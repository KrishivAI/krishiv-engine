// SPDX-License-Identifier: Apache-2.0
//! Morsel dispatch (HyPer-style), the finalize countdown and the flush
//! barrier.

use crate::sync::{AtomicUsize, Event, Ordering};

/// Hands out morsel indices `0..len` to drivers, each exactly once: one
/// atomic increment per claim, so drivers balance load by pulling. Seeded,
/// the order is a fixed permutation (deterministic test mode).
#[derive(Debug)]
pub struct MorselCursor {
    next: AtomicUsize,
    len: usize,
    order: Option<Vec<usize>>,
}

impl MorselCursor {
    /// `len` morsels, claimed in order, or in a permutation fixed by `seed`.
    pub fn new(len: usize, seed: Option<u64>) -> Self {
        Self {
            next: AtomicUsize::new(0),
            len,
            order: seed.map(|s| permutation(len, s)),
        }
    }

    /// Number of morsels.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no morsels.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The next unclaimed morsel, or `None` once all are claimed.
    pub fn claim(&self) -> Option<usize> {
        // Relaxed: the index only selects work; the morsel's data is shared
        // immutable state published before the drivers started.
        let i = self.next.fetch_add(1, Ordering::Relaxed);
        if i >= self.len {
            return None;
        }
        Some(self.order.as_ref().map_or(i, |o| o[i]))
    }
}

/// A Fisher-Yates shuffle of `0..len` driven by SplitMix64.
fn permutation(len: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut v: Vec<usize> = (0..len).collect();
    for i in (1..len).rev() {
        let j = usize::try_from(next() % (i as u64 + 1)).unwrap_or(i);
        v.swap(i, j);
    }
    v
}

/// Counts drivers down to the one that runs `finalize`: every driver calls
/// [`arrive`](Self::arrive) after its `combine`; exactly one sees `true`,
/// and that one observes every other driver's `combine` (release/acquire).
#[derive(Debug)]
pub(crate) struct Countdown(AtomicUsize);

impl Countdown {
    pub(crate) fn new(drivers: usize) -> Self {
        Self(AtomicUsize::new(drivers))
    }

    /// `true` for the last driver.
    pub(crate) fn arrive(&self) -> bool {
        self.0.fetch_sub(1, Ordering::AcqRel) == 1
    }
}

/// The flush barrier of one operator (see
/// [`Operator::needs_flush`](crate::Operator::needs_flush)): every driver
/// [`arrive`](Self::arrive)s once, and [`wait`](Self::wait) completes once
/// all have, observing everything each did before arriving.
#[derive(Debug)]
pub(crate) struct Barrier {
    left: AtomicUsize,
    event: Event,
}

impl Barrier {
    pub(crate) fn new(drivers: usize) -> Self {
        Self {
            left: AtomicUsize::new(drivers),
            event: Event::default(),
        }
    }

    pub(crate) fn arrive(&self) {
        if self.left.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.event.notify();
        }
    }

    /// Completes once every driver has arrived.
    pub(crate) async fn wait(&self) {
        loop {
            let listener = self.event.listen();
            if self.left.load(Ordering::Acquire) == 0 {
                return;
            }
            listener.await;
        }
    }
}
