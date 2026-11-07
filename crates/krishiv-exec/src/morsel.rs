// SPDX-License-Identifier: Apache-2.0
//! Morsel dispatch (HyPer-style), the finalize countdown, the flush
//! barrier and the deterministic fan-in's baton.

use std::sync::Arc;

use crate::sync::{AtomicUsize, Event, Mutex, Ordering, lock};

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
        self.claim_ordered().map(|(_, m)| m)
    }

    /// As [`claim`](Self::claim), with the claim's ordinal: claims are
    /// numbered `0..len` in the order they happen, so a later claim always
    /// has a higher ordinal. Unseeded, the ordinal is the morsel index.
    pub fn claim_ordered(&self) -> Option<(u64, usize)> {
        // Relaxed: the index only selects work; the morsel's data is shared
        // immutable state published before the drivers started.
        let i = self.next.fetch_add(1, Ordering::Relaxed);
        if i >= self.len {
            return None;
        }
        Some((i as u64, self.order.as_ref().map_or(i, |o| o[i])))
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
            if self.is_open() {
                return;
            }
            listener.await;
        }
    }

    /// Whether every driver has arrived.
    pub(crate) fn is_open(&self) -> bool {
        self.left.load(Ordering::Acquire) == 0
    }
}

/// The turn of the members of a fan-in whose shared operators need a flush,
/// under `krishiv.deterministicOrder` (one driver per member): only the
/// member holding the baton runs, so the rows reach the shared operators in
/// one order on every run. Member 0 starts with it. A member keeps it until
/// it waits at a flush barrier that is still closed ([`pass`](Self::pass))
/// or ends ([`leave`](Self::leave)); the baton then goes to the next member
/// after it, in member order, that has not ended and is not waiting at a
/// closed barrier. Every arrival at a barrier happens under the baton, so
/// which barriers are open when it moves is the same on every run, and so
/// is the whole schedule.
///
/// A member gives the baton up only between operators, never inside an
/// operator's flush, so no member waits for work another member holds (a
/// grace join's claimed partition) while the holder waits for it.
#[derive(Debug)]
pub(crate) struct Baton {
    state: Mutex<BatonState>,
    event: Event,
}

#[derive(Debug)]
struct BatonState {
    holder: Option<usize>,
    ended: Vec<bool>,
    /// Per member: the closed barrier it waits at, if any.
    blocked: Vec<Option<Arc<Barrier>>>,
    /// Every member left waits at a closed barrier: nobody can run.
    stuck: bool,
}

impl Baton {
    /// A baton for `members` members, held by member 0.
    pub(crate) fn new(members: usize) -> Self {
        Self {
            state: Mutex::new(BatonState {
                holder: (members > 0).then_some(0),
                ended: vec![false; members],
                blocked: vec![None; members],
                stuck: false,
            }),
            event: Event::default(),
        }
    }

    /// Completes once `member` holds the baton: `true`, or `false` when no
    /// member can ever run again (every member left waits at a closed
    /// barrier).
    pub(crate) async fn acquire(&self, member: usize) -> bool {
        loop {
            let listener = self.event.listen();
            {
                let s = lock(&self.state);
                if s.holder == Some(member) {
                    return true;
                }
                if s.stuck {
                    return false;
                }
            }
            listener.await;
        }
    }

    /// `member`, holding the baton, waits at `barrier`, which is closed:
    /// the baton moves on, and comes back once `barrier` has opened.
    pub(crate) fn pass(&self, member: usize, barrier: Arc<Barrier>) {
        {
            let mut s = lock(&self.state);
            if s.holder != Some(member) {
                return;
            }
            if let Some(b) = s.blocked.get_mut(member) {
                *b = Some(barrier);
            }
            s.hand_on(member);
        }
        self.event.notify();
    }

    /// `member` ended (or failed, or was aborted): it never runs again, and
    /// if it holds the baton, the baton moves on.
    pub(crate) fn leave(&self, member: usize) {
        {
            let mut s = lock(&self.state);
            if let Some(e) = s.ended.get_mut(member) {
                *e = true;
            }
            if s.holder != Some(member) {
                return;
            }
            s.hand_on(member);
        }
        self.event.notify();
    }
}

impl BatonState {
    /// Gives the baton to the first member after `from`, in member order,
    /// that can run.
    fn hand_on(&mut self, from: usize) {
        let n = self.ended.len();
        self.holder = None;
        let mut waiting = false;
        for d in 1..=n {
            let j = (from + d) % n;
            if self.ended[j] {
                continue;
            }
            if self.blocked[j].as_ref().is_some_and(|b| !b.is_open()) {
                waiting = true;
                continue;
            }
            self.blocked[j] = None;
            self.holder = Some(j);
            return;
        }
        self.stuck = waiting;
    }
}
