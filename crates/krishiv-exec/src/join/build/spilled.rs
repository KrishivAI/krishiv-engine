// SPDX-License-Identifier: Apache-2.0
//! The fixed Bloom a guessed build keeps past its key cap or through a spill
//! (`KeyFilterOptions::bloom_spill_bytes`, TPC-H q09 at SF1000): one per
//! runtime filter of a task's build, shared by its partials, which drain
//! their hashes into it ([`Partials::settle`]).

use std::sync::atomic::{AtomicU64, Ordering};

use krishiv_io::KeyFilterBuilder;
use krishiv_io::runtime_filter::{BLOCK_BYTES, RuntimeBloom};
use krishiv_memory::Reservation;

use super::Partials;
use crate::sync::{Mutex, lock};

/// A spill request creates the fixed Bloom only once the build's keys would
/// take at least this share of its size as hashes (8 bytes a key): a small
/// build's hashes are shed as before, rather than reserve a filter many
/// times their size.
const FORCED_DRAIN_SHARE: u64 = 16;

/// A filter's fixed Bloom ([`Partials::spilled`]).
#[derive(Debug, Default)]
pub(super) struct SpilledBloom {
    state: Mutex<Spilled>,
    /// Non-null keys every partial of the build added: the build's size,
    /// whichever partial is asked to shed (review finding 4).
    seen: AtomicU64,
    /// What happened to it, as [`events`] bits (`EXPLAIN ANALYZE`'s
    /// `fixed_bloom`).
    events: AtomicU64,
}

/// [`SpilledBloom`]'s outcome bits, reported by the build sink as
/// `fixed_bloom` ([`events::names`]).
pub(crate) mod events {
    /// A partial drained into it: the filter was created.
    pub(crate) const ACTIVE: u64 = 1;
    /// Its memory was refused when a partial first needed it.
    pub(crate) const REFUSED: u64 = 2;
    /// It passed the keys it holds within `bloom_max_fpp`.
    pub(crate) const FULL: u64 = 4;
    /// The build lost the Bloom while the filter was held.
    pub(crate) const LOST: u64 = 8;
    /// It went out as the filter's Bloom.
    pub(crate) const SHIPPED: u64 = 16;
    /// It existed at the end but the filter went out without a Bloom.
    pub(crate) const DROPPED: u64 = 32;
    /// The plan gave the filter one (set by the build when it reports).
    pub(crate) const PLANNED: u64 = 64;

    /// The names of the bits set in `bits`, in order.
    pub(crate) fn names(bits: u64) -> String {
        [
            (PLANNED, "planned"),
            (ACTIVE, "active"),
            (REFUSED, "refused"),
            (FULL, "full"),
            (LOST, "lost"),
            (SHIPPED, "bloom"),
            (DROPPED, "range"),
        ]
        .iter()
        .filter(|(b, _)| bits & b != 0)
        .map(|(_, n)| *n)
        .collect::<Vec<_>>()
        .join("; ")
    }
}

#[derive(Debug, Default)]
enum Spilled {
    /// No partial needed it yet.
    #[default]
    Unused,
    /// The filter, the reservation holding it, the keys inserted and the
    /// most it holds within `bloom_max_fpp`.
    Active {
        bloom: RuntimeBloom,
        memory: Reservation,
        inserted: u64,
        capacity: u64,
    },
    /// Its memory was refused, it filled, or the build lost the Bloom:
    /// the partials drop their hashes as before.
    Closed,
    /// Taken by the publish.
    Taken,
}

/// What a drain did.
enum Drained {
    /// The hashes are in the filter; the bytes newly reserved for it.
    Into(usize),
    /// No filter (refused, closed, published): nothing changed.
    Not,
    /// The filter passed its capacity and was freed.
    Full,
}

/// The most keys a filter of `bytes` holds within `max_fpp`.
fn capacity(bloom: &RuntimeBloom, max_fpp: f64) -> u64 {
    let (mut lo, mut hi) = (0u64, (bloom.num_bytes() as u64).saturating_mul(8));
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if bloom.expected_fpp(mid) <= max_fpp {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

impl SpilledBloom {
    /// The filter and its reservation, once: `None` when no partial drained
    /// into it, or it was closed.
    pub(super) fn take(&self) -> Option<(RuntimeBloom, Reservation)> {
        let mut st = lock(&self.state);
        match std::mem::replace(&mut *st, Spilled::Taken) {
            Spilled::Active { bloom, memory, .. } => Some((bloom, memory)),
            other => {
                *st = other;
                None
            }
        }
    }

    /// Frees the filter and its memory for good: the build lost the Bloom.
    pub(super) fn close(&self) {
        let mut st = lock(&self.state);
        if matches!(*st, Spilled::Active { .. }) {
            self.note(events::LOST);
        }
        if matches!(*st, Spilled::Unused | Spilled::Active { .. }) {
            *st = Spilled::Closed;
        }
    }

    /// Records [`events`] bits.
    pub(super) fn note(&self, bits: u64) {
        self.events.fetch_or(bits, Ordering::Relaxed);
    }

    /// The [`events`] bits so far and the build's keys.
    pub(super) fn outcome(&self) -> (u64, u64) {
        (
            self.events.load(Ordering::Relaxed),
            self.seen.load(Ordering::Relaxed),
        )
    }

    /// Whether a partial already drains into it.
    pub(super) fn active(&self) -> bool {
        matches!(*lock(&self.state), Spilled::Active { .. })
    }

    /// Counts `rows` more keys added by a partial of the build.
    pub(super) fn saw(&self, rows: u64) {
        self.seen.fetch_add(rows, Ordering::Relaxed);
    }

    /// Whether the build's keys, as hashes, reach a sixteenth of `bytes`.
    fn big(&self, bytes: usize) -> bool {
        self.seen
            .load(Ordering::Relaxed)
            .saturating_mul(8)
            .saturating_mul(FORCED_DRAIN_SHARE)
            >= bytes as u64
    }

    /// Inserts `b`'s hashes (taking them from it), first creating the
    /// filter of `bytes` on a new reservation of `memory`'s consumer.
    fn drain(&self, b: &mut KeyFilterBuilder, bytes: usize, memory: &mut Reservation) -> Drained {
        let mut st = lock(&self.state);
        let mut reserved = 0;
        if matches!(*st, Spilled::Unused) {
            let bytes = bytes.div_ceil(BLOCK_BYTES).max(1) * BLOCK_BYTES;
            let mut r = memory.split(0);
            *st = if r.try_grow(bytes).is_ok() {
                reserved = bytes;
                self.note(events::ACTIVE);
                let bloom = RuntimeBloom::with_bytes(bytes);
                Spilled::Active {
                    capacity: capacity(&bloom, b.options().bloom_max_fpp),
                    bloom,
                    memory: r,
                    inserted: 0,
                }
            } else {
                self.note(events::REFUSED);
                Spilled::Closed
            };
        }
        let Spilled::Active {
            bloom,
            inserted,
            capacity,
            ..
        } = &mut *st
        else {
            return Drained::Not;
        };
        let hashes = b.drain_hashes().unwrap_or_default();
        *inserted += hashes.len() as u64;
        if *inserted > *capacity {
            // More keys than it holds within the false-positive limit: it
            // would publish as a range anyway, so stop paying for it now
            // (review finding 2).
            self.note(events::FULL);
            *st = Spilled::Closed;
            return Drained::Full;
        }
        for h in hashes {
            bloom.insert_hash(h);
        }
        Drained::Into(reserved)
    }
}

impl Partials {
    /// Drains filter `i`'s hashes into its fixed Bloom ([`Self::spilled`])
    /// when it has one (`bloom_spill_bytes`) and either the partial passed
    /// `bloom_max_items`, another partial drains already, or it is `force`d
    /// (memory refused, a spill request) while the whole build's keys reach a
    /// sixteenth of the filter as hashes. Past the cap (or forced) without
    /// the filter, or once it is full, the hashes are dropped and the Bloom
    /// lost for the build ([`Self::mark_lost`]). Returns `None` when no
    /// hashes left the partial, else the bytes newly reserved for the
    /// filter.
    pub(super) fn settle(&mut self, i: usize, force: bool) -> Option<usize> {
        let (Some(Some(b)), Some(spilled)) = (self.builders.get_mut(i), self.spilled.get(i)) else {
            return None;
        };
        let bytes = b.options().bloom_spill_bytes? as usize;
        let must = (force && spilled.big(bytes)) || b.is_drained() || b.past_bloom_cap();
        if !b.has_hashes() || !(must || spilled.active()) {
            return None;
        }
        let drained = match self.memory.first_mut() {
            Some(m) => spilled.drain(b, bytes, m),
            None => Drained::Not,
        };
        match drained {
            Drained::Into(reserved) => Some(reserved),
            Drained::Not if !must => None,
            Drained::Not | Drained::Full => {
                b.drop_hashes();
                self.mark_lost(i);
                Some(0)
            }
        }
    }

    /// [`Self::settle`] for every filter: whether hashes left, and the bytes
    /// newly reserved for fixed filters.
    pub(super) fn settle_all(&mut self, force: bool) -> (bool, usize) {
        let (mut left, mut reserved) = (false, 0);
        for i in 0..self.builders.len() {
            if let Some(r) = self.settle(i, force) {
                left = true;
                reserved += r;
            }
        }
        (left, reserved)
    }

    /// Filter `i` can no longer be a Bloom: tells the build's other
    /// partials and frees its fixed filter, if any.
    pub(super) fn mark_lost(&self, i: usize) {
        if let Some(lost) = self.bloom_lost.get(i) {
            lost.store(true, Ordering::Relaxed);
        }
        if let Some(s) = self.spilled.get(i) {
            s.close();
        }
    }
}
