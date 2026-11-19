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
    /// whichever partial is asked to shed.
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
    /// It existed at the end but the filter went out without a Bloom (an
    /// exact set, or a range when the end-of-build reservation failed).
    pub(crate) const NOT_SHIPPED: u64 = 32;
    /// The plan gave the filter one (set by the build when it reports).
    pub(crate) const PLANNED: u64 = 64;
    /// Its memory was reserved before any partial needed it, once the
    /// build's keys were big enough.
    pub(crate) const RESERVED: u64 = 128;

    /// The names of the bits set in `bits`, in order.
    pub(crate) fn names(bits: u64) -> String {
        [
            (PLANNED, "planned"),
            (RESERVED, "reserved"),
            (ACTIVE, "active"),
            (REFUSED, "refused"),
            (FULL, "full"),
            (LOST, "lost"),
            (SHIPPED, "bloom"),
            (NOT_SHIPPED, "no-bloom"),
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
    /// Its memory, reserved while the pool had room ([`SpilledBloom::reserve`]);
    /// the filter is made on it when a partial first drains.
    Reserved(Reservation),
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
    /// The hashes are in the filter; the bytes taken from the pool for it.
    Into(usize),
    /// No filter (refused, closed, published): nothing changed.
    Not,
    /// The filter passed its capacity and was freed: the bytes this drain
    /// took from the pool, and the filter's bytes returned to it.
    Full { taken: usize, freed: usize },
}

/// What [`Partials::settle`] moved between the partials, their fixed
/// filters and the pool (review F2 of f574c368f: a filter made and filled
/// by one drain returns more than it took).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct Moved {
    /// Bytes the pool gave a new filter.
    pub(super) taken: usize,
    /// Bytes of a filled filter the pool got back.
    pub(super) freed: usize,
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
    /// Memory reserved but never drained into is freed here, before the
    /// publish reserves the builder's own filter.
    pub(super) fn take(&self) -> Option<(RuntimeBloom, Reservation)> {
        let mut st = lock(&self.state);
        match std::mem::replace(&mut *st, Spilled::Taken) {
            Spilled::Active { bloom, memory, .. } => Some((bloom, memory)),
            Spilled::Reserved(_) => None,
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
        if matches!(
            *st,
            Spilled::Unused | Spilled::Reserved(_) | Spilled::Active { .. }
        ) {
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

    /// Reserves the filter's memory (`bytes`, from `memory`'s pool) once the
    /// build's keys are big enough that a spill request would make it
    /// ([`Self::big`]): early in the build the pool has room, while by the
    /// time a partial passes its cap or is asked to spill it is full (TPC-H
    /// q09 SF100 at 2g, peak 2.097 of 2.147 GB, was `planned; refused`), and
    /// one partial's hashes cover only part of the filter (review F1 of
    /// f574c368f). A refusal is tried again on a later batch.
    pub(super) fn reserve(&self, bytes: usize, memory: &mut Reservation) {
        if !self.big(bytes) {
            return;
        }
        let mut st = lock(&self.state);
        if !matches!(*st, Spilled::Unused) {
            return;
        }
        let mut r = memory.split(0);
        if r.try_grow(rounded(bytes)).is_ok() {
            self.note(events::RESERVED);
            *st = Spilled::Reserved(r);
        }
    }

    /// Inserts `b`'s hashes (taking them from it), first creating the
    /// filter of `bytes`: on the memory [`Self::reserve`] holds, or else on
    /// a new reservation of `memory`'s consumer, up to `fund` bytes of it
    /// from `memory` itself (the reserved bytes of the hashes being drained,
    /// which the filter replaces) and only the rest from the pool.
    fn drain(
        &self,
        b: &mut KeyFilterBuilder,
        bytes: usize,
        fund: usize,
        memory: &mut Reservation,
    ) -> Drained {
        let mut st = lock(&self.state);
        let mut taken = 0;
        if matches!(*st, Spilled::Reserved(_))
            && let Spilled::Reserved(r) = std::mem::take(&mut *st)
        {
            *st = self.activate(r, b);
        }
        if matches!(*st, Spilled::Unused) {
            let bytes = rounded(bytes);
            let fund = fund.min(bytes).min(memory.size());
            // Grow by what `fund` does not cover, then move the whole filter
            // to a reservation of its own: a refusal changes nothing.
            *st = if memory.try_grow(bytes - fund).is_ok() {
                taken = bytes - fund;
                self.activate(memory.split(bytes), b)
            } else {
                self.note(events::REFUSED);
                Spilled::Closed
            };
        }
        let Spilled::Active {
            bloom,
            memory: own,
            inserted,
            capacity,
        } = &mut *st
        else {
            return Drained::Not;
        };
        let hashes = b.drain_hashes().unwrap_or_default();
        *inserted += hashes.len() as u64;
        if *inserted > *capacity {
            // More keys than it holds within the false-positive limit: it
            // would publish as a range anyway, so stop paying for it now.
            let freed = own.size();
            self.note(events::FULL);
            *st = Spilled::Closed;
            return Drained::Full { taken, freed };
        }
        for h in hashes {
            bloom.insert_hash(h);
        }
        Drained::Into(taken)
    }

    /// The filter on reservation `memory`, which it fills.
    fn activate(&self, memory: Reservation, b: &KeyFilterBuilder) -> Spilled {
        self.note(events::ACTIVE);
        let bloom = RuntimeBloom::with_bytes(memory.size());
        Spilled::Active {
            capacity: capacity(&bloom, b.options().bloom_max_fpp),
            bloom,
            memory,
            inserted: 0,
        }
    }
}

/// A filter's size in whole Bloom blocks.
fn rounded(bytes: usize) -> usize {
    bytes.div_ceil(BLOCK_BYTES).max(1) * BLOCK_BYTES
}

impl Partials {
    /// Drains filter `i`'s hashes into its fixed Bloom ([`Self::spilled`])
    /// when it has one (`bloom_spill_bytes`) and either the partial passed
    /// `bloom_max_items`, another partial drains already, or it is `force`d
    /// (memory refused, a spill request) while the whole build's keys reach a
    /// sixteenth of the filter as hashes. Past the cap (or forced) without
    /// the filter, or once it is full, the hashes are dropped and the Bloom
    /// lost for the build ([`Self::mark_lost`]). Returns `None` when no
    /// hashes left the partial, else what moved to and from the pool
    /// ([`Moved`]). A filter not reserved earlier ([`Self::reserve_fixed`])
    /// is paid first from the partials' reservations, up to what they hold
    /// beyond everything but these hashes, so the reservations may shrink:
    /// callers re-read them.
    pub(super) fn settle(&mut self, i: usize, force: bool) -> Option<Moved> {
        let (Some(Some(b)), Some(spilled)) = (self.builders.get(i), self.spilled.get(i)) else {
            return None;
        };
        let bytes = b.options().bloom_spill_bytes? as usize;
        let must = (force && spilled.big(bytes)) || b.is_drained() || b.past_bloom_cap();
        if !b.has_hashes() || !(must || spilled.active()) {
            return None;
        }
        // The reserved bytes beyond what the partials hold besides these
        // hashes: the filter replaces them, so they fund it first.
        let held: usize = self.memory.iter().map(Reservation::size).sum();
        let fund = held.saturating_sub(self.heap_bytes().saturating_sub(b.hash_bytes()));
        let (Some(Some(b)), Some(spilled)) = (self.builders.get_mut(i), self.spilled.get(i)) else {
            return None;
        };
        let drained = match self.memory.first_mut() {
            Some(m) => spilled.drain(b, bytes, fund, m),
            None => Drained::Not,
        };
        let moved = match drained {
            Drained::Into(taken) => return Some(Moved { taken, freed: 0 }),
            Drained::Not if !must => return None,
            Drained::Not => Moved::default(),
            Drained::Full { taken, freed } => Moved { taken, freed },
        };
        b.drop_hashes();
        self.mark_lost(i);
        Some(moved)
    }

    /// Reserves each filter's fixed Bloom once the build is big enough
    /// ([`SpilledBloom::reserve`]), while its partial still collects hashes.
    pub(super) fn reserve_fixed(&mut self) {
        for (b, s) in self.builders.iter().zip(self.spilled.iter()) {
            let Some(b) = b.as_ref().filter(|b| b.has_hashes()) else {
                continue;
            };
            if let (Some(bytes), Some(m)) = (b.options().bloom_spill_bytes, self.memory.first_mut())
            {
                s.reserve(bytes as usize, m);
            }
        }
    }

    /// [`Self::settle`] for every filter: whether hashes left, and what
    /// moved to and from the pool in all.
    pub(super) fn settle_all(&mut self, force: bool) -> (bool, Moved) {
        let (mut left, mut all) = (false, Moved::default());
        for i in 0..self.builders.len() {
            if let Some(m) = self.settle(i, force) {
                left = true;
                all.taken += m.taken;
                all.freed += m.freed;
            }
        }
        (left, all)
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
