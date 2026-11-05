// SPDX-License-Identifier: Apache-2.0
//! [`MergeIndex`]: the key index of a large final-phase merge table, a salted
//! directory like the large [`crate::rows::JoinIndex`]. Its slots hold a
//! 16-bit salt and `row reference + 1`, probed linearly at load at most 1/2.
//!
//! A merge is mostly inserts into a table too large for the caches, so each
//! lookup misses. `hashbrown` has no way to prefetch; this directory's slots
//! are prefetched for a chunk of rows at a time, so the misses overlap. TPC-H
//! SF10 q18 (15M groups, 16 tasks of about 1M): 1.162 to 1.034 s, q13 1.184 to
//! 1.164 s, q03 unchanged (gap ledger, "Note 2026-11-01", T159).

use krishiv_kernels::prefetch_read;

use crate::error::{Result, internal};

/// Salt bits: the top 16 bits of `g`, kept in the slot.
const SALT_MASK: u64 = 0xFFFF << 48;
/// The `row reference + 1` part of a slot (0 = empty).
const ID_MASK: u64 = !SALT_MASK;
/// Fewest slots.
const MIN_SLOTS: usize = 1 << 10;

/// Slots for `keys` keys at load at most 1/2.
fn slots_for(keys: usize) -> usize {
    keys.saturating_mul(2)
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX)
        .max(MIN_SLOTS)
}

#[inline]
fn low(g: u64, mask: usize) -> usize {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "only the low bits below the mask are used"
    )]
    let low = g as usize;
    low & mask
}

/// What a lookup found in one slot.
pub(super) enum Slot {
    /// Empty: the key is not indexed; insert it at this position.
    Empty(usize),
    /// The salt matches: compare the row's key. On a different key, go on
    /// from the next position.
    Candidate(usize, u32),
}

/// A salted directory of row references: see the module docs.
#[derive(Debug)]
pub(super) struct MergeIndex {
    slots: Vec<u64>,
    keys: usize,
}

impl MergeIndex {
    /// Bytes of an index sized for `keys` keys.
    pub(super) fn bytes_for(keys: usize) -> usize {
        slots_for(keys).saturating_mul(size_of::<u64>())
    }

    /// An empty index sized for `keys` keys.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if it cannot be allocated.
    pub(super) fn new(keys: usize) -> Result<Self> {
        Ok(Self {
            slots: zeroed(slots_for(keys))?,
            keys: 0,
        })
    }

    /// Slots (the index's capacity, for the load factor).
    pub(super) fn slots(&self) -> usize {
        self.slots.len()
    }

    /// Bytes held.
    pub(super) fn bytes(&self) -> usize {
        self.slots.capacity() * size_of::<u64>()
    }

    /// Whether `new` more keys need a larger directory.
    fn must_grow(&self, new: usize) -> bool {
        self.keys.saturating_add(new).saturating_mul(2) > self.slots.len()
    }

    /// Bytes at the peak of [`reserve`](Self::reserve) for `new` more keys:
    /// while it grows it holds both directories.
    pub(super) fn projected(&self, new: usize) -> usize {
        if self.must_grow(new) {
            self.bytes()
                .saturating_add(Self::bytes_for(self.keys.saturating_add(new)))
        } else {
            self.bytes()
        }
    }

    /// Makes room for `new` more keys, re-inserting from the hashes `g` of
    /// the indexed rows (no key reads); returns whether it grew.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if the larger directory cannot be allocated.
    pub(super) fn reserve(&mut self, new: usize, g_of: impl Fn(u32) -> u64) -> Result<bool> {
        if !self.must_grow(new) {
            return Ok(false);
        }
        let mut slots = zeroed(slots_for(self.keys.saturating_add(new)))?;
        let mask = slots.len() - 1;
        for &s in self.slots.iter().filter(|&&s| s != 0) {
            let mut i = low(g_of(row_ref(s)), mask);
            while slots[i] != 0 {
                i = (i + 1) & mask;
            }
            slots[i] = s;
        }
        self.slots = slots;
        Ok(true)
    }

    /// Asks the CPU to load the slot of `g`.
    #[inline]
    pub(super) fn prefetch_slot(&self, g: u64) {
        prefetch_read(&self.slots, low(g, self.slots.len() - 1));
    }

    /// The first position of a lookup of `g`.
    #[inline]
    pub(super) fn start(&self, g: u64) -> usize {
        low(g, self.slots.len() - 1)
    }

    /// The next position after `at`.
    #[inline]
    pub(super) fn after(&self, at: usize) -> usize {
        (at + 1) & (self.slots.len() - 1)
    }

    /// The first empty or salt-matching slot of `g` from position `at`.
    /// Load at most 1/2 means an empty slot always ends the walk.
    #[inline]
    pub(super) fn walk(&self, g: u64, mut at: usize) -> Slot {
        loop {
            let s = self.slots[at];
            if s == 0 {
                return Slot::Empty(at);
            }
            if (s ^ g) & SALT_MASK == 0 {
                return Slot::Candidate(at, row_ref(s));
            }
            at = self.after(at);
        }
    }

    /// Indexes row `r` with hash `g` at the empty position `at`.
    #[inline]
    pub(super) fn insert(&mut self, at: usize, g: u64, r: u32) {
        self.slots[at] = (g & SALT_MASK) | (u64::from(r) + 1);
        self.keys += 1;
    }
}

/// The row reference of an occupied slot.
#[inline]
fn row_ref(slot: u64) -> u32 {
    u32::try_from((slot & ID_MASK).saturating_sub(1)).unwrap_or(u32::MAX)
}

fn zeroed(len: usize) -> Result<Vec<u64>> {
    let mut slots: Vec<u64> = Vec::new();
    slots
        .try_reserve_exact(len)
        .map_err(|e| internal(e.to_string()))?;
    slots.resize(len, 0);
    Ok(slots)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_walk_passes_other_salts_and_ends_at_an_empty_slot() {
        let mut index = MergeIndex::new(1).unwrap();
        let (a, b) = (0x0001_0000_0000_0005_u64, 0x0002_0000_0000_0005_u64);
        let at = index.start(a);
        assert!(matches!(index.walk(a, at), Slot::Empty(x) if x == at));
        index.insert(at, a, 7);
        // Another salt on the same slot: passed over.
        assert!(matches!(index.walk(b, at), Slot::Empty(x) if x == index.after(at)));
        // The same salt: a candidate, and the next position is empty.
        assert!(matches!(index.walk(a, at), Slot::Candidate(x, 7) if x == at));
        assert!(matches!(index.walk(a, index.after(at)), Slot::Empty(_)));
    }

    #[test]
    fn growth_keeps_every_key_and_is_projected() {
        let g = |r: u32| u64::from(r) << 3;
        let mut index = MergeIndex::new(10).unwrap();
        for r in 0..10 {
            let Slot::Empty(at) = index.walk(g(r), index.start(g(r))) else {
                panic!("key {r} seen twice");
            };
            index.insert(at, g(r), r);
        }
        assert!(!index.reserve(100, g).unwrap());
        assert_eq!(index.projected(100), index.bytes());
        let peak = index.projected(10_000);
        assert_eq!(peak, index.bytes() + MergeIndex::bytes_for(10_010));
        assert!(index.reserve(10_000, g).unwrap());
        assert_eq!(index.bytes(), MergeIndex::bytes_for(10_010));
        for r in 0..10 {
            assert!(matches!(
                index.walk(g(r), index.start(g(r))),
                Slot::Candidate(_, x) if x == r
            ));
        }
        assert_eq!(MergeIndex::bytes_for(usize::MAX), usize::MAX);
    }
}
