// SPDX-License-Identifier: Apache-2.0
//! [`MergeIndex`]: the key index of a large final-phase merge table, a salted
//! directory like the large [`crate::rows::JoinIndex`]. Its slots hold a
//! 16-bit salt and `row reference + 1`, probed linearly. It is sized for
//! load 1/2 at the estimate, exactly rather than to a power of two (a key's
//! first slot is a multiply-shift of its hash, not a mask), and grows only past
//! load 5/8, so an estimate a little low does not double it. A power of two
//! made it up to twice that: ClickBench q33's merge held 268M slots for 100M
//! groups (`design/P3-A-wide-string-groups.md`).
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
/// Bits of `g` below the salt that pick a key's first slot.
const POS_BITS: u32 = 48;

/// Slots for `keys` keys at load 1/2 (a whole number of cache lines).
fn slots_for(keys: usize) -> usize {
    keys.saturating_mul(2)
        .checked_next_multiple_of(8)
        .unwrap_or(usize::MAX)
        .max(MIN_SLOTS)
}

/// The first slot of `g` among `len`: the low [`POS_BITS`] bits of `g`
/// scaled to `len` (multiply-shift; the salt bits are not used). `g` must be
/// a mixed hash (the table's `fin`): one with entropy only in its low bits
/// would put every key in the first slots.
#[inline]
fn first(g: u64, len: usize) -> usize {
    let x = u128::from(g & ID_MASK);
    // Below `len` (a 48-bit fraction of it), so the cast keeps every bit.
    ((x * len as u128) >> POS_BITS) as usize
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

    /// Whether `new` more keys need a larger directory: past load 5/8. A
    /// miss there walks about 4.1 slots (2.5 at load 1/2, where the
    /// directory is sized), still mostly inside one cache line.
    fn must_grow(&self, new: usize) -> bool {
        self.keys.saturating_add(new).saturating_mul(8) > self.slots.len().saturating_mul(5)
    }

    /// The slots a growth for `new` more keys allocates: load 1/2, and at
    /// least twice the old directory (so repeated growth stays linear).
    fn grown_slots(&self, new: usize) -> usize {
        slots_for(self.keys.saturating_add(new)).max(self.slots.len().saturating_mul(2))
    }

    /// Bytes at the peak of [`reserve`](Self::reserve) for `new` more keys:
    /// while it grows it holds both directories.
    pub(super) fn projected(&self, new: usize) -> usize {
        if self.must_grow(new) {
            self.bytes()
                .saturating_add(self.grown_slots(new).saturating_mul(size_of::<u64>()))
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
        let mut slots = zeroed(self.grown_slots(new))?;
        let len = slots.len();
        for &s in self.slots.iter().filter(|&&s| s != 0) {
            let mut i = first(g_of(row_ref(s)), len);
            while slots[i] != 0 {
                i = next(i, len);
            }
            slots[i] = s;
        }
        self.slots = slots;
        Ok(true)
    }

    /// Asks the CPU to load the slot of `g`.
    #[inline]
    pub(super) fn prefetch_slot(&self, g: u64) {
        prefetch_read(&self.slots, first(g, self.slots.len()));
    }

    /// The first position of a lookup of `g`.
    #[inline]
    pub(super) fn start(&self, g: u64) -> usize {
        first(g, self.slots.len())
    }

    /// The next position after `at`.
    #[inline]
    pub(super) fn after(&self, at: usize) -> usize {
        next(at, self.slots.len())
    }

    /// The first empty or salt-matching slot of `g` from position `at`.
    /// Load at most 5/8 means an empty slot always ends the walk.
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

/// The position after `at` among `len`, wrapping at the end.
#[inline]
fn next(at: usize, len: usize) -> usize {
    if at + 1 == len { 0 } else { at + 1 }
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
        let g = |r: u32| u64::from(r).wrapping_mul(0x9E37_79B9_7F4A_7C15);
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

    /// Inserts `g(r)` for each `r` of `rows` (all new keys).
    fn fill(index: &mut MergeIndex, rows: std::ops::Range<u32>, g: impl Fn(u32) -> u64) {
        for r in rows {
            // A same-salt slot holds another key (keys are distinct here).
            let mut at = index.start(g(r));
            let free = loop {
                match index.walk(g(r), at) {
                    Slot::Empty(free) => break free,
                    Slot::Candidate(x, _) => at = index.after(x),
                }
            };
            index.insert(free, g(r), r);
        }
    }

    #[test]
    fn the_directory_is_sized_to_the_estimate_not_a_power_of_two() {
        // q33's merge: 100M groups took 2^28 slots; now 2 per key.
        assert_eq!(MergeIndex::bytes_for(100_000_000), 200_000_000 * 8);
        assert_eq!(MergeIndex::new(3_000).unwrap().slots(), 6_000);
        assert_eq!(MergeIndex::new(3_001).unwrap().slots(), 6_008);
        assert_eq!(MergeIndex::new(1).unwrap().slots(), MIN_SLOTS);
    }

    #[test]
    fn keys_past_the_estimate_fit_up_to_load_five_eighths() {
        // A 1,000-key estimate (2,000 slots) takes 1,250 keys without growing,
        // so an estimate a little low or a last merge chunk does not double it.
        let g = |r: u32| u64::from(r).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut index = MergeIndex::new(1_000).unwrap();
        assert!(!index.reserve(1_250, g).unwrap());
        fill(&mut index, 0..1_250, g);
        assert_eq!(index.slots(), 2_000);
        assert!(index.reserve(1, g).unwrap());
        // A growth at least doubles.
        assert_eq!(index.slots(), 4_000);
        fill(&mut index, 1_250..2_000, g);
        for r in 0..2_000 {
            assert!(matches!(
                index.walk(g(r), index.start(g(r))),
                Slot::Candidate(_, x) if x == r
            ));
        }
    }

    proptest::proptest! {
        #[test]
        fn every_key_survives_any_growth_sequence(
            estimate in 1_usize..5_000,
            steps in proptest::collection::vec(1_u32..700, 1..8),
            len in 1_usize..100_000_000,
            g in proptest::num::u64::ANY,
        ) {
            proptest::prop_assert!(first(g, len) < len);
            let mix = |r: u32| u64::from(r).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (u64::from(r) << 40);
            let mut index = MergeIndex::new(estimate).unwrap();
            let mut next = 0_u32;
            for n in steps {
                let before = index.bytes();
                let peak = index.projected(n as usize);
                let grew = index.reserve(n as usize, mix).unwrap();
                proptest::prop_assert_eq!(peak, if grew { before + index.bytes() } else { before });
                fill(&mut index, next..next + n, mix);
                next += n;
                proptest::prop_assert!(index.keys * 8 <= index.slots() * 5);
            }
            for r in 0..next {
                let mut at = index.start(mix(r));
                let found = loop {
                    match index.walk(mix(r), at) {
                        Slot::Empty(_) => break false,
                        Slot::Candidate(_, x) if x == r => break true,
                        Slot::Candidate(x, _) => at = index.after(x),
                    }
                };
                proptest::prop_assert!(found, "key {} lost", r);
            }
        }
    }

    #[test]
    fn a_walk_from_the_last_slot_wraps_to_the_first() {
        let mut index = MergeIndex::new(1).unwrap();
        let last = MIN_SLOTS - 1;
        // The highest 48-bit position maps to the last slot.
        let (a, b) = (0x0001_FFFF_FFFF_FFFF_u64, 0x0002_FFFF_FFFF_FFFF_u64);
        assert_eq!(index.start(a), last);
        index.insert(last, a, 3);
        assert!(matches!(index.walk(b, index.start(b)), Slot::Empty(0)));
        index.insert(0, b, 4);
        assert!(matches!(index.walk(b, last), Slot::Candidate(0, 4)));
    }
}
