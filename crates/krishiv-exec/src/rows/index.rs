// SPDX-License-Identifier: Apache-2.0
//! [`JoinIndex`]: the key index of one hash join partition over rows of a
//! [`RowLayout`] (P2-T158), size-adaptive between two designs:
//!
//! - **flat**: `hashbrown::HashTable<u32>` of row ids (SIMD group probing,
//!   about 5 bytes per bucket at up to 7/8 load) while the index is small
//!   enough to stay in a core's own cache, where its lower instruction count
//!   wins;
//! - **salted**: a power-of-two directory of `u64` slots, each a 16-bit
//!   salt plus `row id + 1`, probed linearly at load ≤ 1/2 (DuckDB's
//!   `ht_entry_t`, with row ids instead of pointers), above that size,
//!   where a probe costs one directory miss and reads a row only when its
//!   salt matches. Its lookups prefetch each chunk's slots, then its first
//!   salt-matching rows, so the misses overlap; `hashbrown` has no way to
//!   prefetch its buckets.
//!
//! The size rule looks at one table, but a probe batch visits every radix
//! partition of a join, so what competes for the cache is all the tables
//! together, and their rows.
//! The kind is chosen once per table from its row count, the upper bound
//! of its distinct keys ([`FLAT_MAX_BYTES`]).
//!
//! Both kinds are driven by the same table-local hash `g = fin(h ^ seed)`
//! of each row's stored hash `h` ([`finalize`]): the salted
//! directory takes its salt from the top 16 bits of `g` and its slot from
//! the low bits; `hashbrown` takes its tag from the top 7 bits and its
//! bucket from the low bits. Neither reads the bits a join's grace
//! sub-partitions use (`g` bits 30..48) or the radix bits of `h`.
//!
//! Each distinct key points to its newest row; older rows with the same
//! key are chained through the caller's `next` array, so the chain order
//! (and hence the join's output order) is the same for both kinds.

use hashbrown::HashTable;
use hashbrown::hash_table::Entry;
use krishiv_kernels::prefetch_read;

use super::layout::{EncodedKeys, RowLayout, finalize};
use crate::error::{Result, internal};
use crate::hash::ProbeCount;

/// "No row" in a lookup result or a chain.
pub const NO_MATCH: u32 = u32::MAX;

/// Rows a salted lookup handles per chunk.
const CHUNK: usize = 256;
/// Fewest slots in a salted directory.
const MIN_SLOTS: usize = 1 << 10;
/// Salt bits: the top 16 bits of `g`, kept in the slot.
const SALT_MASK: u64 = 0xFFFF << 48;
/// The `row id + 1` part of a slot (0 = empty).
const ID_MASK: u64 = u32::MAX as u64;

/// Largest flat index of one table, in bytes, before it takes the salted
/// directory: 512 KiB over the default 8 radix partitions. TPC-H SF10 probe CPU
/// from flat to salted with prefetch: q13 6.23 to 3.58 s, q09 6.47 to 5.43 s,
/// q03 0.72 to 0.53 s. At 256 KiB q09 kept 0.5 s of that loss, its 100,000 rows
/// staying flat (gap ledger, "Note 2026-11-01").
pub const FLAT_MAX_BYTES: usize = 64 << 10;

/// Where an index finds its rows.
pub trait RowSource {
    /// Row `id`'s bytes and the heap its long views point into.
    fn row(&self, id: u32) -> (&[u8], &[u8]);
    /// Row `id`'s stored hash `h`.
    fn hash(&self, id: u32) -> u64;
    /// Whether row `id` goes into the index: a row a join keeps but no key
    /// may find (a NULL `=` key) is stored and never indexed.
    fn indexed(&self, id: u32) -> bool {
        let _ = id;
        true
    }
    /// Asks the CPU to load the start of row `id`: a hint, with no effect on
    /// results.
    fn prefetch(&self, id: u32) {
        let _ = id;
    }
}

/// Which index a table uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexChoice {
    /// By size: flat up to [`FLAT_MAX_BYTES`], salted above.
    Adaptive,
    /// Always flat (tests and measurement).
    Flat,
    /// Always salted (tests and measurement).
    Salted,
}

#[derive(Debug)]
enum Kind {
    Flat(HashTable<u32>),
    Salted(Vec<u64>),
}

/// The key index of one join partition: see the module docs.
#[derive(Debug)]
pub struct JoinIndex {
    seed: u64,
    kind: Kind,
    keys: usize,
}

/// Buckets `hashbrown` allocates for `n` entries.
fn flat_buckets(n: usize) -> usize {
    if n < 4 {
        4
    } else if n < 8 {
        8
    } else {
        (n.saturating_mul(8) / 7)
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX)
    }
}

/// Bytes of a flat index for `n` keys: a `u32` and a control byte per
/// bucket, plus one group of trailing control bytes.
pub fn flat_bytes(n: usize) -> usize {
    flat_buckets(n)
        .saturating_mul(size_of::<u32>() + 1)
        .saturating_add(16)
}

/// Slots of a salted directory for `n` keys at load ≤ 1/2.
fn salted_slots(n: usize) -> usize {
    n.saturating_mul(2)
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX)
        .max(MIN_SLOTS)
}

/// Bytes of a salted directory for `n` keys.
pub fn salted_bytes(n: usize) -> usize {
    salted_slots(n).saturating_mul(size_of::<u64>())
}

impl IndexChoice {
    /// Whether a table of up to `rows` keys takes the flat index.
    fn flat(self, rows: usize) -> bool {
        match self {
            Self::Adaptive => flat_bytes(rows) <= FLAT_MAX_BYTES,
            Self::Flat => true,
            Self::Salted => false,
        }
    }

    /// Bytes the index over `rows` rows takes at most (it never grows).
    pub fn bytes(self, rows: usize) -> usize {
        if self.flat(rows) {
            flat_bytes(rows)
        } else {
            salted_bytes(rows)
        }
    }
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

#[inline]
fn slot_id(slot: u64) -> u32 {
    u32::try_from((slot & ID_MASK).saturating_sub(1)).unwrap_or(NO_MATCH)
}

impl JoinIndex {
    /// Indexes rows `0..rows` of `src`, sized for `rows` keys (no resize).
    /// For each row whose key is already indexed, `next[row]` is set to the
    /// previous newest row with that key and the index points to `row`
    /// (newest first); other entries of `next` are left unchanged.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if `next` is shorter than `rows`, `rows` exceeds the
    /// row id space, or the index cannot be allocated.
    pub fn build(
        layout: &RowLayout,
        src: &impl RowSource,
        rows: usize,
        seed: u64,
        choice: IndexChoice,
        next: &mut [u32],
    ) -> Result<Self> {
        Self::build_expecting(layout, src, rows, rows, seed, choice, next)
    }

    /// [`build`](Self::build), sized for about `keys` distinct keys (a
    /// cardinality estimate, at most `rows`) instead of one per row: the
    /// kind is chosen for `keys`, and the index grows (re-inserting from the
    /// stored hashes, never reading keys) if the estimate was low.
    ///
    /// # Errors
    /// As [`build`](Self::build).
    pub fn build_expecting(
        layout: &RowLayout,
        src: &impl RowSource,
        rows: usize,
        keys: usize,
        seed: u64,
        choice: IndexChoice,
        next: &mut [u32],
    ) -> Result<Self> {
        let expect = keys.clamp(1, rows.max(1));
        let n = u32::try_from(rows)
            .ok()
            .filter(|&n| n < NO_MATCH && rows <= next.len())
            .ok_or_else(|| internal(format!("cannot index {rows} join rows")))?;
        let oom = |e: std::collections::TryReserveError| internal(e.to_string());
        let g_of = |id: u32| finalize(src.hash(id), seed);
        let eq = |a: u32, b: u32| {
            let (ar, ah) = src.row(a);
            let (br, bh) = src.row(b);
            layout.eq(ar, ah, br, bh)
        };
        let mut keys = 0;
        let kind = if choice.flat(expect) {
            let mut t: HashTable<u32> = HashTable::new();
            t.try_reserve(expect, |&id| g_of(id))
                .map_err(|e| internal(format!("{e:?}")))?;
            for id in (0..n).filter(|&id| src.indexed(id)) {
                match t.entry(g_of(id), |&o| eq(id, o), |&o| g_of(o)) {
                    Entry::Occupied(mut o) => {
                        next[id as usize] = *o.get();
                        *o.get_mut() = id;
                    }
                    Entry::Vacant(v) => {
                        v.insert(id);
                        keys += 1;
                    }
                }
            }
            Kind::Flat(t)
        } else {
            let len = salted_slots(expect);
            let mut slots: Vec<u64> = Vec::new();
            slots.try_reserve_exact(len).map_err(oom)?;
            slots.resize(len, 0);
            let mut mask = len - 1;
            for id in (0..n).filter(|&id| src.indexed(id)) {
                if (keys + 1) * 2 > slots.len() {
                    slots = grow_salted(&slots, g_of)?;
                    mask = slots.len() - 1;
                }
                let g = g_of(id);
                let salt = g & SALT_MASK;
                let slot = salt | (u64::from(id) + 1);
                let mut i = low(g, mask);
                loop {
                    let s = slots[i];
                    if s == 0 {
                        slots[i] = slot;
                        keys += 1;
                        break;
                    }
                    if s & SALT_MASK == salt && eq(id, slot_id(s)) {
                        next[id as usize] = slot_id(s);
                        slots[i] = slot;
                        break;
                    }
                    i = (i + 1) & mask;
                }
            }
            Kind::Salted(slots)
        };
        Ok(Self { seed, kind, keys })
    }

    /// Whether this index is the flat one.
    pub fn is_flat(&self) -> bool {
        matches!(self.kind, Kind::Flat(_))
    }

    /// Buckets of the flat index or slots of the salted directory (its
    /// capacity, for the load factor).
    pub fn buckets(&self) -> usize {
        match &self.kind {
            Kind::Flat(t) => t.num_buckets(),
            Kind::Salted(s) => s.len(),
        }
    }

    /// Distinct keys indexed.
    pub fn len(&self) -> usize {
        self.keys
    }

    /// Whether no key is indexed.
    pub fn is_empty(&self) -> bool {
        self.keys == 0
    }

    /// Bytes held.
    pub fn memory_size(&self) -> usize {
        match &self.kind {
            Kind::Flat(t) => flat_bytes(t.capacity()),
            Kind::Salted(s) => s.capacity() * size_of::<u64>(),
        }
    }

    /// For each probe row `which[j]` of `keys` (encoded with the same
    /// layout, with seedless hashes), the newest row of `src` with an equal
    /// key, or [`NO_MATCH`], appended to `out`.
    pub fn find(
        &self,
        layout: &RowLayout,
        keys: &EncodedKeys,
        which: &[u32],
        src: &impl RowSource,
        out: &mut Vec<u32>,
    ) {
        self.find_with(layout, keys, which, src, out, true);
    }

    /// [`find`](Self::find), choosing whether a salted lookup first prefetches
    /// each chunk's slots and first salt-matching rows (P2-T157 §8 enables it
    /// only where an interleaved benchmark shows a win; `join_index_matrix`
    /// re-measures it). The flat index ignores `prefetch`.
    pub fn find_with(
        &self,
        layout: &RowLayout,
        keys: &EncodedKeys,
        which: &[u32],
        src: &impl RowSource,
        out: &mut Vec<u32>,
        prefetch: bool,
    ) {
        self.lookup::<false>(
            layout,
            keys,
            which,
            src,
            out,
            prefetch,
            &mut ProbeCount::default(),
        );
    }

    /// As [`find`](Self::find), counting the keys compared per lookup into
    /// `count` (a detailed profile: `ExecConfig::detailed_metrics`).
    pub fn find_counted(
        &self,
        layout: &RowLayout,
        keys: &EncodedKeys,
        which: &[u32],
        src: &impl RowSource,
        out: &mut Vec<u32>,
        count: &mut ProbeCount,
    ) {
        self.lookup::<true>(layout, keys, which, src, out, true, count);
    }

    /// The lookups of [`find_with`](Self::find_with); `COUNT` adds each
    /// lookup's key comparisons to `count` (compiled out otherwise).
    #[expect(
        clippy::too_many_arguments,
        reason = "find_with's arguments plus the count"
    )]
    fn lookup<const COUNT: bool>(
        &self,
        layout: &RowLayout,
        keys: &EncodedKeys,
        which: &[u32],
        src: &impl RowSource,
        out: &mut Vec<u32>,
        prefetch: bool,
        count: &mut ProbeCount,
    ) {
        let hashes = keys.hashes();
        let heap = keys.heap();
        let g = |k: u32| finalize(hashes.get(k as usize).copied().unwrap_or(0), self.seed);
        match &self.kind {
            Kind::Flat(t) => out.extend(which.iter().map(|&k| {
                let key = keys.row(k as usize);
                let mut compares = 0_u64;
                let found = t
                    .find(g(k), |&id| {
                        if COUNT {
                            compares += 1;
                        }
                        let (row, h) = src.row(id);
                        layout.eq(key, heap, row, h)
                    })
                    .copied()
                    .unwrap_or(NO_MATCH);
                if COUNT {
                    count.add(compares);
                }
                found
            })),
            Kind::Salted(slots) => {
                let mask = slots.len() - 1;
                let mut gs = [0u64; CHUNK];
                for chunk in which.chunks(CHUNK) {
                    for (dst, &k) in gs.iter_mut().zip(chunk) {
                        *dst = g(k);
                    }
                    let gs = &gs[..chunk.len()];
                    if prefetch {
                        prefetch_chunk(slots, gs, src);
                    }
                    out.extend(chunk.iter().zip(gs).map(|(&k, &g)| {
                        let key = keys.row(k as usize);
                        let salt = g & SALT_MASK;
                        let mut i = low(g, mask);
                        let mut compares = 0_u64;
                        let found = loop {
                            let s = slots[i];
                            if s == 0 {
                                break NO_MATCH;
                            }
                            if s & SALT_MASK == salt {
                                let id = slot_id(s);
                                let (row, h) = src.row(id);
                                if COUNT {
                                    compares += 1;
                                }
                                if layout.eq(key, heap, row, h) {
                                    break id;
                                }
                            }
                            i = (i + 1) & mask;
                        };
                        if COUNT {
                            count.add(compares);
                        }
                        found
                    }));
                }
            }
        }
    }
}

#[cfg(test)]
impl JoinIndex {
    /// The probe run of a row with stored hash `h` in a salted directory:
    /// the occupied slots from its first slot to the next empty one, which
    /// bounds the slots any lookup of it reads; `None` for the flat index.
    pub(crate) fn probe_run(&self, h: u64) -> Option<usize> {
        let Kind::Salted(slots) = &self.kind else {
            return None;
        };
        let mask = slots.len() - 1;
        let mut i = low(finalize(h, self.seed), mask);
        let mut run = 0;
        while slots[i] != 0 && run < slots.len() {
            run += 1;
            i = (i + 1) & mask;
        }
        Some(run)
    }
}

/// A directory of twice the slots holding the slots of `old`, re-inserted
/// from the table-local hashes `g` of their rows (no key reads).
fn grow_salted(old: &[u64], g: impl Fn(u32) -> u64) -> Result<Vec<u64>> {
    let len = old.len().saturating_mul(2).max(MIN_SLOTS);
    let mut slots: Vec<u64> = Vec::new();
    slots
        .try_reserve_exact(len)
        .map_err(|e| internal(e.to_string()))?;
    slots.resize(len, 0);
    let mask = len - 1;
    for &s in old.iter().filter(|&&s| s != 0) {
        let mut i = low(g(slot_id(s)), mask);
        while slots[i] != 0 {
            i = (i + 1) & mask;
        }
        slots[i] = s;
    }
    Ok(slots)
}

/// Prefetches every slot of a chunk, then the first salt-matching row of
/// each, so their cache misses overlap before the chunk is resolved in order.
fn prefetch_chunk(slots: &[u64], gs: &[u64], src: &impl RowSource) {
    let mask = slots.len() - 1;
    for &g in gs {
        prefetch_read(slots, low(g, mask));
    }
    for &g in gs {
        let salt = g & SALT_MASK;
        let mut i = low(g, mask);
        loop {
            let s = slots[i];
            if s == 0 {
                break;
            }
            if s & SALT_MASK == salt {
                src.prefetch(slot_id(s));
                break;
            }
            i = (i + 1) & mask;
        }
    }
}
