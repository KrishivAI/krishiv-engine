// SPDX-License-Identifier: Apache-2.0
//! [`JoinIndex`]: the key index of one hash join partition over rows of a
//! [`RowLayout`] (P2-T158), size-adaptive between two designs:
//!
//! - **flat**: `hashbrown::HashTable<u32>` of row ids (SIMD group probing,
//!   about 5 bytes per bucket at up to 7/8 load) while the index is small
//!   enough to stay in the core-private caches, where its lower
//!   instruction count wins;
//! - **salted**: a power-of-two directory of `u64` slots, each a 16-bit
//!   salt plus `row id + 1`, probed linearly at load ≤ 1/2 (DuckDB's
//!   `ht_entry_t`, with row ids instead of pointers), above that size,
//!   where a probe costs one directory miss and reads a row only when its
//!   salt matches.
//!
//! Measured over the same rows, the salted directory probes faster once the
//! index outgrows the last-level cache and slower while it fits; the size
//! rule keeps each regime's winner.
//! The kind is chosen once per table from its row count, the upper bound
//! of its distinct keys ([`FLAT_MAX_BYTES`] was set from the
//! `join_index_matrix` sweep).
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

use super::layout::{EncodedKeys, RowLayout, finalize};
use crate::error::{Result, internal};

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

/// Largest flat index, in bytes, before a table takes the salted
/// directory. On the `join_index_matrix` sweep (`Int64` keys, 2N probes,
/// interleaved ABBA, n = 5, one thread, 12 MiB L3) the flat index probed
/// 1.0-1.4x faster up to 768 Ki keys (2^20 buckets, 5 MiB); from 1.2 Mi
/// keys (2^21 buckets, 10.5 MiB) the touched salted directory was 1.25-1.8x
/// faster on hits and even or faster on half misses. A size between the
/// two bucket counts splits them. The best value depends on the cache
/// sizes; `join_index_matrix` re-measures it.
pub const FLAT_MAX_BYTES: usize = 8 << 20;
/// Smallest salted directory, in bytes, whose lookups first touch a
/// chunk's slots and rows: the same sweep measured the touch pass at
/// 0.6-1.0x (a loss) up to 8 MiB directories and 1.03-1.28x from 16 MiB.
/// Every directory the size rule picks is at least this large.
pub const TOUCH_MIN_BYTES: usize = 16 << 20;

/// Where an index finds its rows.
pub trait RowSource {
    /// Row `id`'s bytes and the heap its long views point into.
    fn row(&self, id: u32) -> (&[u8], &[u8]);
    /// Row `id`'s stored hash `h`.
    fn hash(&self, id: u32) -> u64;
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
    /// Indexes rows `0..rows` of `src`, sized once for `rows` keys (no
    /// resize). For each row whose key is already indexed, `next[row]` is
    /// set to the previous newest row with that key and the index points to
    /// `row` (newest first); other entries of `next` are left unchanged.
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
        let kind = if choice.flat(rows) {
            let mut t: HashTable<u32> = HashTable::new();
            t.try_reserve(rows, |&id| g_of(id))
                .map_err(|e| internal(format!("{e:?}")))?;
            for id in 0..n {
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
            let len = salted_slots(rows);
            let mut slots: Vec<u64> = Vec::new();
            slots.try_reserve_exact(len).map_err(oom)?;
            slots.resize(len, 0);
            let mask = len - 1;
            for id in 0..n {
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
    /// key, or [`NO_MATCH`], appended to `out`. A salted directory of at
    /// least [`TOUCH_MIN_BYTES`] touches each chunk first.
    pub fn find(
        &self,
        layout: &RowLayout,
        keys: &EncodedKeys,
        which: &[u32],
        src: &impl RowSource,
        out: &mut Vec<u32>,
    ) {
        let touch =
            matches!(&self.kind, Kind::Salted(s) if s.len() * size_of::<u64>() >= TOUCH_MIN_BYTES);
        self.find_with(layout, keys, which, src, out, touch);
    }

    /// [`find`](Self::find), choosing whether a salted lookup first touches
    /// each chunk's slots and first salt-matching rows (software
    /// prefetching with demand loads: P2-T157 §8 enables it only where an
    /// interleaved benchmark shows a win, which [`find`](Self::find)
    /// follows). The flat index ignores `touch`.
    pub fn find_with(
        &self,
        layout: &RowLayout,
        keys: &EncodedKeys,
        which: &[u32],
        src: &impl RowSource,
        out: &mut Vec<u32>,
        touch: bool,
    ) {
        let hashes = keys.hashes();
        let heap = keys.heap();
        let g = |k: u32| finalize(hashes.get(k as usize).copied().unwrap_or(0), self.seed);
        match &self.kind {
            Kind::Flat(t) => out.extend(which.iter().map(|&k| {
                let key = keys.row(k as usize);
                t.find(g(k), |&id| {
                    let (row, h) = src.row(id);
                    layout.eq(key, heap, row, h)
                })
                .copied()
                .unwrap_or(NO_MATCH)
            })),
            Kind::Salted(slots) => {
                let mask = slots.len() - 1;
                let mut gs = [0u64; CHUNK];
                for chunk in which.chunks(CHUNK) {
                    for (dst, &k) in gs.iter_mut().zip(chunk) {
                        *dst = g(k);
                    }
                    let gs = &gs[..chunk.len()];
                    if touch {
                        touch_chunk(slots, gs, src);
                    }
                    out.extend(chunk.iter().zip(gs).map(|(&k, &g)| {
                        let key = keys.row(k as usize);
                        let salt = g & SALT_MASK;
                        let mut i = low(g, mask);
                        loop {
                            let s = slots[i];
                            if s == 0 {
                                return NO_MATCH;
                            }
                            if s & SALT_MASK == salt {
                                let id = slot_id(s);
                                let (row, h) = src.row(id);
                                if layout.eq(key, heap, row, h) {
                                    return id;
                                }
                            }
                            i = (i + 1) & mask;
                        }
                    }));
                }
            }
        }
    }
}

/// Loads every slot of a chunk, then the first salt-matching row of each,
/// so their cache misses overlap before the chunk is resolved in order.
fn touch_chunk(slots: &[u64], gs: &[u64], src: &impl RowSource) {
    let mask = slots.len() - 1;
    for &g in gs {
        std::hint::black_box(slots[low(g, mask)]);
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
                std::hint::black_box(src.row(slot_id(s)).0.first());
                break;
            }
            i = (i + 1) & mask;
        }
    }
}
