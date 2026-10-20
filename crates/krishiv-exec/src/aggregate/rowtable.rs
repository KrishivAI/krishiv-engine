// SPDX-License-Identifier: Apache-2.0
//! Row-layout aggregation tables (P2-T159, `krishiv.exec.agg.rowLayout`).
//!
//! A [`RowTable`] keeps its groups' keys as fixed-width [`RowLayout`] rows,
//! one row per group, in [`PARTS`] radix partitions chosen by the top bits
//! of the row hash; the aggregates' states stay in the same
//! [`Accumulator`]s as the default path, indexed by a group id written into
//! the row. A row is
//!
//! ```text
//! | h u64 | key validity | keys ... | group id u32 | pad |
//! ```
//!
//! # Hashes (P2-T157 Appendix C)
//! - `h`, stored at the start of the row, is seedless: the row layout's
//!   column-wise hash with seed 0, the same for every operator instance and
//!   worker. Radix partitions are its top [`RADIX_BITS`]; the per-partition
//!   [`Hll`] sketches are over it, so they merge across tables.
//! - The index and spill partitions use `g = fin(h ^ seed)`, one folded
//!   multiply with the operator's seed, so crafted keys cannot be aimed at
//!   one index region or one spill file without the seed. `g` never leaves
//!   the operator instance.
//!
//! # Index modes
//! The index is a flat `hashbrown::HashTable<u32>` of row references
//! (`partition << ROW_BITS | row`) hashed by `g`.
//! - **Fixed** (thread-local tables): a fixed capacity, cleared when full;
//!   the rows stay in their partitions, so a group seen again later becomes
//!   a second row (a partial duplicate merged by the final phase). Every
//!   [`ADAPT_ROWS`] input rows, an [`Hll`] of a 1/8 sample of the input
//!   hashes grows the capacity 4x when deduplication could halve the rows,
//!   or switches to **append** (no lookups) when over 95% of the keys are
//!   unique. This is DuckDB's thread-local abandon (fixed-size thread-local
//!   tables, <https://duckdb.org/2022/03/07/aggregate-hashtable.html>;
//!   `radix_partitioned_hashtable.cpp`) with its capacities: 2^15 slots
//!   above two threads, 2^18 at two.
//! - **Grow**: a classic growing table that never abandons. Deterministic
//!   mode (ADR 0023) uses it for thread-local tables, because when a
//!   fixed table abandons depends on which morsels a driver received. One
//!   thread uses it too: with no other driver to merge with, partial
//!   duplicates only cost (a laptop ABBA measured TPC-H Q17 16% slower at
//!   one thread with a fixed index). The final phase and the spill merge
//!   use it for their merge tables.
//!
//! # Memory
//! [`projected`](RowTable::projected) bounds the bytes after an apply
//! before anything changes (ADR 0007): partition rows by the batch's rows
//! per partition, the index by its mode, the accumulators as on the default
//! path.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use hashbrown::HashTable;
use krishiv_types::batch::{Batch, Weights};

use super::accumulator::Accumulator;
use super::table::{AggCore, batch};
use super::{AggregateOutput, unsupported};
use crate::error::{Result, internal};
use crate::hash::{HashSeed, PARTITIONS, grown, spill_partition};
use crate::rows::{EncodedKeys, RowLayout};

/// Radix bits of `h` per table (DuckDB's initial 4 bits; the final phase
/// merges ranges of these partitions, see [`final_tasks`]).
pub(crate) const RADIX_BITS: u32 = 4;
/// Radix partitions per table.
pub(crate) const PARTS: usize = 1 << RADIX_BITS;
/// Bits of a row reference for the row within its partition.
const ROW_BITS: u32 = 32 - RADIX_BITS;
const ROW_MASK: u32 = (1 << ROW_BITS) - 1;
/// Bytes after the key region: the group id and padding to 8.
const TAIL: usize = 8;
/// Index capacity (slots) of a thread-local table above two threads, and at
/// most two threads (DuckDB's sink capacities); entries up to 2/3 of it.
const SINK_SLOTS: usize = 1 << 15;
const SINK_SLOTS_FEW_THREADS: usize = 1 << 18;
/// Largest capacity the adaptation grows a fixed index to.
const MAX_SINK_SLOTS: usize = 1 << 22;
/// Input rows between adaptations of a fixed index.
pub(crate) const ADAPT_ROWS: usize = 1 << 20;
/// Bytes per index bucket: a `u32` reference and hashbrown's control byte.
const BUCKET_BYTES: usize = size_of::<u32>() + 1;
/// Odd constant of the `g` finalizer (the fractional digits of sqrt(3)).
const G_MIX: u64 = 0xBB67_AE85_84CA_A73B;

/// How a table's index treats a full index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Grows with its groups.
    Grow,
    /// At most `limit` keys, cleared when full.
    Fixed { limit: usize },
    /// No index: every row is a new group.
    Append,
}

impl Mode {
    /// A thread-local table's starting mode (see the module docs).
    pub(crate) fn local(deterministic: bool, threads: usize) -> Self {
        if deterministic || threads <= 1 {
            Self::Grow
        } else {
            let slots = if threads <= 2 {
                SINK_SLOTS_FEW_THREADS
            } else {
                SINK_SLOTS
            };
            Self::Fixed {
                limit: slots * 2 / 3,
            }
        }
    }
}

/// `g = fin(h ^ seed)` (P2-T157 Appendix C): one folded multiply.
#[inline]
fn fin(h: u64, seed: u64) -> u64 {
    let p = u128::from(h ^ seed) * u128::from(G_MIX);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the low half of the product is intended"
    )]
    let low = p as u64;
    low ^ ((p >> 64) as u64)
}

/// The radix partition of `h`.
#[inline]
pub(crate) fn part_of(h: u64) -> usize {
    usize::try_from(h >> (64 - RADIX_BITS)).unwrap_or(0)
}

#[inline]
fn word(row: &[u8], at: usize) -> u64 {
    row.get(at..at + 8)
        .and_then(|s| <[u8; 8]>::try_from(s).ok())
        .map_or(0, u64::from_le_bytes)
}

/// A HyperLogLog sketch (Flajolet et al. 2007) with `2^bits` one-byte
/// registers over the bits of `h` below the top 8 (so the radix bits, equal
/// within a partition, do not bias it), with linear counting for small
/// counts. Seedless and mergeable: the sketches of one partition from every
/// table merge into that partition's estimate. Standard error
/// `1.04 / 2^(bits/2)`: 3.3% for the partition sketches ([`PART_HLL_BITS`],
/// enough to presize an index), 1.6% for the adaptation's ([`ADAPT_HLL_BITS`],
/// which decides at 95% unique).
#[derive(Clone, Debug)]
pub(crate) struct Hll {
    bits: u32,
    registers: Box<[u8]>,
}

/// Register bits of the per-partition sketches.
const PART_HLL_BITS: u32 = 10;
/// Register bits of a fixed index's input sample.
const ADAPT_HLL_BITS: u32 = 12;
/// Top bits of `h` the sketch skips (radix bits, up to 8).
const HLL_SKIP: u32 = 8;

impl Hll {
    pub(crate) fn new(bits: u32) -> Self {
        Self {
            bits,
            registers: vec![0; 1 << bits].into_boxed_slice(),
        }
    }

    #[inline]
    pub(crate) fn add(&mut self, h: u64) {
        let rest = h << HLL_SKIP;
        let i = usize::try_from(rest >> (64 - self.bits)).unwrap_or(0);
        let w = rest << self.bits;
        let max = 64 - self.bits - HLL_SKIP + 1;
        let rank = u8::try_from((w.leading_zeros() + 1).min(max)).unwrap_or(u8::MAX);
        if let Some(r) = self.registers.get_mut(i) {
            *r = (*r).max(rank);
        }
    }

    /// Merges a sketch with as many registers (others are ignored).
    pub(crate) fn merge(&mut self, o: &Self) {
        if o.bits == self.bits {
            for (a, b) in self.registers.iter_mut().zip(o.registers.iter()) {
                *a = (*a).max(*b);
            }
        }
    }

    /// The distinct count estimate.
    #[expect(
        clippy::cast_precision_loss,
        reason = "an estimate: register counts are small"
    )]
    pub(crate) fn estimate(&self) -> f64 {
        let m = self.registers.len() as f64;
        let (mut sum, mut zeros) = (0.0, 0usize);
        for &r in self.registers.iter() {
            sum += (-f64::from(r)).exp2();
            zeros += usize::from(r == 0);
        }
        let e = 0.7213 / (1.0 + 1.079 / m) * m * m / sum;
        if e <= 2.5 * m && zeros > 0 {
            m * (m / zeros as f64).ln()
        } else {
            e
        }
    }

    pub(crate) fn bytes(&self) -> usize {
        self.registers.len()
    }
}

/// The input sample a fixed index adapts from.
#[derive(Debug)]
struct Adapt {
    sketch: Hll,
    rows: usize,
    /// Rows pushed (groups created) so far.
    pushed: usize,
    next: usize,
}

/// A batch prepared for [`RowTable::apply`].
pub(crate) struct RowPrepared {
    keys: EncodedKeys,
    /// Rows per radix partition (bounds each partition's growth).
    counts: [usize; PARTS],
    cols: Vec<ArrayRef>,
    weights: Weights,
    states: bool,
}

/// A chunk of another table's rows to merge (the final phase).
pub(crate) struct MergeChunk {
    /// The radix partition of every row.
    part: usize,
    /// The source rows (references into the source table).
    refs: Vec<u32>,
    /// Their states, in the state layout (every aggregate's columns).
    states: Vec<ArrayRef>,
}

/// Groups as rows in radix partitions, with their accumulators.
#[derive(Debug)]
pub(crate) struct RowTable {
    core: Arc<AggCore>,
    /// The seed of `g`.
    gseed: u64,
    key_stride: usize,
    stride: usize,
    parts: Vec<Vec<u8>>,
    /// Per-partition sketches of the rows' `h` (thread-local tables).
    sketches: Option<Vec<Hll>>,
    index: HashTable<u32>,
    mode: Mode,
    adapt: Option<Adapt>,
    accs: Vec<Accumulator>,
    groups: usize,
    /// No key has two rows: the index was never cleared or dropped with
    /// rows in the table, and no row was appended without a lookup.
    unique: bool,
    ids: Vec<u32>,
}

impl RowTable {
    /// A table in `mode`; `local` tables keep per-partition sketches for
    /// the final phase, and adapt a fixed index.
    pub(crate) fn new(
        core: &Arc<AggCore>,
        seed: &HashSeed,
        mode: Mode,
        local: bool,
    ) -> Result<Self> {
        let layout = layout(core)?;
        let accs = core
            .spec
            .aggregates
            .iter()
            .map(|a| Accumulator::new(a, core.spec.ansi))
            .collect::<Result<Vec<_>>>()?;
        let key_stride = layout.stride();
        Ok(Self {
            core: Arc::clone(core),
            gseed: seed.hash(b"krishiv.agg.g"),
            key_stride,
            stride: key_stride + TAIL,
            parts: (0..PARTS).map(|_| Vec::new()).collect(),
            sketches: local.then(|| (0..PARTS).map(|_| Hll::new(PART_HLL_BITS)).collect()),
            index: HashTable::new(),
            mode,
            adapt: (local && matches!(mode, Mode::Fixed { .. })).then(|| Adapt {
                sketch: Hll::new(ADAPT_HLL_BITS),
                rows: 0,
                pushed: 0,
                next: ADAPT_ROWS,
            }),
            accs,
            groups: 0,
            unique: true,
            ids: Vec::new(),
        })
    }

    /// Number of rows (groups; a group may have several rows in a table
    /// whose fixed index was cleared).
    pub(crate) fn len(&self) -> usize {
        self.groups
    }

    /// Whether every group has one row (no partial duplicates).
    pub(crate) fn is_unique(&self) -> bool {
        self.unique
    }

    /// Rows in partition `p`.
    pub(crate) fn part_len(&self, p: usize) -> usize {
        self.parts.get(p).map_or(0, |v| v.len() / self.stride)
    }

    /// The per-partition sketches (thread-local tables).
    pub(crate) fn sketches(&self) -> Option<&[Hll]> {
        self.sketches.as_deref()
    }

    pub(crate) fn memory_size(&self) -> usize {
        self.parts.iter().map(Vec::capacity).sum::<usize>()
            + buckets(&self.index) * BUCKET_BYTES
            + self.accs.iter().map(Accumulator::size).sum::<usize>()
            + self.ids.capacity() * size_of::<u32>()
            + self.sketch_bytes()
    }

    fn sketch_bytes(&self) -> usize {
        self.sketches
            .iter()
            .flatten()
            .map(Hll::bytes)
            .sum::<usize>()
            + self.adapt.as_ref().map_or(0, |a| a.sketch.bytes())
    }

    /// Encodes the keys of a batch (`states`: in the state layout).
    pub(crate) fn prepare(&self, b: &Batch, states: bool) -> Result<RowPrepared> {
        let cols = b.data().columns();
        let keys: Vec<ArrayRef> = if states {
            cols.get(..self.core.spec.group_by.len())
                .ok_or_else(|| internal("state batch without its keys".to_owned()))?
                .to_vec()
        } else {
            self.core
                .spec
                .group_by
                .iter()
                .map(|&k| {
                    cols.get(k)
                        .cloned()
                        .ok_or_else(|| internal(format!("grouping column {k} missing")))
                })
                .collect::<Result<_>>()?
        };
        // Seed 0: `h` is seedless (Appendix C).
        let keys = layout(&self.core)?.encode(&keys, 0)?;
        let mut counts = [0; PARTS];
        for &h in keys.hashes() {
            counts[part_of(h)] += 1;
        }
        Ok(RowPrepared {
            keys,
            counts,
            cols: cols.to_vec(),
            weights: b.weights().clone(),
            states,
        })
    }

    /// The mode the next apply runs in (a fixed index adapts every
    /// [`ADAPT_ROWS`] input rows).
    fn next_mode(&self) -> Mode {
        let (Mode::Fixed { limit }, Some(a)) = (self.mode, &self.adapt) else {
            return self.mode;
        };
        if a.rows < a.next {
            return self.mode;
        }
        #[expect(clippy::cast_precision_loss, reason = "row counts in a ratio")]
        let (rows, pushed) = (a.rows as f64, a.pushed.max(1) as f64);
        // The sample is 1/8 of the hashes.
        let distinct = (a.sketch.estimate() * 8.0).max(1.0);
        if distinct / rows > 0.95 {
            Mode::Append
        } else if pushed >= 2.0 * distinct && limit * 3 / 2 < MAX_SINK_SLOTS {
            Mode::Fixed { limit: limit * 4 }
        } else {
            self.mode
        }
    }

    /// Index bytes in `mode` after `new` more keys: a growing index grows
    /// with its keys, a fixed one up to its limit (it is cleared there).
    fn index_bytes(&self, mode: Mode, new: usize) -> usize {
        let len = match mode {
            Mode::Append => return 0,
            Mode::Fixed { limit } => self.index.len().saturating_add(new).min(limit),
            Mode::Grow => self.index.len().saturating_add(new),
        };
        let buckets = if len > self.index.capacity() {
            buckets_for(len.max(self.index.capacity() + 1))
        } else {
            buckets(&self.index)
        };
        buckets * BUCKET_BYTES
    }

    /// Grows the index for `new` more keys, as [`index_bytes`](Self::index_bytes)
    /// assumed.
    fn reserve_index(&mut self, new: usize) {
        let len = match self.mode {
            Mode::Append => return,
            Mode::Fixed { limit } => self.index.len().saturating_add(new).min(limit),
            Mode::Grow => self.index.len().saturating_add(new),
        };
        let (parts, stride, gseed) = (&self.parts, self.stride, self.gseed);
        self.index
            .reserve(len.saturating_sub(self.index.len()), |&r| {
                fin(word(row_at(parts, stride, r), 0), gseed)
            });
    }

    /// [`memory_size`](Self::memory_size) after [`apply`](Self::apply)ing
    /// `p` (an upper bound).
    pub(crate) fn projected(&self, p: &RowPrepared) -> usize {
        let n = p.keys.len();
        let rows: usize = self
            .parts
            .iter()
            .zip(p.counts)
            .map(|(v, c)| grown(v, c * self.stride))
            .sum();
        let accs: usize = self
            .accs
            .iter()
            .zip(&self.core.ranges)
            .map(|(a, r)| {
                let at = if p.states {
                    Some(r.start)
                } else {
                    a.args().first().copied()
                };
                a.projected(n, at.and_then(|i| p.cols.get(i)))
            })
            .sum();
        rows + self.index_bytes(self.next_mode(), n)
            + accs
            + grown(&self.ids, n) * size_of::<u32>()
            + self.sketch_bytes()
    }

    /// Switches the index to `mode` (append-only tables drop it; a fixed
    /// index keeps its keys under a new limit).
    fn switch(&mut self, mode: Mode) {
        if mode == Mode::Append && self.mode != Mode::Append {
            self.unique &= self.groups == 0;
            self.index = HashTable::new();
        }
        self.mode = mode;
        if let Some(a) = self.adapt.as_mut()
            && a.rows >= a.next
        {
            a.next = a.rows + ADAPT_ROWS;
        }
    }

    /// Folds a prepared batch in. Call after reserving
    /// [`projected`](Self::projected) bytes.
    pub(crate) fn apply(&mut self, p: &RowPrepared) -> Result<()> {
        let n = p.keys.len();
        self.switch(self.next_mode());
        if let Some(a) = self.adapt.as_mut() {
            a.rows += n;
            for &h in p.keys.hashes().iter().filter(|&&h| h & 7 == 0) {
                a.sketch.add(h);
            }
        }
        for (v, c) in self.parts.iter_mut().zip(p.counts) {
            v.reserve(c * self.stride);
        }
        let mut ids = std::mem::take(&mut self.ids);
        ids.clear();
        ids.reserve(n);
        let before = self.groups;
        self.reserve_index(n);
        let result = self.lookup(&p.keys, &mut ids);
        if let Some(a) = self.adapt.as_mut() {
            a.pushed += self.groups - before;
        }
        self.accs.iter_mut().for_each(|a| a.resize(self.groups));
        let result = result.and_then(|()| {
            if p.states {
                let core = Arc::clone(&self.core);
                self.accs
                    .iter_mut()
                    .zip(&core.ranges)
                    .try_for_each(|(a, r)| a.merge(&ids, &p.cols[r.clone()]))
            } else {
                self.accs
                    .iter_mut()
                    .try_for_each(|a| a.update(&ids, &p.cols, &p.weights))
            }
        });
        self.ids = ids;
        result
    }

    /// The group id of each row of `keys`, inserting new groups.
    fn lookup(&mut self, keys: &EncodedKeys, ids: &mut Vec<u32>) -> Result<()> {
        let hashes = keys.hashes();
        match self.mode {
            Mode::Append => {
                self.unique &= keys.is_empty();
                for (k, &h) in hashes.iter().enumerate() {
                    let r = self.push(keys.row(k), h)?;
                    ids.push(self.gid(r));
                }
            }
            Mode::Fixed { limit } => {
                for (k, &h) in hashes.iter().enumerate() {
                    ids.push(self.find_or_insert(keys.row(k), h, limit)?);
                }
            }
            Mode::Grow => {
                for (k, &h) in hashes.iter().enumerate() {
                    ids.push(self.find_or_insert(keys.row(k), h, usize::MAX)?);
                }
            }
        }
        Ok(())
    }

    /// The group of key row `key` (with hash `h`), inserting it if new; an
    /// index holding `limit` keys is cleared before it takes another (the
    /// rows stay: a key seen again gets a new row).
    #[inline(always)]
    fn find_or_insert(&mut self, key: &[u8], h: u64, limit: usize) -> Result<u32> {
        let Self {
            parts,
            sketches,
            groups,
            index,
            unique,
            stride,
            key_stride,
            gseed,
            ..
        } = self;
        let (stride, key_stride, gseed) = (*stride, *key_stride, *gseed);
        let g = fin(h, gseed);
        // A lookup first (hashbrown inlines `find`; most rows are hits),
        // then an insert on a miss.
        let found = index
            .find(g, |&r| same_key(row_at(parts, stride, r), key, key_stride))
            .copied();
        let r = match found {
            Some(r) => r,
            None => {
                let r = push_row(parts, sketches, groups, stride, key, h)?;
                if index.len() >= limit {
                    index.clear();
                    *unique = false;
                }
                index.insert_unique(g, r, |&o| fin(word(row_at(parts, stride, o), 0), gseed));
                r
            }
        };
        Ok(gid_at(parts, stride, key_stride, r))
    }

    fn push(&mut self, key: &[u8], h: u64) -> Result<u32> {
        push_row(
            &mut self.parts,
            &mut self.sketches,
            &mut self.groups,
            self.stride,
            key,
            h,
        )
    }

    /// The group id stored in row `r`.
    #[inline]
    fn gid(&self, r: u32) -> u32 {
        gid_at(&self.parts, self.stride, self.key_stride, r)
    }

    /// The row references at positions `range` of the partition order
    /// (partition 0's rows, then partition 1's, ...).
    pub(crate) fn ids(&self, range: Range<usize>) -> Vec<u32> {
        let mut out = Vec::with_capacity(range.len());
        let mut start = 0;
        for p in 0..PARTS {
            let len = self.part_len(p);
            let (from, to) = (range.start.max(start), range.end.min(start + len));
            out.extend((from..to).map(|i| ref_of(p, i - start)));
            start += len;
        }
        out
    }

    /// The references of rows `rows` of partition `p`.
    pub(crate) fn part_ids(p: usize, rows: Range<usize>) -> Vec<u32> {
        rows.map(|i| ref_of(p, i)).collect()
    }

    /// Row references per spill partition at `level` (by `g`) of the rows
    /// at positions `from..` of the partition order (see [`ids`](Self::ids)).
    pub(crate) fn ids_by_partition(&self, level: u32, from: usize) -> Vec<Vec<u32>> {
        let mut out = vec![Vec::new(); PARTITIONS];
        let mut start = 0;
        for p in 0..PARTS {
            let skip = from.saturating_sub(start);
            start += self.part_len(p);
            for (i, row) in self.parts[p]
                .chunks_exact(self.stride)
                .enumerate()
                .skip(skip)
            {
                let g = fin(word(row, 0), self.gseed);
                out[spill_partition(g, level)].push(ref_of(p, i));
            }
        }
        out
    }

    /// Row indices of a state batch per spill partition at `level`.
    pub(crate) fn partitions_of(&self, b: &RecordBatch, level: u32) -> Result<Vec<Vec<u32>>> {
        let keys = b
            .columns()
            .get(..self.core.spec.group_by.len())
            .ok_or_else(|| internal("state batch without its keys".to_owned()))?;
        let keys = layout(&self.core)?.encode(keys, 0)?;
        let mut parts = vec![Vec::new(); PARTITIONS];
        for (i, &h) in (0u32..).zip(keys.hashes()) {
            parts[spill_partition(fin(h, self.gseed), level)].push(i);
        }
        Ok(parts)
    }

    fn rows_of(&self, refs: &[u32]) -> Result<Vec<&[u8]>> {
        refs.iter()
            .map(|&r| {
                let row = row_at(&self.parts, self.stride, r);
                if row.is_empty() {
                    Err(internal(format!("row reference {r:#x} out of range")))
                } else {
                    Ok(row)
                }
            })
            .collect()
    }

    fn gids_of(&self, refs: &[u32]) -> Vec<u32> {
        refs.iter().map(|&r| self.gid(r)).collect()
    }

    fn key_columns(&self, refs: &[u32]) -> Result<Vec<ArrayRef>> {
        layout(&self.core)?.decode(&self.rows_of(refs)?)
    }

    /// The partial states of rows `refs`, in the state layout.
    pub(crate) fn state_batch(&self, refs: &[u32]) -> Result<RecordBatch> {
        let mut cols = self.key_columns(refs)?;
        let gids = self.gids_of(refs);
        for a in &self.accs {
            cols.extend(a.state(&gids)?);
        }
        batch(&self.core.state_schema, cols, refs.len())
    }

    /// The node's output for rows `refs`: final values or partial states.
    pub(crate) fn output_batch(&self, refs: &[u32]) -> Result<RecordBatch> {
        if self.core.spec.output == AggregateOutput::Partial {
            return self.state_batch(refs);
        }
        let mut cols = self.key_columns(refs)?;
        let gids = self.gids_of(refs);
        for a in &self.accs {
            cols.push(a.evaluate(&gids)?);
        }
        batch(&self.core.output_schema, cols, refs.len())
    }

    /// Rows `rows` of partition `p` of this table, with their states, to
    /// merge into another table of the same aggregate.
    pub(crate) fn merge_chunk(&self, p: usize, rows: Range<usize>) -> Result<MergeChunk> {
        let refs = Self::part_ids(p, rows);
        let gids = self.gids_of(&refs);
        let mut states = Vec::new();
        for a in &self.accs {
            states.extend(a.state(&gids)?);
        }
        Ok(MergeChunk {
            part: p,
            refs,
            states,
        })
    }

    /// [`memory_size`](Self::memory_size) after
    /// [`apply_merge`](Self::apply_merge) of `c` from `src`.
    pub(crate) fn projected_merge(&self, c: &MergeChunk) -> usize {
        let n = c.refs.len();
        let rows: usize = self
            .parts
            .iter()
            .enumerate()
            .map(|(p, v)| {
                if p == c.part {
                    grown(v, n * self.stride)
                } else {
                    v.capacity()
                }
            })
            .sum();
        let accs: usize = self
            .accs
            .iter()
            .zip(&self.core.ranges)
            .map(|(a, r)| {
                let at = r.start - self.core.spec.group_by.len();
                a.projected(n, c.states.get(at))
            })
            .sum();
        rows + self.index_bytes(self.mode, n)
            + accs
            + grown(&self.ids, n) * size_of::<u32>()
            + self.sketch_bytes()
    }

    /// Merges a chunk of `src`'s rows (rows are copied, states merged).
    /// Call after reserving [`projected_merge`](Self::projected_merge).
    pub(crate) fn apply_merge(&mut self, src: &Self, c: &MergeChunk) -> Result<()> {
        if self.mode != Mode::Grow {
            return Err(internal(
                "merging into a table that is not growing".to_owned(),
            ));
        }
        let core = Arc::clone(&self.core);
        let mut ids = std::mem::take(&mut self.ids);
        ids.clear();
        ids.reserve(c.refs.len());
        self.reserve_index(c.refs.len());
        if let Some(v) = self.parts.get_mut(c.part) {
            v.reserve(c.refs.len() * self.stride);
        }
        let mut result = Ok(());
        for &r in &c.refs {
            let row = row_at(&src.parts, src.stride, r);
            let key = &row[..src.key_stride.min(row.len())];
            match self.find_or_insert(key, word(row, 0), usize::MAX) {
                Ok(id) => ids.push(id),
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        self.accs.iter_mut().for_each(|a| a.resize(self.groups));
        let result = result.and_then(|()| {
            let k = core.spec.group_by.len();
            self.accs
                .iter_mut()
                .zip(&core.ranges)
                .try_for_each(|(a, r)| a.merge(&ids, &c.states[r.start - k..r.end - k]))
        });
        self.ids = ids;
        result
    }
}

/// The table's row layout (present whenever row tables are made).
fn layout(core: &AggCore) -> Result<&RowLayout> {
    core.rows
        .as_ref()
        .ok_or_else(|| unsupported("row-layout aggregation of these key types".to_owned()))
}

/// The partition and row of a row reference.
pub(crate) fn split_ref(r: u32) -> (usize, usize) {
    ((r >> ROW_BITS) as usize, (r & ROW_MASK) as usize)
}

#[inline]
fn ref_of(p: usize, row: usize) -> u32 {
    let p = u32::try_from(p).unwrap_or(0);
    (p << ROW_BITS) | u32::try_from(row).unwrap_or(ROW_MASK)
}

/// Row `r` of `parts` (empty if out of range).
#[inline]
fn row_at(parts: &[Vec<u8>], stride: usize, r: u32) -> &[u8] {
    let p = (r >> ROW_BITS) as usize;
    let at = (r & ROW_MASK) as usize * stride;
    parts
        .get(p)
        .and_then(|v| v.get(at..at + stride))
        .unwrap_or_default()
}

/// Whether `row` holds the key of key row `key`: the same hash and key
/// bytes. Fixed-width rows hold the whole key, NULLs and padding as zero
/// bytes and floats normalized, so equal keys have equal bytes (the
/// layout's word compare without views).
#[inline(always)]
fn same_key(row: &[u8], key: &[u8], key_stride: usize) -> bool {
    match (row.get(..key_stride), key.get(..key_stride)) {
        (Some(a), Some(b)) => {
            let (a, _) = a.as_chunks::<8>();
            let (b, _) = b.as_chunks::<8>();
            a.iter().zip(b).all(|(x, y)| x == y)
        }
        _ => false,
    }
}

/// The group id stored in row `r`.
#[inline(always)]
fn gid_at(parts: &[Vec<u8>], stride: usize, key_stride: usize, r: u32) -> u32 {
    row_at(parts, stride, r)
        .get(key_stride..key_stride + 4)
        .and_then(|s| <[u8; 4]>::try_from(s).ok())
        .map_or(u32::MAX, u32::from_le_bytes)
}

/// Appends a row for key row `key` (hash `h`) with a new group id.
#[inline]
fn push_row(
    parts: &mut [Vec<u8>],
    sketches: &mut Option<Vec<Hll>>,
    groups: &mut usize,
    stride: usize,
    key: &[u8],
    h: u64,
) -> Result<u32> {
    let p = part_of(h);
    let v = &mut parts[p];
    let row = v.len() / stride;
    let gid = u32::try_from(*groups)
        .ok()
        .filter(|&g| g < u32::MAX)
        .ok_or_else(|| internal("more than 2^32 groups in one table".to_owned()))?;
    if row > ROW_MASK as usize {
        return Err(unsupported(format!(
            "more than 2^{ROW_BITS} groups in one radix partition of an aggregation table"
        )));
    }
    let key_stride = stride - TAIL;
    v.extend_from_slice(key.get(..key_stride).unwrap_or(key));
    v.extend_from_slice(&gid.to_le_bytes());
    v.resize(v.len() + (stride - key_stride - 4), 0);
    if let Some(s) = sketches.as_mut() {
        s[p].add(h);
    }
    *groups += 1;
    Ok(ref_of(p, row))
}

/// The buckets an index allocated (an empty one shares a static control
/// group: none).
fn buckets(t: &HashTable<u32>) -> usize {
    if t.capacity() == 0 {
        0
    } else {
        t.num_buckets()
    }
}

/// hashbrown's bucket count for `cap` entries (load factor 7/8).
fn buckets_for(cap: usize) -> usize {
    if cap == 0 {
        0
    } else if cap < 4 {
        4
    } else if cap < 8 {
        8
    } else {
        (cap.saturating_mul(8) / 7).next_power_of_two()
    }
}

/// One final-phase task: radix partitions `parts` of every input table.
#[derive(Debug, Clone)]
pub(crate) struct FinalTask {
    pub(crate) parts: Range<usize>,
    /// Estimated groups (merged sketches, capped by the rows).
    pub(crate) groups: usize,
}

/// Groups per final-phase task the radix bits aim for: a task's index and
/// rows then fit a core's share of the last-level cache (DuckDB sizes
/// over-partitions for the same reason,
/// <https://duckdb.org/2024/03/29/external-aggregation.html>); a starting
/// point, to be tuned.
const TASK_GROUPS: f64 = 32_768.0;

/// The final phase's tasks: the radix bits (`0..=RADIX_BITS`) follow the
/// estimated group count, so small results merge in one task and large
/// ones in up to [`PARTS`] partitions merged in parallel. Partitions
/// without rows get no task.
pub(crate) fn final_tasks(tables: &[&RowTable]) -> Vec<FinalTask> {
    let mut est = Vec::with_capacity(PARTS);
    for p in 0..PARTS {
        let rows: usize = tables.iter().map(|t| t.part_len(p)).sum();
        let mut sketch = Hll::new(PART_HLL_BITS);
        let mut sketched = false;
        for s in tables.iter().filter_map(|t| t.sketches()) {
            sketch.merge(&s[p]);
            sketched = true;
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a non-negative estimate, capped by the row count"
        )]
        let groups = if sketched {
            ((sketch.estimate() * 1.1) as usize + 16).min(rows)
        } else {
            rows
        };
        est.push((rows, groups));
    }
    #[expect(clippy::cast_precision_loss, reason = "a group-count estimate")]
    let total = est.iter().map(|e| e.1).sum::<usize>() as f64;
    let mut bits = 0;
    while bits < RADIX_BITS && total / f64::from(1u32 << bits) > TASK_GROUPS {
        bits += 1;
    }
    let width = PARTS >> bits;
    (0..PARTS)
        .step_by(width)
        .map(|start| {
            let parts = start..start + width;
            let rows: usize = est[parts.clone()].iter().map(|e| e.0).sum();
            let groups = est[parts.clone()].iter().map(|e| e.1).sum();
            (rows, FinalTask { parts, groups })
        })
        .filter(|(rows, _)| *rows > 0)
        .map(|(_, t)| t)
        .collect()
}

impl RowTable {
    /// Presizes a merge table's index for `groups` keys (the final phase's
    /// estimate); returns the bytes that adds.
    pub(crate) fn presize(&mut self, groups: usize) {
        let (parts, stride, gseed) = (&self.parts, self.stride, self.gseed);
        self.index
            .reserve(groups, |&r| fin(word(row_at(parts, stride, r), 0), gseed));
    }

    /// [`memory_size`](Self::memory_size) after [`presize`](Self::presize).
    pub(crate) fn projected_presize(&self, groups: usize) -> usize {
        self.memory_size() - buckets(&self.index) * BUCKET_BYTES
            + self.index_bytes(Mode::Grow, groups)
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::Int64Array;
    use arrow_schema::{DataType, Field, Schema};

    use super::*;
    use crate::aggregate::{AggregateExpr, AggregateFunction, AggregateSpec};

    fn core() -> Arc<AggCore> {
        let s = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("v", DataType::Int64, true),
        ]));
        let spec = AggregateSpec {
            input: Arc::clone(&s),
            group_by: vec![0],
            aggregates: vec![
                AggregateExpr::count_star("c"),
                AggregateExpr::new(AggregateFunction::Sum, 1, DataType::Int64, "s"),
            ],
            output: AggregateOutput::Final,
            ansi: true,
            spill: None,
            require_unit_weights: false,
        };
        Arc::new(AggCore::new(spec).unwrap())
    }

    fn batch(core: &AggCore, keys: impl Iterator<Item = i64>) -> Batch {
        let k: Vec<i64> = keys.collect();
        let v = Int64Array::from(vec![1; k.len()]);
        let cols: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(k)), Arc::new(v)];
        Batch::new(RecordBatch::try_new(Arc::clone(&core.spec.input), cols).unwrap())
    }

    /// Applies after reserving: the projection bounds the bytes after.
    fn apply(t: &mut RowTable, b: &Batch) {
        let p = t.prepare(b, false).unwrap();
        let want = t.projected(&p);
        t.apply(&p).unwrap();
        assert!(t.memory_size() <= want, "{} > {want}", t.memory_size());
    }

    /// A seedless 64-bit mix (splitmix64's finalizer) for test hashes.
    fn mix(mut x: u64) -> u64 {
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    #[test]
    fn hll_estimates_and_merges() {
        for n in [0u64, 10, 1_000, 100_000] {
            let mut h = Hll::new(PART_HLL_BITS);
            for i in 0..n {
                h.add(mix(i));
                h.add(mix(i));
            }
            #[expect(clippy::cast_precision_loss, reason = "test counts")]
            let n = n as f64;
            let e = h.estimate();
            assert!((e - n).abs() <= n * 0.1 + 1.0, "{e} for {n}");
        }
        let (mut a, mut b) = (Hll::new(PART_HLL_BITS), Hll::new(PART_HLL_BITS));
        (0..50_000).for_each(|i| a.add(mix(i)));
        (25_000..75_000).for_each(|i| b.add(mix(i)));
        a.merge(&b);
        assert!(
            (a.estimate() - 75_000.0).abs() < 7_500.0,
            "{}",
            a.estimate()
        );
    }

    /// A fixed index is cleared when full: a key seen again gets a second
    /// row, so the table is no longer unique; growing and append-only
    /// tables behave as their modes say.
    #[test]
    fn modes_and_partial_duplicates() {
        let core = core();
        let seed = HashSeed::new(Some(1));
        let mut grow = RowTable::new(&core, &seed, Mode::Grow, false).unwrap();
        let mut fixed = RowTable::new(&core, &seed, Mode::Fixed { limit: 8 }, true).unwrap();
        let mut append = RowTable::new(&core, &seed, Mode::Append, true).unwrap();
        for t in [&mut grow, &mut fixed, &mut append] {
            apply(t, &batch(&core, 0..20));
            apply(t, &batch(&core, (0..20).map(|i| i % 4)));
        }
        assert_eq!((grow.len(), grow.is_unique()), (20, true));
        // Cleared at 8 keys: 16..19 are left, 0..3 come again.
        assert_eq!((fixed.len(), fixed.is_unique()), (24, false));
        assert_eq!((append.len(), append.is_unique()), (40, false));
        // Every row is in its partition of `h`, and positions cover them.
        let all = append.ids(0..append.len());
        assert_eq!(all.len(), 40);
        // Spilling from a position skips exactly the rows before it.
        for from in [0, 1, 17, 39, 40, 41] {
            let mut got: Vec<u32> = append.ids_by_partition(0, from).concat();
            let mut want = append.ids(from..append.len());
            got.sort_unstable();
            want.sort_unstable();
            assert_eq!(got, want, "from {from}");
        }
        let per: usize = (0..PARTS).map(|p| append.part_len(p)).sum();
        assert_eq!(per, 40);
        let out = grow.output_batch(&grow.ids(0..grow.len())).unwrap();
        assert_eq!(out.num_rows(), 20);
        // A merge table takes another table's rows and merges duplicates.
        let mut merged = RowTable::new(&core, &seed, Mode::Grow, false).unwrap();
        // Different register counts do not merge.
        let mut one = Hll::new(PART_HLL_BITS);
        one.add(1);
        one.merge(&Hll::new(ADAPT_HLL_BITS));
        assert_eq!(one.registers.iter().filter(|&&r| r > 0).count(), 1);
        for p in 0..PARTS {
            let c = append.merge_chunk(p, 0..append.part_len(p)).unwrap();
            let want = merged.projected_merge(&c);
            merged.apply_merge(&append, &c).unwrap();
            assert!(merged.memory_size() <= want);
        }
        assert_eq!((merged.len(), merged.is_unique()), (20, true));
        let out = merged.output_batch(&merged.ids(0..20)).unwrap();
        let keys = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let counts = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        for (k, c) in keys.iter().zip(counts) {
            assert_eq!(c, Some(if k < Some(4) { 6 } else { 1 }), "key {k:?}");
        }
        // Only growing tables merge.
        let c = append.merge_chunk(0, 0..0).unwrap();
        let e = fixed.apply_merge(&append, &c).unwrap_err();
        assert!(e.to_string().contains("not growing"), "{e}");
    }

    /// After `ADAPT_ROWS` rows, unique keys switch a fixed index to
    /// append-only, and keys that repeat beyond its capacity grow it.
    #[test]
    fn fixed_index_adapts() {
        let core = core();
        let seed = HashSeed::new(Some(2));
        let mut unique = RowTable::new(&core, &seed, Mode::Fixed { limit: 1024 }, true).unwrap();
        let mut repeat = RowTable::new(&core, &seed, Mode::Fixed { limit: 1024 }, true).unwrap();
        let step = 8192;
        for start in (0..ADAPT_ROWS + step).step_by(step) {
            let s = i64::try_from(start).unwrap();
            apply(&mut unique, &batch(&core, s..s + 8192));
            apply(
                &mut repeat,
                &batch(&core, (s..s + 8192).map(|i| i % 10_000)),
            );
        }
        assert_eq!(unique.mode, Mode::Append);
        assert_eq!(repeat.mode, Mode::Fixed { limit: 4096 });
        // Deterministic and two-thread defaults.
        assert_eq!(Mode::local(true, 12), Mode::Grow);
        assert_eq!(Mode::local(false, 1), Mode::Grow);
        assert_eq!(
            Mode::local(false, 2),
            Mode::Fixed {
                limit: (1 << 18) * 2 / 3
            }
        );
        assert_eq!(
            Mode::local(false, 3),
            Mode::Fixed {
                limit: (1 << 15) * 2 / 3
            }
        );
    }

    /// The final phase merges small results in one task and splits large
    /// ones over every partition; empty partitions get no task.
    #[test]
    fn final_tasks_follow_the_estimate() {
        let core = core();
        let seed = HashSeed::new(Some(3));
        let mut small = RowTable::new(&core, &seed, Mode::Fixed { limit: 64 }, true).unwrap();
        apply(&mut small, &batch(&core, 0..1_000));
        let tasks = final_tasks(&[&small]);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].parts, 0..PARTS);
        assert!(
            tasks[0].groups >= 1_000 && tasks[0].groups <= 1_200,
            "{:?}",
            tasks[0]
        );
        let mut big = RowTable::new(&core, &seed, Mode::Append, true).unwrap();
        for s in (0..600_000).step_by(8192) {
            apply(&mut big, &batch(&core, s..(s + 8192).min(600_000)));
        }
        let tasks = final_tasks(&[&big, &small]);
        assert_eq!(tasks.len(), PARTS);
        assert!(tasks.iter().all(|t| t.parts.len() == 1));
        assert!(final_tasks(&[]).is_empty());
    }
}
