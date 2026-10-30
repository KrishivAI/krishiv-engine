// SPDX-License-Identifier: Apache-2.0
//! Row-layout aggregation tables (P2-T159, `krishiv.exec.agg.rowLayout`).
//!
//! A [`RowTable`] keeps its groups' keys as fixed-stride [`RowLayout`] rows,
//! one row per group, in up to [`PARTS`] radix partitions chosen by the top
//! bits of the row hash. Each partition owns its rows and its aggregates'
//! states (the same [`Accumulator`]s as the columnar path): a group's id in
//! its partition's accumulators is its row's position in the partition, so
//! a row is only its key,
//!
//! ```text
//! | h u64 | key validity | views 16 B ... | fixed keys ... |
//! ```
//!
//! A string or binary key is a 16-byte view: its length, then the bytes
//! inline (up to 12), or a 4-byte prefix and the offset of the bytes in the
//! partition's heap. Equal keys have equal bytes but for the heap offset,
//! so fixed-width keys compare as words and views compare their length and
//! prefix word first, then the inline word or the heap bytes.
//!
//! and a partition is self-contained: the final phase moves partitions
//! between tables and frees each one once it is merged (P2-T171 I2: state
//! is partition-local and mergeable).
//!
//! # Hashes (P2-T157 Appendix C)
//! - `h`, stored at the start of the row, is seedless: the row layout's
//!   column-wise hash, the same for every operator instance and worker
//!   (not the routing hash `H_kg`, which only exchanges and key groups
//!   use). Radix partitions are its top bits; the per-partition [`Hll`]
//!   sketches are over it, so they merge across tables.
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
//! # Updates
//! A batch's rows are looked up first; then each touched partition's
//! accumulators fold that partition's rows. When a batch touches several
//! partitions, the argument columns are gathered once in partition order
//! (a stable counting sort, so each group still sees its rows in input
//! order) and each partition folds a zero-copy slice. A table with one
//! partition never gathers.
//!
//! # Width
//! A thread-local table starts with **one partition** and widens to its
//! radix partitions ([`RowTable::widen`]: rows, long key bytes and states
//! are spread by `h`, in row order) once it holds [`WIDEN_GROUPS`] groups,
//! so the final phase merges partitions in parallel and a low-cardinality
//! aggregate never pays the per-partition fold. When it widens is a cost
//! choice between that fold (a gather of the argument columns per touched
//! partition, every batch) and the copy (every row of the table, once):
//! - while the input is folded, when its input keeps making new rows
//!   (nearly all rows so far, or a quarter of the rows since a settled
//!   fixed index last adapted): the table grows with its input, and a
//!   late copy costs several times the fold per row it copies;
//! - otherwise when its input ends (`combine`, on each driver in
//!   parallel): most rows find their group, so the fold would be paid on
//!   every row for a copy of few groups.
//!
//! Measured on TPC-DS SF10 at eight threads: q22 (a `ROLLUP` of 132.6M
//! expanded rows into 204K groups; a driver's table ends at 1.8M rows from
//! 66M input rows) is faster widening at the end; q39, q65, q67 and q75
//! (a quarter to all input rows new) are faster widening early.
//! A table that cannot reserve the copy stays narrow (the final phase asks
//! again). One thread never widens (nothing merges in parallel).
//!
//! # Memory
//! [`projected`](RowTable::projected) bounds the bytes after an apply
//! before anything changes (ADR 0007): partition rows by the batch's rows
//! per partition, the index by its mode, the accumulators as on the default
//! path, plus the gathered argument columns.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow_array::{Array, ArrayRef, RecordBatch, UInt32Array};
use hashbrown::HashTable;
use krishiv_types::batch::{Batch, Weights};

use super::accumulator::Accumulator;
use super::table::{AggCore, batch};
use super::{AggregateOutput, unsupported};
use crate::error::{Result, internal};
use crate::hash::{HashSeed, PARTITIONS, ProbeCount, ProbeTotals, grown, spill_partition};
use crate::rows::{EncodedKeys, Hll, RowLayout};

/// Radix bits of `h` of a partitioned table (DuckDB's initial 4 bits; the
/// final phase merges ranges of these partitions, see [`final_tasks`]).
pub(crate) const RADIX_BITS: u32 = 4;
/// Most radix partitions of a table.
pub(crate) const PARTS: usize = 1 << RADIX_BITS;
/// How a table's `u32` row references split into partition and row: the
/// top `bits` bits are the partition, the rest the row, so a table of one
/// partition holds up to 2^32 groups as before and a partitioned one up to
/// 2^(32 - bits) per partition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Refs {
    bits: u32,
}

impl Refs {
    fn new(bits: u32) -> Self {
        Self {
            bits: bits.min(RADIX_BITS),
        }
    }

    #[inline]
    fn row_bits(self) -> u32 {
        32 - self.bits
    }

    /// The partition of `r`.
    #[inline]
    pub(crate) fn part(self, r: u32) -> usize {
        usize::try_from(u64::from(r) >> self.row_bits()).unwrap_or(usize::MAX)
    }

    /// The row of `r` within its partition.
    #[inline]
    pub(crate) fn row(self, r: u32) -> u32 {
        r & self.mask()
    }

    /// The largest row of a partition.
    #[inline]
    fn mask(self) -> u32 {
        u32::try_from((1u64 << self.row_bits()) - 1).unwrap_or(u32::MAX)
    }

    /// The reference of row `row` of partition `p`.
    #[inline]
    fn make(self, p: usize, row: usize) -> u32 {
        let p = u64::try_from(p).unwrap_or(0) << self.row_bits();
        u32::try_from(p).unwrap_or(0) | u32::try_from(row).unwrap_or(self.mask())
    }
}
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

    /// Radix bits of the thread-local tables: none when one growing table
    /// is the whole result (one thread: nothing to merge in parallel).
    pub(crate) fn radix_bits(self, threads: usize) -> u32 {
        if self == Self::Grow && threads <= 1 {
            0
        } else {
            RADIX_BITS
        }
    }
}

/// `g = fin(h ^ seed)` (P2-T157 Appendix C), shared with the join.
#[inline]
fn fin(h: u64, seed: u64) -> u64 {
    crate::rows::finalize(h, seed)
}

/// The radix partition of `h` among `1 << bits`.
#[inline]
fn part_of(h: u64, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        usize::try_from(h >> (64 - bits)).unwrap_or(0)
    }
}

#[inline]
fn word(row: &[u8], at: usize) -> u64 {
    row.get(at..at + 8)
        .and_then(|s| <[u8; 8]>::try_from(s).ok())
        .map_or(0, u64::from_le_bytes)
}

/// Groups from which a narrow thread-local table widens to its radix
/// partitions (the final phase's groups per task; when, see the module's
/// width docs): below it the table design does not matter (P2-T159 step 0)
/// and one partition saves the per-partition fold.
pub(crate) const WIDEN_GROUPS: usize = 32_768;

/// Rows from which a thread-local table keeps per-partition sketches: a
/// smaller table's row count is its own estimate, and 16 sketches would
/// cost more than its rows under a tight memory limit.
const SKETCH_ROWS: usize = 4096;

/// Register bits of the per-partition sketches.
const PART_HLL_BITS: u32 = 10;
/// Register bits of a fixed index's input sample.
const ADAPT_HLL_BITS: u32 = 12;

/// The input sample a fixed index adapts from.
#[derive(Debug)]
struct Adapt {
    sketch: Hll,
    rows: usize,
    /// Rows pushed (groups created) so far.
    pushed: usize,
    next: usize,
    /// `rows` and `pushed` when the current interval started.
    window: (usize, usize),
    /// At the last adaptation the index was settled (it did not grow) and
    /// at least [`EARLY_RATE`] of the interval's rows were pushed: see
    /// [`RowTable::wants_widen_early`].
    early: bool,
}

/// Share of input rows that must make new rows, once a fixed index has
/// settled, for a table to widen while its input is still folded. The
/// late copy costs about 7x the early per-row fold per row it copies
/// (TPC-DS SF10 profiles: a widen copies about 5M rows a second per
/// driver; the per-partition fold adds about 30 ns per input row), so
/// early wins above about 1/7 of rows new; a quarter leaves a margin.
const EARLY_RATE: f64 = 0.25;
/// Share of input rows that made new rows from which a table widens as
/// soon as it is due: nearly every row is new (no adaptation needed).
const ALL_NEW: f64 = 0.9;

/// A batch prepared for [`RowTable::apply`].
pub(crate) struct RowPrepared {
    keys: EncodedKeys,
    /// Rows per radix partition (bounds each partition's growth).
    counts: [usize; PARTS],
    /// Long key bytes per radix partition (bounds each heap's growth;
    /// empty without string or binary keys).
    heap_counts: Vec<usize>,
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
    /// Their long key bytes.
    heap: usize,
}

/// One radix partition: its rows and its groups' states (group `i` is row
/// `i`). A partition moved out of a table ([`RowTable::split_parts`]) or
/// freed ([`RowTable::release_part`]) is empty, with no accumulators.
#[derive(Debug, Default)]
struct Part {
    rows: Vec<u8>,
    /// The bytes of the rows' long string and binary keys (views longer
    /// than inline point here). Append-only: a group's key never changes.
    heap: Vec<u8>,
    accs: Vec<Accumulator>,
}

impl Part {
    fn bytes(&self) -> usize {
        self.rows.capacity()
            + self.heap.capacity()
            + self.accs.iter().map(Accumulator::size).sum::<usize>()
    }
}

/// Which columns a fold reads, and how.
#[derive(Clone, Copy)]
enum Fold {
    /// Raw input: each aggregate's arguments.
    Raw,
    /// Partial states: aggregate `i`'s columns are its state range minus
    /// `offset` (the key columns before them, if any).
    States { offset: usize },
}

/// Groups as rows in radix partitions, with their accumulators.
#[derive(Debug)]
pub(crate) struct RowTable {
    core: Arc<AggCore>,
    /// The seed of `g`.
    gseed: u64,
    stride: usize,
    /// Radix bits: `1 << bits` partitions.
    bits: u32,
    /// How row references split.
    refs: Refs,
    /// The key layout when it has string or binary views (their compare
    /// and copy need it); `None` for fixed-width keys.
    views: Option<RowLayout>,
    parts: Vec<Part>,
    /// Per-partition sketches of the rows' `h` (thread-local tables, from
    /// [`SKETCH_ROWS`] rows).
    sketches: Option<Vec<Hll>>,
    /// Whether the table keeps sketches once large enough (a thread-local
    /// table with several partitions).
    sketching: bool,
    /// A thread-local table (sketches, adaptation, widening).
    local: bool,
    /// Radix bits a narrow table widens to at `widen_at` groups (none when
    /// equal to `bits`).
    widen_to: u32,
    widen_at: usize,
    index: HashTable<u32>,
    mode: Mode,
    adapt: Option<Adapt>,
    groups: usize,
    /// Input rows applied (see [`wants_widen_early`](Self::wants_widen_early)).
    rows_in: usize,
    /// No key has two rows: the index was never cleared or dropped with
    /// rows in the table, and no row was appended without a lookup.
    unique: bool,
    /// Counts clears of a full fixed index (`EXPLAIN ANALYZE`).
    abandons: Option<Arc<AtomicU64>>,
    /// Counts reallocations of the index (`EXPLAIN ANALYZE`).
    resizes: Option<Arc<AtomicU64>>,
    /// Where probe lengths go, for a detailed profile only.
    probes: Option<Arc<ProbeTotals>>,
}

impl RowTable {
    /// A table in `mode` with `1 << bits` radix partitions (`bits` at most
    /// [`RADIX_BITS`]); `local` tables keep per-partition sketches for the
    /// final phase, and adapt a fixed index.
    pub(crate) fn new(
        core: &Arc<AggCore>,
        seed: &HashSeed,
        mode: Mode,
        local: bool,
        bits: u32,
    ) -> Result<Self> {
        let layout = layout(core)?;
        let bits = bits.min(RADIX_BITS);
        let nparts = 1 << bits;
        let parts = (0..nparts)
            .map(|_| {
                Ok(Part {
                    rows: Vec::new(),
                    heap: Vec::new(),
                    accs: new_accs(core)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            core: Arc::clone(core),
            gseed: seed.hash(b"krishiv.agg.g"),
            stride: layout.stride(),
            bits,
            refs: Refs::new(bits),
            views: layout.has_views().then(|| layout.clone()),
            parts,
            sketches: None,
            sketching: local && nparts > 1,
            local,
            widen_to: bits,
            widen_at: WIDEN_GROUPS,
            index: HashTable::new(),
            mode,
            adapt: (local && matches!(mode, Mode::Fixed { .. })).then(|| Adapt {
                sketch: Hll::new(ADAPT_HLL_BITS),
                rows: 0,
                pushed: 0,
                next: ADAPT_ROWS,
                window: (0, 0),
                early: false,
            }),
            groups: 0,
            rows_in: 0,
            unique: true,
            abandons: None,
            resizes: None,
            probes: None,
        })
    }

    /// Number of rows (groups; a group may have several rows in a table
    /// whose fixed index was cleared).
    pub(crate) fn len(&self) -> usize {
        self.groups
    }

    /// Counts the clears of a full fixed index into `counter`.
    pub(crate) fn with_abandons(mut self, counter: Arc<AtomicU64>) -> Self {
        self.abandons = Some(counter);
        self
    }

    /// Counts the reallocations of the index into `counter`.
    pub(crate) fn with_resizes(mut self, counter: Arc<AtomicU64>) -> Self {
        self.resizes = Some(counter);
        self
    }

    /// Counts probe lengths into `totals` (`ExecConfig::detailed_metrics`;
    /// `None`: not counted).
    pub(crate) fn with_probes(mut self, totals: Option<Arc<ProbeTotals>>) -> Self {
        self.probes = totals;
        self
    }

    /// Widens the table to `bits` radix partitions once it holds `at`
    /// groups (see [`widen`](Self::widen)).
    pub(crate) fn widening(mut self, bits: u32, at: usize) -> Self {
        self.widen_to = bits.min(RADIX_BITS).max(self.bits);
        self.widen_at = at;
        self
    }

    /// Whether the table is due to widen ([`widen`](Self::widen)): the
    /// caller widens it when it can reserve
    /// [`widen_bytes`](Self::widen_bytes); a table that cannot stays narrow
    /// (only a cost choice) and the final phase asks again.
    pub(crate) fn wants_widen(&self) -> bool {
        self.widen_to > self.bits && self.groups >= self.widen_at
    }

    /// Whether the table is due to widen while its input is still being
    /// folded ([`wants_widen`](Self::wants_widen), and its input keeps
    /// making new rows): nearly every input row so far made a new row
    /// ([`ALL_NEW`]), or, at the last adaptation of a settled fixed index,
    /// at least [`EARLY_RATE`] of the interval's rows did. While a fixed
    /// index still grows, its rows overstate the keys (a cleared index
    /// makes new rows of keys it has seen), so it does not decide then.
    /// See the module's width docs.
    pub(crate) fn wants_widen_early(&self) -> bool {
        #[expect(clippy::cast_precision_loss, reason = "row counts in a ratio")]
        let all_new = self.groups as f64 >= ALL_NEW * self.rows_in as f64;
        self.wants_widen() && (all_new || self.adapt.as_ref().is_some_and(|a| a.early))
    }

    /// Spreads the rows (with their long key bytes) and states over
    /// `1 << widen_to` radix partitions of `h`, in row order, so each
    /// group's states are unchanged. A growing index is rebuilt over the
    /// new rows; a fixed one is cleared (as when full: later rows of its
    /// keys become partial duplicates); append mode has none. Partition
    /// sketches start from the rows.
    ///
    /// The copy is built beside the table, which changes only once it is
    /// complete: on an error the table keeps its rows, states and width.
    pub(crate) fn widen(&mut self) -> Result<()> {
        if self.widen_to <= self.bits {
            return Ok(());
        }
        let (bits, refs) = (self.widen_to, Refs::new(self.widen_to));
        let stride = self.stride;
        let nparts = 1usize << bits;
        let mut parts = (0..nparts)
            .map(|_| {
                Ok(Part {
                    rows: Vec::new(),
                    heap: Vec::new(),
                    accs: new_accs(&self.core)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for (copied, part) in self.parts.iter().enumerate() {
            // Rows per new partition first, so each grows once and exactly
            // (no doubling slack: widening runs under a memory limit too).
            let mut ids: Vec<Vec<u32>> = vec![Vec::new(); nparts];
            let mut heap = vec![0; nparts];
            for (i, row) in (0u32..).zip(part.rows.chunks_exact(stride)) {
                let p = part_of(word(row, 0), bits);
                ids[p].push(i);
                if let Some(l) = &self.views {
                    heap[p] += l.long_bytes(row);
                }
            }
            for ((dst, ids), &bytes) in parts.iter_mut().zip(&ids).zip(&heap) {
                if ids.is_empty() {
                    continue;
                }
                dst.rows.reserve_exact(ids.len() * stride);
                dst.heap.reserve_exact(bytes);
                for &i in ids {
                    let at = i as usize * stride;
                    let row = part
                        .rows
                        .get(at..at + stride)
                        .ok_or_else(|| internal("a row outside its partition".to_owned()))?;
                    match &self.views {
                        Some(l) => l.push_row(row, &part.heap, &mut dst.rows, &mut dst.heap)?,
                        None => dst.rows.extend_from_slice(row),
                    }
                }
                let start = dst.rows.len() / stride - ids.len();
                let to: Vec<u32> = (start..start + ids.len())
                    .map(|r| u32::try_from(r).unwrap_or(u32::MAX))
                    .collect();
                for (acc, from) in dst.accs.iter_mut().zip(&part.accs) {
                    acc.resize(start + ids.len());
                    acc.merge(&to, &from.state(ids)?)?;
                }
            }
            widen_fault(copied)?;
        }
        self.parts = parts;
        self.bits = bits;
        self.refs = refs;
        self.sketching = self.local;
        self.index = HashTable::new();
        match self.mode {
            Mode::Grow => {
                let (gseed, all) = (self.gseed, self.ids(0..self.groups));
                // The rebuilt index is a reallocation too.
                self.index.reserve(all.len(), |_| 0);
                self.count_resize(0);
                for r in all {
                    let g = fin(word(row_at(&self.parts, stride, refs, r), 0), gseed);
                    let parts = &self.parts;
                    self.index.insert_unique(g, r, |&o| {
                        fin(word(row_at(parts, stride, refs, o), 0), gseed)
                    });
                }
            }
            Mode::Fixed { .. } => self.unique &= self.groups == 0,
            Mode::Append => {}
        }
        self.start_sketches(0);
        Ok(())
    }

    /// Bytes [`widen`](Self::widen) needs beyond the table's while it
    /// runs: the copy of its rows and states (at most the table's bytes),
    /// the state arrays of one new partition at a time (at most the
    /// states' bytes, twice for validity and offsets), the row lists and
    /// references (two `u32` per group, plus per-partition list headers),
    /// and the sketches it starts.
    pub(crate) fn widen_bytes(&self) -> usize {
        let nparts = 1usize << self.widen_to;
        let accs: usize = self
            .parts
            .iter()
            .flat_map(|p| &p.accs)
            .map(Accumulator::size)
            .sum();
        let lists = 2 * self.groups * size_of::<u32>()
            + nparts * (2 * size_of::<Vec<u32>>() + size_of::<usize>());
        self.memory_size() + 2 * accs + lists + nparts * Hll::new(PART_HLL_BITS).bytes()
    }

    /// Buckets of the index (its capacity).
    pub(crate) fn buckets(&self) -> usize {
        buckets(&self.index)
    }

    /// Whether every group has one row (no partial duplicates).
    pub(crate) fn is_unique(&self) -> bool {
        self.unique
    }

    /// Radix partitions of this table.
    pub(crate) fn part_count(&self) -> usize {
        self.parts.len()
    }

    /// Rows in partition `p`.
    pub(crate) fn part_len(&self, p: usize) -> usize {
        self.parts.get(p).map_or(0, |v| v.rows.len() / self.stride)
    }

    /// The per-partition sketches (thread-local tables).
    pub(crate) fn sketches(&self) -> Option<&[Hll]> {
        self.sketches.as_deref()
    }

    pub(crate) fn memory_size(&self) -> usize {
        self.parts.iter().map(Part::bytes).sum::<usize>()
            + buckets(&self.index) * BUCKET_BYTES
            + self.sketch_bytes()
    }

    /// The scratch of folding `n` rows: each row's reference, the gather
    /// order and the rows' positions in their partitions.
    fn scratch(n: usize) -> usize {
        3 * n * size_of::<u32>()
    }

    fn sketch_bytes(&self) -> usize {
        self.sketches
            .iter()
            .flatten()
            .map(Hll::bytes)
            .sum::<usize>()
            + self.adapt.as_ref().map_or(0, |a| a.sketch.bytes())
    }

    /// Whether an apply of `n` more rows starts the sketches.
    fn starts_sketches(&self, n: usize) -> bool {
        self.sketching && self.sketches.is_none() && self.groups + n >= SKETCH_ROWS
    }

    /// [`sketch_bytes`](Self::sketch_bytes) after an apply of `n` rows.
    fn projected_sketch_bytes(&self, n: usize) -> usize {
        let new = if self.starts_sketches(n) {
            self.parts.len() * Hll::new(PART_HLL_BITS).bytes()
        } else {
            0
        };
        self.sketch_bytes() + new
    }

    /// Starts the per-partition sketches from the rows so far, if an apply
    /// of `n` rows reaches [`SKETCH_ROWS`].
    fn start_sketches(&mut self, n: usize) {
        if !self.starts_sketches(n) {
            return;
        }
        let stride = self.stride;
        let sketches = self
            .parts
            .iter()
            .map(|part| {
                let mut h = Hll::new(PART_HLL_BITS);
                part.rows
                    .chunks_exact(stride)
                    .for_each(|row| h.add(word(row, 0)));
                h
            })
            .collect();
        self.sketches = Some(sketches);
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
        // `h` is seedless (Appendix C).
        let keys = layout(&self.core)?.encode(&keys)?;
        let mut counts = [0; PARTS];
        let mut heap_counts = Vec::new();
        // One partition (a narrow table) needs no per-row pass: all rows,
        // and all the batch heap's bytes (only long values go there), are
        // its; a batch without long values adds no heap bytes anywhere.
        if self.bits == 0 {
            counts[0] = keys.len();
        } else {
            for &h in keys.hashes() {
                counts[part_of(h, self.bits)] += 1;
            }
        }
        if let Some(l) = &self.views {
            heap_counts = vec![0; PARTS];
            if self.bits == 0 {
                heap_counts[0] = keys.heap().len();
            } else if !keys.heap().is_empty() {
                for (i, &h) in keys.hashes().iter().enumerate() {
                    heap_counts[part_of(h, self.bits)] += l.long_bytes(keys.row(i));
                }
            }
        }
        Ok(RowPrepared {
            keys,
            counts,
            heap_counts,
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
        } else if pushed >= 2.0 * distinct && limit * 4 * 3 / 2 <= MAX_SINK_SLOTS {
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
        let (parts, stride, gseed, refs) = (&self.parts, self.stride, self.gseed, self.refs);
        let before = self.index.capacity();
        self.index
            .reserve(len.saturating_sub(self.index.len()), |&r| {
                fin(word(row_at(parts, stride, refs, r), 0), gseed)
            });
        self.count_resize(before);
    }

    /// Counts a reallocation of the index (`hash_resizes`) when its
    /// capacity changed from `before`: every path that reserves calls it.
    fn count_resize(&self, before: usize) {
        if self.index.capacity() != before
            && let Some(c) = &self.resizes
        {
            c.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The accumulators' bytes after `new[p]` more groups in each partition
    /// `p` and a fold of `cols` (`fold` says which columns each aggregate
    /// reads; a string `min`/`max` may keep all their bytes, once).
    fn projected_accs(&self, new: &[usize], cols: &[ArrayRef], fold: Fold) -> usize {
        let parts: usize = self
            .parts
            .iter()
            .zip(new.iter().copied().chain(std::iter::repeat(0)))
            .map(|(part, n)| {
                part.accs
                    .iter()
                    .map(|a| a.projected(n, None, None))
                    .sum::<usize>()
            })
            .sum();
        let kept: usize = self.parts.first().map_or(0, |p| {
            p.accs
                .iter()
                .zip(&self.core.ranges)
                .map(|(a, r)| {
                    // The value column, then (`max_by`/`min_by`) the key.
                    let at = |n: usize| {
                        let i = match fold {
                            Fold::States { offset } => r.start.checked_sub(offset).map(|i| i + n),
                            Fold::Raw => a.args().get(n).copied(),
                        };
                        i.and_then(|i| cols.get(i))
                    };
                    a.projected(0, at(0), at(1))
                        .saturating_sub(a.projected(0, None, None))
                })
                .sum()
        });
        parts + kept
    }

    /// Bytes of the gathered argument columns of a fold touching several
    /// partitions, `counts` rows each: one partition's at a time, so the
    /// largest partition's share of the fixed-width columns, and the full
    /// columns without a primitive width (their values are not spread
    /// evenly: a partition with few rows can gather most of their bytes).
    fn gather_bytes(&self, cols: &[ArrayRef], fold: Fold, w: &Weights, counts: &[usize]) -> usize {
        let used = self.used(cols.len(), fold);
        let (mut fixed, mut varied) = (0usize, 0usize);
        for (c, u) in cols.iter().zip(&used) {
            if !u {
                continue;
            }
            if c.data_type().primitive_width().is_some() {
                fixed += c.get_array_memory_size();
            } else {
                varied += c.get_array_memory_size();
            }
        }
        // Weights are `Int64`: one value per row, so a proportional share.
        fixed += match w {
            Weights::Column(c) => c.get_array_memory_size(),
            Weights::Const(_) => 0,
        };
        let rows: usize = counts.iter().sum();
        let most = counts.iter().copied().max().unwrap_or(0);
        varied + fixed.saturating_mul(most).div_ceil(rows.max(1))
    }

    /// [`memory_size`](Self::memory_size) after [`apply`](Self::apply)ing
    /// `p` (an upper bound).
    pub(crate) fn projected(&self, p: &RowPrepared) -> usize {
        let n = p.keys.len();
        let rows: usize = self
            .parts
            .iter()
            .zip(
                p.counts
                    .iter()
                    .zip(p.heap_counts.iter().chain(std::iter::repeat(&0))),
            )
            .map(|(v, (c, &b))| grown(&v.rows, c * self.stride) + grown(&v.heap, b))
            .sum();
        rows + self.index_bytes(self.next_mode(), n)
            + self.projected_accs(&p.counts, &p.cols, Self::fold_of(p))
            + self.projected_sketch_bytes(n)
    }

    /// Bytes [`apply`](Self::apply)ing `p` needs while it runs and frees
    /// before it returns (an upper bound): the scratch of the fold and the
    /// gathered columns of one partition.
    pub(crate) fn transient(&self, p: &RowPrepared) -> usize {
        let gather = if p.counts.iter().filter(|&&c| c > 0).count() > 1 {
            self.gather_bytes(&p.cols, Self::fold_of(p), &p.weights, &p.counts)
        } else {
            0
        };
        Self::scratch(p.keys.len()) + gather + p.keys.bytes()
    }

    fn fold_of(p: &RowPrepared) -> Fold {
        if p.states {
            Fold::States { offset: 0 }
        } else {
            Fold::Raw
        }
    }

    /// Switches the index to `mode` (append-only tables drop it; a fixed
    /// index keeps its keys under a new limit).
    fn switch(&mut self, mode: Mode) {
        if mode == Mode::Append && self.mode != Mode::Append {
            self.unique &= self.groups == 0;
            self.index = HashTable::new();
        }
        let grew = mode != self.mode && mode != Mode::Append;
        self.mode = mode;
        if let Some(a) = self.adapt.as_mut()
            && a.rows >= a.next
        {
            #[expect(clippy::cast_precision_loss, reason = "row counts in a ratio")]
            let rate = (a.pushed - a.window.1) as f64 / (a.rows - a.window.0).max(1) as f64;
            a.early = !grew && rate >= EARLY_RATE;
            a.window = (a.rows, a.pushed);
            a.next = a.rows + ADAPT_ROWS;
        }
    }

    /// Folds a prepared batch in. Call after reserving
    /// [`projected`](Self::projected) bytes.
    pub(crate) fn apply(&mut self, p: &RowPrepared) -> Result<()> {
        let n = p.keys.len();
        self.rows_in = self.rows_in.saturating_add(n);
        self.switch(self.next_mode());
        self.start_sketches(n);
        if let Some(a) = self.adapt.as_mut() {
            a.rows += n;
            for &h in p.keys.hashes().iter().filter(|&&h| h & 7 == 0) {
                a.sketch.add(h);
            }
        }
        for (v, c) in self.parts.iter_mut().zip(p.counts) {
            v.rows.reserve(c * self.stride);
        }
        // Heaps grow once per batch, as `projected` assumed (not by
        // doubling per value).
        for (v, &b) in self.parts.iter_mut().zip(&p.heap_counts) {
            v.heap.reserve(b);
        }
        let mut refs = Vec::with_capacity(n);
        let before = self.groups;
        self.reserve_index(n);
        let result = self.lookup(&p.keys, &mut refs);
        if let Some(a) = self.adapt.as_mut() {
            a.pushed += self.groups - before;
        }
        result.and_then(|()| self.fold(&refs, &p.counts, &p.cols, &p.weights, Self::fold_of(p)))
    }

    /// The columns a fold reads.
    fn used(&self, n: usize, fold: Fold) -> Vec<bool> {
        let mut used = vec![false; n];
        let mut mark = |i: usize| {
            if let Some(u) = used.get_mut(i) {
                *u = true;
            }
        };
        match fold {
            Fold::Raw => self
                .core
                .spec
                .aggregates
                .iter()
                .flat_map(|a| &a.args)
                .for_each(|&i| mark(i)),
            Fold::States { offset } => self
                .core
                .ranges
                .iter()
                .flat_map(Clone::clone)
                .filter_map(|i| i.checked_sub(offset))
                .for_each(&mut mark),
        }
        used
    }

    /// Folds rows `refs` (one per row of `cols`; `counts` of them per
    /// partition) into their partitions' accumulators, after their
    /// partitions grew to their rows.
    fn fold(
        &mut self,
        refs: &[u32],
        counts: &[usize],
        cols: &[ArrayRef],
        w: &Weights,
        fold: Fold,
    ) -> Result<()> {
        let stride = self.stride;
        for part in &mut self.parts {
            let len = part.rows.len() / stride;
            part.accs.iter_mut().for_each(|a| a.resize(len));
        }
        let touched: Vec<usize> = (0..self.parts.len())
            .filter(|&p| counts.get(p).is_some_and(|&c| c > 0))
            .collect();
        let mut local = Vec::with_capacity(refs.len());
        if let [p] = touched[..] {
            local.extend(refs.iter().map(|&r| self.refs.row(r)));
            self.fold_part(p, &local, cols, w, fold)
        } else if touched.is_empty() {
            Ok(())
        } else {
            self.fold_gathered(refs, counts, cols, w, fold, &touched, &mut local)
        }
    }

    /// [`fold`](Self::fold) over several partitions: the rows are gathered
    /// in partition order (stable), and each partition folds its slice.
    #[expect(clippy::too_many_arguments, reason = "one fold's inputs")]
    fn fold_gathered(
        &mut self,
        refs: &[u32],
        counts: &[usize],
        cols: &[ArrayRef],
        w: &Weights,
        fold: Fold,
        touched: &[usize],
        local: &mut Vec<u32>,
    ) -> Result<()> {
        let n = refs.len();
        let mut starts = vec![0; self.parts.len()];
        let mut at = 0;
        for (s, &c) in starts.iter_mut().zip(counts) {
            *s = at;
            at += c;
        }
        if at != n {
            return Err(internal(
                "partition counts do not match the rows".to_owned(),
            ));
        }
        let mut order = vec![0; n];
        local.resize(n, 0);
        let mut next = starts.clone();
        for (i, &r) in (0u32..).zip(refs) {
            let p = self.refs.part(r);
            let slot = next
                .get_mut(p)
                .ok_or_else(|| internal("bad row reference".to_owned()))?;
            order[*slot] = i;
            local[*slot] = self.refs.row(r);
            *slot += 1;
        }
        let idx = UInt32Array::from(order);
        let used = self.used(cols.len(), fold);
        // One partition's rows at a time: the gathered copy is at most the
        // largest partition's share of the batch.
        touched.iter().try_for_each(|&p| {
            let (s, c) = (starts[p], counts[p]);
            let idx = idx.slice(s, c);
            let take = |c: &dyn Array| {
                arrow_select::take::take(c, &idx, None).map_err(|e| internal(e.to_string()))
            };
            let gathered: Vec<ArrayRef> = cols
                .iter()
                .zip(&used)
                .map(|(col, &u)| {
                    if u {
                        take(col.as_ref())
                    } else {
                        Ok(col.slice(s, c))
                    }
                })
                .collect::<Result<_>>()?;
            let w = match w {
                Weights::Const(k) => Weights::Const(*k),
                Weights::Column(a) => Weights::Column(
                    take(a)?
                        .as_any()
                        .downcast_ref::<arrow_array::Int64Array>()
                        .cloned()
                        .ok_or_else(|| internal("weights are not Int64".to_owned()))?,
                ),
            };
            self.fold_part(p, &local[s..s + c], &gathered, &w, fold)
        })
    }

    /// Folds rows of `cols` into groups `ids` of partition `p`.
    fn fold_part(
        &mut self,
        p: usize,
        ids: &[u32],
        cols: &[ArrayRef],
        w: &Weights,
        fold: Fold,
    ) -> Result<()> {
        let core = Arc::clone(&self.core);
        let part = self
            .parts
            .get_mut(p)
            .ok_or_else(|| internal(format!("radix partition {p} out of range")))?;
        match fold {
            Fold::Raw => part
                .accs
                .iter_mut()
                .try_for_each(|a| a.update(ids, cols, w)),
            Fold::States { offset } => {
                part.accs
                    .iter_mut()
                    .zip(&core.ranges)
                    .try_for_each(|(a, r)| {
                        let cols = cols
                            .get(r.start - offset..r.end - offset)
                            .ok_or_else(|| internal("state columns missing".to_owned()))?;
                        a.merge(ids, cols)
                    })
            }
        }
    }

    /// The row of each row of `keys`, inserting new groups.
    fn lookup(&mut self, keys: &EncodedKeys, refs: &mut Vec<u32>) -> Result<()> {
        let (hashes, heap) = (keys.hashes(), keys.heap());
        match self.mode {
            Mode::Append => {
                self.unique &= keys.is_empty();
                for (k, &h) in hashes.iter().enumerate() {
                    refs.push(self.push(keys.row(k), heap, h)?);
                }
            }
            Mode::Fixed { limit } => self.probe_all(keys, refs, limit)?,
            Mode::Grow => self.probe_all(keys, refs, usize::MAX)?,
        }
        Ok(())
    }

    /// [`find_or_insert`](Self::find_or_insert) for every row of `keys`, in
    /// one loop per key shape: the probe's compare of fixed-width keys has
    /// no view branch, so it stays small enough to inline.
    fn probe_all(&mut self, keys: &EncodedKeys, refs: &mut Vec<u32>, limit: usize) -> Result<()> {
        let mut count = ProbeCount::default();
        match (self.views.is_some(), self.probes.is_some()) {
            (true, false) => self.probe_rows::<true, false>(keys, refs, limit, &mut count)?,
            (false, false) => self.probe_rows::<false, false>(keys, refs, limit, &mut count)?,
            (true, true) => self.probe_rows::<true, true>(keys, refs, limit, &mut count)?,
            (false, true) => self.probe_rows::<false, true>(keys, refs, limit, &mut count)?,
        }
        if let Some(totals) = &self.probes
            && count.lookups > 0
        {
            totals.add(&count);
        }
        Ok(())
    }

    /// The loop of [`probe_all`](Self::probe_all) for one key shape;
    /// `COUNT` adds each lookup's key comparisons to `count`.
    fn probe_rows<const VIEWS: bool, const COUNT: bool>(
        &mut self,
        keys: &EncodedKeys,
        refs: &mut Vec<u32>,
        limit: usize,
        count: &mut ProbeCount,
    ) -> Result<()> {
        let (hashes, heap) = (keys.hashes(), keys.heap());
        for (k, &h) in hashes.iter().enumerate() {
            refs.push(self.find_or_insert::<VIEWS, COUNT>(keys.row(k), heap, h, limit, count)?);
        }
        Ok(())
    }

    /// The row of key row `key` (with hash `h`; its long values in
    /// `key_heap`), inserting it if new; an index holding `limit` keys is
    /// cleared before it takes another (the rows stay: a key seen again
    /// gets a new row). `VIEWS`: the layout has string or binary views;
    /// `COUNT`: the lookup's key comparisons are added to `count`.
    #[inline(always)]
    fn find_or_insert<const VIEWS: bool, const COUNT: bool>(
        &mut self,
        key: &[u8],
        key_heap: &[u8],
        h: u64,
        limit: usize,
        count: &mut ProbeCount,
    ) -> Result<u32> {
        let Self {
            parts,
            sketches,
            groups,
            index,
            unique,
            abandons,
            stride,
            gseed,
            bits,
            refs,
            views,
            ..
        } = self;
        let (stride, gseed) = (*stride, *gseed);
        let g = fin(h, gseed);
        // A lookup first (hashbrown inlines `find`; most rows are hits),
        // then an insert on a miss.
        let mut compares = 0_u64;
        let found = index
            .find(g, |&r| {
                if COUNT {
                    compares += 1;
                }
                match views {
                    Some(l) if VIEWS => {
                        let (row, heap) = row_heap_at(parts, stride, *refs, r);
                        !row.is_empty() && l.eq(row, heap, key, key_heap)
                    }
                    _ => same_key(row_at(parts, stride, *refs, r), key, stride),
                }
            })
            .copied();
        if COUNT {
            count.add(compares);
        }
        if let Some(r) = found {
            return Ok(r);
        }
        let shape = (stride, *bits, *refs, views.as_ref());
        let r = push_row(parts, sketches, groups, shape, (key, key_heap), h)?;
        if index.len() >= limit {
            index.clear();
            *unique = false;
            if let Some(c) = abandons {
                c.fetch_add(1, Ordering::Relaxed);
            }
        }
        let refs = *refs;
        index.insert_unique(g, r, |&o| {
            fin(word(row_at(parts, stride, refs, o), 0), gseed)
        });
        Ok(r)
    }

    fn push(&mut self, key: &[u8], key_heap: &[u8], h: u64) -> Result<u32> {
        push_row(
            &mut self.parts,
            &mut self.sketches,
            &mut self.groups,
            (self.stride, self.bits, self.refs, self.views.as_ref()),
            (key, key_heap),
            h,
        )
    }

    /// The row references at positions `range` of the partition order
    /// (partition 0's rows, then partition 1's, ...).
    pub(crate) fn ids(&self, range: Range<usize>) -> Vec<u32> {
        let mut out = Vec::with_capacity(range.len());
        let mut start = 0;
        for p in 0..self.parts.len() {
            let len = self.part_len(p);
            let (from, to) = (range.start.max(start), range.end.min(start + len));
            out.extend((from..to).map(|i| self.refs.make(p, i - start)));
            start += len;
        }
        out
    }

    /// The references of rows `rows` of partition `p`.
    pub(crate) fn part_ids(&self, p: usize, rows: Range<usize>) -> Vec<u32> {
        rows.map(|i| self.refs.make(p, i)).collect()
    }

    /// Row references per spill partition at `level` (by `g`) of the rows
    /// at positions `from..` of the partition order (see [`ids`](Self::ids)).
    pub(crate) fn ids_by_partition(&self, level: u32, from: usize) -> Vec<Vec<u32>> {
        let mut out = vec![Vec::new(); PARTITIONS];
        let mut start = 0;
        for (p, part) in self.parts.iter().enumerate() {
            let skip = from.saturating_sub(start);
            start += self.part_len(p);
            for (i, row) in part.rows.chunks_exact(self.stride).enumerate().skip(skip) {
                let g = fin(word(row, 0), self.gseed);
                out[spill_partition(g, level)].push(self.refs.make(p, i));
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
        let keys = layout(&self.core)?.encode(keys)?;
        let mut parts = vec![Vec::new(); PARTITIONS];
        for (i, &h) in (0u32..).zip(keys.hashes()) {
            parts[spill_partition(fin(h, self.gseed), level)].push(i);
        }
        Ok(parts)
    }

    fn rows_of(&self, refs: &[u32]) -> Result<Vec<&[u8]>> {
        refs.iter()
            .map(|&r| {
                let row = row_at(&self.parts, self.stride, self.refs, r);
                if row.is_empty() {
                    Err(internal(format!("row reference {r:#x} out of range")))
                } else {
                    Ok(row)
                }
            })
            .collect()
    }

    fn key_columns(&self, refs: &[u32]) -> Result<Vec<ArrayRef>> {
        let rows = self.rows_of(refs)?;
        match &self.views {
            None => layout(&self.core)?.decode(&rows),
            Some(l) => {
                let with_heaps: Vec<(&[u8], &[u8])> = refs
                    .iter()
                    .zip(rows)
                    .map(|(&r, row)| {
                        let heap = self
                            .parts
                            .get(self.refs.part(r))
                            .map_or(&[][..], |p| p.heap.as_slice());
                        (row, heap)
                    })
                    .collect();
                l.decode_rows(&with_heaps)
            }
        }
    }

    /// `f` of each partition's accumulators and its groups among `refs`
    /// (taken in runs of one partition), concatenated column by column in
    /// the order of `refs`.
    fn per_part(
        &self,
        refs: &[u32],
        f: impl Fn(&[Accumulator], &[u32]) -> Result<Vec<ArrayRef>>,
    ) -> Result<Vec<ArrayRef>> {
        let mut pieces: Vec<Vec<ArrayRef>> = Vec::new();
        let mut local = Vec::new();
        let mut rest = refs;
        while let Some(&first) = rest.first() {
            let p = self.refs.part(first);
            let run = rest
                .iter()
                .position(|&r| self.refs.part(r) != p)
                .unwrap_or(rest.len());
            let part = self
                .parts
                .get(p)
                .ok_or_else(|| internal(format!("row reference {first:#x} out of range")))?;
            local.clear();
            local.extend(rest[..run].iter().map(|&r| self.refs.row(r)));
            if local
                .iter()
                .any(|&i| i as usize >= part.rows.len() / self.stride)
            {
                return Err(internal(format!("row reference {first:#x} out of range")));
            }
            pieces.push(f(&part.accs, &local)?);
            rest = &rest[run..];
        }
        match pieces.len() {
            // No rows: the right types from any accumulators.
            0 => {
                let accs = new_accs(&self.core)?;
                f(&accs, &[])
            }
            1 => Ok(pieces.swap_remove(0)),
            _ => (0..pieces[0].len())
                .map(|c| {
                    let arrays: Vec<&dyn Array> = pieces.iter().map(|p| p[c].as_ref()).collect();
                    arrow_select::concat::concat(&arrays).map_err(|e| internal(e.to_string()))
                })
                .collect(),
        }
    }

    /// The partial states of rows `refs`, in the state layout.
    pub(crate) fn state_batch(&self, refs: &[u32]) -> Result<RecordBatch> {
        let mut cols = self.key_columns(refs)?;
        cols.extend(self.per_part(refs, |accs, ids| {
            let mut out = Vec::new();
            for a in accs {
                out.extend(a.state(ids)?);
            }
            Ok(out)
        })?);
        batch(&self.core.state_schema, cols, refs.len())
    }

    /// The node's output for rows `refs`: final values or partial states.
    pub(crate) fn output_batch(&self, refs: &[u32]) -> Result<RecordBatch> {
        if self.core.spec.output == AggregateOutput::Partial {
            return self.state_batch(refs);
        }
        let mut cols = self.key_columns(refs)?;
        cols.extend(self.per_part(refs, |accs, ids| {
            accs.iter().map(|a| a.evaluate(ids)).collect()
        })?);
        batch(&self.core.output_schema, cols, refs.len())
    }

    /// Rows `rows` of partition `p` of this table, with their states, to
    /// merge into another table of the same aggregate.
    pub(crate) fn merge_chunk(&self, p: usize, rows: Range<usize>) -> Result<MergeChunk> {
        let refs = self.part_ids(p, rows.clone());
        let part = self
            .parts
            .get(p)
            .ok_or_else(|| internal(format!("radix partition {p} out of range")))?;
        if rows.end > self.part_len(p) {
            return Err(internal(format!(
                "rows {rows:?} of partition {p} out of range"
            )));
        }
        let ids: Vec<u32> = refs.iter().map(|&r| self.refs.row(r)).collect();
        let mut states = Vec::new();
        for a in &part.accs {
            states.extend(a.state(&ids)?);
        }
        let heap = match &self.views {
            Some(l) => refs
                .iter()
                .map(|&r| l.long_bytes(row_at(&self.parts, self.stride, self.refs, r)))
                .sum(),
            None => 0,
        };
        Ok(MergeChunk {
            part: p,
            refs,
            states,
            heap,
        })
    }

    /// [`memory_size`](Self::memory_size) after
    /// [`apply_merge`](Self::apply_merge) of `c` from `src`, plus the
    /// merge's scratch (an upper bound for the merge to reserve).
    /// The partition of this (merge) table that chunk `c` lands in: its
    /// own partition, or the only one of a table with one partition (which
    /// merges any input, whatever its width).
    fn merge_part(&self, c: &MergeChunk) -> usize {
        if self.bits == 0 { 0 } else { c.part }
    }

    pub(crate) fn projected_merge(&self, c: &MergeChunk) -> usize {
        let n = c.refs.len();
        let part = self.merge_part(c);
        let rows: usize = self
            .parts
            .iter()
            .enumerate()
            .map(|(p, v)| {
                if p == part {
                    grown(&v.rows, n * self.stride) + grown(&v.heap, c.heap)
                } else {
                    v.rows.capacity() + v.heap.capacity()
                }
            })
            .sum();
        let mut new = vec![0; self.parts.len()];
        if let Some(x) = new.get_mut(part) {
            *x = n;
        }
        let k = self.core.spec.group_by.len();
        rows + self.index_bytes(self.mode, n)
            + self.projected_accs(&new, &c.states, Fold::States { offset: k })
            + Self::scratch(n)
            + self.sketch_bytes()
    }

    /// Merges a chunk of `src`'s rows (rows are copied, states merged).
    /// Call after reserving [`projected_merge`](Self::projected_merge). Both
    /// tables have the same radix bits, so the rows stay in partition
    /// `c.part`.
    pub(crate) fn apply_merge(&mut self, src: &Self, c: &MergeChunk) -> Result<()> {
        if self.mode != Mode::Grow {
            return Err(internal(
                "merging into a table that is not growing".to_owned(),
            ));
        }
        if self.bits != 0 && self.bits != src.bits {
            return Err(internal(
                "merging tables of different radix bits".to_owned(),
            ));
        }
        let part = self.merge_part(c);
        let mut refs = Vec::with_capacity(c.refs.len());
        self.reserve_index(c.refs.len());
        if let Some(v) = self.parts.get_mut(part) {
            v.rows.reserve(c.refs.len() * self.stride);
            v.heap.reserve(c.heap);
        }
        let mut result = Ok(());
        let mut count = ProbeCount::default();
        let counted = self.probes.is_some();
        for &r in &c.refs {
            let (row, heap) = row_heap_at(&src.parts, src.stride, src.refs, r);
            let h = word(row, 0);
            let found = match (self.views.is_some(), counted) {
                (true, false) => {
                    self.find_or_insert::<true, false>(row, heap, h, usize::MAX, &mut count)
                }
                (false, false) => {
                    self.find_or_insert::<false, false>(row, heap, h, usize::MAX, &mut count)
                }
                (true, true) => {
                    self.find_or_insert::<true, true>(row, heap, h, usize::MAX, &mut count)
                }
                (false, true) => {
                    self.find_or_insert::<false, true>(row, heap, h, usize::MAX, &mut count)
                }
            };
            match found {
                Ok(id) => refs.push(id),
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        if let Some(totals) = &self.probes
            && count.lookups > 0
        {
            totals.add(&count);
        }
        let mut counts = vec![0; self.parts.len()];
        if let Some(x) = counts.get_mut(part) {
            *x = refs.len();
        }
        let k = self.core.spec.group_by.len();
        result.and_then(|()| {
            self.fold(
                &refs,
                &counts,
                &c.states,
                &Weights::Const(1),
                Fold::States { offset: k },
            )
        })
    }

    /// Moves partitions `ranges` out into tables of their own (in order):
    /// each holds its range's rows and states, no index or sketch,
    /// and takes no more rows (append mode, never merged into). This table
    /// keeps the partitions no range names, and drops its index.
    pub(crate) fn split_parts(&mut self, ranges: &[Range<usize>]) -> Vec<Self> {
        self.index = HashTable::new();
        self.sketches = None;
        self.sketching = false;
        self.adapt = None;
        ranges
            .iter()
            .map(|range| {
                let mut parts: Vec<Part> = (0..self.parts.len()).map(|_| Part::default()).collect();
                let mut groups = 0;
                for p in range.clone() {
                    if let (Some(from), Some(to)) = (self.parts.get_mut(p), parts.get_mut(p)) {
                        groups += from.rows.len() / self.stride;
                        *to = std::mem::take(from);
                    }
                }
                self.groups -= groups;
                Self {
                    core: Arc::clone(&self.core),
                    gseed: self.gseed,
                    stride: self.stride,
                    bits: self.bits,
                    refs: self.refs,
                    views: self.views.clone(),
                    parts,
                    sketches: None,
                    sketching: false,
                    local: false,
                    widen_to: self.bits,
                    widen_at: usize::MAX,
                    index: HashTable::new(),
                    mode: Mode::Append,
                    adapt: None,
                    groups,
                    rows_in: 0,
                    unique: self.unique,
                    abandons: None,
                    resizes: None,
                    probes: None,
                }
            })
            .collect()
    }

    /// Frees partition `p` (rows and states); its rows no longer count.
    pub(crate) fn release_part(&mut self, p: usize) {
        if let Some(part) = self.parts.get_mut(p) {
            self.groups -= part.rows.len() / self.stride;
            *part = Part::default();
        }
    }
}

#[cfg(test)]
thread_local! {
    /// Tests: [`RowTable::widen`] fails once it has copied the old
    /// partition of this index.
    static WIDEN_FAULT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// The fault tests inject into [`RowTable::widen`] after it copied old
/// partition `copied` (none outside tests).
#[cfg(not(test))]
#[inline]
fn widen_fault(_copied: usize) -> Result<()> {
    Ok(())
}

#[cfg(test)]
fn widen_fault(copied: usize) -> Result<()> {
    if WIDEN_FAULT.get() == Some(copied) {
        return Err(internal("injected widening fault".to_owned()));
    }
    Ok(())
}

/// One accumulator per aggregate of `core`.
fn new_accs(core: &AggCore) -> Result<Vec<Accumulator>> {
    core.spec
        .aggregates
        .iter()
        .map(|a| core.accumulator(a))
        .collect()
}

/// The table's row layout (present whenever row tables are made).
fn layout(core: &AggCore) -> Result<&RowLayout> {
    core.rows
        .as_ref()
        .ok_or_else(|| unsupported("row-layout aggregation of these key types".to_owned()))
}

impl RowTable {
    /// Whether two row references are in one radix partition.
    pub(crate) fn same_part(&self, a: u32, b: u32) -> bool {
        self.refs.part(a) == self.refs.part(b)
    }

    /// The partition and row of a row reference.
    pub(crate) fn split_ref(&self, r: u32) -> (usize, usize) {
        (self.refs.part(r), self.refs.row(r) as usize)
    }
}

/// Row `r` of `parts` and its partition's heap (both empty if out of
/// range).
#[inline(always)]
fn row_heap_at(parts: &[Part], stride: usize, refs: Refs, r: u32) -> (&[u8], &[u8]) {
    let at = refs.row(r) as usize * stride;
    parts
        .get(refs.part(r))
        .and_then(|v| Some((v.rows.get(at..at + stride)?, v.heap.as_slice())))
        .unwrap_or_default()
}

/// Row `r` of `parts` (empty if out of range).
#[inline]
fn row_at(parts: &[Part], stride: usize, refs: Refs, r: u32) -> &[u8] {
    let p = refs.part(r);
    let at = refs.row(r) as usize * stride;
    parts
        .get(p)
        .and_then(|v| v.rows.get(at..at + stride))
        .unwrap_or_default()
}

/// Whether `row` holds the key of key row `key`: the same hash and key
/// bytes. Fixed-width rows hold the whole key, NULLs and padding as zero
/// bytes and floats normalized, so equal keys have equal bytes (the
/// layout's word compare without views).
#[inline(always)]
fn same_key(row: &[u8], key: &[u8], stride: usize) -> bool {
    match (row.get(..stride), key.get(..stride)) {
        (Some(a), Some(b)) => {
            let (a, _) = a.as_chunks::<8>();
            let (b, _) = b.as_chunks::<8>();
            a.iter().zip(b).all(|(x, y)| x == y)
        }
        _ => false,
    }
}

/// Appends a row for key row `key` (hash `h`; its long values in
/// `key_heap`, copied to the partition's heap). `shape` is the table's
/// stride, radix bits, reference split and view layout.
#[inline]
fn push_row(
    parts: &mut [Part],
    sketches: &mut Option<Vec<Hll>>,
    groups: &mut usize,
    (stride, bits, refs, views): (usize, u32, Refs, Option<&RowLayout>),
    (key, key_heap): (&[u8], &[u8]),
    h: u64,
) -> Result<u32> {
    let p = part_of(h, bits);
    let part = parts
        .get_mut(p)
        .ok_or_else(|| internal(format!("radix partition {p} out of range")))?;
    let row = part.rows.len() / stride;
    if row > refs.mask() as usize {
        return Err(unsupported(format!(
            "more than 2^{} groups in one radix partition of an aggregation table",
            refs.row_bits()
        )));
    }
    match views {
        Some(l) => l.push_row(key, key_heap, &mut part.rows, &mut part.heap)?,
        None => {
            let key = key
                .get(..stride)
                .ok_or_else(|| internal("a key row shorter than the stride".to_owned()))?;
            part.rows.extend_from_slice(key);
        }
    }
    if let Some(s) = sketches.as_mut().and_then(|s| s.get_mut(p)) {
        s.add(h);
    }
    *groups += 1;
    Ok(refs.make(p, row))
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

/// The final phase's tasks over tables of one radix width: the task count
/// follows the estimated group count, so small results merge in one task
/// and large ones in up to one task per partition, merged in parallel.
/// Partitions without rows get no task.
pub(crate) fn final_tasks(tables: &[&RowTable]) -> Vec<FinalTask> {
    let nparts = tables.first().map_or(1, |t| t.part_count());
    let mut est = Vec::with_capacity(nparts);
    for p in 0..nparts {
        let rows: usize = tables.iter().map(|t| t.part_len(p)).sum();
        // Tables without sketches (small ones) count their rows.
        let mut sketch = Hll::new(PART_HLL_BITS);
        let (mut sketched, mut unsketched) = (false, 0);
        for t in tables {
            match t.sketches().and_then(|s| s.get(p)) {
                Some(s) => {
                    sketch.merge(s);
                    sketched = true;
                }
                None => unsketched += t.part_len(p),
            }
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a non-negative estimate, capped by the row count"
        )]
        let groups = if sketched {
            ((sketch.estimate() * 1.1) as usize + 16 + unsketched).min(rows)
        } else {
            rows
        };
        est.push((rows, groups));
    }
    #[expect(clippy::cast_precision_loss, reason = "a group-count estimate")]
    let total = est.iter().map(|e| e.1).sum::<usize>() as f64;
    let mut bits = 0;
    while (1 << bits) < nparts && total / f64::from(1u32 << bits) > TASK_GROUPS {
        bits += 1;
    }
    let width = nparts >> bits;
    (0..nparts)
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
    /// estimate).
    pub(crate) fn presize(&mut self, groups: usize) {
        let (parts, stride, gseed, refs) = (&self.parts, self.stride, self.gseed, self.refs);
        let before = self.index.capacity();
        self.index.reserve(groups, |&r| {
            fin(word(row_at(parts, stride, refs, r), 0), gseed)
        });
        self.count_resize(before);
    }

    /// [`memory_size`](Self::memory_size) after [`presize`](Self::presize).
    pub(crate) fn projected_presize(&self, groups: usize) -> usize {
        self.memory_size() - buckets(&self.index) * BUCKET_BYTES
            + self.index_bytes(Mode::Grow, groups)
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Int64Array, StringArray};
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
            legacy_statistical: false,
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

    /// A table grouping by a `Utf8` key (`v` an `Int64`).
    fn core_str() -> Arc<AggCore> {
        let s = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Utf8, true),
            Field::new("v", DataType::Int64, true),
        ]));
        let spec = AggregateSpec {
            input: Arc::clone(&s),
            group_by: vec![0],
            aggregates: vec![AggregateExpr::count_star("c")],
            output: AggregateOutput::Final,
            ansi: true,
            spill: None,
            require_unit_weights: false,
            legacy_statistical: false,
        };
        Arc::new(AggCore::new(spec).unwrap())
    }

    /// Keys of 200 bytes (heap values) for [`core_str`].
    fn long_keys(core: &AggCore, keys: impl Iterator<Item = i64>) -> Batch {
        let k: Vec<String> = keys.map(|i| format!("{i:0>200}")).collect();
        let v = Int64Array::from(vec![1; k.len()]);
        let k = arrow_array::StringArray::from(k);
        let cols: Vec<ArrayRef> = vec![Arc::new(k), Arc::new(v)];
        Batch::new(RecordBatch::try_new(Arc::clone(&core.spec.input), cols).unwrap())
    }

    /// Long string keys live in the partitions' heaps: the table counts
    /// them, and the projections of an apply and of a merge bound them.
    #[test]
    fn heap_bytes_are_accounted() {
        let core = core_str();
        let seed = HashSeed::new(Some(9));
        for mode in [Mode::Grow, Mode::Fixed { limit: 64 }, Mode::Append] {
            let mut t = RowTable::new(&core, &seed, mode, true, RADIX_BITS).unwrap();
            for start in (0..2_000).step_by(500) {
                apply(&mut t, &long_keys(&core, start..start + 500));
            }
            // Every key is new: 2,000 values of 200 bytes in the heaps.
            assert!(
                t.memory_size() >= 2_000 * 200,
                "{mode:?}: {}",
                t.memory_size()
            );
            let mut merged = RowTable::new(&core, &seed, Mode::Grow, false, RADIX_BITS).unwrap();
            for p in 0..PARTS {
                let c = t.merge_chunk(p, 0..t.part_len(p)).unwrap();
                let want = merged.projected_merge(&c);
                merged.apply_merge(&t, &c).unwrap();
                assert!(
                    merged.memory_size() <= want,
                    "{} > {want}",
                    merged.memory_size()
                );
            }
            assert!(merged.memory_size() >= 2_000 * 200);
            let out = merged.output_batch(&merged.ids(0..merged.len())).unwrap();
            assert_eq!(out.num_rows(), 2_000);
        }
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
        let mut grow = RowTable::new(&core, &seed, Mode::Grow, false, RADIX_BITS).unwrap();
        let mut fixed =
            RowTable::new(&core, &seed, Mode::Fixed { limit: 8 }, true, RADIX_BITS).unwrap();
        let mut append = RowTable::new(&core, &seed, Mode::Append, true, RADIX_BITS).unwrap();
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
        let mut merged = RowTable::new(&core, &seed, Mode::Grow, false, RADIX_BITS).unwrap();
        // Different register counts do not merge.
        let mut one = Hll::new(PART_HLL_BITS);
        one.add(1);
        one.merge(&Hll::new(ADAPT_HLL_BITS));
        assert_eq!(one.used(), 1);
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
        let mut unique =
            RowTable::new(&core, &seed, Mode::Fixed { limit: 1024 }, true, RADIX_BITS).unwrap();
        let mut repeat =
            RowTable::new(&core, &seed, Mode::Fixed { limit: 1024 }, true, RADIX_BITS).unwrap();
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
        // Growth stops at MAX_SINK_SLOTS (limit = 2/3 of the slots): from
        // 2^21 slots a 4x step would pass it; from 2^20 it lands on it.
        let grow_from = |slots: usize| {
            let mut t = RowTable::new(
                &core,
                &seed,
                Mode::Fixed {
                    limit: slots * 2 / 3,
                },
                true,
                RADIX_BITS,
            )
            .unwrap();
            let a = t.adapt.as_mut().unwrap();
            (a.rows, a.pushed) = (ADAPT_ROWS, ADAPT_ROWS);
            for h in 0..64 {
                a.sketch.add(mix(h));
            }
            t.next_mode()
        };
        assert_eq!(
            grow_from(1 << 21),
            Mode::Fixed {
                limit: (1 << 21) * 2 / 3
            }
        );
        assert_eq!(
            grow_from(1 << 20),
            Mode::Fixed {
                limit: (1 << 20) * 2 / 3 * 4
            }
        );
        const { assert!(((1 << 20) * 2 / 3 * 4) * 3 / 2 <= MAX_SINK_SLOTS) };
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
        let mut small =
            RowTable::new(&core, &seed, Mode::Fixed { limit: 64 }, true, RADIX_BITS).unwrap();
        apply(&mut small, &batch(&core, 0..1_000));
        let tasks = final_tasks(&[&small]);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].parts, 0..PARTS);
        assert!(
            tasks[0].groups >= 1_000 && tasks[0].groups <= 1_200,
            "{:?}",
            tasks[0]
        );
        let mut big = RowTable::new(&core, &seed, Mode::Append, true, RADIX_BITS).unwrap();
        for s in (0..600_000).step_by(8192) {
            apply(&mut big, &batch(&core, s..(s + 8192).min(600_000)));
        }
        let tasks = final_tasks(&[&big, &small]);
        assert_eq!(tasks.len(), PARTS);
        assert!(tasks.iter().all(|t| t.parts.len() == 1));
        assert!(final_tasks(&[]).is_empty());
    }

    /// The sorted (key, count, sum) rows of a final output batch.
    fn sorted(b: &RecordBatch) -> Vec<(Option<i64>, Option<i64>, Option<i64>)> {
        let col = |i: usize| b.column(i).as_any().downcast_ref::<Int64Array>().unwrap();
        let mut v: Vec<_> = (0..b.num_rows())
            .map(|i| {
                let get = |c: &Int64Array| c.is_valid(i).then(|| c.value(i));
                (get(col(0)), get(col(1)), get(col(2)))
            })
            .collect();
        v.sort_unstable();
        v
    }

    /// One partition (one thread) and sixteen fold the same groups; output
    /// across partitions concatenates each partition's states in the order
    /// of the references.
    #[test]
    fn one_partition_equals_sixteen() {
        let core = core();
        let seed = HashSeed::new(Some(4));
        let mut one = RowTable::new(&core, &seed, Mode::Grow, false, 0).unwrap();
        let mut many = RowTable::new(&core, &seed, Mode::Grow, false, RADIX_BITS).unwrap();
        assert_eq!((one.part_count(), many.part_count()), (1, PARTS));
        for t in [&mut one, &mut many] {
            apply(t, &batch(&core, (0..3_000).map(|i| i % 700)));
            apply(t, &batch(&core, 0..10));
        }
        let all = |t: &RowTable| t.output_batch(&t.ids(0..t.len())).unwrap();
        assert_eq!(sorted(&all(&one)), sorted(&all(&many)));
        assert_eq!(one.len(), 700);
        // References out of partition order still line up with their keys.
        let mut refs = many.ids(0..many.len());
        refs.reverse();
        let b = many.output_batch(&refs).unwrap();
        assert_eq!(sorted(&b), sorted(&all(&one)));
        let states = many.state_batch(&refs[..5]).unwrap();
        assert_eq!(states.num_rows(), 5);
        // No rows: empty columns of the right types.
        assert_eq!(many.output_batch(&[]).unwrap().num_rows(), 0);
        // A reference past a partition's rows is an internal error.
        let bad = many.refs.make(0, many.part_len(0) + 1);
        assert!(many.output_batch(&[bad]).is_err());
        assert!(many.merge_chunk(0, 0..many.part_len(0) + 1).is_err());
        // A one-partition table merges every partition of a wider one
        // (the final phase's merge of inputs of mixed widths)...
        let mut grow = RowTable::new(&core, &seed, Mode::Grow, false, 0).unwrap();
        for p in 0..PARTS {
            let c = many.merge_chunk(p, 0..many.part_len(p)).unwrap();
            assert!(grow.projected_merge(&c) >= grow.memory_size());
            grow.apply_merge(&many, &c).unwrap();
        }
        assert_eq!(sorted(&all(&grow)), sorted(&all(&one)));
        // ... but a partitioned one needs the same radix bits.
        let c = many.merge_chunk(0, 0..1).unwrap();
        let mut two = RowTable::new(&core, &seed, Mode::Grow, false, 1).unwrap();
        let e = two.apply_merge(&many, &c).unwrap_err();
        assert!(e.to_string().contains("radix bits"), "{e}");
    }

    /// A thread-local table starts its partition sketches at
    /// `SKETCH_ROWS` rows, from the rows it already has: smaller tables
    /// hold none (their row counts are the final phase's estimate).
    #[test]
    fn sketches_start_with_enough_rows() {
        let core = core();
        let seed = HashSeed::new(Some(6));
        let mut t = RowTable::new(&core, &seed, Mode::Grow, true, RADIX_BITS).unwrap();
        apply(&mut t, &batch(&core, 0..1_000));
        assert!(t.sketches().is_none());
        let before = t.memory_size();
        let n = i64::try_from(SKETCH_ROWS).unwrap();
        apply(&mut t, &batch(&core, 1_000..n));
        assert!(t.memory_size() > before);
        let s = t.sketches().expect("sketches from SKETCH_ROWS rows");
        let mut all = Hll::new(PART_HLL_BITS);
        s.iter().for_each(|h| all.merge(h));
        #[expect(clippy::cast_precision_loss, reason = "test counts")]
        let rows = SKETCH_ROWS as f64;
        assert!(
            (all.estimate() - rows).abs() < rows * 0.1,
            "{}",
            all.estimate()
        );
        // One partition, or a merge table: never.
        let mut one = RowTable::new(&core, &seed, Mode::Grow, true, 0).unwrap();
        apply(&mut one, &batch(&core, 0..n));
        assert!(one.sketches().is_none());
        // A small table among sketched ones counts its rows.
        let mut small = RowTable::new(&core, &seed, Mode::Grow, true, RADIX_BITS).unwrap();
        apply(&mut small, &batch(&core, n..n + 100));
        let groups: usize = final_tasks(&[&t, &small]).iter().map(|f| f.groups).sum();
        assert!(
            (SKETCH_ROWS + 100..=(SKETCH_ROWS + 100) * 12 / 10).contains(&groups),
            "{groups}"
        );
    }

    /// Widening spreads a narrow table's rows, long key bytes and states
    /// over the radix partitions without changing any group: a growing
    /// index still finds every key, a fixed one is cleared, and sketches
    /// start.
    #[test]
    fn widening_keeps_every_group() {
        let core = core();
        let seed = HashSeed::new(Some(8));
        for mode in [Mode::Grow, Mode::Fixed { limit: 5_000 }, Mode::Append] {
            let mut t = RowTable::new(&core, &seed, mode, true, 0)
                .unwrap()
                .widening(RADIX_BITS, 4_000);
            let mut narrow = RowTable::new(&core, &seed, mode, true, 0).unwrap();
            for tb in [&mut t, &mut narrow] {
                apply(tb, &batch(&core, (0..6_000).map(|i| i % 4_500)));
            }
            assert!(t.wants_widen() && !narrow.wants_widen());
            let all = |t: &RowTable| sorted(&t.output_batch(&t.ids(0..t.len())).unwrap());
            let want = all(&t);
            let bytes = t.memory_size();
            // The reservation covers the copy, the state arrays and the
            // row lists the widening holds while it runs.
            let accs: usize = t
                .parts
                .iter()
                .flat_map(|p| &p.accs)
                .map(Accumulator::size)
                .sum();
            let lists = 2 * t.len() * size_of::<u32>();
            assert!(t.widen_bytes() >= bytes + 2 * accs + lists);
            t.widen().unwrap();
            assert_eq!((t.part_count(), t.wants_widen()), (PARTS, false));
            assert_eq!(all(&t), want, "{mode:?}");
            // The copy grows each partition once, exactly: no more bytes
            // than before but for the sketches it starts.
            let sketches = PARTS * Hll::new(PART_HLL_BITS).bytes();
            assert!(t.memory_size() <= bytes + sketches, "{mode:?}");
            assert!(t.sketches().is_some());
            let groups = t.len();
            apply(&mut t, &batch(&core, 0..4_500));
            match mode {
                // Every key is found again: no new rows, still unique.
                Mode::Grow => assert_eq!((t.len(), t.is_unique()), (groups, true)),
                _ => assert!(t.len() > groups && !t.is_unique(), "{mode:?}"),
            }
            // Widening an already wide table changes nothing.
            let n = t.len();
            t.widen().unwrap();
            assert_eq!(t.len(), n);
        }
    }

    /// A widening that fails part-way leaves the table as it was: every
    /// group with its states, its width and an index that still finds
    /// every key; a later widening succeeds.
    #[test]
    fn failed_widening_keeps_the_table() {
        let core = core();
        let seed = HashSeed::new(Some(8));
        for (bits, copied) in [(0, 0), (2, 1)] {
            let mut t = RowTable::new(&core, &seed, Mode::Grow, true, bits)
                .unwrap()
                .widening(RADIX_BITS, 64);
            apply(&mut t, &batch(&core, (0..3_000).map(|i| i % 2_500)));
            assert!(t.wants_widen());
            let all = |t: &RowTable| sorted(&t.output_batch(&t.ids(0..t.len())).unwrap());
            let (want, groups, parts) = (all(&t), t.len(), t.part_count());
            WIDEN_FAULT.set(Some(copied));
            let err = t.widen().unwrap_err();
            WIDEN_FAULT.set(None);
            assert!(err.to_string().contains("injected widening fault"), "{err}");
            assert_eq!((t.len(), t.part_count()), (groups, parts), "bits {bits}");
            assert_eq!(all(&t), want, "bits {bits}");
            // The index still finds every key: no new groups.
            apply(&mut t, &batch(&core, 0..2_500));
            assert_eq!((t.len(), t.is_unique()), (groups, true), "bits {bits}");
            t.widen().unwrap();
            assert_eq!(t.part_count(), PARTS);
            let mut twice = want.clone();
            for row in &mut twice {
                row.1 = row.1.map(|c| c + 1);
                row.2 = row.2.map(|v| v + 1);
            }
            assert_eq!(all(&t), twice, "bits {bits}");
        }
    }

    /// Row references split into partition and row by the table's radix
    /// bits: one partition keeps all 32 bits for rows (2^32 groups).
    #[test]
    fn refs_split_by_radix_bits() {
        let one = Refs::new(0);
        assert_eq!(
            (one.mask(), one.part(u32::MAX), one.row(u32::MAX)),
            (u32::MAX, 0, u32::MAX)
        );
        assert_eq!(one.make(0, 5), 5);
        let many = Refs::new(RADIX_BITS);
        assert_eq!(many.mask(), (1 << 28) - 1);
        let r = many.make(PARTS - 1, 7);
        assert_eq!((many.part(r), many.row(r)), (PARTS - 1, 7));
        // Never more than RADIX_BITS.
        assert_eq!(Refs::new(9), many);
    }

    /// A partition with few rows can hold most of a variable-width
    /// argument column's bytes, so the gather bound charges the full
    /// column for types without a primitive width.
    #[test]
    fn gather_bytes_covers_skewed_var_width() {
        let s = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("v", DataType::Utf8, true),
        ]));
        let spec = AggregateSpec {
            input: Arc::clone(&s),
            group_by: vec![0],
            aggregates: vec![AggregateExpr::new(
                AggregateFunction::Min,
                1,
                DataType::Utf8,
                "m",
            )],
            output: AggregateOutput::Final,
            ansi: true,
            spill: None,
            require_unit_weights: false,
            legacy_statistical: false,
        };
        let core = Arc::new(AggCore::new(spec).unwrap());
        let seed = HashSeed::new(Some(6));
        let t = RowTable::new(&core, &seed, Mode::Grow, false, RADIX_BITS).unwrap();
        // Exact-capacity values: a grown `Vec` can leave slack in the
        // values buffer, which would hide the skew.
        let mut values = Vec::with_capacity(4103);
        values.extend(std::iter::repeat_n(b'x', 4096));
        values.extend(std::iter::repeat_n(b'y', 7));
        let mut offsets = Vec::with_capacity(9);
        offsets.push(0i32);
        offsets.push(4096);
        for i in 1..=7 {
            offsets.push(4096 + i);
        }
        let strings: ArrayRef = Arc::new(StringArray::new(
            arrow_buffer::OffsetBuffer::new(offsets.into()),
            arrow_buffer::Buffer::from_vec(values),
            None,
        ));
        let keys: ArrayRef = Arc::new(Int64Array::from(vec![0; 8]));
        let cols = vec![keys, Arc::clone(&strings)];
        let mut counts = vec![0; PARTS];
        counts[0] = 1;
        counts[1] = 7;
        let got = t.gather_bytes(&cols, Fold::Raw, &Weights::Const(1), &counts);
        let take = |idx: &[u32]| {
            arrow_select::take::take(strings.as_ref(), &UInt32Array::from(idx.to_vec()), None)
                .unwrap()
                .get_array_memory_size()
        };
        let small = take(&[0]);
        let large = take(&[1, 2, 3, 4, 5, 6, 7]);
        assert!(got >= small, "{got} < {small}");
        assert!(got >= large, "{got} < {large}");
    }

    /// Partitions split out of a table carry their rows and states; the
    /// rest stays; a released partition frees its memory.
    #[test]
    fn split_and_release_parts() {
        let core = core();
        let seed = HashSeed::new(Some(5));
        let mut t =
            RowTable::new(&core, &seed, Mode::Fixed { limit: 100 }, true, RADIX_BITS).unwrap();
        apply(&mut t, &batch(&core, (0..5_000).map(|i| i % 1_000)));
        let total = t.len();
        let want = sorted(&t.output_batch(&t.ids(0..total)).unwrap());
        let mut slices = t.split_parts(&[0..4, 4..PARTS]);
        assert_eq!(t.len(), 0);
        assert_eq!(t.buckets(), 0);
        assert_eq!(slices.iter().map(RowTable::len).sum::<usize>(), total);
        let mut got = Vec::new();
        for s in &slices {
            got.extend(sorted(&s.output_batch(&s.ids(0..s.len())).unwrap()));
        }
        got.sort_unstable();
        assert_eq!(got, want);
        let s = &mut slices[1];
        let before = s.memory_size();
        let n4 = s.part_len(4);
        s.release_part(4);
        assert_eq!(s.part_len(4), 0);
        assert!(s.memory_size() < before || n4 == 0);
        for p in 5..PARTS {
            s.release_part(p);
        }
        assert_eq!((s.len(), s.memory_size()), (0, 0));
    }
}
