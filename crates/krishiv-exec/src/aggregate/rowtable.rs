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
//!   a second row (a partial duplicate merged by the final phase). An
//!   [`Hll`] of a 1/8 sample of the input hashes estimates the keys. After
//!   an index filled, it grows to the estimate's capacity (at least 2x)
//!   once the partial duplicates outweigh [`GROW_DUPLICATE_FACTOR`] times
//!   the larger index (DuckDB 2.0's `RadixHTAdaptivity::TryGrow`), if the
//!   keys fit in [`MAX_SINK_SLOTS`], and is rebuilt over the rows. Every
//!   [`ADAPT_ROWS`] input rows, it switches to **append** (no lookups)
//!   when over 95% of the keys are unique. This is DuckDB's thread-local
//!   abandon (fixed-size thread-local tables,
//!   <https://duckdb.org/2022/03/07/aggregate-hashtable.html>;
//!   `radix_partitioned_hashtable.cpp`) with its capacities: 2^15 slots
//!   above two threads, 2^18 at two. Growing only every [`ADAPT_ROWS`]
//!   rows, by 4x, left TPC-DS q22's index (204K groups a driver) clearing
//!   for its first 3M rows: ten times the columnar path's partial rows.
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
use krishiv_kernels::prefetch_read;
use krishiv_types::batch::{Batch, Weights};

use super::accumulator::{Accumulator, paired};
use super::merge_index::{MergeIndex, Slot};
use super::table::{AggCore, batch, own_bytes};
use super::{AggregateOutput, unsupported};
use crate::error::{Result, internal};
use crate::hash::{HashSeed, PARTITIONS, ProbeCount, ProbeTotals, grown, spill_partition};
use crate::rows::{EncodedKeys, FLAT_MAX_BYTES, Hll, RowLayout};

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
/// Largest capacity a fixed index grows to: 2.5 MiB of index, about one
/// thread's share of a 32 MiB L3 at 12 threads. Past the caches a lookup
/// costs about as much as the partial duplicate it saves (P2-T159 step 0:
/// 3-5 LLC misses a row in a growing table): TPC-DS q67, about 5.8M groups,
/// ran 8% slower growing to 2^22 slots.
const MAX_SINK_SLOTS: usize = 1 << 19;
/// Input rows between append decisions of a fixed index.
pub(crate) const ADAPT_ROWS: usize = 1 << 20;
/// Bytes of partial duplicates per byte of the grown index from which a
/// full fixed index grows (DuckDB 2.0's `GROWTH_DUPLICATE_BYTES_FACTOR`).
const GROW_DUPLICATE_FACTOR: usize = 4;
/// Bytes per index bucket: a `u32` reference and hashbrown's control byte.
const BUCKET_BYTES: usize = size_of::<u32>() + 1;

/// Rows of a merge whose slots and candidate rows are prefetched together
/// (the join's lookups measured the same at 32, 64 and 256).
const PREFETCH_ROWS: usize = 64;

/// The most distinct keys a batch may hold for the batch-local key map
/// ([`RowTable::probe_local`]): beyond it the batch takes the per-row probe.
const LOCAL_MAX: usize = 256;

/// Batches the per-row probe runs for after a batch exceeded [`LOCAL_MAX`],
/// before the local map is tried again.
const LOCAL_BACKOFF: u8 = 16;

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
// A fused aggregation reports these sketches as its exchange's.
const _: () = assert!(PART_HLL_BITS == crate::EXCHANGE_HLL_BITS);
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
    /// The index was cleared since the last apply decided its mode.
    filled: bool,
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
    /// A large merge table's index, instead of `index` ([`Self::presize`]).
    merge_index: Option<MergeIndex>,
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
    /// Batches left before the batch-local key map is tried again, after a
    /// batch held more than [`LOCAL_MAX`] distinct keys.
    local_skip: u8,
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
            merge_index: None,
            mode,
            adapt: (local && matches!(mode, Mode::Fixed { .. })).then(|| Adapt {
                sketch: Hll::new(ADAPT_HLL_BITS),
                rows: 0,
                pushed: 0,
                next: ADAPT_ROWS,
                window: (0, 0),
                early: false,
                filled: false,
            }),
            groups: 0,
            rows_in: 0,
            unique: true,
            abandons: None,
            resizes: None,
            probes: None,
            local_skip: 0,
        })
    }

    /// Number of rows (groups; a group may have several rows in a table
    /// whose fixed index was cleared).
    pub(crate) fn len(&self) -> usize {
        self.groups
    }

    /// Calls `f` with the stored `h` of every row, partition by partition.
    pub(crate) fn for_each_hash(&self, mut f: impl FnMut(u64)) {
        for part in &self.parts {
            for row in part.rows.chunks_exact(self.stride) {
                f(word(row, 0));
            }
        }
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
    /// new rows, a fixed one over up to its limit of them
    /// ([`reindex`](Self::reindex)); append mode has none. Partition
    /// sketches start from the rows.
    ///
    /// The copy is built beside the table, which changes only once it is
    /// complete: on an error the table keeps its rows, states and width.
    pub(crate) fn widen(&mut self) -> Result<()> {
        if self.widen_to <= self.bits {
            return Ok(());
        }
        self.only_hashbrown()?;
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
            // Rebuilt like a grown one: an empty index would make one more
            // partial row of every key seen again (q39, q65, q67 and q75
            // widen early). A unique table keeps all its keys (they fit).
            Mode::Fixed { limit } => self.reindex(limit),
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
    /// and the sketches it starts. The rebuilt index is not counted again:
    /// the old one is dropped first and the new one is no larger (a fixed
    /// index holds at most the keys its capacity held).
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
        self.merge_index
            .as_ref()
            .map_or_else(|| buckets(&self.index), MergeIndex::slots)
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
            + self.merge_index.as_ref().map_or(0, MergeIndex::bytes)
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

    /// The mode the next apply runs in: a fixed index turns append-only
    /// every [`ADAPT_ROWS`] input rows if nearly every key is unique, and
    /// grows after it filled ([`grown_limit`](Self::grown_limit)).
    fn next_mode(&self) -> Mode {
        let (Mode::Fixed { limit }, Some(a)) = (self.mode, &self.adapt) else {
            return self.mode;
        };
        if a.rows >= a.next {
            #[expect(clippy::cast_precision_loss, reason = "row counts in a ratio")]
            let rows = a.rows as f64;
            if Self::distinct(a) / rows > 0.95 {
                return Mode::Append;
            }
        }
        match a.filled.then(|| self.grown_limit(limit, a)).flatten() {
            Some(limit) => Mode::Fixed { limit },
            None => self.mode,
        }
    }

    /// The input's distinct keys, estimated from the sample (1/8 of the
    /// hashes); at least 1.
    fn distinct(a: &Adapt) -> f64 {
        (a.sketch.estimate() * 8.0).max(1.0)
    }

    /// The limit a full fixed index of `limit` keys grows to (DuckDB 2.0's
    /// `TryGrow`, research C11): room for the distinct keys at load 2/3,
    /// and at least twice the slots; `None` when that passes
    /// [`MAX_SINK_SLOTS`] (the keys would not fit, so a larger index only
    /// adds cache misses: DuckDB grows to its memory limit instead), and
    /// while the partial duplicates (rows beyond the distinct keys) take
    /// fewer than [`GROW_DUPLICATE_FACTOR`] times the grown index's bytes.
    /// The growth stops a table's index from clearing over and over when
    /// its keys repeat but do not fit: every row it makes then is a partial
    /// duplicate that the final phase copies and merges again.
    fn grown_limit(&self, limit: usize, a: &Adapt) -> Option<usize> {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a non-negative estimate, capped by the row count"
        )]
        let distinct = (Self::distinct(a) as usize).min(self.groups);
        let slots = (limit * 3 / 2).next_power_of_two();
        let grown = (distinct * 3 / 2).next_power_of_two().max(slots * 2);
        if grown > MAX_SINK_SLOTS {
            return None;
        }
        let grown = grown * 2 / 3;
        let index = buckets_for(grown) * BUCKET_BYTES;
        ((self.groups - distinct) * self.stride >= GROW_DUPLICATE_FACTOR * index).then_some(grown)
    }

    /// Index bytes in `mode` after `new` more keys: a growing index grows
    /// with its keys, a fixed one up to its limit (it is cleared there).
    fn index_bytes(&self, mode: Mode, new: usize) -> usize {
        if let Some(index) = &self.merge_index {
            return index.projected(new);
        }
        let len = match mode {
            Mode::Append => return 0,
            // A grown index is rebuilt over the rows ([`reindex`](Self::reindex)).
            Mode::Fixed { limit } if mode != self.mode && !self.unique => {
                self.groups.saturating_add(new).min(limit)
            }
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
        if self.merge_index.is_some() {
            return;
        }
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
        for (c, _) in cols.iter().zip(&used).filter(|(_, u)| **u) {
            let n = if c.data_type().primitive_width().is_some() {
                &mut fixed
            } else {
                &mut varied
            };
            *n += own_bytes(c.as_ref());
        }
        // Weights are `Int64`: one value per row, so a proportional share.
        fixed += match w {
            Weights::Column(c) => own_bytes(c),
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
            self.merge_index = None;
        }
        let grew = mode != self.mode && mode != Mode::Append;
        self.mode = mode;
        if grew
            && !self.unique
            && let Mode::Fixed { limit } = mode
        {
            self.reindex(limit);
        }
        let Some(a) = self.adapt.as_mut() else {
            return;
        };
        a.filled = false;
        if grew {
            // The early-widening rate counts from here: the rows of the
            // clears before are not the input's new keys.
            a.window = (a.rows, a.pushed);
            a.early = false;
        }
        if a.rows >= a.next {
            #[expect(clippy::cast_precision_loss, reason = "row counts in a ratio")]
            let rate = (a.pushed - a.window.1) as f64 / (a.rows - a.window.0).max(1) as f64;
            a.early = !grew && rate >= EARLY_RATE;
            a.window = (a.rows, a.pushed);
            a.next = a.rows + ADAPT_ROWS;
        }
    }

    /// Rebuilds a grown fixed index over the table's rows, up to `limit`
    /// keys, one entry per key: the last partition's rows first, each
    /// partition's newest first (so in a one-partition table, the newest
    /// keys are the ones kept). Without it, every key whose
    /// entry an earlier clear dropped would make one more partial duplicate
    /// when it next came (on TPC-DS q22 about as many rows again as the
    /// table's keys). DuckDB clears a table it grows: its rows are already
    /// spread over partitions there, ours are still in the table.
    fn reindex(&mut self, limit: usize) {
        let Self {
            parts,
            index,
            stride,
            gseed,
            refs,
            views,
            resizes,
            ..
        } = self;
        let (stride, gseed, refs, views) = (*stride, *gseed, *refs, views.as_ref());
        let parts: &[Part] = parts;
        let before = index.capacity();
        index.clear();
        let groups = parts.iter().map(|p| p.rows.len() / stride).sum::<usize>();
        index.reserve(groups.min(limit), |&o| {
            fin(word(row_at(parts, stride, refs, o), 0), gseed)
        });
        if index.capacity() != before
            && let Some(c) = resizes
        {
            c.fetch_add(1, Ordering::Relaxed);
        }
        'parts: for (p, part) in parts.iter().enumerate().rev() {
            for row in (0..part.rows.len() / stride).rev() {
                if index.len() >= limit {
                    break 'parts;
                }
                let r = refs.make(p, row);
                let (key, heap) = row_heap_at(parts, stride, refs, r);
                let g = fin(word(key, 0), gseed);
                let found = index.find(g, |&o| match views {
                    Some(l) => {
                        let (other, other_heap) = row_heap_at(parts, stride, refs, o);
                        !other.is_empty() && l.eq(other, other_heap, key, heap)
                    }
                    None => same_key(row_at(parts, stride, refs, o), key, stride),
                });
                if found.is_none() {
                    index.insert_unique(g, r, |&o| {
                        fin(word(row_at(parts, stride, refs, o), 0), gseed)
                    });
                }
            }
        }
    }

    /// Folds a prepared batch in. Call after reserving
    /// [`projected`](Self::projected) bytes.
    pub(crate) fn apply(&mut self, p: &RowPrepared) -> Result<()> {
        self.only_hashbrown()?;
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
            Fold::Raw => paired::update_all(&mut part.accs, &core.pairs, ids, cols, w),
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

    /// Moves partitions `ranges` out into tables of their own (in order):
    /// each holds its range's rows and states, no index or sketch,
    /// and takes no more rows (append mode, never merged into). This table
    /// keeps the partitions no range names, and drops its index.
    pub(crate) fn split_parts(&mut self, ranges: &[Range<usize>]) -> Vec<Self> {
        self.index = HashTable::new();
        self.merge_index = None;
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
                    merge_index: None,
                    mode: Mode::Append,
                    adapt: None,
                    groups,
                    rows_in: 0,
                    unique: self.unique,
                    abandons: None,
                    resizes: None,
                    probes: None,
                    local_skip: 0,
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
    /// estimate). One over [`FLAT_MAX_BYTES`] is a [`MergeIndex`].
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if a [`MergeIndex`] cannot be allocated.
    pub(crate) fn presize(&mut self, groups: usize) -> Result<()> {
        if Self::salted_merge(groups) {
            self.merge_index = Some(MergeIndex::new(groups)?);
            return Ok(());
        }
        let (parts, stride, gseed, refs) = (&self.parts, self.stride, self.gseed, self.refs);
        let before = self.index.capacity();
        self.index.reserve(groups, |&r| {
            fin(word(row_at(parts, stride, refs, r), 0), gseed)
        });
        self.count_resize(before);
        Ok(())
    }

    /// Whether a merge table of `groups` keys takes a [`MergeIndex`]: the
    /// join's size rule, [`FLAT_MAX_BYTES`].
    fn salted_merge(groups: usize) -> bool {
        buckets_for(groups).saturating_mul(BUCKET_BYTES) > FLAT_MAX_BYTES
    }

    /// [`memory_size`](Self::memory_size) after [`presize`](Self::presize).
    pub(crate) fn projected_presize(&self, groups: usize) -> usize {
        if Self::salted_merge(groups) {
            return self.memory_size() - buckets(&self.index) * BUCKET_BYTES
                + MergeIndex::bytes_for(groups);
        }
        self.memory_size() - buckets(&self.index) * BUCKET_BYTES
            + self.index_bytes(Mode::Grow, groups)
    }
}

#[cfg(test)]
mod tests;

mod merge;

mod probe;
