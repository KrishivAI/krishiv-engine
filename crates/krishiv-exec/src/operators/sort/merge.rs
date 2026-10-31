// SPDX-License-Identifier: Apache-2.0
//! Sorted runs and their k-way merge.
//!
//! A run is a sorted stream of rows: a buffered input batch with its sorted
//! row order (sorted when it was sunk), or a spill file whose batches are in
//! order. The [`Merger`] keeps a loser tree of run cursors keyed by
//! `(key row, run index)` (one comparison per level and output row), so rows that compare equal come out in run order:
//! the merge is stable when runs are numbered in input order.

use std::collections::VecDeque;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, RecordBatchOptions};
use arrow_row::{Row, Rows};
use arrow_schema::SchemaRef;
use arrow_select::interleave::interleave;
use krishiv_memory::{
    ConsumerHandle, MemoryError, Reservation, SpillError, SpillFile, SpillReader, SpillScope,
};
use krishiv_types::batch::{Batch, Weights};

use super::keys::KeyEncoder;
use crate::context::QueryContext;
use crate::error::{Result, internal};
use crate::hash::HashSeed;

/// A sunk batch with its key rows and its sorted row order (ties in row
/// order), grouped by bucket when the sort has buckets ([`Bucketing`]):
/// `order[bounds[b]..bounds[b + 1]]` are bucket `b`'s rows, sorted. Shared,
/// so a spill can merge it while the sink still owns it.
#[derive(Debug)]
pub(crate) struct SortedBatch {
    pub(crate) batch: Batch,
    pub(crate) rows: Rows,
    pub(crate) order: Vec<u32>,
    /// Where each bucket's rows start in `order`, and the end: one bucket
    /// (`[0, rows]`) without buckets.
    pub(crate) bounds: Vec<u32>,
}

/// A sort's split of its rows into `2^bits` buckets by the hash of their
/// partitioning keys (a prefix of the sort keys): the top bits of the
/// seedless hash `h` of the keys' normalized rows. Rows with equal
/// partitioning keys share a bucket, so each bucket sorts and merges on
/// its own, and in parallel.
#[derive(Debug)]
pub(crate) struct Bucketing {
    keys: Arc<KeyEncoder>,
    bits: u32,
}

/// The seed of the bucket hash: none, so equal keys share a bucket in every
/// process (design P2-T171 §5.1).
const SEEDLESS: u64 = 0;

impl Bucketing {
    /// Buckets by `keys` (encoded as the sort orders them), `2^bits` of
    /// them.
    pub(crate) fn new(keys: Arc<KeyEncoder>, bits: u32) -> Self {
        Self { keys, bits }
    }

    /// The number of buckets.
    pub(crate) fn buckets(&self) -> usize {
        1 << self.bits
    }

    /// Each row's bucket.
    fn of(&self, batch: &RecordBatch) -> Result<Vec<u32>> {
        let rows = self.keys.encode(batch)?;
        let h = HashSeed::new(Some(SEEDLESS));
        let shift = 64 - self.bits;
        Ok((0..rows.num_rows())
            .map(|i| {
                let bucket = h.hash(rows.row(i).data()).checked_shr(shift).unwrap_or(0);
                u32::try_from(bucket).unwrap_or(0)
            })
            .collect())
    }
}

impl SortedBatch {
    /// Encodes and sorts `batch`, bucket by bucket with `buckets`.
    pub(crate) fn new(
        batch: Batch,
        keys: &KeyEncoder,
        buckets: Option<&Bucketing>,
    ) -> Result<Self> {
        let rows = keys.encode(batch.data())?;
        let n = u32::try_from(batch.num_rows())
            .map_err(|_| internal("a batch has more than u32::MAX rows".to_owned()))?;
        let mut keyed: Vec<(u64, u32)> = Vec::with_capacity(n as usize);
        let bounds = match buckets.filter(|b| b.bits > 0) {
            None => {
                keyed.extend((0..n).map(|i| (prefix(rows.row(i as usize).data()), i)));
                vec![0, n]
            }
            Some(b) => {
                // A counting sort by bucket, then each bucket by key.
                let of = b.of(batch.data())?;
                let mut bounds = vec![0_u32; b.buckets() + 1];
                for &x in &of {
                    bounds[x as usize + 1] += 1;
                }
                for i in 1..bounds.len() {
                    bounds[i] += bounds[i - 1];
                }
                let mut at = bounds.clone();
                keyed.resize(n as usize, (0, 0));
                for (i, &x) in (0..n).zip(&of) {
                    let slot = &mut at[x as usize];
                    keyed[*slot as usize] = (prefix(rows.row(i as usize).data()), i);
                    *slot += 1;
                }
                bounds
            }
        };
        // Unstable sort plus the row index as tie-break: a stable order
        // without the stable sort's scratch buffer. Most comparisons end at
        // the 8-byte key prefix.
        for w in bounds.windows(2) {
            keyed[w[0] as usize..w[1] as usize].sort_unstable_by(|&(pa, a), &(pb, b)| {
                pa.cmp(&pb)
                    .then_with(|| rows.row(a as usize).cmp(&rows.row(b as usize)))
                    .then(a.cmp(&b))
            });
        }
        let order = keyed.into_iter().map(|(_, i)| i).collect();
        Ok(Self {
            batch,
            rows,
            order,
            bounds,
        })
    }

    /// The positions in `order` of bucket `b`'s rows.
    pub(crate) fn bucket(&self, b: usize) -> std::ops::Range<usize> {
        match (self.bounds.get(b), self.bounds.get(b + 1)) {
            (Some(&lo), Some(&hi)) => lo as usize..hi as usize,
            _ => 0..0,
        }
    }

    /// Bytes this batch holds: data, key rows and order.
    pub(crate) fn bytes(&self) -> usize {
        self.batch.data().get_array_memory_size()
            + self.batch.num_rows() * (size_of::<u32>() + weight_bytes(&self.batch))
            + self.rows.size()
            + self.bounds.len() * size_of::<u32>()
    }
}

/// The first 16 bytes of a key row, big-endian (zero-padded), as
/// [`prefix`]: a sort merge compares many rows of one partition, whose
/// first 8 bytes are mostly the partitioning key.
fn wide_prefix(row: &[u8]) -> u128 {
    let mut bytes = [0u8; 16];
    let n = row.len().min(16);
    bytes[..n].copy_from_slice(&row[..n]);
    u128::from_be_bytes(bytes)
}

/// The first 8 bytes of a key row, big-endian (zero-padded): comparing
/// prefixes orders rows as their bytes do, up to prefix ties.
fn prefix(row: &[u8]) -> u64 {
    let mut bytes = [0u8; 8];
    let n = row.len().min(8);
    bytes[..n].copy_from_slice(&row[..n]);
    u64::from_be_bytes(bytes)
}

/// The bytes of `batch`'s rows (what a copy of them, or a spill frame,
/// holds), without the slack of the buffers it shares: a per-row size that
/// does not depend on the batch's length.
pub(crate) fn data_bytes(batch: &Batch) -> usize {
    let columns: usize = batch
        .data()
        .columns()
        .iter()
        .map(|c| {
            c.to_data()
                .get_slice_memory_size()
                .unwrap_or_else(|_| c.get_array_memory_size())
        })
        .sum();
    columns + batch.num_rows() * weight_bytes(batch)
}

fn weight_bytes(batch: &Batch) -> usize {
    match batch.weights() {
        Weights::Const(_) => 0,
        Weights::Column(_) => size_of::<i64>(),
    }
}

/// A spilled sorted run.
#[derive(Debug)]
pub(crate) struct SpillRun {
    pub(crate) file: SpillFile,
    /// The largest batch written, in memory bytes: what reading it costs.
    pub(crate) max_batch_bytes: usize,
    /// The most memory bytes per row of a written batch.
    pub(crate) row_bytes: usize,
}

impl SpillRun {
    /// Memory a merge needs per open run: the reader's frame (encoded and
    /// decoded batch) and the batch's key rows.
    pub(crate) fn merge_bytes(&self) -> usize {
        self.max_batch_bytes.saturating_mul(3)
    }
}

/// Where a cursor's rows come from.
enum Input {
    Memory(Arc<SortedBatch>),
    Spill(Box<SpillInput>),
}

/// A spilled run being read: its current batch and key rows.
struct SpillInput {
    reader: SpillReader,
    batch: Batch,
    rows: Rows,
    /// Accounts `rows`.
    keys: Reservation,
}

/// The current position in one run.
struct Cursor {
    input: Input,
    pos: usize,
    len: usize,
    /// Index of the cursor's batch in [`Merger::sources`].
    source: usize,
    /// The current key's [`prefix`].
    prefix: u64,
}

impl Cursor {
    fn row_index(&self) -> usize {
        match &self.input {
            Input::Memory(s) => s.order.get(self.pos).map_or(0, |&i| i as usize),
            Input::Spill(_) => self.pos,
        }
    }

    fn key(&self) -> Row<'_> {
        let i = self.row_index();
        match &self.input {
            Input::Memory(s) => s.rows.row(i),
            Input::Spill(s) => s.rows.row(i),
        }
    }

    /// Caches the current key's prefix (after a move).
    fn refresh(&mut self) {
        if self.pos < self.len {
            self.prefix = prefix(self.key().data());
        }
    }

    fn batch(&self) -> &Batch {
        match &self.input {
            Input::Memory(s) => &s.batch,
            Input::Spill(s) => &s.batch,
        }
    }
}

/// Reads the next non-empty batch of `reader` with its key rows, accounted
/// in `keys`.
fn next_batch(
    reader: &mut SpillReader,
    encoder: &KeyEncoder,
    keys: &mut Reservation,
) -> Result<Option<(Batch, Rows)>> {
    for batch in reader.by_ref() {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let rows = encoder.encode(batch.data())?;
        keys.shrink(keys.size());
        keys.try_grow(rows.size())?;
        return Ok(Some((batch, rows)));
    }
    Ok(None)
}

/// Persistent per-source metadata a [`Merger`] holds: the output `slot` and
/// generation `stamp` that number an output's sources in one pass. Reserved
/// in the merge's budget (see `merge_budget`), not per output.
pub(crate) fn metadata_bytes(sources: usize) -> usize {
    sources.saturating_mul(size_of::<u64>().saturating_add(size_of::<u32>()))
}

/// Bytes per row of a sort merge's keyed rows (16-byte prefix, slice, row).
const KEYED_ROW_BYTES: usize = size_of::<(u128, u32, u32)>();
/// Bytes per row of a sort merge's output order.
const ORDER_ROW_BYTES: usize = size_of::<(u32, u32)>();

/// A k-way merge of sorted runs.
pub(crate) struct Merger {
    schema: SchemaRef,
    encoder: Arc<KeyEncoder>,
    cursors: Vec<Cursor>,
    /// Loser tree over the cursors by `(key, cursor index)`: `tree[0]` is
    /// the winner, inner node `n` the loser of its match.
    tree: Vec<usize>,
    /// The batches the pending output rows point into.
    sources: Vec<Batch>,
    /// Whether a cursor moved to a new batch since the last output, so
    /// `sources` holds batches no cursor is on any more.
    retired: bool,
    /// Per source: the output slot it was given in the batch being built
    /// (valid while its `stamp` equals `generation`): the sources an output
    /// uses are numbered in one pass, without sorting them.
    slot: Vec<u32>,
    /// A 64-bit count of output batches: it never wraps.
    stamp: Vec<u64>,
    generation: u64,
    /// A merge of many in-memory slices done as one sort of their rows
    /// (see [`Merger::bucket`]): no cursors, the rows in output order.
    sorted: Option<SortedRows>,
}

/// The rows of a sorted merge, in output order, and the memory they hold.
struct SortedRows {
    /// `(source, row)` in output order.
    #[expect(
        clippy::rc_buffer,
        reason = "`Arc<[_]>` would copy a large order once more"
    )]
    order: Arc<Vec<(u32, u32)>>,
    /// The next row of `order` to emit or gather.
    pos: usize,
    /// What [`GatherJob`]s read.
    sources: Arc<[Batch]>,
    /// Output batches gathered ahead of `pos`, in order.
    ready: VecDeque<Batch>,
    /// Output batches to gather at once: 1 gathers on the reader's step.
    gather: usize,
    /// Accounts `order` (and, while sorting, its keyed form).
    _scratch: Reservation,
}

/// In-memory slices above which a bucket's merge sorts its rows instead of
/// running a loser tree: with many small slices each tree step is a chain
/// of cache misses, while one sort of the rows' key prefixes stays in
/// cache (most comparisons end at a 16-byte prefix). An unbucketed sort has
/// one reader, so above this many batches it sorts on several threads
/// ([`parallel_merge`]) when they and the memory allow, else runs the tree.
const SORT_MIN_SLICES: usize = 64;

/// Fewest rows worth a part of a parallel merge of its own.
const PARALLEL_MIN_PART_ROWS: usize = 1 << 14;

/// Samples per part to place a parallel merge's cuts: a part's size is off by
/// about `1 / sqrt(SAMPLES_PER_PART)` of itself.
const SAMPLES_PER_PART: usize = 128;

/// `2^64 / phi`: multiples of it spread evenly over the `u64` range.
const GOLDEN_RATIO: u64 = 0x9E37_79B9_7F4A_7C15;

/// A range of a sorted batch's `order`: one input of a merge.
pub(super) type Slice = (Arc<SortedBatch>, std::ops::Range<usize>);

/// `(key row, slice, row)`, the total order of a merge: ties in a slice keep
/// its order (its own ties are in row order), and slices keep their run
/// order.
fn merge_cmp(memory: &[Slice], (sa, ra): (u32, u32), (sb, rb): (u32, u32)) -> std::cmp::Ordering {
    let a = memory[sa as usize].0.rows.row(ra as usize);
    let b = memory[sb as usize].0.rows.row(rb as usize);
    a.cmp(&b).then((sa, ra).cmp(&(sb, rb)))
}

/// The leading bytes all the rows of `memory` share: those of its smallest and
/// largest row, which each slice's ends give. A part of a merge has a narrow
/// key range, so its 16-byte prefixes do more past these bytes.
fn shared_prefix_len(memory: &[Slice]) -> usize {
    let mut ends = memory
        .iter()
        .filter(|(_, r)| !r.is_empty())
        .flat_map(|(s, r)| [r.start, r.end - 1].map(|at| s.rows.row(s.order[at] as usize)));
    let Some(first) = ends.next() else {
        return 0;
    };
    let (lo, hi) = ends.fold((first, first), |(lo, hi), row| (lo.min(row), hi.max(row)));
    let (lo, hi) = (lo.data(), hi.data());
    lo.iter().zip(hi).take_while(|(a, b)| a == b).count()
}

/// The rows of `memory`, as `(index into memory, row)`, in the merge's order.
pub(super) fn sort_slices(memory: &[Slice]) -> Result<Vec<(u32, u32)>> {
    let too_many = || internal("a sort merge of more than u32::MAX rows".to_owned());
    let rows: usize = memory.iter().map(|(_, r)| r.len()).sum();
    let shared = shared_prefix_len(memory);
    let mut keyed: Vec<(u128, u32, u32)> = Vec::with_capacity(rows);
    for (i, (s, range)) in memory.iter().enumerate() {
        let i = u32::try_from(i).map_err(|_| too_many())?;
        keyed.extend(
            s.order[range.clone()]
                .iter()
                .map(|&r| (wide_prefix(&s.rows.row(r as usize).data()[shared..]), i, r)),
        );
    }
    keyed.sort_unstable_by(|&(pa, sa, ra), &(pb, sb, rb)| {
        pa.cmp(&pb)
            .then_with(|| merge_cmp(memory, (sa, ra), (sb, rb)))
    });
    Ok(keyed.into_iter().map(|(_, s, r)| (s, r)).collect())
}

/// The most rows one part sort holds, about: a sort cannot stop part way, so
/// this bounds how long a cancelled query waits for it.
const SORT_PART_ROWS: usize = 1 << 20;

/// The parts to sort `rows` in with `parallelism` threads: one per thread,
/// or more when those would pass [`SORT_PART_ROWS`].
pub(super) fn sort_parts(rows: usize, parallelism: usize) -> usize {
    parallelism.max(rows.div_ceil(SORT_PART_ROWS))
}

/// A wide in-memory merge to run in parallel.
pub(super) struct ParallelMerge {
    pub(super) memory: Vec<Slice>,
    /// The threads to use, and the output batches to gather at once.
    pub(super) parts: usize,
    /// The parts to sort, at most `parts` of them at a time.
    pub(super) sort_parts: usize,
    /// Holds the keyed rows, the order and the output batches gathered ahead.
    pub(super) scratch: Reservation,
    /// Of `scratch`, the bytes of the output batches gathered ahead.
    pub(super) ready_bytes: usize,
}

/// The parallel form of a wide in-memory merge, with its memory: `None` when
/// there are few slices or rows for `parallelism` to pay, or no memory for
/// the keyed rows and the output gathered ahead (the loser tree needs
/// neither). The parts it sorts come from [`merge_splitters`] and [`merge_part`].
pub(super) fn parallel_merge(
    batches: &[Arc<SortedBatch>],
    parallelism: usize,
    batch_rows: usize,
    handle: &Arc<ConsumerHandle>,
) -> Option<ParallelMerge> {
    let unbucketed = batches.iter().all(|s| s.bounds.len() == 2);
    if !unbucketed || batches.len() <= SORT_MIN_SLICES {
        return None;
    }
    let memory: Vec<Slice> = batches
        .iter()
        .map(|s| (Arc::clone(s), 0..s.order.len()))
        .filter(|(_, r)| !r.is_empty())
        .collect();
    let rows: usize = memory.iter().map(|(_, r)| r.len()).sum();
    let parts = parallelism.min(rows / PARALLEL_MIN_PART_ROWS);
    if parts < 2 {
        return None;
    }
    let data: usize = memory.iter().map(|(s, _)| data_bytes(&s.batch)).sum();
    // About: by the rows' mean width, like the headroom of an output batch.
    let ready_bytes = data
        .div_ceil(rows)
        .saturating_mul(batch_rows.saturating_mul(parts).min(rows));
    let mut scratch = handle.reservation();
    scratch
        .try_grow(
            rows.saturating_mul(KEYED_ROW_BYTES + ORDER_ROW_BYTES)
                .saturating_add(ready_bytes),
        )
        .ok()?;
    Some(ParallelMerge {
        memory,
        parts,
        sort_parts: sort_parts(rows, parts),
        scratch,
        ready_bytes,
    })
}

/// One output batch of a sorted merge, to gather on a thread of its own.
pub(super) struct GatherJob {
    schema: SchemaRef,
    sources: Arc<[Batch]>,
    #[expect(
        clippy::rc_buffer,
        reason = "`Arc<[_]>` would copy a large order once more"
    )]
    order: Arc<Vec<(u32, u32)>>,
    range: std::ops::Range<usize>,
}

impl GatherJob {
    /// The output batch, reading only the sources it uses.
    pub(super) fn run(&self) -> Result<Batch> {
        let mut slot = vec![u32::MAX; self.sources.len()];
        let mut used: Vec<&Batch> = Vec::new();
        let mut indices = Vec::with_capacity(self.range.len());
        for &(s, r) in &self.order[self.range.clone()] {
            let at = &mut slot[s as usize];
            if *at == u32::MAX {
                *at = u32::try_from(used.len())
                    .map_err(|_| internal("too many merge sources".to_owned()))?;
                used.push(&self.sources[s as usize]);
            }
            indices.push((*at as usize, r as usize));
        }
        interleave_batches(&self.schema, &used, &indices)
    }
}

/// The rows between `parts` parts of `memory`: `(index into memory, row)`,
/// at most `parts - 1` of them, ascending in the merge's order, which is total,
/// so cuts fall between rows of equal keys too. They come from about
/// [`SAMPLES_PER_PART`] samples per part, placed across the slices so that many
/// slices with one sample each still see their whole key range, and weighted
/// by the rows each stands for.
pub(super) fn merge_splitters(memory: &[Slice], parts: usize) -> Result<Vec<(u32, u32)>> {
    let too_many = || internal("a sort merge of more than u32::MAX slices".to_owned());
    let per_slice = parts
        .saturating_mul(SAMPLES_PER_PART)
        .div_ceil(memory.len().max(1));
    let mut samples: Vec<((u32, u32), usize)> = Vec::new();
    for (i, (s, range)) in memory.iter().enumerate() {
        let index = u32::try_from(i).map_err(|_| too_many())?;
        let taken = per_slice.min(range.len());
        samples.extend((0..taken).map(|j| {
            // A fraction of the slice from the golden-ratio sequence, so
            // that slices with one sample each cover the whole key range.
            let spot = u128::from(((i + j) as u64).wrapping_mul(GOLDEN_RATIO));
            let at = (spot * range.len() as u128) >> 64;
            let at = range.start + usize::try_from(at).unwrap_or(0);
            ((index, s.order[at]), range.len() / taken)
        }));
    }
    samples.sort_unstable_by(|a, b| merge_cmp(memory, a.0, b.0));
    let total: usize = samples.iter().map(|&(_, w)| w).sum();
    let mut splitters = Vec::with_capacity(parts.saturating_sub(1));
    let mut seen = 0;
    for &(sample, weight) in &samples {
        seen += weight;
        if seen.saturating_mul(parts) >= total.saturating_mul(splitters.len() + 1)
            && splitters.len() + 1 < parts
        {
            splitters.push(sample);
        }
    }
    Ok(splitters)
}

/// The rows of `memory` from splitter `lo` (none: the first row) up to
/// splitter `hi` (none: past the last row), as slices of the same indices:
/// sorting the parts between consecutive splitters one by one and joining the
/// results is the whole merge.
pub(super) fn merge_part(
    memory: &[Slice],
    lo: Option<(u32, u32)>,
    hi: Option<(u32, u32)>,
) -> Result<Vec<Slice>> {
    let too_many = || internal("a sort merge of more than u32::MAX slices".to_owned());
    let mut part = Vec::with_capacity(memory.len());
    for (i, (s, range)) in memory.iter().enumerate() {
        let i = u32::try_from(i).map_err(|_| too_many())?;
        let at = |start: usize, splitter: Option<(u32, u32)>| match splitter {
            Some(to) => {
                start
                    + s.order[start..range.end].partition_point(|&r| {
                        merge_cmp(memory, (i, r), to) == std::cmp::Ordering::Less
                    })
            }
            None => range.end,
        };
        let start = if lo.is_some() {
            at(range.start, lo)
        } else {
            range.start
        };
        part.push((Arc::clone(s), start..at(start, hi)));
    }
    Ok(part)
}

impl Merger {
    /// A merge of `memory` runs followed by `spilled` runs (in that run
    /// order). Opening a spill file is blocking I/O: call it from a
    /// blocking closure when `spilled` is not empty. Readers and key rows
    /// are accounted in reservations of `handle`.
    pub(crate) fn new(
        schema: SchemaRef,
        encoder: Arc<KeyEncoder>,
        memory: Vec<Arc<SortedBatch>>,
        spilled: &[SpillRun],
        handle: &Arc<ConsumerHandle>,
    ) -> Result<Self> {
        let memory = memory
            .into_iter()
            .map(|s| {
                let all = 0..s.order.len();
                (s, all)
            })
            .collect();
        Self::slices(schema, encoder, memory, spilled, handle)
    }

    /// A merge of bucket `bucket` of each of `memory`'s batches: with hash
    /// buckets and many slices a sort of the rows, else the loser tree of
    /// [`Merger::new`].
    pub(crate) fn bucket(
        schema: SchemaRef,
        encoder: Arc<KeyEncoder>,
        memory: &[Arc<SortedBatch>],
        bucket: usize,
        handle: &Arc<ConsumerHandle>,
    ) -> Result<Self> {
        let memory: Vec<_> = memory
            .iter()
            .map(|s| (Arc::clone(s), s.bucket(bucket)))
            .filter(|(_, r)| !r.is_empty())
            .collect();
        let bucketed = memory.iter().any(|(s, _)| s.bounds.len() > 2);
        if bucketed && memory.len() > SORT_MIN_SLICES {
            let rows: usize = memory.iter().map(|(_, r)| r.len()).sum();
            let mut scratch = handle.reservation();
            // Without the memory for the keyed rows, the loser tree (which
            // needs none) merges them.
            if scratch
                .try_grow(rows.saturating_mul(KEYED_ROW_BYTES + ORDER_ROW_BYTES))
                .is_ok()
            {
                return Self::sorted(schema, encoder, &memory, scratch);
            }
        }
        Self::slices(schema, encoder, memory, &[], handle)
    }

    /// The merge of `memory`'s slices as one stable sort of their rows by
    /// `(key, slice, position)`, `scratch` holding its memory.
    fn sorted(
        schema: SchemaRef,
        encoder: Arc<KeyEncoder>,
        memory: &[Slice],
        scratch: Reservation,
    ) -> Result<Self> {
        let order = sort_slices(memory)?;
        Ok(Self::from_order(
            schema,
            encoder,
            memory,
            order,
            (scratch, 0),
            1,
        ))
    }

    /// A merge whose rows are already in output `order` (`(index into
    /// memory, row)`), gathering `gather` output batches at once. `scratch`
    /// holds its memory, of which it keeps the order's and the given bytes.
    pub(super) fn from_order(
        schema: SchemaRef,
        encoder: Arc<KeyEncoder>,
        memory: &[Slice],
        order: Vec<(u32, u32)>,
        (mut scratch, ready_bytes): (Reservation, usize),
        gather: usize,
    ) -> Self {
        let keep = (order.len() * ORDER_ROW_BYTES).saturating_add(ready_bytes);
        scratch.shrink(scratch.size().saturating_sub(keep));
        let sources: Vec<Batch> = memory.iter().map(|(s, _)| s.batch.clone()).collect();
        let shared: Arc<[Batch]> = sources.as_slice().into();
        Self {
            schema,
            encoder,
            cursors: Vec::new(),
            tree: Vec::new(),
            sources,
            retired: false,
            slot: Vec::new(),
            stamp: Vec::new(),
            generation: 0,
            sorted: Some(SortedRows {
                order: Arc::new(order),
                pos: 0,
                sources: shared,
                ready: VecDeque::new(),
                gather,
                _scratch: scratch,
            }),
        }
    }

    /// Whether this merge sorts its rows instead of running the loser tree.
    #[cfg(test)]
    pub(super) fn sorts_rows(&self) -> bool {
        self.sorted.is_some()
    }

    /// The next output batches of a sorted merge that gathers several at
    /// once, none while some are still to read: run them, in any order, and
    /// hand the results to [`Merger::gathered`] in the jobs' order.
    pub(super) fn gather_jobs(&mut self, rows: usize) -> Vec<GatherJob> {
        let Some(s) = self.sorted.as_mut().filter(|s| s.gather > 1) else {
            return Vec::new();
        };
        if !s.ready.is_empty() {
            return Vec::new();
        }
        let mut jobs = Vec::with_capacity(s.gather);
        while jobs.len() < s.gather && s.pos < s.order.len() {
            let end = s.order.len().min(s.pos + rows.max(1));
            jobs.push(GatherJob {
                schema: Arc::clone(&self.schema),
                sources: Arc::clone(&s.sources),
                order: Arc::clone(&s.order),
                range: s.pos..end,
            });
            s.pos = end;
        }
        jobs
    }

    /// Takes the results of [`Merger::gather_jobs`].
    pub(super) fn gathered(&mut self, batches: Vec<Batch>) {
        if let Some(s) = &mut self.sorted {
            s.ready.extend(batches);
        }
    }

    /// The next batch gathered ahead, if any.
    pub(super) fn pop_ready(&mut self) -> Option<Batch> {
        self.sorted.as_mut()?.ready.pop_front()
    }

    /// A merge of the `order` ranges of `memory` followed by `spilled`
    /// runs.
    pub(super) fn slices(
        schema: SchemaRef,
        encoder: Arc<KeyEncoder>,
        memory: Vec<Slice>,
        spilled: &[SpillRun],
        handle: &Arc<ConsumerHandle>,
    ) -> Result<Self> {
        let mut cursors = Vec::with_capacity(memory.len() + spilled.len());
        for (s, range) in memory {
            cursors.push(Cursor {
                input: Input::Memory(s),
                pos: range.start,
                len: range.end,
                source: cursors.len(),
                prefix: 0,
            });
        }
        for run in spilled {
            // Frames and key rows come and go: what they free stays with
            // the consumer as credit for the next ones (see `Sort`).
            let mut reader = run
                .file
                .reader(handle.reservation().releasing_to_credit())?;
            // A refusal here fails the merge (no spill to fall back on).
            let mut keys = handle.reservation().releasing_to_credit().unguarded();
            if let Some((batch, rows)) = next_batch(&mut reader, &encoder, &mut keys)? {
                cursors.push(Cursor {
                    pos: 0,
                    len: batch.num_rows(),
                    source: cursors.len(),
                    prefix: 0,
                    input: Input::Spill(Box::new(SpillInput {
                        reader,
                        batch,
                        rows,
                        keys,
                    })),
                });
            }
        }
        cursors.iter_mut().for_each(Cursor::refresh);
        let sources = cursors.iter().map(|c| c.batch().clone()).collect();
        let mut merger = Self {
            schema,
            encoder,
            tree: vec![usize::MAX; cursors.len()],
            cursors,
            sources,
            retired: false,
            slot: Vec::new(),
            stamp: Vec::new(),
            generation: 0,
            sorted: None,
        };
        for i in 0..merger.cursors.len() {
            // Plays `i` up until it meets an empty node (initial build).
            let (mut node, mut winner) = ((merger.cursors.len() + i) / 2, i);
            while node != 0 && merger.tree[node] != usize::MAX {
                if merger.beats(merger.tree[node], winner) {
                    winner = std::mem::replace(&mut merger.tree[node], winner);
                }
                node /= 2;
            }
            merger.tree[node] = winner;
        }
        Ok(merger)
    }

    /// Whether cursor `a` comes before `b`; an exhausted cursor never does.
    fn beats(&self, a: usize, b: usize) -> bool {
        let (ca, cb) = (&self.cursors[a], &self.cursors[b]);
        match (ca.pos < ca.len, cb.pos < cb.len) {
            (true, true) => ca
                .prefix
                .cmp(&cb.prefix)
                .then_with(|| ca.key().cmp(&cb.key()))
                .then(a.cmp(&b))
                .is_lt(),
            (live, _) => live,
        }
    }

    /// Replays the winner's path after it moved.
    fn replay(&mut self) {
        let mut winner = self.tree[0];
        let mut node = (self.cursors.len() + winner) / 2;
        while node != 0 {
            if self.beats(self.tree[node], winner) {
                winner = std::mem::replace(&mut self.tree[node], winner);
            }
            node /= 2;
        }
        self.tree[0] = winner;
    }

    /// Moves cursor `c` past its current row; `false` once its run ends.
    fn advance(&mut self, c: usize) -> Result<bool> {
        let cursor = &mut self.cursors[c];
        cursor.pos += 1;
        if cursor.pos < cursor.len {
            cursor.refresh();
            return Ok(true);
        }
        let Input::Spill(s) = &mut cursor.input else {
            return Ok(false);
        };
        let Some((next, next_rows)) = next_batch(&mut s.reader, &self.encoder, &mut s.keys)? else {
            return Ok(false);
        };
        cursor.pos = 0;
        cursor.len = next.num_rows();
        cursor.source = self.sources.len();
        self.sources.push(next.clone());
        s.batch = next;
        s.rows = next_rows;
        cursor.refresh();
        self.retired = true;
        Ok(true)
    }

    /// The next merged batch of at most `max_rows` rows (at least one), or
    /// `None` once every run is exhausted. Reading a spilled run is blocking
    /// I/O.
    pub(crate) fn next(&mut self, max_rows: usize) -> Result<Option<Batch>> {
        if let Some(s) = &mut self.sorted {
            // Batches gathered ahead are the reader's size. A caller wanting
            // fewer rows (a spill's small batches) gathers them again.
            if s.ready
                .front()
                .is_some_and(|b| b.num_rows() > max_rows.max(1))
            {
                let rows: usize = s.ready.iter().map(Batch::num_rows).sum();
                s.pos -= rows;
                s.ready.clear();
            }
        }
        if let Some(batch) = self.pop_ready() {
            return Ok(Some(batch));
        }
        let mut out = Vec::with_capacity(max_rows.min(1 << 16));
        if let Some(sorted) = &mut self.sorted {
            let end = sorted.order.len().min(sorted.pos + max_rows.max(1));
            out.extend(
                sorted.order[sorted.pos..end]
                    .iter()
                    .map(|&(s, r)| (s as usize, r as usize)),
            );
            sorted.pos = end;
        }
        while self.sorted.is_none() && out.len() < max_rows.max(1) {
            let Some(&top) = self.tree.first() else {
                break;
            };
            let cursor = &self.cursors[top];
            if cursor.pos >= cursor.len {
                break;
            }
            out.push((cursor.source, cursor.row_index()));
            self.advance(top)?;
            self.replay();
        }
        if out.is_empty() {
            return Ok(None);
        }
        // Interleave only the batches this output uses: an in-memory merge
        // has one source per buffered batch, and passing them all would
        // make every output batch cost O(buffered batches). They are
        // numbered in order of first use and borrowed, not cloned.
        let n = self.sources.len();
        if self.stamp.len() < n {
            self.stamp.resize(n, 0);
            self.slot.resize(n, 0);
        }
        self.generation += 1;
        let mut used: Vec<&Batch> = Vec::new();
        for o in &mut out {
            let s = o.0;
            let (Some(stamp), Some(slot)) = (self.stamp.get_mut(s), self.slot.get_mut(s)) else {
                return Err(internal(format!("merge source {s} is not in use")));
            };
            if *stamp != self.generation {
                *stamp = self.generation;
                *slot = u32::try_from(used.len())
                    .map_err(|_| internal("too many merge sources".to_owned()))?;
                used.push(&self.sources[s]);
            }
            o.0 = *slot as usize;
        }
        let batch = interleave_batches(&self.schema, &used, &out)?;
        if self.retired {
            // Only the batches cursors are on stay referenced.
            self.sources = self.cursors.iter().map(|c| c.batch().clone()).collect();
            for (i, c) in self.cursors.iter_mut().enumerate() {
                c.source = i;
            }
            self.retired = false;
        }
        Ok(Some(batch))
    }
}

/// The rows at `indices` (`(source batch, row)`) of `sources`, weights
/// included: a constant weight shared by every source stays constant.
pub(crate) fn interleave_batches<B: std::borrow::Borrow<Batch>>(
    schema: &SchemaRef,
    sources: &[B],
    indices: &[(usize, usize)],
) -> Result<Batch> {
    let columns = (0..schema.fields().len())
        .map(|c| {
            let arrays: Vec<&dyn Array> = sources
                .iter()
                .map(|b| b.borrow().data().column(c).as_ref())
                .collect();
            interleave(&arrays, indices)
        })
        .collect::<Result<Vec<ArrayRef>, _>>()
        .map_err(|e| internal(e.to_string()))?;
    let options = RecordBatchOptions::new().with_row_count(Some(indices.len()));
    let data = RecordBatch::try_new_with_options(Arc::clone(schema), columns, &options)
        .map_err(|e| internal(e.to_string()))?;
    let first = sources.first().map(|b| b.borrow().weights());
    let weights = match first {
        Some(Weights::Const(w))
            if sources
                .iter()
                .all(|b| b.borrow().weights() == &Weights::Const(*w)) =>
        {
            Weights::Const(*w)
        }
        _ => Weights::Column(
            indices
                .iter()
                .map(
                    |&(b, r)| match sources.get(b).map(|b| b.borrow().weights()) {
                        Some(Weights::Const(w)) => Ok(*w),
                        Some(Weights::Column(c)) if r < c.len() => Ok(c.value(r)),
                        _ => Err(internal(format!("merge index ({b}, {r}) is out of range"))),
                    },
                )
                .collect::<Result<Int64Array>>()?,
        ),
    };
    Batch::with_weights(data, weights)
}

/// Merges `merger` into a new spill file, in batches of `batch_rows`,
/// checking cancellation between batches. The writer's scratch memory comes
/// from `scratch`. `Ok(Err(_))` when the memory pool refused the scratch
/// memory: nothing was spilled and the caller keeps its data.
pub(crate) fn write_run(
    merger: &mut Merger,
    scope: &SpillScope,
    scratch: Reservation,
    batch_rows: usize,
    query: &QueryContext,
) -> Result<Result<SpillRun, MemoryError>> {
    let refused = |e: SpillError| match e {
        SpillError::Memory(m) => Ok(Err(m)),
        other => Err(other.into()),
    };
    let mut writer = match scope.writer(Arc::clone(&merger.schema), scratch) {
        Ok(w) => w,
        Err(e) => return refused(e),
    };
    let (mut max_batch_bytes, mut row_bytes) = (0, 1);
    while let Some(batch) = merger.next(batch_rows)? {
        query.check_cancelled()?;
        let bytes = batch.data().get_array_memory_size();
        max_batch_bytes = max_batch_bytes.max(bytes);
        row_bytes = row_bytes.max(data_bytes(&batch).div_ceil(batch.num_rows().max(1)));
        if let Err(e) = writer.write(&batch) {
            return refused(e);
        }
    }
    match writer.finish() {
        Ok(file) => Ok(Ok(SpillRun {
            file,
            max_batch_bytes,
            row_bytes,
        })),
        Err(e) => refused(e),
    }
}
