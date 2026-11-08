// SPDX-License-Identifier: Apache-2.0
//! The `ORDER BY` breaker: buffer and sort per batch, spill sorted runs
//! under memory pressure, merge them in `finalize` down to a bounded fan-in,
//! and stream the final k-way merge from its source side.
//!
//! # Hash buckets
//! A sort below a window over a hash exchange (`Sort` over
//! `Exchange(Hash(keys))`, instantiated on one worker) splits its rows into
//! buckets by the hash of those keys ([`Sort::with_hash_buckets`], design
//! P2-T055 4a): each sorted batch is sorted bucket by bucket, a spill
//! writes one run per bucket, and the source merges each bucket on its own,
//! the buckets claimed by as many drivers as there are buckets. A bucket
//! holds whole partitions in sort order, which is all a window reads; the
//! order across buckets is not kept. Without hash keys, or with
//! `krishiv.exec.sort.hashBuckets` off, there is one bucket and one reader,
//! and the output is totally ordered. Readers of a spilled result merge in
//! parallel only with memory to spare (see `Readers`): at a tight limit
//! they need no more than one reader does.

use std::mem;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use arrow_schema::SchemaRef;
use futures::StreamExt as _;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_memory::{
    ConsumerHandle, MemoryError, Reservation, Seat, SeatState, SpillScope, SpillSignal, SpillWriter,
};
use krishiv_types::batch::Batch;

use super::budget::{
    MERGE_COHORT, SORTS_COHORT, headroom, holding, lending, lending_for_write, merge_budget,
    merge_need, spill_rows,
};
use super::keys::{KeyEncoder, SortKey};
use super::merge::{
    Bucketing, Merger, ParallelMerge, SortedBatch, SpillRun, data_bytes, interleave_batches,
    merge_part, merge_splitters, owned_rows, parallel_merge, sort_slices, write_run,
};
use super::{blocking, finalized};
use crate::context::{QueryContext, TaskContext};
use crate::error::{Result, cancelled, internal};
use crate::executor::{Executor, run_blocking};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus, Source};
use crate::operators::{global_ref, local_mut};
use crate::output_spill::{OutputSpill, SpillUnread, Unfinished, refused};
use crate::sync::{Event, Mutex, lock};
use crate::window::{GroupLimitPlan, RankLike, Ranker};

#[path = "held.rs"]
mod held;
use held::{Bucket, Held, HeldState, Take};

/// Default bound on the runs one merge reads at once.
pub const DEFAULT_MAX_MERGE_FAN_IN: usize = 64;

/// Fewest bucket bits of a bucketed sort (8 buckets).
const MIN_BUCKET_BITS: u32 = 3;
/// Most bucket bits (4096 buckets, the key groups of design P2-T171).
const MAX_BUCKET_BITS: u32 = 12;

/// Bucket bits for `drivers` drivers: at least four buckets per driver, so
/// that buckets of different sizes still keep every driver busy; none for
/// one driver.
pub(crate) fn bucket_bits(drivers: usize) -> u32 {
    if drivers <= 1 {
        return 0;
    }
    drivers
        .saturating_mul(4)
        .checked_next_power_of_two()
        .map_or(MAX_BUCKET_BITS, usize::trailing_zeros)
        .clamp(MIN_BUCKET_BITS, MAX_BUCKET_BITS)
}

/// Counters of one sort for `EXPLAIN ANALYZE`, next to the driver's
/// [`OperatorMetrics`](crate::OperatorMetrics) of its sink and source sides.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SortMetrics {
    /// Rows sunk (and sorted).
    pub rows_sorted: u64,
    /// Sorted runs written to spill files by spills (one per bucket).
    pub runs_spilled: u64,
    /// Bytes written to spill files, merge passes included.
    pub spilled_bytes: u64,
    /// Merge passes before the final merge: levels of intermediate merges,
    /// each rewriting up to the fan-in's runs into one.
    pub merge_passes: u64,
    /// Runs read at once by the final merge (of the widest bucket).
    pub final_fan_in: u64,
    /// Hash buckets the rows were split into (1: none).
    pub buckets: u64,
    /// Most spilled buckets merged at once by the readers (0: nothing
    /// spilled).
    pub merge_readers: u64,
    /// Rows a window group limit dropped from the sorted batches.
    pub rows_pruned: u64,
    /// Times a merge budget the pool refused waited for another sort's
    /// merge in the query to end (the merge cohort).
    pub merge_waits: u64,
}

#[derive(Debug, Default)]
struct Counters {
    rows: AtomicU64,
    runs: AtomicU64,
    bytes: AtomicU64,
    passes: AtomicU64,
    fan_in: AtomicU64,
    buckets: AtomicU64,
    readers: AtomicU64,
    pruned: AtomicU64,
    merge_waits: AtomicU64,
}

/// A window group limit applied to each sorted batch (the `Partial` mode of
/// [`crate::window::WindowGroupLimit`]): rows past the limit within their
/// batch are past it in their partition.
#[derive(Debug)]
struct Prune {
    /// The partitioning keys, a prefix of the sort keys (none: one
    /// partition).
    partition: Option<Arc<KeyEncoder>>,
    function: RankLike,
    limit: u64,
}

/// Sorted batches a driver looks at before keeping pruning only if it
/// drops at least a tenth of their rows.
const PRUNE_PROBE_BATCHES: u32 = 8;

/// One driver's pruning so far.
#[derive(Debug, Default)]
struct PruneStats {
    batches: u32,
    rows_in: u64,
    rows_out: u64,
    /// Too few rows dropped: pruning stopped.
    off: bool,
}

impl PruneStats {
    /// Counts a pruned batch the sink took: `rows` looked at (none: not
    /// pruned), `dropped` of them dropped.
    fn count(&mut self, rows: usize, dropped: usize) {
        if rows == 0 {
            return;
        }
        self.batches = self.batches.saturating_add(1);
        self.rows_in = self.rows_in.saturating_add(rows as u64);
        self.rows_out = self.rows_out.saturating_add((rows - dropped) as u64);
        if self.batches >= PRUNE_PROBE_BATCHES
            && self.rows_out.saturating_mul(10) > self.rows_in.saturating_mul(9)
        {
            self.off = true;
        }
    }
}

/// `ORDER BY` as a pipeline breaker: the sink side of one pipeline and the
/// source side of the next (one driver, so the order holds; one per bucket
/// with hash buckets).
///
/// Spark's order: NULLS FIRST for ASC and LAST for DESC by default, every
/// NaN equal and above all values, `-0.0 = 0.0`, strings by bytes. Stable
/// within a driver (ties keep input order); weights travel with their rows.
/// Spills sorted runs under memory pressure and merges them with a bounded
/// fan-in (see the `operators::sort` module source for the details).
#[derive(Debug)]
pub struct Sort {
    schema: SchemaRef,
    encoder: Arc<KeyEncoder>,
    spill: Option<SpillScope>,
    max_fan_in: usize,
    counters: Arc<Counters>,
    /// The keys to bucket by ([`Sort::with_hash_buckets`]).
    bucket_keys: Option<Arc<KeyEncoder>>,
    prune: Option<Prune>,
    /// Read bucket by bucket ([`Sort::bucket_reader`]), not claimed.
    read_by_bucket: bool,
}

impl Sort {
    /// Sorts batches of `schema` by `keys`. With a `spill` scope the sort
    /// spills sorted runs under memory pressure; without one it can only
    /// hold its input in memory. Without keys every row ties: the sort
    /// gathers its input into one stream (a window over all rows).
    ///
    /// # Errors
    /// As [`SortKey`] validation: `KRISHIV_UNSUPPORTED_FEATURE` for a key
    /// type the sort does not order yet, `INTERNAL_ERROR` for an
    /// out-of-range column.
    pub fn new(schema: SchemaRef, keys: &[SortKey], spill: Option<SpillScope>) -> Result<Self> {
        Ok(Self {
            encoder: KeyEncoder::new(&schema, keys)?,
            schema,
            spill,
            max_fan_in: DEFAULT_MAX_MERGE_FAN_IN,
            counters: Arc::default(),
            bucket_keys: None,
            prune: None,
            read_by_bucket: false,
        })
    }

    /// Keeps a spilled result's runs by bucket, for
    /// [`Sort::bucket_reader`] (a merge join reads the same bucket of two
    /// sorts); the source side's claims then find nothing spilled.
    #[must_use]
    pub(crate) fn read_by_bucket(mut self) -> Self {
        self.read_by_bucket = true;
        self
    }

    /// The buckets of a finalized sort's `global` state.
    pub(crate) fn bucket_count(global: &GlobalState) -> Result<usize> {
        Ok(global_ref::<SortGlobal>(global, "Sort")?.buckets)
    }

    /// The memory merging bucket `b` of a finalized sort takes up front (no
    /// more than its output batch when nothing spilled).
    pub(crate) fn bucket_need(global: &GlobalState, b: usize, batch_rows: usize) -> Result<usize> {
        let g = global_ref::<SortGlobal>(global, "Sort")?;
        let s = lock(&g.readers.state);
        Ok(s.by_bucket.get(b).map_or(0, |runs| {
            if runs.is_empty() {
                0
            } else {
                merge_need(runs, runs.len(), batch_rows, false)
            }
        }))
    }

    /// A reader of bucket `b` alone, for [`Source::get_data`]: no claim, and
    /// its merge waits for its memory with arbitration.
    pub(crate) fn bucket_reader(
        &self,
        env: &crate::join::MergeEnv,
        handle: &Arc<ConsumerHandle>,
        b: usize,
    ) -> LocalState {
        Box::new(SortSource {
            state: SourceState::Start,
            bucket: Some(b),
            query: Arc::clone(&env.query),
            executor: Arc::clone(&env.executor),
            handle: Arc::clone(handle),
            batch_rows: env.query.config().batch_size.max(1),
        })
    }

    /// Drops, from each sorted batch, the rows past `limit`'s rank within
    /// their partition (a window group limit's `Partial` mode); the sort's
    /// keys must be `limit`'s partitioning keys ascending, then its
    /// ordering. A driver stops pruning when its first batches lose less
    /// than a tenth of their rows.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when the sort's keys are not `limit`'s, as
    /// [`Sort::new`] for the partitioning keys.
    pub fn with_group_limit(mut self, keys: &[SortKey], limit: &GroupLimitPlan) -> Result<Self> {
        if !limit.sorted_by(keys) || limit.limit == 0 {
            return Err(internal(
                "a window group limit over a sort by other keys".to_owned(),
            ));
        }
        let partition = limit.partition_keys();
        self.prune = Some(Prune {
            partition: (!partition.is_empty())
                .then(|| KeyEncoder::new(&self.schema, &partition))
                .transpose()?,
            function: limit.function,
            limit: limit.limit,
        });
        Ok(self)
    }

    /// `sorted` without its rows past the group limit (see
    /// [`Sort::with_group_limit`]).
    /// Returns the batch and how many rows were looked at (0 without
    /// pruning) and dropped; the caller counts them once the batch is
    /// taken ([`PruneStats::count`]).
    fn prune(
        &self,
        sorted: SortedBatch,
        stats: &PruneStats,
    ) -> Result<(SortedBatch, usize, usize)> {
        let Some(p) = &self.prune else {
            return Ok((sorted, 0, 0));
        };
        if stats.off || !sorted.batch.is_unweighted() {
            return Ok((sorted, 0, 0));
        }
        let partition = p
            .partition
            .as_ref()
            .map(|e| e.encode(sorted.batch.data()))
            .transpose()?;
        let mut ranker = Ranker::new(p.function, p.limit);
        let peers = ranker.needs_peers();
        let mut kept: Vec<u32> = Vec::with_capacity(sorted.order.len());
        let mut bounds: Vec<u32> = Vec::with_capacity(sorted.bounds.len());
        bounds.push(0);
        // Bucket by bucket: rows of different buckets are of different
        // partitions.
        for w in sorted.bounds.windows(2) {
            let mut prev: Option<usize> = None;
            for &i in &sorted.order[w[0] as usize..w[1] as usize] {
                let at = i as usize;
                let (part_start, peer_start) = match prev {
                    None => (true, true),
                    Some(p) => (
                        partition.as_ref().is_some_and(|r| r.row(at) != r.row(p)),
                        // Equal full keys: equal partition and ordering.
                        peers && sorted.rows.row(at) != sorted.rows.row(p),
                    ),
                };
                if ranker.keep(part_start, peer_start) {
                    kept.push(i);
                }
                prev = Some(at);
            }
            // At most the batch's rows, which fit in `u32` (`order`).
            bounds.push(
                u32::try_from(kept.len())
                    .map_err(|_| internal("a batch has more than u32::MAX rows".to_owned()))?,
            );
        }
        let n = sorted.order.len();
        let dropped = n - kept.len();
        if dropped == 0 {
            return Ok((sorted, n, 0));
        }
        let indices: Vec<(usize, usize)> = kept.iter().map(|&i| (0, i as usize)).collect();
        let batch = interleave_batches(&self.schema, &[&sorted.batch], &indices)?;
        let bytes = kept
            .iter()
            .map(|&i| sorted.rows.row(i as usize).as_ref().len())
            .sum();
        let mut rows = self.encoder.empty_rows(kept.len(), bytes);
        for &i in &kept {
            rows.push(sorted.rows.row(i as usize));
        }
        let m = u32::try_from(kept.len())
            .map_err(|_| internal("a batch has more than u32::MAX rows".to_owned()))?;
        Ok((
            SortedBatch {
                batch,
                rows,
                order: (0..m).collect(),
                bounds,
            },
            n,
            dropped,
        ))
    }

    /// Bounds the runs one merge reads at once (at least 2); more runs are
    /// merged in several passes.
    #[must_use]
    pub fn with_max_fan_in(mut self, max_fan_in: usize) -> Self {
        self.max_fan_in = max_fan_in.max(2);
        self
    }

    /// Splits the rows into buckets by the hash of `columns` (a hash
    /// exchange's keys below the sort), which are sorted, spilled and
    /// merged on their own, the buckets read by several drivers (see the
    /// module docs): rows with equal `columns` come out together and in
    /// order, but the order across them is not kept. Whether a query
    /// buckets is `krishiv.exec.sort.hashBuckets`; one driver never does.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` without columns or for one out of range;
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a type the sort cannot order.
    pub fn with_hash_buckets(mut self, columns: &[usize]) -> Result<Self> {
        if columns.is_empty() {
            return Err(internal("hash buckets without keys".to_owned()));
        }
        let keys: Vec<SortKey> = columns.iter().map(|&c| SortKey::asc(c)).collect();
        self.bucket_keys = Some(KeyEncoder::new(&self.schema, &keys)?);
        Ok(self)
    }

    /// Whether the sort may split its rows into hash buckets (its output
    /// is then ordered within each bucket only).
    pub fn bucketed(&self) -> bool {
        self.bucket_keys.is_some()
    }

    /// The counters so far.
    pub fn metrics(&self) -> SortMetrics {
        let c = &self.counters;
        SortMetrics {
            rows_sorted: c.rows.load(Ordering::Relaxed),
            runs_spilled: c.runs.load(Ordering::Relaxed),
            spilled_bytes: c.bytes.load(Ordering::Relaxed),
            merge_passes: c.passes.load(Ordering::Relaxed),
            final_fan_in: c.fan_in.load(Ordering::Relaxed),
            buckets: c.buckets.load(Ordering::Relaxed),
            merge_readers: c.readers.load(Ordering::Relaxed),
            rows_pruned: c.pruned.load(Ordering::Relaxed),
            merge_waits: c.merge_waits.load(Ordering::Relaxed),
        }
    }
}

/// Buffered, per-batch-sorted input with its reservations.
#[derive(Debug)]
struct Buffer {
    batches: Vec<Arc<SortedBatch>>,
    /// Accounts `batches`.
    reservation: Reservation,
    /// Memory kept free for spilling this buffer (the merged output batch
    /// and the writer's scratch), lent to them as grant credit.
    headroom: Reservation,
    /// The most bytes per row seen, for the headroom.
    row_bytes: usize,
    handle: Arc<ConsumerHandle>,
}

impl Buffer {
    fn new(handle: &Arc<ConsumerHandle>) -> Self {
        Self {
            batches: Vec::new(),
            reservation: handle.reservation(),
            // Filled from credit just moved off the batch's reservation; a
            // refusal fails the sink, so the resident-set guard never cuts it.
            headroom: handle.reservation().unguarded(),
            row_bytes: 0,
            handle: Arc::clone(handle),
        }
    }
}

/// One spill of a buffer: a sorted run per bucket (`None` for a bucket
/// without rows).
type Runs = Vec<Option<SpillRun>>;

struct SortLocal {
    buffer: Buffer,
    runs: Vec<Runs>,
    /// This driver's spill signal, in the sort's list of drivers still
    /// sorting input.
    live: Arc<Live>,
    batch_rows: usize,
    bucketing: Option<Arc<Bucketing>>,
    prune: PruneStats,
    /// The sort's global state: a spill of this driver's also writes out
    /// the buffers drivers that ended already combined into it, whose own
    /// spill hooks ended with them.
    global: GlobalState,
}

/// A combined driver's state, in combine order.
#[derive(Debug)]
enum Part {
    Run(Runs),
    Buffer(Buffer),
}

#[derive(Default)]
struct SortShared {
    parts: Vec<Part>,
    /// The in-memory result, read by every reader.
    held: Option<Arc<Held>>,
}

/// A driver still sorting input: its spill signal and its buffer's bytes.
struct Live {
    signal: Arc<SpillSignal>,
    own: AtomicUsize,
}

impl SortGlobal {
    /// Whether another sort of the query is present: its state exists and
    /// its spilled result is not read to the end yet. Every `Sort` sink is
    /// a member, bucketed (a merge join's or a window's) or not.
    fn shares_query(&self) -> bool {
        lock(&self.present)
            .as_ref()
            .is_some_and(|s| s.cohort().members() > 1)
    }

    /// This sort's spilled result is read: its merges' budgets are gone,
    /// so it no longer counts as present for the query's other sorts.
    fn leave(&self) {
        drop(lock(&self.present).take());
    }

    /// Reports the bytes driver `live` could spill: its buffer's `own`
    /// and the combined buffers.
    fn report(&self, live: &Live, own: usize) {
        live.own.store(own, Ordering::Relaxed);
        let combined = self.combined.load(Ordering::Relaxed);
        live.signal
            .set_spillable_bytes(own.saturating_add(combined));
    }

    /// The combined buffers changed: every driver still sorting input
    /// reports them (one waiting for memory reports nothing on its own).
    /// A round of another consumer counts at most a consumer's own `used`
    /// bytes of what it reports (`arbitrate::request_spills`): a driver
    /// holding little exposes little of the combined bytes to it (the
    /// driver's own requests are not capped so).
    fn report_all(&self) {
        let combined = self.combined.load(Ordering::Relaxed);
        for l in lock(&self.live).iter().filter_map(Weak::upgrade) {
            l.signal
                .set_spillable_bytes(l.own.load(Ordering::Relaxed).saturating_add(combined));
        }
    }
}

struct SortGlobal {
    shared: Mutex<SortShared>,
    /// The bytes of the buffers among `shared.parts`, which every driver
    /// still sorting input reports as spillable with its own.
    combined: AtomicUsize,
    /// The drivers still sorting input (until they combine; weak, so a
    /// driver that ends with an error leaves no hook behind).
    live: Mutex<Vec<Weak<Live>>>,
    /// The spilled result's buckets and their readers.
    readers: Arc<Readers>,
    /// The query's bucketing (none: one bucket).
    bucketing: Option<Arc<Bucketing>>,
    buckets: usize,
    /// This sort's idle seat in the query's [`SORTS_COHORT`] (none outside
    /// a query), while the state lives and until its spilled result is
    /// read ([`SortGlobal::leave`]). A merge join's bucket readers and an
    /// in-memory result keep it while the state lives.
    present: Mutex<Option<Seat>>,
}

/// The readers of a spilled result: the buckets left to merge, and the
/// readers merging one. Each reader's merge reserves its runs' frames from
/// the pool. A reader alone waits for them with arbitration, as the one
/// reader of an unbucketed sort does; the others take only memory the
/// pool has free (leaving half of a limited pool to the operators above),
/// else give their bucket back and wait for a merge to end. So a result
/// one reader can merge never fails for its parallel readers: at a tight
/// limit fewer of them merge at once, down to one at a time.
#[derive(Default)]
struct Readers {
    state: Mutex<ReadersState>,
    /// A merge ended, the lone reader has its memory, or no reader holds
    /// a bucket any more: waiting readers claim again.
    changed: Event,
}

#[derive(Default)]
struct ReadersState {
    /// Each bucket's runs, claimed from the end.
    spilled: Vec<Vec<SpillRun>>,
    /// Each bucket's runs by bucket number (`Sort::read_by_bucket`).
    by_bucket: Vec<Vec<SpillRun>>,
    /// Readers holding a bucket (reserving or merging it).
    active: usize,
    /// Readers merging a bucket.
    merging: usize,
    /// The lone reader is waiting for its merge's memory: the others wait
    /// rather than take memory it waits for.
    reserving: bool,
}

/// A reader's claim on the spilled result.
enum Claim {
    /// No bucket is left to claim; `last` when no reader holds one either
    /// (decided under the claim's lock, so a bucket given back meanwhile
    /// is never missed): the result is read.
    Done { last: bool },
    /// The lone reader is reserving: try again once it has.
    Wait,
    /// A bucket's runs to merge, and the reader's slot.
    Bucket(Vec<SpillRun>, ReaderSlot),
}

impl Readers {
    fn claim(self: &Arc<Self>) -> Claim {
        let mut s = lock(&self.state);
        if s.reserving {
            return Claim::Wait;
        }
        let Some(runs) = s.spilled.pop() else {
            return Claim::Done {
                last: s.active == 0,
            };
        };
        let alone = s.active == 0;
        s.active += 1;
        s.reserving = alone;
        Claim::Bucket(
            runs,
            ReaderSlot {
                readers: Arc::clone(self),
                alone,
                counted: true,
                merging: false,
            },
        )
    }
}

/// A reader holding a bucket of a spilled result (see [`Readers`]): gone
/// (dropped, also on error and cancel) once its merge ends, which lets
/// waiting readers try again.
struct ReaderSlot {
    readers: Arc<Readers>,
    /// No other reader held a bucket when this one claimed it: its merge
    /// waits for memory with arbitration.
    alone: bool,
    /// Still counted in [`ReadersState::active`].
    counted: bool,
    /// Counted in [`ReadersState::merging`].
    merging: bool,
}

impl ReaderSlot {
    /// The merge has its memory and starts. Returns the readers merging.
    fn merging(&mut self) -> usize {
        self.merging = true;
        let mut s = lock(&self.readers.state);
        s.merging += 1;
        s.merging
    }

    /// The lone reader has its merge's memory: the others may try again.
    fn reserved(&self) {
        if self.alone {
            lock(&self.readers.state).reserving = false;
            self.readers.changed.notify();
        }
    }

    /// Gives `runs` back unmerged (the pool refused their memory). Wakes the
    /// waiting readers when no reader holds a bucket any more, so one of
    /// them claims it alone.
    fn give_back(mut self, runs: Vec<SpillRun>) {
        let idle = {
            let mut s = lock(&self.readers.state);
            s.spilled.push(runs);
            s.active = s.active.saturating_sub(1);
            s.active == 0
        };
        self.counted = false;
        if idle {
            self.readers.changed.notify();
        }
    }
}

impl Drop for ReaderSlot {
    fn drop(&mut self) {
        if !self.counted {
            return;
        }
        {
            let mut s = lock(&self.readers.state);
            s.active = s.active.saturating_sub(1);
            if self.merging {
                s.merging = s.merging.saturating_sub(1);
            }
            // Only a lone reader reserves, and no other claims meanwhile.
            if self.alone {
                s.reserving = false;
            }
        }
        self.readers.changed.notify();
    }
}

impl Sort {
    /// Writes out the buffers among the combined parts; returns the bytes
    /// freed, and every driver still sorting input reports what is left.
    /// Out of the lock while spilling: a driver spills them only before it
    /// combines, so the last to combine finalizes after every such spill
    /// gave its parts back (`finalize` takes them, and is not polled during
    /// the spill it services itself); an error fails the query (the parts
    /// go). Two drivers spilling at once may put whole combines back out of
    /// combine order (one driver's runs and buffer stay together, in
    /// order); a driver's runs merge as one input either way.
    async fn spill_combined(
        &self,
        g: &SortGlobal,
        batch_rows: usize,
        ctx: &TaskContext,
    ) -> Result<usize> {
        let mut parts = mem::take(&mut lock(&g.shared).parts);
        let mut freed = 0;
        for part in &mut parts {
            let Part::Buffer(b) = part else { continue };
            let before = b.reservation.size();
            if let Ok(runs) = self.spill_buffer(b, g.buckets, batch_rows, ctx).await? {
                freed += before;
                *part = Part::Run(runs);
            }
        }
        let mut shared = lock(&g.shared);
        parts.append(&mut shared.parts);
        shared.parts = parts;
        g.combined.store(buffered(&shared.parts), Ordering::Relaxed);
        drop(shared);
        g.report_all();
        Ok(freed)
    }

    /// Spills `buffer` (not empty) as one sorted run per bucket on the
    /// blocking pool. Returns the runs and empties the buffer, or
    /// `Ok(Err(_))` and keeps it when the pool refused the writer's scratch
    /// memory.
    async fn spill_buffer(
        &self,
        buffer: &mut Buffer,
        buckets: usize,
        batch_rows: usize,
        ctx: &TaskContext,
    ) -> Result<Result<Runs, MemoryError>> {
        let target = headroom(buffer.row_bytes, batch_rows);
        let rows = spill_rows(buffer.row_bytes, batch_rows);
        let (batches, handle) = (buffer.batches.clone(), Arc::clone(&buffer.handle));
        let written = self.write_buffer(batches, buckets, rows, &handle, ctx);
        // Until the buffer's reservation shrinks below, its rows count as
        // about to be released, so another driver's round that cannot spill
        // waits for them instead of failing. The write is one blocking job
        // with no arbitration (a merge pass frees nothing, so it takes none).
        let _writing = handle.spilling();
        let runs = match lending_for_write(&mut buffer.headroom, &handle, target, written).await? {
            Ok(runs) => runs,
            Err(refused) => return Ok(Err(refused)),
        };
        buffer.batches.clear();
        let freed = buffer.reservation.size();
        buffer.reservation.shrink(freed);
        let written = runs.iter().flatten().count() as u64;
        self.counters.runs.fetch_add(written, Ordering::Relaxed);
        Ok(Ok(runs))
    }

    /// The in-memory result of `buffers` in `buckets` buckets, charged to
    /// a consumer of its own whose hook spills the rest of it (with a
    /// spill scope).
    fn held(
        &self,
        mut buffers: Vec<Buffer>,
        buckets: usize,
        ctx: &TaskContext,
    ) -> Result<Arc<Held>> {
        let (hook, handle) = match &self.spill {
            Some(scope) => {
                let (hook, handle) = OutputSpill::register(ctx, "Sort output");
                for b in &mut buffers {
                    for r in [&mut b.reservation, &mut b.headroom] {
                        r.transfer_to(&handle).map_err(KrishivError::from)?;
                    }
                }
                (Some((hook, scope.clone())), handle)
            }
            None => (None, Arc::clone(ctx.consumer().consumer())),
        };
        let bytes = buffers.iter().map(|b| b.reservation.size()).sum();
        let batches: Arc<[Arc<SortedBatch>]> = buffers
            .iter()
            .flat_map(|b| b.batches.iter().cloned())
            .collect();
        let held = Arc::new(Held {
            state: Mutex::new(HeldState {
                batches,
                buffers,
                buckets: (0..buckets).map(|_| Bucket::Unread).collect(),
                next: 0,
                spilling: false,
                failed: None,
            }),
            changed: Event::default(),
            spill: hook,
            schema: Arc::clone(&self.schema),
            encoder: Arc::clone(&self.encoder),
            handle,
            counters: Arc::clone(&self.counters),
            query: Arc::clone(ctx.query()),
            executor: Arc::clone(ctx.executor()),
            batch_rows: ctx.query().config().batch_size.max(1),
        });
        if let Some((hook, _)) = &held.spill {
            let weak: Weak<dyn SpillUnread> = Arc::downgrade(&held) as Weak<Held>;
            hook.attach(weak);
            hook.set_spillable_bytes(bytes);
        }
        Ok(held)
    }

    /// Merges each bucket of `memory` into a new spill file of batches of
    /// `rows` rows, on the blocking pool; scratch memory comes from
    /// `handle` (credit first). `Ok(Err(_))` if the pool refused it, with
    /// nothing written.
    async fn write_buffer(
        &self,
        memory: Vec<Arc<SortedBatch>>,
        buckets: usize,
        rows: usize,
        handle: &Arc<ConsumerHandle>,
        ctx: &TaskContext,
    ) -> Result<Result<Runs, MemoryError>> {
        let scope = self.spill.clone();
        let scope = scope.ok_or_else(|| internal("a sort spilled without a scope".to_owned()))?;
        let (schema, encoder) = (Arc::clone(&self.schema), Arc::clone(&self.encoder));
        let (h, query) = (Arc::clone(handle), Arc::clone(ctx.query()));
        let written = ctx
            .run_blocking(move || {
                let mut runs: Runs = Vec::with_capacity(buckets);
                for b in 0..buckets {
                    if memory.iter().all(|s| s.bucket(b).is_empty()) {
                        runs.push(None);
                        continue;
                    }
                    let mut merger =
                        Merger::bucket(Arc::clone(&schema), Arc::clone(&encoder), &memory, b, &h)?;
                    let scratch = h.reservation().releasing_to_credit().for_spill_scratch();
                    match write_run(&mut merger, &scope, scratch, rows, &query)? {
                        Ok(run) => runs.push(Some(run)),
                        // Refused: the runs written go (their files with
                        // them) and the buffer stays.
                        Err(refused) => return Ok(Err(refused)),
                    }
                }
                Ok(Ok(runs))
            })
            .await?;
        if let Ok(runs) = &written {
            for run in runs.iter().flatten() {
                self.counters
                    .bytes
                    .fetch_add(run.file.bytes(), Ordering::Relaxed);
            }
        }
        Ok(written)
    }

    /// Merges `spilled` runs into one new spill file of batches of `rows`
    /// rows, on the blocking pool; scratch memory comes from `handle`
    /// (credit first). `Ok(Err(_))` if the pool refused it.
    async fn write(
        &self,
        spilled: Vec<SpillRun>,
        rows: usize,
        env: &MergeEnv<'_>,
    ) -> Result<Result<SpillRun, MemoryError>> {
        let scope = self.spill.clone();
        let scope = scope.ok_or_else(|| internal("a sort spilled without a scope".to_owned()))?;
        let (schema, encoder) = (Arc::clone(&self.schema), Arc::clone(&self.encoder));
        let (h, query) = (Arc::clone(env.handle), Arc::clone(env.query));
        let written = blocking(env.query, env.executor, move || {
            let mut merger = Merger::new(schema, encoder, Vec::new(), &spilled, &h)?;
            let scratch = h.reservation().releasing_to_credit().for_spill_scratch();
            write_run(&mut merger, &scope, scratch, rows, &query)
        })
        .await?;
        if let Ok(run) = &written {
            self.counters
                .bytes
                .fetch_add(run.file.bytes(), Ordering::Relaxed);
        }
        Ok(written)
    }

    /// Merges consecutive runs, up to `fan_in` at a time and level by level,
    /// until at most `fan_in` (what the memory pool allows) remain. A merged
    /// group takes its first run's place, so runs keep their relative order
    /// (and ties their input order).
    ///
    /// `seat` (idle, in the query's [`MERGE_COHORT`]) holds while the
    /// budget is held and is idle again once it is released. On an error it
    /// may stay holding after the budget is gone: the caller's drop of the
    /// seat (finalize's) then wakes the cohort.
    async fn reduce(
        &self,
        runs: &mut Vec<SpillRun>,
        batch_rows: usize,
        env: &MergeEnv<'_>,
        mut seat: Option<&mut Seat>,
        shares_query: bool,
    ) -> Result<()> {
        if runs.len() <= 2 {
            return Ok(());
        }
        let handle = Arc::clone(env.handle);
        let mut max_fan_in = self.max_fan_in;
        // With another sort in the query, the final merge keeps to half the
        // pool, which stays for that sort (a merge holds its budget until it
        // ends, and the other sort may need the rest to buffer, finalize or
        // merge). A merge join reads two final merges at once: each keeps to
        // a quarter, so the two and the join's group fit together. A sort
        // alone in its query takes the fan-in the pool allows.
        let limit = handle.pool().effective_limit();
        let share = limit.and_then(|limit| {
            if self.read_by_bucket {
                Some(limit / 4)
            } else {
                shares_query.then_some(limit / 2)
            }
        });
        if let Some(share) = share {
            while max_fan_in > 2
                && merge_need(runs, max_fan_in.min(runs.len()), batch_rows, false) > share
            {
                max_fan_in /= 2;
            }
        }
        let waits = &self.counters.merge_waits;
        let (mut budget, fan_in) = merge_budget(
            runs,
            max_fan_in,
            batch_rows,
            &handle,
            true,
            seat.as_deref_mut().map(|s| (s, waits)),
        )
        .await?;
        let target = budget.size();
        while runs.len() > fan_in {
            self.counters.passes.fetch_add(1, Ordering::Relaxed);
            let mut level = mem::take(runs).into_iter();
            while level.len() > 0 {
                // Stop merging once the rest fits the fan-in.
                if runs.len() + level.len() <= fan_in {
                    runs.extend(level.by_ref());
                    break;
                }
                let take = group_size(runs.len() + level.len(), fan_in);
                let group: Vec<SpillRun> = level.by_ref().take(take).collect();
                let row_bytes = group.iter().map(|r| r.row_bytes).max().unwrap_or(1);
                let rows = spill_rows(row_bytes, batch_rows);
                let merged = self.write(group, rows, env);
                let run = lending_for_write(&mut budget, &handle, target, merged).await?;
                runs.push(run.map_err(KrishivError::from)?);
            }
        }
        // The memory goes before the seat wakes the cohort.
        drop(budget);
        if let Some(seat) = seat {
            seat.set(SeatState::Idle);
        }
        Ok(())
    }
}

/// What a merge pass runs with: the query, its executor (the blocking
/// pool) and the consumer whose memory the merges take. A sink's finalize
/// and a source's reader both have one.
struct MergeEnv<'a> {
    query: &'a Arc<QueryContext>,
    executor: &'a Arc<dyn Executor>,
    handle: &'a Arc<ConsumerHandle>,
}

/// Runs to merge next when `total` remain at a level (more than `fan_in`):
/// a group of `g` leaves `g - 1` fewer, so never more than it takes to get
/// down to the fan-in, and as few bytes as possible are written again.
pub(super) fn group_size(total: usize, fan_in: usize) -> usize {
    fan_in.min(total.saturating_sub(fan_in) + 1)
}

impl Sink for Sort {
    fn name(&self) -> &'static str {
        "Sort"
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        let bits = if ctx.query().config().sort_hash_buckets {
            bucket_bits(ctx.executor().parallelism())
        } else {
            0
        };
        let bucketing = self
            .bucket_keys
            .as_ref()
            .filter(|_| bits > 0)
            .map(|keys| Arc::new(Bucketing::new(Arc::clone(keys), bits)));
        let buckets = bucketing.as_ref().map_or(1, |b| b.buckets());
        self.counters
            .buckets
            .store(buckets as u64, Ordering::Relaxed);
        Ok(Arc::new(SortGlobal {
            shared: Mutex::default(),
            combined: AtomicUsize::new(0),
            live: Mutex::default(),
            readers: Arc::default(),
            bucketing,
            buckets,
            present: Mutex::new(
                // Always idle: no grant credit (`Cohort::join`).
                ctx.consumer()
                    .consumer()
                    .pool()
                    .cohort(SORTS_COHORT)
                    .map(|c| c.join()),
            ),
        }))
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        let g = global_ref::<SortGlobal>(global, "Sort")?;
        let memory = ctx.consumer();
        Ok(Box::new(SortLocal {
            buffer: Buffer::new(memory.consumer()),
            runs: Vec::new(),
            live: {
                let live = Arc::new(Live {
                    signal: Arc::clone(memory.signal()),
                    own: AtomicUsize::new(0),
                });
                lock(&g.live).push(Arc::downgrade(&live));
                live
            },
            batch_rows: ctx.query().config().batch_size.max(1),
            bucketing: g.bucketing.clone(),
            prune: PruneStats::default(),
            global: Arc::clone(global),
        }))
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        _global: &GlobalState,
    ) -> Result<SinkStatus> {
        let l = local_mut::<SortLocal>(local, "Sort")?;
        let rows = input.num_rows();
        if rows == 0 {
            return Ok(SinkStatus::NeedMoreInput);
        }
        let data = data_bytes(&input);
        let row_bytes = l.buffer.row_bytes.max(data.div_ceil(rows));
        // A small slice of a large batch, or views into an exchange's shared
        // pages, would keep (and account) the whole buffers.
        let input = owned_rows(input)?;
        let (sorted, looked_at, dropped) = self.prune(
            SortedBatch::new(input, &self.encoder, l.bucketing.as_deref())?,
            &l.prune,
        )?;
        let bytes = sorted.bytes();
        let extra = if self.spill.is_some() {
            headroom(row_bytes, l.batch_rows).saturating_sub(l.buffer.headroom.size())
        } else {
            0
        };
        let b = &mut l.buffer;
        if b.reservation.try_grow(bytes + extra).is_err() {
            return Ok(SinkStatus::NeedMemory {
                bytes: bytes + extra,
            });
        }
        // Moves `extra` to the headroom (through the consumer's credit).
        b.reservation.shrink_to_credit(extra);
        b.headroom.try_grow(extra)?;
        b.row_bytes = row_bytes;
        b.batches.push(Arc::new(sorted));
        self.counters.rows.fetch_add(rows as u64, Ordering::Relaxed);
        // Counted once taken: a batch refused for memory comes back.
        l.prune.count(looked_at, dropped);
        self.counters
            .pruned
            .fetch_add(dropped as u64, Ordering::Relaxed);
        if self.spill.is_some() {
            global_ref::<SortGlobal>(&l.global, "Sort")?.report(&l.live, b.reservation.size());
        }
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        let g = global_ref::<SortGlobal>(global, "Sort")?;
        let l = local
            .downcast::<SortLocal>()
            .map_err(|_| internal("wrong local state for Sort".to_owned()))?;
        let SortLocal {
            buffer, runs, live, ..
        } = *l;
        {
            let mut shared = lock(&g.shared);
            shared.parts.extend(runs.into_iter().map(Part::Run));
            if !buffer.batches.is_empty() {
                shared.parts.push(Part::Buffer(buffer));
            }
            // Without a spill scope nothing spills: nothing is offered.
            if self.spill.is_some() {
                g.combined.store(buffered(&shared.parts), Ordering::Relaxed);
            }
        }
        lock(&g.live)
            .retain(|l| l.strong_count() > 0 && !std::ptr::eq(l.as_ptr(), Arc::as_ptr(&live)));
        // The buffers stay charged to this driver's consumer, but its spill
        // hook ends with the driver: the drivers still sorting input report
        // and spill them, and the last to combine finalizes (spilling them
        // itself when external), so this signal offers nothing.
        live.signal.set_spillable_bytes(0);
        if self.spill.is_some() {
            g.report_all();
        }
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let g = global_ref::<SortGlobal>(global, "Sort")?;
            let parts = mem::take(&mut lock(&g.shared).parts);
            g.combined.store(0, Ordering::Relaxed);
            ctx.consumer().signal().set_spillable_bytes(0);
            let batch_rows = ctx.query().config().batch_size.max(1);
            if !parts.iter().any(|p| matches!(p, Part::Run(_))) {
                let buffers = parts
                    .into_iter()
                    .filter_map(|p| match p {
                        Part::Buffer(mut b) => {
                            // Without a spill scope nothing spills any
                            // more: the headroom goes. With one, the
                            // result's spill writes through it.
                            if self.spill.is_none() {
                                b.headroom.shrink(b.headroom.size());
                            }
                            Some(b)
                        }
                        Part::Run(_) => None,
                    })
                    .collect();
                let held = self.held(buffers, g.buckets, ctx)?;
                lock(&g.shared).held = Some(held);
                return Ok(());
            }
            // External: spill what is still buffered, so the output phase
            // holds only the merges' frames. The buffers are memory this
            // step frees on its own: a merge budget of another task waits
            // for it in the cohort ([`MERGE_COHORT`]), and the seat then
            // serves the merges (idle in between).
            let mut seat = holding(ctx.consumer().consumer());
            let mut spilled: Vec<Vec<SpillRun>> = (0..g.buckets).map(|_| Vec::new()).collect();
            for part in parts {
                let runs = match part {
                    Part::Run(r) => r,
                    Part::Buffer(mut b) => {
                        let mut runs = self
                            .spill_buffer(&mut b, g.buckets, batch_rows, ctx)
                            .await?;
                        if runs.is_err() {
                            // The scratch outgrew the headroom: widen it,
                            // with arbitration this time, and try again. The
                            // buffer's consumer is an ended driver's, so the
                            // seat goes idle meanwhile: a round never waits
                            // for the member asking in it.
                            let more = headroom(b.row_bytes, batch_rows);
                            if let Some(seat) = &mut seat {
                                seat.set(SeatState::Idle);
                            }
                            b.headroom.grow(more).await?;
                            if let Some(seat) = &mut seat {
                                seat.set(SeatState::Holding);
                            }
                            runs = self
                                .spill_buffer(&mut b, g.buckets, batch_rows, ctx)
                                .await?;
                        }
                        runs?
                    }
                };
                for (bucket, run) in spilled.iter_mut().zip(runs) {
                    bucket.extend(run);
                }
            }
            if let Some(seat) = &mut seat {
                seat.set(SeatState::Idle);
            }
            let env = MergeEnv {
                query: ctx.query(),
                executor: ctx.executor(),
                handle: ctx.consumer().consumer(),
            };
            for runs in &mut spilled {
                self.reduce(runs, batch_rows, &env, seat.as_mut(), g.shares_query())
                    .await?;
            }
            if self.read_by_bucket {
                lock(&g.readers.state).by_bucket = spilled;
                return Ok(());
            }
            spilled.retain(|runs| !runs.is_empty());
            // Readers claim from the end: the first bucket first.
            spilled.reverse();
            lock(&g.readers.state).spilled = spilled;
            Ok(())
        })
    }

    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        _target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let l = local_mut::<SortLocal>(local, "Sort")?;
            if self.spill.is_none() {
                return Ok(0);
            }
            let g = global_ref::<SortGlobal>(&l.global, "Sort")?;
            // The buffers ended drivers combined first: no other hook
            // reaches them before finalize.
            let mut freed = self.spill_combined(g, l.batch_rows, ctx).await?;
            let before = l.buffer.reservation.size();
            if !l.buffer.batches.is_empty() {
                let buckets = l.bucketing.as_ref().map_or(1, |b| b.buckets());
                // Refused scratch memory: this buffer frees nothing this
                // time.
                if let Ok(runs) = self
                    .spill_buffer(&mut l.buffer, buckets, l.batch_rows, ctx)
                    .await?
                {
                    l.runs.push(runs);
                    freed += before;
                }
            }
            g.report(&l.live, l.buffer.reservation.size());
            Ok(freed)
        })
    }

    fn spill_global<'a>(
        &'a self,
        global: &'a GlobalState,
        _target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let g = global_ref::<SortGlobal>(global, "Sort")?;
            if self.spill.is_none() {
                return Ok(0);
            }
            let batch_rows = ctx.query().config().batch_size.max(1);
            let freed = self.spill_combined(g, batch_rows, ctx).await?;
            ctx.consumer()
                .signal()
                .set_spillable_bytes(g.combined.load(Ordering::Relaxed));
            Ok(freed)
        })
    }

    fn extra_metrics(&self) -> crate::metrics::ExtraMetrics {
        if !self.bucketed() {
            return Vec::new();
        }
        let buckets = self.counters.buckets.load(Ordering::Relaxed);
        vec![(
            std::borrow::Cow::Borrowed("buckets"),
            crate::metrics::Metric::Max(buckets),
        )]
    }
}

/// Bytes held by the buffers among `parts`.
fn buffered(parts: &[Part]) -> usize {
    parts
        .iter()
        .map(|p| match p {
            Part::Buffer(b) => b.reservation.size(),
            Part::Run(_) => 0,
        })
        .sum()
}

/// The source side's per-driver state.
struct SortSource {
    state: SourceState,
    /// The one bucket this reader reads ([`Sort::bucket_reader`]).
    bucket: Option<usize>,
    query: Arc<QueryContext>,
    executor: Arc<dyn Executor>,
    handle: Arc<ConsumerHandle>,
    batch_rows: usize,
}

/// A merge of spilled runs, holding the budget for the runs' frames.
struct Merging {
    merger: Box<Merger>,
    budget: Reservation,
    /// A spilled bucket's reader slot (none: the rest of an in-memory
    /// bucket), dropped after the budget.
    slot: Option<ReaderSlot>,
    /// Its seat in the query's merge cohort, holding while the budget is
    /// held (every reader; a merge join's bucket reader joins once granted
    /// and never waits), dropped after the budget.
    seat: Option<Seat>,
}

/// Where a reader goes once a merge of spilled runs ends.
enum After {
    /// Back to the in-memory result's buckets.
    Held(Arc<Held>),
    /// To the next spilled bucket.
    Spilled,
}

enum SourceState {
    Start,
    /// Reading an in-memory result's buckets: the one claimed, if any.
    Held(Arc<Held>, Option<usize>),
    Merging(Merging, After),
    /// Claiming the spilled buckets.
    Spilled,
    Done,
}

/// One step of reading a bucket of an in-memory result.
enum Step {
    Batch(Batch),
    /// The bucket is read.
    Done,
    /// The rest was spilled: merge it from disk.
    Run(SpillRun),
}

impl Source for Sort {
    fn name(&self) -> &'static str {
        "Sort"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        finalized::<SortGlobal>(ctx, "Sort")
    }

    fn max_parallelism(&self, global: &GlobalState) -> usize {
        global_ref::<SortGlobal>(global, "Sort").map_or(1, |g| g.buckets)
    }

    fn local_state(&self, ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(SortSource {
            state: SourceState::Start,
            bucket: None,
            query: Arc::clone(ctx.query()),
            executor: Arc::clone(ctx.executor()),
            handle: Arc::clone(ctx.consumer().consumer()),
            batch_rows: ctx.query().config().batch_size.max(1),
        }))
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(async move {
            let l = local_mut::<SortSource>(local, "Sort")?;
            let g = global_ref::<SortGlobal>(global, "Sort")?;
            loop {
                match mem::replace(&mut l.state, SourceState::Done) {
                    SourceState::Start => {
                        let held = lock(&g.shared).held.clone();
                        l.state = match (held, l.bucket) {
                            (Some(held), b) => SourceState::Held(held, b),
                            (None, None) => SourceState::Spilled,
                            (None, Some(b)) => {
                                let runs = lock(&g.readers.state)
                                    .by_bucket
                                    .get_mut(b)
                                    .map(mem::take)
                                    .unwrap_or_default();
                                if runs.is_empty() {
                                    return Ok(None);
                                }
                                let merging = self.open_runs(l, g, runs, None).await?;
                                let merging = merging.map_err(|_| {
                                    internal("a waiting merge was refused memory".to_owned())
                                })?;
                                SourceState::Merging(merging, After::Spilled)
                            }
                        };
                    }
                    SourceState::Held(held, bucket) => {
                        let next = match (bucket, l.bucket) {
                            (Some(b), _) => Some(b),
                            // Its one bucket is read.
                            (None, Some(_)) => return Ok(None),
                            (None, None) => held.claim()?,
                        };
                        let Some(b) = next else {
                            continue;
                        };
                        match read_bucket(l, &held, b).await? {
                            Step::Batch(batch) => {
                                l.state = SourceState::Held(held, Some(b));
                                return Ok(Some(batch));
                            }
                            Step::Done => l.state = SourceState::Held(held, None),
                            Step::Run(run) => {
                                let merging = self.open_runs(l, g, vec![run], None).await?;
                                let merging = merging.map_err(|_| {
                                    internal("a waiting merge was refused memory".to_owned())
                                })?;
                                l.state = SourceState::Merging(merging, After::Held(held));
                            }
                        }
                    }
                    // Its one bucket is read.
                    SourceState::Spilled if l.bucket.is_some() => return Ok(None),
                    SourceState::Spilled => {
                        // Listen before claiming, so a merge ending in
                        // between is not missed.
                        let changed = g.readers.changed.listen();
                        match g.readers.claim() {
                            // The last reader is done: the result is read.
                            Claim::Done { last } => {
                                if last {
                                    g.leave();
                                }
                            }
                            Claim::Wait => {
                                changed.await;
                                l.state = SourceState::Spilled;
                            }
                            Claim::Bucket(runs, slot) => {
                                match self.open_runs(l, g, runs, Some(slot)).await? {
                                    Ok(merging) => {
                                        l.state = SourceState::Merging(merging, After::Spilled);
                                    }
                                    // Other readers hold the memory: one
                                    // fewer reader until a merge ends.
                                    Err((runs, slot)) => {
                                        slot.give_back(runs);
                                        changed.await;
                                        l.state = SourceState::Spilled;
                                    }
                                }
                            }
                        }
                    }
                    SourceState::Merging(
                        Merging {
                            mut merger,
                            mut budget,
                            slot,
                            seat,
                        },
                        after,
                    ) => {
                        // Each output batch on the blocking pool: a wide
                        // merge over ~1 ms, or reading spilled runs.
                        let rows = l.batch_rows;
                        let step = blocking(&l.query, &l.executor, move || {
                            let next = merger.next(rows)?;
                            Ok((merger, next))
                        });
                        // No top-up: the merger keeps the frames it drew,
                        // and frames it frees come back as credit for the
                        // next step.
                        let (merger, next) = lending(&mut budget, &l.handle, 0, step).await?;
                        if let Some(batch) = next {
                            l.state = SourceState::Merging(
                                Merging {
                                    merger,
                                    budget,
                                    slot,
                                    seat,
                                },
                                after,
                            );
                            return Ok(Some(batch));
                        }
                        // Spilled runs' last frames went back as credit;
                        // then waiting readers may take the memory.
                        drop((merger, budget));
                        l.handle.release_credit();
                        drop((slot, seat));
                        l.state = match after {
                            After::Held(held) => SourceState::Held(held, None),
                            After::Spilled => SourceState::Spilled,
                        };
                    }
                    SourceState::Done => return Ok(None),
                }
            }
        })
    }
}

impl Sort {
    /// Whether a final merge of `runs` must first be reduced to half the
    /// pool: another sort of the query is present now and the merge needs
    /// more. Not for a merge join's sorts, which finalize keeps to a
    /// quarter whatever is present.
    fn recheck_share(
        &self,
        g: &SortGlobal,
        runs: &[SpillRun],
        batch_rows: usize,
        handle: &ConsumerHandle,
    ) -> bool {
        if self.read_by_bucket || runs.len() <= 2 {
            return false;
        }
        handle.pool().effective_limit().is_some_and(|limit| {
            g.shares_query() && merge_need(runs, runs.len(), batch_rows, false) > limit / 2
        })
    }

    /// Opens the final merge of spilled `runs`. A reader `slot` not alone
    /// takes only free memory: refused, it gets the runs and slot back.
    ///
    /// A reader that may wait (alone, or the rest of an in-memory result)
    /// first checks the query's other sorts again: finalize decided the
    /// fan-in by the sorts present then, and one may have started since (a
    /// later dependency wave, or a task of the query started later). If one
    /// is present now and the merge needs over half the pool, more passes
    /// bring it down to half first, as finalize would have.
    async fn open_runs(
        &self,
        l: &SortSource,
        g: &SortGlobal,
        mut runs: Vec<SpillRun>,
        slot: Option<ReaderSlot>,
    ) -> Result<std::result::Result<Merging, (Vec<SpillRun>, ReaderSlot)>> {
        let (schema, encoder) = (Arc::clone(&self.schema), Arc::clone(&self.encoder));
        let handle = Arc::clone(&l.handle);
        let (mut budget, slot, seat) = match slot {
            Some(slot) if !slot.alone => {
                let mut budget = handle.reservation();
                let need = merge_need(&runs, runs.len(), l.batch_rows, false);
                // Only memory to spare: half the pool stays for the
                // operators above, whose memory is not reclaimed from a
                // merge in progress. The limit may be a pool's above the
                // query's (a session's query pool has none of its own).
                let pool = handle.pool();
                let spare = pool
                    .effective_limit()
                    .is_none_or(|limit| pool.reserved().saturating_add(need) <= limit / 2);
                // It never waits, but holds a merge budget a waiter in the
                // cohort can wait for: holding before it takes the memory,
                // so a request refused meanwhile waits too (idle again on a
                // refusal, the seat's drop).
                let seat = holding(&handle);
                if !spare || budget.try_grow(need).is_err() {
                    drop(seat);
                    return Ok(Err((runs, slot)));
                }
                (budget, Some(slot), seat)
            }
            slot => {
                // A merge join reads two buckets' merges in one driver: one
                // must never wait for the other, so a bucket reader joins
                // the cohort only once its budget is granted.
                let mut seat = if l.bucket.is_none() {
                    handle.join_cohort(MERGE_COHORT)
                } else {
                    None
                };
                // A lone reader reduces while it reserves: the other readers
                // wait (`Claim::Wait`) for these whole passes, as they wait
                // for its merge budget.
                if self.recheck_share(g, &runs, l.batch_rows, &handle) {
                    let env = MergeEnv {
                        query: &l.query,
                        executor: &l.executor,
                        handle: &handle,
                    };
                    let reduced = self
                        .reduce(&mut runs, l.batch_rows, &env, seat.as_mut(), true)
                        .await;
                    // On an error the slot's drop ends the reader's wait
                    // and the seat's drop wakes the cohort.
                    reduced?;
                }
                let waits = &self.counters.merge_waits;
                let budget = merge_budget(
                    &runs,
                    runs.len(),
                    l.batch_rows,
                    &handle,
                    false,
                    seat.as_mut().map(|s| (s, waits)),
                )
                .await;
                if l.bucket.is_some() && budget.is_ok() {
                    seat = holding(&handle);
                }
                // The lone reader's wait is over (also on error).
                if let Some(slot) = &slot {
                    slot.reserved();
                }
                (budget?.0, slot, seat)
            }
        };
        let mut slot = slot;
        if let Some(slot) = &mut slot {
            let merging = slot.merging() as u64;
            self.counters.readers.fetch_max(merging, Ordering::Relaxed);
        }
        self.counters
            .fan_in
            .fetch_max(runs.len() as u64, Ordering::Relaxed);
        let open = blocking(&l.query, &l.executor, move || {
            Merger::new(schema, encoder, Vec::new(), &runs, &handle)
        });
        let merger = match lending(&mut budget, &l.handle, 0, open).await {
            Ok(merger) => merger,
            Err(e) => {
                // The memory goes before the seat wakes the cohort.
                drop(budget);
                drop((slot, seat));
                return Err(e);
            }
        };
        Ok(Ok(Merging {
            merger: Box::new(merger),
            budget,
            slot,
            seat,
        }))
    }
}

/// Asks a reader's in-memory result to spill again, as a request made on
/// stale spillable bytes would (tests).
#[cfg(test)]
pub(super) async fn spill_result_again(local: &mut LocalState) -> Result<usize> {
    let l = local_mut::<SortSource>(local, "Sort")?;
    let SourceState::Held(held, _) = &l.state else {
        return Err(internal("the reader holds no in-memory result".to_owned()));
    };
    Arc::clone(held).spill_unread().await
}

/// One step of reading bucket `b` of an in-memory result: the next batch,
/// the end of the bucket, or the run its rest was spilled to.
async fn read_bucket(l: &SortSource, held: &Arc<Held>, b: usize) -> Result<Step> {
    let (merger, batches) = loop {
        let changed = held.changed.listen();
        match held.take(b)? {
            Take::Open(batches) => break (None, Some(batches)),
            Take::Merge(m) => break (Some(m), None),
            Take::Run(run) => return Ok(Step::Run(run)),
            Take::Done => return Ok(Step::Done),
            // A spill holds the result.
            Take::Wait => changed.await,
        }
    };
    let opened = match &batches {
        Some(batches) => open_parallel(l, held, batches).await,
        None => Ok(None),
    };
    let mut merger = match opened {
        Ok(Some(opened)) => Some(opened),
        Ok(None) => merger,
        Err(e) => {
            held.fail(e.clone());
            return Err(e);
        }
    };
    let rows = l.batch_rows;
    if let Some(m) = &mut merger {
        let jobs = m.gather_jobs(rows);
        if !jobs.is_empty() {
            let gather = jobs
                .into_iter()
                .map(|job| blocking(&l.query, &l.executor, move || job.run()));
            match futures::future::try_join_all(gather).await {
                Ok(batches) => m.gathered(batches),
                Err(e) => {
                    held.fail(e.clone());
                    return Err(e);
                }
            }
        }
        if let Some(batch) = m.pop_ready() {
            if let Some(m) = merger {
                held.put(b, Bucket::Merging(m));
            }
            return Ok(Step::Batch(batch));
        }
    }
    let (schema, encoder, handle) = (
        Arc::clone(&held.schema),
        Arc::clone(&held.encoder),
        Arc::clone(&held.handle),
    );
    let step = blocking(&l.query, &l.executor, move || {
        let merger = match (merger, batches) {
            (Some(m), _) => m,
            (None, batches) => {
                let batches = batches.unwrap_or_else(|| Arc::new([]));
                Box::new(Merger::bucket(schema, encoder, &batches, b, &handle)?)
            }
        };
        let mut merger = merger;
        let next = merger.next(rows);
        Ok((merger, next))
    })
    .await;
    match step {
        Ok((merger, Ok(Some(batch)))) => {
            held.put(b, Bucket::Merging(merger));
            Ok(Step::Batch(batch))
        }
        Ok((merger, Ok(None))) => {
            drop(merger);
            held.put(b, Bucket::Done);
            Ok(Step::Done)
        }
        Ok((_, Err(e))) | Err(e) => {
            held.fail(e.clone());
            Err(e)
        }
    }
}

/// The merge of an unbucketed sort's wide in-memory result, computed as one
/// blocking task per part, its output gathered a batch per part at once: a
/// loser tree over thousands of batches costs a chain of cache misses per
/// row, on one thread. `None`: the tree it is.
async fn open_parallel(
    l: &SortSource,
    held: &Arc<Held>,
    batches: &Arc<[Arc<SortedBatch>]>,
) -> Result<Option<Box<Merger>>> {
    let parallelism = l.executor.parallelism();
    let Some(ParallelMerge {
        memory,
        parts,
        sort_parts,
        scratch,
        ready_bytes,
    }) = parallel_merge(batches, parallelism, l.batch_rows, &held.handle)
    else {
        return Ok(None);
    };
    let memory = Arc::new(memory);
    let shared = Arc::clone(&memory);
    let splitters = blocking(&l.query, &l.executor, move || {
        merge_splitters(&shared, sort_parts)
    })
    .await?;
    let lows = std::iter::once(None).chain(splitters.iter().copied().map(Some));
    let highs = splitters.iter().copied().map(Some).chain([None]);
    // Each running sort holds a share of `scratch`, so it outlives this
    // future if the read is dropped.
    let scratch = Arc::new(scratch);
    let sorted = lows.zip(highs).map(|(lo, hi)| {
        let query = Arc::clone(&l.query);
        let memory = Arc::clone(&memory);
        let scratch = Arc::clone(&scratch);
        blocking(&l.query, &l.executor, move || {
            let _scratch = scratch;
            query.check_cancelled()?;
            sort_slices(&merge_part(&memory, lo, hi)?)
        })
    });
    let mut sorted = futures::stream::iter(sorted).buffered(parts);
    // Every part, even after an error: a failed read frees `scratch` only
    // once no sort is using it.
    let mut orders = Vec::with_capacity(sort_parts);
    let mut failed = None;
    while let Some(part) = sorted.next().await {
        match part {
            Ok(order) => orders.push(order),
            Err(e) => failed = failed.or(Some(e)),
        }
    }
    drop(sorted);
    if let Some(e) = failed {
        return Err(e);
    }
    let scratch = Arc::into_inner(scratch)
        .ok_or_else(|| internal("a part sort outlived its read".to_owned()))?;
    let mut order = Vec::with_capacity(orders.iter().map(Vec::len).sum());
    orders.iter().for_each(|o| order.extend_from_slice(o));
    drop(orders);
    Ok(Some(Box::new(Merger::from_order(
        Arc::clone(&held.schema),
        Arc::clone(&held.encoder),
        &memory,
        order,
        (scratch, ready_bytes),
        parts,
    ))))
}

/// Writes the rest of `merger` with `writer` in batches of `rows` rows,
/// checking cancellation between batches (blocking I/O).
fn write_rest(
    merger: &mut Merger,
    mut writer: krishiv_memory::SpillWriter,
    rows: usize,
    query: &QueryContext,
) -> Result<SpillRun> {
    let (mut max_batch_bytes, mut row_bytes) = (0, 1);
    while let Some(batch) = merger.next(rows)? {
        query.check_cancelled()?;
        // What reading the frame back costs: the writer compacts view
        // columns, whose batch here still shares its inputs' buffers.
        let bytes = SpillWriter::frame_bytes(&batch);
        max_batch_bytes = max_batch_bytes.max(bytes);
        row_bytes = row_bytes.max(bytes.div_ceil(batch.num_rows().max(1)));
        writer.write(&batch).map_err(KrishivError::from)?;
    }
    let file = writer.finish().map_err(KrishivError::from)?;
    Ok(SpillRun {
        file,
        max_batch_bytes,
        row_bytes,
    })
}
