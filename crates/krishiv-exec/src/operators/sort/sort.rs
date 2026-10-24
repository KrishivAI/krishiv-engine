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
//! and the output is totally ordered.

use std::future::Future;
use std::mem;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use arrow_schema::SchemaRef;
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_memory::{ConsumerHandle, MemoryError, Reservation, SpillScope, SpillSignal};
use krishiv_types::batch::Batch;

use super::keys::{KeyEncoder, SortKey};
use super::merge::{
    Bucketing, Merger, SortedBatch, SpillRun, data_bytes, interleave_batches, write_run,
};
use super::{BATCH_SLACK_BYTES, SPILL_BATCH_BYTES, SPILL_BUFFER_BYTES, blocking, finalized};
use crate::context::{QueryContext, TaskContext};
use crate::error::{Result, cancelled, internal};
use crate::executor::{Executor, run_blocking};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus, Source};
use crate::operators::{global_ref, local_mut};
use crate::output_spill::{OutputSpill, SpillUnread, Unfinished, refused};
use crate::sync::{Event, Mutex, lock};
use crate::window::{GroupLimitPlan, RankLike, Ranker};

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
    /// Rows a window group limit dropped from the sorted batches.
    pub rows_pruned: u64,
}

#[derive(Debug, Default)]
struct Counters {
    rows: AtomicU64,
    runs: AtomicU64,
    bytes: AtomicU64,
    passes: AtomicU64,
    fan_in: AtomicU64,
    buckets: AtomicU64,
    pruned: AtomicU64,
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
            rows_pruned: c.pruned.load(Ordering::Relaxed),
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
            headroom: handle.reservation(),
            row_bytes: 0,
            handle: Arc::clone(handle),
        }
    }
}

/// Lends `res` to the next growths of its consumer (as grant credit) while
/// `work` runs, so the scratch memory of a spill or merge is never taken by
/// another consumer; then takes the credit back and tops `res` up to
/// `target` if the pool allows (best effort: a later reservation asks for
/// the rest).
async fn lending<T>(
    res: &mut Reservation,
    handle: &ConsumerHandle,
    target: usize,
    work: impl Future<Output = T>,
) -> T {
    let lent = res.size();
    res.shrink_to_credit(lent);
    let out = work.await;
    // Credit only: this never reaches the pool, so it cannot fail.
    let _ = res.try_grow(handle.credit());
    let _ = res.try_grow(target.saturating_sub(res.size()));
    out
}

/// [`lending`] for a spill write: takes back at most what it lent (or
/// `target`) and returns the rest of the consumer's credit to the pool. The
/// writer's scratch beyond the lent bytes is temporary, and may have come
/// from the pool's spill-scratch floor, which must not stay with one
/// consumer.
async fn lending_for_write<T>(
    res: &mut Reservation,
    handle: &ConsumerHandle,
    target: usize,
    work: impl Future<Output = T>,
) -> T {
    let lent = res.size();
    res.shrink_to_credit(lent);
    let out = work.await;
    // Credit only: this never reaches the pool, so it cannot fail.
    let _ = res.try_grow(handle.credit().min(lent.max(target)));
    handle.release_credit();
    let _ = res.try_grow(target.saturating_sub(res.size()));
    out
}

/// Buffer slack a sunk batch may carry over its data before it is
/// compacted.
const SLICE_SLACK_BYTES: usize = 16 << 10;

/// Rows per spilled batch: about [`SPILL_BATCH_BYTES`], at most a batch.
fn spill_rows(row_bytes: usize, batch_rows: usize) -> usize {
    (SPILL_BATCH_BYTES / row_bytes.max(1)).clamp(1, batch_rows.max(1))
}

/// Headroom to spill one buffer: one merged spill batch of `row_bytes`
/// data bytes per row (plus its buffers' alignment), the writer's scratch
/// (twice that) and its write buffer.
fn headroom(row_bytes: usize, batch_rows: usize) -> usize {
    row_bytes
        .saturating_mul(spill_rows(row_bytes, batch_rows))
        .saturating_add(BATCH_SLACK_BYTES)
        .saturating_mul(3)
        .saturating_add(SPILL_BUFFER_BYTES)
}

/// One spill of a buffer: a sorted run per bucket (`None` for a bucket
/// without rows).
type Runs = Vec<Option<SpillRun>>;

struct SortLocal {
    buffer: Buffer,
    runs: Vec<Runs>,
    signal: Arc<SpillSignal>,
    batch_rows: usize,
    bucketing: Option<Arc<Bucketing>>,
    prune: PruneStats,
}

/// A combined driver's state, in combine order.
#[derive(Debug)]
enum Part {
    Run(Runs),
    Buffer(Buffer),
}

/// An in-memory result, spillable until read (see
/// [`crate::output_spill`]): the buffers, and per bucket its merge between
/// the reader's steps. A spill writes the rest of each bucket's merge as a
/// run of its own.
struct Held {
    state: Mutex<HeldState>,
    /// Notified when a reader's step or a spill gives a merge back.
    changed: Event,
    /// The hook and the scope a spill writes to (with a spill scope only).
    spill: Option<(Arc<OutputSpill>, SpillScope)>,
    schema: SchemaRef,
    encoder: Arc<KeyEncoder>,
    /// The result's consumer: the buffers are charged to it.
    handle: Arc<ConsumerHandle>,
    counters: Arc<Counters>,
    query: Arc<QueryContext>,
    executor: Arc<dyn Executor>,
    batch_rows: usize,
}

struct HeldState {
    /// The buffered batches the in-memory buckets read, and the buffers
    /// accounting them: dropped once no bucket reads them.
    batches: Arc<[Arc<SortedBatch>]>,
    buffers: Vec<Buffer>,
    buckets: Vec<Bucket>,
    /// The next bucket a reader claims.
    next: usize,
    /// A spill waits for the steps in progress or writes: readers wait.
    spilling: bool,
    /// The merge was lost (a failed spill or step): every reader fails
    /// with the error, never with a short result.
    failed: Option<KrishivError>,
}

/// One bucket of a [`Held`] result.
enum Bucket {
    /// Not opened yet.
    Unread,
    /// The merge between the reader's steps.
    Merging(Box<Merger>),
    /// A reader's step or a spill holds the merge.
    Busy,
    /// The rest of the merge, spilled: the reader takes the run.
    Spilled(SpillRun),
    Done,
}

impl HeldState {
    fn failed(e: KrishivError) -> Self {
        Self {
            batches: Arc::new([]),
            buffers: Vec::new(),
            buckets: Vec::new(),
            next: 0,
            spilling: false,
            failed: Some(e),
        }
    }

    /// Whether an in-memory bucket still reads the buffers.
    fn reads_buffers(&self) -> bool {
        self.buckets
            .iter()
            .any(|b| matches!(b, Bucket::Unread | Bucket::Merging(_) | Bucket::Busy))
    }
}

/// What a reader's next step on its bucket is.
enum Take {
    /// Open the merge of the bucket of these batches.
    Open(Arc<[Arc<SortedBatch>]>),
    /// Continue the merge.
    Merge(Box<Merger>),
    /// Read the rest from this run.
    Run(SpillRun),
    /// A spill holds the result: wait for it.
    Wait,
    /// The bucket is read.
    Done,
}

impl Held {
    fn set_spillable(&self, bytes: usize) {
        if let Some((hook, _)) = &self.spill {
            hook.set_spillable_bytes(bytes);
        }
    }

    /// The next bucket for a reader, if any is left; the error once the
    /// result was lost.
    fn claim(&self) -> Result<Option<usize>> {
        let mut s = lock(&self.state);
        if let Some(e) = &s.failed {
            return Err(e.clone());
        }
        if s.next >= s.buckets.len() {
            return Ok(None);
        }
        s.next += 1;
        Ok(Some(s.next - 1))
    }

    /// A reader's next step on bucket `b` (leaving it `Busy` for a merge).
    fn take(&self, b: usize) -> Result<Take> {
        let mut s = lock(&self.state);
        if let Some(e) = &s.failed {
            return Err(e.clone());
        }
        if s.spilling {
            return Ok(Take::Wait);
        }
        let batches = Arc::clone(&s.batches);
        let Some(bucket) = s.buckets.get_mut(b) else {
            return Err(internal(format!("sort bucket {b} out of range")));
        };
        Ok(match mem::replace(bucket, Bucket::Busy) {
            Bucket::Unread => Take::Open(batches),
            Bucket::Merging(m) => Take::Merge(m),
            Bucket::Spilled(run) => {
                *bucket = Bucket::Done;
                Take::Run(run)
            }
            // A spill holds it.
            Bucket::Busy => Take::Wait,
            Bucket::Done => {
                *bucket = Bucket::Done;
                Take::Done
            }
        })
    }

    /// Ends a step on bucket `b`: `next` is its new state. The buffers go
    /// once no bucket reads them.
    fn put(&self, b: usize, next: Bucket) {
        let freed = {
            let mut s = lock(&self.state);
            if let Some(bucket) = s.buckets.get_mut(b) {
                *bucket = next;
            }
            if s.reads_buffers() {
                None
            } else {
                let batches = mem::replace(&mut s.batches, Arc::new([]));
                Some((batches, mem::take(&mut s.buffers)))
            }
        };
        if let Some(freed) = freed {
            self.set_spillable(0);
            drop(freed);
        }
        self.changed.notify();
    }

    /// Fails every reader with `e`; the buffers and merges go.
    fn fail(&self, e: KrishivError) {
        let old = mem::replace(&mut *lock(&self.state), HeldState::failed(e));
        self.set_spillable(0);
        drop(old);
        self.changed.notify();
    }
}

/// A bucket's merge taken by a spill, to write: the merge, or `None` to
/// open it.
type Unwritten = (usize, Option<Box<Merger>>);

impl SpillUnread for Held {
    fn spill_unread(self: Arc<Self>) -> BoxFuture<'static, Result<usize>> {
        Box::pin(async move {
            let Some((_, scope)) = &self.spill else {
                return Ok(0);
            };
            // Readers stop taking steps; wait for the steps in progress.
            let mut mine = false;
            let (work, batches, mut buffers) = loop {
                let changed = self.changed.listen();
                {
                    let mut s = lock(&self.state);
                    if s.failed.is_some() || (s.spilling && !mine) || !s.reads_buffers() {
                        if mine {
                            s.spilling = false;
                            drop(s);
                            self.changed.notify();
                        }
                        return Ok(0);
                    }
                    s.spilling = true;
                    mine = true;
                    if !s.buckets.iter().any(|b| matches!(b, Bucket::Busy)) {
                        let mut work: Vec<Unwritten> = Vec::new();
                        for (i, b) in s.buckets.iter_mut().enumerate() {
                            match mem::replace(b, Bucket::Busy) {
                                Bucket::Unread => work.push((i, None)),
                                Bucket::Merging(m) => work.push((i, Some(m))),
                                other => *b = other,
                            }
                        }
                        let batches = Arc::clone(&s.batches);
                        break (work, batches, mem::take(&mut s.buffers));
                    }
                }
                changed.await;
            };
            // The buffers' headroom is lent to the writers' scratch (as
            // grant credit, so no other consumer takes it meanwhile). The
            // first writer first: if its scratch is refused, nothing was
            // merged yet and every merge goes back as it was.
            let mut freed = 0;
            for b in &mut buffers {
                let lent = b.headroom.size();
                freed += lent + b.reservation.size();
                b.headroom.shrink_to_credit(lent);
            }
            let scratch = self.handle.reservation().for_spill_scratch();
            let writer = match scope.writer(Arc::clone(&self.schema), scratch) {
                Ok(w) => w,
                Err(e) => {
                    self.handle.release_credit();
                    self.give_back(work, buffers);
                    return refused(e.into());
                }
            };
            self.set_spillable(0);
            // Dropped before the spill ends (the query was cancelled): the
            // merges are gone with the blocking job, so readers fail rather
            // than wait on `Busy`.
            let unfinished = Unfinished(Some(&self.state), &self.changed, || {
                HeldState::failed(cancelled())
            });
            let row_bytes = buffers.iter().map(|b| b.row_bytes).max().unwrap_or(1);
            let rows = spill_rows(row_bytes, self.batch_rows);
            let query = Arc::clone(&self.query);
            let guard = self.query.blocking_started();
            let (schema, encoder) = (Arc::clone(&self.schema), Arc::clone(&self.encoder));
            let (handle, scope) = (Arc::clone(&self.handle), scope.clone());
            let written = run_blocking(&*self.executor, move || {
                let out = write_buckets(
                    work, &batches, writer, &scope, &schema, &encoder, &handle, rows, &query,
                );
                // The guard travels with the runs (files), so the query
                // does not end before they are dropped.
                Ok((out, guard))
            })
            .await
            .and_then(|(out, _guard)| out);
            unfinished.disarm();
            self.handle.release_credit();
            match written {
                Ok((runs, rest)) => {
                    for (_, run) in &runs {
                        self.counters.runs.fetch_add(1, Ordering::Relaxed);
                        let bytes = run.file.bytes();
                        self.counters.bytes.fetch_add(bytes, Ordering::Relaxed);
                    }
                    let all = rest.is_empty();
                    {
                        let mut s = lock(&self.state);
                        for (b, run) in runs {
                            if let Some(bucket) = s.buckets.get_mut(b) {
                                *bucket = Bucket::Spilled(run);
                            }
                        }
                    }
                    if all {
                        // Every merge is on disk: the buffers go.
                        drop(buffers);
                        self.give_back(Vec::new(), Vec::new());
                        Ok(freed)
                    } else {
                        // The pool refused a later writer's scratch: the
                        // other buckets stay in memory, with the buffers.
                        self.give_back(rest, buffers);
                        Ok(0)
                    }
                }
                Err(e) => {
                    // Rows already merged are lost: fail the query before
                    // the readers can see the state, and the readers too.
                    self.query.fail(e.clone());
                    self.fail(e.clone());
                    Err(e)
                }
            }
        })
    }
}

impl Held {
    /// Ends a spill: the merges in `work` and `buffers` go back, readers
    /// continue.
    fn give_back(&self, work: Vec<Unwritten>, buffers: Vec<Buffer>) {
        let bytes: usize = buffers.iter().map(|b| b.reservation.size()).sum();
        {
            let mut s = lock(&self.state);
            for (b, merger) in work {
                if let Some(bucket) = s.buckets.get_mut(b) {
                    *bucket = merger.map_or(Bucket::Unread, Bucket::Merging);
                }
            }
            s.buffers.extend(buffers);
            s.spilling = false;
            if !s.reads_buffers() {
                s.batches = Arc::new([]);
            }
        }
        self.set_spillable(bytes);
        self.changed.notify();
    }
}

/// The runs a spill wrote (by bucket) and the merges it left in memory.
type Written = (Vec<(usize, SpillRun)>, Vec<Unwritten>);

/// Writes the rest of each bucket's merge in `work` as a run of its own
/// (blocking I/O): `writer` for the first, a new one for each next. Returns
/// the runs and, when the pool refused a later writer's scratch, the
/// merges not written.
#[expect(clippy::too_many_arguments, reason = "one blocking job's inputs")]
fn write_buckets(
    work: Vec<Unwritten>,
    batches: &[Arc<SortedBatch>],
    writer: krishiv_memory::SpillWriter,
    scope: &SpillScope,
    schema: &SchemaRef,
    encoder: &Arc<KeyEncoder>,
    handle: &Arc<ConsumerHandle>,
    rows: usize,
    query: &QueryContext,
) -> Result<Written> {
    let mut runs = Vec::with_capacity(work.len());
    let mut first = Some(writer);
    let mut work = work.into_iter();
    while let Some((b, merger)) = work.next() {
        let writer = match first.take() {
            Some(w) => w,
            None => {
                match scope.writer(Arc::clone(schema), handle.reservation().for_spill_scratch()) {
                    Ok(w) => w,
                    Err(krishiv_memory::SpillError::Memory(_)) => {
                        let mut rest = vec![(b, merger)];
                        rest.extend(work);
                        return Ok((runs, rest));
                    }
                    Err(e) => return Err(e.into()),
                }
            }
        };
        let mut merger = match merger {
            Some(m) => m,
            None => Box::new(Merger::bucket(
                Arc::clone(schema),
                Arc::clone(encoder),
                batches,
                b,
                handle,
            )?),
        };
        runs.push((b, write_rest(&mut merger, writer, rows, query)?));
    }
    Ok((runs, Vec::new()))
}

#[derive(Default)]
struct SortShared {
    parts: Vec<Part>,
    /// The in-memory result, read by every reader.
    held: Option<Arc<Held>>,
    /// The spilled result: each bucket's runs, claimed by the readers.
    spilled: Vec<Vec<SpillRun>>,
}

struct SortGlobal {
    shared: Mutex<SortShared>,
    /// The query's bucketing (none: one bucket).
    bucketing: Option<Arc<Bucketing>>,
    buckets: usize,
}

impl Sort {
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
        handle: &Arc<ConsumerHandle>,
        ctx: &TaskContext,
    ) -> Result<Result<SpillRun, MemoryError>> {
        let scope = self.spill.clone();
        let scope = scope.ok_or_else(|| internal("a sort spilled without a scope".to_owned()))?;
        let (schema, encoder) = (Arc::clone(&self.schema), Arc::clone(&self.encoder));
        let (h, query) = (Arc::clone(handle), Arc::clone(ctx.query()));
        let written = ctx
            .run_blocking(move || {
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
    async fn reduce(
        &self,
        runs: &mut Vec<SpillRun>,
        batch_rows: usize,
        ctx: &TaskContext,
    ) -> Result<()> {
        if runs.len() <= 2 {
            return Ok(());
        }
        let handle = Arc::clone(ctx.consumer().consumer());
        let (mut budget, fan_in) =
            merge_budget(runs, self.max_fan_in, batch_rows, &handle, true).await?;
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
                let merged = self.write(group, rows, &handle, ctx);
                let run = lending_for_write(&mut budget, &handle, target, merged).await?;
                runs.push(run.map_err(KrishivError::from)?);
            }
        }
        Ok(())
    }
}

/// Runs to merge next when `total` remain at a level (more than `fan_in`):
/// a group of `g` leaves `g - 1` fewer, so never more than it takes to get
/// down to the fan-in, and as few bytes as possible are written again.
pub(super) fn group_size(total: usize, fan_in: usize) -> usize {
    fan_in.min(total.saturating_sub(fan_in) + 1)
}

/// Reserves (with arbitration) the memory to merge up to `max_fan_in` of
/// `runs` at once. An intermediate merge (`writing` a run) halves the
/// fan-in down to 2 while the pool refuses and also reserves the spill
/// writer's scratch; the final merge needs every run.
pub(super) async fn merge_budget(
    runs: &[SpillRun],
    max_fan_in: usize,
    batch_rows: usize,
    handle: &Arc<ConsumerHandle>,
    writing: bool,
) -> Result<(Reservation, usize)> {
    let per_run = runs.iter().map(SpillRun::merge_bytes).max().unwrap_or(0);
    let row_bytes = runs.iter().map(|r| r.row_bytes).max().unwrap_or(1);
    let out = if writing {
        headroom(row_bytes, batch_rows)
    } else {
        row_bytes.saturating_mul(batch_rows)
    };
    let mut fan_in = max_fan_in.min(runs.len());
    let mut budget = handle.reservation();
    loop {
        // The merger's per-source `slot`/`stamp` metadata (see
        // [`super::merge::metadata_bytes`]): small next to the frames, but
        // reserved with them so the merge never grows past its budget.
        let meta = super::merge::metadata_bytes(fan_in);
        let need = per_run
            .saturating_mul(fan_in)
            .saturating_add(out)
            .saturating_add(meta);
        match budget.grow(need).await {
            Ok(()) => return Ok((budget, fan_in)),
            Err(_) if writing && fan_in > 2 => fan_in = (fan_in / 2).max(2),
            Err(e) => return Err(e.into()),
        }
    }
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
            bucketing,
            buckets,
        }))
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        let g = global_ref::<SortGlobal>(global, "Sort")?;
        let memory = ctx.consumer();
        Ok(Box::new(SortLocal {
            buffer: Buffer::new(memory.consumer()),
            runs: Vec::new(),
            signal: Arc::clone(memory.signal()),
            batch_rows: ctx.query().config().batch_size.max(1),
            bucketing: g.bucketing.clone(),
            prune: PruneStats::default(),
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
        // A small slice of a large batch would keep (and account) the whole
        // batch: keep a copy of its rows instead.
        let input = if input.data().get_array_memory_size() > 2 * data + SLICE_SLACK_BYTES {
            let rows: Vec<(usize, usize)> = (0..rows).map(|r| (0, r)).collect();
            interleave_batches(&input.schema(), &[input], &rows)?
        } else {
            input
        };
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
            l.signal.set_spillable_bytes(b.reservation.size());
        }
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        let g = global_ref::<SortGlobal>(global, "Sort")?;
        let l = local
            .downcast::<SortLocal>()
            .map_err(|_| internal("wrong local state for Sort".to_owned()))?;
        let SortLocal {
            buffer,
            runs,
            signal,
            ..
        } = *l;
        let mut shared = lock(&g.shared);
        shared.parts.extend(runs.into_iter().map(Part::Run));
        if !buffer.batches.is_empty() {
            shared.parts.push(Part::Buffer(buffer));
        }
        // The last driver to combine finalizes: its signal offers the whole
        // global state to `spill_global` meanwhile.
        if self.spill.is_some() {
            signal.set_spillable_bytes(buffered(&shared.parts));
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
            // holds only the merges' frames.
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
                            // with arbitration this time, and try again.
                            let more = headroom(b.row_bytes, batch_rows);
                            b.headroom.grow(more).await?;
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
            for runs in &mut spilled {
                self.reduce(runs, batch_rows, ctx).await?;
            }
            spilled.retain(|runs| !runs.is_empty());
            // Readers claim from the end: the first bucket first.
            spilled.reverse();
            lock(&g.shared).spilled = spilled;
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
            let before = l.buffer.reservation.size();
            if l.buffer.batches.is_empty() || self.spill.is_none() {
                return Ok(0);
            }
            let buckets = l.bucketing.as_ref().map_or(1, |b| b.buckets());
            let Ok(runs) = self
                .spill_buffer(&mut l.buffer, buckets, l.batch_rows, ctx)
                .await?
            else {
                // Refused scratch memory: nothing freed this time.
                return Ok(0);
            };
            l.runs.push(runs);
            l.signal.set_spillable_bytes(0);
            Ok(before)
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
            // Out of the lock while spilling; `finalize` is not polled
            // meanwhile, and an error fails the query (the parts go).
            let mut parts = mem::take(&mut lock(&g.shared).parts);
            let batch_rows = ctx.query().config().batch_size.max(1);
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
            ctx.consumer()
                .signal()
                .set_spillable_bytes(buffered(&shared.parts));
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
    query: Arc<QueryContext>,
    executor: Arc<dyn Executor>,
    handle: Arc<ConsumerHandle>,
    batch_rows: usize,
}

/// A merge of spilled runs, holding the budget for the runs' frames.
struct Merging {
    merger: Box<Merger>,
    budget: Reservation,
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
                        l.state = match held {
                            Some(held) => SourceState::Held(held, None),
                            None => SourceState::Spilled,
                        };
                    }
                    SourceState::Held(held, bucket) => {
                        let next = match bucket {
                            Some(b) => Some(b),
                            None => held.claim()?,
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
                                let merging = self.open_runs(l, vec![run]).await?;
                                l.state = SourceState::Merging(merging, After::Held(held));
                            }
                        }
                    }
                    SourceState::Spilled => {
                        let runs = lock(&g.shared).spilled.pop();
                        if let Some(runs) = runs {
                            let merging = self.open_runs(l, runs).await?;
                            l.state = SourceState::Merging(merging, After::Spilled);
                        }
                    }
                    SourceState::Merging(
                        Merging {
                            mut merger,
                            mut budget,
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
                            l.state = SourceState::Merging(Merging { merger, budget }, after);
                            return Ok(Some(batch));
                        }
                        // Spilled runs' last frames went back as credit.
                        drop((merger, budget));
                        l.handle.release_credit();
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
    /// Opens the final merge of spilled `runs`.
    async fn open_runs(&self, l: &SortSource, runs: Vec<SpillRun>) -> Result<Merging> {
        let (schema, encoder) = (Arc::clone(&self.schema), Arc::clone(&self.encoder));
        let handle = Arc::clone(&l.handle);
        let mut budget = merge_budget(&runs, runs.len(), l.batch_rows, &handle, false)
            .await?
            .0;
        self.counters
            .fan_in
            .fetch_max(runs.len() as u64, Ordering::Relaxed);
        let open = blocking(&l.query, &l.executor, move || {
            Merger::new(schema, encoder, Vec::new(), &runs, &handle)
        });
        let merger = lending(&mut budget, &l.handle, 0, open).await?;
        Ok(Merging {
            merger: Box::new(merger),
            budget,
        })
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
    let rows = l.batch_rows;
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
        max_batch_bytes = max_batch_bytes.max(batch.data().get_array_memory_size());
        row_bytes = row_bytes.max(data_bytes(&batch).div_ceil(batch.num_rows().max(1)));
        writer.write(&batch).map_err(KrishivError::from)?;
    }
    let file = writer.finish().map_err(KrishivError::from)?;
    Ok(SpillRun {
        file,
        max_batch_bytes,
        row_bytes,
    })
}
