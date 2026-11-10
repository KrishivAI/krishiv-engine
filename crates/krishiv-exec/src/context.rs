// SPDX-License-Identifier: Apache-2.0
//! Per-query and per-task context: configuration, cancellation, the first
//! error, the query's memory pool and its per-operator consumers.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use krishiv_common::config::{
    Config, EXEC_AGG_ROW_LAYOUT, EXEC_BATCH_SIZE, EXEC_JOIN_RADIX_BITS, EXEC_JOIN_ROW_LAYOUT,
    EXEC_JOIN_SHARED_BROADCAST_BYTES, EXEC_JOIN_SMJ_BUFFER_BYTES, EXEC_SEED,
    EXEC_SORT_HASH_BUCKETS,
};
use krishiv_common::error::KrishivError;
use krishiv_memory::{
    ConsumerHandle, ConsumerId, MemoryPool, Reservation, SpillOwner, SpillSignal, SpillStats,
};

use crate::cancel::CancellationToken;
use crate::error::{Result, cancelled};
use crate::executor::{Executor, WorkloadClass, run_blocking};
use crate::metrics::{PipelineMetrics, QueryMemory, QueryMetrics};
use crate::operator::GlobalState;
use crate::sync::{AtomicBool, AtomicUsize, Event, Mutex, Ordering, lock};

/// Rows per batch (`krishiv.exec.batchSize`; ADR 0034, spec 03, E01;
/// provisional until E19).
pub const DEFAULT_BATCH_SIZE: usize = 4096;
/// Rows per morsel: the unit drivers claim from a source (ADR 0006 §2).
pub const DEFAULT_MORSEL_ROWS: usize = 128 * 1024;

/// Execution settings of one query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecConfig {
    /// Rows per batch.
    pub batch_size: usize,
    /// Rows per morsel.
    pub morsel_rows: usize,
    /// Result batches buffered per worker slot before the result sink
    /// blocks its drivers (backpressure).
    pub result_buffer_per_slot: usize,
    /// A driver poll longer than this is a stall (E01: ≈ 1 ms of work per
    /// batch; provisional 10 ms, recorded with the batch size).
    pub stall_threshold: Duration,
    /// Fail the query with `INTERNAL_ERROR` on a stall instead of only
    /// counting it (tests and CI). A slice then leaves out the time its
    /// thread waited on a run queue, so machine load cannot make a stall.
    pub fail_on_stall: bool,
    /// Seeds morsel order (deterministic test mode); `None` claims morsels
    /// in order.
    pub seed: Option<u64>,
    /// Grouped aggregations with fixed-width keys use row-layout,
    /// radix-partitioned tables (`krishiv.exec.agg.rowLayout`; P2-T159).
    pub agg_row_layout: bool,
    /// Collect the costlier metrics of a detailed profile (`EXPLAIN
    /// ANALYZE`): per-lookup hash-table probe lengths. The always-on
    /// counters do not depend on it.
    pub detailed_metrics: bool,
    /// Hash joins use the row layout and its size-adaptive index
    /// (`krishiv.exec.join.rowLayout`; P2-T158).
    pub join_row_layout: bool,
    /// Level-0 radix bits of a row-layout join build
    /// (`krishiv.exec.join.radixBits`, 3 to 6).
    pub join_radix_bits: u32,
    /// A sort below a window splits its rows into hash buckets read in
    /// parallel (`krishiv.exec.sort.hashBuckets`; P2-T055).
    pub sort_hash_buckets: bool,
    /// The bytes of one key's right-side rows a sort-merge join keeps in
    /// memory (`krishiv.exec.join.smj.bufferBytes`; 0: a share of the pool).
    pub join_merge_buffer_bytes: usize,
    /// The most one query's shared broadcast tables may hold on a worker
    /// (`krishiv.exec.join.sharedBroadcastBytes`; `None`: no cap beyond the
    /// query's memory). A build past it falls back to one per task.
    pub shared_broadcast_bytes: Option<usize>,
    /// How long a finished hash join build waits for its probe before the
    /// arbitrator asks it to spill ahead of running consumers
    /// (`krishiv.exec.join.idleSpillAfter`).
    pub join_idle_spill_after: Duration,
    /// Every pipeline runs on one driver, so the order of merges and of the
    /// output rows does not depend on how the drivers interleave, and the key
    /// hashes take a fixed seed (a constant unless
    /// `krishiv.exec.seed` names one) (`krishiv.deterministicOrder`).
    pub deterministic_order: bool,
}

/// The key-hash seed of `krishiv.deterministicOrder` when `krishiv.exec.seed`
/// is not set: any fixed value; it is never observable (Spark's observable
/// hash is Murmur3, in the kernels).
pub(crate) const DETERMINISTIC_SEED: u64 = 0x6b72_6973_6869_7601;

impl Default for ExecConfig {
    fn default() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
            morsel_rows: DEFAULT_MORSEL_ROWS,
            result_buffer_per_slot: 2,
            stall_threshold: Duration::from_millis(10),
            fail_on_stall: false,
            seed: None,
            agg_row_layout: true,
            detailed_metrics: false,
            join_row_layout: true,
            join_radix_bits: 3,
            sort_hash_buckets: true,
            join_merge_buffer_bytes: 0,
            shared_broadcast_bytes: None,
            join_idle_spill_after: Duration::from_secs(1),
            deterministic_order: false,
        }
    }
}

impl ExecConfig {
    /// Reads the `krishiv.exec.*` entries, falling back to the defaults.
    pub fn from_config(config: &Config) -> Self {
        let batch_size = config
            .get(&EXEC_BATCH_SIZE)
            .and_then(krishiv_common::config::ConfigValue::as_int)
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| *n > 0)
            .unwrap_or(DEFAULT_BATCH_SIZE);
        let agg_row_layout = config
            .get(&EXEC_AGG_ROW_LAYOUT)
            .and_then(krishiv_common::config::ConfigValue::as_bool)
            .unwrap_or(true);
        let join_row_layout = config
            .get(&EXEC_JOIN_ROW_LAYOUT)
            .and_then(krishiv_common::config::ConfigValue::as_bool)
            .unwrap_or(true);
        let join_radix_bits = config
            .get(&EXEC_JOIN_RADIX_BITS)
            .and_then(krishiv_common::config::ConfigValue::as_int)
            .and_then(|n| u32::try_from(n.clamp(3, 6)).ok())
            .unwrap_or(3);
        let sort_hash_buckets = config
            .get(&EXEC_SORT_HASH_BUCKETS)
            .and_then(krishiv_common::config::ConfigValue::as_bool)
            .unwrap_or(true);
        let join_merge_buffer_bytes = config
            .get(&EXEC_JOIN_SMJ_BUFFER_BYTES)
            .and_then(krishiv_common::config::ConfigValue::as_size)
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(0);
        let shared_broadcast_bytes = config
            .get(&EXEC_JOIN_SHARED_BROADCAST_BYTES)
            .and_then(krishiv_common::config::ConfigValue::as_size)
            .and_then(|n| usize::try_from(n).ok())
            .filter(|n| *n > 0);
        let join_idle_spill_after = config
            .get(&krishiv_common::config::EXEC_JOIN_IDLE_SPILL_AFTER)
            .and_then(krishiv_common::config::ConfigValue::as_duration)
            .unwrap_or(Duration::from_secs(1));
        let deterministic_order = config
            .get(&krishiv_common::config::DETERMINISTIC_ORDER)
            .and_then(krishiv_common::config::ConfigValue::as_bool)
            .unwrap_or(false);
        // Under `krishiv.deterministicOrder` the key hashes are seeded too
        // (`krishiv.exec.seed` when set, else a fixed seed): spill partitions
        // follow the seeded hash, so a random seed per query would order the
        // groups of a spilled aggregation differently on every run.
        let seed = config
            .get(&EXEC_SEED)
            .and_then(krishiv_common::config::ConfigValue::as_int)
            .map(i64::cast_unsigned)
            .or(deterministic_order.then_some(DETERMINISTIC_SEED));
        Self {
            batch_size,
            seed,
            deterministic_order,
            sort_hash_buckets,
            agg_row_layout,
            join_row_layout,
            join_radix_bits,
            join_merge_buffer_bytes,
            shared_broadcast_bytes,
            join_idle_spill_after,
            ..Self::default()
        }
    }
}

/// Identifies a query.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct QueryId(pub u64);

/// The scans' decode working sets may hold `1 / SCAN_GATE_SHARE` of the
/// query's memory limit at once.
const SCAN_GATE_SHARE: usize = 4;

/// State shared by every task of one query.
pub struct QueryContext {
    id: QueryId,
    class: WorkloadClass,
    config: ExecConfig,
    pool: Arc<dyn MemoryPool>,
    cancel: CancellationToken,
    /// The first error; later ones (mostly "cancelled") are dropped.
    error: Mutex<Option<KrishivError>>,
    /// The error of a cancellation with a reason ([`Self::cancel_with`]),
    /// set only when it was the query's first error: drivers end with it
    /// instead of a plain `OPERATION_CANCELED`.
    cancel_error: OnceLock<KrishivError>,
    /// When the query was cancelled with a reason ([`Self::cancel_with`]):
    /// a dropped stream or a failure is not a cancellation to time.
    cancelled_at: OnceLock<Instant>,
    metrics: Mutex<QueryMetrics>,
    /// When execution started (the runner sets it): pipeline times are
    /// offsets from it.
    epoch: OnceLock<Instant>,
    /// Per pipeline, the threads that polled its drivers (worker-local, to
    /// count them; [`QueryMetrics`] keeps only the count).
    threads: Mutex<BTreeMap<u32, BTreeSet<u64>>>,
    /// Spill files written per operator, shared by its consumers in every
    /// driver (read into the metrics by [`QueryContext::metrics`]).
    spills: Mutex<BTreeMap<OperatorId, Arc<SpillStats>>>,
    /// The scans' decode working sets (`krishiv_io::ScanGate`), made on
    /// first use with a budget of a quarter of the query's memory limit.
    scan_gate: OnceLock<Arc<krishiv_io::ScanGate>>,
    finished: AtomicBool,
    finished_event: Event,
    /// Blocking closures in flight; the query ends only once they have.
    blocking: AtomicUsize,
    blocking_idle: Event,
}

impl fmt::Debug for QueryContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueryContext")
            .field("id", &self.id)
            .field("class", &self.class)
            .field("cancelled", &self.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl QueryContext {
    /// A query context. `pool` is the query's memory pool (a child of the
    /// process root).
    pub fn new(
        id: QueryId,
        class: WorkloadClass,
        config: ExecConfig,
        pool: Arc<dyn MemoryPool>,
    ) -> Arc<Self> {
        Arc::new(Self {
            id,
            class,
            config,
            pool,
            cancel: CancellationToken::new(),
            error: Mutex::new(None),
            cancel_error: OnceLock::new(),
            cancelled_at: OnceLock::new(),
            metrics: Mutex::new(QueryMetrics::default()),
            epoch: OnceLock::new(),
            threads: Mutex::new(BTreeMap::new()),
            spills: Mutex::new(BTreeMap::new()),
            scan_gate: OnceLock::new(),
            finished: AtomicBool::new(false),
            finished_event: Event::default(),
            blocking: AtomicUsize::new(0),
            blocking_idle: Event::default(),
        })
    }

    /// The query id.
    pub fn id(&self) -> QueryId {
        self.id
    }

    /// The workload class.
    pub fn class(&self) -> WorkloadClass {
        self.class
    }

    /// The execution settings.
    pub fn config(&self) -> &ExecConfig {
        &self.config
    }

    /// The query's memory pool.
    pub fn pool(&self) -> &Arc<dyn MemoryPool> {
        &self.pool
    }

    /// Cancels the query (`OPERATION_CANCELED`, unless it already failed or
    /// finished). Drivers stop at their next batch boundary or await.
    pub fn cancel(&self) {
        if !self.is_finished() {
            self.fail(cancelled());
        }
    }

    /// Cancels the query with `error`, a cancellation's own error (a
    /// timeout, an operator's kill): unless the query already finished or
    /// failed, it becomes the query's error and every driver's, so the
    /// result stream ends with it. Returns whether this call cancelled it
    /// (the first cancellation wins; later ones are no-ops).
    pub fn cancel_with(&self, error: KrishivError) -> bool {
        if self.is_finished() {
            return false;
        }
        let first = {
            let mut slot = lock(&self.error);
            if slot.is_none() {
                // Before the error is visible: a driver that sees the
                // cancellation finds the reason.
                let _ = self.cancel_error.set(error.clone());
                *slot = Some(error);
                true
            } else {
                false
            }
        };
        if first {
            self.cancelled_at.get_or_init(Instant::now);
        }
        self.cancel.cancel();
        first
    }

    /// When the query was cancelled with a reason, if it was.
    pub fn cancelled_at(&self) -> Option<Instant> {
        self.cancelled_at.get().copied()
    }

    /// Fails the query with `error` unless an earlier error won, then
    /// cancels every task.
    pub fn fail(&self, error: KrishivError) {
        {
            let mut slot = lock(&self.error);
            if slot.is_none() {
                *slot = Some(error);
            }
        }
        self.cancel.cancel();
    }

    /// The error the query failed with, if any.
    pub fn error(&self) -> Option<KrishivError> {
        lock(&self.error).clone()
    }

    /// Whether the query was cancelled or failed (one atomic load).
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// `Err` once cancelled or failed: the reason's error of a
    /// [`Self::cancel_with`] cancellation, else `OPERATION_CANCELED`.
    ///
    /// # Errors
    /// As above, once the query was cancelled or failed.
    pub fn check_cancelled(&self) -> Result<()> {
        if self.is_cancelled() {
            return Err(self.cancellation());
        }
        Ok(())
    }

    /// The error a driver stopped by cancellation ends with: the reason's
    /// error of a [`Self::cancel_with`] cancellation, else
    /// `OPERATION_CANCELED`.
    pub(crate) fn cancellation(&self) -> KrishivError {
        self.cancel_error.get().cloned().unwrap_or_else(cancelled)
    }

    /// Completes once the query is cancelled or failed.
    pub async fn cancelled(&self) {
        self.cancel.cancelled().await;
    }

    /// Whether every task of the query has ended and released its memory.
    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }

    /// Completes once [`is_finished`](Self::is_finished).
    pub async fn finished(&self) {
        let listener = self.finished_event.listen();
        if self.is_finished() {
            return;
        }
        listener.await;
    }

    /// Completes once no blocking closure of the query is running (they
    /// may hold memory; a cancelled query's closures stop at their next
    /// cancellation check).
    pub(crate) async fn blocking_done(&self) {
        loop {
            let listener = self.blocking_idle.listen();
            if self.blocking.load(Ordering::Acquire) == 0 {
                return;
            }
            listener.await;
        }
    }

    /// Counts a blocking closure until the returned guard drops.
    pub(crate) fn blocking_started(self: &Arc<Self>) -> InFlight {
        self.blocking.fetch_add(1, Ordering::AcqRel);
        InFlight(Arc::clone(self))
    }

    pub(crate) fn mark_finished(&self) {
        self.finished.store(true, Ordering::Release);
        self.finished_event.notify();
    }

    /// Per-operator metrics for `EXPLAIN ANALYZE`, complete once the query
    /// finished (each driver merges its counters once, when it ends).
    pub fn metrics(&self) -> QueryMetrics {
        let mut m = lock(&self.metrics).clone();
        for (id, stats) in lock(&self.spills).iter() {
            let om = &mut m.operators.entry(*id).or_default().metrics;
            om.spill_files = stats.files();
            om.spill_write_bytes = stats.bytes();
        }
        m.memory = QueryMemory {
            limit: self.pool.effective_limit().map(|l| l as u64),
            peak_reserved: self.pool.peak_reserved() as u64,
        };
        m
    }

    /// The query's scan decode gate (see `krishiv_io::ScanGate`): its
    /// working sets may take up to a quarter of the query's memory limit at
    /// once, leaving the rest to operators that cannot wait for a scan.
    pub fn scan_gate(&self) -> &Arc<krishiv_io::ScanGate> {
        self.scan_gate.get_or_init(|| {
            let budget = self
                .pool
                .effective_limit()
                .map_or(usize::MAX, |l| l / SCAN_GATE_SHARE);
            Arc::new(krishiv_io::ScanGate::new(&self.pool, budget))
        })
    }

    /// Where `op`'s consumers count their spill files.
    pub(crate) fn spill_stats(&self, op: OperatorId) -> Arc<SpillStats> {
        Arc::clone(lock(&self.spills).entry(op).or_default())
    }

    pub(crate) fn merge_metrics(&self, f: impl FnOnce(&mut QueryMetrics)) {
        f(&mut lock(&self.metrics));
    }

    /// Marks the start of execution at `now` (once; later calls keep it).
    pub(crate) fn start_clock(&self, now: Instant) {
        let _ = self.epoch.set(now);
    }

    /// `now` as an offset from the start of execution.
    pub(crate) fn offset(&self, now: Instant) -> Duration {
        self.epoch
            .get()
            .map_or(Duration::ZERO, |e| now.saturating_duration_since(*e))
    }

    /// Records one driver of pipeline `pipeline`: it ran from `started`
    /// to `ended` (offsets) on `threads`, its longest poll `max_slice`, with
    /// `stalls` polls over the stall threshold.
    pub(crate) fn record_driver(
        &self,
        pipeline: u32,
        started: Duration,
        ended: Duration,
        threads: &[u64],
        (max_slice, stalls): (Duration, u64),
    ) {
        let count = {
            let mut all = lock(&self.threads);
            let set = all.entry(pipeline).or_default();
            set.extend(threads.iter().copied());
            set.len() as u64
        };
        self.merge_metrics(|m| {
            let p = m.pipelines.entry(pipeline).or_default();
            p.merge(&PipelineMetrics {
                drivers: 1,
                threads: 0,
                started,
                ended,
                max_slice,
                stalls,
            });
            // The set only grows, but a driver that read it earlier may
            // merge later: keep the larger count.
            p.threads = p.threads.max(count);
        });
    }
}

/// Identifies an operator (source, operator or sink) within a query: the
/// key of its metrics. Assigned by the
/// [`PipelineBuilder`](crate::PipelineBuilder).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct OperatorId(pub u32);

/// What one driver passes to one operator: the query, the executor (for
/// `spawn_blocking` and the I/O runtime) and the operator's memory
/// consumer.
pub struct TaskContext {
    query: Arc<QueryContext>,
    executor: Arc<dyn Executor>,
    owner: Arc<SpillOwner>,
    op: OperatorId,
    name: &'static str,
    memory: OnceLock<MemoryHandle>,
    dependency: Option<GlobalState>,
    second_dependency: Option<GlobalState>,
    /// The driver's scheduler-assigned globally unique partition id within
    /// its pipeline group (Spark's partition index for the partition the
    /// driver computes): stateful functions own one value stream per
    /// partition. The first driver of a group reads partition 0.
    partition: u32,
    /// The split the driver's source claimed last (see [`SplitCell`]).
    split: SplitCell,
}

/// The source split (morsel claim ordinal) a driver is reading, shared by the
/// driver and its operators. Seeded random functions draw their stream by the
/// split a row belongs to, as Spark draws by the partition of the input, not by
/// which driver happened to run it (drivers claim morsels dynamically).
#[derive(Debug, Clone, Default)]
pub struct SplitCell(Arc<std::sync::atomic::AtomicU64>);

impl SplitCell {
    const NONE: u64 = u64::MAX;

    /// A cell with no split yet.
    pub fn new() -> Self {
        Self(Arc::new(std::sync::atomic::AtomicU64::new(Self::NONE)))
    }

    /// Records the split the source claimed.
    pub(crate) fn set(&self, split: u64) {
        let split = u64::from(u32::try_from(split).unwrap_or(u32::MAX - 1));
        self.0.store(split, std::sync::atomic::Ordering::Relaxed);
    }

    /// The split being read, or `fallback` (the driver's partition) when the
    /// source reports none.
    pub fn get(&self, fallback: u32) -> u32 {
        match self.0.load(std::sync::atomic::Ordering::Relaxed) {
            Self::NONE => fallback,
            v => u32::try_from(v).unwrap_or(fallback),
        }
    }
}

impl fmt::Debug for TaskContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TaskContext")
            .field("query", &self.query.id())
            .field("op", &self.op)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl TaskContext {
    pub(crate) fn new(
        query: Arc<QueryContext>,
        executor: Arc<dyn Executor>,
        owner: Arc<SpillOwner>,
        op: OperatorId,
        name: &'static str,
    ) -> Self {
        Self {
            query,
            executor,
            owner,
            op,
            name,
            memory: OnceLock::new(),
            dependency: None,
            second_dependency: None,
            partition: 0,
            split: SplitCell::new(),
        }
    }

    /// The same context for the driver `partition` (its scheduler-assigned
    /// globally unique id within its pipeline group).
    #[must_use]
    pub(crate) fn with_partition(mut self, partition: u32) -> Self {
        self.partition = partition;
        self
    }

    /// The driver's scheduler-assigned partition id (set when the driver is
    /// spawned; the first driver of a group reads 0).
    pub fn partition(&self) -> u32 {
        self.partition
    }

    /// The driver's split cell: the source split it reads now.
    pub fn split_cell(&self) -> SplitCell {
        self.split.clone()
    }

    pub(crate) fn with_split(mut self, split: SplitCell) -> Self {
        self.split = split;
        self
    }

    pub(crate) fn with_dependency(mut self, state: GlobalState) -> Self {
        self.dependency = Some(state);
        self
    }

    /// For a breaker's source side (in `Source::global_state`): the
    /// finalized global state of its sink side (see
    /// [`PipelineBuilder::set_breaker_source`](crate::PipelineBuilder::set_breaker_source)).
    pub fn dependency(&self) -> Option<&GlobalState> {
        self.dependency.as_ref()
    }

    pub(crate) fn with_second_dependency(mut self, state: GlobalState) -> Self {
        self.second_dependency = Some(state);
        self
    }

    /// For a source over two breakers (a sort-merge join), the second one's
    /// finalized global state (see
    /// [`PipelineBuilder::set_merge_source`](crate::PipelineBuilder::set_merge_source)).
    pub fn second_dependency(&self) -> Option<&GlobalState> {
        self.second_dependency.as_ref()
    }

    /// The query.
    pub fn query(&self) -> &Arc<QueryContext> {
        &self.query
    }

    /// The executor.
    pub fn executor(&self) -> &Arc<dyn Executor> {
        &self.executor
    }

    /// Runs non-yielding work over ~1 ms on the bounded blocking pool and
    /// returns its result. Long closures check
    /// [`QueryContext::is_cancelled`] between chunks of ≤ 10 ms; the query
    /// does not end before its closures have.
    ///
    /// # Errors
    /// The closure's error, or `INTERNAL_ERROR` if it panicked.
    pub async fn run_blocking<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        let guard = self.query.blocking_started();
        // The guard travels with the result, so a result that holds memory
        // (a reservation) and is dropped because the caller went away (a
        // cancelled query) is dropped before the query can count the
        // closure as done (a tuple drops its fields in order).
        let (out, guard) = run_blocking(&*self.executor, move || f().map(|v| (v, guard))).await?;
        drop(guard);
        Ok(out)
    }

    /// The operator this context belongs to.
    pub fn op(&self) -> OperatorId {
        self.op
    }

    /// This operator's memory consumer in this driver, registered on first
    /// use with a spill signal of the driver's
    /// [`SpillOwner`](krishiv_memory::SpillOwner).
    pub fn consumer(&self) -> &MemoryHandle {
        self.memory.get_or_init(|| {
            let signal = self.owner.signal();
            let consumer = ConsumerHandle::register_counted(
                Arc::clone(self.query.pool()),
                self.name,
                Some(Arc::clone(&signal) as Arc<dyn krishiv_memory::Spillable>),
                self.query.spill_stats(self.op),
            );
            MemoryHandle { consumer, signal }
        })
    }

    /// A further consumer of this operator that never spills, for memory kept
    /// apart from the driver's consumer: grant credit is per consumer, so two
    /// merges on one would take each other's. It registers with a signal of
    /// the driver's owner that is dropped at once (no spillable bytes), so
    /// arbitration knows which driver it belongs to.
    pub(crate) fn register_plain_consumer(&self, name: &str) -> Arc<ConsumerHandle> {
        let signal = self.owner.signal();
        ConsumerHandle::register_counted(
            Arc::clone(self.query.pool()),
            name,
            Some(signal as Arc<dyn krishiv_memory::Spillable>),
            self.query.spill_stats(self.op),
        )
    }

    /// The consumer, if the operator registered one.
    pub(crate) fn registered(&self) -> Option<&MemoryHandle> {
        self.memory.get()
    }
}

/// One blocking closure in flight (dropped with the closure, or with the
/// waiting future if the closure never started).
pub(crate) struct InFlight(Arc<QueryContext>);

impl Drop for InFlight {
    fn drop(&mut self) {
        if self.0.blocking.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.blocking_idle.notify();
        }
    }
}

/// One operator's memory consumer in one driver: the consumer (with its
/// grant credit) and its spill signal. Operators make their
/// [`Reservation`]s here and keep the signal's spillable bytes current.
#[derive(Debug)]
pub struct MemoryHandle {
    consumer: Arc<ConsumerHandle>,
    signal: Arc<SpillSignal>,
}

impl MemoryHandle {
    /// The consumer id.
    pub fn id(&self) -> ConsumerId {
        self.consumer.id()
    }

    /// An empty reservation that takes the driver's grant credit first
    /// (see `NeedMemory`) and keeps the consumer registered while it lives,
    /// even after it moves into a breaker's global state.
    pub fn reservation(&self) -> Reservation {
        self.consumer.reservation()
    }

    /// The spill signal: publish spillable bytes here.
    pub fn signal(&self) -> &Arc<SpillSignal> {
        &self.signal
    }

    pub(crate) fn consumer(&self) -> &Arc<ConsumerHandle> {
        &self.consumer
    }
}
