// SPDX-License-Identifier: Apache-2.0
//! Per-query and per-task context: configuration, cancellation, the first
//! error, the query's memory pool and its per-operator consumers.

use std::fmt;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use krishiv_common::config::{Config, EXEC_BATCH_SIZE};
use krishiv_common::error::KrishivError;
use krishiv_memory::{
    ConsumerHandle, ConsumerId, MemoryPool, Reservation, SpillOwner, SpillSignal,
};

use crate::cancel::CancellationToken;
use crate::error::{Result, cancelled};
use crate::executor::{Executor, WorkloadClass, run_blocking};
use crate::metrics::QueryMetrics;
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
    /// counting it (tests and CI).
    pub fail_on_stall: bool,
    /// Seeds morsel order (deterministic test mode); `None` claims morsels
    /// in order.
    pub seed: Option<u64>,
}

impl Default for ExecConfig {
    fn default() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
            morsel_rows: DEFAULT_MORSEL_ROWS,
            result_buffer_per_slot: 2,
            stall_threshold: Duration::from_millis(10),
            fail_on_stall: false,
            seed: None,
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
        Self {
            batch_size,
            ..Self::default()
        }
    }
}

/// Identifies a query.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct QueryId(pub u64);

/// State shared by every task of one query.
pub struct QueryContext {
    id: QueryId,
    class: WorkloadClass,
    config: ExecConfig,
    pool: Arc<dyn MemoryPool>,
    cancel: CancellationToken,
    /// The first error; later ones (mostly "cancelled") are dropped.
    error: Mutex<Option<KrishivError>>,
    metrics: Mutex<QueryMetrics>,
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
            metrics: Mutex::new(QueryMetrics::default()),
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

    /// `Err(OPERATION_CANCELED)` once cancelled or failed.
    ///
    /// # Errors
    /// `OPERATION_CANCELED` once the query was cancelled or failed.
    pub fn check_cancelled(&self) -> Result<()> {
        if self.is_cancelled() {
            return Err(cancelled());
        }
        Ok(())
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
        lock(&self.metrics).clone()
    }

    pub(crate) fn merge_metrics(&self, f: impl FnOnce(&mut QueryMetrics)) {
        f(&mut lock(&self.metrics));
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
        }
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
            let consumer = ConsumerHandle::register(
                Arc::clone(self.query.pool()),
                self.name,
                Some(Arc::clone(&signal) as Arc<dyn krishiv_memory::Spillable>),
            );
            MemoryHandle { consumer, signal }
        })
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
