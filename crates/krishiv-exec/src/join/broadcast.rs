// SPDX-License-Identifier: Apache-2.0
//! A broadcast hash join's build shared by the tasks of one worker (design
//! P2-T177 §2.5, unit 3).
//!
//! On one worker a broadcast exchange is lowered into the join over it, and
//! the join's drivers share its one build. When the exchange is cut into
//! stages, every task of the reading stage reads the whole broadcast; with
//! [`BroadcastTables`] on the worker, the first task that reaches the
//! join's build starts one build of the broadcast's rows for the job on
//! this worker, and every task of the job on the worker probes that table:
//! rows read once and a table built once per worker and query.
//!
//! - **Ownership.** The build runs as a query of its own on the resource's
//!   executor, over a child of the resource's pool (the job's pool on the
//!   worker, I11), with the first task's settings. Its table and memory stay
//!   until the resource is dropped (at the job's end); the tasks' own pools
//!   hold none of it.
//! - **No spill; per-task fallback.** The shared build has no spill scope
//!   (a spilled build's grace state holds one task's probe rows and cannot
//!   be shared), and its reservations never abort a query: when it does not
//!   fit (`UNABLE_TO_ACQUIRE_MEMORY`), every
//!   task of the job on the worker builds its own table from the broadcast's
//!   rows instead, with its own spill scope, as one task always did (owner
//!   decision 2026-11-06, row 30c (b)). Any other failure of the shared build
//!   fails the tasks waiting on it.
//! - **Immutable.** After `finalize` a build's tables are only read; joins
//!   that mark build rows (those that build their preserved side) never
//!   share one.
//! - **Runtime filters.** The shared build publishes into a value service
//!   of its own; each task republishes the finished filters as its own
//!   partial, so its scans see what its own build would have published.
//! - **Retries.** The build is keyed by the exchange and the attempt of the
//!   stage that wrote it, never by a task: a retried task reads the same
//!   table (I10).

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use arrow_schema::SchemaRef;
use futures::StreamExt as _;
use futures::future::BoxFuture;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_io::{DisabledReason, FilterId, FilterSnapshot};
use krishiv_memory::{ConsumerId, MemoryError, MemoryPool, PoolSnapshot, Spillable};
use krishiv_types::batch::Batch;

use super::build::BuildGlobal;
use super::{FilterPublisher, HashJoin, Shared};
use crate::ExecConfig;
use crate::context::{QueryContext, QueryId, TaskContext};
use crate::error::{Result, internal};
use crate::executor::{Executor, WorkloadClass};
use crate::operator::{GlobalState, LocalState, Sink, SinkStatus, Source};
use crate::physical::ExchangeId;
use crate::pipeline::{ExecNode, PipelineBuilder};
use crate::runner::QueryRunner;
use crate::sync::{Event, Mutex, lock};

/// Names one shared build: the broadcast exchange it reads and the attempt
/// of the stage that wrote it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BroadcastKey {
    /// The broadcast exchange.
    pub exchange: ExchangeId,
    /// The attempt of the stage that wrote it.
    pub stage_attempt: u32,
}

/// The broadcast builds of one job on one worker (see the module docs): a
/// live resource of the worker, never plan data.
pub struct BroadcastTables {
    executor: Arc<dyn Executor>,
    pool: Arc<dyn MemoryPool>,
    builds: Mutex<HashMap<BroadcastKey, Arc<Slot>>>,
    next: AtomicU64,
    /// Tasks that built their own table without attempting the shared one,
    /// because it could never fit ([`Decision`]).
    skipped: AtomicU64,
}

impl fmt::Debug for BroadcastTables {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BroadcastTables")
            .field("builds", &lock(&self.builds).len())
            .finish_non_exhaustive()
    }
}

/// One build: running until `outcome` is set.
#[derive(Default)]
struct Slot {
    outcome: Mutex<Option<Result<Arc<SharedBuild>>>>,
    done: Event,
    /// The build's query while it runs, cancelled with the resource.
    query: Mutex<Option<Arc<QueryContext>>>,
}

impl Slot {
    fn finish(&self, outcome: Result<Arc<SharedBuild>>) {
        let mut o = lock(&self.outcome);
        if o.is_none() {
            *o = Some(outcome);
        }
        drop(o);
        self.done.notify();
    }
}

/// Sets the slot's outcome if the build's task ends without one (a panic,
/// an abort), so no task waits forever.
struct FinishOnDrop(Arc<Slot>);

impl Drop for FinishOnDrop {
    fn drop(&mut self) {
        self.0.finish(Err(internal(
            "a broadcast build ended without a result".to_owned(),
        )));
    }
}

/// A finished shared build.
pub(crate) struct SharedBuild {
    /// The build's state: what the probe reads.
    build: Arc<BuildGlobal>,
    /// The runtime filters it published.
    filters: Vec<(FilterId, FilterSnapshot)>,
}

impl BroadcastTables {
    /// The builds of one job, run on `executor` and reserving from `pool`
    /// (the job's pool on this worker).
    #[must_use]
    pub fn new(executor: Arc<dyn Executor>, pool: Arc<dyn MemoryPool>) -> Self {
        Self {
            executor,
            pool,
            builds: Mutex::default(),
            next: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
        }
    }

    /// The builds started so far (finished, failed or running).
    #[must_use]
    pub fn builds(&self) -> usize {
        lock(&self.builds).len()
    }

    /// The builds that did not fit, so each task built its own table.
    #[must_use]
    pub fn fallbacks(&self) -> usize {
        lock(&self.builds)
            .values()
            .filter(|s| matches!(&*lock(&s.outcome), Some(Err(e)) if does_not_fit(e)))
            .count()
    }

    /// The tasks that skipped the shared build because it could never fit
    /// (each built its own table; not among [`Self::fallbacks`]).
    #[must_use]
    pub fn skipped(&self) -> u64 {
        self.skipped.load(Ordering::Relaxed)
    }

    /// The build `key`, started from `job` with `query`'s settings and
    /// class unless a task started it already, once it finished.
    async fn get(
        &self,
        key: BroadcastKey,
        job: impl FnOnce() -> Option<BuildJob>,
        query: &QueryContext,
    ) -> Result<Arc<SharedBuild>> {
        let (slot, start) = {
            let mut builds = lock(&self.builds);
            match builds.get(&key) {
                Some(s) => (Arc::clone(s), None),
                None => {
                    // This task handed its job over already, and its slot is
                    // gone: a newer attempt of the writing stage replaced it.
                    // The task reads rows nobody keeps; no slot is made for
                    // them (it would stand for that attempt from now on).
                    let Some(job) = job() else {
                        return Err(internal(format!(
                            "the broadcast of exchange {} attempt {} was superseded by a newer attempt",
                            key.exchange.0, key.stage_attempt
                        )));
                    };
                    // A newer attempt of the stage that wrote the exchange:
                    // no task reads an older attempt's rows any more, so
                    // its table goes (a retry never doubles the memory).
                    builds.retain(|k, s| {
                        let older =
                            k.exchange == key.exchange && k.stage_attempt < key.stage_attempt;
                        if older && let Some(q) = lock(&s.query).take() {
                            q.cancel();
                        }
                        !older
                    });
                    let s = Arc::new(Slot::default());
                    builds.insert(key, Arc::clone(&s));
                    (s, Some(job))
                }
            }
        };
        if let Some(job) = start {
            self.start(key, &slot, job, query.config(), query.class());
        }
        loop {
            let listener = slot.done.listen();
            if let Some(outcome) = lock(&slot.outcome).clone() {
                return outcome;
            }
            listener.await;
        }
    }

    fn start(
        &self,
        key: BroadcastKey,
        slot: &Arc<Slot>,
        job: BuildJob,
        config: &ExecConfig,
        class: WorkloadClass,
    ) {
        let n = self.next.fetch_add(1, Ordering::Relaxed);
        let query = QueryContext::new(
            // Apart from the worker's task queries (diagnostics only).
            QueryId(u64::MAX - n),
            class,
            config.clone(),
            Arc::new(NoAbortPool(
                self.pool
                    .child(&format!("broadcast-{}", key.exchange.0), None),
            )),
        );
        *lock(&slot.query) = Some(Arc::clone(&query));
        let runner = QueryRunner::new(Arc::clone(&self.executor));
        let guard = FinishOnDrop(Arc::clone(slot));
        let build = Box::pin(async move {
            let outcome = job.run(&runner, &query).await;
            *lock(&guard.0.query) = None;
            guard.0.finish(outcome);
        });
        // Detached: it runs to its end whichever task waits on it.
        drop(self.executor.spawn(class, build));
    }
}

impl Drop for BroadcastTables {
    fn drop(&mut self) {
        for slot in lock(&self.builds).values() {
            if let Some(q) = lock(&slot.query).take() {
                q.cancel();
            }
        }
    }
}

/// Whether a shared build failed only because it did not fit in memory (its
/// pool never aborts, [`NoAbortPool`]): the tasks then build their own
/// tables. An abort of the user's query is not a fallback.
fn does_not_fit(e: &KrishivError) -> bool {
    e.error_class() == ErrorClass::UnableToAcquireMemory
}

/// What a broadcast's built table holds at least, per byte of its exchange
/// frames: its rows. A lower bound, so no build that could fit is skipped.
/// Measured on TPC-H SF0.01 over a `LocalShuffle` (2026-11-07, design
/// `P2-row30c-m5-shared-skip`): every table's pool peak was 2x to over 1000x
/// its frames (the row pages' floor dominates small builds), so no upper
/// factor holds.
const EXPANSION: usize = 1;

/// How a task's join gets its table.
#[derive(Clone)]
enum Choice {
    /// The worker's shared build.
    Shared(Arc<SharedBuild>),
    /// The shared build did not fit: the task builds its own.
    Own,
}

/// A task's view of one shared build: started by the first of its readers
/// (a driver of the build pipeline or the sink's `finalize`) unless another
/// task started it, and the task's choice once it finished.
pub(crate) struct Decision {
    tables: Arc<BroadcastTables>,
    key: BroadcastKey,
    /// The frame bytes the broadcast's committed map outputs hold for this
    /// task, when its exchange knows them ([`Exchange::committed_bytes`](crate::exchange::Exchange::committed_bytes)).
    committed: Option<u64>,
    /// Handed over if this task starts the build.
    job: std::sync::Mutex<Option<BuildJob>>,
    chosen: OnceLock<Result<Choice>>,
}

impl Decision {
    pub(crate) fn new(
        tables: Arc<BroadcastTables>,
        key: BroadcastKey,
        committed: Option<u64>,
        job: BuildJob,
    ) -> Self {
        Self {
            tables,
            key,
            committed,
            job: std::sync::Mutex::new(Some(job)),
            chosen: OnceLock::new(),
        }
    }

    /// Whether the shared table could never fit: the broadcast's bytes times
    /// [`EXPANSION`] (a lower bound) past the most the job's broadcast pool can hold (its
    /// `krishiv.exec.join.sharedBroadcastBytes` cap or a limit above it).
    /// What other consumers hold now does not count: they can spill.
    fn cannot_fit(&self) -> bool {
        match (self.committed, self.tables.pool.effective_limit()) {
            (Some(bytes), Some(limit)) => {
                let table = usize::try_from(bytes)
                    .unwrap_or(usize::MAX)
                    .saturating_mul(EXPANSION);
                table > limit
            }
            _ => false,
        }
    }

    async fn choose(&self, query: &QueryContext) -> Result<Choice> {
        if let Some(c) = self.chosen.get() {
            return c.clone();
        }
        let job = || {
            self.job
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take()
        };
        if self.cannot_fit() {
            // Never attempted: the attempt would spill the query's other
            // operators before failing (design `P2-row30c-m5-shared-skip`).
            self.tables.skipped.fetch_add(1, Ordering::Relaxed);
            return self.chosen.get_or_init(|| Ok(Choice::Own)).clone();
        }
        let choice = match self.tables.get(self.key, job, query).await {
            Ok(built) => Ok(Choice::Shared(built)),
            Err(e) if does_not_fit(&e) => Ok(Choice::Own),
            Err(e) => Err(e),
        };
        // Every reader of one task sees the outcome of the same slot.
        self.chosen.get_or_init(|| choice).clone()
    }
}

/// The shared build's pool: `pool` whose slow-path reservations never abort
/// a query. The build cannot spill, and its tasks have a cheaper fallback
/// than an abort (their own builds), so a request spills others and then
/// fails with `UNABLE_TO_ACQUIRE_MEMORY` ([`MemoryPool::reserve_without_abort`]).
/// Without this the arbitrator would abort the query holding the build:
/// the user's query, whose pool contains it.
#[derive(Debug)]
struct NoAbortPool(Arc<dyn MemoryPool>);

impl MemoryPool for NoAbortPool {
    fn child(&self, name: &str, limit: Option<usize>) -> Arc<dyn MemoryPool> {
        Arc::new(Self(self.0.child(name, limit)))
    }

    fn register(&self, name: &str, spill: Option<Arc<dyn Spillable>>) -> ConsumerId {
        self.0.register(name, spill)
    }

    fn unregister(&self, id: ConsumerId) {
        self.0.unregister(id);
    }

    fn try_reserve(&self, id: ConsumerId, bytes: usize) -> Result<(), MemoryError> {
        self.0.try_reserve(id, bytes)
    }

    fn try_reserve_scratch(&self, id: ConsumerId, bytes: usize) -> Result<(), MemoryError> {
        self.0.try_reserve_scratch(id, bytes)
    }

    fn try_reserve_unguarded(&self, id: ConsumerId, bytes: usize) -> Result<(), MemoryError> {
        self.0.try_reserve_unguarded(id, bytes)
    }

    fn reserve<'a>(
        &'a self,
        id: ConsumerId,
        bytes: usize,
    ) -> BoxFuture<'a, Result<(), MemoryError>> {
        self.0.reserve_without_abort(id, bytes)
    }

    fn reserve_without_abort<'a>(
        &'a self,
        id: ConsumerId,
        bytes: usize,
    ) -> BoxFuture<'a, Result<(), MemoryError>> {
        self.0.reserve_without_abort(id, bytes)
    }

    fn release(&self, id: ConsumerId, bytes: usize) {
        self.0.release(id, bytes);
    }

    fn transfer(&self, from: ConsumerId, to: ConsumerId, bytes: usize) -> Result<(), MemoryError> {
        self.0.transfer(from, to, bytes)
    }

    fn reserved(&self) -> usize {
        self.0.reserved()
    }

    fn snapshot(&self) -> PoolSnapshot {
        self.0.snapshot()
    }

    fn peak_reserved(&self) -> usize {
        self.0.peak_reserved()
    }

    fn limit(&self) -> Option<usize> {
        self.0.limit()
    }

    fn effective_limit(&self) -> Option<usize> {
        self.0.effective_limit()
    }

    fn check_aborted(&self) -> Result<(), MemoryError> {
        self.0.check_aborted()
    }

    fn aborted(&self) -> BoxFuture<'_, MemoryError> {
        self.0.aborted()
    }
}

/// What the first task hands over to start a shared build: the join (its
/// runtime filters on a value service of its own) and its input (the
/// broadcast exchange's rows).
pub(crate) struct BuildJob {
    join: HashJoin,
    input: Arc<dyn ExecNode>,
}

impl BuildJob {
    /// A build of `input` by `join` (built without a spill scope, its
    /// runtime filters on a value service of its own).
    pub(crate) fn new(join: HashJoin, input: Arc<dyn ExecNode>) -> Self {
        Self { join, input }
    }

    async fn run(
        self,
        runner: &QueryRunner,
        query: &Arc<QueryContext>,
    ) -> Result<Arc<SharedBuild>> {
        let capture = Arc::new(Capture {
            inner: self.join.build_sink(),
            schema: self.join.schema(),
            state: OnceLock::new(),
        });
        let node = Arc::new(BuildOnly {
            sink: Arc::clone(&capture),
            input: self.input,
        });
        let mut outcome = Ok(());
        match runner.execute(node, Arc::clone(query)).await {
            Ok(mut stream) => {
                while let Some(b) = stream.next().await {
                    if let Err(e) = b {
                        outcome = Err(e);
                        break;
                    }
                }
            }
            Err(e) => outcome = Err(e),
        }
        query.finished().await;
        outcome?;
        if let Some(e) = query.error() {
            return Err(e);
        }
        let state =
            capture.state.get().cloned().ok_or_else(|| {
                internal("a broadcast build finished without its table".to_owned())
            })?;
        let build = state
            .downcast::<BuildGlobal>()
            .map_err(|_| internal("a broadcast build of another kind".to_owned()))?;
        let filters = self
            .join
            .runtime_filters()
            .iter()
            .map(|f| (f.filter.id(), f.filter.published().snapshot.clone()))
            .collect();
        Ok(Arc::new(SharedBuild { build, filters }))
    }
}

/// The shared build's graph: its input into the join's build sink, which
/// yields nothing.
struct BuildOnly {
    sink: Arc<Capture>,
    input: Arc<dyn ExecNode>,
}

impl ExecNode for BuildOnly {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        let sink = b.child_pipeline(
            Arc::clone(&self.sink) as Arc<dyn Sink>,
            Arc::clone(&self.input),
        )?;
        b.set_breaker_source(Arc::clone(&self.sink) as Arc<dyn Source>, sink)?;
        Ok(())
    }
}

/// The join's build sink, keeping its finalized state.
struct Capture {
    inner: Arc<dyn Sink>,
    schema: SchemaRef,
    state: OnceLock<GlobalState>,
}

impl Sink for Capture {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        self.inner.global_state(ctx)
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        self.inner.local_state(ctx, global)
    }

    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        global: &GlobalState,
    ) -> Result<SinkStatus> {
        self.inner.sink(input, local, global)
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        self.inner.combine(local, global)
    }

    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.inner.finalize(global, ctx).await?;
            let _ = self.state.set(Arc::clone(global));
            Ok(())
        })
    }

    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        self.inner.spill(local, target, ctx)
    }

    fn spill_global<'a>(
        &'a self,
        global: &'a GlobalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        self.inner.spill_global(global, target, ctx)
    }

    fn extra_metrics(&self) -> crate::metrics::ExtraMetrics {
        self.inner.extra_metrics()
    }
}

impl Source for Capture {
    fn name(&self) -> &'static str {
        "BroadcastBuildResult"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(()))
    }

    fn max_parallelism(&self, _global: &GlobalState) -> usize {
        1
    }

    fn local_state(&self, _ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn get_data<'a>(
        &'a self,
        _local: &'a mut LocalState,
        _global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(std::future::ready(Ok(None)))
    }
}

/// A task's build side over a shared build: an empty input, and a
/// `finalize` that waits for the worker's build and republishes its
/// runtime filters as this task's partials.
pub(crate) struct PrebuiltSink {
    /// The task's join (its filters and where it publishes them).
    shared: Arc<Shared>,
    /// The task's own build, used when the shared one did not fit.
    own: Arc<dyn Sink>,
    decision: Arc<Decision>,
}

impl PrebuiltSink {
    pub(crate) fn new(join: &HashJoin, decision: Arc<Decision>) -> Self {
        Self {
            shared: Arc::clone(&join.0),
            own: join.build_sink(),
            decision,
        }
    }
}

/// A [`PrebuiltSink`]'s global state: the shared build once it finished.
pub(crate) struct Prebuilt {
    pub(crate) build: OnceLock<Arc<BuildGlobal>>,
    /// The task's own build state (fed only after a fallback). On the
    /// shared path it is dropped unfinalized, and its drop publishes
    /// `Disabled` for the task's filters: ignored only because `finalize`
    /// republished the shared build's filters first and the value service
    /// keeps a producer's first partial. A change to that rule must keep
    /// this path publishing first.
    own: GlobalState,
    filters: Vec<FilterId>,
    publisher: Option<FilterPublisher>,
    published: AtomicBool,
}

impl Drop for Prebuilt {
    fn drop(&mut self) {
        // As a build that failed before finalize: consumers stop waiting.
        if !self.published.load(Ordering::Acquire)
            && let Some(p) = &self.publisher
        {
            for &id in &self.filters {
                p.publish(id, FilterSnapshot::Disabled(DisabledReason::ProducerFailed));
            }
        }
    }
}

/// The finished build a probe reads: a task's own, or the shared one.
pub(crate) fn build_of(dep: &GlobalState) -> Result<Arc<BuildGlobal>> {
    if let Ok(own) = Arc::clone(dep).downcast::<BuildGlobal>() {
        return Ok(own);
    }
    dep.downcast_ref::<Prebuilt>()
        .and_then(|p| p.build.get().cloned())
        .ok_or_else(|| internal("wrong build state for the hash join probe".to_owned()))
}

impl Sink for PrebuiltSink {
    fn name(&self) -> &'static str {
        "HashJoinBuild"
    }

    fn published_filters(&self) -> Vec<FilterId> {
        if self.shared.publisher.is_none() {
            return Vec::new();
        }
        self.shared
            .runtime
            .iter()
            .filter(|f| f.options.scan_waits)
            .map(|f| f.filter.id())
            .collect()
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(Prebuilt {
            build: OnceLock::new(),
            own: self.own.global_state(ctx)?,
            filters: self.shared.runtime.iter().map(|f| f.filter.id()).collect(),
            publisher: self.shared.publisher.clone(),
            published: AtomicBool::new(false),
        }))
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        self.own.local_state(ctx, &prebuilt(global)?.own)
    }

    // Rows arrive only after a fallback (`SharedOrOwnRows`).
    fn sink(
        &self,
        input: Batch,
        local: &mut LocalState,
        global: &GlobalState,
    ) -> Result<SinkStatus> {
        self.own.sink(input, local, &prebuilt(global)?.own)
    }

    fn combine(&self, local: LocalState, global: &GlobalState) -> Result<()> {
        self.own.combine(local, &prebuilt(global)?.own)
    }

    fn spill<'a>(
        &'a self,
        local: &'a mut LocalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        self.own.spill(local, target, ctx)
    }

    fn spill_global<'a>(
        &'a self,
        global: &'a GlobalState,
        target: usize,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let g = prebuilt(global)?;
            self.own.spill_global(&g.own, target, ctx).await
        })
    }

    /// The own build's counters, plus which table the task probed:
    /// `broadcast_shared` (the worker's) or `broadcast_fallback` (its own,
    /// after the shared build did not fit), summed over tasks.
    fn extra_metrics(&self) -> crate::metrics::ExtraMetrics {
        let mut m = self.own.extra_metrics();
        let probed = match self.decision.chosen.get() {
            Some(Ok(Choice::Shared(_))) => Some("broadcast_shared"),
            Some(Ok(Choice::Own)) => Some("broadcast_fallback"),
            Some(Err(_)) | None => None,
        };
        if let Some(name) = probed {
            m.push((
                std::borrow::Cow::Borrowed(name),
                crate::metrics::Metric::Count(1),
            ));
        }
        m
    }

    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let g = prebuilt(global)?;
            let built = match self.decision.choose(ctx.query()).await? {
                Choice::Shared(built) => built,
                Choice::Own => {
                    // The task's own build publishes its own filters.
                    self.own.finalize(&g.own, ctx).await?;
                    let _ = g.build.set(build_of(&g.own)?);
                    g.published.store(true, Ordering::Release);
                    return Ok(());
                }
            };
            let _ = g.build.set(Arc::clone(&built.build));
            if let Some(p) = &g.publisher {
                for (id, snapshot) in &built.filters {
                    let snapshot = match snapshot {
                        // Never published by the shared build: off.
                        FilterSnapshot::Pending => FilterSnapshot::Disabled(DisabledReason::Off),
                        s => s.clone(),
                    };
                    p.publish(*id, snapshot);
                }
            }
            g.published.store(true, Ordering::Release);
            Ok(())
        })
    }
}

fn prebuilt(global: &GlobalState) -> Result<&Prebuilt> {
    global
        .downcast_ref::<Prebuilt>()
        .ok_or_else(|| internal("wrong state for a shared broadcast build".to_owned()))
}

/// The input of a [`PrebuiltSink`]: no rows while the worker's shared build
/// serves the task, the broadcast's rows (read by this task) once the shared
/// build did not fit. Each driver waits for the choice on its first call.
pub(crate) struct SharedOrOwnRows {
    decision: Arc<Decision>,
    rows: Arc<dyn Source>,
}

impl SharedOrOwnRows {
    pub(crate) fn new(decision: Arc<Decision>, rows: Arc<dyn Source>) -> Self {
        Self { decision, rows }
    }
}

/// [`SharedOrOwnRows`]' global state: the task's query (whose settings a
/// build it starts takes) and the rows' state.
struct GateGlobal {
    query: Arc<QueryContext>,
    rows: GlobalState,
}

fn gate(global: &GlobalState) -> Result<&GateGlobal> {
    global
        .downcast_ref::<GateGlobal>()
        .ok_or_else(|| internal("wrong state for a shared broadcast's rows".to_owned()))
}

impl Source for SharedOrOwnRows {
    fn name(&self) -> &'static str {
        "SharedBroadcast"
    }

    fn schema(&self) -> SchemaRef {
        self.rows.schema()
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(GateGlobal {
            query: Arc::clone(ctx.query()),
            rows: self.rows.global_state(ctx)?,
        }))
    }

    fn max_parallelism(&self, global: &GlobalState) -> usize {
        global
            .downcast_ref::<GateGlobal>()
            .map_or(1, |g| self.rows.max_parallelism(&g.rows))
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        self.rows.local_state(ctx, &gate(global)?.rows)
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        let g = match gate(global) {
            Ok(g) => g,
            Err(e) => return Box::pin(std::future::ready(Err(e))),
        };
        // Once chosen, a call costs what the rows' own read costs.
        match self.decision.chosen.get() {
            Some(Ok(Choice::Own)) => return self.rows.get_data(local, &g.rows),
            Some(Ok(Choice::Shared(_))) => return Box::pin(std::future::ready(Ok(None))),
            Some(Err(e)) => return Box::pin(std::future::ready(Err(e.clone()))),
            None => {}
        }
        Box::pin(async move {
            match self.decision.choose(&g.query).await? {
                Choice::Shared(_) => Ok(None),
                Choice::Own => self.rows.get_data(local, &g.rows).await,
            }
        })
    }

    fn morsels_claimed(&self, local: &LocalState) -> u64 {
        self.rows.morsels_claimed(local)
    }
}

/// A hash join whose build is the worker's shared broadcast build.
pub struct SharedBuildJoinNode {
    pub(crate) join: HashJoin,
    pub(crate) sink: Arc<PrebuiltSink>,
    pub(crate) rows: Arc<SharedOrOwnRows>,
    pub(crate) probe: Arc<dyn ExecNode>,
}

impl fmt::Debug for SharedBuildJoinNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedBuildJoinNode")
            .field("join", &self.join)
            .field("key", &self.sink.decision.key)
            .finish_non_exhaustive()
    }
}

struct RowsNode(Arc<SharedOrOwnRows>);

impl ExecNode for RowsNode {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        b.set_source(Arc::clone(&self.0) as Arc<dyn Source>)
            .map(drop)
    }
}

impl ExecNode for SharedBuildJoinNode {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        let sink = b.child_pipeline(
            Arc::clone(&self.sink) as Arc<dyn Sink>,
            Arc::new(RowsNode(Arc::clone(&self.rows))),
        )?;
        Arc::clone(&self.probe).build_pipelines(b)?;
        b.push_operator_reading(self.join.probe(), sink).map(drop)
    }
}

#[cfg(all(test, not(loom)))]
#[expect(
    clippy::disallowed_methods,
    clippy::unwrap_used,
    reason = "test-only runtime plumbing via #[tokio::test] and test assertions; never runs on a compute worker"
)]
mod tests {
    use futures::FutureExt as _;
    use krishiv_memory::{MemoryConfig, TreePool};

    use super::*;

    /// A query's pool under a 1 MiB root, with another consumer of the
    /// query holding 768 KiB that cannot spill.
    fn crowded() -> (Arc<dyn MemoryPool>, Arc<dyn MemoryPool>, ConsumerId) {
        let root: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig {
            limit: Some(1 << 20),
            ..MemoryConfig::default()
        });
        let query = root.child("q", None);
        let other = query.register("other", None);
        query.try_reserve(other, 768 << 10).unwrap();
        (root, query, other)
    }

    /// A decision of a task over the slot `key` of `tables`, whose build
    /// already ended with `outcome` (no job to hand over).
    fn decided(
        tables: &Arc<BroadcastTables>,
        key: BroadcastKey,
        outcome: Result<Arc<SharedBuild>>,
    ) -> Arc<Decision> {
        let slot = Arc::new(Slot::default());
        slot.finish(outcome);
        lock(&tables.builds).insert(key, slot);
        Arc::new(Decision {
            tables: Arc::clone(tables),
            key,
            committed: None,
            job: std::sync::Mutex::new(None),
            chosen: OnceLock::new(),
        })
    }

    /// The three outcomes of a shared build as its tasks see them: a build
    /// that did not fit is a fallback to the task's own build; any other
    /// failure (here a reader's) fails the task with it, and the gate in
    /// front of the task's rows returns that error; neither counts the
    /// other. A task whose slot was superseded fails.
    #[tokio::test]
    async fn a_shared_build_that_failed_otherwise_fails_its_tasks() {
        let (root, query, other) = crowded();
        let exec = crate::TokioExecutor::new(&crate::ExecutorConfig::default()).unwrap();
        let tables = Arc::new(BroadcastTables::new(
            Arc::clone(&exec) as Arc<dyn Executor>,
            query.child("broadcasts", None),
        ));
        let full = NoAbortPool(query.child("broadcast-0", None));
        let b = full.register("build", None);
        let oom = KrishivError::from(full.reserve(b, 512 << 10).await.unwrap_err());
        full.unregister(b);
        let key = |exchange| BroadcastKey {
            exchange: ExchangeId(exchange),
            stage_attempt: 0,
        };
        let task = QueryContext::new(
            QueryId(1),
            WorkloadClass::Interactive,
            ExecConfig::default(),
            query.child("task", None),
        );

        let fell_back = decided(&tables, key(1), Err(oom));
        assert!(matches!(fell_back.choose(&task).await, Ok(Choice::Own)));
        let broken = decided(&tables, key(2), Err(internal("a reader broke".to_owned())));
        let e = broken.choose(&task).await.err().unwrap();
        assert!(e.message().contains("a reader broke"), "{e}");
        assert_eq!(tables.fallbacks(), 1);
        assert_eq!(tables.builds(), 2);

        // The rows gate after the choice: the error again, no rows read.
        let rows = crate::testing::RangeSource::new(10);
        let gate = SharedOrOwnRows::new(Arc::clone(&broken), Arc::clone(&rows) as _);
        let ctx = TaskContext::new(
            Arc::clone(&task),
            Arc::clone(&exec) as Arc<dyn Executor>,
            krishiv_memory::SpillOwner::new(),
            crate::OperatorId(0),
            "gate",
        );
        let global = gate.global_state(&ctx).unwrap();
        let mut local = gate.local_state(&ctx, &global).unwrap();
        let e = gate.get_data(&mut local, &global).await.unwrap_err();
        assert!(e.message().contains("a reader broke"), "{e}");
        assert_eq!(rows.produced.load(Ordering::Relaxed), 0);

        // A newer attempt replaced the slot this task's job went to.
        let gone = Decision {
            tables: Arc::clone(&tables),
            key: key(3),
            committed: None,
            job: std::sync::Mutex::new(None),
            chosen: OnceLock::new(),
        };
        let e = gone.choose(&task).await.err().unwrap();
        assert!(e.message().contains("superseded"), "{e}");

        drop((gate, ctx, task, tables));
        query.release(other, usize::MAX);
        assert_eq!(root.reserved(), 0);
    }

    /// The shared build's pool refuses a request that does not fit
    /// instead of aborting the query holding it, where the plain pool
    /// aborts the query; every other call is the wrapped pool's.
    #[tokio::test]
    async fn the_shared_builds_pool_never_aborts_its_query() {
        let (_root, query, other) = crowded();
        let pool = NoAbortPool(query.child("broadcast-1", None));
        let b = pool.register("build", None);
        let e = KrishivError::from(pool.reserve(b, 512 << 10).await.unwrap_err());
        assert!(does_not_fit(&e), "{e}");
        assert!(query.check_aborted().is_ok(), "the query was aborted");
        let e = pool.reserve_without_abort(b, 512 << 10).await.unwrap_err();
        assert!(does_not_fit(&KrishivError::from(e)));

        // Delegation.
        pool.try_reserve(b, 1024).unwrap();
        pool.try_reserve_scratch(b, 1024).unwrap();
        pool.try_reserve_unguarded(b, 1024).unwrap();
        pool.reserve(b, 1024).await.unwrap();
        assert!(pool.reserved() >= 4096);
        assert!(pool.peak_reserved() >= pool.reserved());
        assert!(!pool.snapshot().entries.is_empty());
        assert_eq!(pool.limit(), None);
        assert_eq!(pool.effective_limit(), Some(1 << 20));
        assert!(pool.check_aborted().is_ok());
        assert!(pool.aborted().now_or_never().is_none());
        let c = pool.register("other-build", None);
        pool.transfer(b, c, 1024).unwrap();
        pool.release(b, usize::MAX);
        pool.release(c, usize::MAX);
        assert_eq!(pool.reserved(), 0);
        // A child keeps the rule.
        let child = pool.child("part", None);
        let d = child.register("d", None);
        let e = KrishivError::from(child.reserve(d, 512 << 10).await.unwrap_err());
        assert!(does_not_fit(&e), "{e}");
        assert!(query.check_aborted().is_ok());
        child.unregister(d);
        pool.unregister(b);
        pool.unregister(c);
        query.release(other, usize::MAX);
        assert!(!format!("{pool:?}").is_empty());

        // The plain pool aborts the query to make room.
        let (_root, query, _other) = crowded();
        let plain = query.child("broadcast-1", None);
        let b = plain.register("build", None);
        assert!(plain.reserve(b, 512 << 10).await.is_err());
        assert!(
            query.check_aborted().is_err(),
            "the plain pool did not abort"
        );
    }
}
