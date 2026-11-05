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
//! - **No spill.** As in Spark, a broadcast relation must fit in memory: the
//!   shared build has no spill scope, so a build that does not fit fails
//!   every task waiting on it. (A spilled build's grace state holds one
//!   task's probe rows and cannot be shared.)
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
use krishiv_io::{DisabledReason, FilterId, FilterSnapshot};
use krishiv_memory::MemoryPool;
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
        }
    }

    /// The builds started so far (finished, failed or running).
    #[must_use]
    pub fn builds(&self) -> usize {
        lock(&self.builds).len()
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
                Some(s) => (Arc::clone(s), false),
                None => {
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
                    (s, true)
                }
            }
        };
        if start {
            match job() {
                Some(job) => self.start(key, &slot, job, query.config(), query.class()),
                None => slot.finish(Err(internal(format!(
                    "the broadcast build of exchange {} was started twice by one task",
                    key.exchange.0
                )))),
            }
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
            self.pool
                .child(&format!("broadcast-{}", key.exchange.0), None),
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
    tables: Arc<BroadcastTables>,
    key: BroadcastKey,
    /// Handed over if this task starts the build.
    job: std::sync::Mutex<Option<BuildJob>>,
}

impl PrebuiltSink {
    pub(crate) fn new(
        join: &HashJoin,
        tables: Arc<BroadcastTables>,
        key: BroadcastKey,
        job: BuildJob,
    ) -> Self {
        Self {
            shared: Arc::clone(&join.0),
            tables,
            key,
            job: std::sync::Mutex::new(Some(job)),
        }
    }
}

/// A [`PrebuiltSink`]'s global state: the shared build once it finished.
pub(crate) struct Prebuilt {
    pub(crate) build: OnceLock<Arc<BuildGlobal>>,
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

    fn global_state(&self, _ctx: &TaskContext) -> Result<GlobalState> {
        Ok(Arc::new(Prebuilt {
            build: OnceLock::new(),
            filters: self.shared.runtime.iter().map(|f| f.filter.id()).collect(),
            publisher: self.shared.publisher.clone(),
            published: AtomicBool::new(false),
        }))
    }

    fn local_state(&self, _ctx: &TaskContext, _global: &GlobalState) -> Result<LocalState> {
        Ok(Box::new(()))
    }

    fn sink(
        &self,
        _input: Batch,
        _local: &mut LocalState,
        _global: &GlobalState,
    ) -> Result<SinkStatus> {
        Err(internal(
            "rows sunk into a shared broadcast build".to_owned(),
        ))
    }

    fn combine(&self, _local: LocalState, _global: &GlobalState) -> Result<()> {
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        global: &'a GlobalState,
        ctx: &'a TaskContext,
    ) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let g = global
                .downcast_ref::<Prebuilt>()
                .ok_or_else(|| internal("wrong state for a shared broadcast build".to_owned()))?;
            let job = || {
                self.job
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take()
            };
            let built = self.tables.get(self.key, job, ctx.query()).await?;
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

/// A source of no rows: the input of a [`PrebuiltSink`].
pub(crate) struct NoRows(pub(crate) SchemaRef);

impl Source for NoRows {
    fn name(&self) -> &'static str {
        "SharedBroadcast"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.0)
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

/// A hash join whose build is the worker's shared broadcast build.
pub struct SharedBuildJoinNode {
    pub(crate) join: HashJoin,
    pub(crate) sink: Arc<PrebuiltSink>,
    pub(crate) build_schema: SchemaRef,
    pub(crate) probe: Arc<dyn ExecNode>,
}

impl fmt::Debug for SharedBuildJoinNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedBuildJoinNode")
            .field("join", &self.join)
            .field("key", &self.sink.key)
            .finish_non_exhaustive()
    }
}

struct NoRowsNode(SchemaRef);

impl ExecNode for NoRowsNode {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        b.set_source(Arc::new(NoRows(Arc::clone(&self.0))))
            .map(drop)
    }
}

impl ExecNode for SharedBuildJoinNode {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        let sink = b.child_pipeline(
            Arc::clone(&self.sink) as Arc<dyn Sink>,
            Arc::new(NoRowsNode(Arc::clone(&self.build_schema))),
        )?;
        Arc::clone(&self.probe).build_pipelines(b)?;
        b.push_operator_reading(self.join.probe(), sink).map(drop)
    }
}
