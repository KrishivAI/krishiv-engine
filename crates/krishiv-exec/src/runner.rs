// SPDX-License-Identifier: Apache-2.0
//! Running a query in process: [`Scheduler`] and its embedded
//! implementation, [`QueryRunner`].
//!
//! `submit` instantiates the [`PhysicalPlan`] with the caller's
//! [`WorkerContext`] ([`instantiate`](crate::physical::instantiate)) on
//! the executor's blocking pool when its future runs,
//! splits the operator graph into pipelines and spawns a coordinator task,
//! which starts each pipeline once its dependencies have finalized, and
//! once the pipelines publishing the runtime filters its scan applies have
//! ([`with_filter_order`]), so a scan does not read rows a filter about to
//! be published would drop:
//! `min(worker slots, source parallelism)` drivers per pipeline (one if it
//! must keep order). The pipelines of a fan-in (`UNION ALL`) end in one
//! sink: they run one after another, in pipeline order, when none of
//! their shared operators needs a cross-driver flush, and start together
//! otherwise (taking turns with a [`Baton`] under `krishiv.deterministicOrder`),
//! sharing the sink's state either way. The coordinator awaits every driver; the first error
//! (a driver's error, a panic, a memory abort, cancellation) wins and
//! cancels the rest, which are also aborted. The result stream ends only
//! after every task has ended and every pipeline's state is dropped, so the
//! query's memory pool is empty by then.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Weak};

use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::{BoxStream, FuturesUnordered};
use krishiv_io::FilterId;
use krishiv_memory::SpillOwner;
use krishiv_types::batch::Batch;

use crate::channel::{Channel, ResultSink, ResultStream};
use crate::context::{OperatorId, QueryContext, QueryId, TaskContext};
use crate::driver::{PipelineRun, spawn_driver};
use crate::error::{Result, cancelled, internal};
use crate::executor::{Executor, TaskHandle};
use crate::metrics::OperatorEntry;
use crate::morsel::{Barrier, Baton, CombineOrder, Countdown};
use crate::numbering::{MemberShape, Numbering};
use crate::operator::{GlobalState, Source};
use crate::physical::{PhysicalPlan, WorkerContext, instantiate};
use crate::pipeline::{ExecNode, Origin, Pipeline, PipelineBuilder};
use crate::sync::{Mutex, lock};

/// Plans in, running queries out (spec 13 §4). Embedded and single-node:
/// [`QueryRunner`]; distributed: the coordinator (P5).
pub trait Scheduler: Send + Sync {
    /// Starts `plan` and returns its result stream. Dropping the stream
    /// cancels the query. The plan is data; `worker` holds what this
    /// process resolves its references with (tables, spill scope, I/O
    /// context, formats and configuration) for the parts of the plan that run here.
    ///
    /// # Errors
    /// Instantiation errors; execution errors arrive on the stream.
    fn submit(
        &self,
        plan: PhysicalPlan,
        worker: WorkerContext,
        query: Arc<QueryContext>,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<Batch>>>>;
    /// Cancels a running query; unknown or finished ids are ignored.
    fn cancel(&self, query: QueryId);
}

type Running = Arc<Mutex<HashMap<QueryId, Weak<QueryContext>>>>;

/// The in-process [`Scheduler`].
#[derive(Debug)]
pub struct QueryRunner {
    executor: Arc<dyn Executor>,
    running: Running,
}

impl QueryRunner {
    /// A runner on `executor`.
    pub fn new(executor: Arc<dyn Executor>) -> Self {
        Self {
            executor,
            running: Arc::default(),
        }
    }

    fn start(
        &self,
        plan: Arc<dyn ExecNode>,
        query: Arc<QueryContext>,
    ) -> Result<BoxStream<'static, Result<Batch>>> {
        let capacity = query.config().result_buffer_per_slot * self.executor.parallelism();
        let channel = Channel::new(capacity);
        // Under `krishiv.deterministicOrder` every pipeline runs one driver
        // (lanes aside, `start_members`), and a plan chosen from this value
        // must not make its output order follow thread timing: one.
        let parallelism = if query.config().deterministic_order {
            1
        } else {
            self.executor.parallelism()
        };
        let mut b = PipelineBuilder::new(Arc::new(ResultSink {
            channel: Arc::clone(&channel),
        }))
        .with_parallelism(parallelism);
        plan.build_pipelines(&mut b)?;
        let (pipelines, names) = b.finish()?;
        let mut member: HashMap<OperatorId, Vec<u32>> = HashMap::new();
        for p in &pipelines {
            let ops = std::iter::once(p.source.0)
                .chain(p.operators.iter().map(|(id, _)| *id))
                .chain(std::iter::once(p.sink.0));
            for id in ops {
                member.entry(id).or_default().push(p.index);
            }
        }
        query.merge_metrics(|m| {
            m.workers = 1;
            for n in names {
                let mut pipelines = member.remove(&n.id).unwrap_or_default();
                pipelines.sort_unstable();
                pipelines.dedup();
                m.operators.insert(
                    n.id,
                    OperatorEntry {
                        name: Cow::Borrowed(n.name),
                        node: n.node,
                        role: n.role,
                        pipelines,
                        ..OperatorEntry::default()
                    },
                );
            }
        });
        query.start_clock(self.executor.now());
        {
            let mut running = lock(&self.running);
            running.retain(|_, q| q.upgrade().is_some_and(|q| !q.is_finished()));
            if running.contains_key(&query.id()) {
                return Err(internal(format!(
                    "query {} is already running",
                    query.id().0
                )));
            }
            running.insert(query.id(), Arc::downgrade(&query));
        }
        let finish = Finish {
            query: Arc::clone(&query),
            executor: Arc::clone(&self.executor),
            channel: Arc::clone(&channel),
            running: Arc::clone(&self.running),
            completed: false,
        };
        let executor = Arc::clone(&self.executor);
        let q = Arc::clone(&query);
        // The pipeline task runs inside the span that submitted it (the caller's `krishiv.execute`).
        let span = tracing::Span::current();
        drop(self.executor.spawn(
            query.class(),
            Box::pin(tracing::Instrument::instrument(
                async move {
                    if let Err(e) = run_pipelines(&executor, &q, pipelines).await {
                        q.fail(e);
                    }
                    q.blocking_done().await;
                    finish.complete();
                },
                span,
            )),
        ));
        Ok(Box::pin(ResultStream { channel, query }))
    }
}

impl QueryRunner {
    /// Runs an instantiated operator graph (worker-local; a [`Scheduler`]
    /// takes plans) and returns its result stream. Dropping the stream
    /// cancels the query.
    ///
    /// # Errors
    /// Pipeline construction errors; execution errors arrive on the
    /// stream.
    pub fn execute(
        &self,
        node: Arc<dyn ExecNode>,
        query: Arc<QueryContext>,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<Batch>>>> {
        let started = self.start(node, query);
        Box::pin(async move { started })
    }
}

impl Scheduler for QueryRunner {
    fn submit(
        &self,
        plan: PhysicalPlan,
        worker: WorkerContext,
        query: Arc<QueryContext>,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<Batch>>>> {
        let runner = Self {
            executor: Arc::clone(&self.executor),
            running: Arc::clone(&self.running),
        };
        Box::pin(async move {
            // Instantiation builds operators and starts a deep-stack
            // thread: on the blocking pool, not on an async worker.
            let built = Arc::new(Mutex::new(None));
            let out = Arc::clone(&built);
            runner
                .executor
                .spawn_blocking(Box::new(move || {
                    *lock(&out) = Some(instantiate(&plan, &worker));
                }))
                .await?;
            let node = lock(&built)
                .take()
                .ok_or_else(|| internal("an instantiation without a result".to_owned()))??;
            runner.start(node, query)
        })
    }

    fn cancel(&self, query: QueryId) {
        let q = lock(&self.running).get(&query).and_then(Weak::upgrade);
        if let Some(q) = q {
            q.cancel();
        }
    }
}

/// Ends the query when the coordinator finishes, or is dropped (its
/// runtime shut down): marks it finished and closes the result stream.
struct Finish {
    query: Arc<QueryContext>,
    executor: Arc<dyn Executor>,
    channel: Arc<Channel>,
    running: Running,
    completed: bool,
}

impl Finish {
    /// The coordinator ran to the end (consumes the guard, which then
    /// drops).
    fn complete(mut self) {
        self.completed = true;
    }
}

impl Drop for Finish {
    fn drop(&mut self) {
        if !self.completed {
            self.query.fail(cancelled());
        }
        let ended = self.query.offset(self.executor.now());
        self.query.merge_metrics(|m| m.wall_time = ended);
        self.query.mark_finished();
        lock(&self.running).remove(&self.query.id());
        self.channel.close(self.query.error());
    }
}

/// `deps` with each pipeline that applies a runtime filter (`applies`) also
/// depending on the pipelines that publish it and say it is worth the wait
/// (`publishes`), unless a
/// publisher already runs after that pipeline, or with it: a filter never
/// makes a query wait for itself. Pipelines with one sink (`sinks`) start
/// together, so depending on one means depending on all.
fn with_filter_order(
    mut deps: Vec<Vec<usize>>,
    sinks: &[OperatorId],
    applies: &[Vec<FilterId>],
    publishes: &[Vec<FilterId>],
) -> Vec<Vec<usize>> {
    for (consumer, filters) in applies.iter().enumerate() {
        for filter in filters {
            for (publisher, published) in publishes.iter().enumerate() {
                if published.contains(filter)
                    && !deps[consumer].contains(&publisher)
                    && !runs_with_or_after(publisher, consumer, &deps, sinks)
                {
                    deps[consumer].push(publisher);
                }
            }
        }
    }
    deps
}

/// Whether `pipeline` starts with `other`, or only after it has finalized.
fn runs_with_or_after(
    pipeline: usize,
    other: usize,
    deps: &[Vec<usize>],
    sinks: &[OperatorId],
) -> bool {
    let mut seen = vec![false; deps.len()];
    let mut stack = vec![pipeline];
    while let Some(next) = stack.pop() {
        for member in (0..sinks.len()).filter(|&m| sinks[m] == sinks[next]) {
            if member == other {
                return true;
            }
            if !seen[member] {
                seen[member] = true;
                stack.extend(&deps[member]);
            }
        }
    }
    false
}

/// Runs the pipelines in dependency waves (or, under
/// `krishiv.exec.dataflowScheduling`, [`run_dataflow`]). The coordinator owns every
/// breaker's state: a finalized sink's global state is kept here and handed
/// to the pipelines that read it, dropped (with the memory it holds) once
/// the last of them has ended, and whatever is left is dropped when this
/// returns, on success, error, cancel or panic alike, so the query's pool is
/// empty even if the caller still holds the plan.
async fn run_pipelines(
    executor: &Arc<dyn Executor>,
    query: &Arc<QueryContext>,
    pipelines: Vec<Pipeline>,
) -> Result<()> {
    // Pipelines ending in one sink (a fan-in) start together and run as
    // one; a group is ready once every member's dependencies finalized.
    let sinks: Vec<OperatorId> = pipelines.iter().map(|p| p.sink.0).collect();
    let group =
        |i: usize| -> Vec<usize> { (0..sinks.len()).filter(|&j| sinks[j] == sinks[i]).collect() };
    let deps = with_filter_order(
        pipelines.iter().map(|p| p.deps.clone()).collect(),
        &sinks,
        &pipelines
            .iter()
            .map(|p| p.source.1.runtime_filters())
            .collect::<Vec<_>>(),
        &pipelines
            .iter()
            .map(|p| p.sink.1.published_filters())
            .collect::<Vec<_>>(),
    );
    let mut todo: Vec<Option<Pipeline>> = pipelines.into_iter().map(Some).collect();
    let mut done = vec![false; todo.len()];
    let mut finalized: HashMap<OperatorId, GlobalState> = HashMap::new();
    // Pipelines yet to end that read each breaker's state.
    let mut readers: HashMap<OperatorId, usize> = HashMap::new();
    for p in todo.iter().flatten() {
        for input in inputs(p) {
            *readers.entry(input).or_default() += 1;
        }
    }
    if query.config().dataflow_scheduling {
        return run_dataflow(executor, query, todo, &deps, &sinks, finalized, readers).await;
    }
    while done.iter().any(|d| !d) {
        let ready: Vec<Vec<usize>> = (0..todo.len())
            .filter(|&i| !done[i] && group(i)[0] == i)
            .map(group)
            .filter(|g| g.iter().all(|&m| deps[m].iter().all(|&d| done[d])))
            .collect();
        if ready.is_empty() {
            // Unreachable: dependencies form a DAG (a shared plan is read by several pipelines).
            return Err(internal("pipeline dependencies form a cycle".to_owned()));
        }
        let mut handles = Drivers(FuturesUnordered::new());
        let mut runs = Vec::new();
        for g in ready {
            let members: Vec<Pipeline> = g
                .iter()
                .filter_map(|&i| {
                    done[i] = true;
                    todo[i].take()
                })
                .collect();
            let group = match start_group(executor, query, members, &finalized) {
                Ok(group) => group,
                Err(e) => {
                    query.fail(e);
                    break;
                }
            };
            if fan_in_is_sequential(&group) {
                match run_group_sequential(executor, query, group).await {
                    Ok(rs) => runs.extend(rs),
                    Err(e) => {
                        query.fail(e);
                        break;
                    }
                }
            } else {
                match spawn_group(executor, query, group) {
                    Ok((rs, hs)) => {
                        handles.0.extend(hs);
                        runs.extend(rs);
                    }
                    Err(e) => {
                        query.fail(e);
                        break;
                    }
                }
            }
        }
        await_drivers(query, &mut handles).await;
        record_extra_metrics(query, &runs);
        query.check_cancelled()?;
        for run in &runs {
            finalized.insert(run.pipeline.sink.0, Arc::clone(&run.sink_global));
        }
        for run in runs {
            for input in inputs(&run.pipeline) {
                let left = readers.get_mut(&input).map(|n| {
                    *n -= 1;
                    *n
                });
                if left == Some(0) {
                    finalized.remove(&input);
                }
            }
        }
    }
    Ok(())
}

/// A started group of the dataflow runner: its members, their runs, and
/// whether it ended cleanly.
type GroupRun = BoxFuture<'static, (Vec<usize>, Vec<Arc<PipelineRun>>, bool)>;

/// [`run_pipelines`] without the wave barrier (`krishiv.exec.dataflowScheduling`):
/// a group starts as soon as every member's dependencies have ended, not
/// when the whole wave before it has, so a long pipeline no longer holds
/// back the pipelines that do not read it. The bookkeeping (finalized
/// states, their readers, metrics) happens per group as each one ends.
/// After a failure or cancellation no group starts, and every running one
/// is still awaited (its drivers aborted and drained), so the drivers have
/// released everything when this returns, as in the wave runner.
async fn run_dataflow(
    executor: &Arc<dyn Executor>,
    query: &Arc<QueryContext>,
    mut todo: Vec<Option<Pipeline>>,
    deps: &[Vec<usize>],
    sinks: &[OperatorId],
    mut finalized: HashMap<OperatorId, GlobalState>,
    mut readers: HashMap<OperatorId, usize>,
) -> Result<()> {
    let group =
        |i: usize| -> Vec<usize> { (0..sinks.len()).filter(|&j| sinks[j] == sinks[i]).collect() };
    let mut started = vec![false; todo.len()];
    let mut finished = vec![false; todo.len()];
    let mut running: FuturesUnordered<GroupRun> = FuturesUnordered::new();
    let mut stopped = false;
    loop {
        stopped |= query.is_cancelled();
        if !stopped {
            let ready: Vec<Vec<usize>> = (0..todo.len())
                .filter(|&i| !started[i] && group(i)[0] == i)
                .map(group)
                .filter(|g| g.iter().all(|&m| deps[m].iter().all(|&d| finished[d])))
                .collect();
            for g in ready {
                if let Err(e) = start_dataflow_group(
                    executor,
                    query,
                    &mut todo,
                    &mut started,
                    &finalized,
                    g,
                    &running,
                ) {
                    query.fail(e);
                    stopped = true;
                    break;
                }
            }
        }
        let Some((g, runs, ok)) = running.next().await else {
            break;
        };
        // Before the stop, as the wave runner records a failed wave's: a
        // failed or cancelled query's profile keeps these counters.
        record_extra_metrics(query, &runs);
        if !ok || query.is_cancelled() {
            // Keep draining the running groups; their states are dropped on return.
            stopped = true;
            continue;
        }
        for &m in &g {
            finished[m] = true;
        }
        for run in &runs {
            finalized.insert(run.pipeline.sink.0, Arc::clone(&run.sink_global));
        }
        for run in runs {
            for input in inputs(&run.pipeline) {
                let left = readers.get_mut(&input).map(|n| {
                    *n -= 1;
                    *n
                });
                if left == Some(0) {
                    finalized.remove(&input);
                }
            }
        }
    }
    query.check_cancelled()?;
    if finished.iter().any(|f| !f) {
        // Unreachable: dependencies form a DAG (a shared plan is read by several pipelines).
        return Err(internal("pipeline dependencies form a cycle".to_owned()));
    }
    Ok(())
}

/// Starts group `g` of [`run_dataflow`] and adds it to `running`.
fn start_dataflow_group(
    executor: &Arc<dyn Executor>,
    query: &Arc<QueryContext>,
    todo: &mut [Option<Pipeline>],
    started: &mut [bool],
    finalized: &HashMap<OperatorId, GlobalState>,
    g: Vec<usize>,
    running: &FuturesUnordered<GroupRun>,
) -> Result<()> {
    let members: Vec<Pipeline> = g
        .iter()
        .filter_map(|&i| {
            started[i] = true;
            todo[i].take()
        })
        .collect();
    let start = start_group(executor, query, members, finalized)?;
    let (e, q) = (Arc::clone(executor), Arc::clone(query));
    if fan_in_is_sequential(&start) {
        running.push(Box::pin(async move {
            match run_group_sequential(&e, &q, start).await {
                Ok(rs) => (g, rs, true),
                Err(err) => {
                    q.fail(err);
                    (g, Vec::new(), false)
                }
            }
        }));
    } else {
        let (rs, hs) = spawn_group(executor, query, start)?;
        let mut handles = Drivers(hs.into_iter().collect());
        running.push(Box::pin(async move {
            await_drivers(&q, &mut handles).await;
            (g, rs, true)
        }));
    }
    Ok(())
}

/// The breakers whose finalized state `p` reads, once each.
fn inputs(p: &Pipeline) -> Vec<OperatorId> {
    let mut ids: Vec<OperatorId> = p
        .input
        .iter()
        .chain(&p.second_input)
        .chain(p.operator_inputs.iter().flatten())
        .copied()
        .collect();
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Reads the operator-specific counters of each finished pipeline's
/// source, operators and sink into the query's metrics.
fn record_extra_metrics(query: &QueryContext, runs: &[Arc<PipelineRun>]) {
    query.merge_metrics(|m| {
        for run in runs {
            let p = &run.pipeline;
            let source = std::iter::once((p.source.0, p.source.1.extra_metrics()));
            let ops = p.operators.iter().map(|(id, op)| (*id, op.extra_metrics()));
            let sink = std::iter::once((p.sink.0, p.sink.1.extra_metrics()));
            for (id, extra) in source.chain(ops).chain(sink) {
                // An operator shared by the pipelines of a fan-in reports
                // its query-wide counters once.
                let entry = m.operators.entry(id).or_default();
                if !extra.is_empty() && entry.extra.is_empty() {
                    entry.extra = extra;
                }
            }
        }
    });
}

/// Starts the pipelines of one sink: one pipeline, or the pipelines of a
/// fan-in, which share the sink's state, its finalize countdown and the
/// flush barriers of the operators they share. Pipelines of a fan-in
/// whose branches share no operator needing a cross-driver flush run one
/// after another, in pipeline order (a fan-in's branches, left to right),
/// so the union emits its branches' rows in order; any other group starts
/// together and runs as one.
fn start_group(
    executor: &Arc<dyn Executor>,
    query: &Arc<QueryContext>,
    members: Vec<Pipeline>,
    finalized: &HashMap<OperatorId, GlobalState>,
) -> Result<GroupStart> {
    // Global-state contexts get an owner nobody services: memory reserved
    // in `global_state` is never asked to spill (see `Sink::global_state`).
    let ctx = |(id, name)| {
        TaskContext::new(
            Arc::clone(query),
            Arc::clone(executor),
            SpillOwner::new(),
            id,
            name,
        )
    };
    let read = |input: &OperatorId| {
        let state = finalized.get(input).cloned();
        state.ok_or_else(|| internal("a breaker read before it finalized".to_owned()))
    };
    let Some(first) = members.first() else {
        return Err(internal("a pipeline group without pipelines".to_owned()));
    };
    let (sink_id, sink) = &first.sink;
    // A sink sees the breaker its pipeline's source reads, as the source
    // does (for a fan-in, the first member's): a sink that continues that
    // breaker's state adopts it (`aggregate/distinct.rs` `KeyFeed`); every
    // other sink ignores it.
    let mut sink_ctx = ctx((*sink_id, sink.name()));
    if let Some(input) = &first.input {
        sink_ctx = sink_ctx.with_dependency(read(input)?);
    }
    let sink_global = sink.global_state(&sink_ctx)?;
    let sink = Arc::clone(sink);
    let mut started = Vec::with_capacity(members.len());
    // Scheduler-assigned globally unique partition ids: the drivers of
    // every member share one id space, in pipeline order, so two drivers
    // never compute the same partition (stateful functions own one value
    // stream per partition). An index past `u32` is rejected rather than
    // wrapped to `u32::MAX`, which would collide.
    let mut prepared = Vec::with_capacity(members.len());
    for pipeline in members {
        let (source_id, source) = &pipeline.source;
        let mut source_ctx = ctx((*source_id, source.name()));
        if let Some(input) = &pipeline.input {
            source_ctx = source_ctx.with_dependency(read(input)?);
        }
        if let Some(input) = &pipeline.second_input {
            source_ctx = source_ctx.with_second_dependency(read(input)?);
        }
        let operator_deps = pipeline
            .operator_inputs
            .iter()
            .map(|i| i.as_ref().map(read).transpose())
            .collect::<Result<Vec<_>>>()?;
        let source_global = source.global_state(&source_ctx)?;
        let lane_count = lane_count(query, &pipeline, source, &source_global);
        prepared.push((pipeline, operator_deps, source_global, lane_count));
    }
    // A fan-in whose members do not all split keeps one driver per member (the baton
    // needs it, and a flush barrier is shared: design §6.5), so no member runs in lanes.
    let mixed = prepared.len() > 1 && prepared.iter().any(|p| p.3.is_none());
    let mut next_partition: u64 = 0;
    for (pipeline, operator_deps, source_global, lane_count) in prepared {
        let (_, source) = &pipeline.source;
        let lane_count = if mixed { None } else { lane_count };
        // One driver when the pipeline must keep order, and for every pipeline under
        // `krishiv.deterministicOrder`: no interleaving of drivers, no run-to-run
        // difference in the order of merges or output rows.
        let drivers = if let Some(n) = lane_count {
            n
        } else if pipeline.preserve_order || query.config().deterministic_order {
            1
        } else {
            executor
                .parallelism()
                .min(source.max_parallelism(&source_global))
                .max(1)
        };
        let base = u32::try_from(next_partition)
            .map_err(|_| internal("too many drivers for u32 partition ids".to_owned()))?;
        // The next member starts past every split of this one (a source can
        // have more splits than drivers), so split ids never overlap. The
        // 2^24 cap only bounds the id space: a source reporting more splits
        // than that would see members' split ids (and seeded draws) overlap.
        let span = drivers.max(source.partitions(&source_global).min(1 << 24));
        // `span` fits `usize`; the end index must fit `u32` as well.
        u32::try_from(next_partition + span as u64)
            .map_err(|_| internal("too many drivers for u32 partition ids".to_owned()))?;
        next_partition += span as u64;
        started.push(MemberStart {
            pipeline,
            operator_deps,
            source_global,
            drivers,
            base_partition: base,
            span: span as u64,
            operator_bases: Vec::new(),
            lanes: lane_count.map(|n| (base, n)),
        });
    }
    // Each operator numbers partitions from its origin (`Pipeline::origins`):
    // its own member below a `UNION ALL`, the union's numbering above it
    // (`numbering.rs`).
    let shapes: Vec<MemberShape<'_>> = started
        .iter()
        .map(|m| MemberShape {
            pipeline: usize::try_from(m.pipeline.index).unwrap_or(usize::MAX),
            base: u64::from(m.base_partition),
            span: m.span,
            single_source: m.pipeline.source.1.single_partition(&m.source_global),
            origins: &m.pipeline.origins,
            keeps: m
                .pipeline
                .operators
                .iter()
                .map(|(_, op)| op.keeps_partitioning())
                .collect(),
            fans: &m.pipeline.fans,
        })
        .collect();
    // A coalesced union's branches share index 0's streams above it, so they
    // must run in turn (Spark's one task). A group whose members run together
    // (a shared operator needs a flush, e.g. a hash-join probe above the
    // union) keeps the plain numbering, which gives each member its own
    // streams: deterministic, though not Spark's one stream (known gap,
    // `numbering.rs`).
    let sequential = started.len() < 2 || members_run_in_turn(&started);
    let numbering = Numbering::new(
        &shapes,
        query.config().union_output_partitioning && sequential,
    );
    // The group's sink is above every fan-in of the first member.
    let top = shapes.first().map_or(
        Origin {
            pipeline: 0,
            level: 0,
        },
        |m| Origin {
            pipeline: m.pipeline,
            level: m.fans.len(),
        },
    );
    let bases: Vec<Vec<u32>> = shapes.iter().map(|m| numbering.bases(m, top)).collect();
    for (m, bases) in started.iter_mut().zip(bases) {
        m.operator_bases = bases;
    }
    // The members whose rows have a source order (an ordered source, and
    // no operator that emits after a cross-driver flush, where a driver
    // held back to keep the order could stall the others) ask the sink to
    // keep it: by member, then by the source's position.
    for (member, m) in started.iter().enumerate() {
        let ops_stream = m.pipeline.operators.iter().all(|(_, op)| !op.needs_flush());
        if m.pipeline.source.1.reports_positions() && ops_stream {
            let member = u32::try_from(member)
                .map_err(|_| internal("too many fan-in members for u32 ids".to_owned()))?;
            let end = m.base_partition + u32::try_from(m.drivers).unwrap_or(u32::MAX);
            sink.expect_ordered(&sink_global, member, m.base_partition..end);
        }
    }
    // One barrier per operator, over the drivers of every pipeline it is in.
    let mut barrier_drivers: HashMap<OperatorId, usize> = HashMap::new();
    for m in &started {
        for (id, _) in &m.pipeline.operators {
            *barrier_drivers.entry(*id).or_default() += m.drivers;
        }
    }
    let barriers: HashMap<OperatorId, Arc<Barrier>> = barrier_drivers
        .into_iter()
        .map(|(id, n)| (id, Arc::new(Barrier::new(n))))
        .collect();
    let total = started.iter().map(|s| s.drivers).sum();
    let countdown = Arc::new(Countdown::new(total));
    Ok(GroupStart {
        sink_global,
        countdown,
        barriers,
        members: started,
    })
}

/// One pipeline of a group: its breaker inputs resolved, its drivers counted.
struct MemberStart {
    pipeline: Pipeline,
    operator_deps: Vec<Option<GlobalState>>,
    source_global: GlobalState,
    drivers: usize,
    /// The first scheduler-assigned partition id of its drivers.
    base_partition: u32,
    /// The partition ids it takes (see `start_group`).
    span: u64,
    /// Per operator: the first partition id it numbers partitions from.
    operator_bases: Vec<u32>,
    /// `(first partition id, lane count)` when its drivers are logical partitions.
    lanes: Option<(u32, usize)>,
}

/// The logical partitions of a pipeline under `krishiv.deterministicOrder`
/// (T094 G1), or `None` for the one-driver mode: the source offers a split, the
/// sink merges in lane order without help, and no operator waits for
/// another driver (a flush barrier). The count is a property of the source,
/// never of the executor's thread count.
fn lane_count(
    query: &QueryContext,
    pipeline: &Pipeline,
    source: &Arc<dyn Source>,
    global: &GlobalState,
) -> Option<usize> {
    if !query.config().deterministic_order || pipeline.preserve_order {
        return None;
    }
    // Spill decisions follow the shared pool, which the lanes' timing moves (G2: a private
    // per-query budget lifts this): under a small limit the one-driver mode keeps its guarantee.
    if query
        .pool()
        .effective_limit()
        .is_some_and(|l| l < LANES_MIN_MEMORY_LIMIT)
    {
        return None;
    }
    if !source.reports_positions()
        || !pipeline.sink.1.lane_safe()
        || pipeline
            .operators
            .iter()
            .any(|(_, op)| op.needs_flush() || !op.lane_safe())
    {
        return None;
    }
    let n = source
        .deterministic_split(global)?
        .min(DETERMINISTIC_LOGICAL_PARTITIONS);
    (n > 1).then_some(n)
}

/// Below this pool limit a deterministic pipeline keeps one driver (see `lane_count`).
const LANES_MIN_MEMORY_LIMIT: usize = 4 << 30;

/// The most logical partitions a deterministic pipeline runs.
pub(crate) const DETERMINISTIC_LOGICAL_PARTITIONS: usize = 16;

/// A pipeline group planned but not yet running: the state its members
/// share and the members in pipeline order.
struct GroupStart {
    sink_global: GlobalState,
    countdown: Arc<Countdown>,
    barriers: HashMap<OperatorId, Arc<Barrier>>,
    members: Vec<MemberStart>,
}

/// Whether the members of `group` run one after another, in pipeline
/// order: a fan-in (`UNION ALL`) whose branches share no operator needing
/// a cross-driver flush. Sequential branches emit their rows in branch
/// order; concurrent ones interleave batches by arrival, so a downstream
/// that observes row order (`LIMIT` without `ORDER BY`, the golden harness
/// over a sorted child) sees nondeterministic rows. A flush barrier waits
/// for every driver's arrival, so a group sharing one stays concurrent.
fn fan_in_is_sequential(group: &GroupStart) -> bool {
    group.members.len() > 1 && members_run_in_turn(&group.members)
}

/// Whether no operator of `members` needs a cross-driver flush (see
/// [`fan_in_is_sequential`]).
fn members_run_in_turn(members: &[MemberStart]) -> bool {
    members
        .iter()
        .all(|m| m.pipeline.operators.iter().all(|(_, op)| !op.needs_flush()))
}

/// The barriers of `member`'s operators.
fn member_barriers(
    member: &MemberStart,
    barriers: &HashMap<OperatorId, Arc<Barrier>>,
) -> Result<Vec<Arc<Barrier>>> {
    member
        .pipeline
        .operators
        .iter()
        .map(|(id, _)| barriers.get(id).cloned())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| internal("an operator without a barrier".to_owned()))
}

/// The scheduler-assigned partition ids of a member's drivers,
/// `base..base + count`. Rejects an id past `u32` instead of wrapping it.
/// Checked before any driver is spawned: a driver spawned before an error
/// would be detached (a dropped [`TaskHandle`]) and never awaited.
fn partitions(base: u32, count: usize) -> Result<std::ops::Range<u32>> {
    let too_many = || internal("too many drivers for u32 partition ids".to_owned());
    let end = u32::try_from(count)
        .ok()
        .and_then(|n| base.checked_add(n))
        .ok_or_else(too_many)?;
    Ok(base..end)
}

/// One driver per partition id of `ids` (from [`partitions`]).
fn drivers(run: &Arc<PipelineRun>, ids: std::ops::Range<u32>) -> Vec<TaskHandle> {
    let base = ids.start;
    ids.map(|partition| spawn_driver(Arc::clone(run), partition, base))
        .collect()
}

/// Starts every member of `group` together, sharing the sink's state.
/// Under `krishiv.deterministicOrder` (one driver per member) the members
/// of a fan-in take turns with a [`Baton`], in member order, so the shared
/// operators see their rows in the same order on every run.
fn spawn_group(
    executor: &Arc<dyn Executor>,
    query: &Arc<QueryContext>,
    group: GroupStart,
) -> Result<(Vec<Arc<PipelineRun>>, Vec<TaskHandle>)> {
    let GroupStart {
        sink_global,
        countdown,
        barriers,
        members,
    } = group;
    let total = members.iter().map(|m| m.drivers).sum();
    let baton = (query.config().deterministic_order
        && members.len() > 1
        && members.iter().all(|m| m.drivers == 1))
    .then(|| Arc::new(Baton::new(members.len())));
    // Every check first, so an error spawns no driver (see [`partitions`]).
    let mut planned = Vec::with_capacity(members.len());
    for m in members {
        let member_barriers = member_barriers(&m, &barriers)?;
        let ids = partitions(m.base_partition, m.drivers)?;
        planned.push((m, member_barriers, ids));
    }
    let mut runs = Vec::with_capacity(planned.len());
    let mut handles = Vec::with_capacity(total);
    for (member, (m, member_barriers, ids)) in planned.into_iter().enumerate() {
        let run = Arc::new(PipelineRun {
            pipeline: m.pipeline,
            operator_deps: m.operator_deps,
            operator_bases: m.operator_bases,
            barriers: member_barriers,
            source_global: m.source_global,
            sink_global: Arc::clone(&sink_global),
            countdown: Arc::clone(&countdown),
            turn: baton.as_ref().map(|b| (Arc::clone(b), member)),
            lanes: m.lanes,
            combine_order: m.lanes.map(|_| Arc::new(CombineOrder::new())),
            query: Arc::clone(query),
            executor: Arc::clone(executor),
        });
        handles.extend(drivers(&run, ids));
        runs.push(run);
    }
    Ok((runs, handles))
}

/// Runs the members of `group` one after another, in pipeline order (see
/// [`fan_in_is_sequential`]): the next member starts once the previous
/// one's drivers have ended, sharing the sink's state, its finalize
/// countdown and the flush barriers exactly as concurrent members do.
async fn run_group_sequential(
    executor: &Arc<dyn Executor>,
    query: &Arc<QueryContext>,
    group: GroupStart,
) -> Result<Vec<Arc<PipelineRun>>> {
    let GroupStart {
        sink_global,
        countdown,
        barriers,
        members,
    } = group;
    let mut runs = Vec::with_capacity(members.len());
    for m in members {
        let member_barriers = member_barriers(&m, &barriers)?;
        let ids = partitions(m.base_partition, m.drivers)?;
        let run = Arc::new(PipelineRun {
            pipeline: m.pipeline,
            operator_deps: m.operator_deps,
            operator_bases: m.operator_bases,
            barriers: member_barriers,
            source_global: m.source_global,
            sink_global: Arc::clone(&sink_global),
            countdown: Arc::clone(&countdown),
            turn: None,
            lanes: m.lanes,
            combine_order: m.lanes.map(|_| Arc::new(CombineOrder::new())),
            query: Arc::clone(query),
            executor: Arc::clone(executor),
        });
        let mut handles = Drivers(FuturesUnordered::new());
        handles.0.extend(drivers(&run, ids));
        await_drivers(query, &mut handles).await;
        record_extra_metrics(query, std::slice::from_ref(&run));
        query.check_cancelled()?;
        runs.push(run);
    }
    Ok(runs)
}

/// A wave's drivers, aborted if the coordinator is dropped mid-wave (so
/// they are never detached).
struct Drivers(FuturesUnordered<TaskHandle>);

impl Drop for Drivers {
    fn drop(&mut self) {
        for h in &self.0 {
            h.abort();
        }
    }
}

/// Awaits every driver. A panic fails the query (`INTERNAL_ERROR`); once
/// the query is cancelled or failed, the remaining drivers are aborted and
/// still awaited, so they have released everything when this returns.
async fn await_drivers(query: &QueryContext, handles: &mut Drivers) {
    let mut aborted = false;
    loop {
        tokio::select! {
            biased;
            next = handles.0.next() => match next {
                None => return,
                Some(Err(e)) => query.fail(e),
                Some(Ok(())) => {}
            },
            () = query.cancelled(), if !aborted => {
                aborted = true;
                for h in &handles.0 {
                    h.abort();
                }
            }
        }
    }
}

#[cfg(test)]
mod filter_order_tests {
    use krishiv_io::FilterId;

    use super::with_filter_order;
    use crate::OperatorId;

    fn sinks(ids: &[u32]) -> Vec<OperatorId> {
        ids.iter().map(|&i| OperatorId(i)).collect()
    }

    /// A consumer starts after the publisher of the filter it applies.
    #[test]
    fn a_consumer_waits_for_its_filters_publisher() {
        let f = FilterId(1);
        let deps = with_filter_order(
            vec![vec![], vec![], vec![0, 1]],
            &sinks(&[0, 1, 2]),
            &[vec![], vec![f], vec![]],
            &[vec![f], vec![], vec![]],
        );
        assert_eq!(deps, [vec![], vec![0], vec![0, 1]]);
    }

    /// Without a publisher in the query (another fragment's filter), or
    /// when it already comes first, nothing changes.
    #[test]
    fn nothing_changes_without_a_publisher_or_when_it_comes_first() {
        let f = FilterId(1);
        let same = vec![vec![], vec![0]];
        let order = |applies: &[Vec<FilterId>], publishes: &[Vec<FilterId>]| {
            with_filter_order(same.clone(), &sinks(&[0, 1]), applies, publishes)
        };
        assert_eq!(order(&[vec![], vec![f]], &[vec![], vec![]]), same);
        assert_eq!(order(&[vec![], vec![f]], &[vec![f], vec![]]), same);
    }

    /// A publisher that runs after the consumer, directly, transitively or
    /// through a fan-in member, adds nothing: the query would never start.
    #[test]
    fn a_filter_never_makes_a_cycle() {
        let f = FilterId(1);
        // 2 depends on 1, which depends on 0: 0 applies 2's filter.
        let chain = vec![vec![], vec![0], vec![1]];
        let deps = with_filter_order(
            chain.clone(),
            &sinks(&[0, 1, 2]),
            &[vec![f], vec![], vec![]],
            &[vec![], vec![], vec![f]],
        );
        assert_eq!(deps, chain);
        // 0 and 1 share a sink and start together; 2 depends on 1, so 0
        // cannot wait for 2.
        let fan_in = vec![vec![], vec![], vec![1]];
        let deps = with_filter_order(
            fan_in.clone(),
            &sinks(&[9, 9, 2]),
            &[vec![f], vec![], vec![]],
            &[vec![], vec![], vec![f]],
        );
        assert_eq!(deps, fan_in);
        // The same filter's own pipeline.
        let deps = with_filter_order(vec![vec![]], &sinks(&[0]), &[vec![f]], &[vec![f]]);
        assert_eq!(deps, [Vec::<usize>::new()]);
        // Two filters in opposite directions: only the first edge is added.
        let g = FilterId(2);
        let deps = with_filter_order(
            vec![vec![], vec![]],
            &sinks(&[0, 1]),
            &[vec![f], vec![g]],
            &[vec![g], vec![f]],
        );
        assert_eq!(deps, [vec![1], vec![]]);
    }
}
