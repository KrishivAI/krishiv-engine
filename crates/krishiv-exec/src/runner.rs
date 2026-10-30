// SPDX-License-Identifier: Apache-2.0
//! Running a query in process: [`Scheduler`] and its embedded
//! implementation, [`QueryRunner`].
//!
//! `submit` instantiates the [`PhysicalPlan`] with the caller's
//! [`WorkerContext`] ([`instantiate`](crate::physical::instantiate)) on
//! the executor's blocking pool when its future runs,
//! splits the operator graph into pipelines and spawns a coordinator task,
//! which starts each pipeline once its dependencies have finalized:
//! `min(worker slots, source parallelism)` drivers per pipeline (one if it
//! must keep order). The pipelines of a fan-in (`UNION ALL`) end in one
//! sink: they run one after another, in pipeline order, when none of
//! their shared operators needs a cross-driver flush, and start together
//! otherwise, sharing the sink's state either way. The coordinator awaits every driver; the first error
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
use krishiv_memory::SpillOwner;
use krishiv_types::batch::Batch;

use crate::channel::{Channel, ResultSink, ResultStream};
use crate::context::{OperatorId, QueryContext, QueryId, TaskContext};
use crate::driver::{PipelineRun, spawn_driver};
use crate::error::{Result, cancelled, internal};
use crate::executor::{Executor, TaskHandle};
use crate::metrics::OperatorEntry;
use crate::morsel::{Barrier, Countdown};
use crate::operator::GlobalState;
use crate::physical::{PhysicalPlan, WorkerContext, instantiate};
use crate::pipeline::{ExecNode, Pipeline, PipelineBuilder};
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
        let mut b = PipelineBuilder::new(Arc::new(ResultSink {
            channel: Arc::clone(&channel),
        }));
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
        drop(self.executor.spawn(
            query.class(),
            Box::pin(async move {
                if let Err(e) = run_pipelines(&executor, &q, pipelines).await {
                    q.fail(e);
                }
                q.blocking_done().await;
                finish.complete();
            }),
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

/// Runs the pipelines in dependency waves. The coordinator owns every
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
    let deps: Vec<Vec<usize>> = pipelines.iter().map(|p| p.deps.clone()).collect();
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
    while done.iter().any(|d| !d) {
        let ready: Vec<Vec<usize>> = (0..todo.len())
            .filter(|&i| !done[i] && group(i)[0] == i)
            .map(group)
            .filter(|g| g.iter().all(|&m| deps[m].iter().all(|&d| done[d])))
            .collect();
        if ready.is_empty() {
            // Unreachable: dependencies form a tree.
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

/// The breakers whose finalized state `p` reads, once each.
fn inputs(p: &Pipeline) -> Vec<OperatorId> {
    let mut ids: Vec<OperatorId> = p
        .input
        .iter()
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
    let sink_global = sink.global_state(&ctx((*sink_id, sink.name())))?;
    let sink = Arc::clone(sink);
    let mut started = Vec::with_capacity(members.len());
    // Scheduler-assigned globally unique partition ids: the drivers of
    // every member share one id space, in pipeline order, so two drivers
    // never compute the same partition (stateful functions own one value
    // stream per partition). An index past `u32` is rejected rather than
    // wrapped to `u32::MAX`, which would collide.
    let mut next_partition: u64 = 0;
    for pipeline in members {
        let (source_id, source) = &pipeline.source;
        let mut source_ctx = ctx((*source_id, source.name()));
        if let Some(input) = &pipeline.input {
            source_ctx = source_ctx.with_dependency(read(input)?);
        }
        let operator_deps = pipeline
            .operator_inputs
            .iter()
            .map(|i| i.as_ref().map(read).transpose())
            .collect::<Result<Vec<_>>>()?;
        let source_global = source.global_state(&source_ctx)?;
        let drivers = if pipeline.preserve_order {
            1
        } else {
            executor
                .parallelism()
                .min(source.max_parallelism(&source_global))
                .max(1)
        };
        let base = u32::try_from(next_partition)
            .map_err(|_| internal("too many drivers for u32 partition ids".to_owned()))?;
        // `drivers` fits `usize`; the end index must fit `u32` as well.
        u32::try_from(next_partition + drivers as u64)
            .map_err(|_| internal("too many drivers for u32 partition ids".to_owned()))?;
        next_partition += drivers as u64;
        started.push(MemberStart {
            pipeline,
            operator_deps,
            source_global,
            drivers,
            base_partition: base,
        });
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
}

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
    group.members.len() > 1
        && group
            .members
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

/// One driver per index of `member`: the scheduler-assigned partition ids
/// `base..base + count`. Rejects an id past `u32` instead of wrapping it.
fn drivers(run: &Arc<PipelineRun>, base: u32, count: usize) -> Result<Vec<TaskHandle>> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let offset = u32::try_from(i)
            .map_err(|_| internal("too many drivers for u32 partition ids".to_owned()))?;
        let partition = base
            .checked_add(offset)
            .ok_or_else(|| internal("too many drivers for u32 partition ids".to_owned()))?;
        out.push(spawn_driver(Arc::clone(run), partition));
    }
    Ok(out)
}

/// Starts every member of `group` together, sharing the sink's state.
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
    let mut runs = Vec::with_capacity(members.len());
    let mut handles = Vec::with_capacity(total);
    for m in members {
        let member_barriers = member_barriers(&m, &barriers)?;
        let run = Arc::new(PipelineRun {
            pipeline: m.pipeline,
            operator_deps: m.operator_deps,
            barriers: member_barriers,
            source_global: m.source_global,
            sink_global: Arc::clone(&sink_global),
            countdown: Arc::clone(&countdown),
            query: Arc::clone(query),
            executor: Arc::clone(executor),
        });
        handles.extend(drivers(&run, m.base_partition, m.drivers)?);
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
        let run = Arc::new(PipelineRun {
            pipeline: m.pipeline,
            operator_deps: m.operator_deps,
            barriers: member_barriers,
            source_global: m.source_global,
            sink_global: Arc::clone(&sink_global),
            countdown: Arc::clone(&countdown),
            query: Arc::clone(query),
            executor: Arc::clone(executor),
        });
        let mut handles = Drivers(FuturesUnordered::new());
        handles
            .0
            .extend(drivers(&run, m.base_partition, m.drivers)?);
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
