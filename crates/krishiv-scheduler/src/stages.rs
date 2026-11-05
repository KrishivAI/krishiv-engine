// SPDX-License-Identifier: Apache-2.0
//! [`StageRunner`]: a [`Scheduler`] that runs a plan as stages of tasks in
//! this process (design P2-T176 §2.5, unit 1). It fragments the plan
//! ([`fragment`]), runs the stages bottom-up, each to completion (blocking
//! exchanges), every task as its own instantiation of its stage's fragment
//! over a [`LocalShuffle`], and returns the root stage's rows.
//!
//! Each task gets a [`TaskSpec`] and nothing else: its partition, the
//! reduce partitions it reads of each exchange and from how many map tasks,
//! the reduce partitions its own exchange writes, and its scans' splits. So
//! a task that ran here runs the same anywhere its fragment's bytes and its
//! spec are shipped (the simulation, P2-T176 unit 2).
//!
//! Task counts: the root stage is one task, on the caller's query context
//! (its result is the query's). A map stage reading a hash or round-robin
//! exchange runs `partitions` tasks, one per reduce partition; a map stage
//! that only scans runs one task per split group when a [`StageSplits`]
//! deals its scans, else one; a stage that must run whole
//! ([`Stage::single`]), or that both scans and reads a hash exchange, runs
//! one task, which reads every reduce partition. An exchange writes as many
//! reduce partitions as the stage reading it runs tasks (one for a gather or
//! broadcast, design Q3); task `t` reads reduce partition `t`.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};

use futures::StreamExt as _;
use futures::future::{BoxFuture, Either};
use futures::stream::BoxStream;
use futures::stream::FuturesUnordered;
use krishiv_common::error::{InternalError, KrishivError, OperationCanceled};
use krishiv_exec::exchange::reduce_partitions;
use krishiv_exec::physical::Partitioning;
use krishiv_exec::physical::{
    ExchangeInput, PhysicalPlan, ScanId, ScanSplits, TaskSpec, WorkerContext, exchange_schemas,
    instantiate_task,
};
use krishiv_exec::{Executor, QueryContext, QueryId, QueryRunner, Scheduler};
use krishiv_shuffle::LocalShuffle;
use krishiv_types::batch::Batch;

use crate::fragment::{Stage, fragment};

/// The map tasks of each running query, by query (the query itself first).
type Running = Mutex<HashMap<QueryId, Vec<Weak<QueryContext>>>>;

type Result<T> = std::result::Result<T, KrishivError>;

/// Deals a stage's scans to its tasks: the coordinator's split assignment
/// (P2-T178), over the tables the query opened.
///
/// Only the scans of the fragment's streaming pipeline may be dealt: the
/// probe side of each join, never a build side, a subquery's plan, a shared
/// plan's body or a late fetch, which every task reads whole (no entry). A
/// dealt build side would join each task's probe rows with part of the
/// build and lose rows.
pub trait StageSplits: Send + Sync + fmt::Debug {
    /// The splits of each task of a stage running `fragment`, at most
    /// `tasks` of them; `None` to read every scan whole in one task.
    ///
    /// # Errors
    /// A scan whose splits cannot be planned.
    fn splits(
        &self,
        fragment: &PhysicalPlan,
        tasks: u32,
    ) -> Result<Option<Vec<BTreeMap<ScanId, ScanSplits>>>>;
}

/// The in-process stage scheduler (see the module docs). The worker it is
/// given must not set a value service of its own
/// ([`WorkerContext::with_values`]): every task's instantiation needs fresh
/// values, as one worker's does.
pub struct StageRunner {
    executor: Arc<dyn Executor>,
    /// Runs map tasks, each on a query context of its own.
    tasks: Arc<QueryRunner>,
    /// Runs root stages, on the caller's query context.
    roots: Arc<QueryRunner>,
    partitions: u32,
    splits: Option<Arc<dyn StageSplits>>,
    next_task: Arc<AtomicU64>,
    /// The map tasks running for each query, cancelled with it.
    running: Arc<Running>,
}

impl fmt::Debug for StageRunner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StageRunner")
            .field("partitions", &self.partitions)
            .field("splits", &self.splits)
            .finish_non_exhaustive()
    }
}

/// What the runner decided for one stage.
#[derive(Debug, Clone)]
struct Plan {
    tasks: u32,
    /// The reduce partitions its output exchange writes.
    writes: u32,
    /// Each task's scan splits (empty: read whole).
    splits: Vec<BTreeMap<ScanId, ScanSplits>>,
}

impl StageRunner {
    /// A runner on `executor`, with `partitions` reduce partitions per hash
    /// exchange (design §2.5: `2^k`, at least the worker slots).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for no partitions.
    pub fn new(executor: Arc<dyn Executor>, partitions: u32) -> Result<Self> {
        if partitions == 0 {
            return Err(internal("a stage runner without partitions".to_owned()));
        }
        Ok(Self {
            tasks: Arc::new(QueryRunner::new(Arc::clone(&executor))),
            roots: Arc::new(QueryRunner::new(Arc::clone(&executor))),
            executor,
            partitions,
            splits: None,
            next_task: Arc::default(),
            running: Arc::default(),
        })
    }

    /// The same runner dealing scans to tasks through `splits`.
    #[must_use]
    pub fn with_splits(mut self, splits: Arc<dyn StageSplits>) -> Self {
        self.splits = Some(splits);
        self
    }
}

/// The task specs of each stage of `stages` (by stage id), every task at
/// attempt 0 of stage attempt 0: the in-process runner's task plan (see the
/// module docs), which a coordinator of several workers ships as is. Hash
/// exchanges get `partitions` reduce partitions; `splits` deals the scans of
/// leaf stages.
///
/// # Errors
/// `INTERNAL_ERROR` for no partitions or an exchange no stage reads;
/// `splits`' errors.
pub fn stage_tasks(
    stages: &[Stage],
    partitions: u32,
    splits: Option<&dyn StageSplits>,
) -> Result<Vec<Vec<TaskSpec>>> {
    if partitions == 0 {
        return Err(internal("a task plan without partitions".to_owned()));
    }
    let plans = plans(stages, partitions, splits)?;
    stages
        .iter()
        .map(|stage| {
            let tasks = plans.get(stage.id).map_or(1, |p| p.tasks);
            (0..tasks).map(|t| spec(stage, &plans, t)).collect()
        })
        .collect()
}

/// The task count, written partitions and splits of each stage.
fn plans(stages: &[Stage], partitions: u32, splits: Option<&dyn StageSplits>) -> Result<Vec<Plan>> {
    let mut plans: Vec<Option<Plan>> = vec![None; stages.len()];
    // Readers come after what they read: decide from the root down.
    for stage in stages.iter().rev() {
        // A subquery's input is read whole by every task: it does not
        // partition the stage.
        let reads_hash = stage.inputs.iter().any(|i| {
            !i.subquery
                && matches!(
                    i.partitioning,
                    Partitioning::Hash { .. } | Partitioning::RoundRobin
                )
        });
        let (tasks, splits) =
            if stage.output.is_none() || stage.single || (reads_hash && stage.scans) {
                (1, Vec::new())
            } else if reads_hash {
                (partitions, Vec::new())
            } else if stage.scans
                && let Some(splits) = splits
                && let Some(dealt) = splits.splits(&stage.plan, partitions)?
                && !dealt.is_empty()
            {
                let tasks = u32::try_from(dealt.len())
                    .map_err(|_| internal(format!("{} tasks", dealt.len())))?;
                (tasks, dealt)
            } else {
                (1, Vec::new())
            };
        let writes = match &stage.output {
            None => 0,
            Some((exchange, partitioning)) => {
                let reader = stages
                    .iter()
                    .find(|s| s.inputs.iter().any(|i| i.exchange == *exchange))
                    .and_then(|s| plans.get(s.id).cloned().flatten())
                    .ok_or_else(|| {
                        internal(format!("exchange {} has no reading stage", exchange.0))
                    })?;
                reduce_partitions(partitioning, reader.tasks)
            }
        };
        if let Some(slot) = plans.get_mut(stage.id) {
            *slot = Some(Plan {
                tasks,
                writes,
                splits,
            });
        }
    }
    plans
        .into_iter()
        .map(|p| p.ok_or_else(|| internal("a stage without a plan".to_owned())))
        .collect()
}

/// Task `partition`'s spec of `stage`.
fn spec(stage: &Stage, plans: &[Plan], partition: u32) -> Result<TaskSpec> {
    let plan = plans
        .get(stage.id)
        .ok_or_else(|| internal("a stage without a plan".to_owned()))?;
    let mut exchanges = BTreeMap::new();
    for input in &stage.inputs {
        let writer = plans
            .get(input.stage)
            .ok_or_else(|| internal("an input stage without a plan".to_owned()))?;
        let reduce = if writer.writes > 1 && plan.tasks > 1 && !input.subquery {
            vec![partition]
        } else {
            (0..writer.writes).collect()
        };
        exchanges.insert(
            input.exchange,
            ExchangeInput {
                stage_attempt: 0,
                maps: writer.tasks,
                reduce,
            },
        );
    }
    Ok(TaskSpec {
        stage: u32::try_from(stage.id).map_err(|_| internal(format!("stage {}", stage.id)))?,
        stage_attempt: 0,
        partition,
        partitions: plan.tasks,
        attempt: 0,
        scans: plan
            .splits
            .get(partition as usize)
            .cloned()
            .unwrap_or_default(),
        exchanges,
        output: stage.output.as_ref().map(|_| plan.writes),
    })
}

impl StageRunner {
    /// `plan` instantiated as `task` on the blocking pool (instantiation
    /// starts a deep-stack thread).
    async fn instantiate(
        executor: &Arc<dyn Executor>,
        plan: PhysicalPlan,
        worker: WorkerContext,
        task: TaskSpec,
    ) -> Result<Arc<dyn krishiv_exec::ExecNode>> {
        let built = Arc::new(Mutex::new(None));
        let out = Arc::clone(&built);
        executor
            .spawn_blocking(Box::new(move || {
                *out.lock().unwrap_or_else(PoisonError::into_inner) =
                    Some(instantiate_task(&plan, &worker, &task));
            }))
            .await?;
        built
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .ok_or_else(|| internal("an instantiation without a result".to_owned()))?
    }

    /// Runs one map task to its end on a query context of its own, a child
    /// of `query`'s pool.
    async fn map_task(
        &self,
        stage: &Stage,
        worker: WorkerContext,
        spec: TaskSpec,
        query: &Arc<QueryContext>,
    ) -> Result<()> {
        query.check_cancelled()?;
        let name = format!("stage-{}-task-{}", spec.stage, spec.partition);
        let id = QueryId(self.next_task.fetch_add(1, Ordering::Relaxed));
        let task = QueryContext::new(
            id,
            query.class(),
            query.config().clone(),
            query.pool().child(&name, None),
        );
        // Registered before anything runs: a cancel from here on reaches it.
        self.running
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(query.id())
            .or_default()
            .push(Arc::downgrade(&task));
        let node = Self::instantiate(&self.executor, stage.plan.clone(), worker, spec).await?;
        query.check_cancelled()?;
        let mut stream = self.tasks.execute(node, Arc::clone(&task)).await?;
        let outcome = {
            let drained = std::pin::pin!(async {
                while let Some(batch) = stream.next().await {
                    if batch?.num_rows() > 0 {
                        return Err(internal(format!("map task {name} returned rows")));
                    }
                }
                Ok(())
            });
            // The task ends with the query: a cancel or a sibling's failure.
            let cancelled = std::pin::pin!(query.cancelled());
            match futures::future::select(drained, cancelled).await {
                Either::Left((outcome, _)) => outcome,
                Either::Right(((), _)) => Err(query.error().unwrap_or_else(canceled)),
            }
        };
        if outcome.is_err() {
            task.cancel();
        }
        drop(stream);
        // Its drivers have ended and its memory is back before it returns.
        task.finished().await;
        outcome?;
        task.error().map_or(Ok(()), Err)
    }

    async fn run(
        &self,
        plan: PhysicalPlan,
        worker: WorkerContext,
        query: Arc<QueryContext>,
    ) -> Result<BoxStream<'static, Result<Batch>>> {
        // One scope for the query: its shuffle and every task spill there.
        let (worker, shuffle) = match worker.spill_scope()? {
            Some(scope) => (
                worker.with_spill(scope.clone()),
                LocalShuffle::on_disk(scope),
            ),
            None => (worker, LocalShuffle::in_memory()),
        };
        let schemas = {
            let (plan, worker) = (plan.clone(), worker.clone());
            let out = Arc::new(Mutex::new(None));
            let slot = Arc::clone(&out);
            self.executor
                .spawn_blocking(Box::new(move || {
                    *slot.lock().unwrap_or_else(PoisonError::into_inner) =
                        Some(exchange_schemas(&plan, &worker));
                }))
                .await?;
            let schemas = out
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take()
                .ok_or_else(|| internal("exchange schemas without a result".to_owned()))?;
            schemas?
        };
        let stages = fragment(&plan, &schemas)?;
        let tasks = stage_tasks(&stages, self.partitions, self.splits.as_deref())?;
        let worker = worker.with_exchange(Arc::new(shuffle));
        let (root, maps) = stages
            .split_last()
            .ok_or_else(|| internal("a plan without stages".to_owned()))?;
        let specs = |stage: &Stage| {
            tasks
                .get(stage.id)
                .cloned()
                .ok_or_else(|| internal("a stage without tasks".to_owned()))
        };
        for stage in maps {
            let mut runs = specs(stage)?
                .into_iter()
                .map(|spec| {
                    let worker = worker.clone();
                    let query = &query;
                    async move { self.map_task(stage, worker, spec, query).await }
                })
                .collect::<FuturesUnordered<_>>();
            // The first failure fails the query, which ends the other tasks;
            // each returns only once its memory is released.
            let mut failed = None;
            while let Some(outcome) = runs.next().await {
                if let Err(e) = outcome
                    && failed.is_none()
                {
                    query.fail(e.clone());
                    failed = Some(e);
                }
            }
            if let Some(e) = failed {
                return Err(e);
            }
            query.check_cancelled()?;
        }
        let spec = specs(root)?
            .into_iter()
            .next()
            .ok_or_else(|| internal("a root stage without a task".to_owned()))?;
        let node = Self::instantiate(&self.executor, root.plan.clone(), worker, spec).await?;
        query.check_cancelled()?;
        self.roots.execute(node, query).await
    }
}

impl StageRunner {
    /// The number of queries whose submission is running: what a cancel
    /// can still reach through the runner.
    #[must_use]
    pub fn running_queries(&self) -> usize {
        self.running
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

/// A query's entry in [`StageRunner::running`], removed when dropped: when
/// its submission ends, or is dropped unfinished (a timeout, a client gone).
struct Registered {
    running: Arc<Running>,
    id: QueryId,
}

impl Registered {
    fn new(running: &Arc<Running>, query: &Arc<QueryContext>) -> Self {
        let id = query.id();
        running
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(id, vec![Arc::downgrade(query)]);
        Self {
            running: Arc::clone(running),
            id,
        }
    }
}

impl Drop for Registered {
    fn drop(&mut self) {
        self.running
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&self.id);
    }
}

impl Scheduler for StageRunner {
    fn submit(
        &self,
        plan: PhysicalPlan,
        worker: WorkerContext,
        query: Arc<QueryContext>,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<Batch>>>> {
        let runner = Self {
            executor: Arc::clone(&self.executor),
            tasks: Arc::clone(&self.tasks),
            roots: Arc::clone(&self.roots),
            partitions: self.partitions,
            splits: self.splits.clone(),
            next_task: Arc::clone(&self.next_task),
            running: Arc::clone(&self.running),
        };
        // The query itself first: a cancel before or between its stages
        // ends it too.
        let registered = Registered::new(&self.running, &query);
        Box::pin(async move {
            let out = runner.run(plan, worker, query).await;
            drop(registered);
            out
        })
    }

    fn cancel(&self, query: QueryId) {
        let tasks = self
            .running
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&query)
            .cloned()
            .unwrap_or_default();
        for t in tasks.iter().filter_map(Weak::upgrade) {
            t.cancel();
        }
        self.roots.cancel(query);
    }
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

fn canceled() -> KrishivError {
    OperationCanceled {}.into()
}
