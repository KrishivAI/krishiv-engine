// SPDX-License-Identifier: Apache-2.0
//! The simulated coordinator (design P2-T176 §2.5, unit 2): a [`Scheduler`]
//! that fragments a plan, ships each map stage's fragment as
//! `krishiv.plan.v1` bytes with a task descriptor per task to worker
//! `partition mod W` over simulated TCP, runs the stages bottom-up (blocking
//! exchanges), and runs the root stage itself, reading every map output from
//! the workers. A failed task runs again as its next attempt, up to
//! [`Coordinator::MAX_ATTEMPTS`]; the first committed attempt of each map
//! partition is what readers read (I10).
//!
//! The coordinator records a trace of what it shipped and what came back
//! ([`Coordinator::trace`]): the same seed must give the same trace. A
//! submitted query must carry an execution seed (`ExecConfig::seed`): its
//! root stage runs on the caller's context with it, and map tasks derive
//! theirs from it.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::task::{Context, Poll};

use futures::Stream;
use futures::StreamExt as _;
use futures::future::BoxFuture;
use futures::stream::{BoxStream, FuturesUnordered};
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_exec::physical::{
    PhysicalPlan, TaskSpec, WorkerContext, instantiate_task, plan_schemas,
};
use krishiv_exec::{ExecConfig, Executor, QueryContext, QueryId, QueryRunner, Scheduler};
use krishiv_scheduler::{
    Partials, StageSplits, StageValues, fragment, merge_partials, plan_data_schemas,
    split_nested_loops, stage_tasks,
};
use krishiv_types::batch::Batch;
use turmoil::net::TcpStream;

use crate::exchange::{HostExchange, Workers};
use crate::task::{decode_partials, encode_task};
use crate::wire::{Reply, Request};

type Result<T> = std::result::Result<T, KrishivError>;

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// The coordinator of a simulated cluster (see the module docs). It runs on
/// the coordinator host's runtime.
#[derive(Clone)]
pub struct Coordinator {
    inner: Arc<Inner>,
}

struct Inner {
    workers: Workers,
    partitions: u32,
    roots: QueryRunner,
    trace: Mutex<Vec<String>>,
    next_job: AtomicU64,
    /// The queries whose submission runs, so a cancel reaches their map
    /// stages too (checked between stages; a running stage's tasks finish).
    running: Mutex<HashMap<QueryId, Weak<QueryContext>>>,
    /// Deals each stage's scans to its tasks (P2-T178); `None` runs every
    /// scanning stage as one task.
    splits: Mutex<Option<Arc<dyn StageSplits>>>,
    /// The highest task attempt any submission started: a write's commit
    /// sweeps the files of failed attempts only when one above 0 ran.
    max_attempt: AtomicU32,
}

impl std::fmt::Debug for Coordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Coordinator")
            .field("workers", &self.inner.workers)
            .field("partitions", &self.inner.partitions)
            .finish_non_exhaustive()
    }
}

impl Coordinator {
    /// The most attempts of one task.
    pub const MAX_ATTEMPTS: u32 = 3;

    /// A coordinator over `workers`, with `partitions` reduce partitions per
    /// hash exchange, running root stages on `executor`.
    #[must_use]
    pub fn new(executor: Arc<dyn Executor>, workers: Workers, partitions: u32) -> Self {
        Self {
            inner: Arc::new(Inner {
                workers,
                partitions,
                roots: QueryRunner::new(executor),
                trace: Mutex::default(),
                next_job: AtomicU64::new(0),
                running: Mutex::default(),
                splits: Mutex::default(),
                max_attempt: AtomicU32::new(0),
            }),
        }
    }

    /// Deals the scans of each stage to its tasks through `splits` from the
    /// next submission on: the tasks' file splits travel in their
    /// descriptors to the workers that read them (P2-T178).
    pub fn deal_scans(&self, splits: Arc<dyn StageSplits>) {
        *self
            .inner
            .splits
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(splits);
    }

    /// The highest task attempt any submission so far started (0 when no
    /// task was retried). A file write run here passes it to
    /// `DirectoryWrite::note_attempt` before its commit, so the files a
    /// failed attempt left are swept instead of committed next to the
    /// retry's.
    #[must_use]
    pub fn max_attempt(&self) -> u32 {
        self.inner.max_attempt.load(Ordering::Relaxed)
    }

    /// The events so far: each task shipped and its outcome, in order.
    #[must_use]
    pub fn trace(&self) -> Vec<String> {
        self.inner
            .trace
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl Inner {
    fn record(&self, event: String) {
        self.trace
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(event);
    }

    /// Runs `task` of a stage on its worker until an attempt succeeds; the
    /// filter partials of the attempt that did (I10: one per partition).
    async fn run_task(
        &self,
        job: u64,
        fragment: Arc<[u8]>,
        mut task: TaskSpec,
        settings: &ExecConfig,
    ) -> Result<Partials> {
        let owner = self.workers.owner(task.partition)?;
        let host = self
            .workers
            .names
            .get(owner)
            .cloned()
            .ok_or_else(|| internal(format!("no worker {owner}")))?;
        loop {
            let event = format!(
                "job {job} stage {} task {} attempt {} on {host}",
                task.stage, task.partition, task.attempt
            );
            let reply = async {
                let mut stream = TcpStream::connect((host.as_str(), self.workers.port))
                    .await
                    .map_err(|e| internal(format!("connect to {host}: {e}")))?;
                Request::Run {
                    job,
                    fragment: fragment.to_vec(),
                    task: encode_task(&task, settings)?,
                }
                .write(&mut stream)
                .await?;
                Reply::read(&mut stream).await
            }
            .await;
            match reply {
                Ok(Reply::Done(partials)) => {
                    self.record(format!("{event}: done"));
                    return decode_partials(&partials);
                }
                Ok(Reply::Failed(message)) => {
                    self.record(format!("{event}: failed: {message}"));
                    if task.attempt + 1 >= Coordinator::MAX_ATTEMPTS {
                        return Err(internal(format!("{event}: {message}")));
                    }
                    task.attempt += 1;
                    self.max_attempt.fetch_max(task.attempt, Ordering::Relaxed);
                }
                Ok(other) => return Err(internal(format!("{event}: answered {other:?}"))),
                Err(e) => return Err(e),
            }
        }
    }

    async fn submit(
        &self,
        plan: PhysicalPlan,
        worker: WorkerContext,
        query: Arc<QueryContext>,
    ) -> Result<BoxStream<'static, Result<Batch>>> {
        // The root stage runs on the caller's query context, so its seed is
        // the query's: an unseeded query would run its root stage's tables
        // on OS-random seeds, and the run would not replay.
        if query.config().seed.is_none() {
            return Err(internal(
                "a simulated query needs an execution seed (ExecConfig::seed) to replay".to_owned(),
            ));
        }
        let job = self.next_job.fetch_add(1, Ordering::Relaxed);
        let release = Release {
            job,
            workers: self.workers.clone(),
        };
        // The plan's own schemas; a plan written without them is
        // instantiated, unrun, to learn them.
        let mut schemas = match plan_data_schemas(&plan) {
            Some(s) => s,
            None => plan_schemas(&plan, &worker)?,
        };
        // One task per stage: nothing to split.
        let plan = if self.partitions > 1 {
            split_nested_loops(&plan, &mut schemas)?
        } else {
            plan
        };
        let stages = fragment(&plan, &schemas.exchanges)?;
        let splits = self
            .splits
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let tasks = stage_tasks(&stages, self.partitions, splits.as_deref())?;
        let (root, maps) = stages
            .split_last()
            .ok_or_else(|| internal("a plan without stages".to_owned()))?;
        // The filters and bitmaps earlier stages computed, merged here and
        // shipped in the later tasks' descriptors.
        let mut values = StageValues::default();
        for stage in maps {
            let bytes: Arc<[u8]> = krishiv_plan_codec::encode(&stage.plan)
                .map_err(|e| internal(format!("stage {}: {e}", stage.id)))?
                .into();
            let specs = tasks.get(stage.id).cloned().unwrap_or_default();
            let mut runs: FuturesUnordered<_> = specs
                .into_iter()
                .map(|mut spec| {
                    values.give(&mut spec);
                    self.run_task(job, Arc::clone(&bytes), spec, query.config())
                })
                .collect();
            let mut failed = None;
            let mut partials = Vec::new();
            while let Some(outcome) = runs.next().await {
                match outcome {
                    Ok(p) => partials.push(p),
                    Err(e) => {
                        failed.get_or_insert(e);
                    }
                }
            }
            if let Some(e) = failed {
                return Err(e);
            }
            values.extend(merge_partials(&stage.plan, &partials)?);
            query.check_cancelled()?;
        }
        let mut spec = tasks
            .get(root.id)
            .and_then(|t| t.first())
            .cloned()
            .ok_or_else(|| internal("a root stage without a task".to_owned()))?;
        values.give(&mut spec);
        let worker = worker.with_exchange(Arc::new(HostExchange::coordinator(
            job,
            self.workers.clone(),
        )));
        let node = instantiate_task(&root.plan, &worker, &spec)?;
        let inner = self.roots.execute(node, query).await?;
        Ok(Box::pin(Releasing {
            inner,
            release: Some(release),
        }))
    }
}

impl Scheduler for Coordinator {
    fn submit(
        &self,
        plan: PhysicalPlan,
        worker: WorkerContext,
        query: Arc<QueryContext>,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<Batch>>>> {
        let inner = Arc::clone(&self.inner);
        Box::pin(async move {
            let id = query.id();
            inner
                .running
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(id, Arc::downgrade(&query));
            let out = inner.submit(plan, worker, query).await;
            inner
                .running
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&id);
            out
        })
    }

    fn cancel(&self, query: QueryId) {
        let running = self
            .inner
            .running
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&query)
            .and_then(Weak::upgrade);
        if let Some(q) = running {
            q.cancel();
        }
        self.inner.roots.cancel(query);
    }
}

/// Releases a job's map outputs on every worker when dropped: when its
/// result stream ends or is dropped, or its submission fails.
struct Release {
    job: u64,
    workers: Workers,
}

impl Drop for Release {
    fn drop(&mut self) {
        let (job, workers) = (self.job, self.workers.clone());
        // On the coordinator host's runtime; outside one (a test dropping a
        // stream after the simulation) the workers are gone anyway.
        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            rt.spawn(async move {
                for host in workers.names.iter() {
                    if let Ok(mut s) = TcpStream::connect((host.as_str(), workers.port)).await
                        && (Request::Release { job }).write(&mut s).await.is_ok()
                    {
                        let _ = Reply::read(&mut s).await;
                    }
                }
            });
        }
    }
}

/// The root stage's rows, releasing the job as soon as they end (the workers
/// drop its map outputs while the caller may still hold the stream), or once
/// dropped before that.
struct Releasing {
    inner: BoxStream<'static, Result<Batch>>,
    release: Option<Release>,
}

impl Stream for Releasing {
    type Item = Result<Batch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let next = self.inner.as_mut().poll_next(cx);
        if matches!(next, Poll::Ready(None)) {
            // `Release` sends from its `Drop`, on this host's runtime.
            self.release = None;
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use krishiv_exec::TokioExecutor;

    use super::*;
    use crate::testing;

    /// Runs one task against a fake worker `srv` that answers with `reply`,
    /// and returns the task's error and the coordinator's trace.
    fn run_one(reply: fn() -> Reply, host: &'static str) -> (String, Vec<String>) {
        let mut sim = testing::sim();
        testing::answering(&mut sim, "srv", reply);
        let out = Arc::new(Mutex::new((String::new(), Vec::new())));
        let seen = Arc::clone(&out);
        sim.client("coordinator", async move {
            let coordinator = Coordinator::new(
                TokioExecutor::deterministic(1)?,
                testing::workers(&[host]),
                1,
            );
            let e = coordinator
                .inner
                .run_task(
                    0,
                    Arc::from(Vec::new()),
                    TaskSpec::whole(),
                    &ExecConfig::default(),
                )
                .await
                .unwrap_err();
            *seen.lock().unwrap() = (e.message().to_owned(), coordinator.trace());
            Ok::<(), Box<dyn std::error::Error>>(())
        });
        sim.run().unwrap();
        out.lock().unwrap().clone()
    }

    #[test]
    fn a_task_that_always_fails_ends_the_run_after_its_attempts() {
        let (message, trace) = run_one(|| Reply::Failed("INTERNAL_ERROR: boom".to_owned()), "srv");
        assert!(
            message.contains("attempt 2") && message.contains("boom"),
            "{message}"
        );
        assert_eq!(trace.len(), Coordinator::MAX_ATTEMPTS as usize, "{trace:?}");
        assert!(
            trace[0].ends_with("failed: INTERNAL_ERROR: boom"),
            "{trace:?}"
        );
    }

    #[test]
    fn a_reply_that_is_not_an_answer_to_a_run_is_refused() {
        let (message, _) = run_one(|| Reply::End, "srv");
        assert!(message.contains("answered End"), "{message}");
    }

    #[test]
    fn an_unreachable_worker_fails_the_task() {
        let (message, trace) = run_one(|| Reply::Done(Vec::new()), "ghost");
        assert!(message.contains("connect to ghost"), "{message}");
        assert!(trace.is_empty(), "{trace:?}");
    }

    #[tokio::test]
    async fn a_coordinator_prints_its_shape_and_ignores_a_cancel_of_an_unknown_query() {
        let coordinator = Coordinator::new(
            TokioExecutor::deterministic(1).unwrap(),
            testing::workers(&["a", "b"]),
            4,
        );
        let text = format!("{coordinator:?}");
        assert!(
            text.contains("Coordinator") && text.contains("partitions: 4"),
            "{text}"
        );
        coordinator.cancel(QueryId(99));
        assert!(coordinator.trace().is_empty());
    }
}
