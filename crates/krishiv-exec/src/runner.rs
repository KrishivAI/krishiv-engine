// SPDX-License-Identifier: Apache-2.0
//! Running a query in process: [`Scheduler`] and its embedded
//! implementation, [`QueryRunner`].
//!
//! `submit` splits the plan into pipelines and spawns a coordinator task,
//! which starts each pipeline once its dependencies have finalized:
//! `min(worker slots, source parallelism)` drivers per pipeline (one if it
//! must keep order). The coordinator awaits every driver; the first error
//! (a driver's error, a panic, a memory abort, cancellation) wins and
//! cancels the rest, which are also aborted. The result stream ends only
//! after every task has ended and every pipeline's state is dropped, so the
//! query's memory pool is empty by then.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::{BoxStream, FuturesUnordered};
use krishiv_memory::SpillOwner;
use krishiv_types::batch::Batch;

use crate::channel::{Channel, ResultSink, ResultStream};
use crate::context::{QueryContext, QueryId, TaskContext};
use crate::driver::{PipelineRun, spawn_driver};
use crate::error::{Result, cancelled, internal};
use crate::executor::{Executor, TaskHandle};
use crate::metrics::OperatorEntry;
use crate::morsel::Countdown;
use crate::pipeline::{ExecNode, Pipeline, PipelineBuilder};
use crate::sync::{Mutex, lock};

/// Plans in, running queries out (spec 13 §4). Embedded and single-node:
/// [`QueryRunner`]; distributed: the coordinator (P5).
pub trait Scheduler: Send + Sync {
    /// Starts `plan` and returns its result stream. Dropping the stream
    /// cancels the query.
    ///
    /// # Errors
    /// Planning errors; execution errors arrive on the stream.
    fn submit(
        &self,
        plan: Arc<dyn ExecNode>,
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
        query.merge_metrics(|m| {
            for (id, name) in names {
                m.operators.insert(
                    id,
                    OperatorEntry {
                        name,
                        ..OperatorEntry::default()
                    },
                );
            }
        });
        {
            let mut running = lock(&self.running);
            running.retain(|_, q| q.strong_count() > 0);
            running.insert(query.id(), Arc::downgrade(&query));
        }
        let finish = Finish {
            query: Arc::clone(&query),
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

impl Scheduler for QueryRunner {
    fn submit(
        &self,
        plan: Arc<dyn ExecNode>,
        query: Arc<QueryContext>,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<Batch>>>> {
        let started = self.start(plan, query);
        Box::pin(async move { started })
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
        self.query.mark_finished();
        lock(&self.running).remove(&self.query.id());
        self.channel.close(self.query.error());
    }
}

/// Runs the pipelines in dependency waves; every pipeline (and its global
/// state) is dropped when this returns.
async fn run_pipelines(
    executor: &Arc<dyn Executor>,
    query: &Arc<QueryContext>,
    pipelines: Vec<Pipeline>,
) -> Result<()> {
    let deps: Vec<Vec<usize>> = pipelines.iter().map(|p| p.deps.clone()).collect();
    let mut todo: Vec<Option<Pipeline>> = pipelines.into_iter().map(Some).collect();
    let mut done = vec![false; todo.len()];
    while done.iter().any(|d| !d) {
        let ready: Vec<usize> = (0..todo.len())
            .filter(|&i| !done[i] && deps[i].iter().all(|&d| done[d]))
            .collect();
        if ready.is_empty() {
            // Unreachable: dependencies form a tree.
            return Err(internal("pipeline dependencies form a cycle".to_owned()));
        }
        let mut handles = FuturesUnordered::new();
        for i in ready {
            done[i] = true;
            if let Some(p) = todo[i].take() {
                match start_pipeline(executor, query, p) {
                    Ok(hs) => handles.extend(hs),
                    Err(e) => {
                        query.fail(e);
                        break;
                    }
                }
            }
        }
        await_drivers(query, handles).await;
        query.check_cancelled()?;
    }
    Ok(())
}

fn start_pipeline(
    executor: &Arc<dyn Executor>,
    query: &Arc<QueryContext>,
    pipeline: Pipeline,
) -> Result<Vec<TaskHandle>> {
    let ctx = |(id, name)| {
        TaskContext::new(
            Arc::clone(query),
            Arc::clone(executor),
            SpillOwner::new(),
            id,
            name,
        )
    };
    let (source_id, source) = &pipeline.source;
    let (sink_id, sink) = &pipeline.sink;
    let source_global = source.global_state(&ctx((*source_id, source.name())))?;
    let sink_global = sink.global_state(&ctx((*sink_id, sink.name())))?;
    let drivers = if pipeline.preserve_order {
        1
    } else {
        executor
            .parallelism()
            .min(source.max_parallelism(&source_global))
            .max(1)
    };
    let run = Arc::new(PipelineRun {
        pipeline,
        source_global,
        sink_global,
        countdown: Countdown::new(drivers),
        query: Arc::clone(query),
        executor: Arc::clone(executor),
    });
    Ok((0..drivers)
        .map(|_| spawn_driver(Arc::clone(&run)))
        .collect())
}

/// Awaits every driver. A panic fails the query (`INTERNAL_ERROR`); once
/// the query is cancelled or failed, the remaining drivers are aborted and
/// still awaited, so they have released everything when this returns.
async fn await_drivers(query: &QueryContext, mut handles: FuturesUnordered<TaskHandle>) {
    let mut aborted = false;
    loop {
        tokio::select! {
            biased;
            next = handles.next() => match next {
                None => return,
                Some(Err(e)) => query.fail(e),
                Some(Ok(())) => {}
            },
            () = query.cancelled(), if !aborted => {
                aborted = true;
                for h in &handles {
                    h.abort();
                }
            }
        }
    }
}
