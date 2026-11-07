// SPDX-License-Identifier: Apache-2.0
//! A simulated worker host (design P2-T176 §2.6): inside its host closure
//! it builds its own deterministic executor, memory pool, exchange and
//! worker context (tables opened by [`HostSetup`] on the host's runtime),
//! then serves [`Request`]s on its port: `Run` decodes a fragment's
//! `krishiv.plan.v1` bytes and its task descriptor, instantiates and runs
//! the map task; `Fetch` streams one reduce partition of a map output it
//! committed.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use futures::StreamExt as _;
use futures::future::BoxFuture;
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_exec::exchange::{Exchange, ExchangeFrame};
use krishiv_exec::physical::{ExchangeId, WorkerContext, instantiate_task};
use krishiv_exec::values::{LocalValues, ValueService};
use krishiv_exec::{
    BroadcastTables, ExecConfig, Executor, QueryContext, QueryId, QueryRunner, TokioClock,
    TokioExecutor, WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
use krishiv_scheduler::produced;
use krishiv_shuffle::frame::batch_body;
use turmoil::net::{TcpListener, TcpStream};

use crate::exchange::{Fault, HostExchange, Workers};
use crate::task::{decode_task, encode_partials};
use crate::wire::{Reply, Request};

type Result<T> = std::result::Result<T, KrishivError>;

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// What a host builds for itself: called inside the host's closure, on its
/// runtime, so nothing it returns is shared with another host unless the
/// implementation shares it (only the object store should be).
pub trait HostSetup: Send + Sync + 'static {
    /// The host's worker context: its own tables, I/O context and
    /// configuration, over the cluster's shared object store. The
    /// simulation adds the exchange.
    fn worker(&self) -> BoxFuture<'static, Result<WorkerContext>>;
}

/// The execution seed of a task: from the simulation's seed, its job,
/// stage and partition, never its attempt, so a rerun replays the same
/// morsel order and table seeds (I10).
#[must_use]
pub fn task_seed(seed: u64, job: u64, stage: u32, partition: u32) -> u64 {
    // SplitMix64 over the parts: distinct tasks get unrelated seeds.
    let mut x = seed;
    for part in [job, u64::from(stage), u64::from(partition)] {
        x = x.wrapping_add(part).wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
    }
    x
}

/// A failed request as its wire text: the error's class and message.
pub(crate) fn failure(e: &KrishivError) -> String {
    format!("{}: {}", e.error_class().name(), e.message())
}

/// The state of one worker host.
struct Host {
    /// The simulation's seed: each task's execution seed derives from it.
    seed: u64,
    index: usize,
    workers: Workers,
    fault: Option<Fault>,
    executor: Arc<dyn Executor>,
    pool: Arc<dyn MemoryPool>,
    runner: QueryRunner,
    /// Each running job's exchange and broadcast builds, until the
    /// coordinator releases it.
    jobs: Mutex<BTreeMap<u64, Job>>,
    worker: WorkerContext,
    next_query: AtomicU64,
}

/// What a host keeps of one job: its map outputs and its broadcast tables
/// (built once on this host and shared by the job's tasks here, design
/// P2-T177 unit 3), both dropped at `Release`.
struct Job {
    exchange: Arc<HostExchange>,
    broadcasts: Arc<BroadcastTables>,
}

/// Runs worker `index` of `workers` until the simulation ends: the body of
/// its turmoil host.
///
/// # Errors
/// The host's setup failing, or its port not binding.
pub async fn serve(
    seed: u64,
    index: usize,
    workers: Workers,
    fault: Option<Fault>,
    setup: Arc<dyn HostSetup>,
    slots: usize,
) -> Result<()> {
    let executor: Arc<dyn Executor> = TokioExecutor::deterministic(slots)?;
    let pool: Arc<dyn MemoryPool> =
        TreePool::with_clock(&MemoryConfig::default(), Arc::new(TokioClock));
    let worker = setup.worker().await?;
    let host = Arc::new(Host {
        seed,
        index,
        workers: workers.clone(),
        fault,
        runner: QueryRunner::new(Arc::clone(&executor)),
        executor,
        pool,
        jobs: Mutex::default(),
        worker,
        next_query: AtomicU64::new(0),
    });
    let listener = TcpListener::bind(("0.0.0.0", workers.port))
        .await
        .map_err(|e| internal(format!("worker {index}: bind: {e}")))?;
    loop {
        let (stream, _) = listener
            .accept()
            .await
            .map_err(|e| internal(format!("worker {index}: accept: {e}")))?;
        let host = Arc::clone(&host);
        tokio::spawn(async move {
            // A connection that fails mid-reply is the requester's error.
            let _ = host.answer(stream).await;
        });
    }
}

impl Host {
    /// `job`'s exchange on this host, if it has one (a fetch never makes
    /// one: a fetch after the job's release fails).
    fn existing(&self, job: u64) -> Option<Arc<HostExchange>> {
        self.jobs
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&job)
            .map(|j| Arc::clone(&j.exchange))
    }

    /// `job`'s exchange and broadcast tables on this host, made on first
    /// use.
    fn job(&self, job: u64) -> (Arc<HostExchange>, Arc<BroadcastTables>) {
        let mut jobs = self.jobs.lock().unwrap_or_else(PoisonError::into_inner);
        let j = jobs.entry(job).or_insert_with(|| Job {
            exchange: Arc::new(HostExchange::worker(
                job,
                self.index,
                self.workers.clone(),
                self.fault,
            )),
            broadcasts: Arc::new(BroadcastTables::new(
                Arc::clone(&self.executor),
                self.pool.child(&format!("job-{job}-broadcasts"), None),
            )),
        });
        (Arc::clone(&j.exchange), Arc::clone(&j.broadcasts))
    }

    async fn answer(&self, mut stream: TcpStream) -> Result<()> {
        match Request::read(&mut stream).await? {
            Request::Release { job } => {
                self.jobs
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .remove(&job);
                Reply::Done(Vec::new()).write(&mut stream).await
            }
            Request::Run {
                job,
                fragment,
                task,
            } => {
                let reply = match self.run(job, &fragment, &task).await {
                    Ok(partials) => Reply::Done(partials),
                    Err(e) => Reply::Failed(failure(&e)),
                };
                reply.write(&mut stream).await
            }
            Request::Fetch {
                job,
                exchange,
                stage_attempt,
                map,
                reduce,
            } => {
                let outcome = match self.existing(job) {
                    Some(shuffle) => {
                        self.fetch(
                            &mut stream,
                            (shuffle, ExchangeId(exchange)),
                            stage_attempt,
                            map,
                            reduce,
                        )
                        .await
                    }
                    None => Err(internal(format!("job {job} holds no map output here"))),
                };
                if let Err(e) = outcome {
                    Reply::Failed(failure(&e)).write(&mut stream).await?;
                }
                Ok(())
            }
        }
    }

    /// Runs one map task to its end; its output is committed by then. Its
    /// exchange's filter partials, encoded.
    async fn run(&self, job: u64, fragment: &[u8], task: &[u8]) -> Result<Vec<u8>> {
        let plan = krishiv_plan_codec::decode(fragment)
            .map_err(|e| internal(format!("a fragment: {e}")))?;
        let (task, settings) = decode_task(task)?;
        let name = format!(
            "stage-{}-task-{}-attempt-{}",
            task.stage, task.partition, task.attempt
        );
        let (exchange, broadcasts) = self.job(job);
        let worker = self
            .worker
            .clone()
            .with_exchange(exchange as Arc<dyn Exchange>)
            .with_broadcasts(broadcasts);
        // The task's own values: its exchange's filter partials are read
        // from here once it ended.
        let values = Arc::new(LocalValues::new());
        let worker = worker.with_values(Arc::clone(&values) as Arc<dyn ValueService>);
        let node = instantiate_task(&plan, &worker, &task)?;
        let query = QueryContext::new(
            QueryId(self.next_query.fetch_add(1, Ordering::Relaxed)),
            WorkloadClass::Interactive,
            ExecConfig {
                // The query's seed when it has one, else the simulation's:
                // either way per task, never per attempt.
                seed: Some(task_seed(
                    settings.seed.unwrap_or(self.seed),
                    job,
                    task.stage,
                    task.partition,
                )),
                ..settings
            },
            self.pool.child(&name, None),
        );
        let mut stream = self.runner.execute(node, Arc::clone(&query)).await?;
        let mut outcome = Ok(());
        while let Some(batch) = stream.next().await {
            match batch {
                Ok(b) if b.num_rows() == 0 => {}
                Ok(_) => outcome = Err(internal(format!("map task {name} returned rows"))),
                Err(e) => outcome = Err(e),
            }
            if outcome.is_err() {
                query.cancel();
                break;
            }
        }
        drop(stream);
        // Its drivers have ended and its memory is back before it answers.
        query.finished().await;
        outcome?;
        query.error().map_or(Ok(()), Err)?;
        Ok(encode_partials(&produced(&plan, values.as_ref())))
    }

    async fn fetch(
        &self,
        stream: &mut TcpStream,
        (shuffle, exchange): (Arc<HostExchange>, ExchangeId),
        stage_attempt: u32,
        map: u32,
        reduce: u32,
    ) -> Result<()> {
        let mut reader = shuffle.reader(exchange, stage_attempt, map, reduce)?;
        loop {
            match reader.next().await? {
                ExchangeFrame::Data(batch) => {
                    let body =
                        batch_body(&batch).map_err(|e| internal(format!("a frame body: {e}")))?;
                    Reply::Frame(body).write(stream).await?;
                }
                ExchangeFrame::End => return Reply::End.write(stream).await,
            }
        }
    }
}

impl std::fmt::Debug for Host {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Host")
            .field("executor", &self.executor)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use krishiv_exec::planner::MemoryTables;
    use turmoil::net::TcpStream;

    use super::*;
    use crate::testing;

    /// Worker hosts with no tables at all.
    struct Empty;

    impl HostSetup for Empty {
        fn worker(&self) -> BoxFuture<'static, Result<WorkerContext>> {
            Box::pin(async { Ok(WorkerContext::new(Arc::new(MemoryTables::new()))) })
        }
    }

    async fn ask(request: Request) -> Reply {
        let mut stream = TcpStream::connect(("worker-0", testing::PORT))
            .await
            .unwrap();
        request.write(&mut stream).await.unwrap();
        Reply::read(&mut stream).await.unwrap()
    }

    #[test]
    fn a_worker_refuses_what_it_cannot_serve_and_prints_its_shape() {
        let mut sim = testing::sim();
        sim.host("worker-0", || async {
            let workers = testing::workers(&["worker-0"]);
            serve(1, 0, workers, None, Arc::new(Empty), 1)
                .await
                .map_err(Into::into)
        });
        sim.client("coordinator", async {
            // A fetch of a job this host never ran.
            let reply = ask(Request::Fetch {
                job: 99,
                exchange: 1,
                stage_attempt: 0,
                map: 0,
                reduce: 0,
            })
            .await;
            assert!(
                matches!(&reply, Reply::Failed(m) if m.contains("job 99 holds no map output here")),
                "{reply:?}"
            );
            // A fragment that is no plan.
            let reply = ask(Request::Run {
                job: 1,
                fragment: vec![1, 2, 3],
                task: Vec::new(),
            })
            .await;
            assert!(
                matches!(&reply, Reply::Failed(m) if m.contains("a fragment")),
                "{reply:?}"
            );
            // Releasing a job it does not hold is fine.
            assert_eq!(
                ask(Request::Release { job: 99 }).await,
                Reply::Done(Vec::new())
            );
            Ok::<(), Box<dyn std::error::Error>>(())
        });
        sim.run().unwrap();
    }

    #[tokio::test]
    async fn a_host_prints_its_executor() {
        let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
        let host = Host {
            seed: 1,
            index: 0,
            workers: testing::workers(&["worker-0"]),
            fault: None,
            runner: QueryRunner::new(Arc::clone(&executor)),
            executor,
            pool: TreePool::with_clock(&MemoryConfig::default(), Arc::new(TokioClock)),
            jobs: Mutex::default(),
            worker: Empty.worker().await.unwrap(),
            next_query: AtomicU64::new(0),
        };
        assert!(format!("{host:?}").starts_with("Host"));
        assert!(host.existing(5).is_none());
        // A job's exchange and broadcast tables are made once and reused.
        let (first, second) = (host.job(5), host.job(5));
        assert!(Arc::ptr_eq(&first.0, &second.0));
        assert!(Arc::ptr_eq(&first.1, &second.1));
        assert!(host.existing(5).is_some());
    }
}
