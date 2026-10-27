// SPDX-License-Identifier: Apache-2.0
//! The [`Executor`] seam and its Tokio implementation (E01, ADR 0006 §6,
//! spec 13 §4): pipeline drivers run as tasks on a multi-thread compute
//! runtime, object-store and network I/O on a small separate runtime, and
//! non-yielding work over ~1 ms on Tokio's blocking pool, gated by a
//! semaphore so at most a fixed number of such closures run at once.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use futures::future::BoxFuture;
use krishiv_memory::{Clock, MemoryError};
use tokio::runtime::{Handle, Runtime, RuntimeFlavor};
use tokio::sync::{Semaphore, oneshot};
use tokio::task::{JoinError, JoinHandle};

use crate::error::{Result, cancelled, internal};

/// Workload class of a query. P2a runs every class on one compute runtime;
/// the class is threaded through so workload management (per-class
/// runtimes or admission budgets) drops in behind [`Executor::spawn`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkloadClass {
    /// Short interactive queries.
    Interactive,
    /// Long batch queries.
    Batch,
    /// Maintenance work (compaction, statistics).
    Background,
    /// Queries issued by agents.
    Agent,
}

impl WorkloadClass {
    /// `interactive`, `batch`, `background` or `agent` (the `class` label).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Batch => "batch",
            Self::Background => "background",
            Self::Agent => "agent",
        }
    }
}

/// Where tasks run. Keeps a dedicated CPU pool possible later without
/// changing operators or the driver.
pub trait Executor: Send + Sync + fmt::Debug {
    /// Runs a pipeline driver (or another engine task) on the compute
    /// runtime.
    fn spawn(&self, class: WorkloadClass, fut: BoxFuture<'static, ()>) -> TaskHandle;
    /// Runs non-yielding CPU work over ~1 ms on the bounded blocking pool;
    /// never run blocking code elsewhere. The closure keeps its pool slot
    /// until it returns, even if the returned future is dropped, so long
    /// closures must poll their query's cancellation between chunks
    /// (≤ 10 ms each).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if the closure panicked.
    fn spawn_blocking(
        &self,
        f: Box<dyn FnOnce() + Send + 'static>,
    ) -> BoxFuture<'static, Result<()>>;
    /// The I/O runtime, for object-store and network futures.
    fn io_handle(&self) -> &Handle;
    /// Worker slots: the most drivers one pipeline runs at once.
    fn parallelism(&self) -> usize;
    /// The time source of the stall detector and the metrics (the paused
    /// Tokio clock in deterministic mode).
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// Tokio's clock as an arbitration [`Clock`] for the memory pool. A
/// deterministic run needs it (with the paused clock), so spill-wait and
/// abort deadlines replay too.
#[derive(Debug, Default, Clone, Copy)]
pub struct TokioClock;

impl Clock for TokioClock {
    fn now(&self) -> Instant {
        tokio::time::Instant::now().into_std()
    }

    fn sleep_until(
        &self,
        deadline: Instant,
    ) -> std::result::Result<BoxFuture<'static, ()>, MemoryError> {
        Ok(Box::pin(tokio::time::sleep_until(
            tokio::time::Instant::from_std(deadline),
        )))
    }
}

/// Runs `f` through [`Executor::spawn_blocking`] and returns its result
/// (operators use [`TaskContext::run_blocking`](crate::TaskContext::run_blocking)).
pub(crate) async fn run_blocking<T, F>(executor: &dyn Executor, f: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    let (tx, rx) = oneshot::channel();
    executor
        .spawn_blocking(Box::new(move || {
            drop(tx.send(f()));
            // What the closure freed lives in this pool thread's allocator
            // caches: under memory pressure, return it to the OS too.
            krishiv_memory::release_if_pressured_here();
        }))
        .await?;
    rx.await
        .map_err(|_| internal("a blocking task ended without a result".to_owned()))?
}

/// A spawned task. Awaiting it yields the task's outcome: a panic becomes
/// `INTERNAL_ERROR` for the task's query only (no `catch_unwind`; Tokio
/// isolates the panic), an abort becomes `OPERATION_CANCELED`. Dropping the
/// handle detaches the task.
#[derive(Debug)]
pub struct TaskHandle {
    join: JoinHandle<()>,
}

impl TaskHandle {
    /// Wraps a Tokio join handle.
    pub fn new(join: JoinHandle<()>) -> Self {
        Self { join }
    }

    /// Aborts the task: it is dropped at its next await, releasing what it
    /// holds.
    pub fn abort(&self) {
        self.join.abort();
    }
}

impl Future for TaskHandle {
    type Output = Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<()>> {
        Pin::new(&mut self.join)
            .poll(cx)
            .map(|r| r.map_err(join_error))
    }
}

fn join_error(e: JoinError) -> krishiv_common::error::KrishivError {
    if !e.is_panic() {
        return cancelled();
    }
    let payload = e.into_panic();
    let message = payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_default();
    internal(format!("a query task panicked: {message}"))
}

/// Sizes of the [`TokioExecutor`] runtimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutorConfig {
    /// Compute worker threads (default: available cores).
    pub compute_threads: usize,
    /// I/O runtime worker threads (default 2).
    pub io_threads: usize,
    /// Blocking closures that may run at once (default: compute threads).
    pub blocking_threads: usize,
    /// Most blocking threads of the I/O runtime, where a local file store
    /// reads and writes (`krishiv.io.blockingThreads`, default 8; see
    /// [`TokioExecutor::new`]).
    pub io_blocking_threads: usize,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        let cores = std::thread::available_parallelism().map_or(1, usize::from);
        Self {
            compute_threads: cores,
            io_threads: 2,
            blocking_threads: cores,
            io_blocking_threads: DEFAULT_IO_BLOCKING_THREADS,
        }
    }
}

/// The default of `krishiv.io.blockingThreads`.
const DEFAULT_IO_BLOCKING_THREADS: usize = 8;

impl ExecutorConfig {
    /// Takes the executor settings of `config`: `krishiv.io.blockingThreads`
    /// (validated as a positive integer when it was set).
    #[must_use]
    pub fn with_config(mut self, config: &krishiv_common::config::Config) -> Self {
        let blocking = config
            .get(&krishiv_common::config::IO_BLOCKING_THREADS)
            .and_then(krishiv_common::config::ConfigValue::as_int)
            .and_then(|n| usize::try_from(n).ok());
        self.io_blocking_threads = blocking.unwrap_or(DEFAULT_IO_BLOCKING_THREADS);
        self
    }
}

/// Owned runtimes, shut down without blocking when the executor drops (so
/// dropping it inside async code is fine).
struct Runtimes {
    compute: Option<Runtime>,
    io: Option<Runtime>,
}

impl Drop for Runtimes {
    fn drop(&mut self) {
        for rt in [self.compute.take(), self.io.take()].into_iter().flatten() {
            rt.shutdown_background();
        }
    }
}

/// The v1 [`Executor`]: Tokio for compute (E01), a small I/O runtime and a
/// semaphore-gated blocking pool.
pub struct TokioExecutor {
    compute: Handle,
    io: Handle,
    parallelism: usize,
    blocking: Arc<Semaphore>,
    /// Deterministic mode: blocking closures run as tasks on the caller's
    /// runtime and time is Tokio's clock.
    deterministic: bool,
    _owned: Option<Runtimes>,
}

impl fmt::Debug for TokioExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokioExecutor")
            .field("parallelism", &self.parallelism)
            .field("blocking_permits", &self.blocking.available_permits())
            .finish_non_exhaustive()
    }
}

impl TokioExecutor {
    /// Starts a compute and an I/O runtime.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if a runtime cannot be started.
    pub fn new(config: &ExecutorConfig) -> Result<Arc<Self>> {
        let compute_threads = config.compute_threads.max(1);
        let blocking_threads = config.blocking_threads.max(1);
        let build = |builder: &mut tokio::runtime::Builder, name: &str| {
            builder
                .thread_name(name)
                .enable_all()
                .build()
                .map_err(|e| internal(format!("cannot start the {name} runtime: {e}")))
        };
        let compute = build(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(compute_threads)
                .max_blocking_threads(blocking_threads),
            "krishiv-compute",
        )?;
        // The I/O runtime's blocking pool is bounded too
        // (`io_blocking_threads`). A local file store reads each range on a
        // blocking thread, so the buffer lives in that thread's allocator
        // heap; it is freed later on a compute thread, and the page it sat
        // in stays resident in the reading thread's heap until that thread
        // allocates again. Tokio's default pool grows to 512 threads (TPC-H
        // Q21 at SF10 under a 2 GiB limit ran up to ~200 threads and held
        // ~690 MiB more resident memory than the live heap). Requests beyond
        // the bound queue for a thread. Reads a scan issues outside the I/O
        // runtime (prefetch misses, footers) still run on the compute
        // runtime's blocking pool, bounded by `blocking_threads`.
        let io = build(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(config.io_threads.max(1))
                .max_blocking_threads(config.io_blocking_threads.max(1)),
            "krishiv-io",
        )?;
        Ok(Arc::new(Self {
            compute: compute.handle().clone(),
            io: io.handle().clone(),
            parallelism: compute_threads,
            blocking: Arc::new(Semaphore::new(blocking_threads)),
            deterministic: false,
            _owned: Some(Runtimes {
                compute: Some(compute),
                io: Some(io),
            }),
        }))
    }

    /// Deterministic test mode (spec 13 §4, ADR 0029): every task, blocking
    /// closures included, runs on the caller's current-thread runtime, and
    /// the stall detector and metrics read Tokio's clock. With the paused
    /// clock, a seeded [`ExecConfig`](crate::ExecConfig) and a memory pool
    /// built with [`TokioClock`], a run's interleavings, morsel order and
    /// spill/cancel timing replay exactly (a paused clock never advances
    /// inside a poll, so no slice counts as a stall). `slots` drivers per
    /// pipeline interleave at their yields.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` outside a current-thread Tokio runtime.
    pub fn deterministic(slots: usize) -> Result<Arc<Self>> {
        let handle = Handle::try_current()
            .ok()
            .filter(|h| h.runtime_flavor() == RuntimeFlavor::CurrentThread)
            .ok_or_else(|| {
                internal("deterministic mode needs a current-thread Tokio runtime".to_owned())
            })?;
        Ok(Arc::new(Self {
            compute: handle.clone(),
            io: handle,
            parallelism: slots.max(1),
            blocking: Arc::new(Semaphore::new(1)),
            deterministic: true,
            _owned: None,
        }))
    }
}

impl Executor for TokioExecutor {
    fn spawn(&self, _class: WorkloadClass, fut: BoxFuture<'static, ()>) -> TaskHandle {
        TaskHandle::new(self.compute.spawn(fut))
    }

    fn spawn_blocking(
        &self,
        f: Box<dyn FnOnce() + Send + 'static>,
    ) -> BoxFuture<'static, Result<()>> {
        let permits = Arc::clone(&self.blocking);
        let compute = self.compute.clone();
        if self.deterministic {
            // On the controlled runtime, scheduled like any task.
            return Box::pin(TaskHandle::new(compute.spawn(async move { f() })));
        }
        Box::pin(async move {
            // The semaphore is never closed, so `acquire` cannot fail.
            let permit = permits
                .acquire_owned()
                .await
                .map_err(|_| internal("the blocking pool is closed".to_owned()))?;
            let task = compute.spawn_blocking(move || {
                let _slot = permit;
                f();
            });
            TaskHandle::new(task).await
        })
    }

    fn io_handle(&self) -> &Handle {
        &self.io
    }

    fn parallelism(&self) -> usize {
        self.parallelism
    }

    fn now(&self) -> Instant {
        if self.deterministic {
            return tokio::time::Instant::now().into_std();
        }
        Instant::now()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use krishiv_common::config::Config;

    use super::*;

    /// The most blocking closures that ran at once on `executor`'s I/O
    /// runtime, out of `n` that each hold their thread for a while.
    fn most_at_once_on_io(executor: &TokioExecutor, n: usize) -> usize {
        let running = Arc::new(AtomicUsize::new(0));
        let most = Arc::new(AtomicUsize::new(0));
        let tasks: Vec<_> = (0..n)
            .map(|_| {
                let (running, most) = (Arc::clone(&running), Arc::clone(&most));
                executor.io_handle().spawn_blocking(move || {
                    let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                    most.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(50));
                    running.fetch_sub(1, Ordering::SeqCst);
                })
            })
            .collect();
        executor.io_handle().block_on(async {
            for t in tasks {
                t.await.unwrap();
            }
        });
        most.load(Ordering::SeqCst)
    }

    #[test]
    fn the_io_runtime_runs_at_most_its_blocking_threads_at_once() {
        let config = ExecutorConfig {
            compute_threads: 1,
            io_threads: 1,
            blocking_threads: 1,
            io_blocking_threads: 3,
        };
        let executor = TokioExecutor::new(&config).unwrap();
        let most = most_at_once_on_io(&executor, 12);
        assert!((1..=3).contains(&most), "{most} blocking closures at once");
    }

    #[test]
    fn the_blocking_thread_bound_comes_from_the_config() {
        assert_eq!(ExecutorConfig::default().io_blocking_threads, 8);
        let (set, _) = Config::from_pairs([("krishiv.io.blockingThreads", "2")]).unwrap();
        let config = ExecutorConfig::default().with_config(&set);
        assert_eq!(config.io_blocking_threads, 2);
        let executor = TokioExecutor::new(&config).unwrap();
        assert!(most_at_once_on_io(&executor, 8) <= 2);
        for bad in ["0", "-2", "many"] {
            assert!(
                Config::from_pairs([("krishiv.io.blockingThreads", bad)]).is_err(),
                "{bad}"
            );
        }
    }
}
