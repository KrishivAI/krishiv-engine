// SPDX-License-Identifier: Apache-2.0
//! The bounded pool that runs every kernel call (design note P3-T021 section 3.2).
//!
//! The kernel's API is synchronous and owns no runtime, while Krishiv's I/O is async. A kernel
//! call therefore runs on one of the pool's own OS threads, never on a runtime worker, and its
//! handlers bridge back to `IoContext` with `Handle::block_on`. The threads are the pool's, not
//! the runtime's blocking pool: the I/O a handler waits for (local files, local listings) runs
//! on that blocking pool, which `krishiv.io.blockingThreads` bounds, so kernel calls holding its
//! threads could wait forever for I/O queued behind them. At most [`LogPool::threads`] calls run at once,
//! one pool for the whole process (`krishiv.delta.logThreads`); a query waits for a slot in a
//! wait its cancellation ends, and the wait is measured ([`WAIT_METRIC`]) so slow log replays
//! holding every slot are visible. A statement's deadline ends the wait too: the session cancels
//! the query's token when the deadline passes.

use std::cell::Cell;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, mpsc};
use std::time::Instant;

use krishiv_common::config::{Config, ConfigValue, DELTA_LOG_THREADS};
use krishiv_common::error::{InternalError, KrishivError, OperationCanceled};
use tokio::sync::{Semaphore, oneshot};

/// The name of the slot-wait metric.
pub const WAIT_METRIC: &str = "delta.log_pool.wait_ms";

thread_local! {
    /// Set while a log-pool thread runs a kernel call.
    static ON_LOG_POOL: Cell<bool> = const { Cell::new(false) };
}

/// Whether the current thread is running a kernel call for the log pool.
#[must_use]
pub fn on_log_pool() -> bool {
    ON_LOG_POOL.with(Cell::get)
}

/// Marks the current thread as a log-pool thread until dropped.
struct PoolThread;

impl PoolThread {
    fn enter() -> Self {
        ON_LOG_POOL.with(|f| f.set(true));
        Self
    }
}

impl Drop for PoolThread {
    fn drop(&mut self) {
        ON_LOG_POOL.with(|f| f.set(false));
    }
}

/// The slot waits of a pool ([`WAIT_METRIC`]): how many, their total and their longest, in
/// milliseconds.
#[derive(Debug, Default)]
pub struct WaitStats {
    count: AtomicU64,
    total_ms: AtomicU64,
    max_ms: AtomicU64,
}

impl WaitStats {
    fn record(&self, ms: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_ms.fetch_add(ms, Ordering::Relaxed);
        self.max_ms.fetch_max(ms, Ordering::Relaxed);
    }

    /// Waits recorded (one per call that got a slot).
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Total milliseconds waited.
    #[must_use]
    pub fn total_ms(&self) -> u64 {
        self.total_ms.load(Ordering::Relaxed)
    }

    /// The longest wait, in milliseconds.
    #[must_use]
    pub fn max_ms(&self) -> u64 {
        self.max_ms.load(Ordering::Relaxed)
    }
}

/// A job the pool's threads run.
type Job = Box<dyn FnOnce() + Send + 'static>;

/// A bounded pool of OS threads for kernel calls.
#[derive(Debug)]
pub struct LogPool {
    slots: Arc<Semaphore>,
    threads: usize,
    jobs: mpsc::Sender<Job>,
    waits: WaitStats,
}

/// The loop of one pool thread: run jobs until the pool is dropped.
fn pool_thread(jobs: &Mutex<mpsc::Receiver<Job>>) {
    let _thread = PoolThread::enter();
    loop {
        let next = jobs.lock().unwrap_or_else(PoisonError::into_inner).recv();
        match next {
            Ok(job) => job(),
            Err(mpsc::RecvError) => return,
        }
    }
}

static PROCESS_POOL: OnceLock<LogPool> = OnceLock::new();

impl LogPool {
    /// A pool of `threads` slots and OS threads (at least one). The threads end when the pool
    /// is dropped. A thread that cannot be started leaves fewer threads than slots: calls then
    /// queue for a thread (none can deadlock, the threads wait on nothing but their jobs), and
    /// with no thread at all every call fails with `INTERNAL_ERROR`.
    #[must_use]
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let (jobs, queue) = mpsc::channel::<Job>();
        let queue = Arc::new(Mutex::new(queue));
        for i in 0..threads {
            let queue = Arc::clone(&queue);
            // A failed spawn drops its clone of the queue; see the doc above.
            let _spawned = std::thread::Builder::new()
                .name(format!("krishiv-delta-log-{i}"))
                .spawn(move || pool_thread(&queue));
        }
        Self {
            slots: Arc::new(Semaphore::new(threads)),
            threads,
            jobs,
            waits: WaitStats::default(),
        }
    }

    /// The process's pool, sized by `krishiv.delta.logThreads` of the first `config` that asks.
    #[must_use]
    pub fn process(config: &Config) -> &'static Self {
        PROCESS_POOL.get_or_init(|| {
            let threads = config
                .get(&DELTA_LOG_THREADS)
                .and_then(ConfigValue::as_int)
                .and_then(|n| usize::try_from(n).ok())
                .unwrap_or(4);
            Self::new(threads)
        })
    }

    /// How many kernel calls may run at once.
    #[must_use]
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Slots free right now.
    #[must_use]
    pub fn free(&self) -> usize {
        self.slots.available_permits()
    }

    /// The slot waits so far ([`WAIT_METRIC`]).
    #[must_use]
    pub fn waits(&self) -> &WaitStats {
        &self.waits
    }

    /// Runs `f` on a pool thread once a slot is free. The wait for the slot ends early with
    /// `OPERATION_CANCELED` when `cancelled` completes (a token's or a query's cancellation). A call that has started is always awaited to
    /// its real outcome: it may have put a commit, and a commit that landed must not be reported
    /// as cancelled. It still ends soon after a cancel, because every handler call checks the
    /// token first and reads race it.
    ///
    /// # Errors
    /// `OPERATION_CANCELED`; `INTERNAL_ERROR` if `f` panics or the pool has no thread.
    pub async fn run<T, F>(
        &self,
        cancelled: impl Future<Output = ()>,
        f: F,
    ) -> Result<T, KrishivError>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let started = Instant::now();
        let permit = tokio::select! {
            biased;
            () = cancelled => return Err(canceled()),
            permit = Arc::clone(&self.slots).acquire_owned() => permit.map_err(|_| canceled())?,
        };
        self.waits
            .record(u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX));
        let (done, result) = oneshot::channel();
        let job: Job = Box::new(move || {
            let out = std::panic::catch_unwind(AssertUnwindSafe(f));
            // The slot is free before the caller hears of the result: a caller that ran a call
            // to its end finds the slot back (the pool's tests count on it).
            drop(permit);
            // The caller's future may have been dropped; the result is then dropped here.
            let _unread = done.send(out);
        });
        self.jobs
            .send(job)
            .map_err(|_| internal("the Delta log pool has no thread"))?;
        match result.await {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(_)) => Err(internal("a Delta kernel call panicked")),
            Err(_) => Err(internal("a Delta kernel call ended without a result")),
        }
    }
}

fn internal(message: &str) -> KrishivError {
    InternalError {
        message: message.to_owned(),
    }
    .into()
}

fn canceled() -> KrishivError {
    OperationCanceled {}.into()
}
