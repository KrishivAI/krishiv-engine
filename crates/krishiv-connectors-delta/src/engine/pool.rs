// SPDX-License-Identifier: Apache-2.0
//! The bounded pool that runs every kernel call (design note P3-T021 section 3.2).
//!
//! The kernel's API is synchronous and owns no runtime, while Krishiv's I/O is async. A kernel
//! call therefore runs on a blocking thread, never on a runtime worker, and its handlers bridge
//! back to `IoContext` with `Handle::block_on`. At most [`LogPool::threads`] calls run at once,
//! one pool for the whole process (`krishiv.delta.logThreads`); a query waits for a slot in a
//! wait its cancellation ends, and the wait is measured ([`WAIT_METRIC`]) so slow log replays
//! holding every slot are visible. A statement's deadline ends the wait too: the session cancels
//! the query's token when the deadline passes.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use krishiv_common::config::{Config, ConfigValue, DELTA_LOG_THREADS};
use krishiv_common::error::{InternalError, KrishivError, OperationCanceled};
use krishiv_exec::CancellationToken;
use tokio::runtime::Handle;
use tokio::sync::Semaphore;

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

/// A bounded pool of blocking threads for kernel calls.
#[derive(Debug)]
pub struct LogPool {
    slots: Arc<Semaphore>,
    threads: usize,
    waits: WaitStats,
}

static PROCESS_POOL: OnceLock<LogPool> = OnceLock::new();

impl LogPool {
    /// A pool of `threads` slots (at least one).
    #[must_use]
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        Self {
            slots: Arc::new(Semaphore::new(threads)),
            threads,
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

    /// Runs `f` on a blocking thread of `handle` once a slot is free. The wait for the slot and
    /// the call itself end early with `OPERATION_CANCELED` when `cancel` fires; a call already
    /// running keeps its slot until the kernel returns (every handler call checks the token
    /// first, so it returns soon).
    ///
    /// # Errors
    /// `OPERATION_CANCELED`; `INTERNAL_ERROR` if `f` panics.
    pub async fn run<T, F>(
        &self,
        handle: &Handle,
        cancel: &CancellationToken,
        f: F,
    ) -> Result<T, KrishivError>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let started = Instant::now();
        let permit = tokio::select! {
            biased;
            () = cancel.cancelled() => return Err(OperationCanceled {}.into()),
            permit = Arc::clone(&self.slots).acquire_owned() => permit.map_err(|_| canceled())?,
        };
        self.waits
            .record(u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX));
        let task = handle.spawn_blocking(move || {
            let _slot = permit;
            let _thread = PoolThread::enter();
            f()
        });
        tokio::select! {
            biased;
            () = cancel.cancelled() => Err(OperationCanceled {}.into()),
            joined = task => joined.map_err(|e| {
                InternalError { message: format!("a Delta kernel call failed: {e}") }.into()
            }),
        }
    }
}

fn canceled() -> KrishivError {
    OperationCanceled {}.into()
}
