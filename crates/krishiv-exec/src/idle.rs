// SPDX-License-Identifier: Apache-2.0
//! Returning freed memory when the process goes idle (design
//! `0.4-memory-query-end-release`, study 56 F2).
//!
//! The allocator keeps memory a finished query freed in the heap of every
//! thread that freed it. A running query holds a [`QueryRun`]; when the last
//! one ends, the executor is told ([`Executor::queries_idle`]), and a
//! [`TokioExecutor`](crate::TokioExecutor) wakes its *releaser* thread. The
//! releaser waits [`IDLE_DELAY`]; if no query started meanwhile, it starts an
//! idle epoch ([`krishiv_memory::note_idle`]) and runs one closure on every
//! worker and every idle pool thread of both runtimes, each calling
//! [`krishiv_memory::release_if_due_here`] on its own thread. Workers also
//! check when they park. Nothing runs while a query runs: under sustained
//! load, memory under a limit is the pressure epoch's job.
//!
//! The releaser is a thread of its own, outside both runtimes, so its spawns
//! go to each runtime's shared queue (none lands in a busy worker's local
//! slot), and a request is a counter under a mutex, so none is lost.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, PoisonError, Weak};
use std::time::{Duration, Instant};

use tokio::runtime::Handle;

use crate::executor::Executor;

/// How long the process must stay without a running query before the
/// releaser starts a round: statements closer together than this (a
/// script, a burst of metadata queries) never trigger one.
pub const IDLE_DELAY: Duration = Duration::from_millis(10);

/// The longest a round's closure waits for its peers to land on threads of
/// their own (it releases first, so the wait only spreads the closures).
const PEER_WAIT: Duration = Duration::from_millis(5);

/// The longest step of a closure's wait: a query start ends it within this.
const WAIT_STEP: Duration = Duration::from_millis(1);

/// The longest the releaser waits for the closures of one pool to report;
/// one that has not run by then (its pool is busy with other work) runs
/// later and finds nothing due.
const POOL_WAIT: Duration = Duration::from_millis(100);

/// Queries running in this process.
static RUNNING: AtomicUsize = AtomicUsize::new(0);
/// Queries started in this process: a round started before a query began
/// sees it moved and stops.
static STARTS: AtomicU64 = AtomicU64::new(0);

/// A running query, from admission until its tasks have ended. Dropping the
/// last one in the process tells its executor the process is idle.
#[must_use = "the query counts as running only while the run is held"]
#[derive(Debug)]
pub struct QueryRun {
    executor: Weak<dyn Executor>,
}

/// Marks a query as running on `executor` until the returned run drops.
pub fn query_run(executor: &Arc<dyn Executor>) -> QueryRun {
    STARTS.fetch_add(1, Ordering::AcqRel);
    RUNNING.fetch_add(1, Ordering::AcqRel);
    QueryRun {
        executor: Arc::downgrade(executor),
    }
}

/// Queries running in this process (holders of a [`QueryRun`]).
#[must_use]
pub fn queries_running() -> usize {
    RUNNING.load(Ordering::Acquire)
}

impl Drop for QueryRun {
    fn drop(&mut self) {
        if RUNNING.fetch_sub(1, Ordering::AcqRel) == 1
            && let Some(executor) = self.executor.upgrade()
        {
            executor.queries_idle();
        }
    }
}

/// No query running, and none started since `starts` was read.
fn still_idle(starts: u64) -> bool {
    RUNNING.load(Ordering::Acquire) == 0 && STARTS.load(Ordering::Acquire) == starts
}

/// Threads a runtime started and stopped (`on_thread_start`/`on_thread_stop`
/// fire for its workers and its blocking threads alike).
#[derive(Debug, Default)]
pub(crate) struct ThreadCount {
    started: AtomicUsize,
    stopped: AtomicUsize,
}

impl ThreadCount {
    pub(crate) fn start(&self) {
        self.started.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn stop(&self) {
        self.stopped.fetch_add(1, Ordering::Relaxed);
    }

    /// Threads alive now.
    fn alive(&self) -> usize {
        self.started
            .load(Ordering::Relaxed)
            .saturating_sub(self.stopped.load(Ordering::Relaxed))
    }
}

/// One runtime as the releaser sees it.
#[derive(Debug)]
pub(crate) struct Pool {
    pub(crate) handle: Handle,
    pub(crate) workers: usize,
    /// The most blocking threads the runtime runs.
    pub(crate) blocking_cap: usize,
    pub(crate) threads: Arc<ThreadCount>,
}

impl Pool {
    /// An upper bound of the blocking threads that exist now: tokio starts
    /// workers through its blocking spawner, so they count in `threads`.
    fn blocking_alive(&self) -> usize {
        self.threads
            .alive()
            .saturating_sub(self.workers)
            .min(self.blocking_cap)
    }
}

#[derive(Debug, Default)]
struct State {
    requested: u64,
    shutdown: bool,
}

#[derive(Debug, Default)]
struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    /// Rounds that ran (after the debounce).
    rounds: AtomicU64,
}

/// The releaser thread of one executor; stops when dropped.
#[derive(Debug)]
pub(crate) struct Releaser {
    shared: Arc<Shared>,
}

impl Releaser {
    /// Starts the releaser over `pools` (the compute runtime first).
    ///
    /// # Errors
    /// The OS refused the thread.
    pub(crate) fn start(pools: Vec<Pool>) -> std::io::Result<Self> {
        let shared = Arc::new(Shared::default());
        let theirs = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("krishiv-release".to_owned())
            .spawn(move || run(&theirs, &pools))?;
        Ok(Self { shared })
    }

    /// The process went idle: run a round once it stays so.
    pub(crate) fn request(&self) {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        state.requested = state.requested.wrapping_add(1);
        self.shared.wake.notify_all();
    }

    /// Rounds that ran so far.
    pub(crate) fn rounds(&self) -> u64 {
        self.shared.rounds.load(Ordering::Acquire)
    }
}

impl Drop for Releaser {
    fn drop(&mut self) {
        self.shared
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .shutdown = true;
        self.shared.wake.notify_all();
    }
}

/// The releaser's loop: one round per request that the debounce confirms.
fn run(shared: &Shared, pools: &[Pool]) {
    let mut seen = 0;
    loop {
        let state = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
        let state = shared
            .wake
            .wait_while(state, |s| s.requested == seen && !s.shutdown)
            .unwrap_or_else(PoisonError::into_inner);
        if state.shutdown {
            return;
        }
        seen = state.requested;
        drop(state);
        let starts = STARTS.load(Ordering::Acquire);
        // The debounce: a request arriving meanwhile is seen on the next turn.
        let state = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
        let (state, _) = shared
            .wake
            .wait_timeout_while(state, IDLE_DELAY, |s| !s.shutdown)
            .unwrap_or_else(PoisonError::into_inner);
        if state.shutdown {
            return;
        }
        drop(state);
        if !still_idle(starts) || !krishiv_memory::release_hook_installed() {
            continue;
        }
        krishiv_memory::note_idle();
        shared.rounds.fetch_add(1, Ordering::AcqRel);
        round(pools, starts);
    }
}

/// Every worker, then every idle blocking thread, of each pool releases.
fn round(pools: &[Pool], starts: u64) {
    for pool in pools {
        for (n, blocking) in [(pool.workers, false), (pool.blocking_alive(), true)] {
            if !still_idle(starts) {
                return;
            }
            rendezvous(&pool.handle, n, blocking, starts);
        }
    }
}

/// Runs `n` closures on `handle` (as tasks, or on its blocking pool), each
/// releasing on its thread and then waiting briefly for its peers so that
/// each lands on a thread of its own. Returns how many reported in time.
fn rendezvous(handle: &Handle, n: usize, blocking: bool, starts: u64) -> usize {
    if n == 0 {
        return 0;
    }
    let gate = Arc::new((Mutex::new(0_usize), Condvar::new()));
    let (tx, rx) = mpsc::channel();
    for _ in 0..n {
        let gate = Arc::clone(&gate);
        let tx = tx.clone();
        let f = move || {
            krishiv_memory::release_if_due_here();
            let (count, arrived) = &*gate;
            let mut c = count.lock().unwrap_or_else(PoisonError::into_inner);
            *c += 1;
            arrived.notify_all();
            let deadline = Instant::now() + PEER_WAIT;
            while *c < n && still_idle(starts) {
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                c = arrived
                    .wait_timeout(c, (deadline - now).min(WAIT_STEP))
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
            drop(c);
            tx.send(()).ok();
        };
        // On a runtime that shut down, the handle comes back cancelled.
        if blocking {
            drop(handle.spawn_blocking(f));
        } else {
            drop(handle.spawn(async move { f() }));
        }
    }
    drop(tx);
    let deadline = Instant::now() + POOL_WAIT;
    let mut reported = 0;
    while reported < n {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() || rx.recv_timeout(left).is_err() {
            break;
        }
        reported += 1;
    }
    reported
}
