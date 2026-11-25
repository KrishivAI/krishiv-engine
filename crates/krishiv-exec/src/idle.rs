// SPDX-License-Identifier: Apache-2.0
//! Returning freed memory when the process goes idle (design
//! `0.4-memory-query-end-release`, study 56 F2).
//!
//! The allocator keeps memory a finished query freed in the heap of every
//! thread that freed it. A running query holds a [`QueryRun`]; when the last
//! one ends, the executor is told ([`Executor::queries_idle`]), and a
//! [`TokioExecutor`](crate::TokioExecutor) wakes its *releaser* thread. The
//! releaser waits [`IDLE_DELAY`] (1 s); if no query started meanwhile, it starts an
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
/// script, a BI tool's burst, a client's stream of small queries) never
/// trigger one. A release costs the next query the page faults of
/// rebuilding its threads' heaps (TPC-DS SF10 small queries: 3-13 ms after a
/// release, measured with 15 ms and 50 ms gaps), so it is worth paying only
/// after real idleness.
pub const IDLE_DELAY: Duration = Duration::from_secs(1);

/// The longest a round's closure waits for its peers to land on threads of
/// their own (it releases first, so the wait only spreads the closures).
const PEER_WAIT: Duration = Duration::from_millis(5);

/// The releaser's waits: [`IDLE_DELAY`] and the peer wait in production;
/// tests use longer ones so that a loaded machine cannot fail them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Waits {
    pub(crate) debounce: Duration,
    pub(crate) peer: Duration,
}

impl Default for Waits {
    fn default() -> Self {
        Self {
            debounce: IDLE_DELAY,
            peer: PEER_WAIT,
        }
    }
}

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
    /// Closures whose peer wait ended on its deadline: their pool had fewer
    /// idle threads than closures (one busy with other work), so a thread
    /// may have been reached twice and another not at all.
    short: Arc<AtomicU64>,
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
    pub(crate) fn start(pools: Vec<Pool>, waits: Waits) -> std::io::Result<Self> {
        let shared = Arc::new(Shared::default());
        let theirs = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("krishiv-release".to_owned())
            .spawn(move || run(&theirs, &pools, waits))?;
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

    /// Round closures whose peer wait ran out (see `Shared::short`).
    pub(crate) fn short_waits(&self) -> u64 {
        self.shared.short.load(Ordering::Acquire)
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
fn run(shared: &Shared, pools: &[Pool], waits: Waits) {
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
            .wait_timeout_while(state, waits.debounce, |s| !s.shutdown)
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
        round(pools, starts, waits.peer, &shared.short);
    }
}

/// Every worker, then every idle blocking thread, of each pool releases.
fn round(pools: &[Pool], starts: u64, peer: Duration, short: &Arc<AtomicU64>) {
    for pool in pools {
        for (n, blocking) in [(pool.workers, false), (pool.blocking_alive(), true)] {
            if !still_idle(starts) {
                return;
            }
            rendezvous(&pool.handle, n, blocking, starts, peer, short);
        }
    }
}

/// Runs `n` closures on `handle` (as tasks, or on its blocking pool), each
/// releasing on its thread and then waiting briefly for its peers so that
/// each lands on a thread of its own. Returns how many reported in time.
fn rendezvous(
    handle: &Handle,
    n: usize,
    blocking: bool,
    starts: u64,
    peer: Duration,
    short: &Arc<AtomicU64>,
) -> usize {
    if n == 0 {
        return 0;
    }
    let gate = Arc::new((Mutex::new(0_usize), Condvar::new()));
    let (tx, rx) = mpsc::channel();
    for _ in 0..n {
        let gate = Arc::clone(&gate);
        let tx = tx.clone();
        let short = Arc::clone(short);
        let f = move || {
            // A closure that runs late, after a query started, leaves the
            // query's threads alone.
            if !still_idle(starts) {
                tx.send(()).ok();
                return;
            }
            krishiv_memory::release_if_due_here();
            let (count, arrived) = &*gate;
            let mut c = count.lock().unwrap_or_else(PoisonError::into_inner);
            *c += 1;
            arrived.notify_all();
            let deadline = Instant::now() + peer;
            while *c < n && still_idle(starts) {
                let now = Instant::now();
                if now >= deadline {
                    short.fetch_add(1, Ordering::Relaxed);
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

#[cfg(all(test, not(loom)))]
#[expect(
    clippy::disallowed_methods,
    reason = "test threads sleep and wait; never on a compute worker"
)]
#[expect(clippy::unwrap_used, reason = "test helpers build runtimes")]
mod tests {
    use std::collections::HashSet;
    use std::sync::Barrier;
    use std::thread::ThreadId;

    use super::*;

    /// Threads that called the release hook: (thread name, id).
    static HITS: Mutex<Vec<(String, ThreadId)>> = Mutex::new(Vec::new());
    /// The tests share the hook and the records.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn hook() {
        let t = std::thread::current();
        HITS.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((t.name().unwrap_or("").to_owned(), t.id()));
    }

    fn hits(name: &str) -> HashSet<ThreadId> {
        HITS.lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|(n, _)| n == name)
            .map(|(_, id)| *id)
            .collect()
    }

    /// A runtime of `workers` workers and up to two blocking threads, named
    /// `name`, counted and with the park check, as `TokioExecutor::new`
    /// builds them; two blocking threads are started.
    fn pool(name: &str, workers: usize) -> (tokio::runtime::Runtime, Pool) {
        let count = Arc::new(ThreadCount::default());
        let (up, down) = (Arc::clone(&count), Arc::clone(&count));
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(workers)
            .max_blocking_threads(2)
            .thread_name(name)
            .on_thread_park(|| {
                krishiv_memory::release_if_idle_here();
            })
            .on_thread_start(move || up.start())
            .on_thread_stop(move || down.stop())
            .build()
            .unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let warm: Vec<_> = (0..2)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                rt.spawn_blocking(move || {
                    barrier.wait();
                })
            })
            .collect();
        rt.block_on(async {
            for w in warm {
                w.await.unwrap();
            }
        });
        let pool = Pool {
            handle: rt.handle().clone(),
            workers,
            blocking_cap: 2,
            threads: count,
        };
        (rt, pool)
    }

    /// Waits up to ten seconds for `done`.
    fn eventually(mut done: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if done() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        done()
    }

    /// Waits long enough that a loaded machine never cuts a rendezvous short.
    const PATIENT: Waits = Waits {
        debounce: Duration::from_millis(10),
        peer: Duration::from_secs(2),
    };

    #[test]
    fn a_round_reaches_every_worker_and_blocking_thread_and_starts_none() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        krishiv_memory::set_release_hook(hook);
        let (_c, compute) = pool("idle-test-c", 3);
        let (_i, io) = pool("idle-test-io", 2);
        HITS.lock().unwrap_or_else(PoisonError::into_inner).clear();
        let releaser = Releaser::start(vec![compute, io], PATIENT).unwrap();
        releaser.request();
        // Three workers and two blocking threads; two and two.
        assert!(
            eventually(|| hits("idle-test-c").len() >= 5 && hits("idle-test-io").len() >= 4),
            "{:?} {:?}",
            hits("idle-test-c"),
            hits("idle-test-io")
        );
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(hits("idle-test-c").len(), 5, "no thread started");
        assert_eq!(hits("idle-test-io").len(), 4, "no thread started");
        assert_eq!(releaser.rounds(), 1);
        assert_eq!(releaser.short_waits(), 0);
    }

    #[test]
    fn no_request_is_lost() {
        let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
        krishiv_memory::set_release_hook(hook);
        let (_c, compute) = pool("idle-test-r", 2);
        let releaser = Releaser::start(vec![compute], PATIENT).unwrap();
        releaser.request();
        assert!(eventually(|| releaser.rounds() == 1));
        // A request once that round started gets a round of its own.
        releaser.request();
        assert!(eventually(|| releaser.rounds() == 2));
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(releaser.rounds(), 2);
    }
}
