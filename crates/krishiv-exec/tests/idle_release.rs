// SPDX-License-Identifier: Apache-2.0
//! Releasing freed memory when the process goes idle (`krishiv_exec::idle`,
//! design `0.4-memory-query-end-release`): the releaser's round reaches every
//! worker and every idle blocking thread without starting new ones, a query
//! that starts during the debounce cancels it, and nothing happens while a
//! query runs or with the setting off. The running-query counter and the
//! release hook are process-wide, so the tests take one lock.
#![cfg(not(loom))]
#![expect(clippy::unwrap_used, reason = "test code")]
#![expect(
    clippy::disallowed_methods,
    reason = "test threads sleep and wait; never on a compute worker"
)]

use std::collections::HashSet;
use std::sync::{Arc, Barrier, Mutex, MutexGuard, PoisonError};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use krishiv_exec::{Executor, ExecutorConfig, TokioExecutor, queries_running, query_run};

/// Threads that called the release hook: (thread name, id).
static HITS: Mutex<Vec<(String, ThreadId)>> = Mutex::new(Vec::new());
static SERIAL: Mutex<()> = Mutex::new(());

fn hook() {
    let t = std::thread::current();
    HITS.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push((t.name().unwrap_or("").to_owned(), t.id()));
}

fn setup() -> MutexGuard<'static, ()> {
    let guard = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    // Installed once per process; later calls return false.
    krishiv_memory::set_release_hook(hook);
    guard
}

fn executor(release: bool) -> Arc<TokioExecutor> {
    TokioExecutor::new(&ExecutorConfig {
        compute_threads: 3,
        io_threads: 2,
        blocking_threads: 2,
        io_blocking_threads: 2,
        release_when_idle: release,
    })
    .unwrap()
}

fn as_dyn(e: &Arc<TokioExecutor>) -> Arc<dyn Executor> {
    Arc::clone(e) as Arc<dyn Executor>
}

/// Waits up to five seconds for `done`.
fn eventually(mut done: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if done() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    done()
}

/// Distinct threads named `name` that called the hook.
fn hit_threads(name: &str) -> HashSet<ThreadId> {
    HITS.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .filter(|(n, _)| n == name)
        .map(|(_, id)| *id)
        .collect()
}

/// Starts two blocking threads on `handle` (two closures that meet) and
/// returns their ids.
fn warm_blocking(handle: &tokio::runtime::Handle) -> HashSet<ThreadId> {
    let barrier = Arc::new(Barrier::new(2));
    let tasks: Vec<_> = (0..2)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            handle.spawn_blocking(move || {
                barrier.wait();
                std::thread::current().id()
            })
        })
        .collect();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    rt.block_on(async {
        let mut ids = HashSet::new();
        for t in tasks {
            ids.insert(t.await.unwrap());
        }
        ids
    })
}

#[test]
fn a_round_reaches_every_worker_and_idle_blocking_thread() {
    let _serial = setup();
    let exec = executor(true);
    let compute_blocking = warm_blocking(exec.compute_handle());
    let io_blocking = warm_blocking(exec.io_handle());
    HITS.lock().unwrap_or_else(PoisonError::into_inner).clear();

    drop(query_run(&as_dyn(&exec)));
    // Three workers and the two blocking threads of each runtime.
    assert!(
        eventually(
            || hit_threads("krishiv-compute").len() >= 5 && hit_threads("krishiv-io").len() >= 4
        ),
        "compute {:?}, io {:?}",
        hit_threads("krishiv-compute"),
        hit_threads("krishiv-io")
    );
    assert_eq!(exec.idle_rounds(), 1);
    // The warmed blocking threads released, and no other thread was started.
    let compute = hit_threads("krishiv-compute");
    let io = hit_threads("krishiv-io");
    assert!(compute_blocking.is_subset(&compute), "{compute:?}");
    assert!(io_blocking.is_subset(&io), "{io:?}");
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(hit_threads("krishiv-compute").len(), 5);
    assert_eq!(hit_threads("krishiv-io").len(), 4);
}

#[test]
fn a_query_starting_during_the_debounce_cancels_the_round() {
    let _serial = setup();
    let exec = executor(true);
    let first = query_run(&as_dyn(&exec));
    drop(first);
    let ended = Instant::now();
    let second = query_run(&as_dyn(&exec));
    if ended.elapsed() >= Duration::from_millis(5) {
        // The test thread was descheduled past half the debounce: the
        // round may have run legitimately, so nothing can be asserted.
        return;
    }
    std::thread::sleep(Duration::from_millis(60));
    assert_eq!(exec.idle_rounds(), 0, "a query started within the debounce");
    drop(second);
    assert!(eventually(|| exec.idle_rounds() == 1));
}

#[test]
fn nothing_is_released_while_a_query_runs() {
    let _serial = setup();
    let exec = executor(true);
    let long = query_run(&as_dyn(&exec));
    let short = query_run(&as_dyn(&exec));
    assert_eq!(queries_running(), 2);
    let epoch = krishiv_memory::idle_epoch();
    drop(short);
    std::thread::sleep(Duration::from_millis(60));
    assert_eq!(exec.idle_rounds(), 0);
    assert_eq!(krishiv_memory::idle_epoch(), epoch);
    drop(long);
    assert!(eventually(|| exec.idle_rounds() == 1));
    assert!(krishiv_memory::idle_epoch() > epoch);
    assert_eq!(queries_running(), 0);
}

#[test]
fn the_setting_off_starts_no_round() {
    let _serial = setup();
    let exec = executor(false);
    let epoch = krishiv_memory::idle_epoch();
    drop(query_run(&as_dyn(&exec)));
    std::thread::sleep(Duration::from_millis(60));
    assert_eq!(exec.idle_rounds(), 0);
    assert_eq!(krishiv_memory::idle_epoch(), epoch);
}

#[test]
fn a_run_outliving_its_executor_ends_quietly() {
    let _serial = setup();
    let exec = executor(true);
    let run = query_run(&as_dyn(&exec));
    drop(exec);
    drop(run);
    assert_eq!(queries_running(), 0);
}

#[test]
fn a_deterministic_executor_runs_no_round() {
    let _serial = setup();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let exec = rt
        .block_on(async { TokioExecutor::deterministic(2) })
        .unwrap();
    let epoch = krishiv_memory::idle_epoch();
    drop(query_run(&as_dyn(&exec)));
    std::thread::sleep(Duration::from_millis(60));
    assert_eq!(exec.idle_rounds(), 0);
    assert_eq!(krishiv_memory::idle_epoch(), epoch);
}
