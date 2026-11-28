// SPDX-License-Identifier: Apache-2.0
//! Releasing freed memory when the process goes idle (`krishiv_exec::idle`,
//! design `0.4-memory-query-end-release`): the last query's end runs a round
//! on the executor's threads (full coverage: `idle::tests`), a query
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
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
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

#[test]
fn the_last_query_ending_runs_a_round_on_the_executors_threads() {
    let _serial = setup();
    let exec = executor(true);
    HITS.lock().unwrap_or_else(PoisonError::into_inner).clear();
    drop(query_run(&as_dyn(&exec)));
    assert!(eventually(|| exec.idle_rounds() == 1));
    // Coverage of every thread is checked with patient waits in the unit
    // tests (`idle::tests`); here, that the round reached this executor.
    assert!(eventually(
        || !hit_threads("krishiv-compute").is_empty() && !hit_threads("krishiv-io").is_empty()
    ));
}

#[test]
fn a_query_starting_during_the_debounce_cancels_the_round() {
    let _serial = setup();
    let exec = executor(true);
    let first = query_run(&as_dyn(&exec));
    drop(first);
    let ended = Instant::now();
    let second = query_run(&as_dyn(&exec));
    if ended.elapsed() >= Duration::from_millis(500) {
        // The test thread was descheduled past half the debounce: the
        // round may have run legitimately, so nothing can be asserted.
        return;
    }
    std::thread::sleep(Duration::from_millis(1200));
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
    std::thread::sleep(Duration::from_millis(1200));
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
    std::thread::sleep(Duration::from_millis(1200));
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
    std::thread::sleep(Duration::from_millis(1200));
    assert_eq!(exec.idle_rounds(), 0);
    assert_eq!(krishiv_memory::idle_epoch(), epoch);
}
