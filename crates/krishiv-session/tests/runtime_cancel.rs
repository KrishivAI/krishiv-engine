// SPDX-License-Identifier: Apache-2.0
//! Cancellation with reasons and the statement timeout under one engine
//! runtime (design P2-E1, unit R5): the stream ends with the reason's error,
//! the first reason wins, `system.queries` keeps `state = 'cancelled'` and
//! shows `cancel_reason`.
#![cfg(test)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{Array, RecordBatch, StringArray};
use futures::StreamExt;
use futures::executor::block_on;
use krishiv_session::{
    CancelReason, EngineRuntime, QueryId, QueryOptions, RecordBatchStream, Session,
};

/// A query that runs until it is cancelled.
const LONG: &str = "SELECT id FROM range(0, 4000000000)";
const OP: &str = "00112233-4455-6677-8899-aabbccddeeff";

/// Polls `stream` until it ends; the class of the error that ended it.
fn end_of(mut stream: RecordBatchStream) -> Option<String> {
    while let Some(item) = block_on(stream.next()) {
        if let Err(e) = item {
            return Some(e.error_class().name().to_owned());
        }
    }
    None
}

/// A running stream of `LONG` named `OP`, one batch already read.
fn running(s: &Session, options: QueryOptions) -> RecordBatchStream {
    let df = block_on(s.sql_with(LONG, options.operation_id(OP))).unwrap();
    let mut stream = block_on(df.execute_stream()).unwrap();
    assert!(block_on(stream.next()).unwrap().is_ok());
    stream
}

/// `state|cancel_reason` of the rows whose text is `text`, oldest first.
fn outcome(s: &Session, text: &str) -> Vec<String> {
    let df = block_on(s.sql(&format!(
        "SELECT concat(state, '|', coalesce(cancel_reason, '-')) FROM system.queries \
         WHERE query_text = '{}'",
        text.replace('\'', "''")
    )))
    .unwrap();
    let batches: Vec<RecordBatch> = block_on(df.collect()).unwrap();
    batches
        .iter()
        .flat_map(|b| {
            let any = b.column(0).as_any();
            let len = b.column(0).len();
            (0..len)
                .map(|i| {
                    any.downcast_ref::<StringArray>().map_or_else(
                        || {
                            any.downcast_ref::<arrow_array::StringViewArray>()
                                .map(|a| a.value(i).to_owned())
                                .unwrap_or_default()
                        },
                        |a| a.value(i).to_owned(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Waits for the row of `text` to leave `running`.
fn settled(s: &Session, text: &str) -> Vec<String> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let rows = outcome(s, text);
        if rows.iter().all(|r| !r.starts_with("running|")) {
            return rows;
        }
        assert!(Instant::now() < deadline, "still running: {rows:?}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn a_statement_timeout_ends_the_stream_with_krishiv_query_timeout() {
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.query.maxExecutionTime", "300ms")
        .unwrap();
    let started = Instant::now();
    let stream = running(&s, QueryOptions::new());
    assert_eq!(end_of(stream).as_deref(), Some("KRISHIV_QUERY_TIMEOUT"));
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "{:?}",
        started.elapsed()
    );
    assert_eq!(settled(&s, LONG), ["cancelled|timeout"]);
}

#[test]
fn a_statement_deadline_tightens_the_session_limit_and_never_loosens_it() {
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.query.maxExecutionTime", "1h")
        .unwrap();
    let stream = running(&s, QueryOptions::new().deadline(Duration::from_millis(200)));
    assert_eq!(end_of(stream).as_deref(), Some("KRISHIV_QUERY_TIMEOUT"));
    // A longer statement deadline does not lift a shorter session limit.
    s.conf()
        .set("krishiv.query.maxExecutionTime", "200ms")
        .unwrap();
    let df = block_on(s.sql_with(
        "SELECT id FROM range(0, 4000000001)",
        QueryOptions::new().deadline(Duration::from_secs(3600)),
    ))
    .unwrap();
    let stream = block_on(df.execute_stream()).unwrap();
    assert_eq!(end_of(stream).as_deref(), Some("KRISHIV_QUERY_TIMEOUT"));
}

#[test]
fn a_query_that_ended_before_its_deadline_stays_finished() {
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.query.maxExecutionTime", "2s")
        .unwrap();
    let text = "SELECT 1 AS quick";
    block_on(block_on(s.sql(text)).unwrap().collect()).unwrap();
    std::thread::sleep(Duration::from_millis(2500));
    assert_eq!(outcome(&s, text), ["finished|-"]);
}

#[test]
fn a_users_cancel_from_another_thread_is_operation_canceled() {
    let s = Session::new().unwrap();
    let stream = running(&s, QueryOptions::new());
    let id = QueryId::parse(OP).unwrap();
    let canceller = {
        let (s, id) = (s.clone(), id.clone());
        std::thread::spawn(move || s.cancel(&id))
    };
    assert!(canceller.join().unwrap());
    assert_eq!(end_of(stream).as_deref(), Some("OPERATION_CANCELED"));
    assert_eq!(settled(&s, LONG), ["cancelled|user"]);
    // An ended or unknown id: nothing to cancel.
    assert!(!s.cancel(&id));
    assert!(!s.cancel(&QueryId::parse("1-2-3-4-5").unwrap()));
}

#[test]
fn an_operators_kill_reaches_any_session_with_krishiv_query_killed() {
    let rt = EngineRuntime::builder().build().unwrap();
    let a = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let b = a.new_session();
    let stream = running(&a, QueryOptions::new());
    let id = QueryId::parse(OP).unwrap();
    // Not this session's query: `Session::cancel` does not reach it.
    assert!(!b.cancel(&id));
    assert!(rt.cancel(&id, CancelReason::Admin));
    assert_eq!(end_of(stream).as_deref(), Some("KRISHIV_QUERY_KILLED"));
    assert_eq!(settled(&a, LONG), ["cancelled|admin"]);
    assert!(!rt.cancel(&id, CancelReason::Drain), "it already ended");
}

#[test]
fn the_first_reason_wins_under_concurrent_cancels() {
    let rt = EngineRuntime::builder().build().unwrap();
    let s = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let stream = running(&s, QueryOptions::new());
    let id = QueryId::parse(OP).unwrap();
    let reasons = [
        CancelReason::User,
        CancelReason::Admin,
        CancelReason::Drain,
        CancelReason::Timeout,
    ];
    let won: Vec<bool> = std::thread::scope(|scope| {
        let handles: Vec<_> = reasons
            .iter()
            .map(|r| {
                let (rt, id) = (&rt, &id);
                scope.spawn(move || rt.cancel(id, *r))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(won.iter().filter(|w| **w).count(), 1, "{won:?}");
    let winner = reasons[won.iter().position(|w| *w).unwrap()];
    let class = end_of(stream).unwrap();
    let expected = match winner {
        CancelReason::User => "OPERATION_CANCELED",
        CancelReason::Timeout => "KRISHIV_QUERY_TIMEOUT",
        _ => "KRISHIV_QUERY_KILLED",
    };
    assert_eq!(class, expected);
    assert_eq!(settled(&s, LONG), [format!("cancelled|{winner}")]);
}

#[test]
fn interrupt_all_cancels_the_sessions_queries_and_returns_their_ids() {
    let rt = EngineRuntime::builder().build().unwrap();
    let a = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let b = a.new_session();
    let first = running(&a, QueryOptions::new());
    let other = "1-2-3-4-5";
    let df = block_on(a.sql_with(
        "SELECT id FROM range(0, 4000000002)",
        QueryOptions::new().operation_id(other),
    ))
    .unwrap();
    let mut second = block_on(df.execute_stream()).unwrap();
    assert!(block_on(second.next()).unwrap().is_ok());
    let mut untouched = running(&b, QueryOptions::new());
    let mut ids: Vec<String> = a
        .interrupt_all()
        .iter()
        .map(|i| i.as_str().to_owned())
        .collect();
    ids.sort();
    assert_eq!(ids, ["00000001-0002-0003-0004-000000000005", OP]);
    assert_eq!(end_of(first).as_deref(), Some("OPERATION_CANCELED"));
    assert_eq!(end_of(second).as_deref(), Some("OPERATION_CANCELED"));
    assert!(block_on(untouched.next()).unwrap().is_ok());
    assert!(a.interrupt_all().is_empty());
}

#[test]
fn closing_a_session_records_client_gone() {
    let s = Session::new().unwrap();
    let stream = running(&s, QueryOptions::new());
    let watcher = s.new_session();
    assert_eq!(s.close(), 1);
    assert_eq!(end_of(stream).as_deref(), Some("OPERATION_CANCELED"));
    assert_eq!(settled(&watcher, LONG), ["cancelled|client_gone"]);
}

#[test]
fn the_timer_never_keeps_the_runtime_alive() {
    let rt = EngineRuntime::builder()
        .config("krishiv.query.maxExecutionTime", "1h")
        .build()
        .unwrap();
    let weak = Arc::downgrade(&rt);
    let s = Session::builder().runtime(rt).build().unwrap();
    // A query with a deadline starts the timer.
    block_on(block_on(s.sql("SELECT 1")).unwrap().collect()).unwrap();
    drop(s);
    let deadline = Instant::now() + Duration::from_secs(30);
    while weak.upgrade().is_some() {
        assert!(
            Instant::now() < deadline,
            "the runtime outlived its sessions"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_cancel_feeds_the_cancel_latency_histogram() {
    let s = Session::builder()
        .config("krishiv.metrics.enabled", "true")
        .build()
        .unwrap();
    let stream = running(&s, QueryOptions::new());
    assert!(s.cancel(&QueryId::parse(OP).unwrap()));
    assert!(end_of(stream).is_some());
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let found = s
            .metrics()
            .histograms
            .iter()
            .any(|h| h.name == "krishiv.query.cancel.latency" && h.count == 1);
        if found {
            break;
        }
        assert!(Instant::now() < deadline, "no cancel latency observed");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// No part file of a write that a timeout or an operator stopped.
fn part_files(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir).map_or(0, |entries| {
        entries
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().starts_with("part-"))
            .count()
    })
}

#[test]
fn a_write_cancelled_for_any_reason_leaves_no_files() {
    let out = tempfile::tempdir().unwrap();
    // Timeout.
    let s = Session::new().unwrap();
    let path = out.path().join("timeout");
    let df = block_on(s.sql_with(
        LONG,
        QueryOptions::new().deadline(Duration::from_millis(300)),
    ))
    .unwrap();
    let e = block_on(df.write().format("parquet").save(path.to_str().unwrap())).unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_QUERY_TIMEOUT");
    assert_eq!(part_files(&path), 0);
    // An operator's kill, from another thread while the write runs.
    let rt = EngineRuntime::builder().build().unwrap();
    let s = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let path = out.path().join("admin");
    let watcher = s.clone();
    let writer = {
        let path = path.clone();
        std::thread::spawn(move || {
            let df = block_on(s.sql_with(LONG, QueryOptions::new().operation_id(OP))).unwrap();
            block_on(df.write().format("parquet").save(path.to_str().unwrap()))
        })
    };
    // Kill only once the write's own query runs: a cancel that comes while the statement is still
    // being analyzed ends the `sql_with` call itself (`action.rs`), which is not what this checks.
    let until = Instant::now() + Duration::from_secs(60);
    while watcher.running_queries() == 0 {
        assert!(
            !writer.is_finished(),
            "the write ended before it ran: {:?}",
            writer.join()
        );
        assert!(Instant::now() < until, "the write never started");
        std::thread::sleep(Duration::from_millis(10));
    }
    let id = QueryId::parse(OP).unwrap();
    assert!(
        rt.cancel(&id, CancelReason::Admin),
        "a running write is cancelled"
    );
    let e = writer.join().unwrap().unwrap_err();
    assert_eq!(e.error_class().name(), "KRISHIV_QUERY_KILLED");
    assert_eq!(part_files(&path), 0);
}

#[test]
fn a_dropped_stream_is_not_a_timed_cancellation() {
    let s = Session::builder()
        .config("krishiv.metrics.enabled", "true")
        .build()
        .unwrap();
    let stream = running(&s, QueryOptions::new());
    drop(stream);
    // The query's fold has run once its row settled.
    assert_eq!(settled(&s, LONG), ["cancelled|-"]);
    let samples: u64 = s
        .metrics()
        .histograms
        .iter()
        .filter(|h| h.name == "krishiv.query.cancel.latency")
        .map(|h| h.count)
        .sum();
    assert_eq!(samples, 0);
}

/// `krishiv.query.maxQueuedTime` is reserved for admission: a duration is
/// accepted and changes nothing (no admission queue exists yet).
#[test]
fn the_reserved_queue_time_is_accepted_and_has_no_effect() {
    let s = Session::new().unwrap();
    s.conf().set("krishiv.query.maxQueuedTime", "1ms").unwrap();
    let rows = block_on(async { s.sql("SELECT 41 + 1").await?.collect().await }).unwrap();
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    assert_eq!(settled(&s, "SELECT 41 + 1"), ["finished|-"]);
    assert!(s.conf().set("krishiv.query.maxQueuedTime", "soon").is_err());
}
