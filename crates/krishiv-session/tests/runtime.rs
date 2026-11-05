// SPDX-License-Identifier: Apache-2.0
//! One engine runtime shared by many sessions (design P2-E1): what the
//! sessions share, what each keeps, how a session closes and when the
//! runtime ends.
#![cfg(test)]

use std::sync::Arc;

use futures::StreamExt;
use futures::executor::block_on;
use krishiv_plan::analyzer::{Principal, PrincipalKind};
use krishiv_session::{EngineRuntime, Session};

/// A query that runs long enough to be cancelled while its stream is open.
const LONG: &str = "SELECT id FROM range(0, 2000000000)";

/// Polls `stream` until it ends; the error that ended it, if any.
fn drain(mut stream: krishiv_session::RecordBatchStream) -> Option<String> {
    while let Some(item) = block_on(stream.next()) {
        if let Err(e) = item {
            return Some(e.error_class().name().to_owned());
        }
    }
    None
}

#[test]
fn fifty_sessions_share_one_runtime() {
    let rt = EngineRuntime::builder()
        .config("krishiv.metrics.enabled", "true")
        .build()
        .unwrap();
    let sessions: Vec<Session> = (0..50)
        .map(|_| Session::builder().runtime(Arc::clone(&rt)).build().unwrap())
        .collect();
    for s in &sessions {
        assert!(Arc::ptr_eq(s.runtime(), &rt));
        assert!(Arc::ptr_eq(s.runtime().executor(), rt.executor()));
        assert!(Arc::ptr_eq(&s.runtime().memory(), &rt.memory()));
        let df = block_on(s.sql("SELECT 1 AS one")).unwrap();
        assert_eq!(block_on(df.collect()).unwrap()[0].num_rows(), 1);
    }
    // One registry: every session's query folded into it, and a scrape from
    // any session sees all of them.
    let queries = |snapshot: &krishiv_exec::MetricsSnapshot| -> f64 {
        snapshot
            .counters
            .iter()
            .filter(|c| c.name == "krishiv.query.count")
            .map(|c| c.value)
            .sum()
    };
    assert!(queries(&sessions[0].metrics()) >= 50.0);
    assert_eq!(
        queries(&sessions[0].metrics()),
        queries(&sessions[49].metrics())
    );
    // One gauge set, not one per session.
    let gauges = rt
        .metrics()
        .gauges
        .iter()
        .filter(|g| g.name == "krishiv.memory.limit")
        .count();
    assert_eq!(gauges, 1);
}

#[test]
fn runtime_keys_on_a_session_of_a_supplied_runtime_are_refused() {
    let rt = EngineRuntime::builder()
        .config("krishiv.memory.limit", "1g")
        .build()
        .unwrap();
    for key in ["krishiv.memory.limit", "krishiv.spill.maxBytes"] {
        let e = Session::builder()
            .runtime(Arc::clone(&rt))
            .config(key, "64m")
            .build()
            .unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "CANNOT_MODIFY_STATIC_CONFIG",
            "{key}"
        );
        assert_eq!(e.sql_state(), Some("46110"));
    }
    let e = Session::builder()
        .runtime(Arc::clone(&rt))
        .executor(Arc::clone(rt.executor()))
        .build()
        .unwrap_err();
    assert_eq!(e.error_class().name(), "CANNOT_MODIFY_STATIC_CONFIG");
    // Session keys are fine, and start from the runtime's.
    let s = Session::builder()
        .runtime(Arc::clone(&rt))
        .config("spark.sql.ansi.enabled", "false")
        .build()
        .unwrap();
    assert_eq!(
        s.conf().get("spark.sql.ansi.enabled").as_deref(),
        Some("false")
    );
    assert_eq!(s.conf().get("krishiv.memory.limit").as_deref(), Some("1GB"));
    // A session built alone keeps taking every key (embedded mode).
    let alone = Session::builder()
        .config("krishiv.memory.limit", "64m")
        .build()
        .unwrap();
    assert_eq!(alone.runtime().memory().limit(), Some(64 << 20));
}

#[test]
fn runtime_session_defaults_reach_every_session() {
    let rt = EngineRuntime::builder()
        .config("spark.sql.ansi.enabled", "false")
        .build()
        .unwrap();
    let a = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    assert_eq!(
        a.conf().get("spark.sql.ansi.enabled").as_deref(),
        Some("false")
    );
    a.conf().set("spark.sql.ansi.enabled", "true").unwrap();
    let b = a.new_session();
    assert_eq!(
        b.conf().get("spark.sql.ansi.enabled").as_deref(),
        Some("false")
    );
}

#[test]
fn a_new_session_has_fresh_state_and_the_same_principal() {
    let a = Session::builder()
        .principal(Principal::new("alice", PrincipalKind::User))
        .build()
        .unwrap();
    for stmt in [
        "CREATE TEMPORARY VIEW v AS SELECT 1 AS x",
        "DECLARE VARIABLE var1 INT DEFAULT 7",
        "CREATE TEMPORARY FUNCTION f1() RETURNS INT RETURN 1",
    ] {
        block_on(block_on(a.sql(stmt)).unwrap().collect()).unwrap();
    }
    a.set_tag("team", "etl").unwrap();
    let b = a.new_session();
    assert!(Arc::ptr_eq(a.runtime(), b.runtime()));
    assert_eq!(
        format!("{b:?}"),
        format!("{a:?}"),
        "same principal and profile"
    );
    for (query, class) in [
        ("SELECT * FROM v", "TABLE_OR_VIEW_NOT_FOUND"),
        ("SELECT var1", "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION"),
        ("SELECT f1()", "UNRESOLVED_ROUTINE"),
    ] {
        let e = block_on(b.sql(query)).unwrap_err();
        assert_eq!(e.error_class().name(), class, "{query}");
    }
    assert!(b.tags().is_empty());
    // The parent still has all of them.
    assert!(block_on(a.sql("SELECT * FROM v, (SELECT var1), (SELECT f1())")).is_ok());
}

#[test]
fn an_open_stream_keeps_the_runtime_alive_until_it_drops() {
    let rt = EngineRuntime::builder().build().unwrap();
    let weak = Arc::downgrade(&rt);
    let s = Session::builder().runtime(rt).build().unwrap();
    let df = block_on(s.sql(LONG)).unwrap();
    let mut stream = block_on(df.execute_stream()).unwrap();
    assert!(block_on(stream.next()).unwrap().is_ok());
    drop(df);
    drop(s);
    assert!(weak.upgrade().is_some(), "the stream holds the runtime");
    // Dropping the session's handles cancelled nothing.
    assert!(block_on(stream.next()).unwrap().is_ok());
    drop(stream);
    // The query's tasks end once cancelled; the runtime with them.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while weak.upgrade().is_some() {
        assert!(
            std::time::Instant::now() < deadline,
            "the runtime outlived its last stream"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

#[test]
fn closing_a_session_cancels_its_open_stream_and_refuses_new_statements() {
    let rt = EngineRuntime::builder().build().unwrap();
    let a = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let b = a.new_session();
    let stream_a = block_on(block_on(a.sql(LONG)).unwrap().execute_stream()).unwrap();
    let mut stream_b = block_on(block_on(b.sql(LONG)).unwrap().execute_stream()).unwrap();
    let df_a = block_on(a.sql("SELECT 1")).unwrap();
    assert_eq!(a.close(), 1);
    assert_eq!(a.close(), 0, "closing again is a no-op");
    assert_eq!(drain(stream_a).as_deref(), Some("OPERATION_CANCELED"));
    let e = block_on(a.sql("SELECT 1")).unwrap_err();
    assert_eq!(e.error_class().name(), "INVALID_HANDLE.SESSION_CLOSED");
    let e = block_on(df_a.collect()).unwrap_err();
    assert_eq!(e.error_class().name(), "INVALID_HANDLE.SESSION_CLOSED");
    // The other session of the runtime runs on.
    assert!(block_on(stream_b.next()).unwrap().is_ok());
    assert!(block_on(block_on(b.sql("SELECT 1")).unwrap().collect()).is_ok());
}
