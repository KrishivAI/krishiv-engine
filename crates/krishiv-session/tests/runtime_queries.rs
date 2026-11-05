// SPDX-License-Identifier: Apache-2.0
//! Query ids and `system.queries` under one engine runtime (design P2-E1,
//! unit R4): ids unique across sessions, caller operation ids scoped per
//! session, one history bounded per principal and read per principal.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::{Array, LargeStringArray, RecordBatch, StringArray, StringViewArray};
use futures::StreamExt;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::{BoxFuture, PolicyVersion};
use krishiv_session::{
    AccessDecision, AccessPolicy, EngineRuntime, Principal, PrincipalKind, Privilege, QueryOptions,
    Securable, Session,
};

const OP: &str = "00112233-4455-6677-8899-aabbccddeeff";
const LONG: &str = "SELECT id FROM range(0, 2000000000)";

/// Allows everything except `MONITOR` on the `system` securable, which only
/// `admin` holds.
struct MonitorForAdmin;

impl AccessPolicy for MonitorForAdmin {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        principal: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        let decision = match (securable, privilege) {
            (Securable::System(_), Privilege::Monitor) if principal.name != "admin" => {
                AccessDecision::Deny {
                    reason: "monitor".to_owned(),
                }
            }
            _ => AccessDecision::Allow,
        };
        Box::pin(std::future::ready(Ok(decision)))
    }
}

fn session_of(rt: &Arc<EngineRuntime>, who: &str) -> Session {
    Session::builder()
        .runtime(Arc::clone(rt))
        .principal(Principal::new(who, PrincipalKind::User))
        .access_policy(Arc::new(MonitorForAdmin))
        .build()
        .unwrap()
}

/// One text column of `query`'s rows, in order.
fn column(s: &Session, query: &str) -> Vec<Option<String>> {
    let batches: Vec<RecordBatch> = block_on(block_on(s.sql(query)).unwrap().collect()).unwrap();
    batches
        .iter()
        .flat_map(|b| {
            let c = b.column(0);
            let any = c.as_any();
            (0..c.len())
                .map(|i| {
                    c.is_valid(i).then(|| {
                        if let Some(a) = any.downcast_ref::<StringArray>() {
                            a.value(i).to_owned()
                        } else if let Some(a) = any.downcast_ref::<LargeStringArray>() {
                            a.value(i).to_owned()
                        } else {
                            let a = any.downcast_ref::<StringViewArray>().unwrap();
                            a.value(i).to_owned()
                        }
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn run(s: &Session, query: &str) {
    block_on(block_on(s.sql(query)).unwrap().collect()).unwrap();
}

#[test]
fn two_sessions_may_run_one_operation_id_at_once() {
    let rt = EngineRuntime::builder().build().unwrap();
    let a = session_of(&rt, "alice");
    let b = a.new_session();
    // Both statements take the id at the same moment.
    let barrier = std::sync::Barrier::new(2);
    let streams: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = [&a, &b]
            .into_iter()
            .map(|s| {
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    let df =
                        block_on(s.sql_with(LONG, QueryOptions::new().operation_id(OP))).unwrap();
                    let mut stream = block_on(df.execute_stream()).unwrap();
                    assert!(block_on(stream.next()).unwrap().is_ok());
                    stream
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let ids = column(
        &a,
        "SELECT query_id FROM system.queries WHERE state = 'running' ORDER BY session_id",
    );
    assert_eq!(ids.iter().filter(|i| i.as_deref() == Some(OP)).count(), 2);
    drop(streams);
}

#[test]
fn an_operation_id_is_taken_while_its_statement_lives_and_for_a_while_after() {
    let rt = EngineRuntime::builder().build().unwrap();
    let a = session_of(&rt, "alice");
    let options = QueryOptions::new().operation_id(OP);
    let df = block_on(a.sql_with(LONG, options.clone())).unwrap();
    let mut running = block_on(df.execute_stream()).unwrap();
    assert!(block_on(running.next()).unwrap().is_ok());
    let e = block_on(a.sql_with("SELECT 1", options.clone())).unwrap_err();
    assert_eq!(
        e.error_class().name(),
        "INVALID_HANDLE.OPERATION_ALREADY_EXISTS"
    );
    assert_eq!(e.sql_state(), Some("HY000"));
    // Another session may use it.
    let b = a.new_session();
    block_on(
        block_on(b.sql_with("SELECT 1", options.clone()))
            .unwrap()
            .collect(),
    )
    .unwrap();
    // The statement's DataFrame and stream gone, the operation finished:
    // Spark Connect keeps the id taken as a recently finished operation.
    drop(df);
    drop(running);
    let e = block_on(a.sql_with("SELECT 1", options)).unwrap_err();
    assert_eq!(e.error_class().name(), "INVALID_HANDLE.OPERATION_ABANDONED");
    let ids = column(
        &a,
        "SELECT DISTINCT query_id FROM system.queries WHERE query_text LIKE 'SELECT id FROM range%'",
    );
    assert_eq!(ids, [Some(OP.to_owned())]);
}

#[test]
fn operation_ids_are_read_as_spark_connect_reads_them() {
    let s = Session::new().unwrap();
    for bad in [
        "",
        "abc",
        "1-2-3-4",
        "-1-2-3-4-5",
        "00112233-4455-6677-8899-aabbccddeeff0",
    ] {
        let e =
            block_on(s.sql_with("SELECT 1", QueryOptions::new().operation_id(bad))).unwrap_err();
        assert_eq!(e.error_class().name(), "INVALID_HANDLE.FORMAT", "{bad:?}");
    }
    // Java's `UUID.fromString(id).toString`: either case, short groups.
    for (given, shown) in [
        (OP.to_uppercase(), OP.to_owned()),
        (
            "1-2-3-4-5".to_owned(),
            "00000001-0002-0003-0004-000000000005".to_owned(),
        ),
    ] {
        let df = block_on(s.sql_with("SELECT 'x' AS op", QueryOptions::new().operation_id(given)))
            .unwrap();
        block_on(df.collect()).unwrap();
        let ids = column(
            &s,
            "SELECT query_id FROM system.queries WHERE query_text = 'SELECT ''x'' AS op'",
        );
        assert_eq!(ids.last(), Some(&Some(shown)));
    }
    // A statement refused before it ran keeps the caller's id too.
    let refused = "7-7-7-7-7";
    assert!(block_on(s.sql_with("SELECT (", QueryOptions::new().operation_id(refused))).is_err());
    let ids = column(
        &s,
        "SELECT query_id FROM system.queries WHERE query_text = 'SELECT ('",
    );
    assert_eq!(
        ids,
        [Some("00000007-0007-0007-0007-000000000007".to_owned())]
    );
}

#[test]
fn each_principal_sees_its_own_queries_and_monitor_sees_all() {
    let rt = EngineRuntime::builder().build().unwrap();
    let alice = session_of(&rt, "alice");
    let bob = session_of(&rt, "bob");
    let admin = session_of(&rt, "admin");
    run(&alice, "SELECT 'alice' AS who");
    run(&bob, "SELECT 'bob' AS who");
    let principals = |s: &Session| {
        let mut p = column(s, "SELECT DISTINCT principal FROM system.queries");
        p.sort();
        p
    };
    assert_eq!(principals(&alice), [Some("alice".to_owned())]);
    assert_eq!(principals(&bob), [Some("bob".to_owned())]);
    assert_eq!(
        principals(&admin),
        [
            Some("admin".to_owned()),
            Some("alice".to_owned()),
            Some("bob".to_owned())
        ]
    );
    // Operators follow their queries.
    let ops = |s: &Session| {
        column(
            s,
            "SELECT DISTINCT q.principal FROM system.operators o \
             JOIN system.queries q ON o.query_id = q.query_id",
        )
    };
    assert!(ops(&alice).iter().all(|p| p.as_deref() == Some("alice")));
    assert!(ops(&admin).iter().any(|p| p.as_deref() == Some("bob")));
}

#[test]
fn a_busy_principal_does_not_flush_another_principals_history() {
    let rt = EngineRuntime::builder().build().unwrap();
    let alice = session_of(&rt, "alice");
    let bob = session_of(&rt, "bob");
    run(&bob, "SELECT 'kept' AS marker");
    // Refused statements are cheap rows of their own.
    for _ in 0..10_000 {
        assert!(block_on(alice.sql("SELECT (")).is_err());
    }
    let kept = column(&bob, "SELECT query_text FROM system.queries");
    assert!(
        kept.contains(&Some("SELECT 'kept' AS marker".to_owned())),
        "{kept:?}"
    );
    let alice_rows = column(&alice, "SELECT query_id FROM system.queries");
    assert!(alice_rows.len() <= 1001, "{}", alice_rows.len());
}

#[test]
fn tags_and_traces_stay_with_their_session() {
    let rt = EngineRuntime::builder().build().unwrap();
    let a = session_of(&rt, "alice");
    let b = a.new_session();
    a.set_tag("team", "etl").unwrap();
    a.set_trace_context("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
        .unwrap();
    run(&a, "SELECT 'a' AS who");
    run(&b, "SELECT 'b' AS who");
    let rows = column(
        &a,
        "SELECT concat(query_text, '|', coalesce(trace_id, '-'), '|', \
         coalesce(element_at(tags, 'team'), '-')) FROM system.queries \
         WHERE query_text LIKE 'SELECT ''_'' AS who' ORDER BY query_text",
    );
    assert_eq!(
        rows,
        [
            Some("SELECT 'a' AS who|4bf92f3577b34da6a3ce929d0e0e4736|etl".to_owned()),
            Some("SELECT 'b' AS who|-|-".to_owned()),
        ]
    );
    // One id per session, the same on every row of it.
    let sessions = column(
        &a,
        "SELECT DISTINCT session_id FROM system.queries WHERE query_text LIKE 'SELECT ''_'' AS who'",
    );
    assert_eq!(sessions.len(), 2);
}

#[test]
fn query_ids_are_unique_across_sessions() {
    let rt = EngineRuntime::builder().build().unwrap();
    let sessions: Vec<Session> = (0..8).map(|_| session_of(&rt, "alice")).collect();
    std::thread::scope(|scope| {
        for s in &sessions {
            scope.spawn(move || {
                for _ in 0..50 {
                    run(s, "SELECT 1");
                }
            });
        }
    });
    let ids = column(&sessions[0], "SELECT query_id FROM system.queries");
    let mut unique = ids.clone();
    unique.sort();
    unique.dedup();
    assert!(ids.len() >= 400);
    assert_eq!(unique.len(), ids.len());
}

#[test]
fn at_capacity_the_heaviest_principal_loses_its_oldest_rows() {
    let rt = EngineRuntime::builder()
        .config("krishiv.system.queries.capacity", "50")
        .build()
        .unwrap();
    let alice = session_of(&rt, "alice");
    let bob = session_of(&rt, "bob");
    for i in 0..5 {
        run(&bob, &format!("SELECT {i} AS bob_marker"));
    }
    for _ in 0..100 {
        assert!(block_on(alice.sql("SELECT (")).is_err());
    }
    let admin = session_of(&rt, "admin");
    let rows = column(&admin, "SELECT principal FROM system.queries");
    assert!(rows.len() <= 51, "{}", rows.len());
    let bobs = rows.iter().filter(|p| p.as_deref() == Some("bob")).count();
    assert_eq!(bobs, 5, "bob's rows stay while alice holds the most");
}
