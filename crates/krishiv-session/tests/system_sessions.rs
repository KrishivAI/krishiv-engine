// SPDX-License-Identifier: Apache-2.0
//! `system.sessions` and the session registry (P3-T064): listing, per-principal filtering, the
//! lifecycle states, `kill_session` and the bounded ended ring.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::{Array, LargeStringArray, RecordBatch, StringArray, StringViewArray};
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::{BoxFuture, PolicyVersion};
use krishiv_session::{
    AccessDecision, AccessPolicy, CancelReason, EngineRuntime, Principal, PrincipalKind, Privilege,
    Securable, Session, Transport,
};

/// Allows everything except `MONITOR` on `system`, which only `admin` holds.
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

/// The first column of `query`, in order.
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
                            any.downcast_ref::<StringViewArray>()
                                .unwrap()
                                .value(i)
                                .to_owned()
                        }
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn texts(s: &Session, query: &str) -> Vec<String> {
    column(s, query).into_iter().flatten().collect()
}

fn id_of(s: &Session, who: &str) -> String {
    texts(
        s,
        &format!(
            "SELECT session_id FROM system.sessions WHERE principal = '{who}' AND state = 'active'"
        ),
    )
    .remove(0)
}

#[test]
fn a_session_lists_itself_with_what_the_front_end_said() {
    let rt = EngineRuntime::builder().build().unwrap();
    let a = session_of(&rt, "alice");
    a.set_client_info(
        Transport::Connect,
        Some("spark/4.3.0"),
        Some("u1"),
        Some("srv-1"),
    );
    a.set_client_info(Transport::Connect, Some("other"), None, None);
    a.set_tag("team", "etl").unwrap();
    let catalog = texts(&a, "SELECT current_catalog()").remove(0);
    let namespace = texts(&a, "SELECT current_schema()").remove(0);
    let rows = texts(
        &a,
        "SELECT concat_ws('|', transport, client_type, client_user_id, server_session_id, state, \
         current_catalog, current_namespace, tags['team']) FROM system.sessions",
    );
    assert_eq!(
        rows,
        [format!(
            "connect|spark/4.3.0|u1|srv-1|active|{catalog}|{namespace}|etl"
        )]
    );
}

#[test]
fn a_principal_sees_its_own_sessions_and_monitor_sees_all() {
    let rt = EngineRuntime::builder().build().unwrap();
    let a = session_of(&rt, "alice");
    let b = session_of(&rt, "bob");
    let admin = session_of(&rt, "admin");
    b.close();
    let q = "SELECT principal FROM system.sessions ORDER BY principal";
    assert_eq!(texts(&a, q), ["alice"]);
    // Bob's closed session is hidden from alice too.
    assert!(!texts(&a, q).contains(&"bob".to_owned()));
    assert_eq!(texts(&admin, q), ["admin", "alice", "bob"]);
}

#[test]
fn the_lifecycle_states_and_reasons() {
    let rt = EngineRuntime::builder().build().unwrap();
    let admin = session_of(&rt, "admin");
    let released = session_of(&rt, "p1");
    let expired = session_of(&rt, "p2");
    let killed = session_of(&rt, "p3");
    let (e, k) = (id_of(&admin, "p2"), id_of(&admin, "p3"));
    released.close();
    assert_eq!(rt.expire_session(&e), Some(0));
    assert_eq!(rt.kill_session(&k, CancelReason::Admin), Some(0));
    assert_eq!(rt.kill_session(&k, CancelReason::Admin), None);
    assert!(rt.session_ended(&k));
    assert!(!rt.session_ended("never-seen"));
    let rows = texts(
        &admin,
        "SELECT concat_ws('|', principal, state, close_reason) FROM system.sessions \
         WHERE principal LIKE 'p%' ORDER BY principal",
    );
    assert_eq!(
        rows,
        [
            "p1|closed|release_session",
            "p2|expired|idle_timeout",
            "p3|killed|admin_kill"
        ]
    );
    // A killed or expired session takes no new statement.
    for s in [&expired, &killed] {
        let e = block_on(s.sql("SELECT 1")).unwrap_err();
        assert_eq!(e.error_class().name(), "INVALID_HANDLE.SESSION_CLOSED");
    }
}

#[test]
fn a_running_query_is_counted_and_cancelled_by_kill() {
    let rt = EngineRuntime::builder().build().unwrap();
    let admin = session_of(&rt, "admin");
    let victim = session_of(&rt, "victim");
    let id = id_of(&admin, "victim");
    let df = block_on(victim.sql("SELECT id FROM range(0, 2000000000)")).unwrap();
    let stream = block_on(df.execute_stream()).unwrap();
    assert_eq!(
        texts(
            &admin,
            "SELECT CAST(queries_started AS STRING) FROM system.sessions WHERE principal = 'victim'"
        ),
        ["1"]
    );
    assert_eq!(rt.kill_session(&id, CancelReason::Admin), Some(1));
    // A second kill finds the session ended.
    assert_eq!(rt.kill_session(&id, CancelReason::Admin), None);
    assert_eq!(rt.expire_session(&id), None);
    // The query itself ends with the admin's reason, not merely with its dropped stream.
    let mut stream = stream;
    let err = block_on(async {
        use futures::StreamExt as _;
        loop {
            match stream.next().await {
                Some(Ok(_)) => {}
                Some(Err(e)) => break Some(e),
                None => break None,
            }
        }
    })
    .expect("the killed query ends with an error");
    assert_eq!(err.error_class().name(), "KRISHIV_QUERY_KILLED");
}

#[test]
fn dropped_sessions_leave_a_bounded_ring() {
    let rt = EngineRuntime::builder().build().unwrap();
    let admin = session_of(&rt, "admin");
    for _ in 0..1100 {
        session_of(&rt, "churn").close();
    }
    let n = texts(
        &admin,
        "SELECT CAST(count(*) AS STRING) FROM system.sessions WHERE principal = 'churn'",
    );
    assert_eq!(n, ["1000"]);
}

#[test]
fn a_dropped_handle_is_remembered_as_client_gone() {
    let rt = EngineRuntime::builder().build().unwrap();
    let admin = session_of(&rt, "admin");
    drop(session_of(&rt, "gone"));
    assert_eq!(
        texts(
            &admin,
            "SELECT concat_ws('|', state, close_reason) FROM system.sessions WHERE principal = 'gone'"
        ),
        ["closed|client_gone"]
    );
}

#[test]
fn queries_running_counts_live_queries_only_and_activity_moves() {
    let rt = EngineRuntime::builder().build().unwrap();
    let admin = session_of(&rt, "admin");
    let s = session_of(&rt, "runner");
    let counts = "SELECT concat_ws('|', CAST(queries_running AS STRING), CAST(queries_started AS STRING), \
                  CAST(idle_timeout_ms AS STRING)) FROM system.sessions WHERE principal = 'runner'";
    let active = "SELECT CAST(unix_micros(last_active_at) AS STRING) FROM system.sessions \
                  WHERE principal = 'runner'";
    s.set_idle_timeout_ms(Some(60_000));
    let df = block_on(s.sql("SELECT id FROM range(0, 2000000000)")).unwrap();
    let stream = block_on(df.execute_stream()).unwrap();
    assert_eq!(texts(&admin, counts), ["1|1|60000"]);
    drop(stream);
    let before: i64 = texts(&admin, active)[0].parse().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(20));
    block_on(block_on(s.sql("SELECT 1")).unwrap().collect()).unwrap();
    let after: i64 = texts(&admin, active)[0].parse().unwrap();
    assert!(
        after > before,
        "last_active_at did not move: {before} -> {after}"
    );
    assert_eq!(texts(&admin, counts), ["0|2|60000"]);
}

#[test]
fn client_strings_are_capped_at_a_character_boundary() {
    let rt = EngineRuntime::builder().build().unwrap();
    let s = session_of(&rt, "capper");
    let long = "\u{e9}".repeat(400);
    s.set_client_info(Transport::Ui, Some(&long), None, None);
    let n = texts(
        &s,
        "SELECT CAST(length(client_type) AS STRING) FROM system.sessions",
    );
    assert_eq!(n, ["128"]);
}

#[test]
fn a_callers_own_session_kill_is_recorded_as_a_release() {
    let rt = EngineRuntime::builder().build().unwrap();
    let admin = session_of(&rt, "admin");
    let _s = session_of(&rt, "me");
    let id = id_of(&admin, "me");
    assert_eq!(rt.kill_session(&id, CancelReason::ClientGone), Some(0));
    assert_eq!(
        texts(
            &admin,
            "SELECT concat_ws('|', state, close_reason) FROM system.sessions WHERE principal = 'me'"
        ),
        ["closed|release_session"]
    );
}

#[test]
fn equal_names_in_two_tenants_do_not_see_each_other() {
    let rt = EngineRuntime::builder().build().unwrap();
    let mk = |tenant: &str| {
        let mut p = Principal::new("pat", PrincipalKind::User);
        p.tenant = Some(tenant.to_owned());
        Session::builder()
            .runtime(Arc::clone(&rt))
            .principal(p)
            .access_policy(Arc::new(MonitorForAdmin))
            .build()
            .unwrap()
    };
    let (a, _b) = (mk("acme"), mk("globex"));
    assert_eq!(
        texts(&a, "SELECT CAST(count(*) AS STRING) FROM system.sessions"),
        ["1"]
    );
}

/// An impersonated session shows the end user as its principal and the service that acts for
/// it in `delegated_by`, in `system.sessions` and in every row it adds to `system.queries`
/// (P3-T061 §4.3, amendment A5); a session of its own shows NULL there. Own rows are keyed by
/// the service too (amendment A11): a service acting for `spark` never sees the rows of the
/// `spark` principal that logged in itself, nor the other way round.
#[test]
fn delegated_sessions_show_the_delegator() {
    let rt = EngineRuntime::builder().build().unwrap();
    let via = Session::builder()
        .runtime(Arc::clone(&rt))
        .principal(Principal::new("spark", PrincipalKind::User).with_delegator("portal"))
        .access_policy(Arc::new(MonitorForAdmin))
        .build()
        .unwrap();
    let own = session_of(&rt, "spark");
    let admin = session_of(&rt, "admin");
    block_on(block_on(via.sql("SELECT 1")).unwrap().collect()).unwrap();
    block_on(block_on(own.sql("SELECT 2")).unwrap().collect()).unwrap();

    let sessions = column(
        &admin,
        "SELECT coalesce(delegated_by, '-') FROM system.sessions WHERE principal = 'spark' \
         ORDER BY delegated_by NULLS LAST",
    );
    assert_eq!(sessions, [Some("portal".to_owned()), Some("-".to_owned())]);
    let queries = column(
        &admin,
        "SELECT concat(query_text, '|', coalesce(delegated_by, '-')) FROM system.queries \
         WHERE query_text IN ('SELECT 1', 'SELECT 2') ORDER BY query_text",
    );
    assert_eq!(
        queries,
        [
            Some("SELECT 1|portal".to_owned()),
            Some("SELECT 2|-".to_owned())
        ]
    );

    for (s, mine) in [(&own, "-"), (&via, "portal")] {
        let sessions = column(s, "SELECT coalesce(delegated_by, '-') FROM system.sessions");
        assert_eq!(
            sessions,
            [Some(mine.to_owned())],
            "{mine}: own sessions only"
        );
        let queries = column(
            s,
            "SELECT coalesce(delegated_by, '-') FROM system.queries \
             WHERE query_text IN ('SELECT 1', 'SELECT 2')",
        );
        assert_eq!(queries, [Some(mine.to_owned())], "{mine}: own queries only");
    }
}

#[test]
fn a_draining_runtime_refuses_new_queries_but_runs_system_tables_and_cancels_the_rest() {
    let rt = EngineRuntime::builder().build().unwrap();
    let s = session_of(&rt, "alice");
    let df = block_on(s.sql("SELECT id FROM range(0, 2000000000)")).unwrap();
    let running = block_on(df.execute_stream()).unwrap();
    rt.set_draining(true);
    assert!(rt.is_draining());
    // A new query is refused with the drain class; one that reads only `system.*` runs.
    let df = block_on(s.sql("SELECT 1")).unwrap();
    let err = block_on(df.collect()).unwrap_err();
    assert_eq!(err.error_class().name(), "KRISHIV_SERVER_DRAINING");
    assert_eq!(texts(&s, "SELECT state FROM system.sessions").len(), 1);
    // The refused query's row is closed as failed with the drain class, not left live.
    assert_eq!(
        texts(
            &s,
            "SELECT error_class FROM system.queries WHERE state = 'failed' \
             AND error_class = 'KRISHIV_SERVER_DRAINING'"
        ),
        ["KRISHIV_SERVER_DRAINING"]
    );
    // The deadline cancels what is left, with the drain reason.
    assert!(rt.cancel_all(CancelReason::Drain) >= 1);
    drop(running);
    rt.set_draining(false);
    assert_eq!(texts(&s, "SELECT CAST(1 AS STRING)"), ["1"]);
}

/// A stored result is not replayed while the server drains: the drain refuses new work, and a
/// cache hit would be new work served (design P3-T066 §5.1). Serving resumes after `ResumeServing`.
#[test]
fn a_draining_runtime_does_not_replay_a_cached_result() {
    let rt = EngineRuntime::builder().build().unwrap();
    let s = session_of(&rt, "alice");
    s.conf().set("krishiv.resultCache.enabled", "true").unwrap();
    let sql = "SELECT 41 + 1 AS drain_cache_probe";
    for _ in 0..2 {
        block_on(async { s.sql(sql).await.unwrap().collect().await.unwrap() });
    }
    rt.set_draining(true);
    let df = block_on(s.sql(sql)).unwrap();
    let err = block_on(df.collect()).unwrap_err();
    assert_eq!(err.error_class().name(), "KRISHIV_SERVER_DRAINING");
    rt.set_draining(false);
    assert_eq!(texts(&s, "SELECT CAST(41 + 1 AS STRING)"), ["42"]);
    block_on(async { s.sql(sql).await.unwrap().collect().await.unwrap() });
}

/// The drain deadline counts what it cancels by (session, operation id): two sessions may run a
/// query under the same client-chosen operation id, and both end (design P3-T066 §5.1).
#[test]
fn cancel_all_counts_the_same_operation_id_in_two_sessions_twice() {
    const OP: &str = "00112233-4455-6677-8899-aabbccddeeff";
    let rt = EngineRuntime::builder().build().unwrap();
    let mut running = Vec::new();
    for user in ["alice", "bob"] {
        let s = session_of(&rt, user);
        let df = block_on(s.sql_with(
            "SELECT id FROM range(0, 2000000000)",
            krishiv_session::QueryOptions::new().operation_id(OP),
        ))
        .unwrap();
        running.push((s, block_on(df.execute_stream()).unwrap()));
    }
    assert_eq!(rt.cancel_all(CancelReason::Drain), 2);
    drop(running);
}

/// `DROP TABLE` is refused while the server drains before it touches the catalog: the table is
/// still there after `ResumeServing`.
#[test]
fn a_draining_runtime_refuses_drop_table_and_leaves_the_table() {
    let rt = EngineRuntime::builder().build().unwrap();
    let s = session_of(&rt, "alice");
    block_on(async {
        s.sql("CREATE TABLE drain_drop_t (a INT)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
    });
    rt.set_draining(true);
    let Err(err) = block_on(s.sql("DROP TABLE drain_drop_t")) else {
        panic!("DROP TABLE must be refused while draining");
    };
    assert_eq!(err.error_class().name(), "KRISHIV_SERVER_DRAINING");
    rt.set_draining(false);
    assert_eq!(
        texts(&s, "SELECT CAST(count(*) AS STRING) FROM drain_drop_t"),
        ["0"]
    );
}

/// A bare `DROP VIEW` of a stored view is the qualified drop (DDL): refused while draining, the view
/// kept; a bare drop of a temporary view stays session-local and is admitted.
#[test]
fn a_draining_runtime_refuses_a_bare_drop_of_a_stored_view_and_keeps_it() {
    let rt = EngineRuntime::builder().build().unwrap();
    let s = session_of(&rt, "alice");
    let run = |sql: &str| block_on(async { s.sql(sql).await.unwrap().collect().await.unwrap() });
    run("CREATE VIEW drain_bare_v AS SELECT 1 AS a");
    run("CREATE TEMPORARY VIEW drain_bare_tv AS SELECT 2 AS a");
    rt.set_draining(true);
    let Err(err) = block_on(s.sql("DROP VIEW drain_bare_v")) else {
        panic!("a bare DROP VIEW of a stored view must be refused while draining");
    };
    assert_eq!(err.error_class().name(), "KRISHIV_SERVER_DRAINING");
    // The command runs in `sql`; its result frame is a new query, which the drain refuses.
    assert!(block_on(s.sql("DROP VIEW drain_bare_tv")).is_ok());
    rt.set_draining(false);
    assert_eq!(
        texts(&s, "SELECT CAST(a AS STRING) FROM drain_bare_v"),
        ["1"]
    );
    let Err(gone) = block_on(s.sql("SELECT a FROM drain_bare_tv")) else {
        panic!("the temporary view must have been dropped");
    };
    assert_eq!(gone.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
}

/// Every statement that writes, creates or drops is refused while the server drains, before its
/// handler changes anything, and leaves a failed `system.queries` row as a refused query does
/// (design P3-T066 §5.1).
#[test]
fn a_draining_runtime_refuses_every_changing_statement_before_it_changes_anything() {
    let rt = EngineRuntime::builder().build().unwrap();
    let s = session_of(&rt, "alice");
    let run = |sql: &str| block_on(async { s.sql(sql).await.unwrap().collect().await.unwrap() });
    run("CREATE TABLE drain_all_t (a INT)");
    run("INSERT INTO drain_all_t VALUES (1), (2)");
    run("CREATE NAMESPACE drain_all_ns");
    rt.set_draining(true);
    for sql in [
        "TRUNCATE TABLE drain_all_t",
        "ALTER TABLE drain_all_t RENAME TO drain_all_t2",
        "DROP NAMESPACE drain_all_ns CASCADE",
        "CREATE NAMESPACE drain_all_ns2",
        "CREATE TABLE drain_all_new (a INT)",
        "INSERT INTO drain_all_t VALUES (3)",
        "DROP TABLE drain_all_t",
    ] {
        let Err(err) = block_on(s.sql(sql)) else {
            panic!("{sql} must be refused while draining");
        };
        assert_eq!(err.error_class().name(), "KRISHIV_SERVER_DRAINING", "{sql}");
    }
    // A read of `system.*` still runs while draining.
    assert_eq!(texts(&s, "SELECT state FROM system.sessions").len(), 1);
    assert_eq!(
        texts(
            &s,
            "SELECT CAST(count(*) AS STRING) FROM system.queries WHERE state = 'failed' \
             AND error_class = 'KRISHIV_SERVER_DRAINING' AND query_text LIKE 'TRUNCATE%'"
        ),
        ["1"]
    );
    rt.set_draining(false);
    assert_eq!(
        texts(&s, "SELECT CAST(count(*) AS STRING) FROM drain_all_t"),
        ["2"]
    );
    run("DROP NAMESPACE drain_all_ns CASCADE");
}
