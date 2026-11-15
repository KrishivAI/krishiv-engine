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
