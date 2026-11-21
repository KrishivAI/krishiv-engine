// SPDX-License-Identifier: Apache-2.0
//! The editor's per-caller sessions (P3-T050, ADR 0046): a role change gets the new role's policy, a static token
//! and an OIDC subject of the same name are different callers, the run limits answer 429, and a path outside the
//! allowed roots is refused. Drives `SessionSource` through `UiSource` with hand-made callers.
#![cfg(feature = "session")]
#![expect(
    clippy::unwrap_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use krishiv_common::config::Config;
use krishiv_common::principal::{Principal, PrincipalKind};
use krishiv_server::guard::Caller;
use krishiv_server::internal::{InboundTrace, UiSource};
use krishiv_server::session_source::SessionSource;
use krishiv_session::auth::{Authenticated, Method, Role, RolePolicy};
use krishiv_session::{EngineRuntime, PathPrefixPolicy, PolicyVersion, Session};
use serde_json::{Value, json};

const SLOW: &str = "SELECT count(*) FROM range(0, 1000000000) a CROSS JOIN range(0, 1000000000) b";

fn caller(name: &str, role: Role, method: Method) -> Caller {
    let who = Authenticated::new(
        Principal::new(name, PrincipalKind::User),
        Some(role),
        method,
    );
    Caller {
        principal: name.to_owned(),
        role: Some(role.as_str().to_owned()),
        method: "token".to_owned(),
        identity: Some(Arc::new(who)),
    }
}

fn source(allowed: Option<&std::path::Path>) -> SessionSource {
    let runtime = EngineRuntime::builder().build().unwrap();
    let base = Session::builder()
        .runtime(Arc::clone(&runtime))
        .build()
        .unwrap();
    let mut config = Config::new();
    if let Some(dir) = allowed {
        config
            .set("krishiv.io.local.allowedRoots", dir.to_str().unwrap())
            .unwrap();
    }
    let factory = move |who: &Authenticated| {
        let roles = RolePolicy::new(who.role, Role::Admin, PolicyVersion(0));
        let policy = PathPrefixPolicy::from_config(&config, Arc::new(roles))?;
        Ok(base.new_session_as(who.principal.clone(), Arc::new(policy)))
    };
    SessionSource::new(runtime)
        .unwrap()
        .with_caller_sessions(Arc::new(factory))
}

async fn finish(s: &SessionSource, c: &Caller, sql: &str) -> Value {
    let id = s.sql_start(sql, Some(c)).await.unwrap();
    let start = Instant::now();
    loop {
        let v = s.sql_status(&id, Some(c)).await.unwrap().unwrap();
        if v["state"] != "running" {
            return v;
        }
        assert!(start.elapsed() < Duration::from_secs(30), "{v}");
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
}

#[tokio::test]
async fn a_caller_whose_role_changed_gets_a_session_with_the_new_policy() {
    let s = source(None);
    let admin = caller("ann", Role::Admin, Method::Oidc);
    let create = "CREATE TABLE t50_role (a INT) USING parquet";
    let before = finish(&s, &admin, "DROP TABLE IF EXISTS t50_role").await;
    assert_eq!(before["state"], "finished", "an admin may drop: {before}");
    // The same OIDC subject, now an operator: the admin session is not reused.
    let operator = caller("ann", Role::Operator, Method::Oidc);
    let after = finish(&s, &operator, create).await;
    assert_eq!(after["state"], "failed", "{after}");
    assert_eq!(
        after["error_class"], "KRISHIV_INSUFFICIENT_PERMISSIONS",
        "{after}"
    );
    let drop = finish(&s, &operator, "DROP TABLE IF EXISTS t50_role").await;
    assert_eq!(
        drop["error_class"], "KRISHIV_INSUFFICIENT_PERMISSIONS",
        "{drop}"
    );
}

#[tokio::test]
async fn a_static_token_and_an_oidc_subject_of_one_name_do_not_share_runs() {
    let s = source(None);
    let token_ann = caller("ann", Role::Operator, Method::Token);
    let oidc_ann = caller("ann", Role::Operator, Method::Oidc);
    let id = s.sql_start(SLOW, Some(&token_ann)).await.unwrap();
    assert!(s.sql_status(&id, Some(&token_ann)).await.unwrap().is_some());
    assert!(s.sql_status(&id, Some(&oidc_ann)).await.unwrap().is_none());
    assert!(!s.sql_cancel(&id, Some(&oidc_ann)).await.unwrap());
    assert!(s.sql_cancel(&id, Some(&token_ann)).await.unwrap());
}

#[tokio::test]
async fn too_many_running_statements_are_refused_with_busy() {
    let s = source(None);
    let ann = caller("ann", Role::Operator, Method::Token);
    let mut ids = Vec::new();
    for _ in 0..4 {
        ids.push(s.sql_start(SLOW, Some(&ann)).await.unwrap());
    }
    let fifth = s.sql_start(SLOW, Some(&ann)).await.unwrap_err();
    assert!(fifth.busy, "{fifth}");
    // Another caller still has room.
    let bo = caller("bo", Role::Operator, Method::Token);
    let other = s.sql_start(SLOW, Some(&bo)).await.unwrap();
    ids.push(other);
    for id in &ids[..4] {
        s.sql_cancel(id, Some(&ann)).await.unwrap();
    }
    s.sql_cancel(&ids[4], Some(&bo)).await.unwrap();
}

#[tokio::test]
async fn when_every_slot_is_running_a_new_statement_is_refused_and_none_is_forgotten() {
    let s = source(None);
    let mut started = Vec::new();
    for i in 0..32 {
        let c = caller(&format!("u{i}"), Role::Operator, Method::Token);
        let id = s.sql_start(SLOW, Some(&c)).await.unwrap();
        started.push((c, id));
    }
    let late = caller("late", Role::Operator, Method::Token);
    let refused = s.sql_start("SELECT 1", Some(&late)).await.unwrap_err();
    assert!(refused.busy, "{refused}");
    // Every run is still known, and cancellable.
    for (c, id) in &started {
        assert!(s.sql_cancel(id, Some(c)).await.unwrap(), "{id}");
    }
}

#[tokio::test]
async fn a_path_outside_the_allowed_roots_is_refused_through_the_editor() {
    let tmp = tempfile::tempdir().unwrap();
    let allowed = tmp.path().join("allowed");
    let secret = tmp.path().join("secret");
    std::fs::create_dir(&allowed).unwrap();
    std::fs::create_dir(&secret).unwrap();
    std::fs::write(allowed.join("a.csv"), "1\n2\n").unwrap();
    std::fs::write(secret.join("s.csv"), "3\n").unwrap();
    let s = source(Some(&allowed));
    let op = caller("ann", Role::Operator, Method::Token);
    let outside = finish(
        &s,
        &op,
        &format!("SELECT * FROM csv.`{}`", secret.join("s.csv").display()),
    )
    .await;
    assert_eq!(outside["state"], "failed", "{outside}");
    assert_eq!(
        outside["error_class"], "KRISHIV_INSUFFICIENT_PERMISSIONS",
        "{outside}"
    );
    let inside = finish(
        &s,
        &op,
        &format!("SELECT * FROM csv.`{}`", allowed.join("a.csv").display()),
    )
    .await;
    assert_eq!(inside["state"], "finished", "{inside}");
    // Spark's CSV reader without a header: one string column `_c0`, the file's lines in order.
    assert_eq!(inside["columns"], json!(["_c0"]), "{inside}");
    assert_eq!(inside["rows"], json!([["1"], ["2"]]), "{inside}");
}

/// P3-T068: the HTTP request's `traceparent` names the trace of the statement it starts (`system.queries.trace_id`);
/// a malformed header or none means no trace and does not fail the run.
#[tokio::test]
async fn a_requests_traceparent_names_the_trace_of_the_statement_it_starts() {
    let s = source(None);
    let admin = caller("ann", Role::Admin, Method::Oidc);
    let trace = "4bf92f3577b34da6a3ce929d0e0e4736";
    let traced = InboundTrace {
        traceparent: Some(format!("00-{trace}-00f067aa0ba902b7-01")),
        baggage: Some("tenant=acme".to_owned()),
    };
    let malformed = InboundTrace {
        traceparent: Some("not-a-traceparent".to_owned()),
        baggage: None,
    };
    for (sql, t) in [
        ("SELECT 41 AS t68_traced", &traced),
        ("SELECT 42 AS t68_malformed", &malformed),
        ("SELECT 43 AS t68_plain", &InboundTrace::default()),
    ] {
        let id = s.sql_start_traced(sql, Some(&admin), t).await.unwrap();
        let start = Instant::now();
        loop {
            let v = s.sql_status(&id, Some(&admin)).await.unwrap().unwrap();
            if v["state"] != "running" {
                assert_eq!(v["state"], "finished", "{v}");
                break;
            }
            assert!(start.elapsed() < Duration::from_secs(30), "{v}");
            tokio::time::sleep(Duration::from_millis(30)).await;
        }
    }
    // The text of another caller's query is not shown, so count by trace: only the traced statement has one.
    let seen = finish(
        &s,
        &admin,
        "SELECT trace_id FROM system.queries WHERE trace_id IS NOT NULL",
    )
    .await;
    assert_eq!(seen["rows"], json!([[trace]]), "{seen}");
    let all = finish(&s, &admin, "SELECT count(*) FROM system.queries").await;
    assert!(all["rows"][0][0].as_i64().unwrap() >= 3, "{all}");
}
