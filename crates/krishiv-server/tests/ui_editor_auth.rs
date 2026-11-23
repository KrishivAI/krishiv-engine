// SPDX-License-Identifier: Apache-2.0
//! The UI's SQL editor with authentication on (P3-T050, ADR 0046): a statement runs as the authenticated caller
//! under that caller's policy, never as a principal of the UI's own, with the real `SessionGuard`.
#![cfg(all(unix, feature = "session", feature = "metrics-prometheus"))]
#![expect(
    clippy::unwrap_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::net::SocketAddr;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use krishiv_server::admin::{HttpLimits, serve_capped, with_limits};
use krishiv_server::guard::Guard;
use krishiv_server::internal::router_with_ui_guard;
use krishiv_server::session_guard::SessionGuard;
use krishiv_server::session_source::SessionSource;
use krishiv_session::auth::{AuthConfig, Authenticated, Role, RolePolicy, ServerAuth};
use krishiv_session::{EngineRuntime, PathPrefixPolicy, PolicyVersion, Session};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const VIEWER: &str = "viewer-token-0123456789abcdef0123456789";
const ANN: &str = "ann-token-0123456789abcdef0123456789abc";
const BO: &str = "bo-token-0123456789abcdef0123456789abcd";

fn secret(dir: &std::path::Path, name: &str, value: &str) -> String {
    let path = dir.join(name);
    std::fs::write(&path, value).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    format!("file:{}", path.display())
}

async fn start() -> SocketAddr {
    let dir = tempfile::tempdir().unwrap();
    let (v, a, b) = (
        secret(dir.path(), "v", VIEWER),
        secret(dir.path(), "a", ANN),
        secret(dir.path(), "b", BO),
    );
    let settings = [
        ("krishiv.auth.tokens.v.secret", v.as_str()),
        ("krishiv.auth.tokens.v.principal", "vera"),
        ("krishiv.auth.tokens.v.role", "viewer"),
        ("krishiv.auth.tokens.a.secret", a.as_str()),
        ("krishiv.auth.tokens.a.principal", "ann"),
        ("krishiv.auth.tokens.a.role", "operator"),
        ("krishiv.auth.tokens.b.secret", b.as_str()),
        ("krishiv.auth.tokens.b.principal", "bo"),
        ("krishiv.auth.tokens.b.role", "operator"),
    ];
    let auth = ServerAuth::new(&AuthConfig::from_settings(settings).unwrap()).unwrap();
    let guard: Arc<dyn Guard> = Arc::new(SessionGuard::new(Arc::new(auth)));
    let runtime = EngineRuntime::builder().build().unwrap();
    let base = Session::builder()
        .runtime(Arc::clone(&runtime))
        .build()
        .unwrap();
    let config = krishiv_common::config::Config::new();
    // What `krishiv server` builds with `krishiv_connect_server::role_sessions`.
    let factory = move |who: &Authenticated| {
        let roles = RolePolicy::new(who.role, Role::Admin, PolicyVersion(0));
        let policy = PathPrefixPolicy::from_config(&config, Arc::new(roles))?;
        Ok(base.new_session_as(who.principal.clone(), Arc::new(policy)))
    };
    let source = SessionSource::new(runtime)
        .unwrap()
        .with_caller_sessions(Arc::new(factory));
    let app = router_with_ui_guard(Arc::new(source), guard);
    let limits = HttpLimits::default();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(serve_capped(
        listener,
        with_limits(app, limits),
        limits.max_connections,
        std::future::pending(),
    ));
    addr
}

async fn call(addr: SocketAddr, method: &str, path: &str, token: &str, body: &str) -> (u16, Value) {
    call_with(addr, method, path, token, body, "").await
}

/// `call` with extra raw header lines (each ending in `\r\n`).
async fn call_with(
    addr: SocketAddr,
    method: &str,
    path: &str,
    token: &str,
    body: &str,
    extra: &str,
) -> (u16, Value) {
    let mut s = TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {token}\r\nX-Krishiv-Csrf: 1\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut out = Vec::new();
    tokio::time::timeout(Duration::from_secs(20), s.read_to_end(&mut out))
        .await
        .unwrap()
        .unwrap();
    let text = String::from_utf8(out).unwrap();
    let (head, body) = text.split_once("\r\n\r\n").unwrap();
    (
        head.split_whitespace().nth(1).unwrap().parse().unwrap(),
        serde_json::from_str(body).unwrap_or(Value::Null),
    )
}

async fn run(addr: SocketAddr, token: &str, sql: &str) -> Value {
    run_with(addr, token, sql, "").await
}

async fn run_with(addr: SocketAddr, token: &str, sql: &str, extra: &str) -> Value {
    let (status, body) = call_with(
        addr,
        "POST",
        "/api/internal/sql",
        token,
        &json!({"sql": sql}).to_string(),
        extra,
    )
    .await;
    assert_eq!(status, 202, "{body}");
    let id = body["run_id"].as_str().unwrap().to_owned();
    let start = Instant::now();
    loop {
        let (_, v) = call(addr, "GET", &format!("/api/internal/sql/{id}"), token, "").await;
        if v["state"] != "running" {
            return v;
        }
        assert!(start.elapsed() < Duration::from_secs(30), "{v}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn an_operators_statement_runs_as_the_operator_and_is_recorded_under_them() {
    let addr = start().await;
    let done = run(addr, ANN, "SELECT 41 + 1 AS answer").await;
    assert_eq!(done["state"], "finished", "{done}");
    assert_eq!(done["columns"], json!(["answer"]), "{done}");
    assert_eq!(done["rows"], json!([[42]]), "{done}");
    let (_, history) = call(addr, "GET", "/api/internal/history", ANN, "").await;
    let row = history["queries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|q| q["principal"] == "ann" && q["rows_produced"] == 1)
        .unwrap_or_else(|| panic!("{history}"));
    assert_eq!(
        row["principal"], "ann",
        "the query belongs to the caller: {row}"
    );
}

#[tokio::test]
async fn an_operator_cannot_use_the_editor_to_do_what_only_an_admin_may() {
    let addr = start().await;
    for sql in [
        "CREATE TABLE t50_denied (a INT) USING parquet",
        "DROP TABLE IF EXISTS t50_denied",
        "INSERT INTO t50_denied VALUES (1)",
    ] {
        let r = run(addr, ANN, sql).await;
        assert_eq!(r["state"], "failed", "{sql}: {r}");
        assert_eq!(
            r["error_class"], "KRISHIV_INSUFFICIENT_PERMISSIONS",
            "{sql}: {r}"
        );
    }
}

#[tokio::test]
async fn a_run_belongs_to_its_caller_and_a_viewer_cannot_start_one() {
    let addr = start().await;
    let (_, body) = call(
        addr,
        "POST",
        "/api/internal/sql",
        ANN,
        &json!({"sql": "SELECT 1"}).to_string(),
    )
    .await;
    let id = body["run_id"].as_str().unwrap();
    let (own, _) = call(addr, "GET", &format!("/api/internal/sql/{id}"), ANN, "").await;
    assert_eq!(own, 200);
    // Another operator does not see the run, and cannot cancel it.
    let (other, _) = call(addr, "GET", &format!("/api/internal/sql/{id}"), BO, "").await;
    assert_eq!(other, 404);
    let (other, _) = call(
        addr,
        "POST",
        &format!("/api/internal/sql/{id}/cancel"),
        BO,
        "",
    )
    .await;
    assert_eq!(other, 404);
    // A viewer has no access to the editor, or to the pages.
    let (viewer, _) = call(
        addr,
        "POST",
        "/api/internal/sql",
        VIEWER,
        &json!({"sql": "SELECT 1"}).to_string(),
    )
    .await;
    assert_eq!(viewer, 403);
    let (viewer, _) = call(addr, "POST", "/api/internal/queries/x/cancel", VIEWER, "").await;
    assert_eq!(viewer, 403);
}

#[tokio::test]
async fn an_operators_own_cancel_ends_their_query_as_a_user_cancel() {
    let addr = start().await;
    let sql = "SELECT count(*) FROM range(0, 1000000000) a CROSS JOIN range(0, 1000000000) b";
    let (status, body) = call(
        addr,
        "POST",
        "/api/internal/sql",
        ANN,
        &json!({"sql": sql}).to_string(),
    )
    .await;
    assert_eq!(status, 202, "{body}");
    let id = body["run_id"].as_str().unwrap().to_owned();
    // Wait until the engine has registered the tagged query (a cancel earlier than that takes the editor's
    // pre-execution path and would not test `cancel_by_tag`).
    let registered = Instant::now();
    loop {
        let (_, q) = call(addr, "GET", "/api/internal/queries", ANN, "").await;
        let live = q["queries"].as_array().is_some_and(|rows| {
            rows.iter()
                .any(|r| r["principal"] == "ann" && r["state"] == "running")
        });
        if live {
            break;
        }
        assert!(registered.elapsed() < Duration::from_secs(20), "{q}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let (status, _) = call(
        addr,
        "POST",
        &format!("/api/internal/sql/{id}/cancel"),
        ANN,
        "",
    )
    .await;
    assert_eq!(status, 200);
    let start = Instant::now();
    let done = loop {
        let (_, v) = call(addr, "GET", &format!("/api/internal/sql/{id}"), ANN, "").await;
        if v["state"] != "running" {
            break v;
        }
        assert!(start.elapsed() < Duration::from_secs(2), "{v}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert_eq!(done["state"], "cancelled", "{done}");
    assert_eq!(done["error_class"], "OPERATION_CANCELED", "{done}");
}

/// P3-T068: `POST /api/internal/sql` reads the request's `traceparent`; a malformed one is no trace and the run
/// still finishes.
#[tokio::test]
async fn the_sql_route_carries_the_requests_traceparent_into_the_statements_trace() {
    let addr = start().await;
    let trace = "4bf92f3577b34da6a3ce929d0e0e4736";
    let traced = run_with(
        addr,
        ANN,
        "SELECT 41 AS t68_traced",
        &format!("traceparent: 00-{trace}-00f067aa0ba902b7-01\r\nbaggage: tenant=acme\r\n"),
    )
    .await;
    assert_eq!(traced["state"], "finished", "{traced}");
    let bad = run_with(
        addr,
        ANN,
        "SELECT 42 AS t68_malformed",
        "traceparent: not-a-traceparent\r\n",
    )
    .await;
    assert_eq!(bad["state"], "finished", "{bad}");
    let seen = run(
        addr,
        ANN,
        "SELECT trace_id FROM system.queries WHERE trace_id IS NOT NULL",
    )
    .await;
    assert_eq!(seen["rows"], json!([[trace]]), "{seen}");
}
