// SPDX-License-Identifier: Apache-2.0
//! The T050 page endpoints over a real engine runtime on a real socket: catalog browser, environment with
//! secrets masked (REST and event stream), the SQL editor's run and cancel, the UI session cookie, the host and
//! origin checks, and a viewer who cannot cancel.
#![cfg(feature = "session")]
#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::BoxFuture;
use krishiv_server::guard::{Access, Guard, GuardRequest, Refusal, TokenGuard};
use krishiv_server::internal::{router_with_ui, router_with_ui_guard, router_with_ui_policy};
use krishiv_server::pages::UiPolicy;
use krishiv_server::session_source::SessionSource;
use krishiv_session::EngineRuntime;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const TOKEN: &str = "page-token";
const CANARIES: [&str; 3] = ["tok-canary-91", "pw-canary-92", "s3-canary-93"];

async fn serve(app: axum::Router) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await });
    addr
}

async fn start() -> std::net::SocketAddr {
    let rt = EngineRuntime::builder()
        .config("my.app.token", CANARIES[0])
        .config("my.app.password", CANARIES[1])
        .config("spark.hadoop.fs.s3a.secret.key", CANARIES[2])
        .build()
        .unwrap();
    let source = SessionSource::new(rt).unwrap();
    serve(router_with_ui(Arc::new(source), TOKEN)).await
}

/// One request: `headers` are raw lines; returns status, lower-cased head and body text.
async fn raw(
    addr: std::net::SocketAddr,
    method: &str,
    path: &str,
    headers: &[&str],
    body: &str,
) -> (u16, String, String) {
    let mut s = TcpStream::connect(addr).await.unwrap();
    let extra: String = headers.iter().map(|h| format!("{h}\r\n")).collect();
    let req = format!(
        "{method} {path} HTTP/1.1\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut out = Vec::new();
    tokio::time::timeout(Duration::from_secs(20), s.read_to_end(&mut out))
        .await
        .expect("read timed out")
        .unwrap();
    let text = String::from_utf8(out).unwrap();
    let (head, body) = text.split_once("\r\n\r\n").unwrap();
    (
        head.split_whitespace().nth(1).unwrap().parse().unwrap(),
        head.to_lowercase(),
        body.to_owned(),
    )
}

fn auth() -> String {
    format!("Authorization: Bearer {TOKEN}")
}

async fn get(addr: std::net::SocketAddr, path: &str) -> Value {
    let a = auth();
    let (status, _, body) = raw(addr, "GET", path, &["Host: localhost", &a], "").await;
    assert_eq!(status, 200, "{path}: {body}");
    serde_json::from_str(&body).unwrap()
}

async fn post(addr: std::net::SocketAddr, path: &str, body: &str) -> (u16, Value) {
    let a = auth();
    let (status, _, text) = raw(
        addr,
        "POST",
        path,
        &["Host: localhost", &a, "X-Krishiv-Csrf: 1"],
        body,
    )
    .await;
    (status, serde_json::from_str(&text).unwrap_or(Value::Null))
}

#[tokio::test]
async fn the_catalog_browser_lists_catalogs_schemas_tables_and_columns() {
    let addr = start().await;
    let (status, _) = post(
        addr,
        "/api/internal/sql",
        &json!({"sql": "SELECT 1"}).to_string(),
    )
    .await;
    assert_eq!(status, 202);
    let catalogs = get(addr, "/api/internal/catalog/catalogs").await;
    let names: Vec<&str> = catalogs["catalogs"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c| c["name"].as_str())
        .collect();
    assert!(names.contains(&"spark_catalog"), "{catalogs}");
    let schemas = get(addr, "/api/internal/catalog/spark_catalog/schemas").await;
    assert!(
        schemas["schemas"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s["name"] == "default"),
        "{schemas}"
    );
    let tables = get(addr, "/api/internal/catalog/spark_catalog/default/tables").await;
    assert!(tables["tables"].is_array(), "{tables}");
    let cols = get(
        addr,
        "/api/internal/catalog/spark_catalog/system/queries/columns",
    )
    .await;
    assert!(
        cols["columns"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["name"] == "query_id" && c["data_type"].is_string()),
        "{cols}"
    );
    // A name with a quoting character is refused before it reaches SQL.
    let a = auth();
    let (status, ..) = raw(
        addr,
        "GET",
        "/api/internal/catalog/a%60b/schemas",
        &["Host: localhost", &a],
        "",
    )
    .await;
    assert_eq!(status, 400);
}

#[tokio::test]
async fn secrets_show_as_stars_in_the_rest_json_and_the_event_stream() {
    let addr = start().await;
    let env = get(addr, "/api/internal/environment").await.to_string();
    for c in CANARIES {
        assert!(!env.contains(c), "{c} leaked into the REST body: {env}");
    }
    assert!(env.contains("***"), "{env}");
    assert!(env.contains("my.app.password"), "the key stays: {env}");

    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(
        format!(
            "GET /api/internal/events HTTP/1.1\r\nHost: localhost\r\n{}\r\n\r\n",
            auth()
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    let mut seen = String::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline && !seen.contains("event: queries") {
        let mut buf = [0_u8; 8192];
        let n = tokio::time::timeout(Duration::from_secs(5), s.read(&mut buf))
            .await
            .unwrap()
            .unwrap();
        seen.push_str(&String::from_utf8_lossy(&buf[..n]));
    }
    assert!(seen.contains("event: environment"), "{seen}");
    for c in CANARIES {
        assert!(!seen.contains(c), "{c} leaked into the stream: {seen}");
    }
    assert!(seen.contains("***"), "{seen}");
}

async fn wait_state(addr: std::net::SocketAddr, run: &str, want: &str, within: Duration) -> Value {
    let start = Instant::now();
    loop {
        let v = get(addr, &format!("/api/internal/sql/{run}")).await;
        if v["state"] == want {
            return v;
        }
        assert!(
            start.elapsed() < within,
            "run is {v}, wanted {want} within {within:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn the_editor_runs_a_statement_and_returns_rows() {
    let addr = start().await;
    let (status, body) = post(
        addr,
        "/api/internal/sql",
        &json!({"sql": "SELECT 41 + 1 AS answer, 'x' AS s, DATE '2024-02-03' AS d"}).to_string(),
    )
    .await;
    assert_eq!(status, 202, "{body}");
    let run = body["run_id"].as_str().unwrap().to_owned();
    let done = wait_state(addr, &run, "finished", Duration::from_secs(20)).await;
    assert_eq!(done["columns"], json!(["answer", "s", "d"]));
    assert_eq!(done["rows"], json!([[42, "x", "2024-02-03"]]));

    let (_, body) = post(addr, "/api/internal/sql", r#"{"sql": "SELECT FROM WHERE"}"#).await;
    let run = body["run_id"].as_str().unwrap().to_owned();
    let failed = wait_state(addr, &run, "failed", Duration::from_secs(20)).await;
    assert!(failed["error_class"].is_string(), "{failed}");

    let (status, _) = post(addr, "/api/internal/sql", "not json").await;
    assert_eq!(status, 400);
}

#[tokio::test]
async fn cancelling_a_running_statement_shows_cancelled_within_two_seconds() {
    let addr = start().await;
    // A cross join no machine finishes soon.
    let sql = "SELECT count(*) FROM range(0, 1000000000) a CROSS JOIN range(0, 1000000000) b";
    let (_, body) = post(addr, "/api/internal/sql", &json!({"sql": sql}).to_string()).await;
    let run = body["run_id"].as_str().unwrap().to_owned();
    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert_eq!(
        get(addr, &format!("/api/internal/sql/{run}")).await["state"],
        "running"
    );
    let t = Instant::now();
    let (status, _) = post(addr, &format!("/api/internal/sql/{run}/cancel"), "").await;
    assert_eq!(status, 200);
    let v = wait_state(addr, &run, "cancelled", Duration::from_secs(2)).await;
    assert!(t.elapsed() < Duration::from_secs(2), "{v}");
    let (status, _) = post(addr, &format!("/api/internal/sql/{run}/cancel"), "").await;
    assert_eq!(status, 404, "a finished run is not running");
}

#[tokio::test]
async fn the_session_cookie_is_http_only_strict_and_carries_gets_and_csrf_marked_posts() {
    let addr = start().await;
    let a = auth();
    let (status, head, _) = raw(
        addr,
        "POST",
        "/api/internal/session",
        &["Host: localhost", &a, "X-Krishiv-Csrf: 1"],
        "",
    )
    .await;
    assert_eq!(status, 204);
    assert!(head.contains("httponly"), "{head}");
    assert!(head.contains("samesite=strict"), "{head}");
    let cookie = format!("Cookie: krishiv_ui_token={TOKEN}");
    let (status, ..) = raw(
        addr,
        "GET",
        "/api/internal/health",
        &["Host: localhost", &cookie],
        "",
    )
    .await;
    assert_eq!(status, 200);
    // A change with the cookie alone (a cross-site form) is refused; with the page's header it passes.
    let (status, ..) = raw(
        addr,
        "POST",
        "/api/internal/sql",
        &["Host: localhost", &cookie],
        r#"{"sql":"SELECT 1"}"#,
    )
    .await;
    assert_eq!(status, 401);
    let (status, ..) = raw(
        addr,
        "POST",
        "/api/internal/sql",
        &["Host: localhost", &cookie, "X-Krishiv-Csrf: 1"],
        r#"{"sql":"SELECT 1"}"#,
    )
    .await;
    assert_eq!(status, 202);
    // The same with a foreign Origin is refused.
    let (status, ..) = raw(
        addr,
        "POST",
        "/api/internal/sql",
        &[
            "Host: localhost",
            &cookie,
            "X-Krishiv-Csrf: 1",
            "Origin: http://evil.example",
        ],
        r#"{"sql":"SELECT 1"}"#,
    )
    .await;
    assert_eq!(status, 403);
    // A token that cannot sit in a cookie is not turned into one.
    let (status, ..) = raw(
        addr,
        "POST",
        "/api/internal/session",
        &[
            "Host: localhost",
            "Authorization: Bearer a;b",
            "X-Krishiv-Csrf: 1",
        ],
        "",
    )
    .await;
    assert!(status == 400 || status == 401, "{status}");
}

#[tokio::test]
async fn a_host_name_that_is_not_ours_is_refused_before_anything_is_read() {
    let addr = start().await;
    let a = auth();
    for host in ["Host: evil.example.com", "Host: localhost.evil.com:80"] {
        let (status, _, body) = raw(addr, "GET", "/api/internal/health", &[host, &a], "").await;
        assert_eq!(status, 403, "{host}");
        assert!(body.contains("bad_host"), "{body}");
    }
    let (status, ..) = raw(
        addr,
        "GET",
        "/api/internal/environment",
        &["Host: 127.0.0.1:9", &a],
        "",
    )
    .await;
    assert_eq!(status, 200);
}

#[tokio::test]
async fn the_page_endpoints_are_documented_in_the_openapi_document() {
    let addr = start().await;
    let doc = get(addr, "/api/internal/openapi.json").await;
    let paths = doc["paths"].as_object().unwrap();
    for p in [
        "/api/internal/catalog/catalogs",
        "/api/internal/environment",
        "/api/internal/sql",
        "/api/internal/sql/{run}/cancel",
        "/api/internal/session",
    ] {
        assert!(paths.contains_key(p), "{p} missing from {doc}");
    }
}

/// Admits `viewer-token` for reads and everyone else for everything.
struct ViewerGuard;

impl Guard for ViewerGuard {
    fn check<'a>(&'a self, r: GuardRequest<'a>) -> BoxFuture<'a, Result<(), Refusal>> {
        Box::pin(async move {
            match (r.token, r.access) {
                (Some("viewer-token"), Access::Operate) => Err(Refusal::Denied),
                (Some("viewer-token" | "op-token"), _) => Ok(()),
                _ => Err(Refusal::Unauthenticated),
            }
        })
    }
}

#[tokio::test]
async fn a_viewer_reads_but_can_neither_cancel_nor_run_sql() {
    let rt = EngineRuntime::builder().build().unwrap();
    let source = SessionSource::new(rt).unwrap();
    let addr = serve(router_with_ui_guard(
        Arc::new(source),
        Arc::new(ViewerGuard),
    ))
    .await;
    let viewer = [
        "Host: localhost",
        "Authorization: Bearer viewer-token",
        "X-Krishiv-Csrf: 1",
    ];
    let (status, ..) = raw(addr, "GET", "/api/internal/environment", &viewer, "").await;
    assert_eq!(status, 200);
    let (status, ..) = raw(addr, "POST", "/api/internal/session", &viewer, "").await;
    assert_eq!(status, 204, "a viewer may open a session");
    let (status, ..) = raw(addr, "POST", "/api/internal/queries/x/cancel", &viewer, "").await;
    assert_eq!(status, 403);
    let (status, ..) = raw(
        addr,
        "POST",
        "/api/internal/sql",
        &viewer,
        r#"{"sql":"SELECT 1"}"#,
    )
    .await;
    assert_eq!(status, 403);
    let (status, ..) = raw(addr, "POST", "/api/internal/sql/r/cancel", &viewer, "").await;
    assert_eq!(status, 403);
}

async fn start_with(policy: UiPolicy) -> std::net::SocketAddr {
    let rt = EngineRuntime::builder().build().unwrap();
    let source = SessionSource::new(rt).unwrap();
    serve(router_with_ui_policy(
        Arc::new(source),
        Arc::new(TokenGuard::new(TOKEN)),
        policy,
    ))
    .await
}

#[tokio::test]
async fn the_session_cookie_is_secure_when_the_listener_is_tls() {
    let a = auth();
    for (secure, want) in [(true, true), (false, false)] {
        let addr = start_with(UiPolicy::for_listener(true, []).secure_cookie(secure)).await;
        let (status, head, _) = raw(
            addr,
            "POST",
            "/api/internal/session",
            &["Host: localhost", &a, "X-Krishiv-Csrf: 1"],
            "",
        )
        .await;
        assert_eq!(status, 204);
        let cookie = head.lines().find(|l| l.starts_with("set-cookie")).unwrap();
        assert_eq!(cookie.contains("; secure"), want, "{cookie}");
        assert!(
            cookie.contains("httponly") && cookie.contains("samesite=strict"),
            "{cookie}"
        );
    }
}

#[tokio::test]
async fn a_listener_on_another_address_does_not_refuse_remote_host_names() {
    let a = auth();
    // Bound to 0.0.0.0 with no names configured: a client reaching the machine by its address or any name works.
    let open = start_with(UiPolicy::for_listener(false, [])).await;
    for host in ["Host: 10.1.2.3:8080", "Host: ui.internal.example"] {
        let (status, ..) = raw(open, "GET", "/api/internal/health", &[host, &a], "").await;
        assert_eq!(status, 200, "{host}");
    }
    // With krishiv.ui.allowedHosts set, only those names (and loopback).
    let named = start_with(UiPolicy::for_listener(false, ["ui.example.com"])).await;
    let (status, ..) = raw(
        named,
        "GET",
        "/api/internal/health",
        &["Host: ui.example.com", &a],
        "",
    )
    .await;
    assert_eq!(status, 200);
    let (status, ..) = raw(
        named,
        "GET",
        "/api/internal/health",
        &["Host: evil.example", &a],
        "",
    )
    .await;
    assert_eq!(status, 403);
    // A loopback bind always checks.
    let lo = start_with(UiPolicy::for_listener(true, [])).await;
    let (status, ..) = raw(
        lo,
        "GET",
        "/api/internal/health",
        &["Host: evil.example", &a],
        "",
    )
    .await;
    assert_eq!(status, 403);
}

#[tokio::test]
async fn the_documented_properties_are_the_ones_the_responses_carry() {
    let addr = start().await;
    let doc = get(addr, "/api/internal/openapi.json").await;
    let props = |name: &str| -> Vec<String> {
        doc["components"]["schemas"][name]["properties"]
            .as_object()
            .unwrap_or_else(|| panic!("no schema {name}"))
            .keys()
            .cloned()
            .collect()
    };
    for (path, schema) in [
        ("/api/internal/catalog/catalogs", "Catalogs"),
        ("/api/internal/catalog/spark_catalog/schemas", "Schemas"),
        (
            "/api/internal/catalog/spark_catalog/default/tables",
            "Tables",
        ),
        (
            "/api/internal/catalog/spark_catalog/system/queries/columns",
            "Columns",
        ),
        ("/api/internal/environment", "Environment"),
    ] {
        let body = get(addr, path).await;
        let mut got: Vec<String> = body.as_object().unwrap().keys().cloned().collect();
        let mut want = props(schema);
        got.sort();
        want.sort();
        assert_eq!(got, want, "{path}");
    }
    let (_, body) = post(addr, "/api/internal/sql", r#"{"sql": "SELECT FROM WHERE"}"#).await;
    let failed = wait_state(
        addr,
        body["run_id"].as_str().unwrap(),
        "failed",
        Duration::from_secs(20),
    )
    .await;
    let documented = props("SqlRun");
    for k in failed.as_object().unwrap().keys() {
        assert!(
            documented.contains(k),
            "{k} is not documented: {documented:?}"
        );
    }
}
