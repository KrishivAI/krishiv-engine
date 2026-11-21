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
    let (status, body) = post(
        addr,
        "/api/internal/sql",
        &json!({"sql": "CREATE TABLE catalog_probe_table (a INT) USING parquet"}).to_string(),
    )
    .await;
    assert_eq!(status, 202);
    let run = body["run_id"].as_str().unwrap().to_owned();
    wait_state(addr, &run, "finished", Duration::from_secs(20)).await;
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
    let names: Vec<&str> = tables["tables"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    assert_eq!(names, ["catalog_probe_table"], "{tables}");
    let cols = get(
        addr,
        "/api/internal/catalog/spark_catalog/default/catalog_probe_table/columns",
    )
    .await;
    assert_eq!(
        cols["columns"],
        json!([{"name": "a", "data_type": "int"}]),
        "{cols}"
    );
    // Every statement names its catalog: `system` is a catalog, not a schema of `spark_catalog`.
    let a = auth();
    let (status, ..) = raw(
        addr,
        "GET",
        "/api/internal/catalog/spark_catalog/system/queries/columns",
        &["Host: localhost", &a],
        "",
    )
    .await;
    assert_ne!(status, 200);
    // A name with a quoting character is refused before it reaches SQL.
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
    let (status, _) = post(addr, &format!("/api/internal/sql/{run}/cancel"), "").await;
    assert_eq!(status, 200);
    // `wait_state` fails the test when the state is not reached within the 2 s bound.
    wait_state(addr, &run, "cancelled", Duration::from_secs(2)).await;
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
    // A token that cannot sit in a cookie is not turned into one: this server's own token passes the guard, so
    // the request reaches the cookie check.
    let semi = serve(router_with_ui(
        Arc::new(SessionSource::new(EngineRuntime::builder().build().unwrap()).unwrap()),
        "a;b",
    ))
    .await;
    let (status, head, _) = raw(
        semi,
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
    assert_eq!(status, 400);
    assert!(!head.contains("set-cookie"), "{head}");
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
    let (_, body) = post(
        addr,
        "/api/internal/sql",
        &json!({"sql": "CREATE TABLE documented_columns (a INT) USING parquet"}).to_string(),
    )
    .await;
    let run = body["run_id"].as_str().unwrap().to_owned();
    wait_state(addr, &run, "finished", Duration::from_secs(20)).await;
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

/// A browser keeps cookies per host name, not per port. Two consoles on one host (one per embedded session,
/// each with its own port and token) therefore each name their cookie after their port, and read only that one.
#[tokio::test]
async fn the_session_cookie_is_named_after_the_port_so_two_consoles_share_a_browser() {
    let addr = start_with(UiPolicy::for_listener(true, [])).await;
    let port = addr.port();
    let host = format!("Host: localhost:{port}");
    let a = auth();
    let (status, head, _) = raw(
        addr,
        "POST",
        "/api/internal/session",
        &[&host, &a, "X-Krishiv-Csrf: 1"],
        "",
    )
    .await;
    assert_eq!(status, 204);
    let cookie = head.lines().find(|l| l.starts_with("set-cookie")).unwrap();
    assert!(
        cookie.starts_with(&format!("set-cookie: krishiv_ui_token_{port}=")),
        "{cookie}"
    );
    let other = port.wrapping_add(1);
    for (sent, want) in [
        // The browser sends both consoles' cookies to both; each reads its own.
        (
            format!("krishiv_ui_token_{other}=another-console; krishiv_ui_token_{port}={TOKEN}"),
            200,
        ),
        (
            format!("krishiv_ui_token_{port}={TOKEN}; krishiv_ui_token_{other}=another-console"),
            200,
        ),
        // Another console's cookie, or one with no port in its name, is not this console's.
        (format!("krishiv_ui_token_{other}={TOKEN}"), 401),
        (format!("krishiv_ui_token={TOKEN}"), 401),
        (format!("krishiv_ui_token_{port}0={TOKEN}"), 401),
    ] {
        let cookie = format!("Cookie: {sent}");
        let (status, ..) = raw(addr, "GET", "/api/internal/health", &[&host, &cookie], "").await;
        assert_eq!(status, want, "{sent}");
    }
    // An IPv6 address keeps its brackets in `Host`; the port follows them.
    let (status, head, _) = raw(
        addr,
        "POST",
        "/api/internal/session",
        &[&format!("Host: [::1]:{port}"), &a, "X-Krishiv-Csrf: 1"],
        "",
    )
    .await;
    assert_eq!(status, 204);
    assert!(
        head.contains(&format!("set-cookie: krishiv_ui_token_{port}=")),
        "{head}"
    );
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
            cookie.starts_with("set-cookie: krishiv_ui_token=")
                && cookie.contains("; path=/api/internal")
                && cookie.contains("; max-age=28800"),
            "{cookie}"
        );
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
    let (_, body) = post(
        addr,
        "/api/internal/sql",
        &json!({"sql": "CREATE TABLE documented_columns (a INT) USING parquet"}).to_string(),
    )
    .await;
    let run = body["run_id"].as_str().unwrap().to_owned();
    wait_state(addr, &run, "finished", Duration::from_secs(20)).await;
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
            "/api/internal/catalog/spark_catalog/default/documented_columns/columns",
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

#[tokio::test]
async fn an_editor_result_is_cut_at_the_row_cap_and_says_so() {
    let addr = start().await;
    let (_, body) = post(
        addr,
        "/api/internal/sql",
        &json!({"sql": "SELECT id FROM range(0, 5000)"}).to_string(),
    )
    .await;
    let run = body["run_id"].as_str().unwrap().to_owned();
    let done = wait_state(addr, &run, "finished", Duration::from_secs(20)).await;
    assert_eq!(done["rows"].as_array().unwrap().len(), 1000, "{done}");
    assert_eq!(done["truncated"], json!(true), "{done}");
}

#[tokio::test]
async fn a_query_id_stored_twice_is_listed_once_as_its_newest_profile() {
    use krishiv_server::internal::{SystemTable, UiSource};
    use krishiv_server::session_source::{ProfilesSource, SessionSource};
    let dir = tempfile::tempdir().unwrap();
    {
        let rt = EngineRuntime::builder()
            .config("krishiv.history.path", dir.path().to_str().unwrap())
            .build()
            .unwrap();
        let session = krishiv_session::Session::builder()
            .runtime(Arc::clone(&rt))
            .build()
            .unwrap();
        let df = session
            .sql("SELECT id % 3 AS k, count(*) AS n FROM range(0, 300) GROUP BY id % 3")
            .await
            .unwrap();
        df.collect().await.unwrap();
    }
    // The same id stored by two runs of a server (a client that reuses its operation id): an older profile
    // that finished, a newer one that failed.
    let older = krishiv_session::read_profile_history(dir.path())
        .unwrap()
        .remove(0);
    let mut newer = older.clone();
    newer.state = "failed".to_owned();
    newer.finished_at_ms += 60_000;
    let stored = [older.clone(), newer];
    let doc: Value = serde_json::from_str(&older.profile).unwrap();
    let operators = doc["operators"].as_array().unwrap().len();
    assert!(operators > 0);

    let profiles = ProfilesSource::from_profiles(&stored);
    let restarted = SessionSource::new(EngineRuntime::builder().build().unwrap())
        .unwrap()
        .with_stored_history(&stored);
    let sources: [(&str, &dyn UiSource); 2] = [("profiles", &profiles), ("server", &restarted)];
    for (name, source) in sources {
        let queries: Vec<_> = source
            .rows(SystemTable::Queries)
            .await
            .unwrap()
            .into_iter()
            .filter(|q| q["query_id"] == json!(older.query_id))
            .collect();
        assert_eq!(queries.len(), 1, "{name}: {queries:?}");
        assert_eq!(queries[0]["state"], json!("failed"), "{name}");
        let of_query = source
            .rows(SystemTable::Operators)
            .await
            .unwrap()
            .into_iter()
            .filter(|o| o["query_id"] == json!(older.query_id))
            .count();
        assert_eq!(of_query, operators, "{name}");
    }
}

#[tokio::test]
async fn stored_profiles_give_the_history_and_operator_rows_and_cannot_cancel() {
    use krishiv_server::internal::{SystemTable, UiSource};
    use krishiv_server::session_source::ProfilesSource;
    let dir = tempfile::tempdir().unwrap();
    {
        let rt = EngineRuntime::builder()
            .config("krishiv.history.path", dir.path().to_str().unwrap())
            .build()
            .unwrap();
        let session = krishiv_session::Session::builder()
            .runtime(Arc::clone(&rt))
            .build()
            .unwrap();
        let df = session
            .sql("SELECT id % 3 AS k, count(*) AS n FROM range(0, 300) GROUP BY id % 3")
            .await
            .unwrap();
        df.collect().await.unwrap();
        // Dropping the runtime flushes the history writer.
    }
    let source = ProfilesSource::open(dir.path()).unwrap();
    let queries = source.rows(SystemTable::Queries).await.unwrap();
    assert_eq!(queries.len(), 1, "{queries:?}");
    assert_eq!(queries[0]["state"], json!("finished"));
    assert!(queries[0]["ended_at"].as_i64().unwrap() > 0);
    let operators = source.rows(SystemTable::Operators).await.unwrap();
    // The oracle is the stored profile document itself: every operator, its name and its row counts.
    let stored = krishiv_session::read_profile_history(dir.path()).unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(queries[0]["query_id"], json!(stored[0].query_id));
    assert_eq!(queries[0]["wall_ns"], json!(stored[0].duration_ns));
    assert_eq!(queries[0]["principal"], json!(stored[0].principal));
    let doc: Value = serde_json::from_str(&stored[0].profile).unwrap();
    let want: Vec<(Value, Value, Value, Value)> = doc["operators"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| {
            (
                o["id"].clone(),
                o["name"].clone(),
                o["metrics"]["rows_in"].clone(),
                o["metrics"]["rows_out"].clone(),
            )
        })
        .collect();
    let got: Vec<(Value, Value, Value, Value)> = operators
        .iter()
        .map(|o| {
            (
                o["operator_id"].clone(),
                o["name"].clone(),
                o["rows_in"].clone(),
                o["rows_out"].clone(),
            )
        })
        .collect();
    assert!(!want.is_empty());
    assert_eq!(got, want);
    assert!(got.iter().all(|(_, name, ..)| name.is_string()), "{got:?}");
    // The query ends in 3 groups.
    assert!(got.iter().any(|(.., out)| *out == json!(3)), "{got:?}");
    assert!(
        operators
            .iter()
            .all(|o| o["query_id"] == queries[0]["query_id"])
    );
    assert!(source.rows(SystemTable::Memory).await.unwrap().is_empty());
    assert!(source.rows(SystemTable::Caches).await.unwrap().is_empty());
    assert!(!source.cancel("anything").await.unwrap());
    assert!(ProfilesSource::open(std::path::Path::new("/nonexistent/krishiv-profiles")).is_err());

    // The page asks the server which mode it is in, and hides what saved profiles cannot give (ADR 0024).
    assert_eq!(source.mode(), "profiles");
    assert_eq!(source.session_id(), None);
    let addr = serve(router_with_ui(Arc::new(source), TOKEN)).await;
    let health = get(addr, "/api/internal/health").await;
    assert_eq!(health["mode"], json!("profiles"), "{health}");
    assert!(health["session_id"].is_null(), "{health}");
}

struct Bare;
impl krishiv_server::internal::UiSource for Bare {
    fn rows(
        &self,
        _t: krishiv_server::internal::SystemTable,
    ) -> krishiv_server::internal::UiFuture<'_, Vec<krishiv_server::internal::Row>> {
        Box::pin(async { Ok(Vec::new()) })
    }
    fn cancel<'a>(&'a self, _q: &'a str) -> krishiv_server::internal::UiFuture<'a, bool> {
        Box::pin(async { Ok(false) })
    }
}

#[tokio::test]
async fn a_source_with_only_rows_and_cancel_offers_no_pages_and_a_caller_prints_its_name() {
    use krishiv_server::internal::UiSource;
    let s = Bare;
    assert!(s.catalog(&[]).await.is_err());
    assert!(s.environment().await.is_err());
    assert!(s.sql_start("SELECT 1", None).await.is_err());
    assert!(s.sql_status("r", None).await.unwrap().is_none());
    assert!(!s.sql_cancel("r", None).await.unwrap());
}

#[tokio::test]
async fn the_origin_scheme_must_match_the_listener() {
    let a = auth();
    for (tls, origin, want) in [
        (true, "Origin: https://localhost", 202),
        (true, "Origin: http://localhost", 403),
        (false, "Origin: http://localhost", 202),
        (false, "Origin: https://localhost", 403),
    ] {
        let addr = start_with(UiPolicy::for_listener(true, []).secure_cookie(tls)).await;
        let (status, ..) = raw(
            addr,
            "POST",
            "/api/internal/sql",
            &["Host: localhost", &a, "X-Krishiv-Csrf: 1", origin],
            r#"{"sql":"SELECT 1"}"#,
        )
        .await;
        assert_eq!(status, want, "tls={tls} {origin}");
    }
}

#[tokio::test]
async fn only_an_identified_caller_or_the_local_admin_may_start_a_run() {
    use krishiv_server::guard::Caller;
    use krishiv_server::internal::UiSource;
    let source = SessionSource::new(EngineRuntime::builder().build().unwrap()).unwrap();
    // A guard that authenticates nobody by name gives an anonymous caller: no rights, no editor.
    let anonymous = Caller::anonymous();
    assert!(
        source
            .sql_start("SELECT 1", Some(&anonymous))
            .await
            .is_err()
    );
    // The start-up token's local administrator runs in the ordinary session.
    let local = Caller::local_admin();
    let id = source.sql_start("SELECT 1", Some(&local)).await.unwrap();
    assert!(!id.is_empty());
}

#[tokio::test]
async fn the_editor_result_is_exact_at_the_row_cap_boundary() {
    let addr = start().await;
    for (rows, truncated) in [(1000, false), (1001, true)] {
        let (_, body) = post(
            addr,
            "/api/internal/sql",
            &json!({"sql": format!("SELECT id FROM range(0, {rows}) ORDER BY id")}).to_string(),
        )
        .await;
        let run = body["run_id"].as_str().unwrap().to_owned();
        let done = wait_state(addr, &run, "finished", Duration::from_secs(20)).await;
        let got = done["rows"].as_array().unwrap();
        assert_eq!(got.len(), 1000, "{rows} rows");
        assert_eq!(got[0], json!([0]));
        assert_eq!(got[999], json!([999]));
        assert_eq!(done["truncated"], json!(truncated), "{rows} rows");
    }
}

/// One planted query, three surfaces (P3-T073): the profile of the run, the `hints` the query page's API
/// returns, and the `system.queries` row all carry the same hints.
#[tokio::test]
async fn the_query_page_the_system_table_and_the_profile_carry_the_same_hints() {
    let rt = EngineRuntime::builder().build().unwrap();
    let user = krishiv_session::Session::builder()
        .runtime(Arc::clone(&rt))
        .build()
        .unwrap();
    for ddl in [
        "CREATE OR REPLACE TEMP VIEW lineitem AS SELECT id % 100000 AS l_partkey FROM range(3000000)",
        "CREATE OR REPLACE TEMP VIEW part AS SELECT id AS p_partkey, \
         CASE WHEN id % 100 = 0 THEN 'forest' ELSE 'other' END AS p_name FROM range(100000)",
    ] {
        user.sql(ddl).await.unwrap().collect().await.unwrap();
    }
    let sql = "SELECT count(*) FROM lineitem JOIN part ON l_partkey + 0 = p_partkey \
               WHERE p_name LIKE 'forest%'";
    let (_, profile) = user
        .sql(sql)
        .await
        .unwrap()
        .collect_with_profile()
        .await
        .unwrap();
    let from_profile: Vec<(String, Option<i64>, Option<i64>)> = profile
        .hints
        .iter()
        .map(|h| {
            (
                h.code.as_str().to_owned(),
                h.node.map(i64::from),
                h.operator.map(|o| i64::from(o.0)),
            )
        })
        .collect();
    assert!(!from_profile.is_empty(), "the planted query raised no hint");

    let source = SessionSource::new(rt).unwrap();
    let addr = serve(router_with_ui(Arc::new(source), TOKEN)).await;
    let list = get(addr, "/api/internal/history").await;
    let rows = list["queries"].as_array().unwrap();
    let row = rows
        .iter()
        .find(|r| {
            r["query_text"]
                .as_str()
                .is_some_and(|t| t.starts_with("SELECT count(*) FROM lineitem JOIN part"))
        })
        .unwrap_or_else(|| panic!("the planted query is not listed: {list}"));
    let id = row["query_id"].as_str().unwrap().to_owned();
    let detail = get(addr, &format!("/api/internal/queries/{id}")).await;
    let cells = |v: &Value| -> Vec<(String, Option<i64>, Option<i64>)> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|h| {
                (
                    h["code"].as_str().unwrap().to_owned(),
                    h["node_id"].as_i64(),
                    h["operator_id"].as_i64(),
                )
            })
            .collect()
    };
    assert_eq!(cells(&detail["query"]["hints"]), from_profile, "{detail}");
    assert_eq!(cells(&row["hints"]), from_profile, "{row}");
}
