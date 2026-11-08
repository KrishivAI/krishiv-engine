// SPDX-License-Identifier: Apache-2.0
//! `/api/internal/*` over a real socket with a fake `UiSource`: the token, the security headers, each page's
//! data, the allowlist, cancel, and the event stream.
#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;
use std::sync::Mutex;

use krishiv_server::internal::{Row, SystemTable, UiError, UiFuture, UiSource, router_with_ui};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const TOKEN: &str = "test-token";

#[expect(
    clippy::needless_pass_by_value,
    reason = "takes the json! literal as written"
)]
fn rows(v: Value) -> Vec<Row> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_object().unwrap().clone())
        .collect()
}

struct Fake {
    cancelled: Mutex<Vec<String>>,
    fail: bool,
    /// How many times the live-query table was read.
    reads: std::sync::atomic::AtomicUsize,
}

impl UiSource for Fake {
    fn rows(&self, table: SystemTable) -> UiFuture<'_, Vec<Row>> {
        let fail = self.fail;
        Box::pin(async move {
            if table == SystemTable::Queries {
                self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
            if fail {
                return Err(UiError::with_class(
                    "system.queries is unavailable",
                    "KRISHIV_TEST_FAILURE",
                    Some("XX000"),
                ));
            }
            Ok(match table {
                SystemTable::Queries => rows(json!([
                    {"query_id": "q-run", "state": "running", "principal": "ann", "submitted_at": 2_000,
                     "query_text": "SELECT 1 FROM t WHERE pw = 's3cr3t-canary'", "secret_option": "hunter2", "workers": 1},
                    {"query_id": "q-wait", "state": "queued", "principal": "bob", "submitted_at": 1_000},
                    {"query_id": "q-a", "state": "finished", "principal": "ann", "submitted_at": 100,
                     "wall_ns": 1_000, "peak_memory_bytes": 10, "rows_produced": 5, "spill_bytes": 0},
                    {"query_id": "q-b", "state": "finished", "principal": "ann", "submitted_at": 200,
                     "wall_ns": 1_500, "peak_memory_bytes": 30, "rows_produced": 5, "spill_bytes": 7},
                    {"query_id": "q-bad", "state": "failed", "principal": "bob", "submitted_at": 300,
                     "error_class": "DIVIDE_BY_ZERO"},
                    {"query_id": "q-min", "state": "finished", "principal": "cy", "submitted_at": 50, "wall_ns": -9_223_372_036_854_775_807_i64},
                    {"query_id": "q-max", "state": "finished", "principal": "cy", "submitted_at": 60, "wall_ns": 9_223_372_036_854_775_807_i64}
                ])),
                SystemTable::Operators => rows(json!([
                    {"query_id": "q-b", "operator_id": 1, "name": "Scan", "drivers": 1, "compute_ns": 100,
                     "max_driver_compute_ns": 100, "spilled_bytes": 0, "blocked_memory_ns": 0},
                    {"query_id": "q-b", "operator_id": 2, "name": "HashJoin", "drivers": 4, "compute_ns": 900,
                     "max_driver_compute_ns": 800, "spilled_bytes": 4096, "blocked_memory_ns": 0},
                    {"query_id": "q-b", "operator_id": 3, "name": "Sort", "drivers": 1, "compute_ns": 50,
                     "max_driver_compute_ns": 50, "spilled_bytes": 0, "blocked_memory_ns": 400},
                    {"query_id": "other", "operator_id": 1, "name": "Scan", "drivers": 1, "compute_ns": 1}
                ])),
                SystemTable::Memory => rows(json!([
                    {"worker_id": "local", "path": "root", "parent": null, "kind": "root",
                     "reserved_bytes": 100, "peak_bytes": 200, "limit_bytes": 1000, "spillable": false},
                    {"worker_id": "w2", "path": "root/q", "parent": "root", "kind": "query",
                     "reserved_bytes": 5, "peak_bytes": 6, "limit_bytes": null, "spillable": true},
                    {"worker_id": "w2", "path": "root", "parent": null, "kind": "root",
                     "reserved_bytes": 20, "peak_bytes": 30, "limit_bytes": 500, "spillable": false},
                    {"worker_id": "local", "path": "root/q", "parent": "root", "kind": "query",
                     "reserved_bytes": 50, "peak_bytes": 60, "limit_bytes": null, "spillable": true}
                ])),
                SystemTable::Caches => rows(json!([
                    {"worker_id": "local", "cache": "metadata", "entries": 3, "bytes": 30, "hits": 3, "misses": 1},
                    {"worker_id": "local", "cache": "block", "entries": 0, "bytes": 0, "hits": 0, "misses": 0},
                    {"worker_id": "local", "cache": "huge", "entries": 1, "bytes": 1, "hits": 9_223_372_036_854_775_807_i64, "misses": 5}
                ])),
            })
        })
    }

    fn cancel<'a>(&'a self, query_id: &'a str) -> UiFuture<'a, bool> {
        Box::pin(async move {
            self.cancelled.lock().unwrap().push(query_id.to_owned());
            Ok(query_id == "q-run")
        })
    }
}

async fn start(fail: bool) -> (std::net::SocketAddr, Arc<Fake>) {
    let fake = Arc::new(Fake {
        cancelled: Mutex::new(Vec::new()),
        fail,
        reads: std::sync::atomic::AtomicUsize::new(0),
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router_with_ui(Arc::<Fake>::clone(&fake), TOKEN);
    tokio::spawn(async move { axum::serve(listener, app).await });
    (addr, fake)
}

async fn request(
    addr: std::net::SocketAddr,
    method: &str,
    path: &str,
    auth: bool,
) -> (u16, String, Value) {
    let mut s = TcpStream::connect(addr).await.unwrap();
    let bearer = if auth {
        format!("Authorization: Bearer {TOKEN}\r\n")
    } else {
        String::new()
    };
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\n{bearer}Content-Length: 0\r\nConnection: close\r\n\r\n"
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut out = Vec::new();
    tokio::time::timeout(std::time::Duration::from_secs(10), s.read_to_end(&mut out))
        .await
        .expect("read timed out")
        .unwrap();
    let text = String::from_utf8(out).unwrap();
    let (head, body) = text.split_once("\r\n\r\n").unwrap();
    let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    let json = serde_json::from_str(body.trim()).unwrap_or(Value::Null);
    (status, head.to_lowercase(), json)
}

async fn get(addr: std::net::SocketAddr, path: &str) -> Value {
    let (status, _, body) = request(addr, "GET", path, true).await;
    assert_eq!(status, 200, "{path}: {body}");
    body
}

#[tokio::test]
async fn every_internal_endpoint_needs_the_token() {
    let (addr, _) = start(false).await;
    for path in [
        "/api/internal/health",
        "/api/internal/queries",
        "/api/internal/history",
        "/api/internal/events",
    ] {
        let (status, _, body) = request(addr, "GET", path, false).await;
        assert_eq!(status, 401, "{path}");
        assert_eq!(body["error"], "unauthorized");
    }
    // The token also works as a cookie and as a query parameter (an EventSource cannot set headers).
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(format!("GET /api/internal/queries?token={TOKEN} HTTP/1.1\r\nHost: l\r\nConnection: close\r\n\r\n").as_bytes())
        .await
        .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).await.unwrap();
    assert!(out.starts_with("HTTP/1.1 200"), "{out}");
    let (status, ..) = request(addr, "GET", "/api/internal/queries?token=wrong", false).await;
    assert_eq!(status, 401);
    for (cookie, expected) in [
        (format!("a=b; krishiv_ui_token={TOKEN}"), "HTTP/1.1 200"),
        ("a=b; krishiv_ui_token=wrong".to_owned(), "HTTP/1.1 401"),
    ] {
        let mut s = TcpStream::connect(addr).await.unwrap();
        s.write_all(
            format!("GET /api/internal/queries HTTP/1.1\r\nHost: l\r\nCookie: {cookie}\r\nConnection: close\r\n\r\n").as_bytes(),
        )
        .await
        .unwrap();
        let mut out = String::new();
        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            s.read_to_string(&mut out),
        )
        .await
        .expect("read timed out")
        .unwrap();
        assert!(out.starts_with(expected), "{cookie}: {out}");
    }
    // The public version route and the SPA stay open.
    let (status, ..) = request(addr, "GET", "/api/v1/version", false).await;
    assert_eq!(status, 200);
}

#[tokio::test]
async fn every_response_carries_the_security_headers() {
    let (addr, _) = start(false).await;
    for path in ["/", "/api/v1/version", "/api/internal/health"] {
        let (_, head, _) = request(addr, "GET", path, true).await;
        assert!(
            head.contains("content-security-policy: default-src 'self'"),
            "{path}: {head}"
        );
        assert!(head.contains("frame-ancestors 'none'"), "{path}");
        assert!(head.contains("x-frame-options: deny"), "{path}");
        assert!(head.contains("x-content-type-options: nosniff"), "{path}");
        assert!(head.contains("cache-control: no-store"), "{path}");
    }
}

#[tokio::test]
async fn the_live_list_has_running_and_queued_queries_oldest_first_and_only_allowlisted_columns() {
    let (addr, _) = start(false).await;
    let body = get(addr, "/api/internal/queries").await;
    let ids: Vec<&str> = body["queries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| q["query_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["q-wait", "q-run"]);
    let run = &body["queries"][1];
    assert_eq!(
        run["query_text"], "SELECT ? FROM t WHERE pw = ?",
        "literals never reach the page"
    );
    assert!(!body.to_string().contains("s3cr3t-canary"), "{body}");
    assert!(
        run.get("secret_option").is_none(),
        "a column outside the allowlist is dropped: {run}"
    );
}

#[tokio::test]
async fn a_query_detail_names_the_hot_operator_and_explains_spill_skew_and_memory_waits() {
    let (addr, _) = start(false).await;
    let body = get(addr, "/api/internal/queries/q-b").await;
    assert_eq!(body["hot_operator"]["operator"], "HashJoin");
    let ops = body["operators"].as_array().unwrap();
    assert_eq!(ops.len(), 3, "only this query's operators");
    let kinds: Vec<(&str, &str)> = body["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| (w["operator"].as_str().unwrap(), w["kind"].as_str().unwrap()))
        .collect();
    assert!(kinds.contains(&("HashJoin", "spill")), "{kinds:?}");
    let spill_hint = body["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .find(|w| w["kind"] == "spill")
        .map(|w| w["hint"].as_str().unwrap().to_owned())
        .unwrap();
    assert!(
        spill_hint.contains("spilled 4.0 KiB to disk"),
        "bytes are written like the pages write them: {spill_hint}"
    );
    assert!(kinds.contains(&("HashJoin", "skew")), "{kinds:?}");
    assert!(kinds.contains(&("Sort", "memory")), "{kinds:?}");
    assert!(!kinds.iter().any(|(op, _)| *op == "Scan"));
    assert!(
        body["warnings"][0]["hint"].as_str().unwrap().len() > 20,
        "a plain-language hint"
    );
    let (status, _, missing) = request(addr, "GET", "/api/internal/queries/nope", true).await;
    assert_eq!(status, 404);
    assert_eq!(missing["error"], "no_such_query");
}

#[tokio::test]
async fn history_lists_finished_queries_newest_first_with_filters_and_a_limit() {
    let (addr, _) = start(false).await;
    let all = get(addr, "/api/internal/history").await;
    let ids: Vec<&str> = all["queries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| q["query_id"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        ["q-bad", "q-b", "q-a", "q-max", "q-min"],
        "running and queued queries are not history"
    );
    let failed = get(addr, "/api/internal/history?state=failed").await;
    assert_eq!(failed["queries"].as_array().unwrap().len(), 1);
    let ann = get(addr, "/api/internal/history?principal=ann&limit=1").await;
    assert_eq!(ann["queries"][0]["query_id"], "q-b");
    assert_eq!(ann["queries"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn compare_gives_both_runs_and_the_change_in_each_number() {
    let (addr, _) = start(false).await;
    let body = get(addr, "/api/internal/compare?a=q-a&b=q-b").await;
    assert_eq!(body["delta"]["wall_ns"], 500);
    assert_eq!(body["delta"]["peak_memory_bytes"], 20);
    assert_eq!(body["delta"]["spill_bytes"], 7);
    assert_eq!(body["a"]["query_id"], "q-a");
    let big = get(addr, "/api/internal/compare?a=q-min&b=q-max").await;
    assert!(
        big["delta"].get("wall_ns").is_none(),
        "an overflowing difference is left out: {big}"
    );
    let (status, ..) = request(addr, "GET", "/api/internal/compare?a=q-a", true).await;
    assert_eq!(status, 400);
    let (status, ..) = request(addr, "GET", "/api/internal/compare?a=q-a&b=zzz", true).await;
    assert_eq!(status, 404);
}

#[tokio::test]
async fn health_and_resources_report_memory_and_cache_hit_rates() {
    let (addr, _) = start(false).await;
    let h = get(addr, "/api/internal/health").await;
    assert_eq!(h["running_queries"], 1);
    assert_eq!(h["queued_queries"], 1);
    assert_eq!(h["workers"], 2, "two worker ids in the memory table");
    assert_eq!(
        h["memory"]["reserved_bytes"], 120,
        "every worker's root pool is added up"
    );
    assert_eq!(h["memory"]["peak_bytes"], 230);
    assert_eq!(h["memory"]["limit_bytes"], 1500);
    assert_eq!(h["spill_bytes_total"], 7);
    assert_eq!(h["caches"][0]["hit_rate"], 0.75);
    assert_eq!(
        h["caches"][1]["hit_rate"],
        Value::Null,
        "no lookups yet is not a 0% hit rate"
    );
    let r = get(addr, "/api/internal/resources").await;
    assert_eq!(r["memory"].as_array().unwrap().len(), 4);
    let keys: std::collections::BTreeSet<String> = r["memory"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            format!(
                "{}/{}",
                m["worker_id"].as_str().unwrap(),
                m["path"].as_str().unwrap()
            )
        })
        .collect();
    assert_eq!(
        keys.len(),
        4,
        "worker id and path together identify a pool (the page keys its rows by them): {keys:?}"
    );
    assert_eq!(r["caches"][0]["cache"], "metadata");
}

#[tokio::test]
async fn cancel_reaches_the_source_and_reports_whether_the_query_was_running() {
    let (addr, fake) = start(false).await;
    let (status, _, body) = request(addr, "POST", "/api/internal/queries/q-run/cancel", true).await;
    assert_eq!((status, &body["cancelled"]), (200, &json!(true)));
    let (status, _, body) = request(addr, "POST", "/api/internal/queries/q-a/cancel", true).await;
    assert_eq!((status, body["error"].as_str()), (404, Some("not_running")));
    assert_eq!(*fake.cancelled.lock().unwrap(), ["q-run", "q-a"]);
    // GET is not a way to cancel.
    let (status, ..) = request(addr, "GET", "/api/internal/queries/q-run/cancel", true).await;
    assert_eq!(status, 405);
}

#[tokio::test]
async fn a_failing_source_is_a_json_500_not_a_hang() {
    let (addr, _) = start(true).await;
    let (status, _, body) = request(addr, "GET", "/api/internal/queries", true).await;
    assert_eq!(status, 500);
    assert_eq!(body["error"], "source_failed");
    assert_eq!(body["error_class"], "KRISHIV_TEST_FAILURE");
    assert_eq!(body["sql_state"], "XX000");
    assert!(body["message"].as_str().unwrap().contains("unavailable"));
}

#[tokio::test]
async fn the_event_stream_sends_the_live_list_about_once_a_second() {
    let (addr, _) = start(false).await;
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(
        format!(
            "GET /api/internal/events HTTP/1.1\r\nHost: l\r\nAuthorization: Bearer {TOKEN}\r\n\r\n"
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    let mut seen = String::new();
    let mut arrivals = Vec::new();
    let mut buf = [0_u8; 4096];
    // Two events: the first comes at once, the second must follow about a second later.
    while arrivals.len() < 2 {
        let n = tokio::time::timeout(std::time::Duration::from_secs(20), s.read(&mut buf))
            .await
            .expect("no event in 20 s")
            .unwrap();
        assert!(n > 0, "stream closed: {seen}");
        seen.push_str(&String::from_utf8_lossy(&buf[..n]));
        while arrivals.len() < seen.matches("event: queries").count() {
            arrivals.push(std::time::Instant::now());
        }
    }
    let lower = seen.to_lowercase();
    assert!(lower.contains("content-type: text/event-stream"), "{seen}");
    assert!(seen.contains("\"query_id\":\"q-run\""), "{seen}");
    let gap = arrivals[1] - arrivals[0];
    assert!(
        gap >= std::time::Duration::from_millis(900),
        "events {gap:?} apart, faster than once a second"
    );
}

#[tokio::test]
async fn nothing_that_changes_state_accepts_a_token_in_the_address_or_a_cookie() {
    let (addr, fake) = start(false).await;
    for (name, extra, path) in [
        ("no token", "", "/api/internal/queries/q-run/cancel"),
        (
            "token in the address",
            "",
            &format!("/api/internal/queries/q-run/cancel?token={TOKEN}"),
        ),
        (
            "token in a cookie",
            &format!("Cookie: krishiv_ui_token={TOKEN}\r\n"),
            "/api/internal/queries/q-run/cancel",
        ),
    ] {
        let mut s = TcpStream::connect(addr).await.unwrap();
        s.write_all(format!("POST {path} HTTP/1.1\r\nHost: l\r\n{extra}Content-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).await.unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).await.unwrap();
        assert!(out.starts_with("HTTP/1.1 401"), "{name}: {out}");
    }
    assert!(
        fake.cancelled.lock().unwrap().is_empty(),
        "nothing was cancelled"
    );
}

async fn open_stream(addr: std::net::SocketAddr) -> TcpStream {
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(
        format!(
            "GET /api/internal/events HTTP/1.1\r\nHost: l\r\nAuthorization: Bearer {TOKEN}\r\n\r\n"
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    s
}

async fn first_line(s: &mut TcpStream) -> String {
    let mut buf = [0_u8; 256];
    let n = tokio::time::timeout(std::time::Duration::from_secs(20), s.read(&mut buf))
        .await
        .expect("no response")
        .unwrap();
    String::from_utf8_lossy(&buf[..n])
        .lines()
        .next()
        .unwrap_or("")
        .to_owned()
}

#[tokio::test]
async fn the_event_streams_are_capped_and_a_slot_comes_back_when_a_client_leaves() {
    let (addr, _) = start(false).await;
    let mut open = Vec::new();
    for _ in 0..krishiv_server::internal::MAX_EVENT_CLIENTS {
        let mut s = open_stream(addr).await;
        assert_eq!(first_line(&mut s).await, "HTTP/1.1 200 OK");
        open.push(s);
    }
    let mut extra = open_stream(addr).await;
    assert_eq!(
        first_line(&mut extra).await,
        "HTTP/1.1 503 Service Unavailable",
        "the stream over the cap is refused"
    );
    drop(open.pop());
    // The server notices the closed connection when it next writes to it (within a period or two).
    let mut accepted = false;
    for _ in 0..10 {
        tokio::time::sleep(std::time::Duration::from_millis(700)).await;
        let mut again = open_stream(addr).await;
        if first_line(&mut again).await == "HTTP/1.1 200 OK" {
            accepted = true;
            break;
        }
    }
    assert!(accepted, "a client that left did not free its place");
}

#[tokio::test]
async fn several_open_tabs_share_one_read_of_the_live_queries_a_period() {
    let (addr, fake) = start(false).await;
    let mut tabs = Vec::new();
    for _ in 0..6 {
        tabs.push(open_stream(addr).await);
    }
    for t in &mut tabs {
        assert_eq!(first_line(t).await, "HTTP/1.1 200 OK");
    }
    // Wait for the first event of every tab (one period), then count the reads.
    let mut buf = [0_u8; 4096];
    for t in &mut tabs {
        let mut seen = String::new();
        while !seen.contains("event: queries") {
            let n = tokio::time::timeout(std::time::Duration::from_secs(20), t.read(&mut buf))
                .await
                .expect("no event")
                .unwrap();
            seen.push_str(&String::from_utf8_lossy(&buf[..n]));
        }
    }
    let reads = fake.reads.load(std::sync::atomic::Ordering::SeqCst);
    assert!(
        reads <= 3,
        "6 tabs caused {reads} reads of the live queries; they should share one a period"
    );
}
