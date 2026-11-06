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
}

impl UiSource for Fake {
    fn rows(&self, table: SystemTable) -> UiFuture<'_, Vec<Row>> {
        let fail = self.fail;
        Box::pin(async move {
            if fail {
                return Err(UiError("system.queries is unavailable".to_owned()));
            }
            Ok(match table {
                SystemTable::Queries => rows(json!([
                    {"query_id": "q-run", "state": "running", "principal": "ann", "submitted_at": 2_000,
                     "query_text": "SELECT 1", "secret_option": "hunter2", "workers": 1},
                    {"query_id": "q-wait", "state": "queued", "principal": "bob", "submitted_at": 1_000},
                    {"query_id": "q-a", "state": "finished", "principal": "ann", "submitted_at": 100,
                     "wall_ns": 1_000, "peak_memory_bytes": 10, "rows_produced": 5, "spill_bytes": 0},
                    {"query_id": "q-b", "state": "finished", "principal": "ann", "submitted_at": 200,
                     "wall_ns": 1_500, "peak_memory_bytes": 30, "rows_produced": 5, "spill_bytes": 7},
                    {"query_id": "q-bad", "state": "failed", "principal": "bob", "submitted_at": 300,
                     "error_class": "DIVIDE_BY_ZERO"}
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
                    {"worker_id": "local", "path": "root/q", "parent": "root", "kind": "query",
                     "reserved_bytes": 50, "peak_bytes": 60, "limit_bytes": null, "spillable": true}
                ])),
                SystemTable::Caches => rows(json!([
                    {"worker_id": "local", "cache": "metadata", "entries": 3, "bytes": 30, "hits": 3, "misses": 1},
                    {"worker_id": "local", "cache": "block", "entries": 0, "bytes": 0, "hits": 0, "misses": 0}
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
    assert_eq!(run["query_text"], "SELECT 1");
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
        ["q-bad", "q-b", "q-a"],
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
    let (status, ..) = request(addr, "GET", "/api/internal/compare?a=q-a", true).await;
    assert_eq!(status, 400);
    let (status, ..) = request(addr, "GET", "/api/internal/compare?a=q-a&b=zzz", true).await;
    assert_eq!(status, 404);
}

#[tokio::test]
async fn health_and_resources_report_memory_and_cache_hit_rates() {
    let (addr, _) = start(false).await;
    let h = get(addr, "/api/internal/health").await;
    assert_eq!(h["workers"], 1);
    assert_eq!(h["running_queries"], 1);
    assert_eq!(h["queued_queries"], 1);
    assert_eq!(h["memory"]["reserved_bytes"], 100);
    assert_eq!(h["memory"]["limit_bytes"], 1000);
    assert_eq!(h["spill_bytes_total"], 7);
    assert_eq!(h["caches"][0]["hit_rate"], 0.75);
    assert_eq!(
        h["caches"][1]["hit_rate"],
        Value::Null,
        "no lookups yet is not a 0% hit rate"
    );
    let r = get(addr, "/api/internal/resources").await;
    assert_eq!(r["memory"].as_array().unwrap().len(), 2);
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
    let started = std::time::Instant::now();
    let mut seen = String::new();
    let mut buf = [0_u8; 4096];
    // Two events, then stop: they must be at least roughly a second apart.
    while seen.matches("event: queries").count() < 2 {
        let n = tokio::time::timeout(std::time::Duration::from_secs(5), s.read(&mut buf))
            .await
            .expect("no event in 5 s")
            .unwrap();
        assert!(n > 0, "stream closed: {seen}");
        seen.push_str(&String::from_utf8_lossy(&buf[..n]));
    }
    let lower = seen.to_lowercase();
    assert!(lower.contains("content-type: text/event-stream"), "{seen}");
    assert!(seen.contains("\"query_id\":\"q-run\""), "{seen}");
    assert!(
        started.elapsed() >= std::time::Duration::from_millis(900),
        "events came faster than once a second"
    );
}
