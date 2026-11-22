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
    closed: Mutex<Vec<String>>,
    fail: bool,
    /// Two sessions each run a query under the id `q-dup`.
    dup: bool,
    /// How many times the live-query table was read.
    reads: std::sync::atomic::AtomicUsize,
}

impl UiSource for Fake {
    fn rows(&self, table: SystemTable) -> UiFuture<'_, Vec<Row>> {
        let (fail, dup) = (self.fail, self.dup);
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
            if dup && table == SystemTable::Queries {
                return Ok(rows(json!([
                    {"query_id": "q-dup", "state": "running", "principal": "ann", "submitted_at": 10, "session_id": "s-a",
                     "query_text": "SELECT a FROM t"},
                    {"query_id": "q-dup", "state": "running", "principal": "bob", "submitted_at": 20, "session_id": "s-b",
                     "query_text": "SELECT b FROM t"},
                    {"query_id": "q-solo", "state": "running", "principal": "bob", "submitted_at": 30, "session_id": "s-b"},
                    {"query_id": "q-old", "state": "finished", "principal": "ann", "submitted_at": 1, "session_id": "s-a"},
                    {"query_id": "q-old", "state": "running", "principal": "cy", "submitted_at": 40, "session_id": "s-c"}
                ])));
            }
            Ok(match table {
                SystemTable::Queries => rows(json!([
                    {"query_id": "q-run", "state": "running", "principal": "ann", "submitted_at": 2_000, "session_id": "s-busy",
                     "query_text": "SELECT 1 FROM t WHERE pw = 's3cr3t-canary'", "secret_option": "hunter2", "workers": 1},
                    {"query_id": "q-wait", "state": "queued", "principal": "bob", "submitted_at": 1_000},
                    {"query_id": "q-a", "state": "finished", "principal": "ann", "submitted_at": 100, "session_id": "s-old",
                     "client_type": "_SPARK_CONNECT_PYTHON spark/4.2.0", "tags": {"team": "risk", "nightly": ""},
                     "query_text": "SELECT sum(amount) FROM Orders WHERE region = 'emea-canary'",
                     "wall_ns": 1_000, "peak_memory_bytes": 10, "rows_produced": 5, "spill_bytes": 0},
                    {"query_id": "q-b", "state": "finished", "principal": "ann", "submitted_at": 200, "session_id": "s-old",
                     "client_type": "krishiv-cli", "tags": {"team": "ads", "krishiv-ui-run-7": ""},
                     "query_text": "SELECT count(*) FROM clicks",
                     "wall_ns": 1_500, "peak_memory_bytes": 30, "rows_produced": 5, "spill_bytes": 7,
                     "hints": [{"code": "SPILL", "node_id": 2, "operator_id": 2, "message": "HashJoin spilled 4.0 KiB"}]},
                    {"query_id": "q-bad", "state": "failed", "principal": "bob@example.com", "submitted_at": 300,
                     "client_type": null, "tags": {}, "error_class": "DIVIDE_BY_ZERO"},
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
                SystemTable::Sessions => rows(json!([
                    {"session_id": "s-old", "principal": "ann", "state": "closed", "transport": "connect",
                     "created_at": 10, "last_active_at": 900, "closed_at": 950, "close_reason": "release_session",
                     "tags": {}, "queries_running": 0, "queries_started": 4},
                    {"session_id": "s-idle", "principal": "bob", "state": "active", "transport": "connect",
                     "client_type": "krishiv-cli", "created_at": 20, "last_active_at": 100,
                     "tags": {"team": "risk", "krishiv-ui-run-3": ""}, "queries_running": 0, "queries_started": 1,
                     "current_catalog": "spark_catalog", "current_namespace": "default", "secret_option": "hunter2"},
                    {"session_id": "s-busy", "principal": "ann", "state": "active", "transport": "flight_sql",
                     "created_at": 30, "last_active_at": 800, "tags": {}, "queries_running": 2, "queries_started": 9},
                    {"session_id": "s-gone", "principal": "cy", "state": "expired", "transport": "connect",
                     "created_at": 5, "last_active_at": 990, "closed_at": 995, "tags": {}, "queries_running": 0,
                     "queries_started": 0}
                ])),
                SystemTable::ResourceGroups => rows(json!([
                    {"group_id": "root", "parent_id": null, "state": "active", "scheduling_policy": "weighted_fair",
                     "scheduling_weight": 1, "priority": 0, "workload_class": "interactive",
                     "hard_concurrency_limit": 8, "max_queued": 100, "soft_memory_limit_bytes": 1000,
                     "running": 3, "queued": 1, "memory_reserved_bytes": 400, "internal_note": "hunter2"},
                    {"group_id": "root.etl", "parent_id": "root", "state": "active",
                     "scheduling_policy": "query_priority", "scheduling_weight": 2, "priority": 1,
                     "workload_class": "batch", "hard_concurrency_limit": 2, "max_queued": 10,
                     "running": 2, "queued": 1, "memory_reserved_bytes": 300},
                    {"group_id": "root.adhoc", "parent_id": "root", "state": "removed",
                     "scheduling_policy": "weighted_fair", "scheduling_weight": 1, "priority": 0,
                     "workload_class": "interactive", "hard_concurrency_limit": 4, "max_queued": 10,
                     "running": 1, "queued": 0, "memory_reserved_bytes": 100}
                ])),
                // As the table lists them: the best suggestion of each table first.
                SystemTable::PartitionAdvice => rows(json!([
                    {"table_name": "spark_catalog.default.ev", "rank": 1, "column_name": "d", "table_queries": 120,
                     "column_queries": 100, "partitions": 365, "pruned_fraction": 0.9973, "internal_note": "hunter2"},
                    {"table_name": "spark_catalog.sales.orders", "rank": 1, "column_name": "region",
                     "table_queries": 40, "column_queries": 30, "partitions": 12, "pruned_fraction": 0.9167},
                    {"table_name": "spark_catalog.default.ev", "rank": 2, "column_name": "c", "table_queries": 120,
                     "column_queries": 20, "partitions": 3, "pruned_fraction": 0.6667}
                ])),
            })
        })
    }

    fn close_session<'a>(&'a self, session_id: &'a str) -> UiFuture<'a, Option<usize>> {
        Box::pin(async move {
            self.closed.lock().unwrap().push(session_id.to_owned());
            if self.fail {
                return Err(UiError::new("the engine is unavailable"));
            }
            Ok((session_id == "s-busy").then_some(2))
        })
    }

    fn cancel<'a>(&'a self, query_id: &'a str) -> UiFuture<'a, bool> {
        Box::pin(async move {
            self.cancelled.lock().unwrap().push(query_id.to_owned());
            Ok(matches!(query_id, "q-run" | "q-solo" | "q-old"))
        })
    }

    fn cancel_in_session<'a>(
        &'a self,
        query_id: &'a str,
        session_id: &'a str,
    ) -> UiFuture<'a, bool> {
        Box::pin(async move {
            self.cancelled
                .lock()
                .unwrap()
                .push(format!("{query_id}@{session_id}"));
            if self.fail {
                return Err(UiError::new("the engine is unavailable"));
            }
            Ok(query_id == "q-dup" && matches!(session_id, "s-a" | "s-b"))
        })
    }
}

async fn start(fail: bool) -> (std::net::SocketAddr, Arc<Fake>) {
    start_with(fail, false).await
}

/// A source where two sessions each run a query under one id.
async fn start_dup() -> (std::net::SocketAddr, Arc<Fake>) {
    start_with(false, true).await
}

async fn start_with(fail: bool, dup: bool) -> (std::net::SocketAddr, Arc<Fake>) {
    let fake = Arc::new(Fake {
        cancelled: Mutex::new(Vec::new()),
        closed: Mutex::new(Vec::new()),
        fail,
        dup,
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
        "/api/internal/sessions",
        "/api/internal/groups",
        "/api/internal/advice",
    ] {
        let (status, _, body) = request(addr, "GET", path, false).await;
        assert_eq!(status, 401, "{path}");
        assert_eq!(body["error"], "unauthorized");
    }
    // The token also works as a cookie and as a query parameter (an EventSource cannot set headers).
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(format!("GET /api/internal/queries?token={TOKEN} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n").as_bytes())
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
            format!("GET /api/internal/queries HTTP/1.1\r\nHost: localhost\r\nCookie: {cookie}\r\nConnection: close\r\n\r\n").as_bytes(),
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
async fn a_query_detail_names_the_hot_operator_and_carries_the_advisor_hints() {
    let (addr, _) = start(false).await;
    let body = get(addr, "/api/internal/queries/q-b").await;
    assert_eq!(body["hot_operator"]["operator"], "HashJoin");
    let ops = body["operators"].as_array().unwrap();
    assert_eq!(ops.len(), 3, "only this query's operators");
    assert!(
        body.get("warnings").is_none(),
        "the advisor hints replace the old warnings"
    );
    assert_eq!(
        body["query"]["hints"][0]["code"], "SPILL",
        "the advisor's hints reach the query page as system.queries lists them"
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

/// The server console's filters (ADR 0024): by user, by client type and by tag, as the page sends them.
#[tokio::test]
async fn history_filters_by_user_client_type_and_tag() {
    let (addr, _) = start(false).await;
    let ids = |path: &'static str| async move {
        let body = get(addr, path).await;
        body["queries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| q["query_id"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>()
    };
    // A name with a character the page escapes.
    assert_eq!(
        ids("/api/internal/history?principal=bob%40example.com").await,
        ["q-bad"]
    );
    // Any part of the client's name, in either case; a query with no client type matches no client.
    assert_eq!(ids("/api/internal/history?client=python").await, ["q-a"]);
    assert_eq!(
        ids("/api/internal/history?client=krishiv-CLI").await,
        ["q-b"]
    );
    assert_eq!(
        ids("/api/internal/history?client=spark+connect").await,
        Vec::<String>::new(),
        "a space is a space, not an underscore"
    );
    // A tag by its key, or by key and value; a request tag is a key with no value.
    assert_eq!(ids("/api/internal/history?tag=team").await, ["q-b", "q-a"]);
    assert_eq!(ids("/api/internal/history?tag=team%3Drisk").await, ["q-a"]);
    assert_eq!(ids("/api/internal/history?tag=nightly").await, ["q-a"]);
    assert_eq!(
        ids("/api/internal/history?tag=team%3D").await,
        Vec::<String>::new()
    );
    assert_eq!(
        ids("/api/internal/history?tag=team&client=cli&state=finished").await,
        ["q-b"],
        "filters combine"
    );
    // An empty filter is no filter, and a value that does not decode is none either.
    assert_eq!(ids("/api/internal/history?tag=&client=").await.len(), 5);
    assert_eq!(ids("/api/internal/history?tag=%zz").await.len(), 5);
    // The row carries what was filtered on.
    let row = &get(addr, "/api/internal/history?tag=nightly").await["queries"][0];
    assert_eq!(row["client_type"], "_SPARK_CONNECT_PYTHON spark/4.2.0");
    assert_eq!(row["tags"]["team"], "risk");
    // The tag the editor gives its own run is not sent.
    let from_editor = &get(addr, "/api/internal/history?client=cli").await["queries"][0];
    assert_eq!(from_editor["tags"], json!({"team": "ads"}));
}

#[tokio::test]
async fn history_filters_by_a_part_of_the_statement_as_it_is_shown() {
    let (addr, _) = start(false).await;
    let ids = |path: &'static str| async move {
        let body = get(addr, path).await;
        body["queries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| q["query_id"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>()
    };
    // Any part of the statement, in either case; a space arrives as `+` or `%20`.
    assert_eq!(ids("/api/internal/history?text=orders").await, ["q-a"]);
    assert_eq!(ids("/api/internal/history?text=FROM+ORDERS").await, ["q-a"]);
    assert_eq!(
        ids("/api/internal/history?text=select").await,
        ["q-b", "q-a"]
    );
    assert_eq!(
        ids("/api/internal/history?text=count(*)%20from").await,
        ["q-b"]
    );
    // A query whose text was not recorded matches no text.
    assert_eq!(
        ids("/api/internal/history?text=divide").await,
        Vec::<String>::new()
    );
    // The literal is masked in the list, so it finds nothing: a search must not tell which literal a statement
    // holds. The mask itself is what the list shows, and is found.
    let shown = &get(addr, "/api/internal/history?text=orders").await["queries"][0];
    assert_eq!(
        shown["query_text"],
        "SELECT sum(amount) FROM Orders WHERE region = ?"
    );
    assert_eq!(
        ids("/api/internal/history?text=emea").await,
        Vec::<String>::new()
    );
    assert_eq!(
        ids("/api/internal/history?text=region+%3D+%3F").await,
        ["q-a"]
    );
    // It combines with the other filters; empty or undecodable, it is no filter.
    assert_eq!(
        ids("/api/internal/history?text=select&tag=nightly").await,
        ["q-a"]
    );
    assert_eq!(
        ids("/api/internal/history?text=select&client=cli").await,
        ["q-b"]
    );
    assert_eq!(ids("/api/internal/history?text=").await.len(), 5);
    assert_eq!(ids("/api/internal/history?text=%zz").await.len(), 5);
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
            "GET /api/internal/events HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {TOKEN}\r\n\r\n"
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
        s.write_all(format!("POST {path} HTTP/1.1\r\nHost: localhost\r\n{extra}Content-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).await.unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).await.unwrap();
        assert!(out.starts_with("HTTP/1.1 401"), "{name}: {out}");
    }
    assert!(
        fake.cancelled.lock().unwrap().is_empty(),
        "nothing was cancelled"
    );
    for (name, extra, path) in [
        ("no token", "", "/api/internal/sessions/s-busy/close"),
        (
            "token in the address",
            "",
            &format!("/api/internal/sessions/s-busy/close?token={TOKEN}"),
        ),
        (
            "token in a cookie",
            &format!("Cookie: krishiv_ui_token={TOKEN}\r\n"),
            "/api/internal/sessions/s-busy/close",
        ),
    ] {
        let mut s = TcpStream::connect(addr).await.unwrap();
        s.write_all(format!("POST {path} HTTP/1.1\r\nHost: localhost\r\n{extra}Content-Length: 0\r\nConnection: close\r\n\r\n").as_bytes()).await.unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).await.unwrap();
        assert!(out.starts_with("HTTP/1.1 401"), "{name}: {out}");
    }
    assert!(
        fake.closed.lock().unwrap().is_empty(),
        "no session was ended"
    );
}

async fn open_stream(addr: std::net::SocketAddr) -> TcpStream {
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(
        format!(
            "GET /api/internal/events HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {TOKEN}\r\n\r\n"
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

#[tokio::test]
async fn sessions_list_the_open_ones_first_by_last_activity_with_only_allowlisted_columns() {
    let (addr, _) = start(false).await;
    let body = get(addr, "/api/internal/sessions").await;
    let ids: Vec<&str> = body["sessions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["session_id"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        ["s-busy", "s-idle", "s-gone", "s-old"],
        "open first, then ended; each part by its last activity, newest first"
    );
    assert_eq!(body["truncated"], false);
    let idle = &body["sessions"][1];
    assert_eq!(idle["client_type"], "krishiv-cli");
    assert_eq!(idle["current_namespace"], "default");
    assert_eq!(
        idle["tags"],
        json!({"team": "risk"}),
        "the editor's own tag is the console's bookkeeping"
    );
    assert!(
        idle.get("secret_option").is_none(),
        "a column outside the allowlist does not reach the page"
    );
    assert!(!body.to_string().contains("hunter2"));
    assert_eq!(body["sessions"][3]["close_reason"], "release_session");
}

#[tokio::test]
async fn groups_keep_the_tables_order_and_only_allowlisted_columns() {
    let (addr, _) = start(false).await;
    let body = get(addr, "/api/internal/groups").await;
    let groups = body["groups"].as_array().unwrap();
    let ids: Vec<&str> = groups
        .iter()
        .map(|g| g["group_id"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        ["root", "root.etl", "root.adhoc"],
        "a parent before its children, as the table lists them (not sorted by name)"
    );
    assert_eq!(groups[0]["hard_concurrency_limit"], 8);
    assert_eq!(groups[0]["running"], 3);
    assert_eq!(groups[1]["parent_id"], "root");
    assert_eq!(groups[2]["state"], "removed");
    assert!(!body.to_string().contains("hunter2"));
}

#[tokio::test]
async fn closing_a_session_reaches_the_source_and_reports_what_was_cancelled() {
    let (addr, fake) = start(false).await;
    let (status, _, body) =
        request(addr, "POST", "/api/internal/sessions/s-busy/close", true).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body, json!({"closed": true, "queries_cancelled": 2}));
    let (status, _, body) = request(addr, "POST", "/api/internal/sessions/s-old/close", true).await;
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["error"], "not_open");
    assert_eq!(*fake.closed.lock().unwrap(), ["s-busy", "s-old"]);
    // Reading is not closing.
    let (status, ..) = request(addr, "GET", "/api/internal/sessions/s-busy/close", true).await;
    assert_eq!(status, 405);
    assert_eq!(fake.closed.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn a_failing_source_fails_the_session_and_group_routes_as_json() {
    let (addr, _) = start(true).await;
    for path in [
        "/api/internal/sessions",
        "/api/internal/groups",
        "/api/internal/advice",
    ] {
        let (status, _, body) = request(addr, "GET", path, true).await;
        assert_eq!(status, 500, "{path}");
        assert_eq!(body["error"], "source_failed", "{path}");
    }
    let (status, _, body) =
        request(addr, "POST", "/api/internal/sessions/s-busy/close", true).await;
    assert_eq!(status, 500);
    assert_eq!(body["error"], "source_failed");
}

#[tokio::test]
async fn the_lists_name_a_session_and_history_filters_by_it_but_the_query_page_does_not() {
    let (addr, _) = start(false).await;
    let live = get(addr, "/api/internal/queries").await;
    assert_eq!(live["queries"][1]["query_id"], "q-run");
    assert_eq!(live["queries"][1]["session_id"], "s-busy");
    let of = get(addr, "/api/internal/history?session=s-old").await;
    let ids: Vec<&str> = of["queries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| q["query_id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["q-b", "q-a"]);
    let none = get(addr, "/api/internal/history?session=nobody").await;
    assert_eq!(none["queries"], json!([]));
    // A signed embed link reads a query's page with no session: the id of the client's session stays out.
    let detail = get(addr, "/api/internal/queries/q-b").await;
    assert!(detail["query"].get("session_id").is_none(), "{detail}");
}

#[tokio::test]
async fn advice_is_grouped_by_table_in_the_tables_order_with_only_allowlisted_columns() {
    let (addr, _) = start(false).await;
    let body = get(addr, "/api/internal/advice").await;
    assert_eq!(body["truncated"], false);
    let tables = body["tables"].as_array().unwrap();
    let names: Vec<&str> = tables
        .iter()
        .map(|t| t["table_name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        ["spark_catalog.default.ev", "spark_catalog.sales.orders"],
        "one entry a table, in the order the table first names each"
    );
    assert_eq!(tables[0]["table_queries"], 120);
    assert_eq!(
        tables[0]["columns"],
        json!([
            {"rank": 1, "column_name": "d", "column_queries": 100, "partitions": 365, "pruned_fraction": 0.9973},
            {"rank": 2, "column_name": "c", "column_queries": 20, "partitions": 3, "pruned_fraction": 0.6667}
        ]),
        "a table's suggestions by rank, however far apart the table lists them"
    );
    assert_eq!(tables[1]["columns"].as_array().unwrap().len(), 1);
    assert!(!body.to_string().contains("hunter2"));
    // Advice is read, never applied.
    let (status, ..) = request(addr, "POST", "/api/internal/advice", true).await;
    assert_eq!(status, 405);
}

#[tokio::test]
async fn a_query_page_can_name_the_session_of_a_query_whose_id_two_sessions_share() {
    let (addr, _) = start_dup().await;
    let of_a = get(addr, "/api/internal/queries/q-dup?session=s-a").await;
    assert_eq!(of_a["query"]["principal"], "ann", "{of_a}");
    let of_b = get(addr, "/api/internal/queries/q-dup?session=s-b").await;
    assert_eq!(of_b["query"]["principal"], "bob", "{of_b}");
    assert_eq!(of_b["query"]["query_text"], "SELECT b FROM t");
    // Without a session the first of that id answers, as before; a session that has no such query is a 404.
    let first = get(addr, "/api/internal/queries/q-dup").await;
    assert_eq!(first["query"]["principal"], "ann");
    for path in [
        "/api/internal/queries/q-dup?session=s-c",
        "/api/internal/queries/q-solo?session=s-a",
    ] {
        let (status, _, body) = request(addr, "GET", path, true).await;
        assert_eq!(status, 404, "{path}: {body}");
        assert_eq!(body["error"], "no_such_query");
    }
}

#[tokio::test]
async fn compare_can_name_the_session_of_each_run() {
    let (addr, _) = start_dup().await;
    let path = "/api/internal/compare?a=q-dup&a_session=s-b&b=q-dup&b_session=s-a";
    let body = get(addr, path).await;
    assert_eq!(body["a"]["principal"], "bob", "{body}");
    assert_eq!(body["b"]["principal"], "ann", "{body}");
    // Without a session the first run of that id answers, as before.
    let first = get(addr, "/api/internal/compare?a=q-dup&b=q-solo").await;
    assert_eq!(first["a"]["principal"], "ann", "{first}");
    // A session that has no such run is not in the history.
    let path = "/api/internal/compare?a=q-dup&a_session=s-c&b=q-solo";
    let (status, _, body) = request(addr, "GET", path, true).await;
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["error"], "no_such_query");
}

#[tokio::test]
async fn cancel_ends_only_the_named_sessions_query_and_refuses_an_id_two_sessions_run() {
    let (addr, fake) = start_dup().await;
    // An id two sessions are running, with no session named: refused before the source is asked.
    let (status, _, body) = request(addr, "POST", "/api/internal/queries/q-dup/cancel", true).await;
    assert_eq!(status, 409, "{body}");
    assert_eq!(body["error"], "ambiguous_query");
    assert!(fake.cancelled.lock().unwrap().is_empty());

    let path = "/api/internal/queries/q-dup/cancel?session=s-b";
    let (status, _, body) = request(addr, "POST", path, true).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body, json!({"cancelled": true}));
    let path = "/api/internal/queries/q-dup/cancel?session=s-c";
    let (status, _, body) = request(addr, "POST", path, true).await;
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["error"], "not_running");

    // An id only one session is running needs no session: an ended query of another session under the
    // same id does not make it ambiguous.
    for id in ["q-solo", "q-old"] {
        let path = format!("/api/internal/queries/{id}/cancel");
        let (status, _, body) = request(addr, "POST", &path, true).await;
        assert_eq!(status, 200, "{id}: {body}");
    }
    assert_eq!(
        *fake.cancelled.lock().unwrap(),
        ["q-dup@s-b", "q-dup@s-c", "q-solo", "q-old"]
    );
}

#[tokio::test]
async fn a_session_qualifier_that_is_empty_or_does_not_decode_is_refused_not_dropped() {
    let (addr, fake) = start_dup().await;
    // `%FF` is no UTF-8 text. Dropping the qualifier would end, show or compare the first query of that id.
    for (method, path) in [
        ("POST", "/api/internal/queries/q-solo/cancel?session=%FF"),
        ("POST", "/api/internal/queries/q-solo/cancel?session="),
        ("GET", "/api/internal/queries/q-dup?session=%FF"),
        ("GET", "/api/internal/queries/q-dup?session="),
        (
            "GET",
            "/api/internal/compare?a=q-dup&a_session=%FF&b=q-solo",
        ),
        (
            "GET",
            "/api/internal/compare?a=q-dup&b=q-solo&b_session=%zz",
        ),
        // A key with no `=` names an empty session too.
        ("POST", "/api/internal/queries/q-solo/cancel?session"),
        ("GET", "/api/internal/queries/q-dup?session"),
        ("GET", "/api/internal/compare?a=q-dup&b=q-solo&a_session"),
        ("GET", "/api/internal/queries/q-dup/embed-link?session=%FF"),
        ("GET", "/api/internal/queries/q-dup/embed-link?session"),
        // The history's filter: not decodable is refused, where dropping it would list every session's.
        ("GET", "/api/internal/history?session=%FF"),
    ] {
        let (status, _, body) = request(addr, method, path, true).await;
        assert_eq!(status, 400, "{path}: {body}");
        assert_eq!(body["error"], "bad_request", "{path}");
    }
    assert!(fake.cancelled.lock().unwrap().is_empty());
}

/// An empty history filter is no filter, as the page sends it when the box is cleared.
#[tokio::test]
async fn an_empty_session_filter_lists_the_history_of_every_session() {
    let (addr, _) = start_dup().await;
    let all = get(addr, "/api/internal/history").await;
    assert_eq!(get(addr, "/api/internal/history?session=").await, all);
}

/// A signed link issued for one session's query opens that query, and no other reading of the same link does:
/// not the other session's query of the id, not the id alone.
#[tokio::test]
async fn a_signed_link_opens_the_query_of_the_session_it_names() {
    let (addr, _) = start_dup().await;
    let link = |session: String| async move {
        let path = format!("/api/internal/queries/q-dup/embed-link?ttl=60{session}");
        let (status, _, body) = request(addr, "GET", &path, true).await;
        assert_eq!(status, 200, "{body}");
        let url = body["url"].as_str().unwrap().to_owned();
        (url.split_once('?').unwrap().1.to_owned(), body)
    };
    let open = |query: String| async move {
        let path = format!("/api/internal/embed/queries/q-dup?{query}");
        request(addr, "GET", &path, false).await
    };
    for (session, principal) in [("s-a", "ann"), ("s-b", "bob")] {
        let (query, issued) = link(format!("&session={session}")).await;
        assert_eq!(issued["session_id"], session);
        assert!(query.ends_with(&format!("&session={session}")), "{query}");
        let (status, _, body) = open(query.clone()).await;
        assert_eq!(status, 200, "{body}");
        assert_eq!(body["query"]["principal"], principal, "{body}");
        // The same signature under another session, or under none, is not a link.
        let other = if session == "s-a" { "s-b" } else { "s-a" };
        let swapped = query.replace(&format!("session={session}"), &format!("session={other}"));
        assert_eq!(open(swapped).await.0, 401);
        let stripped = query.replace(&format!("&session={session}"), "");
        assert_eq!(open(stripped).await.0, 401);
    }
    // A link for the id alone opens the first query of the id, and gains no session afterwards.
    let (query, issued) = link(String::new()).await;
    assert_eq!(issued["session_id"], Value::Null);
    assert!(!query.contains("session"), "{query}");
    let (status, _, body) = open(query.clone()).await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["query"]["principal"], "ann", "{body}");
    assert_eq!(open(format!("{query}&session=s-b")).await.0, 401);
    assert_eq!(open(format!("{query}&session=%FF")).await.0, 401);
    assert_eq!(open(format!("{query}&session=")).await.0, 401, "empty");
    // A link for a session with no query of the id is signed (nothing is read to issue it) and finds nothing.
    let (query, _) = link("&session=s-c".to_owned()).await;
    assert_eq!(open(query).await.0, 404);
}

#[tokio::test]
async fn a_failing_source_fails_a_cancel_as_json_with_or_without_a_session() {
    let (addr, fake) = start(true).await;
    // Without a session the running queries are read first, and that read fails.
    let (status, _, body) = request(addr, "POST", "/api/internal/queries/q-run/cancel", true).await;
    assert_eq!(status, 500, "{body}");
    assert_eq!(body["error"], "source_failed");
    assert!(fake.cancelled.lock().unwrap().is_empty());
    let path = "/api/internal/queries/q-run/cancel?session=s-busy";
    let (status, _, body) = request(addr, "POST", path, true).await;
    assert_eq!(status, 500, "{body}");
    assert_eq!(body["error"], "source_failed");
}

/// A source that implements only what the trait requires.
struct Plain;

impl UiSource for Plain {
    fn rows(&self, _table: SystemTable) -> UiFuture<'_, Vec<Row>> {
        Box::pin(async { Ok(Vec::new()) })
    }

    fn cancel<'a>(&'a self, _query_id: &'a str) -> UiFuture<'a, bool> {
        Box::pin(async { Ok(true) })
    }
}

#[tokio::test]
async fn a_source_that_does_not_tell_sessions_apart_cancels_nothing_in_a_named_session() {
    // The default is not the unqualified cancel: that would end another session's query of the same id.
    assert!(Plain.cancel("q").await.unwrap());
    assert!(!Plain.cancel_in_session("q", "s").await.unwrap());
}
