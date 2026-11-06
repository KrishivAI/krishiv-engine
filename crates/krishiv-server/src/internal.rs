// SPDX-License-Identifier: Apache-2.0
//! The UI's backend (`/api/internal/*`, P2-T133 part a): JSON and server-sent events over the system tables.
//!
//! The server holds no engine state. Everything it shows comes from a [`UiSource`] (the system tables
//! `queries`, `operators`, `memory`, `caches`, ...), read as JSON rows once per request, so a handler never
//! takes an engine lock beyond the snapshot the source builds (design T133 item 7). Every endpoint needs the
//! start-up token; columns outside each endpoint's allowlist are dropped, so a table that grows a column does not
//! put it on the page by accident.
//!
//! Timestamps are microseconds since the epoch, as the system tables hold them.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, Uri, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures::stream::Stream;
use serde_json::{Map, Value, json};

/// One system-table row as JSON (column name to value).
pub type Row = Map<String, Value>;

/// A boxed future the source returns (object safe, no `async_trait`).
pub type UiFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, UiError>> + Send + 'a>>;

/// The system tables the UI reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemTable {
    /// `system.queries`.
    Queries,
    /// `system.operators`.
    Operators,
    /// `system.memory`.
    Memory,
    /// `system.caches`.
    Caches,
}

/// A source failed to read a table or act on a query. Carries the engine's Spark error class and SQLSTATE
/// when the failure came from the engine, so an operator sees the same class a client would.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UiError {
    /// What went wrong.
    pub message: String,
    /// The Spark error class, when the engine raised one.
    pub error_class: Option<String>,
    /// The SQLSTATE of that class, when Spark defines one.
    pub sql_state: Option<String>,
}

impl UiError {
    /// A failure with no engine error class (a transport or set-up problem).
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            error_class: None,
            sql_state: None,
        }
    }

    /// A failure the engine raised, with its class and SQLSTATE.
    #[must_use]
    pub fn with_class(message: impl Into<String>, class: &str, sql_state: Option<&str>) -> Self {
        Self {
            message: message.into(),
            error_class: Some(class.to_owned()),
            sql_state: sql_state.map(str::to_owned),
        }
    }
}

impl std::fmt::Display for UiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.error_class {
            Some(c) => write!(f, "[{c}] {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for UiError {}

/// Where the UI's data comes from: an engine session in embedded mode, a fake in tests.
pub trait UiSource: Send + Sync + 'static {
    /// Every row of `table`, a snapshot.
    fn rows(&self, table: SystemTable) -> UiFuture<'_, Vec<Row>>;
    /// Cancels the query as an operator's kill (the `admin` reason: the query fails with
    /// `KRISHIV_QUERY_KILLED`); `false` when it is not running.
    fn cancel<'a>(&'a self, query_id: &'a str) -> UiFuture<'a, bool>;
}

#[derive(Clone)]
struct UiState {
    source: Arc<dyn UiSource>,
    token: Arc<str>,
    started: Instant,
}

/// A fresh random token: 32 bytes from the operating system, hex encoded.
///
/// # Errors
/// Returns an error when the system has no random source.
pub fn random_token() -> Result<String, UiError> {
    let mut bytes = [0_u8; 32];
    getrandom::fill(&mut bytes).map_err(|e| UiError::new(format!("no random source: {e}")))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

/// The headers every response of a UI-serving router carries (spec 09 section 5): no framing, no inline
/// script, no third-party content.
const CSP: &str = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; \
connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; object-src 'none'; form-action 'self'";

async fn security_headers(req: axum::extract::Request, next: Next) -> Response {
    let mut res = next.run(req).await;
    let h = res.headers_mut();
    h.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(CSP),
    );
    h.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    h.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    h.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    res
}

/// The admin router with the UI backend: [`crate::router`] plus `/api/internal/*`, behind `token`, with the
/// security headers on every response.
pub fn router_with_ui(source: Arc<dyn UiSource>, token: &str) -> Router {
    let state = UiState {
        source,
        token: token.into(),
        started: Instant::now(),
    };
    let internal = Router::new()
        .route("/api/internal/health", get(health))
        .route("/api/internal/queries", get(queries))
        .route("/api/internal/queries/{id}", get(query_detail))
        .route("/api/internal/queries/{id}/cancel", post(cancel))
        .route("/api/internal/history", get(history))
        .route("/api/internal/compare", get(compare))
        .route("/api/internal/resources", get(resources))
        .route("/api/internal/events", get(events))
        .with_state(state.clone())
        .layer(middleware::from_fn_with_state(state, require_token));
    crate::router()
        .merge(internal)
        .layer(middleware::from_fn(security_headers))
}

fn json_response(status: StatusCode, body: &Value) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

fn error(status: StatusCode, code: &str, message: &str) -> Response {
    json_response(status, &json!({"error": code, "message": message}))
}

fn source_error(e: &UiError) -> Response {
    json_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        &json!({
            "error": "source_failed",
            "message": e.message,
            "error_class": e.error_class,
            "sql_state": e.sql_state,
        }),
    )
}

/// Constant-time equality of two tokens.
fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut diff = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        diff |= usize::from(a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0));
    }
    diff == 0
}

/// A query-string parameter's raw value (no percent-decoding: the parameters here are ids, numbers and names).
fn param<'a>(uri: &'a Uri, name: &str) -> Option<&'a str> {
    uri.query()?
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .find(|(k, _)| *k == name)
        .map(|(_, v)| v)
}

/// The presented token: `Authorization: Bearer`, the `krishiv_ui_token` cookie, or `?token=` (an
/// `EventSource` cannot set headers).
fn presented(headers: &HeaderMap, uri: &Uri) -> Option<String> {
    if let Some(v) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        && let Some(t) = v.strip_prefix("Bearer ")
    {
        return Some(t.trim().to_owned());
    }
    if let Some(c) = headers.get(header::COOKIE).and_then(|v| v.to_str().ok())
        && let Some(t) = c
            .split(';')
            .filter_map(|p| p.trim().strip_prefix("krishiv_ui_token="))
            .next()
    {
        return Some(t.to_owned());
    }
    param(uri, "token").map(str::to_owned)
}

async fn require_token(
    State(s): State<UiState>,
    req: axum::extract::Request,
    next: Next,
) -> Response {
    match presented(req.headers(), req.uri()) {
        Some(t) if same(&t, &s.token) => next.run(req).await,
        _ => error(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "a valid UI token is required",
        ),
    }
}

/// Keeps only the allowlisted columns of a row.
fn pick(row: &Row, columns: &[&str]) -> Row {
    columns
        .iter()
        .filter_map(|c| row.get(*c).map(|v| ((*c).to_owned(), v.clone())))
        .collect()
}

const QUERY_COLUMNS: &[&str] = &[
    "query_id",
    "state",
    "principal",
    "statement_class",
    "query_text",
    "query_text_truncated",
    "submitted_at",
    "ended_at",
    "wall_ns",
    "cpu_ns",
    "peak_memory_bytes",
    "rows_produced",
    "bytes_read",
    "bytes_written",
    "spill_bytes",
    "spill_files",
    "workload_class",
    "error_class",
    "cancel_reason",
    "pipelines",
    "workers",
    "rows_read",
    "files_read",
    "partial",
];

const OPERATOR_COLUMNS: &[&str] = &[
    "query_id",
    "operator_id",
    "node_id",
    "name",
    "role",
    "drivers",
    "rows_in",
    "rows_out",
    "compute_ns",
    "max_driver_compute_ns",
    "blocked_io_ns",
    "blocked_memory_ns",
    "blocked_sink_ns",
    "peak_reserved_bytes",
    "spills",
    "spilled_bytes",
];

const MEMORY_COLUMNS: &[&str] = &[
    "worker_id",
    "path",
    "parent",
    "kind",
    "query_id",
    "reserved_bytes",
    "peak_bytes",
    "limit_bytes",
    "spillable",
];

const CACHE_COLUMNS: &[&str] = &[
    "worker_id",
    "cache",
    "entries",
    "bytes",
    "capacity_bytes",
    "hits",
    "misses",
    "evictions",
    "invalidations",
    "loads_in_flight",
];

fn text<'a>(row: &'a Row, column: &str) -> &'a str {
    row.get(column).and_then(Value::as_str).unwrap_or("")
}

fn int(row: &Row, column: &str) -> i64 {
    row.get(column).and_then(Value::as_i64).unwrap_or(0)
}

fn is_live(row: &Row) -> bool {
    matches!(text(row, "state"), "running" | "queued")
}

fn live_queries(rows: &[Row]) -> Vec<Value> {
    let mut live: Vec<&Row> = rows.iter().filter(|r| is_live(r)).collect();
    live.sort_by_key(|r| int(r, "submitted_at"));
    live.into_iter()
        .map(|r| Value::Object(pick(r, QUERY_COLUMNS)))
        .collect()
}

async fn table(s: &UiState, t: SystemTable) -> Result<Vec<Row>, Box<Response>> {
    s.source
        .rows(t)
        .await
        .map_err(|e| Box::new(source_error(&e)))
}

/// `hits / (hits + misses)`, `null` before the first lookup.
fn hit_rate(row: &Row) -> Value {
    let (h, m) = (int(row, "hits"), int(row, "misses"));
    match h.checked_add(m) {
        None | Some(0) => Value::Null,
        Some(total) => {
            #[expect(clippy::cast_precision_loss, reason = "a ratio for display")]
            let r = h as f64 / total as f64;
            json!(r)
        }
    }
}

/// The memory of every worker's root pool added up. With several workers, `peak_bytes` is the sum of the
/// workers' own peaks (an upper bound of the cluster's peak, since they need not coincide) and `limit_bytes`
/// is `null` unless every root has a limit.
fn memory_total(memory: &[Row]) -> Value {
    let roots: Vec<&Row> = memory
        .iter()
        .filter(|r| r.get("parent").is_none_or(Value::is_null))
        .collect();
    if roots.is_empty() {
        return Value::Null;
    }
    let sum = |column: &str| {
        roots
            .iter()
            .fold(0_i64, |acc, r| acc.saturating_add(int(r, column)))
    };
    let limits: Option<Vec<i64>> = roots
        .iter()
        .map(|r| r.get("limit_bytes").and_then(Value::as_i64))
        .collect();
    json!({
        "reserved_bytes": sum("reserved_bytes"),
        "peak_bytes": sum("peak_bytes"),
        "limit_bytes": limits.map_or(Value::Null, |l| json!(l.iter().fold(0_i64, |a, b| a.saturating_add(*b)))),
    })
}

async fn health(State(s): State<UiState>) -> Response {
    let (memory, caches, queries) = match (
        table(&s, SystemTable::Memory).await,
        table(&s, SystemTable::Caches).await,
        table(&s, SystemTable::Queries).await,
    ) {
        (Ok(m), Ok(c), Ok(q)) => (m, c, q),
        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => return *e,
    };
    let workers: std::collections::BTreeSet<&str> =
        memory.iter().map(|r| text(r, "worker_id")).collect();
    let spill_bytes = queries
        .iter()
        .fold(0_i64, |acc, r| acc.saturating_add(int(r, "spill_bytes")));
    json_response(
        StatusCode::OK,
        &json!({
            "version": crate::VERSION,
            "uptime_seconds": s.started.elapsed().as_secs(),
            "workers": workers.len().max(1),
            "running_queries": queries.iter().filter(|r| text(r, "state") == "running").count(),
            "queued_queries": queries.iter().filter(|r| text(r, "state") == "queued").count(),
            "memory": memory_total(&memory),
            "spill_bytes_total": spill_bytes,
            "caches": caches.iter().map(|r| {
                let mut c = pick(r, &["cache", "entries", "bytes", "capacity_bytes", "hits", "misses"]);
                c.insert("hit_rate".to_owned(), hit_rate(r));
                Value::Object(c)
            }).collect::<Vec<_>>(),
        }),
    )
}

async fn queries(State(s): State<UiState>) -> Response {
    match table(&s, SystemTable::Queries).await {
        Ok(rows) => json_response(StatusCode::OK, &json!({"queries": live_queries(&rows)})),
        Err(e) => *e,
    }
}

/// A byte count for people: `128.0 MiB`, as the pages write them.
fn human_bytes(n: i64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    if n.unsigned_abs() < 1024 {
        return format!("{n} B");
    }
    #[expect(clippy::cast_precision_loss, reason = "a size for display")]
    let mut v = n as f64;
    let mut i = 0;
    while v.abs() >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.1} {}", UNITS[i])
}

/// A plain-language hint per operator that spilled, was skewed, or waited on memory.
fn warnings(operators: &[Row]) -> Vec<Value> {
    let mut out = Vec::new();
    for op in operators {
        let name = text(op, "name");
        let spilled = int(op, "spilled_bytes");
        if spilled > 0 {
            out.push(json!({
                "operator_id": int(op, "operator_id"), "operator": name, "kind": "spill",
                "hint": format!("{name} spilled {} to disk. Give the query more memory, or reduce the rows it holds at once (a filter or an earlier aggregation).", human_bytes(spilled)),
            }));
        }
        let (drivers, total, slowest) = (
            int(op, "drivers"),
            int(op, "compute_ns"),
            int(op, "max_driver_compute_ns"),
        );
        // One driver doing more than twice the average share of the work: the data is skewed.
        if drivers > 1 && total > 0 && slowest.saturating_mul(drivers) > total.saturating_mul(2) {
            out.push(json!({
                "operator_id": int(op, "operator_id"), "operator": name, "kind": "skew",
                "hint": format!("One of {drivers} parallel drivers of {name} did most of the work, so the others waited. A few keys probably hold most of the rows; check the join or group-by keys for skew."),
            }));
        }
        if int(op, "blocked_memory_ns") > int(op, "compute_ns") && int(op, "blocked_memory_ns") > 0
        {
            out.push(json!({
                "operator_id": int(op, "operator_id"), "operator": name, "kind": "memory",
                "hint": format!("{name} spent more time waiting for memory than computing. The memory limit is too small for this query."),
            }));
        }
    }
    out
}

/// The operator with the most compute time, `null` when there is none.
fn hot_operator(operators: &[Row]) -> Value {
    operators
        .iter()
        .max_by_key(|r| int(r, "compute_ns"))
        .map_or(Value::Null, |r| {
            json!({"operator_id": int(r, "operator_id"), "operator": text(r, "name"), "compute_ns": int(r, "compute_ns")})
        })
}

async fn query_detail(State(s): State<UiState>, Path(id): Path<String>) -> Response {
    let (queries, operators) = match (
        table(&s, SystemTable::Queries).await,
        table(&s, SystemTable::Operators).await,
    ) {
        (Ok(q), Ok(o)) => (q, o),
        (Err(e), _) | (_, Err(e)) => return *e,
    };
    let Some(query) = queries.iter().find(|r| text(r, "query_id") == id) else {
        return error(
            StatusCode::NOT_FOUND,
            "no_such_query",
            "no query with that id",
        );
    };
    let mut ops: Vec<Row> = operators
        .into_iter()
        .filter(|r| text(r, "query_id") == id)
        .collect();
    ops.sort_by_key(|r| int(r, "operator_id"));
    json_response(
        StatusCode::OK,
        &json!({
            "query": pick(query, QUERY_COLUMNS),
            "operators": ops.iter().map(|r| Value::Object(pick(r, OPERATOR_COLUMNS))).collect::<Vec<_>>(),
            "hot_operator": hot_operator(&ops),
            "warnings": warnings(&ops),
        }),
    )
}

async fn cancel(State(s): State<UiState>, Path(id): Path<String>) -> Response {
    match s.source.cancel(&id).await {
        Ok(true) => json_response(StatusCode::OK, &json!({"cancelled": true})),
        Ok(false) => error(
            StatusCode::NOT_FOUND,
            "not_running",
            "that query is not running",
        ),
        Err(e) => source_error(&e),
    }
}

async fn history(State(s): State<UiState>, uri: Uri) -> Response {
    let rows = match table(&s, SystemTable::Queries).await {
        Ok(r) => r,
        Err(e) => return *e,
    };
    let limit = param(&uri, "limit")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(100)
        .clamp(1, 1000);
    let state = param(&uri, "state");
    let principal = param(&uri, "principal");
    let mut done: Vec<&Row> = rows
        .iter()
        .filter(|r| !is_live(r))
        .filter(|r| state.is_none_or(|s| text(r, "state") == s))
        .filter(|r| principal.is_none_or(|p| text(r, "principal") == p))
        .collect();
    done.sort_by_key(|r| std::cmp::Reverse(int(r, "submitted_at")));
    done.truncate(limit);
    json_response(
        StatusCode::OK,
        &json!({"queries": done.into_iter().map(|r| Value::Object(pick(r, QUERY_COLUMNS))).collect::<Vec<_>>()}),
    )
}

/// Two runs side by side with the change in each number (`b - a`).
async fn compare(State(s): State<UiState>, uri: Uri) -> Response {
    let (Some(a), Some(b)) = (param(&uri, "a"), param(&uri, "b")) else {
        return error(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "compare needs ?a=<query_id>&b=<query_id>",
        );
    };
    let rows = match table(&s, SystemTable::Queries).await {
        Ok(r) => r,
        Err(e) => return *e,
    };
    let find = |id: &str| rows.iter().find(|r| text(r, "query_id") == id);
    let (Some(ra), Some(rb)) = (find(a), find(b)) else {
        return error(
            StatusCode::NOT_FOUND,
            "no_such_query",
            "one of the queries is not in the history",
        );
    };
    let mut delta = BTreeMap::new();
    for c in [
        "wall_ns",
        "cpu_ns",
        "peak_memory_bytes",
        "rows_produced",
        "bytes_read",
        "spill_bytes",
    ] {
        if let (Some(x), Some(y)) = (
            ra.get(c).and_then(Value::as_i64),
            rb.get(c).and_then(Value::as_i64),
        ) && let Some(d) = y.checked_sub(x)
        {
            delta.insert(c, d);
        }
    }
    json_response(
        StatusCode::OK,
        &json!({"a": pick(ra, QUERY_COLUMNS), "b": pick(rb, QUERY_COLUMNS), "delta": delta}),
    )
}

async fn resources(State(s): State<UiState>) -> Response {
    let (memory, caches) = match (
        table(&s, SystemTable::Memory).await,
        table(&s, SystemTable::Caches).await,
    ) {
        (Ok(m), Ok(c)) => (m, c),
        (Err(e), _) | (_, Err(e)) => return *e,
    };
    json_response(
        StatusCode::OK,
        &json!({
            "memory": memory.iter().map(|r| Value::Object(pick(r, MEMORY_COLUMNS))).collect::<Vec<_>>(),
            "caches": caches.iter().map(|r| {
                let mut c = pick(r, CACHE_COLUMNS);
                c.insert("hit_rate".to_owned(), hit_rate(r));
                Value::Object(c)
            }).collect::<Vec<_>>(),
        }),
    )
}

/// The refresh period of the event stream: one event a second per client at most (design T133 item 7).
const EVENT_PERIOD: Duration = Duration::from_secs(1);

/// `GET /api/internal/events`: the live query list as an `queries` event, once a second.
async fn events(
    State(s): State<UiState>,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let ticker = tokio::time::interval(EVENT_PERIOD);
    let stream = futures::stream::unfold((ticker, s), |(mut ticker, s)| async move {
        ticker.tick().await;
        let body = match s.source.rows(SystemTable::Queries).await {
            Ok(rows) => json!({"queries": live_queries(&rows)}),
            Err(e) => json!({"error": e.message, "error_class": e.error_class}),
        };
        let event = Event::default().event("queries").data(body.to_string());
        // A slow snapshot must not make the next ticks fire back to back: the period restarts here.
        ticker.reset();
        Some((Ok(event), (ticker, s)))
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[cfg(test)]
mod tests {
    use super::human_bytes;

    #[test]
    fn byte_sizes_read_like_the_pages_write_them() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(1023), "1023 B");
        assert_eq!(human_bytes(1024), "1.0 KiB");
        assert_eq!(human_bytes(1536), "1.5 KiB");
        assert_eq!(human_bytes(134_217_728), "128.0 MiB");
        assert_eq!(human_bytes(-2048), "-2.0 KiB");
        assert_eq!(human_bytes(i64::MAX), "8388608.0 TiB");
    }
}
