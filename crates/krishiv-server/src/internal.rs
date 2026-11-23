// SPDX-License-Identifier: Apache-2.0
//! The UI's backend (`/api/internal/*`, P2-T133 part a): JSON and server-sent events over the system tables.
//!
//! The server holds no engine state. Everything it shows comes from a [`UiSource`] (the system tables
//! `queries`, `operators`, `memory`, `caches`, ...), read as JSON rows once per request, so a handler never
//! takes an engine lock beyond the snapshot the source builds (design T133 item 7). Every endpoint asks the
//! router's [`Guard`] (the start-up token, or the server's authentication); columns outside each endpoint's allowlist are dropped, so a table that grows a column does not
//! put it on the page by accident.
//!
//! Timestamps are microseconds since the epoch, as the system tables hold them.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use axum::Extension;
use axum::Router;
use axum::extract::{Path, RawQuery, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, Uri, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Map, Value, json};

use crate::guard::{
    Access, Caller, Guard, GuardRequest, GuardSurface, Refusal, TokenGuard, peer_leaf_of, peer_of,
};

/// One system-table row as JSON (column name to value).
pub type Row = Map<String, Value>;

/// The W3C `traceparent` and `baggage` header values of an HTTP request: untrusted text the session validates
/// (a malformed value means no trace and never fails the request).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InboundTrace {
    /// The `traceparent` header value.
    pub traceparent: Option<String>,
    /// The `baggage` header value.
    pub baggage: Option<String>,
}

impl InboundTrace {
    /// The trace headers of `headers` (a non-ASCII value is no header at all).
    pub fn from_headers(headers: &axum::http::HeaderMap) -> Self {
        let text = |name: &str| {
            headers
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned)
        };
        Self {
            traceparent: text("traceparent"),
            baggage: text("baggage"),
        }
    }
}

/// A boxed future the source returns (object safe, no `async_trait`).
pub type UiFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, UiError>> + Send + 'a>>;

/// The system tables the UI reads. More are added as pages are: a [`UiSource`] outside this crate gives no rows
/// for a table it does not know.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SystemTable {
    /// `system.queries`.
    Queries,
    /// `system.operators`.
    Operators,
    /// `system.memory`.
    Memory,
    /// `system.caches`.
    Caches,
    /// `system.sessions`: the sessions of every client, open and lately ended.
    Sessions,
    /// `system.resource_groups`: the workload groups with their limits and live counters.
    ResourceGroups,
    /// `system.partition_advice`: the partition columns the query history suggests, per table.
    PartitionAdvice,
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
    /// The source has too much running to take more (HTTP 429).
    pub busy: bool,
}

impl UiError {
    /// A failure with no engine error class (a transport or set-up problem).
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            error_class: None,
            sql_state: None,
            busy: false,
        }
    }

    /// A refusal because too much is running already.
    #[must_use]
    pub fn busy(message: impl Into<String>) -> Self {
        Self {
            busy: true,
            ..Self::new(message)
        }
    }

    /// A failure the engine raised, with its class and SQLSTATE.
    #[must_use]
    pub fn with_class(message: impl Into<String>, class: &str, sql_state: Option<&str>) -> Self {
        Self {
            message: message.into(),
            error_class: Some(class.to_owned()),
            sql_state: sql_state.map(str::to_owned),
            busy: false,
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

    /// [`Self::cancel`] limited to the query `query_id` of session `session_id`: a query another session runs
    /// under the same id (a client chooses its operation ids) is left running. `false` when that session runs
    /// no such query, and from a source that does not tell sessions apart.
    fn cancel_in_session<'a>(
        &'a self,
        query_id: &'a str,
        session_id: &'a str,
    ) -> UiFuture<'a, bool> {
        let _ = (query_id, session_id);
        Box::pin(async { Ok(false) })
    }

    /// Ends session `session_id` as an operator's kill: its running queries fail with `KRISHIV_QUERY_KILLED`
    /// and the session's next call is refused. The number of queries cancelled; `None` when no open session
    /// has that id, or when the source does not end sessions (a console over one session, or over stored
    /// profiles).
    fn close_session<'a>(&'a self, session_id: &'a str) -> UiFuture<'a, Option<usize>> {
        let _ = session_id;
        Box::pin(async { Ok(None) })
    }

    /// The plan tree of the finished query whose `system.queries` row is `query`, as the profile document
    /// stores it (an array of nodes: `id`, `name`, `children` as indexes into the array), `None` when this
    /// source keeps no plan for it. The page joins the nodes to the query's operators by `node_id`, and lists
    /// the operators alone when there is none. Only a node's id, name and children reach the page
    /// (`plan_nodes`).
    fn plan<'a>(&'a self, query: &'a Row) -> UiFuture<'a, Option<Value>> {
        let _ = query;
        Box::pin(async { Ok(None) })
    }

    /// Whether the plan of the ended query whose row is `query` can still arrive: the source writes a history
    /// store and the query ended a moment ago, so its profile may not be written yet. The page asks again
    /// only while this holds; a source that will never have the plan says `false` and the page says so at once.
    fn plan_pending(&self, query: &Row) -> bool {
        let _ = query;
        false
    }

    /// The Python UDF worker pool now, for the Resources page; `None` when this source has no pool (a console
    /// over one session or over stored profiles, a server without one).
    fn udf_workers(&self) -> UiFuture<'_, Option<crate::udf_workers::UdfWorkers>> {
        Box::pin(async { Ok(None) })
    }

    /// Which console this is, as `/api/internal/health` reports it and the page reads it: `server` (every
    /// session of a server), `embedded` (one session of the host process) or `profiles` (stored profiles, no
    /// engine).
    fn mode(&self) -> &'static str {
        "server"
    }

    /// The id of the one session an `embedded` console shows.
    fn session_id(&self) -> Option<String> {
        None
    }

    /// The objects one level below `path` in the catalog tree, as rows with at least a `name`: no path lists the
    /// catalogs, `[catalog]` its schemas, `[catalog, schema]` its tables, `[catalog, schema, table]` its columns
    /// (`name`, `data_type`, `comment`). A source with no catalog has none.
    fn catalog<'a>(&'a self, path: &'a [String]) -> UiFuture<'a, Vec<Row>> {
        let _ = path;
        Box::pin(async { Err(UiError::new("this source has no catalog")) })
    }

    /// The engine's settings as key and value pairs, as they are: the page masks the secrets.
    fn environment(&self) -> UiFuture<'_, Vec<(String, String)>> {
        Box::pin(async { Err(UiError::new("this source has no environment")) })
    }

    /// Starts `sql` for the editor as `caller` (the authenticated principal, `None` when the guard knows only a
    /// shared secret) and returns the run's id at once. A run belongs to its caller: another caller sees and
    /// cancels nothing of it.
    fn sql_start<'a>(&'a self, sql: &'a str, caller: Option<&'a Caller>) -> UiFuture<'a, String> {
        let _ = (sql, caller);
        Box::pin(async { Err(UiError::new("this source runs no SQL")) })
    }

    /// As [`Self::sql_start`], with the W3C trace headers of the HTTP request (P3-T068): the statement's queries join
    /// that trace. A source that does not trace starts the run as `sql_start` does.
    fn sql_start_traced<'a>(
        &'a self,
        sql: &'a str,
        caller: Option<&'a Caller>,
        trace: &'a InboundTrace,
    ) -> UiFuture<'a, String> {
        let _ = trace;
        self.sql_start(sql, caller)
    }

    /// The state of an editor run (`state` is `running`, `finished`, `failed` or `cancelled`; a finished run
    /// carries `columns` and `rows`, a failed one `error`); `None` for an id the source does not know.
    fn sql_status<'a>(
        &'a self,
        run_id: &'a str,
        caller: Option<&'a Caller>,
    ) -> UiFuture<'a, Option<Value>> {
        let _ = (run_id, caller);
        Box::pin(async { Ok(None) })
    }

    /// Cancels an editor run; `false` when it is not running.
    fn sql_cancel<'a>(&'a self, run_id: &'a str, caller: Option<&'a Caller>) -> UiFuture<'a, bool> {
        let _ = (run_id, caller);
        Box::pin(async { Ok(false) })
    }
}

#[derive(Clone)]
pub(crate) struct UiState {
    pub(crate) source: Arc<dyn UiSource>,
    guard: Arc<dyn Guard>,
    pub(crate) policy: Arc<crate::pages::UiPolicy>,
    started: Instant,
    /// Event streams open now (capped at [`MAX_EVENT_CLIENTS`]).
    streams: Arc<AtomicUsize>,
    /// The last live-query snapshot the event streams share, so the number of open tabs does not change how many
    /// planned queries the engine runs.
    snapshot: Arc<tokio::sync::Mutex<Option<(Instant, Value)>>>,
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

/// Whether `req` is the embeddable view: the profile page of one query (`/queries/{id}/profile`, the only
/// `embeddable` route of `ui/routes.json`) with exactly one `embed` parameter, and it is `1`. Anything else (another
/// page, `embed=0&embed=1`, an API path) is never framable.
fn is_embed_page(req: &axum::extract::Request) -> bool {
    let mut seg = req.uri().path().trim_matches('/').split('/');
    let profile = matches!(
        (seg.next(), seg.next(), seg.next(), seg.next()),
        (Some("queries"), Some(id), Some("profile"), None) if !id.is_empty()
    );
    // Names and values are decoded as the page decodes them (`emb%65d` is `embed`).
    let mut embeds = req.uri().query().unwrap_or("").split('&').filter_map(|kv| {
        let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
        (crate::embed::percent_decode(k).as_deref() == Some(crate::embed::EMBED_PARAM))
            .then(|| crate::embed::percent_decode(v))
    });
    profile && embeds.next() == Some(Some("1".to_owned())) && embeds.next().is_none()
}

async fn security_headers(
    State(policy): State<Arc<crate::pages::UiPolicy>>,
    req: axum::extract::Request,
    next: Next,
) -> Response {
    // Only the embeddable view may be framed, and only by the listed origins (T071); the default is `'none'`.
    let framable = is_embed_page(&req) && !policy.embed().origins().is_empty();
    let mut res = next.run(req).await;
    let h = res.headers_mut();
    let csp = framable
        .then(|| {
            CSP.replace(
                "frame-ancestors 'none'",
                &format!("frame-ancestors {}", policy.embed().frame_ancestors()),
            )
        })
        .and_then(|c| HeaderValue::from_str(&c).ok());
    // A policy that cannot be written falls back to the static one and `DENY`: never open.
    let framable = csp.is_some();
    h.insert(
        header::CONTENT_SECURITY_POLICY,
        csp.unwrap_or_else(|| HeaderValue::from_static(CSP)),
    );
    if framable {
        // X-Frame-Options cannot name several origins; frame-ancestors decides.
        h.remove(header::X_FRAME_OPTIONS);
    } else {
        h.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    }
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
    router_with_ui_guard(source, Arc::new(TokenGuard::new(token)))
}

/// [`router_with_ui`] behind `guard`: reading other principals' work needs [`Access::Monitor`], a cancel
/// [`Access::Operate`] (ADR 0046 §3: both operator), the settings [`Access::Manage`] (admin). The token is taken
/// from the same places as with a start-up token, so the page works unchanged with a configured token.
pub fn router_with_ui_guard(source: Arc<dyn UiSource>, guard: Arc<dyn Guard>) -> Router {
    router_with_ui_policy(source, guard, crate::pages::UiPolicy::loopback())
}

/// [`router_with_ui_guard`] answering to the hosts `policy` names (the default is loopback only: DNS rebinding).
pub fn router_with_ui_policy(
    source: Arc<dyn UiSource>,
    guard: Arc<dyn Guard>,
    policy: crate::pages::UiPolicy,
) -> Router {
    use crate::pages;
    let state = UiState {
        source,
        guard,
        policy: Arc::new(policy),
        started: Instant::now(),
        streams: Arc::new(AtomicUsize::new(0)),
        snapshot: Arc::new(tokio::sync::Mutex::new(None)),
    };
    let internal = Router::new()
        .route("/api/internal/health", get(health))
        .route("/api/internal/queries", get(queries))
        .route("/api/internal/queries/{id}", get(query_detail))
        .route("/api/internal/queries/{id}/cancel", post(cancel))
        .route("/api/internal/history", get(history))
        .route("/api/internal/compare", get(compare))
        .route("/api/internal/resources", get(resources))
        .route(
            "/api/internal/udf-workers",
            get(crate::udf_workers::udf_workers),
        )
        .route("/api/internal/sessions", get(crate::workload::sessions))
        .route(
            "/api/internal/sessions/{id}/close",
            post(crate::workload::close_session),
        )
        .route("/api/internal/groups", get(crate::workload::groups))
        .route("/api/internal/advice", get(crate::advice::advice))
        .route("/api/internal/events", get(events))
        .route("/api/internal/catalog/catalogs", get(pages::catalogs))
        .route(
            "/api/internal/catalog/{catalog}/schemas",
            get(pages::schemas),
        )
        .route(
            "/api/internal/catalog/{catalog}/{schema}/tables",
            get(pages::tables),
        )
        .route(
            "/api/internal/catalog/{catalog}/{schema}/{table}/columns",
            get(pages::columns),
        )
        .route("/api/internal/environment", get(pages::environment))
        .route("/api/internal/sql", post(pages::sql_start))
        .route("/api/internal/sql/{run}", get(pages::sql_status))
        .route("/api/internal/sql/{run}/cancel", post(pages::sql_cancel))
        .route("/api/internal/session", post(pages::create_session))
        .route("/api/internal/openapi.json", get(pages::openapi_json))
        .route("/api/internal/queries/{id}/embed-link", get(embed_link))
        .with_state(state.clone())
        .layer(middleware::from_fn_with_state(state.clone(), require_token))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            pages::host_guard,
        ));
    let policy = Arc::clone(&state.policy);
    // The embedded view's own routes: no token (the signed link is the credential, one query, read-only), but the
    // same host check.
    let embed_open = Router::new()
        .route("/api/internal/embed/config", get(embed_config))
        .route("/api/internal/embed/queries/{id}", get(embed_query))
        .with_state(state.clone())
        .layer(middleware::from_fn_with_state(state, pages::host_guard));
    crate::router()
        .merge(internal)
        .merge(embed_open)
        .layer(middleware::from_fn_with_state(policy, security_headers))
}

pub(crate) fn json_response(status: StatusCode, body: &Value) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

pub(crate) fn error(status: StatusCode, code: &str, message: &str) -> Response {
    json_response(status, &json!({"error": code, "message": message}))
}

pub(crate) fn source_error(e: &UiError) -> Response {
    json_response(
        if e.busy {
            StatusCode::TOO_MANY_REQUESTS
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        },
        &json!({
            "error": "source_failed",
            "message": e.message,
            "error_class": e.error_class,
            "sql_state": e.sql_state,
        }),
    )
}

/// A query-string parameter's raw value (no percent-decoding: the parameters here are ids, numbers and names).
fn param<'a>(uri: &'a Uri, name: &str) -> Option<&'a str> {
    uri.query()?
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .find(|(k, _)| *k == name)
        .map(|(_, v)| v)
}

/// A filter's value as the page sent it (`URLSearchParams`): `+` is a space and every `%XX` its byte. A value
/// that does not decode is no filter.
fn filter_param(uri: &Uri, name: &str) -> Option<String> {
    percent_decode(&param(uri, name)?.replace('+', " ")).filter(|v| !v.is_empty())
}

/// The session a request names for a query (`name` is `session`, `a_session` or `b_session`): `None` when the
/// request names none. A qualifier that is there but empty or does not decode is refused (400): dropping it
/// would act on whichever session's query comes first.
fn session_param(uri: &Uri, name: &str) -> Result<Option<String>, Box<Response>> {
    // A bare key (`?session`, no `=`) names an empty session too.
    let bare = uri
        .query()
        .is_some_and(|q| q.split('&').any(|kv| kv == name));
    if !bare && param(uri, name).is_none() {
        return Ok(None);
    }
    match filter_param(uri, name).filter(|_| !bare) {
        Some(session) => Ok(Some(session)),
        None => Err(Box::new(error(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "the session named for the query is empty or not valid text",
        ))),
    }
}

/// Whether a `system.queries` row carries `tag`: a tag's key, or `key=value`. A request tag (Spark Connect's
/// `addTag`) is a key with an empty value.
fn has_tag(row: &Row, tag: &str) -> bool {
    row.get("tags")
        .and_then(Value::as_object)
        .is_some_and(|tags| {
            tags.iter().any(|(k, v)| {
                k == tag
                    || tag
                        .strip_prefix(k.as_str())
                        .and_then(|rest| rest.strip_prefix('='))
                        .is_some_and(|value| v.as_str() == Some(value))
            })
        })
}

/// The presented token. `Authorization: Bearer` is accepted for every method. The cookie and `?token=` (an
/// `EventSource` cannot set headers) are accepted for `GET` only, so a request that changes something (cancel)
/// cannot be made by a link or a cross-site form that carries the token in its address or a cookie.
fn presented(method: &Method, headers: &HeaderMap, uri: &Uri) -> Option<String> {
    if let Some(v) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        && let Some(t) = v.strip_prefix("Bearer ")
    {
        return Some(t.trim().to_owned());
    }
    // A cookie carries a request that changes something only with the page's own header (a cross-site form or
    // link cannot add it); a `?token=` is for reads only.
    if method != Method::GET && !crate::pages::has_csrf_header(headers) {
        return None;
    }
    // Only the cookie of this address counts: another console on the same host name has its own (see
    // `session_cookie_name`).
    let name = crate::pages::session_cookie_name(headers);
    if let Some(c) = headers.get(header::COOKIE).and_then(|v| v.to_str().ok())
        && let Some(t) = c.split(';').find_map(|p| {
            p.trim()
                .strip_prefix(name.as_str())
                .and_then(|rest| rest.strip_prefix('='))
        })
    {
        return Some(t.to_owned());
    }
    if method != Method::GET {
        return None;
    }
    param(uri, "token").and_then(percent_decode)
}

/// A `?token=` value as the page sent it (`encodeURIComponent`): every `%XX` becomes its byte. `+` stays `+` (the
/// page never writes a space as `+`, and a token may hold `+`). A broken escape or a value that is not UTF-8 after
/// decoding presents no token.
fn percent_decode(raw: &str) -> Option<String> {
    let b = raw.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            let hex = b.get(i + 1..i + 3)?;
            let hi = char::from(hex[0]).to_digit(16)?;
            let lo = char::from(hex[1]).to_digit(16)?;
            out.push(u8::try_from(hi * 16 + lo).ok()?);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

async fn require_token(
    State(s): State<UiState>,
    req: axum::extract::Request,
    next: Next,
) -> Response {
    let token = presented(req.method(), req.headers(), req.uri());
    let malformed = token.is_none() && req.headers().contains_key(header::AUTHORIZATION);
    // The settings are the administrator's. Anything else but a read changes something: a cancel acts on another
    // principal's query.
    let path = req.uri().path();
    let access = if path == "/api/internal/environment" {
        Access::Manage
    } else if req.method() == Method::GET || path == "/api/internal/session" {
        Access::Monitor
    } else {
        Access::Operate
    };
    let request = GuardRequest::new(GuardSurface::Ui, access)
        .with_token(token.as_deref())
        .with_malformed(malformed)
        .with_peer(peer_of(req.extensions()))
        .with_peer_leaf(peer_leaf_of(req.extensions()));
    match s.guard.authenticate(request).await {
        Ok(caller) => {
            let mut req = req;
            req.extensions_mut().insert(caller);
            next.run(req).await
        }
        Err(Refusal::Unauthenticated) => {
            let mut res = error(
                StatusCode::UNAUTHORIZED,
                "unauthorized",
                "a valid UI token is required",
            );
            res.headers_mut()
                .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
            res
        }
        Err(Refusal::Denied) => error(
            StatusCode::FORBIDDEN,
            "forbidden",
            "the principal's role does not allow this",
        ),
        Err(Refusal::Unavailable) => error(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable",
            "authentication is not possible now; retry later",
        ),
        Err(_) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "misconfigured",
            "the server cannot authenticate this request",
        ),
    }
}

/// `text` with every literal replaced by `?`: quoted strings (`'it''s'`, `"x"`, with backslash escapes) and
/// numbers (`42`, `1.5e3`, `10L`, `0xFF`), outside of identifiers and backquoted names. A query's text can hold
/// secrets and personal data, and the page is a screen: the literals stay in the engine's own record.
#[must_use]
pub fn mask_literals(text: &str) -> String {
    let masked = mask_pass(text);
    // `SET key = value`: the value is a secret as often as not. Looked for in the masked text (comments gone,
    // any whitespace allowed after SET) so a leading comment or a tab cannot hide it: keep the key, drop the value.
    let head = masked.trim_start();
    let is_set = head.len() > 3
        && head.is_char_boundary(3)
        && head[..3].eq_ignore_ascii_case("set")
        && head[3..].starts_with(char::is_whitespace);
    if is_set && let Some(eq) = masked.find('=') {
        return format!("{}=?", &masked[..eq]);
    }
    masked
}

/// One pass over `text` masking strings, numbers and comments (see [`mask_literals`]).
fn mask_pass(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    let ident = |c: char| c.is_alphanumeric() || c == '_' || c == '$';
    while i < chars.len() {
        let c = chars[i];
        let prev_ident = i > 0 && ident(chars[i - 1]);
        match c {
            '\'' | '"' => {
                let quote = c;
                i += 1;
                while i < chars.len() {
                    if chars[i] == '\\' {
                        i += 2;
                    } else if chars[i] == quote {
                        if chars.get(i + 1) == Some(&quote) {
                            i += 2;
                        } else {
                            i += 1;
                            break;
                        }
                    } else {
                        i += 1;
                    }
                }
                out.push('?');
            }
            '`' => {
                // A quoted name stays. One that never closes (a cut text) is masked to its end.
                match chars[i + 1..].iter().position(|&x| x == '`') {
                    Some(len) => {
                        out.extend(&chars[i..=i + 1 + len]);
                        i += len + 2;
                    }
                    None => {
                        out.push('?');
                        i = chars.len();
                    }
                }
            }
            // Comments can hold anything: drop them.
            '-' if chars.get(i + 1) == Some(&'-') => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
                out.push(' ');
            }
            '/' if chars.get(i + 1) == Some(&'*') => {
                i += 2;
                while i < chars.len() && !(chars[i] == '*' && chars.get(i + 1) == Some(&'/')) {
                    i += 1;
                }
                i = (i + 2).min(chars.len());
                out.push(' ');
            }
            // A number: `42`, `1.5e3`, `10L`, `0xFF`, and `.5`.
            d if (d.is_ascii_digit() && !prev_ident && (i == 0 || chars[i - 1] != '.'))
                || (d == '.'
                    && !prev_ident
                    && chars.get(i + 1).is_some_and(char::is_ascii_digit)) =>
            {
                i += 1;
                while i < chars.len() && (ident(chars[i]) || chars[i] == '.') {
                    i += 1;
                }
                out.push('?');
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    out
}

/// What the tag of a statement run from the console's editor starts with (the editor cancels its run by this
/// tag). It is the console's own bookkeeping: a query's row is sent without it.
pub(crate) const EDITOR_TAG_PREFIX: &str = "krishiv-ui-run-";

/// [`pick`] for a `system.queries` row, with the literals of its text masked and without the editor's own tag.
fn pick_query(row: &Row) -> Row {
    let mut picked = pick(row, QUERY_COLUMNS);
    if let Some(Value::Object(tags)) = picked.get_mut("tags") {
        tags.retain(|key, _| !key.starts_with(EDITOR_TAG_PREFIX));
    }
    if let Some(Value::String(text)) = picked.get("query_text") {
        let masked = mask_literals(text);
        picked.insert("query_text".to_owned(), Value::String(masked));
    }
    picked
}

/// [`pick_query`] for a list of queries (the live list, the history): the row also names its session, so the
/// Sessions page can point at a session's queries. A query's own page does not carry it: a signed embed link
/// reads that response with no session of its own.
fn pick_listed(row: &Row) -> Value {
    let mut picked = pick_query(row);
    if let Some(session) = row.get("session_id") {
        picked.insert("session_id".to_owned(), session.clone());
    }
    Value::Object(picked)
}

/// Keeps only the allowlisted columns of a row.
pub(crate) fn pick(row: &Row, columns: &[&str]) -> Row {
    columns
        .iter()
        .filter_map(|c| row.get(*c).map(|v| ((*c).to_owned(), v.clone())))
        .collect()
}

const QUERY_COLUMNS: &[&str] = &[
    "query_id",
    "state",
    "principal",
    "client_type",
    "tags",
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
    "hints",
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

pub(crate) fn text<'a>(row: &'a Row, column: &str) -> &'a str {
    row.get(column).and_then(Value::as_str).unwrap_or("")
}

pub(crate) fn int(row: &Row, column: &str) -> i64 {
    row.get(column).and_then(Value::as_i64).unwrap_or(0)
}

fn is_live(row: &Row) -> bool {
    matches!(text(row, "state"), "running" | "queued")
}

fn live_queries(rows: &[Row]) -> Vec<Value> {
    let mut live: Vec<&Row> = rows.iter().filter(|r| is_live(r)).collect();
    live.sort_by_key(|r| int(r, "submitted_at"));
    live.into_iter().map(pick_listed).collect()
}

pub(crate) async fn table(s: &UiState, t: SystemTable) -> Result<Vec<Row>, Box<Response>> {
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

async fn health(State(s): State<UiState>, caller: Option<Extension<Caller>>) -> Response {
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
            "mode": s.source.mode(),
            "session_id": s.source.session_id(),
            // The page hides what the caller's role cannot open; the routes decide by themselves.
            "role": caller.and_then(|c| c.0.role),
            "uptime_seconds": s.started.elapsed().as_secs(),
            "workers": workers.len().max(1),
            "running_queries": queries.iter().filter(|r| text(r, "state") == "running").count(),
            "queued_queries": queries.iter().filter(|r| text(r, "state") == "queued").count(),
            "memory": memory_total(&memory),
            "spill_bytes_total": spill_bytes,
            "caches": caches.iter().map(|r| {
                let mut c = pick(r, &["worker_id", "cache", "entries", "bytes", "capacity_bytes", "hits", "misses"]);
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

/// The operator with the most compute time, `null` when there is none.
fn hot_operator(operators: &[Row]) -> Value {
    operators
        .iter()
        .max_by_key(|r| int(r, "compute_ns"))
        .map_or(Value::Null, |r| {
            json!({"operator_id": int(r, "operator_id"), "operator": text(r, "name"), "compute_ns": int(r, "compute_ns")})
        })
}

/// `GET /api/internal/queries/{id}[?session=]`: one query's page. Two sessions may run a query under one id
/// (a client chooses its operation ids), so a list's link names the row's session; without one the first
/// query of that id answers.
async fn query_detail(State(s): State<UiState>, Path(id): Path<String>, uri: Uri) -> Response {
    match session_param(&uri, "session") {
        Ok(session) => detail(&s, &id, session.as_deref()).await,
        Err(e) => *e,
    }
}

/// Whether a `system.queries` row is query `id`, of session `session` when one is named.
fn is_query(row: &Row, id: &str, session: Option<&str>) -> bool {
    text(row, "query_id") == id && session.is_none_or(|s| text(row, "session_id") == s)
}

async fn detail(s: &UiState, id: &str, session: Option<&str>) -> Response {
    let (queries, operators) = match (
        table(s, SystemTable::Queries).await,
        table(s, SystemTable::Operators).await,
    ) {
        (Ok(q), Ok(o)) => (q, o),
        (Err(e), _) | (_, Err(e)) => return *e,
    };
    let Some(query) = queries.iter().find(|r| is_query(r, id, session)) else {
        return error(
            StatusCode::NOT_FOUND,
            "no_such_query",
            "no query with that id",
        );
    };
    // Operator rows and stored plans are kept by query id alone. When another session ran a query under this
    // id they would be both runs', and a signed link is for one session's query: neither run's are given.
    let other_run = |rows: &[Row]| {
        rows.iter().any(|r| {
            text(r, "query_id") == id && text(r, "session_id") != text(query, "session_id")
        })
    };
    let mut shared = other_run(&queries);
    let mut ops: Vec<Row> = operators
        .into_iter()
        .filter(|r| !shared && text(r, "query_id") == id)
        .collect();
    ops.sort_by_key(|r| int(r, "operator_id"));
    // A source with no plan for the query leaves the page its flat operator list. A query that has not ended
    // has no stored plan, and its page asks again every second: the store is not searched for it.
    let ended = !matches!(text(query, "state"), "running" | "queued");
    let mut plan = if !ended || shared {
        None
    } else {
        s.source
            .plan(query)
            .await
            .ok()
            .flatten()
            .as_ref()
            .and_then(plan_nodes)
    };
    // Another session may have begun a run of the id after the queries were read, and the operators or the
    // plan read since may be that run's. A run has its operators and its plan only once it is listed, so
    // the queries as they are now name it: they are read again whenever there is something to give.
    if !shared && (!ops.is_empty() || plan.is_some()) {
        match table(s, SystemTable::Queries).await {
            Ok(now) if other_run(&now) => {
                shared = true;
                ops.clear();
                plan = None;
            }
            Ok(_) => {}
            Err(e) => return *e,
        }
    }
    json_response(
        StatusCode::OK,
        &json!({
            "query": pick_query(query),
            "operators": ops.iter().map(|r| Value::Object(pick(r, OPERATOR_COLUMNS))).collect::<Vec<_>>(),
            "hot_operator": hot_operator(&ops),
            // A query with no operator rows (one refused before it ran) has nothing to draw a plan over: no wait.
            "plan_pending": ended && plan.is_none() && !ops.is_empty() && s.source.plan_pending(query),
            "plan": plan,
            "operators_withheld": shared,
        }),
    )
}

/// The nodes of a stored plan as the page gets them: `id`, `name` and `children`, nothing else. A node's
/// `detail` and `facts` are the plan's own text: with `krishiv.history.literals = keep` they hold the
/// statement's literals, which the page never shows (the statement's text is masked on the same page, and a
/// signed link reads this response with no session); with the default they are blanked to `?`. `None` when
/// `plan` is not an array of nodes.
fn plan_nodes(plan: &Value) -> Option<Value> {
    let nodes = plan.as_array()?;
    Some(Value::Array(
        nodes
            .iter()
            .map(|n| {
                json!({
                    "id": n.get("id").filter(|v| v.is_u64()),
                    "name": n.get("name").and_then(Value::as_str).unwrap_or(""),
                    "children": n
                        .get("children")
                        .and_then(Value::as_array)
                        .map(|c| c.iter().filter(|i| i.is_u64()).cloned().collect::<Vec<_>>())
                        .unwrap_or_default(),
                })
            })
            .collect(),
    ))
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The value of `name` in a query string, percent-decoded.
fn query_param(raw: Option<&str>, name: &str) -> Option<String> {
    raw?.split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=')?;
        (k == name)
            .then(|| crate::embed::percent_decode(v))
            .flatten()
    })
}

/// `GET /api/internal/queries/{id}/embed-link[?ttl=seconds&theme=json&session=id]`: a signed read-only link to
/// one query's profile for an embedding console (T071). The life is the asked one, at most
/// `krishiv.ui.embed.linkTtl`. With `session` the link opens that session's query only (two sessions may run a
/// query under one id); without it, the first query of the id.
async fn embed_link(
    State(s): State<UiState>,
    Path(id): Path<String>,
    RawQuery(raw): RawQuery,
    uri: Uri,
) -> Response {
    let session = match session_param(&uri, "session") {
        Ok(session) => session,
        Err(e) => return *e,
    };
    let embed = s.policy.embed();
    let max = embed.ttl().as_secs();
    let ttl = match query_param(raw.as_deref(), "ttl") {
        None => max,
        Some(t) => match t.parse::<u64>() {
            Ok(n) if n > 0 => n.min(max),
            _ => {
                return error(
                    StatusCode::BAD_REQUEST,
                    "bad_request",
                    "ttl is a whole number of seconds",
                );
            }
        },
    };
    let theme = match query_param(raw.as_deref(), "theme") {
        None => None,
        Some(t) if t.len() > crate::embed::MAX_THEME_BYTES => {
            return error(
                StatusCode::BAD_REQUEST,
                "bad_theme",
                "the theme is too long",
            );
        }
        Some(t) => match serde_json::from_str::<Value>(&t)
            .map_err(|e| e.to_string())
            .and_then(|v| crate::embed::validate_theme(&v))
        {
            Ok(theme) => Some(theme.to_json().to_string()),
            Err(m) => return error(StatusCode::BAD_REQUEST, "bad_theme", &m),
        },
    };
    let expires = now_secs().saturating_add(ttl);
    let Some(sig) = embed.sign_in(&id, session.as_deref(), expires) else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "unavailable",
            "the server has no signing key",
        );
    };
    let mut url = format!(
        "/queries/{}/profile?{}=1&exp={expires}&sig={sig}",
        crate::embed::percent_encode(&id),
        crate::embed::EMBED_PARAM
    );
    if let Some(session) = &session {
        url.push_str("&session=");
        url.push_str(&crate::embed::percent_encode(session));
    }
    if let Some(t) = theme {
        url.push_str("&theme=");
        url.push_str(&crate::embed::percent_encode(&t));
    }
    json_response(
        StatusCode::OK,
        &json!({"url": url, "expires_at": expires, "query_id": id, "session_id": session}),
    )
}

/// `GET /api/internal/embed/config`: what the embedded page needs before it can trust a theme message.
async fn embed_config(State(s): State<UiState>) -> Response {
    let e = s.policy.embed();
    json_response(
        StatusCode::OK,
        &json!({"allowed_origins": e.origins(), "hide_attribution": e.hide_attribution()}),
    )
}

/// `GET /api/internal/embed/queries/{id}?exp=&sig=[&session=]`: the query detail for a valid signed link,
/// nothing else. The session is part of what was signed: a link opens the query of the session it was issued
/// for, and one issued for the id alone opens no named session's.
async fn embed_query(
    State(s): State<UiState>,
    Path(id): Path<String>,
    RawQuery(raw): RawQuery,
    uri: Uri,
) -> Response {
    let exp = query_param(raw.as_deref(), "exp").and_then(|e| e.parse::<u64>().ok());
    let sig = query_param(raw.as_deref(), "sig");
    // A session that is there but empty or does not decode was never signed: the link is not valid.
    let session = session_param(&uri, "session").ok();
    match (exp, sig, session) {
        (Some(exp), Some(sig), Some(session))
            if s.policy
                .embed()
                .verify_in(&id, session.as_deref(), exp, &sig, now_secs()) =>
        {
            detail(&s, &id, session.as_deref()).await
        }
        _ => error(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "the link is not valid, or has expired",
        ),
    }
}

/// `POST /api/internal/queries/{id}/cancel[?session=]`: an operator's kill of a running query; a query that
/// was cancelled is one audit record. With `session` only that session's query ends. Without it, an id that
/// several sessions are running is refused (409): the caller must say whose query it means.
async fn cancel(
    State(s): State<UiState>,
    caller: Option<Extension<Caller>>,
    Path(id): Path<String>,
    uri: Uri,
) -> Response {
    let session = match session_param(&uri, "session") {
        Ok(session) => session,
        Err(e) => return *e,
    };
    let done = match &session {
        Some(session) => s.source.cancel_in_session(&id, session).await,
        None => {
            let rows = match table(&s, SystemTable::Queries).await {
                Ok(r) => r,
                Err(e) => return *e,
            };
            let mut running = rows.iter().filter(|r| is_live(r) && is_query(r, &id, None));
            let first = running.next().map(|r| text(r, "session_id"));
            if running.any(|r| Some(text(r, "session_id")) != first) {
                return error(
                    StatusCode::CONFLICT,
                    "ambiguous_query",
                    "more than one session runs a query with that id: name the session",
                );
            }
            s.source.cancel(&id).await
        }
    };
    match done {
        Ok(true) => {
            let caller = caller.map(|c| c.0);
            tracing::info!(
                target: "krishiv::audit",
                principal = caller.as_ref().map_or("unknown", |c| c.principal.as_str()),
                role = caller.as_ref().and_then(|c| c.role.as_deref()).unwrap_or("none"),
                surface = "ui",
                action = "cancel_query",
                query = %id,
                session = session.as_deref().unwrap_or(""),
                "query cancelled"
            );
            json_response(StatusCode::OK, &json!({"cancelled": true}))
        }
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
    let state = filter_param(&uri, "state");
    let principal = filter_param(&uri, "principal");
    // A client names itself at length (`_SPARK_CONNECT_PYTHON spark/4.2.0 os/linux python/3.12`): any part
    // of the name finds it, in either case.
    let client = filter_param(&uri, "client").map(|c| c.to_lowercase());
    let tag = filter_param(&uri, "tag");
    // Any part of a statement finds it, in either case. It is looked for in the text as the list shows it, with
    // the literals masked: a search of the stored text would tell a caller which literal a statement holds.
    let statement = filter_param(&uri, "text").map(|t| t.to_lowercase());
    let session = filter_param(&uri, "session");
    // An empty filter is no filter, as for the others. One that does not decode is refused: dropping it would
    // list every session's queries under an address that names one.
    if session.is_none() && param(&uri, "session").is_some_and(|v| !v.is_empty()) {
        return error(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "the session filter is not valid text",
        );
    }
    let mut done: Vec<&Row> = rows
        .iter()
        .filter(|r| !is_live(r))
        .filter(|r| state.as_deref().is_none_or(|s| text(r, "state") == s))
        .filter(|r| {
            principal
                .as_deref()
                .is_none_or(|p| text(r, "principal") == p)
        })
        .filter(|r| {
            client
                .as_deref()
                .is_none_or(|c| text(r, "client_type").to_lowercase().contains(c))
        })
        .filter(|r| tag.as_deref().is_none_or(|t| has_tag(r, t)))
        .filter(|r| {
            statement.as_deref().is_none_or(|t| {
                mask_literals(text(r, "query_text"))
                    .to_lowercase()
                    .contains(t)
            })
        })
        .filter(|r| {
            session
                .as_deref()
                .is_none_or(|s| text(r, "session_id") == s)
        })
        .collect();
    done.sort_by_key(|r| std::cmp::Reverse(int(r, "submitted_at")));
    done.truncate(limit);
    json_response(
        StatusCode::OK,
        &json!({"queries": done.into_iter().map(pick_listed).collect::<Vec<_>>()}),
    )
}

/// `GET /api/internal/compare?a=&b=[&a_session=&b_session=]`: two runs side by side with the change in each
/// number (`b - a`). Two sessions may each have run a query under one id, so the page names each run's
/// session; without one the first run of that id answers.
async fn compare(State(s): State<UiState>, uri: Uri) -> Response {
    let (Some(a), Some(b)) = (filter_param(&uri, "a"), filter_param(&uri, "b")) else {
        return error(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "compare needs ?a=<query_id>&b=<query_id>",
        );
    };
    let (of_a, of_b) = match (
        session_param(&uri, "a_session"),
        session_param(&uri, "b_session"),
    ) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => return *e,
    };
    let rows = match table(&s, SystemTable::Queries).await {
        Ok(r) => r,
        Err(e) => return *e,
    };
    let find = |id: &str, session: Option<String>| {
        rows.iter().find(|r| is_query(r, id, session.as_deref()))
    };
    let (Some(ra), Some(rb)) = (find(&a, of_a), find(&b, of_b)) else {
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
        &json!({"a": pick_query(ra), "b": pick_query(rb), "delta": delta}),
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

/// Event streams open at once: each is a connection the server holds and a periodic read of the system tables.
pub const MAX_EVENT_CLIENTS: usize = 16;

/// Releases an event stream's place when the stream ends (the client left or the connection dropped).
struct StreamSlot(Arc<AtomicUsize>);

impl Drop for StreamSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The live-query list as an event body, normally read once per period however many streams ask (streams that
/// arrive while a read is in flight may each read once more). The lock is
/// held only to look at and to store the snapshot, never across the read: a slow read delays the stream that
/// asked for it and no other.
async fn shared_snapshot(s: &UiState) -> Value {
    if let Some((at, body)) = s.snapshot.lock().await.as_ref()
        && at.elapsed() < EVENT_PERIOD / 2
    {
        return body.clone();
    }
    let body = match s.source.rows(SystemTable::Queries).await {
        Ok(rows) => json!({"queries": live_queries(&rows)}),
        Err(e) => json!({"error": e.message, "error_class": e.error_class}),
    };
    *s.snapshot.lock().await = Some((Instant::now(), body.clone()));
    body
}

/// `GET /api/internal/events`: the live query list as a `queries` event, once a second. At most
/// [`MAX_EVENT_CLIENTS`] streams at a time (a further one is a `503`); the streams share one snapshot a period.
/// An administrator's stream opens with an `environment` event.
async fn events(State(s): State<UiState>, caller: Option<Extension<Caller>>) -> Response {
    if s.streams.fetch_add(1, Ordering::SeqCst) >= MAX_EVENT_CLIENTS {
        s.streams.fetch_sub(1, Ordering::SeqCst);
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "too_many_streams",
            "too many live views are open; close one and retry",
        );
    }
    let slot = StreamSlot(Arc::clone(&s.streams));
    let ticker = tokio::time::interval(EVENT_PERIOD);
    // The stream opens with the environment (secrets masked), so a page can show it without a second request.
    // Only for an administrator: the stream itself is open to operators, the settings are not
    // (`/api/internal/environment` asks for `Access::Manage`).
    let first = if caller.is_some_and(|c| c.0.is_admin()) {
        Some(match crate::pages::environment_json(&s).await {
            Ok(v) => Event::default().event("environment").data(v.to_string()),
            Err(e) => Event::default()
                .event("environment")
                .data(json!({"error": e.message}).to_string()),
        })
    } else {
        None
    };
    let opening = futures::stream::iter(first.map(Ok::<_, std::convert::Infallible>));
    let ticks = futures::stream::unfold((ticker, s, slot), |(mut ticker, s, slot)| async move {
        ticker.tick().await;
        let body = shared_snapshot(&s).await;
        let event = Event::default().event("queries").data(body.to_string());
        // A slow client must not make the next ticks fire back to back: the period restarts here.
        ticker.reset();
        Some((Ok::<_, std::convert::Infallible>(event), (ticker, s, slot)))
    });
    let stream = futures::StreamExt::chain(opening, ticks);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::{CSP, InboundTrace, mask_literals, percent_decode};

    /// P3-T068: the trace headers of an HTTP request; a non-ASCII value is no header.
    #[test]
    fn the_trace_headers_are_read_as_text_and_a_bad_one_is_none() {
        use axum::http::{HeaderMap, HeaderValue};
        assert_eq!(
            InboundTrace::from_headers(&HeaderMap::new()),
            InboundTrace::default()
        );
        let mut h = HeaderMap::new();
        h.insert("traceparent", HeaderValue::from_static("00-a-b-01"));
        h.insert("baggage", HeaderValue::from_bytes(&[0xE9]).unwrap());
        let t = InboundTrace::from_headers(&h);
        assert_eq!(t.traceparent.as_deref(), Some("00-a-b-01"));
        assert_eq!(t.baggage, None);
    }

    /// `ui/tests/fixture_server.py` repeats the policy; the page check is only honest while both agree.
    #[test]
    fn the_fixture_server_sends_the_same_csp() {
        let py = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../ui/tests/fixture_server.py"
        ))
        .expect("fixture server source");
        let block = py
            .split("CSP = (")
            .nth(1)
            .and_then(|r| r.split("\n)").next())
            .expect("CSP block");
        let joined: String = block
            .lines()
            .filter_map(|l| l.trim().strip_prefix('"').and_then(|l| l.strip_suffix('"')))
            .collect();
        assert_eq!(joined, CSP);
    }

    #[test]
    fn a_query_string_token_is_percent_decoded_as_the_page_encodes_it() {
        assert_eq!(
            percent_decode("ab%2Bc%2Fd%3D%3D").as_deref(),
            Some("ab+c/d==")
        );
        assert_eq!(percent_decode("a+b").as_deref(), Some("a+b"));
        assert_eq!(percent_decode("%e2%82%ac").as_deref(), Some("\u{20ac}"));
        assert_eq!(percent_decode("plain0123").as_deref(), Some("plain0123"));
        assert_eq!(percent_decode("").as_deref(), Some(""));
        for broken in ["%", "%2", "%zz", "a%G1", "%ff"] {
            assert_eq!(percent_decode(broken), None, "{broken}");
        }
    }

    #[test]
    fn literals_are_masked_and_names_are_not() {
        let cases = [
            ("SELECT 1", "SELECT ?"),
            (
                "SELECT * FROM t WHERE name = 'ann' AND age > 42",
                "SELECT * FROM t WHERE name = ? AND age > ?",
            ),
            ("select 'it''s a secret', \"also secret\"", "select ?, ?"),
            ("select 'back\\'slash', 2", "select ?, ?"),
            (
                "select l_orderkey1, t2.c3, `a 1` from t2 where x = 1.5e3",
                "select l_orderkey1, t2.c3, `a 1` from t2 where x = ?",
            ),
            ("select 10L, 1BD, 0xFF, 3.14", "select ?, ?, ?, ?"),
            (
                "select sum(l_quantity) from lineitem group by 1, 2",
                "select sum(l_quantity) from lineitem group by ?, ?",
            ),
            ("select 'unterminated", "select ?"),
            ("select .5, x.y from t", "select ?, x.y from t"),
            (
                "select 1 -- the password is hunter2\nfrom t",
                "select ?  \nfrom t",
            ),
            ("select /* secret */ 1", "select   ?"),
            ("select a from `unclosed", "select a from ?"),
            ("select 'é', 'ü'", "select ?, ?"),
            ("select 'ends in a backslash\\", "select ?"),
            ("select a from `x`", "select a from `x`"),
            ("setup the thing 42", "setup the thing ?"),
            ("", ""),
            ("`", "?"),
            ("\\", "\\"),
            ("/*", " "),
            ("a--b", "a "),
        ];
        for (input, want) in cases {
            assert_eq!(mask_literals(input), want, "{input:?}");
        }
    }

    #[test]
    fn a_set_statement_keeps_its_key_and_loses_its_value_however_it_is_written() {
        for (input, want) in [
            (
                "SET spark.hadoop.fs.s3a.secret.key = abc123",
                "SET spark.hadoop.fs.s3a.secret.key =?",
            ),
            ("  set x=1", "  set x=?"),
            ("SET\tk=hunter2", "SET\tk=?"),
            ("SET\nk=hunter2", "SET\nk=?"),
            ("/*c*/ SET k=hunter2", "  SET k=?"),
            ("SET -- tok abc\nk = v", "SET  \nk =?"),
            ("SET k = 'quoted secret'", "SET k =?"),
            ("SET TIME ZONE 'UTC'", "SET TIME ZONE ?"),
        ] {
            let got = mask_literals(input);
            assert_eq!(got, want, "{input:?}");
            assert!(
                !got.contains("hunter2")
                    && !got.contains("abc123")
                    && !got.contains("quoted secret"),
                "{got}"
            );
        }
    }

    #[test]
    fn a_canary_literal_never_survives() {
        for input in [
            "INSERT INTO users VALUES ('s3cr3t-canary', 123456789)",
            "SELECT * FROM t WHERE token = \"s3cr3t-canary\"",
            "SELECT 1 /* s3cr3t-canary */",
            "SELECT 1 -- s3cr3t-canary",
            "SET x = s3cr3t-canary",
        ] {
            let got = mask_literals(input);
            assert!(
                !got.contains("s3cr3t") && !got.contains("123456789"),
                "{input:?} -> {got:?}"
            );
        }
    }
}
