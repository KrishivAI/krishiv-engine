// SPDX-License-Identifier: Apache-2.0
//! The UI's page endpoints beyond the query lists (P3-T050): the catalog browser, the environment page, the SQL
//! editor's run and cancel, the UI session, the DNS-rebinding host check and the OpenAPI document of these routes.
//!
//! Everything reads through [`UiSource`](crate::internal::UiSource); nothing here holds engine state. Configuration shown on a page, in the
//! REST JSON or in the event stream goes through [`redact_entries`]: a key the engine classifies as sensitive
//! shows [`MASK`], and every other value has the secrets in it replaced ([`krishiv_common::redact::redact`]).
//!
//! Threat model (P3 security notes, section 5): the `Host` header must be on the allowlist (T-A9 / T-W1: a page on
//! another name that resolves to this address cannot read the API), a cookie never carries a request that changes
//! something without the page's own `X-Krishiv-Csrf` header and a same-origin `Origin` (T-W4), the session cookie is
//! `HttpOnly; SameSite=Strict` so no script and no link can read or replay it (T-W5), the token leaves the address
//! bar at once and is never sent in a URL by the page (T-D7).

use std::sync::Arc;

use axum::Extension;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::response::Response;
use krishiv_common::redact::redact;
use krishiv_common::secret::is_sensitive_key;
use serde_json::{Value, json};
use utoipa::{OpenApi, ToSchema};

use crate::guard::Caller;
use crate::internal::{UiState, error, json_response, source_error};

/// What a sensitive configuration value shows as.
pub const MASK: &str = "***";

/// The header the page puts on every request that changes something. A cross-site page cannot add it without a
/// CORS preflight this server never grants.
pub const CSRF_HEADER: &str = "x-krishiv-csrf";

/// The longest statement the editor runs.
const MAX_SQL_BYTES: usize = 256 * 1024;

/// The names this server answers to, and how its cookie is set.
#[derive(Debug, Clone)]
pub struct UiPolicy {
    hosts: Vec<String>,
    /// Whether the `Host` header is checked at all (see [`UiPolicy::for_listener`]).
    enforce: bool,
    secure_cookie: bool,
    embed: Arc<crate::embed::EmbedPolicy>,
}

impl Default for UiPolicy {
    fn default() -> Self {
        Self::loopback()
    }
}

impl UiPolicy {
    /// Loopback names only (`localhost`, `127.0.0.1`, `[::1]`), any port.
    #[must_use]
    pub fn loopback() -> Self {
        Self {
            hosts: ["localhost", "127.0.0.1", "[::1]"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            enforce: true,
            secure_cookie: false,
            embed: Arc::new(crate::embed::EmbedPolicy::default()),
        }
    }

    /// Lets the embeddable views be framed by `embed`'s origins and signs their links (P3-T071).
    #[must_use]
    pub fn with_embed(mut self, embed: crate::embed::EmbedPolicy) -> Self {
        self.embed = Arc::new(embed);
        self
    }

    /// The embedding settings.
    #[must_use]
    pub fn embed(&self) -> &crate::embed::EmbedPolicy {
        &self.embed
    }

    /// The policy of a listener: on a loopback address the loopback names and `names` (DNS rebinding, T-W5: a
    /// page on another name that resolves to the loopback address must not read the API); on another address
    /// (which already requires authentication) the `Host` header is checked only when `names` is not empty,
    /// so a client that reaches the machine by its address or any of its names is not refused.
    #[must_use]
    pub fn for_listener<'a>(loopback: bool, names: impl IntoIterator<Item = &'a str>) -> Self {
        let mut policy = Self::loopback();
        let mut any = false;
        for n in names {
            // A usable name counts even when it is already a default (`localhost`).
            any |= is_usable_host(n);
            policy = policy.allow_host(n);
        }
        policy.enforce = loopback || any;
        policy
    }

    /// Also answers to `host` (a name or an address, with no port; an IPv6 address in brackets).
    #[must_use]
    pub fn allow_host(mut self, host: &str) -> Self {
        let h = host.trim().to_ascii_lowercase();
        if is_usable_host(&h) && !self.hosts.contains(&h) {
            self.hosts.push(h);
        }
        self
    }

    /// Marks the session cookie `Secure` (the listener is TLS). It also fixes the scheme a browser's `Origin` must
    /// carry on a change (`https` when set, else `http`): behind a proxy that ends TLS in front of a plain listener
    /// the browser's `https` origin is refused (fail closed).
    #[must_use]
    pub fn secure_cookie(mut self, on: bool) -> Self {
        self.secure_cookie = on;
        self
    }

    /// Whether a `Host` header value names this server.
    #[must_use]
    pub fn allows(&self, host_header: &str) -> bool {
        if !self.enforce {
            return true;
        }
        let h = host_header.trim().to_ascii_lowercase();
        self.hosts.contains(&strip_port(&h).to_owned())
    }
}

/// `host[:port]` without the port; an IPv6 literal keeps its brackets.
fn strip_port(h: &str) -> &str {
    if h.starts_with('[') {
        return h.find(']').map_or(h, |end| &h[..=end]);
    }
    match h.rsplit_once(':') {
        Some((name, port)) if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => name,
        _ => h,
    }
}

/// The port of a `host[:port]` value, when it names one.
fn port_of(h: &str) -> Option<&str> {
    let rest = if h.starts_with('[') {
        &h[h.find(']')? + 1..]
    } else {
        h
    };
    match rest.rsplit_once(':') {
        Some((_, port)) if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => {
            Some(port)
        }
        _ => None,
    }
}

/// The name of the session cookie for the address a request was sent to: `krishiv_ui_token_<port>`, or
/// `krishiv_ui_token` when the `Host` header names no port. A browser keeps cookies per host name, not per port,
/// so two consoles on `127.0.0.1` (one per embedded session, each with its own port and token) would otherwise
/// overwrite each other's cookie and sign each other out.
pub(crate) fn session_cookie_name(headers: &HeaderMap) -> String {
    let port = headers
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .and_then(|h| port_of(h.trim()));
    match port {
        Some(port) => format!("krishiv_ui_token_{port}"),
        None => "krishiv_ui_token".to_owned(),
    }
}

/// The host check and the same-origin check, before any route (and so before any token check) of the API.
pub(crate) async fn host_guard(
    State(s): State<UiState>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let host = req
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !s.policy.allows(host) {
        return error(
            StatusCode::FORBIDDEN,
            "bad_host",
            "this address is not one the server answers to",
        );
    }
    if req.method() != Method::GET
        && let Some(origin) = req.headers().get(header::ORIGIN)
    {
        let same = origin.to_str().ok().is_some_and(|o| {
            let scheme = if s.policy.secure_cookie {
                "https"
            } else {
                "http"
            };
            o.split_once("://").is_some_and(|(got, authority)| {
                !authority.is_empty()
                    && got.eq_ignore_ascii_case(scheme)
                    && authority.eq_ignore_ascii_case(host)
            })
        });
        if !same {
            return error(
                StatusCode::FORBIDDEN,
                "bad_origin",
                "the request comes from another origin",
            );
        }
    }
    next.run(req).await
}

/// Whether a request that changes something carries the page's own header.
pub(crate) fn has_csrf_header(headers: &HeaderMap) -> bool {
    headers
        .get(CSRF_HEADER)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v == "1")
}

/// A configuration as the pages show it: the key, the value with secrets masked, and whether it was masked.
#[must_use]
pub fn redact_entries(entries: &[(String, String)]) -> Vec<Value> {
    let mut out: Vec<Value> = entries
        .iter()
        .map(|(k, v)| {
            let (value, masked) = if is_sensitive_key(k) {
                (MASK.to_owned(), true)
            } else {
                let r = redact(v);
                let changed = r != v.as_str();
                (r.into_owned(), changed)
            };
            json!({"key": k, "value": value, "redacted": masked})
        })
        .collect();
    out.sort_by(|a, b| a["key"].as_str().cmp(&b["key"].as_str()));
    out
}

/// The environment document: the REST body and the event the stream opens with.
pub(crate) async fn environment_json(s: &UiState) -> Result<Value, crate::internal::UiError> {
    let entries = s.source.environment().await?;
    Ok(json!({"version": crate::VERSION, "settings": redact_entries(&entries)}))
}

/// A name that may go into a catalog path: no quoting characters, no control characters, bounded.
fn valid_name(n: &str) -> bool {
    !n.is_empty()
        && n.len() <= 255
        && !n
            .chars()
            .any(|c| c.is_control() || matches!(c, '`' | '\'' | '"' | ';' | '\\'))
}

async fn catalog_at(s: &UiState, path: Vec<String>, key: &str) -> Response {
    if !path.iter().all(|n| valid_name(n)) {
        return error(StatusCode::BAD_REQUEST, "bad_name", "not a valid name");
    }
    match s.source.catalog(&path).await {
        Ok(rows) => json_response(
            StatusCode::OK,
            &json!({key: rows.into_iter().map(Value::Object).collect::<Vec<_>>()}),
        ),
        Err(e) => source_error(&e),
    }
}

/// `GET /api/internal/catalog/catalogs`.
#[utoipa::path(get, path = "/api/internal/catalog/catalogs", responses((status = 200, description = "The catalogs", body = Catalogs)))]
pub(crate) async fn catalogs(State(s): State<UiState>) -> Response {
    catalog_at(&s, vec![], "catalogs").await
}

/// `GET /api/internal/catalog/{catalog}/schemas`.
#[utoipa::path(get, path = "/api/internal/catalog/{catalog}/schemas", params(("catalog" = String, Path)), responses((status = 200, description = "The schemas of a catalog", body = Schemas)))]
pub(crate) async fn schemas(State(s): State<UiState>, Path(c): Path<String>) -> Response {
    catalog_at(&s, vec![c], "schemas").await
}

/// `GET /api/internal/catalog/{catalog}/{schema}/tables`.
#[utoipa::path(get, path = "/api/internal/catalog/{catalog}/{schema}/tables", params(("catalog" = String, Path), ("schema" = String, Path)), responses((status = 200, description = "The tables of a schema", body = Tables)))]
pub(crate) async fn tables(
    State(s): State<UiState>,
    Path((c, sc)): Path<(String, String)>,
) -> Response {
    catalog_at(&s, vec![c, sc], "tables").await
}

/// `GET /api/internal/catalog/{catalog}/{schema}/{table}/columns`.
#[utoipa::path(get, path = "/api/internal/catalog/{catalog}/{schema}/{table}/columns", params(("catalog" = String, Path), ("schema" = String, Path), ("table" = String, Path)), responses((status = 200, description = "The columns of a table", body = Columns)))]
pub(crate) async fn columns(
    State(s): State<UiState>,
    Path((c, sc, t)): Path<(String, String, String)>,
) -> Response {
    catalog_at(&s, vec![c, sc, t], "columns").await
}

/// `GET /api/internal/environment`: the version and the settings, secrets masked.
#[utoipa::path(get, path = "/api/internal/environment", responses((status = 200, description = "Settings with secrets masked", body = Environment)))]
pub(crate) async fn environment(State(s): State<UiState>) -> Response {
    match environment_json(&s).await {
        Ok(v) => json_response(StatusCode::OK, &v),
        Err(e) => source_error(&e),
    }
}

/// `POST /api/internal/sql`: starts a statement; the body is `{"sql": "..."}`.
#[utoipa::path(post, path = "/api/internal/sql", request_body = SqlRequest, params(("traceparent" = Option<String>, Header, description = "W3C trace context; the statement joins this trace"), ("baggage" = Option<String>, Header, description = "W3C baggage")), responses((status = 202, description = "Started", body = SqlStarted)))]
pub(crate) async fn sql_start(
    State(s): State<UiState>,
    caller: Option<Extension<Caller>>,
    headers: HeaderMap,
    body: String,
) -> Response {
    let sql = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|v| v.get("sql").and_then(Value::as_str).map(str::to_owned));
    let Some(sql) = sql.filter(|q| !q.trim().is_empty()) else {
        return error(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "the body must be {\"sql\": \"<statement>\"}",
        );
    };
    if sql.len() > MAX_SQL_BYTES {
        return error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "too_long",
            "the statement is too long",
        );
    }
    match s
        .source
        .sql_start_traced(
            &sql,
            caller.as_ref().map(|c| &c.0),
            &crate::internal::InboundTrace::from_headers(&headers),
        )
        .await
    {
        Ok(id) => json_response(StatusCode::ACCEPTED, &json!({"run_id": id})),
        Err(e) => source_error(&e),
    }
}

/// `GET /api/internal/sql/{run}`: the run's state and, once it ended, its rows or its error.
#[utoipa::path(get, path = "/api/internal/sql/{run}", params(("run" = String, Path)), responses((status = 200, description = "The run", body = SqlRun)))]
pub(crate) async fn sql_status(
    State(s): State<UiState>,
    caller: Option<Extension<Caller>>,
    Path(id): Path<String>,
) -> Response {
    match s
        .source
        .sql_status(&id, caller.as_ref().map(|c| &c.0))
        .await
    {
        Ok(Some(v)) => json_response(StatusCode::OK, &v),
        Ok(None) => error(StatusCode::NOT_FOUND, "no_such_run", "no such run"),
        Err(e) => source_error(&e),
    }
}

/// `POST /api/internal/sql/{run}/cancel`.
#[utoipa::path(post, path = "/api/internal/sql/{run}/cancel", params(("run" = String, Path)), responses((status = 200, description = "Cancelled")))]
pub(crate) async fn sql_cancel(
    State(s): State<UiState>,
    caller: Option<Extension<Caller>>,
    Path(id): Path<String>,
) -> Response {
    match s
        .source
        .sql_cancel(&id, caller.as_ref().map(|c| &c.0))
        .await
    {
        Ok(true) => json_response(StatusCode::OK, &json!({"cancelled": true})),
        Ok(false) => error(
            StatusCode::NOT_FOUND,
            "not_running",
            "that run is not running",
        ),
        Err(e) => source_error(&e),
    }
}

/// Whether `token` can sit in a cookie value as it is.
fn cookie_safe(token: &str) -> bool {
    !token.is_empty()
        && token.len() <= 4096
        && token
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._~+/=-".contains(&b))
}

/// `POST /api/internal/session`: trades the presented token for the session cookie, so the page keeps no token
/// and sends none in an address.
#[utoipa::path(post, path = "/api/internal/session", responses((status = 204, description = "The cookie is set")))]
pub(crate) async fn create_session(State(s): State<UiState>, headers: HeaderMap) -> Response {
    let token = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim);
    let Some(token) = token.filter(|t| cookie_safe(t)) else {
        return error(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "present the token as Authorization: Bearer",
        );
    };
    let secure = if s.policy.secure_cookie {
        "; Secure"
    } else {
        ""
    };
    let name = session_cookie_name(&headers);
    let cookie = format!(
        "{name}={token}; HttpOnly; SameSite=Strict; Path=/api/internal; Max-Age=28800{secure}"
    );
    let mut res = Response::new(axum::body::Body::empty());
    *res.status_mut() = StatusCode::NO_CONTENT;
    if let Ok(v) = HeaderValue::from_str(&cookie) {
        res.headers_mut().insert(header::SET_COOKIE, v);
    }
    res
}

/// `GET /api/internal/openapi.json`.
pub(crate) async fn openapi_json() -> Response {
    json_response(
        StatusCode::OK,
        &serde_json::to_value(ApiDoc::openapi()).unwrap_or(Value::Null),
    )
}

/// The OpenAPI document of the page endpoints of this module.
#[derive(Debug, OpenApi)]
#[openapi(
    info(
        title = "Krishiv UI API",
        description = "The REST behind the web UI pages (P3-T050)."
    ),
    paths(
        catalogs,
        schemas,
        tables,
        columns,
        environment,
        sql_start,
        sql_status,
        sql_cancel,
        create_session
    ),
    components(schemas(
        CatalogItem,
        Catalogs,
        Schemas,
        Tables,
        Columns,
        Environment,
        EnvironmentSetting,
        SqlRequest,
        SqlStarted,
        SqlRun
    ))
)]
pub struct ApiDoc;

/// The OpenAPI document as JSON.
#[must_use]
pub fn openapi() -> Value {
    serde_json::to_value(ApiDoc::openapi()).unwrap_or(Value::Null)
}

/// One object of the catalog tree.
#[derive(Debug, ToSchema)]
#[expect(dead_code, reason = "documents a response body")]
pub struct CatalogItem {
    /// The name.
    name: String,
    /// A column's type.
    data_type: Option<String>,
    /// A column's comment.
    comment: Option<String>,
    /// Whether a table is a temporary view.
    is_temporary: Option<bool>,
}

/// The reply of `catalogs`.
#[derive(Debug, ToSchema)]
#[expect(dead_code, reason = "documents a response body")]
pub struct Catalogs {
    /// The catalogs.
    catalogs: Vec<CatalogItem>,
}

/// The reply of `schemas`.
#[derive(Debug, ToSchema)]
#[expect(dead_code, reason = "documents a response body")]
pub struct Schemas {
    /// The schemas of the catalog.
    schemas: Vec<CatalogItem>,
}

/// The reply of `tables`.
#[derive(Debug, ToSchema)]
#[expect(dead_code, reason = "documents a response body")]
pub struct Tables {
    /// The tables of the schema.
    tables: Vec<CatalogItem>,
}

/// The reply of `columns`.
#[derive(Debug, ToSchema)]
#[expect(dead_code, reason = "documents a response body")]
pub struct Columns {
    /// The columns of the table.
    columns: Vec<CatalogItem>,
}

/// One setting of the environment page.
#[derive(Debug, ToSchema)]
#[expect(dead_code, reason = "documents a response body")]
pub struct EnvironmentSetting {
    /// The key.
    key: String,
    /// The value; `***` for a secret.
    value: String,
    /// Whether anything was masked.
    redacted: bool,
}

/// The environment page's body.
#[derive(Debug, ToSchema)]
#[expect(dead_code, reason = "documents a response body")]
pub struct Environment {
    /// The engine version.
    version: String,
    /// The settings, secrets masked.
    settings: Vec<EnvironmentSetting>,
}

/// The editor's request body.
#[derive(Debug, ToSchema)]
#[expect(dead_code, reason = "documents a request body")]
pub struct SqlRequest {
    /// The statement.
    sql: String,
}

/// The reply to starting a statement.
#[derive(Debug, ToSchema)]
#[expect(dead_code, reason = "documents a response body")]
pub struct SqlStarted {
    /// The run's id, for status and cancel.
    run_id: String,
}

/// A run of the editor.
#[derive(Debug, ToSchema)]
#[expect(dead_code, reason = "documents a response body")]
pub struct SqlRun {
    /// `running`, `finished`, `failed` or `cancelled`.
    state: String,
    /// The result's column names, once finished.
    columns: Option<Vec<String>>,
    /// The result's rows, at most 1,000, once finished.
    rows: Option<Vec<Vec<serde_json::Value>>>,
    /// Whether more rows were cut.
    truncated: Option<bool>,
    /// The error, redacted, when failed or cancelled.
    error: Option<String>,
    /// The error class.
    error_class: Option<String>,
    /// The error class's SQLSTATE.
    sql_state: Option<String>,
}

/// A configured host name that can be matched: not blank and not the wildcard `*` (which is never allowed).
fn is_usable_host(host: &str) -> bool {
    let h = host.trim();
    !h.is_empty() && h != "*"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_host_allowlist_ignores_ports_and_case() {
        let p = UiPolicy::loopback().allow_host("Ui.Example.com");
        assert!(p.allows("localhost:8080"));
        assert!(p.allows("127.0.0.1"));
        assert!(p.allows("[::1]:9"));
        assert!(p.allows("UI.example.com:443"));
        assert!(!p.allows("evil.example.com"));
        assert!(!p.allows("localhost.evil.com"));
        assert!(!p.allows(""));
    }

    #[test]
    fn sensitive_keys_and_embedded_secrets_are_masked() {
        let rows = redact_entries(&[
            ("krishiv.auth.token".into(), "tok-canary".into()),
            ("spark.hadoop.fs.s3a.secret.key".into(), "s3-canary".into()),
            ("db.password".into(), "pw-canary".into()),
            ("spark.sql.shuffle.partitions".into(), "200".into()),
            ("url".into(), "jdbc:x://u:urlpw-canary@h/db".into()),
        ]);
        let text = Value::Array(rows).to_string();
        for canary in ["tok-canary", "s3-canary", "pw-canary", "urlpw-canary"] {
            assert!(!text.contains(canary), "{canary} leaked: {text}");
        }
        assert!(text.contains("200"));
    }

    #[test]
    fn a_policy_for_a_listener_checks_loopback_binds_and_non_loopback_only_with_names() {
        assert!(!UiPolicy::for_listener(true, []).allows("evil.example"));
        assert!(UiPolicy::for_listener(true, ["ui.example.com"]).allows("ui.example.com:8080"));
        // Bound to 0.0.0.0 with nothing configured: any name or address reaches it.
        assert!(UiPolicy::for_listener(false, []).allows("10.1.2.3:8080"));
        assert!(UiPolicy::for_listener(false, [" ", "*"]).allows("10.1.2.3"));
        // With names configured, only those (and loopback).
        let p = UiPolicy::for_listener(false, ["ui.example.com"]);
        assert!(p.allows("ui.example.com"));
        assert!(!p.allows("evil.example"));
        // A configured name that is already a default still turns the check on.
        let p = UiPolicy::for_listener(false, ["localhost"]);
        assert!(p.allows("localhost:8080"));
        assert!(!p.allows("evil.example"));
    }

    #[test]
    fn names_with_quoting_characters_are_refused() {
        assert!(valid_name("main"));
        assert!(!valid_name("a`b"));
        assert!(!valid_name(""));
        assert!(!valid_name("a;b"));
    }
}
