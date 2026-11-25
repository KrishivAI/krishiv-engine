// SPDX-License-Identifier: Apache-2.0
//! [`RestClient`]: Krishiv's own read client of one Iceberg REST catalog (decision 8, T067 §3.4, owner decision
//! 2026-11-15): `GET /v1/config`, the namespace and table listings, `loadTable` and `loadCredentials`.
//!
//! - **Config.** Fetched once, on first use (with the declared `warehouse`). Only `prefix` and
//!   `namespace-separator` are taken from it, after a check that neither can leave `/v1/` (they go into the path as
//!   the server wrote them, as in Java's `ResourcePaths`). A `uri` it returns is ignored and logged: the client
//!   only ever talks to the declared host, so the caller's token never goes anywhere else.
//! - **Tokens.** One source per catalog, asked per request (`auth`) for the call's [`RestCaller`] (only a delegated
//!   source reads it; the `*_as` methods take it, the others call as the server); a 401 with a client-credentials or
//!   exchanged token drops it and retries once.
//! - **Locations.** `metadata-location`, the metadata's `location` and every storage credential's `prefix` pass
//!   [`check_location`] before the table is returned.
//! - **Refusals.** A table whose `read-restrictions` impose anything is refused (`KRISHIV_UNSUPPORTED_FEATURE`, T067
//!   §4.1: Krishiv cannot apply them yet, and the spec says such a reader must fail), as is one whose `config` says
//!   `scan-planning-mode=server`.
//! - **Limits.** Each call has one `request_timeout`, the first call's config fetch and every page of a listing
//!   included; listings
//!   stop at `max_pages` and `max_list_entries` and refuse a page token the server repeats or one past 4 KiB.
//! - **Names.** An empty, `.` or `..` namespace level or table name is refused before a URL is built, and any path
//!   the URL parser would rewrite is refused (it would address another resource).
//! - **Not here yet (the `Catalog` step, PR-58a/b).** Which table `config` keys reach a store: endpoint and signer
//!   overrides (`s3.endpoint`, `signer.uri`, refresh endpoints), `remote-signing-config` and the delegated-token
//!   keys are returned as they are (secret-typed) and the `Catalog` over this client decides on them, with the
//!   endpoint-override case of `catalog_location_confined`. `load_credentials` sends the caller's token; using a
//!   table's vended `token` for its later requests (T067 §4.1) comes with PR-58b. Copies of the client secret and
//!   token inside `reqwest` (the form body, the header value) are not wiped.
//!
//! Calls block the caller (the `Catalog` trait is synchronous); see the `transport` module.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use krishiv_common::error::{
    KrishivCatalogTableUnsupported, KrishivError, KrishivUnsupportedFeature,
};
use krishiv_common::secret::{SecretString, wipe_bytes};
use serde_json::Value;
use url::Url;

use super::auth::Tokens;
use super::config::RestConfig;
use super::delegated::RestCaller;
use super::encode::{
    DEFAULT_NAMESPACE_SEPARATOR, namespace_path, namespace_query, path_segment, prefix_is_safe,
    separator_is_safe,
};
use super::error::{RestError, RestLimit, quoted};
use super::transport::{Request, Transport};
use super::types::{self, CatalogConfig, LoadedTable, Page, StorageCredential, TableIdent};
use crate::location::{LocationPolicy, check_location, check_scheme};

/// The endpoint `loadCredentials` is advertised as.
const LOAD_CREDENTIALS: &str = "GET /v1/{prefix}/namespaces/{namespace}/tables/{table}/credentials";

/// `<base>/<segments...>`; each segment is already encoded (a `prefix` may hold `/`).
pub(crate) fn join(base: &Url, segments: &[&str]) -> Result<Url, RestError> {
    let mut url = base.clone();
    let mut path = base.path().trim_end_matches('/').to_owned();
    for s in segments {
        path.push('/');
        path.push_str(s);
    }
    url.set_path(&path);
    // The URL parser removes dot segments (`..`, `.`, `%2e%2e`) and would address another resource than the one
    // built; refuse any path it changed.
    if url.path() != path {
        return Err(RestError::Protocol(
            "a name that the URL parser rewrites (a `.` or `..` segment)".to_owned(),
        ));
    }
    if url.host_str() != base.host_str()
        || url.scheme() != base.scheme()
        || url.port() != base.port()
    {
        return Err(RestError::Protocol(
            "a path that changes the host".to_owned(),
        ));
    }
    Ok(url)
}

/// The 404 types that mean a namespace does not exist.
const NO_SUCH_NAMESPACE: &[&str] = &["NoSuchNamespaceException"];
/// The 404 types that mean a table (or its namespace) does not exist.
const NO_SUCH_TABLE: &[&str] = &["NoSuchTableException", "NoSuchNamespaceException"];
/// The `absent` kinds that make every 404 of the route itself absence, whatever its error model.
const ANY_404: &[&str] = &["*"];

/// The longest `next-page-token` accepted; a token is an opaque cursor, not a payload.
const MAX_PAGE_TOKEN_BYTES: usize = 4096;

/// Refuses a namespace level or table name a path cannot carry as itself: empty, `.` or `..` (a URL parser
/// collapses them, so the request would address another resource).
fn check_names(levels: &[String], table: Option<&str>) -> Result<(), RestError> {
    let bad = |n: &str| n.is_empty() || n == "." || n == "..";
    if levels.iter().any(|l| bad(l)) || table.is_some_and(bad) {
        return Err(RestError::Protocol(
            "an empty, `.` or `..` namespace level or table name".to_owned(),
        ));
    }
    Ok(())
}

/// As [`check_names`], for a call that addresses a namespace in the path: it needs at least one level (an empty
/// one would send `namespaces//tables`).
fn check_namespace(levels: &[String], table: Option<&str>) -> Result<(), RestError> {
    if levels.is_empty() {
        return Err(RestError::Protocol(
            "a table call without a namespace".to_owned(),
        ));
    }
    check_names(levels, table)
}

/// Whether a storage-credential prefix is a bare scheme (`s3`, `gs`, `abfss`; T067 §4.1).
fn is_bare_scheme(prefix: &str) -> bool {
    let mut bytes = prefix.bytes();
    bytes.next().is_some_and(|b| b.is_ascii_alphabetic())
        && bytes.all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.'))
}

/// Checks a storage-credential prefix: a full URL or bucket prefix like any location, a bare scheme prefix by
/// whether the policy allows that scheme ([`LocationPolicy::check_scheme`]).
fn check_prefix(
    catalog: &str,
    what: &str,
    prefix: &str,
    policy: &dyn LocationPolicy,
) -> Result<(), KrishivError> {
    if is_bare_scheme(prefix) {
        check_scheme(catalog, what, prefix, policy)
    } else {
        check_location(catalog, what, prefix, policy)
    }
}

/// Most `/v1/config` answers one catalog keeps (one per principal and claims on a delegated
/// catalog).
pub(crate) const MAX_CONFIGS: usize = 4096;

/// The `/v1/config` answers of one catalog, least recently used out past [`MAX_CONFIGS`].
#[derive(Debug, Default)]
pub(crate) struct ConfigCache {
    entries: HashMap<Option<String>, (Arc<Server>, u64)>,
    tick: u64,
}

impl ConfigCache {
    fn get(&mut self, key: &Option<String>) -> Option<Arc<Server>> {
        self.tick += 1;
        let tick = self.tick;
        self.entries.get_mut(key).map(|(s, used)| {
            *used = tick;
            Arc::clone(s)
        })
    }

    /// Stores `fetched` unless a racing caller stored one first (that one wins), evicting the
    /// least recently used entry when full.
    fn insert(&mut self, key: Option<String>, fetched: Arc<Server>) -> Arc<Server> {
        if let Some(s) = self.get(&key) {
            return s;
        }
        if self.entries.len() >= MAX_CONFIGS
            && let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(k, _)| k.clone())
        {
            self.entries.remove(&oldest);
        }
        self.entries.insert(key, (Arc::clone(&fetched), self.tick));
        fetched
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

/// The `snapshots` parameter of `loadTable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Snapshots {
    /// `snapshots=all`: every snapshot.
    All,
    /// `snapshots=refs`: only the snapshots refs point at.
    Refs,
}

impl Snapshots {
    fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Refs => "refs",
        }
    }
}

/// The catalog's answer to a commit request ([`RestClient::commit_table_as`]).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CommitReply {
    /// The HTTP status.
    pub status: u16,
    /// `Retry-After` in seconds, when the answer had one.
    pub retry_after: Option<std::time::Duration>,
    /// The body (a `LoadTableResult` on 200, an `ErrorModel` otherwise).
    pub body: Vec<u8>,
}

/// A commit request that got no usable answer.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CommitFailure {
    /// Whether the request may have reached the catalog (`false`: provably never sent).
    pub sent: bool,
    /// Why, for the log and messages (Krishiv's text; never a token or a server's message).
    pub reason: String,
}

/// What the server's `/v1/config` settled.
#[derive(Debug)]
struct Server {
    prefix: Option<String>,
    separator: String,
    config: CatalogConfig,
}

/// A read client of one Iceberg REST catalog.
pub struct RestClient {
    name: String,
    config: RestConfig,
    http: Transport,
    tokens: Tokens,
    policy: Arc<dyn LocationPolicy + Send + Sync>,
    /// `/v1/config` by who fetched it: `None` for the catalog's own identity (every mode but
    /// delegated), else the caller's principal key and claims fingerprint, so one user's answer
    /// never shapes another user's calls (T067 §3.5); least recently used out past
    /// [`MAX_CONFIGS`].
    server: Mutex<ConfigCache>,
}

impl std::fmt::Debug for RestClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RestClient")
            .field("name", &self.name)
            .field("uri", &self.config.uri.as_str())
            .finish_non_exhaustive()
    }
}

/// A namespace as error messages name it.
fn relation(levels: &[String], table: Option<&str>) -> String {
    let mut parts: Vec<String> = levels.iter().map(|l| quoted(l)).collect();
    if let Some(t) = table {
        parts.push(quoted(t));
    }
    if parts.is_empty() {
        "the namespace listing".to_owned()
    } else {
        parts.join(".")
    }
}

impl RestClient {
    /// A client named `name` (the Krishiv catalog name) for the catalog `config` declares; `policy` decides which
    /// locations it may return. Starts the client thread; nothing is sent until the first call.
    ///
    /// # Errors
    /// The declaration is refused ([`RestConfig::validate`]), or the client thread cannot start.
    pub fn new(
        name: impl Into<String>,
        config: RestConfig,
        policy: Arc<dyn LocationPolicy + Send + Sync>,
    ) -> Result<Self, KrishivError> {
        let name = name.into();
        config.validate()?;
        let start = || -> Result<(Transport, Tokens), RestError> {
            let http = Transport::start(
                &config.connector(),
                config.allow_http,
                config.limits.request_timeout,
                config.limits.max_response_bytes,
            )?;
            let mut tokens =
                Tokens::new(&name, &config.auth, &config.uri, config.limits.max_depth)?;
            tokens.set_actor_timeout(config.limits.request_timeout);
            Ok((http, tokens))
        };
        let (http, tokens) =
            start().map_err(|e| e.into_error(&name, "the catalog client", &config.limits))?;
        Ok(Self {
            name,
            config,
            http,
            tokens,
            policy,
            server: Mutex::new(ConfigCache::default()),
        })
    }

    /// The Krishiv catalog name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    fn fail(&self, relation: &str) -> impl FnOnce(RestError) -> KrishivError + '_ {
        let relation = relation.to_owned();
        move |e| e.into_error(&self.name, &relation, &self.config.limits)
    }

    /// The deadline of a call that starts now.
    fn deadline(&self) -> Instant {
        let now = Instant::now();
        // `validate` caps the timeout, so this never saturates in practice.
        now.checked_add(self.config.limits.request_timeout)
            .unwrap_or(now)
    }

    /// One GET within `deadline`: a token, the request, one retry after a 401 when the token can be renewed.
    /// `Ok(None)` is a 404 whose `ErrorModel` type is one of `absent` (the `NoSuch*` answers that fit this call);
    /// any other 404 (a proxy, a wrong prefix, an HTML page) is an error. Stricter than Java, whose handlers read
    /// every 404 of a table call as "no such table".
    fn get(
        &self,
        caller: &RestCaller,
        url: &Url,
        headers: &[(&'static str, String)],
        deadline: Instant,
        absent: &[&str],
    ) -> Result<Option<Value>, RestError> {
        self.get_with(
            &self.tokens,
            caller,
            &mut None,
            url,
            headers,
            deadline,
            absent,
        )
    }

    /// [`Self::get`] with the token of `tokens`, or first with `table_token` (a catalog-supplied table token, design
    /// P3-T067 §4.1: it replaces the caller's token for that table's requests). A 401 for the table token (it
    /// expired) retries once with the caller's token and sets `table_token` to `None`, so the caller can drop it.
    #[expect(
        clippy::too_many_arguments,
        reason = "one GET: its token source, caller, table token, URL, headers, deadline and absent kinds"
    )]
    fn get_with(
        &self,
        tokens: &Tokens,
        caller: &RestCaller,
        table_token: &mut Option<&SecretString>,
        url: &Url,
        headers: &[(&'static str, String)],
        deadline: Instant,
        absent: &[&str],
    ) -> Result<Option<Value>, RestError> {
        let mut retried = false;
        loop {
            let mut request = Request::get(url.clone());
            request.headers = headers.to_vec();
            let bearer = match *table_token {
                Some(t) => Some(t.clone()),
                None => tokens.token(caller, &self.http, deadline)?,
            };
            request.bearer.clone_from(&bearer);
            let mut response = self.http.call(request, deadline)?;
            let result = match response.status {
                200 => types::parse_limited(&response.body, self.config.limits.max_depth).map(Some),
                // Iceberg REST 1.12 lets a catalog answer an expired token with 419 as well as 401.
                401 | 419 if table_token.is_some() => {
                    *table_token = None;
                    wipe_bytes(&mut response.body);
                    continue;
                }
                401 if !retried && tokens.refreshable() => {
                    if let Some(refused) = &bearer {
                        tokens.invalidate(refused);
                    }
                    retried = true;
                    continue;
                }
                404 if absent == ANY_404
                    || types::error_model(&response.body)
                        .0
                        .is_some_and(|k| absent.contains(&k.as_str())) =>
                {
                    Ok(None)
                }
                status => {
                    let (kind, message) = types::error_model(&response.body);
                    Err(RestError::Status {
                        status,
                        kind,
                        message,
                    })
                }
            };
            wipe_bytes(&mut response.body);
            return result;
        }
    }

    /// The server's configuration, fetched on first use. For every mode but `delegated` it is the
    /// catalog's own, fetched with the catalog's token. A delegated catalog has no identity of its
    /// own at the catalog (the server's actor token is for the IdP only and never reaches the
    /// catalog host): `/v1/config` is fetched with the caller's own delegated token (exchanged or
    /// forwarded), as the Iceberg REST spec authenticates it like any call, and kept per
    /// principal (without groups) and exchange claims.
    fn server(&self, caller: &RestCaller, deadline: Instant) -> Result<Arc<Server>, KrishivError> {
        let key = if self.tokens.is_delegated() {
            match (&caller.principal, &caller.subject) {
                (Some(p), Some(s)) => Some(format!("{}:{p}|{:02x?}", p.len(), s.claims)),
                // Refused by the token source before anything is sent.
                _ => caller.identity.clone(),
            }
        } else {
            None
        };
        let slot = || {
            self.server
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        };
        if let Some(s) = slot().get(&key) {
            return Ok(s);
        }
        // Fetched without the lock, so a stalled `/v1/config` holds no other caller; when two callers race, the
        // first answer stored wins.
        let fail = self.fail("the catalog configuration");
        let fetch = || -> Result<Server, RestError> {
            let mut url = join(&self.config.uri, &["v1", "config"])?;
            if let Some(w) = &self.config.warehouse {
                url.query_pairs_mut().append_pair("warehouse", w);
            }
            let value = self
                .get(caller, &url, &[], deadline, &[])?
                .ok_or(RestError::Status {
                    status: 404,
                    kind: Some("NoSuchWarehouseException".to_owned()),
                    message: None,
                })?;
            let config = CatalogConfig::from_json(value)?;
            if config.get("uri").is_some() {
                tracing::info!(
                    target: "krishiv::catalog",
                    catalog = self.name.as_str(),
                    "the Iceberg REST catalog's config names a `uri`; it is ignored (requests go to the declared URI only)"
                );
            }
            let prefix = match config.get("prefix").filter(|p| !p.is_empty()) {
                Some(p) if prefix_is_safe(p) => Some(p.to_owned()),
                Some(_) => {
                    return Err(RestError::Protocol(
                        "an unsafe `prefix` in the catalog config".to_owned(),
                    ));
                }
                None => None,
            };
            let separator = match config.get("namespace-separator") {
                Some(s) if separator_is_safe(s) => s.to_owned(),
                Some(_) => {
                    return Err(RestError::Protocol(
                        "an unsafe `namespace-separator` in the catalog config".to_owned(),
                    ));
                }
                None => DEFAULT_NAMESPACE_SEPARATOR.to_owned(),
            };
            Ok(Server {
                prefix,
                separator,
                config,
            })
        };
        let fetched = Arc::new(fetch().map_err(fail)?);
        Ok(slot().insert(key, fetched))
    }

    /// The server's merged configuration (after the first call).
    ///
    /// # Errors
    /// The `/v1/config` call failed.
    pub fn catalog_config(&self) -> Result<CatalogConfig, KrishivError> {
        Ok(self
            .server(&RestCaller::server(), self.deadline())?
            .config
            .clone())
    }

    fn url(&self, server: &Server, tail: &[&str]) -> Result<Url, RestError> {
        let mut segments = vec!["v1"];
        if let Some(p) = &server.prefix {
            segments.push(p);
        }
        segments.extend_from_slice(tail);
        join(&self.config.uri, &segments)
    }

    /// Every page of a listing, within the page and entry limits. `Ok(None)` is a 404 on the first page.
    fn pages<T>(
        &self,
        caller: &RestCaller,
        base: &Url,
        parse: impl Fn(Value) -> Result<Page<T>, RestError>,
        deadline: Instant,
    ) -> Result<Option<Vec<T>>, RestError> {
        let limits = &self.config.limits;
        let mut out = Vec::new();
        let mut token: Option<String> = None;
        let mut seen = HashSet::new();
        for _ in 0..limits.max_pages {
            let mut url = base.clone();
            if let Some(t) = &token {
                url.query_pairs_mut().append_pair("pageToken", t);
            }
            let Some(value) = self.get(caller, &url, &[], deadline, NO_SUCH_NAMESPACE)? else {
                return if token.is_none() {
                    Ok(None)
                } else {
                    Err(RestError::Protocol(
                        "a later page of a listing is not found".to_owned(),
                    ))
                };
            };
            let page = parse(value)?;
            if out.len().saturating_add(page.items.len()) > limits.max_list_entries {
                return Err(RestError::Limit(RestLimit::Entries));
            }
            out.extend(page.items);
            match page.next.filter(|t| !t.is_empty()) {
                None => return Ok(Some(out)),
                Some(next) => {
                    if next.len() > MAX_PAGE_TOKEN_BYTES {
                        return Err(RestError::Protocol(
                            "a page token longer than 4 KiB".to_owned(),
                        ));
                    }
                    if !seen.insert(next.clone()) {
                        return Err(RestError::Protocol(
                            "a listing repeats its page token".to_owned(),
                        ));
                    }
                    token = Some(next);
                }
            }
        }
        Err(RestError::Limit(RestLimit::Pages))
    }

    /// The namespaces directly under `parent` (top level when empty), each as its levels. `Ok(None)` when `parent`
    /// does not exist.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE` or `KRISHIV_CATALOG_LIMIT_EXCEEDED` naming the catalog.
    pub fn list_namespaces(
        &self,
        parent: &[String],
    ) -> Result<Option<Vec<Vec<String>>>, KrishivError> {
        self.list_namespaces_as(&RestCaller::server(), parent)
    }

    /// [`list_namespaces`](Self::list_namespaces) for `caller`.
    ///
    /// # Errors
    /// As [`list_namespaces`](Self::list_namespaces); `KRISHIV_CATALOG_DELEGATION_UNAVAILABLE` when a delegated
    /// catalog has no usable token for `caller`.
    pub fn list_namespaces_as(
        &self,
        caller: &RestCaller,
        parent: &[String],
    ) -> Result<Option<Vec<Vec<String>>>, KrishivError> {
        // One deadline for the whole call: the config fetch on first use, every page, the token.
        let deadline = self.deadline();
        let server = self.server(caller, deadline)?;
        let what = relation(parent, None);
        let run = || -> Result<Option<Vec<Vec<String>>>, RestError> {
            check_names(parent, None)?;
            let mut url = self.url(&server, &["namespaces"])?;
            if !parent.is_empty() {
                let value =
                    namespace_query(parent, &server.separator).map_err(RestError::Protocol)?;
                url.query_pairs_mut().append_pair("parent", &value);
            }
            self.pages(caller, &url, types::namespaces_page, deadline)
        };
        run().map_err(self.fail(&what))
    }

    /// The tables in `namespace`. `Ok(None)` when the namespace does not exist.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE` or `KRISHIV_CATALOG_LIMIT_EXCEEDED` naming the catalog and the namespace.
    pub fn list_tables(
        &self,
        namespace: &[String],
    ) -> Result<Option<Vec<TableIdent>>, KrishivError> {
        self.list_tables_as(&RestCaller::server(), namespace)
    }

    /// [`list_tables`](Self::list_tables) for `caller`.
    ///
    /// # Errors
    /// As [`list_tables`](Self::list_tables); `KRISHIV_CATALOG_DELEGATION_UNAVAILABLE` when a delegated catalog has
    /// no usable token for `caller`.
    pub fn list_tables_as(
        &self,
        caller: &RestCaller,
        namespace: &[String],
    ) -> Result<Option<Vec<TableIdent>>, KrishivError> {
        // One deadline for the whole call: the config fetch on first use, every page, the token.
        let deadline = self.deadline();
        let server = self.server(caller, deadline)?;
        let what = relation(namespace, None);
        let run = || -> Result<Option<Vec<TableIdent>>, RestError> {
            check_namespace(namespace, None)?;
            let ns = namespace_path(namespace, &server.separator);
            let url = self.url(&server, &["namespaces", &ns, "tables"])?;
            self.pages(caller, &url, types::tables_page, deadline)
        };
        run().map_err(self.fail(&what))
    }

    /// Loads `namespace.table`. `Ok(None)` when it does not exist.
    ///
    /// # Errors
    /// `KRISHIV_CATALOG_FAILURE`/`KRISHIV_CATALOG_LIMIT_EXCEEDED` for a failed call;
    /// `KRISHIV_CATALOG_LOCATION_REFUSED` for a location the policy refuses; `KRISHIV_UNSUPPORTED_FEATURE` for read
    /// restrictions or server-side scan planning.
    pub fn load_table(
        &self,
        namespace: &[String],
        table: &str,
    ) -> Result<Option<LoadedTable>, KrishivError> {
        self.load_table_as(&RestCaller::server(), namespace, table)
    }

    /// [`load_table`](Self::load_table) for `caller`.
    ///
    /// # Errors
    /// As [`load_table`](Self::load_table); `KRISHIV_CATALOG_DELEGATION_UNAVAILABLE` when a delegated catalog has no
    /// usable token for `caller`.
    pub fn load_table_as(
        &self,
        caller: &RestCaller,
        namespace: &[String],
        table: &str,
    ) -> Result<Option<LoadedTable>, KrishivError> {
        self.load_table_snapshots_as(caller, namespace, table, None)
    }

    /// [`load_table_as`](Self::load_table_as) with the `snapshots` query parameter: [`Snapshots::All`] every
    /// snapshot (a commit's probe, design T026/T027 §7.2.1), [`Snapshots::Refs`] only those refs point at; `None`
    /// sends none (the server's default, `all`).
    ///
    /// # Errors
    /// As [`load_table_as`](Self::load_table_as).
    pub fn load_table_snapshots_as(
        &self,
        caller: &RestCaller,
        namespace: &[String],
        table: &str,
        snapshots: Option<Snapshots>,
    ) -> Result<Option<LoadedTable>, KrishivError> {
        self.load_table_with(caller, &mut None, namespace, table, snapshots)
    }

    /// [`load_table_as`](Self::load_table_as), sent with the catalog-supplied `table_token` of this table when given
    /// (design P3-T067 §4.1; a 401 for it falls back to the caller's token and sets `table_token` to `None`).
    pub(crate) fn load_table_for(
        &self,
        caller: &RestCaller,
        table_token: &mut Option<&SecretString>,
        namespace: &[String],
        table: &str,
    ) -> Result<Option<LoadedTable>, KrishivError> {
        self.load_table_with(caller, table_token, namespace, table, None)
    }

    /// The one `loadTable` request: `table_token` as [`load_table_for`](Self::load_table_for), `snapshots` as
    /// [`load_table_snapshots_as`](Self::load_table_snapshots_as).
    fn load_table_with(
        &self,
        caller: &RestCaller,
        table_token: &mut Option<&SecretString>,
        namespace: &[String],
        table: &str,
        snapshots: Option<Snapshots>,
    ) -> Result<Option<LoadedTable>, KrishivError> {
        // One deadline for the whole call: the config fetch on first use, every page, the token.
        let deadline = self.deadline();
        let server = self.server(caller, deadline)?;
        let what = relation(namespace, Some(table));
        let mut run = || -> Result<Option<LoadedTable>, RestError> {
            check_namespace(namespace, Some(table))?;
            let ns = namespace_path(namespace, &server.separator);
            let name = path_segment(table);
            let mut url = self.url(&server, &["namespaces", &ns, "tables", &name])?;
            if let Some(s) = snapshots {
                url.query_pairs_mut().append_pair("snapshots", s.as_str());
            }
            let headers = self.delegation_headers();
            self.get_with(
                &self.tokens,
                caller,
                &mut *table_token,
                &url,
                &headers,
                deadline,
                NO_SUCH_TABLE,
            )?
            .map(LoadedTable::from_json)
            .transpose()
        };
        let Some(loaded) = run().map_err(self.fail(&what))? else {
            return Ok(None);
        };
        // Krishiv's own text (the analyzer shows it, unlike a provider's failure).
        let unsupported = |feature: &str| -> KrishivError {
            KrishivCatalogTableUnsupported {
                catalog: quoted(&self.name),
                relation_name: what.clone(),
                feature: feature.to_owned(),
            }
            .into()
        };
        if loaded.restricted {
            return Err(unsupported("Iceberg REST read restrictions"));
        }
        if loaded
            .config
            .get("scan-planning-mode")
            .is_some_and(|m| m.expose_secret().eq_ignore_ascii_case("server"))
        {
            return Err(unsupported("Iceberg REST server-side scan planning"));
        }
        self.check_locations(&what, &loaded)?;
        Ok(Some(loaded))
    }

    /// `X-Iceberg-Access-Delegation` as `loadTable` and a commit send it.
    fn delegation_headers(&self) -> Vec<(&'static str, String)> {
        if self.config.vended_credentials {
            vec![(
                "X-Iceberg-Access-Delegation",
                "vended-credentials".to_owned(),
            )]
        } else {
            Vec::new()
        }
    }

    /// Sends one commit request (`updateTable`, `POST /v1/{prefix}/namespaces/{ns}/tables/{table}`) with `body`
    /// (a `CommitTableRequest`, JSON) for `caller`: the same token, egress policy and
    /// `X-Iceberg-Access-Delegation` as [`load_table_as`](Self::load_table_as), one deadline. The request is
    /// **never retried** except once after a 401 when the token can be renewed (a 401 means the catalog did not
    /// apply it); a 5xx, a timeout or a reset is returned as it is, for the commit loop to probe. A 200's
    /// `metadata-location` and `location` pass [`check_location`]: a refused one is a failure after sending (the
    /// commit may stand, so the caller must treat it as unknown).
    ///
    /// # Errors
    /// No whole answer, or a refused location in a 200 ([`CommitFailure::sent`] says whether the request may have
    /// reached the catalog); a body over `max_response_bytes` is refused before sending.
    pub fn commit_table_as(
        &self,
        caller: &RestCaller,
        namespace: &[String],
        table: &str,
        body: &[u8],
    ) -> Result<CommitReply, CommitFailure> {
        let what = relation(namespace, Some(table));
        let unsent = |reason: String| CommitFailure {
            sent: false,
            reason,
        };
        if body.len() > self.config.limits.max_response_bytes {
            return Err(unsent(format!(
                "a commit request of {} bytes is over the {}-byte limit",
                body.len(),
                self.config.limits.max_response_bytes
            )));
        }
        let deadline = self.deadline();
        let server = self
            .server(caller, deadline)
            .map_err(|e| unsent(format!("the catalog configuration: {e}")))?;
        let url = check_namespace(namespace, Some(table))
            .and_then(|()| {
                let ns = namespace_path(namespace, &server.separator);
                let name = path_segment(table);
                self.url(&server, &["namespaces", &ns, "tables", &name])
            })
            .map_err(|e| unsent(e.detail()))?;
        let headers = self.delegation_headers();
        let mut retried = false;
        loop {
            let bearer = self
                .tokens
                .token(caller, &self.http, deadline)
                .map_err(|e| unsent(format!("no token: {}", e.detail())))?;
            let mut request = Request::get(url.clone());
            request.post = true;
            request.json = Some(body.to_vec());
            request.headers.clone_from(&headers);
            request.bearer.clone_from(&bearer);
            let response = match self.http.call_tracked(request, deadline) {
                Ok(r) => r,
                Err(f) => {
                    tracing::warn!(
                        target: "krishiv::catalog",
                        catalog = self.name.as_str(),
                        table = what.as_str(),
                        sent = f.sent,
                        detail = f.error.detail().as_str(),
                        "an Iceberg REST commit request got no answer"
                    );
                    return Err(CommitFailure {
                        sent: f.sent,
                        reason: f.error.detail(),
                    });
                }
            };
            if response.status == 401 && !retried && self.tokens.refreshable() {
                if let Some(refused) = &bearer {
                    self.tokens.invalidate(refused);
                }
                retried = true;
                continue;
            }
            if response.status == 200 {
                // A location the policy refuses is never handed on, whatever the catalog applied.
                // A 200 that does not validate is unknown, not committed: the connector probes it.
                let loaded = types::parse_limited(&response.body, self.config.limits.max_depth)
                    .and_then(LoadedTable::from_json)
                    .map_err(|e| CommitFailure {
                        sent: true,
                        reason: format!("a 200 answer that does not validate ({})", e.detail()),
                    })?;
                self.check_locations(&what, &loaded)
                    .map_err(|e| CommitFailure {
                        sent: true,
                        reason: e.to_string(),
                    })?;
            }
            return Ok(CommitReply {
                status: response.status,
                retry_after: response.retry_after,
                body: response.body,
            });
        }
    }

    fn check_locations(&self, what: &str, t: &LoadedTable) -> Result<(), KrishivError> {
        let policy = &*self.policy;
        if let Some(m) = &t.metadata_location {
            check_location(&self.name, what, m, policy)?;
        }
        check_location(&self.name, what, &t.location, policy)?;
        for c in &t.storage_credentials {
            check_prefix(&self.name, what, &c.prefix, policy)?;
        }
        Ok(())
    }

    /// The vended credentials of `namespace.table` (`loadCredentials`), when the server advertises the endpoint.
    /// `Ok(None)` when the table does not exist.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` when the server does not advertise `loadCredentials`; otherwise as
    /// [`load_table`](Self::load_table).
    pub fn load_credentials(
        &self,
        namespace: &[String],
        table: &str,
    ) -> Result<Option<Vec<StorageCredential>>, KrishivError> {
        self.load_credentials_as(&RestCaller::server(), namespace, table)
    }

    /// [`load_credentials`](Self::load_credentials) for `caller`.
    ///
    /// # Errors
    /// As [`load_credentials`](Self::load_credentials); `KRISHIV_CATALOG_DELEGATION_UNAVAILABLE` when a delegated
    /// catalog has no usable token for `caller`.
    pub fn load_credentials_as(
        &self,
        caller: &RestCaller,
        namespace: &[String],
        table: &str,
    ) -> Result<Option<Vec<StorageCredential>>, KrishivError> {
        self.load_credentials_for(caller, &mut None, namespace, table)
    }

    /// [`load_credentials_as`](Self::load_credentials_as), sent with the catalog-supplied `table_token` of this table
    /// when given (design P3-T067 §4.1; a 401 for it falls back to the caller's token and sets `table_token` to
    /// `None`).
    pub(crate) fn load_credentials_for(
        &self,
        caller: &RestCaller,
        table_token: &mut Option<&SecretString>,
        namespace: &[String],
        table: &str,
    ) -> Result<Option<Vec<StorageCredential>>, KrishivError> {
        // One deadline for the whole call: the config fetch on first use, every page, the token.
        let deadline = self.deadline();
        let server = self.server(caller, deadline)?;
        let what = relation(namespace, Some(table));
        if !server.config.supports(LOAD_CREDENTIALS) {
            return Err(KrishivUnsupportedFeature {
                feature: format!(
                    "loading credentials from the Iceberg REST catalog {}",
                    quoted(&self.name)
                ),
            }
            .into());
        }
        let mut run = || -> Result<Option<Vec<StorageCredential>>, RestError> {
            check_namespace(namespace, Some(table))?;
            let ns = namespace_path(namespace, &server.separator);
            let name = path_segment(table);
            let url = self.url(
                &server,
                &["namespaces", &ns, "tables", &name, "credentials"],
            )?;
            self.get_with(
                &self.tokens,
                caller,
                &mut *table_token,
                &url,
                &[],
                deadline,
                // Any 404 of the credentials route itself is `None`, not only a `NoSuch*` one: the caller reloads
                // the table, which tells whether it is really gone (a catalog may advertise the route and answer it
                // with a plain 404). A 404 of anything else on the way (the token endpoint) stays a failure.
                ANY_404,
            )?
            .map(types::credentials)
            .transpose()
        };
        let found = run().map_err(self.fail(&what))?;
        let Some(creds) = found else {
            return Ok(None);
        };
        for c in &creds {
            check_prefix(&self.name, &what, &c.prefix, &*self.policy)?;
        }
        Ok(Some(creds))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_join_under_the_base_path_and_never_change_the_host() {
        let base = Url::parse("https://onelake.table.fabric.microsoft.com/iceberg").unwrap();
        let u = join(
            &base,
            &[
                "v1",
                "ws/item.Lakehouse",
                "namespaces",
                "a%1Fb",
                "tables",
                "t%20x",
            ],
        )
        .unwrap();
        assert_eq!(
            u.as_str(),
            "https://onelake.table.fabric.microsoft.com/iceberg/v1/ws/item.Lakehouse/namespaces/a%1Fb/tables/t%20x"
        );
        let base = Url::parse("http://127.0.0.1:8181/").unwrap();
        assert_eq!(
            join(&base, &["v1", "config"]).unwrap().as_str(),
            "http://127.0.0.1:8181/v1/config"
        );
    }

    /// Review 58b1-r2 N3: past the cap the least recently used answer goes, not all of them.
    #[test]
    fn the_config_cache_evicts_the_least_recently_used() {
        let server = || {
            Arc::new(Server {
                prefix: None,
                separator: DEFAULT_NAMESPACE_SEPARATOR.to_owned(),
                config: CatalogConfig::default(),
            })
        };
        let mut cache = ConfigCache::default();
        for i in 0..MAX_CONFIGS {
            cache.insert(Some(i.to_string()), server());
        }
        // Touch the oldest: the second oldest is the one evicted.
        assert!(cache.get(&Some("0".to_owned())).is_some());
        cache.insert(Some("new".to_owned()), server());
        assert_eq!(cache.len(), MAX_CONFIGS);
        assert!(cache.get(&Some("0".to_owned())).is_some());
        assert!(cache.get(&Some("1".to_owned())).is_none());
        assert!(cache.get(&Some("new".to_owned())).is_some());
    }

    #[test]
    fn relations_are_quoted() {
        assert_eq!(
            relation(&["a".into(), "b`c".into()], Some("t")),
            "`a`.`b``c`.`t`"
        );
        assert_eq!(relation(&[], None), "the namespace listing");
    }
}
