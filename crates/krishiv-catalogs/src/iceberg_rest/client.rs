// SPDX-License-Identifier: Apache-2.0
//! [`RestClient`]: Krishiv's own read client of one Iceberg REST catalog (decision 8, T067 §3.4, owner decision
//! 2026-11-15): `GET /v1/config`, the namespace and table listings, `loadTable` and `loadCredentials`.
//!
//! - **Config.** Fetched once, on first use (with the declared `warehouse`). Only `prefix` and
//!   `namespace-separator` are taken from it, after a check that neither can leave `/v1/` (they go into the path as
//!   the server wrote them, as in Java's `ResourcePaths`). A `uri` it returns is ignored and logged: the client
//!   only ever talks to the declared host, so the caller's token never goes anywhere else.
//! - **Tokens.** One source per catalog, asked per request (`auth`); a 401 with a client-credentials token drops it
//!   and retries once.
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

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_common::secret::wipe_bytes;
use serde_json::Value;
use url::Url;

use super::auth::Tokens;
use super::config::RestConfig;
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
    server: Mutex<Option<Arc<Server>>>,
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
            Ok((
                http,
                Tokens::new(&config.auth, &config.uri, config.limits.max_depth)?,
            ))
        };
        let (http, tokens) =
            start().map_err(|e| e.into_error(&name, "the catalog client", &config.limits))?;
        Ok(Self {
            name,
            config,
            http,
            tokens,
            policy,
            server: Mutex::new(None),
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
        url: &Url,
        headers: &[(&'static str, String)],
        deadline: Instant,
        absent: &[&str],
    ) -> Result<Option<Value>, RestError> {
        let mut retried = false;
        loop {
            let mut request = Request::get(url.clone());
            request.headers = headers.to_vec();
            let bearer = self.tokens.token(&self.http, deadline)?;
            request.bearer.clone_from(&bearer);
            let mut response = self.http.call(request, deadline)?;
            let result = match response.status {
                200 => types::parse_limited(&response.body, self.config.limits.max_depth).map(Some),
                401 if !retried && self.tokens.refreshable() => {
                    if let Some(refused) = &bearer {
                        self.tokens.invalidate(refused);
                    }
                    retried = true;
                    continue;
                }
                404 if types::error_model(&response.body)
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

    /// The server's configuration, fetched on first use.
    fn server(&self, deadline: Instant) -> Result<Arc<Server>, KrishivError> {
        let slot = || {
            self.server
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        };
        if let Some(s) = slot().as_ref() {
            return Ok(Arc::clone(s));
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
                .get(&url, &[], deadline, &[])?
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
        Ok(Arc::clone(slot().get_or_insert(fetched)))
    }

    /// The server's merged configuration (after the first call).
    ///
    /// # Errors
    /// The `/v1/config` call failed.
    pub fn catalog_config(&self) -> Result<CatalogConfig, KrishivError> {
        Ok(self.server(self.deadline())?.config.clone())
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
            let Some(value) = self.get(&url, &[], deadline, NO_SUCH_NAMESPACE)? else {
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
        // One deadline for the whole call: the config fetch on first use, every page, the token.
        let deadline = self.deadline();
        let server = self.server(deadline)?;
        let what = relation(parent, None);
        let run = || -> Result<Option<Vec<Vec<String>>>, RestError> {
            check_names(parent, None)?;
            let mut url = self.url(&server, &["namespaces"])?;
            if !parent.is_empty() {
                let value =
                    namespace_query(parent, &server.separator).map_err(RestError::Protocol)?;
                url.query_pairs_mut().append_pair("parent", &value);
            }
            self.pages(&url, types::namespaces_page, deadline)
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
        // One deadline for the whole call: the config fetch on first use, every page, the token.
        let deadline = self.deadline();
        let server = self.server(deadline)?;
        let what = relation(namespace, None);
        let run = || -> Result<Option<Vec<TableIdent>>, RestError> {
            check_namespace(namespace, None)?;
            let ns = namespace_path(namespace, &server.separator);
            let url = self.url(&server, &["namespaces", &ns, "tables"])?;
            self.pages(&url, types::tables_page, deadline)
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
        // One deadline for the whole call: the config fetch on first use, every page, the token.
        let deadline = self.deadline();
        let server = self.server(deadline)?;
        let what = relation(namespace, Some(table));
        let run = || -> Result<Option<LoadedTable>, RestError> {
            check_namespace(namespace, Some(table))?;
            let ns = namespace_path(namespace, &server.separator);
            let name = path_segment(table);
            let url = self.url(&server, &["namespaces", &ns, "tables", &name])?;
            let headers = if self.config.vended_credentials {
                vec![(
                    "X-Iceberg-Access-Delegation",
                    "vended-credentials".to_owned(),
                )]
            } else {
                Vec::new()
            };
            self.get(&url, &headers, deadline, NO_SUCH_TABLE)?
                .map(LoadedTable::from_json)
                .transpose()
        };
        let Some(loaded) = run().map_err(self.fail(&what))? else {
            return Ok(None);
        };
        if loaded.restricted {
            return Err(KrishivUnsupportedFeature {
                feature: format!("Iceberg REST read restrictions on table {what}"),
            }
            .into());
        }
        if loaded
            .config
            .get("scan-planning-mode")
            .is_some_and(|m| m.expose_secret().eq_ignore_ascii_case("server"))
        {
            return Err(KrishivUnsupportedFeature {
                feature: format!("Iceberg REST server-side scan planning for table {what}"),
            }
            .into());
        }
        self.check_locations(&what, &loaded)?;
        Ok(Some(loaded))
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
        // One deadline for the whole call: the config fetch on first use, every page, the token.
        let deadline = self.deadline();
        let server = self.server(deadline)?;
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
        let run = || -> Result<Option<Vec<StorageCredential>>, RestError> {
            check_namespace(namespace, Some(table))?;
            let ns = namespace_path(namespace, &server.separator);
            let name = path_segment(table);
            let url = self.url(
                &server,
                &["namespaces", &ns, "tables", &name, "credentials"],
            )?;
            self.get(&url, &[], deadline, NO_SUCH_TABLE)?
                .map(types::credentials)
                .transpose()
        };
        let Some(creds) = run().map_err(self.fail(&what))? else {
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

    #[test]
    fn relations_are_quoted() {
        assert_eq!(
            relation(&["a".into(), "b`c".into()], Some("t")),
            "`a`.`b``c`.`t`"
        );
        assert_eq!(relation(&[], None), "the namespace listing");
    }
}
