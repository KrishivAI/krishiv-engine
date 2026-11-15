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
//! - **Limits.** Each request has the whole `request_timeout`; listings stop at `max_pages` and `max_list_entries`
//!   and refuse a page token the server repeats.
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
    DEFAULT_NAMESPACE_SEPARATOR, json_depth_within, namespace_path, namespace_query, path_segment,
    prefix_is_safe, separator_is_safe,
};
use super::error::{RestError, RestLimit, quoted};
use super::transport::{Request, Transport};
use super::types::{self, CatalogConfig, LoadedTable, Page, StorageCredential, TableIdent};
use crate::location::{LocationPolicy, check_location};

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
                config.allow_http,
                config.limits.request_timeout,
                config.limits.max_response_bytes,
            )?;
            Ok((http, Tokens::new(&config.auth, &config.uri)?))
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

    /// One GET: a token, the request, one retry after a 401 when the token can be renewed. `Ok(None)` is a 404.
    fn get(
        &self,
        url: &Url,
        headers: &[(&'static str, String)],
    ) -> Result<Option<Value>, RestError> {
        let deadline = Instant::now() + self.config.limits.request_timeout;
        let mut retried = false;
        loop {
            let mut request = Request::get(url.clone());
            request.headers = headers.to_vec();
            request.bearer = self.tokens.token(&self.http, deadline)?;
            let mut response = self.http.call(request, deadline)?;
            let result = match response.status {
                200 => {
                    if json_depth_within(&response.body, self.config.limits.max_depth) {
                        types::parse(&response.body).map(Some)
                    } else {
                        Err(RestError::Limit(RestLimit::Depth))
                    }
                }
                404 => Ok(None),
                401 if !retried && self.tokens.refreshable() => {
                    self.tokens.invalidate();
                    retried = true;
                    continue;
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
    fn server(&self) -> Result<Arc<Server>, KrishivError> {
        let mut slot = self
            .server
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(s) = slot.as_ref() {
            return Ok(Arc::clone(s));
        }
        let fail = self.fail("the catalog configuration");
        let fetch = || -> Result<Server, RestError> {
            let mut url = join(&self.config.uri, &["v1", "config"])?;
            if let Some(w) = &self.config.warehouse {
                url.query_pairs_mut().append_pair("warehouse", w);
            }
            let value = self.get(&url, &[])?.ok_or(RestError::Status {
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
        let server = Arc::new(fetch().map_err(fail)?);
        *slot = Some(Arc::clone(&server));
        Ok(server)
    }

    /// The server's merged configuration (after the first call).
    ///
    /// # Errors
    /// The `/v1/config` call failed.
    pub fn catalog_config(&self) -> Result<CatalogConfig, KrishivError> {
        Ok(self.server()?.config.clone())
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
            let Some(value) = self.get(&url, &[])? else {
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
        let server = self.server()?;
        let what = relation(parent, None);
        let run = || -> Result<Option<Vec<Vec<String>>>, RestError> {
            let mut url = self.url(&server, &["namespaces"])?;
            if !parent.is_empty() {
                let value =
                    namespace_query(parent, &server.separator).map_err(RestError::Protocol)?;
                url.query_pairs_mut().append_pair("parent", &value);
            }
            self.pages(&url, types::namespaces_page)
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
        let server = self.server()?;
        let what = relation(namespace, None);
        let run = || -> Result<Option<Vec<TableIdent>>, RestError> {
            let ns = namespace_path(namespace, &server.separator);
            let url = self.url(&server, &["namespaces", &ns, "tables"])?;
            self.pages(&url, types::tables_page)
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
        let server = self.server()?;
        let what = relation(namespace, Some(table));
        let run = || -> Result<Option<LoadedTable>, RestError> {
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
            self.get(&url, &headers)?
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
            check_location(&self.name, what, &c.prefix, policy)?;
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
        let server = self.server()?;
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
            let ns = namespace_path(namespace, &server.separator);
            let name = path_segment(table);
            let url = self.url(
                &server,
                &["namespaces", &ns, "tables", &name, "credentials"],
            )?;
            self.get(&url, &[])?.map(types::credentials).transpose()
        };
        let Some(creds) = run().map_err(self.fail(&what))? else {
            return Ok(None);
        };
        for c in &creds {
            check_location(&self.name, &what, &c.prefix, &*self.policy)?;
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
