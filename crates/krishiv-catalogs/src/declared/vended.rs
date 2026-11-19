// SPDX-License-Identifier: Apache-2.0
//! Reads of a declared Iceberg REST table with `storage = vended` (design P3-T067 §4.1, §4.3, §4.4).
//!
//! - **Answers.** The vended part of a `loadTable` answer (its `storage-credentials`, the credential keys of its
//!   `config`, the table location, when it was fetched) is kept in [`VendedCache`] per identity key and table: the
//!   caller's on a `delegated` catalog, the catalog's otherwise, never shared across identity keys. The cache is one
//!   per process, bounded by `krishiv.catalog.vended.maxEntries` (least recently used out first); an entry is dropped
//!   at its answer's expiry (the earliest credential expiry, else fetched plus `krishiv.catalog.vendedDefaultTtl`),
//!   on `REFRESH TABLE`, when its catalog is dropped, or on eviction. Not when the last session of its identity
//!   ends: the expiry bounds it (design §4.3 as amended in unit 3).
//! - **Table token.** A `token` in the `loadTable` answer's `config` is kept with the answer and sent as the bearer
//!   of that table's `loadCredentials` and reload only (the declared catalog host; a 401 falls back to the caller's
//!   token), never to the token endpoint or to storage (§4.1).
//! - **Refresh.** [`VendedSource`] is the `CredentialSource` of one scope of one query: the cached credential of the
//!   longest prefix covering the scope while it is valid for longer than the refresh skew, else a new answer from
//!   `loadCredentials` with the same caller, or a reload of the table when the catalog does not advertise the endpoint
//!   or answers it 404. A second fetch while the cache still holds the answer this source last served means the
//!   store refused that credential (`CredentialCache::invalidate`), so it fetches a new answer instead of serving it
//!   again (§4.3, the forced refresh after a 403).
//!   Refresh endpoints and signer URLs a catalog sends are never followed (they are not even kept).
//! - **Scopes.** One store scope per vended prefix: a full URL or bucket prefix is its own scope; a bare scheme prefix
//!   becomes the buckets of the table's location and metadata location in that family; with no `storage-credentials`,
//!   the table location. [`VendedPaths`] refuses any location outside those scopes
//!   (`KRISHIV_CATALOG_LOCATION_REFUSED`, before any request), so no file of the table is read with the server's own
//!   or ambient credentials. A vended prefix opens nothing to path reads: only this table's scans use these stores.
//! - **Block-cache scope (§4.4).** `Principal` of the length-prefixed (catalog, principal id, tenant) on a `delegated`
//!   catalog, of the catalog alone otherwise; `delegated_by` is left out.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_common::secret::SecretString;
use krishiv_exec::files::{LocatedPath, PathLocator};
use krishiv_session::catalog_registry::IdentityKey;
use krishiv_store::{
    CacheScope, CredentialScope, CredentialSource, ParsedUrl, ScopedCredential, parse_object_url,
};

use crate::iceberg_rest::types::StorageCredential;
use crate::iceberg_rest::vended::{
    VendedPrefix, answer_expires_at, covering, credential, keep_credentials, table_token,
};
use crate::iceberg_rest::{LoadedTable, RestCaller, RestClient};
use crate::location::{LocationRule, refused};

/// A vended-answer cache key: the catalog instance (two runtimes declaring a catalog of one name against different
/// servers never share an answer), the identity, the namespace and the table as asked.
pub(crate) type Key = (u64, IdentityKey, Vec<String>, String);

/// The vended part of one catalog answer (module docs). Holds secrets: its `Debug` shows none.
pub(crate) struct VendedAnswer {
    credentials: Vec<StorageCredential>,
    config: BTreeMap<String, SecretString>,
    location: String,
    fetched: SystemTime,
    /// The table's own bearer token from the `loadTable` answer (§4.1), when the catalog sent one.
    table_token: Option<SecretString>,
}

impl std::fmt::Debug for VendedAnswer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VendedAnswer")
            .field("credentials", &self.credentials.len())
            .finish_non_exhaustive()
    }
}

impl VendedAnswer {
    /// The vended part of a `loadTable` answer.
    pub(crate) fn of_table(t: &LoadedTable) -> Self {
        let mut answer = Self::new(
            t.storage_credentials.clone(),
            t.config.clone(),
            t.location.clone(),
        );
        answer.table_token = table_token(&t.config);
        answer
    }

    fn new(
        credentials: Vec<StorageCredential>,
        config: BTreeMap<String, SecretString>,
        location: String,
    ) -> Self {
        Self {
            table_token: None,
            credentials: credentials
                .into_iter()
                .map(|c| StorageCredential {
                    prefix: c.prefix,
                    config: keep_credentials(c.config),
                })
                .collect(),
            config: keep_credentials(config),
            location,
            fetched: SystemTime::now(),
        }
    }

    /// The table location the answer was given for.
    pub(crate) fn location(&self) -> &str {
        &self.location
    }

    /// When the answer stops being good: its cache entry is dropped then (module docs).
    fn expires_at(&self, default_ttl: Duration) -> SystemTime {
        answer_expires_at(
            self.credentials
                .iter()
                .map(|c| &c.config)
                .chain(std::iter::once(&self.config)),
            self.fetched,
            default_ttl,
        )
    }

    /// Whether the answer vends anything at all.
    pub(crate) fn is_empty(&self) -> bool {
        self.credentials.is_empty() && self.config.is_empty()
    }
}

/// One cached answer: the answer, the tick it was last used at, when it expires.
type Entry = (Arc<VendedAnswer>, u64, SystemTime);

#[derive(Debug, Default)]
struct State {
    entries: HashMap<Key, Entry>,
    tick: u64,
    /// Catalog instances that were dropped: an answer fetched for one by a query still running is never kept.
    /// Instance ids are never reused, so this only grows by one id per dropped catalog.
    dead: HashSet<u64>,
}

impl State {
    fn drop_expired(&mut self, now: SystemTime) {
        self.entries.retain(|_, (_, _, expires)| *expires > now);
    }

    fn put(&mut self, key: Key, answer: Arc<VendedAnswer>, expires: SystemTime, max: usize) {
        if self.dead.contains(&key.0) {
            return;
        }
        let now = SystemTime::now();
        self.drop_expired(now);
        if expires <= now {
            // Already expired: the next `get` would drop it, so it must not evict a live entry when full. It still
            // supersedes what the key held.
            self.entries.remove(&key);
            return;
        }
        self.tick += 1;
        // The least-recently-used scan is O(entries) under the lock once full; fine at the default 10,000 (it runs
        // once per query-table open that missed), revisit if `maxEntries` grows by orders of magnitude.
        while self.entries.len() >= max.max(1) && !self.entries.contains_key(&key) {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, (_, used, _))| *used)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            self.entries.remove(&oldest);
        }
        self.entries.insert(key, (answer, self.tick, expires));
    }
}

/// The vended answers of every declared catalog of the process (module docs).
#[derive(Debug, Default)]
pub(crate) struct VendedCache {
    inner: Mutex<State>,
}

impl VendedCache {
    /// The process's cache.
    pub(crate) fn global() -> Arc<Self> {
        static GLOBAL: OnceLock<Arc<VendedCache>> = OnceLock::new();
        Arc::clone(GLOBAL.get_or_init(|| Arc::new(Self::default())))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The answer held for `key` unless it expired (an expired one is dropped).
    pub(crate) fn get(&self, key: &Key) -> Option<Arc<VendedAnswer>> {
        let mut state = self.lock();
        state.tick += 1;
        let now = state.tick;
        match state.entries.get_mut(key) {
            Some((_, _, expires)) if *expires <= SystemTime::now() => {
                state.entries.remove(key);
                None
            }
            Some((a, used, _)) => {
                *used = now;
                Some(Arc::clone(a))
            }
            None => None,
        }
    }

    /// Keeps `answer` for `key` until its expiry (`default_ttl` for a credential without one), the least recently
    /// used entry out first past `max` entries; expired entries are dropped first.
    pub(crate) fn put(
        &self,
        key: Key,
        answer: Arc<VendedAnswer>,
        default_ttl: Duration,
        max: usize,
    ) {
        let expires = answer.expires_at(default_ttl);
        self.lock().put(key, answer, expires, max);
    }

    /// Keeps `answer` for `key` unless the entry held was fetched later (a `loadTable` answer that raced a refresh),
    /// under one lock.
    pub(crate) fn put_if_newer(
        &self,
        key: Key,
        answer: Arc<VendedAnswer>,
        default_ttl: Duration,
        max: usize,
    ) {
        let expires = answer.expires_at(default_ttl);
        let mut state = self.lock();
        let newer = state
            .entries
            .get(&key)
            .is_none_or(|(held, _, _)| held.fetched < answer.fetched);
        if newer {
            state.put(key, answer, expires, max);
        }
    }

    /// Drops every identity's answer for `namespace.name` of the catalog `instance` (`REFRESH TABLE`).
    pub(crate) fn drop_table(&self, instance: u64, namespace: &[String], name: &str) {
        self.lock()
            .entries
            .retain(|(i, _, ns, n), _| !(*i == instance && ns == namespace && n == name));
    }

    /// Drops every answer of the catalog `instance` (the catalog was dropped); none is kept for it later.
    pub(crate) fn drop_instance(&self, instance: u64) {
        let mut state = self.lock();
        state.dead.insert(instance);
        state.entries.retain(|(i, _, _, _), _| *i != instance);
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lock().entries.len()
    }
}

/// How one table's credentials are asked for again, and how its faults are named.
#[derive(Clone)]
pub(crate) struct Refetch {
    pub(crate) client: Arc<RestClient>,
    pub(crate) caller: RestCaller,
    pub(crate) namespace: Vec<String>,
    pub(crate) table: String,
    pub(crate) catalog: String,
    pub(crate) relation: String,
    pub(crate) location: String,
}

impl std::fmt::Debug for Refetch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Refetch")
            .field("relation", &self.relation)
            .finish_non_exhaustive()
    }
}

impl Refetch {
    fn refused(&self, rule: LocationRule) -> KrishivError {
        refused(&self.catalog, &self.relation, rule)
    }

    /// A new answer: `loadCredentials`, or the table reloaded when the catalog does not advertise it or answers it
    /// 404 (the reload tells whether the table is really gone). Both are sent with the table's own `token` when the
    /// last answer carried one; a `loadCredentials` answer keeps that token unless the catalog refused it (401), in
    /// which case neither the reload nor the new answer carries it.
    fn fetch_blocking(&self, token: Option<&SecretString>) -> Result<VendedAnswer, KrishivError> {
        let (ns, t) = (&self.namespace, self.table.as_str());
        let mut token = token;
        let reload = |mut token: Option<&SecretString>| match self.client.load_table_for(
            &self.caller,
            &mut token,
            ns,
            t,
        )? {
            Some(table) => Ok(VendedAnswer::of_table(&table)),
            None => Err(self.refused(LocationRule::NoVendedCredential)),
        };
        match self
            .client
            .load_credentials_for(&self.caller, &mut token, ns, t)
        {
            Ok(Some(credentials)) => {
                let mut answer =
                    VendedAnswer::new(credentials, BTreeMap::new(), self.location.clone());
                answer.table_token = token.cloned();
                Ok(answer)
            }
            Ok(None) => reload(token),
            Err(e) if e.error_class() == ErrorClass::KrishivUnsupportedFeature => reload(token),
            Err(e) => Err(e),
        }
    }

    /// [`Self::fetch_blocking`] off the compute workers.
    pub(crate) async fn fetch(
        &self,
        token: Option<SecretString>,
    ) -> Result<VendedAnswer, KrishivError> {
        let this = self.clone();
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => handle
                .spawn_blocking(move || this.fetch_blocking(token.as_ref()))
                .await
                .map_err(|_| self.refused(LocationRule::NoVendedCredential))?,
            Err(_) => this.fetch_blocking(token.as_ref()),
        }
    }
}

/// What one table's vended reads share: the cache, its key and bound, the default TTL and the refresh skew.
#[derive(Debug, Clone)]
pub(crate) struct VendedTable {
    pub(crate) cache: Arc<VendedCache>,
    pub(crate) key: Key,
    pub(crate) max_entries: usize,
    pub(crate) default_ttl: Duration,
    pub(crate) skew: Duration,
    pub(crate) refetch: Refetch,
}

impl VendedTable {
    /// The cached answer, else a new one (kept).
    pub(crate) async fn answer(&self) -> Result<Arc<VendedAnswer>, KrishivError> {
        if let Some(a) = self.cache.get(&self.key) {
            return Ok(a);
        }
        self.refreshed(None).await
    }

    /// A new answer, kept; `previous` is the answer it replaces (its table token is sent with the refresh).
    async fn refreshed(
        &self,
        previous: Option<&VendedAnswer>,
    ) -> Result<Arc<VendedAnswer>, KrishivError> {
        let token = previous.and_then(|p| p.table_token.clone());
        let fresh = Arc::new(self.refetch.fetch(token).await?);
        self.cache.put(
            self.key.clone(),
            Arc::clone(&fresh),
            self.default_ttl,
            self.max_entries,
        );
        Ok(fresh)
    }

    /// Keeps the `loadTable` answer `answer` unless the cache holds a newer one.
    pub(crate) fn offer(&self, answer: Arc<VendedAnswer>) {
        self.cache
            .put_if_newer(self.key.clone(), answer, self.default_ttl, self.max_entries);
    }

    /// The credential `answer` gives `url`, `None` when it gives none.
    fn pick(
        &self,
        answer: &VendedAnswer,
        url: &ParsedUrl,
    ) -> Result<Option<ScopedCredential>, KrishivError> {
        let fault = |detail: String| {
            tracing::warn!(target: "krishiv::catalog", catalog = self.refetch.catalog.as_str(), table = self.refetch.relation.as_str(), detail, "an Iceberg REST catalog vended a credential Krishiv cannot use");
            self.refetch.refused(LocationRule::NoVendedCredential)
        };
        let Some((_, config)) =
            covering(&answer.credentials, &answer.config, &answer.location, url)
                .map_err(|f| fault(f.0))?
        else {
            return Ok(None);
        };
        credential(config, url, answer.fetched, self.default_ttl).map_err(|f| fault(f.0))
    }

    /// The store scopes of `answer` (module docs): each as a location text, deduplicated, in a stable order.
    pub(crate) fn scopes(
        &self,
        answer: &VendedAnswer,
        metadata_location: &str,
    ) -> Result<Vec<String>, KrishivError> {
        let fault = || self.refetch.refused(LocationRule::NoVendedCredential);
        let mut out = Vec::new();
        let mut add = |s: String| {
            if !out.contains(&s) {
                out.push(s);
            }
        };
        if answer.credentials.is_empty() {
            add(answer.location.clone());
            return Ok(out);
        }
        for c in &answer.credentials {
            match VendedPrefix::parse(&c.prefix).map_err(|_| fault())? {
                VendedPrefix::Scheme(family) => {
                    for own in [answer.location.as_str(), metadata_location] {
                        if let Ok(url) = parse_object_url(own)
                            && url.scheme == family
                        {
                            add(url.store_id.key().to_owned());
                        }
                    }
                }
                VendedPrefix::Bucket { .. } | VendedPrefix::Path { .. } => add(c.prefix.clone()),
            }
        }
        Ok(out)
    }

    /// The source of the scope at `location`.
    pub(crate) fn source(&self, location: &str) -> Result<VendedSource, KrishivError> {
        let url = parse_object_url(location)
            .map_err(|_| self.refetch.refused(LocationRule::NoVendedCredential))?;
        Ok(VendedSource {
            table: self.clone(),
            url,
            served: Mutex::new(None),
        })
    }
}

/// The `CredentialSource` of one vended scope of one query (module docs).
#[derive(Debug)]
pub(crate) struct VendedSource {
    table: VendedTable,
    url: ParsedUrl,
    /// The answer this source last served a credential from.
    served: Mutex<Option<Arc<VendedAnswer>>>,
}

impl VendedSource {
    /// The credential now, fetching a new answer when the cached one gives none, one about to expire, or is the one
    /// this source served before (the store asks again only when it refused it or it is about to expire).
    pub(crate) async fn current(&self) -> Result<ScopedCredential, KrishivError> {
        let t = &self.table;
        let held = t.cache.get(&t.key);
        let served_before = {
            let served = self.served.lock().unwrap_or_else(PoisonError::into_inner);
            matches!((&*served, &held), (Some(s), Some(h)) if Arc::ptr_eq(s, h))
        };
        if let Some(a) = &held
            && !served_before
            && let Some(c) = t.pick(a, &self.url)?
            && c.expires_at()
                .is_none_or(|e| e > SystemTime::now() + t.skew)
        {
            self.serve(a);
            return Ok(c);
        }
        let fresh = t.refreshed(held.as_deref()).await?;
        let c = t
            .pick(&fresh, &self.url)?
            .ok_or_else(|| t.refetch.refused(LocationRule::NoVendedCredential))?;
        self.serve(&fresh);
        Ok(c)
    }

    fn serve(&self, answer: &Arc<VendedAnswer>) {
        *self.served.lock().unwrap_or_else(PoisonError::into_inner) = Some(Arc::clone(answer));
    }
}

#[async_trait]
impl CredentialSource for VendedSource {
    async fn fetch(&self, _scope: &CredentialScope) -> krishiv_store::Result<ScopedCredential> {
        self.current().await
    }
}

/// The block-cache scope of a vended read (§4.4): every part length-prefixed, an absent part `-`.
pub(crate) fn cache_scope(identity: &IdentityKey, delegated: bool) -> CacheScope {
    let mut parts = vec![Some(identity.catalog.as_str())];
    if delegated {
        parts.push(identity.principal_id.as_deref());
        parts.push(identity.tenant.as_deref());
    }
    let mut out = String::from("vended:");
    for p in parts {
        match p {
            None => out.push('-'),
            Some(p) => out.push_str(&format!("{}:{p}", p.len())),
        }
    }
    CacheScope::Principal(out)
}

/// The locator of one vended table read: only locations under a registered scope, else refused (module docs).
#[derive(Debug)]
pub(crate) struct VendedPaths {
    pub(crate) inner: Arc<dyn PathLocator>,
    pub(crate) prefixes: Vec<VendedPrefix>,
    pub(crate) catalog: String,
    pub(crate) relation: String,
}

impl VendedPaths {
    fn check(&self, path: &str) -> Result<(), KrishivError> {
        let covered = parse_object_url(path)
            .ok()
            .is_some_and(|url| self.prefixes.iter().any(|p| p.covers(&url)));
        if covered {
            Ok(())
        } else {
            Err(refused(
                &self.catalog,
                &self.relation,
                LocationRule::OutsideVendedPrefixes,
            ))
        }
    }
}

impl PathLocator for VendedPaths {
    fn locate(&self, path: &str) -> Result<LocatedPath, KrishivError> {
        self.check(path)?;
        self.inner.locate(path)
    }

    fn locate_with_options(
        &self,
        path: &str,
        options: &BTreeMap<String, krishiv_common::secret::OptionValue>,
    ) -> Result<LocatedPath, KrishivError> {
        self.check(path)?;
        self.inner.locate_with_options(path, options)
    }

    fn qualified_display(&self, path: &str) -> Result<String, KrishivError> {
        self.check(path)?;
        self.inner.qualified_display(path)
    }
}

#[cfg(test)]
#[path = "vended_tests.rs"]
mod tests;
