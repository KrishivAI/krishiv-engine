// SPDX-License-Identifier: Apache-2.0
//! The catalog registry: the declared catalogs of
//! one engine runtime, each built by the [`CatalogFactory`] of its type on first use and then
//! shared by every session of the process.
//!
//! - **One code path.** A [`CatalogSetup`] carries the declaration keys, the mode and the
//!   factories this build compiles in; [`crate::EngineRuntimeBuilder::catalogs`] hands it to the
//!   runtime, which parses it with [`CatalogDeclarations::parse`] (the egress rule checking every
//!   declared literal endpoint), resolves every secret reference and asks each factory to
//!   validate its declarations. Any failure stops the build: a server does not start. Nothing
//!   connects: a catalog that is down does not stop the server.
//! - **Factories.** The session crate has no connector dependency: the server and the facade
//!   register the factories their cargo features compile in. A type without one is refused at
//!   startup, naming the type and the catalog, never skipped.
//! - **Lazy, shared, never poisoned.** A catalog is built on its first use; a build that fails
//!   fails that query and is tried again by the next one.
//! - **Per tenant.** A catalog that lists tenants does not exist for a principal of any
//!   other tenant, or of none: listings omit it and a name lookup fails as for an undeclared name.
//! - **Callers.** A built catalog answers one [`CatalogCaller`] at a time: the caller's
//!   [`IdentityKey`] and, for a `delegated` catalog, the session's latest user token
//!   ([`crate::Session::set_inbound_credential`]). Each query's view of the declared catalogs
//!   carries its session's caller, so the analyzer's `Catalog` trait is unchanged. A catalog of
//!   another mode ignores the caller: every principal reaches it as the server.
//! - **Server mode** (decisions 1 and 7): the declaration keys are startup only. A session's
//!   `SET`/`RESET` of one, or a session builder's, is `CANNOT_MODIFY_STATIC_CONFIG` for every
//!   role (`conf.rs`, `runtime.rs`). Embedded sessions are unchanged.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use krishiv_common::config::{
    CATALOG_DELEGATION_MAX_DISTINCT_EXCHANGES_PER_MINUTE,
    CATALOG_DELEGATION_MAX_EXCHANGES_PER_MINUTE, CATALOG_MAX_METADATA_BYTES,
    CATALOG_REQUEST_TIMEOUT, CATALOG_VENDED_DEFAULT_TTL, Config, ConfigValue,
};
use krishiv_common::error::KrishivError;
use krishiv_common::principal::Principal;
use krishiv_exec::files::PathLocator;
use krishiv_exec::planner::TableProvider;
use krishiv_io::IoContext;
use krishiv_io::egress::EgressPolicy;
use krishiv_plan::TableVersion;
use krishiv_plan::analyzer::{BoxFuture, Catalog, NameForm, Securable, TableIdent, TableMeta};

use crate::auth::{InboundCredential, Method};
use crate::catalog_decl::{
    CATALOG_PREFIX, CatalogAuth, CatalogDecl, CatalogDeclarations, CatalogStorage, CatalogType,
    DeclMode, Delegation,
};

/// Who a declared catalog answers: for a `delegated` catalog the catalog,
/// the principal id ([`Principal::id`]: kind, tenant, name, groups), its tenant and the service
/// acting for it; for every other mode the catalog alone (every caller is the server). No token,
/// token hash or scope is part of it: a refreshed token of the same identity keeps its caches.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub struct IdentityKey {
    /// The catalog name.
    pub catalog: String,
    /// [`Principal::id`]; `None` for the server identity.
    pub principal_id: Option<String>,
    /// The principal's tenant.
    pub tenant: Option<String>,
    /// `Principal::delegated_by`: the service acting for the principal.
    pub delegated_by: Option<String>,
    /// The principal without its groups (kind, tenant, name, `delegated_by`): what per-principal
    /// limits count by, so a caller cannot get a fresh budget by varying its groups.
    pub limit_key: Option<String>,
}

impl IdentityKey {
    /// The server's own identity at `catalog`.
    #[must_use]
    pub fn server(catalog: impl Into<String>) -> Self {
        Self {
            catalog: catalog.into(),
            principal_id: None,
            tenant: None,
            delegated_by: None,
            limit_key: None,
        }
    }

    /// `principal` at `catalog`.
    #[must_use]
    pub fn of(catalog: impl Into<String>, principal: &Principal) -> Self {
        let mut bare = principal.clone();
        bare.groups.clear();
        let by = principal
            .delegated_by
            .as_deref()
            .map_or_else(|| "-".to_owned(), |d| format!("{}:{d}", d.len()));
        Self {
            catalog: catalog.into(),
            principal_id: Some(principal.id()),
            tenant: principal.tenant.clone(),
            delegated_by: principal.delegated_by.clone(),
            limit_key: Some(format!("{} by {by}", bare.id())),
        }
    }
}

/// One call's caller of a declared catalog: its identity key and, for a `delegated`
/// catalog, the session's latest user token and how the session authenticated. `Debug` never
/// shows the token.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CatalogCaller {
    /// The identity key.
    pub identity: IdentityKey,
    /// The user token (OIDC or a signed assertion), when the session has one.
    pub credential: Option<InboundCredential>,
    /// How the session's latest RPC authenticated; `None` for an embedded session.
    pub method: Option<Method>,
}

impl CatalogCaller {
    /// The server itself at `catalog` (no user token).
    #[must_use]
    pub fn server(catalog: impl Into<String>) -> Self {
        Self {
            identity: IdentityKey::server(catalog),
            credential: None,
            method: None,
        }
    }

    /// A user caller.
    #[must_use]
    pub fn new(
        identity: IdentityKey,
        method: Option<Method>,
        credential: Option<InboundCredential>,
    ) -> Self {
        Self {
            identity,
            credential,
            method,
        }
    }

    /// How error messages name the session's authentication.
    ///
    /// How error messages name the session's authentication: the credential's kind when it has
    /// one, else the method (`token`, `mtls`, `none`).
    #[must_use]
    pub fn authenticated_with(&self) -> &'static str {
        match (&self.credential, self.method) {
            (Some(c), _) => c.kind().as_str(),
            // Plain act-as metadata carries the service's method, not the end user's.
            (None, Some(_)) if self.identity.delegated_by.is_some() => {
                "act-as metadata (no signed assertion)"
            }
            (None, Some(m)) => m.as_str(),
            (None, None) => "no user token (an embedded session)",
        }
    }
}

/// What a declared catalog's table is read with in one query: the query's I/O, its path
/// placement (the session's stores; in server mode confined to `krishiv.io.local.allowedRoots`
/// and `krishiv.io.objectStore.allowedPrefixes`), its configuration and the
/// worker slots splits are cut for.
#[derive(Clone)]
#[non_exhaustive]
pub struct ScanInputs {
    /// The query's I/O.
    pub io: IoContext,
    /// Where the table's files live.
    pub paths: Arc<dyn PathLocator>,
    /// The query's configuration.
    pub config: Arc<Config>,
    /// The drivers splits are planned for.
    pub worker_slots: u64,
    /// What a table read with credentials of its own builds its stores from (vended).
    pub stores: ScopedStores,
}

/// What a table read with credentials of its own (vended) builds its stores from: the query's
/// configuration and the runtime's I/O scheduler. It holds no credential.
#[derive(Clone)]
pub struct ScopedStores {
    config: Arc<Config>,
    sched: Arc<krishiv_io::IoScheduler>,
    cwd: String,
}

impl fmt::Debug for ScopedStores {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScopedStores").finish_non_exhaustive()
    }
}

impl ScopedStores {
    /// The stores of one query: `config` its snapshot, `sched` the runtime's scheduler, `cwd` where relative local
    /// paths resolve.
    #[must_use]
    pub fn new(
        config: Arc<Config>,
        sched: Arc<krishiv_io::IoScheduler>,
        cwd: impl Into<String>,
    ) -> Self {
        Self {
            config,
            sched,
            cwd: cwd.into(),
        }
    }

    /// A path locator whose stores under each scope of `sources` sign with that scope's source, registered under
    /// `cache_scope`, and connect through the runtime's (egress policy) HTTP clients. A location outside every scope
    /// would use the query's static store configuration, so the caller refuses those before it locates them. The
    /// scoped stores are unregistered when the locator is dropped.
    #[must_use]
    pub fn paths(
        &self,
        cache_scope: krishiv_io::CacheScope,
        sources: Vec<(
            krishiv_io::CredentialScope,
            Arc<dyn krishiv_io::CredentialSource>,
        )>,
    ) -> Arc<dyn PathLocator> {
        let mut registry = krishiv_io::query_registry(Arc::clone(&self.config), &self.sched)
            .with_cache_scope(cache_scope);
        for (scope, source) in sources {
            registry = registry.with_source(scope, source);
        }
        Arc::new(krishiv_exec::files::ObjectPaths::new(
            krishiv_io::StoreId::new(crate::session::LOCAL_STORE),
            self.cwd.clone(),
            Arc::clone(&self.sched),
            Arc::new(registry),
        ))
    }
}

impl fmt::Debug for ScanInputs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScanInputs")
            .field("paths", &self.paths)
            .field("worker_slots", &self.worker_slots)
            .finish_non_exhaustive()
    }
}

impl ScanInputs {
    /// The inputs of one query.
    #[must_use]
    pub fn new(
        io: IoContext,
        paths: Arc<dyn PathLocator>,
        config: Arc<Config>,
        worker_slots: u64,
        stores: ScopedStores,
    ) -> Self {
        Self {
            io,
            paths,
            config,
            worker_slots,
            stores,
        }
    }
}

/// A table a declared catalog loaded for one caller at one version: what a scan of it opens.
/// Opened before planning, on the executor's I/O runtime.
pub trait DeclaredTable: fmt::Debug + Send + Sync {
    /// The table's provider, read with `inputs`.
    ///
    /// # Errors
    /// A refusal of the read (`KRISHIV_UNSUPPORTED_FEATURE`, a location outside the confinement)
    /// or the failure to open it.
    fn open(
        self: Arc<Self>,
        inputs: ScanInputs,
    ) -> BoxFuture<'static, Result<Arc<dyn TableProvider>, KrishivError>>;
}

/// What [`CallerCatalog::load_declared`] found: the analyzer's metadata and, when the catalog
/// can read the table, the [`DeclaredTable`] its scans open.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LoadedDeclared {
    /// The table's metadata.
    pub meta: TableMeta,
    /// What a scan opens; `None` for a catalog that resolves names only.
    pub table: Option<Arc<dyn DeclaredTable>>,
}

impl LoadedDeclared {
    /// `meta`, read through `table`.
    #[must_use]
    pub fn new(meta: TableMeta, table: Option<Arc<dyn DeclaredTable>>) -> Self {
        Self { meta, table }
    }
}

/// A built declared catalog: the analyzer's `Catalog` with the caller of each call (§3.2).
pub trait CallerCatalog: fmt::Debug + Send + Sync {
    /// As `Catalog::epoch`.
    fn epoch(&self) -> u64 {
        0
    }

    /// As `Catalog::load_table`, for `caller`.
    fn load_table<'a>(
        &'a self,
        caller: &'a CatalogCaller,
        id: &'a TableIdent,
        version: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>>;

    /// [`Self::load_table`] with what a scan of the table opens. The default resolves the name
    /// only: a scan of the table is refused.
    fn load_declared<'a>(
        &'a self,
        caller: &'a CatalogCaller,
        id: &'a TableIdent,
        version: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<LoadedDeclared>, KrishivError>> {
        Box::pin(async move {
            Ok(self
                .load_table(caller, id, version, form)
                .await?
                .map(|meta| LoadedDeclared::new(meta, None)))
        })
    }

    /// `REFRESH TABLE` of `id` for `caller`: drops what the catalog keeps of the table (its
    /// metadata cache entries, for every caller) and moves [`Self::epoch`], so plans and frames
    /// that read it are analyzed again. Whether the table exists. The default keeps nothing and
    /// only looks the table up.
    ///
    /// # Errors
    /// The catalog's failure to load the table.
    fn refresh<'a>(
        &'a self,
        caller: &'a CatalogCaller,
        id: &'a TableIdent,
    ) -> BoxFuture<'a, Result<bool, KrishivError>> {
        let form = NameForm {
            case_sensitive: false,
            qualified: true,
            temporary_only: false,
        };
        Box::pin(async move { Ok(self.load_table(caller, id, None, form).await?.is_some()) })
    }

    /// The namespaces directly under `parent` (top level when empty), for `caller`; `None` when
    /// `parent` does not exist. The default lists nothing: `KRISHIV_UNSUPPORTED_FEATURE`.
    ///
    /// # Errors
    /// The catalog's failure; the default's refusal.
    fn list_namespaces<'a>(
        &'a self,
        caller: &'a CatalogCaller,
        parent: &'a [String],
    ) -> BoxFuture<'a, Result<Option<Vec<Vec<String>>>, KrishivError>> {
        let _ = (caller, parent);
        Box::pin(async move { Err(unlisted()) })
    }

    /// The table names in `namespace`, for `caller`; `None` when the namespace does not exist.
    /// The default lists nothing: `KRISHIV_UNSUPPORTED_FEATURE`.
    ///
    /// # Errors
    /// The catalog's failure; the default's refusal.
    fn list_tables<'a>(
        &'a self,
        caller: &'a CatalogCaller,
        namespace: &'a [String],
    ) -> BoxFuture<'a, Result<Option<Vec<String>>, KrishivError>> {
        let _ = (caller, namespace);
        Box::pin(async move { Err(unlisted()) })
    }
}

fn unlisted() -> KrishivError {
    krishiv_common::error::KrishivUnsupportedFeature {
        feature: "listing the namespaces and tables of this catalog".to_owned(),
    }
    .into()
}

/// A catalog that answers every caller as the server: any analyzer `Catalog` as a
/// [`CallerCatalog`] (a factory of a mode without delegation, a test).
pub struct ServerIdentity(pub Arc<dyn Catalog>);

impl fmt::Debug for ServerIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ServerIdentity")
            .field(&self.0.name())
            .finish()
    }
}

impl CallerCatalog for ServerIdentity {
    fn epoch(&self) -> u64 {
        self.0.epoch()
    }

    fn load_table<'a>(
        &'a self,
        _caller: &'a CatalogCaller,
        id: &'a TableIdent,
        version: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        self.0.load_table(id, version, form)
    }
}

/// How a session's latest RPC authenticated, and the user token it carried.
#[derive(Debug, Clone)]
pub(crate) struct Inbound {
    method: Method,
    credential: Option<InboundCredential>,
}

impl Inbound {
    pub(crate) fn new(method: Method, credential: Option<InboundCredential>) -> Self {
        Self { method, credential }
    }
}

/// Builds the catalog of one declared type. Implemented by the connector
/// crates (`krishiv-catalogs`: `iceberg-rest`), registered by the binary that compiles them in.
pub trait CatalogFactory: fmt::Debug + Send + Sync {
    /// The type it builds.
    fn kind(&self) -> CatalogType;

    /// Checks `decl` at startup without connecting.
    ///
    /// Checks `decl` at startup without connecting (what [`Self::build`] would refuse for the
    /// declaration alone: an option this factory does not implement, a URL its client refuses).
    /// The default accepts everything.
    ///
    /// # Errors
    /// The startup refusal, naming the catalog.
    fn validate(&self, decl: &CatalogDecl, deps: &CatalogDeps) -> Result<(), KrishivError> {
        let _ = (decl, deps);
        Ok(())
    }

    /// The catalog `decl` declares, on its first use. It must not connect to validate itself:
    /// the first request fails the query that made it, and the next query builds again.
    ///
    /// # Errors
    /// A secret that no longer resolves, a client that cannot start.
    fn build(
        &self,
        decl: &CatalogDecl,
        deps: &CatalogDeps,
    ) -> Result<Arc<dyn CallerCatalog>, KrishivError>;
}

/// What a factory builds a catalog with: the mode, the runtime's egress policy (a catalog's
/// clients scope it to its declared hosts, `CatalogDecl::declared_hosts`), the admin's egress
/// proxy and the catalog client limits (`krishiv.catalog.requestTimeout`,
/// `krishiv.catalog.maxMetadataBytes`). `Debug` never shows the proxy URL.
#[derive(Clone)]
#[non_exhaustive]
pub struct CatalogDeps {
    /// Server or embedded.
    pub mode: DeclMode,
    /// The runtime's egress policy.
    pub egress: Arc<EgressPolicy>,
    /// `krishiv.io.egress.proxy`, when the admin set one.
    pub proxy: Option<String>,
    /// The longest one catalog call may take.
    pub request_timeout: Duration,
    /// The largest response a catalog call may read.
    pub max_response_bytes: u64,
    /// `krishiv.catalog.delegation.maxExchangesPerMinute`.
    pub max_exchanges_per_minute: usize,
    /// `krishiv.catalog.delegation.maxDistinctExchangesPerMinute`.
    pub max_distinct_exchanges_per_minute: usize,
}

impl fmt::Debug for CatalogDeps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CatalogDeps")
            .field("mode", &self.mode)
            .field("egress_enforced", &self.egress.enforced())
            .field("proxy", &self.proxy.is_some())
            .field("request_timeout", &self.request_timeout)
            .field("max_response_bytes", &self.max_response_bytes)
            .field("max_exchanges_per_minute", &self.max_exchanges_per_minute)
            .field(
                "max_distinct_exchanges_per_minute",
                &self.max_distinct_exchanges_per_minute,
            )
            .finish()
    }
}

impl CatalogDeps {
    /// Dependencies with the default limits and no proxy.
    ///
    /// Dependencies with the default limits (30 s, 256 MiB, 60 and 20 exchanges a minute) and no
    /// proxy.
    #[must_use]
    pub fn new(mode: DeclMode, egress: Arc<EgressPolicy>) -> Self {
        Self {
            mode,
            egress,
            proxy: None,
            request_timeout: Duration::from_secs(30),
            max_response_bytes: 256 << 20,
            max_exchanges_per_minute: 60,
            max_distinct_exchanges_per_minute: 20,
        }
    }

    /// The same with the admin's egress proxy.
    #[must_use]
    pub fn with_proxy(mut self, proxy: Option<String>) -> Self {
        self.proxy = proxy;
        self
    }

    /// The dependencies of a runtime with `config` and `egress`.
    fn of(mode: DeclMode, egress: &Arc<EgressPolicy>, config: &Config) -> Self {
        let mut deps =
            Self::new(mode, Arc::clone(egress)).with_proxy(crate::egress::egress_proxy(config));
        if let Some(t) = config
            .get(&CATALOG_REQUEST_TIMEOUT)
            .and_then(ConfigValue::as_duration)
        {
            deps.request_timeout = t;
        }
        if let Some(n) = config
            .get(&CATALOG_MAX_METADATA_BYTES)
            .and_then(ConfigValue::as_size)
        {
            deps.max_response_bytes = n;
        }
        let count = |entry| {
            config
                .get(entry)
                .and_then(ConfigValue::as_int)
                .and_then(|n| usize::try_from(n).ok())
        };
        if let Some(n) = count(&CATALOG_DELEGATION_MAX_EXCHANGES_PER_MINUTE) {
            deps.max_exchanges_per_minute = n;
        }
        if let Some(n) = count(&CATALOG_DELEGATION_MAX_DISTINCT_EXCHANGES_PER_MINUTE) {
            deps.max_distinct_exchanges_per_minute = n;
        }
        deps
    }
}

/// Refuses a zero `krishiv.catalog.vendedDefaultTtl` for a runtime with a vended catalog: a credential the catalog
/// gives no expiry for would expire as it is fetched, and every refetch again.
fn check_vended_ttl(config: &Config) -> Result<(), KrishivError> {
    let ttl = config
        .get(&CATALOG_VENDED_DEFAULT_TTL)
        .and_then(ConfigValue::as_duration);
    if ttl.is_some_and(|t| t.is_zero()) {
        return Err(krishiv_common::error::KrishivInvalidConfigValue {
            value: "'0s'".to_owned(),
            key: format!("\"{}\"", CATALOG_VENDED_DEFAULT_TTL.key),
            reason: "a catalog with storage = vended needs a positive default credential lifetime"
                .to_owned(),
        }
        .into());
    }
    Ok(())
}

/// The catalogs a runtime is built with: the declaration keys, where they come from and the
/// factories of this build (see the module docs).
#[must_use]
pub struct CatalogSetup {
    mode: DeclMode,
    settings: Vec<(String, String)>,
    factories: Vec<Arc<dyn CatalogFactory>>,
}

impl fmt::Debug for CatalogSetup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Never the settings: a declaration may carry a secret reference or a URI userinfo.
        f.debug_struct("CatalogSetup")
            .field("mode", &self.mode)
            .field("settings", &self.settings.len())
            .field("factories", &self.factories)
            .finish()
    }
}

impl CatalogSetup {
    /// Where the declarations come from.
    pub(crate) fn mode(&self) -> DeclMode {
        self.mode
    }

    /// No declarations yet, for `mode`.
    pub fn new(mode: DeclMode) -> Self {
        Self {
            mode,
            settings: Vec::new(),
            factories: Vec::new(),
        }
    }

    /// One setting; keys that are not catalog declarations are ignored.
    pub fn setting(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.settings.push((key.into(), value.into()));
        self
    }

    /// The factory of one catalog type (a later one of the same type replaces it).
    pub fn factory(mut self, factory: Arc<dyn CatalogFactory>) -> Self {
        self.factories.push(factory);
        self
    }
}

/// One declared catalog: its declaration, its factory and, once used, the built catalog.
struct Declared {
    decl: CatalogDecl,
    factory: Arc<dyn CatalogFactory>,
    deps: Arc<CatalogDeps>,
    built: Mutex<Option<Arc<dyn CallerCatalog>>>,
}

impl fmt::Debug for Declared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Declared")
            .field("decl", &self.decl)
            .field("built", &self.is_built())
            .finish_non_exhaustive()
    }
}

impl Declared {
    fn is_built(&self) -> bool {
        self.built
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
    }

    /// The built catalog, building it now when this is its first use. A failure is not kept:
    /// the next use builds again. Two first uses at once may both build; the first stored wins.
    fn catalog(&self) -> Result<Arc<dyn CallerCatalog>, KrishivError> {
        if let Some(c) = self
            .built
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
        {
            return Ok(Arc::clone(c));
        }
        // Built outside the lock: a slow build holds no other query of the catalog.
        let fresh = self.factory.build(&self.decl, &self.deps)?;
        let mut slot = self.built.lock().unwrap_or_else(PoisonError::into_inner);
        Ok(Arc::clone(slot.get_or_insert(fresh)))
    }
}

/// A declared catalog as the analyzer sees it for one caller: its name at once, the real catalog
/// on its first table load; the tables it loads are kept in the query's [`DeclaredLoads`].
#[derive(Debug)]
struct DeclaredCatalog(Arc<Declared>, CatalogCaller, Arc<DeclaredLoads>);

/// The key of a declared table in one query: the parts a scan of it names (catalog, namespace,
/// table, as the catalog spells them) and the version it was loaded at.
pub(crate) type DeclaredKey = (Vec<String>, Option<TableVersion>);

/// The declared tables one query loaded and opened: the analysis records each table a
/// declared catalog loaded; before planning each scanned one is opened once, so every scan of it
/// in the query reads one provider.
#[derive(Default)]
pub(crate) struct DeclaredLoads {
    /// What the query's scans read with: `None` outside a query (a listing), where nothing opens.
    inputs: Option<QueryInputs>,
    loaded: Mutex<BTreeMap<DeclaredKey, Arc<dyn DeclaredTable>>>,
    opened: Mutex<BTreeMap<DeclaredKey, Arc<dyn TableProvider>>>,
}

/// A query's scan inputs, before the server-mode confinement is applied (it is built on the
/// first open, so a query that reads no declared table never reads the roots).
#[derive(Clone)]
pub(crate) struct QueryInputs {
    pub(crate) inputs: ScanInputs,
    /// Server mode: catalog-supplied locations are confined (ADR 0046 §3a).
    pub(crate) confine: bool,
}

impl fmt::Debug for DeclaredLoads {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeclaredLoads")
            .field(
                "loaded",
                &self
                    .loaded
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .keys()
                    .collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

impl DeclaredLoads {
    pub(crate) fn new(inputs: QueryInputs) -> Self {
        Self {
            inputs: Some(inputs),
            ..Self::default()
        }
    }

    pub(crate) fn inputs(&self) -> Option<&QueryInputs> {
        self.inputs.as_ref()
    }

    fn record(&self, key: DeclaredKey, table: Arc<dyn DeclaredTable>) {
        self.loaded
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(key, table);
    }

    /// The table a scan of `key` reads, when a declared catalog loaded it in this query.
    pub(crate) fn loaded(&self, key: &DeclaredKey) -> Option<Arc<dyn DeclaredTable>> {
        self.loaded
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(key)
            .cloned()
    }

    /// The provider opened for `key`.
    pub(crate) fn opened(&self, key: &DeclaredKey) -> Option<Arc<dyn TableProvider>> {
        self.opened
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(key)
            .cloned()
    }

    pub(crate) fn put_opened(&self, key: DeclaredKey, provider: Arc<dyn TableProvider>) {
        self.opened
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(key, provider);
    }
}

impl Catalog for DeclaredCatalog {
    fn name(&self) -> &str {
        &self.0.decl.name
    }

    fn epoch(&self) -> u64 {
        self.0
            .built
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .map_or(0, |c| c.epoch())
    }

    fn load_table<'a>(
        &'a self,
        id: &'a TableIdent,
        version: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        Box::pin(async move {
            let catalog = self.0.catalog()?;
            let Some(loaded) = catalog.load_declared(&self.1, id, version, form).await? else {
                return Ok(None);
            };
            if let Some(table) = loaded.table {
                self.2
                    .record((loaded.meta.ident.parts(), loaded.meta.version), table);
            }
            Ok(Some(loaded.meta))
        })
    }
}

/// The declared catalogs of one runtime (see the module docs).
pub struct CatalogRegistry {
    mode: DeclMode,
    default: Option<String>,
    warnings: Vec<String>,
    /// By name, sorted.
    catalogs: BTreeMap<String, Arc<Declared>>,
}

impl fmt::Debug for CatalogRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CatalogRegistry")
            .field("mode", &self.mode)
            .field("default", &self.default)
            .field("catalogs", &self.catalogs.values().collect::<Vec<_>>())
            .finish()
    }
}

/// The settings key of `field` of catalog `name`.
fn key_of(name: &str, field: &str) -> String {
    format!("{CATALOG_PREFIX}{name}.{field}")
}

/// A built declared catalog with the caller its calls carry.
pub(crate) type BuiltCatalog = (Arc<dyn CallerCatalog>, CatalogCaller);

impl CatalogRegistry {
    /// No declared catalogs (an embedded runtime built without a [`CatalogSetup`]).
    #[must_use]
    pub fn empty(mode: DeclMode) -> Self {
        Self {
            mode,
            default: None,
            warnings: Vec::new(),
            catalogs: BTreeMap::new(),
        }
    }

    /// The registry of `setup` for a runtime with `config` and `egress` (startup validation,
    /// R14). Warnings (options accepted and ignored) are logged once, never with a value.
    ///
    /// # Errors
    /// Every refusal of [`CatalogDeclarations::parse`] (a type without a factory in this build
    /// is `KRISHIV_UNSUPPORTED_FEATURE` naming the type and the catalog); a secret reference
    /// that does not resolve (`KRISHIV_INVALID_CONFIG_VALUE` naming the key); a factory's own
    /// refusal ([`CatalogFactory::validate`]).
    pub(crate) fn build(
        setup: CatalogSetup,
        config: &Config,
        egress: &Arc<EgressPolicy>,
    ) -> Result<Self, KrishivError> {
        let factories: BTreeMap<CatalogType, Arc<dyn CatalogFactory>> =
            setup.factories.into_iter().map(|f| (f.kind(), f)).collect();
        let compiled: Vec<CatalogType> = factories.keys().copied().collect();
        let declarations = CatalogDeclarations::parse(
            setup.settings.iter().map(|(k, v)| (k.as_str(), v.as_str())),
            setup.mode,
            &compiled,
            Some(egress),
        )?;
        let deps = Arc::new(CatalogDeps::of(setup.mode, egress, config));
        let mut catalogs = BTreeMap::new();
        for decl in declarations.catalogs {
            // `parse` refused every type without a factory.
            let Some(factory) = factories.get(&decl.kind) else {
                return Err(crate::session::internal(format!(
                    "no factory for catalog type {}",
                    decl.kind.name()
                )));
            };
            check_secrets(&decl)?;
            if decl.storage == CatalogStorage::Vended {
                check_vended_ttl(config)?;
            }
            factory.validate(&decl, &deps)?;
            catalogs.insert(
                decl.name.clone(),
                Arc::new(Declared {
                    decl,
                    factory: Arc::clone(factory),
                    deps: Arc::clone(&deps),
                    built: Mutex::new(None),
                }),
            );
        }
        for w in &declarations.warnings {
            tracing::warn!(target: "krishiv::catalog", "{w}");
        }
        for d in catalogs.values() {
            if d.decl.auth.delegation() == Some(Delegation::Forward) {
                // §3.3: an operator sees every catalog that receives users' own tokens.
                tracing::info!(
                    target: "krishiv::catalog",
                    catalog = d.decl.name.as_str(),
                    "catalog forwards users' own tokens (auth = delegated, delegation = forward)"
                );
            }
        }
        Ok(Self {
            mode: setup.mode,
            default: declarations
                .default
                .filter(|d| d != crate::catalog::CATALOG),
            warnings: declarations.warnings,
            catalogs,
        })
    }

    /// Where the declarations came from: in server mode the declaration keys are startup only.
    #[must_use]
    pub fn mode(&self) -> DeclMode {
        self.mode
    }

    /// `krishiv.catalog.default` when it names a declared catalog (`None`: `spark_catalog`).
    #[must_use]
    pub fn default_catalog(&self) -> Option<&str> {
        self.default.as_deref()
    }

    /// The startup warnings (options accepted and ignored), never carrying a value.
    #[must_use]
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Every declaration, sorted by name, whoever sees it (readiness checks, T066).
    pub fn declarations(&self) -> impl Iterator<Item = &CatalogDecl> {
        self.catalogs.values().map(|d| &d.decl)
    }

    /// The declarations a principal of `tenant` sees, sorted by name.
    pub fn visible<'a>(&'a self, tenant: Option<&'a str>) -> impl Iterator<Item = &'a CatalogDecl> {
        self.declarations().filter(move |d| d.visible_to(tenant))
    }

    /// The catalog `name` as a principal of `tenant` sees it, or `None`.
    ///
    /// The catalog `name` (any letter case) as a principal of `tenant` sees it: `None` when it is
    /// not declared or not visible, the same answer either way (hidden means absent).
    /// It answers as the server: a `delegated` catalog fails every call for want of a user token.
    #[must_use]
    pub fn catalog(&self, name: &str, tenant: Option<&str>) -> Option<Arc<dyn Catalog>> {
        self.catalog_for(name, tenant, None, None, &Arc::default())
    }

    /// [`Self::catalog`] for `principal` with its session's latest authentication `inbound`.
    fn catalog_for(
        &self,
        name: &str,
        tenant: Option<&str>,
        principal: Option<&Principal>,
        inbound: Option<&Inbound>,
        loads: &Arc<DeclaredLoads>,
    ) -> Option<Arc<dyn Catalog>> {
        let declared = self.catalogs.get(&name.to_ascii_lowercase())?;
        if !declared.decl.visible_to(tenant) {
            return None;
        }
        let caller = caller_of(&declared.decl, principal, inbound);
        Some(Arc::new(DeclaredCatalog(
            Arc::clone(declared),
            caller,
            Arc::clone(loads),
        )) as Arc<dyn Catalog>)
    }

    /// The built catalog `name` (any letter case) a principal of `tenant` sees, with its caller.
    ///
    /// # Errors
    /// The catalog's build failure.
    fn built_for(
        &self,
        name: &str,
        tenant: Option<&str>,
        principal: Option<&Principal>,
        inbound: Option<&Inbound>,
    ) -> Option<Result<BuiltCatalog, KrishivError>> {
        let declared = self.catalogs.get(&name.to_ascii_lowercase())?;
        if !declared.decl.visible_to(tenant) {
            return None;
        }
        let caller = caller_of(&declared.decl, principal, inbound);
        Some(declared.catalog().map(|c| (c, caller)))
    }
}

/// Who calls `decl`'s catalog (§3.2): only a delegated catalog tells callers apart; for the others
/// the key is the catalog and no user token goes anywhere.
fn caller_of(
    decl: &CatalogDecl,
    principal: Option<&Principal>,
    inbound: Option<&Inbound>,
) -> CatalogCaller {
    match (&decl.auth, principal) {
        (CatalogAuth::Delegated { .. }, Some(p)) => CatalogCaller::new(
            IdentityKey::of(decl.name.clone(), p),
            inbound.map(|i| i.method),
            inbound.and_then(|i| i.credential.clone()),
        ),
        _ => CatalogCaller::server(decl.name.clone()),
    }
}

/// Resolves every secret reference of `decl` once, so a missing variable or an unsafe file stops
/// startup instead of the first query; the values are dropped (the factory resolves again).
fn check_secrets(decl: &CatalogDecl) -> Result<(), KrishivError> {
    match &decl.auth {
        CatalogAuth::None => {}
        CatalogAuth::Token(r) => drop(r.resolve(&key_of(&decl.name, "token"))?),
        CatalogAuth::ClientCredentials { client_secret, .. } => {
            drop(client_secret.resolve(&key_of(&decl.name, "clientSecret"))?);
        }
        CatalogAuth::Delegated { actor, .. } => {
            if let Some(a) = actor {
                drop(
                    a.client_secret
                        .resolve(&key_of(&decl.name, "clientSecret"))?,
                );
            }
        }
    }
    Ok(())
}

/// The declared catalogs as one session's principal sees them: what name resolution, the
/// listings and `system.catalogs` read.
#[derive(Debug, Clone)]
pub(crate) struct DeclaredView {
    registry: Arc<CatalogRegistry>,
    tenant: Option<String>,
    /// The session's principal, for delegated catalogs' identity keys.
    principal: Option<Arc<Principal>>,
    /// The session's latest authentication, snapshotted for this view's query.
    inbound: Option<Inbound>,
    /// The declared tables this view's query loaded.
    loads: Arc<DeclaredLoads>,
}

impl Default for DeclaredView {
    fn default() -> Self {
        Self {
            registry: Arc::new(CatalogRegistry::empty(DeclMode::Embedded)),
            tenant: None,
            principal: None,
            inbound: None,
            loads: Arc::default(),
        }
    }
}

/// One row of `system.catalogs`: no URI, no credential, ever.
pub(crate) struct CatalogRow {
    pub(crate) name: String,
    pub(crate) kind: &'static str,
    pub(crate) auth: &'static str,
    pub(crate) storage: &'static str,
    /// The delegation form of a `delegated` catalog; `None` otherwise.
    pub(crate) delegation: Option<&'static str>,
    pub(crate) default: bool,
}

impl DeclaredView {
    pub(crate) fn new(
        registry: Arc<CatalogRegistry>,
        principal: Arc<Principal>,
        inbound: Option<Inbound>,
    ) -> Self {
        Self {
            registry,
            tenant: principal.tenant.clone(),
            principal: Some(principal),
            inbound,
            loads: Arc::default(),
        }
    }

    /// This view for one query reading with `inputs`.
    pub(crate) fn for_query(mut self, inputs: ScanInputs) -> Self {
        let confine = self.registry.mode() == DeclMode::Server;
        self.loads = Arc::new(DeclaredLoads::new(QueryInputs { inputs, confine }));
        self
    }

    /// The declared tables this view's query loaded.
    pub(crate) fn loads(&self) -> &Arc<DeclaredLoads> {
        &self.loads
    }

    /// The visible catalog `name`, answering this view's caller.
    pub(crate) fn catalog(&self, name: &str) -> Option<Arc<dyn Catalog>> {
        self.registry.catalog_for(
            name,
            self.tenant.as_deref(),
            self.principal.as_deref(),
            self.inbound.as_ref(),
            &self.loads,
        )
    }

    /// The visible catalog `name`, built, with this view's caller (refresh, listings).
    pub(crate) fn built(&self, name: &str) -> Option<Result<BuiltCatalog, KrishivError>> {
        self.registry.built_for(
            name,
            self.tenant.as_deref(),
            self.principal.as_deref(),
            self.inbound.as_ref(),
        )
    }

    /// The canonical name of the visible catalog `name` (any letter case).
    pub(crate) fn canonical(&self, name: &str) -> Option<String> {
        self.registry
            .visible(self.tenant.as_deref())
            .find(|d| d.name.eq_ignore_ascii_case(name))
            .map(|d| d.name.clone())
    }

    /// The visible catalogs' names, sorted.
    pub(crate) fn names(&self) -> Vec<String> {
        self.registry
            .visible(self.tenant.as_deref())
            .map(|d| d.name.clone())
            .collect()
    }

    /// The rows of `system.catalogs`.
    pub(crate) fn rows(&self) -> Vec<CatalogRow> {
        let default = self.registry.default_catalog();
        self.registry
            .visible(self.tenant.as_deref())
            .map(|d| CatalogRow {
                name: d.name.clone(),
                kind: d.kind.name(),
                auth: d.auth.name(),
                storage: d.storage.name(),
                delegation: d.auth.delegation().map(Delegation::name),
                default: default == Some(d.name.as_str()),
            })
            .collect()
    }
}

impl crate::Session {
    /// The declared catalogs as this session's principal sees them.
    pub(crate) fn declared_catalogs(&self) -> DeclaredView {
        // The statement's own request first, else the session's latest RPC. A statement handle
        // that outlives its RPC (a long operation) keeps that token, secret-typed and wiped on
        // drop, until the handle goes: the snapshot is the design's (one credential per query),
        // and once the token expires no new exchange is made with it.
        let inbound = self.request_inbound.clone().or_else(|| {
            self.state
                .inbound
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        });
        DeclaredView::new(
            Arc::clone(&self.state.runtime.catalogs),
            Arc::clone(&self.state.principal),
            inbound,
        )
    }

    /// Records how the session's latest RPC authenticated and the user token it carried (an
    /// OIDC token or a signed assertion; `None` for a static token, a certificate or plain act-as
    /// metadata), for its `delegated` catalogs. The most recent call wins,
    /// as the per-RPC baggage does: a fresher token replaces the stored one (a token of another
    /// principal never reaches this session, whose owner is fixed). Queries take it when they
    /// start.
    pub fn set_inbound_credential(&self, method: Method, credential: Option<InboundCredential>) {
        *self
            .state
            .inbound
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(Inbound { method, credential });
    }

    /// The namespaces directly under `parent` of the declared catalog `catalog` (any letter case)
    /// as the current caller sees them (design P3-T067 §3.5, PR-58b), only those this session's
    /// principal may list (P3-T062 §2.4: the rule of [`Self::declared_namespace_listed`]);
    /// `None` when `catalog` is no declared catalog this principal sees.
    ///
    /// # Errors
    /// The catalog's failure; `SCHEMA_NOT_FOUND` for a `parent` it does not have or the principal
    /// may not list (ADR 0046 §8: no existence oracle).
    pub(crate) async fn declared_namespaces(
        &self,
        catalog: &str,
        parent: &[String],
    ) -> Option<Result<Vec<Vec<String>>, KrishivError>> {
        let built = self.declared_catalogs().built(catalog)?;
        let name = self.declared_catalogs().canonical(catalog)?;
        Some(
            async {
                let (built, caller) = built?;
                if !parent.is_empty() && !self.declared_namespace_listed(&name, parent).await {
                    return Err(schema_not_found(catalog, parent));
                }
                let Some(found) = built
                    .list_namespaces(&caller, parent)
                    .await
                    .map_err(|e| names_catalog(e, catalog))?
                else {
                    return Err(schema_not_found(catalog, parent));
                };
                let mut out = Vec::with_capacity(found.len());
                for n in found {
                    if self.declared_namespace_listed(&name, &n).await {
                        out.push(n);
                    }
                }
                Ok(out)
            }
            .await,
        )
    }

    /// The tables of `namespace` of the declared catalog `catalog`, as [`Self::declared_namespaces`]:
    /// only those this session's principal may list ([`Session::may_list`]); `SCHEMA_NOT_FOUND`
    /// for a namespace it may not list.
    pub(crate) async fn declared_tables(
        &self,
        catalog: &str,
        namespace: &[String],
    ) -> Option<Result<Vec<String>, KrishivError>> {
        let built = self.declared_catalogs().built(catalog)?;
        let name = self.declared_catalogs().canonical(catalog)?;
        Some(
            async {
                let (built, caller) = built?;
                // A namespace the grants hide answers as a missing one before the catalog is
                // asked (no existence oracle, no outbound call for it, ADR 0046 §8). A table the
                // grants list implies its namespace is listed, so nothing listable is lost.
                if self.state.grants.is_some()
                    && !self.declared_namespace_listed(&name, namespace).await
                {
                    return Err(schema_not_found(catalog, namespace));
                }
                let Some(found) = built
                    .list_tables(&caller, namespace)
                    .await
                    .map_err(|e| names_catalog(e, catalog))?
                else {
                    return Err(schema_not_found(catalog, namespace));
                };
                Ok(self.listed_declared_tables(&name, namespace, found).await)
            }
            .await,
        )
    }

    /// The tables among `names` of `namespace` of the declared catalog `catalog` (its canonical
    /// name) that this session's principal may list.
    async fn listed_declared_tables(
        &self,
        catalog: &str,
        namespace: &[String],
        names: Vec<String>,
    ) -> Vec<String> {
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            let securable = Securable::Table(TableIdent {
                catalog: catalog.to_owned(),
                namespace: namespace.to_vec(),
                name: name.clone(),
            });
            if self.may_list(&securable).await {
                out.push(name);
            }
        }
        out
    }

    /// Whether this session's principal may see the namespace `namespace` of the declared catalog
    /// `catalog` (its canonical name) in a listing, by the session catalog's rule
    /// ([`Session::may_list_namespace`]): it may list the namespace, or, under the `grants`
    /// provider, it owns or holds a grant on something under it at any depth (a nested
    /// namespace, a table, a column; `GrantsPolicy::namespace_listable`). Decided from the grant
    /// state alone: no catalog call, so one namespace the catalog refuses cannot fail a listing.
    async fn declared_namespace_listed(&self, catalog: &str, namespace: &[String]) -> bool {
        let securable = declared_namespace(catalog, namespace);
        if let Some(listed) = self
            .state
            .grants
            .as_ref()
            .and_then(|g| g.namespace_listable(&self.state.principal, &securable))
        {
            return listed;
        }
        self.may_list(&securable).await
    }

    /// Every catalog this session sees, sorted: `spark_catalog`, `system` and the visible
    /// declared ones (`SHOW CATALOGS`, `spark.catalog.listCatalogs`).
    pub(crate) fn catalog_names(&self) -> Vec<String> {
        let mut names = vec![
            crate::catalog::CATALOG.to_owned(),
            crate::system::SYSTEM.to_owned(),
        ];
        names.extend(self.declared_catalogs().names());
        names.sort();
        names
    }

    /// `SHOW CURRENT NAMESPACE`'s row: the current catalog and namespace; `system` and a
    /// declared catalog show no namespace (a declared catalog's namespaces are its own, PR-58b).
    pub(crate) fn current_namespace_row(&self) -> Vec<Option<String>> {
        let catalog = self.state.catalog.current_catalog();
        if catalog == crate::catalog::CATALOG {
            vec![Some(catalog), Some(self.state.catalog.current_namespace())]
        } else {
            vec![Some(catalog), Some(String::new())]
        }
    }
}

/// `SCHEMA_NOT_FOUND` for `namespace` of the declared catalog `catalog`.
pub(crate) fn schema_not_found(catalog: &str, namespace: &[String]) -> KrishivError {
    let quote = |p: &str| format!("`{}`", p.replace('`', "``"));
    krishiv_common::error::SchemaNotFound {
        schema_name: std::iter::once(catalog)
            .chain(namespace.iter().map(String::as_str))
            .map(quote)
            .collect::<Vec<_>>()
            .join("."),
    }
    .into()
}

/// `e` naming `catalog` when it is the refusal of a catalog that lists nothing.
fn names_catalog(e: KrishivError, catalog: &str) -> KrishivError {
    if e.error_class().name() == "KRISHIV_UNSUPPORTED_FEATURE" {
        return krishiv_common::error::KrishivUnsupportedFeature {
            feature: format!(
                "listing the namespaces and tables of the declared catalog `{}`",
                catalog.replace('`', "``")
            ),
        }
        .into();
    }
    e
}

/// The securable of the namespace `namespace` of the catalog `catalog`.
fn declared_namespace(catalog: &str, namespace: &[String]) -> Securable {
    let mut parts = Vec::with_capacity(namespace.len() + 1);
    parts.push(catalog.to_owned());
    parts.extend_from_slice(namespace);
    Securable::Namespace(parts)
}
