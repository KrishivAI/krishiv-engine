// SPDX-License-Identifier: Apache-2.0
//! The catalog registry (design P3-T067 §2.3, §2.4; ADR 0036 R11, R14): the declared catalogs of
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
//! - **Per tenant (§2.4).** A catalog that lists tenants does not exist for a principal of any
//!   other tenant, or of none: listings omit it and a name lookup fails as for an undeclared name.
//! - **Server mode** (decisions 1 and 7): the declaration keys are startup only. A session's
//!   `SET`/`RESET` of one, or a session builder's, is `CANNOT_MODIFY_STATIC_CONFIG` for every
//!   role (`conf.rs`, `runtime.rs`). Embedded sessions are unchanged.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use krishiv_common::config::{
    CATALOG_MAX_METADATA_BYTES, CATALOG_REQUEST_TIMEOUT, Config, ConfigValue,
};
use krishiv_common::error::KrishivError;
use krishiv_io::egress::EgressPolicy;
use krishiv_plan::TableVersion;
use krishiv_plan::analyzer::{BoxFuture, Catalog, NameForm, TableIdent, TableMeta};

use crate::catalog_decl::{
    CATALOG_PREFIX, CatalogAuth, CatalogDecl, CatalogDeclarations, CatalogType, DeclMode,
};

/// Builds the catalog of one declared type (design P3-T067 §2.3). Implemented by the connector
/// crates (`krishiv-catalogs`: `iceberg-rest`), registered by the binary that compiles them in.
pub trait CatalogFactory: fmt::Debug + Send + Sync {
    /// The type it builds.
    fn kind(&self) -> CatalogType;

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
    ) -> Result<Arc<dyn Catalog>, KrishivError>;
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
}

impl fmt::Debug for CatalogDeps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CatalogDeps")
            .field("mode", &self.mode)
            .field("egress_enforced", &self.egress.enforced())
            .field("proxy", &self.proxy.is_some())
            .field("request_timeout", &self.request_timeout)
            .field("max_response_bytes", &self.max_response_bytes)
            .finish()
    }
}

impl CatalogDeps {
    /// Dependencies with the default limits (30 s, 256 MiB) and no proxy.
    #[must_use]
    pub fn new(mode: DeclMode, egress: Arc<EgressPolicy>) -> Self {
        Self {
            mode,
            egress,
            proxy: None,
            request_timeout: Duration::from_secs(30),
            max_response_bytes: 256 << 20,
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
        deps
    }
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
    built: Mutex<Option<Arc<dyn Catalog>>>,
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
    fn catalog(&self) -> Result<Arc<dyn Catalog>, KrishivError> {
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

/// A declared catalog as the analyzer sees it: its name at once, the real catalog on its first
/// table load.
#[derive(Debug)]
struct DeclaredCatalog(Arc<Declared>);

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
            catalog.load_table(id, version, form).await
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

    /// The declarations a principal of `tenant` sees, sorted by name (§2.4).
    pub fn visible<'a>(&'a self, tenant: Option<&'a str>) -> impl Iterator<Item = &'a CatalogDecl> {
        self.declarations().filter(move |d| d.visible_to(tenant))
    }

    /// The catalog `name` (any letter case) as a principal of `tenant` sees it: `None` when it is
    /// not declared or not visible, the same answer either way (hidden means absent).
    #[must_use]
    pub fn catalog(&self, name: &str, tenant: Option<&str>) -> Option<Arc<dyn Catalog>> {
        let declared = self.catalogs.get(&name.to_ascii_lowercase())?;
        declared
            .decl
            .visible_to(tenant)
            .then(|| Arc::new(DeclaredCatalog(Arc::clone(declared))) as Arc<dyn Catalog>)
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
    }
    Ok(())
}

/// The declared catalogs as one session's principal sees them: what name resolution, the
/// listings and `system.catalogs` read.
#[derive(Debug, Clone)]
pub(crate) struct DeclaredView {
    registry: Arc<CatalogRegistry>,
    tenant: Option<String>,
}

impl Default for DeclaredView {
    fn default() -> Self {
        Self {
            registry: Arc::new(CatalogRegistry::empty(DeclMode::Embedded)),
            tenant: None,
        }
    }
}

/// One row of `system.catalogs`: no URI, no credential, ever.
pub(crate) struct CatalogRow {
    pub(crate) name: String,
    pub(crate) kind: &'static str,
    pub(crate) auth: &'static str,
    pub(crate) storage: &'static str,
    /// The delegation form of a `delegated` catalog (PR-58b); `None` otherwise.
    pub(crate) delegation: Option<&'static str>,
    pub(crate) default: bool,
}

impl DeclaredView {
    pub(crate) fn new(registry: Arc<CatalogRegistry>, tenant: Option<String>) -> Self {
        Self { registry, tenant }
    }

    /// The visible catalog `name`.
    pub(crate) fn catalog(&self, name: &str) -> Option<Arc<dyn Catalog>> {
        self.registry.catalog(name, self.tenant.as_deref())
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
                delegation: None,
                default: default == Some(d.name.as_str()),
            })
            .collect()
    }
}

impl crate::Session {
    /// The declared catalogs as this session's principal sees them.
    pub(crate) fn declared_catalogs(&self) -> DeclaredView {
        DeclaredView::new(
            Arc::clone(&self.state.runtime.catalogs),
            self.state.principal.tenant.clone(),
        )
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
