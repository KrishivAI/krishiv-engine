// SPDX-License-Identifier: Apache-2.0
//! A declared Iceberg REST catalog as the analyzer sees it (design P3-T067 PR-58b): `loadTable`
//! mapped to the analyzer's metadata, and the table's scans opened through the Iceberg connector.
//!
//! - **Metadata.** The `metadata` document of the `loadTable` response is the table's (as Java's
//!   REST catalog takes it): its schema becomes the columns (Iceberg `doc` as the column comment),
//!   the snapshot read becomes the version (the current snapshot, or the one a time travel names).
//!   Read restrictions and server-side planning are refused by the client before anything is
//!   returned.
//! - **Table `config`.** No key of it reaches a store: the reads use the server's store
//!   configuration, or (`storage = vended`) only the vended credential keys. A catalog-supplied
//!   `token` is kept only with a vended answer, as the bearer of that table's `loadCredentials` and
//!   reload (`vended.rs`); otherwise it is dropped (and the RFC 8693 token-type keys beside it
//!   always are); the refresh-endpoint and `signer.*` keys are dropped and never followed. A response asking for remote signing
//!   (`s3.remote-signing-enabled`, or `signer.*` keys without credentials) is refused when the
//!   table is read (`KRISHIV_UNSUPPORTED_FEATURE`, §4.5).
//! - **Metadata cache (§3.5).** A `loadTable` answer is kept [`METADATA_TTL`] per table and
//!   identity key (the caller's on a `delegated` catalog, the catalog's otherwise), so a revoked
//!   grant is seen within the TTL; `REFRESH TABLE` drops the table's entries for every identity
//!   and moves the catalog's epoch.
//! - **Reads.** `storage = allowed-prefixes`: the query's I/O and path placement (confined by the
//!   session in server mode). `storage = vended` is refused when the table is read (unit 3); no
//!   ambient fallback is tried.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use krishiv_common::error::{
    KrishivCatalogTableUnsupported, KrishivError, KrishivUnsupportedFeature,
};
use krishiv_connectors_iceberg::plan::VersionSpec;
use krishiv_connectors_iceberg::{
    IcebergMetadata, IcebergTable, IcebergTableProvider, StorageRegistration,
};
use krishiv_exec::planner::TableProvider;
use krishiv_plan::TableVersion;
use krishiv_plan::analyzer::{
    BoxFuture, FileMetadata, NameForm, TableColumn, TableIdent, TableKind, TableMeta,
};
use krishiv_session::catalog_decl::{CatalogAuth, CatalogDecl, CatalogStorage};
use krishiv_session::catalog_registry::{
    CallerCatalog, CatalogCaller, DeclaredTable, IdentityKey, LoadedDeclared, ScanInputs,
};

use super::vended::{Refetch, VendedAnswer, VendedCache, VendedPaths, VendedTable, cache_scope};
use super::{quoted, rest_caller};
use crate::iceberg_rest::vended::VendedPrefix;
use crate::iceberg_rest::{LoadedTable, RestClient, RestError, RestLimits};
use crate::location::{LocationPolicy, LocationRule, RemoteLocationPolicy, refused};

/// Iceberg's metadata table names (`MetadataTableType` at apache-iceberg-1.12.0), lower case.
const METADATA_TABLES: [&str; 16] = [
    "entries",
    "files",
    "data_files",
    "delete_files",
    "history",
    "metadata_log_entries",
    "snapshots",
    "refs",
    "manifests",
    "partitions",
    "all_data_files",
    "all_delete_files",
    "all_files",
    "all_manifests",
    "all_entries",
    "position_deletes",
];

/// How long a `loadTable` answer is kept (design P3-T067 §3.5, LH §2.3 `ttl(30s)`).
pub(crate) const METADATA_TTL: Duration = Duration::from_secs(30);

/// Most `loadTable` answers one catalog keeps; past it the oldest goes.
const MAX_ENTRIES: usize = 4096;

/// The table `config` keys that ask for remote signing on their own.
const SIGNING: &str = "s3.remote-signing-enabled";

fn unsupported(feature: String) -> KrishivError {
    KrishivUnsupportedFeature { feature }.into()
}

/// What Krishiv keeps of one `loadTable` answer: the parsed metadata and what a read refuses.
#[derive(Debug)]
struct Loaded {
    metadata: IcebergMetadata,
    metadata_location: String,
    /// The answer asks for remote signing (§4.5).
    signing: bool,
    /// The answer's vended credentials (`storage = vended` only; module docs of `vended.rs`).
    vended: Option<Arc<VendedAnswer>>,
}

/// A `loadTable` cache key: the identity, the namespace and the table as asked.
type Key = (IdentityKey, Vec<String>, String);

/// A declared Iceberg REST catalog (module docs).
#[derive(Debug)]
pub(crate) struct RestCatalog {
    name: String,
    client: Arc<RestClient>,
    storage: CatalogStorage,
    delegated: bool,
    ttl: Duration,
    epoch: AtomicU64,
    cache: Mutex<HashMap<Key, (Instant, Arc<Loaded>)>>,
    /// The vended answers (one cache per process, §4.3), keyed with `instance`.
    vended: Arc<VendedCache>,
    /// This catalog's own part of every vended key: unique in the process.
    instance: u64,
}

impl Drop for RestCatalog {
    fn drop(&mut self) {
        self.vended.drop_instance(self.instance);
    }
}

impl RestCatalog {
    pub(crate) fn new(decl: &CatalogDecl, client: Arc<RestClient>, ttl: Duration) -> Self {
        Self {
            name: decl.name.clone(),
            client,
            storage: decl.storage,
            delegated: matches!(decl.auth, CatalogAuth::Delegated { .. }),
            ttl,
            epoch: AtomicU64::new(0),
            cache: Mutex::new(HashMap::new()),
            vended: VendedCache::global(),
            instance: {
                static NEXT: AtomicU64 = AtomicU64::new(0);
                NEXT.fetch_add(1, Ordering::Relaxed)
            },
        }
    }

    /// The cache key of `caller`'s load of `namespace.name`: per caller on a delegated
    /// catalog, per catalog otherwise (§3.5).
    fn key(&self, caller: &CatalogCaller, namespace: &[String], name: &str) -> Key {
        let identity = if self.delegated {
            caller.identity.clone()
        } else {
            IdentityKey::server(self.name.clone())
        };
        (identity, namespace.to_vec(), name.to_owned())
    }

    /// Design §4.1: an Iceberg metadata table of a declared table (`cat.ns.t.files`) is refused,
    /// as its bounds would show values a masked column hides and reading it reads manifests. The
    /// name is refused when its last part is a metadata table name (Iceberg 1.12.0
    /// `MetadataTableType`, any letter case) and its prefix is a table this caller can load;
    /// otherwise it is an ordinary table name in a nested namespace. The prefix is asked first, so
    /// this is more conservative than Iceberg's own order (the full name, then the metadata
    /// table): a real table `files` in namespace `db.t` beside a table `db.t` is refused, and a
    /// caller who may not load the prefix gets the prefix's error.
    async fn refuse_metadata_table(
        &self,
        caller: &CatalogCaller,
        id: &TableIdent,
    ) -> Result<(), KrishivError> {
        let [parent @ .., table] = id.namespace.as_slice() else {
            return Ok(());
        };
        if parent.is_empty()
            || !METADATA_TABLES
                .iter()
                .any(|m| m.eq_ignore_ascii_case(&id.name))
            || self.loaded(caller, parent, table).await?.is_none()
        {
            return Ok(());
        }
        let relation = id
            .namespace
            .iter()
            .map(|p| quoted(p))
            .collect::<Vec<_>>()
            .join(".");
        Err(KrishivCatalogTableUnsupported {
            catalog: quoted(&self.name),
            relation_name: relation,
            feature: format!("the Iceberg metadata table {}", quoted(&id.name)),
        }
        .into())
    }

    /// Runs the blocking client call `call` off the compute workers (on the blocking pool inside
    /// a tokio runtime, else on the caller's thread, which is blocking already).
    async fn blocking<T: Send + 'static>(
        &self,
        call: impl FnOnce(&RestClient) -> Result<T, KrishivError> + Send + 'static,
    ) -> Result<T, KrishivError> {
        let client = Arc::clone(&self.client);
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle
                    .spawn_blocking(move || call(&client))
                    .await
                    .map_err(|_| {
                        RestError::Transport("the catalog call did not finish".to_owned())
                            .into_error(self.client.name(), "the table", &RestLimits::default())
                    })?
            }
            Err(_) => call(&client),
        }
    }

    /// `caller`'s answer for `namespace.name`: cached while fresh, else loaded and kept.
    async fn loaded(
        &self,
        caller: &CatalogCaller,
        namespace: &[String],
        name: &str,
    ) -> Result<Option<Arc<Loaded>>, KrishivError> {
        let key = self.key(caller, namespace, name);
        if let Some((at, hit)) = self
            .cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&key)
            && at.elapsed() < self.ttl
        {
            return Ok(Some(Arc::clone(hit)));
        }
        let who = rest_caller(caller);
        let (ns, table) = (namespace.to_vec(), name.to_owned());
        // Read before the request: a `REFRESH TABLE` after it keeps this answer out of the vended cache.
        let drops = self.vended.drops();
        let Some(found) = self
            .blocking(move |c| c.load_table_as(&who, &ns, &table))
            .await?
        else {
            return Ok(None);
        };
        let loaded = Arc::new(self.keep(namespace, name, found, drops)?);
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        // A `REFRESH TABLE` since the request (any table's, as in the vended cache): this answer serves its own
        // query only, so no later query reuses metadata the refresh discarded (review 2ed68fa67 nit 1). Checked under
        // the cache lock; `refresh` moves the count before its `retain`, so no stale insert survives.
        if self.vended.drops() != drops {
            return Ok(Some(loaded));
        }
        if cache.len() >= MAX_ENTRIES && !cache.contains_key(&key) {
            let oldest = cache
                .iter()
                .min_by_key(|(_, (at, _))| *at)
                .map(|(k, _)| k.clone());
            if let Some(k) = oldest {
                cache.remove(&k);
            }
        }
        cache.insert(key, (Instant::now(), Arc::clone(&loaded)));
        Ok(Some(loaded))
    }

    /// What is kept of `found` (module docs): the parsed metadata, and with `storage = vended` its
    /// vended part ([`VendedAnswer::of_table`]: the credential keys and the table token only), which
    /// then lives in this 30 s table cache beside the `VendedCache`. Every other `config` value is
    /// dropped here, and so is every credential when the storage is not vended.
    fn keep(
        &self,
        namespace: &[String],
        name: &str,
        found: LoadedTable,
        drops: u64,
    ) -> Result<Loaded, KrishivError> {
        let what = self.relation(namespace, name);
        let Some(metadata_location) = found.metadata_location.clone() else {
            return Err(self.fault(&what, "a table without a metadata location (staged)"));
        };
        let signing = asks_for_signing(&found);
        let vended = (self.storage == CatalogStorage::Vended)
            .then(|| Arc::new(VendedAnswer::of_table(&found, drops)));
        let metadata = IcebergMetadata::from_json(found.metadata)
            .map_err(|_| self.fault(&what, "table metadata that does not parse"))?;
        Ok(Loaded {
            metadata,
            metadata_location,
            signing,
            vended,
        })
    }

    fn relation(&self, namespace: &[String], name: &str) -> String {
        std::iter::once(self.name.as_str())
            .chain(namespace.iter().map(String::as_str))
            .chain(std::iter::once(name))
            .map(quoted)
            .collect::<Vec<_>>()
            .join(".")
    }

    /// A catalog fault for `what`: the detail in the log, never in the error.
    fn fault(&self, what: &str, detail: &str) -> KrishivError {
        tracing::warn!(target: "krishiv::catalog", catalog = self.name.as_str(), table = what, detail, "an Iceberg REST catalog returned a table Krishiv cannot read");
        krishiv_common::error::KrishivCatalogFailure {
            catalog: quoted(&self.name),
            relation_name: what.to_owned(),
        }
        .into()
    }
}

/// Whether a `loadTable` answer asks for remote signing (§4.5): the S3 signing flag, or signer
/// keys with no credential to read with.
fn asks_for_signing(t: &LoadedTable) -> bool {
    let flag = t
        .config
        .get(SIGNING)
        .is_some_and(|v| v.expose_secret().trim().eq_ignore_ascii_case("true"));
    let signer = t.config.keys().any(|k| k.starts_with("signer."));
    let credentials =
        !t.storage_credentials.is_empty() || t.config.contains_key("s3.access-key-id");
    flag || (signer && !credentials)
}

impl CallerCatalog for RestCatalog {
    fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::SeqCst)
    }

    fn load_table<'a>(
        &'a self,
        caller: &'a CatalogCaller,
        id: &'a TableIdent,
        version: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        Box::pin(async move {
            Ok(self
                .load_declared(caller, id, version, form)
                .await?
                .map(|l| l.meta))
        })
    }

    fn load_declared<'a>(
        &'a self,
        caller: &'a CatalogCaller,
        id: &'a TableIdent,
        version: Option<TableVersion>,
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<LoadedDeclared>, KrishivError>> {
        Box::pin(async move {
            // No temporary relation lives here, and a table needs a namespace.
            if form.temporary_only || id.namespace.is_empty() {
                return Ok(None);
            }
            self.refuse_metadata_table(caller, id).await?;
            let Some(loaded) = self.loaded(caller, &id.namespace, &id.name).await? else {
                return Ok(None);
            };
            let spec = version.map_or(VersionSpec::Current, |v| VersionSpec::Id(v.0));
            let snapshot = loaded.metadata.snapshot(&spec)?;
            let mut columns = Vec::new();
            let mut column_comments = std::collections::BTreeMap::new();
            for f in snapshot.schema.fields() {
                columns.push(TableColumn::new(
                    f.name(),
                    f.data_type().clone(),
                    f.nullable(),
                ));
                if let Some(c) = f.metadata().comment() {
                    column_comments.insert(f.name().to_owned(), c.to_owned());
                }
            }
            let meta = TableMeta {
                ident: TableIdent {
                    catalog: self.name.clone(),
                    namespace: id.namespace.clone(),
                    name: id.name.clone(),
                },
                columns,
                version: snapshot.snapshot_id.map(TableVersion),
                kind: TableKind::Table,
                file_metadata: FileMetadata::None,
                column_comments,
            };
            let relation = self.relation(&id.namespace, &id.name);
            let vended = (self.storage == CatalogStorage::Vended).then(|| VendedRead {
                cache: Arc::clone(&self.vended),
                key: {
                    let (identity, namespace, name) = self.key(caller, &id.namespace, &id.name);
                    (self.instance, identity, namespace, name)
                },
                identity: caller.identity.clone(),
                delegated: self.delegated,
                refetch: Refetch {
                    client: Arc::clone(&self.client),
                    caller: rest_caller(caller),
                    namespace: id.namespace.clone(),
                    table: id.name.clone(),
                    catalog: self.name.clone(),
                    relation: relation.clone(),
                    location: loaded
                        .vended
                        .as_ref()
                        .map(|a| a.location().to_owned())
                        .unwrap_or_default(),
                },
            });
            let table = RestTable {
                relation,
                vended,
                namespace: id.namespace.clone(),
                name: id.name.clone(),
                loaded,
                spec: snapshot
                    .snapshot_id
                    .map_or(VersionSpec::Current, VersionSpec::Id),
                registrations: Mutex::new(Vec::new()),
            };
            Ok(Some(LoadedDeclared::new(meta, Some(Arc::new(table)))))
        })
    }

    fn refresh<'a>(
        &'a self,
        caller: &'a CatalogCaller,
        id: &'a TableIdent,
    ) -> BoxFuture<'a, Result<bool, KrishivError>> {
        Box::pin(async move {
            // REST identifiers go to the catalog as written, by the analyzer and by `REFRESH
            // TABLE` alike (the catalog decides what a spelling names), so the entries of this
            // exact spelling are the ones a refresh drops. The count moves first: a load that read the old one and
            // inserts before this `retain` is removed by it, and one inserting after sees the new count (`loaded`).
            self.vended
                .drop_table(self.instance, &id.namespace, &id.name);
            self.cache
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .retain(|(_, ns, name), _| !(ns == &id.namespace && name == &id.name));
            self.epoch.fetch_add(1, Ordering::SeqCst);
            if id.namespace.is_empty() {
                return Ok(false);
            }
            Ok(self
                .loaded(caller, &id.namespace, &id.name)
                .await?
                .is_some())
        })
    }

    fn list_namespaces<'a>(
        &'a self,
        caller: &'a CatalogCaller,
        parent: &'a [String],
    ) -> BoxFuture<'a, Result<Option<Vec<Vec<String>>>, KrishivError>> {
        let who = rest_caller(caller);
        let parent = parent.to_vec();
        Box::pin(self.blocking(move |c| c.list_namespaces_as(&who, &parent)))
    }

    fn list_tables<'a>(
        &'a self,
        caller: &'a CatalogCaller,
        namespace: &'a [String],
    ) -> BoxFuture<'a, Result<Option<Vec<String>>, KrishivError>> {
        let who = rest_caller(caller);
        let namespace = namespace.to_vec();
        Box::pin(async move {
            Ok(self
                .blocking(move |c| c.list_tables_as(&who, &namespace))
                .await?
                .map(|ids| ids.into_iter().map(|t| t.name).collect()))
        })
    }
}

/// One loaded table of a declared REST catalog at one snapshot: what its scans open.
struct RestTable {
    relation: String,
    /// What a `storage = vended` read needs (`None` for `allowed-prefixes`).
    vended: Option<VendedRead>,
    namespace: Vec<String>,
    name: String,
    loaded: Arc<Loaded>,
    spec: VersionSpec,
    /// The storage registrations of the providers opened, kept while the query holds the table.
    registrations: Mutex<Vec<StorageRegistration>>,
}

impl std::fmt::Debug for RestTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RestTable")
            .field("relation", &self.relation)
            .field("spec", &self.spec)
            .finish_non_exhaustive()
    }
}

/// What a vended read of one table needs (module docs of `vended.rs`).
#[derive(Debug)]
struct VendedRead {
    cache: Arc<VendedCache>,
    key: super::vended::Key,
    identity: IdentityKey,
    delegated: bool,
    refetch: Refetch,
}

impl RestTable {
    /// The path locator of a vended read (module docs of `vended.rs`).
    async fn vended_paths(
        &self,
        v: &VendedRead,
        inputs: &ScanInputs,
    ) -> Result<Arc<dyn krishiv_exec::files::PathLocator>, KrishivError> {
        use krishiv_common::config::{
            CATALOG_VENDED_DEFAULT_TTL, CATALOG_VENDED_MAX_ENTRIES, ConfigValue,
            IO_CREDENTIALS_REFRESH_SKEW,
        };
        let config = &inputs.config;
        let duration = |e| config.get(e).and_then(ConfigValue::as_duration);
        let table = VendedTable {
            cache: Arc::clone(&v.cache),
            key: v.key.clone(),
            max_entries: config
                .get(&CATALOG_VENDED_MAX_ENTRIES)
                .and_then(ConfigValue::as_int)
                .and_then(|n| usize::try_from(n).ok())
                .unwrap_or(10_000),
            default_ttl: duration(&CATALOG_VENDED_DEFAULT_TTL).unwrap_or(Duration::from_secs(900)),
            skew: duration(&IO_CREDENTIALS_REFRESH_SKEW)
                .unwrap_or(krishiv_store::DEFAULT_REFRESH_SKEW),
            refetch: v.refetch.clone(),
        };
        // The answer `loadTable` gave, kept unless the cache holds a newer one for this identity or a `REFRESH TABLE`
        // came since its request; either way this query reads with it when the cache holds nothing.
        let own = self.loaded.vended.clone();
        if let Some(a) = &own {
            table.offer(Arc::clone(a));
        }
        let answer = table.answer(own.as_ref()).await?;
        let refuse = |rule| refused(&table.refetch.catalog, &self.relation, rule);
        if answer.is_empty() {
            return Err(refuse(LocationRule::NoVendedCredential));
        }
        let scopes = table.scopes(&answer, &self.loaded.metadata_location)?;
        let mut sources = Vec::new();
        let mut prefixes = Vec::new();
        for location in &scopes {
            let scope = krishiv_store::CredentialScope::new(location)
                .map_err(|_| refuse(LocationRule::NoVendedCredential))?;
            let source: Arc<dyn krishiv_store::CredentialSource> =
                Arc::new(table.source(location, own.clone())?);
            sources.push((scope, source));
            prefixes.push(
                VendedPrefix::parse(location)
                    .map_err(|_| refuse(LocationRule::NoVendedCredential))?,
            );
        }
        let inner = inputs
            .stores
            .paths(cache_scope(&v.identity, v.delegated), sources);
        Ok(Arc::new(VendedPaths {
            inner,
            prefixes,
            catalog: table.refetch.catalog.clone(),
            relation: self.relation.clone(),
        }))
    }
}

impl DeclaredTable for RestTable {
    fn open(
        self: Arc<Self>,
        inputs: ScanInputs,
    ) -> BoxFuture<'static, Result<Arc<dyn TableProvider>, KrishivError>> {
        Box::pin(async move {
            if self.loaded.signing {
                return Err(unsupported(format!(
                    "Iceberg REST remote signing for the table {}",
                    self.relation
                )));
            }
            // `storage = vended`: only the vended credentials, only under their prefixes, refused before any
            // storage request otherwise; never the server's own or ambient credentials.
            let paths = match &self.vended {
                Some(v) => self.vended_paths(v, &inputs).await?,
                None => Arc::clone(&inputs.paths),
            };
            let registration = StorageRegistration::new(inputs.io.clone(), paths);
            let this = Arc::clone(&self);
            let config = Arc::clone(&inputs.config);
            let slots = inputs.worker_slots;
            // The connector's planning future is not provably `Send` (a closure over manifest
            // entries), so it runs to completion on one blocking-pool thread, off the compute
            // workers; outside a runtime, on the caller's thread.
            let handle = tokio::runtime::Handle::try_current().ok();
            let work =
                move || -> Result<(IcebergTableProvider, StorageRegistration), KrishivError> {
                    let table = IcebergTable::from_metadata(
                        &registration,
                        &this.loaded.metadata,
                        &this.loaded.metadata_location,
                        &this.namespace,
                        &this.name,
                        &config,
                    )?;
                    let open = IcebergTableProvider::open(&table, &this.spec, &config, slots);
                    let provider = match handle {
                        Some(h) => h.block_on(open),
                        None => tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .map_err(|e| {
                                unsupported(format!("reading the Iceberg REST table ({e})"))
                            })?
                            .block_on(open),
                    }?;
                    Ok((provider, registration))
                };
            let (provider, registration) = match tokio::runtime::Handle::try_current() {
                Ok(handle) => handle.spawn_blocking(work).await.map_err(|_| {
                    unsupported(
                        "reading the Iceberg REST table (the open did not finish)".to_owned(),
                    )
                })??,
                Err(_) => work()?,
            };
            self.registrations
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(registration);
            Ok(Arc::new(provider) as Arc<dyn TableProvider>)
        })
    }
}

/// The locations an embedded catalog may return: the remote rules, plus absolute local paths
/// (`file:/` or `file:///`) without escapes. Embedded use is not confined by path.
#[derive(Debug, Default)]
pub(crate) struct EmbeddedLocations(RemoteLocationPolicy);

impl LocationPolicy for EmbeddedLocations {
    fn check(&self, location: &str) -> Result<(), LocationRule> {
        let lower = location.get(..5).map(str::to_ascii_lowercase);
        if lower.as_deref() != Some("file:") {
            return self.0.check(location);
        }
        if location.chars().any(|c| c.is_control() || c == '\\') {
            return Err(LocationRule::Malformed);
        }
        let path = location.get(5..).unwrap_or_default();
        let path = path.strip_prefix("//").unwrap_or(path);
        if !path.starts_with('/') {
            return Err(LocationRule::NoScheme);
        }
        let escapes = path.split('/').any(|seg| {
            let seg = seg.to_ascii_lowercase();
            seg == ".." || seg.contains("%2e") || seg.contains("%2f") || seg.contains("%5c")
        });
        if escapes {
            return Err(LocationRule::PathEscape);
        }
        Ok(())
    }

    fn check_scheme(&self, scheme: &str) -> Result<(), LocationRule> {
        if scheme.eq_ignore_ascii_case("file") {
            return Ok(());
        }
        self.0.check_scheme(scheme)
    }
}
