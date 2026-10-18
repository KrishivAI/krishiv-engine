// SPDX-License-Identifier: Apache-2.0
//! The session catalog: temporary tables and views, by name, in
//! `spark_catalog.default` (Spark's session catalog), and what one query
//! sees of it.
//!
//! A temporary table holds batches in memory ([`Session::register_batches`](crate::Session::register_batches));
//! a temporary view over files is a read by path
//! ([`DataFrame::create_or_replace_temp_view`](crate::DataFrame::create_or_replace_temp_view)),
//! listed again by every query that reads it. Each query loads an entry
//! once and keeps it ([`QueryCatalog`]), so a view replaced while the query
//! runs does not change what it reads.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::files::FileReads;
use krishiv_exec::planner::{MemoryTable, TableProvider, TableProviderResolver};
use krishiv_plan::analyzer::{
    BoxFuture, Catalog, CatalogResolver, FileSchemaResolver, FunctionLookup, FunctionRegistry,
    SparkProfile, TableColumn, TableIdent, TableKind, TableMeta,
};
use krishiv_plan::{ReadSource, Scan, TableVersion};

/// The session catalog's name, as Spark names it.
pub(crate) const CATALOG: &str = "spark_catalog";
/// The namespace temporary tables live in.
pub(crate) const NAMESPACE: &str = "default";

/// A temporary table or view.
#[derive(Debug)]
pub(crate) enum TempTable {
    /// Batches in memory.
    Memory {
        columns: Vec<TableColumn>,
        table: Arc<MemoryTable>,
    },
    /// A read by path.
    Files(ReadSource),
}

/// The temporary tables and views of a session, by lower-case name.
#[derive(Debug, Default)]
pub(crate) struct SessionCatalog {
    tables: RwLock<BTreeMap<String, (String, Arc<TempTable>)>>,
    epoch: AtomicU64,
}

impl SessionCatalog {
    /// Adds or replaces `name`.
    pub(crate) fn put(&self, name: &str, table: TempTable) {
        self.tables
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(name.to_lowercase(), (name.to_owned(), Arc::new(table)));
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// Removes `name`; whether it existed.
    pub(crate) fn drop_table(&self, name: &str) -> bool {
        let removed = self
            .tables
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&name.to_lowercase())
            .is_some();
        if removed {
            self.epoch.fetch_add(1, Ordering::AcqRel);
        }
        removed
    }

    /// The names, as registered, in order.
    pub(crate) fn names(&self) -> Vec<String> {
        self.tables
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .map(|(n, _)| n.clone())
            .collect()
    }

    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// The entry `name`: case-insensitively unless `case_sensitive`, when
    /// the name must be as registered.
    fn get(&self, name: &str, case_sensitive: bool) -> Option<Arc<TempTable>> {
        let tables = self.tables.read().unwrap_or_else(PoisonError::into_inner);
        let (registered, table) = tables.get(&name.to_lowercase())?;
        (!case_sensitive || registered == name).then(|| Arc::clone(table))
    }
}

/// The session catalog as one query sees it: the catalog resolver and the
/// table providers of its scans.
pub(crate) struct QueryCatalog {
    session: Arc<SessionCatalog>,
    files: Arc<FileReads>,
    /// Entries loaded by this query, by lower-case name.
    loaded: Mutex<HashMap<String, Arc<TempTable>>>,
    epoch: u64,
}

impl fmt::Debug for QueryCatalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueryCatalog")
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}

impl QueryCatalog {
    pub(crate) fn new(session: Arc<SessionCatalog>, files: Arc<FileReads>) -> Self {
        let epoch = session.epoch();
        Self {
            session,
            files,
            loaded: Mutex::new(HashMap::new()),
            epoch,
        }
    }

    fn loaded(&self, name: &str) -> Option<Arc<TempTable>> {
        self.loaded
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&name.to_lowercase())
            .cloned()
    }

    /// The provider of a read by path this query listed.
    fn files_provider(&self) -> Result<Arc<dyn TableProvider>, KrishivError> {
        // Scans of listed files arrive with the file table provider of
        // the physical planner's next part (P2-T074).
        let _ = &self.files;
        Err(unsupported("running queries that read files"))
    }
}

fn unsupported(feature: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.to_owned(),
    }
    .into()
}

impl Catalog for QueryCatalog {
    fn name(&self) -> &str {
        CATALOG
    }

    fn epoch(&self) -> u64 {
        self.epoch
    }

    fn load_table<'a>(
        &'a self,
        id: &'a TableIdent,
        _: Option<TableVersion>,
        case_sensitive: bool,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        Box::pin(async move {
            let in_default = match id.namespace.as_slice() {
                [ns] if case_sensitive => ns == NAMESPACE,
                [ns] => ns.eq_ignore_ascii_case(NAMESPACE),
                _ => false,
            };
            let found = if in_default {
                self.session.get(&id.name, case_sensitive)
            } else {
                None
            };
            let Some(table) = found else {
                return Ok(None);
            };
            let columns = match &*table {
                TempTable::Memory { columns, .. } => columns.clone(),
                TempTable::Files(ReadSource::Files {
                    format,
                    paths,
                    options,
                }) => self.files.infer(format, paths, options).await?,
                TempTable::Files(_) => return Err(unsupported("this temporary view")),
            };
            self.loaded
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(id.name.to_lowercase(), table);
            Ok(Some(TableMeta {
                ident: TableIdent {
                    catalog: CATALOG.to_owned(),
                    namespace: vec![NAMESPACE.to_owned()],
                    name: id.name.to_lowercase(),
                },
                columns,
                version: None,
                kind: TableKind::Table,
            }))
        })
    }
}

/// The resolver of one query: the session catalog only.
#[derive(Debug)]
pub(crate) struct QueryCatalogs(pub(crate) Arc<QueryCatalog>);

impl CatalogResolver for QueryCatalogs {
    fn current_catalog(&self) -> &str {
        CATALOG
    }

    fn current_namespace(&self) -> &[String] {
        static DEFAULT: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
        DEFAULT.get_or_init(|| vec![NAMESPACE.to_owned()])
    }

    fn catalog(&self, name: &str) -> Option<Arc<dyn Catalog>> {
        name.eq_ignore_ascii_case(CATALOG)
            .then(|| Arc::clone(&self.0) as Arc<dyn Catalog>)
    }

    fn epoch(&self) -> u64 {
        0
    }
}

impl TableProviderResolver for QueryCatalog {
    fn provider(&self, scan: &Scan) -> Result<Arc<dyn TableProvider>, KrishivError> {
        match &scan.source {
            ReadSource::Table { name, .. } => {
                let last = name.last().map_or("", String::as_str);
                // Analysis loaded every table a scan names.
                let Some(table) = self.loaded(last) else {
                    return Err(InternalError {
                        message: format!("the scanned table {last} was not loaded"),
                    }
                    .into());
                };
                match &*table {
                    TempTable::Memory { table, .. } => {
                        Ok(Arc::clone(table) as Arc<dyn TableProvider>)
                    }
                    TempTable::Files(_) => self.files_provider(),
                }
            }
            ReadSource::Files { .. } => self.files_provider(),
            _ => Err(unsupported("this scan")),
        }
    }
}

/// The built-in functions until the registry (P2-T153) lands: every name is
/// known to Spark but not implemented, so a call fails with
/// `KRISHIV_UNSUPPORTED_FEATURE`. Operators need no registry.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PendingFunctions;

impl FunctionRegistry for PendingFunctions {
    fn lookup(&self, _: &[String], _: SparkProfile) -> FunctionLookup {
        FunctionLookup::KnownUnsupported
    }

    fn version(&self) -> u64 {
        0
    }
}
