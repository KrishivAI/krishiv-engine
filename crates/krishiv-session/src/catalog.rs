// SPDX-License-Identifier: Apache-2.0
//! The session catalog: temporary tables and views, by name, in
//! `spark_catalog.default` (Spark's session catalog), and what one query
//! sees of it.
//!
//! A temporary table holds batches in memory ([`Session::register_batches`](crate::Session::register_batches));
//! a temporary view over files is a read by path
//! ([`DataFrame::create_or_replace_temp_view`](crate::DataFrame::create_or_replace_temp_view))
//! with the columns it had when it was created: as in Spark, files changed
//! later do not change the view's schema, so analysing a query over it reads
//! no files. Each query loads an entry once and keeps it ([`QueryCatalog`]),
//! so a view replaced while the query runs does not change what it reads.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::files::FileReads;
use krishiv_exec::planner::{MemoryTable, TableProvider, TableProviderResolver};
use krishiv_plan::analyzer::{
    BoxFuture, Catalog, CatalogResolver, NameForm, TableColumn, TableIdent, TableKind, TableMeta,
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
    /// A read by path, with the columns inferred when the view was created.
    Files {
        source: ReadSource,
        columns: Vec<TableColumn>,
    },
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

    /// The provider of the read by path `source`, which lists its files
    /// when the query runs.
    fn files_provider(&self, source: &ReadSource) -> Result<Arc<dyn TableProvider>, KrishivError> {
        // Scans of listed files arrive with the file table provider of
        // the physical planner's next part (P2-T074).
        let _ = (&self.files, source);
        Err(unsupported("running queries that read files"))
    }
}

/// The table `name`: the one this query already loaded, so every spelling
/// of a name binds one table even if another thread replaces it meanwhile,
/// else the session's.
fn resolve(
    loaded: &Mutex<HashMap<String, Arc<TempTable>>>,
    session: &SessionCatalog,
    name: &str,
    form: NameForm,
) -> Option<Arc<TempTable>> {
    let first = loaded
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&name.to_lowercase())
        .cloned();
    match first {
        Some(t) if !form.case_sensitive || session.get(name, true).is_some() => Some(t),
        _ => session.get(name, form.case_sensitive),
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
        form: NameForm,
    ) -> BoxFuture<'a, Result<Option<TableMeta>, KrishivError>> {
        Box::pin(async move {
            // Temporary views resolve for unqualified names only: Spark
            // finds `v` but not `default.v` or `spark_catalog.default.v`.
            let found = if form.qualified {
                None
            } else {
                resolve(&self.loaded, &self.session, &id.name, form)
            };
            let Some(table) = found else {
                return Ok(None);
            };
            let columns = match &*table {
                TempTable::Memory { columns, .. } | TempTable::Files { columns, .. } => {
                    columns.clone()
                }
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
                    TempTable::Files { source, .. } => self.files_provider(source),
                }
            }
            source @ ReadSource::Files { .. } => self.files_provider(source),
            _ => Err(unsupported("this scan")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use krishiv_types::DataType;

    fn files(col: &str) -> TempTable {
        TempTable::Files {
            source: ReadSource::Files {
                format: "parquet".to_owned(),
                paths: vec![],
                options: Default::default(),
            },
            columns: vec![TableColumn {
                name: col.to_owned(),
                data_type: DataType::Integer,
                nullable: false,
            }],
        }
    }

    fn first_column(t: &TempTable) -> &str {
        match t {
            TempTable::Files { columns, .. } | TempTable::Memory { columns, .. } => {
                &columns[0].name
            }
        }
    }

    #[test]
    fn every_spelling_binds_the_table_loaded_first() {
        let session = SessionCatalog::default();
        session.put("people", files("a"));
        let loaded = Mutex::new(HashMap::new());
        let form = NameForm {
            case_sensitive: false,
            qualified: false,
        };
        let t = resolve(&loaded, &session, "people", form).unwrap();
        loaded.lock().unwrap().insert("people".to_owned(), t);
        // Another thread replaces the table.
        session.put("people", files("b"));
        let again = resolve(&loaded, &session, "PEOPLE", form).unwrap();
        assert_eq!(first_column(&again), "a");
        // A case-sensitive spelling that is not registered does not resolve
        // through the loaded entry.
        let cs = NameForm {
            case_sensitive: true,
            ..form
        };
        assert!(resolve(&loaded, &session, "PEOPLE", cs).is_none());
    }
}
