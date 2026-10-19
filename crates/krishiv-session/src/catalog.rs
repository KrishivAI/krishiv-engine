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
//! A query lists the files of each read (and view) once, before it is
//! planned, and scans them as a [`FileTable`] whose footers give the
//! optimizer its statistics.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use arrow_schema::{Schema, SchemaRef};
use krishiv_common::config::Config;
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_exec::files::FileReads;
use krishiv_exec::planner::{FileTable, MemoryTable, TableProvider, TableProviderResolver};
use krishiv_io::IoContext;
use krishiv_plan::analyzer::{
    BoxFuture, Catalog, CatalogResolver, NameForm, TableColumn, TableIdent, TableKind, TableMeta,
};
use krishiv_plan::{ExprNode, LogicalPlan, PlanNode, ReadSource, Scan, TableVersion};

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

/// What a query reads files for: a view (by lower-case name) or a read by
/// path.
#[derive(Debug, Clone, PartialEq)]
enum FilesKey {
    View(String),
    Read(ReadSource),
}

/// How a query opens the file tables of its scans.
#[derive(Debug, Clone)]
pub(crate) struct FileScans {
    /// The I/O the scans read through.
    pub(crate) io: IoContext,
    /// The query's configuration (split sizes, ...).
    pub(crate) config: Arc<Config>,
    /// The executor's drivers, for split planning.
    pub(crate) slots: u64,
    /// The I/O runtime, entered by the thread that opens file tables.
    pub(crate) runtime: tokio::runtime::Handle,
}

/// The session catalog as one query sees it: the catalog resolver and the
/// table providers of its scans.
pub(crate) struct QueryCatalog {
    session: Arc<SessionCatalog>,
    files: Arc<FileReads>,
    scans: FileScans,
    /// Entries loaded by this query, by lower-case name.
    loaded: Mutex<HashMap<String, Arc<TempTable>>>,
    /// The file tables of the query's scans, opened before planning.
    opened: Mutex<Vec<(FilesKey, Arc<FileTable>)>>,
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
    pub(crate) fn new(
        session: Arc<SessionCatalog>,
        files: Arc<FileReads>,
        scans: FileScans,
    ) -> Self {
        let epoch = session.epoch();
        Self {
            session,
            files,
            scans,
            loaded: Mutex::new(HashMap::new()),
            opened: Mutex::new(Vec::new()),
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

    /// Lists the files of every read by path and file view that the
    /// analyzed `plan` scans (subqueries included), and opens them as file
    /// tables for the planner. A read the analyzer listed keeps that
    /// listing; a view lists its files now.
    ///
    /// # Errors
    /// Listing errors (`PATH_NOT_FOUND` once a view's files are gone, ...)
    /// and errors reading the files' footers.
    pub(crate) async fn open_files(&self, plan: &LogicalPlan) -> Result<(), KrishivError> {
        for (key, source, table) in self.file_scans(plan) {
            if self.opened(&key).is_some() {
                continue;
            }
            let ReadSource::Files {
                format,
                paths,
                options,
            } = &source
            else {
                continue;
            };
            let listing = self.files.listing(format, paths, options).await?;
            let opened = open_table(
                Arc::clone(&listing.format),
                self.scans.clone(),
                table,
                listing.files.to_vec(),
                listing.options.clone(),
            )
            .await?;
            self.opened
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push((key, Arc::new(opened)));
        }
        Ok(())
    }

    /// The scans of files in `plan`: what each reads, and the columns the
    /// analyzer bound (the view's, for a view), which the files map to by
    /// name.
    fn file_scans(&self, plan: &LogicalPlan) -> Vec<(FilesKey, ReadSource, SchemaRef)> {
        let mut scans = Vec::new();
        collect_scans(plan, &mut scans);
        scans
            .into_iter()
            .filter_map(|scan| {
                let (key, source) = match &scan.source {
                    ReadSource::Table { name, .. } => {
                        let last = name.last().map_or("", String::as_str);
                        let table = self.loaded(last)?;
                        let TempTable::Files { source, .. } = &*table else {
                            return None;
                        };
                        (FilesKey::View(last.to_lowercase()), source.clone())
                    }
                    source @ ReadSource::Files { .. } => {
                        (FilesKey::Read(source.clone()), source.clone())
                    }
                    _ => return None,
                };
                let table = Arc::new(Schema::new(
                    scan.output
                        .iter()
                        .map(|c| c.data_type.to_arrow_field(c.name.clone(), c.nullable))
                        .collect::<Vec<_>>(),
                ));
                Some((key, source, table))
            })
            .collect()
    }

    /// Whether this query loaded a file view: its files are listed again
    /// by the next query, so an analysis that did must not be reused.
    pub(crate) fn loaded_file_views(&self) -> bool {
        self.loaded
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .any(|t| matches!(&**t, TempTable::Files { .. }))
    }

    fn opened(&self, key: &FilesKey) -> Option<Arc<FileTable>> {
        self.opened
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, t)| Arc::clone(t))
    }

    /// The file table of `key`, opened by [`Self::open_files`].
    fn files_provider(&self, key: &FilesKey) -> Result<Arc<dyn TableProvider>, KrishivError> {
        match self.opened(key) {
            Some(t) => Ok(t as Arc<dyn TableProvider>),
            None => Err(InternalError {
                message: "a scan of files that was not opened before planning".to_owned(),
            }
            .into()),
        }
    }
}

/// [`FileTable::open`] on a thread of its own. Its future cannot be proven
/// `Send` (a higher-ranked closure in its footer stream), so it cannot run
/// on the session's executor; the thread enters the I/O runtime, whose
/// tasks and timers its reads use (on a deterministic executor that is the
/// caller's current-thread runtime, which drives them while the action
/// awaits this thread). The
/// thread only waits for footers, so no compute or blocking-pool slot is
/// held. Dropping the returned future abandons the open.
async fn open_table(
    format: Arc<dyn krishiv_io::FileFormat>,
    scans: FileScans,
    table: SchemaRef,
    files: Vec<krishiv_io::FileRef>,
    options: krishiv_io::FormatOptions,
) -> Result<FileTable, KrishivError> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    std::thread::Builder::new()
        .name("krishiv-file-table".to_owned())
        .spawn(move || {
            let _runtime = scans.runtime.enter();
            let opened = futures::executor::block_on(FileTable::open(
                format,
                scans.io,
                table,
                files,
                options,
                &scans.config,
                scans.slots,
            ));
            // The query may be gone; its table is then not needed.
            let _ = tx.send(opened);
        })
        .map_err(|e| {
            internal(format!(
                "could not start a thread to read file footers: {e}"
            ))
        })?;
    rx.await
        .map_err(|_| internal("reading file footers ended without a result".to_owned()))?
}

fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// The scans of `plan` and of the subqueries in its expressions.
fn collect_scans<'a>(plan: &'a LogicalPlan, out: &mut Vec<&'a Scan>) {
    fn in_expr<'a>(e: &'a Arc<krishiv_plan::Expr>, out: &mut Vec<&'a Scan>) {
        if let ExprNode::Subquery { plan, .. } = &e.node {
            collect_scans(plan, out);
        }
        e.for_each_child(&mut |c| in_expr(c, out));
    }
    if let PlanNode::Scan(scan) = &plan.node {
        out.push(scan);
    }
    plan.for_each_expr(&mut |e| in_expr(e, out));
    plan.for_each_child(&mut |c| collect_scans(c, out));
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

    fn single_part_namespace(&self) -> bool {
        true
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
                    TempTable::Files { .. } => {
                        self.files_provider(&FilesKey::View(last.to_lowercase()))
                    }
                }
            }
            source @ ReadSource::Files { .. } => {
                self.files_provider(&FilesKey::Read(source.clone()))
            }
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
