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

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};

use arrow_schema::{Schema, SchemaRef};
use krishiv_common::config::Config;
use krishiv_common::error::{
    InternalError, KrishivError, KrishivReservedTempViewName, KrishivUnsupportedFeature,
    RecursiveView, TableOrViewNotFound,
};
use krishiv_exec::files::FileReads;
use krishiv_exec::planner::{FileTable, MemoryTable, TableProvider, TableProviderResolver};
use krishiv_io::IoContext;
use krishiv_plan::analyzer::{
    BoxFuture, Catalog, CatalogResolver, NameForm, TableColumn, TableIdent, TableKind, TableMeta,
};
use krishiv_plan::{
    Cte, Expr, ExprNode, LogicalPlan, Origin, PlanNode, ReadSource, Scan, TableVersion,
};

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
    /// A view over a query: the unresolved plan and the output columns it
    /// had when it was created (renamed when the view declares aliases).
    /// Queries inline the plan as a subquery, so a view replaced while a
    /// query runs does not change what it reads, as for tables.
    ///
    /// A DataFrame view also owns the hidden names its snapshot pinned
    /// ([`pin_temp_tables`]): replacing or dropping the view removes the
    /// pins no other live view still owns. SQL views keep the unresolved
    /// query, follow the tables they read, and own no pins.
    View {
        plan: Arc<LogicalPlan>,
        columns: Vec<TableColumn>,
        /// The hidden names this view pinned, owned by this view.
        pins: Vec<String>,
    },
}

/// The prefix of the hidden names snapshots pin tables under. No user
/// input spells them: [`expand_temp_views`] reports a read of one as a
/// missing table, and creating a temporary table or view with such a
/// name fails with `KRISHIV_RESERVED_TEMP_VIEW_NAME`, which names this
/// prefix. [`SessionCatalog::names`] hides them.
pub(crate) const HIDDEN_PREFIX: &str = "__krishiv_snapshot_";

/// Whether `name` is a hidden snapshot pin: it starts with
/// [`HIDDEN_PREFIX`], in any letter case.
pub(crate) fn is_hidden_name(name: &str) -> bool {
    name.get(..HIDDEN_PREFIX.len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(HIDDEN_PREFIX))
}

/// The temporary tables and views of a session, by lower-case name.
#[derive(Debug, Default)]
pub(crate) struct SessionCatalog {
    tables: RwLock<BTreeMap<String, (String, Arc<TempTable>)>>,
    epoch: AtomicU64,
    /// Hidden names handed out by [`SessionCatalog::pin`].
    hidden: AtomicU64,
}

impl SessionCatalog {
    /// Adds or replaces `name`, removing the snapshot pins the old entry
    /// owned that no other live view still owns.
    pub(crate) fn put(&self, name: &str, table: TempTable) {
        let mut tables = self.tables.write().unwrap_or_else(PoisonError::into_inner);
        tables.insert(name.to_lowercase(), (name.to_owned(), Arc::new(table)));
        Self::collect_garbage(&mut tables);
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// Removes every hidden entry no live view pins anymore. The caller
    /// holds the write lock and bumps the epoch itself.
    fn collect_garbage(tables: &mut BTreeMap<String, (String, Arc<TempTable>)>) {
        let mut live = BTreeSet::new();
        for (_, table) in tables.values() {
            if let TempTable::View { pins, .. } = &**table {
                live.extend(pins.iter().cloned());
            }
        }
        tables.retain(|key, _| !is_hidden_name(key) || live.contains(key));
    }

    /// Pins `table` (an entry of this catalog) under a hidden name that no
    /// SQL text spells, returning the name: snapshot views read it however
    /// the session's names change afterwards.
    pub(crate) fn pin(&self, table: &Arc<TempTable>) -> String {
        let name = format!(
            "{HIDDEN_PREFIX}{}",
            self.hidden.fetch_add(1, Ordering::AcqRel)
        );
        self.tables
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(name.clone(), (name.clone(), Arc::clone(table)));
        self.epoch.fetch_add(1, Ordering::AcqRel);
        name
    }

    /// Removes `name` (and the snapshot pins it owned that no other live
    /// view still owns); whether it existed. Hidden names are never
    /// removed this way: dropping one behaves as if it did not exist.
    pub(crate) fn drop_table(&self, name: &str) -> bool {
        if is_hidden_name(name) {
            return false;
        }
        let mut tables = self.tables.write().unwrap_or_else(PoisonError::into_inner);
        let removed = tables.remove(&name.to_lowercase()).is_some();
        if removed {
            Self::collect_garbage(&mut tables);
            self.epoch.fetch_add(1, Ordering::AcqRel);
        }
        removed
    }

    /// The names, as registered, in order, without snapshot pins.
    pub(crate) fn names(&self) -> Vec<String> {
        self.tables
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .map(|(n, _)| n.clone())
            .filter(|n| !n.starts_with(HIDDEN_PREFIX))
            .collect()
    }

    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// The entry `name`: case-insensitively unless `case_sensitive`, when
    /// the name must be as registered. Unlike [`exists`](Self::exists),
    /// hidden names resolve here: snapshot views read their pins through
    /// this. User input never reaches a pin: [`expand_temp_views`]
    /// reports a read of one as a missing table before analysis loads
    /// anything, and creating or dropping one behaves as missing too.
    pub(crate) fn get(&self, name: &str, case_sensitive: bool) -> Option<Arc<TempTable>> {
        let tables = self.tables.read().unwrap_or_else(PoisonError::into_inner);
        let (registered, table) = tables.get(&name.to_lowercase())?;
        (!case_sensitive || registered == name).then(|| Arc::clone(table))
    }

    /// Whether an entry `name` exists: case-insensitively unless
    /// `case_sensitive`, when the name must be as registered. Hidden
    /// names never exist for user input, however they are spelled.
    pub(crate) fn exists(&self, name: &str, case_sensitive: bool) -> bool {
        if is_hidden_name(name) {
            return false;
        }
        let tables = self.tables.read().unwrap_or_else(PoisonError::into_inner);
        match tables.get(&name.to_lowercase()) {
            Some((registered, _)) => !case_sensitive || registered == name,
            None => false,
        }
    }

    /// The number of catalog entries, snapshot pins included: tests only.
    /// [`names`](Self::names) stays the user-visible count.
    #[cfg(test)]
    pub(crate) fn entry_count(&self) -> usize {
        self.tables
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
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
    /// The executor's I/O runtime, which opens the file tables.
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

/// [`FileTable::open`] on the executor's I/O runtime: footer reads are
/// object-store requests, which belong there and not on the compute
/// runtime. Dropping the returned future aborts the open.
async fn open_table(
    format: Arc<dyn krishiv_io::FileFormat>,
    scans: FileScans,
    table: SchemaRef,
    files: Vec<krishiv_io::FileRef>,
    options: krishiv_io::FormatOptions,
) -> Result<FileTable, KrishivError> {
    let runtime = scans.runtime.clone();
    run_on(&runtime, async move {
        FileTable::open(
            format,
            scans.io,
            table,
            files,
            options,
            &scans.config,
            scans.slots,
        )
        .await
    })
    .await
}

/// Aborts its task when dropped.
struct AbortGuard<T>(tokio::task::JoinHandle<T>);

impl<T> Drop for AbortGuard<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Runs `work` as a task of `runtime` and waits for it; dropping the
/// returned future aborts the task.
async fn run_on<T: Send + 'static>(
    runtime: &tokio::runtime::Handle,
    work: impl Future<Output = Result<T, KrishivError>> + Send + 'static,
) -> Result<T, KrishivError> {
    let mut guard = AbortGuard(runtime.spawn(work));
    match (&mut guard.0).await {
        Ok(result) => result,
        Err(e) => Err(KrishivError::from(InternalError {
            message: format!("reading file footers ended without a result: {e}"),
        })),
    }
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
            let (columns, kind) = match &*table {
                TempTable::Memory { columns, .. } | TempTable::Files { columns, .. } => {
                    (columns.clone(), TableKind::Table)
                }
                TempTable::View { columns, .. } => (columns.clone(), TableKind::View),
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
                kind,
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
                    // Views are inlined as subqueries before analysis, so
                    // no scan ever reads one.
                    TempTable::View { .. } => Err(InternalError {
                        message: "a scan of a temporary view that was not inlined".to_owned(),
                    }
                    .into()),
                }
            }
            source @ ReadSource::Files { .. } => {
                self.files_provider(&FilesKey::Read(source.clone()))
            }
            _ => Err(unsupported("this scan")),
        }
    }
}

/// Inlines the temporary views `plan` reads: a read of a view (a
/// single-part name with no options, not shadowed by a `WITH` definition)
/// becomes the view's plan as a subquery named for the view, with the
/// columns the view had when it was created. Tables, file views and reads
/// by path are left for the analyzer, as are reads a `WITH` definition
/// shadows (definitions shadow tables, as in Spark).
///
/// Views over views inline recursively; a view reading itself (through
/// `OR REPLACE`) is an error, as the engine has no recursive views.
///
/// # Errors
/// `INTERNAL_ERROR` for a view that reads itself.
pub(crate) fn expand_temp_views(
    catalog: &SessionCatalog,
    plan: &Arc<LogicalPlan>,
    case_sensitive: bool,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    Expander {
        catalog,
        case_sensitive,
        path: Vec::new(),
    }
    .go(Arc::clone(plan), &mut Vec::new())
}

/// [`expand_temp_views`]: the views on the current expansion path, by
/// lower-case name, for cycle detection.
struct Expander<'a> {
    catalog: &'a SessionCatalog,
    case_sensitive: bool,
    path: Vec<String>,
}

impl Expander<'_> {
    /// `plan` with its views inlined; `scope` holds the `WITH`
    /// definitions visible where `plan` reads.
    fn go(
        &mut self,
        plan: Arc<LogicalPlan>,
        scope: &mut Vec<String>,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        if let PlanNode::Read(ReadSource::Table { name, options }) = &plan.node
            && let [view] = name.as_slice()
            && options.is_empty()
            && !scope
                .iter()
                .any(|c| same_name(c, view, self.case_sensitive))
        {
            // Snapshot pins are unreachable from user input: a read of
            // one behaves exactly as a missing table. (The pins snapshot
            // views read are scans, which pass through untouched below.)
            if is_hidden_name(view) {
                return Err(TableOrViewNotFound {
                    relation_name: quote(view),
                    search_path: search_path(),
                }
                .into());
            }
            if let Some(entry) = self.catalog.get(view, self.case_sensitive)
                && let TempTable::View {
                    plan: definition,
                    columns,
                    ..
                } = &*entry
            {
                let key = view.to_lowercase();
                if self.path.contains(&key) {
                    return Err(InternalError {
                        message: format!("temporary view `{key}` reads itself"),
                    }
                    .into());
                }
                self.path.push(key);
                // A view's definitions are its own: outer `WITH` names do not
                // reach into it.
                let input = self.go(Arc::clone(definition), &mut Vec::new());
                self.path.pop();
                let columns = columns.iter().map(|c| c.name.clone()).collect();
                return Ok(Arc::new(
                    LogicalPlan::new(PlanNode::SubqueryAlias {
                        input: input?,
                        alias: view.clone(),
                        columns,
                    })
                    .with_origin(plan.origin),
                ));
            }
        }
        if let PlanNode::WithCte { input, ctes } = &plan.node {
            let (input, ctes) = (Arc::clone(input), ctes.clone());
            return self.go_with_cte(plan.origin, input, ctes, scope);
        }
        self.children_and_exprs(plan, scope)
    }

    /// `WITH` query parts (of `origin`) with their views inlined: each
    /// definition expands with only the preceding definitions in scope, as
    /// in Spark, so a body still reads a view of its own name.
    fn go_with_cte(
        &mut self,
        origin: Origin,
        input: Arc<LogicalPlan>,
        ctes: Vec<Cte>,
        scope: &mut Vec<String>,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        let outer = scope.len();
        let mut expanded = Vec::with_capacity(ctes.len());
        for cte in ctes {
            let definition = self.go(Arc::clone(&cte.plan), scope)?;
            scope.push(cte.name.clone());
            expanded.push(Cte {
                name: cte.name,
                plan: definition,
            });
        }
        let input = self.go(input, scope)?;
        scope.truncate(outer);
        Ok(Arc::new(
            LogicalPlan::new(PlanNode::WithCte {
                input,
                ctes: expanded,
            })
            .with_origin(origin),
        ))
    }

    /// `plan` with its child relations and the subquery plans in its
    /// expressions expanded.
    fn children_and_exprs(
        &mut self,
        plan: Arc<LogicalPlan>,
        scope: &mut Vec<String>,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        let plan = plan.map_children(|child| self.go(child, scope))?;
        plan.map_expressions(|expr| self.expand_expr(expr, scope))
    }

    /// `expr` with the views read by the subquery plans inside it inlined.
    fn expand_expr(
        &mut self,
        expr: Arc<Expr>,
        scope: &mut Vec<String>,
    ) -> Result<Arc<Expr>, KrishivError> {
        let expr = expr.map_children(|child| self.expand_expr(child, scope))?;
        if let ExprNode::Subquery { kind, plan } = &expr.node {
            let expanded = self.go(Arc::clone(plan), scope)?;
            if !Arc::ptr_eq(&expanded, plan) {
                return Ok(Arc::new(Expr {
                    node: ExprNode::Subquery {
                        kind: kind.clone(),
                        plan: expanded,
                    },
                    origin: expr.origin,
                }));
            }
        }
        Ok(expr)
    }
}

/// Whether creating the temporary view `name` over `plan` would close a
/// recursive reference: `plan` reads `name` itself, or a view that reads
/// it transitively. Spark reports that when the view is created, as
/// `RECURSIVE_VIEW`; the read-time guard in [`expand_temp_views`] stays
/// for anything analysis meets another way.
///
/// A read of `name` counts only when the view exists (an `OR REPLACE`
/// whose query sees the old definition): a fresh view over its own name
/// resolves nowhere, and analysis reports it as missing.
///
/// # Errors
/// `RECURSIVE_VIEW` naming the view and the cycle.
pub(crate) fn check_view_cycle(
    catalog: &SessionCatalog,
    name: &str,
    plan: &Arc<LogicalPlan>,
    case_sensitive: bool,
) -> Result<(), KrishivError> {
    Checker {
        catalog,
        case_sensitive,
        target: name,
        stack: vec![name.to_owned()],
        scope: Vec::new(),
    }
    .check(plan)
}

/// [`check_view_cycle`]: the views on the current expansion path from the
/// new view down (by name as read), and the `WITH` definitions in scope.
struct Checker<'a> {
    catalog: &'a SessionCatalog,
    case_sensitive: bool,
    target: &'a str,
    stack: Vec<String>,
    scope: Vec<String>,
}

impl Checker<'_> {
    fn check(&mut self, plan: &Arc<LogicalPlan>) -> Result<(), KrishivError> {
        if let PlanNode::Read(ReadSource::Table { name, options }) = &plan.node
            && let [view] = name.as_slice()
            && options.is_empty()
            && !self
                .scope
                .iter()
                .any(|c| same_name(c, view, self.case_sensitive))
        {
            if same_name(view, self.target, self.case_sensitive) {
                if self.catalog.get(self.target, self.case_sensitive).is_some() {
                    return Err(recursive_view(self.target, &self.stack));
                }
            } else if let Some(entry) = self.catalog.get(view, self.case_sensitive)
                && let TempTable::View {
                    plan: definition, ..
                } = &*entry
            {
                let key = view.to_lowercase();
                if self.stack.iter().any(|v| v.to_lowercase() == key) {
                    return Err(recursive_view(self.target, &self.stack));
                }
                self.stack.push(view.clone());
                // A view's definitions are its own: outer `WITH` names do
                // not reach into it.
                let outer = std::mem::take(&mut self.scope);
                let found = self.check(definition);
                self.scope = outer;
                self.stack.pop();
                found?;
            }
            return Ok(());
        }
        if let PlanNode::WithCte { input, ctes } = &plan.node {
            let outer = self.scope.len();
            for cte in ctes.iter() {
                self.check(&cte.plan)?;
                self.scope.push(cte.name.clone());
            }
            self.check(input)?;
            self.scope.truncate(outer);
            return Ok(());
        }
        let mut children = Vec::new();
        plan.for_each_child(&mut |child| children.push(Arc::clone(child)));
        for child in &children {
            self.check(child)?;
        }
        let mut exprs = Vec::new();
        plan.for_each_expr(&mut |expr| exprs.push(Arc::clone(expr)));
        for expr in &exprs {
            self.check_expr(expr)?;
        }
        Ok(())
    }

    /// `expr` with the views read by the subquery plans inside it checked.
    fn check_expr(&mut self, expr: &Arc<Expr>) -> Result<(), KrishivError> {
        let mut children = Vec::new();
        expr.for_each_child(&mut |child| children.push(Arc::clone(child)));
        for child in &children {
            self.check_expr(child)?;
        }
        if let ExprNode::Subquery { plan, .. } = &expr.node {
            self.check(plan)?;
        }
        Ok(())
    }
}

/// `RECURSIVE_VIEW` for a cycle through `stack` back to `target`.
fn recursive_view(target: &str, stack: &[String]) -> KrishivError {
    let mut path: Vec<String> = stack.iter().map(|v| quote(v)).collect();
    path.push(quote(target));
    RecursiveView {
        view_ident: quote(target),
        new_path: path.join(" -> "),
    }
    .into()
}

/// Snapshots the temporary memory tables the analyzed `plan` reads: pins
/// each under a hidden name and rewrites its scans to it, so the plan stays
/// bound to the tables it was analyzed against however the session's names
/// change afterwards (a view built from itself reads the old definition).
/// Reads of files are self-contained and need no pinning.
///
/// A scan that already reads a pin (inlined from another view) reuses it,
/// so views over views share the pin instead of stacking a new one per
/// view. Returns the pinned plan and the hidden names it reads: the new
/// view owns them (see [`TempTable::View`]).
///
/// # Errors
/// `INTERNAL_ERROR` for a scan of a temporary view that was not inlined.
pub(crate) fn pin_temp_tables(
    catalog: &SessionCatalog,
    plan: &Arc<LogicalPlan>,
    case_sensitive: bool,
) -> Result<(Arc<LogicalPlan>, Vec<String>), KrishivError> {
    let mut pinner = Pinner {
        catalog,
        case_sensitive,
        pins: HashMap::new(),
        owned: BTreeSet::new(),
    };
    let plan = pinner.run(plan)?;
    Ok((plan, pinner.owned.into_iter().collect()))
}

/// [`pin_temp_tables`]: the pins handed out so far, by lower-case table
/// name, and every hidden name the pinned plan reads.
struct Pinner<'a> {
    catalog: &'a SessionCatalog,
    case_sensitive: bool,
    pins: HashMap<String, String>,
    owned: BTreeSet<String>,
}

impl Pinner<'_> {
    fn run(&mut self, plan: &Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, KrishivError> {
        if let PlanNode::Scan(scan) = &plan.node
            && let ReadSource::Table { name, .. } = &scan.source
            && let Some(last) = name.last()
        {
            if is_hidden_name(last) {
                self.owned.insert(last.to_lowercase());
                return Ok(Arc::clone(plan));
            }
            let key = last.to_lowercase();
            if !self.pins.contains_key(&key) {
                let pin = match self.catalog.get(last, self.case_sensitive) {
                    Some(table) if matches!(&*table, TempTable::Memory { .. }) => {
                        let hidden = self.catalog.pin(&table);
                        self.owned.insert(hidden.clone());
                        hidden
                    }
                    _ => String::new(),
                };
                self.pins.insert(key.clone(), pin);
            }
            if let Some(hidden) = self.pins.get(&key).filter(|h| !h.is_empty()) {
                let mut pinned = scan.clone();
                if let ReadSource::Table { name, .. } = &mut pinned.source {
                    *name = vec![hidden.clone()];
                }
                return Ok(Arc::new(
                    LogicalPlan::new(PlanNode::Scan(pinned)).with_origin(plan.origin),
                ));
            }
            return Ok(Arc::clone(plan));
        }
        let plan = Arc::clone(plan).map_children(|child| self.run(&child))?;
        plan.map_expressions(|expr| self.run_expr(&expr))
    }

    /// `expr` with the scans read by the subquery plans inside it pinned.
    fn run_expr(&mut self, expr: &Arc<Expr>) -> Result<Arc<Expr>, KrishivError> {
        let expr = Arc::clone(expr).map_children(|child| self.run_expr(&child))?;
        if let ExprNode::Subquery { kind, plan } = &expr.node {
            let pinned = self.run(plan)?;
            if !Arc::ptr_eq(&pinned, plan) {
                return Ok(Arc::new(Expr {
                    node: ExprNode::Subquery {
                        kind: kind.clone(),
                        plan: pinned,
                    },
                    origin: expr.origin,
                }));
            }
        }
        Ok(expr)
    }
}

/// Whether `a` and `b` name the same view: exactly when `case_sensitive`,
/// else ignoring ASCII case, as the session catalog stores names.
fn same_name(a: &str, b: &str, case_sensitive: bool) -> bool {
    a == b || !case_sensitive && a.eq_ignore_ascii_case(b)
}

/// A name as Spark quotes it in errors: `` `name` ``.
fn quote(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

/// Spark's search path, as analysis reports it for a missing table.
fn search_path() -> String {
    format!("[`system`.`builtin`, `system`.`session`, `{CATALOG}`.`{NAMESPACE}`]")
}

/// `name` rejected as a temporary table or view name: it starts with the
/// reserved snapshot prefix ([`HIDDEN_PREFIX`]), in any letter case. The
/// `KRISHIV_RESERVED_TEMP_VIEW_NAME` error names the reserved prefix in
/// its `prefix` parameter.
pub(crate) fn reserved_temp_view_name(name: &str) -> KrishivError {
    KrishivReservedTempViewName {
        relation_name: quote(name),
        prefix: format!("`{HIDDEN_PREFIX}`"),
    }
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array as _;
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::Field;
    use futures::executor::block_on;
    use krishiv_common::error::ErrorClass;
    use krishiv_types::DataType;

    use crate::session::Session;

    #[test]
    fn work_runs_on_the_given_runtime_and_a_drop_aborts_it() {
        let io = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("test-io")
            .enable_all()
            .build()
            .unwrap();
        let name = futures::executor::block_on(run_on(io.handle(), async {
            Ok(std::thread::current().name().map(str::to_owned))
        }))
        .unwrap();
        assert_eq!(name.as_deref(), Some("test-io"));
        // Dropping the waiting future aborts a task that never finishes.
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let mut waiting = Box::pin(run_on(io.handle(), async move {
            let _held = tx;
            std::future::pending::<Result<(), KrishivError>>().await
        }));
        let waker = futures::task::noop_waker();
        let _ = waiting
            .as_mut()
            .poll(&mut std::task::Context::from_waker(&waker));
        drop(waiting);
        assert_eq!(
            rx.recv_timeout(std::time::Duration::from_secs(5)),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected),
            "the aborted task must drop its sender"
        );
    }

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
            TempTable::Files { columns, .. }
            | TempTable::Memory { columns, .. }
            | TempTable::View { columns, .. } => &columns[0].name,
        }
    }

    #[test]
    fn expanding_a_cycle_is_an_internal_error() {
        // `CREATE TEMPORARY VIEW` rejects recursive views itself; the
        // expansion guard stays for anything else that closes a cycle.
        let session = SessionCatalog::default();
        let view = |name: &str| TempTable::View {
            plan: Arc::new(LogicalPlan::table([name])),
            columns: Vec::new(),
            pins: Vec::new(),
        };
        session.put("a", view("b"));
        session.put("b", view("a"));
        let plan = Arc::new(LogicalPlan::table(["a"]));
        let e = expand_temp_views(&session, &plan, false).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError);
        assert!(e.message().contains("`a`"), "{e}");
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

    /// A session with the memory table `t(id)` holding `values`.
    fn int_session(values: Vec<i32>) -> Session {
        let session = Session::new().unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "id",
            arrow_schema::DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(values))],
        )
        .unwrap();
        session.register_batches("t", schema, vec![batch]).unwrap();
        session
    }

    /// The `id` column of `sql`, ordered.
    fn select_ids(session: &Session, sql: &str) -> Vec<i32> {
        let df = block_on(session.sql(sql)).unwrap();
        let batches = block_on(df.collect()).unwrap();
        let mut out = Vec::new();
        for batch in &batches {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            out.extend((0..batch.num_rows()).map(|i| ids.value(i)));
        }
        out.sort_unstable();
        out
    }

    #[test]
    fn reregistering_a_snapshot_view_does_not_grow_the_catalog() {
        let s = int_session(vec![1]);
        block_on(s.table("t").create_or_replace_temp_view("v")).unwrap();
        // The table, its pin and the view: replacing the view must not add
        // to that.
        assert_eq!(s.state.catalog.entry_count(), 3);
        for _ in 0..100 {
            block_on(s.table("t").create_or_replace_temp_view("v")).unwrap();
        }
        assert_eq!(s.state.catalog.entry_count(), 3);
        assert_eq!(select_ids(&s, "SELECT id FROM v"), vec![1]);
    }

    #[test]
    fn dropping_a_view_removes_the_pins_it_owned() {
        let s = int_session(vec![1]);
        block_on(s.table("t").create_or_replace_temp_view("v")).unwrap();
        assert_eq!(s.state.catalog.entry_count(), 3);
        block_on(s.sql("DROP VIEW v")).unwrap();
        assert_eq!(s.state.catalog.entry_count(), 1);
    }

    #[test]
    fn a_pin_shared_by_two_views_survives_dropping_one() {
        let s = int_session(vec![1]);
        block_on(s.table("t").create_or_replace_temp_view("v1")).unwrap();
        // A view over a view reuses the pin instead of stacking a new one.
        block_on(s.table("v1").create_or_replace_temp_view("v2")).unwrap();
        assert_eq!(s.state.catalog.entry_count(), 4);
        assert!(s.drop_temp_view("v1"));
        assert_eq!(s.state.catalog.entry_count(), 3);
        // The surviving view still reads the snapshot, however `t` changes.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "id",
            arrow_schema::DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![99]))],
        )
        .unwrap();
        s.register_batches("t", schema, vec![batch]).unwrap();
        assert_eq!(select_ids(&s, "SELECT id FROM v2"), vec![1]);
    }

    #[test]
    fn hidden_names_resolve_as_missing_tables() {
        let s = int_session(vec![1]);
        block_on(s.table("t").create_or_replace_temp_view("v")).unwrap();
        // Exactly as a missing table: same class, state and message shape.
        let missing = block_on(s.sql("SELECT * FROM nope")).unwrap_err();
        assert_eq!(missing.error_class(), ErrorClass::TableOrViewNotFound);
        let e = block_on(s.sql("SELECT * FROM `__krishiv_snapshot_0`")).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
        assert_eq!(e.sql_state(), missing.sql_state());
        assert_eq!(
            e.message().replace("`__krishiv_snapshot_0`", "`nope`"),
            missing.message()
        );
        // Any letter case, quoted or not.
        let e = block_on(s.sql("SELECT * FROM __KRISHIV_SNAPSHOT_0")).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
    }

    #[test]
    fn view_built_from_itself_reads_the_old_definition() {
        let s = int_session(vec![1]);
        block_on(s.table("t").create_or_replace_temp_view("t")).unwrap();
        assert_eq!(select_ids(&s, "SELECT id FROM t"), vec![1]);
        // Rebuilding the self-view reuses the pin instead of orphaning one.
        block_on(s.table("t").create_or_replace_temp_view("t")).unwrap();
        assert_eq!(select_ids(&s, "SELECT id FROM t"), vec![1]);
        assert_eq!(s.state.catalog.entry_count(), 2);
    }

    #[test]
    fn hidden_names_are_rejected_for_new_views_and_tables() {
        let s = int_session(vec![1]);
        let e = block_on(
            s.table("t")
                .create_or_replace_temp_view("__krishiv_snapshot_7"),
        )
        .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivReservedTempViewName);
        assert_eq!(e.error_class().name(), "KRISHIV_RESERVED_TEMP_VIEW_NAME");
        assert_eq!(e.sql_state(), Some("42602"));
        assert!(e.message().contains("`__krishiv_snapshot_`"), "{e}");
        let e = block_on(s.sql("CREATE TEMP VIEW `__krishiv_snapshot_7` AS SELECT id FROM t"))
            .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivReservedTempViewName);
        let e =
            block_on(s.sql("CREATE OR REPLACE TEMP VIEW V__KRISHIV_SNAPSHOT_ AS SELECT id FROM t"));
        // A name that merely contains the prefix is fine.
        assert!(e.is_ok(), "{e:?}");
        block_on(s.sql("DROP VIEW V__KRISHIV_SNAPSHOT_")).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "id",
            arrow_schema::DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();
        let e = s
            .register_batches("__KRISHIV_SNAPSHOT_7", schema, vec![batch])
            .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivReservedTempViewName);
    }
}
