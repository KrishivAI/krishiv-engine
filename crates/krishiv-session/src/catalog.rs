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

use crate::store::{CatalogStore, CatalogTable, InMemoryCatalogStore, ViewDef, Warehouse};
use arrow_schema::{Schema, SchemaRef};
use krishiv_common::config::Config;
use krishiv_common::error::{
    InternalError, KrishivError, KrishivReservedTempViewName, KrishivUnsupportedFeature,
    RecursiveView, TableOrViewNotFound,
};
use krishiv_exec::files::FileReads;
use krishiv_exec::physical::{
    PhysNode, PhysicalPlan, ScanId, SplitFileRef, TaskSpec, assign_batch_splits,
    assign_file_splits, plan_file_partitions,
};
use krishiv_exec::planner::{FileTable, MemoryTable, TableProvider, TableProviderResolver};
use krishiv_io::IoContext;
use krishiv_io::split::SplitConfig;
use krishiv_plan::analyzer::{
    BoxFuture, Catalog, CatalogResolver, NameForm, TableColumn, TableIdent, TableKind, TableMeta,
};
use krishiv_plan::{
    ALIAS_PART_SEP, Cte, Expr, ExprNode, LogicalPlan, Origin, PlanNode, ReadSource, Scan,
    TableVersion,
};

use crate::cache::{RelationCache, is_cache_name};

/// The session catalog's name, as Spark names it.
pub(crate) const CATALOG: &str = "spark_catalog";
/// The namespace temporary tables live in.
pub(crate) const NAMESPACE: &str = "default";
/// The namespace global temporary views live in.
pub(crate) const GLOBAL_NAMESPACE: &str = "global_temp";

/// The key a global temporary view `name` is stored under in [`SessionCatalog`]: a prefix no SQL
/// identifier spells (it starts with a control character), so it never collides with a temporary
/// view and [`SessionCatalog::names`] can leave it out.
pub(crate) fn global_key(name: &str) -> String {
    format!("{GLOBAL_PREFIX}{name}")
}

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
        /// The columns are a permanent table's own, not inferred: reading
        /// it lists its files without inferring a schema, and a table with
        /// no files is empty.
        known: bool,
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

/// What every [`global_key`] starts with.
const GLOBAL_PREFIX: &str = "\u{1}global_temp\u{1}";

/// The temporary tables and views of a session, by lower-case name.
#[derive(Debug, Default)]
pub(crate) struct SessionCatalog {
    tables: RwLock<BTreeMap<String, (String, Arc<TempTable>)>>,
    epoch: AtomicU64,
    /// Hidden names handed out for snapshot pins (see [`SessionCatalog::put_view`]).
    hidden: AtomicU64,
    /// The permanent tables, apart from the temporary views above.
    pub(crate) store: InMemoryCatalogStore,
    /// Where managed tables live.
    pub(crate) warehouse: Warehouse,
    /// The namespace bare table names resolve in (`USE`).
    current_namespace: RwLock<String>,
    /// The entries `SET PATH` stored; `None` is the default path.
    path: RwLock<Option<Vec<crate::sql_path::Entry>>>,
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

    /// Adds or replaces `name` with a view over `plan` and `columns` that
    /// owns `pins`, inserting `new_pins` (the hidden entries the view pins)
    /// in the same critical section: no concurrent `put` or `drop_table`
    /// can collect a new pin between its insert and the view's, which would
    /// leave the view reading a missing entry.
    pub(crate) fn put_view(
        &self,
        name: &str,
        plan: Arc<LogicalPlan>,
        columns: Vec<TableColumn>,
        pins: Vec<String>,
        new_pins: Vec<(String, Arc<TempTable>)>,
    ) {
        let mut tables = self.tables.write().unwrap_or_else(PoisonError::into_inner);
        for (hidden, table) in new_pins {
            tables.insert(hidden.clone(), (hidden, table));
        }
        tables.insert(
            name.to_lowercase(),
            (
                name.to_owned(),
                Arc::new(TempTable::View {
                    plan,
                    columns,
                    pins,
                }),
            ),
        );
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

    /// Hands out the next hidden name snapshot pins use: no SQL text
    /// spells it, so snapshot views read it however the session's names
    /// change afterwards. The caller inserts the entry with
    /// [`SessionCatalog::put_view`], which stores the pins and the owning
    /// view atomically.
    fn next_hidden_name(&self) -> String {
        format!(
            "{HIDDEN_PREFIX}{}",
            self.hidden.fetch_add(1, Ordering::AcqRel)
        )
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
            .filter(|n| !n.starts_with(HIDDEN_PREFIX) && !n.starts_with(GLOBAL_PREFIX))
            .collect()
    }

    /// The names of the global temporary views, as registered, in order.
    pub(crate) fn global_names(&self) -> Vec<String> {
        self.tables
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .filter_map(|(n, _)| n.strip_prefix(GLOBAL_PREFIX).map(str::to_owned))
            .collect()
    }

    /// The namespace bare table names resolve in.
    pub(crate) fn current_namespace(&self) -> String {
        let ns = self
            .current_namespace
            .read()
            .unwrap_or_else(PoisonError::into_inner);
        if ns.is_empty() {
            NAMESPACE.to_owned()
        } else {
            ns.clone()
        }
    }

    /// The stored path entries, `None` for the default.
    pub(crate) fn stored_path(&self) -> Option<Vec<crate::sql_path::Entry>> {
        self.path
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// `SET PATH`: stores the expanded entries.
    pub(crate) fn set_path(&self, entries: Vec<crate::sql_path::Entry>) {
        *self.path.write().unwrap_or_else(PoisonError::into_inner) = Some(entries);
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// The SQL path as parts, `CURRENT_SCHEMA` resolved; `default_conf` is
    /// `krishiv.sql.defaultPath`, read when no `SET PATH` ran.
    pub(crate) fn sql_path(&self, default_conf: &str) -> Vec<Vec<String>> {
        let entries = self
            .stored_path()
            .unwrap_or_else(|| crate::sql_path::default_entries(default_conf));
        let namespace = self.current_namespace();
        entries.iter().map(|e| e.parts(&namespace)).collect()
    }

    /// `USE namespace`.
    pub(crate) fn use_namespace(&self, namespace: &str) {
        namespace.clone_into(
            &mut self
                .current_namespace
                .write()
                .unwrap_or_else(PoisonError::into_inner),
        );
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    /// What a cached analysis depends on: temporary views and permanent
    /// tables both bump it.
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire) + self.store.epoch()
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
    /// A permanent table, by `namespace.table`.
    Table(String),
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
    cache: Arc<RelationCache>,
    /// The cached snapshots this query's plan was rewritten to scan, by
    /// hidden name: held so a concurrent `CLEAR CACHE` cannot pull them out
    /// from under the query between substitution and planning.
    pinned: Mutex<HashMap<String, Arc<MemoryTable>>>,
    files: Arc<FileReads>,
    scans: FileScans,
    /// Entries loaded by this query, by lower-case name.
    loaded: Mutex<HashMap<String, Arc<TempTable>>>,
    /// The file tables of the query's scans, opened before planning.
    opened: Mutex<Vec<(FilesKey, Arc<FileTable>)>>,
    epoch: u64,
    /// The current namespace (`USE`) as this query sees it.
    namespace: Vec<String>,
    /// The SQL path (`SET PATH`) as this query sees it.
    path: Vec<Vec<String>>,
}

impl QueryCatalog {
    /// The query's SQL path snapshot.
    pub(crate) fn sql_path_snapshot(&self) -> Vec<Vec<String>> {
        self.path.clone()
    }

    /// The query's namespace snapshot.
    pub(crate) fn namespace(&self) -> &[String] {
        &self.namespace
    }
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
        cache: Arc<RelationCache>,
        files: Arc<FileReads>,
        scans: FileScans,
        default_path: &str,
    ) -> Self {
        let epoch = session.epoch();
        let namespace = vec![session.current_namespace()];
        let path = session.sql_path(default_path);
        Self {
            session,
            cache,
            pinned: Mutex::new(HashMap::new()),
            files,
            scans,
            loaded: Mutex::new(HashMap::new()),
            opened: Mutex::new(Vec::new()),
            epoch,
            namespace,
            path,
        }
    }

    /// Keeps the snapshot `table` for this query's scans of the hidden `name`.
    pub(crate) fn pin_cached(&self, name: String, table: Arc<MemoryTable>) {
        self.pinned
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(name, table);
    }

    /// The entry a scan of `name` (as the analyzer kept it) reads: a
    /// temporary view or table by its single part, a permanent table by
    /// `namespace.table`.
    fn loaded_for(&self, name: &[String]) -> Option<Arc<TempTable>> {
        self.entry_of(name).map(|(_, t)| t)
    }

    /// The key and entry a scan of `name` reads: `namespace.table` for a
    /// permanent table; else the bare name of a temporary entry (a temporary
    /// view or table is found by its bare name however the scan spells it).
    fn entry_of(&self, name: &[String]) -> Option<(String, Arc<TempTable>)> {
        let last = name.last()?.to_lowercase();
        [entry_key(name)?, last]
            .into_iter()
            .find_map(|key| self.loaded(&key).map(|t| (key, t)))
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
            let listing = match &key {
                FilesKey::Table(_) => {
                    self.files
                        .listing_known(format, paths, options, Arc::clone(&table))
                        .await?
                }
                _ => self.files.listing(format, paths, options).await?,
            };
            let opened = open_table(
                Arc::clone(&listing.format),
                self.scans.clone(),
                table,
                listing.files.to_vec(),
                listing.options.clone(),
            )
            .await?
            .with_partition_columns(listing.partition_columns.to_vec());
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
                        let (key, table) = self.entry_of(name)?;
                        let TempTable::Files { source, known, .. } = &*table else {
                            return None;
                        };
                        let key = if *known {
                            FilesKey::Table(key)
                        } else {
                            FilesKey::View(key)
                        };
                        (key, source.clone())
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

    /// The scan splits of one stage over `plan`: one [`TaskSpec`] per task
    /// of `tasks`, dealing the streaming pipeline's scans (or relation
    /// batches) in contiguous index ranges, deterministically.
    ///
    /// Only the scans of the probe or streaming pipeline are split: the
    /// build side of any join and the plans of scalar and `EXISTS`
    /// subqueries read whole (no entry), so a task joins its probe share
    /// against the full build and evaluates subqueries over full inputs.
    /// Splitting those would join `t_i` against only its build share or
    /// evaluate `(SELECT max(x) FROM u)` over part of `u`.
    ///
    /// The files are the opened listing (listed once, before planning);
    /// their partitions are planned here with `total_slots` worker slots,
    /// so the same listing and config give every task the same splits, and
    /// a retried task reads the same ones. A split carries a byte range,
    /// never row-group indices: the coordinator reads no footers. A scan
    /// the stage reads whole (a table without a relation lease) gets no
    /// entry.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for no tasks, a task count past `u32`, or a scan
    /// whose files were not opened before planning.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the distributed scheduler is the caller; unit tests cover it until then"
        )
    )]
    pub(crate) fn task_specs(
        &self,
        plan: &PhysicalPlan,
        tasks: usize,
        total_slots: u64,
    ) -> Result<Vec<TaskSpec>, KrishivError> {
        if tasks == 0 {
            return Err(InternalError {
                message: "a stage without tasks".to_owned(),
            }
            .into());
        }
        let partitions = u32::try_from(tasks).map_err(|_| InternalError {
            message: format!("a stage of {tasks} tasks"),
        })?;
        // The streaming pipeline's scans, in id order: the assignment is
        // deterministic. Builds and subquery plans read whole.
        let (mut scans, streams_range) = streaming_leaves(plan.root());
        // A `Range` source slices its rows over the task's own drivers and
        // has no split assignment yet: several tasks would each emit the
        // whole range. Until slices are assigned across tasks (a follow-up
        // of the distributed scheduler), such a stage is a single task.
        if streams_range && tasks > 1 {
            return Err(unsupported("a range source across several tasks"));
        }
        // Each task would run the shared plan again and read all its rows,
        // until shared plans lower to exchanges.
        if !plan.shared().is_empty() && tasks > 1 {
            return Err(unsupported("a shared plan across several tasks"));
        }
        scans.sort_by_key(|s| s.id);
        let mut assigned: Vec<(ScanId, ScanSplitsKind)> = Vec::new();
        for scan in scans {
            match &scan.scan.source {
                ReadSource::Table { name, .. } => {
                    let last = name.last().map_or("", String::as_str);
                    let Some(table) = self.loaded(last) else {
                        return Err(InternalError {
                            message: format!("the scanned table {last} was not loaded"),
                        }
                        .into());
                    };
                    match &*table {
                        TempTable::Memory { table, .. } => {
                            if let Some(lease) = table.relation_lease() {
                                assigned.push((
                                    scan.id,
                                    ScanSplitsKind::Batches {
                                        relation: lease.reference(),
                                        batches: lease.batches(),
                                    },
                                ));
                            }
                        }
                        TempTable::Files { .. } => {
                            let key = FilesKey::View(last.to_lowercase());
                            let table = self.opened(&key).ok_or_else(|| InternalError {
                                message: "a scan of files that was not opened before planning"
                                    .to_owned(),
                            })?;
                            assigned.push((scan.id, self.file_splits(&table, total_slots)?));
                        }
                        TempTable::View { .. } => {
                            return Err(InternalError {
                                message: "a scan of a temporary view that was not inlined"
                                    .to_owned(),
                            }
                            .into());
                        }
                    }
                }
                source @ ReadSource::Files { .. } => {
                    let key = FilesKey::Read(source.clone());
                    let table = self.opened(&key).ok_or_else(|| InternalError {
                        message: "a scan of files that was not opened before planning".to_owned(),
                    })?;
                    assigned.push((scan.id, self.file_splits(&table, total_slots)?));
                }
                _ => {
                    return Err(unsupported("this scan"));
                }
            }
        }
        // `partitions` already holds `tasks` as a `u32`: iterating it
        // cannot fail the conversion the `tasks` range would repeat.
        let mut specs: Vec<TaskSpec> = (0..partitions)
            .map(|partition| TaskSpec {
                stage: 0,
                stage_attempt: 0,
                partition,
                partitions,
                attempt: 0,
                scans: BTreeMap::new(),
            })
            .collect();
        for (id, kind) in assigned {
            match kind {
                ScanSplitsKind::Files { files, partitions } => {
                    for (splits, spec) in assign_file_splits(&files, &partitions, tasks)
                        .into_iter()
                        .zip(specs.iter_mut())
                    {
                        spec.scans.insert(id, splits);
                    }
                }
                ScanSplitsKind::Batches { relation, batches } => {
                    for (splits, spec) in assign_batch_splits(relation, batches, tasks)
                        .into_iter()
                        .zip(specs.iter_mut())
                    {
                        spec.scans.insert(id, splits);
                    }
                }
            }
        }
        Ok(specs)
    }

    /// The file splits of an opened file table: the listed files with
    /// their versions, planned with `total_slots` worker slots.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the distributed scheduler is the caller; unit tests cover it until then"
        )
    )]
    fn file_splits(
        &self,
        table: &FileTable,
        total_slots: u64,
    ) -> Result<ScanSplitsKind, KrishivError> {
        let config = SplitConfig::from_config(&self.scans.config, total_slots);
        let format = table.format();
        let options = table.options();
        let files: Vec<SplitFileRef> = table
            .files()
            .iter()
            .map(SplitFileRef::from_file_ref)
            .collect();
        let partitions = plan_file_partitions(&config, &files, |f| {
            format.capabilities(&options).is_splittable(f.path.as_ref())
        });
        Ok(ScanSplitsKind::Files { files, partitions })
    }
}

/// What [`QueryCatalog::task_specs`] deals to a stage's tasks: a file
/// scan's planned partitions, or a memory scan's batch count.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the distributed scheduler is the caller; unit tests cover it until then"
    )
)]
enum ScanSplitsKind {
    /// The listed files with their versions, and their planned partitions.
    Files {
        /// The scan's files, in listing order.
        files: Vec<SplitFileRef>,
        /// Their partitions.
        partitions: Vec<krishiv_io::split::FilePartition>,
    },
    /// A relation's content hash and batch count.
    Batches {
        /// The relation, by content hash.
        relation: krishiv_memory::RelationRef,
        /// The number of stored batches.
        batches: usize,
    },
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

/// Whether the analyzed `plan` reads the permanent table `namespace.name` (directly, or through a view or a
/// subquery): `INSERT OVERWRITE` of a table that is also read would delete what it reads.
pub(crate) fn reads_stored_table(
    plan: &LogicalPlan,
    namespace: &str,
    name: &str,
    current: &str,
) -> bool {
    let mut scans = Vec::new();
    collect_scans(plan, &mut scans);
    scans.iter().any(|scan| {
        let ReadSource::Table { name: parts, .. } = &scan.source else {
            return false;
        };
        let (ns, table) = match parts.as_slice() {
            [table] => (current, table),
            [.., ns, table] => (ns.as_str(), table),
            [] => return false,
        };
        ns.eq_ignore_ascii_case(namespace) && table.eq_ignore_ascii_case(name)
    })
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

/// The scans of the streaming pipeline of `root`: those a task splits.
/// Joins contribute only their probe side and nodes only their streaming
/// input; subquery plans and join builds read whole (no entry), so tasks
/// join probe shares against full builds and evaluate subqueries over full
/// inputs.
fn streaming_leaves(root: &PhysNode) -> (Vec<&krishiv_exec::physical::ScanNode>, bool) {
    #[derive(Default)]
    struct Leaves<'a> {
        scans: Vec<&'a krishiv_exec::physical::ScanNode>,
        range: bool,
    }
    fn go<'a>(node: &'a PhysNode, out: &mut Leaves<'a>) {
        match node {
            PhysNode::Scan(scan) => out.scans.push(scan),
            PhysNode::OneRow | PhysNode::SharedRead { .. } => {}
            PhysNode::Range(_) => out.range = true,
            PhysNode::Filter { input, .. }
            | PhysNode::Project { input, .. }
            | PhysNode::ReplicateRows { input, .. }
            | PhysNode::Generate { input, .. }
            | PhysNode::Expand { input, .. }
            | PhysNode::Limit { input, .. }
            | PhysNode::Sort { input, .. }
            | PhysNode::TopN { input, .. }
            | PhysNode::Window { input, .. }
            | PhysNode::WindowGroupLimit { input, .. }
            | PhysNode::Aggregate { input, .. }
            | PhysNode::WriteFiles { input, .. } => go(input, out),
            PhysNode::Exchange(e) => go(&e.input, out),
            PhysNode::HashJoin(join) => go(&join.probe, out),
            PhysNode::NestedLoopJoin(join) => go(&join.probe, out),
            PhysNode::Union { inputs } => {
                for input in inputs {
                    go(input, out);
                }
            }
        }
    }
    let mut out = Leaves::default();
    go(root, &mut out);
    (out.scans, out.range)
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

/// The key a query loads an entry under, from a scan's name: a single
/// part is a temporary entry (or a permanent table found by its bare name);
/// `namespace.table` (a leading catalog part dropped) a permanent table.
fn entry_key(name: &[String]) -> Option<String> {
    match name {
        [] => None,
        [one] => Some(one.to_lowercase()),
        [.., namespace, table] => Some(format!(
            "{}.{}",
            namespace.to_lowercase(),
            table.to_lowercase()
        )),
    }
}

impl QueryCatalog {
    /// The permanent table `id` as a read of its files with the columns it
    /// was created with (analysis infers nothing and reads no files), or
    /// `None` when the store has no such table. The table is kept for the
    /// query under `namespace.table`, and under its bare name when the
    /// query wrote it bare.
    fn load_stored(&self, id: &TableIdent, form: NameForm) -> Option<TableMeta> {
        let namespace = id.namespace.first()?;
        let (namespace, name) = if form.case_sensitive {
            (namespace.clone(), id.name.clone())
        } else {
            (namespace.to_lowercase(), id.name.to_lowercase())
        };
        let table = self.session.store.table(&namespace, &name)?;
        if table.view.is_some() {
            return None; // views are expanded before analysis
        }
        let columns: Vec<TableColumn> = table
            .columns
            .iter()
            .map(|c| TableColumn {
                name: c.name.clone(),
                data_type: c.data_type.clone(),
                nullable: c.nullable,
            })
            .collect();
        let entry = Arc::new(TempTable::Files {
            source: table_source(&table),
            columns: columns.clone(),
            known: true,
        });
        let mut loaded = self.loaded.lock().unwrap_or_else(PoisonError::into_inner);
        // Keys are lower case whatever the session's case sensitivity: `entry_key` spells scans that way.
        loaded.insert(
            format!("{}.{}", namespace.to_lowercase(), name.to_lowercase()),
            Arc::clone(&entry),
        );
        if !form.qualified {
            loaded.insert(name.to_lowercase(), entry);
        }
        Some(TableMeta {
            ident: TableIdent {
                catalog: CATALOG.to_owned(),
                namespace: vec![namespace],
                name,
            },
            columns,
            version: None,
            kind: TableKind::Table,
        })
    }
}

/// The read of a table's files.
pub(crate) fn table_source(table: &crate::store::CatalogTable) -> ReadSource {
    ReadSource::Files {
        format: table.provider.clone(),
        paths: vec![krishiv_common::secret::OptionValue::from_user(
            "path",
            table.location.display().to_string(),
        )],
        options: table
            .options
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    krishiv_common::secret::OptionValue::from_user(k, v),
                )
            })
            .collect(),
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
                return Ok(self.load_stored(id, form));
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
        &self.0.namespace
    }

    fn sql_path(&self) -> Vec<Vec<String>> {
        self.0.path.clone()
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
                let Some(table) = self.loaded_for(name) else {
                    // ...except a cached snapshot substituted after
                    // analysis, which reads the relation cache instead.
                    let pinned = self
                        .pinned
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .get(last)
                        .cloned();
                    if let Some(cached) = pinned.or_else(|| self.cache.table(last)) {
                        return Ok(cached as Arc<dyn TableProvider>);
                    }
                    return Err(InternalError {
                        message: format!("the scanned table {last} was not loaded"),
                    }
                    .into());
                };
                match &*table {
                    TempTable::Memory { table, .. } => {
                        Ok(Arc::clone(table) as Arc<dyn TableProvider>)
                    }
                    TempTable::Files { known, .. } => {
                        let key = self
                            .entry_of(name)
                            .map_or_else(|| last.to_lowercase(), |(k, _)| k);
                        self.files_provider(&if *known {
                            FilesKey::Table(key)
                        } else {
                            FilesKey::View(key)
                        })
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
#[cfg(test)]
pub(crate) fn expand_temp_views(
    catalog: &SessionCatalog,
    plan: &Arc<LogicalPlan>,
    case_sensitive: bool,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    let expansion = expand_views(catalog, plan, case_sensitive, None)?;
    match expansion.deferred {
        Some(e) => Err(e),
        None => Ok(expansion.plan),
    }
}

/// What expanding a permanent view needs: its SQL is parsed again with the session's parser options.
pub(crate) struct StoredViews<'a> {
    pub(crate) parser: krishiv_sql::ParserOptions,
    pub(crate) next_source: &'a dyn Fn() -> krishiv_plan::SourceId,
    /// The current namespace as the query snapshot saw it: bare names resolve here, whatever `USE` does later.
    pub(crate) namespace: String,
    /// The SQL path as the query snapshot saw it: a bare name belongs to the first of its schemas that has it.
    pub(crate) path: Vec<Vec<String>>,
    /// `spark.sql.legacy.viewSchemaBindingMode` of the query's snapshot: views read by their schema binding mode.
    pub(crate) binding_enabled: bool,
}

/// [`expand_temp_views`], and with `stored` also the permanent views of the catalog store: a read of one becomes
/// its query as a subquery, with its bare table names resolved in the namespace the view was created in.
///
/// # Errors
/// As [`expand_temp_views`]; `INTERNAL_ERROR` for a stored view whose text no longer parses as a query.
pub(crate) fn expand_views(
    catalog: &SessionCatalog,
    plan: &Arc<LogicalPlan>,
    case_sensitive: bool,
    stored: Option<&StoredViews<'_>>,
) -> Result<Expansion, KrishivError> {
    expand_views_in(catalog, plan, case_sensitive, stored, None, None)
}

/// [`expand_views`] for `plan` written inside `namespace`: its bare table names belong to that namespace, never
/// to a temporary view (the body of a stored function), resolved through `view_path` when it has one.
pub(crate) fn expand_views_in(
    catalog: &SessionCatalog,
    plan: &Arc<LogicalPlan>,
    case_sensitive: bool,
    stored: Option<&StoredViews<'_>>,
    namespace: Option<String>,
    view_path: Option<Vec<Vec<String>>>,
) -> Result<Expansion, KrishivError> {
    let mut expander = Expander {
        catalog,
        case_sensitive,
        path: Vec::new(),
        stored,
        view_namespace: namespace,
        view_path,
        reads: Vec::new(),
        deferred: None,
        stored_reads: Vec::new(),
    };
    let plan = expander.go(Arc::clone(plan), &mut Vec::new())?;
    Ok(Expansion {
        plan,
        reads: expander.reads,
        deferred: expander.deferred,
        stored_reads: expander.stored_reads,
    })
}

/// A table-like name the expansion looked up in the catalog store, in plan order. The analyzer never
/// sees an inlined permanent view, so the session asks the policy about every candidate itself, as the
/// analyzer asks about a table: the name as written and folded first (whether or not it exists), then,
/// for a permanent view, the view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReadCandidate {
    /// The name parts as the plan spells them.
    pub(crate) written: Vec<String>,
    /// The namespace and name the catalog stores, when the name is a permanent view.
    pub(crate) view: Option<(String, String)>,
}

/// What [`expand_views`] found besides the expanded plan.
pub(crate) struct Expansion {
    pub(crate) plan: Arc<LogicalPlan>,
    /// The catalog-store lookups, in plan order, without repeats.
    pub(crate) reads: Vec<ReadCandidate>,
    /// A failure inside a stored view's text (it no longer parses, reads itself, ...). It is raised only
    /// after the reads are authorized, so a principal who may not read the view never sees its text's errors.
    pub(crate) deferred: Option<KrishivError>,
    /// The stored views inlined, inner views before the views that read them: their schema binding is checked
    /// against their analyzed queries (`view_schema::check_read`).
    pub(crate) stored_reads: Vec<crate::view_schema::StoredRead>,
}

/// [`expand_temp_views`]: the views on the current expansion path, by
/// lower-case name, for cycle detection.
struct Expander<'a> {
    catalog: &'a SessionCatalog,
    case_sensitive: bool,
    path: Vec<String>,
    stored: Option<&'a StoredViews<'a>>,
    /// Inside a stored view: the namespace its bare table names belong to.
    view_namespace: Option<String>,
    /// Inside a stored view: the SQL path frozen with it.
    view_path: Option<Vec<Vec<String>>>,
    /// The catalog-store lookups so far, in plan order, without repeats.
    reads: Vec<ReadCandidate>,
    /// The first failure inside a stored view's text, kept for [`Expansion::deferred`].
    deferred: Option<KrishivError>,
    /// The stored views inlined so far, for [`Expansion::stored_reads`].
    stored_reads: Vec<crate::view_schema::StoredRead>,
}

impl Expander<'_> {
    /// `plan` with its views inlined; `scope` holds the `WITH`
    /// definitions visible where `plan` reads.
    fn go(
        &mut self,
        plan: Arc<LogicalPlan>,
        scope: &mut Vec<String>,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        // Snapshot pins and cached snapshots are unreachable from user
        // input however a read spells them: a read of one behaves exactly
        // as a missing table, with or without read options. (The pins
        // snapshot views read are scans, which pass through untouched
        // below.)
        if let PlanNode::Read(ReadSource::Table { name, .. }) = &plan.node
            && let [view] = name.as_slice()
            && !scope
                .iter()
                .any(|c| same_name(c, view, self.case_sensitive))
            && (is_hidden_name(view) || is_cache_name(view))
        {
            return Err(TableOrViewNotFound {
                relation_name: quote(view),
                search_path: search_path(),
            }
            .into());
        }
        // Inside a stored view a bare table name is the view's own namespace's, never a temporary view.
        if let Some(namespace) = &self.view_namespace
            && let PlanNode::Read(ReadSource::Table { name, options }) = &plan.node
            && let [bare] = name.as_slice()
            && !scope
                .iter()
                .any(|c| same_name(c, bare, self.case_sensitive))
        {
            // The first schema of the view's frozen path holding the name. A name no schema of a
            // non-empty frozen path holds stays missing (the first schema reports it), never
            // read from a namespace the path left out; a view without a catalog schema in its
            // path reads its own namespace.
            let namespace = self
                .view_path
                .as_deref()
                .and_then(|path| {
                    self.path_schema_of(path, bare).or_else(|| {
                        path.iter().find_map(|entry| match entry.as_slice() {
                            [c, n] if c.eq_ignore_ascii_case(CATALOG) => Some(n.clone()),
                            _ => None,
                        })
                    })
                })
                .unwrap_or_else(|| namespace.clone());
            let qualified = ReadSource::Table {
                name: vec![namespace, bare.clone()],
                options: options.clone(),
            };
            let plan =
                Arc::new(LogicalPlan::new(PlanNode::Read(qualified)).with_origin(plan.origin));
            return self.expand_stored(plan, scope);
        }
        if let PlanNode::Read(ReadSource::Table { name, options }) = &plan.node
            && options.is_empty()
        {
            // A bare name (not shadowed by a `WITH` definition) is a temporary view; `global_temp.name`
            // is a global one.
            let found = match name.as_slice() {
                [view]
                    if !scope
                        .iter()
                        .any(|c| same_name(c, view, self.case_sensitive)) =>
                {
                    // (Hidden names are rejected above, before this view lookup.)
                    Some((view.as_str(), self.catalog.get(view, self.case_sensitive)))
                }
                [namespace, view] if namespace.eq_ignore_ascii_case(GLOBAL_NAMESPACE) => Some((
                    view.as_str(),
                    self.catalog
                        .get(&global_key(view), self.case_sensitive)
                        .filter(|_| !is_hidden_name(view) && !is_cache_name(view)),
                )),
                _ => None,
            };
            if let Some((view, Some(entry))) = found
                && let TempTable::View {
                    plan: definition,
                    columns,
                    ..
                } = &*entry
            {
                let key = name
                    .iter()
                    .map(|p| p.to_lowercase())
                    .collect::<Vec<_>>()
                    .join(".");
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
                        // A global view's qualifier is `global_temp.v`, two parts (see
                        // `ALIAS_PART_SEP`).
                        alias: if name.len() == 2 {
                            format!("{GLOBAL_NAMESPACE}{ALIAS_PART_SEP}{view}")
                        } else {
                            view.to_owned()
                        },
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
        let plan = self.children_and_exprs(plan, scope)?;
        self.expand_stored(plan, scope)
    }

    /// The first schema of `path` (in the catalog store) holding the table or view `table`.
    fn path_schema_of(&self, path: &[Vec<String>], table: &str) -> Option<String> {
        path.iter().find_map(|entry| match entry.as_slice() {
            [c, n] if c.eq_ignore_ascii_case(CATALOG) => {
                let (n_key, t_key) = if self.case_sensitive {
                    (n.clone(), table.to_owned())
                } else {
                    (n.to_lowercase(), table.to_lowercase())
                };
                self.catalog.store.table(&n_key, &t_key).map(|_| n.clone())
            }
            _ => None,
        })
    }

    /// `plan`, or when it reads a stored view, that view's query as a subquery named for the view.
    fn expand_stored(
        &mut self,
        plan: Arc<LogicalPlan>,
        scope: &mut [String],
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        let Some(stored) = self.stored else {
            return Ok(plan);
        };
        let PlanNode::Read(ReadSource::Table { name, options }) = &plan.node else {
            return Ok(plan);
        };
        if !options.is_empty()
            || scope
                .iter()
                .any(|c| name.len() == 1 && same_name(c, &name[0], self.case_sensitive))
        {
            return Ok(plan);
        }
        // A bare name that is a temporary table or view never reaches a permanent view (Spark's resolution order).
        if self.view_namespace.is_none()
            && name.len() == 1
            && self.catalog.exists(&name[0], self.case_sensitive)
        {
            return Ok(plan);
        }
        let (namespace, table) = match name.as_slice() {
            [table] => {
                // Outside a stored view, the first schema of the path holding the name; inside one,
                // the view's own namespace.
                let path = self.view_path.as_deref().unwrap_or(&stored.path);
                let from_path = (self.view_namespace.is_none() || self.view_path.is_some())
                    .then(|| self.path_schema_of(path, table))
                    .flatten();
                match (from_path, &self.view_namespace) {
                    (Some(namespace), _) => (namespace, table),
                    (None, Some(_)) => (stored.namespace.clone(), table),
                    // No schema of the path holds the name: no view to inline.
                    (None, None) => return Ok(plan),
                }
            }
            [namespace, table] => (namespace.clone(), table),
            [catalog, namespace, table] if catalog.eq_ignore_ascii_case(CATALOG) => {
                (namespace.clone(), table)
            }
            _ => return Ok(plan),
        };
        let (namespace, table_key) = if self.case_sensitive {
            (namespace, table.clone())
        } else {
            (namespace.to_lowercase(), table.to_lowercase())
        };
        let entry = self.catalog.store.table(&namespace, &table_key);
        let candidate = ReadCandidate {
            written: name.clone(),
            view: entry
                .as_ref()
                .filter(|e| e.view.is_some())
                .map(|e| (e.namespace.clone(), e.name.clone())),
        };
        if !self.reads.contains(&candidate) {
            self.reads.push(candidate);
        }
        let Some(entry) = entry else {
            return Ok(plan);
        };
        let Some(view) = &entry.view else {
            return Ok(plan);
        };
        match self.inline_stored(&plan, &entry, view, &namespace, &table_key, scope) {
            Ok(done) => Ok(done),
            Err(e) => {
                // Keep the read as it is; the failure is raised once the reads are authorized.
                self.deferred.get_or_insert(e);
                Ok(plan)
            }
        }
    }

    /// The query of the stored view `entry`, inlined for the read `plan`.
    fn inline_stored(
        &mut self,
        plan: &Arc<LogicalPlan>,
        entry: &Arc<CatalogTable>,
        view: &ViewDef,
        namespace: &str,
        table_key: &str,
        _scope: &mut [String],
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        let Some(stored) = self.stored else {
            return Ok(Arc::clone(plan));
        };
        let table = match &plan.node {
            PlanNode::Read(ReadSource::Table { name, .. }) => {
                name.last().cloned().unwrap_or_default()
            }
            _ => return Ok(Arc::clone(plan)),
        };
        let key = format!("{namespace}.{table_key}");
        if self.path.contains(&key) {
            return Err(InternalError {
                message: format!("view `{key}` reads itself"),
            }
            .into());
        }
        let statement =
            krishiv_sql::parse_statement(&view.sql, &stored.parser).map_err(|e| e.error)?;
        let krishiv_sql::ast::Statement::Query(query) = statement else {
            return Err(InternalError {
                message: format!("the text of view `{key}` is not a query"),
            }
            .into());
        };
        let lowered = krishiv_sql::lower(&query, (stored.next_source)()).map_err(|e| e.error)?;
        self.path.push(key);
        let previous = self.view_namespace.replace(view.namespace.clone());
        let previous_path = self.view_path.replace(view.path.clone());
        let input = self.go(Arc::clone(lowered.root()), &mut Vec::new());
        self.view_namespace = previous;
        self.view_path = previous_path;
        self.path.pop();
        let mut input = input?;
        let mode = crate::view_schema::effective_mode(view.schema_mode, stored.binding_enabled);
        self.stored_reads.push(crate::view_schema::StoredRead {
            entry: Arc::clone(entry),
            query: Arc::clone(&input),
            mode,
        });
        let projection = crate::view_schema::projection(mode);
        let mut columns: Vec<String> = entry.columns.iter().map(|c| c.name.clone()).collect();
        let unique = view
            .query_columns
            .iter()
            .map(|c| c.to_lowercase())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == view.query_columns.len();
        if projection == crate::view_schema::ReadProjection::Output {
            // `EVOLUTION`: the view is its query's output, whatever the stored schema says.
            columns.clear();
        } else if unique && view.query_columns.len() == columns.len() {
            // The stored columns, by their query names: a base table that gained columns since does not change
            // what the view returns. In their stored types (the binding check ran on the analyzed query), or in
            // the query's types under `TYPE EVOLUTION`.
            let exprs = view
                .query_columns
                .iter()
                .zip(&entry.columns)
                .map(|(q, c)| {
                    // The name is one part: `col(q)` would split a dot or read a trailing `*`.
                    let stored = crate::column::Column::new(Expr::new(ExprNode::Column(
                        krishiv_plan::ColumnRef {
                            name: vec![q.clone()],
                            plan_id: None,
                        },
                    )));
                    let typed = if projection == crate::view_schema::ReadProjection::Cast {
                        stored.cast_ansi(c.data_type.clone())
                    } else {
                        stored
                    };
                    Arc::new(typed.alias(&c.name).expr)
                })
                .collect();
            input = Arc::new(
                LogicalPlan::new(PlanNode::Project { input, exprs }).with_origin(plan.origin),
            );
        }
        // Otherwise (a query that repeats a column name, or a column list of another length) the columns are only
        // renamed by position: a repeated name cannot be cast, since `ColumnRef` has no ordinal. The binding check
        // matches by name and ordinal as Spark does, so a type change under such a view is not cast here.
        Ok(Arc::new(
            LogicalPlan::new(PlanNode::SubqueryAlias {
                input,
                alias: table,
                columns,
            })
            .with_origin(plan.origin),
        ))
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

/// The first temporary view or table that `plan` reads by its bare name (outside `WITH` definitions that shadow it),
/// for the check that a permanent view reads none (Spark's `INVALID_TEMP_OBJ_REFERENCE`).
pub(crate) fn reads_temporary(
    catalog: &SessionCatalog,
    plan: &Arc<LogicalPlan>,
    case_sensitive: bool,
) -> Option<String> {
    fn go(
        catalog: &SessionCatalog,
        plan: &Arc<LogicalPlan>,
        cs: bool,
        scope: &mut Vec<String>,
    ) -> Option<String> {
        if let PlanNode::Read(ReadSource::Table { name, .. }) = &plan.node
            && let [bare] = name.as_slice()
            && !scope.iter().any(|c| same_name(c, bare, cs))
        {
            return catalog.exists(bare, cs).then(|| bare.clone());
        }
        if let PlanNode::Read(ReadSource::Table { name, .. }) = &plan.node
            && let [namespace, view] = name.as_slice()
            && namespace.eq_ignore_ascii_case(GLOBAL_NAMESPACE)
        {
            return catalog
                .exists(&global_key(view), cs)
                .then(|| format!("{GLOBAL_NAMESPACE}`.`{view}"));
        }
        if let PlanNode::WithCte { input, ctes } = &plan.node {
            let outer = scope.len();
            for cte in ctes.iter() {
                if let Some(found) = go(catalog, &cte.plan, cs, scope) {
                    return Some(found);
                }
                scope.push(cte.name.clone());
            }
            let found = go(catalog, input, cs, scope);
            scope.truncate(outer);
            return found;
        }
        let mut children = Vec::new();
        plan.for_each_child(&mut |child| children.push(Arc::clone(child)));
        if let Some(found) = children.iter().find_map(|c| go(catalog, c, cs, scope)) {
            return Some(found);
        }
        let mut exprs = Vec::new();
        plan.for_each_expr(&mut |expr| exprs.push(Arc::clone(expr)));
        exprs.iter().find_map(|e| in_expr(catalog, e, cs, scope))
    }
    fn in_expr(
        catalog: &SessionCatalog,
        expr: &Arc<Expr>,
        cs: bool,
        scope: &mut Vec<String>,
    ) -> Option<String> {
        let mut children = Vec::new();
        expr.for_each_child(&mut |child| children.push(Arc::clone(child)));
        if let Some(found) = children.iter().find_map(|c| in_expr(catalog, c, cs, scope)) {
            return Some(found);
        }
        if let ExprNode::Subquery { plan, .. } = &expr.node {
            return go(catalog, plan, cs, scope);
        }
        None
    }
    go(catalog, plan, case_sensitive, &mut Vec::new())
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
            && options.is_empty()
            && let Some(view) = match name.as_slice() {
                [view]
                    if !self
                        .scope
                        .iter()
                        .any(|c| same_name(c, view, self.case_sensitive)) =>
                {
                    Some(view.clone())
                }
                [namespace, view] if namespace.eq_ignore_ascii_case(GLOBAL_NAMESPACE) => {
                    Some(global_key(view))
                }
                _ => None,
            }
        {
            if same_name(&view, self.target, self.case_sensitive) {
                if self.catalog.get(self.target, self.case_sensitive).is_some() {
                    return Err(recursive_view(self.target, &self.stack));
                }
            } else if let Some(entry) = self.catalog.get(&view, self.case_sensitive)
                && let TempTable::View {
                    plan: definition, ..
                } = &*entry
            {
                let key = view.to_lowercase();
                if self.stack.iter().any(|v| v.to_lowercase() == key) {
                    return Err(recursive_view(self.target, &self.stack));
                }
                self.stack.push(view);
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

/// A view's catalog key as Spark names it in errors (`` `global_temp`.`v` `` for a global view).
fn shown_view(key: &str) -> String {
    match key.strip_prefix(GLOBAL_PREFIX) {
        Some(view) => format!("{}.{}", quote(GLOBAL_NAMESPACE), quote(view)),
        None => quote(key),
    }
}

/// `RECURSIVE_VIEW` for a cycle through `stack` back to `target`.
fn recursive_view(target: &str, stack: &[String]) -> KrishivError {
    let mut path: Vec<String> = stack.iter().map(|v| shown_view(v)).collect();
    path.push(shown_view(target));
    RecursiveView {
        view_ident: shown_view(target),
        new_path: path.join(" -> "),
    }
    .into()
}

/// The pinned plan, the hidden names it reads (the new view owns them,
/// see [`TempTable::View`]) and the hidden entries to store with it.
type PinnedPlan = (Arc<LogicalPlan>, Vec<String>, Vec<(String, Arc<TempTable>)>);

/// Snapshots the temporary tables the analyzed `plan` reads: pins each
/// memory or file-backed entry under a hidden name and rewrites its scans
/// to it, so the plan stays bound to the tables it was analyzed against
/// however the session's names change afterwards (a view built from itself
/// reads the old definition; a view over a file-backed view survives its
/// replacement or removal). Reads by path are self-contained and need no
/// pinning.
///
/// A scan that already reads a pin (inlined from another view) reuses it,
/// so views over views share the pin instead of stacking a new one per
/// view. Returns the pinned plan, the hidden names it reads (the new view
/// owns them, see [`TempTable::View`]) and the hidden entries to store
/// with it: the caller inserts both atomically with
/// [`SessionCatalog::put_view`].
///
/// # Errors
/// `INTERNAL_ERROR` for a scan of a temporary view that was not inlined.
pub(crate) fn pin_temp_tables(
    catalog: &SessionCatalog,
    plan: &Arc<LogicalPlan>,
    case_sensitive: bool,
) -> Result<PinnedPlan, KrishivError> {
    let mut pinner = Pinner {
        catalog,
        case_sensitive,
        pins: HashMap::new(),
        owned: BTreeSet::new(),
        new_pins: Vec::new(),
    };
    let plan = pinner.run(plan)?;
    Ok((plan, pinner.owned.into_iter().collect(), pinner.new_pins))
}

/// [`pin_temp_tables`]: the pins handed out so far, by lower-case table
/// name, every hidden name the pinned plan reads, and the hidden entries
/// to store with the owning view.
struct Pinner<'a> {
    catalog: &'a SessionCatalog,
    case_sensitive: bool,
    pins: HashMap<String, String>,
    owned: BTreeSet<String>,
    new_pins: Vec<(String, Arc<TempTable>)>,
}

impl Pinner<'_> {
    fn run(&mut self, plan: &Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, KrishivError> {
        if let PlanNode::Scan(scan) = &plan.node
            && let ReadSource::Table { name, .. } = &scan.source
            && let Some(last) = name.last()
        {
            if is_hidden_name(last) || is_cache_name(last) {
                self.owned.insert(last.to_lowercase());
                return Ok(Arc::clone(plan));
            }
            let key = last.to_lowercase();
            if !self.pins.contains_key(&key) {
                let pin = match self.catalog.get(last, self.case_sensitive) {
                    Some(table)
                        if matches!(
                            &*table,
                            TempTable::Memory { .. } | TempTable::Files { .. }
                        ) =>
                    {
                        let hidden = self.catalog.next_hidden_name();
                        self.owned.insert(hidden.clone());
                        self.new_pins.push((hidden.clone(), table));
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
pub(crate) fn search_path() -> String {
    format!("[`system`.`builtin`, `system`.`session`, `{CATALOG}`.`{NAMESPACE}`]")
}

/// `name` rejected as a temporary table or view name: it starts with the
/// reserved snapshot prefix ([`HIDDEN_PREFIX`]) or the cache prefix
/// ([`CACHE_PREFIX`](crate::cache::CACHE_PREFIX)), in any letter case. The
/// `KRISHIV_RESERVED_TEMP_VIEW_NAME` error names the matched prefix in its
/// `prefix` parameter.
pub(crate) fn reserved_temp_view_name(name: &str) -> KrishivError {
    let prefix = if is_cache_name(name) {
        crate::cache::CACHE_PREFIX
    } else {
        HIDDEN_PREFIX
    };
    KrishivReservedTempViewName {
        relation_name: quote(name),
        prefix: format!("`{prefix}`"),
    }
    .into()
}

#[cfg(test)]
mod tests {
    /// The scans of the streaming pipeline of `root`.
    fn streaming_scans(root: &PhysNode) -> Vec<&krishiv_exec::physical::ScanNode> {
        super::streaming_leaves(root).0
    }

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
            known: false,
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
    fn view_creation_stores_pins_and_view_together() {
        // The pins and the owning view insert in one critical section, so
        // one epoch bump: no concurrent `put` or `drop_table` can collect
        // a new pin before its view arrives.
        let s = int_session(vec![1]);
        let before = s.state.catalog.epoch();
        block_on(s.table("t").create_or_replace_temp_view("v")).unwrap();
        assert_eq!(s.state.catalog.epoch(), before + 1);
        assert_eq!(s.state.catalog.entry_count(), 3);
        assert_eq!(select_ids(&s, "SELECT id FROM v"), vec![1]);
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
        // With read options too, while a pin of that name exists: the
        // options do not reach the pinned snapshot.
        let missing_options = block_on(s.sql("SELECT * FROM nope WITH ('a' = 'b')")).unwrap_err();
        assert_eq!(
            missing_options.error_class(),
            ErrorClass::TableOrViewNotFound
        );
        let e = block_on(s.sql("SELECT * FROM __krishiv_snapshot_0 WITH ('a' = 'b')")).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::TableOrViewNotFound);
        assert_eq!(e.sql_state(), missing_options.sql_state());
        assert_eq!(
            e.message().replace("`__krishiv_snapshot_0`", "`nope`"),
            missing_options.message()
        );
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

    /// Writes `rows` `id` values to `path` as Parquet with row groups of at
    /// most `row_group` rows.
    fn write_parquet(path: &std::path::Path, rows: i32, row_group: usize) {
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "id",
            arrow_schema::DataType::Int32,
            false,
        )]));
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(row_group))
            .build();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::clone(&schema), Some(props)).unwrap();
        let mut at = 0;
        while at < rows {
            let take = (rows - at).min(50);
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int32Array::from(
                    (at..at + take).collect::<Vec<_>>(),
                ))],
            )
            .unwrap();
            writer.write(&batch).unwrap();
            at += take;
        }
        writer.close().unwrap();
    }

    /// The `id` column of `batches`, sorted.
    fn ids_of(batches: &[RecordBatch]) -> Vec<i32> {
        let mut out = Vec::new();
        for batch in batches {
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
    fn task_specs_deal_file_scans_deterministically_and_cover_everything() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 500, 100);
        write_parquet(&dir.path().join("b.parquet"), 300, 100);
        let session = Session::builder()
            // Small splits: every file is cut into several byte ranges.
            .config("spark.sql.files.maxPartitionBytes", "1024")
            .build()
            .unwrap();
        let df = session.read().parquet([dir.path().to_str().unwrap()]);
        let analysis = rt.block_on(df.analysis()).unwrap();
        let (_, physical) = rt
            .block_on(session.physical(&analysis.analyzed, &analysis.env))
            .unwrap();
        let catalog = &analysis.env.catalog;
        // The same listing and config give the same assignment, so a
        // retried task reads the same splits.
        let first = catalog.task_specs(&physical, 3, 3).unwrap();
        let second = catalog.task_specs(&physical, 3, 3).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 3);
        // Every task's partitions are contiguous, and together the tasks
        // cover every planned partition exactly once.
        let mut indexes = Vec::new();
        for task in &first {
            assert_eq!(task.scans.len(), 1, "one scan: {task:?}");
            let splits = task.scans.values().next().unwrap();
            let krishiv_exec::physical::ScanSplits::Files { files, partitions } = splits else {
                panic!("file splits: {splits:?}");
            };
            assert!(!files.is_empty());
            let mut last: Option<usize> = None;
            for part in partitions {
                if let Some(previous) = last {
                    assert_eq!(part.index, previous + 1, "contiguous: {partitions:?}");
                }
                last = Some(part.index);
                indexes.push(part.index);
            }
        }
        indexes.sort_unstable();
        assert!(!indexes.is_empty(), "several splits");
        assert_eq!(
            indexes,
            (0..indexes.len()).collect::<Vec<_>>(),
            "every partition exactly once"
        );
        // The union of 1..=4 tasks is the single run, with no duplicates.
        let whole = ids_of(&rt.block_on(df.collect()).unwrap());
        assert_eq!(whole.len(), 800);
        for tasks in 1..=4_usize {
            let specs = catalog
                .task_specs(&physical, tasks, u64::try_from(tasks).unwrap())
                .unwrap();
            assert_eq!(specs.len(), tasks);
            let mut union = Vec::new();
            for task in &specs {
                union.extend(ids_of(
                    &rt.block_on(session.run_task(&physical, &analysis.env, task))
                        .unwrap(),
                ));
            }
            union.sort_unstable();
            assert_eq!(union, whole, "{tasks} tasks");
        }
    }

    #[test]
    fn task_specs_deal_a_view_over_files() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 500, 100);
        write_parquet(&dir.path().join("b.parquet"), 300, 100);
        let session = Session::builder()
            // Small splits: every file is cut into several byte ranges.
            .config("spark.sql.files.maxPartitionBytes", "1024")
            .build()
            .unwrap();
        // A view over a files read keeps listing them: the scan names the
        // view, and its splits come from the opened listing.
        let files = session.read().parquet([dir.path().to_str().unwrap()]);
        rt.block_on(files.create_or_replace_temp_view("pv"))
            .unwrap();
        let df = block_on(session.sql("SELECT id FROM pv")).unwrap();
        let analysis = rt.block_on(df.analysis()).unwrap();
        assert!(analysis.env.catalog.loaded_file_views());
        let (_, physical) = rt
            .block_on(session.physical(&analysis.analyzed, &analysis.env))
            .unwrap();
        let catalog = &analysis.env.catalog;
        // The same listing and config give the same assignment, so a
        // retried task reads the same splits.
        let first = catalog.task_specs(&physical, 3, 3).unwrap();
        let second = catalog.task_specs(&physical, 3, 3).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 3);
        // The union of 1..=3 tasks is the single run, with no duplicates:
        // every task reads exactly its splits, never the whole view.
        let whole = ids_of(&rt.block_on(df.collect()).unwrap());
        assert_eq!(whole.len(), 800);
        for tasks in 1..=3_usize {
            let specs = catalog
                .task_specs(&physical, tasks, u64::try_from(tasks).unwrap())
                .unwrap();
            assert_eq!(specs.len(), tasks);
            let mut union = Vec::new();
            for task in &specs {
                assert_eq!(task.scans.len(), 1, "one scan: {task:?}");
                union.extend(ids_of(
                    &rt.block_on(session.run_task(&physical, &analysis.env, task))
                        .unwrap(),
                ));
            }
            union.sort_unstable();
            assert_eq!(union, whole, "{tasks} tasks");
        }
    }

    /// A streaming `Range` has no split assignment: one task is fine,
    /// several would each emit the whole range, so they are refused.
    #[test]
    fn task_specs_refuse_a_range_across_several_tasks() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let session = Session::new().unwrap();
        let df = session.range(0, 10, 1, None);
        let analysis = rt.block_on(df.analysis()).unwrap();
        let (_, physical) = rt
            .block_on(session.physical(&analysis.analyzed, &analysis.env))
            .unwrap();
        assert_eq!(
            analysis
                .env
                .catalog
                .task_specs(&physical, 1, 1)
                .unwrap()
                .len(),
            1
        );
        let e = analysis
            .env
            .catalog
            .task_specs(&physical, 2, 2)
            .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    }

    #[test]
    fn a_file_changed_between_listing_and_reading_fails() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("a.parquet"), 200, 100);
        let session = Session::builder()
            .config("spark.sql.files.maxPartitionBytes", "4096")
            .build()
            .unwrap();
        let df = session.read().parquet([dir.path().to_str().unwrap()]);
        let analysis = rt.block_on(df.analysis()).unwrap();
        let (_, physical) = rt
            .block_on(session.physical(&analysis.analyzed, &analysis.env))
            .unwrap();
        let specs = analysis.env.catalog.task_specs(&physical, 1, 1).unwrap();
        assert_eq!(specs.len(), 1);
        // New bytes under an old name: the listed version is stale (the
        // size differs too, so the failure does not depend on timestamp
        // granularity).
        write_parquet(&dir.path().join("a.parquet"), 10, 100);
        let e = rt
            .block_on(session.run_task(&physical, &analysis.env, &specs[0]))
            .unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivFileChangedDuringRead,
            "{e}"
        );
    }

    /// The `id` values of `sql` over `session`, sorted.
    fn table_ids(session: &Session, sql: &str) -> Vec<i32> {
        let df = block_on(session.sql(sql)).unwrap();
        ids_of(&block_on(df.collect()).unwrap())
    }

    #[test]
    fn task_specs_deal_memory_tables_and_release_them() {
        use krishiv_exec::physical::TaskSpec;
        let session = Session::new().unwrap();
        let base = session.memory_reserved();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "id",
            arrow_schema::DataType::Int32,
            false,
        )]));
        let batches_of = |values: Vec<i32>| {
            values
                .chunks(3)
                .map(|chunk| {
                    RecordBatch::try_new(
                        Arc::clone(&schema),
                        vec![Arc::new(Int32Array::from(chunk.to_vec()))],
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
        };
        let values: Vec<i32> = (0..14).collect();
        session
            .register_batches("t", Arc::clone(&schema), batches_of(values.clone()))
            .unwrap();
        // The same data under another name shares its content hash.
        session
            .register_batches("u", Arc::clone(&schema), batches_of(values))
            .unwrap();
        session
            .register_batches("v", Arc::clone(&schema), batches_of(vec![99]))
            .unwrap();
        let hash_of = |name: &str| {
            let entry = session.state.catalog.get(name, false).unwrap();
            let TempTable::Memory { table, .. } = &*entry else {
                panic!("a memory table: {name}");
            };
            table.relation_lease().unwrap().hash()
        };
        assert_eq!(hash_of("t"), hash_of("u"), "same data, same hash");
        assert_ne!(hash_of("t"), hash_of("v"), "different data, different hash");
        assert_eq!(session.relations().len(), 2, "t and u share one entry");
        assert!(
            session.relations().reserved() > 0,
            "the entries hold store bytes while leased"
        );
        // Holding tables never consumes the query pool's budget (tight
        // spill tests register large tables): the store has its own pool.
        assert_eq!(
            session.memory_reserved(),
            base,
            "registration leaves the session pool alone"
        );
        // The union of 1..=4 tasks is the table, with no duplicates; the
        // same listing gives the same assignment.
        let df = session.table("t");
        let analysis = block_on(df.analysis()).unwrap();
        let (_, physical) = block_on(session.physical(&analysis.analyzed, &analysis.env)).unwrap();
        let whole = table_ids(&session, "SELECT id FROM t");
        assert_eq!(whole, (0..14).collect::<Vec<_>>());
        for tasks in 1..=4_usize {
            let catalog = &analysis.env.catalog;
            let specs = catalog
                .task_specs(&physical, tasks, u64::try_from(tasks).unwrap())
                .unwrap();
            assert_eq!(
                specs,
                catalog
                    .task_specs(&physical, tasks, u64::try_from(tasks).unwrap())
                    .unwrap(),
                "deterministic"
            );
            assert_eq!(specs.len(), tasks);
            let mut union = Vec::new();
            for task in &specs {
                assert_eq!(task.scans.len(), 1, "one scan: {task:?}");
                union.extend(ids_of(
                    &block_on(session.run_task(&physical, &analysis.env, task)).unwrap(),
                ));
            }
            union.sort_unstable();
            assert_eq!(union, whole, "{tasks} tasks");
        }
        // A scan without an entry reads the whole table, exactly as before.
        let single = ids_of(
            &block_on(session.run_task(&physical, &analysis.env, &TaskSpec::whole())).unwrap(),
        );
        assert_eq!(single, whole);
        // The test's analysis holds a table lease through its loaded
        // tables: release it before the tables are dropped.
        drop(analysis);
        drop(df);
        // Dropping the tables releases their entries and their bytes.
        assert!(session.drop_temp_view("t"));
        assert_eq!(session.relations().len(), 2, "u still holds the entry");
        assert!(session.drop_temp_view("u"));
        assert_eq!(session.relations().len(), 1);
        assert!(session.drop_temp_view("v"));
        assert!(session.relations().is_empty());
        assert_eq!(
            session.relations().reserved(),
            0,
            "the last lease returns the bytes"
        );
        assert_eq!(
            session.memory_reserved(),
            base,
            "queries and drops leave the session pool as it was"
        );
    }

    #[test]
    fn task_specs_splits_only_the_streaming_pipeline() {
        // Broadcast builds and scalar subqueries read whole: tasks join
        // probe shares against the full build and evaluate the subquery
        // over its full input, so their union is the single run.
        let session = Session::new().unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "id",
            arrow_schema::DataType::Int32,
            false,
        )]));
        let batches_of = |values: Vec<i32>| {
            values
                .chunks(3)
                .map(|chunk| {
                    RecordBatch::try_new(
                        Arc::clone(&schema),
                        vec![Arc::new(Int32Array::from(chunk.to_vec()))],
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
        };
        session
            .register_batches("t", Arc::clone(&schema), batches_of((0..14).collect()))
            .unwrap();
        session
            .register_batches("u", Arc::clone(&schema), batches_of((0..14).collect()))
            .unwrap();
        session
            .register_batches("v", Arc::clone(&schema), batches_of(vec![5]))
            .unwrap();
        let sql = "SELECT t.id FROM t JOIN u ON t.id = u.id WHERE t.id > (SELECT max(id) FROM v)";
        let whole = table_ids(&session, sql);
        assert_eq!(whole, (6..14).collect::<Vec<_>>());
        let df = block_on(session.sql(sql)).unwrap();
        let analysis = block_on(df.analysis()).unwrap();
        let (_, physical) = block_on(session.physical(&analysis.analyzed, &analysis.env)).unwrap();
        // A shared plan runs in one task until it lowers to an exchange.
        let shared = physical
            .clone()
            .with_shared(vec![krishiv_exec::physical::SharedPlan {
                id: krishiv_exec::physical::SharedId(0),
                plan: PhysNode::OneRow,
                explain: krishiv_exec::planner::ExplainNode::leaf("OneRowRelation", ""),
                deterministic: true,
                ordered: false,
            }])
            .unwrap();
        let e = analysis.env.catalog.task_specs(&shared, 2, 2).unwrap_err();
        assert!(e.to_string().contains("a shared plan across"), "{e}");
        assert_eq!(
            analysis
                .env
                .catalog
                .task_specs(&shared, 1, 1)
                .unwrap()
                .len(),
            1
        );
        for tasks in [2_usize, 3_usize] {
            let specs = analysis
                .env
                .catalog
                .task_specs(&physical, tasks, u64::try_from(tasks).unwrap())
                .unwrap();
            assert_eq!(specs.len(), tasks);
            // Only the streaming scan is dealt ranges; builds and the
            // subquery's scan read whole (no entry).
            let streaming = streaming_scans(physical.root());
            assert_eq!(streaming.len(), 1, "one streaming scan");
            for task in &specs {
                assert_eq!(task.scans.len(), 1, "probe only: {task:?}");
                assert!(
                    task.scans.contains_key(&streaming[0].id),
                    "the streaming scan: {task:?}"
                );
            }
            let mut union = Vec::new();
            for task in &specs {
                union.extend(ids_of(
                    &block_on(session.run_task(&physical, &analysis.env, task)).unwrap(),
                ));
            }
            union.sort_unstable();
            assert_eq!(union, whole, "{tasks} tasks");
        }
        drop(analysis);
        drop(df);
    }

    #[test]
    fn task_specs_rejects_no_tasks() {
        let session = Session::new().unwrap();
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
        session.register_batches("t", schema, vec![batch]).unwrap();
        let df = session.table("t");
        let analysis = block_on(df.analysis()).unwrap();
        let (_, physical) = block_on(session.physical(&analysis.analyzed, &analysis.env)).unwrap();
        let e = analysis
            .env
            .catalog
            .task_specs(&physical, 0, 0)
            .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
        // The analysis holds the old table through its loaded tables.
        drop(analysis);
        drop(df);
        // Replacing the table releases the old entry: one entry either way.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "id",
            arrow_schema::DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![2]))],
        )
        .unwrap();
        session.register_batches("t", schema, vec![batch]).unwrap();
        assert_eq!(session.relations().len(), 1);
        assert_eq!(table_ids(&session, "SELECT id FROM t"), [2]);
    }

    /// A scan plan over `source`: what `task_specs` deals splits for.
    fn split_scan_plan(source: ReadSource) -> PhysicalPlan {
        use krishiv_exec::RowStageSettings;
        use krishiv_exec::physical::{ExprEnv, ScanNode};
        use krishiv_exec::planner::ExplainNode;
        use krishiv_io::ColumnPath;
        use krishiv_plan::{AttrId, BoundColumn};

        let scan = Scan {
            source,
            output: vec![BoundColumn {
                id: AttrId(1),
                name: "id".to_owned(),
                data_type: DataType::Integer,
                nullable: true,
            }],
            version: None,
        };
        PhysicalPlan::new(
            PhysNode::Scan(ScanNode {
                id: ScanId(1),
                scan,
                projection: vec![ColumnPath::new("id")],
                filter: None,
                runtime_filters: Vec::new(),
                row_stage: RowStageSettings::default(),
            }),
            Arc::new(Schema::new(vec![Field::new(
                "id",
                arrow_schema::DataType::Int32,
                true,
            )])),
            ExplainNode::leaf("Scan", ""),
            ExprEnv {
                ansi: true,
                legacy_statistical_aggregate: false,
                zone: None,
                zone_name: None,
                query_start_micros: 0,
                user: Arc::from("u"),
                map_key_dedup_last_win: false,
                legacy_java_charsets: false,
                legacy_coding_error_action: false,
            },
        )
    }

    /// A table scan source over `name`.
    fn table_source(name: &str) -> ReadSource {
        ReadSource::Table {
            name: vec![
                "spark_catalog".to_owned(),
                "default".to_owned(),
                name.to_owned(),
            ],
            options: BTreeMap::new(),
        }
    }

    #[test]
    fn task_specs_rejects_what_it_cannot_split() {
        let session = Session::new().unwrap();
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
        session.register_batches("t", schema, vec![batch]).unwrap();
        block_on(session.sql("CREATE TEMPORARY VIEW w AS SELECT 1 AS id")).unwrap();
        // The analysis references the view (inlined afterwards), so the
        // catalog holds its entry while no scan names it.
        let df = block_on(session.sql("SELECT id FROM w")).unwrap();
        let analysis = block_on(df.analysis()).unwrap();
        let catalog = &analysis.env.catalog;
        // The view's own entry, as an analysis that kept it would hold
        // it: a view reaching split planning was not inlined.
        let view = session.state.catalog.get("w", false).unwrap();
        catalog.loaded.lock().unwrap().insert("w".to_owned(), view);

        // A scan of a table the analysis never loaded.
        let e = catalog
            .task_specs(&split_scan_plan(table_source("nope")), 2, 2)
            .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
        assert!(e.to_string().contains("was not loaded"), "{e}");

        // A scan of a temporary view: views inline during analysis, so a
        // view reaching split planning is a coordinator bug.
        let e = catalog
            .task_specs(&split_scan_plan(table_source("w")), 2, 2)
            .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
        assert!(e.to_string().contains("was not inlined"), "{e}");

        // A scan of files that were never opened.
        let e = catalog
            .task_specs(
                &split_scan_plan(ReadSource::Files {
                    format: "parquet".to_owned(),
                    paths: Vec::new(),
                    options: BTreeMap::new(),
                }),
                2,
                2,
            )
            .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
        assert!(e.to_string().contains("not opened"), "{e}");

        // More tasks than a `u32` holds: rejected before anything is dealt.
        let e = catalog
            .task_specs(
                &split_scan_plan(table_source("t")),
                usize::try_from(u32::MAX).unwrap() + 1,
                1,
            )
            .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
        drop(analysis);
        drop(df);
    }
}
