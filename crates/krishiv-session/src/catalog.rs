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
    BoxFuture, Catalog, CatalogResolver, FileMetadata, NameForm, TableColumn, TableIdent,
    TableKind, TableMeta,
};
use krishiv_plan::{
    ALIAS_PART_SEP, Cte, Expr, ExprNode, LogicalPlan, Origin, PlanNode, ReadSource, Scan,
    TableVersion,
};

use crate::cache::{RelationCache, is_cache_name};

mod declared;
#[cfg(test)]
mod declared_tests;
mod file_tables;
mod key_facts;
mod ndv_sampling;
mod scans;
mod script_results;
mod session_catalog;

use self::file_tables::{FileScanOf, reference_key};
pub(crate) use self::scans::reads_source;
pub(crate) use self::scans::reads_stored_table;
#[cfg(feature = "delta")]
pub(crate) use self::scans::{reads_delta_root, reads_files_at};

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
    /// Any table provider under a hidden name: a row-level statement's target at the version the
    /// statement read, with its row id (`RowIdScan`).
    Provider {
        columns: Vec<TableColumn>,
        table: Arc<dyn TableProvider>,
    },
    /// A read by path, with the columns inferred when the view was created.
    Files {
        source: ReadSource,
        columns: Vec<TableColumn>,
        /// The columns are a permanent table's own, not inferred: reading
        /// it lists its files without inferring a schema, and a table with
        /// no files is empty.
        known: bool,
        /// An entry of the permanent catalog: only a reference to one takes read options of its
        /// own (`FROM t WITH (..)`); Spark drops them for a temporary view.
        permanent: bool,
        /// A permanent table's partition columns and registered partitions
        /// (`None` when they are not all at their default directories, so a
        /// listing under the root cannot read them as Spark does). Empty
        /// otherwise.
        partitions: Option<krishiv_exec::files::KnownPartitions>,
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
        /// The query text of a view `CREATE TEMPORARY VIEW` made (`SHOW TABLE EXTENDED` prints it as
        /// `View Text`); `None` for a DataFrame view.
        text: Option<String>,
        /// What the frame that made a DataFrame view kept (its `Observation` handles): a query that
        /// inlines this entry runs its observed nodes, so it completes them. Part of the entry, so a
        /// query reads the handles of the definition it expanded, and they are published with it.
        keeps: ViewKeeps,
    },
}

impl TempTable {
    /// A view `CREATE TEMPORARY VIEW` made over `plan`, with its query `text`: no pins, no keeps.
    pub(crate) fn sql_view(
        plan: Arc<LogicalPlan>,
        columns: Vec<TableColumn>,
        text: Option<String>,
    ) -> Self {
        Self::View {
            plan,
            columns,
            pins: Vec::new(),
            text,
            keeps: ViewKeeps::default(),
        }
    }
}

/// A keep-alive handle a frame holds (an `Observation`'s shared state, a sink).
pub(crate) type Keep = Arc<dyn std::any::Any + Send + Sync>;

/// [`TempTable::View::keeps`]: empty for a SQL view.
#[derive(Clone, Default)]
pub(crate) struct ViewKeeps(pub(crate) Vec<Keep>);

impl std::fmt::Debug for ViewKeeps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ViewKeeps({})", self.0.len())
    }
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

/// Whether `name` is spelled like a [`global_key`], in any letter case (catalog keys are
/// lower-cased): no user-written name may be, or a local view could overwrite, read or drop a
/// global one without the global view's checks.
pub(crate) fn is_global_key(name: &str) -> bool {
    name.get(..GLOBAL_PREFIX.len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(GLOBAL_PREFIX))
}

/// Whether a user may not name a local view `name`: a snapshot pin, a cache entry or a global key.
pub(crate) fn is_reserved_view_name(name: &str) -> bool {
    is_hidden_name(name) || is_cache_name(name) || is_global_key(name)
}

/// The temporary tables and views of a session, by lower-case name.
#[derive(Debug, Default)]
pub(crate) struct SessionCatalog {
    tables: RwLock<BTreeMap<String, (String, Arc<TempTable>)>>,
    epoch: AtomicU64,
    /// Hidden names handed out for snapshot pins (see [`SessionCatalog::put_view`]).
    hidden: AtomicU64,
    /// The permanent tables, apart from the temporary views above: the
    /// runtime's, shared by all its sessions as Spark's `newSession()`
    /// shares the external catalog.
    pub(crate) store: Arc<InMemoryCatalogStore>,
    /// Where managed tables live (the runtime's).
    pub(crate) warehouse: Arc<Warehouse>,
    /// The current catalog (`USE system`; empty is `spark_catalog`) and the namespace bare
    /// table names resolve in (`USE`; empty is the default), under one lock so a reader never
    /// sees half of a `USE`.
    position: RwLock<(String, String)>,
    /// The entries `SET PATH` stored; `None` is the default path.
    path: RwLock<Option<Vec<crate::sql_path::Entry>>>,
}

/// What a query reads files for: a view (by lower-case name) or a read by
/// path.
#[derive(Debug, Clone, PartialEq)]
enum FilesKey {
    View(String),
    /// A permanent table, by `namespace.table`.
    Table(String),
    Read(ReadSource),
    /// A reference to a file table with read options of its own: the table's identity (and
    /// whether its columns are known) plus the files read with the merged options.
    Reference {
        key: String,
        known: bool,
        source: ReadSource,
    },
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
    /// The session's relation store, which holds the data of local relations.
    pub(crate) relations: Arc<krishiv_memory::RelationStore>,
    /// The session's memory pool (under the runtime's root and
    /// `krishiv.session.memoryLimit`): plan-time work that holds memory, the
    /// key counts, reserves from a child of it, since the query's own pool
    /// exists only once planning is done.
    pub(crate) memory: Arc<dyn krishiv_memory::MemoryPool>,
}

impl FileScans {
    /// A lease on the entry of the local relation `l`: its own lease, else one
    /// found by hash in the session's store.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` naming the hash when the store does not hold it (a
    /// decoded plan run where its data is not): never an empty result.
    pub(crate) fn local_lease(
        &self,
        l: &krishiv_plan::LocalRelation,
    ) -> Result<krishiv_memory::RelationLease, KrishivError> {
        if let Some(lease) = l.lease() {
            return Ok(lease.clone());
        }
        let hash = l.relation().hash;
        self.relations.lease(&hash).ok_or_else(|| {
            InternalError {
                message: format!("the local relation {hash} is not in this session's store"),
            }
            .into()
        })
    }
}

/// A snapshot pinned by hidden name, and how many actions pinned it.
type Pinned = (Arc<dyn TableProvider>, usize);

/// The session catalog as one query sees it: the catalog resolver and the
/// table providers of its scans.
pub(crate) struct QueryCatalog {
    session: Arc<SessionCatalog>,
    cache: Arc<RelationCache>,
    /// The cached snapshots this query's plan was rewritten to scan, by
    /// hidden name: held so a concurrent `CLEAR CACHE` cannot pull them out
    /// from under the query between substitution and planning. Counted:
    /// each action over the analysis pins and unpins (`SnapshotLease`).
    pinned: Mutex<HashMap<String, Pinned>>,
    files: Arc<FileReads>,
    /// The Delta tables of the query's scans, opened before planning.
    #[cfg_attr(
        not(feature = "delta"),
        expect(
            dead_code,
            reason = "read by the Delta reads of the `delta` feature only"
        )
    )]
    pub(crate) delta: Arc<crate::delta::DeltaReads>,
    scans: FileScans,
    /// Entries loaded by this query, by lower-case name.
    loaded: Mutex<HashMap<String, Arc<TempTable>>>,
    /// The file tables of the query's scans, opened before planning.
    opened: Mutex<Vec<(FilesKey, Arc<FileTable>)>>,
    epoch: u64,
    /// The current namespace (`USE`) as this query sees it.
    namespace: Vec<String>,
    /// The current catalog (`USE system`) as this query sees it.
    catalog: String,
    /// The SQL path (`SET PATH`) as this query sees it.
    path: Vec<Vec<String>>,
    /// What the `system` tables read.
    system: crate::system::Sources,
    /// Which kinds of table the planned scans read: a `system.*` table, anything else. A query
    /// that read only `system.*` tables skips admission (`reads_only_system`).
    scanned: Scanned,
    /// The estimate of the query's optimized plan, kept for the admission request when the
    /// controller asked for one (`AdmissionController::uses_estimate`).
    estimate: Mutex<Option<crate::explain::Estimate>>,
}

/// The kinds of table a query's scans read (set when each scan's provider is resolved).
#[derive(Debug, Default)]
struct Scanned {
    system: std::sync::atomic::AtomicBool,
    other: std::sync::atomic::AtomicBool,
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
        delta: Arc<crate::delta::DeltaReads>,
        scans: FileScans,
        default_path: &str,
        system: crate::system::Sources,
    ) -> Self {
        let epoch = session.epoch();
        // One read of the position, so the snapshot never mixes two `USE`s.
        let (catalog, current_namespace) = session.position_pair();
        let namespace = if crate::system::is_system(&catalog) {
            Vec::new()
        } else {
            vec![current_namespace]
        };
        let path = session.sql_path_in(default_path, &catalog, &namespace);
        Self {
            session,
            cache,
            pinned: Mutex::new(HashMap::new()),
            files,
            delta,
            scans,
            loaded: Mutex::new(HashMap::new()),
            opened: Mutex::new(Vec::new()),
            epoch,
            namespace,
            catalog,
            path,
            system,
            scanned: Scanned::default(),
            estimate: Mutex::new(None),
        }
    }

    /// Whether the planned scans and `plan`'s leaves read `system.*` tables and nothing else
    /// (`system::plan_reads_only_system`): such a query is never queued behind the load.
    pub(crate) fn reads_only_system(&self, plan: &krishiv_exec::physical::PhysicalPlan) -> bool {
        use std::sync::atomic::Ordering::SeqCst;
        (self.scanned.system.load(SeqCst) && !self.scanned.other.load(SeqCst))
            && crate::system::plan_reads_only_system(plan)
    }

    /// Keeps `estimate` for the query's admission request.
    pub(crate) fn set_estimate(&self, estimate: crate::explain::Estimate) {
        *self.estimate.lock().unwrap_or_else(PoisonError::into_inner) = Some(estimate);
    }

    /// The kept estimate, `None` when none was computed (unknown).
    pub(crate) fn estimate(&self) -> Option<crate::explain::Estimate> {
        self.estimate
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Keeps the snapshot `table` for this query's scans of the hidden `name`.
    pub(crate) fn pin_cached(&self, name: String, table: Arc<dyn TableProvider>) {
        let mut pinned = self.pinned.lock().unwrap_or_else(PoisonError::into_inner);
        let slot = pinned
            .entry(name)
            .or_insert_with(|| (Arc::clone(&table), 0));
        slot.0 = table;
        slot.1 += 1;
    }

    /// Lets go of one pin of the hidden `name` ([`Self::pin_cached`]); the
    /// last one drops the snapshot from this query's catalog.
    pub(crate) fn unpin_cached(&self, name: &str) {
        let mut pinned = self.pinned.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(slot) = pinned.get_mut(name) {
            slot.1 = slot.1.saturating_sub(1);
            if slot.1 == 0 {
                pinned.remove(name);
            }
        }
    }

    /// The `_metadata` column `scan` can read (`input_file_name()` reads it): its file source's, none for a
    /// table that is no file source or a cached snapshot substituted after analysis (Spark's cached data has no
    /// file either). Iceberg and Delta scans are no `TempTable::Files`: none (a recorded divergence,
    /// `krishiv-plan` `analyzer/input_file.rs`).
    pub(crate) fn scan_file_metadata(&self, scan: &Scan) -> FileMetadata {
        let format = match &scan.source {
            ReadSource::Files { format, .. } => format.clone(),
            ReadSource::Table { name, .. } if !crate::system::is_system_table(name) => {
                match self.loaded_for(name).as_deref() {
                    Some(TempTable::Files {
                        source: ReadSource::Files { format, .. },
                        ..
                    }) => format.clone(),
                    _ => return FileMetadata::None,
                }
            }
            _ => return FileMetadata::None,
        };
        file_metadata(&format)
    }

    /// The root of the Delta table `scan` reads, as [`crate::delta::DeltaReads::hadoop_root`] names it
    /// (`file:/..`): a read by path (`` delta.`p` ``, `format("delta")`) or a catalog table whose entry
    /// this query opened as a Delta read. `None`: not a Delta read (the hidden target of a row-level
    /// statement is a provider entry, so it is none). `Some(Err)`: a Delta read whose root cannot be
    /// resolved.
    #[cfg(feature = "delta")]
    pub(crate) fn delta_root(&self, scan: &Scan) -> Option<Result<String, KrishivError>> {
        let source = match &scan.source {
            source @ ReadSource::Files { .. } => source.clone(),
            ReadSource::Table { name, .. } => match self.loaded_for(name).as_deref() {
                Some(TempTable::Files { source, .. }) => source.clone(),
                _ => return None,
            },
            _ => return None,
        };
        crate::delta::is_delta(&source).then(|| self.delta.hadoop_root(&source))
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

    /// Whether this query loaded a file view: its files are listed again
    /// by the next query, so an analysis that did must not be reused.
    pub(crate) fn loaded_file_views(&self) -> bool {
        self.loaded
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .any(|t| matches!(&**t, TempTable::Files { .. }))
    }

    /// Every file table opened for the query, in opening order.
    fn opened_tables(&self) -> Vec<(FilesKey, Arc<FileTable>)> {
        self.opened
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
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

    /// Whether any scan of `plan` reads a Delta table: every tree
    /// ([`PhysicalPlan::roots`]) and every node's inputs and subquery plans
    /// ([`PhysNode::children`]), not only the streaming pipeline.
    #[cfg(feature = "delta")]
    fn reads_delta(&self, plan: &PhysicalPlan) -> bool {
        let mut stack: Vec<&PhysNode> = plan.roots().collect();
        while let Some(node) = stack.pop() {
            if let PhysNode::Scan(scan) = node {
                let delta = match &scan.scan.source {
                    ReadSource::Table { name, .. } => self.loaded_for(name).is_some_and(
                        |t| matches!(&*t, TempTable::Files { source, .. } if crate::delta::is_delta(source)),
                    ),
                    source => crate::delta::is_delta(source),
                };
                if delta {
                    return true;
                }
            }
            stack.extend(node.children());
        }
        false
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
    /// whose files were not opened before planning;
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a plan that reads Delta anywhere
    /// (a worker cannot open the session's snapshots), a range source or a
    /// shared plan across several tasks, and a scan it cannot split.
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
        // A worker cannot open the session's Delta snapshots, so no stage
        // that reads Delta anywhere (a build side, a subquery or a shared
        // plan included, not only the streaming pipeline) leaves the driver.
        #[cfg(feature = "delta")]
        if self.reads_delta(plan) {
            return Err(unsupported("a Delta scan in a distributed stage"));
        }
        for scan in scans {
            match &scan.scan.source {
                ReadSource::Table { name, options } => {
                    let last = name.last().map_or("", String::as_str);
                    // A declared catalog's table is opened on the driver with the session's
                    // stores; a worker cannot open it yet (design P3-T067 PR-58b).
                    if self.is_declared_scan(name, scan.scan.version) {
                        return Err(self.declared_refusal(name, " in a distributed stage"));
                    }
                    let Some(table) = self.loaded(last) else {
                        // A relation cache snapshot lives on the driver (design P3-T083
                        // §3.8): a memory snapshot is dealt as a memory table without a
                        // lease is (no assignment: a stage of one task reads it whole);
                        // a disk snapshot is a driver-local file no worker can open, so
                        // such a plan is refused, not failed as an internal error.
                        if crate::cache::is_cache_name(last) {
                            if self.cache.is_on_disk(last) {
                                return Err(unsupported(
                                    "a scan of a disk-cached relation in a distributed stage",
                                ));
                            }
                            continue;
                        }
                        if let Some(e) = self.declared_read(name) {
                            return Err(e);
                        }
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
                        TempTable::Files {
                            source,
                            known,
                            permanent,
                            ..
                        } => {
                            let key = if options.is_empty() {
                                FilesKey::View(last.to_lowercase())
                            } else {
                                // The key `open_files` and the provider build: the entry's own.
                                reference_key(
                                    self.entry_of(name)
                                        .map_or_else(|| last.to_lowercase(), |(k, _)| k),
                                    *known,
                                    *permanent,
                                    source,
                                    options,
                                )
                                .0
                            };
                            let table = self.opened(&key).ok_or_else(|| InternalError {
                                message: "a scan of files that was not opened before planning"
                                    .to_owned(),
                            })?;
                            assigned.push((scan.id, self.file_splits(&table, total_slots)?));
                        }
                        // A row-level target plans its own splits through its scan.
                        TempTable::Provider { .. } => {}
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
                ReadSource::Local(l) => {
                    // The batches live in the session's store; the splits name
                    // them by hash. A relation the store lost is an engine
                    // bug, never an empty scan.
                    let lease = self.scans.local_lease(l)?;
                    assigned.push((
                        scan.id,
                        ScanSplitsKind::Batches {
                            relation: lease.reference(),
                            batches: lease.batches(),
                        },
                    ));
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
                partition,
                partitions,
                ..TaskSpec::whole()
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
            PhysNode::OneRow
            | PhysNode::SharedRead { .. }
            | PhysNode::LoopRef { .. }
            | PhysNode::ExchangeSource { .. } => {}
            PhysNode::UnionLoop {
                anchor, recursive, ..
            } => {
                go(anchor, out);
                go(recursive, out);
            }
            PhysNode::Range(_) => out.range = true,
            PhysNode::Filter { input, .. }
            | PhysNode::Project { input, .. }
            | PhysNode::ReplicateRows { input, .. }
            | PhysNode::Sample { input, .. }
            | PhysNode::CollectMetrics { input, .. }
            | PhysNode::Generate { input, .. }
            | PhysNode::Expand { input, .. }
            | PhysNode::Limit { input, .. }
            | PhysNode::Tail { input, .. }
            | PhysNode::Sort { input, .. }
            | PhysNode::TopN { input, .. }
            | PhysNode::LateFetch { input, .. }
            | PhysNode::Window { input, .. }
            | PhysNode::WindowGroupLimit { input, .. }
            | PhysNode::Aggregate { input, .. }
            | PhysNode::PythonEval { input, .. }
            | PhysNode::PythonMap { input, .. }
            | PhysNode::PythonGroupMap { input, .. }
            | PhysNode::PythonCoGroupMap { input, .. }
            | PhysNode::PythonUdtf { input, .. }
            | PhysNode::PythonAggregate { input, .. }
            | PhysNode::WriteFiles { input, .. }
            | PhysNode::TableWrite { input, .. } => go(input, out),
            PhysNode::Exchange(e) => go(&e.input, out),
            PhysNode::HashJoin(join) => go(&join.probe, out),
            PhysNode::SortMergeJoin(join) => go(&join.probe, out),
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
                char_varchar: c.char_varchar.clone(),
            })
            .collect();
        let entry = Arc::new(TempTable::Files {
            source: table_source(&table),
            columns: columns.clone(),
            known: true,
            permanent: true,
            partitions: file_tables::known_partitions(&table),
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
            file_metadata: file_metadata(&table.provider),
            column_comments: table
                .columns
                .iter()
                .filter_map(|c| Some((c.name.clone(), c.comment.clone()?)))
                .collect(),
        })
    }
}

/// The `_metadata` column of a table read through the file source
/// `provider` (Spark: file sources; row positions only from Parquet).
pub(crate) fn file_metadata(provider: &str) -> FileMetadata {
    match provider.to_ascii_lowercase().as_str() {
        "parquet" => FileMetadata::WithRowIndex,
        "orc" | "csv" | "json" | "text" | "avro" | "xml" | "binaryfile" => FileMetadata::Basic,
        _ => FileMetadata::None,
    }
}

/// The read of a table's files.
pub(crate) fn table_source(table: &crate::store::CatalogTable) -> ReadSource {
    ReadSource::files(
        table.provider.clone(),
        vec![krishiv_common::secret::OptionValue::from_user(
            "path",
            table.location.display().to_string(),
        )],
        table
            .options
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    krishiv_common::secret::OptionValue::from_user(k, v),
                )
            })
            .collect(),
    )
}

pub(super) fn unsupported(feature: &str) -> KrishivError {
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
                if form.temporary_only {
                    return Ok(None);
                }
                return Ok(self.load_stored(id, form));
            };
            let (columns, kind) = match &*table {
                TempTable::Memory { columns, .. }
                | TempTable::Files { columns, .. }
                | TempTable::Provider { columns, .. } => (columns.clone(), TableKind::Table),
                TempTable::View { columns, .. } => (columns.clone(), TableKind::View),
            };
            let file_metadata = match &*table {
                TempTable::Files {
                    source: ReadSource::Files { format, .. },
                    ..
                } => file_metadata(format),
                _ => FileMetadata::None,
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
                file_metadata,
                column_comments: BTreeMap::new(),
            }))
        })
    }
}

/// The resolver of one query: the session catalog, `system` and the declared catalogs.
#[derive(Debug)]
pub(crate) struct QueryCatalogs(pub(crate) Arc<QueryCatalog>);

impl CatalogResolver for QueryCatalogs {
    fn current_catalog(&self) -> &str {
        &self.0.catalog
    }

    fn current_namespace(&self) -> &[String] {
        &self.0.namespace
    }

    fn sql_path(&self) -> Vec<Vec<String>> {
        self.0.path.clone()
    }

    fn catalog(&self, name: &str) -> Option<Arc<dyn Catalog>> {
        if crate::system::is_system(name) {
            return Some(Arc::new(self.0.system_catalog()));
        }
        if name.eq_ignore_ascii_case(CATALOG) {
            return Some(Arc::clone(&self.0) as Arc<dyn Catalog>);
        }
        // A declared catalog the principal sees (design P3-T067 §2.3, §2.4); a hidden one is
        // absent, exactly as an undeclared name.
        self.0.system.catalogs.catalog(name)
    }

    fn epoch(&self) -> u64 {
        0
    }
}

impl TableProviderResolver for QueryCatalog {
    fn provider(&self, scan: &Scan) -> Result<Arc<dyn TableProvider>, KrishivError> {
        let kind = match &scan.source {
            ReadSource::Table { name, .. } if crate::system::is_system_table(name) => {
                &self.scanned.system
            }
            _ => &self.scanned.other,
        };
        kind.store(true, std::sync::atomic::Ordering::SeqCst);
        // A declared catalog's table, opened before planning (`open_declared`).
        if let ReadSource::Table { name, .. } = &scan.source
            && let Some(provider) = self.declared_provider(name, scan.version)
        {
            return Ok(provider);
        }
        match &scan.source {
            ReadSource::Table { name, .. } if crate::system::is_system_table(name) => {
                let table = name.last().map_or("", String::as_str);
                crate::system::provider(table, &self.system)?
                    .map(|t| t as Arc<dyn TableProvider>)
                    .ok_or_else(|| {
                        InternalError {
                            message: format!("the system table {table} does not exist"),
                        }
                        .into()
                    })
            }
            ReadSource::Table { name, options } => {
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
                        .map(|(table, _)| Arc::clone(table));
                    if let Some(cached) = pinned.or_else(|| self.cache.table(last)) {
                        return Ok(cached);
                    }
                    if let Some(e) = self.declared_read(name) {
                        return Err(e);
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
                    TempTable::Provider { table, .. } => Ok(Arc::clone(table)),
                    TempTable::Files {
                        known,
                        permanent,
                        source,
                        ..
                    } => {
                        let key = self
                            .entry_of(name)
                            .map_or_else(|| last.to_lowercase(), |(k, _)| k);
                        // The source `file_scans` opened: a permanent table's
                        // reference options are merged in, and for a cloud
                        // path they are part of the store's key.
                        let (key, effective) =
                            reference_key(key, *known, *permanent, source, options);
                        #[cfg(feature = "delta")]
                        if crate::delta::is_delta(&effective) {
                            return self.delta.provider(&effective);
                        }
                        #[cfg(not(feature = "delta"))]
                        let _ = effective;
                        self.files_provider(&key)
                    }
                    // Views are inlined as subqueries before analysis, so
                    // no scan ever reads one.
                    TempTable::View { .. } => Err(InternalError {
                        message: "a scan of a temporary view that was not inlined".to_owned(),
                    }
                    .into()),
                }
            }
            #[cfg(feature = "delta")]
            source @ ReadSource::Files { .. } if crate::delta::is_delta(source) => {
                self.delta.provider(source)
            }
            source @ ReadSource::Files { .. } => {
                self.files_provider(&FilesKey::Read(source.clone()))
            }
            ReadSource::Local(l) => {
                let lease = self.scans.local_lease(l)?;
                Ok(Arc::new(krishiv_exec::planner::MemoryTable::from_lease(
                    Arc::clone(l.schema()),
                    l.rows(),
                    lease,
                )))
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
    /// Whether the current catalog was another than `spark_catalog` (`system` or a declared one) as
    /// the query snapshot saw it.
    pub(crate) outside_session: bool,
    /// `spark.sql.legacy.ctePrecedencePolicy` in the query's snapshot. A stored view's text resolves its own
    /// `WITH`s under the policy captured with the view, never this one (review r4 Y1): unless both are
    /// `CORRECTED` they are inlined in the text alone before it is expanded
    /// (`krishiv_plan::analyzer::inline_view_ctes`), so the reader's pass over the whole plan finds none of
    /// them (review r3 M2) and a definer view's query is trusted by identity as built (review r2 R2).
    pub(crate) cte_precedence: krishiv_plan::analyzer::CtePrecedencePolicy,
    /// Qualifies the bare routine calls of an expression written in a namespace under a frozen path
    /// (`None` when nothing changed): the calls of a stored view's query resolve where the view was
    /// created, not where it is read.
    pub(crate) qualify: Option<&'a QualifyCalls<'a>>,
    /// As [`Self::qualify`], for the name of a table function a stored view's query calls (review
    /// r6 R6-1): the qualified name, its parts joined by `TABLE_FUNCTION_NAME_SEP`.
    pub(crate) qualify_table: Option<&'a QualifyTableCall<'a>>,
    /// The plan expanded is the body of a stored function (its namespace is the expansion's
    /// `namespace`): its own calls, scalar and table ones at every level, resolve through the
    /// function's frozen path or namespace as [`Self::qualify`] qualifies them for a body, never
    /// through the reader's session (review r7 R7-1; Spark v4.2.0 `Analyzer.scala` lines 260-270).
    pub(crate) stored_body: bool,
}

/// See [`StoredViews::qualify`]: the expression, the namespace it was written in, the path frozen with it and
/// the rule of the text it is in (a stored function's body, an invoker or a definer view's text).
pub(crate) type QualifyCalls<'a> = dyn Fn(&Arc<Expr>, &str, &[Vec<String>], crate::sql_functions::CallRule<'_>) -> Option<Arc<Expr>>
    + 'a;

/// See [`StoredViews::qualify_table`]: the table function's name as written, the namespace it was
/// written in, the path frozen with it and the rule of the text it is in.
pub(crate) type QualifyTableCall<'a> =
    dyn Fn(&str, &str, &[Vec<String>], crate::sql_functions::CallRule<'_>) -> Option<String> + 'a;

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
    let mut expander = Expander::new(catalog, case_sensitive, stored, namespace, view_path);
    let plan = expander.go(Arc::clone(plan), &mut Vec::new())?;
    Ok(expander.finish(plan))
}

/// A `SQL SECURITY DEFINER` view the expansion inlined (P3-T062 §2.4): its query reads with its
/// definer's rights. The reader is asked only about the view itself (among
/// [`Expansion::reads`]); the session asks [`Self::reads`] and analyzes [`Self::query`] as the
/// definer, then has the reader's analysis skip what that very node reads
/// (`AnalysisContext::definer_texts`: trusted by node identity, review B1).
#[derive(Debug, Clone)]
pub(crate) struct DefinerRead {
    /// The view's name parts (`spark_catalog`, namespace, name).
    pub(crate) name: Vec<String>,
    /// Whose rights read its query.
    pub(crate) definer: String,
    /// Its query, the views it reads inlined; `None` until inlined.
    pub(crate) query: Option<Arc<LogicalPlan>>,
    /// The catalog-store lookups of its text (not those of a definer view inside it), in order.
    pub(crate) reads: Vec<ReadCandidate>,
    /// Its entry among [`Expansion::views`].
    pub(crate) view: usize,
    /// The namespace current when it was created (`ViewDef::namespace`): its text is analyzed there.
    pub(crate) namespace: String,
    /// The SQL path frozen with it (`ViewDef::path`).
    pub(crate) path: Vec<Vec<String>>,
    /// The functions it bound when created or last altered (`ViewDef::functions`; empty for one
    /// stored without, so every stored call it reaches must be its definer's).
    pub(crate) bound: crate::store::BoundFunctions,
    /// The stored views its text inlines (not those inside a definer view in it), each checked
    /// against [`Self::bound`] and asked about its own reads as its owner (review r10 M10).
    pub(crate) views: Vec<ReachedView>,
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
    /// Whether [`Self::deferred`] arose inside a `SQL SECURITY DEFINER` view's text.
    pub(crate) deferred_in_definer: bool,
    /// The stored views inlined, inner views before the views that read them: their schema binding is checked
    /// against their analyzed queries (`view_schema::check_read`).
    pub(crate) stored_reads: Vec<crate::view_schema::StoredRead>,
    /// The views inlined, for the lineage side table (`AnalyzedPlan::views`, located once analyzed).
    pub(crate) views: Vec<krishiv_plan::analyzer::views::ViewExpansion>,
    /// The `SQL SECURITY DEFINER` views inlined, outer views first.
    pub(crate) definers: Vec<DefinerRead>,
    /// What the temporary views inlined kept ([`TempTable::View::keeps`]), from the entries expanded.
    pub(crate) keeps: Vec<Keep>,
    /// Every stored view inlined, at any depth (review r10 M10).
    pub(crate) reached: Vec<ReachedView>,
    /// The inlined query of each temporary view read (`AnalysisContext::temp_view_texts`).
    pub(crate) temp_texts: Vec<Arc<LogicalPlan>>,
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
    let plan = crate::run_site::detach_sources(pinner.run(plan)?);
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
            // A `system` table is the engine's, whatever the session names
            // its own tables: it is read as it is when the view is queried.
            if crate::system::is_system_table(name) {
                return Ok(Arc::clone(plan));
            }
            let key = last.to_lowercase();
            if !self.pins.contains_key(&key) {
                let pin = match self.catalog.get(last, self.case_sensitive) {
                    Some(table)
                        if matches!(
                            &*table,
                            TempTable::Memory { .. }
                                | TempTable::Files { .. }
                                | TempTable::Provider { .. }
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

/// A SQL path as Spark quotes it in `searchPath`: `[`a`.`b`, `c`]`.
pub(crate) fn render_search_path(path: &[Vec<String>]) -> String {
    let entries: Vec<String> = path.iter().map(|e| quote_parts(e)).collect();
    format!("[{}]", entries.join(", "))
}

fn quote_parts(parts: &[String]) -> String {
    parts.iter().map(|p| quote(p)).collect::<Vec<_>>().join(".")
}

/// Spark's search path, as analysis reports it for a missing table.
///
/// The last entry is the current namespace, as `USE` left it.
pub(crate) fn search_path_in(namespace: &str) -> String {
    format!(
        "[`system`.`builtin`, `system`.`session`, `{CATALOG}`.`{}`]",
        namespace.replace('`', "``")
    )
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
mod tests;

mod expand;
pub(crate) use expand::ReachedView;
use expand::*;
