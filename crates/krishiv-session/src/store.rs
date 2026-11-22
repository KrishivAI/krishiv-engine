// SPDX-License-Identifier: Apache-2.0
//! The permanent catalog: namespaces and tables, apart from the session's
//! temporary views ([`SessionCatalog`](crate::catalog::SessionCatalog)).
//!
//! The metadata of a table (its columns, provider, location and options)
//! lives in a [`CatalogStore`]; its rows are files in the table's
//! directory, so a table is read as a read of files and any worker that can
//! reach the location can read it. `CREATE TABLE` never creates an
//! in-memory table.
//!
//! P2b has one implementation, [`InMemoryCatalogStore`] (Spark's
//! `in-memory` catalog, the one the Spark 4.2 goldens were produced with).
//! Other catalogs implement the same trait and the commands do not change.
//! Every change bumps the store's epoch, which is part of what a cached
//! plan was analyzed against.
//!
//! Table data is ordered after its metadata so that metadata never points
//! at missing data: the directory is created before the table is
//! inserted, and a dropped managed table's directory is deleted after its
//! metadata is gone.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

use krishiv_common::error::{KrishivError, SchemaNotFound, TableOrViewAlreadyExists};
use krishiv_types::DataType;
use std::sync::Arc;

use crate::sql_functions::Entry;

/// The namespace that always exists and cannot be dropped.
pub(crate) const DEFAULT_NAMESPACE: &str = "default";

/// A column of a catalog table.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StoreColumn {
    pub(crate) name: String,
    pub(crate) data_type: DataType,
    pub(crate) nullable: bool,
    /// The `DEFAULT` expression as SQL text, analysed again at each
    /// `INSERT` (Spark keeps column defaults as text).
    pub(crate) default: Option<String>,
    pub(crate) comment: Option<String>,
    /// The Spark catalog string of a `CHAR(n)` / `VARCHAR(n)` column (`char(5)`, `array<varchar(2)>`; Spark's
    /// `__CHAR_VARCHAR_TYPE_STRING` field metadata). The column's type is `string`: the length lives here only,
    /// and the analyzer reads it for the write check and the padding rule (design P2-T001b).
    pub(crate) char_varchar: Option<String>,
    /// A metric view's column (P3 4.2-MV): whether it is a dimension or a measure and its
    /// expression as written (Spark's `metric_view.type` / `metric_view.expr` field metadata).
    pub(crate) metric: Option<MetricColumn>,
}

/// [`StoreColumn::metric`]: what a metric view's column is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetricColumn {
    /// A dimension or a measure.
    pub(crate) kind: MetricColumnKind,
    /// The YAML `expr`, as written.
    pub(crate) expr: String,
}

/// A metric view's column kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetricColumnKind {
    /// Computed per source row; the view's grouping keys.
    Dimension,
    /// An aggregate, read through `measure(..)`.
    Measure,
}

impl StoreColumn {
    /// `NOT_SUPPORTED_CHANGE_COLUMN`: `CHANGE COLUMN .. TYPE` to a type other than this column's own (a session
    /// catalog table takes only the identical type, a `CHAR(n)` length included). `to_declared` is the
    /// `CHAR(n)` / `VARCHAR(n)` string the new type was written with.
    pub(crate) fn change_type_error(
        &self,
        table: String,
        to: &DataType,
        to_declared: Option<&str>,
    ) -> KrishivError {
        let sql = |t: &DataType, declared: Option<&str>| {
            format!(
                "\"{}\"",
                declared.map_or_else(|| t.sql(), str::to_uppercase)
            )
        };
        let id = format!("`{}`", self.name.replace('`', "``"));
        krishiv_common::error::NotSupportedChangeColumn {
            table,
            origin_name: id.clone(),
            origin_type: sql(&self.data_type, self.char_varchar.as_deref()),
            new_name: id,
            new_type: sql(to, to_declared),
        }
        .into()
    }

    /// A nullable column as DDL declares it: `CHAR`/`VARCHAR` read as `STRING`, the length kept in `char_varchar`.
    pub(crate) fn declared(
        name: String,
        spec: &krishiv_sql::ast::DataTypeSpec,
        default: Option<String>,
        comment: Option<String>,
    ) -> Result<Self, KrishivError> {
        let (data_type, char_varchar) = spec.to_column_type()?;
        Ok(Self {
            name,
            data_type,
            nullable: true,
            default,
            comment,
            char_varchar,
            metric: None,
        })
    }

    /// A column with no default, comment or `CHAR`/`VARCHAR` length: what a query's output gives a table.
    pub(crate) fn plain(name: String, data_type: DataType, nullable: bool) -> Self {
        Self {
            name,
            data_type,
            nullable,
            default: None,
            comment: None,
            char_varchar: None,
            metric: None,
        }
    }
}

/// A permanent view: its query as SQL text, parsed again at each read (so it follows the tables it reads), and the
/// namespace that was current when it was created, in which its bare table names resolve when its frozen path does not hold them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ViewDef {
    pub(crate) sql: String,
    pub(crate) namespace: String,
    /// The SQL path when the view was created (Spark freezes it with the view): the bare names of its text
    /// resolve through it first, whatever `SET PATH` does later, and fall back to `namespace` when no schema
    /// of the path holds the name.
    pub(crate) path: Vec<Vec<String>>,
    /// The names the query's output had when the view was created (before a column list renamed them): a
    /// read projects the stored columns from the query's output by these names, so a base table that gained
    /// columns does not change the view (Spark's schema compensation).
    pub(crate) query_columns: Vec<String>,
    /// The spans in `sql` of the names the session substituted for `IDENTIFIER(..)` clauses when
    /// the view was created: a read marks their columns as the creating query did, so the view's
    /// output names stay the ones Spark gives (`c1.c2` for `IDENTIFIER('c1.c2')`).
    pub(crate) identifier_spans: Vec<krishiv_sql::ast::Span>,
    /// The query as written, its `IDENTIFIER(..)` clauses kept (`sql` has the names they
    /// evaluated to): Spark's `View Text`, `View Original Text` and `SHOW CREATE TABLE` print it.
    pub(crate) written: String,
    /// `WITH SCHEMA ...` as stored; `None` for a view created while `spark.sql.legacy.viewSchemaBindingMode` was
    /// off (Spark then stores no mode, and the view reads as `BINDING` once the feature is on).
    pub(crate) schema_mode: Option<ViewSchemaMode>,
    /// `SQL SECURITY DEFINER` (P3-T062 §2.4): the principal whose rights read the view's query (its
    /// creator, or whoever last replaced the query); `None` for `INVOKER`, the default, read with
    /// the rights of whoever reads the view.
    pub(crate) definer: Option<String>,
    /// `spark.sql.legacy.ctePrecedencePolicy` when the view was created or its query last
    /// replaced (Spark captures the session's SQL configs with a view, `view.sqlConfig.*`, and
    /// resolves its text under them): the view's own `WITH`s resolve under it, never the reader's
    /// (review r4 Y1). `None` (a view stored without it) reads as `CORRECTED`, Spark's default.
    pub(crate) cte_precedence: Option<krishiv_plan::analyzer::CtePrecedencePolicy>,
    /// For a `SQL SECURITY DEFINER` view (`Some`, `None` for `INVOKER`): its functions, bound
    /// when its query was created or last replaced (review r8 item 1, ADR 0048 A1). Every name
    /// its text calls bare (scalar or table) that no built-in answers was resolved then, through
    /// [`Self::path`] and [`Self::namespace`], and a read calls it by the name stored here, never
    /// resolving it again: a function created later earlier on the path does not change the view,
    /// a dropped one fails it. An invoker view's calls resolve at each read, as Spark's do.
    pub(crate) functions: Option<BoundFunctions>,
    /// The store's version of the view's text (review r10 M10, ADR 0048 A1): set by the store, from
    /// its epoch under its lock, when the view is created or replaced and when an entry carrying 0
    /// (`ALTER VIEW .. AS`) replaces it; any other change keeps it. A definer view whose text reaches
    /// this view pins it ([`BoundFunctions::views`]).
    pub(crate) revision: u64,
    /// The principal who wrote this version of the text (created, replaced or altered it); kept
    /// with [`Self::revision`].
    pub(crate) author: String,
    /// A SQL view, or a metric view whose [`Self::sql`] is its YAML body (P3 4.2-MV).
    pub(crate) kind: ViewKind,
}

impl ViewDef {
    /// Whether this is a metric view (Spark's table type `METRIC_VIEW`), whose [`Self::sql`] is YAML.
    pub(crate) const fn is_metric(&self) -> bool {
        matches!(self.kind, ViewKind::Metric(_))
    }
}

/// [`ViewDef::kind`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ViewKind {
    /// `CREATE VIEW .. AS query`.
    Sql,
    /// `CREATE VIEW .. WITH METRICS LANGUAGE YAML AS $$..$$` (Spark's table type `METRIC_VIEW`). Only
    /// `sql`, `written`, `namespace`, `path`, `revision` and `author` of the [`ViewDef`] apply; the
    /// YAML is read again at each read, as Spark does, under the parse options captured here.
    Metric(MetricViewDef),
}

/// [`ViewKind::Metric`]: what a metric view keeps beside its YAML text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MetricViewDef {
    /// The parse options when the view was created (Spark's captured `view.sqlConfig.*` apply only
    /// while its YAML and expressions are parsed, design note `P3-metric-views.md` §4).
    pub(crate) parser: krishiv_sql::ParserOptions,
    /// `spark.sql.ansi.enabled` when the view was created: the operators and casts its YAML
    /// expressions write (`+`, `/`, `CAST`, which Spark's parser builds with the mode it parses
    /// under) evaluate in it at every read; functions resolved at a read (`sum`) and the casts the
    /// analyzer inserts follow the reader's mode (design note `P3-metric-views.md`, golden results 1).
    pub(crate) ansi: bool,
}

/// [`ViewDef::functions`]: what a definer view bound when it was created or last altered.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct BoundFunctions {
    /// Each bare call's name as written in the view's text, with the catalog and schema of the
    /// stored function it resolved to (review r8 item 1).
    pub(crate) names: BTreeMap<String, Vec<String>>,
    /// Every stored function the view reached then (namespace, name as stored): those its text
    /// calls, bare or qualified, those its invoker views call and those their bodies call, with
    /// the [`StoredFunction::revision`] it had (review r9 item 1, ADR 0048 A1). A read may call a
    /// stored function only at that revision, or one its definer owns and last defined.
    pub(crate) versions: BTreeMap<(String, String), u64>,
    /// Every stored view the view's text reached then (namespace, name as stored): those it reads,
    /// those they read and those the bodies it calls read, with the [`ViewDef::revision`] it had
    /// (review r10 M10). A read may inline a stored view in its trusted text only at that
    /// revision, or one its definer owns and last defined.
    pub(crate) views: BTreeMap<(String, String), u64>,
}

/// How a permanent view follows the schema of its query (Spark's `ViewSchemaMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ViewSchemaMode {
    /// Columns are found by name and only up-cast to the stored types.
    Binding,
    /// Columns are found by name and cast to the stored types with ANSI casts.
    Compensation,
    /// Columns are found by name and keep the query's types, which the view stores.
    TypeEvolution,
    /// The view is the query's output, whose schema the view stores.
    Evolution,
}

impl ViewSchemaMode {
    /// The mode as `DESCRIBE` and `SHOW CREATE TABLE` print it.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Binding => "BINDING",
            Self::Compensation => "COMPENSATION",
            Self::TypeEvolution => "TYPE EVOLUTION",
            Self::Evolution => "EVOLUTION",
        }
    }
}

/// A table in the catalog.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CatalogTable {
    pub(crate) namespace: String,
    pub(crate) name: String,
    /// A managed table owns its directory: dropping the table deletes it.
    pub(crate) managed: bool,
    /// The file format: `parquet`, `csv`, `json` or `text`.
    pub(crate) provider: String,
    /// The provider as written (`USING CSV`), which `DESCRIBE` and
    /// `SHOW CREATE TABLE` print: Spark keeps its case.
    pub(crate) provider_written: String,
    /// The directory the table's files live in.
    pub(crate) location: PathBuf,
    pub(crate) columns: Vec<StoreColumn>,
    /// `OPTIONS (...)` as written, plus the options the engine adds so that
    /// the files it writes read back as the table's columns.
    pub(crate) options: BTreeMap<String, String>,
    pub(crate) properties: BTreeMap<String, String>,
    pub(crate) comment: Option<String>,
    pub(crate) created_ms: i64,
    /// The warehouse root the table was created under: `spark.sql.warehouse.dir` can change later, and the
    /// files must be deleted against the root they were created in.
    pub(crate) root: PathBuf,
    /// Set for a view: it has no provider, location or files.
    pub(crate) view: Option<ViewDef>,
    /// Partitioning, bucketing and the partitions the catalog tracks.
    pub(crate) layout: TableLayout,
    /// `ANALYZE TABLE .. FOR COLUMNS` results by column name (Spark's `CatalogStatistics.colStats`).
    pub(crate) column_stats: BTreeMap<String, ColumnStats>,
    /// Which entry of its name this is: set by the store when the table is created (a table dropped and created
    /// again, or replaced by `CREATE OR REPLACE`, gets a new one) and kept by [`CatalogStore::replace_table`]. A
    /// command that read the table and publishes later checks it, so a result never lands on a new table of the
    /// same name. Built entries carry 0 until the store sets it.
    pub(crate) incarnation: u64,
    /// Moves on every change of the table's data ([`CatalogTable::data_changed`]) and is never lowered by
    /// [`CatalogStore::replace_table`]. `ANALYZE` reads it with the table and publishes its statistics only if it
    /// has not moved, so statistics of rows a concurrent write replaced are never stored.
    pub(crate) data_version: u64,
    /// The entry's version for `CatalogWrite::alter_table`: taken from the store's epoch, under its lock, when
    /// the entry is created or replaced, so no two entries (or states of one) share it. A caller that read one
    /// stamp and finds another has been overtaken by a change, a drop and create, or an `ANALYZE`.
    pub(crate) stamp: u64,
}

impl CatalogTable {
    /// The table's version for `CatalogWrite::alter_table` (see [`stamp`](Self::stamp)).
    pub(crate) fn version(&self) -> krishiv_plan::TableVersion {
        krishiv_plan::TableVersion(i64::try_from(self.stamp).unwrap_or(i64::MAX))
    }

    /// Records that the table's data changed: its column statistics no longer describe it (`updateTableStats`).
    pub(crate) fn data_changed(&mut self) {
        self.column_stats.clear();
        self.data_version += 1;
    }

    /// Whether the table has partition columns.
    pub(crate) fn is_partitioned(&self) -> bool {
        !self.layout.partition_columns.is_empty()
    }

    /// The columns that are not partition columns (the files' columns), in table order.
    pub(crate) fn data_columns(&self) -> impl Iterator<Item = &StoreColumn> {
        self.columns
            .iter()
            .filter(|c| !self.layout.partition_columns.contains(&c.name))
    }

    /// The partition columns in partition order.
    pub(crate) fn partition_schema(&self) -> Vec<&StoreColumn> {
        self.layout
            .partition_columns
            .iter()
            .filter_map(|p| self.columns.iter().find(|c| &c.name == p))
            .collect()
    }
}

/// What `ANALYZE TABLE .. FOR COLUMNS` found for a column, as `DESCRIBE EXTENDED t col` prints it. A `min` or `max`
/// is `None` for a string or binary column and for one with no value; a timestamp's is microseconds since the
/// epoch, rendered in the session time zone of the `DESCRIBE`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ColumnStats {
    pub(crate) min: Option<String>,
    pub(crate) max: Option<String>,
    pub(crate) num_nulls: String,
    pub(crate) distinct_count: String,
    pub(crate) avg_col_len: String,
    pub(crate) max_col_len: String,
}

/// How a table's files are laid out (Spark's `partitionColumnNames`, `bucketSpec` and, for a partitioned
/// table, the partitions its catalog tracks: `tracksPartitionsInCatalog`). The partition columns are the last
/// columns of the table, in this order.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct TableLayout {
    /// The partition columns, as the table's columns spell them, in `PARTITIONED BY` order.
    pub(crate) partition_columns: Vec<String>,
    /// `CLUSTERED BY ... [SORTED BY ...] INTO n BUCKETS`.
    pub(crate) bucket: Option<BucketMeta>,
    /// The partitions, in the order they were added.
    pub(crate) partitions: Vec<CatalogPartition>,
}

/// A table's bucket specification, column names as the table spells them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BucketMeta {
    pub(crate) num_buckets: u64,
    pub(crate) columns: Vec<String>,
    pub(crate) sort_columns: Vec<String>,
}

/// A partition of a partitioned table (Spark's `CatalogTablePartition`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CatalogPartition {
    /// `(column, value)` in partition-column order; values are Spark's partition value strings.
    pub(crate) spec: Vec<(String, String)>,
    /// The partition's directory.
    pub(crate) location: PathBuf,
    /// `Partition Parameters` (file statistics a recovery gathered).
    pub(crate) parameters: BTreeMap<String, String>,
    pub(crate) created_ms: i64,
    /// `ANALYZE TABLE ... PARTITION` statistics: size in bytes and, unless `NOSCAN`, the row count.
    pub(crate) stats: Option<PartitionStats>,
}

/// Statistics of a partition (Spark's `CatalogStatistics` without column statistics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PartitionStats {
    pub(crate) size_in_bytes: i128,
    pub(crate) row_count: Option<u64>,
}

/// A namespace's metadata.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct NamespaceMeta {
    /// The schema's `DEFAULT COLLATION` (its canonical name), when schema-level collations are on.
    pub(crate) collation: Option<String>,
    pub(crate) comment: Option<String>,
    pub(crate) location: Option<String>,
    pub(crate) properties: BTreeMap<String, String>,
}

/// A SQL-bodied function of a namespace (`CREATE FUNCTION` without
/// `TEMPORARY`), with the statement that defined it.
#[derive(Clone)]
pub(crate) struct StoredFunction {
    pub(crate) namespace: String,
    pub(crate) entry: Entry,
    /// This definition's identity: the store's function version that defined it, set by
    /// [`StoredFunctions::with`], so every `CREATE [OR REPLACE] FUNCTION` gives a new one. A
    /// definer view pins it ([`BoundFunctions::versions`], review r9 item 1).
    pub(crate) revision: u64,
    /// The principal whose statement defined this version (review r9 item 1: a definer view
    /// follows only its definer's own replacements).
    pub(crate) author: String,
}

/// The stored functions at one version: replaced, never edited, so a query
/// reads one map whatever other statements run meanwhile.
#[derive(Clone, Default)]
pub(crate) struct StoredFunctions {
    version: u64,
    /// By namespace and lower-case name.
    entries: BTreeMap<(String, String), StoredFunction>,
}

impl StoredFunctions {
    /// The names of the functions stored in `namespace`, sorted.
    pub(crate) fn in_namespace(&self, namespace: &str) -> Vec<String> {
        self.entries
            .keys()
            .filter(|(ns, _)| ns == namespace)
            .map(|(_, n)| n.clone())
            .collect()
    }
}

impl std::fmt::Debug for StoredFunctions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredFunctions")
            .field("version", &self.version)
            .field("functions", &self.entries.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl StoredFunctions {
    /// The function `name` of `namespace`.
    pub(crate) fn get(&self, namespace: &str, name: &str) -> Option<&StoredFunction> {
        self.entries
            .get(&(namespace.to_owned(), name.to_lowercase()))
    }

    /// Every function, by namespace and name.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &StoredFunction> {
        self.entries.values()
    }

    /// How many changes the store has made to its functions.
    pub(crate) fn version(&self) -> u64 {
        self.version
    }

    /// This map with `function` defined (replacing any of its name).
    pub(crate) fn with(&self, mut function: StoredFunction) -> Self {
        let mut next = self.clone();
        next.version += 1;
        function.revision = next.version;
        next.entries.insert(
            (function.namespace.clone(), function.entry.def.name.clone()),
            function,
        );
        next
    }
}

/// Why a namespace could not be dropped.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DropNamespaceError {
    /// `default` is never dropped.
    Default,
    /// The namespace does not exist.
    Missing,
    /// It holds tables and the drop was a `RESTRICT`.
    NotEmpty,
    /// Its tables are not the ones the caller checked (`drop_namespace_matching`): nothing was dropped.
    Changed,
}

/// What `CREATE TABLE` does when the table exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CreateMode {
    ErrorIfExists,
    IfNotExists,
    /// Replaces an existing entry (views only: file tables cannot be replaced).
    Replace,
}

/// The contract of a permanent catalog (see the module docs).
pub(crate) trait CatalogStore: Send + Sync {
    /// Bumped on every change.
    fn epoch(&self) -> u64;
    /// Whether the namespace exists.
    fn has_namespace(&self, namespace: &str) -> bool;
    /// Adds a namespace; `false` when it exists already (nothing changes).
    fn create_namespace(&self, namespace: &str, meta: NamespaceMeta) -> bool;
    /// Changes a namespace's metadata with `change`; `false` when it does not exist.
    fn alter_namespace(&self, namespace: &str, change: &dyn Fn(&mut NamespaceMeta)) -> bool;
    /// Removes a namespace; with `cascade` also its tables, which are returned so that the caller can delete
    /// the managed ones' data after the metadata change.
    ///
    /// # Errors
    /// [`DropNamespaceError`] for `default`, a missing namespace, or a `RESTRICT` of a namespace with tables.
    fn drop_namespace(
        &self,
        namespace: &str,
        cascade: bool,
    ) -> Result<Vec<Arc<CatalogTable>>, DropNamespaceError> {
        self.drop_namespace_matching(namespace, cascade, None)
    }
    /// [`drop_namespace`](Self::drop_namespace) that, given the tables the caller checked before deleting
    /// anything, drops only if the namespace still holds exactly those entries.
    ///
    /// # Errors
    /// As `drop_namespace`, and [`DropNamespaceError::Changed`] when its tables are not `expected`.
    fn drop_namespace_matching(
        &self,
        namespace: &str,
        cascade: bool,
        expected: Option<&[Arc<CatalogTable>]>,
    ) -> Result<Vec<Arc<CatalogTable>>, DropNamespaceError>;
    /// Every namespace with its metadata, by name.
    fn namespaces(&self) -> Vec<(String, NamespaceMeta)>;
    /// The tables and views of `namespace`, by name.
    fn tables_in(&self, namespace: &str) -> Vec<Arc<CatalogTable>>;
    /// The table `namespace.name`.
    fn table(&self, namespace: &str, name: &str) -> Option<Arc<CatalogTable>>;
    /// Adds `table`; `false` when nothing was added (`IfNotExists` on an existing table).
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_ALREADY_EXISTS` for `ErrorIfExists`.
    fn create_table(&self, table: CatalogTable, mode: CreateMode) -> Result<bool, KrishivError>;
    /// Replaces the entry `expected` was read as with `table` and returns the entry it stored (with its new
    /// stamp), taken under the same lock; `None` when that entry was dropped or replaced since, so a concurrent
    /// `DROP` or `ALTER` is never undone by a stale copy.
    fn replace_table_entry(
        &self,
        expected: &Arc<CatalogTable>,
        table: CatalogTable,
    ) -> Option<Arc<CatalogTable>>;
    /// [`replace_table_entry`](Self::replace_table_entry) as a yes or no.
    fn replace_table(&self, expected: &Arc<CatalogTable>, table: CatalogTable) -> bool {
        self.replace_table_entry(expected, table).is_some()
    }
    /// Removes the table, returning it.
    fn drop_table(&self, namespace: &str, name: &str) -> Option<Arc<CatalogTable>>;
    /// Removes the entry `expected` was read as, returning it; `None` when that entry was dropped or replaced
    /// since, so a delete checked against one entry never removes (or deletes the files of) another.
    fn drop_table_entry(&self, expected: &Arc<CatalogTable>) -> Option<Arc<CatalogTable>>;
    /// Renames the entry `expected` was read as to `renamed` (same namespace, another name), under one lock: the
    /// old name is gone and the new one is there, or nothing changed. The table keeps its incarnation and gets a
    /// new stamp. `Ok(None)` when that entry was dropped or replaced since.
    ///
    /// # Errors
    /// `TABLE_OR_VIEW_ALREADY_EXISTS` when the new name is taken, `SCHEMA_NOT_FOUND` when its namespace is gone.
    fn rename_table_entry(
        &self,
        expected: &Arc<CatalogTable>,
        renamed: CatalogTable,
    ) -> Result<Option<Arc<CatalogTable>>, KrishivError>;
    /// The stored functions as of now.
    fn functions(&self) -> Arc<StoredFunctions>;
    /// Defines `function`, replacing one of its name. `expected` is the
    /// version of [`functions`](Self::functions) the caller checked the
    /// definition against: when another statement changed the functions
    /// meanwhile nothing is defined and `false` is returned.
    fn define_function(&self, function: StoredFunction, expected: u64) -> bool;
    /// Removes the function `namespace.name`; whether it existed.
    fn drop_function(&self, namespace: &str, name: &str) -> bool;
}

/// Spark's `in-memory` catalog: metadata in memory, one per engine runtime,
/// shared by its sessions (design P2-E1; Spark's `SharedState`).
#[derive(Debug)]
pub(crate) struct InMemoryCatalogStore {
    state: Mutex<StoreState>,
    epoch: AtomicU64,
    /// Where a table change reaches every cache of the runtime: its result
    /// cache and each session's relation cache.
    hub: Arc<crate::cache::Invalidation>,
    /// One read/write lock per partitioned table's location, held exclusively by a command that moves or deletes
    /// its partition directories or log entries (`RENAME` / `DROP PARTITION`, `MSCK REPAIR`, `TRUNCATE`) from
    /// before it reads the entry to after its catalog swap, and shared by an `INSERT` into it from its write to
    /// its partition registration: no two such commands, and no such command and a write, interleave their
    /// directory moves, log entries and catalog swaps on one table. Writes never wait for each other, and other
    /// tables never wait (entries are weak: a lock lives while a statement holds it).
    partition_locks:
        Mutex<std::collections::HashMap<PathBuf, std::sync::Weak<tokio::sync::RwLock<()>>>>,
}

#[derive(Debug)]
struct StoreState {
    namespaces: BTreeMap<String, NamespaceMeta>,
    tables: BTreeMap<(String, String), Arc<CatalogTable>>,
    functions: Arc<StoredFunctions>,
}

impl Default for InMemoryCatalogStore {
    fn default() -> Self {
        Self::with_hub(crate::cache::Invalidation::new(Arc::clone(
            crate::result_cache::global(),
        )))
    }
}

impl InMemoryCatalogStore {
    /// An empty store whose changes go through `hub`.
    pub(crate) fn with_hub(hub: Arc<crate::cache::Invalidation>) -> Self {
        Self {
            hub,
            state: Mutex::new(StoreState {
                namespaces: BTreeMap::from([(
                    DEFAULT_NAMESPACE.to_owned(),
                    NamespaceMeta::default(),
                )]),
                tables: BTreeMap::new(),
                functions: Arc::new(StoredFunctions::default()),
            }),
            epoch: AtomicU64::new(0),
            partition_locks: Mutex::default(),
        }
    }

    /// The partition lock of the table at `location` (see the field).
    fn partition_lock(&self, location: &Path) -> Arc<tokio::sync::RwLock<()>> {
        let mut locks = self
            .partition_locks
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        locks.retain(|_, lock| lock.strong_count() > 0);
        if let Some(lock) = locks.get(location).and_then(std::sync::Weak::upgrade) {
            return lock;
        }
        let lock = Arc::new(tokio::sync::RwLock::new(()));
        locks.insert(location.to_path_buf(), Arc::downgrade(&lock));
        lock
    }

    /// The exclusive hold of a command that moves or deletes the partition directories of the table at
    /// `location` (see the field).
    pub(crate) async fn partition_ddl(
        &self,
        location: &Path,
    ) -> tokio::sync::OwnedRwLockWriteGuard<()> {
        self.partition_lock(location).write_owned().await
    }

    /// The shared hold of a write into the partitioned table at `location` (see the field). Taken before the
    /// write's admission: admission never takes this lock, so the order cannot deadlock.
    pub(crate) async fn partition_write(
        &self,
        location: &Path,
    ) -> tokio::sync::OwnedRwLockReadGuard<()> {
        self.partition_lock(location).read_owned().await
    }

    /// Marks a change to a table's data (an `INSERT`, a `TRUNCATE`): cached
    /// analyses are stale though the metadata is the same.
    pub(crate) fn bump(&self) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, StoreState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl CatalogStore for InMemoryCatalogStore {
    fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    fn has_namespace(&self, namespace: &str) -> bool {
        self.lock().namespaces.contains_key(namespace)
    }

    fn namespaces(&self) -> Vec<(String, NamespaceMeta)> {
        self.lock()
            .namespaces
            .iter()
            .map(|(n, m)| (n.clone(), m.clone()))
            .collect()
    }

    fn tables_in(&self, namespace: &str) -> Vec<Arc<CatalogTable>> {
        self.lock()
            .tables
            .iter()
            .filter(|((ns, _), _)| ns == namespace)
            .map(|(_, t)| Arc::clone(t))
            .collect()
    }

    fn create_namespace(&self, namespace: &str, meta: NamespaceMeta) -> bool {
        let mut state = self.lock();
        if state.namespaces.contains_key(namespace) {
            return false;
        }
        state.namespaces.insert(namespace.to_owned(), meta);
        self.epoch.fetch_add(1, Ordering::AcqRel);
        true
    }

    fn alter_namespace(&self, namespace: &str, change: &dyn Fn(&mut NamespaceMeta)) -> bool {
        let mut state = self.lock();
        let Some(meta) = state.namespaces.get_mut(namespace) else {
            return false;
        };
        change(meta);
        self.epoch.fetch_add(1, Ordering::AcqRel);
        true
    }

    fn drop_namespace_matching(
        &self,
        namespace: &str,
        cascade: bool,
        expected: Option<&[Arc<CatalogTable>]>,
    ) -> Result<Vec<Arc<CatalogTable>>, DropNamespaceError> {
        if namespace == DEFAULT_NAMESPACE {
            return Err(DropNamespaceError::Default);
        }
        let mut state = self.lock();
        if !state.namespaces.contains_key(namespace) {
            return Err(DropNamespaceError::Missing);
        }
        let held: Vec<(String, String)> = state
            .tables
            .keys()
            .filter(|(ns, _)| ns == namespace)
            .cloned()
            .collect();
        let has_functions = state.functions.iter().any(|f| f.namespace == namespace);
        if (!held.is_empty() || has_functions) && !cascade {
            return Err(DropNamespaceError::NotEmpty);
        }
        if let Some(expected) = expected {
            let same = held.len() == expected.len()
                && expected.iter().all(|e| {
                    state
                        .tables
                        .get(&(e.namespace.clone(), e.name.clone()))
                        .is_some_and(|t| Arc::ptr_eq(t, e))
                });
            if !same {
                return Err(DropNamespaceError::Changed);
            }
        }
        if has_functions {
            let mut next = StoredFunctions::clone(&state.functions);
            next.entries.retain(|(ns, _), _| ns != namespace);
            next.version += 1;
            state.functions = Arc::new(next);
        }
        let dropped = held
            .into_iter()
            .filter_map(|key| state.tables.remove(&key))
            .collect();
        state.namespaces.remove(namespace);
        self.epoch.fetch_add(1, Ordering::AcqRel);
        // What the caches hold over a dropped table is stale, for every caller of this drop.
        let dropped: Vec<Arc<CatalogTable>> = dropped;
        for t in &dropped {
            self.hub
                .invalidate_table(&format!("{}.{}", t.namespace, t.name));
        }
        Ok(dropped)
    }

    fn table(&self, namespace: &str, name: &str) -> Option<Arc<CatalogTable>> {
        self.lock()
            .tables
            .get(&(namespace.to_owned(), name.to_owned()))
            .cloned()
    }

    fn create_table(&self, table: CatalogTable, mode: CreateMode) -> Result<bool, KrishivError> {
        let key = (table.namespace.clone(), table.name.clone());
        let mut state = self.lock();
        // Checked under the lock the insert takes, so a namespace dropped meanwhile cannot be left holding a table.
        if !state.namespaces.contains_key(&table.namespace) {
            return Err(SchemaNotFound {
                schema_name: format!("`{}`", table.namespace),
            }
            .into());
        }
        match (state.tables.contains_key(&key), mode) {
            (true, CreateMode::ErrorIfExists) => {
                return Err(TableOrViewAlreadyExists {
                    relation_name: format!(
                        "`spark_catalog`.`{}`.`{}`",
                        table.namespace, table.name
                    ),
                }
                .into());
            }
            (true, CreateMode::IfNotExists) => return Ok(false),
            (true, CreateMode::Replace) | (false, _) => {}
        }
        let changed = format!("{}.{}", key.0, key.1);
        // The epoch moves under the state lock on every create, so each created entry gets its own value.
        let mut table = table;
        table.incarnation = self.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        table.stamp = table.incarnation;
        if let Some(view) = &mut table.view {
            view.revision = table.incarnation;
        }
        state.tables.insert(key, Arc::new(table));
        // What the result cache holds for this table is stale (it also covers the results
        // computed between a table's data being written and its partitions being registered).
        self.hub.invalidate_table(&changed);
        Ok(true)
    }

    fn replace_table_entry(
        &self,
        expected: &Arc<CatalogTable>,
        table: CatalogTable,
    ) -> Option<Arc<CatalogTable>> {
        let key = (table.namespace.clone(), table.name.clone());
        let mut state = self.lock();
        match state.tables.get(&key) {
            Some(current) if Arc::ptr_eq(current, expected) => {}
            _ => return None,
        }
        let changed = format!("{}.{}", key.0, key.1);
        // The same table, changed: it keeps its incarnation.
        let mut table = table;
        table.incarnation = expected.incarnation;
        table.data_version = table.data_version.max(expected.data_version);
        table.stamp = self.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        // A new text (`ALTER VIEW .. AS` stores 0) is a new version; any other change keeps the
        // version and author the entry had (review r10 M10).
        if let Some(view) = &mut table.view {
            match &expected.view {
                Some(old) if view.revision != 0 => {
                    view.revision = old.revision;
                    view.author.clone_from(&old.author);
                }
                _ => view.revision = table.stamp,
            }
        }
        let entry = Arc::new(table);
        state.tables.insert(key, Arc::clone(&entry));
        self.hub.invalidate_table(&changed);
        Some(entry)
    }

    fn drop_table(&self, namespace: &str, name: &str) -> Option<Arc<CatalogTable>> {
        let removed = self
            .lock()
            .tables
            .remove(&(namespace.to_owned(), name.to_owned()));
        if removed.is_some() {
            self.epoch.fetch_add(1, Ordering::AcqRel);
            self.hub.invalidate_table(&format!("{namespace}.{name}"));
        }
        removed
    }

    fn drop_table_entry(&self, expected: &Arc<CatalogTable>) -> Option<Arc<CatalogTable>> {
        let key = (expected.namespace.clone(), expected.name.clone());
        let mut state = self.lock();
        if !state
            .tables
            .get(&key)
            .is_some_and(|t| Arc::ptr_eq(t, expected))
        {
            return None;
        }
        let removed = state.tables.remove(&key);
        self.epoch.fetch_add(1, Ordering::AcqRel);
        self.hub.invalidate_table(&format!("{}.{}", key.0, key.1));
        removed
    }

    fn rename_table_entry(
        &self,
        expected: &Arc<CatalogTable>,
        renamed: CatalogTable,
    ) -> Result<Option<Arc<CatalogTable>>, KrishivError> {
        let old = (expected.namespace.clone(), expected.name.clone());
        let new = (renamed.namespace.clone(), renamed.name.clone());
        let mut state = self.lock();
        if !state
            .tables
            .get(&old)
            .is_some_and(|t| Arc::ptr_eq(t, expected))
        {
            return Ok(None);
        }
        if !state.namespaces.contains_key(&new.0) {
            return Err(SchemaNotFound {
                schema_name: format!("`{}`", new.0),
            }
            .into());
        }
        if state.tables.contains_key(&new) {
            return Err(TableOrViewAlreadyExists {
                relation_name: format!("`spark_catalog`.`{}`.`{}`", new.0, new.1),
            }
            .into());
        }
        let mut renamed = renamed;
        renamed.incarnation = expected.incarnation;
        renamed.data_version = renamed.data_version.max(expected.data_version);
        renamed.stamp = self.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        if let (Some(view), Some(old)) = (&mut renamed.view, &expected.view) {
            view.revision = old.revision;
            view.author.clone_from(&old.author);
        }
        let entry = Arc::new(renamed);
        state.tables.remove(&old);
        state.tables.insert(new.clone(), Arc::clone(&entry));
        self.hub.invalidate_table(&format!("{}.{}", old.0, old.1));
        self.hub.invalidate_table(&format!("{}.{}", new.0, new.1));
        Ok(Some(entry))
    }

    fn functions(&self) -> Arc<StoredFunctions> {
        Arc::clone(&self.lock().functions)
    }

    fn define_function(&self, function: StoredFunction, expected: u64) -> bool {
        let mut state = self.lock();
        if state.functions.version() != expected
            || !state.namespaces.contains_key(&function.namespace)
        {
            return false;
        }
        state.functions = Arc::new(state.functions.with(function));
        self.epoch.fetch_add(1, Ordering::AcqRel);
        true
    }

    fn drop_function(&self, namespace: &str, name: &str) -> bool {
        let mut state = self.lock();
        if state.functions.get(namespace, name).is_none() {
            return false;
        }
        let mut next = StoredFunctions::clone(&state.functions);
        next.entries
            .remove(&(namespace.to_owned(), name.to_lowercase()));
        next.version += 1;
        state.functions = Arc::new(next);
        self.epoch.fetch_add(1, Ordering::AcqRel);
        true
    }
}

/// Where managed tables live: `spark.sql.warehouse.dir` when set, else a
/// directory of the runtime's own, removed when the runtime ends (the
/// tables of an in-memory catalog do not outlive it, and a leftover
/// directory would make the next runtime's `CREATE TABLE` fail on an
/// existing location).
#[derive(Debug, Default)]
pub(crate) struct Warehouse {
    own: Mutex<Option<OwnDir>>,
}

/// The runtime's own warehouse directory: its name, chosen once, and whether it was made.
#[derive(Debug)]
struct OwnDir {
    dir: PathBuf,
    made: bool,
}

impl Warehouse {
    /// The warehouse directory for `configured`, creating the runtime's own
    /// on first use.
    ///
    /// # Errors
    /// The I/O error creating the directory.
    pub(crate) fn root(&self, configured: Option<&str>) -> std::io::Result<PathBuf> {
        if let Some(dir) = configured_dir(configured) {
            return Ok(dir);
        }
        let mut own = self.own.lock().unwrap_or_else(PoisonError::into_inner);
        let own = own.get_or_insert_with(OwnDir::named);
        if !own.made {
            std::fs::create_dir_all(&own.dir)?;
            own.made = true;
        }
        Ok(own.dir.clone())
    }

    /// The path [`Self::root`] gives for `configured`, without making the runtime's own directory
    /// (its name is chosen now, so a later `root` gives the same path): what the analyze API lists.
    pub(crate) fn path(&self, configured: Option<&str>) -> PathBuf {
        if let Some(dir) = configured_dir(configured) {
            return dir;
        }
        let mut own = self.own.lock().unwrap_or_else(PoisonError::into_inner);
        own.get_or_insert_with(OwnDir::named).dir.clone()
    }
}

/// `spark.sql.warehouse.dir` as a path, when set and not empty.
fn configured_dir(configured: Option<&str>) -> Option<PathBuf> {
    let dir = configured.filter(|d| !d.is_empty())?;
    Some(PathBuf::from(dir.strip_prefix("file:").unwrap_or(dir)))
}

impl OwnDir {
    /// A new name under the temporary directory, not made yet.
    fn named() -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        Self {
            dir: std::env::temp_dir()
                .join(format!("krishiv-warehouse-{}-{stamp}", std::process::id())),
            made: false,
        }
    }
}

impl Drop for Warehouse {
    fn drop(&mut self) {
        let own = self.own.get_mut().unwrap_or_else(PoisonError::into_inner);
        if let Some(OwnDir { dir, made: true }) = own.take() {
            // Best effort: a leftover directory is the temp directory's to clean.
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

/// Spark's managed location of `namespace.name` under `warehouse`:
/// `<warehouse>/<table>` in `default`, else `<warehouse>/<namespace>.db/<table>`.
pub(crate) fn managed_location(warehouse: &Path, namespace: &str, name: &str) -> PathBuf {
    if namespace == DEFAULT_NAMESPACE {
        warehouse.join(name)
    } else {
        warehouse.join(format!("{namespace}.db")).join(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(ns: &str, name: &str) -> CatalogTable {
        CatalogTable {
            namespace: ns.to_owned(),
            name: name.to_owned(),
            managed: true,
            provider: "parquet".to_owned(),
            provider_written: "parquet".to_owned(),
            location: PathBuf::from("/w").join(name),
            columns: vec![StoreColumn {
                name: "a".to_owned(),
                data_type: DataType::Integer,
                nullable: true,
                default: None,
                comment: None,
                char_varchar: None,
                metric: None,
            }],
            options: BTreeMap::new(),
            properties: BTreeMap::new(),
            comment: None,
            created_ms: 0,
            root: PathBuf::new(),
            view: None,
            layout: TableLayout::default(),
            column_stats: BTreeMap::new(),
            incarnation: 0,
            data_version: 0,
            stamp: 0,
        }
    }

    #[test]
    fn a_table_cannot_be_created_in_a_namespace_that_is_not_there() {
        let s = InMemoryCatalogStore::default();
        let e = s
            .create_table(table("nope", "t"), CreateMode::ErrorIfExists)
            .unwrap_err();
        assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::SchemaNotFound
        );
        assert!(s.table("nope", "t").is_none());
        // Dropped meanwhile: the namespace is checked where the insert happens.
        s.create_namespace("db", NamespaceMeta::default());
        s.drop_namespace("db", false).unwrap();
        assert!(
            s.create_table(table("db", "t"), CreateMode::ErrorIfExists)
                .is_err()
        );
    }

    #[test]
    fn the_store_takes_a_table_in_a_collated_namespace_so_a_rename_can_always_restore_it() {
        // `Session::rename_table` drops the source entry and creates the new one, and puts the source
        // back if that fails: neither create may be refused by the store. The SQL create refuses a *new*
        // table in a collated schema before it gets here.
        let s = InMemoryCatalogStore::default();
        s.create_namespace(
            "ci",
            NamespaceMeta {
                collation: Some("UTF8_LCASE".into()),
                ..NamespaceMeta::default()
            },
        );
        assert!(
            s.create_table(table("ci", "t"), CreateMode::ErrorIfExists)
                .unwrap()
        );
        assert!(s.table("ci", "t").is_some());
    }

    #[test]
    fn a_removal_checked_against_one_entry_never_removes_another() {
        let s = InMemoryCatalogStore::default();
        s.create_table(table("default", "t"), CreateMode::ErrorIfExists)
            .unwrap();
        let read = s.table("default", "t").unwrap();
        s.drop_table("default", "t");
        s.create_table(table("default", "t"), CreateMode::ErrorIfExists)
            .unwrap();
        // `read` is the dropped incarnation: the new table stays.
        assert!(s.drop_table_entry(&read).is_none());
        assert!(s.table("default", "t").is_some());
        let current = s.table("default", "t").unwrap();
        let epoch = s.epoch();
        assert!(s.drop_table_entry(&current).is_some());
        assert!(s.table("default", "t").is_none());
        assert!(s.epoch() > epoch);
        assert!(s.drop_table_entry(&current).is_none());
    }

    #[test]
    fn a_namespace_drop_bound_to_the_tables_checked_refuses_a_changed_set() {
        let s = InMemoryCatalogStore::default();
        s.create_namespace("db", NamespaceMeta::default());
        s.create_table(table("db", "a"), CreateMode::ErrorIfExists)
            .unwrap();
        let checked = s.tables_in("db");
        // A table appears (or is replaced) after the check: nothing is dropped.
        s.create_table(table("db", "b"), CreateMode::ErrorIfExists)
            .unwrap();
        assert!(matches!(
            s.drop_namespace_matching("db", true, Some(&checked)),
            Err(DropNamespaceError::Changed)
        ));
        assert!(s.has_namespace("db") && s.table("db", "b").is_some());
        let again = s.tables_in("db");
        let replaced = s.table("db", "a").unwrap();
        assert!(s.replace_table(&replaced, table("db", "a")));
        assert!(matches!(
            s.drop_namespace_matching("db", true, Some(&again)),
            Err(DropNamespaceError::Changed)
        ));
        let fresh = s.tables_in("db");
        let dropped = s.drop_namespace_matching("db", true, Some(&fresh)).unwrap();
        assert_eq!(dropped.len(), 2);
        assert!(!s.has_namespace("db"));
    }

    /// Review r10 M10: a view's text version moves on a create, a replace and a new text
    /// (`ALTER VIEW .. AS` stores 0), never back, and any other change or a rename keeps it and
    /// its author.
    #[test]
    fn a_views_revision_moves_only_with_a_new_text() {
        let view = |author: &str| {
            let mut t = table("default", "v");
            t.view = Some(ViewDef {
                sql: "SELECT 1".into(),
                namespace: "default".into(),
                path: Vec::new(),
                query_columns: Vec::new(),
                identifier_spans: Vec::new(),
                written: "SELECT 1".into(),
                schema_mode: None,
                definer: None,
                cte_precedence: None,
                functions: None,
                revision: 0,
                author: author.into(),
                kind: ViewKind::Sql,
            });
            t
        };
        let revision = |s: &InMemoryCatalogStore, name: &str| {
            let t = s.table("default", name).unwrap();
            let v = t.view.as_ref().unwrap();
            (v.revision, v.author.clone())
        };
        let s = InMemoryCatalogStore::default();
        s.create_table(view("ana"), CreateMode::ErrorIfExists)
            .unwrap();
        let (first, _) = revision(&s, "v");
        assert!(first > 0);
        // Another change (a property) keeps the version and author, whatever the copy says.
        let current = s.table("default", "v").unwrap();
        let mut changed = (*current).clone();
        changed.properties.insert("k".into(), "v".into());
        if let Some(v) = &mut changed.view {
            v.revision = 1_000;
            v.author = "bob".into();
        }
        assert!(s.replace_table(&current, changed));
        assert_eq!(revision(&s, "v"), (first, "ana".to_owned()));
        // A new text is a new version, by its author.
        let current = s.table("default", "v").unwrap();
        assert!(s.replace_table(&current, view("bob")));
        let (second, author) = revision(&s, "v");
        assert!(second > first && author == "bob");
        // `CREATE OR REPLACE` is a new version too; a rename keeps it.
        s.create_table(view("ana"), CreateMode::Replace).unwrap();
        let (third, _) = revision(&s, "v");
        assert!(third > second);
        let current = s.table("default", "v").unwrap();
        let mut renamed = (*current).clone();
        renamed.name = "w".into();
        s.rename_table_entry(&current, renamed).unwrap();
        assert_eq!(revision(&s, "w"), (third, "ana".to_owned()));
    }

    #[test]
    fn replace_table_refuses_a_stale_copy() {
        let s = InMemoryCatalogStore::default();
        s.create_table(table("default", "t"), CreateMode::ErrorIfExists)
            .unwrap();
        let read = s.table("default", "t").unwrap();
        let epoch = s.epoch();
        assert!(s.replace_table(&read, table("default", "t")));
        assert_eq!(s.epoch(), epoch + 1);
        // A replacement never lowers the data version, even when it was built fresh (version 0).
        let current = s.table("default", "t").unwrap();
        let mut changed = (*current).clone();
        changed.data_changed();
        assert!(s.replace_table(&current, changed));
        let current = s.table("default", "t").unwrap();
        assert_eq!(current.data_version, 1);
        assert!(s.replace_table(&current, table("default", "t")));
        assert_eq!(s.table("default", "t").unwrap().data_version, 1);
        // `read` is stale now.
        assert!(!s.replace_table(&read, table("default", "t")));
        // And so is a copy of a dropped table: it is not recreated.
        let current = s.table("default", "t").unwrap();
        s.drop_table("default", "t");
        assert!(!s.replace_table(&current, table("default", "t")));
        assert!(s.table("default", "t").is_none());
    }

    #[test]
    fn create_and_drop_bump_the_epoch() {
        let s = InMemoryCatalogStore::default();
        assert!(s.has_namespace("default"));
        assert!(!s.has_namespace("other"));
        assert_eq!(s.epoch(), 0);
        assert!(
            s.create_table(table("default", "t"), CreateMode::ErrorIfExists)
                .unwrap()
        );
        assert_eq!(s.epoch(), 1);
        let err = s
            .create_table(table("default", "t"), CreateMode::ErrorIfExists)
            .unwrap_err();
        assert_eq!(err.error_class().name(), "TABLE_OR_VIEW_ALREADY_EXISTS");
        assert_eq!(s.epoch(), 1);
        assert!(
            !s.create_table(table("default", "t"), CreateMode::IfNotExists)
                .unwrap()
        );
        assert_eq!(s.epoch(), 1);
        assert!(s.table("default", "t").is_some());
        assert!(s.table("default", "u").is_none());
        s.bump();
        assert_eq!(s.epoch(), 2);
        assert!(s.drop_table("default", "t").is_some());
        assert_eq!(s.epoch(), 3);
        assert!(s.drop_table("default", "t").is_none());
        assert_eq!(s.epoch(), 3);
    }

    #[test]
    fn managed_locations_follow_sparks_layout() {
        let w = Path::new("/w");
        assert_eq!(managed_location(w, "default", "t"), Path::new("/w/t"));
        assert_eq!(managed_location(w, "db", "t"), Path::new("/w/db.db/t"));
    }

    #[test]
    fn the_warehouse_is_the_configured_dir_or_a_session_dir_removed_on_drop() {
        let w = Warehouse::default();
        assert_eq!(
            w.root(Some("file:/data/wh")).unwrap(),
            Path::new("/data/wh")
        );
        let own = w.root(None).unwrap();
        assert!(own.is_dir());
        assert_eq!(w.root(Some("")).unwrap(), own);
        assert_eq!(w.path(None), own);
        drop(w);
        assert!(!own.exists());
        // `path` names the directory `root` makes later, without making it.
        let w = Warehouse::default();
        assert_eq!(w.path(Some("/data/wh")), Path::new("/data/wh"));
        let named = w.path(None);
        assert!(!named.exists());
        assert_eq!(w.root(None).unwrap(), named);
        assert!(named.is_dir());
        drop(w);
        assert!(!named.exists());
    }
}
