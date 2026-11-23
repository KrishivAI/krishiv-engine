// SPDX-License-Identifier: Apache-2.0
//! Row-level changes to a table: `DELETE`, `UPDATE` and `MERGE` (design `P3-PR-DML-row-level.md`, T021 note §6.1).
//!
//! A table that can change rows offers a [`RowLevelTable`] for an operation
//! ([`TableProvider::row_level`](super::TableProvider::row_level)). The session plans the change
//! as two queries over one pinned snapshot: one finds the files holding a row the condition hits
//! (scan rows carry `_krishiv_row_file` and `_krishiv_row_index`), the other rewrites those files
//! into the table's [`rewriter`](RowLevelTable::rewriter), whose finalize commits the swap. A
//! table without one keeps Spark's v1 refusal.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_io::{ColumnPath, FilterId};
use krishiv_plan::Expr;
use krishiv_types::batch::Batch;

use crate::context::TaskContext;
use crate::error::Result;
use crate::operator::{GlobalState, LocalState, Sink, Source};
use crate::operators::{
    FilterPushdown, PushedFilter, ROW_FILE_COLUMN, ROW_POSITION_COLUMN, ScanPushdown, ScanPushdowns,
};

/// The statement that changes rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RowLevelOperation {
    /// `DELETE FROM`.
    Delete,
    /// `UPDATE ... SET`.
    Update,
    /// `MERGE INTO`.
    Merge,
}

impl RowLevelOperation {
    /// The statement's name as Spark's errors print it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Delete => "DELETE",
            Self::Update => "UPDATE",
            Self::Merge => "MERGE INTO",
        }
    }
}

/// How `MERGE` treats a target row that more than one source row matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CardinalityRule {
    /// Delta's `DELTA_MULTIPLE_SOURCE_ROW_MATCHING_TARGET_ROW_IN_MERGE`, waived when the only
    /// matched clause is one unconditional `DELETE`.
    DeltaWithUnconditionalDeleteExemption,
    /// Spark's `MERGE_CARDINALITY_VIOLATION`, what Iceberg tables raise in Spark.
    SparkMergeCardinality,
}

/// What a [`RowLevelTable`] can do. Copy-on-write is always offered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct RowLevelCapabilities {
    /// Changed rows can be recorded as deletes (Delta deletion vectors, Iceberg position deletes)
    /// instead of rewriting their files.
    pub merge_on_read: bool,
    /// `MERGE`'s multiple-match check.
    pub cardinality: CardinalityRule,
}

impl RowLevelCapabilities {
    /// Copy-on-write only, with `cardinality` for `MERGE`.
    #[must_use]
    pub fn copy_on_write(cardinality: CardinalityRule) -> Self {
        Self {
            merge_on_read: false,
            cardinality,
        }
    }
}

/// The counts of one row-level statement, for its result rows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct RowLevelCounts {
    /// Rows the statement deleted, updated or inserted.
    pub affected: u64,
    /// Rows a `MERGE` updated.
    pub updated: u64,
    /// Rows a `MERGE` deleted.
    pub deleted: u64,
    /// Rows a `MERGE` inserted.
    pub inserted: u64,
}

impl RowLevelCounts {
    /// The counts of a `DELETE` or `UPDATE` that changed `affected` rows.
    #[must_use]
    pub fn affected(affected: u64) -> Self {
        Self {
            affected,
            ..Self::default()
        }
    }
}

/// What a copy-on-write rewrite asks of a table.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RowLevelWrite {
    /// The statement.
    pub op: RowLevelOperation,
    /// The target as Spark's errors print it.
    pub table_name: String,
    /// The schema of the rows the sink receives: the table's columns.
    pub input_schema: SchemaRef,
    /// The files the rewrite replaces: values of `_krishiv_row_file` (the scan's encoded file
    /// references), each once.
    pub touched: Vec<Vec<u8>>,
    /// The statement's condition, for the table's conflict check (`None`: every row was read).
    pub read_filter: Option<Arc<Expr>>,
    /// The statement's options (`WITH (...)`), keys as written.
    pub options: BTreeMap<String, String>,
    /// The idempotency key of the commit.
    pub write_id: Option<String>,
}

impl RowLevelWrite {
    /// A rewrite of `touched` in `table_name` by `op`, with no options and no write id.
    #[must_use]
    pub fn new(
        op: RowLevelOperation,
        table_name: impl Into<String>,
        input_schema: SchemaRef,
        touched: Vec<Vec<u8>>,
        read_filter: Option<Arc<Expr>>,
    ) -> Self {
        Self {
            op,
            table_name: table_name.into(),
            input_schema,
            touched,
            read_filter,
            options: BTreeMap::new(),
            write_id: None,
        }
    }
}

/// A problem a row-level statement found, for the table to raise in its own terms
/// ([`RowLevelTable::error`]): a table format may raise its own classes where Spark's
/// analyzer would raise another (delta-spark's `DELTA_*` classes).
#[derive(Debug)]
#[non_exhaustive]
pub enum RowLevelProblem<'a> {
    /// A name of the statement that does not resolve: the analyzer's error.
    Unresolved(&'a KrishivError),
    /// A non-deterministic condition: `what` names it as delta-spark does (`search condition`,
    /// `DELETE condition`), `condition` is its text as written.
    NonDeterministic {
        /// The condition's role.
        what: &'a str,
        /// The condition as written.
        condition: &'a str,
    },
    /// A condition holding a subquery.
    Subquery {
        /// The condition's role.
        what: &'a str,
        /// The condition as written.
        condition: &'a str,
    },
    /// A `MERGE` `UPDATE` clause sets a column twice: the repeated keys as written, in order
    /// (Spark's `DUPLICATE_ASSIGNMENTS` by default).
    DuplicateAssignment {
        /// The repeated keys.
        columns: &'a [String],
    },
    /// A `MERGE` `INSERT` clause names a column twice (refused by default).
    DuplicateInsertColumn,
    /// An error raised while the rewrite ran: `overflow` is the assigned column whose
    /// store-assignment cast can overflow, when the error is arithmetic (SQLSTATE 22003 or
    /// 22012).
    WriteFailed {
        /// The error as it came.
        error: &'a KrishivError,
        /// The column, the new value's type and the column's type.
        overflow: Option<(
            &'a str,
            &'a krishiv_types::DataType,
            &'a krishiv_types::DataType,
        )>,
    },
}

impl RowLevelWrite {
    /// The caller's `krishiv.write.idempotencyKey` (E34), carried in the options.
    #[must_use]
    pub fn idempotency_key(&self) -> Option<String> {
        super::write::idempotency_key_option(&self.options)
    }

    /// [`Self::read_filter`] as a typed predicate over the input's columns by name, for a
    /// table's conflict check: a superset of the rows it keeps (an untranslatable conjunct is
    /// left out), `None` when there is no filter or nothing of it translates (every row).
    #[must_use]
    pub fn read_filter_pushdown(&self) -> Option<krishiv_io::PushdownExpr> {
        let filter = self.read_filter.as_ref()?;
        let mut columns = Vec::new();
        bound_columns(filter, &mut columns);
        columns.retain(|c: &krishiv_plan::BoundColumn| {
            self.input_schema
                .fields()
                .iter()
                .any(|f| f.name() == &c.name)
        });
        super::pushdown::translate(filter, &columns)
    }
}

/// The resolved columns of `e`'s function tree, each once.
pub(super) fn bound_columns(e: &Expr, out: &mut Vec<krishiv_plan::BoundColumn>) {
    use krishiv_plan::{ExprNode, ResolvedExpr};
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => {
            if !out.iter().any(|o| o.id == c.id) {
                out.push(c.clone());
            }
        }
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            for a in &f.args {
                bound_columns(a, out);
            }
        }
        _ => {}
    }
}

/// One table, at the version a row-level statement read, that the statement can change.
pub trait RowLevelTable: Send + Sync + fmt::Debug {
    /// What the table can do.
    fn capabilities(&self) -> RowLevelCapabilities;

    /// The table at the statement's version. Every scan of one statement reads the same snapshot;
    /// a scan built with `row_id` emits `_krishiv_row_file` and `_krishiv_row_index`.
    ///
    /// # Errors
    /// Errors opening the table.
    fn scan(&self) -> Result<Arc<dyn ScanPushdown>>;

    /// [`scan`](Self::scan) restricted to `files` (values of `_krishiv_row_file`): the input of a
    /// rewrite.
    ///
    /// # Errors
    /// Errors opening the table; a file not in the snapshot.
    fn scan_files(&self, files: &[Vec<u8>]) -> Result<Arc<dyn ScanPushdown>>;

    /// The sink of a copy-on-write rewrite. Its finalize commits atomically, with
    /// `request.write_id` as the idempotency key: `request.touched` are removed and the files
    /// written from the sink's rows added, after the table's conflict check against
    /// `request.read_filter`. A failed or cancelled statement leaves nothing visible. It is
    /// called with no touched files too: whether an empty change commits is the table's choice.
    /// The commit is based on this table's own pinned snapshot, the one [`scan`](Self::scan)
    /// read (newer commits are conflict-checked against it); the planner keeps one instance for
    /// the whole statement. A distributed run, where a worker resolves the table again, needs
    /// the read version carried in the request first.
    ///
    /// # Errors
    /// `UNSUPPORTED_FEATURE.TABLE_OPERATION` for an operation the table refuses.
    fn rewriter(&self, request: &RowLevelWrite) -> Result<Arc<dyn Sink>>;

    /// The statement's result rows from its counts; `None` gives an empty result, as Spark does
    /// for a v2 table.
    fn output(&self, counts: &RowLevelCounts) -> Option<RecordBatch> {
        let _ = counts;
        None
    }

    /// The error the table raises for `problem` of `op`; `None` leaves the planner's own
    /// (Spark's): the analyzer's error, the rewrite's error as it came, a refusal of a
    /// non-deterministic condition and of a `MERGE` condition with a subquery (a subquery in a
    /// `DELETE` or `UPDATE` condition runs).
    fn error(&self, op: RowLevelOperation, problem: &RowLevelProblem<'_>) -> Option<KrishivError> {
        let _ = (op, problem);
        None
    }
}

/// A scan of a [`RowLevelTable`] whose schema names the row id as two ordinary columns,
/// [`ROW_FILE_COLUMN`] and [`ROW_POSITION_COLUMN`] (appended last), so a statement's SQL can
/// select them: asking for either reads the row id (`ScanPushdowns::row_id`). The file column
/// holds the encoded file reference that [`RowLevelTable::scan_files`] takes back.
///
/// Every filter stays above the scan ([`FilterPushdown::Unsupported`]): a filter may name the
/// row id, which the table's scan does not know.
pub struct RowIdScan {
    inner: Arc<dyn ScanPushdown>,
    schema: SchemaRef,
}

impl fmt::Debug for RowIdScan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RowIdScan")
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

impl RowIdScan {
    /// `inner` with the row id columns. Its scan must
    /// [`supports_take`](ScanPushdown::supports_take).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` when `inner` cannot read row ids or already has a column of either name.
    pub fn new(inner: Arc<dyn ScanPushdown>) -> Result<Self> {
        if !inner.supports_take() {
            return Err(crate::error::internal(
                "a row-level scan must read row ids".to_owned(),
            ));
        }
        let table = inner.table_schema();
        if table
            .fields()
            .iter()
            .any(|f| f.name() == ROW_FILE_COLUMN || f.name() == ROW_POSITION_COLUMN)
        {
            return Err(crate::error::internal(
                "a row-level table already has a row id column".to_owned(),
            ));
        }
        let mut fields: Vec<FieldRef> = table.fields().iter().cloned().collect();
        // The type the file scan's row id column has (`FileScan::with_row_id`).
        fields.push(Arc::new(Field::new(
            ROW_FILE_COLUMN,
            DataType::BinaryView,
            false,
        )));
        fields.push(Arc::new(Field::new(
            ROW_POSITION_COLUMN,
            DataType::Int64,
            false,
        )));
        Ok(Self {
            inner,
            schema: Arc::new(Schema::new_with_metadata(fields, table.metadata().clone())),
        })
    }

    /// The inner pushdowns and the order of the output's top-level columns, by name.
    fn split(&self, pushdowns: ScanPushdowns) -> (ScanPushdowns, Option<Vec<String>>) {
        let is_row_id =
            |c: &ColumnPath| c.root() == ROW_FILE_COLUMN || c.root() == ROW_POSITION_COLUMN;
        let mut inner = pushdowns;
        inner.filters.clear();
        match inner.projection.take() {
            None => {
                inner.row_id = true;
                (inner, None)
            }
            Some(paths) => {
                inner.row_id = paths.iter().any(is_row_id);
                let mut order: Vec<String> = Vec::new();
                for p in &paths {
                    if !order.iter().any(|o| o == p.root()) {
                        order.push(p.root().to_owned());
                    }
                }
                inner.projection = Some(paths.into_iter().filter(|p| !is_row_id(p)).collect());
                (inner, Some(order))
            }
        }
    }
}

impl ScanPushdown for RowIdScan {
    fn table_schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn filter_pushdown(&self, filter: &PushedFilter) -> FilterPushdown {
        let _ = filter;
        FilterPushdown::Unsupported
    }

    fn build(&self, pushdowns: ScanPushdowns) -> Result<Arc<dyn Source>> {
        let (inner, order) = self.split(pushdowns);
        let source = self.inner.build(inner)?;
        let Some(order) = order else {
            return Ok(source);
        };
        let built = source.schema();
        let indices = order
            .iter()
            .map(|name| {
                built.index_of(name).map_err(|_| {
                    crate::error::internal(format!("the row-level scan did not return {name}"))
                })
            })
            .collect::<Result<Vec<usize>>>()?;
        if indices.iter().copied().eq(0..built.fields().len()) {
            return Ok(source);
        }
        Ok(Arc::new(Reordered::new(source, indices)?))
    }
}

/// A source whose batches' top-level columns are taken in another order.
struct Reordered {
    inner: Arc<dyn Source>,
    indices: Vec<usize>,
    schema: SchemaRef,
}

impl Reordered {
    fn new(inner: Arc<dyn Source>, indices: Vec<usize>) -> Result<Self> {
        let schema =
            Arc::new(inner.schema().project(&indices).map_err(|e| {
                crate::error::internal(format!("reordering a row-level scan: {e}"))
            })?);
        Ok(Self {
            inner,
            indices,
            schema,
        })
    }
}

impl Source for Reordered {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn global_state(&self, ctx: &TaskContext) -> Result<GlobalState> {
        self.inner.global_state(ctx)
    }

    fn max_parallelism(&self, global: &GlobalState) -> usize {
        self.inner.max_parallelism(global)
    }

    fn deterministic_split(&self, global: &GlobalState) -> Option<usize> {
        self.inner.deterministic_split(global)
    }

    fn partitions(&self, global: &GlobalState) -> usize {
        self.inner.partitions(global)
    }

    fn local_state(&self, ctx: &TaskContext, global: &GlobalState) -> Result<LocalState> {
        self.inner.local_state(ctx, global)
    }

    fn get_data<'a>(
        &'a self,
        local: &'a mut LocalState,
        global: &'a GlobalState,
    ) -> BoxFuture<'a, Result<Option<Batch>>> {
        Box::pin(async move {
            match self.inner.get_data(local, global).await? {
                Some(b) => Ok(Some(b.project(&self.indices)?)),
                None => Ok(None),
            }
        })
    }

    fn morsels_claimed(&self, local: &LocalState) -> u64 {
        self.inner.morsels_claimed(local)
    }

    fn reports_positions(&self) -> bool {
        self.inner.reports_positions()
    }

    fn position(&self, local: &LocalState) -> Option<u64> {
        self.inner.position(local)
    }

    fn runtime_filters(&self) -> Vec<FilterId> {
        self.inner.runtime_filters()
    }

    fn extra_metrics(&self) -> crate::metrics::ExtraMetrics {
        self.inner.extra_metrics()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use arrow_array::{ArrayRef, BinaryViewArray, Int32Array, Int64Array};
    use krishiv_types::batch::Batch as KBatch;

    use super::*;
    use crate::planner::{MemoryTable, TableProvider as _};

    /// A table of `a INT` whose scan reads the row id as the file scan does (appended last,
    /// `BinaryView` file, `Int64` position), and records the pushdowns it was built with.
    #[derive(Debug, Default)]
    struct Taking {
        built: Mutex<Vec<(Option<Vec<String>>, bool)>>,
    }

    fn table_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]))
    }

    impl ScanPushdown for Taking {
        fn table_schema(&self) -> SchemaRef {
            table_schema()
        }

        fn filter_pushdown(&self, _: &PushedFilter) -> FilterPushdown {
            FilterPushdown::Exact
        }

        fn supports_take(&self) -> bool {
            true
        }

        fn build(&self, pushdowns: ScanPushdowns) -> Result<Arc<dyn Source>> {
            self.built
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push((
                    pushdowns
                        .projection
                        .as_ref()
                        .map(|p| p.iter().map(|c| c.root().to_owned()).collect()),
                    pushdowns.row_id,
                ));
            let mut fields = vec![Field::new("a", DataType::Int32, true)];
            let mut columns: Vec<ArrayRef> = vec![Arc::new(Int32Array::from(vec![1, 2]))];
            if pushdowns.row_id {
                fields.push(Field::new(ROW_FILE_COLUMN, DataType::BinaryView, false));
                fields.push(Field::new(ROW_POSITION_COLUMN, DataType::Int64, false));
                columns.push(Arc::new(BinaryViewArray::from_iter_values([b"f", b"f"])));
                columns.push(Arc::new(Int64Array::from(vec![0, 1])));
            }
            let schema = Arc::new(Schema::new(fields));
            let batch = RecordBatch::try_new(Arc::clone(&schema), columns)
                .map_err(|e| crate::error::internal(e.to_string()))?;
            let inner = ScanPushdowns {
                projection: pushdowns.projection.clone().map(|mut p| {
                    if pushdowns.row_id {
                        p.push(ColumnPath::new(ROW_FILE_COLUMN));
                        p.push(ColumnPath::new(ROW_POSITION_COLUMN));
                    }
                    p
                }),
                ..ScanPushdowns::default()
            };
            MemoryTable::new(schema, vec![KBatch::new(batch)])
                .scan()?
                .build(inner)
        }
    }

    #[test]
    fn the_row_id_is_two_columns_of_the_file_scans_types_and_no_filter_goes_below() {
        let scan = RowIdScan::new(Arc::new(Taking::default())).unwrap();
        let schema = scan.table_schema();
        let types: Vec<_> = schema
            .fields()
            .iter()
            .map(|f| (f.name().clone(), f.data_type().clone()))
            .collect();
        assert_eq!(
            types,
            [
                ("a".to_owned(), DataType::Int32),
                (ROW_FILE_COLUMN.to_owned(), DataType::BinaryView),
                (ROW_POSITION_COLUMN.to_owned(), DataType::Int64),
            ]
        );
        let filter = PushedFilter::opaque(Arc::new(crate::operators::FnPredicate(
            |b: &RecordBatch| Ok(arrow_array::BooleanArray::from(vec![true; b.num_rows()])),
        )));
        assert_eq!(scan.filter_pushdown(&filter), FilterPushdown::Unsupported);
    }

    #[test]
    fn a_projection_asks_for_the_row_id_only_when_it_names_it_and_keeps_its_order() {
        let inner = Arc::new(Taking::default());
        let scan = RowIdScan::new(Arc::clone(&inner) as Arc<dyn ScanPushdown>).unwrap();
        let p = ScanPushdowns {
            projection: Some(vec![ColumnPath::new(ROW_FILE_COLUMN), ColumnPath::new("a")]),
            ..ScanPushdowns::default()
        };
        let source = scan.build(p).unwrap();
        let names: Vec<String> = source
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, [ROW_FILE_COLUMN.to_owned(), "a".to_owned()]);
        let p = ScanPushdowns {
            projection: Some(vec![ColumnPath::new("a")]),
            ..ScanPushdowns::default()
        };
        let source = scan.build(p).unwrap();
        assert_eq!(source.schema().fields().len(), 1);
        let built = inner.built.lock().unwrap().clone();
        assert_eq!(
            built,
            [
                (Some(vec!["a".to_owned()]), true),
                (Some(vec!["a".to_owned()]), false),
            ]
        );
        // No projection: every column and the row id.
        let source = scan.build(ScanPushdowns::default()).unwrap();
        assert_eq!(source.schema().fields().len(), 3);
    }

    #[test]
    fn a_scan_without_row_ids_or_with_a_row_id_column_is_refused() {
        let plain = MemoryTable::new(table_schema(), Vec::new()).scan().unwrap();
        assert!(RowIdScan::new(plain).is_err());
        #[derive(Debug)]
        struct Named(SchemaRef);
        impl ScanPushdown for Named {
            fn table_schema(&self) -> SchemaRef {
                Arc::clone(&self.0)
            }
            fn filter_pushdown(&self, _: &PushedFilter) -> FilterPushdown {
                FilterPushdown::Unsupported
            }
            fn supports_take(&self) -> bool {
                true
            }
            fn build(&self, _: ScanPushdowns) -> Result<Arc<dyn Source>> {
                Err(crate::error::internal("unused".to_owned()))
            }
        }
        let clash = Arc::new(Schema::new(vec![Field::new(
            ROW_POSITION_COLUMN,
            DataType::Int64,
            false,
        )]));
        assert!(RowIdScan::new(Arc::new(Named(clash))).is_err());
    }

    #[test]
    fn a_reordered_scan_reads_its_columns_in_the_asked_order_and_passes_the_rest_through() {
        use futures::FutureExt as _;
        use krishiv_memory::{MemoryConfig, MemoryPool as _, SpillOwner, TreePool};

        use crate::{
            ExecConfig, ExecutorConfig, OperatorId, QueryContext, QueryId, TaskContext,
            TokioExecutor, WorkloadClass,
        };
        let scan = RowIdScan::new(Arc::new(Taking::default())).unwrap();
        let source = scan
            .build(ScanPushdowns {
                projection: Some(vec![
                    ColumnPath::new(ROW_POSITION_COLUMN),
                    ColumnPath::new("a"),
                ]),
                ..ScanPushdowns::default()
            })
            .unwrap();
        let root = TreePool::new(&MemoryConfig::default());
        let q = QueryContext::new(
            QueryId(1),
            WorkloadClass::Batch,
            ExecConfig::default(),
            root.child("q", None),
        );
        let exec = TokioExecutor::new(&ExecutorConfig::default()).unwrap();
        let ctx = TaskContext::new(q, exec, SpillOwner::new(), OperatorId(0), "op");
        let global = source.global_state(&ctx).unwrap();
        assert_eq!(source.name(), "MemoryScan");
        assert_eq!(source.max_parallelism(&global), 1);
        assert_eq!(source.partitions(&global), 1);
        let _ = source.deterministic_split(&global);
        assert!(source.runtime_filters().is_empty());
        let _ = source.reports_positions();
        let _ = source.extra_metrics();
        let mut local = source.local_state(&ctx, &global).unwrap();
        let batch = source
            .get_data(&mut local, &global)
            .now_or_never()
            .unwrap()
            .unwrap()
            .unwrap();
        let data = batch.data();
        assert_eq!(data.schema().field(0).name(), ROW_POSITION_COLUMN);
        assert_eq!(
            data.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
            &[0, 1]
        );
        assert_eq!(
            data.column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values(),
            &[1, 2]
        );
        let _ = source.position(&local);
        let _ = source.morsels_claimed(&local);
        assert!(
            source
                .get_data(&mut local, &global)
                .now_or_never()
                .unwrap()
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn the_read_filter_translates_over_the_input_columns() {
        use krishiv_io::{CmpOp, PushdownExpr, ScalarValue};
        use krishiv_plan::{
            AttrId, BoundColumn, Builtin, ExprNode, FunctionId, FunctionKind, Literal,
            ResolvedExpr, ResolvedFunction,
        };
        let column = |name: &str, id| {
            Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
                BoundColumn {
                    id: AttrId(id),
                    name: name.to_owned(),
                    data_type: krishiv_types::DataType::Integer,
                    nullable: true,
                },
            ))))
        };
        let eq = |c: Arc<Expr>| {
            Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                ResolvedFunction {
                    id: FunctionId::Builtin(Builtin::Eq),
                    kind: FunctionKind::Scalar,
                    args: vec![
                        c,
                        Arc::new(Expr::new(ExprNode::Literal(Literal::Integer(1)))),
                    ],
                    distinct: false,
                    filter: None,
                    within_group: Vec::new(),
                    data_type: krishiv_types::DataType::Boolean,
                    nullable: true,
                    deterministic: true,
                },
            ))))
        };
        let mut w = RowLevelWrite::new(
            RowLevelOperation::Delete,
            "t",
            table_schema(),
            Vec::new(),
            Some(eq(column("a", 1))),
        );
        assert_eq!(
            w.read_filter_pushdown(),
            Some(PushdownExpr::cmp(
                ColumnPath::new("a"),
                CmpOp::Eq,
                ScalarValue::Int32(1)
            ))
        );
        // A column the input lacks, or no filter: every row.
        w.read_filter = Some(eq(column("b", 2)));
        assert_eq!(w.read_filter_pushdown(), None);
        w.read_filter = None;
        assert_eq!(w.read_filter_pushdown(), None);
    }

    #[test]
    fn operations_name_themselves_as_sparks_errors_do() {
        assert_eq!(RowLevelOperation::Delete.name(), "DELETE");
        assert_eq!(RowLevelOperation::Update.name(), "UPDATE");
        assert_eq!(RowLevelOperation::Merge.name(), "MERGE INTO");
        let c = RowLevelCapabilities::copy_on_write(CardinalityRule::SparkMergeCardinality);
        assert!(!c.merge_on_read);
    }
}
