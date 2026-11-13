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
#[derive(Debug, Clone)]
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
    pub touched: Vec<String>,
    /// The statement's condition, for the table's conflict check (`None`: every row was read).
    pub read_filter: Option<Expr>,
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
        touched: Vec<String>,
        read_filter: Option<Expr>,
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
    fn scan_files(&self, files: &[String]) -> Result<Arc<dyn ScanPushdown>>;

    /// The sink of a copy-on-write rewrite. Its finalize commits atomically, with
    /// `request.write_id` as the idempotency key: `request.touched` are removed and the files
    /// written from the sink's rows added, after the table's conflict check against
    /// `request.read_filter`. A failed or cancelled statement leaves nothing visible. It is
    /// called with no touched files too: whether an empty change commits is the table's choice.
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
        fields.push(Arc::new(Field::new(
            ROW_FILE_COLUMN,
            DataType::Binary,
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
