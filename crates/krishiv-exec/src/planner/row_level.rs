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
use arrow_schema::SchemaRef;
use krishiv_plan::Expr;

use crate::error::Result;
use crate::operator::Sink;
use crate::operators::ScanPushdown;

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
    /// The files the rewrite replaces (values of `_krishiv_row_file`), each once.
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
