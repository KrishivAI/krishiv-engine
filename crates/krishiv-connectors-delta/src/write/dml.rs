// SPDX-License-Identifier: Apache-2.0
//! Copy-on-write `DELETE`, `UPDATE` and `MERGE` on a Delta table (design note P3-T021 sections
//! 2.3, 6.1 and 6.2; PR-37b part 2, PR-37c): [`DeltaRowLevel`], the table's [`RowLevelTable`]; what
//! a `MERGE` commits is in [`super::merge`].
//!
//! The session plans the statement (krishiv-session `table_commands/row_level.rs`): a find query
//! over [`scan`](DeltaRowLevel) (the live rows, deletion vectors applied, with
//! `_krishiv_row_file` = the file's path as the log names it and `_krishiv_row_index` = the row's
//! position in the file) gives the touched files, then the rows of those files to keep
//! (`DELETE`) or every row of them with the new values (`UPDATE`) go through the
//! [`rewriter`](DeltaRowLevel), the write path's sink in a row-level mode. Its finalize:
//! 1. ends the written files as any write does;
//! 2. selects the files to remove from the snapshot's scan metadata (batch by batch, keeping
//!    only the batches holding a removed file), so each remove carries the file's deletion
//!    vector as the log has it: the touched files, and for a whole-file `DELETE` (one that wrote
//!    no row) the untouched files of the matching partitions whose statistics prove no live row
//!    (every live file without a condition). The find query decides what is deleted; the
//!    partition predicate's translation never removes a file holding a live row (review 1, B1);
//! 3. commits `DELETE` or `UPDATE` through the commit loop, with delta-spark's
//!    `operationParameters` and `operationMetrics`, the removed files in the read set and the
//!    statement read by its condition's partition conjuncts (`filter::partition_conjuncts`;
//!    delta-spark's read predicates), else as a whole-table read, so a concurrent append to a
//!    partition read is `DELTA_CONCURRENT_APPEND`, a remove of a read file
//!    `DELTA_CONCURRENT_DELETE_READ` and of a removed file `DELTA_CONCURRENT_DELETE_DELETE`;
//! 4. records the outcome for the statement's result row (`num_affected_rows`).
//!
//! A statement that changes nothing commits nothing (delta-spark's `commitIfNeeded`).
//!
//! # Delta's rules ported
//! - A `DELETE` without a condition, or whose condition names only partition columns, removes
//!   whole files without rewriting them (`DeleteCommand.scala:196-251`, cases 1 and 2): its
//!   `numDeletedRows` comes from the files' statistics (`numRecords` less the deletion vector's
//!   cardinality, `DeleteCommand.scala:81-99`), its metrics drop `numDeletedRows`,
//!   `numCopiedRows` and `numAddedFiles` when that count is not known (`DeltaOperations.scala:
//!   266-275`), and its result is `-1` then (`DeleteCommand.scala:159-163`).
//! - Otherwise a `DELETE` rewrites the touched files' other rows (`DeleteCommand.scala:313-380`):
//!   `numCopiedRows` = rows written, `numDeletionVectorsRemoved` = removed files with a
//!   deletion vector.
//! - `UPDATE` rewrites every row of the touched files (`UpdateCommand.scala:340-395`):
//!   `numUpdatedRows` = the rows the condition holds for, `numCopiedRows` = the others.
//! - The result row is `num_affected_rows BIGINT` (`DeleteCommand.scala:124`,
//!   `UpdateCommand.scala:67-69`).
//!
//! # Refusals (before any file is written)
//! - Krishiv's writer allow-list and `DELTA_CANNOT_MODIFY_APPEND_ONLY` (delta-spark's
//!   `DeltaLog.assertRemovable`), when the statement's scan is built.
//! - On a table with the change data feed on, `UPDATE` and a `DELETE` that rewrites rows:
//!   `UNSUPPORTED_FEATURE.TABLE_OPERATION` (design section 6.2; the kernel refuses a commit with
//!   adds and removes there). A whole-file `DELETE` commits.
//!
//! # Known differences
//! - `operationParameters.predicate` is the condition as written, parenthesised, in a one-element
//!   JSON array (`["true"]` for a `DELETE` and `[]` for an `UPDATE` without one, as delta-spark
//!   records them); delta-spark prints Catalyst's `simpleString` of the resolved condition, with
//!   expression ids and unquoted strings (`["(id#1345L = 0)"]`, `["(p#28895 = x)"]`).
//! - A whole-file `DELETE` is recognised only when its condition translates as a partition
//!   predicate (comparisons, `IN`, `IS [NOT] NULL`, `AND`/`OR`/`NOT` of a partition column and a
//!   literal; a column one-part or qualified by the statement's alias, never a struct field; no
//!   string compared with a `TIMESTAMP` column); delta-spark also takes functions of partition
//!   columns. Such a statement rewrites instead (the same rows go; its metrics are a rewrite's,
//!   and a change-data-feed table refuses it). A matching partition's file with no live row is
//!   removed only when its statistics prove it.
//! - The read predicates are the conjuncts that translate; delta-spark takes every
//!   partition-only conjunct, so a conjunct of another form makes Krishiv's read wider (safe).
//! - Deletion vectors are never written: on a table with `delta.enableDeletionVectors = true`
//!   delta-spark marks the changed rows in deletion vectors (its default
//!   `...deletionVectors.persistent = true`); Krishiv rewrites the files (merge-on-read is
//!   PR-37d). The rewrite drops the old files' vectors with them.
//! - `UPDATE` whose condition names only partition columns rewrites the files that hold a row,
//!   not every file of the partitions (a file whose rows are all deleted is kept).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use arrow_array::cast::AsArray;
use arrow_array::{Array, Int64Array, MapArray, RecordBatch, StructArray};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use delta_kernel::expressions::PredicateRef;
use krishiv_common::error::{
    DeltaCastOverflowInTableWrite, DeltaConflictSetColumn, DeltaDuplicateColumnsOnInsert,
    DeltaNonDeterministicFunctionNotSupported, DeltaUnsupportedSubquery, InternalError,
    KrishivError, TaskWriteFailed, UnsupportedFeatureTableOperation,
};
use krishiv_exec::planner::{
    CardinalityRule, RowIdScan, RowLevelCapabilities, RowLevelCounts, RowLevelOperation,
    RowLevelProblem, RowLevelTable, RowLevelWrite,
};
use krishiv_exec::{ScanPushdown, Sink};
use krishiv_plan::{Expr, ExprNode};

use super::files::Written;
use super::filter::{partition_conjuncts, partition_predicate};
use super::sink::{ExactCheck, Failure, append_only_refusal, for_each_scan_batch, row_level_sink};
use crate::commit::winner::FileAction;
use crate::commit::{CommitOperation, ReadSet, StagedCommit, StagedRemoves};
use crate::engine::DeltaEngine;
use crate::features::{WriteProtocol, check_write};
use crate::table::{DeltaScan, DeltaTable, ScanUnit};

/// The row-level option holding the rows the statement's condition holds for (the find query's
/// count), a decimal number.
pub const ROW_LEVEL_AFFECTED_OPTION: &str = "__krishiv_row_level_affected";

/// The row-level option holding the statement's condition as written (absent: no condition).
pub const ROW_LEVEL_CONDITION_OPTION: &str = "__krishiv_row_level_condition";

/// The row-level option holding the milliseconds the find query took (`scanTimeMs`).
pub const ROW_LEVEL_SCAN_MS_OPTION: &str = "__krishiv_row_level_scan_ms";

/// The row-level option holding the name the statement reads the table under (its alias, else
/// the table's last name part): a two-part column name of the condition starting with it names
/// a column of the table (absent: only one-part names do).
pub const ROW_LEVEL_ALIAS_OPTION: &str = "__krishiv_row_level_alias";

/// Whether `e` holds a subquery (`IN (SELECT ..)`, `EXISTS`, a scalar subquery) anywhere.
/// delta-spark refuses a `DELETE` or `UPDATE` whose condition has one
/// (`PreprocessTableDelete.scala:35-45`, `PreprocessTableUpdate.scala:44`).
#[must_use]
pub fn has_subquery(e: &Expr) -> bool {
    let mut stack: Vec<&Expr> = vec![e];
    while let Some(x) = stack.pop() {
        if matches!(x.node, ExprNode::Subquery { .. }) {
            return true;
        }
        x.for_each_child(&mut |c| stack.push(c));
    }
    false
}

/// What a finished row-level change did, for its result row.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Outcome {
    /// Files removed.
    removed_files: u64,
    /// delta-spark's `numDeletedRows` (`DELETE`) or `numUpdatedRows` (`UPDATE`) metric; `None`
    /// when a whole-file delete's files lack the statistic.
    rows: Option<u64>,
    /// A `MERGE`'s counts.
    merge: Option<super::merge::MergeOutcome>,
}

/// A Delta table at the version a `DELETE` or `UPDATE` read (see the module docs).
#[derive(Debug)]
pub struct DeltaRowLevel {
    table: DeltaTable,
    op: RowLevelOperation,
    outcome: Arc<Mutex<Option<Outcome>>>,
}

impl DeltaRowLevel {
    /// The row-level table of `op` over `table`; `None` for an operation not built.
    #[must_use]
    pub fn new(table: DeltaTable, op: RowLevelOperation) -> Option<Self> {
        matches!(
            op,
            RowLevelOperation::Delete | RowLevelOperation::Update | RowLevelOperation::Merge
        )
        .then(|| Self {
            table,
            op,
            outcome: Arc::new(Mutex::new(None)),
        })
    }

    /// The refusals that need no condition: the writer allow-list and append-only (a `MERGE`
    /// may only insert: its refusal waits for the touched files, [`Self::rewriter`]).
    fn check_table(&self) -> Result<(), KrishivError> {
        check_write(self.table.name(), &WriteProtocol::of(self.table.snapshot()))?;
        if self.op == RowLevelOperation::Merge {
            return Ok(());
        }
        append_only_refusal(&self.table)
    }

    fn row_scan(&self, inner: DeltaScan) -> Result<Arc<dyn ScanPushdown>, KrishivError> {
        Ok(Arc::new(RowIdScan::new(Arc::new(inner))?))
    }
}

/// The paths `files` (values of `_krishiv_row_file`) name, each checked to be a live file of
/// the snapshot.
fn touched_paths(table: &DeltaTable, files: &[Vec<u8>]) -> Result<HashSet<String>, KrishivError> {
    let live: HashSet<&str> = table.units().iter().map(|u| u.path.as_str()).collect();
    files
        .iter()
        .map(|f| {
            let path = std::str::from_utf8(f)
                .map_err(|_| internal("a Delta row id file that is not UTF-8"))?;
            if live.contains(path) {
                Ok(path.to_owned())
            } else {
                Err(internal(&format!(
                    "the Delta row id file {path} is not a live file of the snapshot"
                )))
            }
        })
        .collect()
}

impl RowLevelTable for DeltaRowLevel {
    fn capabilities(&self) -> RowLevelCapabilities {
        RowLevelCapabilities::copy_on_write(CardinalityRule::DeltaWithUnconditionalDeleteExemption)
    }

    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        self.check_table()?;
        self.row_scan(self.table.row_level_scan(None)?)
    }

    fn scan_files(&self, files: &[Vec<u8>]) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        let paths = touched_paths(&self.table, files)?;
        self.row_scan(self.table.row_level_scan(Some(&paths))?)
    }

    fn rewriter(&self, request: &RowLevelWrite) -> krishiv_exec::Result<Arc<dyn Sink>> {
        self.check_table()?;
        let write_id = request
            .write_id
            .clone()
            .ok_or_else(|| internal("a Delta row-level change without a write id"))?;
        let touched = touched_paths(&self.table, &request.touched)?;
        let option = |key: &str| request.options.get(key);
        let affected = option(ROW_LEVEL_AFFECTED_OPTION)
            .map(|v| v.parse::<u64>())
            .transpose()
            .map_err(|_| internal("a row-level count that is not a number"))?
            .unwrap_or(0);
        let scan_ms = option(ROW_LEVEL_SCAN_MS_OPTION)
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
        let condition = option(ROW_LEVEL_CONDITION_OPTION).cloned();
        let qualifiers: Vec<&str> = option(ROW_LEVEL_ALIAS_OPTION)
            .map(String::as_str)
            .into_iter()
            .collect();
        let (schema, partitions) = (
            self.table.snapshot().schema(),
            self.table.partition_columns(),
        );
        // A `DELETE` without a condition or with a partition-only one removes whole files.
        let whole_files = match (request.op, &request.read_filter) {
            (RowLevelOperation::Delete, None) => Some(None),
            (RowLevelOperation::Delete, Some(filter)) => {
                partition_predicate(filter, &schema, &partitions, &qualifiers)
                    .ok()
                    .map(|p| Some(Arc::new(p)))
            }
            _ => None,
        };
        // What the statement read, for the conflict check: the condition's partition conjuncts
        // (delta-spark's read predicates), else the whole table.
        // A subquery reads the table too (review 2, B2): such a condition is a whole-table read.
        let read_predicate: Option<PredicateRef> = request
            .read_filter
            .as_ref()
            .filter(|f| !has_subquery(f))
            .and_then(|f| partition_conjuncts(f, &schema, &partitions, &qualifiers).map(Arc::new));
        let cdf = self
            .table
            .snapshot()
            .table_properties()
            .enable_change_data_feed
            == Some(true);
        let merge = if request.op == RowLevelOperation::Merge {
            let text = |key: &str| {
                option(key)
                    .cloned()
                    .ok_or_else(|| internal(&format!("a Delta MERGE without {key}")))
            };
            if !touched.is_empty() {
                // A MERGE that removes a file: refused where delta-spark refuses removes, and
                // where it writes change data files (design section 6.2).
                append_only_refusal(&self.table)?;
                if cdf {
                    return Err(cdf_refusal(
                        &request.table_name,
                        "MERGE INTO that changes or deletes rows",
                    )
                    .into());
                }
            }
            Some(super::merge::MergeInfo::parse(
                &text(super::merge::MERGE_COUNTS_OPTION)?,
                &text(super::merge::MERGE_CLAUSES_OPTION)?,
                &text(ROW_LEVEL_CONDITION_OPTION)?,
            )?)
        } else {
            None
        };
        if cdf && whole_files.is_none() && merge.is_none() {
            let what = match request.op {
                RowLevelOperation::Delete => "DELETE of rows (not of whole partitions)",
                _ => "UPDATE",
            };
            return Err(cdf_refusal(&request.table_name, what).into());
        }
        let plan = RowLevelPlan {
            op: request.op,
            touched,
            whole_files,
            read_predicate,
            condition,
            affected,
            scan_ms,
            refuse_rows: (cdf && merge.is_none()).then(|| request.table_name.clone()),
            merge,
            outcome: Arc::clone(&self.outcome),
        };
        Ok(row_level_sink(
            &self.table,
            plan,
            write_id,
            request.idempotency_key(),
            self.table.retry_limits(),
        ))
    }

    /// delta-spark's classes for what the session's planner found (krishiv-session
    /// `row_level.rs`, `merge_checks.rs`):
    /// - a `MERGE` name that does not resolve: `INTERNAL_ERROR` "Cannot find main error class
    ///   'DELTA_MERGE_UNRESOLVED_EXPRESSION'", the analyzer's error as its cause (delta-spark
    ///   4.4.1's class file lacks the class it raises, `ResolveDeltaMergeInto.scala:44-52`;
    ///   `merge/MERGE.json`);
    /// - a non-deterministic `MERGE` condition: `DELTA_NON_DETERMINISTIC_FUNCTION_NOT_SUPPORTED`
    ///   (`PreprocessTableMerge.scala:66-71`);
    /// - a subquery: `DELTA_UNSUPPORTED_SUBQUERY` (`PreprocessTableDelete.scala:35-45`,
    ///   `PreprocessTableUpdate.scala:44`, `PreprocessTableMerge.scala:76-78`);
    /// - an error of a task writing rows: Spark's `TASK_WRITE_FAILED` naming the table's root
    ///   (`FileFormatWriter`), caused by the error, or by `DELTA_CAST_OVERFLOW_IN_TABLE_WRITE`
    ///   for an arithmetic error under a store-assignment cast that can overflow
    ///   (`UpdateExpressionsSupport.scala:729-757`); conflicts, Krishiv's refusals and limits,
    ///   internal errors and cancellations stay as they came.
    fn error(&self, op: RowLevelOperation, problem: &RowLevelProblem<'_>) -> Option<KrishivError> {
        let name = |op: RowLevelOperation| match op {
            RowLevelOperation::Update => "UPDATE",
            RowLevelOperation::Merge => "MERGE",
            _ => "DELETE",
        };
        match problem {
            RowLevelProblem::Unresolved(e) if op == RowLevelOperation::Merge => Some(
                KrishivError::from(InternalError {
                    message: "Cannot find main error class 'DELTA_MERGE_UNRESOLVED_EXPRESSION'"
                        .to_owned(),
                })
                .with_cause((*e).clone()),
            ),
            RowLevelProblem::NonDeterministic { what, condition }
                if op == RowLevelOperation::Merge =>
            {
                Some(
                    DeltaNonDeterministicFunctionNotSupported {
                        operation: format!("{what} of MERGE operation"),
                        expression: format!("(condition = {condition})"),
                    }
                    .into(),
                )
            }
            RowLevelProblem::Subquery { what, condition } => Some(
                DeltaUnsupportedSubquery {
                    operation: if op == RowLevelOperation::Merge {
                        format!("{what} of MERGE operation")
                    } else {
                        name(op).to_owned()
                    },
                    cond: (*condition).to_owned(),
                }
                .into(),
            ),
            // `PreprocessTableMerge.scala:136-162` and delta-spark's `UPDATE` alignment.
            RowLevelProblem::DuplicateAssignment { columns } if op == RowLevelOperation::Merge => {
                Some(
                    DeltaConflictSetColumn {
                        column_list: format!(
                            "[{}]",
                            columns
                                .iter()
                                .map(|c| format!("`{}`", c.replace('`', "``")))
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                    }
                    .into(),
                )
            }
            RowLevelProblem::DuplicateInsertColumn => Some(DeltaDuplicateColumnsOnInsert {}.into()),
            RowLevelProblem::WriteFailed { error, overflow } => {
                Some(task_write_failed(&self.table, error, *overflow))
            }
            _ => None,
        }
    }

    /// delta-spark's result row: `num_affected_rows`, `-1` for a whole-file delete whose count is
    /// not known.
    fn output(&self, counts: &RowLevelCounts) -> Option<RecordBatch> {
        let outcome = *self.outcome.lock().unwrap_or_else(PoisonError::into_inner);
        if self.op == RowLevelOperation::Merge {
            // delta-spark's four counts (`MergeIntoCommand.scala:78-82, 183-192`).
            let (u, d, i) = match outcome.and_then(|o| o.merge) {
                Some(m) => (m.updated, m.deleted, m.inserted),
                None => (counts.updated, counts.deleted, counts.inserted),
            };
            return Some(merge_row(u, d, i));
        }
        let n = match outcome {
            // Nothing was committed: no row matched.
            None => counts.affected,
            Some(o) => match (self.op, o.rows) {
                (RowLevelOperation::Delete, None | Some(0)) if o.removed_files > 0 => {
                    return Some(affected_row(-1));
                }
                (_, rows) => rows.unwrap_or(0),
            },
        };
        Some(affected_row(i64::try_from(n).unwrap_or(i64::MAX)))
    }
}

fn affected_row(n: i64) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "num_affected_rows",
        ArrowType::Int64,
        true,
    )]));
    RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![n]))])
        .unwrap_or_else(|_| RecordBatch::new_empty(Arc::new(Schema::empty())))
}

/// `TASK_WRITE_FAILED` around `e` (see [`DeltaRowLevel`]'s `error`).
fn task_write_failed(
    table: &DeltaTable,
    e: &KrishivError,
    overflow: Option<(&str, &krishiv_types::DataType, &krishiv_types::DataType)>,
) -> KrishivError {
    let class = e.error_class().name();
    if class.starts_with("DELTA_")
        || class.starts_with("KRISHIV_")
        || class.starts_with("UNSUPPORTED_FEATURE")
        || class == "INTERNAL_ERROR"
        || e.sql_state() == Some("HY008")
    {
        return e.clone();
    }
    let sql = |t: &krishiv_types::DataType| format!("\"{}\"", t.catalog_string().to_uppercase());
    let cause = match overflow {
        Some((column, value, target)) if matches!(e.sql_state(), Some("22003" | "22012")) => {
            DeltaCastOverflowInTableWrite {
                source_type: sql(value),
                target_type: sql(target),
                column_name: format!("`{}`", column.replace('`', "``")),
                store_assignment_policy_flag: "spark.sql.storeAssignmentPolicy".to_owned(),
                update_and_merge_casting_follows_ansi_enabled_flag:
                    "spark.databricks.delta.updateAndMergeCastingFollowsAnsiEnabledFlag".to_owned(),
                ansi_enabled_flag: "spark.sql.ansi.enabled".to_owned(),
            }
            .into()
        }
        _ => e.clone(),
    };
    KrishivError::from(TaskWriteFailed {
        path: crate::table::hadoop_path(table.engine().table_root()),
    })
    .with_cause(cause)
}

fn merge_row(updated: u64, deleted: u64, inserted: u64) -> RecordBatch {
    let n = |v: u64| i64::try_from(v).unwrap_or(i64::MAX);
    let names = [
        "num_affected_rows",
        "num_updated_rows",
        "num_deleted_rows",
        "num_inserted_rows",
    ];
    let values = [
        n(updated.saturating_add(deleted).saturating_add(inserted)),
        n(updated),
        n(deleted),
        n(inserted),
    ];
    let schema = Arc::new(Schema::new(
        names
            .iter()
            .map(|c| Field::new(*c, ArrowType::Int64, true))
            .collect::<Vec<_>>(),
    ));
    let columns: Vec<arrow_array::ArrayRef> = values
        .iter()
        .map(|v| Arc::new(Int64Array::from(vec![*v])) as arrow_array::ArrayRef)
        .collect();
    RecordBatch::try_new(schema, columns)
        .unwrap_or_else(|_| RecordBatch::new_empty(Arc::new(Schema::empty())))
}

fn cdf_refusal(table: &str, what: &str) -> UnsupportedFeatureTableOperation {
    UnsupportedFeatureTableOperation {
        table_name: table.to_owned(),
        operation: format!("{what} on a Delta table with change data feed enabled"),
    }
}

/// What a row-level sink commits (see the module docs).
#[derive(Debug)]
pub(crate) struct RowLevelPlan {
    op: RowLevelOperation,
    /// The touched files' paths, as the log names them.
    touched: HashSet<String>,
    /// A whole-file `DELETE`: every file (`Some(None)`) or the files of the partitions the
    /// predicate holds for. It is one only when no row was written (every touched file's live
    /// rows all matched); then the untouched files it selects are removed too when they hold no
    /// live row (with no condition: whatever they hold, the find query read every file).
    whole_files: Option<Option<PredicateRef>>,
    /// The read predicate of the conflict check (`None`: the whole table).
    read_predicate: Option<PredicateRef>,
    /// The condition as written.
    condition: Option<String>,
    /// The rows the condition holds for.
    affected: u64,
    /// The find query's milliseconds.
    scan_ms: u64,
    /// A whole-file `DELETE` on a change-data-feed table: no row may be written (the table's
    /// name for the refusal).
    refuse_rows: Option<String>,
    /// A `MERGE`'s counts and clauses.
    merge: Option<super::merge::MergeInfo>,
    outcome: Arc<Mutex<Option<Outcome>>>,
}

impl RowLevelPlan {
    /// The refusal of a written row, before it is written.
    pub(crate) fn check_rows(&self) -> Result<(), KrishivError> {
        match &self.refuse_rows {
            Some(table) => {
                Err(cdf_refusal(table, "DELETE of rows (not of whole partitions)").into())
            }
            None => Ok(()),
        }
    }

    fn operation(&self) -> CommitOperation {
        match self.op {
            RowLevelOperation::Update => CommitOperation::Update,
            RowLevelOperation::Merge => CommitOperation::Merge,
            _ => CommitOperation::Delete,
        }
    }

    /// `operationParameters.predicate`: a JSON array of the condition (see the known
    /// differences).
    fn predicate_parameter(&self) -> String {
        // delta-spark records `DELETE FROM t` as the condition `true` and an `UPDATE` without
        // one as `[]` (`tests/fixtures/delta-4.4.1/dml/DML.json`).
        let all = (self.op == RowLevelOperation::Delete).then(|| "true".to_owned());
        let items: Vec<String> = self
            .condition
            .clone()
            .or(all)
            .iter()
            .map(|c| {
                let c = c.trim();
                if c == "true" || (c.starts_with('(') && c.ends_with(')')) {
                    c.to_owned()
                } else {
                    format!("({c})")
                }
            })
            .collect();
        serde_json::to_string(&items).unwrap_or_else(|_| "[]".to_owned())
    }
}

/// The removed files' totals.
#[derive(Debug, Default)]
struct Removed {
    files: u64,
    bytes: u64,
    with_dv: u64,
    /// Live rows by the statistics (`None`: a file without `numRecords`).
    rows: Option<u64>,
    actions: Vec<FileAction>,
}

/// The commit of a row-level change, or `None` when it changes nothing.
pub(super) async fn stage(
    engine: &DeltaEngine,
    started: Instant,
    table: &DeltaTable,
    plan: &RowLevelPlan,
    written: Written,
    write_id: &str,
) -> Result<Option<StagedCommit>, Failure> {
    let (snapshot, name, partitions) = (
        Arc::clone(table.snapshot()),
        table.name().to_owned(),
        table.partition_columns(),
    );
    // A whole-file plan that wrote rows was not one (the find query proved some touched row
    // survives): it commits as a rewrite of the touched files only.
    let whole_file = plan.whole_files.is_some() && written.rows == 0;
    let whole = if whole_file {
        plan.whole_files.clone()
    } else {
        None
    };
    // The untouched files a partition predicate may add: only those whose statistics prove no
    // live row is left (the find query read no row of them; the predicate's translation is never
    // trusted with a live row).
    let empty: HashSet<String> = match &whole {
        Some(Some(_)) => table
            .units()
            .iter()
            .filter(|u| no_live_rows(u))
            .map(|u| u.path.clone())
            .collect(),
        _ => HashSet::new(),
    };
    let touched = plan.touched.clone();
    let read_predicate = plan.read_predicate.clone();
    let (removes, actions, also_read) = engine
        .delta_blocking(move |e| {
            let exact = match &whole {
                Some(Some(p)) => Some(ExactCheck::new(e, &snapshot, &partitions, p, &name)?),
                _ => None,
            };
            // The statement's candidate files (delta-spark's `filterFiles` by the read
            // predicates): every one is a read file of the conflict check, touched or not.
            let read = match &read_predicate {
                Some(p) => Some(ExactCheck::new(e, &snapshot, &partitions, p, &name)?),
                None => None,
            };
            let mut out = Vec::new();
            let mut actions = Vec::new();
            let mut also_read = Vec::new();
            let mut found = 0_usize;
            // Batch by batch: only the batches holding a removed file are kept.
            for_each_scan_batch(e, &snapshot, &name, |scan_rows, selection| {
                let extra = match (&whole, &exact) {
                    (Some(None), _) => selection.clone(),
                    (Some(Some(_)), Some(x)) => {
                        let mut s = selection.clone();
                        x.narrow(&scan_rows, &mut s)?;
                        s
                    }
                    _ => vec![false; selection.len()],
                };
                let candidate = match &read {
                    Some(x) => {
                        let mut s = selection.clone();
                        x.narrow(&scan_rows, &mut s)?;
                        s
                    }
                    None => Vec::new(),
                };
                let paths = paths_of(&scan_rows)?;
                let mut keep = vec![false; selection.len()];
                for (i, k) in keep.iter_mut().enumerate() {
                    let live = selection.get(i).copied().unwrap_or(false);
                    let Some(Some(p)) = paths.get(i) else {
                        continue;
                    };
                    let hit = touched.contains(p.as_str());
                    if hit && live {
                        found += 1;
                    }
                    let added = extra.get(i).is_some_and(|x| {
                        *x && (matches!(whole, Some(None)) || empty.contains(p.as_str()))
                    });
                    *k = live && (hit || added);
                    if *k {
                        actions.push(FileAction {
                            path: p.clone(),
                            partition_values: partition_values_of(&scan_rows, i)?,
                        });
                    } else if live && candidate.get(i).copied().unwrap_or(false) {
                        also_read.push(FileAction {
                            path: p.clone(),
                            partition_values: partition_values_of(&scan_rows, i)?,
                        });
                    }
                }
                if keep.iter().any(|k| *k) {
                    out.push(StagedRemoves {
                        scan_rows,
                        selection: keep,
                    });
                }
                Ok(())
            })?;
            if found != touched.len() {
                return Err(internal(
                    "a touched file of a Delta row-level change is not in the snapshot's scan",
                ));
            }
            Ok((out, actions, also_read))
        })
        .await
        .and_then(|r| r)
        .map_err(Failure::NotCommitted)?;
    #[cfg(test)]
    if super::test_hooks::fails(write_id) {
        return Err(Failure::NotCommitted(internal(
            "a failure injected into the Delta row-level stage",
        )));
    }
    let removed = totals(table.units(), actions);
    if removed.files == 0 && written.files == 0 {
        // Nothing to commit.
        *plan.outcome.lock().unwrap_or_else(PoisonError::into_inner) = None;
        return Ok(None);
    }
    let rewrite_ms = millis(started.elapsed());
    if let Some(info) = &plan.merge {
        *plan.outcome.lock().unwrap_or_else(PoisonError::into_inner) = Some(Outcome {
            removed_files: removed.files,
            rows: None,
            merge: Some(info.outcome(written.rows)),
        });
        let metrics = info.metrics(
            written.rows,
            written.files,
            written.bytes,
            removed.files,
            removed.bytes,
            plan.scan_ms,
            rewrite_ms,
        );
        let mut read_set = ReadSet::default();
        match &plan.read_predicate {
            Some(p) => read_set.predicates.push(Arc::clone(p)),
            None => read_set.whole_table = true,
        }
        read_set.files = removed.actions;
        read_set.files.extend(also_read);
        return Ok(Some(StagedCommit {
            table: table.name().to_owned(),
            write_id: write_id.to_owned(),
            idempotency_key: None,
            operation: plan.operation(),
            operation_parameters: info.parameters(),
            operation_metrics: metrics
                .into_iter()
                .map(|(k, v)| (k, Some(v.to_string())))
                .collect(),
            adds: written.adds,
            removes,
            read_set,
        }));
    }
    let mut metrics: Vec<(String, u64)> = Vec::new();
    let mut put = |k: &str, v: u64| metrics.push((k.to_owned(), v));
    put("numRemovedFiles", removed.files);
    put("numRemovedBytes", removed.bytes);
    put("numAddedFiles", written.files);
    put("numAddedBytes", written.bytes);
    put("numAddedChangeFiles", 0);
    put("numDeletionVectorsAdded", 0);
    put("numDeletionVectorsUpdated", 0);
    put("scanTimeMs", plan.scan_ms);
    let outcome_rows = match plan.op {
        RowLevelOperation::Update => {
            put("numUpdatedRows", plan.affected);
            put("numCopiedRows", written.rows.saturating_sub(plan.affected));
            put("numDeletionVectorsRemoved", 0);
            put("rewriteTimeMs", rewrite_ms);
            Some(plan.affected)
        }
        _ if whole_file => {
            put("numDeletionVectorsRemoved", removed.with_dv);
            put("rewriteTimeMs", 0);
            let deleted = removed.rows.unwrap_or(0);
            if deleted == 0 && removed.files != 0 {
                // delta-spark drops these when the count is not known.
                metrics.retain(|(k, _)| k != "numAddedFiles");
            } else {
                metrics.push(("numDeletedRows".to_owned(), deleted));
                metrics.push(("numCopiedRows".to_owned(), 0));
            }
            removed.rows
        }
        _ => {
            put("numDeletedRows", plan.affected);
            put("numCopiedRows", written.rows);
            put("numDeletionVectorsRemoved", removed.with_dv);
            put("rewriteTimeMs", rewrite_ms);
            Some(plan.affected)
        }
    };
    metrics.push((
        "executionTimeMs".to_owned(),
        plan.scan_ms.saturating_add(rewrite_ms),
    ));
    *plan.outcome.lock().unwrap_or_else(PoisonError::into_inner) = Some(Outcome {
        removed_files: removed.files,
        rows: outcome_rows,
        merge: None,
    });
    let mut read_set = ReadSet::default();
    match &plan.read_predicate {
        Some(p) => read_set.predicates.push(Arc::clone(p)),
        None => read_set.whole_table = true,
    }
    read_set.files = removed.actions;
    read_set.files.extend(also_read);
    Ok(Some(StagedCommit {
        table: table.name().to_owned(),
        write_id: write_id.to_owned(),
        idempotency_key: None,
        operation: plan.operation(),
        operation_parameters: vec![("predicate".to_owned(), Some(plan.predicate_parameter()))],
        operation_metrics: metrics
            .into_iter()
            .map(|(k, v)| (k, Some(v.to_string())))
            .collect(),
        adds: written.adds,
        removes,
        read_set,
    }))
}

/// Whether `unit`'s statistics prove it holds no live row (`numRecords` less its deletion
/// vector's cardinality is 0; unknown: no).
fn no_live_rows(unit: &ScanUnit) -> bool {
    let deleted = if unit.dv.has_vector() {
        match unit.dv.cardinality() {
            Ok(Some(n)) => n,
            _ => return false,
        }
    } else {
        0
    };
    unit.records
        .is_some_and(|records| records.saturating_sub(deleted) == 0)
}

/// The removed files' sizes, deletion vectors and live rows, from the table's listing.
fn totals(units: &[ScanUnit], actions: Vec<FileAction>) -> Removed {
    let by_path: HashMap<&str, &ScanUnit> = units.iter().map(|u| (u.path.as_str(), u)).collect();
    let mut r = Removed {
        rows: Some(0),
        ..Removed::default()
    };
    for a in &actions {
        r.files += 1;
        let Some(u) = by_path.get(a.path.as_str()) else {
            r.rows = None;
            continue;
        };
        r.bytes += u.size;
        if u.dv.has_vector() {
            r.with_dv += 1;
        }
        let deleted = u.dv.cardinality().ok().flatten().unwrap_or(0);
        r.rows = match (r.rows, u.records) {
            (Some(n), Some(records)) => Some(n + records.saturating_sub(deleted)),
            _ => None,
        };
    }
    r.actions = actions;
    r
}

/// The `path` column of scan metadata rows.
fn paths_of(batch: &RecordBatch) -> Result<Vec<Option<String>>, KrishivError> {
    let column = batch
        .column_by_name("path")
        .ok_or_else(|| internal("scan metadata without a path"))?;
    let column = arrow_cast::cast(column, &ArrowType::Utf8)
        .map_err(|e| internal(&format!("a scan metadata path: {e}")))?;
    let strings = column.as_string::<i32>();
    Ok((0..strings.len())
        .map(|i| strings.is_valid(i).then(|| strings.value(i).to_owned()))
        .collect())
}

fn millis(d: std::time::Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

fn internal(message: &str) -> KrishivError {
    InternalError {
        message: message.to_owned(),
    }
    .into()
}

/// The partition values of scan metadata row `i`, by physical name.
fn partition_values_of(
    batch: &RecordBatch,
    i: usize,
) -> Result<crate::commit::PartitionValues, KrishivError> {
    let values = batch
        .column_by_name("fileConstantValues")
        .and_then(|c| c.as_any().downcast_ref::<StructArray>())
        .and_then(|s| s.column_by_name("partitionValues"))
        .and_then(|c| c.as_any().downcast_ref::<MapArray>());
    match values {
        Some(m) if m.is_valid(i) => crate::commit::map_row(m, i),
        _ => Ok(HashMap::new()),
    }
}
