// SPDX-License-Identifier: Apache-2.0
//! The row-level planner over a test provider (design `P3-PR-DML-row-level.md` §5): files as
//! batches, a scan with the row id, and a commit that swaps the touched files for one new file.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, BinaryViewArray, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::executor::block_on;
use futures::future::BoxFuture;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_exec::planner::{
    CardinalityRule, MemoryTable, RowLevelCapabilities, RowLevelCounts, RowLevelOperation,
    RowLevelTable, RowLevelWrite, TableProvider,
};
use krishiv_exec::{
    GlobalState, LocalState, ROW_FILE_COLUMN, ROW_POSITION_COLUMN, ScanPushdown, Sink, SinkStatus,
    TaskContext,
};
use krishiv_plan::analyzer::TableColumn;
use krishiv_types::batch::Batch;

use super::Assignment;
use crate::session::Session;

/// Each rewrite's target and whether its request kept a `read_filter` (the partitions read), in
/// order: the end-to-end side of the re-read rule (review 681c2e038 nit 1).
pub(super) static READ_FILTERS: Mutex<Vec<(String, bool)>> = Mutex::new(Vec::new());

/// The table's files: name and rows (`id BIGINT, s STRING`).
#[derive(Debug)]
struct Files {
    files: Mutex<Vec<(Vec<u8>, RecordBatch)>>,
    commits: AtomicUsize,
    next: AtomicUsize,
    /// The table records changes as deletes by position (merge-on-read).
    mor: AtomicBool,
    /// Each rewrite request's `read_filter`, printed (`None`: every row).
    reads: Mutex<Vec<Option<String>>>,
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int64, true),
        Field::new("s", ArrowType::Utf8View, true),
    ]))
}

fn rows(ids: &[i64]) -> RecordBatch {
    let s: StringViewArray = ids.iter().map(|i| Some(format!("s{i}"))).collect();
    RecordBatch::try_new(
        schema(),
        vec![Arc::new(Int64Array::from(ids.to_vec())), Arc::new(s)],
    )
    .unwrap()
}

impl Files {
    /// Three files: `f0` holds 1..=3, `f1` 4..=6, `f2` 7..=9.
    fn new() -> Arc<Self> {
        Arc::new(Self {
            files: Mutex::new(vec![
                (b"f0".to_vec(), rows(&[1, 2, 3])),
                (b"f1".to_vec(), rows(&[4, 5, 6])),
                (b"f2".to_vec(), rows(&[7, 8, 9])),
            ]),
            commits: AtomicUsize::new(0),
            next: AtomicUsize::new(3),
            mor: AtomicBool::new(false),
            reads: Mutex::new(Vec::new()),
        })
    }

    fn names(&self) -> Vec<String> {
        self.lock()
            .iter()
            .map(|(n, _)| String::from_utf8_lossy(n).into_owned())
            .collect()
    }

    fn ids(&self) -> Vec<i64> {
        let mut out: Vec<i64> = self
            .lock()
            .iter()
            .flat_map(|(_, b)| b.column(0).as_primitive::<Int64Type>().values().to_vec())
            .collect();
        out.sort_unstable();
        out
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<(Vec<u8>, RecordBatch)>> {
        self.files.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The files `only` names (all without it), with or without the row id.
    fn table(&self, only: Option<&[Vec<u8>]>, row_id: bool) -> MemoryTable {
        let files = self.lock();
        let mut fields: Vec<Field> = schema().fields().iter().map(|f| (**f).clone()).collect();
        if row_id {
            fields.push(Field::new(ROW_FILE_COLUMN, ArrowType::BinaryView, false));
            fields.push(Field::new(ROW_POSITION_COLUMN, ArrowType::Int64, false));
        }
        let with = Arc::new(Schema::new(fields));
        let batches = files
            .iter()
            .filter(|(n, _)| only.is_none_or(|o| o.contains(n)))
            .map(|(n, b)| {
                let mut columns: Vec<ArrayRef> = b.columns().to_vec();
                if row_id {
                    let len = b.num_rows();
                    columns.push(Arc::new(BinaryViewArray::from_iter_values(
                        std::iter::repeat_n(n.as_slice(), len),
                    )));
                    columns.push(Arc::new(Int64Array::from_iter_values(0..len as i64)));
                }
                Batch::new(RecordBatch::try_new(Arc::clone(&with), columns).unwrap())
            })
            .collect();
        MemoryTable::new(with, batches)
    }
}

#[derive(Debug)]
struct Table {
    files: Arc<Files>,
    row_level: bool,
    output: bool,
}

impl TableProvider for Table {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        self.files.table(None, false).scan()
    }

    fn row_level(&self, _op: RowLevelOperation) -> Option<Arc<dyn RowLevelTable>> {
        self.row_level.then(|| {
            Arc::new(RowLevel(Arc::clone(&self.files), self.output)) as Arc<dyn RowLevelTable>
        })
    }
}

#[derive(Debug)]
struct RowLevel(Arc<Files>, bool);

impl RowLevelTable for RowLevel {
    fn capabilities(&self) -> RowLevelCapabilities {
        let mut c = RowLevelCapabilities::copy_on_write(
            CardinalityRule::DeltaWithUnconditionalDeleteExemption,
        );
        c.merge_on_read = self.0.mor.load(Ordering::Relaxed);
        c
    }

    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        self.0.table(None, true).scan()
    }

    fn scan_files(&self, files: &[Vec<u8>]) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        self.0.table(Some(files), true).scan()
    }

    fn rewriter(&self, request: &RowLevelWrite) -> krishiv_exec::Result<Arc<dyn Sink>> {
        if self.0.mor.load(Ordering::Relaxed) {
            return Ok(Arc::new(Swap {
                files: Arc::clone(&self.0),
                touched: Vec::new(),
                written: Mutex::new(Vec::new()),
                by_position: true,
            }));
        }
        self.0
            .reads
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(request.read_filter.as_ref().map(ToString::to_string));
        Ok(Arc::new(Swap {
            files: Arc::clone(&self.0),
            touched: request.touched.clone(),
            written: Mutex::new(Vec::new()),
            by_position: false,
        }))
    }

    fn output(&self, counts: &RowLevelCounts) -> Option<RecordBatch> {
        if !self.1 {
            return None;
        }
        let schema = Arc::new(Schema::new(vec![Field::new(
            "num_affected_rows",
            ArrowType::Int64,
            false,
        )]));
        let n = i64::try_from(counts.affected).unwrap();
        Some(RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![n]))]).unwrap())
    }
}

/// Collects the rewrite's rows; its finalize swaps the touched files for one new file.
#[derive(Debug)]
struct Swap {
    files: Arc<Files>,
    touched: Vec<Vec<u8>>,
    written: Mutex<Vec<RecordBatch>>,
    /// Merge-on-read: the last two columns of each row name a row to delete; any columns
    /// before them are the new values.
    by_position: bool,
}

impl Sink for Swap {
    fn global_state(&self, _ctx: &TaskContext) -> krishiv_exec::Result<GlobalState> {
        Ok(Arc::new(()))
    }

    fn local_state(
        &self,
        _ctx: &TaskContext,
        _global: &GlobalState,
    ) -> krishiv_exec::Result<LocalState> {
        Ok(Box::new(()))
    }

    fn sink(
        &self,
        input: Batch,
        _local: &mut LocalState,
        _global: &GlobalState,
    ) -> krishiv_exec::Result<SinkStatus> {
        self.written
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(input.into_parts().0);
        Ok(SinkStatus::NeedMoreInput)
    }

    fn combine(&self, _local: LocalState, _global: &GlobalState) -> krishiv_exec::Result<()> {
        Ok(())
    }

    fn finalize<'a>(
        &'a self,
        _global: &'a GlobalState,
        _ctx: &'a TaskContext,
    ) -> BoxFuture<'a, krishiv_exec::Result<()>> {
        let written =
            std::mem::take(&mut *self.written.lock().unwrap_or_else(PoisonError::into_inner));
        if self.by_position {
            let mut files = self.files.lock();
            let mut added = Vec::new();
            let mut deleted: std::collections::HashMap<Vec<u8>, Vec<i64>> =
                std::collections::HashMap::new();
            for b in &written {
                let n = b.num_columns();
                let file = b.column(n - 2).as_binary_view();
                let pos = b.column(n - 1).as_primitive::<Int64Type>();
                for r in 0..b.num_rows() {
                    deleted
                        .entry(file.value(r).to_vec())
                        .or_default()
                        .push(pos.value(r));
                }
                if n > 2 {
                    added.push(b.project(&(0..n - 2).collect::<Vec<_>>()).unwrap());
                }
            }
            for (name, rows) in files.iter_mut() {
                if let Some(gone) = deleted.get(name) {
                    let keep: Vec<bool> = (0..rows.num_rows())
                        .map(|i| !gone.contains(&(i as i64)))
                        .collect();
                    *rows = arrow_select::filter::filter_record_batch(
                        rows,
                        &arrow_array::BooleanArray::from(keep),
                    )
                    .unwrap();
                }
            }
            if !added.is_empty() {
                let batch = arrow_select::concat::concat_batches(&schema(), &added).unwrap();
                let n = self.files.next.fetch_add(1, Ordering::Relaxed);
                files.push((format!("f{n}").into_bytes(), batch));
            }
            files.retain(|(_, b)| b.num_rows() > 0);
            self.files.commits.fetch_add(1, Ordering::Relaxed);
            return Box::pin(std::future::ready(Ok(())));
        }
        let mut files = self.files.lock();
        files.retain(|(n, _)| !self.touched.contains(n));
        let rows = written.iter().map(RecordBatch::num_rows).sum::<usize>();
        if rows > 0 {
            let batch = arrow_select::concat::concat_batches(&schema(), &written).unwrap();
            let n = self.files.next.fetch_add(1, Ordering::Relaxed);
            files.push((format!("f{n}").into_bytes(), batch));
        }
        self.files.commits.fetch_add(1, Ordering::Relaxed);
        Box::pin(std::future::ready(Ok(())))
    }
}

/// A session reading `files` as the temporary table `t`.
fn session(files: &Arc<Files>) -> (Session, String) {
    session_with(files, true, true)
}

/// [`session`] with or without a row-level table and an output.
fn session_with(files: &Arc<Files>, row_level: bool, output: bool) -> (Session, String) {
    let session = Session::new().unwrap();
    let columns = vec![
        TableColumn {
            name: "id".to_owned(),
            data_type: krishiv_types::DataType::Long,
            nullable: true,
            char_varchar: None,
        },
        TableColumn {
            name: "s".to_owned(),
            data_type: krishiv_types::DataType::String,
            nullable: true,
            char_varchar: None,
        },
    ];
    session.state.catalog.put(
        "t",
        crate::catalog::TempTable::Provider {
            columns,
            table: Arc::new(Table {
                files: Arc::clone(files),
                row_level,
                output,
            }),
        },
    );
    let name = "t".to_owned();
    (session, name)
}

fn run(
    session: &Session,
    name: &str,
    op: RowLevelOperation,
    condition: Option<&str>,
    assignments: Vec<Assignment>,
) -> Vec<RecordBatch> {
    let df = try_run(session, name, op, "t", condition, assignments)
        .unwrap()
        .unwrap();
    block_on(df.collect()).unwrap()
}

fn try_run(
    session: &Session,
    name: &str,
    op: RowLevelOperation,
    alias: &str,
    condition: Option<&str>,
    assignments: Vec<Assignment>,
) -> Result<Option<crate::dataframe::DataFrame>, krishiv_common::error::KrishivError> {
    let s = session.clone();
    let quoted = format!("`{name}`");
    let alias = alias.to_owned();
    let condition = condition.map(str::to_owned);
    block_on(session.on_executor(async move {
        s.run_row_level(op, &quoted, &alias, condition.as_deref(), &assignments)
            .await
    }))
}

fn set(column: &str, value: &str) -> Assignment {
    Assignment {
        column: column.to_owned(),
        value: Some(value.to_owned()),
    }
}

fn affected(batches: &[RecordBatch]) -> i64 {
    batches[0].column(0).as_primitive::<Int64Type>().value(0)
}

#[test]
fn row_level_delete_rewrites_only_touched_files() {
    let files = Files::new();
    let (session, name) = session(&files);
    let out = run(
        &session,
        &name,
        RowLevelOperation::Delete,
        Some("t.id IN (2, 5, 6)"),
        vec![],
    );
    assert_eq!(affected(&out), 3);
    assert_eq!(files.ids(), [1, 3, 4, 7, 8, 9]);
    // f2 untouched; f0 and f1 replaced by one new file.
    assert_eq!(files.names(), ["f2", "f3"]);
    assert_eq!(files.commits.load(Ordering::Relaxed), 1);
}

#[test]
fn row_level_update_case_per_row() {
    let files = Files::new();
    let (session, name) = session(&files);
    // A NULL condition keeps the row as it is.
    let out = run(
        &session,
        &name,
        RowLevelOperation::Update,
        Some("CASE WHEN id = 9 THEN NULL ELSE id > 7 END"),
        vec![Assignment {
            column: "id".to_owned(),
            value: Some("id * 100".to_owned()),
        }],
    );
    assert_eq!(affected(&out), 1);
    assert_eq!(files.ids(), [1, 2, 3, 4, 5, 6, 7, 9, 800]);
    assert_eq!(files.names(), ["f0", "f1", "f3"]);
}

#[test]
fn row_level_empty_match_calls_rewriter() {
    let files = Files::new();
    let (session, name) = session(&files);
    let out = run(
        &session,
        &name,
        RowLevelOperation::Delete,
        Some("id > 100"),
        vec![],
    );
    assert_eq!(affected(&out), 0);
    assert_eq!(files.names(), ["f0", "f1", "f2"]);
    assert_eq!(files.commits.load(Ordering::Relaxed), 1);
}

#[test]
fn row_level_delete_without_condition_empties_the_table() {
    let files = Files::new();
    let (session, name) = session(&files);
    let out = run(&session, &name, RowLevelOperation::Delete, None, vec![]);
    assert_eq!(affected(&out), 9);
    assert!(files.ids().is_empty());
}

#[test]
fn row_level_failed_rewrite_commits_nothing() {
    let files = Files::new();
    let (session, name) = session(&files);
    // `s4` is no number: the ANSI store assignment fails the hit row.
    let e = try_run(
        &session,
        &name,
        RowLevelOperation::Update,
        "t",
        Some("id = 4"),
        vec![set("id", "CAST(s AS BIGINT) + 0")],
    )
    .unwrap_err();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::CastInvalidInput,
        "{e}"
    );
    assert_eq!(files.commits.load(Ordering::Relaxed), 0);
    assert_eq!(files.ids(), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn row_level_refusal_unchanged() {
    let files = Files::new();
    let (session, name) = session_with(&files, false, true);
    let none = try_run(
        &session,
        &name,
        RowLevelOperation::Delete,
        "t",
        None,
        vec![],
    )
    .unwrap();
    assert!(none.is_none());
    assert_eq!(files.commits.load(Ordering::Relaxed), 0);
}

#[test]
fn a_session_idempotency_key_is_refused_on_row_level_writes() {
    let files = Files::new();
    let (session, name) = session(&files);
    session
        .conf()
        .set("krishiv.write.idempotencyKey", "job-1")
        .unwrap();
    let e = try_run(
        &session,
        &name,
        RowLevelOperation::Delete,
        "t",
        Some("id = 1"),
        vec![],
    )
    .expect_err("refused");
    assert!(e.message().contains("idempotencyKey"), "{e}");
    assert_eq!(files.commits.load(Ordering::Relaxed), 0);
}

#[test]
fn a_session_idempotency_key_is_refused_on_merge_on_read_writes() {
    let files = Files::new();
    files.mor.store(true, Ordering::Relaxed);
    let (session, name) = session(&files);
    session
        .conf()
        .set("krishiv.write.idempotencyKey", "job-1")
        .unwrap();
    let e = try_run(
        &session,
        &name,
        RowLevelOperation::Delete,
        "t",
        Some("id = 1"),
        vec![],
    )
    .expect_err("refused");
    assert!(e.message().contains("idempotencyKey"), "{e}");
    assert_eq!(files.commits.load(Ordering::Relaxed), 0);
}

#[test]
fn row_level_counts_output() {
    // A table without an output: no rows, no columns.
    let files = Files::new();
    let (session, name) = session_with(&files, true, false);
    let df = try_run(
        &session,
        &name,
        RowLevelOperation::Delete,
        "t",
        Some("id = 1"),
        vec![],
    )
    .unwrap()
    .unwrap();
    let batches = block_on(df.collect()).unwrap();
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
    assert_eq!(files.ids(), [2, 3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn the_condition_and_values_resolve_under_the_statements_alias() {
    let files = Files::new();
    let (session, name) = session(&files);
    let df = try_run(
        &session,
        &name,
        RowLevelOperation::Update,
        "x",
        Some("x.id = 2"),
        vec![set("s", "concat(x.s, '!')")],
    )
    .unwrap()
    .unwrap();
    let batches = block_on(df.collect()).unwrap();
    assert_eq!(affected(&batches), 1);
}

/// The new value is computed for hit rows only: here it divides by zero on the row before.
#[test]
fn a_new_value_is_evaluated_on_hit_rows_only() {
    let files = Files::new();
    let (session, name) = session(&files);
    let out = run(
        &session,
        &name,
        RowLevelOperation::Update,
        Some("id = 5"),
        vec![set("id", "10 div (id - 4)")],
    );
    assert_eq!(affected(&out), 1);
    assert_eq!(files.ids(), [1, 2, 3, 4, 6, 7, 8, 9, 10]);
}

#[test]
fn assignments_are_checked_before_anything_runs() {
    use krishiv_common::error::ErrorClass;
    let files = Files::new();
    let (session, name) = session(&files);
    let class = |cond: Option<&str>, sets: Vec<Assignment>| {
        try_run(&session, &name, RowLevelOperation::Update, "t", cond, sets)
            .unwrap_err()
            .error_class()
    };
    assert_eq!(
        class(None, vec![set("id", "1"), set("ID", "2")]),
        ErrorClass::DuplicateAssignments
    );
    assert_eq!(
        class(None, vec![set("id", "'x'")]),
        ErrorClass::IncompatibleDataForTableCannotSafelyCast
    );
    assert_eq!(
        class(Some("rand() > 0.5"), vec![set("id", "1")]),
        ErrorClass::KrishivUnsupportedFeature
    );
    assert_eq!(files.commits.load(Ordering::Relaxed), 0);
}

#[test]
fn merge_on_read_delete_records_positions_and_rewrites_nothing() {
    let files = Files::new();
    files.mor.store(true, Ordering::Relaxed);
    let (session, name) = session(&files);
    let out = run(
        &session,
        &name,
        RowLevelOperation::Delete,
        Some("t.id IN (2, 5, 6)"),
        vec![],
    );
    // The hit count is not known before the commit.
    assert_eq!(affected(&out), 0);
    assert_eq!(files.ids(), [1, 3, 4, 7, 8, 9]);
    // The files keep their names: no copy-on-write swap.
    assert_eq!(files.names(), ["f0", "f1", "f2"]);
    assert_eq!(files.commits.load(Ordering::Relaxed), 1);
}

#[test]
fn merge_on_read_update_deletes_old_rows_and_adds_new_ones() {
    let files = Files::new();
    files.mor.store(true, Ordering::Relaxed);
    let (session, name) = session(&files);
    run(
        &session,
        &name,
        RowLevelOperation::Update,
        Some("t.id IN (2, 8)"),
        vec![set("s", "'x'")],
    );
    assert_eq!(files.ids(), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
    assert_eq!(files.names(), ["f0", "f1", "f2", "f3"]);
    let f3 = files.lock().last().unwrap().1.clone();
    assert_eq!(f3.num_rows(), 2);
    assert_eq!(files.commits.load(Ordering::Relaxed), 1);
}

/// `has_subquery` finds a subquery at any depth of a condition, and only then.
#[test]
fn a_condition_holds_a_subquery_only_when_one_is_written() {
    let parsed = |c: &str| crate::functions::expr(c).map(|e| e.expr);
    for c in [
        "id IN (SELECT 1)",
        "s = 'a' AND NOT EXISTS (SELECT 1)",
        "id > (SELECT max(1))",
    ] {
        assert!(super::has_subquery(&parsed(c).unwrap()), "{c}");
    }
    for c in ["id = 1", "s IN ('a', 'b') OR id > 2"] {
        assert!(!super::has_subquery(&parsed(c).unwrap()), "{c}");
    }
}

fn overflow() -> KrishivError {
    krishiv_common::error::ArithmeticOverflow {
        message: "overflow".to_owned(),
        alternative: String::new(),
        config: "\"spark.sql.ansi.enabled\"".to_owned(),
    }
    .into()
}

fn cols() -> Vec<TableColumn> {
    use krishiv_types::DataType;
    vec![
        TableColumn::new("s", DataType::String, true),
        TableColumn::new("small", DataType::Short, true),
        TableColumn::new(
            "`d`",
            DataType::Decimal(krishiv_types::DecimalType::try_new(5, 2).unwrap()),
            true,
        ),
    ]
}

/// A Delta rewrite's own errors stay as they came: the commit's `DELTA_*` conflicts, Krishiv's
/// refusals, `UNSUPPORTED_FEATURE`, internal errors and cancellations.
#[test]
fn delta_rewrite_errors_that_are_not_the_tasks_are_not_wrapped() {
    use krishiv_common::error::{
        DeltaConcurrentAppendWithoutHint, DeltaUnsupportedSubquery, InternalError,
        KrishivUnsupportedFeature, OperationCanceled, UnsupportedFeatureTableOperation,
    };
    let kept: Vec<KrishivError> = vec![
        // A commit conflict.
        DeltaConcurrentAppendWithoutHint {
            operation: "UPDATE".into(),
            table_name: "t".into(),
            version: "3".into(),
            doc_link: "d".into(),
        }
        .into(),
        DeltaUnsupportedSubquery {
            operation: "DELETE".into(),
            cond: "x".into(),
        }
        .into(),
        KrishivUnsupportedFeature {
            feature: "f".into(),
        }
        .into(),
        // The change-data-feed refusal: `UNSUPPORTED_FEATURE.TABLE_OPERATION`.
        UnsupportedFeatureTableOperation {
            table_name: "t".into(),
            operation: "UPDATE".into(),
        }
        .into(),
        InternalError {
            message: "m".into(),
        }
        .into(),
        OperationCanceled {}.into(),
    ];
    for e in kept {
        let class = e.error_class();
        let out = super::delta_task_write_failed(e, "file:/t", &cols(), &[]);
        assert_eq!(out.error_class(), class);
        assert!(out.cause().is_none());
    }
}

/// Any other error of the rewrite is `TASK_WRITE_FAILED` naming the root, caused by the error.
#[test]
fn a_delta_rewrite_error_is_task_write_failed_caused_by_it() {
    let e: KrishivError = krishiv_common::error::DivideByZero {
        config: "\"spark.sql.ansi.enabled\"".into(),
    }
    .into();
    // No assigned value under a cast that can overflow: the cause is the error itself.
    let out = super::delta_task_write_failed(
        e,
        "file:/t",
        &cols(),
        &[("s".into(), krishiv_types::DataType::String)],
    );
    assert_eq!(out.error_class(), ErrorClass::TaskWriteFailed);
    assert!(
        out.message_parameters()
            .iter()
            .any(|(k, v)| *k == "path" && v == "file:/t")
    );
    assert_eq!(
        out.cause().map(KrishivError::error_class),
        Some(ErrorClass::DivideByZero)
    );
}

/// An arithmetic error with an assigned value whose store-assignment cast can overflow (a
/// narrowing integral cast, or into a decimal) is `DELTA_CAST_OVERFLOW_IN_TABLE_WRITE` naming
/// the first such column in table order; an up-cast or the same type never is.
#[test]
fn an_overflow_under_a_narrowing_store_cast_is_delta_cast_overflow() {
    use krishiv_types::DataType;
    let out = super::delta_task_write_failed(
        overflow(),
        "file:/t",
        &cols(),
        // Assigned out of table order: the first such column in table order is named.
        &[
            ("`d`".into(), DataType::Long),
            ("SMALL".into(), DataType::Integer),
        ],
    );
    assert_eq!(out.error_class(), ErrorClass::TaskWriteFailed);
    let cause = out.cause().unwrap();
    assert_eq!(
        cause.error_class(),
        ErrorClass::DeltaCastOverflowInTableWrite
    );
    let p = |k: &str| {
        cause
            .message_parameters()
            .iter()
            .find(|(n, _)| *n == k)
            .map(|(_, v)| v.clone())
    };
    assert_eq!(p("columnName").as_deref(), Some("`small`"));
    assert_eq!(p("sourceType").as_deref(), Some("\"INT\""));
    assert_eq!(p("targetType").as_deref(), Some("\"SMALLINT\""));

    // Into a decimal from an integer: can overflow; a backquote in the name is doubled.
    let cause =
        super::store_assignment_overflow(&overflow(), &cols(), &[("`d`".into(), DataType::Long)])
            .unwrap();
    assert!(
        cause
            .message_parameters()
            .iter()
            .any(|(k, v)| *k == "columnName" && v == "```d```")
    );

    // An up-cast, the same type, a non-integral target, or a non-arithmetic error: no rewrap.
    for values in [
        vec![("small".to_owned(), DataType::Byte)],
        vec![("small".to_owned(), DataType::Short)],
        vec![("s".to_owned(), DataType::Integer)],
        vec![],
    ] {
        assert!(super::store_assignment_overflow(&overflow(), &cols(), &values).is_none());
    }
    let not_arith: KrishivError = krishiv_common::error::InternalError {
        message: "m".into(),
    }
    .into();
    assert!(
        super::store_assignment_overflow(
            &not_arith,
            &cols(),
            &[("small".into(), DataType::Integer)]
        )
        .is_none()
    );
}

/// Review of #909 (Codex P1; 66ad1730d, P2): an `UPDATE` whose subquery reads the target table
/// again reads every row, so a concurrent change anywhere in the table conflicts; a subquery over
/// another table, or none, leaves the condition as the read set (delta-spark records nothing on
/// the target for it).
#[test]
fn only_a_subquery_over_the_target_reads_the_whole_table() {
    let files = Files::new();
    let (session, name) = session(&files);
    let other = Arc::new(Schema::new(vec![Field::new("x", ArrowType::Utf8, true)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&other),
        vec![Arc::new(arrow_array::StringArray::from(vec!["z"])) as ArrayRef],
    )
    .unwrap();
    session.register_batches("o", other, vec![batch]).unwrap();
    let update = |value: &str| {
        run(
            &session,
            &name,
            RowLevelOperation::Update,
            Some("id = 1"),
            vec![set("s", value)],
        );
        files
            .reads
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop()
            .unwrap()
    };
    assert_eq!(update("(SELECT max(s) FROM t)"), None);
    assert_eq!(update("(SELECT max(s) FROM t WHERE id = 5)"), None);
    let read = update("(SELECT max(x) FROM o)");
    assert!(
        read.as_deref().is_some_and(|f| f.contains("id")),
        "{read:?}"
    );
    let read = update("'v'");
    assert!(
        read.as_deref().is_some_and(|f| f.contains("id")),
        "{read:?}"
    );
}

/// Review of #909 (9bc733547, P1): a cast can overflow exactly when delta-spark's
/// `canCauseCastOverflow` says so: the target holds an integral or decimal type at any depth and
/// the value does not `Cast.canUpCast` to it.
#[test]
fn a_cast_can_overflow_as_delta_spark_decides() {
    use krishiv_types::{ArrayType, DataType as T, DecimalType};
    let dec = |p, s| T::Decimal(DecimalType::try_new(p, s).unwrap());
    let array = |t: T| T::Array(Arc::new(ArrayType::new(t, true)));
    for (value, target, rewrap) in [
        (T::Integer, dec(10, 0), false),
        (T::Integer, dec(9, 0), true),
        (T::Long, dec(20, 0), false),
        (dec(5, 2), dec(10, 2), false),
        (dec(10, 2), dec(5, 2), true),
        (dec(5, 0), T::Integer, false),
        (T::Null, T::Short, false),
        (T::Integer, T::Short, true),
        (T::Short, T::Integer, false),
        (array(T::Long), array(T::Integer), true),
        (array(T::Integer), array(T::Long), false),
        (T::Double, T::String, false),
    ] {
        let columns = [TableColumn::new("c", target.clone(), true)];
        let out =
            super::store_assignment_overflow(&overflow(), &columns, &[("c".into(), value.clone())]);
        assert_eq!(out.is_some(), rewrap, "{value:?} into {target:?}");
    }
}

/// Review of #909 (055310eb4, P1): a Delta target is the table at its resolved root, so a subquery that
/// reads it under another spelling (`/t/`, `file:/t`) or by the catalog name over that location is a
/// re-read of the target (the read set is then the whole table), while another table, or no read at
/// all, is not.
#[cfg(feature = "delta")]
#[test]
fn a_delta_target_is_compared_by_its_root_not_its_spelling() {
    let dir = tempfile::tempdir().unwrap();
    let (root, other) = (dir.path().join("t"), dir.path().join("o"));
    let s = Session::builder().build().unwrap();
    s.conf()
        .set(
            "spark.sql.warehouse.dir",
            &dir.path().join("wh").display().to_string(),
        )
        .unwrap();
    for (name, path) in [("dt", &root), ("other_t", &other)] {
        let create = format!(
            "CREATE TABLE {name} USING delta LOCATION '{}' AS SELECT 1 AS p, 2 AS v",
            path.display()
        );
        block_on(async { s.sql(&create).await?.collect().await }).unwrap();
    }
    let t = root.display().to_string();
    let target = block_on(async {
        let a = s
            .analyze_variable_query(&format!("SELECT * FROM delta.`{t}`"))
            .await?;
        let (_, physical) = s.physical(&a.analyzed, &a.env).await?;
        let scan = super::first_scan(physical.root()).unwrap();
        a.env.catalog.delta.hadoop_root(&scan.source)
    })
    .unwrap();
    let rereads = |subquery: &str| {
        block_on(async {
            let a = s
                .analyze_variable_query(&format!("SELECT 1 AS x WHERE 2 = ({subquery})"))
                .await?;
            s.physical(&a.analyzed, &a.env).await?;
            Ok::<_, KrishivError>(crate::catalog::reads_delta_root(
                a.analyzed.plan.root(),
                &target,
                &a.env.catalog,
            ))
        })
        .unwrap()
    };
    for same in [
        format!("SELECT max(v) FROM delta.`{t}`"),
        format!("SELECT max(v) FROM delta.`{t}/`"),
        format!("SELECT max(v) FROM delta.`file:{t}`"),
        "SELECT max(v) FROM dt".to_owned(),
    ] {
        assert!(rereads(&same), "{same}");
    }
    for different in [
        format!("SELECT max(v) FROM delta.`{}`", other.display()),
        "SELECT max(v) FROM other_t".to_owned(),
        "SELECT 2".to_owned(),
    ] {
        assert!(!rereads(&different), "{different}");
    }
}

/// On a real rewrite plan (the hidden provider in it), an `UPDATE` whose `SET` subquery reads
/// another Delta table keeps its condition as the read filter, and one that reads the target again
/// under another spelling reads the whole table (review 681c2e038 nit 1).
#[cfg(feature = "delta")]
#[test]
fn a_delta_update_reads_the_whole_table_only_when_a_subquery_reads_the_target() {
    let dir = tempfile::tempdir().unwrap();
    let (root, other) = (dir.path().join("rt"), dir.path().join("ro"));
    let s = Session::builder().build().unwrap();
    let sql = |text: &str| {
        block_on(async { s.sql(text).await?.collect().await }).unwrap();
    };
    s.conf()
        .set(
            "spark.sql.warehouse.dir",
            &dir.path().join("wh").display().to_string(),
        )
        .unwrap();
    for (name, path) in [("rt", &root), ("ro", &other)] {
        sql(&format!(
            "CREATE TABLE {name} USING delta PARTITIONED BY (p) LOCATION '{}' AS SELECT 1 AS v, 1 AS p UNION ALL SELECT 2, 2",
            path.display()
        ));
    }
    let (t, o) = (root.display().to_string(), other.display().to_string());
    let filters_of = |update: &str, target: &str| {
        let before = READ_FILTERS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len();
        sql(update);
        let seen = READ_FILTERS.lock().unwrap_or_else(PoisonError::into_inner);
        seen[before..]
            .iter()
            .filter(|(name, _)| name.contains(target))
            .map(|(_, kept)| *kept)
            .collect::<Vec<_>>()
    };
    let filters = |update: &str| filters_of(update, &t);
    assert_eq!(
        filters(&format!(
            "UPDATE delta.`{t}` SET v = (SELECT max(v) FROM delta.`{o}`) WHERE p = 1"
        )),
        vec![true]
    );
    assert_eq!(
        filters(&format!(
            "UPDATE delta.`{t}` SET v = (SELECT max(v) FROM delta.`file:{t}/`) WHERE p = 1"
        )),
        vec![false]
    );
    // A catalog table over a Delta location is the same table as its path (#909 review thread):
    // its own root is resolved through the catalog, so a re-read by path reads the whole table.
    assert_eq!(
        filters_of(
            &format!("UPDATE rt SET v = (SELECT max(v) FROM delta.`{t}`) WHERE p = 1"),
            "`rt`"
        ),
        vec![false]
    );
    assert_eq!(
        filters_of(
            &format!("UPDATE rt SET v = (SELECT max(v) FROM delta.`{o}`) WHERE p = 1"),
            "`rt`"
        ),
        vec![true]
    );
}
