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
use krishiv_common::error::KrishivError;
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
pub(crate) static READ_FILTERS: Mutex<Vec<(String, bool)>> = Mutex::new(Vec::new());

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
    put_table(Session::new().unwrap(), files, row_level, output)
}

/// `session` with the table `t` over `files`.
fn put_table(
    session: Session,
    files: &Arc<Files>,
    row_level: bool,
    output: bool,
) -> (Session, String) {
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
        nested: false,
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
            nested: false,
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

/// Under `spark.sql.ansi.doubleQuotedIdentifiers` a double-quoted name in the condition and in a
/// value is the column, in the rewrite as in the analysis: the texts are read again under the
/// statement's parser options, for a copy-on-write and a merge-on-read table alike (#962's
/// CodeRabbit finding on MERGE, here for DELETE and UPDATE).
#[test]
fn a_double_quoted_name_reads_the_column_under_ansi_identifiers() {
    let strings = |batches: &[RecordBatch]| -> Vec<String> {
        let mut out: Vec<(i64, String)> = batches
            .iter()
            .flat_map(|b| {
                let ids = b.column(0).as_primitive::<Int64Type>().clone();
                let s = b.column(1).as_string_view().clone();
                (0..b.num_rows())
                    .map(|i| (ids.value(i), s.value(i).to_owned()))
                    .collect::<Vec<_>>()
            })
            .collect();
        out.sort();
        out.into_iter().map(|(_, s)| s).collect()
    };
    for mor in [false, true] {
        let files = Files::new();
        files.mor.store(mor, Ordering::Relaxed);
        let (session, name) = session(&files);
        session
            .conf()
            .set("spark.sql.ansi.doubleQuotedIdentifiers", "true")
            .unwrap();
        run(
            &session,
            &name,
            RowLevelOperation::Update,
            Some("\"id\" IN (2, 8)"),
            vec![set("s", "concat(\"s\", '!')")],
        );
        let batches: Vec<RecordBatch> = files.lock().iter().map(|(_, b)| b.clone()).collect();
        if mor {
            // The new rows only: the old ones are deleted by position.
            assert_eq!(strings(&batches[3..]), ["s2!", "s8!"]);
        } else {
            let want: Vec<String> = (1..=9)
                .map(|i| match i {
                    2 | 8 => format!("s{i}!"),
                    _ => format!("s{i}"),
                })
                .collect();
            assert_eq!(strings(&batches), want);
            run(
                &session,
                &name,
                RowLevelOperation::Delete,
                Some("\"id\" = 5"),
                vec![],
            );
            assert_eq!(files.ids(), [1, 2, 3, 4, 6, 7, 8, 9]);
        }
    }
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

/// An arithmetic error with an assigned value whose store-assignment cast can overflow (a
/// narrowing integral cast, or into a decimal) names the first such column in table order with
/// the value's and the column's types; an up-cast, the same type, a non-integral target or a
/// non-arithmetic error never does. The table turns it into its own class (Delta:
/// `DELTA_CAST_OVERFLOW_IN_TABLE_WRITE`, krishiv-connectors-delta `write/tests/dml.rs`).
#[test]
fn an_overflow_under_a_narrowing_store_cast_names_the_column() {
    use krishiv_types::DataType;
    let (error, columns) = (overflow(), cols());
    // Assigned out of table order: the first such column in table order is named.
    let values = [
        ("`d`".to_owned(), DataType::Long),
        ("SMALL".to_owned(), DataType::Integer),
    ];
    let found = super::store_assignment_overflow(&error, &columns, &values);
    assert_eq!(found, Some(("small", &DataType::Integer, &DataType::Short)));
    let values = [("`d`".to_owned(), DataType::Long)];
    let found = super::store_assignment_overflow(&error, &columns, &values);
    assert_eq!(found.map(|(c, _, _)| c), Some("`d`"));
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
        let (error, values) = (overflow(), [("c".to_owned(), value.clone())]);
        let out = super::store_assignment_overflow(&error, &columns, &values);
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
    // The same rule for a MERGE (#962 review nit 2): a source that reads the target again under
    // another spelling of its root, or through the catalog table over it, reads every row.
    let merge = |source: &str| {
        format!(
            "MERGE INTO delta.`{t}` t USING ({source}) s ON t.p = s.p AND t.p = 1 \
             WHEN MATCHED THEN UPDATE SET v = s.v"
        )
    };
    assert_eq!(
        filters(&merge(&format!("SELECT v, p FROM delta.`{o}`"))),
        vec![true]
    );
    for again in [
        format!("delta.`file:{t}/`"),
        format!("delta.`{t}/`"),
        "rt".to_owned(),
    ] {
        assert_eq!(
            filters(&merge(&format!("SELECT v, p FROM {again}"))),
            vec![false],
            "{again}"
        );
    }
}

mod merge {
    use super::*;
    use crate::table_commands::merge::{ClauseAction, ClauseKind, MergeClause, MergeStatement};

    const SOURCE: &str =
        "SELECT CAST(id AS BIGINT) AS id, s FROM VALUES (2, 'u2'), (10, 'n10') AS v(id, s)";

    fn clause(kind: ClauseKind, condition: Option<&str>, action: ClauseAction) -> MergeClause {
        MergeClause {
            kind,
            condition: condition.map(str::to_owned),
            action,
        }
    }

    fn statement(source: &str, clauses: Vec<MergeClause>) -> MergeStatement {
        MergeStatement {
            target: "`t`".to_owned(),
            target_alias: "t".to_owned(),
            source: source.to_owned(),
            source_alias: "src".to_owned(),
            on: "t.id = src.id".to_owned(),
            clauses,
        }
    }

    fn try_merge(
        session: &Session,
        m: MergeStatement,
    ) -> Result<Option<crate::dataframe::DataFrame>, krishiv_common::error::KrishivError> {
        let s = session.clone();
        // As the statement runs it: the hidden tables go at its end (`Swept`).
        block_on(session.on_executor(async move {
            crate::table_commands::row_level::Swept::new(
                Arc::clone(&s.state.catalog),
                s.run_merge(&m),
            )
            .await
        }))
    }

    fn merge(session: &Session, m: MergeStatement) -> Vec<RecordBatch> {
        let df = try_merge(session, m).unwrap().unwrap();
        block_on(df.collect()).unwrap()
    }

    fn strings(files: &Files) -> Vec<(i64, String)> {
        let mut out: Vec<(i64, String)> = files
            .lock()
            .iter()
            .flat_map(|(_, b)| {
                let ids = b.column(0).as_primitive::<Int64Type>().values().to_vec();
                let s: Vec<String> = (0..b.num_rows())
                    .map(|i| b.column(1).as_string_view().value(i).to_owned())
                    .collect();
                ids.into_iter().zip(s).collect::<Vec<_>>()
            })
            .collect();
        out.sort();
        out
    }

    #[test]
    fn a_sql_merge_builds_qualified_insert_and_set_columns_before_its_refusal() {
        // `exec_merge` builds every clause from the SQL text (an `INSERT` column qualified by the
        // target's alias, or without one by its name, `namespace.name` or
        // `spark_catalog.namespace.name`; a `SET` column qualified the same way) before it asks the
        // table for a row-level table. A Parquet table has none, so the statement reaches the old
        // refusal with every clause built.
        let dir = tempfile::tempdir().unwrap();
        let session = Session::new().unwrap();
        block_on(session.sql(&format!(
            "CREATE TABLE p (id BIGINT, s STRING) USING parquet LOCATION '{}'",
            dir.path().display()
        )))
        .unwrap();
        for (target, q) in [
            ("p AS x", "x"),
            ("p", "p"),
            ("p", "default.p"),
            ("p", "spark_catalog.default.p"),
            ("p", "P"),
        ] {
            let sql = format!(
                "MERGE INTO {target} USING ({SOURCE}) src ON {q}.id = src.id \
                 WHEN MATCHED THEN UPDATE SET {q}.s = src.s \
                 WHEN NOT MATCHED THEN INSERT ({q}.id, s) VALUES (src.id, DEFAULT)"
            );
            let e = block_on(async { session.sql(&sql).await?.collect().await }).unwrap_err();
            assert_eq!(
                format!("{:?}", e.error_class()),
                "UnsupportedFeatureTableOperation",
                "{target} / {q}: {e}"
            );
        }
    }

    #[test]
    fn merge_updates_matched_rows_and_inserts_the_others_rewriting_only_touched_files() {
        let files = Files::new();
        let (session, _) = session(&files);
        let out = merge(
            &session,
            statement(
                SOURCE,
                vec![
                    clause(
                        ClauseKind::Matched,
                        None,
                        ClauseAction::Update(vec![set("s", "src.s")]),
                    ),
                    clause(ClauseKind::NotMatched, None, ClauseAction::InsertStar),
                ],
            ),
        );
        assert_eq!(affected(&out), 2);
        assert_eq!(files.ids(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let rows = strings(&files);
        assert!(rows.contains(&(2, "u2".to_owned())), "{rows:?}");
        assert!(rows.contains(&(10, "n10".to_owned())), "{rows:?}");
        // f0 (ids 1..=3) replaced; f1, f2 untouched.
        assert_eq!(files.names(), ["f1", "f2", "f3"]);
        assert_eq!(files.commits.load(Ordering::Relaxed), 1);
        assert_eq!(session.memory_reserved(), 0);
    }

    #[test]
    fn merge_clauses_fire_in_order_within_their_kind_and_by_source_rows_are_reached() {
        let files = Files::new();
        let (session, _) = session(&files);
        let out = merge(
            &session,
            statement(
                SOURCE,
                vec![
                    clause(
                        ClauseKind::Matched,
                        Some("src.s = 'u2'"),
                        ClauseAction::Delete,
                    ),
                    clause(
                        ClauseKind::Matched,
                        None,
                        ClauseAction::Update(vec![set("s", "'never'")]),
                    ),
                    clause(
                        ClauseKind::NotMatchedBySource,
                        Some("t.id > 8"),
                        ClauseAction::Delete,
                    ),
                    clause(
                        ClauseKind::NotMatchedBySource,
                        Some("t.id = 1"),
                        ClauseAction::Update(vec![set("s", "'by source'")]),
                    ),
                ],
            ),
        );
        // id 2 deleted (first matched clause), 9 deleted, 1 updated; 10 not inserted.
        assert_eq!(affected(&out), 3);
        assert_eq!(files.ids(), [1, 3, 4, 5, 6, 7, 8]);
        assert!(strings(&files).contains(&(1, "by source".to_owned())));
        // A `NOT MATCHED BY SOURCE` clause touches every file.
        assert_eq!(files.names(), ["f3"]);
    }

    #[test]
    fn one_sided_clauses_read_unqualified_names_from_their_own_side() {
        // Target and source both have `id` and `s`. A `NOT MATCHED` clause sees only the source
        // and a `NOT MATCHED BY SOURCE` clause only the target, so their bare names are not
        // ambiguous, even though the rewrite reads the join of both (review of #962).
        let files = Files::new();
        let (session, _) = session(&files);
        let out = merge(
            &session,
            statement(
                "SELECT CAST(id AS BIGINT) AS id, s FROM VALUES (2, 'u2'), (10, 'n10') AS v(id, s)",
                vec![
                    clause(
                        ClauseKind::Matched,
                        None,
                        ClauseAction::Update(vec![set("s", "src.s")]),
                    ),
                    clause(
                        ClauseKind::NotMatched,
                        Some("s = 'n10'"),
                        ClauseAction::Insert(vec![set("id", "id"), set("s", "concat(s, '!')")]),
                    ),
                    clause(
                        ClauseKind::NotMatchedBySource,
                        Some("id = 9"),
                        ClauseAction::Update(vec![set("s", "concat(s, '?')")]),
                    ),
                ],
            ),
        );
        assert_eq!(affected(&out), 3);
        assert_eq!(files.ids(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let rows = strings(&files);
        assert!(rows.contains(&(2, "u2".to_owned())), "{rows:?}");
        assert!(rows.contains(&(10, "n10!".to_owned())), "{rows:?}");
        assert!(rows.contains(&(9, "s9?".to_owned())), "{rows:?}");
        assert_eq!(session.memory_reserved(), 0);
    }

    #[test]
    fn a_lambda_variable_shadows_a_column_only_inside_its_lambda() {
        // `id` is both a column of the two sides and the lambda's variable. The bare `id` outside
        // the lambda is the target's (qualified by its side); the variable inside stays as it is
        // (review of #962: a name a lambda bound was left unqualified in the whole clause, so the
        // outer `id` was ambiguous in the join).
        let files = Files::new();
        let (session, _) = session(&files);
        let out = merge(
            &session,
            statement(
                "SELECT CAST(id AS BIGINT) AS id, s FROM VALUES (2, 'u2') AS v(id, s)",
                vec![clause(
                    ClauseKind::NotMatchedBySource,
                    Some("id = 9 AND NOT exists(array(1, 2), id -> id > 5)"),
                    ClauseAction::Update(vec![set(
                        "s",
                        "concat(s, CAST(size(filter(array(7, 8, 9), id -> id >= 8)) AS STRING))",
                    )]),
                )],
            ),
        );
        assert_eq!(affected(&out), 1);
        let rows = strings(&files);
        assert!(rows.contains(&(9, "s92".to_owned())), "{rows:?}");
        assert_eq!(session.memory_reserved(), 0);
    }

    /// A MERGE cancelled right after its source's first spill write (the copy waits there, a
    /// test hook) ends with the cancel's class, commits nothing, gives the pool back and leaves
    /// no spill file.
    #[test]
    fn a_merge_cancelled_after_its_first_source_spill_write_leaves_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let spill = tmp.path().join("spill");
        let runtime = crate::EngineRuntime::builder()
            .config("krishiv.spill.dir", spill.to_str().unwrap())
            .build()
            .unwrap();
        let files = Files::new();
        let (session, _) = put_table(
            Session::builder().runtime(runtime).build().unwrap(),
            &files,
            true,
            true,
        );
        let (reached, at_write) = tokio::sync::oneshot::channel();
        let (resume, resumed) = tokio::sync::oneshot::channel();
        *crate::table_commands::merge_source::AFTER_FIRST_WRITE
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some((reached, resumed));
        let canceller = {
            let s = session.clone();
            std::thread::spawn(move || {
                // An error: the merge ended without reaching the hook (cleared below).
                if block_on(at_write).is_ok() {
                    s.interrupt_all();
                    resume.send(()).unwrap();
                }
            })
        };
        let e = try_merge(
            &session,
            statement(
                SOURCE,
                vec![
                    clause(
                        ClauseKind::Matched,
                        None,
                        ClauseAction::Update(vec![set("s", "src.s")]),
                    ),
                    clause(ClauseKind::NotMatched, None, ClauseAction::InsertStar),
                ],
            ),
        )
        .unwrap_err();
        let unreached = crate::table_commands::merge_source::AFTER_FIRST_WRITE
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .is_some();
        canceller.join().unwrap();
        assert!(!unreached, "the source was never spilled");
        assert_eq!(e.sql_state(), Some("HY008"), "{e}");
        assert_eq!(session.memory_reserved(), 0);
        assert_eq!(files.commits.load(Ordering::Relaxed), 0);
        assert_eq!(files.names(), ["f0", "f1", "f2"]);
        let mut left = Vec::new();
        let mut dirs = vec![spill];
        while let Some(d) = dirs.pop() {
            let Ok(entries) = std::fs::read_dir(&d) else {
                continue;
            };
            for entry in entries {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    dirs.push(path);
                } else if path.extension().is_none_or(|e| e != "lock") {
                    // The `.lock` is the runtime's own, held while it lives.
                    left.push(path);
                }
            }
        }
        assert!(left.is_empty(), "{left:?}");
    }

    #[test]
    fn multiple_matches_raise_deltas_class_unless_the_only_matched_clause_deletes() {
        use krishiv_common::error::ErrorClass;
        let dup = "SELECT CAST(id AS BIGINT) AS id, s FROM VALUES (2, 'a'), (2, 'b') AS v(id, s)";
        let files = Files::new();
        let (session, _) = session(&files);
        let e = try_merge(
            &session,
            statement(
                dup,
                vec![clause(
                    ClauseKind::Matched,
                    Some("src.s = 'a'"),
                    ClauseAction::Update(vec![set("s", "src.s")]),
                )],
            ),
        )
        .unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::DeltaMultipleSourceRowMatchingTargetRowInMerge
        );
        assert_eq!(files.commits.load(Ordering::Relaxed), 0);
        // One unconditional DELETE: exempt, and the row counts once.
        let out = merge(
            &session,
            statement(
                dup,
                vec![clause(ClauseKind::Matched, None, ClauseAction::Delete)],
            ),
        );
        assert_eq!(affected(&out), 1);
        assert_eq!(files.ids(), [1, 3, 4, 5, 6, 7, 8, 9]);
        // Insert-only: no check, no file touched.
        merge(
            &session,
            statement(
                "SELECT CAST(id AS BIGINT) AS id, s FROM VALUES (2, 'a'), (2, 'b'), (11, 'c') AS v(id, s)",
                vec![clause(
                    ClauseKind::NotMatched,
                    None,
                    ClauseAction::InsertStar,
                )],
            ),
        );
        assert_eq!(files.ids(), [1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 11]);
        // The files the delete left stay; the inserts are a new file.
        assert_eq!(files.names(), ["f1", "f2", "f3", "f4"]);
    }

    #[test]
    fn a_failed_merge_rewrite_commits_nothing() {
        let files = Files::new();
        let (session, _) = session(&files);
        let e = try_merge(
            &session,
            statement(
                SOURCE,
                vec![clause(
                    ClauseKind::Matched,
                    None,
                    ClauseAction::Update(vec![set("id", "CAST(t.s AS BIGINT) + 0")]),
                )],
            ),
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
    fn a_table_without_a_row_level_table_keeps_the_refusal() {
        let files = Files::new();
        let (session, _) = session_with(&files, false, true);
        let none = try_merge(
            &session,
            statement(
                SOURCE,
                vec![clause(ClauseKind::Matched, None, ClauseAction::Delete)],
            ),
        )
        .unwrap();
        assert!(none.is_none());
    }

    #[test]
    fn clause_scopes_and_order_are_checked_before_anything_runs() {
        use krishiv_common::error::ErrorClass;
        let files = Files::new();
        let (session, _) = session(&files);
        let class = |clauses| {
            try_merge(&session, statement(SOURCE, clauses))
                .unwrap_err()
                .error_class()
        };
        assert_eq!(
            class(vec![
                clause(ClauseKind::Matched, None, ClauseAction::Delete),
                clause(ClauseKind::Matched, None, ClauseAction::Delete),
            ]),
            ErrorClass::NonLastMatchedClauseOmitCondition
        );
        // A `NOT MATCHED` clause sees only the source (a test table is no Delta table: the
        // analyzer's own error).
        assert_eq!(
            class(vec![clause(
                ClauseKind::NotMatched,
                None,
                ClauseAction::Insert(vec![set("id", "t.id")]),
            )]),
            ErrorClass::UnresolvedColumnWithSuggestion
        );
        assert_eq!(files.commits.load(Ordering::Relaxed), 0);
    }

    /// Without a table hook (`RowLevelTable::error`), a subquery or a non-deterministic
    /// condition is refused by Krishiv's own class; no table format's class appears.
    #[test]
    fn a_table_without_an_error_hook_gets_the_planners_own_refusals() {
        use krishiv_common::error::ErrorClass;
        let files = Files::new();
        let (session, _) = session(&files);
        let class = |condition: &str| {
            try_merge(
                &session,
                statement(
                    SOURCE,
                    vec![clause(
                        ClauseKind::Matched,
                        Some(condition),
                        ClauseAction::Delete,
                    )],
                ),
            )
            .unwrap_err()
            .error_class()
        };
        assert_eq!(
            class("src.id IN (SELECT 2)"),
            ErrorClass::KrishivUnsupportedFeature
        );
        assert_eq!(class("rand() < 2"), ErrorClass::KrishivUnsupportedFeature);
        assert_eq!(files.commits.load(Ordering::Relaxed), 0);
    }

    /// Under `spark.databricks.delta.schema.autoMerge.enabled` only a MERGE that would add a
    /// column is refused; `*` takes the target columns the source has.
    #[test]
    fn schema_evolution_refuses_only_a_merge_that_adds_a_column() {
        use krishiv_common::error::ErrorClass;
        let files = Files::new();
        let (session, _) = session(&files);
        block_on(session.sql("SET spark.databricks.delta.schema.autoMerge.enabled = true"))
            .unwrap();
        let wide = "SELECT CAST(2 AS BIGINT) AS id, 'w' AS s, 1 AS extra";
        let e = try_merge(
            &session,
            statement(
                wide,
                vec![clause(ClauseKind::Matched, None, ClauseAction::UpdateStar)],
            ),
        )
        .unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{e}"
        );
        // A narrow source: `INSERT *` gives the missing column NULL.
        merge(
            &session,
            statement(
                "SELECT CAST(20 AS BIGINT) AS id",
                vec![clause(
                    ClauseKind::NotMatched,
                    None,
                    ClauseAction::InsertStar,
                )],
            ),
        );
        assert_eq!(files.ids(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 20]);
        let s_of_20: Vec<bool> = files
            .lock()
            .iter()
            .flat_map(|(_, b)| {
                let ids = b.column(0).as_primitive::<Int64Type>();
                (0..b.num_rows())
                    .filter(|&i| ids.value(i) == 20)
                    .map(|i| b.column(1).is_null(i))
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(s_of_20, [true], "the inserted row's `s` is NULL");
        // Without the conf the narrow source is an unresolved name.
        block_on(session.sql("SET spark.databricks.delta.schema.autoMerge.enabled = false"))
            .unwrap();
        let e = try_merge(
            &session,
            statement(
                "SELECT CAST(21 AS BIGINT) AS id",
                vec![clause(
                    ClauseKind::NotMatched,
                    None,
                    ClauseAction::InsertStar,
                )],
            ),
        )
        .unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::UnresolvedColumnWithSuggestion,
            "{e}"
        );
    }

    /// The conflict read set is the `ON` condition's, unless the rewrite reads the target again:
    /// an assignment subquery or a source over `t` reads every row (#962 review, Codex P2).
    #[test]
    fn a_merge_that_reads_the_target_again_reads_every_row() {
        let files = Files::new();
        let (session, _) = session(&files);
        let other = Arc::new(Schema::new(vec![Field::new("x", ArrowType::Utf8, true)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&other),
            vec![Arc::new(arrow_array::StringArray::from(vec!["z"])) as ArrayRef],
        )
        .unwrap();
        session.register_batches("o", other, vec![batch]).unwrap();
        let read = |source: &str, value: &str| {
            merge(
                &session,
                statement(
                    source,
                    vec![clause(
                        ClauseKind::Matched,
                        None,
                        ClauseAction::Update(vec![set("s", value)]),
                    )],
                ),
            );
            files
                .reads
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .pop()
                .unwrap()
        };
        let kept = read(SOURCE, "(SELECT max(x) FROM o)");
        assert_eq!(kept.as_deref(), Some("(t.id == src.id)"));
        assert_eq!(read(SOURCE, "(SELECT max(s) FROM t)"), None);
        assert_eq!(read("SELECT id, s FROM t WHERE id = 2", "src.s"), None);
    }

    /// A session `krishiv.write.idempotencyKey` is refused before the MERGE runs, as on
    /// `UPDATE` and `DELETE` (#962 review, Codex P2).
    #[test]
    fn a_session_idempotency_key_is_refused_on_merge() {
        let files = Files::new();
        let (session, _) = session(&files);
        session
            .conf()
            .set("krishiv.write.idempotencyKey", "job-1")
            .unwrap();
        let e = try_merge(
            &session,
            statement(
                SOURCE,
                vec![clause(
                    ClauseKind::Matched,
                    None,
                    ClauseAction::Update(vec![set("s", "src.s")]),
                )],
            ),
        )
        .unwrap_err();
        assert!(e.message().contains("idempotencyKey"), "{e}");
        assert_eq!(files.commits.load(Ordering::Relaxed), 0);
    }

    /// The class of the refusal of `clauses` over the test table (no table classes of its own,
    /// so the session's own ones), with nothing committed.
    fn refused_class(clauses: Vec<MergeClause>) -> String {
        let files = Files::new();
        let (session, _) = session(&files);
        let e = try_merge(&session, statement(SOURCE, clauses)).unwrap_err();
        assert_eq!(files.commits.load(Ordering::Relaxed), 0, "{e}");
        format!("{:?}", e.error_class())
    }

    fn update(assignments: Vec<Assignment>) -> MergeClause {
        clause(ClauseKind::Matched, None, ClauseAction::Update(assignments))
    }

    #[test]
    fn the_analysis_refusals_are_the_session_s_own_classes() {
        // An `UPDATE` setting a column twice (Spark's `DUPLICATE_ASSIGNMENTS`).
        assert_eq!(
            refused_class(vec![update(vec![set("s", "src.s"), set("S", "'x'")])]),
            "DuplicateAssignments"
        );
        // An `INSERT` naming a column twice: the test table has no class for it.
        assert_eq!(
            refused_class(vec![clause(
                ClauseKind::NotMatched,
                None,
                ClauseAction::Insert(vec![set("id", "src.id"), set("id", "1")]),
            )]),
            "KrishivUnsupportedFeature"
        );
        // A nested field, `= DEFAULT`, and a column the table does not have.
        let nested = Assignment {
            nested: true,
            column: "s.x".to_owned(),
            value: Some("'x'".to_owned()),
        };
        assert_eq!(
            refused_class(vec![update(vec![nested])]),
            "KrishivUnsupportedFeature"
        );
        let default = Assignment {
            nested: false,
            column: "s".to_owned(),
            value: None,
        };
        assert_eq!(
            refused_class(vec![update(vec![default])]),
            "KrishivUnsupportedFeature"
        );
        assert_eq!(
            refused_class(vec![update(vec![set("nope", "src.s")])]),
            "UnresolvedColumnWithSuggestion"
        );
        // A value ANSI store assignment cannot put in the column (a string into `id BIGINT`).
        assert_eq!(
            refused_class(vec![update(vec![set("id", "src.s")])]),
            "IncompatibleDataForTableCannotSafelyCast"
        );
        // A clause of each `NOT MATCHED` kind left without a condition before another.
        assert_eq!(
            refused_class(vec![
                clause(ClauseKind::NotMatched, None, ClauseAction::InsertStar),
                clause(
                    ClauseKind::NotMatched,
                    Some("src.id > 0"),
                    ClauseAction::InsertStar
                ),
            ]),
            "NonLastNotMatchedByTargetClauseOmitCondition"
        );
        assert_eq!(
            refused_class(vec![
                clause(ClauseKind::NotMatchedBySource, None, ClauseAction::Delete),
                clause(
                    ClauseKind::NotMatchedBySource,
                    Some("t.id > 0"),
                    ClauseAction::Delete,
                ),
            ]),
            "NonLastNotMatchedBySourceClauseOmitCondition"
        );
    }

    #[test]
    fn a_case_sensitive_session_is_refused() {
        let files = Files::new();
        let (session, _) = session(&files);
        block_on(session.sql("SET spark.sql.caseSensitive = true")).unwrap();
        let e = try_merge(
            &session,
            statement(
                SOURCE,
                vec![clause(ClauseKind::Matched, None, ClauseAction::Delete)],
            ),
        )
        .unwrap_err();
        assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::KrishivUnsupportedFeature
        );
    }

    /// A pinned provider whose scan applies filters exactly (a `MemoryTable`): a filtered source
    /// over it is a bare scan with the filter pushed in, never the whole table.
    #[derive(Debug)]
    struct Pinned(MemoryTable);

    impl TableProvider for Pinned {
        fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
            self.0.scan()
        }

        fn pinned(&self) -> bool {
            true
        }
    }

    #[test]
    fn a_filtered_source_over_a_pinned_exact_filtering_table_merges_only_its_rows() {
        let files = Files::new();
        let (session, _) = session(&files);
        let src = Pinned(MemoryTable::new(schema(), vec![Batch::new(rows(&[2, 3]))]));
        session.state.catalog.put(
            "src_table",
            crate::catalog::TempTable::Provider {
                columns: vec![
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
                ],
                table: Arc::new(src),
            },
        );
        let update = || {
            vec![clause(
                ClauseKind::Matched,
                None,
                ClauseAction::Update(vec![set("s", "concat(src.s, '!')")]),
            )]
        };
        let out = merge(
            &session,
            statement("SELECT * FROM src_table WHERE id = 2", update()),
        );
        assert_eq!(affected(&out), 1);
        let changed: Vec<_> = strings(&files)
            .into_iter()
            .filter(|(_, s)| s.ends_with('!'))
            .collect();
        assert_eq!(changed, [(2, "s2!".to_owned())]);
        // The bare scan is read as the pinned provider: both rows.
        let out = merge(&session, statement("SELECT * FROM src_table", update()));
        assert_eq!(affected(&out), 2);
    }
}
