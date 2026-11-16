// SPDX-License-Identifier: Apache-2.0
//! The row-level planner over a test provider (design `P3-PR-DML-row-level.md` §5): files as
//! batches, a scan with the row id, and a commit that swaps the touched files for one new file.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, BinaryViewArray, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use futures::executor::block_on;
use futures::future::BoxFuture;
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

/// The table's files: name and rows (`id BIGINT, s STRING`).
#[derive(Debug)]
struct Files {
    files: Mutex<Vec<(Vec<u8>, RecordBatch)>>,
    commits: AtomicUsize,
    next: AtomicUsize,
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
        RowLevelCapabilities::copy_on_write(CardinalityRule::DeltaWithUnconditionalDeleteExemption)
    }

    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        self.0.table(None, true).scan()
    }

    fn scan_files(&self, files: &[Vec<u8>]) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        self.0.table(Some(files), true).scan()
    }

    fn rewriter(&self, request: &RowLevelWrite) -> krishiv_exec::Result<Arc<dyn Sink>> {
        Ok(Arc::new(Swap {
            files: Arc::clone(&self.0),
            touched: request.touched.clone(),
            written: Mutex::new(Vec::new()),
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
        block_on(session.on_executor(async move { s.run_merge(&m).await }))
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
        assert!(files.ids().contains(&20));
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
}
