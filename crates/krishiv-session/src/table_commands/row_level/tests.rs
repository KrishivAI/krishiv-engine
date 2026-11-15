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
struct Table(Arc<Files>);

impl TableProvider for Table {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        self.0.table(None, false).scan()
    }

    fn row_level(&self, _op: RowLevelOperation) -> Option<Arc<dyn RowLevelTable>> {
        Some(Arc::new(RowLevel(Arc::clone(&self.0))))
    }
}

#[derive(Debug)]
struct RowLevel(Arc<Files>);

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

/// A session reading `files` as the hidden table it returns the name of.
fn session(files: &Arc<Files>) -> (Session, String) {
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
            table: Arc::new(Table(Arc::clone(files))),
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
    let s = session.clone();
    let quoted = format!("`{name}`");
    let condition = condition.map(str::to_owned);
    let df = block_on(session.on_executor(async move {
        s.run_row_level(op, &quoted, "t", condition.as_deref(), &assignments)
            .await
    }))
    .unwrap()
    .unwrap();
    block_on(df.collect()).unwrap()
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
    let s = session.clone();
    let quoted = format!("`{name}`");
    let result = block_on(session.on_executor(async move {
        s.run_row_level(
            RowLevelOperation::Update,
            &quoted,
            "t",
            Some("id = 4"),
            &[Assignment {
                column: "id".to_owned(),
                value: Some("CAST(s AS BIGINT)".to_owned()),
            }],
        )
        .await
    }));
    assert!(result.is_err());
    assert_eq!(files.commits.load(Ordering::Relaxed), 0);
    assert_eq!(files.ids(), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
