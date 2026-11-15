// SPDX-License-Identifier: Apache-2.0
//! `PhysNode::TableWrite` (design `P3-PR-DML-row-level.md` §7): a query's rows written into a
//! table provider's sink, resolved when the plan is instantiated; the sink finalizes (commits)
//! once, and the plan returns no rows.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::RecordBatch;
use futures::future::BoxFuture;
use krishiv_common::error::ErrorClass;
use krishiv_exec::physical::{PhysNode, TableWriteRequest};
use krishiv_exec::planner::{
    CardinalityRule, MemoryTable, RowLevelCapabilities, RowLevelOperation, RowLevelTable,
    RowLevelWrite, TableCapabilities, TableProvider, TableProviderResolver, WriteMode,
    WriteRequest,
};
use krishiv_exec::{GlobalState, LocalState, ScanPushdown, Sink, SinkStatus, TaskContext};
use krishiv_types::batch::Batch;

use super::{Runnable, env, run, table};

/// What a sink received, and how many times it finalized.
#[derive(Debug, Default)]
struct Received {
    batches: Mutex<Vec<RecordBatch>>,
    finalized: AtomicUsize,
}

impl Received {
    fn rows(&self) -> usize {
        self.batches
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .map(RecordBatch::num_rows)
            .sum()
    }
}

#[derive(Debug)]
struct Collect(Arc<Received>);

impl Sink for Collect {
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
        self.0
            .batches
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
        self.0.finalized.fetch_add(1, Ordering::Relaxed);
        Box::pin(std::future::ready(Ok(())))
    }
}

/// The table `t`, writable, with a row-level table for `DELETE` only.
#[derive(Debug)]
struct Writable {
    table: MemoryTable,
    received: Arc<Received>,
}

impl TableProvider for Writable {
    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        self.table.scan()
    }

    fn capabilities(&self) -> TableCapabilities {
        let mut c = TableCapabilities::default();
        c.batch_write = true;
        c
    }

    fn writer(&self, request: &WriteRequest) -> krishiv_exec::Result<Arc<dyn Sink>> {
        self.capabilities()
            .check(&request.mode, &request.table_name)?;
        Ok(Arc::new(Collect(Arc::clone(&self.received))))
    }

    fn row_level(&self, op: RowLevelOperation) -> Option<Arc<dyn RowLevelTable>> {
        (op == RowLevelOperation::Delete).then(|| {
            Arc::new(Rewritable {
                table: self.table.clone(),
                received: Arc::clone(&self.received),
            }) as Arc<dyn RowLevelTable>
        })
    }
}

#[derive(Debug)]
struct Rewritable {
    table: MemoryTable,
    received: Arc<Received>,
}

impl RowLevelTable for Rewritable {
    fn capabilities(&self) -> RowLevelCapabilities {
        RowLevelCapabilities::copy_on_write(CardinalityRule::SparkMergeCardinality)
    }

    fn scan(&self) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        self.table.scan()
    }

    fn scan_files(&self, _files: &[String]) -> krishiv_exec::Result<Arc<dyn ScanPushdown>> {
        self.table.scan()
    }

    fn rewriter(&self, _request: &RowLevelWrite) -> krishiv_exec::Result<Arc<dyn Sink>> {
        Ok(Arc::new(Collect(Arc::clone(&self.received))))
    }
}

#[derive(Debug)]
struct One(Arc<dyn TableProvider>);

impl TableProviderResolver for One {
    fn provider(&self, _: &krishiv_plan::Scan) -> krishiv_exec::Result<Arc<dyn TableProvider>> {
        Ok(Arc::clone(&self.0))
    }
}

/// `SELECT * FROM t` (10 rows) written as `request(schema)` asks, into a fresh [`Writable`].
fn written(
    request: impl FnOnce(arrow_schema::SchemaRef) -> TableWriteRequest,
) -> (Runnable, Arc<Received>) {
    let env = env();
    let plan = env.plan("SELECT * FROM t").unwrap();
    let scan = find_scan(plan.root()).unwrap();
    let schema = Arc::clone(plan.schema());
    let (_, batches) = table(10);
    let received = Arc::new(Received::default());
    let provider = Arc::new(Writable {
        table: MemoryTable::new(Arc::clone(&schema), batches),
        received: Arc::clone(&received),
    });
    let plan = plan.plan.into_table_write(scan, request(schema));
    (Runnable::new(plan, Arc::new(One(provider))), received)
}

fn find_scan(n: &PhysNode) -> Option<krishiv_plan::Scan> {
    match n {
        PhysNode::Scan(s) => Some(s.scan.clone()),
        _ => n.children().into_iter().find_map(find_scan),
    }
}

#[test]
fn a_write_sends_every_row_to_the_providers_sink_and_finalizes_once() {
    let (plan, received) = written(|schema| {
        TableWriteRequest::Write(WriteRequest::new(WriteMode::Append, "`t`", schema))
    });
    assert!(matches!(plan.root(), PhysNode::TableWrite { .. }), "{plan}");
    assert!(plan.schema().fields().is_empty());
    assert!(plan.to_string().contains("TableWrite"), "{plan}");
    let out = run(&plan).unwrap();
    assert_eq!(out.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
    assert_eq!(received.rows(), 10);
    assert_eq!(received.finalized.load(Ordering::Relaxed), 1);
}

#[test]
fn a_row_level_write_goes_to_the_rewriter() {
    let (plan, received) = written(|schema| {
        TableWriteRequest::RowLevel(RowLevelWrite::new(
            RowLevelOperation::Delete,
            "`t`",
            schema,
            Vec::new(),
            None,
        ))
    });
    run(&plan).unwrap();
    assert_eq!(received.rows(), 10);
    assert_eq!(received.finalized.load(Ordering::Relaxed), 1);
}

#[test]
fn a_table_without_the_mode_or_the_statement_refuses_before_writing() {
    let (plan, received) = written(|schema| {
        TableWriteRequest::Write(WriteRequest::new(WriteMode::Truncate, "`t`", schema))
    });
    let e = run(&plan).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::UnsupportedFeatureTableOperation,
        "{e}"
    );
    let (plan, _) = written(|schema| {
        TableWriteRequest::RowLevel(RowLevelWrite::new(
            RowLevelOperation::Update,
            "`t`",
            schema,
            Vec::new(),
            None,
        ))
    });
    let e = run(&plan).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::UnsupportedFeatureTableOperation,
        "{e}"
    );
    assert!(e.to_string().contains("UPDATE"), "{e}");
    assert_eq!(received.rows(), 0);
}

#[test]
fn a_request_whose_columns_differ_from_the_input_is_refused() {
    let (plan, received) = written(|_| {
        let other = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "x",
            arrow_schema::DataType::Utf8,
            true,
        )]));
        TableWriteRequest::Write(WriteRequest::new(WriteMode::Append, "`t`", other))
    });
    let e = run(&plan).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError, "{e}");
    assert_eq!(received.rows(), 0);
}
