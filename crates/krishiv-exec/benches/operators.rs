// SPDX-License-Identifier: Apache-2.0
//! Throughput of the stateless operators over a prebuilt in-memory table, on
//! one deterministic worker (simulation-friendly: no thread parking in the
//! instruction counts).
//!
//! - `operators/filter_1m_rows/{50_percent,all_rows}`: scan → filter → count.
//!   The predicate is a stand-in closure; the cost measured is the mask
//!   handling, Arrow's filter kernel and the driver. `all_rows` keeps every
//!   row, so the filter shares buffers instead of copying: its extra cost over
//!   `project_1m_rows/columns` is the mask itself.
//! - `operators/project_1m_rows/{columns,expr}`: scan → project → count.
//!   `columns` reorders two columns (zero-copy); `expr` adds a computed one.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::hint::black_box;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, BooleanArray, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::StreamExt;
use krishiv_exec::testing::{Break, Count, Pipe, Scan};
use krishiv_exec::{
    ExecConfig, ExecNode, Filter, FnPredicate, MemoryScan, Operator, Project, ProjectItem,
    QueryContext, QueryId, QueryRunner, Result, ScalarExpr, Scheduler, Source, TokioExecutor,
    WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
use krishiv_types::batch::Batch;

const BATCHES: usize = 256;
const BATCH_ROWS: usize = 4096;
const ROWS: u64 = (BATCHES * BATCH_ROWS) as u64;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("v", DataType::Int64, false),
        Field::new("w", DataType::Int64, false),
    ]))
}

#[expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]
fn table() -> Vec<Batch> {
    (0..BATCHES)
        .map(|k| {
            let v =
                Int64Array::from_iter_values((0..BATCH_ROWS).map(|i| (k * BATCH_ROWS + i) as i64));
            let w = Int64Array::from_iter_values((0..BATCH_ROWS).map(|i| i as i64));
            Batch::new(
                RecordBatch::try_new(schema(), vec![Arc::new(v), Arc::new(w)]).expect("batch"),
            )
        })
        .collect()
}

#[derive(Debug)]
struct Double;

impl ScalarExpr for Double {
    fn data_type(&self) -> DataType {
        DataType::Int64
    }

    fn nullable(&self) -> bool {
        false
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        let v = batch.column(0).as_primitive::<Int64Type>();
        Ok(Arc::new(Int64Array::from_iter_values(
            v.values().iter().map(|x| x.wrapping_mul(2)),
        )))
    }
}

#[expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]
fn plan<O: Operator + 'static>(table: &[Batch], op: &Arc<O>) -> Arc<dyn ExecNode> {
    let scan: Arc<dyn Source> =
        Arc::new(MemoryScan::try_new(schema(), table.to_vec()).expect("scan"));
    Arc::new(Break(
        Arc::new(Pipe(
            Arc::new(Scan(scan)),
            Arc::clone(op) as Arc<dyn Operator>,
        )),
        Count::new(),
        false,
    ))
}

#[expect(
    clippy::expect_used,
    reason = "a failing query must fail the bench loudly, not skip it"
)]
fn run(runner: &QueryRunner, caller: &tokio::runtime::Runtime, plan: Arc<dyn ExecNode>) -> usize {
    let root = TreePool::new(&MemoryConfig::default());
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("bench", None),
    );
    caller.block_on(async {
        let mut stream = runner.submit(plan, Arc::clone(&query)).await.expect("plan");
        let mut rows = 0;
        while let Some(b) = stream.next().await {
            rows += b.expect("batch").num_rows();
        }
        query.finished().await;
        rows
    })
}

#[expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]
fn bench(c: &mut Criterion) {
    let caller = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("caller runtime");
    let executor = {
        let _in_runtime = caller.enter();
        TokioExecutor::deterministic(1).expect("executor")
    };
    let runner = QueryRunner::new(executor);
    let table = table();

    let half = |b: &RecordBatch| -> Result<BooleanArray> {
        Ok(b.column(0)
            .as_primitive::<Int64Type>()
            .values()
            .iter()
            .map(|v| v & 1 == 0)
            .collect())
    };
    let all = |b: &RecordBatch| -> Result<BooleanArray> {
        Ok(BooleanArray::from(vec![true; b.num_rows()]))
    };
    let mut g = c.benchmark_group("operators");
    g.throughput(Throughput::Elements(ROWS));
    g.bench_function("filter_1m_rows/50_percent", |b| {
        let f = Filter::new(schema(), Arc::new(FnPredicate(half)));
        b.iter(|| black_box(run(&runner, &caller, plan(&table, &f))));
    });
    g.bench_function("filter_1m_rows/all_rows", |b| {
        let f = Filter::new(schema(), Arc::new(FnPredicate(all)));
        b.iter(|| black_box(run(&runner, &caller, plan(&table, &f))));
    });
    g.bench_function("project_1m_rows/columns", |b| {
        let p = Project::try_new(
            &schema(),
            vec![ProjectItem::column(1), ProjectItem::column(0)],
        )
        .expect("project");
        b.iter(|| black_box(run(&runner, &caller, plan(&table, &p))));
    });
    g.bench_function("project_1m_rows/expr", |b| {
        let p = Project::try_new(
            &schema(),
            vec![
                ProjectItem::column(0),
                ProjectItem::expr("d", Arc::new(Double)),
            ],
        )
        .expect("project");
        b.iter(|| black_box(run(&runner, &caller, plan(&table, &p))));
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
