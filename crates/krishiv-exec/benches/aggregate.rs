// SPDX-License-Identifier: Apache-2.0
//! Hash aggregation benchmarks, single-thread and deterministic (CodSpeed
//! simulation mode):
//!
//! - `aggregate/grouped_sum_{64,262144}_keys`: `select k, sum(v), count(*)
//!   group by k` over 1 Mi rows (`Int64` keys and values), low and high
//!   cardinality: probe-bound vs insert-bound.
//! - `aggregate/grouped_sum_spill`: the high-cardinality query under a
//!   4 MiB budget with a spill directory: spill runs, then the partitioned
//!   merge from disk.
//! - `aggregate/decimal_{sum,avg}_{1,64,262144}_keys`: `sum`/`avg` of a
//!   `decimal(15,2)` column (TPC-H price-like values) over 1 Mi rows: the
//!   per-batch overflow proof path of the decimal accumulators (P2-T160)
//!   with one key (accumulator-bound), few and many keys.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]
#![expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]

use std::hint::black_box;
use std::sync::Arc;

use arrow_array::{ArrayRef, Decimal128Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::StreamExt;
use krishiv_exec::aggregate::{
    AggregateExpr, AggregateFunction, AggregateNode, AggregateOutput, AggregateSpec, HashAggregate,
};
use krishiv_exec::testing::{Batches, Scan};
use krishiv_exec::{
    DEFAULT_BATCH_SIZE, ExecConfig, ExecNode, QueryContext, QueryId, QueryRunner, TokioExecutor,
    WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, SpillConfig, SpillManager, SpillScope, TreePool};
use krishiv_types::batch::Batch;

const ROWS: usize = 1 << 20;

fn schema() -> SchemaRef {
    schema_of(DataType::Int64)
}

fn schema_of(v: DataType) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", v, false),
    ]))
}

/// `decimal(15,2)`, TPC-H's money type.
const DECIMAL: DataType = DataType::Decimal128(15, 2);

/// `ROWS` rows, `k` a scrambled key in `0..keys`, `v` a `decimal(15,2)` in
/// `[900.00, 105000.00)` (the range of `l_extendedprice / l_quantity`).
fn decimal_input(keys: u64) -> Vec<Batch> {
    input(keys)
        .into_iter()
        .map(|b| {
            let r = b.data();
            let v = r
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("values")
                .iter()
                .map(|v| v.map(|v| 90_000 + i128::from(v.wrapping_mul(7919) % 10_410_000)));
            let v = Decimal128Array::from_iter(v).with_data_type(DECIMAL);
            let cols: Vec<ArrayRef> = vec![Arc::clone(r.column(0)), Arc::new(v)];
            Batch::new(RecordBatch::try_new(schema_of(DECIMAL), cols).expect("batch"))
        })
        .collect()
}

/// `select k, f(v) group by k` over `batches` (`v` a decimal).
fn decimal_plan(batches: &[Batch], f: AggregateFunction) -> Arc<dyn ExecNode> {
    let s = schema_of(DECIMAL);
    let spec = AggregateSpec {
        input: Arc::clone(&s),
        group_by: vec![0],
        aggregates: vec![AggregateExpr::new(f, 1, DECIMAL, "a")],
        output: AggregateOutput::Final,
        ansi: true,
        spill: None,
        require_unit_weights: false,
    };
    let scan = Arc::new(Scan(Batches::new(s, batches.to_vec())));
    AggregateNode::new(scan, HashAggregate::new(spec).expect("spec"))
}

/// `ROWS` rows, `k` a scrambled key in `0..keys`, `v = i`.
fn input(keys: u64) -> Vec<Batch> {
    (0..ROWS)
        .step_by(DEFAULT_BATCH_SIZE)
        .map(|start| {
            let r = start as u64..(start + DEFAULT_BATCH_SIZE).min(ROWS) as u64;
            let k = r.clone().map(|i| {
                let x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11;
                i64::try_from(x % keys).expect("key")
            });
            let v = r.map(|i| i64::try_from(i).expect("value"));
            let cols: Vec<ArrayRef> = vec![
                Arc::new(Int64Array::from_iter_values(k)),
                Arc::new(Int64Array::from_iter_values(v)),
            ];
            Batch::new(RecordBatch::try_new(schema(), cols).expect("batch"))
        })
        .collect()
}

fn plan(batches: &[Batch], spill: Option<SpillScope>) -> Arc<dyn ExecNode> {
    let s = schema();
    let spec = AggregateSpec {
        input: Arc::clone(&s),
        group_by: vec![0],
        aggregates: vec![
            AggregateExpr::new(AggregateFunction::Sum, 1, DataType::Int64, "s"),
            AggregateExpr::count_star("c"),
        ],
        output: AggregateOutput::Final,
        ansi: true,
        spill,
        require_unit_weights: false,
    };
    let scan = Arc::new(Scan(Batches::new(s, batches.to_vec())));
    AggregateNode::new(scan, HashAggregate::new(spec).expect("spec"))
}

/// Runs `plan` under a per-query `limit`; returns the output rows.
fn run(
    runner: &QueryRunner,
    caller: &tokio::runtime::Runtime,
    plan: Arc<dyn ExecNode>,
    limit: usize,
) -> usize {
    let root = TreePool::new(&MemoryConfig::default());
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig {
            seed: Some(1),
            ..ExecConfig::default()
        },
        root.child("bench", Some(limit)),
    );
    caller.block_on(async {
        let mut stream = runner
            .execute(plan, Arc::clone(&query))
            .await
            .expect("plan");
        let mut rows = 0;
        while let Some(b) = stream.next().await {
            rows += b.expect("batch").num_rows();
        }
        query.finished().await;
        rows
    })
}

fn bench(c: &mut Criterion) {
    let caller = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("caller runtime");
    let runner = {
        let _in_runtime = caller.enter();
        QueryRunner::new(TokioExecutor::deterministic(1).expect("executor"))
    };
    let mut g = c.benchmark_group("aggregate");
    g.throughput(Throughput::Elements(ROWS as u64));
    for keys in [64u64, 1 << 18] {
        let batches = input(keys);
        g.bench_function(format!("grouped_sum_{keys}_keys"), |b| {
            b.iter(|| black_box(run(&runner, &caller, plan(&batches, None), 1 << 30)));
        });
    }
    for keys in [1u64, 64, 1 << 18] {
        let batches = decimal_input(keys);
        for (name, f) in [
            ("sum", AggregateFunction::Sum),
            ("avg", AggregateFunction::Avg),
        ] {
            g.bench_function(format!("decimal_{name}_{keys}_keys"), |b| {
                b.iter(|| black_box(run(&runner, &caller, decimal_plan(&batches, f), 1 << 30)));
            });
        }
    }
    let batches = input(1 << 18);
    // Removed after the manager when the bench returns.
    let dir = tempfile::Builder::new()
        .prefix("krishiv-bench-agg-")
        .tempdir()
        .expect("temporary directory");
    let spill = SpillManager::open(&SpillConfig {
        // A path inside the temporary directory: the manager creates it
        // private (0700), whatever the umask.
        dir: dir.path().join("spill"),
        ..SpillConfig::default()
    })
    .expect("spill directory");
    g.bench_function("grouped_sum_spill", |b| {
        b.iter(|| {
            let scope = spill.scope().expect("scope");
            black_box(run(&runner, &caller, plan(&batches, Some(scope)), 4 << 20))
        });
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
