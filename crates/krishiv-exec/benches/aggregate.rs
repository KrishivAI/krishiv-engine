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

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::StreamExt;
use krishiv_exec::aggregate::{
    AggregateExpr, AggregateFunction, AggregateNode, AggregateOutput, AggregateSpec, HashAggregate,
};
use krishiv_exec::testing::{Batches, Scan};
use krishiv_exec::{
    DEFAULT_BATCH_SIZE, ExecConfig, ExecNode, QueryContext, QueryId, QueryRunner, Scheduler,
    TokioExecutor, WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, SpillConfig, SpillManager, SpillScope, TreePool};
use krishiv_types::batch::Batch;

const ROWS: usize = 1 << 20;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]))
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
        let mut stream = runner.submit(plan, Arc::clone(&query)).await.expect("plan");
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
