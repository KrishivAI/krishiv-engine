// SPDX-License-Identifier: Apache-2.0
//! Hash aggregation benchmarks, single-thread and deterministic (CodSpeed
//! simulation mode):
//!
//! - `aggregate/grouped_sum_{64,262144}_keys`: `select k, sum(v), count(*)
//!   group by k` over 1 Mi rows (`Int64` keys and values), low and high
//!   cardinality: probe-bound vs insert-bound.
//! - `aggregate/grouped_sum_string_keys`: the same over a 12-byte `Utf8` key,
//!   `grouped_sum_two_keys` over two `Int64` keys (about 262,144 groups) and
//!   `grouped_multi_agg` computing `sum`, `min`, `max`, `avg` and `count(*)`.
//! - `aggregate/grouped_sum_spill`: the high-cardinality query under a
//!   4 MiB budget with a spill directory: spill runs, then the partitioned
//!   merge from disk.
//! - `aggregate/decimal_{sum,avg}_{1,64,262144}_keys`: `sum`/`avg` of a
//!   `decimal(15,2)` column (TPC-H price-like values) over 1 Mi rows: the
//!   per-batch overflow proof path of the decimal accumulators (P2-T160)
//!   with one key (accumulator-bound), few and many keys.
//! - `…_rows`: the same queries on the row-layout tables
//!   (`krishiv.exec.agg.rowLayout`, P2-T159).
//!
//! `AGG_BENCH_THREADS=N` runs them (but the spill one) on `N` worker
//! threads in the shipped (non-deterministic) mode instead: thread-local
//! tables that abandon, and the parallel final phase.

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

use arrow_array::{ArrayRef, Decimal128Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::StreamExt;
use krishiv_exec::aggregate::{
    AggregateExpr, AggregateFunction, AggregateNode, AggregateOutput, AggregateSpec, HashAggregate,
};
use krishiv_exec::testing::{Batches, Scan};
use krishiv_exec::{
    DEFAULT_BATCH_SIZE, ExecConfig, ExecNode, ExecutorConfig, QueryContext, QueryId, QueryRunner,
    TokioExecutor, WorkloadClass,
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
fn decimal_plan(batches: &[Batch], f: &AggregateFunction) -> Arc<dyn ExecNode> {
    let s = schema_of(DECIMAL);
    let spec = AggregateSpec {
        input: Arc::clone(&s),
        group_by: vec![0],
        aggregates: vec![AggregateExpr::new(f.clone(), 1, DECIMAL, "a")],
        output: AggregateOutput::Final,
        ansi: true,
        spill: None,
        require_unit_weights: false,
        legacy_statistical: false,
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

/// `(k1, k2, s, v)`: `k1` in `0..512`, `k2` in `0..512` (together about
/// 262,144 groups), `s` a 12-byte string of the pair, `v = i`.
fn wide_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k1", DataType::Int64, false),
        Field::new("k2", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
        Field::new("v", DataType::Int64, false),
    ]))
}

fn wide_input() -> Vec<Batch> {
    (0..ROWS)
        .step_by(DEFAULT_BATCH_SIZE)
        .map(|start| {
            let r = start as u64..(start + DEFAULT_BATCH_SIZE).min(ROWS) as u64;
            let pair = |i: u64| {
                let x = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11;
                (x % 512, (x >> 9) % 512)
            };
            let k1 = Int64Array::from_iter_values(r.clone().map(|i| pair(i).0 as i64));
            let k2 = Int64Array::from_iter_values(r.clone().map(|i| pair(i).1 as i64));
            let s = StringArray::from_iter_values(r.clone().map(|i| {
                let (a, b) = pair(i);
                format!("key-{a:03}-{b:03}")
            }));
            let v = Int64Array::from_iter_values(r.map(|i| i as i64));
            let cols: Vec<ArrayRef> = vec![Arc::new(k1), Arc::new(k2), Arc::new(s), Arc::new(v)];
            Batch::new(RecordBatch::try_new(wide_schema(), cols).expect("batch"))
        })
        .collect()
}

/// `group by <group_by>` computing `aggregates` over `wide_input` batches.
fn wide_plan(
    batches: &[Batch],
    group_by: Vec<usize>,
    aggregates: Vec<AggregateExpr>,
) -> Arc<dyn ExecNode> {
    let s = wide_schema();
    let spec = AggregateSpec {
        input: Arc::clone(&s),
        group_by,
        aggregates,
        output: AggregateOutput::Final,
        ansi: true,
        spill: None,
        require_unit_weights: false,
        legacy_statistical: false,
    };
    let scan = Arc::new(Scan(Batches::new(s, batches.to_vec())));
    AggregateNode::new(scan, HashAggregate::new(spec).expect("spec"))
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
        legacy_statistical: false,
    };
    let scan = Arc::new(Scan(Batches::new(s, batches.to_vec())));
    AggregateNode::new(scan, HashAggregate::new(spec).expect("spec"))
}

/// How the benchmark runs: the deterministic single-thread executor, or
/// `AGG_BENCH_THREADS` workers in the shipped mode.
#[derive(Clone, Copy)]
struct Mode {
    /// The row-layout aggregate tables.
    rows: bool,
    /// Deterministic (seeded) mode.
    seeded: bool,
}

/// Runs `plan` under a per-query `limit`; returns the output rows.
fn run(
    runner: &QueryRunner,
    caller: &tokio::runtime::Runtime,
    plan: Arc<dyn ExecNode>,
    limit: usize,
    mode: Mode,
) -> usize {
    let root = TreePool::new(&MemoryConfig::default());
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig {
            seed: mode.seeded.then_some(1),
            agg_row_layout: mode.rows,
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
    let threads: Option<usize> = std::env::var("AGG_BENCH_THREADS")
        .ok()
        .map(|t| t.parse().expect("AGG_BENCH_THREADS: a thread count"))
        .filter(|&t| t > 0);
    let runner = {
        let _in_runtime = caller.enter();
        let executor = match threads {
            Some(t) => TokioExecutor::new(&ExecutorConfig {
                compute_threads: t,
                blocking_threads: t,
                io_blocking_threads: 2,
                ..ExecutorConfig::default()
            }),
            None => TokioExecutor::deterministic(1),
        };
        QueryRunner::new(executor.expect("executor"))
    };
    let seeded = threads.is_none();
    let modes =
        [("", false), ("_rows", true)].map(|(suffix, rows)| (suffix, Mode { rows, seeded }));
    let mut g = c.benchmark_group("aggregate");
    g.throughput(Throughput::Elements(ROWS as u64));
    for keys in [64u64, 1 << 18] {
        let batches = input(keys);
        for (suffix, mode) in modes {
            g.bench_function(format!("grouped_sum_{keys}_keys{suffix}"), |b| {
                b.iter(|| black_box(run(&runner, &caller, plan(&batches, None), 1 << 30, mode)));
            });
        }
    }
    let wide = wide_input();
    let sum = || AggregateExpr::new(AggregateFunction::Sum, 3, DataType::Int64, "s");
    let multi = || {
        vec![
            sum(),
            AggregateExpr::new(AggregateFunction::Min, 3, DataType::Int64, "lo"),
            AggregateExpr::new(AggregateFunction::Max, 3, DataType::Int64, "hi"),
            AggregateExpr::new(AggregateFunction::Avg, 3, DataType::Int64, "a"),
            AggregateExpr::count_star("c"),
        ]
    };
    type Make<'a> = Box<dyn Fn() -> Arc<dyn ExecNode> + 'a>;
    let wide_cases: [(&str, Make<'_>); 3] = [
        (
            "grouped_sum_string_keys",
            Box::new(|| wide_plan(&wide, vec![2], vec![sum()])),
        ),
        (
            "grouped_sum_two_keys",
            Box::new(|| wide_plan(&wide, vec![0, 1], vec![sum()])),
        ),
        (
            "grouped_multi_agg",
            Box::new(|| wide_plan(&wide, vec![0, 1], multi())),
        ),
    ];
    for (name, make) in &wide_cases {
        for (suffix, mode) in modes {
            g.bench_function(format!("{name}{suffix}"), |b| {
                b.iter(|| black_box(run(&runner, &caller, make(), 1 << 30, mode)));
            });
        }
    }
    for keys in [1u64, 64, 1 << 18] {
        let batches = decimal_input(keys);
        for (name, f) in [
            ("sum", AggregateFunction::Sum),
            ("avg", AggregateFunction::Avg),
        ] {
            for (suffix, mode) in modes {
                g.bench_function(format!("decimal_{name}_{keys}_keys{suffix}"), |b| {
                    b.iter(|| {
                        let p = decimal_plan(&batches, &f);
                        black_box(run(&runner, &caller, p, 1 << 30, mode))
                    });
                });
            }
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
    // 4 MiB is too little for many drivers' tables on either layout: the
    // spill benchmark runs single-threaded only.
    for (suffix, mode) in modes.into_iter().filter(|_| threads.is_none()) {
        g.bench_function(format!("grouped_sum_spill{suffix}"), |b| {
            b.iter(|| {
                let scope = spill.scope().expect("scope");
                black_box(run(
                    &runner,
                    &caller,
                    plan(&batches, Some(scope)),
                    4 << 20,
                    mode,
                ))
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
