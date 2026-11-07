// SPDX-License-Identifier: Apache-2.0
//! Hash aggregation benchmarks, single-thread and deterministic (CodSpeed
//! simulation mode):
//!
//! - `aggregate/grouped_sum_{64,262144}_keys`: `select k, sum(v), count(*)
//!   group by k` over 1 Mi rows (`Int64` keys and values), low and high
//!   cardinality: probe-bound vs insert-bound.
//! - `aggregate/grouped_sum_string_keys`: the same over a 12-byte `Utf8View` key,
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
//! - `aggregate/{count,udaf_count}_{64,262144}_keys`: `count(v)` against a
//!   native Rust aggregate UDF computing the same (ADR 0045): the cost of the
//!   UDF boundary (guarded calls, the engine's null and filter handling, the
//!   state checks). The ADR's target is within 1.5x of `count`.
//!
//! - `aggregate/q01_shape_rows`: TPC-H q01's aggregate (P2b E2): two
//!   one-character string keys (`l_returnflag`, `l_linestatus`: 4 groups),
//!   four decimal sums, three averages and `count(*)` over 1 Mi rows of
//!   `decimal(15,2)` / `decimal(32,4)` / `decimal(38,6)` columns, as the
//!   query's projection hands them to the aggregate. `AGG_BENCH_Q01_REPS=N`
//!   runs it `N` times outside criterion and exits (for `perf stat`:
//!   instructions per row = instructions / (N * 2^20)); `AGG_BENCH_REPS_CASE=few|many`
//!   runs `grouped_sum_{64,262144}_keys_rows` instead.
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

use arrow_array::{ArrayRef, Decimal128Array, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::StreamExt;
use krishiv_exec::aggregate::{
    AggInput, AggregateExpr, AggregateFunction, AggregateNode, AggregateOutput, AggregateSpec,
    HashAggregate, UdafCall, UdafEnv,
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
        udafs: None,
    };
    let scan = Arc::new(Scan(Batches::new(s, batches.to_vec())));
    AggregateNode::new(scan, HashAggregate::new(spec).expect("spec"))
}

/// `ucount(v)`: the non-NULL rows of a group, as `count(v)`.
#[derive(Debug)]
struct UCount(krishiv_udf_sdk::Signature);

impl krishiv_udf_sdk::AggregateUdf for UCount {
    fn name(&self) -> &str {
        "ucount"
    }

    fn version(&self) -> u32 {
        1
    }

    fn signature(&self) -> &krishiv_udf_sdk::Signature {
        &self.0
    }

    fn return_field(
        &self,
        _: &[krishiv_udf_sdk::ArgField],
    ) -> Result<krishiv_udf_sdk::ReturnField, krishiv_udf_sdk::UdfError> {
        Ok(krishiv_udf_sdk::ReturnField {
            data_type: krishiv_types::DataType::Long,
            nullable: false,
        })
    }

    fn state_fields(
        &self,
        _: &[krishiv_udf_sdk::ArgField],
    ) -> Result<Vec<krishiv_udf_sdk::StateField>, krishiv_udf_sdk::UdfError> {
        Ok(vec![krishiv_udf_sdk::StateField {
            name: "count".to_owned(),
            data_type: krishiv_types::DataType::Long,
            nullable: false,
        }])
    }

    fn accumulator(
        &self,
        _: &[krishiv_udf_sdk::ArgField],
        _: &krishiv_udf_sdk::InvokeContext,
    ) -> Result<Box<dyn krishiv_udf_sdk::GroupsAccumulator>, krishiv_udf_sdk::UdfError> {
        Ok(Box::new(Counts(Vec::new())))
    }
}

#[derive(Debug)]
struct Counts(Vec<i64>);

impl krishiv_udf_sdk::GroupsAccumulator for Counts {
    fn resize(&mut self, groups: usize) {
        self.0.resize(groups, 0);
    }

    fn update(
        &mut self,
        _: &[krishiv_udf_sdk::UdfArg<'_>],
        groups: &[u32],
    ) -> Result<(), krishiv_udf_sdk::UdfError> {
        for &g in groups {
            self.0[g as usize] += 1;
        }
        Ok(())
    }

    fn merge(
        &mut self,
        states: &[ArrayRef],
        groups: &[u32],
    ) -> Result<(), krishiv_udf_sdk::UdfError> {
        let c = states[0]
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("counts");
        for (i, &g) in groups.iter().enumerate() {
            self.0[g as usize] += c.value(i);
        }
        Ok(())
    }

    fn state(&self, groups: &[u32]) -> Result<Vec<ArrayRef>, krishiv_udf_sdk::UdfError> {
        Ok(vec![self.evaluate(groups)?])
    }

    fn evaluate(&self, groups: &[u32]) -> Result<ArrayRef, krishiv_udf_sdk::UdfError> {
        Ok(Arc::new(Int64Array::from_iter_values(
            groups.iter().map(|&g| self.0[g as usize]),
        )))
    }

    fn size(&self) -> usize {
        self.0.capacity() * size_of::<i64>()
    }
}

/// `select k, count(v) group by k`, or the same through [`UCount`].
fn count_plan(batches: &[Batch], udaf: bool) -> Arc<dyn ExecNode> {
    let s = schema();
    let (aggregate, udafs) = if udaf {
        let library = krishiv_udf_sdk::UdfLibrary::new();
        library
            .register_aggregate(Arc::new(UCount(krishiv_udf_sdk::Signature::Exact(vec![
                krishiv_udf_sdk::UdfInputType::Exact(krishiv_types::DataType::Long),
            ]))))
            .expect("register");
        let call = UdafCall {
            name: "ucount".to_owned(),
            version: 1,
            args: vec![Field::new("arg0", DataType::Int64, false)],
            result: krishiv_types::DataType::Long,
            nullable: false,
            state: vec![Field::new("count", DataType::Int64, false)],
        };
        let expr = AggregateExpr {
            function: AggregateFunction::Udf(Box::new(call)),
            args: vec![1],
            input: AggInput::Raw,
            arg_type: DataType::Null,
            name: "c".to_owned(),
            mask: None,
            element: None,
        };
        let env = UdafEnv {
            library: Arc::new(library),
            time_zone: None,
        };
        (expr, Some(env))
    } else {
        (
            AggregateExpr::new(AggregateFunction::Count, 1, DataType::Int64, "c"),
            None,
        )
    };
    let spec = AggregateSpec {
        input: Arc::clone(&s),
        group_by: vec![0],
        aggregates: vec![aggregate],
        output: AggregateOutput::Final,
        ansi: true,
        spill: None,
        require_unit_weights: false,
        legacy_statistical: false,
        udafs,
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
        Field::new("s", DataType::Utf8View, false),
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
            let s = StringViewArray::from_iter_values(r.clone().map(|i| {
                let (a, b) = pair(i);
                format!("key-{a:03}-{b:03}")
            }));
            let v = Int64Array::from_iter_values(r.map(|i| i as i64));
            let cols: Vec<ArrayRef> = vec![Arc::new(k1), Arc::new(k2), Arc::new(s), Arc::new(v)];
            Batch::new(RecordBatch::try_new(wide_schema(), cols).expect("batch"))
        })
        .collect()
}

/// TPC-H q01's aggregate input: `(flag, status, qty, price, disc_price,
/// charge, discount)`, 4 key pairs.
fn q01_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("flag", DataType::Utf8View, false),
        Field::new("status", DataType::Utf8View, false),
        Field::new("qty", DECIMAL, false),
        Field::new("price", DECIMAL, false),
        Field::new("disc_price", DataType::Decimal128(32, 4), false),
        Field::new("charge", DataType::Decimal128(38, 6), false),
        Field::new("discount", DECIMAL, false),
    ]))
}

fn q01_input() -> Vec<Batch> {
    (0..ROWS)
        .step_by(DEFAULT_BATCH_SIZE)
        .map(|start| {
            let r = start as u64..(start + DEFAULT_BATCH_SIZE).min(ROWS) as u64;
            let mix = |i: u64| i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11;
            let flag = StringViewArray::from_iter_values(
                r.clone()
                    .map(|i| ["A", "N", "N", "R"][(mix(i) % 4) as usize]),
            );
            let status = StringViewArray::from_iter_values(
                r.clone()
                    .map(|i| ["F", "F", "O", "F"][(mix(i) % 4) as usize]),
            );
            let dec = |f: &dyn Fn(u64) -> i128, t: DataType| -> ArrayRef {
                Arc::new(Decimal128Array::from_iter_values(r.clone().map(f)).with_data_type(t))
            };
            let qty = |i: u64| i128::from(100 + (mix(i) >> 3) % 5_000);
            let price = |i: u64| i128::from(90_000 + (mix(i) >> 5) % 10_410_000);
            let disc = |i: u64| i128::from((mix(i) >> 7) % 11);
            let cols: Vec<ArrayRef> = vec![
                Arc::new(flag),
                Arc::new(status),
                dec(&qty, DECIMAL),
                dec(&price, DECIMAL),
                dec(&|i| price(i) * (100 - disc(i)), DataType::Decimal128(32, 4)),
                dec(
                    &|i| price(i) * (100 - disc(i)) * (100 + i128::from((mix(i) >> 9) % 9)),
                    DataType::Decimal128(38, 6),
                ),
                dec(&disc, DECIMAL),
            ];
            Batch::new(RecordBatch::try_new(q01_schema(), cols).expect("batch"))
        })
        .collect()
}

/// q01's `group by flag, status` with its eight aggregates.
fn q01_plan(batches: &[Batch]) -> Arc<dyn ExecNode> {
    let s = q01_schema();
    let d2 = DECIMAL;
    let spec = AggregateSpec {
        input: Arc::clone(&s),
        group_by: vec![0, 1],
        aggregates: vec![
            AggregateExpr::new(AggregateFunction::Sum, 2, d2.clone(), "sum_qty"),
            AggregateExpr::new(AggregateFunction::Sum, 3, d2.clone(), "sum_base_price"),
            AggregateExpr::new(
                AggregateFunction::Sum,
                4,
                DataType::Decimal128(32, 4),
                "sum_disc_price",
            ),
            AggregateExpr::new(
                AggregateFunction::Sum,
                5,
                DataType::Decimal128(38, 6),
                "sum_charge",
            ),
            AggregateExpr::new(AggregateFunction::Avg, 2, d2.clone(), "avg_qty"),
            AggregateExpr::new(AggregateFunction::Avg, 3, d2.clone(), "avg_price"),
            AggregateExpr::new(AggregateFunction::Avg, 6, d2, "avg_disc"),
            AggregateExpr::count_star("count_order"),
        ],
        output: AggregateOutput::Final,
        ansi: true,
        spill: None,
        require_unit_weights: false,
        legacy_statistical: false,
        udafs: None,
    };
    let scan = Arc::new(Scan(Batches::new(s, batches.to_vec())));
    AggregateNode::new(scan, HashAggregate::new(spec).expect("spec"))
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
        udafs: None,
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
        udafs: None,
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
    let q01 = q01_input();
    if let Ok(reps) = std::env::var("AGG_BENCH_Q01_REPS") {
        let mode = Mode { rows: true, seeded };
        // `AGG_BENCH_REPS_CASE`: `q01` (default), or `grouped_sum_{64,262144}_keys_rows`
        // as `few` / `many`, for instruction counts of the other shapes.
        let case = std::env::var("AGG_BENCH_REPS_CASE").unwrap_or_default();
        let grouped = match case.as_str() {
            "few" => Some(input(64)),
            "many" => Some(input(1 << 18)),
            other => {
                assert!(
                    matches!(other, "" | "q01"),
                    "AGG_BENCH_REPS_CASE: q01, few or many, not {other}"
                );
                None
            }
        };
        for _ in 0..reps.parse::<usize>().expect("AGG_BENCH_Q01_REPS: a count") {
            let p = grouped
                .as_ref()
                .map_or_else(|| q01_plan(&q01), |b| plan(b, None));
            black_box(run(&runner, &caller, p, 1 << 30, mode));
        }
        std::process::exit(0);
    }
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
    for keys in [64u64, 1 << 18] {
        let batches = input(keys);
        for (name, udaf) in [("count", false), ("udaf_count", true)] {
            let mode = modes[0].1;
            g.bench_function(format!("{name}_{keys}_keys"), |b| {
                b.iter(|| {
                    black_box(run(
                        &runner,
                        &caller,
                        count_plan(&batches, udaf),
                        1 << 30,
                        mode,
                    ))
                });
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
    g.bench_function("q01_shape_rows", |b| {
        let mode = Mode { rows: true, seeded };
        b.iter(|| black_box(run(&runner, &caller, q01_plan(&q01), 1 << 30, mode)));
    });
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
