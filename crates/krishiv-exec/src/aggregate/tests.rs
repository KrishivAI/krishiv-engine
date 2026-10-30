// SPDX-License-Identifier: Apache-2.0
//! Aggregation tests. Expected values come from Spark 4.2.0 runs (the
//! queries are quoted next to each case); the property test compares with
//! a row-at-a-time oracle.
#![cfg(not(loom))]
#![expect(
    clippy::disallowed_methods,
    reason = "test-only runtime plumbing; never runs on a compute worker"
)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Decimal128Type, Float64Type, Int8Type, Int32Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, Date32Array, Decimal128Array, Float64Array, Int8Array, Int32Array, Int64Array,
    RecordBatch, StringViewArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use krishiv_common::error::ErrorClass;
use krishiv_memory::{
    MemoryConfig, MemoryPool, SpillConfig, SpillManager, SpillOwner, SpillScope, TreePool,
};
use krishiv_types::batch::{Batch, Weights};
use proptest::prelude::*;

use super::accumulator::plain;
use super::*;
use crate::testing::{Batches, Scan};
use crate::{
    ExecConfig, ExecNode, Executor, OperatorId, QueryContext, QueryId, QueryMetrics, QueryRunner,
    Sink, Source, TaskContext, TokioClock, TokioExecutor, WorkloadClass,
};

use AggregateFunction::{Avg, Count, Max, Min, Sum};

/// Run options.
#[derive(Clone, Copy)]
struct Opts {
    slots: usize,
    limit: usize,
    /// Also run with `krishiv.exec.agg.rowLayout` on and require the same
    /// rows (or the same error).
    both_layouts: bool,
    /// The deterministic seed (morsel order); `None` claims morsels in
    /// order.
    seed: Option<u64>,
    /// Collect a detailed profile's counters (`ExecConfig::detailed_metrics`).
    detailed: bool,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            slots: 3,
            limit: 1 << 30,
            both_layouts: true,
            seed: Some(7),
            detailed: false,
        }
    }
}

fn pool(limit: usize) -> Arc<TreePool> {
    TreePool::with_clock(
        &MemoryConfig {
            limit: Some(limit),
            quantum: 16 << 10,
            spill_wait_timeout: Duration::from_secs(5),
            abort_release_timeout: Duration::from_secs(30),
            spill_scratch_reserve: krishiv_memory::DEFAULT_SPILL_SCRATCH_RESERVE,
            resident_guard: true,
        },
        Arc::new(TokioClock),
    )
}

/// Runs `plan` on a deterministic executor (seed `o.seed`); checks the pool is
/// empty afterwards. Unless `o.both_layouts` is off, runs it again with the
/// row-layout aggregate tables and checks the rows (or the error) are the
/// same; returns the columnar layout's outcome.
async fn exec(plan: Arc<dyn ExecNode>, o: Opts) -> crate::Result<(Vec<Batch>, QueryMetrics)> {
    let off = exec_with(Arc::clone(&plan), o, false).await;
    if o.both_layouts {
        let on = exec_with(plan, o, true).await;
        match (&off, &on) {
            (Ok(a), Ok(b)) => assert_eq!(rows(&a.0), rows(&b.0), "row layout changed the rows"),
            // Memory errors quote the bytes asked for, which differ.
            (Err(a), Err(b)) if a.error_class() == ErrorClass::UnableToAcquireMemory => {
                assert_eq!(b.error_class(), ErrorClass::UnableToAcquireMemory, "{b}");
            }
            (Err(a), Err(b)) => {
                assert_eq!(a.to_string(), b.to_string(), "row layout changed the error");
            }
            (a, b) => panic!(
                "row layout changed the outcome: {:?} vs {:?}",
                a.as_ref().err(),
                b.as_ref().err()
            ),
        }
    }
    off
}

/// Runs a fresh plan from `make` on each layout (operator counters add up
/// across runs of one plan) and requires the same rows; returns both
/// metrics, columnar first, so tests check each layout's spill counters.
async fn exec_each(
    make: impl Fn() -> Arc<dyn ExecNode>,
    o: Opts,
) -> (Vec<String>, [QueryMetrics; 2]) {
    let (off, m_off) = exec_with(make(), o, false).await.unwrap();
    let (on, m_on) = exec_with(make(), o, true).await.unwrap();
    assert_eq!(rows(&on), rows(&off), "row layout changed the rows");
    (rows(&off), [m_off, m_on])
}

/// [`exec`] with `krishiv.exec.agg.rowLayout` set to `rows`.
async fn exec_with(
    plan: Arc<dyn ExecNode>,
    o: Opts,
    rows: bool,
) -> crate::Result<(Vec<Batch>, QueryMetrics)> {
    let runner = QueryRunner::new(TokioExecutor::deterministic(o.slots)?);
    let root = pool(o.limit);
    let config = ExecConfig {
        seed: o.seed,
        agg_row_layout: rows,
        detailed_metrics: o.detailed,
        ..ExecConfig::default()
    };
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        config,
        root.child("q", Some(o.limit)),
    );
    let mut stream = runner.execute(plan, Arc::clone(&q)).await?;
    let mut out = Vec::new();
    let mut err = None;
    while let Some(b) = stream.next().await {
        match b {
            Ok(b) => out.push(b),
            Err(e) => err = Some(e),
        }
    }
    drop(stream);
    q.finished().await;
    assert_eq!(root.reserved(), 0, "memory left after the query");
    err.map_or(Ok((out, q.metrics())), Err)
}

/// A spill manager on its own temporary directory. Fields drop in order: the
/// manager removes its session, then the directory itself goes (also when
/// the test panics).
struct TestSpill {
    manager: SpillManager,
    _dir: tempfile::TempDir,
}

/// A spill directory for one test.
fn spill_dir() -> (TestSpill, SpillScope) {
    spill_dir_capped(SpillConfig::default().max_bytes)
}

/// [`spill_dir`] holding at most `max_bytes` of spill files.
fn spill_dir_capped(max_bytes: u64) -> (TestSpill, SpillScope) {
    let dir = tempfile::Builder::new()
        .prefix("krishiv-agg-")
        .tempdir()
        .unwrap();
    let manager = SpillManager::open(&SpillConfig {
        // A path inside the temporary directory: the manager creates it
        // private (0700), whatever the umask.
        dir: dir.path().join("spill"),
        max_bytes,
        ..SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    (TestSpill { manager, _dir: dir }, scope)
}

fn cell(a: &ArrayRef, i: usize) -> String {
    if a.is_null(i) {
        return "NULL".to_owned();
    }
    match a.data_type() {
        DataType::Int64 => a.as_primitive::<Int64Type>().value(i).to_string(),
        DataType::Int32 => a.as_primitive::<Int32Type>().value(i).to_string(),
        DataType::Int8 => a.as_primitive::<Int8Type>().value(i).to_string(),
        DataType::Float64 => format!("{:?}", a.as_primitive::<Float64Type>().value(i)),
        DataType::Decimal128(_, s) => plain(a.as_primitive::<Decimal128Type>().value(i), *s),
        DataType::Date32 => a.as_primitive::<Date32Type>().value(i).to_string(),
        DataType::Utf8View => a.as_string_view().value(i).to_owned(),
        _ => format!("{:?}", a.slice(i, 1)).replace('\n', " "),
    }
}

/// Rows as `a|b|c`, sorted.
fn rows(batches: &[Batch]) -> Vec<String> {
    let mut out: Vec<String> = batches
        .iter()
        .flat_map(|b| {
            let d = b.data();
            (0..d.num_rows()).map(move |r| {
                let cells: Vec<String> = d.columns().iter().map(|c| cell(c, r)).collect();
                cells.join("|")
            })
        })
        .collect();
    out.sort();
    out
}

fn schema(fields: Vec<(&str, DataType)>) -> SchemaRef {
    let fields: Vec<Field> = fields
        .into_iter()
        .map(|(n, t)| Field::new(n, t, true))
        .collect();
    Arc::new(Schema::new(fields))
}

fn scan(s: &SchemaRef, batches: Vec<Vec<ArrayRef>>) -> Arc<dyn ExecNode> {
    let b = batches
        .into_iter()
        .map(|c| Batch::new(RecordBatch::try_new(Arc::clone(s), c).unwrap()))
        .collect();
    Arc::new(Scan(Batches::new(Arc::clone(s), b)))
}

fn agg(f: AggregateFunction, col: usize, s: &SchemaRef) -> AggregateExpr {
    let t = s.field(col).data_type().clone();
    AggregateExpr::new(f, col, t, format!("{f:?}_{col}"))
}

fn spec(s: &SchemaRef, group_by: Vec<usize>, aggregates: Vec<AggregateExpr>) -> AggregateSpec {
    AggregateSpec {
        input: Arc::clone(s),
        group_by,
        aggregates,
        output: AggregateOutput::Final,
        ansi: true,
        spill: None,
        require_unit_weights: false,
        legacy_statistical: false,
    }
}

fn node(input: Arc<dyn ExecNode>, spec: AggregateSpec) -> Arc<dyn ExecNode> {
    AggregateNode::new(input, HashAggregate::new(spec).unwrap())
}

async fn run_rows(plan: Arc<dyn ExecNode>) -> Vec<String> {
    rows(&exec(plan, Opts::default()).await.unwrap().0)
}

fn i64s(v: &[Option<i64>]) -> ArrayRef {
    Arc::new(Int64Array::from(v.to_vec()))
}

fn dec(v: &[&str], p: u8, s: i8) -> ArrayRef {
    let parse = |t: &str| {
        let (int, frac) = t.split_once('.').unwrap_or((t, ""));
        let frac = format!("{frac:0<width$}", width = usize::try_from(s).unwrap());
        format!("{int}{frac}").parse::<i128>().unwrap()
    };
    let a = Decimal128Array::from(v.iter().map(|t| parse(t)).collect::<Vec<_>>());
    Arc::new(a.with_precision_and_scale(p, s).unwrap())
}

fn err_text(r: crate::Result<(Vec<Batch>, QueryMetrics)>) -> String {
    r.err().map(|e| e.to_string()).unwrap_or_default()
}

/// One of an aggregate's own `EXPLAIN ANALYZE` counters.
fn sink_metric(agg: &HashAggregate, name: &str) -> u64 {
    Sink::extra_metrics(agg)
        .iter()
        .filter(|(n, _)| *n == name)
        .map(|(_, v)| match v {
            crate::metrics::Metric::Count(n) | crate::metrics::Metric::Max(n) => *n,
            crate::metrics::Metric::Text(_) => 0,
        })
        .sum()
}

fn extra(m: &QueryMetrics, op: &str, name: &str) -> u64 {
    m.operators
        .values()
        .filter(|e| e.name == op)
        .flat_map(|e| e.extra.iter())
        .filter(|(n, _)| *n == name)
        .map(|(_, v)| match v {
            crate::metrics::Metric::Count(n) | crate::metrics::Metric::Max(n) => *n,
            crate::metrics::Metric::Text(_) => 0,
        })
        .sum()
}

#[test]
fn result_types_follow_spark_4_2() {
    let t = |f, arg: DataType| {
        AggregateExpr::new(f, 0, arg, "x")
            .result_type()
            .map_err(|e| e.error_class())
    };
    let d = DataType::Decimal128;
    assert_eq!(t(Sum, DataType::Int8), Ok(DataType::Int64));
    assert_eq!(t(Sum, DataType::Float32), Ok(DataType::Float64));
    assert_eq!(t(Sum, d(15, 2)), Ok(d(25, 2)));
    assert_eq!(t(Sum, d(38, 18)), Ok(d(38, 18)));
    assert_eq!(t(Avg, DataType::Int32), Ok(DataType::Float64));
    assert_eq!(t(Avg, DataType::Float32), Ok(DataType::Float64));
    assert_eq!(t(Avg, d(15, 2)), Ok(d(19, 6)));
    assert_eq!(t(Avg, d(38, 18)), Ok(d(38, 22)));
    assert_eq!(t(Avg, d(30, 10)), Ok(d(34, 14)));
    assert_eq!(t(Count, DataType::Utf8View), Ok(DataType::Int64));
    assert_eq!(t(Min, DataType::Date32), Ok(DataType::Date32));
    assert_eq!(t(Max, DataType::Utf8View), Ok(DataType::Utf8View));
    let unsupported = Err(ErrorClass::KrishivUnsupportedFeature);
    assert_eq!(t(Sum, DataType::Utf8View), unsupported);
    // Negative decimal scales (Spark refuses them by default).
    assert_eq!(t(Avg, d(5, -2)), unsupported);
    assert_eq!(t(Sum, d(5, -2)), unsupported);
    assert_eq!(t(Max, d(5, -2)), Ok(d(5, -2)));
    let e = AggregateExpr::new(Avg, 0, d(15, 2), "a");
    assert_eq!(e.merging(3).args, vec![3, 4]);
    let s = schema(vec![("k", DataType::Int64)]);
    assert!(spec(&s, vec![4], vec![]).output_schema().is_err());
}

/// `select count(*), count(x), sum(x), avg(x), min(x), max(x) from t where
/// false` → `0 0 NULL NULL NULL NULL`; grouped → no rows.
#[tokio::test]
async fn empty_input() {
    let s = schema(vec![("k", DataType::Int64), ("x", DataType::Int64)]);
    let aggs = || {
        vec![
            AggregateExpr::count_star("c"),
            agg(Count, 1, &s),
            agg(Sum, 1, &s),
            agg(Avg, 1, &s),
            agg(Min, 1, &s),
            agg(Max, 1, &s),
        ]
    };
    let empty = || scan(&s, vec![vec![i64s(&[]), i64s(&[])]]);
    let global = run_rows(node(empty(), spec(&s, vec![], aggs()))).await;
    assert_eq!(global, ["0|0|NULL|NULL|NULL|NULL"]);
    assert!(
        run_rows(node(empty(), spec(&s, vec![0], aggs())))
            .await
            .is_empty()
    );
    let none = run_rows(node(scan(&s, vec![]), spec(&s, vec![], aggs()))).await;
    assert_eq!(none, ["0|0|NULL|NULL|NULL|NULL"]);
}

/// `select k, count(*), count(x), sum(x), min(x) from values (null, 1),
/// (null, 2), (1, null), (1, null) t(k, x) group by k` →
/// `NULL 2 2 3 1` and `1 2 0 NULL NULL`.
#[tokio::test]
async fn null_keys_group_together_and_all_null_groups() {
    let s = schema(vec![("k", DataType::Int64), ("x", DataType::Int64)]);
    let input = scan(
        &s,
        vec![
            vec![i64s(&[None, Some(1)]), i64s(&[Some(1), None])],
            vec![i64s(&[None, Some(1)]), i64s(&[Some(2), None])],
        ],
    );
    let aggs = vec![
        AggregateExpr::count_star("c"),
        agg(Count, 1, &s),
        agg(Sum, 1, &s),
        agg(Min, 1, &s),
    ];
    let out = run_rows(node(input, spec(&s, vec![0], aggs))).await;
    assert_eq!(out, ["1|2|0|NULL|NULL", "NULL|2|2|3|1"]);
}

/// `group by x` over `0.0, -0.0, NaN, NaN` → `0.0 2`, `NaN 2`; `min, max`
/// of `1.0, NaN, -1.0` → `-1.0 NaN`; of `0.0, -0.0` the first is kept
/// (`0.0 0.0`), of `-0.0, 0.0` → `-0.0 -0.0`; strings by bytes (`'' ä`).
#[tokio::test]
async fn float_string_and_date_ordering() {
    let s = schema(vec![("x", DataType::Float64)]);
    let f = |v: Vec<f64>| -> ArrayRef { Arc::new(Float64Array::from(v)) };
    let keys = scan(&s, vec![vec![f(vec![0.0, -0.0, f64::NAN, -f64::NAN])]]);
    let count = vec![AggregateExpr::count_star("c")];
    let out = run_rows(node(keys, spec(&s, vec![0], count))).await;
    assert_eq!(out, ["0.0|2", "NaN|2"]);
    let one = |v| {
        let mm = vec![agg(Min, 0, &s), agg(Max, 0, &s)];
        node(scan(&s, vec![vec![f(v)]]), spec(&s, vec![], mm))
    };
    assert_eq!(run_rows(one(vec![1.0, f64::NAN, -1.0])).await, ["-1.0|NaN"]);
    assert_eq!(run_rows(one(vec![0.0, -0.0])).await, ["0.0|0.0"]);
    assert_eq!(run_rows(one(vec![-0.0, 0.0])).await, ["-0.0|-0.0"]);

    let s = schema(vec![("s", DataType::Utf8View), ("d", DataType::Date32)]);
    let strs: ArrayRef = Arc::new(StringViewArray::from(vec![
        Some("b"),
        Some("a"),
        Some("ä"),
        Some("Z"),
        Some(""),
        None,
    ]));
    let days = vec![Some(18262), Some(-1), None, None, None, None];
    let dates: ArrayRef = Arc::new(Date32Array::from(days));
    let aggs = vec![
        agg(Min, 0, &s),
        agg(Max, 0, &s),
        agg(Min, 1, &s),
        agg(Max, 1, &s),
    ];
    let plan = node(scan(&s, vec![vec![strs, dates]]), spec(&s, vec![], aggs));
    assert_eq!(run_rows(plan).await, ["|ä|-1|18262"]);
}

/// Decimal `avg` and `sum`, and overflow in ANSI and legacy mode (Spark
/// 4.2.0 messages).
#[tokio::test]
async fn decimals_and_overflow() {
    let one = |t: DataType, col: ArrayRef, f: AggregateFunction, ansi: bool| {
        let s = schema(vec![("x", t)]);
        let mut sp = spec(&s, vec![], vec![agg(f, 0, &s)]);
        sp.ansi = ansi;
        node(scan(&s, vec![vec![col]]), sp)
    };
    let d = DataType::Decimal128;
    let o = Opts::default();
    let plan = one(d(15, 2), dec(&["1", "2", "2"], 15, 2), Avg, true);
    assert_eq!(run_rows(plan).await, ["1.666667"]);
    let plan = one(d(38, 18), dec(&["1", "2", "2"], 38, 18), Avg, true);
    assert_eq!(run_rows(plan).await, ["1.6666666666666666666667"]);
    let plan = one(d(38, 0), dec(&["-1", "-2", "-2"], 38, 0), Avg, true);
    assert_eq!(run_rows(plan).await, ["-1.6667"]);
    // The sum fits decimal(38,18), the average does not fit decimal(38,22)
    // (Spark's message: the quotient at scale 39).
    let big = || dec(&["100000000000000000", "100000000000000000"], 38, 18);
    let r = exec(one(d(38, 18), big(), Avg, true), o).await;
    assert_eq!(
        err_text(r),
        "[NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION]  \
         100000000000000000.000000000000000000000000000000000000000 cannot be represented \
         as Decimal(38, 22). If necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass \
         this error, and return NULL instead. SQLSTATE: 22003"
    );
    assert_eq!(run_rows(one(d(38, 18), big(), Avg, false)).await, ["NULL"]);
    // A running sum past decimal(38,18) overflows (strict, see below).
    let past = dec(&["99999999999999999999.999999999999999999", "1"], 38, 18);
    let r = err_text(exec(one(d(38, 18), past, Avg, true), o).await);
    assert!(r.starts_with("[ARITHMETIC_OVERFLOW] Overflow in sum of decimals. Use 'try_avg'"));
    let nines = "9".repeat(38);
    let over = || dec(&[&nines, "1"], 38, 0);
    // The running sum may not leave the sum type's precision, grouped or
    // not, even if later rows bring it back (Spark's single-partition global
    // aggregate tolerates it and gives …995: a divergence, ADR 0023).
    let back = || dec(&[&nines, "1", "-5"], 38, 0);
    for col in [over(), back()] {
        let r = err_text(exec(one(d(38, 0), Arc::clone(&col), Sum, true), o).await);
        assert!(
            r.starts_with("[ARITHMETIC_OVERFLOW] Overflow in sum of decimals. Use 'try_sum'"),
            "{r}"
        );
        assert_eq!(run_rows(one(d(38, 0), col, Sum, false)).await, ["NULL"]);
    }
    // Grouped, the running sum may not leave the precision even if it comes
    // back: `select k, sum(x) / avg(x) from values (1, 9…9 (38 digits)),
    // (1, 1), (1, -5) as t(k, x) group by k` → ARITHMETIC_OVERFLOW (legacy:
    // NULL), with or without whole-stage codegen; in the order 9…9, -5, 1
    // it never leaves and gives 9…95.
    let grouped = |xs: &[&str], f: AggregateFunction, ansi: bool| {
        let s = schema(vec![("k", DataType::Int64), ("x", d(38, 0))]);
        let k = i64s(&vec![Some(1); xs.len()]);
        let mut sp = spec(&s, vec![0], vec![agg(f, 1, &s)]);
        sp.ansi = ansi;
        node(scan(&s, vec![vec![k, dec(xs, 38, 0)]]), sp)
    };
    let up = [nines.as_str(), "1", "-5"];
    for (f, hint) in [(Sum, "try_sum"), (Avg, "try_avg")] {
        let r = err_text(exec(grouped(&up, f, true), o).await);
        assert!(
            r.starts_with(&format!(
                "[ARITHMETIC_OVERFLOW] Overflow in sum of decimals. Use '{hint}'"
            )),
            "{r}"
        );
        assert_eq!(run_rows(grouped(&up, f, false)).await, ["1|NULL"]);
    }
    let down = [nines.as_str(), "-5", "1"];
    let want = format!("1|{}5", "9".repeat(37));
    assert_eq!(run_rows(grouped(&down, Sum, true)).await, [want]);
    // Past i128: "Overflow in sum of decimals" (sum and avg).
    let huge = || dec(&[nines.as_str(); 4], 38, 0);
    for (f, hint) in [(Sum, "try_sum"), (Avg, "try_avg")] {
        let r = exec(one(d(38, 0), huge(), f, true), o).await;
        assert_eq!(
            err_text(r),
            format!(
                "[ARITHMETIC_OVERFLOW] Overflow in sum of decimals. Use '{hint}' to tolerate \
                 overflow and return NULL instead. If necessary set \"spark.sql.ansi.enabled\" \
                 to \"false\" to bypass this error. SQLSTATE: 22003"
            )
        );
        assert_eq!(run_rows(one(d(38, 0), huge(), f, false)).await, ["NULL"]);
    }
    // bigint: ANSI raises, legacy wraps; avg sums in double.
    let longs = || i64s(&[Some(i64::MAX), Some(1)]);
    let r = exec(one(DataType::Int64, longs(), Sum, true), o).await;
    assert_eq!(
        err_text(r),
        "[ARITHMETIC_OVERFLOW] overflow. Use 'try_add' to tolerate overflow and return NULL \
         instead. If necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error. \
         SQLSTATE: 22003"
    );
    let wrapped = run_rows(one(DataType::Int64, longs(), Sum, false)).await;
    assert_eq!(wrapped, [i64::MIN.to_string()]);
    let two_max = i64s(&[Some(i64::MAX), Some(i64::MAX)]);
    let avg = run_rows(one(DataType::Int64, two_max, Avg, true)).await;
    assert_eq!(avg, ["9.223372036854776e18"]);
    let ints: ArrayRef = Arc::new(Int32Array::from(vec![i32::MAX, i32::MAX]));
    assert_eq!(
        run_rows(one(DataType::Int32, ints, Sum, true)).await,
        ["4294967294"]
    );
}

/// `select sum(distinct x), count(distinct x), avg(distinct x) from values
/// (1), (1), (2), (null)` → `3 2 1.5`; `count(distinct x, y)` counts
/// non-NULL pairs; `select k, count(distinct x) … group by k`; UNION.
#[tokio::test]
async fn distinct_aggregates_and_union() {
    let s = schema(vec![
        ("k", DataType::Int64),
        ("x", DataType::Int64),
        ("y", DataType::Int64),
    ]);
    let k = i64s(&[Some(1), Some(1), Some(1), None, None]);
    let x = i64s(&[Some(1), Some(1), None, Some(2), Some(2)]);
    let y = i64s(&[Some(1), Some(1), Some(1), None, Some(4)]);
    let input = || {
        scan(
            &s,
            vec![vec![Arc::clone(&k), Arc::clone(&x), Arc::clone(&y)]],
        )
    };
    let all = vec![agg(Sum, 1, &s), agg(Count, 1, &s), agg(Avg, 1, &s)];
    let plan = distinct_plan(input(), spec(&s, vec![], all), &[true, true, true]).unwrap();
    assert_eq!(run_rows(plan).await, ["3|2|1.5"]);
    let mut pair = agg(Count, 1, &s);
    pair.args = vec![1, 2];
    let plan = distinct_plan(input(), spec(&s, vec![], vec![pair]), &[true]).unwrap();
    assert_eq!(run_rows(plan).await, ["2"]);
    // count(distinct x), sum(y), count(*) group by k.
    let mixed = vec![
        agg(Count, 1, &s),
        agg(Sum, 2, &s),
        AggregateExpr::count_star("c"),
    ];
    let flags = [true, false, false];
    let plan = distinct_plan(input(), spec(&s, vec![0], mixed), &flags).unwrap();
    assert_eq!(run_rows(plan).await, ["1|1|3|3", "NULL|1|4|2"]);
    let plain = distinct_plan(
        input(),
        spec(&s, vec![0], vec![agg(Count, 1, &s)]),
        &[false],
    );
    assert_eq!(run_rows(plain.unwrap()).await, ["1|2", "NULL|2"]);
    let two = vec![agg(Count, 1, &s), agg(Count, 2, &s)];
    let e = distinct_plan(input(), spec(&s, vec![], two), &[true, true]).err();
    assert_eq!(
        e.map(|e| e.error_class()),
        Some(ErrorClass::KrishivUnsupportedFeature)
    );
    assert!(distinct_plan(input(), spec(&s, vec![], vec![]), &[true]).is_err());

    // `select x from values (1), (1), (null), (null), (2) union select 3`.
    let u = schema(vec![("x", DataType::Int64)]);
    let rows_in = scan(
        &u,
        vec![
            vec![i64s(&[Some(1), Some(1), None])],
            vec![i64s(&[None, Some(2), Some(3)])],
        ],
    );
    let plan = union_distinct(rows_in, Arc::clone(&u), None).unwrap();
    assert_eq!(run_rows(plan).await, ["1", "2", "3", "NULL"]);
}

fn masked(f: AggregateFunction, col: usize, mask: usize, s: &SchemaRef) -> AggregateExpr {
    AggregateExpr {
        mask: Some(mask),
        ..agg(f, col, s)
    }
}

fn bools(v: &[Option<bool>]) -> ArrayRef {
    Arc::new(arrow_array::BooleanArray::from(v.to_vec()))
}

/// `FILTER (WHERE ...)`: a masked aggregate reads only the rows where the
/// mask column is true; a NULL mask is not true. Expected values are
/// Spark 4.2's (`group-by-filter.sql`: `count(x) FILTER (WHERE m)` counts
/// the passing rows with non-NULL `x`, `count(*)` the passing rows, `sum`
/// of no rows is NULL).
#[tokio::test]
async fn filter_mask_reads_only_passing_rows() {
    let s = schema(vec![("x", DataType::Int64), ("m", DataType::Boolean)]);
    // x: 1, 2, NULL, 4, 5; m: true, false, true, NULL, true.
    let x = i64s(&[Some(1), Some(2), None, Some(4), Some(5)]);
    let m = bools(&[Some(true), Some(false), Some(true), None, Some(true)]);
    let input = || scan(&s, vec![vec![Arc::clone(&x), Arc::clone(&m)]]);
    let aggs = || {
        vec![
            AggregateExpr::count_star("c"),
            masked(Count, 0, 1, &s),
            masked(Sum, 0, 1, &s),
            masked(Avg, 0, 1, &s),
            masked(Min, 0, 1, &s),
            masked(Max, 0, 1, &s),
        ]
    };
    // Unmasked counts beside masked calls: the counts read all rows,
    // the other calls only the passing ones (rows 0, 2, 4).
    let mut all = aggs();
    all[0].mask = None;
    all[1].mask = None;
    assert_eq!(
        run_rows(node(input(), spec(&s, vec![], all))).await,
        ["5|4|6|3.0|1|5"]
    );
    let mut only_masked = aggs();
    for a in &mut only_masked {
        a.mask = Some(1);
    }
    // count(*): rows 0, 2, 4; count(x): rows 0, 4; sum: 1 + 5.
    assert_eq!(
        run_rows(node(input(), spec(&s, vec![], only_masked))).await,
        ["3|2|6|3.0|1|5"]
    );
    // A mask that passes nothing: counts are 0, other values NULL.
    let none = bools(&[Some(false), None, Some(false), Some(false), None]);
    let input_none = || scan(&s, vec![vec![Arc::clone(&x), Arc::clone(&none)]]);
    // The unmasked `count(*)` still reads all rows; the masked calls
    // read none.
    assert_eq!(
        run_rows(node(input_none(), spec(&s, vec![], aggs()))).await,
        ["5|0|NULL|NULL|NULL|NULL"]
    );
    let mut masked_none = aggs();
    for a in &mut masked_none {
        a.mask = Some(1);
    }
    assert_eq!(
        run_rows(node(input_none(), spec(&s, vec![], masked_none))).await,
        ["0|0|NULL|NULL|NULL|NULL"]
    );
    // Grouped: k holds 10 and 20 in group 1, 30 in group 2; the mask
    // drops the 20.
    let g = schema(vec![
        ("k", DataType::Int64),
        ("x", DataType::Int64),
        ("m", DataType::Boolean),
    ]);
    let grouped = scan(
        &g,
        vec![vec![
            i64s(&[Some(1), Some(1), Some(2)]),
            i64s(&[Some(10), Some(20), Some(30)]),
            bools(&[Some(true), Some(false), Some(true)]),
        ]],
    );
    let gaggs = vec![
        masked(Count, 1, 2, &g),
        masked(Sum, 1, 2, &g),
        masked(Min, 1, 2, &g),
    ];
    assert_eq!(
        run_rows(node(grouped, spec(&g, vec![0], gaggs))).await,
        ["1|1|10|10", "2|1|30|30"]
    );
}

/// A masked `DISTINCT` never reaches the single-distinct plan: the
/// analyzer rewrites every one (group formation itself must skip the
/// rows the mask drops), so this plan rejects them instead of
/// miscounting. `merging` drops the mask: states hold filtered rows.
#[tokio::test]
async fn filter_mask_with_distinct_is_rejected() {
    let s = schema(vec![("x", DataType::Int64), ("m", DataType::Boolean)]);
    // `count(distinct x) FILTER (WHERE m)`: groups would form from the
    // rows the mask drops, so the count would include them.
    let x = i64s(&[Some(1), Some(2)]);
    let m = bools(&[Some(false), Some(true)]);
    let input = || scan(&s, vec![vec![Arc::clone(&x), Arc::clone(&m)]]);
    let e = distinct_plan(
        input(),
        spec(&s, vec![], vec![masked(Count, 0, 1, &s)]),
        &[true],
    )
    .err();
    assert_eq!(
        e.map(|e| e.error_class()),
        Some(ErrorClass::KrishivUnsupportedFeature)
    );
    assert!(masked(Count, 0, 1, &s).merging(0).mask.is_none());
}

/// Z-set weights: `count` and `sum` scale; `min`/`max` and `DISTINCT`
/// reject weights other than 1 until IVM.
#[tokio::test]
async fn weights_scale_sum_and_count() {
    let s = schema(vec![("k", DataType::Int64), ("x", DataType::Int64)]);
    let cols = vec![
        i64s(&[Some(1), Some(1), Some(2)]),
        i64s(&[Some(10), Some(5), None]),
    ];
    let data = RecordBatch::try_new(Arc::clone(&s), cols).unwrap();
    let weighted = |w: Weights| -> Arc<dyn ExecNode> {
        let b = Batch::with_weights(data.clone(), w).unwrap();
        Arc::new(Scan(Batches::new(Arc::clone(&s), vec![b])))
    };
    let w = || Weights::Column(Int64Array::from(vec![2, -1, 3]));
    let aggs = vec![
        AggregateExpr::count_star("c"),
        agg(Count, 1, &s),
        agg(Sum, 1, &s),
        agg(Avg, 1, &s),
    ];
    let plan = node(weighted(w()), spec(&s, vec![0], aggs));
    assert_eq!(run_rows(plan).await, ["1|1|1|15|15.0", "2|3|0|NULL|NULL"]);
    let plan = node(
        weighted(Weights::Const(3)),
        spec(&s, vec![], vec![agg(Sum, 1, &s)]),
    );
    assert_eq!(run_rows(plan).await, ["45"]);
    let unsupported = Some(ErrorClass::KrishivUnsupportedFeature);
    let plan = node(weighted(w()), spec(&s, vec![0], vec![agg(Min, 1, &s)]));
    let e = exec(plan, Opts::default()).await.err();
    assert_eq!(e.map(|e| e.error_class()), unsupported);
    let plan = union_distinct(weighted(w()), Arc::clone(&s), None).unwrap();
    let e = exec(plan, Opts::default()).await.err();
    assert_eq!(e.map(|e| e.error_class()), unsupported);
}

/// `n` rows `k = i % keys`, `x = i` (NULL every 7th), in batches of 1000.
fn grouped_input(n: i64, keys: i64) -> (SchemaRef, Vec<Vec<ArrayRef>>) {
    let s = schema(vec![("k", DataType::Int64), ("x", DataType::Int64)]);
    let batches = (0..n)
        .step_by(1000)
        .map(|start| {
            let r = start..(start + 1000).min(n);
            let k: Vec<_> = r.clone().map(|i| Some(i % keys)).collect();
            let x: Vec<_> = r.map(|i| (i % 7 != 0).then_some(i)).collect();
            vec![i64s(&k), i64s(&x)]
        })
        .collect();
    (s, batches)
}

fn grouped_aggs(s: &SchemaRef) -> Vec<AggregateExpr> {
    vec![
        AggregateExpr::count_star("c"),
        agg(Sum, 1, s),
        agg(Min, 1, s),
        agg(Avg, 1, s),
    ]
}

/// The final phase merging `first`'s partial states.
fn second_phase(first: &AggregateSpec, spill: Option<SpillScope>) -> AggregateSpec {
    let mid = first.output_schema().unwrap();
    let mut at = first.group_by.len();
    let merging = first
        .aggregates
        .iter()
        .map(|a| {
            let m = a.merging(at);
            at += m.args.len();
            m
        })
        .collect();
    let mut second = spec(&mid, (0..first.group_by.len()).collect(), merging);
    second.spill = spill;
    second
}

/// Partial → final over the partial states.
fn two_phase(
    input: Arc<dyn ExecNode>,
    s: &SchemaRef,
    policy: BypassPolicy,
    spill: Option<SpillScope>,
) -> (Arc<dyn ExecNode>, Arc<PartialAggregate>) {
    two_phase_of(input, spec(s, vec![0], grouped_aggs(s)), policy, spill)
}

/// [`two_phase`] of the aggregation `first` (raw input, final output).
fn two_phase_of(
    input: Arc<dyn ExecNode>,
    mut first: AggregateSpec,
    policy: BypassPolicy,
    spill: Option<SpillScope>,
) -> (Arc<dyn ExecNode>, Arc<PartialAggregate>) {
    first.output = AggregateOutput::Partial;
    let mut second = second_phase(&first, spill);
    second.legacy_statistical = first.legacy_statistical;
    second.ansi = first.ansi;
    let partial = PartialAggregate::new(first, policy, Some(7)).unwrap();
    let plan = node(PartialNode::new(input, Arc::clone(&partial)), second);
    (plan, partial)
}

fn single(s: &SchemaRef, b: &[Vec<ArrayRef>]) -> Arc<dyn ExecNode> {
    node(scan(s, b.to_vec()), spec(s, vec![0], grouped_aggs(s)))
}

/// Two-phase equals single-phase; high-cardinality keys switch the partial
/// phase to pass-through, low-cardinality ones do not.
#[tokio::test]
async fn two_phase_and_adaptive_bypass() {
    let policy = BypassPolicy {
        min_rows: 10_000,
        ..BypassPolicy::default()
    };
    for (keys, bypass) in [(1_000_000, true), (50, false)] {
        let (s, b) = grouped_input(60_000, keys);
        let expected = run_rows(single(&s, &b)).await;
        let (plan, partial) = two_phase(scan(&s, b), &s, policy, None);
        let (out, m) = exec(plan, Opts::default()).await.unwrap();
        assert_eq!(rows(&out), expected);
        assert_eq!(partial.bypassed_drivers() > 0, bypass, "keys {keys}");
        assert_eq!(
            extra(&m, "PartialAggregate", "bypassed_drivers") > 0,
            bypass
        );
        assert_eq!(extra(&m, "PartialAggregate", "bypassed_rows") > 0, bypass);
        assert_eq!(extra(&m, "HashAggregate", "groups"), expected.len() as u64);
    }
    // A tiny table cap flushes instead of growing.
    let (s, b) = grouped_input(20_000, 5_000);
    let expected = run_rows(single(&s, &b)).await;
    let tiny = BypassPolicy {
        max_table_bytes: 64 << 10,
        min_rows: u64::MAX,
        ..BypassPolicy::default()
    };
    let (plan, _) = two_phase(scan(&s, b), &s, tiny, None);
    let (out, m) = exec(plan, Opts::default()).await.unwrap();
    assert_eq!(rows(&out), expected);
    assert!(extra(&m, "PartialAggregate", "flushes") > 3);
    // Each flushed table grew from empty; its successor counts into the
    // same total.
    assert!(extra(&m, "PartialAggregate", "hash_resizes") > 3);
    // Unique keys with a table that flushes long before `min_rows`: the
    // reduction is judged across flushes, so the partial still bypasses.
    let (s, b) = grouped_input(40_000, 1_000_000);
    let expected = run_rows(single(&s, &b)).await;
    let flushing = BypassPolicy {
        max_table_bytes: 64 << 10,
        min_rows: 10_000,
        ..BypassPolicy::default()
    };
    let (plan, p) = two_phase(scan(&s, b), &s, flushing, None);
    assert_eq!(
        rows(&exec(plan, Opts::default()).await.unwrap().0),
        expected
    );
    assert!(p.bypassed_drivers() > 0);
    assert!(PartialAggregate::new(spec(&s, vec![0], vec![]), tiny, None).is_err());
}

/// Under a tiny budget the aggregate spills runs and merges them partition
/// by partition; the result equals the in-memory one.
#[tokio::test]
async fn spill_matches_in_memory() {
    let (s, b) = grouped_input(100_000, 75_000);
    let expected = run_rows(single(&s, &b)).await;
    for limit in [4 << 20, 2 << 20] {
        let (_m, scope) = spill_dir();
        let mut sp = spec(&s, vec![0], grouped_aggs(&s));
        sp.spill = Some(scope.clone());
        let o = Opts {
            limit,
            ..Opts::default()
        };
        let (out, metrics) = exec_each(|| node(scan(&s, b.clone()), sp.clone()), o).await;
        assert_eq!(out, expected, "limit {limit}");
        // Both layouts spilled.
        for m in &metrics {
            assert!(extra(m, "HashAggregate", "spill_runs") > 0, "limit {limit}");
            assert!(extra(m, "HashAggregate", "spill_bytes") > 0);
            assert_eq!(extra(m, "HashAggregate", "groups"), 75_000);
        }
        let policy = BypassPolicy::default();
        let (plan, _) = two_phase(scan(&s, b.clone()), &s, policy, Some(scope));
        assert_eq!(rows(&exec(plan, o).await.unwrap().0), expected);
    }
}

/// A partition that still does not fit is split by the next hash bits.
#[tokio::test]
async fn spill_repartitions_a_partition_that_does_not_fit() {
    let (s, b) = grouped_input(60_000, 60_000);
    let expected = run_rows(single(&s, &b)).await;
    let (_m, scope) = spill_dir();
    let mut sp = spec(&s, vec![0], grouped_aggs(&s));
    sp.spill = Some(scope);
    let o = Opts {
        limit: 512 << 10,
        slots: 2,
        ..Opts::default()
    };
    let (out, metrics) = exec_each(|| node(scan(&s, b.clone()), sp.clone()), o).await;
    assert_eq!(out, expected);
    // Both layouts split a partition.
    for m in &metrics {
        assert!(extra(m, "HashAggregate", "repartitions") > 0);
    }
}

/// Without a spill directory a table that outgrows the pool fails the
/// query with a memory error and releases everything.
#[tokio::test]
async fn unspillable_out_of_memory_fails_cleanly() {
    let (s, b) = grouped_input(100_000, 100_000);
    let o = Opts {
        limit: 1 << 20,
        ..Opts::default()
    };
    assert!(exec(single(&s, &b), o).await.is_err());
}

/// `spill_global` during `finalize` (an arbitration request after the last
/// driver combined) moves the merge to disk; the result is unchanged.
#[tokio::test]
async fn spill_global_during_finalize() {
    let (s, b) = grouped_input(30_000, 20_000);
    let expected = run_rows(single(&s, &b)).await;
    for rows_layout in [false, true] {
        let (_m, scope) = spill_dir();
        let mut sp = spec(&s, vec![0], grouped_aggs(&s));
        sp.spill = Some(scope);
        let agg = HashAggregate::new(sp).unwrap();
        let root = pool(1 << 30);
        let config = ExecConfig {
            agg_row_layout: rows_layout,
            ..ExecConfig::default()
        };
        let q = QueryContext::new(
            QueryId(9),
            WorkloadClass::Batch,
            config,
            root.child("q", None),
        );
        let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
        let ctx = |id, name| {
            let e = Arc::clone(&executor);
            TaskContext::new(Arc::clone(&q), e, SpillOwner::new(), OperatorId(id), name)
        };
        let sink_ctx = ctx(0, "agg");
        let g = Sink::global_state(&*agg, &sink_ctx).unwrap();
        for part in b.chunks(10) {
            let mut l = Sink::local_state(&*agg, &sink_ctx, &g).unwrap();
            for cols in part {
                let batch = Batch::new(RecordBatch::try_new(Arc::clone(&s), cols.clone()).unwrap());
                agg.sink(batch, &mut l, &g).unwrap();
            }
            agg.combine(l, &g).unwrap();
        }
        assert!(agg.spill_global(&g, 1, &sink_ctx).await.unwrap() > 0);
        assert_eq!(agg.spill_global(&g, 1, &sink_ctx).await.unwrap(), 0);
        agg.finalize(&g, &sink_ctx).await.unwrap();
        assert_eq!(agg.spill_global(&g, 1, &sink_ctx).await.unwrap(), 0);
        let src = ctx(1, "src").with_dependency(Arc::clone(&g));
        let sg = Source::global_state(&*agg, &src).unwrap();
        assert_eq!(agg.max_parallelism(&sg), 1);
        let mut sl = Source::local_state(&*agg, &src, &sg).unwrap();
        let mut out = Vec::new();
        while let Some(b) = agg.get_data(&mut sl, &sg).await.unwrap() {
            out.push(b);
        }
        assert_eq!(rows(&out), expected);
        assert!(agg.metrics().spill_runs >= 2);
        drop((sl, sg, src, g, sink_ctx));
        assert_eq!(root.reserved(), 0);
    }
}

/// Same seed, same output order and metrics (deterministic mode).
#[tokio::test]
async fn deterministic_replay() {
    let (s, b) = grouped_input(50_000, 9_000);
    let run = || async {
        let (_m, scope) = spill_dir();
        let mut sp = spec(&s, vec![0], grouped_aggs(&s));
        sp.spill = Some(scope);
        let o = Opts {
            limit: 2 << 20,
            ..Opts::default()
        };
        let (out, m) = exec(node(scan(&s, b.clone()), sp), o).await.unwrap();
        let ordered: Vec<String> = out.iter().map(|b| format!("{:?}", b.data())).collect();
        (ordered, extra(&m, "HashAggregate", "spill_runs"))
    };
    let first = run().await;
    assert!(first.1 > 0);
    assert_eq!(first, run().await);
}

/// An error mid-aggregation releases every reservation and spill file.
#[tokio::test]
async fn error_mid_aggregation_releases_memory() {
    let (s, mut b) = grouped_input(40_000, 40_000);
    b.push(vec![
        i64s(&[Some(1), Some(1)]),
        i64s(&[Some(i64::MAX), Some(i64::MAX)]),
    ]);
    let (_m, scope) = spill_dir();
    let mut sp = spec(&s, vec![0], vec![agg(Sum, 1, &s)]);
    sp.spill = Some(scope);
    let o = Opts {
        limit: 2 << 20,
        ..Opts::default()
    };
    let e = exec(node(scan(&s, b), sp), o).await.err();
    assert_eq!(
        e.map(|e| e.error_class()),
        Some(ErrorClass::ArithmeticOverflow)
    );
}

type OracleRow = (i64, i64, Option<i64>, Option<i32>, Option<i32>);

/// The row-at-a-time oracle: count(*), count(x), sum(x), min(x), max(x)
/// per key.
fn oracle(keys: &[Option<i8>], xs: &[Option<i32>]) -> Vec<String> {
    let mut m: BTreeMap<Option<i8>, OracleRow> = BTreeMap::new();
    for (k, x) in keys.iter().zip(xs) {
        let e = m.entry(*k).or_insert((0, 0, None, None, None));
        e.0 += 1;
        if let Some(x) = *x {
            e.1 += 1;
            e.2 = Some(e.2.unwrap_or(0) + i64::from(x));
            e.3 = Some(e.3.map_or(x, |v| v.min(x)));
            e.4 = Some(e.4.map_or(x, |v| v.max(x)));
        }
    }
    let o = |v: Option<String>| v.unwrap_or_else(|| "NULL".to_owned());
    let mut out: Vec<String> = m
        .into_iter()
        .map(|(k, (c, cx, s, mn, mx))| {
            let k = o(k.map(|k| k.to_string()));
            let s = o(s.map(|v| v.to_string()));
            let mn = o(mn.map(|v| v.to_string()));
            let mx = o(mx.map(|v| v.to_string()));
            format!("{k}|{c}|{cx}|{s}|{mn}|{mx}")
        })
        .collect();
    out.sort();
    out
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]
    #[test]
    fn matches_the_oracle(
        data in proptest::collection::vec(
            (proptest::option::of(any::<i8>()), proptest::option::of(any::<i32>())),
            0..3000,
        ),
        batch in 1usize..700,
        slots in 1usize..4,
        two in any::<bool>(),
        spill in any::<bool>(),
    ) {
        let (keys, xs): (Vec<_>, Vec<_>) = data.iter().copied().unzip();
        let s = schema(vec![("k", DataType::Int8), ("x", DataType::Int32)]);
        let batches = keys
            .chunks(batch)
            .zip(xs.chunks(batch))
            .map(|(k, x)| {
                let k: ArrayRef = Arc::new(Int8Array::from(k.to_vec()));
                let x: ArrayRef = Arc::new(Int32Array::from(x.to_vec()));
                vec![k, x]
            })
            .collect();
        let aggs = vec![
            AggregateExpr::count_star("c"),
            agg(Count, 1, &s),
            agg(Sum, 1, &s),
            agg(Min, 1, &s),
            agg(Max, 1, &s),
        ];
        let (_m, scope) = spill_dir();
        let mut sp = spec(&s, vec![0], aggs);
        sp.spill = spill.then_some(scope);
        let plan = if two {
            let mut first = sp.clone();
            first.output = AggregateOutput::Partial;
            first.spill = None;
            let second = second_phase(&first, sp.spill);
            let policy = BypassPolicy { min_rows: 100, ..BypassPolicy::default() };
            let p = PartialAggregate::new(first, policy, Some(1)).unwrap();
            node(PartialNode::new(scan(&s, batches), p), second)
        } else {
            node(scan(&s, batches), sp)
        };
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let limit = if spill { 1 << 20 } else { 1 << 30 };
        let out = rt.block_on(exec(plan, Opts { slots, limit, ..Opts::default() })).unwrap().0;
        prop_assert_eq!(rows(&out), oracle(&keys, &xs));
    }
}

/// Every argument type through both phases (raw updates, state merges and
/// evaluation): the two-phase result equals the single-phase one.
#[tokio::test]
async fn every_type_through_both_phases() {
    use arrow_array::{BinaryArray, BinaryViewArray, BooleanArray, Float32Array, Int16Array};
    use arrow_array::{StringArray, TimestampMicrosecondArray};
    let s = schema(vec![
        ("k", DataType::Int64),
        ("i8", DataType::Int8),
        ("i16", DataType::Int16),
        ("b", DataType::Boolean),
        ("d", DataType::Decimal128(10, 2)),
        (
            "ts",
            DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
        ),
        ("s", DataType::Utf8),
        ("bin", DataType::Binary),
        ("bv", DataType::BinaryView),
        ("f", DataType::Float32),
        ("sv", DataType::Utf8View),
    ]);
    let n = 40usize;
    let col = |f: &dyn Fn(usize) -> Option<i64>| -> Vec<Option<i64>> { (0..n).map(f).collect() };
    let v = col(&|i| (i % 5 != 0).then_some(i64::try_from(i).unwrap() * 7 % 23 - 11));
    let text: Vec<Option<String>> = v.iter().map(|x| x.map(|x| format!("v{x}"))).collect();
    let bytes: Vec<Option<&[u8]>> = text
        .iter()
        .map(|t| t.as_deref().map(str::as_bytes))
        .collect();
    let i8s: Vec<Option<i8>> = v
        .iter()
        .map(|x| x.map(|x| i8::try_from(x).unwrap()))
        .collect();
    let i16s: Vec<Option<i16>> = v
        .iter()
        .map(|x| x.map(|x| i16::try_from(x * 100).unwrap()))
        .collect();
    #[expect(clippy::cast_precision_loss, reason = "small test values")]
    let f32s: Vec<Option<f32>> = v.iter().map(|x| x.map(|x| x as f32 / 4.0)).collect();
    let cols: Vec<ArrayRef> = vec![
        i64s(&col(&|i| Some(i64::try_from(i % 3).unwrap()))),
        Arc::new(Int8Array::from(i8s)),
        Arc::new(Int16Array::from(i16s)),
        Arc::new(BooleanArray::from(
            v.iter().map(|x| x.map(|x| x > 0)).collect::<Vec<_>>(),
        )),
        Arc::new(
            Decimal128Array::from(v.iter().map(|x| x.map(i128::from)).collect::<Vec<_>>())
                .with_precision_and_scale(10, 2)
                .unwrap(),
        ),
        Arc::new(TimestampMicrosecondArray::from(v.clone()).with_timezone("UTC")),
        Arc::new(StringArray::from(text.clone())),
        Arc::new(BinaryArray::from(bytes.clone())),
        Arc::new(BinaryViewArray::from(bytes)),
        Arc::new(Float32Array::from(f32s)),
        Arc::new(StringViewArray::from(text)),
    ];
    let mut aggs = Vec::new();
    for c in 1..=10 {
        aggs.push(agg(Min, c, &s));
        aggs.push(agg(Max, c, &s));
    }
    for (f, c) in [
        (Sum, 1),
        (Sum, 2),
        (Sum, 9),
        (Avg, 9),
        (Avg, 1),
        (Sum, 4),
        (Avg, 4),
    ] {
        aggs.push(agg(f, c, &s));
    }
    let halves = vec![
        cols.iter().map(|c| c.slice(0, n / 2)).collect(),
        cols.iter().map(|c| c.slice(n / 2, n / 2)).collect(),
    ];
    let sp = spec(&s, vec![0], aggs);
    let expected = run_rows(node(scan(&s, halves.clone()), sp.clone())).await;
    assert_eq!(expected.len(), 3);
    let bools = "BooleanArray [   false, ]|BooleanArray [   true, ]";
    assert!(expected.iter().all(|r| r.contains(bools)), "{expected:?}");
    let mut first = sp.clone();
    first.output = AggregateOutput::Partial;
    let second = second_phase(&first, None);
    for min_rows in [1, 1_000_000] {
        let policy = BypassPolicy {
            min_rows,
            ..BypassPolicy::default()
        };
        let p = PartialAggregate::new(first.clone(), policy, None).unwrap();
        assert_eq!(crate::Operator::schema(&*p), first.output_schema().unwrap());
        let plan = node(
            PartialNode::new(scan(&s, halves.clone()), p),
            second.clone(),
        );
        assert_eq!(run_rows(plan).await, expected, "min_rows {min_rows}");
    }
    assert!(format!("{:?}", node(scan(&s, vec![]), sp)).contains("ExecNode"));
}

/// The bypass decision on the last batch leaves the flushed table to
/// `finish`; a LIMIT that stops during `finish` ends the pipeline.
#[tokio::test]
async fn bypass_on_the_last_batch_and_limit_during_finish() {
    use crate::testing::{Break, Count, Limit, Pipe};
    let (s, b) = grouped_input(20_000, 1_000_000);
    let expected = run_rows(single(&s, &b)).await;
    let policy = BypassPolicy {
        min_rows: 20_000,
        ..BypassPolicy::default()
    };
    let o = Opts {
        slots: 1,
        ..Opts::default()
    };
    let (plan, p) = two_phase(scan(&s, b.clone()), &s, policy, None);
    assert_eq!(rows(&exec(plan, o).await.unwrap().0), expected);
    // Once per layout (`exec` runs both).
    assert_eq!(p.bypassed_drivers(), 2);
    let mut first = spec(&s, vec![0], grouped_aggs(&s));
    first.output = AggregateOutput::Partial;
    let p = PartialAggregate::new(first, BypassPolicy::default(), None).unwrap();
    let limited = Arc::new(Pipe(PartialNode::new(scan(&s, b), p), Limit::new(5)));
    let plan = Arc::new(Break(limited, Count::new(), false));
    // No hash aggregate, and the limit counts across runs: one run.
    let o = Opts {
        both_layouts: false,
        ..o
    };
    let out = exec(plan, o).await.unwrap().0;
    assert_eq!(
        out[0].data().column(0).as_primitive::<Int64Type>().value(0),
        5
    );
}

/// `finalize` without a spill directory: a merge that does not fit fails
/// with a memory error; a global aggregate and an empty grouped one spilled
/// by `spill_global` still return their rows.
#[tokio::test]
async fn finalize_edge_cases() {
    let (s, b) = grouped_input(30_000, 30_000);
    for rows_layout in [false, true] {
        // Three disjoint tables that fit, whose merge does not.
        let r = finalize_directly(
            spec(&s, vec![0], grouped_aggs(&s)),
            &s,
            &b,
            3,
            3 << 20,
            false,
            rows_layout,
        )
        .await;
        assert_eq!(
            r.err().map(|e| e.error_class()),
            Some(ErrorClass::UnableToAcquireMemory)
        );
        let (_m, scope) = spill_dir();
        let mut global = spec(&s, vec![], grouped_aggs(&s));
        global.spill = Some(scope.clone());
        let out = finalize_directly(global, &s, &b, 3, 1 << 30, true, rows_layout)
            .await
            .unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].starts_with("30000|"), "{out:?}");
        let mut empty = spec(&s, vec![0], grouped_aggs(&s));
        empty.spill = Some(scope);
        let out = finalize_directly(empty, &s, &[], 1, 1 << 30, true, rows_layout)
            .await
            .unwrap();
        assert!(out.is_empty());
    }
}

/// A driver whose own table is not worth spilling (a global aggregate's one
/// group) still spills the tables other drivers combined, and its own table
/// stays accounted.
#[tokio::test]
async fn local_spill_keeps_an_unspilled_table_accounted() {
    let (s, b) = grouped_input(1_000, 1_000);
    let (_m, scope) = spill_dir();
    let mut sp = spec(&s, vec![], grouped_aggs(&s));
    sp.spill = Some(scope);
    let agg = HashAggregate::new(sp).unwrap();
    let root = pool(1 << 30);
    let q = QueryContext::new(
        QueryId(10),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("q", Some(1 << 30)),
    );
    let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
    let ctx = TaskContext::new(q, executor, SpillOwner::new(), OperatorId(0), "agg");
    let g = Sink::global_state(&*agg, &ctx).unwrap();
    let batch = || Batch::new(RecordBatch::try_new(Arc::clone(&s), b[0].clone()).unwrap());
    let mut first = Sink::local_state(&*agg, &ctx, &g).unwrap();
    agg.sink(batch(), &mut first, &g).unwrap();
    agg.combine(first, &g).unwrap();
    let mut second = Sink::local_state(&*agg, &ctx, &g).unwrap();
    agg.sink(batch(), &mut second, &g).unwrap();
    let before = root.reserved();
    let freed = agg.spill(&mut second, 1, &ctx).await.unwrap();
    assert!(freed > 0 && freed < before, "freed {freed} of {before}");
    // The kept table still holds its bytes and spill headroom.
    assert!(root.reserved() >= HEADROOM_FOR_TESTS, "{}", root.reserved());
    agg.combine(second, &g).unwrap();
    agg.finalize(&g, &ctx).await.unwrap();
    // The combined table's run, then finalize spills the kept one.
    assert_eq!(agg.metrics().spill_runs, 2);
    let src = ctx.with_dependency(Arc::clone(&g));
    let sg = Source::global_state(&*agg, &src).unwrap();
    let mut sl = Source::local_state(&*agg, &src, &sg).unwrap();
    let mut out = Vec::new();
    while let Some(b) = agg.get_data(&mut sl, &sg).await.unwrap() {
        out.push(b);
    }
    let out = rows(&out);
    assert!(out.len() == 1 && out[0].starts_with("2000|"), "{out:?}");
    drop((sl, sg, src, g));
    assert_eq!(root.reserved(), 0);
}

/// The spill headroom a spillable table keeps reserved (`sink::HEADROOM`).
const HEADROOM_FOR_TESTS: usize = 128 << 10;

/// Sinks `b` through `drivers` local states, combines, optionally asks
/// `spill_global`, finalizes and reads the source side.
async fn finalize_directly(
    sp: AggregateSpec,
    s: &SchemaRef,
    b: &[Vec<ArrayRef>],
    drivers: usize,
    limit: usize,
    spill_first: bool,
    rows_layout: bool,
) -> crate::Result<Vec<String>> {
    let agg = HashAggregate::new(sp).unwrap();
    let root = pool(1 << 30);
    let config = ExecConfig {
        agg_row_layout: rows_layout,
        ..ExecConfig::default()
    };
    let q = QueryContext::new(
        QueryId(9),
        WorkloadClass::Batch,
        config,
        root.child("q", Some(limit)),
    );
    let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
    let ctx = |id, name| {
        let e = Arc::clone(&executor);
        TaskContext::new(Arc::clone(&q), e, SpillOwner::new(), OperatorId(id), name)
    };
    let sink_ctx = ctx(0, "agg");
    let g = Sink::global_state(&*agg, &sink_ctx).unwrap();
    let per = b.len().div_ceil(drivers).max(1);
    let mut parts: Vec<&[Vec<ArrayRef>]> = b.chunks(per).collect();
    parts.resize(drivers.max(parts.len()), &[]);
    for part in parts {
        let mut l = Sink::local_state(&*agg, &sink_ctx, &g).unwrap();
        for cols in part {
            let batch = Batch::new(RecordBatch::try_new(Arc::clone(s), cols.clone()).unwrap());
            agg.sink(batch, &mut l, &g)?;
        }
        agg.combine(l, &g).unwrap();
    }
    if spill_first {
        agg.spill_global(&g, 1, &sink_ctx).await?;
    }
    agg.finalize(&g, &sink_ctx).await?;
    let src = ctx(1, "src").with_dependency(Arc::clone(&g));
    let sg = Source::global_state(&*agg, &src).unwrap();
    assert!(!Source::schema(&*agg).fields().is_empty() || b.is_empty());
    let mut sl = Source::local_state(&*agg, &src, &sg).unwrap();
    let mut out = Vec::new();
    while let Some(b) = agg.get_data(&mut sl, &sg).await? {
        out.push(b);
    }
    Ok(rows(&out))
}

/// A LIMIT upstream of a partial aggregate stops the scan, but the partial
/// table is still flushed; a partial phase of a DISTINCT/UNION spec rejects
/// weights; a bad grouping column is an error, not a panic.
#[tokio::test]
async fn limit_before_partial_and_spec_errors() {
    use crate::testing::{Limit, Pipe};
    let (s, b) = grouped_input(10_000, 3);
    let mut first = spec(&s, vec![0], vec![AggregateExpr::count_star("c")]);
    first.output = AggregateOutput::Partial;
    let second = second_phase(&first, None);
    // The limit counts across runs: a fresh plan per layout.
    let plan = || {
        let p = PartialAggregate::new(first.clone(), BypassPolicy::default(), None).unwrap();
        let limited = Arc::new(Pipe(scan(&s, b.clone()), Limit::new(5)));
        node(PartialNode::new(limited, p), second.clone())
    };
    let o = Opts {
        slots: 1,
        both_layouts: false,
        ..Opts::default()
    };
    // Five rows of one (seeded) batch: three keys, five rows in all.
    let out = rows(&exec(plan(), o).await.unwrap().0);
    let total: i64 = out.iter().map(|r| r[2..].parse::<i64>().unwrap()).sum();
    assert_eq!((out.len(), total), (3, 5), "{out:?}");
    assert_eq!(rows(&exec_with(plan(), o, true).await.unwrap().0), out);

    let data =
        RecordBatch::try_new(Arc::clone(&s), vec![i64s(&[Some(1)]), i64s(&[Some(1)])]).unwrap();
    let weighted = Batch::with_weights(data, Weights::Const(2)).unwrap();
    first.require_unit_weights = true;
    let p = PartialAggregate::new(first, BypassPolicy::default(), None).unwrap();
    let input = Arc::new(Scan(Batches::new(Arc::clone(&s), vec![weighted])));
    let plan = Arc::new(Pipe(input, p as Arc<dyn crate::Operator>));
    let e = exec(
        Arc::new(crate::testing::Break(
            plan,
            crate::testing::Count::new(),
            false,
        )),
        o,
    )
    .await;
    assert_eq!(
        e.err().map(|e| e.error_class()),
        Some(ErrorClass::KrishivUnsupportedFeature)
    );

    let bad = HashAggregate::new(spec(&s, vec![9], vec![])).err();
    assert_eq!(
        bad.map(|e| e.error_class()),
        Some(ErrorClass::InternalError)
    );
}

/// Tables combined by drivers that ended are re-charged to the last driver
/// to combine: its consumer holds all of them (what the arbitrator sizes
/// spill requests by), and nothing is lost or double-counted.
#[tokio::test]
async fn combined_tables_move_to_the_last_driver() {
    let (s, b) = grouped_input(9_000, 9_000);
    for rows_layout in [false, true] {
        let agg = HashAggregate::new(spec(&s, vec![0], grouped_aggs(&s))).unwrap();
        let root = pool(1 << 30);
        let config = ExecConfig {
            agg_row_layout: rows_layout,
            ..ExecConfig::default()
        };
        let q = QueryContext::new(
            QueryId(3),
            WorkloadClass::Batch,
            config,
            root.child("q", None),
        );
        let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
        let ctx = |id| {
            let e = Arc::clone(&executor);
            TaskContext::new(Arc::clone(&q), e, SpillOwner::new(), OperatorId(id), "agg")
        };
        let drivers: Vec<TaskContext> = (0..3).map(ctx).collect();
        let g = Sink::global_state(&*agg, &drivers[0]).unwrap();
        for (d, part) in drivers.iter().zip(b.chunks(3)) {
            let mut l = Sink::local_state(&*agg, d, &g).unwrap();
            for cols in part {
                let batch = Batch::new(RecordBatch::try_new(Arc::clone(&s), cols.clone()).unwrap());
                agg.sink(batch, &mut l, &g).unwrap();
            }
            agg.combine(l, &g).unwrap();
        }
        let held = root.reserved();
        let last = drivers[2].consumer().consumer().peak();
        assert!(last * 10 >= held * 9, "last driver holds {last} of {held}");
        // Another driver may finalize: the tables move to its consumer first.
        let first = drivers[0].consumer().consumer().peak();
        agg.finalize(&g, &drivers[0]).await.unwrap();
        let after = drivers[0].consumer().consumer().peak();
        assert!(first < held / 2, "driver 0 held {first} of {held} before");
        assert!(
            after >= held * 9 / 10,
            "finalizing driver holds {after} of {held}"
        );
        drop((g, drivers));
        assert_eq!(root.reserved(), 0);
    }
}

fn picks_input() -> (SchemaRef, Vec<Vec<ArrayRef>>) {
    let s = schema(vec![
        ("k", DataType::Int64),
        ("x", DataType::Int64),
        ("s", DataType::Utf8View),
    ]);
    let k = i64s(&[Some(1), Some(1), Some(1), Some(2), Some(2), Some(3)]);
    let x = i64s(&[None, Some(5), Some(7), None, None, Some(9)]);
    let t: ArrayRef = Arc::new(StringViewArray::from(vec![
        Some("a"),
        None,
        Some("c"),
        None,
        None,
        Some("z"),
    ]));
    // Two batches, so the group state carries across them.
    let split = |a: &ArrayRef| (a.slice(0, 2), a.slice(2, 4));
    let ((k1, k2), (x1, x2), (t1, t2)) = (split(&k), split(&x), split(&t));
    (s, vec![vec![k1, x1, t1], vec![k2, x2, t2]])
}

/// One slot, so batches reach the aggregate in order.
async fn run_one(plan: Arc<dyn ExecNode>) -> Vec<String> {
    let o = Opts {
        slots: 1,
        ..Opts::default()
    };
    rows(&exec(plan, o).await.unwrap().0)
}

fn picks() -> Vec<(AggregateFunction, usize)> {
    use AggregateFunction::{First, Last};
    let mut out = Vec::new();
    for col in [1, 2] {
        for f in [
            First {
                ignore_nulls: false,
            },
            Last {
                ignore_nulls: false,
            },
            First { ignore_nulls: true },
            Last { ignore_nulls: true },
        ] {
            out.push((f, col));
        }
    }
    out
}

/// `first`/`last` of a group in input order, a NULL counting unless
/// `ignoreNulls` (Spark 4.2: `SELECT k, first(x), last(x), first(x, true),
/// last(x, true), first(s), ... FROM t GROUP BY k` over this input).
#[tokio::test]
async fn first_and_last_pick_rows_in_input_order() {
    let (s, b) = picks_input();
    let aggs: Vec<_> = picks().into_iter().map(|(f, c)| agg(f, c, &s)).collect();
    let single = || node(scan(&s, b.clone()), spec(&s, vec![0], aggs.clone()));
    let expected = [
        // k | first x, last x, first x ignoring NULLs, last x ignoring NULLs,
        //   | the same for s
        "1|NULL|7|5|7|a|c|a|c",
        "2|NULL|NULL|NULL|NULL|NULL|NULL|NULL|NULL",
        "3|9|9|9|9|z|z|z|z",
    ];
    assert_eq!(run_one(single()).await, expected);
    // Partial states merge to the same result (one partial phase keeps the
    // input order).
    let mut first = spec(&s, vec![0], aggs.clone());
    first.output = AggregateOutput::Partial;
    let second = second_phase(&first, None);
    let partial = PartialAggregate::new(first, BypassPolicy::default(), Some(7)).unwrap();
    let plan = node(PartialNode::new(scan(&s, b.clone()), partial), second);
    assert_eq!(run_one(plan).await, expected);
    // A global aggregate over no rows is NULL, not an error.
    let none = node(
        scan(
            &s,
            vec![vec![
                i64s(&[]),
                i64s(&[]),
                Arc::new(StringViewArray::from(Vec::<&str>::new())),
            ]],
        ),
        spec(&s, vec![], aggs),
    );
    assert_eq!(
        run_one(none).await,
        ["NULL|NULL|NULL|NULL|NULL|NULL|NULL|NULL"]
    );
}

mod simple;
mod structs;

#[test]
fn first_and_last_types_and_states() {
    use AggregateFunction::{First, Last};
    let s = schema(vec![("x", DataType::Int32), ("a", DataType::Boolean)]);
    for f in [
        First { ignore_nulls: true },
        Last {
            ignore_nulls: false,
        },
    ] {
        let a = agg(f, 0, &s);
        assert_eq!(a.result_type().unwrap(), DataType::Int32);
        let names: Vec<_> = a
            .state_fields()
            .unwrap()
            .iter()
            .map(|f| (f.name().clone(), f.data_type().clone()))
            .collect();
        assert_eq!(names[1].1, DataType::Boolean);
        assert_eq!(a.merging(3).args, [3, 4]);
    }
    let unsupported = AggregateExpr::new(
        First {
            ignore_nulls: false,
        },
        0,
        DataType::Null,
        "f",
    );
    assert_eq!(
        unsupported.result_type().unwrap_err().error_class(),
        ErrorClass::KrishivUnsupportedFeature
    );
}

/// Input of the row-layout property test: keys `(k, f, b)` (an `Int32`
/// from `0..keys`, a `Float64` from a set with `-0.0` and NaN, a
/// `Boolean`), values `x` (`Int64`), `d` (`decimal(12,2)`), `s` (strings)
/// and `g = x / 2` (`Float64`).
type RowCase = (
    Option<i32>,
    Option<u8>,
    Option<bool>,
    Option<i64>,
    Option<i64>,
    Option<u16>,
);

fn row_case_batches(data: &[RowCase], keys: i32, batch: usize) -> (SchemaRef, Vec<Vec<ArrayRef>>) {
    use arrow_array::BooleanArray;
    let s = schema(vec![
        ("k", DataType::Int32),
        ("f", DataType::Float64),
        ("b", DataType::Boolean),
        ("x", DataType::Int64),
        ("d", DataType::Decimal128(12, 2)),
        ("s", DataType::Utf8View),
        ("g", DataType::Float64),
    ]);
    const FLOATS: [f64; 5] = [0.0, -0.0, f64::NAN, 1.5, -2.25];
    let batches = data
        .chunks(batch.max(1))
        .map(|c| {
            let k: Int32Array = c.iter().map(|r| r.0.map(|v| v.rem_euclid(keys))).collect();
            let f: Float64Array = c
                .iter()
                .map(|r| r.1.map(|v| FLOATS[usize::from(v) % FLOATS.len()]))
                .collect();
            let b: BooleanArray = c.iter().map(|r| r.2).collect();
            let x: Int64Array = c.iter().map(|r| r.3).collect();
            let d = Decimal128Array::from_iter(c.iter().map(|r| r.4.map(i128::from)))
                .with_precision_and_scale(12, 2)
                .unwrap();
            // Inline (up to 12 bytes) and heap strings.
            let s = StringViewArray::from_iter(c.iter().map(|r| {
                r.5.map(|v| {
                    if v % 3 == 0 {
                        format!("v{}", v % 977)
                    } else {
                        format!("value-{:05}-with-a-long-tail", v % 977)
                    }
                })
            }));
            // No -0.0 among the values: `min`/`max` of -0.0 and 0.0 keep
            // whichever comes first, so they depend on the merge order.
            #[expect(clippy::cast_precision_loss, reason = "small integers")]
            let g: Float64Array = c.iter().map(|r| r.3.map(|v| v as f64 * 0.5)).collect();
            let cols: Vec<ArrayRef> = vec![
                Arc::new(k),
                Arc::new(f),
                Arc::new(b),
                Arc::new(x),
                Arc::new(d),
                Arc::new(s),
                Arc::new(g),
            ];
            cols
        })
        .collect();
    (s, batches)
}

/// Aggregates of every accumulator kind; `picks` adds `first`/`last`,
/// which depend on the order of the drivers' partial states.
fn row_case_aggs(s: &SchemaRef, picks: bool) -> Vec<AggregateExpr> {
    use AggregateFunction::{First, Last};
    let mut a = vec![
        AggregateExpr::count_star("c"),
        agg(Count, 3, s),
        agg(Sum, 3, s),
        agg(Avg, 3, s),
        agg(Sum, 4, s),
        agg(Avg, 4, s),
        agg(Min, 6, s),
        agg(Max, 6, s),
        agg(Max, 4, s),
        agg(Min, 5, s),
        agg(Max, 5, s),
    ];
    if picks {
        a.extend([
            agg(
                First {
                    ignore_nulls: false,
                },
                3,
                s,
            ),
            agg(Last { ignore_nulls: true }, 5, s),
            agg(First { ignore_nulls: true }, 4, s),
        ]);
    }
    a
}

fn row_mode(kind: u8, limit: usize) -> rowtable::Mode {
    match kind {
        0 => rowtable::Mode::Grow,
        1 => rowtable::Mode::Fixed { limit },
        _ => rowtable::Mode::Append,
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, ..ProptestConfig::default() })]
    /// The row-layout tables return exactly the columnar tables' rows: one
    /// to three keys (with NULL, `-0.0`, NaN), every accumulator, partial
    /// duplicates from cleared indexes and append-only tables, one and
    /// several drivers, two phases, spilling under a tight memory limit.
    #[test]
    fn row_layout_equals_default(
        data in proptest::collection::vec(
            (
                proptest::option::of(any::<i32>()),
                proptest::option::of(any::<u8>()),
                proptest::option::of(any::<bool>()),
                proptest::option::of(-1_000_000i64..1_000_000),
                proptest::option::of(-10_000_000i64..10_000_000),
                proptest::option::of(any::<u16>()),
            ),
            0..4000,
        ),
        keys in 1i32..3000,
        nkeys in 1usize..=6,
        batch in 1usize..900,
        slots in 1usize..4,
        mode in 0u8..3,
        limit in 1usize..64,
        two in any::<bool>(),
        tight in any::<bool>(),
    ) {
        // Tight (see `row_layout_tight_memory` for tighter).
        row_case(&data, keys, nkeys, batch, slots, mode, limit, two, tight.then_some(1 << 20))?;
    }
}

/// Cases of `row_layout_tight_memory`: 64 per PR, more where
/// `KRISHIV_TIGHT_MEMORY_CASES` says (the nightly runs 256).
fn tight_memory_cases() -> u32 {
    std::env::var("KRISHIV_TIGHT_MEMORY_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(64)
}

/// One case of [`row_layout_equals_default`]: `tight` is the memory limit
/// with a spill scope (`None`: unlimited, unspillable).
#[expect(clippy::too_many_arguments, reason = "the property's parameters")]
fn row_case(
    data: &[RowCase],
    keys: i32,
    nkeys: usize,
    batch: usize,
    slots: usize,
    mode: u8,
    limit: usize,
    two: bool,
    tight: Option<usize>,
) -> Result<(), TestCaseError> {
    let (s, batches) = row_case_batches(data, keys, batch);
    let (_m, scope) = spill_dir();
    // Fixed-width keys, then sets with the string column (5).
    let keys = match nkeys {
        1 => vec![0],
        2 => vec![0, 1],
        3 => vec![0, 1, 2],
        4 => vec![5],
        5 => vec![0, 5],
        _ => vec![5, 1, 2],
    };
    let mut sp = spec(&s, keys, row_case_aggs(&s, slots == 1));
    sp.spill = tight.map(|_| scope);
    let plan = |agg: Arc<HashAggregate>, sp: &AggregateSpec| -> Arc<dyn ExecNode> {
        if two {
            let mut first = sp.clone();
            first.output = AggregateOutput::Partial;
            first.spill = None;
            let policy = BypassPolicy {
                min_rows: 100,
                ..BypassPolicy::default()
            };
            let p = PartialAggregate::new(first, policy, Some(1)).expect("partial spec");
            AggregateNode::new(PartialNode::new(scan(&s, batches.clone()), p), agg)
        } else {
            AggregateNode::new(scan(&s, batches.clone()), agg)
        }
    };
    let final_spec = if two {
        let mut first = sp.clone();
        first.output = AggregateOutput::Partial;
        second_phase(&first, sp.spill.clone())
    } else {
        sp.clone()
    };
    let o = Opts {
        slots,
        limit: tight.unwrap_or(1 << 30),
        both_layouts: false,
        ..Opts::default()
    };
    let fail = |e: &dyn std::fmt::Display| TestCaseError::fail(e.to_string());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| fail(&e))?;
    let off = plan(
        HashAggregate::new(final_spec.clone()).map_err(|e| fail(&e))?,
        &sp,
    );
    // Only cases the columnar tables finish compare; running out of memory
    // is the one expected columnar failure.
    let want = match rt.block_on(exec(off, o)) {
        Ok((want, _)) => want,
        Err(e) if e.error_class() == ErrorClass::UnableToAcquireMemory => {
            return Err(TestCaseError::reject(
                "the columnar tables ran out of memory",
            ));
        }
        Err(e) => return Err(fail(&e)),
    };
    let agg =
        HashAggregate::with_row_mode(final_spec, row_mode(mode, limit)).map_err(|e| fail(&e))?;
    let got = rt
        .block_on(exec_with(plan(agg, &sp), o, true))
        .map_err(|e| {
            TestCaseError::fail(format!("row layout failed where the columnar did not: {e}"))
        })?;
    prop_assert_eq!(rows(&got.0), rows(&want));
    Ok(())
}

/// One driver whose table spills several times still picks `first`/`last`
/// in input order: spill runs merge in the order their rows arrived, on
/// both layouts (the row tables also through partial duplicates).
#[tokio::test]
async fn first_and_last_keep_input_order_through_spills() {
    use AggregateFunction::{First, Last};
    let (n, keys) = (60_000i64, 20_000i64);
    let (s, b) = grouped_input(n, keys);
    let (_m, scope) = spill_dir();
    let aggs = vec![
        agg(First { ignore_nulls: true }, 1, &s),
        agg(Last { ignore_nulls: true }, 1, &s),
        agg(
            First {
                ignore_nulls: false,
            },
            1,
            &s,
        ),
    ];
    let mut sp = spec(&s, vec![0], aggs);
    sp.spill = Some(scope);
    // No seed: one driver claims the batches in order.
    let o = Opts {
        slots: 1,
        limit: 1 << 20,
        seed: None,
        ..Opts::default()
    };
    let (out, metrics) = exec_each(|| node(scan(&s, b.clone()), sp.clone()), o).await;
    // Both layouts spilled several runs.
    for m in &metrics {
        assert!(extra(m, "HashAggregate", "spill_runs") > 1);
    }
    let expected: Vec<String> = {
        let mut v: Vec<String> = (0..keys)
            .map(|k| {
                let valid: Vec<i64> = (k..n)
                    .step_by(usize::try_from(keys).unwrap())
                    .filter(|i| i % 7 != 0)
                    .collect();
                let first = if k % 7 == 0 {
                    "NULL".to_owned()
                } else {
                    k.to_string()
                };
                format!("{k}|{}|{}|{first}", valid[0], valid[valid.len() - 1])
            })
            .collect();
        v.sort();
        v
    };
    assert_eq!(out, expected);
    let agg = HashAggregate::with_row_mode(sp, rowtable::Mode::Fixed { limit: 300 }).unwrap();
    let on = exec_with(AggregateNode::new(scan(&s, b), agg), o, true)
        .await
        .unwrap();
    assert_eq!(rows(&on.0), expected);
    assert!(extra(&on.1, "HashAggregate", "spill_runs") > 1);
}

/// The row tables' final phase: several partition tasks in parallel over
/// tables with partial duplicates; when the merge cannot get memory (here
/// another consumer holds the pool), everything left goes to spill runs
/// and the disk phase returns the same rows, with nothing leaked.
#[tokio::test]
async fn row_final_phase_parallel_and_spill_fallback() {
    let (s, b) = grouped_input(200_000, 90_000);
    let expected = run_rows(single(&s, &b)).await;
    for hog in [false, true] {
        let (_m, scope) = spill_dir();
        let mut sp = spec(&s, vec![0], grouped_aggs(&s));
        sp.spill = Some(scope);
        let agg = HashAggregate::with_row_mode(sp, rowtable::Mode::Fixed { limit: 5_000 }).unwrap();
        let root = pool(1 << 30);
        let config = ExecConfig {
            agg_row_layout: true,
            ..ExecConfig::default()
        };
        let limit = 256 << 20;
        let q = QueryContext::new(
            QueryId(4),
            WorkloadClass::Batch,
            config,
            root.child("q", Some(limit)),
        );
        let executor: Arc<dyn Executor> = TokioExecutor::deterministic(4).unwrap();
        let ctx = |id, name| {
            let e = Arc::clone(&executor);
            TaskContext::new(Arc::clone(&q), e, SpillOwner::new(), OperatorId(id), name)
        };
        let sink_ctx = ctx(0, "agg");
        let g = Sink::global_state(&*agg, &sink_ctx).unwrap();
        for part in b.chunks(70) {
            let mut l = Sink::local_state(&*agg, &sink_ctx, &g).unwrap();
            for cols in part {
                let batch = Batch::new(RecordBatch::try_new(Arc::clone(&s), cols.clone()).unwrap());
                assert!(matches!(
                    agg.sink(batch, &mut l, &g).unwrap(),
                    crate::SinkStatus::NeedMoreInput
                ));
            }
            agg.combine(l, &g).unwrap();
        }
        // An unspillable consumer takes all but about 1 MiB: tasks start,
        // stall partway, and cannot go on.
        let hog_ctx = ctx(1, "hog");
        let mut held = hog_ctx.consumer().reservation();
        if hog {
            let free = limit - root.reserved();
            held.try_grow(free - (1 << 20)).unwrap();
        }
        agg.finalize(&g, &sink_ctx).await.unwrap();
        drop(held);
        let src = ctx(2, "src").with_dependency(Arc::clone(&g));
        let sg = Source::global_state(&*agg, &src).unwrap();
        let parallel = agg.max_parallelism(&sg);
        let mut sl = Source::local_state(&*agg, &src, &sg).unwrap();
        let mut out = Vec::new();
        while let Some(b) = agg.get_data(&mut sl, &sg).await.unwrap() {
            out.push(b);
        }
        assert_eq!(rows(&out), expected, "hog {hog}");
        if hog {
            assert!(agg.metrics().spill_runs > 0);
            assert_eq!(parallel, 1);
        } else {
            assert_eq!(agg.metrics().spill_runs, 0);
            assert!(parallel > 1, "{parallel}");
        }
        assert_eq!(agg.metrics().groups, 90_000);
        drop((sl, sg, src, g, sink_ctx));
        assert_eq!(root.reserved(), 0);
    }
}

/// Every key type with a row encoding through the row tables (fixed
/// indexes that abandon, several drivers, up to three keys at a time)
/// returns the columnar tables' rows, NULL keys included: fixed widths, and
/// strings and binaries of every carrier, empty, inline, 12 and 13 bytes
/// and long (heap), binaries that are not UTF-8.
#[tokio::test]
async fn row_layout_key_types_equal_default() {
    use arrow_array::{
        BooleanArray, Decimal64Array, Float32Array, TimestampMicrosecondArray, UInt16Array,
    };
    let n = 5_000usize;
    let i =
        |m: usize| (0..n).map(move |r| (r % 11 != 0).then(|| i64::try_from(r * 7919 % m).unwrap()));
    let mut cols: Vec<(&str, ArrayRef)> = vec![
        (
            "d128",
            Arc::new(
                Decimal128Array::from_iter(i(300).map(|v| v.map(|v| i128::from(v) - 150)))
                    .with_precision_and_scale(20, 3)
                    .unwrap(),
            ),
        ),
        (
            "d64",
            Arc::new(
                Decimal64Array::from_iter(i(70))
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
        ),
        (
            "date",
            Arc::new(Date32Array::from_iter(
                i(400).map(|v| v.map(|v| i32::try_from(v - 200).unwrap())),
            )),
        ),
        (
            "ts",
            Arc::new(
                TimestampMicrosecondArray::from_iter(i(90).map(|v| v.map(|v| v * 1_000)))
                    .with_timezone("UTC"),
            ),
        ),
        (
            "f32",
            Arc::new(Float32Array::from_iter(
                i(9).map(|v| v.map(|v| f32::from(i16::try_from(v).unwrap()) - 4.0)),
            )),
        ),
        (
            "u16",
            Arc::new(UInt16Array::from_iter(
                i(50).map(|v| v.map(|v| u16::try_from(v).unwrap())),
            )),
        ),
        (
            "b",
            Arc::new(BooleanArray::from_iter(i(2).map(|v| v.map(|v| v == 1)))),
        ),
    ];
    // Strings and binaries of every view shape: empty, inline (up to 12
    // bytes), exactly 12 and 13 bytes, and long; binaries not UTF-8.
    let text = |m: usize| {
        i(m).map(|v| {
            v.map(|v| match v % 5 {
                0 => String::new(),
                1 => format!("{v}"),
                2 => format!("{v:0>12}"),
                3 => format!("{v:0>13}"),
                _ => format!("a-long-string-key-{v:0>30}"),
            })
        })
    };
    let bytes = |m: usize| {
        text(m).map(|v| {
            v.map(|v| {
                let mut b = vec![0xFF_u8, 0x00];
                b.extend(v.into_bytes());
                b
            })
        })
    };
    let strs = |m| text(m).collect::<Vec<_>>();
    let bins = |m| bytes(m).collect::<Vec<_>>();
    cols.extend([
        (
            "utf8",
            Arc::new(arrow_array::StringArray::from(strs(200))) as ArrayRef,
        ),
        (
            "large_utf8",
            Arc::new(arrow_array::LargeStringArray::from(strs(60))),
        ),
        ("utf8_view", Arc::new(StringViewArray::from(strs(500)))),
        (
            "binary",
            Arc::new(arrow_array::BinaryArray::from_iter(bins(80))),
        ),
        (
            "binary_view",
            Arc::new(arrow_array::BinaryViewArray::from_iter(bins(300))),
        ),
        (
            "large_binary",
            Arc::new(arrow_array::LargeBinaryArray::from_iter(bins(40))),
        ),
        ("x", Arc::new(Int64Array::from_iter(i(1_000)))),
    ]);
    let s: SchemaRef = Arc::new(Schema::new(
        cols.iter()
            .map(|(n, c)| Field::new(*n, c.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ));
    let arrays: Vec<ArrayRef> = cols.into_iter().map(|(_, c)| c).collect();
    let batches: Vec<Vec<ArrayRef>> = (0..n)
        .step_by(700)
        .map(|at| {
            arrays
                .iter()
                .map(|a| a.slice(at, 700.min(n - at)))
                .collect()
        })
        .collect();
    let x = s.fields().len() - 1;
    let aggs = || {
        vec![
            AggregateExpr::count_star("c"),
            agg(Sum, x, &s),
            agg(Min, 0, &s),
            agg(Max, 2, &s),
        ]
    };
    for keys in [
        vec![0],
        vec![1],
        vec![2],
        vec![3],
        vec![4],
        vec![5],
        vec![6],
        vec![0, 3],
        vec![2, 6],
        vec![7],
        vec![8],
        vec![9],
        vec![10],
        vec![11],
        vec![12],
        vec![9, 0],
        vec![7, 6, 11],
        vec![8, 12, 2],
    ] {
        let sp = spec(&s, keys.clone(), aggs());
        let o = Opts {
            both_layouts: false,
            ..Opts::default()
        };
        let want = rows(
            &exec(node(scan(&s, batches.clone()), sp.clone()), o)
                .await
                .unwrap()
                .0,
        );
        assert!(!want.is_empty());
        for mode in [
            rowtable::Mode::Grow,
            rowtable::Mode::Fixed { limit: 7 },
            rowtable::Mode::Append,
        ] {
            let agg = HashAggregate::with_row_mode(sp.clone(), mode).unwrap();
            let (got, m) = exec_with(AggregateNode::new(scan(&s, batches.clone()), agg), o, true)
                .await
                .unwrap();
            assert_eq!(rows(&got), want, "keys {keys:?} {mode:?}");
            // The row tables ran (no fallback to the columnar tables).
            let layout = m
                .operators
                .values()
                .flat_map(|e| e.extra.iter())
                .any(|(n, v)| {
                    *n == "table_layout" && *v == crate::metrics::Metric::Text("rows".to_owned())
                });
            assert!(layout, "keys {keys:?}: not on the row tables");
        }
    }
}

/// A spec whose key field is non-nullable over batches that hold NULL keys
/// (their own schema is nullable): both layouts fail the same way, when
/// the output batch is checked against the declared schema (the row
/// tables give every key a validity bit, so they group the NULLs first,
/// as the columnar tables do).
#[tokio::test]
async fn row_layout_nullable_batches_under_a_non_nullable_spec() {
    let nullable = schema(vec![("k", DataType::Int64), ("x", DataType::Int64)]);
    let strict: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("x", DataType::Int64, true),
    ]));
    let b = vec![
        vec![
            i64s(&[Some(1), None, Some(1), None]),
            i64s(&[Some(1), Some(2), Some(3), Some(4)]),
        ],
        vec![i64s(&[None, Some(2)]), i64s(&[Some(5), Some(6)])],
    ];
    let plan = node(
        scan(&nullable, b),
        spec(
            &strict,
            vec![0],
            vec![AggregateExpr::count_star("c"), agg(Sum, 1, &strict)],
        ),
    );
    // `exec` runs both layouts and requires the same error.
    let e = exec(plan, Opts::default()).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    assert!(
        e.to_string()
            .contains("declared as non-nullable but contains null"),
        "{e}"
    );
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: tight_memory_cases(),
        rng_seed: proptest::test_runner::RngSeed::Fixed(20_261_020),
        ..ProptestConfig::default()
    })]
    /// At 768 KiB the row tables finish wherever the columnar tables do,
    /// with partial duplicates from tiny fixed indexes (one row per input
    /// row at `limit` 1): the final phase frees each driver partition once
    /// merged and writes finished partitions to the output file instead of
    /// spilling them, and a batch's scatter uses the idle spill headroom.
    /// This fixed seed found the final-phase gap (2x memory) and then the
    /// scatter's transient bytes.
    #[test]
    fn row_layout_tight_memory(
        data in proptest::collection::vec(
            (
                proptest::option::of(any::<i32>()),
                proptest::option::of(any::<u8>()),
                proptest::option::of(any::<bool>()),
                proptest::option::of(-1_000_000i64..1_000_000),
                proptest::option::of(-10_000_000i64..10_000_000),
                proptest::option::of(any::<u16>()),
            ),
            0..4000,
        ),
        keys in 1i32..3000,
        nkeys in 1usize..=6,
        batch in 1usize..900,
        slots in 1usize..4,
        mode in 0u8..3,
        limit in 1usize..64,
        two in any::<bool>(),
        // As in `row_layout_equals_default`, so its saved seeds (one of them
        // is this gap, found at 768 KiB) replay here first.
        _tight in any::<bool>(),
    ) {
        row_case(&data, keys, nkeys, batch, slots, mode, limit, two, Some(768 << 10))?;
    }
}

/// A spill request during the row tables' parallel merge writes the
/// finished tasks' groups to the output file (final, never merged again)
/// and the merge goes on; the rest of the result follows them to the file.
/// The rows are unchanged and nothing leaks.
#[tokio::test]
async fn row_merge_spills_on_request() {
    let (s, b) = grouped_input(200_000, 90_000);
    let expected = run_rows(single(&s, &b)).await;
    let (_m, scope) = spill_dir();
    let mut sp = spec(&s, vec![0], grouped_aggs(&s));
    sp.spill = Some(scope);
    let agg = HashAggregate::with_row_mode(sp, rowtable::Mode::Fixed { limit: 5_000 }).unwrap();
    let root = pool(1 << 30);
    let config = ExecConfig {
        agg_row_layout: true,
        ..ExecConfig::default()
    };
    let q = QueryContext::new(
        QueryId(5),
        WorkloadClass::Batch,
        config,
        root.child("q", None),
    );
    // One worker: the merge runs one task at a time.
    let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
    let ctx = |id, name| {
        let e = Arc::clone(&executor);
        TaskContext::new(Arc::clone(&q), e, SpillOwner::new(), OperatorId(id), name)
    };
    let sink_ctx = ctx(0, "agg");
    let g = Sink::global_state(&*agg, &sink_ctx).unwrap();
    for part in b.chunks(70) {
        let mut l = Sink::local_state(&*agg, &sink_ctx, &g).unwrap();
        for cols in part {
            let batch = Batch::new(RecordBatch::try_new(Arc::clone(&s), cols.clone()).unwrap());
            agg.sink(batch, &mut l, &g).unwrap();
        }
        agg.combine(l, &g).unwrap();
    }
    let request = async {
        while !super::row_merge_running(&g) {
            tokio::task::yield_now().await;
        }
        // Asked, not freed yet: the merge frees between its tasks.
        assert_eq!(agg.spill_global(&g, 1, &sink_ctx).await.unwrap(), 0);
    };
    let (done, ()) = tokio::join!(agg.finalize(&g, &sink_ctx), request);
    done.unwrap();
    let m = agg.metrics();
    assert_eq!((m.spill_runs, m.groups), (0, 90_000));
    assert!(m.spill_files > 0);
    assert!(sink_metric(&agg, "row_final_flushes") > 0);
    assert_eq!(sink_metric(&agg, "row_final_spills"), 0);
    let src = ctx(1, "src").with_dependency(Arc::clone(&g));
    let sg = Source::global_state(&*agg, &src).unwrap();
    let mut sl = Source::local_state(&*agg, &src, &sg).unwrap();
    let mut out = Vec::new();
    while let Some(b) = agg.get_data(&mut sl, &sg).await.unwrap() {
        out.push(b);
    }
    assert_eq!(rows(&out), expected);
    drop((sl, sg, src, g, sink_ctx));
    assert_eq!(root.reserved(), 0);
}

/// Both table layouts count their index reallocations, and, in a detailed
/// profile only, their probe lengths.
#[tokio::test]
async fn resizes_and_probe_lengths_on_both_layouts() {
    let (s, b) = grouped_input(200_000, 90_000);
    let sp = spec(&s, vec![0], grouped_aggs(&s));
    for rows in [false, true] {
        let o = Opts {
            both_layouts: false,
            detailed: true,
            ..Opts::default()
        };
        let (_, m) = exec_with(node(scan(&s, b.clone()), sp.clone()), o, rows)
            .await
            .unwrap();
        assert!(
            extra(&m, "HashAggregate", "hash_resizes") > 0,
            "rows {rows}"
        );
        let lookups = extra(&m, "HashAggregate", "probe_lookups");
        assert!(lookups > 0, "rows {rows}");
        assert!(
            extra(&m, "HashAggregate", "probe_key_compares") > 0,
            "rows {rows}"
        );
        assert!(
            extra(&m, "HashAggregate", "max_probe_length") >= 1,
            "rows {rows}"
        );
        let o = Opts {
            both_layouts: false,
            ..Opts::default()
        };
        let (_, m) = exec_with(node(scan(&s, b.clone()), sp.clone()), o, rows)
            .await
            .unwrap();
        assert!(
            extra(&m, "HashAggregate", "hash_resizes") > 0,
            "rows {rows}"
        );
        assert_eq!(
            extra(&m, "HashAggregate", "probe_lookups"),
            0,
            "rows {rows}"
        );
    }
}

/// `EXPLAIN ANALYZE` counters of the row tables: abandons of full fixed
/// indexes, final-phase tasks, the groups and the layout; the default
/// tables report none of the row counters.
#[tokio::test]
async fn row_layout_metrics() {
    let (s, b) = grouped_input(200_000, 90_000);
    let sp = spec(&s, vec![0], grouped_aggs(&s));
    let o = Opts {
        both_layouts: false,
        ..Opts::default()
    };
    let agg =
        HashAggregate::with_row_mode(sp.clone(), rowtable::Mode::Fixed { limit: 5_000 }).unwrap();
    let (_, m) = exec_with(AggregateNode::new(scan(&s, b.clone()), agg), o, true)
        .await
        .unwrap();
    assert!(extra(&m, "HashAggregate", "row_abandons") > 0);
    assert!(extra(&m, "HashAggregate", "row_final_tasks") > 1);
    assert_eq!(extra(&m, "HashAggregate", "row_final_spills"), 0);
    assert_eq!(extra(&m, "HashAggregate", "row_final_flushes"), 0);
    assert_eq!(extra(&m, "HashAggregate", "groups"), 90_000);
    assert_eq!(extra(&m, "HashAggregate", "hash_entries"), 90_000);
    let layout = m
        .operators
        .values()
        .flat_map(|e| e.extra.iter())
        .any(|(n, v)| {
            *n == "table_layout" && *v == crate::metrics::Metric::Text("rows".to_owned())
        });
    assert!(layout);
    let (_, m) = exec_with(node(scan(&s, b), sp), o, false).await.unwrap();
    let names: Vec<_> = m
        .operators
        .values()
        .flat_map(|e| e.extra.iter())
        .map(|(n, _)| n.to_string())
        .collect();
    assert!(
        !names
            .iter()
            .any(|n| n.starts_with("row_") || n == "table_layout"),
        "{names:?}"
    );
}

/// What [`by_hand`] observed.
struct ByHand {
    rows: Vec<String>,
    /// Bytes the query pool held once every driver combined.
    combined: usize,
    /// The final phase's peak bytes.
    peak: usize,
}

/// Runs `agg` (row layout on) by hand on one worker: `drivers` drivers
/// sink `b` in turn and combine, then the aggregate finalizes and is read.
/// Checks that nothing leaks.
async fn by_hand(
    agg: &Arc<HashAggregate>,
    s: &SchemaRef,
    b: &[Vec<ArrayRef>],
    drivers: usize,
) -> ByHand {
    let root = pool(1 << 30);
    let config = ExecConfig {
        agg_row_layout: true,
        ..ExecConfig::default()
    };
    let q = QueryContext::new(
        QueryId(6),
        WorkloadClass::Batch,
        config,
        root.child("q", None),
    );
    // One worker: the merge runs one task at a time.
    let executor: Arc<dyn Executor> = TokioExecutor::deterministic(1).unwrap();
    let ctx = |id, name| {
        let e = Arc::clone(&executor);
        TaskContext::new(Arc::clone(&q), e, SpillOwner::new(), OperatorId(id), name)
    };
    let sink_ctx = ctx(0, "agg");
    let g = Sink::global_state(&**agg, &sink_ctx).unwrap();
    for part in b.chunks(b.len().div_ceil(drivers)) {
        let mut l = Sink::local_state(&**agg, &sink_ctx, &g).unwrap();
        for cols in part {
            let batch = Batch::new(RecordBatch::try_new(Arc::clone(s), cols.clone()).unwrap());
            agg.sink(batch, &mut l, &g).unwrap();
        }
        agg.combine(l, &g).unwrap();
    }
    let combined = root.reserved();
    // The final phase runs on a consumer of its own (the tables are charged
    // to it), so its peak is the final phase's, not the drivers' (who widen
    // their tables when they combine).
    let final_ctx = ctx(0, "agg");
    agg.finalize(&g, &final_ctx).await.unwrap();
    let peak = final_ctx.consumer().consumer().peak();
    let src = ctx(1, "src").with_dependency(Arc::clone(&g));
    let sg = Source::global_state(&**agg, &src).unwrap();
    let mut sl = Source::local_state(&**agg, &src, &sg).unwrap();
    let mut out = Vec::new();
    while let Some(b) = agg.get_data(&mut sl, &sg).await.unwrap() {
        out.push(b);
    }
    drop((sl, sg, src, g, sink_ctx, final_ctx));
    assert_eq!(root.reserved(), 0);
    ByHand {
        rows: rows(&out),
        combined,
        peak,
    }
}

/// When the final phase runs short of memory it first writes the finished
/// tasks' groups to the output file; when it runs short again with no
/// task finished, the rest spills and the disk phase appends to that same
/// file. Every group comes out once, with its values.
#[tokio::test]
async fn row_merge_flushes_then_spills() {
    let (s, b) = grouped_input(200_000, 90_000);
    let expected = run_rows(single(&s, &b)).await;
    let (_m, scope) = spill_dir();
    let mut sp = spec(&s, vec![0], grouped_aggs(&s));
    sp.spill = Some(scope);
    // After the first task, no task gets memory: flush it, then spill.
    let agg =
        HashAggregate::with_final_stall(sp, rowtable::Mode::Fixed { limit: 5_000 }, 1).unwrap();
    let got = by_hand(&agg, &s, &b, 3).await;
    assert_eq!(got.rows.len(), 90_000);
    assert_eq!(got.rows, expected);
    assert!(sink_metric(&agg, "row_final_tasks") > 1);
    assert_eq!(sink_metric(&agg, "row_final_flushes"), 1);
    assert_eq!(sink_metric(&agg, "row_final_spills"), 1);
    let m = agg.metrics();
    assert!(m.spill_runs > 0);
    assert_eq!(m.groups, 90_000);
}

/// The final phase frees each driver partition once merged: with every key
/// distinct (the merged tables are as large as the inputs), the peak stays
/// well below holding the inputs and the merged tables together.
#[tokio::test]
async fn row_merge_frees_inputs_as_it_merges() {
    let (s, b) = grouped_input(120_000, 120_000);
    let expected = run_rows(single(&s, &b)).await;
    let sp = spec(&s, vec![0], grouped_aggs(&s));
    let agg = HashAggregate::with_row_mode(sp, rowtable::Mode::Append).unwrap();
    let got = by_hand(&agg, &s, &b, 3).await;
    assert_eq!(got.rows, expected);
    assert!(sink_metric(&agg, "row_final_tasks") > 2);
    assert!(
        got.peak < got.combined * 3 / 2,
        "peak {} vs {} combined",
        got.peak,
        got.combined
    );
}

/// The plan of the unread-result tests: `sum(x) group by k` over `groups`
/// keys, read into a breaker that cannot spill (as a join build or a
/// result buffer would be in its place).
fn unread_plan(groups: i64, spill: Option<SpillScope>) -> Arc<dyn ExecNode> {
    unread_plan_with(groups, spill, None)
}

/// [`unread_plan`] whose result's spill waits at `gate`.
fn unread_plan_with(
    groups: i64,
    spill: Option<SpillScope>,
    gate: Option<Arc<crate::testing::BlockingGate>>,
) -> Arc<dyn ExecNode> {
    let s = schema(vec![("k", DataType::Int64), ("x", DataType::Int64)]);
    let batches = (0..groups)
        .step_by(1024)
        .map(|start| {
            let keys: Vec<Option<i64>> = (start..(start + 1024).min(groups)).map(Some).collect();
            vec![i64s(&keys), i64s(&keys)]
        })
        .collect();
    let mut sp = spec(&s, vec![0], vec![agg(Sum, 1, &s)]);
    sp.spill = spill;
    let agg = match gate {
        Some(gate) => HashAggregate::with_output_gate(sp, gate).unwrap(),
        None => HashAggregate::new(sp).unwrap(),
    };
    Arc::new(crate::testing::Break(
        AggregateNode::new(scan(&s, batches), agg),
        crate::testing::Buffer::new(false, None),
        false,
    ))
}

/// An in-memory result is spillable until read. Its reader feeds a
/// breaker that cannot spill; the memory that breaker needs comes from
/// spilling the result's unread chunks. Before, the result held its memory
/// unspillable until the query ended: a limit at which the aggregate fits
/// failed or passed depending on how much of the result had to coexist
/// with the reader's state (1.8 MB failed while 1.6 MB, where the aggregate
/// spilled while building, passed). Now every limit in the range completes,
/// with each layout and with one or several readers.
#[tokio::test]
async fn an_unread_result_spills_for_its_reader() {
    let (_d, scope) = spill_dir();
    let groups = 20_000;
    let free = Opts {
        slots: 1,
        ..Opts::default()
    };
    let (want, _) = exec_with(unread_plan(groups, None), free, false)
        .await
        .unwrap();
    let want = rows(&want);
    assert_eq!(want.len(), 20_000);
    for slots in [1, 3] {
        for limit in (5..=10).map(|fifth| fifth * 200_000) {
            for layout in [false, true] {
                let o = Opts {
                    slots,
                    limit,
                    ..Opts::default()
                };
                let plan = unread_plan(groups, Some(scope.clone()));
                let r = exec_with(plan, o, layout).await;
                let (got, m) = r.unwrap_or_else(|e| panic!("{slots} slots, {limit} B: {e}"));
                assert_eq!(rows(&got), want, "{slots} slots, {limit} B");
                if slots == 1 && limit == 1_800_000 && !layout {
                    // The aggregate finished in memory; only its result
                    // was written to disk, by its own consumer.
                    assert_eq!(extra(&m, "HashAggregate", "spill_runs"), 0);
                    let written: u64 = m
                        .operators
                        .values()
                        .filter(|e| e.name == "HashAggregate")
                        .map(|e| e.metrics.spill_write_bytes)
                        .sum();
                    assert!(written > 0, "{m:?}");
                }
            }
        }
    }
}

/// A query cancelled while its aggregate's result is being spilled ends
/// only once the spill has let go of the result's memory and its files.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cancel_during_a_result_spill_releases_everything() {
    let (spill, scope) = spill_dir();
    let gate = crate::testing::BlockingGate::new();
    let plan = unread_plan_with(20_000, Some(scope), Some(Arc::clone(&gate)));
    // Threads of their own: the spill blocks at the gate on the blocking
    // pool while the query is cancelled.
    let executor = TokioExecutor::new(&crate::ExecutorConfig {
        compute_threads: 1,
        io_threads: 1,
        blocking_threads: 1,
        io_blocking_threads: 2,
    })
    .unwrap();
    let runner = QueryRunner::new(executor);
    let root = pool(1_800_000);
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig {
            seed: Some(7),
            agg_row_layout: false,
            ..ExecConfig::default()
        },
        root.child("q", Some(1_800_000)),
    );
    let mut stream = runner.execute(plan, Arc::clone(&q)).await.unwrap();
    let drained = tokio::spawn(async move {
        let mut rows = 0;
        while let Some(b) = stream.next().await {
            rows += b.map_or(0, |b| b.num_rows());
        }
        rows
    });
    while !gate.reached() {
        tokio::task::yield_now().await;
    }
    q.cancel();
    // What the pool holds the moment the query reports it has ended.
    let (fq, froot) = (Arc::clone(&q), Arc::clone(&root));
    let at_end = tokio::spawn(async move {
        fq.finished().await;
        froot.reserved()
    });
    // The gate stays shut while the query winds down: the query must not
    // end while the spill job still holds the result's memory. (The wait
    // is bounded by count; the query ends here only if it does not wait
    // for the spill.)
    for _ in 0..200 {
        if at_end.is_finished() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    gate.open();
    // Nothing the spill held outlives the query.
    assert_eq!(at_end.await.unwrap(), 0, "memory left when the query ended");
    assert_eq!(root.reserved(), 0, "memory left after the query");
    assert_eq!(spill.manager.used_bytes(), 0, "spill files left");
    assert!(drained.await.unwrap() < 20_000);
    assert_eq!(
        q.error().map(|e| e.error_class()),
        Some(ErrorClass::OperationCanceled)
    );
}

/// A spill of an aggregate's result that fails (its spill directory is
/// full) fails the query with that error; nothing is left behind.
#[tokio::test]
async fn a_failed_result_spill_fails_the_query() {
    let (spill, scope) = spill_dir_capped(4096);
    let o = Opts {
        slots: 1,
        limit: 1_800_000,
        ..Opts::default()
    };
    let r = exec_with(unread_plan(20_000, Some(scope)), o, false).await;
    let e = r
        .err()
        .unwrap_or_else(|| unreachable!("the result could not spill"));
    assert_eq!(e.error_class(), ErrorClass::KrishivSpillCapExceeded, "{e}");
    assert_eq!(spill.manager.used_bytes(), 0, "spill files left");
}

/// A thread-local table widens to its radix partitions once large, and the
/// final phase widens the tables that stayed narrow before it merges, so
/// a driver with many groups and one with few merge to the same result.
/// When the final phase cannot reserve that widening, every driver table
/// spills and the disk phase merges them.
#[tokio::test]
async fn narrow_and_wide_tables_merge() {
    let (s, mut b) = grouped_input(1_000, 1_000);
    let (_, small) = grouped_input(40, 10);
    b.extend(small);
    let expected = run_rows(single(&s, &b)).await;
    let mode = rowtable::Mode::Fixed { limit: 5_000 };
    // One driver per batch: 1,000 groups (widens at 64) and 10 (narrow).
    let sp = spec(&s, vec![0], grouped_aggs(&s));
    let agg = HashAggregate::with_row_mode(sp, mode).unwrap();
    let got = by_hand(&agg, &s, &b, 2).await;
    assert_eq!(got.rows, expected);
    assert_eq!(sink_metric(&agg, "row_final_spills"), 0);
    // The widening of the narrow table is refused: spill and merge on disk.
    let (_m, scope) = spill_dir();
    let mut sp = spec(&s, vec![0], grouped_aggs(&s));
    sp.spill = Some(scope);
    let agg = HashAggregate::with_final_stall(sp, mode, 0).unwrap();
    let got = by_hand(&agg, &s, &b, 2).await;
    assert_eq!(got.rows, expected);
    assert_eq!(sink_metric(&agg, "row_final_spills"), 1);
    assert!(agg.metrics().spill_runs >= 2);
}

/// Without a spill directory, a narrow table that cannot reserve its
/// widening stays narrow (waiting for that memory could wait forever: at
/// a tight limit nothing else frees it), and one task with one partition
/// merges every partition of every table instead of one task per range.
#[tokio::test]
async fn a_refused_widening_merges_in_one_task() {
    let (s, mut b) = grouped_input(80_000, 70_000);
    let (_, small) = grouped_input(40, 10);
    b.extend(small);
    let expected = run_rows(single(&s, &b)).await;
    // A fixed index (radix partitions at one thread too) that never fills.
    let mode = rowtable::Mode::Fixed { limit: 100_000 };
    // Widened: the 70,000 groups merge in several tasks.
    let agg = HashAggregate::with_row_mode(spec(&s, vec![0], grouped_aggs(&s)), mode).unwrap();
    let got = by_hand(&agg, &s, &b, 81).await;
    assert_eq!(got.rows, expected);
    assert!(sink_metric(&agg, "row_final_tasks") > 1);
    // Refused (and no spill directory): one task. (A stall after
    // `usize::MAX` tasks refuses only the widening.)
    let sp = spec(&s, vec![0], grouped_aggs(&s));
    let agg = HashAggregate::with_final_stall(sp, mode, usize::MAX).unwrap();
    let got = by_hand(&agg, &s, &b, 81).await;
    assert_eq!(got.rows, expected);
    assert_eq!(sink_metric(&agg, "row_final_tasks"), 1);
    assert_eq!(sink_metric(&agg, "row_final_spills"), 0);
}

// ---- the moment aggregates ----

const MOMENTS: [AggregateFunction; 9] = [
    AggregateFunction::StddevSamp,
    AggregateFunction::StddevPop,
    AggregateFunction::VarSamp,
    AggregateFunction::VarPop,
    AggregateFunction::Skewness,
    AggregateFunction::Kurtosis,
    AggregateFunction::Corr,
    AggregateFunction::CovarSamp,
    AggregateFunction::CovarPop,
];

/// Every moment aggregate over `x` (column 1) and, for the pair
/// statistics, `y` (column 2).
fn moment_aggs() -> Vec<AggregateExpr> {
    MOMENTS
        .iter()
        .map(|&f| AggregateExpr {
            function: f,
            args: if super::arity(f) == 2 {
                vec![1, 2]
            } else {
                vec![1]
            },
            input: AggInput::Raw,
            arg_type: DataType::Float64,
            mask: None,
            name: format!("{f:?}"),
        })
        .collect()
}

fn f64s(v: Vec<Option<f64>>) -> ArrayRef {
    Arc::new(Float64Array::from(v))
}

/// `n` rows `k = i % keys`, a `double` x (NULL every 7th) and a `double` y
/// (NULL every 11th), in batches of 1000.
fn moment_input(n: i64, keys: i64) -> (SchemaRef, Vec<Vec<ArrayRef>>) {
    let s = schema(vec![
        ("k", DataType::Int64),
        ("x", DataType::Float64),
        ("y", DataType::Float64),
    ]);
    #[expect(clippy::cast_precision_loss, reason = "small test values")]
    let batches = (0..n)
        .step_by(1000)
        .map(|start| {
            let r = start..(start + 1000).min(n);
            let k: Vec<_> = r.clone().map(|i| Some(i % keys)).collect();
            let x = r
                .clone()
                .map(|i| (i % 7 != 0).then(|| (i * 37 % 101) as f64 / 3.0));
            let y = r.map(|i| (i % 11 != 0).then(|| (i * 53 % 97) as f64 / 7.0 - (i % 5) as f64));
            vec![i64s(&k), f64s(x.collect()), f64s(y.collect())]
        })
        .collect();
    (s, batches)
}

type Moments = BTreeMap<String, Vec<Option<f64>>>;

/// The output of a moment aggregation by key (`""` without keys).
fn moment_rows(batches: &[Batch], keyed: bool) -> Moments {
    rows(batches)
        .iter()
        .map(|r| {
            let mut cells = r.split('|');
            let key = if keyed { cells.next().unwrap() } else { "" };
            let values = cells
                .map(|c| (c != "NULL").then(|| c.parse::<f64>().unwrap()))
                .collect();
            (key.to_owned(), values)
        })
        .collect()
}

/// The aggregates of every group folded in one pass by one accumulator
/// each: what a single driver computes.
fn moment_oracle(b: &[Vec<ArrayRef>], keys: i64, keyed: bool, legacy: bool) -> Moments {
    let groups = if keyed {
        usize::try_from(keys).unwrap()
    } else {
        1
    };
    let mut accs: Vec<Accumulator> = moment_aggs()
        .iter()
        .map(|a| {
            let mut acc = Accumulator::new(a, true)
                .unwrap()
                .legacy_statistical(legacy);
            acc.resize(groups);
            acc
        })
        .collect();
    for cols in b {
        let ids: Vec<u32> = cols[0]
            .as_primitive::<Int64Type>()
            .values()
            .iter()
            .map(|&k| if keyed { u32::try_from(k).unwrap() } else { 0 })
            .collect();
        for a in &mut accs {
            a.update(&ids, cols, &Weights::Const(1)).unwrap();
        }
    }
    let ids: Vec<u32> = (0..u32::try_from(groups).unwrap()).collect();
    let columns: Vec<ArrayRef> = accs.iter().map(|a| a.evaluate(&ids).unwrap()).collect();
    ids.iter()
        .map(|&g| {
            let values = columns
                .iter()
                .map(|c| {
                    let c = c.as_primitive::<Float64Type>();
                    (!c.is_null(g as usize)).then(|| c.value(g as usize))
                })
                .collect();
            (if keyed { g.to_string() } else { String::new() }, values)
        })
        .collect()
}

/// The values agree to `1e-9` (merging partial states reassociates the
/// arithmetic), NULLs and NaNs exactly.
fn assert_moments_close(got: &Moments, want: &Moments, what: &str) {
    assert_eq!(
        got.keys().collect::<Vec<_>>(),
        want.keys().collect::<Vec<_>>(),
        "{what}"
    );
    for (key, want) in want {
        for (i, (g, w)) in got[key].iter().zip(want).enumerate() {
            let ok = match (g, w) {
                (None, None) => true,
                (Some(g), Some(w)) if g.is_nan() || w.is_nan() => g.is_nan() && w.is_nan(),
                (Some(g), Some(w)) => (g - w).abs() <= 1e-9 * g.abs().max(w.abs()).max(1.0),
                _ => false,
            };
            assert!(ok, "{what}: key {key:?} column {i}: {g:?} vs {w:?}");
        }
    }
}

/// Runs `plan` on each layout and checks both against `want`.
async fn check_both_layouts(
    make: impl Fn() -> Arc<dyn ExecNode>,
    o: Opts,
    keyed: bool,
    want: &Moments,
) {
    for rows_layout in [false, true] {
        let (out, _) = exec_with(make(), o, rows_layout).await.unwrap();
        assert_moments_close(
            &moment_rows(&out, keyed),
            want,
            &format!("row layout {rows_layout}"),
        );
    }
}

/// Grouped and global moment aggregates, on both table layouts, against a
/// single-pass reference.
#[tokio::test]
async fn moments_grouped_and_global() {
    let (s, b) = moment_input(8_000, 13);
    let o = Opts {
        both_layouts: false,
        ..Opts::default()
    };
    let grouped = || node(scan(&s, b.clone()), spec(&s, vec![0], moment_aggs()));
    check_both_layouts(grouped, o, true, &moment_oracle(&b, 13, true, false)).await;
    let global = || node(scan(&s, b.clone()), spec(&s, vec![], moment_aggs()));
    check_both_layouts(global, o, false, &moment_oracle(&b, 13, false, false)).await;
    // A global aggregate of no rows is one row of NULLs; a grouped one has none.
    let empty = |keys: Vec<usize>| node(scan(&s, vec![]), spec(&s, keys, moment_aggs()));
    let (out, _) = exec(empty(vec![]), o).await.unwrap();
    assert_eq!(rows(&out), ["NULL|NULL|NULL|NULL|NULL|NULL|NULL|NULL|NULL"]);
    assert!(rows(&exec(empty(vec![0]), o).await.unwrap().0).is_empty());
}

/// `spark.sql.legacy.statisticalAggregate` reaches the accumulators of
/// both layouts and of the final phase of a two-phase aggregation: groups
/// of one row give NULL, or NaN with the flag.
#[tokio::test]
async fn moments_legacy_flag_reaches_every_accumulator() {
    // Every group has one row, so the sample statistics divide by zero.
    let (s, b) = moment_input(40, 40);
    let o = Opts {
        both_layouts: false,
        ..Opts::default()
    };
    for legacy in [false, true] {
        let mut sp = spec(&s, vec![0], moment_aggs());
        sp.legacy_statistical = legacy;
        let want = moment_oracle(&b, 40, true, legacy);
        // The oracle itself shows the flag: some groups' cells are NaN or NULL.
        let nan_cells = want
            .values()
            .flatten()
            .filter(|v| v.is_some_and(f64::is_nan))
            .count();
        assert_eq!(nan_cells > 0, legacy);
        let single = || node(scan(&s, b.clone()), sp.clone());
        check_both_layouts(single, o, true, &want).await;
        let phased = || {
            two_phase_of(
                scan(&s, b.clone()),
                sp.clone(),
                BypassPolicy::default(),
                None,
            )
            .0
        };
        check_both_layouts(phased, o, true, &want).await;
    }
}

type Plan<'a> = Box<dyn Fn(&[Vec<ArrayRef>]) -> Arc<dyn ExecNode> + 'a>;

/// `corr` of several rows over a constant column: `DIVIDE_BY_ZERO` with ANSI
/// and NULL without, on both layouts and across a partial and a final
/// phase, whatever the legacy flag says; the other groups are unaffected
/// when none is constant.
#[tokio::test]
async fn corr_of_a_constant_column_follows_ansi() {
    let s = schema(vec![
        ("k", DataType::Int64),
        ("x", DataType::Float64),
        ("y", DataType::Float64),
    ]);
    let batch = |k: &[i64], x: &[f64], y: &[f64]| {
        let some = |v: &[f64]| f64s(v.iter().map(|&v| Some(v)).collect());
        let k: Vec<_> = k.iter().map(|&k| Some(k)).collect();
        vec![i64s(&k), some(x), some(y)]
    };
    let corr = || vec![moment_aggs().remove(6)];
    let o = Opts {
        both_layouts: false,
        ..Opts::default()
    };
    let varying = batch(
        &[1, 1, 2, 2, 2],
        &[1.0, 2.0, 1.0, 2.0, 4.0],
        &[3.0, 1.0, 2.0, 5.0, 6.0],
    );
    let constant = batch(
        &[1, 1, 2, 2, 2],
        &[1.0, 2.0, 7.0, 7.0, 7.0],
        &[3.0, 1.0, 2.0, 5.0, 6.0],
    );
    for legacy in [false, true] {
        for ansi in [false, true] {
            let mut sp = spec(&s, vec![0], corr());
            sp.ansi = ansi;
            sp.legacy_statistical = legacy;
            let plans: [Plan<'_>; 2] = [
                Box::new(|b| node(scan(&s, b.to_vec()), sp.clone())),
                Box::new(|b| {
                    two_phase_of(
                        scan(&s, b.to_vec()),
                        sp.clone(),
                        BypassPolicy::default(),
                        None,
                    )
                    .0
                }),
            ];
            for plan in &plans {
                for rows_layout in [false, true] {
                    let ok = exec_with(plan(std::slice::from_ref(&varying)), o, rows_layout)
                        .await
                        .unwrap()
                        .0;
                    assert!(moment_rows(&ok, true).values().all(|v| v[0].is_some()));
                    let got =
                        exec_with(plan(std::slice::from_ref(&constant)), o, rows_layout).await;
                    if ansi {
                        let e = got.unwrap_err();
                        assert_eq!(e.error_class(), ErrorClass::DivideByZero, "{e}");
                    } else {
                        let m = moment_rows(&got.unwrap().0, true);
                        assert_eq!(m["2"], [None], "legacy={legacy} layout={rows_layout}");
                        assert!(m["1"][0].is_some());
                    }
                }
            }
        }
    }
}

/// Partial → final over partial states equals one phase, also when the
/// partial phase flushes and bypasses.
#[tokio::test]
async fn moments_two_phase_equals_single_phase() {
    let (s, b) = moment_input(60_000, 700);
    let want = moment_oracle(&b, 700, true, false);
    let o = Opts {
        both_layouts: false,
        ..Opts::default()
    };
    let tiny = BypassPolicy {
        max_table_bytes: 64 << 10,
        min_rows: u64::MAX,
        ..BypassPolicy::default()
    };
    let policies = [BypassPolicy::default(), tiny];
    for policy in policies {
        let make = || {
            two_phase_of(
                scan(&s, b.clone()),
                spec(&s, vec![0], moment_aggs()),
                policy,
                None,
            )
            .0
        };
        check_both_layouts(make, o, true, &want).await;
    }
    // Unique keys: the partial phase passes rows through, one state per row.
    let (s, b) = moment_input(40_000, 1_000_000);
    let want = moment_oracle(&b, 40_000, true, false);
    let bypass = BypassPolicy {
        min_rows: 10_000,
        ..BypassPolicy::default()
    };
    let (plan, partial) = two_phase_of(
        scan(&s, b.clone()),
        spec(&s, vec![0], moment_aggs()),
        bypass,
        None,
    );
    let (out, _) = exec_with(plan, o, false).await.unwrap();
    assert!(partial.bypassed_drivers() > 0);
    assert_moments_close(&moment_rows(&out, true), &want, "bypassed");
}

/// Under a small budget the moment states spill with the others, merge from
/// the runs and give the in-memory result.
#[tokio::test]
async fn moments_spill_and_merge() {
    let (s, b) = moment_input(30_000, 25_000);
    // Three of the aggregates (one of each state width): the budget below
    // holds about a third of their groups.
    let aggs: Vec<AggregateExpr> = [0, 4, 6]
        .into_iter()
        .map(|i| moment_aggs().remove(i))
        .collect();
    let keep = |m: Moments| -> Moments {
        m.into_iter()
            .map(|(k, v)| (k, [0, 4, 6].iter().map(|&i| v[i]).collect()))
            .collect()
    };
    let want = keep(moment_oracle(&b, 25_000, true, false));
    let (_m, scope) = spill_dir();
    let mut sp = spec(&s, vec![0], aggs);
    sp.spill = Some(scope.clone());
    let o = Opts {
        limit: 2 << 20,
        both_layouts: false,
        ..Opts::default()
    };
    for rows_layout in [false, true] {
        let (out, m) = exec_with(node(scan(&s, b.clone()), sp.clone()), o, rows_layout)
            .await
            .unwrap();
        assert!(
            extra(&m, "HashAggregate", "spill_runs") > 0,
            "rows {rows_layout}"
        );
        assert_eq!(extra(&m, "HashAggregate", "groups"), 25_000);
        assert_moments_close(&moment_rows(&out, true), &want, "spilled");
    }
    let (plan, _) = two_phase_of(
        scan(&s, b.clone()),
        sp,
        BypassPolicy::default(),
        Some(scope),
    );
    let (out, _) = exec_with(plan, o, false).await.unwrap();
    assert_moments_close(&moment_rows(&out, true), &want, "spilled two-phase");
}

/// `DISTINCT` pairs: `corr(DISTINCT x, y)` over the distinct non-NULL
/// pairs, next to a plain `count(*)`.
#[tokio::test]
async fn moments_over_distinct_arguments() {
    let s = schema(vec![
        ("k", DataType::Int64),
        ("x", DataType::Float64),
        ("y", DataType::Float64),
    ]);
    let k = i64s(&[Some(1); 6]);
    let x = f64s(vec![
        Some(1.0),
        Some(1.0),
        Some(2.0),
        Some(3.0),
        None,
        Some(3.0),
    ]);
    let y = f64s(vec![
        Some(2.0),
        Some(2.0),
        Some(4.5),
        Some(5.0),
        Some(1.0),
        Some(5.0),
    ]);
    let input = || {
        scan(
            &s,
            vec![vec![Arc::clone(&k), Arc::clone(&x), Arc::clone(&y)]],
        )
    };
    let mut corr = moment_aggs().remove(6);
    assert_eq!(corr.function, AggregateFunction::Corr);
    corr.name = "c".to_owned();
    let flags = [true, false];
    let plan = distinct_plan(
        input(),
        spec(
            &s,
            vec![],
            vec![corr.clone(), AggregateExpr::count_star("n")],
        ),
        &flags,
    )
    .unwrap();
    // The distinct pairs (1,2), (2,4.5), (3,5), with a correlation of its own.
    let pairs = vec![Some(1.0), Some(2.0), Some(3.0)];
    let want = moment_oracle(
        &[vec![
            i64s(&[Some(1); 3]),
            f64s(pairs),
            f64s(vec![Some(2.0), Some(4.5), Some(5.0)]),
        ]],
        1,
        false,
        false,
    );
    let got = moment_rows(&exec(plan, Opts::default()).await.unwrap().0, false);
    let corr_at = |m: &Moments, i: usize| m[""][i];
    let want_corr = corr_at(&want, 6).unwrap();
    assert!((got[""][0].unwrap() - want_corr).abs() < 1e-12, "{got:?}");
    assert_eq!(got[""][1], Some(6.0));
}
