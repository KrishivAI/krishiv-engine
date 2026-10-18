// SPDX-License-Identifier: Apache-2.0
//! Aggregation tests. Expected values come from Spark 4.2.0 runs (the
//! queries are quoted next to each case); the property test compares with
//! a row-at-a-time oracle.
#![cfg(not(loom))]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
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
    Scheduler, Sink, Source, TaskContext, TokioClock, TokioExecutor, WorkloadClass,
};

use AggregateFunction::{Avg, Count, Max, Min, Sum};

/// Run options.
#[derive(Clone, Copy)]
struct Opts {
    slots: usize,
    limit: usize,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            slots: 3,
            limit: 1 << 30,
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
        },
        Arc::new(TokioClock),
    )
}

/// Runs `plan` on a deterministic executor (seed 7); checks the pool is
/// empty afterwards.
async fn exec(plan: Arc<dyn ExecNode>, o: Opts) -> crate::Result<(Vec<Batch>, QueryMetrics)> {
    let runner = QueryRunner::new(TokioExecutor::deterministic(o.slots)?);
    let root = pool(o.limit);
    let config = ExecConfig {
        seed: Some(7),
        ..ExecConfig::default()
    };
    let q = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        config,
        root.child("q", Some(o.limit)),
    );
    let mut stream = runner.submit(plan, Arc::clone(&q)).await?;
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

/// A spill directory for one test (removed when the manager drops).
fn spill_dir() -> (SpillManager, SpillScope) {
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "krishiv-agg-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let m = SpillManager::open(&SpillConfig {
        dir,
        ..SpillConfig::default()
    })
    .unwrap();
    let s = m.scope().unwrap();
    (m, s)
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

fn extra(m: &QueryMetrics, op: &str, name: &str) -> u64 {
    m.operators
        .values()
        .filter(|e| e.name == op)
        .flat_map(|e| e.extra.iter())
        .filter(|(n, _)| *n == name)
        .map(|(_, v)| *v)
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
    let mut first = spec(s, vec![0], grouped_aggs(s));
    first.output = AggregateOutput::Partial;
    let second = second_phase(&first, spill);
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
        let (out, m) = exec(node(scan(&s, b.clone()), sp), o).await.unwrap();
        assert_eq!(rows(&out), expected, "limit {limit}");
        assert!(
            extra(&m, "HashAggregate", "spill_runs") > 0,
            "limit {limit}"
        );
        assert!(extra(&m, "HashAggregate", "spill_bytes") > 0);
        assert_eq!(extra(&m, "HashAggregate", "groups"), 75_000);
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
    };
    let (out, m) = exec(node(scan(&s, b), sp), o).await.unwrap();
    assert_eq!(rows(&out), expected);
    assert!(extra(&m, "HashAggregate", "repartitions") > 0);
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
    let (_m, scope) = spill_dir();
    let mut sp = spec(&s, vec![0], grouped_aggs(&s));
    sp.spill = Some(scope);
    let agg = HashAggregate::new(sp).unwrap();
    let root = pool(1 << 30);
    let config = ExecConfig::default();
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
        let out = rt.block_on(exec(plan, Opts { slots, limit })).unwrap().0;
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
    assert_eq!(p.bypassed_drivers(), 1);
    let mut first = spec(&s, vec![0], grouped_aggs(&s));
    first.output = AggregateOutput::Partial;
    let p = PartialAggregate::new(first, BypassPolicy::default(), None).unwrap();
    let limited = Arc::new(Pipe(PartialNode::new(scan(&s, b), p), Limit::new(5)));
    let plan = Arc::new(Break(limited, Count::new(), false));
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
    // Three disjoint tables that fit, whose merge does not.
    let r = finalize_directly(
        spec(&s, vec![0], grouped_aggs(&s)),
        &s,
        &b,
        3,
        3 << 20,
        false,
    )
    .await;
    assert_eq!(
        r.err().map(|e| e.error_class()),
        Some(ErrorClass::UnableToAcquireMemory)
    );
    let (_m, scope) = spill_dir();
    let mut global = spec(&s, vec![], grouped_aggs(&s));
    global.spill = Some(scope.clone());
    let out = finalize_directly(global, &s, &b, 3, 1 << 30, true)
        .await
        .unwrap();
    assert_eq!(out.len(), 1);
    assert!(out[0].starts_with("30000|"), "{out:?}");
    let mut empty = spec(&s, vec![0], grouped_aggs(&s));
    empty.spill = Some(scope);
    let out = finalize_directly(empty, &s, &[], 1, 1 << 30, true)
        .await
        .unwrap();
    assert!(out.is_empty());
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
) -> crate::Result<Vec<String>> {
    let agg = HashAggregate::new(sp).unwrap();
    let root = pool(1 << 30);
    let config = ExecConfig::default();
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
    let p = PartialAggregate::new(first.clone(), BypassPolicy::default(), None).unwrap();
    let limited = Arc::new(Pipe(scan(&s, b), Limit::new(5)));
    let plan = node(PartialNode::new(limited, p), second);
    let o = Opts {
        slots: 1,
        ..Opts::default()
    };
    // Five rows of one (seeded) batch: three keys, five rows in all.
    let out = rows(&exec(plan, o).await.unwrap().0);
    let total: i64 = out.iter().map(|r| r[2..].parse::<i64>().unwrap()).sum();
    assert_eq!((out.len(), total), (3, 5), "{out:?}");

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
    let agg = HashAggregate::new(spec(&s, vec![0], grouped_aggs(&s))).unwrap();
    let root = pool(1 << 30);
    let config = ExecConfig::default();
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
    drop((g, drivers));
    assert_eq!(root.reserved(), 0);
}
