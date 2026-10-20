// SPDX-License-Identifier: Apache-2.0
//! The aggregation shapes of TPC-H Q1, Q13, Q16 and Q18 as hand-wired
//! pipelines (the physical planner and the Parquet scan arrive later). The
//! inputs are what Spark 4.2.0 computes below each aggregate over TPC-H
//! SF 0.01, and the expected rows are Spark's results
//! (`tests/fixtures/tpch-sf0.01/shapes.sql`).
#![cfg(not(loom))]
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::fs::File;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Decimal128Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef};
use arrow_ipc::reader::FileReader;
use arrow_schema::{DataType, SchemaRef};
use futures::StreamExt;
use krishiv_exec::aggregate::{
    AggregateExpr, AggregateFunction, AggregateNode, AggregateOutput, AggregateSpec, BypassPolicy,
    HashAggregate, PartialAggregate, PartialNode, distinct_plan,
};
use krishiv_exec::testing::{Batches, Scan};
use krishiv_exec::{
    ExecConfig, ExecNode, QueryContext, QueryId, QueryMetrics, QueryRunner, TokioClock,
    TokioExecutor, WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, SpillConfig, SpillManager, SpillScope, TreePool};
use krishiv_types::batch::Batch;

use AggregateFunction::{Avg, Count, Sum};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tpch-sf0.01");

fn input(name: &str) -> (SchemaRef, Arc<dyn ExecNode>) {
    let file = File::open(format!("{DIR}/{name}.arrow")).unwrap();
    let reader = FileReader::try_new(file, None).unwrap();
    let schema = reader.schema();
    let batches = reader.map(|b| Batch::new(b.unwrap())).collect();
    let scan = Arc::new(Scan(Batches::new(Arc::clone(&schema), batches)));
    (schema, scan)
}

fn expected(name: &str) -> Vec<String> {
    let text = std::fs::read_to_string(format!("{DIR}/{name}.csv")).unwrap();
    let mut rows: Vec<String> = text.lines().map(|l| l.replace(',', "|")).collect();
    rows.sort();
    rows
}

/// `YYYY-MM-DD` of days since 1970-01-01 (proleptic Gregorian).
fn date(days: i32) -> String {
    let z = i64::from(days) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

fn decimal(v: i128, scale: i8) -> String {
    let s = usize::try_from(scale).unwrap();
    let digits = format!("{:0>width$}", v.unsigned_abs(), width = s + 1);
    let (int, frac) = digits.split_at(digits.len() - s);
    let sign = if v < 0 { "-" } else { "" };
    if s == 0 {
        format!("{sign}{int}")
    } else {
        format!("{sign}{int}.{frac}")
    }
}

fn cell(a: &ArrayRef, i: usize) -> String {
    match a.data_type() {
        _ if a.is_null(i) => String::new(),
        DataType::Int64 => a.as_primitive::<Int64Type>().value(i).to_string(),
        DataType::Int32 => a.as_primitive::<Int32Type>().value(i).to_string(),
        DataType::Decimal128(_, s) => decimal(a.as_primitive::<Decimal128Type>().value(i), *s),
        DataType::Date32 => date(a.as_primitive::<Date32Type>().value(i)),
        DataType::Utf8 => a.as_string::<i32>().value(i).to_owned(),
        t => format!("<{t}>"),
    }
}

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

/// Runs `plan` on `slots` deterministic drivers under a per-query `limit`.
async fn run(plan: Arc<dyn ExecNode>, slots: usize, limit: usize) -> (Vec<Batch>, QueryMetrics) {
    let runner = QueryRunner::new(TokioExecutor::deterministic(slots).unwrap());
    let root = TreePool::with_clock(
        &MemoryConfig {
            limit: Some(1 << 30),
            quantum: 16 << 10,
            spill_wait_timeout: Duration::from_secs(5),
            abort_release_timeout: Duration::from_secs(30),
            spill_scratch_reserve: krishiv_memory::DEFAULT_SPILL_SCRATCH_RESERVE,
        },
        Arc::new(TokioClock),
    );
    let config = ExecConfig {
        seed: Some(3),
        ..ExecConfig::default()
    };
    let pool = root.child("q", Some(limit));
    let q = QueryContext::new(QueryId(1), WorkloadClass::Batch, config, pool);
    let mut stream = runner.execute(plan, Arc::clone(&q)).await.unwrap();
    let mut out = Vec::new();
    while let Some(b) = stream.next().await {
        out.push(b.unwrap());
    }
    drop(stream);
    q.finished().await;
    assert_eq!(root.reserved(), 0);
    (out, q.metrics())
}

/// A spill manager on its own temporary directory. Fields drop in order: the
/// manager removes its session, then the directory itself goes (also when
/// the test panics).
struct TestSpill {
    _manager: SpillManager,
    _dir: tempfile::TempDir,
}

fn spill_scope() -> (TestSpill, SpillScope) {
    let dir = tempfile::Builder::new()
        .prefix("krishiv-tpch-agg-")
        .tempdir()
        .unwrap();
    let manager = SpillManager::open(&SpillConfig {
        // A path inside the temporary directory: the manager creates it
        // private (0700), whatever the umask.
        dir: dir.path().join("spill"),
        ..SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    (
        TestSpill {
            _manager: manager,
            _dir: dir,
        },
        scope,
    )
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

fn spilled(m: &QueryMetrics) -> u64 {
    m.operators
        .values()
        .flat_map(|e| e.extra.iter())
        .filter(|(n, _)| *n == "spill_runs")
        .filter_map(|(_, v)| v.value())
        .sum()
}

fn q1_spec(s: &SchemaRef) -> AggregateSpec {
    let aggs = vec![
        agg(Sum, 2, s),
        agg(Sum, 3, s),
        agg(Sum, 4, s),
        agg(Sum, 5, s),
        agg(Avg, 2, s),
        agg(Avg, 3, s),
        agg(Avg, 6, s),
        AggregateExpr::count_star("count_order"),
    ];
    spec(s, vec![0, 1], aggs)
}

/// Q1: eight aggregates (decimal sums and averages) over two string keys,
/// single-phase, two-phase, and spilling under a 1 MiB query budget.
#[tokio::test]
async fn q1_pricing_summary() {
    let want = expected("q1_expected");
    let (s, scan) = input("q1_input");
    let (out, _) = run(node(Arc::clone(&scan), q1_spec(&s)), 4, 1 << 30).await;
    assert_eq!(rows(&out), want);

    let mut first = q1_spec(&s);
    first.output = AggregateOutput::Partial;
    let mid = first.output_schema().unwrap();
    let mut at = 2;
    let merging = first
        .aggregates
        .iter()
        .map(|a| {
            let m = a.merging(at);
            at += m.args.len();
            m
        })
        .collect();
    let partial = PartialAggregate::new(first, BypassPolicy::default(), Some(3)).unwrap();
    let plan = node(
        PartialNode::new(Arc::clone(&scan), partial),
        spec(&mid, vec![0, 1], merging),
    );
    assert_eq!(rows(&run(plan, 4, 1 << 30).await.0), want);

    // Q18's inner aggregate (15,000 groups) under a 1 MiB budget spills.
    let (_m, scope) = spill_scope();
    let (s, lineitem) = input("q18_lineitem");
    let mut sp = spec(&s, vec![0], vec![agg(Sum, 1, &s)]);
    sp.spill = Some(scope);
    let (out, m) = run(node(lineitem, sp), 4, 1 << 20).await;
    assert!(spilled(&m) > 0);
    let big: Vec<String> = rows(&out)
        .into_iter()
        .filter(|r| r.split('|').nth(1).and_then(|v| v.parse::<f64>().ok()) > Some(300.0))
        .collect();
    assert_eq!(big, expected("q18_big_orders"));
}

/// Q13: the per-customer order count (NULL orders of the outer join not
/// counted), then the distribution of those counts: two aggregates
/// stacked.
#[tokio::test]
async fn q13_customer_distribution() {
    let (s, scan) = input("q13_input");
    let inner = node(scan, spec(&s, vec![0], vec![agg(Count, 1, &s)]));
    let mid = spec(&s, vec![0], vec![agg(Count, 1, &s)])
        .output_schema()
        .unwrap();
    let plan = node(
        inner,
        spec(&mid, vec![1], vec![AggregateExpr::count_star("custdist")]),
    );
    assert_eq!(
        rows(&run(plan, 3, 1 << 30).await.0),
        expected("q13_expected")
    );
}

/// Q16: `count(DISTINCT ps_suppkey)` grouped by brand, type and size.
#[tokio::test]
async fn q16_distinct_supplier_count() {
    let (s, scan) = input("q16_input");
    let sp = spec(&s, vec![0, 1, 2], vec![agg(Count, 3, &s)]);
    let plan = distinct_plan(scan, sp, &[true]).unwrap();
    assert_eq!(
        rows(&run(plan, 3, 1 << 30).await.0),
        expected("q16_expected")
    );
}

/// Q18: the outer aggregate over five keys (string, integers, date,
/// decimal).
#[tokio::test]
async fn q18_large_volume_customers() {
    let (s, scan) = input("q18_input");
    let plan = node(scan, spec(&s, vec![0, 1, 2, 3, 4], vec![agg(Sum, 5, &s)]));
    assert_eq!(
        rows(&run(plan, 2, 1 << 30).await.0),
        expected("q18_expected")
    );
}
