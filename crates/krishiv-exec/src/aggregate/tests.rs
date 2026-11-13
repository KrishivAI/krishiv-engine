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
    /// `krishiv.exec.agg.directKeys`.
    direct: bool,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            slots: 3,
            limit: 1 << 30,
            both_layouts: true,
            seed: Some(7),
            detailed: false,
            direct: true,
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
        agg_direct_keys: o.direct,
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
        DataType::Interval(_) => a
            .as_primitive::<arrow_array::types::IntervalYearMonthType>()
            .value(i)
            .to_string(),
        DataType::Duration(_) => a
            .as_primitive::<arrow_array::types::DurationMicrosecondType>()
            .value(i)
            .to_string(),
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

fn agg(f: impl std::borrow::Borrow<AggregateFunction>, col: usize, s: &SchemaRef) -> AggregateExpr {
    let f = f.borrow();
    let t = s.field(col).data_type().clone();
    AggregateExpr::new(f.clone(), col, t, format!("{f:?}_{col}"))
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
        udafs: None,
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

fn masked(f: AggregateFunction, col: usize, mask: usize, s: &SchemaRef) -> AggregateExpr {
    AggregateExpr {
        mask: Some(mask),
        ..agg(f, col, s)
    }
}

fn bools(v: &[Option<bool>]) -> ArrayRef {
    Arc::new(arrow_array::BooleanArray::from(v.to_vec()))
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
        agg(&Sum, 1, s),
        agg(&Min, 1, s),
        agg(&Avg, 1, s),
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
mod collect;
mod disk;
mod misc;
mod simple;
mod spill_writers;
mod structs;

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
        agg(&Count, 3, s),
        agg(&Sum, 3, s),
        agg(&Avg, 3, s),
        agg(&Sum, 4, s),
        agg(&Avg, 4, s),
        agg(&Min, 6, s),
        agg(&Max, 6, s),
        agg(&Max, 4, s),
        agg(&Min, 5, s),
        agg(&Max, 5, s),
    ];
    if picks {
        a.extend([
            agg(
                &First {
                    ignore_nulls: false,
                },
                3,
                s,
            ),
            agg(&Last { ignore_nulls: true }, 5, s),
            agg(&First { ignore_nulls: true }, 4, s),
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
    let mut sp = spec(&s, vec![0], vec![agg(&Sum, 1, &s)]);
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
        .map(|f| AggregateExpr {
            function: f.clone(),
            args: if super::arity(f) == 2 {
                vec![1, 2]
            } else {
                vec![1]
            },
            input: AggInput::Raw,
            arg_type: DataType::Float64,
            mask: None,
            name: format!("{f:?}"),
            element: None,
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

type Plan<'a> = Box<dyn Fn(&[Vec<ArrayRef>]) -> Arc<dyn ExecNode> + 'a>;

mod part1;
mod part2;
