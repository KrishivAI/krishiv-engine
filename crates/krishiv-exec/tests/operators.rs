// SPDX-License-Identifier: Apache-2.0
//! The stateless operators in real pipelines (P2-T051 acceptance), in
//! deterministic mode: scan pushdown, filter, project, `LIMIT` stopping its
//! upstream, the morsel metric, and the `UNION` semantics of `union.sql`
//! at operator level.
#![cfg(not(loom))]
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Decimal128Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use krishiv_common::error::ErrorClass;
use krishiv_exec::testing::{Break, Count, Pipe, Scan};
use krishiv_exec::{
    ExecConfig, ExecNode, Filter, FnPredicate, Limit, MemoryScan, Predicate, Project, ProjectItem,
    PushedFilter, QueryContext, QueryId, QueryRunner, Result, ScalarExpr, ScanPushdowns,
    TokioClock, TokioExecutor, UnionAlign, WorkloadClass,
};
use krishiv_io::ColumnPath;
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
use krishiv_types::batch::{Batch, Weights};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]))
}

/// `batches` batches of `rows` rows; `a` counts up from 0.
fn table(batches: usize, rows: usize) -> Vec<Batch> {
    (0..batches)
        .map(|k| {
            let a = Int64Array::from_iter_values((0..rows).map(|i| (k * rows + i) as i64));
            let s = StringArray::from_iter_values((0..rows).map(|i| format!("r{}", k * rows + i)));
            Batch::new(RecordBatch::try_new(schema(), vec![Arc::new(a), Arc::new(s)]).unwrap())
        })
        .collect()
}

fn root() -> Arc<TreePool> {
    TreePool::with_clock(
        &MemoryConfig {
            limit: Some(1 << 30),
            quantum: 64 << 10,
            spill_wait_timeout: Duration::from_secs(2),
            abort_release_timeout: Duration::from_secs(30),
            spill_scratch_reserve: krishiv_memory::DEFAULT_SPILL_SCRATCH_RESERVE,
        },
        Arc::new(TokioClock),
    )
}

fn query(root: &TreePool, config: ExecConfig) -> Arc<QueryContext> {
    QueryContext::new(
        QueryId(1),
        WorkloadClass::Interactive,
        config,
        root.child("q", None),
    )
}

/// Runs `plan` to the end; the query's pool must be empty afterwards.
async fn run(
    runner: &QueryRunner,
    plan: Arc<dyn ExecNode>,
    q: &Arc<QueryContext>,
) -> Result<Vec<Batch>> {
    let mut stream = runner.execute(plan, Arc::clone(q)).await?;
    let mut out = Vec::new();
    let mut error = None;
    while let Some(b) = stream.next().await {
        match b {
            Ok(b) => out.push(b),
            Err(e) => error = Some(e),
        }
    }
    drop(stream);
    q.finished().await;
    error.map_or(Ok(out), Err)
}

fn deterministic() -> QueryRunner {
    QueryRunner::new(TokioExecutor::deterministic(1).unwrap())
}

fn pipe(child: Arc<dyn ExecNode>, op: Arc<dyn krishiv_exec::Operator>) -> Arc<dyn ExecNode> {
    Arc::new(Pipe(child, op))
}

fn a_values(batches: &[Batch]) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            b.data()
                .column(0)
                .as_primitive::<Int64Type>()
                .values()
                .to_vec()
        })
        .collect()
}

fn rows(batches: &[Batch]) -> usize {
    batches.iter().map(Batch::num_rows).sum()
}

fn mod_pred(modulus: i64, keep: i64) -> Arc<dyn Predicate> {
    Arc::new(FnPredicate(move |b: &RecordBatch| {
        Ok(b.column(0)
            .as_primitive::<Int64Type>()
            .iter()
            .map(|v| v.map(|v| v % modulus == keep))
            .collect::<BooleanArray>())
    }))
}

fn scan_node(scan: &Arc<MemoryScan>) -> Arc<dyn ExecNode> {
    Arc::new(Scan(Arc::clone(scan) as Arc<dyn krishiv_exec::Source>))
}

#[tokio::test(start_paused = true)]
async fn scan_with_pushdown_then_filter_and_project() {
    let base = MemoryScan::try_new(schema(), table(10, 100)).unwrap();
    let pushed = ScanPushdowns {
        projection: Some(vec![ColumnPath::new("s"), ColumnPath::new("a")]),
        filters: vec![PushedFilter::opaque(mod_pred(2, 0))],
        ..ScanPushdowns::default()
    };
    let scan = Arc::new(base.with_pushdowns(pushed).unwrap());
    let filter = Filter::new(
        krishiv_exec::Source::schema(&*scan),
        Arc::new(FnPredicate(|b: &RecordBatch| {
            // Column 1 is `a` after the pushed projection: keep a % 3 == 0.
            Ok(b.column(1)
                .as_primitive::<Int64Type>()
                .iter()
                .map(|v| v.map(|v| v % 3 == 0))
                .collect::<BooleanArray>())
        })),
    );
    let project = Project::try_new(
        &krishiv_exec::Operator::schema(&*filter),
        vec![ProjectItem::aliased(1, "id")],
    )
    .unwrap();
    let plan = pipe(pipe(scan_node(&scan), filter), project);
    let root = root();
    let q = query(&root, ExecConfig::default());
    let out = run(&deterministic(), plan, &q).await.unwrap();
    let expected: Vec<i64> = (0..1000).filter(|v| v % 6 == 0).collect();
    assert_eq!(a_values(&out), expected);
    assert_eq!(out[0].data().schema().field(0).name(), "id");
    assert_eq!(scan.rows_read(), 1000);
    assert_eq!(root.reserved(), 0);
}

/// Acceptance: a limit stops upstream reading (rows read <= limit + one batch).
#[tokio::test(start_paused = true)]
async fn limit_stops_the_scan() {
    for (offset, limit) in [(0, 2_500), (0, 1), (700, 2_000), (0, 0)] {
        let scan = Arc::new(MemoryScan::try_new(schema(), table(1_000, 1_000)).unwrap());
        let l = Limit::with_offset(schema(), offset, limit);
        let plan = pipe(scan_node(&scan), l);
        let root = root();
        let config = ExecConfig {
            batch_size: 1_000,
            ..ExecConfig::default()
        };
        let q = query(&root, config);
        let out = run(&deterministic(), plan, &q).await.unwrap();
        assert_eq!(
            a_values(&out),
            (offset as i64..(offset + limit) as i64).collect::<Vec<_>>()
        );
        assert!(
            scan.rows_read() <= offset + limit + 1_000,
            "read {} rows for offset {offset} limit {limit}",
            scan.rows_read()
        );
        assert!(scan.rows_read() < 1_000_000);
        assert_eq!(root.reserved(), 0);
    }
}

/// With several drivers the count is exact and each other driver reads at
/// most the batch it had in flight.
#[tokio::test]
async fn limit_is_exact_with_parallel_drivers() {
    let executor = TokioExecutor::new(&krishiv_exec::ExecutorConfig {
        compute_threads: 4,
        io_threads: 1,
        blocking_threads: 2,
        io_blocking_threads: 2,
    })
    .unwrap();
    let runner = QueryRunner::new(executor);
    let scan = Arc::new(MemoryScan::try_new(schema(), table(400, 1_000)).unwrap());
    let plan = pipe(scan_node(&scan), Limit::new(schema(), 12_345));
    let root = root();
    let config = ExecConfig {
        batch_size: 1_000,
        morsel_rows: 5_000,
        ..ExecConfig::default()
    };
    let q = query(&root, config);
    let out = run(&runner, plan, &q).await.unwrap();
    assert_eq!(rows(&out), 12_345);
    assert!(scan.rows_read() <= 12_345 + 4 * 1_000 + 4 * 5_000);
    assert_eq!(root.reserved(), 0);
}

/// Scans report the morsels each driver claimed.
#[tokio::test(start_paused = true)]
async fn scans_report_the_morsel_metric() {
    let scan = Arc::new(MemoryScan::try_new(schema(), table(10, 100)).unwrap());
    let root = root();
    let q = query(
        &root,
        ExecConfig {
            morsel_rows: 250,
            batch_size: 100,
            ..ExecConfig::default()
        },
    );
    let out = run(&deterministic(), scan_node(&scan), &q).await.unwrap();
    assert_eq!(rows(&out), 1_000);
    let entry = q
        .metrics()
        .operators
        .values()
        .find(|e| e.name == "MemoryScan")
        .cloned()
        .unwrap();
    // 3 batches reach 300 >= 250: morsels of 3, 3, 3 and 1 batches.
    assert_eq!(entry.metrics.morsels, 4);
    assert_eq!(entry.metrics.rows_out, 1_000);
}

/// A seeded run replays the same morsel order; another seed permutes it.
#[tokio::test(start_paused = true)]
async fn deterministic_mode_replays_scan_order() {
    async fn firsts(seed: Option<u64>) -> Vec<i64> {
        let scan = Arc::new(MemoryScan::try_new(schema(), table(16, 10)).unwrap());
        let q = query(
            &root(),
            ExecConfig {
                morsel_rows: 10,
                seed,
                ..ExecConfig::default()
            },
        );
        let out = run(&deterministic(), scan_node(&scan), &q).await.unwrap();
        a_values(&out)
    }
    let a = firsts(Some(3)).await;
    assert_eq!(a, firsts(Some(3)).await);
    assert_ne!(a, firsts(Some(4)).await);
    assert_eq!(firsts(None).await, (0..160).collect::<Vec<_>>());
}

/// Weights survive scan, filter, project and union alignment end to end.
#[tokio::test(start_paused = true)]
async fn weights_flow_through_the_pipeline() {
    let data = table(1, 6);
    let (record, _) = data.into_iter().next().unwrap().into_parts();
    let w = Int64Array::from(vec![1, -1, 2, 3, -4, 5]);
    let batch = Batch::with_weights(record, Weights::Column(w)).unwrap();
    let scan = Arc::new(MemoryScan::try_new(schema(), vec![batch]).unwrap());
    let filter = Filter::new(schema(), mod_pred(2, 1)); // a = 1, 3, 5
    let project = Project::try_new(&schema(), vec![ProjectItem::column(0)]).unwrap();
    let plan = pipe(pipe(scan_node(&scan), filter), project);
    let q = query(&root(), ExecConfig::default());
    let out = run(&deterministic(), plan, &q).await.unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(a_values(&out), [1, 3, 5]);
    assert_eq!(
        out[0].weights(),
        &Weights::Column(Int64Array::from(vec![-1, 3, 5]))
    );
}

/// `LIMIT` rejects a weighted stream with its error class; the query still
/// releases its memory.
#[tokio::test(start_paused = true)]
async fn limit_rejects_weighted_input_in_a_pipeline() {
    let (record, _) = table(1, 6).into_iter().next().unwrap().into_parts();
    let weighted = Batch::with_weights(record, Weights::Const(2)).unwrap();
    let scan = Arc::new(MemoryScan::try_new(schema(), vec![weighted]).unwrap());
    let plan = pipe(scan_node(&scan), Limit::new(schema(), 3));
    let root = root();
    let q = query(&root, ExecConfig::default());
    let e = run(&deterministic(), plan, &q).await.unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert_eq!(root.reserved(), 0);
}

// ---- union.sql at operator level ------------------------------------------

/// One `UNION ALL` branch: scan, optional pre-union steps, alignment to the
/// union schema. The physical plan (P2-T074) wires several of these into
/// one sink; until then each branch runs as its own pipeline and the
/// outputs are concatenated, which is what the fan-in does.
async fn branch(
    input: SchemaRef,
    batches: Vec<Batch>,
    union: &SchemaRef,
    steps: Vec<Arc<dyn krishiv_exec::Operator>>,
) -> Vec<Batch> {
    let scan = Arc::new(MemoryScan::try_new(input, batches).unwrap());
    let mut plan = scan_node(&scan);
    let mut last: SchemaRef = krishiv_exec::Source::schema(&*scan);
    for step in steps {
        last = step.schema();
        plan = pipe(plan, step);
    }
    plan = pipe(plan, UnionAlign::try_new(&last, Arc::clone(union)).unwrap());
    let q = query(&root(), ExecConfig::default());
    run(&deterministic(), plan, &q).await.unwrap()
}

fn row_strings(batches: &[Batch]) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            let cols: Vec<String> = b.data().columns().iter().map(|c| display(c, r)).collect();
            out.push(cols.join("\t"));
        }
    }
    out.sort();
    out
}

fn display(col: &ArrayRef, row: usize) -> String {
    if col.is_null(row) {
        return "NULL".to_owned();
    }
    match col.data_type() {
        DataType::Int64 => col.as_primitive::<Int64Type>().value(row).to_string(),
        DataType::Utf8 => col.as_string::<i32>().value(row).to_owned(),
        DataType::Decimal128(_, _) => col.as_primitive::<Decimal128Type>().value(row).to_string(),
        other => unreachable!("{other}"),
    }
}

fn two_cols(c1: &[i64], c2: &[&str]) -> Batch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("c1", DataType::Int64, false),
        Field::new("c2", DataType::Utf8, false),
    ]));
    Batch::new(
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(c1.to_vec())),
                Arc::new(StringArray::from(c2.to_vec())),
            ],
        )
        .unwrap(),
    )
}

fn t1_schema() -> SchemaRef {
    two_cols(&[], &[]).schema()
}

/// `SELECT * FROM (SELECT * FROM t1 UNION ALL SELECT * FROM t1)`:
/// expected `1 a, 1 a, 2 b, 2 b`.
#[tokio::test(start_paused = true)]
async fn union_all_simple() {
    let t1 = || vec![two_cols(&[1, 2], &["a", "b"])];
    let union = UnionAlign::unify(&[t1_schema(), t1_schema()]).unwrap();
    let mut all = branch(t1_schema(), t1(), &union, vec![]).await;
    all.extend(branch(t1_schema(), t1(), &union, vec![]).await);
    assert_eq!(row_strings(&all), ["1\ta", "1\ta", "2\tb", "2\tb"]);
}

/// SPARK-18622 / SPARK-18841 shape: `SELECT 0 a UNION ALL SELECT SUM(1) a
/// UNION ALL SELECT 0 a` gives `0, 0, 1` with `a` nullable bigint (the
/// analyzer already cast the literals; the aggregate output is nullable),
/// and `SELECT 1 AS x, col` over two branches gives `1 1` twice, the
/// projection computed per branch.
#[tokio::test(start_paused = true)]
async fn union_all_nullability_and_projection_per_branch() {
    let non_null = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let nullable = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
    let one = |s: &SchemaRef, v: i64| {
        vec![Batch::new(
            RecordBatch::try_new(Arc::clone(s), vec![Arc::new(Int64Array::from(vec![v]))]).unwrap(),
        )]
    };
    let union = UnionAlign::unify(&[
        Arc::clone(&non_null),
        Arc::clone(&nullable),
        Arc::clone(&non_null),
    ])
    .unwrap();
    assert!(union.field(0).is_nullable());
    let mut all = branch(Arc::clone(&non_null), one(&non_null, 0), &union, vec![]).await;
    all.extend(branch(Arc::clone(&nullable), one(&nullable, 1), &union, vec![]).await);
    all.extend(branch(Arc::clone(&non_null), one(&non_null, 0), &union, vec![]).await);
    assert_eq!(row_strings(&all), ["0", "0", "1"]);

    // SELECT 1 AS x, col FROM p1 / p3: the constant is a project expression.
    #[derive(Debug)]
    struct One;
    impl ScalarExpr for One {
        fn data_type(&self) -> DataType {
            DataType::Int64
        }
        fn nullable(&self) -> bool {
            false
        }
        fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
            Ok(Arc::new(Int64Array::from(vec![1; batch.num_rows()])))
        }
    }
    let p = || {
        Project::try_new(
            &non_null,
            vec![
                ProjectItem::expr("x", Arc::new(One)),
                ProjectItem::column(0),
            ],
        )
        .unwrap() as Arc<dyn krishiv_exec::Operator>
    };
    let out_schema = p().schema();
    let union = UnionAlign::unify(&[Arc::clone(&out_schema), Arc::clone(&out_schema)]).unwrap();
    let mut all = branch(Arc::clone(&non_null), one(&non_null, 1), &union, vec![p()]).await;
    all.extend(branch(Arc::clone(&non_null), one(&non_null, 1), &union, vec![p()]).await);
    assert_eq!(row_strings(&all), ["1\t1", "1\t1"]);
}

/// SPARK-32638: `SELECT v FROM t3 UNION ALL SELECT v + v AS v FROM t3`
/// gives `1, 2` (decimal(11,0)); `SELECT SUM(v) FROM (... UNION ...)` gives
/// `3`. `UNION` (distinct) is `UNION ALL` plus a distinct aggregate, owned
/// by the aggregate task: here the distinct step is a reference one over the
/// `UNION ALL` output, and the sum runs through the test `Count` breaker.
#[tokio::test(start_paused = true)]
async fn union_and_union_all_over_decimals() {
    let dec = Arc::new(Schema::new(vec![Field::new(
        "v",
        DataType::Decimal128(11, 0),
        false,
    )]));
    let one = |v: i128| {
        vec![Batch::new(
            RecordBatch::try_new(
                Arc::clone(&dec),
                vec![Arc::new(
                    Decimal128Array::from(vec![v])
                        .with_precision_and_scale(11, 0)
                        .unwrap(),
                )],
            )
            .unwrap(),
        )]
    };
    let union = UnionAlign::unify(&[Arc::clone(&dec), Arc::clone(&dec)]).unwrap();
    let mut all = branch(Arc::clone(&dec), one(1), &union, vec![]).await;
    all.extend(branch(Arc::clone(&dec), one(2), &union, vec![]).await);
    assert_eq!(row_strings(&all), ["1", "2"]);
    let distinct: BTreeSet<String> = row_strings(&all).into_iter().collect();
    let sum: i128 = distinct.iter().map(|s| s.parse::<i128>().unwrap()).sum();
    assert_eq!(sum, 3);
}

/// Duplicates survive `UNION ALL` (which is why `UNION` needs the breaker),
/// and weights pass through the alignment: the retraction of a row from one
/// branch stays a retraction in the union.
#[tokio::test(start_paused = true)]
async fn union_all_keeps_duplicates_and_weights() {
    let (record, _) = two_cols(&[1, 1], &["a", "a"]).into_parts();
    let retract =
        Batch::with_weights(record, Weights::Column(Int64Array::from(vec![-1, 2]))).unwrap();
    let union = UnionAlign::unify(&[t1_schema(), t1_schema()]).unwrap();
    let mut all = branch(t1_schema(), vec![two_cols(&[1], &["a"])], &union, vec![]).await;
    all.extend(branch(t1_schema(), vec![retract], &union, vec![]).await);
    let total: i64 = all
        .iter()
        .map(|b| match b.weights() {
            Weights::Const(c) => c * b.num_rows() as i64,
            Weights::Column(w) => w.values().iter().sum(),
        })
        .sum();
    assert_eq!(total, 2);
    assert_eq!(rows(&all), 3);
}

/// A limit above a breaker-free union branch still finishes the query and
/// leaves no memory behind, including with a breaker above it.
#[tokio::test(start_paused = true)]
async fn limit_below_a_breaker_ends_cleanly() {
    let scan = Arc::new(MemoryScan::try_new(schema(), table(100, 100)).unwrap());
    let limited = pipe(scan_node(&scan), Limit::new(schema(), 250));
    let plan = Arc::new(Break(limited, Count::new(), false));
    let root = root();
    let q = query(
        &root,
        ExecConfig {
            batch_size: 100,
            ..ExecConfig::default()
        },
    );
    let out = run(&deterministic(), plan, &q).await.unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(
        out[0].data().column(0).as_primitive::<Int64Type>().value(0),
        250
    );
    assert!(scan.rows_read() <= 350);
    assert_eq!(root.reserved(), 0);
}
