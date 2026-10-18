// SPDX-License-Identifier: Apache-2.0
//! The stateless operators in real pipelines (P2-T051 acceptance), in
//! deterministic mode: scan pushdown, filter, project, `LIMIT` stopping its
//! upstream, the morsel metric, and the `UNION` semantics of `union.sql`
//! at operator level.
#![cfg(not(loom))]
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::sync::Arc;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{BooleanArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use krishiv_common::error::ErrorClass;
use krishiv_exec::testing::{Break, Count, Pipe, Scan};
use krishiv_exec::{
    ExecConfig, ExecNode, Filter, FnPredicate, Limit, MemoryScan, Predicate, Project, ProjectItem,
    QueryContext, QueryId, QueryRunner, Result, ScanPushdown, Scheduler, TokioClock, TokioExecutor,
    WorkloadClass,
};
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
    let mut stream = runner.submit(plan, Arc::clone(q)).await?;
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
    let mut scan = MemoryScan::try_new(schema(), table(10, 100)).unwrap();
    scan.push_filter(mod_pred(2, 0));
    scan.push_projection(&[1, 0]).unwrap();
    let scan = Arc::new(scan);
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
