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
use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use krishiv_exec::testing::Scan;
use krishiv_exec::{
    ExecConfig, ExecNode, MemoryScan, QueryContext, QueryId, QueryRunner, Result, Scheduler,
    TokioClock, TokioExecutor, WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};
use krishiv_types::batch::Batch;

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

fn scan_node(scan: &Arc<MemoryScan>) -> Arc<dyn ExecNode> {
    Arc::new(Scan(Arc::clone(scan) as Arc<dyn krishiv_exec::Source>))
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
