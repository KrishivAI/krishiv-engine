// SPDX-License-Identifier: Apache-2.0
//! Helpers of the hash join tests.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Decimal128Type, Float64Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, SchemaRef};
use futures::StreamExt;
use krishiv_exec::testing::{Batches, Scan};
use krishiv_exec::{
    ExecConfig, ExecNode, ExecutorConfig, HashJoin, HashJoinNode, JoinSpec, QueryContext, QueryId,
    QueryRunner, Result, TokioClock, TokioExecutor, WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, SpillConfig, SpillManager, SpillScope, TreePool};
use krishiv_types::batch::{Batch, Weights};

pub(crate) const MIB: usize = 1 << 20;

pub(crate) fn root(limit: usize) -> Arc<TreePool> {
    TreePool::with_clock(
        &MemoryConfig {
            limit: Some(limit),
            quantum: 16 << 10,
            spill_wait_timeout: Duration::from_secs(2),
            abort_release_timeout: Duration::from_secs(30),
            spill_scratch_reserve: krishiv_memory::DEFAULT_SPILL_SCRATCH_RESERVE,
        },
        Arc::new(TokioClock),
    )
}

/// `KRISHIV_TEST_JOIN_ROW_LAYOUT=1` runs every join of this suite with
/// `krishiv.exec.join.rowLayout` on (P2-T158: the suite with the flag on
/// and off; tests that pick the format themselves are unaffected when it
/// is unset).
pub(crate) fn row_layout_forced() -> bool {
    std::env::var("KRISHIV_TEST_JOIN_ROW_LAYOUT").is_ok_and(|v| v == "1") || ROW_LAYOUT.get()
}

thread_local! {
    /// The row layout for the joins [`query`] creates on this thread (the
    /// `#[tokio::test]` runtimes are single-threaded).
    static ROW_LAYOUT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs the joins of this thread's next queries with the row layout on or
/// off.
pub(crate) fn set_row_layout(on: bool) {
    ROW_LAYOUT.set(on);
}

pub(crate) fn query(root: &TreePool, id: u64, mut config: ExecConfig) -> Arc<QueryContext> {
    config.join_row_layout |= row_layout_forced();
    config.fail_on_stall = true;
    config.stall_threshold = Duration::from_secs(5);
    QueryContext::new(
        QueryId(id),
        WorkloadClass::Interactive,
        config,
        root.child(&format!("q{id}"), None),
    )
}

pub(crate) fn runner(threads: usize) -> QueryRunner {
    let executor = TokioExecutor::new(&ExecutorConfig {
        compute_threads: threads,
        io_threads: 1,
        blocking_threads: threads,
        io_blocking_threads: 2,
    });
    QueryRunner::new(executor.unwrap())
}

/// Budgets below this do not hold the tests' joins: they spill.
pub(crate) const SPILL_FREE: usize = 64 * MIB;

/// The runner for a query under a `limit`-byte budget. A spilling run uses
/// the deterministic executor (`slots` drivers on the test's
/// current-thread runtime, blocking work scheduled like any task), whose
/// paused clock also times the pool's spill and abort waits: a deadline
/// then passes only while every task waits, never because the machine is
/// busy, so a run's outcome does not depend on machine load (and replays).
/// Unbudgeted runs use real threads.
pub(crate) fn runner_for(slots: usize, limit: usize) -> QueryRunner {
    if limit < SPILL_FREE {
        QueryRunner::new(TokioExecutor::deterministic(slots).unwrap())
    } else {
        runner(slots)
    }
}

/// Runs `plan` to the end; returns every batch or the query's error, once
/// every task has ended.
pub(crate) async fn run(
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

/// A spill manager in its own directory under the target's temp dir.
pub(crate) fn spill() -> (SpillManager, SpillScope) {
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "join-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let manager = SpillManager::open(&SpillConfig {
        dir,
        ..SpillConfig::default()
    })
    .unwrap();
    let scope = manager.scope().unwrap();
    (manager, scope)
}

/// Files left in a spill scope's directory.
pub(crate) fn spill_files(scope: &SpillScope) -> usize {
    std::fs::read_dir(scope.dir()).map_or(0, Iterator::count)
}

pub(crate) fn scan(schema: &SchemaRef, batches: Vec<Batch>) -> Arc<dyn ExecNode> {
    Arc::new(Scan(Batches::new(Arc::clone(schema), batches)))
}

pub(crate) fn join(
    spec: JoinSpec,
    build: Arc<dyn ExecNode>,
    probe: Arc<dyn ExecNode>,
) -> Arc<dyn ExecNode> {
    let join = HashJoin::try_new(spec).unwrap();
    Arc::new(HashJoinNode { join, build, probe })
}

/// Splits `data` into batches of `rows` rows, unweighted.
pub(crate) fn batches(data: &RecordBatch, rows: usize) -> Vec<Batch> {
    (0..data.num_rows())
        .step_by(rows.max(1))
        .map(|o| Batch::new(data.slice(o, rows.min(data.num_rows() - o))))
        .collect()
}

pub(crate) fn cell(a: &ArrayRef, i: usize) -> String {
    if a.is_null(i) {
        return "∅".to_owned();
    }
    match a.data_type() {
        DataType::Int64 => a.as_primitive::<Int64Type>().value(i).to_string(),
        DataType::Int32 => a.as_primitive::<Int32Type>().value(i).to_string(),
        DataType::Date32 => a.as_primitive::<Date32Type>().value(i).to_string(),
        DataType::Float64 => format!("{:.4}", a.as_primitive::<Float64Type>().value(i)),
        DataType::Decimal128(_, _) => a.as_primitive::<Decimal128Type>().value_as_string(i),
        DataType::Utf8 => a.as_string::<i32>().value(i).to_owned(),
        DataType::Boolean => a.as_boolean().value(i).to_string(),
        t => format!("<{t}>"),
    }
}

/// Every output row as `a|b|...` with its weight (`*w`) unless 1, sorted.
pub(crate) fn render(batches: &[Batch]) -> Vec<String> {
    let mut rows = Vec::new();
    for b in batches {
        for i in 0..b.num_rows() {
            let mut row: Vec<String> = b.data().columns().iter().map(|c| cell(c, i)).collect();
            let w = match b.weights() {
                Weights::Const(w) => *w,
                Weights::Column(c) => c.value(i),
            };
            if w != 1 {
                row.push(format!("*{w}"));
            }
            rows.push(row.join("|"));
        }
    }
    rows.sort();
    rows
}

/// A custom metric of the operator named `op`.
pub(crate) fn metric(q: &QueryContext, op: &str, name: &str) -> u64 {
    q.metrics()
        .operators
        .values()
        .filter(|e| e.name == op)
        .flat_map(|e| {
            e.extra
                .iter()
                .filter(|(n, _)| *n == name)
                .filter_map(|(_, v)| v.value())
        })
        .sum()
}

pub(crate) fn assert_released(root: &TreePool) {
    assert_eq!(
        root.reserved(),
        0,
        "the query's memory is released at its end"
    );
}
