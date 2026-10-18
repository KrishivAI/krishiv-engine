// SPDX-License-Identifier: Apache-2.0
//! The morsel executor's benchmarks. Under CodSpeed (simulation mode,
//! `--cfg codspeed`) only the single-thread deterministic benches run:
//! instruction counts of a multi-thread runtime would mostly measure
//! parking. Locally the 4-thread throughput bench runs as well (wall time).
//!
//! - `executor/scan_filter_count_1m_rows/{1_thread,4_threads}`: throughput.
//! - `driver/{256,4096}_batches_4_empty_operators`: one shared 4,096-row
//!   batch repeated through 4 pass-through operators into a count. The
//!   per-batch driver cost is the slope:
//!   `(t(4096) - t(256)) / 3840`, which cancels the per-query setup.

#![expect(
    missing_docs,
    reason = "criterion_group! generates an undocumented function"
)]

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use futures::StreamExt;
use krishiv_exec::testing::{Break, Count, Filter, PassThrough, Pipe, RangeSource, Repeat, Scan};
use krishiv_exec::{
    DEFAULT_BATCH_SIZE, ExecConfig, ExecNode, QueryContext, QueryId, QueryRunner, Scheduler,
    TokioExecutor, WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};

const ROWS: u64 = 1 << 20;

/// Runs `plan` to the end on `runner`, driven by `caller`, and returns the
/// result rows.
#[expect(
    clippy::expect_used,
    reason = "a failing query must fail the bench loudly, not skip it"
)]
fn run(runner: &QueryRunner, caller: &tokio::runtime::Runtime, plan: Arc<dyn ExecNode>) -> usize {
    let root = TreePool::new(&MemoryConfig::default());
    let query = QueryContext::new(
        QueryId(1),
        WorkloadClass::Batch,
        ExecConfig::default(),
        root.child("bench", None),
    );
    caller.block_on(async {
        let mut stream = runner.submit(plan, Arc::clone(&query)).await.expect("plan");
        let mut rows = 0;
        while let Some(b) = stream.next().await {
            rows += b.expect("batch").num_rows();
        }
        query.finished().await;
        rows
    })
}

fn scan_filter_count() -> Arc<dyn ExecNode> {
    let scan = Arc::new(Scan(RangeSource::new(ROWS)));
    let filter = Arc::new(Pipe(scan, Filter::new(|v| v % 3 == 0)));
    Arc::new(Break(filter, Count::new(), false))
}

/// A current-thread runtime and a deterministic one-slot runner on it.
#[expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]
fn single_thread() -> (tokio::runtime::Runtime, QueryRunner) {
    let caller = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("caller runtime");
    let executor = {
        let _in_runtime = caller.enter();
        TokioExecutor::deterministic(1).expect("executor")
    };
    (caller, QueryRunner::new(executor))
}

fn bench(c: &mut Criterion) {
    let (caller, runner) = single_thread();
    let mut g = c.benchmark_group("executor");
    g.throughput(Throughput::Elements(ROWS));
    g.bench_function("scan_filter_count_1m_rows/1_thread", |b| {
        b.iter(|| black_box(run(&runner, &caller, scan_filter_count())));
    });
    #[cfg(not(codspeed))]
    {
        #[expect(
            clippy::expect_used,
            reason = "a broken fixture must fail the bench loudly, not skip it"
        )]
        let executor = TokioExecutor::new(&krishiv_exec::ExecutorConfig {
            compute_threads: 4,
            ..krishiv_exec::ExecutorConfig::default()
        })
        .expect("executor");
        let multi = QueryRunner::new(executor);
        g.bench_function("scan_filter_count_1m_rows/4_threads", |b| {
            b.iter(|| black_box(run(&multi, &caller, scan_filter_count())));
        });
    }
    g.finish();

    let mut g = c.benchmark_group("driver");
    for batches in [256u64, 4096] {
        g.throughput(Throughput::Elements(batches));
        g.bench_function(format!("{batches}_batches_4_empty_operators"), |b| {
            b.iter(|| {
                #[expect(
                    clippy::expect_used,
                    reason = "a broken fixture must fail the bench loudly, not skip it"
                )]
                let source = Repeat::new(DEFAULT_BATCH_SIZE, batches).expect("batch");
                let mut plan: Arc<dyn ExecNode> = Arc::new(Scan(source));
                for _ in 0..4 {
                    plan = Arc::new(Pipe(plan, Arc::new(PassThrough)));
                }
                black_box(run(
                    &runner,
                    &caller,
                    Arc::new(Break(plan, Count::new(), false)),
                ))
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
