// SPDX-License-Identifier: Apache-2.0
//! The morsel executor under CodSpeed: scan → filter → count throughput at
//! 1 and N worker slots, and the driver's per-batch overhead (one shared
//! 4,096-row batch, repeated, through empty operators into a count).

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
    DEFAULT_BATCH_SIZE, ExecConfig, ExecNode, ExecutorConfig, QueryContext, QueryId, QueryRunner,
    Scheduler, TokioExecutor, WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};

const ROWS: u64 = 1 << 20;

/// Runs `plan` to the end on `runner` and returns the result rows.
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
        let mut stream = runner.submit(plan, query).await.expect("plan");
        let mut rows = 0;
        while let Some(b) = stream.next().await {
            rows += b.expect("batch").num_rows();
        }
        rows
    })
}

#[expect(
    clippy::expect_used,
    reason = "a broken fixture must fail the bench loudly, not skip it"
)]
fn bench(c: &mut Criterion) {
    let caller = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("caller runtime");
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    let mut g = c.benchmark_group("executor");
    g.throughput(Throughput::Elements(ROWS));
    for threads in [1, cores] {
        let executor = TokioExecutor::new(&ExecutorConfig {
            compute_threads: threads,
            ..ExecutorConfig::default()
        })
        .expect("executor");
        let runner = QueryRunner::new(executor);
        g.bench_function(
            format!("scan_filter_count_1m_rows/{threads}_threads"),
            |b| {
                b.iter(|| {
                    let scan = Arc::new(Scan(RangeSource::new(ROWS)));
                    let filter = Arc::new(Pipe(scan, Filter::new(|v| v % 3 == 0)));
                    black_box(run(
                        &runner,
                        &caller,
                        Arc::new(Break(filter, Count::new(), false)),
                    ))
                });
            },
        );
    }
    g.finish();

    // Per-batch scheduling overhead: 256 batches through 4 empty operators
    // on one driver; divide by 256 for the cost per batch.
    let batches = 256u64;
    let executor = TokioExecutor::new(&ExecutorConfig {
        compute_threads: 1,
        ..ExecutorConfig::default()
    })
    .expect("executor");
    let runner = QueryRunner::new(executor);
    let mut g = c.benchmark_group("driver");
    g.throughput(Throughput::Elements(batches));
    g.bench_function("overhead_256_batches_4_empty_operators", |b| {
        b.iter(|| {
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
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
