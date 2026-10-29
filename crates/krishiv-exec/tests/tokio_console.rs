// SPDX-License-Identifier: Apache-2.0
//! tokio-console task naming and the planted slow operator (P2-T129).
#![cfg(feature = "tokio-console-test")]

use std::sync::Arc;

use futures::StreamExt;
use krishiv_exec::console::{driver_task_name, io_task_name};
use krishiv_exec::testing::{BusyLoop, Pipe, RangeSource, Scan};
use krishiv_exec::{
    ExecConfig, ExecNode, Executor, ExecutorConfig, QueryContext, QueryId, QueryRunner,
    TokioExecutor, WorkloadClass,
};
use krishiv_memory::{MemoryConfig, MemoryPool, TreePool};

fn query(pool: &TreePool) -> Arc<QueryContext> {
    QueryContext::new(
        QueryId(1),
        WorkloadClass::Interactive,
        ExecConfig {
            fail_on_stall: false,
            ..ExecConfig::default()
        },
        pool.child("q1", None),
    )
}

#[test]
fn task_names_match_what_the_console_shows() {
    assert_eq!(driver_task_name(QueryId(1), 0, 0), "q1/0/driver0");
    assert_eq!(driver_task_name(QueryId(7), 2, 3), "q7/2/driver3");
    assert_eq!(io_task_name("file-write"), "io/file-write");
    assert_eq!(io_task_name("spill"), "io/spill");
}

#[tokio::test]
async fn a_busy_loop_counts_a_stall() {
    let pool = TreePool::new(&MemoryConfig::default());
    let q = query(&pool);
    let executor: Arc<dyn Executor> = TokioExecutor::new(&ExecutorConfig {
        compute_threads: 2,
        io_threads: 1,
        blocking_threads: 2,
        io_blocking_threads: 2,
    })
    .unwrap();
    let runner = QueryRunner::new(Arc::clone(&executor));
    // One batch: one 50 ms non-yielding slice in the driver.
    let source = RangeSource::new(krishiv_exec::DEFAULT_BATCH_SIZE as u64);
    let plan: Arc<dyn ExecNode> = Arc::new(Pipe(Arc::new(Scan(source)), BusyLoop::new()));
    let mut stream = runner.execute(plan, Arc::clone(&q)).await.unwrap();
    let mut rows = 0;
    while let Some(batch) = stream.next().await {
        rows += batch.unwrap().num_rows();
    }
    drop(stream);
    q.finished().await;
    assert_eq!(rows, krishiv_exec::DEFAULT_BATCH_SIZE);
    let stalls = q.metrics().drivers.stalls;
    assert!(
        stalls >= 1,
        "a 50 ms busy loop must count a stall, got {stalls}"
    );
    assert_eq!(
        pool.reserved(),
        0,
        "the query's memory is released at its end"
    );
}
