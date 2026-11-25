// SPDX-License-Identifier: Apache-2.0
//! Every admitted query counts as running (`krishiv_exec::queries_running`)
//! from admission until its tasks have ended, whether it succeeds, fails or
//! its stream is dropped: the last one to end lets the executor return freed
//! memory (design `0.4-memory-query-end-release`). The counter is
//! process-wide, so this binary runs its cases in one test.
#![cfg(test)]

use std::time::{Duration, Instant};

use futures::StreamExt;
use krishiv_exec::queries_running;
use krishiv_session::Session;

/// Waits up to five seconds for no query to be running.
async fn settles() -> bool {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if queries_running() == 0 {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    false
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_query_ends_its_run() {
    let s = Session::new().unwrap();
    assert_eq!(queries_running(), 0);

    s.sql("SELECT 1 AS one")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert!(settles().await, "a finished query");

    s.sql("SET spark.sql.ansi.enabled = true").await.unwrap();
    let failed = s
        .sql("SELECT 1 / (id - id) FROM range(10)")
        .await
        .unwrap()
        .collect()
        .await;
    assert!(failed.is_err());
    assert!(settles().await, "a failed query");

    let mut stream = s
        .sql("SELECT id FROM range(100000000)")
        .await
        .unwrap()
        .execute_stream()
        .await
        .unwrap();
    assert!(stream.next().await.is_some());
    assert!(queries_running() >= 1, "a query with an open stream runs");
    drop(stream);
    assert!(settles().await, "a dropped stream");
}
