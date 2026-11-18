// SPDX-License-Identifier: Apache-2.0
//! `QueryOptions::query_start` (P3-T101 §4.1): a statement run twice with the same fixed start
//! reads the same clock; without it each run reads its own.
#![cfg(test)]

use arrow_array::RecordBatch;
use krishiv_session::{QueryOptions, Session};

const CLOCKS: &str = "SELECT current_timestamp() AS ts, now() AS n, current_date() AS d, \
                      localtimestamp() AS lt";

/// 2001-09-09 01:46:40 UTC.
const FIXED: i64 = 1_000_000_000_000_000;

async fn run(s: &Session, options: QueryOptions) -> Vec<RecordBatch> {
    s.sql_with(CLOCKS, options)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
}

fn micros(batches: &[RecordBatch], column: usize) -> i64 {
    let col = batches[0].column(column);
    let ts = col
        .as_any()
        .downcast_ref::<arrow_array::TimestampMicrosecondArray>()
        .unwrap();
    ts.value(0)
}

#[tokio::test(flavor = "multi_thread")]
async fn a_fixed_query_start_gives_every_run_the_same_clock() {
    let s = Session::new().unwrap();
    let first = run(&s, QueryOptions::new().query_start(FIXED)).await;
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    let second = run(&s, QueryOptions::new().query_start(FIXED)).await;
    assert_eq!(first, second);
    assert_eq!(micros(&first, 0), FIXED);
    assert_eq!(micros(&first, 1), FIXED);
}

#[tokio::test(flavor = "multi_thread")]
async fn without_a_fixed_start_each_run_reads_the_clock() {
    let s = Session::new().unwrap();
    let first = run(&s, QueryOptions::new()).await;
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    let second = run(&s, QueryOptions::new()).await;
    assert!(micros(&second, 0) > micros(&first, 0));
    assert!(micros(&first, 0) > FIXED);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_fixed_start_does_not_leak_into_the_sessions_next_statement() {
    let s = Session::new().unwrap();
    let _ = run(&s, QueryOptions::new().query_start(FIXED)).await;
    let plain = s.sql(CLOCKS).await.unwrap().collect().await.unwrap();
    assert!(micros(&plain, 0) > FIXED);
}
