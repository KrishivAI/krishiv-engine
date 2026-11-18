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

fn days(batches: &[RecordBatch], column: usize) -> i32 {
    let col = batches[0].column(column);
    col.as_any()
        .downcast_ref::<arrow_array::Date32Array>()
        .unwrap()
        .value(0)
}

/// Microseconds since the epoch now.
fn wall_micros() -> i64 {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    i64::try_from(d.as_micros()).unwrap()
}

/// A session in UTC, so the local clock functions read the fixed instant unshifted.
async fn utc_session() -> Session {
    let s = Session::new().unwrap();
    s.sql("SET spark.sql.session.timeZone = UTC")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    s
}

/// Runs `CLOCKS` without a fixed start and checks it read the clock during the run (a test-side
/// bracket, so it holds whatever the wall clock does between runs); returns its reading.
async fn run_reads_the_clock(s: &Session) -> i64 {
    let before = wall_micros();
    let out = run(s, QueryOptions::new()).await;
    let after = wall_micros();
    let ts = micros(&out, 0);
    assert!(
        (before..=after).contains(&ts),
        "{ts} not within the run [{before}, {after}]"
    );
    ts
}

#[tokio::test(flavor = "multi_thread")]
async fn a_fixed_query_start_gives_every_run_the_same_clock() {
    let s = utc_session().await;
    let first = run(&s, QueryOptions::new().query_start(FIXED)).await;
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    let second = run(&s, QueryOptions::new().query_start(FIXED)).await;
    assert_eq!(first, second);
    assert_eq!(micros(&first, 0), FIXED);
    assert_eq!(micros(&first, 1), FIXED);
    // 2001-09-09 is day 11,574 of the epoch; in UTC the local timestamp is the instant itself.
    assert_eq!(days(&first, 2), 11_574);
    assert_eq!(micros(&first, 3), FIXED);
}

#[tokio::test(flavor = "multi_thread")]
async fn without_a_fixed_start_each_run_reads_the_clock() {
    let s = Session::new().unwrap();
    run_reads_the_clock(&s).await;
    run_reads_the_clock(&s).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn the_fixed_start_does_not_leak_into_the_sessions_next_statement() {
    let s = Session::new().unwrap();
    let fixed = run(&s, QueryOptions::new().query_start(FIXED)).await;
    assert_eq!(micros(&fixed, 0), FIXED);
    run_reads_the_clock(&s).await;
}
