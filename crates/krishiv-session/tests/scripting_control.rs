// SPDX-License-Identifier: Apache-2.0
//! A script is one query for cancellation and the time limit, and holds its rows in memory with no
//! fixed caps (P2b row 15, R3 and R4; Spark 4.2.0: `tests/spark-golden/v4.2.0/scripting/interrupt`).
#![cfg(test)]

use std::time::{Duration, Instant};

use arrow_array::{Array, RecordBatch};
use futures::executor::block_on;
use krishiv_session::{QueryId, QueryOptions, Session};

const OP: &str = "00112233-4455-6677-8899-aabbccddeeff";

/// Runs `script` under the operation `OP` on a thread, calls `then` on the session after
/// `after`, and returns the class of the error the script ended with.
fn ended_by(s: &Session, script: &str, after: Duration, then: impl FnOnce(&Session)) -> String {
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            block_on(s.sql_with(script, QueryOptions::new().operation_id(OP)))
                .map(|_| ())
                .map_err(|e| e.error_class().name().to_owned())
        });
        std::thread::sleep(after);
        then(s);
        worker.join().unwrap().unwrap_err()
    })
}

fn total(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

fn count(s: &Session, sql: &str) -> i64 {
    let batches: Vec<RecordBatch> = block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    let column = batches[0].column(0);
    column
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .map_or_else(
            || {
                i64::from(
                    column
                        .as_any()
                        .downcast_ref::<arrow_array::Int32Array>()
                        .unwrap()
                        .value(0),
                )
            },
            |a| a.value(0),
        )
}

#[test]
fn a_cancel_between_two_statements_ends_the_script() {
    let s = Session::new().unwrap();
    let started = Instant::now();
    let class = ended_by(
        &s,
        "BEGIN DECLARE i INT DEFAULT 0; WHILE TRUE DO SET i = i + 1; END WHILE; END",
        Duration::from_millis(400),
        |s| assert!(s.cancel(&QueryId::parse(OP).unwrap())),
    );
    assert_eq!(class, "OPERATION_CANCELED");
    assert!(started.elapsed() < Duration::from_secs(30));
    // The script's row records the reason; the operation id is free again.
    assert_eq!(
        count(
            &s,
            "SELECT count(*) FROM system.queries WHERE query_id = '00112233-4455-6677-8899-aabbccddeeff' \
             AND state = 'cancelled' AND cancel_reason = 'user'"
        ),
        1
    );
    assert!(!s.cancel(&QueryId::parse(OP).unwrap()));
}

#[test]
fn no_handler_catches_a_cancel_and_no_later_statement_runs() {
    let s = Session::new().unwrap();
    block_on(s.sql("DECLARE hit INT DEFAULT 0")).unwrap();
    let class = ended_by(
        &s,
        "BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION SET session.hit = 1; \
         WHILE TRUE DO SET session.hit = 0; END WHILE; SET session.hit = 2; END",
        Duration::from_millis(400),
        |s| assert!(s.cancel(&QueryId::parse(OP).unwrap())),
    );
    assert_eq!(class, "OPERATION_CANCELED");
    assert_ne!(count(&s, "SELECT session.hit"), 1, "the handler ran");
    assert_ne!(count(&s, "SELECT session.hit"), 2, "a later statement ran");
}

#[test]
fn interrupt_all_ends_a_script() {
    let s = Session::new().unwrap();
    let class = ended_by(
        &s,
        "BEGIN WHILE TRUE DO SELECT 1; END WHILE; END",
        Duration::from_millis(400),
        |s| assert!(!s.interrupt_all().is_empty()),
    );
    assert_eq!(class, "OPERATION_CANCELED");
}

#[test]
fn the_time_limit_counts_for_the_whole_script() {
    let s = Session::new().unwrap();
    s.conf()
        .set("krishiv.query.maxExecutionTime", "700ms")
        .unwrap();
    // Every statement is far inside the limit; the loop is not.
    let started = Instant::now();
    let error = block_on(
        s.sql("BEGIN DECLARE i INT DEFAULT 0; WHILE TRUE DO SET i = i + 1; END WHILE; END"),
    )
    .map(|_| ())
    .unwrap_err();
    assert_eq!(error.error_class().name(), "KRISHIV_QUERY_TIMEOUT");
    assert!(started.elapsed() < Duration::from_secs(30));
    // No handler catches it.
    let error = block_on(s.sql(
        "BEGIN DECLARE EXIT HANDLER FOR SQLEXCEPTION SELECT 'caught'; WHILE TRUE DO SELECT 1; END WHILE; END",
    ))
    .map(|_| ())
    .unwrap_err();
    assert_eq!(error.error_class().name(), "KRISHIV_QUERY_TIMEOUT");
}

#[test]
fn a_script_under_an_operation_id_runs_its_statements_and_shows_them_all() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql_with(
        "BEGIN DECLARE x INT DEFAULT 1; SET x = x + 1; SELECT x; END",
        QueryOptions::new().operation_id(OP),
    ))
    .unwrap();
    let rows: Vec<RecordBatch> = block_on(df.collect()).unwrap();
    assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    // The script's row has the operation's id; its statements have rows of their own.
    assert!(
        count(
            &s,
            "SELECT count(*) FROM system.queries WHERE query_id = '00112233-4455-6677-8899-aabbccddeeff' \
             AND state = 'finished'"
        ) >= 1
    );
    assert!(
        count(
            &s,
            "SELECT count(*) FROM system.queries WHERE state = 'finished'"
        ) >= 2
    );
}

#[test]
fn a_result_has_no_fixed_cap_on_rows_or_types() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql(
        "BEGIN SELECT id, array(id, id + 1) AS a, map('k', id) AS m, struct(id AS x) AS st, \
         CAST(CAST(id AS STRING) AS BINARY) AS b FROM range(3000); END",
    ))
    .unwrap();
    let batches: Vec<RecordBatch> = block_on(df.collect()).unwrap();
    assert_eq!(
        batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        3000
    );
    assert_eq!(batches[0].num_columns(), 5);
}

#[test]
fn a_for_loop_has_no_fixed_cap_on_rows() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql(
        "BEGIN DECLARE n INT DEFAULT 0; FOR r AS SELECT id FROM range(600) DO SET n = n + 1; END FOR; SELECT n; END",
    ))
    .unwrap();
    let batches: Vec<RecordBatch> = block_on(df.collect()).unwrap();
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(count(&s, "SELECT 600"), 600);
}

#[test]
fn a_result_frame_survives_changes_to_the_catalog() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql("BEGIN SELECT id FROM range(5); END")).unwrap();
    block_on(s.sql("CREATE TEMP VIEW v1 AS SELECT 1 AS a")).unwrap();
    block_on(s.sql("CREATE TEMP VIEW v2 AS SELECT 2 AS a")).unwrap();
    block_on(s.sql("DROP VIEW v1")).unwrap();
    assert_eq!(total(&block_on(df.collect()).unwrap()), 5);
    drop(df);
    let derived = block_on(s.sql("BEGIN SELECT id FROM range(5); END"))
        .unwrap()
        .limit(2);
    block_on(s.sql("DROP VIEW v2")).unwrap();
    block_on(s.sql("CREATE TEMP VIEW v3 AS SELECT 3 AS a")).unwrap();
    assert_eq!(total(&block_on(derived.collect()).unwrap()), 2);
}

#[test]
fn execute_immediate_outside_a_script_keeps_the_operation_id() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql_with(
        "EXECUTE IMMEDIATE 'SELECT 1 AS one'",
        QueryOptions::new().operation_id(OP),
    ))
    .unwrap();
    block_on(df.collect()).unwrap();
    assert!(
        count(
            &s,
            "SELECT count(*) FROM system.queries WHERE query_id = '00112233-4455-6677-8899-aabbccddeeff'"
        ) >= 1
    );
}
