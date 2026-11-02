// SPDX-License-Identifier: Apache-2.0
//! `krishiv.deterministicOrder` (P2-T094): the same query returns the same bytes on every run.
#![cfg(test)]

use arrow_array::RecordBatch;
use futures::executor::block_on;
use krishiv_session::Session;

fn run(s: &Session, sql: &str) -> Vec<RecordBatch> {
    block_on(async { s.sql(sql).await.unwrap().collect().await.unwrap() })
}

/// The rows as text, in the order they came, batch boundaries included.
fn shape(batches: &[RecordBatch]) -> String {
    batches
        .iter()
        .map(|b| format!("{b:?}"))
        .collect::<Vec<_>>()
        .join("|")
}

#[test]
fn repeated_runs_of_unordered_queries_are_byte_identical() {
    let s = Session::new().unwrap();
    s.conf().set("krishiv.deterministicOrder", "true").unwrap();
    run(
        &s,
        "CREATE TABLE d USING parquet AS \
         SELECT id, id % 7 AS g, CAST(id AS DECIMAL(20,2)) AS m FROM range(60000)",
    );
    for sql in [
        "SELECT id, m FROM d WHERE id % 3 = 0",
        "SELECT g, sum(m), avg(m), count(*) FROM d GROUP BY g",
        "SELECT a.id, b.g FROM d a JOIN d b ON a.id = b.id WHERE a.id < 5000",
        "SELECT id FROM d WHERE id % 5 = 0 UNION ALL SELECT id FROM d WHERE id % 7 = 0",
        "SELECT g, count(DISTINCT id), sum(m) FROM d GROUP BY g",
    ] {
        let first = shape(&run(&s, sql));
        for i in 1..6 {
            assert_eq!(first, shape(&run(&s, sql)), "run {i} of `{sql}` differs");
        }
    }
}

#[test]
fn the_setting_is_off_by_default_and_a_session_setting() {
    let s = Session::new().unwrap();
    let get = |s: &Session| {
        let r = run(s, "SET krishiv.deterministicOrder");
        format!("{r:?}")
    };
    assert!(get(&s).contains("false"), "{}", get(&s));
    s.conf().set("krishiv.deterministicOrder", "true").unwrap();
    assert!(get(&s).contains("true"));
}
