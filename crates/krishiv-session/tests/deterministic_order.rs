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

/// Enough groups that a partial table fills and flushes into its radix
/// partitions, whose order follows the per-query hash seed: the order of
/// the groups is still the same on every run.
#[test]
fn many_groups_come_out_in_the_same_order_on_every_run() {
    let s = Session::new().unwrap();
    s.conf().set("krishiv.deterministicOrder", "true").unwrap();
    run(
        &s,
        "CREATE TABLE w USING parquet AS \
         SELECT id, id % 200003 AS k, CAST(id AS DOUBLE) / 7 AS x FROM range(1000000)",
    );
    for sql in [
        "SELECT k, count(*), sum(x) FROM w GROUP BY k",
        "SELECT DISTINCT k FROM w",
    ] {
        let first = shape(&run(&s, sql));
        for i in 1..4 {
            assert!(first == shape(&run(&s, sql)), "run {i} of `{sql}` differs");
        }
    }
}

/// A `UNION ALL` whose branches feed a partial aggregation (a fan-in the
/// runner may start together) and a grouped float `sum` that spills under
/// a small memory limit: still the same bytes on every run.
#[test]
fn fan_ins_and_spills_give_the_same_bytes_on_every_run() {
    let s = Session::new().unwrap();
    s.conf().set("krishiv.deterministicOrder", "true").unwrap();
    run(
        &s,
        "CREATE TABLE f USING parquet AS \
         SELECT id, id % 50021 AS k, CAST(id AS DOUBLE) / 3 AS x FROM range(400000)",
    );
    let union = "SELECT k, sum(x), count(*) FROM \
                 (SELECT k, x FROM f WHERE id % 2 = 0 UNION ALL SELECT k, x * 1.5 FROM f WHERE id % 3 = 0) \
                 GROUP BY k";
    let first = shape(&run(&s, union));
    for i in 1..4 {
        assert!(
            first == shape(&run(&s, union)),
            "run {i} of the union differs"
        );
    }
    let spill = Session::builder()
        .config("krishiv.memory.limit", "8m")
        .build()
        .unwrap();
    spill
        .conf()
        .set("krishiv.deterministicOrder", "true")
        .unwrap();
    run(
        &spill,
        "CREATE TEMPORARY VIEW g AS SELECT id % 150001 AS k, CAST(id AS DOUBLE) / 7 AS x FROM range(600000)",
    );
    let grouped = "SELECT k, sum(x), avg(x) FROM g GROUP BY k";
    let first = shape(&run(&spill, grouped));
    for i in 1..3 {
        assert!(
            first == shape(&run(&spill, grouped)),
            "run {i} under spill differs"
        );
    }
}
