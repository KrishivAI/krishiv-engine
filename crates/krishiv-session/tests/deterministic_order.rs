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

/// A hash join probing a `UNION ALL`: its probe operator needs a flush, so
/// the runner starts the branches together; the rows still come out in the
/// same order on every run.
#[test]
fn a_join_probing_a_union_gives_the_same_order_on_every_run() {
    let s = Session::new().unwrap();
    s.conf().set("krishiv.deterministicOrder", "true").unwrap();
    run(
        &s,
        "CREATE TABLE p USING parquet AS \
         SELECT id, id % 1000 AS k, CAST(id AS DOUBLE) / 3 AS x FROM range(300000)",
    );
    run(
        &s,
        "CREATE TABLE q USING parquet AS SELECT id AS k, id * 2 AS v FROM range(1000)",
    );
    let sql = "SELECT u.id, u.x, q.v FROM \
               (SELECT id, k, x FROM p WHERE id % 2 = 0 UNION ALL SELECT id, k, x FROM p WHERE id % 3 = 0) u \
               LEFT JOIN q ON u.k = q.k";
    let first = shape(&run(&s, sql));
    for i in 1..5 {
        assert!(first == shape(&run(&s, sql)), "run {i} differs");
    }
}

/// Logical partitions (T094 G1): a deterministic query returns the same bytes whatever the
/// number of compute threads, because the morsels are dealt to a fixed set of lanes.
#[test]
fn the_result_does_not_depend_on_the_thread_count() {
    use krishiv_exec::{ExecutorConfig, TokioExecutor};
    let queries = [
        "SELECT id, CAST(id AS DOUBLE) / 7 AS x FROM range(0, 400000, 1, 24) WHERE id % 3 = 0",
        "SELECT id % 11 AS g, sum(CAST(id AS DOUBLE) / 7), avg(CAST(id AS DOUBLE) / 3), count(*) \
         FROM range(0, 400000, 1, 24) GROUP BY id % 11",
        "SELECT id FROM range(0, 400000, 1, 24) WHERE id % 5 = 0 LIMIT 100",
        "SELECT collect_list(id) FROM (SELECT id FROM range(0, 5000, 1, 24))",
    ];
    let mut shapes: Vec<Vec<String>> = Vec::new();
    for threads in [1, 4, 12] {
        let executor = TokioExecutor::new(&ExecutorConfig {
            compute_threads: threads,
            ..ExecutorConfig::default()
        })
        .unwrap();
        let s = Session::builder()
            .config("krishiv.deterministicOrder", "true")
            .executor(executor)
            .build()
            .unwrap();
        shapes.push(
            queries
                .iter()
                .map(|q| {
                    let first = shape(&run(&s, q));
                    assert_eq!(first, shape(&run(&s, q)), "two runs of `{q}` differ");
                    first
                })
                .collect(),
        );
    }
    for (i, q) in queries.iter().enumerate() {
        assert!(
            shapes[0][i] == shapes[1][i] && shapes[1][i] == shapes[2][i],
            "`{q}` depends on the thread count"
        );
    }
}
