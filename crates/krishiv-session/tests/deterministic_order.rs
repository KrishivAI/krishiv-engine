// SPDX-License-Identifier: Apache-2.0
//! `krishiv.deterministicOrder` (P2-T094): the same query returns the same bytes on every run.
#![cfg(test)]

use arrow_array::RecordBatch;
use futures::executor::block_on;
use krishiv_session::Session;

fn run(s: &Session, sql: &str) -> Vec<RecordBatch> {
    block_on(async { s.sql(sql).await.unwrap().collect().await.unwrap() })
}

/// Runs `sql`, checks it returns exactly `rows` rows (so a repeat of the
/// same wrong answer cannot pass) and returns its shape.
fn first_run(s: &Session, sql: &str, rows: usize) -> String {
    let batches = run(s, sql);
    let got: usize = batches.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(got, rows, "row count of `{sql}`");
    shape(&batches)
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
    for (sql, rows) in [
        ("SELECT id, m FROM d WHERE id % 3 = 0", 20_000),
        ("SELECT g, sum(m), avg(m), count(*) FROM d GROUP BY g", 7),
        (
            "SELECT a.id, b.g FROM d a JOIN d b ON a.id = b.id WHERE a.id < 5000",
            5_000,
        ),
        (
            "SELECT id FROM d WHERE id % 5 = 0 UNION ALL SELECT id FROM d WHERE id % 7 = 0",
            12_000 + 8_572,
        ),
        ("SELECT g, count(DISTINCT id), sum(m) FROM d GROUP BY g", 7),
    ] {
        let first = first_run(&s, sql, rows);
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
        let first = first_run(&s, sql, 200_003);
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
    // Every residue of 50021 (odd) has an even id below 400000.
    let first = first_run(&s, union, 50_021);
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
    let first = first_run(&spill, grouped, 150_001);
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
    // 150000 even ids + 100000 multiples of 3; each k matches one q row.
    let first = first_run(&s, sql, 250_000);
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
        "SELECT id, id % 7 AS k FROM range(0, 400000, 1, 24) ORDER BY k LIMIT 50",
        "SELECT id, id % 7 AS k, CAST(id AS DOUBLE) / 3 AS x FROM range(0, 300000, 1, 24) ORDER BY k",
        "SELECT a.id, b.id FROM range(0, 20000, 1, 24) a JOIN (SELECT id, id % 100 AS k FROM range(0, 20000, 1, 24)) b ON a.id % 100 = b.k WHERE a.id < 300 LIMIT 500",
        // A fan-in mixing a branch that splits into lanes with one holding a flush barrier
        // (review B1): every member keeps one driver.
        "SELECT x FROM (SELECT a.id AS x FROM range(0,20000,1,24) a FULL JOIN range(0,1000,1,4) b ON a.id=b.id*3 \
         UNION ALL SELECT id AS x FROM range(0,40000,1,24))",
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

/// The most drivers any pipeline of `sql` ran, from its profile.
fn max_drivers(s: &Session, sql: &str) -> u64 {
    let df = block_on(s.sql(sql)).unwrap();
    let p = block_on(df.explain_analyze()).unwrap();
    p.metrics
        .pipelines
        .values()
        .map(|m| m.drivers)
        .max()
        .unwrap_or(0)
}

/// Review M1: the lanes are engaged where they should be and only there. This fails if
/// `lane_count` silently returns `None` (every result test would still pass: the old mode
/// ran one driver at every thread count).
#[test]
fn lanes_engage_one_driver_per_logical_partition() {
    use krishiv_exec::{ExecutorConfig, TokioExecutor};
    let executor = TokioExecutor::new(&ExecutorConfig {
        compute_threads: 12,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let s = Session::builder()
        .config("krishiv.deterministicOrder", "true")
        .config("krishiv.memory.limit", "16GB")
        .executor(executor)
        .build()
        .unwrap();
    // 24 partitions: capped at 16 lanes.
    let split = "SELECT id % 11 AS g, sum(CAST(id AS DOUBLE) / 7) FROM range(0, 400000, 1, 24) GROUP BY id % 11";
    assert_eq!(max_drivers(&s, split), 16, "{split}");
    // Fewer partitions than the cap: one lane each.
    let few = "SELECT id % 11 AS g, sum(CAST(id AS DOUBLE) / 7) FROM range(0, 400000, 1, 5) GROUP BY id % 11";
    assert_eq!(max_drivers(&s, few), 5, "{few}");
    // A fan-in sharing a flush barrier keeps every member at one driver (B1): the 24-partition
    // members would show 16; only the 4-partition join build, outside the group, uses lanes.
    let mixed = "SELECT x FROM (SELECT a.id AS x FROM range(0,20000,1,24) a FULL JOIN range(0,1000,1,4) b ON a.id=b.id*3 \
                 UNION ALL SELECT id AS x FROM range(0,40000,1,24))";
    assert_eq!(max_drivers(&s, mixed), 4, "{mixed}");
}

/// Every stateless operator, a hash join probe, a partial aggregate and a pushed LIMIT under the
/// mode with lanes available: each query returns the same bytes on repeated runs and at two thread
/// counts, which also exercises each operator's `lane_safe` answer and the scans' splits.
#[test]
fn lane_eligible_pipelines_over_every_operator_are_repeatable() {
    use arrow_array::Int64Array;
    use arrow_schema::{DataType, Field, Schema};
    use krishiv_exec::{ExecutorConfig, TokioExecutor};
    use std::sync::Arc;
    let queries = [
        // Expand (rollup), Generate (explode), ReplicateRows (intersect all), UnionAlign, probe.
        "SELECT g, count(*), sum(v) FROM (SELECT id % 5 AS g, id AS v FROM range(0, 20000, 1, 24)) GROUP BY ROLLUP(g)",
        "SELECT count(*), sum(e) FROM (SELECT explode(array(id, id + 1)) AS e FROM range(0, 20000, 1, 24))",
        "SELECT count(*) FROM (SELECT id FROM range(0, 20000, 1, 24) INTERSECT ALL SELECT id FROM range(0, 20000, 1, 24))",
        "SELECT count(*), sum(x) FROM (SELECT id AS x FROM range(0, 8000, 1, 24) UNION ALL SELECT id AS x FROM range(0, 8000, 1, 24))",
        "SELECT count(*), sum(a.id) FROM range(0, 20000, 1, 24) a JOIN range(0, 500, 1, 3) b ON a.id % 500 = b.id",
        // A pushed-down LIMIT on an in-memory table: that scan refuses to split.
        "SELECT count(*), sum(k) FROM (SELECT k FROM t LIMIT 700)",
        "SELECT k % 7, count(*), sum(k) FROM t GROUP BY k % 7",
    ];
    let mut shapes: Vec<Vec<String>> = Vec::new();
    for threads in [2, 12] {
        let executor = TokioExecutor::new(&ExecutorConfig {
            compute_threads: threads,
            ..ExecutorConfig::default()
        })
        .unwrap();
        let s = Session::builder()
            .config("krishiv.deterministicOrder", "true")
            .config("krishiv.memory.limit", "16GB")
            .executor(executor)
            .build()
            .unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
        let batches = (0..8)
            .map(|b| {
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(Int64Array::from(
                        (0..500).map(|i| b * 500 + i).collect::<Vec<i64>>(),
                    ))],
                )
                .unwrap()
            })
            .collect();
        s.register_batches("t", schema, batches).unwrap();
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
        assert_eq!(
            shapes[0][i], shapes[1][i],
            "`{q}` depends on the thread count"
        );
    }
}
