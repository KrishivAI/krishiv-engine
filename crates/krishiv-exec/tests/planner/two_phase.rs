// SPDX-License-Identifier: Apache-2.0
//! Two-phase aggregates (design P2-T174 unit 2): an aggregate whose input
//! must move is planned as partial -> exchange -> final, and one worker runs
//! the three nodes as one breaker. Run as written (the operators each side of
//! a cut exchange runs), they return the same rows, spilled or not.

use krishiv_exec::OperatorRole;
use krishiv_memory::{MemoryConfig, MemoryPool as _, SpillConfig, SpillManager, TreePool};
use proptest::prelude::*;

use super::{Env, Runnable, cell, env, exchanges::env as exchanges_env};

/// The rows of `plan` run under `root`, sorted, and the number of aggregate
/// sinks and spill files the query had.
fn rows_of(plan: &Runnable, root: &TreePool) -> (Vec<String>, usize, u64) {
    let (out, query) = super::run_in(plan, root);
    let batches = out.unwrap_or_else(|e| panic!("{plan}: {e}"));
    let mut rows = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b.columns().iter().map(|c| cell(c.as_ref(), r)).collect();
            rows.push(cells.join("|"));
        }
    }
    rows.sort();
    let metrics = query.metrics();
    let sinks = metrics
        .operators
        .values()
        .filter(|o| o.name == "HashAggregate" && o.role == OperatorRole::Sink)
        .count();
    let files = metrics
        .operators
        .values()
        .map(|o| o.metrics.spill_files)
        .sum();
    assert_eq!(root.reserved(), 0, "the pool is empty at the end");
    (rows, sinks, files)
}

/// `sql` run fused and as written, with the same rows; returns them and the
/// aggregate sinks of each run.
fn both(env: &Env, sql: &str) -> (Vec<String>, usize, usize) {
    let plan = env.plan(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let fused = rows_of(&plan, &TreePool::new(&MemoryConfig::default()));
    let unfused = plan.with_worker(|w| w.with_fused_aggregates(false));
    let written = rows_of(&unfused, &TreePool::new(&MemoryConfig::default()));
    assert_eq!(fused.0, written.0, "{sql}");
    (fused.0, fused.1, written.1)
}

/// The physical plan of `sql` as `EXPLAIN` prints it.
fn explain(env: &Env, sql: &str) -> String {
    env.plan(sql)
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .explain()
        .to_string()
}

#[test]
fn an_aggregate_over_rows_that_move_has_a_partial_phase() {
    let env = env();
    insta::assert_snapshot!(explain(&env, "SELECT a, count(*), sum(b) FROM t GROUP BY a"), @"
    Project [a#0, count(1)#4611686018427387904 AS count(1)#3, sum(b#1)#4611686018427387905 AS sum(b)#4]
    +- HashAggregate keys=[a#0], functions=[count(1), sum(b#1)]
       +- Exchange hashpartitioning(a#0)
          +- PartialHashAggregate keys=[a#0], functions=[count(1), sum(b#1)]
             +- Scan spark_catalog.default.t [a#0, b#1]
    ");
    insta::assert_snapshot!(explain(&env, "SELECT max(s) FROM t"), @"
    Project [max(s#2)#4611686018427387904 AS max(s)#3]
    +- HashAggregate keys=[], functions=[max(s#2)]
       +- Exchange SinglePartition
          +- PartialHashAggregate keys=[], functions=[max(s#2)]
             +- Scan spark_catalog.default.t [s#2]
    ");
}

/// A `DISTINCT` aggregate over moved rows has Spark's four phases: partial
/// over the keys and the distinct argument, the exchange on the keys, a
/// partial merge removing the pairs several tasks sent, and the final phase.
/// An aggregate whose input is already placed on its keys keeps one node.
#[test]
fn distinct_has_a_partial_merge_and_placed_inputs_keep_one_aggregate() {
    let env = env();
    insta::assert_snapshot!(explain(&env, "SELECT a, count(DISTINCT b), sum(b) FROM t GROUP BY a"), @"
    Project [a#0, count(DISTINCT b#1)#4611686018427387904 AS count(DISTINCT b)#3, sum(b#1)#4611686018427387905 AS sum(b)#4]
    +- HashAggregate keys=[a#0], functions=[count(DISTINCT b#1), sum(b#1)]
       +- PartialMergeHashAggregate keys=[a#0], distinct=[b], functions=[sum(b#1)]
          +- Exchange hashpartitioning(a#0)
             +- PartialHashAggregate keys=[a#0], distinct=[b], functions=[sum(b#1)]
                +- Scan spark_catalog.default.t [a#0, b#1]
    ");
    let (rows, fused, written) = both(
        &env,
        "SELECT a, count(DISTINCT b), sum(b) FROM t GROUP BY a",
    );
    assert_eq!(rows.len(), 9, "{rows:?}");
    assert_eq!((fused, written), (2, 3));
    let (rows, ..) = both(&env, "SELECT count(DISTINCT a), max(s) FROM t");
    assert_eq!(rows, ["8|s9"]);
    let plan = explain(
        &env,
        "SELECT a, count(*) FROM (SELECT a, sum(b) AS b FROM t GROUP BY a) GROUP BY a",
    );
    assert_eq!(plan.matches("PartialHashAggregate").count(), 1, "{plan}");
    assert_eq!(plan.matches("Exchange").count(), 1, "{plan}");
}

#[test]
fn one_worker_runs_one_breaker_and_the_written_plan_two() {
    let env = env();
    let (rows, fused, written) = both(&env, "SELECT a, count(*), sum(b) FROM t GROUP BY a");
    assert_eq!(rows.len(), 9, "{rows:?}");
    assert!(rows.contains(&"NULL|2|100".to_owned()), "{rows:?}");
    assert_eq!((fused, written), (1, 2));
    let (rows, fused, written) = both(&env, "SELECT count(*), max(s) FROM t WHERE a > 100");
    assert_eq!(rows, ["0|NULL"]);
    assert_eq!((fused, written), (1, 2));
}

/// An aggregate over a partitioned join fuses the same way.
#[test]
fn a_partitioned_join_below_the_aggregate_agrees() {
    let env = exchanges_env();
    let (rows, ..) = both(
        &env,
        "SELECT x.a, count(*), sum(y.b), min(y.s) FROM big x JOIN big y ON x.a = y.a GROUP BY x.a",
    );
    assert_eq!(rows.len(), 8, "{rows:?}");
}

/// Large enough to spill under 8 MiB: both runs spill and agree.
#[test]
fn fused_and_written_agree_when_they_spill() {
    let env = env();
    let sql = "SELECT id % 200003 AS k, count(*), sum(id), max(CAST(id AS STRING)) \
               FROM range(600000) GROUP BY id % 200003";
    let dir = tempfile::tempdir().unwrap();
    let scope = || {
        SpillManager::open(&SpillConfig {
            dir: dir.path().join("spill"),
            ..SpillConfig::default()
        })
        .unwrap()
        .scope()
        .unwrap()
    };
    let small = || {
        TreePool::new(&MemoryConfig {
            limit: Some(8 << 20),
            ..MemoryConfig::default()
        })
    };
    let plan = env.plan(sql).unwrap();
    let fused = rows_of(&env.plan(sql).unwrap().with_spill(scope()), &small());
    let written = rows_of(
        &plan
            .with_spill(scope())
            .with_worker(|w| w.with_fused_aggregates(false)),
        &small(),
    );
    assert_eq!(fused.0.len(), 200_003);
    assert_eq!(fused.0, written.0);
    assert!(fused.2 > 0 && written.2 > 0, "{} {}", fused.2, written.2);
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]

    /// Random keys (with NULLs), values and filters: fused and written plans
    /// return the same rows, grouped and global, empty input included.
    #[test]
    fn fused_and_written_plans_agree(
        n in 0_i64..3000,
        groups in 1_i64..400,
        nulls in 2_i64..20,
        keep in 0_i64..3000,
    ) {
        let env = env();
        let rows = format!(
            "SELECT CASE WHEN id % {nulls} = 0 THEN NULL ELSE id % {groups} END AS k, \
                    CASE WHEN id % 7 = 3 THEN NULL ELSE id * 31 % 1009 - 500 END AS v, \
                    CAST(id % 53 AS STRING) AS s \
             FROM range({n}) WHERE id < {keep}"
        );
        both(&env, &format!(
            "SELECT k, count(*), count(v), sum(v), min(s), max(v), \
                    CAST(avg(v) AS STRING), count(v) FILTER (WHERE v > 0), \
                    CAST(first(s) IS NULL AS STRING) \
             FROM ({rows}) GROUP BY k"
        ));
        both(&env, &format!(
            "SELECT k, s, sum(v), CAST(stddev(v) IS NULL AS STRING) FROM ({rows}) GROUP BY k, s"
        ));
        both(&env, &format!(
            "SELECT count(*), sum(v), max(s), CAST(avg(v) AS STRING) FROM ({rows})"
        ));
        both(&env, &format!(
            "SELECT k, count(DISTINCT v), sum(v) FROM ({rows}) GROUP BY k"
        ));
    }
}

/// The fused breaker records the exchange it stands for: every driver's
/// partial rows once, keys estimated, and `EXPLAIN ANALYZE` shows its sink
/// (with those counters) under the partial node and its source under the final one.
#[test]
fn a_fused_aggregate_reports_its_exchange() {
    use krishiv_exec::profile::{QueryProfile, TextOptions};

    let env = env();
    let sql = "SELECT id % 5000 AS k, count(*) FROM range(200000) GROUP BY id % 5000";
    let plan = env.plan(sql).unwrap();
    let root = TreePool::new(&MemoryConfig::default());
    let (out, query) = super::run_in(&plan, &root);
    let rows: usize = out
        .unwrap()
        .iter()
        .map(arrow_array::RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 5000);
    let metrics = query.metrics();
    let sink = metrics
        .operators
        .values()
        .find(|o| o.name == "HashAggregate" && o.role == OperatorRole::Sink)
        .unwrap();
    let count = |n: &str| sink.extra(n).and_then(krishiv_exec::Metric::value).unwrap();
    // Each driver ships each key it met at least once.
    let tasks = count("exchange_map_tasks");
    assert!(tasks >= 1);
    let shipped = count("exchange_rows");
    assert!(
        (5000..=5000 * tasks).contains(&shipped),
        "{shipped} over {tasks}"
    );
    let keys = count("exchange_keys");
    assert!((4500..=5500).contains(&keys), "{keys}");
    assert!(count("exchange_bytes") > 0);
    assert!(count("exchange_max_partition_rows") <= shipped);
    let text = QueryProfile::new(&plan, metrics, false).to_text(TextOptions::masked());
    let partial = text.find("PartialHashAggregate").unwrap();
    let final_ = text.find("HashAggregate keys").unwrap();
    let sink_at = text.find("sink   #").unwrap();
    assert!(final_ < partial && partial < sink_at, "{text}");
    assert!(text.contains("exchange_rows=<n>"), "{text}");
    // The written plan (two breakers) records none.
    let written = plan.with_worker(|w| w.with_fused_aggregates(false));
    let (_, query) = super::run_in(&written, &root);
    assert!(
        query
            .metrics()
            .operators
            .values()
            .all(|o| o.extra("exchange_rows").is_none())
    );
}
