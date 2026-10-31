// SPDX-License-Identifier: Apache-2.0
//! Nondeterministic CTEs are computed once per query and read in full by each reference (P2-T198 unit 2).

use super::env;

/// The self-join of a `rand()` CTE: both sides read the same rows and values.
#[test]
fn both_references_of_a_rand_cte_see_the_same_values() {
    let env = env();
    let sql = "WITH c AS (SELECT a, rand() AS r FROM t) \
               SELECT x.a, x.r = y.r FROM c x JOIN c y ON x.a = y.a";
    let plan = env.plan(sql).unwrap();
    assert_eq!(plan.plan.shared().len(), 1, "{}", plan.plan.explain_all());
    let reads = format!("{}", plan.plan.explain_all())
        .matches("SharedRead")
        .count();
    assert_eq!(reads, 2);
    let (_, rows) = env.rows(sql);
    // Rows with a NULL key do not join: the rest all match themselves.
    assert!(!rows.is_empty());
    assert!(rows.iter().all(|r| r.ends_with("|true")), "{rows:?}");
}

/// A `UNION ALL` of two references holds each value twice, not twenty different ones.
#[test]
fn a_union_of_two_references_repeats_the_same_rows() {
    let env = env();
    let (_, rows) = env.rows(
        "WITH c AS (SELECT rand() AS r FROM t) \
         SELECT count(*), count(DISTINCT r) FROM (SELECT r FROM c UNION ALL SELECT r FROM c)",
    );
    assert_eq!(rows, vec!["20|10"]);
}

/// A reference in a scalar subquery and one in the main query read one body.
#[test]
fn a_subquery_and_the_main_query_read_one_body() {
    let env = env();
    let (_, rows) = env.rows(
        "WITH c AS (SELECT rand() AS r FROM t) \
         SELECT count(*) FROM c WHERE r IN (SELECT r FROM c)",
    );
    assert_eq!(rows, vec!["10"]);
}

/// An ordered body keeps its order for the reader: both references see the same rows, each in the body's order.
#[test]
fn an_ordered_body_is_read_in_order() {
    let env = env();
    let (_, rows) = env.rows(
        "WITH c AS (SELECT rand() AS r FROM t ORDER BY r LIMIT 5) \
         SELECT CAST(r AS STRING) FROM c UNION ALL SELECT CAST(r AS STRING) FROM c",
    );
    assert_eq!(rows.len(), 10, "{rows:?}");
    let values: Vec<f64> = rows.iter().map(|r| r.parse().unwrap()).collect();
    let (first, second) = values.split_at(5);
    assert_eq!(
        first, second,
        "both references read the same list: {rows:?}"
    );
    assert!(first.windows(2).all(|w| w[0] <= w[1]), "{rows:?}");
}

/// Under a small memory limit the shared list spills (the readers continue from the files) and every reader
/// still gets every row; the pool is empty at the end.
#[test]
fn a_spilled_shared_list_is_read_in_full_by_each_reader() {
    use krishiv_memory::{MemoryConfig, MemoryPool as _, SpillConfig, SpillManager, TreePool};

    let env = env();
    let dir = tempfile::tempdir().unwrap();
    let scope = SpillManager::open(&SpillConfig {
        dir: dir.path().join("spill"),
        ..SpillConfig::default()
    })
    .unwrap()
    .scope()
    .unwrap();
    let sql = "WITH c AS (SELECT id, rand() AS r FROM range(300000)) \
               SELECT id FROM c UNION ALL SELECT id FROM c";
    let plan = env.plan(sql).unwrap().with_spill(scope);
    let root = TreePool::new(&MemoryConfig {
        limit: Some(6 << 20),
        ..MemoryConfig::default()
    });
    let (out, query) = super::run_in(&plan, &root);
    let rows: usize = out
        .unwrap()
        .iter()
        .map(arrow_array::RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 600_000);
    let files: u64 = query
        .metrics()
        .operators
        .values()
        .map(|o| o.metrics.spill_files)
        .sum();
    assert!(files > 0, "the shared list did not spill");
    assert_eq!(root.reserved(), 0, "the pool is empty at the end");
}

/// A failing body fails the query for every reader, and the pool is empty afterwards.
#[test]
fn a_failing_shared_body_fails_the_query_and_releases_its_memory() {
    use krishiv_memory::{MemoryConfig, MemoryPool as _, TreePool};

    let env = env();
    let sql = "WITH c AS (SELECT id, rand() AS r, 1 / (id - 5) AS q FROM range(10)) \
               SELECT x.id FROM c x JOIN c y ON x.id = y.id";
    let plan = env.plan(sql).unwrap();
    let root = TreePool::new(&MemoryConfig::default());
    let (out, _query) = super::run_in(&plan, &root);
    assert!(out.is_err(), "division by zero must fail the query");
    assert_eq!(root.reserved(), 0, "the pool is empty after the failure");
}

// -------------------------------------------------------------- deterministic instances (unit 3)

/// `sql` optimized as a session does (with a configuration: `ShareCte` runs), planned and run.
fn sharing(env: &super::Env, sql: &str, conf: &[(&str, &str)]) -> (usize, Vec<String>) {
    use std::sync::Arc;

    use krishiv_common::config::Config;
    use krishiv_exec::planner::{DefaultPlanner, PhysicalPlanner as _, PlanningContext};
    use krishiv_optimizer::NoStatistics;
    use krishiv_udf_sdk::UdfLibrary;

    let mut config = Config::new();
    for (k, v) in conf {
        config.set(k, v).unwrap();
    }
    let analyzed = env.analyze(sql);
    let plan = krishiv_optimizer::optimize_with_udfs(
        &analyzed,
        &NoStatistics,
        &UdfLibrary::new(),
        &config,
    )
    .unwrap();
    let cx = PlanningContext::new(Arc::clone(&env.tables) as _, Arc::new(config));
    let runnable = env.runnable(DefaultPlanner.plan(&plan, &cx).unwrap());
    let shared = runnable.plan.shared().len();
    let batches = super::run(&runnable).unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| super::cell(c.as_ref(), r))
                .collect();
            rows.push(cells.join("|"));
        }
    }
    rows.sort();
    (shared, rows)
}

const GROUPED: &str = "WITH c AS (SELECT a % 3 AS k, count(*) AS n FROM t GROUP BY a % 3) ";

/// Two equal instances of an aggregate body are computed once.
#[test]
fn equal_instances_with_a_breaker_are_shared() {
    let env = env();
    let sql = format!("{GROUPED} SELECT k, n FROM c UNION ALL SELECT k, n FROM c");
    let (shared, rows) = sharing(&env, &sql, &[]);
    assert_eq!(shared, 1);
    let (copies, same) = sharing(
        &env,
        &sql,
        &[("krishiv.sql.optimizer.shareCteInstances", "false")],
    );
    assert_eq!(copies, 0, "the switch keeps the copies");
    assert_eq!(rows, same, "sharing changes no result");
}

/// A group whose member receives a runtime filter from a join outside it keeps its copies.
#[test]
fn a_member_that_receives_a_runtime_filter_is_not_shared() {
    let env = env();
    let sql = "WITH c AS (SELECT a AS k, count(*) AS n FROM t GROUP BY a) \
               SELECT s.a, c1.n FROM (SELECT a FROM t WHERE a = 1) s JOIN c c1 ON s.a = c1.k \
               UNION ALL SELECT k, n FROM c";
    let (shared, rows) = sharing(&env, sql, &[]);
    let (copies, same) = sharing(
        &env,
        sql,
        &[("krishiv.sql.optimizer.shareCteInstances", "false")],
    );
    assert_eq!(copies, 0);
    assert_eq!(rows, same);
    assert_eq!(shared, 0, "a filter from the join reaches a copy");
}

/// Groups nested in another group's body are found too.
#[test]
fn nested_groups_are_shared_inside_the_outer_definition() {
    let env = env();
    let sql = "WITH a AS (SELECT a % 3 AS k, count(*) AS n FROM t GROUP BY a % 3), \
                    b AS (SELECT k, sum(n) AS s FROM a GROUP BY k) \
               SELECT k, s FROM b UNION ALL SELECT k, s FROM b UNION ALL SELECT k, n FROM a";
    let (shared, rows) = sharing(&env, sql, &[]);
    let (_, same) = sharing(
        &env,
        sql,
        &[("krishiv.sql.optimizer.shareCteInstances", "false")],
    );
    assert_eq!(rows, same);
    assert!(shared >= 1);
}

/// A body without a pipeline breaker is not shared (rescanning is cheaper than buffering).
#[test]
fn a_scan_only_body_is_not_shared() {
    let env = env();
    let (shared, _) = sharing(
        &env,
        "WITH c AS (SELECT a FROM t WHERE a > 1) SELECT a FROM c UNION ALL SELECT a FROM c",
        &[],
    );
    assert_eq!(shared, 0);
}
