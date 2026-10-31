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
    let reads = format!("{}", plan.plan.explain_all()).matches("SharedRead").count();
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

/// An ordered body keeps its order for the reader.
#[test]
fn an_ordered_body_is_read_in_order() {
    let env = env();
    let (_, rows) = env.rows(
        "WITH c AS (SELECT a, rand() AS r FROM t ORDER BY a DESC LIMIT 4) \
         SELECT a FROM c UNION ALL SELECT a FROM c",
    );
    let mut sorted = rows.clone();
    sorted.sort();
    assert_eq!(sorted.len(), 8, "{rows:?}");
    // Each half (one reference's rows) is in the body's descending order.
    let firsts: Vec<i64> = rows.iter().map(|r| r.parse().unwrap()).collect();
    let runs = firsts.windows(2).filter(|w| w[0] < w[1]).count();
    assert!(runs <= 1, "{rows:?}");
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
    let rows: usize = out.unwrap().iter().map(arrow_array::RecordBatch::num_rows).sum();
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
