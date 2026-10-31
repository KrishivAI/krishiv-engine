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
