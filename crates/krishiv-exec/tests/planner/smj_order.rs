// SPDX-License-Identifier: Apache-2.0
//! P2-T056 acceptance #6b (amended by the owner, 2026-11-04, `design/P2-T056-6b-output-ordering.md`): an
//! `ORDER BY` on the join key after a sort-merge join keeps its exchange and sort, as in Spark 4.2 (a global
//! `ORDER BY` needs a range partitioning, which drops the join's order). The plan shape is pinned here; the
//! results of the forced sort-merge plans equal the hash join's (TPC-H 22 in `tpch.rs`).

use std::path::Path;
use std::sync::Arc;

use super::tpch::parquet_tables;

/// The planner's text for `sql` with every equi-join a sort-merge join.
fn sort_merge_plan(sql: &str) -> String {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tpch-sf0.01-parquet");
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut tpch = parquet_tables(&dir, &rt);
    let mut config = (*tpch.config).clone();
    config
        .set("krishiv.exec.join.strategy", "sortMerge")
        .unwrap();
    tpch.config = Arc::new(config);
    tpch.plan_sql(sql).unwrap().to_string()
}

/// The line number of the first line containing `needle`.
fn line_of(plan: &str, needle: &str) -> usize {
    plan.lines()
        .position(|l| l.contains(needle))
        .unwrap_or_else(|| panic!("no `{needle}` in\n{plan}"))
}

#[test]
fn order_by_the_join_key_keeps_its_exchange_and_sort() {
    for (kind, join) in [
        ("inner", "JOIN"),
        ("left", "LEFT JOIN"),
        ("full", "FULL JOIN"),
    ] {
        let sql = format!(
            "SELECT o_orderkey, l_linenumber FROM orders {join} lineitem ON o_orderkey = l_orderkey \
             ORDER BY o_orderkey"
        );
        let plan = sort_merge_plan(&sql);
        // Sort above the range exchange above the join, as Spark's `Sort`, `Exchange rangepartitioning`,
        // `SortMergeJoin`.
        let (sort, exchange, merge) = (
            line_of(&plan, "Sort ["),
            line_of(&plan, "Exchange rangepartitioning"),
            line_of(&plan, "SortMergeJoin"),
        );
        assert!(sort < exchange && exchange < merge, "{kind}:\n{plan}");
        insta::assert_snapshot!(format!("order_by_after_{kind}_sort_merge_join"), plan);
    }
}

#[test]
fn without_the_order_by_there_is_no_range_exchange() {
    let plan = sort_merge_plan(
        "SELECT o_orderkey, l_linenumber FROM orders JOIN lineitem ON o_orderkey = l_orderkey",
    );
    assert!(plan.contains("SortMergeJoin"), "{plan}");
    assert!(!plan.contains("rangepartitioning"), "{plan}");
}
