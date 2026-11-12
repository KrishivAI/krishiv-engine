// SPDX-License-Identifier: Apache-2.0
//! `SubqueryAsWindow` (P3 W1 shape (a), form M): q17 and q02 become a window over the outer
//! rows, and every refusal of the design note's section 2.2 leaves the plan as it was.

use krishiv_common::config::Config;
use krishiv_optimizer::optimize_with_udfs;
use krishiv_plan::Plan;
use krishiv_udf_sdk::UdfLibrary;

use crate::join_order::fixture;
use crate::support;

fn run(sql: &str, on: bool) -> Plan {
    let plan = support::resolve(sql, &support::ctx());
    let mut config = Config::new();
    if on {
        config
            .set("krishiv.optimizer.subqueryAsWindow", "true")
            .unwrap();
    }
    optimize_with_udfs(&plan, &fixture(), &UdfLibrary::new(), &config).unwrap()
}

/// Whether the key changes the plan.
fn windowed(sql: &str) -> bool {
    !run(sql, true).structurally_eq(&run(sql, false))
}

/// q17 with `agg` as the subquery's aggregate, `extra_outer` and `extra_inner` conjuncts, and the
/// correlation `corr`.
fn q17(agg: &str, extra_outer: &str, extra_inner: &str, corr: &str) -> String {
    format!(
        "SELECT sum(l_extendedprice) / 7.0 AS avg_yearly FROM lineitem, part \
         WHERE p_partkey = l_partkey AND p_brand = 'Brand#23' AND p_container = 'MED BOX' \
         {extra_outer} AND l_quantity < (SELECT 0.2 * {agg} FROM lineitem \
         WHERE {corr} {extra_inner})"
    )
}

fn plain_q17() -> String {
    q17("avg(l_quantity)", "", "", "l_partkey = p_partkey")
}

#[test]
fn q17_and_q02_become_windows_only_with_the_key() {
    for q in [17, 2] {
        let sql = support::tpch_sql(q);
        assert!(
            !run(&sql, false).root().to_string().contains("Window ["),
            "q{q} off"
        );
        let on = run(&sql, true);
        insta::assert_snapshot!(format!("win_magic_q{q}"), on.root().to_string());
        Plan::try_new(std::sync::Arc::clone(on.root())).unwrap();
        // A fixed point.
        let again = run(&sql, true);
        assert!(again.structurally_eq(&on), "q{q}: {}", again.root());
    }
}

#[test]
fn the_plain_variant_fires() {
    assert!(windowed(&plain_q17()));
}

#[test]
fn an_outer_only_filter_on_the_subquery_s_table_is_refused() {
    // The window would aggregate a subset of the group (condition 4).
    let sql = q17(
        "avg(l_quantity)",
        "AND l_shipdate > DATE '1995-01-01'",
        "",
        "l_partkey = p_partkey",
    );
    assert!(!windowed(&sql));
}

#[test]
fn an_outer_filter_the_subquery_shares_fires() {
    let sql = q17(
        "avg(l_quantity)",
        "AND l_shipdate > DATE '1995-01-01'",
        "AND l_shipdate > DATE '1995-01-01'",
        "l_partkey = p_partkey",
    );
    assert!(windowed(&sql));
}

#[test]
fn a_column_equality_the_subquery_lacks_is_refused() {
    let sql = q17(
        "avg(l_quantity)",
        "AND l_suppkey = l_partkey",
        "",
        "l_partkey = p_partkey",
    );
    assert!(!windowed(&sql));
}

#[test]
fn a_non_uniform_join_is_refused_in_form_m() {
    // `p_partkey = l_suppkey` multiplies `S` rows by 0 or m (form J would be needed).
    let sql = q17(
        "avg(l_quantity)",
        "AND p_partkey = l_suppkey",
        "",
        "l_partkey = p_partkey",
    );
    assert!(!windowed(&sql));
}

#[test]
fn calls_form_m_does_not_take_are_refused() {
    for agg in [
        "sum(l_quantity)",
        "count(l_quantity)",
        "avg(DISTINCT l_quantity)",
    ] {
        let sql = q17(agg, "", "", "l_partkey = p_partkey");
        assert!(!windowed(&sql), "{agg}");
    }
}

#[test]
fn a_null_safe_correlation_is_refused() {
    let sql = q17("avg(l_quantity)", "", "", "l_partkey <=> p_partkey");
    assert!(!windowed(&sql));
}

#[test]
fn a_limit_in_the_subquery_is_refused() {
    let sql = "SELECT sum(l_extendedprice) FROM lineitem, part WHERE p_partkey = l_partkey \
               AND l_quantity < (SELECT avg(q) FROM (SELECT l_quantity AS q FROM lineitem \
               WHERE l_partkey = p_partkey LIMIT 5))";
    assert!(!windowed(sql));
}
