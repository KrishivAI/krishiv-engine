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
        let mut config = Config::new();
        config
            .set("krishiv.optimizer.subqueryAsWindow", "true")
            .unwrap();
        let again = optimize_with_udfs(&on, &fixture(), &UdfLibrary::new(), &config).unwrap();
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

#[test]
fn without_statistics_the_guard_does_not_fire() {
    let plan = support::resolve(&support::tpch_sql(17), &support::ctx());
    let mut config = Config::new();
    config
        .set("krishiv.optimizer.subqueryAsWindow", "true")
        .unwrap();
    let out = optimize_with_udfs(
        &plan,
        &krishiv_optimizer::NoStatistics,
        &UdfLibrary::new(),
        &config,
    )
    .unwrap();
    assert!(!out.root().to_string().contains("Window ["));
}

#[test]
fn avg_of_an_integer_is_refused() {
    // Without `avg_is_exact` the window's average could differ from the aggregate's.
    let sql = q17("avg(l_linenumber)", "", "", "l_partkey = p_partkey");
    assert!(!windowed(&sql));
}

#[test]
fn a_floating_point_key_is_refused() {
    let sql = q17(
        "avg(l_quantity)",
        "",
        "",
        "CAST(l_partkey AS DOUBLE) = CAST(p_partkey AS DOUBLE)",
    );
    assert!(!windowed(&sql));
}

#[test]
fn a_firing_is_noted_in_the_trace() {
    let plan = support::resolve(&support::tpch_sql(17), &support::ctx());
    let mut config = Config::new();
    config
        .set("krishiv.optimizer.subqueryAsWindow", "true")
        .unwrap();
    let (_, steps) = krishiv_optimizer::optimize_traced_with_udfs(
        &plan,
        &fixture(),
        &UdfLibrary::new(),
        &config,
    )
    .unwrap();
    assert!(
        steps
            .iter()
            .any(|s| { s.note.contains("SubqueryAsWindow: form M") }),
    );
}

#[test]
fn an_or_filter_on_the_outer_block_is_refused() {
    // An OR over the subquery's table is not a conjunct the subquery shares.
    let sql = q17(
        "avg(l_quantity)",
        "AND (l_shipdate > DATE '1995-01-01' OR l_discount > 0.05)",
        "",
        "l_partkey = p_partkey",
    );
    assert!(!windowed(&sql));
}

#[test]
fn a_coalesced_subquery_value_is_refused() {
    // `coalesce(.., 0)` gives a group with no match a value the window cannot give.
    let sql = "SELECT sum(l_extendedprice) FROM lineitem, part WHERE p_partkey = l_partkey \
               AND l_quantity < coalesce((SELECT avg(l_quantity) FROM lineitem \
               WHERE l_partkey = p_partkey), 0)";
    assert!(!windowed(sql));
}

#[test]
fn a_second_instance_of_the_table_in_the_outer_block_is_refused() {
    let sql = "SELECT sum(a.l_extendedprice) FROM lineitem a, lineitem b, part \
               WHERE p_partkey = a.l_partkey AND b.l_orderkey = a.l_orderkey \
               AND a.l_quantity < (SELECT avg(l_quantity) FROM lineitem \
               WHERE l_partkey = p_partkey)";
    assert!(!windowed(sql));
}

#[test]
fn a_raising_or_nondeterministic_call_is_refused() {
    for agg in [
        "min(CAST(l_comment AS INT))",
        "max(l_quantity / (l_linenumber - 1))",
    ] {
        let sql = q17(agg, "", "", "l_partkey = p_partkey");
        assert!(!windowed(&sql), "{agg}");
    }
}

#[test]
fn a_raising_expression_around_the_call_is_refused() {
    // The window evaluates it on every outer row, the aggregate once per group.
    let sql = q17(
        "(10 / (max(l_quantity) - 9))",
        "",
        "",
        "l_partkey = p_partkey",
    );
    assert!(!windowed(&sql));
}

#[test]
fn min_and_max_of_a_double_are_refused() {
    // NaN and -0.0 make the extreme depend on the order the rows arrive in.
    for agg in [
        "min(CAST(l_quantity AS DOUBLE))",
        "max(CAST(l_quantity AS DOUBLE))",
    ] {
        let sql = q17(agg, "", "", "l_partkey = p_partkey");
        assert!(!windowed(&sql), "{agg}");
    }
}
