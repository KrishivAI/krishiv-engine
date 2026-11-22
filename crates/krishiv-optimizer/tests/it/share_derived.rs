// SPDX-License-Identifier: Apache-2.0
//! An aggregate a derived table computes twice, with different outputs or
//! key filters, is computed once (perf-3 lever D1, TPC-DS q65):
//! `krishiv.optimizer.shareRepeatedSubtrees`, `share_cte/derived.rs`.

use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_optimizer::optimize_with_udfs;
use krishiv_plan::{LogicalPlan, Plan};
use krishiv_udf_sdk::UdfLibrary;

use crate::join_order::fixture;
use crate::support;

fn config(on: bool) -> Config {
    let mut config = Config::new();
    config
        .set(
            "krishiv.optimizer.shareRepeatedSubtrees",
            if on { "true" } else { "false" },
        )
        .unwrap();
    config
}

fn run(sql: &str, on: bool) -> Plan {
    let ctx = support::ctx_over(crate::tpcds::catalog());
    let plan = support::resolve(sql, &ctx);
    optimize_with_udfs(&plan, &fixture(), &UdfLibrary::new(), &config(on)).unwrap()
}

/// How many times the plan computes `sum(ss_sales_price)` (a subtree
/// shared through `Arc` prints once).
fn sums(plan: &Arc<LogicalPlan>) -> usize {
    plan.to_string().matches("sum(ss_sales_price").count()
}

/// Shared once, a valid plan and a fixed point.
fn shared_once(plan: &Plan) -> String {
    let text = plan.root().to_string();
    assert_eq!(sums(plan.root()), 1, "{text}");
    assert_eq!(text.matches("CteRef __shared_0").count(), 2, "{text}");
    Plan::try_new(Arc::clone(plan.root())).unwrap();
    // The first optimization over the shared plan may swap one join's
    // inputs (they tie on estimates, and the canonical order reads the
    // `CteRef` leaves' ids: `join_cost` exempts the shared queries); the
    // plan it gives is a fixed point.
    let again = optimize_with_udfs(plan, &fixture(), &UdfLibrary::new(), &config(true)).unwrap();
    let third = optimize_with_udfs(&again, &fixture(), &UdfLibrary::new(), &config(true)).unwrap();
    assert!(third.structurally_eq(&again), "{}", third.root());
    assert_eq!(
        again.root().resolved_output(),
        plan.root().resolved_output()
    );
    text
}

/// q65: `sc` (a `Shift` of `sa`, with `isnotnull(ss_item_sk)` from the join
/// with `item` and every output) and `sa` under `sb` (no item filter, item
/// pruned) are one aggregate over the weaker input; `sc` keeps its filter
/// over its reference (inside its `Shift`), `sa` a projection.
#[test]
fn q65_computes_the_store_item_revenue_once() {
    let off = run(&crate::tpcds::sql(65), false);
    assert_eq!(sums(off.root()), 2, "{}", off.root());
    assert!(!off.root().to_string().contains("CteRef"));
    let text = shared_once(&run(&crate::tpcds::sql(65), true));
    assert!(text.contains("Filter isnotnull(ss_item_sk#53)\n"), "{text}");
    insta::assert_snapshot!("share_derived_q65", text);
}

const REVENUE: &str = "(SELECT ss_store_sk, ss_item_sk, sum(ss_sales_price) AS revenue \
                       FROM store_sales GROUP BY ss_store_sk, ss_item_sk)";

/// The direct copy is the filtered one (joined with `item`), the `Shift` the
/// narrower; neither has a filter of its own below the aggregate.
#[test]
fn the_direct_copy_may_be_the_filtered_one() {
    let sql = format!(
        "SELECT a.ss_store_sk, a.revenue, i_brand \
         FROM {REVENUE} a JOIN item ON i_item_sk = a.ss_item_sk \
         JOIN (SELECT ss_store_sk, avg(revenue) AS ave FROM {REVENUE} x GROUP BY ss_store_sk) b \
         ON a.ss_store_sk = b.ss_store_sk WHERE a.revenue <= 0.1 * b.ave"
    );
    let off = run(&sql, false);
    assert_eq!(sums(off.root()), 2, "{}", off.root());
    shared_once(&run(&sql, true));
}

/// `ROLLUP` is an `Expand` under a plain `GROUP BY` (with the grouping id as
/// a key): a filter of the keys over the `Expand` still commutes with the
/// aggregate, so the copies share it.
#[test]
fn a_rollup_shares_above_its_expand() {
    let rollup = "(SELECT ss_store_sk, ss_item_sk, sum(ss_sales_price) AS revenue \
                  FROM store_sales GROUP BY ROLLUP(ss_store_sk, ss_item_sk))";
    let sql = format!(
        "SELECT a.ss_store_sk, a.revenue, i_brand \
         FROM {rollup} a JOIN item ON i_item_sk = a.ss_item_sk \
         JOIN (SELECT ss_store_sk, avg(revenue) AS ave FROM {rollup} x GROUP BY ss_store_sk) b \
         ON a.ss_store_sk = b.ss_store_sk WHERE a.revenue <= 0.1 * b.ave"
    );
    shared_once(&run(&sql, true));
}

/// Each copy reads an output the other does not (`revenue`, `n`) over the same input: the
/// definition outputs both.
#[test]
fn the_definition_outputs_both_lists() {
    let both = "(SELECT ss_store_sk, ss_item_sk, sum(ss_sales_price) AS revenue, max(ss_sales_price) AS n \
                FROM store_sales GROUP BY ss_store_sk, ss_item_sk)";
    let sql = format!(
        "SELECT a.ss_store_sk, a.revenue, i_brand \
         FROM {both} a JOIN item ON i_item_sk = a.ss_item_sk \
         JOIN (SELECT ss_store_sk, avg(n) AS ave FROM {both} x GROUP BY ss_store_sk) b \
         ON a.ss_store_sk = b.ss_store_sk WHERE a.revenue <= 0.1 * b.ave"
    );
    let text = shared_once(&run(&sql, true));
    assert!(text.contains("max(ss_sales_price#13) AS n#24"), "{text}");
}

/// A key filter the join with `item` adds below an outer join is not
/// followed (only filters, projections and inner joins are): no pair.
#[test]
fn a_filter_below_an_outer_join_is_not_followed() {
    let outer = "(SELECT ss_store_sk, ss_item_sk, sum(ss_sales_price) AS revenue \
                 FROM store_sales LEFT JOIN date_dim ON ss_sold_date_sk = d_date_sk \
                 GROUP BY ss_store_sk, ss_item_sk)";
    let sql = format!(
        "SELECT a.ss_store_sk, a.revenue, i_brand \
         FROM {outer} a JOIN item ON i_item_sk = a.ss_item_sk \
         JOIN (SELECT ss_store_sk, avg(revenue) AS ave FROM {outer} x GROUP BY ss_store_sk) b \
         ON a.ss_store_sk = b.ss_store_sk WHERE a.revenue <= 0.1 * b.ave"
    );
    let on = run(&sql, true);
    assert_eq!(sums(on.root()), 2, "{}", on.root());
}

/// Both copies read every output (`b.m` reads `ss_item_sk` through `max`):
/// the definition is the unfiltered aggregate itself, the filtered copy reads
/// it under its filter with no projection.
#[test]
fn equal_outputs_share_the_unfiltered_aggregate() {
    let sql = format!(
        "SELECT a.ss_store_sk, a.revenue, i_brand, b.m \
         FROM {REVENUE} a JOIN item ON i_item_sk = a.ss_item_sk \
         JOIN (SELECT ss_store_sk, avg(revenue) AS ave, max(ss_item_sk) AS m FROM {REVENUE} x \
         GROUP BY ss_store_sk) b \
         ON a.ss_store_sk = b.ss_store_sk WHERE a.revenue <= 0.1 * b.ave"
    );
    let text = shared_once(&run(&sql, true));
    assert!(
        text.contains("Filter isnotnull(ss_item_sk#2)\n            :  +- CteRef __shared_0"),
        "{text}"
    );
    assert!(
        !text.contains("Project [ss_store_sk#7, ss_item_sk#2, revenue#23]\n"),
        "{text}"
    );
}

/// A group key that is not a bare column (`ss_quantity % 2`) is grouped
/// by and output, and the copies still share.
#[test]
fn a_computed_group_key_still_shares() {
    let keyed = "(SELECT ss_store_sk, ss_item_sk, ss_quantity % 2 AS q, sum(ss_sales_price) AS revenue \
                 FROM store_sales GROUP BY ss_store_sk, ss_item_sk, ss_quantity % 2)";
    let sql = format!(
        "SELECT a.ss_store_sk, a.revenue, a.q, i_brand \
         FROM {keyed} a JOIN item ON i_item_sk = a.ss_item_sk \
         JOIN (SELECT ss_store_sk, avg(revenue) AS ave FROM {keyed} x GROUP BY ss_store_sk) b \
         ON a.ss_store_sk = b.ss_store_sk WHERE a.revenue <= 0.1 * b.ave"
    );
    let text = shared_once(&run(&sql, true));
    assert!(text.contains("(ss_quantity#10 % 2) AS q#23"), "{text}");
}

/// The derived table written four times (twice in each `UNION ALL` branch):
/// the two direct copies and the `Shift` body share one definition.
#[test]
fn a_derived_table_written_more_than_twice_is_computed_once() {
    let branch = |call: &str, cmp: &str| {
        format!(
            "SELECT a.ss_store_sk, a.revenue, i_brand \
             FROM {REVENUE} a JOIN item ON i_item_sk = a.ss_item_sk \
             JOIN (SELECT ss_store_sk, {call}(revenue) AS ave FROM {REVENUE} x GROUP BY ss_store_sk) b \
             ON a.ss_store_sk = b.ss_store_sk WHERE a.revenue {cmp} b.ave"
        )
    };
    let sql = format!(
        "{} UNION ALL {}",
        branch("avg", "<= 0.1 *"),
        branch("max", ">= 0.9 *")
    );
    let off = run(&sql, false);
    assert_eq!(sums(off.root()), 3, "{}", off.root());
    let on = run(&sql, true);
    let text = on.root().to_string();
    assert_eq!(sums(on.root()), 1, "{text}");
    assert_eq!(text.matches("CteRef __shared_0").count(), 3, "{text}");
    Plan::try_new(Arc::clone(on.root())).unwrap();
    let again = optimize_with_udfs(&on, &fixture(), &UdfLibrary::new(), &config(true)).unwrap();
    let third = optimize_with_udfs(&again, &fixture(), &UdfLibrary::new(), &config(true)).unwrap();
    assert!(third.structurally_eq(&again), "{}", third.root());
}
