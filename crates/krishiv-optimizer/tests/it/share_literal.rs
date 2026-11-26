// SPDX-License-Identifier: Apache-2.0
//! `ShareLiteralInstances`: instances of one CTE that differ only in the
//! literal a group key is compared with read one definition over all the
//! literals (`krishiv.optimizer.shareLiteralInstances`, TPC-DS q31, q75).

use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_optimizer::{NoStatistics, optimize_with_udfs};
use krishiv_plan::{LogicalPlan, Plan, PlanNode};
use krishiv_udf_sdk::UdfLibrary;

use crate::{support, tpcds};

fn config(on: bool) -> Config {
    let mut c = Config::new();
    c.set(
        "krishiv.optimizer.shareLiteralInstances",
        if on { "true" } else { "false" },
    )
    .unwrap();
    c
}

fn optimize(sql: &str, on: bool) -> Plan {
    let plan = support::resolve(sql, &support::ctx_over(tpcds::catalog()));
    let out = optimize_with_udfs(&plan, &NoStatistics, &UdfLibrary::new(), &config(on)).unwrap();
    assert_eq!(
        out.root().resolved_output(),
        plan.root().resolved_output(),
        "{sql}"
    );
    out
}

/// How many times `table` is scanned when the plan runs: paths from the root,
/// each `WITH` definition counted once (a `CteRef` computes nothing).
fn scans(plan: &Arc<LogicalPlan>, table: &str) -> usize {
    let prefix = format!("Scan spark_catalog.default.{table} ");
    fn go(p: &Arc<LogicalPlan>, prefix: &str) -> usize {
        let mut n =
            usize::from(matches!(&p.node, PlanNode::Scan(_) if p.to_string().starts_with(prefix)));
        p.for_each_child(&mut |c| n += go(c, prefix));
        n
    }
    go(plan, &prefix)
}

/// The plan is valid and a fixed point.
fn fixed_point(on: &Plan) {
    Plan::try_new(Arc::clone(on.root())).unwrap();
    let again = optimize_with_udfs(on, &NoStatistics, &UdfLibrary::new(), &config(true)).unwrap();
    assert!(again.structurally_eq(on), "{}", again.root());
}

/// q31 reads `ss` and `ws` three times each, with `d_qoy = 1`, `2`, `3`:
/// each CTE is computed once over `d_qoy IN (1, 2, 3)`.
#[test]
fn q31_scans_the_facts_once() {
    let sql = tpcds::sql(31);
    let off = optimize(&sql, false);
    assert_eq!(scans(off.root(), "store_sales"), 3, "{}", off.root());
    assert_eq!(scans(off.root(), "web_sales"), 3, "{}", off.root());

    let on = optimize(&sql, true);
    let text = on.root().to_string();
    assert_eq!(scans(on.root(), "store_sales"), 1, "{text}");
    assert_eq!(scans(on.root(), "web_sales"), 1, "{text}");
    assert!(text.contains("d_qoy#"), "{text}");
    assert!(text.contains("IN (1, 2, 3)"), "{text}");
    fixed_point(&on);
    insta::assert_snapshot!("share_literal_q31", text);
}

/// q75 reads `all_sales` (a distinct union of three channels under an
/// aggregate) for `d_year = 2002` and `2001`: one computation over both.
#[test]
fn q75_reads_the_channels_once() {
    let sql = tpcds::sql(75);
    let off = optimize(&sql, false);
    let on = optimize(&sql, true);
    for t in ["catalog_sales", "store_sales", "web_sales"] {
        assert_eq!(scans(off.root(), t), 2, "{t}: {}", off.root());
        assert_eq!(scans(on.root(), t), 1, "{t}: {}", on.root());
    }
    fixed_point(&on);
}

/// q39 reads `inventory` for `d_moy = 1` and `2`, each reader with its own
/// filter (`cov > 1`) above the rule's `d_moy = k`: the two merge in the same
/// run (the Tidy batch after the rule), so one optimization is a fixed point.
#[test]
fn q39_readers_filters_merge_in_one_run() {
    let sql = tpcds::sql(39);
    let off = optimize(&sql, false);
    let on = optimize(&sql, true);
    assert_eq!(scans(off.root(), "inventory"), 2, "{}", off.root());
    assert_eq!(scans(on.root(), "inventory"), 1, "{}", on.root());
    fixed_point(&on);
}

const BY_KEY: &str = "WITH c AS (SELECT ss_store_sk, ss_quantity, sum(ss_net_paid) AS paid \
     FROM store_sales GROUP BY ss_store_sk, ss_quantity) \
     SELECT a.ss_store_sk, a.paid, b.paid FROM c a, c b \
     WHERE a.ss_store_sk = b.ss_store_sk AND a.ss_quantity = 1 AND b.ss_quantity = 2";

/// Two instances with different literals on a group key are enough.
#[test]
fn two_literals_share() {
    let off = optimize(BY_KEY, false);
    assert_eq!(scans(off.root(), "store_sales"), 2, "{}", off.root());
    let on = optimize(BY_KEY, true);
    assert_eq!(scans(on.root(), "store_sales"), 1, "{}", on.root());
    fixed_point(&on);
}

/// A literal on a column the aggregate does not group by cannot move above
/// it: each instance keeps its own computation.
#[test]
fn a_literal_on_a_non_key_keeps_the_instances() {
    let sql = "WITH c AS (SELECT ss_store_sk, ss_quantity, ss_net_paid FROM store_sales) \
         SELECT * FROM \
         (SELECT ss_store_sk, sum(ss_net_paid) AS p FROM c WHERE ss_quantity = 1 \
          GROUP BY ss_store_sk) a, \
         (SELECT ss_store_sk, sum(ss_net_paid) AS p FROM c WHERE ss_quantity = 2 \
          GROUP BY ss_store_sk) b WHERE a.ss_store_sk = b.ss_store_sk";
    let on = optimize(sql, true);
    assert_eq!(scans(on.root(), "store_sales"), 2, "{}", on.root());
}

/// A group key computed by an expression is not followed: the literal
/// compares the expression's value, which the rule does not trace.
#[test]
fn an_expression_key_keeps_the_instances() {
    let sql = "WITH c AS (SELECT ss_store_sk, ss_quantity + 1 AS q, sum(ss_net_paid) AS paid \
         FROM store_sales GROUP BY ss_store_sk, ss_quantity + 1) \
         SELECT a.ss_store_sk, a.paid, b.paid FROM c a, c b \
         WHERE a.ss_store_sk = b.ss_store_sk AND a.q = 1 AND b.q = 2";
    let on = optimize(sql, true);
    assert_eq!(scans(on.root(), "store_sales"), 2, "{}", on.root());
}

/// On by default; `false` keeps the instances.
#[test]
fn on_by_default() {
    let plan = support::resolve(BY_KEY, &support::ctx_over(tpcds::catalog()));
    let out = optimize_with_udfs(&plan, &NoStatistics, &UdfLibrary::new(), &Config::new()).unwrap();
    assert_eq!(scans(out.root(), "store_sales"), 1, "{}", out.root());
    let off = optimize(BY_KEY, false);
    assert_eq!(scans(off.root(), "store_sales"), 2, "{}", off.root());
}

/// An aggregate only one instance reads would run on the other literal's
/// groups: a `sum` (which may overflow under ANSI there) keeps both scans.
#[test]
fn a_sum_only_one_instance_reads_keeps_the_instances() {
    let sql = "WITH c AS (SELECT ss_store_sk, ss_quantity, sum(ss_net_paid) AS s, \
         count(ss_net_paid) AS n FROM store_sales GROUP BY ss_store_sk, ss_quantity) \
         SELECT a.ss_store_sk, a.s, b.n FROM c a, c b \
         WHERE a.ss_store_sk = b.ss_store_sk AND a.ss_quantity = 1 AND b.ss_quantity = 2";
    let on = optimize(sql, true);
    assert_eq!(scans(on.root(), "store_sales"), 2, "{}", on.root());
}

/// Extra items that cannot raise (`count`, `min`, `max` over columns) are
/// computed for every literal: the instances share.
#[test]
fn a_count_or_max_only_one_instance_reads_still_shares() {
    let sql = "WITH c AS (SELECT ss_store_sk, ss_quantity, sum(ss_net_paid) AS s, \
         count(ss_net_paid) AS n, max(ss_net_paid) AS m FROM store_sales \
         GROUP BY ss_store_sk, ss_quantity) \
         SELECT a.ss_store_sk, a.s, b.s, b.n, b.m FROM c a, c b \
         WHERE a.ss_store_sk = b.ss_store_sk AND a.ss_quantity = 1 AND b.ss_quantity = 2";
    let on = optimize(sql, true);
    assert_eq!(scans(on.root(), "store_sales"), 1, "{}", on.root());
    fixed_point(&on);
}

/// A user CTE already named `__shared_lit_0` (kept as a `WITH`: it is
/// nondeterministic and read twice) keeps its name; the rule's definition
/// takes the next free one.
#[test]
fn a_user_cte_named_like_the_rule_s_keeps_its_name() {
    let sql = "WITH __shared_lit_0 AS (SELECT rand() AS r), \
         c AS (SELECT ss_store_sk, ss_quantity, sum(ss_net_paid) AS paid \
         FROM store_sales GROUP BY ss_store_sk, ss_quantity) \
         SELECT a.ss_store_sk, a.paid, b.paid, x.r, y.r FROM c a, c b, __shared_lit_0 x, \
         __shared_lit_0 y \
         WHERE a.ss_store_sk = b.ss_store_sk AND a.ss_quantity = 1 AND b.ss_quantity = 2";
    let on = optimize(sql, true);
    let text = on.root().to_string();
    assert_eq!(scans(on.root(), "store_sales"), 1, "{text}");
    assert!(text.contains("CteRef __shared_lit_0 [r#"), "{text}");
    assert!(
        text.contains("CteRef __shared_lit_1 [ss_store_sk#"),
        "{text}"
    );
}

/// TPC-DS q11, q74 (two channels) and q04 (three): `year_total` is a
/// `UNION ALL` of channel aggregates tagged by a constant `sale_type`, read
/// once per channel and year. Each reference's other channels fold to
/// `false` and are dropped (`DropEmptyUnionInputs`), so the two years of a
/// channel share one definition: each fact table and `customer` per channel
/// are scanned once.
#[test]
fn year_over_year_scans_each_fact_once() {
    for (q, facts) in [
        (11, &["store_sales", "web_sales"][..]),
        (74, &["store_sales", "web_sales"][..]),
        (4, &["store_sales", "web_sales", "catalog_sales"][..]),
    ] {
        let sql = tpcds::sql(q);
        let off = optimize(&sql, false);
        let on = optimize(&sql, true);
        for t in facts {
            assert_eq!(scans(off.root(), t), 2, "q{q} {t}: {}", off.root());
            assert_eq!(scans(on.root(), t), 1, "q{q} {t}: {}", on.root());
        }
        assert_eq!(
            scans(off.root(), "customer"),
            2 * facts.len(),
            "q{q}: {}",
            off.root()
        );
        assert_eq!(
            scans(on.root(), "customer"),
            facts.len(),
            "q{q}: {}",
            on.root()
        );
        // q74 included: a second run's join ordering estimates a reader as
        // the instance it replaced, not as the whole two-year definition
        // (design P3-perf031-FA, open item 1).
        fixed_point(&on);
    }
}
