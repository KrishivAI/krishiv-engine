// SPDX-License-Identifier: Apache-2.0
//! Shape (b) of P3 W1: a subtree the main block reads and a subquery reads
//! again is computed once (`krishiv.optimizer.shareRepeatedSubtrees`).

use krishiv_common::config::Config;
use krishiv_optimizer::optimize_with_udfs;
use krishiv_plan::Plan;
use krishiv_udf_sdk::UdfLibrary;

use crate::join_order::fixture;
use crate::support;

fn q11(conf: &[(&str, &str)]) -> Plan {
    let plan = support::resolve(&support::tpch_sql(11), &support::ctx());
    let mut config = Config::new();
    for (k, v) in conf {
        config.set(k, v).unwrap();
    }
    optimize_with_udfs(&plan, &fixture(), &UdfLibrary::new(), &config).unwrap()
}

/// With the key off, q11 keeps the pre-W1 plan (two evaluations of the join).
#[test]
fn q11_is_unchanged_without_the_key() {
    let plan = q11(&[("krishiv.optimizer.shareRepeatedSubtrees", "false")]);
    assert!(
        !plan.root().to_string().contains("CteRef"),
        "{}",
        plan.root()
    );
}

/// On, the join under the main aggregate and the subquery's is one definition read twice.
#[test]
fn q11_shares_the_join_the_subquery_rereads() {
    let plan = q11(&[("krishiv.optimizer.shareRepeatedSubtrees", "true")]);
    let text = plan.root().to_string();
    assert_eq!(text.matches("CteRef").count(), 2, "{text}");
    assert_eq!(
        text.matches("Scan spark_catalog.default.partsupp").count(),
        1,
        "{text}"
    );
    insta::assert_snapshot!("share_subtree_q11", text);
    // `CteRef` with `output == source` is a valid plan.
    Plan::try_new(std::sync::Arc::clone(plan.root())).unwrap();
    // A fixed point.
    let again = optimize_with_udfs(&plan, &fixture(), &UdfLibrary::new(), &{
        let mut c = Config::new();
        c.set("krishiv.optimizer.shareRepeatedSubtrees", "true")
            .unwrap();
        c
    })
    .unwrap();
    assert!(again.structurally_eq(&plan), "{}", again.root());
}

fn run(sql: &str, on: bool) -> Plan {
    let plan = support::resolve(sql, &support::ctx());
    let mut config = Config::new();
    config
        .set(
            "krishiv.optimizer.shareRepeatedSubtrees",
            if on { "true" } else { "false" },
        )
        .unwrap();
    optimize_with_udfs(&plan, &fixture(), &UdfLibrary::new(), &config).unwrap()
}

/// A subquery that rereads a join the main block also reads (`direct_members`: a body reached
/// both through a `Shift` and directly) shares it only with the key on, and the result is a
/// valid fixed point.
#[test]
fn a_reread_join_is_shared_once_and_only_with_the_key() {
    let sql = "SELECT ps_partkey, sum(ps_supplycost * ps_availqty) AS v \
               FROM partsupp JOIN supplier ON ps_suppkey = s_suppkey GROUP BY ps_partkey \
               HAVING sum(ps_supplycost * ps_availqty) > \
                 (SELECT sum(ps_supplycost * ps_availqty) * 0.001 \
                  FROM partsupp JOIN supplier ON ps_suppkey = s_suppkey)";
    let off = run(sql, false);
    assert!(!off.root().to_string().contains("CteRef"), "{}", off.root());
    let on = run(sql, true);
    let text = on.root().to_string();
    assert_eq!(
        text.matches("Scan spark_catalog.default.partsupp").count(),
        1,
        "{text}"
    );
    Plan::try_new(std::sync::Arc::clone(on.root())).unwrap();
    assert!(run(sql, true).structurally_eq(&on));
}

/// Two different joins are not shared.
#[test]
fn different_subtrees_are_not_shared() {
    let sql = "SELECT ps_partkey FROM partsupp JOIN supplier ON ps_suppkey = s_suppkey \
               WHERE ps_supplycost > (SELECT avg(ps_supplycost) FROM partsupp WHERE ps_availqty > 5)";
    let text = run(sql, true).root().to_string();
    assert!(!text.contains("CteRef"), "{text}");
}

/// TPC-H q15 reads the `revenue0` body twice (the join and the `max` subquery). The join's
/// inferred `isnotnull(l_suppkey)` stays off the shared body, so both reads stay one
/// definition and `lineitem` is scanned once (the #598 regression: the inferred filter made
/// the two copies differ and each was computed).
#[test]
fn q15_computes_the_revenue_body_once() {
    let plan = run(&support::tpch_sql(15), true);
    let text = plan.root().to_string();
    assert!(text.starts_with("WithCte"), "{text}");
    assert_eq!(
        text.matches("Scan spark_catalog.default.lineitem").count(),
        1,
        "{text}"
    );
}

/// How many times `table` is scanned when the plan runs: paths from the root, with each `WITH`
/// definition counted once (a `CteRef` computes nothing). q47 and q57 hold no subquery.
fn scans(plan: &std::sync::Arc<krishiv_plan::LogicalPlan>, table: &str) -> usize {
    use krishiv_plan::PlanNode;
    let mut n = 0;
    if let PlanNode::Scan(_) = &plan.node
        && plan.to_string().starts_with(&format!("Scan {table} "))
    {
        n += 1;
    }
    plan.for_each_child(&mut |c| n += scans(c, table));
    n
}

fn tpcds(q: usize) -> Plan {
    let plan = support::resolve(
        &crate::tpcds::sql(q),
        &support::ctx_over(crate::tpcds::catalog()),
    );
    optimize_with_udfs(
        &plan,
        &krishiv_optimizer::NoStatistics,
        &UdfLibrary::new(),
        &Config::new(),
    )
    .unwrap()
}

/// TPC-DS q47 and q57 read `v1` three times; the main instance has the outer filter pushed in
/// above the windows, so its body is not equal to the lag/lead instances'. The windowed
/// aggregate below that filter is the same, and it is computed once (TPC-DS q47, q57).
#[test]
fn an_instance_with_its_own_filter_shares_the_common_subtree() {
    for (q, fact) in [(47, "store_sales"), (57, "catalog_sales")] {
        let plan = tpcds(q);
        let table = format!("spark_catalog.default.{fact}");
        assert_eq!(scans(plan.root(), &table), 1, "q{q}:\n{}", plan.root());
        assert_eq!(
            plan.root().resolved_output(),
            support::resolve(
                &crate::tpcds::sql(q),
                &support::ctx_over(crate::tpcds::catalog())
            )
            .root()
            .resolved_output()
        );
        Plan::try_new(std::sync::Arc::clone(plan.root())).unwrap();
        let again = optimize_with_udfs(
            &plan,
            &krishiv_optimizer::NoStatistics,
            &UdfLibrary::new(),
            &Config::new(),
        )
        .unwrap();
        assert!(
            again.structurally_eq(&plan),
            "q{q} is a fixed point:\n{}",
            again.root()
        );
    }
}

/// The same shape over TPC-H: one instance keeps a filter on a window column, the other reads
/// the CTE whole. The window over the aggregate is one definition, and the plan is a fixed point.
#[test]
fn a_filtered_instance_reads_the_shared_window() {
    let sql = "WITH v AS (SELECT o_custkey, o_orderstatus, sum(o_totalprice) AS s, \
                 avg(sum(o_totalprice)) OVER (PARTITION BY o_custkey) AS a \
                 FROM orders GROUP BY o_custkey, o_orderstatus) \
               SELECT x.o_custkey, x.s, y.s FROM v x JOIN v y \
                 ON x.o_custkey = y.o_custkey AND x.o_orderstatus = y.o_orderstatus WHERE x.a > 10";
    let plan = run(sql, false);
    assert_eq!(
        scans(plan.root(), "spark_catalog.default.orders"),
        1,
        "{}",
        plan.root()
    );
    Plan::try_new(std::sync::Arc::clone(plan.root())).unwrap();
    assert!(run(sql, false).structurally_eq(&plan));
    let again = optimize_with_udfs(&plan, &fixture(), &UdfLibrary::new(), &Config::new()).unwrap();
    assert!(
        again.structurally_eq(&plan),
        "a fixed point:\n{}",
        again.root()
    );
    // Off, both instances compute the window.
    let mut off = Config::new();
    off.set("krishiv.sql.optimizer.shareCteInstances", "false")
        .unwrap();
    let unshared = optimize_with_udfs(
        &support::resolve(sql, &support::ctx()),
        &fixture(),
        &UdfLibrary::new(),
        &off,
    )
    .unwrap();
    assert_eq!(scans(unshared.root(), "spark_catalog.default.orders"), 2);
}

/// Equal bodies whose filter above the shared aggregate is nondeterministic do not share that
/// filter: the aggregate is one definition and each instance evaluates its own `rand()`, as the
/// copies do (`ShareCte` never shares a nondeterministic part).
#[test]
fn a_nondeterministic_chain_over_a_shared_core_stays_per_instance() {
    let sql = "WITH a AS (SELECT o_custkey k, count(*) n FROM orders GROUP BY o_custkey \
                 HAVING rand() < 0.5), b AS (SELECT k, n FROM a) \
               SELECT x.k, x.n, y.n FROM b x JOIN b y ON x.k = y.k";
    let plan = run(sql, false);
    let text = plan.root().to_string();
    assert_eq!(
        scans(plan.root(), "spark_catalog.default.orders"),
        1,
        "{text}"
    );
    // The definition is the aggregate alone; the `rand()` filter stays above its `CteRef` in
    // the instances (the physical planner plans each instance's path apart).
    let krishiv_plan::PlanNode::WithCte { ctes, .. } = &plan.root().node else {
        panic!("the aggregate is shared:\n{text}");
    };
    assert_eq!(ctes.len(), 1, "{text}");
    let def = ctes[0].plan.to_string();
    assert!(def.starts_with("Aggregate"), "{def}");
    assert!(!def.contains("rand("), "{def}");
    assert!(text.contains("rand() < 0.5D"), "{text}");
    Plan::try_new(std::sync::Arc::clone(plan.root())).unwrap();
    let again = optimize_with_udfs(&plan, &fixture(), &UdfLibrary::new(), &Config::new()).unwrap();
    assert!(
        again.structurally_eq(&plan),
        "a fixed point:\n{}",
        again.root()
    );
}

/// A mixed group where one instance's body is the shared core itself (a CTE that ends in its
/// aggregate) and the other keeps a filter above it: the core is the definition, the bare
/// instance reads it as a shifted `CteRef` in place of its `Shift`, and only the filtered one
/// keeps its `Shift` (`share_cte.rs`, the `equal(&m.body, &def)` arm of a mixed group).
#[test]
fn an_instance_equal_to_the_core_reads_it_without_its_shift() {
    let sql = "WITH a AS (SELECT o_custkey, count(*) AS n FROM orders GROUP BY o_custkey) \
               SELECT x.o_custkey, x.n, y.n FROM a x \
                 JOIN (SELECT o_custkey, n FROM a WHERE n > 3) y ON x.o_custkey = y.o_custkey";
    let plan = run(sql, false);
    let text = plan.root().to_string();
    assert_eq!(
        scans(plan.root(), "spark_catalog.default.orders"),
        1,
        "{text}"
    );
    let krishiv_plan::PlanNode::WithCte { ctes, .. } = &plan.root().node else {
        panic!("the aggregate is shared:\n{text}");
    };
    assert_eq!(ctes.len(), 1, "{text}");
    assert!(ctes[0].plan.to_string().starts_with("Aggregate"), "{text}");
    assert_eq!(text.matches("CteRef").count(), 2, "{text}");
    // The filtered member reads the core in its body's own ids under its `Shift`; the bare one
    // reads it in shifted ids, with no `Shift` of its own (the `Shift +` left is the subquery
    // alias's, above the filtered member).
    let lines: Vec<&str> = text.lines().collect();
    let under: Vec<&str> = (1..lines.len())
        .filter(|&i| lines[i].contains("CteRef"))
        .map(|i| lines[i - 1].trim_start_matches([' ', ':', '+', '-']))
        .collect();
    assert_eq!(under.len(), 2, "{text}");
    assert!(under.iter().all(|l| !l.starts_with("Shift")), "{text}");
    assert_eq!(
        under.iter().filter(|l| l.starts_with("Filter")).count(),
        1,
        "{text}"
    );
    Plan::try_new(std::sync::Arc::clone(plan.root())).unwrap();
    let again = optimize_with_udfs(&plan, &fixture(), &UdfLibrary::new(), &Config::new()).unwrap();
    assert!(
        again.structurally_eq(&plan),
        "a fixed point:\n{}",
        again.root()
    );
}

/// TPC-DS q01 reads `customer_total_return` twice; the main instance carries an inferred
/// `isnotnull(sr_customer_sk)` below the aggregate (it joins `customer` on that key), the
/// average's instance does not. The filter reads only a group key, so it commutes with the
/// aggregate: the aggregate without it is computed once, and the main instance filters the
/// shared groups (L4).
#[test]
fn a_filter_on_group_keys_below_the_aggregate_shares_the_aggregate() {
    let plan = tpcds(1);
    let table = "spark_catalog.default.store_returns";
    assert_eq!(scans(plan.root(), table), 1, "q01:\n{}", plan.root());
    assert_eq!(
        plan.root().resolved_output(),
        support::resolve(
            &crate::tpcds::sql(1),
            &support::ctx_over(crate::tpcds::catalog())
        )
        .root()
        .resolved_output()
    );
    Plan::try_new(std::sync::Arc::clone(plan.root())).unwrap();
    let again = optimize_with_udfs(
        &plan,
        &krishiv_optimizer::NoStatistics,
        &UdfLibrary::new(),
        &Config::new(),
    )
    .unwrap();
    assert!(
        again.structurally_eq(&plan),
        "q01 is a fixed point:\n{}",
        again.root()
    );
}

/// q01's shape over TPC-H: the CTE's aggregate is read by a join on its customer key (so that
/// instance holds an inferred `isnotnull(o_custkey)`) and by a per-status average that never
/// reads the customer key (so pruning drops it from that instance's outputs). The aggregate is
/// computed once, and the rows are the unshared plan's.
#[test]
fn null_tests_and_pruned_keys_share_one_aggregate_with_the_same_rows() {
    use crate::support::{data, eval};
    let sql = "WITH t AS (SELECT o_custkey AS k, o_orderstatus AS s, sum(o_totalprice) AS tot \
                 FROM orders GROUP BY o_custkey, o_orderstatus) \
               SELECT c_name, t1.s, t1.tot FROM t t1 JOIN customer ON t1.k = c_custkey \
               WHERE t1.tot > (SELECT avg(t2.tot) * 1.2 FROM t t2 WHERE t2.s = t1.s)";
    let plan = run(sql, false);
    let orders = "spark_catalog.default.orders";
    assert_eq!(scans(plan.root(), orders), 1, "{}", plan.root());
    let text = plan.root().to_string();
    assert_eq!(text.matches("CteRef").count(), 2, "{text}");
    Plan::try_new(std::sync::Arc::clone(plan.root())).unwrap();
    let again = optimize_with_udfs(&plan, &fixture(), &UdfLibrary::new(), &Config::new()).unwrap();
    assert!(
        again.structurally_eq(&plan),
        "a fixed point:\n{}",
        again.root()
    );
    let db = data::tpch(7, 150);
    let written = support::resolve(sql, &support::ctx());
    let want = eval::rows(&db, written.root());
    assert!(!want.is_empty());
    assert_eq!(eval::rows(&db, plan.root()), want, "{}", plan.root());
}

/// L4 over data with NULL group keys (`data::small`, ~20% NULL `a`): the join on `x1.a` gives the
/// main instance an inferred `isnotnull(a)` the average's instance does not hold, and the average
/// reads no `a` (pruned). The shared aggregate must be the unfiltered one, so the average still
/// sees the NULL-`a` groups; the seeds include one where those groups change the answer.
#[test]
fn null_key_groups_stay_in_the_unfiltered_instance() {
    use crate::support::{data, eval};
    let sql = |extra: &str| {
        format!(
            "WITH x AS (SELECT a, b, sum(c) AS s FROM t GROUP BY a, b) \
             SELECT u.d, x1.s FROM x x1 JOIN u ON x1.a = u.a \
             WHERE x1.s >= (SELECT avg(x2.s) FROM x x2 WHERE x2.b = x1.b{extra})"
        )
    };
    let plan = run(&sql(""), false);
    let text = plan.root().to_string();
    assert_eq!(scans(plan.root(), "spark_catalog.default.t"), 1, "{text}");
    assert_eq!(text.matches("CteRef").count(), 2, "{text}");
    Plan::try_new(std::sync::Arc::clone(plan.root())).unwrap();
    let again = optimize_with_udfs(&plan, &fixture(), &UdfLibrary::new(), &Config::new()).unwrap();
    assert!(
        again.structurally_eq(&plan),
        "a fixed point:\n{}",
        again.root()
    );
    let written = support::resolve(&sql(""), &support::ctx());
    // The answer had the average skipped the NULL-`a` groups (the wrong definition).
    let without = support::resolve(&sql(" AND x2.a IS NOT NULL"), &support::ctx());
    let mut telling = 0;
    for seed in 1..=16 {
        let db = data::small(seed);
        let want = eval::rows(&db, written.root());
        assert_eq!(eval::rows(&db, plan.root()), want, "seed {seed}:\n{text}");
        if eval::rows(&db, without.root()) != want {
            telling += 1;
        }
    }
    assert!(
        telling > 0,
        "no seed where the NULL-`a` groups change the answer"
    );
}
