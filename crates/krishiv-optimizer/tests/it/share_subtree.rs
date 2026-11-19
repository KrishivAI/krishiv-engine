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
