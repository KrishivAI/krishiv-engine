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

/// Off by default: q11 keeps today's plan (two evaluations of the join).
#[test]
fn q11_is_unchanged_without_the_key() {
    let plan = q11(&[]);
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
