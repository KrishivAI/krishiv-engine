// SPDX-License-Identifier: Apache-2.0
//! Equal derived tables that are not CTEs are computed once (perf-3 lever
//! D1, TPC-DS q65): `krishiv.optimizer.shareRepeatedSubtrees`.

use krishiv_common::config::Config;
use krishiv_optimizer::optimize_with_udfs;
use krishiv_plan::Plan;
use krishiv_udf_sdk::UdfLibrary;

use crate::join_order::fixture;
use crate::support;

fn tpcds(q: usize, on: bool) -> Plan {
    let ctx = support::ctx_over(crate::tpcds::catalog());
    let plan = support::resolve(&crate::tpcds::sql(q), &ctx);
    let mut config = Config::new();
    config
        .set(
            "krishiv.optimizer.shareRepeatedSubtrees",
            if on { "true" } else { "false" },
        )
        .unwrap();
    optimize_with_udfs(&plan, &fixture(), &UdfLibrary::new(), &config).unwrap()
}

#[test]
fn print_q65() {
    println!("{}", tpcds(65, true).root());
}
