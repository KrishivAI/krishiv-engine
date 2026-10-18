// SPDX-License-Identifier: Apache-2.0
//! The 22 TPC-H queries, parsed, resolved and optimised: plan snapshots,
//! unchanged outputs and idempotence.

use krishiv_optimizer::optimize;

use crate::support;

#[test]
fn tpch_plans() {
    let ctx = support::ctx();
    for q in 1..=22 {
        let plan = support::resolve(&support::tpch_sql(q), &ctx);
        let optimized = optimize(&plan).unwrap();
        assert_eq!(
            optimized.root().resolved_output(),
            plan.root().resolved_output(),
            "q{q:02} keeps its output"
        );
        let again = optimize(&optimized).unwrap();
        assert!(
            again.structurally_eq(&optimized),
            "q{q:02} is a fixed point:\n{}",
            again.root()
        );
        insta::assert_snapshot!(format!("q{q:02}"), optimized.root().to_string());
    }
}
