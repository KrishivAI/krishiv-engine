// SPDX-License-Identifier: Apache-2.0
//! The 22 TPC-H queries, parsed, resolved and optimised: plan snapshots,
//! unchanged outputs, idempotence, and the same rows as the unoptimised
//! plans.

use krishiv_optimizer::{NoStatistics, optimize, optimize_traced};

use crate::support::{self, data, eval};

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

/// Tracing (`EXPLAIN VERBOSE`) returns the plan optimising returns, and
/// its last changed step is that plan.
#[test]
fn tpch_traced_plans_match() {
    let ctx = support::ctx();
    let mut rules = std::collections::BTreeSet::new();
    for q in 1..=22 {
        let plan = support::resolve(&support::tpch_sql(q), &ctx);
        let optimized = optimize(&plan).unwrap();
        let (traced, steps) = optimize_traced(&plan, &NoStatistics).unwrap();
        assert!(traced.structurally_eq(&optimized), "q{q:02}");
        let last = steps
            .last()
            .unwrap_or_else(|| panic!("q{q:02} changed nothing"));
        assert_eq!(
            last.plan.to_string(),
            optimized.root().to_string(),
            "q{q:02}"
        );
        assert!(steps.iter().all(|s| s.round >= 1 && !s.rule.is_empty()));
        rules.extend(steps.iter().map(|s| (s.batch, s.rule)));
    }
    for want in [
        ("Rewrite", "PushDownFilter"),
        ("JoinOrder", "JoinOrder"),
        ("Prune", "Prune"),
    ] {
        assert!(rules.contains(&want), "no {want:?} step in TPC-H");
    }
}

/// Optimised and unoptimised plans return the same rows on generated data.
#[test]
fn tpch_results_match() {
    let ctx = support::ctx();
    let db = data::tpch(7, 150);
    let mut nonempty = 0;
    for q in 1..=22 {
        let plan = support::resolve(&support::tpch_sql(q), &ctx);
        let optimized = optimize(&plan).unwrap();
        let want = eval::rows(&db, plan.root());
        assert_eq!(eval::rows(&db, optimized.root()), want, "q{q:02}");
        nonempty += usize::from(!want.is_empty());
    }
    assert!(nonempty >= 20, "only {nonempty} queries return rows");
}
