// SPDX-License-Identifier: Apache-2.0
//! perf-3 D1 (TPC-DS q65 shape): a derived table written more than once, its
//! copies changed differently by pruning and inferred join filters (a key
//! filter `isnotnull(a)` from the join with `u`, null `a` groups kept by the
//! unfiltered copies), is computed once with the same rows.

use krishiv_common::config::Config;

use super::*;

const R: &str = "(SELECT a, s, sum(b) AS v FROM t GROUP BY a, s)";

fn query() -> String {
    let branch = |call: &str| {
        format!(
            "SELECT r.a AS a, r.v AS v FROM {R} r JOIN t u ON u.a = r.a \
             JOIN (SELECT s, {call}(v) AS m FROM {R} x GROUP BY s) q ON r.s = q.s \
             WHERE r.v >= q.m * 0.5"
        )
    };
    format!(
        "SELECT * FROM ({} UNION ALL {}) ORDER BY a, v",
        branch("avg"),
        branch("max")
    )
}

fn rows_under(env: &Env, on: bool) -> (Vec<String>, usize) {
    let mut config = Config::new();
    config
        .set(
            "krishiv.optimizer.shareRepeatedSubtrees",
            if on { "true" } else { "false" },
        )
        .unwrap();
    // A runtime filter from `u` reaches the filtered copies, and the planner
    // keeps the copies of a body one reaches (`shared.rs`): off, so the
    // shared body runs.
    config
        .set("krishiv.runtimeFilter.enabled", "false")
        .unwrap();
    let analyzed = env.analyze(&query());
    let plan = krishiv_optimizer::optimize_with_udfs(
        &analyzed,
        &krishiv_optimizer::NoStatistics,
        &krishiv_udf_sdk::UdfLibrary::new(),
        &config,
    )
    .unwrap();
    let cx = PlanningContext::new(Arc::clone(&env.tables) as _, Arc::new(config));
    let runnable = env.runnable(DefaultPlanner.plan(&plan, &cx).unwrap());
    let shared = runnable.plan.shared().len();
    let batches = run(&runnable).unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b.columns().iter().map(|c| cell(c.as_ref(), r)).collect();
            rows.push(cells.join("|"));
        }
    }
    (rows, shared)
}

/// Flag on or off, the rows are the same; on, the aggregate is one shared
/// body.
#[test]
fn sharing_the_derived_aggregate_keeps_the_rows() {
    let env = env_of(40);
    let (off, shared_off) = rows_under(&env, false);
    let (on, shared_on) = rows_under(&env, true);
    assert_eq!(off, on);
    assert!(!off.is_empty());
    assert_eq!(shared_off, 0);
    assert_eq!(shared_on, 1);
}
