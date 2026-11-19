// SPDX-License-Identifier: Apache-2.0
//! P3 W1 shape (b): a subtree the main block and an uncorrelated subquery both
//! read is computed once, with the same rows either way.

use krishiv_common::config::Config;

use super::*;

const Q: &str = "SELECT x.a AS a, sum(y.b) AS total FROM t x JOIN t y ON x.a = y.a GROUP BY x.a \
                 HAVING sum(y.b) > (SELECT sum(y.b) * 0.1 FROM t x JOIN t y ON x.a = y.a) \
                 ORDER BY a";

fn rows_under(env: &Env, conf: &[(&str, &str)]) -> (Vec<String>, usize) {
    let mut config = Config::new();
    for (k, v) in conf {
        config.set(k, v).unwrap();
    }
    let analyzed = env.analyze(Q);
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

/// Flag on or off, the rows are the same; on, the join is a shared body.
#[test]
fn sharing_the_repeated_join_keeps_the_rows() {
    let env = env();
    let (off, shared_off) = rows_under(
        &env,
        &[("krishiv.optimizer.shareRepeatedSubtrees", "false")],
    );
    let (on, shared_on) = rows_under(&env, &[("krishiv.optimizer.shareRepeatedSubtrees", "true")]);
    assert_eq!(off, on);
    assert!(!off.is_empty());
    assert_eq!(shared_off, 0);
    assert_eq!(shared_on, 1);
}
