// SPDX-License-Identifier: Apache-2.0
//! `ShareLiteralInstances` (TPC-DS q31's shape): instances of one CTE that
//! differ only in a group key's literal read one definition over all the
//! literals, with the same rows as one computation per instance.

use krishiv_common::config::Config;
use krishiv_optimizer::NoStatistics;

use super::*;

/// One aggregate read for three literals of its key `a` (one that matches no
/// row: `a` is null where `v % 4 == 3`), and a fourth reader of a different
/// key. `ShareCte` alone shares none of them.
const Q: &str = "WITH c AS (SELECT a, s, count(*) AS n, sum(b) AS sb FROM t GROUP BY a, s) \
     SELECT 'x', * FROM c WHERE a = 1 \
     UNION ALL SELECT 'y', * FROM c WHERE a = 6 \
     UNION ALL SELECT 'z', * FROM c WHERE a = 3 \
     UNION ALL SELECT 'w', a, s, n, sb FROM c WHERE s = 's9'";

/// `sql`'s rows with the rule on or off, and the shared plans; `refs`, when
/// given, is the number of `__shared_lit_*` references the plan must hold.
fn rows_under(env: &Env, sql: &str, on: bool, refs_on: Option<usize>) -> (Vec<String>, usize) {
    let mut config = Config::new();
    config
        .set(
            "krishiv.optimizer.shareLiteralInstances",
            if on { "true" } else { "false" },
        )
        .unwrap();
    let analyzed = env.analyze(sql);
    let plan = krishiv_optimizer::optimize_with_udfs(
        &analyzed,
        &NoStatistics,
        &krishiv_udf_sdk::UdfLibrary::new(),
        &config,
    )
    .unwrap();
    fn refs(p: &Arc<krishiv_plan::LogicalPlan>) -> usize {
        let mut n = usize::from(matches!(&p.node,
            krishiv_plan::PlanNode::CteRef(r) if krishiv_optimizer::is_literal_shared(&r.name)));
        p.for_each_child(&mut |c| n += refs(c));
        n
    }
    if let Some(n) = refs_on {
        assert_eq!(refs(plan.root()), if on { n } else { 0 }, "{}", plan.root());
    }
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
    rows.sort();
    (rows, shared)
}

/// On or off, the rows are the same; on, the three literals read one shared
/// body.
#[test]
fn sharing_the_literal_instances_keeps_the_rows() {
    let env = env_of(200);
    let (off, shared_off) = rows_under(&env, Q, false, Some(3));
    let (on, shared_on) = rows_under(&env, Q, true, Some(3));
    assert_eq!(off, on);
    assert_eq!(off.len(), 3, "{off:?}");
    assert_eq!(shared_off, 0);
    assert_eq!(shared_on, 1);
}

/// The key from the null-producing side of an outer join, and an aggregate
/// only one reader keeps (`sb`): whatever the rule does with them, the rows
/// stay those of one computation per instance.
#[test]
fn an_outer_join_key_and_an_unshared_sum_keep_the_rows() {
    let env = env_of(200);
    for sql in [
        "WITH c AS (SELECT r.a, count(*) AS n, sum(l.b) AS sb FROM t l \
         LEFT JOIN (SELECT * FROM t WHERE b > 0) r ON l.s = r.s GROUP BY r.a) \
         SELECT 'x', a, n, sb FROM c WHERE a = 1 UNION ALL SELECT 'y', a, n, sb FROM c WHERE a = 6",
        "WITH c AS (SELECT a, s, count(*) AS n, sum(b) AS sb FROM t GROUP BY a, s) \
         SELECT 'x', a, s, sb FROM c WHERE a = 1 UNION ALL SELECT 'y', a, s, n FROM c WHERE a = 6",
    ] {
        let (off, _) = rows_under(&env, sql, false, None);
        let (on, _) = rows_under(&env, sql, true, None);
        assert_eq!(off, on, "{sql}");
        assert!(!off.is_empty(), "{sql}");
    }
}
