// SPDX-License-Identifier: Apache-2.0
//! `SemiAntiOrder` (`krishiv.optimizer.semiAntiJoinOrder`): a stack of semi/anti joins over one
//! input runs its most selective join first; with the key off, or when a condition may raise,
//! the stack keeps the order of the text. Both orders return the same rows.

use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_optimizer::optimize_with_udfs;
use krishiv_plan::{JoinKind, LogicalPlan, Plan, PlanNode};
use krishiv_udf_sdk::UdfLibrary;

use crate::join_order::fixture;
use crate::support::{self, data, eval};

fn run(plan: &Plan, on: bool) -> Plan {
    let mut config = Config::new();
    config
        .set(
            "krishiv.optimizer.semiAntiJoinOrder",
            if on { "true" } else { "false" },
        )
        .unwrap();
    optimize_with_udfs(plan, &fixture(), &UdfLibrary::new(), &config).unwrap()
}

/// The kinds of the semi/anti joins of `p`, outermost first.
fn stack(p: &Arc<LogicalPlan>) -> Vec<JoinKind> {
    let mut out = Vec::new();
    let mut todo = vec![Arc::clone(p)];
    while let Some(n) = todo.pop() {
        if let PlanNode::Join {
            kind: kind @ (JoinKind::LeftSemi | JoinKind::LeftAnti),
            ..
        } = &n.node
        {
            out.push(*kind);
        }
        n.for_each_child(&mut |c| todo.push(Arc::clone(c)));
    }
    out
}

/// q21 with `extra` added to its `NOT EXISTS` subquery.
fn q21(extra: &str) -> String {
    support::tpch_sql(21).replacen(
        "AND l3.l_receiptdate > l3.l_commitdate",
        &format!("AND l3.l_receiptdate > l3.l_commitdate {extra}"),
        1,
    )
}

#[test]
fn q21_runs_its_anti_join_first_only_with_the_key() {
    assert_ne!(
        q21("AND true"),
        support::tpch_sql(21),
        "the edit point exists"
    );
    let sql = q21("");
    let plan = support::resolve(&sql, &support::ctx());
    let off = run(&plan, false);
    assert_eq!(
        stack(off.root()),
        [JoinKind::LeftAnti, JoinKind::LeftSemi],
        "as written: {}",
        off.root()
    );
    let on = run(&plan, true);
    assert_eq!(
        stack(on.root()),
        [JoinKind::LeftSemi, JoinKind::LeftAnti],
        "{}",
        on.root()
    );
    // A fixed point.
    let again = run(&on, true);
    assert!(again.structurally_eq(&on), "{}", again.root());
    // Both orders return the rows the plan as written returns.
    let db = data::tpch(7, 150);
    let want = eval::rows(&db, plan.root());
    assert!(!want.is_empty(), "the data has q21 rows");
    assert_eq!(eval::rows(&db, on.root()), want);
    assert_eq!(eval::rows(&db, off.root()), want);
}

#[test]
fn a_condition_that_may_raise_keeps_the_written_order() {
    // Integral division raises on a zero divisor under ANSI: run first, the anti join would see
    // rows the semi join drops.
    let sql = q21("AND l3.l_linenumber div (l3.l_suppkey - l1.l_suppkey) > 0");
    let plan = support::resolve(&sql, &support::ctx());
    let on = run(&plan, true);
    assert_eq!(
        stack(on.root()),
        [JoinKind::LeftAnti, JoinKind::LeftSemi],
        "{}",
        on.root()
    );
    assert!(on.structurally_eq(&run(&plan, false)));
}
