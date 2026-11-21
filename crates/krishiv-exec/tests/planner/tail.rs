// SPDX-License-Identifier: Apache-2.0
//! `Dataset.tail(n)`: the last `n` rows in plan order, through the optimizer, the planner and the executor.

use krishiv_plan::{Expr, Literal, LogicalPlan, Plan};

use super::{DefaultPlanner, Env, PhysicalPlanner as _, env, run};

/// The rows of `sql` with a `Tail n` above it, as text, and the plan's text.
fn tail(env: &Env, sql: &str, n: i32) -> (Vec<String>, String) {
    let analyzed = env.analyze(sql);
    let root = LogicalPlan::clone(analyzed.root()).tail(Expr::from(Literal::Integer(n)));
    let plan = krishiv_optimizer::optimize(&Plan::try_new(root).unwrap()).unwrap();
    let physical = DefaultPlanner.plan(&plan, &env.cx()).unwrap();
    let runnable = env.runnable(physical);
    let text = runnable.to_string();
    let mut rows = Vec::new();
    for b in run(&runnable).unwrap() {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| super::cell(c.as_ref(), r))
                .collect();
            rows.push(cells.join("|"));
        }
    }
    (rows, text)
}

#[test]
fn tail_returns_the_last_rows_in_plan_order() {
    let env = env();
    let (rows, text) = tail(&env, "SELECT b FROM t", 4);
    assert_eq!(rows, ["60", "70", "80", "90"]);
    assert!(text.starts_with("Tail 4"), "{text}");
    // Within one batch and across the batches of three.
    assert_eq!(tail(&env, "SELECT b FROM t", 1).0, ["90"]);
    assert_eq!(tail(&env, "SELECT b FROM t", 3).0, ["70", "80", "90"]);
}

#[test]
fn tail_zero_is_empty_and_a_large_tail_is_everything() {
    let env = env();
    assert!(tail(&env, "SELECT b FROM t", 0).0.is_empty());
    let all: Vec<String> = (0..10).map(|i| (i * 10).to_string()).collect();
    assert_eq!(tail(&env, "SELECT b FROM t", 10).0, all);
    assert_eq!(tail(&env, "SELECT b FROM t", 1000).0, all);
}

#[test]
fn tail_is_not_a_sort_and_a_limit() {
    let env = env();
    // The last rows of the sorted stream, kept in its order.
    let (rows, _) = tail(&env, "SELECT b FROM t ORDER BY b DESC", 3);
    assert_eq!(rows, ["20", "10", "0"]);
    // The rows after a filter, not the filter's smallest.
    let (rows, _) = tail(&env, "SELECT b FROM t WHERE b >= 30", 2);
    assert_eq!(rows, ["80", "90"]);
}

#[test]
fn tail_over_a_union_is_the_tail_of_the_union() {
    let env = env();
    // One tail per branch would give 80, 90, 81, 91.
    let (rows, _) = tail(&env, "SELECT b FROM t UNION ALL SELECT b + 1 FROM t", 2);
    assert_eq!(rows, ["81", "91"]);
    // The last rows span the branches.
    let (rows, _) = tail(&env, "SELECT b FROM t UNION ALL SELECT b + 1 FROM t", 12);
    let all: Vec<String> = (0..10)
        .map(|i| (i * 10).to_string())
        .chain((0..10).map(|i| (i * 10 + 1).to_string()))
        .collect();
    assert_eq!(rows, all[8..]);
    // A filter on one branch.
    let (rows, _) = tail(
        &env,
        "SELECT b FROM t UNION ALL SELECT b + 1 FROM t WHERE b < 30",
        3,
    );
    assert_eq!(rows, ["1", "11", "21"]);
}
