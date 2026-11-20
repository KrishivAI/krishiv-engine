// SPDX-License-Identifier: Apache-2.0
//! `OneRowAbove`: a `max`/`min` of a column of a one-row relation, in a grouped aggregate over a
//! cross join with it, is read above the aggregate (TPC-DS q23's `best_ss_customer`). The rows
//! equal the unoptimized plan's; the shapes the rule must not touch keep their aggregate.

use krishiv_optimizer::optimize_with;
use krishiv_plan::{ExprNode, Plan, PlanNode, ResolvedExpr};

use crate::join_order::fixture;
use crate::support::{self, data, eval};

/// `max_total` is a one-row relation: the largest customer total.
fn query(select: &str, tail: &str) -> String {
    format!(
        "WITH m AS (SELECT max(s) cmax, min(s) cmin FROM \
         (SELECT o_custkey k, sum(o_totalprice) s FROM orders GROUP BY o_custkey) x) \
         SELECT {select} FROM orders, customer, m WHERE o_custkey = c_custkey {tail}"
    )
}

fn optimized(sql: &str) -> (Plan, Plan) {
    let plan = support::resolve(sql, &support::ctx());
    let out = optimize_with(&plan, &fixture()).unwrap();
    (plan, out)
}

/// The `max`/`min` calls in the aggregates of `p`.
fn extremes(p: &Plan) -> usize {
    let mut n = 0;
    let mut todo = vec![p.root()];
    while let Some(node) = todo.pop() {
        node.for_each_child(&mut |c| todo.push(c));
        let PlanNode::Aggregate { aggregates, .. } = &node.node else {
            continue;
        };
        for a in aggregates {
            let ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) = &a.node else {
                continue;
            };
            if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &expr.node {
                n += usize::from(matches!(f.id.name(), "max" | "min"));
            }
        }
    }
    n
}

fn same_rows(plan: &Plan, out: &Plan) {
    let db = data::tpch(7, 150);
    assert_eq!(
        eval::rows(&db, out.root()),
        eval::rows(&db, plan.root()),
        "{}",
        out.root()
    );
}

#[test]
fn max_and_min_of_a_one_row_relation_leave_the_aggregate() {
    let sql = query(
        "o_custkey, sum(o_totalprice) ss, max(cmax) hi, min(cmin) lo",
        "GROUP BY o_custkey HAVING sum(o_totalprice) > (50/100.0) * max(cmax)",
    );
    let (plan, out) = optimized(&sql);
    // Only the one-row relation's own `max(s)` and `min(s)` remain.
    assert_eq!(extremes(&out), 2, "{}", out.root());
    assert_eq!(out.root().resolved_output(), plan.root().resolved_output());
    same_rows(&plan, &out);
    let again = optimize_with(&out, &fixture()).unwrap();
    assert!(
        again.structurally_eq(&out),
        "fixed point:\n{}",
        again.root()
    );
}

#[test]
fn a_global_aggregate_keeps_its_call() {
    // No grouping key: over an empty join the `max` is NULL, the column is not.
    let sql = query("sum(o_totalprice) ss, max(cmax) hi", "AND o_custkey < 0");
    let (plan, out) = optimized(&sql);
    assert_eq!(extremes(&out), 2, "{}", out.root());
    same_rows(&plan, &out);
}

#[test]
fn another_use_of_the_one_row_relation_keeps_the_call() {
    let sql = query(
        "o_custkey, max(cmax) hi, sum(cmax) total",
        "GROUP BY o_custkey",
    );
    let (plan, out) = optimized(&sql);
    assert_eq!(extremes(&out), 2, "{}", out.root());
    same_rows(&plan, &out);
}

#[test]
fn a_distinct_max_keeps_its_call() {
    let sql = query("o_custkey, max(DISTINCT cmax) hi", "GROUP BY o_custkey");
    let (plan, out) = optimized(&sql);
    assert_eq!(extremes(&out), 2, "{}", out.root());
    same_rows(&plan, &out);
}

#[test]
fn a_many_row_relation_keeps_its_call() {
    let sql = "SELECT o_custkey, max(s) hi FROM orders, \
               (SELECT o_custkey k, sum(o_totalprice) s FROM orders GROUP BY o_custkey) x \
               GROUP BY o_custkey";
    let (plan, out) = optimized(sql);
    assert_eq!(extremes(&out), 1, "{}", out.root());
    same_rows(&plan, &out);
}
