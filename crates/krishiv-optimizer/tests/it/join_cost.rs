// SPDX-License-Identifier: Apache-2.0
//! `krishiv.optimizer.joinCost` (E5b, design `P2b-E5b-join-bytes.md` §6):
//! costed by `bytes`, each join costs its result rows times `ROW_BYTES`
//! plus its build input's bytes (rows times the width of the columns still
//! read), and the input with fewer rows stays the build unless it holds more
//! than 2x the other's bytes. Plans return the same rows either way, keep
//! the crate's invariants (a fixed point, no new cross product, the same
//! output) and are planned canonically.

use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_optimizer::{NoStatistics, RuleStep, Statistics, optimize_traced_with_udfs};
use krishiv_plan::{LogicalPlan, Plan, PlanNode};
use krishiv_udf_sdk::UdfLibrary;

use crate::join_order::{fixture, no_cross_product};
use crate::support::{self, data, eval};
use crate::tpcds;

const BYTES: &[(&str, &str)] = &[("krishiv.optimizer.joinCost", "bytes")];

/// The queries whose plans with a session configuration are not fixed
/// points costed by rows either (`ShareCte`'s shared CTE plans); every
/// other plan must be one.
const TPCH_SHARED: &[usize] = &[15];
const TPCDS_SHARED: &[usize] = &[1, 2, 14, 23, 47, 57, 59, 64, 81, 95];

const GREEDY_BYTES: &[(&str, &str)] = &[
    ("krishiv.optimizer.joinCost", "bytes"),
    ("krishiv.optimizer.joinEnumerator", "greedy"),
];

/// Whether `p` (or a subquery plan in it) holds a join without a condition.
fn has_cross_product(p: &Arc<LogicalPlan>) -> bool {
    let mut stack = vec![Arc::clone(p)];
    let mut found = false;
    while let Some(n) = stack.pop() {
        n.for_each_child(&mut |c| stack.push(Arc::clone(c)));
        n.for_each_expr(&mut |e| {
            if let krishiv_plan::ExprNode::Subquery { plan, .. } = &e.node {
                stack.push(Arc::clone(plan));
            }
        });
        found |= matches!(
            n.node,
            PlanNode::Join {
                condition: krishiv_plan::JoinCondition::None,
                ..
            }
        );
    }
    found
}

/// The crate's invariants for `optimized` (from `plan` under `conf`): the
/// same output, a fixed point under the same settings, and no cross
/// product the plan as written does not have.
fn invariants(
    plan: &Plan,
    optimized: &Plan,
    stats: &dyn Statistics,
    conf: &[(&str, &str)],
    what: &str,
    exempt: bool,
) {
    assert_eq!(
        optimized.root().resolved_output(),
        plan.root().resolved_output(),
        "{what}"
    );
    // A fixed point, but for the listed queries (`TPCH_SHARED`,
    // `TPCDS_SHARED`), which are not one costed by rows either: with a
    // session configuration `ShareCte` runs too (on main as here).
    let (again, _) = plan_with(optimized, stats, conf);
    if !again.structurally_eq(optimized) {
        assert!(exempt, "{what} is not a fixed point:\n{}", again.root());
        let by_rows: Vec<(&str, &str)> = conf
            .iter()
            .copied()
            .filter(|(k, _)| *k != "krishiv.optimizer.joinCost")
            .collect();
        let (rows, _) = plan_with(plan, stats, &by_rows);
        let (rows_again, _) = plan_with(&rows, stats, &by_rows);
        assert!(
            !rows_again.structurally_eq(&rows),
            "{what} is a fixed point by rows but not by bytes:\n{}",
            again.root()
        );
    }
    if !has_cross_product(plan.root()) {
        no_cross_product(optimized.root(), what);
    }
}

fn plan_with(plan: &Plan, stats: &dyn Statistics, conf: &[(&str, &str)]) -> (Plan, Vec<RuleStep>) {
    let mut config = Config::new();
    for (k, v) in conf {
        config.set(k, v).unwrap();
    }
    optimize_traced_with_udfs(plan, stats, &UdfLibrary::new(), &config).unwrap()
}

/// The tree notes' `measure` fields.
fn measures(steps: &[RuleStep]) -> Vec<String> {
    steps
        .iter()
        .filter(|s| s.rule == "JoinOrder" && s.note.starts_with("tree "))
        .filter_map(|s| {
            s.note
                .split(' ')
                .find_map(|f| f.strip_prefix("measure="))
                .map(str::to_owned)
        })
        .collect()
}

/// Whether `p` scans the table `name`.
fn reads(p: &LogicalPlan, name: &str) -> bool {
    p.to_string().contains(&format!(".{name} ["))
}

/// For the first join found with one side reading `a` and the other `b`:
/// whether the side reading `a` is its left input.
fn sides(p: &Arc<LogicalPlan>, a: &str, b: &str) -> Option<bool> {
    let mut stack = vec![Arc::clone(p)];
    while let Some(n) = stack.pop() {
        if let PlanNode::Join { left, right, .. } = &n.node {
            match (
                reads(left, a),
                reads(right, a),
                reads(left, b),
                reads(right, b),
            ) {
                (true, false, false, true) => return Some(true),
                (false, true, true, false) => return Some(false),
                _ => {}
            }
        }
        n.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    None
}

/// Every TPC-H tree is costed by bytes when asked, and the plans return the
/// same rows as written.
#[test]
fn tpch_bytes_plans_return_the_same_rows() {
    let ctx = support::ctx();
    let stats = fixture();
    let db = data::tpch(7, 150);
    for q in 1..=22 {
        let plan = support::resolve(&support::tpch_sql(q), &ctx);
        let (rows, rows_steps) = plan_with(&plan, &stats, &[]);
        let (bytes, bytes_steps) = plan_with(&plan, &stats, BYTES);
        let (greedy, _) = plan_with(&plan, &stats, GREEDY_BYTES);
        let exempt = TPCH_SHARED.contains(&q);
        invariants(
            &plan,
            &bytes,
            &stats,
            BYTES,
            &format!("q{q:02} bytes"),
            exempt,
        );
        let what = format!("q{q:02} greedy bytes");
        invariants(&plan, &greedy, &stats, GREEDY_BYTES, &what, exempt);
        assert!(measures(&rows_steps).iter().all(|m| m == "rows"), "q{q:02}");
        let m = measures(&bytes_steps);
        assert_eq!(m.len(), measures(&rows_steps).len(), "q{q:02}");
        assert!(m.iter().all(|m| m == "bytes"), "q{q:02}");
        let want = eval::rows(&db, plan.root());
        assert_eq!(eval::rows(&db, bytes.root()), want, "q{q:02}");
        assert_eq!(eval::rows(&db, greedy.root()), want, "q{q:02}");
        assert_eq!(eval::rows(&db, rows.root()), want, "q{q:02}");
    }
}

/// q10 at SF10 (study 05 §7, design §3): by rows the tree joins customer
/// and orders first and builds that wide intermediate against lineitem; by
/// bytes it joins orders and lineitem first and builds them (narrow)
/// against customer, as DuckDB does.
#[test]
fn q10_by_bytes_builds_the_narrow_side() {
    let plan = support::resolve(&support::tpch_sql(10), &support::ctx());
    let stats = fixture();
    let (rows, _) = plan_with(&plan, &stats, &[]);
    let (bytes, _) = plan_with(&plan, &stats, BYTES);
    // By rows: customer is in the build (right) side of its join with
    // lineitem.
    assert_eq!(sides(rows.root(), "customer", "lineitem"), Some(false));
    // By bytes: customer is the probe (left), lineitem's side the build.
    assert_eq!(sides(bytes.root(), "customer", "lineitem"), Some(true));
}

/// At SF10, costing by bytes changes only q02 and q10 of TPC-H (design §4
/// predicted at most four, q05/q07/q08/q09 kept). q09 keeps its lineitem-side
/// build: a join key used up inside a join is not carried above it, and the
/// input with fewer rows stays the build within 2x of the other's bytes.
#[test]
fn tpch_plans_that_bytes_changes() {
    let ctx = support::ctx();
    let stats = fixture();
    let changed: Vec<usize> = (1..=22)
        .filter(|&q| {
            let plan = support::resolve(&support::tpch_sql(q), &ctx);
            let (rows, _) = plan_with(&plan, &stats, &[]);
            let (bytes, _) = plan_with(&plan, &stats, BYTES);
            !rows.structurally_eq(&bytes)
        })
        .collect();
    assert_eq!(changed, [2, 10]);
}

/// TPC-DS under bytes (default row counts: no statistics here) keeps the
/// crate's invariants, for both enumerators.
#[test]
fn tpcds_bytes_plans_keep_the_invariants() {
    let ctx = support::ctx_over(tpcds::catalog());
    let mut planned = 0;
    for q in 1..=99 {
        let Some(plan) = support::try_resolve(&tpcds::sql(q), &ctx) else {
            continue;
        };
        for conf in [BYTES, GREEDY_BYTES] {
            let (bytes, _) = plan_with(&plan, &NoStatistics, conf);
            let what = format!("q{q:02} {conf:?}");
            let exempt = TPCDS_SHARED.contains(&q);
            invariants(&plan, &bytes, &NoStatistics, conf, &what, exempt);
        }
        planned += 1;
    }
    assert!(planned > 90, "{planned} queries");
}
