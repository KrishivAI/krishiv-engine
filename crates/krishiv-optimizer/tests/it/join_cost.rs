// SPDX-License-Identifier: Apache-2.0
//! `krishiv.optimizer.joinCost` (E5b, design `P2b-E5b-join-bytes.md`):
//! costed by `bytes`, a join tree minimises its intermediate results' rows
//! times the width of the columns read above the tree, and builds the input
//! with fewer bytes. Plans return the same rows either way.

use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_optimizer::{RuleStep, Statistics, optimize_traced_with_udfs};
use krishiv_plan::{LogicalPlan, Plan, PlanNode};
use krishiv_udf_sdk::UdfLibrary;

use crate::join_order::fixture;
use crate::support::{self, data, eval};

const BYTES: &[(&str, &str)] = &[("krishiv.optimizer.joinCost", "bytes")];

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
    p.to_string().contains(name)
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
        assert!(measures(&rows_steps).iter().all(|m| m == "rows"), "q{q:02}");
        let m = measures(&bytes_steps);
        assert_eq!(m.len(), measures(&rows_steps).len(), "q{q:02}");
        assert!(m.iter().all(|m| m == "bytes"), "q{q:02}");
        let want = eval::rows(&db, plan.root());
        assert_eq!(eval::rows(&db, bytes.root()), want, "q{q:02}");
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
            rows.root().to_string() != bytes.root().to_string()
        })
        .collect();
    assert_eq!(changed, [2, 10]);
}
