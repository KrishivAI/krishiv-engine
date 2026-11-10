// SPDX-License-Identifier: Apache-2.0
//! Whether an analyzed plan reads a permanent table: the scans of a plan and
//! of the subqueries in its expressions, each shared expression node visited once.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_plan::{Expr, ExprNode, LogicalPlan, PlanNode, ReadSource, Scan};

/// Whether the analyzed `plan` reads the permanent table `namespace.name` (directly, or through a view or a
/// subquery): `INSERT OVERWRITE` of a table that is also read would delete what it reads.
pub(crate) fn reads_stored_table(
    plan: &LogicalPlan,
    namespace: &str,
    name: &str,
    current: &str,
) -> bool {
    let mut scans = Vec::new();
    collect_scans(plan, &mut scans);
    scans.iter().any(|scan| {
        let ReadSource::Table { name: parts, .. } = &scan.source else {
            return false;
        };
        let (ns, table) = match parts.as_slice() {
            [table] => (current, table),
            [.., ns, table] => (ns.as_str(), table),
            [] => return false,
        };
        ns.eq_ignore_ascii_case(namespace) && table.eq_ignore_ascii_case(name)
    })
}

/// The scans of `plan` and of the subqueries in its expressions.
///
/// Expressions are DAGs (the analyzer shares a subexpression by `Arc`, as a
/// nested `decode` does in every search condition), so each expression node
/// is visited once: a walk that revisits shared nodes is exponential in the
/// nesting depth.
pub(super) fn collect_scans<'a>(plan: &'a LogicalPlan, out: &mut Vec<&'a Scan>) {
    collect_scans_in(plan, out, &mut HashSet::new());
}

fn collect_scans_in<'a>(plan: &'a LogicalPlan, out: &mut Vec<&'a Scan>, seen: &mut HashSet<usize>) {
    fn in_expr<'a>(e: &'a Arc<Expr>, out: &mut Vec<&'a Scan>, seen: &mut HashSet<usize>) {
        if !seen.insert(Arc::as_ptr(e).addr()) {
            return;
        }
        if let ExprNode::Subquery { plan, .. } = &e.node {
            collect_scans_in(plan, out, seen);
        }
        e.for_each_child(&mut |c| in_expr(c, out, seen));
    }
    if let PlanNode::Scan(scan) = &plan.node {
        out.push(scan);
    }
    plan.for_each_expr(&mut |e| in_expr(e, out, seen));
    plan.for_each_child(&mut |c| collect_scans_in(c, out, seen));
}
