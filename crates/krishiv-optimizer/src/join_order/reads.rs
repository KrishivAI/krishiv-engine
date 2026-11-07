// SPDX-License-Identifier: Apache-2.0
//! Which columns a plan reads, and where (E5b, `krishiv.optimizer.joinCost`
//! = `bytes`).
//!
//! Join ordering runs before column pruning, so a join input's output still
//! carries columns nothing reads. Costing a join tree by bytes counts only
//! the columns read outside the tree: [`Reads`] counts, per column, the plan
//! nodes whose expressions read it (each distinct node once) over the whole
//! plan and over one tree; a column read more often in the plan than inside
//! the tree is read above it. Positional reads (the query's output, CTE
//! definitions, renaming aliases, union inputs, recursive terms and
//! subquery plans) read every output column; they count [`POSITIONAL`]
//! each, so no count of node reads reaches them.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use krishiv_plan::{AttrId, ExprNode, LogicalPlan, PlanNode};

use crate::expr::{Attrs, references_into};

/// What one positional read of a column counts.
const POSITIONAL: u64 = 1 << 32;

/// Reads per column under one plan.
#[derive(Debug, Default)]
pub(super) struct Reads(HashMap<AttrId, u64>);

impl Reads {
    /// The reads of the query `root`, its output included.
    pub(super) fn of_query(root: &Arc<LogicalPlan>) -> Self {
        let mut r = Self::default();
        r.positional(root);
        r.walk(root);
        r
    }

    /// The reads inside `tree` (not its own output).
    pub(super) fn of_tree(tree: &Arc<LogicalPlan>) -> Self {
        let mut r = Self::default();
        r.walk(tree);
        r
    }

    /// The columns read more often in `self` than in `inside`: read
    /// outside the subtree `inside` counts.
    pub(super) fn outside(&self, inside: &Self) -> Attrs {
        self.0
            .iter()
            .filter(|(id, n)| **n > inside.0.get(id).copied().unwrap_or(0))
            .map(|(id, _)| *id)
            .collect()
    }

    fn positional(&mut self, p: &LogicalPlan) {
        for c in p.resolved_output().unwrap_or_default() {
            let n = self.0.entry(c.id).or_default();
            *n = n.saturating_add(POSITIONAL);
        }
    }

    fn walk(&mut self, root: &Arc<LogicalPlan>) {
        let mut seen: HashSet<*const LogicalPlan> = HashSet::new();
        let mut stack = vec![Arc::clone(root)];
        while let Some(p) = stack.pop() {
            if !seen.insert(Arc::as_ptr(&p)) {
                continue;
            }
            let mut ids = Attrs::new();
            let mut subqueries = Vec::new();
            p.for_each_expr(&mut |e| {
                references_into(e, &mut ids);
                subquery_plans(e, &mut subqueries);
            });
            for id in ids {
                let n = self.0.entry(id).or_default();
                *n = n.saturating_add(1);
            }
            for s in subqueries {
                self.positional(&s);
                stack.push(s);
            }
            // As `Reorder::children` reads them: the CTE definitions, the
            // input of a renaming alias and union inputs by position.
            let (by_position, ctes) = match &p.node {
                PlanNode::SubqueryAlias { columns, .. } => (!columns.is_empty(), 0),
                PlanNode::Union { .. } | PlanNode::SetOp { .. } | PlanNode::UnionLoop { .. } => {
                    (true, 0)
                }
                PlanNode::WithCte { ctes, .. } => (false, ctes.len()),
                _ => (false, 0),
            };
            let mut i = 0;
            p.for_each_child(&mut |c| {
                i += 1;
                if by_position || i <= ctes {
                    self.positional(c);
                }
                stack.push(Arc::clone(c));
            });
        }
    }
}

/// The subquery plans in `e`.
fn subquery_plans(e: &Arc<krishiv_plan::Expr>, out: &mut Vec<Arc<LogicalPlan>>) {
    let mut stack = vec![e];
    let mut seen = HashSet::new();
    while let Some(x) = stack.pop() {
        if !seen.insert(Arc::as_ptr(x)) {
            continue;
        }
        if let ExprNode::Subquery { plan, .. } = &x.node {
            out.push(Arc::clone(plan));
        }
        x.for_each_child(&mut |c| stack.push(c));
    }
}

#[cfg(test)]
mod tests {
    use krishiv_plan::AttrId;

    use super::Reads;
    use crate::expr::Attrs;

    #[test]
    fn a_read_above_the_subtree_counts_and_one_inside_it_does_not() {
        let mut query = Reads::default();
        query.0.extend([
            (AttrId(1), 3),
            (AttrId(2), 1),
            (AttrId(3), super::POSITIONAL),
        ]);
        let mut tree = Reads::default();
        tree.0.extend([(AttrId(1), 2), (AttrId(2), 1)]);
        assert_eq!(query.outside(&tree), Attrs::from([AttrId(1), AttrId(3)]));
    }
}
