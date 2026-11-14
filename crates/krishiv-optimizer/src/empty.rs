// SPDX-License-Identifier: Apache-2.0
//! Spark's `PruneFilters` + `PropagateEmptyRelation`, as far as `Dataset.observe` sees them
//! (P3-T084c): a relation proven empty (a filter that is `false` or NULL, a `LIMIT 0`, and what
//! Spark propagates it through) is replaced by an empty relation *with its whole subtree*, so an
//! observed node anywhere inside it is gone and its `Observation` reports `{}`. The relation's
//! rows are unchanged (none), so the rule only drops the [`PlanNode::CollectMetrics`] nodes under
//! it. An empty input *below* an observed node keeps it (`CollectMetrics` is not propagated
//! through), as do an outer join's preserved side, an anti join with an empty right side and a
//! union with a non-empty branch.

use std::sync::Arc;

use krishiv_plan::{ExprNode, Grouping, JoinKind, Literal, LogicalPlan, PlanNode};

use crate::dag;
use crate::limit::limit_value;

/// Whether `plan` is proven to return no rows at planning time, by Spark's rules.
pub(crate) fn proven_empty(plan: &LogicalPlan) -> bool {
    match &plan.node {
        PlanNode::Filter { input, predicate } => {
            matches!(
                &predicate.node,
                ExprNode::Literal(Literal::Boolean(false) | Literal::Null(_))
            ) || proven_empty(input)
        }
        PlanNode::Limit { input, limit } => limit_value(limit) == Some(0) || proven_empty(input),
        PlanNode::Aggregate {
            input,
            grouping: Grouping::GroupBy(keys),
            ..
        } => !keys.is_empty() && proven_empty(input),
        PlanNode::Project { input, .. }
        | PlanNode::WithColumns { input, .. }
        | PlanNode::Offset { input, .. }
        | PlanNode::Sort { input, .. }
        | PlanNode::Distinct { input }
        | PlanNode::SubqueryAlias { input, .. }
        | PlanNode::WithCte { input, .. }
        | PlanNode::JoinHint { input, .. }
        | PlanNode::Repartition { input, .. }
        | PlanNode::Window { input, .. } => proven_empty(input),
        PlanNode::Join {
            left, right, kind, ..
        } => match kind {
            JoinKind::Inner | JoinKind::Cross | JoinKind::LeftSemi => {
                proven_empty(left) || proven_empty(right)
            }
            JoinKind::LeftOuter | JoinKind::LeftAnti => proven_empty(left),
            JoinKind::RightOuter => proven_empty(right),
            JoinKind::FullOuter => proven_empty(left) && proven_empty(right),
            _ => false,
        },
        PlanNode::Union { inputs, .. } => inputs.iter().all(|i| proven_empty(i)),
        _ => false,
    }
}

fn observes(plan: &LogicalPlan) -> bool {
    matches!(plan.node, PlanNode::CollectMetrics { .. }) || {
        let mut found = false;
        plan.for_each_child(&mut |i| found = found || observes(i));
        found
    }
}

/// The rule at any relation: when it is proven empty, its observed nodes are dropped.
pub(crate) fn prune(node: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    if !observes(&node) || !proven_empty(&node) {
        return node;
    }
    // A failure cannot happen (the closure never fails); keep the plan if it did.
    dag::rewrite_up(Arc::clone(&node), &mut |p| {
        Ok(match &p.node {
            PlanNode::CollectMetrics { input, .. } => Arc::clone(input),
            _ => p,
        })
    })
    .unwrap_or(node)
}
