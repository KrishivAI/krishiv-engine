// SPDX-License-Identifier: Apache-2.0
//! Limit pushdown and top-N detection.
//!
//! A limit merges with a limit below it, moves below a deterministic
//! projection (so `LIMIT` over `ORDER BY` meets its sort), and is copied
//! onto the preserved side of an outer join, which needs at most that many
//! of its rows. [`TopN::detect`] finds the `LIMIT` over a total sort that the
//! physical planner runs as one top-N operator.

use std::sync::Arc;

use krishiv_plan::{Expr, ExprNode, Grouping, JoinKind, Literal, LogicalPlan, PlanNode, SortOrder};

use crate::dag::with_input;
use crate::expr::is_deterministic;

/// A `LIMIT n` directly over a total `ORDER BY`.
#[derive(Debug, Clone, Copy)]
pub struct TopN<'a> {
    /// The sort keys, most significant first.
    pub order: &'a [SortOrder],
    /// The row count.
    pub limit: u64,
    /// The sorted relation.
    pub input: &'a Arc<LogicalPlan>,
}

impl<'a> TopN<'a> {
    /// The top-N `plan` computes, when it is a literal `LIMIT` directly over
    /// a global sort.
    #[must_use]
    pub fn detect(plan: &'a LogicalPlan) -> Option<Self> {
        let PlanNode::Limit { input, limit } = &plan.node else {
            return None;
        };
        let PlanNode::Sort {
            input,
            order,
            global: true,
        } = &input.node
        else {
            return None;
        };
        Some(Self {
            order,
            limit: limit_value(limit)?,
            input,
        })
    }
}

/// The value of a literal, non-negative row count.
pub(crate) fn limit_value(e: &Expr) -> Option<u64> {
    match &e.node {
        ExprNode::Literal(Literal::Integer(v)) => u64::try_from(*v).ok(),
        _ => None,
    }
}

/// The most rows `plan` can return, when bounded.
pub(crate) fn max_rows(plan: &LogicalPlan) -> Option<u64> {
    match &plan.node {
        PlanNode::Limit { input, limit } => {
            let n = limit_value(limit)?;
            Some(max_rows(input).map_or(n, |m| m.min(n)))
        }
        PlanNode::Project { input, .. }
        | PlanNode::Filter { input, .. }
        | PlanNode::Sort { input, .. }
        | PlanNode::SubqueryAlias { input, .. }
        | PlanNode::WithCte { input, .. }
        | PlanNode::Shift { input, .. }
        | PlanNode::Window { input, .. }
        | PlanNode::JoinHint { input, .. } => max_rows(input),
        PlanNode::Union { inputs, .. } => inputs
            .iter()
            .try_fold(0_u64, |n, i| n.checked_add(max_rows(i)?)),
        PlanNode::Aggregate {
            grouping: Grouping::GroupBy(keys),
            ..
        } if keys.is_empty() => Some(1),
        PlanNode::Join {
            left,
            kind: JoinKind::LeftSemi | JoinKind::LeftAnti | JoinKind::Existence(_),
            ..
        } => max_rows(left),
        _ => None,
    }
}

/// The rule at a `Limit`.
pub(crate) fn push_limit(node: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    let PlanNode::Limit { input, limit } = &node.node else {
        return node;
    };
    let Some(n) = limit_value(limit) else {
        return node;
    };
    match &input.node {
        PlanNode::Limit {
            input: below,
            limit: inner,
        } => match limit_value(inner) {
            Some(m) => {
                let keep = if m <= n { inner } else { limit };
                limited(&node, below, Arc::clone(keep))
            }
            None => node,
        },
        PlanNode::Project { exprs, .. } if exprs.iter().all(|e| is_deterministic(e)) => {
            // Below the whole run of deterministic projections at once.
            let mut chain = vec![input];
            while let PlanNode::Project { input: next, .. } = &chain[chain.len() - 1].node {
                match &next.node {
                    PlanNode::Project { exprs, .. }
                        if exprs.iter().all(|e| is_deterministic(e)) =>
                    {
                        chain.push(next);
                    }
                    _ => break,
                }
            }
            let PlanNode::Project { input: below, .. } = &chain[chain.len() - 1].node else {
                return node;
            };
            let limited = limited(&node, below, Arc::clone(limit));
            chain
                .into_iter()
                .rev()
                .fold(limited, |acc, p| with_input(p, acc))
        }
        PlanNode::Join {
            left,
            right,
            kind: kind @ (JoinKind::LeftOuter | JoinKind::RightOuter | JoinKind::Existence(_)),
            condition,
        } => {
            let side = if *kind != JoinKind::RightOuter {
                left
            } else {
                right
            };
            if max_rows(side).is_some_and(|m| m <= n) {
                return node;
            }
            let side = limited(&node, side, Arc::clone(limit));
            let (left, right) = if *kind != JoinKind::RightOuter {
                (side, Arc::clone(right))
            } else {
                (Arc::clone(left), side)
            };
            let join = Arc::new(LogicalPlan {
                node: PlanNode::Join {
                    left,
                    right,
                    kind: *kind,
                    condition: condition.clone(),
                },
                origin: input.origin,
            });
            with_input(&node, join)
        }
        _ => node,
    }
}

/// `LIMIT limit` over `input`, with `like`'s origin.
fn limited(like: &LogicalPlan, input: &Arc<LogicalPlan>, limit: Arc<Expr>) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan {
        node: PlanNode::Limit {
            input: Arc::clone(input),
            limit,
        },
        origin: like.origin,
    })
}
