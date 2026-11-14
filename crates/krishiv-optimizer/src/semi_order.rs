// SPDX-License-Identifier: Apache-2.0
//! `SemiAntiOrder`: a stack of `LEFT SEMI` / `LEFT ANTI` joins over one input runs its most
//! selective join first (`krishiv.optimizer.semiAntiJoinOrder`, on).
//!
//! Each join of such a stack keeps or drops a row of the stack's input by that row alone: its
//! output is its left input's columns, so no condition reads another join's right side, and
//! the joins commute. Join ordering plans inner-join trees only, so a stack stays in the order of
//! the text: TPC-H q21 runs its `EXISTS` (which keeps nearly every row) before its `NOT EXISTS`
//! (which keeps one in ten), and the semi join builds and probes ten times the rows it needs.
//!
//! The rule estimates every join of a stack over the stack's input alone (not over the joins
//! below it), so a join's estimate does not depend on where it stands and the sorted stack is a
//! fixed point; it sorts by estimated output rows, ties in their written order. A join moves only
//! when its condition is deterministic and raises on no row ([`runs_on_any_row`], under ANSI):
//! after the swap it sees rows the join that used to run first would have dropped.

use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{JoinCondition, JoinKind, LogicalPlan, PlanNode, runs_on_any_row};

use crate::dag;
use crate::estimate::Estimator;
use crate::stats::Statistics;

/// Sorts every semi/anti stack of `root` (subquery plans included) by estimated survivors.
pub(crate) fn reorder(
    root: Arc<LogicalPlan>,
    stats: &dyn Statistics,
    propagate: bool,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    let mut est = Estimator::with_propagation(stats, propagate);
    dag::rewrite_up(root, &mut |p| Ok(order_stack(p, &mut est)))
}

/// A semi or anti join whose condition may run on any row of its left input.
fn movable(p: &LogicalPlan) -> bool {
    match &p.node {
        PlanNode::Join {
            kind: JoinKind::LeftSemi | JoinKind::LeftAnti,
            condition,
            ..
        } => match condition {
            JoinCondition::None => true,
            JoinCondition::On(c) => runs_on_any_row(c, true),
            _ => false,
        },
        _ => false,
    }
}

fn left(p: &LogicalPlan) -> Option<&Arc<LogicalPlan>> {
    match &p.node {
        PlanNode::Join { left, .. } => Some(left),
        _ => None,
    }
}

/// `join` (a join) over `input` instead of its left input, keeping its origin.
fn with_left(join: &LogicalPlan, input: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    let mut out = join.clone();
    if let PlanNode::Join { left, .. } = &mut out.node {
        *left = input;
    }
    Arc::new(out)
}

/// `p` with the stack it tops sorted, or `p` itself when the stack has one join or is in order.
fn order_stack(p: Arc<LogicalPlan>, est: &mut Estimator<'_>) -> Arc<LogicalPlan> {
    if !movable(&p) || !left(&p).is_some_and(|l| movable(l)) {
        return p;
    }
    // Top down: `stack[0]` is `p`, the last element the join over the base.
    let mut stack = vec![Arc::clone(&p)];
    let mut base = left(&p).map(Arc::clone);
    while let Some(b) = base.take_if(|b| movable(b)) {
        base = left(&b).map(Arc::clone);
        stack.push(b);
    }
    let Some(base) = base else {
        return p;
    };
    // Bottom up, each with its rows estimated over the base alone.
    let mut keyed: Vec<(f64, Arc<LogicalPlan>)> = stack
        .into_iter()
        .rev()
        .map(|j| {
            let alone = with_left(&j, Arc::clone(&base));
            (est.plan(&alone).rows, j)
        })
        .collect();
    let written: Vec<usize> = keyed.iter().map(|(_, j)| Arc::as_ptr(j).addr()).collect();
    keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
    if keyed
        .iter()
        .map(|(_, j)| Arc::as_ptr(j).addr())
        .eq(written.iter().copied())
    {
        return p;
    }
    keyed
        .into_iter()
        .fold(base, |input, (_, join)| with_left(&join, input))
}
