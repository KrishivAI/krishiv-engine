// SPDX-License-Identifier: Apache-2.0
//! A bottom-up rewrite over every relation of a plan, subquery plans
//! included, that rewrites a shared subtree once.

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{Expr, ExprNode, LogicalPlan, PlanNode};

/// A rewrite of one relation whose children are already rewritten.
pub(crate) type Step<'a> =
    dyn FnMut(Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, KrishivError> + 'a;

/// Rewrites relations children first, entering subquery plans; each
/// distinct relation (by address) is rewritten once and its rewrite shared.
pub(crate) struct Rewriter {
    /// Original → rewritten; the original is kept alive so its address
    /// stays unique.
    done: HashMap<*const LogicalPlan, (Arc<LogicalPlan>, Arc<LogicalPlan>)>,
}

impl Rewriter {
    pub(crate) fn new() -> Self {
        Self {
            done: HashMap::new(),
        }
    }

    /// Applies `step` to every relation under `plan`, children first.
    pub(crate) fn up(
        &mut self,
        plan: Arc<LogicalPlan>,
        step: &mut Step<'_>,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        if let Some((_, out)) = self.done.get(&Arc::as_ptr(&plan)) {
            return Ok(Arc::clone(out));
        }
        let rebuilt = Arc::clone(&plan)
            .map_children(|c| self.up(c, step))?
            .map_expressions(|e| self.subqueries(e, step))?;
        let out = step(rebuilt)?;
        self.done
            .insert(Arc::as_ptr(&plan), (plan, Arc::clone(&out)));
        Ok(out)
    }

    /// `e` with its subquery plans rewritten.
    pub(crate) fn subqueries(
        &mut self,
        e: Arc<Expr>,
        step: &mut Step<'_>,
    ) -> Result<Arc<Expr>, KrishivError> {
        if !crate::expr::has_subquery(&e) {
            return Ok(e);
        }
        e.transform_up(&mut |x| match &x.node {
            ExprNode::Subquery { kind, plan } => {
                let rewritten = self.up(Arc::clone(plan), step)?;
                // A subquery's consumer reads its output by position or
                // width: a rewrite that widened its root (CSE's extracted
                // columns under a filter) is projected back.
                let rewritten =
                    crate::restore_output(rewritten, &plan.resolved_output(), plan.origin);
                if Arc::ptr_eq(&rewritten, plan) {
                    return Ok(x);
                }
                Ok(Arc::new(Expr {
                    node: ExprNode::Subquery {
                        kind: kind.clone(),
                        plan: rewritten,
                    },
                    origin: x.origin,
                }))
            }
            _ => Ok(x),
        })
    }
}

/// Applies `step` to every relation of `root`, children first.
pub(crate) fn rewrite_up(
    root: Arc<LogicalPlan>,
    step: &mut Step<'_>,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    Rewriter::new().up(root, step)
}

/// `node` (a relation with an `input`) over `input` instead.
pub(crate) fn with_input(node: &Arc<LogicalPlan>, input: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    let mut out = LogicalPlan::clone(node);
    match &mut out.node {
        PlanNode::Project { input: slot, .. }
        | PlanNode::Filter { input: slot, .. }
        | PlanNode::Aggregate { input: slot, .. }
        | PlanNode::Sort { input: slot, .. }
        | PlanNode::Limit { input: slot, .. }
        | PlanNode::Offset { input: slot, .. }
        | PlanNode::SubqueryAlias { input: slot, .. }
        | PlanNode::WithCte { input: slot, .. }
        | PlanNode::Shift { input: slot, .. }
        | PlanNode::JoinHint { input: slot, .. }
        | PlanNode::Window { input: slot, .. }
        | PlanNode::ReplicateRows { input: slot, .. } => *slot = input,
        _ => return Arc::clone(node),
    }
    Arc::new(out)
}
