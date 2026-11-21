// SPDX-License-Identifier: Apache-2.0
//! A bottom-up rewrite over every relation of a plan, subquery plans
//! included, that rewrites a shared subtree once.
//!
//! A [`PlanNode::SecurityBarrier`]'s input is a plan of its own (design
//! `P3-E14-security-barrier.md` G3'): a rewrite leaves it as it is unless the caller says what runs
//! there. Every caller is one of two kinds, and says which:
//! - an **optimization** ([`rewrite_up`], [`rewrite_up_traced`], [`Rewriter::new`]) fails closed:
//!   it does not run inside a barrier, or runs there only the step it passes to
//!   [`rewrite_up_inside`] (the rules that opted in, guarded where they move expressions);
//! - a **lowering** ([`rewrite_up_lowering`]), a rewrite whose absence changes results or leaves a
//!   plan the executor cannot run, runs inside with the same step (`collation_keys` alone today).

use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{Expr, ExprNode, LogicalPlan, PlanNode};

/// A rewrite of one relation whose children are already rewritten.
pub(crate) type Step<'a> =
    dyn FnMut(Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, KrishivError> + 'a;

/// As [`Step`], also given the relation of the input the rebuilt one comes from.
pub(crate) type TracedStep<'a> =
    dyn FnMut(&Arc<LogicalPlan>, Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, KrishivError> + 'a;

/// What runs on the relations under a security barrier.
enum Inside<'r, 'a> {
    /// Nothing: the barrier's input is returned as it is.
    Skip,
    /// The step that runs outside (a lowering).
    Same,
    /// A step of its own.
    Step(&'r mut TracedStep<'a>),
}

/// Rewrites relations children first, entering subquery plans; each
/// distinct relation (by address) is rewritten once and its rewrite shared.
pub(crate) struct Rewriter {
    /// Original → rewritten, outside (`[0]`) and under a barrier (`[1]`); the original is kept
    /// alive so its address stays unique.
    done: [Done; 2],
}

/// One mode's memo: original → (original, rewritten).
type Done = crate::expr::AddrMap<LogicalPlan, (Arc<LogicalPlan>, Arc<LogicalPlan>)>;

impl Rewriter {
    /// A rewriter for an optimization: nothing runs under a barrier.
    pub(crate) fn new() -> Self {
        Self {
            done: Default::default(),
        }
    }

    /// Applies `step` to every relation under `plan` outside any barrier, children first.
    pub(crate) fn up(
        &mut self,
        plan: Arc<LogicalPlan>,
        step: &mut Step<'_>,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        self.up_traced(plan, &mut |_, rebuilt| step(rebuilt))
    }

    /// As [`Self::up`], `step` also seeing the input relation each rebuilt one comes from.
    pub(crate) fn up_traced(
        &mut self,
        plan: Arc<LogicalPlan>,
        step: &mut TracedStep<'_>,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        self.walk(plan, step, &mut Inside::Skip, false)
    }

    fn walk(
        &mut self,
        plan: Arc<LogicalPlan>,
        step: &mut TracedStep<'_>,
        inside: &mut Inside<'_, '_>,
        under: bool,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        let mode = usize::from(under);
        if let Some((_, out)) = self.done[mode].get(&Arc::as_ptr(&plan)) {
            return Ok(Arc::clone(out));
        }
        let rebuilt = if matches!(plan.node, PlanNode::SecurityBarrier { .. }) {
            // A barrier holds no expression; its input is rewritten in the inside mode.
            match inside {
                Inside::Skip => Arc::clone(&plan),
                Inside::Same | Inside::Step(_) => {
                    Arc::clone(&plan).map_children(|c| self.walk(c, step, inside, true))?
                }
            }
        } else {
            Arc::clone(&plan)
                .map_children(|c| self.walk(c, step, inside, under))?
                .map_expressions(|e| self.subqueries(e, step, inside, under))?
        };
        let out = match (under, &mut *inside) {
            (false, _) | (true, Inside::Same) => step(&plan, rebuilt)?,
            (true, Inside::Step(s)) => s(&plan, rebuilt)?,
            (true, Inside::Skip) => rebuilt,
        };
        self.done[mode].insert(Arc::as_ptr(&plan), (plan, Arc::clone(&out)));
        Ok(out)
    }

    /// `e` with its subquery plans rewritten, in the mode of the relation holding `e`.
    fn subqueries(
        &mut self,
        e: Arc<Expr>,
        step: &mut TracedStep<'_>,
        inside: &mut Inside<'_, '_>,
        under: bool,
    ) -> Result<Arc<Expr>, KrishivError> {
        if !crate::expr::has_subquery(&e) {
            return Ok(e);
        }
        e.transform_up(&mut |x| match &x.node {
            ExprNode::Subquery { kind, plan } => {
                let rewritten = self.walk(Arc::clone(plan), step, inside, under)?;
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

/// Applies `step` to every relation of `root` outside any barrier, children first (an
/// optimization that did not opt in).
pub(crate) fn rewrite_up(
    root: Arc<LogicalPlan>,
    step: &mut Step<'_>,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    Rewriter::new().up(root, step)
}

/// As [`rewrite_up`], `step` also seeing the input relation each rebuilt one comes from.
pub(crate) fn rewrite_up_traced(
    root: Arc<LogicalPlan>,
    step: &mut TracedStep<'_>,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    Rewriter::new().up_traced(root, step)
}

/// As [`rewrite_up`], and `inside` on every relation under a barrier (an optimization that opted
/// in; `inside` is its guarded form where it moves expressions).
pub(crate) fn rewrite_up_inside(
    root: Arc<LogicalPlan>,
    step: &mut Step<'_>,
    inside: &mut Step<'_>,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    let mut inner = |_: &Arc<LogicalPlan>, p: Arc<LogicalPlan>| inside(p);
    Rewriter::new().walk(
        root,
        &mut |_, p| step(p),
        &mut Inside::Step(&mut inner),
        false,
    )
}

/// As [`rewrite_up`], with `step` also on every relation under a barrier (a lowering).
pub(crate) fn rewrite_up_lowering(
    root: Arc<LogicalPlan>,
    step: &mut Step<'_>,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    Rewriter::new().walk(root, &mut |_, p| step(p), &mut Inside::Same, false)
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
        | PlanNode::SecurityBarrier { input: slot, .. }
        | PlanNode::JoinHint { input: slot, .. }
        | PlanNode::Repartition { input: slot, .. }
        | PlanNode::CollectMetrics { input: slot, .. }
        | PlanNode::Window { input: slot, .. }
        | PlanNode::Generate { input: slot, .. }
        | PlanNode::Sample { input: slot, .. }
        | PlanNode::ReplicateRows { input: slot, .. } => *slot = input,
        _ => return Arc::clone(node),
    }
    Arc::new(out)
}
