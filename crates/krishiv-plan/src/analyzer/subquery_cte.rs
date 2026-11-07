// SPDX-License-Identifier: Apache-2.0
//! `WITH` inside a correlated subquery (`(WITH t AS (SELECT * FROM t2 WHERE
//! c1 = t1.c1) SELECT SUM(c2) FROM t)`): the subquery's own definitions are
//! inlined before it is decorrelated, as Spark's `InlineCTE` does for a
//! subquery's CTEs, so the correlation reaches the operators the rewrite
//! knows. A reference becomes a projection renaming the definition's
//! columns to its own. A definition read again gets fresh ids: a
//! [`PlanNode::Shift`] when it reads no enclosing query, the correlated
//! instance (`instance.rs`) when it does. A nondeterministic definition read
//! twice stays a CTE (it is evaluated once), and that subquery stays refused.

use std::collections::HashMap;
use std::sync::Arc;

use super::AnalysisError;
use super::bind::unsupported;
use super::instance::{Span, renumbered};
use super::ops::{MAX_ID, col_expr};
use super::subquery::{has_outer, hull, node, plan_is_deterministic};
use krishiv_common::error::FailedToParseTooComplex;

use crate::walk::Node;
use crate::{
    AttrId, BoundColumn, CteRef, Expr, ExprNode, LogicalPlan, Origin, PlanNode, ResolvedExpr,
};

type R<T> = Result<T, AnalysisError>;

/// One definition being inlined.
struct Def {
    source: Vec<AttrId>,
    plan: Arc<LogicalPlan>,
    reads: usize,
}

/// `plan`, a correlated subquery's plan, with each correlated `WITH` in it
/// inlined; `next_id` is past every id in use and stays so.
pub(super) fn inline(plan: &Arc<LogicalPlan>, next_id: &mut u64) -> R<Arc<LogicalPlan>> {
    if !has_outer(Node::Plan(plan)) || !has_with(plan) {
        return Ok(Arc::clone(plan));
    }
    Arc::clone(plan).transform_up(&mut |p: Arc<LogicalPlan>| match &p.node {
        PlanNode::WithCte { input, ctes } if has_outer(Node::Plan(&p)) => {
            let mut defs: Vec<Def> = Vec::new();
            for c in ctes {
                let body = references(&c.plan, &mut defs, next_id)?;
                let source = c
                    .plan
                    .resolved_output()
                    .ok_or_else(|| unsupported("a subquery over an unresolved plan", p.origin))?
                    .into_iter()
                    .map(|b| b.id)
                    .collect();
                defs.push(Def {
                    source,
                    plan: body,
                    reads: 0,
                });
            }
            references(input, &mut defs, next_id)
        }
        _ => Ok(p),
    })
}

/// The span of attribute ids `p` defines, subquery plans inside its
/// expressions included ([`hull`] leaves those out, but a copy of `p` shifts
/// them too, so its offset and reservation must cover them).
fn full_hull(p: &Arc<LogicalPlan>) -> Option<(u64, u64)> {
    let mut span = hull(p);
    let mut subs: Vec<Arc<LogicalPlan>> = Vec::new();
    let _ = crate::walk::visit_distinct(Node::Plan(p), |n| {
        if let Node::Expr(e) = n
            && let ExprNode::Subquery { plan, .. } = &e.node
        {
            subs.push(Arc::clone(plan));
        }
        std::ops::ControlFlow::<()>::Continue(())
    });
    for sub in &subs {
        span = match (span, hull(sub)) {
            (Some((a, b)), Some((c, d))) => Some((a.min(c), b.max(d))),
            (x, y) => x.or(y),
        };
    }
    span
}

/// Moves `next_id` past `top` (the highest id of a copy just made), or fails
/// as too complex when that passes [`MAX_ID`].
fn reserve_past(next_id: &mut u64, top: Option<u64>, origin: Origin) -> R<()> {
    if let Some(t) = top {
        if t >= MAX_ID {
            return Err(AnalysisError::new(FailedToParseTooComplex {}, origin));
        }
        *next_id = (*next_id).max(t + 1);
    }
    Ok(())
}

fn has_with(plan: &Arc<LogicalPlan>) -> bool {
    crate::walk::visit_distinct(Node::Plan(plan), |n| match n {
        Node::Plan(p) if matches!(p.node, PlanNode::WithCte { .. }) => {
            std::ops::ControlFlow::Break(())
        }
        _ => std::ops::ControlFlow::Continue(()),
    })
    .is_break()
}

/// `p` with every reference to one of `defs` (subquery plans included)
/// replaced by the definition.
fn references(p: &Arc<LogicalPlan>, defs: &mut Vec<Def>, next_id: &mut u64) -> R<Arc<LogicalPlan>> {
    Arc::clone(p).transform_up(&mut |n: Arc<LogicalPlan>| {
        let n = n.map_expressions(|e| {
            e.transform_up(&mut |x: Arc<Expr>| match &x.node {
                ExprNode::Subquery { kind, plan } => {
                    let plan = references(plan, defs, next_id)?;
                    let node = ExprNode::Subquery {
                        kind: kind.clone(),
                        plan,
                    };
                    Ok(Arc::new(Expr::new(node).with_origin(x.origin)))
                }
                _ => Ok(x),
            })
        })?;
        match &n.node {
            PlanNode::CteRef(r) => match defs.iter_mut().find(|d| d.source == r.source) {
                Some(d) => read(d, r, n.origin, next_id),
                None => Ok(n),
            },
            // The analyzer's instance of a reference read twice (`t a JOIN
            // t b`): over a correlated definition a `Shift` would renumber
            // the outer references too, so it becomes the correlated
            // instance (every id but theirs shifted).
            PlanNode::Shift { input, offset } if has_outer(Node::Plan(input)) => {
                // A nondeterministic definition is one value read twice.
                if !plan_is_deterministic(input) {
                    return Err(unsupported(
                        "a CTE read twice in a correlated subquery",
                        n.origin,
                    ));
                }
                let span = Span {
                    lo: 0,
                    hi: MAX_ID,
                    offset: *offset,
                };
                let copy = renumbered(input, span, &mut HashMap::new())?;
                reserve_past(next_id, full_hull(&copy).map(|(_, hi)| hi), n.origin)?;
                Ok(copy)
            }
            _ => Ok(n),
        }
    })
}

/// The reference `r` to `d` as a projection over the definition (a fresh
/// instance of it after the first read).
fn read(d: &mut Def, r: &CteRef, origin: Origin, next_id: &mut u64) -> R<Arc<LogicalPlan>> {
    let refused = || unsupported("a CTE read twice in a correlated subquery", origin);
    let (body, span) = if d.reads == 0 {
        (Arc::clone(&d.plan), None)
    } else {
        if !plan_is_deterministic(&d.plan) {
            return Err(refused());
        }
        let (lo, hi) = full_hull(&d.plan).ok_or_else(refused)?;
        let offset = next_id.saturating_sub(lo);
        if hi.checked_add(offset).is_none_or(|top| top > MAX_ID) {
            return Err(AnalysisError::new(FailedToParseTooComplex {}, origin));
        }
        *next_id += hi - lo + 1;
        let span = Span { lo, hi, offset };
        let body = if has_outer(Node::Plan(&d.plan)) {
            renumbered(&d.plan, span, &mut HashMap::new())?
        } else {
            node(
                PlanNode::Shift {
                    input: Arc::clone(&d.plan),
                    offset,
                },
                Origin::UNKNOWN,
            )
        };
        (body, Some(span))
    };
    d.reads += 1;
    let out = d
        .plan
        .resolved_output()
        .ok_or_else(|| unsupported("a subquery over an unresolved plan", origin))?;
    if out.len() != r.output.len() {
        return Err(refused());
    }
    let exprs = out
        .iter()
        .zip(&r.output)
        .map(|(c, want)| {
            let c: BoundColumn = span.map_or_else(|| c.clone(), |s| s.column(c));
            Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                expr: col_expr(&c, Origin::UNKNOWN),
                output: want.clone(),
            })))
        })
        .collect();
    Ok(node(PlanNode::Project { input: body, exprs }, origin))
}
