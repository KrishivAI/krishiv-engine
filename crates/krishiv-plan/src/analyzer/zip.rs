// SPDX-License-Identifier: Apache-2.0
//! `DataFrame.zip` (Spark 4.3's `Zip`, resolved by `ResolveZip`): the columns of two DataFrames side by
//! side, row by row.
//!
//! Both inputs must derive from one base through chains of projections (`select`, `withColumn`, ...).
//! Their bases must be the same plan; the two chains are merged into one layered chain of
//! [`PlanNode::Project`]s over that base, one layer per dependency depth of the aliases either chain
//! defines (a producer both sides share is kept once), and a last projection with the columns of the left
//! input followed by those of the right. Two inputs that cannot be merged are
//! `ZIP_PLANS_NOT_MERGEABLE`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use krishiv_common::error::ZipPlansNotMergeable;

use super::AnalysisError;
use super::ops::col_expr;
use super::resolve::{Attr, Env, Rel, Resolver};
use crate::{
    AttrId, BoundColumn, Expr, ExprNode, LogicalPlan, Origin, Plan, PlanNode, ResolvedExpr,
};

/// A resolved projection chain: its aliases (innermost first), the outermost list, and the base under it.
struct Chain {
    aliases: Vec<(BoundColumn, Arc<Expr>)>,
    top: Vec<Arc<Expr>>,
    base: Arc<LogicalPlan>,
}

/// The alias `e` is, as its output column and its body.
fn alias_of(e: &Expr) -> Option<(&BoundColumn, &Arc<Expr>)> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => Some((output, expr)),
        _ => None,
    }
}

/// `plan` as a chain of projections over a base; `None` when a projection does something else than
/// compute one value per row (the analyzer keeps generators, aggregates and windows out of a `Project`).
fn chain(plan: &Arc<LogicalPlan>) -> Chain {
    match &plan.node {
        PlanNode::Project { input, exprs } => {
            let mut inner = chain(input);
            inner.aliases.extend(
                exprs
                    .iter()
                    .filter_map(|e| alias_of(e).map(|(o, b)| (o.clone(), Arc::clone(b)))),
            );
            inner.top.clone_from(exprs);
            inner
        }
        _ => Chain {
            aliases: Vec::new(),
            top: Vec::new(),
            base: Arc::clone(plan),
        },
    }
}

/// The unresolved plan under its projection chain: what both inputs must share.
fn unresolved_base(plan: &Arc<LogicalPlan>) -> &Arc<LogicalPlan> {
    match &plan.node {
        PlanNode::Project { input, .. } | PlanNode::WithColumns { input, .. } => {
            unresolved_base(input)
        }
        _ => plan,
    }
}

/// `e` with each column in `map` replaced by its target.
fn remap(e: &Arc<Expr>, map: &HashMap<AttrId, BoundColumn>) -> Arc<Expr> {
    if map.is_empty() {
        return Arc::clone(e);
    }
    let done: Result<_, std::convert::Infallible> = Arc::clone(e).transform_up(&mut |x| {
        Ok(match &x.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => match map.get(&c.id) {
                Some(to) => col_expr(to, x.origin),
                None => x,
            },
            _ => x,
        })
    });
    done.unwrap_or_else(|never| match never {})
}

/// The ids of the columns `e` reads.
fn reads(e: &Arc<Expr>) -> Vec<AttrId> {
    let mut out = Vec::new();
    let _: Result<_, std::convert::Infallible> = Arc::clone(e).transform_up(&mut |x| {
        if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &x.node {
            out.push(c.id);
        }
        Ok(x)
    });
    out
}

impl Resolver<'_> {
    /// A `Zip` as described in the module docs. Out of line: it keeps the recursive `node` frame small.
    #[inline(never)]
    pub(super) fn zip_relations(
        &mut self,
        p: &Arc<LogicalPlan>,
        left: &Arc<LogicalPlan>,
        right: &Arc<LogicalPlan>,
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let o = p.origin;
        let refuse = || AnalysisError::new(ZipPlansNotMergeable {}, o);
        let l = self.child(left, env)?;
        let r = self.child(right, env)?;
        // The same base: what is under the projections, as written.
        let same_base = {
            let a = Plan::try_new(Arc::clone(unresolved_base(left)));
            let b = Plan::try_new(Arc::clone(unresolved_base(right)));
            matches!((a, b), (Ok(a), Ok(b)) if a.structurally_eq(&b))
        };
        if !same_base {
            return Err(refuse());
        }
        let (lc, rc) = (chain(&l.plan), chain(&r.plan));
        let (Some(lb), Some(rb)) = (lc.base.resolved_output(), rc.base.resolved_output()) else {
            return Err(refuse());
        };
        if lb.len() != rb.len() {
            return Err(refuse());
        }
        // Both outputs are the lists they were resolved to: one expression per column.
        let list = |c: &Chain, rel: &Rel, base: &[BoundColumn]| -> Vec<Arc<Expr>> {
            if c.top.is_empty() {
                base.iter().map(|b| col_expr(b, Origin::UNKNOWN)).collect()
            } else {
                debug_assert_eq!(c.top.len(), rel.out.len());
                c.top.clone()
            }
        };
        let (l_top, r_top) = (list(&lc, &l, &lb), list(&rc, &r, &rb));
        if l_top.len() != l.out.len() || r_top.len() != r.out.len() {
            return Err(refuse());
        }
        // The right side reads the left base's columns.
        let mut map: HashMap<AttrId, BoundColumn> = rb
            .iter()
            .zip(&lb)
            .map(|(from, to)| (from.id, to.clone()))
            .collect();
        // Every alias, once per producer: a producer both sides share is kept for the first.
        let mut kept: Vec<(BoundColumn, Arc<Expr>)> = Vec::new();
        let mut survivors: HashMap<AttrId, BoundColumn> = HashMap::new();
        let all = lc.aliases.iter().chain(&rc.aliases).map(|(o, b)| (o, b));
        for (output, body) in all {
            let body = remap(&remap(body, &map), &survivors);
            match kept.iter().find(|(_, b)| **b == *body) {
                Some((first, _)) => {
                    survivors.insert(output.id, first.clone());
                }
                None => kept.push((output.clone(), body)),
            }
        }
        map.extend(survivors.iter().map(|(k, v)| (*k, v.clone())));
        // One layer per dependency depth.
        let by_id: BTreeMap<AttrId, usize> = kept
            .iter()
            .enumerate()
            .map(|(i, (o, _))| (o.id, i))
            .collect();
        let mut depth = vec![0_usize; kept.len()];
        for i in 0..kept.len() {
            // An alias reads only aliases defined before it, so one forward pass settles every depth.
            let d = reads(&kept[i].1)
                .iter()
                .filter_map(|id| by_id.get(id))
                .map(|j| depth[*j])
                .max()
                .unwrap_or(0);
            depth[i] = d + 1;
        }
        let layers = depth.iter().copied().max().unwrap_or(0);
        let base_attrs: Vec<Attr> = lb
            .iter()
            .map(|c| Attr {
                col: c.clone(),
                qual: Arc::from(Vec::<String>::new()),
                source: None,
                duplicate: false,
            })
            .collect();
        let mut rel = self.wrap(Arc::clone(&lc.base), None, base_attrs.clone(), &[]);
        let mut current = base_attrs;
        for layer in 1..=layers {
            let mut items: Vec<Arc<Expr>> =
                current.iter().map(|a| self.pass_expr(&a.col)).collect();
            let mut out = current.clone();
            for (i, (output, body)) in kept.iter().enumerate() {
                if depth[i] != layer {
                    continue;
                }
                items.push(Arc::new(Expr::new(ExprNode::Resolved(
                    ResolvedExpr::Alias {
                        expr: Arc::clone(body),
                        output: output.clone(),
                    },
                ))));
                out.push(Attr {
                    col: output.clone(),
                    qual: Arc::from(Vec::<String>::new()),
                    source: None,
                    duplicate: false,
                });
            }
            rel = self.projection(o, &rel, items, out.clone());
            current = out;
        }
        // The columns of the left input, then those of the right.
        let mut items = Vec::with_capacity(l_top.len() + r_top.len());
        let mut out = Vec::with_capacity(items.capacity());
        let sides = [(&l_top, &l, false), (&r_top, &r, true)];
        for (top, side, from_right) in sides {
            for (e, attr) in top.iter().zip(side.out.iter()) {
                let e = if from_right {
                    remap(e, &map)
                } else {
                    remap(e, &survivors)
                };
                let item = match alias_of(&e) {
                    // An alias kept by the layers is read from there; one a shared producer made
                    // redundant is the survivor's value under its own name and identity.
                    Some((output, body)) if survivors.contains_key(&output.id) => {
                        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                            expr: Arc::clone(body),
                            output: output.clone(),
                        })))
                    }
                    Some((output, _)) => self.pass_expr(output),
                    None => e,
                };
                let col = match (&item.node, alias_of(&item)) {
                    (_, Some((output, _))) => output.clone(),
                    (ExprNode::Resolved(ResolvedExpr::Column(c)), None) => c.clone(),
                    _ => attr.col.clone(),
                };
                items.push(item);
                out.push(Attr {
                    col,
                    qual: Arc::clone(&attr.qual),
                    source: attr.source.clone(),
                    duplicate: attr.duplicate,
                });
            }
        }
        Ok(self.projection(o, &rel, items, out))
    }
}
