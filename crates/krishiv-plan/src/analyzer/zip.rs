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

/// A resolved projection chain: its aliases (innermost first), the outermost list (`None` when the plan is
/// the base itself: a projection may be empty, a zero-column `select()`), and the base under it.
struct Chain {
    aliases: Vec<(BoundColumn, Arc<Expr>)>,
    top: Option<Vec<Arc<Expr>>>,
    base: Arc<LogicalPlan>,
}

/// The alias `e` is, as its output column and its body.
fn alias_of(e: &Expr) -> Option<(&BoundColumn, &Arc<Expr>)> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => Some((output, expr)),
        _ => None,
    }
}

/// `plan` as a chain of projections over a base (the plan itself when it is no projection: a generator,
/// aggregate or window stays in the base, the analyzer keeps them out of a `Project`).
fn chain(plan: &Arc<LogicalPlan>) -> Chain {
    match &plan.node {
        PlanNode::Project { input, exprs } => {
            let mut inner = chain(input);
            inner.aliases.extend(
                exprs
                    .iter()
                    .filter_map(|e| alias_of(e).map(|(o, b)| (o.clone(), Arc::clone(b)))),
            );
            inner.top = Some(exprs.clone());
            inner
        }
        _ => Chain {
            aliases: Vec::new(),
            top: None,
            base: Arc::clone(plan),
        },
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

/// Whether `e` calls a random function without a seed: Spark gives each such call a seed of its own
/// (`ResolveRandomSeed`), so two of them are never one producer even when they read alike.
fn has_unseeded_random(e: &Arc<Expr>) -> bool {
    let mut found = false;
    let _ = crate::walk::visit_distinct(crate::walk::Node::Expr(e), |n| {
        if let crate::walk::Node::Expr(x) = n
            && let ExprNode::Resolved(ResolvedExpr::Function(f)) = &x.node
            && let crate::FunctionId::Registry { name, .. } = &f.id
        {
            // The argument count a call has when it names its seed.
            let seeded_at = match name.as_str() {
                "rand" | "random" | "randn" => Some(1),
                "shuffle" | "randstr" => Some(2),
                "uniform" => Some(3),
                "uuid" => None,
                _ => return std::ops::ControlFlow::Continue(()),
            };
            if seeded_at.is_none_or(|n| f.args.len() < n) {
                found = true;
                return std::ops::ControlFlow::Break(());
            }
        }
        std::ops::ControlFlow::Continue(())
    });
    found
}

/// The ids of the columns `e` reads, those a correlated subquery inside it reads from outside included.
fn reads(e: &Arc<Expr>) -> Vec<AttrId> {
    let mut out = Vec::new();
    let _ = crate::walk::visit_distinct(crate::walk::Node::Expr(e), |n| {
        if let crate::walk::Node::Expr(x) = n
            && let ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c)) =
                &x.node
        {
            out.push(c.id);
        }
        std::ops::ControlFlow::Continue(())
    });
    out
}

/// The ids of the columns a correlated subquery inside `e` reads from outside it.
fn outer_reads(e: &Arc<Expr>) -> Vec<AttrId> {
    let mut out = Vec::new();
    let _ = crate::walk::visit_distinct(crate::walk::Node::Expr(e), |n| {
        if let crate::walk::Node::Expr(x) = n
            && let ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) = &x.node
        {
            out.push(c.id);
        }
        std::ops::ControlFlow::Continue(())
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
        let (lc, rc) = (chain(&l.plan), chain(&r.plan));
        // The same base once resolved, as Spark's `ResolveZip` compares them (`sameResult`): the plans under
        // the projection chains, attribute ids normalised by position. A projection holding an aggregate or
        // a window resolves to an `Aggregate` or `Window` that stays in the base, so it is told apart.
        let same_base = {
            let a = Plan::try_new(Arc::clone(&lc.base));
            let b = Plan::try_new(Arc::clone(&rc.base));
            matches!((a, b), (Ok(a), Ok(b)) if a.structurally_eq_modulo_attr_ids(&b))
        };
        if !same_base {
            return Err(refuse());
        }
        let (Some(lb), Some(rb)) = (lc.base.resolved_output(), rc.base.resolved_output()) else {
            return Err(refuse());
        };
        if lb.len() != rb.len() {
            return Err(refuse());
        }
        // Both outputs are the lists they were resolved to: one expression per column.
        let list = |c: &Chain, rel: &Rel, base: &[BoundColumn]| -> Vec<Arc<Expr>> {
            if let Some(top) = &c.top {
                debug_assert_eq!(top.len(), rel.out.len());
                top.clone()
            } else {
                base.iter().map(|b| col_expr(b, Origin::UNKNOWN)).collect()
            }
        };
        let (l_top, r_top) = (list(&lc, &l, &lb), list(&rc, &r, &rb));
        if l_top.len() != l.out.len() || r_top.len() != r.out.len() {
            return Err(refuse());
        }
        // The right side reads the left base's columns.
        let map: HashMap<AttrId, BoundColumn> = rb
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
            // Equal bodies are one producer, as `ResolveZip` dedups by canonicalized child. A seedless
            // `rand()` has a seed of its own, so two independently written ones stay two.
            let apart = has_unseeded_random(&body);
            // The same output id on both sides (a shared parent) is one producer, seeded or not.
            match kept
                .iter()
                .find(|(k, b)| k.id == output.id || (!apart && **b == *body))
            {
                Some((first, _)) => {
                    if first.id != output.id {
                        survivors.insert(output.id, first.clone());
                    }
                }
                None => kept.push((output.clone(), body)),
            }
        }
        // `remap` rewrites the columns of an expression, not those a correlated subquery inside it reads
        // from outside (`OuterColumn`). A right-side subquery reading a column that was renamed to the left
        // base's, or an alias merged into a survivor, would keep a stale id: refused, never answered wrong.
        let moved =
            |id: &AttrId| survivors.contains_key(id) || map.get(id).is_some_and(|to| to.id != *id);
        let right_exprs = rc.aliases.iter().map(|(_, b)| b).chain(&r_top);
        for e in right_exprs {
            if outer_reads(e).iter().any(moved) {
                return Err(refuse());
            }
        }
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
                    Arc::clone(e)
                };
                // A column a shared producer made redundant is the survivor's value under its
                // own name and identity (Spark's `ResolveZip`), read from the survivor, not computed again.
                let redundant = |c: &BoundColumn| {
                    survivors.get(&c.id).map(|s| {
                        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                            expr: col_expr(s, Origin::UNKNOWN),
                            output: c.clone(),
                        })))
                    })
                };
                let item = match (alias_of(&e), &e.node) {
                    (Some((output, _)), _) => {
                        redundant(output).unwrap_or_else(|| self.pass_expr(output))
                    }
                    (None, ExprNode::Resolved(ResolvedExpr::Column(c))) => {
                        redundant(c).unwrap_or_else(|| Arc::clone(&e))
                    }
                    _ => e,
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
        let shown: Arc<[Attr]> = out.clone().into();
        let mut res = self.projection(o, &rel, items, out);
        // A DataFrame reference to either input (`left["a"]`) still resolves: the inputs' carriers stay,
        // their columns read as the merged plan names them.
        let named = |a: &Attr| {
            let col = map.get(&a.col.id).cloned().unwrap_or_else(|| a.col.clone());
            Attr { col, ..a.clone() }
        };
        let mut carriers = (*res.carriers).clone();
        for side in [&l, &r] {
            for (id, outs) in side.carriers.iter() {
                carriers
                    .entry(*id)
                    .or_default()
                    .extend(outs.iter().map(|c| -> Arc<[Attr]> {
                        c.iter().map(named).collect::<Vec<_>>().into()
                    }));
            }
        }
        if let Some(id) = o.plan_id {
            carriers.insert(id, vec![shown]);
        }
        res.carriers = Arc::new(carriers);
        // The result reads whatever either input reads from the enclosing query.
        res.outer = [l.outer, r.outer].into_iter().flatten().min();
        Ok(res)
    }
}
