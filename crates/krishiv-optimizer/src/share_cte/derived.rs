// SPDX-License-Identifier: Apache-2.0
//! An aggregate computed twice with different outputs or filters (perf-3
//! lever D1, TPC-DS q65), part of `ShareCte` under `direct`.
//!
//! The analysis cache makes a derived table written twice one body read
//! through a `Shift` and once directly, but pruning and inferred join filters
//! then change the two copies differently: q65's `sa` (under `sb`) outputs
//! `[ss_store_sk, revenue]` and its `sc` copy `[ss_store_sk, ss_item_sk,
//! revenue]` with `isnotnull(ss_item_sk)` from the join with `item` added to
//! its `store_sales` filter. Neither is equal to the other, so no group forms.
//!
//! A share forms from two or more `GROUP BY` aggregates of the plan with
//! equal grouping, ids included (a derived table written more than once: the
//! `Shift` member or an aggregate inside its body, in the body's ids, and the
//! direct copies), when
//! - the outputs two of them hold are equal expressions, ids included
//!   (written in the same ids, they compute the same column), so the
//!   definition may output every list;
//! - each one's input equals one weakest input except for extra filter
//!   conjuncts (in a filter of its own or added to one), found only through filters,
//!   projections and inner joins, which pass each row's columns up
//!   unchanged; each extra reads only bare group-key columns of the
//!   aggregate, in the wider output list, and runs on any row (deterministic,
//!   raises on no row, no subquery).
//!
//! A filter of group-key columns commutes with the aggregate: dropping a row
//! drops its whole group. So the definition is the aggregate over the weakest
//! input with every output list, and each place reads it through a
//! `CteRef`, under its extra conjuncts and, when narrower, a column
//! projection to its own outputs.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_plan::{
    AttrId, BoundColumn, CteRef, Expr, ExprNode, Grouping, JoinKind, LogicalPlan, PlanNode,
    ResolvedExpr, runs_on_any_row,
};

use super::{Member, walk, worth_sharing};
use crate::expr::{Attrs, and_all, conjuncts, references};

/// One place that reads a shared definition.
pub(super) struct Use {
    /// The aggregate it replaces.
    pub(super) at: Arc<LogicalPlan>,
    /// Its outputs.
    output: Vec<BoundColumn>,
    /// The conjuncts it applies over the definition.
    extras: Vec<Arc<Expr>>,
}

/// A definition, its outputs and the places that read it (two or more).
pub(super) struct Shared {
    pub(super) def: Arc<LogicalPlan>,
    pub(super) out: Vec<BoundColumn>,
    pub(super) uses: Vec<Use>,
}

/// The aggregates (indices into the candidates) that extend one input, each
/// with the conjuncts it filters by beyond it.
type Over = Vec<(usize, Vec<Arc<Expr>>)>;

/// A `GROUP BY` aggregate of the plan, its parts.
struct Agg {
    plan: Arc<LogicalPlan>,
    input: Arc<LogicalPlan>,
    grouping: Grouping,
    keys: Vec<Arc<Expr>>,
    aggregates: Vec<Arc<Expr>>,
}

/// The shared aggregates of `root`, each aggregate in at most one, none of
/// them inside the body of a member in a group (`grouped`). In walk order,
/// each aggregate not yet taken shares with every later one that computes it
/// (a derived table written three times shares all three).
pub(super) fn shared(
    root: &Arc<LogicalPlan>,
    members: &[Member],
    grouped: &HashSet<usize>,
) -> Vec<Shared> {
    let mut excluded: HashSet<*const LogicalPlan> = HashSet::new();
    for m in grouped.iter().filter_map(|&i| members.get(i)) {
        walk(&m.body, &mut |p| {
            excluded.insert(Arc::as_ptr(p));
        });
    }
    let mut rest: Vec<Agg> = Vec::new();
    walk(root, &mut |p| {
        if let PlanNode::Aggregate {
            input,
            grouping: grouping @ Grouping::GroupBy(keys),
            aggregates,
        } = &p.node
            && !excluded.contains(&Arc::as_ptr(p))
        {
            rest.push(Agg {
                plan: Arc::clone(p),
                input: Arc::clone(input),
                grouping: grouping.clone(),
                keys: keys.clone(),
                aggregates: aggregates.clone(),
            });
        }
    });
    let mut out = Vec::new();
    let mut i = 0;
    while i < rest.len() {
        if let Some((taken, p)) = share(&rest, i) {
            for &j in taken.iter().rev() {
                rest.remove(j);
            }
            out.push(p);
        } else {
            i += 1;
        }
    }
    out
}

/// The aggregate `rest[i]` shared with the later ones that compute it up to
/// outputs and key filters, and their indices (ascending), if any does.
/// Their grouping is equal, ids included, only when the analysis wrote them
/// in the same ids (a derived table written more than once); two references
/// to one `Arc` are one aggregate. The definition reads the weakest input:
/// the one among them that most of them extend, `rest[i]`'s included.
fn share(rest: &[Agg], i: usize) -> Option<(Vec<usize>, Shared)> {
    let a = &rest[i];
    let same: Vec<usize> =
        std::iter::once(i)
            .chain((i + 1..rest.len()).filter(|&j| {
                rest[j].grouping == a.grouping && !Arc::ptr_eq(&rest[j].plan, &a.plan)
            }))
            .collect();
    let mut best: Option<(usize, Over)> = None;
    for &w in &same {
        let over: Over = same
            .iter()
            .filter_map(|&x| extras(&rest[w].input, &rest[x].input).map(|e| (x, e)))
            .collect();
        if over.first().is_some_and(|(x, _)| *x == i)
            && over.len() >= 2
            && best.as_ref().is_none_or(|(_, b)| over.len() > b.len())
        {
            best = Some((w, over));
        }
    }
    let (w, over) = best?;
    let weak = &rest[w];
    // Every output list in one: the weak side's, widened by each other's in
    // turn (in the order of a list that holds the one so far).
    let mut wide = weak.aggregates.clone();
    for (x, _) in &over {
        let list = &rest[*x].aggregates;
        if wide.iter().all(|e| list.contains(e)) {
            wide.clone_from(list);
        } else {
            let more: Vec<_> = list.iter().filter(|e| !wide.contains(e)).cloned().collect();
            wide.extend(more);
        }
    }
    let key_ids: Attrs = a
        .keys
        .iter()
        .filter_map(|k| match &k.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
            _ => None,
        })
        .collect();
    let def = if wide == weak.aggregates {
        Arc::clone(&weak.plan)
    } else {
        Arc::new(LogicalPlan::new(PlanNode::Aggregate {
            input: Arc::clone(&weak.input),
            grouping: a.grouping.clone(),
            aggregates: wide,
        }))
    };
    let out = def.resolved_output()?;
    let def_ids: HashSet<AttrId> = out.iter().map(|c| c.id).collect();
    let fits = |e: &Arc<Expr>| {
        runs_on_any_row(e, true)
            && references(e)
                .iter()
                .all(|id| key_ids.contains(id) && def_ids.contains(id))
    };
    if !over.iter().flat_map(|(_, e)| e).all(fits) || !worth_sharing(&def) {
        return None;
    }
    let mut taken = Vec::with_capacity(over.len());
    let mut uses = Vec::with_capacity(over.len());
    for (x, extras) in over {
        let at = &rest[x].plan;
        uses.push(Use {
            at: Arc::clone(at),
            output: at.resolved_output()?,
            extras,
        });
        taken.push(x);
    }
    Some((taken, Shared { def, out, uses }))
}

/// The conjuncts `strong` filters by beyond `weak`, when the two are equal
/// otherwise; descends only through filters, projections and inner joins.
fn extras(weak: &Arc<LogicalPlan>, strong: &Arc<LogicalPlan>) -> Option<Vec<Arc<Expr>>> {
    if Arc::ptr_eq(weak, strong) || **weak == **strong {
        return Some(Vec::new());
    }
    match (&weak.node, &strong.node) {
        (
            PlanNode::Filter {
                input: wi,
                predicate: wp,
            },
            PlanNode::Filter {
                input: si,
                predicate: sp,
            },
        ) => {
            // The weak conjuncts, in order, among the strong ones.
            let want = conjuncts(wp);
            let mut next = want.iter().peekable();
            let mut out = Vec::new();
            for c in conjuncts(sp) {
                if next.peek().is_some_and(|w| **w == c) {
                    next.next();
                } else {
                    out.push(c);
                }
            }
            if next.peek().is_some() {
                return extras(weak, si).map(|mut below| {
                    below.splice(0..0, conjuncts(sp));
                    below
                });
            }
            let mut below = extras(wi, si)?;
            out.append(&mut below);
            Some(out)
        }
        (
            _,
            PlanNode::Filter {
                input: si,
                predicate: sp,
            },
        ) => {
            let mut out = conjuncts(sp);
            out.append(&mut extras(weak, si)?);
            Some(out)
        }
        (
            PlanNode::Project {
                input: wi,
                exprs: we,
            },
            PlanNode::Project {
                input: si,
                exprs: se,
            },
        ) if we == se => extras(wi, si),
        (
            PlanNode::Join {
                left: wl,
                right: wr,
                kind: JoinKind::Inner,
                condition: wc,
            },
            PlanNode::Join {
                left: sl,
                right: sr,
                kind: JoinKind::Inner,
                condition: sc,
            },
        ) if wc == sc => {
            let mut out = extras(wl, sl)?;
            out.append(&mut extras(wr, sr)?);
            Some(out)
        }
        _ => None,
    }
}

/// What replaces `u.at`: the definition `name` (outputs `def_out`) read
/// through a `CteRef`, under `u`'s extra conjuncts, projected to `u`'s
/// outputs when they differ from the definition's.
pub(super) fn reference(name: &str, def_out: &[BoundColumn], u: &Use) -> Arc<LogicalPlan> {
    let mut node = Arc::new(LogicalPlan::new(PlanNode::CteRef(CteRef {
        name: name.to_owned(),
        source: def_out.iter().map(|c| c.id).collect(),
        output: def_out.to_vec(),
    })));
    if let Some(predicate) = and_all(u.extras.iter().cloned()) {
        node = Arc::new(LogicalPlan::new(PlanNode::Filter {
            input: node,
            predicate,
        }));
    }
    if !u
        .output
        .iter()
        .map(|c| c.id)
        .eq(def_out.iter().map(|c| c.id))
    {
        let exprs = u
            .output
            .iter()
            .map(|c| {
                Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
                    c.clone(),
                ))))
            })
            .collect();
        node = Arc::new(LogicalPlan::new(PlanNode::Project { input: node, exprs }));
    }
    node
}

#[cfg(test)]
mod tests;
