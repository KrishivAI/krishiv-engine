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
//! A pair forms from two `GROUP BY` aggregates of the plan with equal
//! grouping, ids included (a derived table written twice: the `Shift` member
//! or an aggregate inside its body, in the body's ids, and the direct copy),
//! when
//! - the outputs both hold are equal expressions, ids included (written in
//!   the same ids, they compute the same column), so the definition may
//!   output both lists;
//! - one's input equals the other's except for extra filter conjuncts (in a
//!   filter of its own or added to one), found only through filters,
//!   projections and inner joins, which pass each row's columns up
//!   unchanged; each extra reads only bare group-key columns of the
//!   aggregate, in the wider output list, and runs on any row (deterministic,
//!   raises on no row, no subquery).
//!
//! A filter of group-key columns commutes with the aggregate: dropping a row
//! drops its whole group. So the definition is the aggregate over the weaker
//! input with both output lists, and each place reads it through a
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

/// One place that reads a pair's definition.
pub(super) struct Use {
    /// The aggregate it replaces.
    pub(super) at: Arc<LogicalPlan>,
    /// Its outputs.
    output: Vec<BoundColumn>,
    /// The conjuncts it applies over the definition.
    extras: Vec<Arc<Expr>>,
}

/// A definition and the two places that read it.
pub(super) struct Pair {
    pub(super) def: Arc<LogicalPlan>,
    pub(super) uses: [Use; 2],
}

/// The pairs of `root`'s aggregates, each in at most one, none of them the
/// body of a member in a group (`grouped`).
pub(super) fn pairs(
    root: &Arc<LogicalPlan>,
    members: &[Member],
    grouped: &HashSet<usize>,
) -> Vec<Pair> {
    let excluded: HashSet<*const LogicalPlan> = grouped
        .iter()
        .filter_map(|&i| members.get(i))
        .map(|m| Arc::as_ptr(&m.body))
        .collect();
    let mut aggs: Vec<Arc<LogicalPlan>> = Vec::new();
    walk(root, &mut |p| {
        if matches!(
            p.node,
            PlanNode::Aggregate {
                grouping: Grouping::GroupBy(_),
                ..
            }
        ) && !excluded.contains(&Arc::as_ptr(p))
        {
            aggs.push(Arc::clone(p));
        }
    });
    let mut taken = vec![false; aggs.len()];
    let mut out = Vec::new();
    for i in 0..aggs.len() {
        for j in i + 1..aggs.len() {
            if taken[i] {
                break;
            }
            if taken[j] {
                continue;
            }
            if let Some(pair) = pair(&aggs[i], &aggs[j]) {
                taken[i] = true;
                taken[j] = true;
                out.push(pair);
            }
        }
    }
    out
}

/// The pair of the aggregates `b` and `d`, if they compute one aggregate up
/// to outputs and key filters. Their grouping is equal, ids included, only
/// when the analysis wrote both in the same ids (a derived table written
/// twice); two references to one `Arc` are not a pair.
fn pair(b: &Arc<LogicalPlan>, d: &Arc<LogicalPlan>) -> Option<Pair> {
    let PlanNode::Aggregate {
        input: bi,
        grouping: bg @ Grouping::GroupBy(keys),
        aggregates: ba,
    } = &b.node
    else {
        return None;
    };
    let PlanNode::Aggregate {
        input: di,
        grouping: dg,
        aggregates: da,
    } = &d.node
    else {
        return None;
    };
    if Arc::ptr_eq(b, d) || dg != bg {
        return None;
    }
    let (b_extra, d_extra, weak) = if let Some(x) = extras(di, bi) {
        (x, Vec::new(), di)
    } else {
        (Vec::new(), extras(bi, di)?, bi)
    };
    // Both output lists: the weaker side's, then the other's it lacks.
    let (first, second) = if Arc::ptr_eq(weak, di) {
        (da, ba)
    } else {
        (ba, da)
    };
    // In the order of the list holding the other, if one does.
    let wide = if first.iter().all(|e| second.contains(e)) {
        second.clone()
    } else {
        let mut w = first.clone();
        w.extend(second.iter().filter(|e| !first.contains(e)).cloned());
        w
    };
    let key_ids: Attrs = keys
        .iter()
        .filter_map(|k| match &k.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
            _ => None,
        })
        .collect();
    let def = if wide == *first {
        Arc::clone(if Arc::ptr_eq(weak, di) { d } else { b })
    } else {
        Arc::new(LogicalPlan::new(PlanNode::Aggregate {
            input: Arc::clone(weak),
            grouping: bg.clone(),
            aggregates: wide,
        }))
    };
    let def_out = def.resolved_output()?;
    let def_ids: HashSet<AttrId> = def_out.iter().map(|c| c.id).collect();
    let fits = |e: &Arc<Expr>| {
        runs_on_any_row(e, true)
            && references(e)
                .iter()
                .all(|id| key_ids.contains(id) && def_ids.contains(id))
    };
    if !b_extra.iter().chain(&d_extra).all(fits) || !worth_sharing(&def) {
        return None;
    }
    Some(Pair {
        uses: [
            Use {
                at: Arc::clone(b),
                output: b.resolved_output()?,
                extras: b_extra,
            },
            Use {
                at: Arc::clone(d),
                output: d.resolved_output()?,
                extras: d_extra,
            },
        ],
        def,
    })
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
