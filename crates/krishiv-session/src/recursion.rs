// SPDX-License-Identifier: Apache-2.0
//! Recursive CTEs: [`PlanNode::UnionLoop`] runs as an operator of the
//! physical plan (`krishiv-exec`, `loops.rs`). What the session adds is the
//! row count each loop's readers need: when every read of a loop's CTE sits
//! under a `LIMIT n`, the loop is told to stop after `n` rows
//! ([`push_limits`]).

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::analyzer::AnalyzedPlan;
use krishiv_plan::{LogicalPlan, Plan, PlanNode};

/// The plans of the subqueries inside `e`.
fn subquery_plans(e: &Arc<krishiv_plan::Expr>, out: &mut Vec<Arc<LogicalPlan>>) {
    if let krishiv_plan::ExprNode::Subquery { plan, .. } = &e.node {
        out.push(Arc::clone(plan));
    }
    e.for_each_child(&mut |c| subquery_plans(c, out));
}

/// `plan` with every relation `f` rewrites replaced, in subquery plans too.
fn rewrite(
    plan: Arc<LogicalPlan>,
    f: &mut dyn FnMut(Arc<LogicalPlan>) -> Arc<LogicalPlan>,
) -> Arc<LogicalPlan> {
    fn in_expr(
        e: Arc<krishiv_plan::Expr>,
        f: &mut dyn FnMut(Arc<LogicalPlan>) -> Arc<LogicalPlan>,
    ) -> Arc<krishiv_plan::Expr> {
        let e = e
            .map_children(|c| Ok::<_, std::convert::Infallible>(in_expr(c, f)))
            .unwrap_or_else(|never| match never {});
        if let krishiv_plan::ExprNode::Subquery { kind, plan } = &e.node {
            let replaced = rewrite(Arc::clone(plan), f);
            if !Arc::ptr_eq(&replaced, plan) {
                return Arc::new(krishiv_plan::Expr {
                    node: krishiv_plan::ExprNode::Subquery {
                        kind: kind.clone(),
                        plan: replaced,
                    },
                    origin: e.origin,
                });
            }
        }
        e
    }
    plan.transform_up(&mut |n| {
        let n = f(n);
        Ok::<_, std::convert::Infallible>(
            n.map_expressions(|e| Ok::<_, std::convert::Infallible>(in_expr(e, f)))
                .unwrap_or_else(|never| match never {}),
        )
    })
    .unwrap_or_else(|never| match never {})
}

/// The row count each loop needs when every read of its CTE sits under a
/// `LIMIT n` (through projections and aliases only): Spark pushes the
/// largest into the loop, so an unbounded recursion under `LIMIT n` ends and
/// `LIMIT ALL` (here `i32::MAX`) lifts the row limit. A loop with a read
/// outside any limit (another recursive CTE's included) is absent. Reads in
/// the loop's own recursive term are not outside readers: there the name is
/// defined as the loop's reference, which shadows the CTE.
fn pushed_limits(root: &Arc<LogicalPlan>) -> HashMap<u32, usize> {
    /// Per loop id: the largest limit over its reads, `None` once one read
    /// has none.
    type Reads = HashMap<u32, Option<usize>>;

    /// The definitions in scope, innermost first.
    struct Def {
        name: String,
        /// The loop id when the definition is a loop (through projections).
        id: Option<u32>,
        plan: Arc<LogicalPlan>,
        outer: Defs,
    }
    type Defs = Option<Rc<Def>>;
    type Seen = HashSet<(usize, Option<usize>)>;

    fn through(p: &Arc<LogicalPlan>) -> &Arc<LogicalPlan> {
        let mut d = p;
        while let PlanNode::SubqueryAlias { input, .. } | PlanNode::Project { input, .. } = &d.node
        {
            d = input;
        }
        d
    }

    fn plan(
        p: &Arc<LogicalPlan>,
        limit: Option<usize>,
        defs: &Defs,
        reads: &mut Reads,
        seen: &mut Seen,
    ) {
        if !seen.insert((Arc::as_ptr(p) as usize, limit)) {
            return;
        }
        match &p.node {
            PlanNode::CteRef(r) => {
                let mut cur = defs.as_ref();
                while let Some(d) = cur {
                    if d.name.eq_ignore_ascii_case(&r.name) {
                        match d.id {
                            Some(id) => {
                                reads
                                    .entry(id)
                                    .and_modify(|m| *m = m.zip(limit).map(|(a, b)| a.max(b)))
                                    .or_insert(limit);
                            }
                            // A plain CTE passes its reader's limit on to
                            // what it reads.
                            None => plan(&d.plan, limit, &d.outer, reads, seen),
                        }
                        break;
                    }
                    cur = d.outer.as_ref();
                }
            }
            PlanNode::WithCte { input, ctes } => {
                let mut scope = defs.clone();
                for c in ctes {
                    let id = match &through(&c.plan).node {
                        PlanNode::UnionLoop { id, .. } => Some(*id),
                        _ => None,
                    };
                    // A loop's own terms are read once, without a limit; a
                    // plain definition is read at each of its references.
                    if id.is_some() {
                        plan(&c.plan, None, &scope, reads, seen);
                    }
                    scope = Some(Rc::new(Def {
                        name: c.name.clone(),
                        id,
                        plan: Arc::clone(&c.plan),
                        outer: scope,
                    }));
                }
                plan(input, limit, &scope, reads, seen);
            }
            PlanNode::Limit { input, limit: l } => {
                let n = match &l.node {
                    krishiv_plan::ExprNode::Literal(krishiv_plan::Literal::Integer(n)) => {
                        usize::try_from(*n).ok()
                    }
                    _ => None,
                };
                let next = match (n, limit) {
                    (Some(n), Some(m)) => Some(n.min(m)),
                    (Some(n), None) => Some(n),
                    (None, m) => m,
                };
                plan(input, next, defs, reads, seen);
                p.for_each_expr(&mut |e| expr(e, defs, reads, seen));
            }
            PlanNode::Project { input, .. } | PlanNode::SubqueryAlias { input, .. } => {
                plan(input, limit, defs, reads, seen);
                p.for_each_expr(&mut |e| expr(e, defs, reads, seen));
            }
            _ => {
                p.for_each_child(&mut |c| plan(c, None, defs, reads, seen));
                p.for_each_expr(&mut |e| expr(e, defs, reads, seen));
            }
        }
    }

    fn expr(e: &Arc<krishiv_plan::Expr>, defs: &Defs, reads: &mut Reads, seen: &mut Seen) {
        if let krishiv_plan::ExprNode::Subquery { plan: p, .. } = &e.node {
            plan(p, None, defs, reads, seen);
        }
        e.for_each_child(&mut |c| expr(c, defs, reads, seen));
    }

    let mut reads = Reads::new();
    plan(root, None, &None, &mut reads, &mut HashSet::new());
    reads
        .into_iter()
        .filter_map(|(id, n)| n.map(|n| (id, n)))
        .collect()
}

/// Whether `p` holds a recursive CTE.
pub(crate) fn has_loop(p: &Arc<LogicalPlan>) -> bool {
    fn walk(p: &Arc<LogicalPlan>, seen: &mut HashSet<usize>) -> bool {
        if !seen.insert(Arc::as_ptr(p) as usize) {
            return false;
        }
        if matches!(p.node, PlanNode::UnionLoop { .. }) {
            return true;
        }
        let mut kids = Vec::new();
        p.for_each_child(&mut |c| kids.push(Arc::clone(c)));
        p.for_each_expr(&mut |e| subquery_plans(e, &mut kids));
        kids.iter().any(|k| walk(k, seen))
    }
    walk(p, &mut HashSet::new())
}

/// `analyzed` with each loop told the rows its readers need.
pub(crate) fn push_limits(analyzed: &AnalyzedPlan) -> Result<AnalyzedPlan, KrishivError> {
    let root = Arc::clone(analyzed.plan.root());
    let limits = pushed_limits(&root);
    if limits.is_empty() {
        return Ok(analyzed.clone());
    }
    let plan = rewrite(root, &mut |n| match &n.node {
        PlanNode::UnionLoop {
            id,
            anchor,
            recursive,
            max_level,
            output,
            ..
        } if limits.contains_key(id) => Arc::new(
            LogicalPlan::new(PlanNode::UnionLoop {
                id: *id,
                anchor: Arc::clone(anchor),
                recursive: Arc::clone(recursive),
                max_level: *max_level,
                limit: limits.get(id).and_then(|n| u64::try_from(*n).ok()),
                output: output.clone(),
            })
            .with_origin(n.origin),
        ),
        _ => n,
    });
    Ok(AnalyzedPlan {
        plan: Plan::try_new(plan)?,
        ..analyzed.clone()
    })
}
