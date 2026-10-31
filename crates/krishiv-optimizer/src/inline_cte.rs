// SPDX-License-Identifier: Apache-2.0
//! CTE inlining (Spark's `InlineCTE`): a reference to a `WITH` definition
//! becomes a fresh instance of the definition, so filters, column pruning
//! and join ordering see through it, per reference.
//!
//! A definition is inlined when it is deterministic or referenced once. A
//! nondeterministic definition with several references stays a
//! [`PlanNode::CteRef`], kept for P2-T198 (a CTE computed once and shared).
//! Spark 4.2 inlines it too, with one shared seed per nondeterministic call
//! (`WITH t AS (SELECT rand() r FROM range(3)) SELECT count(*) FROM t a JOIN
//! t b ON a.r = b.r` is 3); the physical planner computes the kept
//! reference's body once per query and every reference reads it
//! (a `SharedRead`), so Krishiv returns 3 as well. A definition
//! that holds an outer reference is inlined whatever its determinism, as
//! Spark's `shouldInline` does.
//!
//! Determinism is that of the definition's own expressions, before its own
//! references are inlined: a deterministic definition that reads a
//! nondeterministic one is inlined, and the nondeterministic reference
//! inside each instance stays a reference (Spark's observed result is the
//! same).
//!
//! An instance is `Project[source + offset AS output]` over
//! `Shift { definition, offset }`: the shift gives the copy attribute ids no
//! other part of the plan holds (each instance takes the next free range
//! above the plan's largest id), and the projection gives the copy the
//! reference's own output ids, so nothing above the reference changes.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{
    AttrId, BoundColumn, CteRef, Expr, ExprNode, LogicalPlan, PlanNode, ResolvedExpr,
};

use crate::dag;
use crate::expr::{has_outer_reference, is_deterministic};
use crate::restrict::{MAX_ID, max_id, subquery_plans};

/// `root` with its CTE references inlined where Spark inlines them.
pub(crate) fn inline(root: Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, KrishivError> {
    // Most plans have no `WITH`: one walk, then nothing to do.
    if !has_with(&root) {
        return Ok(root);
    }
    let mut defs: HashMap<Vec<AttrId>, Arc<LogicalPlan>> = HashMap::new();
    let mut refs: HashMap<Vec<AttrId>, usize> = HashMap::new();
    survey(&root, &mut defs, &mut refs);
    let inline: HashSet<Vec<AttrId>> = defs
        .iter()
        .filter(|(key, def)| {
            // Spark's `shouldInline`: a definition holding an outer reference cannot be computed once per
            // query, so it is inlined however it is read.
            refs.get(*key).copied().unwrap_or(0) <= 1 || deterministic(def) || has_outer(def)
        })
        .map(|(key, _)| key.clone())
        .collect();
    if inline.is_empty() {
        return Ok(root);
    }
    let next = max_id(&root, &mut HashMap::new()).saturating_add(1);
    let mut inliner = Inliner {
        defs,
        inline,
        bodies: HashMap::new(),
        next,
    };
    inliner.plan(root)
}

/// Whether a `WITH` is anywhere in `root` (subquery plans included).
fn has_with(root: &Arc<LogicalPlan>) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(root)];
    while let Some(p) = stack.pop() {
        if matches!(p.node, PlanNode::WithCte { .. }) {
            return true;
        }
        if !seen.insert(Arc::as_ptr(&p)) {
            continue;
        }
        p.for_each_expr(&mut |e| stack.extend(subquery_plans(e)));
        p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    false
}

/// Every definition (by its output ids) and how many references read it,
/// subquery plans included. A subtree shared through `Arc` (a self-join
/// instance under a `Shift`) counts once per place it is used: two places
/// read the definition twice.
fn survey(
    root: &Arc<LogicalPlan>,
    defs: &mut HashMap<Vec<AttrId>, Arc<LogicalPlan>>,
    refs: &mut HashMap<Vec<AttrId>, usize>,
) {
    let mut memo = HashMap::new();
    *refs = occurrences(root, defs, &mut memo);
}

/// The references under `p`, per definition, counted per path from `p`.
fn occurrences(
    p: &Arc<LogicalPlan>,
    defs: &mut HashMap<Vec<AttrId>, Arc<LogicalPlan>>,
    memo: &mut HashMap<*const LogicalPlan, HashMap<Vec<AttrId>, usize>>,
) -> HashMap<Vec<AttrId>, usize> {
    if let Some(m) = memo.get(&Arc::as_ptr(p)) {
        return m.clone();
    }
    let mut out: HashMap<Vec<AttrId>, usize> = HashMap::new();
    match &p.node {
        PlanNode::WithCte { ctes, .. } => {
            for c in ctes {
                if let Some(o) = c.plan.resolved_output() {
                    defs.insert(o.iter().map(|x| x.id).collect(), Arc::clone(&c.plan));
                }
            }
        }
        PlanNode::CteRef(r) => *out.entry(r.source.clone()).or_insert(0) += 1,
        _ => {}
    }
    let mut kids = Vec::new();
    p.for_each_expr(&mut |e| kids.extend(subquery_plans(e)));
    p.for_each_child(&mut |c| kids.push(Arc::clone(c)));
    for k in &kids {
        for (key, n) in occurrences(k, defs, memo) {
            let slot = out.entry(key).or_insert(0);
            *slot = slot.saturating_add(n);
        }
    }
    memo.insert(Arc::as_ptr(p), out.clone());
    out
}

/// Whether an expression of `plan` (subquery plans included) holds an outer reference.
fn has_outer(plan: &Arc<LogicalPlan>) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(plan)];
    let mut found = false;
    while let Some(p) = stack.pop() {
        if found {
            break;
        }
        if !seen.insert(Arc::as_ptr(&p)) {
            continue;
        }
        p.for_each_expr(&mut |e| {
            found |= has_outer_reference(e);
            stack.extend(subquery_plans(e));
        });
        p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    found
}

/// Whether every expression of `plan` (subquery plans included) is
/// deterministic.
pub(crate) fn deterministic(plan: &Arc<LogicalPlan>) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(plan)];
    let mut ok = true;
    while let Some(p) = stack.pop() {
        if !ok {
            break;
        }
        if !seen.insert(Arc::as_ptr(&p)) {
            continue;
        }
        p.for_each_expr(&mut |e| ok &= is_deterministic(e));
        p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    ok
}

struct Inliner {
    /// Every definition, by its output ids.
    defs: HashMap<Vec<AttrId>, Arc<LogicalPlan>>,
    /// The definitions to inline.
    inline: HashSet<Vec<AttrId>>,
    /// A definition with its own references inlined, built once.
    bodies: HashMap<Vec<AttrId>, Arc<LogicalPlan>>,
    /// The first attribute id no instance holds yet.
    next: u64,
}

impl Inliner {
    fn plan(&mut self, plan: Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, KrishivError> {
        let mut rewriter = dag::Rewriter::new();
        rewriter.up(plan, &mut |n| self.step(n))
    }

    fn step(&mut self, node: Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, KrishivError> {
        match &node.node {
            PlanNode::CteRef(r) if self.inline.contains(&r.source) => {
                Ok(self.instance(r)?.unwrap_or(node))
            }
            PlanNode::WithCte { input, ctes } => {
                // A definition stays while a reference to it does: one not
                // inlined, or an instance that would pass `MAX_ID`.
                let mut live = HashSet::new();
                collect_refs(input, &mut live);
                for c in ctes {
                    collect_refs(&c.plan, &mut live);
                }
                let kept: Vec<_> = ctes
                    .iter()
                    .filter(|c| {
                        c.plan.resolved_output().is_none_or(|out| {
                            live.contains(&out.iter().map(|o| o.id).collect::<Vec<_>>())
                        })
                    })
                    .cloned()
                    .collect();
                if kept.len() == ctes.len() {
                    Ok(node)
                } else if kept.is_empty() {
                    Ok(Arc::clone(input))
                } else {
                    Ok(Arc::new(LogicalPlan::new(PlanNode::WithCte {
                        input: Arc::clone(input),
                        ctes: kept,
                    })))
                }
            }
            _ => Ok(node),
        }
    }

    /// A fresh instance of `r`'s definition under `r`'s output ids; `None`
    /// when its ids would pass [`MAX_ID`].
    fn instance(&mut self, r: &CteRef) -> Result<Option<Arc<LogicalPlan>>, KrishivError> {
        let body = match self.bodies.get(&r.source) {
            Some(b) => Arc::clone(b),
            None => {
                let Some(def) = self.defs.get(&r.source).cloned() else {
                    return Ok(None);
                };
                let b = self.plan(def)?;
                self.bodies.insert(r.source.clone(), Arc::clone(&b));
                b
            }
        };
        let span = max_id(&body, &mut HashMap::new());
        let offset = self.next;
        let Some(top) = offset.checked_add(span).filter(|top| *top <= MAX_ID) else {
            return Ok(None);
        };
        self.next = top + 1;
        let exprs = r
            .source
            .iter()
            .zip(&r.output)
            .map(|(s, o)| {
                let column = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
                    BoundColumn {
                        id: AttrId(s.0 + offset),
                        ..o.clone()
                    },
                ))));
                Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                    expr: column,
                    output: o.clone(),
                })))
            })
            .collect();
        let shifted = Arc::new(LogicalPlan::new(PlanNode::Shift {
            input: body,
            offset,
        }));
        Ok(Some(Arc::new(LogicalPlan::new(PlanNode::Project {
            input: shifted,
            exprs,
        }))))
    }
}

/// The definitions the references under `plan` read (subquery plans
/// included).
fn collect_refs(plan: &Arc<LogicalPlan>, out: &mut HashSet<Vec<AttrId>>) {
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(plan)];
    while let Some(p) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&p)) {
            continue;
        }
        if let PlanNode::CteRef(r) = &p.node {
            out.insert(r.source.clone());
        }
        p.for_each_expr(&mut |e| stack.extend(subquery_plans(e)));
        p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
}
