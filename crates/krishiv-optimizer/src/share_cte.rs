// SPDX-License-Identifier: Apache-2.0
//! Sharing identical deterministic CTE instances (P2-T198, design §3.1 kind 2).
//!
//! InlineCTE leaves every reference of a deterministic definition as a fresh
//! instance (`Shift { body, offset }`), so the rules can see through it. When
//! two or more instances of one body survive optimization equal, each would
//! compute the same rows again; this rule, run once after the final prune and
//! tidy, turns them into references to one `WITH` definition that the
//! physical planner computes once per query (`SharedRead`).
//!
//! A group is shared when
//! - it has two or more members (instances, subquery plans included) whose
//!   bodies are equal: bucketed by the normalized subtree hash, decided by
//!   `Arc::ptr_eq` or plan equality (bodies are written in their source's own
//!   ids, so equal bodies compute equal rows);
//! - the body is deterministic, holds no outer reference, and holds a pipeline
//!   breaker (aggregate, join, window, sort, distinct, set operation): a scan,
//!   filter and projection chain is cheaper to rescan than to buffer, and it
//!   keeps its runtime filters.
//!
//! The rewrite replaces each member `Shift` alone by a `CteRef` that
//! produces exactly the ids the shift produced, so no parent changes, and
//! wraps the root in one `WithCte` holding the definitions. Members nested
//! inside another group's body are found too, and shared inside that body's
//! definition. Whether a group is really shared is decided last by the
//! physical planner, which keeps the copies when a runtime filter reaches
//! one of them.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{
    AttrId, BoundColumn, Cte, CteRef, Expr, ExprNode, LogicalPlan, PlanNode,
    normalized_subtree_hash,
};

use crate::expr::has_outer_reference;
use crate::inline_cte::deterministic;
use crate::restrict::subquery_plans;

/// The name prefix of the definitions this rule adds.
pub(crate) const SHARED_PREFIX: &str = "__shared_";

/// One instance: the `Shift` node, its body and offset.
struct Member {
    shift: Arc<LogicalPlan>,
    body: Arc<LogicalPlan>,
    offset: u64,
    /// How many paths from the root reach it: a `Shift` shared through
    /// `Arc` (both sides of a `UNION ALL` of one CTE) is read that many times.
    uses: u64,
}

/// `root` with its groups of equal instances shared.
pub(crate) fn share(root: Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, KrishivError> {
    let members = shifts(&root);
    if members.iter().map(|m| m.uses).sum::<u64>() < 2 {
        return Ok(root);
    }
    let groups = groups(&members);
    if groups.is_empty() {
        return Ok(root);
    }
    let mut replace: HashMap<*const LogicalPlan, Arc<LogicalPlan>> = HashMap::new();
    let mut bodies: Vec<(String, Arc<LogicalPlan>)> = Vec::new();
    let mut keys: HashSet<Vec<AttrId>> = HashSet::new();
    for group in groups {
        let first = &members[group[0]];
        let Some(output) = first.body.resolved_output() else {
            continue;
        };
        let source: Vec<AttrId> = output.iter().map(|c| c.id).collect();
        if !keys.insert(source.clone()) {
            continue;
        }
        let name = format!("{SHARED_PREFIX}{}", bodies.len());
        bodies.push((name.clone(), Arc::clone(&first.body)));
        for &i in &group {
            let m = &members[i];
            let shifted: Option<Vec<BoundColumn>> = output
                .iter()
                .map(|c| {
                    c.id.0.checked_add(m.offset).map(|id| BoundColumn {
                        id: AttrId(id),
                        ..c.clone()
                    })
                })
                .collect();
            let Some(shifted) = shifted else {
                continue;
            };
            replace.insert(
                Arc::as_ptr(&m.shift),
                Arc::new(LogicalPlan::new(PlanNode::CteRef(CteRef {
                    name: name.clone(),
                    source: source.clone(),
                    output: shifted,
                }))),
            );
        }
    }
    if replace.is_empty() {
        return Ok(root);
    }
    let mut sharer = Sharer {
        replace: &replace,
        done: HashMap::new(),
    };
    let input = sharer.go(Arc::clone(&root))?;
    // Each definition is its body with the groups nested inside it shared too.
    let mut ctes = Vec::with_capacity(bodies.len());
    for (name, body) in bodies {
        ctes.push(Cte {
            name,
            plan: sharer.go(body)?,
        });
    }
    // A group none of whose members was replaced (an id overflow) leaves an
    // unused definition; the planner never plans it.
    Ok(Arc::new(LogicalPlan::new(PlanNode::WithCte {
        input,
        ctes,
    })))
}

/// Every `Shift` of the plan (subquery plans included), once each, with the
/// number of paths that reach it.
fn shifts(root: &Arc<LogicalPlan>) -> Vec<Member> {
    // The relations in post-order (children before parents), each with its
    // children, then path counts from the root down.
    let mut order: Vec<Arc<LogicalPlan>> = Vec::new();
    let mut kids: HashMap<*const LogicalPlan, Vec<Arc<LogicalPlan>>> = HashMap::new();
    let mut seen: HashSet<*const LogicalPlan> = HashSet::new();
    let mut stack: Vec<(Arc<LogicalPlan>, bool)> = vec![(Arc::clone(root), false)];
    while let Some((p, expanded)) = stack.pop() {
        if expanded {
            order.push(p);
            continue;
        }
        if !seen.insert(Arc::as_ptr(&p)) {
            continue;
        }
        let mut children = Vec::new();
        p.for_each_expr(&mut |e| children.extend(subquery_plans(e)));
        p.for_each_child(&mut |c| children.push(Arc::clone(c)));
        stack.push((Arc::clone(&p), true));
        for c in &children {
            stack.push((Arc::clone(c), false));
        }
        kids.insert(Arc::as_ptr(&p), children);
    }
    let mut paths: HashMap<*const LogicalPlan, u64> = HashMap::new();
    paths.insert(Arc::as_ptr(root), 1);
    let mut out = Vec::new();
    for p in order.iter().rev() {
        let n = paths.get(&Arc::as_ptr(p)).copied().unwrap_or(0);
        if let PlanNode::Shift { input, offset } = &p.node {
            out.push(Member {
                shift: Arc::clone(p),
                body: Arc::clone(input),
                offset: *offset,
                uses: n,
            });
            // Instances inside the body are members too (nested groups); a
            // group shared from inside a shared body is rewritten there.
        }
        for c in kids.get(&Arc::as_ptr(p)).into_iter().flatten() {
            let slot = paths.entry(Arc::as_ptr(c)).or_insert(0);
            *slot = slot.saturating_add(n);
        }
    }
    out
}

/// The groups worth sharing, as indices into `members`.
fn groups(members: &[Member]) -> Vec<Vec<usize>> {
    let mut buckets: HashMap<u128, Vec<Vec<usize>>> = HashMap::new();
    for (i, m) in members.iter().enumerate() {
        let groups = buckets.entry(normalized_subtree_hash(&m.body)).or_default();
        match groups
            .iter_mut()
            .find(|g| equal(&members[g[0]].body, &m.body))
        {
            Some(g) => g.push(i),
            None => groups.push(vec![i]),
        }
    }
    let mut out: Vec<Vec<usize>> = buckets
        .into_values()
        .flatten()
        .filter(|g| {
            g.iter().map(|&i| members[i].uses).sum::<u64>() >= 2
                && worth_sharing(&members[g[0]].body)
        })
        .collect();
    // A stable order: by first member.
    out.sort_by_key(|g| g[0]);
    out
}

fn equal(a: &Arc<LogicalPlan>, b: &Arc<LogicalPlan>) -> bool {
    Arc::ptr_eq(a, b) || **a == **b
}

/// Whether `body` is deterministic, holds no outer reference and a pipeline
/// breaker.
fn worth_sharing(body: &Arc<LogicalPlan>) -> bool {
    deterministic(body) && !has_outer(body) && has_breaker(body)
}

fn has_outer(plan: &Arc<LogicalPlan>) -> bool {
    let mut found = false;
    walk(plan, &mut |p| {
        p.for_each_expr(&mut |e| found |= has_outer_reference(e));
    });
    found
}

fn has_breaker(plan: &Arc<LogicalPlan>) -> bool {
    let mut found = false;
    walk(plan, &mut |p| {
        found |= matches!(
            p.node,
            PlanNode::Aggregate { .. }
                | PlanNode::PartialAggregate { .. }
                | PlanNode::Join { .. }
                | PlanNode::Window { .. }
                | PlanNode::Sort { .. }
                | PlanNode::Distinct { .. }
                | PlanNode::SetOp { .. }
        );
    });
    found
}

/// Calls `f` on every relation under `plan` (subquery plans included), once
/// each.
fn walk(plan: &Arc<LogicalPlan>, f: &mut impl FnMut(&Arc<LogicalPlan>)) {
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(plan)];
    while let Some(p) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&p)) {
            continue;
        }
        f(&p);
        p.for_each_expr(&mut |e| stack.extend(subquery_plans(e)));
        p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
}

/// A top-down rewrite that replaces the members and rebuilds what is above.
struct Sharer<'a> {
    replace: &'a HashMap<*const LogicalPlan, Arc<LogicalPlan>>,
    /// Original → rewritten; the original is kept alive so its address stays
    /// unique.
    done: HashMap<*const LogicalPlan, (Arc<LogicalPlan>, Arc<LogicalPlan>)>,
}

impl Sharer<'_> {
    fn go(&mut self, plan: Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, KrishivError> {
        let key = Arc::as_ptr(&plan);
        if let Some(r) = self.replace.get(&key) {
            return Ok(Arc::clone(r));
        }
        if let Some((_, out)) = self.done.get(&key) {
            return Ok(Arc::clone(out));
        }
        let rebuilt = Arc::clone(&plan)
            .map_children(|c| self.go(c))?
            .map_expressions(|e| self.subqueries(e))?;
        self.done.insert(key, (plan, Arc::clone(&rebuilt)));
        Ok(rebuilt)
    }

    fn subqueries(&mut self, e: Arc<Expr>) -> Result<Arc<Expr>, KrishivError> {
        if !crate::expr::has_subquery(&e) {
            return Ok(e);
        }
        e.transform_up(&mut |x| match &x.node {
            ExprNode::Subquery { kind, plan } => {
                let rewritten = self.go(Arc::clone(plan))?;
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
