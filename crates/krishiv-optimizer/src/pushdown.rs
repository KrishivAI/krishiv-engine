// SPDX-License-Identifier: Apache-2.0
//! Predicate pushdown: filters move below projections, aggregates (onto
//! their grouping keys), sorts, aliases, CTE bodies, self-join instances and
//! joins; inner-join filters become join conditions; join conditions on one
//! side move into that side; outer joins whose null rows a filter rejects
//! become inner (or one-sided) joins.
//!
//! Only deterministic predicates move, and only the deterministic prefix of
//! a conjunction (a nondeterministic predicate must see the rows it saw
//! before).

use std::collections::HashMap;
use std::sync::Arc;

use krishiv_plan::{
    AttrId, Builtin, Expr, ExprNode, Grouping, JoinCondition, JoinKind, Literal, LogicalPlan,
    PlanNode, ResolvedExpr,
};

use crate::dag::with_input;
use crate::expr::{
    Attrs, and_all, builtin, call, conjuncts, has_aggregate, has_outer_reference, has_subquery,
    implied, is_deterministic, lit, references, same, strip_alias, substitute, unshift,
};
use crate::fold::fold;
use krishiv_types::DataType;

/// The rule at a `Filter`: pushes its conjuncts down; drops true and
/// duplicate conjuncts.
pub(crate) fn push_filter(node: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    let PlanNode::Filter { input, predicate } = &node.node else {
        return node;
    };
    let original = conjuncts(predicate);
    let (below, rest) = push(input, conjuncts(&null_to_false(predicate)));
    let rest = dedupe(rest);
    let same_preds =
        rest.len() == original.len() && rest.iter().zip(&original).all(|(a, b)| Arc::ptr_eq(a, b));
    if Arc::ptr_eq(&below, input) && same_preds {
        return node;
    }
    wrap(below, rest, &node)
}

/// A predicate in a filter or join condition: a NULL operand of its
/// `and`/`or` tree acts as false (Spark's `ReplaceNullWithFalseInPredicate`),
/// so the tree folds further. `predicate` itself when there is none.
fn null_to_false(predicate: &Arc<Expr>) -> Arc<Expr> {
    let Some((op @ (Builtin::And | Builtin::Or), [a, b], _)) = builtin(predicate) else {
        return match &predicate.node {
            ExprNode::Literal(Literal::Null(_)) => lit(Literal::Boolean(false)),
            _ => Arc::clone(predicate),
        };
    };
    let (x, y) = (null_to_false(a), null_to_false(b));
    if Arc::ptr_eq(&x, a) && Arc::ptr_eq(&y, b) {
        return Arc::clone(predicate);
    }
    fold(&call(op, vec![x, y], DataType::Boolean))
}

/// The rule at a `Join`: moves one-sided condition conjuncts into the side
/// they read, where the join type allows.
pub(crate) fn push_join_condition(node: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    let PlanNode::Join {
        left,
        right,
        kind,
        condition: JoinCondition::On(cond),
    } = &node.node
    else {
        return node;
    };
    let (lo, ro) = (ids(left), ids(right));
    let (to_left, to_right) = match kind {
        JoinKind::Inner | JoinKind::Cross | JoinKind::LeftSemi => (true, true),
        JoinKind::LeftOuter | JoinKind::LeftAnti => (false, true),
        JoinKind::RightOuter => (true, false),
        _ => (false, false),
    };
    let simplified = null_to_false(cond);
    let (mut l, mut r, mut stay) = (Vec::new(), Vec::new(), Vec::new());
    for c in conjuncts(&simplified) {
        let refs = local(&c, &lo, &ro);
        if !is_deterministic(&c) {
            stay.push(c);
        } else if to_left && refs.is_subset(&lo) {
            l.push(c);
        } else if to_right && refs.is_subset(&ro) {
            r.push(c);
        } else {
            stay.push(c);
        }
    }
    if l.is_empty() && r.is_empty() && Arc::ptr_eq(&simplified, cond) {
        return node;
    }
    let (left, right) = (into(left, l), into(right, r));
    join(&node, left, right, *kind, stay)
}

/// The rule at an inner or cross `Join`, run once before the rewrite batch:
/// filters each input with what the condition's disjunctions imply on it
/// alone (`(a1 and b1) or (a2 and b2)` adds `a1 or a2` below the join, as
/// Spark's `PushExtraPredicateThroughJoin` does). A disjunction moved from a
/// `WHERE` into a join gets them as it moves (`push_join`); one written in
/// `ON` gets them here. Join ordering divides them out of the disjunction's
/// selectivity, so both kinds must have them. Once only: the disjunction
/// stays in the condition, so a fixed point would add them again.
pub(crate) fn imply_join_condition(node: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    let PlanNode::Join {
        left,
        right,
        kind: kind @ (JoinKind::Inner | JoinKind::Cross),
        condition: JoinCondition::On(cond),
    } = &node.node
    else {
        return node;
    };
    let (mut l, mut r) = (Vec::new(), Vec::new());
    for c in conjuncts(cond).iter().filter(|c| is_deterministic(c)) {
        l.extend(implied_by_input(c, left));
        r.extend(implied_by_input(c, right));
    }
    if l.is_empty() && r.is_empty() {
        return node;
    }
    let (left, right) = (into(left, l), into(right, r));
    join(&node, left, right, *kind, conjuncts(cond))
}

/// Pushes `preds` (over `plan`'s output) into `plan` as far as legal.
/// Returns the new plan and the predicates that stay above it, in order.
fn push(plan: &Arc<LogicalPlan>, preds: Vec<Arc<Expr>>) -> (Arc<LogicalPlan>, Vec<Arc<Expr>>) {
    let split = preds
        .iter()
        .position(|p| !is_deterministic(p))
        .unwrap_or(preds.len());
    let mut movable = preds;
    let fixed = movable.split_off(split);
    if movable.is_empty() {
        return (Arc::clone(plan), fixed);
    }
    let (out, mut stay) = match &plan.node {
        PlanNode::Filter { input, predicate } if is_deterministic(predicate) => {
            let mut all = conjuncts(predicate);
            all.extend(movable);
            let (below, rest) = push(input, all);
            (wrap(below, dedupe(rest), plan), Vec::new())
        }
        PlanNode::Project { input, exprs } if exprs.iter().all(|e| is_deterministic(e)) => {
            let aliases = alias_map(exprs);
            through(plan, input, movable, |p| {
                let refs = references(p);
                let hits = refs.iter().any(|id| aliases.contains_key(id));
                (!(hits && has_subquery(p))).then(|| substitute(p, &aliases))
            })
        }
        PlanNode::Aggregate {
            input,
            grouping: Grouping::GroupBy(keys),
            aggregates,
        } if !keys.is_empty() && aggregates.iter().all(|e| is_deterministic(e)) => {
            let (own, child) = (ids(plan), ids(input));
            let aliases: HashMap<AttrId, Arc<Expr>> = alias_map(aggregates)
                .into_iter()
                .filter(|(_, e)| !has_aggregate(e))
                .collect();
            through(plan, input, movable, |p| {
                let refs: Attrs = references(p)
                    .into_iter()
                    .filter(|id| own.contains(id))
                    .collect();
                let grouped = refs
                    .iter()
                    .all(|id| aliases.contains_key(id) || child.contains(id));
                let hits = refs.iter().any(|id| aliases.contains_key(id));
                (!refs.is_empty() && grouped && !(hits && has_subquery(p)))
                    .then(|| substitute(p, &aliases))
            })
        }
        PlanNode::Sort { input, .. }
        | PlanNode::SubqueryAlias { input, .. }
        | PlanNode::WithCte { input, .. }
        | PlanNode::JoinHint { input, .. } => {
            through(plan, input, movable, |p| Some(Arc::clone(p)))
        }
        PlanNode::Shift { input, offset } => through(plan, input, movable, |p| {
            // The subtree under a shift holds no outer references.
            if has_subquery(p) || has_outer_reference(p) {
                None
            } else {
                unshift(p, *offset)
            }
        }),
        PlanNode::Join {
            left,
            right,
            kind,
            condition,
        } => push_join(plan, left, right, *kind, condition, movable),
        _ => (Arc::clone(plan), movable),
    };
    stay.extend(fixed);
    (out, stay)
}

/// Moves each predicate `map` accepts below `plan` (onto `input`, rewritten
/// as `map` returns it); the others stay.
fn through(
    plan: &Arc<LogicalPlan>,
    input: &Arc<LogicalPlan>,
    preds: Vec<Arc<Expr>>,
    mut map: impl FnMut(&Arc<Expr>) -> Option<Arc<Expr>>,
) -> (Arc<LogicalPlan>, Vec<Arc<Expr>>) {
    let (mut down, mut stay) = (Vec::new(), Vec::new());
    for p in preds {
        match map(&p) {
            Some(q) => down.push(fold(&q)),
            None => stay.push(p),
        }
    }
    if down.is_empty() {
        return (Arc::clone(plan), stay);
    }
    (with_input(plan, into(input, down)), stay)
}

/// `input` with `preds` pushed into it, the rest filtering on top.
fn into(input: &Arc<LogicalPlan>, preds: Vec<Arc<Expr>>) -> Arc<LogicalPlan> {
    if preds.is_empty() {
        return Arc::clone(input);
    }
    let (below, rest) = push(input, preds);
    wrap(below, dedupe(rest), input)
}

fn push_join(
    plan: &Arc<LogicalPlan>,
    left: &Arc<LogicalPlan>,
    right: &Arc<LogicalPlan>,
    kind: JoinKind,
    condition: &JoinCondition,
    preds: Vec<Arc<Expr>>,
) -> (Arc<LogicalPlan>, Vec<Arc<Expr>>) {
    let (lo, ro) = (ids(left), ids(right));
    let original = kind;
    let kind = eliminate_outer(kind, &preds, left, right, &lo, &ro);
    let (mut l, mut r, mut on, mut stay) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for p in preds {
        let refs = local(&p, &lo, &ro);
        match kind {
            JoinKind::Inner | JoinKind::Cross => {
                if refs.is_subset(&lo) {
                    l.push(p);
                } else if refs.is_subset(&ro) {
                    r.push(p);
                } else if has_subquery(&p) {
                    // A join condition holds no subquery: it stays a filter.
                    stay.push(p);
                } else {
                    l.extend(implied_by_input(&p, left));
                    r.extend(implied_by_input(&p, right));
                    on.push(p);
                }
            }
            JoinKind::LeftOuter if refs.is_subset(&lo) => l.push(p),
            JoinKind::RightOuter if refs.is_subset(&ro) => r.push(p),
            JoinKind::LeftSemi | JoinKind::LeftAnti => l.push(p),
            _ => stay.push(p),
        }
    }
    if l.is_empty() && r.is_empty() && on.is_empty() && kind == original {
        return (Arc::clone(plan), stay);
    }
    let mut cond = match condition {
        JoinCondition::On(c) => conjuncts(c),
        _ => Vec::new(),
    };
    cond.extend(on);
    let out = join(plan, into(left, l), into(right, r), kind, dedupe(cond));
    (out, stay)
}

/// A join like `node` over new inputs and condition conjuncts; a cross join
/// with a condition becomes an inner join.
fn join(
    node: &Arc<LogicalPlan>,
    left: Arc<LogicalPlan>,
    right: Arc<LogicalPlan>,
    kind: JoinKind,
    cond: Vec<Arc<Expr>>,
) -> Arc<LogicalPlan> {
    let condition = and_all(cond).map_or(JoinCondition::None, JoinCondition::On);
    let kind = match (kind, &condition) {
        (JoinKind::Cross, JoinCondition::On(_)) => JoinKind::Inner,
        _ => kind,
    };
    Arc::new(LogicalPlan {
        node: PlanNode::Join {
            left,
            right,
            kind,
            condition,
        },
        origin: node.origin,
    })
}

/// The join type after a filter `preds` above it: an outer side whose
/// all-NULL rows some predicate rejects is no longer preserved.
fn eliminate_outer(
    kind: JoinKind,
    preds: &[Arc<Expr>],
    left: &LogicalPlan,
    right: &LogicalPlan,
    lo: &Attrs,
    ro: &Attrs,
) -> JoinKind {
    let rejects =
        |side: &LogicalPlan, ids: &Attrs| preds.iter().any(|p| rejects_nulls(p, side, ids));
    match kind {
        JoinKind::LeftOuter if rejects(right, ro) => JoinKind::Inner,
        JoinKind::RightOuter if rejects(left, lo) => JoinKind::Inner,
        JoinKind::FullOuter => match (rejects(left, lo), rejects(right, ro)) {
            (true, true) => JoinKind::Inner,
            (true, false) => JoinKind::LeftOuter,
            (false, true) => JoinKind::RightOuter,
            (false, false) => kind,
        },
        _ => kind,
    }
}

/// Whether `pred` is false or NULL whenever every column of `side` is
/// NULL: it then removes the rows an outer join pads with NULLs.
fn rejects_nulls(pred: &Arc<Expr>, side: &LogicalPlan, ids: &Attrs) -> bool {
    if has_subquery(pred) || !is_deterministic(pred) {
        return false;
    }
    let refs = references(pred);
    if refs.is_disjoint(ids) {
        return false;
    }
    let Some(out) = side.resolved_output() else {
        return false;
    };
    let nulls: HashMap<AttrId, Arc<Expr>> = out
        .into_iter()
        .filter(|c| refs.contains(&c.id))
        .map(|c| (c.id, lit(Literal::Null(Some(c.data_type)))))
        .collect();
    matches!(
        fold(&substitute(pred, &nulls)).node,
        ExprNode::Literal(Literal::Null(_) | Literal::Boolean(false))
    )
}

/// What `pred` implies on each input of the inner-join tree under `side`
/// ([`implied`]), one predicate per input that it constrains. Per input,
/// not per side: the inputs stay the same however join ordering regroups
/// the tree, so the filters derived before and after it agree (and
/// optimizing an optimized plan derives nothing new); join ordering
/// divides out exactly these per-input filters.
fn implied_by_input(pred: &Arc<Expr>, side: &Arc<LogicalPlan>) -> Vec<Arc<Expr>> {
    let mut out = Vec::new();
    let mut stack = vec![side];
    while let Some(n) = stack.pop() {
        match &n.node {
            PlanNode::Join {
                left,
                right,
                kind: JoinKind::Inner | JoinKind::Cross,
                condition,
            } if !matches!(condition, JoinCondition::On(c) if !is_deterministic(c)) => {
                stack.push(right);
                stack.push(left);
            }
            PlanNode::Filter { input, predicate } if is_deterministic(predicate) => {
                stack.push(input);
            }
            PlanNode::Project { input, exprs }
                if exprs
                    .iter()
                    .all(|e| matches!(e.node, ExprNode::Resolved(ResolvedExpr::Column(_)))) =>
            {
                stack.push(input);
            }
            _ => out.extend(implied(pred, &ids(n))),
        }
    }
    out
}

/// The output ids of a resolved relation (none for an unresolved one, so
/// nothing moves into it).
pub(crate) fn ids(plan: &LogicalPlan) -> Attrs {
    plan.resolved_output()
        .unwrap_or_default()
        .into_iter()
        .map(|c| c.id)
        .collect()
}

/// The references of `p` among the join's own attributes (outer references
/// to enclosing queries are constants here).
fn local(p: &Expr, lo: &Attrs, ro: &Attrs) -> Attrs {
    references(p)
        .into_iter()
        .filter(|id| lo.contains(id) || ro.contains(id))
        .collect()
}

/// Output id → the expression an alias computes.
fn alias_map(exprs: &[Arc<Expr>]) -> HashMap<AttrId, Arc<Expr>> {
    exprs
        .iter()
        .filter_map(|e| match &e.node {
            ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => {
                Some((output.id, Arc::clone(strip_alias(e))))
            }
            _ => None,
        })
        .collect()
}

/// Drops literal-true and repeated deterministic conjuncts.
fn dedupe(preds: Vec<Arc<Expr>>) -> Vec<Arc<Expr>> {
    let mut out: Vec<Arc<Expr>> = Vec::with_capacity(preds.len());
    for p in preds {
        if matches!(p.node, ExprNode::Literal(Literal::Boolean(true))) {
            continue;
        }
        if is_deterministic(&p) && out.iter().any(|q| same(q, &p)) {
            continue;
        }
        out.push(p);
    }
    out
}

/// `plan` filtered by `preds` (none: `plan` itself), with `like`'s origin.
fn wrap(plan: Arc<LogicalPlan>, preds: Vec<Arc<Expr>>, like: &LogicalPlan) -> Arc<LogicalPlan> {
    match and_all(preds) {
        None => plan,
        Some(predicate) => Arc::new(LogicalPlan {
            node: PlanNode::Filter {
                input: plan,
                predicate,
            },
            origin: like.origin,
        }),
    }
}
