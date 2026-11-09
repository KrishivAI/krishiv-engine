// SPDX-License-Identifier: Apache-2.0
//! Inferred join filters (design P2-T072b §2.3, unit 3): a subset of
//! Spark's `InferFiltersFromConstraints`.
//!
//! At each join, the classes of columns equal by the join condition and by
//! the constraints of its preserved side(s) carry those constraints to the
//! other members: `a = b AND b = 5` filters the side of `a` with `a = 5`.
//! Each nullable `=` key also gets `isnotnull` on the sides that may be
//! filtered (a null key never matches); as in Spark, a column that cannot
//! be null gets none, and no `isnotnull` moves onto one. File scans read
//! every column as nullable in both engines (`krishiv-io` `schema_map`, as
//! Spark's `asNullable`), so over files both add the same filters.
//!
//! - **Sides** follow Spark: inner, cross and left semi joins filter both
//!   inputs; a right outer join its left, a left outer or left anti join its
//!   right; a full outer, single or existence join none.
//! - **Sources** are the constraints of the preserved side(s) (both inputs
//!   of an inner join) and the condition. Classes are built per join, so a
//!   constraint never passes up through an outer or anti join's
//!   null-supplying or removed side.
//! - **What moves**: comparisons of a column with a literal, literal `IN`
//!   lists and `isnotnull`, onto a column of the same type joined by `=`
//!   (classes from [`crate::expr::equality`]). These cannot raise. Nothing
//!   is substituted into other expressions (Spark does, which can add a
//!   predicate that raises; the difference is listed in ADR 0023).
//! - A predicate a side already holds is not added again, so `optimize`
//!   stays a fixed point.
//!
//! Runs after the rewrite batch (which moves `WHERE` equalities into the
//! joins; the batch runs again to push what it added), then in each round
//! of join ordering and restriction
//! (`order_and_restrict`: the joins they form and restriction's semi joins),
//! and once after restriction when there is no join ordering. What a side
//! already implies is never added again ([`held`]), so a second `optimize`
//! adds nothing.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, Grouping, JoinCondition, JoinKind, Literal,
    LogicalPlan, PlanNode, ResolvedExpr,
};
use krishiv_types::DataType;

use crate::expr::{
    Attrs, Classes, builtin, call_builtin, column, conjuncts, equality, has_aggregate,
    has_subquery, is_deterministic, references, same, substitute, unshift,
};
use crate::pushdown::{alias_map, into_side_new, rebuild_join};

/// The rule at a `Join` (see the module docs).
pub(crate) fn infer_join_filters(node: Arc<LogicalPlan>, memo: &mut Memo) -> Arc<LogicalPlan> {
    let PlanNode::Join {
        left,
        right,
        kind,
        condition: JoinCondition::On(cond),
    } = &node.node
    else {
        return node;
    };
    if !is_deterministic(cond) {
        return node;
    }
    // (filter the left, filter the right, sources from the left, from the right)
    let (to_left, to_right, from_left, from_right) = match kind {
        JoinKind::Inner | JoinKind::Cross | JoinKind::LeftSemi => (true, true, true, true),
        JoinKind::RightOuter => (true, false, false, true),
        JoinKind::LeftOuter | JoinKind::LeftAnti => (false, true, true, false),
        _ => return node,
    };
    let cond = conjuncts(cond);
    // Everything moves across an `=` of the condition: without one there is
    // nothing to do (and no input needs walking).
    if !cond.iter().any(|c| equality(c).is_some()) {
        return node;
    }
    let mut sources: Vec<Arc<Expr>> = cond.clone();
    if from_left {
        sources.extend(memo.constraints(left).iter().cloned());
    }
    if from_right {
        sources.extend(memo.constraints(right).iter().cloned());
    }
    let mut classes = Classes::default();
    let mut members: HashMap<AttrId, Vec<AttrId>> = HashMap::new();
    for s in &sources {
        if let Some([a, b]) = equality(s) {
            classes.union([a.id, b.id]);
            for id in [a.id, b.id] {
                members.entry(id).or_default();
            }
        }
    }
    let ids_of: Vec<AttrId> = members.keys().copied().collect();
    let mut by_root: HashMap<AttrId, Vec<AttrId>> = HashMap::new();
    for id in ids_of {
        by_root.entry(classes.find(id)).or_default().push(id);
    }
    let (lo, ro) = (memo.ids(left), memo.ids(right));
    let (mut add_left, mut add_right) = (Vec::new(), Vec::new());
    let mut push = |target: AttrId, pred: Arc<Expr>| {
        if to_left && lo.contains(&target) && !holds(left, &add_left, &pred, memo) {
            add_left.push(pred);
        } else if to_right && ro.contains(&target) && !holds(right, &add_right, &pred, memo) {
            add_right.push(pred);
        }
    };
    // A null key never matches: `isnotnull` on each nullable `=` key of the
    // condition (Spark's `constructIsNotNullConstraints` skips non-nullable
    // attributes).
    // In column-id order, so the plan does not depend on the order the
    // condition was written in.
    let mut keys: Vec<&BoundColumn> = cond
        .iter()
        .filter_map(|c| equality(c))
        .flatten()
        .filter(|k| k.nullable)
        .collect();
    keys.sort_by_key(|k| k.id);
    keys.dedup_by_key(|k| k.id);
    // Each member's `isnotnull` is a candidate once per join: from its key,
    // or from one `isnotnull` source of its class (a join chain puts every
    // key in one class, and each of its `isnotnull` filters would otherwise
    // be moved onto every member).
    let mut nn_done: HashSet<AttrId> = HashSet::new();
    for k in keys {
        nn_done.insert(k.id);
        push(k.id, is_not_null_of(k));
    }
    // The column each class member's `=` writes (once, not per member).
    let mut cols: HashMap<AttrId, &BoundColumn> = HashMap::new();
    for s in &sources {
        if let Some([a, b]) = equality(s) {
            cols.entry(a.id).or_insert(a);
            cols.entry(b.id).or_insert(b);
        }
    }
    let mut nn_classes: HashSet<AttrId> = HashSet::new();
    // The class's literal constraints, moved onto its other members.
    for s in &sources {
        let Some(on) = literal_constraint(s) else {
            continue;
        };
        let root = classes.find(on);
        let Some(class) = by_root.get(&root) else {
            continue;
        };
        let nn = is_not_null(s);
        if nn && !nn_classes.insert(root) {
            continue;
        }
        for &m in class.iter().filter(|&&m| m != on) {
            let Some(to) = cols.get(&m).copied() else {
                continue;
            };
            if nn {
                if !to.nullable || !nn_done.insert(m) {
                    continue;
                }
                push(m, is_not_null_of(to));
                continue;
            }
            let map = HashMap::from([(on, column(to))]);
            push(m, substitute(s, &map));
        }
    }
    if add_left.is_empty() && add_right.is_empty() {
        return node;
    }
    // A side with nothing to add stays as it is (the push would merge
    // its stacked filters, a change that never ends join ordering's loop).
    let side = |s: &Arc<LogicalPlan>, add: Vec<Arc<Expr>>| {
        if add.is_empty() {
            Arc::clone(s)
        } else {
            into_side_new(s, add)
        }
    };
    let new_left = side(left, add_left);
    let new_right = side(right, add_right);
    if Arc::ptr_eq(&new_left, left) && Arc::ptr_eq(&new_right, right) {
        return node;
    }
    rebuild_join(&node, new_left, new_right, *kind, cond)
}

/// Whether `pred` is already queued in `added` or `side`'s rows already
/// satisfy it ([`held`]).
fn holds(side: &Arc<LogicalPlan>, added: &[Arc<Expr>], pred: &Arc<Expr>, memo: &mut Memo) -> bool {
    added.iter().any(|k| same(k, pred)) || held(side, pred, memo)
}

/// A node of [`held`]'s walk, the candidate and the conjuncts met above it,
/// spelled over the node's output.
type Step<'a> = (&'a Arc<LogicalPlan>, Arc<Expr>, Vec<Arc<Expr>>);

/// Whether the deterministic filters and join conditions of `side` already
/// imply `pred`. The walk follows `pred` down through projections (aliases
/// replaced by what they compute), shifts, grouped aggregates, positional
/// unions, joins and row-preserving nodes. Each conjunct is compared where
/// it is met, and a filter's once more under the projection right below it
/// (so a conjunct over a `cse_*` column compares equal to the expression it
/// names); nothing is carried further, which keeps the walk linear.
/// That is what keeps `optimize` a fixed point: later rules (pushdown,
/// common-subexpression elimination, join ordering) move and respell the
/// filters this rule added, and a second run must still find them.
fn held(side: &Arc<LogicalPlan>, pred: &Arc<Expr>, memo: &mut Memo) -> bool {
    held_under(side, Arc::clone(pred), Vec::new(), memo)
}

/// [`held`] of `pred` at `side`, below conjuncts `known` (spelled over
/// `side`'s output).
fn held_under(
    side: &Arc<LogicalPlan>,
    pred: Arc<Expr>,
    known: Vec<Arc<Expr>>,
    memo: &mut Memo,
) -> bool {
    let mut stack: Vec<Step<'_>> = vec![(side, pred, known)];
    while let Some((n, p, mut known)) = stack.pop() {
        if implies(&known, &p) {
            return true;
        }
        match &n.node {
            PlanNode::Filter { input, predicate } if is_deterministic(predicate) => {
                known.extend(conjuncts(predicate));
                stack.push((input, p, known));
            }
            PlanNode::Project { input, exprs } if exprs.iter().all(|e| is_deterministic(e)) => {
                let defs = alias_map(exprs);
                let p = substitute(&p, &defs);
                // The filters right above, spelled over this input too (a
                // `cse_*` filter over its own projection); then dropped, so
                // nothing is carried through a whole subtree.
                let known: Vec<_> = known.iter().map(|k| substitute(k, &defs)).collect();
                if implies(&known, &p) {
                    return true;
                }
                stack.push((input, p, Vec::new()));
            }
            // A grouped aggregate passes its key columns through: what
            // holds of them below holds of their outputs (pushdown moves
            // filters on them below it).
            PlanNode::Aggregate {
                input,
                grouping: Grouping::GroupBy(keys),
                aggregates,
            } if !keys.is_empty() && aggregates.iter().all(|e| is_deterministic(e)) => {
                let (defs, calls): (HashMap<AttrId, Arc<Expr>>, HashMap<AttrId, Arc<Expr>>) =
                    alias_map(aggregates)
                        .into_iter()
                        .partition(|(_, e)| !has_aggregate(e));
                // Not over an aggregate call's result.
                let passes = |e: &Arc<Expr>| references(e).iter().all(|id| !calls.contains_key(id));
                if passes(&p) {
                    stack.push((input, substitute(&p, &defs), Vec::new()));
                }
            }
            // A positional union holds `p` when every input does, its
            // columns read by position (pushdown moves filters into each).
            PlanNode::Union {
                inputs,
                by_name: false,
                ..
            } => {
                let (Some(out), Some(outs)) = (
                    n.resolved_output(),
                    inputs
                        .iter()
                        .map(|i| i.resolved_output())
                        .collect::<Option<Vec<_>>>(),
                ) else {
                    continue;
                };
                let all = outs.len() == inputs.len()
                    && inputs.iter().zip(&outs).all(|(input, cols)| {
                        if cols.len() != out.len() {
                            return false;
                        }
                        let map: HashMap<AttrId, Arc<Expr>> = out
                            .iter()
                            .zip(cols)
                            .map(|(o, c)| (o.id, column(c)))
                            .collect();
                        held_under(input, substitute(&p, &map), Vec::new(), memo)
                    });
                if all {
                    return true;
                }
            }
            PlanNode::Shift { input, offset } => {
                if let Some(p) = unshift(&p, *offset) {
                    stack.push((input, p, Vec::new()));
                }
            }
            PlanNode::Join {
                left,
                right,
                kind,
                condition,
            } => {
                let refs = references(&p);
                // Which inputs' rows reach the output unextended, and
                // whether the condition holds of every output row.
                let (to_left, to_right, cond_holds) = match kind {
                    JoinKind::Inner | JoinKind::Cross => (true, true, true),
                    JoinKind::LeftSemi => (true, false, true),
                    JoinKind::LeftOuter | JoinKind::LeftAnti | JoinKind::Single => {
                        (true, false, false)
                    }
                    JoinKind::Existence(id) => (!refs.contains(id), false, false),
                    JoinKind::RightOuter => (false, true, false),
                    _ => (false, false, false),
                };
                if cond_holds && let JoinCondition::On(c) = condition {
                    if !is_deterministic(c) {
                        continue;
                    }
                    if implies(&conjuncts(c), &p) {
                        return true;
                    }
                }
                // Only the open input that has `p`'s columns (one path down,
                // ids memoised per pass).
                for (input, open) in [(left, to_left), (right, to_right)] {
                    if open && refs.is_subset(&memo.ids(input)) {
                        stack.push((input, Arc::clone(&p), Vec::new()));
                    }
                }
            }
            // Row-preserving nodes: what holds of the input columns below
            // holds above.
            PlanNode::Sort { input, .. }
            | PlanNode::SubqueryAlias { input, .. }
            | PlanNode::WithCte { input, .. }
            | PlanNode::JoinHint { input, .. }
            | PlanNode::Window { input, .. } => stack.push((input, p, known)),
            _ => {}
        }
    }
    false
}

/// Whether one of `known` is `pred` or a literal `false`, or, for `pred = isnotnull(e)`, rejects
/// a null `e`: a comparison with `e` on either side, `e IN (..)` or
/// `isnotnull(e)`.
fn implies(known: &[Arc<Expr>], pred: &Arc<Expr>) -> bool {
    // A filter folded to `false` (or NULL) keeps no rows, so it implies
    // everything; folding would drop anything added beside it again.
    if known.iter().any(|k| {
        matches!(
            k.node,
            ExprNode::Literal(Literal::Boolean(false) | Literal::Null(_))
        ) || same(k, pred)
    }) {
        return true;
    }
    let Some((Builtin::IsNotNull, [e], _)) = builtin(pred) else {
        return false;
    };
    known.iter().any(|k| match builtin(k) {
        Some((
            Builtin::Eq | Builtin::Lt | Builtin::LtEq | Builtin::Gt | Builtin::GtEq,
            [a, b],
            _,
        )) => same(a, e) || same(b, e),
        Some((Builtin::In | Builtin::IsNotNull, [a, ..], _)) => same(a, e),
        _ => false,
    })
}

/// `isnotnull(c)` as the analyzer resolves it: never null, whatever `c` is
/// (a call built with `nullable` from its argument would differ from the
/// user's own `c IS NOT NULL`, and the plan would not be a fixed point).
fn is_not_null_of(c: &BoundColumn) -> Arc<Expr> {
    let mut f = call_builtin(Builtin::IsNotNull, vec![column(c)], DataType::Boolean);
    f.nullable = false;
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f))))
}

/// Whether `e` is `isnotnull(..)`.
fn is_not_null(e: &Expr) -> bool {
    matches!(builtin(e), Some((Builtin::IsNotNull, _, _)))
}

/// The attribute a constraint that may move is about: `col <op> literal`
/// (either order) for `=`, `<`, `<=`, `>`, `>=`, `col IN (literals)` and
/// `isnotnull(col)`. Anything else, or anything with a subquery, stays.
fn literal_constraint(e: &Expr) -> Option<AttrId> {
    // The shape first: most sources are neither (the walks below are not
    // free). Sources are deterministic already (the condition is checked,
    // `Memo::constraints` keeps deterministic conjuncts only).
    let (op, args, _) = builtin(e)?;
    if !matches!(
        op,
        Builtin::IsNotNull
            | Builtin::Eq
            | Builtin::Lt
            | Builtin::LtEq
            | Builtin::Gt
            | Builtin::GtEq
            | Builtin::In
    ) || has_subquery(e)
    {
        return None;
    }
    let col = |x: &Expr| match &x.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
        _ => None,
    };
    let lit = |x: &Expr| matches!(x.node, ExprNode::Literal(_));
    match (op, args) {
        (Builtin::IsNotNull, [a]) => col(a),
        (Builtin::Eq | Builtin::Lt | Builtin::LtEq | Builtin::Gt | Builtin::GtEq, [a, b]) => {
            match (col(a), col(b)) {
                (Some(id), None) if lit(b) => Some(id),
                (None, Some(id)) if lit(a) => Some(id),
                _ => None,
            }
        }
        (Builtin::In, [a, rest @ ..]) if !rest.is_empty() && rest.iter().all(|x| lit(x)) => col(a),
        _ => None,
    }
}

/// A node's constraints, shared between the memo and its callers.
type Conjuncts = Arc<[Arc<Expr>]>;

/// What one pass of the rule (`lib.rs` `infer_join_filters`) learns about
/// relations, memoised by address: each node's constraints and output ids
/// are computed once from its inputs', not by a walk per join. Every key's
/// `Arc` is kept in the entry, so an address is never reused while the memo
/// lives (the plan crate's rule for address-keyed memos).
#[derive(Default)]
pub(crate) struct Memo {
    constraints: HashMap<*const LogicalPlan, (Arc<LogicalPlan>, Conjuncts)>,
    ids: HashMap<*const LogicalPlan, (Arc<LogicalPlan>, bool, Arc<Attrs>)>,
}

impl Memo {
    /// The deterministic conjuncts known to hold of `plan`'s output, found
    /// through filters, inner joins, column-only projections and shifts (the
    /// walk of the implied filters of `pushdown`, plus self-join
    /// instances), in pre-order: a node's own, then its inputs' left first.
    fn constraints(&mut self, plan: &Arc<LogicalPlan>) -> Conjuncts {
        if let Some((_, c)) = self.constraints.get(&Arc::as_ptr(plan)) {
            return Arc::clone(c);
        }
        let deterministic =
            |e: &Arc<Expr>| conjuncts(e).into_iter().filter(|c| is_deterministic(c));
        let out: Vec<Arc<Expr>> = match &plan.node {
            PlanNode::Filter { input, predicate } => {
                let mut out: Vec<_> = deterministic(predicate).collect();
                out.extend(self.constraints(input).iter().cloned());
                out
            }
            PlanNode::Join {
                left,
                right,
                kind: JoinKind::Inner | JoinKind::Cross,
                condition,
            } => {
                let mut out: Vec<_> = match condition {
                    JoinCondition::On(c) => deterministic(c).collect(),
                    _ => Vec::new(),
                };
                out.extend(self.constraints(left).iter().cloned());
                out.extend(self.constraints(right).iter().cloned());
                out
            }
            PlanNode::Project { input, exprs }
                if exprs
                    .iter()
                    .all(|e| matches!(e.node, ExprNode::Resolved(ResolvedExpr::Column(_)))) =>
            {
                self.constraints(input).to_vec()
            }
            // A self-join instance: its constraints, renumbered as it reads.
            PlanNode::Shift { input, offset } => self
                .constraints(input)
                .iter()
                .filter_map(|c| shift(c, *offset))
                .collect(),
            _ => Vec::new(),
        };
        let out: Conjuncts = out.into();
        self.constraints
            .insert(Arc::as_ptr(plan), (Arc::clone(plan), Arc::clone(&out)));
        out
    }

    /// `plan`'s output ids (`pushdown::ids`, once per node).
    fn ids(&mut self, plan: &Arc<LogicalPlan>) -> Arc<Attrs> {
        Arc::clone(&self.ids_of(plan).1)
    }

    /// The output ids of `plan` and whether it is resolved, built from its
    /// inputs' for the nodes that pass them on (a join's output would
    /// otherwise be recomputed, columns cloned, from its whole subtree at
    /// every join of a chain). An unresolved relation has none, as in
    /// `pushdown::ids`.
    fn ids_of(&mut self, plan: &Arc<LogicalPlan>) -> (bool, Arc<Attrs>) {
        if let Some((_, resolved, a)) = self.ids.get(&Arc::as_ptr(plan)) {
            return (*resolved, Arc::clone(a));
        }
        let (resolved, set) = match &plan.node {
            PlanNode::Filter { input, .. }
            | PlanNode::Sort { input, .. }
            | PlanNode::Limit { input, .. }
            | PlanNode::Offset { input, .. }
            | PlanNode::WithCte { input, .. }
            | PlanNode::JoinHint { input, .. }
            | PlanNode::Repartition { input, .. }
            | PlanNode::ReplicateRows { input, .. }
            | PlanNode::SubqueryAlias { input, .. }
                if !matches!(
                    &plan.node,
                    PlanNode::SubqueryAlias { columns, .. } if !columns.is_empty()
                ) =>
            {
                self.ids_of(input)
            }
            PlanNode::Join {
                left, right, kind, ..
            } => {
                let (lr, l) = self.ids_of(left);
                let (rr, r) = self.ids_of(right);
                if !(lr && rr) {
                    (false, Arc::new(Attrs::default()))
                } else {
                    match kind {
                        JoinKind::LeftSemi | JoinKind::LeftAnti => (true, l),
                        JoinKind::Existence(id) => {
                            let mut all = (*l).clone();
                            all.insert(JoinKind::existence_column(*id).id);
                            (true, Arc::new(all))
                        }
                        _ => {
                            let mut all = (*l).clone();
                            all.extend(r.iter().copied());
                            (true, Arc::new(all))
                        }
                    }
                }
            }
            _ => match plan.resolved_output() {
                Some(out) => (true, Arc::new(out.into_iter().map(|c| c.id).collect())),
                None => (false, Arc::new(Attrs::default())),
            },
        };
        self.ids.insert(
            Arc::as_ptr(plan),
            (Arc::clone(plan), resolved, Arc::clone(&set)),
        );
        (resolved, set)
    }
}

/// `e` with every column id raised by `offset` (what a [`PlanNode::Shift`]
/// does to its subtree's output); `None` on overflow.
fn shift(e: &Arc<Expr>, offset: u64) -> Option<Arc<Expr>> {
    let out: Result<_, ()> = Arc::clone(e).transform_up(&mut |x| match &x.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => {
            let id = AttrId(c.id.0.checked_add(offset).ok_or(())?);
            Ok(column(&BoundColumn { id, ..c.clone() }))
        }
        _ => Ok(x),
    });
    out.ok()
}
