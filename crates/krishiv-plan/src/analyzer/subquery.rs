// SPDX-License-Identifier: Apache-2.0
//! Subqueries to joins, after Resolve (P2-T071c), in the shapes Spark's
//! `RewritePredicateSubquery` and `RewriteCorrelatedScalarSubquery` plan:
//!
//! - A conjunct `EXISTS (q)` / `NOT EXISTS (q)` of a filter, `q`
//!   correlated: a left semi / anti join of the filter's input with `q`,
//!   on `q`'s correlated predicates.
//! - A conjunct `v IN (q)`: a left semi join on `v = q's columns` and `q`'s
//!   correlated predicates; `v NOT IN (q)`: a left anti join on
//!   `(v = c) OR isnull(v = c)` (NULL-aware), as Spark plans it.
//! - A correlated scalar subquery `(SELECT agg ... WHERE inner = outer)` in
//!   a filter or a projection: a left outer join with `q`'s aggregate,
//!   grouped by its correlated keys; the subquery reads the aggregate's
//!   value (`count` reads 0 where no row matched).
//!
//! Correlated predicates move up from the filters at the top of `q`,
//! through projections and aliases. Uncorrelated scalar subqueries and
//! `EXISTS` stay expressions, computed once. Every other correlated shape
//! fails with `KRISHIV_UNSUPPORTED_FEATURE`, or Spark's class where Spark
//! rejects it too, never with a wrong result.
//!
//! The rewritten joins keep the analyzer's invariants: a join's sides never
//! share an attribute id (a right side that would is a [`PlanNode::Shift`]),
//! and re-analysing the result changes nothing (no correlated subquery is
//! left; uncorrelated ones are untouched).

use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;
use std::sync::Arc;

use krishiv_common::error::{
    FailedToParseTooComplex, UnsupportedSubqueryExpressionCategoryCorrelatedReference,
};
use krishiv_types::DataType;

use super::AnalysisError;
use super::bind::{addr, unsupported};
use super::exprs::{literal_type, pretty};
use super::ops::{MAX_ID, col_expr};
use crate::walk::{Node, visit_distinct};
use crate::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, FunctionKind, Grouping,
    JoinCondition, JoinKind, Literal, LogicalPlan, Origin, PlanNode, ResolvedExpr,
    ResolvedFunction, SortDirection, SubqueryKind,
};
use krishiv_common::stack::{STEP, on_big_stack};

type R<T> = Result<T, AnalysisError>;

/// A subquery's input with its correlation lifted out: the plan, the keys it
/// groups by, and the outer expressions they match.
type Lifted = (Arc<LogicalPlan>, Vec<Arc<Expr>>, Vec<Arc<Expr>>);

/// Rewrites the subqueries of `root`; `next_id` is one past every
/// attribute id in use and is kept so.
pub(super) fn rewrite(root: &Arc<LogicalPlan>, next_id: &mut u64) -> R<Arc<LogicalPlan>> {
    if !has_subquery(Node::Plan(root)) {
        return Ok(Arc::clone(root));
    }
    let mut r = Rewriter {
        next_id: *next_id,
        done: HashMap::new(),
        correlated: HashMap::new(),
        depth: 0,
    };
    let out = r.plan(root)?;
    *next_id = r.next_id;
    Ok(out)
}

struct Rewriter {
    next_id: u64,
    /// Each relation rewritten, by address (the original kept alive), so a
    /// relation shared by several parents or subqueries is rewritten once.
    done: HashMap<usize, (Arc<LogicalPlan>, Arc<LogicalPlan>)>,
    /// Whether a subquery plan refers to its enclosing query, by address.
    correlated: HashMap<usize, (Arc<LogicalPlan>, bool)>,
    depth: usize,
}

/// A predicate subquery conjunct, as a join.
struct Pred {
    kind: JoinKind,
    /// `NOT IN`: NULL-aware equalities.
    null_aware: bool,
    values: Vec<Arc<Expr>>,
    plan: Arc<LogicalPlan>,
}

/// How a join's right side was renumbered (see [`Rewriter::disjoint`]).
#[derive(Clone, Copy)]
struct Renumber {
    lo: u64,
    hi: u64,
    offset: u64,
}

impl Renumber {
    const NONE: Self = Self {
        lo: 1,
        hi: 0,
        offset: 0,
    };

    fn id(self, id: AttrId) -> AttrId {
        if (self.lo..=self.hi).contains(&id.0) {
            AttrId(id.0 + self.offset)
        } else {
            id
        }
    }

    /// `e`, a predicate of a branch's attributes and of outer references,
    /// reading the outer references as the renumbered domain columns.
    fn outer(self, e: &Arc<Expr>) -> Arc<Expr> {
        let out = Arc::clone(e).transform_up(&mut |x: Arc<Expr>| {
            Ok::<_, std::convert::Infallible>(match &x.node {
                ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) => {
                    let id = self.id(c.id);
                    col_expr(&BoundColumn { id, ..c.clone() }, x.origin)
                }
                _ => x,
            })
        });
        out.unwrap_or_else(|never| match never {})
    }

    /// `e`, an expression of the right side's attributes and of the outer
    /// query's (the left side): right ones renumbered, outer references
    /// read as plain columns of the left side.
    fn inner(self, e: &Arc<Expr>) -> Arc<Expr> {
        let out = Arc::clone(e).transform_up(&mut |x: Arc<Expr>| {
            Ok::<_, std::convert::Infallible>(match &x.node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) => {
                    let id = self.id(c.id);
                    if id == c.id {
                        x
                    } else {
                        col_expr(&BoundColumn { id, ..c.clone() }, x.origin)
                    }
                }
                ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) => col_expr(c, x.origin),
                _ => x,
            })
        });
        out.unwrap_or_else(|never| match never {})
    }
}

/// Whether `e` (subquery plans included) refers to an enclosing query.
fn has_outer(node: Node<'_>) -> bool {
    visit_distinct(node, |n| match n {
        Node::Expr(e) if matches!(e.node, ExprNode::Resolved(ResolvedExpr::OuterColumn(_))) => {
            ControlFlow::Break(())
        }
        _ => ControlFlow::Continue(()),
    })
    .is_break()
}

fn has_subquery(node: Node<'_>) -> bool {
    visit_distinct(node, |n| match n {
        Node::Expr(e) if matches!(e.node, ExprNode::Subquery { .. }) => ControlFlow::Break(()),
        _ => ControlFlow::Continue(()),
    })
    .is_break()
}

/// The subquery expressions directly in `e` (not inside subquery plans),
/// each once.
fn subqueries(e: &Arc<Expr>) -> Vec<Arc<Expr>> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        if !seen.insert(addr(&*x)) {
            continue;
        }
        if matches!(x.node, ExprNode::Subquery { .. }) {
            out.push(Arc::clone(&x));
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    out
}

/// `e` without the origins of its nodes, so equal expressions written in
/// two places compare equal.
fn without_origins(e: &Arc<Expr>) -> Arc<Expr> {
    let out = Arc::clone(e).transform_up(&mut |x: Arc<Expr>| {
        Ok::<_, std::convert::Infallible>(Arc::new(Expr::new(x.node.clone())))
    });
    out.unwrap_or_else(|never| match never {})
}

/// The disjuncts of a predicate.
fn disjuncts(e: &Arc<Expr>) -> Vec<Arc<Expr>> {
    let mut out = Vec::new();
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        match &x.node {
            ExprNode::Resolved(ResolvedExpr::Function(f))
                if f.id == FunctionId::Builtin(Builtin::Or) =>
            {
                stack.extend(f.args.iter().rev().cloned());
            }
            _ => out.push(x),
        }
    }
    out
}

/// `e` with the conjuncts every disjunct of a top-level `OR` holds taken
/// out: `(a AND b) OR (a AND c)` is `a AND (b OR c)`, which is three-valued
/// exact. `e` itself when it is no disjunction or the disjuncts share no
/// conjunct. A nondeterministic predicate is left whole: factoring moves
/// part of it below the join, where it would run once per inner row instead
/// of once per joined pair.
fn factor_or(e: &Arc<Expr>) -> Arc<Expr> {
    let parts = disjuncts(e);
    if parts.len() < 2 || !super::sqlfn::is_deterministic(e) {
        return Arc::clone(e);
    }
    let rows: Vec<Vec<(Arc<Expr>, Arc<Expr>)>> = parts
        .iter()
        .map(|d| {
            conjuncts(d)
                .into_iter()
                .map(|c| (without_origins(&c), c))
                .collect()
        })
        .collect();
    let common: Vec<Arc<Expr>> = rows[0]
        .iter()
        .filter(|(key, _)| rows[1..].iter().all(|r| r.iter().any(|(k, _)| k == key)))
        .map(|(_, c)| Arc::clone(c))
        .collect();
    if common.is_empty() {
        return Arc::clone(e);
    }
    let keys: Vec<Arc<Expr>> = common.iter().map(without_origins).collect();
    let mut rests = Vec::new();
    for row in &rows {
        let rest: Vec<Arc<Expr>> = row
            .iter()
            .filter(|(k, _)| !keys.contains(k))
            .map(|(_, c)| Arc::clone(c))
            .collect();
        // `a OR (a AND b)` is `a`.
        match and_all(rest) {
            Some(r) => rests.push(r),
            None => return and_all(common).unwrap_or_else(|| Arc::clone(e)),
        }
    }
    let disjunction = rests
        .into_iter()
        .reduce(|a, b| boolean(Builtin::Or, vec![a, b]));
    let mut all = common;
    all.extend(disjunction);
    and_all(all).unwrap_or_else(|| Arc::clone(e))
}

/// The conjuncts of a predicate.
fn conjuncts(e: &Arc<Expr>) -> Vec<Arc<Expr>> {
    let mut out = Vec::new();
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        match &x.node {
            ExprNode::Resolved(ResolvedExpr::Function(f))
                if f.id == FunctionId::Builtin(Builtin::And) =>
            {
                stack.extend(f.args.iter().rev().cloned());
            }
            _ => out.push(x),
        }
    }
    out
}

/// `e`'s operand when `e` is `NOT x`.
fn not_arg(e: &Arc<Expr>) -> Option<&Arc<Expr>> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Function(f))
            if f.id == FunctionId::Builtin(Builtin::Not) && f.args.len() == 1 =>
        {
            Some(&f.args[0])
        }
        _ => None,
    }
}

/// The `IN` subquery when `e` is `NOT (v IN (q))`.
fn negated_in(e: &Arc<Expr>) -> Option<&Arc<Expr>> {
    not_arg(e).filter(|x| {
        matches!(
            &x.node,
            ExprNode::Subquery {
                kind: SubqueryKind::In(_),
                ..
            }
        )
    })
}

/// Whether `e` is a NULL literal or a cast of one.
fn null_literal(e: &Expr) -> bool {
    match &e.node {
        ExprNode::Literal(v) => matches!(v, Literal::Null(_)),
        ExprNode::Cast { expr, .. } | ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => {
            null_literal(expr)
        }
        _ => false,
    }
}

/// Whether `p` is one row of expressions over no relation: `SELECT 1`,
/// `VALUES (1)`, through aliases and projections.
fn one_row_literals(p: &LogicalPlan) -> bool {
    match &p.node {
        PlanNode::OneRowRelation => true,
        PlanNode::InlineTable { rows, .. } => rows.len() == 1,
        PlanNode::Project { input, .. } | PlanNode::SubqueryAlias { input, .. } => {
            one_row_literals(input)
        }
        _ => false,
    }
}

/// The `IN` forms Spark folds before any join (`NullPropagation`, the
/// one-row literal rewrite): `v IN (q)` with a NULL `v`, or over a one-row
/// literal relation. They are refused rather than answered two-valued.
fn check_in(values: &[Arc<Expr>], plan: &Arc<LogicalPlan>, origin: Origin) -> R<()> {
    if values.iter().any(|v| null_literal(v)) {
        return Err(unsupported("IN (subquery) with a NULL value", origin));
    }
    if one_row_literals(plan) {
        return Err(unsupported(
            "IN (subquery) over a one-row relation of literals",
            origin,
        ));
    }
    Ok(())
}

/// The type and nullability of a resolved expression.
pub(super) fn typed(e: &Expr) -> (Option<DataType>, bool) {
    match &e.node {
        ExprNode::Resolved(
            ResolvedExpr::Column(c)
            | ResolvedExpr::OuterColumn(c)
            | ResolvedExpr::Alias { output: c, .. },
        ) => (Some(c.data_type.clone()), c.nullable),
        ExprNode::Resolved(ResolvedExpr::Function(f)) => (Some(f.data_type.clone()), f.nullable),
        ExprNode::Literal(v) => (literal_type(v), matches!(v, Literal::Null(_))),
        ExprNode::Cast { expr, to, .. } => (Some(to.clone()), typed(expr).1),
        ExprNode::Window(w) => typed(&w.function),
        _ => (None, true),
    }
}

/// A resolved call of operator `op`.
fn call(op: Builtin, args: Vec<Arc<Expr>>, data_type: DataType, nullable: bool) -> Arc<Expr> {
    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
        ResolvedFunction {
            id: FunctionId::Builtin(op),
            kind: FunctionKind::Scalar,
            args,
            distinct: false,
            filter: None,
            within_group: Vec::new(),
            data_type,
            nullable,
            deterministic: true,
        },
    ))))
}

fn boolean(op: Builtin, args: Vec<Arc<Expr>>) -> Arc<Expr> {
    let nullable = op != Builtin::IsNull && args.iter().any(|a| typed(a).1);
    call(op, args, DataType::Boolean, nullable)
}

/// The conjunction of `parts` (at least one).
fn and_all(parts: Vec<Arc<Expr>>) -> Option<Arc<Expr>> {
    parts
        .into_iter()
        .reduce(|a, b| boolean(Builtin::And, vec![a, b]))
}

fn node(node: PlanNode, origin: Origin) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(node).with_origin(origin))
}

/// A projection of `out` over `input`.
fn columns(input: Arc<LogicalPlan>, out: &[BoundColumn]) -> Arc<LogicalPlan> {
    let exprs = out.iter().map(|c| col_expr(c, Origin::UNKNOWN)).collect();
    node(PlanNode::Project { input, exprs }, Origin::UNKNOWN)
}

/// The output of a resolved relation the rewrite builds on.
fn output(p: &LogicalPlan) -> R<Vec<BoundColumn>> {
    p.resolved_output()
        .ok_or_else(|| unsupported("a subquery over an unresolved plan", p.origin))
}

/// The span of attribute ids `p`'s relations define, as Resolve counts
/// them for a join's sides (a shift counts at its offset; subquery plans
/// inside expressions do not count).
fn hull(p: &Arc<LogicalPlan>) -> Option<(u64, u64)> {
    fn add(h: &mut Option<(u64, u64)>, lo: u64, hi: u64) {
        *h = Some(h.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
    }
    let mut memo: HashMap<usize, Option<(u64, u64)>> = HashMap::new();
    let mut stack = vec![(Arc::clone(p), false)];
    while let Some((x, expanded)) = stack.pop() {
        if memo.contains_key(&addr(&*x)) {
            continue;
        }
        let mut kids = Vec::new();
        x.for_each_child(&mut |c| kids.push(Arc::clone(c)));
        if !expanded {
            stack.push((Arc::clone(&x), true));
            stack.extend(kids.into_iter().map(|c| (c, false)));
            continue;
        }
        let mut h = None;
        if let PlanNode::Shift { offset, .. } = &x.node {
            if let Some((lo, hi)) = kids
                .first()
                .and_then(|c| memo.get(&addr(&**c)).copied().flatten())
            {
                add(
                    &mut h,
                    lo.saturating_add(*offset),
                    hi.saturating_add(*offset),
                );
            }
        } else {
            for c in &kids {
                if let Some((lo, hi)) = memo.get(&addr(&**c)).copied().flatten() {
                    add(&mut h, lo, hi);
                }
            }
            for c in x.resolved_output().unwrap_or_default() {
                add(&mut h, c.id.0, c.id.0);
            }
        }
        memo.insert(addr(&*x), h);
    }
    memo.get(&addr(&**p)).copied().flatten()
}

impl Rewriter {
    /// `root` rewritten, children first, once per distinct relation; every
    /// `STEP` levels on a fresh stack.
    fn plan(&mut self, root: &Arc<LogicalPlan>) -> R<Arc<LogicalPlan>> {
        if let Some((_, out)) = self.done.get(&addr(&**root)) {
            return Ok(Arc::clone(out));
        }
        self.depth += 1;
        let out = if self.depth.is_multiple_of(STEP) {
            on_big_stack(|| self.plan_here(root))
                .unwrap_or_else(|| Err(AnalysisError::new(FailedToParseTooComplex {}, root.origin)))
        } else {
            self.plan_here(root)
        };
        self.depth -= 1;
        let out = out?;
        self.done
            .insert(addr(&**root), (Arc::clone(root), Arc::clone(&out)));
        Ok(out)
    }

    fn plan_here(&mut self, root: &Arc<LogicalPlan>) -> R<Arc<LogicalPlan>> {
        let rebuilt = Arc::clone(root).map_children(|c| self.plan(&c))?;
        self.relation(rebuilt)
    }

    /// Whether the subquery `e` refers to its enclosing query.
    fn correlated(&mut self, e: &Expr) -> bool {
        let ExprNode::Subquery { plan, .. } = &e.node else {
            return false;
        };
        if let Some((_, c)) = self.correlated.get(&addr(&**plan)) {
            return *c;
        }
        let c = has_outer(Node::Plan(plan));
        self.correlated.insert(addr(&**plan), (Arc::clone(plan), c));
        c
    }

    fn relation(&mut self, p: Arc<LogicalPlan>) -> R<Arc<LogicalPlan>> {
        // Subqueries first: what is left correlated refers to this node.
        let p = p.map_expressions(|e| self.nested(e))?;
        match &p.node {
            PlanNode::Filter { input, predicate } => self.filter(&p, Arc::clone(input), predicate),
            PlanNode::Project { input, exprs } => {
                self.project(&p, Arc::clone(input), exprs.clone())
            }
            _ => {
                let mut bad = false;
                p.for_each_expr(&mut |e| bad |= self.reads_correlated(e));
                if bad {
                    return Err(unsupported(
                        "correlated subqueries outside a filter or a projection",
                        p.origin,
                    ));
                }
                Ok(p)
            }
        }
    }

    /// `e` with the plan of each of its subqueries rewritten.
    fn nested(&mut self, e: Arc<Expr>) -> R<Arc<Expr>> {
        if subqueries(&e).is_empty() {
            return Ok(e);
        }
        e.transform_up(&mut |x: Arc<Expr>| match &x.node {
            ExprNode::Subquery { kind, plan } => {
                let rewritten = self.plan(plan)?;
                if Arc::ptr_eq(&rewritten, plan) {
                    return Ok(x);
                }
                let node = ExprNode::Subquery {
                    kind: kind.clone(),
                    plan: rewritten,
                };
                Ok(Arc::new(Expr::new(node).with_origin(x.origin)))
            }
            _ => Ok(x),
        })
    }

    /// A predicate subquery conjunct this rewrite turns into a join.
    fn predicate(&mut self, c: &Arc<Expr>) -> Option<Pred> {
        let (negated, inner) = match &c.node {
            ExprNode::Resolved(ResolvedExpr::Function(f))
                if f.id == FunctionId::Builtin(Builtin::Not) && f.args.len() == 1 =>
            {
                (true, &f.args[0])
            }
            _ => (false, c),
        };
        let ExprNode::Subquery { kind, plan } = &inner.node else {
            return None;
        };
        let correlated = self.correlated(inner);
        let (kind, values, null_aware) = match kind {
            // An uncorrelated EXISTS stays an expression, computed once.
            SubqueryKind::Exists if !correlated => return None,
            SubqueryKind::Exists if negated => (JoinKind::LeftAnti, Vec::new(), false),
            SubqueryKind::Exists => (JoinKind::LeftSemi, Vec::new(), false),
            SubqueryKind::In(values) if negated => (JoinKind::LeftAnti, values.clone(), true),
            SubqueryKind::In(values) => (JoinKind::LeftSemi, values.clone(), false),
            SubqueryKind::Scalar => return None,
        };
        Some(Pred {
            kind,
            null_aware,
            values,
            plan: Arc::clone(plan),
        })
    }

    /// A filter's predicate subqueries as semi / anti joins and its
    /// correlated scalar subqueries as outer joins. Conjuncts that read the
    /// enclosing query go above the joins, where its own rewrite pulls them
    /// up; the others stay below.
    fn filter(
        &mut self,
        p: &Arc<LogicalPlan>,
        input: Arc<LogicalPlan>,
        predicate: &Arc<Expr>,
    ) -> R<Arc<LogicalPlan>> {
        let mut plain = Vec::new();
        let mut above = Vec::new();
        let mut joins = Vec::new();
        let mut scalars = Vec::new();
        let mut existential = Vec::new();
        for c in conjuncts(predicate) {
            if let Some(pred) = self.predicate(&c) {
                joins.push(pred);
                continue;
            }
            if self.existential(&c) {
                existential.push(c);
                continue;
            }
            let subs = subqueries(&c);
            if subs.iter().any(|s| self.correlated(s)) {
                scalars.push(c);
            } else if has_outer(Node::Expr(&c)) {
                above.push(c);
            } else {
                plain.push(c);
            }
        }
        if joins.is_empty() && scalars.is_empty() && existential.is_empty() {
            return Ok(Arc::clone(p));
        }
        let out = output(&input)?;
        let mut base = match and_all(plain) {
            Some(predicate) => node(PlanNode::Filter { input, predicate }, p.origin),
            None => input,
        };
        // Whether outer joins added columns the filter does not output.
        let mut widened = false;
        for mut pred in joins {
            // `(correlated scalar subquery) IN (q)`: its value first.
            if pred.values.iter().any(|v| self.reads_correlated(v)) {
                let (joined, values) = self.scalars(base, std::mem::take(&mut pred.values))?;
                (base, pred.values, widened) = (joined, values, true);
            }
            base = self.semi(base, &pred)?;
        }
        if !existential.is_empty() {
            let (joined, conds) = self.existence(base, existential)?;
            (base, widened) = (joined, true);
            scalars.extend(conds);
        }
        if !scalars.is_empty() {
            let (joined, conds) = self.scalars(base, scalars)?;
            (base, widened) = (joined, true);
            above.extend(conds);
        }
        if let Some(predicate) = and_all(above) {
            base = node(
                PlanNode::Filter {
                    input: base,
                    predicate,
                },
                p.origin,
            );
        }
        Ok(if widened { columns(base, &out) } else { base })
    }

    /// Whether `e` holds a correlated subquery.
    fn reads_correlated(&mut self, e: &Arc<Expr>) -> bool {
        subqueries(e).iter().any(|s| self.correlated(s))
    }

    fn project(
        &mut self,
        p: &Arc<LogicalPlan>,
        input: Arc<LogicalPlan>,
        exprs: Vec<Arc<Expr>>,
    ) -> R<Arc<LogicalPlan>> {
        if !exprs
            .iter()
            .any(|e| self.reads_correlated(e) || self.existential(e))
        {
            return Ok(Arc::clone(p));
        }
        let (base, exprs) = self.existence(input, exprs)?;
        let (base, exprs) = self.scalars(base, exprs)?;
        Ok(node(PlanNode::Project { input: base, exprs }, p.origin))
    }

    /// Whether `e` holds an `EXISTS` or `IN` subquery that an existence
    /// join computes: a correlated `EXISTS`, or any `IN`. (An uncorrelated
    /// `EXISTS` stays an expression computed once.)
    fn existential(&mut self, e: &Arc<Expr>) -> bool {
        subqueries(e).iter().any(|s| match &s.node {
            ExprNode::Subquery {
                kind: SubqueryKind::Exists,
                ..
            } => self.correlated(s),
            ExprNode::Subquery {
                kind: SubqueryKind::In(_),
                ..
            } => true,
            _ => false,
        })
    }

    /// The `EXISTS` / `IN` subqueries of `exprs` that [`Self::existential`]
    /// finds, each a left existence join of `base` with it (one per
    /// distinct subquery expression, and one more when the same `IN` is also
    /// wrapped directly by `NOT`); `exprs` read
    /// the joins' `exists` columns. A `NOT` that directly wraps an `IN`
    /// reads `NOT exists` over a null-aware join, as Spark plans it.
    fn existence(
        &mut self,
        base: Arc<LogicalPlan>,
        exprs: Vec<Arc<Expr>>,
    ) -> R<(Arc<LogicalPlan>, Vec<Arc<Expr>>)> {
        let mut base = base;
        // (subquery address, null-aware) -> its `exists` column.
        let mut found: HashMap<(usize, bool), Arc<Expr>> = HashMap::new();
        for e in &exprs {
            for (sub, null_aware) in self.predicates(e)? {
                if found.contains_key(&(addr(&*sub), null_aware)) {
                    continue;
                }
                let ExprNode::Subquery { kind, plan } = &sub.node else {
                    continue;
                };
                let values = match kind {
                    SubqueryKind::In(values) => values.clone(),
                    _ => Vec::new(),
                };
                let id = self.fresh();
                let mut pred = Pred {
                    kind: JoinKind::Existence(id),
                    null_aware,
                    values,
                    plan: Arc::clone(plan),
                };
                // `(correlated scalar subquery) IN (q)`: its value first.
                if pred.values.iter().any(|v| self.reads_correlated(v)) {
                    let (joined, values) = self.scalars(base, std::mem::take(&mut pred.values))?;
                    (base, pred.values) = (joined, values);
                }
                base = self.semi(base, &pred)?;
                found.insert(
                    (addr(&*sub), null_aware),
                    col_expr(&JoinKind::existence_column(id), Origin::UNKNOWN),
                );
            }
        }
        let mut out = Vec::new();
        for e in exprs {
            let replaced = e.transform_down(&mut |x: Arc<Expr>| {
                Ok::<_, std::convert::Infallible>(match negated_in(&x) {
                    Some(sub) => found.get(&(addr(&**sub), true)).map_or_else(
                        || Arc::clone(&x),
                        |col| boolean(Builtin::Not, vec![Arc::clone(col)]),
                    ),
                    None => found.get(&(addr(&*x), false)).cloned().unwrap_or(x),
                })
            });
            out.push(replaced.unwrap_or_else(|never| match never {}));
        }
        Ok((base, out))
    }

    /// The subqueries of `e` an existence join computes, in the order they
    /// read (left to right, depth first), each with whether a `NOT` wraps
    /// it directly (an `IN` only: the null-aware form). Refuses the forms
    /// Spark decides before any join, where a two-valued join would answer
    /// FALSE for Spark's NULL.
    fn predicates(&mut self, e: &Arc<Expr>) -> R<Vec<(Arc<Expr>, bool)>> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        // Expressions are DAGs: each node once (the subquery key above is
        // separate, so a subquery under `NOT` and bare yields both).
        let mut visited = HashSet::new();
        let mut stack = vec![Arc::clone(e)];
        while let Some(x) = stack.pop() {
            if !visited.insert(addr(&*x)) {
                continue;
            }
            if not_arg(&x).is_some_and(|y| negated_in(y).is_some()) {
                return Err(unsupported("NOT IN under a further NOT", x.origin));
            }
            let (sub, null_aware) = match negated_in(&x) {
                Some(sub) => (Arc::clone(sub), true),
                None => (Arc::clone(&x), false),
            };
            if let ExprNode::Subquery { kind, plan } = &sub.node {
                let wanted = match kind {
                    SubqueryKind::Exists => self.correlated(&sub),
                    SubqueryKind::In(values) => {
                        check_in(values, plan, sub.origin)?;
                        true
                    }
                    SubqueryKind::Scalar => false,
                };
                if wanted && seen.insert((addr(&*sub), null_aware)) {
                    out.push((Arc::clone(&sub), null_aware));
                }
            }
            let mut kids = Vec::new();
            sub.for_each_child(&mut |k| kids.push(Arc::clone(k)));
            stack.extend(kids.into_iter().rev());
        }
        Ok(out)
    }

    /// A predicate subquery as a semi or anti join of `left` with it.
    fn semi(&mut self, left: Arc<LogicalPlan>, pred: &Pred) -> R<Arc<LogicalPlan>> {
        let keep = !pred.values.is_empty();
        let (inner, corr) = self.pull(&pred.plan, keep)?;
        let out = output(&inner)?;
        let (right, renumber) = self.disjoint(&left, inner)?;
        let mut cond = Vec::new();
        for (v, c) in pred.values.iter().zip(&out) {
            let c = renumber.inner(&col_expr(c, Origin::UNKNOWN));
            let eq = boolean(Builtin::Eq, vec![Arc::clone(v), c]);
            cond.push(if pred.null_aware {
                let unknown = boolean(Builtin::IsNull, vec![Arc::clone(&eq)]);
                boolean(Builtin::Or, vec![eq, unknown])
            } else {
                eq
            });
        }
        cond.extend(corr.iter().map(|c| renumber.inner(c)));
        let condition = and_all(cond).map_or(JoinCondition::None, JoinCondition::On);
        let join = PlanNode::Join {
            left,
            right,
            kind: pred.kind,
            condition,
        };
        Ok(node(join, Origin::UNKNOWN))
    }

    /// `plan` without the correlated conjuncts of the filters at its top
    /// (through projections and aliases), and those conjuncts. With `keep`,
    /// projections keep their columns and add the ones the conjuncts read;
    /// without (`EXISTS`), they go.
    fn pull(
        &mut self,
        plan: &Arc<LogicalPlan>,
        keep: bool,
    ) -> R<(Arc<LogicalPlan>, Vec<Arc<Expr>>)> {
        if !has_outer(Node::Plan(plan)) {
            return Ok((Arc::clone(plan), Vec::new()));
        }
        match &plan.node {
            PlanNode::Filter { input, predicate } => {
                // `(k = o AND p) OR (k = o AND q)` is `k = o AND (p OR q)`:
                // the correlation reads as a conjunct (Spark's optimizer
                // factors the common conjuncts of a disjunction too).
                let predicate = &factor_or(predicate);
                let (corr, rest): (Vec<_>, Vec<_>) = conjuncts(predicate)
                    .into_iter()
                    .partition(|c| has_outer(Node::Expr(c)));
                if corr.iter().any(|c| has_subquery(Node::Expr(c))) {
                    return Err(unsupported(
                        "correlated predicates with subqueries",
                        plan.origin,
                    ));
                }
                let (inner, mut more) = self.pull(input, true)?;
                more.extend(corr);
                let inner = match and_all(rest) {
                    Some(predicate) => node(
                        PlanNode::Filter {
                            input: inner,
                            predicate,
                        },
                        plan.origin,
                    ),
                    None => inner,
                };
                Ok((inner, more))
            }
            PlanNode::Project { input, exprs } => {
                // EXISTS reads no column: its select list goes.
                if !keep {
                    return self.pull(input, true);
                }
                if exprs.iter().any(|e| has_outer(Node::Expr(e))) {
                    return Err(unsupported(
                        "outer references in a subquery's select list",
                        plan.origin,
                    ));
                }
                let (inner, corr) = self.pull(input, true)?;
                let mut exprs = exprs.clone();
                let mut have: HashSet<AttrId> = output(plan)?.iter().map(|c| c.id).collect();
                for c in &corr {
                    for col in inner_columns(c) {
                        if have.insert(col.id) {
                            exprs.push(col_expr(&col, Origin::UNKNOWN));
                        }
                    }
                }
                Ok((
                    node(
                        PlanNode::Project {
                            input: inner,
                            exprs,
                        },
                        plan.origin,
                    ),
                    corr,
                ))
            }
            PlanNode::SubqueryAlias {
                input,
                alias,
                columns,
            } if columns.is_empty() => {
                let (inner, corr) = self.pull(input, keep)?;
                let node_ = PlanNode::SubqueryAlias {
                    input: inner,
                    alias: alias.clone(),
                    columns: Vec::new(),
                };
                Ok((node(node_, plan.origin), corr))
            }
            PlanNode::Sort { input, order, .. } => {
                let outer: Vec<String> = order
                    .iter()
                    .filter(|o| has_outer(Node::Expr(&o.expr)))
                    .map(|o| {
                        let dir = match o.direction {
                            SortDirection::Ascending => "ASC",
                            SortDirection::Descending => "DESC",
                        };
                        let nulls = match o.nulls {
                            crate::NullOrdering::NullsFirst => "NULLS FIRST",
                            crate::NullOrdering::NullsLast => "NULLS LAST",
                        };
                        format!("\"{} {dir} {nulls}\"", pretty(&o.expr, None))
                    })
                    .collect();
                if outer.is_empty() {
                    // Without a limit, a subquery's order means nothing (as
                    // Spark's `RemoveSortInSubquery` has it): it goes.
                    return self.pull(input, keep);
                }
                Err(AnalysisError::new(
                    UnsupportedSubqueryExpressionCategoryCorrelatedReference {
                        sql_exprs: outer.join(", "),
                    },
                    plan.origin,
                ))
            }
            // Spark rejects a correlated reference in a nested aggregate.
            PlanNode::Aggregate { aggregates, .. }
                if aggregates.iter().any(|e| has_outer(Node::Expr(e))) =>
            {
                let outer: Vec<String> = aggregates
                    .iter()
                    .filter(|e| has_outer(Node::Expr(e)))
                    .map(|e| format!("\"{}\"", pretty(e, None)))
                    .collect();
                Err(AnalysisError::new(
                    UnsupportedSubqueryExpressionCategoryCorrelatedReference {
                        sql_exprs: outer.join(", "),
                    },
                    plan.origin,
                ))
            }
            _ => Err(unsupported(
                "outer references below a join, aggregate, limit or set operation",
                plan.origin,
            )),
        }
    }

    /// `right`, renumbered where its attribute ids would meet `left`'s (so
    /// a join of the two never shares an id).
    fn disjoint(
        &mut self,
        left: &Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
    ) -> R<(Arc<LogicalPlan>, Renumber)> {
        let (Some((a, b)), Some((lo, hi))) = (hull(left), hull(&right)) else {
            return Ok((right, Renumber::NONE));
        };
        if b < lo || hi < a {
            return Ok((right, Renumber::NONE));
        }
        let offset = self.next_id.saturating_sub(lo);
        if hi.checked_add(offset).is_none_or(|top| top > MAX_ID) {
            return Err(AnalysisError::new(FailedToParseTooComplex {}, right.origin));
        }
        self.next_id += hi - lo + 1;
        let shift = PlanNode::Shift {
            input: right,
            offset,
        };
        Ok((node(shift, Origin::UNKNOWN), Renumber { lo, hi, offset }))
    }

    /// The correlated scalar subqueries of `exprs`, each a left outer join
    /// of `base` with its aggregate; `exprs` read the aggregates' values.
    fn scalars(
        &mut self,
        base: Arc<LogicalPlan>,
        exprs: Vec<Arc<Expr>>,
    ) -> R<(Arc<LogicalPlan>, Vec<Arc<Expr>>)> {
        let mut base = base;
        let mut values: HashMap<usize, Arc<Expr>> = HashMap::new();
        for e in &exprs {
            for s in subqueries(e) {
                if !self.correlated(&s) || values.contains_key(&addr(&*s)) {
                    continue;
                }
                let ExprNode::Subquery {
                    kind: SubqueryKind::Scalar,
                    plan,
                } = &s.node
                else {
                    return Err(unsupported(
                        "correlated EXISTS and IN subqueries inside OR, NOT or other expressions",
                        s.origin,
                    ));
                };
                let (joined, value) = self.scalar(base, plan)?;
                base = joined;
                values.insert(addr(&*s), value);
            }
        }
        let mut out = Vec::new();
        for e in exprs {
            let replaced = e.transform_up(&mut |x: Arc<Expr>| {
                Ok::<_, std::convert::Infallible>(values.get(&addr(&*x)).cloned().unwrap_or(x))
            });
            out.push(replaced.unwrap_or_else(|never| match never {}));
        }
        Ok((base, out))
    }

    /// A correlated scalar subquery as a left outer join of `base` with
    /// its aggregate grouped by its correlated keys, and the expression
    /// that reads its value.
    fn scalar(
        &mut self,
        base: Arc<LogicalPlan>,
        plan: &Arc<LogicalPlan>,
    ) -> R<(Arc<LogicalPlan>, Arc<Expr>)> {
        let PlanNode::Aggregate {
            input,
            grouping: Grouping::GroupBy(keys),
            aggregates,
        } = &plan.node
        else {
            return Err(unsupported(
                "correlated scalar subqueries without an aggregate",
                plan.origin,
            ));
        };
        let [item] = &aggregates[..] else {
            return Err(unsupported(
                "a correlated scalar subquery of this shape",
                plan.origin,
            ));
        };
        let ExprNode::Resolved(ResolvedExpr::Alias {
            expr,
            output: value,
        }) = &item.node
        else {
            return Err(unsupported(
                "a correlated scalar subquery of this shape",
                plan.origin,
            ));
        };
        if !keys.is_empty() {
            return Err(unsupported(
                "correlated scalar subqueries with GROUP BY",
                plan.origin,
            ));
        }
        // Spark rejects outer references in the value
        // (`AGGREGATE_FUNCTION_MIXED_OUTER_LOCAL_REFERENCES`, or an outer
        // aggregate's `CORRELATED_REFERENCE`); the join could not bind them.
        if has_outer(Node::Expr(item)) {
            return Err(unsupported(
                "outer references in a correlated scalar subquery's value",
                item.origin,
            ));
        }
        let empty = on_no_rows(expr).ok_or_else(|| {
            unsupported(
                "a correlated scalar subquery whose value over no rows is not NULL or a count",
                plan.origin,
            )
        })?;
        if let Some(union) = under_union(input) {
            let (inner, inner_keys, outer_keys) = self.domain_union(&base, input, union)?;
            return self.scalar_join(
                base,
                plan,
                item,
                value,
                empty,
                inner,
                &inner_keys,
                &outer_keys,
            );
        }
        let (inner, corr) = self.pull(input, true)?;
        // Each correlated predicate is `inner = outer`: a grouping key.
        let mut inner_keys = Vec::new();
        let mut outer_keys = Vec::new();
        for c in &corr {
            let pair = match &c.node {
                ExprNode::Resolved(ResolvedExpr::Function(f))
                    if f.id == FunctionId::Builtin(Builtin::Eq) && f.args.len() == 2 =>
                {
                    let (a, b) = (&f.args[0], &f.args[1]);
                    match (has_outer(Node::Expr(a)), has_outer(Node::Expr(b))) {
                        (false, true) if inner_columns(b).is_empty() => Some((a, b)),
                        (true, false) if inner_columns(a).is_empty() => Some((b, a)),
                        _ => None,
                    }
                }
                _ => None,
            };
            let Some((i, o)) = pair else {
                return Err(unsupported(
                    "correlated scalar subqueries whose correlation is not an equality",
                    plan.origin,
                ));
            };
            inner_keys.push(Arc::clone(i));
            outer_keys.push(Arc::clone(o));
        }
        self.scalar_join(
            base,
            plan,
            item,
            value,
            empty,
            inner,
            &inner_keys,
            &outer_keys,
        )
    }

    /// The left outer join of `base` with the aggregate of `inner` (the
    /// subquery's input, its correlation lifted out) grouped by
    /// `inner_keys`, matched to `outer_keys`; and the value it reads.
    #[expect(clippy::too_many_arguments, reason = "the pieces `scalar` has built")]
    fn scalar_join(
        &mut self,
        base: Arc<LogicalPlan>,
        plan: &Arc<LogicalPlan>,
        item: &Arc<Expr>,
        value: &BoundColumn,
        empty: Option<Literal>,
        inner: Arc<LogicalPlan>,
        inner_keys: &[Arc<Expr>],
        outer_keys: &[Arc<Expr>],
    ) -> R<(Arc<LogicalPlan>, Arc<Expr>)> {
        // The aggregate outputs its value and its keys.
        let mut aggregates = vec![Arc::clone(item)];
        let mut key_cols: Vec<BoundColumn> = Vec::new();
        let mut grouping: Vec<Arc<Expr>> = Vec::new();
        for k in inner_keys {
            let col = match &k.node {
                // A column met again (`u.a = t.a AND u.a = t.x`) is one key:
                // the aggregate outputs each attribute once.
                ExprNode::Resolved(ResolvedExpr::Column(c)) => {
                    if !key_cols.iter().any(|o| o.id == c.id) {
                        aggregates.push(Arc::clone(k));
                        grouping.push(Arc::clone(k));
                    }
                    c.clone()
                }
                _ => {
                    let (data_type, nullable) = typed(k);
                    let data_type = data_type
                        .ok_or_else(|| unsupported("a correlation key of this form", k.origin))?;
                    let output = BoundColumn {
                        id: self.fresh(),
                        name: pretty(k, None),
                        data_type,
                        nullable,
                    };
                    aggregates.push(Arc::new(Expr::new(ExprNode::Resolved(
                        ResolvedExpr::Alias {
                            expr: Arc::clone(k),
                            output: output.clone(),
                        },
                    ))));
                    grouping.push(Arc::clone(k));
                    output
                }
            };
            key_cols.push(col);
        }
        let agg = node(
            PlanNode::Aggregate {
                input: inner,
                grouping: Grouping::GroupBy(grouping),
                aggregates,
            },
            plan.origin,
        );
        let (right, renumber) = self.disjoint(&base, agg)?;
        let cond = outer_keys
            .iter()
            .zip(&key_cols)
            .map(|(o, k)| {
                let k = renumber.inner(&col_expr(k, Origin::UNKNOWN));
                boolean(Builtin::Eq, vec![renumber.inner(o), k])
            })
            .collect();
        let join = PlanNode::Join {
            left: base,
            right,
            kind: JoinKind::LeftOuter,
            condition: and_all(cond).map_or(JoinCondition::None, JoinCondition::On),
        };
        // The value, NULL where no row matched (an outer join's right side).
        let value = BoundColumn {
            id: renumber.id(value.id),
            nullable: true,
            ..value.clone()
        };
        let read = col_expr(&value, Origin::UNKNOWN);
        let read = match empty {
            None => read,
            Some(zero) => {
                let missing = boolean(Builtin::IsNull, vec![Arc::clone(&read)]);
                let zero = Arc::new(Expr::literal(zero));
                call(
                    Builtin::When,
                    vec![missing, zero, read],
                    value.data_type,
                    false,
                )
            }
        };
        Ok((node(join, Origin::UNKNOWN), read))
    }

    /// The input of a correlated scalar aggregate that is a `UNION ALL`
    /// (`union`, under `input`'s aliases), decorrelated through the domain
    /// of the outer columns it reads: each branch joins the distinct
    /// values of those columns on its correlated predicates and outputs
    /// them, so the aggregate groups by them. Returns the new input, the
    /// grouping keys and the outer columns they match.
    fn domain_union(
        &mut self,
        base: &Arc<LogicalPlan>,
        input: &Arc<LogicalPlan>,
        union: &Arc<LogicalPlan>,
    ) -> R<Lifted> {
        let PlanNode::Union {
            inputs,
            by_name: false,
            ..
        } = &union.node
        else {
            return Err(unsupported("a set operation by name", union.origin));
        };
        let mut pulled = Vec::new();
        let mut domain: Vec<BoundColumn> = Vec::new();
        for branch in inputs {
            // A select list that reads the outer query reads the domain.
            let (inner, corr, select) = match &branch.node {
                PlanNode::Project { input, exprs }
                    if exprs.iter().any(|e| has_outer(Node::Expr(e))) =>
                {
                    let (inner, corr) = self.pull(input, true)?;
                    (inner, corr, Some(exprs.clone()))
                }
                _ => {
                    let (inner, corr) = self.pull(branch, true)?;
                    (inner, corr, None)
                }
            };
            for c in corr.iter().chain(select.iter().flatten()) {
                for col in outer_columns(c) {
                    if !domain.iter().any(|d| d.id == col.id) {
                        domain.push(col);
                    }
                }
            }
            pulled.push((branch, inner, corr, select));
        }
        if domain.is_empty() {
            return Err(unsupported(
                "a set operation without correlation",
                union.origin,
            ));
        }
        let keys = columns(Arc::clone(base), &domain);
        let exprs: Vec<_> = domain
            .iter()
            .map(|c| col_expr(c, Origin::UNKNOWN))
            .collect();
        let distinct = node(
            PlanNode::Aggregate {
                input: keys,
                grouping: Grouping::GroupBy(exprs.clone()),
                aggregates: exprs,
            },
            Origin::UNKNOWN,
        );
        let mut branches = Vec::new();
        let mut first_domain = Vec::new();
        for (i, (branch, inner, corr, select)) in pulled.into_iter().enumerate() {
            // The first branch outputs the domain under its own ids, the
            // others under fresh ones; no branch's join shares an id.
            let (right, renumber) = if i == 0 {
                self.disjoint(&inner, Arc::clone(&distinct))?
            } else {
                self.shifted(Arc::clone(&distinct))?
            };
            let cond = corr.iter().map(|c| renumber.outer(c)).collect::<Vec<_>>();
            let join = node(
                PlanNode::Join {
                    left: inner,
                    right,
                    kind: JoinKind::Inner,
                    condition: and_all(cond).map_or(JoinCondition::None, JoinCondition::On),
                },
                Origin::UNKNOWN,
            );
            let out = output(branch)?;
            let dom: Vec<BoundColumn> = domain
                .iter()
                .map(|c| BoundColumn {
                    id: renumber.id(c.id),
                    ..c.clone()
                })
                .collect();
            if i == 0 {
                first_domain.clone_from(&dom);
            }
            let exprs: Vec<Arc<Expr>> = match select {
                Some(select) => select.iter().map(|e| renumber.outer(e)).collect(),
                None => out.iter().map(|c| col_expr(c, Origin::UNKNOWN)).collect(),
            };
            let exprs = exprs
                .into_iter()
                .chain(dom.iter().map(|c| col_expr(c, Origin::UNKNOWN)))
                .collect();
            branches.push(node(
                PlanNode::Project { input: join, exprs },
                Origin::UNKNOWN,
            ));
        }
        let new_union = node(
            PlanNode::Union {
                inputs: branches,
                by_name: false,
                allow_missing_columns: false,
            },
            union.origin,
        );
        let inner_keys: Vec<Arc<Expr>> = first_domain
            .iter()
            .map(|c| col_expr(c, Origin::UNKNOWN))
            .collect();
        let inner = rewrap(input, &new_union, &inner_keys);
        let outer_keys = domain
            .iter()
            .map(|c| {
                Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::OuterColumn(
                    c.clone(),
                ))))
            })
            .collect();
        Ok((inner, inner_keys, outer_keys))
    }

    /// `right` renumbered to fresh attribute ids whatever it overlaps.
    fn shifted(&mut self, right: Arc<LogicalPlan>) -> R<(Arc<LogicalPlan>, Renumber)> {
        let Some((lo, hi)) = hull(&right) else {
            return Ok((right, Renumber::NONE));
        };
        let offset = self.next_id.saturating_sub(lo);
        if hi.checked_add(offset).is_none_or(|top| top > MAX_ID) {
            return Err(AnalysisError::new(FailedToParseTooComplex {}, right.origin));
        }
        self.next_id += hi - lo + 1;
        let shift = PlanNode::Shift {
            input: right,
            offset,
        };
        Ok((node(shift, Origin::UNKNOWN), Renumber { lo, hi, offset }))
    }

    fn fresh(&mut self) -> AttrId {
        self.next_id += 1;
        AttrId(self.next_id - 1)
    }
}

/// The union under `input`'s plain aliases and distinct aggregates (`UNION`
/// without `ALL`), if that is what `input` is.
fn under_union(input: &Arc<LogicalPlan>) -> Option<&Arc<LogicalPlan>> {
    match &input.node {
        PlanNode::Union { .. } => Some(input),
        PlanNode::SubqueryAlias { input, columns, .. } if columns.is_empty() => under_union(input),
        PlanNode::Aggregate {
            input: below,
            grouping: Grouping::GroupBy(keys),
            aggregates,
        } if is_distinct(below, keys, aggregates) => under_union(below),
        _ => None,
    }
}

/// Whether an aggregate of `below` with these keys and outputs is
/// `SELECT DISTINCT`: it groups by every column and outputs them.
fn is_distinct(below: &Arc<LogicalPlan>, keys: &[Arc<Expr>], aggregates: &[Arc<Expr>]) -> bool {
    let column = |e: &Arc<Expr>| match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
        _ => None,
    };
    let Some(out) = below.resolved_output() else {
        return false;
    };
    keys.len() == out.len()
        && aggregates.len() == out.len()
        && out
            .iter()
            .zip(keys.iter().zip(aggregates))
            .all(|(o, (k, a))| column(k) == Some(o.id) && column(a) == Some(o.id))
}

/// `input` with the union under it replaced; the distinct aggregates on
/// the way also group by the `extra` columns.
fn rewrap(
    input: &Arc<LogicalPlan>,
    union: &Arc<LogicalPlan>,
    extra: &[Arc<Expr>],
) -> Arc<LogicalPlan> {
    match &input.node {
        PlanNode::SubqueryAlias {
            input: below,
            alias,
            columns,
        } => node(
            PlanNode::SubqueryAlias {
                input: rewrap(below, union, extra),
                alias: alias.clone(),
                columns: columns.clone(),
            },
            input.origin,
        ),
        PlanNode::Aggregate {
            input: below,
            grouping: Grouping::GroupBy(keys),
            aggregates,
        } => {
            let mut keys = keys.clone();
            keys.extend(extra.iter().cloned());
            let mut aggregates = aggregates.clone();
            aggregates.extend(extra.iter().cloned());
            node(
                PlanNode::Aggregate {
                    input: rewrap(below, union, extra),
                    grouping: Grouping::GroupBy(keys),
                    aggregates,
                },
                input.origin,
            )
        }
        _ => Arc::clone(union),
    }
}

/// The outer columns `e` reads.
fn outer_columns(e: &Arc<Expr>) -> Vec<BoundColumn> {
    let mut out: Vec<BoundColumn> = Vec::new();
    let _ = visit_distinct(Node::Expr(e), |n| {
        if let Node::Expr(x) = n
            && let ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) = &x.node
            && !out.iter().any(|o| o.id == c.id)
        {
            out.push(c.clone());
        }
        ControlFlow::<()>::Continue(())
    });
    out
}

/// The columns (not outer references) `e` reads.
fn inner_columns(e: &Arc<Expr>) -> Vec<BoundColumn> {
    let mut out: Vec<BoundColumn> = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        if !seen.insert(addr(&*x)) {
            continue;
        }
        if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &x.node
            && !out.iter().any(|o| o.id == c.id)
        {
            out.push(c.clone());
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    out
}

/// What a scalar subquery's value `e` is over no rows: `None` when NULL,
/// `Some(0)` for a bare `count`; `None` of the outer option when unknown
/// (Spark evaluates other forms; Krishiv does not yet).
fn on_no_rows(e: &Arc<Expr>) -> Option<Option<Literal>> {
    // A bare count is 0 over no rows.
    if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node
        && f.kind == FunctionKind::Aggregate
        && f.id.name() == "count"
    {
        return Some(Some(Literal::Long(0)));
    }
    // Otherwise NULL when every path to a leaf meets an aggregate that is
    // NULL over no rows through operators that keep a NULL.
    let mut stack = vec![Arc::clone(e)];
    let mut any_aggregate = false;
    while let Some(x) = stack.pop() {
        match &x.node {
            ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate => {
                if !matches!(f.id.name(), "sum" | "avg" | "min" | "max" | "any_value") {
                    return None;
                }
                any_aggregate = true;
            }
            ExprNode::Resolved(ResolvedExpr::Function(f))
                if matches!(
                    f.id,
                    FunctionId::Builtin(
                        Builtin::Add
                            | Builtin::Subtract
                            | Builtin::Multiply
                            | Builtin::Divide
                            | Builtin::Remainder
                            | Builtin::IntegralDivide
                            | Builtin::Negative
                    )
                ) =>
            {
                // NULL if either side is: one aggregate side suffices.
                let mut nulls = false;
                for a in &f.args {
                    if on_no_rows(a) == Some(None) {
                        nulls = true;
                    }
                }
                if nulls {
                    any_aggregate = true;
                } else {
                    return None;
                }
            }
            ExprNode::Cast { expr, .. } => stack.push(Arc::clone(expr)),
            _ => return None,
        }
    }
    any_aggregate.then_some(None)
}
