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
//! through projections and aliases. A set operation or a join on the way
//! runs once per distinct value of the outer columns instead: those values
//! join under it and travel up beside its output, and `q` matches them
//! null-safely. Uncorrelated scalar subqueries and
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
    FailedToParseTooComplex,
    UnsupportedSubqueryExpressionCategoryAccessingOuterQueryColumnIsNotAllowed,
    UnsupportedSubqueryExpressionCategoryCorrelatedReference,
    UnsupportedSubqueryExpressionCategoryUnsupportedCorrelatedExpressionInJoinCondition,
};
use krishiv_types::DataType;

use super::bind::{addr, unsupported};
use super::catalog::{FunctionLookup, FunctionRegistry};
use super::exprs::{literal_type, pretty};
use super::ops::{MAX_ID, col_expr};
use super::{AnalysisError, SparkProfile};
use crate::walk::{Node, visit_distinct};
use crate::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FrameUnits, FunctionId, FunctionKind, Grouping,
    JoinCondition, JoinKind, Literal, LogicalPlan, Origin, PlanNode, ResolvedExpr,
    ResolvedFunction, SortDirection, SortOrder, SubqueryKind, WindowExpr, WindowFrame, WindowSpec,
    WindowSpecRef,
};
use krishiv_common::stack::{STEP, on_big_stack};

type R<T> = Result<T, AnalysisError>;

/// A subquery's input with its correlation lifted out: the plan, the keys it
/// groups by, and the outer expressions they match.
type Lifted = (Arc<LogicalPlan>, Vec<Arc<Expr>>, Vec<Arc<Expr>>);
/// The inner and the outer sides of the correlated equalities, pairwise.
type Keys = (Vec<Arc<Expr>>, Vec<Arc<Expr>>);

/// Rewrites the subqueries of `root`; `next_id` is one past every
/// attribute id in use and is kept so.
pub(super) fn rewrite(
    root: &Arc<LogicalPlan>,
    next_id: &mut u64,
    functions: &dyn FunctionRegistry,
    profile: SparkProfile,
) -> R<Arc<LogicalPlan>> {
    if !has_subquery(Node::Plan(root)) {
        return Ok(Arc::clone(root));
    }
    let mut r = Rewriter {
        next_id: *next_id,
        done: HashMap::new(),
        correlated: HashMap::new(),
        depth: 0,
        row_number: match functions.lookup(&["row_number".to_owned()], profile) {
            FunctionLookup::Found(entry) => Some((entry.id, entry.kind)),
            _ => None,
        },
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
    /// The registry's `row_number`, for a subquery's per-row `LIMIT`.
    row_number: Option<(FunctionId, FunctionKind)>,
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

/// `inner_column = outer_expression`, either way round.
fn is_outer_equality(e: &Arc<Expr>) -> bool {
    let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node else {
        return false;
    };
    if f.id != FunctionId::Builtin(Builtin::Eq) || f.args.len() != 2 {
        return false;
    }
    let plain = |a: &Arc<Expr>, b: &Arc<Expr>| {
        matches!(&a.node, ExprNode::Resolved(ResolvedExpr::Column(_)))
            && !has_outer(Node::Expr(a))
            && has_outer(Node::Expr(b))
            && inner_columns(b).is_empty()
    };
    plain(&f.args[0], &f.args[1]) || plain(&f.args[1], &f.args[0])
}

/// Whether `e` is an integer literal above zero.
fn positive_literal(e: &Expr) -> bool {
    match &e.node {
        ExprNode::Literal(v) => match v {
            Literal::Byte(n) => *n > 0,
            Literal::Short(n) => *n > 0,
            Literal::Integer(n) => *n > 0,
            Literal::Long(n) => *n > 0,
            _ => false,
        },
        ExprNode::Cast { expr, .. } => positive_literal(expr),
        _ => false,
    }
}

/// Whether an aggregate sits under `p`'s projections and filters.
fn aggregates_below(p: &LogicalPlan) -> bool {
    match &p.node {
        PlanNode::Aggregate { .. } => true,
        PlanNode::Project { input, .. }
        | PlanNode::SubqueryAlias { input, .. }
        | PlanNode::Filter { input, .. }
        | PlanNode::Sort { input, .. } => aggregates_below(input),
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
            PlanNode::Join {
                left,
                right,
                kind,
                condition: JoinCondition::On(on),
            } if self.existential(on) => self.join(&p, left, right, kind, on),
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

    /// `v [NOT] IN (SELECT agg ...)` over a correlated global aggregate:
    /// the subquery has exactly one row for every outer row, so it is
    /// `v = (scalar subquery)` (negated: `NOT (v = ...)`, the same NULLs).
    fn single_row_in(&mut self, c: &Arc<Expr>) -> Option<Arc<Expr>> {
        let (negated, inner) = match not_arg(c) {
            Some(x) => (true, x),
            None => (false, c),
        };
        let ExprNode::Subquery {
            kind: SubqueryKind::In(values),
            plan,
        } = &inner.node
        else {
            return None;
        };
        let [value] = &values[..] else { return None };
        if !self.correlated(inner) || self.reads_correlated(value) || has_outer(Node::Expr(value)) {
            return None;
        }
        let PlanNode::Aggregate {
            grouping: Grouping::GroupBy(keys),
            aggregates,
            ..
        } = &plan.node
        else {
            return None;
        };
        if !keys.is_empty() || aggregates.len() != 1 {
            return None;
        }
        let sub = Arc::new(Expr::new(ExprNode::Subquery {
            kind: SubqueryKind::Scalar,
            plan: Arc::clone(plan),
        }));
        let eq = boolean(Builtin::Eq, vec![Arc::clone(value), sub]);
        Some(if negated {
            boolean(Builtin::Not, vec![eq])
        } else {
            eq
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
            let c = self.single_row_in(&c).unwrap_or(c);
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

    /// A join whose `ON` holds `EXISTS` / `IN` subqueries: each is an
    /// existence join under the side whose columns it reads, and the `ON`
    /// reads the `exists` column (Spark plans the same `ExistenceJoin`
    /// below the join). A subquery that reads both sides stays unsupported.
    fn join(
        &mut self,
        p: &Arc<LogicalPlan>,
        left: &Arc<LogicalPlan>,
        right: &Arc<LogicalPlan>,
        kind: &JoinKind,
        on: &Arc<Expr>,
    ) -> R<Arc<LogicalPlan>> {
        let out = output(p)?;
        let left_ids: HashSet<AttrId> = output(left)?.iter().map(|c| c.id).collect();
        let right_ids: HashSet<AttrId> = output(right)?.iter().map(|c| c.id).collect();
        let (mut plain, mut on_left, mut on_right) = (Vec::new(), Vec::new(), Vec::new());
        for c in conjuncts(on) {
            if !self.existential(&c) {
                plain.push(c);
                continue;
            }
            let mut ids = outer_ids(&c);
            // An `IN` value reads the side it belongs to, too.
            for s in subqueries(&c) {
                if let ExprNode::Subquery {
                    kind: SubqueryKind::In(values),
                    ..
                } = &s.node
                {
                    ids.extend(values.iter().flat_map(inner_columns).map(|c| c.id));
                }
            }
            if ids.iter().all(|i| left_ids.contains(i)) {
                on_left.push(c);
            } else if ids.iter().all(|i| right_ids.contains(i)) {
                on_right.push(c);
            } else if ids
                .iter()
                .all(|i| left_ids.contains(i) || right_ids.contains(i))
            {
                let shown = subqueries(&c)
                    .iter()
                    .map(|s| pretty(s, None))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(AnalysisError::new(
                    UnsupportedSubqueryExpressionCategoryUnsupportedCorrelatedExpressionInJoinCondition {
                        subquery_expression: shown,
                    },
                    c.origin,
                ));
            } else {
                return Err(unsupported(
                    "correlated subqueries in a join condition that read an enclosing query",
                    c.origin,
                ));
            }
        }
        let (mut l, mut r) = (Arc::clone(left), Arc::clone(right));
        if !on_left.is_empty() {
            let (joined, conds) = self.existence(l, on_left)?;
            l = joined;
            plain.extend(conds);
        }
        if !on_right.is_empty() {
            let (joined, conds) = self.existence(r, on_right)?;
            r = joined;
            plain.extend(conds);
        }
        if plain.iter().any(|c| self.reads_correlated(c)) {
            return Err(unsupported(
                "correlated subqueries outside a filter or a projection",
                p.origin,
            ));
        }
        let condition = and_all(plain).map_or(JoinCondition::None, JoinCondition::On);
        let joined = node(
            PlanNode::Join {
                left: l,
                right: r,
                kind: *kind,
                condition,
            },
            p.origin,
        );
        Ok(columns(joined, &out))
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
        let (inner, corr) = if on_path(&pred.plan, set_op_or_join) {
            let (inner, inner_keys, outer_keys) = self.domain(&left, &pred.plan)?;
            let corr = inner_keys
                .into_iter()
                .zip(outer_keys)
                .map(|(i, o)| boolean(Builtin::EqNullSafe, vec![i, o]))
                .collect();
            (inner, corr)
        } else {
            self.pull(&pred.plan, keep)?
        };
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
                    return self.pull(input, false);
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
                if !order.iter().any(|o| has_outer(Node::Expr(&o.expr))) {
                    // Without a limit, a subquery's order means nothing (as
                    // Spark's `RemoveSortInSubquery` has it): it goes.
                    return self.pull(input, keep);
                }
                Err(correlated_order(order, plan.origin))
            }
            // Spark rejects a correlated reference in a nested aggregate.
            PlanNode::Aggregate { aggregates, .. }
                if aggregates.iter().any(|e| has_outer(Node::Expr(e))) =>
            {
                Err(correlated_reference(aggregates, plan.origin))
            }
            // Nor may it group by one.
            PlanNode::Aggregate {
                grouping: Grouping::GroupBy(keys),
                ..
            } if keys.iter().any(|k| has_outer(Node::Expr(k))) => {
                Err(correlated_reference(keys, plan.origin))
            }
            // EXISTS reads no column: a grouped aggregate with no HAVING
            // has a row exactly when its input has one, and a global one
            // always has exactly one row, whatever the correlation.
            PlanNode::Aggregate {
                input,
                grouping: Grouping::GroupBy(keys),
                ..
            } if !keep && !keys.iter().any(|k| has_outer(Node::Expr(k))) => {
                if keys.is_empty() {
                    // The input's rows do not matter, but what Spark rejects
                    // in it is still rejected.
                    if let Err(e) = self.pull(input, false)
                        && e.error.error_class().name() != "KRISHIV_UNSUPPORTED_FEATURE"
                    {
                        return Err(e);
                    }
                    Ok((node(PlanNode::OneRowRelation, plan.origin), Vec::new()))
                } else {
                    self.pull(input, false)
                }
            }
            // `IN (SELECT agg ... WHERE k = o GROUP BY g)`: the aggregate
            // runs per `(g, k)`, so each outer row meets the groups of its
            // own `k`. Only equalities keep that reading, and a global
            // aggregate (one row even for no input) would not.
            PlanNode::Aggregate {
                input,
                grouping: Grouping::GroupBy(keys),
                aggregates,
            } if keep && !keys.is_empty() && !keys.iter().any(|k| has_outer(Node::Expr(k))) => {
                let (inner, corr) = self.pull(input, true)?;
                if !corr.iter().all(is_outer_equality) {
                    return Err(unsupported(
                        "outer references below a grouped aggregate that are not equalities",
                        plan.origin,
                    ));
                }
                let mut keys = keys.clone();
                let mut aggregates = aggregates.clone();
                // A key need not be selected: what the correlation reads is
                // grouped by and output, each only if it is not already.
                let mut grouped: HashSet<AttrId> = keys
                    .iter()
                    .filter_map(|k| match &k.node {
                        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
                        _ => None,
                    })
                    .collect();
                let mut selected: HashSet<AttrId> = output(plan)?.iter().map(|c| c.id).collect();
                for c in &corr {
                    for col in inner_columns(c) {
                        let e = col_expr(&col, Origin::UNKNOWN);
                        if grouped.insert(col.id) {
                            keys.push(Arc::clone(&e));
                        }
                        if selected.insert(col.id) {
                            aggregates.push(e);
                        }
                    }
                }
                let agg = PlanNode::Aggregate {
                    input: inner,
                    grouping: Grouping::GroupBy(keys),
                    aggregates,
                };
                Ok((node(agg, plan.origin), corr))
            }
            // EXISTS asks only whether a row exists: a positive LIMIT keeps
            // that answer (an OFFSET or `LIMIT 0` would not).
            PlanNode::Limit { input, limit }
                if !keep && positive_literal(limit) && !has_outer(Node::Expr(limit)) =>
            {
                self.pull(input, false)
            }
            // `IN (... WHERE k = o LIMIT n)` limits the rows per outer row:
            // the first `n` of each `k` by the subquery's order, so a
            // `row_number()` over `k` that keeps ranks up to `n`.
            PlanNode::Limit { input, limit }
                if keep && positive_literal(limit) && !has_outer(Node::Expr(limit)) =>
            {
                self.limited(plan, input, limit)
            }
            _ => Err(unsupported(
                "outer references below a join, aggregate, limit or set operation",
                plan.origin,
            )),
        }
    }

    /// `LIMIT limit` over `input` (a subquery read for its values) with
    /// its correlation lifted out: the rows ranked up to `limit` within
    /// each value of the correlated keys, in `input`'s order.
    fn limited(
        &mut self,
        plan: &Arc<LogicalPlan>,
        input: &Arc<LogicalPlan>,
        limit: &Arc<Expr>,
    ) -> R<(Arc<LogicalPlan>, Vec<Arc<Expr>>)> {
        let unsupported_limit =
            || unsupported("a correlated subquery's LIMIT of this form", plan.origin);
        // The analyzer puts a Project between the Limit and the Sort when
        // the sort columns are not selected: the window ranks below it.
        let ordered = |sort: &Arc<LogicalPlan>| match &sort.node {
            PlanNode::Sort { input, order, .. }
                if !order.iter().any(|o| has_outer(Node::Expr(&o.expr))) =>
            {
                Some((Arc::clone(input), order.clone()))
            }
            _ => None,
        };
        // The projection moves above the ranking filter, so a nondeterministic
        // one would see only the kept rows.
        if let PlanNode::Project { exprs, .. } = &input.node
            && !exprs.iter().all(super::sqlfn::is_deterministic)
        {
            return Err(unsupported_limit());
        }
        // The select lists and derived tables above the sort rebuild above
        // the ranking; they keep its order.
        let mut layers = Vec::new();
        let mut at = input;
        loop {
            match &at.node {
                PlanNode::Project { input, exprs }
                    if !exprs.iter().any(|e| has_outer(Node::Expr(e))) =>
                {
                    layers.push(at);
                    at = input;
                }
                PlanNode::SubqueryAlias { input, columns, .. } if columns.is_empty() => {
                    layers.push(at);
                    at = input;
                }
                _ => break,
            }
        }
        let (below, order) = match ordered(at) {
            Some((below, order)) => {
                let moves = |l: &&Arc<LogicalPlan>| match &l.node {
                    PlanNode::Project { exprs, .. } => {
                        !exprs.iter().all(super::sqlfn::is_deterministic)
                    }
                    _ => false,
                };
                if layers.iter().any(moves) {
                    return Err(unsupported_limit());
                }
                (below, order)
            }
            None => {
                layers.clear();
                (Arc::clone(input), Vec::new())
            }
        };
        // `pull` drops a sort, which is right only with no limit above it.
        if order.is_empty() && sorts(&below) {
            return Err(unsupported_limit());
        }
        let (inner, corr) = self.pull(&below, true)?;
        let mut partition_by = Vec::new();
        for c in &corr {
            let ExprNode::Resolved(ResolvedExpr::Function(f)) = &c.node else {
                return Err(unsupported_limit());
            };
            if f.id != FunctionId::Builtin(Builtin::Eq) || f.args.len() != 2 {
                return Err(unsupported_limit());
            }
            let (a, b) = (&f.args[0], &f.args[1]);
            // The window partitions on one evaluation, the join reads another.
            if !super::sqlfn::is_deterministic(a) || !super::sqlfn::is_deterministic(b) {
                return Err(unsupported_limit());
            }
            match (has_outer(Node::Expr(a)), has_outer(Node::Expr(b))) {
                (false, true) if inner_columns(b).is_empty() => partition_by.push(Arc::clone(a)),
                (true, false) if inner_columns(a).is_empty() => partition_by.push(Arc::clone(b)),
                _ => return Err(unsupported_limit()),
            }
        }
        let Some((id, kind)) = self.row_number.clone() else {
            return Err(unsupported_limit());
        };
        let out = output(&inner)?;
        let rank = BoundColumn {
            id: self.fresh(),
            name: "row_number".to_owned(),
            data_type: DataType::Integer,
            nullable: false,
        };
        let function = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id,
                kind,
                args: Vec::new(),
                distinct: false,
                filter: None,
                within_group: Vec::new(),
                data_type: DataType::Integer,
                nullable: false,
                deterministic: true,
            },
        ))));
        let window = Arc::new(Expr::new(ExprNode::Window(WindowExpr {
            function,
            window: WindowSpecRef::Spec(WindowSpec {
                partition_by: partition_by.clone(),
                order_by: order.clone(),
                frame: Some(WindowFrame::running(FrameUnits::Rows)),
            }),
        })));
        let alias = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
            expr: window,
            output: rank.clone(),
        })));
        let ranked = node(
            PlanNode::Window {
                input: inner,
                exprs: vec![alias],
                partition_by,
                order_by: order,
            },
            plan.origin,
        );
        let keep = boolean(
            Builtin::LtEq,
            vec![col_expr(&rank, Origin::UNKNOWN), Arc::clone(limit)],
        );
        let filtered = node(
            PlanNode::Filter {
                input: ranked,
                predicate: keep,
            },
            plan.origin,
        );
        let mut out_plan = columns(filtered, &out);
        for layer in layers.into_iter().rev() {
            let rebuilt = match &layer.node {
                PlanNode::Project { exprs, .. } => {
                    let mut exprs = exprs.clone();
                    let mut have: HashSet<AttrId> = output(layer)?.iter().map(|c| c.id).collect();
                    for c in &corr {
                        for col in inner_columns(c) {
                            if have.insert(col.id) {
                                exprs.push(col_expr(&col, Origin::UNKNOWN));
                            }
                        }
                    }
                    PlanNode::Project {
                        input: out_plan,
                        exprs,
                    }
                }
                PlanNode::SubqueryAlias { alias, .. } => PlanNode::SubqueryAlias {
                    input: out_plan,
                    alias: alias.clone(),
                    columns: Vec::new(),
                },
                _ => return Err(unsupported_limit()),
            };
            out_plan = node(rebuilt, layer.origin);
        }
        Ok((out_plan, corr))
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
            return self.scalar_single(base, plan);
        };
        if keys.iter().any(|k| has_outer(Node::Expr(k))) {
            return Err(correlated_reference(keys, plan.origin));
        }
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
        // Spark rejects outer references in the value
        // (`AGGREGATE_FUNCTION_MIXED_OUTER_LOCAL_REFERENCES`, or an outer
        // aggregate's `CORRELATED_REFERENCE`); the join could not bind them.
        if has_outer(Node::Expr(item)) {
            return Err(unsupported(
                "outer references in a correlated scalar subquery's value",
                item.origin,
            ));
        }
        // Grouped, the subquery has no row for an outer row without matches:
        // its value is NULL, whatever the expression (a bare `count` too).
        let empty = if keys.is_empty() {
            on_no_rows(expr).ok_or_else(|| {
                unsupported(
                    "a correlated scalar subquery whose value over no rows is not NULL or a count",
                    plan.origin,
                )
            })?
        } else {
            None
        };
        if on_path(input, set_op_or_join) {
            if !keys.is_empty() {
                return Err(unsupported(
                    "correlated scalar subqueries with GROUP BY",
                    plan.origin,
                ));
            }
            let (inner, inner_keys, outer_keys) = self.domain(&base, input)?;
            return self.scalar_join(
                base,
                plan,
                item,
                value,
                empty,
                inner,
                &inner_keys,
                &outer_keys,
                true,
            );
        }
        let (inner, corr) = self.pull(input, true)?;
        let (inner_keys, outer_keys) = equality_keys(&corr, plan.origin)?;
        // Each grouping expression is a constant or a correlated key, so
        // there is at most one group per outer row.
        let determined = |k: &Arc<Expr>| match &k.node {
            ExprNode::Literal(_) => true,
            ExprNode::Resolved(ResolvedExpr::Column(c)) => inner_keys.iter().any(
                |i| matches!(&i.node, ExprNode::Resolved(ResolvedExpr::Column(o)) if o.id == c.id),
            ),
            _ => false,
        };
        if !keys.iter().all(determined) {
            return Err(unsupported(
                "correlated scalar subqueries with GROUP BY",
                plan.origin,
            ));
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
            false,
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
        null_safe: bool,
    ) -> R<(Arc<LogicalPlan>, Arc<Expr>)> {
        // Over a union the keys are the domain of outer values, NULL
        // included, and a NULL outer value owns the NULL group (Spark joins
        // with `<=>`); a lifted `inner = outer` leaves no NULL group, and
        // the inner NULL keys must not match.
        let op = if null_safe {
            Builtin::EqNullSafe
        } else {
            Builtin::Eq
        };
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
                boolean(op, vec![renumber.inner(o), k])
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

    /// A correlated scalar subquery without an aggregate, as a `Single`
    /// join of `base` with the subquery (its correlation lifted out) on the
    /// correlated equalities: a left row matching more than one row fails
    /// with `SCALAR_SUBQUERY_TOO_MANY_ROWS`, none reads NULL.
    fn scalar_single(
        &mut self,
        base: Arc<LogicalPlan>,
        plan: &Arc<LogicalPlan>,
    ) -> R<(Arc<LogicalPlan>, Arc<Expr>)> {
        let out = output(plan)?;
        let [value] = &out[..] else {
            return Err(unsupported(
                "a correlated scalar subquery of this shape",
                plan.origin,
            ));
        };
        let value = value.clone();
        let through_domain = on_path(plan, set_op_or_join);
        if !through_domain && aggregates_below(plan) {
            return Err(unsupported(
                "a correlated scalar subquery of this shape",
                plan.origin,
            ));
        }
        let (inner, inner_keys, outer_keys) = if through_domain {
            self.domain(&base, plan)?
        } else {
            let (inner, corr) = self.pull(plan, true)?;
            let (inner_keys, outer_keys) = equality_keys(&corr, plan.origin)?;
            (inner, inner_keys, outer_keys)
        };
        let op = if through_domain {
            Builtin::EqNullSafe
        } else {
            Builtin::Eq
        };
        if inner_keys.is_empty() {
            return Err(unsupported(
                "a correlated scalar subquery of this shape",
                plan.origin,
            ));
        }
        let mut key_cols = Vec::new();
        for k in &inner_keys {
            let ExprNode::Resolved(ResolvedExpr::Column(c)) = &k.node else {
                return Err(unsupported("a correlation key of this form", k.origin));
            };
            key_cols.push(c.clone());
        }
        let (right, renumber) = self.disjoint(&base, inner)?;
        let cond = outer_keys
            .iter()
            .zip(&key_cols)
            .map(|(o, k)| {
                let k = renumber.inner(&col_expr(k, Origin::UNKNOWN));
                boolean(op, vec![renumber.inner(o), k])
            })
            .collect();
        let join = PlanNode::Join {
            left: base,
            right,
            kind: JoinKind::Single,
            condition: and_all(cond).map_or(JoinCondition::None, JoinCondition::On),
        };
        // NULL where no row matched (an outer join's right side).
        let value = BoundColumn {
            id: renumber.id(value.id),
            nullable: true,
            ..value
        };
        Ok((
            node(join, Origin::UNKNOWN),
            col_expr(&value, Origin::UNKNOWN),
        ))
    }

    /// `input` decorrelated through the domain of the outer columns it
    /// reads (Spark's `DecorrelateInnerQuery` on a set operation): each
    /// correlated part joins the distinct values of those columns and they
    /// travel up beside its output, so a set operation or a join runs once
    /// per value. Returns the new input, the domain columns in it and the
    /// outer columns they match, null-safely (a NULL outer value owns its
    /// own rows).
    fn domain(&mut self, base: &Arc<LogicalPlan>, input: &Arc<LogicalPlan>) -> R<Lifted> {
        let domain = plan_outer_columns(input);
        let visible: HashSet<AttrId> = output(base)?.iter().map(|c| c.id).collect();
        if !domain.iter().all(|c| visible.contains(&c.id)) {
            return Err(unsupported(
                "a set operation or join that reads a query further out",
                input.origin,
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
        let (inner, renumber) = self.domained(input, &distinct, &domain)?;
        let inner_keys = domain_columns(&domain, renumber)
            .iter()
            .map(|c| col_expr(c, Origin::UNKNOWN))
            .collect();
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

    /// `plan` with the domain columns beside its output and no outer
    /// reference left; the [`Renumber`] gives their ids here.
    fn domained(
        &mut self,
        plan: &Arc<LogicalPlan>,
        distinct: &Arc<LogicalPlan>,
        domain: &[BoundColumn],
    ) -> R<(Arc<LogicalPlan>, Renumber)> {
        // Below the last set operation or join, the correlation lifts as
        // it does anywhere else and meets the domain in a join's condition.
        if !on_path(plan, needs_domain) {
            let (inner, corr) = self.pull(plan, true)?;
            let (right, renumber) = self.shifted(Arc::clone(distinct))?;
            let cond = corr.iter().map(|c| renumber.outer(c)).collect();
            let join = node(
                PlanNode::Join {
                    left: inner,
                    right,
                    kind: JoinKind::Inner,
                    condition: and_all(cond).map_or(JoinCondition::None, JoinCondition::On),
                },
                Origin::UNKNOWN,
            );
            let out = with_domain(output(plan)?, domain, renumber);
            return Ok((columns(join, &out), renumber));
        }
        match &plan.node {
            PlanNode::Filter { input, predicate } => {
                if conjuncts(predicate)
                    .iter()
                    .any(|c| has_outer(Node::Expr(c)) && has_subquery(Node::Expr(c)))
                {
                    return Err(unsupported(
                        "correlated predicates with subqueries",
                        plan.origin,
                    ));
                }
                let (inner, renumber) = self.domained(input, distinct, domain)?;
                let filter = PlanNode::Filter {
                    input: inner,
                    predicate: renumber.outer(predicate),
                };
                Ok((node(filter, plan.origin), renumber))
            }
            PlanNode::Project { input, exprs } => {
                let (inner, renumber) = self.domained(input, distinct, domain)?;
                let exprs = exprs
                    .iter()
                    .map(|e| renumber.outer(e))
                    .chain(
                        domain_columns(domain, renumber)
                            .iter()
                            .map(|c| col_expr(c, Origin::UNKNOWN)),
                    )
                    .collect();
                let project = PlanNode::Project {
                    input: inner,
                    exprs,
                };
                Ok((node(project, plan.origin), renumber))
            }
            PlanNode::SubqueryAlias {
                input,
                alias,
                columns,
            } if columns.is_empty() => {
                let (inner, renumber) = self.domained(input, distinct, domain)?;
                let aliased = PlanNode::SubqueryAlias {
                    input: inner,
                    alias: alias.clone(),
                    columns: Vec::new(),
                };
                Ok((node(aliased, plan.origin), renumber))
            }
            PlanNode::Sort { input, order, .. } => {
                if order.iter().any(|o| has_outer(Node::Expr(&o.expr))) {
                    return Err(correlated_order(order, plan.origin));
                }
                // No limit above (it stops the path): the order means nothing.
                self.domained(input, distinct, domain)
            }
            PlanNode::Aggregate {
                input,
                grouping: Grouping::GroupBy(keys),
                aggregates,
            } if !keys.is_empty() => {
                if aggregates.iter().any(|e| has_outer(Node::Expr(e))) {
                    return Err(correlated_reference(aggregates, plan.origin));
                }
                if keys.iter().any(|k| has_outer(Node::Expr(k))) {
                    return Err(correlated_reference(keys, plan.origin));
                }
                let (inner, renumber) = self.domained(input, distinct, domain)?;
                let extra: Vec<Arc<Expr>> = domain_columns(domain, renumber)
                    .iter()
                    .map(|c| col_expr(c, Origin::UNKNOWN))
                    .collect();
                let agg = PlanNode::Aggregate {
                    input: inner,
                    grouping: Grouping::GroupBy(keys.iter().chain(&extra).cloned().collect()),
                    aggregates: aggregates.iter().chain(&extra).cloned().collect(),
                };
                Ok((node(agg, plan.origin), renumber))
            }
            PlanNode::ReplicateRows { input, count } => {
                let (inner, renumber) = self.domained(input, distinct, domain)?;
                let replicate = PlanNode::ReplicateRows {
                    input: inner,
                    count: Arc::clone(count),
                };
                Ok((node(replicate, plan.origin), renumber))
            }
            PlanNode::Union {
                inputs,
                by_name: false,
                ..
            } => {
                let mut branches = Vec::with_capacity(inputs.len());
                let mut first = None;
                for branch in inputs {
                    let (inner, renumber) = self.domained(branch, distinct, domain)?;
                    // A union matches by position.
                    let out = with_domain(output(branch)?, domain, renumber);
                    branches.push(columns(inner, &out));
                    first.get_or_insert(renumber);
                }
                let union = PlanNode::Union {
                    inputs: branches,
                    by_name: false,
                    allow_missing_columns: false,
                };
                Ok((node(union, plan.origin), first.unwrap_or(Renumber::NONE)))
            }
            PlanNode::Join {
                left,
                right,
                kind,
                condition,
            } => self.domained_join(plan, left, right, *kind, condition, distinct, domain),
            _ => Err(unsupported(
                "outer references below a join, aggregate, limit or set operation",
                plan.origin,
            )),
        }
    }

    /// A join on the correlation path: the domain goes under the side a
    /// correlated condition must not drop rows of, and under both sides when
    /// both read the outer query. What Spark's `CheckAnalysis` refuses for
    /// the join type is refused with its class.
    #[expect(clippy::too_many_arguments, reason = "the parts of the join")]
    fn domained_join(
        &mut self,
        plan: &Arc<LogicalPlan>,
        left: &Arc<LogicalPlan>,
        right: &Arc<LogicalPlan>,
        kind: JoinKind,
        condition: &JoinCondition,
        distinct: &Arc<LogicalPlan>,
        domain: &[BoundColumn],
    ) -> R<(Arc<LogicalPlan>, Renumber)> {
        let on = match condition {
            JoinCondition::On(on) => Some(on),
            JoinCondition::None => None,
            _ => return Err(unsupported("this join condition", plan.origin)),
        };
        let (in_left, in_right) = (has_outer(Node::Plan(left)), has_outer(Node::Plan(right)));
        // Which sides take the domain.
        let (under_left, under_right) = match kind {
            JoinKind::Inner => (in_left || !in_right, in_right),
            // `INTERSECT` / `EXCEPT` are a semi / anti join here; Spark checks
            // them as set operations, so both sides may be correlated.
            JoinKind::LeftSemi | JoinKind::LeftAnti
                if in_right && is_set_operation(left, right, on) =>
            {
                (true, true)
            }
            JoinKind::LeftOuter | JoinKind::LeftSemi | JoinKind::LeftAnti if !in_right => {
                (true, false)
            }
            JoinKind::RightOuter if !in_left => (false, true),
            JoinKind::LeftOuter
            | JoinKind::LeftSemi
            | JoinKind::LeftAnti
            | JoinKind::RightOuter
            | JoinKind::FullOuter
                if in_left || in_right =>
            {
                let side = if in_right && kind != JoinKind::RightOuter {
                    right
                } else {
                    left
                };
                return Err(AnalysisError::new(
                    UnsupportedSubqueryExpressionCategoryAccessingOuterQueryColumnIsNotAllowed {
                        tree_node: side.to_string(),
                    },
                    plan.origin,
                ));
            }
            // Spark's answer for a FULL JOIN correlated only in `ON` is not
            // the per-row join: refused rather than answered differently.
            _ => {
                return Err(unsupported(
                    "outer references in the condition of this join type",
                    plan.origin,
                ));
            }
        };
        let (l, left_domain) = if under_left {
            let (l, r) = self.domained(left, distinct, domain)?;
            (l, Some(r))
        } else {
            (Arc::clone(left), None)
        };
        let (r, right_domain) = if under_right {
            let (r, n) = self.domained(right, distinct, domain)?;
            (r, Some(n))
        } else {
            (Arc::clone(right), None)
        };
        let renumber = left_domain.or(right_domain).unwrap_or(Renumber::NONE);
        // The domain widens the left side's ids over the right side's, so
        // the right side moves (a join's sides never overlap, and re-analysing
        // must change nothing). Above the join its columns read their old ids.
        let right_ids: HashSet<AttrId> = output(&r)?.iter().map(|c| c.id).collect();
        let (r, moved) = self.disjoint(&l, r)?;
        let moved_column = |c: &BoundColumn| BoundColumn {
            id: if right_ids.contains(&c.id) {
                moved.id(c.id)
            } else {
                c.id
            },
            ..c.clone()
        };
        let mut cond: Vec<Arc<Expr>> = on.iter().map(|c| renumber.outer(c)).collect();
        if let (Some(a), Some(b)) = (left_domain, right_domain) {
            let pairs = domain_columns(domain, a)
                .into_iter()
                .zip(domain_columns(domain, b));
            for (x, y) in pairs {
                let args = vec![col_expr(&x, Origin::UNKNOWN), col_expr(&y, Origin::UNKNOWN)];
                cond.push(boolean(Builtin::EqNullSafe, args));
            }
        }
        let cond = cond
            .into_iter()
            .map(|c| {
                c.transform_up(&mut |x: Arc<Expr>| {
                    Ok::<_, std::convert::Infallible>(match &x.node {
                        ExprNode::Resolved(ResolvedExpr::Column(col)) => {
                            col_expr(&moved_column(col), x.origin)
                        }
                        _ => x,
                    })
                })
                .unwrap_or_else(|never| match never {})
            })
            .collect();
        let join = node(
            PlanNode::Join {
                left: l,
                right: r,
                kind,
                condition: and_all(cond).map_or(JoinCondition::None, JoinCondition::On),
            },
            plan.origin,
        );
        let exprs = with_domain(output(plan)?, domain, renumber)
            .iter()
            .map(|c| {
                let read = col_expr(&moved_column(c), Origin::UNKNOWN);
                if moved_column(c).id == c.id {
                    read
                } else {
                    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                        expr: read,
                        output: c.clone(),
                    })))
                }
            })
            .collect();
        let project = PlanNode::Project { input: join, exprs };
        Ok((node(project, plan.origin), renumber))
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

/// The attribute ids of the enclosing query's columns the subqueries in `e`
/// read.
fn outer_ids(e: &Arc<Expr>) -> HashSet<AttrId> {
    let mut ids = HashSet::new();
    let _ = visit_distinct(Node::Expr(e), |n| {
        if let Node::Expr(x) = n
            && let ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) = &x.node
        {
            ids.insert(c.id);
        }
        ControlFlow::<()>::Continue(())
    });
    ids
}

/// Whether a node holds what only a domain join can decorrelate: a set
/// operation (as the analyzer rewrites them) or a join.
fn set_op_or_join(p: &LogicalPlan) -> bool {
    matches!(
        p.node,
        PlanNode::Union { .. } | PlanNode::ReplicateRows { .. } | PlanNode::Join { .. }
    )
}

/// [`set_op_or_join`], or a select list that reads the outer query: below a
/// domain it reads the domain's columns, and `pull` cannot lift it.
fn needs_domain(p: &LogicalPlan) -> bool {
    set_op_or_join(p)
        || matches!(&p.node, PlanNode::Project { exprs, .. }
            if exprs.iter().any(|e| has_outer(Node::Expr(e))))
}

/// Whether `hit` holds on the correlation path from `p` down, through what
/// [`Rewriter::domained`] goes through. A global aggregate ends the path: it
/// has one row for an outer row without matches, which a domain would lose.
fn on_path(p: &Arc<LogicalPlan>, hit: fn(&LogicalPlan) -> bool) -> bool {
    if !has_outer(Node::Plan(p)) {
        return false;
    }
    if hit(p) {
        return true;
    }
    match &p.node {
        PlanNode::Project { input, .. }
        | PlanNode::Filter { input, .. }
        | PlanNode::Sort { input, .. } => on_path(input, hit),
        PlanNode::SubqueryAlias { input, columns, .. } if columns.is_empty() => on_path(input, hit),
        PlanNode::Aggregate {
            input,
            grouping: Grouping::GroupBy(keys),
            ..
        } if !keys.is_empty() => on_path(input, hit),
        _ => false,
    }
}

/// Whether a semi or anti join is what the analyzer makes of `INTERSECT` or
/// `EXCEPT`: the null-safe equality of every column, pairwise.
fn is_set_operation(
    left: &Arc<LogicalPlan>,
    right: &Arc<LogicalPlan>,
    on: Option<&Arc<Expr>>,
) -> bool {
    let (Some(on), Some(l), Some(r)) = (on, left.resolved_output(), right.resolved_output()) else {
        return false;
    };
    let parts = conjuncts(on);
    parts.len() == l.len()
        && l.len() == r.len()
        && parts.iter().zip(l.iter().zip(&r)).all(|(c, (a, b))| {
            let ExprNode::Resolved(ResolvedExpr::Function(f)) = &c.node else {
                return false;
            };
            f.id == FunctionId::Builtin(Builtin::EqNullSafe)
                && matches!(&f.args[..], [x, y]
                    if column_id(x) == Some(a.id) && column_id(y) == Some(b.id))
        })
}

fn column_id(e: &Expr) -> Option<AttrId> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
        _ => None,
    }
}

/// The domain's columns as `renumber` placed them.
fn domain_columns(domain: &[BoundColumn], renumber: Renumber) -> Vec<BoundColumn> {
    domain
        .iter()
        .map(|c| BoundColumn {
            id: renumber.id(c.id),
            ..c.clone()
        })
        .collect()
}

/// `out` and the domain's columns after it.
fn with_domain(
    mut out: Vec<BoundColumn>,
    domain: &[BoundColumn],
    renumber: Renumber,
) -> Vec<BoundColumn> {
    out.extend(domain_columns(domain, renumber));
    out
}

/// The outer columns `plan` reads, in the order they are met.
fn plan_outer_columns(plan: &Arc<LogicalPlan>) -> Vec<BoundColumn> {
    let mut out: Vec<BoundColumn> = Vec::new();
    let _ = visit_distinct(Node::Plan(plan), |n| {
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
    if let Some(v) = integer_on_no_rows(e) {
        return Some(v.map(Literal::Long));
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

/// `e` over no rows when it is integer arithmetic over `count`s (0), other
/// aggregates (NULL) and integer literals: `Some(None)` is NULL.
fn integer_on_no_rows(e: &Arc<Expr>) -> Option<Option<i64>> {
    match &e.node {
        ExprNode::Literal(Literal::Long(v)) => Some(Some(*v)),
        ExprNode::Literal(Literal::Integer(v)) => Some(Some(i64::from(*v))),
        ExprNode::Cast {
            expr,
            to: DataType::Long,
            ..
        } => integer_on_no_rows(expr),
        ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate => {
            match f.id.name() {
                "count" => Some(Some(0)),
                "sum" | "avg" | "min" | "max" | "any_value" => Some(None),
                _ => None,
            }
        }
        ExprNode::Resolved(ResolvedExpr::Function(f)) if f.args.len() == 2 => {
            let op: fn(i64, i64) -> Option<i64> = match f.id {
                FunctionId::Builtin(Builtin::Add) => i64::checked_add,
                FunctionId::Builtin(Builtin::Subtract) => i64::checked_sub,
                FunctionId::Builtin(Builtin::Multiply) => i64::checked_mul,
                _ => return None,
            };
            let a = integer_on_no_rows(&f.args[0])?;
            let b = integer_on_no_rows(&f.args[1])?;
            match (a, b) {
                (Some(a), Some(b)) => op(a, b).map(Some),
                _ => Some(None),
            }
        }
        _ => None,
    }
}

/// Each correlated predicate is `inner = outer`: the inner and the outer
/// sides, pairwise.
fn equality_keys(corr: &[Arc<Expr>], origin: Origin) -> R<Keys> {
    let mut inner_keys = Vec::new();
    let mut outer_keys = Vec::new();
    for c in corr {
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
                origin,
            ));
        };
        inner_keys.push(Arc::clone(i));
        outer_keys.push(Arc::clone(o));
    }
    Ok((inner_keys, outer_keys))
}

/// Spark's refusal of a correlated reference where it cannot be lifted, naming
/// those of `exprs` that hold one.
fn correlated_reference(exprs: &[Arc<Expr>], origin: Origin) -> AnalysisError {
    let outer: Vec<String> = exprs
        .iter()
        .filter(|e| has_outer(Node::Expr(e)))
        .map(|e| format!("\"{}\"", pretty(e, None)))
        .collect();
    AnalysisError::new(
        UnsupportedSubqueryExpressionCategoryCorrelatedReference {
            sql_exprs: outer.join(", "),
        },
        origin,
    )
}

/// Spark's refusal of an order that reads the outer query.
fn correlated_order(order: &[SortOrder], origin: Origin) -> AnalysisError {
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
    AnalysisError::new(
        UnsupportedSubqueryExpressionCategoryCorrelatedReference {
            sql_exprs: outer.join(", "),
        },
        origin,
    )
}

/// Whether a sort lies on the path from `plan` down that keeps rows in order.
fn sorts(plan: &Arc<LogicalPlan>) -> bool {
    match &plan.node {
        PlanNode::Sort { .. } => true,
        PlanNode::Project { input, .. }
        | PlanNode::Filter { input, .. }
        | PlanNode::SubqueryAlias { input, .. } => sorts(input),
        _ => false,
    }
}
