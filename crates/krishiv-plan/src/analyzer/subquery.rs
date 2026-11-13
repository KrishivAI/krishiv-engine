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
use super::exprs::{builtin_name, literal_type, pretty};
use super::ops::{MAX_ID, col_expr};
use super::subquery_shapes::{
    Bounded, aggregates_below, bounded, check_in, constant_side, fold_one_row_bounds,
    in_join_condition_ins, outer_ids, three_valued_ins,
};
pub(super) use super::subquery_shapes::{null_literal, one_row_aggregate};
use super::{AnalysisError, ResolutionConf, SparkProfile};
use crate::walk::{Node, visit_distinct};
use crate::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FrameUnits, FunctionId, FunctionKind, Grouping,
    JoinCondition, JoinKind, Literal, LogicalPlan, Origin, PlanNode, ResolvedExpr,
    ResolvedFunction, SortDirection, SortOrder, SubqueryKind, WindowExpr, WindowFrame, WindowSpec,
    WindowSpecRef,
};
use krishiv_common::stack::{STEP, on_big_stack};

#[path = "subquery_rank.rs"]
mod rank;
use rank::{outer_only, repartitioned, window_reads_outer};

pub(super) type R<T> = Result<T, AnalysisError>;

#[path = "subquery_split.rs"]
mod split;

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
    conf: &ResolutionConf,
) -> R<Arc<LogicalPlan>> {
    if !has_subquery(Node::Plan(root)) && !has_outer(Node::Plan(root)) {
        return Ok(Arc::clone(root));
    }
    let mut r = Rewriter {
        next_id: *next_id,
        done: HashMap::new(),
        correlated: HashMap::new(),
        depth: 0,
        lateral: false,
        legacy_exists_in_count: conf.legacy_exists_in_count,
        legacy_null_in_empty: conf.legacy_null_in_empty_list,
        in_join_as_join: conf.in_join_condition_as_join,
        legacy_scalar_count: conf.legacy_scalar_count,
        nested_count_fix: conf.nested_count_fix(),
        single_join: conf.scalar_single_join && profile != SparkProfile::V3_5,
        legacy_group_by: conf.legacy_group_by_non_equality,
        decorrelate_limit: conf.decorrelate_limit && profile != SparkProfile::V3_5,
        decorrelate_offset: conf.decorrelate_offset && profile != SparkProfile::V3_5,
        decorrelate_window: profile != SparkProfile::V3_5,
        row_number: match functions.lookup(&["row_number".to_owned()], profile) {
            FunctionLookup::Found(entry) => Some((entry.id, entry.kind)),
            _ => None,
        },
    };
    let out = r.plan(root)?;
    *next_id = r.next_id;
    Ok(out)
}

pub(super) struct Rewriter {
    next_id: u64,
    /// Each relation rewritten, by address (the original kept alive), so a
    /// relation shared by several parents or subqueries is rewritten once.
    done: HashMap<usize, (Arc<LogicalPlan>, Arc<LogicalPlan>)>,
    /// Whether a subquery plan refers to its enclosing query, by address.
    correlated: HashMap<usize, (Arc<LogicalPlan>, bool)>,
    depth: usize,
    /// The registry's `row_number`, for a subquery's per-row `LIMIT`.
    row_number: Option<(FunctionId, FunctionKind)>,
    /// Decorrelating a lateral join's right side: a global aggregate on the
    /// way keeps its one row per domain value.
    lateral: bool,
    /// `spark.sql.optimizer.decorrelateExistsSubqueryLegacyIncorrectCountHandling.enabled`.
    legacy_exists_in_count: bool,
    /// `spark.sql.legacy.nullInEmptyListBehavior`.
    legacy_null_in_empty: bool,
    in_join_as_join: bool,
    /// `spark.sql.optimizer.decorrelateSubqueryLegacyIncorrectCountHandling.enabled`.
    legacy_scalar_count: bool,
    nested_count_fix: bool,
    /// `spark.sql.optimizer.scalarSubqueryUseSingleJoin` (always off for 3.5).
    single_join: bool,
    /// `spark.sql.legacy.scalarSubqueryAllowGroupByNonEqualityCorrelatedPredicate`.
    legacy_group_by: bool,
    /// `spark.sql.optimizer.decorrelateLimit.enabled` (always off for 3.5,
    /// which has no correlation below a `LIMIT`).
    decorrelate_limit: bool,
    /// `spark.sql.optimizer.decorrelateOffset.enabled` (likewise).
    decorrelate_offset: bool,
    /// Whether a correlation may sit below a window: not on Spark 3.5, whose
    /// `CheckAnalysis.checkPlan` has no case for `Window` and so forbids outer
    /// references below it, as below a `LIMIT`.
    decorrelate_window: bool,
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
pub(super) fn has_outer(node: Node<'_>) -> bool {
    visit_distinct(node, |n| match n {
        Node::Expr(e) if matches!(e.node, ExprNode::Resolved(ResolvedExpr::OuterColumn(_))) => {
            ControlFlow::Break(())
        }
        _ => ControlFlow::Continue(()),
    })
    .is_break()
}

/// Whether `e` reads an outer attribute that is not in `defined`.
fn escapes(e: &Arc<Expr>, defined: &HashSet<AttrId>) -> bool {
    visit_distinct(Node::Expr(e), |n| match n {
        Node::Expr(x)
            if matches!(&x.node, ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) if !defined.contains(&c.id)) =>
        {
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
pub(super) fn subqueries(e: &Arc<Expr>) -> Vec<Arc<Expr>> {
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
pub(super) fn conjuncts(e: &Arc<Expr>) -> Vec<Arc<Expr>> {
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

/// `e`'s builtin operator and arguments, if it is a builtin call.
fn builtin_args(e: &Expr) -> Option<(Builtin, &[Arc<Expr>])> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) => match f.id {
            FunctionId::Builtin(b) => Some((b, f.args.as_slice())),
            _ => None,
        },
        _ => None,
    }
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

/// `inner_column = outer_expression`, either way round.
pub(super) fn is_outer_equality(e: &Arc<Expr>) -> bool {
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

pub(super) fn boolean(op: Builtin, args: Vec<Arc<Expr>>) -> Arc<Expr> {
    let nullable = op != Builtin::IsNull && args.iter().any(|a| typed(a).1);
    call(op, args, DataType::Boolean, nullable)
}

/// The conjunction of `parts` (at least one).
pub(super) fn and_all(parts: Vec<Arc<Expr>>) -> Option<Arc<Expr>> {
    parts
        .into_iter()
        .reduce(|a, b| boolean(Builtin::And, vec![a, b]))
}

pub(super) fn node(node: PlanNode, origin: Origin) -> Arc<LogicalPlan> {
    Arc::new(LogicalPlan::new(node).with_origin(origin))
}

/// A projection of `out` over `input`.
fn columns(input: Arc<LogicalPlan>, out: &[BoundColumn]) -> Arc<LogicalPlan> {
    let exprs = out.iter().map(|c| col_expr(c, Origin::UNKNOWN)).collect();
    node(PlanNode::Project { input, exprs }, Origin::UNKNOWN)
}

/// The output of a resolved relation the rewrite builds on.
pub(super) fn output(p: &LogicalPlan) -> R<Vec<BoundColumn>> {
    p.resolved_output()
        .ok_or_else(|| unsupported("a subquery over an unresolved plan", p.origin))
}

/// The span of attribute ids `p`'s relations define, as Resolve counts
/// them for a join's sides (a shift counts at its offset; subquery plans
/// inside expressions do not count).
pub(super) fn hull(p: &Arc<LogicalPlan>) -> Option<(u64, u64)> {
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
    pub(super) fn correlated(&mut self, e: &Expr) -> bool {
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
            PlanNode::Aggregate {
                input,
                grouping: Grouping::GroupBy(keys),
                aggregates,
            } => self.aggregate(&p, Arc::clone(input), keys, aggregates),
            PlanNode::Join {
                left,
                right,
                kind,
                condition: JoinCondition::On(on),
            } if self.joins_here(on, kind, left, right) => self.join(&p, left, right, kind, on),
            PlanNode::Join {
                left,
                right,
                kind: kind @ (JoinKind::Inner | JoinKind::LeftOuter),
                condition,
            } if reads_left(left, right) => self.lateral(&p, left, right, *kind, condition),
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
                // After the inlining: Spark checks the plan with its CTEs
                // inlined, so a bound behind a `WITH` reference counts.
                let inlined = super::subquery_cte::inline(plan, &mut self.next_id)?;
                self.check_bounds(&inlined)?;
                let mut rewritten = self.plan(&inlined)?;
                if self.legacy_exists_in_count
                    && matches!(kind, SubqueryKind::Exists | SubqueryKind::In(_))
                {
                    rewritten = super::subquery_legacy::legacy_shape(&rewritten, x.origin)?;
                }
                rewritten = fold_one_row_bounds(&rewritten);
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

    /// Spark's check of a `LIMIT` / `OFFSET` on a correlation path when its
    /// decorrelation is off (`spark.sql.optimizer.decorrelateLimit.enabled`,
    /// `decorrelateOffset.enabled`, and always on Spark 3.5): the first
    /// operator below it that reads the outer query is refused
    /// (`ACCESSING_OUTER_QUERY_COLUMN_IS_NOT_ALLOWED`).
    fn check_bounds(&self, plan: &Arc<LogicalPlan>) -> R<()> {
        if self.decorrelate_limit && self.decorrelate_offset && self.decorrelate_window {
            return Ok(());
        }
        let mut seen = HashSet::new();
        let mut stack = vec![Arc::clone(plan)];
        while let Some(p) = stack.pop() {
            if !seen.insert(addr(&*p)) {
                continue;
            }
            let below = match &p.node {
                PlanNode::Limit { input, .. } if !self.decorrelate_limit => Some(input),
                PlanNode::Offset { input, .. } if !self.decorrelate_offset => Some(input),
                PlanNode::Window { input, .. } if !self.decorrelate_window => Some(input),
                _ => None,
            };
            if let Some(input) = below
                && let Some(host) = outer_host(input)
            {
                return Err(AnalysisError::new(
                    UnsupportedSubqueryExpressionCategoryAccessingOuterQueryColumnIsNotAllowed {
                        tree_node: host.to_string(),
                    },
                    host.origin,
                ));
            }
            p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
        }
        Ok(())
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
        if let Some(e) = self.in_composite(value, plan) {
            return Some(if negated {
                boolean(Builtin::Not, vec![e])
            } else {
                e
            });
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
        for c in conjuncts(&split::push_not(predicate)) {
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

    /// A join whose `ON` holds `EXISTS` / `IN` subqueries: each is an existence join under
    /// the side it reads (Spark's `ExistenceJoin`); one reading both sides stays unsupported.
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
        let on = &three_valued_ins(on, self.legacy_null_in_empty)?;
        let flags = (self.in_join_as_join, self.legacy_null_in_empty);
        let on = &in_join_condition_ins(on, *kind, [&left_ids, &right_ids], flags, &mut || {
            self.fresh()
        })?;
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
                on_left.push(constant_side(c, left)?);
            } else if ids.iter().all(|i| right_ids.contains(i)) {
                on_right.push(constant_side(c, right)?);
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
        let (plain, l, r) = self.lift_sides(plain, kind, l, r, [&left_ids, &right_ids])?;
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

    /// `left JOIN LATERAL right`, `right` reading `left`'s columns: an
    /// ordinary join of the two on `right`'s correlated predicates (lifted
    /// out as for an `IN` subquery) and the join's own condition. The
    /// output is the join's, as it was.
    fn lateral(
        &mut self,
        p: &Arc<LogicalPlan>,
        left: &Arc<LogicalPlan>,
        right: &Arc<LogicalPlan>,
        kind: JoinKind,
        condition: &JoinCondition,
    ) -> R<Arc<LogicalPlan>> {
        self.check_bounds(right)?;
        let out = output(p)?;
        // A right side that is a select list over no relation is one row per
        // left row: the expressions read over the left side's own columns. No
        // domain, so a nondeterministic left side is evaluated once.
        if (matches!(kind, JoinKind::Inner) || matches!(condition, JoinCondition::None))
            && let Some(row) = one_row_select(right)
            && !row.iter().any(|e| has_subquery(Node::Expr(e)))
        {
            let mut exprs: Vec<Arc<Expr>> = output(left)?
                .iter()
                .map(|c| col_expr(c, Origin::UNKNOWN))
                .collect();
            exprs.extend(row.iter().map(|e| Renumber::NONE.inner(e)));
            let mut plan = node(
                PlanNode::Project {
                    input: Arc::clone(left),
                    exprs,
                },
                p.origin,
            );
            if let JoinCondition::On(on) = condition {
                plan = node(
                    PlanNode::Filter {
                        input: plan,
                        predicate: Arc::clone(on),
                    },
                    p.origin,
                );
            }
            return Ok(plan);
        }
        if !plan_is_deterministic(left) && on_path_through(right, lateral_hit, true) {
            return Err(unsupported(
                "a lateral join over a nondeterministic input that its right side correlates with",
                p.origin,
            ));
        }
        let (inner, corr) = if on_path_through(right, lateral_hit, true) {
            self.lateral = true;
            let domained = self.domain(left, right);
            self.lateral = false;
            let (inner, inner_keys, outer_keys) = domained?;
            let corr = inner_keys
                .into_iter()
                .zip(outer_keys)
                .map(|(i, o)| boolean(Builtin::EqNullSafe, vec![i, o]))
                .collect();
            (inner, corr)
        } else {
            self.pull(right, true)?
        };
        let (inner, renumber) = self.disjoint(left, inner)?;
        let mut cond: Vec<Arc<Expr>> = corr.iter().map(|c| renumber.inner(c)).collect();
        if let JoinCondition::On(on) = condition {
            // The condition reads the right side's columns as renumbered.
            cond.push(renumber.inner(on));
        }
        let joined = node(
            PlanNode::Join {
                left: Arc::clone(left),
                right: inner,
                kind,
                condition: and_all(cond).map_or(JoinCondition::None, JoinCondition::On),
            },
            p.origin,
        );
        // The right side's columns come back under the attributes the plan
        // above reads them as, where the join renumbered them.
        let exprs = out
            .iter()
            .map(|c| {
                let id = renumber.id(c.id);
                if id == c.id {
                    col_expr(c, Origin::UNKNOWN)
                } else {
                    let read = BoundColumn { id, ..c.clone() };
                    alias(col_expr(&read, Origin::UNKNOWN), c)
                }
            })
            .collect();
        Ok(node(
            PlanNode::Project {
                input: joined,
                exprs,
            },
            Origin::UNKNOWN,
        ))
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

    /// An aggregate whose keys or calls (or their `FILTER`s) read a
    /// correlated subquery: the subquery joins to the input below the
    /// aggregate, which reads the join's column (Spark pushes the
    /// expression into a projection under the aggregate).
    fn aggregate(
        &mut self,
        p: &Arc<LogicalPlan>,
        input: Arc<LogicalPlan>,
        keys: &[Arc<Expr>],
        aggregates: &[Arc<Expr>],
    ) -> R<Arc<LogicalPlan>> {
        let aggregates = &split::share_keys(keys, aggregates);
        let all: Vec<Arc<Expr>> = keys.iter().chain(aggregates).cloned().collect();
        if !all
            .iter()
            .any(|e| self.reads_correlated(e) || self.existential(e))
        {
            return Ok(Arc::clone(p));
        }
        // Spark rejects a correlated scalar subquery here
        // (`MUST_AGGREGATE_CORRELATED_SCALAR_SUBQUERY` for a key).
        if all.iter().any(|e| {
            subqueries(e).iter().any(|s| {
                matches!(
                    s.node,
                    ExprNode::Subquery {
                        kind: SubqueryKind::Scalar,
                        ..
                    }
                ) && self.correlated(s)
            })
        }) {
            return Err(unsupported(
                "correlated subqueries outside a filter or a projection",
                p.origin,
            ));
        }
        let before = output(&input)?;
        split::check_grouped_reads(keys, aggregates, &before, p.origin)?;
        if keys.is_empty()
            && let Some(out) = self.global_with_subqueries(p, &input, aggregates)?
        {
            return Ok(out);
        }
        let (base, all) = self.existence(input, all)?;
        let (base, mut all) = self.scalars(base, all)?;
        let aggregates = all.split_off(keys.len());
        Ok(node(
            PlanNode::Aggregate {
                input: base,
                grouping: Grouping::GroupBy(split::group_joined(all, &aggregates, &before)),
                aggregates,
            },
            p.origin,
        ))
    }

    /// Whether `e` holds an `EXISTS` or `IN` subquery that an existence
    /// join computes: a correlated `EXISTS`, or any `IN`. (An uncorrelated
    /// `EXISTS` stays an expression computed once.)
    pub(super) fn existential(&mut self, e: &Arc<Expr>) -> bool {
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
        let exprs: Vec<Arc<Expr>> = exprs.into_iter().map(|e| split::push_not(&e)).collect();
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
            if let ExprNode::Subquery { kind, .. } = &sub.node {
                let wanted = match kind {
                    SubqueryKind::Exists => self.correlated(&sub),
                    SubqueryKind::In(values) => {
                        check_in(values, sub.origin)?;
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
        // EXISTS asks only whether a row exists: a positive LIMIT is pulled
        // away, not ranked through the domain.
        let hit: fn(&LogicalPlan) -> bool = if keep { split::limit_hit } else { exists_hit };
        let (inner, corr) = if on_path(&pred.plan, hit) {
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
            // Spark's Category 2 (`checkCorrelationsInSubquery`): allowed on
            // the path, its expressions free of outer references (the
            // analyzer refused those). Unlike a sort it stays: Spark's
            // shuffle still evaluates its expressions, over the rows below
            // the lifted correlation.
            PlanNode::Repartition { input, exprs } => {
                let (inner, corr) = self.pull(input, true)?;
                let node_ = PlanNode::Repartition {
                    input: inner,
                    exprs: exprs.clone(),
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
                if !corr.iter().all(|c| is_outer_equality(c) || outer_only(c)) {
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
            // A window never adds or drops rows: EXISTS reads through it.
            // For its values, the window runs per correlation value too
            // (Spark's `DecorrelateInnerQuery` adds the correlated inner
            // columns to its partitioning), which only equalities on a bare
            // inner column allow.
            PlanNode::Window {
                input,
                exprs,
                partition_by,
                order_by,
            } if !window_reads_outer(exprs, partition_by, order_by) => {
                if !keep {
                    return self.pull(input, false);
                }
                let (inner, corr) = self.pull(input, true)?;
                let mut extra: Vec<BoundColumn> = Vec::new();
                for c in &corr {
                    if inner_columns(c).is_empty() {
                        continue;
                    }
                    if !is_outer_equality(c) {
                        return Err(unsupported(
                            "outer references below a window that are not equalities",
                            plan.origin,
                        ));
                    }
                    for col in inner_columns(c) {
                        if !extra.iter().any(|e| e.id == col.id) {
                            extra.push(col);
                        }
                    }
                }
                let window = repartitioned(plan, inner, exprs, partition_by, order_by, &extra)?;
                Ok((window, corr))
            }
            PlanNode::Limit { .. } | PlanNode::Offset { .. } if bounded(plan).is_some() => {
                let Some(b) = bounded(plan) else {
                    return Err(unsupported(
                        "outer references below a join, aggregate, limit or set operation",
                        plan.origin,
                    ));
                };
                // EXISTS asks only whether a row exists: a positive LIMIT
                // keeps that answer (an OFFSET or `LIMIT 0` would not).
                if !keep && b.offset == 0 && b.limit.is_some_and(|l| l > 0) {
                    return self.pull(&b.below, false);
                }
                // `IN (... WHERE k = o LIMIT n OFFSET m)` limits the rows per
                // outer row: ranks `m + 1` to `m + n` of each `k` by the
                // subquery's order, a `row_number()` over `k`.
                self.limited(plan, &b)
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

    pub(super) fn fresh(&mut self) -> AttrId {
        self.next_id += 1;
        AttrId(self.next_id - 1)
    }
}

mod support;
pub(in crate::analyzer) use support::*;

mod domain;

mod scalar;
