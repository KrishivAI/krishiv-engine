// SPDX-License-Identifier: Apache-2.0
//! Correlated subqueries over a global aggregate whose value is not one
//! aggregate call: an expression of several calls (`count(*) is null`), a
//! `HAVING` that reads another call than the value (`SELECT count(a) ...
//! HAVING count(*) > -1`), or `v IN` of such a subquery. Spark evaluates
//! them through `RewriteCorrelatedScalarSubquery`'s count-bug handling (its
//! `evalSubqueryOnZeroTups`, `splitSubquery`): over no matching rows each
//! call takes its value over no rows (a count's 0, NULL for the others), and
//! the expression and the `HAVING` are computed from those.
//!
//! Here each distinct call becomes a one-call scalar subquery, which the
//! scalar path decorrelates with its count-bug handling, and the value and
//! the `HAVING` are rebuilt over their reads: `CASE WHEN having THEN value
//! END` (a rejected row is no row: NULL), and for `v IN`, `CASE WHEN having
//! THEN v = value ELSE false END` (no row is not a match; `NOT IN` is its
//! negation, true over no row).
//!
//! It also holds the grouped correlated scalar subqueries whose grouping
//! the correlation does not fix to one group per outer row (Spark's
//! `nonEquivalentGroupbyCols`): Spark 4.x runs them as a single join, which
//! fails with `SCALAR_SUBQUERY_TOO_MANY_ROWS` at run time on a second row;
//! with `spark.sql.optimizer.scalarSubqueryUseSingleJoin` off (and on Spark
//! 3.5) analysis refuses them with `NON_CORRELATED_COLUMNS_IN_GROUP_BY`,
//! unless the legacy flag
//! `spark.sql.legacy.scalarSubqueryAllowGroupByNonEqualityCorrelatedPredicate`
//! lets them run as an outer join.

use std::collections::HashMap;
use std::sync::Arc;

use super::{
    R, Rewriter, boolean, call, conjuncts, has_outer, has_subquery, inner_columns, node,
    on_no_rows, typed,
};
use crate::analyzer::AnalysisError;
use crate::analyzer::exprs::pretty;
use crate::walk::Node;
use crate::{
    BoundColumn, Builtin, Expr, ExprNode, FunctionKind, Grouping, JoinKind, Literal, LogicalPlan,
    Origin, PlanNode, ResolvedExpr, SubqueryKind,
};
use krishiv_common::error::{
    UnsupportedSubqueryExpressionCategoryMustAggregateCorrelatedScalarSubquery,
    UnsupportedSubqueryExpressionCategoryNonCorrelatedColumnsInGroupBy,
};
use krishiv_types::DataType;

/// A correlated global aggregate's value and `HAVING`, both over its
/// aggregate calls.
pub(super) struct Composite {
    /// The aggregate's input, correlation included.
    input: Arc<LogicalPlan>,
    origin: Origin,
    having: Option<Arc<Expr>>,
    value: Arc<Expr>,
}

/// `plan` as a [`Composite`]: `[Project [e]] [Filter h] Aggregate` with no
/// grouping keys, when the one-call scalar path does not take it (several
/// calls, a projection or `HAVING` over them, or a value whose result over
/// no rows it cannot tell). `None` for every other plan.
pub(super) fn composite(plan: &Arc<LogicalPlan>) -> Option<Composite> {
    let (project, agg) = match &plan.node {
        PlanNode::Project { input, exprs } => match &exprs[..] {
            [e] => (Some(e), input),
            _ => return None,
        },
        _ => (None, plan),
    };
    let (having, agg) = match &agg.node {
        PlanNode::Filter { input, predicate } => (Some(predicate), input),
        _ => (None, agg),
    };
    let PlanNode::Aggregate {
        input,
        grouping: Grouping::GroupBy(keys),
        aggregates,
    } = &agg.node
    else {
        return None;
    };
    if !keys.is_empty() || !has_outer(Node::Plan(input)) {
        return None;
    }
    let mut items: HashMap<crate::AttrId, Arc<Expr>> = HashMap::new();
    for item in aggregates {
        let ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) = &item.node else {
            return None;
        };
        items.insert(output.id, Arc::clone(expr));
    }
    let value = match project {
        Some(e) => inline(unalias(e), &items)?,
        None => match &aggregates[..] {
            [_] => items.values().next().cloned()?,
            _ => return None,
        },
    };
    let having = match having {
        Some(h) => Some(inline(h, &items)?),
        None => None,
    };
    let one_call =
        having.is_none() && project.is_none() && (is_call(&value) || on_no_rows(&value).is_some());
    let outer =
        has_outer(Node::Expr(&value)) || having.as_ref().is_some_and(|h| has_outer(Node::Expr(h)));
    // With no call anywhere the subquery does not read its aggregate.
    let no_call = calls(&value).is_empty() && having.as_ref().is_none_or(|h| calls(h).is_empty());
    if one_call || outer || no_call {
        return None;
    }
    Some(Composite {
        input: Arc::clone(input),
        origin: agg.origin,
        having,
        value,
    })
}

fn unalias(e: &Arc<Expr>) -> &Arc<Expr> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => expr,
        _ => e,
    }
}

/// `e` with each reference to an aggregate's output replaced by the
/// output's expression; `None` if it reads another column.
fn inline(e: &Arc<Expr>, items: &HashMap<crate::AttrId, Arc<Expr>>) -> Option<Arc<Expr>> {
    let mut unknown = false;
    let out = Arc::clone(e).transform_up(&mut |x: Arc<Expr>| {
        Ok::<_, std::convert::Infallible>(match &x.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => match items.get(&c.id) {
                Some(def) => Arc::clone(def),
                None => {
                    unknown = true;
                    x
                }
            },
            _ => x,
        })
    });
    let out = out.unwrap_or_else(|never| match never {});
    (!unknown).then_some(out)
}

fn is_call(e: &Expr) -> bool {
    matches!(
        &e.node,
        ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate
    )
}

/// The distinct aggregate calls in `e` (not inside one another), in order.
fn calls(e: &Arc<Expr>) -> Vec<Arc<Expr>> {
    let mut out: Vec<Arc<Expr>> = Vec::new();
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        if is_call(&x) {
            if !out.iter().any(|o| **o == *x) {
                out.push(x);
            }
            continue;
        }
        let mut kids = Vec::new();
        x.for_each_child(&mut |k| kids.push(Arc::clone(k)));
        stack.extend(kids.into_iter().rev());
    }
    out
}

/// `e` with each call of `reads` replaced by its read.
fn replace(e: &Arc<Expr>, reads: &[(Arc<Expr>, Arc<Expr>)]) -> Arc<Expr> {
    let out = Arc::clone(e).transform_down(&mut |x: Arc<Expr>| {
        Ok::<_, std::convert::Infallible>(
            reads
                .iter()
                .find(|(c, _)| **c == *x)
                .map_or(x, |(_, read)| Arc::clone(read)),
        )
    });
    out.unwrap_or_else(|never| match never {})
}

impl Rewriter {
    /// A one-call global aggregate over `c`'s input, as a scalar subquery
    /// plan.
    fn one_call(&mut self, c: &Composite, expr: Arc<Expr>) -> R<Arc<LogicalPlan>> {
        let (data_type, nullable) = typed(&expr);
        let data_type = data_type.ok_or_else(|| {
            super::unsupported("a correlated scalar subquery of this shape", c.origin)
        })?;
        let output = BoundColumn {
            id: self.fresh(),
            name: pretty(&expr, None),
            data_type,
            nullable,
        };
        let item = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
            expr,
            output,
        })));
        Ok(node(
            PlanNode::Aggregate {
                input: Arc::clone(&c.input),
                grouping: Grouping::GroupBy(Vec::new()),
                aggregates: vec![item],
            },
            c.origin,
        ))
    }

    /// `c` as a left outer join per aggregate call and the expression that
    /// reads its value per outer row.
    pub(super) fn scalar_composite(
        &mut self,
        base: Arc<LogicalPlan>,
        c: &Composite,
    ) -> R<(Arc<LogicalPlan>, Arc<Expr>)> {
        let mut wanted = calls(&c.value);
        if let Some(h) = &c.having {
            for k in calls(h) {
                if !wanted.iter().any(|w| **w == *k) {
                    wanted.push(k);
                }
            }
        }
        let mut base = base;
        let mut reads = Vec::new();
        for k in wanted {
            let plan = self.one_call(c, Arc::clone(&k))?;
            let (joined, read) = self.scalar(base, &plan)?;
            base = joined;
            reads.push((k, read));
        }
        let value = replace(&c.value, &reads);
        let Some(h) = &c.having else {
            return Ok((base, value));
        };
        let data_type = typed(&value).0.unwrap_or(DataType::Null);
        let null = Arc::new(Expr::literal(Literal::Null(Some(data_type.clone()))));
        let guarded = call(
            Builtin::When,
            vec![replace(h, &reads), value, null],
            data_type,
            true,
        );
        Ok((base, guarded))
    }

    /// `v IN (plan)` over a [`Composite`] with a `HAVING`, as scalar
    /// subqueries: `CASE WHEN having THEN v = value ELSE false END`.
    pub(super) fn in_composite(
        &mut self,
        value: &Arc<Expr>,
        plan: &Arc<LogicalPlan>,
    ) -> Option<Arc<Expr>> {
        let c = composite(plan)?;
        let having = c.having.clone()?;
        let scalar = |p: Arc<LogicalPlan>| {
            Arc::new(Expr::new(ExprNode::Subquery {
                kind: SubqueryKind::Scalar,
                plan: p,
            }))
        };
        // A value without a call (`SELECT 1 ... HAVING ...`) is a constant.
        let v = if calls(&c.value).is_empty() {
            Arc::clone(&c.value)
        } else {
            scalar(self.one_call(&c, Arc::clone(&c.value)).ok()?)
        };
        let h = scalar(self.one_call(&c, having).ok()?);
        let eq = boolean(Builtin::Eq, vec![Arc::clone(value), v]);
        let no = Arc::new(Expr::literal(Literal::Boolean(false)));
        let nullable = typed(&eq).1;
        Some(call(
            Builtin::When,
            vec![h, eq, no],
            DataType::Boolean,
            nullable,
        ))
    }
}

/// Whether grouping expression `k` of an aggregate over `input` is fixed per
/// outer row: equal, in a filter on the way down, to an expression of outer
/// columns or constants (Spark's `getCorrelatedEquivalentInnerExpressions`,
/// with `scalarSubqueryAllowGroupByColumnEqualToConstant` at its default).
pub(super) fn fixed(k: &Arc<Expr>, input: &LogicalPlan) -> bool {
    match &input.node {
        PlanNode::Filter { input, predicate } => {
            conjuncts(predicate).iter().any(|c| {
                let ExprNode::Resolved(ResolvedExpr::Function(f)) = &c.node else {
                    return false;
                };
                let [a, b] = &f.args[..] else { return false };
                f.id == crate::FunctionId::Builtin(Builtin::Eq)
                    && ((**a == **k && inner_columns(b).is_empty())
                        || (**b == **k && inner_columns(a).is_empty()))
            }) || fixed(k, input)
        }
        PlanNode::Project { input, .. }
        | PlanNode::SubqueryAlias { input, .. }
        | PlanNode::Sort { input, .. } => fixed(k, input),
        PlanNode::Join {
            left, right, kind, ..
        } => match kind {
            JoinKind::Inner | JoinKind::Cross => fixed(k, left) || fixed(k, right),
            JoinKind::LeftOuter | JoinKind::LeftSemi | JoinKind::LeftAnti => fixed(k, left),
            JoinKind::RightOuter => fixed(k, right),
            _ => false,
        },
        _ => false,
    }
}

impl Rewriter {
    /// A correlated scalar subquery grouped by `keys` over `input`, where
    /// the correlation may leave several groups for an outer row: a single
    /// join through the domain of the outer columns (an outer join under
    /// the legacy flag), or Spark's refusal with the single join off.
    pub(super) fn scalar_grouped(
        &mut self,
        base: Arc<LogicalPlan>,
        plan: &Arc<LogicalPlan>,
        input: &Arc<LogicalPlan>,
        keys: &[Arc<Expr>],
    ) -> R<(Arc<LogicalPlan>, Arc<Expr>)> {
        let loose: Vec<&Arc<Expr>> = keys
            .iter()
            .filter(|k| !inner_columns(k).is_empty() && !fixed(k, input))
            .collect();
        // Spark's legacy count handling gives such a subquery its zero-row
        // value where nothing matched, which the single join does not.
        if self.legacy_scalar_count && !loose.is_empty() {
            return Err(super::unsupported(
                "a loosely grouped correlated scalar subquery under \
                 spark.sql.optimizer.decorrelateSubqueryLegacyIncorrectCountHandling.enabled",
                plan.origin,
            ));
        }
        let kind = if loose.is_empty() || self.single_join {
            JoinKind::Single
        } else if self.legacy_group_by {
            JoinKind::LeftOuter
        } else {
            let names: Vec<String> = loose
                .iter()
                .map(|k| match &k.node {
                    ExprNode::Resolved(ResolvedExpr::Column(c)) => c.name.clone(),
                    _ => pretty(k, None),
                })
                .collect();
            return Err(AnalysisError::new(
                UnsupportedSubqueryExpressionCategoryNonCorrelatedColumnsInGroupBy {
                    value: names.join(","),
                },
                plan.origin,
            ));
        };
        // The domain goes through a grouped aggregate over a correlation
        // that is not an equality (as for a lateral join's right side),
        // grouping by the domain columns too.
        let lateral = std::mem::replace(&mut self.lateral, true);
        let out = self.scalar_single(base, plan, Some(kind));
        self.lateral = lateral;
        out
    }
}

/// Whether `p` has at most one row whatever its input (Spark's
/// `maxRows <= 1`): a global aggregate, `LIMIT 0` or `1`, one row of
/// literals, through the operators that never add rows.
fn at_most_one_row(p: &LogicalPlan) -> bool {
    match &p.node {
        PlanNode::OneRowRelation => true,
        PlanNode::Aggregate {
            grouping: Grouping::GroupBy(keys),
            ..
        } => keys.is_empty(),
        PlanNode::Limit { input, limit } => {
            matches!(limit.node, ExprNode::Literal(Literal::Integer(0 | 1)))
                || at_most_one_row(input)
        }
        PlanNode::Project { input, .. }
        | PlanNode::SubqueryAlias { input, .. }
        | PlanNode::Filter { input, .. }
        | PlanNode::Sort { input, .. } => at_most_one_row(input),
        _ => false,
    }
}

impl Rewriter {
    /// Spark's check of a correlated scalar subquery that is not an
    /// aggregate when the single join is off (and on Spark 3.5, which has
    /// none): it must have at most one row.
    pub(super) fn check_single(&self, plan: &Arc<LogicalPlan>) -> R<()> {
        // Spark strips projections and aliases first; an aggregate (or a
        // `HAVING` over one) gets the aggregate's checks instead.
        let mut clean = &**plan;
        while let PlanNode::Project { input, .. } | PlanNode::SubqueryAlias { input, .. } =
            &clean.node
        {
            clean = input;
        }
        let aggregate = match &clean.node {
            PlanNode::Aggregate { .. } => true,
            PlanNode::Filter { input, .. } => matches!(input.node, PlanNode::Aggregate { .. }),
            _ => false,
        };
        if self.single_join || aggregate || at_most_one_row(plan) {
            return Ok(());
        }
        Err(AnalysisError::new(
            UnsupportedSubqueryExpressionCategoryMustAggregateCorrelatedScalarSubquery {},
            plan.origin,
        ))
    }
}

/// `keys` extended by the columns the subquery joins under an aggregate
/// added (an existence join's `exists`, a scalar subquery's value) that
/// `aggregates` read outside an aggregate call and outside a grouping key
/// (a key that holds the column already decides it:
/// `GROUP BY k > 1 OR k IN (..)` must not group by the `IN` alone too).
/// [`check_grouped_reads`] made sure such a subquery reads only grouped
/// columns, so its column is fixed within a group and grouping by it
/// changes no group.
pub(super) fn group_joined(
    mut keys: Vec<Arc<Expr>>,
    aggregates: &[Arc<Expr>],
    before: &[BoundColumn],
) -> Vec<Arc<Expr>> {
    let given = keys.len();
    let mut stack: Vec<Arc<Expr>> = aggregates.to_vec();
    while let Some(x) = stack.pop() {
        if is_call(&x) || keys[..given].iter().any(|k| **k == *x) {
            continue;
        }
        if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &x.node {
            let joined = !before.iter().any(|b| b.id == c.id);
            let grouped = keys.iter().any(
                |k| matches!(&k.node, ExprNode::Resolved(ResolvedExpr::Column(o)) if o.id == c.id),
            );
            if joined && !grouped {
                keys.push(Arc::clone(&x));
            }
            continue;
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    keys
}

/// Spark's `MISSING_AGGREGATION` check for a subquery in an aggregate's
/// output, outside a call and a grouping key: it may read only grouping
/// columns of the aggregate's input (`exists(.. where b.k = a.v) .. group by
/// a.k` is refused). Refused here rather than answered per input row.
pub(super) fn check_grouped_reads(
    keys: &[Arc<Expr>],
    aggregates: &[Arc<Expr>],
    input: &[BoundColumn],
    origin: Origin,
) -> R<()> {
    // The columns the grouping keys read, inside their subqueries too (a
    // key and an output may hold separately resolved copies of one
    // subquery, `GROUP BY cast(exists(..) as int)`).
    let mut key_reads: Vec<crate::AttrId> = Vec::new();
    for k in keys {
        let _ = crate::walk::visit_distinct(Node::Expr(k), |n| {
            if let Node::Expr(e) = n
                && let ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c)) =
                    &e.node
            {
                key_reads.push(c.id);
            }
            std::ops::ControlFlow::Continue(())
        });
    }
    let grouped = |id: crate::AttrId| key_reads.contains(&id);
    let from_input = |id: crate::AttrId| input.iter().any(|c| c.id == id);
    let mut stack: Vec<Arc<Expr>> = aggregates.to_vec();
    while let Some(x) = stack.pop() {
        if is_call(&x) || keys.iter().any(|k| **k == *x) {
            continue;
        }
        if let ExprNode::Subquery { kind, plan } = &x.node {
            let mut reads: Vec<crate::AttrId> = Vec::new();
            let _ = crate::walk::visit_distinct(Node::Plan(plan), |n| {
                if let Node::Expr(e) = n
                    && let ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) = &e.node
                {
                    reads.push(c.id);
                }
                std::ops::ControlFlow::Continue(())
            });
            if let SubqueryKind::In(values) = kind {
                reads.extend(values.iter().flat_map(inner_columns).map(|c| c.id));
            }
            if reads.iter().any(|id| from_input(*id) && !grouped(*id)) {
                return Err(super::unsupported(
                    "a subquery in an aggregate's output that reads a column not grouped by",
                    origin,
                ));
            }
            continue;
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    Ok(())
}

/// `e` with every `NOT` above an `AND`, an `OR` or another `NOT` that holds
/// a subquery pushed down to its operands (De Morgan's laws and `NOT NOT x =
/// x`, exact in three-valued logic), as
/// Spark's `BooleanSimplification` does before its subquery rewrite: a
/// `NOT` then wraps the `IN` itself, whose null-aware join gives the NULL
/// that `NOT (a > 5 OR a IN (... NULL ...))` must keep (an existence join's
/// column alone is never NULL).
pub(super) fn push_not(e: &Arc<Expr>) -> Arc<Expr> {
    fn not_of(e: &Arc<Expr>) -> Option<&Arc<Expr>> {
        match super::builtin_args(e) {
            Some((Builtin::Not, [x])) => Some(x),
            _ => None,
        }
    }
    fn push(e: &Arc<Expr>, negate: bool) -> Arc<Expr> {
        if !has_subquery(Node::Expr(e)) {
            return if negate {
                boolean(Builtin::Not, vec![Arc::clone(e)])
            } else {
                Arc::clone(e)
            };
        }
        if let Some(x) = not_of(e) {
            return push(x, !negate);
        }
        match super::builtin_args(e) {
            Some((op @ (Builtin::And | Builtin::Or), [a, b])) => {
                let op = match (op, negate) {
                    (Builtin::And, true) => Builtin::Or,
                    (Builtin::Or, true) => Builtin::And,
                    (op, _) => op,
                };
                boolean(op, vec![push(a, negate), push(b, negate)])
            }
            _ if negate => boolean(Builtin::Not, vec![Arc::clone(e)]),
            _ => Arc::clone(e),
        }
    }
    // Only a `NOT` over a connective or a `NOT` with a subquery inside
    // changes.
    let mut changes = false;
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        if let Some(inner) = not_of(&x)
            && matches!(
                super::builtin_args(inner),
                Some((Builtin::And | Builtin::Or | Builtin::Not, _))
            )
            && has_subquery(Node::Expr(inner))
        {
            changes = true;
            break;
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    if !changes {
        return Arc::clone(e);
    }
    let out = Arc::clone(e).transform_down(&mut |x: Arc<Expr>| {
        Ok::<_, std::convert::Infallible>(match not_of(&x) {
            Some(inner)
                if matches!(
                    super::builtin_args(inner),
                    Some((Builtin::And | Builtin::Or | Builtin::Not, _))
                ) && has_subquery(Node::Expr(inner)) =>
            {
                push(inner, true)
            }
            _ => x,
        })
    });
    out.unwrap_or_else(|never| match never {})
}

/// [`super::needs_domain`], or a positive `LIMIT` over a correlation that
/// is not an equality (`WHERE t2b <= t1d LIMIT 2`): its rows are ranked
/// within each value of the outer columns, which only the domain gives (an
/// equality's keys partition the ranking without it).
///
/// A conjunct that reads only outer columns (`WHERE x1 < x2 ... LIMIT 1`)
/// is left out: Spark 4.2.0 answers it with the first row in input order,
/// ignoring the subquery's `ORDER BY`, which this does not reproduce, so it
/// stays refused rather than answered differently.
pub(super) fn limit_hit(p: &LogicalPlan) -> bool {
    super::needs_domain(p)
        || matches!(&p.node, PlanNode::Limit { input, limit }
            if super::non_equality_correlation(input)
                && reads_inner(input)
                && super::positive_literal(limit)
                && !has_outer(Node::Expr(limit)))
}

/// Whether every correlated conjunct of the filters on `p`'s path down
/// (through projections, sorts and aliases) reads an inner column too.
fn reads_inner(p: &LogicalPlan) -> bool {
    match &p.node {
        PlanNode::Filter { input, predicate } => {
            conjuncts(predicate)
                .iter()
                .filter(|c| has_outer(Node::Expr(c)))
                .all(|c| !inner_columns(c).is_empty())
                && reads_inner(input)
        }
        PlanNode::Project { input, .. }
        | PlanNode::Sort { input, .. }
        | PlanNode::SubqueryAlias { input, .. } => reads_inner(input),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AttrId;

    fn col(id: u64, name: &str) -> Arc<Expr> {
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
            BoundColumn {
                id: AttrId(id),
                name: name.to_owned(),
                data_type: DataType::Integer,
                nullable: true,
            },
        ))))
    }

    fn outer(id: u64, name: &str) -> Arc<Expr> {
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::OuterColumn(
            BoundColumn {
                id: AttrId(id),
                name: name.to_owned(),
                data_type: DataType::Integer,
                nullable: true,
            },
        ))))
    }

    fn lit(v: i32) -> Arc<Expr> {
        Arc::new(Expr::literal(Literal::Integer(v)))
    }

    fn in_sub(v: Arc<Expr>) -> Arc<Expr> {
        Arc::new(Expr::new(ExprNode::Subquery {
            kind: SubqueryKind::In(vec![v]),
            plan: Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
        }))
    }

    fn op(b: Builtin, args: Vec<Arc<Expr>>) -> Arc<Expr> {
        boolean(b, args)
    }

    fn shape(e: &Expr) -> String {
        match &e.node {
            ExprNode::Subquery { .. } => "IN".to_owned(),
            ExprNode::Resolved(ResolvedExpr::Column(c)) => c.name.clone(),
            ExprNode::Literal(_) => "lit".to_owned(),
            ExprNode::Resolved(ResolvedExpr::Function(f)) => {
                let args: Vec<String> = f.args.iter().map(|a| shape(a)).collect();
                format!("{:?}({})", f.id, args.join(", "))
            }
            _ => "?".to_owned(),
        }
    }

    #[test]
    fn a_not_moves_down_to_the_subquery() {
        let a = col(1, "a");
        let gt = op(Builtin::Gt, vec![Arc::clone(&a), lit(5)]);
        let e = op(
            Builtin::Not,
            vec![op(
                Builtin::Or,
                vec![Arc::clone(&gt), in_sub(Arc::clone(&a))],
            )],
        );
        let pushed = push_not(&e);
        let want = op(
            Builtin::And,
            vec![
                op(Builtin::Not, vec![gt]),
                op(Builtin::Not, vec![in_sub(Arc::clone(&a))]),
            ],
        );
        assert_eq!(shape(&pushed), shape(&want));
        // `NOT NOT x` is `x`.
        let twice = op(
            Builtin::Not,
            vec![op(Builtin::Not, vec![in_sub(Arc::clone(&a))])],
        );
        assert_eq!(shape(&push_not(&twice)), "IN");
        // Without a subquery nothing changes.
        let plain = op(
            Builtin::Not,
            vec![op(Builtin::Or, vec![Arc::clone(&a), lit(1)])],
        );
        assert!(Arc::ptr_eq(&push_not(&plain), &plain));
    }

    #[test]
    fn keys_equal_to_constants_or_outer_expressions_are_fixed() {
        let y1 = col(1, "y1");
        let y2 = col(2, "y2");
        let scan = Arc::new(LogicalPlan::new(PlanNode::OneRowRelation));
        let filter = |predicate: Arc<Expr>| {
            LogicalPlan::new(PlanNode::Filter {
                input: Arc::clone(&scan),
                predicate,
            })
        };
        let by_constant = filter(op(Builtin::Eq, vec![Arc::clone(&y2), lit(1)]));
        assert!(fixed(&y2, &by_constant));
        assert!(!fixed(&y1, &by_constant));
        let by_outer = filter(op(Builtin::Eq, vec![outer(9, "x1"), Arc::clone(&y1)]));
        assert!(fixed(&y1, &by_outer));
        let greater = filter(op(Builtin::Gt, vec![Arc::clone(&y1), outer(9, "x1")]));
        assert!(!fixed(&y1, &greater));
    }

    #[test]
    fn joined_columns_read_outside_a_call_are_grouped() {
        let k = col(1, "k");
        let joined = col(7, "exists");
        let before = vec![BoundColumn {
            id: AttrId(1),
            name: "k".to_owned(),
            data_type: DataType::Integer,
            nullable: true,
        }];
        let keys = group_joined(vec![Arc::clone(&k)], &[Arc::clone(&joined)], &before);
        assert_eq!(keys.len(), 2);
        // Inside an aggregate call it is aggregated, not grouped.
        let count = Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            crate::ResolvedFunction {
                id: crate::FunctionId::Builtin(Builtin::Not),
                kind: FunctionKind::Aggregate,
                args: vec![joined],
                distinct: false,
                filter: None,
                within_group: Vec::new(),
                data_type: DataType::Long,
                nullable: false,
                deterministic: true,
            },
        ))));
        let keys = group_joined(vec![k], &[count], &before);
        assert_eq!(keys.len(), 1);
    }

    #[test]
    fn the_subquery_flags_follow_session_confs() {
        use crate::analyzer::ResolutionConf;
        let conf = |kv: &[(&str, &str)]| {
            let kv: Vec<(String, String)> = kv
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect();
            ResolutionConf::from_conf(|key| {
                kv.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
            })
        };
        let d = ResolutionConf::default();
        assert!(d.scalar_single_join && !d.legacy_group_by_non_equality);
        assert!(!d.legacy_exists_in_count && !d.legacy_scalar_count);
        let single = conf(&[
            ("spark.sql.optimizer.scalarSubqueryUseSingleJoin", "false"),
            (
                "spark.sql.legacy.scalarSubqueryAllowGroupByNonEqualityCorrelatedPredicate",
                "true",
            ),
        ]);
        assert!(!single.scalar_single_join && single.legacy_group_by_non_equality);
        let counts = conf(&[
            (
                "spark.sql.optimizer.decorrelateExistsSubqueryLegacyIncorrectCountHandling.enabled",
                "true",
            ),
            (
                "spark.sql.optimizer.decorrelateSubqueryLegacyIncorrectCountHandling.enabled",
                " TRUE ",
            ),
        ]);
        assert!(counts.legacy_exists_in_count && counts.legacy_scalar_count);
    }
}
