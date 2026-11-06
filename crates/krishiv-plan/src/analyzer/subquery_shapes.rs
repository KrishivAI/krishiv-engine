// SPDX-License-Identifier: Apache-2.0
//! Shape tests over subquery plans and expressions that the subquery
//! rewrite (`subquery.rs`) and the expression binder ask: one-row
//! aggregates, literal relations, the `IN` forms Spark folds first.

use std::sync::Arc;

use super::AnalysisError;
use super::bind::unsupported;
use super::subquery::has_outer;
use crate::walk::Node;
use crate::{Expr, ExprNode, Grouping, Literal, LogicalPlan, Origin, PlanNode, ResolvedExpr};

type R<T> = Result<T, AnalysisError>;

/// Whether `e` is a NULL literal or a cast of one.
pub(super) fn null_literal(e: &Expr) -> bool {
    match &e.node {
        ExprNode::Literal(v) => matches!(v, Literal::Null(_)),
        ExprNode::Cast { expr, .. } | ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => {
            null_literal(expr)
        }
        _ => false,
    }
}

/// The value of `e` when it is a non-negative integer literal (or a cast of
/// one): a `LIMIT` or `OFFSET` count.
pub(super) fn count_literal(e: &Expr) -> Option<i64> {
    let n = match &e.node {
        ExprNode::Literal(v) => match v {
            Literal::Byte(n) => i64::from(*n),
            Literal::Short(n) => i64::from(*n),
            Literal::Integer(n) => i64::from(*n),
            Literal::Long(n) => *n,
            _ => return None,
        },
        ExprNode::Cast { expr, .. } => return count_literal(expr),
        _ => return None,
    };
    (n >= 0).then_some(n)
}

/// A `LIMIT`, a `LIMIT .. OFFSET` or an `OFFSET` (Spark's `Limit` over an
/// `Offset`) over `below`, with literal counts that read no outer column.
pub(super) struct Bounded {
    /// What the counts apply to.
    pub(super) below: Arc<LogicalPlan>,
    /// The `LIMIT`, if any.
    pub(super) limit: Option<i64>,
    /// The `OFFSET`, 0 if none.
    pub(super) offset: i64,
}

/// `p` as a [`Bounded`], if it is one.
pub(super) fn bounded(p: &LogicalPlan) -> Option<Bounded> {
    let count = |e: &Arc<Expr>| {
        if has_outer(Node::Expr(e)) {
            None
        } else {
            count_literal(e)
        }
    };
    match &p.node {
        PlanNode::Limit { input, limit } => {
            let limit = count(limit)?;
            match &input.node {
                PlanNode::Offset { input, offset } => Some(Bounded {
                    below: Arc::clone(input),
                    limit: Some(limit),
                    offset: count(offset)?,
                }),
                _ => Some(Bounded {
                    below: Arc::clone(input),
                    limit: Some(limit),
                    offset: 0,
                }),
            }
        }
        PlanNode::Offset { input, offset } => Some(Bounded {
            below: Arc::clone(input),
            limit: None,
            offset: count(offset)?,
        }),
        _ => None,
    }
}

/// `p` without a `LIMIT` / `OFFSET` over a correlated global aggregate (one
/// row per outer row): a positive `LIMIT` alone keeps that row, so it goes;
/// an `OFFSET` or `LIMIT 0` drops it, so the subquery has no row for any
/// outer row and becomes an uncorrelated empty relation with the same
/// output (NULL for a scalar, false for `EXISTS` and `IN`). Spark
/// decorrelates it to the same rows (`DecorrelateInnerQuery`: a row number
/// per correlation value, which is 1).
pub(super) fn fold_one_row_bounds(p: &Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    let Some(b) = bounded(p) else {
        return Arc::clone(p);
    };
    if !one_row_aggregate(&b.below) {
        return Arc::clone(p);
    }
    if b.offset == 0 && b.limit.is_none_or(|l| l > 0) {
        return b.below;
    }
    let Some(out) = b.below.resolved_output() else {
        return Arc::clone(p);
    };
    let none = super::subquery::node(
        PlanNode::Filter {
            input: super::subquery::node(PlanNode::OneRowRelation, p.origin),
            predicate: Arc::new(Expr::literal(Literal::Boolean(false))),
        },
        p.origin,
    );
    let exprs = out
        .into_iter()
        .map(|c| {
            let null = Arc::new(Expr::literal(Literal::Null(Some(c.data_type.clone()))));
            Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Alias {
                expr: null,
                output: c,
            })))
        })
        .collect();
    super::subquery::node(PlanNode::Project { input: none, exprs }, p.origin)
}

/// Whether `p` reads the outer query and is one row for each outer row:
/// a global aggregate under projections and aliases (no filter, so no
/// `HAVING`).
pub(super) fn one_row_aggregate(p: &Arc<LogicalPlan>) -> bool {
    if !has_outer(Node::Plan(p)) {
        return false;
    }
    fn walk(p: &LogicalPlan) -> bool {
        match &p.node {
            PlanNode::Aggregate {
                grouping: Grouping::GroupBy(keys),
                ..
            } => keys.is_empty(),
            PlanNode::Project { input, .. } => walk(input),
            PlanNode::SubqueryAlias { input, columns, .. } if columns.is_empty() => walk(input),
            _ => false,
        }
    }
    walk(p)
}

/// Whether an aggregate sits under `p`'s projections and filters.
pub(super) fn aggregates_below(p: &LogicalPlan) -> bool {
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
pub(super) fn check_in(values: &[Arc<Expr>], plan: &Arc<LogicalPlan>, origin: Origin) -> R<()> {
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
