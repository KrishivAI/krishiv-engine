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

/// Whether `e` is an integer literal above zero.
pub(super) fn positive_literal(e: &Expr) -> bool {
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
pub(super) fn one_row_literals(p: &LogicalPlan) -> bool {
    match &p.node {
        PlanNode::OneRowRelation => true,
        PlanNode::InlineTable { rows, .. } => rows.len() == 1,
        PlanNode::Project { input, .. } | PlanNode::SubqueryAlias { input, .. } => {
            one_row_literals(input)
        }
        _ => false,
    }
}

/// `IN` forms Spark folds before any join (`NullPropagation`, one-row literals)
/// left here: a NULL among several values, or several over one literal row
/// (one value over one row is an equality). Refused, not answered two-valued.
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
