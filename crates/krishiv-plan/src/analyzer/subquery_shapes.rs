// SPDX-License-Identifier: Apache-2.0
//! Shape tests over subquery plans and expressions that the subquery
//! rewrite (`subquery.rs`) and the expression binder ask: one-row
//! aggregates, literal relations, the `IN` forms Spark folds first.

use std::collections::HashSet;
use std::ops::ControlFlow;
use std::sync::Arc;

use super::AnalysisError;
use super::bind::unsupported;
use super::ops::col_expr;
use super::subquery::has_outer;
use crate::walk::{Node, visit_distinct};
use crate::{
    AttrId, Builtin, Expr, ExprNode, Grouping, Literal, LogicalPlan, Origin, PlanNode,
    ResolvedExpr, SubqueryKind,
};
use krishiv_types::DataType;

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

/// The `IN` form Spark folds before any join (`NullPropagation`): `v IN
/// (q)` with a NULL `v`. It is refused rather than answered two-valued. Over
/// a one-row relation of literals an `IN` is an existence join as over any
/// other (`x IN (SELECT 1)` is false for a NULL `x` in Spark 4.2.0).
pub(super) fn check_in(values: &[Arc<Expr>], origin: Origin) -> R<()> {
    if values.iter().any(|v| null_literal(v)) {
        return Err(unsupported("IN (subquery) with a NULL value", origin));
    }
    Ok(())
}

/// The attribute ids of the enclosing query's columns the subqueries in `e`
/// read.
pub(super) fn outer_ids(e: &Arc<Expr>) -> HashSet<AttrId> {
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

/// The columns an `IN` subquery `s` reads outside its own plan: its values'
/// and its outer references.
fn in_reads(s: &Arc<Expr>, values: &[Arc<Expr>]) -> HashSet<AttrId> {
    let mut ids = outer_ids(s);
    let _ = values.iter().try_for_each(|v| {
        visit_distinct(Node::Expr(v), |n| {
            if let Node::Expr(x) = n
                && let ExprNode::Resolved(ResolvedExpr::Column(c)) = &x.node
            {
                ids.insert(c.id);
            }
            ControlFlow::<()>::Continue(())
        })
    });
    ids
}

/// The join condition `on` with each `IN` subquery that reads neither join
/// input (`ON (3 IN (SELECT b FROM t)) IS NULL`) three-valued: Spark's
/// `RewritePredicateSubquery` leaves such a subquery an expression, NULL
/// when no row equals the value and some comparison is NULL. Over the
/// subquery `q` with column `c`, `v IN q` is `EXISTS (q WHERE c = v) OR
/// (EXISTS (q WHERE (c = v) IS NULL) AND NULL)`, both uncorrelated. A
/// multi-value `IN` of this kind, or one under
/// `spark.sql.legacy.nullInEmptyListBehavior` (NULL over no rows), is
/// refused. A NULL literal value is left to [`check_in`].
pub(super) fn three_valued_ins(on: &Arc<Expr>, legacy: bool) -> R<Arc<Expr>> {
    Arc::clone(on).transform_down(&mut |x: Arc<Expr>| {
        let ExprNode::Subquery {
            kind: SubqueryKind::In(values),
            plan,
        } = &x.node
        else {
            return Ok(x);
        };
        if !in_reads(&x, values).is_empty() || values.iter().any(|v| null_literal(v)) {
            return Ok(x);
        }
        let ([v], [c], false) = (&values[..], &super::subquery::output(plan)?[..], legacy) else {
            return Err(unsupported(
                "IN (subquery) in a join condition that reads neither side, with several values or a legacy NULL",
                x.origin,
            ));
        };
        let eq = super::subquery::boolean(Builtin::Eq, vec![Arc::clone(v), col_expr(c, Origin::UNKNOWN)]);
        let exists = |predicate: Arc<Expr>| {
            Arc::new(Expr::new(ExprNode::Subquery {
                kind: SubqueryKind::Exists,
                plan: super::subquery::node(
                    PlanNode::Filter {
                        input: Arc::clone(plan),
                        predicate,
                    },
                    plan.origin,
                ),
            }))
        };
        let unknown = exists(super::subquery::boolean(Builtin::IsNull, vec![Arc::clone(&eq)]));
        let null = Arc::new(Expr::new(ExprNode::Literal(Literal::Null(Some(DataType::Boolean)))));
        Ok(super::subquery::boolean(
            Builtin::Or,
            vec![exists(eq), super::subquery::boolean(Builtin::And, vec![unknown, null])],
        ))
    })
}

/// The join condition conjunct `c`, an `IN` / `EXISTS` read from one join
/// side `side`, refused when an `IN` in it reads a side that is a select
/// list over no relation (through projections and aliases): Spark folds such a constant column into
/// the condition (`SELECT 1 AS a`), and the `IN` then reads neither side
/// and is three-valued (in-nullability), where an existence join is not.
pub(super) fn constant_side(c: Arc<Expr>, side: &Arc<LogicalPlan>) -> R<Arc<Expr>> {
    let has_in = super::subquery::subqueries(&c).iter().any(|s| {
        matches!(
            s.node,
            ExprNode::Subquery {
                kind: SubqueryKind::In(_),
                ..
            }
        )
    });
    fn constant(p: &LogicalPlan) -> bool {
        match &p.node {
            PlanNode::OneRowRelation => true,
            PlanNode::Project { input, .. } | PlanNode::SubqueryAlias { input, .. } => {
                constant(input)
            }
            _ => false,
        }
    }
    if has_in && constant(side) {
        return Err(unsupported(
            "IN (subquery) in a join condition over a side with constant columns",
            c.origin,
        ));
    }
    Ok(c)
}
