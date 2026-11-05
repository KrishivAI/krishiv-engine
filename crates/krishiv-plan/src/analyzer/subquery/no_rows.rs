// SPDX-License-Identifier: Apache-2.0
//! The value of a correlated scalar subquery over no rows: what an outer row without a match reads.

use crate::{Builtin, Expr, ExprNode, FunctionId, FunctionKind, Literal, ResolvedExpr};
use krishiv_types::DataType;
use std::sync::Arc;

/// What a scalar subquery's value `e` is over no rows: `None` when NULL,
/// `Some(0)` for a bare `count`; `None` of the outer option when unknown
/// (Spark evaluates other forms; Krishiv does not yet).
pub(super) fn on_no_rows(e: &Arc<Expr>) -> Option<Option<Literal>> {
    // A bare count is 0 over no rows.
    if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node
        && f.kind == FunctionKind::Aggregate
        && f.id.name() == "count"
    {
        return Some(Some(Literal::Long(0)));
    }
    // `x IS [NOT] NULL` over integer arithmetic of aggregates is a boolean: `(count(*)) IS NULL` is false.
    if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node
        && matches!(
            f.id,
            FunctionId::Builtin(Builtin::IsNull | Builtin::IsNotNull)
        )
        && let [arg] = &f.args[..]
        && let Some(v) = integer_on_no_rows(arg)
    {
        let is_null = f.id == FunctionId::Builtin(Builtin::IsNull);
        return Some(Some(Literal::Boolean(v.is_none() == is_null)));
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
