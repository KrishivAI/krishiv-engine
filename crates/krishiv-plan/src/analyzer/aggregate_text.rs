// SPDX-License-Identifier: Apache-2.0
//! The aggregate calls left in an expression, as Spark's `sql` prints them (for `INVALID_WHERE_CONDITION` and
//! `UNSUPPORTED_EXPR_FOR_OPERATOR`).

use std::collections::HashSet;
use std::sync::Arc;

use super::exprs::pretty;
use super::ops::qualified;
use super::resolve::Attr;
use crate::expr::{Expr, ExprNode, ResolvedExpr};
use crate::function::FunctionKind;

/// The aggregate calls in `e` as Spark's `sql` prints them, columns fully
/// qualified (for `INVALID_WHERE_CONDITION`).
pub(super) fn aggregate_sql(e: &Arc<Expr>, scope: &[Attr]) -> String {
    let mut found = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        if !seen.insert(super::bind::addr(&*x)) {
            continue;
        }
        if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &x.node
            && f.kind == FunctionKind::Aggregate
        {
            found.push(pretty(&qualified(&x, scope), None));
            continue;
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    found.join(", ")
}

/// The aggregate calls in `e` as Spark's `sql` prints them (columns unqualified), in the order `e` reads them.
pub(super) fn aggregate_calls(e: &Arc<Expr>) -> Vec<String> {
    let mut found = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        if !seen.insert(super::bind::addr(&*x)) {
            continue;
        }
        if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &x.node
            && f.kind == FunctionKind::Aggregate
        {
            found.push(pretty(&x, None));
            continue;
        }
        let mut children = Vec::new();
        x.for_each_child(&mut |k| children.push(Arc::clone(k)));
        stack.extend(children.into_iter().rev());
    }
    found
}
