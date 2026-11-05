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

use std::collections::HashMap;
use std::sync::Arc;

use super::{R, Rewriter, boolean, call, has_outer, node, on_no_rows, typed};
use crate::analyzer::exprs::pretty;
use crate::walk::Node;
use crate::{
    BoundColumn, Builtin, Expr, ExprNode, FunctionKind, Grouping, Literal, LogicalPlan, Origin,
    PlanNode, ResolvedExpr, SubqueryKind,
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
