// SPDX-License-Identifier: Apache-2.0
//! Subqueries in a `HAVING` over an aggregate: the aggregate calls of a
//! subquery's `WHERE` over only the aggregate's input are that aggregate's
//! (Spark's outer aggregates, `HAVING EXISTS (.. WHERE b < min(emp.salary))`),
//! and the grouping attributes a subquery reads must be in the filter's input.

use std::collections::{HashMap, HashSet};
use std::ops::ControlFlow;
use std::sync::Arc;

use super::AnalysisError;
use super::bind::unsupported;
use super::exprs::{Bound, semantic_key};
use super::ops::{Grouped, col_expr, no_qual};
use super::resolve::{Attr, Env, Resolver};
use crate::walk::{Node, visit_distinct};
use crate::{AttrId, Expr, ExprNode, FunctionKind, LogicalPlan, Origin, PlanNode, ResolvedExpr};
use krishiv_common::error::{
    MissingAttributesResolvedAttributeMissingFromInput,
    UnsupportedSubqueryExpressionCategoryAggregateFunctionMixedOuterLocalReferences,
};

type R<T> = Result<T, AnalysisError>;

fn is_aggregate(x: &Expr) -> bool {
    matches!(&x.node, ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate)
}

/// Whether the aggregate call `x` reads columns of `input` (through outer
/// references) and nothing else: no column of its own query, no subquery.
fn is_outer_aggregate(x: &Arc<Expr>, input: &[Attr]) -> bool {
    let mut outer = false;
    let local = visit_distinct(Node::Expr(x), |n| match n {
        Node::Expr(y) => match &y.node {
            ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) => {
                if input.iter().any(|a| a.col.id == c.id) {
                    outer = true;
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(())
                }
            }
            ExprNode::Resolved(ResolvedExpr::Column(_) | ResolvedExpr::LambdaVar(_))
            | ExprNode::Subquery { .. } => ControlFlow::Break(()),
            _ => ControlFlow::Continue(()),
        },
        Node::Plan(_) => ControlFlow::Break(()),
    });
    outer && local.is_continue()
}

/// Whether every aggregate call in the `WHERE` condition `e` is an outer
/// aggregate of the `HAVING` the subquery sits in (the innermost enclosing
/// query): Spark computes it there, so the condition is no
/// `INVALID_WHERE_CONDITION`.
pub(super) fn outer_aggregates_only(e: &Arc<Expr>, env: &Env) -> bool {
    let Some(input) = env.outer.last().and_then(|l| l.having.as_deref()) else {
        return false;
    };
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&x)) {
            continue;
        }
        if is_aggregate(&x) {
            if !is_outer_aggregate(&x, input) {
                return false;
            }
            continue;
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    true
}

/// `x` with its outer references read as columns of the aggregate's input.
fn as_local(x: &Arc<Expr>) -> Arc<Expr> {
    let local: Result<_, std::convert::Infallible> =
        Arc::clone(x).transform_up(&mut |y: Arc<Expr>| match &y.node {
            ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) => Ok(col_expr(c, y.origin)),
            _ => Ok(y),
        });
    match local {
        Ok(e) => e,
    }
}

impl Resolver<'_> {
    /// The attribute the aggregate outputs the aggregate call `x` as (`key`
    /// is its `semantic_key`), added to the aggregate when it does not
    /// compute it yet, and passed up when the filter's input does not show
    /// it.
    pub(super) fn aggregate_output(
        &mut self,
        x: &Arc<Expr>,
        key: u128,
        st: &mut Grouped<'_>,
    ) -> R<Attr> {
        let ExprNode::Resolved(ResolvedExpr::Function(f)) = &x.node else {
            return Err(unsupported("an aggregate call", x.origin));
        };
        // A call over a column of this query and an outer reference is
        // Spark's `AGGREGATE_FUNCTION_MIXED_OUTER_LOCAL_REFERENCES`, here in
        // the `HAVING` or `ORDER BY` of a subquery's own aggregate.
        if let Some(m) = mixed_aggregate(x) {
            return Err(AnalysisError::new(
                UnsupportedSubqueryExpressionCategoryAggregateFunctionMixedOuterLocalReferences {
                    function: mixed_sql(&m, &st.info.input.out, st.env),
                },
                x.origin,
            ));
        }
        let a = match st.known.iter().find(|(kk, _)| *kk == key) {
            Some((_, a)) => a.clone(),
            None if st.info.distinct_rewritten
                && !f.distinct
                && !matches!(super::exprs::builtin_name(f), "min" | "max") =>
            {
                // Spark rewrites this call with the select list's. Planned over the
                // rewrite's own rows (one per distinct set) it would count each group
                // once per set (`HAVING count(*) > 1` kept every group), so it is
                // refused until the rewrite takes calls from `HAVING` and `ORDER BY`.
                // A `DISTINCT` call, `min` and `max` do not read a row twice.
                return Err(unsupported(
                    "an aggregate in HAVING or ORDER BY that the select list does not hold, over DISTINCT aggregates of different arguments",
                    x.origin,
                ));
            }
            None => {
                let ty = super::ArgType {
                    data_type: Some(f.data_type.clone()),
                    nullable: f.nullable,
                };
                let output = self.attr(&super::exprs::pretty(x, self.ctx.conf.session_zone), &ty);
                st.grown
                    .aggregates
                    .push(Arc::new(Expr::new(ExprNode::Resolved(
                        ResolvedExpr::Alias {
                            expr: Arc::clone(x),
                            output: output.clone(),
                        },
                    ))));
                let a = Attr {
                    col: output,
                    qual: no_qual(),
                    source: None,
                    duplicate: false,
                };
                st.grown.out.push(a.clone());
                st.known.push((key, a.clone()));
                a
            }
        };
        if !st.visible.contains(&a.col.id) && !st.pass.iter().any(|p| p.col.id == a.col.id) {
            st.pass.push(a.clone());
        }
        Ok(a)
    }

    /// The subqueries of `bounds` (a `HAVING` condition) with their outer
    /// aggregates computed by the aggregate, then Spark's check that the
    /// grouping attributes they read are in the filter's input.
    pub(super) fn having_subqueries(
        &mut self,
        bounds: &mut [Bound],
        st: &mut Grouped<'_>,
        origin: Origin,
    ) -> R<()> {
        let input = Arc::clone(&st.info.input.out);
        for b in bounds.iter_mut() {
            b.expr = Arc::clone(&b.expr).transform_up(&mut |x: Arc<Expr>| match &x.node {
                ExprNode::Subquery { kind, plan } => {
                    let lifted = self.lift_outer_aggregates(plan, &input, st)?;
                    if Arc::ptr_eq(&lifted, plan) {
                        return Ok(x);
                    }
                    let node = ExprNode::Subquery {
                        kind: kind.clone(),
                        plan: lifted,
                    };
                    Ok(Arc::new(Expr::new(node).with_origin(x.origin)))
                }
                _ => Ok(x),
            })?;
        }
        self.check_having_reads(bounds, st, origin)
    }

    /// `plan` (a subquery of the `HAVING`) with each outer aggregate of its
    /// filters replaced by an outer reference to the aggregate's output.
    /// Aggregate calls of the subquery's own aggregate stay (Spark's
    /// `CORRELATED_REFERENCE`); so do nested subqueries, which never hold
    /// one (`outer_aggregates_only` reads the innermost level only).
    fn lift_outer_aggregates(
        &mut self,
        plan: &Arc<LogicalPlan>,
        input: &[Attr],
        st: &mut Grouped<'_>,
    ) -> R<Arc<LogicalPlan>> {
        Arc::clone(plan).transform_up(&mut |n: Arc<LogicalPlan>| {
            if !matches!(n.node, PlanNode::Filter { .. }) {
                return Ok(n);
            }
            n.map_expressions(|e| {
                e.transform_down(&mut |x: Arc<Expr>| {
                    if !is_aggregate(&x) || !is_outer_aggregate(&x, input) {
                        return Ok(x);
                    }
                    let local = as_local(&x);
                    // A fresh memo: `local` may be dropped, and the memo is
                    // keyed by address.
                    let key = semantic_key(&local, &mut HashMap::new());
                    let a = self.aggregate_output(&local, key, st)?;
                    Ok(Arc::new(
                        Expr::new(ExprNode::Resolved(ResolvedExpr::OuterColumn(a.col)))
                            .with_origin(x.origin),
                    ))
                })
            })
        })
    }

    /// Grouping attributes a subquery reads from here. A column of the
    /// aggregate's input that is neither grouped nor visible is not in the
    /// filter's input: Spark's `MISSING_ATTRIBUTES` (`HAVING t2c IN (SELECT ..
    /// HAVING t3b > t2b)` grouped by `t2c`). An outer column inside an
    /// aggregate call of the subquery (`EXISTS (SELECT first(t1.col2) = 0)`)
    /// is Spark's `CORRELATED_REFERENCE` instead, raised later.
    fn check_having_reads(
        &mut self,
        bounds: &[Bound],
        st: &mut Grouped<'_>,
        origin: Origin,
    ) -> R<()> {
        let mut aggregated: HashSet<AttrId> = HashSet::new();
        for b in bounds {
            let _ = visit_distinct(Node::Expr(&b.expr), |n| {
                if let Node::Expr(e) = n
                    && is_aggregate(e)
                {
                    let _ = visit_distinct(Node::Expr(e), |m| {
                        if let Node::Expr(x) = m
                            && let ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) = &x.node
                        {
                            aggregated.insert(c.id);
                        }
                        ControlFlow::Continue(())
                    });
                }
                ControlFlow::Continue(())
            });
        }
        let info = st.info;
        let mut missing: Option<String> = None;
        for b in bounds {
            let _ = visit_distinct(Node::Expr(&b.expr), |n| {
                if let Node::Expr(e) = n
                    && let ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) = &e.node
                    && !st.visible.contains(&c.id)
                    && !st.pass.iter().any(|a| a.col.id == c.id)
                {
                    if let Some(k) = info.keys.iter().find(|k| k.col.id == c.id) {
                        if !st.grown.out.iter().any(|o| o.col.id == c.id) {
                            st.grown.aggregates.push(col_expr(&k.col, Origin::UNKNOWN));
                            st.grown.out.push(k.clone());
                        }
                        st.pass.push(k.clone());
                    } else if missing.is_none()
                        && !aggregated.contains(&c.id)
                        && info.input.out.iter().any(|a| a.col.id == c.id)
                    {
                        missing = Some(c.name.clone());
                    }
                }
                ControlFlow::Continue(())
            });
        }
        if let Some(name) = missing {
            let input = info
                .out
                .iter()
                .map(|a| format!("\"{}\"", a.col.name))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(AnalysisError::new(
                MissingAttributesResolvedAttributeMissingFromInput {
                    missing_attributes: format!("\"{name}\""),
                    input,
                    operator: "Filter".to_owned(),
                },
                origin,
            ));
        }
        Ok(())
    }
}

/// The first aggregate call in `e` (outside subqueries) that reads both a
/// column of its own query and an outer reference: Spark's
/// `AGGREGATE_FUNCTION_MIXED_OUTER_LOCAL_REFERENCES`.
pub(super) fn mixed_aggregate(e: &Arc<Expr>) -> Option<Arc<Expr>> {
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&x)) {
            continue;
        }
        if is_aggregate(&x) {
            let (mut outer, mut local) = (false, false);
            let _ = visit_distinct(Node::Expr(&x), |n| {
                match n {
                    Node::Expr(y) => match &y.node {
                        ExprNode::Resolved(ResolvedExpr::OuterColumn(_)) => outer = true,
                        ExprNode::Resolved(ResolvedExpr::Column(_)) => local = true,
                        _ => {}
                    },
                    Node::Plan(_) => return ControlFlow::Break(()),
                }
                ControlFlow::Continue(())
            });
            if outer && local {
                return Some(x);
            }
            continue;
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    None
}

/// The aggregate call `x` as Spark's `sql` prints it in
/// `AGGREGATE_FUNCTION_MIXED_OUTER_LOCAL_REFERENCES`: columns of `scope`
/// fully qualified, outer references as `outer(<qualified name>)`.
pub(super) fn mixed_sql(x: &Arc<Expr>, scope: &[Attr], env: &Env) -> String {
    let local = super::ops::qualified(x, scope);
    let shown: Result<_, std::convert::Infallible> =
        local.transform_up(&mut |y: Arc<Expr>| match &y.node {
            ExprNode::Resolved(ResolvedExpr::OuterColumn(c)) => {
                let mut parts: Vec<String> = env
                    .outer
                    .iter()
                    .rev()
                    .flat_map(|l| l.scope.iter().chain(l.fallback.iter()))
                    .find(|a| a.col.id == c.id)
                    .map(|a| a.qual.to_vec())
                    .unwrap_or_default();
                parts.push(c.name.clone());
                let mut col = c.clone();
                col.name = format!("outer({})", parts.join("."));
                Ok(Arc::new(
                    Expr::new(ExprNode::Resolved(ResolvedExpr::OuterColumn(col)))
                        .with_origin(y.origin),
                ))
            }
            _ => Ok(y),
        });
    match shown {
        Ok(e) => super::exprs::pretty(&e, None),
    }
}
