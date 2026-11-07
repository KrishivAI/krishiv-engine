// SPDX-License-Identifier: Apache-2.0
//! Spark's legacy decorrelation of `EXISTS` / `IN` subqueries over an
//! aggregate, under
//! `spark.sql.optimizer.decorrelateExistsSubqueryLegacyIncorrectCountHandling.enabled`
//! (default off; Spark 4.0+).
//!
//! With the flag, Spark 4.2 decorrelates these subqueries with
//! `PullupCorrelatedPredicates.pullOutCorrelatedPredicates` instead of
//! `DecorrelateInnerQuery` (`subquery.scala`, `rewriteSubQueries`): the
//! correlated conjuncts of the filter below a global aggregate move above
//! it, and the aggregate groups by the inner columns they read. So
//! `EXISTS (SELECT count(*) FROM t2 WHERE t2.c1 = t1.c1)` holds only for
//! the rows of `t1` with a match, where the correct answer is every row
//! (a count over no rows is still a row). The answers this gives are
//! Spark's (`exists-count-bug.sql`, `in-count-bug.sql`).

use std::sync::Arc;

use super::AnalysisError;
use super::bind::unsupported;
use super::subquery::{and_all, conjuncts, has_outer, inner_columns, is_outer_equality, node};
use crate::walk::Node;
use crate::{Grouping, LogicalPlan, Origin, PlanNode};

/// `plan`, an `EXISTS` / `IN` subquery's plan under the flag, in the
/// legacy shape ([`pull_through_aggregate`]), or as it is when it has no
/// aggregate or limit over its correlation. Spark's legacy rule raises or
/// answers differently for the shapes it cannot reshape here (a `LIMIT` over
/// the correlation, a correlation that is not an equality under an
/// aggregate, one below a projection under it): those are refused, never
/// answered the non-legacy way.
pub(super) fn legacy_shape(
    plan: &Arc<LogicalPlan>,
    origin: Origin,
) -> Result<Arc<LogicalPlan>, AnalysisError> {
    if let Some(shaped) = pull_through_aggregate(plan) {
        return Ok(shaped);
    }
    let mut refused = false;
    let mut stack = vec![Arc::clone(plan)];
    while let Some(p) = stack.pop() {
        if let PlanNode::Limit { input, .. } | PlanNode::Aggregate { input, .. } = &p.node
            && has_outer(Node::Plan(input))
        {
            refused = true;
            break;
        }
        p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    if refused {
        return Err(unsupported(
            "this correlated EXISTS or IN under \
             spark.sql.optimizer.decorrelateExistsSubqueryLegacyIncorrectCountHandling.enabled",
            origin,
        ));
    }
    Ok(Arc::clone(plan))
}

/// `plan`, an `EXISTS` / `IN` subquery's plan, in the legacy shape: when
/// it is a correlated global aggregate (under projections, aliases and a
/// `HAVING` filter) over a filter with correlated conjuncts, those
/// conjuncts filter the aggregate's result instead, the aggregate grouped
/// by the inner columns they read, and a projection keeps the aggregate's
/// columns. `None` for any other plan: it decorrelates as without the flag.
pub(super) fn pull_through_aggregate(plan: &Arc<LogicalPlan>) -> Option<Arc<LogicalPlan>> {
    match &plan.node {
        PlanNode::Project { input, .. } => with_input(plan, pull_through_aggregate(input)?),
        PlanNode::SubqueryAlias { input, columns, .. } if columns.is_empty() => {
            with_input(plan, pull_through_aggregate(input)?)
        }
        PlanNode::Filter { input, predicate } if !has_outer(Node::Expr(predicate)) => {
            with_input(plan, pull_through_aggregate(input)?)
        }
        PlanNode::Aggregate {
            input,
            grouping: Grouping::GroupBy(keys),
            aggregates,
        } if keys.is_empty() => {
            let PlanNode::Filter {
                input: below,
                predicate,
            } = &input.node
            else {
                return None;
            };
            let (correlated, local): (Vec<_>, Vec<_>) = conjuncts(predicate)
                .into_iter()
                .partition(|c| has_outer(Node::Expr(c)));
            // Spark's legacy rule refuses a correlation that is not an
            // equality under an aggregate.
            if correlated.is_empty()
                || !correlated
                    .iter()
                    .all(|c| is_outer_equality(c) || inner_columns(c).is_empty())
            {
                return None;
            }
            let out = plan.resolved_output()?;
            let mut grouping = Vec::new();
            let mut cols = Vec::new();
            for c in correlated.iter().flat_map(inner_columns) {
                if !cols.iter().any(|o: &crate::BoundColumn| o.id == c.id) {
                    grouping.push(super::ops::col_expr(&c, Origin::UNKNOWN));
                    cols.push(c);
                }
            }
            let input = match and_all(local) {
                Some(predicate) => node(
                    PlanNode::Filter {
                        input: Arc::clone(below),
                        predicate,
                    },
                    input.origin,
                ),
                None => Arc::clone(below),
            };
            let mut items = aggregates.clone();
            items.extend(grouping.iter().cloned());
            let grouped = node(
                PlanNode::Aggregate {
                    input,
                    grouping: Grouping::GroupBy(grouping),
                    aggregates: items,
                },
                plan.origin,
            );
            let filtered = node(
                PlanNode::Filter {
                    input: grouped,
                    predicate: and_all(correlated)?,
                },
                Origin::UNKNOWN,
            );
            let exprs = out
                .iter()
                .map(|c| super::ops::col_expr(c, Origin::UNKNOWN))
                .collect();
            Some(node(
                PlanNode::Project {
                    input: filtered,
                    exprs,
                },
                plan.origin,
            ))
        }
        _ => None,
    }
}

/// `p`, a node with one input, over `input`.
fn with_input(p: &Arc<LogicalPlan>, input: Arc<LogicalPlan>) -> Option<Arc<LogicalPlan>> {
    let mut input = Some(input);
    Arc::clone(p)
        .map_children(|c| Ok::<_, std::convert::Infallible>(input.take().unwrap_or(c)))
        .ok()
}
