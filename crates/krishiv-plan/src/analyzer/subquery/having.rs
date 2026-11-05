// SPDX-License-Identifier: Apache-2.0
//! `(SELECT agg .. WHERE <correlation> HAVING pred(aggs))`: a global aggregate under a `HAVING`.
//!
//! The aggregate has one row for every outer row, and the `HAVING` can reject it (no row, so a NULL value). The
//! `HAVING` may test other aggregates than the one selected (`SELECT count(a) .. HAVING count(*) > -1`): each is
//! decorrelated on its own, as a scalar subquery, and the predicate reads them all.

use std::collections::HashMap;
use std::sync::Arc;

use super::{R, Rewriter, call, has_outer, has_subquery, node, unsupported};
use crate::walk::Node;
use crate::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, Grouping, Literal, LogicalPlan, PlanNode,
    ResolvedExpr,
};

/// `plan` as a global aggregate under a `HAVING` (a filter on its values, possibly under a projection of the one
/// value it selects): the aggregate, the predicate and the selected value's column.
pub(super) fn having_over_global(
    plan: &Arc<LogicalPlan>,
) -> Option<(Arc<LogicalPlan>, Arc<Expr>, BoundColumn)> {
    let (filter, selected) = match &plan.node {
        PlanNode::Project { input, exprs } if exprs.len() == 1 => {
            // The projection must be one of the aggregate's values itself.
            let ExprNode::Resolved(ResolvedExpr::Column(c)) = &exprs[0].node else {
                return None;
            };
            (input, Some(c.id))
        }
        PlanNode::Filter { .. } => (plan, None),
        _ => return None,
    };
    let PlanNode::Filter { input, predicate } = &filter.node else {
        return None;
    };
    let PlanNode::Aggregate {
        grouping: Grouping::GroupBy(keys),
        aggregates,
        ..
    } = &input.node
    else {
        return None;
    };
    if !keys.is_empty() {
        return None;
    }
    let outputs: Vec<&BoundColumn> = aggregates
        .iter()
        .map(|item| match &item.node {
            ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => Some(output),
            _ => None,
        })
        .collect::<Option<_>>()?;
    let value = match (selected, &outputs[..]) {
        (Some(id), _) => outputs.iter().find(|o| o.id == id)?,
        (None, [only]) => only,
        _ => return None,
    };
    Some((Arc::clone(input), Arc::clone(predicate), (*value).clone()))
}

impl Rewriter {
    /// `(SELECT agg FROM .. WHERE <correlation> HAVING pred(aggs))`: the selected aggregate's value per outer row,
    /// NULL where the `HAVING` rejects it (a global aggregate has one row, so a rejected row is no row). A count
    /// over no matches is 0 and meets the `HAVING` as such.
    pub(super) fn scalar_having(
        &mut self,
        mut base: Arc<LogicalPlan>,
        agg: &Arc<LogicalPlan>,
        predicate: &Arc<Expr>,
        value: &BoundColumn,
    ) -> R<(Arc<LogicalPlan>, Arc<Expr>)> {
        let shape = || {
            unsupported(
                "a correlated scalar subquery of this shape",
                predicate.origin,
            )
        };
        let PlanNode::Aggregate {
            input,
            grouping,
            aggregates,
        } = &agg.node
        else {
            return Err(shape());
        };
        // Each aggregate over the same correlated input, read per outer row.
        let mut read: HashMap<AttrId, Arc<Expr>> = HashMap::new();
        for item in aggregates {
            let ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) = &item.node else {
                return Err(shape());
            };
            let single = node(
                PlanNode::Aggregate {
                    input: Arc::clone(input),
                    grouping: grouping.clone(),
                    aggregates: vec![Arc::clone(item)],
                },
                agg.origin,
            );
            let (joined, v) = self.scalar(base, &single)?;
            base = joined;
            read.insert(output.id, v);
        }
        let mut other = false;
        let pred = Arc::clone(predicate).transform_up(&mut |x: Arc<Expr>| {
            Ok::<_, std::convert::Infallible>(match &x.node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) => match read.get(&c.id) {
                    Some(v) => Arc::clone(v),
                    None => {
                        other = true;
                        x
                    }
                },
                _ => x,
            })
        });
        let pred = pred.unwrap_or_else(|never| match never {});
        let Some(v) = read.get(&value.id) else {
            return Err(shape());
        };
        if other || has_outer(Node::Expr(&pred)) || has_subquery(Node::Expr(&pred)) {
            return Err(shape());
        }
        let null = Arc::new(Expr::literal(Literal::Null(Some(value.data_type.clone()))));
        let guarded = call(
            Builtin::When,
            vec![pred, Arc::clone(v), null],
            value.data_type.clone(),
            true,
        );
        Ok((base, guarded))
    }
}
