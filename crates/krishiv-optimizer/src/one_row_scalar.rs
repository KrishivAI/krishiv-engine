// SPDX-License-Identifier: Apache-2.0
//! `OneRowAbove`: a `max`/`min` of a column of a one-row relation, in a grouped aggregate over a
//! cross join with that relation, is the column itself, read above the aggregate.
//!
//! TPC-DS q23's `best_ss_customer` is `FROM store_sales, customer, max_store_sales GROUP BY
//! c_customer_sk HAVING sum(..) > 0.5 * max(tpcds_cmax)`, where `max_store_sales` is a global
//! aggregate (one row). The cross join copies its one value into every one of the 27 million
//! joined rows, the aggregate takes the `max` of 27 million equal values per group, and the
//! executor pays a nested-loop join, a projection and a `max` per group for it.
//!
//! The rewrite is exact when
//! - the aggregate has at least one grouping key (`GroupBy`, not `ROLLUP`/`CUBE`/grouping sets),
//!   so every group holds at least one row of the join; a global aggregate over an empty input
//!   returns a NULL `max` where the column is not NULL;
//! - the join is an inner join without a condition and one side is a global aggregate (`GROUP BY`
//!   nothing) under projections, shifts and aliases: it returns exactly one row, whatever its
//!   input holds, so the join repeats it per row of the other side and drops no row;
//! - the call is a plain `max` or `min` (no `DISTINCT`, no `FILTER`) of a bare column of the
//!   one-row side, of its own type; and no other item of the aggregate reads that side.
//!
//! Then the aggregate runs over the other side alone, and the one-row side joins its output; a
//! projection restores the original output attributes, each rewritten call now the column. The
//! rewritten plan has no such call left, so the rule is a fixed point.

use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{
    BoundColumn, Expr, ExprNode, FunctionKind, Grouping, JoinCondition, JoinKind, LogicalPlan,
    PlanNode, ResolvedExpr,
};

use crate::dag;
use crate::expr::{column, output_of, references};

/// Rewrites every such aggregate of `root` (subquery plans included).
///
/// # Errors
/// As [`dag::rewrite_up`].
pub(crate) fn rewrite(root: Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, KrishivError> {
    dag::rewrite_up(root, &mut |p| Ok(rewrite_aggregate(&p).unwrap_or(p)))
}

/// Whether `p` returns exactly one row: a global aggregate under projections, shifts and aliases.
fn one_row(p: &LogicalPlan) -> bool {
    match &p.node {
        PlanNode::Aggregate {
            grouping: Grouping::GroupBy(keys),
            ..
        } => keys.is_empty(),
        PlanNode::Project { input, .. } | PlanNode::Shift { input, .. } => one_row(input),
        PlanNode::SubqueryAlias { input, columns, .. } if columns.is_empty() => one_row(input),
        _ => false,
    }
}

/// `max(c)` or `min(c)` of a bare column `c` among `one`'s attributes: that column.
fn constant_call<'a>(item: &'a Expr, one: &'a [BoundColumn]) -> Option<&'a BoundColumn> {
    let ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) = &item.node else {
        return None;
    };
    let ExprNode::Resolved(ResolvedExpr::Function(f)) = &expr.node else {
        return None;
    };
    if f.kind != FunctionKind::Aggregate
        || !matches!(f.id.name(), "max" | "min")
        || f.distinct
        || f.filter.is_some()
    {
        return None;
    }
    let [arg] = f.args.as_slice() else {
        return None;
    };
    let ExprNode::Resolved(ResolvedExpr::Column(c)) = &arg.node else {
        return None;
    };
    one.iter()
        .find(|o| o.id == c.id)
        .filter(|o| o.data_type == f.data_type)
}

fn rewrite_aggregate(p: &Arc<LogicalPlan>) -> Option<Arc<LogicalPlan>> {
    let PlanNode::Aggregate {
        input,
        grouping: grouping @ Grouping::GroupBy(keys),
        aggregates,
    } = &p.node
    else {
        return None;
    };
    if keys.is_empty() {
        return None;
    }
    let PlanNode::Join {
        left,
        right,
        kind: JoinKind::Inner,
        condition: JoinCondition::None,
    } = &input.node
    else {
        return None;
    };
    // Either side may be the one-row relation.
    let (many, one, one_first) = if one_row(right) && !one_row(left) {
        (left, right, false)
    } else if one_row(left) && !one_row(right) {
        (right, left, true)
    } else {
        return None;
    };
    let one_out = one.resolved_output()?;
    let mut kept: Vec<Arc<Expr>> = Vec::new();
    let mut project: Vec<Arc<Expr>> = Vec::new();
    let mut moved = false;
    for item in aggregates {
        let out = output_of(item)?;
        if let Some(c) = constant_call(item, &one_out) {
            moved = true;
            project.push(Arc::new(Expr::new(ExprNode::Resolved(
                ResolvedExpr::Alias {
                    expr: column(c),
                    output: out.clone(),
                },
            ))));
        } else if references(item)
            .iter()
            .any(|a| one_out.iter().any(|o| o.id == *a))
        {
            return None;
        } else {
            kept.push(Arc::clone(item));
            project.push(column(out));
        }
    }
    if !moved {
        return None;
    }
    let aggregate = Arc::new(LogicalPlan::new(PlanNode::Aggregate {
        input: Arc::clone(many),
        grouping: grouping.clone(),
        aggregates: kept,
    }));
    let (l, r) = if one_first {
        (Arc::clone(one), aggregate)
    } else {
        (aggregate, Arc::clone(one))
    };
    let join = Arc::new(LogicalPlan::new(PlanNode::Join {
        left: l,
        right: r,
        kind: JoinKind::Inner,
        condition: JoinCondition::None,
    }));
    let mut out = LogicalPlan::new(PlanNode::Project {
        input: join,
        exprs: project,
    });
    out.origin = p.origin;
    Some(Arc::new(out))
}
