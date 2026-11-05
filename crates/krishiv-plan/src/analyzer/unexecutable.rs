// SPDX-License-Identifier: Apache-2.0
//! Types the engine parses, maps and serializes but cannot evaluate yet
//! (`TIME` while its gate is off): a resolved plan that holds one
//! anywhere (a scanned column, a cast, a typed `NULL`, a function result)
//! fails with `KRISHIV_UNSUPPORTED_FEATURE` before it reaches a planner.

use std::ops::ControlFlow;

use krishiv_common::error::{UnsupportedGeneratorOutsideSelect, UnsupportedTimeType};
use krishiv_types::DataType;

use super::AnalysisError;
use super::bind::unsupported;
use crate::walk::{Node, visit_distinct};
use crate::{ExprNode, Literal, LogicalPlan, PlanNode, ResolvedExpr};

/// The feature name of the `TIME` type.
const TIME: &str = "the TIME data type";

fn feature(t: &DataType, time_type: bool) -> Option<&'static str> {
    // The first unevaluable type in the type or inside it (an array
    // element, a map key or value, a struct field).
    (!time_type && t.any_type(&|x| matches!(x, DataType::Time { .. }))).then_some(TIME)
}

/// The type of a resolved expression, when it is a column or a call.
fn expr_type(e: &crate::Expr) -> Option<&DataType> {
    match &e.node {
        ExprNode::Literal(Literal::Null(Some(t))) | ExprNode::Cast { to: t, .. } => Some(t),
        ExprNode::Resolved(
            ResolvedExpr::Column(c)
            | ResolvedExpr::OuterColumn(c)
            | ResolvedExpr::Alias { output: c, .. },
        ) => Some(&c.data_type),
        ExprNode::Resolved(ResolvedExpr::Function(f)) => Some(&f.data_type),
        ExprNode::Resolved(ResolvedExpr::Field { data_type, .. }) => Some(data_type),
        _ => None,
    }
}

fn is_spatial(t: Option<&DataType>) -> bool {
    t.is_some_and(|t| {
        t.any_type(&|x| matches!(x, DataType::Geometry { .. } | DataType::Geography { .. }))
    })
}

fn is_collated(t: Option<&DataType>) -> bool {
    t.is_some_and(|t| t.any_type(&|x| matches!(x, DataType::CollatedString(_))))
}

/// The use of a spatial value as a key at `node`, with its origin.
fn spatial_key(node: &Node<'_>) -> Option<(&'static str, crate::Origin)> {
    const KEY: &str =
        "a GEOMETRY or GEOGRAPHY value as a grouping, distinct, window or min/max key";
    key_use(
        node,
        is_spatial,
        KEY,
        &["min", "max", "min_by", "max_by", "mode", "collect_set"],
    )
}

/// The execution guard of the collation units (ADR 0043, design P2-collations section 3.2):
/// until the planner rewrites the comparing positions onto collation keys, a collated value
/// there would be compared as bytes, silently wrong. Unit 3a lifts the scalar comparisons,
/// unit 3b the keys of aggregates, joins, sorts, windows and set operations.
fn collated_key(node: &Node<'_>) -> Option<(&'static str, crate::Origin)> {
    const KEY: &str = "a collated string as a grouping, distinct, sort, window or join key, a set \
                       operation, or a min/max/count(DISTINCT)/collect_set/listagg/mode argument";
    const COMPARISON: &str = "a comparison of collated strings";
    if let Node::Plan(p) = node
        && let PlanNode::Sort { order, .. } = &p.node
        && order.iter().any(|o| is_collated(expr_type(&o.expr)))
    {
        return Some((KEY, p.origin));
    }
    if let Node::Expr(e) = node
        && let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node
        && f.kind != crate::FunctionKind::Aggregate
        && matches!(
            f.id.name(),
            "=" | "<=>"
                | "<"
                | "<="
                | ">"
                | ">="
                | "in"
                | "between"
                | "greatest"
                | "least"
                | "nullif"
        )
        && f.args.iter().any(|a| is_collated(expr_type(a)))
    {
        return Some((COMPARISON, e.origin));
    }
    key_use(
        node,
        is_collated,
        KEY,
        &[
            "min",
            "max",
            "min_by",
            "max_by",
            "mode",
            "collect_set",
            "listagg",
            "approx_count_distinct",
        ],
    )
}

/// The use of a value of a type `held` accepts as a key at `node`, with its origin: a grouping,
/// distinct or window key, or an argument of a comparing aggregate (`names`, or any `DISTINCT`
/// one).
fn key_use(
    node: &Node<'_>,
    held: fn(Option<&DataType>) -> bool,
    key: &'static str,
    names: &[&str],
) -> Option<(&'static str, crate::Origin)> {
    match node {
        Node::Plan(p) => match &p.node {
            PlanNode::Aggregate { grouping, .. } => {
                let exprs: Vec<&std::sync::Arc<crate::Expr>> = match grouping {
                    crate::Grouping::GroupBy(e)
                    | crate::Grouping::Rollup(e)
                    | crate::Grouping::Cube(e) => e.iter().collect(),
                    crate::Grouping::GroupingSets { exprs, .. } => exprs.iter().collect(),
                };
                exprs
                    .iter()
                    .any(|e| held(expr_type(e)))
                    .then_some((key, p.origin))
            }
            // A window partitions and orders by its keys.
            PlanNode::Window {
                partition_by,
                order_by,
                ..
            } => partition_by
                .iter()
                .chain(order_by.iter().map(|o| &o.expr))
                .any(|e| held(expr_type(e)))
                .then_some((key, p.origin)),
            PlanNode::Distinct { input } => match &input.node {
                PlanNode::Project { exprs, .. } => exprs
                    .iter()
                    .any(|e| held(expr_type(e)))
                    .then_some((key, p.origin)),
                _ => None,
            },
            _ => None,
        },
        Node::Expr(e) => match &e.node {
            ExprNode::Resolved(ResolvedExpr::Function(f))
                if f.kind == crate::FunctionKind::Aggregate
                    && (f.distinct || names.contains(&f.id.name()))
                    && f.args.iter().any(|a| held(expr_type(a))) =>
            {
                Some((key, e.origin))
            }
            _ => None,
        },
    }
}

/// Fails on the first node of `root` (in pre-order) that holds a type the
/// engine cannot evaluate.
pub(super) fn check(root: &LogicalPlan, time_type: bool) -> Result<(), AnalysisError> {
    let mut error = None;
    let _ = visit_distinct(Node::Plan(root), |node| {
        // A generator the select-list rewrite did not take (in a filter, a
        // sort key, a grouping, an aggregate argument, ...) is outside the
        // SELECT clause, which Spark rejects at analysis.
        if let Node::Expr(e) = node
            && let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node
            && f.kind == crate::FunctionKind::Generator
        {
            error = Some(AnalysisError::new(
                UnsupportedGeneratorOutsideSelect {
                    plan: format!("'{}'", super::exprs::pretty(e, None)),
                },
                e.origin,
            ));
            return ControlFlow::Break(());
        }
        // A geometry or geography neither compares nor orders (Spark's
        // `INVALID_ORDERING_TYPE`), so it is no grouping key, distinct row or
        // `min`/`max` argument; the operators below would compare its
        // internal struct.
        if let Some((what, origin)) = spatial_key(&node).or_else(|| collated_key(&node)) {
            error = Some(unsupported(what, origin));
            return ControlFlow::Break(());
        }
        let (types, origin) = match node {
            Node::Plan(p) => match &p.node {
                PlanNode::Scan(scan) => {
                    (scan.output.iter().map(|c| &c.data_type).collect(), p.origin)
                }
                _ => (Vec::new(), p.origin),
            },
            Node::Expr(e) => {
                let t = match &e.node {
                    ExprNode::Literal(Literal::Null(Some(t))) | ExprNode::Cast { to: t, .. } => {
                        Some(t)
                    }
                    ExprNode::Resolved(
                        ResolvedExpr::Column(c)
                        | ResolvedExpr::OuterColumn(c)
                        | ResolvedExpr::Alias { output: c, .. },
                    ) => Some(&c.data_type),
                    ExprNode::Resolved(ResolvedExpr::Function(f)) => Some(&f.data_type),
                    _ => None,
                };
                (t.into_iter().collect(), e.origin)
            }
        };
        match types.into_iter().find_map(|t| feature(t, time_type)) {
            // Spark rejects `TIME` with its own class while
            // `spark.sql.timeType.enabled` is false (the default; the type
            // cannot be evaluated here either way).
            Some(TIME) => {
                error = Some(AnalysisError::new(UnsupportedTimeType {}, origin));
                ControlFlow::Break(())
            }
            Some(f) => {
                error = Some(unsupported(f, origin));
                ControlFlow::Break(())
            }
            None => ControlFlow::Continue(()),
        }
    });
    error.map_or(Ok(()), Err)
}
