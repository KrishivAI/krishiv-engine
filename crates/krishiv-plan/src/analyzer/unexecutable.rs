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
    t.is_some_and(krishiv_types::DataType::contains_collation)
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

/// Whether a collated leaf of `t` has no key here: ICU collations without the `collations-icu`
/// feature, and an indeterminate one.
fn no_key(t: Option<&DataType>) -> bool {
    t.is_some_and(|t| {
        t.any_type(&|x| {
            matches!(x, DataType::CollatedString(c) if !matches!(
                c.family(),
                krishiv_types::CollationFamily::Utf8Binary
                    | krishiv_types::CollationFamily::Utf8Lcase
            ) && !(cfg!(feature = "collations-icu")
                && matches!(c.family(), krishiv_types::CollationFamily::Icu { .. }))
                || c.is_indeterminate())
        })
    })
}

/// Whether a collated leaf of `t` is an ICU collation.
fn has_icu(t: Option<&DataType>) -> bool {
    t.is_some_and(|t| {
        t.any_type(&|x| {
            matches!(x, DataType::CollatedString(c)
                if matches!(c.family(), krishiv_types::CollationFamily::Icu { .. }))
        })
    })
}

/// A collated string inside an array, map or struct: it has an equality key but no ordering yet.
fn nested_order(t: Option<&DataType>) -> bool {
    t.is_some_and(|t| t.contains_collation() && !t.is_collated_string())
}

/// Every aggregate call under `e` (not those inside another call's arguments).
fn aggregate_calls<'a>(e: &'a crate::Expr, out: &mut Vec<&'a crate::ResolvedFunction>) {
    if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node
        && f.kind == crate::FunctionKind::Aggregate
    {
        out.push(f);
        return;
    }
    e.for_each_child(&mut |c| aggregate_calls(c, out));
}

/// Whether `f` is a `DISTINCT` aggregate whose collated values must survive (`listagg`,
/// `collect_list`): the optimizer dedupes them by key in an aggregate of their own
/// (`krishiv_optimizer::collation_keys`), so the analyzer's distinct rewrite leaves it alone.
pub(super) fn keeps_collated(f: &crate::ResolvedFunction) -> bool {
    // `listagg` casts its value to a plain string: the collation is the child's.
    let value_collated = |a: &crate::Expr| match &a.node {
        ExprNode::Cast {
            expr,
            to: DataType::String,
            ..
        } if is_collated(expr_type(expr)) => true,
        _ => is_collated(expr_type(a)),
    };
    f.kind == crate::FunctionKind::Aggregate
        && f.distinct
        && !matches!(f.id.name(), "count" | "approx_count_distinct")
        && f.args.first().is_some_and(|a| value_collated(a))
}

/// A [`keeps_collated`] call beside another aggregate call or with a filter: the optimizer's
/// plan for it does not take either yet.
fn distinct_kept_beside(aggregates: &[std::sync::Arc<crate::Expr>]) -> bool {
    let mut calls = Vec::new();
    aggregates
        .iter()
        .for_each(|a| aggregate_calls(a, &mut calls));
    match calls.as_slice() {
        [one] => keeps_collated(one) && one.filter.is_some(),
        many => many.iter().any(|f| keeps_collated(f)),
    }
}

/// The execution guard of the collation units (ADR 0043, design `P2-collations.md` section 3.2),
/// the one table of what still has no key. Everything else runs on keys
/// (`krishiv_optimizer::collation_keys`, and `exprs/collation.rs` for scalars):
///
/// | Position | Refused | Owner |
/// |---|---|---|
/// | group, distinct, window partition | a leaf without a key | 6 |
/// | sort, window order, `min`/`max`/`min_by`/`max_by` ordering | a leaf without a key, nested | 6, later |
/// | `DISTINCT`, `listagg`, `approx_count_distinct` | a leaf without a key | 6 |
/// | `mode` | any collated string | 5 |
/// | `DISTINCT` `listagg`/`collect_list` beside another call or with a filter | any collated string | later |
/// | a comparison the analyzer did not key | any collated string | 6 |
fn collated_key(node: &Node<'_>) -> Option<(&'static str, crate::Origin)> {
    const KEY: &str = "a collated string as a grouping, distinct, sort, window or join key, a set \
                       operation, or a min/max/count(DISTINCT)/collect_set/listagg/mode argument";
    const COMPARISON: &str = "a comparison of collated strings";
    const DISTINCT: &str =
        "a DISTINCT aggregate of a collated string beside another aggregate or with a filter";
    let order_refused = |t: Option<&DataType>| no_key(t) || nested_order(t);
    match node {
        Node::Plan(p) => {
            let bad = match &p.node {
                PlanNode::Sort { order, .. } => {
                    order.iter().any(|o| order_refused(expr_type(&o.expr)))
                }
                PlanNode::Aggregate { aggregates, .. } if distinct_kept_beside(aggregates) => {
                    return Some((DISTINCT, p.origin));
                }
                PlanNode::Aggregate { grouping, .. } => match grouping {
                    crate::Grouping::GroupBy(e)
                    | crate::Grouping::Rollup(e)
                    | crate::Grouping::Cube(e) => e.iter().any(|x| no_key(expr_type(x))),
                    crate::Grouping::GroupingSets { exprs, .. } => {
                        exprs.iter().any(|x| no_key(expr_type(x)))
                    }
                },
                PlanNode::Window {
                    partition_by,
                    order_by,
                    ..
                } => {
                    partition_by.iter().any(|x| no_key(expr_type(x)))
                        || order_by.iter().any(|o| order_refused(expr_type(&o.expr)))
                }
                PlanNode::Distinct { input } => {
                    matches!(&input.node, PlanNode::Project { exprs, .. }
                    if exprs.iter().any(|x| no_key(expr_type(x))))
                }
                _ => false,
            };
            bad.then_some((KEY, p.origin))
        }
        Node::Expr(e) => {
            let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node else {
                return None;
            };
            if f.kind != crate::FunctionKind::Aggregate {
                let comparing = matches!(
                    f.id.name(),
                    "==" | "<=>"
                        | "<"
                        | "<="
                        | ">"
                        | ">="
                        | "in"
                        | "between"
                        | "greatest"
                        | "least"
                        | "nullif"
                        | "equal_null"
                ) && f.args.iter().any(|a| is_collated(expr_type(a)));
                return comparing.then_some((COMPARISON, e.origin));
            }
            // `listagg` casts its value to a plain string: the collation is the child's.
            let peeled = |a: &crate::Expr| match &a.node {
                ExprNode::Cast {
                    expr,
                    to: DataType::String,
                    ..
                } if is_collated(expr_type(expr)) => expr_type(expr).cloned(),
                _ => expr_type(a).cloned(),
            };
            let arg = |i: usize| f.args.get(i).and_then(|a| expr_type(a));
            let name = super::exprs::builtin_name(f);
            let bad = match name {
                "mode" => is_collated(arg(0)),
                "min" | "max" => order_refused(arg(0)),
                "min_by" | "max_by" => order_refused(arg(1)),
                "listagg" => f.args.iter().any(|a| no_key(peeled(a).as_ref())),
                // ADR 0043 Decision 6: a sketch over an ICU collation is refused even where the
                // feature gives ICU a key.
                "approx_count_distinct" => f
                    .args
                    .iter()
                    .any(|a| no_key(peeled(a).as_ref()) || has_icu(peeled(a).as_ref())),
                _ => f.distinct && f.args.iter().any(|a| no_key(expr_type(a))),
            };
            bad.then_some((KEY, e.origin))
        }
    }
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
                    && (f.distinct || names.contains(&super::exprs::builtin_name(f)))
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
