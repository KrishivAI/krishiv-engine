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
