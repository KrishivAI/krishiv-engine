// SPDX-License-Identifier: Apache-2.0
//! Types the engine parses, maps and serializes but cannot evaluate yet
//! (`TIME`, `GEOMETRY`, `GEOGRAPHY`): a resolved plan that holds one
//! anywhere (a scanned column, a cast, a typed `NULL`, a function result)
//! fails with `KRISHIV_UNSUPPORTED_FEATURE` before it reaches a planner.

use std::ops::ControlFlow;

use krishiv_common::error::UnsupportedTimeType;
use krishiv_types::DataType;

use super::AnalysisError;
use super::bind::unsupported;
use crate::walk::{Node, visit_distinct};
use crate::{ExprNode, Literal, LogicalPlan, PlanNode, ResolvedExpr};

/// The feature name of the `TIME` type.
const TIME: &str = "the TIME data type";

fn feature(t: &DataType) -> Option<&'static str> {
    match t {
        DataType::Time { .. } => Some(TIME),
        DataType::Geometry { .. } => Some("the GEOMETRY data type"),
        DataType::Geography { .. } => Some("the GEOGRAPHY data type"),
        _ => None,
    }
}

/// Fails on the first node of `root` (in pre-order) that holds a type the
/// engine cannot evaluate.
pub(super) fn check(root: &LogicalPlan) -> Result<(), AnalysisError> {
    let mut error = None;
    let _ = visit_distinct(Node::Plan(root), |node| {
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
        match types.into_iter().find_map(feature) {
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
