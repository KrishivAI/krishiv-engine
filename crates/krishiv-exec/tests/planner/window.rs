// SPDX-License-Identifier: Apache-2.0
//! Window relations the planner must refuse rather than compute as
//! something else.

use std::convert::Infallible;
use std::sync::Arc;

use krishiv_common::error::ErrorClass;
use krishiv_exec::planner::{DefaultPlanner, PhysicalPlanner as _};
use krishiv_plan::{Expr, ExprNode, Plan, ResolvedExpr};

use super::env;

/// `count(DISTINCT x) OVER (...)` is an analysis error in Spark, so a
/// plan that reaches the planner with the flag set is malformed: it fails
/// instead of planning a plain `count`.
#[test]
fn a_distinct_window_aggregate_is_refused() {
    let env = env();
    let analyzed = env.analyze("SELECT count(a) OVER (PARTITION BY b) FROM t");
    let mut flagged = 0;
    let root = Arc::clone(analyzed.root())
        .transform_down(&mut |p| {
            p.map_expressions(|e| {
                e.transform_down(&mut |x: Arc<Expr>| {
                    let ExprNode::Resolved(ResolvedExpr::Function(f)) = &x.node else {
                        return Ok::<_, Infallible>(x);
                    };
                    flagged += 1;
                    let mut f = f.clone();
                    f.distinct = true;
                    Ok(Arc::new(Expr::new(ExprNode::Resolved(
                        ResolvedExpr::Function(f),
                    ))))
                })
            })
        })
        .unwrap_or_else(|never| match never {});
    assert_eq!(flagged, 1, "the window call is the only function");
    let plan = Plan::try_new(root).unwrap();
    let err = DefaultPlanner.plan(&plan, &env.cx()).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert!(
        err.to_string().contains("DISTINCT window aggregates"),
        "{err}"
    );
}
