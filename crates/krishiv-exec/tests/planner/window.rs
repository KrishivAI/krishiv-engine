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

/// A window over a hash exchange runs over the sort's hash buckets, read by
/// several drivers: each partition is whole and in order, and `EXPLAIN
/// ANALYZE` shows the buckets.
#[test]
fn a_partitioned_window_reads_the_sort_buckets() {
    let env = env();
    let sql = "SELECT b, row_number() OVER (PARTITION BY b % 3 ORDER BY b DESC) AS r FROM t";
    let plan = env.plan(sql).unwrap();
    let root = krishiv_memory::TreePool::new(&krishiv_memory::MemoryConfig::default());
    // Four compute threads, whatever the machine: 16 buckets.
    let executor = krishiv_exec::ExecutorConfig {
        compute_threads: 4,
        io_threads: 1,
        blocking_threads: 2,
    };
    let (out, query) = super::run_on(&plan, &root, &executor);
    let mut rows: Vec<String> = Vec::new();
    for b in &out.unwrap() {
        for r in 0..b.num_rows() {
            let cells: Vec<String> = b
                .columns()
                .iter()
                .map(|c| super::cell(c.as_ref(), r))
                .collect();
            rows.push(cells.join("|"));
        }
    }
    rows.sort();
    // b % 3 = 0: 90, 60, 30, 0; 1: 70, 40, 10; 2: 80, 50, 20.
    let want = [
        "0|4", "10|3", "20|3", "30|3", "40|2", "50|2", "60|2", "70|1", "80|1", "90|1",
    ];
    assert_eq!(rows, want);
    let metrics = query.metrics();
    let buckets = metrics
        .operators
        .values()
        .find(|e| e.name == "Sort")
        .and_then(|e| e.extra("buckets").cloned());
    assert_eq!(buckets, Some(krishiv_exec::Metric::Max(16)));
}
