// SPDX-License-Identifier: Apache-2.0
//! E2 (P3 spike, branch only): planning speed at Connect scale. A chain of `withColumn` steps over a
//! three-column table, each step referencing the previous DataFrame by `plan_id`, as the PySpark client
//! sends it. Measures the analysis of the finished chain and the client loop that calls `df.columns` after
//! every step (one analysis per step). Run: `cargo test -p krishiv-plan --release e2_ -- --ignored --nocapture`.

use std::time::Instant;

use crate::analyzer::{Analyzer, DefaultAnalyzer};
use crate::{ColumnRef, Expr, ExprNode, LogicalPlan, Origin, Plan, PlanId, PlanNode};

use super::dsl::{add, int};
use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{MemoryCatalog, ResolutionConf, TableColumn};

fn ctx() -> crate::analyzer::AnalysisContext {
    let cols = ["a", "b", "c"]
        .iter()
        .map(|n| TableColumn::new((*n).to_owned(), krishiv_types::DataType::Integer, true))
        .collect();
    context(
        MemoryCatalog::new("spark_catalog").with_table(&["default"], "t", cols, None),
        ResolutionConf::default(),
    )
}

fn col(name: &str, id: i64) -> Expr {
    Expr::new(ExprNode::Column(ColumnRef {
        plan_id: Some(PlanId(id)),
        ..ColumnRef::new(vec![name.to_owned()])
    }))
}

/// The chain after `steps` `withColumn` calls; step `i` adds `c{i} = <previous column> + 1`.
fn chain(steps: i64) -> LogicalPlan {
    let mut plan = LogicalPlan::table(["t"]).with_origin(Origin::plan(PlanId(0)));
    for i in 1..=steps {
        let prev = if i == 1 {
            "a".to_owned()
        } else {
            format!("c{}", i - 1)
        };
        let expr = add(col(&prev, i - 1), int(1));
        plan = LogicalPlan::new(PlanNode::WithColumns {
            input: std::sync::Arc::new(plan),
            columns: vec![(format!("c{i}"), std::sync::Arc::new(expr))],
        })
        .with_origin(Origin::plan(PlanId(i)));
    }
    plan
}

/// The `p` percentile (`p` in per cent) by rank.
fn pct(v: &mut [f64], p: usize) -> f64 {
    v.sort_by(f64::total_cmp);
    v[(v.len() * p).div_ceil(100).clamp(1, v.len()) - 1]
}

fn analyze_ms(plan: &LogicalPlan, c: &crate::analyzer::AnalysisContext) -> (f64, f64) {
    let t = Instant::now();
    let p = Plan::try_new(plan.clone()).unwrap();
    let build = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let a = block_on(DefaultAnalyzer.analyze(&p, c)).unwrap();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    std::hint::black_box(a);
    (build, ms)
}

#[test]
#[ignore = "timed spike; run with --release --ignored --nocapture"]
fn e2_planning_speed() {
    let c = ctx();
    let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    println!("host load before: {}", load.trim());
    for depth in [10, 50, 100, 200, 400] {
        let plan = chain(depth);
        let (mut b, mut a): (Vec<f64>, Vec<f64>) = (0..200).map(|_| analyze_ms(&plan, &c)).unzip();
        println!(
            "depth {depth:>3}: Plan::try_new p50 {:.3} p99 {:.3} ms | analyze p50 {:.3} p99 {:.3} ms",
            pct(&mut b, 50),
            pct(&mut b, 99),
            pct(&mut a, 50),
            pct(&mut a, 99)
        );
    }
    // The client loop: withColumn then df.columns, 200 times; per-call analysis latency at each depth.
    let mut per_call = Vec::new();
    let t = Instant::now();
    for i in 1..=200 {
        per_call.push(analyze_ms(&chain(i), &c).1);
    }
    println!(
        "loop of 200 (withColumn + df.columns): total {:.1} ms, call p50 {:.3} p99 {:.3} max {:.3} ms",
        t.elapsed().as_secs_f64() * 1e3,
        pct(&mut per_call.clone(), 50),
        pct(&mut per_call.clone(), 99),
        per_call.iter().cloned().fold(0.0, f64::max)
    );
    println!(
        "host load after: {}",
        std::fs::read_to_string("/proc/loadavg")
            .unwrap_or_default()
            .trim()
    );
}
