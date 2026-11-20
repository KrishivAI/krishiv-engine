// SPDX-License-Identifier: Apache-2.0
//! `PlanNode::MetricViewRead` in the analyzer (design note `P3-metric-views.md` §4): the read's
//! output, Spark's `ResolveMetricView` over an aggregate (grouped, global, grouping sets, a
//! measure read outside `measure(..)`), the printer, and `refuse_unplannable`'s order. The Spark
//! goldens run end to end in `krishiv-session/tests/metric_views.rs`.

use std::sync::Arc;

use super::analyzer::{ctx, run};
use super::dsl::{c, f};
use crate::analyzer::{is_unplannable, refuse_unplannable};
use crate::{Expr, Grouping, LogicalPlan, MetricColumn, PlanNode};
use krishiv_types::DataType;

fn column(name: &str, expr: Expr, data_type: DataType) -> MetricColumn {
    MetricColumn {
        name: name.to_owned(),
        expr: Arc::new(expr),
        data_type,
        nullable: true,
    }
}

/// A metric view over `part`: dimension `brand` (`p_brand`), measure `total` (`sum(p_size)`).
fn read() -> LogicalPlan {
    LogicalPlan::new(PlanNode::MetricViewRead {
        input: Arc::new(LogicalPlan::table(["part"])),
        name: vec!["spark_catalog".into(), "default".into(), "mv".into()],
        dimensions: vec![column("brand", c("p_brand"), DataType::String)],
        measures: vec![column("total", f("sum", vec![c("p_size")]), DataType::Long)],
        output: Vec::new(),
    })
}

fn measure(name: &str) -> Expr {
    f("measure", vec![c(name)])
}

fn class(plan: LogicalPlan) -> String {
    run(plan, &ctx())
        .expect_err("must fail")
        .error
        .error_class()
        .name()
        .to_owned()
}

fn has_read(plan: &LogicalPlan) -> bool {
    let mut found = matches!(plan.node, PlanNode::MetricViewRead { .. });
    plan.for_each_child(&mut |c| found |= has_read(c));
    found
}

#[test]
fn a_read_outputs_the_stored_columns_and_prints() {
    assert_eq!(
        read().to_string(),
        "MetricViewRead spark_catalog.default.mv [brand = p_brand, total = sum(p_size)], []\n+- Read part\n"
    );
    let analyzed = run(read(), &ctx()).unwrap();
    let names: Vec<&str> = analyzed.output.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["brand", "total"]);
    assert!(has_read(analyzed.plan.root()));
    // Analyzable, never plannable.
    let e = refuse_unplannable(analyzed.plan.root()).unwrap_err();
    assert!(is_unplannable(&e));
    assert!(e.to_string().contains("phase planning failed"), "{e}");
}

#[test]
fn an_aggregate_over_a_read_computes_the_measure_over_the_source() {
    let plan = read().aggregate(vec![c("brand")], vec![c("brand"), measure("total")]);
    let analyzed = run(plan, &ctx()).unwrap();
    assert!(!has_read(analyzed.plan.root()), "{}", analyzed.plan.root());
    refuse_unplannable(analyzed.plan.root()).unwrap();
    let names: Vec<&str> = analyzed.output.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["brand", "measure(total)"]);
    assert_eq!(analyzed.output[1].data_type, DataType::Long);
    // A global aggregate (`SELECT measure(total) FROM mv`, a projection) too.
    let analyzed = run(read().project(vec![measure("total")]), &ctx()).unwrap();
    assert!(!has_read(analyzed.plan.root()));
}

#[test]
fn sparks_rule_errors_are_reproduced() {
    // A grouping-set expansion passes the measures on.
    let rollup = LogicalPlan::new(PlanNode::Aggregate {
        input: Arc::new(read()),
        grouping: Grouping::Rollup(vec![Arc::new(c("brand"))]),
        aggregates: vec![Arc::new(c("brand")), Arc::new(measure("total"))],
    });
    assert_eq!(
        class(rollup),
        "MISSING_ATTRIBUTES.RESOLVED_ATTRIBUTE_MISSING_FROM_INPUT"
    );
    // A measure read as a column (a filter under the aggregate) is missing once the read goes.
    let filtered = read()
        .filter(super::dsl::gt(c("total"), super::dsl::int(1)))
        .aggregate(vec![c("brand")], vec![measure("total")]);
    assert_eq!(
        class(filtered),
        "MISSING_ATTRIBUTES.RESOLVED_ATTRIBUTE_MISSING_FROM_INPUT"
    );
    // `measure(<dimension>)` is looked up among the measures.
    assert_eq!(
        class(read().project(vec![measure("brand")])),
        "INTERNAL_ERROR"
    );
}

#[test]
fn a_leftover_measure_is_refused_before_a_leftover_read() {
    // `measure(..)` over a plain table: no rule fires.
    let analyzed = run(
        LogicalPlan::table(["part"]).project(vec![f("measure", vec![c("p_size")])]),
        &ctx(),
    )
    .unwrap();
    let e = refuse_unplannable(analyzed.plan.root()).unwrap_err();
    assert!(e.to_string().contains("UnevaluableAggregateFunc"), "{e}");
    assert!(is_unplannable(&e));
}
