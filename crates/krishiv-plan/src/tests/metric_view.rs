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
        ansi: None,
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
    // The fixture's declared `sum(p_size)` over the source's own `p_size` is the oracle.
    let top = |a: &crate::analyzer::AnalyzedPlan| {
        a.plan
            .root()
            .to_string()
            .lines()
            .next()
            .unwrap_or_default()
            .to_owned()
    };
    assert_eq!(
        top(&analyzed),
        "Aggregate [brand#13], [brand#13, sum(p_size#5) AS measure(total)#14]"
    );
    // A global aggregate (`SELECT measure(total) FROM mv`, a projection) too.
    let analyzed = run(read().project(vec![measure("total")]), &ctx()).unwrap();
    assert!(!has_read(analyzed.plan.root()));
    assert!(
        top(&analyzed).ends_with("[sum(p_size#5) AS measure(total)#15]"),
        "{}",
        analyzed.plan.root()
    );
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

/// A read over a source that repeats a name (#941 review 49995de79). A dimension or measure that
/// names it is ambiguous, as Spark 4.2.1's `CREATE` over `SELECT 1 AS a, 2 AS a` is
/// (`[AMBIGUOUS_REFERENCE] Reference `a` is ambiguous, could be: [`a`, `a`]. SQLSTATE: 42704`,
/// for `sum(a)` and for `a + 0`); one that does not reads (`count(1)` gives `(1)`). The session
/// refuses such a view at `CREATE`, so only this test reaches the read's branch.
#[test]
fn a_repeated_source_name_is_ambiguous_only_where_named() {
    let over = |dimensions: Vec<MetricColumn>, measures: Vec<MetricColumn>| {
        LogicalPlan::new(PlanNode::MetricViewRead {
            input: Arc::new(
                LogicalPlan::table(["part"])
                    .project(vec![c("p_size").alias("a"), c("p_partkey").alias("a")]),
            ),
            name: vec!["spark_catalog".into(), "default".into(), "mv".into()],
            dimensions,
            measures,
            output: Vec::new(),
            ansi: None,
        })
    };
    let count = || column("c", f("count", vec![super::dsl::int(1)]), DataType::Long);
    let sum = over(
        Vec::new(),
        vec![column("s", f("sum", vec![c("a")]), DataType::Long)],
    );
    assert_eq!(
        class(sum.project(vec![measure("s")])),
        "AMBIGUOUS_REFERENCE"
    );
    let dim = over(
        vec![column(
            "d",
            super::dsl::add(c("a"), super::dsl::int(0)),
            DataType::Integer,
        )],
        vec![count()],
    );
    assert_eq!(
        class(dim.aggregate(vec![c("d")], vec![c("d"), measure("c")])),
        "AMBIGUOUS_REFERENCE"
    );
    let plain = over(Vec::new(), vec![count()]).project(vec![measure("c")]);
    let analyzed = run(plain, &ctx()).unwrap();
    assert!(!has_read(analyzed.plan.root()), "{}", analyzed.plan.root());
}

/// A read whose measure nests `n` deep over the `part` table: depth `n + 1`.
fn read_with_deep_measure(n: usize) -> LogicalPlan {
    let deep = (1..n).fold(c("p_size"), |e, _| {
        Expr::builtin(crate::Builtin::Not, vec![e])
    });
    LogicalPlan::new(PlanNode::MetricViewRead {
        input: Arc::new(LogicalPlan::table(["part"])),
        name: vec!["spark_catalog".into(), "default".into(), "mv".into()],
        dimensions: vec![column("brand", c("p_brand"), DataType::String)],
        measures: vec![column("total", deep, DataType::Long)],
        output: Vec::new(),
        ansi: None,
    })
}

#[test]
fn the_depth_check_measures_dimension_and_measure_expressions() {
    assert!(crate::check_depth(&read_with_deep_measure(crate::MAX_DEPTH - 1)).is_ok());
    let err = crate::check_depth(&read_with_deep_measure(crate::MAX_DEPTH)).unwrap_err();
    assert_eq!(
        err.error_class(),
        krishiv_common::error::ErrorClass::FailedToParseTooComplex
    );
    assert!(crate::Plan::try_new(read_with_deep_measure(crate::MAX_DEPTH)).is_err());
}

#[test]
fn reads_whose_column_types_differ_in_nested_nullability_are_different_plans() {
    let with = |contains_null: bool| {
        let t = DataType::Array(Arc::new(krishiv_types::ArrayType::new(
            DataType::Long,
            contains_null,
        )));
        crate::Plan::try_new(LogicalPlan::new(PlanNode::MetricViewRead {
            input: Arc::new(LogicalPlan::table(["part"])),
            name: vec!["mv".into()],
            dimensions: vec![column("xs", c("p_xs"), t)],
            measures: Vec::new(),
            output: Vec::new(),
            ansi: None,
        }))
        .unwrap()
    };
    assert!(with(true).structurally_eq(&with(true)));
    assert!(!with(true).structurally_eq(&with(false)));
}
