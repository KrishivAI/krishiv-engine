// SPDX-License-Identifier: Apache-2.0
//! `PlanNode::CollectMetrics` (`Dataset.observe`) in the analyzer: the
//! error classes and SQLSTATEs of Spark 4.2.0's `CheckAnalysis`
//! (`tests/fixtures/spark-4.2.0/observe-goldens.json`, the `err_*` cases of
//! the classic leg), the uniqueness of names across relations, and the
//! node's pass-through output.

use super::analyzer::{ctx, run};
use super::dsl::{c, count_star, f, int, sum};
use crate::{Expr, ExprNode, LogicalPlan, PlanNode, WindowExpr, WindowSpec, WindowSpecRef};
use std::sync::Arc;

fn observe(input: LogicalPlan, name: &str, metrics: Vec<Expr>) -> LogicalPlan {
    LogicalPlan::new(PlanNode::CollectMetrics {
        input: Arc::new(input),
        name: name.to_owned(),
        metrics: metrics.into_iter().map(Arc::new).collect(),
    })
}

fn part() -> LogicalPlan {
    LogicalPlan::table(["part"])
}

/// The error class and SQLSTATE of observing `metrics` over `part`.
fn fails(metrics: Vec<Expr>) -> (String, Option<String>) {
    let e = run(observe(part(), "m", metrics), &ctx()).expect_err("must fail");
    (
        e.error.error_class().name().to_owned(),
        e.error.sql_state().map(str::to_owned),
    )
}

fn invalid(class: &str) -> (String, Option<String>) {
    (class.to_owned(), Some("42K0E".to_owned()))
}

#[test]
fn allowed_metrics_pass_rows_through() {
    let out = run(
        observe(
            part(),
            "m",
            vec![
                count_star().alias("n"),
                sum(c("p_partkey")).alias("s"),
                int(42),
                f("sum", vec![f("rand", vec![])]).alias("r"),
                f("max", vec![c("p_partkey")]),
                // A nondeterministic call inside an aggregate stays allowed under arithmetic.
                crate::Expr::builtin(
                    crate::Builtin::Add,
                    vec![f("sum", vec![f("rand", vec![])]), int(1)],
                )
                .alias("r2"),
            ],
        ),
        &ctx(),
    )
    .expect("analyzes");
    let root = out.plan.root();
    let PlanNode::CollectMetrics { metrics, input, .. } = &root.node else {
        panic!("root is {root}");
    };
    assert_eq!(metrics.len(), 6);
    // Pass-through: exactly the input's columns (ids, names, types, nullability).
    let output = root.resolved_output().expect("the node's output");
    let below = input.resolved_output().expect("the input's output");
    assert!(!output.is_empty());
    assert_eq!(output, below);
}

/// The `CollectMetrics` nodes of `plan` in pre-order: (name, metric output names, the node kind
/// of the input).
fn observed_nodes(plan: &Arc<LogicalPlan>) -> Vec<(String, Vec<String>, String)> {
    let mut out = Vec::new();
    let mut stack = vec![plan];
    while let Some(p) = stack.pop() {
        if let PlanNode::CollectMetrics {
            name,
            metrics,
            input,
        } = &p.node
        {
            let names = metrics
                .iter()
                .map(|m| match &m.node {
                    ExprNode::Resolved(crate::ResolvedExpr::Alias { output, .. }) => {
                        output.name.clone()
                    }
                    other => format!("not an alias: {other:?}"),
                })
                .collect();
            let kind = format!("{:?}", input.node)
                .split([' ', '(', '{'])
                .next()
                .unwrap_or_default()
                .to_owned();
            out.push((name.clone(), names, kind));
        }
        let mut kids = Vec::new();
        p.for_each_child(&mut |c| kids.push(c));
        stack.extend(kids.into_iter().rev());
    }
    out
}

#[test]
fn attribute_outside_an_aggregate() {
    assert_eq!(
        fails(vec![c("p_partkey")]),
        invalid("INVALID_OBSERVED_METRICS.NON_AGGREGATE_FUNC_ARG_IS_ATTRIBUTE")
    );
}

#[test]
fn nested_aggregate() {
    assert_eq!(
        fails(vec![sum(f("max", vec![c("p_partkey")])).alias("n")]),
        invalid("INVALID_OBSERVED_METRICS.NESTED_AGGREGATES_UNSUPPORTED")
    );
}

#[test]
fn distinct_and_filter() {
    let mut call = f("count", vec![c("p_partkey")]);
    if let ExprNode::Function(fc) = &mut call.node {
        fc.distinct = true;
    }
    assert_eq!(
        fails(vec![call.alias("d")]),
        invalid("INVALID_OBSERVED_METRICS.AGGREGATE_EXPRESSION_WITH_DISTINCT_UNSUPPORTED")
    );
    let mut call = f("count", vec![c("p_partkey")]);
    if let ExprNode::Function(fc) = &mut call.node {
        fc.filter = Some(Arc::new(f("isnotnull", vec![c("p_partkey")])));
    }
    assert_eq!(
        fails(vec![call.alias("f")]),
        invalid("INVALID_OBSERVED_METRICS.AGGREGATE_EXPRESSION_WITH_FILTER_UNSUPPORTED")
    );
}

#[test]
fn non_deterministic_outside_an_aggregate() {
    assert_eq!(
        fails(vec![f("rand", vec![]).alias("r")]),
        invalid("INVALID_OBSERVED_METRICS.NON_AGGREGATE_FUNC_ARG_IS_NON_DETERMINISTIC")
    );
    assert_eq!(
        fails(vec![
            crate::Expr::builtin(crate::Builtin::Add, vec![count_star(), f("rand", vec![])])
                .alias("r")
        ]),
        invalid("INVALID_OBSERVED_METRICS.NON_AGGREGATE_FUNC_ARG_IS_NON_DETERMINISTIC")
    );
}

#[test]
fn window_expression() {
    let w = Expr::new(ExprNode::Window(WindowExpr {
        function: Arc::new(sum(c("p_partkey"))),
        window: WindowSpecRef::Spec(WindowSpec {
            partition_by: vec![],
            order_by: vec![],
            frame: None,
        }),
    }));
    assert_eq!(
        fails(vec![w.alias("w")]),
        invalid("INVALID_OBSERVED_METRICS.WINDOW_EXPRESSIONS_UNSUPPORTED")
    );
}

#[test]
fn generator_is_an_analyzer_error() {
    assert_eq!(
        fails(vec![f("explode", vec![c("p_partkey")]).alias("g")]).1,
        Some("42K0E".to_owned())
    );
}

#[test]
fn missing_name() {
    let e = run(observe(part(), "", vec![count_star().alias("n")]), &ctx()).expect_err("fails");
    assert_eq!(
        e.error.error_class().name(),
        "INVALID_OBSERVED_METRICS.MISSING_NAME"
    );
}

#[test]
fn unknown_column_is_unresolved() {
    let e = run(
        observe(part(), "m", vec![sum(c("nope")).alias("n")]),
        &ctx(),
    )
    .expect_err("fails");
    assert!(
        e.error
            .error_class()
            .name()
            .starts_with("UNRESOLVED_COLUMN")
    );
}

#[test]
fn names_are_unique_across_relations_but_not_for_one_relation() {
    let one = observe(part(), "dup", vec![count_star().alias("n")]);
    let other = observe(
        part().limit(Arc::new(int(1))),
        "dup",
        vec![count_star().alias("n")],
    );
    let both = |l: &LogicalPlan, r: &LogicalPlan| {
        l.clone().join(
            r.clone(),
            crate::JoinKind::Cross,
            crate::JoinCondition::None,
        )
    };
    let e = run(both(&one, &other), &ctx()).expect_err("two relations, one name");
    assert_eq!(e.error.error_class().name(), "DUPLICATED_METRICS_NAME");
    assert_eq!(e.error.sql_state(), Some("42710"));
    // The same relation twice (a self-join) is legal.
    let shared = Arc::new(one);
    let twice = LogicalPlan::new(PlanNode::Join {
        left: Arc::clone(&shared),
        right: shared,
        kind: crate::JoinKind::Cross,
        condition: crate::JoinCondition::None,
    });
    let out = run(twice, &ctx()).expect("self-join of one observed relation");
    // Both sides keep the observation.
    let PlanNode::Join { left, right, .. } = &out.plan.root().node else {
        panic!("root is {}", out.plan.root());
    };
    for side in [left, right] {
        assert_eq!(
            observed_nodes(side),
            [("dup".to_owned(), vec!["n".to_owned()], kind_of(&part()))]
        );
    }
}

/// The node kind `observed_nodes` reports for an analyzed `plan` as an input.
fn kind_of(plan: &LogicalPlan) -> String {
    let analyzed = run(plan.clone(), &ctx()).expect("analyzes");
    format!("{:?}", analyzed.plan.root().node)
        .split([' ', '(', '{'])
        .next()
        .unwrap_or_default()
        .to_owned()
}

#[test]
fn nested_observations_with_different_names_are_independent() {
    let inner = observe(part(), "a", vec![count_star().alias("n")]);
    let outer = observe(inner, "b", vec![count_star().alias("n")]);
    let out = run(outer, &ctx()).expect("analyzes");
    // `b` at the root over `a` over the table, each with its own metric.
    let PlanNode::CollectMetrics { name, input, .. } = &out.plan.root().node else {
        panic!("root is {}", out.plan.root());
    };
    assert_eq!(name, "b");
    assert_eq!(
        observed_nodes(out.plan.root()),
        [
            (
                "b".to_owned(),
                vec!["n".to_owned()],
                "CollectMetrics".to_owned()
            ),
            ("a".to_owned(), vec!["n".to_owned()], kind_of(&part())),
        ]
    );
    assert!(matches!(input.node, PlanNode::CollectMetrics { .. }));
}
