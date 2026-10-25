// SPDX-License-Identifier: Apache-2.0
//! `ResolvePivot` and `ResolveUnpivot` on hand-built plans: output names
//! against Spark 4.2's rules, and every error class with its message
//! parameters.

use std::sync::Arc;

use krishiv_types::DataType;
use serde_json::Value;

use super::dsl::{add, c, eq, f, int, s};
use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    AnalysisError, AnalyzedPlan, Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf,
    TableColumn,
};
use crate::{
    Expr, ExprNode, Literal, LogicalPlan, Origin, PivotValue, Plan, PlanNode, UnpivotColumnSet,
};

fn catalog() -> MemoryCatalog {
    let col = |name: &str, data_type: DataType, nullable: bool| TableColumn {
        name: name.to_owned(),
        data_type,
        nullable,
    };
    MemoryCatalog::new("spark_catalog")
        .with_table(
            &["default"],
            "sales",
            vec![
                col("course", DataType::String, true),
                col("year", DataType::Integer, false),
                col("earnings", DataType::Integer, true),
            ],
            None,
        )
        .with_table(
            &["default"],
            "flags",
            vec![
                col("id", DataType::Integer, false),
                col("big", DataType::Long, true),
                col("i2", DataType::Integer, true),
                col("i3", DataType::Integer, true),
                col("i4", DataType::Integer, true),
                col("ok", DataType::Boolean, true),
            ],
            None,
        )
        .with_table(
            &["default"],
            "events",
            vec![
                col("day", DataType::Date, false),
                col("v", DataType::Integer, true),
            ],
            None,
        )
}

fn ctx() -> crate::analyzer::AnalysisContext {
    context(catalog(), ResolutionConf::default())
}

fn run(plan: LogicalPlan) -> Result<AnalyzedPlan, AnalysisError> {
    let plan = Plan::try_new(plan).map_err(|error| AnalysisError {
        error,
        origin: Origin::UNKNOWN,
    })?;
    block_on(DefaultAnalyzer.analyze(&plan, &ctx()))
}

/// Output names, or the error class and parameters.
fn summary(r: &Result<AnalyzedPlan, AnalysisError>) -> Value {
    match r {
        Ok(a) => serde_json::json!(a.output.iter().map(|c| c.name.clone()).collect::<Vec<_>>()),
        Err(e) => serde_json::json!([
            e.error.error_class().name(),
            e.error
                .message_parameters()
                .iter()
                .map(|(k, v)| (*k, v))
                .collect::<std::collections::BTreeMap<_, _>>()
        ]),
    }
}

fn names(r: &Result<AnalyzedPlan, AnalysisError>) -> Vec<String> {
    r.as_ref()
        .unwrap_or_else(|e| panic!("expected rows, got {e}"))
        .output
        .iter()
        .map(|c| c.name.clone())
        .collect()
}

/// `PIVOT (aggs FOR col IN (...))` over `sales`.
fn pivot(cols: &[&str], values: Vec<(Vec<Expr>, Option<&str>)>, aggs: Vec<Expr>) -> LogicalPlan {
    LogicalPlan::new(PlanNode::Pivot {
        input: Arc::new(LogicalPlan::table(["sales"])),
        pivot_columns: cols.iter().map(|n| Arc::new(c(n))).collect(),
        values: values
            .into_iter()
            .map(|(exprs, alias)| PivotValue {
                exprs: exprs.into_iter().map(Arc::new).collect(),
                alias: alias.map(str::to_owned),
            })
            .collect(),
        aggregates: aggs.into_iter().map(Arc::new).collect(),
    })
}

fn sum_earnings() -> Expr {
    f("sum", vec![c("earnings")])
}

#[test]
fn pivot_names_single_aggregate() {
    let r = run(pivot(
        &["course"],
        vec![(vec![s("dotNET")], None), (vec![s("Java")], None)],
        vec![sum_earnings()],
    ));
    assert_eq!(names(&r), ["year", "dotNET", "Java"]);
    assert!(r.unwrap().plan.root().is_resolved());
}

#[test]
fn pivot_names_several_aggregates() {
    let r = run(pivot(
        &["course"],
        vec![(vec![s("dotNET")], None), (vec![s("Java")], None)],
        vec![sum_earnings(), f("avg", vec![c("earnings")])],
    ));
    assert_eq!(
        names(&r),
        [
            "year",
            "dotNET_sum(earnings)",
            "dotNET_avg(earnings)",
            "Java_sum(earnings)",
            "Java_avg(earnings)"
        ]
    );
}

#[test]
fn pivot_names_aggregate_aliases() {
    let r = run(pivot(
        &["year"],
        vec![(vec![int(2012)], None), (vec![int(2013)], None)],
        vec![
            f("sum", vec![c("earnings")]).alias("s"),
            f("avg", vec![c("earnings")]).alias("a"),
        ],
    ));
    assert_eq!(
        names(&r),
        ["course", "2012_s", "2012_a", "2013_s", "2013_a"]
    );
}

#[test]
fn pivot_names_value_aliases() {
    let r = run(pivot(
        &["course"],
        vec![
            (vec![s("dotNET")], Some("net")),
            (vec![s("Java")], Some("jv")),
        ],
        vec![sum_earnings()],
    ));
    assert_eq!(names(&r), ["year", "net", "jv"]);
}

#[test]
fn pivot_names_generated_aggregate_sql() {
    // `ceil` prints upper case, as Spark's `UnaryMathExpression` does.
    let r = run(pivot(
        &["course"],
        vec![(vec![s("dotNET")], None), (vec![s("Java")], None)],
        vec![
            f("ceil", vec![sum_earnings()]),
            f("avg", vec![c("earnings")]).alias("a1"),
        ],
    ));
    assert_eq!(
        names(&r),
        [
            "year",
            "dotNET_CEIL(sum(earnings))",
            "dotNET_a1",
            "Java_CEIL(sum(earnings))",
            "Java_a1"
        ]
    );
}

#[test]
fn pivot_without_grouping_columns_is_global() {
    let plan = LogicalPlan::new(PlanNode::Pivot {
        input: Arc::new(LogicalPlan::table(["sales"]).project(vec![c("course"), c("earnings")])),
        pivot_columns: vec![Arc::new(c("course"))],
        values: vec![
            PivotValue {
                exprs: vec![Arc::new(s("dotNET"))],
                alias: None,
            },
            PivotValue {
                exprs: vec![Arc::new(s("Java"))],
                alias: None,
            },
        ],
        aggregates: vec![Arc::new(sum_earnings())],
    });
    assert_eq!(names(&run(plan)), ["dotNET", "Java"]);
}

#[test]
fn pivot_folds_value_expressions_for_names() {
    let r = run(pivot(
        &["year"],
        vec![
            (
                vec![Expr::builtin(crate::Builtin::Add, vec![int(2011), int(1)])],
                None,
            ),
            (vec![int(2013)], None),
        ],
        vec![sum_earnings()],
    ));
    assert_eq!(names(&r), ["course", "2012", "2013"]);
}

#[test]
fn pivot_rejects_a_bare_column_aggregate() {
    let r = run(pivot(
        &["course"],
        vec![(vec![s("dotNET")], None)],
        vec![sum_earnings(), c("year")],
    ));
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "_LEGACY_ERROR_TEMP_1006",
            {"sql": "sales.year"},
        ])
    );
}

#[test]
fn pivot_rejects_a_scalar_aggregate() {
    let r = run(pivot(
        &["year"],
        vec![(vec![int(2012)], None)],
        vec![f("abs", vec![c("earnings")])],
    ));
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "_LEGACY_ERROR_TEMP_1006",
            {"sql": "sales.earnings"},
        ])
    );
}

#[test]
fn pivot_rejects_a_nested_aggregate() {
    let r = run(pivot(
        &["course"],
        vec![(vec![s("dotNET")], None)],
        vec![f("sum", vec![f("avg", vec![c("earnings")])])],
    ));
    assert_eq!(
        summary(&r),
        serde_json::json!(["NESTED_AGGREGATE_FUNCTION", {}])
    );
}

#[test]
fn pivot_rejects_an_unresolvable_pivot_column() {
    let r = run(pivot(
        &["s"],
        vec![(vec![int(2012)], None)],
        vec![sum_earnings()],
    ));
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "UNRESOLVED_COLUMN.WITH_SUGGESTION",
            {"objectName": "`s`", "proposal": "`year`, `course`, `earnings`"},
        ])
    );
}

#[test]
fn pivot_rejects_an_unresolvable_value() {
    let r = run(pivot(
        &["year"],
        vec![(vec![c("s")], None)],
        vec![sum_earnings()],
    ));
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "UNRESOLVED_COLUMN.WITH_SUGGESTION",
            {"objectName": "`s`", "proposal": "`year`, `course`, `earnings`"},
        ])
    );
}

#[test]
fn pivot_rejects_a_non_literal_value() {
    let r = run(pivot(
        &["year"],
        vec![(vec![c("course")], None)],
        vec![sum_earnings()],
    ));
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "NON_LITERAL_PIVOT_VALUES",
            {"expression": "\"course\""},
        ])
    );
}

#[test]
fn pivot_rejects_an_aliased_non_literal_value() {
    let r = run(pivot(
        &["year"],
        vec![(vec![c("course")], Some("c"))],
        vec![sum_earnings()],
    ));
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "NON_LITERAL_PIVOT_VALUES",
            {"expression": "\"course AS c\""},
        ])
    );
}

#[test]
fn pivot_rejects_a_value_of_the_wrong_type() {
    // An `INT` value for a `DATE` pivot column: neither casts to the
    // other.
    let plan = LogicalPlan::new(PlanNode::Pivot {
        input: Arc::new(LogicalPlan::table(["events"])),
        pivot_columns: vec![Arc::new(c("day"))],
        values: vec![PivotValue {
            exprs: vec![Arc::new(int(2012))],
            alias: None,
        }],
        aggregates: vec![Arc::new(f("sum", vec![c("v")]))],
    });
    assert_eq!(
        summary(&run(plan)),
        serde_json::json!([
            "PIVOT_VALUE_DATA_TYPE_MISMATCH",
            {
                "pivotType": "date",
                "value": "2012",
                "valueType": "int",
            },
        ])
    );
}

#[test]
fn pivot_rejects_a_folding_error_in_a_value() {
    // ANSI `2147483647 + 1` overflows while folding.
    let r = run(pivot(
        &["year"],
        vec![(
            vec![Expr::builtin(
                crate::Builtin::Add,
                vec![int(2_147_483_647), int(1)],
            )],
            None,
        )],
        vec![sum_earnings()],
    ));
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "ARITHMETIC_OVERFLOW",
            {
                "alternative": " Use 'try_add' to tolerate overflow and return NULL instead.",
                "config": "\"spark.sql.ansi.enabled\"",
                "message": "overflow",
            },
        ])
    );
}

#[test]
fn pivot_without_pivot_columns_is_unsupported() {
    let r = run(pivot(
        &[],
        vec![(vec![int(2012)], None)],
        vec![sum_earnings()],
    ));
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "KRISHIV_UNSUPPORTED_FEATURE",
            {"feature": "PIVOT without pivot columns"},
        ])
    );
}

#[test]
fn pivot_on_several_columns_is_unsupported() {
    let r = run(pivot(
        &["course", "year"],
        vec![
            (vec![s("dotNET"), int(2012)], None),
            (vec![s("Java"), int(2013)], None),
        ],
        vec![sum_earnings()],
    ));
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "KRISHIV_UNSUPPORTED_FEATURE",
            {"feature": "multi-column PIVOT"},
        ])
    );
}

#[test]
fn pivot_on_several_columns_checks_value_types_first() {
    let r = run(pivot(
        &["course", "year"],
        vec![(vec![s("dotNET")], None)],
        vec![sum_earnings()],
    ));
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "PIVOT_VALUE_DATA_TYPE_MISMATCH",
            {
                "pivotType": "struct<course:string,year:int>",
                "value": "dotNET",
                "valueType": "string",
            },
        ])
    );
}

#[test]
fn pivot_on_several_columns_checks_values_are_literal_first() {
    let r = run(pivot(
        &["course", "year"],
        vec![(vec![c("earnings"), int(2012)], None)],
        vec![sum_earnings()],
    ));
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "NON_LITERAL_PIVOT_VALUES",
            {"expression": "\"earnings\""},
        ])
    );
}

#[test]
fn pivot_counts_matching_rows() {
    let r = run(pivot(
        &["course"],
        vec![(vec![s("dotNET")], None), (vec![s("Java")], None)],
        vec![f("count", vec![c("earnings")])],
    ));
    assert_eq!(names(&r), ["year", "dotNET", "Java"]);
}

#[test]
fn pivot_first_ignores_the_nulls_the_if_introduces() {
    let r = run(pivot(
        &["course"],
        vec![(vec![s("dotNET")], None)],
        vec![f("first", vec![c("earnings")])],
    ));
    assert_eq!(names(&r), ["year", "dotNET"]);
    // The rewritten `first` ignores nulls: its second argument is `true`.
    let mut found = false;
    let _ = crate::walk::visit_distinct(crate::walk::Node::Plan(r.unwrap().plan.root()), |node| {
        if let crate::walk::Node::Expr(e) = node
            && let ExprNode::Resolved(crate::ResolvedExpr::Function(f)) = &e.node
            && f.id.name() == "first"
        {
            found = f.args.len() == 2
                && matches!(&f.args[1].node, ExprNode::Literal(Literal::Boolean(true)));
        }
        std::ops::ControlFlow::Continue(())
    });
    assert!(found, "first(value, true)");
}

/// `UNPIVOT (value FOR name IN (cols))` over `sales`.
fn unpivot(
    values: &[&str],
    name: &str,
    sets: Vec<(Vec<&str>, Option<&str>)>,
    include_nulls: bool,
) -> LogicalPlan {
    LogicalPlan::new(PlanNode::Unpivot {
        input: Arc::new(LogicalPlan::table(["sales"])),
        value_columns: values.iter().map(|v| (*v).to_owned()).collect(),
        name_column: name.to_owned(),
        column_sets: sets
            .into_iter()
            .map(|(cols, alias)| UnpivotColumnSet {
                columns: cols.into_iter().map(|n| Arc::new(c(n))).collect(),
                alias: alias.map(str::to_owned),
            })
            .collect(),
        include_nulls,
    })
}

#[test]
fn unpivot_names_its_output() {
    let r = run(unpivot(
        &["earnings_year"],
        "year",
        vec![(vec!["year"], None)],
        false,
    ));
    assert_eq!(names(&r), ["course", "earnings", "year", "earnings_year"]);
    assert!(r.unwrap().plan.root().is_resolved());
}

#[test]
fn unpivot_labels_rows_with_literals_and_aliases() {
    // Labels only show in the rows; the plan still resolves, with and
    // without `INCLUDE NULLS`.
    for include_nulls in [false, true] {
        let r = run(unpivot(
            &["v"],
            "n",
            vec![(vec!["year"], Some("y2012")), (vec!["earnings"], None)],
            include_nulls,
        ));
        assert_eq!(names(&r), ["course", "n", "v"]);
    }
}

#[test]
fn unpivot_rejects_unshared_types() {
    let plan = LogicalPlan::new(PlanNode::Unpivot {
        input: Arc::new(LogicalPlan::table(["flags"])),
        value_columns: vec!["v".to_owned()],
        name_column: "n".to_owned(),
        column_sets: vec![
            UnpivotColumnSet {
                columns: vec![Arc::new(c("id"))],
                alias: None,
            },
            UnpivotColumnSet {
                columns: vec![Arc::new(c("ok"))],
                alias: None,
            },
        ],
        include_nulls: true,
    });
    assert_eq!(
        summary(&run(plan)),
        serde_json::json!([
            "UNPIVOT_VALUE_DATA_TYPE_MISMATCH",
            {"types": "\"BOOLEAN\" (`ok`), \"INT\" (`id`)"},
        ])
    );
}

#[test]
fn unpivot_without_value_columns_fails() {
    let r = run(unpivot(&[], "n", vec![], false));
    assert_eq!(
        summary(&r),
        serde_json::json!(["UNPIVOT_REQUIRES_VALUE_COLUMNS", {}])
    );
}

#[test]
fn unpivot_with_mismatched_set_sizes_fails() {
    let r = run(unpivot(&["e", "s"], "n", vec![(vec!["year"], None)], false));
    assert_eq!(
        summary(&r),
        serde_json::json!(["UNPIVOT_VALUE_SIZE_MISMATCH", {"names": "2"}])
    );
}

#[test]
fn unpivot_with_a_non_column_fails() {
    let plan = LogicalPlan::new(PlanNode::Unpivot {
        input: Arc::new(LogicalPlan::table(["sales"])),
        value_columns: vec!["v".to_owned()],
        name_column: "n".to_owned(),
        column_sets: vec![UnpivotColumnSet {
            columns: vec![Arc::new(Expr::new(ExprNode::Literal(Literal::Integer(1))))],
            alias: None,
        }],
        include_nulls: false,
    });
    assert_eq!(
        summary(&run(plan)),
        serde_json::json!([
            "UNPIVOT_REQUIRES_ATTRIBUTES",
            {"empty": "id", "expressions": "\"1\"", "given": "value"},
        ])
    );
}

#[test]
fn pivot_names_a_date_value_as_spark_casts_it() {
    use crate::TypedLiteralKind;
    let plan = LogicalPlan::new(PlanNode::Pivot {
        input: Arc::new(LogicalPlan::table(["events"])),
        pivot_columns: vec![Arc::new(c("day"))],
        values: vec![PivotValue {
            exprs: vec![Arc::new(Expr::new(ExprNode::TypedLiteral {
                kind: TypedLiteralKind::Date,
                value: "2012-01-15".to_owned(),
            }))],
            alias: None,
        }],
        aggregates: vec![Arc::new(f("sum", vec![c("v")]))],
    });
    assert_eq!(names(&run(plan)), ["2012-01-15"]);
}

#[test]
fn pivot_names_an_aliased_value_node() {
    // An `Alias` value without a value alias (as Spark's parser builds
    // them): the name is the alias.
    let plan = LogicalPlan::new(PlanNode::Pivot {
        input: Arc::new(LogicalPlan::table(["sales"])),
        pivot_columns: vec![Arc::new(c("course"))],
        values: vec![PivotValue {
            exprs: vec![Arc::new(s("dotNET").alias("net"))],
            alias: None,
        }],
        aggregates: vec![Arc::new(sum_earnings())],
    });
    assert_eq!(names(&run(plan)), ["year", "net"]);
}

#[test]
fn pivot_counts_star() {
    use super::dsl::count_star;
    let r = run(pivot(
        &["course"],
        vec![(vec![s("dotNET")], None)],
        vec![count_star()],
    ));
    // `count(*)` references no column, so `earnings` stays a grouping
    // column, as in Spark.
    assert_eq!(names(&r), ["year", "earnings", "dotNET"]);
    // The rewrite counts matching rows: `count(if(course <=> 'dotNET',
    // 1, NULL))`, not `count(1)`.
    let analyzed = r.unwrap();
    let mut found = false;
    let _ = crate::walk::visit_distinct(crate::walk::Node::Plan(analyzed.plan.root()), |node| {
        if let crate::walk::Node::Expr(e) = node
            && let ExprNode::Resolved(crate::ResolvedExpr::Function(f)) = &e.node
            && f.id.name().eq_ignore_ascii_case("count")
            && let [arg] = &f.args[..]
            // `if` binds to the `when` operator with three arguments.
            && let ExprNode::Resolved(crate::ResolvedExpr::Function(g)) = &arg.node
            && g.id.name().eq_ignore_ascii_case("when")
            && g.args.len() == 3
            && matches!(&g.args[1].node, ExprNode::Literal(Literal::Integer(1)))
        {
            found = true;
        }
        std::ops::ControlFlow::Continue(())
    });
    assert!(found, "count(if(pivot <=> value, 1, NULL))");
}

#[test]
fn pivot_on_several_columns_checks_value_types() {
    use crate::TypedLiteralKind;
    // A `DATE` value for the `INT` pivot column: Spark's cast check fails,
    // so this is `PIVOT_VALUE_DATA_TYPE_MISMATCH`, not the unsupported
    // multi-column path.
    let date = Expr::new(ExprNode::TypedLiteral {
        kind: TypedLiteralKind::Date,
        value: "2012-01-15".to_owned(),
    });
    let r = run(pivot(
        &["course", "year"],
        vec![(vec![s("dotNET"), date], None)],
        vec![sum_earnings()],
    ));
    let summary = summary(&r);
    assert_eq!(
        summary.as_array().map(|a| a[0].clone()),
        Some(serde_json::json!("PIVOT_VALUE_DATA_TYPE_MISMATCH")),
    );
    assert_eq!(
        summary
            .as_array()
            .and_then(|a| a[1].get("pivotType").cloned()),
        Some(serde_json::json!("struct<course:string,year:int>")),
    );
}

#[test]
fn pivot_rejects_a_multi_part_value_for_one_column() {
    let r = run(pivot(
        &["year"],
        vec![(vec![int(2012), int(2013)], None)],
        vec![sum_earnings()],
    ));
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "PIVOT_VALUE_DATA_TYPE_MISMATCH",
            {
                "pivotType": "int",
                "value": "{2012, 2013}",
                "valueType": "struct<col1:int,col2:int>",
            },
        ])
    );
}

#[test]
fn pivot_keeps_an_aliased_mismatched_value_whole() {
    let r = run(pivot(
        &["course", "year"],
        vec![(vec![s("dotNET")], Some("net"))],
        vec![sum_earnings()],
    ));
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "PIVOT_VALUE_DATA_TYPE_MISMATCH",
            {
                "pivotType": "struct<course:string,year:int>",
                "value": "net",
                "valueType": "string",
            },
        ])
    );
}

#[test]
fn pivot_names_a_non_literal_fractional_mismatch_as_written() {
    let r = run(pivot(
        &["course", "year"],
        vec![(
            vec![Expr::builtin(
                crate::Builtin::Add,
                vec![Expr::new(ExprNode::Literal(Literal::Double(1.5))), int(0)],
            )],
            None,
        )],
        vec![sum_earnings()],
    ));
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "PIVOT_VALUE_DATA_TYPE_MISMATCH",
            {
                "pivotType": "struct<course:string,year:int>",
                "value": "(1.5 + 0)",
                "valueType": "double",
            },
        ])
    );
}

#[test]
fn pivot_on_several_literal_columns_is_unsupported() {
    let plan = LogicalPlan::new(PlanNode::Pivot {
        input: Arc::new(LogicalPlan::table(["sales"])),
        pivot_columns: vec![Arc::new(int(1)), Arc::new(int(2))],
        values: vec![PivotValue {
            exprs: vec![Arc::new(int(1)), Arc::new(int(2))],
            alias: None,
        }],
        aggregates: vec![Arc::new(sum_earnings())],
    });
    assert_eq!(
        summary(&run(plan)),
        serde_json::json!([
            "KRISHIV_UNSUPPORTED_FEATURE",
            {"feature": "multi-column PIVOT"},
        ])
    );
}

#[test]
fn pivot_quotes_a_non_bare_value_alias() {
    let r = run(pivot(
        &["year"],
        vec![(vec![c("course")], Some("twenty twelve"))],
        vec![sum_earnings()],
    ));
    assert_eq!(
        summary(&r),
        serde_json::json!([
            "NON_LITERAL_PIVOT_VALUES",
            {"expression": "\"course AS `twenty twelve`\""},
        ])
    );
}

#[test]
fn pivot_names_columns_without_a_qualifier_bare() {
    // Over an aggregate the grouping attributes carry no qualifier; a
    // bare column there is still not an aggregate.
    let plan = LogicalPlan::new(PlanNode::Pivot {
        input: Arc::new(
            LogicalPlan::table(["sales"])
                .aggregate(vec![c("year")], vec![c("year"), sum_earnings().alias("s")]),
        ),
        pivot_columns: vec![Arc::new(c("year"))],
        values: vec![PivotValue {
            exprs: vec![Arc::new(int(2012))],
            alias: None,
        }],
        aggregates: vec![Arc::new(Expr::builtin(
            crate::Builtin::Add,
            vec![c("s"), int(1)],
        ))],
    });
    assert_eq!(
        summary(&run(plan)),
        serde_json::json!([
            "_LEGACY_ERROR_TEMP_1006",
            {"sql": "s"},
        ])
    );
}

#[test]
fn pivot_folds_computed_values_for_names() {
    // `3`, `true` and `dotNET` all fold; the names use the folded values.
    let r = run(pivot(
        &["year"],
        vec![(vec![add(int(1), int(2))], None)],
        vec![sum_earnings()],
    ));
    assert_eq!(names(&r), ["course", "3"]);
    let r = run(pivot(
        &["course"],
        vec![(vec![eq(int(1), int(1))], None)],
        vec![sum_earnings()],
    ));
    assert_eq!(names(&r), ["year", "true"]);
    let r = run(pivot(
        &["course"],
        vec![(
            vec![Expr::builtin(
                crate::Builtin::Concat,
                vec![s("dot"), s("NET")],
            )],
            None,
        )],
        vec![sum_earnings()],
    ));
    assert_eq!(names(&r), ["year", "dotNET"]);
}

#[test]
fn unresolved_pivot_and_unpivot_print() {
    let plan = pivot(
        &["course"],
        vec![(vec![s("dotNET")], Some("net"))],
        vec![sum_earnings()],
    );
    let text = plan.to_string();
    assert!(text.contains("Pivot [course]"), "{text}");
    let plan = unpivot(&["v"], "n", vec![(vec!["year"], None)], false);
    let text = plan.to_string();
    assert!(text.contains("Unpivot [v FOR n IN"), "{text}");
    let plan = unpivot(&["v"], "n", vec![(vec!["year"], None)], true);
    assert!(plan.to_string().contains("UnpivotIncludeNulls"), "{text}");
}

#[test]
fn unpivot_multi_value_sets_label_and_widen() {
    let r = run(unpivot(
        &["e", "s"],
        "n",
        vec![(vec!["year", "earnings"], None)],
        false,
    ));
    assert_eq!(names(&r), ["course", "n", "e", "s"]);
    let r = run(unpivot(
        &["e", "s"],
        "n",
        vec![(vec!["year", "earnings"], Some("y2012"))],
        true,
    ));
    assert_eq!(names(&r), ["course", "n", "e", "s"]);
}

#[test]
fn unpivot_casts_members_to_the_wider_type() {
    // `id` is `INT`, `big` is `BIGINT`: the value column is `BIGINT`
    // and the `INT` member is cast.
    let plan = LogicalPlan::new(PlanNode::Unpivot {
        input: Arc::new(LogicalPlan::table(["flags"])),
        value_columns: vec!["v".to_owned()],
        name_column: "n".to_owned(),
        column_sets: vec![
            UnpivotColumnSet {
                columns: vec![Arc::new(c("id"))],
                alias: None,
            },
            UnpivotColumnSet {
                columns: vec![Arc::new(c("big"))],
                alias: None,
            },
        ],
        include_nulls: true,
    });
    let r = run(plan);
    assert_eq!(names(&r), ["i2", "i3", "i4", "ok", "n", "v"]);
    let types: Vec<DataType> = r
        .unwrap()
        .output
        .iter()
        .map(|c| c.data_type.clone())
        .collect();
    assert_eq!(types.last(), Some(&DataType::Long));
}

#[test]
fn unpivot_truncates_long_type_lists() {
    let plan = LogicalPlan::new(PlanNode::Unpivot {
        input: Arc::new(LogicalPlan::table(["flags"])),
        value_columns: vec!["v".to_owned()],
        name_column: "n".to_owned(),
        column_sets: ["id", "i2", "i3", "i4", "ok"]
            .into_iter()
            .map(|name| UnpivotColumnSet {
                columns: vec![Arc::new(c(name))],
                alias: None,
            })
            .collect(),
        include_nulls: true,
    });
    assert_eq!(
        summary(&run(plan)),
        serde_json::json!([
            "UNPIVOT_VALUE_DATA_TYPE_MISMATCH",
            {"types": "\"BOOLEAN\" (`ok`), \"INT\" (`i2`, `i3`, `i4`, ...)"},
        ])
    );
}

#[test]
fn pivot_resolves_without_ansi_casts() {
    use crate::analyzer::ResolutionConf;
    let conf = ResolutionConf {
        ansi: false,
        ..ResolutionConf::default()
    };
    let plan = Plan::try_new(pivot(
        &["course"],
        vec![(vec![s("dotNET")], None)],
        vec![sum_earnings()],
    ))
    .unwrap();
    let analyzed = block_on(DefaultAnalyzer.analyze(&plan, &context(catalog(), conf))).unwrap();
    assert_eq!(
        analyzed
            .output
            .iter()
            .map(|c| c.name.clone())
            .collect::<Vec<_>>(),
        ["year", "dotNET"]
    );
}

#[test]
fn pivot_shares_subtrees_linearly() {
    // One `sum` shared by both sides of the `+`: the reference walks
    // visit it once.
    let sum = sum_earnings();
    let r = run(pivot(
        &["course"],
        vec![(vec![s("dotNET")], None)],
        vec![Expr::builtin(crate::Builtin::Add, vec![sum.clone(), sum])],
    ));
    assert_eq!(names(&r), ["year", "dotNET"]);
}

#[test]
fn pivot_names_an_aliased_non_literal_node() {
    // An `Alias` value node without a value alias: the error keeps the
    // alias, as Spark's `Alias.sql` does.
    let plan = LogicalPlan::new(PlanNode::Pivot {
        input: Arc::new(LogicalPlan::table(["sales"])),
        pivot_columns: vec![Arc::new(c("year"))],
        values: vec![PivotValue {
            exprs: vec![Arc::new(c("course").alias("c"))],
            alias: None,
        }],
        aggregates: vec![Arc::new(sum_earnings())],
    });
    assert_eq!(
        summary(&run(plan)),
        serde_json::json!([
            "NON_LITERAL_PIVOT_VALUES",
            {"expression": "\"course AS c\""},
        ])
    );
}

#[test]
fn pivot_names_an_aliased_mismatched_node() {
    // An `Alias` value node without a value alias, of the wrong type:
    // the mismatch names the alias.
    let plan = LogicalPlan::new(PlanNode::Pivot {
        input: Arc::new(LogicalPlan::table(["events"])),
        pivot_columns: vec![Arc::new(c("day"))],
        values: vec![PivotValue {
            exprs: vec![Arc::new(int(2012).alias("y2012"))],
            alias: None,
        }],
        aggregates: vec![Arc::new(f("sum", vec![c("v")]))],
    });
    assert_eq!(
        summary(&run(plan)),
        serde_json::json!([
            "PIVOT_VALUE_DATA_TYPE_MISMATCH",
            {
                "pivotType": "date",
                "value": "y2012",
                "valueType": "int",
            },
        ])
    );
}

#[test]
fn pivot_names_timestamp_values_as_spark_casts_them() {
    let stamp = |us: i64| Expr::new(ExprNode::Literal(Literal::Timestamp(us)));
    let ntz = |us: i64| Expr::new(ExprNode::Literal(Literal::TimestampNtz(us)));
    for value in [stamp(1_326_621_600_000_000), ntz(1_326_621_600_000_000)] {
        let r = run(LogicalPlan::new(PlanNode::Pivot {
            input: Arc::new(LogicalPlan::table(["sales"])),
            pivot_columns: vec![Arc::new(c("course"))],
            values: vec![PivotValue {
                exprs: vec![Arc::new(value)],
                alias: None,
            }],
            aggregates: vec![Arc::new(sum_earnings())],
        }));
        assert_eq!(names(&r), ["year", "2012-01-15 10:00:00"]);
    }
}

#[test]
fn pivot_names_a_null_cast_null() {
    let null = Expr::new(ExprNode::Cast {
        expr: Arc::new(Expr::new(ExprNode::Literal(Literal::Null(None)))),
        to: DataType::Integer,
        mode: crate::CastMode::Default,
    });
    let r = run(pivot(
        &["year"],
        vec![(vec![null], None)],
        vec![sum_earnings()],
    ));
    assert_eq!(names(&r), ["course", "null"]);
}

#[test]
fn unresolved_pivot_nodes_rewrite_children_expressions_and_have_no_output() {
    use std::convert::Infallible;
    let plan = Arc::new(pivot(
        &["course"],
        vec![(vec![s("dotNET")], None)],
        vec![sum_earnings()],
    ));
    assert!(!plan.is_resolved());
    assert_eq!(plan.resolved_output(), None);
    let renamed = Arc::clone(&plan).map_children(Ok::<_, Infallible>).unwrap();
    assert!(Arc::ptr_eq(&renamed, &plan));
    let touched = Arc::clone(&plan)
        .map_expressions(|_| {
            Ok::<_, Infallible>(Arc::new(Expr::new(ExprNode::Literal(Literal::Null(None)))))
        })
        .unwrap();
    assert!(!Arc::ptr_eq(&touched, &plan));
    let unpivot = Arc::new(unpivot(&["v"], "n", vec![(vec!["year"], None)], false));
    assert!(!unpivot.is_resolved());
    assert_eq!(unpivot.resolved_output(), None);
    let touched = Arc::clone(&unpivot)
        .map_expressions(|_| {
            Ok::<_, Infallible>(Arc::new(Expr::new(ExprNode::Literal(Literal::Null(None)))))
        })
        .unwrap();
    assert!(!Arc::ptr_eq(&touched, &unpivot));
}
