// SPDX-License-Identifier: Apache-2.0
//! `PlanNode::ColumnNames` (`toDF`, `withColumnsRenamed`, `drop(names)`) on a hand-built plan over
//! `sales(course, year, earnings)`, against Spark 4.2.0's rules (`UnresolvedSubqueryColumnAliases`,
//! `UnresolvedStarWithColumnsRenames`, `Dataset.drop(colNames*)`).

use std::sync::Arc;

use krishiv_types::DataType;

use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    AnalysisError, AnalyzedPlan, Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf,
    TableColumn,
};
use crate::{ColumnNamesOp, Expr, JoinCondition, JoinKind, LogicalPlan, Origin, Plan, PlanNode};

fn run(case_sensitive: bool, op: ColumnNamesOp) -> Result<AnalyzedPlan, AnalysisError> {
    run_on(case_sensitive, LogicalPlan::table(["sales"]), op, |p| p)
}

/// `op` over `input`, then `after` (a projection reading the result, say).
fn run_on(
    case_sensitive: bool,
    input: LogicalPlan,
    op: ColumnNamesOp,
    after: impl FnOnce(LogicalPlan) -> LogicalPlan,
) -> Result<AnalyzedPlan, AnalysisError> {
    let col = |name: &str, data_type: DataType| TableColumn {
        name: name.to_owned(),
        data_type,
        nullable: true,
        char_varchar: None,
    };
    let catalog = MemoryCatalog::new("spark_catalog").with_table(
        &["default"],
        "sales",
        vec![
            col("course", DataType::String),
            col("year", DataType::Integer),
            col("earnings", DataType::Integer),
        ],
        None,
    );
    let conf = ResolutionConf {
        case_sensitive,
        ..ResolutionConf::default()
    };
    let plan = Plan::try_new(after(LogicalPlan::new(PlanNode::ColumnNames {
        input: Arc::new(input),
        op,
    })))
    .map_err(|error| AnalysisError {
        error,
        origin: Origin::UNKNOWN,
    })?;
    block_on(DefaultAnalyzer.analyze(&plan, &context(catalog, conf)))
}

fn names(r: Result<AnalyzedPlan, AnalysisError>) -> Vec<String> {
    r.unwrap_or_else(|e| panic!("expected columns, got {e}"))
        .output
        .iter()
        .map(|c| c.name.clone())
        .collect()
}

fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_owned()).collect()
}

fn drop_names(n: &[&str]) -> ColumnNamesOp {
    ColumnNamesOp::Drop {
        names: strings(n),
        columns: vec![],
    }
}

fn drop_columns(n: &[&str]) -> ColumnNamesOp {
    ColumnNamesOp::Drop {
        names: vec![],
        columns: strings(n),
    }
}

/// `sales AS l CROSS JOIN sales AS r`: every column name twice.
fn doubled() -> LogicalPlan {
    LogicalPlan::table(["sales"]).alias("l").join(
        LogicalPlan::table(["sales"]).alias("r"),
        JoinKind::Cross,
        JoinCondition::None,
    )
}

#[test]
fn to_df_renames_by_position() {
    let r = run(false, ColumnNamesOp::ToDf(strings(&["a", "b", "c"])));
    assert_eq!(names(r), ["a", "b", "c"]);
}

#[test]
fn to_df_with_a_wrong_count_is_refused() {
    // `toDF` is always a derived-table column list: Spark's arity error, parameters in its order.
    let e = run(false, ColumnNamesOp::ToDf(strings(&["a"]))).expect_err("a wrong count");
    assert_eq!(e.error.error_class().name(), "ASSIGNMENT_ARITY_MISMATCH");
    let params: std::collections::BTreeMap<_, _> = e
        .error
        .message_parameters()
        .iter()
        .map(|(k, v)| (*k, v.as_str()))
        .collect();
    assert_eq!(params["numTarget"], "3");
    assert_eq!(params["numExpr"], "1");
}

#[test]
fn rename_ignores_unknown_names_and_matches_case_insensitively() {
    let op = ColumnNamesOp::Rename(vec![
        ("YEAR".to_owned(), "y".to_owned()),
        ("nope".to_owned(), "z".to_owned()),
    ]);
    assert_eq!(names(run(false, op.clone())), ["course", "y", "earnings"]);
    // Case sensitive: `YEAR` matches nothing.
    assert_eq!(names(run(true, op)), ["course", "year", "earnings"]);
}

#[test]
fn rename_applies_the_pairs_in_turn() {
    let op = ColumnNamesOp::Rename(vec![
        ("year".to_owned(), "y".to_owned()),
        ("y".to_owned(), "w".to_owned()),
    ]);
    assert_eq!(names(run(false, op)), ["course", "w", "earnings"]);
}

#[test]
fn drop_leaves_out_matches_and_ignores_the_rest() {
    let op = drop_names(&["Year", "nope"]);
    assert_eq!(names(run(false, op.clone())), ["course", "earnings"]);
    assert_eq!(names(run(true, op)), ["course", "year", "earnings"]);
}

#[test]
fn drop_of_every_column_leaves_none() {
    let op = drop_names(&["course", "year", "earnings"]);
    assert!(names(run(false, op)).is_empty());
}

#[test]
fn rename_and_drop_by_name_act_on_every_duplicate() {
    let rename = ColumnNamesOp::Rename(vec![("year".to_owned(), "y".to_owned())]);
    let r = run_on(false, doubled(), rename, |p| p);
    assert_eq!(
        names(r),
        ["course", "y", "earnings", "course", "y", "earnings"]
    );
    let r = run_on(false, doubled(), drop_names(&["year"]), |p| p);
    assert_eq!(names(r), ["course", "earnings", "course", "earnings"]);
}

#[test]
fn drop_of_a_column_reference_is_resolved_like_any_reference() {
    // Ambiguous over a join that repeats the name: Spark's `AMBIGUOUS_REFERENCE`, not a drop of both.
    let e = run_on(false, doubled(), drop_columns(&["year"]), |p| p).expect_err("ambiguous");
    assert_eq!(e.error.error_class().name(), "AMBIGUOUS_REFERENCE");
    // One match drops that column alone; none is ignored.
    assert_eq!(
        names(run(false, drop_columns(&["Year"]))),
        ["course", "earnings"]
    );
    assert_eq!(
        names(run(false, drop_columns(&["nope"]))),
        ["course", "year", "earnings"]
    );
}

#[test]
fn rename_and_drop_keep_the_qualifier_of_untouched_columns_and_to_df_clears_it() {
    let read_year = |p: LogicalPlan| p.project(vec![Expr::column(["t", "year"])]);
    let input = || LogicalPlan::table(["sales"]).alias("t");
    let rename = ColumnNamesOp::Rename(vec![("course".to_owned(), "c".to_owned())]);
    assert_eq!(names(run_on(false, input(), rename, read_year)), ["year"]);
    assert_eq!(
        names(run_on(false, input(), drop_names(&["course"]), read_year)),
        ["year"]
    );
    let to_df = ColumnNamesOp::ToDf(strings(&["a", "year", "c"]));
    let Err(e) = run_on(false, input(), to_df, read_year) else {
        panic!("toDF drops the qualifier")
    };
    assert!(e.to_string().contains("UNRESOLVED_COLUMN"), "{e}");
}

fn literal(l: crate::Literal) -> Arc<Expr> {
    Arc::new(Expr::literal(l))
}

fn fill(cols: &[&str], values: Vec<Arc<Expr>>) -> ColumnNamesOp {
    ColumnNamesOp::FillNa {
        cols: strings(cols),
        values,
    }
}

#[test]
fn drop_na_keeps_the_columns_and_filters() {
    for min in [None, Some(1), Some(0)] {
        let op = ColumnNamesOp::DropNa {
            subset: vec![],
            min_non_nulls: min,
        };
        assert_eq!(
            names(run(false, op)),
            strings(&["course", "year", "earnings"])
        );
    }
    let op = ColumnNamesOp::DropNa {
        subset: strings(&["YEAR"]),
        min_non_nulls: None,
    };
    let plan = run(false, op).expect("a subset by name");
    assert!(
        format!("{plan:?}").contains("Filter"),
        "a filter over the input: {plan:?}"
    );
}

#[test]
fn drop_na_of_an_unknown_column_is_unresolved() {
    let op = ColumnNamesOp::DropNa {
        subset: strings(&["nope"]),
        min_non_nulls: None,
    };
    let Err(e) = run(false, op) else {
        panic!("expected UNRESOLVED_COLUMN")
    };
    assert!(e.to_string().contains("UNRESOLVED_COLUMN"), "{e}");
}

#[test]
fn fill_na_keeps_names_and_types_and_reaches_by_type() {
    use crate::Literal;
    let all = |v| run(false, fill(&[], vec![literal(v)])).expect("fill all columns");
    // A number reaches the two integer columns; a string only `course`.
    for plan in [all(Literal::Long(0)), all(Literal::String("x".into()))] {
        let types: Vec<_> = plan.output.iter().map(|c| c.data_type.clone()).collect();
        assert_eq!(
            types,
            vec![DataType::String, DataType::Integer, DataType::Integer]
        );
    }
    let numeric = format!("{:?}", all(Literal::Long(0)));
    let text = format!("{:?}", all(Literal::String("x".into())));
    assert!(numeric.contains("coalesce") && text.contains("coalesce"));
    assert_ne!(numeric, text);
    // A named column, case-insensitively; a boolean reaches none, so the input is unchanged.
    let named = run(false, fill(&["YEAR"], vec![literal(Literal::Long(7))])).expect("named");
    assert_eq!(named.output.len(), 3);
    let none = run(false, fill(&[], vec![literal(Literal::Boolean(true))])).expect("no target");
    assert!(!format!("{none:?}").contains("coalesce"), "{none:?}");
}

#[test]
fn fill_na_pairs_values_with_columns() {
    use crate::Literal;
    let op = fill(
        &["course", "year"],
        vec![
            literal(Literal::String("none".into())),
            literal(Literal::Long(1)),
        ],
    );
    assert_eq!(
        names(run(false, op)),
        strings(&["course", "year", "earnings"])
    );
}

#[test]
fn fill_na_with_a_value_per_column_casts_each_whatever_its_kind() {
    use crate::Literal;
    let coalesces = |op| {
        format!("{:?}", run(false, op).expect("fill"))
            .matches("coalesce")
            .count()
    };
    // One string value for an integer column fits no kind: `fill(value, subset)` leaves it.
    let single = fill(&["year"], vec![literal(Literal::String("5".into()))]);
    assert_eq!(coalesces(single), 0);
    // The map form casts "5" to the column's type, as Spark's `fillMap` does.
    let each = fill(
        &["year", "course"],
        vec![
            literal(Literal::String("5".into())),
            literal(Literal::String("x".into())),
        ],
    );
    assert_eq!(coalesces(each), 2);
}

#[test]
fn fill_na_of_an_unknown_column_is_unresolved() {
    let Err(e) = run(
        false,
        fill(&["nope"], vec![literal(crate::Literal::Long(1))]),
    ) else {
        panic!("a misspelled column must not be a no-op")
    };
    assert!(e.to_string().contains("UNRESOLVED_COLUMN"), "{e}");
}

#[test]
fn fill_na_of_an_untyped_value_is_unsupported() {
    let Err(e) = run(false, fill(&[], vec![literal(crate::Literal::Null(None))])) else {
        panic!("expected an error")
    };
    assert!(e.to_string().contains("UNSUPPORTED"), "{e}");
}

fn replace(cols: &[&str], pairs: Vec<Arc<Expr>>) -> ColumnNamesOp {
    ColumnNamesOp::Replace {
        cols: strings(cols),
        pairs,
    }
}

#[test]
fn replace_keeps_names_and_types_and_reaches_by_type() {
    use crate::Literal;
    let num = vec![literal(Literal::Long(1)), literal(Literal::Long(10))];
    let plan = run(false, replace(&[], num.clone())).expect("a numeric replace");
    let types: Vec<_> = plan.output.iter().map(|c| c.data_type.clone()).collect();
    assert_eq!(
        types,
        vec![DataType::String, DataType::Integer, DataType::Integer]
    );
    assert!(
        format!("{plan:?}").contains("CaseWhen")
            || format!("{plan:?}").to_lowercase().contains("when")
    );
    // A string key reaches only `course`; a boolean key reaches none, so the plan is unchanged.
    let text = run(
        false,
        replace(
            &[],
            vec![
                literal(Literal::String("a".into())),
                literal(Literal::String("z".into())),
            ],
        ),
    )
    .expect("a string replace");
    assert_ne!(format!("{plan:?}"), format!("{text:?}"));
    let none = run(
        false,
        replace(
            &[],
            vec![
                literal(Literal::Boolean(true)),
                literal(Literal::Boolean(false)),
            ],
        ),
    )
    .expect("no target");
    assert!(
        !format!("{none:?}").to_lowercase().contains("when"),
        "{none:?}"
    );
    // A named column, case-insensitively; a missing name is an error.
    run(false, replace(&["YEAR"], num.clone())).expect("named");
    assert!(run(false, replace(&["nope"], num)).is_err());
}

#[test]
fn replace_refuses_a_null_or_mixed_type_old_value() {
    use crate::Literal;
    let null_key = vec![literal(Literal::Null(None)), literal(Literal::Long(1))];
    assert!(run(false, replace(&[], null_key)).is_err());
    let mixed = vec![
        literal(Literal::Long(1)),
        literal(Literal::Long(2)),
        literal(Literal::String("a".into())),
        literal(Literal::String("b".into())),
    ];
    assert!(run(false, replace(&[], mixed)).is_err());
}

#[test]
fn na_subset_names_are_attribute_references() {
    // A backquoted name is the column it quotes (`df.col("`year`")`), as in fill, replace and drop.
    let quoted = strings(&["`year`"]);
    let drop = ColumnNamesOp::DropNa {
        subset: quoted,
        min_non_nulls: None,
    };
    assert!(format!("{:?}", run(false, drop).expect("quoted")).contains("Filter"));
    let fill = fill(&["`year`"], vec![literal(crate::Literal::Long(0))]);
    assert_eq!(
        names(run(false, fill)),
        strings(&["course", "year", "earnings"])
    );
    assert!(run(false, replace(&["`year`"], vec![])).is_ok());
    // Quotes that do not close name no column.
    let bad = ColumnNamesOp::DropNa {
        subset: strings(&["`ye``ar`"]),
        min_non_nulls: None,
    };
    assert!(run(false, bad).is_err());
}

#[test]
fn drop_na_over_no_columns_keeps_no_row_when_a_value_is_needed() {
    let none = LogicalPlan::new(PlanNode::ColumnNames {
        input: Arc::new(LogicalPlan::table(["sales"])),
        op: drop_columns(&["course", "year", "earnings"]),
    });
    for (min, filtered) in [(Some(1), true), (None, false), (Some(0), false)] {
        let op = ColumnNamesOp::DropNa {
            subset: vec![],
            min_non_nulls: min,
        };
        let plan = run_on(false, none.clone(), op, |p| p).expect("analyzes");
        assert!(plan.output.is_empty());
        assert_eq!(
            format!("{plan:?}").contains("Filter"),
            filtered,
            "min {min:?}"
        );
    }
}
