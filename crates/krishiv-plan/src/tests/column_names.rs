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
    assert!(run_on(false, input(), to_df, read_year).is_err());
}
