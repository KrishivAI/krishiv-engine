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
use crate::{ColumnNamesOp, LogicalPlan, Origin, Plan, PlanNode};

fn run(case_sensitive: bool, op: ColumnNamesOp) -> Result<AnalyzedPlan, AnalysisError> {
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
    let plan = Plan::try_new(LogicalPlan::new(PlanNode::ColumnNames {
        input: Arc::new(LogicalPlan::table(["sales"])),
        op,
    }))
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

#[test]
fn to_df_renames_by_position() {
    let r = run(false, ColumnNamesOp::ToDf(strings(&["a", "b", "c"])));
    assert_eq!(names(r), ["a", "b", "c"]);
}

#[test]
fn to_df_with_a_wrong_count_is_refused() {
    let r = run(false, ColumnNamesOp::ToDf(strings(&["a"])));
    assert!(r.is_err());
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
    let op = ColumnNamesOp::Drop(strings(&["Year", "nope"]));
    assert_eq!(names(run(false, op.clone())), ["course", "earnings"]);
    assert_eq!(names(run(true, op)), ["course", "year", "earnings"]);
}

#[test]
fn drop_of_every_column_leaves_none() {
    let op = ColumnNamesOp::Drop(strings(&["course", "year", "earnings"]));
    assert!(names(run(false, op)).is_empty());
}
