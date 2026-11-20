// SPDX-License-Identifier: Apache-2.0
//! `ColumnEdit` (`withColumnsRenamed`, `drop`) and an alias-less `SubqueryAlias` (`toDF`) on
//! hand-built plans, against the answers of a local Spark 4.2.0 `Dataset` (probe 2026-11-20).

use std::sync::Arc;

use krishiv_types::DataType;

use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    AnalysisError, Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use crate::{ColumnEdit, LogicalPlan, Origin, Plan, PlanNode};

fn catalog() -> MemoryCatalog {
    let col = |name: &str| TableColumn {
        name: name.to_owned(),
        data_type: DataType::Integer,
        nullable: true,
        char_varchar: None,
    };
    MemoryCatalog::new("spark_catalog").with_table(
        &["default"],
        "t",
        vec![col("a"), col("B"), col("a2")],
        None,
    )
}

fn run(plan: LogicalPlan, case_sensitive: bool) -> Result<Vec<String>, AnalysisError> {
    let plan = Plan::try_new(plan).map_err(|error| AnalysisError {
        error,
        origin: Origin::UNKNOWN,
    })?;
    let conf = ResolutionConf {
        case_sensitive,
        ..ResolutionConf::default()
    };
    let a = block_on(DefaultAnalyzer.analyze(&plan, &context(catalog(), conf)))?;
    assert!(a.plan.root().is_resolved());
    Ok(a.output.iter().map(|c| c.name.clone()).collect())
}

fn edit(edit: ColumnEdit) -> LogicalPlan {
    LogicalPlan::new(PlanNode::ColumnEdit {
        input: Arc::new(LogicalPlan::table(["t"])),
        edit,
    })
}

fn rename(pairs: &[(&str, &str)]) -> LogicalPlan {
    edit(ColumnEdit::Rename(
        pairs
            .iter()
            .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
            .collect(),
    ))
}

fn drop_of(names: &[&str]) -> LogicalPlan {
    edit(ColumnEdit::Drop(
        names.iter().map(|n| (*n).to_owned()).collect(),
    ))
}

fn to_df(names: &[&str]) -> LogicalPlan {
    LogicalPlan::new(PlanNode::SubqueryAlias {
        input: Arc::new(LogicalPlan::table(["t"])),
        alias: String::new(),
        columns: names.iter().map(|n| (*n).to_owned()).collect(),
    })
}

fn ok(plan: LogicalPlan) -> Vec<String> {
    run(plan, false).unwrap_or_else(|e| panic!("expected columns, got {e}"))
}

#[test]
fn rename_matches_ignoring_case_and_skips_unknown_names() {
    assert_eq!(ok(rename(&[("b", "q")])), ["a", "q", "a2"]);
    assert_eq!(ok(rename(&[("zz", "q")])), ["a", "B", "a2"]);
}

#[test]
fn renames_apply_one_after_another() {
    // `{a: B, B: a}` is not a swap: the second rename sees the first's result.
    assert_eq!(ok(rename(&[("a", "B"), ("B", "a")])), ["a", "a", "a2"]);
    // The second key no longer matches anything once `a` is `m`.
    assert_eq!(ok(rename(&[("a", "m"), ("A", "n")])), ["m", "B", "a2"]);
    // Duplicate new names are allowed.
    assert_eq!(ok(rename(&[("a", "k"), ("B", "k")])), ["k", "k", "a2"]);
}

#[test]
fn rename_respects_case_sensitivity() {
    let r = run(rename(&[("b", "q")]), true).unwrap();
    assert_eq!(r, ["a", "B", "a2"]);
    let r = run(rename(&[("B", "q")]), true).unwrap();
    assert_eq!(r, ["a", "q", "a2"]);
}

#[test]
fn drop_matches_ignoring_case_and_skips_unknown_names() {
    assert_eq!(ok(drop_of(&["b"])), ["a", "a2"]);
    assert_eq!(ok(drop_of(&["zz"])), ["a", "B", "a2"]);
    assert_eq!(ok(drop_of(&["a.b"])), ["a", "B", "a2"]);
    assert!(ok(drop_of(&["a", "B", "a2"])).is_empty());
    assert_eq!(run(drop_of(&["b"]), true).unwrap(), ["a", "B", "a2"]);
}

#[test]
fn to_df_renames_by_position_and_keeps_repeated_names() {
    assert_eq!(ok(to_df(&["x", "y", "x"])), ["x", "y", "x"]);
}

#[test]
fn to_df_with_the_wrong_number_of_names_is_an_arity_mismatch() {
    // (No names at all means "keep the columns" to `SubqueryAlias`; the Connect translator
    // refuses an empty `toDF`.)
    for names in [&["x", "y"][..], &["x", "y", "z", "w"]] {
        let n = names.len();
        let e = run(to_df(names), false).unwrap_err();
        assert_eq!(e.error.error_class().name(), "ASSIGNMENT_ARITY_MISMATCH");
        let params = e.error.message_parameters();
        assert!(params.iter().any(|(k, v)| *k == "numTarget" && v == "3"));
        assert!(
            params
                .iter()
                .any(|(k, v)| *k == "numExpr" && *v == n.to_string())
        );
    }
}
