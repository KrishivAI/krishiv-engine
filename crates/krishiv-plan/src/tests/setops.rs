// SPDX-License-Identifier: Apache-2.0
//! `unionByName` and `INTERSECT` / `EXCEPT` as DataFrames build them:
//! output names, types and nullability, and errors, as Spark 4.2 gives
//! them (probed with PySpark against Spark Connect: `a` is
//! `SELECT 1 AS k, 'a' AS x, 10L AS y`, and so on).

use std::sync::Arc;

use krishiv_types::DataType;
use serde_json::{Value, json};

use crate::analyzer::testing::{block_on, context};
use crate::analyzer::{
    AnalysisError, AnalyzedPlan, Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf,
    TableColumn,
};
use crate::{LogicalPlan, Plan, PlanNode, SetOperation};

fn table(cat: MemoryCatalog, name: &str, cols: &[(&str, DataType, bool)]) -> MemoryCatalog {
    let columns = cols
        .iter()
        .map(|(n, t, nullable)| TableColumn {
            name: (*n).to_owned(),
            data_type: t.clone(),
            nullable: *nullable,
            char_varchar: None,
        })
        .collect();
    cat.with_table(&["default"], name, columns, None)
}

fn catalog() -> MemoryCatalog {
    use DataType::{Integer as I, Long as L, String as S};
    let cat = MemoryCatalog::new("spark_catalog");
    let cat = table(
        cat,
        "a",
        &[("k", I, false), ("x", S, false), ("y", L, false)],
    );
    let cat = table(
        cat,
        "b",
        &[("X", S, false), ("K", L, false), ("y", I, false)],
    );
    let cat = table(cat, "c", &[("k", I, false), ("x", S, false)]);
    let cat = table(
        cat,
        "d",
        &[("k", I, false), ("x", S, false), ("z", I, false)],
    );
    let cat = table(cat, "e", &[("k", I, false), ("k", I, false)]);
    let cat = table(cat, "f", &[("K", I, false), ("k", I, false)]);
    let cat = table(cat, "l", &[("k", I, true), ("x", S, true)]);
    table(cat, "n", &[("k", I, false), ("x", S, false)])
}

fn analyze(plan: LogicalPlan, case_sensitive: bool) -> Result<AnalyzedPlan, AnalysisError> {
    let conf = ResolutionConf {
        case_sensitive,
        ..ResolutionConf::default()
    };
    let ctx = context(catalog(), conf);
    let plan = Plan::try_new(plan).map_err(|error| AnalysisError {
        error,
        origin: crate::Origin::UNKNOWN,
    })?;
    block_on(DefaultAnalyzer.analyze(&plan, &ctx))
}

/// `name type nullable` per column, or the error class and parameters.
fn summary(r: &Result<AnalyzedPlan, AnalysisError>) -> Value {
    match r {
        Ok(a) => json!(
            a.output
                .iter()
                .map(|c| format!("{} {} {}", c.name, c.data_type, c.nullable))
                .collect::<Vec<_>>()
        ),
        Err(e) => json!([
            e.error.error_class().name(),
            e.error
                .message_parameters()
                .iter()
                .map(|(k, v)| (*k, v))
                .collect::<std::collections::BTreeMap<_, _>>()
        ]),
    }
}

fn by_name(l: &str, r: &str, allow_missing_columns: bool) -> LogicalPlan {
    LogicalPlan::new(PlanNode::Union {
        inputs: vec![
            Arc::new(LogicalPlan::table([l])),
            Arc::new(LogicalPlan::table([r])),
        ],
        by_name: true,
        allow_missing_columns,
    })
}

fn set_op(l: &str, r: &str, op: SetOperation, all: bool) -> LogicalPlan {
    LogicalPlan::new(PlanNode::SetOp {
        left: Arc::new(LogicalPlan::table([l])),
        right: Arc::new(LogicalPlan::table([r])),
        op,
        all,
    })
}

#[test]
fn union_by_name_matches_spark() {
    let cases = [
        (
            by_name("a", "b", false),
            false,
            json!(["k bigint false", "x string false", "y bigint false"]),
        ),
        (
            by_name("b", "a", false),
            false,
            json!(["X string false", "K bigint false", "y bigint false"]),
        ),
        (
            by_name("a", "c", false),
            false,
            json!([
                "UNRESOLVED_COLUMN_AMONG_FIELD_NAMES",
                {"colName": "y", "fieldNames": "k, x"}
            ]),
        ),
        (
            by_name("c", "a", false),
            false,
            json!([
                "NUM_COLUMNS_MISMATCH",
                {
                    "firstNumColumns": "2",
                    "invalidNumColumns": "3",
                    "invalidOrdinalNum": "second",
                    "operator": "UNION"
                }
            ]),
        ),
        (
            by_name("a", "c", true),
            false,
            json!(["k int false", "x string false", "y bigint true"]),
        ),
        (
            by_name("c", "a", true),
            false,
            json!(["k int false", "x string false", "y bigint true"]),
        ),
        (
            by_name("c", "d", true),
            false,
            json!(["k int false", "x string false", "z int true"]),
        ),
        (
            by_name("e", "e", false),
            false,
            json!(["COLUMN_ALREADY_EXISTS", {"columnName": "`k`"}]),
        ),
        (
            by_name("a", "f", true),
            false,
            json!(["COLUMN_ALREADY_EXISTS", {"columnName": "`k`"}]),
        ),
        (
            by_name("a", "a", false),
            false,
            json!(["k int false", "x string false", "y bigint false"]),
        ),
        (
            by_name("a", "b", false),
            true,
            json!([
                "UNRESOLVED_COLUMN_AMONG_FIELD_NAMES",
                {"colName": "k", "fieldNames": "X, K, y"}
            ]),
        ),
        (
            by_name("a", "b", true),
            true,
            json!([
                "k int true",
                "x string true",
                "y bigint false",
                "X string true",
                "K bigint true"
            ]),
        ),
    ];
    for (i, (plan, cs, want)) in cases.into_iter().enumerate() {
        assert_eq!(summary(&analyze(plan, cs)), want, "case {i}");
    }
}

#[test]
fn intersect_and_except_outputs_match_spark() {
    use SetOperation::{Except, Intersect};
    let cases = [
        // An intersection's column is nullable only where both sides' are.
        (
            set_op("n", "l", Intersect, false),
            json!(["k int false", "x string false"]),
        ),
        (
            set_op("l", "n", Intersect, true),
            json!(["k int false", "x string false"]),
        ),
        // An except keeps the left side's.
        (
            set_op("l", "n", Except, true),
            json!(["k int true", "x string true"]),
        ),
        (
            set_op("n", "l", Except, false),
            json!(["k int false", "x string false"]),
        ),
        (
            set_op("c", "a", Except, true),
            json!([
                "NUM_COLUMNS_MISMATCH",
                {
                    "firstNumColumns": "2",
                    "invalidNumColumns": "3",
                    "invalidOrdinalNum": "second",
                    "operator": "EXCEPT ALL"
                }
            ]),
        ),
        (
            set_op("c", "b", Intersect, false),
            json!([
                "NUM_COLUMNS_MISMATCH",
                {
                    "firstNumColumns": "2",
                    "invalidNumColumns": "3",
                    "invalidOrdinalNum": "second",
                    "operator": "INTERSECT"
                }
            ]),
        ),
    ];
    for (i, (plan, want)) in cases.into_iter().enumerate() {
        assert_eq!(summary(&analyze(plan, false)), want, "case {i}");
    }
}

#[test]
fn set_operations_resolve_away() {
    // The resolved plan has no set operation left (the physical planner
    // knows none), and analyzing it again changes nothing.
    for all in [false, true] {
        for op in [SetOperation::Intersect, SetOperation::Except] {
            let a = analyze(set_op("l", "l", op, all), false).unwrap();
            let text = a.plan.to_string();
            assert!(
                !text.contains("Intersect") && !text.contains("Except"),
                "{text}"
            );
            assert_eq!(text.contains("replicaterows"), all, "{text}");
            let again = analyze(LogicalPlan::clone(a.plan.root()), false).unwrap();
            assert_eq!(again.output, a.output);
            assert_eq!(again.plan.to_string(), text);
        }
    }
}

#[test]
fn set_operations_display_as_spark_names_them() {
    use SetOperation::{Except, Intersect};
    for (op, all, want) in [
        (Intersect, false, "Intersect false"),
        (Intersect, true, "Intersect All true"),
        (Except, false, "Except false"),
        (Except, true, "Except All true"),
    ] {
        let text = set_op("l", "n", op, all).to_string();
        assert!(text.starts_with(want), "{text}");
    }
}

#[test]
fn union_by_name_duplicates_compare_as_the_session_does() {
    // Case-sensitive, `K` and `k` are two columns; `k` twice is not.
    assert_eq!(
        summary(&analyze(by_name("f", "f", false), true)),
        json!(["K int false", "k int false"])
    );
    assert_eq!(
        summary(&analyze(by_name("e", "a", false), true)),
        json!(["COLUMN_ALREADY_EXISTS", {"columnName": "`k`"}])
    );
}

/// A union input that repeats a column gets a projection re-aliasing the
/// repeat (Spark's `DeduplicateUnionChildOutput`: `Project [k#1, k#1 AS
/// k#n]`), `k` above the union is the first copy, and re-analysis is a
/// fixed point. The analyzed union, analyzed again under another operator
/// (a DataFrame temporary view stores the analyzed plan), still reads `k`
/// as the first copy.
#[test]
fn a_union_input_that_repeats_a_column_gets_distinct_attributes() {
    use super::dsl::c;
    let union = |first: LogicalPlan, second: LogicalPlan| {
        LogicalPlan::new(PlanNode::Union {
            inputs: vec![Arc::new(first), Arc::new(second)],
            by_name: false,
            allow_missing_columns: false,
        })
    };
    // Both `INT`: no widening cast replaces the repeat.
    let u = union(
        LogicalPlan::table(["a"]).project(vec![c("k"), c("k")]),
        LogicalPlan::table(["c"]).project(vec![c("k"), c("k")]),
    );
    let a = analyze(u.clone().project(vec![c("k")]), false).unwrap();
    assert_eq!(a.output.len(), 1);
    let text = a.plan.root().to_string();
    assert!(
        text.lines()
            .any(|l| l.trim_start().starts_with(":- Project [k#") && l.contains(" AS k#")),
        "the repeat is re-aliased:\n{text}"
    );
    let ctx = context(catalog(), ResolutionConf::default());
    let again = block_on(DefaultAnalyzer.analyze(&a.plan, &ctx)).unwrap();
    assert!(again.plan.structurally_eq(&a.plan), "a fixed point");
    // The resolved union, read again by name.
    let resolved = analyze(u, false).unwrap();
    let over = (**resolved.plan.root()).clone().project(vec![c("k")]);
    let r = analyze(over, false).unwrap();
    assert_eq!(r.output.len(), 1);
}
