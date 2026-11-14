// SPDX-License-Identifier: Apache-2.0
//! Generators in the DataFrame API (`select`, `select_expr`, `with_column`): the same rows and
//! errors as SQL `SELECT explode(..)`, Spark 4.2.0's.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::functions::{call_function, col};
use krishiv_session::{DataFrame, Session};

const T: &str = "values (1, array(10, 20)), (2, array()), (3, cast(null as array<int>)) as t(a, b)";

fn input(s: &Session) -> DataFrame {
    block_on(s.sql(&format!("select * from {T}"))).unwrap()
}

/// Every row of `df` as text, sorted.
fn rows(df: &DataFrame) -> Vec<Vec<String>> {
    let batches = block_on(df.collect()).unwrap_or_else(|e| panic!("{e}"));
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|c| {
                        if c.is_null(i) {
                            "NULL".to_owned()
                        } else {
                            c.as_primitive::<arrow_array::types::Int32Type>()
                                .value(i)
                                .to_string()
                        }
                    })
                    .collect(),
            );
        }
    }
    out.sort();
    out
}

fn names(df: &DataFrame) -> Vec<String> {
    block_on(df.schema())
        .unwrap()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

fn error_of(df: &DataFrame) -> KrishivError {
    block_on(df.collect()).expect_err("ran")
}

fn t(rows: &[&[&str]]) -> Vec<Vec<String>> {
    rows.iter()
        .map(|r| r.iter().map(|c| (*c).to_owned()).collect())
        .collect()
}

#[test]
fn select_lifts_a_generator_beside_other_columns() {
    let s = Session::new().unwrap();
    let df = input(&s).select([col("a"), call_function("explode", [col("b")])]);
    assert_eq!(names(&df), ["a", "col"]);
    assert_eq!(rows(&df), t(&[&["1", "10"], &["1", "20"]]));
    let outer = input(&s).select([
        col("a"),
        call_function("explode_outer", [col("b")]).alias("v"),
    ]);
    assert_eq!(names(&outer), ["a", "v"]);
    assert_eq!(
        rows(&outer),
        t(&[&["1", "10"], &["1", "20"], &["2", "NULL"], &["3", "NULL"]])
    );
}

#[test]
fn select_expr_takes_alias_lists_and_several_generators() {
    let s = Session::new().unwrap();
    let df = input(&s)
        .select_expr(["a", "posexplode(b) as (p, v)"])
        .unwrap();
    assert_eq!(names(&df), ["a", "p", "v"]);
    assert_eq!(rows(&df), t(&[&["1", "0", "10"], &["1", "1", "20"]]));
    let cross = input(&s)
        .select_expr(["explode(b) as x", "explode(b) as y"])
        .unwrap();
    assert_eq!(
        rows(&cross),
        t(&[&["10", "10"], &["10", "20"], &["20", "10"], &["20", "20"]])
    );
}

#[test]
fn alias_all_names_a_generators_columns() {
    let s = Session::new().unwrap();
    let df = input(&s).select([
        col("a"),
        call_function("posexplode", [col("b")]).alias_all(vec!["p".into(), "v".into()]),
    ]);
    assert_eq!(names(&df), ["a", "p", "v"]);
    // Two names over a one-column generator: Spark's `UDTF_ALIAS_NUMBER_MISMATCH`.
    let bad = input(&s)
        .select([call_function("explode", [col("b")]).alias_all(vec!["p".into(), "v".into()])]);
    assert!(
        error_of(&bad)
            .to_string()
            .contains("UDTF_ALIAS_NUMBER_MISMATCH")
    );
    // Over anything else: `MULTI_ALIAS_WITHOUT_GENERATOR`.
    let plain = input(&s).select([col("a").alias_all(vec!["p".into(), "v".into()])]);
    assert_eq!(
        error_of(&plain).error_class(),
        ErrorClass::MultiAliasWithoutGenerator
    );
}

#[test]
fn with_column_replaces_in_place_or_appends() {
    let s = Session::new().unwrap();
    let appended = input(&s).with_column("x", call_function("explode", [col("b")]));
    assert_eq!(names(&appended), ["a", "b", "x"]);
    assert_eq!(appended_count(&appended), 2);
    let replaced = input(&s).with_column("b", call_function("explode", [col("b")]));
    assert_eq!(names(&replaced), ["a", "b"]);
    assert_eq!(rows(&replaced), t(&[&["1", "10"], &["1", "20"]]));
    // A generator of several columns under one name.
    let bad = input(&s).with_column("x", call_function("posexplode", [col("b")]));
    assert!(
        error_of(&bad)
            .to_string()
            .contains("UDTF_ALIAS_NUMBER_MISMATCH")
    );
}

fn appended_count(df: &DataFrame) -> usize {
    block_on(df.collect())
        .unwrap()
        .iter()
        .map(arrow_array::RecordBatch::num_rows)
        .sum()
}

#[test]
fn a_generator_in_an_expression_is_refused() {
    let s = Session::new().unwrap();
    let df = input(&s).select_expr(["explode(b) + 1"]).unwrap();
    assert!(
        error_of(&df)
            .to_string()
            .contains("UNSUPPORTED_GENERATOR.NESTED_IN_EXPRESSIONS")
    );
}
