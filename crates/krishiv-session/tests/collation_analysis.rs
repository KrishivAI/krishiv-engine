// SPDX-License-Identifier: Apache-2.0
//! `COLLATE` and `collate()` through the analyzer (collations unit 2a, ADR 0043), with the
//! names, types and errors of the Spark 4.2.0 `collations-basic` golden.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

fn run(sql: &str) -> Result<Vec<String>, KrishivError> {
    let s = Session::new()?;
    let df = block_on(s.sql(sql))?;
    let batches = block_on(df.collect())?;
    let mut out = Vec::new();
    for b in &batches {
        let c = &b.columns()[0];
        for i in 0..c.len() {
            out.push(if c.is_null(i) {
                "NULL".to_owned()
            } else if let Some(v) = c.as_string_view_opt() {
                v.value(i).to_owned()
            } else if let Some(v) = c.as_string_opt::<i32>() {
                v.value(i).to_owned()
            } else {
                format!("{c:?}")
            });
        }
    }
    Ok(out)
}

fn one(sql: &str) -> String {
    run(sql).unwrap().remove(0)
}

fn class(sql: &str) -> ErrorClass {
    run(sql).unwrap_err().error_class()
}

#[test]
fn collate_gives_the_collated_type() {
    assert_eq!(
        one("SELECT typeof('a' COLLATE utf8_lcase)"),
        "string collate UTF8_LCASE"
    );
    assert_eq!(
        one("SELECT typeof(collate('a', 'UNICODE_CI'))"),
        "string collate UNICODE_CI"
    );
    // UTF8_BINARY is the plain string type.
    assert_eq!(one("SELECT typeof('a' COLLATE UTF8_BINARY)"), "string");
    // Names are case-insensitive and may be qualified.
    assert_eq!(
        one("SELECT typeof('a' COLLATE SYSTEM.BUILTIN.utf8_lcase)"),
        "string collate UTF8_LCASE"
    );
}

#[test]
fn collate_keeps_the_value() {
    assert_eq!(one("SELECT 'AbC' COLLATE utf8_lcase"), "AbC");
}

#[test]
fn collate_checks_its_input_and_its_name() {
    assert_eq!(
        class("SELECT 1 COLLATE utf8_lcase"),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert_eq!(
        class("SELECT 'a' COLLATE no_such_collation"),
        ErrorClass::CollationInvalidName
    );
}

/// Until the planner rewrites comparing positions onto collation keys, every one of them
/// refuses a collated value instead of comparing bytes (the execution guard of unit 2a).
#[test]
fn comparing_positions_refuse_a_collated_value() {
    let c = "('a' COLLATE utf8_lcase)";
    for sql in [
        // scalar comparisons
        format!("SELECT {c} = 'A'"),
        format!("SELECT {c} <=> 'A'"),
        format!("SELECT {c} < 'A'"),
        format!("SELECT {c} BETWEEN 'a' AND 'b'"),
        format!("SELECT {c} IN ('A', 'b')"),
        // keys
        format!("SELECT col1 FROM (SELECT {c} AS col1) GROUP BY col1"),
        format!("SELECT DISTINCT {c}"),
        format!("SELECT col1 FROM (SELECT {c} AS col1) ORDER BY col1"),
        format!("SELECT count(DISTINCT c) FROM (SELECT {c} AS c)"),
        format!("SELECT max(c) FROM (SELECT {c} AS c)"),
        format!("SELECT c, row_number() OVER (PARTITION BY c ORDER BY c) FROM (SELECT {c} AS c)"),
        format!("SELECT {c} UNION SELECT {c}"),
        format!("SELECT {c} INTERSECT SELECT {c}"),
        format!("SELECT {c} EXCEPT SELECT {c}"),
        format!("SELECT * FROM (SELECT {c} AS c) a JOIN (SELECT {c} AS c) b ON a.c = b.c"),
    ] {
        assert_eq!(
            class(&sql),
            ErrorClass::KrishivUnsupportedFeature,
            "{sql} must be refused, not compared as bytes"
        );
    }
}

/// What does not compare may still use a collated value.
#[test]
fn non_comparing_positions_accept_a_collated_value() {
    assert_eq!(one("SELECT 'a' COLLATE utf8_lcase"), "a");
    assert_eq!(
        one("SELECT c FROM (SELECT 'Ab' COLLATE utf8_lcase AS c) WHERE true"),
        "Ab"
    );
    assert_eq!(
        one("SELECT CAST(count(c) AS STRING) FROM (SELECT 'a' COLLATE utf8_lcase AS c)"),
        "1"
    );
}

#[test]
fn collation_names_the_collation_of_its_argument() {
    assert_eq!(
        one("SELECT collation('a' COLLATE utf8_lcase)"),
        "SYSTEM.BUILTIN.UTF8_LCASE"
    );
    assert_eq!(one("SELECT collation('a')"), "SYSTEM.BUILTIN.UTF8_BINARY");
    assert_eq!(
        one("SELECT collation('a' COLLATE en_ci)"),
        "SYSTEM.BUILTIN.en_CI"
    );
    assert_eq!(
        class("SELECT collation(1)"),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
}

/// A table with collated columns: created, written, read back with the type kept.
#[test]
fn collated_columns_in_tables_keep_their_type() {
    let s = Session::new().unwrap();
    for sql in [
        "create table t1(a string collate utf8_binary, b string collate utf8_lcase) using parquet",
        "insert into t1 values('aaa', 'Bbb')",
    ] {
        block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    }
    let first = |sql: &str| -> String {
        let b = block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
        let c = &b[0].columns()[0];
        c.as_string_opt::<i32>().map_or_else(
            || c.as_string_view().value(0).to_owned(),
            |v| v.value(0).to_owned(),
        )
    };
    assert_eq!(first("select typeof(a) from t1"), "string");
    assert_eq!(
        first("select typeof(b) from t1"),
        "string collate UTF8_LCASE"
    );
    assert_eq!(first("select b from t1"), "Bbb");
    assert_eq!(
        first("select collation(b) from t1"),
        "SYSTEM.BUILTIN.UTF8_LCASE"
    );
}

/// A string function over collated strings takes their one collation (Spark's
/// `CollationTypeCoercion`): explicit beats implicit beats default.
#[test]
fn string_functions_take_the_collation_of_their_arguments() {
    assert_eq!(
        one("SELECT typeof(concat('a' COLLATE utf8_lcase, 'b'))"),
        "string collate UTF8_LCASE"
    );
    assert_eq!(
        one("SELECT typeof(upper('a' COLLATE unicode_ci))"),
        "string collate UNICODE_CI"
    );
    // The same explicit collation twice is fine; the value is untouched.
    assert_eq!(
        one("SELECT concat('a' COLLATE utf8_lcase, 'B' COLLATE utf8_lcase)"),
        "aB"
    );
    // A function with a non-string result keeps it.
    assert_eq!(one("SELECT typeof(length('a' COLLATE utf8_lcase))"), "int");
    // `collation()` itself is a plain string.
    assert_eq!(
        one("SELECT typeof(collation('a' COLLATE utf8_lcase))"),
        "string"
    );
}

#[test]
fn conflicting_collations_are_refused_like_spark() {
    assert_eq!(
        class("SELECT concat('a' COLLATE utf8_lcase, 'b' COLLATE unicode_ci)"),
        ErrorClass::CollationMismatchExplicit
    );
    // An explicit collation outranks an implicit one (a column).
    let s = Session::new().unwrap();
    for sql in [
        "create table t1(a string collate utf8_binary, b string collate utf8_lcase) using parquet",
        "insert into t1 values('x', 'y')",
    ] {
        block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    }
    let ok = block_on(
        block_on(s.sql("select typeof(concat(b, 'z' collate unicode_ci)) from t1"))
            .unwrap()
            .collect(),
    );
    assert!(ok.is_ok());
    // Two different implicit collations are indeterminate.
    let err = block_on(s.sql("select trim(a, b) from t1")).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::IndeterminateCollationInExpression
    );
}

/// Accent-insensitive ICU collations refuse string search; CI ones do not. Only the calls Spark
/// lists fail on operands without one collation; others (`levenshtein`) just run.
#[test]
fn search_and_indeterminate_rules_follow_spark() {
    assert_eq!(
        class("SELECT contains('Ab' COLLATE unicode_ai, 'a')"),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert_eq!(
        class("SELECT str_to_map('a:b' COLLATE unicode_ai)"),
        ErrorClass::DatatypeMismatchUnexpectedInputType
    );
    assert!(run("SELECT replace('aAb' COLLATE unicode_ci, 'a', 'x')").is_ok());
    let s = Session::new().unwrap();
    for sql in [
        "create table t1(a string collate utf8_binary, b string collate utf8_lcase) using parquet",
        "insert into t1 values('abc', 'abd')",
    ] {
        block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    }
    assert!(block_on(s.sql("select levenshtein(a, b) from t1")).is_ok());
}
