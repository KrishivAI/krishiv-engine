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
    // A written UTF8_BINARY is the display-only marker: Spark prints it, nothing else sees it.
    assert_eq!(
        one("SELECT typeof('a' COLLATE UTF8_BINARY)"),
        "string collate UTF8_BINARY"
    );
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

/// A collated value with no key (ICU, unit 6) is refused in a comparing position instead of being
/// compared as bytes; `collation_keys.rs` runs every key-bearing operator.
#[test]
fn comparing_positions_refuse_a_collated_value() {
    let ci = "('a' COLLATE unicode_ci)";
    for sql in [
        format!("SELECT {ci} = 'A'"),
        format!("SELECT {ci} <=> 'A'"),
        format!("SELECT {ci} < 'A'"),
        format!("SELECT {ci} BETWEEN 'a' AND 'b'"),
        format!("SELECT {ci} IN ('A', 'b')"),
        format!("SELECT * FROM (SELECT {ci} AS c) a JOIN (SELECT {ci} AS c) b ON a.c = b.c"),
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
    // An untyped NULL is the default string.
    assert_eq!(one("SELECT collation(NULL)"), "SYSTEM.BUILTIN.UTF8_BINARY");
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
    assert_eq!(
        first("select typeof(a) from t1"),
        "string collate UTF8_BINARY"
    );
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
        one("SELECT typeof(upper('a' COLLATE utf8_lcase))"),
        "string collate UTF8_LCASE"
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
    // An explicit collation outranks the column's implicit one.
    assert_eq!(
        value_of(
            &s,
            "select typeof(concat(b, 'z' collate unicode_ci)) from t1"
        ),
        "string collate UNICODE_CI"
    );
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
    let s = Session::new().unwrap();
    for sql in [
        "create table t1(a string collate utf8_binary, b string collate utf8_lcase) using parquet",
        "insert into t1 values('abc', 'abcd')",
    ] {
        block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    }
    // `levenshtein` is not on Spark's indeterminate list: it just runs ('abc' to 'abcd').
    assert_eq!(
        value_of(&s, "select cast(levenshtein(a, b) as string) from t1"),
        "1"
    );
}

/// Calls whose answer depends on the collation refuse a non-binary one instead of matching
/// bytes (unit 5 brings the collation-aware kernels; `contains`, `startswith` and `endswith`
/// over `UTF8_LCASE` are there, unit 5a, and so are the trim family, `replace`, `translate`,
/// `split_part` and `substring_index`, unit 5c); binary strings are untouched.
#[test]
fn collation_dependent_calls_refuse_instead_of_matching_bytes() {
    let l = "('Ab' COLLATE utf8_lcase)";
    for sql in [
        format!("SELECT {l} LIKE 'a%'"),
        format!("SELECT {l} RLIKE '^a'"),
        format!("SELECT regexp_replace({l}, 'a', 'x')"),
        "SELECT replace('aAb' COLLATE unicode_ci, 'a', 'x')".to_owned(),
        "SELECT translate('aAb' COLLATE unicode_ci, 'a', 'x')".to_owned(),
        "SELECT trim('a' FROM 'aAb' COLLATE unicode_ci)".to_owned(),
        "SELECT split_part('aAb' COLLATE unicode_ci, 'x', 1)".to_owned(),
        "SELECT substring_index('aAb' COLLATE unicode_ci, 'x', 1)".to_owned(),
        "SELECT 'a ' COLLATE utf8_binary_rtrim LIKE 'a'".to_owned(),
        "SELECT upper('a' COLLATE unicode_ci)".to_owned(),
    ] {
        assert_eq!(
            class(&sql),
            ErrorClass::KrishivUnsupportedFeature,
            "{sql} must be refused, not matched as bytes"
        );
    }
    // Plain strings, an explicit UTF8_BINARY and UTF8_LCASE case mapping are unchanged.
    assert_eq!(one("SELECT CAST(contains('Ab', 'a') AS STRING)"), "false");
    assert_eq!(
        one("SELECT CAST(contains('Ab' COLLATE utf8_binary, 'a') AS STRING)"),
        "false"
    );
    assert_eq!(one("SELECT upper('ab' COLLATE utf8_lcase)"), "AB");
}

/// `İ` (U+0130) lower-cases to `i` + U+0307 and Spark (`lowercaseMatchLengthFrom`,
/// `lowercaseMatchUntil`) refuses a match that starts or ends inside the pair, so a pattern that
/// could do that is not run on the lower-cased key as a plain compare (unit 5a review B1); since
/// unit 5b such a pattern, and a per-row pattern, runs the lower-case search kernel and answers
/// false as Spark does.
#[test]
fn lcase_search_answers_a_pattern_that_could_split_the_dotted_capital_i() {
    for (call, want) in [
        ("contains('\u{130}' COLLATE utf8_lcase, 'i')", "false"),
        ("startswith('\u{130}' COLLATE utf8_lcase, 'i')", "false"),
        ("endswith('\u{130}' COLLATE utf8_lcase, '\u{307}')", "false"),
        ("contains('Ab' COLLATE utf8_lcase, concat('a', ''))", "true"),
    ] {
        assert_eq!(
            one(&format!("SELECT CAST({call} AS STRING)")),
            want,
            "{call}"
        );
    }
    assert_eq!(
        one("SELECT CAST(contains('Ab' COLLATE utf8_lcase, 'a') AS STRING)"),
        "true"
    );
    // Values of a real Spark 4.2.0 run (`ops/scripts/spark-local.sh`): a pattern that cannot split
    // the pair is answered on the key; a split of it is false in Spark too (refused here).
    assert_eq!(
        one("SELECT CAST(contains('a\u{130}b' COLLATE utf8_lcase, 'b') AS STRING)"),
        "true"
    );
    assert_eq!(
        one("SELECT CAST(contains('\u{130}' COLLATE utf8_lcase, 'i\u{307}') AS STRING)"),
        "true"
    );
}

/// A collated string inside an array, map or struct would lose its collation (or be compared
/// Indeterminate input gives no string result here; a non-string result is still produced.
#[test]
fn indeterminate_string_results_are_refused() {
    let s = Session::new().unwrap();
    for sql in [
        "create table t1(a string collate utf8_binary, b string collate utf8_lcase) using parquet",
        "insert into t1 values('abc', 'abcd')",
    ] {
        block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    }
    let e = block_on(s.sql("select concat(a, b) from t1")).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    assert_eq!(
        class_of(&s, "select length(concat(a, b)) from t1"),
        ErrorClass::KrishivUnsupportedFeature
    );
    // `levenshtein` is not on Spark's indeterminate list: it just runs ('abc' to 'abcd').
    assert_eq!(
        value_of(&s, "select cast(levenshtein(a, b) as string) from t1"),
        "1"
    );
}

/// A written `CAST(.. AS STRING COLLATE c)` is an explicit collation like `COLLATE`.
#[test]
fn an_explicit_cast_is_an_explicit_collation() {
    assert_eq!(
        class("SELECT concat(CAST('a' AS STRING COLLATE utf8_lcase), 'b' COLLATE unicode_ci)"),
        ErrorClass::CollationMismatchExplicit
    );
}

/// Defaults and switches the engine does not honour yet are refused instead of being ignored
/// (ADR 0043 Decisions 7 and 9): a silently ignored `DEFAULT COLLATION` would type strings
/// wrongly.
#[test]
fn unsupported_collation_defaults_and_switches_are_refused() {
    let s = Session::new().unwrap();
    for (sql, class) in [
        (
            "CREATE SCHEMA s1 DEFAULT COLLATION UTF8_LCASE",
            ErrorClass::UnsupportedFeatureSchemaLevelCollations,
        ),
        (
            "CREATE TABLE tt (a STRING) DEFAULT COLLATION UTF8_LCASE",
            ErrorClass::UnsupportedFeatureObjectLevelCollations,
        ),
        (
            "CREATE TABLE tm (m MAP<STRING COLLATE UTF8_LCASE, INT>)",
            ErrorClass::UnsupportedFeatureCollationsInMapKeys,
        ),
        (
            "CREATE VIEW vv DEFAULT COLLATION UTF8_LCASE AS SELECT 'a' AS c",
            ErrorClass::KrishivUnsupportedFeature,
        ),
        (
            "CREATE TEMPORARY VIEW tv DEFAULT COLLATION UNICODE AS SELECT 'a' AS c",
            ErrorClass::KrishivUnsupportedFeature,
        ),
        (
            "SET spark.sql.legacy.collationAwareHashFunctions = true",
            ErrorClass::KrishivUnsupportedFeature,
        ),
        (
            "SET spark.sql.collation.allowInMapKeys = true",
            ErrorClass::KrishivUnsupportedFeature,
        ),
    ] {
        let e = block_on(s.sql(sql)).unwrap_err();
        assert_eq!(e.error_class(), class, "{sql}");
    }
    // The defaults themselves, a binary view default, and the switches off are fine; the
    // session default collation does not change literals (Spark 4.2).
    for sql in [
        "SET spark.sql.collation.allowInMapKeys = false",
        "SET spark.sql.session.collation.default = UTF8_LCASE",
        "CREATE VIEW vb DEFAULT COLLATION UTF8_BINARY AS SELECT 'a' AS c",
    ] {
        block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    }
    assert_eq!(
        class_of(&s, "SET COLLATION UTF8_LCASE"),
        ErrorClass::InvalidSetSyntax
    );
}

fn class_of(s: &Session, sql: &str) -> ErrorClass {
    block_on(s.sql(sql)).unwrap_err().error_class()
}

/// Operands the strength pass does not read through (subqueries, variables, fields, elements)
/// count as implicit, never as absent; a collated scalar against a plain array is refused.
#[test]
fn unusual_operands_and_array_functions_do_not_compare_bytes() {
    for sql in [
        "SELECT (SELECT 'Ab' COLLATE utf8_lcase) LIKE 'a%'",
        "SELECT replace(CAST(array('aAb') AS ARRAY<STRING COLLATE unicode_ci>)[0], 'a', 'x')",
        "SELECT array_contains(array('a', 'b'), 'A' COLLATE utf8_lcase)",
        "SELECT array_position(array('a'), 'A' COLLATE unicode_ci)",
        "SELECT map_contains_key(map('a', 1), 'A' COLLATE utf8_lcase)",
    ] {
        assert_eq!(class(sql), ErrorClass::KrishivUnsupportedFeature, "{sql}");
    }
    // An element of a collated array is read by its collation (unit 5c).
    assert_eq!(
        one("SELECT replace(CAST(array('aAb') AS ARRAY<STRING COLLATE utf8_lcase>)[0], 'a', 'x')"),
        "xxb"
    );
    // A search over a subquery or a struct field of a collated type reads its key (unit 5a).
    for sql in [
        "SELECT CAST(contains((SELECT 'Ab' COLLATE utf8_lcase), 'a') AS STRING)",
        "SELECT CAST(contains(CAST(named_struct('f','Ab') AS STRUCT<f: STRING COLLATE utf8_lcase>).f, 'a') AS STRING)",
    ] {
        assert_eq!(one(sql), "true", "{sql}");
    }
    // Plain arrays with plain needles are untouched.
    assert_eq!(
        one("SELECT CAST(array_contains(array('a', 'b'), 'a') AS STRING)"),
        "true"
    );
}

/// A value window function over a collated column is refused with the engine's class (the
/// re-tag cast would otherwise hide the window from the binder).
#[test]
fn value_window_functions_over_collated_columns_are_refused() {
    let s = Session::new().unwrap();
    for sql in [
        "create table w1(a string collate utf8_lcase, c int) using parquet",
        "insert into w1 values('x', 1), ('y', 2)",
    ] {
        block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    }
    assert_eq!(
        class_of(&s, "select lag(a) over (order by c) from w1"),
        ErrorClass::KrishivUnsupportedFeature
    );
    // A window over a plain column is as before.
    assert!(block_on(s.sql("select lag(c) over (order by c) from w1")).is_ok());
}

/// A collated map key is refused in `ALTER TABLE ... ADD COLUMNS` as in `CREATE TABLE`.
#[test]
fn alter_table_refuses_collated_map_keys() {
    let s = Session::new().unwrap();
    block_on(
        block_on(s.sql("create table am(a int) using parquet"))
            .unwrap()
            .collect(),
    )
    .unwrap();
    assert_eq!(
        class_of(
            &s,
            "ALTER TABLE am ADD COLUMNS (m MAP<STRING COLLATE UTF8_LCASE, INT>)"
        ),
        ErrorClass::UnsupportedFeatureCollationsInMapKeys
    );
}

/// Codex review of #493: a lookup in a map with collated keys compares bytes, a function that
/// reads a collation does not pass it on, and the builder checks the switches like `SET`.
#[test]
fn map_key_lookups_boundaries_and_builder_switches() {
    let nested = "STRUCT<k: STRING COLLATE UTF8_LCASE>";
    let sql = format!(
        "SELECT CAST(map(named_struct('k', 'a'), 1) AS MAP<{nested}, INT>)\
         [CAST(named_struct('k', 'A') AS {nested})]"
    );
    assert_eq!(class(&sql), ErrorClass::KrishivUnsupportedFeature);
    assert_eq!(
        class("SELECT map('a', 1)['A' COLLATE utf8_lcase]"),
        ErrorClass::KrishivUnsupportedFeature
    );
    // `collation()` returns a plain string: it brings no explicit collation of its own.
    assert_eq!(
        one("SELECT typeof(concat(collation('a' COLLATE UTF8_LCASE), 'b' COLLATE UNICODE_CI))"),
        "string collate UNICODE_CI"
    );
    let built = Session::builder()
        .config("spark.sql.legacy.collationAwareHashFunctions", "true")
        .build();
    assert_eq!(
        built.err().map(|e| e.error_class()),
        Some(ErrorClass::KrishivUnsupportedFeature)
    );
}

/// Codex review of #493 (second round): binary-result functions keep a plain `STRING`, a written
/// cast to `STRING` ends the inherited collation, and `equal_null` is guarded like `<=>`.
#[test]
fn result_boundaries_and_equal_null() {
    assert_eq!(
        one("SELECT typeof(soundex('Robert' COLLATE UTF8_LCASE))"),
        "string"
    );
    assert_eq!(
        one("SELECT typeof(format_number(123, '###' COLLATE UTF8_LCASE))"),
        "string"
    );
    // Codex review of #493 (third round): a binary-result call nested in another string call does
    // not pass its argument's collation on, so it cannot conflict with another explicit one.
    assert_eq!(
        one(
            "SELECT collation(concat(soundex('Robert' COLLATE UTF8_LCASE), 'b' COLLATE UNICODE_CI))"
        ),
        "SYSTEM.BUILTIN.UNICODE_CI"
    );
    // A cast back to plain STRING: the other operand decides, with no false mismatch.
    assert_eq!(
        one(
            "SELECT typeof(concat(CAST('a' COLLATE UTF8_LCASE AS STRING), 'b' COLLATE UNICODE_CI))"
        ),
        "string collate UNICODE_CI"
    );
    assert_eq!(
        format!(
            "{:?}",
            run("SELECT equal_null('a' COLLATE UTF8_LCASE, 'A' COLLATE UTF8_LCASE)").unwrap()
        ),
        "[\"BooleanArray\\n[\\n  true,\\n]\"]"
    );
}

/// `nvl2`'s first argument only selects: it does not decide the result's collation (Codex review).
#[test]
fn nvl2_selector_does_not_unify() {
    assert_eq!(
        one("SELECT typeof(nvl2('x' COLLATE UTF8_LCASE, 'a', 'b'))"),
        "string"
    );
    assert_eq!(
        one("SELECT typeof(nvl2('x', 'a' COLLATE UNICODE_CI, 'b' COLLATE UNICODE_CI))"),
        "string collate UNICODE_CI"
    );
    // The selector's collation does not leak into an enclosing call either.
    assert_eq!(
        one(
            "SELECT typeof(concat(nvl2('x' COLLATE UTF8_LCASE, 'a', 'b'), 'c' COLLATE UNICODE_CI))"
        ),
        "string collate UNICODE_CI"
    );
}

/// Unit 3a: comparisons of `UTF8_LCASE` / `_RTRIM` strings compare collation keys, with the
/// answers of the Spark 4.2 runs in design `P2-collations.md` section 2.4.
#[test]
fn lcase_and_rtrim_comparisons_compare_keys() {
    let t = |sql: &str| one(&format!("SELECT CAST(({sql}) AS STRING)"));
    for (sql, want) in [
        ("'a' COLLATE utf8_lcase = 'A'", "true"),
        ("'A' = 'a' COLLATE utf8_lcase", "true"),
        ("'a' COLLATE utf8_lcase = 'b'", "false"),
        ("'a' COLLATE utf8_lcase < 'B'", "true"),
        ("'b' COLLATE utf8_lcase < 'A'", "false"),
        ("'B' COLLATE utf8_lcase >= 'a'", "true"),
        ("'a' COLLATE utf8_lcase <=> 'A'", "true"),
        ("CAST(NULL AS STRING) COLLATE utf8_lcase <=> 'A'", "false"),
        // The special cases of `lowerCaseCodePoints`.
        ("'\u{3c2}' COLLATE utf8_lcase = '\u{3c3}'", "true"),
        ("'\u{130}' COLLATE utf8_lcase = 'i\u{307}'", "true"),
        ("'\u{df}' COLLATE utf8_lcase = 'ss'", "false"),
        // RTRIM drops trailing spaces only.
        ("'a ' COLLATE utf8_binary_rtrim = 'a'", "true"),
        ("' a' COLLATE utf8_binary_rtrim = 'a'", "false"),
        ("'A ' COLLATE utf8_lcase_rtrim = 'a'", "true"),
        ("'a' COLLATE utf8_lcase IN ('B', 'A')", "true"),
        ("'a' COLLATE utf8_lcase IN ('B', 'C')", "false"),
        ("'b' COLLATE utf8_lcase BETWEEN 'A' AND 'C'", "true"),
        ("'d' COLLATE utf8_lcase BETWEEN 'A' AND 'C'", "false"),
        ("'b' COLLATE utf8_lcase NOT BETWEEN 'A' AND 'C'", "false"),
        ("'d' COLLATE utf8_lcase NOT BETWEEN 'A' AND 'C'", "true"),
        ("'a' COLLATE utf8_lcase NOT IN ('B', 'A')", "false"),
        ("'a' COLLATE utf8_lcase NOT IN ('B', 'C')", "true"),
        (
            "CAST(NULL AS STRING) COLLATE utf8_lcase <=> CAST(NULL AS STRING)",
            "true",
        ),
        ("CAST(NULL AS STRING) COLLATE utf8_lcase = 'A'", "NULL"),
        (
            "CASE WHEN 'a' COLLATE utf8_lcase = 'A' THEN 'y' ELSE 'n' END",
            "y",
        ),
        // Only trailing spaces are dropped, tabs and the no-break space are not.
        ("'a\t' COLLATE utf8_binary_rtrim = 'a'", "false"),
        ("'a\u{a0}' COLLATE utf8_binary_rtrim = 'a'", "false"),
    ] {
        assert_eq!(t(sql), want, "{sql}");
    }
    // The column name is the call as written.
    let s = Session::new().unwrap();
    let df = block_on(s.sql("SELECT 'a' COLLATE utf8_lcase = 'A'")).unwrap();
    assert_eq!(
        block_on(df.schema()).unwrap().field(0).name(),
        "(collate(a, utf8_lcase) = A)"
    );
}

/// Filters, joins and projections over collated columns use the keys; ICU collations, which have
/// no key yet, stay refused.
#[test]
fn tables_filter_and_join_on_collation_keys() {
    let s = Session::new().unwrap();
    for sql in [
        "create table c1(a string collate utf8_lcase, n int) using parquet",
        "insert into c1 values('aaa', 1), ('AAA', 2), ('bbb', 3), ('BBB', 4)",
        "create table c2(a string collate utf8_lcase) using parquet",
        "insert into c2 values('aaa'), ('ccc')",
    ] {
        block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    }
    let count = |sql: &str| value_of(&s, &format!("select cast(count(*) as string) from {sql}"));
    assert_eq!(count("c1 where a = 'aaa' collate utf8_lcase"), "2");
    assert_eq!(count("c1 where a = 'AAA'"), "2");
    assert_eq!(count("c1 where a < 'bbb'"), "2");
    assert_eq!(count("c1 join c2 on c1.a = c2.a"), "2");
    assert_eq!(count("c1 where a in ('BBB', 'zzz')"), "2");
    // An explicit `UTF8_BINARY` beats the column's implicit `UTF8_LCASE`: bytes.
    assert_eq!(count("c1 where a = 'AAA' collate utf8_binary"), "1");
    // `BETWEEN` coerces each half on its own (`collation_keys.rs` runs the mixed form).
    assert_eq!(count("c1 where a between 'AAA' and 'bbb'"), "4");
    // Two different explicit collations never compare.
    assert_eq!(
        class_of(
            &s,
            "select 1 where 'a' collate utf8_lcase = 'a' collate utf8_binary"
        ),
        ErrorClass::CollationMismatchExplicit
    );
    // Opus review of #530: against a timestamp, number or binary the other operand is cast to the
    // string's type, so the lowered key must never be cast instead of the text: those comparisons
    // stay with the guard.
    for sql in [
        "select count(*) from (select timestamp '2020-01-01 10:00:00' as ts) where ts = '2020-01-01T10:00:00' collate utf8_lcase",
        "select count(*) from c1 where n = '1' collate utf8_lcase",
    ] {
        assert_eq!(
            class_of(&s, sql),
            ErrorClass::KrishivUnsupportedFeature,
            "{sql}"
        );
    }
    assert_eq!(
        class_of(
            &s,
            "select count(*) from c1 where a = 'aaa' collate unicode_ci"
        ),
        ErrorClass::KrishivUnsupportedFeature
    );
}

/// The first value of the first column of `sql`, as text.
fn value_of(s: &Session, sql: &str) -> String {
    let b = block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    let c = &b[0].columns()[0];
    c.as_string_opt::<i32>().map_or_else(
        || c.as_string_view().value(0).to_owned(),
        |v| v.value(0).to_owned(),
    )
}

/// A written `UTF8_BINARY` keeps its name in the schema (ADR 0043, update 2026-11-07) and is a
/// plain string everywhere else: functions, comparisons, grouping and joins work as on `STRING`.
#[test]
fn an_explicit_utf8_binary_is_a_display_only_marker() {
    let s = Session::new().unwrap();
    for sql in [
        "create table m(c string collate utf8_binary, p string) using parquet",
        "insert into m values ('aB', 'aB'), ('ab', 'x'), ('aB', 'y')",
    ] {
        block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    }
    let col = |sql: &str| -> Vec<String> {
        let b = block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
        let mut out = Vec::new();
        for batch in &b {
            let c = &batch.columns()[0];
            for i in 0..c.len() {
                out.push(if let Some(v) = c.as_string_view_opt() {
                    v.value(i).to_owned()
                } else if let Some(v) = c.as_string_opt::<i32>() {
                    v.value(i).to_owned()
                } else if let Some(v) = c.as_primitive_opt::<arrow_array::types::Int64Type>() {
                    v.value(i).to_string()
                } else {
                    format!("{c:?}")
                });
            }
        }
        out
    };
    // The schema line: the marker follows the first operand of the winning strength.
    let ty = |sql: &str| col(sql).remove(0);
    assert_eq!(
        ty("select typeof(upper(c)) from m limit 1"),
        "string collate UTF8_BINARY"
    );
    assert_eq!(ty("select typeof(upper(p)) from m limit 1"), "string");
    // With one collation among the operands nothing is cast and the head operand decides.
    assert_eq!(ty("select typeof(concat(p, c)) from m limit 1"), "string");
    assert_eq!(
        ty("select typeof(concat(c, p)) from m limit 1"),
        "string collate UTF8_BINARY"
    );
    assert_eq!(ty("select typeof(concat('a', c)) from m limit 1"), "string");
    // An operand already `UTF8_BINARY` is not cast: the head operand's own type stays.
    assert_eq!(
        ty("select typeof(concat(p, c, 'y' collate utf8_binary)) from m limit 1"),
        "string"
    );
    // Mixed collations cast to the winner, which is the written type.
    assert_eq!(
        ty("select typeof(concat(c, 'x' collate utf8_lcase)) from m limit 1"),
        "string collate UTF8_LCASE"
    );
    // Behaviour is a plain string's.
    assert_eq!(col("select upper(c) from m order by 1")[0], "AB");
    assert_eq!(col("select count(*) from m where c = 'aB'"), ["2"]);
    assert_eq!(col("select count(*) from m where c = p"), ["1"]);
    assert_eq!(col("select count(distinct c) from m"), ["2"]);
    assert_eq!(col("select c from m group by c order by c"), ["aB", "ab"]);
    assert_eq!(col("select count(*) from m a join m b on a.c = b.c"), ["5"]);
}

/// Type unification treats the marker as the plain string it equals (`UNION`, `IN (subquery)`,
/// array functions, casts to the marker type, a `from_json` map key).
#[test]
fn the_marker_unifies_with_string() {
    let s = Session::new().unwrap();
    for sql in [
        "create table u(c string collate utf8_binary, p string, a array<string collate utf8_binary>) using parquet",
        "insert into u values ('x', 'x', array('x', 'y'))",
    ] {
        block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
    }
    let n = |sql: &str| -> usize {
        block_on(block_on(s.sql(sql)).unwrap().collect())
            .unwrap()
            .iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum()
    };
    assert_eq!(n("select c from u union select p from u"), 1);
    assert_eq!(n("select p from u where p in (select c from u)"), 1);
    assert_eq!(n("select c from u where c in (select p from u)"), 1);
    assert_eq!(n("select array_contains(a, 'x') from u"), 1);
    assert_eq!(n("select coalesce(a, array(1)) from u where false"), 0);
    assert_eq!(n("select c from u where false union all select 1"), 1);
    assert_eq!(n("select 1 in (select c from u where false) from u"), 1);
    assert_eq!(
        n("select cast(date'2024-01-01' as string collate utf8_binary)"),
        1
    );
    assert_eq!(
        n("select from_json('{\"a\":1}', 'map<string collate utf8_binary,int>')"),
        1
    );
}

/// `contains`, `startswith` and `endswith` under `UTF8_LCASE` compare the lower-cased text
/// (Spark's `lowercaseContains` and friends); other collations keep the guard (unit 5a).
#[test]
fn searches_under_utf8_lcase_ignore_case() {
    assert_eq!(
        one("SELECT CAST(contains('Hello World' COLLATE UTF8_LCASE, 'WORLD') AS STRING)"),
        "true"
    );
    assert_eq!(
        one("SELECT CAST(startswith('Hello' COLLATE UTF8_LCASE, 'HE') AS STRING)"),
        "true"
    );
    assert_eq!(
        one("SELECT CAST(endswith('Hello' COLLATE UTF8_LCASE, 'LLO') AS STRING)"),
        "true"
    );
    assert_eq!(
        one("SELECT CAST(contains('Hello' COLLATE UTF8_LCASE, 'xyz') AS STRING)"),
        "false"
    );
    // The binary collation stays case-sensitive.
    assert_eq!(
        one("SELECT CAST(contains('Hello', 'HELLO') AS STRING)"),
        "false"
    );
    // NULL stays NULL.
    assert_eq!(
        one("SELECT CAST(contains(CAST(NULL AS STRING) COLLATE UTF8_LCASE, 'a') AS STRING)"),
        "NULL"
    );
    // The rtrim and ICU variants are still refused until their units land.
    for sql in [
        "SELECT contains('a ' COLLATE UTF8_LCASE_RTRIM, 'A')",
        "SELECT startswith('a' COLLATE UNICODE_CI, 'A')",
    ] {
        assert_eq!(class(sql), ErrorClass::KrishivUnsupportedFeature, "{sql}");
    }
}

/// A string literal under the implicit collation cast is quoted in the column name, as Spark's
/// `Literal.sql` prints it.
#[test]
fn the_implicit_collation_cast_quotes_a_literal_in_the_name() {
    let s = Session::new().unwrap();
    let df = block_on(s.sql("SELECT contains('Ab' COLLATE UTF8_LCASE, 'a')")).unwrap();
    let names: Vec<String> = block_on(df.schema())
        .unwrap()
        .fields()
        .iter()
        .map(|f| f.name().to_owned())
        .collect();
    assert_eq!(
        names,
        ["contains(collate(Ab, UTF8_LCASE), 'a' collate UTF8_LCASE)"]
    );
}

/// `x IN (subquery)` compares in the collated side's collation; values are those of a real
/// Spark 4.2.0 run (`ops/scripts/spark-local.sh`, 2026-11-07).
#[test]
fn in_subquery_compares_through_the_collation_keys() {
    for (sql, want) in [
        ("('a' COLLATE utf8_lcase) IN (SELECT 'A')", "true"),
        ("('b' COLLATE utf8_lcase) IN (SELECT 'A')", "false"),
        ("'a' IN (SELECT 'A' COLLATE utf8_lcase)", "true"),
        ("'a ' COLLATE utf8_binary_rtrim IN (SELECT 'a')", "true"),
        ("'b' IN (SELECT 'A' COLLATE utf8_lcase)", "false"),
        ("('a' COLLATE utf8_binary) IN (SELECT 'a')", "true"),
        ("('a' COLLATE utf8_binary) IN (SELECT 'A')", "false"),
    ] {
        assert_eq!(
            one(&format!("SELECT CAST(({sql}) AS STRING)")),
            want,
            "{sql}"
        );
    }
}

/// `instr` / `locate` over `UTF8_LCASE` strings: values of a Spark 4.2.0 local run (2026-11-08).
#[test]
fn positions_over_lcase_strings_match_spark() {
    let l = "COLLATE utf8_lcase";
    for (sql, want) in [
        (format!("instr('xAbAy' {l}, 'ab')"), "2"),
        (format!("locate('ab', 'xAbAy' {l})"), "2"),
        (format!("locate('a', 'xAbAy' {l}, 3)"), "4"),
        (format!("instr('a\u{130}' {l}, 'i')"), "0"),
        (format!("instr('a\u{130}' {l}, 'i\u{307}')"), "2"),
        (format!("instr('a\u{130}b' {l}, 'b')"), "3"),
        (format!("instr('x\u{3c2}' {l}, '\u{3c3}')"), "2"),
        (format!("locate('', 'xAbAy' {l}, 3)"), "1"),
        (format!("find_in_set('B' {l}, 'a,b,c')"), "2"),
        (format!("find_in_set('b', 'A,B,C' {l})"), "2"),
        (format!("find_in_set('a,b' {l}, 'a,b')"), "0"),
        (
            format!("find_in_set('\u{3a3}' {l}, '\u{3c2},\u{3c3}')"),
            "1",
        ),
    ] {
        assert_eq!(one(&format!("SELECT CAST({sql} AS STRING)")), want, "{sql}");
    }
}

/// A column (or `İ`-splitting) pattern over `UTF8_LCASE`: values of a Spark 4.2.0 local run.
#[test]
fn searches_with_column_patterns_over_lcase_match_spark() {
    let l = "COLLATE utf8_lcase";
    for (sql, want) in [
        (format!("contains(t {l}, p)"), "true"),
        (format!("startswith(t {l}, p)"), "false"),
        (format!("endswith(t {l}, p)"), "true"),
    ] {
        let q = format!("SELECT CAST({sql} AS STRING) FROM (SELECT 'AbC' AS t, 'bC' AS p)");
        assert_eq!(one(&q), want, "{sql}");
    }
}

/// The `İ` rows of the kernel test, through SQL: `İ` lower-cases to `i` + U+0307, so a match may
/// not end inside the pair (values of a Spark 4.2.0 local run).
#[test]
fn searches_over_lcase_split_the_dotted_capital_i_like_spark() {
    let l = "COLLATE utf8_lcase";
    for (pat, want) in [
        ("a", ["true", "true", "false"]),
        ("ai", ["false", "false", "false"]),
        ("ai\u{307}", ["true", "true", "true"]),
    ] {
        for (f, w) in ["contains", "startswith", "endswith"].iter().zip(want) {
            let q = format!(
                "SELECT CAST({f}(t {l}, p) AS STRING) FROM (SELECT 'a\u{130}' AS t, '{pat}' AS p)"
            );
            assert_eq!(one(&q), w, "{f} {pat:?}");
        }
    }
}

#[test]
fn equal_null_compares_lcase_keys() {
    for (call, want) in [
        ("equal_null('A' COLLATE utf8_lcase, 'a')", "true"),
        ("equal_null('A' COLLATE utf8_lcase, 'b')", "false"),
        ("equal_null('A' COLLATE utf8_lcase, NULL)", "false"),
    ] {
        assert_eq!(
            one(&format!("SELECT CAST({call} AS STRING)")),
            want,
            "{call}"
        );
    }
}

/// `trim`, `replace`, `translate`, `split_part` and `substring_index` over `UTF8_LCASE` and the
/// `_RTRIM` collations (unit 5c); values of a Spark 4.2.0 local run (2026-11-10).
#[test]
fn trim_replace_translate_split_substring_index_over_lcase_and_rtrim() {
    for (sql, want) in [
        ("TRIM('b' FROM 'ABab' COLLATE utf8_lcase)", "ABa"),
        (
            "BTRIM('\u{130}o' COLLATE utf8_lcase, 'i\u{307}o')",
            "\u{130}",
        ),
        ("BTRIM('i\u{307}o' COLLATE utf8_lcase, '\u{130}o')", ""),
        ("LTRIM('a', 'AAb' COLLATE utf8_lcase)", "b"),
        ("RTRIM('B', 'aAbB' COLLATE utf8_lcase)", "aA"),
        ("BTRIM('ab  ' COLLATE utf8_binary_rtrim, 'b')", "a  "),
        ("BTRIM('ab  ' COLLATE utf8_binary_rtrim, 'b ')", "a"),
        ("RTRIM('ab ', 'xab  ' COLLATE utf8_lcase_rtrim)", "x"),
        ("RTRIM('  ab  ' COLLATE utf8_binary_rtrim)", "  ab"),
        ("REPLACE('aAa' COLLATE utf8_lcase, 'A', 'x')", "xxx"),
        (
            "REPLACE('\u{130}o' COLLATE utf8_lcase, 'i', 'x')",
            "\u{130}o",
        ),
        ("REPLACE('A  ' COLLATE utf8_lcase_rtrim, 'a ', 'x')", "x "),
        ("REPLACE('a  ' COLLATE utf8_binary_rtrim, 'a ', 'x')", "x "),
        ("TRANSLATE('aAbB' COLLATE utf8_lcase, 'Aab', 'xy')", "xx"),
        (
            "TRANSLATE('i\u{307}i' COLLATE utf8_lcase, '\u{130}', '1')",
            "1i",
        ),
        (
            "TRANSLATE('A b ' COLLATE utf8_lcase_rtrim, ' ', '1')",
            "A1b1",
        ),
        ("SPLIT_PART('aAbAAc' COLLATE utf8_lcase, 'a', 3)", "b"),
        ("SPLIT_PART('aAbAAc' COLLATE utf8_lcase, 'a', -1)", "c"),
        (
            "SPLIT_PART('a  b' COLLATE utf8_binary_rtrim, 'a ', 2)",
            "  b",
        ),
        ("SPLIT_PART('Ab  ' COLLATE utf8_lcase_rtrim, 'B ', 1)", "A"),
        (
            "SUBSTRING_INDEX('aAbAAc' COLLATE utf8_lcase, 'a', 3)",
            "aAb",
        ),
        (
            "SUBSTRING_INDEX('aAbAAc' COLLATE utf8_lcase, 'a', -2)",
            "Ac",
        ),
        (
            "SUBSTRING_INDEX('a  b' COLLATE utf8_binary_rtrim, ' ', 1)",
            "",
        ),
        (
            "SUBSTRING_INDEX('Ab  ' COLLATE utf8_lcase_rtrim, 'B ', 1)",
            "A",
        ),
    ] {
        assert_eq!(one(&format!("SELECT {sql}")), want, "{sql}");
    }
    // The result keeps the collation.
    assert_eq!(
        one("SELECT typeof(replace('aAa' COLLATE utf8_lcase, 'A', 'x'))"),
        "string collate UTF8_LCASE"
    );
    assert_eq!(
        one("SELECT typeof(btrim('a ' COLLATE utf8_binary_rtrim, 'a'))"),
        "string collate UTF8_BINARY_RTRIM"
    );
    // NULL stays NULL.
    assert_eq!(
        one("SELECT CAST(btrim(CAST(NULL AS STRING) COLLATE utf8_lcase, 'a') AS STRING)"),
        "NULL"
    );
}

/// `nullif`, `least` and `greatest` over strings of one `UTF8_LCASE` or `_RTRIM` type compare by
/// the collation: the earlier operand wins a tie and NULLs are skipped (Spark 4.2.0 values).
#[test]
fn nullif_least_and_greatest_pick_by_the_collation() {
    let l = "collate utf8_lcase";
    for (sql, want) in [
        (format!("select nullif('aaa' {l}, 'AAA' {l})"), "NULL"),
        (format!("select nullif('aaa' {l}, 'bbb' {l})"), "aaa"),
        (
            format!("select nullif('aaa' {l}, cast(null as string {l}))"),
            "aaa",
        ),
        (
            format!("select nullif(cast(null as string {l}), 'aaa' {l})"),
            "NULL",
        ),
        (format!("select least('b' {l}, 'B' {l})"), "b"),
        (format!("select least('B' {l}, 'b' {l})"), "B"),
        (format!("select greatest('a' {l}, 'A' {l})"), "a"),
        (format!("select greatest('A' {l}, 'a' {l})"), "A"),
        (format!("select least('aaa' {l}, 'AAA' {l}, 'a' {l})"), "a"),
        (
            format!("select greatest('aaa' {l}, 'AAA' {l}, 'B' {l})"),
            "B",
        ),
        (
            format!("select least(cast(null as string {l}), 'b' {l}, 'A' {l})"),
            "A",
        ),
        (
            format!("select greatest(cast(null as string {l}), cast(null as string {l}))"),
            "NULL",
        ),
        (
            format!("select typeof(least('a' {l}, 'b' {l}))"),
            "string collate UTF8_LCASE",
        ),
        (
            "select nullif('a ' collate utf8_binary_rtrim, 'a' collate utf8_binary_rtrim)".into(),
            "NULL",
        ),
        (
            "select least('b ' collate utf8_binary_rtrim, 'b' collate utf8_binary_rtrim)".into(),
            "b ",
        ),
        (
            "select greatest('b ' collate utf8_binary_rtrim, 'B' collate utf8_binary_rtrim)".into(),
            "b ",
        ),
    ] {
        assert_eq!(one(&sql), want, "{sql}");
    }
    // Any number of operands pick by key (`__collation_least`).
    assert_eq!(
        one(&format!(
            "select least('b' {l}, 'a' {l}, 'C' {l}, 'd' {l}, 'E' {l})"
        )),
        "a"
    );
    // The call keeps its own name and result type.
    let s = Session::new().unwrap();
    let df = block_on(s.sql(&format!("select nullif('aaa' {l}, 'AAA' {l})"))).unwrap();
    let name = block_on(df.schema()).unwrap().field(0).name().clone();
    assert_eq!(
        name,
        "nullif(collate(aaa, utf8_lcase), collate(AAA, utf8_lcase))"
    );
}

#[test]
fn listagg_distinct_casts_follow_spark() {
    // A cast the user wrote over a cast is not the implicit one: Spark names the value, not the
    // cast type (MISMATCH_WITH_DISTINCT_INPUT),.
    let e = run(
        "SELECT listagg(DISTINCT CAST(CAST(col AS STRING COLLATE UTF8_LCASE) AS STRING)) \
         WITHIN GROUP (ORDER BY col) FROM VALUES ('ABC'), ('abc') AS t(col)",
    )
    .unwrap_err();
    assert!(
        e.to_string().contains("MISMATCH_WITH_DISTINCT_INPUT")
            && !e.to_string().contains("UNSAFE_CAST"),
        "{e}"
    );
    let e = run(
        "SELECT listagg(DISTINCT CAST(col AS STRING COLLATE UTF8_LCASE)) \
         WITHIN GROUP (ORDER BY col) FROM VALUES ('ABC'), ('abc') AS t(col)",
    )
    .unwrap_err();
    assert!(e.to_string().contains("UNSAFE_CAST"), "{e}");
}
