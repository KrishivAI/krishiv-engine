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
    // ICU collations have no key yet (unit 6): their scalar comparisons and joins are refused.
    let ci = "('a' COLLATE unicode_ci)";
    for sql in [
        // scalar comparisons
        format!("SELECT {ci} = 'A'"),
        format!("SELECT {ci} <=> 'A'"),
        format!("SELECT {ci} < 'A'"),
        format!("SELECT {ci} BETWEEN 'a' AND 'b'"),
        format!("SELECT {ci} IN ('A', 'b')"),
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
/// bytes (unit 5 brings the collation-aware kernels); binary strings are untouched.
#[test]
fn collation_dependent_calls_refuse_instead_of_matching_bytes() {
    let l = "('Ab' COLLATE utf8_lcase)";
    for sql in [
        format!("SELECT contains({l}, 'a')"),
        format!("SELECT startswith({l}, 'a')"),
        format!("SELECT instr({l}, 'a')"),
        format!("SELECT replace({l}, 'a', 'x')"),
        format!("SELECT translate({l}, 'a', 'x')"),
        format!("SELECT trim('a' FROM {l})"),
        format!("SELECT split_part({l}, 'x', 1)"),
        format!("SELECT {l} LIKE 'a%'"),
        format!("SELECT {l} RLIKE '^a'"),
        format!("SELECT regexp_replace({l}, 'a', 'x')"),
        "SELECT replace('aAb' COLLATE unicode_ci, 'a', 'x')".to_owned(),
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

/// A collated string inside an array, map or struct would lose its collation (or be compared
/// as bytes), so building or reading one is refused for now; counting and picking are fine.
#[test]
fn nested_values_of_collated_strings_are_refused() {
    for sql in [
        "SELECT array('a' COLLATE utf8_lcase)",
        "SELECT map('a' COLLATE utf8_lcase, 1)",
        "SELECT named_struct('f', 'a' COLLATE utf8_lcase)",
        "SELECT array('a' COLLATE utf8_lcase) = array('A')",
    ] {
        assert_eq!(class(sql), ErrorClass::KrishivUnsupportedFeature, "{sql}");
    }
}

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
        (
            "SET spark.sql.collation.schemaLevel.enabled = true",
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
        "SELECT contains((SELECT 'Ab' COLLATE utf8_lcase), 'a')",
        "SELECT (SELECT 'Ab' COLLATE utf8_lcase) LIKE 'a%'",
        "SELECT replace(CAST(array('aAb') AS ARRAY<STRING COLLATE utf8_lcase>)[0], 'a', 'x')",
        "SELECT contains(CAST(named_struct('f','Ab') AS STRUCT<f: STRING COLLATE utf8_lcase>).f, 'a')",
        "SELECT array_contains(array('a', 'b'), 'A' COLLATE utf8_lcase)",
        "SELECT array_position(array('a'), 'A' COLLATE unicode_ci)",
        "SELECT map_contains_key(map('a', 1), 'A' COLLATE utf8_lcase)",
    ] {
        assert_eq!(class(sql), ErrorClass::KrishivUnsupportedFeature, "{sql}");
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
    // Opus review of #493 (head f1239bef): the three-argument `decode` is a CASE whose result keeps
    // the collation of its branches; only the charset form `decode(bin, 'utf-8')` is a plain STRING.
    assert_eq!(
        one("SELECT typeof(decode(1, 1, 'a' COLLATE UTF8_LCASE, 'b' COLLATE UTF8_LCASE))"),
        "string collate UTF8_LCASE"
    );
    assert_eq!(
        one("SELECT typeof(decode(encode('a' COLLATE UTF8_LCASE, 'utf-8'), 'utf-8'))"),
        "string"
    );
    // A call returning a non-string carries no collation: `length` of a collated string does not
    // make `substr`'s result collated.
    assert_eq!(
        one("SELECT typeof(upper(substr('abc', 1, length('x' COLLATE UTF8_LCASE))))"),
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
        class("SELECT equal_null('a' COLLATE UTF8_LCASE, 'A' COLLATE UTF8_LCASE)"),
        ErrorClass::KrishivUnsupportedFeature
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
/// no key yet, stay refused, as do `greatest` and the keyed positions of unit 3b.
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
    let count = |sql: &str| -> String {
        let b = block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap();
        let c = &b[0].columns()[0];
        format!("{c:?}")
    };
    assert!(count("select count(*) from c1 where a = 'aaa' collate utf8_lcase").contains('2'));
    assert!(count("select count(*) from c1 where a = 'AAA'").contains('2'));
    assert!(count("select count(*) from c1 where a < 'bbb'").contains('2'));
    assert!(count("select count(*) from c1 join c2 on c1.a = c2.a").contains('2'));
    assert!(count("select count(*) from c1 where a in ('BBB', 'zzz')").contains('2'));
    for sql in [
        "select count(*) from c1 where a = 'aaa' collate unicode_ci",
        "select count(*) from c1 group by a",
    ] {
        assert_eq!(
            class_of(&s, sql),
            ErrorClass::KrishivUnsupportedFeature,
            "{sql}"
        );
    }
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
