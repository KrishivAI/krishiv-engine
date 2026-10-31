// SPDX-License-Identifier: Apache-2.0
//! Oracle-style `decode(expr, search, result, ..., [default])` (a CASE over
//! null-safe equality) and `format_string`/`printf`. Expected values come
//! from the Spark 4.2.0 oracle.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

fn session() -> Session {
    Session::new().unwrap()
}

/// The rows of a `SELECT`, rendered as Spark renders them (columns
/// joined by tabs).
fn rows_of(batches: &[RecordBatch]) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        for i in 0..b.num_rows() {
            let mut row = Vec::new();
            for col in b.columns() {
                if col.is_null(i) {
                    row.push("NULL".to_owned());
                    continue;
                }
                row.push(match col.data_type() {
                    ArrowType::Utf8View | ArrowType::Utf8 | ArrowType::LargeUtf8 => {
                        match col.data_type() {
                            ArrowType::Utf8View => col.as_string_view().value(i).to_owned(),
                            ArrowType::LargeUtf8 => col.as_string::<i64>().value(i).to_owned(),
                            _ => col.as_string::<i32>().value(i).to_owned(),
                        }
                    }
                    ArrowType::Int64 => col
                        .as_primitive::<arrow_array::types::Int64Type>()
                        .value(i)
                        .to_string(),
                    ArrowType::Int32 => col
                        .as_primitive::<arrow_array::types::Int32Type>()
                        .value(i)
                        .to_string(),
                    t => panic!("unrendered {t}"),
                });
            }
            out.push(row.join("\t"));
        }
    }
    out
}

/// Runs `sql`, expecting rows.
fn run(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    rows_of(&block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}")))
}

/// The output names of `sql`.
fn names(s: &Session, sql: &str) -> Vec<String> {
    let df = block_on(s.sql(sql)).unwrap();
    block_on(df.schema())
        .unwrap()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

fn one(s: &Session, sql: &str) -> String {
    run(s, sql).remove(0)
}

#[test]
fn decode_picks_the_first_match_or_the_default() {
    let s = session();
    let case = |q: &str| one(&s, q);
    assert_eq!(case("select decode(1, 1, 'Southlake')"), "Southlake");
    assert_eq!(case("select decode(2, 1, 'Southlake')"), "NULL");
    assert_eq!(
        case("select decode(6, 1, 'a', 2, 'b', 3, 'c', 4, 'd', 'Non domestic')"),
        "Non domestic"
    );
    assert_eq!(
        case("select decode(6, 1, 'a', 2, 'b', 3, 'c', 4, 'd')"),
        "NULL"
    );
    assert_eq!(case("select decode(2, 1, 'a', 2, 'b', 3, 'c')"), "b");
    // A NULL expression matches a NULL search; a NULL search never matches a value.
    assert_eq!(
        case("select decode(null, 6, 'Spark', NULL, 'SQL', 4, 'rocks')"),
        "SQL"
    );
    assert_eq!(
        case("select decode(null, 6, 'Spark', NULL, 'SQL', 4, 'rocks', NULL, '.')"),
        "SQL"
    );
    assert_eq!(case("select decode(1, null, 'n', 1, 'one')"), "one");
    assert_eq!(
        case("select decode(x, 1, 'a', 'b') from values (cast(null as int)) t(x)"),
        "b"
    );
    // The search is compared at the common type.
    assert_eq!(case("select decode(1, '1', 'a')"), "a");
    assert_eq!(case("select decode(1, 1.0, 'a', 'b')"), "a");
    assert_eq!(case("select typeof(decode(1, 1, 2, 3, 4L))"), "bigint");
    // Without a default the CASE ends in a string NULL.
    assert_eq!(case("select typeof(decode(1, 2, 2, 3))"), "int");
    assert_eq!(case("select typeof(decode(1, 1, 2))"), "bigint");
    assert_eq!(case("select decode(1, 1, 2)"), "2");
    assert_eq!(case("select typeof(decode(1, 1, 'a'))"), "string");
}

#[test]
fn decode_is_named_by_the_call() {
    let s = session();
    assert_eq!(
        names(
            &s,
            "select decode(2, 1, 'Southlake', 2, 'San Francisco', 'x')"
        ),
        ["decode(2, 1, Southlake, 2, San Francisco, x)"]
    );
    assert_eq!(names(&s, "select decode(1, 2, 'a') as c"), ["c"]);
}

#[test]
fn decode_still_decodes_binary_with_two_arguments() {
    let s = session();
    assert_eq!(one(&s, "select decode(X'68656c6c6f', 'UTF-8')"), "hello");
}

#[test]
fn format_string_matches_java() {
    let s = session();
    let f = |q: &str| one(&s, &format!("select {q}"));
    for (q, want) in [
        ("format_string('Hello %s', 'World')", "Hello World"),
        ("format_string('Hello %%%%')", "Hello %%"),
        ("format_string('%1$s %3$s', 1, 2, 3)", "1 3"),
        (
            "format_string('Hello %s %1$s %s', 'World', 'Hello again')",
            "Hello World World Hello again",
        ),
        ("format_string('>>%10s<<', 'Hello')", ">>     Hello<<"),
        ("format_string('>>%-10s<<', NULL)", ">>null      <<"),
        ("format_string('Hello %x', 20)", "Hello 14"),
        (
            "format_string('%x %X %o %#x', -1, 255L, 8, 255)",
            "ffffffff FF 10 0xff",
        ),
        ("format_string('%x', cast(-1 as tinyint))", "ff"),
        (
            "format_string('%,d|%+d|% d|%(d|%05d', 1234567, 5, 5, -5, -42)",
            "1,234,567|+5| 5|(5)|-0042",
        ),
        ("format_string('%c%C', 65, 98)", "AB"),
        ("printf('%s %s', 1.0D, cast(0.1 as float))", "1.0 0.1"),
        ("format_string('%f', cast(0.1 as float))", "0.100000"),
        (
            "format_string('%10.3f|%-10.1f|', 3.14159D, 2.0D)",
            "     3.142|2.0       |",
        ),
        ("format_string('%e', 12345.6789D)", "1.234568e+04"),
        ("format_string('%.2E', 0.000123D)", "1.23E-04"),
        (
            "format_string('%g|%g|%g', 123.456D, 0.00001234D, 0.0D)",
            "123.456|1.23400e-05|0.00000",
        ),
        ("format_string('%s', cast(1.5 as decimal(5,2)))", "1.50"),
        (
            "format_string('%b %b %b', null, 'x', false)",
            "false true false",
        ),
        ("format_string('%5%|%-5%|')", "    %|%    |"),
        (
            "format_string('%+f %(f', cast('Infinity' as double), -2.5D)",
            "+Infinity (2.500000)",
        ),
        ("format_string('%f', 1e20D)", "100000000000000000000.000000"),
        ("format_string('%d|%5s|', null, null)", "null| null|"),
        ("format_string('%s', 1e10D)", "1.0E10"),
        ("format_string('%.2f', 1e-10D)", "0.00"),
        ("format_string(NULL)", "NULL"),
    ] {
        assert_eq!(f(q), want, "{q}");
    }
}

fn class_of(s: &Session, sql: &str) -> ErrorClass {
    let err: KrishivError = match block_on(s.sql(sql)) {
        Ok(df) => block_on(df.collect()).expect_err(sql),
        Err(e) => e,
    };
    err.error_class()
}

#[test]
fn format_string_error_classes() {
    use ErrorClass::{
        InvalidParameterValueZeroIndex as Zero, KrishivInvalidFormatString as Bad,
        WrongNumArgsWithoutSuggestion as Arity,
    };
    let s = session();
    for (q, want) in [
        ("format_string('Hello %s')", Bad),
        ("format_string('%1$s %4$s', 1, 2, 3)", Bad),
        ("format_string('%d', 1.5D)", Bad),
        ("format_string('%.1f', 1.5)", Bad),
        ("format_string('>>%-s<<', 'Hello')", Bad),
        ("format_string('>>%10L<<', NULL)", Bad),
        ("format_string('%1$', 1)", Bad),
        ("format_string('%0$s', 'Hello')", Zero),
        ("format_string()", Arity),
    ] {
        assert_eq!(class_of(&s, &format!("select {q}")), want, "{q}");
    }
}

#[test]
fn decode_refuses_distinct_and_filter() {
    let s = session();
    for q in [
        "decode(DISTINCT 1, 1, 2)",
        "decode(1, 1, 2) FILTER (WHERE true)",
    ] {
        assert_eq!(
            class_of(&s, &format!("select {q}")),
            ErrorClass::InvalidSqlSyntaxFunctionWithUnsupportedSyntax,
            "{q}"
        );
    }
}

/// Each level of nesting puts the inner `decode` in every search condition.
/// The inner expression is shared, not copied, and compiled once per use, so
/// depth with several searches must stay cheap.
#[test]
fn nested_decode_stays_linear_to_plan() {
    let s = session();
    let mut q = String::from("x + 0");
    for _ in 0..12 {
        q = format!("decode({q}, 1, 2, 2, 3, 3, 4, 4, 5)");
    }
    let start = std::time::Instant::now();
    let df = block_on(s.sql(&format!("select {q} from values (1) t(x)"))).expect("plans");
    block_on(df.collect()).expect("runs");
    assert!(start.elapsed().as_secs() < 20, "{:?}", start.elapsed());
}
