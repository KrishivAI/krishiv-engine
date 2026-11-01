// SPDX-License-Identifier: Apache-2.0
//! `make_time`, `time_from_*`, `time_to_*`, `to_time`, `try_to_time` and
//! `time()` end to end, with the values and errors of the Spark 4.2.0
//! `time.sql` golden.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_session::Session;

fn session() -> Session {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    s
}

fn run(expr: &str) -> Result<String, KrishivError> {
    let s = session();
    let df = block_on(s.sql(&format!("SELECT CAST(({expr}) AS STRING)")))?;
    let b = block_on(df.collect())?;
    let c = &b[0].columns()[0];
    Ok(if c.is_null(0) {
        "NULL".to_owned()
    } else {
        c.as_string_view().value(0).to_owned()
    })
}

fn value(expr: &str) -> String {
    run(expr).unwrap()
}

fn class(expr: &str) -> ErrorClass {
    run(expr).unwrap_err().error_class()
}

#[test]
fn make_time_builds_a_time_or_names_the_bad_field() {
    assert_eq!(value("make_time(1, 18, 19.87)"), "01:18:19.87");
    assert_eq!(value("make_time(23, 59, 59.999999)"), "23:59:59.999999");
    for expr in [
        "make_time(NULL, 18, 19.87)",
        "make_time(1, NULL, 19.87)",
        "make_time(1, 18, NULL)",
    ] {
        assert_eq!(value(expr), "NULL", "{expr}");
    }
    for expr in [
        "make_time(-1, 18, 19.87)",
        "make_time(1, 60, 19.87)",
        "make_time(1, 18, 60.0)",
        "make_time(1, 0, -0.5)",
        "make_time(0, 0, -0.000001)",
        "make_time(1, 18, 9999999999.999999)",
    ] {
        assert_eq!(
            class(expr),
            ErrorClass::DatetimeFieldOutOfBoundsWithoutSuggestion,
            "{expr}"
        );
    }
}

#[test]
fn time_from_and_to_units_agree_with_the_golden() {
    for (expr, want) in [
        ("time_from_seconds(0)", "00:00:00"),
        ("time_from_seconds(52200.5)", "14:30:00.5"),
        ("time_from_seconds(86399.999999)", "23:59:59.999999"),
        ("time_from_millis(52200500)", "14:30:00.5"),
        ("time_from_micros(52200500000)", "14:30:00.5"),
        ("time_from_micros(NULL)", "NULL"),
        ("time_to_seconds(TIME '14:30:00.5')", "52200.500000"),
        ("time_to_seconds(NULL)", "NULL"),
        ("time_to_millis(TIME '14:30:00.5')", "52200500"),
        ("time_to_millis(TIME '23:59:59.999999')", "86399999"),
        ("time_to_micros(TIME '23:59:59.999999')", "86399999999"),
        (
            "time_to_micros(time_from_micros(52200500000))",
            "52200500000",
        ),
    ] {
        assert_eq!(value(expr), want, "{expr}");
    }
    for expr in [
        "time_from_seconds(-1)",
        "time_from_seconds(86400)",
        "time_from_millis(-1)",
        "time_from_millis(86400000)",
        "time_from_micros(86400000000)",
    ] {
        assert_eq!(
            class(expr),
            ErrorClass::DatetimeFieldOutOfBoundsWithoutSuggestion,
            "{expr}"
        );
    }
}

#[test]
fn to_time_try_to_time_and_time() {
    for (expr, want) in [
        ("to_time('01:02:03')", "01:02:03"),
        (
            "to_time('23-59-59.999999', 'HH-mm-ss.SSSSSS')",
            "23:59:59.999999",
        ),
        ("to_time('11', 'HH')", "11:00:00"),
        ("to_time(NULL)", "NULL"),
        ("to_time('01:02:03', NULL)", "NULL"),
        ("try_to_time('00:12:00')", "00:12:00"),
        ("try_to_time('12:48:31 abc')", "NULL"),
        ("try_to_time('02-69', 'HH-mm')", "NULL"),
        ("time('12:34:56.789')", "12:34:56.789"),
        ("time(NULL)", "NULL"),
    ] {
        assert_eq!(value(expr), want, "{expr}");
    }
    for expr in ["to_time('13-60', 'HH-mm')", "to_time('abc')"] {
        assert_eq!(class(expr), ErrorClass::CannotParseTime, "{expr}");
    }
    assert_eq!(class("time('abc')"), ErrorClass::CastInvalidInput);
}
