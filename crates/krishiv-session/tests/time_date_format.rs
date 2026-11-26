// SPDX-License-Identifier: Apache-2.0
//! `date_format` of a `TIME` under the Spark 4.3 profile: Spark 4.3's `DateFormatClass` prints a
//! time through `TimeFormatter` over its nanoseconds, and a pattern with a date or zone field the
//! time cannot fill raises `INVALID_PARAMETER_VALUE.PATTERN`. Every expectation is from a Spark
//! 4.3.0 run (local mode, `spark.sql.timeType.enabled=true`, PySpark 4.3.0, 2026-11-26).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

fn session() -> Session {
    let s = Session::builder()
        .profile(SparkProfile::V4_3)
        .build()
        .expect("session");
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).expect("conf");
    s
}

/// Every row's first column, `None` for NULL.
fn rows_on(s: &Session, sql: &str) -> Result<Vec<Option<String>>, KrishivError> {
    let df = block_on(s.sql(sql))?;
    let mut out = Vec::new();
    for b in block_on(df.collect())? {
        let c = &b.columns()[0];
        for i in 0..c.len() {
            out.push(c.is_valid(i).then(|| match c.data_type() {
                arrow_schema::DataType::Utf8View => c.as_string_view().value(i).to_owned(),
                _ => c.as_string::<i32>().value(i).to_owned(),
            }));
        }
    }
    Ok(out)
}

fn rows(sql: &str) -> Result<Vec<Option<String>>, KrishivError> {
    rows_on(&session(), sql)
}

fn one(sql: &str) -> String {
    rows(sql).unwrap()[0].clone().unwrap()
}

#[test]
fn a_time_prints_its_time_fields() {
    for (sql, want) in [
        ("SELECT date_format(TIME'14:30:45', 'HH:mm:ss')", "14:30:45"),
        (
            "SELECT date_format(TIME'12:00:00.123456789', 'HH:mm:ss.SSSSSSSSS')",
            "12:00:00.123456789",
        ),
        (
            "SELECT date_format(TIME'12:00:00.123456789', 'ss.SSS')",
            "00.123",
        ),
        ("SELECT date_format(TIME'00:05:00', 'hh:mm a')", "12:05 AM"),
        (
            "SELECT date_format(TIME'23:05:00', 'h:m a k K')",
            "11:5 PM 23 11",
        ),
        ("SELECT date_format(TIME'01:02:03', 'HH''h''mm')", "01h02"),
        // An optional section with a date field is skipped.
        ("SELECT date_format(TIME'01:02:03', 'HH[ yyyy]')", "01"),
    ] {
        assert_eq!(one(sql), want, "{sql}");
    }
}

#[test]
fn a_pattern_column_is_read_per_row_and_null_gives_null() {
    assert_eq!(
        rows(
            "SELECT date_format(t, f) FROM VALUES (TIME'01:02:03', 'HH:mm'), \
             (TIME'04:05:06', 'mm:ss') AS v(t, f)"
        )
        .unwrap(),
        [Some("01:02".to_owned()), Some("05:06".to_owned())]
    );
    for sql in [
        "SELECT date_format(CAST(NULL AS TIME), 'HH')",
        "SELECT date_format(TIME'01:02:03', CAST(NULL AS STRING))",
    ] {
        assert_eq!(rows(sql).unwrap(), [None], "{sql}");
    }
}

#[test]
fn a_date_or_zone_field_is_an_invalid_pattern() {
    for (sql, pattern) in [
        ("SELECT date_format(TIME'01:02:03', 'yyyy HH')", "yyyy HH"),
        ("SELECT date_format(TIME'01:02:03', 'HH Z')", "HH Z"),
        ("SELECT date_format(TIME'01:02:03', 'MMM')", "MMM"),
        ("SELECT date_format(TIME'01:02:03', 'EEE')", "EEE"),
        ("SELECT date_format(TIME'01:02:03', 'HH qqq')", "HH qqq"),
        (
            "SELECT date_format(t, f) FROM VALUES (TIME'01:02:03', 'HH:mm'), \
             (TIME'04:05:06', 'yyyy') AS v(t, f)",
            "yyyy",
        ),
    ] {
        let e = rows(sql).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::InvalidParameterValuePattern,
            "{sql}: {e}"
        );
        assert_eq!(
            e.message(),
            format!("The value of parameter(s) `format` in `date_format` is invalid: '{pattern}'."),
            "{sql}"
        );
    }
    // A pattern the formatter cannot compile: over a `TIME` always this class, with no legacy check
    // (a `TIMESTAMP` gives `INCONSISTENT_BEHAVIOR_CROSS_VERSION.*` or `ILLEGAL_CHARACTER` for these).
    for pattern in [
        "HH:mm:ss.SSSSSSSSSS",
        "HH YYYY",
        "HH c",
        "HH A",
        "HHH",
        "HH {",
    ] {
        let e = rows(&format!(
            "SELECT date_format(TIME'01:02:03.5', '{pattern}')"
        ))
        .unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::InvalidDatetimePatternWithSuggestion,
            "{pattern}: {e}"
        );
    }
}

#[test]
fn the_legacy_parser_policy_does_not_change_a_time() {
    let s = session();
    block_on(s.sql("SET spark.sql.legacy.timeParserPolicy = LEGACY")).expect("conf");
    assert_eq!(
        rows_on(
            &s,
            "SELECT date_format(TIME'01:02:03.123456789', 'HH:mm:ss.SSSSSSSSS')"
        )
        .unwrap(),
        [Some("01:02:03.123456789".to_owned())]
    );
}
