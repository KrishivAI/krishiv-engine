// SPDX-License-Identifier: Apache-2.0
//! `timestamp_seconds`/`_millis`/`_micros`, `unix_seconds`/`_millis`/`_micros`
//! and `date_from_unix_date` end to end, under the fixed offset -08:00 (Los Angeles in winter; region zones need #317). Values
//! come from a Spark 4.2.0 run in that zone.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;

fn session() -> Session {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.session.timeZone = -08:00")).unwrap();
    s
}

fn one(s: &Session, expr: &str) -> Option<String> {
    let sql = format!("SELECT CAST({expr} AS STRING)");
    let df = block_on(s.sql(&sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let batches = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let c = batches[0].column(0);
    (!c.is_null(0)).then(|| match c.data_type() {
        arrow_schema::DataType::Utf8View => c.as_string_view().value(0).to_owned(),
        _ => c.as_string::<i32>().value(0).to_owned(),
    })
}

fn error_of(s: &Session, expr: &str) -> krishiv_common::error::KrishivError {
    let sql = format!("SELECT {expr}");
    block_on(async { s.sql(&sql).await?.collect().await })
        .err()
        .unwrap_or_else(|| panic!("{sql} ran"))
}

#[test]
fn timestamp_seconds_reads_integers_decimals_and_floats() {
    let s = session();
    for (expr, want) in [
        ("timestamp_seconds(1230219000)", Some("2008-12-25 07:30:00")),
        (
            "timestamp_seconds(CAST(1 AS TINYINT))",
            Some("1969-12-31 16:00:01"),
        ),
        ("timestamp_seconds(1.5)", Some("1969-12-31 16:00:01.5")),
        (
            "timestamp_seconds(CAST(1.5 AS FLOAT))",
            Some("1969-12-31 16:00:01.5"),
        ),
        // Past microseconds a float or double is cut, a decimal is refused.
        (
            "timestamp_seconds(CAST(1.9999999 AS DOUBLE))",
            Some("1969-12-31 16:00:01.999999"),
        ),
        (
            "timestamp_seconds(CAST(-0.0000005 AS DOUBLE))",
            Some("1969-12-31 16:00:00"),
        ),
        (
            "timestamp_seconds(CAST(1 AS DECIMAL(10,2)))",
            Some("1969-12-31 16:00:01"),
        ),
        (
            "timestamp_seconds(CAST(-1.2345 AS DECIMAL(10,4)))",
            Some("1969-12-31 15:59:58.7655"),
        ),
        (
            "timestamp_seconds(CAST(1 AS DECIMAL(38,10)))",
            Some("1969-12-31 16:00:01"),
        ),
        ("timestamp_seconds(CAST('NaN' AS DOUBLE))", None),
        ("timestamp_seconds(CAST('Infinity' AS DOUBLE))", None),
        ("timestamp_seconds(NULL)", None),
    ] {
        assert_eq!(one(&s, expr).as_deref(), want, "{expr}");
    }
}

#[test]
fn timestamp_millis_and_micros_take_integers_only() {
    let s = session();
    assert_eq!(
        one(&s, "timestamp_millis(1230219000123)").as_deref(),
        Some("2008-12-25 07:30:00.123")
    );
    assert_eq!(
        one(&s, "timestamp_millis(-1230219000123)").as_deref(),
        Some("1931-01-07 00:29:59.877")
    );
    assert_eq!(
        one(&s, "timestamp_micros(1230219000123123)").as_deref(),
        Some("2008-12-25 07:30:00.123123")
    );
    assert_eq!(one(&s, "timestamp_micros(NULL)"), None);
    for expr in [
        "timestamp_millis(1.5)",
        "timestamp_micros(CAST(5 AS DECIMAL(10,0)))",
        "timestamp_millis('5')",
    ] {
        assert_eq!(
            error_of(&s, expr).error_class(),
            ErrorClass::DatatypeMismatchUnexpectedInputType,
            "{expr}"
        );
    }
}

#[test]
fn epoch_overflow_is_datetime_overflow() {
    let s = session();
    for expr in [
        "timestamp_seconds(1230219000123123)",
        "timestamp_seconds(CAST(9223372036854775807 AS BIGINT))",
        "timestamp_millis(92233720368547758)",
        "timestamp_millis(-92233720368547758)",
        // A decimal with digits past microseconds ("Rounding necessary").
        "timestamp_seconds(0.1234567)",
    ] {
        assert_eq!(
            error_of(&s, expr).error_class(),
            ErrorClass::DatetimeOverflow,
            "{expr}"
        );
    }
}

#[test]
fn unix_time_floors_and_wants_a_timestamp() {
    let s = session();
    for (expr, want) in [
        (
            "unix_seconds(TIMESTAMP '2020-12-01 14:30:08Z')",
            "1606833008",
        ),
        (
            "unix_seconds(TIMESTAMP '2020-12-01 14:30:08.999999Z')",
            "1606833008",
        ),
        (
            "unix_millis(TIMESTAMP '2020-12-01 14:30:08.999999Z')",
            "1606833008999",
        ),
        (
            "unix_micros(TIMESTAMP '2020-12-01 14:30:08.999999Z')",
            "1606833008999999",
        ),
        // Rounded toward negative infinity.
        ("unix_seconds(TIMESTAMP '1969-12-31 23:59:59.5Z')", "-1"),
        ("unix_millis(TIMESTAMP '1969-12-31 23:59:59.9995Z')", "-1"),
    ] {
        assert_eq!(one(&s, expr).as_deref(), Some(want), "{expr}");
    }
    assert_eq!(one(&s, "unix_seconds(NULL)"), None);
    for expr in [
        "unix_seconds(DATE '2020-12-01')",
        "unix_seconds(TIMESTAMP_NTZ '2020-12-01 14:30:08')",
        "unix_seconds('2020-12-01 14:30:08')",
    ] {
        assert_eq!(
            error_of(&s, expr).error_class(),
            ErrorClass::DatatypeMismatchUnexpectedInputType,
            "{expr}"
        );
    }
}

#[test]
fn date_from_unix_date_counts_days() {
    let s = session();
    for (expr, want) in [
        ("date_from_unix_date(1000)", Some("1972-09-27")),
        ("date_from_unix_date(0)", Some("1970-01-01")),
        ("date_from_unix_date(-1)", Some("1969-12-31")),
        (
            "date_from_unix_date(CAST(5.9 AS DOUBLE))",
            Some("1970-01-06"),
        ),
        ("date_from_unix_date('5')", Some("1970-01-06")),
        ("date_from_unix_date(1.5)", Some("1970-01-02")),
        ("date_from_unix_date(NULL)", None),
    ] {
        assert_eq!(one(&s, expr).as_deref(), want, "{expr}");
    }
    for expr in [
        "date_from_unix_date(1000000000)",
        "date_from_unix_date(-1000000000)",
    ] {
        assert_eq!(
            error_of(&s, expr).error_class(),
            ErrorClass::DatetimeOverflow,
            "{expr}"
        );
    }
    // The limits themselves are valid (the string form of such a year is the
    // cast kernel's business, so read the day count back).
    for days in ["999999999", "-999999999"] {
        assert_eq!(
            one(&s, &format!("unix_date(date_from_unix_date({days}))")).as_deref(),
            Some(days)
        );
    }
}
