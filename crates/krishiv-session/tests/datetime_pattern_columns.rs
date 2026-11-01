// SPDX-License-Identifier: Apache-2.0
//! Datetime functions with per-row patterns, non-string first arguments
//! and zone conversions end to end. Expected rows come from a Spark 4.2.0
//! run in the UTC session zone.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_session::Session;

/// The single column of `SELECT CAST(<expr> AS STRING) <from>` as text.
fn column(s: &Session, expr: &str, from: &str) -> Vec<Option<String>> {
    let sql = format!("SELECT CAST({expr} AS STRING) {from}");
    let df = block_on(s.sql(&sql)).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let batches = block_on(df.collect()).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let mut out = Vec::new();
    for b in &batches {
        let c = b.column(0);
        for i in 0..b.num_rows() {
            out.push((!c.is_null(i)).then(|| match c.data_type() {
                arrow_schema::DataType::Utf8View => c.as_string_view().value(i).to_owned(),
                _ => c.as_string::<i32>().value(i).to_owned(),
            }));
        }
    }
    out
}

fn session() -> Session {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.session.timeZone = UTC")).unwrap();
    s
}

fn text(rows: &[Option<&str>]) -> Vec<Option<String>> {
    rows.iter().map(|r| r.map(str::to_owned)).collect()
}

const FORMATS: &str = "FROM VALUES ('yyyy'), ('MM/dd'), (CAST(NULL AS STRING)) AS t(f)";
const PARSE_DATES: &str = "FROM VALUES ('2024-01-05', 'yyyy-MM-dd'), ('05/01/2024', 'dd/MM/yyyy'), \
                           (CAST(NULL AS STRING), 'yyyy') AS t(s, f)";
const PARSE_STAMPS: &str = "FROM VALUES ('2024-01-05 10:00:00', 'yyyy-MM-dd HH:mm:ss'), \
                            ('05.01.2024', 'dd.MM.yyyy') AS t(s, f)";

/// A pattern per row formats and parses per row.
#[test]
fn patterns_may_vary_per_row() {
    let s = session();
    assert_eq!(
        column(
            &s,
            "date_format(TIMESTAMP '2024-03-05 10:20:30', f)",
            FORMATS
        ),
        text(&[Some("2024"), Some("03/05"), None])
    );
    assert_eq!(
        column(&s, "to_date(s, f)", PARSE_DATES),
        text(&[Some("2024-01-05"), Some("2024-01-05"), None])
    );
    assert_eq!(
        column(
            &s,
            "try_to_date(s, f)",
            "FROM VALUES ('2024-01-05', 'yyyy-MM-dd'), ('zz', 'dd/MM/yyyy') AS t(s, f)"
        ),
        text(&[Some("2024-01-05"), None])
    );
    assert_eq!(
        column(&s, "to_timestamp(s, f)", PARSE_STAMPS),
        text(&[Some("2024-01-05 10:00:00"), Some("2024-01-05 00:00:00")])
    );
    assert_eq!(
        column(
            &s,
            "try_to_timestamp(s, f)",
            "FROM VALUES ('2024-01-05 10:00:00', 'yyyy-MM-dd HH:mm:ss'), ('x', 'dd.MM.yyyy') AS t(s, f)"
        ),
        text(&[Some("2024-01-05 10:00:00"), None])
    );
    assert_eq!(
        column(&s, "to_timestamp_ntz(s, f)", PARSE_STAMPS),
        text(&[Some("2024-01-05 10:00:00"), Some("2024-01-05 00:00:00")])
    );
    assert_eq!(
        column(
            &s,
            "to_timestamp_ltz(s, f)",
            "FROM VALUES ('2024-01-05 10:00:00', 'yyyy-MM-dd HH:mm:ss') AS t(s, f)"
        ),
        text(&[Some("2024-01-05 10:00:00")])
    );
    assert_eq!(
        column(&s, "unix_timestamp(s, f)", PARSE_STAMPS),
        text(&[Some("1704448800"), Some("1704412800")])
    );
    assert_eq!(
        column(
            &s,
            "to_unix_timestamp(s, f)",
            "FROM VALUES ('2024-01-05', 'yyyy-MM-dd') AS t(s, f)"
        ),
        text(&[Some("1704412800")])
    );
    assert_eq!(
        column(
            &s,
            "from_unixtime(n, f)",
            "FROM VALUES (0L, 'yyyy-MM-dd'), (86400L, 'HH:mm') AS t(n, f)"
        ),
        text(&[Some("1970-01-01"), Some("00:00")])
    );
    assert_eq!(
        column(
            &s,
            "date_format(DATE '2024-01-05', f)",
            "FROM VALUES ('EEE'), ('D') AS t(f)"
        ),
        text(&[Some("Fri"), Some("5")])
    );
}

/// Dates and timestamps ignore the pattern; a NULL pattern still gives NULL.
#[test]
fn non_string_inputs_ignore_the_pattern() {
    let s = session();
    for (expr, expected) in [
        ("to_date(DATE '2024-01-05', 'yyyy')", "2024-01-05"),
        (
            "to_date(TIMESTAMP '2024-01-05 10:00:00', 'yyyy')",
            "2024-01-05",
        ),
        (
            "to_timestamp(TIMESTAMP '2024-01-05 10:00:00', 'yyyy')",
            "2024-01-05 10:00:00",
        ),
        (
            "to_timestamp(DATE '2024-01-05', 'yyyy')",
            "2024-01-05 00:00:00",
        ),
        (
            "to_timestamp_ntz(TIMESTAMP '2024-01-05 10:00:00', 'yyyy')",
            "2024-01-05 10:00:00",
        ),
        (
            "unix_timestamp(TIMESTAMP '2024-01-05 10:00:00', 'yyyy')",
            "1704448800",
        ),
        ("unix_timestamp(DATE '2024-01-05', 'yyyy')", "1704412800"),
        (
            "from_unixtime(0L, 'yyyy-MM-dd HH:mm:ss')",
            "1970-01-01 00:00:00",
        ),
    ] {
        assert_eq!(column(&s, expr, ""), text(&[Some(expected)]), "{expr}");
    }
    assert_eq!(column(&s, "from_unixtime(NULL, 'yyyy')", ""), text(&[None]));
    assert_eq!(
        column(&s, "to_date(DATE '2024-01-05', CAST(NULL AS STRING))", ""),
        text(&[None])
    );
}

/// Region zones convert with daylight saving.
#[test]
fn region_zones_convert() {
    let s = session();
    assert_eq!(
        column(
            &s,
            "convert_timezone('Europe/Brussels', 'America/New_York', TIMESTAMP_NTZ '2024-01-05 10:00:00')",
            ""
        ),
        text(&[Some("2024-01-05 04:00:00")])
    );
    assert_eq!(
        column(
            &s,
            "from_utc_timestamp(TIMESTAMP '2024-07-05 10:00:00', 'Asia/Kolkata')",
            ""
        ),
        text(&[Some("2024-07-05 15:30:00")])
    );
    assert_eq!(
        column(
            &s,
            "to_utc_timestamp(TIMESTAMP '2024-07-05 10:00:00', 'Europe/Paris')",
            ""
        ),
        text(&[Some("2024-07-05 08:00:00")])
    );
}

/// Numeric arguments (seconds), zone fields, day of year, am/pm and
/// two-digit years parse as Spark does.
#[test]
fn numeric_and_zone_pattern_inputs() {
    let s = session();
    let cases: &[(&str, Option<&str>)] = &[
        (
            "to_timestamp(CAST(1 AS TINYINT))",
            Some("1970-01-01 00:00:01"),
        ),
        (
            "to_timestamp(CAST(2 AS SMALLINT))",
            Some("1970-01-01 00:00:02"),
        ),
        ("to_timestamp(3)", Some("1970-01-01 00:00:03")),
        ("to_timestamp(4L)", Some("1970-01-01 00:00:04")),
        (
            "to_timestamp(CAST(1.5 AS FLOAT))",
            Some("1970-01-01 00:00:01.5"),
        ),
        ("to_timestamp(1.5D)", Some("1970-01-01 00:00:01.5")),
        (
            "to_timestamp(CAST(1.5 AS DECIMAL(5,1)))",
            Some("1970-01-01 00:00:01.5"),
        ),
        (
            "to_timestamp('2024-01-05 10:00:00 +01:00', 'yyyy-MM-dd HH:mm:ss XXX')",
            Some("2024-01-05 09:00:00"),
        ),
        (
            "to_timestamp('2024-01-05 10:00:00 Europe/Brussels', 'yyyy-MM-dd HH:mm:ss VV')",
            Some("2024-01-05 09:00:00"),
        ),
        (
            "to_timestamp('2024-01-05 10:00:00 CET', 'yyyy-MM-dd HH:mm:ss z')",
            Some("2024-01-05 09:00:00"),
        ),
        (
            "to_timestamp('2024-01-05 10:00:00 GMT+02:00', 'yyyy-MM-dd HH:mm:ss O')",
            Some("2024-01-05 08:00:00"),
        ),
        (
            "to_timestamp('2024-01-05 10:00:00 +0100', 'yyyy-MM-dd HH:mm:ss Z')",
            Some("2024-01-05 09:00:00"),
        ),
        (
            "to_timestamp('2024-01-05 10:00:00.123456', 'yyyy-MM-dd HH:mm:ss.SSSSSS')",
            Some("2024-01-05 10:00:00.123456"),
        ),
        (
            "to_timestamp('2024-01-05 10:00:00 PM', 'yyyy-MM-dd hh:mm:ss a')",
            Some("2024-01-05 22:00:00"),
        ),
        (
            "to_timestamp('2024-01-05 03:04:05 PM', 'yyyy-MM-dd hh:mm:ss a')",
            Some("2024-01-05 15:04:05"),
        ),
        (
            "to_timestamp('2024 5', 'yyyy D')",
            Some("2024-01-05 00:00:00"),
        ),
        (
            "to_timestamp('24-1-5', 'yy-M-d')",
            Some("2024-01-05 00:00:00"),
        ),
        (
            "to_timestamp('January 5, 2024', 'MMMM d, yyyy')",
            Some("2024-01-05 00:00:00"),
        ),
        (
            "to_timestamp_ntz('2024-01-05 10:00:00 +01:00', 'yyyy-MM-dd HH:mm:ss XXX')",
            Some("2024-01-05 10:00:00"),
        ),
        ("try_to_timestamp('2024-13-45', 'yyyy-MM-dd')", None),
        (
            "to_date('2024-01-05 10:00:00 +01:00', 'yyyy-MM-dd HH:mm:ss XXX')",
            Some("2024-01-05"),
        ),
        (
            "date_format(TIMESTAMP '2024-01-05 15:04:05.123456', 'hh:mm:ss a SSS EEEE MMMM D z VV XXX O')",
            Some("03:04:05 PM 123 Friday January 5 UTC UTC Z GMT"),
        ),
    ];
    for (expr, expected) in cases {
        assert_eq!(column(&s, expr, ""), text(&[*expected]), "{expr}");
    }
}

/// Digits beyond the micros of a decimal number of seconds are dropped
/// toward zero (Spark: `00:00:01.5`), not refused.
#[test]
fn decimal_seconds_truncate_to_micros() {
    let s = session();
    assert_eq!(
        column(&s, "to_timestamp(CAST(1.5000001 AS DECIMAL(8,7)))", ""),
        text(&[Some("1970-01-01 00:00:01.5")])
    );
    assert_eq!(
        column(&s, "to_timestamp(CAST(-1.5000009 AS DECIMAL(8,7)))", ""),
        text(&[Some("1969-12-31 23:59:58.5")])
    );
}

/// `Etc/GMT+5` is a zone; a zero-padded hour is not.
#[test]
fn etc_gmt_zones_reject_a_padded_hour() {
    let s = session();
    assert_eq!(
        column(
            &s,
            "from_utc_timestamp(TIMESTAMP '2024-07-05 10:00:00', 'Etc/GMT+5')",
            ""
        ),
        text(&[Some("2024-07-05 05:00:00")])
    );
    let df =
        block_on(s.sql("SELECT from_utc_timestamp(TIMESTAMP '2024-07-05 10:00:00', 'Etc/GMT+05')"));
    let err = match df {
        Ok(df) => block_on(df.collect()).unwrap_err(),
        Err(e) => e,
    };
    assert_eq!(
        err.error_class(),
        krishiv_common::error::ErrorClass::InvalidTimezone
    );
}

/// `spark.sql.legacy.timeParserPolicy = LEGACY` reads and prints patterns
/// with the `SimpleDateFormat` family; the other policies behave as before.
#[test]
fn the_legacy_time_parser_reads_and_prints_legacy_patterns() {
    let s = session();
    block_on(s.sql("SET spark.sql.legacy.timeParserPolicy = LEGACY")).unwrap();
    // `S` is the millisecond count, not fraction digits.
    assert_eq!(
        column(
            &s,
            "date_format(TIMESTAMP '2024-01-05 10:00:00.123456', 'SSSS')",
            ""
        ),
        text(&[Some("0123")])
    );
    // A two-digit year reads in the 80/20 window.
    assert_eq!(
        column(&s, "to_date('99-01-05', 'yy-MM-dd')", ""),
        text(&[Some("1999-01-05")])
    );
    // `Q` is no legacy pattern letter.
    let err = block_on(s.sql("SELECT date_format(TIMESTAMP '2024-01-05 10:00:00', 'Q')"))
        .and_then(|df| block_on(df.collect()))
        .expect_err("Q is rejected");
    assert!(
        err.to_string().contains("INVALID_DATETIME_PATTERN"),
        "{err}"
    );
    // One-argument forms are casts, not patterns.
    assert_eq!(
        column(&s, "to_timestamp('2024-01-05')", ""),
        text(&[Some("2024-01-05 00:00:00")])
    );
    block_on(s.sql("SET spark.sql.legacy.timeParserPolicy = CORRECTED")).unwrap();
    assert_eq!(
        column(&s, "to_date('2024-01-05', 'yyyy-MM-dd')", ""),
        text(&[Some("2024-01-05")])
    );
}

/// The legacy parser is not lenient but stops where the pattern does, the
/// legacy printer knows the week letters, and region offsets before 1900
/// are `java.util.TimeZone`'s raw offset (each as Spark 4.2.0 answers).
#[test]
fn the_legacy_time_parser_matches_spark_edges() {
    let s = session();
    block_on(s.sql("SET spark.sql.legacy.timeParserPolicy = LEGACY")).unwrap();
    for (expr, want) in [
        ("try_to_timestamp('2020-13-01', 'yyyy-MM-dd')", None),
        (
            "try_to_timestamp('2020-01-01 25:00', 'yyyy-MM-dd HH:mm')",
            None,
        ),
        (
            "unix_timestamp('2020-01-01x', 'yyyy-MM-dd')",
            Some("1577836800"),
        ),
        (
            "try_to_timestamp('2020-01-01 trailing', 'yyyy-MM-dd')",
            Some("2020-01-01 00:00:00"),
        ),
        (
            "date_format(TIMESTAMP '2020-01-05 10:00:00', 'YYYY ww W u F E')",
            Some("2020 02 2 7 1 Sun"),
        ),
    ] {
        assert_eq!(column(&s, expr, ""), text(&[want]), "{expr}");
    }
    block_on(s.sql("SET spark.sql.session.timeZone = Asia/Kolkata")).unwrap();
    for (at, want) in [
        ("1850-06-01 00:00:00", "1850-06-01 00:00:00 +0530"),
        ("1900-06-01 00:00:00", "1900-06-01 00:00:00 +0521"),
    ] {
        assert_eq!(
            column(
                &s,
                &format!("date_format(TIMESTAMP '{at}', 'yyyy-MM-dd HH:mm:ss Z')"),
                ""
            ),
            text(&[Some(want)]),
            "{at}"
        );
    }
}

/// `current_timezone()` returns the configured string and the patterns
/// print its zone id, as Spark 4.2.0 does for each of these.
#[test]
fn the_session_zone_name_is_the_configured_one() {
    let at = "TIMESTAMP '2024-01-01 12:00:00'";
    for (zone, id, name, long) in [
        ("UTC", "UTC", "UTC", "Coordinated Universal Time"),
        ("GMT", "GMT", "GMT", "Greenwich Mean Time"),
        ("Z", "Z", "Z", "Z"),
        ("UT", "UT", "UT", "UT"),
        ("+00:00", "Z", "Z", "Z"),
        ("-00:00", "Z", "Z", "Z"),
        ("GMT+00:00", "GMT", "GMT", "Greenwich Mean Time"),
        ("Etc/UTC", "Etc/UTC", "UTC", "Coordinated Universal Time"),
        ("+05:30", "+05:30", "+05:30", "+05:30"),
        ("+5:30", "+05:30", "+05:30", "+05:30"),
        ("GMT+5", "GMT+05:00", "GMT+05:00", "GMT+05:00"),
        ("UTC+05:30", "UTC+05:30", "UTC+05:30", "UTC+05:30"),
    ] {
        let s = Session::new().unwrap();
        block_on(s.sql(&format!("SET spark.sql.session.timeZone = {zone}"))).unwrap();
        assert_eq!(column(&s, "current_timezone()", ""), text(&[Some(zone)]));
        for (pattern, want) in [("VV", id), ("z", name), ("zzzz", long)] {
            assert_eq!(
                column(&s, &format!("date_format({at}, '{pattern}')"), ""),
                text(&[Some(want)]),
                "{zone} {pattern}"
            );
        }
    }
}

/// A region zone has no rules here yet, but `current_timezone()` reads only
/// its name.
#[test]
fn a_region_session_zone_has_its_name() {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.session.timeZone = America/Los_Angeles")).unwrap();
    assert_eq!(
        column(&s, "current_timezone()", ""),
        text(&[Some("America/Los_Angeles")])
    );
}

/// Under a region session zone the calls that read it (strings with an
/// offset, zone letters, dynamic patterns) run on the database's rules; the
/// values are Spark 4.2.0's for `America/Los_Angeles`.
#[test]
fn a_region_session_zone_serves_the_calls_that_read_it() {
    let s = Session::new().unwrap();
    block_on(s.sql("SET spark.sql.session.timeZone = America/Los_Angeles")).unwrap();
    for (expr, want) in [
        (
            "date_format('2024-01-05 10:00:00+08:00', 'yyyy-MM-dd')",
            "2024-01-04",
        ),
        (
            "date_format(DATE '2024-01-05', 'VV')",
            "America/Los_Angeles",
        ),
        (
            "date_format(TIMESTAMP_NTZ '2024-01-05 10:00:00', 'yyyy-MM-dd XXX')",
            "2024-01-05 -08:00",
        ),
        (
            "to_date('2024-01-05 +08:00', 'yyyy-MM-dd XXX')",
            "2024-01-04",
        ),
        ("date_format(DATE '2024-01-05', 'yyyy-MM-dd')", "2024-01-05"),
        (
            "date_format(TIMESTAMP_NTZ '2024-01-05 10:00:00', 'yyyy-MM-dd HH')",
            "2024-01-05 10",
        ),
        ("to_date('2024-01-05', 'yyyy-MM-dd')", "2024-01-05"),
        ("to_date('2024-01-05')", "2024-01-05"),
    ] {
        assert_eq!(column(&s, expr, ""), text(&[Some(want)]), "{expr}");
    }
    assert_eq!(
        column(
            &s,
            "date_format(DATE '2024-01-05', f)",
            "FROM VALUES ('yyyy'), ('MM') AS t(f)"
        ),
        text(&[Some("2024"), Some("01")])
    );
    assert_eq!(
        column(
            &s,
            "try_to_date('2024-01-05', f)",
            "FROM VALUES ('yyyy-MM-dd'), ('dd') AS t(f)"
        ),
        text(&[Some("2024-01-05"), None])
    );
    // A fixed-offset zone has no such limit.
    block_on(s.sql("SET spark.sql.session.timeZone = +08:00")).unwrap();
    assert_eq!(
        column(&s, "to_date('2024-01-05 +08:00', 'yyyy-MM-dd XXX')", ""),
        text(&[Some("2024-01-05")])
    );
}
