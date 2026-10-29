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
