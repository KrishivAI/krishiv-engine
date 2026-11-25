// SPDX-License-Identifier: Apache-2.0
//! `TIME` to and from `TIMESTAMP` and `TIMESTAMP_NTZ` casts (Spark 4.3). Values and errors are
//! those of Spark 4.3.0 (`sql-tests/results/cast.sql.out` and a local run with the session zones
//! used here); Spark 4.2.0 refuses every one of these casts.
#![forbid(unsafe_code)]
#![cfg(test)]

use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

fn session(profile: SparkProfile, zone: &str, ansi: bool) -> Session {
    let s = Session::builder().profile(profile).build().unwrap();
    for set in [
        "SET spark.sql.timeType.enabled = true".to_owned(),
        format!("SET spark.sql.session.timeZone = {zone}"),
        format!("SET spark.sql.ansi.enabled = {ansi}"),
    ] {
        block_on(s.sql(&set)).unwrap();
    }
    s
}

/// The one value of `expr` as text.
fn text(s: &Session, expr: &str) -> Result<Option<String>, KrishivError> {
    let df = block_on(s.sql(&format!("SELECT CAST(({expr}) AS STRING)")))?;
    let batches = block_on(df.collect())?;
    let c = batches[0].column(0);
    let valid = c.is_valid(0);
    Ok(valid.then(|| match c.data_type() {
        arrow_schema::DataType::Utf8 => c.as_string::<i32>().value(0).to_owned(),
        arrow_schema::DataType::Utf8View => c.as_string_view().value(0).to_owned(),
        other => panic!("unexpected result type {other}"),
    }))
}

#[track_caller]
fn check(s: &Session, cases: &[(&str, Option<&str>)]) {
    for (expr, want) in cases {
        let got = text(s, expr).unwrap_or_else(|e| panic!("{expr}: {e}"));
        assert_eq!(got.as_deref(), *want, "{expr}");
    }
}

#[test]
fn a_timestamp_casts_to_its_wall_clock_time() {
    for ansi in [true, false] {
        let s = session(SparkProfile::V4_3, "UTC", ansi);
        check(
            &s,
            &[
                (
                    "CAST(TIMESTAMP_NTZ'2020-05-17 12:34:56.789012' AS TIME(6))",
                    Some("12:34:56.789012"),
                ),
                (
                    "CAST(TIMESTAMP_NTZ'2020-05-17 12:34:56.789012' AS TIME(3))",
                    Some("12:34:56.789"),
                ),
                (
                    "CAST(TIMESTAMP_NTZ'2020-05-17 12:34:56.789012' AS TIME(0))",
                    Some("12:34:56"),
                ),
                (
                    "CAST(TIMESTAMP_NTZ'1969-12-31 23:59:59.123456' AS TIME(6))",
                    Some("23:59:59.123456"),
                ),
                (
                    "TIMESTAMP_NTZ'2020-05-17 12:34:56.789012' :: TIME(6)",
                    Some("12:34:56.789012"),
                ),
                (
                    "CAST(TIMESTAMP'2020-05-17 12:34:56.789012' AS TIME(3))",
                    Some("12:34:56.789"),
                ),
                ("CAST(CAST(NULL AS TIMESTAMP_NTZ) AS TIME(6))", None),
                ("CAST(CAST(NULL AS TIMESTAMP) AS TIME(6))", None),
                (
                    "TRY_CAST(TIMESTAMP_NTZ'2020-01-01 10:00:00.5' AS TIME(0))",
                    Some("10:00:00"),
                ),
                (
                    "CAST(named_struct('t', TIMESTAMP_NTZ'2020-01-01 10:00:00.25') AS STRUCT<t: TIME(1)>)",
                    Some("{10:00:00.2}"),
                ),
            ],
        );
    }
}

#[test]
fn a_timestamp_is_read_in_the_session_zone_and_a_timestamp_ntz_is_not() {
    for (zone, local, before_epoch) in [
        ("America/Los_Angeles", "05:34:56.789012", "15:59:59.123"),
        ("+05:30", "18:04:56.789012", "05:29:59.123"),
    ] {
        let s = session(SparkProfile::V4_3, zone, true);
        check(
            &s,
            &[
                (
                    "CAST(TIMESTAMP'2020-05-17 12:34:56.789012Z' AS TIME(6))",
                    Some(local),
                ),
                (
                    "CAST(TIMESTAMP'1969-12-31 23:59:59.123456Z' AS TIME(3))",
                    Some(before_epoch),
                ),
                (
                    "CAST(TIMESTAMP_NTZ'2020-05-17 12:34:56' AS TIME(6))",
                    Some("12:34:56"),
                ),
            ],
        );
    }
}

#[test]
fn a_time_casts_to_that_time_on_the_current_date() {
    for zone in ["UTC", "America/Los_Angeles", "+05:30"] {
        let s = session(SparkProfile::V4_3, zone, true);
        // 12:34 exists on every date, DST transitions included.
        check(
            &s,
            &[
                (
                    "CAST(CAST(TIME'12:34:56.789012' AS TIMESTAMP_NTZ) AS STRING) \
                     = concat(CAST(current_date() AS STRING), ' 12:34:56.789012')",
                    Some("true"),
                ),
                (
                    "CAST(CAST(TIME'12:34:56.789012' AS TIMESTAMP) AS STRING) \
                     = concat(CAST(current_date() AS STRING), ' 12:34:56.789012')",
                    Some("true"),
                ),
                (
                    "CAST(CAST(TIME'12:34:56' AS TIMESTAMP_NTZ) AS DATE) = current_date()",
                    Some("true"),
                ),
                (
                    "CAST(CAST(TIME'12:34:56.789012' AS TIMESTAMP) AS TIME(6))",
                    Some("12:34:56.789012"),
                ),
                (
                    "typeof(CAST(TIME'12:34:56' AS TIMESTAMP_NTZ))",
                    Some("timestamp_ntz"),
                ),
                (
                    "typeof(CAST(TIME'12:34:56' AS TIMESTAMP_LTZ))",
                    Some("timestamp"),
                ),
                ("CAST(CAST(NULL AS TIME) AS TIMESTAMP)", None),
                ("CAST(CAST(NULL AS TIME) AS TIMESTAMP_NTZ)", None),
                (
                    "TRY_CAST(TIME'01:02:03' AS TIMESTAMP) IS NOT NULL",
                    Some("true"),
                ),
                (
                    "CAST(array(TIME'12:34:56') AS ARRAY<TIMESTAMP_NTZ>)[0] \
                     = CAST(TIME'12:34:56' AS TIMESTAMP_NTZ)",
                    Some("true"),
                ),
            ],
        );
        // A column, so the cast runs in the executor rather than on a constant.
        let df = block_on(s.sql(
            "SELECT CAST(CAST(CAST(x AS TIMESTAMP_NTZ) AS TIME(6)) AS STRING) \
             FROM VALUES (TIME'12:34:56.789012') t(x)",
        ))
        .unwrap();
        let batches = block_on(df.collect()).unwrap();
        assert_eq!(
            batches[0].column(0).as_string_view().value(0),
            "12:34:56.789012"
        );
    }
}

#[test]
fn spark_4_2_refuses_the_casts() {
    let s = session(SparkProfile::V4_2, "UTC", true);
    for expr in [
        "CAST(TIME'12:34:56' AS TIMESTAMP)",
        "CAST(TIME'12:34:56' AS TIMESTAMP_NTZ)",
        "CAST(TIMESTAMP'2020-05-17 12:34:56' AS TIME(6))",
        "CAST(TIMESTAMP_NTZ'2020-05-17 12:34:56' AS TIME(6))",
        "CAST(array(TIME'12:34:56') AS ARRAY<TIMESTAMP>)",
    ] {
        let e = text(&s, expr).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::DatatypeMismatchCastWithoutSuggestion,
            "{expr}: {e}"
        );
    }
}

#[test]
fn spark_4_3_still_refuses_a_time_to_a_date() {
    let s = session(SparkProfile::V4_3, "UTC", true);
    let e = text(&s, "CAST(TIME'12:34:56' AS DATE)").unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::DatatypeMismatchCastWithoutSuggestion,
        "{e}"
    );
}
