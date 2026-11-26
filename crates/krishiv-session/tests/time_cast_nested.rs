// SPDX-License-Identifier: Apache-2.0
//! TIME inside nested values, long-fraction literals and `listagg` over a
//! TIME cast, against what Spark 4.2.0 gave (local-mode run, TIME enabled).
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

fn text(sql: &str) -> String {
    let s = Session::builder()
        .profile(SparkProfile::V4_2)
        .build()
        .unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    let df = block_on(s.sql(&format!("SELECT CAST(({sql}) AS STRING)"))).unwrap();
    let b = block_on(df.collect()).unwrap();
    let c = &b[0].columns()[0];
    if c.is_null(0) {
        "NULL".to_owned()
    } else {
        c.as_string_view().value(0).to_owned()
    }
}

#[test]
fn nested_time_casts_drop_digits_below_the_precision() {
    for (sql, want) in [
        (
            "CAST(array(TIME '12:34:56.789123') AS array<time(3)>)",
            "[12:34:56.789]",
        ),
        (
            "CAST(named_struct('a', TIME '12:34:56.789123') AS struct<a: time(3)>)",
            "{12:34:56.789}",
        ),
    ] {
        assert_eq!(text(sql), want, "{sql}");
    }
}

#[test]
fn a_time_literal_with_a_long_fraction_truncates_to_microseconds() {
    for sql in [
        "TIME '12:34:56.12345678901'",
        "TIME '12:34:56.1234567890123'",
    ] {
        assert_eq!(text(sql), "12:34:56.123456", "{sql}");
    }
}

/// Spark accepts `listagg(DISTINCT CAST(t AS STRING)) ... ORDER BY t` for a
/// TIME (`01:00:0002:00:00`). The analyzer must too; ordering by a TIME key
/// is not built yet, which is a different error.
#[test]
fn listagg_distinct_over_a_time_cast_passes_the_analyzer() {
    let sql = "SELECT listagg(DISTINCT CAST(t AS STRING)) WITHIN GROUP (ORDER BY t) \
               FROM VALUES (TIME '01:00:00'), (TIME '02:00:00') AS v(t)";
    let s = Session::builder()
        .profile(SparkProfile::V4_2)
        .build()
        .unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    let df = block_on(s.sql(sql)).unwrap();
    let e = block_on(df.collect()).unwrap_err();
    assert_eq!(
        e.error_class(),
        krishiv_common::error::ErrorClass::KrishivUnsupportedFeature
    );
}

/// Spark types a TIME cast to `tinyint`/`smallint` as nullable (the seconds
/// may overflow, NULL with ANSI off) and `int`/`bigint`/`time(p)` as not; a
/// map key that may become NULL is refused.
#[test]
fn narrowing_time_casts_are_nullable() {
    let s = Session::builder()
        .profile(SparkProfile::V4_2)
        .build()
        .unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    block_on(s.sql("SET spark.sql.ansi.enabled = false")).unwrap();
    for (to, nullable) in [
        ("tinyint", true),
        ("smallint", true),
        ("int", false),
        ("bigint", false),
        ("time(3)", false),
    ] {
        let df = block_on(s.sql(&format!(
            "SELECT CAST(t AS {to}) c FROM (SELECT TIME '12:00:00' t)"
        )))
        .unwrap();
        let schema = block_on(df.schema()).unwrap();
        assert_eq!(schema.field(0).is_nullable(), nullable, "{to}");
    }
    let e =
        block_on(s.sql("SELECT CAST(map(TIME '23:59:59', 1) AS map<tinyint, int>)")).unwrap_err();
    assert_ne!(
        e.error_class(),
        krishiv_common::error::ErrorClass::InternalError,
        "{e}"
    );
}

/// Spark 4.3 casts timestamps to `TIME` inside arrays, structs and maps element by element
/// (Spark 4.3.0 local run, session zone UTC), and a `TIME` array to `array<timestamp_ntz>`;
/// Spark 4.2 refuses both.
#[test]
fn nested_timestamps_cast_to_time_under_spark_4_3() {
    let s = Session::builder()
        .profile(SparkProfile::V4_3)
        .build()
        .unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    block_on(s.sql("SET spark.sql.session.timeZone = UTC")).unwrap();
    let one = |sql: &str| -> String {
        let df = block_on(s.sql(sql)).unwrap();
        let b = block_on(df.collect()).unwrap();
        b[0].columns()[0].as_string_view().value(0).to_owned()
    };
    for (sql, want) in [
        (
            "SELECT CAST(CAST(array(TIMESTAMP_NTZ'2020-05-17 12:34:56.789012', NULL) AS array<time(3)>) AS STRING)",
            "[12:34:56.789, null]",
        ),
        (
            "SELECT CAST(CAST(named_struct('a', TIMESTAMP'2020-05-17 12:34:56.789012') AS struct<a: time(6)>) AS STRING)",
            "{12:34:56.789012}",
        ),
        (
            "SELECT CAST(CAST(map('k', TIMESTAMP_NTZ'1969-12-31 23:59:59.5') AS map<string, time(1)>) AS STRING)",
            "{k -> 23:59:59.5}",
        ),
        (
            "SELECT typeof(CAST(array(TIME'01:02:03') AS array<timestamp_ntz>))",
            "array<timestamp_ntz>",
        ),
    ] {
        assert_eq!(one(sql), want, "{sql}");
    }
    let s42 = Session::builder()
        .profile(SparkProfile::V4_2)
        .build()
        .unwrap();
    block_on(s42.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    let e = block_on(s42.sql("SELECT CAST(array(TIME'01:02:03') AS array<timestamp_ntz>)"))
        .err()
        .map(|e| e.error_class().name().to_owned());
    assert_eq!(
        e.as_deref(),
        Some("DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION")
    );
}

/// A refused implicit TIME cast is reported before an ordinary argument
/// mismatch, wherever the arguments stand: Spark inserts the cast, keeps the
/// mismatched argument as it is, and CheckAnalysis (`foreachUp`) reaches the
/// child cast before the call. Spark 4.2.0 and 4.3.0 runs, ANSI on and off.
/// (`date_add`/`date_sub` take no such cast in Spark: queued separately.)
#[test]
fn a_refused_time_cast_is_reported_before_an_earlier_mismatch() {
    const CAST: &str = "DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION";
    const UNEXPECTED: &str = "DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE";
    for (profile, to_timestamp) in [(SparkProfile::V4_2, CAST), (SparkProfile::V4_3, UNEXPECTED)] {
        for ansi in ["true", "false"] {
            let s = Session::builder().profile(profile).build().unwrap();
            block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
            block_on(s.sql(&format!("SET spark.sql.ansi.enabled = {ansi}"))).unwrap();
            for (sql, want) in [
                ("SELECT datediff(array(1), TIME'12:00:00')", CAST),
                ("SELECT datediff(TIME'12:00:00', array(1))", CAST),
                ("SELECT add_months(array(1), TIME'12:00:00')", UNEXPECTED),
                ("SELECT next_day(array(1), TIME'12:00:00')", UNEXPECTED),
                (
                    "SELECT months_between(array(1), TIME'12:00:00')",
                    to_timestamp,
                ),
                (
                    "SELECT months_between(TIME'12:00:00', array(1))",
                    to_timestamp,
                ),
                ("SELECT date_format(TIME'12:00:00', array(1))", to_timestamp),
                ("SELECT datediff(TIME'12:00:00', TIME'13:00:00')", CAST),
            ] {
                let got = match block_on(s.sql(sql)) {
                    Ok(df) => block_on(df.collect())
                        .err()
                        .map(|e| e.error_class().name().to_owned()),
                    Err(e) => Some(e.error_class().name().to_owned()),
                };
                assert_eq!(got.as_deref(), Some(want), "{profile:?} ansi={ansi}: {sql}");
            }
        }
    }
}
