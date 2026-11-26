// SPDX-License-Identifier: Apache-2.0
//! `date_format`, `to_char` and `to_varchar` of a `TIME` (Spark 4.3). Values, names and errors are
//! those of a local Spark 4.3.0 run; Spark 4.2.0 refuses every one of these calls.
#![forbid(unsafe_code)]
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

fn session(profile: SparkProfile) -> Session {
    let s = Session::builder().profile(profile).build().unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    s
}

/// The rows of the first column of `query`, and that column's name.
fn run(s: &Session, query: &str) -> Result<(String, Vec<Option<String>>), KrishivError> {
    let df = block_on(s.sql(query))?;
    let batches = block_on(df.collect())?;
    let name = batches[0].schema().field(0).name().clone();
    let mut rows = Vec::new();
    for b in &batches {
        let c = b.column(0);
        for i in 0..c.len() {
            rows.push((!c.is_null(i)).then(|| match c.data_type() {
                arrow_schema::DataType::Utf8 => c.as_string::<i32>().value(i).to_owned(),
                arrow_schema::DataType::Utf8View => c.as_string_view().value(i).to_owned(),
                other => panic!("unexpected result type {other}"),
            }));
        }
    }
    Ok((name, rows))
}

#[test]
fn a_time_formats_with_its_time_of_day_fields() {
    let s = session(SparkProfile::V4_3);
    for (pattern, want) in [
        ("HH:mm:ss.SSSSSS", "13:04:05.123456"),
        ("hh:mm a", "01:04 PM"),
        ("h:m:s S", "1:4:5 1"),
        ("H", "13"),
        ("K k", "1 13"),
        ("HH:mm:ss.SSSSSSSSS", "13:04:05.123456000"),
        ("", ""),
        ("\\'T\\'HH", "T13"),
        ("[HH]", "13"),
        ("HH[ yyyy]", "13"),
        ("HH[:mm]", "13:04"),
    ] {
        let q = format!("SELECT date_format(TIME'13:04:05.123456', '{pattern}')");
        let (_, rows) = run(&s, &q).unwrap_or_else(|e| panic!("{q}: {e}"));
        assert_eq!(rows, [Some(want.to_owned())], "{q}");
    }
    let (_, rows) = run(
        &s,
        "SELECT date_format(CAST(TIME'01:02:03.5' AS TIME(0)), 'ss.S')",
    )
    .unwrap();
    assert_eq!(rows, [Some("03.0".to_owned())]);
}

#[test]
fn to_char_and_to_varchar_are_date_format() {
    let s = session(SparkProfile::V4_3);
    for f in ["to_char", "to_varchar"] {
        let (name, rows) = run(&s, &format!("SELECT {f}(TIME'01:02:03', 'HH:mm')")).unwrap();
        assert_eq!(name, "date_format(TIME '01:02:03', HH:mm)");
        assert_eq!(rows, [Some("01:02".to_owned())]);
    }
}

#[test]
fn a_pattern_field_a_time_lacks_is_an_invalid_parameter() {
    let s = session(SparkProfile::V4_3);
    for pattern in [
        "yyyy", "MM", "MMM", "L", "dd", "E", "D", "Q", "q", "F", "G", "z", "VV", "XXX", "O", "x",
        "Z",
    ] {
        for q in [
            format!("SELECT date_format(TIME'13:04:05', '{pattern}')"),
            format!("SELECT to_char(x, p) FROM VALUES (TIME'13:04:05', '{pattern}') t(x, p)"),
        ] {
            let e = run(&s, &q).unwrap_err();
            assert_eq!(
                e.error_class(),
                ErrorClass::InvalidParameterValuePattern,
                "{q}: {e}"
            );
            let message = e.to_string();
            for part in ["`format`", "`date_format`", &format!("'{pattern}'")] {
                assert!(message.contains(part), "{q}: {message}");
            }
        }
    }
}

#[test]
fn the_legacy_parser_policy_does_not_reach_a_time() {
    let s = session(SparkProfile::V4_3);
    block_on(s.sql("SET spark.sql.legacy.timeParserPolicy = LEGACY")).unwrap();
    let (_, rows) = run(&s, "SELECT date_format(TIME'13:02:03', 'HH:mm a')").unwrap();
    assert_eq!(rows, [Some("13:02 PM".to_owned())]);
    let e = run(&s, "SELECT date_format(TIME'13:02:03', 'yyyy')").unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::InvalidParameterValuePattern,
        "{e}"
    );
}

#[test]
fn nulls_are_null_before_the_pattern_is_read() {
    let s = session(SparkProfile::V4_3);
    for q in [
        "SELECT date_format(CAST(NULL AS TIME), 'yyyy')",
        "SELECT date_format(TIME'01:02:03', CAST(NULL AS STRING))",
    ] {
        assert_eq!(run(&s, q).unwrap().1, [None], "{q}");
    }
    let (_, rows) = run(
        &s,
        "SELECT date_format(x, p) FROM VALUES (TIME'01:02:03', 'HH'), (TIME'01:02:03', NULL), \
         (NULL, 'yyyy') t(x, p)",
    )
    .unwrap();
    assert_eq!(rows, [Some("01".to_owned()), None, None]);
}

#[test]
fn spark_4_2_refuses_to_cast_the_time_to_a_timestamp() {
    let s = session(SparkProfile::V4_2);
    for q in [
        "SELECT date_format(TIME'01:02:03', 'HH')",
        "SELECT to_char(TIME'01:02:03', 'HH')",
        "SELECT to_varchar(CAST(NULL AS TIME(3)), 'HH')",
    ] {
        let e = run(&s, q).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::DatatypeMismatchCastWithoutSuggestion,
            "{q}: {e}"
        );
    }
}
