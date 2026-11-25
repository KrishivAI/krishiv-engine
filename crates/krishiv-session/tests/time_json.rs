// SPDX-License-Identifier: Apache-2.0
//! TIME in `from_json`/`to_json`, beyond the Spark 4.2.0 `json-functions`
//! golden: a nested TIME, a `timeFormat` that drops the date, a pattern miss
//! and a NULL.
#![cfg(test)]

use arrow_array::Array;
use arrow_array::cast::AsArray;
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

fn text(sql: &str) -> Option<String> {
    let s = Session::builder()
        .profile(SparkProfile::V4_2)
        .build()
        .unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    let df = block_on(s.sql(&format!("SELECT CAST(({sql}) AS STRING)"))).unwrap();
    let b = block_on(df.collect()).unwrap();
    let c = &b[0].columns()[0];
    (!c.is_null(0)).then(|| c.as_string_view().value(0).to_owned())
}

#[test]
fn to_json_writes_default_and_formatted_times() {
    assert_eq!(
        text("to_json(named_struct('t', TIME'01:02:03.5'))").as_deref(),
        Some(r#"{"t":"01:02:03.5"}"#)
    );
    assert_eq!(
        text("to_json(array(TIME'01:02:03'), map('timeFormat', 'HH.mm.ss'))").as_deref(),
        Some(r#"["01.02.03"]"#)
    );
}

#[test]
fn from_json_reads_nested_times_and_misses() {
    assert_eq!(
        text("from_json('{\"a\":[\"01:02:03\",\"04:05:06\"]}', 'a array<time>')").as_deref(),
        Some("{[01:02:03, 04:05:06]}")
    );
    assert_eq!(
        text("from_json('{\"t\":\"01:02:03\"}', 't TIME', map('timeFormat', 'HH.mm.ss'))")
            .as_deref(),
        Some("{null}")
    );
    assert_eq!(
        text("from_json('{\"t\":\"01.02.03\"}', 't TIME', map('timeFormat', 'HH.mm.ss'))")
            .as_deref(),
        Some("{01:02:03}")
    );
}

/// Spark 4.2.0 reads a JSON time to microseconds whatever the declared `TIME(p)`: the lane's
/// 4.2.0 run (`time-precision-goldens.json` case `ddl_time3_value`) gives `12:00:00.123456` for
/// `from_json('{"a":"12:00:00.123456789"}', 'a time(3)')`, so nothing cuts to `p`.
#[test]
fn from_json_keeps_microseconds_whatever_the_declared_precision_under_spark_4_2() {
    assert_eq!(
        text("from_json('{\"t\":\"09:08:01.987654\"}', 't TIME(0)')").as_deref(),
        Some("{09:08:01.987654}")
    );
    assert_eq!(
        text("from_json('{\"t\":[\"09:08:01.987654\"]}', 't array<time(2)>')").as_deref(),
        Some("{[09:08:01.987654]}")
    );
    assert_eq!(
        text("from_json('{\"t\":\"09:08:01.987654789\"}', 't TIME')").as_deref(),
        Some("{09:08:01.987654}")
    );
}

/// Spark 4.3.0 (lane review probe `probe-430.out`): the reader keeps every digit it reads,
/// whatever the declared `TIME(p)`; the session's profile reaches the executor's bindings.
#[test]
fn from_json_keeps_nine_digits_under_spark_4_3() {
    let s = Session::builder()
        .profile(SparkProfile::V4_3)
        .build()
        .unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    let df = block_on(
        s.sql("SELECT CAST(from_json('{\"a\":\"12:00:00.123456789\"}', 'a time(3)').a AS STRING)"),
    )
    .unwrap();
    let b = block_on(df.collect()).unwrap();
    assert_eq!(
        b[0].columns()[0].as_string_view().value(0),
        "12:00:00.123456789"
    );
}

/// Spark 4.3.0 (lane probe of 2026-11-24: `error` is `'<type name>'`, `hint` empty, SQLSTATE 42601, message
/// `Syntax error at or near '<type name>'.`): a DDL schema string reports a bad
/// timestamp or time precision as `PARSE_SYNTAX_ERROR` at the type name, since
/// `DataType.parseTypeWithFallback` throws the direct parse's error when the field-list
/// parse fails too. `timestamp(6)` and `timestamp_ltz(6)` are the microsecond type.
#[test]
fn a_ddl_schema_precision_error_is_a_syntax_error_under_spark_4_3() {
    let s = Session::builder()
        .profile(SparkProfile::V4_3)
        .build()
        .unwrap();
    block_on(s.sql("SET spark.sql.timeType.enabled = true")).unwrap();
    let run = |sql: &str| block_on(async { s.sql(sql).await?.collect().await });
    for (ty, near) in [
        ("timestamp_ntz(9)", "timestamp_ntz"),
        ("timestamp(9)", "timestamp"),
        ("timestamp_ntz(3)", "timestamp_ntz"),
        ("time(10)", "time"),
    ] {
        let e = run(&format!("SELECT from_json('{{\"a\":1}}', 'a {ty}')")).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::ParseSyntaxError, "{ty}: {e}");
        assert_eq!(e.sql_state(), Some("42601"), "{ty}");
        assert_eq!(
            e.message(),
            format!("Syntax error at or near '{near}'."),
            "{ty}"
        );
        let params: Vec<(String, String)> = e
            .message_parameters()
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect();
        assert_eq!(
            params,
            [
                ("error".to_owned(), format!("'{near}'")),
                ("hint".to_owned(), String::new())
            ],
            "{ty}"
        );
    }
    for ty in ["timestamp_ltz(6)", "timestamp(6)"] {
        let b = run(&format!(
            "SELECT typeof(from_json('{{\"a\":1}}', 'a {ty}'))"
        ))
        .unwrap();
        assert_eq!(
            b[0].columns()[0].as_string_view().value(0),
            "struct<a:timestamp>",
            "{ty}"
        );
    }
}
