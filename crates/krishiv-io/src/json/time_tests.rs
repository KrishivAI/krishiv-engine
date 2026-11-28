// SPDX-License-Identifier: Apache-2.0
//! `TIME` columns in the JSON reader and writer against Spark 4.3.0
//! (`crate::text::time_golden`).

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, Time64NanosecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};

use krishiv_common::error::ErrorClass;

use super::tests::{encode, infer, options, read};
use crate::request::ScanRequest;
use crate::text::time_golden::{
    INFER_JSON, case, check_reads, expected, fixture, inferred_t, rows, schema, spark_t, with_gate,
};

#[test]
fn time_reads_match_spark() {
    check_reads("json", &|bytes, opts, schema| {
        read(bytes, &options(opts), &ScanRequest::all(schema))
    });
}

/// A `TIME` field reads only a JSON string: a number, a boolean, an object or
/// an array makes the record malformed; `null` and a missing field are NULL.
#[test]
fn time_reads_only_strings() {
    let doc = fixture();
    let file = concat!(
        r#"{"id":0,"t":45296}"#,
        "\n",
        r#"{"id":1,"t":true}"#,
        "\n",
        r#"{"id":2,"t":null}"#,
        "\n",
        r#"{"id":3,"t":12.5}"#,
        "\n",
        r#"{"id":4}"#,
        "\n",
        r#"{"id":5,"t":{"h":1}}"#,
        "\n",
        r#"{"id":6,"t":["12:00"]}"#,
        "\n",
    );
    let ddl = "id INT, t TIME, _corrupt_record STRING";
    let got = rows(
        &read(
            file.as_bytes(),
            &options(&serde_json::json!({})),
            &ScanRequest::all(schema(ddl)),
        )
        .expect("read"),
    );
    assert_eq!(
        got,
        crate::text::time_golden::expected(case(&doc, "json tokens TIME"))
    );
}

/// The generator's write query: `id`, `TIME(9)` `12:34:56.123456789`, `TIME(3)`
/// `12:34:56.120`, `TIME(6)` `01:02:03`, `TIME(0)` midnight and a NULL `TIME`.
fn written() -> RecordBatch {
    let t = || Field::new("t", DataType::Time64(TimeUnit::Nanosecond), true);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("t9", t().data_type().clone(), true),
        Field::new("t3", t().data_type().clone(), true),
        Field::new("t6", t().data_type().clone(), true),
        Field::new("t0", t().data_type().clone(), true),
        Field::new("tn", t().data_type().clone(), true),
    ]));
    let time = |v: Option<i64>| -> ArrayRef { Arc::new(Time64NanosecondArray::from(vec![v])) };
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1])),
            time(Some(45_296_123_456_789)),
            time(Some(45_296_120_000_000)),
            time(Some(3_723_000_000_000)),
            time(Some(0)),
            time(None),
        ],
    )
    .expect("batch")
}

/// The written text of a write case (one line).
fn written_line(doc: &serde_json::Value, name: &str) -> String {
    case(doc, name)["rows"][0][0]
        .as_str()
        .expect("line")
        .to_owned()
}

#[test]
fn time_writes_match_spark() {
    let doc = fixture();
    for (name, opts) in [
        ("write json", serde_json::json!({})),
        (
            "write json timeFormat",
            serde_json::json!({"timeFormat": "HH.mm.ss"}),
        ),
        (
            "write json timeFormat fraction",
            serde_json::json!({"timeFormat": "HH:mm:ss.SSS"}),
        ),
    ] {
        let text = String::from_utf8(
            encode(
                &written(),
                &options(&crate::text::time_golden::gate_on(opts)),
            )
            .expect(name),
        )
        .expect("utf-8");
        assert_eq!(text.trim_end(), written_line(&doc, name), "{name}");
    }
    for (name, pattern) in [
        ("write json timeFormat invalid", "qqq HH"),
        ("write json timeFormat date fields", "yyyy-MM-dd HH:mm"),
    ] {
        assert_eq!(case(&doc, name)["error_class"], "TASK_WRITE_FAILED");
        let e = encode(
            &written(),
            &options(&crate::text::time_golden::gate_on(
                serde_json::json!({"timeFormat": pattern}),
            )),
        )
        .expect_err(name);
        assert_eq!(e.error_class(), ErrorClass::TaskWriteFailed, "{name}: {e}");
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("t", DataType::Time64(TimeUnit::Nanosecond), true),
    ]));
    let nulls = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Time64NanosecondArray::from(vec![None::<i64>])),
        ],
    )
    .expect("batch");
    let text = String::from_utf8(
        encode(
            &nulls,
            &options(&crate::text::time_golden::gate_on(
                serde_json::json!({"timeFormat": "qqq HH"}),
            )),
        )
        .expect("nulls"),
    )
    .expect("utf-8");
    assert_eq!(
        text.trim_end(),
        written_line(&doc, "write json timeFormat invalid nulls only")
    );
}

/// Inference gives `TIME(6)` only under `inferTimestamp` with the gate on
/// (Spark 4.3.0 `JsonInferSchema`, before a timestamp is tried).
#[test]
fn time_inference_matches_spark() {
    let doc = fixture();
    for (name, extra) in [
        ("infer json default", serde_json::json!({})),
        (
            "infer json inferTimestamp",
            serde_json::json!({"inferTimestamp": "true"}),
        ),
        (
            "infer json inferTimestamp gate off",
            serde_json::json!({"inferTimestamp": "true"}),
        ),
    ] {
        let c = case(&doc, name);
        let opts = options(&with_gate(c, extra));
        let got = infer(INFER_JSON.as_bytes(), &opts).expect(name);
        assert_eq!(inferred_t(&got), spark_t(c), "{name}");
        if spark_t(c) == "time(6)" {
            let got_rows =
                rows(&read(INFER_JSON.as_bytes(), &opts, &ScanRequest::all(got)).expect(name));
            assert_eq!(got_rows, expected(c), "{name} rows");
        }
    }
}

/// With the gate off a `TIME` column is not written (fixture case
/// `gate off json write`).
#[test]
fn a_time_column_needs_the_gate_to_be_written() {
    let doc = fixture();
    assert_eq!(
        case(&doc, "gate off json write")["error_class"],
        "UNSUPPORTED_TIME_TYPE"
    );
    let e = encode(&written(), &options(&serde_json::json!({}))).expect_err("gate off");
    assert_eq!(e.error_class(), ErrorClass::UnsupportedTimeType, "{e}");
}
