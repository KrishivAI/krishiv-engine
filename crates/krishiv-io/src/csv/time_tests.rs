// SPDX-License-Identifier: Apache-2.0
//! `TIME` columns in the CSV reader and writer against Spark 4.3.0
//! (`crate::text::time_golden`).

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, Time64NanosecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};

use krishiv_common::error::ErrorClass;

use super::tests::{encode, infer, options, read};
use crate::request::ScanRequest;
use crate::text::time_golden::{
    INFER_CSV, INFER_DOTS_CSV, INFER_MIXED_CSV, case, check_reads, expected, fixture, inferred_t,
    rows, schema, spark_t, with_gate,
};

#[test]
fn time_reads_match_spark() {
    check_reads("csv", &|bytes, opts, schema| {
        let mut o = opts.clone();
        o["header"] = serde_json::json!("true");
        read(bytes, &options(&o), &ScanRequest::all(schema))
    });
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
        ("write csv", serde_json::json!({})),
        (
            "write csv timeFormat",
            serde_json::json!({"timeFormat": "HH.mm.ss"}),
        ),
        (
            "write csv timeFormat fraction",
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
    // A pattern that cannot format a time fails the write at the first value.
    for (name, pattern) in [
        ("write csv timeFormat invalid", "qqq HH"),
        ("write csv timeFormat date fields", "yyyy-MM-dd HH:mm"),
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
    let text = String::from_utf8(
        encode(
            &nulls_only(),
            &options(&crate::text::time_golden::gate_on(
                serde_json::json!({"timeFormat": "qqq HH"}),
            )),
        )
        .expect("nulls"),
    )
    .expect("utf-8");
    assert_eq!(
        text.trim_end(),
        written_line(&doc, "write csv timeFormat invalid nulls only")
    );
}

/// One row, `id` 1 and a NULL `TIME`.
fn nulls_only() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("t", DataType::Time64(TimeUnit::Nanosecond), true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Time64NanosecondArray::from(vec![None::<i64>])),
        ],
    )
    .expect("batch")
}

/// An empty field, `nullValue` and `emptyValue` (a quoted empty field) on a
/// `TIME` column.
#[test]
fn time_null_and_empty_values_match_spark() {
    let doc = fixture();
    let file = b"id,t\n0,\n1,bad\n2,12:00\n3,\"\"\n";
    for (name, opts) in [
        ("csv nulls default", serde_json::json!({"header": "true"})),
        (
            "csv nullValue",
            serde_json::json!({"header": "true", "nullValue": "bad"}),
        ),
        (
            "csv emptyValue",
            serde_json::json!({"header": "true", "emptyValue": "12:00"}),
        ),
    ] {
        let got = rows(
            &read(
                file,
                &options(&opts),
                &ScanRequest::all(schema("id INT, t TIME")),
            )
            .expect(name),
        );
        assert_eq!(got, expected(case(&doc, name), false), "{name}");
    }
}

/// Inference gives `TIME(6)` with the gate on and `preferDate`, where every
/// value is a time (Spark 4.3.0 `CSVInferSchema.tryParseTime`).
#[test]
fn time_inference_matches_spark() {
    let doc = fixture();
    for (name, file, extra) in [
        ("infer csv single shape", INFER_CSV, serde_json::json!({})),
        ("infer csv gate off", INFER_CSV, serde_json::json!({})),
        (
            "infer csv preferDate false",
            INFER_CSV,
            serde_json::json!({"preferDate": "false"}),
        ),
        (
            "infer csv timeFormat",
            INFER_DOTS_CSV,
            serde_json::json!({"timeFormat": "HH.mm.ss"}),
        ),
        (
            "infer csv timeFormat default texts",
            INFER_CSV,
            serde_json::json!({"timeFormat": "HH.mm.ss"}),
        ),
        ("infer csv mixed", INFER_MIXED_CSV, serde_json::json!({})),
    ] {
        let c = case(&doc, name);
        let mut opts = with_gate(c, extra);
        opts["header"] = serde_json::json!("true");
        opts["inferSchema"] = serde_json::json!("true");
        let got = infer(file.as_bytes(), &options(&opts)).expect(name);
        assert_eq!(inferred_t(&got), spark_t(c), "{name}");
        if spark_t(c) == "time(6)" {
            let rows =
                rows(&read(file.as_bytes(), &options(&opts), &ScanRequest::all(got)).expect(name));
            assert_eq!(rows, expected(c, false), "{name} rows");
        }
    }
}

/// With the gate off a `TIME` column is not written (Spark 4.3.0:
/// `UNSUPPORTED_TIME_TYPE`, fixture case `gate off csv write`).
#[test]
fn a_time_column_needs_the_gate_to_be_written() {
    let doc = fixture();
    assert_eq!(
        case(&doc, "gate off csv write")["error_class"],
        "UNSUPPORTED_TIME_TYPE"
    );
    let e = encode(&written(), &options(&serde_json::json!({}))).expect_err("gate off");
    assert_eq!(e.error_class(), ErrorClass::UnsupportedTimeType, "{e}");
}
