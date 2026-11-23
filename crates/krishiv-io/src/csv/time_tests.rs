// SPDX-License-Identifier: Apache-2.0
//! `TIME` columns in the CSV reader and writer against Spark 4.3.0
//! (`crate::text::time_golden`).

use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, Time64NanosecondArray};
use arrow_schema::{DataType, Field, Schema, TimeUnit};

use super::tests::{encode, options, read};
use crate::request::ScanRequest;
use crate::text::time_golden::{case, check_reads, fixture};

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
        let text =
            String::from_utf8(encode(&written(), &options(&opts)).expect(name)).expect("utf-8");
        assert_eq!(text.trim_end(), written_line(&doc, name), "{name}");
    }
}
