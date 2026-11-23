// SPDX-License-Identifier: Apache-2.0
//! The CSV and JSON readers' `TIME` columns against a real Spark 4.3.0 run
//! (`tests/fixtures/spark-4.3.0/time-file-reads.json`, made by
//! `tests/scripts/gen_time_file_reads.py`): the files the generator writes are
//! rebuilt here from its text list, read with each case's schema and options,
//! and compared row by row (the `TIME` cast to text as Spark shows it).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type, Time64NanosecondType};
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, SchemaRef};
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_kernels::cast::format_time;
use serde_json::Value;

/// The fixture.
pub(crate) const FIXTURE: &str =
    include_str!("../../../../tests/fixtures/spark-4.3.0/time-file-reads.json");

/// The `timeFormat` files of the generator (`fmt/a.csv`, `fmt_json/a.json`).
const FORMATTED: &[&str] = &[
    "12:34:56",
    "12.34.56",
    "1:05 PM",
    "12:34:56.123",
    "12:34:56.123456789",
    "2024-01-01 12:34:56",
    "12:34",
    "1:2:3",
];

/// The parsed fixture.
pub(crate) fn fixture() -> Value {
    serde_json::from_str(FIXTURE).expect("fixture")
}

/// The case named `name`.
pub(crate) fn case<'a>(doc: &'a Value, name: &str) -> &'a Value {
    doc["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("no case {name}"))
}

fn texts(doc: &Value) -> Vec<String> {
    doc["texts"]
        .as_array()
        .expect("texts")
        .iter()
        .map(|t| t.as_str().expect("text").to_owned())
        .collect()
}

/// Python's `json.dumps` of a string (`ensure_ascii`): the generator's JSON
/// lines, so a malformed record's text is the same.
fn py_json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            c if c.is_ascii() => out.push(c),
            c => {
                let mut units = [0_u16; 2];
                for u in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
        }
    }
    out.push('"');
    out
}

/// The CSV file of the text table (`texts/a.csv`).
pub(crate) fn texts_csv(doc: &Value) -> Vec<u8> {
    let mut s = String::from("id,t\n");
    for (i, t) in texts(doc).iter().enumerate() {
        s.push_str(&format!("{i},\"{t}\"\n"));
    }
    s.into_bytes()
}

/// The JSON file of the text table (`texts_json/a.json`).
pub(crate) fn texts_json(doc: &Value) -> Vec<u8> {
    let mut s = String::new();
    for (i, t) in texts(doc).iter().enumerate() {
        s.push_str(&format!("{{\"id\": {i}, \"t\": {}}}\n", py_json_str(t)));
    }
    s.into_bytes()
}

/// The `timeFormat` CSV file.
pub(crate) fn formatted_csv() -> Vec<u8> {
    let mut s = String::from("id,t\n");
    for (i, t) in FORMATTED.iter().enumerate() {
        s.push_str(&format!("{i},{t}\n"));
    }
    s.into_bytes()
}

/// The `timeFormat` JSON file.
pub(crate) fn formatted_json() -> Vec<u8> {
    let mut s = String::new();
    for (i, t) in FORMATTED.iter().enumerate() {
        s.push_str(&format!("{{\"id\":{i},\"t\":\"{t}\"}}\n"));
    }
    s.into_bytes()
}

/// A DDL schema as Arrow. Every `TIME(p)` is `Time64(ns)`, so `TIME(9)` is
/// read as `TIME` (the engine's `TIME` precision above 6 arrives with #981).
pub(crate) fn schema(ddl: &str) -> SchemaRef {
    let ddl = ddl.replace("TIME(9)", "TIME").replace("TIME(0)", "TIME");
    let t = krishiv_types::parse_data_type(&format!("struct<{ddl}>")).expect("ddl");
    let DataType::Struct(fields) = t.to_arrow() else {
        panic!("struct");
    };
    Arc::new(arrow_schema::Schema::new(fields))
}

fn cell(a: &dyn Array, i: usize) -> Option<String> {
    if a.is_null(i) {
        return None;
    }
    Some(match a.data_type() {
        DataType::Int32 => a.as_primitive::<Int32Type>().value(i).to_string(),
        DataType::Int64 => a.as_primitive::<Int64Type>().value(i).to_string(),
        DataType::Time64(_) => format_time(a.as_primitive::<Time64NanosecondType>().value(i)),
        DataType::Utf8View => a.as_string_view().value(i).to_owned(),
        DataType::Utf8 => a.as_string::<i32>().value(i).to_owned(),
        other => panic!("{other}"),
    })
}

/// The rows of `batches` as text, sorted as the generator sorts them.
pub(crate) fn rows(batches: &[RecordBatch]) -> Vec<Vec<Option<String>>> {
    let mut out: Vec<Vec<Option<String>>> = batches
        .iter()
        .flat_map(|b| {
            (0..b.num_rows())
                .map(|i| b.columns().iter().map(|c| cell(c.as_ref(), i)).collect())
                .collect::<Vec<_>>()
        })
        .collect();
    out.sort();
    out
}

/// A `TIME` text with its fraction cut to microseconds: what the engine's
/// `TIME` grammar reads until #981 takes it to nanoseconds.
fn micros(s: &str) -> String {
    match s.split_once('.') {
        Some((clock, frac)) if frac.len() > 6 && frac.bytes().all(|b| b.is_ascii_digit()) => {
            let kept = frac[..6].trim_end_matches('0');
            if kept.is_empty() {
                clock.to_owned()
            } else {
                format!("{clock}.{kept}")
            }
        }
        _ => s.to_owned(),
    }
}

/// The expected rows of `case`; with `until_981`, the `TIME` column (index
/// 1) cut to microseconds.
pub(crate) fn expected(case: &Value, until_981: bool) -> Vec<Vec<Option<String>>> {
    let mut out: Vec<Vec<Option<String>>> = case["rows"]
        .as_array()
        .unwrap_or_else(|| panic!("rows of {}", case["name"]))
        .iter()
        .map(|r| {
            r.as_array()
                .expect("row")
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    v.as_str().map(|s| {
                        if until_981 && i == 1 {
                            micros(s)
                        } else {
                            s.to_owned()
                        }
                    })
                })
                .collect()
        })
        .collect();
    out.sort();
    out
}

/// The error classes of `e` and its causes.
pub(crate) fn classes(e: &KrishivError) -> Vec<ErrorClass> {
    std::iter::once(e)
        .chain(e.causes())
        .map(KrishivError::error_class)
        .collect()
}

/// A reader under test: the file, the case's options, the table schema.
pub(crate) type Reader<'a> =
    &'a dyn Fn(&[u8], &Value, SchemaRef) -> Result<Vec<RecordBatch>, KrishivError>;

/// Runs the reader cases of `fmt` (`csv` or `json`) through `read`, with
/// `options` building a case's options.
pub(crate) fn check_reads(fmt: &str, read: Reader<'_>) {
    let doc = fixture();
    let texts = if fmt == "csv" {
        texts_csv(&doc)
    } else {
        texts_json(&doc)
    };
    let mut checked = 0;
    for ty in ["TIME", "TIME(0)", "TIME(3)", "TIME(9)"] {
        let c = case(&doc, &format!("texts {fmt} {ty} PERMISSIVE"));
        let ddl = format!("id INT, t {ty}, _corrupt_record STRING");
        let got = rows(&read(&texts, &serde_json::json!({}), schema(&ddl)).expect(&ddl));
        assert_eq!(got, expected(c, true), "{fmt} {ty} PERMISSIVE");
        checked += 1;
    }
    let c = case(&doc, &format!("texts {fmt} TIME(3) no corrupt column"));
    let got =
        rows(&read(&texts, &serde_json::json!({}), schema("id INT, t TIME(3)")).expect("read"));
    assert_eq!(got, expected(c, true), "{fmt} no corrupt column");
    let c = case(&doc, &format!("texts {fmt} TIME(3) DROPMALFORMED"));
    let got = rows(
        &read(
            &texts,
            &serde_json::json!({"mode": "DROPMALFORMED"}),
            schema("id INT, t TIME(3)"),
        )
        .expect("read"),
    );
    assert_eq!(got, expected(c, true), "{fmt} DROPMALFORMED");
    let c = case(&doc, &format!("texts {fmt} TIME FAILFAST"));
    let e = read(
        &texts,
        &serde_json::json!({"mode": "FAILFAST"}),
        schema("id INT, t TIME"),
    )
    .expect_err("FAILFAST");
    // Spark: FAILED_READ_FILE.NO_HINT > MALFORMED_RECORD_IN_PARSING.WITHOUT_SUGGESTION >
    // CAST_INVALID_INPUT; the engine names no cast under the record.
    let got = classes(&e);
    let want = c["error_chain"].as_array().expect("chain");
    assert_eq!(
        want[..2],
        [
            "FAILED_READ_FILE.NO_HINT",
            "MALFORMED_RECORD_IN_PARSING.WITHOUT_SUGGESTION"
        ]
    );
    assert_eq!(
        got[..2],
        [
            ErrorClass::FailedReadFileNoHint,
            ErrorClass::MalformedRecordInParsingWithoutSuggestion
        ],
        "{fmt} FAILFAST"
    );
    checked += 3;
    // `timeFormat`: an exact pattern, no fallback to the `TIME` grammar.
    let formatted = if fmt == "csv" {
        formatted_csv()
    } else {
        formatted_json()
    };
    for pattern in [
        "HH:mm:ss",
        "HH.mm.ss",
        "h:mm a",
        "HH:mm:ss.SSS",
        "HH:mm:ss.SSSSSSSSS",
        "yyyy-MM-dd HH:mm:ss",
        "H:m:s",
    ] {
        let c = case(&doc, &format!("timeFormat {fmt} {pattern}"));
        let ddl = "id INT, t TIME(9), _corrupt_record STRING";
        let got = rows(
            &read(
                &formatted,
                &serde_json::json!({"timeFormat": pattern}),
                schema(ddl),
            )
            .expect(pattern),
        );
        assert_eq!(got, expected(c, true), "{fmt} timeFormat {pattern}");
        checked += 1;
    }
    // A pattern Spark's formatter cannot build fails each value, not the read.
    let c = case(&doc, &format!("timeFormat {fmt} invalid pattern"));
    let ddl = "id INT, t TIME, _corrupt_record STRING";
    let got = rows(
        &read(
            &formatted,
            &serde_json::json!({"timeFormat": "qqq HH"}),
            schema(ddl),
        )
        .expect("invalid pattern"),
    );
    assert_eq!(got, expected(c, false), "{fmt} timeFormat invalid pattern");
    let c = case(&doc, &format!("timeFormat {fmt} FAILFAST"));
    let e = read(
        &formatted,
        &serde_json::json!({"timeFormat": "HH:mm:ss", "mode": "FAILFAST"}),
        schema("id INT, t TIME"),
    )
    .expect_err("timeFormat FAILFAST");
    assert_eq!(c["error_chain"][0], "FAILED_READ_FILE.NO_HINT");
    assert_eq!(
        classes(&e)[..2],
        [
            ErrorClass::FailedReadFileNoHint,
            ErrorClass::MalformedRecordInParsingWithoutSuggestion
        ],
        "{fmt} timeFormat FAILFAST"
    );
    checked += 2;
    assert_eq!(checked, 16);
}

/// The inference files of the generator (`infer/a.csv`, `infer_dots/a.csv`,
/// `infer_mixed/a.csv`, `infer_json/a.json`).
pub(crate) const INFER_CSV: &str = "id,t\n0,12:34:56\n1,01:02:03.5\n2,\n";
/// See [`INFER_CSV`].
pub(crate) const INFER_DOTS_CSV: &str = "id,t\n0,12.34.56\n1,01.02.03\n";
/// See [`INFER_CSV`].
pub(crate) const INFER_MIXED_CSV: &str = "id,t\n0,12:34:56\n1,7\n";
/// See [`INFER_CSV`].
pub(crate) const INFER_JSON: &str =
    "{\"id\":0,\"t\":\"12:34:56\"}\n{\"id\":1,\"t\":\"01:02:03.5\"}\n{\"id\":2,\"t\":null}\n";

/// Spark's simple name of column `t`'s inferred Arrow type.
pub(crate) fn inferred_t(schema: &SchemaRef) -> String {
    let f = schema.field_with_name("t").expect("column t");
    match f.data_type() {
        DataType::Time64(_) => "time(6)".to_owned(),
        DataType::Timestamp(..) => "timestamp".to_owned(),
        DataType::Utf8View | DataType::Utf8 => "string".to_owned(),
        other => other.to_string(),
    }
}

/// The type Spark inferred for column `t` in `case`.
pub(crate) fn spark_t(case: &Value) -> String {
    case["schema"]
        .as_array()
        .expect("schema")
        .iter()
        .find(|f| f["name"] == "t")
        .and_then(|f| f["type"].as_str())
        .expect("t")
        .to_owned()
}

/// A case's options with the `TIME` gate of its `conf`.
pub(crate) fn with_gate(case: &Value, mut opts: Value) -> Value {
    let gate = case["conf"][crate::text::TIME_TYPE_CONF].clone();
    opts[crate::text::TIME_TYPE_CONF] = gate;
    opts
}

/// `opts` with the `TIME` gate on (a session with `spark.sql.timeType.enabled`).
pub(crate) fn gate_on(mut opts: Value) -> Value {
    opts[crate::text::TIME_TYPE_CONF] = Value::from("true");
    opts
}
