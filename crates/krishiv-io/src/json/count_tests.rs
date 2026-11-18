// SPDX-License-Identifier: Apache-2.0
//! A JSON read of no columns (`count()`): records are parsed and no value is
//! converted, as Spark's `JacksonParser` with an empty required schema. A
//! `DROPMALFORMED` count drops only the records that are not JSON objects,
//! and a `FAILFAST` count fails only on those. A `multiLine` file is one
//! record, its first value (rows, count, corrupt text and inference). Expected
//! results from a Spark 4.2.0 run of the same files (`spark.read.schema(..)
//! .option("mode", m).json(p)`, local mode, 2026-11-18).

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_common::error::KrishivError;

use super::tests::{infer_named, options, read, read_split, rows_of, schema_string};
use crate::io::ByteRange;
use crate::request::ScanRequest;

/// Objects, a value of the wrong type, a line that is not JSON, a top-level
/// array of two objects, a blank line, a scalar and a nested value where an
/// int is declared.
const MIXED: &str = "{\"id\":1,\"name\":\"a\"}\n{\"id\":\"x\",\"name\":\"b\"}\nnot json\n\
{\"id\":3,\"name\":4}\n{\"id\":4}\n[{\"id\":5},{\"id\":6}]\n\n7\n{\"id\":{\"z\":1}}\n";

/// Only values of the wrong type: every record is JSON.
const TYPES_ONLY: &str = "{\"id\":1,\"name\":\"a\"}\n{\"id\":\"x\",\"name\":\"b\"}\n{\"id\":{\"z\":1}}\n[{\"id\":5},{\"id\":\"y\"}]\n";

/// Top-level scalars and arrays of scalars (an empty array has no rows).
const SCALARS: &str = "{\"id\":1}\n7\n\"s\"\n[1,2]\n[]\n{\"id\":2}\n";

fn schema(names: &[&str]) -> SchemaRef {
    let fields: Vec<Field> = names
        .iter()
        .map(|n| {
            let t = if *n == "id" {
                DataType::Int32
            } else {
                DataType::Utf8View
            };
            Field::new(*n, t, true)
        })
        .collect();
    Arc::new(Schema::new(fields))
}

fn count(text: &str, names: &[&str], extra: &[(&str, &str)]) -> Result<usize, KrishivError> {
    let mut m = serde_json::json!({});
    for (k, v) in extra {
        m[*k] = serde_json::Value::from(*v);
    }
    let none = ScanRequest::new(schema(names), Vec::new());
    let batches = read(text.as_bytes(), &options(&m), &none)?;
    Ok(batches.iter().map(RecordBatch::num_rows).sum())
}

#[test]
fn a_read_of_no_columns_counts_records_as_spark() {
    let both = ["id", "name"];
    let id = ["id"];
    let mode = |m| [("mode", m)];
    assert_eq!(count(MIXED, &both, &[]).expect("permissive"), 9);
    assert_eq!(
        count(MIXED, &both, &mode("DROPMALFORMED")).expect("drop"),
        7
    );
    assert!(count(MIXED, &both, &mode("FAILFAST")).is_err());
    assert_eq!(
        count(TYPES_ONLY, &both, &mode("FAILFAST")).expect("failfast"),
        5
    );
    assert_eq!(
        count(TYPES_ONLY, &both, &mode("DROPMALFORMED")).expect("drop"),
        5
    );
    assert_eq!(count(SCALARS, &id, &[]).expect("permissive"), 5);
    assert_eq!(
        count(SCALARS, &id, &mode("DROPMALFORMED")).expect("drop"),
        2
    );
    assert!(count(SCALARS, &id, &mode("FAILFAST")).is_err());
}

#[test]
fn a_multi_line_read_of_no_columns_counts_the_first_value() {
    // The whole file is one record: Spark counts 1 in every mode, the text
    // after the first object unread.
    for m in ["PERMISSIVE", "DROPMALFORMED", "FAILFAST"] {
        let got = count(MIXED, &["id"], &[("mode", m), ("multiLine", "true")]);
        assert_eq!(got.expect("multiLine"), 1, "{m}");
    }
}

#[test]
fn every_split_counts_each_record_once() {
    let none = ScanRequest::new(schema(&["id", "name"]), Vec::new());
    let o = options(&serde_json::json!({"mode": "DROPMALFORMED"}));
    let size = MIXED.len() as u64;
    for cut in 1..size {
        let rows: usize = [ByteRange::new(0, cut), ByteRange::new(cut, size)]
            .into_iter()
            .map(|s| {
                read_split(MIXED.as_bytes(), &o, &none, s)
                    .expect("split")
                    .iter()
                    .map(RecordBatch::num_rows)
                    .sum::<usize>()
            })
            .sum();
        assert_eq!(rows, 7, "cut {cut}");
    }
}

#[test]
fn a_multi_line_file_is_its_first_value() {
    let multi = options(&serde_json::json!({"multiLine": "true"}));
    let corrupt = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("_corrupt_record", DataType::Utf8View, true),
    ]));
    let req = ScanRequest::all(Arc::clone(&corrupt));
    let rows = |text: &str| rows_of(&read(text.as_bytes(), &multi, &req).expect("read"));
    assert_eq!(
        rows("{\"id\":1}\n{\"id\":2}\n"),
        [vec![Some("1".into()), None]]
    );
    // Text after the first value is not read, even when it is not JSON.
    assert_eq!(
        rows("{\"id\":1} trailing garbage {"),
        [vec![Some("1".into()), None]]
    );
    // A top-level array is still a row per element.
    assert_eq!(
        rows("[{\"id\":5},{\"id\":6}]\n{\"id\":3}\n"),
        [vec![Some("5".into()), None], vec![Some("6".into()), None]]
    );
    // The corrupt text of a malformed first value is the whole file.
    let file = "{\"id\":\"x\"}\n{\"id\":2}\n";
    assert_eq!(rows(file), [vec![None, Some(file.into())]]);
    let file = "not json\n{\"id\":3}\n";
    assert_eq!(rows(file), [vec![None, Some(file.into())]]);
    let drop = options(&serde_json::json!({"multiLine": "true", "mode": "DROPMALFORMED"}));
    assert!(rows_of(&read(file.as_bytes(), &drop, &req).expect("read")).is_empty());
    assert!(
        read(b"", &multi, &req)
            .expect("empty")
            .iter()
            .all(|b| b.num_rows() == 0)
    );
    // Inference reads the first value of each file too.
    let schema = infer_named(b"{\"a\":1}\n{\"b\":2}\n", "t.json", &multi).expect("infer");
    assert_eq!(schema_string(&schema), "struct<a:Int64>");
}
