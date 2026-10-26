// SPDX-License-Identifier: Apache-2.0
//! The JSON functions with options built as a map literal, as the Python
//! wrappers build them: the options reach the registry and take effect.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use futures::executor::block_on;
use krishiv_common::error::ErrorClass;
use krishiv_session::Session;
use krishiv_session::functions::{call_function, col, lit, map_literal};

fn session() -> Session {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("j", ArrowType::Utf8, true)]));
    let batch = |rows: &[Option<&str>]| {
        let json: ArrayRef = Arc::new(StringArray::from(rows.to_vec()));
        RecordBatch::try_new(Arc::clone(&schema), vec![json]).unwrap()
    };
    s.register_batches(
        "v",
        Arc::clone(&schema),
        vec![batch(&[Some("{\"a\":1}"), Some("{\"a\":2}")])],
    )
    .unwrap();
    s.register_batches(
        "t",
        Arc::clone(&schema),
        vec![batch(&[Some("{\"a\":1}"), Some("not json")])],
    )
    .unwrap();
    s
}

/// The options of one `from_json` call, with two entries (a dropped entry
/// is the old bug).
fn options() -> krishiv_session::Column {
    map_literal(vec![
        ("mode".to_owned(), "FAILFAST".to_owned()),
        ("timestampFormat".to_owned(), "dd/MM/yyyy".to_owned()),
    ])
}

#[test]
fn from_json_reads_options_from_a_map_literal() {
    let s = session();
    // The mode fails the malformed row through the whole stack.
    let df = s.table("t").select([call_function(
        "from_json",
        [col("j"), lit("a INT"), options()],
    )]);
    let err = block_on(df.collect()).unwrap_err();
    assert_eq!(
        err.error_class(),
        ErrorClass::MalformedRecordInParsingWithoutSuggestion
    );
    // The valid rows read their values.
    let df = s.table("v").select([call_function(
        "from_json",
        [col("j"), lit("a INT"), options()],
    )]);
    let batches = block_on(df.collect()).unwrap();
    assert_eq!(batches.len(), 1);
    let st = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::StructArray>()
        .unwrap();
    let a: Vec<Option<i32>> = st.column(0).as_primitive::<Int32Type>().iter().collect();
    assert_eq!(a, [Some(1), Some(2)]);
}

#[test]
fn schema_of_json_reads_options_from_a_map_literal() {
    let s = session();
    let options = map_literal(vec![("primitivesAsString".to_owned(), "true".to_owned())]);
    let df = s
        .table("v")
        .select([call_function("schema_of_json", [lit("{\"a\":1}"), options])]);
    let batches = block_on(df.collect()).unwrap();
    assert_eq!(batches.len(), 1);
    let got: Vec<Option<String>> = match batches[0].column(0).data_type() {
        ArrowType::Utf8View => batches[0]
            .column(0)
            .as_string_view()
            .iter()
            .map(|v| v.map(str::to_owned))
            .collect(),
        _ => batches[0]
            .column(0)
            .as_string::<i32>()
            .iter()
            .map(|v| v.map(str::to_owned))
            .collect(),
    };
    assert_eq!(
        got,
        [
            Some("STRUCT<a: STRING>".to_owned()),
            Some("STRUCT<a: STRING>".to_owned())
        ]
    );
}
