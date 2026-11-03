// SPDX-License-Identifier: Apache-2.0
//! The dictionary evaluator: eligibility from the encoding statistics, the
//! per-entry mask, and the mask over a dictionary decoded from a real file.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::{ArrowReaderOptions, ParquetRecordBatchReaderBuilder};
use parquet::file::metadata::ParquetMetaDataReader;
use parquet::file::properties::WriterProperties;

use super::*;
use crate::request::{CmpOp, Collation};

fn file(strings: &[&str], props: WriterProperties) -> Bytes {
    let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(strings.to_vec())) as ArrayRef],
    )
    .unwrap();
    let mut out = Vec::new();
    {
        let mut w = ArrowWriter::try_new(&mut out, schema, Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }
    Bytes::from(out)
}

fn column(bytes: &Bytes) -> ColumnChunkMetaData {
    let meta = ParquetMetaDataReader::new()
        .parse_and_finish(bytes)
        .unwrap();
    meta.row_group(0).column(0).clone()
}

#[test]
fn a_chunk_of_dictionary_pages_is_eligible() {
    let few = file(
        &["a", "b", "a", "c", "b", "a"],
        WriterProperties::builder().build(),
    );
    assert!(all_dictionary(&column(&few)));
    // Dictionary encoding off: plain pages only.
    let plain = file(
        &["a", "b", "a"],
        WriterProperties::builder()
            .set_dictionary_enabled(false)
            .build(),
    );
    assert!(!all_dictionary(&column(&plain)));
}

#[test]
fn a_chunk_that_fell_back_to_plain_pages_is_not_eligible() {
    // A tiny dictionary page limit: the writer starts with a dictionary and
    // falls back to plain pages once it is full.
    let many: Vec<String> = (0..2_000).map(|i| format!("value-{i:06}")).collect();
    let refs: Vec<&str> = many.iter().map(String::as_str).collect();
    let props = WriterProperties::builder()
        .set_dictionary_page_size_limit(64)
        .set_data_page_row_count_limit(100)
        .set_write_batch_size(100)
        .build();
    assert!(!all_dictionary(&column(&file(&refs, props))));
}

fn lit(s: &str) -> ScalarValue {
    ScalarValue::Utf8(s.to_owned())
}

#[test]
fn the_mask_drops_the_entries_a_conjunct_rejects() {
    let v = ColumnPath::new("v");
    let entries = [lit("apple"), lit("banana"), lit("cherry")];
    let eq = PushdownExpr::eq(v.clone(), lit("banana"));
    assert_eq!(passes(&[&eq], &v, &entries), [false, true, false]);
    let ge = PushdownExpr::cmp(v.clone(), CmpOp::GtEq, lit("banana"));
    assert_eq!(passes(&[&ge], &v, &entries), [false, true, true]);
    // Conjuncts combine: every one must hold.
    let lt = PushdownExpr::cmp(v.clone(), CmpOp::Lt, lit("cherry"));
    assert_eq!(passes(&[&ge, &lt], &v, &entries), [false, true, false]);
    let any_of = PushdownExpr::In {
        column: v.clone(),
        values: vec![lit("apple"), lit("cherry")],
        collation: Collation::Utf8Binary,
    };
    assert_eq!(passes(&[&any_of], &v, &entries), [true, false, true]);
    let not_b = PushdownExpr::Not(Box::new(PushdownExpr::eq(v.clone(), lit("banana"))));
    assert_eq!(passes(&[&not_b], &v, &entries), [true, false, true]);
}

#[test]
fn what_cannot_be_decided_passes() {
    let x = ColumnPath::new("x");
    // A NaN entry has no comparable bound: it passes whatever the predicate.
    let entries = [ScalarValue::Float64(1.0), ScalarValue::Float64(f64::NAN)];
    let gt = PushdownExpr::cmp(x.clone(), CmpOp::Gt, ScalarValue::Float64(5.0));
    assert_eq!(passes(&[&gt], &x, &entries), [false, true]);
    // A collation other than binary is not decided by bytes.
    let v = ColumnPath::new("v");
    let ci = PushdownExpr::Cmp {
        column: v.clone(),
        op: CmpOp::Eq,
        value: lit("BANANA"),
        collation: Collation::Other("UTF8_LCASE".into()),
    };
    assert_eq!(passes(&[&ci], &v, &[lit("banana")]), [true]);
}

/// The mask over the dictionary of a real file: the entries are read back
/// through arrow-rs as the dictionary of a dictionary-typed column.
#[test]
fn the_mask_over_a_decoded_dictionary() {
    let bytes = file(
        &["pear", "apple", "pear", "fig", "apple", "pear"],
        WriterProperties::builder().build(),
    );
    assert!(all_dictionary(&column(&bytes)));
    let hint = Arc::new(Schema::new(vec![Field::new(
        "v",
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        false,
    )]));
    let reader = ParquetRecordBatchReaderBuilder::try_new_with_options(
        bytes,
        ArrowReaderOptions::new().with_schema(hint),
    )
    .unwrap()
    .build()
    .unwrap();
    let batch = reader.map(Result::unwrap).next().unwrap();
    let dict = batch.column(0).as_dictionary::<Int32Type>();
    let values = dict.values().as_string::<i32>();
    let entries: Vec<ScalarValue> = (0..values.len()).map(|i| lit(values.value(i))).collect();
    let v = ColumnPath::new("v");
    let eq = PushdownExpr::eq(v.clone(), lit("fig"));
    let mask = passes(&[&eq], &v, &entries);
    assert_eq!(mask.iter().filter(|p| **p).count(), 1);
    // The rows that pass are exactly the rows holding `fig`.
    let keys = dict.keys();
    let code = |i: usize| usize::try_from(keys.value(i)).unwrap();
    let kept: Vec<&str> = (0..keys.len())
        .filter(|i| mask[code(*i)])
        .map(|i| values.value(code(i)))
        .collect();
    assert_eq!(kept, ["fig"]);
}
