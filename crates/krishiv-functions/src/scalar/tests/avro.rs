// SPDX-License-Identifier: Apache-2.0
//! The Avro bindings: `to_avro` bytes and `from_avro` values against the
//! Spark goldens in `tests/spark-golden/functions/from_to_avro`.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BinaryArray, Int32Array};
use krishiv_common::error::ErrorClass;

use super::{BindArg, BindOptions, bind_scalar, class, col, lit, strs};

fn binary(v: &[Option<&[u8]>]) -> ArrayRef {
    Arc::new(BinaryArray::from(v.to_vec()))
}

fn hex(a: &ArrayRef) -> Vec<Option<String>> {
    a.as_binary_view()
        .iter()
        .map(|x| x.map(|b| b.iter().map(|x| format!("{x:02X}")).collect()))
        .collect()
}

#[test]
fn to_avro_infers_a_plain_type_for_a_literal_and_a_union_for_a_nullable_column() {
    let one: ArrayRef = Arc::new(Int32Array::from(vec![1]));
    let f = bind_scalar(
        "to_avro",
        &[BindArg::Literal(Arc::clone(&one))],
        &BindOptions::new(),
    )
    .unwrap();
    assert_eq!(
        hex(&f.invoke(&[lit(&one)], 1).unwrap()),
        [Some("02".into())]
    );
    let nullable: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None]));
    let f = bind_scalar(
        "to_avro",
        &[BindArg::Column {
            ty: nullable.data_type().clone(),
            nullable: true,
        }],
        &BindOptions::new(),
    )
    .unwrap();
    assert_eq!(
        hex(&f.invoke(&[col(&nullable)], 2).unwrap()),
        [Some("0002".into()), None]
    );
}

#[test]
fn from_avro_reads_a_literal_schema_and_fails_fast_by_default() {
    let data = binary(&[Some(&[2]), None]);
    let schema = strs(&[Some(r#""int""#)]);
    let f = bind_scalar(
        "from_avro",
        &[
            data.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&schema)),
        ],
        &BindOptions::new(),
    )
    .unwrap();
    let out = f.invoke(&[col(&data), lit(&schema)], 2).unwrap();
    let ints = out.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(ints.value(0), 1);
    assert!(ints.is_null(1));
    let bad = binary(&[Some(&[0x80])]);
    let err = f.invoke(&[col(&bad), lit(&schema)], 1).unwrap_err();
    assert_eq!(
        class(&err),
        ErrorClass::MalformedRecordInParsingWithoutSuggestion
    );
}

#[test]
fn from_avro_refuses_options_it_does_not_implement_and_dropmalformed() {
    let schema = strs(&[Some(r#""int""#)]);
    let data = binary(&[Some(&[2])]);
    let opts = |k: &str, v: &str| {
        let keys: ArrayRef = strs(&[Some(k)]);
        let values: ArrayRef = strs(&[Some(v)]);
        let fields: Vec<arrow_schema::Field> = vec![
            arrow_schema::Field::new("key", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("value", arrow_schema::DataType::Utf8, true),
        ];
        let entries = arrow_array::StructArray::new(fields.into(), vec![keys, values], None);
        let m: ArrayRef = Arc::new(
            arrow_array::MapArray::try_new(
                Arc::new(arrow_schema::Field::new(
                    "entries",
                    entries.data_type().clone(),
                    false,
                )),
                arrow_buffer::OffsetBuffer::new(arrow_buffer::ScalarBuffer::from(vec![0, 1])),
                entries,
                None,
                false,
            )
            .unwrap(),
        );
        m
    };
    let bind = |k: &str, v: &str| {
        bind_scalar(
            "from_avro",
            &[
                data.data_type().clone().into(),
                BindArg::Literal(Arc::clone(&schema)),
                BindArg::Literal(opts(k, v)),
            ],
            &BindOptions::new(),
        )
    };
    assert_eq!(
        class(&bind("mode", "DROPMALFORMED").err().unwrap()),
        ErrorClass::ParseModeUnsupported
    );
    assert!(bind("recursiveFieldMaxDepth", "3").is_err());
    // An unrecognised mode reads as PERMISSIVE, as Spark's does.
    let f = bind("mode", "bogus").unwrap();
    let bad = binary(&[Some(&[0x80])]);
    let out = f
        .invoke(&[col(&bad), lit(&schema), lit(&opts("mode", "bogus"))], 1)
        .unwrap();
    assert!(out.is_null(0));
}
