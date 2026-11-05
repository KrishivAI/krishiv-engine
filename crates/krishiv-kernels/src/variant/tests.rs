// SPDX-License-Identifier: Apache-2.0
//! Variant kernels against Spark 4.2.0 (the `variant-field-extractions.sql`
//! goldens and the design note's oracle probes).

use std::sync::Arc;

use arrow_array::types::{Decimal128Type, Int32Type};
use arrow_array::{Array, ArrayRef, Int32Array, ListArray, StringArray, StructArray};
use arrow_schema::{DataType as ArrowType, Field};
use krishiv_common::error::ErrorClass;
use krishiv_types::variant::{VariantBuilder, VariantRef};
use krishiv_types::{ArrayType, DataType, DecimalType, StructField, StructType};

use super::*;
use crate::cast::FixedOffset;

fn utc() -> FixedOffset {
    FixedOffset::parse("+00:00").unwrap()
}

fn column(json: &[Option<&str>]) -> ArrayRef {
    let mut b = VariantColumnBuilder::with_capacity(json.len());
    for j in json {
        match j {
            Some(j) => {
                let (m, v) = VariantBuilder::from_json(j).unwrap();
                b.append(&m, &v);
            }
            None => b.append_null(),
        }
    }
    b.finish().unwrap()
}

fn json_of(col: &ArrayRef) -> Vec<Option<String>> {
    let out = to_json(col.as_ref(), &utc()).unwrap();
    out.as_string_view()
        .iter()
        .map(|s| s.map(str::to_owned))
        .collect()
}

#[test]
fn prints_as_spark() {
    let col = column(&[
        Some(r#"{"b":1,"a":2}"#),
        Some("1.0"),
        Some("1.50"),
        Some("[0.10, 100, 1.00e1]"),
        Some(r#""a\"b\u0001""#),
        Some("null"),
        None,
        Some(r#"{ "price": 12345.678 }"#),
    ]);
    assert_eq!(
        json_of(&col),
        vec![
            Some(r#"{"a":2,"b":1}"#.to_owned()),
            Some("1".to_owned()),
            Some("1.5".to_owned()),
            Some("[0.1,100,10.0]".to_owned()),
            Some(r#""a\"b\u0001""#.to_owned()),
            Some("null".to_owned()),
            None,
            Some(r#"{"price":12345.678}"#.to_owned()),
        ]
    );
}

#[test]
fn parses_paths_as_spark() {
    use PathSegment::{Index, Key};
    assert_eq!(parse_path("$"), Some(vec![]));
    assert_eq!(
        parse_path("$.item[0].model"),
        Some(vec![Key("item".into()), Index(0), Key("model".into())])
    );
    assert_eq!(
        parse_path("$['field.name']"),
        Some(vec![Key("field.name".into())])
    );
    assert_eq!(parse_path(r#"$["a b"]"#), Some(vec![Key("a b".into())]));
    assert_eq!(
        parse_path("$.field-name"),
        Some(vec![Key("field-name".into())])
    );
    for bad in [
        "",
        "a",
        "$.",
        "$[",
        "$[x]",
        "$[-1]",
        "$[99999999999]",
        "$..a",
        "$['a",
    ] {
        assert_eq!(parse_path(bad), None, "{bad}");
    }
    let e = parse_path_or_raise("$[x]", "variant_get").unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InvalidVariantGetPath);
}

fn first<'a>(col: &'a VariantColumn) -> VariantRef<'a> {
    col.get(0).unwrap().unwrap()
}

#[test]
fn extracts_by_path() {
    let col = column(&[Some(
        r#"{ "item": [ { "model" : "basic", "price" : 6.12 }, { "model" : "medium" } ] }"#,
    )]);
    let col = VariantColumn::new(col.as_ref()).unwrap();
    let v = first(&col);
    let get = |p: &str| {
        extract(v, &parse_path(p).unwrap()).unwrap().map(|x| {
            let mut s = String::new();
            write_json(&x, &utc(), &mut s).unwrap();
            s
        })
    };
    assert_eq!(get("$.item[0].model").as_deref(), Some(r#""basic""#));
    assert_eq!(get("$.item[1].price"), None);
    assert_eq!(get("$.item[2]"), None);
    assert_eq!(get("$.item.model"), None);
    assert_eq!(get("$[0]"), None);
}

fn cast_rows(json: &[Option<&str>], to: &DataType, fail: bool) -> Result<ArrayRef, KrishivError> {
    let col = column(json);
    cast_from_variant(col.as_ref(), to, fail, &utc())
}

#[test]
fn casts_scalars_as_spark() {
    let dec = DataType::Decimal(DecimalType::try_new(5, 2).unwrap());
    let out = cast_rows(
        &[Some("30"), Some("\"12\""), Some("null"), None],
        &dec,
        true,
    )
    .unwrap();
    let out = out.as_primitive::<arrow_array::types::Decimal64Type>();
    assert_eq!(out.value(0), 3000);
    assert_eq!(out.value(1), 1200);
    assert!(out.is_null(2) && out.is_null(3));

    let e = cast_rows(
        &[Some("1"), Some("12345.678")],
        &DataType::Decimal(DecimalType::try_new(3, 2).unwrap()),
        true,
    )
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InvalidVariantCast);
    assert!(e.to_string().contains("12345.678"), "{e}");
    assert!(e.to_string().contains("DECIMAL(3,2)"), "{e}");

    let out = cast_rows(
        &[Some("\"x\""), Some("3.7"), Some("[1]")],
        &DataType::Integer,
        false,
    )
    .unwrap();
    let out = out.as_primitive::<Int32Type>();
    assert!(out.is_null(0));
    assert_eq!(out.value(1), 3);
    assert!(out.is_null(2));

    let out = cast_rows(
        &[Some("30"), Some(r#"{"b":1,"a":[true]}"#), Some("\"s\"")],
        &DataType::String,
        true,
    )
    .unwrap();
    let out: Vec<_> = out.as_string_view().iter().collect();
    assert_eq!(
        out,
        vec![Some("30"), Some(r#"{"a":[true],"b":1}"#), Some("s")]
    );

    let out = cast_rows(&[Some("true")], &DataType::Boolean, true).unwrap();
    assert!(out.as_boolean().value(0));
    // A boolean is no date.
    assert!(cast_rows(&[Some("true")], &DataType::Date, true).is_err());
    let ts = cast_rows(&[Some("2")], &DataType::Timestamp, true).unwrap();
    assert_eq!(
        ts.as_primitive::<arrow_array::types::TimestampMicrosecondType>()
            .value(0),
        2_000_000
    );
}

#[test]
fn casts_containers() {
    let arr = DataType::Array(Arc::new(ArrayType::new(DataType::Integer, true)));
    let out = cast_rows(&[Some("[85, 92, null]"), Some("{}"), None], &arr, false).unwrap();
    let list = out.as_any().downcast_ref::<ListArray>().unwrap();
    assert_eq!(list.value_length(0), 3);
    assert!(list.is_null(1) && list.is_null(2));
    let st = DataType::Struct(StructType::new(vec![
        StructField::new("a", DataType::Long, true),
        StructField::new("z", DataType::String, true),
    ]));
    let out = cast_rows(&[Some(r#"{"z":"q","a":7,"x":1}"#)], &st, true).unwrap();
    let s = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(
        s.column(0)
            .as_primitive::<arrow_array::types::Int64Type>()
            .value(0),
        7
    );
    assert_eq!(s.column(1).as_string_view().value(0), "q");
}

#[test]
fn variant_target_rebuilds_the_dictionary() {
    let col = column(&[Some(r#"{"a":{"b":1},"c":2}"#)]);
    let c = VariantColumn::new(col.as_ref()).unwrap();
    let inner = extract(first(&c), &parse_path("$.a").unwrap())
        .unwrap()
        .unwrap();
    let out = cast_values(&[Some(inner)], &DataType::Variant, true, &utc()).unwrap();
    let out = VariantColumn::new(out.as_ref()).unwrap();
    let v = first(&out);
    // Only `b` is left in the dictionary.
    assert_eq!(v.metadata(), &[0x01, 0x01, 0x00, 0x01, b'b']);
    let mut s = String::new();
    write_json(&v, &utc(), &mut s).unwrap();
    assert_eq!(s, r#"{"b":1}"#);
}

#[test]
fn builds_from_values() {
    let ints: ArrayRef = Arc::new(Int32Array::from(vec![Some(30), None, Some(-300)]));
    let out = cast_to_variant(ints.as_ref(), &DataType::Integer).unwrap();
    assert_eq!(
        json_of(&out),
        vec![Some("30".into()), None, Some("-300".into())]
    );
    let strings: ArrayRef = Arc::new(StringArray::from(vec!["a"]));
    let out = cast_to_variant(strings.as_ref(), &DataType::String).unwrap();
    assert_eq!(json_of(&out), vec![Some(r#""a""#.into())]);
    let field = Arc::new(Field::new("element", ArrowType::Int32, true));
    let list: ArrayRef = Arc::new(
        ListArray::try_new(
            field,
            arrow_buffer::OffsetBuffer::new(vec![0, 2].into()),
            Arc::new(Int32Array::from(vec![Some(1), None])),
            None,
        )
        .unwrap(),
    );
    let t = DataType::Array(Arc::new(ArrayType::new(DataType::Integer, true)));
    let out = cast_to_variant(list.as_ref(), &t).unwrap();
    assert_eq!(json_of(&out), vec![Some("[1,null]".into())]);
    let dec: ArrayRef = Arc::new(
        arrow_array::PrimitiveArray::<Decimal128Type>::from(vec![1050_i128])
            .with_precision_and_scale(20, 2)
            .unwrap(),
    );
    let out = cast_to_variant(
        dec.as_ref(),
        &DataType::Decimal(DecimalType::try_new(20, 2).unwrap()),
    )
    .unwrap();
    assert_eq!(json_of(&out), vec![Some("10.5".into())]);
}

#[test]
fn duplicate_struct_keys_are_refused() {
    let st = DataType::Struct(StructType::new(vec![
        StructField::new("a", DataType::Integer, true),
        StructField::new("a", DataType::Integer, true),
    ]));
    let arrow = st.to_arrow();
    let ArrowType::Struct(fields) = arrow else {
        unreachable!()
    };
    let one: ArrayRef = Arc::new(Int32Array::from(vec![1]));
    let s: ArrayRef =
        Arc::new(StructArray::try_new(fields, vec![Arc::clone(&one), one], None).unwrap());
    let e = cast_to_variant(s.as_ref(), &st).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::VariantDuplicateKey);
}
