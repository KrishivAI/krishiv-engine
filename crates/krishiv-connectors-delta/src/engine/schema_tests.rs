// SPDX-License-Identifier: Apache-2.0
//! Schema and value conversion tests.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, TimestampMicrosecondType};
use arrow_array::{Array, Int32Array, StringArray, TimestampMicrosecondArray};
use arrow_schema::{DataType as ArrowType, Field as ArrowField, TimeUnit};
use delta_kernel::expressions::{ArrayData, MapData, Scalar, StructData};
use delta_kernel::schema::{ArrayType, DataType, MapType, MetadataValue, StructField, StructType};

use super::*;

fn all_types() -> StructType {
    StructType::try_new([
        StructField::nullable("s", DataType::STRING),
        StructField::not_null("l", DataType::LONG),
        StructField::nullable("i", DataType::INTEGER),
        StructField::nullable("sh", DataType::SHORT),
        StructField::nullable("b", DataType::BYTE),
        StructField::nullable("f", DataType::FLOAT),
        StructField::nullable("d", DataType::DOUBLE),
        StructField::nullable("bo", DataType::BOOLEAN),
        StructField::nullable("bi", DataType::BINARY),
        StructField::nullable("da", DataType::DATE),
        StructField::nullable("ts", DataType::TIMESTAMP),
        StructField::nullable("ntz", DataType::TIMESTAMP_NTZ),
        StructField::nullable("dec", DataType::decimal(10, 2).unwrap()),
        StructField::nullable("arr", ArrayType::new(DataType::STRING, true)),
        StructField::nullable("m", MapType::new(DataType::STRING, DataType::STRING, true)),
        StructField::nullable(
            "st",
            StructType::try_new([StructField::nullable("x", DataType::INTEGER)]).unwrap(),
        ),
    ])
    .unwrap()
}

#[test]
fn every_type_round_trips() {
    let kernel = all_types();
    let arrow = to_arrow_schema(&kernel).unwrap();
    assert_eq!(
        arrow.field_with_name("ts").unwrap().data_type(),
        &ArrowType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
    );
    assert_eq!(
        arrow.field_with_name("ntz").unwrap().data_type(),
        &ArrowType::Timestamp(TimeUnit::Microsecond, None)
    );
    assert!(!arrow.field_with_name("l").unwrap().is_nullable());
    let ArrowType::Map(entries, false) = arrow.field_with_name("m").unwrap().data_type() else {
        panic!("map");
    };
    assert_eq!(entries.name(), "key_value");
    assert!(!entries.is_nullable());
    assert_eq!(to_kernel_schema(&arrow).unwrap(), kernel);
}

#[test]
fn variant_is_the_struct_of_metadata_and_value() {
    let ArrowType::Struct(fields) = to_arrow_type(&DataType::unshredded_variant()).unwrap() else {
        panic!("struct");
    };
    let names: Vec<_> = fields
        .iter()
        .map(|f| (f.name().as_str(), f.data_type().clone(), f.is_nullable()))
        .collect();
    assert_eq!(
        names,
        [
            ("metadata", ArrowType::Binary, false),
            ("value", ArrowType::Binary, false)
        ]
    );
    let shredded = DataType::variant_type([
        StructField::not_null("metadata", DataType::BINARY),
        StructField::nullable("value", DataType::BINARY),
        StructField::nullable("typed_value", DataType::INTEGER),
    ])
    .unwrap();
    assert!(to_arrow_type(&shredded).is_err());
}

#[test]
fn there_is_no_nanosecond_or_millisecond_type() {
    for unit in [
        TimeUnit::Nanosecond,
        TimeUnit::Millisecond,
        TimeUnit::Second,
    ] {
        assert!(
            to_kernel_type(&ArrowType::Timestamp(unit, None)).is_err(),
            "{unit:?}"
        );
        assert!(
            to_kernel_type(&ArrowType::Timestamp(unit, Some("UTC".into()))).is_err(),
            "{unit:?}"
        );
    }
    // Any zone label is an instant; an empty one is not.
    assert_eq!(
        to_kernel_type(&ArrowType::Timestamp(
            TimeUnit::Microsecond,
            Some("+05:30".into())
        ))
        .unwrap(),
        DataType::TIMESTAMP
    );
    assert_eq!(
        to_kernel_type(&ArrowType::Timestamp(
            TimeUnit::Microsecond,
            Some("".into())
        ))
        .unwrap(),
        DataType::TIMESTAMP_NTZ
    );
    for t in [
        ArrowType::UInt8,
        ArrowType::UInt64,
        ArrowType::Date64,
        ArrowType::Float16,
    ] {
        assert!(to_kernel_type(&t).is_err(), "{t}");
    }
    assert!(to_kernel_type(&ArrowType::Decimal128(10, -1)).is_err());
}

#[test]
fn field_ids_and_nested_ids_cross_both_ways() {
    let nested = serde_json::json!({"m.key": 101, "m.value": 102, "m.value.element": 103});
    let field = StructField::nullable(
        "m",
        MapType::new(
            DataType::STRING,
            ArrayType::new(DataType::INTEGER, true),
            true,
        ),
    )
    .with_metadata([
        ("parquet.field.id".to_owned(), MetadataValue::Number(100)),
        (
            "delta.columnMapping.nested.ids".to_owned(),
            MetadataValue::Other(nested),
        ),
        ("comment".to_owned(), MetadataValue::String("c".to_owned())),
    ]);
    let arrow = to_arrow_field(&field).unwrap();
    assert_eq!(
        arrow.metadata().get("PARQUET:field_id").map(String::as_str),
        Some("100")
    );
    assert!(
        !arrow
            .metadata()
            .contains_key("delta.columnMapping.nested.ids")
    );
    let ArrowType::Map(entries, _) = arrow.data_type() else {
        panic!("map")
    };
    let ArrowType::Struct(kv) = entries.data_type() else {
        panic!("entries")
    };
    assert_eq!(
        kv[0].metadata().get("PARQUET:field_id").map(String::as_str),
        Some("101")
    );
    assert_eq!(
        kv[1].metadata().get("PARQUET:field_id").map(String::as_str),
        Some("102")
    );
    let ArrowType::List(element) = kv[1].data_type() else {
        panic!("list")
    };
    assert_eq!(
        element
            .metadata()
            .get("PARQUET:field_id")
            .map(String::as_str),
        Some("103")
    );
    assert_eq!(to_kernel_field(&arrow).unwrap(), field);
    // Two different ids on one field are refused.
    let clash = ArrowField::new("x", ArrowType::Int32, true).with_metadata(
        std::collections::HashMap::from([
            ("PARQUET:field_id".to_owned(), "1".to_owned()),
            ("parquet.field.id".to_owned(), "2".to_owned()),
        ]),
    );
    assert!(to_kernel_field(&clash).is_err());
}

#[test]
fn scalars_become_arrays_and_back() {
    let ts = scalar_to_array(&Scalar::Timestamp(7), 3).unwrap();
    assert_eq!(ts.len(), 3);
    assert_eq!(ts.as_primitive::<TimestampMicrosecondType>().value(2), 7);
    assert_eq!(scalar_at(ts.as_ref(), 1).unwrap(), Scalar::Timestamp(7));
    let ntz = TimestampMicrosecondArray::from(vec![Some(5), None]);
    assert_eq!(scalar_at(&ntz, 0).unwrap(), Scalar::TimestampNtz(5));
    assert_eq!(
        scalar_at(&ntz, 1).unwrap(),
        Scalar::Null(DataType::TIMESTAMP_NTZ)
    );
    let dec = scalar_to_array(&Scalar::decimal(12345, 10, 2).unwrap(), 1).unwrap();
    assert_eq!(dec.data_type(), &ArrowType::Decimal128(10, 2));
    assert_eq!(
        scalar_at(dec.as_ref(), 0).unwrap(),
        Scalar::decimal(12345, 10, 2).unwrap()
    );
    assert_eq!(
        scalar_at(&StringArray::from(vec!["a"]), 0).unwrap(),
        Scalar::from("a")
    );
    assert_eq!(
        scalar_at(&Int32Array::from(vec![4]), 0).unwrap(),
        Scalar::Integer(4)
    );
    assert!(scalar_at(&Int32Array::from(vec![4]), 1).is_err());
    // Nested values and nulls of every shape.
    let st = StructData::try_new(
        vec![
            StructField::nullable("x", DataType::INTEGER),
            StructField::nullable("y", DataType::STRING),
        ],
        vec![Scalar::Integer(1), Scalar::Null(DataType::STRING)],
    )
    .unwrap();
    let arr = scalar_to_array(&Scalar::Struct(st), 2).unwrap();
    let st = arr.as_struct();
    assert_eq!(
        st.column(0).as_primitive::<Int32Type>().values().to_vec(),
        [1, 1]
    );
    assert_eq!(st.column(1).null_count(), 2);
    let list = ArrayData::try_new(
        ArrayType::new(DataType::STRING, true),
        [Scalar::from("a"), Scalar::Null(DataType::STRING)],
    )
    .unwrap();
    let arr = scalar_to_array(&Scalar::Array(list.clone()), 2).unwrap();
    assert_eq!(arr.as_list::<i32>().value(1).len(), 2);
    let elements = array_data_to_arrow(&list).unwrap();
    assert_eq!((elements.len(), elements.null_count()), (2, 1));
    let map = MapData::try_new(
        MapType::new(DataType::STRING, DataType::LONG, true),
        [(Scalar::from("k"), Scalar::Long(1))],
    )
    .unwrap();
    let arr = scalar_to_array(&Scalar::Map(map), 1).unwrap();
    assert_eq!(arr.as_map().keys().as_string::<i32>().value(0), "k");
    for t in [
        DataType::from(all_types()),
        DataType::from(ArrayType::new(DataType::LONG, true)),
        DataType::from(MapType::new(DataType::STRING, DataType::STRING, true)),
        DataType::unshredded_variant(),
        DataType::VOID,
    ] {
        let nulls = scalar_to_array(&Scalar::Null(t.clone()), 3).unwrap();
        assert_eq!(nulls.len(), 3, "{t}");
        assert_eq!(nulls.data_type(), &to_arrow_type(&t).unwrap(), "{t}");
        assert_eq!(nulls.logical_null_count(), 3, "{t}");
    }
    let _unused: Arc<dyn Array> = Arc::new(Int32Array::from(vec![1]));
}

/// Every Arrow layout of a Delta type reads back as that type; other types are refused.
#[test]
fn arrow_layouts_map_to_kernel_types() {
    let item = Arc::new(ArrowField::new("element", ArrowType::Utf8, true));
    let strings = ArrayType::new(DataType::STRING, true);
    for (arrow, kernel) in [
        (ArrowType::LargeUtf8, DataType::STRING),
        (ArrowType::Utf8View, DataType::STRING),
        (ArrowType::Int16, DataType::SHORT),
        (ArrowType::Int8, DataType::BYTE),
        (ArrowType::Null, DataType::VOID),
        (ArrowType::Float32, DataType::FLOAT),
        (
            ArrowType::LargeList(Arc::clone(&item)),
            strings.clone().into(),
        ),
        (
            ArrowType::ListView(Arc::clone(&item)),
            strings.clone().into(),
        ),
        (ArrowType::LargeListView(Arc::clone(&item)), strings.into()),
        (
            ArrowType::Dictionary(Box::new(ArrowType::Int32), Box::new(ArrowType::Utf8)),
            DataType::STRING,
        ),
    ] {
        assert_eq!(to_kernel_type(&arrow).unwrap(), kernel, "{arrow}");
    }
    assert!(to_kernel_type(&ArrowType::UInt32).is_err());
    let bad_map = ArrowType::Map(
        Arc::new(ArrowField::new(
            "entries",
            ArrowType::Struct(vec![ArrowField::new("k", ArrowType::Utf8, false)].into()),
            false,
        )),
        false,
    );
    assert!(to_kernel_type(&bad_map).is_err());
    let field = ArrowField::new("m", bad_map, true);
    assert!(to_kernel_field(&field).is_err());
}

#[test]
fn malformed_field_ids_are_refused() {
    let key = delta_kernel::schema::ColumnMetadataKey::ColumnMappingNestedIds.as_ref();
    let list = ArrayType::new(DataType::STRING, true);
    let not_object = StructField::nullable("a", list.clone())
        .with_metadata([(key, MetadataValue::String("x".to_owned()))]);
    assert!(to_arrow_field(&not_object).is_err());
    let not_int = StructField::nullable("a", list).with_metadata([(
        key,
        MetadataValue::Other(serde_json::json!({ "a.element": "seven" })),
    )]);
    assert!(to_arrow_field(&not_int).is_err());
    let field = ArrowField::new("x", ArrowType::Int32, true).with_metadata(
        std::collections::HashMap::from([("PARQUET:field_id".to_owned(), "seven".to_owned())]),
    );
    assert!(to_kernel_field(&field).is_err());
}

/// `scalar_at` reads every primitive layout, a null as a typed null, and refuses the rest.
#[test]
fn scalar_at_reads_every_layout() {
    use arrow_array::{
        BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Decimal128Array, Float32Array,
        Float64Array, Int8Array, Int16Array, Int64Array, LargeBinaryArray, LargeStringArray,
        StringViewArray,
    };
    let cases: Vec<(Arc<dyn Array>, Scalar)> = vec![
        (Arc::new(Int8Array::from(vec![1])), Scalar::Byte(1)),
        (Arc::new(Int16Array::from(vec![2])), Scalar::Short(2)),
        (Arc::new(Int64Array::from(vec![3])), Scalar::Long(3)),
        (Arc::new(Float32Array::from(vec![1.5])), Scalar::Float(1.5)),
        (Arc::new(Float64Array::from(vec![2.5])), Scalar::Double(2.5)),
        (
            Arc::new(BooleanArray::from(vec![true])),
            Scalar::Boolean(true),
        ),
        (Arc::new(Date32Array::from(vec![10])), Scalar::Date(10)),
        (
            Arc::new(TimestampMicrosecondArray::from(vec![5]).with_timezone("UTC")),
            Scalar::Timestamp(5),
        ),
        (
            Arc::new(TimestampMicrosecondArray::from(vec![6])),
            Scalar::TimestampNtz(6),
        ),
        (
            Arc::new(
                Decimal128Array::from(vec![1234])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
            Scalar::decimal(1234, 10, 2).unwrap(),
        ),
        (
            Arc::new(LargeStringArray::from(vec!["l"])),
            Scalar::String("l".to_owned()),
        ),
        (
            Arc::new(StringViewArray::from(vec!["v"])),
            Scalar::String("v".to_owned()),
        ),
        (
            Arc::new(BinaryArray::from(vec![&b"b"[..]])),
            Scalar::Binary(b"b".to_vec()),
        ),
        (
            Arc::new(LargeBinaryArray::from(vec![&b"lb"[..]])),
            Scalar::Binary(b"lb".to_vec()),
        ),
        (
            Arc::new(BinaryViewArray::from(vec![&b"bv"[..]])),
            Scalar::Binary(b"bv".to_vec()),
        ),
    ];
    for (array, expected) in cases {
        assert_eq!(scalar_at(array.as_ref(), 0).unwrap(), expected);
    }
    let nulls = Int32Array::from(vec![None::<i32>]);
    assert_eq!(
        scalar_at(&nulls, 0).unwrap(),
        Scalar::Null(DataType::INTEGER)
    );
    assert!(scalar_at(&nulls, 1).is_err(), "out of bounds");
    let list =
        arrow_array::ListArray::from_iter_primitive::<Int32Type, _, _>(vec![Some(vec![Some(1)])]);
    assert!(scalar_at(&list, 0).is_err(), "not primitive");
}

/// Scalars of every primitive and nested type, and typed nulls of nested types, become arrays.
#[test]
fn scalars_of_every_type_become_arrays() {
    let st = StructType::try_new([
        StructField::nullable("a", DataType::SHORT),
        StructField::nullable("m", MapType::new(DataType::STRING, DataType::STRING, true)),
    ])
    .unwrap();
    let scalars = [
        Scalar::Short(1),
        Scalar::Byte(2),
        Scalar::Float(1.5),
        Scalar::Double(2.5),
        Scalar::Date(3),
        Scalar::Binary(b"x".to_vec()),
        Scalar::Struct(
            StructData::try_new(
                st.fields().cloned().collect(),
                vec![
                    Scalar::Short(4),
                    Scalar::Null(MapType::new(DataType::STRING, DataType::STRING, true).into()),
                ],
            )
            .unwrap(),
        ),
        Scalar::Null(st.into()),
        Scalar::Null(MapType::new(DataType::STRING, DataType::STRING, true).into()),
        Scalar::Null(ArrayType::new(DataType::STRING, true).into()),
    ];
    for scalar in scalars {
        let array = scalar_to_array(&scalar, 3).unwrap();
        assert_eq!(array.len(), 3, "{scalar:?}");
        assert_eq!(array.null_count() == 3, scalar.is_null(), "{scalar:?}");
    }
}
