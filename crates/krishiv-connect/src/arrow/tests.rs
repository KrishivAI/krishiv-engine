// SPDX-License-Identifier: Apache-2.0
//! The Connect shapes match what Spark 4.2 sends (`df._collect_as_arrow()`
//! over a local Spark 4.2.0, recorded in the expectations below).

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, Int32Array, StructArray, Time64NanosecondArray,
};
use arrow_schema::{DataType as ArrowType, Field, TimeUnit};
use krishiv_common::error::ErrorClass;
use krishiv_types::{DataType, Srid};

use super::{spatial_column, spatial_field};

const POINT: &[u8] = &[
    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xF0, 0x3F, 0, 0, 0, 0, 0, 0, 0, 0x40,
];

/// An in-memory spatial column: a point, a null row, a point with SRID 3857.
fn in_memory(t: &DataType) -> ArrayRef {
    let field = t.to_arrow_field("g", true);
    let ArrowType::Struct(fields) = field.data_type().clone() else {
        panic!("not a struct");
    };
    let srid: ArrayRef = Arc::new(Int32Array::from(vec![0, 0, 3857]));
    let wkb: ArrayRef = Arc::new(BinaryViewArray::from(vec![POINT, b"", POINT]));
    let nulls = Some(vec![true, false, true].into());
    Arc::new(StructArray::try_new(fields, vec![srid, wkb], nulls).unwrap())
}

#[test]
fn spatial_fields_match_spark() {
    let cases = [
        (
            DataType::Geometry {
                srid: Srid::Fixed(0),
            },
            "geometry",
            "0",
        ),
        (
            DataType::Geometry {
                srid: Srid::Fixed(3857),
            },
            "geometry",
            "3857",
        ),
        (DataType::Geometry { srid: Srid::Any }, "geometry", "-1"),
        (
            DataType::Geography {
                srid: Srid::Fixed(4326),
            },
            "geography",
            "4326",
        ),
        (DataType::Geography { srid: Srid::Any }, "geography", "-1"),
    ];
    for (t, tag, srid) in cases {
        let field = spatial_field("g", &t, false).unwrap();
        assert!(!field.is_nullable());
        let ArrowType::Struct(children) = field.data_type() else {
            panic!("{t}: not a struct");
        };
        assert_eq!(
            children[0].as_ref(),
            &Field::new("srid", ArrowType::Int32, false)
        );
        assert_eq!(children[1].name(), "wkb");
        assert_eq!(children[1].data_type(), &ArrowType::Binary);
        assert!(!children[1].is_nullable());
        let mut metadata: Vec<(&str, &str)> = children[1]
            .metadata()
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        metadata.sort_unstable();
        let mut expected = vec![(tag, "true"), ("srid", srid)];
        expected.sort_unstable();
        assert_eq!(metadata, expected, "{t}");
        assert!(field.metadata().is_empty() && children[0].metadata().is_empty());
    }
    assert!(spatial_field("t", &DataType::Time { precision: 6 }, true).is_none());
    // A time column is sent as it is held: nanoseconds, no metadata.
    let time = DataType::Time { precision: 2 }.to_arrow();
    assert_eq!(time, ArrowType::Time64(TimeUnit::Nanosecond));
    let values = Time64NanosecondArray::from(vec![36_000_000_000_000]);
    assert_eq!(values.data_type(), &time);
}

#[test]
fn spatial_columns_convert_to_the_connect_form() {
    let t = DataType::Geometry { srid: Srid::Any };
    let out = spatial_column(&in_memory(&t), &t).unwrap();
    let field = spatial_field("g", &t, true).unwrap();
    assert_eq!(out.data_type(), field.data_type());
    let out = out.as_any().downcast_ref::<StructArray>().unwrap();
    assert_eq!(out.len(), 3);
    assert!(out.is_valid(0) && out.is_null(1) && out.is_valid(2));
    let srid = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(srid.values().as_ref(), [0, 0, 3857]);
    let wkb = out
        .column(1)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap();
    assert_eq!(wkb.value(0), POINT);
    assert_eq!(wkb.value(2), POINT);
    assert_eq!(wkb.null_count(), 0);
}

#[test]
fn a_column_that_is_not_spatial_is_an_engine_error() {
    let g = DataType::Geography {
        srid: Srid::Fixed(4326),
    };
    let class =
        |column: &ArrayRef, t: &DataType| spatial_column(column, t).unwrap_err().error_class();
    let ints: ArrayRef = Arc::new(Int32Array::from(vec![1]));
    assert_eq!(class(&ints, &g), ErrorClass::InternalError);
    assert_eq!(
        class(&in_memory(&g), &DataType::Binary),
        ErrorClass::InternalError
    );
    // Already converted: the WKB child is not a view.
    let converted = spatial_column(&in_memory(&g), &g).unwrap();
    assert_eq!(class(&converted, &g), ErrorClass::InternalError);
    // A null SRID.
    let fields = vec![
        Field::new("srid", ArrowType::Int32, true),
        Field::new("wkb", ArrowType::BinaryView, false),
    ];
    let srid: ArrayRef = Arc::new(Int32Array::from(vec![None::<i32>]));
    let wkb: ArrayRef = Arc::new(BinaryViewArray::from(vec![POINT]));
    let bad = StructArray::try_new(fields.into(), vec![srid, wkb], None).unwrap();
    assert_eq!(
        class(&(Arc::new(bad) as ArrayRef), &g),
        ErrorClass::InternalError
    );
}
