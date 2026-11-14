// SPDX-License-Identifier: Apache-2.0
//! `recarried_array`: decimals of every carrier, alone and nested, keep their
//! values and NULLs; anything but a carrier difference is refused.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal64Type, Decimal128Type};
use arrow_array::{
    ArrayRef, Decimal32Array, Decimal64Array, Decimal128Array, FixedSizeListArray, Int32Array,
    LargeListArray, ListArray, MapArray, StructArray,
};
use arrow_buffer::{NullBuffer, OffsetBuffer};
use arrow_schema::{DataType, Field, Fields};
use krishiv_common::error::ErrorClass;
use krishiv_exec::carrier::recarried_array;

const D64: DataType = DataType::Decimal64(10, 2);
const D128: DataType = DataType::Decimal128(10, 2);

fn d64() -> ArrayRef {
    Arc::new(
        Decimal64Array::from(vec![Some(150), None, Some(-275)])
            .with_precision_and_scale(10, 2)
            .unwrap(),
    )
}

fn d128() -> ArrayRef {
    Arc::new(
        Decimal128Array::from(vec![Some(150), None, Some(-275)])
            .with_precision_and_scale(10, 2)
            .unwrap(),
    )
}

fn item(t: DataType) -> Arc<Field> {
    Arc::new(Field::new("item", t, true))
}

#[test]
fn the_same_type_is_the_same_array() {
    let a = d64();
    assert!(Arc::ptr_eq(&a, &recarried_array(&a, &D64).unwrap()));
}

#[test]
fn every_carrier_keeps_values_and_nulls() {
    assert_eq!(
        recarried_array(&d64(), &D128).unwrap().as_ref(),
        d128().as_ref()
    );
    assert_eq!(
        recarried_array(&d128(), &D64).unwrap().as_ref(),
        d64().as_ref()
    );
    let d32: ArrayRef = Arc::new(
        Decimal32Array::from(vec![Some(150), None, Some(-275)])
            .with_precision_and_scale(9, 2)
            .unwrap(),
    );
    let narrow = recarried_array(&d32, &DataType::Decimal64(9, 2)).unwrap();
    assert_eq!(
        narrow
            .as_primitive::<Decimal64Type>()
            .iter()
            .collect::<Vec<_>>(),
        [Some(150), None, Some(-275)]
    );
    let wide = recarried_array(&d32, &DataType::Decimal128(9, 2)).unwrap();
    assert_eq!(
        wide.as_primitive::<Decimal128Type>()
            .iter()
            .collect::<Vec<_>>(),
        [Some(150), None, Some(-275)]
    );
}

#[test]
fn a_value_beyond_the_narrow_carrier_is_refused() {
    let big: ArrayRef = Arc::new(
        Decimal128Array::from(vec![i128::from(i64::MAX) + 1])
            .with_precision_and_scale(38, 0)
            .unwrap(),
    );
    let e = recarried_array(&big, &DataType::Decimal64(18, 0)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
}

/// Values under NULLs are arbitrary: only valid slots must fit.
#[test]
fn a_wide_value_under_a_null_does_not_matter() {
    let a: ArrayRef = Arc::new(
        Decimal128Array::new(
            vec![i128::MAX, 5].into(),
            Some(NullBuffer::from(vec![false, true])),
        )
        .with_precision_and_scale(10, 2)
        .unwrap(),
    );
    let narrow = recarried_array(&a, &D64).unwrap();
    assert_eq!(
        narrow
            .as_primitive::<Decimal64Type>()
            .iter()
            .collect::<Vec<_>>(),
        [None, Some(5)]
    );
}

#[test]
fn only_carriers_may_differ() {
    let ints: ArrayRef = Arc::new(Int32Array::from(vec![1]));
    for to in [D64, D128, DataType::Utf8] {
        let e = recarried_array(&ints, &to).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError, "{to}");
    }
    // `Decimal32` is read, never written.
    let e = recarried_array(&d64(), &DataType::Decimal32(9, 2)).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
    // Map entries that are not a struct.
    let map = DataType::Map(item(DataType::Int32), false);
    let e = recarried_array(&ints, &map).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::InternalError);
}

#[test]
fn lists_recarry_their_items() {
    let lengths = || OffsetBuffer::<i32>::from_lengths([2, 0, 1]);
    let nulls = || Some(NullBuffer::from(vec![true, false, true]));
    let list =
        |t, values| -> ArrayRef { Arc::new(ListArray::new(item(t), lengths(), values, nulls())) };
    assert_eq!(
        recarried_array(&list(D64, d64()), &DataType::List(item(D128)))
            .unwrap()
            .as_ref(),
        list(D128, d128()).as_ref(),
    );

    let large_lengths = || OffsetBuffer::<i64>::from_lengths([3]);
    let large = |t, values| -> ArrayRef {
        Arc::new(LargeListArray::new(item(t), large_lengths(), values, None))
    };
    assert_eq!(
        recarried_array(&large(D64, d64()), &DataType::LargeList(item(D128)))
            .unwrap()
            .as_ref(),
        large(D128, d128()).as_ref(),
    );

    let fixed =
        |t, values| -> ArrayRef { Arc::new(FixedSizeListArray::new(item(t), 3, values, None)) };
    assert_eq!(
        recarried_array(&fixed(D128, d128()), &DataType::FixedSizeList(item(D64), 3))
            .unwrap()
            .as_ref(),
        fixed(D64, d64()).as_ref(),
    );
}

#[test]
fn structs_and_maps_recarry_their_fields() {
    let fields = |t: DataType| -> Fields {
        vec![
            Field::new("k", DataType::Int32, false),
            Field::new("v", t, true),
        ]
        .into()
    };
    let row = |t, values| {
        StructArray::new(
            fields(t),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
                values,
            ],
            None,
        )
    };
    let wide: ArrayRef = Arc::new(row(D128, d128()));
    assert_eq!(
        recarried_array(&wide, &DataType::Struct(fields(D64)))
            .unwrap()
            .as_ref(),
        &row(D64, d64()),
    );

    let entries = |t: DataType| Arc::new(Field::new("entries", DataType::Struct(fields(t)), false));
    let map = |t: DataType, values| -> ArrayRef {
        Arc::new(MapArray::new(
            entries(t.clone()),
            OffsetBuffer::from_lengths([3]),
            row(t, values),
            None,
            false,
        ))
    };
    assert_eq!(
        recarried_array(&map(D64, d64()), &DataType::Map(entries(D128), false))
            .unwrap()
            .as_ref(),
        map(D128, d128()).as_ref(),
    );
}

fn vectors(rows: &[Option<[f32; 3]>]) -> ArrayRef {
    let values: Vec<Option<f32>> = rows
        .iter()
        .flat_map(|r| match r {
            Some(v) => v.iter().copied().map(Some).collect::<Vec<_>>(),
            None => vec![None; 3],
        })
        .collect();
    let field = Arc::new(Field::new("item", DataType::Float32, true));
    let nulls = rows
        .iter()
        .any(Option::is_none)
        .then(|| NullBuffer::from(rows.iter().map(Option::is_some).collect::<Vec<_>>()));
    Arc::new(
        FixedSizeListArray::try_new(
            field,
            3,
            Arc::new(arrow_array::Float32Array::from(values)),
            nulls,
        )
        .unwrap(),
    )
}

fn plain_list() -> DataType {
    DataType::List(item(DataType::Float32))
}

/// Design P3-T075 §5: a vector column read by an expression or an edge is the plain list of the same values; with no
/// null rows the values buffer is the carrier's own.
#[test]
fn a_vector_without_null_rows_becomes_a_list_sharing_its_values() {
    let v = vectors(&[Some([1.0, 2.0, 3.0]), Some([4.0, 5.0, 6.0])]);
    let out = recarried_array(&v, &plain_list()).unwrap();
    let list = out.as_list::<i32>();
    assert_eq!(
        list.offsets().iter().copied().collect::<Vec<_>>(),
        [0, 3, 6]
    );
    assert!(list.nulls().is_none());
    let carrier = v.as_fixed_size_list();
    assert_eq!(
        list.values().to_data().buffers()[0].as_ptr(),
        carrier.values().to_data().buffers()[0].as_ptr(),
        "the values are shared, not copied"
    );
    assert_eq!(
        list.values()
            .as_primitive::<arrow_array::types::Float32Type>()
            .values(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

/// A null vector is an empty null slot, as a list builder writes it; the other rows keep their values.
#[test]
fn a_null_vector_becomes_an_empty_null_slot() {
    let v = vectors(&[Some([1.0, 2.0, 3.0]), None, Some([7.0, 8.0, 9.0])]);
    let out = recarried_array(&v, &plain_list()).unwrap();
    let list = out.as_list::<i32>();
    assert_eq!(
        list.offsets().iter().copied().collect::<Vec<_>>(),
        [0, 3, 3, 6]
    );
    assert!(list.is_valid(0) && list.is_null(1) && list.is_valid(2));
    assert_eq!(
        list.values()
            .as_primitive::<arrow_array::types::Float32Type>()
            .values(),
        &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0]
    );
}

/// A sliced carrier converts what it shows, not what it hides.
#[test]
fn a_sliced_vector_converts_only_its_rows() {
    let v = vectors(&[Some([1.0, 2.0, 3.0]), None, Some([7.0, 8.0, 9.0])]).slice(1, 2);
    let out = recarried_array(&v, &plain_list()).unwrap();
    let list = out.as_list::<i32>();
    assert_eq!(list.len(), 2);
    assert!(list.is_null(0));
    assert_eq!(
        list.value(1)
            .as_primitive::<arrow_array::types::Float32Type>()
            .values(),
        &[7.0, 8.0, 9.0]
    );
}

/// A vector inside a struct reaches the edge as a list too.
#[test]
fn a_vector_inside_a_struct_becomes_a_list() {
    let fields = Fields::from(vec![Field::new("v", plain_list(), true)]);
    let s: ArrayRef = Arc::new(
        StructArray::try_new(
            Fields::from(vec![Field::new(
                "v",
                DataType::FixedSizeList(item(DataType::Float32), 3),
                true,
            )]),
            vec![vectors(&[Some([1.0, 2.0, 3.0])])],
            None,
        )
        .unwrap(),
    );
    let out = recarried_array(&s, &DataType::Struct(fields)).unwrap();
    assert_eq!(out.as_struct().column(0).as_list::<i32>().len(), 1);
}
