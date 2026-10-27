// SPDX-License-Identifier: Apache-2.0
//! Tests for the canonical key encodings and total row orderings: each
//! Arrow carrier encodes distinctly and equally for equal Spark values,
//! and [`super::cmp_rows`] orders NULLs, NaNs and nested values as Spark
//! does.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Int32Array, Int64Array, LargeStringArray, ListArray, StringArray,
    StringViewArray, StructArray,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, Fields};

use super::*;

fn i32s(values: Vec<Option<i32>>) -> ArrayRef {
    Arc::new(Int32Array::from(values))
}

fn strs(values: Vec<Option<&str>>) -> ArrayRef {
    Arc::new(StringArray::from(values))
}

/// Encodes `array[idx]`; `None` for NULL.
fn code(array: &ArrayRef, idx: usize) -> Option<Vec<u8>> {
    normalize_key(array.as_ref(), idx).unwrap()
}

#[test]
fn null_is_no_key() {
    assert_eq!(code(&i32s(vec![None]), 0), None);
    assert_eq!(code(&strs(vec![None]), 0), None);
}

#[test]
fn equal_values_encode_equally() {
    // Integers widen to a common width (same-type keys only ever meet,
    // so the width never decides); strings read across carriers.
    let a = i32s(vec![Some(7)]);
    let b: ArrayRef = Arc::new(Int64Array::from(vec![Some(7)]));
    assert_eq!(code(&a, 0), code(&b, 0));
    assert_eq!(code(&a, 0), code(&i32s(vec![Some(7)]), 0));
    assert_ne!(code(&a, 0), code(&i32s(vec![Some(8)]), 0));
    let view: ArrayRef = Arc::new(StringViewArray::from(vec![Some("k")]));
    assert_eq!(code(&strs(vec![Some("k")]), 0), code(&view, 0));
    let large: ArrayRef = Arc::new(LargeStringArray::from(vec![Some("k")]));
    assert_eq!(code(&strs(vec![Some("k")]), 0), code(&large, 0));
}

#[test]
fn floats_normalize() {
    use arrow_array::Float32Array;
    use arrow_array::Float64Array;
    let f32 = |v: f32| Arc::new(Float32Array::from(vec![v])) as ArrayRef;
    let f64 = |v: f64| Arc::new(Float64Array::from(vec![v])) as ArrayRef;
    // `-0.0` folds to `0.0`, every NaN folds to one canonical NaN.
    assert_eq!(code(&f32(0.0), 0), code(&f32(-0.0), 0));
    assert_eq!(code(&f32(f32::NAN), 0), code(&f32(-f32::NAN), 0));
    assert_eq!(code(&f64(0.0), 0), code(&f64(-0.0), 0));
    assert_eq!(code(&f64(f64::NAN), 0), code(&f64(-f64::NAN), 0));
    assert_ne!(code(&f32(1.0), 0), code(&f32(2.0), 0));
}

#[test]
fn nested_values_encode_recursively() {
    let field = Arc::new(Field::new("element", DataType::Int32, true));
    let list = |values: Vec<i32>| {
        Arc::new(ListArray::new(
            Arc::clone(&field),
            OffsetBuffer::new(ScalarBuffer::from(vec![
                0,
                i32::try_from(values.len()).unwrap(),
            ])),
            Arc::new(Int32Array::from(values)),
            None,
        )) as ArrayRef
    };
    assert_eq!(code(&list(vec![1, 2]), 0), code(&list(vec![1, 2]), 0));
    assert_ne!(code(&list(vec![1, 2]), 0), code(&list(vec![1, 3]), 0));
    assert_ne!(code(&list(vec![1]), 0), code(&list(vec![1, 2]), 0));
}

#[test]
fn rows_order_as_spark_does() {
    use arrow_array::Float64Array;
    let f = |v: Vec<Option<f64>>| Arc::new(Float64Array::from(v)) as ArrayRef;
    let nan = f(vec![Some(f64::NAN)]);
    let one = f(vec![Some(1.0)]);
    let neg = f(vec![Some(-0.0)]);
    let zero = f(vec![Some(0.0)]);
    let null = f(vec![None]);
    // NULL is less than every value, NaN greater than every non-NaN,
    // `-0.0` equals `0.0`.
    assert_eq!(
        cmp_rows(null.as_ref(), 0, one.as_ref(), 0).unwrap(),
        Ordering::Less
    );
    assert_eq!(
        cmp_rows(one.as_ref(), 0, null.as_ref(), 0).unwrap(),
        Ordering::Greater
    );
    assert_eq!(
        cmp_rows(nan.as_ref(), 0, one.as_ref(), 0).unwrap(),
        Ordering::Greater
    );
    assert_eq!(
        cmp_rows(one.as_ref(), 0, nan.as_ref(), 0).unwrap(),
        Ordering::Less
    );
    assert_eq!(
        cmp_rows(nan.as_ref(), 0, nan.as_ref(), 0).unwrap(),
        Ordering::Equal
    );
    assert_eq!(
        cmp_rows(neg.as_ref(), 0, zero.as_ref(), 0).unwrap(),
        Ordering::Equal
    );
    // Strings compare byte-wise.
    let (a, b) = (strs(vec![Some("a")]), strs(vec![Some("b")]));
    assert_eq!(
        cmp_rows(a.as_ref(), 0, b.as_ref(), 0).unwrap(),
        Ordering::Less
    );
}

#[test]
fn structs_order_lexicographically() {
    let fields: Fields = vec![
        Field::new("x", DataType::Int32, true),
        Field::new("y", DataType::Utf8, true),
    ]
    .into();
    let of = |x: Option<i32>, y: Option<&str>| {
        Arc::new(StructArray::new(
            fields.clone(),
            vec![
                Arc::new(Int32Array::from(vec![x])) as ArrayRef,
                Arc::new(StringArray::from(vec![y])) as ArrayRef,
            ],
            None,
        )) as ArrayRef
    };
    let (a, b) = (of(Some(1), Some("b")), of(Some(1), Some("c")));
    assert_eq!(
        cmp_rows(a.as_ref(), 0, b.as_ref(), 0).unwrap(),
        Ordering::Less
    );
    let (a, b) = (of(Some(2), Some("a")), of(Some(1), Some("z")));
    assert_eq!(
        cmp_rows(a.as_ref(), 0, b.as_ref(), 0).unwrap(),
        Ordering::Greater
    );
}

#[test]
fn values_render_like_sparks_tostring() {
    assert_eq!(render_value(i32s(vec![Some(42)]).as_ref(), 0), "42");
    assert_eq!(render_value(strs(vec![Some("k")]).as_ref(), 0), "k");
    assert_eq!(render_value(i32s(vec![None]).as_ref(), 0), "null");
    assert_eq!(
        render_value(
            &(Arc::new(BooleanArray::from(vec![Some(true)])) as ArrayRef),
            0
        ),
        "true"
    );
}
