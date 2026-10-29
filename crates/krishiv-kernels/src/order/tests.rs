// SPDX-License-Identifier: Apache-2.0
//! Tests for the canonical key encodings and total row orderings: each
//! Arrow carrier encodes distinctly and equally for equal Spark values,
//! and [`super::cmp_rows`] orders NULLs, NaNs and nested values as Spark
//! does.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray, Int32Array, StringArray, StructArray};
use arrow_schema::{DataType, Field, Fields};

use super::*;

fn i32s(values: Vec<Option<i32>>) -> ArrayRef {
    Arc::new(Int32Array::from(values))
}

fn strs(values: Vec<Option<&str>>) -> ArrayRef {
    Arc::new(StringArray::from(values))
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
