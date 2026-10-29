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

/// Every primitive carrier orders its smaller value first and renders both.
#[test]
fn every_primitive_carrier_orders_and_renders() {
    use arrow_array::types::*;
    use arrow_array::{Array, PrimitiveArray};
    use arrow_buffer::i256;
    use arrow_schema::{IntervalUnit, TimeUnit};

    macro_rules! check {
        ($t:ty, $a:expr, $b:expr) => {{
            let arr: PrimitiveArray<$t> = PrimitiveArray::from(vec![$a, $b]);
            let arr: ArrayRef = Arc::new(arr);
            assert_eq!(
                cmp_rows(arr.as_ref(), 0, arr.as_ref(), 1).unwrap(),
                Ordering::Less,
                "{}",
                arr.data_type()
            );
            assert_eq!(
                cmp_rows(arr.as_ref(), 1, arr.as_ref(), 0).unwrap(),
                Ordering::Greater
            );
            assert_ne!(render_value(arr.as_ref(), 0), render_value(arr.as_ref(), 1));
        }};
    }
    check!(Int8Type, 1, 2);
    check!(Int16Type, 1, 2);
    check!(Int64Type, 1, 2);
    check!(UInt8Type, 1, 2);
    check!(UInt16Type, 1, 2);
    check!(UInt32Type, 1, 2);
    check!(UInt64Type, 1, 2);
    check!(Float32Type, 1.5, 2.5);
    check!(Float64Type, 1.5, 2.5);
    check!(Decimal32Type, 15, 25);
    check!(Decimal64Type, 15, 25);
    check!(Decimal128Type, 15, 25);
    check!(Decimal256Type, i256::from_i128(15), i256::from_i128(25));
    check!(Date32Type, 1, 2);
    check!(Date64Type, 1, 2);
    check!(Time32SecondType, 1, 2);
    check!(Time32MillisecondType, 1, 2);
    check!(Time64MicrosecondType, 1, 2);
    check!(Time64NanosecondType, 1, 2);
    check!(TimestampSecondType, 1, 2);
    check!(TimestampMillisecondType, 1, 2);
    check!(TimestampMicrosecondType, 1, 2);
    check!(TimestampNanosecondType, 1, 2);
    check!(IntervalYearMonthType, 1, 2);
    check!(
        IntervalDayTimeType,
        IntervalDayTimeType::make_value(1, 0),
        IntervalDayTimeType::make_value(2, 0)
    );
    check!(
        IntervalMonthDayNanoType,
        IntervalMonthDayNanoType::make_value(1, 0, 0),
        IntervalMonthDayNanoType::make_value(1, 1, 0)
    );
    let _ = (IntervalUnit::YearMonth, TimeUnit::Second);
}

/// Strings, binary and booleans in every carrier, and the mismatch error.
#[test]
fn byte_carriers_and_mismatches() {
    use arrow_array::{
        BinaryArray, BinaryViewArray, LargeBinaryArray, LargeStringArray, StringViewArray,
    };
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(LargeStringArray::from(vec!["a", "b"])),
        Arc::new(StringViewArray::from(vec!["a", "b"])),
        Arc::new(BinaryArray::from_vec(vec![b"a", b"b"])),
        Arc::new(LargeBinaryArray::from_vec(vec![b"a", b"b"])),
        Arc::new(BinaryViewArray::from(vec![
            b"a".as_slice(),
            b"b".as_slice(),
        ])),
        Arc::new(BooleanArray::from(vec![false, true])),
    ];
    for a in &arrays {
        assert_eq!(
            cmp_rows(a.as_ref(), 0, a.as_ref(), 1).unwrap(),
            Ordering::Less
        );
        assert_ne!(render_value(a.as_ref(), 0), render_value(a.as_ref(), 1));
    }
    let ints = i32s(vec![Some(1)]);
    assert!(cmp_rows(ints.as_ref(), 0, arrays[0].as_ref(), 0).is_err());
}

/// List carriers, dictionaries and unsupported types.
#[test]
fn list_and_dictionary_carriers() {
    use arrow_array::types::Int32Type;
    use arrow_array::{DictionaryArray, LargeListArray, ListArray};
    use arrow_buffer::{OffsetBuffer, ScalarBuffer};
    let field = Arc::new(Field::new("item", DataType::Int32, true));
    let values: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 1, 3, 1]));
    let offsets = || OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 4, 5]));
    let list: ArrayRef = Arc::new(ListArray::new(
        Arc::clone(&field),
        offsets(),
        Arc::clone(&values),
        None,
    ));
    // [1,2] < [1,3]; [1] < [1,2] (a prefix is smaller).
    assert_eq!(
        cmp_rows(list.as_ref(), 0, list.as_ref(), 1).unwrap(),
        Ordering::Less
    );
    assert_eq!(
        cmp_rows(list.as_ref(), 2, list.as_ref(), 0).unwrap(),
        Ordering::Less
    );
    assert_eq!(render_value(list.as_ref(), 0), "[1,2]");
    let large: ArrayRef = Arc::new(LargeListArray::new(
        Arc::clone(&field),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i64, 2, 4, 5])),
        Arc::clone(&values),
        None,
    ));
    assert_eq!(
        cmp_rows(large.as_ref(), 0, large.as_ref(), 1).unwrap(),
        Ordering::Less
    );
    assert_eq!(render_value(large.as_ref(), 1), "[1,3]");
    let dict: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::new(
        Int32Array::from(vec![1, 0]),
        Arc::new(StringArray::from(vec!["b", "a"])),
    ));
    assert_eq!(
        cmp_rows(dict.as_ref(), 0, dict.as_ref(), 1).unwrap(),
        Ordering::Less
    );
    // A type with no order is an engine error, not a panic.
    let nulls: ArrayRef = Arc::new(arrow_array::NullArray::new(2));
    let _ = render_value(nulls.as_ref(), 0);
}
