// SPDX-License-Identifier: Apache-2.0
//! `bitmap_bucket_number`, `bitmap_bit_position` and `bitmap_count`: the documented examples (`123` is bucket 1, position 122;
//! `X'1010'` has 2 bits) and the edges of Spark's `BitmapExpressionUtils`: positive values count from 1, the rest from 0 with
//! truncating division, and `Long.MIN_VALUE` wraps.

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{ArrayRef, Int64Array};
use std::sync::Arc;

use super::{bytes, run};

fn longs(v: &[Option<i64>]) -> ArrayRef {
    Arc::new(Int64Array::from(v.to_vec()))
}

fn longs_of(a: &ArrayRef) -> Vec<Option<i64>> {
    a.as_primitive::<Int64Type>().iter().collect()
}

#[test]
fn bucket_and_position_follow_spark() {
    let v = [
        Some(123),
        Some(1),
        Some(32768),
        Some(32769),
        Some(0),
        Some(-1),
        Some(-32768),
        Some(-32769),
        Some(i64::MIN),
        Some(i64::MAX),
        None,
    ];
    let input = longs(&v);
    assert_eq!(
        longs_of(&run("bitmap_bucket_number", &[&input])),
        vec![
            Some(1),
            Some(1),
            Some(1),
            Some(2),
            Some(0),
            Some(0),
            Some(-1),
            Some(-1),
            Some(-281_474_976_710_656),
            Some(281_474_976_710_656),
            None
        ]
    );
    assert_eq!(
        longs_of(&run("bitmap_bit_position", &[&input])),
        vec![
            Some(122),
            Some(0),
            Some(32767),
            Some(0),
            Some(0),
            Some(1),
            Some(0),
            Some(1),
            Some(0),
            Some(32766),
            None
        ]
    );
}

#[test]
fn bitmap_count_is_the_popcount() {
    let b = bytes(&[Some(&[0x10, 0x10]), Some(&[]), Some(&[0xFF, 0x01]), None]);
    assert_eq!(
        longs_of(&run("bitmap_count", &[&b])),
        vec![Some(2), Some(0), Some(9), None]
    );
}
