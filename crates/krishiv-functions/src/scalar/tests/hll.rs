// SPDX-License-Identifier: Apache-2.0
//! `hll_sketch_estimate` and `hll_union`: the documented examples, NULLs, the error classes of Spark's goldens
//! (`hll.sql`: `HLL_INVALID_INPUT_SKETCH_BUFFER`, `HLL_UNION_DIFFERENT_LG_K`) and unions across lgK.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, BooleanArray};
use krishiv_common::error::ErrorClass;

use super::{bind, bytes, col};
use crate::sketch::hll;

fn sketch(lg_k: i32, values: &[i64]) -> Vec<u8> {
    let mut s = hll::new_sketch(lg_k).unwrap();
    for v in values {
        hll::update_long(&mut s, *v);
    }
    hll::to_bytes(&s)
}

fn flag(v: bool) -> ArrayRef {
    Arc::new(BooleanArray::from(vec![v]))
}

#[test]
fn estimate_counts_and_keeps_nulls() {
    let a = sketch(12, &[1, 1, 2, 2, 3]);
    let input = bytes(&[Some(&a), None]);
    let out = bind("hll_sketch_estimate", &[&input])
        .invoke(&[col(&input)], 2)
        .unwrap();
    let got: Vec<Option<i64>> = out.as_primitive::<Int64Type>().iter().collect();
    assert_eq!(got, vec![Some(3), None]);
}

#[test]
fn estimate_of_garbage_is_the_buffer_error() {
    let input = bytes(&[Some(b"abc")]);
    let e = bind("hll_sketch_estimate", &[&input])
        .invoke(&[col(&input)], 1)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::HllInvalidInputSketchBuffer);
}

#[test]
fn union_checks_lg_k_unless_allowed() {
    let a = bytes(&[Some(&sketch(12, &[1, 2]))]);
    let b = bytes(&[Some(&sketch(13, &[2, 3]))]);
    let no = flag(false);
    let f = bind("hll_union", &[&a, &b, &no]);
    let e = f.invoke(&[col(&a), col(&b), col(&no)], 1).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::HllUnionDifferentLgK);
    let yes = flag(true);
    let f = bind("hll_union", &[&a, &b, &yes]);
    let u = f.invoke(&[col(&a), col(&b), col(&yes)], 1).unwrap();
    let u = u.as_binary::<i32>();
    assert_eq!(hll::estimate(&hll::from_bytes(u.value(0), "x").unwrap()), 3);
    assert!(!u.is_null(0));
}

#[test]
fn union_of_garbage_is_the_buffer_error() {
    let a = bytes(&[Some(b"abc")]);
    let b = bytes(&[Some(b"def")]);
    let no = flag(false);
    let e = bind("hll_union", &[&a, &b, &no])
        .invoke(&[col(&a), col(&b), col(&no)], 1)
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::HllInvalidInputSketchBuffer);
}
