// SPDX-License-Identifier: Apache-2.0
//! `theta_sketch_estimate`, `theta_union`, `theta_intersection` and `theta_difference`: NULL rows, the error classes,
//! Spark's check order (size, then the first buffer, then the second) and a size per row.

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef};

use super::{bind, bytes, col, ints};
use crate::sketch::theta;

fn sketch(values: &[i64]) -> Vec<u8> {
    let mut s = theta::new_sketch(12, "theta_sketch_agg").unwrap();
    for v in values {
        theta::update_long(&mut s, *v);
    }
    theta::to_bytes(&theta::compact(&mut s))
}

fn est(b: &[u8]) -> i64 {
    theta::estimate(&theta::from_bytes(b, "x").unwrap())
}

#[test]
fn estimate_counts_and_keeps_nulls() {
    let a = sketch(&[1, 1, 2, 3]);
    let input = bytes(&[Some(&a), None]);
    let out = bind("theta_sketch_estimate", &[&input])
        .invoke(&[col(&input)], 2)
        .unwrap();
    let got: Vec<Option<i64>> = out.as_primitive::<Int64Type>().iter().collect();
    assert_eq!(got, vec![Some(3), None]);
}

#[test]
fn estimate_of_garbage_and_of_another_family_are_the_two_buffer_errors() {
    let input = bytes(&[Some(b"abc")]);
    let e = bind("theta_sketch_estimate", &[&input])
        .invoke(&[col(&input)], 1)
        .unwrap_err()
        .to_string();
    assert!(e.contains("THETA_INVALID_INPUT_SKETCH_BUFFER"), "{e}");
    assert!(!e.contains("_FAMILY"), "{e}");
    let kll = bytes(&[Some(&[2_u8, 1, 15, 12, 0, 0, 0, 0])]);
    let e = bind("theta_sketch_estimate", &[&kll])
        .invoke(&[col(&kll)], 1)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("THETA_INVALID_INPUT_SKETCH_BUFFER_FAMILY"),
        "{e}"
    );
    assert!(e.contains("KLL"), "{e}");
}

fn two(name: &str, a: &ArrayRef, b: &ArrayRef, lg: Option<&ArrayRef>, rows: usize) -> ArrayRef {
    let mut args = vec![a, b];
    let mut cols = vec![col(a), col(b)];
    if let Some(l) = lg {
        args.push(l);
        cols.push(col(l));
    }
    bind(name, &args).invoke(&cols, rows).unwrap()
}

#[test]
fn set_operations_with_null_rows() {
    let x = sketch(&[1, 2, 3, 4]);
    let y = sketch(&[3, 4, 5]);
    let a = bytes(&[Some(&x), None, Some(&x)]);
    let b = bytes(&[Some(&y), Some(&y), None]);
    let lg = ints(&[Some(12), Some(12), Some(12)]);
    let u = two("theta_union", &a, &b, Some(&lg), 3);
    let u = u.as_binary::<i32>();
    assert_eq!(est(u.value(0)), 5);
    assert!(u.is_null(1) && u.is_null(2));
    let i = two("theta_intersection", &a, &b, None, 3);
    assert_eq!(est(i.as_binary::<i32>().value(0)), 2);
    assert!(i.as_binary::<i32>().is_null(1));
    let d = two("theta_difference", &a, &b, None, 3);
    assert_eq!(est(d.as_binary::<i32>().value(0)), 2);
    assert!(d.as_binary::<i32>().is_null(2));
}

#[test]
fn union_checks_the_size_first_then_the_first_buffer_then_the_second() {
    let good = sketch(&[1]);
    let a = bytes(&[Some(b"junk")]);
    let b = bytes(&[Some(&good)]);
    let c = bytes(&[Some(&[2_u8, 1, 15, 12, 0, 0, 0, 0])]);
    let run = |a: &ArrayRef, b: &ArrayRef, lg: i32| {
        let l = ints(&[Some(lg)]);
        bind("theta_union", &[a, b, &l])
            .invoke(&[col(a), col(b), col(&l)], 1)
            .unwrap_err()
            .to_string()
    };
    // A bad size wins over a bad buffer.
    assert!(run(&a, &b, 3).contains("SKETCH_INVALID_LG_NOM_ENTRIES"));
    // The first buffer is read before the second: junk first, another family second is the plain error.
    let e = run(&a, &c, 12);
    assert!(!e.contains("_FAMILY"), "{e}");
    // Good first, another family second names the family.
    assert!(run(&b, &c, 12).contains("THETA_INVALID_INPUT_SKETCH_BUFFER_FAMILY"));
}

#[test]
fn the_union_size_is_checked_per_row() {
    let x = sketch(&[1, 2]);
    let a = bytes(&[Some(&x), Some(&x)]);
    let lg = ints(&[Some(12), Some(27)]);
    let e = bind("theta_union", &[&a, &a, &lg])
        .invoke(&[col(&a), col(&a), col(&lg)], 2)
        .unwrap_err()
        .to_string();
    assert!(e.contains("SKETCH_INVALID_LG_NOM_ENTRIES"), "{e}");
}
