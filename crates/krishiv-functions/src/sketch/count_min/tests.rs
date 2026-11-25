// SPDX-License-Identifier: Apache-2.0
use super::*;

/// The hex of `sketch` after counting nothing or the given items.
fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02X}");
        s
    })
}

#[test]
fn the_size_follows_eps_and_confidence_as_spark() {
    assert_eq!(width(0.5), 4);
    assert_eq!(width(0.2), 10);
    assert_eq!(width(2.0), 1);
    assert_eq!(width(1e-12), 2_147_483_647);
    assert_eq!(width(f64::NAN), 0);
    assert_eq!(width(f64::INFINITY), 0);
    assert_eq!(depth(0.5), 1);
    assert_eq!(depth(0.9), 4);
    assert_eq!(depth(0.99), 7);
    assert_eq!(depth(f64::NAN), 0);
}

#[test]
fn a_shape_past_two_gibibytes_does_not_serialize() {
    assert_eq!(serialized_len(1, 4), Some(60));
    assert_eq!(serialized_len(0, 4), Some(20));
    assert_eq!(serialized_len(1, 2_147_483_647), None);
    assert_eq!(serialized_len(u32::MAX, u32::MAX), None);
}

#[test]
fn one_integer_serializes_as_spark_4_3_0() {
    // `count_min_sketch(c, 0.5d, 0.5d, 1)` over the single value 1.
    let shape = Shape::new(depth(0.5), width(0.5), 1);
    let mut counters = vec![0; shape.cells()];
    shape.add_long(&mut counters, 1);
    assert_eq!(
        hex(&shape.serialize(1, &counters)),
        "0000000100000000000000010000000100000004000000005D8D6AB9\
         0000000000000000000000000000000100000000000000000000000000000000"
    );
}

#[test]
fn a_merge_adds_a_sketch_of_the_same_shape_only() {
    let shape = Shape::new(2, 5, 7);
    let mut counters = vec![0; shape.cells()];
    shape.add_bytes(&mut counters, b"abcde");
    shape.add_long(&mut counters, -3);
    let bytes = shape.serialize(2, &counters);
    let (mut total, mut into) = (2, counters.clone());
    assert!(shape.merge(&mut total, &mut into, &bytes));
    assert_eq!(total, 4);
    assert!(into.iter().zip(&counters).all(|(a, b)| *a == 2 * b));
    for other in [
        Shape::new(2, 5, 8),
        Shape::new(1, 10, 7),
        Shape::new(2, 4, 7),
    ] {
        let foreign = other.serialize(1, &vec![0; other.cells()]);
        let (mut total, mut into) = (4, counters.clone());
        assert!(!shape.merge(&mut total, &mut into, &foreign));
        assert_eq!((total, &into), (4, &counters));
    }
    assert!(!shape.merge(&mut total, &mut into, &bytes[..bytes.len() - 1]));
}

#[test]
fn a_shape_with_no_rows_counts_nothing() {
    let shape = Shape::new(0, 4, 1);
    assert!(!shape.has_empty_rows());
    shape.add_long(&mut [], 1);
    shape.add_bytes(&mut [], b"x");
    assert_eq!(shape.serialize(2, &[]).len(), 20);
    assert!(Shape::new(1, 0, 1).has_empty_rows());
}
