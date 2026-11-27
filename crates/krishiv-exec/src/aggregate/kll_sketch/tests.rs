// SPDX-License-Identifier: Apache-2.0
use std::sync::Arc;

use arrow_array::{ArrayRef, BinaryArray, BinaryViewArray, Float32Array, Float64Array, Int64Array};
use krishiv_functions::sketch::kll::{Kll, KllType};

use super::*;

fn longs(v: &[i64]) -> ArrayRef {
    Arc::new(Int64Array::from(v.to_vec()))
}

fn n_of(a: &ArrayRef, row: usize, ty: KllType) -> u64 {
    let v = a.as_any().downcast_ref::<BinaryViewArray>().expect("view");
    Kll::from_bytes(ty, v.value(row), "t").expect("sketch").n()
}

fn state_of(acc: &KllSketchAcc) -> BinaryArray {
    acc.state(&[0])
        .unwrap()
        .remove(0)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap()
        .clone()
}

#[test]
fn a_sketch_counts_rows_per_group_and_skips_filtered_rows() {
    let mut acc = KllSketchAcc::new(KllSketchOp::Sketch(KllType::Bigint, 200)).unwrap();
    acc.resize(2);
    acc.update(&[0, 0, 0, 1, 1], &longs(&[1, 2, 3, 4, 5]), |i| i != 4)
        .unwrap();
    let out = acc.evaluate(&[0, 1]).unwrap();
    assert_eq!(n_of(&out, 0, KllType::Bigint), 3);
    assert_eq!(n_of(&out, 1, KllType::Bigint), 1);
    assert!(!acc.takes_sketches());
    assert!(acc.size() > 0);
}

#[test]
fn float_and_double_columns_feed_their_sketches() {
    let mut f = KllSketchAcc::new(KllSketchOp::Sketch(KllType::Float, 200)).unwrap();
    f.resize(1);
    let a: ArrayRef = Arc::new(Float32Array::from(vec![1.0, 2.0]));
    f.update(&[0, 0], &a, |_| true).unwrap();
    assert_eq!(n_of(&f.evaluate(&[0]).unwrap(), 0, KllType::Float), 2);
    let mut d = KllSketchAcc::new(KllSketchOp::Sketch(KllType::Double, 200)).unwrap();
    d.resize(1);
    let a: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]));
    d.update(&[0, 0, 0], &a, |_| true).unwrap();
    assert_eq!(n_of(&d.evaluate(&[0]).unwrap(), 0, KllType::Double), 3);
}

#[test]
fn partials_merge_through_their_bytes_and_sketches_merge_aggregate() {
    let mut a = KllSketchAcc::new(KllSketchOp::Sketch(KllType::Bigint, 200)).unwrap();
    let mut b = KllSketchAcc::new(KllSketchOp::Sketch(KllType::Bigint, 200)).unwrap();
    a.resize(1);
    b.resize(1);
    a.update(&[0, 0], &longs(&[1, 2]), |_| true).unwrap();
    b.update(&[0, 0, 0], &longs(&[3, 4, 5]), |_| true).unwrap();
    let mut fin = KllSketchAcc::new(KllSketchOp::Sketch(KllType::Bigint, 200)).unwrap();
    fin.resize(1);
    fin.merge(&[0], &state_of(&a)).unwrap();
    fin.merge(&[0], &state_of(&b)).unwrap();
    assert_eq!(n_of(&fin.evaluate(&[0]).unwrap(), 0, KllType::Bigint), 5);

    // kll_merge_agg over the sketches the partials produced.
    let mut m = KllSketchAcc::new(KllSketchOp::Merge(KllType::Bigint, Some(200))).unwrap();
    assert!(m.takes_sketches());
    m.resize(1);
    let sketches: ArrayRef = Arc::new(BinaryArray::from_iter_values([
        state_of(&a).value(0).to_vec(),
        state_of(&b).value(0).to_vec(),
    ]));
    m.update(&[0, 0], &sketches, |_| true).unwrap();
    assert_eq!(n_of(&m.evaluate(&[0]).unwrap(), 0, KllType::Bigint), 5);
}

#[test]
fn the_memory_projection_grows_with_new_groups_and_rows() {
    let mut acc = KllSketchAcc::new(KllSketchOp::Sketch(KllType::Bigint, 200)).unwrap();
    acc.resize(1);
    acc.update(&[0; 100], &longs(&[7; 100]), |_| true).unwrap();
    assert!(acc.state_bound(0) > 0);
    assert!(acc.projected(10, 1000, false, 0) > acc.projected(0, 10, false, 0));
    // The projection made before the groups and rows arrive covers what `size()` then reports.
    let mut grown = KllSketchAcc::new(KllSketchOp::Sketch(KllType::Bigint, 200)).unwrap();
    let rows = 5000;
    let bound = grown.projected(64, rows, false, 0);
    grown.reserve(64);
    grown.resize(64);
    let groups: Vec<u32> = (0..rows).map(|i| u32::try_from(i % 64).unwrap()).collect();
    let values: Vec<i64> = (0..i64::try_from(rows).unwrap()).collect();
    grown.update(&groups, &longs(&values), |_| true).unwrap();
    assert!(
        bound >= grown.size(),
        "projected {bound} < size {}",
        grown.size()
    );
}

#[test]
fn a_mismatched_column_length_and_bad_state_bytes_are_errors() {
    let mut acc = KllSketchAcc::new(KllSketchOp::Sketch(KllType::Bigint, 200)).unwrap();
    acc.resize(1);
    let len_err = acc.update(&[0], &longs(&[1, 2]), |_| true).unwrap_err();
    assert!(len_err.to_string().contains("INTERNAL_ERROR"), "{len_err}");
    let junk = BinaryArray::from_iter_values([vec![1u8, 2, 3]]);
    let err = acc.merge(&[0], &junk).unwrap_err();
    assert!(
        err.to_string().contains("KLL_INVALID_INPUT_SKETCH_BUFFER"),
        "{err}"
    );
}
