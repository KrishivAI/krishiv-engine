// SPDX-License-Identifier: Apache-2.0
use std::sync::Arc;

use arrow_array::types::Int32Type;
use arrow_array::{Array, ArrayRef, BinaryArray, BinaryViewArray, Int64Array, ListArray};
use krishiv_functions::sketch::theta;

use super::*;

fn estimate_of(a: &ArrayRef, row: usize) -> i64 {
    let v = a.as_any().downcast_ref::<BinaryViewArray>().expect("view");
    theta::estimate(&theta::from_bytes(v.value(row), "t").expect("sketch"))
}

fn longs(v: &[Option<i64>]) -> ArrayRef {
    Arc::new(Int64Array::from(v.to_vec()))
}

#[test]
fn a_sketch_counts_distinct_values_per_group_and_skips_nulls() {
    let mut acc = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
    acc.resize(2);
    let arg = longs(&[Some(1), Some(1), Some(2), None, Some(5), Some(6)]);
    acc.update(&[0, 0, 0, 0, 1, 1], &arg, |i| i != 5).unwrap();
    let out = acc.evaluate(&[0, 1]).unwrap();
    assert_eq!(estimate_of(&out, 0), 2);
    assert_eq!(estimate_of(&out, 1), 1, "the filtered row is dropped");
}

#[test]
fn array_values_hash_as_one_value_and_an_empty_array_is_skipped() {
    let mut acc = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
    acc.resize(1);
    let list = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
        Some(vec![Some(1), Some(2)]),
        Some(vec![Some(1), Some(2)]),
        Some(vec![]),
        None,
        Some(vec![Some(3)]),
    ]);
    acc.update(&[0; 5], &(Arc::new(list) as ArrayRef), |_| true)
        .unwrap();
    assert_eq!(estimate_of(&acc.evaluate(&[0]).unwrap(), 0), 2);
}

#[test]
fn partials_merge_through_their_bytes() {
    let mut a = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
    let mut b = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
    a.resize(1);
    b.resize(1);
    a.update(&[0, 0, 0], &longs(&[Some(1), Some(2), Some(3)]), |_| true)
        .unwrap();
    b.update(&[0, 0], &longs(&[Some(3), Some(4)]), |_| true)
        .unwrap();
    let mut fin = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
    fin.resize(1);
    for p in [&a, &b] {
        let state = p.state(&[0]).unwrap().remove(0);
        let state = state
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap()
            .clone();
        fin.merge(&[0], &state).unwrap();
    }
    assert_eq!(estimate_of(&fin.evaluate(&[0]).unwrap(), 0), 4);
}

#[test]
fn an_intersection_with_no_input_is_the_empty_sketch_and_an_empty_state() {
    let mut acc = ThetaSketchAcc::new(ThetaSketchOp::Intersection).unwrap();
    acc.resize(1);
    let state = acc.state(&[0]).unwrap().remove(0);
    let state = state.as_any().downcast_ref::<BinaryArray>().unwrap();
    assert!(state.value(0).is_empty());
    assert_eq!(estimate_of(&acc.evaluate(&[0]).unwrap(), 0), 0);
    // A merge of that empty state changes nothing.
    acc.merge(&[0], state).unwrap();
    assert!(
        acc.state(&[0]).unwrap()[0]
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap()
            .value(0)
            .is_empty()
    );
}

#[test]
fn union_and_intersection_aggregates_fold_sketches() {
    let sketch = |values: &[i64]| {
        let mut acc = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
        acc.resize(1);
        let arg = longs(&values.iter().map(|v| Some(*v)).collect::<Vec<_>>());
        acc.update(&vec![0; values.len()], &arg, |_| true).unwrap();
        let out = acc.evaluate(&[0]).unwrap();
        out.as_any()
            .downcast_ref::<BinaryViewArray>()
            .unwrap()
            .value(0)
            .to_vec()
    };
    let col: ArrayRef = Arc::new(BinaryArray::from_iter(vec![
        Some(sketch(&[1, 2, 3])),
        Some(sketch(&[2, 3, 4])),
        None,
    ]));
    let mut u = ThetaSketchAcc::new(ThetaSketchOp::Union(12)).unwrap();
    u.resize(1);
    u.update(&[0, 0, 0], &col, |_| true).unwrap();
    assert_eq!(estimate_of(&u.evaluate(&[0]).unwrap(), 0), 4);
    let mut i = ThetaSketchAcc::new(ThetaSketchOp::Intersection).unwrap();
    i.resize(1);
    i.update(&[0, 0, 0], &col, |_| true).unwrap();
    assert_eq!(estimate_of(&i.evaluate(&[0]).unwrap(), 0), 2);
}

#[test]
fn a_bad_buffer_is_the_theta_error_and_an_out_of_range_size_is_refused() {
    let mut u = ThetaSketchAcc::new(ThetaSketchOp::Union(12)).unwrap();
    u.resize(1);
    let col: ArrayRef = Arc::new(BinaryArray::from_iter_values([b"junk".to_vec()]));
    let e = u.update(&[0], &col, |_| true).unwrap_err().to_string();
    assert!(e.contains("THETA_INVALID_INPUT_SKETCH_BUFFER"), "{e}");
    assert!(ThetaSketchAcc::new(ThetaSketchOp::Sketch(3)).is_err());
    assert!(ThetaSketchAcc::new(ThetaSketchOp::Union(27)).is_err());
}

#[test]
fn the_projection_covers_what_a_batch_adds() {
    let mut acc = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
    acc.resize(4);
    let before = acc.size();
    assert!(acc.projected(0, 0, false) >= before);
    let arg = longs(&(0..4000).map(Some).collect::<Vec<_>>());
    let projected = acc.projected(0, 4000, false);
    let groups: Vec<u32> = (0..4000).map(|i| i % 4).collect();
    acc.update(&groups, &arg, |_| true).unwrap();
    assert!(acc.size() <= projected, "{} > {projected}", acc.size());
    assert!(arg.len() == 4000);
}
