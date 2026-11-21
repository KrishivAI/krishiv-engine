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

/// Feeds `n` values to group 0, one row per batch, and checks every batch against its projection.
fn one_row_batches_stay_under_the_projection(lg: u8, groups: u32, n: i64) {
    let mut acc = ThetaSketchAcc::new(ThetaSketchOp::Sketch(lg)).unwrap();
    let width = groups as usize;
    acc.resize(width);
    for v in 0..n {
        let arg = longs(&vec![Some(v); width]);
        let ids: Vec<u32> = (0..groups).collect();
        let projected = acc.projected(0, width, false, 0);
        acc.update(&ids, &arg, |_| true).unwrap();
        assert!(
            acc.size() <= projected,
            "lg {lg} row {v}: {} > {projected}",
            acc.size()
        );
    }
}

#[test]
fn a_row_that_crosses_the_resize_edge_is_projected() {
    one_row_batches_stay_under_the_projection(12, 1, 3_000);
    one_row_batches_stay_under_the_projection(5, 3, 200);
}

#[test]
fn new_groups_with_many_rows_are_projected() {
    for rows_per_group in [1_u32, 5, 65, 300] {
        let groups = 7_u32;
        let mut acc = ThetaSketchAcc::new(ThetaSketchOp::Sketch(12)).unwrap();
        let before = acc.size();
        let n = groups * rows_per_group;
        let projected = acc.projected(groups as usize, n as usize, false, 0);
        assert!(projected >= before);
        acc.resize(groups as usize);
        let ids: Vec<u32> = (0..n).map(|i| i % groups).collect();
        let arg = longs(&(0..i64::from(n)).map(Some).collect::<Vec<_>>());
        acc.update(&ids, &arg, |_| true).unwrap();
        assert!(
            acc.size() <= projected,
            "{rows_per_group}: {} > {projected}",
            acc.size()
        );
    }
}

#[test]
fn an_intersection_is_projected_by_its_input_not_by_the_default_size() {
    // Sketches built at lgNomEntries 16 in estimation mode: an intersection's table follows its first input.
    let mut big = theta::new_sketch(16, "theta_sketch_agg").unwrap();
    for v in 0..300_000_i64 {
        theta::update_long(&mut big, v);
    }
    let bytes = theta::to_bytes(&theta::compact(&mut big));
    let col: ArrayRef = Arc::new(BinaryArray::from_iter_values((0..4).map(|_| bytes.clone())));
    let state_bytes = bytes.len() * 4;
    let mut acc = ThetaSketchAcc::new(ThetaSketchOp::Intersection).unwrap();
    acc.resize(4);
    let projected = acc.projected(0, 4, true, state_bytes);
    acc.update(&[0, 1, 2, 3], &col, |_| true).unwrap();
    assert!(acc.size() <= projected, "{} > {projected}", acc.size());
    assert!(
        projected > 4 * (8 << 17),
        "the 64 KiB default table is not the bound"
    );
}

#[test]
fn state_batches_over_two_gib_are_an_error_not_a_wrapped_array() {
    assert!(binary_array(vec![vec![0_u8; 8], vec![1_u8; 8]]).is_ok());
    let max = i32::MAX as usize;
    // Lengths only: nothing of that size is allocated.
    assert!(check_total_len([max].into_iter()).is_ok());
    assert!(check_total_len([max - 1, 1].into_iter()).is_ok());
    assert!(check_total_len([max, 1].into_iter()).is_err());
    assert!(check_total_len([usize::MAX, 1].into_iter()).is_err());
}

/// A raw sketch column (the first stage of `theta_union_agg(s)` / `theta_intersection_agg(s)`, not partial
/// states) is charged by the bytes of its sketches, through the `Accumulator` the row table calls.
#[test]
fn a_raw_sketch_column_is_projected_by_its_bytes_for_union_and_intersection() {
    use super::super::accumulator::Accumulator;
    use super::super::{AggregateExpr, AggregateFunction};
    use arrow_schema::DataType;

    let mut big = theta::new_sketch(20, "theta_sketch_agg").unwrap();
    for v in 0..1_000_000_i64 {
        theta::update_long(&mut big, v);
    }
    let bytes = theta::to_bytes(&theta::compact(&mut big));
    let col: ArrayRef = Arc::new(BinaryArray::from_iter_values((0..4).map(|_| bytes.clone())));
    for function in [
        AggregateFunction::ThetaUnionAgg { lg_nom: 12 },
        AggregateFunction::ThetaIntersectionAgg,
    ] {
        let expr = AggregateExpr::new(function.clone(), 0, DataType::Binary, "t".to_owned());
        let mut acc = Accumulator::new(&expr, false).unwrap();
        acc.resize(1);
        let projected = acc.projected(0, Some(&col), None, false);
        let overhead_only = acc.projected(0, None, None, false);
        assert!(
            projected > 10 * overhead_only,
            "{function:?}: the sketch bytes are not charged ({projected} vs {overhead_only})"
        );
        if matches!(function, AggregateFunction::ThetaIntersectionAgg) {
            // Not capped by lgNomEntries: follows the input.
            assert!(projected > 4 * bytes.len(), "{function:?}: {projected}");
        }
    }
}
