// SPDX-License-Identifier: Apache-2.0
use std::sync::Arc;

use arrow_array::{ArrayRef, BinaryArray, BinaryViewArray, Float64Array, Int32Array, Int64Array};
use krishiv_functions::sketch::tuple::{self, Mode};

use super::*;

fn keys(v: &[i64]) -> ArrayRef {
    Arc::new(Int64Array::from(v.to_vec()))
}

fn doubles(v: &[f64]) -> ArrayRef {
    Arc::new(Float64Array::from(v.to_vec()))
}

fn estimate(a: &ArrayRef, row: usize) -> f64 {
    let v = a.as_any().downcast_ref::<BinaryViewArray>().expect("view");
    tuple::estimate(&tuple::from_bytes::<f64>(v.value(row), "t").expect("sketch"))
}

/// The `Mode::Sum` total of every retained summary of a row's sketch.
fn summary_sum(a: &ArrayRef, row: usize) -> f64 {
    let v = a.as_any().downcast_ref::<BinaryViewArray>().expect("view");
    tuple::summary_double(
        &tuple::from_bytes::<f64>(v.value(row), "t").expect("sketch"),
        Mode::Sum,
    )
}

fn state_of(acc: &TupleSketchAcc) -> BinaryArray {
    acc.state(&[0])
        .unwrap()
        .remove(0)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap()
        .clone()
}

fn sketch_acc(lg: u8) -> TupleSketchAcc {
    let mut acc = TupleSketchAcc::new(TupleSketchOp::Sketch, false, lg, Mode::Sum).unwrap();
    acc.resize(1);
    acc
}

#[test]
fn a_sketch_counts_distinct_keys_per_group_and_skips_filtered_rows() {
    let mut acc = TupleSketchAcc::new(TupleSketchOp::Sketch, false, 12, Mode::Sum).unwrap();
    acc.resize(2);
    let k = keys(&[1, 1, 2, 3, 4]);
    let v = doubles(&[1.0; 5]);
    acc.update(&[0, 0, 0, 1, 1], &[&k, &v], |i| i != 4).unwrap();
    let out = acc.evaluate(&[0, 1]).unwrap();
    assert_eq!(estimate(&out, 0), 2.0);
    assert_eq!(estimate(&out, 1), 1.0);
    assert!(!acc.takes_sketches());
    assert!(acc.size() > 0);
    assert!(acc.state_bound(0) > 0);
    assert!(acc.projected(4, 100, true, 1000) >= acc.projected(0, 1, false, 0));
    acc.reserve(3);
}

#[test]
fn integer_summaries_work_too() {
    let mut acc = TupleSketchAcc::new(TupleSketchOp::Sketch, true, 12, Mode::Sum).unwrap();
    acc.resize(1);
    let k = keys(&[1, 2, 2]);
    let v: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
    acc.update(&[0; 3], &[&k, &v], |_| true).unwrap();
    let out = acc.evaluate(&[0]).unwrap();
    let bytes = out
        .as_any()
        .downcast_ref::<BinaryViewArray>()
        .unwrap()
        .value(0);
    let c = tuple::from_bytes::<i32>(bytes, "t").unwrap();
    assert_eq!(tuple::estimate(&c), 2.0);
    assert_eq!(tuple::summary_integer(&c, Mode::Sum), 6);
}

#[test]
fn partials_merge_and_the_union_and_intersection_aggregates_read_sketches() {
    let mut a = sketch_acc(12);
    let mut b = sketch_acc(12);
    a.update(&[0; 3], &[&keys(&[1, 2, 3]), &doubles(&[1.0; 3])], |_| true)
        .unwrap();
    b.update(&[0; 3], &[&keys(&[3, 4, 5]), &doubles(&[1.0; 3])], |_| true)
        .unwrap();

    let mut fin = sketch_acc(12);
    fin.merge(&[0], &state_of(&a)).unwrap();
    fin.merge(&[0], &state_of(&b)).unwrap();
    assert_eq!(estimate(&fin.evaluate(&[0]).unwrap(), 0), 5.0);

    let sketches: ArrayRef = Arc::new(BinaryArray::from_iter_values([
        state_of(&a).value(0).to_vec(),
        state_of(&b).value(0).to_vec(),
    ]));
    let mut u = TupleSketchAcc::new(TupleSketchOp::Union, false, 12, Mode::Sum).unwrap();
    assert!(u.takes_sketches());
    u.resize(1);
    u.update(&[0, 0], &[&sketches], |_| true).unwrap();
    assert_eq!(estimate(&u.evaluate(&[0]).unwrap(), 0), 5.0);
    assert_eq!(summary_sum(&u.evaluate(&[0]).unwrap(), 0), 6.0);
    // A union after a merge keeps raw sketches pending and still counts them.
    u.merge(&[0], &state_of(&a)).unwrap();
    assert_eq!(estimate(&u.evaluate(&[0]).unwrap(), 0), 5.0);
    // Merging `a` again adds its three summaries: a merge that dropped it would stay at 6.
    assert_eq!(summary_sum(&u.evaluate(&[0]).unwrap(), 0), 9.0);

    let mut i = TupleSketchAcc::new(TupleSketchOp::Intersection, false, 12, Mode::Sum).unwrap();
    i.resize(1);
    i.update(&[0, 0], &[&sketches], |_| true).unwrap();
    assert_eq!(estimate(&i.evaluate(&[0]).unwrap(), 0), 1.0);
    assert!(!state_of(&i).value(0).is_empty());
}

#[test]
fn an_intersection_that_saw_nothing_is_empty_bytes_and_a_sketch_group_is_not_cut() {
    let mut i = TupleSketchAcc::new(TupleSketchOp::Intersection, false, 12, Mode::Sum).unwrap();
    i.resize(1);
    assert!(state_of(&i).value(0).is_empty());

    let mut acc = sketch_acc(4);
    let ks: Vec<i64> = (0..1000).collect();
    acc.update(&[0; 1000], &[&keys(&ks), &doubles(&[1.0; 1000])], |_| true)
        .unwrap();
    let out = acc.evaluate(&[0]).unwrap();
    let bytes = out
        .as_any()
        .downcast_ref::<BinaryViewArray>()
        .unwrap()
        .value(0);
    let c = tuple::from_bytes::<f64>(bytes, "t").unwrap();
    // A never-merged accumulator is not reachable from a query (the final stage always merges and cuts), so
    // this pins only the accumulator: its result is not cut to 2^lgNomEntries.
    assert!(c.num_retained() > 16, "{}", c.num_retained());
}

#[test]
fn a_never_merged_sketch_keeps_its_estimation_mode_entries() {
    // Spark 4.3.0: tuple_sketch_agg_double(id, 1.0D, 5) over range(100) keeps 51 entries (more than 2^5).
    let mut acc = sketch_acc(5);
    let ks: Vec<i64> = (0..100).collect();
    acc.update(&[0; 100], &[&keys(&ks), &doubles(&[1.0; 100])], |_| true)
        .unwrap();
    let out = acc.evaluate(&[0]).unwrap();
    let bytes = out
        .as_any()
        .downcast_ref::<BinaryViewArray>()
        .unwrap()
        .value(0);
    let c = tuple::from_bytes::<f64>(bytes, "t").unwrap();
    // Spark keeps 51 entries at theta 0.3564956426418521 (the library's rebuild threshold decides both).
    assert_eq!(c.num_retained(), 51);
    assert!(
        (c.theta() - 0.356_495_642_641_852_1).abs() < 1e-15,
        "{}",
        c.theta()
    );
}

#[test]
fn bad_arguments_and_state_bytes_are_errors() {
    let class = |e: krishiv_common::error::KrishivError, want: &str| {
        assert!(e.to_string().contains(want), "want {want}: {e}");
    };
    class(
        TupleSketchAcc::new(TupleSketchOp::Sketch, false, 3, Mode::Sum).unwrap_err(),
        "SKETCH_INVALID_LG_NOM_ENTRIES",
    );
    let mut acc = sketch_acc(12);
    // Wrong lengths and the wrong number of argument columns.
    class(
        acc.update(&[0], &[&keys(&[1, 2]), &doubles(&[1.0, 2.0])], |_| true)
            .unwrap_err(),
        "INTERNAL_ERROR",
    );
    class(
        acc.update(&[0], &[&keys(&[1])], |_| true).unwrap_err(),
        "INTERNAL_ERROR",
    );
    let junk = BinaryArray::from_iter_values([vec![1u8, 2, 3]]);
    class(
        acc.merge(&[0], &junk).unwrap_err(),
        "TUPLE_INVALID_INPUT_SKETCH_BUFFER_FAMILY",
    );
    // A summary column of a type the aggregate cannot read.
    class(
        acc.update(&[0], &[&keys(&[1]), &keys(&[1])], |_| true)
            .unwrap_err(),
        "INTERNAL_ERROR",
    );
}
