// SPDX-License-Identifier: Apache-2.0
use arrow_array::{Float64Array, Int32Array, LargeBinaryArray, LargeStringArray, StringArray};

use super::*;

fn acc_of(depth: u32, width: u32, seed: i32) -> CountMinAcc {
    CountMinAcc::new(depth, width, seed).expect("a shape within the budget")
}

fn class(e: &krishiv_common::error::KrishivError) -> &'static str {
    e.error_class().name()
}

fn column(a: impl Array + 'static) -> ArrayRef {
    Arc::new(a)
}

/// The sketches of two groups after `arg`, whose rows go to the groups in turn.
fn sketches(arg: &ArrayRef) -> Vec<Vec<u8>> {
    let mut acc = acc_of(2, 8, 3);
    acc.resize(2);
    let groups: Vec<u32> = (0..arg.len()).map(|i| u32::from(i % 2 == 1)).collect();
    acc.update(&groups, arg, |_| true).expect("update");
    acc.rows(&[0, 1]).expect("rows")
}

#[test]
fn every_carrier_of_a_value_counts_alike() {
    let strings = ["a", "bb", "ccc"];
    let want = sketches(&column(StringArray::from_iter_values(strings)));
    assert_eq!(
        sketches(&column(LargeStringArray::from_iter_values(strings))),
        want
    );
    assert_eq!(
        sketches(&column(LargeBinaryArray::from_iter_values(strings))),
        want
    );
}

#[test]
fn partial_states_merge_into_the_sketch_of_all_rows() {
    let all = column(Int32Array::from(vec![Some(1), None, Some(2), Some(1)]));
    let mut whole = acc_of(2, 8, 3);
    whole.resize(1);
    whole.update(&[0; 4], &all, |_| true).expect("update");
    let mut merged = acc_of(2, 8, 3);
    merged.resize(1);
    for part in [all.slice(0, 2), all.slice(2, 2)] {
        let mut partial = acc_of(2, 8, 3);
        partial.resize(1);
        partial.update(&[0; 2], &part, |_| true).expect("update");
        let state = partial.state(&[0]).expect("state");
        let state = state[0].as_binary::<i32>();
        merged.merge(&[0], state).expect("merge");
    }
    assert_eq!(
        merged.rows(&[0]).expect("rows"),
        whole.rows(&[0]).expect("rows")
    );
    // A NULL state is skipped.
    merged
        .merge(&[0], &BinaryArray::from(vec![None::<&[u8]>]))
        .expect("null");
    assert_eq!(
        merged.rows(&[0]).expect("rows"),
        whole.rows(&[0]).expect("rows")
    );
}

#[test]
fn the_charge_covers_the_groups_it_adds() {
    let mut acc = acc_of(3, 100, 1);
    let projected = acc.projected(10);
    acc.reserve(10);
    acc.resize(10);
    assert!(projected >= 10 * 301 * 8);
    assert!(acc.size() >= 10 * 301 * 8 && acc.size() <= projected);
}

#[test]
fn foreign_input_is_an_error() {
    let mut acc = acc_of(1, 4, 1);
    acc.resize(1);
    let one = column(Int32Array::from(vec![1]));
    let float = column(Float64Array::from(vec![1.0]));
    let bad_state = BinaryArray::from_iter_values([b"x"]);
    let bad_state_twice = BinaryArray::from_iter_values([b"x"]);
    for e in [
        acc.update(&[0, 0], &one, |_| true).expect_err("length"),
        acc.update(&[5], &one, |_| true).expect_err("group"),
        acc.update(&[0], &float, |_| true).expect_err("type"),
        acc.merge(&[0], &bad_state).expect_err("shape"),
        acc.merge(&[0, 0], &bad_state_twice).expect_err("length"),
        acc.rows(&[3]).expect_err("group"),
    ] {
        assert_eq!(class(&e), "INTERNAL_ERROR", "{e}");
    }
    // Rows of no counters (an infinite eps) refuse the first value.
    let mut empty = acc_of(1, 0, 1);
    empty.resize(1);
    let e = empty.update(&[0], &one, |_| true).expect_err("empty rows");
    assert_eq!(class(&e), "KRISHIV_UNSUPPORTED_FEATURE", "{e}");
    // A masked-out row is never counted: the sketch stays the empty one.
    empty.update(&[0], &one, |_| false).expect("masked");
    let oracle = acc_of(1, 0, 1);
    let mut oracle = oracle;
    oracle.resize(1);
    assert_eq!(
        empty.rows(&[0]).expect("rows"),
        oracle.rows(&[0]).expect("rows")
    );
}

#[test]
fn a_depth_past_the_coefficient_budget_is_unsupported_not_allocated() {
    let e = CountMinAcc::new(50_000_000, 4, 1).expect_err("too deep");
    assert_eq!(class(&e), "KRISHIV_UNSUPPORTED_FEATURE", "{e}");
}

#[test]
fn the_charge_covers_a_growth_of_both_vectors() {
    let mut acc = acc_of(2, 16, 1);
    acc.reserve(3);
    acc.resize(3);
    // Spare capacity is small: the next reserve grows both vectors.
    let projected = acc.projected(50);
    acc.reserve(50);
    acc.resize(53);
    assert!(acc.size() >= 53 * 32 * 8);
    assert!(acc.size() <= projected, "{} > {projected}", acc.size());
}

#[test]
fn a_count_min_only_batch_of_states_splits_by_bytes() {
    use crate::aggregate::accumulator::{Accumulator, STATE_BATCH_BYTES, state_prefix};
    use crate::aggregate::{AggregateExpr, AggregateFunction};
    use arrow_schema::DataType;
    let expr = AggregateExpr::new(
        AggregateFunction::CountMinSketch {
            depth: 2,
            width: 1_000_000,
            seed: 1,
        },
        0,
        DataType::Int32,
        "cm",
    );
    let acc = Accumulator::new(&expr, false).expect("accumulator");
    let ids: Vec<u32> = (0..10).collect();
    let n = state_prefix(&[acc], &ids, &mut 0, STATE_BATCH_BYTES);
    // About 16 MB a group: four fit under 64 MiB.
    assert_eq!(n, 4);
}
