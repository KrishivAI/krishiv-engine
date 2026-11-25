// SPDX-License-Identifier: Apache-2.0
use arrow_array::{Float64Array, Int32Array, LargeBinaryArray, LargeStringArray, StringArray};

use super::*;

fn column(a: impl Array + 'static) -> ArrayRef {
    Arc::new(a)
}

/// The sketches of two groups after `arg`, whose rows go to the groups in turn.
fn sketches(arg: &ArrayRef) -> Vec<Vec<u8>> {
    let mut acc = CountMinAcc::new(2, 8, 3);
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
    let mut whole = CountMinAcc::new(2, 8, 3);
    whole.resize(1);
    whole.update(&[0; 4], &all, |_| true).expect("update");
    let mut merged = CountMinAcc::new(2, 8, 3);
    merged.resize(1);
    for part in [all.slice(0, 2), all.slice(2, 2)] {
        let mut partial = CountMinAcc::new(2, 8, 3);
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
    let mut acc = CountMinAcc::new(3, 100, 1);
    let projected = acc.projected(10);
    acc.reserve(10);
    acc.resize(10);
    assert!(projected >= 10 * 301 * 8);
    assert!(acc.size() >= 10 * 301 * 8 && acc.size() <= projected);
}

#[test]
fn foreign_input_is_an_error() {
    let mut acc = CountMinAcc::new(1, 4, 1);
    acc.resize(1);
    let one = column(Int32Array::from(vec![1]));
    assert!(acc.update(&[0, 0], &one, |_| true).is_err());
    assert!(acc.update(&[5], &one, |_| true).is_err());
    assert!(
        acc.update(&[0], &column(Float64Array::from(vec![1.0])), |_| true)
            .is_err()
    );
    assert!(
        acc.merge(&[0], &BinaryArray::from_iter_values([b"x"]))
            .is_err()
    );
    assert!(
        acc.merge(&[0, 0], &BinaryArray::from_iter_values([b"x"]))
            .is_err()
    );
    assert!(acc.rows(&[3]).is_err());
    // Rows of no counters (an infinite eps) refuse the first value.
    let mut empty = CountMinAcc::new(1, 0, 1);
    empty.resize(1);
    assert!(empty.update(&[0], &one, |_| true).is_err());
    assert!(empty.update(&[0], &one, |_| false).is_ok());
}
