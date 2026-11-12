// SPDX-License-Identifier: Apache-2.0
//! Tests of the HyperLogLog accumulator.

use arrow_array::Int64Array;

use super::*;
use crate::aggregate::{Accumulator, AggInput, AggregateExpr, AggregateFunction};

fn acc(p: u32) -> Accumulator {
    let e = AggregateExpr {
        function: AggregateFunction::ApproxCountDistinct { p },
        args: vec![0],
        input: AggInput::Raw,
        arg_type: DataType::Int64,
        name: "h".to_owned(),
        mask: None,
        element: None,
    };
    let mut a = Accumulator::new(&e, true).unwrap();
    a.resize(1);
    a
}

fn estimate(p: u32, values: &[i64], parts: usize) -> i64 {
    let col = Arc::new(Int64Array::from(values.to_vec())) as ArrayRef;
    let mut top = acc(p);
    let mut at = 0;
    for chunk in values.chunks(values.len().div_ceil(parts).max(1)) {
        let mut a = acc(p);
        let col = col.slice(at, chunk.len());
        at += chunk.len();
        a.update(
            &vec![0; chunk.len()],
            &[col],
            &krishiv_types::batch::Weights::Const(1),
        )
        .unwrap();
        let states = a.state(&[0]).unwrap();
        top.merge(&[0], &states).unwrap();
    }
    let out = top.evaluate(&[0]).unwrap();
    out.as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .unwrap()
        .value(0)
}

/// Register merge is element-wise max, so partitioning never changes
/// the estimate; small cardinalities use linear counting exactly.
#[test]
fn count_distinct_merges_identically_and_within_error() {
    let p = precision_for(0.05).unwrap();
    let values: Vec<i64> = (0..200).collect();
    let one = estimate(p, &values, 1);
    let eight = estimate(p, &values, 8);
    assert_eq!(one, eight);
    assert!((one - 200).abs() <= 10, "{one}");
    // Duplicates do not count twice; empty is 0, never NULL.
    assert_eq!(estimate(p, &[1, 1, 2, 2, 2], 2), 2);
    assert_eq!(estimate(p, &[], 1), 0);
}
