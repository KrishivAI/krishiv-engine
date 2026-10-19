// SPDX-License-Identifier: Apache-2.0
//! Differential tests of the decimal `sum`/`avg` per-batch proofs (design
//! P2-T160): the proof path against the per-row path, and `Decimal64`
//! inputs against `Decimal128` ones, over adversarial batches — up to 80
//! groups and 300 rows per batch, sliced columns, garbage (`i128::MIN`,
//! `i128::MAX`) under NULL slots, IVM weights of both signs including
//! `i64::MIN`/`i64::MAX`, ANSI on and off, and repeated, nested merges.

use super::*;
use proptest::prelude::*;

/// `(group, value, valid, weight, garbage under NULL)`.
type Row = (u32, i128, bool, i64, i128);

/// One batch: rows, and junk rows before them that the column slices off.
type Batch = (Vec<Row>, Vec<Row>);

fn decimal_type() -> impl Strategy<Value = (u8, i8)> {
    prop_oneof![
        Just((38, 0)),
        Just((28, 0)),
        Just((30, 5)),
        Just((18, 0)),
        Just((15, 2)),
        Just((8, 2)),
        Just((5, 2)),
        Just((1, 0)),
    ]
}

fn value(p: u8) -> impl Strategy<Value = i128> + Clone {
    let m = 10_i128.pow(u32::from(p)) - 1;
    let two63 = 1_i128 << 63;
    prop_oneof![
        Just(m),
        Just(-m),
        Just(m / 2 + 1),
        Just(1),
        Just(-5),
        Just(0),
        Just((two63 - 1).min(m)),
        Just((-two63).max(-m)),
        -1000..1000_i128,
        -m..=m,
    ]
}

fn weight() -> impl Strategy<Value = i64> + Clone {
    prop_oneof![
        4 => Just(1_i64),
        2 => Just(-1),
        1 => -3..=3_i64,
        1 => Just(i64::MAX),
        1 => Just(i64::MIN),
        1 => any::<i64>(),
    ]
}

fn garbage() -> impl Strategy<Value = i128> + Clone {
    prop_oneof![Just(i128::MIN), Just(i128::MAX), any::<i128>()]
}

/// `(type, groups, weights mode, batches)`.
type Input = ((u8, i8), u32, u8, Vec<Batch>);

fn input() -> impl Strategy<Value = Input> {
    (decimal_type(), 1..=80_u32, 0..3_u8).prop_flat_map(|(t, groups, mode)| {
        // Most rows land in a few groups, so running sums approach the
        // bound; the rest spread over all groups.
        let group = prop_oneof![2 => 0..groups.min(2), 1 => 0..groups];
        let row = (
            group,
            value(t.0),
            proptest::bool::weighted(0.85),
            weight(),
            garbage(),
        );
        // Short batches put a near-bound value alone in a batch, so the
        // next batch's proof depends on the running bound.
        let rows = prop_oneof![
            3 => proptest::collection::vec(row.clone(), 0..4),
            1 => proptest::collection::vec(row.clone(), 0..=300),
        ];
        let batch = (rows, proptest::collection::vec(row, 0..8));
        (
            Just(t),
            Just(groups),
            Just(mode),
            proptest::collection::vec(batch, 1..10),
        )
    })
}

/// The value column of a batch, sliced past its junk rows, on the
/// `Decimal64` carrier when `d64` (garbage saturates into `i64`).
fn column((rows, junk): &Batch, t: (u8, i8), d64: bool) -> ArrayRef {
    let all: Vec<&Row> = junk.iter().chain(rows).collect();
    let raw = |r: &Row| if r.2 { r.1 } else { r.4 };
    let nulls = NullBuffer::from(all.iter().map(|r| r.2).collect::<Vec<_>>());
    let a: ArrayRef = if d64 {
        let sat = |v: i128| i64::try_from(v).unwrap_or(if v < 0 { i64::MIN } else { i64::MAX });
        let v: Vec<i64> = all.iter().map(|r| sat(raw(r))).collect();
        Arc::new(
            PrimitiveArray::<Decimal64Type>::new(v.into(), Some(nulls))
                .with_data_type(DataType::Decimal64(t.0, t.1)),
        )
    } else {
        let v: Vec<i128> = all.iter().map(|r| raw(r)).collect();
        Arc::new(
            Decimal128Array::new(v.into(), Some(nulls))
                .with_data_type(DataType::Decimal128(t.0, t.1)),
        )
    };
    a.slice(junk.len(), rows.len())
}

/// The weights of a batch (mode 2: a sliced column).
fn weights((rows, junk): &Batch, mode: u8) -> Weights {
    match mode {
        0 => Weights::Const(1),
        1 => Weights::Const(rows.first().map_or(1, |r| r.3)),
        _ => {
            let all = Int64Array::from_iter_values(junk.iter().chain(rows).map(|r| r.3));
            Weights::Column(all.slice(junk.len(), rows.len()))
        }
    }
}

fn arg_type(t: (u8, i8), d64: bool) -> DataType {
    if d64 {
        DataType::Decimal64(t.0, t.1)
    } else {
        DataType::Decimal128(t.0, t.1)
    }
}

fn acc(f: AggregateFunction, t: &DataType, ansi: bool, proofs: bool) -> Accumulator {
    let mut a = Accumulator::new(&AggregateExpr::new(f, 0, t.clone(), "a"), ansi).unwrap();
    a.proofs = proofs;
    a
}

/// A decimal column as `Decimal128` (other columns as they are), so
/// `Decimal64` and `Decimal128` runs compare value for value.
fn widen(a: &ArrayRef) -> ArrayRef {
    match a.data_type() {
        DataType::Decimal64(p, s) => Arc::new(
            a.as_primitive::<Decimal64Type>()
                .unary::<_, Decimal128Type>(i128::from)
                .with_data_type(DataType::Decimal128(*p, *s)),
        ),
        _ => Arc::clone(a),
    }
}

type Outcome = (Vec<ArrayRef>, std::result::Result<ArrayRef, String>);

/// `state` and `evaluate` (value or error text) of every group, widened.
fn outcome(a: &Accumulator, groups: u32) -> Outcome {
    let ids: Vec<u32> = (0..groups).collect();
    (
        a.state(&ids).unwrap().iter().map(widen).collect(),
        a.evaluate(&ids)
            .map(|v| widen(&v))
            .map_err(|e| e.to_string()),
    )
}

/// The largest stored `|sum|`, which `max_sum` must bound.
fn stored_max(a: &Accumulator) -> u128 {
    match &a.kind {
        Kind::SumDec { sum, .. } | Kind::AvgDec { sum, .. } => {
            sum.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0)
        }
        _ => 0,
    }
}

/// The carriers a type can take as input.
fn carriers(t: (u8, i8)) -> Vec<bool> {
    if t.0 <= 18 {
        vec![false, true]
    } else {
        vec![false]
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 128,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    /// Raw rows: proof on == proof off, and `Decimal64` == `Decimal128`.
    #[test]
    fn fold_differential((t, groups, mode, batches) in input()) {
        for f in [AggregateFunction::Sum, AggregateFunction::Avg] {
            for ansi in [true, false] {
                let mut outcomes = Vec::new();
                for d64 in carriers(t) {
                    let at = arg_type(t, d64);
                    for proofs in [true, false] {
                        let mut a = acc(f, &at, ansi, proofs);
                        a.resize(groups as usize);
                        for b in &batches {
                            let g: Vec<u32> = b.0.iter().map(|r| r.0).collect();
                            a.update(&g, &[column(b, t, d64)], &weights(b, mode)).unwrap();
                            prop_assert!(stored_max(&a) <= a.max_sum);
                        }
                        outcomes.push(outcome(&a, groups));
                    }
                }
                for o in &outcomes[1..] {
                    prop_assert_eq!(o, &outcomes[0]);
                }
            }
        }
    }

    /// Partial states (overflowed ones included): each merged twice, in
    /// both orders, and through a second level of merges; proof on == off
    /// and `Decimal64` == `Decimal128`.
    #[test]
    fn merge_differential((t, groups, _mode, batches) in input()) {
        let ids: Vec<u32> = (0..groups).collect();
        for f in [AggregateFunction::Sum, AggregateFunction::Avg] {
            for ansi in [true, false] {
                let mut outcomes = Vec::new();
                for d64 in carriers(t) {
                    let at = arg_type(t, d64);
                    let partials: Vec<Vec<ArrayRef>> = batches
                        .iter()
                        .map(|b| {
                            let mut a = acc(f, &at, ansi, false);
                            a.resize(groups as usize);
                            let g: Vec<u32> = b.0.iter().map(|r| r.0).collect();
                            a.update(&g, &[column(b, t, d64)], &Weights::Const(1)).unwrap();
                            a.state(&ids).unwrap()
                        })
                        .collect();
                    for proofs in [true, false] {
                        for reverse in [false, true] {
                            let mut list: Vec<&Vec<ArrayRef>> =
                                partials.iter().chain(&partials).collect();
                            if reverse {
                                list.reverse();
                            }
                            // Two levels: halves merged apart, then together.
                            let half = list.len() / 2;
                            let mut top = acc(f, &at, ansi, proofs);
                            top.resize(groups as usize);
                            for part in [&list[..half], &list[half..]] {
                                let mut mid = acc(f, &at, ansi, proofs);
                                mid.resize(groups as usize);
                                for st in part {
                                    mid.merge(&ids, st).unwrap();
                                    prop_assert!(stored_max(&mid) <= mid.max_sum);
                                }
                                top.merge(&ids, &mid.state(&ids).unwrap()).unwrap();
                                prop_assert!(stored_max(&top) <= top.max_sum);
                            }
                            outcomes.push(outcome(&top, groups));
                        }
                    }
                }
                // Merge order may change whether a transient overflow is
                // seen (as in Spark); within one order every run agrees.
                for r in 0..2 {
                    let same: Vec<&Outcome> = outcomes.iter().skip(r).step_by(2).collect();
                    for o in &same[1..] {
                        prop_assert_eq!(*o, same[0]);
                    }
                }
            }
        }
    }
}

/// The result carrier follows the input's: `sum(decimal(8,2))` is
/// `decimal(18,2)` on `Decimal64`, `avg(decimal(15,2))` is `decimal(19,6)`
/// on `Decimal128`; `Decimal128` inputs keep `Decimal128` results.
#[test]
fn result_carrier_follows_the_input() {
    let ty = |f, t: DataType| AggregateExpr::new(f, 0, t, "a").result_type().unwrap();
    use AggregateFunction::{Avg, Sum};
    assert_eq!(
        ty(Sum, DataType::Decimal64(8, 2)),
        DataType::Decimal64(18, 2)
    );
    assert_eq!(
        ty(Sum, DataType::Decimal64(9, 2)),
        DataType::Decimal128(19, 2)
    );
    assert_eq!(
        ty(Avg, DataType::Decimal64(14, 2)),
        DataType::Decimal64(18, 6)
    );
    assert_eq!(
        ty(Avg, DataType::Decimal64(15, 2)),
        DataType::Decimal128(19, 6)
    );
    assert_eq!(
        ty(Sum, DataType::Decimal128(8, 2)),
        DataType::Decimal128(18, 2)
    );
    assert_eq!(
        ty(AggregateFunction::Min, DataType::Decimal64(8, 2)),
        DataType::Decimal64(8, 2)
    );
}

/// `min`/`max` over a `Decimal64` column keep the carrier and the values.
#[test]
fn min_max_on_decimal64() {
    let t = DataType::Decimal64(10, 2);
    for (f, want) in [
        (AggregateFunction::Min, -12_345_i64),
        (AggregateFunction::Max, 99_999),
    ] {
        let mut a = acc(f, &t, true, true);
        a.resize(1);
        let col: ArrayRef = Arc::new(
            PrimitiveArray::<Decimal64Type>::from(vec![Some(5), None, Some(-12_345), Some(99_999)])
                .with_data_type(t.clone()),
        );
        a.update(&[0, 0, 0, 0], &[col], &Weights::Const(1)).unwrap();
        let v = a.evaluate(&[0]).unwrap();
        assert_eq!(v.data_type(), &t);
        assert_eq!(v.as_primitive::<Decimal64Type>().value(0), want);
    }
}
