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

fn acc(f: &AggregateFunction, t: &DataType, ansi: bool, proofs: bool) -> Accumulator {
    let mut a = Accumulator::new(&AggregateExpr::new(f.clone(), 0, t.clone(), "a"), ansi).unwrap();
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
                        let mut a = acc(&f, &at, ansi, proofs);
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
                            let mut a = acc(&f, &at, ansi, false);
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
                            let mut top = acc(&f, &at, ansi, proofs);
                            top.resize(groups as usize);
                            for part in [&list[..half], &list[half..]] {
                                let mut mid = acc(&f, &at, ansi, proofs);
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
        let mut a = acc(&f, &t, true, true);
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

/// The moment aggregates: `(group, x, y, x is valid, y is valid)`.
type MomentRow = (u32, f64, f64, bool, bool);

fn moment_acc(f: &AggregateFunction, groups: u32) -> Accumulator {
    let args = if crate::aggregate::arity(f) == 2 {
        vec![0, 1]
    } else {
        vec![0]
    };
    let e = AggregateExpr {
        function: f.clone(),
        args,
        input: AggInput::Raw,
        arg_type: DataType::Float64,
        name: "m".to_owned(),
    };
    // Without ANSI: `corr` of a constant column is NULL, not an error.
    let mut a = Accumulator::new(&e, false).unwrap();
    a.resize(groups as usize);
    a
}

/// The columns of `rows` as the arguments of a moment aggregate (garbage
/// under the NULL slots).
fn moment_columns(rows: &[MomentRow]) -> Vec<ArrayRef> {
    let col = |value: fn(&MomentRow) -> f64, valid: fn(&MomentRow) -> bool| -> ArrayRef {
        let nulls = NullBuffer::from(rows.iter().map(valid).collect::<Vec<_>>());
        let v: Vec<f64> = rows
            .iter()
            .map(|r| if valid(r) { value(r) } else { f64::NAN })
            .collect();
        Arc::new(Float64Array::new(v.into(), Some(nulls)))
    };
    vec![col(|r| r.1, |r| r.3), col(|r| r.2, |r| r.4)]
}

fn moment_values(a: &Accumulator, groups: u32) -> Vec<Option<f64>> {
    let ids: Vec<u32> = (0..groups).collect();
    let out = a.evaluate(&ids).unwrap();
    out.as_primitive::<Float64Type>().iter().collect()
}

/// Keys of the `max_by`/`min_by` merge test: NaN (the largest), the
/// infinities, and `-0.0 = 0.0` (a tie).
const KEYS: [f64; 7] = [
    f64::NAN,
    f64::NEG_INFINITY,
    -1.0,
    -0.0,
    0.0,
    1.0,
    f64::INFINITY,
];

/// `(group, value, key)`: a NULL value is legal, a NULL key skips the row.
type ByRow = (u32, Option<i64>, Option<u8>);

/// The value and key columns of `rows`, as strings (`strings`) or as
/// `bigint`/`double` (keys by their index in strings, so ties stay ties).
fn by_columns(rows: &[ByRow], strings: (bool, bool)) -> Vec<ArrayRef> {
    let x: ArrayRef = if strings.0 {
        Arc::new(StringViewArray::from_iter(
            rows.iter().map(|r| r.1.map(|v| format!("v{v:03}"))),
        ))
    } else {
        Arc::new(Int64Array::from_iter(rows.iter().map(|r| r.1)))
    };
    let y: ArrayRef = if strings.1 {
        Arc::new(StringViewArray::from_iter(
            rows.iter().map(|r| r.2.map(|k| format!("k{k}"))),
        ))
    } else {
        Arc::new(Float64Array::from_iter(
            rows.iter().map(|r| r.2.map(|k| KEYS[usize::from(k)])),
        ))
    };
    vec![x, y]
}

fn by_expr(f: AggregateFunction, strings: (bool, bool)) -> AggregateExpr {
    let x = if strings.0 {
        DataType::Utf8View
    } else {
        DataType::Int64
    };
    let y = if strings.1 {
        DataType::Utf8View
    } else {
        DataType::Float64
    };
    match f {
        AggregateFunction::MaxBy | AggregateFunction::MinBy => {
            AggregateExpr::by(f, (0, 1), (x, y), "b")
        }
        _ => AggregateExpr::new(f, 0, x, "a"),
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 96,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    /// Partial states of any split of the input merge, in any order and
    /// through a second level, to the single pass's answer (up to the
    /// reassociation of the floating-point operations).
    #[test]
    fn moments_merge_differential(
        groups in 1..6_u32,
        rows in proptest::collection::vec(
            (0..6_u32, -1000.0..1000.0_f64, -1000.0..1000.0_f64,
             proptest::bool::weighted(0.85), proptest::bool::weighted(0.85)),
            0..160,
        ),
        parts in 1..7_usize,
        rotate in 0..7_usize,
        reverse in proptest::bool::ANY,
    ) {
        let rows: Vec<MomentRow> = rows.into_iter().map(|r| (r.0 % groups, r.1, r.2, r.3, r.4)).collect();
        let ids: Vec<u32> = (0..groups).collect();
        for f in MOMENT_FUNCTIONS {
            let mut once = moment_acc(&f, groups);
            let g: Vec<u32> = rows.iter().map(|r| r.0).collect();
            once.update(&g, &moment_columns(&rows), &Weights::Const(1)).unwrap();
            let mut states: Vec<Vec<ArrayRef>> = rows
                .chunks(rows.len().div_ceil(parts).max(1))
                .map(|chunk| {
                    let mut a = moment_acc(&f, groups);
                    let g: Vec<u32> = chunk.iter().map(|r| r.0).collect();
                    a.update(&g, &moment_columns(chunk), &Weights::Const(1)).unwrap();
                    a.state(&ids).unwrap()
                })
                .collect();
            let len = states.len().max(1);
            states.rotate_left(rotate % len);
            if reverse {
                states.reverse();
            }
            // Two levels: halves merged apart, then together.
            let (lo, hi) = states.split_at(states.len() / 2);
            let mut top = moment_acc(&f, groups);
            for half in [lo, hi] {
                let mut mid = moment_acc(&f, groups);
                for st in half {
                    mid.merge(&ids, st).unwrap();
                }
                top.merge(&ids, &mid.state(&ids).unwrap()).unwrap();
            }
            for (got, want) in moment_values(&top, groups).iter().zip(moment_values(&once, groups)) {
                match (got, want) {
                    (None, None) => {}
                    (Some(g), Some(w)) if g.is_nan() || w.is_nan() => {
                        prop_assert!(g.is_nan() && w.is_nan(), "{:?}: {} vs {}", f, g, w);
                    }
                    (Some(g), Some(w)) => {
                        prop_assert!((g - w).abs() <= 1e-8 * g.abs().max(w.abs()).max(1.0), "{:?}: {} vs {}", f, g, w);
                    }
                    (g, w) => prop_assert!(false, "{:?}: {:?} vs {:?}", f, g, w),
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    /// `max_by`, `min_by` and `any_value`: the input cut anywhere into
    /// consecutive parts, each part's partial states merged in order (also
    /// through a second level), give the single pass's states and result,
    /// ties and NULLs included.
    #[test]
    fn by_merge_differential(
        rows in proptest::collection::vec(
            (0..4_u32, proptest::option::of(0..40_i64), proptest::option::of(0..7_u8)),
            0..60,
        ),
        cuts in proptest::collection::vec(0..61_usize, 0..4),
        strings in (any::<bool>(), any::<bool>()),
    ) {
        let groups = 4_u32;
        let ids: Vec<u32> = (0..groups).collect();
        let mut cuts: Vec<usize> = cuts.into_iter().map(|c| c.min(rows.len())).collect();
        cuts.push(0);
        cuts.push(rows.len());
        cuts.sort_unstable();
        let fold = |range: &[ByRow]| {
            (range.iter().map(|r| r.0).collect::<Vec<u32>>(), by_columns(range, strings))
        };
        for f in [
            AggregateFunction::MaxBy,
            AggregateFunction::MinBy,
            AggregateFunction::AnyValue { ignore_nulls: false },
            AggregateFunction::AnyValue { ignore_nulls: true },
        ] {
            let expr = by_expr(f, strings);
            let new = || {
                let mut a = Accumulator::new(&expr, true).unwrap();
                a.resize(groups as usize);
                a
            };
            let mut single = new();
            let (g, cols) = fold(&rows);
            single.update(&g, &cols, &Weights::Const(1)).unwrap();
            let partials: Vec<Vec<ArrayRef>> = cuts
                .windows(2)
                .map(|w| {
                    let mut a = new();
                    let (g, cols) = fold(&rows[w[0]..w[1]]);
                    a.update(&g, &cols, &Weights::Const(1)).unwrap();
                    a.state(&ids).unwrap()
                })
                .collect();
            let mut flat = new();
            for st in &partials {
                flat.merge(&ids, st).unwrap();
            }
            // Two levels: the halves merged apart, then together.
            let half = partials.len() / 2;
            let mut top = new();
            for part in [&partials[..half], &partials[half..]] {
                let mut mid = new();
                for st in part {
                    mid.merge(&ids, st).unwrap();
                }
                top.merge(&ids, &mid.state(&ids).unwrap()).unwrap();
            }
            let want = (single.state(&ids).unwrap(), single.evaluate(&ids).unwrap());
            for merged in [&flat, &top] {
                prop_assert_eq!(
                    &(merged.state(&ids).unwrap(), merged.evaluate(&ids).unwrap()),
                    &want
                );
            }
        }
    }
}

const MOMENT_FUNCTIONS: [AggregateFunction; 9] = [
    AggregateFunction::StddevSamp,
    AggregateFunction::StddevPop,
    AggregateFunction::VarSamp,
    AggregateFunction::VarPop,
    AggregateFunction::Skewness,
    AggregateFunction::Kurtosis,
    AggregateFunction::Corr,
    AggregateFunction::CovarSamp,
    AggregateFunction::CovarPop,
];
