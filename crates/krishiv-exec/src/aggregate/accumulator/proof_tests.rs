// SPDX-License-Identifier: Apache-2.0
//! The per-batch decimal proofs (design P2-T160) against the per-row path.

use super::*;
use proptest::prelude::*;

/// `10^38 - 1`, Spark's largest decimal.
const M: i128 = 99_999_999_999_999_999_999_999_999_999_999_999_999;

type Row = (u32, i128, bool, i64);

fn acc(f: &AggregateFunction, t: &DataType, ansi: bool, proofs: bool) -> Accumulator {
    let mut a = Accumulator::new(&AggregateExpr::new(f.clone(), 0, t.clone(), "a"), ansi).unwrap();
    a.proofs = proofs;
    a
}

fn column(rows: &[Row], t: &DataType) -> ArrayRef {
    let v = rows.iter().map(|&(_, v, ok, _)| ok.then_some(v));
    Arc::new(Decimal128Array::from_iter(v).with_data_type(t.clone()))
}

fn weights(rows: &[Row], mode: u8) -> Weights {
    match mode {
        0 => Weights::Const(1),
        1 => Weights::Const(rows.first().map_or(1, |r| r.3)),
        _ => Weights::Column(Int64Array::from_iter_values(rows.iter().map(|r| r.3))),
    }
}

fn feed(a: &mut Accumulator, groups: u32, batch: &[Row], t: &DataType, mode: u8) {
    a.resize(groups as usize);
    let g: Vec<u32> = batch.iter().map(|r| r.0).collect();
    a.update(&g, &[column(batch, t)], &weights(batch, mode))
        .unwrap();
}

/// `state` and `evaluate` (value or error text) of every group.
fn outcome(a: &Accumulator, groups: u32) -> (Vec<ArrayRef>, std::result::Result<ArrayRef, String>) {
    let ids: Vec<u32> = (0..groups).collect();
    (
        a.state(&ids).unwrap(),
        a.evaluate(&ids).map_err(|e| e.to_string()),
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

fn decimal_type() -> impl Strategy<Value = (u8, i8)> {
    prop_oneof![
        Just((38, 0)),
        Just((28, 0)),
        Just((5, 2)),
        Just((1, 0)),
        Just((18, 0)),
        Just((30, 5)),
        Just((28, 18)),
    ]
}

fn value(p: u8) -> impl Strategy<Value = i128> + Clone {
    let m = 10_i128.pow(u32::from(p)) - 1;
    prop_oneof![
        Just(m),
        Just(-m),
        Just(m / 2 + 1),
        Just(1),
        Just(-5),
        Just(0),
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

/// `(input type, groups, weights mode, batches)`.
type Input = ((u8, i8), u32, u8, Vec<Vec<Row>>);

fn input() -> impl Strategy<Value = Input> {
    (decimal_type(), 1..=3_u32, 0..3_u8).prop_flat_map(|(t, groups, mode)| {
        let row = (
            0..groups,
            value(t.0),
            proptest::bool::weighted(0.9),
            weight(),
        );
        // Short batches put a near-bound value alone in a batch, so
        // the next batch's proof depends on the running bound.
        let batch = prop_oneof![
            proptest::collection::vec(row.clone(), 0..3),
            proptest::collection::vec(row, 0..24),
        ];
        (
            Just(t),
            Just(groups),
            Just(mode),
            proptest::collection::vec(batch, 1..8),
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig { failure_persistence: None, ..ProptestConfig::default() })]

    /// Raw rows (with IVM weights of both signs): the proof path equals
    /// the per-row path for `sum` and `avg`, ANSI on and off.
    #[test]
    fn fold_proof_equals_per_row(((p, s), groups, mode, batches) in input()) {
        let t = DataType::Decimal128(p, s);
        for f in [&AggregateFunction::Sum, &AggregateFunction::Avg] {
            for ansi in [true, false] {
                let (mut fast, mut slow) = (acc(f, &t, ansi, true), acc(f, &t, ansi, false));
                for b in &batches {
                    feed(&mut fast, groups, b, &t, mode);
                    feed(&mut slow, groups, b, &t, mode);
                    prop_assert!(stored_max(&fast) <= fast.max_sum);
                    prop_assert!(stored_max(&slow) <= slow.max_sum);
                }
                prop_assert_eq!(outcome(&fast, groups), outcome(&slow, groups));
            }
        }
    }

    /// Merged partial states, overflowed ones included (sum NULL with a
    /// nonempty flag or a count): the proof path equals the per-row path
    /// in both merge orders.
    #[test]
    fn merge_proof_equals_per_row(((p, s), groups, _mode, batches) in input()) {
        let t = DataType::Decimal128(p, s);
        for f in [&AggregateFunction::Sum, &AggregateFunction::Avg] {
            for ansi in [true, false] {
                let ids: Vec<u32> = (0..groups).collect();
                let partials: Vec<Vec<ArrayRef>> = batches
                    .iter()
                    .map(|b| {
                        let mut a = acc(f, &t, ansi, false);
                        feed(&mut a, groups, b, &t, 0);
                        a.state(&ids).unwrap()
                    })
                    .collect();
                for order in [false, true] {
                    let (mut fast, mut slow) =
                        (acc(f, &t, ansi, true), acc(f, &t, ansi, false));
                    fast.resize(groups as usize);
                    slow.resize(groups as usize);
                    let mut list: Vec<&Vec<ArrayRef>> = partials.iter().collect();
                    if order {
                        list.reverse();
                    }
                    for st in list {
                        fast.merge(&ids, st).unwrap();
                        slow.merge(&ids, st).unwrap();
                    }
                    prop_assert_eq!(outcome(&fast, groups), outcome(&slow, groups));
                }
            }
        }
    }

    /// One `HALF_UP` division equals Spark's two roundings (scale 39,
    /// then the result scale) whenever it applies.
    #[test]
    fn avg_single_division_equals_digit_loop(
        sum in prop_oneof![Just(M), Just(-M), Just(M / 3), -M..=M, -100_000..100_000_i128],
        ss in 0..=38_i8,
        count in prop_oneof![Just(1_i64), Just(2), Just(3), Just(-7), Just(i64::MAX), Just(i64::MIN), any::<i64>()],
        p in 1..=38_u8,
        extra in 0..=8_i8,
    ) {
        prop_assume!(count != 0);
        let s = (ss + extra).min(38);
        let (slow, _) = avg_decimal(sum, ss, count, p, s);
        match avg_fast(sum, ss, count, p, s) {
            Some(v) => prop_assert_eq!(Some(v), slow),
            None => {
                // The shortcut declines only when it does not apply or
                // the value does not fit.
                let k = u32::try_from(s - ss).unwrap();
                let applies = s <= 19
                    && sum.unsigned_abs().checked_mul(10_u128.pow(k)).is_some();
                prop_assert!(!applies || slow.is_none());
            }
        }
    }
}

/// Spark 4.2's grouped `sum`/`avg` over rows in a fixed order, ANSI on
/// and off (`tests/scripts/gen_decimal_sum_avg.py`): row `i` is in group
/// `i % groups` and in Spark's `range` partition (contiguous ranges).
/// Each partition is folded in one batch or one row per batch (the
/// running bound decides each proof), or on the per-row path; several
/// partitions are merged as partial states in partition order. Every
/// run gives Spark's values or error.
#[test]
fn spark_sum_avg_goldens() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/spark-4.2.0/decimal-sum-avg.json"
    );
    let cases: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(cases.len(), 172);
    let num = |v: &serde_json::Value| usize::try_from(v.as_u64().unwrap()).unwrap();
    for c in &cases {
        let ansi = c["ansi"].as_bool().unwrap();
        let f = match c["fn"].as_str().unwrap() {
            "sum" => AggregateFunction::Sum,
            _ => AggregateFunction::Avg,
        };
        let ty = c["type"].as_array().unwrap();
        let p = u8::try_from(ty[0].as_u64().unwrap()).unwrap();
        let s = i8::try_from(ty[1].as_u64().unwrap()).unwrap();
        let t = DataType::Decimal128(p, s);
        let (groups, parts) = (num(&c["groups"]), num(&c["partitions"]));
        let g32 = u32::try_from(groups).unwrap();
        let rows: Vec<Row> = c["values"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let g = u32::try_from(i % groups).unwrap();
                (g, v.as_str().unwrap().parse().unwrap(), true, 1)
            })
            .collect();
        let n = rows.len();
        let ids: Vec<u32> = (0..g32).collect();
        let mut runs = Vec::new();
        for (proofs, one_batch) in [(true, true), (true, false), (false, true)] {
            let fold = |part: &[Row], a: &mut Accumulator| {
                if one_batch {
                    feed(a, g32, part, &t, 0);
                } else {
                    for r in part {
                        feed(a, g32, std::slice::from_ref(r), &t, 0);
                    }
                }
            };
            let mut a = acc(&f, &t, ansi, proofs);
            a.resize(groups);
            if parts == 1 {
                fold(&rows, &mut a);
            } else {
                for k in 0..parts {
                    let part = &rows[k * n / parts..(k + 1) * n / parts];
                    let mut partial = acc(&f, &t, ansi, proofs);
                    partial.resize(groups);
                    fold(part, &mut partial);
                    a.merge(&ids, &partial.state(&ids).unwrap()).unwrap();
                }
            }
            runs.push(a.evaluate(&ids).map_err(|e| e.to_string()));
        }
        for got in &runs {
            match (got, c.get("error")) {
                (Err(e), Some(want)) => {
                    assert!(e.contains(want.as_str().unwrap()), "{c}: {e}");
                }
                (Ok(v), None) => {
                    let DataType::Decimal128(op, os) = v.data_type() else {
                        panic!("{c}: {:?}", v.data_type());
                    };
                    assert_eq!(c["type_out"], format!("decimal({op},{os})"), "{c}");
                    let v = v.as_primitive::<Decimal128Type>();
                    let want = c["value"].as_array().unwrap();
                    assert_eq!(want.len(), groups.min(n), "{c}");
                    for (g, w) in want.iter().enumerate() {
                        let text = (!v.is_null(g)).then(|| plain(v.value(g), *os));
                        assert_eq!(w.as_str(), text.as_deref(), "{c} group {g}");
                    }
                }
                (got, _) => panic!("{c}: {got:?}"),
            }
        }
    }
}

/// Half points round away from zero, as Spark's `HALF_UP` does.
#[test]
fn avg_single_division_half_points() {
    assert_eq!(avg_fast(5, 0, 2, 10, 0), Some(3));
    assert_eq!(avg_fast(-5, 0, 2, 10, 0), Some(-3));
    assert_eq!(avg_fast(5, 0, -2, 10, 0), Some(-3));
    assert_eq!(avg_fast(-5, 0, 3, 5, 4), Some(-16_667));
    assert_eq!(avg_decimal(-5, 0, 3, 5, 4).0, Some(-16_667));
    // Numerator beyond u128: `M * 10^4`; the digit loop decides.
    assert_eq!(avg_fast(M, 0, 3, 38, 4), None);
    assert_eq!(avg_fast(1, 0, 3, 38, 20), None);
    assert_eq!(avg_fast(1, 2, 3, 38, 1), None);
    assert_eq!(avg_fast(1, 0, 0, 38, 1), None);
    // Does not fit: the caller takes the loop for the error text.
    assert_eq!(avg_fast(99_999, 0, 1, 5, 1), None);
}

/// The proof holds strictly below the bound, Σ|w|·|v| counts weights,
/// and NULL slots do not count.
#[test]
fn proof_boundaries() {
    let b = Some(100_u128);
    assert!(proven(true, b, 89, &[10_i128], None, None));
    assert!(!proven(true, b, 90, &[10_i128], None, None));
    assert!(!proven(false, b, 0, &[1_i128], None, None));
    assert!(!proven(true, None, 0, &[1_i128], None, None));
    let w = Weights::Column(Int64Array::from(vec![-3_i64, 2]));
    assert_eq!(batch_abs(&[-10_i128, 5], None, Some(&w)), 40);
    assert_eq!(batch_abs(&[7_i64], None, Some(&Weights::Const(-4))), 28);
    let nulls = NullBuffer::from(vec![true, false]);
    assert_eq!(batch_abs(&[1, i128::MAX], Some(&nulls), None), 1);
    assert_eq!(batch_abs(&[i128::MIN, i128::MIN], None, None), u128::MAX);
    let big = Weights::Const(i64::MIN);
    assert_eq!(batch_abs(&[M], None, Some(&big)), u128::MAX);
}

/// Spark R6 shape: a partial that overflowed upstream (sum NULL,
/// nonempty) merged with a proof-eligible batch stays overflowed.
#[test]
fn overflowed_partial_is_sticky_on_the_proof_path() {
    let t = DataType::Decimal128(38, 0);
    for f in [&AggregateFunction::Sum, &AggregateFunction::Avg] {
        let mut part = acc(f, &t, true, true);
        feed(&mut part, 1, &[(0, M, true, 1), (0, 1, true, 1)], &t, 0);
        let st = part.state(&[0]).unwrap();
        assert!(st[0].is_null(0), "{f:?} partial overflowed");
        let mut small = acc(f, &t, true, true);
        feed(&mut small, 1, &[(0, -5, true, 1)], &t, 0);
        let small = small.state(&[0]).unwrap();
        for order in [[&st, &small], [&small, &st]] {
            let mut m = acc(f, &t, true, true);
            m.resize(1);
            for s in order {
                m.merge(&[0], s).unwrap();
            }
            let err = m.evaluate(&[0]).unwrap_err().to_string();
            assert!(err.contains("ARITHMETIC_OVERFLOW"), "{err}");
            let mut legacy = acc(f, &t, false, true);
            legacy.resize(1);
            for s in order {
                legacy.merge(&[0], s).unwrap();
            }
            assert!(legacy.evaluate(&[0]).unwrap().is_null(0));
        }
    }
}

/// Spark R6: partials `M | 1 | -5` overflow when `M` meets `1` first and
/// not when `-5` comes between; the outcome follows the merge order (not
/// a Spark golden: Spark's merge order is its shuffle-fetch order).
#[test]
fn r6_partials_follow_merge_order() {
    let t = DataType::Decimal128(38, 0);
    for f in [&AggregateFunction::Sum, &AggregateFunction::Avg] {
        let partial = |v: i128| {
            let mut a = acc(f, &t, true, true);
            feed(&mut a, 1, &[(0, v, true, 1)], &t, 0);
            a.state(&[0]).unwrap()
        };
        let (m, one, minus5) = (partial(M), partial(1), partial(-5));
        for proofs in [true, false] {
            let merged = |order: [&Vec<ArrayRef>; 3]| {
                let mut a = acc(f, &t, true, proofs);
                a.resize(1);
                for st in order {
                    a.merge(&[0], st).unwrap();
                }
                a.evaluate(&[0]).map_err(|e| e.to_string())
            };
            let err = merged([&m, &one, &minus5]).unwrap_err();
            assert!(err.contains("ARITHMETIC_OVERFLOW"), "{err}");
            let ok = merged([&m, &minus5, &one]);
            if f == &AggregateFunction::Sum {
                let v = ok.unwrap();
                assert_eq!(v.as_primitive::<Decimal128Type>().value(0), M - 4);
            } else {
                // The average (M - 4) / 3 is out of decimal(38, 4)'s range.
                assert!(ok.unwrap_err().contains("NUMERIC_VALUE_OUT_OF_RANGE"));
            }
        }
    }
}

/// Spark R1 vs R3: M, 1, -5 overflows in that order (grouped), M, -5, 1
/// does not; the proof path keeps the processing order.
#[test]
fn transient_overflow_follows_row_order() {
    let t = DataType::Decimal128(38, 0);
    let mut a = acc(&AggregateFunction::Sum, &t, true, true);
    feed(
        &mut a,
        1,
        &[(0, M, true, 1), (0, 1, true, 1), (0, -5, true, 1)],
        &t,
        0,
    );
    assert!(a.evaluate(&[0]).is_err());
    // The same across batches: the second batch alone is provable, the
    // running bound is not.
    for f in [&AggregateFunction::Sum, &AggregateFunction::Avg] {
        let mut a = acc(f, &t, true, true);
        feed(&mut a, 1, &[(0, M, true, 1)], &t, 0);
        feed(&mut a, 1, &[(0, 1, true, 1)], &t, 0);
        feed(&mut a, 1, &[(0, -5, true, 1)], &t, 0);
        let err = a.evaluate(&[0]).unwrap_err().to_string();
        assert!(err.contains("ARITHMETIC_OVERFLOW"), "{err}");
    }
    let mut a = acc(&AggregateFunction::Sum, &t, true, true);
    feed(
        &mut a,
        1,
        &[(0, M, true, 1), (0, -5, true, 1), (0, 1, true, 1)],
        &t,
        0,
    );
    let v = a.evaluate(&[0]).unwrap();
    assert_eq!(v.as_primitive::<Decimal128Type>().value(0), M - 4);
    // IVM: a retraction lowers the sum; with no more groups than rows
    // the bound is recomputed and shrinks with it.
    let mut a = acc(&AggregateFunction::Sum, &t, true, true);
    feed(&mut a, 1, &[(0, M, true, 1)], &t, 0);
    assert_eq!(a.max_sum, M.unsigned_abs());
    feed(&mut a, 1, &[(0, M, true, -1)], &t, 2);
    assert_eq!(a.max_sum, 0);
    // Many groups, one row: only the touched group raises the bound.
    assert_eq!(refresh_max(9, &[1, -20, 3], &[2]), 9);
    assert_eq!(refresh_max(9, &[1, -20, 3], &[1]), 20);
    assert_eq!(
        quick_abs(&[3_i128, -2], Some(&Weights::Const(-2))),
        2 * 4 * 2
    );
    let w = Weights::Column(Int64Array::from(vec![1_i64, -3]));
    assert_eq!(quick_abs(&[3_i64, -2], Some(&w)), 2 * 4 * 4);
    feed(&mut a, 1, &[(0, 7, true, 1)], &t, 0);
    let v = a.evaluate(&[0]).unwrap();
    assert_eq!(v.as_primitive::<Decimal128Type>().value(0), 7);
}
