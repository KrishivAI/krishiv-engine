// SPDX-License-Identifier: Apache-2.0
//! The moment aggregates through their accumulators: edge inputs under both
//! values of `spark.sql.legacy.statisticalAggregate`, Spark's recorded
//! results reproduced bit for bit under Spark's partitioning, merges over
//! any split, and the checks of malformed input.
#![cfg(not(loom))]

use std::ops::Range;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Array, ArrayRef, Float32Array, Float64Array, Int32Array};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;
use krishiv_common::error::ErrorClass;
use krishiv_types::batch::Weights;

use super::arity;
use crate::aggregate::{Accumulator, AggInput, AggregateExpr, AggregateFunction};

use AggregateFunction::{
    Corr, CovarPop, CovarSamp, Kurtosis, Skewness, StddevPop, StddevSamp, VarPop, VarSamp,
};

const ALL: [AggregateFunction; 9] = [
    StddevSamp, StddevPop, VarSamp, VarPop, Skewness, Kurtosis, Corr, CovarSamp, CovarPop,
];

/// The functions NULL (or NaN, with the legacy flag) for one row.
const SINGLE_ROW_DIVIDES: [AggregateFunction; 4] = [StddevSamp, VarSamp, CovarSamp, Corr];

fn expr(f: &AggregateFunction) -> AggregateExpr {
    AggregateExpr {
        function: f.clone(),
        args: (0..arity(f)).collect(),
        input: AggInput::Raw,
        arg_type: DataType::Float64,
        name: "m".to_owned(),
    }
}

fn doubles(v: &[Option<f64>]) -> ArrayRef {
    Arc::new(Float64Array::from(v.to_vec()))
}

fn some(v: &[f64]) -> Vec<Option<f64>> {
    v.iter().copied().map(Some).collect()
}

/// An accumulator over `groups` groups, without ANSI.
fn acc(f: &AggregateFunction, legacy: bool, groups: usize) -> Accumulator {
    acc_ansi(f, legacy, false, groups)
}

fn acc_ansi(f: &AggregateFunction, legacy: bool, ansi: bool, groups: usize) -> Accumulator {
    let mut a = Accumulator::new(&expr(f), ansi)
        .unwrap()
        .legacy_statistical(legacy);
    a.resize(groups);
    a
}

/// The columns an aggregate reads: `xs`, and `ys` for the pair statistics.
fn columns(f: &AggregateFunction, xs: &[Option<f64>], ys: &[Option<f64>]) -> Vec<ArrayRef> {
    let mut cols = vec![doubles(xs)];
    if arity(f) == 2 {
        cols.push(doubles(ys));
    }
    cols
}

/// One global group over rows `xs` (and `ys`), in one pass.
fn global(
    f: &AggregateFunction,
    legacy: bool,
    xs: &[Option<f64>],
    ys: &[Option<f64>],
) -> Option<f64> {
    let mut a = acc(f, legacy, 1);
    a.update(&vec![0; xs.len()], &columns(f, xs, ys), &Weights::Const(1))
        .unwrap();
    value(&a, 0)
}

fn value(a: &Accumulator, g: u32) -> Option<f64> {
    let out = a.evaluate(&[g]).unwrap();
    assert_eq!(out.data_type(), &DataType::Float64);
    let out = out.as_primitive::<Float64Type>();
    (!out.is_null(0)).then(|| out.value(0))
}

/// `Some(NaN)` and `Some(v)` compared by bits, `None` as is.
fn same(a: Option<f64>, b: Option<f64>) -> bool {
    a.map(f64::to_bits) == b.map(f64::to_bits)
}

fn close(a: Option<f64>, b: Option<f64>, tolerance: f64) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) if a.is_nan() || b.is_nan() => a.is_nan() && b.is_nan(),
        (Some(a), Some(b)) => (a - b).abs() <= tolerance * a.abs().max(b.abs()).max(1e-300),
        _ => false,
    }
}

fn is_nan(v: Option<f64>) -> bool {
    v.is_some_and(f64::is_nan)
}

/// A deterministic pseudo-random sequence in `[-50, 50)` (an LCG).
fn sequence(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[expect(clippy::cast_precision_loss, reason = "a test value")]
            let v = (s >> 33) as f64 / f64::from(1_u32 << 31) * 100.0 - 50.0;
            v
        })
        .collect()
}

#[test]
fn no_rows_is_null() {
    for f in ALL {
        for legacy in [false, true] {
            assert_eq!(global(&f, legacy, &[], &[]), None, "{f:?} legacy={legacy}");
            // A group that only saw NULLs has no rows either.
            let nulls = [None, None];
            assert_eq!(global(&f, legacy, &nulls, &nulls), None, "{f:?}");
        }
    }
}

#[test]
fn one_row_divides_by_zero_where_spark_does() {
    let (x, y) = (some(&[5.0]), some(&[7.0]));
    for f in ALL {
        let strict = global(&f, false, &x, &y);
        let legacy = global(&f, true, &x, &y);
        if [Skewness, Kurtosis].contains(&f) || SINGLE_ROW_DIVIDES.contains(&f) {
            assert_eq!(strict, None, "{f:?}");
            assert!(is_nan(legacy), "{f:?}: {legacy:?}");
        } else {
            // The population statistics of one row are 0.
            assert_eq!(strict, Some(0.0), "{f:?}");
            assert_eq!(legacy, Some(0.0), "{f:?}");
        }
    }
}

#[test]
fn constant_values_have_no_variance() {
    let (x, y) = (some(&[3.0; 4]), some(&[2.0; 4]));
    for f in ALL {
        let strict = global(&f, false, &x, &y);
        let legacy = global(&f, true, &x, &y);
        match f {
            Skewness | Kurtosis => {
                assert_eq!(strict, None, "{f:?}");
                assert!(is_nan(legacy), "{f:?}");
            }
            // Several rows over a constant column: NULL without ANSI under
            // both settings of the legacy flag, never NaN (see
            // `corr_of_a_constant_column_is_a_division_by_zero_under_ansi`).
            Corr => {
                assert_eq!((strict, legacy), (None, None), "{f:?}");
            }
            _ => {
                assert_eq!(strict, Some(0.0), "{f:?}");
                assert_eq!(legacy, Some(0.0), "{f:?}");
            }
        }
    }
}

#[test]
fn corr_of_a_constant_column_is_a_division_by_zero_under_ansi() {
    let varying = some(&[1.0, 2.0, 4.0]);
    let constant = some(&[3.0; 3]);
    for (xs, ys) in [
        (&constant, &varying),
        (&varying, &constant),
        (&constant, &constant),
    ] {
        for legacy in [false, true] {
            let mut a = acc_ansi(&Corr, legacy, true, 1);
            a.update(&[0; 3], &columns(&Corr, xs, ys), &Weights::Const(1))
                .unwrap();
            let e = a.evaluate(&[0]).unwrap_err();
            assert_eq!(e.error_class(), ErrorClass::DivideByZero, "legacy={legacy}");
            // Without ANSI the same buffer is NULL.
            assert_eq!(global(&Corr, legacy, xs, ys), None);
        }
    }
    // Rows with variance in both columns, and a single row, never raise
    // (the single row is NULL, or NaN with the legacy flag).
    for legacy in [false, true] {
        let mut a = acc_ansi(&Corr, legacy, true, 2);
        a.update(
            &[0, 0, 0, 1],
            &columns(
                &Corr,
                &some(&[1.0, 2.0, 4.0, 5.0]),
                &some(&[2.0, 1.0, 7.0, 5.0]),
            ),
            &Weights::Const(1),
        )
        .unwrap();
        assert!(a.evaluate(&[0]).is_ok());
        assert_eq!(value(&a, 1).is_some(), legacy);
    }
    // A group that raises does not stop the evaluation of the others when
    // they are evaluated apart: only the selected groups are read.
    let mut a = acc_ansi(&Corr, false, true, 2);
    a.update(
        &[0, 0, 1, 1],
        &columns(
            &Corr,
            &some(&[1.0, 2.0, 3.0, 3.0]),
            &some(&[1.0, 2.0, 1.0, 2.0]),
        ),
        &Weights::Const(1),
    )
    .unwrap();
    assert!(value(&a, 0).is_some());
    assert!(a.evaluate(&[1]).is_err());
}

#[test]
fn rows_with_a_null_argument_are_skipped() {
    let xs = [Some(1.0), None, Some(3.0), Some(5.0), Some(9.5), None];
    let ys = [Some(2.0), Some(9.0), None, Some(10.0), Some(1.5), None];
    for f in ALL {
        let complete: Vec<usize> = (0..xs.len())
            .filter(|&i| xs[i].is_some() && (arity(&f) == 1 || ys[i].is_some()))
            .collect();
        let pick = |v: &[Option<f64>]| complete.iter().map(|&i| v[i]).collect::<Vec<_>>();
        for legacy in [false, true] {
            let got = global(&f, legacy, &xs, &ys);
            let want = global(&f, legacy, &pick(&xs), &pick(&ys));
            assert!(same(got, want), "{f:?}: {got:?} vs {want:?}");
            assert!(got.is_some(), "{f:?}");
        }
    }
}

#[test]
fn values_under_null_slots_never_count() {
    // The slots under NULL hold NaN and infinities: never read.
    let values = vec![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0, f64::NEG_INFINITY];
    let valid = NullBuffer::from(vec![true, false, true, false, true, false]);
    let x: ArrayRef = Arc::new(Float64Array::new(
        values.clone().into(),
        Some(valid.clone()),
    ));
    let y: ArrayRef = Arc::new(Float64Array::new(values.into(), Some(valid)));
    for f in ALL {
        let mut a = acc(&f, false, 1);
        let cols = if arity(&f) == 2 {
            vec![Arc::clone(&x), Arc::clone(&y)]
        } else {
            vec![Arc::clone(&x)]
        };
        a.update(&[0; 6], &cols, &Weights::Const(1)).unwrap();
        let (clean_x, clean_y) = (some(&[1.0, 3.0, 5.0]), some(&[1.0, 3.0, 5.0]));
        let want = global(&f, false, &clean_x, &clean_y);
        assert!(same(value(&a, 0), want), "{f:?}");
    }
}

#[test]
fn nan_and_infinity_propagate() {
    let ys = some(&[1.0, 2.0, 3.0]);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let xs = some(&[1.0, bad, 3.0]);
        for f in ALL {
            for legacy in [false, true] {
                let v = global(&f, legacy, &xs, &ys);
                assert!(is_nan(v), "{f:?}({bad}): {v:?}");
            }
        }
    }
    // The pair statistics see a NaN in either argument.
    for f in [Corr, CovarPop, CovarSamp] {
        let v = global(&f, false, &ys, &some(&[1.0, f64::NAN, 3.0]));
        assert!(is_nan(v), "{f:?}");
    }
    // A NaN is a value, not a NULL: it counts as a row. One row still
    // divides by zero first (Spark checks the count before the value).
    let nan = some(&[f64::NAN]);
    assert!(is_nan(global(&StddevPop, false, &nan, &[])));
    assert_eq!(global(&StddevSamp, false, &nan, &[]), None);
    assert!(is_nan(global(&StddevSamp, true, &nan, &[])));
}

/// A two-pass reference: the central moments and co-moments of `xs`, `ys`.
struct Reference {
    n: f64,
    m2: f64,
    m3: f64,
    m4: f64,
    ck: f64,
    y_m2: f64,
}

fn reference(xs: &[f64], ys: &[f64]) -> Reference {
    #[expect(clippy::cast_precision_loss, reason = "a test count")]
    let n = xs.len() as f64;
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let pow = |k: i32| xs.iter().map(|x| (x - mx).powi(k)).sum::<f64>();
    Reference {
        n,
        m2: pow(2),
        m3: pow(3),
        m4: pow(4),
        ck: xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum(),
        y_m2: ys.iter().map(|y| (y - my).powi(2)).sum(),
    }
}

#[test]
fn matches_the_textbook_definitions() {
    let (xs, ys) = (sequence(400, 7), sequence(400, 99));
    let (x, y) = (some(&xs), some(&ys));
    let r = reference(&xs, &ys);
    let n = r.n;
    let want = |f| match f {
        StddevSamp => (r.m2 / (n - 1.0)).sqrt(),
        StddevPop => (r.m2 / n).sqrt(),
        VarSamp => r.m2 / (n - 1.0),
        VarPop => r.m2 / n,
        Skewness => n.sqrt() * r.m3 / r.m2.powf(1.5),
        Kurtosis => n * r.m4 / (r.m2 * r.m2) - 3.0,
        Corr => r.ck / (r.m2 * r.y_m2).sqrt(),
        CovarSamp => r.ck / (n - 1.0),
        CovarPop => r.ck / n,
        _ => f64::NAN,
    };
    for f in ALL {
        let got = global(&f, false, &x, &y);
        assert!(
            close(got, Some(want(f.clone())), 1e-9),
            "{f:?}: {got:?} vs {}",
            want(f.clone())
        );
    }
}

#[test]
fn values_far_from_zero_keep_their_precision() {
    // The spread is 0..=4 around 1e9: the sum-of-squares formula loses it
    // all (1e18 magnitudes, 1e-16 precision).
    let xs: Vec<f64> = (0..1000_u32).map(|i| 1e9 + f64::from(i % 5)).collect();
    let ys: Vec<f64> = (0..1000_u32)
        .map(|i| 3e9 - f64::from(i % 5) * 2.0)
        .collect();
    let (x, y) = (some(&xs), some(&ys));
    // The sum-of-squares formula is hopeless here (it answers 0).
    #[expect(clippy::cast_precision_loss, reason = "a test count")]
    let n = xs.len() as f64;
    let naive = xs.iter().map(|x| x * x).sum::<f64>() / n - (xs.iter().sum::<f64>() / n).powi(2);
    assert!((naive - 2.0).abs() > 1.0, "{naive}");
    // The values 0..=4 equally often have variance 2.
    let within = |v, want| close(v, Some(want), 1e-6);
    assert!(within(global(&VarPop, false, &x, &[]), 2.0));
    assert!(within(global(&StddevPop, false, &x, &[]), 2.0_f64.sqrt()));
    assert!(within(global(&CovarPop, false, &x, &y), -4.0));
    assert!(within(global(&Corr, false, &x, &y), -1.0));
}

/// A recorded result: the function, its arguments and Spark's value.
type Recorded<'a> = (AggregateFunction, &'a [Option<f64>], &'a [Option<f64>], f64);

/// The order partial states are merged in, by position.
type Order = Box<dyn Fn(usize) -> usize>;

/// Spark's slicing of `rows` rows into `p` partitions
/// (`ParallelCollectionRDD.slice`).
fn spark_slices(rows: usize, p: usize) -> Vec<Range<usize>> {
    (0..p).map(|i| i * rows / p..(i + 1) * rows / p).collect()
}

/// Spark's evaluation: each partition folds its rows into a partial buffer;
/// the final stage merges the partials, in order, into an empty buffer.
fn spark_style(
    f: &AggregateFunction,
    legacy: bool,
    xs: &[Option<f64>],
    ys: &[Option<f64>],
    partitions: usize,
) -> Option<f64> {
    let mut last = acc(f, legacy, 1);
    for r in spark_slices(xs.len(), partitions) {
        let mut partial = acc(f, legacy, 1);
        let (px, py) = (&xs[r.clone()], ys.get(r.clone()).unwrap_or_default());
        partial
            .update(&vec![0; r.len()], &columns(f, px, py), &Weights::Const(1))
            .unwrap();
        last.merge(&[0], &partial.state(&[0]).unwrap()).unwrap();
    }
    value(&last, 0)
}

/// Results Spark 4.2.0 recorded, reproduced to the bit by Spark's own
/// partitioning of the rows (which the digits depend on: one pass over the
/// first query's rows differs in the last digits). The partition counts are
/// the ones under which the recorded digits come out: `SQLQueryTestSuite`
/// runs two partitions, the function matrix one row per partition.
#[test]
fn recorded_spark_results_are_reproduced_to_the_bit() {
    let t = |v: &[f64]| some(v);
    let none: Vec<Option<f64>> = Vec::new();
    // group-by.sql: `SELECT SKEWNESS(a), KURTOSIS(a), ..., VARIANCE(a),
    // STDDEV(a) ... FROM testData`, two partitions.
    let a = [
        Some(1.0),
        Some(1.0),
        Some(2.0),
        Some(2.0),
        Some(3.0),
        Some(3.0),
        None,
        Some(3.0),
        None,
    ];
    for (f, want) in [
        (Skewness, -0.272_380_105_814_572_9),
        (Kurtosis, -1.506_920_415_224_913_4),
        (VarSamp, 0.809_523_809_523_809_4),
        (StddevSamp, 0.899_735_410_842_437_2),
    ] {
        let got = spark_style(&f, false, &a, &none, 2);
        assert!(same(got, Some(want)), "{f:?}: {got:?} vs {want:?}");
    }
    // linear-regression.sql: `SELECT k, corr(y, x) ... GROUP BY k`, group
    // k = 2 of the six-row table cut in two partitions: its rows
    // (y, x) = (10, 11), (20, 22) | (25, NULL), (30, 35), (NULL, 40).
    let mut last = acc(&Corr, false, 1);
    for (ys, xs) in [
        (vec![Some(10.0), Some(20.0)], vec![Some(11.0), Some(22.0)]),
        (
            vec![Some(25.0), Some(30.0), None],
            vec![None, Some(35.0), Some(40.0)],
        ),
    ] {
        let mut partial = acc(&Corr, false, 1);
        partial
            .update(
                &[0; 3][..ys.len()],
                &columns(&Corr, &ys, &xs),
                &Weights::Const(1),
            )
            .unwrap();
        last.merge(&[0], &partial.state(&[0]).unwrap()).unwrap();
    }
    assert!(same(value(&last, 0), Some(0.998_844_598_112_153_3)));
    // The function matrix (`function-matrix/cases/agg.jsonl`), one row per
    // partition.
    let (m10, m100) = (
        t(&[-10.0, -20.0, 100.0, 1000.0]),
        t(&[1.0, 10.0, 100.0, 10.0, 1.0]),
    );
    let m1000 = t(&[-1000.0, -100.0, 10.0, 20.0]);
    let one_two_three = t(&[1.0, 2.0, 3.0]);
    let pairs_x = t(&[3.0, 3.0, 6.0]);
    let pairs_y = t(&[2.0, 3.0, 4.0]);
    let cases: [Recorded<'_>; 11] = [
        (Skewness, &m10, &none, 1.113_565_746_902_201_3),
        (Skewness, &m1000, &none, -1.113_565_746_902_201_1),
        (Kurtosis, &m10, &none, -0.701_436_804_752_961_8),
        (Kurtosis, &m100, &none, 0.194_323_231_916_989_86),
        (StddevSamp, &one_two_three, &none, 1.0),
        (StddevPop, &one_two_three, &none, 0.816_496_580_927_726),
        (VarSamp, &one_two_three, &none, 1.0),
        (VarPop, &one_two_three, &none, 0.666_666_666_666_666_6),
        (CovarSamp, &one_two_three, &one_two_three, 1.0),
        (
            CovarPop,
            &one_two_three,
            &one_two_three,
            0.666_666_666_666_666_6,
        ),
        (Corr, &pairs_x, &pairs_y, 0.866_025_403_784_438_7),
    ];
    for (f, x, y, want) in cases {
        let got = spark_style(&f, false, x, y, x.len());
        assert!(
            same(got, Some(want)),
            "{f:?} over {x:?}: {got:?} vs {want:?}"
        );
    }
    // A pass over one partition is another rounding of the same value.
    let one_pass = global(&Skewness, false, &m10, &none);
    assert!(close(one_pass, Some(1.113_565_746_902_201_3), 1e-14));
    assert!(!same(one_pass, Some(1.113_565_746_902_201_3)));
}

/// `values` in `parts` contiguous pieces, each folded on its own and the
/// partial states merged into a fresh accumulator in `order`.
fn split_merge(
    f: &AggregateFunction,
    xs: &[Option<f64>],
    ys: &[Option<f64>],
    parts: usize,
    order: &dyn Fn(usize) -> usize,
) -> Option<f64> {
    let slices = spark_slices(xs.len(), parts);
    let states: Vec<Vec<ArrayRef>> = slices
        .iter()
        .map(|r| {
            let mut a = acc(f, false, 1);
            let (px, py) = (&xs[r.clone()], ys.get(r.clone()).unwrap_or_default());
            a.update(&vec![0; r.len()], &columns(f, px, py), &Weights::Const(1))
                .unwrap();
            a.state(&[0]).unwrap()
        })
        .collect();
    let mut total = acc(f, false, 1);
    for i in 0..parts {
        total.merge(&[0], &states[order(i)]).unwrap();
    }
    value(&total, 0)
}

#[test]
fn any_split_merged_in_any_order_agrees_with_one_pass() {
    // NULLs in the middle, so some pieces are empty or all-NULL.
    let mut xs = some(&sequence(120, 3));
    let mut ys = some(&sequence(120, 4));
    for i in (0..120).step_by(7) {
        xs[i] = None;
    }
    for i in (3..120).step_by(11) {
        ys[i] = None;
    }
    for f in ALL {
        let once = global(&f, false, &xs, &ys);
        for parts in 1..=9 {
            let orders: [(&str, Order); 3] = [
                ("forward", Box::new(|i| i)),
                ("reverse", Box::new(move |i| parts - 1 - i)),
                ("rotated", Box::new(move |i| (i + 2) % parts)),
            ];
            for (name, order) in orders {
                let got = split_merge(&f, &xs, &ys, parts, &order);
                assert!(
                    close(got, once, 1e-9),
                    "{f:?} {parts} {name}: {got:?} vs {once:?}"
                );
                // The same split in the same order is deterministic.
                assert!(same(got, split_merge(&f, &xs, &ys, parts, &order)));
            }
        }
    }
}

#[test]
fn empty_partial_states_merge_to_the_same_value() {
    let (xs, ys) = (some(&sequence(31, 5)), some(&sequence(31, 6)));
    for f in ALL {
        let want = global(&f, false, &xs, &ys);
        let mut full = acc(&f, false, 1);
        full.update(&[0; 31], &columns(&f, &xs, &ys), &Weights::Const(1))
            .unwrap();
        let empty = acc(&f, false, 1).state(&[0]).unwrap();
        let state = full.state(&[0]).unwrap();
        // Spark's final stage merges every partial into an empty buffer;
        // an empty partial merged last changes nothing.
        let mut first = acc(&f, false, 1);
        first.merge(&[0], &state).unwrap();
        let mut last = acc(&f, false, 1);
        last.merge(&[0], &state).unwrap();
        last.merge(&[0], &empty).unwrap();
        for a in [&first, &last] {
            assert!(
                same(value(a, 0), want),
                "{f:?}: {:?} vs {want:?}",
                value(a, 0)
            );
        }
        // Only empty partials: no rows.
        let mut none = acc(&f, false, 1);
        none.merge(&[0], &empty).unwrap();
        none.merge(&[0], &empty).unwrap();
        assert_eq!(value(&none, 0), None, "{f:?}");
    }
}

#[test]
fn groups_are_independent() {
    let xs = some(&[1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 5.0]);
    let ys = some(&[2.0, 5.0, 4.0, 4.0, 7.0, 3.0, 1.0]);
    let groups = [0_u32, 1, 0, 1, 0, 1, 2];
    for f in ALL {
        let mut a = acc(&f, false, 3);
        a.update(&groups, &columns(&f, &xs, &ys), &Weights::Const(1))
            .unwrap();
        let pick = |v: &[Option<f64>], g: u32| {
            groups
                .iter()
                .zip(v)
                .filter(|(k, _)| **k == g)
                .map(|(_, v)| *v)
                .collect::<Vec<_>>()
        };
        for g in 0..3 {
            let want = global(&f, false, &pick(&xs, g), &pick(&ys, g));
            assert!(same(value(&a, g), want), "{f:?} group {g}");
        }
        // Several ids at once, in any order.
        let all = a.evaluate(&[2, 0, 1]).unwrap();
        assert_eq!(all.len(), 3);
        assert!(same(
            all.as_primitive::<Float64Type>().iter().next().unwrap(),
            value(&a, 2)
        ));
    }
}

#[test]
fn integer_and_float_inputs_are_read_as_doubles() {
    let ints: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None, Some(2), Some(4)]));
    let floats: ArrayRef = Arc::new(Float32Array::from(vec![
        Some(1.0),
        None,
        Some(2.0),
        Some(4.0),
    ]));
    let want = global(&VarSamp, false, &some(&[1.0, 2.0, 4.0]), &[]);
    for (col, t) in [(ints, DataType::Int32), (floats, DataType::Float32)] {
        let e = AggregateExpr {
            arg_type: t,
            ..expr(&VarSamp)
        };
        assert_eq!(e.result_type().unwrap(), DataType::Float64);
        let mut a = Accumulator::new(&e, true).unwrap();
        a.resize(1);
        a.update(&[0; 4], &[col], &Weights::Const(1)).unwrap();
        assert!(same(value(&a, 0), want));
    }
}

#[test]
fn state_columns_are_sparks_buffers() {
    let names = |f: &AggregateFunction| -> Vec<String> {
        expr(f)
            .state_fields()
            .unwrap()
            .iter()
            .map(|c| {
                assert_eq!(c.data_type(), &DataType::Float64);
                assert!(!c.is_nullable(), "a buffer is never NULL");
                c.name().clone()
            })
            .collect()
    };
    assert_eq!(names(&StddevSamp), ["m#n", "m#avg", "m#m2"]);
    assert_eq!(names(&VarPop), ["m#n", "m#avg", "m#m2"]);
    assert_eq!(names(&Skewness), ["m#n", "m#avg", "m#m2", "m#m3"]);
    assert_eq!(names(&Kurtosis), ["m#n", "m#avg", "m#m2", "m#m3", "m#m4"]);
    assert_eq!(names(&CovarPop), ["m#n", "m#xAvg", "m#yAvg", "m#ck"]);
    assert_eq!(
        names(&Corr),
        ["m#n", "m#xAvg", "m#yAvg", "m#ck", "m#xMk", "m#yMk"]
    );
    for f in ALL {
        let a = acc(&f, false, 2);
        let state = a.state(&[0, 1]).unwrap();
        assert_eq!(state.len(), names(&f).len(), "{f:?}");
        assert!(state.iter().all(|c| c.len() == 2 && c.null_count() == 0));
        assert_eq!(a.state_len(), state.len());
        // The merging form reads those columns, wherever they sit.
        let merging = expr(&f).merging(4);
        assert_eq!(merging.input, AggInput::State);
        assert_eq!(merging.args, (4..4 + state.len()).collect::<Vec<_>>());
    }
}

#[test]
fn only_numbers_are_read_and_the_result_is_double() {
    for f in ALL {
        for t in [
            DataType::Int8,
            DataType::Int64,
            DataType::Float32,
            DataType::Float64,
        ] {
            let e = AggregateExpr {
                arg_type: t,
                ..expr(&f)
            };
            assert_eq!(e.result_type().unwrap(), DataType::Float64, "{f:?}");
        }
        // The analyzer casts everything else to double before it gets here.
        for t in [
            DataType::Utf8View,
            DataType::Decimal128(10, 2),
            DataType::Boolean,
        ] {
            let e = AggregateExpr {
                arg_type: t,
                ..expr(&f)
            };
            let err = e.result_type().unwrap_err();
            assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
            assert!(e.state_fields().is_err());
        }
    }
}

#[test]
fn row_weights_other_than_one_are_refused() {
    for f in ALL {
        let mut a = acc(&f, false, 1);
        let cols = columns(&f, &some(&[1.0, 2.0]), &some(&[3.0, 4.0]));
        let err = a.update(&[0, 0], &cols, &Weights::Const(2)).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::KrishivUnsupportedFeature);
        let column = Weights::Column(arrow_array::Int64Array::from(vec![1, -1]));
        assert!(a.update(&[0, 0], &cols, &column).is_err());
        assert!(a.update(&[0, 0], &cols, &Weights::Const(1)).is_ok());
        // Nothing of the refused batches was folded in.
        assert_eq!(
            a.state(&[0]).unwrap()[0]
                .as_primitive::<Float64Type>()
                .value(0),
            2.0
        );
    }
}

#[test]
fn malformed_inputs_are_errors_not_panics() {
    for f in ALL {
        let mut a = acc(&f, false, 1);
        let cols = columns(&f, &some(&[1.0, 2.0]), &some(&[1.0, 2.0]));
        // A group that does not exist.
        let err = a.update(&[0, 5], &cols, &Weights::Const(1)).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::InternalError, "{f:?}");
        assert!(a.evaluate(&[3]).is_err());
        assert!(a.state(&[3]).is_err());
        // Columns of another length than the rows.
        assert!(a.update(&[0], &cols, &Weights::Const(1)).is_err());
        // A running fold has no moment form.
        let run = crate::aggregate::Running {
            parts: &[0, 0],
            snapshots: &[crate::aggregate::Running::NONE; 2],
        };
        assert!(a.update_running(&[0, 0], &cols, &run).is_err());
        // States: too few columns, NULLs, another type, another length.
        let good = acc(&f, false, 1).state(&[0]).unwrap();
        assert!(a.merge(&[0], &good[..good.len() - 1]).is_err());
        let mut nulled = good.clone();
        nulled[0] = doubles(&[None]);
        assert!(a.merge(&[0], &nulled).is_err());
        let mut mistyped = good.clone();
        mistyped[0] = Arc::new(Int32Array::from(vec![1]));
        assert!(a.merge(&[0], &mistyped).is_err());
        assert!(a.merge(&[0, 0], &good).is_err());
    }
    // A moment kernel only exists for moment aggregates.
    assert!(super::Moments::new(AggregateFunction::Count, true).is_err());
}
