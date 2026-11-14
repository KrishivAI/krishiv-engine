// SPDX-License-Identifier: Apache-2.0
//! Differential tests of the dense folds (`dense.rs`): a batch folded with
//! no mask (the dense kernels) against the same batch under an all-true
//! `FILTER` mask (the per-row loop), and merged partial states against one
//! raw fold of all the rows. Integer `sum`, `min`, `max`, one- and
//! two-argument `count` over every integer-like carrier, and double
//! `sum`/`avg` (raw rows only: a merge reorders a double sum), sliced columns with
//! garbage under NULL slots, values at the `i64` edges, ANSI on and off,
//! one group and many.

use super::*;
use arrow_array::types::{
    Decimal64Type, DurationMicrosecondType, IntervalYearMonthType, TimestampMicrosecondType,
};
use arrow_array::{
    Date32Array, Float64Array, Int8Array, Int16Array, Int32Array, PrimitiveArray, UInt64Array,
};
use proptest::prelude::*;

/// `(group, value, valid)`.
type Row = (u32, i64, bool);

fn value() -> impl Strategy<Value = i64> + Clone {
    prop_oneof![
        4 => -1000..1000_i64,
        1 => Just(i64::MAX),
        1 => Just(i64::MIN),
        1 => Just(i64::MAX / 2 + 1),
        1 => Just(-(i64::MAX / 2) - 1),
        1 => any::<i64>(),
    ]
}

fn rows(groups: u32) -> impl Strategy<Value = Vec<Row>> {
    prop::collection::vec((0..groups, value(), prop::bool::weighted(0.8)), 0..200)
}

/// The argument types: each integer-like carrier `dense::native_ints` reads.
const TYPES: [DataType; 11] = [
    DataType::Int64,
    DataType::Int16,
    DataType::Duration(TimeUnit::Microsecond),
    DataType::Float64,
    DataType::Int32,
    DataType::Int8,
    DataType::Date32,
    DataType::Timestamp(TimeUnit::Microsecond, None),
    DataType::Interval(IntervalUnit::YearMonth),
    DataType::Decimal64(18, 2),
    DataType::Decimal128(38, 0),
];

/// A column of `t` over `junk` then `rows` (values narrowed to `t`), sliced
/// to `rows`; a NULL slot holds a value too.
fn column(t: &DataType, junk: &[Row], rows: &[Row]) -> ArrayRef {
    let all: Vec<&Row> = junk.iter().chain(rows).collect();
    let nulls = NullBuffer::from(all.iter().map(|r| r.2).collect::<Vec<_>>());
    #[expect(clippy::cast_possible_truncation, reason = "narrowed test values")]
    let a: ArrayRef = match t {
        DataType::Int64 => Arc::new(Int64Array::new(
            all.iter().map(|r| r.1).collect(),
            Some(nulls),
        )),
        #[expect(clippy::cast_precision_loss, reason = "test values")]
        DataType::Float64 => Arc::new(Float64Array::new(
            all.iter()
                .map(|r| match r.1 {
                    0 => -0.0,
                    i64::MAX => f64::NAN,
                    i64::MIN => f64::NEG_INFINITY,
                    x if x == i64::MAX / 2 + 1 => f64::INFINITY,
                    x => x as f64 / 3.0,
                })
                .collect(),
            Some(nulls),
        )),
        DataType::Int32 => Arc::new(Int32Array::new(
            all.iter().map(|r| r.1 as i32).collect(),
            Some(nulls),
        )),
        DataType::Int16 => Arc::new(Int16Array::new(
            all.iter().map(|r| r.1 as i16).collect(),
            Some(nulls),
        )),
        DataType::Duration(_) => Arc::new(PrimitiveArray::<DurationMicrosecondType>::new(
            all.iter().map(|r| r.1).collect(),
            Some(nulls),
        )),
        DataType::Int8 => Arc::new(Int8Array::new(
            all.iter().map(|r| r.1 as i8).collect(),
            Some(nulls),
        )),
        DataType::Date32 => Arc::new(Date32Array::new(
            all.iter().map(|r| r.1 as i32).collect(),
            Some(nulls),
        )),
        DataType::Timestamp(..) => Arc::new(PrimitiveArray::<TimestampMicrosecondType>::new(
            all.iter().map(|r| r.1).collect(),
            Some(nulls),
        )),
        DataType::Interval(_) => Arc::new(PrimitiveArray::<IntervalYearMonthType>::new(
            all.iter().map(|r| r.1 as i32).collect(),
            Some(nulls),
        )),
        DataType::Decimal64(p, s) => Arc::new(
            PrimitiveArray::<Decimal64Type>::new(all.iter().map(|r| r.1).collect(), Some(nulls))
                .with_data_type(DataType::Decimal64(*p, *s)),
        ),
        _ => Arc::new(
            Decimal128Array::new(
                all.iter()
                    .map(|r| i128::from(r.1) * 1_000_000_007)
                    .collect(),
                Some(nulls),
            )
            .with_data_type(t.clone()),
        ),
    };
    a.slice(junk.len(), rows.len())
}

fn functions(t: &DataType) -> Vec<AggregateFunction> {
    match t {
        DataType::Float64 => vec![AggregateFunction::Sum, AggregateFunction::Avg],
        DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => vec![
            AggregateFunction::Min,
            AggregateFunction::Max,
            AggregateFunction::Count,
            AggregateFunction::Sum,
            AggregateFunction::Avg,
        ],
        _ => vec![
            AggregateFunction::Min,
            AggregateFunction::Max,
            AggregateFunction::Count,
        ],
    }
}

/// An accumulator of `f` over column 0 (and column 2 for a two-argument
/// `count`), masked by column 1 when `masked`.
fn acc(f: &AggregateFunction, t: &DataType, ansi: bool, masked: bool, two: bool) -> Accumulator {
    let mut e = AggregateExpr::new(f.clone(), 0, t.clone(), "a");
    if two {
        e.args.push(2);
    }
    if masked {
        e.mask = Some(1);
    }
    Accumulator::new(&e, ansi).unwrap()
}

type Outcome = (Vec<ArrayRef>, std::result::Result<ArrayRef, String>);

/// A double column as its bit patterns, every NaN as one (an add of two
/// NaNs keeps either operand's sign, as the compiler orders them), so -0.0
/// differs from 0.0; any other column as it is.
fn bits(a: ArrayRef) -> ArrayRef {
    match a.as_primitive_opt::<Float64Type>() {
        Some(f) => Arc::new(UInt64Array::new(
            f.values()
                .iter()
                .map(|v| if v.is_nan() { f64::NAN } else { *v }.to_bits())
                .collect(),
            f.nulls().cloned(),
        )),
        None => a,
    }
}

fn outcome(a: &Accumulator, groups: u32) -> Outcome {
    let ids: Vec<u32> = (0..groups).collect();
    (
        a.state(&ids).unwrap().into_iter().map(bits).collect(),
        a.evaluate(&ids).map(bits).map_err(|e| e.to_string()),
    )
}

/// Folds `batches` (each `(junk, rows)`) into a fresh accumulator; `Err` at
/// the first failing batch.
fn fold(
    f: &AggregateFunction,
    t: &DataType,
    ansi: bool,
    masked: bool,
    two: bool,
    groups: u32,
    batches: &[(Vec<Row>, Vec<Row>)],
) -> std::result::Result<Accumulator, String> {
    fold_weighted(f, t, (ansi, masked, two), 0, groups, batches)
}

/// [`fold`] with weights `mode`: 0 `Const(1)`, 1 a column of ones, 2
/// `Const(3)`.
fn fold_weighted(
    f: &AggregateFunction,
    t: &DataType,
    (ansi, masked, two): (bool, bool, bool),
    mode: u8,
    groups: u32,
    batches: &[(Vec<Row>, Vec<Row>)],
) -> std::result::Result<Accumulator, String> {
    let mut a = acc(f, t, ansi, masked, two);
    a.resize(groups as usize);
    for (junk, rows) in batches {
        let ids: Vec<u32> = rows.iter().map(|r| r.0).collect();
        let arg = column(t, junk, rows);
        // The second argument: NULL where the first is valid on every third row.
        let other: Vec<Row> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| (r.0, r.1, i % 3 != 0 || !r.2))
            .collect();
        let cols: Vec<ArrayRef> = vec![
            arg,
            Arc::new(BooleanArray::from(vec![true; rows.len()])),
            column(t, &[], &other),
        ];
        let w = match mode {
            0 => Weights::Const(1),
            1 => Weights::Column(Int64Array::from(vec![1; rows.len()])),
            _ => Weights::Const(3),
        };
        a.update(&ids, &cols, &w).map_err(|e| e.to_string())?;
    }
    Ok(a)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 96,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    /// Raw rows: the dense kernels give the per-row loop's state, result
    /// and error.
    #[test]
    fn dense_equals_per_row(
        groups in prop_oneof![Just(1_u32), 2..20_u32],
        batches in prop::collection::vec((rows(1), rows(1)), 1..4),
        regroup in any::<u32>(),
        mode in 0..3_u8,
    ) {
        // Group ids drawn for `groups`, from the generated rows' values.
        let batches: Vec<(Vec<Row>, Vec<Row>)> = batches
            .into_iter()
            .map(|(junk, rows)| {
                let at = |r: Row| {
                    u32::try_from(r.1.unsigned_abs().wrapping_add(u64::from(regroup)) % u64::from(groups)).unwrap()
                };
                (junk, rows.into_iter().map(|r| (at(r), r.1, r.2)).collect())
            })
            .collect();
        for t in &TYPES {
            for f in functions(t) {
                for two in [false, true] {
                    if two && f != AggregateFunction::Count {
                        continue;
                    }
                    for ansi in [true, false] {
                        let dense = fold_weighted(&f, t, (ansi, false, two), mode, groups, &batches);
                        let per_row = fold_weighted(&f, t, (ansi, true, two), mode, groups, &batches);
                        match (dense, per_row) {
                            (Ok(d), Ok(p)) => prop_assert_eq!(
                                outcome(&d, groups), outcome(&p, groups),
                                "{:?} over {} (two: {}, ansi: {})", f, t, two, ansi
                            ),
                            (Err(d), Err(p)) => prop_assert_eq!(d, p),
                            (d, p) => prop_assert!(
                                false,
                                "{:?} over {}: dense {:?} vs per-row {:?}",
                                f, t, d.map(|_| ()), p.map(|_| ())
                            ),
                        }
                    }
                }
            }
        }
    }

    /// Partial states of two halves merged (the dense merge) equal one raw
    /// fold of every row; under ANSI only when neither overflowed (partial
    /// sums can overflow where the whole does not, and the reverse).
    #[test]
    fn merged_states_equal_one_fold(
        groups in prop_oneof![Just(1_u32), 2..20_u32],
        a in rows(1),
        b in rows(1),
    ) {
        let regroup = |rows: Vec<Row>| -> Vec<Row> {
            rows.into_iter()
                .map(|r| (u32::try_from(r.1.unsigned_abs() % u64::from(groups)).unwrap(), r.1, r.2))
                .collect()
        };
        let (a, b) = (regroup(a), regroup(b));
        // Double sums depend on the order of the adds, which a merge changes.
        for t in TYPES.iter().filter(|t| **t != DataType::Float64) {
            for f in functions(t).into_iter().filter(|f| *f != AggregateFunction::Avg) {
                for ansi in [true, false] {
                    let whole = fold(&f, t, ansi, true, false, groups, &[(vec![], a.iter().chain(&b).copied().collect())]);
                    let parts = [&a, &b].map(|r| fold(&f, t, ansi, false, false, groups, &[(vec![], r.clone())]));
                    let (Ok(whole), [Ok(pa), Ok(pb)]) = (whole, parts) else {
                        prop_assert!(ansi, "a non-ANSI fold failed: {:?} over {}", f, t);
                        continue;
                    };
                    let ids: Vec<u32> = (0..groups).collect();
                    let mut merged = acc(&f, t, ansi, false, false);
                    merged.resize(groups as usize);
                    let mut ok = true;
                    for p in [&pa, &pb] {
                        if merged.merge(&ids, &p.state(&ids).unwrap()).is_err() {
                            ok = false;
                        }
                    }
                    if ok {
                        prop_assert_eq!(outcome(&merged, groups), outcome(&whole, groups), "{:?} over {}", f, t);
                    } else {
                        prop_assert!(ansi, "a non-ANSI merge failed: {:?} over {}", f, t);
                    }
                }
            }
        }
    }
}

#[test]
fn global_sum_overflow_raises_under_ansi_and_wraps_without() {
    let rows: Vec<Row> = vec![
        (0, i64::MAX, true),
        (0, 1, true),
        (0, -5, false),
        (0, -1, true),
    ];
    for masked in [false, true] {
        let e = fold(
            &AggregateFunction::Sum,
            &DataType::Int64,
            true,
            masked,
            false,
            1,
            &[(vec![], rows.clone())],
        );
        assert!(e.is_err(), "ANSI, masked {masked}");
        let a = fold(
            &AggregateFunction::Sum,
            &DataType::Int64,
            false,
            masked,
            false,
            1,
            &[(vec![], rows.clone())],
        )
        .unwrap();
        let v = a.evaluate(&[0]).unwrap();
        assert_eq!(
            v.as_primitive::<Int64Type>().value(0),
            i64::MAX,
            "masked {masked}"
        );
    }
}

#[test]
fn all_null_batch_leaves_groups_unset() {
    let rows: Vec<Row> = vec![(0, 7, false), (1, 9, false)];
    for groups in [1_u32, 2] {
        let rows: Vec<Row> = rows.iter().map(|r| (r.0 % groups, r.1, r.2)).collect();
        for f in [
            AggregateFunction::Sum,
            AggregateFunction::Min,
            AggregateFunction::Max,
        ] {
            let a = fold(
                &f,
                &DataType::Int64,
                true,
                false,
                false,
                groups,
                &[(vec![], rows.clone())],
            )
            .unwrap();
            let ids: Vec<u32> = (0..groups).collect();
            assert_eq!(
                a.evaluate(&ids).unwrap().null_count(),
                groups as usize,
                "{f:?}"
            );
        }
    }
}

/// `count(NULL)` over an untyped NULL column counts nothing, dense or not.
#[test]
fn count_of_an_untyped_null_column() {
    for groups in [1_u32, 3] {
        let mut outs = Vec::new();
        for masked in [false, true] {
            let mut a = acc(
                &AggregateFunction::Count,
                &DataType::Null,
                true,
                masked,
                false,
            );
            a.resize(groups as usize);
            let ids: Vec<u32> = (0..5).map(|i| i % groups).collect();
            let cols: Vec<ArrayRef> = vec![
                Arc::new(arrow_array::NullArray::new(5)),
                Arc::new(BooleanArray::from(vec![true; 5])),
            ];
            a.update(&ids, &cols, &Weights::Const(1)).unwrap();
            outs.push(outcome(&a, groups));
        }
        assert_eq!(outs[0], outs[1]);
        let counts = outs[0].1.as_ref().unwrap();
        assert!(
            counts
                .as_primitive::<Int64Type>()
                .values()
                .iter()
                .all(|&c| c == 0)
        );
    }
}
