// SPDX-License-Identifier: Apache-2.0
//! One-sided ranges ([`RangeSide`]): a bound from a join's comparison
//! residual (design P3-0.3.1 D2) drops only the rows past the kept end, on
//! rows and in statistics, merges as max-of-max / min-of-min, and never
//! drops a row Spark's comparison would keep.

use std::sync::Arc;

use arrow_array::{Array, Decimal128Array, Float64Array, Int32Array, Int64Array, StringArray};
use arrow_schema::DataType;
use proptest::prelude::*;

use crate::runtime_filter::*;
use crate::{
    CmpOp, ColumnPath, ColumnStatistics, FileStatistics, Precision, PushdownExpr, ScalarValue,
    TriState, prune,
};

/// What a bound collects: the range alone.
fn bound(side: RangeSide) -> KeyFilterOptions {
    KeyFilterOptions {
        max_values: 0,
        exact_max_values: 0,
        bloom: false,
        side,
        ..KeyFilterOptions::default()
    }
}

fn built(values: &dyn Array, side: RangeSide) -> KeyFilter {
    let mut b = KeyFilterBuilder::new(values.data_type().clone(), bound(side))
        .unwrap_or_else(|| panic!("unsupported {}", values.data_type()));
    b.add(values, None).unwrap_or_else(|e| panic!("{e}"));
    b.finish()
}

fn pass(f: &KeyFilter, column: &dyn Array) -> Vec<bool> {
    f.evaluate(column)
        .unwrap_or_else(|| panic!("{} does not widen", column.data_type()))
        .iter()
        .map(|b| b == Some(true))
        .collect()
}

fn stats(min: ScalarValue, max: ScalarValue) -> FileStatistics {
    let mut c = ColumnStatistics::new(ColumnPath::new("q"));
    c.min = Precision::Exact(min);
    c.max = Precision::Exact(max);
    c.null_count = Precision::Exact(0);
    let mut s = FileStatistics::with_rows(Precision::Exact(100));
    s.columns.push(c);
    s
}

#[test]
fn an_upper_bound_drops_only_rows_above_the_largest_value() {
    let f = built(&Int32Array::from(vec![5, 100, 40]), RangeSide::Upper);
    assert_eq!(f.form(), FilterForm::Range);
    assert!(!f.exact_rows());
    let probe = Int32Array::from(vec![Some(i32::MIN), Some(4), Some(100), Some(101), None]);
    assert_eq!(pass(&f, &probe), [true, true, true, false, false]);
    // A wider probe column is compared after widening to the key type.
    let wide = Int64Array::from(vec![i64::MIN, 100, 101]);
    assert!(f.evaluate(&wide).is_none(), "int64 does not widen to int32");
}

#[test]
fn a_lower_bound_drops_only_rows_below_the_smallest_value() {
    let f = built(&Int64Array::from(vec![5, 100, 40]), RangeSide::Lower);
    let probe = Int64Array::from(vec![Some(i64::MIN), Some(4), Some(5), Some(i64::MAX), None]);
    assert_eq!(pass(&f, &probe), [false, false, true, true, false]);
}

#[test]
fn null_build_values_are_not_collected_and_no_rows_means_no_match() {
    // The join feeds only rows whose value is not NULL (`x < NULL` matches
    // nothing); a build of none publishes an empty filter.
    let mut b =
        KeyFilterBuilder::new(DataType::Int32, bound(RangeSide::Upper)).unwrap_or_else(|| panic!());
    let f = b.clone().finish();
    assert!(f.is_empty());
    assert_eq!(
        pass(&f, &Int32Array::from(vec![Some(1), None])),
        [false, false]
    );
    b.add(&Int32Array::from(vec![7]), None)
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(!b.finish().is_empty());
}

#[test]
fn partials_merge_to_the_largest_upper_and_the_smallest_lower_bound() {
    for (side, probe, want) in [
        (
            RangeSide::Upper,
            vec![30, 31, -1_000],
            vec![true, false, true],
        ),
        (
            RangeSide::Lower,
            vec![-6, -5, 1_000],
            vec![false, true, true],
        ),
    ] {
        let mut a = KeyFilterBuilder::new(DataType::Int64, bound(side)).unwrap_or_else(|| panic!());
        let mut b = a.clone();
        a.add(&Int64Array::from(vec![-5, 10]), None)
            .unwrap_or_else(|e| panic!("{e}"));
        b.add(&Int64Array::from(vec![0, 30]), None)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut m = a.clone();
        m.merge(b.clone());
        assert_eq!(pass(&m.finish(), &Int64Array::from(probe.clone())), want);
        // Finished filters of several producers merge the same way.
        let merged = a
            .finish()
            .merge(&b.finish(), &bound(side))
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(pass(&merged, &Int64Array::from(probe)), want);
    }
}

#[test]
fn an_upper_bound_lowers_to_one_comparison_statistics_prune_with() {
    let col = ColumnPath::new("q");
    let f = Arc::new(built(&Int32Array::from(vec![5, 100]), RangeSide::Upper));
    let p = f
        .lower(&col, &DataType::Int32, FilterId(1), 1)
        .unwrap_or_else(|| panic!());
    let PushdownExpr::And(parts) = &p else {
        panic!("{p:?}")
    };
    assert!(
        parts.contains(&PushdownExpr::cmp(
            col,
            CmpOp::LtEq,
            ScalarValue::Int32(100)
        )),
        "{parts:?}"
    );
    let above = stats(ScalarValue::Int32(101), ScalarValue::Int32(500));
    let below = stats(ScalarValue::Int32(i32::MIN), ScalarValue::Int32(4));
    assert_eq!(prune(&p, &above), TriState::False);
    assert_eq!(prune(&p, &below), TriState::Unknown);
}

#[test]
fn float_bounds_follow_spark_nan_ordering() {
    // NaN is larger than every value: `NaN < b` is false for every finite
    // `b`, so an upper bound drops it; `-inf` passes.
    let f = built(&Float64Array::from(vec![1.5, 3.0]), RangeSide::Upper);
    let probe = Float64Array::from(vec![f64::NEG_INFINITY, 3.0, 3.5, f64::NAN, -0.0]);
    assert_eq!(pass(&f, &probe), [true, true, false, false, true]);
    // A NaN build value: `x < NaN` holds for every non-NaN `x`: no range.
    let nan = built(&Float64Array::from(vec![1.5, f64::NAN]), RangeSide::Upper);
    assert!(!nan.has_range());
    assert_eq!(pass(&nan, &probe), [true; 5]);
    // A lower bound cannot hold a NaN probe value (larger than all): no range.
    let lower = built(&Float64Array::from(vec![1.5, 3.0]), RangeSide::Lower);
    assert!(!lower.has_range());
    assert_eq!(pass(&lower, &probe), [true; 5]);
}

#[test]
fn decimal_bounds_open_to_the_precision_limit() {
    let d = |v: Vec<i128>| {
        Decimal128Array::from(v)
            .with_precision_and_scale(7, 2)
            .unwrap_or_else(|e| panic!("{e}"))
    };
    let up = built(&d(vec![150, -20]), RangeSide::Upper);
    assert_eq!(
        pass(&up, &d(vec![-9_999_999, 150, 151])),
        [true, true, false]
    );
    let down = built(&d(vec![150, -20]), RangeSide::Lower);
    assert_eq!(
        pass(&down, &d(vec![-21, -20, 9_999_999])),
        [false, true, true]
    );
    let (lo, hi) = down.range_scalars().unwrap_or_else(|| panic!());
    let unscaled = |s: ScalarValue| match s {
        ScalarValue::Decimal128 { unscaled, .. } => unscaled,
        other => panic!("{other:?}"),
    };
    assert_eq!((unscaled(lo), unscaled(hi)), (-20, 9_999_999));
}

#[test]
fn a_string_upper_bound_keeps_every_smaller_string() {
    let f = built(&StringArray::from(vec!["b", "m"]), RangeSide::Upper);
    let probe = StringArray::from(vec!["", "a", "m", "ma", "z"]);
    assert_eq!(pass(&f, &probe), [true, true, true, false, false]);
    let lower = built(&StringArray::from(vec!["b", "m"]), RangeSide::Lower);
    assert!(!lower.has_range());
}

proptest! {
    /// Never a false negative against the residual it stands for: every
    /// probe value with a build value it compares `<` (`<=`) to passes an
    /// upper bound, and `>` (`>=`) a lower one.
    #[test]
    fn a_bound_keeps_every_probe_value_some_build_value_matches(
        build in proptest::collection::vec(proptest::option::of(-50i64..50), 0..20),
        probe in proptest::collection::vec(proptest::option::of(-60i64..60), 0..40),
    ) {
        let b = Int64Array::from(build.clone());
        let keep = b.nulls().cloned();
        for side in [RangeSide::Upper, RangeSide::Lower] {
            let mut builder = KeyFilterBuilder::new(DataType::Int64, bound(side))
                .unwrap_or_else(|| panic!());
            // As the join feeds it: only rows whose value is not NULL.
            builder.add(&b, keep.as_ref()).unwrap_or_else(|e| panic!("{e}"));
            let f = builder.finish();
            let got = pass(&f, &Int64Array::from(probe.clone()));
            for (p, g) in probe.iter().zip(got) {
                let matched = p.is_some_and(|p| build.iter().flatten().any(|&v| match side {
                    RangeSide::Upper => p <= v,
                    _ => p >= v,
                }));
                prop_assert!(g || !matched, "{side:?} dropped {p:?} of {build:?}");
            }
        }
    }
}

/// A one-sided range with the default membership caps and a Bloom asked
/// for still tests the range alone: a value set `{5}` would drop the `4`
/// the `Upper` range keeps (review of #1029).
#[test]
fn a_one_sided_range_takes_no_membership_form() {
    for side in [RangeSide::Upper, RangeSide::Lower] {
        let options = KeyFilterOptions {
            side,
            bloom: true,
            ..KeyFilterOptions::default()
        };
        let mut b = KeyFilterBuilder::new(DataType::Int32, options).unwrap();
        b.add(&Int32Array::from(vec![5]), None).unwrap();
        let f = b.finish();
        assert_eq!(f.form(), FilterForm::Range, "{side:?}");
        let probe = Int32Array::from(vec![4, 5, 6]);
        let want = match side {
            RangeSide::Upper => vec![true, true, false],
            _ => vec![false, true, true],
        };
        assert_eq!(pass(&f, &probe), want, "{side:?}");
    }
}
