// SPDX-License-Identifier: Apache-2.0
//! Unit tests of the accumulator: decimal conversion, running folds and edge cases.

use super::*;

/// A double becomes a decimal through its shortest text, rounded half up
/// (Spark's `BigDecimal.valueOf(d).setScale(scale, HALF_UP)`).
#[test]
fn a_double_becomes_a_decimal_through_its_text() {
    for (v, scale, want) in [
        (1.15, 2, 115),
        (-1.15, 2, -115),
        // 1.005 is below 1.005 in binary, but its text rounds up.
        (1.005, 2, 101),
        (2.5, 0, 3),
        (-2.5, 0, -3),
        (0.0, 2, 0),
        (123.456, 1, 1235),
        (0.000_001, 2, 0),
        (1e20, 2, 10_i128.pow(22)),
    ] {
        assert_eq!(decimal_unscaled(v, scale), want, "{v} at scale {scale}");
    }
    assert_eq!(decimal_unscaled(f64::INFINITY, 2), i128::MAX);
    assert_eq!(decimal_unscaled(f64::NEG_INFINITY, 2), i128::MIN);
}

/// `compact_views` reaches the views inside a `LargeList` of strings: its values are copied
/// into buffers of their own, with the same content.
#[test]
fn compact_views_walks_a_large_list() {
    use arrow_array::{LargeListArray, StringViewArray};
    use arrow_schema::Field;
    let values = StringViewArray::from(vec![
        Some("a string longer than twelve bytes, in the first row"),
        Some("another string longer than twelve bytes, in the second"),
    ]);
    let field = Arc::new(Field::new("item", DataType::Utf8View, true));
    let list = LargeListArray::try_new(
        field,
        arrow_buffer::OffsetBuffer::new(vec![0_i64, 1, 2].into()),
        Arc::new(values),
        None,
    )
    .unwrap();
    let before: ArrayRef = Arc::new(list);
    let compacted = compact_views(Arc::clone(&before));
    let buffer =
        |a: &ArrayRef| a.as_list::<i64>().values().as_string_view().data_buffers()[0].as_ptr();
    assert_ne!(buffer(&compacted), buffer(&before), "the views were copied");
    assert_eq!(
        compacted
            .as_list::<i64>()
            .value(1)
            .as_string_view()
            .value(0),
        "another string longer than twelve bytes, in the second"
    );
}

#[test]
fn avg_decimal_rounds_like_spark() {
    // 5 / 3 at decimal(38,22) (Spark: 1.6666666666666666666667).
    let (v, _) = avg_decimal(5_000_000_000_000_000_000, 18, 3, 38, 22);
    assert_eq!(plain(v.unwrap(), 22), "1.6666666666666666666667");
    let (v, _) = avg_decimal(-5, 0, 3, 38, 4);
    assert_eq!(plain(v.unwrap(), 4), "-1.6667");
    // Spark: avg of 1e20 - 1e-18 and 1 does not fit decimal(38,22).
    let sum = 100_000_000_000_000_000_000_999_999_999_999_999_999i128;
    let (v, text) = avg_decimal(sum, 18, 2, 38, 22);
    assert_eq!(v, None);
    assert_eq!(
        text,
        "50000000000000000000.499999999999999999500000000000000000000"
    );
    let big = 10i128.pow(38) - 1;
    let (v, text) = avg_decimal(big, 0, 1, 38, 4);
    assert_eq!(v, None);
    assert!(text.starts_with("99999999999999999999999999999999999999.000"));
    // Half-up carry through every digit.
    let (v, _) = avg_decimal(2, 0, 3, 10, 0);
    assert_eq!(v, Some(1));
    assert!(round_up(&mut [9, 9]));
    let long = "x".repeat(100);
    let all: ArrayRef = Arc::new(StringArray::from(vec![long.as_str(), "ab"]));
    assert_eq!(value_bytes(&all.slice(1, 1)), 2);
    let views: ArrayRef = Arc::new(StringViewArray::from(vec![long.as_str(), "ab"]));
    assert_eq!(value_bytes(&views.slice(1, 1)), 2);
    let bin: ArrayRef = Arc::new(BinaryArray::from(vec![&b"abc"[..]]));
    assert_eq!(value_bytes(&bin), 3);
    let bv: ArrayRef = Arc::new(BinaryViewArray::from(vec![&b"abcd"[..]]));
    assert_eq!(value_bytes(&bv), 4);
    assert_eq!(plain(-5, 3), "-0.005");
    assert!(float_lt(1.0, f64::NAN) && !float_lt(f64::NAN, f64::NAN) && !float_lt(-0.0, 0.0));
}

/// Beyond `i128`, the exact average rounds as Spark's (`HALF_UP` to
/// scale 39, then to the result's scale; expected values from Python's
/// `decimal`), and reports the scale-39 quotient when it does not fit.
#[test]
fn wide_averages_round_as_spark() {
    let wide = |v: i128, mul: i128| {
        i256::from_i128(v)
            .checked_mul(i256::from_i128(mul))
            .unwrap()
    };
    let three = wide(3, 10_i128.pow(38));
    assert_eq!(
        avg_decimal_wide(three, 0, 7, 38, 0).0,
        Some(42_857_142_857_142_857_142_857_142_857_142_857_143)
    );
    let (none, text) = avg_decimal_wide(three, 0, 7, 38, 2);
    assert_eq!(none, None);
    assert_eq!(
        text,
        "42857142857142857142857142857142857142.857142857142857142857142857142857142857"
    );
    let negative = wide(-5, 10_i128.pow(38)).wrapping_add(i256::from_i128(3));
    assert_eq!(
        avg_decimal_wide(negative, 0, 9, 38, 0).0,
        Some(-55_555_555_555_555_555_555_555_555_555_555_555_555)
    );
    let half = wide(2, 10_i128.pow(38)).wrapping_add(i256::ONE);
    assert_eq!(
        avg_decimal_wide(half, 0, 4, 38, 0).0,
        Some(50_000_000_000_000_000_000_000_000_000_000_000_000)
    );
    // Within `i128` it is `avg_decimal` itself.
    assert_eq!(
        avg_decimal_wide(i256::from_i128(15), 1, 2, 10, 5),
        (Some(75_000), String::new())
    );
    assert_eq!(
        avg_decimal_wide(i256::from_i128(10_i128.pow(38) - 1), 0, 1, 38, 4).0,
        None
    );
}

#[test]
fn wide_sums_are_checked_when_evaluated() {
    let big = i256::from_i128(10_i128.pow(38));
    assert_eq!(wide_sum_value(None, 38, 0, true).unwrap(), None);
    assert_eq!(
        wide_sum_value(Some(i256::from_i128(-7)), 3, 1, true).unwrap(),
        Some(-7)
    );
    assert_eq!(wide_sum_value(Some(big), 38, 0, false).unwrap(), None);
    let e = wide_sum_value(Some(big.wrapping_neg()), 38, 2, true).unwrap_err();
    assert!(
        e.to_string()
            .contains("-1000000000000000000000000000000000000.00"),
        "{e}"
    );
}

/// A running fold keeps, after each row, the state the row-by-row fold
/// of its partition has: every snapshot evaluates as the aggregate of
/// the rows folded so far (NULLs and unfolded rows included), for every
/// kind a window evaluates this way; the rest are refused.
#[test]
fn running_snapshots_equal_row_by_row_folds() {
    use AggregateFunction as F;
    let ints: ArrayRef = Arc::new(Int64Array::from(vec![
        Some(3),
        None,
        Some(-7),
        Some(5),
        None,
        Some(2),
        Some(9),
    ]));
    let floats: ArrayRef = Arc::new(Float64Array::from(vec![
        Some(0.5),
        None,
        Some(-1.25),
        Some(2.0),
        None,
        Some(8.0),
        Some(-3.5),
    ]));
    // Partition 0: rows 0..4, partition 1: rows 4..7; group 2 takes rows
    // the filter drops (rows 3 and 6); a snapshot after every row.
    let (n, parts_of) = (7, [0_u32, 0, 0, 0, 1, 1, 1]);
    let groups = [0_u32, 0, 0, 2, 1, 1, 2];
    let snapshots: Vec<u32> = (3..3 + 7).collect();
    for (col, t) in [(&ints, DataType::Int64), (&floats, DataType::Float64)] {
        for f in [
            &F::Count,
            &F::Sum,
            &F::Avg,
            &F::Min,
            &F::Max,
            &F::First {
                ignore_nulls: false,
            },
            &F::First { ignore_nulls: true },
            &F::Last {
                ignore_nulls: false,
            },
            &F::Last { ignore_nulls: true },
            &F::Mode,
            &F::ModeMin,
            &F::ModeMax,
        ] {
            let expr = AggregateExpr::new(f.clone(), 0, t.clone(), "x");
            let mut fast = Accumulator::new(&expr, true).unwrap();
            fast.resize(3 + n);
            // Two calls: a snapshot never needs rows of the same call.
            for (lo, hi) in [(0, 3), (3, 7)] {
                fast.update_running(
                    &groups[lo..hi],
                    &[col.slice(lo, hi - lo)],
                    &Running {
                        parts: &parts_of[lo..hi],
                        snapshots: &snapshots[lo..hi],
                    },
                )
                .unwrap();
            }
            let ids: Vec<u32> = snapshots.clone();
            let got = fast.evaluate(&ids).unwrap();
            // The same rows, one at a time, evaluated after each.
            let mut slow = Accumulator::new(&expr, true).unwrap();
            slow.resize(3);
            for row in 0..n {
                slow.update(&[groups[row]], &[col.slice(row, 1)], &Weights::Const(1))
                    .unwrap();
                let want = slow.evaluate(&[parts_of[row]]).unwrap();
                assert_eq!(
                    got.slice(row, 1).to_data(),
                    want.to_data(),
                    "{f:?} over {t:?}, row {row}"
                );
            }
        }
    }
    // Rows without a snapshot only fold.
    let expr = AggregateExpr::new(F::Sum, 0, DataType::Int64, "x");
    let mut a = Accumulator::new(&expr, true).unwrap();
    a.resize(2);
    a.update_running(
        &[0, 0],
        &[ints.slice(0, 2)],
        &Running {
            parts: &[0, 0],
            snapshots: &[Running::NONE, Running::NONE],
        },
    )
    .unwrap();
    assert_eq!(
        a.evaluate(&[0])
            .unwrap()
            .as_primitive::<Int64Type>()
            .value(0),
        3
    );
    // Kinds a window evaluates elsewhere, and bad shapes, are errors.
    for (f, t) in [
        (&F::Sum, DataType::Decimal128(10, 2)),
        (&F::Max, DataType::Utf8View),
    ] {
        let mut a =
            Accumulator::new(&AggregateExpr::new(f.clone(), 0, t.clone(), "x"), true).unwrap();
        a.resize(2);
        let col: ArrayRef = Arc::new(Int64Array::from(vec![1]));
        let run = Running {
            parts: &[0],
            snapshots: &[1],
        };
        let e = a.update_running(&[0], &[col], &run).unwrap_err();
        assert!(
            e.to_string().contains("no running fold"),
            "{f:?} {t:?}: {e}"
        );
    }
    let mut a = Accumulator::new(&expr, true).unwrap();
    a.resize(2);
    let short = Running {
        parts: &[0],
        snapshots: &[],
    };
    let e = a
        .update_running(&[0, 0], &[ints.slice(0, 2)], &short)
        .unwrap_err();
    assert!(e.to_string().contains("different lengths"), "{e}");
}
