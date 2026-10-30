// SPDX-License-Identifier: Apache-2.0
//! `timestamp_add`, `timestamp_diff` and `make_timestamp` edge cases:
//! lengths, ranges, DST and the NTZ wall clock.

#![expect(clippy::unwrap_used, reason = "test helpers fail loudly")]

use arrow_array::{Array, Decimal128Array, Int32Array, Int64Array, TimestampMicrosecondArray};
use krishiv_common::error::ErrorClass;
use krishiv_kernels::cast::FixedOffset;
use krishiv_kernels::datetime::{
    TimestampFields, TimestampUnit, Zones, make_timestamp, resolve_zone, timestamp_add,
    timestamp_diff,
};

const DAY: i64 = 86_400_000_000;

fn ts(values: &[Option<i64>]) -> TimestampMicrosecondArray {
    TimestampMicrosecondArray::from(values.to_vec())
}

#[test]
fn units_parse_in_any_case_and_print_upper() {
    for name in [
        "year",
        "Quarter",
        "MONTH",
        "week",
        "day",
        "dayofyear",
        "hour",
        "minute",
        "second",
        "millisecond",
        "microsecond",
    ] {
        let unit = TimestampUnit::parse(name).unwrap();
        assert_eq!(unit.name(), name.to_ascii_uppercase());
    }
    assert_eq!(TimestampUnit::parse("decade"), None);
}

#[test]
fn add_rejects_uneven_lengths_and_nulls_pass_through() {
    let q = Int64Array::from(vec![Some(1), None, Some(2)]);
    let t = ts(&[Some(0), Some(0), None]);
    let out = timestamp_add(TimestampUnit::Day, &q, &t, None).unwrap();
    assert_eq!(out.value(0), DAY);
    assert!(out.is_null(1) && out.is_null(2));
    let short = ts(&[Some(0)]);
    let err = timestamp_add(TimestampUnit::Day, &q, &short, None).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError);
}

#[test]
fn add_overflow_is_datetime_overflow_with_the_zone_free_label_for_ntz() {
    let q = Int64Array::from(vec![i64::MAX]);
    // One microsecond past the epoch, so even the time units overflow.
    let t = ts(&[Some(1)]);
    for unit in [
        TimestampUnit::Microsecond,
        TimestampUnit::Hour,
        TimestampUnit::Day,
        TimestampUnit::Week,
        TimestampUnit::Month,
        TimestampUnit::Quarter,
        TimestampUnit::Year,
    ] {
        let err = timestamp_add(unit, &q, &t, None).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::DatetimeOverflow, "{unit:?}");
        assert!(
            err.error()
                .to_string()
                .contains("TIMESTAMP_NTZ '1970-01-01 00:00:00.000001'")
        );
    }
    let err = timestamp_add(TimestampUnit::Day, &q, &t, Some(&FixedOffset(3600))).unwrap_err();
    assert!(
        err.error()
            .to_string()
            .contains("TIMESTAMP '1970-01-01 01:00:00.000001'")
    );
}

#[test]
fn add_across_a_dst_change_keeps_the_wall_clock() {
    let la = resolve_zone("America/Los_Angeles").unwrap();
    // 2022-03-12 20:00 UTC is noon in Los Angeles; the clocks go forward
    // the next night.
    let start = ts(&[Some(1_647_115_200_000_000)]);
    let q = Int64Array::from(vec![1]);
    let day = timestamp_add(TimestampUnit::Day, &q, &start, Some(&la)).unwrap();
    assert_eq!(day.value(0), 1_647_115_200_000_000 + DAY - 3_600_000_000);
    let q24 = Int64Array::from(vec![24]);
    let hours = timestamp_add(TimestampUnit::Hour, &q24, &start, Some(&la)).unwrap();
    assert_eq!(hours.value(0), 1_647_115_200_000_000 + DAY);
}

#[test]
fn diff_counts_toward_zero_and_adjusts_months_by_time_of_day() {
    let a = ts(&[Some(0), Some(0), Some(0)]);
    let b = ts(&[Some(31 * DAY - 1), Some(-(31 * DAY) + 1), Some(31 * DAY)]);
    let months = timestamp_diff(TimestampUnit::Month, &a, &b, None).unwrap();
    // 1970-01-31 23:59:59.999999 is not a full month after the 1st; the
    // mirror image counts toward zero too; 1970-02-01 is exactly one.
    assert_eq!(months.values().to_vec(), vec![0, 0, 1]);
    let years = timestamp_diff(TimestampUnit::Year, &a, &b, None).unwrap();
    assert_eq!(years.values().to_vec(), vec![0, 0, 0]);
    let weeks = timestamp_diff(TimestampUnit::Week, &a, &b, None).unwrap();
    assert_eq!(weeks.values().to_vec(), vec![4, -4, 4]);
}

#[test]
fn diff_nulls_lengths_and_overflow() {
    let a = ts(&[None, Some(0)]);
    let b = ts(&[Some(0), None]);
    let out = timestamp_diff(TimestampUnit::Day, &a, &b, None).unwrap();
    assert!(out.is_null(0) && out.is_null(1));
    let err = timestamp_diff(TimestampUnit::Day, &a, &ts(&[Some(0)]), None).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError);
    let far = timestamp_diff(
        TimestampUnit::Microsecond,
        &ts(&[Some(i64::MIN)]),
        &ts(&[Some(i64::MAX)]),
        None,
    )
    .unwrap_err();
    assert_eq!(far.error_class(), ErrorClass::DatetimeOverflow);
    // Past the range the zone shifts, a local time does not exist.
    let shifted = timestamp_diff(
        TimestampUnit::Day,
        &ts(&[Some(i64::MIN)]),
        &ts(&[Some(0)]),
        Some(&FixedOffset(-3600)),
    )
    .unwrap_err();
    assert_eq!(shifted.error_class(), ErrorClass::DatetimeOverflow);
}

fn seconds(unscaled: &[Option<i128>]) -> Decimal128Array {
    Decimal128Array::from(unscaled.to_vec())
        .with_precision_and_scale(16, 6)
        .unwrap()
}

fn ints(v: &[Option<i32>]) -> Int32Array {
    Int32Array::from(v.to_vec())
}

#[test]
fn make_timestamp_validates_in_javas_order_and_nulls_win() {
    let y = ints(&[Some(2021), None, Some(2021), Some(2021)]);
    let m = ints(&[Some(7), Some(7), Some(7), Some(7)]);
    let d = ints(&[Some(11), Some(11), Some(11), Some(11)]);
    let h = ints(&[Some(6), Some(6), Some(6), Some(6)]);
    let mi = ints(&[Some(30), Some(30), Some(30), Some(30)]);
    let s = seconds(&[Some(45_678_000), Some(1), None, Some(60_000_000)]);
    let fields = TimestampFields {
        years: &y,
        months: &m,
        days: &d,
        hours: &h,
        minutes: &mi,
        seconds: &s,
    };
    let utc = FixedOffset(0);
    let out = make_timestamp(fields, Some(&Zones::One(&utc)), true).unwrap();
    assert_eq!(out.value(0), 1_625_985_045_678_000);
    assert!(out.is_null(1) && out.is_null(2));
    assert_eq!(out.value(3), 1_625_985_060_000_000);
}

fn fields_of<'a>(
    years: &'a Int32Array,
    ones: &'a Int32Array,
    seconds: &'a Decimal128Array,
) -> TimestampFields<'a> {
    TimestampFields {
        years,
        months: ones,
        days: ones,
        hours: ones,
        minutes: ones,
        seconds,
    }
}

#[test]
fn make_timestamp_rejects_bad_shapes_and_overflow() {
    let y = ints(&[Some(2021)]);
    let one = ints(&[Some(1)]);
    let s = seconds(&[Some(0)]);
    let fields = |years, secs| fields_of(years, &one, secs);
    let uneven = ints(&[Some(1), Some(2)]);
    let err = make_timestamp(fields(&uneven, &s), None, true).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError);
    let coarse = Decimal128Array::from(vec![Some(1_i128)])
        .with_precision_and_scale(10, 0)
        .unwrap();
    let err = make_timestamp(fields(&y, &coarse), None, true).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::InternalError);
    // Negative and fractional seconds are out of bounds; the year past the
    // microsecond range overflows rather than wrapping.
    let neg = seconds(&[Some(-1_500_000)]);
    let err = make_timestamp(fields(&y, &neg), None, true).unwrap_err();
    assert!(
        err.error()
            .to_string()
            .contains("SecondOfMinute (valid values 0 - 59): -2")
    );
    assert!(
        make_timestamp(fields(&y, &neg), None, false)
            .unwrap()
            .is_null(0)
    );
    let far = ints(&[Some(999_999_999)]);
    let err = make_timestamp(fields(&far, &s), None, true).unwrap_err();
    assert_eq!(err.error_class(), ErrorClass::DatetimeOverflow);
    assert!(
        make_timestamp(fields(&far, &s), None, false)
            .unwrap()
            .is_null(0)
    );
    // A per-row NULL zone is a NULL result.
    let zones = [None];
    let out = make_timestamp(fields(&y, &s), Some(&Zones::PerRow(&zones)), true).unwrap();
    assert!(out.is_null(0));
}
