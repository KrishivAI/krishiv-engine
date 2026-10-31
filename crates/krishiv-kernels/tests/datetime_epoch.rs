// SPDX-License-Identifier: Apache-2.0
//! Edge cases of the Unix-time conversion kernels.

use arrow_array::{
    Array, Decimal128Array, Float64Array, Int32Array, Int64Array, TimestampMicrosecondArray,
};
use krishiv_common::error::ErrorClass;
use krishiv_kernels::datetime::{
    date_from_unix_date, timestamp_from_integral, timestamp_from_seconds,
    timestamp_from_seconds_decimal, timestamp_from_seconds_double, unix_time,
};

#[test]
fn integral_scales_and_reports_the_first_overflowing_row() {
    let v = Int64Array::from(vec![Some(2), None, Some(-3)]);
    let out = timestamp_from_integral(&v, 1_000, "timestamp_millis").unwrap();
    assert_eq!(
        (out.value(0), out.is_null(1), out.value(2)),
        (2_000, true, -3_000)
    );
    let out = timestamp_from_seconds(&v).unwrap();
    assert_eq!(out.value(2), -3_000_000);
    let big = Int64Array::from(vec![Some(1), Some(i64::MAX)]);
    let err = timestamp_from_seconds(&big).unwrap_err();
    assert_eq!(
        (err.error_class(), err.row()),
        (ErrorClass::DatetimeOverflow, Some(1))
    );
    let err = timestamp_from_integral(&big, 1_000, "timestamp_millis").unwrap_err();
    assert!(err.error().to_string().contains("timestamp_millis"));
    assert!(timestamp_from_integral(&big, 1, "timestamp_micros").is_ok());
}

#[test]
fn doubles_truncate_saturate_and_null_the_non_finite() {
    let v = Float64Array::from(vec![
        Some(1.9999999),
        Some(-1.5),
        Some(f64::NAN),
        Some(f64::INFINITY),
        Some(f64::NEG_INFINITY),
        Some(1e30),
        Some(-1e30),
        None,
    ]);
    let out = timestamp_from_seconds_double(&v).unwrap();
    assert_eq!(out.value(0), 1_999_999);
    assert_eq!(out.value(1), -1_500_000);
    assert!((2..5).all(|i| out.is_null(i)));
    assert_eq!((out.value(5), out.value(6)), (i64::MAX, i64::MIN));
    assert!(out.is_null(7));
}

#[test]
fn decimals_must_fit_microseconds_exactly() {
    let d = |unscaled: i128, precision: u8, scale: i8| {
        Decimal128Array::from(vec![Some(unscaled)])
            .with_precision_and_scale(precision, scale)
            .unwrap()
    };
    // Scale below, at and above six; trailing zeros past six are exact.
    assert_eq!(
        timestamp_from_seconds_decimal(&d(15, 3, 1))
            .unwrap()
            .value(0),
        1_500_000
    );
    assert_eq!(
        timestamp_from_seconds_decimal(&d(1_234_500, 7, 6))
            .unwrap()
            .value(0),
        1_234_500
    );
    assert_eq!(
        timestamp_from_seconds_decimal(&d(12_345_000_000, 20, 10))
            .unwrap()
            .value(0),
        1_234_500
    );
    for (unscaled, precision, scale) in [(1_234_567, 7, 7), (i128::from(i64::MAX), 38, 0)] {
        let err = timestamp_from_seconds_decimal(&d(unscaled, precision, scale)).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::DatetimeOverflow);
    }
    let nulls = Decimal128Array::from(vec![None::<i128>])
        .with_precision_and_scale(10, 2)
        .unwrap();
    assert!(timestamp_from_seconds_decimal(&nulls).unwrap().is_null(0));
}

#[test]
fn unix_time_rounds_toward_negative_infinity() {
    let t = TimestampMicrosecondArray::from(vec![Some(-1), Some(999_999), Some(1_000_000), None]);
    assert_eq!(
        unix_time(&t, 1_000_000).unwrap().iter().collect::<Vec<_>>(),
        [Some(-1), Some(0), Some(1), None]
    );
    assert_eq!(unix_time(&t, 1_000).unwrap().value(0), -1);
    assert_eq!(unix_time(&t, 1).unwrap().value(0), -1);
}

#[test]
fn unix_dates_pass_every_int_through() {
    let days = Int32Array::from(vec![Some(i32::MAX), Some(i32::MIN), None, Some(0)]);
    let out = date_from_unix_date(&days);
    assert_eq!(
        out.iter().collect::<Vec<_>>(),
        [Some(i32::MAX), Some(i32::MIN), None, Some(0)]
    );
}
