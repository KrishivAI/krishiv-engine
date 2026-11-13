// SPDX-License-Identifier: Apache-2.0
//! `make_ym_interval`, `make_dt_interval` and `make_interval`, following Spark
//! 4.2.0's `IntervalUtils`: any NULL field gives NULL, and each step overflows
//! on its own, in Spark's order, so a later negative field can't rescue an
//! earlier overflow.
//!
//! | Function | Overflow |
//! |---|---|
//! | `make_ym_interval`, `make_dt_interval` | `INTERVAL_ARITHMETIC_OVERFLOW.WITHOUT_SUGGESTION`, ANSI on or off |
//! | `make_interval` | `ARITHMETIC_OVERFLOW` when `fail_on_error`, else NULL |
//!
//! Seconds are a `DECIMAL(18,6)`, so their unscaled value is microseconds.

use arrow_array::builder::PrimitiveBuilder;
use arrow_array::types::{DurationMicrosecondType, IntervalYearMonthType};
use arrow_array::{
    Array, ArrayRef, Decimal128Array, DurationMicrosecondArray, Int32Array, IntervalYearMonthArray,
};

use crate::error::KernelError;

const MONTHS_PER_YEAR: i32 = 12;
const DAYS_PER_WEEK: i32 = 7;
const MICROS_PER_MINUTE: i64 = 60_000_000;
const MICROS_PER_HOUR: i64 = 3_600_000_000;
const MICROS_PER_DAY: i64 = 86_400_000_000;

/// Java's `ArithmeticException` messages, which `make_interval` reports.
const INT_OVERFLOW: &str = "integer overflow";
const LONG_OVERFLOW: &str = "long overflow";

/// `micros + field * per`, each step checked.
fn add_field(micros: i64, field: i32, per: i64) -> Option<i64> {
    micros.checked_add(i64::from(field).checked_mul(per)?)
}

fn same_length(function: &str, rows: usize, lengths: &[usize]) -> Result<(), KernelError> {
    if lengths.iter().all(|&n| n == rows) {
        Ok(())
    } else {
        Err(KernelError::internal(format!(
            "{function} arguments of different lengths"
        )))
    }
}

/// The microseconds of a `DECIMAL(18,6)` seconds value.
fn second_micros(seconds: &Decimal128Array, row: usize) -> Result<i64, KernelError> {
    i64::try_from(seconds.value(row))
        .map_err(|_| KernelError::internal("interval seconds do not fit DECIMAL(18,6)"))
}

fn scale_six(function: &str, seconds: &Decimal128Array) -> Result<(), KernelError> {
    if seconds.scale() == 6 {
        Ok(())
    } else {
        Err(KernelError::internal(format!(
            "{function} seconds are not a decimal of scale 6"
        )))
    }
}

/// `make_ym_interval(years, months)`.
///
/// # Errors
/// `INTERVAL_ARITHMETIC_OVERFLOW.WITHOUT_SUGGESTION`; `INTERNAL_ERROR` for
/// inputs of different lengths.
pub fn make_year_month(
    years: &Int32Array,
    months: &Int32Array,
) -> Result<IntervalYearMonthArray, KernelError> {
    same_length("make_ym_interval", years.len(), &[months.len()])?;
    let mut out = PrimitiveBuilder::<IntervalYearMonthType>::with_capacity(years.len());
    for (row, (y, m)) in years.iter().zip(months).enumerate() {
        if let (Some(y), Some(m)) = (y, m) {
            let total = y
                .checked_mul(MONTHS_PER_YEAR)
                .and_then(|y| m.checked_add(y))
                .ok_or_else(|| KernelError::interval_overflow(row, None))?;
            out.append_value(total);
        } else {
            out.append_null();
        }
    }
    Ok(out.finish())
}

/// The fields of `make_dt_interval`.
#[derive(Clone, Copy, Debug)]
pub struct DayTimeFields<'a> {
    /// Days.
    pub days: &'a Int32Array,
    /// Hours.
    pub hours: &'a Int32Array,
    /// Minutes.
    pub minutes: &'a Int32Array,
    /// Seconds, a `DECIMAL(18,6)`.
    pub seconds: &'a Decimal128Array,
}

/// `make_dt_interval(days, hours, mins, secs)`.
///
/// # Errors
/// `INTERVAL_ARITHMETIC_OVERFLOW.WITHOUT_SUGGESTION`; `INTERNAL_ERROR` for
/// inputs of different lengths or seconds that are not scale 6.
pub fn make_day_time(f: DayTimeFields<'_>) -> Result<DurationMicrosecondArray, KernelError> {
    let rows = f.days.len();
    same_length(
        "make_dt_interval",
        rows,
        &[f.hours.len(), f.minutes.len(), f.seconds.len()],
    )?;
    scale_six("make_dt_interval", f.seconds)?;
    let mut out = PrimitiveBuilder::<DurationMicrosecondType>::with_capacity(rows);
    for row in 0..rows {
        if f.days.is_null(row)
            || f.hours.is_null(row)
            || f.minutes.is_null(row)
            || f.seconds.is_null(row)
        {
            out.append_null();
            continue;
        }
        let micros = add_field(
            second_micros(f.seconds, row)?,
            f.days.value(row),
            MICROS_PER_DAY,
        )
        .and_then(|m| add_field(m, f.hours.value(row), MICROS_PER_HOUR))
        .and_then(|m| add_field(m, f.minutes.value(row), MICROS_PER_MINUTE))
        .ok_or_else(|| KernelError::interval_overflow(row, None))?;
        out.append_value(micros);
    }
    Ok(out.finish())
}

/// The fields of `make_interval`.
#[derive(Clone, Copy, Debug)]
pub struct CalendarFields<'a> {
    /// Years.
    pub years: &'a Int32Array,
    /// Months.
    pub months: &'a Int32Array,
    /// Weeks.
    pub weeks: &'a Int32Array,
    /// Days.
    pub days: &'a Int32Array,
    /// Hours.
    pub hours: &'a Int32Array,
    /// Minutes.
    pub minutes: &'a Int32Array,
    /// Seconds, a `DECIMAL(18,6)`.
    pub seconds: &'a Decimal128Array,
}

impl CalendarFields<'_> {
    fn any_null(&self, row: usize) -> bool {
        [
            self.years,
            self.months,
            self.weeks,
            self.days,
            self.hours,
            self.minutes,
        ]
        .iter()
        .any(|a| a.is_null(row))
            || self.seconds.is_null(row)
    }

    /// Months, days and microseconds, or the message of the overflow.
    fn at(&self, row: usize) -> Result<Result<(i32, i32, i64), &'static str>, KernelError> {
        let micros = second_micros(self.seconds, row)?;
        let months = self
            .years
            .value(row)
            .checked_mul(MONTHS_PER_YEAR)
            .and_then(|y| self.months.value(row).checked_add(y));
        let days = self
            .weeks
            .value(row)
            .checked_mul(DAYS_PER_WEEK)
            .and_then(|w| self.days.value(row).checked_add(w));
        let (Some(months), Some(days)) = (months, days) else {
            return Ok(Err(INT_OVERFLOW));
        };
        let micros = add_field(micros, self.hours.value(row), MICROS_PER_HOUR)
            .and_then(|m| add_field(m, self.minutes.value(row), MICROS_PER_MINUTE));
        Ok(micros.map(|m| (months, days, m)).ok_or(LONG_OVERFLOW))
    }
}

/// `make_interval(years, months, weeks, days, hours, mins, secs)`, on the calendar interval's struct of
/// microseconds (Spark's `CalendarInterval`, so every value Spark computes is held).
///
/// # Errors
/// `ARITHMETIC_OVERFLOW` when `fail_on_error`; `INTERNAL_ERROR` for inputs of different lengths or seconds that
/// are not scale 6.
pub fn make_calendar(f: CalendarFields<'_>, fail_on_error: bool) -> Result<ArrayRef, KernelError> {
    let rows = f.years.len();
    same_length(
        "make_interval",
        rows,
        &[
            f.months.len(),
            f.weeks.len(),
            f.days.len(),
            f.hours.len(),
            f.minutes.len(),
            f.seconds.len(),
        ],
    )?;
    scale_six("make_interval", f.seconds)?;
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        if f.any_null(row) {
            out.push(None);
            continue;
        }
        match f.at(row)? {
            Ok((months, days, micros)) => {
                out.push(Some(krishiv_types::calendar::CalendarValue::new(
                    months, days, micros,
                )));
            }
            Err(message) if fail_on_error => {
                return Err(KernelError::arithmetic_overflow(row, message, None));
            }
            Err(_) => out.push(None),
        }
    }
    Ok(krishiv_types::calendar::calendar_array(out))
}

#[cfg(test)]
mod tests {
    use krishiv_common::error::ErrorClass;

    use super::*;

    fn ints(v: &[Option<i32>]) -> Int32Array {
        Int32Array::from(v.to_vec())
    }

    fn seconds(v: &[Option<i128>]) -> Decimal128Array {
        Decimal128Array::from(v.to_vec())
            .with_precision_and_scale(18, 6)
            .unwrap()
    }

    fn calendar(v: [i32; 6], secs: i128, fail: bool) -> Result<ArrayRef, KernelError> {
        let [years, months, weeks, days, hours, minutes] = v.map(|x| ints(&[Some(x)]));
        make_calendar(
            CalendarFields {
                years: &years,
                months: &months,
                weeks: &weeks,
                days: &days,
                hours: &hours,
                minutes: &minutes,
                seconds: &seconds(&[Some(secs)]),
            },
            fail,
        )
    }

    #[test]
    fn year_month_adds_after_the_years_multiply_without_overflow() {
        let out = make_year_month(
            &ints(&[Some(1), None, Some(-178_956_970)]),
            &ints(&[Some(2), Some(1), Some(-8)]),
        )
        .unwrap();
        assert_eq!(out.value(0), 14);
        assert!(out.is_null(1));
        assert_eq!(out.value(2), i32::MIN);
        // The years alone are past an int, whatever the months would give.
        let e = make_year_month(&ints(&[Some(178_956_971)]), &ints(&[Some(-100)])).unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::IntervalArithmeticOverflowWithoutSuggestion
        );
    }

    #[test]
    fn day_time_sums_its_fields_in_order() {
        let (d, h, m) = (
            ints(&[Some(1), None]),
            ints(&[Some(2), Some(0)]),
            ints(&[Some(3), Some(0)]),
        );
        let s = seconds(&[Some(4_500_000), Some(0)]);
        let f = DayTimeFields {
            days: &d,
            hours: &h,
            minutes: &m,
            seconds: &s,
        };
        let out = make_day_time(f).unwrap();
        assert_eq!(out.value(0), 93_784_500_000);
        assert!(out.is_null(1));

        // `5` hours overflow before `-60` minutes would bring it back.
        let (d, h, m) = (
            ints(&[Some(106_751_991)]),
            ints(&[Some(5)]),
            ints(&[Some(-60)]),
        );
        let s = seconds(&[Some(54_775_807)]);
        let e = make_day_time(DayTimeFields {
            days: &d,
            hours: &h,
            minutes: &m,
            seconds: &s,
        })
        .unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::IntervalArithmeticOverflowWithoutSuggestion
        );
    }

    #[test]
    fn calendar_keeps_its_three_fields_apart() {
        let out = calendar([1, 2, 3, 4, 5, 6], 7_500_000, true).unwrap();
        let v = krishiv_types::calendar::Calendars::of(out.as_ref())
            .unwrap()
            .value(0);
        assert_eq!((v.months, v.days), (14, 25));
        assert_eq!(v.micros, 18_367_500_000);
    }

    #[test]
    fn calendar_overflow_raises_or_is_null() {
        for v in [
            [i32::MAX, 12, 0, 0, 0, 0],
            [0, 0, i32::MAX, 8, 0, 0],
            // Weeks past an int, whatever the days would give.
            [0, 0, 306_783_379, -100, 0, 0],
        ] {
            let e = calendar(v, 0, true).unwrap_err();
            assert_eq!(e.error_class(), ErrorClass::ArithmeticOverflow);
            assert!(e.to_string().contains(INT_OVERFLOW), "{e}");
            assert!(calendar(v, 0, false).unwrap().is_null(0));
        }
    }

    #[test]
    fn calendar_time_past_the_old_nanosecond_field_is_a_value() {
        // The largest fields still fit a long of microseconds, so Spark answers these (`interval` #72's
        // `make_interval(1, 2, 3, 4, 0, 0, 123456789012.123456)` among them).
        let max_seconds = 999_999_999_999_999_999_i128;
        for fail in [true, false] {
            for (v, secs) in [
                ([0, 0, 0, 0, 0, 0], max_seconds),
                ([0, 0, 0, 0, i32::MAX, i32::MAX], max_seconds),
                ([0, 0, 0, 0, i32::MIN, i32::MIN], -max_seconds),
            ] {
                assert!(calendar(v, secs, fail).is_ok_and(|a| a.is_valid(0)));
            }
        }
        let out = calendar([1, 2, 3, 4, 0, 0], 123_456_789_012_123_456, true).unwrap();
        let v = krishiv_types::calendar::Calendars::of(out.as_ref())
            .unwrap()
            .value(0);
        assert_eq!(
            (v.months, v.days, v.micros),
            (14, 25, 123_456_789_012_123_456)
        );
    }

    #[test]
    fn calendar_null_field_is_null_without_reading_the_others() {
        let null = ints(&[None]);
        let one = ints(&[Some(i32::MAX)]);
        let s = seconds(&[Some(0)]);
        let out = make_calendar(
            CalendarFields {
                years: &one,
                months: &one,
                weeks: &one,
                days: &null,
                hours: &one,
                minutes: &one,
                seconds: &s,
            },
            true,
        )
        .unwrap();
        assert!(out.is_null(0));
    }

    #[test]
    fn mismatched_inputs_are_internal_errors() {
        let (one, two) = (ints(&[Some(1)]), ints(&[Some(1), Some(2)]));
        let e = make_year_month(&one, &two).unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError);
        let bad_scale = Decimal128Array::from(vec![Some(1_i128)])
            .with_precision_and_scale(18, 2)
            .unwrap();
        let e = make_day_time(DayTimeFields {
            days: &one,
            hours: &one,
            minutes: &one,
            seconds: &bad_scale,
        })
        .unwrap_err();
        assert_eq!(e.error_class(), ErrorClass::InternalError);
    }
}
