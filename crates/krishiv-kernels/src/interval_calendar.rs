// SPDX-License-Identifier: Apache-2.0
//! Calendar interval arithmetic, following Spark 4.2.0's `IntervalUtils` and
//! `DateTimeUtils`. `exact` is Spark's `ansiEnabled` on the expression, which
//! `try_*` forces on:
//!
//! | Operator | `exact` | Otherwise |
//! |---|---|---|
//! | `+`, `-`, unary `-` | `ARITHMETIC_OVERFLOW` | a field wraps |
//! | `*`, `/` | `ARITHMETIC_OVERFLOW` | a field saturates |
//! | `/ 0` | `INTERVAL_DIVIDED_BY_ZERO` | NULL |
//! | `date + time` | `INVALID_INTERVAL_WITH_MICROSECONDS_ADDITION` | added in UTC, then the day |
//!
//! A value holds more time than the nanosecond field of its Arrow type can
//! (about 292 years) only where Spark's `long` of microseconds still can, so
//! it is refused by name whatever `exact` says: Spark answers it, so NULL
//! would be a wrong result.

use arrow_array::builder::PrimitiveBuilder;
use arrow_array::types::{Date32Type, IntervalMonthDayNanoType, TimestampMicrosecondType};
use arrow_array::{
    Array, ArrowPrimitiveType, Date32Array, Float64Array, IntervalMonthDayNanoArray,
    PrimitiveArray, TimestampMicrosecondArray,
};
use arrow_buffer::IntervalMonthDayNano;

use crate::cast::datetime::{MICROS_PER_DAY, utc_to_local};
use crate::datetime::{TimeZoneRules, add_months, at, local_to_utc_keeping, map2};
use crate::dispatch::KernelCtx;
use crate::error::KernelError;
use crate::interval::ScaleOp;
use crate::ops::Operand;
use crate::version::IntWidth;

const NANOS_PER_MICRO: i64 = 1000;

/// Java's `ArithmeticException` message for `Math.toIntExact` and friends.
const INT_OVERFLOW: &str = "integer overflow";

fn past_nanos() -> KernelError {
    KernelError::unsupported(
        "a calendar interval whose microseconds overflow the nanosecond field of its Arrow type",
    )
}

/// The value at row `i` of a column of `len` rows; a one-row column stands
/// for every row.
fn broadcast<T: ArrowPrimitiveType>(a: &PrimitiveArray<T>, i: usize) -> Option<T::Native> {
    let i = if a.len() == 1 { 0 } else { i };
    a.is_valid(i).then(|| a.value(i))
}

/// The row count of two columns, either of which may stand for every row.
fn rows(a: usize, b: usize) -> Result<usize, KernelError> {
    if a == 0 || b == 0 {
        return Ok(0);
    }
    if a != b && a != 1 && b != 1 {
        return Err(KernelError::internal(format!(
            "operands differ in length: {a} vs {b}"
        )));
    }
    Ok(a.max(b))
}

/// `l + r` or `l - r`, field by field.
///
/// # Errors
/// As in the module docs; `INTERNAL_ERROR` for operands of different lengths.
pub fn add_calendar(
    l: &IntervalMonthDayNanoArray,
    r: &IntervalMonthDayNanoArray,
    subtract: bool,
    exact: bool,
) -> Result<IntervalMonthDayNanoArray, KernelError> {
    let len = rows(l.len(), r.len())?;
    let mut out = PrimitiveBuilder::<IntervalMonthDayNanoType>::with_capacity(len);
    for row in 0..len {
        let Some((a, b)) = broadcast(l, row).zip(broadcast(r, row)) else {
            out.append_null();
            continue;
        };
        let (months, days, nanos) = if subtract {
            (
                field(
                    a.months,
                    b.months,
                    i32::checked_sub,
                    i32::wrapping_sub,
                    exact,
                    row,
                )?,
                field(
                    a.days,
                    b.days,
                    i32::checked_sub,
                    i32::wrapping_sub,
                    exact,
                    row,
                )?,
                a.nanoseconds.checked_sub(b.nanoseconds),
            )
        } else {
            (
                field(
                    a.months,
                    b.months,
                    i32::checked_add,
                    i32::wrapping_add,
                    exact,
                    row,
                )?,
                field(
                    a.days,
                    b.days,
                    i32::checked_add,
                    i32::wrapping_add,
                    exact,
                    row,
                )?,
                a.nanoseconds.checked_add(b.nanoseconds),
            )
        };
        let nanos = nanos.ok_or_else(past_nanos)?;
        out.append_value(IntervalMonthDayNano::new(months, days, nanos));
    }
    Ok(out.finish())
}

fn field(
    a: i32,
    b: i32,
    checked: fn(i32, i32) -> Option<i32>,
    wrapping: fn(i32, i32) -> i32,
    exact: bool,
    row: usize,
) -> Result<i32, KernelError> {
    if exact {
        checked(a, b).ok_or_else(|| KernelError::arithmetic_overflow(row, INT_OVERFLOW, None))
    } else {
        Ok(wrapping(a, b))
    }
}

/// `- a`, field by field.
///
/// # Errors
/// As in the module docs.
pub fn negate_calendar(
    a: &IntervalMonthDayNanoArray,
    exact: bool,
) -> Result<IntervalMonthDayNanoArray, KernelError> {
    let mut out = PrimitiveBuilder::<IntervalMonthDayNanoType>::with_capacity(a.len());
    for row in 0..a.len() {
        if a.is_null(row) {
            out.append_null();
            continue;
        }
        let v = a.value(row);
        let months = field(0, v.months, i32::checked_sub, i32::wrapping_sub, exact, row)?;
        let days = field(0, v.days, i32::checked_sub, i32::wrapping_sub, exact, row)?;
        let nanos = v.nanoseconds.checked_neg().ok_or_else(past_nanos)?;
        out.append_value(IntervalMonthDayNano::new(months, days, nanos));
    }
    Ok(out.finish())
}

/// Java's `Math.round(double)`: halves go up, NaN is 0, and past the `long`
/// range it saturates. Not `x + 0.5`, which rounds the largest double below
/// 0.5 up to 1.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a saturating cast is Java's rule here"
)]
fn java_round(x: f64) -> i64 {
    let floor = x.floor();
    (if x - floor >= 0.5 { floor + 1.0 } else { floor }) as i64
}

/// `Math.toIntExact(x.toLong)`, Spark's exact truncation of a field.
#[expect(
    clippy::cast_possible_truncation,
    reason = "a saturating cast is Java's rule here"
)]
fn exact_int(x: f64, row: usize) -> Result<i32, KernelError> {
    i32::try_from(x as i64).map_err(|_| KernelError::arithmetic_overflow(row, INT_OVERFLOW, None))
}

/// The fields of `v` scaled by a function of each field as a double. A
/// fraction of a month is dropped; a fraction of a day moves to the
/// microseconds.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Spark works in doubles here, and saturating casts are Java's rule"
)]
fn scaled(
    v: IntervalMonthDayNano,
    by: impl Fn(f64) -> f64,
    exact: bool,
    row: usize,
) -> Result<IntervalMonthDayNano, KernelError> {
    let (months, days) = (by(f64::from(v.months)), by(f64::from(v.days)));
    let micros = by((v.nanoseconds / NANOS_PER_MICRO) as f64);
    let (whole_months, whole_days) = if exact {
        (exact_int(months, row)?, exact_int(days, row)?)
    } else {
        (months as i32, days as i32)
    };
    let micros = micros + (MICROS_PER_DAY as f64) * (days - f64::from(whole_days));
    let nanos = java_round(micros)
        .checked_mul(NANOS_PER_MICRO)
        .ok_or_else(past_nanos)?;
    Ok(IntervalMonthDayNano::new(whole_months, whole_days, nanos))
}

/// `a * by` or `a / by`. Spark multiplies each field as a double.
///
/// # Errors
/// As in the module docs; `INTERNAL_ERROR` for operands of different lengths.
pub fn scale_calendar(
    a: &IntervalMonthDayNanoArray,
    by: &Float64Array,
    op: ScaleOp,
    exact: bool,
) -> Result<IntervalMonthDayNanoArray, KernelError> {
    let len = rows(a.len(), by.len())?;
    let mut out = PrimitiveBuilder::<IntervalMonthDayNanoType>::with_capacity(len);
    for row in 0..len {
        let Some((v, n)) = broadcast(a, row).zip(broadcast(by, row)) else {
            out.append_null();
            continue;
        };
        let value = match op {
            ScaleOp::Multiply => scaled(v, |f| f * n, exact, row)?,
            ScaleOp::Divide if n == 0.0 => {
                if exact {
                    return Err(KernelError::interval_divided_by_zero(row));
                }
                out.append_null();
                continue;
            }
            ScaleOp::Divide => scaled(v, |f| f / n, exact, row)?,
        };
        out.append_value(value);
    }
    Ok(out.finish())
}

/// `ZonedDateTime.plusMonths(months).plusDays(days).plus(micros)`: each step
/// resolves on its own, so a DST gap reached by the months moves the time
/// before the days are added. `None` is out of range.
fn add_to_instant(
    t: i64,
    months: i32,
    days: i32,
    micros: i64,
    zone: Option<&dyn TimeZoneRules>,
) -> Option<i64> {
    let on_date = |t: i64, change: &dyn Fn(i64) -> Option<i64>| -> Option<i64> {
        let local = match zone {
            Some(tz) => utc_to_local(t, tz)?,
            None => t,
        };
        let moved = change(local.div_euclid(MICROS_PER_DAY))?
            .checked_mul(MICROS_PER_DAY)?
            .checked_add(local.rem_euclid(MICROS_PER_DAY))?;
        match zone {
            Some(tz) => local_to_utc_keeping(moved, tz, tz.offset_at_utc(t.div_euclid(1_000_000))),
            None => Some(moved),
        }
    };
    let mut t = t;
    if months != 0 {
        t = on_date(t, &|d| add_months(d, i64::from(months)))?;
    }
    if days != 0 {
        t = on_date(t, &|d| d.checked_add(i64::from(days)))?;
    }
    t.checked_add(micros)
}

/// `timestamp + interval`, in `zone` (`None` for `TIMESTAMP_NTZ`). Unlike a
/// day-time interval's, only the days field is calendar days: `24 hours` is
/// elapsed time. A result outside the timestamp range is `DATETIME_OVERFLOW`
/// (Spark: an unclassified `ArithmeticException`).
///
/// # Errors
/// `DATETIME_OVERFLOW` as above; `INTERNAL_ERROR` for operands of different
/// lengths or two literals.
pub fn timestamp_add_calendar(
    start: Operand<'_, TimestampMicrosecondType>,
    i: Operand<'_, IntervalMonthDayNanoType>,
    zone: Option<&dyn TimeZoneRules>,
) -> Result<TimestampMicrosecondArray, KernelError> {
    map2(
        start,
        i,
        |t, v| add_to_instant(t, v.months, v.days, v.nanoseconds / NANOS_PER_MICRO, zone),
        false,
        |row| KernelError::datetime_overflow(row, "timestamp + interval"),
    )
}

/// `date + interval`. Without `exact`, time in the interval is added in UTC
/// and the day taken from the result; with it, time is refused.
///
/// # Errors
/// As in the module docs; `ARITHMETIC_OVERFLOW` for a result outside the
/// `DATE` range; `DATETIME_OVERFLOW` for one outside the timestamp range on
/// the path through UTC; `INTERNAL_ERROR` for operands of different lengths or
/// two literals.
pub fn date_add_calendar(
    start: Operand<'_, Date32Type>,
    i: Operand<'_, IntervalMonthDayNanoType>,
    exact: bool,
    ctx: &KernelCtx,
) -> Result<Date32Array, KernelError> {
    let len = match (&start, &i) {
        (Operand::Array(a), Operand::Array(b)) if a.len() != b.len() => {
            return Err(KernelError::internal(format!(
                "operands differ in length: {} vs {}",
                a.len(),
                b.len()
            )));
        }
        (Operand::Array(a), _) => a.len(),
        (_, Operand::Array(b)) => b.len(),
        _ => return Err(KernelError::internal("a kernel needs a column operand")),
    };
    let message = ctx.spark_version().overflow_message(IntWidth::Int);
    let mut out = PrimitiveBuilder::<Date32Type>::with_capacity(len);
    for row in 0..len {
        let Some((d, v)) = at(&start, row).zip(at(&i, row)) else {
            out.append_null();
            continue;
        };
        if v.nanoseconds == 0 {
            let moved = add_months(i64::from(d), i64::from(v.months))
                .and_then(|m| m.checked_add(i64::from(v.days)))
                .and_then(|m| i32::try_from(m).ok())
                .ok_or_else(|| KernelError::arithmetic_overflow(row, message, None))?;
            out.append_value(moved);
        } else if exact {
            return Err(KernelError::invalid_interval_with_micros_addition(row));
        } else {
            let day = i64::from(d)
                .checked_mul(MICROS_PER_DAY)
                .and_then(|t| {
                    add_to_instant(t, v.months, v.days, v.nanoseconds / NANOS_PER_MICRO, None)
                })
                .ok_or_else(|| KernelError::datetime_overflow(row, "date + interval"))?
                .div_euclid(MICROS_PER_DAY);
            out.append_value(
                i32::try_from(day)
                    .map_err(|_| KernelError::arithmetic_overflow(row, message, None))?,
            );
        }
    }
    Ok(out.finish())
}

#[cfg(test)]
mod tests {
    use krishiv_common::error::ErrorClass;

    use super::*;

    fn cal(months: i32, days: i32, micros: i64) -> IntervalMonthDayNanoArray {
        IntervalMonthDayNanoArray::from(vec![IntervalMonthDayNano::new(
            months,
            days,
            micros.wrapping_mul(NANOS_PER_MICRO),
        )])
    }

    fn fields(a: &IntervalMonthDayNanoArray) -> (i32, i32, i64) {
        let v = a.value(0);
        (v.months, v.days, v.nanoseconds / NANOS_PER_MICRO)
    }

    fn scale(
        v: &IntervalMonthDayNanoArray,
        n: f64,
        op: ScaleOp,
        exact: bool,
    ) -> Result<(i32, i32, i64), ErrorClass> {
        scale_calendar(v, &Float64Array::from(vec![n]), op, exact)
            .map(|a| fields(&a))
            .map_err(|e| e.error_class())
    }

    #[test]
    fn java_round_goes_up_at_a_half_and_saturates() {
        assert_eq!(java_round(1.5), 2);
        assert_eq!(java_round(-1.5), -1);
        assert_eq!(java_round(0.499_999_999_999_999_94), 0);
        assert_eq!(java_round(f64::NAN), 0);
        assert_eq!(java_round(f64::INFINITY), i64::MAX);
    }

    #[test]
    fn a_fraction_of_a_month_drops_and_of_a_day_moves_to_the_time() {
        let v = cal(3, 3, 3);
        assert_eq!(
            scale(&v, 0.5, ScaleOp::Multiply, true),
            Ok((1, 1, 43_200_000_002))
        );
        assert_eq!(
            scale(&v, 2.0, ScaleOp::Divide, false),
            Ok((1, 1, 43_200_000_002))
        );
    }

    #[test]
    fn scaling_past_a_field_raises_only_when_exact() {
        let v = cal(i32::MAX, 0, 0);
        let class = scale(&v, 2.0, ScaleOp::Multiply, true).unwrap_err();
        assert_eq!(class.name(), "ARITHMETIC_OVERFLOW");
        assert_eq!(
            scale(&v, 2.0, ScaleOp::Multiply, false),
            Ok((i32::MAX, 0, 0))
        );
    }

    #[test]
    fn nan_scales_to_nothing() {
        let v = cal(1, 2, 3);
        for exact in [true, false] {
            assert_eq!(scale(&v, f64::NAN, ScaleOp::Multiply, exact), Ok((0, 0, 0)));
            assert_eq!(scale(&v, f64::NAN, ScaleOp::Divide, exact), Ok((0, 0, 0)));
        }
    }

    #[test]
    fn division_by_zero_raises_when_exact_and_is_null_otherwise() {
        let v = cal(1, 0, 0);
        let class = scale(&v, -0.0, ScaleOp::Divide, true).unwrap_err();
        assert_eq!(class.name(), "INTERVAL_DIVIDED_BY_ZERO");
        let by = Float64Array::from(vec![0.0]);
        assert!(scale_calendar(&v, &by, ScaleOp::Divide, false).is_ok_and(|a| a.is_null(0)));
    }

    #[test]
    fn time_past_the_nanosecond_field_is_refused_whatever_the_mode() {
        let v = cal(0, 0, 9_000_000_000_000_000);
        let by = Float64Array::from(vec![2.0]);
        for exact in [true, false] {
            let e = scale_calendar(&v, &by, ScaleOp::Multiply, exact).map_err(|e| e.error_class());
            assert_eq!(
                e.map(|_| ()).unwrap_err().name(),
                "KRISHIV_UNSUPPORTED_FEATURE",
            );
        }
        let big = IntervalMonthDayNanoArray::from(vec![IntervalMonthDayNano::new(0, 0, i64::MAX)]);
        assert!(add_calendar(&big, &big, false, false).is_err());
        assert!(add_calendar(&big, &big, true, false).is_ok());
        let min = IntervalMonthDayNanoArray::from(vec![IntervalMonthDayNano::new(0, 0, i64::MIN)]);
        assert!(negate_calendar(&min, false).is_err());
    }

    #[test]
    fn addition_wraps_unless_exact() {
        let (a, b) = (cal(i32::MAX, 0, 0), cal(1, 0, 0));
        let wrapped = add_calendar(&a, &b, false, false).map(|r| fields(&r));
        assert!(wrapped.is_ok_and(|f| f == (i32::MIN, 0, 0)));
        assert!(add_calendar(&a, &b, false, true).is_err());
        let min = cal(i32::MIN, i32::MIN, 0);
        let negated = negate_calendar(&min, false).map(|r| fields(&r));
        assert!(negated.is_ok_and(|f| f == (i32::MIN, i32::MIN, 0)));
        assert!(negate_calendar(&min, true).is_err());
    }

    #[test]
    fn a_null_on_either_side_is_null_and_a_single_row_stands_for_every_row() {
        let a =
            IntervalMonthDayNanoArray::from(vec![Some(IntervalMonthDayNano::new(1, 0, 0)), None]);
        let added = add_calendar(&a, &cal(1, 0, 0), true, true);
        assert!(added.is_ok_and(|r| r.value(0).months == 0 && r.is_null(1)));
        let short = cal(1, 0, 0);
        let by = Float64Array::from(vec![Some(2.0), None, Some(3.0)]);
        let scaled = scale_calendar(&short, &by, ScaleOp::Multiply, true);
        assert!(scaled.is_ok_and(|r| r.value(2).months == 3 && r.is_null(1)));
        let three = IntervalMonthDayNanoArray::from(vec![IntervalMonthDayNano::ZERO; 3]);
        assert!(add_calendar(&three, &a, false, true).is_err());
    }

    #[test]
    fn a_date_takes_months_then_days_and_time_only_without_exact() {
        let ctx = KernelCtx::default();
        let day = 19_753; // 2024-01-31
        let start = Operand::Scalar(Some(day));
        let i = cal(1, 1, 0);
        let moved = date_add_calendar(start, Operand::Array(&i), true, &ctx);
        // 2024-02-29, then a day.
        assert!(moved.is_ok_and(|a| a.value(0) == 19_783));
        let hour = cal(0, 0, -3_600_000_000);
        let e = date_add_calendar(
            Operand::Scalar(Some(day)),
            Operand::Array(&hour),
            true,
            &ctx,
        )
        .map_err(|e| e.error_class());
        assert_eq!(
            e.map(|_| ()).unwrap_err().name(),
            "INVALID_INTERVAL_WITH_MICROSECONDS_ADDITION"
        );
        let floored = date_add_calendar(
            Operand::Scalar(Some(day)),
            Operand::Array(&hour),
            false,
            &ctx,
        );
        assert!(floored.is_ok_and(|a| a.value(0) == 19_752));
        let far = cal(0, i32::MAX, 0);
        assert!(
            date_add_calendar(
                Operand::Scalar(Some(day)),
                Operand::Array(&far),
                false,
                &ctx
            )
            .is_err()
        );
        let null = Date32Array::from(vec![None]);
        let out = date_add_calendar(Operand::Array(&null), Operand::Array(&i), true, &ctx);
        assert!(out.is_ok_and(|a| a.is_null(0)));
    }

    #[test]
    fn a_timestamp_takes_months_days_and_elapsed_time() {
        let day = 86_400_000_000_i64;
        let t = 19_753_i64.wrapping_mul(day);
        let i = cal(1, 1, 1);
        let moved = timestamp_add_calendar(Operand::Scalar(Some(t)), Operand::Array(&i), None);
        assert!(moved.is_ok_and(|a| a.value(0) == 19_783_i64.wrapping_mul(day).wrapping_add(1)));
        let far = cal(i32::MAX, 0, 0);
        let e = timestamp_add_calendar(Operand::Scalar(Some(t)), Operand::Array(&far), None);
        assert!(e.is_err());
    }
}
