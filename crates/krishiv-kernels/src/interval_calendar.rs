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
//! Values are Spark's `(months, days, microseconds)` on the struct carrier
//! (`krishiv_types::calendar`, ADR 0009 update 2026-11-06). The microseconds
//! follow the months and days: `ARITHMETIC_OVERFLOW` ("long overflow") when
//! `exact`, wrapping otherwise, as `IntervalUtils.addExact`/`add` do.

use arrow_array::builder::PrimitiveBuilder;
use arrow_array::types::{Date32Type, TimestampMicrosecondType};
use arrow_array::{Array, ArrayRef, Date32Array, Float64Array, TimestampMicrosecondArray};
pub use krishiv_types::calendar::{CalendarValue, Calendars, calendar_array};

use crate::cast::datetime::{MICROS_PER_DAY, local_to_utc, utc_to_local};
use crate::datetime::{TimeZoneRules, add_months, at, local_to_utc_keeping};
use crate::dispatch::KernelCtx;
use crate::error::KernelError;
use crate::interval::ScaleOp;
use crate::ops::Operand;
use crate::version::IntWidth;

/// Java's `ArithmeticException` message for `Math.toIntExact` and friends.
const INT_OVERFLOW: &str = "integer overflow";
/// Java's `ArithmeticException` message for `Math.addExact(long, long)` and friends.
const LONG_OVERFLOW: &str = "long overflow";

/// The value at row `i` of a column of `len` rows; a one-row column stands
/// for every row.
fn broadcast(a: &Calendars<'_>, i: usize) -> Option<CalendarValue> {
    a.get(if a.len() == 1 { 0 } else { i })
}

fn broadcast_f64(a: &Float64Array, i: usize) -> Option<f64> {
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
    l: &Calendars<'_>,
    r: &Calendars<'_>,
    subtract: bool,
    exact: bool,
) -> Result<ArrayRef, KernelError> {
    let len = rows(l.len(), r.len())?;
    let mut out = Vec::with_capacity(len);
    for row in 0..len {
        let Some((a, b)) = broadcast(l, row).zip(broadcast(r, row)) else {
            out.push(None);
            continue;
        };
        let value = if subtract {
            CalendarValue::new(
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
                long(
                    a.micros,
                    b.micros,
                    i64::checked_sub,
                    i64::wrapping_sub,
                    exact,
                    row,
                )?,
            )
        } else {
            CalendarValue::new(
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
                long(
                    a.micros,
                    b.micros,
                    i64::checked_add,
                    i64::wrapping_add,
                    exact,
                    row,
                )?,
            )
        };
        out.push(Some(value));
    }
    Ok(calendar_array(out))
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

fn long(
    a: i64,
    b: i64,
    checked: fn(i64, i64) -> Option<i64>,
    wrapping: fn(i64, i64) -> i64,
    exact: bool,
    row: usize,
) -> Result<i64, KernelError> {
    if exact {
        checked(a, b).ok_or_else(|| KernelError::arithmetic_overflow(row, LONG_OVERFLOW, None))
    } else {
        Ok(wrapping(a, b))
    }
}

/// `- a`, field by field.
///
/// # Errors
/// As in the module docs.
pub fn negate_calendar(a: &Calendars<'_>, exact: bool) -> Result<ArrayRef, KernelError> {
    let mut out = Vec::with_capacity(a.len());
    for row in 0..a.len() {
        let Some(v) = a.get(row) else {
            out.push(None);
            continue;
        };
        out.push(Some(CalendarValue::new(
            field(0, v.months, i32::checked_sub, i32::wrapping_sub, exact, row)?,
            field(0, v.days, i32::checked_sub, i32::wrapping_sub, exact, row)?,
            long(0, v.micros, i64::checked_sub, i64::wrapping_sub, exact, row)?,
        )));
    }
    Ok(calendar_array(out))
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
/// microseconds, which round as Java's `Math.round` (saturating).
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "Spark works in doubles here, and saturating casts are Java's rule"
)]
fn scaled(
    v: CalendarValue,
    by: impl Fn(f64) -> f64,
    exact: bool,
    row: usize,
) -> Result<CalendarValue, KernelError> {
    let (months, days) = (by(f64::from(v.months)), by(f64::from(v.days)));
    let micros = by(v.micros as f64);
    let (whole_months, whole_days) = if exact {
        (exact_int(months, row)?, exact_int(days, row)?)
    } else {
        (months as i32, days as i32)
    };
    let micros = micros + (MICROS_PER_DAY as f64) * (days - f64::from(whole_days));
    Ok(CalendarValue::new(
        whole_months,
        whole_days,
        java_round(micros),
    ))
}

/// `a * by` or `a / by`. Spark multiplies each field as a double.
///
/// # Errors
/// As in the module docs; `INTERNAL_ERROR` for operands of different lengths.
pub fn scale_calendar(
    a: &Calendars<'_>,
    by: &Float64Array,
    op: ScaleOp,
    exact: bool,
) -> Result<ArrayRef, KernelError> {
    let len = rows(a.len(), by.len())?;
    let mut out = Vec::with_capacity(len);
    for row in 0..len {
        let Some((v, n)) = broadcast(a, row).zip(broadcast_f64(by, row)) else {
            out.push(None);
            continue;
        };
        let value = match op {
            ScaleOp::Multiply => scaled(v, |f| f * n, exact, row)?,
            ScaleOp::Divide if n == 0.0 => {
                if exact {
                    return Err(KernelError::interval_divided_by_zero(row));
                }
                out.push(None);
                continue;
            }
            ScaleOp::Divide => scaled(v, |f| f / n, exact, row)?,
        };
        out.push(Some(value));
    }
    Ok(calendar_array(out))
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

/// The row count of a temporal operand and an interval column, either of which may stand for every row.
fn operand_rows<T: arrow_array::ArrowPrimitiveType>(
    start: &Operand<'_, T>,
    i: &Calendars<'_>,
) -> Result<usize, KernelError> {
    match start {
        Operand::Array(a) => rows(a.len(), i.len()),
        Operand::Scalar(_) => Ok(i.len()),
    }
}

/// `timestamp + interval`, in `zone` (`None` for `TIMESTAMP_NTZ`). Unlike a
/// day-time interval's, only the days field is calendar days: `24 hours` is
/// elapsed time. A result outside the timestamp range is `DATETIME_OVERFLOW`
/// (Spark: an unclassified `ArithmeticException`).
///
/// # Errors
/// `DATETIME_OVERFLOW` as above; `INTERNAL_ERROR` for operands of different
/// lengths.
pub fn timestamp_add_calendar(
    start: Operand<'_, TimestampMicrosecondType>,
    i: &Calendars<'_>,
    zone: Option<&dyn TimeZoneRules>,
) -> Result<TimestampMicrosecondArray, KernelError> {
    let len = operand_rows(&start, i)?;
    let mut out = PrimitiveBuilder::<TimestampMicrosecondType>::with_capacity(len);
    for row in 0..len {
        let Some((t, v)) = at(&start, row).zip(broadcast(i, row)) else {
            out.append_null();
            continue;
        };
        let moved = add_to_instant(t, v.months, v.days, v.micros, zone)
            .ok_or_else(|| KernelError::datetime_overflow(row, "timestamp + interval"))?;
        out.append_value(moved);
    }
    Ok(out.finish())
}

/// `date + interval`. Without `exact`, time in the interval is added to the
/// date's midnight in `zone` (`None` for UTC) and the day taken from the result
/// there, so a daylight-saving change can move it; with `exact`, time is refused.
///
/// # Errors
/// As in the module docs; `ARITHMETIC_OVERFLOW` for a result outside the
/// `DATE` range; `DATETIME_OVERFLOW` for one outside the timestamp range on
/// the path through a timestamp; `INTERNAL_ERROR` for operands of different
/// lengths.
pub fn date_add_calendar(
    start: Operand<'_, Date32Type>,
    i: &Calendars<'_>,
    exact: bool,
    zone: Option<&dyn TimeZoneRules>,
    ctx: &KernelCtx,
) -> Result<Date32Array, KernelError> {
    let len = operand_rows(&start, i)?;
    let message = ctx.spark_version().overflow_message(IntWidth::Int);
    let mut out = PrimitiveBuilder::<Date32Type>::with_capacity(len);
    for row in 0..len {
        let Some((d, v)) = at(&start, row).zip(broadcast(i, row)) else {
            out.append_null();
            continue;
        };
        if v.micros == 0 {
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
                .and_then(|midnight| match zone {
                    Some(tz) => local_to_utc(midnight, tz),
                    None => Some(midnight),
                })
                .and_then(|t| add_to_instant(t, v.months, v.days, v.micros, zone))
                .and_then(|t| match zone {
                    Some(tz) => utc_to_local(t, tz),
                    None => Some(t),
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
    use crate::datetime::resolve_zone;
    use krishiv_common::error::ErrorClass;

    use super::*;

    fn cal(months: i32, days: i32, micros: i64) -> ArrayRef {
        calendar_array([Some(CalendarValue::new(months, days, micros))])
    }

    fn view(a: &ArrayRef) -> Calendars<'_> {
        Calendars::of(a.as_ref()).unwrap()
    }

    fn fields(a: &ArrayRef) -> (i32, i32, i64) {
        let v = view(a).value(0);
        (v.months, v.days, v.micros)
    }

    fn scale(
        v: &ArrayRef,
        n: f64,
        op: ScaleOp,
        exact: bool,
    ) -> Result<(i32, i32, i64), ErrorClass> {
        scale_calendar(&view(v), &Float64Array::from(vec![n]), op, exact)
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
        assert!(scale_calendar(&view(&v), &by, ScaleOp::Divide, false).is_ok_and(|a| a.is_null(0)));
    }

    #[test]
    fn microseconds_past_the_old_nanosecond_field_are_values() {
        // 9e15 us (about 285 years) doubled is 1.8e16 us: past i64 nanoseconds, a value in Spark.
        let v = cal(0, 0, 9_000_000_000_000_000);
        for exact in [true, false] {
            assert_eq!(
                scale(&v, 2.0, ScaleOp::Multiply, exact),
                Ok((0, 0, 18_000_000_000_000_000))
            );
        }
    }

    #[test]
    fn microseconds_overflow_raises_when_exact_and_wraps_otherwise() {
        // `IntervalUtils.addExact` / `add`: `Math.addExact` on the long, or a plain `+`.
        let big = cal(0, 0, i64::MAX);
        let class =
            add_calendar(&view(&big), &view(&big), false, true).map_err(|e| e.error_class());
        assert_eq!(class.map(|_| ()).unwrap_err().name(), "ARITHMETIC_OVERFLOW");
        assert!(
            add_calendar(&view(&big), &view(&big), false, false).is_ok_and(|a| fields(&a).2 == -2)
        );
        assert!(
            add_calendar(&view(&big), &view(&big), true, true).is_ok_and(|a| fields(&a).2 == 0)
        );
        let min = cal(0, 0, i64::MIN);
        assert!(negate_calendar(&view(&min), true).is_err());
        assert!(negate_calendar(&view(&min), false).is_ok_and(|a| fields(&a).2 == i64::MIN));
    }

    #[test]
    fn addition_wraps_unless_exact() {
        let (a, b) = (cal(i32::MAX, 0, 0), cal(1, 0, 0));
        let wrapped = add_calendar(&view(&a), &view(&b), false, false).map(|r| fields(&r));
        assert!(wrapped.is_ok_and(|f| f == (i32::MIN, 0, 0)));
        assert!(add_calendar(&view(&a), &view(&b), false, true).is_err());
        let min = cal(i32::MIN, i32::MIN, 0);
        let negated = negate_calendar(&view(&min), false).map(|r| fields(&r));
        assert!(negated.is_ok_and(|f| f == (i32::MIN, i32::MIN, 0)));
        assert!(negate_calendar(&view(&min), true).is_err());
    }

    #[test]
    fn a_null_on_either_side_is_null_and_a_single_row_stands_for_every_row() {
        let a = calendar_array([Some(CalendarValue::new(1, 0, 0)), None]);
        let one = cal(1, 0, 0);
        let added = add_calendar(&view(&a), &view(&one), true, true);
        assert!(added.is_ok_and(|r| view(&r).value(0).months == 0 && r.is_null(1)));
        let by = Float64Array::from(vec![Some(2.0), None, Some(3.0)]);
        let scaled = scale_calendar(&view(&one), &by, ScaleOp::Multiply, true);
        assert!(scaled.is_ok_and(|r| view(&r).value(2).months == 3 && r.is_null(1)));
        let three = calendar_array([Some(CalendarValue::default()); 3]);
        assert!(add_calendar(&view(&three), &view(&a), false, true).is_err());
    }

    #[test]
    fn a_date_with_time_runs_through_the_zone_it_is_given() {
        let ctx = KernelCtx::default();
        let tz = resolve_zone("America/Los_Angeles").unwrap();
        let add = |day: i32, hours: i64| {
            let i = cal(0, 0, hours * 3_600_000_000);
            date_add_calendar(
                Operand::Scalar(Some(day)),
                &view(&i),
                false,
                Some(&tz),
                &ctx,
            )
            .unwrap()
            .value(0)
        };
        // 2024-03-10 lost an hour there, so 23 hours reach the next midnight.
        assert_eq!(add(19_792, 23), 19_793);
        // 2024-11-03 gained one, so 24 hours stay on the day.
        assert_eq!(add(20_030, 24), 20_030);
    }

    #[test]
    fn a_date_takes_months_then_days_and_time_only_without_exact() {
        let ctx = KernelCtx::default();
        let day = 19_753; // 2024-01-31
        let i = cal(1, 1, 0);
        let moved = date_add_calendar(Operand::Scalar(Some(day)), &view(&i), true, None, &ctx);
        // 2024-02-29, then a day.
        assert!(moved.is_ok_and(|a| a.value(0) == 19_783));
        let hour = cal(0, 0, -3_600_000_000);
        let e = date_add_calendar(Operand::Scalar(Some(day)), &view(&hour), true, None, &ctx)
            .map_err(|e| e.error_class());
        assert_eq!(
            e.map(|_| ()).unwrap_err().name(),
            "INVALID_INTERVAL_WITH_MICROSECONDS_ADDITION"
        );
        let floored =
            date_add_calendar(Operand::Scalar(Some(day)), &view(&hour), false, None, &ctx);
        assert!(floored.is_ok_and(|a| a.value(0) == 19_752));
        let far = cal(0, i32::MAX, 0);
        assert!(
            date_add_calendar(Operand::Scalar(Some(day)), &view(&far), false, None, &ctx).is_err()
        );
        let null = Date32Array::from(vec![None]);
        let out = date_add_calendar(Operand::Array(&null), &view(&i), true, None, &ctx);
        assert!(out.is_ok_and(|a| a.is_null(0)));
    }

    #[test]
    fn a_timestamp_takes_months_days_and_elapsed_time() {
        let day = 86_400_000_000_i64;
        let t = 19_753_i64.wrapping_mul(day);
        let i = cal(1, 1, 1);
        let moved = timestamp_add_calendar(Operand::Scalar(Some(t)), &view(&i), None);
        assert!(moved.is_ok_and(|a| a.value(0) == 19_783_i64.wrapping_mul(day).wrapping_add(1)));
        let far = cal(i32::MAX, 0, 0);
        let e = timestamp_add_calendar(Operand::Scalar(Some(t)), &view(&far), None);
        assert!(e.is_err());
    }
}
