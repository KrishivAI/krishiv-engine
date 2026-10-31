// SPDX-License-Identifier: Apache-2.0
//! Spark datetime functions on `DATE` (`Date32`, days since the epoch),
//! `TIMESTAMP` (`Timestamp(Microsecond, Some(_))`, an instant) and
//! `TIMESTAMP_NTZ` (`Timestamp(Microsecond, None)`, a local date-time):
//! `date_add`/`date_sub`/`datediff`, field extraction (`year`, `month`,
//! `day`, `extract`/`date_part`), `date`/`timestamp` ± interval,
//! `current_date`/`current_timestamp`, and the calendar functions of the
//! `calendar` submodule (`monthname`, `last_day`, `next_day`, `trunc`,
//! `date_trunc`, `months_between`, `make_date`, ...).
//!
//! Time zones come in as [`TimeZoneRules`] (the session zone resolved by the
//! caller, [`FixedOffset`] for `UTC` and `+hh:mm`, [`ZoneRef`] for region
//! ids resolved against the bundled database by [`resolve_zone`]); this
//! module never parses a zone name except through [`resolve_zone`].
//! `current_date` and `current_timestamp` take the query's start
//! instant, so a kernel stays a pure function and one query reads one clock.
//! Dates are proleptic Gregorian, as in Spark 3+.

use std::sync::Arc;

use arrow_array::types::{Date32Type, Int32Type, Int64Type, TimestampMicrosecondType};
use arrow_array::{
    Array, ArrayRef, ArrowPrimitiveType, Date32Array, Decimal128Array, Int32Array, PrimitiveArray,
    TimestampMicrosecondArray,
};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;

pub(crate) mod calendar;
pub(crate) mod format;
pub(crate) mod make;
pub(crate) mod unit;
pub(crate) mod zone;

pub use crate::cast::{FixedOffset, TimeZoneRules};
pub use calendar::{
    TruncUnit, Zones, convert_timezone, day_name, from_utc_timestamp, last_day, make_date,
    month_name, months_between, next_day, to_utc_timestamp, trunc_date, trunc_timestamp, unix_date,
};
pub use format::{
    ParsePolicy, format_timestamps, parse_date_one, parse_dates, parse_ntz, parse_ntz_one,
    parse_timestamp_one, parse_timestamps, seconds_to_timestamps,
};
pub use make::{TimestampFields, make_timestamp};
pub use unit::{TimestampUnit, timestamp_add, timestamp_diff};
pub use zone::{ZoneRef, resolve_zone, zone_id};

use crate::cast::datetime::{
    MICROS_PER_DAY, civil_from_days, days_from_civil, local_to_utc, utc_to_local,
};
use crate::dispatch::KernelCtx;
use crate::error::{KernelError, with_capacity};
use crate::integer;
use crate::ops::{Add, Operand, Sub};
use crate::policy::{JavaWrap, OverflowMode};
use crate::version::IntWidth;

// ---------------------------------------------------------------------
// date_add, date_sub, datediff
// ---------------------------------------------------------------------

/// The days of a `DATE` operand as the `int` operand integer kernels take
/// (`Date32` and `Int32` share their buffer).
fn as_days(a: &Date32Array) -> Int32Array {
    PrimitiveArray::<Int32Type>::new(a.values().clone(), a.nulls().cloned())
}

fn as_dates(a: &Int32Array) -> Date32Array {
    PrimitiveArray::<Date32Type>::new(a.values().clone(), a.nulls().cloned())
}

/// Runs `f` with the `int` view of a date operand.
fn with_days<R>(d: Operand<'_, Date32Type>, f: impl FnOnce(Operand<'_, Int32Type>) -> R) -> R {
    match d {
        Operand::Array(a) => f(Operand::Array(&as_days(a))),
        Operand::Scalar(v) => f(Operand::Scalar(v)),
    }
}

/// Day arithmetic on dates: Java `int` wrap for the wrapping modes, else the
/// checked result is `ARITHMETIC_OVERFLOW` (ANSI; the Spark-version text, no
/// `try_` hint) or NULL (`try_*`).
fn day_arith<B: arrow_array::ArrowPrimitiveType<Native = i32>>(
    a: Operand<'_, Date32Type>,
    b: Operand<'_, B>,
    checked: fn(i32, i32) -> Option<i32>,
    mode: OverflowMode,
    ctx: &KernelCtx,
) -> Result<Date32Array, KernelError> {
    let message = ctx.spark_version().overflow_message(IntWidth::Int);
    map2(a, b, checked, mode == OverflowMode::NullOnOverflow, |row| {
        KernelError::arithmetic_overflow(row, message, None)
    })
}

/// `date_add(start, days)`: `start + days`. `mode` is the ANSI policy of the
/// session: `AnsiError` raises `ARITHMETIC_OVERFLOW` where the day number
/// leaves the `int` range, `NullOnOverflow` (`try_add`) gives NULL; any other
/// mode wraps like Java's `int` addition, as Spark does with ANSI off
/// (`date_add('2016-07-30', 2147483647)` is a date before the epoch). With
/// ANSI on this differs from Spark, which wraps silently (ADR 0023).
///
/// # Errors
/// `ARITHMETIC_OVERFLOW` as above; `INTERNAL_ERROR` for operands of different
/// lengths or two literals.
pub fn date_add(
    start: Operand<'_, Date32Type>,
    days: Operand<'_, Int32Type>,
    mode: OverflowMode,
    ctx: &KernelCtx,
) -> Result<Date32Array, KernelError> {
    if matches!(mode, OverflowMode::AnsiError | OverflowMode::NullOnOverflow) {
        return day_arith(start, days, i32::checked_add, mode, ctx);
    }
    let out = with_days(start, |s| {
        integer::binary::<Int32Type, Add, JavaWrap>(s, days, ctx)
    })?;
    Ok(as_dates(&out))
}

/// `date_sub(start, days)`: `start - days`, with the overflow modes of
/// [`date_add`].
///
/// # Errors
/// As [`date_add`].
pub fn date_sub(
    start: Operand<'_, Date32Type>,
    days: Operand<'_, Int32Type>,
    mode: OverflowMode,
    ctx: &KernelCtx,
) -> Result<Date32Array, KernelError> {
    if matches!(mode, OverflowMode::AnsiError | OverflowMode::NullOnOverflow) {
        return day_arith(start, days, i32::checked_sub, mode, ctx);
    }
    let out = with_days(start, |s| {
        integer::binary::<Int32Type, Sub, JavaWrap>(s, days, ctx)
    })?;
    Ok(as_dates(&out))
}

/// `datediff(end, start)`: the days from `start` to `end`. The difference of
/// two day numbers can leave the `int` range; the overflow modes are those of
/// [`date_add`].
///
/// # Errors
/// As [`date_add`].
pub fn date_diff(
    end: Operand<'_, Date32Type>,
    start: Operand<'_, Date32Type>,
    mode: OverflowMode,
    ctx: &KernelCtx,
) -> Result<Int32Array, KernelError> {
    if matches!(mode, OverflowMode::AnsiError | OverflowMode::NullOnOverflow) {
        let out = day_arith(end, start, i32::checked_sub, mode, ctx)?;
        return Ok(as_days(&out));
    }
    with_days(end, |e| {
        with_days(start, |s| {
            integer::binary::<Int32Type, Sub, JavaWrap>(e, s, ctx)
        })
    })
}

// ---------------------------------------------------------------------
// field extraction
// ---------------------------------------------------------------------

/// A field of `extract(field FROM x)` / `date_part(field, x)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DatePart {
    /// `YEAR`.
    Year,
    /// `YEAROFWEEK`: the ISO week-numbering year.
    YearOfWeek,
    /// `QUARTER`, 1 to 4.
    Quarter,
    /// `MONTH`, 1 to 12.
    Month,
    /// `WEEK`: the ISO week of the year.
    Week,
    /// `DAY`: the day of the month.
    Day,
    /// `DAYOFWEEK` / `DOW`: Sunday is 1.
    DayOfWeek,
    /// `DAYOFWEEK_ISO` / `DOW_ISO`: Monday is 1.
    DayOfWeekIso,
    /// `DOY`: the day of the year.
    DayOfYear,
    /// `HOUR`.
    Hour,
    /// `MINUTE`.
    Minute,
    /// `SECOND`: seconds with microseconds, `decimal(8, 6)`.
    Second,
}

impl DatePart {
    /// The field `name` denotes (case-insensitive, with Spark's aliases:
    /// `Y`, `YRS`, `MON`, `W`, `D`, `H`, `MIN`, `SEC`, ...).
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name.to_ascii_uppercase().as_str() {
            "YEAR" | "Y" | "YEARS" | "YR" | "YRS" => Self::Year,
            "YEAROFWEEK" => Self::YearOfWeek,
            "QUARTER" | "QTR" => Self::Quarter,
            "MONTH" | "MON" | "MONS" | "MONTHS" => Self::Month,
            "WEEK" | "W" | "WEEKS" => Self::Week,
            "DAY" | "D" | "DAYS" => Self::Day,
            "DAYOFWEEK" | "DOW" => Self::DayOfWeek,
            "DAYOFWEEK_ISO" | "DOW_ISO" => Self::DayOfWeekIso,
            "DOY" => Self::DayOfYear,
            "HOUR" | "H" | "HOURS" | "HR" | "HRS" => Self::Hour,
            "MINUTE" | "M" | "MIN" | "MINS" | "MINUTES" => Self::Minute,
            "SECOND" | "S" | "SEC" | "SECONDS" | "SECS" => Self::Second,
            _ => return None,
        })
    }

    /// The Arrow type of the result: `int`, or `decimal(8, 6)` for `SECOND`.
    #[must_use]
    pub fn result_type(self) -> DataType {
        match self {
            Self::Second => DataType::Decimal128(8, 6),
            _ => DataType::Int32,
        }
    }

    /// Whether the field reads the time of day.
    fn is_time(self) -> bool {
        matches!(self, Self::Hour | Self::Minute | Self::Second)
    }
}

/// ISO weekday of a day number: Monday 1 to Sunday 7 (1970-01-01 is a
/// Thursday).
fn iso_weekday(days: i64) -> i64 {
    days.saturating_add(3).rem_euclid(7).saturating_add(1)
}

/// The ISO week-numbering year and week of a day number: the year and week
/// of the Thursday of its week.
fn iso_week(days: i64) -> (i64, i64) {
    let thursday = days.saturating_sub(iso_weekday(days)).saturating_add(4);
    let (year, _, _) = civil_from_days(thursday);
    let jan1 = days_from_civil(year, 1, 1).unwrap_or(0);
    (
        year,
        thursday
            .saturating_sub(jan1)
            .div_euclid(7)
            .saturating_add(1),
    )
}

/// The date field `part` of day number `days`.
fn date_field(part: DatePart, days: i64) -> i64 {
    let (year, month, day) = civil_from_days(days);
    match part {
        DatePart::Year => year,
        DatePart::YearOfWeek => iso_week(days).0,
        DatePart::Quarter => i64::from(month.saturating_sub(1) / 3).saturating_add(1),
        DatePart::Month => i64::from(month),
        DatePart::Week => iso_week(days).1,
        DatePart::Day => i64::from(day),
        DatePart::DayOfWeek => days.saturating_add(4).rem_euclid(7).saturating_add(1),
        DatePart::DayOfWeekIso => iso_weekday(days),
        DatePart::DayOfYear => {
            let jan1 = days_from_civil(year, 1, 1).unwrap_or(0);
            days.saturating_sub(jan1).saturating_add(1)
        }
        // A DATE has no time of day.
        DatePart::Hour | DatePart::Minute | DatePart::Second => 0,
    }
}

/// The time field `part` of `micros` since local midnight (0 to 86.4e9).
fn time_field(part: DatePart, micros_of_day: i64) -> i64 {
    match part {
        DatePart::Hour => micros_of_day / 3_600_000_000,
        DatePart::Minute => micros_of_day / 60_000_000 % 60,
        // Seconds and microseconds, scale 6.
        DatePart::Second => micros_of_day % 60_000_000,
        _ => 0,
    }
}

fn field_array(
    part: DatePart,
    values: Vec<i64>,
    nulls: Option<NullBuffer>,
) -> Result<ArrayRef, KernelError> {
    if part == DatePart::Second {
        let v: Vec<i128> = values.into_iter().map(i128::from).collect();
        let a = Decimal128Array::new(v.into(), nulls)
            .with_precision_and_scale(8, 6)
            .map_err(|e| KernelError::internal(e.to_string()))?;
        return Ok(Arc::new(a));
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "every date field fits an int (a year needs 24 bits)"
    )]
    let v: Vec<i32> = values.into_iter().map(|x| x as i32).collect();
    Ok(Arc::new(Int32Array::new(v.into(), nulls)))
}

/// `extract(part FROM date)`. `HOUR`, `MINUTE` and `SECOND` of a date are 0.
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be allocated.
pub fn extract_date(part: DatePart, a: &Date32Array) -> Result<ArrayRef, KernelError> {
    let mut out = with_capacity::<i64>(a.len())?;
    out.extend(a.values().iter().map(|d| date_field(part, i64::from(*d))));
    field_array(part, out, a.nulls().cloned())
}

/// `extract(part FROM timestamp)`. `zone` is the session zone for a
/// `TIMESTAMP` (an instant); `None` for `TIMESTAMP_NTZ`, whose value already
/// is the local date-time.
///
/// # Errors
/// `DATETIME_OVERFLOW` for an instant whose local time is out of range;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be allocated.
pub fn extract_timestamp(
    part: DatePart,
    a: &TimestampMicrosecondArray,
    zone: Option<&dyn TimeZoneRules>,
) -> Result<ArrayRef, KernelError> {
    let mut out = with_capacity::<i64>(a.len())?;
    for i in 0..a.len() {
        if a.is_null(i) {
            out.push(0);
            continue;
        }
        let micros = a.value(i);
        let local = match zone {
            Some(tz) => utc_to_local(micros, tz)
                .ok_or_else(|| KernelError::datetime_overflow(i, "extract"))?,
            None => micros,
        };
        out.push(if part.is_time() {
            time_field(part, local.rem_euclid(MICROS_PER_DAY))
        } else {
            date_field(part, local.div_euclid(MICROS_PER_DAY))
        });
    }
    field_array(part, out, a.nulls().cloned())
}

// ---------------------------------------------------------------------
// date / timestamp ± interval
// ---------------------------------------------------------------------

/// The number of days of `month` in `year`.
fn month_length(year: i64, month: u32) -> u32 {
    let next = if month == 12 {
        days_from_civil(year.saturating_add(1), 1, 1)
    } else {
        days_from_civil(year, month.saturating_add(1), 1)
    };
    let this = days_from_civil(year, month, 1);
    next.zip(this)
        .and_then(|(n, t)| u32::try_from(n.saturating_sub(t)).ok())
        .unwrap_or(31)
}

/// Day number `months` calendar months after `days`, the day of the month
/// clamped to the target month (`2020-01-31 + 1 month` is `2020-02-29`), as
/// Java's `LocalDate.plusMonths`.
fn add_months(days: i64, months: i64) -> Option<i64> {
    let (year, month, day) = civil_from_days(days);
    let total = year
        .checked_mul(12)?
        .checked_add(i64::from(month).checked_sub(1)?)?
        .checked_add(months)?;
    let (year, month) = (
        total.div_euclid(12),
        u32::try_from(total.rem_euclid(12).checked_add(1)?).ok()?,
    );
    days_from_civil(year, month, day.min(month_length(year, month)))
}

/// The instant of local time `local` (microseconds) in `tz`, keeping the
/// zone offset `preferred` (seconds) where it is valid for `local`, as
/// `java.time.ZonedDateTime.plusDays`/`plusMonths` do: in a DST overlap the
/// result stays on the side of the transition the start was on. Otherwise
/// (not in an overlap, or in a gap) the usual resolution applies.
fn local_to_utc_keeping(local: i64, tz: &dyn TimeZoneRules, preferred: i32) -> Option<i64> {
    let candidate = local.checked_sub(i64::from(preferred).checked_mul(1_000_000)?)?;
    if tz.offset_at_utc(candidate.div_euclid(1_000_000)) == preferred {
        Some(candidate)
    } else {
        local_to_utc(local, tz)
    }
}

/// `date + months` / `date - months` for a year-month interval (`negate`
/// subtracts). A result outside the `DATE` range is `ARITHMETIC_OVERFLOW`
/// (with the context's Spark-version message), or NULL if `null_on_overflow`
/// (`try_add`).
///
/// # Errors
/// `ARITHMETIC_OVERFLOW` as above; `INTERNAL_ERROR` for operands of different
/// lengths or two literals.
pub fn date_add_months(
    start: Operand<'_, Date32Type>,
    months: Operand<'_, Int32Type>,
    negate: bool,
    null_on_overflow: bool,
    ctx: &KernelCtx,
) -> Result<Date32Array, KernelError> {
    let message = ctx.spark_version().overflow_message(IntWidth::Int);
    map2(
        start,
        months,
        |d, m| {
            let m = if negate {
                i64::from(m).saturating_neg()
            } else {
                i64::from(m)
            };
            add_months(i64::from(d), m).and_then(|r| i32::try_from(r).ok())
        },
        null_on_overflow,
        |row| KernelError::arithmetic_overflow(row, message, None),
    )
}

/// `timestamp + months` / `timestamp - months` for a year-month interval:
/// calendar arithmetic on the local date-time in `zone` (`None` for
/// `TIMESTAMP_NTZ`); a local time that does not exist (a DST gap) moves
/// forward, an ambiguous one keeps the start's offset. A result outside the
/// timestamp range is `DATETIME_OVERFLOW` (Spark: an unclassified
/// `ArithmeticException`), or NULL if `null_on_overflow`.
///
/// # Errors
/// `DATETIME_OVERFLOW` as above; `INTERNAL_ERROR` for operands of different
/// lengths or two literals.
pub fn timestamp_add_months(
    start: Operand<'_, TimestampMicrosecondType>,
    months: Operand<'_, Int32Type>,
    negate: bool,
    zone: Option<&dyn TimeZoneRules>,
    null_on_overflow: bool,
) -> Result<TimestampMicrosecondArray, KernelError> {
    map2(
        start,
        months,
        |t, m| {
            let m = if negate {
                i64::from(m).saturating_neg()
            } else {
                i64::from(m)
            };
            let local = match zone {
                Some(tz) => utc_to_local(t, tz)?,
                None => t,
            };
            let days = add_months(local.div_euclid(MICROS_PER_DAY), m)?;
            let local = days
                .checked_mul(MICROS_PER_DAY)?
                .checked_add(local.rem_euclid(MICROS_PER_DAY))?;
            match zone {
                Some(tz) => {
                    local_to_utc_keeping(local, tz, tz.offset_at_utc(t.div_euclid(1_000_000)))
                }
                None => Some(local),
            }
        },
        null_on_overflow,
        |row| KernelError::datetime_overflow(row, "timestamp + interval"),
    )
}

/// `timestamp + interval` / `timestamp - interval` for a day-time interval of
/// `micros` microseconds. Whole days are calendar days in `zone` (`None` for
/// `TIMESTAMP_NTZ`): across a DST change `+ INTERVAL '1' DAY` keeps the wall
/// clock, as does `INTERVAL '24' HOUR`; the remainder below a day is elapsed
/// time. A result outside the `i64` microsecond range is `DATETIME_OVERFLOW`
/// (Spark: an unclassified `ArithmeticException("long overflow")`), or NULL if
/// `null_on_overflow`.
///
/// # Errors
/// `DATETIME_OVERFLOW` as above; `INTERNAL_ERROR` for operands of different
/// lengths or two literals.
pub fn timestamp_add_day_time(
    start: Operand<'_, TimestampMicrosecondType>,
    micros: Operand<'_, Int64Type>,
    negate: bool,
    zone: Option<&dyn TimeZoneRules>,
    null_on_overflow: bool,
) -> Result<TimestampMicrosecondArray, KernelError> {
    map2(
        start,
        micros,
        |t, m| {
            let m = if negate { m.checked_neg()? } else { m };
            let (days, rest) = (m / MICROS_PER_DAY, m % MICROS_PER_DAY);
            let moved = days.checked_mul(MICROS_PER_DAY)?;
            let base = match zone {
                Some(tz) => {
                    let local = utc_to_local(t, tz)?.checked_add(moved)?;
                    local_to_utc_keeping(local, tz, tz.offset_at_utc(t.div_euclid(1_000_000)))?
                }
                None => t.checked_add(moved)?,
            };
            base.checked_add(rest)
        },
        null_on_overflow,
        |row| KernelError::datetime_overflow(row, "timestamp + interval"),
    )
}

/// The value of an operand at row `i`; `None` is NULL.
fn at<T: ArrowPrimitiveType>(o: &Operand<'_, T>, i: usize) -> Option<T::Native> {
    match o {
        Operand::Array(a) => a.is_valid(i).then(|| a.value(i)),
        Operand::Scalar(v) => *v,
    }
}

/// Applies `f` to the pairs of valid rows of two operands. Where `f` returns
/// `None` (overflow) the row is NULL if `null_on_overflow` (`try_add`), else
/// `overflow(row)` is raised for the first such row. A NULL row is never
/// evaluated.
fn map2<A, B, C>(
    a: Operand<'_, A>,
    b: Operand<'_, B>,
    f: impl Fn(A::Native, B::Native) -> Option<C::Native>,
    null_on_overflow: bool,
    overflow: impl Fn(usize) -> KernelError,
) -> Result<PrimitiveArray<C>, KernelError>
where
    A: ArrowPrimitiveType,
    B: ArrowPrimitiveType,
    C: ArrowPrimitiveType,
{
    let (la, lb) = (
        match &a {
            Operand::Array(x) => Some(x.len()),
            Operand::Scalar(_) => None,
        },
        match &b {
            Operand::Array(x) => Some(x.len()),
            Operand::Scalar(_) => None,
        },
    );
    let len = match (la, lb) {
        (Some(x), Some(y)) if x != y => {
            return Err(KernelError::internal(format!(
                "operands differ in length: {x} vs {y}"
            )));
        }
        (Some(n), _) | (_, Some(n)) => n,
        _ => return Err(KernelError::internal("a kernel needs a column operand")),
    };
    let mut values = with_capacity::<C::Native>(len)?;
    let mut valid = Vec::new();
    let mut any_null = false;
    for i in 0..len {
        match at(&a, i).zip(at(&b, i)) {
            Some((x, y)) => match f(x, y) {
                Some(v) => {
                    values.push(v);
                    valid.push(true);
                }
                None if null_on_overflow => {
                    values.push(C::Native::default());
                    valid.push(false);
                    any_null = true;
                }
                None => return Err(overflow(i)),
            },
            None => {
                values.push(C::Native::default());
                valid.push(false);
                any_null = true;
            }
        }
    }
    let nulls = any_null.then(|| NullBuffer::from(valid));
    Ok(PrimitiveArray::<C>::new(values.into(), nulls))
}

// ---------------------------------------------------------------------
// current_date, current_timestamp
// ---------------------------------------------------------------------

/// `current_date()` at the query-start instant `now_utc_micros`: the date in
/// the session zone.
///
/// # Errors
/// `DATETIME_OVERFLOW` for an instant whose local time is out of range.
pub fn current_date(now_utc_micros: i64, zone: &dyn TimeZoneRules) -> Result<i32, KernelError> {
    let local = utc_to_local(now_utc_micros, zone)
        .ok_or_else(|| KernelError::datetime_overflow(0, "current_date"))?;
    i32::try_from(local.div_euclid(MICROS_PER_DAY))
        .map_err(|_| KernelError::datetime_overflow(0, "current_date"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn part_names() {
        for (names, part) in [
            (&["year", "Y", "YRS"][..], DatePart::Year),
            (&["QTR", "quarter"], DatePart::Quarter),
            (&["dow", "DAYOFWEEK"], DatePart::DayOfWeek),
            (&["seconds", "S"], DatePart::Second),
            (&["mins", "M"], DatePart::Minute),
        ] {
            for n in names {
                assert_eq!(DatePart::parse(n), Some(part), "{n}");
            }
        }
        assert_eq!(DatePart::parse("century"), None);
        assert_eq!(DatePart::Second.result_type(), DataType::Decimal128(8, 6));
        assert_eq!(DatePart::Day.result_type(), DataType::Int32);
    }
}
