// SPDX-License-Identifier: Apache-2.0
//! `timestampadd(unit, n, ts)` and `timestampdiff(unit, a, b)`, and the
//! unit forms of `dateadd`/`date_add` and `datediff`/`date_diff` that
//! Spark's parser maps onto them.
//!
//! Both work on local date-times, as Spark's `DateTimeUtils.timestampAdd`
//! and `timestampDiff` do: for a `TIMESTAMP` the session zone converts the
//! instant to its wall clock and back (a day is a calendar day across a DST
//! change), for a `TIMESTAMP_NTZ` the value already is the wall clock.

use arrow_array::{Array, Int64Array, TimestampMicrosecondArray};

use super::{TimeZoneRules, add_months, local_to_utc_keeping};
use crate::cast::datetime::{MICROS_PER_DAY, civil_from_days, format_timestamp, utc_to_local};
use crate::error::{KernelError, with_capacity};

/// The unit of `timestampadd` and `timestampdiff` (Spark's `datetimeUnit`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimestampUnit {
    /// `YEAR`.
    Year,
    /// `QUARTER`: three months.
    Quarter,
    /// `MONTH`.
    Month,
    /// `WEEK`: seven calendar days.
    Week,
    /// `DAY`: a calendar day.
    Day,
    /// `DAYOFYEAR`: the same as `DAY`.
    DayOfYear,
    /// `HOUR`.
    Hour,
    /// `MINUTE`.
    Minute,
    /// `SECOND`.
    Second,
    /// `MILLISECOND`.
    Millisecond,
    /// `MICROSECOND`.
    Microsecond,
}

impl TimestampUnit {
    /// The unit `text` names, in any case; `None` for anything else.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text.to_ascii_uppercase().as_str() {
            "YEAR" => Self::Year,
            "QUARTER" => Self::Quarter,
            "MONTH" => Self::Month,
            "WEEK" => Self::Week,
            "DAY" => Self::Day,
            "DAYOFYEAR" => Self::DayOfYear,
            "HOUR" => Self::Hour,
            "MINUTE" => Self::Minute,
            "SECOND" => Self::Second,
            "MILLISECOND" => Self::Millisecond,
            "MICROSECOND" => Self::Microsecond,
            _ => return None,
        })
    }

    /// The upper-case name, as error messages spell it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Year => "YEAR",
            Self::Quarter => "QUARTER",
            Self::Month => "MONTH",
            Self::Week => "WEEK",
            Self::Day => "DAY",
            Self::DayOfYear => "DAYOFYEAR",
            Self::Hour => "HOUR",
            Self::Minute => "MINUTE",
            Self::Second => "SECOND",
            Self::Millisecond => "MILLISECOND",
            Self::Microsecond => "MICROSECOND",
        }
    }

    /// The length in microseconds of a unit that is a fixed span of time
    /// (calendar days and weeks are `Day`/`Week`, not fixed under DST).
    const fn fixed_micros(self) -> Option<i64> {
        match self {
            Self::Hour => Some(3_600_000_000),
            Self::Minute => Some(60_000_000),
            Self::Second => Some(1_000_000),
            Self::Millisecond => Some(1_000),
            Self::Microsecond => Some(1),
            _ => None,
        }
    }

    /// The length in months of a calendar-month unit.
    const fn months(self) -> Option<i64> {
        match self {
            Self::Month => Some(1),
            Self::Quarter => Some(3),
            Self::Year => Some(12),
            _ => None,
        }
    }

    /// The length in calendar days of a day unit.
    const fn days(self) -> Option<i64> {
        match self {
            Self::Day | Self::DayOfYear => Some(1),
            Self::Week => Some(7),
            _ => None,
        }
    }
}

/// `start + quantity * unit`. `zone` is the session zone for a `TIMESTAMP`
/// and `None` for a `TIMESTAMP_NTZ`. Calendar units move the local
/// date-time (a local time inside a DST gap moves forward, an ambiguous one
/// keeps the start's offset, as `ZonedDateTime.plusDays` does); time units
/// add elapsed time. NULL in, NULL out.
///
/// # Errors
/// `DATETIME_OVERFLOW` when the result is out of range, with ANSI on or off
/// (Spark raises `timestampAddOverflowError` either way); `INTERNAL_ERROR`
/// for arrays of different lengths.
pub fn timestamp_add(
    unit: TimestampUnit,
    quantity: &Int64Array,
    start: &TimestampMicrosecondArray,
    zone: Option<&dyn TimeZoneRules>,
) -> Result<TimestampMicrosecondArray, KernelError> {
    if quantity.len() != start.len() {
        return Err(KernelError::internal(
            "timestampadd arguments of different lengths",
        ));
    }
    let mut out = with_capacity::<i64>(start.len())?;
    let mut valid = with_capacity::<bool>(start.len())?;
    for row in 0..start.len() {
        if quantity.is_null(row) || start.is_null(row) {
            out.push(0);
            valid.push(false);
            continue;
        }
        let (n, t) = (quantity.value(row), start.value(row));
        let Some(moved) = add_one(unit, n, t, zone) else {
            return Err(overflow(unit, n, t, zone, row));
        };
        out.push(moved);
        valid.push(true);
    }
    let nulls = arrow_buffer::NullBuffer::from(valid);
    Ok(TimestampMicrosecondArray::new(out.into(), Some(nulls)))
}

fn add_one(unit: TimestampUnit, n: i64, t: i64, zone: Option<&dyn TimeZoneRules>) -> Option<i64> {
    if let Some(span) = unit.fixed_micros() {
        return t.checked_add(n.checked_mul(span)?);
    }
    let local = match zone {
        Some(tz) => utc_to_local(t, tz)?,
        None => t,
    };
    let days = local.div_euclid(MICROS_PER_DAY);
    let moved = match (unit.days(), unit.months()) {
        (Some(per), _) => days.checked_add(n.checked_mul(per)?)?,
        (_, Some(per)) => add_months(days, n.checked_mul(per)?)?,
        _ => return None,
    };
    let local = moved
        .checked_mul(MICROS_PER_DAY)?
        .checked_add(local.rem_euclid(MICROS_PER_DAY))?;
    match zone {
        Some(tz) => local_to_utc_keeping(local, tz, tz.offset_at_utc(t.div_euclid(1_000_000))),
        None => Some(local),
    }
}

/// Spark's message: `add 9223372036854775807L YEAR to TIMESTAMP '2022-02-14
/// 01:02:03'`.
fn overflow(
    unit: TimestampUnit,
    n: i64,
    t: i64,
    zone: Option<&dyn TimeZoneRules>,
    row: usize,
) -> KernelError {
    let (keyword, local) = match zone {
        Some(tz) => ("TIMESTAMP", utc_to_local(t, tz).unwrap_or(t)),
        None => ("TIMESTAMP_NTZ", t),
    };
    let mut shown = String::new();
    format_timestamp(local, &mut shown);
    KernelError::datetime_overflow(
        row,
        &format!("add {n}L {} to {keyword} '{shown}'", unit.name()),
    )
}

/// The whole `unit`s from `start` to `end` (negative when `end` is before
/// `start`), truncated toward zero, on the local date-times of the two
/// instants in `zone` (`None` for `TIMESTAMP_NTZ`). A unit that is a span
/// of time counts it on the local clock, so across a DST change a `DAY` is
/// 24 hours of wall clock, as `ChronoUnit.between` on `LocalDateTime`s is.
/// NULL in, NULL out.
///
/// # Errors
/// `DATETIME_OVERFLOW` when the count does not fit a `BIGINT` (a
/// `MICROSECOND` count over a span of more than 292,000 years);
/// `INTERNAL_ERROR` for arrays of different lengths.
pub fn timestamp_diff(
    unit: TimestampUnit,
    start: &TimestampMicrosecondArray,
    end: &TimestampMicrosecondArray,
    zone: Option<&dyn TimeZoneRules>,
) -> Result<Int64Array, KernelError> {
    if start.len() != end.len() {
        return Err(KernelError::internal(
            "timestampdiff arguments of different lengths",
        ));
    }
    let mut out = with_capacity::<i64>(start.len())?;
    let mut valid = with_capacity::<bool>(start.len())?;
    for row in 0..start.len() {
        if start.is_null(row) || end.is_null(row) {
            out.push(0);
            valid.push(false);
            continue;
        }
        let local = |t: i64| match zone {
            Some(tz) => utc_to_local(t, tz),
            None => Some(t),
        };
        let counted = local(start.value(row))
            .zip(local(end.value(row)))
            .and_then(|(a, b)| count(unit, a, b));
        let Some(n) = counted else {
            return Err(KernelError::datetime_overflow(row, "timestampdiff"));
        };
        out.push(n);
        valid.push(true);
    }
    let nulls = arrow_buffer::NullBuffer::from(valid);
    Ok(Int64Array::new(out.into(), Some(nulls)))
}

fn count(unit: TimestampUnit, a: i64, b: i64) -> Option<i64> {
    let span = i128::from(b).checked_sub(i128::from(a))?;
    if let Some(per) = unit.fixed_micros() {
        return i64::try_from(span.checked_div(i128::from(per))?).ok();
    }
    if let Some(per) = unit.days() {
        let micros = per.checked_mul(MICROS_PER_DAY)?;
        return i64::try_from(span.checked_div(i128::from(micros))?).ok();
    }
    // Months: `LocalDate.until` on the end date, moved a day toward the
    // start when the end time of day has not reached the start's.
    let (d1, t1) = (a.div_euclid(MICROS_PER_DAY), a.rem_euclid(MICROS_PER_DAY));
    let (d2, t2) = (b.div_euclid(MICROS_PER_DAY), b.rem_euclid(MICROS_PER_DAY));
    let d2 = if d2 > d1 && t2 < t1 {
        d2.checked_sub(1)?
    } else if d2 < d1 && t2 > t1 {
        d2.checked_add(1)?
    } else {
        d2
    };
    let packed = |days: i64| {
        let (year, month, day) = civil_from_days(days);
        year.checked_mul(12)?
            .checked_add(i64::from(month))?
            .checked_sub(1)?
            .checked_mul(32)?
            .checked_add(i64::from(day))
    };
    let months = packed(d2)?.checked_sub(packed(d1)?)?.checked_div(32)?;
    months.checked_div(unit.months()?)
}
