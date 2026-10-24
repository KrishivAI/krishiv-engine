// SPDX-License-Identifier: Apache-2.0
//! Calendar functions: `monthname`/`dayname`, `unix_date`, `last_day`,
//! `next_day`, `trunc`/`date_trunc`, `months_between`, `make_date` and the
//! zone conversions `from_utc_timestamp`, `to_utc_timestamp` and
//! `convert_timezone`.
//!
//! Dates and timestamps are proleptic Gregorian, as in Spark 3+, so a date
//! before 1582-10-15 reads like any other: there is no Julian cutover. Each
//! function states where Spark's own `int` arithmetic wraps; those results
//! are reproduced, not guarded.

use arrow_array::types::{Date32Type, Float64Type, TimestampMicrosecondType};
use arrow_array::{
    Array, BooleanArray, Date32Array, Float64Array, Int32Array, PrimitiveArray, StringArray,
    TimestampMicrosecondArray,
};
use arrow_buffer::NullBuffer;

use super::{
    FixedOffset, MICROS_PER_DAY, TimeZoneRules, as_days, civil_from_days, days_from_civil,
    iso_weekday, local_to_utc, local_to_utc_keeping, month_length, utc_to_local,
};
use crate::dispatch::KernelCtx;
use crate::error::KernelError;
use crate::error::with_capacity;
use crate::version::IntWidth;

const MONTH_NAMES: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];
const DAY_NAMES: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

const SECONDS_PER_DAY: i64 = 86_400;
const MICROS_PER_MILLI: i64 = 1_000;
const MICROS_PER_SECOND: i64 = 1_000_000;
const MICROS_PER_MINUTE: i64 = 60 * MICROS_PER_SECOND;
const MICROS_PER_HOUR: i64 = 60 * MICROS_PER_MINUTE;

/// `monthname(date)`: the three-letter month, `Jan` to `Dec`.
#[must_use]
pub fn month_name(a: &Date32Array) -> StringArray {
    a.iter()
        .map(|d| {
            let (_, month, _) = civil_from_days(i64::from(d?));
            let index = usize::try_from(month).ok()?.checked_sub(1)?;
            MONTH_NAMES.get(index).copied()
        })
        .collect()
}

/// `dayname(date)`: the three-letter weekday, `Mon` to `Sun`.
#[must_use]
pub fn day_name(a: &Date32Array) -> StringArray {
    a.iter()
        .map(|d| {
            let index = usize::try_from(iso_weekday(i64::from(d?)).checked_sub(1)?).ok()?;
            DAY_NAMES.get(index).copied()
        })
        .collect()
}

/// `unix_date(date)`: the day number, which is the date's own value.
#[must_use]
pub fn unix_date(a: &Date32Array) -> Int32Array {
    as_days(a)
}

/// The last day of the month of day number `days`. Spark adds in `int`: at
/// the top of the date range the sum wraps.
fn last_day_of_month(days: i32) -> i32 {
    let (year, month, day) = civil_from_days(i64::from(days));
    let length = month_length(year, month);
    days.wrapping_sub(i32::try_from(day).unwrap_or_default())
        .wrapping_add(i32::try_from(length).unwrap_or_default())
}

/// `last_day(date)`.
#[must_use]
pub fn last_day(a: &Date32Array) -> Date32Array {
    a.unary::<_, Date32Type>(last_day_of_month)
}

/// Spark's day number of a weekday name for `next_day`: day numbers count
/// from the epoch, a Thursday, so Thursday is 0 and Sunday 3. `None` for a
/// name Spark does not take (`SU`, `SUN`, `SUNDAY`, and so on, any case).
fn day_of_week_from_name(name: &str) -> Option<i32> {
    Some(match name.to_uppercase().as_str() {
        "SU" | "SUN" | "SUNDAY" => 3,
        "MO" | "MON" | "MONDAY" => 4,
        "TU" | "TUE" | "TUESDAY" => 5,
        "WE" | "WED" | "WEDNESDAY" => 6,
        "TH" | "THU" | "THURSDAY" => 0,
        "FR" | "FRI" | "FRIDAY" => 1,
        "SA" | "SAT" | "SATURDAY" => 2,
        _ => return None,
    })
}

/// `next_day(start, name)`: the first date after `start` that is the named
/// weekday. A name Spark does not take is `ILLEGAL_DAY_OF_WEEK` with ANSI on
/// and NULL with it off. Spark adds in `int`: the last days of the range
/// wrap.
///
/// # Errors
/// `ILLEGAL_DAY_OF_WEEK` as above; `INTERNAL_ERROR` for inputs of different
/// lengths; `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be allocated.
pub fn next_day(
    start: &Date32Array,
    names: &StringArray,
    ansi: bool,
) -> Result<Date32Array, KernelError> {
    check_lengths(&[start.len(), names.len()])?;
    let mut values = with_capacity::<i32>(start.len())?;
    let mut valid = with_capacity::<bool>(start.len())?;
    for i in 0..start.len() {
        if start.is_null(i) || names.is_null(i) {
            values.push(0);
            valid.push(false);
            continue;
        }
        let name = names.value(i);
        match day_of_week_from_name(name) {
            Some(dow) => {
                let from = start.value(i);
                let ahead = dow.wrapping_sub(1).wrapping_sub(from).rem_euclid(7);
                values.push(from.wrapping_add(1).wrapping_add(ahead));
                valid.push(true);
            }
            None if ansi => return Err(KernelError::illegal_day_of_week(i, name)),
            None => {
                values.push(0);
                valid.push(false);
            }
        }
    }
    Ok(PrimitiveArray::new(
        values.into(),
        Some(NullBuffer::from(valid)),
    ))
}

/// A level of `trunc` and `date_trunc`, in Spark's order (finest first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TruncUnit {
    /// `MICROSECOND`.
    Microsecond,
    /// `MILLISECOND`.
    Millisecond,
    /// `SECOND`.
    Second,
    /// `MINUTE`.
    Minute,
    /// `HOUR`.
    Hour,
    /// `DAY`, `DD`.
    Day,
    /// `WEEK`: the Monday on or before.
    Week,
    /// `MONTH`, `MON`, `MM`.
    Month,
    /// `QUARTER`.
    Quarter,
    /// `YEAR`, `YYYY`, `YY`.
    Year,
}

impl TruncUnit {
    /// The unit `name` denotes (case-insensitive); `None` for any other
    /// text, which `trunc` and `date_trunc` turn into NULL.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name.to_uppercase().as_str() {
            "YEAR" | "YYYY" | "YY" => Self::Year,
            "QUARTER" => Self::Quarter,
            "MONTH" | "MON" | "MM" => Self::Month,
            "WEEK" => Self::Week,
            "DAY" | "DD" => Self::Day,
            "HOUR" => Self::Hour,
            "MINUTE" => Self::Minute,
            "SECOND" => Self::Second,
            "MILLISECOND" => Self::Millisecond,
            "MICROSECOND" => Self::Microsecond,
            _ => return None,
        })
    }
}

/// The first day of `unit` containing day number `days`; `None` for a unit
/// finer than a week (a date has no such level).
fn truncate_days(days: i64, unit: TruncUnit) -> Option<i64> {
    let (year, month, day) = civil_from_days(days);
    match unit {
        TruncUnit::Week => Some(days.checked_sub(iso_weekday(days).checked_sub(1)?)?),
        TruncUnit::Month => days.checked_sub(i64::from(day))?.checked_add(1),
        TruncUnit::Quarter => days_from_civil(
            year,
            month
                .checked_sub(1)?
                .checked_div(3)?
                .checked_mul(3)?
                .checked_add(1)?,
            1,
        ),
        TruncUnit::Year => days_from_civil(year, 1, 1),
        _ => None,
    }
}

/// Parses the formats of a column, once per run of equal values: `None` for
/// a NULL format, `Some(None)` for one that names no unit.
fn units<'a>(formats: &'a StringArray) -> impl Iterator<Item = Option<Option<TruncUnit>>> + 'a {
    let mut last: Option<(&'a str, Option<TruncUnit>)> = None;
    (0..formats.len()).map(move |i| {
        if formats.is_null(i) {
            return None;
        }
        let text = formats.value(i);
        match last {
            Some((seen, unit)) if seen == text => Some(unit),
            _ => {
                let unit = TruncUnit::parse(text);
                last = Some((text, unit));
                Some(unit)
            }
        }
    })
}

/// `trunc(date, format)`: the first day of the week, month, quarter or year
/// of `date`. NULL for a NULL argument, and for a format that is not one of
/// those four (`DAY` and finer levels are not dates' levels).
///
/// # Errors
/// `DATETIME_OVERFLOW` where the first day leaves the `DATE` range (the
/// first year of the smallest dates); `INTERNAL_ERROR` for inputs of
/// different lengths; `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be
/// allocated.
pub fn trunc_date(dates: &Date32Array, formats: &StringArray) -> Result<Date32Array, KernelError> {
    check_lengths(&[dates.len(), formats.len()])?;
    let mut values = with_capacity::<i32>(dates.len())?;
    let mut valid = with_capacity::<bool>(dates.len())?;
    for (i, unit) in units(formats).enumerate() {
        let day = match (dates.is_valid(i), unit) {
            (true, Some(Some(unit))) if unit >= TruncUnit::Week => {
                let first = truncate_days(i64::from(dates.value(i)), unit)
                    .and_then(|d| i32::try_from(d).ok())
                    .ok_or_else(|| KernelError::datetime_overflow(i, "trunc"))?;
                Some(first)
            }
            _ => None,
        };
        values.push(day.unwrap_or_default());
        valid.push(day.is_some());
    }
    Ok(PrimitiveArray::new(
        values.into(),
        Some(NullBuffer::from(valid)),
    ))
}

/// The instant `unit` truncates `t` to in `zone`. Levels of a minute and
/// above truncate the local date-time: an hour or a day keeps the offset of
/// `t` where it is valid (the same side of a DST overlap, as
/// `ZonedDateTime.truncatedTo`); the date levels start the local day.
fn truncate_instant(t: i64, unit: TruncUnit, zone: &dyn TimeZoneRules) -> Option<i64> {
    let floor = |value: i64, step: i64| value.checked_sub(value.rem_euclid(step));
    match unit {
        TruncUnit::Microsecond => Some(t),
        TruncUnit::Millisecond => floor(t, MICROS_PER_MILLI),
        TruncUnit::Second => floor(t, MICROS_PER_SECOND),
        TruncUnit::Minute | TruncUnit::Hour | TruncUnit::Day => {
            let step = match unit {
                TruncUnit::Minute => MICROS_PER_MINUTE,
                TruncUnit::Hour => MICROS_PER_HOUR,
                _ => MICROS_PER_DAY,
            };
            let truncated = floor(utc_to_local(t, zone)?, step)?;
            let preferred = zone.offset_at_utc(t.div_euclid(MICROS_PER_SECOND));
            local_to_utc_keeping(truncated, zone, preferred)
        }
        TruncUnit::Week | TruncUnit::Month | TruncUnit::Quarter | TruncUnit::Year => {
            let local = utc_to_local(t, zone)?;
            let days = truncate_days(local.div_euclid(MICROS_PER_DAY), unit)?;
            local_to_utc(days.checked_mul(MICROS_PER_DAY)?, zone)
        }
    }
}

/// `date_trunc(format, timestamp)`: `timestamp` cut down to the start of the
/// unit `format` names, in `zone`. NULL for a NULL argument or an unknown
/// format.
///
/// # Errors
/// `DATETIME_OVERFLOW` where the truncated instant leaves the timestamp
/// range; `INTERNAL_ERROR` for inputs of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be allocated.
pub fn trunc_timestamp(
    formats: &StringArray,
    timestamps: &TimestampMicrosecondArray,
    zone: &dyn TimeZoneRules,
) -> Result<TimestampMicrosecondArray, KernelError> {
    check_lengths(&[timestamps.len(), formats.len()])?;
    let mut values = with_capacity::<i64>(timestamps.len())?;
    let mut valid = with_capacity::<bool>(timestamps.len())?;
    for (i, unit) in units(formats).enumerate() {
        let (Some(Some(unit)), true) = (unit, timestamps.is_valid(i)) else {
            values.push(0);
            valid.push(false);
            continue;
        };
        let t = truncate_instant(timestamps.value(i), unit, zone)
            .ok_or_else(|| KernelError::datetime_overflow(i, "date_trunc"))?;
        values.push(t);
        valid.push(true);
    }
    Ok(PrimitiveArray::<TimestampMicrosecondType>::new(
        values.into(),
        Some(NullBuffer::from(valid)),
    ))
}

/// Java's `Math.round`: halves go up, also below zero.
fn java_round(x: f64) -> f64 {
    let floor = x.floor();
    if x - floor >= 0.5 { floor + 1.0 } else { floor }
}

/// `months_between` of two instants in `zone`: Spark's formula. Equal days
/// of the month, or two last days of a month, are a whole number of months;
/// otherwise the difference in days and whole seconds of the day (the
/// microseconds are dropped) counts 31-day months.
fn months_between_instants(
    t1: i64,
    t2: i64,
    round_off: bool,
    zone: &dyn TimeZoneRules,
) -> Option<f64> {
    let days1 = utc_to_local(t1, zone)?.div_euclid(MICROS_PER_DAY);
    let days2 = utc_to_local(t2, zone)?.div_euclid(MICROS_PER_DAY);
    let ((y1, m1, d1), (y2, m2, d2)) = (civil_from_days(days1), civil_from_days(days2));
    let months = y1
        .checked_mul(12)?
        .checked_add(i64::from(m1))?
        .checked_sub(y2.checked_mul(12)?.checked_add(i64::from(m2))?)?;
    let both_last_days = d1 == month_length(y1, m1) && d2 == month_length(y2, m2);
    if d1 == d2 || both_last_days {
        return Some(months as f64);
    }
    // The time into the day counts from the instant the day starts.
    let time1 = t1.checked_sub(local_to_utc(days1.checked_mul(MICROS_PER_DAY)?, zone)?)?;
    let time2 = t2.checked_sub(local_to_utc(days2.checked_mul(MICROS_PER_DAY)?, zone)?)?;
    let seconds = i64::from(d1)
        .checked_sub(i64::from(d2))?
        .checked_mul(SECONDS_PER_DAY)?
        .checked_add(time1 / MICROS_PER_SECOND)?
        .checked_sub(time2 / MICROS_PER_SECOND)?;
    let diff = months as f64 + seconds as f64 / (31 * SECONDS_PER_DAY) as f64;
    Some(if round_off {
        java_round(diff * 1e8) / 1e8
    } else {
        diff
    })
}

/// `months_between(end, start, roundOff)`: the months from `start` to `end`
/// as a `double`, rounded to 8 digits when `round_off`. NULL if any
/// argument is NULL.
///
/// # Errors
/// `DATETIME_OVERFLOW` for an instant whose local time is out of range;
/// `INTERNAL_ERROR` for inputs of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be allocated.
pub fn months_between(
    end: &TimestampMicrosecondArray,
    start: &TimestampMicrosecondArray,
    round_off: &BooleanArray,
    zone: &dyn TimeZoneRules,
) -> Result<Float64Array, KernelError> {
    check_lengths(&[end.len(), start.len(), round_off.len()])?;
    let mut values = with_capacity::<f64>(end.len())?;
    let mut valid = with_capacity::<bool>(end.len())?;
    for i in 0..end.len() {
        if end.is_null(i) || start.is_null(i) || round_off.is_null(i) {
            values.push(0.0);
            valid.push(false);
            continue;
        }
        let months =
            months_between_instants(end.value(i), start.value(i), round_off.value(i), zone)
                .ok_or_else(|| KernelError::datetime_overflow(i, "months_between"))?;
        values.push(months);
        valid.push(true);
    }
    Ok(PrimitiveArray::<Float64Type>::new(
        values.into(),
        Some(NullBuffer::from(valid)),
    ))
}

/// The message of the `DateTimeException` `LocalDate.of(year, month, day)`
/// throws, or `None` for a valid date.
fn invalid_date(year: i32, month: i32, day: i32) -> Option<String> {
    if !(-999_999_999..=999_999_999).contains(&year) {
        return Some(format!(
            "Invalid value for Year (valid values -999999999 - 999999999): {year}"
        ));
    }
    if !(1..=12).contains(&month) {
        return Some(format!(
            "Invalid value for MonthOfYear (valid values 1 - 12): {month}"
        ));
    }
    if !(1..=31).contains(&day) {
        return Some(format!(
            "Invalid value for DayOfMonth (valid values 1 - 28/31): {day}"
        ));
    }
    let month_u = u32::try_from(month).ok()?;
    let day_u = u32::try_from(day).ok()?;
    if day_u <= month_length(i64::from(year), month_u) {
        return None;
    }
    Some(if day == 29 {
        format!("Invalid date 'February 29' as '{year}' is not a leap year")
    } else {
        let name = MONTH_LONG_NAMES.get(usize::try_from(month.checked_sub(1)?).ok()?)?;
        format!("Invalid date '{name} {day}'")
    })
}

const MONTH_LONG_NAMES: [&str; 12] = [
    "JANUARY",
    "FEBRUARY",
    "MARCH",
    "APRIL",
    "MAY",
    "JUNE",
    "JULY",
    "AUGUST",
    "SEPTEMBER",
    "OCTOBER",
    "NOVEMBER",
    "DECEMBER",
];

/// `make_date(year, month, day)`. A field out of range, or a day the month
/// does not have, is `DATETIME_FIELD_OUT_OF_BOUNDS.WITH_SUGGESTION` with
/// `fail_on_error` (ANSI) and NULL without. A year whose dates leave the
/// `DATE` range (about 5.88 million) is `ARITHMETIC_OVERFLOW` in both modes:
/// Spark raises an `ArithmeticException` there that its handler of
/// `DateTimeException` does not catch (no recorded run reaches it).
///
/// # Errors
/// As above; `INTERNAL_ERROR` for inputs of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be allocated.
pub fn make_date(
    years: &Int32Array,
    months: &Int32Array,
    days: &Int32Array,
    fail_on_error: bool,
    ctx: &KernelCtx,
) -> Result<Date32Array, KernelError> {
    check_lengths(&[years.len(), months.len(), days.len()])?;
    let mut values = with_capacity::<i32>(years.len())?;
    let mut valid = with_capacity::<bool>(years.len())?;
    for i in 0..years.len() {
        if years.is_null(i) || months.is_null(i) || days.is_null(i) {
            values.push(0);
            valid.push(false);
            continue;
        }
        let (y, m, d) = (years.value(i), months.value(i), days.value(i));
        if let Some(message) = invalid_date(y, m, d) {
            if fail_on_error {
                return Err(KernelError::datetime_field_out_of_bounds(i, message));
            }
            values.push(0);
            valid.push(false);
            continue;
        }
        let day = u32::try_from(m)
            .ok()
            .zip(u32::try_from(d).ok())
            .and_then(|(m, d)| days_from_civil(i64::from(y), m, d))
            .and_then(|days| i32::try_from(days).ok())
            .ok_or_else(|| {
                let text = ctx.spark_version().overflow_message(IntWidth::Int);
                KernelError::arithmetic_overflow(i, text, None)
            })?;
        values.push(day);
        valid.push(true);
    }
    Ok(PrimitiveArray::<Date32Type>::new(
        values.into(),
        Some(NullBuffer::from(valid)),
    ))
}

/// The zone of each row of a call: one zone for every row, or one per row
/// (`None` is a NULL zone, which gives a NULL result).
#[derive(Debug, Clone, Copy)]
pub enum Zones<'a> {
    /// Every row reads this zone.
    One(&'a dyn TimeZoneRules),
    /// Row `i` reads the zone at `i`.
    PerRow(&'a [Option<FixedOffset>]),
}

impl Zones<'_> {
    /// `INTERNAL_ERROR` unless the zones cover `rows` rows.
    fn check(&self, rows: usize) -> Result<(), KernelError> {
        match self {
            Self::PerRow(zones) if zones.len() != rows => Err(KernelError::internal(format!(
                "{} zones for {rows} rows",
                zones.len()
            ))),
            _ => Ok(()),
        }
    }

    fn at(&self, row: usize) -> Option<&dyn TimeZoneRules> {
        match self {
            Self::One(zone) => Some(*zone),
            Self::PerRow(zones) => zones
                .get(row)
                .and_then(Option::as_ref)
                .map(|zone| zone as &dyn TimeZoneRules),
        }
    }
}

/// Applies `f` to the valid rows of `timestamps`; a row whose zone is NULL
/// is NULL. `f` gets the value and the row's zone, `None` for an overflow.
fn zoned(
    timestamps: &TimestampMicrosecondArray,
    zones: &Zones<'_>,
    operation: &str,
    f: impl Fn(i64, &dyn TimeZoneRules) -> Option<i64>,
) -> Result<TimestampMicrosecondArray, KernelError> {
    zones.check(timestamps.len())?;
    let mut values = with_capacity::<i64>(timestamps.len())?;
    let mut valid = with_capacity::<bool>(timestamps.len())?;
    for i in 0..timestamps.len() {
        let (true, Some(tz)) = (timestamps.is_valid(i), zones.at(i)) else {
            values.push(0);
            valid.push(false);
            continue;
        };
        let t = f(timestamps.value(i), tz)
            .ok_or_else(|| KernelError::datetime_overflow(i, operation))?;
        values.push(t);
        valid.push(true);
    }
    Ok(PrimitiveArray::<TimestampMicrosecondType>::new(
        values.into(),
        Some(NullBuffer::from(valid)),
    ))
}

/// `from_utc_timestamp(timestamp, zone)`: the instant whose UTC wall clock
/// is the wall clock `timestamp` has in the zone of its row (NULL zone, NULL
/// result).
///
/// # Errors
/// `DATETIME_OVERFLOW` where the result leaves the timestamp range;
/// `INTERNAL_ERROR` for a zone column of another length;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be allocated.
pub fn from_utc_timestamp(
    timestamps: &TimestampMicrosecondArray,
    zones: &Zones<'_>,
) -> Result<TimestampMicrosecondArray, KernelError> {
    zoned(timestamps, zones, "from_utc_timestamp", utc_to_local)
}

/// `to_utc_timestamp(timestamp, zone)`: the instant whose wall clock in the
/// zone of its row is the UTC wall clock of `timestamp` (a skipped wall time
/// moves forward, an ambiguous one takes the earlier offset).
///
/// # Errors
/// As [`from_utc_timestamp`].
pub fn to_utc_timestamp(
    timestamps: &TimestampMicrosecondArray,
    zones: &Zones<'_>,
) -> Result<TimestampMicrosecondArray, KernelError> {
    zoned(timestamps, zones, "to_utc_timestamp", local_to_utc)
}

/// `convert_timezone(source, target, ts)` over a `TIMESTAMP_NTZ`: the wall
/// clock in the target zone of the instant `ts` denotes in the source zone.
/// A NULL zone gives NULL.
///
/// # Errors
/// `DATETIME_OVERFLOW` where the result leaves the timestamp range;
/// `INTERNAL_ERROR` for a zone column of another length;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be allocated.
pub fn convert_timezone(
    sources: &Zones<'_>,
    targets: &Zones<'_>,
    wall_clocks: &TimestampMicrosecondArray,
) -> Result<TimestampMicrosecondArray, KernelError> {
    sources.check(wall_clocks.len())?;
    targets.check(wall_clocks.len())?;
    let mut values = with_capacity::<i64>(wall_clocks.len())?;
    let mut valid = with_capacity::<bool>(wall_clocks.len())?;
    for i in 0..wall_clocks.len() {
        let (true, Some(source), Some(target)) =
            (wall_clocks.is_valid(i), sources.at(i), targets.at(i))
        else {
            values.push(0);
            valid.push(false);
            continue;
        };
        let t = local_to_utc(wall_clocks.value(i), source)
            .and_then(|instant| utc_to_local(instant, target))
            .ok_or_else(|| KernelError::datetime_overflow(i, "convert_timezone"))?;
        values.push(t);
        valid.push(true);
    }
    Ok(PrimitiveArray::<TimestampMicrosecondType>::new(
        values.into(),
        Some(NullBuffer::from(valid)),
    ))
}

/// Rejects inputs of different lengths.
fn check_lengths(lengths: &[usize]) -> Result<(), KernelError> {
    match lengths {
        [first, rest @ ..] if rest.iter().any(|n| n != first) => Err(KernelError::internal(
            format!("operands differ in length: {lengths:?}"),
        )),
        _ => Ok(()),
    }
}
