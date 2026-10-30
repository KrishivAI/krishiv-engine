// SPDX-License-Identifier: Apache-2.0
//! `make_timestamp(year, month, day, hour, minute, seconds[, zone])` and its
//! `_ltz`/`_ntz` and `try_` forms, from the six numeric fields.
//!
//! Fields are checked in `java.time`'s order (`LocalDate.of`, then the hour,
//! minute and second of `LocalTime.of`) and raise its messages. `seconds` is
//! a `decimal(16,6)`: 60 with no fraction rolls over to the next minute, 60
//! with a fraction is `INVALID_FRACTION_OF_SECOND`, any other value outside
//! 0..59 is out of bounds. A local time inside a DST gap moves forward, an
//! ambiguous one takes the earlier offset (`ZonedDateTime.of`).

use arrow_array::{Array, Decimal128Array, Int32Array, TimestampMicrosecondArray};
use arrow_buffer::NullBuffer;

use super::Zones;
use super::calendar::invalid_date;
use crate::cast::datetime::{MICROS_PER_DAY, days_from_civil, local_to_utc};
use crate::error::{KernelError, with_capacity};

const MICROS_PER_SECOND: i128 = 1_000_000;

/// The fields of each row, as the `make_timestamp` columns.
#[derive(Debug, Clone, Copy)]
pub struct TimestampFields<'a> {
    /// The year.
    pub years: &'a Int32Array,
    /// The month, 1 to 12.
    pub months: &'a Int32Array,
    /// The day of the month.
    pub days: &'a Int32Array,
    /// The hour, 0 to 23.
    pub hours: &'a Int32Array,
    /// The minute, 0 to 59.
    pub minutes: &'a Int32Array,
    /// Seconds with microseconds: a `decimal(16,6)`.
    pub seconds: &'a Decimal128Array,
}

/// `make_timestamp`. `zones` is the zone the wall clock is read in
/// (`TIMESTAMP`; a NULL zone gives NULL), or `None` for a `TIMESTAMP_NTZ`,
/// which is the wall clock itself. An invalid field is
/// `DATETIME_FIELD_OUT_OF_BOUNDS.WITH_SUGGESTION` (`INVALID_FRACTION_OF_SECOND`
/// for 60 with a fraction) when `fail_on_error`, else NULL.
///
/// # Errors
/// As above; `INTERNAL_ERROR` for inputs of different lengths or a seconds
/// column that is not scale 6.
pub fn make_timestamp(
    fields: TimestampFields<'_>,
    zones: Option<&Zones<'_>>,
    fail_on_error: bool,
) -> Result<TimestampMicrosecondArray, KernelError> {
    let rows = fields.years.len();
    let lengths = [
        fields.months.len(),
        fields.days.len(),
        fields.hours.len(),
        fields.minutes.len(),
        fields.seconds.len(),
    ];
    if lengths.iter().any(|&n| n != rows) {
        return Err(KernelError::internal(
            "make_timestamp arguments of different lengths",
        ));
    }
    if fields.seconds.scale() != 6 {
        return Err(KernelError::internal(
            "make_timestamp seconds are not a decimal of scale 6",
        ));
    }
    let mut values = with_capacity::<i64>(rows)?;
    let mut valid = with_capacity::<bool>(rows)?;
    for row in 0..rows {
        let present = ![
            fields.years.is_null(row),
            fields.months.is_null(row),
            fields.days.is_null(row),
            fields.hours.is_null(row),
            fields.minutes.is_null(row),
            fields.seconds.is_null(row),
        ]
        .contains(&true);
        let zone = match zones {
            Some(z) => z.at(row),
            None => None,
        };
        // A NULL field or zone is NULL without reading the other fields.
        if !present || (zones.is_some() && zone.is_none()) {
            values.push(0);
            valid.push(false);
            continue;
        }
        match one(&fields, row) {
            Ok(local) => {
                let instant = match zone {
                    Some(tz) => local_to_utc(local, tz),
                    None => Some(local),
                };
                if let Some(t) = instant {
                    values.push(t);
                    valid.push(true);
                } else if fail_on_error {
                    return Err(KernelError::datetime_overflow(row, "make_timestamp"));
                } else {
                    values.push(0);
                    valid.push(false);
                }
            }
            Err(e) if fail_on_error => return Err(e),
            Err(_) => {
                values.push(0);
                valid.push(false);
            }
        }
    }
    Ok(TimestampMicrosecondArray::new(
        values.into(),
        Some(NullBuffer::from(valid)),
    ))
}

/// The wall-clock microseconds of row `row`, or the error Spark raises.
fn one(f: &TimestampFields<'_>, row: usize) -> Result<i64, KernelError> {
    let (year, month, day) = (f.years.value(row), f.months.value(row), f.days.value(row));
    let out_of_bounds = |text: String| KernelError::datetime_field_out_of_bounds(row, text);
    if let Some(message) = invalid_date(year, month, day) {
        return Err(out_of_bounds(message));
    }
    let (hour, minute) = (f.hours.value(row), f.minutes.value(row));
    if !(0..=23).contains(&hour) {
        return Err(out_of_bounds(format!(
            "Invalid value for HourOfDay (valid values 0 - 23): {hour}"
        )));
    }
    if !(0..=59).contains(&minute) {
        return Err(out_of_bounds(format!(
            "Invalid value for MinuteOfHour (valid values 0 - 59): {minute}"
        )));
    }
    let unscaled = f.seconds.value(row);
    let whole = unscaled.div_euclid(MICROS_PER_SECOND);
    let micros = unscaled.rem_euclid(MICROS_PER_SECOND);
    let rolled = whole == 60 && micros == 0;
    if whole == 60 && micros != 0 {
        return Err(KernelError::invalid_fraction_of_second(
            row,
            seconds_text(unscaled),
        ));
    }
    if !rolled && !(0..=59).contains(&whole) {
        return Err(out_of_bounds(format!(
            "Invalid value for SecondOfMinute (valid values 0 - 59): {whole}"
        )));
    }
    let overflow = || KernelError::datetime_overflow(row, "make_timestamp");
    let date = u32::try_from(month)
        .ok()
        .zip(u32::try_from(day).ok())
        .and_then(|(m, d)| days_from_civil(i64::from(year), m, d))
        .ok_or_else(overflow)?;
    // The seconds are in 0..=60 and the micros below a second here.
    let of_day = i64::from(hour)
        .saturating_mul(60)
        .saturating_add(i64::from(minute))
        .saturating_mul(60);
    let of_day = of_day
        .checked_add(i64::try_from(whole).map_err(|_| overflow())?)
        .and_then(|s| s.checked_mul(1_000_000))
        .and_then(|us| us.checked_add(i64::try_from(micros).ok()?))
        .ok_or_else(overflow)?;
    date.checked_mul(MICROS_PER_DAY)
        .and_then(|d| d.checked_add(of_day))
        .ok_or_else(overflow)
}

/// `secAndMicros` as Spark prints it: the decimal without trailing zeros.
fn seconds_text(unscaled: i128) -> String {
    let sign = if unscaled < 0 { "-" } else { "" };
    let abs = unscaled.unsigned_abs();
    let (whole, frac) = (abs / 1_000_000, abs % 1_000_000);
    if frac == 0 {
        return format!("{sign}{whole}");
    }
    let digits = format!("{frac:06}");
    format!("{sign}{whole}.{}", digits.trim_end_matches('0'))
}
