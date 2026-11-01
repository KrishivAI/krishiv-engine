// SPDX-License-Identifier: Apache-2.0
//! `TIME` text: Spark's `string -> TIME` grammar and `TIME -> string`.
//!
//! A `TIME` is carried as `Time64(Nanosecond)`, nanoseconds since midnight
//! (design P2-T181 section 2). Behaviour here is from Spark 4.2.0 probes:
//! surrounding whitespace and a leading `T` are accepted; hour and minute
//! are 1 or 2 digits, seconds are optional, a fraction of any length is
//! truncated to microseconds (`TIME(6)` is the widest time) (`12:34:56.` is valid); an `AM`/`PM` suffix is
//! read (`01:02:03 PM` is 13:02:03); `24:00` and `60` minutes or seconds
//! are malformed.

use arrow_array::types::Time64NanosecondType;
use arrow_array::{Array, PrimitiveArray, StringArray};
use krishiv_common::error::InvalidParameterValueTimeUnit;

use crate::error::KernelError;

/// Nanoseconds in a second.
pub(crate) const NANOS_PER_SECOND: i64 = 1_000_000_000;

fn trim_ascii_end(s: &str) -> &str {
    s.trim_end_matches(|c: char| c.is_ascii_whitespace())
}

/// Nanoseconds since midnight for `text`, or `None` when malformed.
pub fn parse_time(text: &str) -> Option<i64> {
    let s = text.trim_matches(|c: char| c.is_ascii_whitespace());
    let s = s.strip_prefix('T').unwrap_or(s);
    // The suffix is separated by a run of ASCII whitespace (a tab will do).
    let (clock, meridiem) = match s.rsplit_once(|c: char| c.is_ascii_whitespace()) {
        Some((clock, m)) if m.eq_ignore_ascii_case("AM") => (trim_ascii_end(clock), Some(false)),
        Some((clock, m)) if m.eq_ignore_ascii_case("PM") => (trim_ascii_end(clock), Some(true)),
        _ => (s, None),
    };
    let field = |p: &str| -> Option<i64> {
        ((1..=2).contains(&p.len()) && p.bytes().all(|b| b.is_ascii_digit()))
            .then(|| p.parse().ok())
            .flatten()
    };
    let mut parts = clock.split(':');
    let hour = parts.next().and_then(field)?;
    let minute = parts.next().and_then(field)?;
    let (second, nanos) = match parts.next() {
        None => (0, 0),
        Some(sec) => {
            let (whole, fraction) = sec.split_once('.').unwrap_or((sec, ""));
            if !fraction.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            let mut nanos = 0_i64;
            for (i, d) in fraction.bytes().take(6).enumerate() {
                let scale = 10_i64.checked_pow(u32::try_from(8_usize.checked_sub(i)?).ok()?)?;
                nanos = nanos.checked_add(i64::from(d.wrapping_sub(b'0')).checked_mul(scale)?)?;
            }
            (field(whole)?, nanos)
        }
    };
    if parts.next().is_some() || minute > 59 || second > 59 {
        return None;
    }
    let hour = match meridiem {
        None if hour < 24 => hour,
        Some(pm) if (1..=12).contains(&hour) => {
            hour.rem_euclid(12).wrapping_add(if pm { 12 } else { 0 })
        }
        _ => return None,
    };
    // Each field is range-checked above, so none of this can wrap.
    let minutes = hour.wrapping_mul(60).wrapping_add(minute);
    let seconds = minutes.wrapping_mul(60).wrapping_add(second);
    Some(seconds.wrapping_mul(NANOS_PER_SECOND).wrapping_add(nanos))
}

/// `HH:mm:ss` with the fraction's trailing zeros dropped, as Spark shows a
/// `TIME`.
pub fn format_time(nanos: i64) -> String {
    let secs = nanos.div_euclid(NANOS_PER_SECOND);
    let frac = nanos.rem_euclid(NANOS_PER_SECOND);
    let mut out = format!("{:02}:{:02}:{:02}", secs / 3600, secs / 60 % 60, secs % 60);
    if frac != 0 {
        let digits = format!("{frac:09}");
        out.push('.');
        out.push_str(digits.trim_end_matches('0'));
    }
    out
}

/// A time unit of `time_trunc` and `time_diff`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeTruncUnit {
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

impl TimeTruncUnit {
    /// The unit a name denotes: Spark's five, in any case and no aliases
    /// (`MS` and a blank are `INVALID_PARAMETER_VALUE.TIME_UNIT`).
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name.to_ascii_uppercase().as_str() {
            "HOUR" => Self::Hour,
            "MINUTE" => Self::Minute,
            "SECOND" => Self::Second,
            "MILLISECOND" => Self::Millisecond,
            "MICROSECOND" => Self::Microsecond,
            _ => return None,
        })
    }

    /// Nanoseconds in one unit.
    fn nanos(self) -> i64 {
        match self {
            Self::Hour => 3_600_000_000_000,
            Self::Minute => 60_000_000_000,
            Self::Second => NANOS_PER_SECOND,
            Self::Millisecond => 1_000_000,
            Self::Microsecond => 1_000,
        }
    }
}

/// `INVALID_PARAMETER_VALUE.TIME_UNIT` for the unit `name` of `function`.
fn bad_unit(row: usize, function: &str, name: &str) -> KernelError {
    KernelError::with_row(
        Some(row),
        InvalidParameterValueTimeUnit {
            function_name: format!("`{function}`"),
            parameter: "`unit`".to_owned(),
            invalid_value: format!("'{name}'"),
        }
        .into(),
    )
}

/// `time_trunc(unit, time)`: `time` cut down to a whole `unit`. A NULL unit
/// or time is NULL; an unknown unit raises for a non-NULL row.
///
/// # Errors
/// `INVALID_PARAMETER_VALUE.TIME_UNIT`.
pub fn time_trunc(
    units: &StringArray,
    times: &PrimitiveArray<Time64NanosecondType>,
) -> Result<PrimitiveArray<Time64NanosecondType>, KernelError> {
    let mut out = Vec::with_capacity(times.len());
    for i in 0..times.len() {
        if units.is_null(i) || times.is_null(i) {
            out.push(None);
            continue;
        }
        let unit = TimeTruncUnit::parse(units.value(i))
            .ok_or_else(|| bad_unit(i, "time_trunc", units.value(i)))?;
        let n = times.value(i);
        out.push(Some(
            n.wrapping_sub(n.checked_rem(unit.nanos()).unwrap_or(0)),
        ));
    }
    Ok(out.into_iter().collect())
}

/// `time_diff(unit, start, end)`: whole `unit`s from `start` to `end`
/// (negative when `end` is earlier), counted toward zero.
///
/// # Errors
/// `INVALID_PARAMETER_VALUE.TIME_UNIT`.
pub fn time_diff(
    units: &StringArray,
    start: &PrimitiveArray<Time64NanosecondType>,
    end: &PrimitiveArray<Time64NanosecondType>,
) -> Result<PrimitiveArray<arrow_array::types::Int64Type>, KernelError> {
    let mut out = Vec::with_capacity(start.len());
    for i in 0..start.len() {
        if units.is_null(i) || start.is_null(i) || end.is_null(i) {
            out.push(None);
            continue;
        }
        let unit = TimeTruncUnit::parse(units.value(i))
            .ok_or_else(|| bad_unit(i, "time_diff", units.value(i)))?;
        out.push(Some(
            end.value(i)
                .wrapping_sub(start.value(i))
                .checked_div(unit.nanos())
                .unwrap_or(0),
        ));
    }
    Ok(out.into_iter().collect())
}

const NANOS_PER_HOUR: i128 = 3_600 * NANOS_PER_SECOND as i128;
const NANOS_PER_DAY: i128 = 24 * NANOS_PER_HOUR;

/// `time_from_seconds|millis|micros`: `values` scaled by `nanos_per_unit`
/// (the seconds form takes a `decimal(14,6)`, so its unit is a microsecond).
///
/// # Errors
/// `DATETIME_FIELD_OUT_OF_BOUNDS.WITHOUT_SUGGESTION` for a value outside one
/// day.
pub fn time_from_units(
    values: impl Iterator<Item = Option<i128>>,
    nanos_per_unit: i128,
) -> Result<PrimitiveArray<Time64NanosecondType>, KernelError> {
    let mut out = Vec::new();
    for (row, v) in values.enumerate() {
        let Some(v) = v else {
            out.push(None);
            continue;
        };
        let nanos = v.saturating_mul(nanos_per_unit);
        if !(0..NANOS_PER_DAY).contains(&nanos) {
            return Err(KernelError::time_field_out_of_bounds(
                row,
                format!(
                    "Invalid TIME value: must be between 00:00:00 and 23:59:59.999999999, but got {nanos} nanoseconds"
                ),
            ));
        }
        out.push(i64::try_from(nanos).ok());
    }
    Ok(out.into_iter().collect())
}

/// `make_time(hour, minute, seconds)`; `seconds` is a `decimal(16,6)` read
/// as microseconds. The fields are checked in `LocalTime.of`'s order.
///
/// # Errors
/// `DATETIME_FIELD_OUT_OF_BOUNDS.WITHOUT_SUGGESTION`.
pub fn make_time(
    hours: &arrow_array::Int32Array,
    minutes: &arrow_array::Int32Array,
    seconds: &arrow_array::Decimal128Array,
) -> Result<PrimitiveArray<Time64NanosecondType>, KernelError> {
    let mut out = Vec::with_capacity(hours.len());
    for row in 0..hours.len() {
        if hours.is_null(row) || minutes.is_null(row) || seconds.is_null(row) {
            out.push(None);
            continue;
        }
        let (h, m, micros) = (
            i64::from(hours.value(row)),
            i64::from(minutes.value(row)),
            seconds.value(row),
        );
        let bad = |field: &str, hi: u8, v: i128| {
            KernelError::time_field_out_of_bounds(
                row,
                format!("Invalid value for {field} (valid values 0 - {hi}): {v}"),
            )
        };
        if !(0..24).contains(&h) {
            return Err(bad("HourOfDay", 23, i128::from(h)));
        }
        if !(0..60).contains(&m) {
            return Err(bad("MinuteOfHour", 59, i128::from(m)));
        }
        let whole = micros.checked_div(1_000_000).unwrap_or(0);
        // A negative value is out of range even when its whole seconds are 0
        // (`-0.5`): Spark shows the truncated whole seconds.
        if micros < 0 || !(0..60).contains(&whole) {
            return Err(bad("SecondOfMinute", 59, whole));
        }
        let nanos = h
            .wrapping_mul(3_600)
            .wrapping_add(m.wrapping_mul(60))
            .wrapping_mul(NANOS_PER_SECOND)
            .wrapping_add(i64::try_from(micros).unwrap_or(0).wrapping_mul(1_000));
        out.push(Some(nanos));
    }
    Ok(out.into_iter().collect())
}

/// `time_to_seconds|millis|micros`: whole `nanos_per_unit`s of each time.
pub fn time_to_units(
    times: &PrimitiveArray<Time64NanosecondType>,
    nanos_per_unit: i64,
) -> PrimitiveArray<arrow_array::types::Int64Type> {
    times.unary(|n| n.checked_div(nanos_per_unit).unwrap_or(0))
}

/// The format Spark names in `CANNOT_PARSE_TIME` for a one-argument
/// `to_time` (its text is read with the default `TIME` grammar).
const DEFAULT_TIME_FORMAT: &str = "HH:mm:ss.SSSSSS";

/// `to_time(text[, format])`: with no `formats`, the default `TIME` grammar;
/// else each row's pattern (a NULL pattern gives NULL). An unparsable row
/// is `CANNOT_PARSE_TIME` with `fail_on_error`, else NULL.
///
/// # Errors
/// `CANNOT_PARSE_TIME`; an invalid pattern's own error.
pub fn to_time(
    strings: &StringArray,
    formats: Option<&StringArray>,
    fail_on_error: bool,
) -> Result<PrimitiveArray<Time64NanosecondType>, KernelError> {
    use crate::datetime_pattern::{DateTimePattern, Purpose};
    const MICROS_PER_DAY: i64 = 86_400_000_000;
    let mut out = Vec::with_capacity(strings.len());
    for i in 0..strings.len() {
        let fmt = formats.map(|f| (!f.is_null(i)).then(|| f.value(i)));
        if strings.is_null(i) || fmt == Some(None) {
            out.push(None);
            continue;
        }
        let text = strings.value(i);
        let parsed = match fmt.flatten() {
            None => parse_time(text),
            Some(f) => DateTimePattern::new(f, Purpose::Parse)
                .map_err(|e| KernelError::with_row(Some(i), e))?
                .parse_local(text, false)
                .map_err(|e| KernelError::with_row(Some(i), e))?
                .map(|m| m.rem_euclid(MICROS_PER_DAY).wrapping_mul(1_000)),
        };
        match parsed {
            Some(n) => out.push(Some(n)),
            None if fail_on_error => {
                let shown = fmt.flatten().unwrap_or(DEFAULT_TIME_FORMAT);
                return Err(KernelError::cannot_parse_time(
                    i,
                    crate::cast::sql_string(text),
                    crate::cast::sql_string(shown),
                ));
            }
            None => out.push(None),
        }
    }
    Ok(out.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: i64 = NANOS_PER_SECOND;

    #[test]
    fn reads_what_spark_reads() {
        for (text, want) in [
            ("12:34", Some((12 * 60 + 34) * 60 * S)),
            (" 12:34 ", Some((12 * 60 + 34) * 60 * S)),
            ("T12:34", Some((12 * 60 + 34) * 60 * S)),
            ("1:2:3", Some(3723 * S)),
            ("12:34:56.", Some(45296 * S)),
            ("12:34:56.1234567", Some(45296 * S + 123_456_000)),
            ("12:34:56.1234567890", Some(45296 * S + 123_456_000)),
            ("01:02:03 PM", Some((13 * 3600 + 2 * 60 + 3) * S)),
            ("12:00 AM", Some(0)),
            ("12:30\tPM", Some((12 * 60 + 30) * 60 * S)),
            ("12:30 \t am", Some(30 * 60 * S)),
            ("12:30\u{a0}AM", None),
            ("12:34 PM", Some((12 * 60 + 34) * 60 * S)),
            ("12", None),
            ("-12:34", None),
            ("24:00", None),
            ("00:60", None),
            ("00:00:60", None),
            ("13:00 PM", None),
            ("00:00 AM", None),
            ("bad", None),
        ] {
            assert_eq!(parse_time(text), want, "{text:?}");
        }
    }

    #[test]
    fn shows_like_spark() {
        assert_eq!(format_time(0), "00:00:00");
        assert_eq!(format_time(45296 * S + 500_000_000), "12:34:56.5");
        assert_eq!(format_time(45296 * S + 789_123_000), "12:34:56.789123");
    }

    fn times(v: &[Option<i64>]) -> PrimitiveArray<Time64NanosecondType> {
        v.iter().copied().collect()
    }

    #[test]
    fn trunc_and_diff_follow_spark() {
        let units = StringArray::from(vec![
            Some("hour"),
            Some("MICROSECOND"),
            None,
            Some("minute"),
        ]);
        let t = times(&[
            Some(45296 * S + 789_123_000),
            Some(45296 * S + 789_123_000),
            Some(0),
            None,
        ]);
        let out = time_trunc(&units, &t).unwrap();
        assert_eq!(out.value(0), 12 * 3600 * S);
        assert_eq!(out.value(1), 45296 * S + 789_123_000);
        assert!(out.is_null(2) && out.is_null(3));

        let start = times(&[Some(0), Some(45296 * S), Some(0)]);
        let end = times(&[Some(45296 * S + 100_000_000), Some(3723 * S), Some(0)]);
        let units = StringArray::from(vec![Some("MILLISECOND"), Some("HOUR"), None]);
        let out = time_diff(&units, &start, &end).unwrap();
        assert_eq!(out.value(0), 45_296_100);
        // Counted toward zero: -11 hours and a bit.
        assert_eq!(out.value(1), -11);
        assert!(out.is_null(2));
    }

    #[test]
    fn an_unknown_unit_raises_for_a_non_null_row_only() {
        let t = times(&[Some(1), None]);
        for bad in ["MS", " ", "INVALID"] {
            let units = StringArray::from(vec![Some(bad), Some(bad)]);
            let e = time_trunc(&units, &t).unwrap_err();
            assert_eq!(
                e.error().error_class(),
                krishiv_common::error::ErrorClass::InvalidParameterValueTimeUnit
            );
        }
        let units = StringArray::from(vec![Some("HOUR"), Some("bad")]);
        assert!(time_trunc(&units, &t).is_ok());
    }
    #[test]
    fn narrow_integral_casts_overflow_like_spark() {
        use crate::cast::{CastMode, CastOptions, FixedOffset, cast};
        use arrow_schema::DataType;
        let zone = FixedOffset(0);
        let t = times(&[Some(112 * S), Some(86_399 * S), None]);
        let run = |mode, to: &DataType| {
            cast(
                &t,
                to,
                &CastOptions {
                    mode,
                    time_zone: &zone,
                },
            )
        };
        let ok = run(CastMode::Legacy, &DataType::Int8).unwrap();
        let ok = ok
            .as_any()
            .downcast_ref::<arrow_array::Int8Array>()
            .unwrap();
        assert_eq!(ok.value(0), 112);
        assert!(ok.is_null(1) && ok.is_null(2));
        let e = run(CastMode::Ansi, &DataType::Int8).unwrap_err();
        assert_eq!(
            e.error().error_class(),
            krishiv_common::error::ErrorClass::CastOverflow
        );
        let wide = run(CastMode::Ansi, &DataType::Int16).unwrap_err();
        assert_eq!(
            wide.error().error_class(),
            krishiv_common::error::ErrorClass::CastOverflow
        );
    }

    #[test]
    fn make_time_checks_fields_in_order() {
        use arrow_array::{Decimal128Array, Int32Array};
        let h = Int32Array::from(vec![Some(1), Some(-1), Some(1), None]);
        let m = Int32Array::from(vec![Some(18), Some(0), Some(60), Some(0)]);
        let sec = Decimal128Array::from(vec![Some(19_870_000), Some(0), Some(0), Some(0)])
            .with_precision_and_scale(16, 6)
            .unwrap();
        // Row 1 fails on the hour first, row 2 on the minute.
        let e = make_time(&h.slice(0, 1), &m.slice(0, 1), &sec.slice(0, 1)).unwrap();
        assert_eq!(e.value(0), (3_600 + 18 * 60) * S + 19_870_000_000);
        for row in [1, 2] {
            let err =
                make_time(&h.slice(row, 1), &m.slice(row, 1), &sec.slice(row, 1)).unwrap_err();
            assert_eq!(
                err.error().error_class(),
                krishiv_common::error::ErrorClass::DatetimeFieldOutOfBoundsWithoutSuggestion
            );
        }
        assert!(
            make_time(&h.slice(3, 1), &m.slice(3, 1), &sec.slice(3, 1))
                .unwrap()
                .is_null(0)
        );
    }

    #[test]
    fn time_from_and_to_units_round_trip_and_refuse_out_of_day() {
        let t = time_from_units([Some(52_200_500_i128), None].into_iter(), 1_000_000).unwrap();
        assert_eq!(t.value(0), 52_200_500 * 1_000_000);
        assert!(t.is_null(1));
        assert_eq!(time_to_units(&t, 1_000_000).value(0), 52_200_500);
        for bad in [-1_i128, 86_400_000] {
            assert!(time_from_units([Some(bad)].into_iter(), 1_000_000).is_err());
        }
    }
    #[test]
    fn to_time_reads_the_default_grammar_or_a_pattern() {
        let text = StringArray::from(vec![Some("01:02:03"), Some("23-59-59.999999"), None]);
        let fmts = StringArray::from(vec![Some("HH:mm:ss"), Some("HH-mm-ss.SSSSSS"), Some("HH")]);
        let plain = to_time(&text.slice(0, 1), None, true).unwrap();
        assert_eq!(plain.value(0), 3723 * S);
        let piped = to_time(&text, Some(&fmts), false).unwrap();
        assert_eq!(piped.value(0), 3723 * S);
        assert_eq!(piped.value(1), 86_399 * S + 999_999_000);
        assert!(piped.is_null(2));
        // A NULL pattern is NULL.
        let none = StringArray::from(vec![None::<&str>]);
        assert!(
            to_time(&text.slice(0, 1), Some(&none), true)
                .unwrap()
                .is_null(0)
        );
    }

    #[test]
    fn to_time_refuses_or_nulls_what_does_not_parse() {
        let text = StringArray::from(vec![Some("13-60"), Some("abc")]);
        let fmts = StringArray::from(vec![Some("HH-mm"), Some("HH:mm")]);
        let e = to_time(&text, Some(&fmts), true).unwrap_err();
        assert_eq!(
            e.error().error_class(),
            krishiv_common::error::ErrorClass::CannotParseTime
        );
        let e = to_time(&text.slice(1, 1), None, true).unwrap_err();
        assert_eq!(
            e.error().error_class(),
            krishiv_common::error::ErrorClass::CannotParseTime
        );
        let nulls = to_time(&text, Some(&fmts), false).unwrap();
        assert_eq!(nulls.null_count(), 2);
    }
}
