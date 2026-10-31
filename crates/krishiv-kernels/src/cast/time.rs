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

/// Nanoseconds since midnight for `text`, or `None` when malformed.
pub fn parse_time(text: &str) -> Option<i64> {
    let s = text.trim_matches(|c: char| c.is_ascii_whitespace());
    let s = s.strip_prefix('T').unwrap_or(s);
    let (clock, meridiem) = match s.rsplit_once(' ') {
        Some((clock, m)) if m.eq_ignore_ascii_case("AM") => (clock.trim_end(), Some(false)),
        Some((clock, m)) if m.eq_ignore_ascii_case("PM") => (clock.trim_end(), Some(true)),
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
                nanos = nanos.checked_add(i64::from(d - b'0').checked_mul(scale)?)?;
            }
            (field(whole)?, nanos)
        }
    };
    if parts.next().is_some() || minute > 59 || second > 59 {
        return None;
    }
    let hour = match meridiem {
        None if hour < 24 => hour,
        Some(pm) if (1..=12).contains(&hour) => hour % 12 + if pm { 12 } else { 0 },
        _ => return None,
    };
    Some(((hour * 60 + minute) * 60 + second) * NANOS_PER_SECOND + nanos)
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
            Self::Hour => 3_600 * NANOS_PER_SECOND,
            Self::Minute => 60 * NANOS_PER_SECOND,
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
        out.push(Some(n - n % unit.nanos()));
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
        out.push(Some((end.value(i) - start.value(i)) / unit.nanos()));
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
}
