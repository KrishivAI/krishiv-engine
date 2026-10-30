// SPDX-License-Identifier: Apache-2.0
//! Values as text, the way Spark's text-based file sources (CSV, JSON) read
//! and write them: Java's number parsers and renderings, which differ from
//! `CAST` (no trimming of integers, no `inf` literals, `BigDecimal` rounding),
//! and Spark's default date and timestamp grammars. Datetime patterns are in
//! [`crate::datetime_pattern`].

use krishiv_common::error::KrishivError;

use crate::cast::TimeZoneRules;
use crate::cast::datetime::{local_to_utc, utc_to_local};
use crate::cast::text::{BigDec, java_digit, java_float_string, parse_date, parse_decimal};
use crate::cast::timestamp::{Parsed, Zone, parse};

/// Java's `Long.parseLong`: an optional sign, then decimal digits (any
/// Unicode `Nd` digit, as `Character.digit`); no whitespace.
#[must_use]
pub fn java_parse_long(s: &str) -> Option<i64> {
    let (negative, digits) = match s.as_bytes().first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() {
        return None;
    }
    let mut v: i128 = 0;
    for c in digits.chars() {
        v = v.checked_mul(10)?.checked_add(i128::from(java_digit(c)?))?;
        if v > i128::from(i64::MAX).saturating_add(1) {
            return None;
        }
    }
    i64::try_from(if negative { v.checked_neg()? } else { v }).ok()
}

/// Java's `Integer.parseInt` / `Short.parseShort` / `Byte.parseByte` for an
/// integral type of `bits` bits.
#[must_use]
pub fn java_parse_integral(s: &str, bits: u32) -> Option<i64> {
    let v = java_parse_long(s)?;
    if bits >= 64 {
        return Some(v);
    }
    let bound = 1_i64.checked_shl(bits.saturating_sub(1))?;
    (v >= bound.saturating_neg() && v < bound).then_some(v)
}

/// Java's `Double.parseDouble` (leading and trailing characters `<= ' '`
/// are ignored).
#[must_use]
pub fn java_parse_double(s: &str) -> Option<f64> {
    crate::cast::text::parse_java_float(java_trim(s))
}

/// Java's `Float.parseFloat`, rounding the decimal once to `f32`.
#[must_use]
pub fn java_parse_float(s: &str) -> Option<f32> {
    crate::cast::text::parse_java_float(java_trim(s))
}

fn java_trim(s: &str) -> &str {
    s.trim_matches(|c: char| c <= ' ')
}

/// `new java.math.BigDecimal(s)` (no whitespace allowed) rounded
/// `HALF_UP` to `scale`, as an unscaled value of at most `precision`
/// digits; `None` when `s` is not a decimal or the value does not fit.
#[must_use]
pub fn java_parse_decimal(s: &str, precision: u8, scale: u8) -> Option<i128> {
    if s.is_empty() || s.chars().any(|c| c <= ' ') {
        return None;
    }
    let d: BigDec = parse_decimal(s)?;
    let v = d.to_scale(scale)?;
    let v = v.to_i128()?;
    let limit = 10_i128.checked_pow(u32::from(precision))?;
    (v.checked_abs()? < limit).then_some(v)
}

/// Java's `Double.toString`.
#[must_use]
pub fn java_double_string(v: f64) -> String {
    java_float_string(v, v.is_nan(), v.is_infinite().then_some(v < 0.0))
}

/// Java's `Float.toString`.
#[must_use]
pub fn java_float_string32(v: f32) -> String {
    java_float_string(v, v.is_nan(), v.is_infinite().then_some(v < 0.0))
}

/// Java's `BigDecimal.toString` of `unscaled × 10^-scale`: plain unless the
/// adjusted exponent is below -6 (`1E-10`, `-1.5E-7`, `0E-10`).
#[must_use]
pub fn java_decimal_string(unscaled: i128, scale: i8) -> String {
    let digits = unscaled.unsigned_abs().to_string();
    let sign = if unscaled < 0 { "-" } else { "" };
    let n = i64::try_from(digits.len()).unwrap_or(i64::MAX);
    let scale = i64::from(scale);
    let adjusted = n.saturating_sub(1).saturating_sub(scale);
    if scale >= 0 && adjusted >= -6 {
        if scale == 0 {
            return format!("{sign}{digits}");
        }
        let scale = usize::try_from(scale).unwrap_or(0);
        return if digits.len() > scale {
            let (int, frac) = digits.split_at(digits.len().saturating_sub(scale));
            format!("{sign}{int}.{frac}")
        } else {
            format!(
                "{sign}0.{}{digits}",
                "0".repeat(scale.saturating_sub(digits.len()))
            )
        };
    }
    let (first, rest) = digits.split_at(1);
    let exp_sign = if adjusted > 0 { "+" } else { "" };
    if rest.is_empty() {
        format!("{sign}{first}E{exp_sign}{adjusted}")
    } else {
        format!("{sign}{first}.{rest}E{exp_sign}{adjusted}")
    }
}

/// Spark's `DateTimeUtils.stringToDate` (the default date grammar of the
/// CSV and JSON readers): days since the epoch, or `None`.
#[must_use]
pub fn spark_string_to_date(s: &str) -> Option<i32> {
    parse_date(s.as_bytes())
}

/// Spark's `DateTimeUtils.stringToTimestamp` (the default timestamp
/// grammar of the CSV and JSON readers) as an instant: a zone or offset in
/// `s`, else `session`. A time of day without a date (`10:00`) is on
/// `today` (days since the epoch in the session zone), as in Spark. A zone
/// id the database cannot resolve reads as no timestamp, as in Spark's
/// lenient readers.
pub fn spark_string_to_timestamp(
    s: &str,
    session: &dyn TimeZoneRules,
    today: i32,
) -> Result<Option<i64>, KrishivError> {
    let parsed = match parse(s.as_bytes()) {
        Parsed::NeedsClock => {
            let time = s.trim_matches(|c: char| c <= ' ' || c == '\u{7f}');
            let time = time.strip_prefix('T').unwrap_or(time);
            let mut text = String::new();
            crate::cast::datetime::format_date(i64::from(today), &mut text);
            text.push('T');
            text.push_str(time);
            parse(text.as_bytes())
        }
        p => p,
    };
    match parsed {
        Parsed::At(local, Zone::Session) => Ok(local_to_utc(local, session)),
        Parsed::At(local, Zone::Offset(o)) => Ok(i64::from(o)
            .checked_mul(1_000_000)
            .and_then(|d| local.checked_sub(d))),
        // A zone id the database cannot resolve reads as no timestamp, as
        // in Spark's lenient readers.
        Parsed::At(local, Zone::Region(id)) => {
            let Ok(zone) = crate::datetime::resolve_zone(&id) else {
                return Ok(None);
            };
            Ok(local_to_utc(local, &zone))
        }
        Parsed::NeedsClock | Parsed::Malformed => Ok(None),
    }
}

/// Spark's `stringToTimestampWithoutTimeZone(s, allowTimeZone = false)`
/// (the readers' default `TIMESTAMP_NTZ` grammar): the local date-time, or
/// `None` when malformed, when `s` names a zone, or for a bare time of day.
#[must_use]
pub fn spark_string_to_timestamp_ntz(s: &str) -> Option<i64> {
    match parse(s.as_bytes()) {
        Parsed::At(local, Zone::Session) => Some(local),
        _ => None,
    }
}

/// Today's date (days since the epoch) in `zone` at the UTC instant
/// `now_micros`.
#[must_use]
pub fn today_in(now_micros: i64, zone: &dyn TimeZoneRules) -> i32 {
    let local = utc_to_local(now_micros, zone).unwrap_or(now_micros);
    i32::try_from(local.div_euclid(86_400_000_000)).unwrap_or(0)
}

#[cfg(test)]
mod tests;
