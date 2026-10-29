// SPDX-License-Identifier: Apache-2.0
//! Batch formatting and parsing with compiled Java datetime patterns
//! (`date_format`, `to_date`, `to_timestamp` and friends), over Arrow
//! `Date32`/`Timestamp` arrays.
//!
//! A literal pattern compiles once per bind; a column pattern compiles per
//! distinct value through a small bounded cache the caller holds (clearing
//! a full cache is deterministic: results never depend on it). A NULL row
//! never raises. Parse failures are `CANNOT_PARSE_TIMESTAMP` with
//! `fail_on_error` (ANSI, or the plain `to_*`) and NULL without it (or
//! with the `try_*` functions); a local date-time past the microsecond
//! range raises `DATETIME_OVERFLOW` either way, as Java's
//! `ArithmeticException` escapes even `try_to_*` in Spark. Pattern errors
//! (`INVALID_DATETIME_PATTERN`, `INCONSISTENT_BEHAVIOR_CROSS_VERSION`)
//! always raise: Spark validates the pattern outside the parse `try`.

use arrow_array::{
    Array, ArrayRef, Date32Array, StringArray, TimestampMicrosecondArray, cast::AsArray,
};
use arrow_schema::DataType;

use super::resolve_zone;
use crate::cast::datetime::{MICROS_PER_DAY, local_to_utc, utc_to_local};
use crate::datetime_pattern::parse::Fail;
use crate::datetime_pattern::{DateTimePattern, Zone};
use crate::error::{KernelError, with_capacity};

/// How one call handles a row whose string does not parse.
#[derive(Debug, Clone, Copy)]
pub struct ParsePolicy {
    /// Raise `CANNOT_PARSE_TIMESTAMP` (`to_*` with ANSI on); else NULL.
    pub fail_on_error: bool,
    /// The `func` parameter of the error (`try_to_timestamp`,
    /// `try_to_date`): what to use to tolerate the input.
    pub func: &'static str,
}

/// `format_timestamps` over one batch: each instant shown in `zone`.
/// `zone.id` is the session zone id (`VV` and `z` print it).
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a `z` display name the database
/// cannot supply; `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be
/// allocated.
pub fn format_timestamps(
    pattern: &DateTimePattern,
    stamps: &TimestampMicrosecondArray,
    zone: Zone<'_>,
) -> Result<StringArray, KernelError> {
    let mut out = with_capacity::<Option<String>>(stamps.len())?;
    for i in 0..stamps.len() {
        if stamps.is_null(i) {
            out.push(None);
            continue;
        }
        let s = pattern
            .format_timestamp(stamps.value(i), zone)
            .map_err(|e| KernelError::with_row(Some(i), e))?;
        out.push(Some(s));
    }
    Ok(StringArray::from(out))
}

/// Parses `strings` as instants with `pattern`: the zone or offset a
/// string names, else `session`.
///
/// # Errors
/// `CANNOT_PARSE_TIMESTAMP` with `fail_on_error`, else NULL per failing
/// row; `DATETIME_OVERFLOW` past the microsecond range either way;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be allocated.
pub fn parse_timestamps(
    pattern: &DateTimePattern,
    strings: &StringArray,
    session: &dyn crate::cast::TimeZoneRules,
    policy: ParsePolicy,
    operation: &str,
) -> Result<TimestampMicrosecondArray, KernelError> {
    let mut values = with_capacity::<i64>(strings.len())?;
    let mut valid = with_capacity::<bool>(strings.len())?;
    for i in 0..strings.len() {
        if strings.is_null(i) {
            values.push(0);
            valid.push(false);
            continue;
        }
        match parse_timestamp_one(pattern, strings.value(i), session, policy, operation, i)? {
            Some(micros) => {
                values.push(micros);
                valid.push(true);
            }
            None => {
                values.push(0);
                valid.push(false);
            }
        }
    }
    finish(values, valid)
}

/// One string of [`parse_timestamps`]: the instant, or `None` where the
/// row is NULL-worthy under `policy`.
///
/// # Errors
/// `CANNOT_PARSE_TIMESTAMP` with `fail_on_error`;
/// `DATETIME_OVERFLOW` past the microsecond range either way.
pub fn parse_timestamp_one(
    pattern: &DateTimePattern,
    s: &str,
    session: &dyn crate::cast::TimeZoneRules,
    policy: ParsePolicy,
    operation: &str,
    row: usize,
) -> Result<Option<i64>, KernelError> {
    match instant_of(pattern, s, session) {
        Ok(micros) => Ok(Some(micros)),
        Err(ParseFail::Parse(_)) if !policy.fail_on_error => Ok(None),
        Err(ParseFail::Parse(message)) => Err(KernelError::cannot_parse_timestamp(
            row,
            message,
            policy.func,
        )),
        Err(ParseFail::Overflow) => Err(KernelError::datetime_overflow(row, operation)),
    }
}

/// Parses `strings` as dates with `pattern` (as `to_date` does: through an
/// instant, read back in `session`).
///
/// # Errors
/// As [`parse_timestamps`].
pub fn parse_dates(
    pattern: &DateTimePattern,
    strings: &StringArray,
    session: &dyn crate::cast::TimeZoneRules,
    policy: ParsePolicy,
    operation: &str,
) -> Result<Date32Array, KernelError> {
    let mut values = with_capacity::<i32>(strings.len())?;
    let mut valid = with_capacity::<bool>(strings.len())?;
    for i in 0..strings.len() {
        if strings.is_null(i) {
            values.push(0);
            valid.push(false);
            continue;
        }
        match parse_date_one(pattern, strings.value(i), session, policy, operation, i)? {
            Some(day) => {
                values.push(day);
                valid.push(true);
            }
            None => {
                values.push(0);
                valid.push(false);
            }
        }
    }
    finish(values, valid)
}

/// One string of [`parse_dates`]: the date, or `None` where the row is
/// NULL-worthy under `policy`.
///
/// # Errors
/// As [`parse_timestamp_one`].
pub fn parse_date_one(
    pattern: &DateTimePattern,
    s: &str,
    session: &dyn crate::cast::TimeZoneRules,
    policy: ParsePolicy,
    operation: &str,
    row: usize,
) -> Result<Option<i32>, KernelError> {
    let stamps = parse_timestamp_one(pattern, s, session, policy, operation, row)?;
    let Some(micros) = stamps else {
        return Ok(None);
    };
    match utc_to_local(micros, session)
        .map(|local| local.div_euclid(MICROS_PER_DAY))
        .and_then(|days| i32::try_from(days).ok())
    {
        Some(day) => Ok(Some(day)),
        // A date past the `DATE` range overflows, as in Spark, even
        // without ANSI (unverified against a Spark run: no golden
        // reaches it).
        None => Err(KernelError::datetime_overflow(row, operation)),
    }
}

/// Parses `strings` as local date-times with `pattern` (as
/// `to_timestamp_ntz` does): a zone or offset in the text is validated
/// and ignored.
///
/// # Errors
/// As [`parse_timestamps`].
pub fn parse_ntz(
    pattern: &DateTimePattern,
    strings: &StringArray,
    policy: ParsePolicy,
    operation: &str,
) -> Result<TimestampMicrosecondArray, KernelError> {
    let mut values = with_capacity::<i64>(strings.len())?;
    let mut valid = with_capacity::<bool>(strings.len())?;
    for i in 0..strings.len() {
        if strings.is_null(i) {
            values.push(0);
            valid.push(false);
            continue;
        }
        match parse_ntz_one(pattern, strings.value(i), policy, operation, i)? {
            Some(micros) => {
                values.push(micros);
                valid.push(true);
            }
            None => {
                values.push(0);
                valid.push(false);
            }
        }
    }
    finish(values, valid)
}

/// One string of [`parse_ntz`]: the local date-time, or `None` where the
/// row is NULL-worthy under `policy`.
///
/// # Errors
/// As [`parse_timestamp_one`].
pub fn parse_ntz_one(
    pattern: &DateTimePattern,
    s: &str,
    policy: ParsePolicy,
    operation: &str,
    row: usize,
) -> Result<Option<i64>, KernelError> {
    match local_of(pattern, s) {
        Ok(micros) => Ok(Some(micros)),
        Err(ParseFail::Parse(_)) if !policy.fail_on_error => Ok(None),
        Err(ParseFail::Parse(message)) => Err(KernelError::cannot_parse_timestamp(
            row,
            message,
            policy.func,
        )),
        Err(ParseFail::Overflow) => Err(KernelError::datetime_overflow(row, operation)),
    }
}

/// A parse failure that is not a hard error.
pub(crate) enum ParseFail {
    /// The string does not parse; Java's message text.
    Parse(String),
    /// The local date-time is past the microsecond range.
    Overflow,
}

/// `s` as an instant: its zone or offset, else `session`.
pub(crate) fn instant_of(
    pattern: &DateTimePattern,
    s: &str,
    session: &dyn crate::cast::TimeZoneRules,
) -> Result<i64, ParseFail> {
    let parsed = pattern.parse_detailed(s).map_err(|fail| match fail {
        Fail::At(i) => ParseFail::Parse(format!("Text '{s}' could not be parsed at index {i}")),
        Fail::Trailing(i) => ParseFail::Parse(format!(
            "Text '{s}' could not be parsed, unparsed text found at index {i}"
        )),
        Fail::InParse(cause) => {
            ParseFail::Parse(format!("Text '{s}' could not be parsed: {cause}"))
        }
        Fail::PostParse(cause) => ParseFail::Parse(cause),
        Fail::Overflow => ParseFail::Overflow,
    })?;
    let local = parsed.local_micros().ok_or(ParseFail::Overflow)?;
    match parsed.zone() {
        crate::datetime_pattern::parse::ParsedZone::None => {
            local_to_utc(local, session).ok_or(ParseFail::Overflow)
        }
        crate::datetime_pattern::parse::ParsedZone::Offset(o) => local
            .checked_sub(
                i64::from(*o)
                    .checked_mul(1_000_000)
                    .ok_or(ParseFail::Overflow)?,
            )
            .ok_or(ParseFail::Overflow),
        crate::datetime_pattern::parse::ParsedZone::Region(id) => {
            let zone = resolve_zone(id)
                .map_err(|_| ParseFail::Parse(format!("Unknown time-zone ID: {id}")))?;
            local_to_utc(local, &zone).ok_or(ParseFail::Overflow)
        }
    }
}

/// `s` as a local date-time: a zone or offset in the text is validated and
/// ignored.
pub(crate) fn local_of(pattern: &DateTimePattern, s: &str) -> Result<i64, ParseFail> {
    let parsed = pattern.parse_detailed(s).map_err(|fail| match fail {
        Fail::At(i) => ParseFail::Parse(format!("Text '{s}' could not be parsed at index {i}")),
        Fail::Trailing(i) => ParseFail::Parse(format!(
            "Text '{s}' could not be parsed, unparsed text found at index {i}"
        )),
        Fail::InParse(cause) => {
            ParseFail::Parse(format!("Text '{s}' could not be parsed: {cause}"))
        }
        Fail::PostParse(cause) => ParseFail::Parse(cause),
        Fail::Overflow => ParseFail::Overflow,
    })?;
    if let crate::datetime_pattern::parse::ParsedZone::Region(id) = parsed.zone() {
        resolve_zone(id).map_err(|_| ParseFail::Parse(format!("Unknown time-zone ID: {id}")))?;
    }
    parsed.local_micros().ok_or(ParseFail::Overflow)
}

fn finish<T: arrow_array::ArrowPrimitiveType>(
    values: Vec<T::Native>,
    valid: Vec<bool>,
) -> Result<arrow_array::PrimitiveArray<T>, KernelError> {
    let nulls = valid
        .iter()
        .any(|v| !v)
        .then(|| arrow_buffer::NullBuffer::from(valid));
    Ok(arrow_array::PrimitiveArray::<T>::new(values.into(), nulls))
}

/// A decimal number of seconds as micros: digits beyond the micros are
/// dropped toward zero (the `longValue` of Spark's `BigDecimal` product),
/// and `None` where the micros leave the `i64` range.
fn decimal_seconds_to_micros(value: i128, scale: i8) -> Option<i64> {
    // Decimal scales stay far from the `i32` limits in practice; the
    // checked steps below make any other scale `None` instead of wrapping.
    let scale = i32::from(scale);
    if scale <= 6 {
        let factor = 10_i128.checked_pow(u32::try_from(6_i32.checked_sub(scale)?).ok()?)?;
        value.checked_mul(factor)?.try_into().ok()
    } else {
        let divisor = 10_i128.checked_pow(u32::try_from(scale.checked_sub(6)?).ok()?)?;
        // Integer division truncates toward zero.
        value.checked_div(divisor)?.try_into().ok()
    }
}

/// A numeric `to_timestamp` argument as an instant: Spark's
/// `longToTimestamp` treats integrals as seconds since the epoch
/// (`to_timestamp(1)` is one second past it), and fractional seconds the
/// same way (digits beyond the micros are dropped). Overflow raises `DATETIME_OVERFLOW`, as Java's
/// `ArithmeticException` escapes even `try_to_timestamp`.
pub fn seconds_to_timestamps(
    a: &ArrayRef,
    operation: &str,
) -> Result<TimestampMicrosecondArray, KernelError> {
    use arrow_array::types::{
        Decimal128Type as D128, Float32Type as F32, Float64Type as F64, Int8Type as I8,
        Int16Type as I16, Int32Type as I32, Int64Type as I64,
    };
    // Micros per row (`None` is NULL input; NaN and infinities are NULL,
    // as in Spark), or `Err` for the rows whose conversion overflows (or,
    // for decimals, is inexact).
    let micros: Vec<Result<Option<i64>, ()>> = match a.data_type() {
        DataType::Int8 => a
            .as_primitive::<I8>()
            .iter()
            .map(|v| match v {
                None => Ok(None),
                Some(s) => i64::from(s).checked_mul(1_000_000).map(Some).ok_or(()),
            })
            .collect(),
        DataType::Int16 => a
            .as_primitive::<I16>()
            .iter()
            .map(|v| match v {
                None => Ok(None),
                Some(s) => i64::from(s).checked_mul(1_000_000).map(Some).ok_or(()),
            })
            .collect(),
        DataType::Int32 => a
            .as_primitive::<I32>()
            .iter()
            .map(|v| match v {
                None => Ok(None),
                Some(s) => i64::from(s).checked_mul(1_000_000).map(Some).ok_or(()),
            })
            .collect(),
        DataType::Int64 => a
            .as_primitive::<I64>()
            .iter()
            .map(|v| match v {
                None => Ok(None),
                Some(s) => s.checked_mul(1_000_000).map(Some).ok_or(()),
            })
            .collect(),
        DataType::Float32 => a
            .as_primitive::<F32>()
            .iter()
            .map(|v| {
                Ok(v.and_then(|x| {
                    if x.is_nan() || x.is_infinite() {
                        None
                    } else {
                        // As Java's narrowing `long` cast, which saturates.
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "Java's `(long)(d * 1e6)` saturates the same way `as` does"
                        )]
                        let micros = (f64::from(x) * 1_000_000.0) as i64;
                        Some(micros)
                    }
                }))
            })
            .collect(),
        DataType::Float64 => a
            .as_primitive::<F64>()
            .iter()
            .map(|v| {
                Ok(v.and_then(|x| {
                    if x.is_nan() || x.is_infinite() {
                        None
                    } else {
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "Java's `(long)(d * 1e6)` saturates the same way `as` does"
                        )]
                        let micros = (x * 1_000_000.0) as i64;
                        Some(micros)
                    }
                }))
            })
            .collect(),
        DataType::Decimal128(_, scale) => {
            let scale = *scale;
            a.as_primitive::<D128>()
                .iter()
                .map(|v| match v {
                    None => Ok(None),
                    Some(d) => decimal_seconds_to_micros(d, scale).map(Some).ok_or(()),
                })
                .collect()
        }
        t => {
            return Err(KernelError::internal(format!(
                "to_timestamp with a non-numeric argument of type {t}"
            )));
        }
    };
    let mut values = with_capacity::<i64>(a.len())?;
    let mut valid = with_capacity::<bool>(a.len())?;
    for (i, m) in micros.iter().enumerate() {
        match m {
            Ok(None) => {
                values.push(0);
                valid.push(false);
            }
            Ok(Some(m)) => {
                values.push(*m);
                valid.push(true);
            }
            // Overflow raises even without ANSI (see above).
            Err(()) => return Err(KernelError::datetime_overflow(i, operation)),
        }
    }
    finish(values, valid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cast::FixedOffset;
    use crate::datetime_pattern::Purpose;

    fn utc() -> FixedOffset {
        FixedOffset(0)
    }

    fn pattern(p: &str) -> DateTimePattern {
        DateTimePattern::new(p, Purpose::Parse).unwrap()
    }

    /// Batch parsing reports Java's failure text per row and NULLs without
    /// ANSI; overflow always raises. Unverified against a Spark run (the
    /// binding tests pin the golden messages).
    #[test]
    fn batches_parse_and_fail_per_row() {
        let policy = ParsePolicy {
            fail_on_error: true,
            func: "try_to_timestamp",
        };
        let p = pattern("yyyy-MM-dd");
        let s = StringArray::from(vec![Some("2020-01-05"), None, Some("xx")]);
        let out = parse_timestamps(&p, &s, &utc(), policy, "to_timestamp").unwrap_err();
        assert_eq!(
            out.error_class(),
            krishiv_common::error::ErrorClass::CannotParseTimestamp
        );
        assert_eq!(out.row(), Some(2));
        let lenient = ParsePolicy {
            fail_on_error: false,
            ..policy
        };
        let out = parse_timestamps(&p, &s, &utc(), lenient, "to_timestamp").unwrap();
        assert_eq!(
            out.iter().collect::<Vec<_>>(),
            [Some(18_266 * 86_400_000_000), None, None]
        );
        // Overflow raises either way.
        let big = pattern("y");
        let s = StringArray::from(vec![Some("294248")]);
        for policy in [policy, lenient] {
            let e = parse_timestamps(&big, &s, &utc(), policy, "to_timestamp").unwrap_err();
            assert_eq!(
                e.error_class(),
                krishiv_common::error::ErrorClass::DatetimeOverflow
            );
        }
        // Dates read back in the session zone.
        let out = parse_dates(&p, &s, &utc(), lenient, "to_date").unwrap();
        assert!(out.is_null(0));
    }

    /// Numerics read as seconds; inexact decimals and overflow raise even
    /// without ANSI. Unverified against a Spark run.
    #[test]
    fn seconds_to_timestamps_cover_the_numeric_types() {
        use arrow_array::{Decimal128Array, Float64Array, Int32Array, Int64Array};
        use std::sync::Arc;

        let ints: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None]));
        let out = seconds_to_timestamps(&ints, "to_timestamp").unwrap();
        assert_eq!(out.iter().collect::<Vec<_>>(), [Some(1_000_000), None]);
        let nan: ArrayRef = Arc::new(Float64Array::from(vec![Some(f64::NAN)]));
        assert!(
            seconds_to_timestamps(&nan, "to_timestamp")
                .unwrap()
                .is_null(0)
        );
        // 1.5 seconds is exact ...
        let exact: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(15)])
                .with_precision_and_scale(2, 1)
                .unwrap(),
        );
        let out = seconds_to_timestamps(&exact, "to_timestamp").unwrap();
        assert_eq!(out.value(0), 1_500_000);
        // ... while a nonzero seventh fractional digit is not.
        let inexact: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(15_000_001)])
                .with_precision_and_scale(8, 7)
                .unwrap(),
        );
        let e = seconds_to_timestamps(&inexact, "to_timestamp").unwrap_err();
        assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::DatetimeOverflow
        );
        assert_eq!(e.row(), Some(0));
        let big: ArrayRef = Arc::new(Int64Array::from(vec![Some(i64::MAX)]));
        let e = seconds_to_timestamps(&big, "to_timestamp").unwrap_err();
        assert_eq!(
            e.error_class(),
            krishiv_common::error::ErrorClass::DatetimeOverflow
        );
    }
}
