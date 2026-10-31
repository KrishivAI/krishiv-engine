// SPDX-License-Identifier: Apache-2.0
//! Conversions between Unix time and Spark's date and timestamp types:
//! `timestamp_seconds` / `timestamp_millis` / `timestamp_micros`,
//! `unix_seconds` / `unix_millis` / `unix_micros` and `date_from_unix_date`.
//!
//! Overflow and a decimal that does not fit microseconds are
//! `DATETIME_OVERFLOW`: Spark raises a bare `ArithmeticException` there
//! (`long overflow`, `Rounding necessary`), which has no error class.

use arrow_array::{
    Array, Date32Array, Decimal128Array, Float64Array, Int32Array, Int64Array,
    TimestampMicrosecondArray,
};
use arrow_buffer::NullBuffer;

use crate::error::{KernelError, with_capacity};

const MICROS_PER_SECOND: i64 = 1_000_000;

/// `timestamp_millis` and `timestamp_micros`: `values * per` microseconds,
/// where `per` is 1,000 for milliseconds and 1 for microseconds.
///
/// # Errors
/// `DATETIME_OVERFLOW` when a product leaves `i64`.
pub fn timestamp_from_integral(
    values: &Int64Array,
    per: i64,
    name: &str,
) -> Result<TimestampMicrosecondArray, KernelError> {
    scale_into_timestamps(values, name, |v| v.checked_mul(per))
}

/// `timestamp_seconds` of an integral column.
///
/// # Errors
/// `DATETIME_OVERFLOW` when a product leaves `i64`.
pub fn timestamp_from_seconds(
    values: &Int64Array,
) -> Result<TimestampMicrosecondArray, KernelError> {
    scale_into_timestamps(values, "timestamp_seconds", |v| {
        v.checked_mul(MICROS_PER_SECOND)
    })
}

fn scale_into_timestamps(
    values: &Int64Array,
    name: &str,
    f: impl Fn(i64) -> Option<i64>,
) -> Result<TimestampMicrosecondArray, KernelError> {
    let mut out = with_capacity::<i64>(values.len())?;
    for row in 0..values.len() {
        if values.is_null(row) {
            out.push(0);
            continue;
        }
        out.push(f(values.value(row)).ok_or_else(|| KernelError::datetime_overflow(row, name))?);
    }
    Ok(TimestampMicrosecondArray::new(
        out.into(),
        values.nulls().cloned(),
    ))
}

/// `timestamp_seconds` of a `double` (a `float` widened to one): the
/// fraction is cut at microseconds, a value past the `i64` range saturates
/// as Java's `(long)` cast does, and NaN and the infinities are NULL.
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be allocated.
pub fn timestamp_from_seconds_double(
    values: &Float64Array,
) -> Result<TimestampMicrosecondArray, KernelError> {
    let mut out = with_capacity::<i64>(values.len())?;
    let mut valid = with_capacity::<bool>(values.len())?;
    for row in 0..values.len() {
        let v = if values.is_null(row) {
            None
        } else {
            Some(values.value(row)).filter(|v| v.is_finite())
        };
        match v {
            Some(v) => {
                out.push(saturating_micros(v));
                valid.push(true);
            }
            None => {
                out.push(0);
                valid.push(false);
            }
        }
    }
    Ok(TimestampMicrosecondArray::new(
        out.into(),
        Some(NullBuffer::from(valid)),
    ))
}

/// `(seconds * 1e6).toLong` with Java's saturation.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the value is truncated toward zero and clamped to the i64 range first, as Java's (long) cast"
)]
fn saturating_micros(seconds: f64) -> i64 {
    let micros = (seconds * 1e6).trunc();
    if micros >= 9_223_372_036_854_775_808.0 {
        i64::MAX
    } else if micros <= -9_223_372_036_854_775_808.0 {
        i64::MIN
    } else {
        micros as i64
    }
}

/// `timestamp_seconds` of a decimal: exact to the microsecond.
///
/// # Errors
/// `DATETIME_OVERFLOW` when the value has digits past microseconds (Spark's
/// `Rounding necessary`) or does not fit `i64` microseconds.
pub fn timestamp_from_seconds_decimal(
    values: &Decimal128Array,
) -> Result<TimestampMicrosecondArray, KernelError> {
    let scale = i32::from(values.scale());
    let mut out = with_capacity::<i64>(values.len())?;
    for row in 0..values.len() {
        if values.is_null(row) {
            out.push(0);
            continue;
        }
        let micros = decimal_micros(values.value(row), scale)
            .ok_or_else(|| KernelError::datetime_overflow(row, "timestamp_seconds"))?;
        out.push(micros);
    }
    Ok(TimestampMicrosecondArray::new(
        out.into(),
        values.nulls().cloned(),
    ))
}

fn decimal_micros(unscaled: i128, scale: i32) -> Option<i64> {
    let micros = if scale <= 6 {
        let factor = 10_i128.checked_pow(u32::try_from(6_i32.checked_sub(scale)?).ok()?)?;
        unscaled.checked_mul(factor)?
    } else {
        let divisor = 10_i128.checked_pow(u32::try_from(scale.checked_sub(6)?).ok()?)?;
        if unscaled.checked_rem(divisor)? != 0 {
            return None;
        }
        unscaled.checked_div(divisor)?
    };
    i64::try_from(micros).ok()
}

/// `unix_seconds`, `unix_millis` and `unix_micros`: the instant in units of
/// `per` microseconds (1,000,000, 1,000 or 1), rounded toward negative
/// infinity (`Math.floorDiv`).
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be allocated.
pub fn unix_time(values: &TimestampMicrosecondArray, per: i64) -> Result<Int64Array, KernelError> {
    let mut out = with_capacity::<i64>(values.len())?;
    for row in 0..values.len() {
        out.push(if values.is_null(row) {
            0
        } else {
            values.value(row).div_euclid(per)
        });
    }
    Ok(Int64Array::new(out.into(), values.nulls().cloned()))
}

/// `date_from_unix_date`: days since the epoch as a `DATE`. Every `int` is a
/// day Spark's `DATE` carries (`unix_date(date_from_unix_date(2147483647))`
/// is 2147483647), so there is nothing to check.
#[must_use]
pub fn date_from_unix_date(days: &Int32Array) -> Date32Array {
    Date32Array::new(days.values().clone(), days.nulls().cloned())
}
