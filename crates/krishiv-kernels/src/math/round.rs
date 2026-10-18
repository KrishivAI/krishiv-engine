// SPDX-License-Identifier: Apache-2.0
//! `round` / `bround` / `ceil` / `floor`.
//!
//! - Integral types: a scale of zero or more is the identity; a negative
//!   scale rounds to tens, hundreds, ...; a result outside the type is
//!   `ARITHMETIC_OVERFLOW` under ANSI and wraps (Java) otherwise.
//! - `double` / `float`: Spark rounds the shortest decimal text of the value
//!   (`BigDecimal.valueOf`), so `round(2.675, 2)` is `2.68`, then converts
//!   back; a `float` goes through its exact `double` value.
//! - Decimals: exact; the result type is [`round_decimal_type`]. A result
//!   that does not fit raises `NUMERIC_VALUE_OUT_OF_RANGE.WITHOUT_SUGGESTION`
//!   in every mode.
//! - `ceil` / `floor` of a `double` or `float` give a `bigint` (saturating,
//!   NaN is 0), of an integral type the same value as `bigint`.

use arrow_array::{Array, Decimal128Array, Int64Array, PrimitiveArray};
use arrow_buffer::ScalarBuffer;
use arrow_schema::DataType;

use crate::error::KernelError;
use crate::integer::SparkIntegral;
use crate::policy::OverflowMode;

/// `10^k` for `k <= 38`.
const POW10: [i128; 39] = {
    let mut t = [1_i128; 39];
    let mut i = 1;
    while i < 39 {
        t[i] = t[i - 1].wrapping_mul(10);
        i += 1;
    }
    t
};

fn pow10(k: u32) -> Option<i128> {
    POW10.get(usize::try_from(k).ok()?).copied()
}

/// `round` (half away from zero) or `bround` (half to even).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoundMode {
    /// `round`: ties go away from zero.
    HalfUp,
    /// `bround`: ties go to the even neighbour.
    HalfEven,
}

/// How a decimal is rounded: `round`, `bround`, `ceil` or `floor`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecimalRounding {
    /// `round`
    HalfUp,
    /// `bround`
    HalfEven,
    /// `ceil`
    Ceiling,
    /// `floor`
    Floor,
}

impl From<RoundMode> for DecimalRounding {
    fn from(m: RoundMode) -> Self {
        match m {
            RoundMode::HalfUp => Self::HalfUp,
            RoundMode::HalfEven => Self::HalfEven,
        }
    }
}

/// `value / 10^drop` rounded as `mode` says (the number of whole units).
fn round_units(value: i128, drop: u32, mode: DecimalRounding) -> i128 {
    let Some(unit) = pow10(drop) else {
        // The unit exceeds 10^38 > |value|: only ceil and floor can move.
        return match mode {
            DecimalRounding::Ceiling => i128::from(value > 0),
            DecimalRounding::Floor => 0_i128.wrapping_sub(i128::from(value < 0)),
            _ => 0,
        };
    };
    let (q, r) = (
        value.checked_div(unit).unwrap_or(0),
        value.checked_rem(unit).unwrap_or(0),
    );
    if r == 0 {
        return q;
    }
    let sign = if value < 0 { -1 } else { 1 };
    let away = match mode {
        DecimalRounding::Ceiling => value > 0,
        DecimalRounding::Floor => value < 0,
        // 2|r| >= unit, written so it cannot overflow.
        DecimalRounding::HalfUp => r.abs() >= unit.wrapping_sub(r.abs()),
        DecimalRounding::HalfEven => {
            let rest = unit.wrapping_sub(r.abs());
            r.abs() > rest || (r.abs() == rest && q & 1 != 0)
        }
    };
    if away { q.wrapping_add(sign) } else { q }
}

/// `BigDecimal.setScale` fails for a scale so far below the value's that the
/// scale difference overflows an `int` ("Underflow") or `10^difference` is
/// too big for a `BigInteger` (its size check). Spark raises these as
/// unclassified `ArithmeticException`s for decimals and floating point, and
/// as `ARITHMETIC_OVERFLOW` for integral types; a zero never does.
fn extreme_scale(value_scale: i64, scale: i32) -> Option<&'static str> {
    /// `BigInteger.pow(10, n)` refuses `n` whose result needs more than
    /// `MAX_MAG_LENGTH` (2^26 + 1) words of 32 bits at 4 bits per digit.
    const BIGINT_LIMIT: i64 = 536_870_920;
    if scale >= 0 {
        return None;
    }
    let diff = value_scale.saturating_sub(i64::from(scale));
    if diff > i64::from(i32::MAX) {
        Some("Underflow")
    } else if diff >= BIGINT_LIMIT {
        Some("BigInteger would overflow supported range")
    } else {
        None
    }
}

/// `round` / `bround` of an integral column to `scale` digits.
///
/// # Errors
/// `ARITHMETIC_OVERFLOW` under `OverflowMode::AnsiError` for a result the
/// type cannot hold; any other mode wraps.
pub fn round_integral<T: SparkIntegral>(
    input: &PrimitiveArray<T>,
    scale: i32,
    mode: RoundMode,
    overflow: OverflowMode,
) -> Result<PrimitiveArray<T>, KernelError> {
    if scale >= 0 {
        return Ok(input.clone());
    }
    let drop = scale.unsigned_abs();
    let extreme = extreme_scale(0, scale);
    let mut out = crate::error::with_capacity::<T::Native>(input.len())?;
    for (row, v) in input.values().iter().enumerate() {
        let valid = input.is_valid(row);
        if let Some(message) = extreme
            && valid
            && T::widen(*v) != 0
        {
            return Err(KernelError::arithmetic_overflow(row, message, None));
        }
        let units = round_units(T::widen(*v), drop, mode.into());
        // The rounded value, in units of 10^drop; drop <= 38 or the unit is 0.
        let rounded = pow10(drop).map_or(0, |u| units.wrapping_mul(u));
        let (n, overflowed) = T::narrow(rounded);
        if valid && overflowed && overflow == OverflowMode::AnsiError {
            return Err(KernelError::arithmetic_overflow(row, "Overflow", None));
        }
        out.push(n);
    }
    Ok(PrimitiveArray::new(
        ScalarBuffer::from(out),
        input.nulls().cloned(),
    ))
}

/// `round` / `bround` of a `double` to `scale` digits.
///
/// # Errors
/// `ARITHMETIC_OVERFLOW` for a scale so far below the value's that Spark's
/// `BigDecimal` rounding fails (see [`round_integral`]).
pub fn round_f64(
    input: &arrow_array::Float64Array,
    scale: i32,
    mode: RoundMode,
) -> Result<arrow_array::Float64Array, KernelError> {
    input
        .iter()
        .enumerate()
        .map(|(row, v)| {
            v.map(|v| {
                round_double(v, scale, mode)
                    .map_err(|m| KernelError::arithmetic_overflow(row, m, None))
            })
            .transpose()
        })
        .collect()
}

/// `round` / `bround` of a `float`: the exact `double` value is rounded and
/// converted back.
///
/// # Errors
/// As [`round_f64`].
pub fn round_f32(
    input: &arrow_array::Float32Array,
    scale: i32,
    mode: RoundMode,
) -> Result<arrow_array::Float32Array, KernelError> {
    input
        .iter()
        .enumerate()
        .map(|(row, v)| {
            v.map(|v| {
                round_double(f64::from(v), scale, mode)
                    .map(narrow_f32)
                    .map_err(|m| KernelError::arithmetic_overflow(row, m, None))
            })
            .transpose()
        })
        .collect()
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "Java's (float) cast rounds to nearest"
)]
fn narrow_f32(v: f64) -> f32 {
    v as f32
}

/// `BigDecimal.valueOf(v).setScale(scale, mode).doubleValue()`.
fn round_double(v: f64, scale: i32, mode: RoundMode) -> Result<f64, &'static str> {
    if v == 0.0 {
        // BigDecimal has no negative zero.
        return Ok(0.0);
    }
    if !v.is_finite() {
        return Ok(v);
    }
    // Shortest round-trip digits: "d.ddde<exp>".
    let text = format!("{:e}", v.abs());
    let Some((mantissa, exp)) = text.split_once('e') else {
        return Ok(v);
    };
    let digits: Vec<u8> = mantissa
        .bytes()
        .filter(u8::is_ascii_digit)
        .map(|b| b.wrapping_sub(b'0'))
        .collect();
    let Ok(exp) = exp.parse::<i64>() else {
        return Ok(v);
    };
    // The value is 0.d1d2...dn * 10^point.
    let point = exp.saturating_add(1);
    let keep = point.saturating_add(i64::from(scale));
    let n = i64::try_from(digits.len()).unwrap_or(i64::MAX);
    // `BigDecimal.valueOf(v)` parses `Double.toString(v)`, which always has a
    // fraction digit ("25.0", "1.0E-5"): that digit counts toward its scale.
    let java_scale = if (1e-3..1e7).contains(&v.abs()) {
        n.saturating_sub(point).max(1)
    } else {
        n.max(2).saturating_sub(point)
    };
    if let Some(message) = extreme_scale(java_scale, scale) {
        return Err(message);
    }
    if keep >= n {
        return Ok(v);
    }
    let (mut kept, mut point) = if keep < 0 {
        (Vec::new(), point)
    } else {
        let k = usize::try_from(keep).unwrap_or(0);
        (digits[..k].to_vec(), point)
    };
    // The digit after the kept ones decides; the rest breaks ties.
    let (next, tail_nonzero) = if keep < 0 {
        (0, false)
    } else {
        let k = usize::try_from(keep).unwrap_or(0);
        (
            digits[k],
            digits[k.wrapping_add(1)..].iter().any(|&d| d != 0),
        )
    };
    let prev_odd = kept.last().is_some_and(|d| d % 2 == 1);
    let up = match mode {
        RoundMode::HalfUp => next >= 5,
        RoundMode::HalfEven => next > 5 || (next == 5 && (tail_nonzero || prev_odd)),
    };
    if up {
        // Add one unit in the last kept place, carrying.
        let mut i = kept.len();
        loop {
            if i == 0 {
                kept.insert(0, 1);
                point = point.saturating_add(1);
                break;
            }
            i = i.wrapping_sub(1);
            if kept[i] == 9 {
                kept[i] = 0;
            } else {
                kept[i] = kept[i].wrapping_add(1);
                break;
            }
        }
    }
    if kept.iter().all(|&d| d == 0) {
        return Ok(0.0);
    }
    let mut text = String::from("0.");
    text.extend(kept.iter().map(|d| char::from(b'0'.wrapping_add(*d))));
    text.push('e');
    text.push_str(&point.to_string());
    let rounded: f64 = text.parse().unwrap_or(v);
    Ok(if v < 0.0 { -rounded } else { rounded })
}

/// The result type of rounding a `decimal(p, s)` to `scale` digits: scale
/// `min(s, scale)` (0 for a negative `scale`) and one more integral digit
/// than the input, or `1 - scale` digits for a negative `scale`, capped at 38.
#[must_use]
pub fn round_decimal_type(precision: u8, scale: i8, input_scale: i32) -> (u8, i8) {
    let (p, s) = (i32::from(precision), i32::from(scale));
    let new_scale = if input_scale < 0 {
        0
    } else {
        s.min(input_scale)
    };
    let integral = if input_scale < 0 {
        // `-scale + 1` wraps for the smallest scale (Spark's Int arithmetic).
        p.saturating_sub(s)
            .saturating_add(1)
            .max(1_i32.wrapping_sub(input_scale))
    } else {
        p.saturating_sub(s).saturating_add(1)
    };
    let precision = integral.saturating_add(new_scale).clamp(1, 38);
    (
        u8::try_from(precision).unwrap_or(38),
        i8::try_from(new_scale).unwrap_or(0),
    )
}

/// Java's `BigDecimal.toString` of `unscaled * 10^-scale`.
fn java_decimal_string(unscaled: i128, scale: i32) -> String {
    let digits = unscaled.unsigned_abs().to_string();
    let sign = if unscaled < 0 { "-" } else { "" };
    if scale == 0 {
        return format!("{sign}{digits}");
    }
    let len = i64::try_from(digits.len()).unwrap_or(i64::MAX);
    let adjusted = len.saturating_sub(1).saturating_sub(i64::from(scale));
    if scale > 0 && adjusted >= -6 {
        let s = usize::try_from(scale).unwrap_or(0);
        return if digits.len() > s {
            let (int, frac) = digits.split_at(digits.len().saturating_sub(s));
            format!("{sign}{int}.{frac}")
        } else {
            format!(
                "{sign}0.{}{digits}",
                "0".repeat(s.saturating_sub(digits.len()))
            )
        };
    }
    let (first, rest) = digits.split_at(1);
    let point = if rest.is_empty() {
        String::new()
    } else {
        format!(".{rest}")
    };
    let plus = if adjusted > 0 { "+" } else { "" };
    format!("{sign}{first}{point}E{plus}{adjusted}")
}

/// `round` / `bround` / `ceil` / `floor` of a decimal column to `scale` digits.
///
/// # Errors
/// `INTERNAL_ERROR` for a column that is not a `decimal(p, s)` with
/// `0 <= s <= p <= 38`; `NUMERIC_VALUE_OUT_OF_RANGE.WITHOUT_SUGGESTION` for
/// a result that does not fit (every mode).
pub fn round_decimal(
    input: &Decimal128Array,
    scale: i32,
    mode: DecimalRounding,
) -> Result<Decimal128Array, KernelError> {
    let DataType::Decimal128(p, s) = input.data_type() else {
        return Err(KernelError::internal(
            "round_decimal needs a decimal column",
        ));
    };
    let s32 = i32::from(*s);
    if *s < 0 || *s > i8::try_from(*p).unwrap_or(0) {
        return Err(KernelError::unsupported(format!(
            "rounding decimal({p},{s})"
        )));
    }
    let (out_p, out_s) = round_decimal_type(*p, *s, scale);
    let limit = pow10(u32::from(out_p)).unwrap_or(i128::MAX);
    let mut out = crate::error::with_capacity::<i128>(input.len())?;
    for (row, u) in input.values().iter().enumerate() {
        if input.is_null(row) {
            out.push(0);
            continue;
        }
        if *u != 0
            && let Some(message) = extreme_scale(i64::from(*s), scale)
        {
            return Err(KernelError::arithmetic_overflow(row, message, None));
        }
        // The rounded value in units of 10^-bd_scale (a BigDecimal whose scale
        // is negative when rounding to tens, hundreds, ...).
        let (q, bd_scale) = if scale >= s32 {
            (*u, s32)
        } else {
            (
                round_units(*u, s32.saturating_sub(scale).unsigned_abs(), mode),
                scale,
            )
        };
        let value = if scale < 0 {
            if q == 0 {
                Some(0)
            } else {
                pow10(scale.unsigned_abs()).and_then(|m| q.checked_mul(m))
            }
        } else {
            Some(q)
        };
        match value {
            Some(v) if v.abs() < limit => out.push(v),
            _ => {
                // Spark: "The <rounded> rounded half up from <original> cannot
                // be represented as Decimal(p, s)" (also for ceil and floor).
                let rounded = if scale < 0 && q != 0 {
                    // Spark prints every zero; past a few hundred, show the exponent.
                    match scale.unsigned_abs() {
                        zeros @ 0..=400 => format!("{q}{}", "0".repeat(zeros as usize)),
                        zeros => format!("{q}E+{zeros}"),
                    }
                } else {
                    plain_string(q, bd_scale.max(0))
                };
                let original = java_decimal_string(q, bd_scale);
                return Err(KernelError::decimal_round_out_of_range(
                    row, rounded, original, out_p, out_s,
                ));
            }
        }
    }
    Decimal128Array::new(ScalarBuffer::from(out), input.nulls().cloned())
        .with_precision_and_scale(out_p, out_s)
        .map_err(|e| KernelError::internal(e.to_string()))
}

/// `BigDecimal.toPlainString` of `unscaled * 10^-scale`.
fn plain_string(unscaled: i128, scale: i32) -> String {
    let digits = unscaled.unsigned_abs().to_string();
    let sign = if unscaled < 0 { "-" } else { "" };
    let s = usize::try_from(scale).unwrap_or(0);
    if s == 0 {
        format!("{sign}{digits}")
    } else if digits.len() > s {
        let (int, frac) = digits.split_at(digits.len().saturating_sub(s));
        format!("{sign}{int}.{frac}")
    } else {
        format!(
            "{sign}0.{}{digits}",
            "0".repeat(s.saturating_sub(digits.len()))
        )
    }
}

/// `ceil` / `floor` of a `double` or `float` (widened to `double`) as a
/// `bigint`: saturating, NaN is 0.
#[must_use]
pub fn ceil_floor_to_long(input: &arrow_array::Float64Array, ceil: bool) -> Int64Array {
    input.unary(|v| saturating_long(if ceil { v.ceil() } else { v.floor() }))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "Java's (long) cast saturates and maps NaN to 0, as Rust's `as` does"
)]
fn saturating_long(v: f64) -> i64 {
    v as i64
}
