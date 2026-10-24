// SPDX-License-Identifier: Apache-2.0
//! `conv(num, from_base, to_base)`: Hive's `NumberConverter.convert`.
//!
//! Invalid bases and empty input are NULL; overflow raises
//! `ARITHMETIC_OVERFLOW` in ANSI mode and yields `2^64 - 1` otherwise; a
//! negative `to_base` formats a signed result.

#![expect(
    clippy::arithmetic_side_effects,
    reason = "the 64-bit conversion checks overflow before each step, as the ported algorithm does"
)]

use arrow_array::types::Int32Type;
use arrow_array::{Array, StringArray};

use crate::error::KernelError;

/// Trims spaces (0x20) at both ends, as Spark's `UTF8String.trim` (note:
///
/// Java's `String.trim` would also strip controls).
fn trim_java(s: &str) -> &str {
    s.trim_matches(' ')
}

/// A base-36 digit, case-insensitive (`None` past the radix, as Java's
/// `Character.digit` reports invalid input).
fn digit_in_base(c: u8, radix: u32) -> Option<u32> {
    let d = match c {
        b'0'..=b'9' => u32::from(c - b'0'),
        b'a'..=b'z' => u32::from(c - b'a') + 10,
        b'A'..=b'Z' => u32::from(c - b'A') + 10,
        _ => return None,
    };
    (d < radix).then_some(d)
}

/// Converts `num` from `from_base` to `to_base`, or NULL.
fn convert_one(
    num: &str,
    from_base: i32,
    to_base: i32,
    ansi: bool,
    row: Option<usize>,
) -> Result<Option<String>, KernelError> {
    if !(2..=36).contains(&from_base)
        || !(2..=36).contains(&to_base.unsigned_abs())
        || num.is_empty()
    {
        return Ok(None);
    }
    let from = from_base.cast_unsigned();
    let to = to_base.unsigned_abs();
    let bytes = num.as_bytes();
    let (mut negative, digits) = match bytes.first() {
        Some(b'-') => (true, &bytes[1..]),
        _ => (false, bytes),
    };
    // Parse as an unsigned 64-bit value, ignoring the suffix from the
    // first invalid digit (like `char2byte` + `encode`).
    let mut value: u64 = 0;
    let bound = (u64::MAX - u64::from(from)) / u64::from(from);
    for c in digits.iter() {
        let Some(d) = digit_in_base(*c, from) else {
            break;
        };
        // `v < 0` (a set top bit) means `v * radix` overflows.
        if value >= 1_u64 << 63 {
            value = convert_overflow(ansi, row)?;
            break;
        }
        if value >= bound && (u64::MAX - u64::from(d)) / u64::from(from) < value {
            value = convert_overflow(ansi, row)?;
            break;
        }
        value = value * u64::from(from) + u64::from(d);
    }
    if negative && to_base > 0 {
        if value >= 1_u64 << 63 {
            value = u64::MAX;
        } else {
            value = (value as i64).wrapping_neg().cast_unsigned();
        }
    }
    if to_base < 0 && value >= 1_u64 << 63 {
        value = (value as i64).wrapping_neg().cast_unsigned();
        negative = true;
    }
    let mut text = if value == 0 {
        "0".to_owned()
    } else {
        let mut chars = Vec::new();
        let mut rest = value;
        while rest > 0 {
            let d = u32::try_from(rest % u64::from(to)).unwrap_or(0);
            chars.push(char::from_digit(d, to).unwrap_or('?').to_ascii_uppercase());
            rest /= u64::from(to);
        }
        chars.iter().rev().collect()
    };
    if negative && to_base < 0 {
        text = format!("-{text}");
    }
    Ok(Some(text))
}

/// Overflow of `conv`: raise in ANSI mode, `2^64 - 1` otherwise.
fn convert_overflow(ansi: bool, row: Option<usize>) -> Result<u64, KernelError> {
    if ansi {
        Err(KernelError::with_row(
            row,
            krishiv_common::error::ArithmeticOverflow {
                message: "Overflow in function conv()".to_owned(),
                alternative: String::new(),
                config: "\"spark.sql.ansi.enabled\"".to_owned(),
            }
            .into(),
        ))
    } else {
        Ok(u64::MAX)
    }
}

/// `conv` over one batch: `num` trimmed per row, bases per row.
pub fn conv(
    nums: &StringArray,
    from: &arrow_array::PrimitiveArray<Int32Type>,
    to: &arrow_array::PrimitiveArray<Int32Type>,
    ansi: bool,
) -> Result<StringArray, KernelError> {
    if nums.len() != from.len() || nums.len() != to.len() {
        return Err(KernelError::internal(
            "conv on columns of different lengths".to_owned(),
        ));
    }
    let mut out = Vec::with_capacity(nums.len());
    for i in 0..nums.len() {
        if nums.is_null(i) || from.is_null(i) || to.is_null(i) {
            out.push(None);
            continue;
        }
        out.push(convert_one(
            trim_java(nums.value(i)),
            from.value(i),
            to.value(i),
            ansi,
            Some(i),
        )?);
    }
    let refs: Vec<Option<&str>> = out.iter().map(|o| o.as_deref()).collect();
    Ok(StringArray::from(refs))
}

#[cfg(test)]
mod tests {
    use super::{conv, convert_one};

    #[test]
    fn overflow_checks_fire_before_wrapping() {
        // Past `bound` but below the top bit: the second check fires.
        let err = convert_one("92233720368547758070", 10, 16, true, Some(0)).unwrap_err();
        assert_eq!(
            err.error_class(),
            krishiv_common::error::ErrorClass::ArithmeticOverflow
        );
        // Legacy overflow formats `2^64 - 1` and keeps converting.
        assert_eq!(
            convert_one("92233720368547758070", 10, 16, false, Some(0)).unwrap(),
            Some("FFFFFFFFFFFFFFFF".to_owned())
        );
    }

    #[test]
    fn negative_values_with_any_base_sign() {
        // Negative with a positive base formats the unsigned magnitude
        // (Hive: non-negative bases are unsigned).
        assert_eq!(
            convert_one("-10", 16, 10, true, Some(0)).unwrap(),
            Some("18446744073709551600".to_owned())
        );
        // A top-bit magnitude with a negative base stays signed.
        assert_eq!(
            convert_one("-9223372036854775808", 10, -10, true, Some(0)).unwrap(),
            Some("-9223372036854775808".to_owned())
        );
        // A bare sign is zero.
        assert_eq!(
            convert_one("-", 10, 10, true, Some(0)).unwrap(),
            Some("0".to_owned())
        );
    }

    #[test]
    fn length_mismatches_are_internal_errors() {
        use arrow_array::{Int32Array, StringArray};
        let nums = StringArray::from(vec!["1", "2"]);
        let one = Int32Array::from(vec![10]);
        let err = conv(&nums, &one, &one, true).unwrap_err();
        assert_eq!(
            err.error_class(),
            krishiv_common::error::ErrorClass::InternalError
        );
    }
}
