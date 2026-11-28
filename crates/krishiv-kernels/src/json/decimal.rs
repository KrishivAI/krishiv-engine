// SPDX-License-Identifier: Apache-2.0
//! The decimal arms of Spark's JSON schema inference (`JsonInferSchema`),
//! shared by `schema_of_json` and the JSON file reader.
//!
//! - A JSON string under `prefersDecimal` is `new BigDecimal(s.replaceAll(",", ""))`
//!   (the US-locale decimal parser) typed `DecimalType(precision, scale)`:
//!   any failure, including a scale that is negative or above the precision
//!   and a precision past 38, leaves the string to the other arms
//!   ([`string_decimal`]).
//! - A number past `Long`, or a floating-point number under `prefersDecimal`,
//!   is `DecimalType(max(precision, scale), scale)`, `DOUBLE` past 38 digits;
//!   a negative scale (`1e3`) fails the query with `NEGATIVE_SCALE_DISALLOWED`,
//!   and a number with no decimal form (`NaN`, the infinities) is Jackson's
//!   `getDecimalValue` failure, `KRISHIV_JSON_NUMBER_NOT_DECIMAL`
//!   ([`number_decimal`]).
//! - Two decimals merge to the wider integral range plus the wider scale,
//!   `DOUBLE` past 38 digits; a `BIGINT` merges as `DECIMAL(20, 0)`
//!   ([`merge_decimals`], [`LONG_DECIMAL`]).
//!
//! Goldens: `tests/fixtures/spark-{4.2.1,4.3.0}/json-prefers-decimal.json`.
//! Known gap: a `locale` other than `en-US` reads through Java's
//! `DecimalFormat` in Spark (`1.000,5` is `DECIMAL(5,1)` under `de-DE`); the
//! US rule applies here whatever the locale.

use std::borrow::Cow;

use krishiv_common::error::{KrishivError, KrishivJsonNumberNotDecimal, NegativeScaleDisallowed};

/// `DecimalType.forType(LongType)`: the decimal a `BIGINT` merges as.
pub const LONG_DECIMAL: (u8, i8) = (20, 0);

/// The largest decimal precision (`DecimalType.MAX_PRECISION`).
const MAX_PRECISION: i64 = 38;

/// `new java.math.BigDecimal(s)`: the precision and scale of a decimal
/// literal, or `None` where Java throws `NumberFormatException`.
///
/// The grammar is Java's: an optional sign, digits with at most one `.` and
/// at least one digit, then an optional `e`/`E` exponent with an optional
/// sign and at least one digit that fits an `int`; no blanks. Digits are
/// `Character.digit(c, 10)` (ASCII and the other basic-multilingual-plane
/// `Nd` digits). Leading zeros do not count toward the precision, a zero
/// has precision 1, and the scale (fraction digits minus the exponent) must
/// fit an `int`.
#[must_use]
pub fn java_big_decimal(s: &str) -> Option<(i64, i64)> {
    let mut chars = s.chars().peekable();
    if matches!(chars.peek(), Some('+' | '-')) {
        chars.next();
    }
    let (mut significant, mut digits, mut frac) = (0_i64, 0_i64, 0_i64);
    let mut dot = false;
    let mut exp = 0_i64;
    while let Some(c) = chars.next() {
        if let Some(d) = crate::string::java_decimal_digit(c) {
            digits = digits.checked_add(1)?;
            if significant > 0 || d != 0 {
                significant = significant.checked_add(1)?;
            }
            if dot {
                frac = frac.checked_add(1)?;
            }
        } else if c == '.' && !dot {
            dot = true;
        } else if matches!(c, 'e' | 'E') && digits > 0 {
            exp = java_exponent(&mut chars)?;
            break;
        } else {
            return None;
        }
    }
    if digits == 0 {
        return None;
    }
    let scale = frac.checked_sub(exp)?;
    i32::try_from(scale).ok()?;
    Some((significant.max(1), scale))
}

/// The exponent after `e`/`E`: an optional sign and at least one digit, the
/// whole rest of the text, within an `int`.
fn java_exponent(chars: &mut impl Iterator<Item = char>) -> Option<i64> {
    let mut neg = false;
    let mut value = 0_i64;
    let mut any = false;
    for (i, c) in chars.enumerate() {
        if i == 0 && matches!(c, '+' | '-') {
            neg = c == '-';
            continue;
        }
        let d = crate::string::java_decimal_digit(c)?;
        value = value.checked_mul(10)?.checked_add(i64::from(d))?;
        // Past `int` in either sign: Java throws. `2^31` itself stays (it is
        // `-2^31` when negated); the final `i32::try_from` refuses `+2^31`,
        // so this early exit only keeps `value` from overflowing.
        if value > 1_i64 << 31 {
            return None;
        }
        any = true;
    }
    if !any {
        return None;
    }
    let exp = if neg { value.checked_neg()? } else { value };
    i32::try_from(exp).ok().map(i64::from)
}

/// Spark's `prefersDecimal` string arm: `new BigDecimal(s.replaceAll(",", ""))`
/// as `DecimalType(precision, scale)`, `None` where Java or `DecimalType`
/// throws (not a decimal, a negative scale, a scale above the precision, a
/// precision past 38).
#[must_use]
pub fn string_decimal(s: &str) -> Option<(u8, i8)> {
    let text: Cow<'_, str> = if s.contains(',') {
        Cow::Owned(s.replace(',', ""))
    } else {
        Cow::Borrowed(s)
    };
    let (p, scale) = java_big_decimal(&text)?;
    if scale < 0 || scale > p || p > MAX_PRECISION {
        return None;
    }
    Some((u8::try_from(p).ok()?, i8::try_from(scale).ok()?))
}

/// The decimal a JSON number infers (a number past `Long`, or a
/// floating-point number under `prefersDecimal`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberDecimal {
    /// `DECIMAL(p, s)`.
    Decimal(u8, i8),
    /// Wider than 38 digits.
    Double,
}

/// Spark's number arm: `DecimalType(max(precision, scale), scale)` of
/// Jackson's `getDecimalValue`, `DOUBLE` past 38 digits.
///
/// # Errors
/// `NEGATIVE_SCALE_DISALLOWED` for a negative scale within 38 digits
/// (`1e3` under `prefersDecimal`), as `DecimalType` refuses it, and
/// `KRISHIV_JSON_NUMBER_NOT_DECIMAL` for a number with no decimal form
/// (`NaN`, `Infinity`, `-Infinity`, an exponent past `int`), where Spark's
/// `getDecimalValue` throws a `NumberFormatException` (Spark 4.2.1 / 4.3.0
/// runs: `schema_of_json` fails; a file read takes the record to the parse
/// mode, [`is_not_decimal`]).
pub fn number_decimal(raw: &str) -> Result<NumberDecimal, KrishivError> {
    let Some((p, scale)) = java_big_decimal(raw) else {
        return Err(KrishivJsonNumberNotDecimal {
            value: format!("\"{raw}\""),
        }
        .into());
    };
    // `DOUBLE` past 38 digits (a width that does not fit a `u8` is past them).
    let Some(width) = u8::try_from(p.max(scale))
        .ok()
        .filter(|w| i64::from(*w) <= MAX_PRECISION)
    else {
        return Ok(NumberDecimal::Double);
    };
    if scale < 0 {
        return Err(NegativeScaleDisallowed {
            scale: scale.to_string(),
            sql_conf: "\"spark.sql.legacy.allowNegativeScaleOfDecimal\"".to_owned(),
        }
        .into());
    }
    // `0 <= scale <= width <= 38`, so the scale fits an `i8`.
    Ok(i8::try_from(scale).map_or(NumberDecimal::Double, |s| NumberDecimal::Decimal(width, s)))
}

/// Whether `e` is [`number_decimal`]'s "no decimal form" failure, a
/// `RuntimeException` that Spark's file inference hands to the parse mode
/// (unlike `NEGATIVE_SCALE_DISALLOWED`, which fails the read).
#[must_use]
pub fn is_not_decimal(e: &KrishivError) -> bool {
    e.error_class() == krishiv_common::error::ErrorClass::KrishivJsonNumberNotDecimal
}

/// Spark's `compatibleType` of two decimals: the wider integral range plus
/// the wider scale, `None` (`DOUBLE`) past 38 digits.
#[must_use]
pub fn merge_decimals(a: (u8, i8), b: (u8, i8)) -> Option<(u8, i8)> {
    let scale = a.1.max(b.1);
    let range_of = |(p, s): (u8, i8)| i16::from(p).saturating_sub(i16::from(s));
    let p = range_of(a)
        .max(range_of(b))
        .saturating_add(i16::from(scale));
    if p > 38 {
        return None;
    }
    Some((u8::try_from(p).ok()?, scale))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn big_decimal_reads_java_grammar() {
        let cases: [(&str, Option<(i64, i64)>); 22] = [
            ("1.5", Some((2, 1))),
            ("0.01", Some((1, 2))),
            ("0.00", Some((1, 2))),
            ("-0.0", Some((1, 1))),
            ("+1.5", Some((2, 1))),
            (".5", Some((1, 1))),
            ("5.", Some((1, 0))),
            ("007.50", Some((3, 2))),
            ("1e3", Some((1, -3))),
            ("1.5e1", Some((2, 0))),
            ("15e-1", Some((2, 1))),
            ("1E+2", Some((1, -2))),
            ("\u{0661}\u{0662}", Some((2, 0))),
            ("1e2147483647", Some((1, -2_147_483_647))),
            ("1e2147483648", None),
            ("1e", None),
            (".", None),
            ("+", None),
            ("e5", None),
            ("1.5 ", None),
            ("1.2.3", None),
            ("NaN", None),
        ];
        for (text, want) in cases {
            assert_eq!(java_big_decimal(text), want, "{text}");
        }
    }

    #[test]
    fn string_decimal_strips_commas_and_needs_a_valid_type() {
        assert_eq!(string_decimal("1,000.5"), Some((5, 1)));
        assert_eq!(string_decimal("1,5"), Some((2, 0)));
        assert_eq!(string_decimal(","), None);
        assert_eq!(string_decimal("0.01"), None);
        assert_eq!(string_decimal("1e3"), None);
        assert_eq!(string_decimal("0.5"), Some((1, 1)));
        assert_eq!(string_decimal(&"9".repeat(38)), Some((38, 0)));
        assert_eq!(string_decimal(&"9".repeat(39)), None);
        // An exponent past `int` is Java's `NumberFormatException`: the text
        // stays a string (Spark 4.2.1 / 4.3.0 runs: `{"a":"1e3000000000"}`
        // under prefersDecimal reads `struct<a:string>`).
        assert_eq!(java_big_decimal("1e3000000000"), None);
        assert_eq!(string_decimal("1e3000000000"), None);
    }

    #[test]
    fn number_decimal_widens_to_the_scale_and_refuses_negative_scales() {
        assert_eq!(
            number_decimal("0.01").ok(),
            Some(NumberDecimal::Decimal(2, 2))
        );
        assert_eq!(
            number_decimal("1.5e1").ok(),
            Some(NumberDecimal::Decimal(2, 0))
        );
        for raw in ["NaN", "Infinity", "-Infinity", "1e3000000000"] {
            let err = number_decimal(raw).err();
            assert!(err.as_ref().is_some_and(is_not_decimal), "{raw}: {err:?}");
            assert_eq!(
                err.map(|e| e.error_class()),
                Some(krishiv_common::error::ErrorClass::KrishivJsonNumberNotDecimal),
                "{raw}"
            );
        }
        assert_eq!(
            number_decimal(&"9".repeat(39)).ok(),
            Some(NumberDecimal::Double)
        );
        assert_eq!(number_decimal("1e-39").ok(), Some(NumberDecimal::Double));
        let err = number_decimal("1e3").err().map(|e| e.to_string());
        assert!(
            err.as_deref()
                .is_some_and(|m| m.contains("NEGATIVE_SCALE_DISALLOWED")),
            "{err:?}"
        );
    }

    #[test]
    fn decimals_merge_by_range_and_scale() {
        assert_eq!(merge_decimals((6, 1), (4, 3)), Some((8, 3)));
        assert_eq!(merge_decimals(LONG_DECIMAL, (2, 1)), Some((21, 1)));
        assert_eq!(merge_decimals(LONG_DECIMAL, (19, 0)), Some((20, 0)));
        assert_eq!(merge_decimals((38, 38), (11, 1)), None);
        assert_eq!(merge_decimals((38, 37), (2, 1)), Some((38, 37)));
    }
}
