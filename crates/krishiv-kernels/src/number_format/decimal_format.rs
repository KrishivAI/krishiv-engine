// SPDX-License-Identifier: Apache-2.0
//! A subset of Java's `DecimalFormat` for `format_number` with a string
//! pattern (Spark's `FormatNumber` with a `STRING` second argument).
//!
//! Supported: the digit placeholders `0` and `#`, the decimal separator
//! `.`, grouping `,`, positive/negative subpatterns (`;`), the multipliers
//! `%`/`‰`, single-quoted literals and any other character as a literal
//! affix. `NaN` formats as `NaN`, infinities as `∞` (with the pattern's
//! affixes). Rounding is half-even, like `DecimalFormat`.
//!
//! Not supported (a [`PatternError`], mapped to
//! `KRISHIV_UNSUPPORTED_FEATURE` like an unimplemented regex delimiter):
//! the exponent `E`, the currency sign `¤`, significant digits `@`, the pad
//! escape `*`, quoted text inside the number part and a second `.`. There
//! is no golden coverage for these; Spark formats them.

#![expect(
    clippy::arithmetic_side_effects,
    reason = "digit-string indexes are bounded by the value and pattern lengths; \
              shifts and roundings use checked or saturating steps"
)]

use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};

/// A value to format.
#[derive(Debug, Clone, Copy)]
pub enum Value {
    /// An integral value.
    Int(i64),
    /// A binary floating-point value (formatted from its shortest
    /// decimal text, as Java formats the `double`).
    Float(f64),
    /// A decimal value: `unscaled * 10^-scale`.
    Decimal {
        /// The unscaled value.
        unscaled: i128,
        /// The scale.
        scale: u32,
    },
}

/// The pattern is not in the supported subset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternError {
    pattern: String,
}

impl PatternError {
    /// `KRISHIV_UNSUPPORTED_FEATURE` for the pattern.
    #[must_use]
    pub fn into_error(self) -> KrishivError {
        KrishivUnsupportedFeature {
            feature: format!("format_number with the pattern {:?}", self.pattern),
        }
        .into()
    }
}

/// A parsed subpattern (one side of `;`).
#[derive(Debug, Clone)]
struct Subpattern {
    prefix: String,
    min_int: usize,
    grouping: usize,
    min_frac: usize,
    max_frac: usize,
    suffix: String,
    /// A decimal-point shift: 2 for `%`, 3 for `‰`.
    shift: u32,
}

/// One character of a pattern: literal text or a placeholder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Char {
    Literal(char),
    Quoted(char),
    Zero,
    Hash,
    Point,
    Comma,
}

/// Splits a pattern side into characters, resolving quotes (`'x'` is
/// literal text, `''` one quote). An unterminated quote is an error.
fn lex(side: &str) -> Result<Vec<Char>, ()> {
    let mut out = Vec::new();
    let mut chars = side.chars().peekable();
    let mut in_quote = false;
    while let Some(c) = chars.next() {
        if c == '\'' {
            if in_quote && chars.peek() == Some(&'\'') {
                out.push(Char::Quoted('\''));
                chars.next();
            } else {
                in_quote = !in_quote;
            }
            continue;
        }
        if in_quote {
            out.push(Char::Quoted(c));
            continue;
        }
        out.push(match c {
            '0' => Char::Zero,
            '#' => Char::Hash,
            '.' => Char::Point,
            ',' => Char::Comma,
            _ => Char::Literal(c),
        });
    }
    if in_quote { Err(()) } else { Ok(out) }
}

/// Parses one side of `;`.
fn parse_side(side: &str) -> Result<Subpattern, PatternError> {
    let bad = || PatternError {
        pattern: side.to_owned(),
    };
    let chars = lex(side).map_err(|()| bad())?;
    let is_number = |c: &Char| matches!(c, Char::Zero | Char::Hash | Char::Point | Char::Comma);
    let first = chars.iter().position(is_number);
    let last = chars.iter().rposition(is_number);
    let (Some(first), Some(last)) = (first, last) else {
        // No number part: the whole side is a prefix (Spark formats
        // `format_number(12332.123456, 'abc')` as `abc12332`).
        let mut prefix = String::new();
        for c in &chars {
            match c {
                Char::Literal(c) | Char::Quoted(c) => prefix.push(*c),
                _ => return Err(bad()),
            }
        }
        return Ok(Subpattern {
            prefix,
            min_int: 0,
            grouping: 0,
            min_frac: 0,
            max_frac: 0,
            suffix: String::new(),
            shift: shift_of(&chars),
        });
    };
    // Quoted text cannot sit inside the number part.
    if chars[first..=last]
        .iter()
        .any(|c| matches!(c, Char::Literal(_) | Char::Quoted(_)))
    {
        // ... except the unquoted multipliers, which stay literal and
        // multiply (a quoted `'%'` is plain text, without a multiplier).
        let inner: Vec<char> = chars[first..=last]
            .iter()
            .filter_map(|c| match c {
                Char::Literal('%' | '‰') => None,
                Char::Literal(c) | Char::Quoted(c) => Some(*c),
                _ => None,
            })
            .collect();
        if !inner.is_empty() {
            return Err(bad());
        }
    }
    let mut prefix = String::new();
    let mut suffix = String::new();
    for (i, c) in chars.iter().enumerate() {
        match c {
            Char::Literal(c) => {
                if i < first {
                    prefix.push(*c);
                } else if i > last || *c == '%' || *c == '‰' {
                    // Multipliers stay literal in the affix, wherever they are.
                    suffix.push(*c);
                }
            }
            Char::Quoted(c) => {
                if i < first {
                    prefix.push(*c);
                } else if i > last {
                    suffix.push(*c);
                }
            }
            _ => {}
        }
    }
    let (mut min_int, mut min_frac, mut max_frac) = (0_usize, 0_usize, 0_usize);
    let mut max_int = 0_usize;
    let mut seen_point = false;
    let mut last_comma = None;
    for c in &chars[first..=last] {
        match c {
            Char::Zero if !seen_point => {
                min_int += 1;
                max_int += 1;
            }
            Char::Hash if !seen_point => max_int += 1,
            Char::Zero => {
                min_frac += 1;
                max_frac += 1;
            }
            Char::Hash => max_frac += 1,
            Char::Point if !seen_point => seen_point = true,
            Char::Comma => last_comma = Some(max_int),
            Char::Literal('%' | '‰') => {}
            _ => return Err(bad()),
        }
    }
    let grouping = match last_comma {
        Some(at) if max_int > at => max_int - at,
        _ => 0,
    };
    Ok(Subpattern {
        prefix,
        min_int,
        grouping,
        min_frac,
        max_frac,
        suffix,
        shift: shift_of(&chars),
    })
}

/// Whether the side multiplies by 100 (`%`) or 1000 (`‰`): the last such
/// sign wins, and it stays literal in the affix.
fn shift_of(chars: &[Char]) -> u32 {
    let mut shift = 0;
    for c in chars {
        match c {
            Char::Literal('%') => shift = 2,
            Char::Literal('‰') => shift = 3,
            _ => {}
        }
    }
    shift
}

/// The shortest decimal digits of a finite `v`: `(digits, exponent)` with
/// value `0.digits * 10^exponent` and no leading or trailing zeros
/// (`"0"` and `1` for zero).
fn shortest_digits(v: f64) -> (String, i32) {
    let text = format!("{v:?}");
    let lower = text.to_ascii_lowercase();
    let (mantissa, exp) = match lower.split_once('e') {
        Some((m, e)) => (m, e.parse().unwrap_or(0)),
        None => (lower.as_str(), 0),
    };
    let point = mantissa.find('.').unwrap_or(mantissa.len());
    let int_len = i32::try_from(
        mantissa[..point]
            .chars()
            .filter(char::is_ascii_digit)
            .count(),
    )
    .unwrap_or(i32::MAX);
    let stripped: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let leading =
        i32::try_from(stripped.chars().take_while(|c| *c == '0').count()).unwrap_or(i32::MAX);
    let digits = stripped.trim_matches('0');
    if digits.is_empty() {
        ("0".to_owned(), 1)
    } else {
        (digits.to_owned(), exp + int_len - leading)
    }
}

/// Whether a shortest-digit tie (`5` followed by zeros) rounds up, from
/// the exact finite `f64` value: above the tie rounds up, below rounds
/// down, exactly on the tie half-even on the kept digit.
fn exact_tie_up(abs: f64, shift: u32, max_frac: usize, int: &str, keep: &str) -> bool {
    // The exact decimal terminates within 1074 fractional digits (the
    // smallest subnormal is `2^-1074`).
    let text = format!("{abs:.1074}");
    let frac_part = text.split_once('.').map_or("", |(_, f)| f);
    let shift = usize::try_from(shift).unwrap_or(usize::MAX);
    let (first, rest) = match frac_part.get(shift..) {
        Some(scaled) if !scaled.is_empty() => (
            scaled.chars().nth(max_frac).unwrap_or('0'),
            scaled.chars().skip(max_frac + 1).any(|c| c != '0'),
        ),
        _ => ('0', false),
    };
    if first > '5' {
        return true;
    }
    if first < '5' {
        return false;
    }
    if rest {
        return true;
    }
    keep.chars()
        .last()
        .or_else(|| int.chars().last())
        .is_some_and(|c| c.to_digit(10).is_some_and(|d| d % 2 == 1))
}

/// Formats `value` with `pattern`.
///
/// # Errors
/// [`PatternError`] for patterns outside the supported subset.
pub fn format(pattern: &str, value: Value) -> Result<String, PatternError> {
    let bad = || PatternError {
        pattern: pattern.to_owned(),
    };
    let parts: Vec<&str> = pattern.split(';').collect();
    if parts.len() > 2 {
        return Err(bad());
    }
    let positive = parse_side(parts[0])?;
    let negative = parts.get(1).map(|p| parse_side(p)).transpose()?;
    let (neg, digits, exp, float_abs) = match value {
        Value::Int(v) => {
            let (neg, abs) = if v < 0 {
                (true, v.unsigned_abs())
            } else {
                (false, v.cast_unsigned())
            };
            let digits = abs.to_string();
            (
                neg,
                digits.clone(),
                i32::try_from(digits.len()).unwrap_or(i32::MAX),
                None,
            )
        }
        Value::Float(v) => {
            if v.is_nan() {
                return Ok("NaN".to_owned());
            }
            if v.is_infinite() {
                if v < 0.0 {
                    return Ok(negative.as_ref().map_or_else(
                        || format!("-{}∞{}", positive.prefix, positive.suffix),
                        |n| format!("{}∞{}", n.prefix, n.suffix),
                    ));
                }
                return Ok(format!("{}∞{}", positive.prefix, positive.suffix));
            }
            let (neg, abs) = if v < 0.0 { (true, -v) } else { (false, v) };
            let (digits, exp) = shortest_digits(abs);
            (neg, digits, exp, Some(abs))
        }
        Value::Decimal { unscaled, scale } => {
            let (neg, abs) = if unscaled < 0 {
                (true, unscaled.unsigned_abs())
            } else {
                (false, unscaled.cast_unsigned())
            };
            let digits = abs.to_string();
            (
                neg,
                digits.clone(),
                i32::try_from(digits.len()).unwrap_or(i32::MAX)
                    - i32::try_from(scale).unwrap_or(i32::MAX),
                None,
            )
        }
    };
    let sub = if neg {
        negative.as_ref().unwrap_or(&positive)
    } else {
        &positive
    };
    // value = 0.digits * 10^exp, shifted for `%`/`‰`.
    let exp = exp + i32::try_from(sub.shift).unwrap_or(0);
    let (mut int, mut frac) = if exp <= 0 {
        (
            "0".to_owned(),
            format!(
                "{}{digits}",
                "0".repeat(usize::try_from(exp.unsigned_abs()).unwrap_or(0))
            ),
        )
    } else if usize::try_from(exp).unwrap_or(usize::MAX) >= digits.len() {
        (
            format!(
                "{digits}{}",
                "0".repeat(
                    usize::try_from(exp)
                        .unwrap_or(0)
                        .saturating_sub(digits.len())
                )
            ),
            String::new(),
        )
    } else {
        let (i, f) = digits.split_at(usize::try_from(exp).unwrap_or(0));
        (i.to_owned(), f.to_owned())
    };
    // Round the fraction half-even to max_frac digits.
    if frac.len() > sub.max_frac {
        let (keep, drop) = frac.split_at(sub.max_frac);
        let mut dropped = drop.chars();
        let first = dropped.next();
        let rest_nonzero = dropped.any(|c| c != '0');
        // The kept digit for half-even ties: the last kept fraction
        // digit, or the last integer digit when nothing is kept.
        let parity_odd = keep
            .chars()
            .last()
            .or_else(|| int.chars().last())
            .is_some_and(|c| c.to_digit(10).is_some_and(|d| d % 2 == 1));
        let up = match first {
            Some(c) if c > '5' => true,
            Some('5') if rest_nonzero => true,
            Some('5') => match float_abs {
                // The shortest digits can show a tie the exact value
                // is not on (`2.675` is `2.67499...`): decide from the
                // exact decimal expansion.
                Some(abs) => exact_tie_up(abs, sub.shift, sub.max_frac, &int, keep),
                None => parity_odd,
            },
            _ => false,
        };
        frac = keep.to_owned();
        if up {
            (int, frac) = round_up(&int, &frac);
        }
    }
    // Optional fraction digits (`#`) drop trailing zeros, down to the
    // required (`0`) ones.
    while frac.len() > sub.min_frac && frac.ends_with('0') {
        frac.pop();
    }
    while frac.len() < sub.min_frac {
        frac.push('0');
    }
    while int.len() < sub.min_int.max(1) {
        int = format!("0{int}");
    }
    if sub.grouping > 0 && int.len() > sub.grouping {
        let mut grouped = String::new();
        for (k, c) in int.chars().rev().enumerate() {
            if k > 0 && k % sub.grouping == 0 {
                grouped.push(',');
            }
            grouped.push(c);
        }
        int = grouped.chars().rev().collect();
    }
    let mut out = String::new();
    out.push_str(&sub.prefix);
    out.push_str(&int);
    // A bare decimal point is never shown.
    if !frac.is_empty() {
        out.push('.');
        out.push_str(&frac);
    }
    out.push_str(&sub.suffix);
    if neg && negative.is_none() {
        out = format!("-{out}");
    }
    Ok(out)
}

/// Adds one to the combined `int.frac` digits, carrying as needed.
fn round_up(int: &str, frac: &str) -> (String, String) {
    let mut digits: Vec<u8> = int.bytes().chain(frac.bytes()).map(|b| b - b'0').collect();
    let mut k = digits.len();
    loop {
        if k == 0 {
            digits.insert(0, 1);
            break;
        }
        k -= 1;
        if digits[k] < 9 {
            digits[k] += 1;
            break;
        }
        digits[k] = 0;
    }
    let f = frac.len();
    let total: String = digits.iter().map(|d| (b'0' + d) as char).collect();
    if total.len() <= f {
        ("0".to_owned(), format!("{:0>width$}", total, width = f))
    } else {
        let at = total.len() - f;
        (total[..at].to_owned(), total[at..].to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::{Value, format};

    #[test]
    fn quoted_literals_are_affixes() {
        assert_eq!(format("'USD'#,##0", Value::Int(5)).unwrap(), "USD5");
        // A quoted `'%'` is plain text: no multiplier.
        assert_eq!(format("#'%'", Value::Int(5)).unwrap(), "5%");
        // An unquoted `#` cannot appear in the suffix.
        assert!(format("#'%'#", Value::Int(5)).is_err());
    }

    #[test]
    fn bad_patterns_refuse() {
        assert!(format("1.2.3", Value::Int(5)).is_err());
        assert!(format("1;2;3", Value::Int(5)).is_err());
        assert!(format("0.0E0", Value::Int(5)).is_err());
    }

    #[test]
    fn non_finite_values() {
        assert_eq!(format("#,##0", Value::Float(f64::NAN)).unwrap(), "NaN");
        // Java returns the bare `NaN` symbol, without the pattern's affixes.
        assert_eq!(format("'$'0", Value::Float(f64::NAN)).unwrap(), "NaN");
        assert_eq!(format("#,##0", Value::Float(f64::INFINITY)).unwrap(), "∞");
        assert_eq!(
            format("#,##0", Value::Float(f64::NEG_INFINITY)).unwrap(),
            "-∞"
        );
        assert_eq!(format("#,##0.00", Value::Float(0.0)).unwrap(), "0.00");
    }

    #[test]
    fn fractions_below_one() {
        assert_eq!(format("0.00", Value::Float(0.05)).unwrap(), "0.05");
        assert_eq!(format("0.00", Value::Float(0.5)).unwrap(), "0.50");
    }

    #[test]
    fn half_even_ties_and_carries() {
        // 5th digit 5, nothing after, last kept even: down.
        assert_eq!(
            format("#,###,###,###,###,###,##0.0000", Value::Float(12332.12345)).unwrap(),
            "12,332.1234"
        );
        // 9.999 to two places carries into the integer part.
        assert_eq!(format("#.##", Value::Float(9.999)).unwrap(), "10");
        // Optional fraction digits drop trailing zeros and a bare point.
        assert_eq!(format("#.##", Value::Float(2.0)).unwrap(), "2");
        assert_eq!(format("#.0#", Value::Float(9.999)).unwrap(), "10.0");
        // Minimum integer digits pad with zeros.
        assert_eq!(format("000", Value::Int(5)).unwrap(), "005");
        // The shortest digits show ties the exact value is just below:
        // `2.675` is `2.67499...`, `0.8055` is `0.80549...`.
        assert_eq!(format("0.00", Value::Float(2.675)).unwrap(), "2.67");
        assert_eq!(format("0.000", Value::Float(0.8055)).unwrap(), "0.805");
    }

    #[test]
    fn shortest_exponents_expand() {
        assert_eq!(
            format("#", Value::Float(1e21)).unwrap(),
            "1000000000000000000000"
        );
    }

    #[test]
    fn negative_decimals() {
        assert_eq!(
            format(
                "#,##0.00",
                Value::Decimal {
                    unscaled: -500,
                    scale: 2
                }
            )
            .unwrap(),
            "-5.00"
        );
    }
}

#[cfg(test)]
mod reach_tests {
    use super::{Value, format};

    #[test]
    fn per_mille_multiplies() {
        assert_eq!(format("#‰#", Value::Int(5)).unwrap(), "5000‰");
    }

    #[test]
    fn negative_infinity_uses_the_negative_pattern() {
        assert_eq!(
            format("0;-0", Value::Float(f64::NEG_INFINITY)).unwrap(),
            "-∞"
        );
    }
}
