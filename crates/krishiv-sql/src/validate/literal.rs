// SPDX-License-Identifier: Apache-2.0
//! Literal values as Spark's `AstBuilder` reads them: numeric literals
//! with their range and precision checks, and the typed literals `DATE`,
//! `TIME`, `TIMESTAMP[_NTZ|_LTZ]` and `X` (`INTERVAL` is in
//! [`super::interval`]). Lowering reads values through the same functions,
//! so a literal the checks accept always lowers.

use krishiv_common::error::{
    DecimalPrecisionExceedsMaxPrecision, InvalidNumericLiteralRange, InvalidTypedLiteral,
    KrishivError, UnsupportedTypedLiteral,
};
use krishiv_kernels::cast::{TimestampText, timestamp_text};

use crate::ast::NumKind;

/// The largest decimal precision.
pub(crate) const MAX_PRECISION: u32 = 38;

/// `Float.MaxValue` as an exact integer, and as Spark prints it.
const FLOAT_MAX: (&str, &str) = (
    "340282346638528859811704183484516925440",
    "3.4028234663852886E+38",
);

/// `Double.MaxValue` as an exact integer, and as Spark prints it.
const DOUBLE_MAX: (&str, &str) = (
    "17976931348623157081452742373170435679807056752584499659891747680315726078002853876058955\
     86327668781715404589535143824642343213268894641827684675467035375169860499105765512820762\
     45490090389328944075868508455133942304583236903222948165808559332123348274797826204144723\
     168738177180919299881250404026184124858368",
    "1.7976931348623157E+308",
);

/// The types Spark's typed literals take, as its error lists them.
const TYPED_LITERAL_TYPES: &str =
    "\"DATE\", \"TIMESTAMP_NTZ\", \"TIMESTAMP_LTZ\", \"TIMESTAMP\", \"INTERVAL\", \"X\", \"TIME\"";

/// The value of a numeric literal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Number {
    Byte(i8),
    Short(i16),
    Integer(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    /// A decimal: its unscaled value, precision and scale.
    Decimal {
        unscaled: i128,
        precision: u32,
        scale: u32,
    },
}

/// A numeric literal's value, as Spark's `AstBuilder` types it: `text` is
/// its digits without the suffix, with a leading `-` if negated.
///
/// # Errors
/// `INVALID_NUMERIC_LITERAL_RANGE` for a suffixed literal outside its type,
/// and `DECIMAL_PRECISION_EXCEEDS_MAX_PRECISION` for a decimal of more than
/// 38 digits.
pub(crate) fn number(text: &str, kind: NumKind) -> Result<Number, KrishivError> {
    let range = |ty: &str, min: String, max: String| -> KrishivError {
        InvalidNumericLiteralRange {
            raw_stripped_qualifier: text.to_owned(),
            type_name: ty.to_owned(),
            min_value: min,
            max_value: max,
        }
        .into()
    };
    Ok(match kind {
        NumKind::Int => match (text.parse::<i32>(), text.parse::<i64>()) {
            (Ok(v), _) => Number::Integer(v),
            (_, Ok(v)) => Number::Long(v),
            _ => decimal(text)?,
        },
        NumKind::Long => Number::Long(
            text.parse()
                .map_err(|_| range("bigint", i64::MIN.to_string(), i64::MAX.to_string()))?,
        ),
        NumKind::Short => Number::Short(
            text.parse()
                .map_err(|_| range("smallint", i16::MIN.to_string(), i16::MAX.to_string()))?,
        ),
        NumKind::Byte => Number::Byte(
            text.parse()
                .map_err(|_| range("tinyint", i8::MIN.to_string(), i8::MAX.to_string()))?,
        ),
        NumKind::Decimal | NumKind::BigDecimal => decimal(text)?,
        NumKind::Float => {
            if exceeds(text, FLOAT_MAX.0) {
                return Err(range(
                    "float",
                    format!("-{}", FLOAT_MAX.1),
                    FLOAT_MAX.1.to_owned(),
                ));
            }
            Number::Float(text.parse().unwrap_or(0.0))
        }
        NumKind::Exponent | NumKind::Double => {
            if exceeds(text, DOUBLE_MAX.0) {
                return Err(range(
                    "double",
                    format!("-{}", DOUBLE_MAX.1),
                    DOUBLE_MAX.1.to_owned(),
                ));
            }
            Number::Double(text.parse().unwrap_or(0.0))
        }
    })
}

/// The digits of a decimal number's text and the power of ten they are
/// scaled by: `-1.20E-38` is `("120", -40)`. Leading zeros are dropped;
/// an all-zero value has no digits.
fn digits_and_exponent(text: &str) -> (String, i64) {
    let body = text.trim_start_matches(['-', '+']);
    let (mantissa, exp) = match body.find(['e', 'E']) {
        Some(i) => (&body[..i], body.get(i + 1..).unwrap_or_default()),
        None => (body, ""),
    };
    // An exponent too large for `i64` saturates: the value is then out of
    // any range.
    let exp = exp.parse::<i64>().unwrap_or_else(|_| {
        if exp.is_empty() {
            0
        } else if exp.starts_with('-') {
            i64::MIN / 2
        } else {
            i64::MAX / 2
        }
    });
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{int}{frac}");
    let frac_len = i64::try_from(frac.len()).unwrap_or(i64::MAX / 2);
    (
        digits.trim_start_matches('0').to_owned(),
        exp.saturating_sub(frac_len),
    )
}

/// Whether the absolute value of the number `text` is above the integer
/// `max`, compared exactly.
fn exceeds(text: &str, max: &str) -> bool {
    let (digits, exp) = digits_and_exponent(text);
    if digits.bytes().all(|b| b == b'0') {
        return false;
    }
    // The value is `digits × 10^exp`: its integer part has this many digits.
    let int_len = i64::try_from(digits.len())
        .unwrap_or(i64::MAX / 2)
        .saturating_add(exp);
    let max_len = i64::try_from(max.len()).unwrap_or(i64::MAX);
    if int_len != max_len {
        return int_len > max_len;
    }
    // As many integer digits as `max`: compare them, then any nonzero
    // fraction left over makes the value larger.
    let width = max.len();
    let int: String = digits
        .chars()
        .chain(std::iter::repeat('0'))
        .take(width)
        .collect();
    match int.as_str().cmp(max) {
        std::cmp::Ordering::Greater => true,
        std::cmp::Ordering::Less => false,
        std::cmp::Ordering::Equal => digits.bytes().skip(width).any(|b| b != b'0'),
    }
}

/// A decimal literal (`1.5`, `1.20E-38BD`, an integer too large for
/// `BIGINT`) as Java's `BigDecimal` reads it, with a negative scale raised
/// to zero: precision is its significant digits, and at least its scale.
fn decimal(text: &str) -> Result<Number, KrishivError> {
    let (digits, exp) = digits_and_exponent(text);
    // A negative scale (a positive exponent) is set to 0: the digits gain
    // that many zeros.
    let (zeros, scale) = if exp > 0 {
        (exp, 0)
    } else {
        (0, exp.unsigned_abs())
    };
    let significant = u64::try_from(digits.len()).unwrap_or(u64::MAX).max(1);
    let precision = significant
        .saturating_add(u64::try_from(zeros).unwrap_or(u64::MAX))
        .max(scale);
    if precision > u64::from(MAX_PRECISION) {
        return Err(DecimalPrecisionExceedsMaxPrecision {
            precision: precision.to_string(),
            max_precision: MAX_PRECISION.to_string(),
        }
        .into());
    }
    // Within 38 digits, so the narrowing and the value fit.
    let (precision, scale) = (
        u32::try_from(precision).unwrap_or(MAX_PRECISION),
        u32::try_from(scale).unwrap_or(MAX_PRECISION),
    );
    let mut unscaled: i128 = if digits.is_empty() {
        0
    } else {
        digits.parse().unwrap_or(0)
    };
    for _ in 0..zeros {
        unscaled = unscaled.saturating_mul(10);
    }
    if text.starts_with('-') {
        unscaled = -unscaled;
    }
    Ok(Number::Decimal {
        unscaled,
        precision,
        scale,
    })
}

/// Spark's `UTF8String.trimAll`: whitespace and control characters.
fn trim_all(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_control() || (c.is_whitespace() && !is_no_break_space(c)))
}

/// The spaces Java's `Character.isWhitespace` excludes.
fn is_no_break_space(c: char) -> bool {
    matches!(c, '\u{A0}' | '\u{2007}' | '\u{202F}')
}

/// The special values of `DATE` and `TIMESTAMP` literals (`now`,
/// `epoch`, ...), which depend on the session's clock and zone.
pub(crate) fn is_special(value: &str) -> bool {
    let v = value.trim().to_ascii_lowercase();
    ["epoch", "now", "today", "yesterday", "tomorrow"].contains(&v.as_str())
}

/// Days since 1970-01-01 of a date string as Spark's `stringToDate` reads
/// it: `[+-]y[yyy...][-m[m][-d[d]]]`; a full date may be followed by a
/// space or `T` and anything.
pub(crate) fn date(s: &str) -> Option<i32> {
    let s = trim_all(s);
    // A space or `T` may end the date only after its day.
    let s = match s.split_once([' ', 'T']) {
        Some((date, _)) if date.trim_start_matches(['-', '+']).matches('-').count() == 2 => date,
        Some(_) => return None,
        None => s,
    };
    let (neg, body) = match s.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let mut parts = body.split('-');
    let field = |p: Option<&str>, lo: usize, hi: usize| -> Option<i64> {
        p.filter(|p| (lo..=hi).contains(&p.len()) && p.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|p| p.parse().ok())
    };
    let year = field(parts.next(), 4, 7)?;
    let month = parts.next().map_or(Some(1), |m| field(Some(m), 1, 2))?;
    let day = parts.next().map_or(Some(1), |d| field(Some(d), 1, 2))?;
    if parts.next().is_some() || !(1..=12).contains(&month) {
        return None;
    }
    let year = if neg { -year } else { year };
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if day < 1 || day > days_in[usize::try_from(month - 1).ok()?] {
        return None;
    }
    // H. Hinnant's `days_from_civil`.
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    i32::try_from(era * 146_097 + doe - 719_468).ok()
}

/// Whether `s` is a time of day as Spark's `stringToTime` reads it:
/// `[T]h[h]:m[m][:s[s][.fraction]]` on a 24-hour clock, or with ` AM` /
/// ` PM` on a 12-hour clock (hours 1 to 12).
fn time(s: &str) -> bool {
    let s = trim_all(s);
    let s = s.strip_prefix('T').unwrap_or(s);
    let (clock, meridiem) = match s.rsplit_once(' ') {
        Some((clock, m)) if m.eq_ignore_ascii_case("AM") || m.eq_ignore_ascii_case("PM") => {
            (clock.trim_end(), true)
        }
        _ => (s, false),
    };
    let number = |p: &str| -> Option<u32> {
        ((1..=2).contains(&p.len()) && p.bytes().all(|b| b.is_ascii_digit()))
            .then(|| p.parse().ok())
            .flatten()
    };
    let mut parts = clock.split(':');
    let (Some(h), Some(m)) = (parts.next().and_then(number), parts.next().and_then(number)) else {
        return false;
    };
    let seconds_ok = match parts.next() {
        None => true,
        Some(sec) => {
            let (whole, fraction) = sec.split_once('.').unwrap_or((sec, ""));
            number(whole).is_some_and(|s| s < 60)
                && fraction.len() <= 9
                && fraction.bytes().all(|b| b.is_ascii_digit())
        }
    };
    let hour_ok = if meridiem {
        (1..=12).contains(&h)
    } else {
        h < 24
    };
    parts.next().is_none() && seconds_ok && hour_ok && m < 60
}

/// The bytes of a hex string (`X'1F'`); an odd length gets a leading 0.
pub(crate) fn hex(s: &str) -> Option<Vec<u8>> {
    let padded = if s.len() % 2 == 1 {
        format!("0{s}")
    } else {
        s.to_owned()
    };
    (0..padded.len())
        .step_by(2)
        .map(|i| {
            padded
                .get(i..i + 2)
                .and_then(|b| u8::from_str_radix(b, 16).ok())
        })
        .collect()
}

/// The kinds of typed literals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TypedKind {
    Date,
    Time,
    Timestamp,
    TimestampNtz,
    TimestampLtz,
    Interval,
    Binary,
}

impl TypedKind {
    /// The kind a literal's type name denotes (`quoted` names none).
    pub(crate) fn of(name: &str, quoted: bool) -> Result<Self, KrishivError> {
        let upper = name.to_uppercase();
        let kind = match upper.as_str() {
            _ if quoted => None,
            "DATE" => Some(Self::Date),
            "TIME" => Some(Self::Time),
            "TIMESTAMP" => Some(Self::Timestamp),
            "TIMESTAMP_NTZ" => Some(Self::TimestampNtz),
            "TIMESTAMP_LTZ" => Some(Self::TimestampLtz),
            "INTERVAL" => Some(Self::Interval),
            "X" => Some(Self::Binary),
            _ => None,
        };
        kind.ok_or_else(|| {
            UnsupportedTypedLiteral {
                unsupported_type: format!("\"{upper}\""),
                supported_types: TYPED_LITERAL_TYPES.to_owned(),
            }
            .into()
        })
    }

    /// The type as Spark's errors name it.
    fn name(self) -> &'static str {
        match self {
            Self::Date => "DATE",
            Self::Time => "TIME",
            Self::Timestamp => "TIMESTAMP",
            Self::TimestampNtz => "TIMESTAMP_NTZ",
            Self::TimestampLtz => "TIMESTAMP_LTZ",
            Self::Interval => "INTERVAL",
            Self::Binary => "X",
        }
    }

    /// `INVALID_TYPED_LITERAL` for `value`.
    pub(crate) fn invalid(self, value: &str) -> KrishivError {
        InvalidTypedLiteral {
            value_type: format!("\"{}\"", self.name()),
            value: format!("'{value}'"),
        }
        .into()
    }
}

/// Checks the value of a typed literal other than `INTERVAL`: `ntz` is
/// `spark.sql.timestampType = TIMESTAMP_NTZ`, under which `TIMESTAMP` means
/// `TIMESTAMP_NTZ` unless the value names a zone.
///
/// # Errors
/// `INVALID_TYPED_LITERAL` for a value the type does not read.
pub(crate) fn typed(kind: TypedKind, value: &str, ntz: bool) -> Result<(), KrishivError> {
    let ok = match kind {
        TypedKind::Date => is_special(value) || date(value).is_some(),
        TypedKind::Time => time(value),
        TypedKind::Timestamp | TypedKind::TimestampLtz | TypedKind::TimestampNtz => {
            let shape = timestamp_text(value);
            let without_zone =
                kind == TypedKind::TimestampNtz || (kind == TypedKind::Timestamp && ntz);
            is_special(value)
                || shape == TimestampText::DateTime
                || (shape == TimestampText::TimeOnly && !without_zone)
        }
        TypedKind::Binary => hex(value).is_some(),
        TypedKind::Interval => true,
    };
    if ok { Ok(()) } else { Err(kind.invalid(value)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_bounds_compare_exactly() {
        assert!(exceeds("3.4028234663852886E38", FLOAT_MAX.0));
        assert!(exceeds("3.4028235E38", FLOAT_MAX.0));
        assert!(!exceeds(
            "340282346638528859811704183484516925440",
            FLOAT_MAX.0
        ));
        assert!(exceeds(
            "340282346638528859811704183484516925440.5",
            FLOAT_MAX.0
        ));
        assert!(exceeds(
            "340282346638528859811704183484516925441",
            FLOAT_MAX.0
        ));
        assert!(!exceeds("-1e-50", FLOAT_MAX.0));
        assert!(!exceeds("0.0", FLOAT_MAX.0));
        assert!(exceeds("1E999999999999999999999", DOUBLE_MAX.0));
        assert!(!exceeds("1E-999999999999999999999", DOUBLE_MAX.0));
        assert!(!exceeds("1.7976931348623157E308", DOUBLE_MAX.0));
        assert!(exceeds("1.7976931348623158E308", DOUBLE_MAX.0));
        assert!(exceeds("100000E304", DOUBLE_MAX.0));
    }

    #[test]
    fn decimals_count_digits_as_big_decimal_does() {
        let precision = |t: &str| match number(t, NumKind::BigDecimal) {
            Ok(Number::Decimal { precision, .. }) => precision.to_string(),
            Ok(_) => unreachable!(),
            Err(e) => e.message_parameters()[0].1.clone(),
        };
        assert_eq!(precision("1.5E2"), "3");
        assert_eq!(precision("1.20E-38"), "40");
        assert_eq!(precision("1E400"), "401");
        assert_eq!(precision("0.000"), "3");
        assert_eq!(precision("-0012.30"), "4");
        assert_eq!(
            number("1.5E2", NumKind::BigDecimal).unwrap(),
            Number::Decimal {
                unscaled: 150,
                precision: 3,
                scale: 0
            }
        );
        assert_eq!(
            number("-1.25", NumKind::Decimal).unwrap(),
            Number::Decimal {
                unscaled: -125,
                precision: 3,
                scale: 2
            }
        );
    }

    #[test]
    fn times_follow_spark() {
        for ok in [
            "00:00:00",
            "23:59:59.999999999",
            "12:00 AM",
            "1:00 PM",
            "0:0:0",
            "T10:00",
            "10:00:00.",
        ] {
            assert!(time(ok), "{ok}");
        }
        for bad in [
            "00",
            "24:00:00",
            "00:30 AM",
            "13:00 PM",
            "10:00:00 Z",
            "10:00:60",
            "1:2:3:4",
            "now",
        ] {
            assert!(!time(bad), "{bad}");
        }
    }
}
