// SPDX-License-Identifier: Apache-2.0
//! Interval literals as Spark's `AstBuilder` reads them, with Spark's
//! `IntervalUtils` string parsers: `INTERVAL 'text'` and `INTERVAL value
//! unit ...` (`stringToInterval`), `INTERVAL 'y-m' YEAR TO MONTH`
//! (`fromYearMonthString`) and the day-time ranges (`fromDayTimeString`),
//! then the type check: an interval mixing year-month and day-time fields
//! is an error unless `spark.sql.legacy.interval.enabled` is on.

use krishiv_common::error::{
    IntervalArithmeticOverflowWithoutSuggestion, InvalidIntervalFormatArithmeticException,
    InvalidIntervalFormatInputIsEmpty, InvalidIntervalFormatIntervalParsing,
    InvalidIntervalFormatInvalidFraction, InvalidIntervalFormatInvalidPrecision,
    InvalidIntervalFormatInvalidPrefix, InvalidIntervalFormatInvalidUnit,
    InvalidIntervalFormatInvalidValue, InvalidIntervalFormatMissingNumber,
    InvalidIntervalFormatMissingUnit, InvalidIntervalFormatUnmatchedFormatString,
    InvalidIntervalFormatUnmatchedFormatStringWithNotice, InvalidIntervalFormatUnrecognizedNumber,
    InvalidIntervalFormatUnsupportedFromToExpression, KrishivError, LegacyErrorTemp0024,
    LegacyErrorTemp0026, LegacyErrorTemp0027, LegacyErrorTemp0029, LegacyErrorTemp0063,
};

use super::literal::TypedKind;
use crate::ast::{Interval, IntervalValue, Span};
use crate::error::ParseError;

const MICROS_PER_SECOND: i64 = 1_000_000;
const MICROS_PER_MINUTE: i64 = 60 * MICROS_PER_SECOND;
const MICROS_PER_HOUR: i64 = 60 * MICROS_PER_MINUTE;
const MICROS_PER_DAY: i64 = 24 * MICROS_PER_HOUR;

/// Spark's `CalendarInterval`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Calendar {
    months: i32,
    days: i32,
    micros: i64,
}

fn at(error: impl Into<KrishivError>, span: Span) -> ParseError {
    ParseError {
        error: error.into(),
        span,
    }
}

/// Checks an interval literal (`INTERVAL value unit ...` or `INTERVAL value
/// from TO to`); `sql` is the statement, for the literal's text.
///
/// # Errors
/// Spark's errors for the literal (the module docs).
pub(crate) fn literal(iv: &Interval, sql: &str, legacy: bool) -> Result<(), ParseError> {
    if let Some(trailing) = &iv.trailing {
        // A range with anything else, or two ranges.
        let span = Span::new(iv.span.start, iv.span.end.max(trailing.span.end));
        return Err(at(LegacyErrorTemp0024 {}, span));
    }
    let body = Span::new(
        iv.units
            .first()
            .map(|(v, _)| v.span.start)
            .or(iv.range.as_ref().map(|r| r.0.span.start))
            .unwrap_or(iv.span.start),
        iv.span.end,
    );
    let calendar = match &iv.range {
        Some((value, from, to)) => range(value, from, to).map_err(|e| at(e, body))?,
        None => multi_units(&iv.units).map_err(|e| at(e, body))?,
    };
    typed_check(calendar, iv.span, sql, legacy)
}

/// Checks the typed literal `INTERVAL 'text'` at `span`.
///
/// # Errors
/// `INVALID_TYPED_LITERAL` when `text` is not an interval, and the mixed
/// fields and overflow errors of [`literal`].
pub(crate) fn typed(value: &str, span: Span, sql: &str, legacy: bool) -> Result<(), ParseError> {
    let calendar =
        string_to_interval(value).map_err(|_| at(TypedKind::Interval.invalid(value), span))?;
    typed_check(calendar, span, sql, legacy)
}

/// Spark's `visitInterval` after parsing: a year-month or a day-time
/// interval, never both (unless legacy).
fn typed_check(c: Calendar, span: Span, sql: &str, legacy: bool) -> Result<(), ParseError> {
    if legacy {
        return Ok(());
    }
    if c.months != 0 {
        if c.days != 0 || c.micros != 0 {
            return Err(at(
                LegacyErrorTemp0029 {
                    literal: span.text(sql).to_owned(),
                },
                span,
            ));
        }
        return Ok(());
    }
    // A day-time interval that does not fit in microseconds fails when it
    // is evaluated, not here.
    Ok(())
}

/// A value with its sign as Spark writes it into the unit string: a
/// negated string loses or gains a leading `-`.
fn signed(v: &IntervalValue) -> String {
    match (v.negative, v.value.strip_prefix('-')) {
        (false, _) => v.value.clone(),
        (true, Some(rest)) if v.is_string => rest.to_owned(),
        (true, _) => format!("-{}", v.value),
    }
}

/// `value unit [value unit ...]`: Spark joins ` value unit` pairs and
/// parses them as one interval string.
fn multi_units(units: &[(IntervalValue, String)]) -> Result<Calendar, KrishivError> {
    let mut text = String::new();
    for (value, unit) in units {
        // A string value holding a unit would read as one once joined.
        if value.is_string && value.value.chars().any(char::is_alphabetic) {
            return Err(LegacyErrorTemp0026 {
                value: value.value.clone(),
            }
            .into());
        }
        text.push(' ');
        text.push_str(&signed(value));
        text.push(' ');
        text.push_str(unit);
    }
    string_to_interval(&text)
}

/// `value from TO to`.
fn range(value: &IntervalValue, from: &str, to: &str) -> Result<Calendar, KrishivError> {
    if !value.is_string {
        return Err(LegacyErrorTemp0027 {}.into());
    }
    let input = signed(value);
    let field = |u: &str| DT_FIELDS.iter().position(|f| *f == u);
    match (from, to) {
        ("year", "month") => year_month(&input),
        _ => match (field(from), field(to)) {
            (Some(f), Some(t)) if f < t => day_time(&input, f, t),
            _ => Err(InvalidIntervalFormatUnsupportedFromToExpression {
                input,
                from: from.to_owned(),
                to: to.to_owned(),
            }
            .into()),
        },
    }
}

/// `[+|-]y-m`, or the same as an `INTERVAL ... YEAR TO MONTH` literal.
fn year_month(input: &str) -> Result<Calendar, KrishivError> {
    let unmatched = || -> KrishivError {
        InvalidIntervalFormatUnmatchedFormatString {
            input: input.to_owned(),
            interval_str: "year-month".to_owned(),
            supported_format: "`[+|-]y-m`, `INTERVAL [+|-]'[+|-]y-m' YEAR TO MONTH`".to_owned(),
            type_name: "interval year to month".to_owned(),
        }
        .into()
    };
    let (sign, body) =
        literal_form(trim_all(input), "YEAR TO MONTH").unwrap_or((false, trim_all(input)));
    let (inner_sign, body) = split_sign(body);
    let (years, months) = body.split_once('-').ok_or_else(unmatched)?;
    if !all_digits(years) || !all_digits(months) {
        return Err(unmatched());
    }
    let negative = sign != inner_sign;
    let parsing = || -> KrishivError {
        InvalidIntervalFormatIntervalParsing {
            input: input.to_owned(),
            interval: "year-month".to_owned(),
        }
        .into()
    };
    let years: i64 = years.parse().map_err(|_| parsing())?;
    let months: i64 = months.parse().map_err(|_| parsing())?;
    if years > i64::from(i32::MAX / 12) || months > 11 {
        return Err(parsing());
    }
    let total = years * 12 + months;
    let total = if negative { -total } else { total };
    let months = i32::try_from(total).map_err(|_| parsing())?;
    Ok(Calendar {
        months,
        ..Calendar::default()
    })
}

/// The day-time fields, in order.
const DT_FIELDS: &[&str] = &["day", "hour", "minute", "second"];

/// The largest leading value of each day-time field: what fits in
/// microseconds.
const DT_MAX: [i64; 4] = [
    i64::MAX / MICROS_PER_DAY,
    i64::MAX / MICROS_PER_HOUR,
    i64::MAX / MICROS_PER_MINUTE,
    i64::MAX / MICROS_PER_SECOND,
];

/// The bound of a field that is not the leading one.
const DT_INNER_MAX: [i64; 4] = [0, 23, 59, 59];

const DT_MICROS: [i64; 4] = [
    MICROS_PER_DAY,
    MICROS_PER_HOUR,
    MICROS_PER_MINUTE,
    MICROS_PER_SECOND,
];

/// `fromDayTimeString(input, from, to)`: the fields `from..=to` written
/// `d h:m:s.n` (day and hour separated by a space, the others by `:`).
fn day_time(input: &str, from: usize, to: usize) -> Result<Calendar, KrishivError> {
    let pattern = ["d", " h", ":m", ":s.n"][from..=to].concat();
    let pattern = pattern.trim_start_matches([' ', ':']);
    let unit = |f: usize| DT_FIELDS[f].to_uppercase();
    let unmatched = || -> KrishivError {
        InvalidIntervalFormatUnmatchedFormatStringWithNotice {
            input: input.to_owned(),
            interval_str: "day-time".to_owned(),
            supported_format: format!(
                "`[+|-]{pattern}`, `INTERVAL [+|-]'[+|-]{pattern}' {} TO {}`",
                unit(from),
                unit(to)
            ),
            type_name: format!("interval {} to {}", DT_FIELDS[from], DT_FIELDS[to]),
        }
        .into()
    };
    let suffix = format!("{} TO {}", unit(from), unit(to));
    let (sign, body) = literal_form(trim_all(input), &suffix).unwrap_or((false, trim_all(input)));
    let (inner_sign, body) = split_sign(body);
    // Split into the fields from `from` to `to`.
    let mut fields: Vec<&str> = Vec::with_capacity(to - from + 1);
    let mut rest = body;
    for f in from..=to {
        if f == to {
            fields.push(rest);
            break;
        }
        let sep = if f == 0 { ' ' } else { ':' };
        let (head, tail) = rest.split_once(sep).ok_or_else(unmatched)?;
        fields.push(head);
        rest = tail;
    }
    let negative = sign != inner_sign;
    let mut micros: i128 = 0;
    let overflow = || -> KrishivError { IntervalArithmeticOverflowWithoutSuggestion {}.into() };
    for (k, text) in fields.iter().enumerate() {
        let f = from + k;
        let (whole, fraction) = if f == 3 {
            match text.split_once('.') {
                Some((w, n)) if !n.is_empty() && n.len() <= 9 && all_digits(n) => (w, Some(n)),
                Some(_) => return Err(unmatched()),
                None => (*text, None),
            }
        } else {
            (*text, None)
        };
        let leading = k == 0;
        let digits_ok = all_digits(whole) && (leading || whole.len() <= 2);
        if !digits_ok {
            return Err(unmatched());
        }
        let value: i64 = whole.parse().unwrap_or(i64::MAX);
        let max = if leading { DT_MAX[f] } else { DT_INNER_MAX[f] };
        if value > max {
            return Err(LegacyErrorTemp0063 {
                msg: format!(
                    "requirement failed: {} {value} outside range [0, {max}]",
                    DT_FIELDS[f]
                ),
            }
            .into());
        }
        // Within the field bounds, so the sum fits in `i128`.
        micros += i128::from(value) * i128::from(DT_MICROS[f])
            + i128::from(fraction.map_or(0, nanos_to_micros));
    }
    if negative {
        micros = -micros;
    }
    let micros = i64::try_from(micros).map_err(|_| overflow())?;
    Ok(Calendar {
        months: 0,
        days: i32::try_from(micros / MICROS_PER_DAY).map_err(|_| overflow())?,
        micros: micros % MICROS_PER_DAY,
    })
}

/// Microseconds of a fraction of up to 9 digits.
fn nanos_to_micros(fraction: &str) -> i64 {
    let padded = format!("{fraction:0<9}");
    padded.get(..6).and_then(|d| d.parse().ok()).unwrap_or(0)
}

/// `INTERVAL [+|-]'body' <suffix>` (case-insensitive): the outer sign and
/// the body, if `s` is written that way.
fn literal_form<'a>(s: &'a str, suffix: &str) -> Option<(bool, &'a str)> {
    let lower = s.to_ascii_lowercase();
    let rest = lower.strip_prefix("interval")?;
    let skipped = s.len() - rest.len();
    let rest_orig = s.get(skipped..)?;
    let trimmed = rest_orig.trim_start();
    if trimmed.len() == rest_orig.len() {
        return None;
    }
    let (sign, quoted) = split_sign(trimmed);
    let quoted = quoted.strip_prefix('\'')?;
    let (body, tail) = quoted.split_once('\'')?;
    let words: Vec<&str> = tail.split_whitespace().collect();
    let expected: Vec<String> = suffix.split(' ').map(str::to_ascii_lowercase).collect();
    let starts_with_space = tail.starts_with(char::is_whitespace);
    (starts_with_space
        && words.len() == expected.len()
        && words
            .iter()
            .zip(&expected)
            .all(|(w, e)| w.eq_ignore_ascii_case(e)))
    .then_some((sign, body))
}

/// A leading `+` or `-`: whether it is `-`, and the rest.
fn split_sign(s: &str) -> (bool, &str) {
    match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    }
}

fn all_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// Spark's `UTF8String.trimAll`: whitespace and control characters.
fn trim_all(s: &str) -> &str {
    s.trim_matches(|c: char| {
        c.is_control() || (c.is_whitespace() && !matches!(c, '\u{A0}' | '\u{2007}' | '\u{202F}'))
    })
}

/// A state of [`string_to_interval`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Prefix,
    TrimBeforeSign,
    Sign,
    TrimBeforeValue,
    Value,
    Fraction,
    TrimBeforeUnit,
    UnitBegin,
    UnitSuffix,
    UnitEnd,
}

/// Spark's `IntervalUtils.stringToInterval`: `[interval] [+|-]value unit
/// ...`, units `year`, `month`, `week`, `day`, `hour`, `minute`, `second`,
/// `millisecond`, `microsecond`, `nanosecond` (each optionally plural); only seconds take
/// a fraction (up to nanoseconds).
fn string_to_interval(input: &str) -> Result<Calendar, KrishivError> {
    let s = trim_all(input).to_lowercase();
    let b = s.as_bytes();
    if b.is_empty() {
        return Err(InvalidIntervalFormatInputIsEmpty {
            input: input.to_owned(),
        }
        .into());
    }
    // The whitespace-separated word at byte `i`, as Spark's errors cite it.
    let word = |i: usize| -> String {
        let start = s[..i.min(s.len())]
            .rfind(char::is_whitespace)
            .map_or(0, |p| p + 1);
        s[start..]
            .split(char::is_whitespace)
            .next()
            .unwrap_or_default()
            .to_owned()
    };
    let input_owned = || input.to_owned();
    let arithmetic = || -> KrishivError {
        InvalidIntervalFormatArithmeticException {
            input: input.to_owned(),
        }
        .into()
    };
    let initial_scale: i64 = 100_000_000;
    let (mut state, mut i) = (State::Prefix, 0_usize);
    let (mut value, mut negative, mut fraction, mut scale) = (0_i64, false, 0_i64, -1_i64);
    let mut point_prefixed = false;
    let mut c = Calendar::default();
    while i < b.len() {
        let ch = b[i];
        let space = ch.is_ascii_whitespace();
        match state {
            State::Prefix => {
                if s.starts_with("interval") {
                    match b.get(8) {
                        None => {
                            return Err(InvalidIntervalFormatInvalidPrefix {
                                input: input_owned(),
                                prefix: word(0),
                            }
                            .into());
                        }
                        Some(n) if !n.is_ascii_whitespace() => {
                            return Err(InvalidIntervalFormatInvalidPrefix {
                                input: input_owned(),
                                prefix: word(0),
                            }
                            .into());
                        }
                        Some(_) => i += 9,
                    }
                }
                state = State::TrimBeforeSign;
            }
            State::TrimBeforeSign | State::TrimBeforeValue | State::TrimBeforeUnit => {
                if space {
                    i += 1;
                } else {
                    state = match state {
                        State::TrimBeforeSign => State::Sign,
                        State::TrimBeforeValue => State::Value,
                        _ => State::UnitBegin,
                    };
                }
            }
            State::Sign => {
                value = 0;
                fraction = 0;
                scale = -1;
                point_prefixed = false;
                state = State::TrimBeforeValue;
                match ch {
                    b'-' => {
                        negative = true;
                        i += 1;
                    }
                    b'+' => {
                        negative = false;
                        i += 1;
                    }
                    b'0'..=b'9' => negative = false,
                    b'.' => {
                        negative = false;
                        scale = initial_scale;
                        point_prefixed = true;
                        i += 1;
                        state = State::Fraction;
                    }
                    _ => {
                        return Err(InvalidIntervalFormatUnrecognizedNumber {
                            input: input_owned(),
                            number: word(i),
                        }
                        .into());
                    }
                }
            }
            State::Value => {
                match ch {
                    b'0'..=b'9' => {
                        value = value
                            .checked_mul(10)
                            .and_then(|v| v.checked_add(i64::from(ch - b'0')))
                            .ok_or_else(arithmetic)?;
                    }
                    _ if space => state = State::TrimBeforeUnit,
                    b'.' => {
                        scale = initial_scale;
                        state = State::Fraction;
                    }
                    _ => {
                        return Err(InvalidIntervalFormatInvalidValue {
                            input: input_owned(),
                            value: word(i),
                        }
                        .into());
                    }
                }
                i += 1;
            }
            State::Fraction => {
                if ch.is_ascii_digit() && scale > 0 {
                    fraction += i64::from(ch - b'0') * scale;
                    scale /= 10;
                } else if space && (!point_prefixed || scale < initial_scale) {
                    fraction /= 1000;
                    state = State::TrimBeforeUnit;
                } else if ch.is_ascii_digit() {
                    return Err(InvalidIntervalFormatInvalidPrecision {
                        input: input_owned(),
                        value: word(i),
                    }
                    .into());
                } else {
                    return Err(InvalidIntervalFormatInvalidValue {
                        input: input_owned(),
                        value: word(i),
                    }
                    .into());
                }
                i += 1;
            }
            State::UnitBegin => {
                if ch != b's' && scale >= 0 {
                    return Err(InvalidIntervalFormatInvalidFraction {
                        input: input_owned(),
                        unit: word(i),
                    }
                    .into());
                }
                if negative {
                    value = -value;
                    fraction = -fraction;
                }
                let rest = &s[i..];
                let invalid_unit = || -> KrishivError {
                    InvalidIntervalFormatInvalidUnit {
                        input: input.to_owned(),
                        unit: word(i),
                    }
                    .into()
                };
                let add_months = |c: &mut Calendar, m: i64| -> Result<(), KrishivError> {
                    let total = i64::from(c.months).checked_add(m).ok_or_else(arithmetic)?;
                    c.months = i32::try_from(total).map_err(|_| arithmetic())?;
                    Ok(())
                };
                let add_days = |c: &mut Calendar, d: i64| -> Result<(), KrishivError> {
                    let total = i64::from(c.days).checked_add(d).ok_or_else(arithmetic)?;
                    c.days = i32::try_from(total).map_err(|_| arithmetic())?;
                    Ok(())
                };
                let add_micros = |c: &mut Calendar, v: i64, per: i64| -> Result<(), KrishivError> {
                    let us = v.checked_mul(per).ok_or_else(arithmetic)?;
                    c.micros = c.micros.checked_add(us).ok_or_else(arithmetic)?;
                    Ok(())
                };
                let units: [(&str, i64); 10] = [
                    ("year", 0),
                    ("month", 1),
                    ("week", 2),
                    ("day", 3),
                    ("hour", MICROS_PER_HOUR),
                    ("minute", MICROS_PER_MINUTE),
                    ("second", MICROS_PER_SECOND),
                    ("millisecond", 1000),
                    ("microsecond", 1),
                    ("nanosecond", 0),
                ];
                let Some((unit, per)) = units.iter().find(|(u, _)| rest.starts_with(u)) else {
                    return Err(invalid_unit());
                };
                match *unit {
                    "year" => add_months(&mut c, value.checked_mul(12).ok_or_else(arithmetic)?)?,
                    "month" => {
                        // `month` is added as an `int`.
                        let v = i32::try_from(value).map_err(|_| arithmetic())?;
                        add_months(&mut c, i64::from(v))?;
                    }
                    "week" => add_days(&mut c, value.checked_mul(7).ok_or_else(arithmetic)?)?,
                    "day" => {
                        let v = i32::try_from(value).map_err(|_| arithmetic())?;
                        add_days(&mut c, i64::from(v))?;
                    }
                    "second" => {
                        add_micros(&mut c, value, *per)?;
                        c.micros = c.micros.checked_add(fraction).ok_or_else(arithmetic)?;
                    }
                    "nanosecond" => {
                        c.micros = c.micros.checked_add(value / 1000).ok_or_else(arithmetic)?;
                    }
                    _ => add_micros(&mut c, value, *per)?,
                }
                i += unit.len();
                state = State::UnitSuffix;
            }
            State::UnitSuffix => {
                match ch {
                    b's' => state = State::UnitEnd,
                    _ if space => state = State::TrimBeforeSign,
                    _ => {
                        return Err(InvalidIntervalFormatInvalidUnit {
                            input: input_owned(),
                            unit: word(i),
                        }
                        .into());
                    }
                }
                i += 1;
            }
            State::UnitEnd => {
                if space {
                    i += 1;
                    state = State::TrimBeforeSign;
                } else {
                    return Err(InvalidIntervalFormatInvalidUnit {
                        input: input_owned(),
                        unit: word(i),
                    }
                    .into());
                }
            }
        }
    }
    match state {
        State::UnitSuffix | State::UnitEnd | State::TrimBeforeSign => Ok(c),
        State::TrimBeforeValue => Err(InvalidIntervalFormatMissingNumber {
            input: input_owned(),
            word: word(i.saturating_sub(1)),
        }
        .into()),
        // The other states never end a trimmed input (Spark's catch-all
        // `UNKNOWN_PARSING_ERROR` cannot happen).
        _ => Err(InvalidIntervalFormatMissingUnit {
            input: input_owned(),
            word: word(i.saturating_sub(1)),
        }
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class(r: Result<Calendar, KrishivError>) -> String {
        r.map_or_else(|e| e.error_class().name().to_owned(), |c| format!("{c:?}"))
    }

    #[test]
    fn strings_parse_as_spark_does() {
        let ok = |s: &str| string_to_interval(s).unwrap();
        assert_eq!(ok("1 day").days, 1);
        assert_eq!(ok("interval 1 year 2 months").months, 14);
        assert_eq!(ok("  2 weeks -3 days ").days, 11);
        assert_eq!(ok("1 hour 70 minutes").micros, 130 * MICROS_PER_MINUTE);
        assert_eq!(ok("4.5 seconds").micros, 4_500_000);
        assert_eq!(ok(".5 second").micros, 500_000);
        assert_eq!(ok("1 millisecond 1 microsecond").micros, 1001);
        assert_eq!(ok("-1.000000001 second").micros, -1_000_000);
        for (s, err) in [
            ("", "INVALID_INTERVAL_FORMAT.INPUT_IS_EMPTY"),
            ("interval", "INVALID_INTERVAL_FORMAT.INVALID_PREFIX"),
            ("intervalx 1 day", "INVALID_INTERVAL_FORMAT.INVALID_PREFIX"),
            ("x day", "INVALID_INTERVAL_FORMAT.UNRECOGNIZED_NUMBER"),
            ("1x day", "INVALID_INTERVAL_FORMAT.INVALID_VALUE"),
            ("1.x second", "INVALID_INTERVAL_FORMAT.INVALID_VALUE"),
            (
                "1.0000000001 second",
                "INVALID_INTERVAL_FORMAT.INVALID_PRECISION",
            ),
            ("1.5 day", "INVALID_INTERVAL_FORMAT.INVALID_FRACTION"),
            ("1 d", "INVALID_INTERVAL_FORMAT.INVALID_UNIT"),
            ("1 dayx", "INVALID_INTERVAL_FORMAT.INVALID_UNIT"),
            ("1 daysx", "INVALID_INTERVAL_FORMAT.INVALID_UNIT"),
            ("1 day -", "INVALID_INTERVAL_FORMAT.MISSING_NUMBER"),
            ("1", "INVALID_INTERVAL_FORMAT.MISSING_UNIT"),
            ("1.2", "INVALID_INTERVAL_FORMAT.MISSING_UNIT"),
            ("+", "INVALID_INTERVAL_FORMAT.MISSING_NUMBER"),
            (
                "178956971 years",
                "INVALID_INTERVAL_FORMAT.ARITHMETIC_EXCEPTION",
            ),
            (
                "9223372036854775808 microseconds",
                "INVALID_INTERVAL_FORMAT.ARITHMETIC_EXCEPTION",
            ),
            (
                "2147483648 days",
                "INVALID_INTERVAL_FORMAT.ARITHMETIC_EXCEPTION",
            ),
            (
                "2147483648 months",
                "INVALID_INTERVAL_FORMAT.ARITHMETIC_EXCEPTION",
            ),
            (
                "9223372036854775807 hours",
                "INVALID_INTERVAL_FORMAT.ARITHMETIC_EXCEPTION",
            ),
            (
                "1 week 2147483647 days",
                "INVALID_INTERVAL_FORMAT.ARITHMETIC_EXCEPTION",
            ),
            (
                "9223372036854775807 seconds",
                "INVALID_INTERVAL_FORMAT.ARITHMETIC_EXCEPTION",
            ),
        ] {
            assert_eq!(class(string_to_interval(s)), err, "{s}");
        }
    }

    #[test]
    fn ranges_parse_as_spark_does() {
        let ym = |s: &str| class(year_month(s));
        assert_eq!(
            ym("10-9"),
            format!(
                "{:?}",
                Calendar {
                    months: 129,
                    ..Calendar::default()
                }
            )
        );
        assert_eq!(
            ym("-178956970-8"),
            format!(
                "{:?}",
                Calendar {
                    months: i32::MIN,
                    ..Calendar::default()
                }
            )
        );
        assert_eq!(
            ym("INTERVAL -'1-2' YEAR TO MONTH"),
            format!(
                "{:?}",
                Calendar {
                    months: -14,
                    ..Calendar::default()
                }
            )
        );
        assert_eq!(
            ym("interval '1-2' year"),
            "INVALID_INTERVAL_FORMAT.UNMATCHED_FORMAT_STRING"
        );
        assert_eq!(ym("1-x"), "INVALID_INTERVAL_FORMAT.UNMATCHED_FORMAT_STRING");
        assert_eq!(
            ym("99999999999999999999-1"),
            "INVALID_INTERVAL_FORMAT.INTERVAL_PARSING"
        );
        let dt = |s: &str, f: usize, t: usize| class(day_time(s, f, t));
        assert_eq!(
            dt("1 2:03:04.5", 0, 3),
            format!(
                "{:?}",
                Calendar {
                    months: 0,
                    days: 1,
                    micros: 2 * MICROS_PER_HOUR + 3 * MICROS_PER_MINUTE + 4_500_000
                }
            )
        );
        assert_eq!(
            dt("INTERVAL '-1 2' DAY TO HOUR", 0, 1),
            format!(
                "{:?}",
                Calendar {
                    months: 0,
                    days: -1,
                    micros: -2 * MICROS_PER_HOUR
                }
            )
        );
        assert_eq!(
            dt("1 2:03:04.", 0, 3),
            "INVALID_INTERVAL_FORMAT.UNMATCHED_FORMAT_STRING_WITH_NOTICE"
        );
        assert_eq!(
            dt("1 2:3:4.1234567890", 0, 3),
            "INVALID_INTERVAL_FORMAT.UNMATCHED_FORMAT_STRING_WITH_NOTICE"
        );
        assert_eq!(
            dt("1 123:00", 0, 2),
            "INVALID_INTERVAL_FORMAT.UNMATCHED_FORMAT_STRING_WITH_NOTICE"
        );
        assert_eq!(
            dt("106751991 23", 0, 1),
            "INTERVAL_ARITHMETIC_OVERFLOW.WITHOUT_SUGGESTION"
        );
        assert_eq!(
            dt("99999999999999999999:00", 1, 2),
            "_LEGACY_ERROR_TEMP_0063"
        );
    }
}
