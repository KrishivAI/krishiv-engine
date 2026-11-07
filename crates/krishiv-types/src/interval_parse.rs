// SPDX-License-Identifier: Apache-2.0
//! Spark's `IntervalUtils` string parsers that run at both plan and run time:
//! `stringToInterval` ([`parse_string`], a calendar interval, also the
//! `INTERVAL 'text'` literal) and the strings a cast reads as an ANSI interval
//! ([`cast_to_year_month`], [`cast_to_day_time`]), which are not the literal
//! grammar.

use krishiv_common::error::{
    InvalidIntervalFormatArithmeticException, InvalidIntervalFormatInputIsEmpty,
    InvalidIntervalFormatIntervalParsing, InvalidIntervalFormatInvalidFraction,
    InvalidIntervalFormatInvalidPrecision, InvalidIntervalFormatInvalidPrefix,
    InvalidIntervalFormatInvalidUnit, InvalidIntervalFormatInvalidValue,
    InvalidIntervalFormatMissingNumber, InvalidIntervalFormatMissingUnit,
    InvalidIntervalFormatUnmatchedFormatString,
    InvalidIntervalFormatUnmatchedFormatStringWithNotice, InvalidIntervalFormatUnrecognizedNumber,
    KrishivError,
};

use crate::{DayTimeField, YearMonthField};

const MICROS_PER_SECOND: i64 = 1_000_000;
const MICROS_PER_MINUTE: i64 = 60 * MICROS_PER_SECOND;
const MICROS_PER_HOUR: i64 = 60 * MICROS_PER_MINUTE;
const MICROS_PER_DAY: i64 = 24 * MICROS_PER_HOUR;

/// Splits `[+|-]rest`; the sign is `-1` or `1`.
fn sign_of(s: &str) -> (i64, &str) {
    match s.as_bytes().first() {
        Some(b'-') => (-1, &s[1..]),
        Some(b'+') => (1, &s[1..]),
        _ => (1, s),
    }
}

/// The parts of `INTERVAL [+|-]'body' qualifier` (any case, any spaces), where
/// `qualifier` is the target type's; or `None` if `s` is not that literal.
fn literal_body(s: &str, qualifier: impl Fn() -> String) -> Option<(i64, &str)> {
    let head = s.get(..8)?;
    if !head.eq_ignore_ascii_case("interval") {
        return None;
    }
    let rest = s[8..].strip_prefix(|c: char| c.is_ascii_whitespace())?;
    let (outer, rest) = sign_of(rest.trim_start());
    let rest = rest.strip_prefix('\'')?;
    let (body, tail) = rest.split_once('\'')?;
    let tail = tail.strip_prefix(|c: char| c.is_ascii_whitespace())?;
    let words: Vec<&str> = tail.split_ascii_whitespace().collect();
    (words.join(" ").eq_ignore_ascii_case(&qualifier())).then_some((outer, body))
}

fn digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// `START` or `START TO END`.
fn qualifier(start: &str, end: &str) -> String {
    if start == end {
        start.to_ascii_uppercase()
    } else {
        format!(
            "{} TO {}",
            start.to_ascii_uppercase(),
            end.to_ascii_uppercase()
        )
    }
}

/// `CAST(string AS <year-month type>)`, in months.
///
/// # Errors
/// `INVALID_INTERVAL_FORMAT.UNMATCHED_FORMAT_STRING` for another form, and
/// `.INTERVAL_PARSING` for a field out of range or a value past the `int` range.
pub fn cast_to_year_month(
    input: &str,
    start: YearMonthField,
    end: YearMonthField,
) -> Result<i32, KrishivError> {
    let qualifier = || qualifier(start.name(), end.name());
    let unmatched = || -> KrishivError {
        let qualifier = qualifier();
        InvalidIntervalFormatUnmatchedFormatString {
            input: input.to_owned(),
            interval_str: "year-month".to_owned(),
            supported_format: if start == end {
                format!("`[+|-]y`, `INTERVAL [+|-]'[+|-]y' {qualifier}`")
            } else {
                format!("`[+|-]y-m`, `INTERVAL [+|-]'[+|-]y-m' {qualifier}`")
            },
            type_name: format!("interval {}", qualifier.to_ascii_lowercase()),
        }
        .into()
    };
    let parsing = || -> KrishivError {
        InvalidIntervalFormatIntervalParsing {
            input: input.to_owned(),
            interval: "year-month".to_owned(),
        }
        .into()
    };
    let s = trim_all(input);
    let (outer, body) = literal_body(s, qualifier).unwrap_or((1, s));
    let (inner, body) = sign_of(body);
    let sign = outer.saturating_mul(inner);
    let months = if start == end {
        if !digits(body) {
            return Err(unmatched());
        }
        let v: i64 = body.parse().map_err(|_| parsing())?;
        match start {
            YearMonthField::Year => v.checked_mul(12).ok_or_else(parsing)?,
            YearMonthField::Month => v,
        }
    } else {
        let (y, m) = body.split_once('-').ok_or_else(unmatched)?;
        if !digits(y) || !digits(m) {
            return Err(unmatched());
        }
        let y: i64 = y.parse().map_err(|_| parsing())?;
        let m: i64 = m.parse().map_err(|_| parsing())?;
        if m > 11 {
            return Err(parsing());
        }
        y.checked_mul(12)
            .and_then(|v| v.checked_add(m))
            .ok_or_else(parsing)?
    };
    months
        .checked_mul(sign)
        .and_then(|v| i32::try_from(v).ok())
        .ok_or_else(parsing)
}

/// Whole microseconds of `.fraction`; digits past the sixth are dropped.
fn fraction_micros(fraction: &str) -> Option<i64> {
    if fraction.is_empty() || fraction.len() > 9 || !digits(fraction) {
        return None;
    }
    let kept = fraction.get(..fraction.len().min(6))?;
    let scale = 10_i64.checked_pow(u32::try_from(6_usize.saturating_sub(kept.len())).ok()?)?;
    kept.parse::<i64>().ok()?.checked_mul(scale)
}

/// `CAST(string AS <day-time type>)`, in microseconds. The first field takes
/// at most 9 digits for a day, 10 for an hour, 12 for a minute and 13 for a
/// second, leading zeros included, and the others stay in their range. One
/// field other than seconds wraps, as Spark's `Long` arithmetic does
/// (`CAST('106751992' AS INTERVAL DAY)` is negative); seconds and a range of
/// fields fail past the `long` range instead.
///
/// # Errors
/// `INVALID_INTERVAL_FORMAT.UNMATCHED_FORMAT_STRING_WITH_NOTICE` for another
/// form, and `.INTERVAL_PARSING` for a field or a sum out of range. For most of
/// those Spark raises a bare `IllegalArgumentException` or `ArithmeticException`;
/// the class is Krishiv's.
pub fn cast_to_day_time(
    input: &str,
    start: DayTimeField,
    end: DayTimeField,
) -> Result<i64, KrishivError> {
    // Field, its microseconds, the largest value after the first field, and
    // as the first field its most digits and its largest value in a range.
    const SIZES: [(DayTimeField, i64, i64, usize, i64); 4] = [
        (DayTimeField::Day, MICROS_PER_DAY, i64::MAX, 9, 106_751_991),
        (DayTimeField::Hour, MICROS_PER_HOUR, 23, 10, 2_562_047_788),
        (
            DayTimeField::Minute,
            MICROS_PER_MINUTE,
            59,
            12,
            153_722_867_280,
        ),
        (
            DayTimeField::Second,
            MICROS_PER_SECOND,
            59,
            13,
            9_223_372_036_854,
        ),
    ];
    let qualifier = || qualifier(start.name(), end.name());
    let unmatched = || -> KrishivError {
        let qualifier = qualifier();
        InvalidIntervalFormatUnmatchedFormatStringWithNotice {
            input: input.to_owned(),
            interval_str: "day-time".to_owned(),
            supported_format: format!(
                "`[+|-]{0}`, `INTERVAL [+|-]'[+|-]{0}' {qualifier}`",
                pattern(start, end)
            ),
            type_name: format!("interval {}", qualifier.to_ascii_lowercase()),
        }
        .into()
    };
    let parsing = || -> KrishivError {
        InvalidIntervalFormatIntervalParsing {
            input: input.to_owned(),
            interval: "day-time".to_owned(),
        }
        .into()
    };
    let s = trim_all(input);
    let (outer, body) = literal_body(s, qualifier).unwrap_or((1, s));
    let (inner, body) = sign_of(body);
    let sign = outer.saturating_mul(inner);
    // `d h:m:s`: a space after a day, a colon after the rest.
    let mut rest = body;
    let mut total = 0_i64;
    // One field wraps as `Long` arithmetic does, but for seconds; a range is
    // checked.
    let exact = start != end || end == DayTimeField::Second;
    let fields = SIZES.iter().filter(|(f, ..)| *f >= start && *f <= end);
    for (at, (field, size, max, first_digits, first_max)) in fields.enumerate() {
        let separator = if *field == DayTimeField::Day {
            ' '
        } else {
            ':'
        };
        let (mut text, tail) = if *field == end {
            (rest, "")
        } else {
            rest.split_once(separator).ok_or_else(unmatched)?
        };
        rest = tail;
        let mut micros = 0;
        if *field == DayTimeField::Second
            && let Some((whole, fraction)) = text.split_once('.')
        {
            micros = fraction_micros(fraction).ok_or_else(unmatched)?;
            text = whole;
        }
        // Only the first field takes more than two digits.
        if !digits(text) || text.len() > if at > 0 { 2 } else { *first_digits } {
            return Err(unmatched());
        }
        let v: i64 = text.parse().map_err(|_| parsing())?;
        if v > if at > 0 {
            *max
        } else if exact {
            *first_max
        } else {
            i64::MAX
        } {
            return Err(parsing());
        }
        total = if exact {
            v.checked_mul(*size)
                .and_then(|x| x.checked_mul(sign))
                .and_then(|x| total.checked_add(x))
                .and_then(|x| x.checked_add(micros.wrapping_mul(sign)))
                .ok_or_else(parsing)?
        } else {
            total
                .wrapping_add(v.wrapping_mul(*size).wrapping_mul(sign))
                .wrapping_add(micros.wrapping_mul(sign))
        };
    }
    Ok(total)
}

/// The form of `start TO end` in Spark's message.
fn pattern(start: DayTimeField, end: DayTimeField) -> String {
    let names = [
        (DayTimeField::Day, "d"),
        (DayTimeField::Hour, "h"),
        (DayTimeField::Minute, "m"),
        (DayTimeField::Second, "s"),
    ];
    let mut out = String::new();
    for (field, name) in names {
        if field >= start && field <= end {
            if !out.is_empty() {
                out.push(if out == "d" { ' ' } else { ':' });
            }
            out.push_str(name);
        }
    }
    out
}

/// Spark's `CalendarInterval`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Calendar {
    /// Months.
    pub months: i32,
    /// Days.
    pub days: i32,
    /// Microseconds.
    pub micros: i64,
}

/// The fields an interval literal names; Spark types the literal by them,
/// whatever its values (`INTERVAL 0 MONTH 1 DAY` mixes fields).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Named {
    /// The lowest and the highest year-month field.
    pub year_month: Option<(YearMonthField, YearMonthField)>,
    /// The lowest and the highest day-time field.
    pub day_time: Option<(DayTimeField, DayTimeField)>,
}

impl Named {
    fn add_year_month(&mut self, f: YearMonthField) {
        let (lo, hi) = self.year_month.unwrap_or((f, f));
        self.year_month = Some((lo.min(f), hi.max(f)));
    }

    fn add_day_time(&mut self, f: DayTimeField) {
        let (lo, hi) = self.day_time.unwrap_or((f, f));
        self.day_time = Some((lo.min(f), hi.max(f)));
    }

    /// The fields of the range `from TO to`, in Spark's unit names.
    #[must_use]
    pub fn of_range(from: &str, to: &str) -> Self {
        let mut named = Self::default();
        for unit in [from, to] {
            named.add_unit(unit);
        }
        named
    }

    /// Records a unit of Spark's interval grammar; a week counts as days, and
    /// milliseconds and microseconds as seconds.
    fn add_unit(&mut self, unit: &str) {
        match unit {
            "year" => self.add_year_month(YearMonthField::Year),
            "month" => self.add_year_month(YearMonthField::Month),
            "week" | "day" => self.add_day_time(DayTimeField::Day),
            "hour" => self.add_day_time(DayTimeField::Hour),
            "minute" => self.add_day_time(DayTimeField::Minute),
            _ => self.add_day_time(DayTimeField::Second),
        }
    }

    /// Whether it names fields of both families.
    #[must_use]
    pub const fn mixed(self) -> bool {
        self.year_month.is_some() && self.day_time.is_some()
    }
}

/// Spark's `UTF8String.trimAll`: whitespace and control characters.
pub fn trim_all(s: &str) -> &str {
    s.trim_matches(|c: char| {
        c.is_control() || (c.is_whitespace() && !matches!(c, '\u{A0}' | '\u{2007}' | '\u{202F}'))
    })
}

/// A state of [`parse_string`].
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

/// Spark's `IntervalUtils.stringToInterval`, with the units the string names:
/// `[interval] [+|-]value unit ...`, units `year`, `month`, `week`, `day`,
/// `hour`, `minute`, `second`, `millisecond`, `microsecond` (each optionally
/// plural); only seconds take a fraction (up to nanoseconds).
///
/// # Errors
/// Spark's `INVALID_INTERVAL_FORMAT.*`.
pub fn parse_string(input: &str) -> Result<(Calendar, Named), KrishivError> {
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
        // `i` is always on a character boundary; `get` keeps a slip from
        // panicking.
        let start = s
            .get(..i.min(s.len()))
            .and_then(|head| head.rfind(char::is_whitespace))
            .map_or(0, |p| p + 1);
        s.get(start..)
            .unwrap_or_default()
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
    let mut named = Named::default();
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
                let rest = s.get(i..).unwrap_or_default();
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
                let units: [(&str, i64); 9] = [
                    ("year", 0),
                    ("month", 1),
                    ("week", 2),
                    ("day", 3),
                    ("hour", MICROS_PER_HOUR),
                    ("minute", MICROS_PER_MINUTE),
                    ("second", MICROS_PER_SECOND),
                    ("millisecond", 1000),
                    ("microsecond", 1),
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
                    _ => add_micros(&mut c, value, *per)?,
                }
                named.add_unit(unit);
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
        State::UnitSuffix | State::UnitEnd | State::TrimBeforeSign => Ok((c, named)),
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

    #[test]
    fn year_month_strings() {
        use YearMonthField::{Month, Year};
        assert_eq!(cast_to_year_month("-1-2", Year, Month).unwrap(), -14);
        assert_eq!(
            cast_to_year_month("INTERVAL -'-1-2' YEAR TO MONTH", Year, Month).unwrap(),
            14
        );
        assert_eq!(cast_to_year_month("5", Month, Month).unwrap(), 5);
        assert_eq!(cast_to_year_month("5", Year, Year).unwrap(), 60);
        for bad in ["1-12", "178956970-8"] {
            let e = cast_to_year_month(bad, Year, Month).unwrap_err();
            assert_eq!(
                e.error_class().name(),
                "INVALID_INTERVAL_FORMAT.INTERVAL_PARSING",
                "{bad}"
            );
        }
        let e = cast_to_year_month("1", Year, Month).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "INVALID_INTERVAL_FORMAT.UNMATCHED_FORMAT_STRING"
        );
    }

    #[test]
    fn day_time_strings() {
        use DayTimeField::{Day, Hour, Minute, Second};
        assert_eq!(
            cast_to_day_time("1 02:03:04.5", Day, Second).unwrap(),
            MICROS_PER_DAY + 2 * MICROS_PER_HOUR + 3 * MICROS_PER_MINUTE + 4_500_000
        );
        assert_eq!(
            cast_to_day_time("100:00", Hour, Minute).unwrap(),
            100 * MICROS_PER_HOUR
        );
        assert_eq!(
            cast_to_day_time("1.1234567", Second, Second).unwrap(),
            1_123_456
        );
        assert!(cast_to_day_time("106751992", Day, Day).unwrap() < 0);
        let class = |s, a, b| {
            cast_to_day_time(s, a, b)
                .unwrap_err()
                .error_class()
                .name()
                .to_owned()
        };
        assert_eq!(
            class("1 24", Day, Hour),
            "INVALID_INTERVAL_FORMAT.INTERVAL_PARSING"
        );
        assert_eq!(
            class("1.5", Minute, Minute),
            "INVALID_INTERVAL_FORMAT.UNMATCHED_FORMAT_STRING_WITH_NOTICE"
        );
        assert_eq!(
            class("INTERVAL '1' HOUR", Day, Day),
            "INVALID_INTERVAL_FORMAT.UNMATCHED_FORMAT_STRING_WITH_NOTICE"
        );
    }
}
