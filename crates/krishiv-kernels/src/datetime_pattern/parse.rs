// SPDX-License-Identifier: Apache-2.0
//! Reading a string with a compiled pattern: `java.time`'s strict,
//! case-insensitive parse, then Spark's resolution of missing fields.

use super::format::{MICROS_PER_DAY, MONTHS};
use super::{Field, Item, Number, OffsetStyle, Part, Sign, Text};
use crate::cast::datetime::days_from_civil;

/// The zone a string named.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ParsedZone {
    None,
    /// Seconds east of UTC.
    Offset(i32),
    /// A region id or zone name (needs the time-zone database).
    Region(String),
}

/// The raw fields of one parse (`java.time`'s `Parsed`).
#[derive(Debug, Clone, Default)]
struct Raw {
    year: Option<i64>,
    year_of_era: Option<i64>,
    era: Option<i64>,
    month: Option<i64>,
    day: Option<i64>,
    day_of_year: Option<i64>,
    hour_of_day: Option<i64>,
    clock_hour_of_day: Option<i64>,
    hour_of_ampm: Option<i64>,
    clock_hour_of_ampm: Option<i64>,
    ampm: Option<i64>,
    minute: Option<i64>,
    second: Option<i64>,
    nanos: Option<i64>,
    zone: Option<ParsedZone>,
}

/// How a parse failed, as Spark's `CANNOT_PARSE_TIMESTAMP` message renders
/// it. `java.time` reports the first failure its matchers meet: [`Fail::At`]
/// for a matcher that does not match, [`Fail::Trailing`] for text after a
/// full match. A full match whose fields do not resolve raises inside
/// `DateTimeFormatter.parse` ([`Fail::InParse`], rendered after
/// `Text 's' could not be parsed: `), unless Spark's own post-parse step
/// (missing-field defaults, `LocalDate.ofYearDay`, the day-of-month
/// cross-check) rejects it ([`Fail::PostParse`], rendered bare).
/// [`Fail::Overflow`] is a local date-time past the microsecond range
/// (Java's `ArithmeticException`, which even `try_to_*` does not catch).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Fail {
    /// No match at this character index.
    At(usize),
    /// The pattern matched but text remains from this index.
    Trailing(usize),
    /// The text matched but `java.time` strict resolution rejects it;
    /// Java's cause text.
    InParse(String),
    /// The text matched and resolved but Spark's post-parse step rejects
    /// it; the bare cause text.
    PostParse(String),
    /// The local date-time is past the microsecond range.
    Overflow,
}

/// Stores `v` in `slot` (`java.time`'s `setParsedField`): a differing value
/// already there fails the match at the field's start `pos`.
fn set(slot: Option<&mut Option<i64>>, v: i64, pos: usize) -> Result<(), Fail> {
    match slot {
        None => Err(Fail::At(pos)),
        Some(s) => match *s {
            Some(old) if old != v => Err(Fail::At(pos)),
            _ => {
                *s = Some(v);
                Ok(())
            }
        },
    }
}

impl Raw {
    fn slot(&mut self, f: Field) -> Option<&mut Option<i64>> {
        Some(match f {
            Field::Year => &mut self.year,
            Field::YearOfEra => &mut self.year_of_era,
            Field::Month => &mut self.month,
            Field::Day => &mut self.day,
            Field::DayOfYear => &mut self.day_of_year,
            Field::HourOfDay => &mut self.hour_of_day,
            Field::ClockHourOfDay => &mut self.clock_hour_of_day,
            Field::HourOfAmPm => &mut self.hour_of_ampm,
            Field::ClockHourOfAmPm => &mut self.clock_hour_of_ampm,
            Field::Minute => &mut self.minute,
            Field::Second => &mut self.second,
            // Not parseable (Spark rejects `Q`, `F` for parsing).
            Field::Quarter | Field::AlignedWeekOfMonth => return None,
        })
    }
}

/// A resolved parse: the local date and time, and the zone.
#[derive(Debug, Clone)]
pub(crate) struct Resolved {
    days: i64,
    time_micros: i64,
    pub(super) zone: ParsedZone,
}

impl Resolved {
    pub(crate) fn days(&self) -> Option<i64> {
        Some(self.days)
    }

    pub(crate) fn local_micros(&self) -> Option<i64> {
        self.days
            .checked_mul(MICROS_PER_DAY)?
            .checked_add(self.time_micros)
    }

    pub(crate) fn zone(&self) -> &ParsedZone {
        &self.zone
    }
}

/// Parses all of `s`; `None` if it does not match or its fields do not
/// resolve to a date-time.
pub(super) fn parse(items: &[Item], s: &str) -> Option<Resolved> {
    parse_detailed(items, s).ok()
}

/// As [`parse`], with Java's failure detail. [`Fail::Overflow`] is a local
/// date-time past the microsecond range.
pub(super) fn parse_detailed(items: &[Item], s: &str) -> Result<Resolved, Fail> {
    let text: Vec<char> = s.chars().collect();
    let mut raw = Raw::default();
    let end = parse_items(items, &text, 0, &mut raw)?;
    if end != text.len() {
        return Err(Fail::Trailing(end));
    }
    let resolved = resolve_detailed(raw)?;
    // Java's `ArithmeticException` past the microsecond range.
    if resolved.local_micros().is_none() {
        return Err(Fail::Overflow);
    }
    Ok(resolved)
}

fn parse_items(items: &[Item], s: &[char], mut pos: usize, raw: &mut Raw) -> Result<usize, Fail> {
    for item in items {
        pos = match item {
            Item::Literal(lit) => literal(s, pos, lit).ok_or(Fail::At(pos))?,
            Item::Number(n) => {
                let (v, end) = number(n, s, pos).ok_or(Fail::At(pos))?;
                set(raw.slot(n.field), v, pos)?;
                end
            }
            Item::Reduced(field) => {
                let (v, end) = fixed_digits(s, pos, 2).ok_or(Fail::At(pos))?;
                set(raw.slot(*field), 2000 + v, pos)?;
                end
            }
            Item::Fraction { max, .. } => {
                let digits = s[pos..]
                    .iter()
                    .take(*max)
                    .take_while(|c| c.is_ascii_digit())
                    .count();
                if digits == 0 {
                    return Err(Fail::At(pos));
                }
                let mut v: i64 = 0;
                for c in &s[pos..pos + digits] {
                    v = v * 10 + i64::from(c.to_digit(10).ok_or(Fail::At(pos))?);
                }
                for _ in digits..9 {
                    v *= 10;
                }
                set(Some(&mut raw.nanos), v, pos)?;
                pos + digits
            }
            Item::Text { text, full } => {
                let (v, end) = text_field(*text, *full, s, pos).ok_or(Fail::At(pos))?;
                let slot = match text {
                    Text::Era => Some(&mut raw.era),
                    Text::Month => Some(&mut raw.month),
                    Text::AmPm => Some(&mut raw.ampm),
                    Text::DayOfWeek | Text::Quarter => None,
                };
                set(slot, v, pos)?;
                end
            }
            Item::Offset(style) => {
                let (o, end) = offset(style, s, pos).ok_or(Fail::At(pos))?;
                set_zone(raw, ParsedZone::Offset(o), pos)?;
                end
            }
            Item::Localized { full } => {
                let (o, end) = localized(*full, s, pos).ok_or(Fail::At(pos))?;
                set_zone(raw, ParsedZone::Offset(o), pos)?;
                end
            }
            Item::ZoneId => {
                let (z, end) = zone_id(s, pos, false).ok_or(Fail::At(pos))?;
                set_zone(raw, z, pos)?;
                end
            }
            Item::ZoneName { .. } => {
                let (z, end) = zone_id(s, pos, true).ok_or(Fail::At(pos))?;
                set_zone(raw, z, pos)?;
                end
            }
            Item::Optional(inner) => {
                let mut trial = raw.clone();
                match parse_items(inner, s, pos, &mut trial) {
                    Ok(end) => {
                        *raw = trial;
                        end
                    }
                    // An optional section that does not match is skipped,
                    // as `java.time` backtracks it.
                    Err(_) => pos,
                }
            }
        };
    }
    Ok(pos)
}

/// Stores `z` (`java.time`'s `setParsed(ZoneId)` overwrites a region, while
/// a second offset conflicts with the first; an offset always wins over a
/// region, as `resolveInstant` prefers it).
fn set_zone(raw: &mut Raw, z: ParsedZone, pos: usize) -> Result<(), Fail> {
    let next = match (&raw.zone, z) {
        (Some(ParsedZone::Region(_)), ParsedZone::Region(new)) => ParsedZone::Region(new),
        (Some(ParsedZone::Region(_)), ParsedZone::Offset(o)) => ParsedZone::Offset(o),
        (Some(ParsedZone::Offset(_)), ParsedZone::Region(_)) => {
            return Ok(());
        }
        (Some(old), new) if *old != new => return Err(Fail::At(pos)),
        (_, new) => new,
    };
    raw.zone = Some(next);
    Ok(())
}

fn eq_ignore_case(a: char, b: char) -> bool {
    a == b || a.to_lowercase().eq(b.to_lowercase())
}

/// `lit` at `pos`, case-insensitively.
fn literal(s: &[char], pos: usize, lit: &str) -> Option<usize> {
    let mut at = pos;
    for c in lit.chars() {
        if !eq_ignore_case(*s.get(at)?, c) {
            return None;
        }
        at += 1;
    }
    Some(at)
}

fn digit(s: &[char], at: usize) -> Option<i64> {
    s.get(at).and_then(|c| c.to_digit(10)).map(i64::from)
}

/// Exactly `width` digits.
fn fixed_digits(s: &[char], pos: usize, width: usize) -> Option<(i64, usize)> {
    let mut v = 0;
    for i in 0..width {
        v = v * 10 + digit(s, pos + i)?;
    }
    Some((v, pos + width))
}

/// `java.time`'s strict `NumberPrinterParser.parse`.
fn number(n: &Number, s: &[char], start: usize) -> Option<(i64, usize)> {
    let (negative, positive, pos) = match s.get(start) {
        Some('+') => {
            let ok = match n.sign {
                Sign::Normal | Sign::NotNegative => false,
                Sign::ExceedsPad => true,
            };
            if !ok {
                return None;
            }
            (false, true, start + 1)
        }
        Some('-') => {
            if n.sign == Sign::NotNegative {
                return None;
            }
            (true, false, start + 1)
        }
        _ => (false, false, start),
    };
    let min_end = pos + n.min;
    if min_end > s.len() {
        return None;
    }
    let mut eff_max = n.max + n.subsequent.unwrap_or(0);
    let mut end = pos;
    let mut total: i128 = 0;
    for pass in 0..2 {
        let max_end = (pos + eff_max).min(s.len());
        end = pos;
        total = 0;
        while end < max_end {
            let Some(d) = digit(s, end) else {
                if end < min_end {
                    return None;
                }
                break;
            };
            // Saturates: a pass-0 total is discarded, and a saturated
            // final total fails the `i64` conversion below.
            total = total.saturating_mul(10).saturating_add(i128::from(d));
            end += 1;
        }
        match n.subsequent {
            Some(w) if w > 0 && pass == 0 => {
                eff_max = n.min.max((end - pos).saturating_sub(w));
            }
            _ => break,
        }
    }
    if end < min_end {
        return None;
    }
    let len = end - pos;
    if negative {
        if total == 0 {
            return None;
        }
        total = -total;
    } else if n.sign == Sign::ExceedsPad {
        if positive && len <= n.min {
            return None;
        }
        if !positive && len > n.min {
            return None;
        }
    }
    Some((i64::try_from(total).ok()?, end))
}

fn text_field(text: Text, full: bool, s: &[char], pos: usize) -> Option<(i64, usize)> {
    let candidates: Vec<(String, i64)> = match text {
        Text::Era if full => vec![("Before Christ".into(), 0), ("Anno Domini".into(), 1)],
        Text::Era => vec![("BC".into(), 0), ("AD".into(), 1)],
        Text::Month => MONTHS
            .iter()
            .zip(1..)
            .map(|(m, i)| {
                (
                    if full {
                        (*m).to_owned()
                    } else {
                        m[..3].to_owned()
                    },
                    i,
                )
            })
            .collect(),
        Text::AmPm => vec![("AM".into(), 0), ("PM".into(), 1)],
        Text::DayOfWeek | Text::Quarter => return None,
    };
    candidates
        .iter()
        .filter_map(|(t, v)| literal(s, pos, t).map(|end| (*v, end)))
        .max_by_key(|(_, end)| *end)
}

/// Two digits after an optional `:` (when `colon`).
fn two_digits(s: &[char], pos: usize, colon: bool) -> Option<(i64, usize)> {
    let at = if colon {
        if s.get(pos) != Some(&':') {
            return None;
        }
        pos + 1
    } else {
        pos
    };
    fixed_digits(s, at, 2)
}

/// `java.time`'s strict `OffsetIdPrinterParser.parse`.
fn offset(style: &OffsetStyle, s: &[char], pos: usize) -> Option<(i32, usize)> {
    if pos >= s.len() {
        return None;
    }
    if let Some(end) = literal(s, pos, style.zero) {
        return Some((0, end));
    }
    let sign = match s.get(pos)? {
        '+' => 1,
        '-' => -1,
        _ => return None,
    };
    let (h, mut end) = fixed_digits(s, pos + 1, 2)?;
    let mut m = 0;
    let mut sec = 0;
    if style.minutes != Part::Absent {
        match two_digits(s, end, style.colon) {
            Some((v, e)) => {
                m = v;
                end = e;
                if style.seconds != Part::Absent {
                    match two_digits(s, end, style.colon) {
                        Some((v, e)) => {
                            sec = v;
                            end = e;
                        }
                        None if style.seconds == Part::Required => return None,
                        None => {}
                    }
                }
            }
            None if style.minutes == Part::Required => return None,
            None => {}
        }
    }
    if h > 23 || m > 59 || sec > 59 {
        return None;
    }
    let total = sign * (h * 3600 + m * 60 + sec);
    if total.abs() > 18 * 3600 {
        return None;
    }
    Some((i32::try_from(total).ok()?, end))
}

/// `GMT`, `GMT+5`, `GMT+5:30` (short) or `GMT+05:30` (full).
fn localized(full: bool, s: &[char], pos: usize) -> Option<(i32, usize)> {
    let mut at = literal(s, pos, "GMT")?;
    let sign = match s.get(at) {
        Some('+') => 1,
        Some('-') => -1,
        _ => return Some((0, at)),
    };
    at += 1;
    let end = s.len();
    let (mut m, mut sec) = (0, 0);
    let h;
    if full {
        let (hh, e) = fixed_digits(s, at, 2)?;
        if s.get(e) != Some(&':') {
            return None;
        }
        let (mm, e) = fixed_digits(s, e + 1, 2)?;
        (h, m, at) = (hh, mm, e);
        if at + 2 < end
            && s[at] == ':'
            && let (Some(a), Some(b)) = (digit(s, at + 1), digit(s, at + 2))
        {
            sec = a * 10 + b;
            at += 3;
        }
    } else {
        let mut hh = digit(s, at)?;
        at += 1;
        if at < end {
            if let Some(h2) = digit(s, at) {
                hh = hh * 10 + h2;
                at += 1;
            }
            if at + 2 < end
                && s[at] == ':'
                && let (Some(a), Some(b)) = (digit(s, at + 1), digit(s, at + 2))
            {
                m = a * 10 + b;
                at += 3;
                if at + 2 < end
                    && s[at] == ':'
                    && let (Some(a), Some(b)) = (digit(s, at + 1), digit(s, at + 2))
                {
                    sec = a * 10 + b;
                    at += 3;
                }
            }
        }
        h = hh;
    }
    if h > 23 || m > 59 || sec > 59 {
        return None;
    }
    // `ZoneOffset.ofTotalSeconds` rejects beyond +-18:00.
    let total = sign * (h * 3600 + m * 60 + sec);
    if total.abs() > 18 * 3600 {
        return None;
    }
    Some((i32::try_from(total).ok()?, at))
}

/// `java.time`'s `ZoneIdPrinterParser` (`names` for `z`, which also reads
/// zone names): fixed offsets, `UTC`/`GMT`/`UT` prefixes, `Z`; any other
/// id is a region.
fn zone_id(s: &[char], pos: usize, names: bool) -> Option<(ParsedZone, usize)> {
    let len = s.len();
    let next = *s.get(pos)?;
    if names {
        for name in [
            "Coordinated Universal Time",
            "Greenwich Mean Time",
            "UTC",
            "GMT",
        ] {
            if let Some(end) = literal(s, pos, name) {
                return Some((ParsedZone::Offset(0), end));
            }
        }
    }
    if next == '+' || next == '-' {
        return offset_based(s, pos, true);
    }
    if len >= pos + 2 {
        let n2 = s[pos + 1];
        if eq_ignore_case(next, 'U') && eq_ignore_case(n2, 'T') {
            if len >= pos + 3 && eq_ignore_case(s[pos + 2], 'C') {
                if !names {
                    return offset_based(s, pos + 3, false);
                }
            } else {
                return offset_based(s, pos + 2, false);
            }
        } else if eq_ignore_case(next, 'G')
            && len >= pos + 3
            && eq_ignore_case(n2, 'M')
            && eq_ignore_case(s[pos + 2], 'T')
        {
            if len >= pos + 4 && s[pos + 3] == '0' {
                return Some((ParsedZone::Offset(0), pos + 4));
            }
            return offset_based(s, pos + 3, false);
        }
    }
    // A region id or zone name: letters, digits and `/_-+~.`.
    let end = s[pos..]
        .iter()
        .take_while(|c| c.is_ascii_alphanumeric() || "/_-+~.".contains(**c))
        .count();
    if end == 0 {
        return None;
    }
    if end == 1 && eq_ignore_case(next, 'Z') {
        return Some((ParsedZone::Offset(0), pos + 1));
    }
    Some((
        ParsedZone::Region(s[pos..pos + end].iter().collect()),
        pos + end,
    ))
}

/// `ZoneIdPrinterParser.parseOffsetBased`: a prefix (`UTC`, `GMT`, `UT`, or
/// none) and an optional `+HH:MM:ss` offset.
fn offset_based(s: &[char], pos: usize, bare: bool) -> Option<(ParsedZone, usize)> {
    if pos >= s.len() || s[pos] == '0' || eq_ignore_case(s[pos], 'Z') {
        return if bare {
            None
        } else {
            Some((ParsedZone::Offset(0), pos))
        };
    }
    let style = OffsetStyle {
        colon: true,
        minutes: Part::Required,
        seconds: Part::Optional,
        zero: if bare { "Z" } else { "0" },
    };
    match offset(&style, s, pos) {
        Some((o, end)) => Some((ParsedZone::Offset(o), end)),
        None if bare => None,
        None => Some((ParsedZone::Offset(0), pos)),
    }
}

/// The offset of a fixed-offset zone id text after any `UTC`/`GMT`/`UT`
/// prefix (`ZoneOffset.of` forms: `+h`, `+hh`, `+hh:mm`, `+hhmm`,
/// `+hh:mm:ss`, `+hhmmss`), or `None`.
pub(super) fn offset_id(rest: &str) -> Option<i32> {
    if rest.is_empty() {
        return Some(0);
    }
    // Byte slicing below needs ASCII (any other text is not an offset).
    if !rest.is_ascii() {
        return None;
    }
    let (sign, digits) = match rest.as_bytes().first()? {
        b'+' => (1, &rest[1..]),
        b'-' => (-1, &rest[1..]),
        _ => return None,
    };
    let num = |t: &str| -> Option<i32> {
        if t.len() == 2 && t.bytes().all(|b| b.is_ascii_digit()) {
            t.parse().ok()
        } else {
            None
        }
    };
    let (h, m, sec) = match digits.len() {
        1 if digits.as_bytes()[0].is_ascii_digit() => (digits.parse().ok()?, 0, 0),
        2 => (num(digits)?, 0, 0),
        4 => (num(&digits[..2])?, num(&digits[2..])?, 0),
        5 if &digits[2..3] == ":" => (num(&digits[..2])?, num(&digits[3..])?, 0),
        6 => (num(&digits[..2])?, num(&digits[2..4])?, num(&digits[4..])?),
        8 if &digits[2..3] == ":" && &digits[5..6] == ":" => {
            (num(&digits[..2])?, num(&digits[3..5])?, num(&digits[6..])?)
        }
        _ => return None,
    };
    if h > 18 || m > 59 || sec > 59 {
        return None;
    }
    let total = sign * (h * 3600 + m * 60 + sec);
    (total.abs() <= 18 * 3600).then_some(total)
}

/// `java.time`'s strict resolution, then Spark's defaults for missing
/// fields (`DateTimeFormatterHelper.toLocalDate` / `toLocalTime`).
/// [`Fail::InParse`] is a strict-resolution failure inside
/// `DateTimeFormatter.parse`; [`Fail::PostParse`] is Spark's post-parse
/// step rejecting the resolved fields.
fn resolve_detailed(mut r: Raw) -> Result<Resolved, Fail> {
    let (date, time) = resolve_java(&mut r).map_err(Fail::InParse)?;
    let days = spark_date(&r, date).map_err(Fail::PostParse)?;
    Ok(Resolved {
        days,
        time_micros: spark_time(&r, time),
        zone: r.zone.unwrap_or(ParsedZone::None),
    })
}

/// `java.time`'s strict resolution (`Parsed.resolve` with
/// `ResolverStyle.STRICT`): the resolved date and time of day, each
/// `None` for insufficient fields. Failures carry Java's cause text.
fn resolve_java(r: &mut Raw) -> Result<(Option<i64>, Option<i64>), String> {
    resolve_year_of_era(r)?;
    let date = resolve_java_date(r)?;
    let time = resolve_java_time(r)?;
    cross_check_date(r, date)?;
    cross_check_time(r, time)?;
    Ok((date, time))
}

/// `IsoChronology.resolveYearOfEra` in strict mode: with an era the year of
/// era becomes the proleptic year, without one it is left for Spark's
/// defaults.
fn resolve_year_of_era(r: &mut Raw) -> Result<(), String> {
    if let Some(yoe) = r.year_of_era {
        check_valid("YearOfEra", yoe, "1 - 999999999", 1, 999_999_999)?;
        if let Some(era) = r.era {
            if era != 0 && era != 1 {
                return Err(format!("Invalid value for era: {era}"));
            }
            r.year = Some(if era == 1 { yoe } else { 1 - yoe });
        }
    }
    Ok(())
}

/// `IsoChronology.resolveDate` in strict mode, without week fields (Spark
/// rejects those patterns): a year-month-day, or a year with a day of
/// year. Anything less resolves nothing here; Spark defaults it later.
fn resolve_java_date(r: &mut Raw) -> Result<Option<i64>, String> {
    if let (Some(y), Some(m), Some(d)) = (r.year, r.month, r.day) {
        let days = ymd_days(y, m, d)?;
        r.year = None;
        r.month = None;
        r.day = None;
        return Ok(Some(days));
    }
    if let (Some(y), Some(doy)) = (r.year, r.day_of_year) {
        check_valid(
            "Year",
            y,
            "-999999999 - 999999999",
            -999_999_999,
            999_999_999,
        )?;
        let days = of_year_day(y, doy)?;
        r.year = None;
        r.day_of_year = None;
        return Ok(Some(days));
    }
    Ok(None)
}

/// `Parsed.resolveTimeFields` plus the strict half of
/// `resolveTimeLenient`: the micros since midnight, or `None` when the
/// fields cannot make a time yet. Consumed fields are removed, as
/// `java.time` removes them.
fn resolve_java_time(r: &mut Raw) -> Result<Option<i64>, String> {
    if let Some(k) = r.clock_hour_of_day.take() {
        check_valid("ClockHourOfDay", k, "1 - 24", 1, 24)?;
        let v = if k == 24 { 0 } else { k };
        if r.hour_of_day.is_some_and(|h| h != v) {
            return Err(conflict_while(
                "HourOfDay",
                r.hour_of_day,
                v,
                "ClockHourOfDay",
            ));
        }
        r.hour_of_day = Some(v);
    }
    if let Some(h) = r.clock_hour_of_ampm.take() {
        check_valid("ClockHourOfAmPm", h, "1 - 12", 1, 12)?;
        let v = if h == 12 { 0 } else { h };
        if r.hour_of_ampm.is_some_and(|old| old != v) {
            return Err(conflict_while(
                "HourOfAmPm",
                r.hour_of_ampm,
                v,
                "ClockHourOfAmPm",
            ));
        }
        r.hour_of_ampm = Some(v);
    }
    if r.ampm.is_some() && r.hour_of_ampm.is_some() {
        let (ap, hap) = (
            r.ampm.take().unwrap_or(0),
            r.hour_of_ampm.take().unwrap_or(0),
        );
        check_valid("AmPmOfDay", ap, "0 - 1", 0, 1)?;
        check_valid("HourOfAmPm", hap, "0 - 11", 0, 11)?;
        let v = ap * 12 + hap;
        if r.hour_of_day.is_some_and(|h| h != v) {
            return Err(conflict_while("HourOfDay", r.hour_of_day, v, "AmPmOfDay"));
        }
        r.hour_of_day = Some(v);
    }
    let mut time = None;
    if let Some(hod) = r.hour_of_day {
        // A minute-less second or nano cannot default (`resolveTimeLenient`
        // leaves the time unresolved).
        let dangling = (r.minute.is_none() && (r.second.is_some() || r.nanos.is_some()))
            || (r.minute.is_some() && r.second.is_none() && r.nanos.is_some());
        if !dangling {
            time = Some(time_of(
                hod,
                r.minute.unwrap_or(0),
                r.second.unwrap_or(0),
                r.nanos.unwrap_or(0),
            )?);
            r.hour_of_day = None;
            r.minute = None;
            r.second = None;
            r.nanos = None;
        }
    }
    // Strict mode validates the time fields left over, whether or not a
    // time was built.
    for (field, range, lo, hi, v) in [
        ("HourOfDay", "0 - 23", 0, 23, r.hour_of_day),
        ("MinuteOfHour", "0 - 59", 0, 59, r.minute),
        ("SecondOfMinute", "0 - 59", 0, 59, r.second),
        ("NanoOfSecond", "0 - 999999999", 0, 999_999_999, r.nanos),
        ("AmPmOfDay", "0 - 1", 0, 1, r.ampm),
        ("HourOfAmPm", "0 - 11", 0, 11, r.hour_of_ampm),
    ] {
        if let Some(v) = v {
            check_valid(field, v, range, lo, hi)?;
        }
    }
    Ok(time)
}

/// `Parsed.crossCheck` over the date: leftover month, day and day-of-year
/// fields must agree with the resolved `date` (days since the epoch).
/// Field order is fixed (month, day, day of year); `java.time` iterates a
/// hash map here.
fn cross_check_date(r: &Raw, date: Option<i64>) -> Result<(), String> {
    let Some(days) = date else { return Ok(()) };
    let (year, month, day) = crate::cast::datetime::civil_from_days(days);
    let day_of_year = days
        .checked_sub(days_from_civil(year, 1, 1).unwrap_or(days))
        .and_then(|d| d.checked_add(1));
    let target = iso_date(days);
    if r.month.is_some_and(|m| m != i64::from(month)) {
        return Err(format!(
            "Conflict found: Field MonthOfYear {} differs from MonthOfYear {} derived from {target}.",
            i64::from(month),
            r.month.unwrap_or(0)
        ));
    }
    if r.day.is_some_and(|d| d != i64::from(day)) {
        return Err(format!(
            "Conflict found: Field DayOfMonth {} differs from DayOfMonth {} derived from {target}.",
            i64::from(day),
            r.day.unwrap_or(0)
        ));
    }
    if let (Some(doy), Some(derived)) = (r.day_of_year, day_of_year)
        && doy != derived
    {
        return Err(format!(
            "Conflict found: Field DayOfYear {derived} differs from DayOfYear {doy} derived from {target}."
        ));
    }
    Ok(())
}

/// `Parsed.crossCheck` over the time: a leftover AM/PM or hour-of-AM/PM
/// must agree with the resolved `time` (micros since midnight).
fn cross_check_time(r: &Raw, time: Option<i64>) -> Result<(), String> {
    let Some(t) = time else { return Ok(()) };
    let target = iso_time(t);
    let hour = t.div_euclid(3_600_000_000);
    if r.ampm.is_some_and(|ap| hour.div_euclid(12) != ap) {
        return Err(format!(
            "Conflict found: Field AmPmOfDay {} differs from AmPmOfDay {} derived from {target}.",
            hour.div_euclid(12),
            r.ampm.unwrap_or(0)
        ));
    }
    if r.hour_of_ampm.is_some_and(|hoa| hour.rem_euclid(12) != hoa) {
        return Err(format!(
            "Conflict found: Field HourOfAmPm {} differs from HourOfAmPm {} derived from {target}.",
            hour.rem_euclid(12),
            r.hour_of_ampm.unwrap_or(0)
        ));
    }
    Ok(())
}

/// Spark's `toLocalDate`: the resolved `date`, else the year (1970 without
/// one) with a day of year or month and day (January 1st without them).
/// Failures are Spark's own, rendered bare.
fn spark_date(r: &Raw, date: Option<i64>) -> Result<i64, String> {
    if let Some(days) = date {
        return Ok(days);
    }
    let year = r.year.unwrap_or(1970);
    if let Some(doy) = r.day_of_year {
        check_valid(
            "Year",
            year,
            "-999999999 - 999999999",
            -999_999_999,
            999_999_999,
        )?;
        let days = of_year_day(year, doy)?;
        let (_, month, day) = crate::cast::datetime::civil_from_days(days);
        let target = iso_date(days);
        if r.month.is_some_and(|m| m != i64::from(month)) {
            return Err(format!(
                "Conflict found: Field MonthOfYear {} differs from MonthOfYear {} derived from {target}.",
                r.month.unwrap_or(0),
                i64::from(month)
            ));
        }
        if r.day.is_some_and(|d| d != i64::from(day)) {
            return Err(format!(
                "Conflict found: Field DayOfMonth {} differs from DayOfMonth {} derived from {target}.",
                r.day.unwrap_or(0),
                i64::from(day)
            ));
        }
        return Ok(days);
    }
    let (y, m, d) = (
        i32::try_from(year).map_err(|_| {
            format!("Invalid value for Year (valid values -999999999 - 999999999): {year}")
        })?,
        i32::try_from(r.month.unwrap_or(1)).map_err(|_| {
            format!(
                "Invalid value for MonthOfYear (valid values 1 - 12): {}",
                r.month.unwrap_or(1)
            )
        })?,
        i32::try_from(r.day.unwrap_or(1)).map_err(|_| {
            format!(
                "Invalid value for DayOfMonth (valid values 1 - 28/31): {}",
                r.day.unwrap_or(1)
            )
        })?,
    );
    if let Some(message) = crate::datetime::calendar::invalid_date(y, m, d) {
        return Err(message);
    }
    // Validated 1..=12 and 1..=31 above.
    days_from_civil(i64::from(y), m.cast_unsigned(), d.cast_unsigned()).ok_or_else(|| {
        crate::datetime::calendar::invalid_date(y, m, d)
            .unwrap_or_else(|| "Invalid date".to_owned())
    })
}

/// Spark's `toLocalTime`: the resolved `time`, else the hour (an AM/PM hour
/// counts as AM, a lone PM is noon) with minutes, seconds and nanos.
fn spark_time(r: &Raw, time: Option<i64>) -> i64 {
    if let Some(t) = time {
        return t;
    }
    let hour = r
        .hour_of_day
        .or(r.hour_of_ampm)
        .unwrap_or(if r.ampm == Some(1) { 12 } else { 0 });
    // Every field was range-checked in strict resolution, so the sum fits.
    hour * 3_600_000_000
        + r.minute.unwrap_or(0) * 60_000_000
        + r.second.unwrap_or(0) * 1_000_000
        + r.nanos.unwrap_or(0) / 1000
}

/// `ChronoField.checkValidValue`: Java's range text, or the value.
/// `range` is the field's `ValueRange` rendering (`1 - 28/31`).
fn check_valid(field: &str, v: i64, range: &str, lo: i64, hi: i64) -> Result<(), String> {
    if (lo..=hi).contains(&v) {
        Ok(())
    } else {
        Err(format!(
            "Invalid value for {field} (valid values {range}): {v}"
        ))
    }
}

/// `Parsed.updateCheckConflict` for a time field: Java's text, including
/// its double space before the resolving field.
fn conflict_while(field: &str, old: Option<i64>, new: i64, target: &str) -> String {
    format!(
        "Conflict found: {field} {} differs from {field} {new} while resolving  {target}",
        old.unwrap_or(new)
    )
}

/// `LocalDate.of(year, month, day)`: the days, or Java's message.
fn ymd_days(year: i64, month: i64, day: i64) -> Result<i64, String> {
    let out_of_range = |field: &str, range: &str, v: i64| {
        format!("Invalid value for {field} (valid values {range}): {v}")
    };
    let (y, m, d) = (
        i32::try_from(year).map_err(|_| out_of_range("Year", "-999999999 - 999999999", year))?,
        i32::try_from(month).map_err(|_| out_of_range("MonthOfYear", "1 - 12", month))?,
        i32::try_from(day).map_err(|_| out_of_range("DayOfMonth", "1 - 28/31", day))?,
    );
    if let Some(message) = crate::datetime::calendar::invalid_date(y, m, d) {
        return Err(message);
    }
    // Validated 1..=12 and 1..=31 above.
    days_from_civil(i64::from(y), m.cast_unsigned(), d.cast_unsigned()).ok_or_else(|| {
        crate::datetime::calendar::invalid_date(y, m, d)
            .unwrap_or_else(|| "Invalid date".to_owned())
    })
}

/// `LocalDate.ofYearDay(year, day_of_year)`: the days, or Java's message.
fn of_year_day(year: i64, doy: i64) -> Result<i64, String> {
    check_valid("DayOfYear", doy, "1 - 365/366", 1, 366)?;
    if doy == 366 && !is_leap(year) {
        return Err(format!(
            "Invalid date 'DayOfYear 366' as '{year}' is not a leap year"
        ));
    }
    days_from_civil(year, 1, 1)
        .and_then(|jan1| jan1.checked_add(doy - 1))
        .ok_or_else(|| format!("Invalid date 'DayOfYear {doy}' as '{year}' is not a leap year"))
}

/// `LocalTime.of(hour, minute, second, nano)`: the micros since midnight,
/// or Java's message.
fn time_of(hour: i64, minute: i64, second: i64, nano: i64) -> Result<i64, String> {
    check_valid("HourOfDay", hour, "0 - 23", 0, 23)?;
    check_valid("MinuteOfHour", minute, "0 - 59", 0, 59)?;
    check_valid("SecondOfMinute", second, "0 - 59", 0, 59)?;
    check_valid("NanoOfSecond", nano, "0 - 999999999", 0, 999_999_999)?;
    Ok(hour * 3_600_000_000 + minute * 60_000_000 + second * 1_000_000 + nano / 1000)
}

/// Whether `year` is a leap year in the proleptic Gregorian calendar.
fn is_leap(year: i64) -> bool {
    year.rem_euclid(4) == 0 && (year.rem_euclid(100) != 0 || year.rem_euclid(400) == 0)
}

/// A day number as `LocalDate.toString` renders it.
fn iso_date(days: i64) -> String {
    let mut out = String::new();
    crate::cast::datetime::format_date(days, &mut out);
    out
}

/// Micros since midnight as `LocalTime.toString` renders it.
fn iso_time(micros: i64) -> String {
    let (h, m, s, n) = (
        micros.div_euclid(3_600_000_000),
        micros.div_euclid(60_000_000).rem_euclid(60),
        micros.div_euclid(1_000_000).rem_euclid(60),
        micros.rem_euclid(1_000_000) * 1000,
    );
    let mut out = format!("{h:02}:{m:02}");
    if s != 0 || n != 0 {
        use std::fmt::Write as _;
        let _ = write!(out, ":{s:02}");
        if n != 0 {
            let digits = format!("{n:09}");
            let _ = write!(out, ".{}", digits.trim_end_matches('0'));
        }
    }
    out
}
