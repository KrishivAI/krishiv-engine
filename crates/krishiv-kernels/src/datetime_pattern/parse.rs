// SPDX-License-Identifier: Apache-2.0
//! Reading a string with a compiled pattern: `java.time`'s strict,
//! case-insensitive parse, then Spark's resolution of missing fields.

use super::format::{MICROS_PER_DAY, MONTHS};
use super::{Field, Item, Number, OffsetStyle, Part, Sign, Text};
use crate::cast::datetime::days_from_civil;

/// The zone a string named.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum ParsedZone {
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

fn set(slot: &mut Option<i64>, v: i64) -> Option<()> {
    match slot {
        Some(old) if *old != v => None,
        _ => {
            *slot = Some(v);
            Some(())
        }
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
pub(super) struct Resolved {
    days: i64,
    time_micros: i64,
    pub(super) zone: ParsedZone,
}

impl Resolved {
    pub(super) fn days(&self) -> Option<i64> {
        Some(self.days)
    }

    pub(super) fn local_micros(&self) -> Option<i64> {
        self.days
            .checked_mul(MICROS_PER_DAY)?
            .checked_add(self.time_micros)
    }
}

/// Parses all of `s`; `None` if it does not match or its fields do not
/// resolve to a date-time.
pub(super) fn parse(items: &[Item], s: &str) -> Option<Resolved> {
    let text: Vec<char> = s.chars().collect();
    let mut raw = Raw::default();
    let end = parse_items(items, &text, 0, &mut raw)?;
    if end != text.len() {
        return None;
    }
    resolve(raw)
}

fn parse_items(items: &[Item], s: &[char], mut pos: usize, raw: &mut Raw) -> Option<usize> {
    for item in items {
        pos = match item {
            Item::Literal(lit) => literal(s, pos, lit)?,
            Item::Number(n) => {
                let (v, end) = number(n, s, pos)?;
                set(raw.slot(n.field)?, v)?;
                end
            }
            Item::Reduced(field) => {
                let (v, end) = fixed_digits(s, pos, 2)?;
                set(raw.slot(*field)?, 2000 + v)?;
                end
            }
            Item::Fraction { max, .. } => {
                let digits = s[pos..]
                    .iter()
                    .take(*max)
                    .take_while(|c| c.is_ascii_digit())
                    .count();
                if digits == 0 {
                    return None;
                }
                let mut v: i64 = 0;
                for c in &s[pos..pos + digits] {
                    v = v * 10 + i64::from(c.to_digit(10)?);
                }
                for _ in digits..9 {
                    v *= 10;
                }
                set(&mut raw.nanos, v)?;
                pos + digits
            }
            Item::Text { text, full } => {
                let (v, end) = text_field(*text, *full, s, pos)?;
                let slot = match text {
                    Text::Era => &mut raw.era,
                    Text::Month => &mut raw.month,
                    Text::AmPm => &mut raw.ampm,
                    Text::DayOfWeek | Text::Quarter => return None,
                };
                set(slot, v)?;
                end
            }
            Item::Offset(style) => {
                let (o, end) = offset(style, s, pos)?;
                set_zone(raw, ParsedZone::Offset(o))?;
                end
            }
            Item::Localized { full } => {
                let (o, end) = localized(*full, s, pos)?;
                set_zone(raw, ParsedZone::Offset(o))?;
                end
            }
            Item::ZoneId => {
                let (z, end) = zone_id(s, pos, false)?;
                set_zone(raw, z)?;
                end
            }
            Item::ZoneName { .. } => {
                let (z, end) = zone_id(s, pos, true)?;
                set_zone(raw, z)?;
                end
            }
            Item::Optional(inner) => {
                let mut trial = raw.clone();
                match parse_items(inner, s, pos, &mut trial) {
                    Some(end) => {
                        *raw = trial;
                        end
                    }
                    None => pos,
                }
            }
        };
    }
    Some(pos)
}

fn set_zone(raw: &mut Raw, z: ParsedZone) -> Option<()> {
    match &raw.zone {
        Some(old) if *old != z => None,
        _ => {
            raw.zone = Some(z);
            Some(())
        }
    }
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

fn check(v: Option<i64>, lo: i64, hi: i64) -> Option<Option<i64>> {
    match v {
        Some(x) if !(lo..=hi).contains(&x) => None,
        other => Some(other),
    }
}

/// `java.time`'s strict resolution, then Spark's defaults for missing
/// fields (`DateTimeFormatterHelper.toLocalDate` / `toLocalTime`).
fn resolve(mut r: Raw) -> Option<Resolved> {
    const MAX_YEAR: i64 = 999_999_999;
    // Every parsed value must be in its field's range.
    check(r.year, -MAX_YEAR, MAX_YEAR)?;
    check(r.year_of_era, 1, MAX_YEAR + 1)?;
    check(r.month, 1, 12)?;
    check(r.day, 1, 31)?;
    check(r.day_of_year, 1, 366)?;
    check(r.hour_of_day, 0, 23)?;
    check(r.clock_hour_of_day, 1, 24)?;
    check(r.hour_of_ampm, 0, 11)?;
    check(r.clock_hour_of_ampm, 1, 12)?;
    check(r.minute, 0, 59)?;
    check(r.second, 0, 59)?;
    // Era and year of era: the proleptic year.
    if let (Some(era), Some(yoe)) = (r.era, r.year_of_era) {
        let y = if era == 1 { yoe } else { 1 - yoe };
        set(&mut r.year, y)?;
    }
    // Hours.
    if let Some(k) = r.clock_hour_of_day {
        set(&mut r.hour_of_day, if k == 24 { 0 } else { k })?;
    }
    if let Some(h) = r.clock_hour_of_ampm {
        set(&mut r.hour_of_ampm, if h == 12 { 0 } else { h })?;
    }
    if let (Some(ap), Some(h)) = (r.ampm, r.hour_of_ampm) {
        set(&mut r.hour_of_day, ap * 12 + h)?;
    }
    if let (Some(ap), Some(h)) = (r.ampm, r.hour_of_day)
        && h / 12 != ap
    {
        return None;
    }
    if let (Some(hod), Some(hoa)) = (r.hour_of_day, r.hour_of_ampm)
        && hod % 12 != hoa
    {
        return None;
    }
    // The date.
    let year = r.year.unwrap_or(1970);
    let days = match (r.month, r.day, r.day_of_year) {
        (m, d, Some(doy)) => {
            let jan1 = days_from_civil(year, 1, 1)?;
            let date = jan1 + doy - 1;
            // Day 366 of a common year is the next year's day 1.
            if days_from_civil(year + 1, 1, 1)? <= date {
                return None;
            }
            let (_, dm, dd) = crate::cast::datetime::civil_from_days(date);
            if m.is_some_and(|m| m != i64::from(dm)) || d.is_some_and(|d| d != i64::from(dd)) {
                return None;
            }
            date
        }
        (m, d, None) => days_from_civil(
            year,
            u32::try_from(m.unwrap_or(1)).ok()?,
            u32::try_from(d.unwrap_or(1)).ok()?,
        )?,
    };
    let hour = r
        .hour_of_day
        .or(r.hour_of_ampm)
        .unwrap_or(if r.ampm == Some(1) { 12 } else { 0 });
    let time_micros = hour * 3_600_000_000
        + r.minute.unwrap_or(0) * 60_000_000
        + r.second.unwrap_or(0) * 1_000_000
        + r.nanos.unwrap_or(0) / 1000;
    Some(Resolved {
        days,
        time_micros,
        zone: r.zone.unwrap_or(ParsedZone::None),
    })
}
