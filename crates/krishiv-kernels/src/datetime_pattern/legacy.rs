// SPDX-License-Identifier: Apache-2.0
//! The pre-3.0 datetime patterns of `spark.sql.legacy.timeParserPolicy =
//! LEGACY`: formatting is commons-lang's `FastDateFormat`, parsing is
//! `java.text.SimpleDateFormat` (non-lenient, so a month 13 or a February 30
//! is no timestamp), both over the Julian/Gregorian hybrid calendar.
//!
//! Spark rebases the proleptic value to the hybrid calendar keeping the
//! date-time labels, so a field that is the label itself (year, month, day,
//! hour, ...) prints as in the proleptic calendar; the calendar-dependent
//! ones (day of week, day of year, week of year) follow the Julian calendar
//! before 1582-10-15.
//!
//! Differences from [`super::DateTimePattern`]: no `[..]` optional sections
//! (`[` is a literal), `S` is the millisecond count padded to the run length,
//! `G` prints `AD`/`BC`, `u` is the day number of the week (Monday = 1),
//! `q`, `Q`, `V`, `O`, `x` and `X` runs over three are illegal, and parsing
//! stops where the pattern does: trailing text is ignored.
#![expect(
    clippy::arithmetic_side_effects,
    reason = "field values are bounded by the digit limits and range checks below"
)]

use std::fmt::Write as _;

use krishiv_common::error::{InvalidDatetimePatternWithSuggestion, KrishivError};

use super::format::{MICROS_PER_DAY, MONTHS, zone_name};
use super::{DOC_ROOT, Zone, unsupported};
use crate::cast::TimeZoneRules;
use crate::cast::datetime::{civil_from_days, days_from_civil, days_in_month};

const MICROS_PER_SECOND: i64 = 1_000_000;
/// First day of the Gregorian calendar, 1582-10-15, in days since the epoch.
const CUTOVER_DAYS: i64 = -141_427;

const SHORT_DAYS: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const LONG_DAYS: [&str; 7] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
];

#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    Literal(String),
    /// A run of one pattern letter.
    Field(char, usize),
}

/// A compiled legacy pattern (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyPattern {
    pattern: String,
    toks: Vec<Tok>,
}

impl LegacyPattern {
    /// Compiles `pattern`.
    ///
    /// # Errors
    /// `INVALID_DATETIME_PATTERN.WITH_SUGGESTION` for a letter the legacy
    /// formatters reject, a run of `X` over three, or an unterminated quote.
    pub fn new(pattern: &str) -> Result<Self, KrishivError> {
        let bad = || -> KrishivError {
            InvalidDatetimePatternWithSuggestion {
                pattern: format!("'{pattern}'"),
                docroot: DOC_ROOT.to_owned(),
            }
            .into()
        };
        let chars: Vec<char> = pattern.chars().collect();
        let mut toks = Vec::new();
        let mut lit = String::new();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c == '\'' {
                if chars.get(i + 1) == Some(&'\'') {
                    lit.push('\'');
                    i += 2;
                    continue;
                }
                i += 1;
                loop {
                    match chars.get(i) {
                        None => return Err(bad()),
                        Some('\'') if chars.get(i + 1) == Some(&'\'') => {
                            lit.push('\'');
                            i += 2;
                        }
                        Some('\'') => {
                            i += 1;
                            break;
                        }
                        Some(&q) => {
                            lit.push(q);
                            i += 1;
                        }
                    }
                }
            } else if c.is_ascii_alphabetic() {
                let mut n = 1;
                while chars.get(i + n) == Some(&c) {
                    n += 1;
                }
                if !"GyYMLwWDdFEuaHkKhmsSzZX".contains(c) || (c == 'X' && n > 3) {
                    return Err(bad());
                }
                if !lit.is_empty() {
                    toks.push(Tok::Literal(std::mem::take(&mut lit)));
                }
                toks.push(Tok::Field(c, n));
                i += n;
            } else {
                lit.push(c);
                i += 1;
            }
        }
        if !lit.is_empty() {
            toks.push(Tok::Literal(lit));
        }
        Ok(Self {
            pattern: pattern.to_owned(),
            toks,
        })
    }

    /// The pattern text.
    #[must_use]
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Whether the pattern prints or reads a zone or offset.
    #[must_use]
    pub fn has_zone(&self) -> bool {
        self.toks
            .iter()
            .any(|t| matches!(t, Tok::Field('z' | 'Z' | 'X', _)))
    }

    /// `micros` (an instant) shown in `zone`.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a timestamp past the year range or
    /// a `z` the zone-name tables do not cover.
    pub fn format_timestamp(&self, micros: i64, zone: Zone<'_>) -> Result<String, KrishivError> {
        let seconds = micros.div_euclid(MICROS_PER_SECOND);
        let offset = zone.rules.offset_at_utc(seconds);
        let local = micros
            .checked_add(i64::from(offset) * MICROS_PER_SECOND)
            .ok_or_else(|| unsupported("a timestamp past the year range in a pattern"))?;
        // Spark rebases the instant to the hybrid calendar keeping its
        // `java.time` date-time labels; the offset fields then print what
        // `java.util.TimeZone` says, which has no transition before 1900.
        let printed = zone.rules.legacy_offset_at_utc(seconds);
        self.write(
            local.div_euclid(MICROS_PER_DAY),
            local.rem_euclid(MICROS_PER_DAY),
            Some((printed, zone.id)),
        )
    }

    /// `local` (a `TIMESTAMP_NTZ`).
    ///
    /// # Errors
    /// A pattern with a zone or offset field.
    pub fn format_local(&self, local: i64) -> Result<String, KrishivError> {
        self.write(
            local.div_euclid(MICROS_PER_DAY),
            local.rem_euclid(MICROS_PER_DAY),
            None,
        )
    }

    /// `days` (a `DATE`): midnight of that day.
    ///
    /// # Errors
    /// A pattern with a zone or offset field.
    pub fn format_date(&self, days: i32) -> Result<String, KrishivError> {
        // Day counts, not microseconds: a Date32 may lie beyond the i64 microsecond range.
        self.write(i64::from(days), 0, None)
    }

    fn write(
        &self,
        days: i64,
        tod: i64,
        zone: Option<(i32, &str)>,
    ) -> Result<String, KrishivError> {
        let local = days.saturating_mul(MICROS_PER_DAY).saturating_add(tod);
        let (year, month, day) = civil_from_days(days);
        let (dow, doy) = hybrid_week_fields(days, year, month, day);
        let hour = tod / 3_600_000_000;
        let minute = tod / 60_000_000 % 60;
        let second = tod / MICROS_PER_SECOND % 60;
        let millis = tod % MICROS_PER_SECOND / 1000;
        let mut out = String::new();
        for t in &self.toks {
            let (c, n) = match t {
                Tok::Literal(s) => {
                    out.push_str(s);
                    continue;
                }
                Tok::Field(c, n) => (*c, *n),
            };
            match c {
                'G' => out.push_str(if year >= 1 { "AD" } else { "BC" }),
                'Y' => {
                    // The week year: the year of the week's Saturday.
                    let (wy, _, _) = civil_from_days(week_year_anchor(days, year, doy));
                    let era_year = if wy >= 1 { wy } else { 1 - wy };
                    if n == 2 {
                        let _ = write!(out, "{:02}", era_year % 100);
                    } else {
                        pad(&mut out, era_year, n);
                    }
                }
                'y' => {
                    let era_year = if year >= 1 { year } else { 1 - year };
                    if n == 2 {
                        let _ = write!(out, "{:02}", era_year % 100);
                    } else {
                        pad(&mut out, era_year, n);
                    }
                }
                'M' | 'L' => match n {
                    1 | 2 => pad(&mut out, i64::from(month), n),
                    3 => out.push_str(&MONTHS[month as usize - 1][..3]),
                    _ => out.push_str(MONTHS[month as usize - 1]),
                },
                'd' => pad(&mut out, i64::from(day), n),
                'D' => pad(&mut out, doy, n),
                'F' => pad(&mut out, (i64::from(day) - 1) / 7 + 1, n),
                'w' => pad(&mut out, week_of_year(days, year, doy), n),
                'W' => pad(&mut out, week_of_month(days, day), n),
                'E' => {
                    let d = usize::try_from(dow - 1).unwrap_or(0);
                    out.push_str(if n >= 4 { LONG_DAYS[d] } else { SHORT_DAYS[d] });
                }
                'u' => pad(&mut out, dow, n),
                'a' => out.push_str(if hour < 12 { "AM" } else { "PM" }),
                'H' => pad(&mut out, hour, n),
                'k' => pad(&mut out, if hour == 0 { 24 } else { hour }, n),
                'K' => pad(&mut out, hour % 12, n),
                'h' => pad(&mut out, if hour % 12 == 0 { 12 } else { hour % 12 }, n),
                'm' => pad(&mut out, minute, n),
                's' => pad(&mut out, second, n),
                'S' => pad(&mut out, millis, n),
                'z' | 'Z' | 'X' => {
                    let Some((offset, id)) = zone else {
                        return Err(unsupported(
                            "a zone or offset field in a pattern that formats a value without a zone",
                        ));
                    };
                    match c {
                        'z' => out.push_str(&zone_name(
                            id,
                            offset,
                            local.div_euclid(MICROS_PER_SECOND),
                            n >= 4,
                        )?),
                        'Z' => write_offset(&mut out, offset, false, true, false),
                        _ => write_offset(&mut out, offset, n == 3, n >= 2, true),
                    }
                }
                _ => {}
            }
        }
        Ok(out)
    }

    /// `s` as an instant: its zone or offset, else `session`. `Ok(None)`
    /// when `s` does not match (Spark raises `CANNOT_PARSE_TIMESTAMP`).
    ///
    /// # Errors
    /// Never.
    pub fn parse_timestamp(
        &self,
        s: &str,
        session: &dyn TimeZoneRules,
    ) -> Result<Option<i64>, KrishivError> {
        let Some((local, zone)) = self.parse_fields(s) else {
            return Ok(None);
        };
        Ok(match zone {
            None => legacy_local_to_utc(local, session),
            Some(Zone_::Offset(o)) => local.checked_sub(i64::from(o) * MICROS_PER_SECOND),
            Some(Zone_::Region(id)) => match crate::datetime::resolve_zone(&id) {
                Ok(z) => legacy_local_to_utc(local, &z),
                Err(_) => None,
            },
        })
    }

    /// `s` as a local date-time; a zone in `s` is read and ignored.
    ///
    /// # Errors
    /// Never.
    pub fn parse_local(&self, s: &str) -> Result<Option<i64>, KrishivError> {
        Ok(self.parse_fields(s).map(|(local, _)| local))
    }

    /// `s` as a date (days since the epoch): time and zone are ignored.
    ///
    /// # Errors
    /// Never.
    pub fn parse_date(&self, s: &str) -> Result<Option<i32>, KrishivError> {
        Ok(self
            .parse_fields(s)
            .and_then(|(local, _)| i32::try_from(local.div_euclid(MICROS_PER_DAY)).ok()))
    }

    fn parse_fields(&self, s: &str) -> Option<(i64, Option<Zone_>)> {
        let text: Vec<char> = s.chars().collect();
        let mut st = Parsed::default();
        let mut pos = 0;
        for (i, t) in self.toks.iter().enumerate() {
            match t {
                Tok::Literal(lit) => {
                    for c in lit.chars() {
                        if c.is_whitespace() {
                            // A blank in the pattern matches any run of blanks.
                            while text.get(pos).is_some_and(|x| x.is_whitespace()) {
                                pos += 1;
                            }
                        } else {
                            if text.get(pos) != Some(&c) {
                                return None;
                            }
                            pos += 1;
                        }
                    }
                }
                Tok::Field(c, n) => {
                    // A number directly followed by another number in the
                    // pattern reads exactly its run length.
                    let abut =
                        matches!(self.toks.get(i + 1), Some(Tok::Field(d, dn)) if is_numeric(*d, *dn))
                            .then_some(*n);
                    pos = st.field(&text, pos, *c, *n, abut)?;
                }
            }
        }
        st.resolve()
    }
}

/// UTC micros of the local time `local`, resolved as a non-lenient
/// `GregorianCalendar` does: a time in a gap is invalid (`None`), and in an
/// overlap the later instant (the smaller offset) wins. Zone offsets are the
/// real rules' (LMT before 1900): Spark rebases the parsed instant so the
/// wall clock survives, while only the printed offset letters use the raw
/// `java.util.TimeZone` offset.
fn legacy_local_to_utc(local: i64, tz: &dyn TimeZoneRules) -> Option<i64> {
    let secs = local.div_euclid(MICROS_PER_SECOND);
    let near = [
        tz.offset_at_utc(secs.saturating_sub(86_400)),
        tz.offset_at_utc(secs.saturating_add(86_400)),
    ];
    let offset = near
        .into_iter()
        .filter(|&o| tz.offset_at_utc(secs.saturating_sub(i64::from(o))) == o)
        .min()?;
    local.checked_sub(i64::from(offset).checked_mul(MICROS_PER_SECOND)?)
}

fn is_numeric(c: char, n: usize) -> bool {
    // Three or more `M` is a month name.
    !matches!(c, 'G' | 'E' | 'a' | 'z' | 'Z' | 'X') && !(matches!(c, 'M' | 'L') && n >= 3)
}

fn pad(out: &mut String, v: i64, n: usize) {
    let _ = write!(out, "{v:0n$}");
}

/// `(day of week, day of year)` of a label: Monday = 1; Julian arithmetic
/// before the Gregorian cutover.
fn hybrid_week_fields(days: i64, year: i64, month: u32, day: u32) -> (i64, i64) {
    if days >= CUTOVER_DAYS {
        let jan1 = days_from_civil(year, 1, 1).unwrap_or(days);
        return ((days + 3).rem_euclid(7) + 1, days - jan1 + 1);
    }
    // Julian day number of the label read as a Julian date.
    let a = (14 - i64::from(month)) / 12;
    let y = year + 4800 - a;
    let m = i64::from(month) + 12 * a - 3;
    let jdn = i64::from(day) + (153 * m + 2) / 5 + 365 * y + y.div_euclid(4) - 32083;
    let dow = jdn.rem_euclid(7) + 1;
    let leap = year.rem_euclid(4) == 0;
    let cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let doy = cum[month as usize - 1] + i64::from(day) + i64::from(leap && month > 2);
    (dow, doy)
}

/// `Calendar.WEEK_OF_YEAR` for `Locale.US` (weeks start on Sunday, week 1
/// is the one with January 1st).
fn week_of_year(days: i64, year: i64, doy: i64) -> i64 {
    let (dow, _) = hybrid_week_fields(days, year, 1, 1);
    let _ = dow;
    // Day of week of January 1st of the year (Sunday = 0).
    let jan1 = days - (doy - 1);
    let jan1_sun0 = (jan1 + 4).rem_euclid(7);
    let week = (doy - 1 + jan1_sun0) / 7 + 1;
    // The last days of December in the week that holds January 1st of the
    // next year belong to its week 1.
    let next_jan1 = jan1 + if is_leap(year) { 366 } else { 365 };
    let next_sun0 = (next_jan1 + 4).rem_euclid(7);
    let dec31 = next_jan1 - 1;
    if days > dec31 - next_sun0 && next_sun0 != 0 {
        1
    } else {
        week
    }
}

/// The day that decides the week year of `days`: itself, unless the week
/// (Sunday first, week 1 holds January 1st) is week 1 of the next year.
fn week_year_anchor(days: i64, year: i64, doy: i64) -> i64 {
    if week_of_year(days, year, doy) == 1 && doy > 300 {
        // Late December in next year's week 1.
        days + 31
    } else {
        days
    }
}

fn is_leap(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// `Calendar.WEEK_OF_MONTH` for `Locale.US`.
fn week_of_month(days: i64, day: u32) -> i64 {
    let first = days - i64::from(day) + 1;
    let first_sun0 = (first + 4).rem_euclid(7);
    (i64::from(day) - 1 + first_sun0) / 7 + 1
}

/// `FastDateFormat`'s offsets: `Z` is `+hhmm`; `X` is `Z` for zero, else
/// `+hh`, `+hhmm` or `+hh:mm` (`minutes`, `colon`).
fn write_offset(out: &mut String, offset: i32, colon: bool, minutes: bool, zulu: bool) {
    if zulu && offset == 0 {
        out.push('Z');
        return;
    }
    out.push(if offset < 0 { '-' } else { '+' });
    let a = offset.unsigned_abs();
    let _ = write!(out, "{:02}", a / 3600);
    if minutes {
        if colon {
            out.push(':');
        }
        let _ = write!(out, "{:02}", a / 60 % 60);
    }
}

#[derive(Debug)]
enum Zone_ {
    Offset(i32),
    Region(String),
}

/// The calendar fields a parse sets, each with the order it was set in (the
/// `Calendar` resolves a clash between date field groups by recency).
#[derive(Default)]
struct Parsed {
    clock: u32,
    year: Option<(i64, u32)>,
    era_bc: bool,
    /// The year came from `Y`: a week-year, checked against the week's.
    year_is_week: bool,
    month0: Option<(i64, u32)>,
    dom: Option<(i64, u32)>,
    doy: Option<(i64, u32)>,
    /// Week of year, week of month, day of week in month, day of week
    /// (`Calendar`'s numbering: Sunday = 1).
    woy: Option<(i64, u32)>,
    wom: Option<(i64, u32)>,
    dowim: Option<(i64, u32)>,
    dow: Option<(i64, u32)>,
    hour_of_day: Option<(i64, u32)>,
    hour12: Option<(i64, u32)>,
    /// A one-based hour (`k`, `h`) read as 0 or past its range.
    bad_hour: bool,
    pm: Option<(bool, u32)>,
    minute: Option<i64>,
    second: Option<i64>,
    millis: Option<i64>,
    zone: Option<Zone_>,
}

impl Parsed {
    fn stamp(&mut self) -> u32 {
        self.clock += 1;
        self.clock
    }

    /// Reads one field at `pos`, returning the new position.
    fn field(
        &mut self,
        text: &[char],
        mut pos: usize,
        c: char,
        n: usize,
        abut: Option<usize>,
    ) -> Option<usize> {
        // Every field skips leading blanks.
        while text.get(pos).is_some_and(|x| x.is_whitespace()) {
            pos += 1;
        }
        let st = self.stamp();
        match c {
            'G' => {
                if let Some(p) = prefix(text, pos, &["AD"]) {
                    return Some(p);
                }
                let p = prefix(text, pos, &["BC"])?;
                self.era_bc = true;
                return Some(p);
            }
            'a' => {
                if let Some(p) = prefix(text, pos, &["AM"]) {
                    self.pm = Some((false, st));
                    return Some(p);
                }
                let p = prefix(text, pos, &["PM"])?;
                self.pm = Some((true, st));
                return Some(p);
            }
            'E' => {
                let (p, i) = name(text, pos, &LONG_DAYS, &SHORT_DAYS)?;
                // The tables start on Monday.
                self.dow = Some((if i == 6 { 1 } else { i as i64 + 2 }, st));
                return Some(p);
            }
            'M' | 'L' if n >= 3 => {
                let short: Vec<&str> = MONTHS.iter().map(|m| &m[..3]).collect();
                let (p, i) = name(text, pos, &MONTHS, &short)?;
                self.month0 = Some((i as i64, st));
                return Some(p);
            }
            'X' => {
                let (p, z) = zone_iso(text, pos, n)?;
                self.zone = Some(z);
                return Some(p);
            }
            'z' | 'Z' => {
                let (p, z) = zone(text, pos)?;
                self.zone = Some(z);
                return Some(p);
            }
            _ => {}
        }
        let (value, end) = number(text, pos, abut)?;
        let digits = end - pos;
        match c {
            'y' | 'Y' => {
                let v = if n <= 2 && digits == 2 {
                    window_year(value)
                } else {
                    value
                };
                self.year = Some((v, st));
                self.year_is_week = c == 'Y';
            }
            'M' | 'L' => self.month0 = Some((value - 1, st)),
            'd' => self.dom = Some((value, st)),
            'D' => self.doy = Some((value, st)),
            'H' => self.hour_of_day = Some((value, st)),
            // 24 reads as 0 (no carry), as `SimpleDateFormat` does.
            'k' => {
                self.bad_hour |= value == 0;
                self.hour_of_day = Some((if value == 24 { 0 } else { value }, st));
            }
            'h' => {
                self.bad_hour |= value == 0;
                self.hour12 = Some((if value == 12 { 0 } else { value }, st));
            }
            'K' => self.hour12 = Some((value, st)),
            'm' => self.minute = Some(value),
            's' => self.second = Some(value),
            'S' => self.millis = Some(value),
            'w' => self.woy = Some((value, st)),
            'W' => self.wom = Some((value, st)),
            'F' => self.dowim = Some((value, st)),
            // `u` counts from Monday = 1; the `Calendar` from Sunday = 1.
            'u' => {
                if !(1..=7).contains(&value) {
                    return None;
                }
                self.dow = Some((if value == 7 { 1 } else { value + 1 }, st));
            }
            _ => {}
        }
        Some(end)
    }

    /// `GregorianCalendar.selectFields`: which group of fields decides the
    /// date when a week field was set. Unset stamps compare lowest.
    fn week_mode(&self) -> Mode {
        let st = |o: Option<(i64, u32)>| o.map(|x| x.1);
        let agg = |a: Option<u32>, b: Option<u32>| a.zip(b).map(|(x, y)| x.max(y));
        let dow = st(self.dow);
        let month = st(self.month0);
        let mut dom = st(self.dom);
        let mut wom = agg(st(self.wom), dow);
        let mut dowim = agg(st(self.dowim), dow);
        let doy = st(self.doy);
        let mut woy = agg(st(self.woy), dow);
        let mut best = dom.max(wom).max(dowim).max(doy).max(woy);
        if best.is_none() {
            wom = st(self.wom);
            dowim = st(self.dowim).max(dow);
            woy = st(self.woy);
            best = wom.max(dowim).max(woy);
            if best.is_none() {
                best = month;
                dom = month;
            }
        }
        let (rw, rd, ry) = (st(self.wom), st(self.dowim), st(self.woy));
        if best == dom || (best == wom && rw >= ry) || (best == dowim && rd >= ry) {
            if best == dom {
                Mode::Dom
            } else if wom == dowim {
                if rw >= rd { Mode::Wom } else { Mode::Dowim }
            } else if best == wom {
                Mode::Wom
            } else {
                Mode::Dowim
            }
        } else if best == doy {
            Mode::Doy
        } else {
            Mode::Woy
        }
    }

    /// The day a week-field group names (`GregorianCalendar.computeTime`,
    /// `Locale.US`: weeks start on Sunday, a week 1 may hold one day).
    fn week_date(&self, year: i64, mode: Mode) -> Option<i64> {
        let on_or_before = |d: i64, dow: i64| d - (dow_sun1(d) - dow).rem_euclid(7);
        let first_of = |month: u32| days_from_civil(year, month, 1);
        let week_start = |first: i64| {
            let mut start = on_or_before(first + 6, 1);
            if start - first >= 1 {
                start -= 7;
            }
            match self.dow {
                Some((dow, _)) => on_or_before(start + 6, dow),
                None => start,
            }
        };
        match mode {
            Mode::Wom => {
                let m = u32::try_from(self.month0.map_or(0, |(v, _)| v) + 1).ok()?;
                let first = first_of(m)?;
                Some(week_start(first) + 7 * (self.wom?.0 - 1))
            }
            Mode::Woy => {
                let first = first_of(1)?;
                Some(week_start(first) + 7 * (self.woy?.0 - 1))
            }
            _ => {
                let m = u32::try_from(self.month0.map_or(0, |(v, _)| v) + 1).ok()?;
                let first = first_of(m)?;
                let dow = self.dow.map_or(1, |(d, _)| d);
                let n = self.dowim.map_or(1, |(v, _)| v);
                if n >= 0 {
                    Some(on_or_before(first + 7 * n - 1, dow))
                } else {
                    let len = i64::from(days_in_month(year, m));
                    Some(on_or_before(first + len + 7 * (n + 1) - 1, dow))
                }
            }
        }
    }

    /// A non-lenient `Calendar` rejects a result that disagrees with a
    /// field that was set.
    fn week_fields_agree(&self, days: i64, year: i64) -> bool {
        let (y, m, d) = civil_from_days(days);
        let doy = days - days_from_civil(y, 1, 1).unwrap_or(days) + 1;
        let year_ok = if self.year_is_week {
            civil_from_days(week_year_anchor(days, y, doy)).0 == year
        } else {
            y == year
        };
        year_ok
            && self.month0.is_none_or(|(v, _)| v == i64::from(m) - 1)
            && self.dom.is_none_or(|(v, _)| v == i64::from(d))
            && self.doy.is_none_or(|(v, _)| v == doy)
            && self
                .woy
                .is_none_or(|(v, _)| v == week_of_year(days, y, doy))
            && self.wom.is_none_or(|(v, _)| v == week_of_month(days, d))
            && self
                .dowim
                .is_none_or(|(v, _)| v == (i64::from(d) - 1) / 7 + 1)
            && self.dow.is_none_or(|(v, _)| v == dow_sun1(days))
    }

    fn resolve(self) -> Option<(i64, Option<Zone_>)> {
        if self.bad_hour {
            return None;
        }
        let mut year = self.year.map_or(1970, |(y, _)| y);
        if self.era_bc {
            year = 1 - year;
        }
        let weeky = self.woy.is_some() || self.wom.is_some() || self.dowim.is_some();
        let mode = if weeky {
            self.week_mode()
        } else if self.doy.is_some_and(|(_, s)| {
            self.month0
                .map(|(_, s)| s)
                .max(self.dom.map(|(_, s)| s))
                .is_none_or(|m| s > m)
        }) {
            Mode::Doy
        } else {
            Mode::Dom
        };
        let days = if matches!(mode, Mode::Wom | Mode::Dowim | Mode::Woy) {
            self.week_date(year, mode)?
        } else if matches!(mode, Mode::Doy) {
            let (d, _) = self.doy?;
            let len = if is_leap(year) { 366 } else { 365 };
            if !(1..=len).contains(&d) {
                return None;
            }
            let (m, dd) = month_day_of(year, d);
            // The other date fields that were set must agree.
            if self.month0.is_some_and(|(v, _)| v != i64::from(m) - 1)
                || self.dom.is_some_and(|(v, _)| v != i64::from(dd))
            {
                return None;
            }
            days_from_civil(year, m, dd)?
        } else {
            let m = self.month0.map_or(0, |(v, _)| v);
            let d = self.dom.map_or(1, |(v, _)| v);
            if !(0..12).contains(&m) || d < 1 {
                return None;
            }
            days_from_civil(year, u32::try_from(m + 1).ok()?, u32::try_from(d).ok()?)?
        };
        // The labels 1582-10-05..14 do not exist in the hybrid calendar.
        let cutover = days_from_civil(1582, 10, 4).unwrap_or(i64::MIN);
        if days > cutover && days < cutover + 11 {
            return None;
        }
        if (weeky || self.dow.is_some()) && !self.week_fields_agree(days, year) {
            return None;
        }
        let hour = match (self.hour_of_day, self.hour12) {
            (Some((h, hs)), other) if other.is_none_or(|(_, os)| hs > os) => {
                if !(0..24).contains(&h) {
                    return None;
                }
                h
            }
            (_, Some((h, _))) => {
                if !(0..12).contains(&h) {
                    return None;
                }
                h + if self.pm.is_some_and(|(pm, _)| pm) {
                    12
                } else {
                    0
                }
            }
            _ => self.pm.map_or(0, |(pm, _)| if pm { 12 } else { 0 }),
        };
        let minute = self.minute.unwrap_or(0);
        let second = self.second.unwrap_or(0);
        let millis = self.millis.unwrap_or(0);
        if !(0..60).contains(&minute) || !(0..60).contains(&second) || !(0..1000).contains(&millis)
        {
            return None;
        }
        let tod = ((hour * 60 + minute) * 60 + second) * MICROS_PER_SECOND + millis * 1000;
        let local = days.checked_mul(MICROS_PER_DAY)?.checked_add(tod)?;
        Some((local, self.zone))
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Dom,
    Doy,
    Wom,
    Dowim,
    Woy,
}

/// Day of week of `days` since the epoch, Sunday = 1.
fn dow_sun1(days: i64) -> i64 {
    (days + 4).rem_euclid(7) + 1
}

/// `(month, day)` of the 1-based `doy` of `year`.
fn month_day_of(year: i64, mut doy: i64) -> (u32, u32) {
    for m in 1..=12u32 {
        let len = i64::from(days_in_month(year, m));
        if doy <= len {
            return (m, u32::try_from(doy).unwrap_or(1));
        }
        doy -= len;
    }
    (12, 31)
}

/// `SimpleDateFormat`'s two-digit year window: the 80 years before now to
/// the 20 after.
fn window_year(two: i64) -> i64 {
    // Miri's isolation has no wall clock, so it reads a fixed 2026-07-01.
    #[cfg(miri)]
    let now_secs = 1_782_864_000_i64;
    #[cfg(not(miri))]
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64);
    let (now_year, _, _) = civil_from_days(now_secs.div_euclid(86_400));
    let start = now_year - 80;
    let mut year = start - start.rem_euclid(100) + two;
    if year < start {
        year += 100;
    }
    year
}

/// An unsigned integer at `pos` (`limit` digits at most), as the digits
/// and where they end.
fn number(text: &[char], pos: usize, limit: Option<usize>) -> Option<(i64, usize)> {
    let mut end = pos;
    let mut value: i64 = 0;
    while let Some(d) = text.get(end).and_then(|c| c.to_digit(10)) {
        if limit.is_some_and(|l| end - pos >= l) {
            break;
        }
        value = value.checked_mul(10)?.checked_add(i64::from(d))?;
        end += 1;
    }
    (end > pos).then_some((value, end))
}

/// Case-insensitive prefix match of any of `options`.
fn prefix(text: &[char], pos: usize, options: &[&str]) -> Option<usize> {
    options.iter().find_map(|o| {
        let len = o.chars().count();
        let slice = text.get(pos..pos + len)?;
        slice
            .iter()
            .zip(o.chars())
            .all(|(a, b)| a.eq_ignore_ascii_case(&b))
            .then_some(pos + len)
    })
}

/// A full or short name (full first), its end and index.
fn name<S: AsRef<str>, T: AsRef<str>>(
    text: &[char],
    pos: usize,
    full: &[S],
    short: &[T],
) -> Option<(usize, usize)> {
    for list in [
        full.iter().map(AsRef::as_ref).collect::<Vec<_>>(),
        short.iter().map(AsRef::as_ref).collect::<Vec<_>>(),
    ] {
        for (i, n) in list.iter().enumerate() {
            if let Some(p) = prefix(text, pos, &[n]) {
                return Some((p, i));
            }
        }
    }
    None
}

/// `SimpleDateFormat`'s ISO 8601 zone (`X`, `XX`, `XXX`): `Z`, or a sign and
/// `hh` (one letter), `hhmm` (two) or `hh:mm` (three); nothing looser.
fn zone_iso(text: &[char], pos: usize, n: usize) -> Option<(usize, Zone_)> {
    if text.get(pos) == Some(&'Z') {
        return Some((pos + 1, Zone_::Offset(0)));
    }
    let sign = *text.get(pos).filter(|c| matches!(c, '+' | '-'))?;
    let (hh, mut e) = number(text, pos + 1, Some(2))?;
    if e - (pos + 1) != 2 {
        return None;
    }
    let mut mm = 0;
    if n >= 2 {
        if n == 3 {
            if text.get(e) != Some(&':') {
                return None;
            }
            e += 1;
        }
        let (m, e2) = number(text, e, Some(2))?;
        if e2 - e != 2 {
            return None;
        }
        mm = m;
        e = e2;
    }
    if hh > 23 || mm > 59 {
        return None;
    }
    let secs = i32::try_from(hh * 3600 + mm * 60).ok()?;
    Some((e, Zone_::Offset(if sign == '-' { -secs } else { secs })))
}

/// A zone: `Z`, `GMT`/`UTC` with an optional offset, `+hh`, `+hhmm`,
/// `+hh:mm`, or a region id.
fn zone(text: &[char], pos: usize) -> Option<(usize, Zone_)> {
    if let Some(p) = prefix(text, pos, &["Z"])
        && !text.get(p).is_some_and(char::is_ascii_alphabetic)
    {
        return Some((p, Zone_::Offset(0)));
    }
    let mut p = pos;
    let named = prefix(text, p, &["GMT", "UTC"]);
    if let Some(q) = named {
        p = q;
    }
    if let Some(&sign) = text.get(p).filter(|c| matches!(c, '+' | '-')) {
        let (hh, e) = number(text, p + 1, Some(2))?;
        let (mm, e) = if text.get(e) == Some(&':') {
            let (m, e2) = number(text, e + 1, Some(2))?;
            (m, e2)
        } else if let Some((m, e2)) = number(text, e, Some(2)).filter(|_| e - (p + 1) == 2) {
            (m, e2)
        } else {
            (0, e)
        };
        if hh > 23 || mm > 59 {
            return None;
        }
        let secs = i32::try_from(hh * 3600 + mm * 60).ok()?;
        return Some((e, Zone_::Offset(if sign == '-' { -secs } else { secs })));
    }
    if let Some(q) = named {
        return Some((q, Zone_::Offset(0)));
    }
    let mut e = pos;
    while text
        .get(e)
        .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '_' | '-' | '+'))
    {
        e += 1;
    }
    if e == pos {
        return None;
    }
    let id: String = text[pos..e].iter().collect();
    // A standard-time display name is that fixed offset all year, as
    // `SimpleDateFormat` reads it (not the region's seasonal rules).
    let fixed = match id.as_str() {
        "PST" => Some(-8),
        "MST" => Some(-7),
        "CST" => Some(-6),
        "EST" => Some(-5),
        "AKST" => Some(-9),
        "HST" => Some(-10),
        _ => None,
    };
    if let Some(h) = fixed {
        return Some((e, Zone_::Offset(h * 3600)));
    }
    crate::datetime::resolve_zone(&id).ok()?;
    Some((e, Zone_::Region(id)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cast::FixedOffset;

    fn fmt(p: &str, local: i64) -> String {
        LegacyPattern::new(p).unwrap().format_local(local).unwrap()
    }

    fn micros(y: i64, mo: u32, d: u32, h: i64, mi: i64, s: i64, ms: i64) -> i64 {
        let days = days_from_civil(y, mo, d).unwrap();
        days * MICROS_PER_DAY + ((h * 60 + mi) * 60 + s) * MICROS_PER_SECOND + ms * 1000
    }

    #[test]
    fn a_date_beyond_the_microsecond_range_formats_from_day_counts() {
        let p = LegacyPattern::new("yyyy-MM-dd").unwrap();
        // 9999-12-31 and a day far past i64 microseconds (2^63 / 86.4e9 days is ~292,000 years).
        assert_eq!(p.format_date(2_932_896).unwrap(), "9999-12-31");
        assert!(p.format_date(i32::MAX).is_ok());
    }

    fn local(p: &str, text: &str) -> Option<i64> {
        LegacyPattern::new(p).unwrap().parse_local(text).unwrap()
    }

    // Spark 4.2.0 under LEGACY (probe 2026-11-02).
    #[test]
    fn week_fields_resolve_like_gregorian_calendar() {
        let d = |y, m, dd| days_from_civil(y, m, dd).unwrap() * MICROS_PER_DAY;
        assert_eq!(local("yyyy-ww-u", "2024-02-1"), Some(d(2024, 1, 8)));
        assert_eq!(local("yyyy-ww-u", "2024-01-2"), Some(d(2024, 1, 2)));
        assert_eq!(local("yyyy-MM-F", "2024-05-3"), Some(d(2024, 5, 19)));
        // Week 1 of February 2024 starts in January: the month disagrees.
        assert_eq!(local("yyyy-MM-W", "2024-02-1"), None);
    }

    #[test]
    fn one_based_hours_reject_zero_and_the_cutover_gap_is_not_a_date() {
        assert_eq!(local("yyyy-MM-dd kk", "2024-01-01 00"), None);
        assert_eq!(local("yyyy-MM-dd hh a", "2024-01-01 00 AM"), None);
        assert!(local("yyyy-MM-dd kk", "2024-01-01 24").is_some());
        assert!(local("yyyy-MM-dd hh a", "2024-01-01 12 AM").is_some());
        assert_eq!(local("yyyy-MM-dd", "1582-10-10"), None);
        assert!(local("yyyy-MM-dd", "1582-10-04").is_some());
        assert!(local("yyyy-MM-dd", "1582-10-15").is_some());
    }

    #[test]
    fn review_edge_cases() {
        let l = |p, t| local(p, t);
        assert_eq!(l("yyyy-MM-dd E", "2024-01-01 Tue"), None);
        assert!(l("yyyy-MM-dd E", "2024-01-01 Mon").is_some());
        assert_eq!(l("yyyy-MM-dd XXX", "2024-01-01 +25:00"), None);
        assert_eq!(l("yyyy-MM-dd XXX", "2024-01-01 +05:99"), None);
        assert!(l("dMMMyyyy", "12Jan2024").is_some());
        // Week-year 2020 week 1 starts on 2019-12-29 (Sunday); day 1 = Sunday.
        assert!(l("YYYY-ww-u", "2020-01-1").is_some());
    }

    #[test]
    fn uppercase_y_is_the_week_year() {
        let t = micros(2019, 12, 30, 0, 0, 0, 0);
        assert_eq!(fmt("YYYY-ww", t), "2020-01");
        assert_eq!(fmt("yyyy-ww", t), "2019-01");
        assert_eq!(fmt("YYYY-ww", micros(2021, 1, 1, 0, 0, 0, 0)), "2021-01");
    }

    #[test]
    fn iso_offsets_have_a_width_each() {
        let ok = |p: &str, t: &str| local(p, t).is_some();
        assert!(ok("yyyy-MM-dd X", "2024-03-05 +05"));
        assert!(ok("yyyy-MM-dd X", "2024-03-05 +05:30")); // trailing text ignored
        assert!(ok("yyyy-MM-dd X", "2024-03-05 Z"));
        assert!(ok("yyyy-MM-dd XX", "2024-03-05 +0530"));
        assert!(!ok("yyyy-MM-dd XX", "2024-03-05 +05"));
        assert!(ok("yyyy-MM-dd XXX", "2024-03-05 +05:30"));
        assert!(!ok("yyyy-MM-dd XXX", "2024-03-05 +0530"));
    }

    #[test]
    fn local_times_resolve_like_a_non_lenient_calendar() {
        let la = crate::datetime::resolve_zone("America/Los_Angeles").unwrap();
        let p = LegacyPattern::new("yyyy-MM-dd HH:mm:ss").unwrap();
        assert_eq!(p.parse_timestamp("2024-03-10 02:30:00", &la).unwrap(), None);
        let t = p
            .parse_timestamp("2024-11-03 01:30:00", &la)
            .unwrap()
            .unwrap();
        assert_eq!(t, micros(2024, 11, 3, 9, 30, 0, 0));
    }

    #[test]
    fn formats_like_fast_date_format() {
        let t = micros(2018, 11, 17, 5, 33, 33, 123);
        assert_eq!(fmt("G GG GGG GGGG", t), "AD AD AD AD");
        assert_eq!(
            fmt("y yy yyy yyyy yyyyy yyyyyy", t),
            "2018 18 2018 2018 02018 002018"
        );
        assert_eq!(fmt("M MM MMM MMMM", t), "11 11 Nov November");
        assert_eq!(fmt("L LL", t), "11 11");
        assert_eq!(fmt("E EE EEE EEEE", t), "Sat Sat Sat Saturday");
        assert_eq!(fmt("F", t), "3");
        assert_eq!(fmt("D DDD", t), "321 321");
        assert_eq!(fmt("h hh k kk K KK a", t), "5 05 5 05 5 05 AM");
        assert_eq!(
            fmt("S SS SSS SSSS SSSSSSSSS", t),
            "123 123 123 0123 000000123"
        );
        assert_eq!(fmt("[yyyy-MM-dd HH:mm:ss]", t), "[2018-11-17 05:33:33]");
        assert_eq!(
            fmt("姚123'GyYqQMLwWuEFDdhHmsSaVzZxXOV'", t),
            "姚123GyYqQMLwWuEFDdhHmsSaVzZxXOV"
        );
        assert_eq!(fmt("''", t), "'");
        assert_eq!(fmt("", t), "");
        assert_eq!(fmt("u", t), "6");
    }

    #[test]
    fn the_julian_calendar_decides_week_fields_before_1582() {
        // 1582-05-31 is a Thursday in the Julian calendar.
        let t = micros(1582, 5, 31, 19, 40, 35, 123);
        assert_eq!(fmt("E D", t), "Thu 151");
        assert_eq!(fmt("E", micros(2100, 1, 1, 1, 33, 33, 123)), "Fri");
        assert_eq!(fmt("E", micros(1969, 12, 31, 15, 0, 0, 0)), "Wed");
    }

    #[test]
    fn illegal_letters_are_invalid_patterns() {
        for p in [
            "q qq",
            "Q QQQ",
            "VV",
            "XXXX XXXXX",
            "O OOOO",
            "x xx",
            "'open",
        ] {
            assert!(LegacyPattern::new(p).is_err(), "{p}");
        }
    }

    #[test]
    fn offsets_print_as_commons_lang_does() {
        let p = LegacyPattern::new("X XX XXX|Z ZZZZZ").unwrap();
        let z = Zone {
            id: "-08:00",
            rules: &FixedOffset(-8 * 3600),
        };
        assert_eq!(
            p.format_timestamp(0, z).unwrap(),
            "-08 -0800 -08:00|-0800 -0800"
        );
        let utc = Zone {
            id: "UTC",
            rules: &FixedOffset(0),
        };
        assert_eq!(p.format_timestamp(0, utc).unwrap(), "Z Z Z|+0000 +0000");
    }

    /// `java.util.TimeZone` drops the transitions before 1900: earlier
    /// instants print the zone's raw offset, later ones the database's
    /// (Spark 4.2.0, `timeParserPolicy=LEGACY`, `date_format` in the
    /// session zone: LA 1800 `-0800`, Kolkata 1850 `+0530` / 1899-12-31
    /// `+0530` / 1900-06 `+0521`, Kathmandu 1850 `+0545`, Amsterdam (a link
    /// to Brussels) 1850 `+0100` / 1910 `+0000`).
    #[test]
    fn region_offsets_before_1900_are_the_raw_offset() {
        let p = LegacyPattern::new("yyyy-MM-dd HH:mm:ss Z").unwrap();
        let show = |zone_id: &str, y, mo, d, h| {
            let rules = crate::datetime::resolve_zone(zone_id).unwrap();
            let local = micros(y, mo, d, h, 0, 0, 0);
            let utc = crate::cast::datetime::local_to_utc(local, &rules).unwrap();
            p.format_timestamp(
                utc,
                Zone {
                    id: zone_id,
                    rules: &rules,
                },
            )
            .unwrap()
        };
        assert_eq!(
            show("America/Los_Angeles", 1800, 1, 1, 0),
            "1800-01-01 00:00:00 -0800"
        );
        assert_eq!(
            show("America/Los_Angeles", 1582, 5, 31, 19),
            "1582-05-31 19:00:00 -0800"
        );
        assert_eq!(
            show("Asia/Kolkata", 1850, 6, 1, 0),
            "1850-06-01 00:00:00 +0530"
        );
        assert_eq!(
            show("Asia/Kolkata", 1899, 12, 31, 23),
            "1899-12-31 23:00:00 +0530"
        );
        assert_eq!(
            show("Asia/Kolkata", 1900, 6, 1, 0),
            "1900-06-01 00:00:00 +0521"
        );
        assert_eq!(
            show("Asia/Kolkata", 1920, 6, 1, 0),
            "1920-06-01 00:00:00 +0530"
        );
        assert_eq!(
            show("Asia/Kathmandu", 1850, 6, 1, 0),
            "1850-06-01 00:00:00 +0545"
        );
        assert_eq!(
            show("Asia/Kathmandu", 1960, 6, 1, 0),
            "1960-06-01 00:00:00 +0530"
        );
        assert_eq!(
            show("Europe/Amsterdam", 1850, 6, 1, 0),
            "1850-06-01 00:00:00 +0100"
        );
        assert_eq!(
            show("Europe/Amsterdam", 1910, 6, 1, 0),
            "1910-06-01 00:00:00 +0000"
        );
    }

    fn parse(p: &str, s: &str) -> Option<i64> {
        LegacyPattern::new(p).unwrap().parse_local(s).unwrap()
    }

    #[test]
    fn parses_like_simple_date_format() {
        let day = |y, m, d| Some(micros(y, m, d, 0, 0, 0, 0));
        assert_eq!(parse("y", "1"), day(1, 1, 1));
        assert_eq!(parse("y", "009999"), day(9999, 1, 1));
        assert_eq!(parse("yy", "00"), day(2000, 1, 1));
        assert_eq!(parse("yy", "99"), day(1999, 1, 1));
        assert_eq!(parse("yyy", "001"), day(1, 1, 1));
        assert_eq!(parse("yyyyyy", "000001"), day(1, 1, 1));
        assert_eq!(parse("D", "9"), day(1970, 1, 9));
        assert_eq!(parse("D", "300"), day(1970, 10, 27));
        assert_eq!(parse("DD", "100"), day(1970, 4, 10));
        assert_eq!(parse("dd-DDD", "31-365"), day(1970, 12, 31));
        assert_eq!(parse("yyyy-DDD", "2020-365"), day(2020, 12, 30));
        assert_eq!(parse("yyyy-MM-DDD", "2020-12-350"), day(2020, 12, 15));
        assert_eq!(parse("yyyy-MM-dd-DDD", "2020-12-31-366"), day(2020, 12, 31));
        assert_eq!(parse("yyyy-MM-dd", "2020-1-2"), day(2020, 1, 2));
        assert_eq!(parse("yyyyMMdd", "20200102"), day(2020, 1, 2));
        assert_eq!(parse("MMM d yyyy", "January 5 2020"), day(2020, 1, 5));
        assert_eq!(parse("dd", "16"), day(1970, 1, 16));
        assert_eq!(parse("yyyy", " 2020"), day(2020, 1, 1));
    }

    #[test]
    fn parsing_is_not_lenient_but_ignores_trailing_text() {
        assert_eq!(parse("MM-dd", "02-29"), None);
        assert_eq!(parse("yyyy-MM-dd", "2020-13-45"), None);
        assert_eq!(parse("yyyy-MM-dd mm", "2020-01-02 75"), None);
        assert_eq!(parse("yyyy", "x2020"), None);
        assert_eq!(parse("y", "-5"), None);
        assert_eq!(parse("yyyy-MM-dd HH:mm:ss", "2020-01-02"), None);
        assert_eq!(
            parse("yyyy-MM-dd HH:mm:ss", "2020-01-02 03:04:05 trailing"),
            Some(micros(2020, 1, 2, 3, 4, 5, 0))
        );
        assert_eq!(parse("dd-DDD", "01-365"), None);
    }

    #[test]
    fn parses_clock_fields() {
        assert_eq!(parse("h a", "5 PM"), Some(micros(1970, 1, 1, 17, 0, 0, 0)));
        assert_eq!(parse("hh a", "12 AM"), Some(micros(1970, 1, 1, 0, 0, 0, 0)));
        assert_eq!(
            parse("yyyy-MM-dd kk:mm:ss", "2020-01-02 24:00:00"),
            Some(micros(2020, 1, 2, 0, 0, 0, 0))
        );
        assert_eq!(
            parse("HH:mm:ss.SSS", "03:04:05.12"),
            Some(micros(1970, 1, 1, 3, 4, 5, 12))
        );
        assert_eq!(
            parse("EEE yyyy-MM-dd", "Mon 2020-01-06"),
            Some(micros(2020, 1, 6, 0, 0, 0, 0))
        );
    }

    #[test]
    fn a_parsed_offset_moves_the_instant() {
        let p = LegacyPattern::new("yyyy-MM-dd'T'HH:mm:ssZ").unwrap();
        assert_eq!(
            p.parse_timestamp("2020-01-02T03:04:05+0530", &FixedOffset(0))
                .unwrap(),
            Some(micros(2020, 1, 1, 21, 34, 5, 0))
        );
    }
}
