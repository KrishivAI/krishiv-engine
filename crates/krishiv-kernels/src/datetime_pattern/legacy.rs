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
use super::legacy_zones::{DIGIT_ZEROS, ZONE_STRINGS, ZoneStrings, zone_row};
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
    /// The date the two-digit year window is anchored to, read once when
    /// the pattern is compiled so every parse of it maps years alike.
    today: (i64, u32, u32),
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
            today: today(),
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
        let real = real_day(days, year, month, day);
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
                    let (wy, _, _) = civil_from_days(week_year_anchor(days, real, year, doy));
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
                'F' => pad(&mut out, dowim_of(year, month, day), n),
                'w' => pad(&mut out, week_of_year(real, year, doy), n),
                'W' => pad(&mut out, week_of_month(real, year, month, day), n),
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
                        'z' => {
                            let fixed = id.starts_with(['+', '-'])
                                || ["UTC+", "UTC-", "GMT+", "GMT-"]
                                    .iter()
                                    .any(|p| id.starts_with(p));
                            if fixed && offset != 0 {
                                // `java.util.TimeZone` names a fixed offset `GMT+hh:mm`.
                                let a = offset.unsigned_abs();
                                let sign = if offset < 0 { '-' } else { '+' };
                                let _ = write!(out, "GMT{sign}{:02}:{:02}", a / 3600, a / 60 % 60);
                            } else {
                                out.push_str(&zone_name(
                                    id,
                                    offset,
                                    local.div_euclid(MICROS_PER_SECOND),
                                    n >= 4,
                                )?);
                            }
                        }
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
        let Some((local, zone)) = self.parse_fields(s, session.region_id()) else {
            return Ok(None);
        };
        Ok(match zone {
            None => legacy_local_to_utc(local, session),
            Some(Zone_::Offset(o)) => local.checked_sub(i64::from(o) * MICROS_PER_SECOND),
            Some(Zone_::Region(id)) => match crate::datetime::resolve_zone(&id) {
                Ok(z) => legacy_local_to_utc(local, &z),
                Err(_) => None,
            },
            Some(Zone_::Standard { id, raw, dst }) => {
                // The calendar reads the standard offset at the wall time.
                let offset = match crate::datetime::resolve_zone(id) {
                    Ok(z) => {
                        let secs = local.div_euclid(MICROS_PER_SECOND);
                        let first = z.standard_offset_at_utc(secs - i64::from(raw));
                        z.standard_offset_at_utc(secs - i64::from(first + dst))
                    }
                    Err(_) => raw,
                };
                local.checked_sub(i64::from(offset + dst) * MICROS_PER_SECOND)
            }
        })
    }

    /// `s` as a local date-time; a zone in `s` is read and ignored.
    ///
    /// # Errors
    /// Never.
    pub fn parse_local(&self, s: &str, allow_zone: bool) -> Result<Option<i64>, KrishivError> {
        Ok(self
            .parse_fields(s, None)
            .filter(|(_, zone)| allow_zone || zone.is_none())
            .map(|(local, _)| local))
    }

    /// `s` as a date (days since the epoch): time and zone are ignored.
    ///
    /// # Errors
    /// Never.
    pub fn parse_date(&self, s: &str) -> Result<Option<i32>, KrishivError> {
        Ok(self
            .parse_fields(s, None)
            .and_then(|(local, _)| i32::try_from(local.div_euclid(MICROS_PER_DAY)).ok()))
    }

    /// The fields of `s`; zone names are looked up in `session`'s own row
    /// first (`None` where the zone is read and ignored).
    fn parse_fields(&self, s: &str, session: Option<&str>) -> Option<(i64, Option<Zone_>)> {
        let text: Vec<char> = s.chars().collect();
        let mut st = Parsed::default();
        let mut pos = 0;
        for (i, t) in self.toks.iter().enumerate() {
            match t {
                Tok::Literal(lit) => {
                    for c in lit.chars() {
                        // A literal matches its own character, a blank too
                        // (extra blanks are skipped before the next field).
                        if text.get(pos) != Some(&c) {
                            return None;
                        }
                        pos += 1;
                    }
                }
                Tok::Field(c, n) => {
                    // `SimpleDateFormat` skips spaces and tabs before a field.
                    while text.get(pos).is_some_and(|x| matches!(x, ' ' | '\t')) {
                        pos += 1;
                    }
                    // A number directly followed by another number in the
                    // pattern reads exactly its run length.
                    let abut =
                        matches!(self.toks.get(i + 1), Some(Tok::Field(d, dn)) if is_numeric(*d, *dn))
                            .then_some(*n);
                    pos = st.field(&text, pos, *c, *n, abut, session)?;
                }
            }
        }
        // `SimpleDateFormat` rejects a parse that consumed nothing (an empty
        // pattern), instead of resolving the default epoch fields.
        if pos == 0 {
            return None;
        }
        st.resolve(self.today)
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
        // 1582 lacks the ten labels 10-05..10-14.
        let skipped = if year == 1582 { 10 } else { 0 };
        return ((days + 3).rem_euclid(7) + 1, days - jan1 + 1 - skipped);
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

/// The epoch day a hybrid-calendar label names: the label itself from the
/// cutover on, before it the day the Julian date is (the labels are
/// Gregorian epoch counts, ten days off in 1582 and more before).
fn real_day(days: i64, year: i64, month: u32, day: u32) -> i64 {
    if days >= CUTOVER_DAYS {
        return days;
    }
    let a = (14 - i64::from(month)) / 12;
    let y = year + 4800 - a;
    let m = i64::from(month) + 12 * a - 3;
    let jdn = i64::from(day) + (153 * m + 2) / 5 + 365 * y + y.div_euclid(4) - 32083;
    jdn - 2_440_588
}

/// `Calendar.DAY_OF_WEEK_IN_MONTH` (the `F` letter): the occurrence number
/// of the weekday in its month, October 1582 lacking the 5th..14th.
fn dowim_of(year: i64, month: u32, day: u32) -> i64 {
    let gap = if year == 1582 && month == 10 && day >= 15 {
        10
    } else {
        0
    };
    (i64::from(day) - 1 - gap) / 7 + 1
}

/// Days in a month of the hybrid calendar.
fn hybrid_month_len(year: i64, month: u32) -> i64 {
    match (year, month) {
        (y, 2) if y < 1582 => 28 + i64::from(y.rem_euclid(4) == 0),
        (1582, 10) => 21,
        _ => i64::from(days_in_month(year, month)),
    }
}

/// The label (Gregorian epoch count of the written date) of a real epoch
/// day: itself from the cutover on, before it the day's Julian date.
fn label_of_real(real: i64) -> i64 {
    if real >= CUTOVER_DAYS {
        return real;
    }
    let c = real + 2_440_588 + 32_082;
    let d = (4 * c + 3).div_euclid(1461);
    let e = c - (1461 * d).div_euclid(4);
    let m = (5 * e + 2).div_euclid(153);
    let day = u32::try_from(e - (153 * m + 2).div_euclid(5) + 1).unwrap_or(1);
    let month = u32::try_from(m + 3 - 12 * m.div_euclid(10)).unwrap_or(1);
    let year = d - 4800 + m.div_euclid(10);
    // A Julian-only leap day has no label: it reads as March 1.
    days_from_civil(year, month, day)
        .or_else(|| days_from_civil(year, 3, 1))
        .unwrap_or(real)
}

/// Days in a year of the hybrid calendar (1582 lacks ten).
fn hybrid_year_len(year: i64) -> i64 {
    match year.cmp(&1582) {
        std::cmp::Ordering::Less => 365 + i64::from(year.rem_euclid(4) == 0),
        std::cmp::Ordering::Equal => 355,
        std::cmp::Ordering::Greater => 365 + i64::from(is_leap(year)),
    }
}

/// `Calendar.WEEK_OF_YEAR` for `Locale.US` (weeks start on Sunday, week 1
/// is the one with January 1st), for the real day `real` whose hybrid
/// calendar year and day of year are `year` and `doy`.
fn week_of_year(real: i64, year: i64, doy: i64) -> i64 {
    // Day of week of January 1st of the year (Sunday = 0).
    let jan1 = real - (doy - 1);
    let jan1_sun0 = (jan1 + 4).rem_euclid(7);
    let week = (doy - 1 + jan1_sun0) / 7 + 1;
    // The last days of December in the week that holds January 1st of the
    // next year belong to its week 1.
    let next_jan1 = jan1 + hybrid_year_len(year);
    let next_sun0 = (next_jan1 + 4).rem_euclid(7);
    let dec31 = next_jan1 - 1;
    if real > dec31 - next_sun0 && next_sun0 != 0 {
        1
    } else {
        week
    }
}

/// The day that decides the week year of `days`: itself, unless the week
/// (Sunday first, week 1 holds January 1st) is week 1 of the next year.
///
/// `days` is the written date's label and `real` its real epoch day (they
/// differ before the cutover); `doy` is the hybrid ordinal.
fn week_year_anchor(days: i64, real: i64, year: i64, doy: i64) -> i64 {
    if week_of_year(real, year, doy) == 1 && doy > 300 {
        // Late December in next year's week 1.
        days + 31
    } else {
        days
    }
}

fn is_leap(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// `Calendar.WEEK_OF_MONTH` for `Locale.US`, for the real day `real` that is
/// day `day` of `month` (October 1582 lacks 5..14).
fn week_of_month(real: i64, year: i64, month: u32, day: u32) -> i64 {
    let gap = if year == 1582 && month == 10 && day >= 15 {
        10
    } else {
        0
    };
    let first = real - i64::from(day) + 1 + gap;
    let first_sun0 = (first + 4).rem_euclid(7);
    (real - first + first_sun0) / 7 + 1
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

#[derive(Debug, Clone)]
enum Zone_ {
    Offset(i32),
    Region(String),
    /// A zone name: the region's standard offset at that time plus `dst`.
    Standard {
        id: &'static str,
        raw: i32,
        dst: i32,
    },
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
    /// The year was two digits (`yy`), read in the 80/20-year window.
    year_two: bool,
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
        session: Option<&str>,
    ) -> Option<usize> {
        // `SimpleDateFormat` skips spaces and tabs before a field, nothing
        // else (a newline or an ideographic space is not a blank).
        while text.get(pos).is_some_and(|x| matches!(x, ' ' | '\t')) {
            pos += 1;
        }
        let st = self.stamp();
        match c {
            'G' => {
                if let Some(p) = prefix(text, pos, &["AD"]) {
                    // The last era field wins.
                    self.era_bc = false;
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
                let (p, z) = zone(text, pos, session)?;
                self.zone = Some(z);
                return Some(p);
            }
            _ => {}
        }
        let (value, end) = number(text, pos, abut)?;
        let digits = end - pos;
        match c {
            'y' | 'Y' => {
                // A two-digit year waits for the rest of the date: the
                // window starts on a day, not at a year.
                self.year_two = n <= 2 && digits == 2;
                self.year = Some((value, st));
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
        self.week_real(year, mode).map(label_of_real)
    }

    /// [`Self::week_date`] as a real epoch day.
    fn week_real(&self, year: i64, mode: Mode) -> Option<i64> {
        let on_or_before = |d: i64, dow: i64| d - (dow_sun1(d) - dow).rem_euclid(7);
        // Real epoch days: the Julian calendar's before the cutover.
        let first_of = |month: u32| {
            days_from_civil(year, month, 1).map(|label| real_day(label, year, month, 1))
        };
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
                Some(week_start(first) + 7 * (self.woy.map_or(1, |w| w.0) - 1))
            }
            _ => {
                let m = u32::try_from(self.month0.map_or(0, |(v, _)| v) + 1).ok()?;
                let first = first_of(m)?;
                let dow = self.dow.map_or(1, |(d, _)| d);
                let n = self.dowim.map_or(1, |(v, _)| v);
                if n >= 0 {
                    Some(on_or_before(first + 7 * n - 1, dow))
                } else {
                    let len = hybrid_month_len(year, m);
                    Some(on_or_before(first + len + 7 * (n + 1) - 1, dow))
                }
            }
        }
    }

    /// A non-lenient `Calendar` rejects a result that disagrees with a
    /// field that was set. `days` is the result's label and `(y, m, d)` the
    /// hybrid-calendar date it names (`1500-02-29` for a Julian-only leap
    /// day, whose label is March 1).
    fn week_fields_agree(&self, days: i64, year: i64, (y, m, d): (i64, u32, u32)) -> bool {
        let (_, hdoy) = hybrid_week_fields(days, y, m, d);
        let real = real_day(days, y, m, d);
        let year_ok = if self.year_is_week {
            civil_from_days(week_year_anchor(days, real, y, hdoy)).0 == year
        } else {
            y == year
        };
        // A winning week year (`Y`) is `Calendar.setWeekDate`, which sets
        // the month and day itself: the parsed ones are ignored, the other
        // fields must still agree.
        let by_week_year = self.year_is_week;
        year_ok
            && (by_week_year || self.month0.is_none_or(|(v, _)| v == i64::from(m) - 1))
            && (by_week_year || self.dom.is_none_or(|(v, _)| v == i64::from(d)))
            && self.doy.is_none_or(|(v, _)| v == hdoy)
            && self
                .woy
                .is_none_or(|(v, _)| v == week_of_year(real, y, hdoy))
            && self
                .wom
                .is_none_or(|(v, _)| v == week_of_month(real, y, m, d))
            && self.dowim.is_none_or(|(v, _)| v == dowim_of(y, m, d))
            && self.dow.is_none_or(|(v, _)| v == dow_sun1(real))
    }

    fn resolve(self, today: (i64, u32, u32)) -> Option<(i64, Option<Zone_>)> {
        if !self.year_two {
            // The year of the era is positive: no year 0 (nor `0000 BC`).
            if self.year.is_some_and(|(y, _)| y <= 0) {
                return None;
            }
            return self.resolve_in(self.year.map_or(1970, |(y, _)| y), self.era_bc);
        }
        // `yy`: the century of the window start, one later when the date
        // falls before the window's first day.
        let two = self.year.map_or(0, |(y, _)| y);
        let (ny, nm, nd) = today;
        let start = ny - 80;
        let start_days = days_from_civil(start, nm, nd.min(days_in_month(start, nm)))?;
        let base = start - start.rem_euclid(100) + two;
        // The provisional century may lack the date (`00-02-29` in 1900).
        // The century is chosen on the year of the era, before a BC era
        // turns the year around (`99 BC` is 1999 BC).
        let chosen = match self.resolve_in(base, false) {
            Some(first) if first.0.div_euclid(MICROS_PER_DAY) >= start_days => base,
            _ => base + 100,
        };
        self.resolve_in(chosen, self.era_bc)
    }

    fn resolve_in(&self, year: i64, bc: bool) -> Option<(i64, Option<Zone_>)> {
        if self.bad_hour {
            return None;
        }
        let mut year = year;
        if bc {
            year = 1 - year;
        }
        // No calendar field reaches this far; a huge one would overflow
        // the week arithmetic below, and a non-lenient calendar refuses it.
        let huge = |f: Option<(i64, u32)>| f.is_some_and(|(v, _)| v.unsigned_abs() > 100_000);
        if huge(self.woy)
            || huge(self.wom)
            || huge(self.dowim)
            || huge(self.dom)
            || huge(self.doy)
            || huge(self.month0)
        {
            return None;
        }
        let weeky = self.woy.is_some() || self.wom.is_some() || self.dowim.is_some();
        // A winning week year (`Y`) is the week date of that week year, its
        // week of year (1 by default) and day of week (Sunday by default),
        // as `CalendarBuilder.establish` calls `setWeekDate`; a month or day
        // beside it is ignored (Spark 4.2: '2020' with 'YYYY' is 2019-12-29,
        // '2020-01-03-05' with 'YYYY-ww-MM-dd' too).
        let mut julian_leap = false;
        let mode = if self.year_is_week {
            Mode::Woy
        } else if weeky {
            self.week_mode()
        } else if self.dow.is_some() && self.dom.is_none() && self.doy.is_none() {
            // A weekday alone (with at most a month) is the first such
            // weekday of the month, as `GregorianCalendar` falls back to
            // `DAY_OF_WEEK_IN_MONTH` 1.
            Mode::Dowim
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
            if !(1..=hybrid_year_len(year)).contains(&d) {
                return None;
            }
            // 1582 lacks the labels 10-05..10-14, so its ordinals past
            // October 4 count ten labels further.
            let (m, dd) = month_day_of(year, if year == 1582 && d > 277 { d + 10 } else { d });
            // The other date fields that were set must agree.
            if self.month0.is_some_and(|(v, _)| v != i64::from(m) - 1)
                || self.dom.is_some_and(|(v, _)| v != i64::from(dd))
            {
                return None;
            }
            // A Julian-only leap day (1500-02-29) reads as March 1.
            days_from_civil(year, m, dd).or_else(|| {
                julian_leap = true;
                days_from_civil(year, 3, 1)
            })?
        } else {
            let m = self.month0.map_or(0, |(v, _)| v);
            let d = self.dom.map_or(1, |(v, _)| v);
            if !(0..12).contains(&m) || d < 1 {
                return None;
            }
            let (mo, dd) = (u32::try_from(m + 1).ok()?, u32::try_from(d).ok()?);
            match days_from_civil(year, mo, dd) {
                Some(days) => days,
                // February 29 of a Julian-only leap year (1500, ...): the
                // hybrid calendar has it, the proleptic one reads March 1.
                None if (mo, dd) == (2, 29) && year <= 1582 && year % 4 == 0 => {
                    julian_leap = true;
                    days_from_civil(year, 3, 1)?
                }
                None => return None,
            }
        };
        // The labels 1582-10-05..14 do not exist in the hybrid calendar.
        let cutover = days_from_civil(1582, 10, 4).unwrap_or(i64::MIN);
        if days > cutover && days < cutover + 11 {
            return None;
        }
        // A Julian-only leap day is checked as the day it is (February 29),
        // before it reads as March 1.
        let label = if julian_leap {
            (year, 2, 29)
        } else {
            civil_from_days(days)
        };
        if (weeky || self.year_is_week || self.dow.is_some() || self.doy.is_some())
            && !self.week_fields_agree(days, year, label)
        {
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
        // Every clock field that was set must agree with the result.
        if self.hour_of_day.is_some_and(|(h, _)| h != hour)
            || self.hour12.is_some_and(|(h, _)| h != hour % 12)
            || (self.hour_of_day.is_some() || self.hour12.is_some())
                && self.pm.is_some_and(|(pm, _)| pm != (hour >= 12))
            || self.hour_of_day.is_some_and(|(h, _)| !(0..24).contains(&h))
            || self.hour12.is_some_and(|(h, _)| !(0..12).contains(&h))
        {
            return None;
        }
        let minute = self.minute.unwrap_or(0);
        let second = self.second.unwrap_or(0);
        let millis = self.millis.unwrap_or(0);
        if !(0..60).contains(&minute) || !(0..60).contains(&second) || !(0..1000).contains(&millis)
        {
            return None;
        }
        let tod = ((hour * 60 + minute) * 60 + second) * MICROS_PER_SECOND + millis * 1000;
        let local = days.checked_mul(MICROS_PER_DAY)?.checked_add(tod)?;
        Some((local, self.zone.clone()))
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
        // Julian leap years (every fourth) before the cutover year.
        let len = if m == 2 && year < 1582 {
            28 + i64::from(year.rem_euclid(4) == 0)
        } else {
            i64::from(days_in_month(year, m))
        };
        if doy <= len {
            return (m, u32::try_from(doy).unwrap_or(1));
        }
        doy -= len;
    }
    (12, 31)
}

/// Today's date, the anchor of `SimpleDateFormat`'s two-digit year window
/// (the 80 years before it to the 20 after).
fn today() -> (i64, u32, u32) {
    // Miri's isolation has no wall clock, so it reads a fixed 2026-07-01.
    #[cfg(miri)]
    let now_secs = 1_782_864_000_i64;
    #[cfg(not(miri))]
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64);
    civil_from_days(now_secs.div_euclid(86_400))
}

/// An unsigned integer at `pos` (`limit` digits at most), as the digits
/// and where they end. Any decimal digit counts, as `DecimalFormat` reads
/// a numeric field (`٢٠٢٤` is 2024).
fn number(text: &[char], pos: usize, limit: Option<usize>) -> Option<(i64, usize)> {
    number_of(text, pos, limit, java_digit)
}

/// [`number`] with ASCII digits only, as the zone fields read them.
fn ascii_number(text: &[char], pos: usize, limit: Option<usize>) -> Option<(i64, usize)> {
    number_of(text, pos, limit, |c| c.to_digit(10))
}

fn number_of(
    text: &[char],
    pos: usize,
    limit: Option<usize>,
    digit: fn(char) -> Option<u32>,
) -> Option<(i64, usize)> {
    let mut end = pos;
    let mut value: i64 = 0;
    while let Some(d) = text.get(end).and_then(|&c| digit(c)) {
        if limit.is_some_and(|l| end - pos >= l) {
            break;
        }
        value = value.checked_mul(10)?.checked_add(i64::from(d))?;
        end += 1;
    }
    (end > pos).then_some((value, end))
}

/// `Character.digit(c, 10)` in the Basic Multilingual Plane (a field is
/// read one UTF-16 unit at a time, so no supplementary digit counts).
fn java_digit(c: char) -> Option<u32> {
    if let Some(d) = c.to_digit(10) {
        return Some(d);
    }
    let cp = u32::from(c);
    let i = DIGIT_ZEROS.partition_point(|&z| z <= cp).checked_sub(1)?;
    let d = cp.checked_sub(*DIGIT_ZEROS.get(i)?)?;
    (d < 10).then_some(d)
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
    let (hh, mut e) = ascii_number(text, pos + 1, Some(2))?;
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
        let (m, e2) = ascii_number(text, e, Some(2))?;
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

/// A `z` or `Z` field, which `SimpleDateFormat.subParse` reads alike: a
/// sign and `hhmm` (RFC 822); `GMT` (any case), alone or with a sign, one or
/// two hour digits, a colon and two minute digits; else a zone name (see
/// [`parse_zone_name`]). Nothing else: no `Z`, `+hh:mm` or region id.
fn zone(text: &[char], pos: usize, session: Option<&str>) -> Option<(usize, Zone_)> {
    let c = *text.get(pos)?;
    if matches!(c, '+' | '-') {
        return numeric_zone(text, pos + 1, c == '-', false);
    }
    if let Some(p) = prefix(text, pos, &["GMT"]) {
        return match text.get(p) {
            Some(&s @ ('+' | '-')) => numeric_zone(text, p + 1, s == '-', true),
            _ => Some((p, Zone_::Offset(0))),
        };
    }
    parse_zone_name(text, pos, session)
}

/// `SimpleDateFormat.subParseNumericZone` from the first hour digit: two
/// hour digits, or one before the colon (`colon`), then two minute digits.
fn numeric_zone(
    text: &[char],
    start: usize,
    negative: bool,
    colon: bool,
) -> Option<(usize, Zone_)> {
    let (hh, mut e) = ascii_number(text, start, Some(2))?;
    if e - start != 2 && !colon {
        return None;
    }
    if colon {
        if text.get(e) != Some(&':') {
            return None;
        }
        e += 1;
    }
    let (mm, end) = ascii_number(text, e, Some(2))?;
    if end - e != 2 || hh > 23 || mm > 59 {
        return None;
    }
    let secs = i32::try_from(hh * 3600 + mm * 60).ok()?;
    Some((end, Zone_::Offset(if negative { -secs } else { secs })))
}

/// `SimpleDateFormat.subParseZoneString`: the session zone's own row of the
/// JVM's zone strings (`session`, its id), else the first row with a name
/// (long standard, short standard, long daylight, short daylight) that
/// starts the text, ignoring case; only that name is read (`PSTfoo` is
/// `PST`). So `IST` is India in `Asia/Kolkata`, Ireland in `Europe/Dublin`
/// and Israel elsewhere. A standard name is the zone's standard offset at
/// that time (`ALMT` is +06:00 in January 2024, +05:00 since March), a
/// daylight one adds its saving; a name its row uses for both kinds of time,
/// or a daylight name of a zone without saving, leaves the zone's own rules
/// to decide. The JVM default zone's row, which Java tries second, is not:
/// it is the host's, not the query's.
fn parse_zone_name(text: &[char], pos: usize, session: Option<&str>) -> Option<(usize, Zone_)> {
    if let Some(row) = session.and_then(zone_row)
        && let Some(found) = match_zone_row(text, pos, &row)
    {
        return Some(found);
    }
    ZONE_STRINGS
        .iter()
        .find_map(|row| match_zone_row(text, pos, row))
}

/// `SimpleDateFormat.matchZoneString` on one row, and the zone it gives.
fn match_zone_row(text: &[char], pos: usize, row: &ZoneStrings) -> Option<(usize, Zone_)> {
    row.names.iter().enumerate().find_map(|(i, name)| {
        let end = name_at(text, pos, name)?;
        let same = row
            .names
            .get(i + 2)
            .is_some_and(|other| java_equals_ignore_case(name, other));
        let daylight = i >= 2;
        let zone = if same || (daylight && row.dst == 0) {
            if crate::datetime::resolve_zone(row.id).is_ok() {
                Zone_::Region(row.id.to_owned())
            } else {
                Zone_::Offset(row.raw)
            }
        } else {
            Zone_::Standard {
                id: row.id,
                raw: row.raw,
                dst: if daylight { row.dst } else { 0 },
            }
        };
        Some((end, zone))
    })
}

/// Where `name` (non-empty) ends if it starts `text` at `pos`, ignoring
/// case as `String.regionMatches(true, ..)` does.
fn name_at(text: &[char], pos: usize, name: &str) -> Option<usize> {
    let mut end = pos;
    for c in name.chars() {
        if !java_same_ignoring_case(*text.get(end)?, c) {
            return None;
        }
        end += 1;
    }
    (end > pos).then_some(end)
}

fn java_equals_ignore_case(a: &str, b: &str) -> bool {
    a.chars().count() == b.chars().count()
        && a.chars()
            .zip(b.chars())
            .all(|(x, y)| java_same_ignoring_case(x, y))
}

/// Java's per-character case-blind equality (single-character mappings).
fn java_same_ignoring_case(a: char, b: char) -> bool {
    let upper = |c: char| {
        let mut u = c.to_uppercase();
        match (u.next(), u.next()) {
            (Some(x), None) => x,
            _ => c,
        }
    };
    let lower = |c: char| {
        let mut l = c.to_lowercase();
        match (l.next(), l.next()) {
            (Some(x), None) => x,
            _ => c,
        }
    };
    a == b || upper(a) == upper(b) || lower(upper(a)) == lower(upper(b))
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
        LegacyPattern::new(p)
            .unwrap()
            .parse_local(text, true)
            .unwrap()
    }

    // Spark 4.2.0 under LEGACY (probe 2026-11-02).
    #[test]
    fn a_day_of_year_must_agree_and_a_lone_weekday_is_the_first_of_the_month() {
        // Spark 4.2.0 under LEGACY (probe 2026-11-02).
        let d = |y, m, dd| days_from_civil(y, m, dd).unwrap() * MICROS_PER_DAY;
        assert_eq!(local("yyyy-DDD-MM-dd", "2024-002-01-01"), None);
        assert_eq!(
            local("yyyy-DDD-MM-dd", "2024-001-01-01"),
            Some(d(2024, 1, 1))
        );
        assert_eq!(local("yyyy E", "2024 Tue"), Some(d(2024, 1, 2)));
        assert_eq!(local("yyyy E", "2024 Mon"), Some(d(2024, 1, 1)));
        assert_eq!(local("yyyy EEEE", "2024 Tuesday"), Some(d(2024, 1, 2)));
        assert_eq!(local("yyyy-MM E", "2024-03 Tue"), Some(d(2024, 3, 5)));
        assert_eq!(local("yyyy-MM E", "2024-03 Fri"), Some(d(2024, 3, 1)));
        assert_eq!(local("yyyy-DDD", "2024-060"), Some(d(2024, 2, 29)));
    }

    // Spark 4.2.0 under LEGACY (probe 2026-11-02).
    // Spark 4.2.0 under LEGACY (probe 2026-11-02).
    #[test]
    fn week_numbers_follow_the_hybrid_calendar() {
        let f = |y, m, d| {
            fmt(
                "yyyy-MM-dd E w W",
                days_from_civil(y, m, d).unwrap() * MICROS_PER_DAY,
            )
        };
        assert_eq!(f(1582, 5, 31), "1582-05-31 Thu 22 5");
        assert_eq!(f(1582, 1, 1), "1582-01-01 Mon 1 1");
        assert_eq!(f(1582, 10, 4), "1582-10-04 Thu 40 1");
        assert_eq!(f(1582, 10, 15), "1582-10-15 Fri 40 1");
        assert_eq!(f(1582, 10, 20), "1582-10-20 Wed 41 2");
        assert!(f(1500, 3, 15).ends_with(" 12 3"));
    }

    #[test]
    fn an_empty_pattern_parses_nothing_and_a_leap_day_finds_its_century() {
        assert_eq!(local("", ""), None);
        assert_eq!(local("", "2024-01-01"), None);
        let d = days_from_civil(2000, 2, 29).unwrap() * MICROS_PER_DAY;
        assert_eq!(local("yy-MM-dd", "00-02-29"), Some(d));
    }

    // Spark 4.2.0 under LEGACY (probe 2026-11-02).
    #[test]
    fn year_zero_and_literal_blanks_follow_simple_date_format() {
        assert_eq!(local("yyyy-MM-dd", "0000-01-01"), None);
        assert_eq!(local("yyyy-MM-dd G", "0000-01-01 BC"), None);
        let p = "yyyy-MM-dd 'at' HH:mm";
        for bad in [
            "2024-01-01at 12:30",
            "2024-01-01  at 12:30",
            "2024-01-01 at12:30",
        ] {
            assert_eq!(local(p, bad), None, "{bad}");
        }
        assert!(local(p, "2024-01-01 at 12:30").is_some());
        assert!(local("yyyy-MM-dd HH:mm", "2024-01-01  12:30").is_some());
        assert_eq!(local("yyyy-MM-dd HH:mm", "2024-01-0112:30"), None);
    }

    // Codex-reported cases against the hybrid calendar; the weekday and the
    // 1582 ordinals agree with the formatter (probed: D=278 on 1582-10-15,
    // 355 on 12-31, 'Thu' on 1582-05-31).
    #[test]
    fn day_of_year_and_weekday_parse_on_the_hybrid_calendar() {
        let d = |m, dd| days_from_civil(1582, m, dd).unwrap() * MICROS_PER_DAY;
        assert_eq!(local("yyyy-DDD", "1582-278"), Some(d(10, 15)));
        assert_eq!(local("yyyy-DDD", "1582-355"), Some(d(12, 31)));
        assert_eq!(local("yyyy-DDD", "1582-356"), None);
        assert_eq!(local("yyyy-DDD", "1582-277"), Some(d(10, 4)));
        assert_eq!(local("yyyy-MM-dd E", "1582-05-31 Thu"), Some(d(5, 31)));
        assert_eq!(local("yyyy-MM-dd E", "1582-05-31 Mon"), None);
    }

    #[test]
    fn week_fields_resolve_on_the_hybrid_calendar() {
        // Spark 4.2.0 under LEGACY: '1582-22-4' (yyyy-ww-u) is May 31.
        let d = days_from_civil(1582, 5, 31).unwrap() * MICROS_PER_DAY;
        assert_eq!(local("yyyy-ww-u", "1582-22-4"), Some(d));
        assert_eq!(
            label_of_real(real_day(days_from_civil(1500, 3, 15).unwrap(), 1500, 3, 15)),
            days_from_civil(1500, 3, 15).unwrap()
        );
    }

    // Spark 4.2.0 under LEGACY (probe 2026-11-02).
    #[test]
    fn numeric_zones_and_the_occurrence_number_follow_simple_date_format() {
        let ok = |p: &str, t: &str| LegacyPattern::new(p).unwrap().parse_local(t, true).unwrap();
        assert!(ok("yyyy-MM-dd Z", "2024-01-01 +0530").is_some());
        assert!(ok("yyyy-MM-dd Z", "2024-01-01 GMT+05:30").is_some());
        for bad in ["2024-01-01 +05:30", "2024-01-01 +05"] {
            assert!(ok("yyyy-MM-dd Z", bad).is_none(), "{bad}");
        }
        for bad in ["2024-01-01 GMT+0530", "2024-01-01 +05:30"] {
            assert!(ok("yyyy-MM-dd z", bad).is_none(), "{bad}");
        }
        let f = |m, dd| fmt("F", days_from_civil(1582, m, dd).unwrap() * MICROS_PER_DAY);
        assert_eq!(f(10, 15), "1");
        assert_eq!(f(10, 31), "3");
        assert!(ok("yyyy-MM-dd F", "1582-10-15 1").is_some());
        assert!(ok("yyyy-MM-dd F", "1582-10-15 3").is_none());
    }

    // Spark 4.2.0 under LEGACY (probe 2026-11-02).
    #[test]
    fn a_lone_week_year_and_julian_leap_days_resolve_like_spark() {
        let d = |y, m, dd| days_from_civil(y, m, dd).unwrap() * MICROS_PER_DAY;
        assert_eq!(local("YYYY", "2020"), Some(d(2019, 12, 29)));
        assert_eq!(local("YYYY-MM-dd", "2020-03-05"), Some(d(2019, 12, 29)));
        assert_eq!(local("YYYY-MM", "2020-03"), Some(d(2019, 12, 29)));
        assert_eq!(local("YYYY-MM-dd", "2021-01-01"), Some(d(2020, 12, 27)));
        assert_eq!(local("YYYY E", "2020 Tue"), Some(d(2019, 12, 31)));
        assert_eq!(local("YYYY-DDD", "2020-053"), None);
        for (p, t) in [
            ("yyyy-MM-dd", "1500-02-29"),
            ("yyyy-DDD", "1500-060"),
            ("yyyy-DDD", "1500-061"),
        ] {
            assert_eq!(local(p, t), Some(d(1500, 3, 1)), "{t}");
        }
        assert_eq!(local("yyyy-MM-dd", "1700-02-29"), None);
    }

    // Spark 4.2.0 under LEGACY (probe 2026-11-02).
    #[test]
    fn zone_display_names_parse_short_and_long_and_the_week_year_uses_the_hybrid_day() {
        let p = LegacyPattern::new("yyyy-MM-dd HH:mm z").unwrap();
        let offset = |t: &str| {
            let (local, _) = p.parse_fields(t, None)?;
            let utc = p.parse_timestamp(t, &FixedOffset(0)).ok()??;
            i32::try_from((local - utc) / MICROS_PER_SECOND).ok()
        };
        // Spark 4.2.0: 12:00 PDT is 19:00 UTC, whether short or long.
        assert_eq!(offset("2024-07-01 12:00 PDT"), Some(-7 * 3600));
        assert_eq!(
            offset("2024-07-01 12:00 Pacific Daylight Time"),
            Some(-7 * 3600)
        );
        assert_eq!(
            offset("2024-07-01 12:00 Central European Summer Time"),
            Some(2 * 3600)
        );
        // 1581-12-27 is in week-year 1581 on the Julian calendar.
        let t = days_from_civil(1581, 12, 27).unwrap() * MICROS_PER_DAY;
        assert_eq!(fmt("yyyy YYYY w", t), "1581 1581 52");
    }

    #[test]
    fn a_two_digit_year_with_bc_picks_its_century_on_the_year_of_the_era() {
        let mut p = LegacyPattern::new("yy G").unwrap();
        p.today = (2026, 11, 2);
        // Spark 4.2.0: '99 BC' is 1999 BC (astronomical year -1998).
        let want = days_from_civil(-1998, 1, 1).unwrap() * MICROS_PER_DAY;
        assert_eq!(p.parse_local("99 BC", true).unwrap(), Some(want));
    }

    #[test]
    fn the_last_era_field_wins() {
        let d = days_from_civil(1, 1, 1).unwrap() * MICROS_PER_DAY;
        // Spark 4.2.0 under LEGACY: '0001-01-01 BC AD' is year 1 AD.
        assert_eq!(local("yyyy-MM-dd G G", "0001-01-01 BC AD"), Some(d));
    }

    #[test]
    fn oversized_fields_fail_instead_of_overflowing() {
        assert_eq!(local("yyyy-ww-u", "2024-9223372036854775807-1"), None);
        assert_eq!(local("yyyy-MM-W-u", "2024-01-9223372036854775807-1"), None);
        assert_eq!(local("yyyy-MM-F-u", "2024-01-9223372036854775807-1"), None);
    }

    #[test]
    fn day_of_year_counts_the_cutover_gap() {
        let f = |days: i64| fmt("D", days * MICROS_PER_DAY);
        let d = |m, dd| days_from_civil(1582, m, dd).unwrap();
        assert_eq!(f(d(10, 4)), "277");
        assert_eq!(f(d(10, 15)), "278");
        assert_eq!(f(d(12, 31)), "355");
        assert_eq!(f(days_from_civil(1583, 1, 1).unwrap()), "1");
    }

    #[test]
    fn every_supplied_hour_field_must_agree() {
        for text in [
            "2024-01-01 23 01 AM",
            "2024-01-01 01 23 AM",
            "2024-01-01 01 13 AM",
            "2024-01-01 25 01 AM",
            "2024-01-01 13 01 AM",
        ] {
            assert_eq!(local("yyyy-MM-dd HH hh a", text), None, "{text}");
        }
        assert!(local("yyyy-MM-dd HH hh a", "2024-01-01 13 01 PM").is_some());
    }

    #[test]
    fn a_zone_is_refused_when_the_caller_allows_none() {
        let p = LegacyPattern::new("yyyy-MM-dd XXX").unwrap();
        assert!(p.parse_local("2024-01-01 +05:30", false).unwrap().is_none());
        assert!(p.parse_local("2024-01-01 +05:30", true).unwrap().is_some());
    }

    #[test]
    fn the_two_digit_year_window_starts_on_a_day() {
        // Spark 4.2.0 under LEGACY with the clock at 2026-11-02 (probe): the
        // window starts on 1946-11-02.
        let mut p = LegacyPattern::new("yy-MM-dd").unwrap();
        p.today = (2026, 11, 2);
        let at = |text: &str| p.parse_local(text, true).unwrap();
        let day = |y: i64, m: u32, d: u32| days_from_civil(y, m, d).unwrap() * MICROS_PER_DAY;
        assert_eq!(at("46-01-01"), Some(day(2046, 1, 1)));
        assert_eq!(at("46-11-01"), Some(day(2046, 11, 1)));
        assert_eq!(at("46-11-02"), Some(day(1946, 11, 2)));
        assert_eq!(at("46-11-03"), Some(day(1946, 11, 3)));
        assert_eq!(at("46-12-31"), Some(day(1946, 12, 31)));
        // A leap day whose provisional century lacks it.
        assert_eq!(at("00-02-29"), Some(day(2000, 2, 29)));
    }

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
    fn daylight_zone_names_are_their_saving_time_offset() {
        let utc = crate::datetime::resolve_zone("UTC").unwrap();
        let p = LegacyPattern::new("yyyy-MM-dd HH:mm z").unwrap();
        let at = |s: &str| p.parse_timestamp(s, &utc).unwrap().unwrap();
        assert_eq!(at("2024-07-01 12:00 PDT"), micros(2024, 7, 1, 19, 0, 0, 0));
        // Spark 4.2.0 under LEGACY: other names are fixed offsets all year.
        for (name, hours) in [
            ("CEST", 2),
            ("CET", 1),
            ("BST", 1),
            ("EEST", 3),
            ("AEST", 10),
            ("JST", 9),
        ] {
            assert_eq!(
                at(&format!("2024-07-01 12:00 {name}")),
                micros(2024, 7, 1, 12 - hours, 0, 0, 0),
                "{name}"
            );
            assert_eq!(
                at(&format!("2024-01-01 12:00 {name}")),
                micros(2024, 1, 1, 12 - hours, 0, 0, 0),
                "{name} in winter"
            );
        }
        assert_eq!(at("2024-07-01 12:00 ACST"), micros(2024, 7, 1, 2, 30, 0, 0));
        assert_eq!(at("2024-01-01 12:00 EDT"), micros(2024, 1, 1, 16, 0, 0, 0));
        assert_eq!(at("2024-07-01 12:00 AKDT"), micros(2024, 7, 1, 20, 0, 0, 0));
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
        LegacyPattern::new(p).unwrap().parse_local(s, true).unwrap()
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
        // Only spaces and tabs are skipped before a field (Spark 4.2.0).
        assert_eq!(parse("yyyy-MM-dd HH", "2024-01-01 \n12"), None);
        assert_eq!(parse("yyyy-MM-dd HH", "2024-01-01 \u{3000}12"), None);
        assert_eq!(parse("yyyy-MM-ddHH", "2024-01-01\n12"), None);
        assert_eq!(
            parse("yyyy-MM-dd HH", "2024-01-01 \t12"),
            Some(micros(2024, 1, 1, 12, 0, 0, 0))
        );
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
