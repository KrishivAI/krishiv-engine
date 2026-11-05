// SPDX-License-Identifier: Apache-2.0
//! The pre-3.0 datetime patterns of `spark.sql.legacy.timeParserPolicy =
//! LEGACY`, over the Julian/Gregorian hybrid calendar. The SQL functions use
//! `java.text.SimpleDateFormat`: its letters print and it parses non-leniently
//! (a month 13 or a February 30 is no timestamp). Spark's JSON, CSV and XML
//! options use commons-lang's `FastDateFormat` instead, a lenient parser and
//! its own printer letters: that is the `fast` flag of [`LegacyPattern`],
//! modelled in the `fast` submodule.
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

mod fast;

use krishiv_common::error::{InvalidDatetimePatternWithSuggestion, KrishivError};

use super::format::{MICROS_PER_DAY, MONTHS, zone_name};
use super::legacy_zones::{DIGIT_ZEROS, ZONE_STRINGS, ZoneStrings, zone_row};
use super::{DOC_ROOT, Zone, unsupported};
use crate::cast::TimeZoneRules;
use crate::cast::datetime::{civil_from_days, days_from_civil, days_in_month};

mod week_year;
use week_year::{week_of_year, week_year_anchor};

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
    /// Set for Spark's JSON, CSV and XML options: commons-lang
    /// `FastDateFormat` prints and a lenient `FastDateParser` reads (see
    /// `fast`), instead of the SQL functions' `SimpleDateFormat`.
    fast: bool,
    /// A fast pattern Spark fails to build while reading: its formatter is
    /// lazy, so the failure surfaces per value and the lenient fallback
    /// takes over. Matches nothing, prints nothing.
    broken: bool,
}

impl LegacyPattern {
    /// Compiles `pattern` as the `SimpleDateFormat` the SQL functions use.
    ///
    /// # Errors
    /// `INVALID_DATETIME_PATTERN.WITH_SUGGESTION` for a letter the legacy
    /// formatters reject, a run of `X` over three, or an unterminated quote.
    pub fn new(pattern: &str) -> Result<Self, KrishivError> {
        Self::compile(pattern, false)
    }

    /// Compiles `pattern` as the `FastDateFormat` Spark's JSON, CSV and XML
    /// options use under `timeParserPolicy=LEGACY`.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn new_fast(pattern: &str) -> Result<Self, KrishivError> {
        Self::compile(pattern, true)
    }

    /// A fast pattern for reading that Spark would fail to build: every
    /// parse is a miss (the caller's lenient fallback then decides).
    #[must_use]
    pub fn new_fast_broken(pattern: &str) -> Self {
        Self {
            pattern: pattern.to_owned(),
            toks: Vec::new(),
            today: today(),
            fast: true,
            broken: true,
        }
    }

    /// The pattern error of a fast pattern that was built lazily.
    #[must_use]
    pub fn build_error(&self) -> Option<KrishivError> {
        if !self.broken {
            return None;
        }
        Some(
            Self::compile(&self.pattern, true)
                .err()
                .unwrap_or_else(|| unsupported("a datetime pattern Spark cannot build")),
        )
    }

    fn compile(pattern: &str, fast: bool) -> Result<Self, KrishivError> {
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
            fast,
            broken: false,
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
        // Spark's `FastDateFormat` date formatter prints zone fields in the
        // JVM default zone, which the probes ran as UTC.
        self.write(i64::from(days), 0, self.fast.then_some((0, "UTC")))
    }

    fn write(
        &self,
        days: i64,
        tod: i64,
        zone: Option<(i32, &str)>,
    ) -> Result<String, KrishivError> {
        if let Some(e) = self.build_error() {
            return Err(e);
        }
        // The hybrid calendar has no 1582-10-05..14: the rebase of such a
        // label lands on the first Gregorian day.
        let days = if self.fast && (CUTOVER_DAYS - 10..CUTOVER_DAYS).contains(&days) {
            CUTOVER_DAYS
        } else {
            days
        };
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
                    // `Calendar.getWeekYear` is not a year of the era: a BC week
                    // year prints as the signed astronomical year (Spark 4.2:
                    // `-0005-06-01` with 'Y YY YYY' is `-5 -05 -005`).
                    let (wy, _, _) = civil_from_days(week_year_anchor(days, real, year, doy));
                    if self.fast {
                        fast::year_field(&mut out, wy, n);
                    } else {
                        if wy < 0 {
                            out.push('-');
                        }
                        if n == 2 {
                            let _ = write!(out, "{:02}", wy.unsigned_abs() % 100);
                        } else {
                            pad(
                                &mut out,
                                i64::try_from(wy.unsigned_abs()).unwrap_or(i64::MAX),
                                n,
                            );
                        }
                    }
                }
                'y' => {
                    let era_year = if year >= 1 { year } else { 1 - year };
                    if n == 2 {
                        let _ = write!(out, "{:02}", era_year % 100);
                    } else {
                        // `FastDatePrinter` pads a year to four digits.
                        pad(&mut out, era_year, if self.fast { n.max(4) } else { n });
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
                'S' if self.fast => pad(&mut out, self.fraction(tod % MICROS_PER_SECOND)?, n),
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
                                || ["UTC+", "UTC-", "GMT+", "GMT-", "UT+", "UT-"]
                                    .iter()
                                    .any(|p| id.starts_with(p));
                            if fixed && offset != 0 {
                                // `java.util.TimeZone` names a fixed offset `GMT+hh:mm`.
                                let a = offset.unsigned_abs();
                                let sign = if offset < 0 { '-' } else { '+' };
                                let _ = write!(out, "GMT{sign}{:02}:{:02}", a / 3600, a / 60 % 60);
                            } else {
                                // `TimeZone.getTimeZone(ZoneId)` reads the offset `Z` as
                                // `UTC`; `UT` is no `TimeZone` id, so it falls back to `GMT`.
                                let id = match id {
                                    "Z" => "UTC",
                                    "UT" => "GMT",
                                    _ if fixed => {
                                        if id.starts_with("GMT")
                                            || id.starts_with("UT+")
                                            || id.starts_with("UT-")
                                        {
                                            "GMT"
                                        } else {
                                            "UTC"
                                        }
                                    }
                                    _ => id,
                                };
                                out.push_str(&zone_name(
                                    id,
                                    offset,
                                    local.div_euclid(MICROS_PER_SECOND),
                                    n >= 4,
                                )?);
                            }
                        }
                        'Z' if self.fast && n == 2 => {
                            write_offset(&mut out, offset, true, true, true);
                        }
                        'Z' if self.fast && n >= 3 => {
                            write_offset(&mut out, offset, true, true, false);
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
    /// when `s` does not match (Spark raises `CANNOT_PARSE_TIMESTAMP`) or
    /// the instant is past the microsecond range.
    ///
    /// # Errors
    /// Never.
    pub fn parse_timestamp(
        &self,
        s: &str,
        session: &dyn TimeZoneRules,
    ) -> Result<Option<i64>, KrishivError> {
        if self.fast {
            return Ok(self.fast_instant(s, session));
        }
        Ok(self.instant(s, session).ok())
    }

    /// `s` as an instant, telling a text that does not match from a date
    /// past the microsecond range: Spark's `millisToMicros` raises the
    /// latter as an `ArithmeticException` whatever the parse-error policy.
    pub(crate) fn instant(&self, s: &str, session: &dyn TimeZoneRules) -> Result<i64, Unmatched> {
        let (local, zone) = self
            .parse_fields(s, session.region_id())
            .ok_or(Unmatched::Text)?;
        let local = local.ok_or(Unmatched::Overflow)?;
        let shift = |offset: i32| {
            local
                .checked_sub(i64::from(offset) * MICROS_PER_SECOND)
                .ok_or(Unmatched::Overflow)
        };
        match zone {
            None => legacy_local_to_utc(local, session),
            Some(Zone_::Offset(o)) => shift(o),
            Some(Zone_::Region(id)) => match crate::datetime::resolve_zone(&id) {
                Ok(z) => legacy_local_to_utc(local, &z),
                Err(_) => Err(Unmatched::Text),
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
                shift(offset + dst)
            }
        }
    }

    /// `s` as a local date-time; a zone in `s` is read and ignored.
    ///
    /// # Errors
    /// Never.
    pub fn parse_local(&self, s: &str, allow_zone: bool) -> Result<Option<i64>, KrishivError> {
        if self.fast {
            return Ok(self
                .fast_fields(s, true)
                .filter(|(_, zone)| allow_zone || zone.is_none())
                .and_then(|(local, _)| local.and_then(|l| l.micros())));
        }
        Ok(self
            .parse_fields(s, None)
            .filter(|(_, zone)| allow_zone || zone.is_none())
            .and_then(|(local, _)| local))
    }

    /// `s` as a date (days since the epoch): time and zone are ignored.
    ///
    /// # Errors
    /// Never.
    pub fn parse_date(&self, s: &str) -> Result<Option<i32>, KrishivError> {
        if self.fast {
            return Ok(self.fast_date(s));
        }
        Ok(self
            .parse_fields(s, None)
            .and_then(|(local, _)| local)
            .and_then(|local| i32::try_from(local.div_euclid(MICROS_PER_DAY)).ok()))
    }

    /// The fields of `s`; zone names are looked up in `session`'s own row
    /// first (`None` where the zone is read and ignored).
    fn parse_fields(&self, s: &str, session: Option<&str>) -> Option<Resolved> {
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
                    let start = pos;
                    while text.get(pos).is_some_and(|x| matches!(x, ' ' | '\t')) {
                        pos += 1;
                    }
                    // A number directly followed by another number in the
                    // pattern reads exactly its run length, counted from the
                    // field's start: skipped blanks use it up (`subParse`
                    // parses up to `start + count`; Spark 4.2: '2024-01-01  123'
                    // with 'yyyy-MM-dd HHmm' is 01:23).
                    let abut =
                        matches!(self.toks.get(i + 1), Some(Tok::Field(d, dn)) if is_numeric(*d, *dn))
                            .then_some(n.saturating_sub(pos - start));
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
fn legacy_local_to_utc(local: i64, tz: &dyn TimeZoneRules) -> Result<i64, Unmatched> {
    let secs = local.div_euclid(MICROS_PER_SECOND);
    let near = [
        tz.offset_at_utc(secs.saturating_sub(86_400)),
        tz.offset_at_utc(secs.saturating_add(86_400)),
    ];
    let offset = near
        .into_iter()
        .filter(|&o| tz.offset_at_utc(secs.saturating_sub(i64::from(o))) == o)
        .min()
        .ok_or(Unmatched::Text)?;
    local
        .checked_sub(i64::from(offset) * MICROS_PER_SECOND)
        .ok_or(Unmatched::Overflow)
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

#[derive(Debug)]
/// Why a LEGACY parse gave no instant.
pub(crate) enum Unmatched {
    /// The text does not match the pattern, or names a date or zone the
    /// calendar refuses.
    Text,
    /// The date is past the microsecond range.
    Overflow,
}

/// The resolved local micros (`None` past the microsecond range) and the
/// zone the text named.
type Resolved = (Option<i64>, Option<Zone_>);

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
    /// A one-based hour (`k`, `h`) read outside its range (1-24, 1-12).
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
        // `DecimalFormat` reads a minus sign, which counts toward a width.
        // Every field's minimum is at least 0 for a non-lenient calendar but
        // the week year's, which `setWeekDate` takes as it is (Spark 4.2:
        // '-5' with 'YYYY' is -0006-12-27, with 'yyyy' NULL).
        let sign = usize::from(text.get(pos) == Some(&'-'));
        let (value, end) = number(text, pos + sign, abut.map(|l| l.saturating_sub(sign)))?;
        let value = if sign == 1 { -value } else { value };
        if value < 0 && c != 'Y' {
            return None;
        }
        let digits = end - pos - sign;
        match c {
            'y' | 'Y' => {
                // A two-digit year waits for the rest of the date: the
                // window starts on a day, not at a year.
                self.year_two = n <= 2 && digits == 2 && sign == 0;
                self.year = Some((value, st));
                self.year_is_week = c == 'Y';
            }
            'M' | 'L' => self.month0 = Some((value - 1, st)),
            'd' => self.dom = Some((value, st)),
            'D' => self.doy = Some((value, st)),
            'H' => self.hour_of_day = Some((value, st)),
            // 24 reads as 0 (no carry), as `SimpleDateFormat` does.
            // `SimpleDateFormat` refuses an out-of-range one-based hour as
            // it reads it, so a later `H` or `K` cannot overwrite it.
            'k' => {
                self.bad_hour |= !(1..=24).contains(&value);
                self.hour_of_day = Some((if value == 24 { 0 } else { value }, st));
            }
            'h' => {
                self.bad_hour |= !(1..=12).contains(&value);
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

    /// The real epoch day a week-field group names
    /// (`GregorianCalendar.computeTime`, `Locale.US`: weeks start on Sunday,
    /// a week 1 may hold one day).
    fn week_real(&self, year: i64, mode: Mode) -> Option<i64> {
        // A lenient `GregorianCalendar` resolves week fields on the Gregorian
        // side first and falls back to the Julian anchors only when that
        // lands before the cutover.
        // The week of the year counts from the year's first day, which is
        // Julian in the cutover year (as the day of the year does).
        if matches!(mode, Mode::Woy) {
            return self.week_real_on(year, mode, false);
        }
        match self.week_real_on(year, mode, true)? {
            real if real >= CUTOVER_DAYS => Some(real),
            _ => self.week_real_on(year, mode, false),
        }
    }

    /// [`Self::week_real`] with the month and year anchors on the Gregorian
    /// calendar (`gregorian`) or the Julian one before the cutover.
    fn week_real_on(&self, year: i64, mode: Mode, gregorian: bool) -> Option<i64> {
        let on_or_before = |d: i64, dow: i64| d - (dow_sun1(d) - dow).rem_euclid(7);
        // Real epoch days: the Julian calendar's before the cutover.
        // A lenient calendar carries a month past December into the year.
        let month0 = self.month0.map_or(0, |(v, _)| v);
        let year_m = year.checked_add(month0.div_euclid(12))?;
        let month_m = u32::try_from(month0.rem_euclid(12) + 1).ok()?;
        let first_of = |month: u32, year: i64| {
            days_from_civil(year, month, 1).map(|label| {
                if gregorian {
                    label
                } else {
                    real_day(label, year, month, 1)
                }
            })
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
                let first = first_of(month_m, year_m)?;
                Some(week_start(first) + 7 * (self.wom?.0 - 1))
            }
            Mode::Woy => {
                let first = first_of(1, year)?;
                Some(week_start(first) + 7 * (self.woy.map_or(1, |w| w.0) - 1))
            }
            _ => {
                let (m, year) = (month_m, year_m);
                let first = first_of(m, year)?;
                let dow = self.dow.map_or(1, |(d, _)| d);
                let n = self.dowim.map_or(1, |(v, _)| v);
                if n >= 0 {
                    Some(on_or_before(first + 7 * n - 1, dow))
                } else {
                    let len = if gregorian {
                        i64::from(days_in_month(year, m))
                    } else {
                        hybrid_month_len(year, m)
                    };
                    Some(on_or_before(first + len + 7 * (n + 1) - 1, dow))
                }
            }
        }
    }

    /// A non-lenient `Calendar` rejects a result that disagrees with a
    /// field that was set. `days` is the result's label and `(y, m, d)` the
    /// hybrid-calendar date it names (`1500-02-29` for a Julian-only leap
    /// day, whose label is March 1). `year` is the astronomical year, or for
    /// a winning week year the week year as parsed: `setWeekDate` compares
    /// `getWeekYear` (astronomical) with it, so a BC era beside a positive
    /// week year (`2020 BC` with `YYYY G`) is refused.
    fn week_fields_agree(&self, days: i64, year: i64, (y, m, d): (i64, u32, u32)) -> bool {
        let (_, hdoy) = hybrid_week_fields(days, y, m, d);
        let real = real_day(days, y, m, d);
        // A year is checked only when the text set one (Spark 4.2: '1' with
        // 'w' is 1969-12-28, the start of 1970's week 1).
        let year_ok = if self.year.is_none() {
            true
        } else if self.year_is_week {
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

    fn resolve(self, today: (i64, u32, u32)) -> Option<Resolved> {
        if !self.year_two {
            // The year of the era is positive: no year 0 (nor `0000 BC`). A
            // week year is no year of the era: `setWeekDate` takes `0` or
            // `-5` as it is (Spark 4.2: '0' with 'YYYY' is -0001-12-28).
            if self.year.is_some_and(|(y, _)| y <= 0) && !(self.year_is_week && !self.era_bc) {
                return None;
            }
            return self.resolve_in(self.year.map_or(1970, |(y, _)| y), self.era_bc);
        }
        // `yy` (`SimpleDateFormat.subParse`): the century of the window
        // start, one later for digits below the start year's; only the start
        // year's own digits are decided by the date, one century later when
        // it falls before the window's first day. A date that fails its
        // fields fails, whatever the century (Spark 4.2: '99-01-01 Thu' with
        // 'yy-MM-dd E' is NULL). The century is chosen on the year of the
        // era, before a BC era turns the year around (`99 BC` is 1999 BC).
        let two = self.year.map_or(0, |(y, _)| y);
        let (ny, nm, nd) = today;
        let start = ny - 80;
        let ambiguous = start.rem_euclid(100);
        let base = start - ambiguous + two + if two >= ambiguous { 0 } else { 100 };
        if two != ambiguous {
            return self.resolve_in(base, self.era_bc);
        }
        let start_days = days_from_civil(start, nm, nd.min(days_in_month(start, nm)))?;
        let chosen = match self.resolve_in(base, false)? {
            (Some(first), _) if first.div_euclid(MICROS_PER_DAY) < start_days => base + 100,
            _ => base,
        };
        self.resolve_in(chosen, self.era_bc)
    }

    fn resolve_in(&self, year: i64, bc: bool) -> Option<Resolved> {
        if self.bad_hour {
            return None;
        }
        // `setWeekDate` checks the result's week year against the week year
        // as parsed, before the era turns it around.
        let parsed = year;
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
        } else if self
            .doy
            .is_some_and(|(_, s)| self.dom.is_none_or(|(_, m)| s > m))
        {
            // `Calendar.selectFields` weighs a day of year against the day of
            // month alone; a month is then only checked against the result.
            Mode::Doy
        } else {
            Mode::Dom
        };
        let days = if matches!(mode, Mode::Wom | Mode::Dowim | Mode::Woy) {
            let real = self.week_real(year, mode)?;
            let days = label_of_real(real);
            // A Julian-only leap day (1500-02-29) has March 1's label, whose
            // real day is the next one.
            let (y, m, d) = civil_from_days(days);
            // A non-lenient `GregorianCalendar` refuses every Gregorian day of
            // the cutover month in a week-of-month or day-of-week-in-month
            // resolution (Spark 4.2: '1582 10 2 1' with 'yyyy MM W u' is NULL,
            // the same with 'F'; the day-of-month path accepts 1582-10-18).
            if matches!(mode, Mode::Wom | Mode::Dowim) && (y, m) == (1582, 10) && d >= 15 {
                return None;
            }
            julian_leap = real_day(days, y, m, d) != real;
            days
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
            (civil_from_days(days).0, 2, 29)
        } else {
            civil_from_days(days)
        };
        if (weeky || self.year_is_week || self.dow.is_some() || self.doy.is_some())
            && !self.week_fields_agree(days, if self.year_is_week { parsed } else { year }, label)
        {
            return None;
        }
        // `GregorianCalendar.computeTime`: the 12-hour group counts as set
        // only with both its fields (`aggregateStamp`), so `hh` without `a`
        // does not displace `HH` (Spark 4.2: '13 01' with 'HH hh' is 13:00);
        // with neither group complete, the later of `h` and `a` decides.
        let hod = self.hour_of_day.map(|(_, s)| s);
        let mut best = hod.max(self.hour12.zip(self.pm).map(|((_, a), (_, b))| a.max(b)));
        if best.is_none() {
            best = self.hour12.map(|(_, s)| s).max(self.pm.map(|(_, s)| s));
        }
        let pm = if self.pm.is_some_and(|(pm, _)| pm) {
            12
        } else {
            0
        };
        let hour = match (best, self.hour_of_day, self.hour12) {
            (None, ..) => 0,
            (b, Some((h, _)), _) if b == hod => {
                if !(0..24).contains(&h) {
                    return None;
                }
                h
            }
            (_, _, Some((h, _))) => {
                if !(0..12).contains(&h) {
                    return None;
                }
                h + pm
            }
            _ => pm,
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
        let local = days
            .checked_mul(MICROS_PER_DAY)
            .and_then(|micros| micros.checked_add(tod));
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

/// `RebaseDateTime.rebaseJulianToGregorianDays`: the day count a hybrid
/// calendar date has, as the day count of the same year-month-day label in
/// the proleptic Gregorian calendar.
#[must_use]
pub fn rebase_julian_days(days: i64) -> i64 {
    label_of_real(days)
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
    let (mut value, mut end) = number_of(text, pos, limit, java_digit)?;
    // `DecimalFormat` also reads an exponent: `E`, an optional `-` and
    // digits, all inside the field's width (Spark 4.2: '2E3' with 'yyyy' is
    // 2000, '25E-1' with 'dd' is 2, '1E+1' is 1). The result is truncated.
    if text.get(end) == Some(&'E') && limit.is_none_or(|l| end + 1 - pos < l) {
        let neg = text.get(end + 1) == Some(&'-');
        let start = end + 1 + usize::from(neg);
        let room = limit.map(|l| l.saturating_sub(start - pos));
        if room != Some(0)
            && let Some((exp, e2)) = number_of(text, start, room, java_digit)
        {
            {
                let scaled = if neg {
                    u32::try_from(exp)
                        .ok()
                        .and_then(|x| 10_i64.checked_pow(x))
                        .map_or(0, |p| value / p)
                } else {
                    u32::try_from(exp)
                        .ok()
                        .and_then(|x| 10_i64.checked_pow(x))
                        .and_then(|p| value.checked_mul(p))?
                };
                value = scaled;
                end = e2;
            }
        }
    }
    Some((wrap_int(value), end))
}

/// `Number.intValue()` of a `Long`: the low 32 bits as a signed `int`
/// (Spark 4.2: '4294969320-01-01' with 'yyyy-MM-dd' is 2024-01-01).
fn wrap_int(value: i64) -> i64 {
    let low = u32::try_from(value & 0xFFFF_FFFF).unwrap_or(0);
    i64::from(i32::from_le_bytes(low.to_le_bytes()))
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

/// Case-insensitive prefix match of any of `options`, as Java's
/// `regionMatches(true, ..)` (Spark 4.2: '2024-APRİL-01' with 'yyyy-MMMM-dd'
/// is April 1).
fn prefix(text: &[char], pos: usize, options: &[&str]) -> Option<usize> {
    options.iter().find_map(|o| {
        let len = o.chars().count();
        let slice = text.get(pos..pos + len)?;
        slice
            .iter()
            .zip(o.chars())
            .all(|(&a, b)| java_same_ignoring_case(a, b))
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
    // Java's simple mapping: the only character whose full lowercase is
    // longer, U+0130 (dotted capital I), lowers to `i`.
    let lower = |c: char| {
        let mut l = c.to_lowercase();
        match (l.next(), l.next()) {
            (Some(x), None) => x,
            _ if c == '\u{130}' => 'i',
            _ => c,
        }
    };
    a == b || upper(a) == upper(b) || lower(upper(a)) == lower(upper(b))
}

#[cfg(test)]
mod tests;
