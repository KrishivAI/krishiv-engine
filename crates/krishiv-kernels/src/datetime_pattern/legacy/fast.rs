// SPDX-License-Identifier: Apache-2.0
//! Spark's `FastDateFormat` pair under `timeParserPolicy=LEGACY`: the
//! JSON, CSV and XML options read and write `DATE` and `TIMESTAMP` values
//! with `LegacyFastTimestampFormatter` / `LegacyFastDateFormatter`, which are
//! commons-lang 3.20's `FastDateParser` and `FastDatePrinter` over a
//! `GregorianCalendar` (a `MicrosCalendar` for timestamps).
//!
//! Against the SQL functions' `SimpleDateFormat` model (the rest of
//! [`super`]):
//! - the calendar is lenient: `2024-02-30` is March 1, month 13 is January of
//!   the next year, `25:61:61` rolls over;
//! - a number is read as `FastDateParser` does: a run followed by another
//!   number in the pattern takes exactly its letter count of digits, any
//!   other skips Java whitespace and takes every digit; there are no signs;
//!   a value past `int` is no match;
//! - a year of one or two letters, or of one or two digits, is mapped into
//!   the 80 years before to the 20 after the current year only by its
//!   *letters* (`y`, `yy`; `Y` is the same, not a week year);
//! - `S` runs are one field, scaled by the pattern's total count of `S` to
//!   a fraction of a second on both parse and print (`MicrosCalendar`);
//! - text fields match case-blind against every long and short name, the
//!   longest first; `a` and `G` also take the one-letter forms;
//! - printing pads `y`/`Y` to four digits, prints `Z` as `+hhmm`, `ZZ` as
//!   `Z`/`+hh:mm` and `ZZZ` as `+hh:mm`, and prints a negative week year
//!   as commons-lang's `appendFullDigits` does (`000.` for -2).
//!
//! Zone names are read from the JVM zone strings that the `SimpleDateFormat`
//! model also uses (first registered name wins, standard offset plus the
//! saving for a daylight name); Spark's `FastDateParser` additionally knows
//! every `TimeZone.getDisplayName`, which is not modelled. A `DATE` is read
//! as the instant's day in UTC (the JVM default zone is the host's, not the
//! query's; the probes ran in UTC).

use super::super::unsupported;
use super::{
    LONG_DAYS, LegacyPattern, MICROS_PER_DAY, MICROS_PER_SECOND, MONTHS, Mode, Parsed, SHORT_DAYS,
    Tok, Unmatched, ZONE_STRINGS, Zone_, days_from_civil, java_digit, label_of_real, name_at,
    prefix, real_day, zone, zone_iso,
};
use crate::cast::{FixedOffset, TimeZoneRules};
use krishiv_common::error::KrishivError;

/// A field as `FastDatePrinter` writes a year (`y` is the year of the era,
/// `Y` the astronomical week year): four digits at least, and a negative
/// value through commons-lang's digit loop.
pub(super) fn year_field(out: &mut String, wy: i64, n: usize) {
    if n == 2 {
        let r = wy % 100;
        out.push(java_char('0', r / 10));
        out.push(java_char('0', r % 10));
    } else if wy >= 0 {
        super::pad(out, wy, n.max(4));
    } else {
        // `appendFullDigits` of a value under 10: zeros, then one char of
        // `value + '0'` (a negative value gives a symbol).
        for _ in 0..n.max(4).saturating_sub(1) {
            out.push('0');
        }
        out.push(java_char('0', wy));
    }
}

/// `(char) (base + delta)`: a Java `char` wraps at 16 bits.
fn java_char(base: char, delta: i64) -> char {
    let v = (i64::from(u32::from(base)) + delta).rem_euclid(65_536);
    u32::try_from(v)
        .ok()
        .and_then(char::from_u32)
        .unwrap_or('\u{fffd}')
}

/// Java whitespace (`Character.isWhitespace`).
fn java_whitespace(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\u{b}' | '\u{c}' | '\r' | '\u{1c}'..='\u{1f}'
    ) || (c.is_whitespace() && !matches!(c, '\u{a0}' | '\u{2007}' | '\u{202f}' | '\u{85}'))
}

impl LegacyPattern {
    /// The `MILLISECOND` field a `MicrosCalendar` holds for `micros` of the
    /// second: scaled by the pattern's whole count of `S` letters.
    ///
    /// # Errors
    /// More than 18 `S` letters (`Decimal.POW_10` has no such power).
    pub(super) fn fraction(&self, micros: i64) -> Result<i64, KrishivError> {
        let digits = self.s_count();
        if digits > 18 {
            return Err(unsupported("a fraction pattern of more than 18 S letters"));
        }
        // Scala: `(micros * POW_10(d) / 1000000).toInt` in `long` arithmetic.
        let scaled = micros.wrapping_mul(10_i64.pow(digits)) / MICROS_PER_SECOND;
        let low = u32::try_from(scaled & 0xFFFF_FFFF).unwrap_or(0);
        Ok(i64::from(i32::from_le_bytes(low.to_le_bytes())))
    }

    fn s_count(&self) -> u32 {
        // `fastDateFormat.getPattern.count(_ == 'S')`: quotes do not matter.
        let n = self.pattern.chars().filter(|&c| c == 'S').count();
        u32::try_from(n.min(30)).unwrap_or(30)
    }

    /// `s` as an instant (`LegacyFastTimestampFormatter.parse`).
    pub(super) fn fast_instant(&self, s: &str, session: &dyn TimeZoneRules) -> Option<i64> {
        let (local, zone) = self.fast_fields(s, true)?;
        zone_instant(local?, zone, session).ok()
    }

    /// `s` as a day (`LegacyFastDateFormatter.parse`): the day of the instant
    /// in UTC.
    pub(super) fn fast_date(&self, s: &str) -> Option<i32> {
        let (local, zone) = self.fast_fields(s, false)?;
        let instant = zone_instant(local?, zone, &FixedOffset(0)).ok()?;
        i32::try_from(instant.div_euclid(MICROS_PER_DAY)).ok()
    }

    /// The local label time and the zone `s` names; `scaled` reads `S` as a
    /// fraction of a second (timestamps) rather than milliseconds (dates).
    pub(super) fn fast_fields(
        &self,
        s: &str,
        scaled: bool,
    ) -> Option<(Option<i64>, Option<Zone_>)> {
        if self.broken {
            return None;
        }
        let text: Vec<char> = s.chars().collect();
        let mut st = Parsed::default();
        let mut pos = 0;
        for (i, t) in self.toks.iter().enumerate() {
            match t {
                Tok::Literal(lit) => {
                    for c in lit.chars() {
                        if text.get(pos) != Some(&c) {
                            return None;
                        }
                        pos += 1;
                    }
                }
                Tok::Field(c, n) => {
                    let width = match self.toks.get(i + 1) {
                        Some(Tok::Field(d, dn)) if is_number(*c, *n) && is_number(*d, *dn) => *n,
                        _ => 0,
                    };
                    pos = self.fast_field(&mut st, &text, pos, *c, *n, width)?;
                }
            }
        }
        resolve(&st, scaled.then(|| self.s_count()), self.today.0)
    }

    fn fast_field(
        &self,
        st: &mut Parsed,
        text: &[char],
        pos: usize,
        c: char,
        n: usize,
        width: usize,
    ) -> Option<usize> {
        let stamp = st.stamp();
        match c {
            'G' => {
                let (end, bc) = names(
                    text,
                    pos,
                    &[
                        ("ad", false),
                        ("anno domini", false),
                        ("a", false),
                        ("bc", true),
                        ("before christ", true),
                        ("b", true),
                    ],
                )?;
                st.era_bc = bc;
                return Some(end);
            }
            'a' => {
                let (end, pm) = names(
                    text,
                    pos,
                    &[("am", false), ("pm", true), ("a", false), ("p", true)],
                )?;
                st.pm = Some((pm, stamp));
                return Some(end);
            }
            'E' => {
                let mut list = Vec::new();
                for (i, d) in LONG_DAYS.iter().enumerate() {
                    list.push((d.to_lowercase(), i));
                }
                for (i, d) in SHORT_DAYS.iter().enumerate() {
                    list.push((d.to_lowercase(), i));
                }
                let (end, i) = names_owned(text, pos, list)?;
                st.dow = Some((if i == 6 { 1 } else { i as i64 + 2 }, stamp));
                return Some(end);
            }
            'M' | 'L' if n >= 3 => {
                let mut list = Vec::new();
                for (i, m) in MONTHS.iter().enumerate() {
                    list.push((m.to_lowercase(), i));
                    list.push((m[..3].to_lowercase(), i));
                }
                let (end, i) = names_owned(text, pos, list)?;
                st.month0 = Some((i as i64, stamp));
                return Some(end);
            }
            'X' => {
                let (end, z) = zone_iso(text, pos, n)?;
                st.zone = Some(z);
                return Some(end);
            }
            'Z' if n == 2 => {
                let (end, z) = zone_iso(text, pos, 3)?;
                st.zone = Some(z);
                return Some(end);
            }
            'z' | 'Z' => {
                let (end, z) = fast_zone(text, pos)?;
                st.zone = Some(z);
                return Some(end);
            }
            _ => {}
        }
        // `NumberStrategy.parse`
        let mut idx = pos;
        let mut last = text.len();
        let mut at = pos;
        if width == 0 {
            while text.get(idx).is_some_and(|&x| java_whitespace(x)) {
                idx += 1;
            }
            at = idx;
        } else {
            last = last.min(idx + width);
        }
        let mut end = at;
        let mut value: i64 = 0;
        while end < last {
            let Some(d) = text.get(end).and_then(|&x| java_digit(x)) else {
                break;
            };
            value = value.checked_mul(10)?.checked_add(i64::from(d))?;
            if value > i64::from(i32::MAX) {
                // `Integer.parseInt` throws.
                return None;
            }
            end += 1;
        }
        if end == at {
            return None;
        }
        match c {
            'y' | 'Y' => {
                let v = if n > 2 || value >= 100 {
                    value
                } else {
                    adjust_year(value, self.today.0)
                };
                st.year = Some((v, stamp));
            }
            'M' | 'L' => st.month0 = Some((value - 1, stamp)),
            'd' => st.dom = Some((value, stamp)),
            'D' => st.doy = Some((value, stamp)),
            'H' => st.hour_of_day = Some((value, stamp)),
            'k' => st.hour_of_day = Some((if value == 24 { 0 } else { value }, stamp)),
            'h' => st.hour12 = Some((if value == 12 { 0 } else { value }, stamp)),
            'K' => st.hour12 = Some((value, stamp)),
            'm' => st.minute = Some(value),
            's' => st.second = Some(value),
            'S' => st.millis = Some(value),
            'w' => st.woy = Some((value, stamp)),
            'W' => st.wom = Some((value, stamp)),
            'F' => st.dowim = Some((value, stamp)),
            'u' => st.dow = Some((if value == 7 { 1 } else { value + 1 }, stamp)),
            _ => {}
        }
        Some(end)
    }
}

/// Whether the letter run is a `NumberStrategy` field.
fn is_number(c: char, n: usize) -> bool {
    match c {
        'D' | 'F' | 'H' | 'K' | 'S' | 'W' | 'd' | 'h' | 'k' | 'm' | 's' | 'u' | 'w' | 'y' | 'Y' => {
            true
        }
        'M' | 'L' => n < 3,
        _ => false,
    }
}

/// `FastDateParser.adjustYear`: the century of the year 80 before now.
fn adjust_year(two: i64, this_year: i64) -> i64 {
    let start = this_year - 80;
    let century = start / 100 * 100;
    let from = start - century;
    if two >= from {
        century + two
    } else {
        century + two + 100
    }
}

/// The first of `options` (longest first, as the reverse-sorted regex tries
/// them) that starts `text` at `pos`, ignoring case.
fn names(text: &[char], pos: usize, options: &[(&str, bool)]) -> Option<(usize, bool)> {
    names_owned(
        text,
        pos,
        options.iter().map(|&(n, v)| (n.to_owned(), v)).collect(),
    )
}

fn names_owned<V: Copy>(
    text: &[char],
    pos: usize,
    mut options: Vec<(String, V)>,
) -> Option<(usize, V)> {
    options.sort_by(|a, b| b.0.cmp(&a.0));
    options.dedup_by(|a, b| a.0 == b.0);
    options
        .into_iter()
        .find_map(|(name, v)| prefix(text, pos, &[&name]).map(|end| (end, v)))
}

/// `FastDateParser`'s `TimeZoneStrategy`: `+hhmm`, `GMT+h:mm`, or a zone
/// name, which sets the zone's raw offset plus (for a daylight name) its
/// saving.
fn fast_zone(text: &[char], pos: usize) -> Option<(usize, Zone_)> {
    if matches!(text.get(pos), Some('+' | '-')) {
        return zone(text, pos, None);
    }
    if let Some(end) = prefix(text, pos, &["GMT"]) {
        // `GMT+05` is no `GMT+h:mm`: the zone name `GMT` is read alone.
        return zone(text, pos, None).or(Some((end, Zone_::Offset(0))));
    }
    let mut best: Option<(usize, i32)> = None;
    for row in ZONE_STRINGS.iter() {
        for (i, name) in row.names.iter().enumerate() {
            let Some(end) = name_at(text, pos, name) else {
                continue;
            };
            if best.is_none_or(|(len, _)| end > len) {
                let dst = if i >= 2 { row.dst } else { 0 };
                best = Some((end, row.raw + dst));
            }
            break;
        }
    }
    best.map(|(end, offset)| (end, Zone_::Offset(offset)))
}

/// The instant of a local label time in `zone`, or the session's.
fn zone_instant(
    local: i64,
    zone: Option<Zone_>,
    session: &dyn TimeZoneRules,
) -> Result<i64, Unmatched> {
    let shift = |offset: i32| {
        local
            .checked_sub(i64::from(offset) * MICROS_PER_SECOND)
            .ok_or(Unmatched::Overflow)
    };
    match zone {
        None => {
            // A lenient calendar puts a time in an overlap on the later
            // instant (the smaller valid offset) and one in a gap on the
            // offset before the change.
            let secs = local.div_euclid(MICROS_PER_SECOND);
            let before = session.offset_at_utc(secs.saturating_sub(86_400));
            let after = session.offset_at_utc(secs.saturating_add(86_400));
            let valid = |o: i32| session.offset_at_utc(secs.saturating_sub(i64::from(o))) == o;
            let offset = [before, after]
                .into_iter()
                .filter(|&o| valid(o))
                .min()
                .unwrap_or(before);
            shift(offset)
        }
        Some(Zone_::Offset(o)) => shift(o),
        Some(Zone_::Region(id)) => match crate::datetime::resolve_zone(&id) {
            Ok(z) => zone_instant(local, None, &z),
            Err(_) => Err(Unmatched::Text),
        },
        Some(Zone_::Standard { raw, dst, .. }) => shift(raw + dst),
    }
}

/// `GregorianCalendar.computeTime` over the fields a lenient parse set.
fn resolve(
    st: &Parsed,
    fraction_digits: Option<u32>,
    this_year: i64,
) -> Option<(Option<i64>, Option<Zone_>)> {
    let _ = this_year;
    let raw_year = st.year.map_or(1970, |(y, _)| y);
    let year = if st.era_bc { 1 - raw_year } else { raw_year };
    let weeky = st.woy.is_some() || st.wom.is_some() || st.dowim.is_some();
    let mode = if weeky {
        st.week_mode()
    } else if st.dow.is_some() && st.dom.is_none() && st.doy.is_none() {
        Mode::Dowim
    } else if st
        .doy
        .is_some_and(|(_, s)| st.dom.is_none_or(|(_, m)| s > m))
    {
        Mode::Doy
    } else {
        Mode::Dom
    };
    let real = match mode {
        Mode::Wom | Mode::Dowim | Mode::Woy => st.week_real(year, mode)?,
        Mode::Doy => {
            let first = days_from_civil(year, 1, 1)?;
            real_day(first, year, 1, 1).checked_add(st.doy?.0 - 1)?
        }
        Mode::Dom => {
            let month0 = st.month0.map_or(0, |(m, _)| m);
            let y = year.checked_add(month0.div_euclid(12))?;
            let month = u32::try_from(month0.rem_euclid(12) + 1).ok()?;
            let first = days_from_civil(y, month, 1)?;
            real_day(first, y, month, 1).checked_add(st.dom.map_or(1, |(d, _)| d) - 1)?
        }
    };
    // `HOUR_OF_DAY`, or `HOUR` with `AM_PM` when `AM_PM` was set last.
    let hod = st.hour_of_day.map(|(_, s)| s);
    let h12 = st.hour12.map(|(_, s)| s);
    let pm = st.pm.map(|(_, s)| s);
    let mut best = hod.max(h12.zip(pm).map(|(a, b)| a.max(b)));
    if best.is_none() {
        best = h12.max(pm);
    }
    let hour = match best {
        None => 0,
        Some(b) if Some(b) == hod => st.hour_of_day.map_or(0, |(h, _)| h),
        Some(_) => {
            st.hour12.map_or(0, |(h, _)| h) + if st.pm.is_some_and(|(p, _)| p) { 12 } else { 0 }
        }
    };
    let ms = st.millis.unwrap_or(0);
    let fraction = match fraction_digits {
        Some(d) if d > 18 => return None,
        Some(d) => i64::try_from(i128::from(ms) * 1_000_000 / 10_i128.checked_pow(d)?).ok()?,
        None => ms.checked_mul(1000)?,
    };
    let seconds = ((hour.checked_mul(60)?.checked_add(st.minute.unwrap_or(0))?).checked_mul(60)?)
        .checked_add(st.second.unwrap_or(0))?;
    let tod = seconds
        .checked_mul(MICROS_PER_SECOND)?
        .checked_add(fraction)?;
    let total = real.checked_mul(MICROS_PER_DAY)?.checked_add(tod)?;
    let days = label_of_real(total.div_euclid(MICROS_PER_DAY));
    let local = days
        .checked_mul(MICROS_PER_DAY)
        .and_then(|d| d.checked_add(total.rem_euclid(MICROS_PER_DAY)));
    Some((local, st.zone.clone()))
}
