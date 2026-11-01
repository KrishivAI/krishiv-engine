// SPDX-License-Identifier: Apache-2.0
//! Printing a local date-time with a compiled pattern.

use std::fmt::Write as _;

use krishiv_common::error::KrishivError;

use super::zone_names;
use super::{Field, Item, Number, OffsetStyle, Part, Sign, Text, unsupported};
use crate::cast::datetime::{civil_from_days, days_from_civil};
use crate::datetime::zone::ZoneRef;

pub(super) const MICROS_PER_DAY: i64 = 86_400_000_000;

pub(super) const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
const DAYS: [&str; 7] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
];
const QUARTERS: [&str; 4] = ["1st quarter", "2nd quarter", "3rd quarter", "4th quarter"];

/// Why a section did not print: the value lacks a field (an optional
/// section skips), or a real error (always propagated).
enum Fail {
    Absent(&'static str),
    Err(KrishivError),
}

impl From<KrishivError> for Fail {
    fn from(e: KrishivError) -> Self {
        Self::Err(e)
    }
}

impl From<Fail> for KrishivError {
    fn from(f: Fail) -> Self {
        match f {
            Fail::Absent(what) => unsupported(what),
            Fail::Err(e) => e,
        }
    }
}

const NO_TIME: &str = "a time-of-day field in a pattern that formats a DATE";
const NO_ZONE: &str = "a zone or offset field in a pattern that formats a value without a zone";

/// The fields of one local date-time.
struct Fields {
    year: i64,
    month: u32,
    day: u32,
    day_of_year: i64,
    /// 1 = Monday.
    day_of_week: i64,
    hour: i64,
    minute: i64,
    second: i64,
    nanos: i64,
    has_time: bool,
    /// Seconds of the local date-time since the epoch.
    local_seconds: i64,
    /// Offset seconds and zone id, for an instant.
    zone: Option<(i32, String)>,
}

impl Fields {
    fn new(local: i64, zone: Option<(i32, &str)>, has_time: bool) -> Self {
        Self::from_days_and_time(
            local.div_euclid(MICROS_PER_DAY),
            local.rem_euclid(MICROS_PER_DAY),
            zone,
            has_time,
        )
    }

    fn from_days_and_time(days: i64, tod: i64, zone: Option<(i32, &str)>, has_time: bool) -> Self {
        let (year, month, day) = civil_from_days(days);
        let jan1 = days_from_civil(year, 1, 1).unwrap_or(days);
        Self {
            year,
            month,
            day,
            day_of_year: days - jan1 + 1,
            day_of_week: (days + 3).rem_euclid(7) + 1,
            hour: tod / 3_600_000_000,
            minute: tod / 60_000_000 % 60,
            second: tod / 1_000_000 % 60,
            nanos: tod % 1_000_000 * 1000,
            has_time,
            local_seconds: days.saturating_mul(86_400).saturating_add(tod / 1_000_000),
            zone: zone.map(|(o, id)| (o, id.to_owned())),
        }
    }

    fn number(&self, f: Field) -> Result<i64, Fail> {
        let time = |v: i64| {
            if self.has_time {
                Ok(v)
            } else {
                Err(Fail::Absent(NO_TIME))
            }
        };
        Ok(match f {
            Field::Year => self.year,
            Field::YearOfEra => {
                if self.year >= 1 {
                    self.year
                } else {
                    1 - self.year
                }
            }
            Field::Month => i64::from(self.month),
            Field::Day => i64::from(self.day),
            Field::DayOfYear => self.day_of_year,
            Field::Quarter => (i64::from(self.month) - 1) / 3 + 1,
            Field::AlignedWeekOfMonth => (i64::from(self.day) - 1) / 7 + 1,
            Field::HourOfDay => time(self.hour)?,
            Field::ClockHourOfDay => time(if self.hour == 0 { 24 } else { self.hour })?,
            Field::HourOfAmPm => time(self.hour % 12)?,
            Field::ClockHourOfAmPm => time(if self.hour % 12 == 0 {
                12
            } else {
                self.hour % 12
            })?,
            Field::Minute => time(self.minute)?,
            Field::Second => time(self.second)?,
        })
    }

    fn zone(&self) -> Result<&(i32, String), Fail> {
        self.zone.as_ref().ok_or(Fail::Absent(NO_ZONE))
    }
}

/// Prints `local` with `items`; `zone` is the instant's offset and zone id
/// (`None` for dates and local date-times); `has_time` is false for dates.
pub(super) fn format(
    items: &[Item],
    local: i64,
    zone: Option<(i32, &str)>,
    has_time: bool,
) -> Result<String, KrishivError> {
    let f = Fields::new(local, zone, has_time);
    let mut out = String::new();
    write_items(items, &f, &mut out)?;
    Ok(out)
}

/// Prints a `DATE` (days since the epoch) without going through
/// microseconds, so every `i32` day formats (the microsecond form
/// overflows `i64` past year 294247).
pub(super) fn format_days(items: &[Item], days: i64) -> Result<String, KrishivError> {
    let f = Fields::from_days_and_time(days, 0, None, false);
    let mut out = String::new();
    write_items(items, &f, &mut out)?;
    Ok(out)
}

fn write_items(items: &[Item], f: &Fields, out: &mut String) -> Result<(), Fail> {
    for item in items {
        match item {
            Item::Literal(s) => out.push_str(s),
            Item::Number(n) => write_number(n, f.number(n.field)?, out),
            Item::Reduced(field) => {
                let v = f.number(*field)?;
                let _ = write!(out, "{:02}", (v % 100).unsigned_abs());
            }
            Item::Fraction { max, .. } => {
                if !f.has_time {
                    return Err(Fail::Absent(NO_TIME));
                }
                let digits = format!("{:09}", f.nanos);
                out.push_str(&digits[..(*max).min(9)]);
            }
            Item::Text { text, full } => write_text(*text, *full, f, out)?,
            Item::Offset(style) => {
                let (offset, _) = f.zone()?;
                write_offset(style, *offset, out);
            }
            Item::Localized { full } => {
                let (offset, _) = f.zone()?;
                write_localized(*full, *offset, out);
            }
            Item::ZoneId => {
                let (_, id) = f.zone()?;
                out.push_str(id);
            }
            Item::ZoneName { full } => {
                let (offset, id) = f.zone()?;
                out.push_str(&zone_name(id, *offset, f.local_seconds, *full)?);
            }
            // A section prints when the value has all its fields (a DATE
            // has no time of day), as java.time's optional sections do;
            // any other error (a region zone name) still fails.
            Item::Optional(inner) => {
                let mut section = String::new();
                match write_items(inner, f, &mut section) {
                    Ok(()) => out.push_str(&section),
                    Err(Fail::Absent(_)) => {}
                    Err(e) => return Err(e),
                }
            }
        }
    }
    Ok(())
}

/// Every field value fits its pattern's maximum width (two-digit fields
/// hold at most 59, `DD` at most 366, years 19 digits), so no width check
/// is needed.
fn write_number(n: &Number, v: i64, out: &mut String) {
    let digits = v.unsigned_abs().to_string();
    if v < 0 {
        out.push('-');
    } else if n.sign == Sign::ExceedsPad && digits.len() > n.min {
        out.push('+');
    }
    for _ in digits.len()..n.min {
        out.push('0');
    }
    out.push_str(&digits);
}

fn write_text(text: Text, full: bool, f: &Fields, out: &mut String) -> Result<(), Fail> {
    let idx = |v: i64| usize::try_from(v - 1).unwrap_or(0);
    match text {
        Text::Era => out.push_str(match (f.year >= 1, full) {
            (true, false) => "AD",
            (true, true) => "Anno Domini",
            (false, false) => "BC",
            (false, true) => "Before Christ",
        }),
        Text::Month => {
            let m = MONTHS[idx(i64::from(f.month))];
            out.push_str(if full { m } else { &m[..3] });
        }
        Text::DayOfWeek => {
            let d = DAYS[idx(f.day_of_week)];
            out.push_str(if full { d } else { &d[..3] });
        }
        Text::Quarter => {
            let q = f.number(Field::Quarter)?;
            if full {
                out.push_str(QUARTERS[idx(q)]);
            } else {
                let _ = write!(out, "Q{q}");
            }
        }
        Text::AmPm => {
            if !f.has_time {
                return Err(Fail::Absent(NO_TIME));
            }
            out.push_str(if f.hour < 12 { "AM" } else { "PM" });
        }
    }
    Ok(())
}

fn write_offset(style: &OffsetStyle, offset: i32, out: &mut String) {
    if offset == 0 {
        out.push_str(style.zero);
        return;
    }
    let abs = offset.unsigned_abs();
    let (h, m, s) = (abs / 3600, abs / 60 % 60, abs % 60);
    out.push(if offset < 0 { '-' } else { '+' });
    let _ = write!(out, "{h:02}");
    let sep = if style.colon { ":" } else { "" };
    let minutes =
        style.minutes == Part::Required || (style.minutes == Part::Optional && (m != 0 || s != 0));
    if minutes {
        let _ = write!(out, "{sep}{m:02}");
        let seconds =
            style.seconds == Part::Required || (style.seconds == Part::Optional && s != 0);
        if seconds {
            let _ = write!(out, "{sep}{s:02}");
        }
    }
}

fn write_localized(full: bool, offset: i32, out: &mut String) {
    out.push_str("GMT");
    if offset == 0 {
        return;
    }
    let abs = offset.unsigned_abs();
    let (h, m, s) = (abs / 3600, abs / 60 % 60, abs % 60);
    out.push(if offset < 0 { '-' } else { '+' });
    if full {
        let _ = write!(out, "{h:02}:{m:02}");
        if s != 0 {
            let _ = write!(out, ":{s:02}");
        }
    } else {
        let _ = write!(out, "{h}");
        if m != 0 || s != 0 {
            let _ = write!(out, ":{m:02}");
            if s != 0 {
                let _ = write!(out, ":{s:02}");
            }
        }
    }
}

/// `z` / `zzzz` for the zones known without the time-zone database.
fn zone_name(
    id: &str,
    offset: i32,
    local_seconds: i64,
    full: bool,
) -> Result<String, KrishivError> {
    Ok(match id {
        "UTC" | "Etc/UTC" => if full {
            "Coordinated Universal Time"
        } else {
            "UTC"
        }
        .to_owned(),
        "GMT" | "Etc/GMT" => if full { "Greenwich Mean Time" } else { "GMT" }.to_owned(),
        "Z" | "UT" => id.to_owned(),
        // `UTC+05:30`, `GMT+05:30` and `UT+05:30` have no display name:
        // `java.time` falls back to the id.
        _ if ["UTC", "GMT", "UT"]
            .iter()
            .filter_map(|p| id.strip_prefix(p))
            .any(|rest| rest.starts_with(['+', '-'])) =>
        {
            id.to_owned()
        }
        _ if id.starts_with('+') || id.starts_with('-') => {
            let mut s = String::new();
            write_offset(
                &OffsetStyle {
                    colon: true,
                    minutes: Part::Required,
                    seconds: Part::Optional,
                    zero: "Z",
                },
                offset,
                &mut s,
            );
            s
        }
        _ => return region_zone_name(id, local_seconds.saturating_sub(i64::from(offset)), full),
    })
}

/// 1990-03-20T22:00Z, where Java's tables move Namibia's standard time to +01:00 (so +02:00 is
/// daylight time from then on).
const WINDHOEK_JAVA_DST_FROM: i64 = 637_970_400;

/// The name of a region zone at the instant `utc_seconds`: daylight names
/// while the zone is on daylight time there.
fn region_zone_name(id: &str, utc_seconds: i64, full: bool) -> Result<String, KrishivError> {
    let unknown = || unsupported(&format!("the name of the time zone {id}"));
    let entry = zone_names::ZONES
        .binary_search_by_key(&id, |(zone, _)| zone)
        .ok()
        .and_then(|i| zone_names::ZONES.get(i))
        .and_then(|(_, group)| zone_names::NAMES.get(usize::from(*group)))
        .ok_or_else(unknown)?;
    // A fixed-offset spelling (`Etc/GMT+0`) never has daylight time.
    let daylight = match crate::datetime::zone::resolve_zone(id).map_err(|_| unknown())? {
        ZoneRef::Named(zone) => {
            // Past Jiff's range the recurring rules repeat, as `offset_at_utc` folds them.
            jiff::Timestamp::from_second(crate::datetime::zone::fold_seconds(utc_seconds))
                .is_ok_and(|t| {
                    let info = zone.to_offset_info(t);
                    if id == "Africa/Windhoek" && utc_seconds >= WINDHOEK_JAVA_DST_FROM {
                        // Java's data keeps +01:00 as Namibia's standard time and calls +02:00 daylight
                        // time from 1990 on (also since 2017, when the winter shift stopped); the tz
                        // database's main form says the reverse before 2017 and has no daylight time after.
                        info.offset().seconds() == 7200
                    } else {
                        info.dst().is_dst()
                    }
                })
        }
        ZoneRef::Fixed(_) => false,
    };
    let at = usize::from(daylight) * 2 + usize::from(full);
    entry
        .get(at)
        .map(|name| (*name).to_owned())
        .ok_or_else(unknown)
}
