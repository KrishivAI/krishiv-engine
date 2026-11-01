// SPDX-License-Identifier: Apache-2.0
//! Named time zones: [`resolve_zone`] turns a Spark zone id (`UTC`, a zone
//! offset, a short id such as `PST`, or a region id such as
//! `America/Los_Angeles`) into a [`ZoneRef`] that implements
//! [`TimeZoneRules`], so kernels stay pure functions of their inputs plus
//! the pinned bundled database.
//!
//! Resolution mirrors Spark's `DateTimeUtils.getZoneId` (`ZoneId.of` with
//! the short ids, as `krishiv-testkit`'s golden formatter parses zones):
//! fixed offsets resolve without the database, short ids map to their
//! region or offset, and anything else is looked up in the bundled TZDB.
//! Unknown ids are `INVALID_TIMEZONE`, with Spark's message. Results never
//! depend on the host: the system time-zone database is never read.
//!
//! Daylight transitions resolve as `java.time.ZonedDateTime.of` does: an
//! overlap keeps the earlier offset, a gap moves forward by the gap's
//! length (the offset before the transition applies to the gap's local
//! times).
#![expect(
    clippy::arithmetic_side_effects,
    reason = "byte indices stay in bounds (`i + 2` below `len`, the rest via `get`), string \
              slices fall on ASCII bytes, and hour/minute/second totals stay below 400,000"
)]

use jiff::Timestamp;
use jiff::tz::TimeZone;
use krishiv_common::error::{InvalidTimezone, KrishivError};

use crate::cast::{FixedOffset, TimeZoneRules};

/// The id `java.time.ZoneId.of` gives a `spark.sql.session.timeZone` value,
/// which `VV` and `z` patterns print: an offset in full (`+05` is `+05:00`,
/// `GMT+5` is `GMT+05:00`), a zero offset as its prefix or `Z`. Anything
/// [`FixedOffset`] does not read stays as written.
#[must_use]
pub fn zone_id(configured: &str) -> String {
    let zone = configured.trim();
    let Some(FixedOffset(offset)) = FixedOffset::parse(zone) else {
        return zone.to_owned();
    };
    if !zone.contains(['+', '-']) {
        return zone.to_owned();
    }
    let prefix = ["UTC", "GMT", "UT"]
        .into_iter()
        .find(|p| zone.starts_with(p))
        .unwrap_or("");
    if offset == 0 {
        return if prefix.is_empty() { "Z" } else { prefix }.to_owned();
    }
    let (sign, abs) = if offset < 0 {
        ('-', -offset)
    } else {
        ('+', offset)
    };
    format!("{prefix}{sign}{:02}:{:02}", abs / 3600, abs / 60 % 60)
}

/// A resolved zone id: a fixed offset, or a region of the bundled time-zone
/// database.
#[derive(Debug, Clone)]
pub enum ZoneRef {
    /// `UTC`, `+05:30` and the other spellings [`FixedOffset`] reads.
    Fixed(FixedOffset),
    /// A region id (`America/Los_Angeles`), short id (`PST`) or `Etc/GMT`
    /// name the bundled database knows.
    Named(TimeZone),
}

impl TimeZoneRules for ZoneRef {
    fn offset_at_utc(&self, utc_seconds: i64) -> i32 {
        match self {
            Self::Fixed(fixed) => fixed.offset_at_utc(utc_seconds),
            Self::Named(zone) => Timestamp::from_second(fold_seconds(utc_seconds))
                .map(|t| zone.to_offset(t).seconds())
                .unwrap_or(0),
        }
    }

    fn offset_for_local(&self, local_seconds: i64) -> i32 {
        match self {
            Self::Fixed(fixed) => fixed.offset_for_local(local_seconds),
            Self::Named(zone) => {
                // The local date-time read as if it were UTC, resolved the
                // way `java.time.ZonedDateTime.of` does: the earlier offset
                // in an overlap, forward by the gap's length in a gap. The
                // offset to apply is then the difference between the local
                // time and the resulting instant.
                let local = fold_seconds(local_seconds);
                let resolved = Timestamp::from_second(local)
                    .map(|t| t.to_zoned(TimeZone::UTC).datetime())
                    .and_then(|dt| zone.to_ambiguous_zoned(dt).compatible());
                resolved
                    .ok()
                    .and_then(|z| i32::try_from(local.checked_sub(z.timestamp().as_second())?).ok())
                    .unwrap_or(0)
            }
        }
    }
}

/// Seconds within `jiff`'s timestamp range: the bundled database ends at
/// the end of year 9999 while `java.time` extends the last rule
/// indefinitely, so out-of-range lookups fold whole 400-year Gregorian
/// cycles (146097 days: the same dates, weekdays and leapness, which is
/// what the yearly-repeating modern rules read). A transition weekday may
/// still drift across the fold.
pub(crate) fn fold_seconds(seconds: i64) -> i64 {
    const CYCLE: i64 = 146_097 * 86_400;
    let (lo, hi) = (Timestamp::MIN.as_second(), Timestamp::MAX.as_second());
    if seconds > hi {
        let cycles = seconds
            .saturating_sub(hi)
            .div_euclid(CYCLE)
            .saturating_add(1);
        seconds.saturating_sub(cycles.saturating_mul(CYCLE))
    } else if seconds < lo {
        let cycles = lo
            .saturating_sub(seconds)
            .div_euclid(CYCLE)
            .saturating_add(1);
        seconds.saturating_add(cycles.saturating_mul(CYCLE))
    } else {
        seconds
    }
}

/// The zone `text` names: a fixed offset without the database, else the
/// bundled database. Untrimmed text never resolves (Spark's `ZoneId.of`
/// does not trim either).
///
/// # Errors
/// `INVALID_TIMEZONE` for text no zone id can have and for unknown region
/// ids.
pub fn resolve_zone(text: &str) -> Result<ZoneRef, KrishivError> {
    /// Zones resolved on this thread: a batch names few distinct zones,
    /// so each is looked up once rather than once per row.
    const CACHE_CAP: usize = 256;
    thread_local! {
        static CACHE: std::cell::RefCell<std::collections::HashMap<String, ZoneRef>> =
            std::cell::RefCell::new(std::collections::HashMap::new());
    }
    if let Some(hit) = CACHE.with(|c| c.borrow().get(text).cloned()) {
        return Ok(hit);
    }
    let zone = resolve_uncached(text)?;
    CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        if cache.len() >= CACHE_CAP {
            cache.clear();
        }
        cache.insert(text.to_owned(), zone.clone());
    });
    Ok(zone)
}

fn resolve_uncached(text: &str) -> Result<ZoneRef, KrishivError> {
    if text != text.trim() || text.is_empty() {
        return Err(invalid(text));
    }
    if let Some(fixed) = FixedOffset::parse(text) {
        return Ok(ZoneRef::Fixed(fixed));
    }
    // Spark's `getZoneId` pads a single hour or minute digit (`+5:30` is
    // `+05:30`) before anything else.
    let padded = pad_hour(text);
    if let Some(id) = SHORT_IDS
        .iter()
        .find(|(short, _)| *short == padded)
        .map(|(_, id)| *id)
    {
        return Ok(match offset_seconds(id) {
            Some(secs) => ZoneRef::Fixed(FixedOffset(secs)),
            // A short id that names a region resolves like one.
            None => ZoneRef::Named(TimeZone::get(id).map_err(|_| invalid(text))?),
        });
    }
    if matches!(padded.as_str(), "Z" | "UTC" | "GMT" | "UT") {
        return Ok(ZoneRef::Fixed(FixedOffset(0)));
    }
    if let Some(secs) = offset_seconds(&padded) {
        return Ok(ZoneRef::Fixed(FixedOffset(secs)));
    }
    for prefix in ["UTC", "GMT", "UT"] {
        if let Some(secs) = padded.strip_prefix(prefix).and_then(offset_seconds) {
            return Ok(ZoneRef::Fixed(FixedOffset(secs)));
        }
    }
    // `Etc/GMT+5` is a fixed UTC-05:00 (POSIX signs flip, whole hours
    // only); the database answers it too, but only within its year
    // range, so certain zones resolve here. Anything else falls through
    // to the database below.
    if let Some(rest) = padded.strip_prefix("Etc/GMT")
        && !rest.is_empty()
        && let Some(digits) = rest.strip_prefix('+').or_else(|| rest.strip_prefix('-'))
        && (1..=2).contains(&digits.len())
        // Spark rejects a zero-padded hour (`Etc/GMT+05`); the database
        // below does too.
        && !(digits.len() == 2 && digits.starts_with('0'))
        && digits.bytes().all(|b| b.is_ascii_digit())
        && let Ok(hours) = digits.parse::<i32>()
        && hours <= 12
    {
        // The sign flips: `Etc/GMT+5` is five hours *behind* UTC.
        let secs = if rest.as_bytes()[0] == b'+' {
            -hours * 3600
        } else {
            hours * 3600
        };
        return Ok(ZoneRef::Fixed(FixedOffset(secs)));
    }
    TimeZone::get(&padded)
        .map_err(|_| invalid(text))
        .and_then(|zone| {
            // `TimeZone::get` matches case-insensitively; Spark's
            // `ZoneId.of` does not, so only the database's own spelling
            // resolves (`etc/utc` is not a zone).
            if zone.iana_name().is_some_and(|name| name == padded) {
                Ok(ZoneRef::Named(zone))
            } else {
                Err(invalid(text))
            }
        })
}

fn invalid(text: &str) -> KrishivError {
    InvalidTimezone {
        time_zone: text.to_owned(),
    }
    .into()
}

/// Spark's two `getZoneId` normalizations, in order: a single hour digit
/// before a colon gains a zero (`+5:30` is `+05:30`), then a single minute
/// digit at the end does (`+10:7` is `+10:07`). Each replaces the first
/// match only.
fn pad_hour(name: &str) -> String {
    let b = name.as_bytes();
    let hour = (0..b.len().saturating_sub(2))
        .find(|&i| matches!(b[i], b'+' | b'-') && b[i + 1].is_ascii_digit() && b[i + 2] == b':');
    let name = match hour {
        Some(i) => format!("{}0{}", &name[..=i], &name[i + 1..]),
        None => name.to_owned(),
    };
    let b = name.as_bytes();
    let minute = (0..b.len().saturating_sub(2)).find(|&i| {
        matches!(b[i], b'+' | b'-')
            && b[i + 1].is_ascii_digit()
            && b[i + 2].is_ascii_digit()
            && b.get(i + 3) == Some(&b':')
            && b.get(i + 4).is_some_and(u8::is_ascii_digit)
            && b.get(i + 5).is_none()
    });
    match minute {
        Some(i) => format!("{}0{}", &name[..i + 4], &name[i + 4..]),
        None => name,
    }
}

/// `+h`, `+hh`, `+hh:mm`, `+hhmm`, `+hh:mm:ss` or `+hhmmss` (or with `-`)
/// as the seconds east of UTC, as Java's `ZoneOffset.of` takes them.
fn offset_seconds(s: &str) -> Option<i32> {
    let (sign, rest) = match s.split_at_checked(1)? {
        ("+", rest) => (1, rest),
        ("-", rest) => (-1, rest),
        _ => return None,
    };
    let colons = rest.matches(':').count();
    let bytes = rest.as_bytes();
    let shaped = colons == 0
        || (bytes.get(2) == Some(&b':')
            && (colons == 1 && rest.len() == 5
                || colons == 2 && rest.len() == 8 && bytes.get(5) == Some(&b':')));
    let digits: String = rest.chars().filter(|c| *c != ':').collect();
    if !shaped || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let part = |i: usize| -> Option<i32> { digits.get(i..i + 2)?.parse().ok() };
    let (h, m, sec) = match digits.len() {
        1 => (digits.parse().ok()?, 0, 0),
        2 => (part(0)?, 0, 0),
        4 => (part(0)?, part(2)?, 0),
        6 => (part(0)?, part(2)?, part(4)?),
        _ => return None,
    };
    // `i32::from_str` accepts a leading sign, so `h * 3600` below is the
    // only arithmetic: hours below 24 and a total within +-18:00 cannot
    // overflow.
    let total = h * 3600 + m * 60 + sec;
    (m < 60 && sec < 60 && total <= 18 * 3600).then_some(sign * total)
}

/// Java's `ZoneId.SHORT_IDS`: the three-letter ids `ZoneId.of` accepts.
/// Offsets stay fixed (`EST` is `-05:00`); the rest name regions.
const SHORT_IDS: [(&str, &str); 28] = [
    ("ACT", "Australia/Darwin"),
    ("AET", "Australia/Sydney"),
    ("AGT", "America/Argentina/Buenos_Aires"),
    ("ART", "Africa/Cairo"),
    ("AST", "America/Anchorage"),
    ("BET", "America/Sao_Paulo"),
    ("BST", "Asia/Dhaka"),
    ("CAT", "Africa/Harare"),
    ("CNT", "America/St_Johns"),
    ("CST", "America/Chicago"),
    ("CTT", "Asia/Shanghai"),
    ("EAT", "Africa/Addis_Ababa"),
    ("ECT", "Europe/Paris"),
    ("IET", "America/Indiana/Indianapolis"),
    ("IST", "Asia/Kolkata"),
    ("JST", "Asia/Tokyo"),
    ("MIT", "Pacific/Apia"),
    ("NET", "Asia/Yerevan"),
    ("NST", "Pacific/Auckland"),
    ("PLT", "Asia/Karachi"),
    ("PNT", "America/Phoenix"),
    ("PRT", "America/Puerto_Rico"),
    ("PST", "America/Los_Angeles"),
    ("SST", "Pacific/Guadalcanal"),
    ("VST", "Asia/Ho_Chi_Minh"),
    ("EST", "-05:00"),
    ("MST", "-07:00"),
    ("HST", "-10:00"),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// `resolve_zone` and [`TimeZoneRules`] agree with Spark's documented
    /// DST behavior; results are asserted, unverified against a Spark run
    /// (no Spark oracle is available here), except the offset spot checks
    /// from the golden formatter (`cell.rs`, pinned against Spark 4.2).
    #[test]
    fn fixed_offsets_resolve_without_the_database() {
        for (text, secs) in [
            ("UTC", 0),
            ("Z", 0),
            ("UT", 0),
            ("Etc/UTC", 0),
            ("+05:30", 19_800),
            ("-8", -28_800),
            ("UTC-08:00", -28_800),
            ("GMT+1", 3600),
            ("+5:30", 19_800),
            ("EST", -18_000),
            ("PST", -28_800),
        ] {
            let zone = resolve_zone(text).expect(text);
            assert_eq!(zone.offset_at_utc(0), secs, "{text}");
            assert_eq!(zone.offset_for_local(0), secs, "{text}");
        }
        // `PST` is Los Angeles, not a fixed offset: it moves with DST.
        let pst = resolve_zone("PST").expect("PST");
        assert_eq!(pst.offset_at_utc(1_700_000_000), -28_800);
        assert_eq!(pst.offset_at_utc(1_715_000_000), -25_200);
    }

    #[test]
    fn unknown_ids_are_invalid_timezones() {
        for text in [
            "",
            " ",
            " UTC",
            "UTC ",
            "xx",
            "abc",
            "etc/utc",
            "Mars/Olympus",
            "America/Foo",
            "+19:00",
            "++5",
        ] {
            let err = resolve_zone(text).expect_err(text);
            assert_eq!(
                err.error_class(),
                krishiv_common::error::ErrorClass::InvalidTimezone,
                "{text}"
            );
        }
    }

    /// Spring-forward gaps move forward, fall-back overlaps keep the
    /// earlier offset (unverified against a Spark run).
    #[test]
    fn dst_edges_follow_java_time() {
        let local = |y, m, d, h, mi| {
            jiff::civil::date(y, m, d)
                .at(h, mi, 0, 0)
                .to_zoned(TimeZone::UTC)
                .expect("test date")
                .timestamp()
                .as_second()
        };
        let la = resolve_zone("America/Los_Angeles").expect("LA");
        // 2024-03-10 02:30 does not exist: the offset before the gap (-8h)
        // moves it to 03:30 PDT.
        assert_eq!(la.offset_for_local(local(2024, 3, 10, 2, 30)), -8 * 3600);
        assert_eq!(la.offset_for_local(local(2024, 3, 10, 3, 30)), -7 * 3600);
        // 2024-11-03 01:30 happens twice: PDT first.
        assert_eq!(la.offset_for_local(local(2024, 11, 3, 1, 30)), -7 * 3600);
        assert_eq!(la.offset_for_local(local(2024, 11, 3, 2, 30)), -8 * 3600);
        assert_eq!(la.offset_at_utc(local(2024, 3, 10, 10, 0)), -7 * 3600);
        assert_eq!(la.offset_at_utc(local(2024, 3, 10, 9, 59)), -8 * 3600);
    }

    /// Spark 4.2.0's `VV` for each session zone.
    #[test]
    fn zone_ids_follow_java_time() {
        for (configured, id) in [
            ("UTC", "UTC"),
            ("GMT", "GMT"),
            ("UT", "UT"),
            ("UT+5", "UT+05:00"),
            ("UT+00:00", "UT"),
            ("Z", "Z"),
            ("Etc/UTC", "Etc/UTC"),
            ("+00:00", "Z"),
            ("-00:00", "Z"),
            ("GMT+00:00", "GMT"),
            ("UTC+00:00", "UTC"),
            ("+05:30", "+05:30"),
            ("+5:30", "+05:30"),
            ("+05", "+05:00"),
            ("-08:00", "-08:00"),
            ("GMT+5", "GMT+05:00"),
            ("UTC+05:30", "UTC+05:30"),
            ("GMT-08:00", "GMT-08:00"),
            ("America/Los_Angeles", "America/Los_Angeles"),
            (" GMT ", "GMT"),
        ] {
            assert_eq!(zone_id(configured), id, "{configured}");
        }
    }

    /// Asia/Kolkata (no DST since 1945) and Australia/Lord_Howe (a
    /// 30-minute DST): winter and summer offsets, a gap and an overlap.
    /// Unverified against a Spark run.
    #[test]
    fn kolkata_and_lord_howe_edges() {
        let utc = |y, mo, d, h, mi| {
            jiff::civil::date(y, mo, d)
                .at(h, mi, 0, 0)
                .to_zoned(TimeZone::UTC)
                .expect("test date")
                .timestamp()
                .as_second()
        };
        let local = |y, mo, d, h, mi| {
            jiff::civil::date(y, mo, d)
                .at(h, mi, 0, 0)
                .to_zoned(TimeZone::UTC)
                .expect("test date")
                .timestamp()
                .as_second()
        };
        let kolkata = resolve_zone("Asia/Kolkata").expect("Kolkata");
        assert_eq!(kolkata.offset_at_utc(utc(2024, 1, 15, 12, 0)), 19_800);
        assert_eq!(kolkata.offset_at_utc(utc(2024, 7, 15, 12, 0)), 19_800);
        assert_eq!(kolkata.offset_for_local(local(2024, 7, 15, 12, 0)), 19_800);
        let howe = resolve_zone("Australia/Lord_Howe").expect("Lord_Howe");
        // +11:00 in January (DST), +10:30 in July.
        assert_eq!(howe.offset_at_utc(utc(2024, 1, 15, 12, 0)), 39_600);
        assert_eq!(howe.offset_at_utc(utc(2024, 7, 15, 12, 0)), 37_800);
        // The 2024-10-06 02:15 wall time does not exist (clocks spring
        // 02:00 forward): the offset before the transition applies.
        assert_eq!(howe.offset_for_local(local(2024, 10, 6, 2, 15)), 37_800);
        // 2024-04-07 01:45 happens twice: the earlier offset first.
        assert_eq!(howe.offset_for_local(local(2024, 4, 7, 1, 45)), 39_600);
        // A singleton TZDB name resolves with DST too.
        let cet = resolve_zone("CET").expect("CET");
        assert_eq!(cet.offset_at_utc(utc(2024, 1, 15, 12, 0)), 3600);
        assert_eq!(cet.offset_at_utc(utc(2024, 7, 15, 12, 0)), 7200);
    }
}
