// SPDX-License-Identifier: Apache-2.0
//! Proleptic Gregorian dates (Spark 3+: `java.time`, no Julian gap), time
//! zone rules, and the `date`/`timestamp` conversions and renderings.
#![expect(
    clippy::arithmetic_side_effects,
    reason = "civil-date arithmetic on |year| < 10^8 and day numbers from i64 micros / 86400e6 \
              (|days| < 1.1e8): far from the i64 limits"
)]

/// Microseconds per day.
pub(crate) const MICROS_PER_DAY: i64 = 86_400_000_000;
const MICROS_PER_SECOND: i64 = 1_000_000;

/// The UTC-offset rules of a session time zone (`spark.sql.session.timeZone`).
///
/// Time zones come in as [`TimeZoneRules`] (the session zone resolved by the
/// caller: [`FixedOffset`] for `UTC` and `+hh:mm`, [`SessionZone`] for
/// region ids through `jiff`'s bundled tz database).
pub trait TimeZoneRules: Send + Sync + std::fmt::Debug {
    /// The offset east of UTC, in seconds, at the UTC instant `utc_seconds`.
    fn offset_at_utc(&self, utc_seconds: i64) -> i32;
    /// The offset to apply to the local date-time `local_seconds`, resolved
    /// as `java.time.ZonedDateTime.of` does: the earlier offset in an
    /// overlap, the offset before the transition in a gap (which moves the
    /// time forward by the gap's length).
    fn offset_for_local(&self, local_seconds: i64) -> i32;
}

/// A zone with one fixed offset (`UTC` is `FixedOffset(0)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FixedOffset(pub i32);

impl TimeZoneRules for FixedOffset {
    fn offset_at_utc(&self, _utc_seconds: i64) -> i32 {
        self.0
    }
    fn offset_for_local(&self, _local_seconds: i64) -> i32 {
        self.0
    }
}

impl FixedOffset {
    /// The zone a `spark.sql.session.timeZone` value names, when it is a
    /// fixed offset: `UTC`, `GMT`, `Z`, `Etc/UTC` or `[UTC|GMT][+-]hh[:mm]`
    /// (hours up to 18). `None` for region ids, which need the time-zone
    /// database, and for malformed zones.
    #[must_use]
    pub fn parse(zone: &str) -> Option<Self> {
        let zone = zone.trim();
        if ["UTC", "GMT", "UT", "Z", "Etc/UTC"].contains(&zone) {
            return Some(Self(0));
        }
        let rest = zone
            .strip_prefix("UTC")
            .or_else(|| zone.strip_prefix("GMT"))
            .unwrap_or(zone);
        let (sign, hm) = match rest.as_bytes().first()? {
            b'+' => (1, &rest[1..]),
            b'-' => (-1, &rest[1..]),
            _ => return None,
        };
        let (h, m) = hm.split_once(':').unwrap_or((hm, "0"));
        let (h, m): (i32, i32) = (h.parse().ok()?, m.parse().ok()?);
        ((0..=18).contains(&h) && (0..60).contains(&m)).then(|| Self(sign * (h * 3600 + m * 60)))
    }
}

/// A session time zone with full region support (`America/Los_Angeles`),
/// backed by `jiff`'s bundled tz database behind [`TimeZoneRules`].
/// Parses the same Spark zone forms as [`FixedOffset`] plus region ids
/// and Java's short ids; unknown names are `None`.
#[derive(Debug, Clone)]
pub struct SessionZone(jiff::tz::TimeZone);

impl SessionZone {
    /// The zone `name` as Spark's `DateTimeUtils.getZoneId` takes it:
    /// `Z`, an offset, `UTC`/`GMT`/`UT` with an optional offset, a short
    /// id or a region name. `None` for unknown names.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        parse_zone(name).map(Self)
    }

    /// UTC.
    #[must_use]
    pub fn utc() -> Self {
        Self(jiff::tz::TimeZone::UTC)
    }
}

impl TimeZoneRules for SessionZone {
    fn offset_at_utc(&self, utc_seconds: i64) -> i32 {
        jiff::Timestamp::from_second(utc_seconds)
            .map(|t| self.0.to_offset(t).seconds())
            .unwrap_or(0)
    }

    fn offset_for_local(&self, local_seconds: i64) -> i32 {
        // The local date-time read as if it were UTC, resolved the way
        // `java.time.ZonedDateTime.of` does: the earlier offset in an
        // overlap, forward by the gap's length in a gap.
        let resolved = jiff::Timestamp::from_second(local_seconds)
            .map(|t| t.to_zoned(jiff::tz::TimeZone::UTC).datetime())
            .and_then(|dt| self.0.to_ambiguous_zoned(dt).compatible());
        resolved
            .ok()
            .and_then(|z| i32::try_from(local_seconds.checked_sub(z.timestamp().as_second())?).ok())
            .unwrap_or(0)
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
    let total = h
        .saturating_mul(3600)
        .saturating_add(m.saturating_mul(60))
        .saturating_add(sec);
    (m < 60 && sec < 60 && total <= 18 * 3600).then_some(sign * total)
}

/// Java's `ZoneId.SHORT_IDS`: the three-letter IDs `ZoneId.of` accepts.
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

/// Spark's two `getZoneId` normalizations (`SparkDateTimeUtils`), in order:
/// `singleHourTz` `(\+|\-)(\d):` → `$10$2:` (`+5:30` is `+05:30`), then
/// `singleMinuteTz` `(\+|\-)(\d\d):(\d)$` → `$1$2:0$3` (`+10:7` is
/// `+10:07`). Each replaces the first match only.
fn pad_hour(name: &str) -> String {
    let b = name.as_bytes();
    let hour = (0..b.len().saturating_sub(2))
        .find(|&i| matches!(b[i], b'+' | b'-') && b[i + 1].is_ascii_digit() && b[i + 2] == b':');
    let name = match hour {
        Some(i) => format!("{}0{}", &name[..=i], &name[i + 1..]),
        None => name.to_owned(),
    };
    let b = name.as_bytes();
    let minute = (0..b.len().saturating_sub(4)).find(|&i| {
        matches!(b[i], b'+' | b'-')
            && b[i + 1].is_ascii_digit()
            && b[i + 2].is_ascii_digit()
            && b[i + 3] == b':'
            && b[i + 4].is_ascii_digit()
            && i + 5 == b.len()
    });
    match minute {
        Some(i) => format!("{}0{}", &name[..i + 4], &name[i + 4..]),
        None => name,
    }
}

/// A zone ID as Spark's `DateTimeUtils.getZoneId` takes it (`ZoneId.of`
/// with the short IDs): `Z`, an offset, `UTC`/`GMT`/`UT` with an optional
/// offset, a short ID or a region name.
fn parse_zone(name: &str) -> Option<jiff::tz::TimeZone> {
    use jiff::tz::{Offset, TimeZone};
    let fixed = |secs: i32| Offset::from_seconds(secs).ok().map(TimeZone::fixed);
    let padded = pad_hour(name);
    let name = SHORT_IDS
        .iter()
        .find(|(short, _)| *short == padded)
        .map_or(padded.as_str(), |(_, id)| id);
    if matches!(name, "Z" | "UTC" | "GMT" | "UT") {
        return Some(TimeZone::UTC);
    }
    if let Some(secs) = offset_seconds(name) {
        return fixed(secs);
    }
    for prefix in ["UTC", "GMT", "UT"] {
        if let Some(secs) = name.strip_prefix(prefix).and_then(offset_seconds) {
            return fixed(secs);
        }
    }
    TimeZone::get(name).ok()
}

/// Days since 1970-01-01 of a proleptic Gregorian date, or `None` for an
/// invalid month/day (Howard Hinnant's `days_from_civil`).
pub(crate) fn days_from_civil(year: i64, month: u32, day: u32) -> Option<i64> {
    if !(1..=12).contains(&month) || day == 0 || day > days_in_month(year, month) {
        return None;
    }
    let y = if month <= 2 {
        year.checked_sub(1)?
    } else {
        year
    };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let m = i64::from(month);
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era.checked_mul(146_097)?
        .checked_add(doe)?
        .checked_sub(719_468)
}

/// `(year, month, day)` of a day number (inverse of [`days_from_civil`]).
pub(crate) fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    // month is 1..=12 and day 1..=31 by construction.
    (
        year,
        u32::try_from(month).unwrap_or(1),
        u32::try_from(day).unwrap_or(1),
    )
}

fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        2 if year.rem_euclid(4) == 0
            && (year.rem_euclid(100) != 0 || year.rem_euclid(400) == 0) =>
        {
            29
        }
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Spark's date rendering: `yyyy-MM-dd`, a `+` before years above 9999 and
/// a `-` before negative ones (astronomical numbering: year 0 exists).
pub(crate) fn format_date(days: i64, out: &mut String) {
    use std::fmt::Write as _;
    let (y, m, d) = civil_from_days(days);
    let _ = match y {
        10_000.. => write!(out, "+{y}-{m:02}-{d:02}"),
        0.. => write!(out, "{y:04}-{m:02}-{d:02}"),
        _ => write!(out, "-{:04}-{m:02}-{d:02}", y.unsigned_abs()),
    };
}

/// A timestamp rendering of local `micros`: the date, `HH:mm:ss`, and the
/// fraction without trailing zeros.
pub(crate) fn format_timestamp(micros: i64, out: &mut String) {
    use std::fmt::Write as _;
    let days = micros.div_euclid(MICROS_PER_DAY);
    let rest = micros.rem_euclid(MICROS_PER_DAY);
    format_date(days, out);
    let secs = rest / MICROS_PER_SECOND;
    let frac = rest % MICROS_PER_SECOND;
    let _ = write!(
        out,
        " {:02}:{:02}:{:02}",
        secs / 3600,
        secs / 60 % 60,
        secs % 60
    );
    if frac != 0 {
        let digits = format!("{frac:06}");
        let _ = write!(out, ".{}", digits.trim_end_matches('0'));
    }
}

/// UTC micros of local `micros` in `tz`, or `None` past the range.
pub(crate) fn local_to_utc(local: i64, tz: &dyn TimeZoneRules) -> Option<i64> {
    let offset = tz.offset_for_local(local.div_euclid(MICROS_PER_SECOND));
    local.checked_sub(i64::from(offset).checked_mul(MICROS_PER_SECOND)?)
}

/// Local micros of UTC `micros` in `tz`, or `None` past the range.
pub(crate) fn utc_to_local(utc: i64, tz: &dyn TimeZoneRules) -> Option<i64> {
    let offset = tz.offset_at_utc(utc.div_euclid(MICROS_PER_SECOND));
    utc.checked_add(i64::from(offset).checked_mul(MICROS_PER_SECOND)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_round_trips_and_spark_values() {
        // Values recorded from Spark 4.2 (`unix_date`).
        assert_eq!(days_from_civil(2024, 2, 29), Some(19_782));
        assert_eq!(days_from_civil(1582, 10, 10), Some(-141_432));
        assert_eq!(days_from_civil(1, 1, 1), Some(-719_162));
        assert_eq!(days_from_civil(0, 1, 1), Some(-719_528));
        assert_eq!(days_from_civil(-1, 1, 1), Some(-719_893));
        assert_eq!(days_from_civil(5_881_580, 7, 11), Some(i64::from(i32::MAX)));
        assert_eq!(
            days_from_civil(-5_877_641, 6, 23),
            Some(i64::from(i32::MIN))
        );
        assert_eq!(days_from_civil(2023, 2, 29), None);
        assert_eq!(days_from_civil(1900, 2, 29), None);
        assert_eq!(days_from_civil(2000, 2, 29), Some(11_016));
        assert_eq!(days_from_civil(2024, 13, 1), None);
        assert_eq!(days_from_civil(2024, 4, 31), None);
        for d in [-800_000_i64, -719_528, -1, 0, 59, 19_782, 2_932_896] {
            let (y, m, day) = civil_from_days(d);
            assert_eq!(days_from_civil(y, m, day), Some(d));
        }
    }

    #[test]
    fn renderings_match_spark() {
        let date = |d| {
            let mut s = String::new();
            format_date(d, &mut s);
            s
        };
        assert_eq!(date(19_782), "2024-02-29");
        assert_eq!(date(-719_893), "-0001-01-01");
        assert_eq!(date(days_from_civil(10_000, 1, 1).unwrap()), "+10000-01-01");
        assert_eq!(date(days_from_civil(-44, 3, 15).unwrap()), "-0044-03-15");
        assert_eq!(date(days_from_civil(0, 6, 1).unwrap()), "0000-06-01");
        let ts = |m| {
            let mut s = String::new();
            format_timestamp(m, &mut s);
            s
        };
        assert_eq!(ts(1_709_210_096_123_456), "2024-02-29 12:34:56.123456");
        assert_eq!(ts(1_709_210_096_000_000), "2024-02-29 12:34:56");
        assert_eq!(ts(1_709_210_096_120_000), "2024-02-29 12:34:56.12");
        assert_eq!(ts(-1), "1969-12-31 23:59:59.999999");
        let la = FixedOffset(-8 * 3600);
        assert_eq!(utc_to_local(0, &la), Some(-8 * 3600 * MICROS_PER_SECOND));
        assert_eq!(local_to_utc(0, &la), Some(8 * 3600 * MICROS_PER_SECOND));
        assert_eq!(local_to_utc(i64::MIN, &FixedOffset(3600)), None);
        assert_eq!(utc_to_local(i64::MAX, &FixedOffset(3600)), None);
    }
}
