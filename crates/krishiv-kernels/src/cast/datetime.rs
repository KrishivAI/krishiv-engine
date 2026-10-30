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
/// Kernels never resolve zone names: the caller supplies the rules
/// ([`FixedOffset`] covers `UTC` and `+hh:mm` zones; a tz-database
/// implementation belongs to the session layer).
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
    /// fixed offset: `UTC`, `GMT`, `UT`, `Z`, `Etc/UTC` or `[UTC|GMT|UT][+-]hh[:mm]`
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
            .or_else(|| zone.strip_prefix("UT"))
            .unwrap_or(zone);
        let (sign, hm) = match rest.as_bytes().first()? {
            b'+' => (1, &rest[1..]),
            b'-' => (-1, &rest[1..]),
            _ => return None,
        };
        let (h_str, m_str) = hm.split_once(':').unwrap_or((hm, "0"));
        // `i32::from_str` accepts a leading `+`/`-`, so `++5` or
        // `+05:+30` would parse; require 1-2 ASCII digits per field.
        if !(1..=2).contains(&h_str.len()) || !(1..=2).contains(&m_str.len()) {
            return None;
        }
        if !h_str.bytes().all(|b| b.is_ascii_digit()) || !m_str.bytes().all(|b| b.is_ascii_digit())
        {
            return None;
        }
        let (h, m): (i32, i32) = (h_str.parse().ok()?, m_str.parse().ok()?);
        let total = h * 3600 + m * 60;
        (h <= 18 && m < 60 && total <= 18 * 3600).then(|| Self(sign * total))
    }
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
    fn session_zones_parse_as_fixed_offsets_or_none() {
        for (zone, offset) in [
            ("UTC", Some(0)),
            (" GMT ", Some(0)),
            ("Z", Some(0)),
            ("Etc/UTC", Some(0)),
            ("+05:30", Some(19_800)),
            ("UTC-08:00", Some(-28_800)),
            ("GMT+1", Some(3600)),
            ("UT", Some(0)),
            ("UT-08:00", Some(-28_800)),
            ("-18:00", Some(-64_800)),
            ("+18:00", Some(64_800)),
        ] {
            assert_eq!(FixedOffset::parse(zone), offset.map(FixedOffset), "{zone}");
        }
        // Region ids need the tz database; malformed or out-of-range
        // offsets are not zones.
        for zone in [
            "America/Los_Angeles",
            "",
            "+19:00",
            "+05:60",
            "+ab",
            "UTC*1",
            "++5",
            "+05:+30",
            "+18:30",
            "+18:01",
            "-18:30",
            "+5:30:00",
            "+",
        ] {
            assert_eq!(FixedOffset::parse(zone), None, "{zone}");
        }
    }

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
