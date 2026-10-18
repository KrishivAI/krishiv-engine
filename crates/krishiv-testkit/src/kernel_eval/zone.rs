// SPDX-License-Identifier: Apache-2.0
//! Named time zones for the kernel evaluator and its tests: `jiff`'s bundled
//! tz database behind the kernels' [`TimeZoneRules`] (the kernels never see a
//! zone name; the session layer supplies the same adapter in P2-T021b).

use jiff::Timestamp;
use jiff::tz::TimeZone;
use krishiv_kernels::cast::TimeZoneRules;

/// A tz-database zone (`America/Los_Angeles`).
#[derive(Debug, Clone)]
pub struct JiffZone(TimeZone);

impl JiffZone {
    /// The zone named `name`, or `None` if the tz database has no such zone.
    #[must_use]
    pub fn new(name: &str) -> Option<Self> {
        TimeZone::get(name).ok().map(Self)
    }
}

impl TimeZoneRules for JiffZone {
    fn offset_at_utc(&self, utc_seconds: i64) -> i32 {
        Timestamp::from_second(utc_seconds)
            .map(|t| self.0.to_offset(t).seconds())
            .unwrap_or(0)
    }

    fn offset_for_local(&self, local_seconds: i64) -> i32 {
        // The local date-time read as if it were UTC, resolved the way
        // `java.time.ZonedDateTime.of` does: the earlier offset in an overlap,
        // forward by the gap's length in a gap. The offset to apply is then
        // the difference between the local time and the resulting instant.
        let resolved = Timestamp::from_second(local_seconds)
            .map(|t| t.to_zoned(TimeZone::UTC).datetime())
            .and_then(|dt| self.0.to_ambiguous_zoned(dt).compatible());
        resolved
            .ok()
            .and_then(|z| i32::try_from(local_seconds.checked_sub(z.timestamp().as_second())?).ok())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dst_gaps_and_overlaps_resolve_like_java() {
        let la = JiffZone::new("America/Los_Angeles").unwrap();
        // 2024-03-10 02:30 does not exist: the offset before the gap (-8h)
        // moves it to 03:30 PDT; 2024-11-03 01:30 happens twice: PDT first.
        let local = |y, m, d, h, mi| {
            jiff::civil::date(y, m, d)
                .at(h, mi, 0, 0)
                .to_zoned(TimeZone::UTC)
                .unwrap()
                .timestamp()
                .as_second()
        };
        assert_eq!(la.offset_for_local(local(2024, 3, 10, 2, 30)), -8 * 3600);
        assert_eq!(la.offset_for_local(local(2024, 3, 10, 3, 30)), -7 * 3600);
        assert_eq!(la.offset_for_local(local(2024, 11, 3, 1, 30)), -7 * 3600);
        assert_eq!(la.offset_for_local(local(2024, 11, 3, 2, 30)), -8 * 3600);
        assert_eq!(la.offset_at_utc(local(2024, 3, 10, 10, 0)), -7 * 3600);
        assert_eq!(la.offset_at_utc(local(2024, 3, 10, 9, 59)), -8 * 3600);
        assert_eq!(la.offset_at_utc(i64::MAX), 0);
        assert_eq!(la.offset_for_local(i64::MAX), 0);
        assert!(JiffZone::new("Mars/Olympus").is_none());
    }
}
