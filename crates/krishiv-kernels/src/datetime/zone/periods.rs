// SPDX-License-Identifier: Apache-2.0
//! A region's offsets from 1900 to 2100 as a sorted list of periods, so the
//! per-row offset lookups of the datetime kernels are a binary search over a
//! few hundred entries instead of a `jiff` lookup (and, for a local time,
//! a civil date-time resolution) each (KN-4: `date_trunc` on `timestamp_ntz`
//! ran about 4 times slower in `Europe/Stockholm` than in `UTC`).
//!
//! The table answers exactly what the `jiff` path answers inside its range
//! (`table_matches_jiff_in_every_zone` checks every zone of the bundled
//! database around each transition); outside it, or for a local time it
//! cannot place, it answers `None` and the caller asks `jiff`.
#![expect(
    clippy::arithmetic_side_effects,
    reason = "seconds stay within 1900..2100 plus a day, far from the i64 range; indices are \
              bounded by the table length"
)]

use jiff::Timestamp;
use jiff::tz::TimeZone;

/// 1900-01-01T00:00:00Z, the first instant the table covers.
const FROM: i64 = -2_208_988_800;
/// 2100-01-01T00:00:00Z, the first instant past the table.
const TO: i64 = 4_102_444_800;
/// More than any offset `jiff` represents (under 26 hours either way): a
/// local time `L` is an instant in `[L - SWING, L + SWING]`.
const SWING: i64 = 26 * 3600;
/// A zone with more transitions in 200 years than this gets no table.
const MAX_PERIODS: usize = 4096;

/// The periods of constant offset of one region between [`FROM`] and [`TO`].
#[derive(Debug)]
pub(super) struct Periods {
    /// The UTC second each period starts at, ascending; the first is [`FROM`].
    starts: Vec<i64>,
    /// The offset (seconds east of UTC) of each period.
    offsets: Vec<i32>,
}

impl Periods {
    /// The table of `zone`; `None` if the database cannot list its
    /// transitions in the range or lists implausibly many.
    pub(super) fn build(zone: &TimeZone) -> Option<Self> {
        let from = Timestamp::from_second(FROM).ok()?;
        let mut starts = vec![FROM];
        let mut offsets = vec![zone.to_offset(from).seconds()];
        for transition in zone.following(from) {
            let start = transition.timestamp().as_second();
            if start >= TO {
                break;
            }
            if starts.len() >= MAX_PERIODS {
                return None;
            }
            // `following` lists transitions after `from`; one at `from`
            // itself is already in the first offset.
            if start > FROM {
                starts.push(start);
                offsets.push(transition.offset().seconds());
            }
        }
        Some(Self { starts, offsets })
    }

    /// The end (exclusive) of period `i`.
    fn end(&self, i: usize) -> i64 {
        self.starts.get(i + 1).copied().unwrap_or(TO)
    }

    /// The offset at the UTC second `utc`, or `None` outside the table.
    pub(super) fn offset_at_utc(&self, utc: i64) -> Option<i32> {
        if !(FROM..TO).contains(&utc) {
            return None;
        }
        // `starts[0] == FROM <= utc`, so at least one start qualifies.
        let i = self.starts.partition_point(|&s| s <= utc).checked_sub(1)?;
        self.offsets.get(i).copied()
    }

    /// The offset of the local second `local` as `ZonedDateTime.of` resolves
    /// it (the earlier offset in an overlap, the offset before the
    /// transition in a gap), or `None` where the table cannot say.
    pub(super) fn offset_for_local(&self, local: i64) -> Option<i32> {
        if !(FROM + SWING..TO - SWING).contains(&local) {
            return None;
        }
        let first = self
            .starts
            .partition_point(|&s| s <= local - SWING)
            .saturating_sub(1);
        let mut i = first;
        while let (Some(&start), Some(&offset)) = (self.starts.get(i), self.offsets.get(i)) {
            if start > local + SWING {
                break;
            }
            let utc = local - i64::from(offset);
            let end = self.end(i);
            if start <= utc && utc < end {
                // The first period that holds the local time gives the
                // earliest instant: the earlier offset of an overlap.
                return Some(offset);
            }
            // A gap at this period's end: the local time is past the end
            // with this offset and before it with the next one. This assumes
            // no later period holds the local time either, true of every
            // zone in the database (`table_matches_jiff_in_every_zone`).
            if utc >= end
                && let Some(&next) = self.offsets.get(i + 1)
                && local - i64::from(next) < end
            {
                return Some(offset);
            }
            i += 1;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cast::TimeZoneRules;
    use crate::datetime::zone::ZoneRef;

    /// The `jiff` answers, by the code path the table replaces.
    fn jiff_at_utc(zone: &TimeZone, utc: i64) -> i32 {
        Timestamp::from_second(utc)
            .map(|t| zone.to_offset(t).seconds())
            .unwrap_or(0)
    }

    fn jiff_for_local(zone: &TimeZone, local: i64) -> i32 {
        Timestamp::from_second(local)
            .map(|t| t.to_zoned(TimeZone::UTC).datetime())
            .and_then(|dt| zone.to_ambiguous_zoned(dt).compatible())
            .ok()
            .and_then(|z| i32::try_from(local - z.timestamp().as_second()).ok())
            .unwrap_or(0)
    }

    fn check(name: &str, zone: &TimeZone, table: &Periods, utc: i64, local: i64) {
        // Inside the range the table must answer, or the fast path is lost.
        if (FROM..TO).contains(&utc) {
            let got = table.offset_at_utc(utc);
            assert_eq!(
                got,
                Some(jiff_at_utc(zone, utc)),
                "{name}: offset at UTC {utc}"
            );
        }
        if (FROM + SWING..TO - SWING).contains(&local) {
            let got = table.offset_for_local(local);
            assert!(got.is_some(), "{name}: no table answer for local {local}");
            let got = got.unwrap_or_default();
            if is_nuuk_2023(name, local) {
                // `jiff` reads a fold here that its own `to_offset` and Java do
                // not have (`nuuk_follows_java_where_jiff_reads_a_fold`).
                assert_eq!(got, -7200, "{name}: offset for local {local}");
            } else {
                assert_eq!(
                    got,
                    jiff_for_local(zone, local),
                    "{name}: offset for local {local}"
                );
            }
        }
    }

    /// 2023-10-28T22:00..2023-10-29T00:00 local in Greenland, where the zone
    /// moved to -02:00 standard time without a change of offset.
    fn is_nuuk_2023(name: &str, local: i64) -> bool {
        matches!(name, "America/Nuuk" | "America/Godthab")
            && (1_698_530_400..1_698_537_600).contains(&local)
    }

    #[test]
    fn table_matches_jiff_in_every_zone() {
        let names = jiff::tz::db().available();
        let mut zones = 0;
        for name in names {
            let name = name.as_str().to_owned();
            let Ok(zone) = TimeZone::get(name.as_str()) else {
                continue;
            };
            let Some(table) = Periods::build(&zone) else {
                panic!("{name}: no table");
            };
            zones += 1;
            for (i, &t) in table.starts.iter().enumerate() {
                let before = i.checked_sub(1).and_then(|j| table.offsets.get(j)).copied();
                let after = table.offsets.get(i).copied().unwrap_or(0);
                let before = i64::from(before.unwrap_or(after));
                let after = i64::from(after);
                for d in [-7200, -3601, -3600, -1, 0, 1, 1799, 1800, 3599, 3600, 7200] {
                    check(&name, &zone, &table, t + d, t + before + d);
                    check(&name, &zone, &table, t + d, t + after + d);
                }
            }
            // A spread of ordinary instants, 1900 to 2100.
            let mut s = FROM + 12_345;
            while s < TO {
                check(&name, &zone, &table, s, s);
                s += 86_400 * 97 + 3_607;
            }
        }
        assert!(zones > 400, "only {zones} zones in the bundled database");
    }

    #[test]
    fn outside_the_table_is_none() {
        let zone = TimeZone::get("Europe/Stockholm").unwrap();
        let table = Periods::build(&zone).unwrap();
        assert_eq!(table.offset_at_utc(FROM - 1), None);
        assert_eq!(table.offset_at_utc(TO), None);
        assert_eq!(table.offset_for_local(FROM), None);
        assert_eq!(table.offset_for_local(TO - 1), None);
        assert_eq!(table.offset_at_utc(1_700_000_000), Some(3600));
    }

    #[test]
    fn region_zones_answer_through_the_table_and_past_it() {
        let ZoneRef::Named(region) = crate::datetime::resolve_zone("Europe/Stockholm").unwrap()
        else {
            panic!("a region");
        };
        let zone = ZoneRef::Named(region);
        // 2024-03-31 02:30 local is in the spring gap: the offset before it.
        let gap = 1_711_852_200;
        assert_eq!(zone.offset_for_local(gap), 3600);
        // 2024-10-27 02:30 local is in the autumn overlap: the earlier offset.
        let overlap = 1_729_996_200;
        assert_eq!(zone.offset_for_local(overlap), 7200);
        // Past 2100 the database still answers.
        assert_eq!(zone.offset_at_utc(TO + 86_400 * 180), 7200);
    }

    /// `ZonedDateTime.of(local, America/Nuuk).getOffset()` on Spark 4.2.1's
    /// JVM (Java 21.0.13, run 2026-11-14): -02:00 for each of these local
    /// times, with one valid offset. `jiff` 0.2.38's `to_ambiguous_zoned`
    /// reads a fold from -01:00 to -02:00 at 2023-10-29T01:00Z and answers
    /// -01:00 for the hour before 23:00 local; the table answers as Java.
    #[test]
    fn nuuk_follows_java_where_jiff_reads_a_fold() {
        let zone = crate::datetime::resolve_zone("America/Nuuk").unwrap();
        // 2023-10-28T21:59:59, 22:00, 22:30, 22:59:59, 23:00, 23:30, 23:59:59, 2023-10-29T00:00.
        for local in [
            1_698_530_399,
            1_698_530_400,
            1_698_532_200,
            1_698_533_999,
            1_698_534_000,
            1_698_535_800,
            1_698_537_599,
            1_698_537_600,
        ] {
            assert_eq!(zone.offset_for_local(local), -7200, "local {local}");
        }
    }
}
