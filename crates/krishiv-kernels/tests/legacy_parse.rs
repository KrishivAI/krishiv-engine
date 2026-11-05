// SPDX-License-Identifier: Apache-2.0
//! The LEGACY parser (`SimpleDateFormat`) against Spark 4.2.0: every zone
//! name of the JVM's tables in January and July, and the edge cases of
//! `tests/scripts/gen_legacy_zone_strings.py`.

use krishiv_kernels::cast::FixedOffset;
use krishiv_kernels::datetime_pattern::legacy::LegacyPattern;

mod fixtures {
    include!("fixtures/legacy_zone_probes.rs");
}

#[test]
fn legacy_parses_match_spark() {
    let mut wrong = Vec::new();
    // Under Miri a sample: each probe scans the zone tables, and the parser
    // has no unsafe code.
    let step = if cfg!(miri) { 64 } else { 1 };
    for &(text, pattern, want) in fixtures::PROBES.iter().step_by(step) {
        let got = LegacyPattern::new(pattern)
            .unwrap()
            .parse_timestamp(text, &FixedOffset(0))
            .unwrap()
            .map(|micros| micros.div_euclid(1_000_000));
        if got != want {
            wrong.push(format!(
                "{text:?} with {pattern:?}: {got:?}, Spark {want:?}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {}:\n{}",
        wrong.len(),
        fixtures::PROBES.len(),
        wrong.join("\n")
    );
}

/// A zone name is first looked up in the session zone's own row
/// (`subParseZoneString`), so `IST` depends on the session zone. Expected
/// values: non-lenient `SimpleDateFormat` (`Locale.US`, Java 21) with the
/// session zone as its time zone, as Spark's `LegacySimpleTimestampFormatter`.
#[test]
fn zone_names_prefer_the_session_zone() {
    let cases = [
        ("Asia/Kolkata", "2024-07-01 12:00 IST", 1_719_815_400),
        ("Asia/Kolkata", "2024-07-01 12:00 ist", 1_719_815_400),
        ("Asia/Calcutta", "2024-07-01 12:00 IST", 1_719_815_400),
        ("Asia/Jerusalem", "2024-07-01 12:00 IST", 1_719_828_000),
        ("UTC", "2024-07-01 12:00 IST", 1_719_828_000),
        ("+05:30", "2024-07-01 12:00 IST", 1_719_828_000),
        ("Europe/Dublin", "2024-07-01 12:00 IST", 1_719_831_600),
        ("Europe/Dublin", "2024-01-01 12:00 IST", 1_704_106_800),
        (
            "Europe/Dublin",
            "2024-07-01 12:00 Greenwich Mean Time",
            1_719_835_200,
        ),
        // Dropped from the global table (Denver's names), found by id.
        ("America/Phoenix", "2024-07-01 12:00 MST", 1_719_860_400),
        ("America/Phoenix", "2024-07-01 12:00 MDT", 1_719_860_400),
        ("America/Denver", "2024-07-01 12:00 MST", 1_719_860_400),
        ("Australia/Brisbane", "2024-07-01 12:00 AEST", 1_719_799_200),
        ("Australia/Brisbane", "2024-01-01 12:00 AEDT", 1_704_074_400),
        ("PST", "2024-07-01 12:00 PDT", 1_719_860_400),
    ];
    let pattern = LegacyPattern::new("yyyy-MM-dd HH:mm z").unwrap();
    for (zone, text, want) in cases {
        let session = krishiv_kernels::datetime::resolve_zone(zone).unwrap();
        let got = pattern
            .parse_timestamp(text, &session)
            .unwrap()
            .map(|micros| micros.div_euclid(1_000_000));
        assert_eq!(got, Some(want), "{text:?} in {zone}");
    }
}

/// Field resolution edge cases from the #417 review, with the epoch
/// seconds a real Spark 4.2.0 run gives (`try_to_timestamp` under
/// `timeParserPolicy=LEGACY`, non-ANSI; `None` is NULL).
#[test]
fn resolution_edge_cases_match_spark() {
    let cases: [(&str, &str, &str, Option<i64>); 26] = [
        // A 24-hour field stays when no AM/PM marker completes the 12-hour one.
        (
            "UTC",
            "2024-01-01 13 01",
            "yyyy-MM-dd HH hh",
            Some(1_704_114_000),
        ),
        (
            "UTC",
            "2024-01-01 01 13",
            "yyyy-MM-dd hh HH",
            Some(1_704_114_000),
        ),
        (
            "UTC",
            "2024-01-01 13 01 PM",
            "yyyy-MM-dd HH hh a",
            Some(1_704_114_000),
        ),
        // A week year takes a sign; other numeric fields don't.
        ("UTC", "-5", "YYYY", Some(-62_325_417_600)),
        ("UTC", "-2019", "YYYY", Some(-125_880_739_200)),
        ("UTC", "-5", "yyyy", None),
        ("UTC", "2024-01--1", "yyyy-MM-dd", None),
        ("UTC", "2024--1-01", "yyyy-MM-dd", None),
        // A signed year is no two-digit year (-20 is week year -20).
        ("UTC", "-20", "Y", Some(-62_798_371_200)),
        ("UTC", "-20", "YY", Some(-62_798_371_200)),
        // `DecimalFormat` reads an exponent inside the field's width.
        ("UTC", "2E3", "yyyy", Some(946_684_800)),
        ("UTC", "2024 25E-1", "yyyy dd", Some(1_704_153_600)),
        ("UTC", "1E1", "HH", Some(36_000)),
        // The cutover month's Gregorian days fail a week-of-month resolution.
        ("UTC", "1582 10 2 1", "yyyy MM W u", None),
        ("UTC", "1582 10 2 1", "yyyy MM F u", None),
        // The calendar year is checked only when the text has one.
        ("UTC", "1", "w", Some(-345_600)),
        ("UTC", "1970 1", "yyyy w", None),
        // Skipped whitespace counts toward an adjacent field's width.
        (
            "UTC",
            "2024-01-01  123",
            "yyyy-MM-dd HHmm",
            Some(1_704_072_180),
        ),
        (
            "UTC",
            "2024-01-01 123",
            "yyyy-MM-dd HHmm",
            Some(1_704_110_580),
        ),
        // A weekday that doesn't match the window's century is no match.
        ("UTC", "99-01-01 Thu", "yy-MM-dd E", None),
        ("UTC", "99-01-01 Fri", "yy-MM-dd E", Some(915_148_800)),
        // Names match under Java's case folding (dotted capital I).
        ("UTC", "2024-APRİL-01", "yyyy-MMMM-dd", Some(1_711_929_600)),
        ("UTC", "2024-april-01", "yyyy-MMMM-dd", Some(1_711_929_600)),
        // London's 1941 double summer time: the standard offset is +00:00.
        (
            "Europe/London",
            "1941-07-01 12:00 British Summer Time",
            "yyyy-MM-dd HH:mm z",
            Some(-899_470_800),
        ),
        (
            "Europe/London",
            "1941-07-01 12:00 BST",
            "yyyy-MM-dd HH:mm z",
            Some(-899_470_800),
        ),
        (
            "Europe/London",
            "1941-07-01 12:00 GMT",
            "yyyy-MM-dd HH:mm z",
            Some(-899_467_200),
        ),
    ];
    let mut wrong = Vec::new();
    for (zone, text, pattern, want) in cases {
        let session = krishiv_kernels::datetime::resolve_zone(zone).unwrap();
        let got = LegacyPattern::new(pattern)
            .unwrap()
            .parse_timestamp(text, &session)
            .unwrap()
            .map(|micros| micros.div_euclid(1_000_000));
        if got != want {
            wrong.push(format!(
                "{text:?} with {pattern:?} in {zone}: {got:?}, Spark {want:?}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// `DecimalFormat` hands a field over as a `Long` and `intValue()` keeps its
/// low 32 bits, so an oversized number wraps (Spark 4.2.0 LEGACY:
/// `to_timestamp('4294969320-01-01', 'yyyy-MM-dd')` is 2024-01-01); past
/// the `Long` range the parse fails.
#[test]
fn oversized_numbers_wrap_to_an_int() {
    let p = LegacyPattern::new("yyyy-MM-dd").unwrap();
    let read = |s: &str| p.parse_timestamp(s, &FixedOffset(0)).unwrap();
    let jan_1_2024 = Some(1_704_067_200_000_000);
    assert_eq!(read("4294969320-01-01"), jan_1_2024);
    assert_eq!(read("2024-4294967297-01"), jan_1_2024);
    assert_eq!(read("2024-01-4294967297"), jan_1_2024);
    assert_eq!(read("9223372036854775807-01-01"), None);
    assert_eq!(read("99999999999999999999-01-01"), None);
}
