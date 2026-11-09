// SPDX-License-Identifier: Apache-2.0
//! Unit checks of datetime patterns; the Spark differential is
//! `krishiv-testkit/tests/datetime_pattern_differential.rs`.

use super::*;
use krishiv_common::error::ErrorClass;

use crate::cast::FixedOffset;

fn utc() -> Zone<'static> {
    Zone {
        id: "UTC",
        rules: &FixedOffset(0),
    }
}

#[test]
fn spark_write_defaults_round_trip() {
    let p = DateTimePattern::new("yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX", Purpose::Format).unwrap();
    let r = DateTimePattern::new("yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX", Purpose::Parse).unwrap();
    for m in [0, 1_577_934_245_123_456, -62_135_596_800_000_001] {
        let s = p.format_timestamp(m, utc()).unwrap();
        assert_eq!(
            r.parse_timestamp(&s, &FixedOffset(0)).unwrap(),
            Some(m),
            "{s}"
        );
    }
    assert!(p.has_zone() && p.has_time());
}

#[test]
fn dates_and_local_values() {
    let p = DateTimePattern::new("yyyy-MM-dd", Purpose::Format).unwrap();
    assert_eq!(p.format_date(18_263).unwrap(), "2020-01-02");
    assert!(!p.has_time() && !p.has_zone());
    let t = DateTimePattern::new("HH:mm", Purpose::Format).unwrap();
    assert_eq!(
        t.format_date(0).unwrap_err().error_class(),
        ErrorClass::KrishivUnsupportedFeature
    );
    let z = DateTimePattern::new("yyyy XXX", Purpose::Format).unwrap();
    assert!(z.format_local(0).is_err());
    let opt = DateTimePattern::new("yyyy[ HH]", Purpose::Format).unwrap();
    assert_eq!(opt.format_date(0).unwrap(), "1970");
    let r = DateTimePattern::new("yyyy-MM-dd HH:mm XXX", Purpose::Parse).unwrap();
    assert_eq!(
        r.parse_local("1970-01-01 01:00 +01:00", false).unwrap(),
        None
    );
    assert_eq!(
        r.parse_local("1970-01-01 01:00 +01:00", true).unwrap(),
        Some(3_600_000_000)
    );
    assert_eq!(r.parse_date("1970-01-02 01:00 Z").unwrap(), Some(1));
    assert_eq!(r.parse_date("1970-01-02").unwrap(), None);
    assert_eq!(r.pattern(), "yyyy-MM-dd HH:mm XXX");
}

#[test]
fn region_zones_resolve_against_the_bundled_database() {
    let p = DateTimePattern::new("yyyy VV", Purpose::Parse).unwrap();
    // Midnight in Los Angeles (UTC-08:00 in January) is 08:00 UTC.
    assert_eq!(
        p.parse_timestamp("2020 America/Los_Angeles", &FixedOffset(0))
            .unwrap(),
        Some(18_262 * 86_400_000_000 + 8 * 3_600_000_000)
    );
    assert_eq!(
        p.parse_local("2020 America/Los_Angeles", true).unwrap(),
        Some(18_262 * 86_400_000_000)
    );
    assert_eq!(
        p.parse_date("2020 America/Los_Angeles").unwrap(),
        Some(18_262)
    );
    // Unknown ids read as no timestamp, as in Spark's lenient readers.
    assert_eq!(
        p.parse_timestamp("2020 Foo/Bar", &FixedOffset(0)).unwrap(),
        None
    );
    assert_eq!(p.parse_local("2020 Foo/Bar", true).unwrap(), None);
    assert_eq!(p.parse_date("2020 Foo/Bar").unwrap(), None);
    assert_eq!(p.parse_local("2020 Foo/Bar", false).unwrap(), None);
    let f = DateTimePattern::new("z", Purpose::Format).unwrap();
    let la = Zone {
        id: "America/Los_Angeles",
        rules: &FixedOffset(-28_800),
    };
    assert_eq!(f.format_timestamp(0, la).unwrap(), "PST");
    assert!(DateTimePattern::new("v", Purpose::Format).is_err());
}

#[test]
fn fixed_zone_ids() {
    for (id, want) in [
        ("UTC", Some(0)),
        ("Etc/UTC", Some(0)),
        ("+05:30", Some(19_800)),
        ("-8", Some(-28_800)),
        ("UTC+01:00", Some(3600)),
        ("GMT-0530", Some(-19_800)),
        ("+05:30:15", Some(19_815)),
        ("+053015", Some(19_815)),
        ("+19", None),
        ("America/Los_Angeles", None),
        ("+5:3", None),
        ("+1é:0", None),
        ("+0é00", None),
    ] {
        assert_eq!(fixed_zone_offset(id), want, "{id}");
    }
}

fn fmt(pattern: &str, micros: i64, id: &str, offset: i32) -> String {
    DateTimePattern::new(pattern, Purpose::Format)
        .unwrap()
        .format_timestamp(
            micros,
            Zone {
                id,
                rules: &FixedOffset(offset),
            },
        )
        .unwrap()
}

#[test]
fn offsets_with_seconds_and_names() {
    let off = 19_815; // +05:30:15
    assert_eq!(fmt("XXXXX", 0, "+05:30:15", off), "+05:30:15");
    assert_eq!(fmt("XXXX", 0, "+05:30:15", off), "+053015");
    assert_eq!(fmt("OOOO", 0, "+05:30:15", off), "GMT+05:30:15");
    assert_eq!(fmt("O", 0, "+05:30:15", off), "GMT+5:30:15");
    assert_eq!(fmt("O", 0, "-05:00", -18_000), "GMT-5");
    assert_eq!(fmt("z", 0, "+05:30:15", off), "+05:30:15");
    assert_eq!(fmt("zzzz", 0, "GMT", 0), "Greenwich Mean Time");
    assert_eq!(fmt("z", 0, "Z", 0), "Z");
    assert_eq!(fmt("VV", 0, "Z", 0), "Z");
}

#[test]
fn dates_reject_time_fields() {
    for p in ["SSS", "a", "hh"] {
        let p = DateTimePattern::new(p, Purpose::Format).unwrap();
        assert!(p.format_date(0).is_err());
    }
    let p = DateTimePattern::new("yyyy[ a][ SSS]", Purpose::Format).unwrap();
    assert!(p.has_time() && !p.has_zone());
    assert_eq!(p.format_date(0).unwrap(), "1970");
    let z = DateTimePattern::new("yyyy[ XXX]", Purpose::Format).unwrap();
    assert!(z.has_zone() && !z.has_time());
}

fn parse(pattern: &str, s: &str) -> Option<i64> {
    DateTimePattern::new(pattern, Purpose::Parse)
        .unwrap()
        .parse_timestamp(s, &FixedOffset(0))
        .unwrap()
}

#[test]
fn parse_edge_cases() {
    const H: i64 = 3_600_000_000;
    // Localized offsets with seconds; a full offset needs `HH:MM`.
    assert_eq!(parse("O", "GMT+5:30:15"), Some(-19_815_000_000));
    assert_eq!(parse("O", "GMT+10"), Some(-10 * H));
    assert_eq!(parse("OOOO", "GMT+05:30:15"), Some(-19_815_000_000));
    assert_eq!(parse("OOOO", "GMT+0530"), None);
    assert_eq!(parse("OOOO", "GMT+24:00"), None);
    // java.time (and so Spark 4.2: CANNOT_PARSE_TIMESTAMP) rejects offsets
    // beyond +-18:00; the bound itself is fine.
    assert_eq!(parse("O", "GMT+20"), None);
    assert_eq!(parse("OOOO", "GMT+19:00"), None);
    assert_eq!(parse("OOOO", "GMT+18:01"), None);
    assert_eq!(parse("OOOO", "GMT+18:00"), Some(-18 * H));
    assert_eq!(parse("O", "GMT+18"), Some(-18 * H));
    // Zone ids: GMT0, UT prefixes; zone names.
    assert_eq!(parse("VV", "GMT0"), Some(0));
    assert_eq!(parse("VV", "GMT+01:00"), Some(-H));
    assert_eq!(parse("VV", "UT-01:00"), Some(H));
    assert_eq!(parse("VV", "UTC"), Some(0));
    assert_eq!(parse("VV", "+1"), None);
    assert_eq!(parse("z", "Greenwich Mean Time"), Some(0));
    // A field or zone read twice must agree.
    assert_eq!(parse("HH HH", "01 01"), Some(H));
    assert_eq!(parse("HH HH", "01 02"), None);
    assert_eq!(parse("XXX XXX", "+01:00 +02:00"), None);
    // Hours that contradict AM/PM.
    assert_eq!(parse("HH a", "13 AM"), None);
    assert_eq!(parse("HH a", "13 PM"), Some(13 * H));
    assert_eq!(parse("HH K", "13 2"), None);
    assert_eq!(parse("HH K", "13 1"), Some(13 * H));
    // Full era names; a reduced year after a variable-width field.
    assert_eq!(parse("GGGG y", "Before Christ 1"), parse("yyyy", "0000"));
    assert_eq!(parse("Dyy", "120"), parse("yyyy", "2020"));
    // Quoted literals with doubled quotes.
    assert_eq!(parse("'it''s' yyyy", "it's 2020"), parse("yyyy", "2020"));
}

#[test]
fn legacy_formatter_decides_the_error_class() {
    let class = |p: &str| {
        DateTimePattern::new(p, Purpose::Format)
            .unwrap_err()
            .error_class()
            .name()
    };
    assert_eq!(
        class("'a''b' yyyyyyy"),
        "INCONSISTENT_BEHAVIOR_CROSS_VERSION.DATETIME_PATTERN_RECOGNITION"
    );
    assert_eq!(
        class("'unterminated VV"),
        "INVALID_DATETIME_PATTERN.WITH_SUGGESTION"
    );
    assert_eq!(class("QQQQQ"), "INVALID_DATETIME_PATTERN.LENGTH");
    assert_eq!(
        class("e"),
        "INCONSISTENT_BEHAVIOR_CROSS_VERSION.DATETIME_WEEK_BASED_PATTERN"
    );
    assert_eq!(class("A"), "INVALID_DATETIME_PATTERN.ILLEGAL_CHARACTER");
    assert_eq!(
        class("yyyy-MM ]"),
        "INCONSISTENT_BEHAVIOR_CROSS_VERSION.DATETIME_PATTERN_RECOGNITION"
    );
}

proptest::proptest! {
    // No failure persistence: it needs the working directory (not under Miri).
    #![proptest_config(proptest::prelude::ProptestConfig {
        failure_persistence: None,
        cases: if cfg!(miri) { 4 } else { 256 },
        ..proptest::prelude::ProptestConfig::default()
    })]
    /// Untrusted strings never panic, whatever the pattern.
    #[test]
    fn parsing_never_panics(
        s in "[0-9a-zA-Z+:.\\- 'éÉ/_]{0,24}",
        p in proptest::sample::select(vec![
            "yyyy-MM-dd'T'HH:mm:ss.SSSXXX", "yyyyMMdd", "Dyy", "G y", "hh a", "O", "OOOO",
            "VV", "z", "XXXXX", "x", "yyyy[-MM[-dd]]", "MMMM d", "y", "kk:mm",
        ]),
    ) {
        let compiled = DateTimePattern::new(p, Purpose::Parse).unwrap();
        let _ = compiled.parse_timestamp(&s, &FixedOffset(3600));
        let _ = compiled.parse_local(&s, false);
        let _ = compiled.parse_date(&s);
        let _ = fixed_zone_offset(&s);
        let _ = DateTimePattern::new(&s, Purpose::Format);
    }
}

#[test]
fn optional_section_keeps_real_errors() {
    // A zone name that is not known must surface instead of the section
    // being dropped.
    let p = DateTimePattern::new("yyyy[ z]", Purpose::Format).unwrap();
    let la = Zone {
        id: "Mars/Olympus_Mons",
        rules: &FixedOffset(-28_800),
    };
    let e = p.format_timestamp(0, la).unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::KrishivUnsupportedFeature);
    // A field the value lacks still skips the section.
    assert_eq!(p.format_date(0).unwrap(), "1970");
    assert_eq!(p.format_local(0).unwrap(), "1970");
    assert_eq!(p.format_timestamp(0, utc()).unwrap(), "1970 UTC");
}

#[test]
fn every_date_formats() {
    // Past year 294247 the microsecond form overflows i64.
    let p = DateTimePattern::new("yyyy-MM-dd", Purpose::Format).unwrap();
    assert_eq!(p.format_date(i32::MAX).unwrap(), "+5881580-07-11");
    assert_eq!(p.format_date(i32::MIN).unwrap(), "-5877641-06-23");
    assert_eq!(p.format_date(106_751_992).unwrap(), "+294247-01-11");
}

#[test]
fn long_adjacent_numbers_do_not_overflow() {
    // 49 digits reserved for the first field would overflow an i128.
    let pattern = format!("y{}", "ddHHmmss".repeat(6));
    let p = DateTimePattern::new(&pattern, Purpose::Parse).unwrap();
    assert_eq!(p.parse_local(&"1".repeat(60), false).unwrap(), None);
}

#[test]
fn region_zone_names_follow_daylight_time() {
    let short = DateTimePattern::new("z", Purpose::Format).unwrap();
    let full = DateTimePattern::new("zzzz", Purpose::Format).unwrap();
    let berlin = |micros: i64, offset: i32| {
        let zone = Zone {
            id: "Europe/Berlin",
            rules: &FixedOffset(offset),
        };
        (
            short.format_timestamp(micros, zone).unwrap(),
            full.format_timestamp(micros, zone).unwrap(),
        )
    };
    // 2024-01-15 and 2024-07-15, noon UTC.
    assert_eq!(
        berlin(1_705_320_000_000_000, 3600),
        ("CET".into(), "Central European Standard Time".into())
    );
    assert_eq!(
        berlin(1_721_044_800_000_000, 7200),
        ("CEST".into(), "Central European Summer Time".into())
    );
}

#[test]
fn region_zone_names_use_historical_and_out_of_range_daylight_time() {
    let full = DateTimePattern::new("zzzz", Purpose::Format).unwrap();
    let name = |id, micros: i64, offset: i32| {
        let zone = Zone {
            id,
            rules: &FixedOffset(offset),
        };
        full.format_timestamp(micros, zone).unwrap()
    };
    // Shanghai observed daylight time in 1990 only; Berlin past year 9999 repeats its rules.
    assert_eq!(
        name("Asia/Shanghai", 648_043_200_000_000, 32400),
        "China Daylight Time"
    );
    assert_eq!(
        name("Europe/Berlin", 1_010_786_126_400_000_000, 7200),
        "Central European Summer Time"
    );
    // Java calls Windhoek's +02:00 daylight time from March 1990 (1991-07-15T12:00Z), standard before.
    assert_eq!(
        name("Africa/Windhoek", 679_579_200_000_000, 7200),
        "Western African Time"
    );
    assert_eq!(
        name("Africa/Windhoek", 15_000_000_000_000, 7200),
        "Central Africa Time"
    );
}

fn fmt_date(pattern: &str, days: i32) -> String {
    DateTimePattern::new(pattern, Purpose::Format)
        .unwrap()
        .format_date(days)
        .unwrap()
}

fn fmt_micros(pattern: &str, micros: i64) -> String {
    DateTimePattern::new(pattern, Purpose::Format)
        .unwrap()
        .format_timestamp(micros, utc())
        .unwrap()
}

/// Day of year and day of week count from the January 1 of the date's year.
#[test]
fn day_of_year_and_week_follow_the_calendar() {
    // 2020-01-01 is a Wednesday, 2020-12-31 a Thursday (day 366), 1969-12-31 a Wednesday.
    assert_eq!(fmt_date("yyyy DDD EEE", 18_262), "2020 001 Wed");
    assert_eq!(fmt_date("yyyy DDD EEE", 18_627), "2020 366 Thu");
    assert_eq!(fmt_date("yyyy DDD EEE", -1), "1969 365 Wed");
    assert_eq!(fmt_date("yyyy DDD EEE", 0), "1970 001 Thu");
}

/// Quarter and `F`, Spark's day of the week in the month (`(day - 1) % 7 + 1`).
#[test]
fn quarter_and_day_of_week_in_month() {
    // 2020-01-01 = day 18262.
    for (days, quarter) in [(18_262, 1), (18_262 + 59, 1), (18_262 + 91, 2), (18_627, 4)] {
        assert_eq!(fmt_date("Q", days), quarter.to_string(), "{days}");
    }
    for (day, f) in [
        (1, 1),
        (7, 7),
        (8, 1),
        (14, 7),
        (15, 1),
        (28, 7),
        (29, 1),
        (31, 3),
    ] {
        assert_eq!(fmt_date("F", 18_262 + day - 1), f.to_string(), "day {day}");
    }
}

/// With an era field, the year counts back from 1 for year 0 and below.
#[test]
fn year_of_era_for_years_before_one() {
    let ad1 = -719_162; // 0001-01-01
    let year = |days: i32| {
        fmt_date("yyyy G", days)
            .split(' ')
            .next()
            .unwrap()
            .to_owned()
    };
    assert_eq!(year(ad1), "0001");
    assert_eq!(year(ad1 - 1), "0001"); // 0000-12-31 is 1 BC
    assert_eq!(year(ad1 - 366), "0001"); // 0000-01-01
    assert_eq!(year(ad1 - 731), "0002"); // -0001-01-01 is 2 BC
    assert_eq!(fmt_date("yyyy", ad1 - 1), "0000");
}

/// Clock hours, am/pm hours and the sub-hour fields.
#[test]
fn hour_variants_and_sub_second_fields() {
    let at = |h: i64, m: i64, s: i64, us: i64| ((h * 60 + m) * 60 + s) * 1_000_000 + us;
    let p = "HH:mm:ss.SSSSSS k h K";
    assert_eq!(
        fmt_micros(p, at(13, 14, 15, 123_456)),
        "13:14:15.123456 13 1 1"
    );
    assert_eq!(fmt_micros(p, at(0, 5, 6, 7)), "00:05:06.000007 24 12 0");
    assert_eq!(fmt_micros(p, at(12, 0, 0, 0)), "12:00:00.000000 12 12 0");
    assert_eq!(
        fmt_micros(p, at(23, 59, 58, 999_999)),
        "23:59:58.999999 23 11 11"
    );
    assert_eq!(fmt_micros(p, at(1, 0, 0, 0)), "01:00:00.000000 1 1 1");
}

/// A zone name reads the saving in force at the instant's own time of day
/// (Berlin springs forward at 01:00 UTC on 2024-03-31).
#[test]
fn zone_name_follows_the_time_of_day_across_a_transition() {
    let short = DateTimePattern::new("z", Purpose::Format).unwrap();
    let at = |secs: i64, offset: i32| {
        let zone = Zone {
            id: "Europe/Berlin",
            rules: &FixedOffset(offset),
        };
        short.format_timestamp(secs * 1_000_000, zone).unwrap()
    };
    assert_eq!(at(1_711_845_000, 3600), "CET"); // 00:30 UTC
    assert_eq!(at(1_711_848_600, 7200), "CEST"); // 01:30 UTC
}
