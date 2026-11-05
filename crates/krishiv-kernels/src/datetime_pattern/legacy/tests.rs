// SPDX-License-Identifier: Apache-2.0
//! Tests of the `SimpleDateFormat` model (`super`).

use super::*;
use crate::cast::FixedOffset;

fn fmt(p: &str, local: i64) -> String {
    LegacyPattern::new(p).unwrap().format_local(local).unwrap()
}

fn micros(y: i64, mo: u32, d: u32, h: i64, mi: i64, s: i64, ms: i64) -> i64 {
    let days = days_from_civil(y, mo, d).unwrap();
    days * MICROS_PER_DAY + ((h * 60 + mi) * 60 + s) * MICROS_PER_SECOND + ms * 1000
}

#[test]
fn a_date_beyond_the_microsecond_range_formats_from_day_counts() {
    let p = LegacyPattern::new("yyyy-MM-dd").unwrap();
    // 9999-12-31 and a day far past i64 microseconds (2^63 / 86.4e9 days is ~292,000 years).
    assert_eq!(p.format_date(2_932_896).unwrap(), "9999-12-31");
    assert!(p.format_date(i32::MAX).is_ok());
}

fn local(p: &str, text: &str) -> Option<i64> {
    LegacyPattern::new(p)
        .unwrap()
        .parse_local(text, true)
        .unwrap()
}

// Spark 4.2.0 under LEGACY (probe 2026-11-02).
#[test]
fn a_day_of_year_must_agree_and_a_lone_weekday_is_the_first_of_the_month() {
    // Spark 4.2.0 under LEGACY (probe 2026-11-02).
    let d = |y, m, dd| days_from_civil(y, m, dd).unwrap() * MICROS_PER_DAY;
    assert_eq!(local("yyyy-DDD-MM-dd", "2024-002-01-01"), None);
    assert_eq!(
        local("yyyy-DDD-MM-dd", "2024-001-01-01"),
        Some(d(2024, 1, 1))
    );
    assert_eq!(local("yyyy E", "2024 Tue"), Some(d(2024, 1, 2)));
    assert_eq!(local("yyyy E", "2024 Mon"), Some(d(2024, 1, 1)));
    assert_eq!(local("yyyy EEEE", "2024 Tuesday"), Some(d(2024, 1, 2)));
    assert_eq!(local("yyyy-MM E", "2024-03 Tue"), Some(d(2024, 3, 5)));
    assert_eq!(local("yyyy-MM E", "2024-03 Fri"), Some(d(2024, 3, 1)));
    assert_eq!(local("yyyy-DDD", "2024-060"), Some(d(2024, 2, 29)));
}

// Spark 4.2.0 under LEGACY (probe 2026-11-02).
// Spark 4.2.0 under LEGACY (probe 2026-11-02).
#[test]
fn week_numbers_follow_the_hybrid_calendar() {
    let f = |y, m, d| {
        fmt(
            "yyyy-MM-dd E w W",
            days_from_civil(y, m, d).unwrap() * MICROS_PER_DAY,
        )
    };
    assert_eq!(f(1582, 5, 31), "1582-05-31 Thu 22 5");
    assert_eq!(f(1582, 1, 1), "1582-01-01 Mon 1 1");
    assert_eq!(f(1582, 10, 4), "1582-10-04 Thu 40 1");
    assert_eq!(f(1582, 10, 15), "1582-10-15 Fri 40 1");
    assert_eq!(f(1582, 10, 20), "1582-10-20 Wed 41 2");
    assert!(f(1500, 3, 15).ends_with(" 12 3"));
}

// Spark 4.2.0 under LEGACY, UTC (probe 2026-11-03, Codex on #417).
#[test]
fn a_bc_week_year_prints_signed_and_a_year_field_keeps_its_run_length() {
    let f = |p, y, m, d| fmt(p, days_from_civil(y, m, d).unwrap() * MICROS_PER_DAY);
    assert_eq!(f("y yyy Y YYY", 1, 1, 1), "1 001 1 001");
    assert_eq!(f("y yyy Y YYY G", -5, 6, 1), "6 006 -5 -005 BC");
    assert_eq!(f("Y YY YYYY yyyy G", -3, 1, 1), "-3 -03 -0003 0004 BC");
    assert_eq!(f("Y YY YYYY yyyy G", -3, 12, 31), "-2 -02 -0002 0004 BC");
    assert_eq!(f("Y YY YYYY yyyy G", -1, 12, 28), "0 00 0000 0002 BC");
    assert_eq!(f("Y YY YYYY yyyy G", 0, 12, 28), "1 01 0001 0001 BC");
    assert_eq!(f("Y YY yyyy", 1000, 12, 31), "1001 01 1000");
}

// Spark 4.2.0 under LEGACY, UTC (probe 2026-11-03, Codex on #417).
#[test]
fn a_week_year_is_checked_as_parsed_against_the_era() {
    let d = |y, m, dd| days_from_civil(y, m, dd).unwrap() * MICROS_PER_DAY;
    for (p, t) in [
        ("YYYY G", "2020 BC"),
        ("YYYY G", "-5 BC"),
        ("YYYY G", "2 BC"),
        ("YYYY G", "1000 BC"),
        ("G YYYY", "BC 5"),
        ("YYYY G ww", "5 BC 20"),
        ("YYYY G u", "2020 BC 2"),
        ("YY G", "20 BC"),
        ("yyyy YYYY G", "2020 2019 BC"),
    ] {
        assert_eq!(local(p, t), None, "{p} {t}");
    }
    assert_eq!(local("YYYY G", "2020 AD"), Some(d(2019, 12, 29)));
    assert_eq!(local("YYYY G", "2021 AD"), Some(d(2020, 12, 27)));
    assert_eq!(local("YYYY", "0"), Some(d(-1, 12, 28)));
    assert_eq!(local("YYYY G", "0 AD"), Some(d(-1, 12, 28)));
    // Not matched yet: Spark reads a signed week year ('-5' with 'YYYY'
    // is -0006-12-27), the numeric fields here take digits only; and
    // `getWeekYear`'s BC year-end arithmetic accepts '0 BC' and refuses
    // '1' with 'YYYY G'/'YYYY', where this check does the opposite.
    // A winning `y` takes the era (`YYYY yyyy G`).
    assert_eq!(local("YYYY yyyy G", "2020 2019 BC"), Some(d(-2018, 1, 1)));
    assert_eq!(local("yyyy G", "2020 BC"), Some(d(-2019, 1, 1)));
}

#[test]
fn an_empty_pattern_parses_nothing_and_a_leap_day_finds_its_century() {
    assert_eq!(local("", ""), None);
    assert_eq!(local("", "2024-01-01"), None);
    let d = days_from_civil(2000, 2, 29).unwrap() * MICROS_PER_DAY;
    assert_eq!(local("yy-MM-dd", "00-02-29"), Some(d));
}

// Spark 4.2.0 under LEGACY (probe 2026-11-02).
#[test]
fn year_zero_and_literal_blanks_follow_simple_date_format() {
    assert_eq!(local("yyyy-MM-dd", "0000-01-01"), None);
    assert_eq!(local("yyyy-MM-dd G", "0000-01-01 BC"), None);
    let p = "yyyy-MM-dd 'at' HH:mm";
    for bad in [
        "2024-01-01at 12:30",
        "2024-01-01  at 12:30",
        "2024-01-01 at12:30",
    ] {
        assert_eq!(local(p, bad), None, "{bad}");
    }
    assert!(local(p, "2024-01-01 at 12:30").is_some());
    assert!(local("yyyy-MM-dd HH:mm", "2024-01-01  12:30").is_some());
    assert_eq!(local("yyyy-MM-dd HH:mm", "2024-01-0112:30"), None);
}

// Codex-reported cases against the hybrid calendar; the weekday and the
// 1582 ordinals agree with the formatter (probed: D=278 on 1582-10-15,
// 355 on 12-31, 'Thu' on 1582-05-31).
#[test]
fn day_of_year_and_weekday_parse_on_the_hybrid_calendar() {
    let d = |m, dd| days_from_civil(1582, m, dd).unwrap() * MICROS_PER_DAY;
    assert_eq!(local("yyyy-DDD", "1582-278"), Some(d(10, 15)));
    assert_eq!(local("yyyy-DDD", "1582-355"), Some(d(12, 31)));
    assert_eq!(local("yyyy-DDD", "1582-356"), None);
    assert_eq!(local("yyyy-DDD", "1582-277"), Some(d(10, 4)));
    assert_eq!(local("yyyy-MM-dd E", "1582-05-31 Thu"), Some(d(5, 31)));
    assert_eq!(local("yyyy-MM-dd E", "1582-05-31 Mon"), None);
}

#[test]
fn week_fields_resolve_on_the_hybrid_calendar() {
    // Spark 4.2.0 under LEGACY: '1582-22-4' (yyyy-ww-u) is May 31.
    let d = days_from_civil(1582, 5, 31).unwrap() * MICROS_PER_DAY;
    assert_eq!(local("yyyy-ww-u", "1582-22-4"), Some(d));
    assert_eq!(
        label_of_real(real_day(days_from_civil(1500, 3, 15).unwrap(), 1500, 3, 15)),
        days_from_civil(1500, 3, 15).unwrap()
    );
}

// Spark 4.2.0 under LEGACY (probe 2026-11-02).
#[test]
fn numeric_zones_and_the_occurrence_number_follow_simple_date_format() {
    let ok = |p: &str, t: &str| LegacyPattern::new(p).unwrap().parse_local(t, true).unwrap();
    assert!(ok("yyyy-MM-dd Z", "2024-01-01 +0530").is_some());
    assert!(ok("yyyy-MM-dd Z", "2024-01-01 GMT+05:30").is_some());
    for bad in ["2024-01-01 +05:30", "2024-01-01 +05"] {
        assert!(ok("yyyy-MM-dd Z", bad).is_none(), "{bad}");
    }
    for bad in ["2024-01-01 GMT+0530", "2024-01-01 +05:30"] {
        assert!(ok("yyyy-MM-dd z", bad).is_none(), "{bad}");
    }
    let f = |m, dd| fmt("F", days_from_civil(1582, m, dd).unwrap() * MICROS_PER_DAY);
    assert_eq!(f(10, 15), "1");
    assert_eq!(f(10, 31), "3");
    assert!(ok("yyyy-MM-dd F", "1582-10-15 1").is_some());
    assert!(ok("yyyy-MM-dd F", "1582-10-15 3").is_none());
}

// Spark 4.2.0 under LEGACY (probe 2026-11-02).
#[test]
fn a_lone_week_year_and_julian_leap_days_resolve_like_spark() {
    let d = |y, m, dd| days_from_civil(y, m, dd).unwrap() * MICROS_PER_DAY;
    assert_eq!(local("YYYY", "2020"), Some(d(2019, 12, 29)));
    assert_eq!(local("YYYY-MM-dd", "2020-03-05"), Some(d(2019, 12, 29)));
    assert_eq!(local("YYYY-MM", "2020-03"), Some(d(2019, 12, 29)));
    assert_eq!(local("YYYY-MM-dd", "2021-01-01"), Some(d(2020, 12, 27)));
    assert_eq!(local("YYYY E", "2020 Tue"), Some(d(2019, 12, 31)));
    assert_eq!(local("YYYY-DDD", "2020-053"), None);
    for (p, t) in [
        ("yyyy-MM-dd", "1500-02-29"),
        ("yyyy-DDD", "1500-060"),
        ("yyyy-DDD", "1500-061"),
    ] {
        assert_eq!(local(p, t), Some(d(1500, 3, 1)), "{t}");
    }
    assert_eq!(local("yyyy-MM-dd", "1700-02-29"), None);
    // Java 21 non-lenient `SimpleDateFormat`: week fields reach the
    // Julian-only leap day too, and its neighbours keep their own labels.
    for (p, t) in [
        ("yyyy-ww-u", "1500-09-6"),
        ("yyyy-MM-W-u", "1500-02-5-6"),
        ("yyyy-MM-F-u", "1500-02-5-6"),
    ] {
        assert_eq!(local(p, t), Some(d(1500, 3, 1)), "{p} {t}");
    }
    assert_eq!(local("yyyy-ww-u", "1500-09-5"), Some(d(1500, 2, 28)));
    assert_eq!(local("yyyy-MM-W-u", "1500-03-1-7"), Some(d(1500, 3, 1)));
    assert_eq!(local("yyyy-MM-F-u", "1500-02-5-5"), None);
}

// Spark 4.2.0 under LEGACY (probe 2026-11-02).
#[test]
fn zone_display_names_parse_short_and_long_and_the_week_year_uses_the_hybrid_day() {
    let p = LegacyPattern::new("yyyy-MM-dd HH:mm z").unwrap();
    let offset = |t: &str| {
        let local = p.parse_fields(t, None)?.0?;
        let utc = p.parse_timestamp(t, &FixedOffset(0)).ok()??;
        i32::try_from((local - utc) / MICROS_PER_SECOND).ok()
    };
    // Spark 4.2.0: 12:00 PDT is 19:00 UTC, whether short or long.
    assert_eq!(offset("2024-07-01 12:00 PDT"), Some(-7 * 3600));
    assert_eq!(
        offset("2024-07-01 12:00 Pacific Daylight Time"),
        Some(-7 * 3600)
    );
    assert_eq!(
        offset("2024-07-01 12:00 Central European Summer Time"),
        Some(2 * 3600)
    );
    // 1581-12-27 is in week-year 1581 on the Julian calendar.
    let t = days_from_civil(1581, 12, 27).unwrap() * MICROS_PER_DAY;
    assert_eq!(fmt("yyyy YYYY w", t), "1581 1581 52");
}

#[test]
fn a_two_digit_year_with_bc_picks_its_century_on_the_year_of_the_era() {
    let mut p = LegacyPattern::new("yy G").unwrap();
    p.today = (2026, 11, 2);
    // Spark 4.2.0: '99 BC' is 1999 BC (astronomical year -1998).
    let want = days_from_civil(-1998, 1, 1).unwrap() * MICROS_PER_DAY;
    assert_eq!(p.parse_local("99 BC", true).unwrap(), Some(want));
}

#[test]
fn the_last_era_field_wins() {
    let d = days_from_civil(1, 1, 1).unwrap() * MICROS_PER_DAY;
    // Spark 4.2.0 under LEGACY: '0001-01-01 BC AD' is year 1 AD.
    assert_eq!(local("yyyy-MM-dd G G", "0001-01-01 BC AD"), Some(d));
}

#[test]
fn oversized_fields_fail_instead_of_overflowing() {
    assert_eq!(local("yyyy-ww-u", "2024-9223372036854775807-1"), None);
    assert_eq!(local("yyyy-MM-W-u", "2024-01-9223372036854775807-1"), None);
    assert_eq!(local("yyyy-MM-F-u", "2024-01-9223372036854775807-1"), None);
}

// Non-lenient `SimpleDateFormat` on Java 21 (2026-11-03): the date
// parses, and `Math.multiplyExact(millis, 1000)` throws.
#[test]
fn a_date_past_the_micros_range_is_an_overflow_not_a_mismatch() {
    let p = LegacyPattern::new("yyyy-MM-dd").unwrap();
    let utc = FixedOffset(0);
    assert!(matches!(
        p.instant("300000-01-01", &utc),
        Err(Unmatched::Overflow)
    ));
    assert!(matches!(
        p.instant("2024-13-01", &utc),
        Err(Unmatched::Text)
    ));
    assert_eq!(p.parse_timestamp("300000-01-01", &utc).unwrap(), None);
}

// Non-lenient `SimpleDateFormat` on Java 21 (2026-11-03): only the day
// of month competes with the day of year; a month is checked after.
#[test]
fn a_month_after_the_day_of_year_does_not_displace_it() {
    let feb29 = Some(days_from_civil(2024, 2, 29).unwrap() * MICROS_PER_DAY);
    assert_eq!(local("yyyy-DDD-MM", "2024-060-02"), feb29);
    assert_eq!(local("yyyy-DDD-MM", "2024-060-03"), None);
    assert_eq!(local("yyyy-MM-DDD", "2024-02-060"), feb29);
    assert_eq!(local("yyyy-DDD-MM-dd", "2024-060-02-29"), feb29);
    assert_eq!(local("yyyy-dd-DDD-MM", "2024-29-060-02"), feb29);
    assert_eq!(local("yyyy-DDD-dd", "2024-060-29"), None);
}

#[test]
fn day_of_year_counts_the_cutover_gap() {
    let f = |days: i64| fmt("D", days * MICROS_PER_DAY);
    let d = |m, dd| days_from_civil(1582, m, dd).unwrap();
    assert_eq!(f(d(10, 4)), "277");
    assert_eq!(f(d(10, 15)), "278");
    assert_eq!(f(d(12, 31)), "355");
    assert_eq!(f(days_from_civil(1583, 1, 1).unwrap()), "1");
}

#[test]
fn every_supplied_hour_field_must_agree() {
    for text in [
        "2024-01-01 23 01 AM",
        "2024-01-01 01 23 AM",
        "2024-01-01 01 13 AM",
        "2024-01-01 25 01 AM",
        "2024-01-01 13 01 AM",
    ] {
        assert_eq!(local("yyyy-MM-dd HH hh a", text), None, "{text}");
    }
    assert!(local("yyyy-MM-dd HH hh a", "2024-01-01 13 01 PM").is_some());
}

#[test]
fn a_zone_is_refused_when_the_caller_allows_none() {
    let p = LegacyPattern::new("yyyy-MM-dd XXX").unwrap();
    assert!(p.parse_local("2024-01-01 +05:30", false).unwrap().is_none());
    assert!(p.parse_local("2024-01-01 +05:30", true).unwrap().is_some());
}

#[test]
fn the_two_digit_year_window_starts_on_a_day() {
    // Spark 4.2.0 under LEGACY with the clock at 2026-11-02 (probe): the
    // window starts on 1946-11-02.
    let mut p = LegacyPattern::new("yy-MM-dd").unwrap();
    p.today = (2026, 11, 2);
    let at = |text: &str| p.parse_local(text, true).unwrap();
    let day = |y: i64, m: u32, d: u32| days_from_civil(y, m, d).unwrap() * MICROS_PER_DAY;
    assert_eq!(at("46-01-01"), Some(day(2046, 1, 1)));
    assert_eq!(at("46-11-01"), Some(day(2046, 11, 1)));
    assert_eq!(at("46-11-02"), Some(day(1946, 11, 2)));
    assert_eq!(at("46-11-03"), Some(day(1946, 11, 3)));
    assert_eq!(at("46-12-31"), Some(day(1946, 12, 31)));
    // A leap day whose provisional century lacks it.
    assert_eq!(at("00-02-29"), Some(day(2000, 2, 29)));
}

#[test]
fn week_fields_resolve_like_gregorian_calendar() {
    let d = |y, m, dd| days_from_civil(y, m, dd).unwrap() * MICROS_PER_DAY;
    assert_eq!(local("yyyy-ww-u", "2024-02-1"), Some(d(2024, 1, 8)));
    assert_eq!(local("yyyy-ww-u", "2024-01-2"), Some(d(2024, 1, 2)));
    assert_eq!(local("yyyy-MM-F", "2024-05-3"), Some(d(2024, 5, 19)));
    // Week 1 of February 2024 starts in January: the month disagrees.
    assert_eq!(local("yyyy-MM-W", "2024-02-1"), None);
}

#[test]
fn one_based_hours_reject_zero_and_the_cutover_gap_is_not_a_date() {
    assert_eq!(local("yyyy-MM-dd kk", "2024-01-01 00"), None);
    assert_eq!(local("yyyy-MM-dd hh a", "2024-01-01 00 AM"), None);
    assert!(local("yyyy-MM-dd kk", "2024-01-01 24").is_some());
    assert!(local("yyyy-MM-dd hh a", "2024-01-01 12 AM").is_some());
    assert_eq!(local("yyyy-MM-dd", "1582-10-10"), None);
    assert!(local("yyyy-MM-dd", "1582-10-04").is_some());
    assert!(local("yyyy-MM-dd", "1582-10-15").is_some());
}

#[test]
fn review_edge_cases() {
    let l = |p, t| local(p, t);
    assert_eq!(l("yyyy-MM-dd E", "2024-01-01 Tue"), None);
    assert!(l("yyyy-MM-dd E", "2024-01-01 Mon").is_some());
    assert_eq!(l("yyyy-MM-dd XXX", "2024-01-01 +25:00"), None);
    assert_eq!(l("yyyy-MM-dd XXX", "2024-01-01 +05:99"), None);
    assert!(l("dMMMyyyy", "12Jan2024").is_some());
    // Week-year 2020 week 1 starts on 2019-12-29 (Sunday); day 1 = Sunday.
    assert!(l("YYYY-ww-u", "2020-01-1").is_some());
}

#[test]
fn uppercase_y_is_the_week_year() {
    let t = micros(2019, 12, 30, 0, 0, 0, 0);
    assert_eq!(fmt("YYYY-ww", t), "2020-01");
    assert_eq!(fmt("yyyy-ww", t), "2019-01");
    assert_eq!(fmt("YYYY-ww", micros(2021, 1, 1, 0, 0, 0, 0)), "2021-01");
}

#[test]
fn iso_offsets_have_a_width_each() {
    let ok = |p: &str, t: &str| local(p, t).is_some();
    assert!(ok("yyyy-MM-dd X", "2024-03-05 +05"));
    assert!(ok("yyyy-MM-dd X", "2024-03-05 +05:30")); // trailing text ignored
    assert!(ok("yyyy-MM-dd X", "2024-03-05 Z"));
    assert!(ok("yyyy-MM-dd XX", "2024-03-05 +0530"));
    assert!(!ok("yyyy-MM-dd XX", "2024-03-05 +05"));
    assert!(ok("yyyy-MM-dd XXX", "2024-03-05 +05:30"));
    assert!(!ok("yyyy-MM-dd XXX", "2024-03-05 +0530"));
}

#[test]
fn daylight_zone_names_are_their_saving_time_offset() {
    let utc = crate::datetime::resolve_zone("UTC").unwrap();
    let p = LegacyPattern::new("yyyy-MM-dd HH:mm z").unwrap();
    let at = |s: &str| p.parse_timestamp(s, &utc).unwrap().unwrap();
    assert_eq!(at("2024-07-01 12:00 PDT"), micros(2024, 7, 1, 19, 0, 0, 0));
    // Spark 4.2.0 under LEGACY: other names are fixed offsets all year.
    for (name, hours) in [
        ("CEST", 2),
        ("CET", 1),
        ("BST", 1),
        ("EEST", 3),
        ("AEST", 10),
        ("JST", 9),
    ] {
        assert_eq!(
            at(&format!("2024-07-01 12:00 {name}")),
            micros(2024, 7, 1, 12 - hours, 0, 0, 0),
            "{name}"
        );
        assert_eq!(
            at(&format!("2024-01-01 12:00 {name}")),
            micros(2024, 1, 1, 12 - hours, 0, 0, 0),
            "{name} in winter"
        );
    }
    assert_eq!(at("2024-07-01 12:00 ACST"), micros(2024, 7, 1, 2, 30, 0, 0));
    assert_eq!(at("2024-01-01 12:00 EDT"), micros(2024, 1, 1, 16, 0, 0, 0));
    assert_eq!(at("2024-07-01 12:00 AKDT"), micros(2024, 7, 1, 20, 0, 0, 0));
}

#[test]
fn local_times_resolve_like_a_non_lenient_calendar() {
    let la = crate::datetime::resolve_zone("America/Los_Angeles").unwrap();
    let p = LegacyPattern::new("yyyy-MM-dd HH:mm:ss").unwrap();
    assert_eq!(p.parse_timestamp("2024-03-10 02:30:00", &la).unwrap(), None);
    let t = p
        .parse_timestamp("2024-11-03 01:30:00", &la)
        .unwrap()
        .unwrap();
    assert_eq!(t, micros(2024, 11, 3, 9, 30, 0, 0));
}

#[test]
fn formats_like_fast_date_format() {
    let t = micros(2018, 11, 17, 5, 33, 33, 123);
    assert_eq!(fmt("G GG GGG GGGG", t), "AD AD AD AD");
    assert_eq!(
        fmt("y yy yyy yyyy yyyyy yyyyyy", t),
        "2018 18 2018 2018 02018 002018"
    );
    assert_eq!(fmt("M MM MMM MMMM", t), "11 11 Nov November");
    assert_eq!(fmt("L LL", t), "11 11");
    assert_eq!(fmt("E EE EEE EEEE", t), "Sat Sat Sat Saturday");
    assert_eq!(fmt("F", t), "3");
    assert_eq!(fmt("D DDD", t), "321 321");
    assert_eq!(fmt("h hh k kk K KK a", t), "5 05 5 05 5 05 AM");
    assert_eq!(
        fmt("S SS SSS SSSS SSSSSSSSS", t),
        "123 123 123 0123 000000123"
    );
    assert_eq!(fmt("[yyyy-MM-dd HH:mm:ss]", t), "[2018-11-17 05:33:33]");
    assert_eq!(
        fmt("姚123'GyYqQMLwWuEFDdhHmsSaVzZxXOV'", t),
        "姚123GyYqQMLwWuEFDdhHmsSaVzZxXOV"
    );
    assert_eq!(fmt("''", t), "'");
    assert_eq!(fmt("", t), "");
    assert_eq!(fmt("u", t), "6");
}

#[test]
fn the_julian_calendar_decides_week_fields_before_1582() {
    // 1582-05-31 is a Thursday in the Julian calendar.
    let t = micros(1582, 5, 31, 19, 40, 35, 123);
    assert_eq!(fmt("E D", t), "Thu 151");
    assert_eq!(fmt("E", micros(2100, 1, 1, 1, 33, 33, 123)), "Fri");
    assert_eq!(fmt("E", micros(1969, 12, 31, 15, 0, 0, 0)), "Wed");
}

#[test]
fn illegal_letters_are_invalid_patterns() {
    for p in [
        "q qq",
        "Q QQQ",
        "VV",
        "XXXX XXXXX",
        "O OOOO",
        "x xx",
        "'open",
    ] {
        assert!(LegacyPattern::new(p).is_err(), "{p}");
    }
}

#[test]
fn offsets_print_as_commons_lang_does() {
    let p = LegacyPattern::new("X XX XXX|Z ZZZZZ").unwrap();
    let z = Zone {
        id: "-08:00",
        rules: &FixedOffset(-8 * 3600),
    };
    assert_eq!(
        p.format_timestamp(0, z).unwrap(),
        "-08 -0800 -08:00|-0800 -0800"
    );
    let utc = Zone {
        id: "UTC",
        rules: &FixedOffset(0),
    };
    assert_eq!(p.format_timestamp(0, utc).unwrap(), "Z Z Z|+0000 +0000");
}

/// `TimeZone.getTimeZone(ZoneId)` names the offset `Z` `UTC` and the
/// unknown id `UT` `GMT` (Java 21 `FastDateFormat`, Codex on #417).
#[test]
fn zero_offset_aliases_take_the_utc_and_gmt_names() {
    let p = LegacyPattern::new("z zzzz").unwrap();
    let show = |id| {
        p.format_timestamp(
            0,
            Zone {
                id,
                rules: &FixedOffset(0),
            },
        )
        .unwrap()
    };
    assert_eq!(show("Z"), "UTC Coordinated Universal Time");
    assert_eq!(show("UT"), "GMT Greenwich Mean Time");
    assert_eq!(show("UTC"), "UTC Coordinated Universal Time");
}

/// A `UT+hh:mm` session zone prints as `GMT+hh:mm` (Spark 4.2.0, LEGACY).
#[test]
fn a_ut_prefixed_offset_prints_as_gmt() {
    let p = LegacyPattern::new("z zzzz").unwrap();
    let z = Zone {
        id: "UT+05:30",
        rules: &FixedOffset(19_800),
    };
    assert_eq!(p.format_timestamp(0, z).unwrap(), "GMT+05:30 GMT+05:30");
}

/// A non-lenient `SimpleDateFormat` refuses `h` outside 1-12 and `k`
/// outside 1-24 as it reads them, whatever field follows.
#[test]
fn an_out_of_range_one_based_hour_fails_even_when_overwritten() {
    assert_eq!(local("yyyy-MM-dd h K", "2024-01-01 13 1"), None);
    assert_eq!(local("yyyy-MM-dd k H", "2024-01-01 25 1"), None);
    assert_eq!(
        local("yyyy-MM-dd k", "2024-01-01 24"),
        Some(micros(2024, 1, 1, 0, 0, 0, 0))
    );
    assert_eq!(
        local("yyyy-MM-dd h K", "2024-01-01 12 1"),
        Some(micros(2024, 1, 1, 1, 0, 0, 0))
    );
}

/// `java.util.TimeZone` drops the transitions before 1900: earlier
/// instants print the zone's raw offset, later ones the database's
/// (Spark 4.2.0, `timeParserPolicy=LEGACY`, `date_format` in the
/// session zone: LA 1800 `-0800`, Kolkata 1850 `+0530` / 1899-12-31
/// `+0530` / 1900-06 `+0521`, Kathmandu 1850 `+0545`, Amsterdam (a link
/// to Brussels) 1850 `+0100` / 1910 `+0000`).
#[test]
fn region_offsets_before_1900_are_the_raw_offset() {
    let p = LegacyPattern::new("yyyy-MM-dd HH:mm:ss Z").unwrap();
    let show = |zone_id: &str, y, mo, d, h| {
        let rules = crate::datetime::resolve_zone(zone_id).unwrap();
        let local = micros(y, mo, d, h, 0, 0, 0);
        let utc = crate::cast::datetime::local_to_utc(local, &rules).unwrap();
        p.format_timestamp(
            utc,
            Zone {
                id: zone_id,
                rules: &rules,
            },
        )
        .unwrap()
    };
    assert_eq!(
        show("America/Los_Angeles", 1800, 1, 1, 0),
        "1800-01-01 00:00:00 -0800"
    );
    assert_eq!(
        show("America/Los_Angeles", 1582, 5, 31, 19),
        "1582-05-31 19:00:00 -0800"
    );
    assert_eq!(
        show("Asia/Kolkata", 1850, 6, 1, 0),
        "1850-06-01 00:00:00 +0530"
    );
    assert_eq!(
        show("Asia/Kolkata", 1899, 12, 31, 23),
        "1899-12-31 23:00:00 +0530"
    );
    assert_eq!(
        show("Asia/Kolkata", 1900, 6, 1, 0),
        "1900-06-01 00:00:00 +0521"
    );
    assert_eq!(
        show("Asia/Kolkata", 1920, 6, 1, 0),
        "1920-06-01 00:00:00 +0530"
    );
    assert_eq!(
        show("Asia/Kathmandu", 1850, 6, 1, 0),
        "1850-06-01 00:00:00 +0545"
    );
    assert_eq!(
        show("Asia/Kathmandu", 1960, 6, 1, 0),
        "1960-06-01 00:00:00 +0530"
    );
    assert_eq!(
        show("Europe/Amsterdam", 1850, 6, 1, 0),
        "1850-06-01 00:00:00 +0100"
    );
    assert_eq!(
        show("Europe/Amsterdam", 1910, 6, 1, 0),
        "1910-06-01 00:00:00 +0000"
    );
}

fn parse(p: &str, s: &str) -> Option<i64> {
    LegacyPattern::new(p).unwrap().parse_local(s, true).unwrap()
}

#[test]
fn parses_like_simple_date_format() {
    let day = |y, m, d| Some(micros(y, m, d, 0, 0, 0, 0));
    assert_eq!(parse("y", "1"), day(1, 1, 1));
    assert_eq!(parse("y", "009999"), day(9999, 1, 1));
    assert_eq!(parse("yy", "00"), day(2000, 1, 1));
    assert_eq!(parse("yy", "99"), day(1999, 1, 1));
    assert_eq!(parse("yyy", "001"), day(1, 1, 1));
    assert_eq!(parse("yyyyyy", "000001"), day(1, 1, 1));
    assert_eq!(parse("D", "9"), day(1970, 1, 9));
    assert_eq!(parse("D", "300"), day(1970, 10, 27));
    assert_eq!(parse("DD", "100"), day(1970, 4, 10));
    assert_eq!(parse("dd-DDD", "31-365"), day(1970, 12, 31));
    assert_eq!(parse("yyyy-DDD", "2020-365"), day(2020, 12, 30));
    assert_eq!(parse("yyyy-MM-DDD", "2020-12-350"), day(2020, 12, 15));
    assert_eq!(parse("yyyy-MM-dd-DDD", "2020-12-31-366"), day(2020, 12, 31));
    assert_eq!(parse("yyyy-MM-dd", "2020-1-2"), day(2020, 1, 2));
    assert_eq!(parse("yyyyMMdd", "20200102"), day(2020, 1, 2));
    assert_eq!(parse("MMM d yyyy", "January 5 2020"), day(2020, 1, 5));
    assert_eq!(parse("dd", "16"), day(1970, 1, 16));
    assert_eq!(parse("yyyy", " 2020"), day(2020, 1, 1));
}

#[test]
fn parsing_is_not_lenient_but_ignores_trailing_text() {
    assert_eq!(parse("MM-dd", "02-29"), None);
    assert_eq!(parse("yyyy-MM-dd", "2020-13-45"), None);
    assert_eq!(parse("yyyy-MM-dd mm", "2020-01-02 75"), None);
    assert_eq!(parse("yyyy", "x2020"), None);
    assert_eq!(parse("y", "-5"), None);
    assert_eq!(parse("yyyy-MM-dd HH:mm:ss", "2020-01-02"), None);
    assert_eq!(
        parse("yyyy-MM-dd HH:mm:ss", "2020-01-02 03:04:05 trailing"),
        Some(micros(2020, 1, 2, 3, 4, 5, 0))
    );
    assert_eq!(parse("dd-DDD", "01-365"), None);
    // Only spaces and tabs are skipped before a field (Spark 4.2.0).
    assert_eq!(parse("yyyy-MM-dd HH", "2024-01-01 \n12"), None);
    assert_eq!(parse("yyyy-MM-dd HH", "2024-01-01 \u{3000}12"), None);
    assert_eq!(parse("yyyy-MM-ddHH", "2024-01-01\n12"), None);
    assert_eq!(
        parse("yyyy-MM-dd HH", "2024-01-01 \t12"),
        Some(micros(2024, 1, 1, 12, 0, 0, 0))
    );
}

#[test]
fn parses_clock_fields() {
    assert_eq!(parse("h a", "5 PM"), Some(micros(1970, 1, 1, 17, 0, 0, 0)));
    assert_eq!(parse("hh a", "12 AM"), Some(micros(1970, 1, 1, 0, 0, 0, 0)));
    assert_eq!(
        parse("yyyy-MM-dd kk:mm:ss", "2020-01-02 24:00:00"),
        Some(micros(2020, 1, 2, 0, 0, 0, 0))
    );
    assert_eq!(
        parse("HH:mm:ss.SSS", "03:04:05.12"),
        Some(micros(1970, 1, 1, 3, 4, 5, 12))
    );
    assert_eq!(
        parse("EEE yyyy-MM-dd", "Mon 2020-01-06"),
        Some(micros(2020, 1, 6, 0, 0, 0, 0))
    );
}

#[test]
fn a_parsed_offset_moves_the_instant() {
    let p = LegacyPattern::new("yyyy-MM-dd'T'HH:mm:ssZ").unwrap();
    assert_eq!(
        p.parse_timestamp("2020-01-02T03:04:05+0530", &FixedOffset(0))
            .unwrap(),
        Some(micros(2020, 1, 1, 21, 34, 5, 0))
    );
}
