// SPDX-License-Identifier: Apache-2.0
//! The `FastDateFormat` model (Spark's JSON/CSV/XML options under
//! `timeParserPolicy=LEGACY`) against a Spark 4.2.0 run: parses with the
//! lenient fallback off, and prints.

use krishiv_kernels::datetime::resolve_zone;
use krishiv_kernels::datetime_pattern::{DateTimePattern, Zone};

mod fixtures {
    include!("fixtures/fast_date_format_probes.rs");
}

const DEFAULT_TS: &str = "yyyy-MM-dd'T'HH:mm:ss.SSSXXX";
const DEFAULT_DATE: &str = "yyyy-MM-dd";

#[test]
fn fast_timestamp_parses_match_spark() {
    let mut wrong = Vec::new();
    for &(zone, text, pattern, want) in fixtures::PARSE_TS {
        let rules = resolve_zone(zone).unwrap();
        let p = DateTimePattern::new_fast(pattern.unwrap_or(DEFAULT_TS));
        let got = p.parse_timestamp(text, &rules).unwrap();
        if got != want {
            wrong.push(format!(
                "{zone} {text:?} with {pattern:?}: {got:?}, Spark {want:?}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {}:\n{}",
        wrong.len(),
        fixtures::PARSE_TS.len(),
        wrong.join("\n")
    );
}

#[test]
fn fast_date_parses_match_spark() {
    let mut wrong = Vec::new();
    for &(text, pattern, want) in fixtures::PARSE_DATE {
        let p = DateTimePattern::new_fast(pattern.unwrap_or(DEFAULT_DATE));
        let got = p.parse_date(text).unwrap();
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
        fixtures::PARSE_DATE.len(),
        wrong.join("\n")
    );
}

#[test]
fn fast_timestamp_prints_match_spark() {
    let mut wrong = Vec::new();
    for &(zone, micros, pattern, want) in fixtures::FORMAT_TS {
        let rules = resolve_zone(zone).unwrap();
        let p = DateTimePattern::new_fast(pattern.unwrap_or(DEFAULT_TS));
        let got = p
            .format_timestamp(
                micros,
                Zone {
                    id: zone,
                    rules: &rules,
                },
            )
            .map_err(|e| e.to_string());
        if got.as_deref() != Ok(want) {
            wrong.push(format!(
                "{zone} {micros} with {pattern:?}: {got:?}, Spark {want:?}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {}:\n{}",
        wrong.len(),
        fixtures::FORMAT_TS.len(),
        wrong.join("\n")
    );
}

#[test]
fn fast_date_prints_match_spark() {
    let mut wrong = Vec::new();
    for &(days, pattern, want) in fixtures::FORMAT_DATE {
        let p = DateTimePattern::new_fast(pattern.unwrap_or(DEFAULT_DATE));
        let got = p.format_date(days).map_err(|e| e.to_string());
        if got.as_deref() != Ok(want) {
            wrong.push(format!("{days} with {pattern:?}: {got:?}, Spark {want:?}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {}:\n{}",
        wrong.len(),
        fixtures::FORMAT_DATE.len(),
        wrong.join("\n")
    );
}

/// Cases from the diff review, each checked on the Spark 4.2.0 oracle.
#[test]
fn fast_parse_review_regressions() {
    let read = |zone: &str, text: &str, pattern: &str| {
        let rules = resolve_zone(zone).unwrap();
        DateTimePattern::new_fast(pattern)
            .parse_timestamp(text, &rules)
            .unwrap()
    };
    // A wall time a few hours after a spring-forward change keeps the new offset.
    assert_eq!(
        read(
            "America/Los_Angeles",
            "2024-03-10 05:00:00",
            "yyyy-MM-dd HH:mm:ss"
        ),
        Some(1_710_072_000_000_000)
    );
    // `a` before `hh` still adds the afternoon.
    assert_eq!(read("UTC", "PM 05", "a hh"), Some(61_200_000_000));
    // `GMT+05` is not `GMT+h:mm`: `GMT` alone is read.
    assert_eq!(
        read("UTC", "2024-01-01 00:00:00 GMT+05", "yyyy-MM-dd HH:mm:ss z"),
        Some(1_704_067_200_000_000)
    );
    // More than 18 `S` letters cannot be built.
    assert_eq!(
        read(
            "UTC",
            "2024-01-01 00:00:00.5",
            "yyyy-MM-dd HH:mm:ss.SSSSSSSSSSSSSSSSSSS"
        ),
        None
    );
}

/// The second review round, each value read from the Spark 4.2.0 oracle
/// (`enableDateTimeParsingFallback=false`).
#[test]
fn fast_parse_second_review_regressions() {
    let ts = |zone: &str, text: &str, pattern: &str| {
        let rules = resolve_zone(zone).unwrap();
        DateTimePattern::new_fast(pattern)
            .parse_timestamp(text, &rules)
            .unwrap()
    };
    let day =
        |text: &str, pattern: &str| DateTimePattern::new_fast(pattern).parse_date(text).unwrap();
    let la = "America/Los_Angeles";
    // A lenient fraction is added after the wall time picks its offset.
    assert_eq!(
        ts(la, "2024-11-03 00:59:59.2000", "yyyy-MM-dd HH:mm:ss.SSS"),
        Some(1_730_620_801_000_000)
    );
    assert_eq!(
        ts(la, "2024-11-03 00:59:59.0000", "yyyy-MM-dd HH:mm:ss.SSS"),
        Some(1_730_620_799_000_000)
    );
    // Month 13 carries into the next year for week-based patterns too.
    assert_eq!(
        ts(la, "2024-13-1-1", "yyyy-MM-W-u"),
        Some(1_735_545_600_000_000)
    );
    // A date past the microsecond range still reads as a day.
    assert_eq!(day("999999-01-01", "yyyy-MM-dd"), Some(364_522_607));
    // The cutover calendar follows the requested day.
    assert_eq!(day("1582-10-15", "yyyy-MM-dd"), Some(-141_427));
    assert_eq!(day("1582-10-05", "yyyy-MM-dd"), Some(-141_427));
    assert_eq!(day("1582-10-31", "yyyy-MM-dd"), Some(-141_411));
    assert_eq!(day("1582-300", "yyyy-DDD"), Some(-141_405));
    assert_eq!(
        ts(la, "1582-10-15 01:02:03", "yyyy-MM-dd HH:mm:ss"),
        Some(-12_219_260_699_000_000)
    );
}
