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
    for &(text, pattern, want) in fixtures::PROBES {
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
