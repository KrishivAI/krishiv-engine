// SPDX-License-Identifier: Apache-2.0
//! The `FastDateFormat` model (Spark's JSON/CSV/XML options under
//! `timeParserPolicy=LEGACY`) against a Spark 4.2.0 run: parses with the
//! lenient fallback off, and prints.

use krishiv_kernels::datetime::resolve_zone;
use krishiv_kernels::datetime_pattern::{DateTimePattern, Purpose, Zone};

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
        let p = DateTimePattern::new_fast(pattern.unwrap_or(DEFAULT_TS), Purpose::Parse).unwrap();
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
        let p = DateTimePattern::new_fast(pattern.unwrap_or(DEFAULT_DATE), Purpose::Parse).unwrap();
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
        let p = DateTimePattern::new_fast(pattern.unwrap_or(DEFAULT_TS), Purpose::Format).unwrap();
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
        let p =
            DateTimePattern::new_fast(pattern.unwrap_or(DEFAULT_DATE), Purpose::Format).unwrap();
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
