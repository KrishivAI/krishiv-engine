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
