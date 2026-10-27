// SPDX-License-Identifier: Apache-2.0
//! `CAST(string AS TIMESTAMP[_NTZ])` against Spark 4.2 on seeded random
//! strings under three session time zones
//! (`tests/fixtures/spark-4.2.0/cast-timestamp-differential.json`, generated
//! by `tests/scripts/gen_cast_goldens.py`). Strings that name a time zone
//! region resolve against the bundled database and compare as values; an
//! unresolvable one is malformed input (`CAST_INVALID_INPUT`, NULL without
//! ANSI).
#![forbid(unsafe_code)]

use arrow_array::cast::AsArray;
use arrow_array::types::TimestampMicrosecondType;
use arrow_array::{Array, StringArray};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::error::ErrorClass;
use krishiv_kernels::cast::{CastMode, CastOptions, cast};
use krishiv_testkit::kernel_eval::JiffZone;
use serde_json::Value;

type R = Result<(), Box<dyn std::error::Error>>;

/// Whether the string starts with a year the adapter cannot place: outside
/// 1..=9998 (its instants stop at the end of 9999 UTC).
fn out_of_range_year(text: &str) -> bool {
    let t = text.trim();
    let digits: String = t
        .trim_start_matches(['+', '-'])
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    t.starts_with(['+', '-']) || !digits.parse::<u32>().is_ok_and(|y| (1..=9998).contains(&y))
}

#[test]
fn string_to_timestamp_matches_spark() -> R {
    let path =
        krishiv_testkit::tests_dir().join("fixtures/spark-4.2.0/cast-timestamp-differential.json");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let cases = fixture["cases"].as_array().ok_or("cases")?;
    let zones = fixture["zones"].as_object().ok_or("zones")?;
    let ltz = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
    let ntz = DataType::Timestamp(TimeUnit::Microsecond, None);
    let (mut failures, mut regions, mut checked) = (Vec::new(), 0_usize, 0_usize);
    for case in cases {
        let text = case["s"].as_str().ok_or("s")?;
        let input = StringArray::from(vec![text]);
        let mut runs: Vec<(String, DataType, JiffZone)> = zones
            .iter()
            .map(|(key, name)| {
                let zone = JiffZone::new(name.as_str().unwrap_or("UTC")).ok_or("zone")?;
                Ok((key.clone(), ltz.clone(), zone))
            })
            .collect::<Result<_, Box<dyn std::error::Error>>>()?;
        runs.push((
            "ntz".to_owned(),
            ntz.clone(),
            JiffZone::new("UTC").ok_or("UTC")?,
        ));
        for (key, to, zone) in runs {
            // The tz database adapter (`jiff`) stops at the end of year 9999; Spark's
            // `java.time` extends the last rule indefinitely.
            if key != "utc" && key != "ntz" && out_of_range_year(text) {
                continue;
            }
            let want = case[key.as_str()].as_i64();
            let run = |mode| {
                cast(
                    &input,
                    &to,
                    &CastOptions {
                        mode,
                        time_zone: &zone,
                    },
                )
            };
            let got = match run(CastMode::Try) {
                Ok(a) => {
                    let a = a.as_primitive::<TimestampMicrosecondType>();
                    a.is_valid(0).then(|| a.value(0))
                }
                Err(e) if e.error_class() == ErrorClass::KrishivUnsupportedFeature => {
                    regions += 1;
                    assert!(
                        text.chars().any(|c| c.is_ascii_alphabetic() && c != 'T')
                            || text.contains('/'),
                        "unsupported without a zone name: {text:?}"
                    );
                    continue;
                }
                Err(e) => return Err(format!("{text:?}: {e}").into()),
            };
            checked += 1;
            if got != want {
                failures.push(format!("{text:?} [{key}]: got {got:?}, Spark {want:?}"));
            }
            // ANSI raises exactly where try_cast gives NULL.
            let ansi = run(CastMode::Ansi);
            let raised = ansi
                .as_ref()
                .err()
                .map(krishiv_kernels::KernelError::error_class);
            if (got.is_none()) != (raised == Some(ErrorClass::CastInvalidInput)) {
                failures.push(format!(
                    "{text:?} [{key}]: ANSI result {raised:?} vs try {got:?}"
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} differences, first: {:#?}",
        failures.len(),
        &failures[..failures.len().min(30)]
    );
    // Region strings resolve against the bundled database and compare as
    // values above; none may need counting out.
    assert!(
        checked > 5_000 && regions == 0,
        "{checked} checked, {regions} regions"
    );
    Ok(())
}
