// SPDX-License-Identifier: Apache-2.0
//! `schema_of_json` over strings that read as times and timestamps, against
//! the Spark 4.2.1 and 4.3.0 runs of `tests/scripts/probe_time_text_profiles.py`
//! (the `schema_of_json` section of `tests/fixtures/spark-<v>/time-text-profiles.json`):
//! under `inferTimestamp` a string infers `TIME(6)` only on the 4.3 profile with
//! `spark.sql.timeType.enabled`, `TIMESTAMP_NTZ` only with
//! `spark.sql.timestampType = TIMESTAMP_NTZ`, then `TIMESTAMP`, and under
//! `timeParserPolicy=LEGACY` the zone-less grammar decides `TIMESTAMP`.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use futures::executor::block_on;
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

/// The settings of the run, as `SET` statements.
fn settings(name: &str) -> &'static [&'static str] {
    match name {
        "default" => &[],
        "time" => &["SET spark.sql.timeType.enabled = true"],
        "ntz" => &["SET spark.sql.timestampType = TIMESTAMP_NTZ"],
        "time ntz" => &[
            "SET spark.sql.timeType.enabled = true",
            "SET spark.sql.timestampType = TIMESTAMP_NTZ",
        ],
        "legacy" => &["SET spark.sql.legacy.timeParserPolicy = LEGACY"],
        other => panic!("unknown setting {other}"),
    }
}

fn quoted(s: &str) -> String {
    format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
}

fn first_text(batches: &[RecordBatch]) -> Option<String> {
    let b = batches.iter().find(|b| b.num_rows() > 0)?;
    let c = b.column(0);
    (!c.is_null(0)).then(|| c.as_string_view().value(0).to_owned())
}

#[test]
fn schema_of_json_infers_times_as_spark_runs() {
    for (profile, version) in [(SparkProfile::V4_2, "4.2.1"), (SparkProfile::V4_3, "4.3.0")] {
        let path = format!(
            "{}/../../tests/fixtures/spark-{version}/time-text-profiles.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let run: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(run["spark"], version);
        let sections = run["schema_of_json"].as_object().unwrap();
        assert_eq!(sections.len(), 5, "{version}: the five settings");
        for (setting, cases) in sections {
            let s = Session::builder().profile(profile).build().unwrap();
            block_on(s.sql("SET spark.sql.session.timeZone = UTC")).unwrap();
            for stmt in settings(setting) {
                block_on(s.sql(stmt)).unwrap();
            }
            for case in cases.as_array().unwrap() {
                let text = case["text"].as_str().unwrap();
                let opts = case["options"].as_object().unwrap();
                let want = case["schema"].as_str().unwrap();
                let map = if opts.is_empty() {
                    String::new()
                } else {
                    let pairs: Vec<String> = opts
                        .iter()
                        .flat_map(|(k, v)| [quoted(k), quoted(v.as_str().unwrap())])
                        .collect();
                    format!(", map({})", pairs.join(", "))
                };
                let sql = format!("SELECT schema_of_json({}{map})", quoted(text));
                let got = block_on(async {
                    let df = s.sql(&sql).await?;
                    df.collect().await
                })
                .map(|b| first_text(&b));
                let got = got.unwrap_or_else(|e| panic!("{version} {setting} {sql}: {e}"));
                assert_eq!(got.as_deref(), Some(want), "{version} {setting} {sql}");
            }
        }
    }
}
