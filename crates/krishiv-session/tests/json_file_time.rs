// SPDX-License-Identifier: Apache-2.0
//! JSON file reads of strings that read as times and timestamps, against the
//! Spark 4.2.1 and 4.3.0 runs of `tests/scripts/probe_time_text_profiles.py`
//! (the `json_file` section of `tests/fixtures/spark-<v>/time-text-profiles.json`):
//! the file reader's inference takes the same arms as `schema_of_json`
//! (`TIME(6)` only on the 4.3 profile with `spark.sql.timeType.enabled`,
//! `TIMESTAMP_NTZ` only with `spark.sql.timestampType = TIMESTAMP_NTZ`, then
//! `TIMESTAMP`, and under `timeParserPolicy=LEGACY` the `FastDateFormat`
//! pattern or the zone-less grammar), and the column reads as Spark reads it.
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

fn texts(batches: &[RecordBatch]) -> Vec<Vec<Option<String>>> {
    let mut out = Vec::new();
    for b in batches {
        for i in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|c| (!c.is_null(i)).then(|| c.as_string_view().value(i).to_owned()))
                    .collect(),
            );
        }
    }
    out
}

fn sql(s: &Session, text: &str) -> Vec<Vec<Option<String>>> {
    let df = block_on(s.sql(text)).unwrap_or_else(|e| panic!("{text}: {e}"));
    texts(&block_on(df.collect()).unwrap_or_else(|e| panic!("{text}: {e}")))
}

/// A value whose leading date the case's text does not hold (a time-only
/// text read as a `TIMESTAMP` takes the day of the run) with that date
/// replaced, so runs on other days compare.
fn undated(value: Option<String>, text: &str) -> Option<String> {
    let v = value?;
    let date = v.get(..10).filter(|d| {
        d.bytes().enumerate().all(|(i, b)| {
            if i == 4 || i == 7 {
                b == b'-'
            } else {
                b.is_ascii_digit()
            }
        })
    });
    Some(match date {
        Some(d) if !text.contains(d) => format!("<today>{}", &v[10..]),
        _ => v,
    })
}

#[test]
fn json_file_reads_infer_times_as_spark_runs() {
    let dir = tempfile::tempdir().unwrap();
    for (profile, version) in [(SparkProfile::V4_2, "4.2.1"), (SparkProfile::V4_3, "4.3.0")] {
        let path = format!(
            "{}/../../tests/fixtures/spark-{version}/time-text-profiles.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let run: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(run["spark"], version);
        let sections = run["json_file"].as_object().unwrap();
        assert_eq!(sections.len(), 5, "{version}: the five settings");
        for (setting, cases) in sections {
            let s = Session::builder().profile(profile).build().unwrap();
            block_on(s.sql("SET spark.sql.session.timeZone = UTC")).unwrap();
            for stmt in settings(setting) {
                block_on(s.sql(stmt)).unwrap();
            }
            for (i, case) in cases.as_array().unwrap().iter().enumerate() {
                let text = case["text"].as_str().unwrap();
                let opts = case["options"].as_object().unwrap();
                let want = case["schema"].as_str().unwrap();
                let file = dir
                    .path()
                    .join(format!("{version}-{i}-{}.json", setting.replace(' ', "-")));
                std::fs::write(&file, format!("{text}\n")).unwrap();
                let at = format!("{version} {setting} {text} {opts:?}");
                let mut reader = s.read().format("json");
                for (k, v) in opts {
                    reader = reader.option(k, v.as_str().unwrap());
                }
                let df = reader.load([file.to_str().unwrap()]);
                block_on(df.create_or_replace_temp_view("v"))
                    .unwrap_or_else(|e| panic!("{at}: {e}"));
                let schema = block_on(df.schema()).unwrap_or_else(|e| panic!("{at}: {e}"));
                let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
                let types = names
                    .iter()
                    .map(|n| format!("typeof(`{n}`)"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let types = &sql(&s, &format!("SELECT {types} FROM v LIMIT 1"))[0];
                let got = format!(
                    "struct<{}>",
                    names
                        .iter()
                        .zip(types)
                        .map(|(n, t)| format!("{n}:{}", t.as_deref().unwrap_or("")))
                        .collect::<Vec<_>>()
                        .join(",")
                );
                assert_eq!(got, want, "{at}: schema");
                let want_values: Vec<Option<String>> = case["values"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| undated(v.as_str().map(str::to_owned), text))
                    .collect();
                let got_values: Vec<Option<String>> = sql(&s, "SELECT CAST(t AS STRING) FROM v")
                    .into_iter()
                    .map(|mut r| undated(r.remove(0), text))
                    .collect();
                assert_eq!(got_values, want_values, "{at}: values");
            }
        }
    }
}
