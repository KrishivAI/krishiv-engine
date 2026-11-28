// SPDX-License-Identifier: Apache-2.0
//! CSV file reads of dates and timestamps under the default parser policy and
//! `spark.sql.legacy.timeParserPolicy = LEGACY`, against the Spark 4.2.1 and
//! 4.3.0 runs of `tests/scripts/probe_time_text_profiles.py` (the `csv_file`
//! section of `tests/fixtures/spark-<v>/time-text-profiles.json`). Each case
//! is read three ways: `inferSchema` (under LEGACY never a `DATE`, and a
//! `TIMESTAMP` only under the `FastDateFormat` pattern), schema `t DATE` and
//! schema `t TIMESTAMP` (LEGACY's lenient `SimpleDateFormat` and fallback).
//! The `json_read` section reads one-record JSON files with the same two
//! schemas: JSON has no strict default date pattern (`JSONOptions`), so with
//! the fallback off and no `dateFormat` it reads `2024-01-02 03:04:05` as a
//! date, where CSV under `preferDate` reads NULL.
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use futures::executor::block_on;
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

fn texts(batches: &[RecordBatch]) -> Vec<Option<String>> {
    let mut out = Vec::new();
    for b in batches {
        let c = b.column(0);
        for i in 0..b.num_rows() {
            out.push((!c.is_null(i)).then(|| c.as_string_view().value(i).to_owned()));
        }
    }
    out
}

fn sql(s: &Session, text: &str) -> Vec<Option<String>> {
    let df = block_on(s.sql(text)).unwrap_or_else(|e| panic!("{text}: {e}"));
    texts(&block_on(df.collect()).unwrap_or_else(|e| panic!("{text}: {e}")))
}

/// A value whose leading date the case's text does not hold (a time-only
/// text read as a `TIMESTAMP` takes the day of the run) with that date
/// replaced, so runs on other days compare.
fn undated(value: Option<String>, text: &str) -> Option<String> {
    let v = value?;
    let dated = v.len() >= 10
        && v.bytes().take(10).enumerate().all(|(i, b)| {
            if i == 4 || i == 7 {
                b == b'-'
            } else {
                b.is_ascii_digit()
            }
        });
    Some(if dated && text.len() < 10 && text.contains(':') {
        format!("<today>{}", &v[10..])
    } else {
        v
    })
}

#[test]
fn csv_file_reads_follow_the_parser_policy_as_spark_runs() {
    reads_match_spark("csv", "csv_file", |text| format!("t\n{text}\n"));
}

#[test]
fn json_file_date_reads_follow_the_parser_policy_as_spark_runs() {
    reads_match_spark("json", "json_read", |text| {
        format!("{}\n", serde_json::json!({ "t": text }))
    });
}

/// Every case of `section` written by `contents` as a `format` file and read
/// as the case's `reads` name, against Spark's schema and values.
fn reads_match_spark(format: &str, section: &str, contents: fn(&str) -> String) {
    let dir = tempfile::tempdir().unwrap();
    for (profile, version) in [(SparkProfile::V4_2, "4.2.1"), (SparkProfile::V4_3, "4.3.0")] {
        let path = format!(
            "{}/../../tests/fixtures/spark-{version}/time-text-profiles.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let run: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(run["spark"], version);
        let sections = run[section].as_object().unwrap();
        assert_eq!(sections.len(), 2, "{version}: default and legacy");
        for (setting, cases) in sections {
            let s = Session::builder().profile(profile).build().unwrap();
            block_on(s.sql("SET spark.sql.session.timeZone = UTC")).unwrap();
            match setting.as_str() {
                "default" => {}
                "legacy" => {
                    block_on(s.sql("SET spark.sql.legacy.timeParserPolicy = LEGACY")).unwrap();
                }
                other => panic!("unknown setting {other}"),
            }
            for (i, case) in cases.as_array().unwrap().iter().enumerate() {
                let text = case["text"].as_str().unwrap();
                let opts = case["options"].as_object().unwrap();
                let file = dir.path().join(format!("{version}-{setting}-{i}.{format}"));
                std::fs::write(&file, contents(text)).unwrap();
                for (read, want) in case["reads"].as_object().unwrap() {
                    let at = format!("{format} {version} {setting} {text:?} {opts:?} {read}");
                    let mut reader = s.read().format(format);
                    if format == "csv" {
                        reader = reader.option("header", "true");
                    }
                    for (k, v) in opts {
                        reader = reader.option(k, v.as_str().unwrap());
                    }
                    reader = match read.as_str() {
                        "infer" => reader.option("inferSchema", "true"),
                        "date" => reader.schema_ddl("t DATE").unwrap(),
                        "timestamp" => reader.schema_ddl("t TIMESTAMP").unwrap(),
                        other => panic!("unknown read {other}"),
                    };
                    let df = reader.load([file.to_str().unwrap()]);
                    block_on(df.create_or_replace_temp_view("v"))
                        .unwrap_or_else(|e| panic!("{at}: {e}"));
                    let got = sql(&s, "SELECT typeof(t) FROM v LIMIT 1");
                    let want_schema = want["schema"].as_str().unwrap();
                    assert_eq!(
                        format!("struct<t:{}>", got[0].as_deref().unwrap_or("")),
                        want_schema,
                        "{at}: schema"
                    );
                    let want_values: Vec<Option<String>> = want["values"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| undated(v.as_str().map(str::to_owned), text))
                        .collect();
                    let got_values: Vec<Option<String>> =
                        sql(&s, "SELECT CAST(t AS STRING) FROM v")
                            .into_iter()
                            .map(|v| undated(v, text))
                            .collect();
                    assert_eq!(got_values, want_values, "{at}: values");
                }
            }
        }
    }
}

/// The `csv_expr` section: the `csv_file` cases as `schema_of_csv(text,
/// options)` and `from_csv(text, 't DATE' | 't TIMESTAMP', options).t` cast to
/// STRING, under the default policy and LEGACY (where `CSVInferSchema` tries
/// only the `FastDateFormat` timestamp pattern, so a date infers as STRING).
#[test]
fn csv_expressions_follow_the_parser_policy_as_spark_runs() {
    fn lit(s: &str) -> String {
        format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
    }
    let mut wrong = Vec::new();
    for (profile, version) in [(SparkProfile::V4_2, "4.2.1"), (SparkProfile::V4_3, "4.3.0")] {
        let path = format!(
            "{}/../../tests/fixtures/spark-{version}/time-text-profiles.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let run: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(run["spark"], version);
        let sections = run["csv_expr"].as_object().unwrap();
        assert_eq!(sections.len(), 2, "{version}: default and legacy");
        for (setting, cases) in sections {
            let s = Session::builder().profile(profile).build().unwrap();
            block_on(s.sql("SET spark.sql.session.timeZone = UTC")).unwrap();
            match setting.as_str() {
                "default" => {}
                "legacy" => {
                    block_on(s.sql("SET spark.sql.legacy.timeParserPolicy = LEGACY")).unwrap();
                }
                other => panic!("unknown setting {other}"),
            }
            for case in cases.as_array().unwrap() {
                let text = case["text"].as_str().unwrap();
                let opts = case["options"].as_object().unwrap();
                let pairs: Vec<String> = opts
                    .iter()
                    .map(|(k, v)| format!("{}, {}", lit(k), lit(v.as_str().unwrap())))
                    .collect();
                let omap = if pairs.is_empty() {
                    String::new()
                } else {
                    format!(", map({})", pairs.join(", "))
                };
                for (name, want) in case["exprs"].as_object().unwrap() {
                    let expr = match name.as_str() {
                        "schema_of_csv" => format!("schema_of_csv({}{omap})", lit(text)),
                        "date" => {
                            format!("CAST(from_csv({}, 't DATE'{omap}).t AS STRING)", lit(text))
                        }
                        "timestamp" => format!(
                            "CAST(from_csv({}, 't TIMESTAMP'{omap}).t AS STRING)",
                            lit(text)
                        ),
                        other => panic!("unknown expression {other}"),
                    };
                    let at = format!("{version} {setting} {text:?} {opts:?} {name}");
                    let got = block_on(async {
                        let df = s.sql(&format!("SELECT {expr}")).await?;
                        df.collect().await
                    });
                    match (want.get("value"), got) {
                        (Some(v), Ok(batches)) => {
                            let got = texts(&batches)
                                .pop()
                                .flatten()
                                .map(|g| undated(Some(g), text));
                            let want = v.as_str().map(|w| undated(Some(w.to_owned()), text));
                            if got != want {
                                wrong.push(format!("{at}: Spark {want:?}, Krishiv {got:?}"));
                            }
                        }
                        (Some(v), Err(e)) => {
                            wrong.push(format!("{at}: Spark {v}, Krishiv fails: {e}"))
                        }
                        (None, Ok(_)) => {
                            wrong.push(format!("{at}: Spark fails ({want}), Krishiv runs"))
                        }
                        (None, Err(_)) => {}
                    }
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} cases differ:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}
