// SPDX-License-Identifier: Apache-2.0
//! JSON decimal inference (`prefersDecimal`, numbers past `Long`, decimal
//! merges) through `schema_of_json` and a JSON file read, against the Spark
//! 4.2.1 and 4.3.0 runs of `tests/scripts/probe_json_prefers_decimal.py`
//! (`tests/fixtures/spark-<v>/json-prefers-decimal.json`): a string reads as
//! `new BigDecimal(s.replaceAll(",", ""))` typed `DECIMAL(precision, scale)`,
//! a number as `DECIMAL(max(p, s), s)` (a negative scale is
//! `NEGATIVE_SCALE_DISALLOWED`), and decimals merge to the wider range plus
//! the wider scale (`BIGINT` as `DECIMAL(20, 0)`, `DOUBLE` past 38 digits).
#![cfg(test)]

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use futures::executor::block_on;
use krishiv_plan::analyzer::SparkProfile;
use krishiv_session::Session;

/// The cases not matched yet, by text and options: a `locale` other than
/// `en-US` parses through Java's `DecimalFormat` in Spark (`1.000,5` is
/// `DECIMAL(5,1)` under `de-DE`); Krishiv applies the US rule. Each must
/// still differ, so a fix removes it from here.
const KNOWN_GAPS: &[&str] = &[r#"{"t":"1.000,5"}"#];

fn quoted(s: &str) -> String {
    format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
}

fn first_text(batches: &[RecordBatch]) -> Option<String> {
    let b = batches.iter().find(|b| b.num_rows() > 0)?;
    let c = b.column(0);
    (!c.is_null(0)).then(|| c.as_string_view().value(0).to_owned())
}

/// The `typeof` of each named column of view `v`, as `struct<n:t,...>`
/// (`struct<>` for none: DROPMALFORMED over only malformed records).
fn view_schema(s: &Session, names: &[&str]) -> String {
    if names.is_empty() {
        return "struct<>".to_owned();
    }
    let types = names
        .iter()
        .map(|n| format!("typeof(`{n}`)"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!("SELECT {types} FROM v LIMIT 1");
    let batches = block_on(async { s.sql(&sql).await?.collect().await }).unwrap();
    let b = batches.iter().find(|b| b.num_rows() > 0).unwrap();
    let fields: Vec<String> = names
        .iter()
        .enumerate()
        .map(|(i, n)| format!("{n}:{}", b.column(i).as_string_view().value(0)))
        .collect();
    format!("struct<{}>", fields.join(","))
}

/// Krishiv's error class for a failure Spark raises as a plain Java
/// exception (no error class): `NumberFormatException` is `new
/// java.math.BigDecimal` refusing `NaN` / `Infinity` or an exponent past
/// `int` under `prefersDecimal`.
fn krishiv_class(spark: &str) -> &str {
    match spark {
        "NumberFormatException" => "KRISHIV_JSON_NUMBER_NOT_DECIMAL",
        class => class,
    }
}

/// `got` is Spark's result `want`: a schema, or `ERROR <class>` matched
/// against the error's class ([`krishiv_class`] for Java exceptions).
fn check(got: &Result<String, String>, want: &str, gap: bool, at: &str) {
    let matches = match (got, want.strip_prefix("ERROR ")) {
        (Err(e), Some(class)) => e.contains(krishiv_class(class)),
        (Ok(g), None) => g == want,
        _ => false,
    };
    if gap {
        assert!(
            !matches,
            "{at}: a known gap now matches ({got:?}); remove it from KNOWN_GAPS"
        );
    } else {
        assert!(matches, "{at}: got {got:?}, Spark {want}");
    }
}

#[test]
fn json_decimal_inference_matches_spark_runs() {
    let dir = tempfile::tempdir().unwrap();
    for (profile, version) in [(SparkProfile::V4_2, "4.2.1"), (SparkProfile::V4_3, "4.3.0")] {
        let path = format!(
            "{}/../../tests/fixtures/spark-{version}/json-prefers-decimal.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let run: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(run["spark"], version);
        let s = Session::builder().profile(profile).build().unwrap();
        block_on(s.sql("SET spark.sql.session.timeZone = UTC")).unwrap();
        let cases = run["cases"].as_array().unwrap();
        assert!(cases.len() > 60, "{version}: {} cases", cases.len());
        for (i, case) in cases.iter().enumerate() {
            let text = case["text"].as_str().unwrap();
            let opts = case["options"].as_object().unwrap();
            let gap = KNOWN_GAPS.contains(&text);
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
            let got = block_on(async { s.sql(&sql).await?.collect().await })
                .map(|b| first_text(&b).unwrap_or_default())
                .map_err(|e| e.to_string());
            let want = case["schema_of_json"].as_str().unwrap();
            check(&got, want, gap, &format!("{version} {sql}"));

            let file = dir.path().join(format!("{version}-{i}.json"));
            std::fs::write(&file, format!("{text}\n")).unwrap();
            let mut reader = s.read().format("json");
            for (k, v) in opts {
                reader = reader.option(k, v.as_str().unwrap());
            }
            let df = reader.load([file.to_str().unwrap()]);
            let got = block_on(async {
                df.create_or_replace_temp_view("v").await?;
                df.schema().await
            })
            .map(|schema| {
                let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
                view_schema(&s, &names)
            })
            .map_err(|e| e.to_string());
            let want = case["file_schema"].as_str().unwrap();
            check(&got, want, gap, &format!("{version} file {text} {opts:?}"));
        }
    }
}

/// The `files` section: a non-numeric number beside good records in one
/// file goes to the parse mode alone (`PERMISSIVE` keeps it in the corrupt
/// column, `DROPMALFORMED` drops it, `FAILFAST` fails the read) while the
/// good records type the column.
#[test]
fn a_non_numeric_record_beside_good_ones_goes_to_the_parse_mode() {
    let dir = tempfile::tempdir().unwrap();
    for (profile, version) in [(SparkProfile::V4_2, "4.2.1"), (SparkProfile::V4_3, "4.3.0")] {
        let path = format!(
            "{}/../../tests/fixtures/spark-{version}/json-prefers-decimal.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let run: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(run["spark"], version);
        let s = Session::builder().profile(profile).build().unwrap();
        block_on(s.sql("SET spark.sql.session.timeZone = UTC")).unwrap();
        let files = run["files"].as_array().unwrap();
        assert!(files.len() >= 6, "{version}: {} files", files.len());
        for (i, case) in files.iter().enumerate() {
            let lines: Vec<&str> = case["lines"]
                .as_array()
                .unwrap()
                .iter()
                .map(|l| l.as_str().unwrap())
                .collect();
            let opts = case["options"].as_object().unwrap();
            let at = format!("{version} file {lines:?} {opts:?}");
            let file = dir.path().join(format!("{version}-lines-{i}.json"));
            std::fs::write(
                &file,
                lines.iter().map(|l| format!("{l}\n")).collect::<String>(),
            )
            .unwrap();
            let mut reader = s.read().format("json");
            for (k, v) in opts {
                reader = reader.option(k, v.as_str().unwrap());
            }
            let df = reader.load([file.to_str().unwrap()]);
            let got = block_on(async {
                df.create_or_replace_temp_view("v").await?;
                let schema = df.schema().await?;
                let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
                let rows = s
                    .sql("SELECT to_json(struct(*)) FROM v")
                    .await?
                    .collect()
                    .await?;
                Ok::<_, krishiv_common::error::KrishivError>((names, rows))
            });
            let want = case["schema"].as_str().unwrap();
            match (got, want.strip_prefix("ERROR ")) {
                (Err(e), Some(class)) => {
                    let name = e.error_class().name();
                    assert!(
                        class == name || class.starts_with(&format!("{name}.")),
                        "{at}: {class} vs {e}"
                    );
                }
                (Ok((names, rows)), None) => {
                    let names: Vec<&str> = names.iter().map(String::as_str).collect();
                    assert_eq!(view_schema(&s, &names), want, "{at}: schema");
                    let got: Vec<String> = rows
                        .iter()
                        .flat_map(|b| {
                            let c = b.column(0).as_string_view();
                            (0..b.num_rows()).map(move |r| c.value(r).to_owned())
                        })
                        .collect();
                    let want: Vec<&str> = case["values"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap())
                        .collect();
                    assert_eq!(got, want, "{at}: values");
                }
                (got, _) => panic!("{at}: got {got:?}, Spark {want}"),
            }
        }
    }
}
