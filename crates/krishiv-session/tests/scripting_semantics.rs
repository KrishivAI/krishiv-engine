// SPDX-License-Identifier: Apache-2.0
//! The Krishiv-authored scripting goldens (`tests/spark-golden/v4.2.0/scripting/semantics/*.json`,
//! design note T196 section 2) against this engine.
//!
//! A case is a list of statements run in order in a fresh session with an empty table `s(i INT)`.
//! A statement passes when it returns the recorded rows, or fails with the recorded error class.
//! `scripting/semantics-known-failures.json` lists the cases that do not pass yet (a reason each,
//! written by hand); the test fails for a case that fails and is not listed, and for a listed one that
//! passes. `KRISHIV_RATCHET_UPDATE=1` rewrites the file, keeping the reasons that exist.
//! The three cases with a `divergence` carry the behaviour this engine has instead of Spark's; the
//! two that name an error class are checked here, the column-name one is a known gap.
#![cfg(test)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_session::Session;
use serde_json::Value;

const FILES: [&str; 5] = [
    "structural",
    "entry_results",
    "variables",
    "control_flow",
    "handlers",
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/spark-golden/v4.2.0/scripting")
}

fn fresh() -> (Session, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    s.conf()
        .set("spark.sql.warehouse.dir", dir.path().to_str().unwrap())
        .unwrap();
    s.conf().set("spark.sql.session.timeZone", "UTC").unwrap();
    block_on(async {
        s.sql("CREATE TABLE s (i INT) USING parquet")
            .await?
            .collect()
            .await
    })
    .unwrap();
    (s, dir)
}

/// The error a statement raises: its class and the message parameters, as strings.
type Failure = (String, Vec<(String, String)>);

fn cells(s: &Session, sql: &str) -> Result<Vec<Vec<String>>, Failure> {
    let batches = block_on(async { s.sql(sql).await?.collect().await }).map_err(|e| {
        (
            e.error_class().name().to_owned(),
            e.message_parameters()
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect(),
        )
    })?;
    let mut out = Vec::new();
    for b in &batches {
        for i in 0..b.num_rows() {
            out.push(
                b.columns()
                    .iter()
                    .map(|c| {
                        if c.is_null(i) {
                            return "NULL".to_owned();
                        }
                        match c.data_type() {
                            ArrowType::Utf8View => c.as_string_view().value(i).to_owned(),
                            ArrowType::Utf8 => c.as_string::<i32>().value(i).to_owned(),
                            ArrowType::Boolean => c.as_boolean().value(i).to_string(),
                            ArrowType::Int32 => c
                                .as_primitive::<arrow_array::types::Int32Type>()
                                .value(i)
                                .to_string(),
                            ArrowType::Int64 => c
                                .as_primitive::<arrow_array::types::Int64Type>()
                                .value(i)
                                .to_string(),
                            _ => "?".to_owned(),
                        }
                    })
                    .collect(),
            );
        }
    }
    Ok(out)
}

fn text(v: &Value) -> String {
    match v {
        Value::Null => "NULL".to_owned(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Why the case differs from the record, or `None`.
fn check(case: &Value) -> Option<String> {
    let (s, _dir) = fresh();
    for statement in case["statements"].as_array().into_iter().flatten() {
        let query = statement["query"].as_str().unwrap_or_default();
        let got = cells(&s, query);
        if statement["error"].is_object() {
            // A failure that is not a Spark error (a Java class) has no class to compare.
            let Some(class) = statement["error"]["class"].as_str() else {
                continue;
            };
            match got {
                Err((c, params)) if c == class => {
                    // The recorded parameters, but for the ones holding a normalised scope or database.
                    for (key, want) in statement["error"]["params"]
                        .as_object()
                        .into_iter()
                        .flatten()
                    {
                        let want = text(want);
                        if want.contains('<') {
                            continue;
                        }
                        let have = params
                            .iter()
                            .find(|(k, _)| k == key)
                            .map(|(_, v)| v.as_str());
                        if have != Some(want.as_str()) {
                            return Some(format!("{query}: {key} is {have:?}, Spark {want:?}"));
                        }
                    }
                }
                Err((c, _)) => return Some(format!("{query}: fails {c}, Spark {class}")),
                Ok(_) => return Some(format!("{query}: runs, Spark fails {class}")),
            }
        } else {
            let want: Vec<Vec<String>> = statement["rows"]
                .as_array()
                .map(|rows| {
                    rows.iter()
                        .map(|r| {
                            r.as_array()
                                .map(|c| c.iter().map(text).collect())
                                .unwrap_or_default()
                        })
                        .collect()
                })
                .unwrap_or_default();
            match got {
                Err((c, _)) => return Some(format!("{query}: fails {c}, Spark returns rows")),
                Ok(rows) => {
                    let same = rows.len() == want.len()
                        && rows.iter().zip(&want).all(|(a, b)| {
                            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x == "?" || x == y)
                        });
                    if !same {
                        return Some(format!("{query}: {rows:?}, Spark {want:?}"));
                    }
                }
            }
        }
    }
    None
}

/// The behaviour this engine has instead of Spark's, for a case with a `divergence`.
fn check_divergence(case: &Value) -> Option<String> {
    let expected = case["divergence"]["krishiv_expected"][0].as_str()?;
    let class = expected.strip_prefix("error ")?.split_whitespace().next()?;
    let query = case["divergence"]["spark_actual_not_expected"][0]["query"].as_str()?;
    let (s, _dir) = fresh();
    match cells(&s, query) {
        Err((c, _)) if c == class => None,
        other => Some(format!("{query}: {other:?}, expected error {class}")),
    }
}

#[test]
fn the_scripting_semantics_match_the_spark_record() {
    let mut failing: BTreeMap<String, String> = BTreeMap::new();
    for file in FILES {
        let doc: Value = serde_json::from_str(
            &std::fs::read_to_string(root().join(format!("semantics/{file}.json"))).unwrap(),
        )
        .unwrap();
        for case in doc["cases"].as_array().into_iter().flatten() {
            let name = case["name"].as_str().unwrap_or_default();
            let key = format!("{file}/{name}");
            let why = if case["divergence"].is_object() {
                check_divergence(case)
            } else {
                check(case)
            };
            if let Some(why) = why {
                failing.insert(key, why);
            }
        }
    }
    let path = root().join("semantics-known-failures.json");
    let known: BTreeMap<String, String> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    if std::env::var_os("KRISHIV_RATCHET_UPDATE").is_some() {
        let updated: BTreeMap<String, String> = failing
            .iter()
            .map(|(k, why)| {
                let reason = known
                    .get(k)
                    .cloned()
                    .unwrap_or_else(|| format!("UNEXPLAINED: {why}"));
                (k.clone(), reason)
            })
            .collect();
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&updated).unwrap() + "\n",
        )
        .unwrap();
        return;
    }
    let new: Vec<_> = failing
        .iter()
        .filter(|(k, _)| !known.contains_key(*k))
        .collect();
    let fixed: Vec<_> = known.keys().filter(|k| !failing.contains_key(*k)).collect();
    assert!(
        new.is_empty(),
        "cases that fail and are not listed: {new:#?}"
    );
    assert!(
        fixed.is_empty(),
        "listed cases that now pass; remove them from semantics-known-failures.json: {fixed:?}"
    );
}
