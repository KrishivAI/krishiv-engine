// SPDX-License-Identifier: Apache-2.0
//! Four correctness checks from Spark 4.3's fixes, end to end against a real
//! Spark 4.3.0 run: `tests/fixtures/spark-4.3.0/spark43-checks.json`
//! (`tests/scripts/gen_spark43_checks_goldens.py`) lists each query with the
//! rows Spark gave. The queries read two of the Spark image's example files,
//! kept beside the fixture.
//!
//! - `nd_*`: a non-deterministic predicate (`rand`) is evaluated after the
//!   scan, never pushed into it.
//! - `gen_*`: with a generator in the SELECT list, HAVING runs before window
//!   functions.
//! - `opts_*`: two references to one relation keep their own `WITH (...)`.
//!
//! A case the engine cannot answer yet is listed in `PENDING` with the way it
//! fails, and the test requires it to keep failing that way, so fixing one
//! forces the list to shrink. (Check 3, a failing partition predicate, needs a
//! partitioned table that the oracle's temp-view-only rule cannot build.)
#![cfg(test)]

use std::path::Path;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use futures::executor::block_on;
use krishiv_session::Session;
use serde_json::Value;

/// Cases that fail today: `(id, how)` where `how` is an error class name, or
/// `"rows"` for a wrong answer.
const PENDING: [(&str, &str); 0] = [];

fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/spark-4.3.0")
}

fn cell(a: &ArrayRef, i: usize) -> String {
    if a.is_null(i) {
        return "NULL".to_owned();
    }
    match a.data_type() {
        DataType::Int32 => a.as_primitive::<Int32Type>().value(i).to_string(),
        DataType::Int64 => a.as_primitive::<Int64Type>().value(i).to_string(),
        DataType::Utf8 => a.as_string::<i32>().value(i).to_owned(),
        DataType::Utf8View => a.as_string_view().value(i).to_owned(),
        t => panic!("unexpected result type {t}"),
    }
}

/// Empty when the case matches Spark, else how it differs.
fn run(dir: &Path, case: &Value) -> String {
    let s = Session::new().unwrap();
    let sub = |sql: &str| {
        sql.replace("{kv}", dir.join("kv1.txt").to_str().unwrap())
            .replace("{people}", dir.join("people.csv").to_str().unwrap())
    };
    let result = block_on(async {
        for stmt in case["setup"].as_array().unwrap() {
            s.sql(&sub(stmt.as_str().unwrap())).await?.collect().await?;
        }
        s.sql(&sub(case["sql"].as_str().unwrap()))
            .await?
            .collect()
            .await
    });
    match (result, case.get("rows")) {
        (Ok(batches), Some(want)) => {
            let mut got: Vec<Vec<String>> = Vec::new();
            for b in &batches {
                for i in 0..b.num_rows() {
                    got.push(b.columns().iter().map(|c| cell(c, i)).collect());
                }
            }
            let mut want: Vec<Vec<String>> = want
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    r.as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_str().unwrap_or("NULL").to_owned())
                        .collect()
                })
                .collect();
            let ordered = case["sql"].as_str().unwrap().contains("order by");
            if !ordered {
                got.sort();
                want.sort();
            }
            if got == want {
                String::new()
            } else {
                "rows".to_owned()
            }
        }
        (Err(e), _) => e.error_class().name().to_owned(),
        (Ok(_), None) => "no error".to_owned(),
    }
}

#[test]
fn spark_4_3_correctness_checks() {
    let dir = fixture_dir();
    let doc: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("spark43-checks.json")).unwrap())
            .unwrap();
    let mut failing = Vec::new();
    for case in doc["cases"].as_array().unwrap() {
        let how = run(&dir, case);
        if !how.is_empty() {
            failing.push((case["id"].as_str().unwrap().to_owned(), how));
        }
    }
    let want: Vec<(String, String)> = PENDING
        .iter()
        .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
        .collect();
    assert_eq!(failing, want, "cases failing today");
}
