// SPDX-License-Identifier: Apache-2.0
//! `INSERT INTO delta.`<path>`` (feature `delta`) against Spark 4.2.0 + delta-spark 4.4.1
//! (`tests/fixtures/delta-4.4.1/INSERT-PATH.json`, `gen_delta_fixtures.py /out insert-path`):
//! every step of the oracle replayed on a Krishiv-made table at a path, its rows, commit and
//! error compared with the oracle's (`insert_by_path_matches_spark`), then the refused targets
//! (`insert_by_path_refusals_match_spark`).
#![cfg(test)]

use std::path::{Path, PathBuf};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;
use serde_json::Value;

fn oracle() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/delta-4.4.1/INSERT-PATH.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn cell(column: &dyn Array, row: usize) -> Value {
    if column.is_null(row) {
        return Value::Null;
    }
    match column.data_type() {
        ArrowType::Int64 => Value::from(column.as_primitive::<Int64Type>().value(row)),
        ArrowType::Utf8 => Value::from(column.as_string::<i32>().value(row)),
        ArrowType::Utf8View => Value::from(column.as_string_view().value(row)),
        other => panic!("no oracle form for {other}"),
    }
}

fn rows_of(batches: &[RecordBatch]) -> Value {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            out.push(Value::Array(
                b.columns().iter().map(|c| cell(c.as_ref(), r)).collect(),
            ));
        }
    }
    Value::Array(out)
}

fn run(s: &Session, sql: &str) -> Result<(), KrishivError> {
    block_on(async { s.sql(sql).await?.collect().await }).map(drop)
}

fn commits(dir: &Path) -> Vec<u64> {
    let mut v: Vec<u64> = std::fs::read_dir(dir.join("_delta_log"))
        .unwrap()
        .filter_map(|e| {
            let n = e.unwrap().file_name().into_string().unwrap();
            n.strip_suffix(".json")?.parse::<u64>().ok()
        })
        .collect();
    v.sort_unstable();
    v
}

fn commit(dir: &Path, version: u64) -> Vec<Value> {
    std::fs::read_to_string(dir.join("_delta_log").join(format!("{version:020}.json")))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn count(actions: &[Value], kind: &str) -> usize {
    actions.iter().filter(|a| a.get(kind).is_some()).count()
}

/// The commit `ours` against the oracle's record `theirs`: operation, parameters, the counts
/// delta-spark and Krishiv both write, blind append, read version, and the files.
fn compare(name: &str, ours: &[Value], theirs: &Value) {
    let oi = ours
        .iter()
        .find_map(|a| a.get("commitInfo").cloned())
        .unwrap();
    let ti = &theirs["commitInfo"];
    assert_eq!(oi["operation"], ti["operation"], "{name}");
    assert_eq!(
        oi["operationParameters"], ti["operationParameters"],
        "{name}"
    );
    for k in ["numFiles", "numOutputRows", "numRemovedFiles"] {
        assert_eq!(
            oi["operationMetrics"].get(k),
            ti["operationMetrics"].get(k),
            "{name} {k}"
        );
    }
    assert_eq!(
        oi["isBlindAppend"].as_bool().unwrap_or(false),
        ti["isBlindAppend"].as_bool().unwrap_or(false),
        "{name}"
    );
    assert_eq!(oi["readVersion"], ti["readVersion"], "{name}");
    assert_eq!(
        count(ours, "add"),
        theirs["adds"].as_array().unwrap().len(),
        "{name}"
    );
    assert_eq!(
        count(ours, "remove"),
        theirs["removes"].as_array().unwrap().len(),
        "{name}"
    );
}

/// The error `e` against the oracle's: class, SQLSTATE and parameters, `<path>` read as `path`.
fn same_error(name: &str, e: &KrishivError, theirs: &Value, placeholder: &str, path: &str) {
    assert_eq!(
        e.error_class().name(),
        theirs["condition"].as_str().unwrap(),
        "{name}: {e}"
    );
    // The PySpark capture of a Delta analysis error carries no SQLSTATE (`DELTA_TABLE_NOT_FOUND`
    // is 42P01 in delta-spark's `delta-error-classes.json`, which the registry vendors).
    if let Some(state) = theirs["sqlstate"].as_str() {
        assert_eq!(e.sql_state(), Some(state), "{name}: {e}");
    }
    let want: serde_json::Map<String, Value> = theirs["parameters"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                Value::from(v.as_str().unwrap().replace(placeholder, path)),
            )
        })
        .collect();
    let got: serde_json::Map<String, Value> = e
        .message_parameters()
        .iter()
        .map(|(k, v)| ((*k).to_owned(), Value::from(v.as_str())))
        .collect();
    assert_eq!(got, want, "{name}: {e}");
}

#[test]
fn insert_by_path_matches_spark() {
    let oracle = oracle();
    let tmp = tempfile::tempdir().unwrap();
    let dir: PathBuf = tmp.path().join("insert-path");
    let path = dir.to_string_lossy().into_owned();
    let s = Session::new().unwrap();
    let t = format!("delta.`{path}`");
    // As the oracle's `CREATE TABLE delta.`<path>``: version 0 with the columns and no files
    // (Krishiv refuses that path form of `CREATE TABLE` yet, so an external catalog table makes
    // it and is dropped again, leaving the files).
    run(
        &s,
        &format!(
            "CREATE TABLE made (id BIGINT, v STRING, p STRING) USING delta PARTITIONED BY (p) \
             LOCATION '{path}'"
        ),
    )
    .unwrap();
    run(&s, "DROP TABLE made").unwrap();
    run(&s, &format!("INSERT INTO {t} VALUES (1, 'a', 'x')")).unwrap();
    for step in oracle["steps"].as_array().unwrap() {
        let name = step["step"].as_str().unwrap();
        let sql = step["sql"].as_str().unwrap().replace("<path>", &path);
        let before = commits(&dir);
        match (run(&s, &sql), step["error"].is_null()) {
            (Ok(()), true) => {}
            (Err(e), false) => same_error(name, &e, &step["error"], "<path>", &path),
            (Ok(()), false) => panic!("{name}: Spark refused it, Krishiv ran it"),
            (Err(e), true) => panic!("{name}: Spark ran it, Krishiv refused: {e}"),
        }
        let after = commits(&dir);
        let theirs = step["commits"].as_array().unwrap();
        assert_eq!(after.len() - before.len(), theirs.len(), "{name}: commits");
        if let (Some(last), Some(theirs)) = (after.last(), theirs.first())
            && after.len() > before.len()
        {
            compare(name, &commit(&dir, *last), theirs);
        }
        let rows = rows_of(
            &block_on(
                block_on(s.sql(&format!("SELECT id, v, p FROM {t} ORDER BY id")))
                    .unwrap()
                    .collect(),
            )
            .unwrap(),
        );
        assert_eq!(rows, step["rows"], "{name}: rows");
    }
}

#[test]
fn insert_by_path_refusals_match_spark() {
    let oracle = oracle();
    let tmp = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    for step in oracle["refused"].as_array().unwrap() {
        let name = step["step"].as_str().unwrap();
        let target = tmp.path().join(name);
        match name {
            "missing" | "no_slash" => {}
            "empty_directory" => std::fs::create_dir(&target).unwrap(),
            "parquet_directory" => block_on(async {
                s.sql("SELECT CAST(1 AS BIGINT) AS id")
                    .await?
                    .write()
                    .parquet(target.to_str().unwrap())
                    .await
            })
            .unwrap(),
            other => panic!("no setup for {other}"),
        }
        let (sql, placeholder, path) = if name == "no_slash" {
            (
                step["sql"]
                    .as_str()
                    .unwrap()
                    .replace("<path>", "krishiv_no_slash"),
                "<path>",
                "krishiv_no_slash".to_owned(),
            )
        } else {
            let path = target.to_string_lossy().into_owned();
            (
                step["sql"].as_str().unwrap().replace("<missing>", &path),
                "<path>",
                path,
            )
        };
        let e = run(&s, &sql).expect_err(name);
        same_error(name, &e, &step["error"], placeholder, &path);
        assert!(
            !target.join("_delta_log").exists(),
            "{name}: a log was made"
        );
    }
}
