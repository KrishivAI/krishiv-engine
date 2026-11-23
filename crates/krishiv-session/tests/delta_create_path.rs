// SPDX-License-Identifier: Apache-2.0
//! `CREATE TABLE delta.`<path>`` against delta-spark 4.4.1 (`tests/fixtures/delta-4.4.1/CREATE-PATH.json`,
//! `gen_delta_fixtures.py /out create-path`; queue row 74): each case's error, or the table it writes
//! at the path (metadata, columns read back) with no catalog entry.
#![cfg(test)]

use std::path::{Path, PathBuf};

use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;
use serde_json::Value;

fn oracle() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/delta-4.4.1/CREATE-PATH.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The cases Krishiv refuses by name where delta-spark creates or replaces a table
/// (`not_null_comments`: the table comment, which the kernel's create cannot write).
const REFUSED: &[&str] = &[
    "not_null_comments",
    "or_replace_new",
    "or_replace_existing",
    "ctas",
    "no_columns",
];

/// The cases whose refused name Krishiv prints lower-cased (a known difference: the session
/// folds a table name before `INVALID_SCHEMA_OR_RELATION_NAME`, Spark prints it as written).
const FOLDED_NAME: &[&str] = &["using_parquet", "no_using"];

/// The cases whose target is a name, not an absolute path (`<path>` in the record stands for it).
const RELATIVE: &[(&str, &str)] = &[
    ("relative_path", "krishiv_rel/dir"),
    ("no_slash", "krishiv_no_slash"),
];

/// The cases that run on the table `plain` creates (the oracle's first path).
const ON_PLAIN: &[&str] = &[
    "plain",
    "again_same",
    "again_other",
    "again_no_columns",
    "if_not_exists_same",
    "if_not_exists_other",
    "again_partitioned",
    "or_replace_existing",
];

fn run(s: &Session, text: &str) -> Result<Vec<arrow_array::RecordBatch>, KrishivError> {
    block_on(async { s.sql(text).await?.collect().await })
}

/// The JSON actions of version `v` of the table at `dir`.
fn commit(dir: &Path, v: u64) -> Vec<Value> {
    std::fs::read_to_string(dir.join("_delta_log").join(format!("{v:020}.json")))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn versions(dir: &Path) -> usize {
    std::fs::read_dir(dir.join("_delta_log")).map_or(0, |d| {
        d.filter(|e| {
            e.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".json")
        })
        .count()
    })
}

fn meta_data(actions: &[Value]) -> &Value {
    actions
        .iter()
        .find_map(|a| a.get("metaData"))
        .expect("a metaData action")
}

/// A schemaString's fields as (name, type, nullable): the metadata of each field is compared
/// apart.
fn fields(meta: &Value) -> Vec<Value> {
    let schema: Value = serde_json::from_str(meta["schemaString"].as_str().unwrap()).unwrap();
    schema["fields"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| serde_json::json!([f["name"], f["type"], f["nullable"]]))
        .collect()
}

#[test]
fn delta_create_by_path_matches_spark() {
    let oracle = oracle();
    let root = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    let plain = root.path().join("plain");
    let mut n = 0;
    let mut checked = 0;
    for case in oracle["cases"].as_array().unwrap() {
        let name = case["case"].as_str().unwrap();
        // The two name cases: the generator elided their relative names as `<path>` too.
        let dir: PathBuf = if let Some((_, relative)) = RELATIVE.iter().find(|(c, _)| *c == name) {
            PathBuf::from(relative)
        } else if ON_PLAIN.contains(&name) {
            plain.clone()
        } else {
            n += 1;
            root.path().join(format!("t{n}"))
        };
        match name {
            "empty_directory" => std::fs::create_dir_all(&dir).unwrap(),
            "parquet_directory" => {
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(dir.join("part-00000.parquet"), b"PAR1").unwrap();
            }
            _ => {}
        }
        let text = case["sql"]
            .as_str()
            .unwrap()
            .replace("<path>", &dir.display().to_string());
        let before = versions(&dir);
        let got = run(&s, &text);
        if REFUSED.contains(&name) {
            let e = got.expect_err(name);
            assert_eq!(
                e.error_class().name(),
                "UNSUPPORTED_FEATURE.TABLE_OPERATION",
                "{name}: {e}"
            );
            assert_eq!(versions(&dir), before, "{name}: a refusal writes nothing");
            continue;
        }
        match (&got, &case["error"]) {
            (Ok(_), Value::Null) => {
                let written = case["commits"].as_array().unwrap();
                assert_eq!(versions(&dir) - before, written.len(), "{name}: commits");
                if let Some(theirs) = written.first() {
                    let ours = commit(&dir, 0);
                    let (ours, theirs) = (
                        meta_data(&ours),
                        meta_data(theirs["actions"].as_array().unwrap()),
                    );
                    assert_eq!(fields(ours), fields(theirs), "{name}: columns");
                    assert_eq!(
                        ours["partitionColumns"], theirs["partitionColumns"],
                        "{name}"
                    );
                    assert_eq!(ours["configuration"], theirs["configuration"], "{name}");
                }
                let back = run(&s, &format!("SELECT * FROM delta.`{}`", dir.display())).unwrap();
                let schema = back
                    .first()
                    .map(arrow_array::RecordBatch::schema)
                    .unwrap_or_else(|| {
                        block_on(async {
                            s.sql(&format!("SELECT * FROM delta.`{}`", dir.display()))
                                .await
                                .unwrap()
                                .schema()
                                .await
                                .unwrap()
                        })
                    });
                let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
                let want: Vec<&str> = case["columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| c[0].as_str().unwrap())
                    .collect();
                assert_eq!(names, want, "{name}: columns read back");
                assert_eq!(
                    back.iter()
                        .map(arrow_array::RecordBatch::num_rows)
                        .sum::<usize>(),
                    0,
                    "{name}"
                );
            }
            (Err(e), Value::Object(theirs)) => {
                assert_eq!(
                    e.error_class().name(),
                    theirs["condition"].as_str().unwrap(),
                    "{name}: {e}"
                );
                assert_eq!(e.sql_state(), theirs["sqlstate"].as_str(), "{name}");
                let message = e
                    .to_string()
                    .lines()
                    .next()
                    .unwrap_or_default()
                    .replace(&dir.display().to_string(), "<path>");
                let want = theirs["message"].as_str().unwrap();
                if FOLDED_NAME.contains(&name) {
                    // The name is case-folded before Krishiv's name check; Spark shows it as
                    // written (queue follow-up).
                    assert_eq!(
                        message,
                        want.replace("<path>", &dir.display().to_string().to_lowercase()),
                        "{name}"
                    );
                } else {
                    assert_eq!(message, want, "{name}");
                }
                assert_eq!(versions(&dir), before, "{name}: an error writes nothing");
            }
            (got, want) => panic!("{name}: Krishiv {got:?}, delta-spark {want}"),
        }
        checked += 1;
    }
    assert_eq!(
        checked + REFUSED.len(),
        oracle["cases"].as_array().unwrap().len()
    );
    // No catalog entry: the oracle's `SHOW TABLES` is empty after every case.
    let tables = run(&s, "SHOW TABLES IN default").unwrap();
    assert_eq!(
        tables
            .iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        oracle["tables_after"].as_array().unwrap().len()
    );
}

/// A column comment and `CHAR`/`VARCHAR` types: delta-spark keeps them as field metadata
/// (`not_null_comments`, run without its table comment, which Krishiv refuses; `types`).
#[test]
fn delta_create_by_path_field_metadata() {
    let oracle = oracle();
    let root = tempfile::tempdir().unwrap();
    let s = Session::new().unwrap();
    for name in ["not_null_comments", "types"] {
        let case = oracle["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["case"] == name)
            .unwrap();
        let dir = root.path().join(name);
        let text = case["sql"]
            .as_str()
            .unwrap()
            .replace("<path>", &dir.display().to_string())
            .replace(" COMMENT 'a table'", "");
        run(&s, &text).unwrap();
        let ours = commit(&dir, 0);
        let theirs = case["commits"][0]["actions"].as_array().unwrap();
        let metadata = |meta: &Value| -> Vec<Value> {
            let schema: Value =
                serde_json::from_str(meta["schemaString"].as_str().unwrap()).unwrap();
            schema["fields"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| f["metadata"].clone())
                .collect()
        };
        assert_eq!(
            metadata(meta_data(&ours)),
            metadata(meta_data(theirs)),
            "{name}: field metadata"
        );
    }
}
