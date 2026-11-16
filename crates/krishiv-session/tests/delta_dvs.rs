// SPDX-License-Identifier: Apache-2.0
//! Merge-on-read `DELETE`, `UPDATE` and `MERGE` of Delta tables with deletion vectors through the
//! session (feature `delta`, P3-T021 PR-37d) against Spark 4.2.0 + delta-spark 4.4.1
//! (`tests/fixtures/delta-4.4.1/dvs/DVS.json`, `gen_delta_fixtures.py /out dvs`): every step runs
//! on its own copy of its scenario's `before` table (delta-spark's own vectors in it), as the
//! oracle ran it, and is compared with the oracle's output, rows read back, commits, deletion
//! vector descriptors and metrics (`delta_dvs_match_spark`); Spark reads the tables Krishiv
//! changed, and derives the change data feed of a vector-only `DELETE` as for its own
//! (`spark_reads_tables_after_krishiv_dvs`); a cancelled statement leaves the version and the
//! files as they were (`a_cancelled_dv_delete_changes_nothing`); a statement under a small pool
//! (`a_dv_delete_under_a_small_pool`). The session's pool is back to zero after every statement.
#![cfg(test)]

use std::path::{Path, PathBuf};

use arrow_array::cast::AsArray;
use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{Array, RecordBatch};
use arrow_schema::DataType as ArrowType;
use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::Session;
use serde_json::Value;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/delta-4.4.1/dvs")
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

fn oracle() -> Value {
    serde_json::from_slice(&std::fs::read(fixtures().join("DVS.json")).unwrap()).unwrap()
}

fn cell(column: &dyn Array, row: usize) -> Value {
    if column.is_null(row) {
        return Value::Null;
    }
    match column.data_type() {
        ArrowType::Int64 => Value::from(column.as_primitive::<Int64Type>().value(row)),
        ArrowType::Int32 => Value::from(column.as_primitive::<Int32Type>().value(row)),
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

fn sql(s: &Session, text: &str) -> Result<(Vec<String>, Value), KrishivError> {
    block_on(async {
        let df = s.sql(text).await?;
        let batches = df.collect().await?;
        let names = batches
            .first()
            .map(|b| {
                b.schema()
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect()
            })
            .unwrap_or_default();
        Ok((names, rows_of(&batches)))
    })
}

/// The JSON commits under `dir` from version `from` on.
fn commits_from(dir: &Path, from: u64) -> Vec<Vec<Value>> {
    let log = dir.join("_delta_log");
    let mut versions: Vec<u64> = std::fs::read_dir(&log)
        .unwrap()
        .filter_map(|e| {
            let n = e.unwrap().file_name().into_string().unwrap();
            n.strip_suffix(".json")?.parse::<u64>().ok()
        })
        .filter(|v| *v >= from)
        .collect();
    versions.sort_unstable();
    versions
        .into_iter()
        .map(|v| {
            std::fs::read_to_string(log.join(format!("{v:020}.json")))
                .unwrap()
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect()
        })
        .collect()
}

fn of_kind<'a>(actions: &'a [Value], kind: &str) -> Vec<&'a Value> {
    actions.iter().filter_map(|a| a.get(kind)).collect()
}

/// The steps Krishiv refuses where delta-spark succeeds (design section 6.2): a change of a
/// change-data-feed table that writes rows (delta-spark writes change data files).
const CDF_REFUSED: &[(&str, &str)] = &[
    ("change-data-feed", "update_rows"),
    ("change-data-feed", "merge_delete"),
];

/// The metrics compared by value (bytes added depend on the writer, times on the run).
const COUNTED: &[&str] = &[
    "numRemovedFiles",
    "numRemovedBytes",
    "numDeletedRows",
    "numCopiedRows",
    "numUpdatedRows",
    "numAddedChangeFiles",
    "numDeletionVectorsAdded",
    "numDeletionVectorsRemoved",
    "numDeletionVectorsUpdated",
    "numSourceRows",
    "numTargetRowsInserted",
    "numTargetRowsUpdated",
    "numTargetRowsMatchedUpdated",
    "numTargetRowsNotMatchedBySourceUpdated",
    "numTargetRowsDeleted",
    "numTargetRowsMatchedDeleted",
    "numTargetRowsNotMatchedBySourceDeleted",
    "numTargetRowsCopied",
    "numOutputRows",
    "numTargetFilesRemoved",
    "numTargetBytesRemoved",
    "numTargetChangeFilesAdded",
    "numTargetDeletionVectorsAdded",
    "numTargetDeletionVectorsRemoved",
    "numTargetDeletionVectorsUpdated",
];

struct Run {
    scenario: String,
    step: String,
    dir: PathBuf,
    versions: u64,
    outcome: Result<(Vec<String>, Value), KrishivError>,
    _tmp: tempfile::TempDir,
}

/// Every step, each on its own copy of its scenario's `before` table, in a session pool of its
/// own, back to exactly zero after the statement.
fn run_all() -> Vec<(Value, Run)> {
    let oracle = oracle();
    let mut out = Vec::new();
    for (scenario, s) in oracle["scenarios"].as_object().unwrap() {
        let versions = s["versions"].as_u64().unwrap();
        for step in s["steps"].as_array().unwrap() {
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path().join("t");
            copy_dir(&fixtures().join(scenario).join("before"), &dir);
            let session = Session::builder()
                .config("krishiv.session.memoryLimit", "1g")
                .build()
                .unwrap();
            for (k, v) in step["confs"].as_object().unwrap() {
                sql(&session, &format!("SET {k} = {}", v.as_str().unwrap())).unwrap();
            }
            let text = step["sql"]
                .as_str()
                .unwrap()
                .replace("<t>", &format!("delta.`{}`", dir.display()));
            let outcome = sql(&session, &text);
            let name = format!("{scenario}/{}", step["step"].as_str().unwrap());
            assert_eq!(session.memory_reserved(), 0, "{name}: memory held");
            out.push((
                step.clone(),
                Run {
                    scenario: scenario.clone(),
                    step: step["step"].as_str().unwrap().to_owned(),
                    dir,
                    versions,
                    outcome,
                    _tmp: tmp,
                },
            ));
        }
    }
    out
}

/// The rows of the table at `dir`, read by a fresh session, by `id`.
fn table_rows(dir: &Path) -> Value {
    sql(
        &Session::new().unwrap(),
        &format!("SELECT * FROM delta.`{}` ORDER BY id", dir.display()),
    )
    .unwrap()
    .1
}

/// A deletion vector descriptor without its offset (delta-spark's order of vectors in its
/// file is its aggregation's; Krishiv writes them by path).
fn descriptor(dv: &Value) -> Value {
    match dv {
        Value::Object(o) => {
            let mut o = o.clone();
            o.remove("offset");
            Value::Object(o)
        }
        other => other.clone(),
    }
}

#[test]
fn delta_dvs_match_spark() {
    let mut checked = 0;
    let mut with_vectors = 0;
    for (step, run) in run_all() {
        let name = format!("{}/{}", run.scenario, run.step);
        if CDF_REFUSED
            .iter()
            .any(|(s, n)| *s == run.scenario && *n == run.step)
        {
            let Err(e) = &run.outcome else {
                panic!("{name}: a change of a change-data-feed table was accepted");
            };
            assert_eq!(
                e.error_class().name(),
                "UNSUPPORTED_FEATURE.TABLE_OPERATION",
                "{name}: {e}"
            );
            assert!(commits_from(&run.dir, run.versions).is_empty(), "{name}");
            continue;
        }
        assert!(step["error"].is_null(), "{name}: an oracle error");
        let (fields, rows) = match &run.outcome {
            Ok(o) => o,
            Err(e) => panic!("{name}: Krishiv failed: {e}"),
        };
        assert_eq!(
            Value::from(fields.clone()),
            step["output_fields"],
            "{name}: the output's columns"
        );
        assert_eq!(rows, &step["output"], "{name}: the command's output");
        assert_eq!(
            table_rows(&run.dir),
            step["rows"],
            "{name}: the rows read back"
        );
        let ours = commits_from(&run.dir, run.versions);
        let theirs = step["commits"].as_array().unwrap();
        assert_eq!(ours.len(), theirs.len(), "{name}: the commits written");
        checked += 1;
        let (Some(commit), Some(want)) = (ours.first(), theirs.first()) else {
            continue;
        };
        let info = commit.iter().find_map(|a| a.get("commitInfo")).unwrap();
        assert_eq!(info["operation"], want["commitInfo"]["operation"], "{name}");
        let (ours_m, theirs_m) = (
            &info["operationMetrics"],
            &want["commitInfo"]["operationMetrics"],
        );
        let keys = |m: &Value| {
            let mut k: Vec<String> = m.as_object().unwrap().keys().cloned().collect();
            k.sort();
            k
        };
        assert_eq!(keys(ours_m), keys(theirs_m), "{name}: the metric names");
        for k in COUNTED {
            assert_eq!(ours_m.get(*k), theirs_m.get(*k), "{name}: {k}");
        }
        // The files removed, each with its deletion vector as the log had it.
        let removed = |items: Vec<(String, Value)>| {
            let mut v = items;
            v.sort_by(|a, b| a.0.cmp(&b.0));
            v
        };
        let ours_removed = removed(
            of_kind(commit, "remove")
                .iter()
                .map(|r| {
                    (
                        r["path"].as_str().unwrap().to_owned(),
                        r.get("deletionVector").cloned().unwrap_or(Value::Null),
                    )
                })
                .collect(),
        );
        let theirs_removed = removed(
            want["remove_dvs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    (
                        r["path"].as_str().unwrap().to_owned(),
                        r["deletionVector"].clone(),
                    )
                })
                .collect(),
        );
        assert_eq!(ours_removed, theirs_removed, "{name}: the files removed");
        // The files added again with a new vector: the same files, the same vectors (but for
        // their offsets), persisted relative to the table root.
        let readded = |items: Vec<(String, Value)>| {
            let mut v: Vec<(String, Value)> = items
                .into_iter()
                .filter(|(_, dv)| !dv.is_null())
                .map(|(p, dv)| (p, descriptor(&dv)))
                .collect();
            v.sort_by(|a, b| a.0.cmp(&b.0));
            v
        };
        let ours_adds = of_kind(commit, "add");
        let ours_readded = readded(
            ours_adds
                .iter()
                .map(|a| {
                    (
                        a["path"].as_str().unwrap().to_owned(),
                        a.get("deletionVector").cloned().unwrap_or(Value::Null),
                    )
                })
                .collect(),
        );
        let theirs_readded = readded(
            want["add_dvs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|a| {
                    (
                        a["path"].as_str().unwrap().to_owned(),
                        a["deletionVector"].clone(),
                    )
                })
                .collect(),
        );
        // The same files and cardinalities; the z85 file id is this write's.
        // `sizeInBytes` is not compared: delta-spark run-optimizes its bitmaps
        // (`RoaringBitmapArray.runOptimize`), the kernel's writer does not (a recorded
        // difference: the same rows, more bytes for runs of rows).
        let strip = |v: &[(String, Value)]| -> Vec<(String, Value)> {
            v.iter()
                .map(|(p, d)| {
                    let mut d = d.clone();
                    if let Some(o) = d.as_object_mut() {
                        o.remove("pathOrInlineDv");
                        o.remove("sizeInBytes");
                    }
                    (p.clone(), d)
                })
                .collect()
        };
        assert_eq!(
            strip(&ours_readded),
            strip(&theirs_readded),
            "{name}: the new deletion vectors"
        );
        for (path, dv) in &ours_readded {
            assert_eq!(dv["storageType"], "u", "{name}: {path}");
            // Each named vector file is in the table root.
            let id = dv["pathOrInlineDv"].as_str().unwrap();
            assert_eq!(id.len(), 20, "{name}: no random prefix");
            assert!(
                std::fs::read_dir(&run.dir).unwrap().any(|e| e
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with("deletion_vector_")),
                "{name}: no vector file"
            );
            with_vectors += 1;
        }
        // A file added again keeps its record count; its bounds are no longer tight.
        for a in &ours_adds {
            if a.get("deletionVector").is_some() {
                let stats: Value = serde_json::from_str(a["stats"].as_str().unwrap()).unwrap();
                assert_eq!(stats["tightBounds"], false, "{name}");
                assert!(stats["numRecords"].is_u64(), "{name}");
            }
        }
        // The new data files: by the partitions written.
        let partitions = |adds: Vec<&Value>| {
            let mut p: Vec<String> = adds
                .iter()
                .map(|a| a["partitionValues"].to_string())
                .collect();
            p.sort();
            p.dedup();
            p
        };
        let theirs_new: Vec<&Value> = want["add_dvs"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|a| a["deletionVector"].is_null())
            .map(|a| {
                want["adds"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|x| x["path"] == a["path"])
                    .unwrap()
            })
            .collect();
        let ours_new: Vec<&Value> = ours_adds
            .iter()
            .copied()
            .filter(|a| a.get("deletionVector").is_none())
            .collect();
        assert_eq!(
            partitions(ours_new),
            partitions(theirs_new),
            "{name}: the partitions written"
        );
        // No change data file: readers derive a vector-only `DELETE`'s change rows.
        assert!(
            commit.iter().all(|a| a.get("cdc").is_none()),
            "{name}: a cdc file"
        );
        assert_eq!(want["cdc"], 0, "{name}");
    }
    assert!(checked > 30, "{checked} steps checked");
    assert!(with_vectors > 20, "{with_vectors} vectors checked");
}

/// The steps whose tables Spark reads back (`READBACK.json`).
const READBACK_STEPS: &[(&str, &str)] = &[
    ("unpartitioned", "delete_combines_existing_dv"),
    ("unpartitioned", "delete_null_condition"),
    ("unpartitioned", "update_rows"),
    ("unpartitioned", "merge_all_kinds"),
    ("partitioned", "update_moves_partition"),
    ("partitioned", "merge_upsert"),
    ("change-data-feed", "delete_rows"),
    ("change-data-feed", "delete_rows_and_file"),
];

fn readback_dir() -> PathBuf {
    fixtures().join("krishiv-written")
}

/// Writes the tables of `READBACK_STEPS` after Krishiv's statement to
/// `$KRISHIV_DELTA_READBACK_OUT/<scenario>--<step>` (run by hand with `--ignored`, then copied
/// to `tests/fixtures/delta-4.4.1/dvs/krishiv-written/` and read by `gen_delta_fixtures.py /out
/// readback /in`).
#[test]
#[ignore = "writes the readback tables; run by hand"]
fn write_dv_readback_tables() {
    let out = PathBuf::from(std::env::var("KRISHIV_DELTA_READBACK_OUT").unwrap());
    let oracle = oracle();
    for (scenario, name) in READBACK_STEPS {
        let s = &oracle["scenarios"][scenario];
        let step = s["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|x| x["step"] == *name)
            .unwrap();
        let dir = out.join(format!("{scenario}--{name}"));
        copy_dir(&fixtures().join(scenario).join("before"), &dir);
        let session = Session::new().unwrap();
        let text = step["sql"]
            .as_str()
            .unwrap()
            .replace("<t>", &format!("delta.`{}`", dir.display()));
        sql(&session, &text).unwrap();
    }
}

/// Spark 4.2.0 + delta-spark 4.4.1 read the tables Krishiv changed with deletion vectors
/// (`krishiv-written/READBACK.json`): the rows delta-spark's own statement left, the statement
/// in `DESCRIBE HISTORY`, the change rows it derives from a vector-only `DELETE` (those of its
/// own `DELETE`), and an append of Spark's own.
#[test]
fn spark_reads_tables_after_krishiv_dvs() {
    let record: Value =
        serde_json::from_slice(&std::fs::read(readback_dir().join("READBACK.json")).unwrap())
            .unwrap();
    let oracle = oracle();
    let tables = record["tables"].as_object().unwrap();
    assert_eq!(tables.len(), READBACK_STEPS.len());
    for (scenario, name) in READBACK_STEPS {
        let key = format!("{scenario}--{name}");
        let t = &tables[&key];
        assert!(t["error"].is_null(), "{key}: {}", t["error"]);
        let s = &oracle["scenarios"][scenario];
        let step = s["steps"]
            .as_array()
            .unwrap()
            .iter()
            .find(|x| x["step"] == *name)
            .unwrap();
        let fields: Vec<String> = s["fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| f.as_str().unwrap().to_owned())
            .collect();
        let rows: Vec<Value> = t["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                let o: Value = serde_json::from_str(r.as_str().unwrap()).unwrap();
                Value::Array(
                    fields
                        .iter()
                        .map(|f| o.get(f).cloned().unwrap_or(Value::Null))
                        .collect(),
                )
            })
            .collect();
        assert_eq!(Value::Array(rows), step["rows"], "{key}");
        let history = t["history"].as_array().unwrap();
        assert_eq!(
            history.last().unwrap()["operation"],
            step["history"]["operation"],
            "{key}"
        );
        if let Some(changes) = step.get("changes") {
            assert_eq!(&t["changes"], changes, "{key}: the change rows");
        }
        assert!(
            t["spark_append"]["error"].is_null(),
            "{key}: {}",
            t["spark_append"]
        );
    }
}

/// The files of `dir` other than the log, sorted.
fn data_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.ends_with("_delta_log") {
                continue;
            }
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "parquet" || x == "bin") {
                out.push(p.strip_prefix(dir).unwrap().to_path_buf());
            }
        }
    }
    out.sort();
    out
}

/// The `unpartitioned` table with `n` more rows in one file, written by Krishiv.
fn large_table(n: u64) -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    copy_dir(&fixtures().join("unpartitioned").join("before"), &dir);
    let s = Session::new().unwrap();
    let rows = block_on(s.sql(&format!(
        "SELECT id + 1000 AS id, CAST(id AS STRING) AS v, CAST(id % 7 AS INT) AS n FROM range({n})"
    )))
    .unwrap();
    block_on(
        rows.write()
            .format("delta")
            .mode("append")
            .save(dir.to_str().unwrap()),
    )
    .unwrap();
    (tmp, dir)
}

/// A `DELETE` cancelled while it runs (its query's time limit) leaves the table at its version,
/// no new data or vector file, and the pool at zero.
#[test]
fn a_cancelled_dv_delete_changes_nothing() {
    let (_tmp, dir) = large_table(3_000_000);
    let session = Session::builder()
        .config("krishiv.session.memoryLimit", "256m")
        .build()
        .unwrap();
    sql(&session, "SET krishiv.query.maxExecutionTime = 200ms").unwrap();
    let before = (commits_from(&dir, 0).len(), data_files(&dir));
    let text = format!(
        "DELETE FROM delta.`{}` WHERE n = 3 OR v LIKE '%7%'",
        dir.display()
    );
    let Err(e) = sql(&session, &text) else {
        panic!("the DELETE finished inside its time limit");
    };
    assert_eq!(e.sql_state(), Some("HY008"), "{e}");
    assert_eq!(session.memory_reserved(), 0);
    // Background deletes of a cancelled statement's files.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while data_files(&dir) != before.1 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert_eq!((commits_from(&dir, 0).len(), data_files(&dir)), before);
}

/// A `DELETE` marking two million rows of one file under a 32 MiB pool: the marks are reserved,
/// the statement commits one vector, and the pool returns to zero.
#[test]
fn a_dv_delete_under_a_small_pool() {
    let (_tmp, dir) = large_table(3_000_000);
    let session = Session::builder()
        .config("krishiv.session.memoryLimit", "32m")
        .build()
        .unwrap();
    let text = format!("DELETE FROM delta.`{}` WHERE id % 3 <> 0", dir.display());
    let (_, out) = sql(&session, &text).unwrap();
    assert_eq!(session.memory_reserved(), 0);
    let deleted = out[0][0].as_i64().unwrap();
    assert!(deleted > 1_900_000, "{deleted}");
    let last = commits_from(&dir, 0).pop().unwrap();
    let readded: Vec<&Value> = of_kind(&last, "add");
    assert!(
        readded.iter().any(|a| a.get("deletionVector").is_some()),
        "{readded:?}"
    );
    let (_, count) = sql(
        &Session::new().unwrap(),
        &format!("SELECT count(*) FROM delta.`{}`", dir.display()),
    )
    .unwrap();
    let left = count[0][0].as_i64().unwrap();
    assert_eq!(left + deleted, 3_000_000 + 90);
}
