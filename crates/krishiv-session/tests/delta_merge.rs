// SPDX-License-Identifier: Apache-2.0
//! Copy-on-write `MERGE INTO` of Delta tables through the session (feature `delta`, P3-T021
//! PR-37c) against Spark 4.2.0 + delta-spark 4.4.1 (`tests/fixtures/delta-4.4.1/merge/MERGE.json`,
//! `gen_delta_fixtures.py /out merge`): every step of every scenario runs on its own copy of the
//! scenario's `before` table, as the oracle ran it, and is compared with the oracle's command
//! output, rows read back, commits and `commitInfo` (`delta_merge_matches_spark`,
//! `delta_merge_metrics_match`), the multiple-match check and its exemptions
//! (`delta_merge_cardinality`), the change-data-feed steps Krishiv refuses
//! (`delta_cdf_merge_refused`) and Spark's read of tables Krishiv merged
//! (`spark_reads_tables_after_krishiv_merge`). The session's own memory pool is back to zero after
//! each statement, and a source larger than it spills (`a_merge_source_larger_than_memory_spills`).
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
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/delta-4.4.1/merge")
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
    serde_json::from_slice(&std::fs::read(fixtures().join("MERGE.json")).unwrap()).unwrap()
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
        ArrowType::Boolean => Value::from(column.as_boolean().value(row)),
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

fn of_kind_json<'a>(commit: &'a Value, kind: &str) -> Vec<&'a Value> {
    commit[kind]
        .as_array()
        .map(|a| a.iter().collect())
        .unwrap_or_default()
}

/// The steps Krishiv refuses where delta-spark succeeds (design section 6.2): a `MERGE` that
/// removes files of a change-data-feed table (delta-spark writes change data files).
const CDF_REFUSED: &[(&str, &str)] = &[
    ("change-data-feed", "upsert"),
    ("change-data-feed", "delete_matched"),
];

/// Steps Krishiv refuses by name where delta-spark succeeds: schema evolution.
const REFUSED: &[(&str, &str, &str)] = &[(
    "unpartitioned",
    "schema_evolution_conf",
    "KRISHIV_UNSUPPORTED_FEATURE",
)];

/// `e`'s message as the oracle records it: the first line, the table's path elided.
fn oracle_message(e: &KrishivError, dir: &Path) -> String {
    e.to_string()
        .lines()
        .next()
        .unwrap_or_default()
        .replace(&dir.display().to_string(), "<path>")
}

/// The metrics compared by value: counts (bytes added depend on the writer, times on the run).
const COUNTED: &[&str] = &[
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

/// `text` with the oracle's placeholders.
fn statement(text: &str, dir: &Path) -> String {
    text.replace("<t>", &format!("delta.`{}`", dir.display()))
        .replace(
            "<src_parquet>",
            &fixtures().join("source-parquet").display().to_string(),
        )
        .replace(
            "<src_delta>",
            &fixtures().join("source-delta").display().to_string(),
        )
}

/// Every step, each on its own copy of its scenario's `before` table.
fn run_all() -> Vec<(Value, Run)> {
    run_where(|_, _| true)
}

/// The steps `keep(scenario, step)` selects, chosen before any table is copied or run.
fn run_where(keep: impl Fn(&str, &str) -> bool) -> Vec<(Value, Run)> {
    let oracle = oracle();
    let mut out = Vec::new();
    for (scenario, s) in oracle["scenarios"].as_object().unwrap() {
        let versions = s["versions"].as_u64().unwrap();
        for step in s["steps"].as_array().unwrap() {
            if !keep(scenario, step["step"].as_str().unwrap()) {
                continue;
            }
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path().join("t");
            copy_dir(&fixtures().join(scenario).join("before"), &dir);
            // A session pool of its own: the runtime's caches (the Parquet footer cache) are not
            // in it, so the statement's memory is exactly what it reserves.
            let session = Session::builder()
                .config("krishiv.session.memoryLimit", "1g")
                .build()
                .unwrap();
            for (k, v) in step["confs"].as_object().unwrap() {
                sql(&session, &format!("SET {k} = {}", v.as_str().unwrap())).unwrap();
            }
            let outcome = sql(&session, &statement(step["sql"].as_str().unwrap(), &dir));
            let name = format!("{scenario}/{}", step["step"].as_str().unwrap());
            assert_eq!(
                session.memory_reserved(),
                0,
                "{name}: memory held after MERGE"
            );
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

fn cdf_refused(run: &Run) -> bool {
    CDF_REFUSED
        .iter()
        .any(|(s, n)| *s == run.scenario && *n == run.step)
}

fn refused_by_name(run: &Run) -> Option<&'static str> {
    REFUSED
        .iter()
        .find(|(s, n, _)| *s == run.scenario && *n == run.step)
        .map(|(_, _, c)| *c)
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

#[test]
fn delta_merge_matches_spark() {
    let mut checked = 0;
    for (step, run) in run_all() {
        let name = format!("{}/{}", run.scenario, run.step);
        if cdf_refused(&run) {
            continue;
        }
        if let Some(class) = refused_by_name(&run) {
            let Err(e) = &run.outcome else {
                panic!("{name}: Krishiv accepted a refused step");
            };
            assert_eq!(e.error_class().name(), class, "{name}: {e}");
            assert!(commits_from(&run.dir, run.versions).is_empty(), "{name}");
            continue;
        }
        match (&run.outcome, &step["error"]) {
            (Err(e), Value::Null) => panic!("{name}: Krishiv failed: {e}"),
            (Ok(_), Value::Object(theirs)) => {
                panic!("{name}: Krishiv succeeded where delta-spark raised {theirs:?}")
            }
            (Err(e), Value::Object(theirs)) => {
                assert_eq!(
                    e.error_class().name(),
                    theirs["condition"].as_str().unwrap(),
                    "{name}: {e}"
                );
                assert_eq!(e.sql_state(), theirs["sqlstate"].as_str(), "{name}");
                let message = theirs["message"].as_str().unwrap();
                // The messages that name no expression (delta-spark prints resolved
                // expressions where Krishiv prints the text as written).
                if !message.is_empty() && !message.contains("condition =") {
                    // delta-spark's own exceptions print no `SQLSTATE` suffix.
                    let ours = oracle_message(e, &run.dir);
                    let ours = match ours.rsplit_once(" SQLSTATE: ") {
                        Some((head, _)) if !message.contains("SQLSTATE") => head.to_owned(),
                        _ => ours,
                    };
                    assert_eq!(ours, message, "{name}: the message");
                }
                if let Some(causes) = theirs.get("causes").and_then(Value::as_array) {
                    let ours: Vec<_> = e.causes().collect();
                    assert_eq!(ours.len(), causes.len(), "{name}: the causes of {e}");
                    for (c, want) in ours.iter().zip(causes) {
                        assert_eq!(c.error_class().name(), want["condition"], "{name}");
                        assert_eq!(
                            oracle_message(c, &run.dir),
                            want["message"].as_str().unwrap(),
                            "{name}: the cause's message"
                        );
                    }
                }
            }
            (Ok((fields, rows)), _) => {
                assert_eq!(
                    Value::from(fields.clone()),
                    step["output_fields"],
                    "{name}: the output's columns"
                );
                assert_eq!(rows, &step["output"], "{name}: the command's output");
            }
            (_, other) => panic!("{name}: an oracle error record {other}"),
        }
        assert_eq!(
            table_rows(&run.dir),
            step["rows"],
            "{name}: the rows read back"
        );
        let ours = commits_from(&run.dir, run.versions);
        let theirs = step["commits"].as_array().unwrap();
        assert_eq!(ours.len(), theirs.len(), "{name}: the commits written");
        checked += 1;
    }
    assert!(checked > 50, "{checked} steps checked");
}

#[test]
fn delta_merge_metrics_match() {
    let mut checked = 0;
    for (step, run) in run_all() {
        let name = format!("{}/{}", run.scenario, run.step);
        if cdf_refused(&run) || run.outcome.is_err() {
            continue;
        }
        let ours = commits_from(&run.dir, run.versions);
        let Some(commit) = ours.first() else {
            continue;
        };
        let theirs = &step["commits"][0];
        let info = commit.iter().find_map(|a| a.get("commitInfo")).unwrap();
        let want = &theirs["commitInfo"];
        assert_eq!(info["operation"], want["operation"], "{name}");
        assert_eq!(info["readVersion"], want["readVersion"], "{name}");
        // The parameters: the same keys; each clause list of the same length with the same
        // action types (the predicates are the text as written, a recorded difference).
        let params = |v: &Value| {
            let mut k: Vec<String> = v["operationParameters"]
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect();
            k.sort();
            k
        };
        assert_eq!(params(info), params(want), "{name}: the parameter names");
        for key in [
            "matchedPredicates",
            "notMatchedPredicates",
            "notMatchedBySourcePredicates",
        ] {
            let shape = |v: &Value| -> Vec<(String, bool)> {
                let list: Vec<Value> =
                    serde_json::from_str(v["operationParameters"][key].as_str().unwrap()).unwrap();
                list.iter()
                    .map(|c| {
                        (
                            c["actionType"].as_str().unwrap().to_owned(),
                            c.get("predicate").is_some(),
                        )
                    })
                    .collect()
            };
            assert_eq!(shape(info), shape(want), "{name}: {key}");
        }
        let (ours_m, theirs_m) = (&info["operationMetrics"], &want["operationMetrics"]);
        let keys = |m: &Value| {
            let mut k: Vec<String> = m.as_object().unwrap().keys().cloned().collect();
            k.sort();
            k
        };
        assert_eq!(keys(ours_m), keys(theirs_m), "{name}: the metric names");
        for k in COUNTED {
            assert_eq!(ours_m.get(*k), theirs_m.get(*k), "{name}: {k}");
        }
        let removed = |c: &[&Value]| {
            let mut p: Vec<(String, Value)> = c
                .iter()
                .map(|r| {
                    (
                        r["path"].as_str().unwrap().to_owned(),
                        r.get("deletionVector").cloned().unwrap_or(Value::Null),
                    )
                })
                .collect();
            p.sort_by(|a, b| a.0.cmp(&b.0));
            p
        };
        assert_eq!(
            removed(&of_kind(commit, "remove")),
            removed(&of_kind_json(theirs, "removes")),
            "{name}: the files removed"
        );
        let partitions = |adds: Vec<&Value>| {
            let mut p: Vec<String> = adds
                .iter()
                .map(|a| a["partitionValues"].to_string())
                .collect();
            p.sort();
            p.dedup();
            p
        };
        let (ours_adds, theirs_adds) = (of_kind(commit, "add"), of_kind_json(theirs, "adds"));
        assert_eq!(
            ours_m["numTargetFilesAdded"].as_str().map(str::to_owned),
            Some(ours_adds.len().to_string()),
            "{name}"
        );
        assert_eq!(
            partitions(ours_adds),
            partitions(theirs_adds),
            "{name}: the partitions written"
        );
        checked += 1;
    }
    assert!(checked > 30, "{checked} steps checked");
}

/// The multiple-match check: Delta's class where delta-spark raised it, and the exemptions
/// (one unconditional `DELETE`, an insert-only `MERGE` under the default conf) commit.
#[test]
fn delta_merge_cardinality() {
    let mut raised = 0;
    let mut exempt = 0;
    for (step, run) in run_where(|_, step| step.starts_with("multiple_match")) {
        let name = format!("{}/{}", run.scenario, run.step);
        match (&run.outcome, &step["error"]) {
            (Err(e), Value::Object(_)) => {
                assert_eq!(
                    e.error_class().name(),
                    "DELTA_MULTIPLE_SOURCE_ROW_MATCHING_TARGET_ROW_IN_MERGE",
                    "{name}"
                );
                assert_eq!(e.sql_state(), Some("21506"));
                assert!(commits_from(&run.dir, run.versions).is_empty(), "{name}");
                raised += 1;
            }
            (Ok((_, rows)), Value::Null) => {
                assert_eq!(rows, &step["output"], "{name}");
                exempt += 1;
            }
            (ours, theirs) => panic!("{name}: {ours:?} against {theirs}"),
        }
    }
    assert_eq!((raised, exempt), (5, 3));
}

#[test]
fn delta_cdf_merge_refused() {
    let mut refusals = 0;
    for (step, run) in run_where(|scenario, step| {
        CDF_REFUSED
            .iter()
            .any(|(s, n)| *s == scenario && *n == step)
    }) {
        let name = format!("{}/{}", run.scenario, run.step);
        // delta-spark commits these with change data files; Krishiv refuses them before any
        // file is written.
        assert!(step["error"].is_null(), "{name}");
        assert!(step["cdc_files"].as_u64().unwrap() > 0, "{name}");
        let Err(e) = &run.outcome else {
            panic!("{name}: a MERGE that removes files of a change-data-feed table was accepted");
        };
        assert_eq!(
            e.error_class().name(),
            "UNSUPPORTED_FEATURE.TABLE_OPERATION",
            "{name}: {e}"
        );
        assert!(e.to_string().contains("change data feed"), "{name}: {e}");
        assert!(commits_from(&run.dir, run.versions).is_empty(), "{name}");
        assert_eq!(
            parquet_files(&run.dir),
            parquet_files(&fixtures().join(&run.scenario).join("before")),
            "{name}: no file written"
        );
        refusals += 1;
    }
    assert_eq!(refusals, CDF_REFUSED.len());
}

/// The data files under `dir`, by relative path.
fn parquet_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "parquet") {
                out.push(p.strip_prefix(dir).unwrap().to_path_buf());
            }
        }
    }
    out.sort();
    out
}

/// The steps whose tables Spark reads back (`READBACK.json`).
const READBACK_STEPS: &[(&str, &str)] = &[
    ("unpartitioned", "all_kinds"),
    ("unpartitioned", "clause_order"),
    ("unpartitioned", "insert_star"),
    ("partitioned", "upsert_new_partition"),
    ("partitioned", "update_moves_partition"),
    ("partitioned", "upsert_and_conditional_nmbs"),
    ("column-mapping", "upsert"),
    ("deletion-vectors", "update_copy_on_write"),
    ("change-data-feed", "insert_only"),
];

fn readback_dir() -> PathBuf {
    fixtures().join("krishiv-written")
}

/// Writes the tables of `READBACK_STEPS` after Krishiv's statement to
/// `$KRISHIV_DELTA_READBACK_OUT/<scenario>--<step>` (run by hand with `--ignored`, then copied
/// to `tests/fixtures/delta-4.4.1/merge/krishiv-written/` and read by `gen_delta_fixtures.py
/// /out readback /in`).
#[test]
#[ignore = "writes the readback tables; run by hand"]
fn write_merge_readback_tables() {
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
        // A table left by an earlier run would be merged twice.
        assert!(
            !dir.exists(),
            "{}: remove the earlier output first",
            dir.display()
        );
        copy_dir(&fixtures().join(scenario).join("before"), &dir);
        let session = Session::new().unwrap();
        sql(&session, &statement(step["sql"].as_str().unwrap(), &dir)).unwrap();
    }
}

/// Spark 4.2.0 + delta-spark 4.4.1 read the tables Krishiv merged (`krishiv-written/
/// READBACK.json`): the rows delta-spark's own statement left, `MERGE` in `DESCRIBE HISTORY`,
/// and an append of Spark's own.
#[test]
fn spark_reads_tables_after_krishiv_merge() {
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
        let fields: Vec<String> = step["fields"]
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
        assert_eq!(history.last().unwrap()["operation"], "MERGE", "{key}");
        assert!(
            t["spark_append"]["error"].is_null(),
            "{key}: {}",
            t["spark_append"]
        );
        // Spark's append of one row reads back on top of the rows the MERGE left.
        assert_eq!(
            t["spark_append"]["rows"].as_u64(),
            Some(t["rows"].as_array().unwrap().len() as u64 + 1),
            "{key}"
        );
    }
}

/// The spill and temporary files under `spill` (none when it was never made); a directory that
/// cannot be read fails the test.
fn spill_files(spill: &Path) -> Vec<PathBuf> {
    let mut left = Vec::new();
    if !spill.exists() {
        return left;
    }
    let mut stack = vec![spill.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "spill" || x == "tmp") {
                left.push(p);
            }
        }
    }
    left
}

/// A source of ~60 MiB under a 16 MiB session pool: the copy of the source spills and is read
/// back a batch at a time; the MERGE commits and the pool returns to zero.
#[test]
fn a_merge_source_larger_than_memory_spills() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    copy_dir(&fixtures().join("unpartitioned").join("before"), &dir);
    let spill = tmp.path().join("spill");
    let runtime = krishiv_session::EngineRuntime::builder()
        .config("krishiv.spill.dir", spill.to_str().unwrap())
        .build()
        .unwrap();
    let session = Session::builder()
        .runtime(runtime)
        .config("krishiv.session.memoryLimit", "16m")
        .build()
        .unwrap();
    let text = format!(
        "MERGE INTO delta.`{}` t USING (SELECT id, repeat('x', 200) AS v, CAST(id AS INT) AS n \
         FROM range(300000)) s ON t.id = s.id WHEN MATCHED THEN UPDATE SET v = substr(s.v, 1, 3)",
        dir.display()
    );
    // Every target row matches: each keeps its id and `n`, `v` becomes "xxx".
    let mut expected = table_rows(&dir);
    for r in expected.as_array_mut().unwrap() {
        r[1] = serde_json::json!("xxx");
    }
    let (_, out) = sql(&session, &text).unwrap();
    assert_eq!(out, serde_json::json!([[6, 6, 0, 0]]));
    assert_eq!(session.memory_reserved(), 0);
    // The copy went to the spill directory (the runtime opens it for the first query that
    // spills: nothing else here spills) and went with the statement.
    assert!(spill.is_dir(), "the source never spilled");
    let left = spill_files(&spill);
    assert!(left.is_empty(), "{left:?}");
    let rows = table_rows(&dir);
    assert_eq!(rows.as_array().map(Vec::len), Some(6), "{rows}");
    assert_eq!(rows, expected);
}

/// Under `spark.sql.ansi.doubleQuotedIdentifiers` a double-quoted name in a clause is the source
/// column, in the rewrite as in the analysis (the clause text is read again under the statement's
/// parser options); a single-quoted one stays a literal.
#[test]
fn a_double_quoted_name_in_a_clause_reads_the_column_under_ansi_identifiers() {
    for (value, written) in [("\"payload\"", "new"), ("'payload'", "payload")] {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("t");
        copy_dir(&fixtures().join("unpartitioned").join("before"), &dir);
        let mut expected = table_rows(&dir);
        for r in expected.as_array_mut().unwrap() {
            r[1] = serde_json::json!(written);
        }
        let session = Session::new().unwrap();
        sql(&session, "SET spark.sql.ansi.enabled = true").unwrap();
        sql(
            &session,
            "SET spark.sql.ansi.doubleQuotedIdentifiers = true",
        )
        .unwrap();
        let text = format!(
            "MERGE INTO delta.`{}` t USING (SELECT id, 'new' AS payload FROM range(1000)) s \
             ON t.id = s.id WHEN MATCHED THEN UPDATE SET v = {value}",
            dir.display()
        );
        sql(&session, &text).unwrap();
        assert_eq!(table_rows(&dir), expected, "{value}");
    }
}

/// A MERGE cancelled through the session while it runs (its query's time limit, during the
/// source's copy) leaves the table at its version, writes no visible file, gives the pool back
/// and leaves no spill file.
#[test]
fn a_merge_cancelled_through_the_session_changes_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    copy_dir(&fixtures().join("unpartitioned").join("before"), &dir);
    let spill = tmp.path().join("spill");
    let runtime = krishiv_session::EngineRuntime::builder()
        .config("krishiv.spill.dir", spill.to_str().unwrap())
        .build()
        .unwrap();
    let session = Session::builder()
        .runtime(runtime)
        .config("krishiv.session.memoryLimit", "64m")
        .build()
        .unwrap();
    sql(&session, "SET krishiv.query.maxExecutionTime = 300ms").unwrap();
    let before = (commits_from(&dir, 0).len(), parquet_files(&dir));
    let text = format!(
        "MERGE INTO delta.`{}` t USING (SELECT id, repeat('x', 200) AS v, CAST(id AS INT) AS n \
         FROM range(3000000)) s ON t.id = s.id WHEN MATCHED THEN UPDATE SET v = s.v",
        dir.display()
    );
    let Err(e) = sql(&session, &text) else {
        panic!("the MERGE finished inside its time limit");
    };
    assert_eq!(e.sql_state(), Some("HY008"), "{e}");
    assert_eq!(session.memory_reserved(), 0);
    assert_eq!((commits_from(&dir, 0).len(), parquet_files(&dir)), before);
    let left = spill_files(&spill);
    assert!(left.is_empty(), "{left:?}");
}

/// A CSV source by path with `header` in its `WITH (...)` options at `<tmp>/src`.
fn csv_source(tmp: &Path) -> PathBuf {
    let src = tmp.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("s.csv"), "id,v\n1,zz\n9,new\n").unwrap();
    src
}

#[test]
fn a_merge_source_reads_with_its_own_options() {
    // Spark reads a table source with its `WITH (...)` options (AstBuilder.scala 1286-1290 at
    // v4.3.0): with `header`, the header line is not a source row.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    copy_dir(&fixtures().join("unpartitioned").join("before"), &dir);
    let src = csv_source(tmp.path());
    let session = Session::new().unwrap();
    let text = format!(
        "MERGE INTO delta.`{}` t USING csv.`{}` WITH ('header' = 'true') s \
         ON t.id = CAST(s.id AS BIGINT) \
         WHEN MATCHED THEN UPDATE SET v = s.v \
         WHEN NOT MATCHED THEN INSERT (id, v, n) VALUES (CAST(s.id AS BIGINT), s.v, 0)",
        dir.display(),
        src.display()
    );
    let (_, out) = sql(&session, &text).unwrap();
    assert_eq!(out, serde_json::json!([[2, 1, 0, 1]]));
    let rows = table_rows(&dir);
    let rows = rows.as_array().unwrap();
    assert_eq!(rows.len(), 7, "{rows:?}");
    assert!(rows.iter().any(|r| r[0] == 1 && r[1] == "zz"), "{rows:?}");
    assert!(rows.iter().any(|r| r[0] == 9 && r[1] == "new"), "{rows:?}");
    assert!(!rows.iter().any(|r| r[1] == "v"), "{rows:?}");
}

#[test]
fn a_merge_source_option_error_is_the_reads_error() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    copy_dir(&fixtures().join("unpartitioned").join("before"), &dir);
    let src = csv_source(tmp.path());
    let session = Session::new().unwrap();
    let before = (table_rows(&dir), commits_from(&dir, 0).len());
    let read = sql(
        &session,
        &format!(
            "SELECT * FROM csv.`{}` WITH ('header' = 'true', 'sep' = '')",
            src.display()
        ),
    )
    .unwrap_err();
    let merge = sql(
        &session,
        &format!(
            "MERGE INTO delta.`{}` t USING csv.`{}` WITH ('header' = 'true', 'sep' = '') s \
             ON t.id = CAST(s.id AS BIGINT) WHEN MATCHED THEN DELETE",
            dir.display(),
            src.display()
        ),
    )
    .unwrap_err();
    assert_eq!(merge.error_class(), read.error_class(), "{merge} / {read}");
    assert_eq!(merge.to_string(), read.to_string());
    // The table is unchanged: the same rows at the same version.
    assert_eq!((table_rows(&dir), commits_from(&dir, 0).len()), before);
}

#[test]
fn a_target_or_source_column_named_like_a_presence_marker_keeps_its_values() {
    // The join's presence markers must not take the place of a user column of the same name
    // (case-insensitively): an UPDATE of `v` leaves the target's own Boolean as it was, and the
    // source's column of the marker's name is an ordinary source column.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    let session = Session::new().unwrap();
    sql(
        &session,
        &format!(
            "CREATE TABLE t USING delta LOCATION '{}' AS SELECT * FROM VALUES \
             (CAST(1 AS BIGINT), false, 'a'), (CAST(2 AS BIGINT), CAST(NULL AS BOOLEAN), 'b') \
             AS x(id, __KRISHIV_MERGE_TARGET_PRESENT, v)",
            dir.display()
        ),
    )
    .unwrap();
    let (_, out) = sql(
        &session,
        "MERGE INTO t USING (SELECT * FROM VALUES (CAST(1 AS BIGINT), false), \
         (CAST(3 AS BIGINT), false) AS s(id, __krishiv_merge_source_present)) s \
         ON t.id = s.id \
         WHEN MATCHED THEN UPDATE SET v = 'z' \
         WHEN NOT MATCHED THEN INSERT (id, __krishiv_merge_target_present, v) \
         VALUES (s.id, s.__krishiv_merge_source_present, 'new')",
    )
    .unwrap();
    assert_eq!(out, serde_json::json!([[2, 1, 0, 1]]));
    assert_eq!(
        table_rows(&dir),
        serde_json::json!([[1, false, "z"], [2, null, "b"], [3, false, "new"]])
    );
}

#[test]
fn an_early_clause_without_a_condition_is_spark_s_class() {
    // Spark's builder: only the last clause of each kind may omit its condition (the golden steps
    // cover them too; these name each class).
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    copy_dir(&fixtures().join("unpartitioned").join("before"), &dir);
    let session = Session::new().unwrap();
    for (clauses, class) in [
        (
            "WHEN MATCHED THEN DELETE WHEN MATCHED AND s.id > 0 THEN DELETE",
            "NON_LAST_MATCHED_CLAUSE_OMIT_CONDITION",
        ),
        (
            "WHEN NOT MATCHED THEN INSERT * WHEN NOT MATCHED AND s.id > 0 THEN INSERT *",
            "NON_LAST_NOT_MATCHED_BY_TARGET_CLAUSE_OMIT_CONDITION",
        ),
        (
            "WHEN NOT MATCHED BY SOURCE THEN DELETE \
             WHEN NOT MATCHED BY SOURCE AND t.id > 0 THEN DELETE",
            "NON_LAST_NOT_MATCHED_BY_SOURCE_CLAUSE_OMIT_CONDITION",
        ),
    ] {
        let e = sql(
            &session,
            &format!(
                "MERGE INTO delta.`{}` t USING delta.`{}` s ON t.id = s.id {clauses}",
                dir.display(),
                dir.display()
            ),
        )
        .unwrap_err();
        assert_eq!(e.error_class().name(), class, "{e}");
        assert_eq!(e.sql_state(), Some("42613"), "{e}");
    }
    assert_eq!(table_rows(&dir).as_array().unwrap().len(), 6);
}

/// A quoted top-level column whose name holds a dot is that column, not a nested field: `SET
/// `a.b` = ..` and `INSERT (`a.b`)` assign it, while `SET st.x = ..` of a struct field stays
/// refused (#962 review).
#[test]
fn a_quoted_column_name_with_a_dot_is_a_top_level_column() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    let session = Session::new().unwrap();
    sql(
        &session,
        &format!(
            "CREATE TABLE t (id BIGINT, `a.b` STRING, st STRUCT<x: INT>) USING delta LOCATION '{}'",
            dir.display()
        ),
    )
    .unwrap();
    sql(
        &session,
        "INSERT INTO t SELECT 1, 'one', named_struct('x', 1)",
    )
    .unwrap();
    let (_, out) = sql(
        &session,
        "MERGE INTO t USING (SELECT CAST(id AS BIGINT) AS id, v FROM VALUES (1, 'uno'), (2, 'dos') \
         AS s(id, v)) s ON t.id = s.id WHEN MATCHED THEN UPDATE SET `a.b` = s.v \
         WHEN NOT MATCHED THEN INSERT (id, `a.b`) VALUES (s.id, s.v)",
    )
    .unwrap();
    assert_eq!(out, serde_json::json!([[2, 1, 0, 1]]));
    let (_, rows) = sql(&session, "SELECT id, `a.b` FROM t ORDER BY id").unwrap();
    assert_eq!(rows, serde_json::json!([[1, "uno"], [2, "dos"]]));
    let e = sql(
        &session,
        "MERGE INTO t USING (SELECT CAST(1 AS BIGINT) AS id) s ON t.id = s.id \
         WHEN MATCHED THEN UPDATE SET st.x = 2",
    )
    .unwrap_err();
    assert!(e.to_string().contains("nested field"), "{e}");
}

/// A `MERGE` inside a SQL script sees the script's locals in its source, its clause conditions
/// and its values, and a local shadows a session variable of the same name (#962 review).
#[test]
fn a_merge_in_a_script_reads_the_script_s_locals() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    let session = Session::new().unwrap();
    sql(
        &session,
        &format!(
            "CREATE TABLE t USING delta LOCATION '{}' AS SELECT * FROM VALUES \
             (CAST(1 AS BIGINT), 'a') AS x(id, v)",
            dir.display()
        ),
    )
    .unwrap();
    sql(&session, "DECLARE VARIABLE k BIGINT DEFAULT 100").unwrap();
    sql(
        &session,
        "BEGIN \
           DECLARE k BIGINT DEFAULT 2; \
           DECLARE w STRING DEFAULT 'local'; \
           MERGE INTO t USING (SELECT k AS id UNION ALL SELECT 1) s ON t.id = s.id \
           WHEN MATCHED AND s.id < k THEN UPDATE SET v = w \
           WHEN NOT MATCHED THEN INSERT (id, v) VALUES (s.id + k, w); \
         END",
    )
    .unwrap();
    assert_eq!(
        table_rows(&dir),
        serde_json::json!([[1, "local"], [4, "local"]])
    );
}
