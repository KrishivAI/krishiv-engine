// SPDX-License-Identifier: Apache-2.0
//! Copy-on-write `DELETE` and `UPDATE` of Delta tables through the session (feature `delta`,
//! P3-T021 PR-37b part 2) against Spark 4.2.0 + delta-spark 4.4.1
//! (`tests/fixtures/delta-4.4.1/dml/DML.json`, `gen_delta_fixtures.py /out dml`): every step of
//! every scenario runs on its own copy of the scenario's `before` table, as the oracle ran it,
//! and is compared with the oracle's command output, rows read back, commits and
//! `commitInfo` (`delta_dml_matches_spark`, `delta_dml_metrics_match`); the change-data-feed
//! steps Krishiv refuses (`delta_cdf_dml_refused`).
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
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/delta-4.4.1/dml")
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
    serde_json::from_slice(&std::fs::read(fixtures().join("DML.json")).unwrap()).unwrap()
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
        // The oracle's form (`gen_delta_fixtures.py` `typed`): Python's `isoformat` of the
        // UTC wall clock.
        ArrowType::Timestamp(arrow_schema::TimeUnit::Microsecond, _) => {
            let micros = column
                .as_primitive::<arrow_array::types::TimestampMicrosecondType>()
                .value(row);
            let (secs, frac) = (micros.div_euclid(1_000_000), micros.rem_euclid(1_000_000));
            let (days, rest) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
            // Civil date from days since 1970-01-01 (H. Hinnant's `civil_from_days`).
            let z = days + 719_468;
            let era = z.div_euclid(146_097);
            let doe = z - era * 146_097;
            let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
            let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
            let mp = (5 * doy + 2) / 153;
            let day = doy - (153 * mp + 2) / 5 + 1;
            let month = if mp < 10 { mp + 3 } else { mp - 9 };
            let year = yoe + era * 400 + i64::from(month <= 2);
            let mut text = format!(
                "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}",
                rest / 3600,
                rest % 3600 / 60,
                rest % 60
            );
            if frac != 0 {
                text.push_str(&format!(".{frac:06}"));
            }
            serde_json::json!({ "timestamp": text })
        }
        ArrowType::Struct(fields) => {
            let s = column.as_struct();
            let mut out = serde_json::Map::new();
            for (f, c) in fields.iter().zip(s.columns()) {
                out.insert(f.name().clone(), cell(c.as_ref(), row));
            }
            serde_json::json!({ "struct": out })
        }
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

/// The steps Krishiv refuses where delta-spark succeeds (design section 6.2): a rewrite of a
/// change-data-feed table.
const CDF_REFUSED: &[(&str, &str)] = &[
    ("change-data-feed", "delete_rows"),
    ("change-data-feed", "update_rows"),
    ("change-data-feed", "update_partition_only"),
];

/// Steps whose error differs from delta-spark's (a recorded difference): the class Krishiv
/// raises. None now: an `UPDATE` overflow is delta-spark's `TASK_WRITE_FAILED` caused by
/// `DELTA_CAST_OVERFLOW_IN_TABLE_WRITE` (review 1, M2).
const KNOWN_ERRORS: &[(&str, &str, &str)] = &[];

/// `e`'s message as the oracle records it: the first line, the table's path elided.
fn oracle_message(e: &KrishivError, dir: &Path) -> String {
    e.to_string()
        .lines()
        .next()
        .unwrap_or_default()
        .replace(&dir.display().to_string(), "<path>")
}

/// Steps whose commits differ from delta-spark's by a recorded difference (the connector's
/// `write::dml` module): delta-spark writes deletion vectors there, Krishiv rewrites files.
fn writes_deletion_vectors(step: &Value) -> bool {
    step["confs"]
        .as_object()
        .is_some_and(serde_json::Map::is_empty)
        && step["commits"].as_array().unwrap().iter().any(|c| {
            of_kind_json(c, "adds")
                .iter()
                .any(|a| a.get("deletionVector").is_some())
        })
        || step["history"]["operationMetrics"]
            .get("numDeletionVectorsAdded")
            .and_then(Value::as_str)
            .is_some_and(|n| n != "0")
            && step["confs"]
                .as_object()
                .is_some_and(serde_json::Map::is_empty)
}

fn of_kind_json<'a>(commit: &'a Value, kind: &str) -> Vec<&'a Value> {
    commit[kind]
        .as_array()
        .map(|a| a.iter().collect())
        .unwrap_or_default()
}

/// The metrics compared by value: counts (bytes added depend on the writer, times on the run).
/// `numAddedFiles` is compared as the partitions written: delta-spark writes one file per
/// partition per task, Krishiv one per partition per driver.
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
];

struct Run {
    scenario: String,
    step: String,
    dir: PathBuf,
    versions: u64,
    outcome: Result<(Vec<String>, Value), KrishivError>,
    _tmp: tempfile::TempDir,
}

/// Every step, each on its own copy of its scenario's `before` table.
fn run_all() -> Vec<(Value, Run)> {
    let oracle = oracle();
    let mut out = Vec::new();
    for (scenario, s) in oracle["scenarios"].as_object().unwrap() {
        let versions = s["versions"].as_u64().unwrap();
        for step in s["steps"].as_array().unwrap() {
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path().join("t");
            copy_dir(&fixtures().join(scenario).join("before"), &dir);
            let session = Session::new().unwrap();
            for (k, v) in step["confs"].as_object().unwrap() {
                sql(&session, &format!("SET {k} = {}", v.as_str().unwrap())).unwrap();
            }
            let text = step["sql"]
                .as_str()
                .unwrap()
                .replace("<t>", &format!("delta.`{}`", dir.display()));
            let outcome = sql(&session, &text);
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

fn refused(run: &Run) -> bool {
    CDF_REFUSED
        .iter()
        .any(|(s, n)| *s == run.scenario && *n == run.step)
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
fn delta_dml_matches_spark() {
    let mut checked = 0;
    for (step, run) in run_all() {
        let name = format!("{}/{}", run.scenario, run.step);
        if refused(&run) {
            continue;
        }
        match (&run.outcome, &step["error"]) {
            (Err(e), Value::Null) => panic!("{name}: Krishiv failed: {e}"),
            (Ok(_), Value::Object(theirs)) => {
                panic!("{name}: Krishiv succeeded where delta-spark raised {theirs:?}")
            }
            (Err(e), Value::Object(_))
                if KNOWN_ERRORS
                    .iter()
                    .any(|(s, n, _)| *s == run.scenario && *n == run.step) =>
            {
                let (_, _, class) = KNOWN_ERRORS
                    .iter()
                    .find(|(s, n, _)| *s == run.scenario && *n == run.step)
                    .unwrap();
                assert_eq!(e.error_class().name(), *class, "{name}: {e}");
            }
            (Err(e), Value::Object(theirs)) => {
                assert_eq!(
                    e.error_class().name(),
                    theirs["condition"].as_str().unwrap(),
                    "{name}: {e}"
                );
                assert_eq!(e.sql_state(), theirs["sqlstate"].as_str(), "{name}");
                // A wrapped error: the message and every classified cause, exactly.
                if let Some(causes) = theirs.get("causes").and_then(Value::as_array) {
                    assert_eq!(
                        oracle_message(e, &run.dir),
                        theirs["message"].as_str().unwrap(),
                        "{name}: the message"
                    );
                    let ours: Vec<_> = e.causes().collect();
                    assert_eq!(ours.len(), causes.len(), "{name}: the causes of {e}");
                    for (c, want) in ours.iter().zip(causes) {
                        assert_eq!(c.error_class().name(), want["condition"], "{name}");
                        assert_eq!(c.sql_state(), want["sqlstate"].as_str(), "{name}");
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
    assert!(checked > 30, "{checked} steps checked");
}

#[test]
fn delta_dml_metrics_match() {
    let mut checked = 0;
    for (step, run) in run_all() {
        let name = format!("{}/{}", run.scenario, run.step);
        if refused(&run) || run.outcome.is_err() {
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
        // The predicate: the condition as written (a recorded difference from delta-spark's
        // `simpleString`); both a one-element array, or both empty.
        let ours_p: Vec<String> =
            serde_json::from_str(info["operationParameters"]["predicate"].as_str().unwrap())
                .unwrap();
        let theirs_p: Vec<String> =
            serde_json::from_str(want["operationParameters"]["predicate"].as_str().unwrap())
                .unwrap();
        assert_eq!(
            ours_p.len(),
            theirs_p.len(),
            "{name}: {ours_p:?} {theirs_p:?}"
        );
        if writes_deletion_vectors(&step) {
            // delta-spark marked rows in deletion vectors; Krishiv rewrote (a recorded
            // difference): only the rows are compared (`delta_dml_matches_spark`).
            continue;
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
        // The files removed, each with its deletion vector as the log had it.
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
            ours_adds.is_empty(),
            theirs_adds.is_empty(),
            "{name}: files added or not"
        );
        assert_eq!(
            ours_m["numAddedFiles"].as_str().map(str::to_owned),
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
    assert!(checked > 20, "{checked} steps checked");
}

#[test]
fn delta_cdf_dml_refused() {
    let mut refusals = 0;
    for (step, run) in run_all() {
        if !refused(&run) {
            continue;
        }
        let name = format!("{}/{}", run.scenario, run.step);
        // delta-spark commits these with change data files; Krishiv refuses them before any
        // file is written.
        assert!(step["error"].is_null(), "{name}");
        let Err(e) = &run.outcome else {
            panic!("{name}: a rewrite of a change-data-feed table was accepted");
        };
        assert_eq!(
            e.error_class().name(),
            "UNSUPPORTED_FEATURE.TABLE_OPERATION",
            "{name}: {e}"
        );
        assert!(e.to_string().contains("change data feed"), "{name}: {e}");
        assert!(commits_from(&run.dir, run.versions).is_empty(), "{name}");
        let data: usize = std::fs::read_dir(&run.dir)
            .unwrap()
            .flat_map(|e| {
                let p = e.unwrap().path();
                if p.is_dir() && !p.ends_with("_delta_log") {
                    std::fs::read_dir(p)
                        .unwrap()
                        .map(|e| e.unwrap().path())
                        .collect()
                } else {
                    vec![p]
                }
            })
            .filter(|p| p.extension().is_some_and(|x| x == "parquet"))
            .count();
        let before: usize = {
            let b = fixtures().join(&run.scenario).join("before");
            std::fs::read_dir(&b)
                .unwrap()
                .flat_map(|e| {
                    let p = e.unwrap().path();
                    if p.is_dir() && !p.ends_with("_delta_log") {
                        std::fs::read_dir(p)
                            .unwrap()
                            .map(|e| e.unwrap().path())
                            .collect()
                    } else {
                        vec![p]
                    }
                })
                .filter(|p| p.extension().is_some_and(|x| x == "parquet"))
                .count()
        };
        assert_eq!(data, before, "{name}: no file written");
        refusals += 1;
    }
    assert_eq!(refusals, CDF_REFUSED.len());
}

/// The steps whose tables Spark reads back (`READBACK_DML`).
const READBACK_STEPS: &[(&str, &str)] = &[
    ("unpartitioned", "delete_rows"),
    ("unpartitioned", "update_store_assignment"),
    ("partitioned", "delete_partition"),
    ("partitioned", "delete_rows_any_partition"),
    ("partitioned", "update_moves_partition"),
    ("partitioned", "update_partition_to_null"),
    ("column-mapping", "update_rows"),
    ("deletion-vectors", "delete_rows_copy_on_write"),
    ("deletion-vectors", "update_rows_with_dvs"),
    ("change-data-feed", "delete_partition"),
];

fn readback_dir() -> PathBuf {
    fixtures().join("krishiv-written")
}

/// Writes the tables of `READBACK_STEPS` after Krishiv's statement to
/// `$KRISHIV_DELTA_READBACK_OUT/<scenario>--<step>` (run by hand with `--ignored`, then copied
/// to `tests/fixtures/delta-4.4.1/dml/krishiv-written/` and read by `gen_delta_fixtures.py /out
/// readback /in`).
#[test]
#[ignore = "writes the readback tables; run by hand"]
fn write_dml_readback_tables() {
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

/// Spark 4.2.0 + delta-spark 4.4.1 read the tables Krishiv changed (`krishiv-written/
/// READBACK.json`): the rows delta-spark's own statement left, the statement in `DESCRIBE
/// HISTORY`, and an append of Spark's own.
#[test]
fn spark_reads_tables_after_krishiv_dml() {
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
        assert!(
            t["spark_append"]["error"].is_null(),
            "{key}: {}",
            t["spark_append"]
        );
    }
}

/// Allows everything but `Modify` on a path (#909 review thread: a Delta table by path is
/// changed only with `Modify` on the path, as a Delta save to it asks).
struct NoPathModify;

impl krishiv_session::AccessPolicy for NoPathModify {
    fn version(&self) -> krishiv_plan::analyzer::PolicyVersion {
        krishiv_plan::analyzer::PolicyVersion(1)
    }

    fn check<'a>(
        &'a self,
        _: &'a krishiv_session::Principal,
        securable: &'a krishiv_session::Securable,
        privilege: krishiv_session::Privilege,
    ) -> krishiv_plan::analyzer::BoxFuture<'a, Result<krishiv_session::AccessDecision, KrishivError>>
    {
        let decision = if matches!(securable, krishiv_session::Securable::Path(_))
            && privilege == krishiv_session::Privilege::Modify
        {
            krishiv_session::AccessDecision::Deny {
                reason: "no path modify".to_owned(),
            }
        } else {
            krishiv_session::AccessDecision::Allow
        };
        Box::pin(std::future::ready(Ok(decision)))
    }
}

#[test]
fn a_delta_dml_by_path_needs_modify_on_the_path() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    copy_dir(&fixtures().join("unpartitioned").join("before"), &dir);
    let log = |d: &Path| std::fs::read_dir(d.join("_delta_log")).unwrap().count();
    let before = log(&dir);
    let session = Session::builder()
        .principal(krishiv_session::Principal::new(
            "ana",
            krishiv_session::PrincipalKind::User,
        ))
        .access_policy(std::sync::Arc::new(NoPathModify))
        .build()
        .unwrap();
    let t = format!("delta.`{}`", dir.display());
    for text in [
        format!("DELETE FROM {t} WHERE id = 1"),
        format!("UPDATE {t} SET id = id + 100"),
    ] {
        let e = sql(&session, &text).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_INSUFFICIENT_PERMISSIONS",
            "{text}: {e}"
        );
    }
    assert_eq!(log(&dir), before, "nothing was committed");
    // Reads by path are still allowed.
    sql(&session, &format!("SELECT * FROM {t}")).unwrap();
}
