// SPDX-License-Identifier: Apache-2.0
//! Delta write options and the kernel's gaps through the session (feature `delta`, P3-T021
//! PR-37e, design sections 3.7, 4.1-4.2, 6.2-6.3):
//!
//! - `delta_write_options_match_spark`: every option case of
//!   `tests/fixtures/delta-4.4.1/OPTIONS.json` (Spark 4.2.0 + delta-spark 4.4.1,
//!   `gen_delta_fixtures.py /out options`) as DataFrame options and as `INSERT .. WITH (..)`:
//!   the same error, or the same commit parameters and `userMetadata`; the options Krishiv
//!   refuses by name (`REFUSED_OPTIONS`) and the one recorded class difference;
//! - `delta_no_internal_error_on_oracle_tables`: every oracle table under
//!   `tests/fixtures/delta-4.4.1/` read, and every write and DML statement kind tried on its own
//!   copy, never raises `INTERNAL_ERROR`, and every refusal carries a documented class;
//! - `no_coordinated_commits`: a table with coordinated commits (the writer feature, or a
//!   `delta.coordinatedCommits.*` key) refuses every read, write and DML statement with
//!   `UNSUPPORTED_FEATURE.TABLE_OPERATION` before anything is written;
//! - `delta_kernel_gaps_are_refused_by_name`: what the kernel cannot do (write-time type
//!   widening, writes to legacy writer 3-6 tables, row tracking, Variant writes, data-column
//!   `replaceWhere`) is refused with its class before anything is written; a table whose files
//!   carry no statistics takes appends past a checkpoint and row-level changes.
#![cfg(test)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use futures::executor::block_on;
use krishiv_common::error::KrishivError;
use krishiv_session::{DataFrame, Session};
use serde_json::{Value, json};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/delta-4.4.1")
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

fn run(s: &Session, sql: &str) -> Result<(), KrishivError> {
    block_on(async { s.sql(sql).await?.collect().await }).map(drop)
}

fn df(s: &Session, sql: &str) -> DataFrame {
    block_on(s.sql(sql)).unwrap()
}

fn count(s: &Session, sql: &str) -> i64 {
    use arrow_array::cast::AsArray;
    let batches = block_on(df(s, sql).collect()).unwrap();
    batches[0]
        .column(0)
        .as_primitive::<arrow_array::types::Int64Type>()
        .value(0)
}

/// The JSON commit versions under `dir`.
fn versions(dir: &Path) -> usize {
    std::fs::read_dir(dir.join("_delta_log"))
        .unwrap()
        .filter(|e| {
            let n = e.as_ref().unwrap().file_name().into_string().unwrap();
            n.strip_suffix(".json")
                .is_some_and(|v| v.parse::<u64>().is_ok())
        })
        .count()
}

/// The commitInfo of the latest commit under `dir`.
fn last_commit_info(dir: &Path) -> Value {
    let v = versions(dir) - 1;
    std::fs::read_to_string(dir.join(format!("_delta_log/{v:020}.json")))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .find_map(|a| a.get("commitInfo").cloned())
        .unwrap()
}

/// The files under `dir` outside the log (data and deletion vector files), by relative path.
fn data_files(dir: &Path) -> BTreeSet<PathBuf> {
    let mut out = BTreeSet::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                if !p.ends_with("_delta_log") {
                    stack.push(p);
                }
            } else if p.extension().is_some_and(|x| x == "parquet" || x == "bin") {
                out.insert(p.strip_prefix(dir).unwrap().to_path_buf());
            }
        }
    }
    out
}

fn options_oracle() -> Value {
    serde_json::from_slice(&std::fs::read(fixtures().join("OPTIONS.json")).unwrap()).unwrap()
}

/// Options delta-spark commits with and Krishiv refuses by name (`KRISHIV_UNSUPPORTED_FEATURE`,
/// recorded in `docs/concepts/known-differences.md`): idempotent writes, `dataChange = false`,
/// `replaceUsing`, `writePartitionColumns`.
const REFUSED_OPTIONS: &[&str] = &[
    "txn_both",
    "data_change_false_append",
    "replace_using",
    "write_partition_columns",
];

/// The oracle's commit parameters as the kernel writes them: every value a string (a recorded
/// difference, as for `isDynamicPartitionOverwrite`).
fn as_strings(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, v)| {
                    let s = match v {
                        Value::String(s) => Value::from(s.clone()),
                        other => Value::from(other.to_string()),
                    };
                    (k.clone(), s)
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Our outcome against the oracle's record of one case on the table at `dir`.
fn compare_option_case(name: &str, ours: Result<(), KrishivError>, theirs: &Value, dir: &Path) {
    let after = versions(dir);
    if REFUSED_OPTIONS.contains(&name) {
        assert!(theirs["error"].is_null(), "{name}: delta-spark commits it");
        let e = ours.expect_err(name);
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_UNSUPPORTED_FEATURE",
            "{name}: {e}"
        );
        assert_eq!(after, 1, "{name}: nothing committed");
        return;
    }
    if name == "max_records_bad" {
        // delta-spark fails with a classless `NumberFormatException`; Krishiv names the option
        // (a recorded difference).
        assert!(theirs["error"]["condition"].is_null(), "{name}");
        let e = ours.expect_err(name);
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_INVALID_FORMAT_OPTION_VALUE",
            "{name}: {e}"
        );
        assert_eq!(after, 1, "{name}");
        return;
    }
    match (&ours, theirs["error"].is_null()) {
        (Err(e), false) => {
            let want = &theirs["error"];
            assert_eq!(e.error_class().name(), want["condition"], "{name}: {e}");
            assert_eq!(
                e.to_string()
                    .lines()
                    .next()
                    .unwrap()
                    .split(" SQLSTATE")
                    .next(),
                want["message"].as_str(),
                "{name}"
            );
            assert_eq!(after, 1, "{name}: nothing committed");
        }
        (Ok(()), true) => {
            assert_eq!(after, 2, "{name}");
            let info = last_commit_info(dir);
            let want = &theirs["commit"];
            assert_eq!(info["operation"], want["operation"], "{name}");
            assert_eq!(
                info["operationParameters"],
                as_strings(&want["operationParameters"]),
                "{name}"
            );
            assert_eq!(
                info.get("userMetadata").cloned().unwrap_or(Value::Null),
                want["userMetadata"],
                "{name}"
            );
        }
        (ours, _) => panic!("{name}: {ours:?} against {}", theirs["error"]),
    }
}

/// A fresh table of two rows (`id`, `p`) at `dir`, written by Krishiv as the oracle wrote its own.
fn two_rows(s: &Session, dir: &Path, partitioned: bool) {
    let w = df(s, "SELECT id, CAST(id % 2 AS INT) AS p FROM range(2)")
        .write()
        .format("delta");
    let w = if partitioned {
        w.partition_by(["p"])
    } else {
        w
    };
    block_on(w.save(&dir.to_string_lossy())).unwrap();
}

#[test]
fn delta_write_options_match_spark() {
    let oracle = options_oracle();
    let mut checked = 0;
    for (name, case) in oracle["cases"].as_object().unwrap() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("t");
        let s = Session::new().unwrap();
        two_rows(&s, &dir, case["partitioned"].as_bool().unwrap());
        let mut w = df(&s, "SELECT id, CAST(id % 2 AS INT) AS p FROM range(10, 12)")
            .write()
            .format("delta")
            .mode(case["mode"].as_str().unwrap());
        for (k, v) in case["options"].as_object().unwrap() {
            w = w.option(k, v.as_str().unwrap());
        }
        let ours = block_on(w.save(&dir.to_string_lossy()));
        compare_option_case(name, ours, case, &dir);
        checked += 1;
    }
    for (name, case) in oracle["statements"].as_object().unwrap() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("t");
        let s = Session::new().unwrap();
        let sql = case["sql"].as_str().unwrap();
        two_rows(&s, &dir, sql.contains("<pt>"));
        run(
            &s,
            &format!("CREATE TABLE t USING delta LOCATION '{}'", dir.display()),
        )
        .unwrap();
        let ours = run(&s, &sql.replace("<t>", "t").replace("<pt>", "t"));
        if name.starts_with("delete") || name.starts_with("update") {
            // Spark 4.2.0 has no `WITH (..)` on `DELETE`/`UPDATE`, nor has Krishiv's parser
            // (the 4.3 syntax is not parsed yet).
            let e = ours.expect_err(name);
            assert_eq!(
                e.error_class().name(),
                case["error"]["condition"],
                "{name}: {e}"
            );
            assert_eq!(versions(&dir), 1, "{name}");
        } else {
            compare_option_case(name, ours, case, &dir);
        }
        checked += 1;
    }
    assert!(checked >= 30, "{checked}");
}

/// Every Delta table under the fixtures: a directory holding `_delta_log`.
fn oracle_tables() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![fixtures()];
    while let Some(d) = stack.pop() {
        if d.join("_delta_log").is_dir() {
            out.push(d);
            continue;
        }
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            }
        }
    }
    out.sort();
    out
}

/// A statement kind tried on a table: SQL over `delta.`<path>`` (`<t>`) whose first column is
/// `<c>`, an `INSERT` of one of its rows into it as a catalog table (Krishiv's `INSERT` takes no
/// path table), or a DataFrame write of one of its rows with a save mode and options.
enum Kind {
    Sql(&'static str),
    Insert,
    Write(&'static str, &'static [(&'static str, &'static str)]),
}

const KINDS: &[(&str, Kind)] = &[
    ("append", Kind::Write("append", &[])),
    ("overwrite", Kind::Write("overwrite", &[])),
    (
        "dynamic overwrite",
        Kind::Write("overwrite", &[("partitionOverwriteMode", "dynamic")]),
    ),
    (
        "replaceWhere",
        Kind::Write("overwrite", &[("replaceWhere", "<c> IS NULL")]),
    ),
    ("insert", Kind::Insert),
    (
        "delete rows",
        Kind::Sql("DELETE FROM <t> WHERE <c> IS NOT NULL"),
    ),
    ("delete all", Kind::Sql("DELETE FROM <t>")),
    (
        "update",
        Kind::Sql("UPDATE <t> SET <c> = <c> WHERE <c> IS NOT NULL"),
    ),
    (
        "merge upsert",
        Kind::Sql(
            "MERGE INTO <t> AS t USING (SELECT * FROM <t> LIMIT 1) AS s ON t.<c> <=> s.<c> \
             WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *",
        ),
    ),
    (
        "merge delete",
        Kind::Sql(
            "MERGE INTO <t> AS t USING (SELECT * FROM <t> LIMIT 1) AS s ON t.<c> <=> s.<c> \
             WHEN MATCHED THEN DELETE",
        ),
    ),
    (
        "merge insert",
        Kind::Sql(
            "MERGE INTO <t> AS t USING (SELECT * FROM <t> LIMIT 1) AS s ON false \
             WHEN NOT MATCHED THEN INSERT *",
        ),
    ),
];

/// The classes a refusal of a read, write or DML statement on a Delta table may carry: the
/// design's (sections 3.5-3.7, 4.1-4.2, 6.2), delta-spark's for what it refuses too, and the
/// engine's own for what Krishiv refuses by name.
const DOCUMENTED: &[&str] = &[
    "DELTA_UNSUPPORTED_FEATURES_FOR_READ",
    "DELTA_UNSUPPORTED_FEATURES_FOR_WRITE",
    "DELTA_INVALID_PROTOCOL_VERSION",
    "DELTA_CANNOT_MODIFY_APPEND_ONLY",
    "DELTA_MULTIPLE_SOURCE_ROW_MATCHING_TARGET_ROW_IN_MERGE",
    "DELTA_REPLACE_WHERE_MISMATCH",
    "UNSUPPORTED_FEATURE.TABLE_OPERATION",
    "KRISHIV_UNSUPPORTED_FEATURE",
    "TASK_WRITE_FAILED",
];

/// `kind` on the table at `dir` (one column `first`).
fn try_kind(s: &Session, dir: &Path, first: &str, kind: &Kind) -> Result<(), KrishivError> {
    let path = dir.to_string_lossy().into_owned();
    let table = format!("delta.`{path}`");
    let fill = |t: &str| t.replace("<t>", &table).replace("<c>", first);
    match kind {
        Kind::Sql(text) => run(s, &fill(text)),
        Kind::Insert => {
            run(
                s,
                &format!("CREATE TABLE ins USING delta LOCATION '{path}'"),
            )?;
            run(s, "INSERT INTO ins SELECT * FROM ins LIMIT 1")
        }
        Kind::Write(mode, options) => block_on(async {
            let mut w = s
                .sql(&format!("SELECT * FROM {table} LIMIT 1"))
                .await?
                .write()
                .format("delta")
                .mode(mode);
            for (k, v) in *options {
                w = w.option(k, &fill(v));
            }
            w.save(&path).await
        }),
    }
}

/// Every oracle table read, then each statement kind on its own copy: no `INTERNAL_ERROR`
/// (design section 3.7: every user-reachable refusal is named), only documented classes.
#[test]
fn delta_no_internal_error_on_oracle_tables() {
    let mut tried = 0;
    let mut seen = BTreeSet::new();
    for table in oracle_tables() {
        let name = table
            .strip_prefix(fixtures())
            .unwrap()
            .display()
            .to_string();
        let s = Session::new().unwrap();
        let read = block_on(async {
            let df = s
                .sql(&format!("SELECT * FROM delta.`{}`", table.display()))
                .await?;
            let schema = df.schema().await?;
            df.collect().await?;
            Ok::<_, KrishivError>(schema)
        });
        let first = match &read {
            Ok(schema) => format!("`{}`", schema.field(0).name().replace('`', "``")),
            Err(e) => {
                assert!(
                    DOCUMENTED.contains(&e.error_class().name()),
                    "{name} read: {e}"
                );
                seen.insert(e.error_class().name().to_owned());
                "id".to_owned()
            }
        };
        for (what, kind) in KINDS {
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path().join("t");
            copy_dir(&table, &dir);
            let s = Session::new().unwrap();
            if let Err(e) = try_kind(&s, &dir, &first, kind) {
                let class = e.error_class().name();
                assert_ne!(class, "INTERNAL_ERROR", "{name} {what}: {e}");
                // Analysis errors of the generic statement on an odd table (a column type
                // the statement cannot assign) are the analyzer's own named classes.
                let analysis = e.sql_state().is_some_and(|s| s.starts_with("42"));
                assert!(
                    DOCUMENTED.contains(&class) || analysis,
                    "{name} {what}: {e}"
                );
                seen.insert(class.to_owned());
            }
            tried += 1;
        }
    }
    assert!(tried >= 80 * KINDS.len(), "{tried}");
    // The refusals the fixtures exercise.
    for class in [
        "DELTA_UNSUPPORTED_FEATURES_FOR_WRITE",
        "UNSUPPORTED_FEATURE.TABLE_OPERATION",
        "DELTA_CANNOT_MODIFY_APPEND_ONLY",
    ] {
        assert!(seen.contains(class), "{class} not seen: {seen:?}");
    }
}

/// `actions` appended to a copy of the oracle table `name` as its next commit.
fn with_commit(name: &str, actions: &[Value]) -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    copy_dir(&fixtures().join(name), &dir);
    let next = versions(&dir);
    let text: String = actions.iter().map(|a| format!("{a}\n")).collect();
    std::fs::write(dir.join(format!("_delta_log/{next:020}.json")), text).unwrap();
    (tmp, dir)
}

/// The first `metaData` action of the oracle table `name`.
fn metadata_of(name: &str) -> Value {
    std::fs::read_to_string(
        fixtures()
            .join(name)
            .join("_delta_log/00000000000000000000.json"),
    )
    .unwrap()
    .lines()
    .map(|l| serde_json::from_str::<Value>(l).unwrap())
    .find(|a| a.get("metaData").is_some())
    .unwrap()
}

/// A table with coordinated commits refuses every statement (design sections 3.5-3.6):
/// `UNSUPPORTED_FEATURE.TABLE_OPERATION`, no commit, no file.
#[test]
fn no_coordinated_commits() {
    let base = "dml/partitioned/before";
    let mut by_config = metadata_of(base);
    by_config["metaData"]["configuration"]["delta.coordinatedCommits.commitCoordinator-preview"] =
        json!("dynamodb");
    let tables = [
        ("configuration", with_commit(base, &[by_config])),
        (
            "writer feature",
            with_commit(
                base,
                &[json!({ "protocol": {
                    "minReaderVersion": 1,
                    "minWriterVersion": 7,
                    "writerFeatures": ["appendOnly", "invariants", "coordinatedCommits-preview"],
                }})],
            ),
        ),
    ];
    for (form, (_tmp, dir)) in &tables {
        let s = Session::new().unwrap();
        let e = run(&s, &format!("SELECT * FROM delta.`{}`", dir.display())).unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "UNSUPPORTED_FEATURE.TABLE_OPERATION",
            "{form}: {e}"
        );
        assert!(e.to_string().contains("coordinated commits"), "{form}: {e}");
        let before = (versions(dir), data_files(dir));
        for (what, kind) in KINDS {
            let e = try_kind(&s, dir, "id", kind).unwrap_err();
            assert_eq!(
                e.error_class().name(),
                "UNSUPPORTED_FEATURE.TABLE_OPERATION",
                "{form} {what}: {e}"
            );
            assert_eq!((versions(dir), data_files(dir)), before, "{form} {what}");
        }
    }
}

/// What the kernel cannot do is refused by name before anything is written (design sections
/// 4.1-4.2); a table without file statistics takes appends past a checkpoint and changes.
#[test]
fn delta_kernel_gaps_are_refused_by_name() {
    struct Gap {
        what: &'static str,
        table: &'static str,
        sql: &'static str,
        options: &'static [(&'static str, &'static str)],
        class: &'static str,
    }
    let gaps = [
        Gap {
            // The kernel has no write-time type widening; delta-spark would merge the types.
            what: "type widening on write",
            table: "dml/unpartitioned/before",
            sql: "SELECT id, v, CAST(n AS BIGINT) AS n FROM <t>",
            options: &[("mergeSchema", "true")],
            class: "KRISHIV_UNSUPPORTED_FEATURE",
        },
        Gap {
            // Legacy writer 3-6: the kernel infers and refuses `checkConstraints`.
            what: "a legacy (1, 3) table",
            table: "write-features/legacy-1-3",
            sql: "SELECT * FROM <t>",
            options: &[],
            class: "DELTA_INVALID_PROTOCOL_VERSION",
        },
        Gap {
            what: "row tracking",
            table: "write-features/row-tracking",
            sql: "SELECT * FROM <t>",
            options: &[],
            class: "DELTA_UNSUPPORTED_FEATURES_FOR_WRITE",
        },
        Gap {
            what: "a Variant write",
            table: "variant",
            sql: "SELECT * FROM <t>",
            options: &[],
            class: "DELTA_UNSUPPORTED_FEATURES_FOR_WRITE",
        },
        Gap {
            what: "replaceWhere on a data column",
            table: "dml/partitioned/before",
            sql: "SELECT * FROM <t> WHERE id = 1",
            options: &[("replaceWhere", "id = 1")],
            class: "UNSUPPORTED_FEATURE.TABLE_OPERATION",
        },
    ];
    for gap in &gaps {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("t");
        copy_dir(&fixtures().join(gap.table), &dir);
        let path = dir.to_string_lossy().into_owned();
        let s = Session::new().unwrap();
        let before = (versions(&dir), data_files(&dir));
        let mode = if gap.options.iter().any(|(k, _)| *k == "replaceWhere") {
            "overwrite"
        } else {
            "append"
        };
        let e = block_on(async {
            let mut w = s
                .sql(&gap.sql.replace("<t>", &format!("delta.`{path}`")))
                .await?
                .write()
                .format("delta")
                .mode(mode);
            for (k, v) in gap.options {
                w = w.option(k, v);
            }
            w.save(&path).await
        })
        .unwrap_err();
        assert_eq!(e.error_class().name(), gap.class, "{}: {e}", gap.what);
        assert_eq!((versions(&dir), data_files(&dir)), before, "{}", gap.what);
    }

    // Files without statistics (delta-spark with `stats.collect = false`): appends past the
    // checkpoint interval (10) write a checkpoint over them, then row-level changes run.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    copy_dir(&fixtures().join("dvs-no-stats/no-stats/before"), &dir);
    let table = format!("delta.`{}`", dir.display());
    let s = Session::new().unwrap();
    let rows = count(&s, &format!("SELECT count(*) FROM {table}"));
    // Krishiv's `INSERT` takes catalog tables only (delta-spark also takes `delta.`path``).
    run(
        &s,
        &format!(
            "CREATE TABLE nostats USING delta LOCATION '{}'",
            dir.display()
        ),
    )
    .unwrap();
    for i in 0..10 {
        run(
            &s,
            &format!("INSERT INTO nostats VALUES ({}, 'x', 1)", 1000 + i),
        )
        .unwrap();
    }
    let last = dir.join("_delta_log/_last_checkpoint");
    for _ in 0..200 {
        if last.exists() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(last.exists(), "no checkpoint");
    let s = Session::new().unwrap();
    assert_eq!(
        count(&s, &format!("SELECT count(*) FROM {table}")),
        rows + 10
    );
    run(&s, &format!("DELETE FROM {table} WHERE id >= 1005")).unwrap();
    run(&s, &format!("UPDATE {table} SET n = 2 WHERE id = 1000")).unwrap();
    assert_eq!(
        count(&s, &format!("SELECT count(*) FROM {table}")),
        rows + 5
    );
    assert_eq!(
        count(
            &s,
            &format!("SELECT count(*) FROM {table} WHERE n = 2 AND id >= 1000")
        ),
        1
    );
}
