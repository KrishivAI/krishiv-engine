// SPDX-License-Identifier: Apache-2.0
//! Spark and Krishiv on the same Delta tables (P3-T028b, PR-53b, round trip): Spark 4.2.0 + delta-spark 4.4.1
//! makes the tables, Krishiv reads them and commits on top, Spark reads that and runs DML on top, Krishiv reads
//! Spark's result.
//!
//! Four steps, run in order by `tests/scripts/delta_roundtrip.sh` (Docker, so both Rust steps are `#[ignore]`d):
//!
//! 1. `gen_delta_roundtrip.py create` (Spark) writes `plain`, `part` (partitioned by `x`), `dv` (a deletion
//!    vector) and `cdf` (change data feed and deletion vectors) under `$KRISHIV_DELTA_RT_OUT/delta`;
//! 2. [`krishiv_commits_on_spark_tables`] reads each table (it must read what Spark read), then commits through
//!    [`crate::commit::commit`]: an append whose data files its Parquet handler writes (one file per partition
//!    value), a `DELETE` removing whole files (partition `x = 2` of `part`; on `dv` Spark's files, one of them with
//!    a deletion vector, and Krishiv's own first file; Krishiv's own first file elsewhere) and a second append; after
//!    each commit it records the version and its read (`KRISHIV-RT.json`);
//! 3. `gen_delta_roundtrip.py readback` (Spark) reads Krishiv's result, the history entries of Krishiv's commits,
//!    every Krishiv version by time travel and the change feed of Krishiv's versions on `cdf`, then runs `DELETE`,
//!    `UPDATE` and `MERGE` on top (each doing work on Krishiv's files, a DV on `dv` and `cdf`) and appends;
//! 4. [`krishiv_reads_spark_on_top`] compares: Spark read what Krishiv read at each of its versions, Spark's history
//!    names Krishiv's operations, read versions and isolation level, the change feed holds Krishiv's rows, Spark's
//!    DML metrics are non-zero, and
//!    Krishiv reads Spark's last version (deletion vectors on Krishiv's files included) and its own versions again.
//!
//! The data files are Krishiv's (its Parquet handler, as PR-37a's writer uses it); the writer itself and its
//! statistics are PR-37a part 2's (their own Spark readback is in `tests/fixtures/delta-4.4.1/krishiv-written/`),
//! so the added files here carry `numRecords` only. Every expected value is the other engine's record.

use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use delta_kernel::expressions::{Expression, MapData, Predicate, Scalar, StructData};
use delta_kernel::schema::{DataType, MapType, StructField, StructType};
use krishiv_common::config::Config;
use krishiv_exec::ScanPushdowns;
use serde_json::{Value, json};

use super::oracle_tests::{Env, rows};
use super::*;
use crate::commit::{
    CommitOperation, ReadSet, RetryLimits, StagedCommit, StagedRemoves, commit as delta_commit,
};
use crate::engine::KrishivEngineData;
use crate::engine::eval::create_many;

const OUT_VAR: &str = "KRISHIV_DELTA_RT_OUT";

/// The tables `gen_delta_roundtrip.py` makes, and whether each is partitioned by `x`.
const TABLES: [(&str, bool); 4] = [
    ("plain", false),
    ("part", true),
    ("dv", false),
    ("cdf", false),
];

fn out_dir() -> PathBuf {
    PathBuf::from(
        std::env::var(OUT_VAR).unwrap_or_else(|_| panic!("{OUT_VAR} names the output directory")),
    )
}

fn read_json(name: &str) -> Value {
    let text = std::fs::read_to_string(out_dir().join(name))
        .unwrap_or_else(|e| panic!("{name}: {e} (run the earlier steps first)"));
    serde_json::from_str(&text).unwrap()
}

fn table_dir(name: &str) -> PathBuf {
    out_dir().join("delta").join(name)
}

/// Krishiv's read of the table at `version` (the latest when `None`) through [`DeltaTable`]'s scan, as rows of
/// `[id, x, name]`.
fn krishiv_read(env: &Env, version: Option<u64>) -> Vec<Value> {
    let table = env
        .block_on(DeltaTable::open(
            env.engine.clone(),
            version,
            &Config::new(),
        ))
        .unwrap();
    let source = table.delta_scan().build(ScanPushdowns::default()).unwrap();
    rows(&env.run(source), &table.schema())
        .into_iter()
        .map(Value::from)
        .collect()
}

/// Spark's rows (`{"id", "x", "name"}` objects, ordered by id) as Krishiv's `[id, x, name]` form.
fn spark_rows(v: &Value) -> Vec<Value> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|r| json!([r["id"], r["x"], r["name"]]))
        .collect()
}

fn sorted(mut v: Vec<Value>) -> Vec<Value> {
    v.sort_by_key(std::string::ToString::to_string);
    v
}

/// The kernel's `add_files_schema`.
fn add_schema() -> StructType {
    let stats = DataType::try_struct_type([
        StructField::nullable("numRecords", DataType::LONG),
        StructField::nullable("nullCount", DataType::try_struct_type([]).unwrap()),
        StructField::nullable("minValues", DataType::try_struct_type([]).unwrap()),
        StructField::nullable("maxValues", DataType::try_struct_type([]).unwrap()),
        StructField::nullable("tightBounds", DataType::BOOLEAN),
    ])
    .unwrap();
    StructType::new_unchecked([
        StructField::not_null("path", DataType::STRING),
        StructField::not_null(
            "partitionValues",
            MapType::new(DataType::STRING, DataType::STRING, true),
        ),
        StructField::not_null("size", DataType::LONG),
        StructField::not_null("modificationTime", DataType::LONG),
        StructField::nullable("stats", stats),
    ])
}

/// Rows `ids` with `x = id % 3` (0 for a NULL id) and `name = n<id>`; without `x` when `partitioned` (Delta keeps
/// partition values in the log only).
fn data(ids: &[Option<i64>], partitioned: bool) -> RecordBatch {
    let string = krishiv_types::DataType::String.to_arrow();
    let names: Vec<String> = ids
        .iter()
        .map(|i| format!("n{}", i.unwrap_or(-1)))
        .collect();
    let names: ArrayRef = match string {
        ArrowType::Utf8View => Arc::new(arrow_array::StringViewArray::from(names)),
        _ => Arc::new(arrow_array::StringArray::from(names)),
    };
    let mut fields = vec![Field::new("id", ArrowType::Int64, true)];
    let mut columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(ids.to_vec()))];
    if !partitioned {
        fields.push(Field::new("x", ArrowType::Int32, true));
        columns.push(Arc::new(Int32Array::from(
            ids.iter()
                .map(|i| i.map_or(0, |i| (i % 3) as i32))
                .collect::<Vec<_>>(),
        )));
    }
    fields.push(Field::new("name", string, true));
    columns.push(names);
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

/// Writes `ids` through Krishiv's Parquet handler as one file per partition value, and returns their add rows.
fn write_files(env: &Env, tag: &str, ids: &[Option<i64>], partitioned: bool) -> Vec<RecordBatch> {
    let groups: Vec<(Option<i32>, Vec<Option<i64>>)> = if partitioned {
        (0..3)
            .map(|x| {
                let of: Vec<_> = ids
                    .iter()
                    .copied()
                    .filter(|i| i.map_or(0, |i| (i % 3) as i32) == x)
                    .collect();
                (Some(x), of)
            })
            .filter(|(_, of)| !of.is_empty())
            .collect()
    } else {
        vec![(None, ids.to_vec())]
    };
    let schema = add_schema();
    let stats = schema.field("stats").unwrap().data_type().clone();
    let DataType::Struct(stats_type) = &stats else {
        panic!("stats is a struct");
    };
    groups
        .into_iter()
        .map(|(x, of)| {
            let rel = match x {
                Some(x) => format!("x={x}/part-00000-krishiv-{tag}.snappy.parquet"),
                None => format!("part-00000-krishiv-{tag}.snappy.parquet"),
            };
            let url = env.engine.table_root().join(&rel).unwrap();
            let batch = data(&of, partitioned);
            let size = env
                .block_on(env.engine.delta_blocking(move |e| {
                    use delta_kernel::ParquetHandler as _;
                    e.parquet().write_parquet_file(
                        url,
                        Box::new(std::iter::once(Ok(Box::new(KrishivEngineData::new(batch))
                            as Box<dyn delta_kernel::EngineData>))),
                    )
                }))
                .unwrap()
                .unwrap();
            let values = MapData::try_new(
                MapType::new(DataType::STRING, DataType::STRING, true),
                x.map(|x| {
                    (
                        Scalar::String("x".to_owned()),
                        Scalar::String(x.to_string()),
                    )
                }),
            )
            .unwrap();
            let stats_value = StructData::try_new(
                stats_type.fields().cloned().collect(),
                vec![
                    Scalar::Long(i64::try_from(of.len()).unwrap()),
                    Scalar::Null(DataType::try_struct_type([]).unwrap()),
                    Scalar::Null(DataType::try_struct_type([]).unwrap()),
                    Scalar::Null(DataType::try_struct_type([]).unwrap()),
                    Scalar::Null(DataType::BOOLEAN),
                ],
            )
            .unwrap();
            let now = i64::try_from(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
            )
            .unwrap();
            create_many(
                &schema,
                &[vec![
                    Scalar::String(rel),
                    Scalar::Map(values),
                    Scalar::Long(i64::try_from(size).unwrap()),
                    Scalar::Long(now),
                    Scalar::Struct(stats_value),
                ]],
            )
            .unwrap()
        })
        .collect()
}

/// The scan metadata rows of `table`'s snapshot, with the live files whose path `pick` accepts selected, and how
/// many of the selected files carry a deletion vector.
fn removes(env: &Env, table: &DeltaTable, pick: fn(&str) -> bool) -> (Vec<StagedRemoves>, usize) {
    let snapshot = Arc::clone(table.snapshot());
    env.block_on(env.engine.delta_blocking(move |e| {
        let scan = snapshot.scan_builder().build().unwrap();
        let mut out = Vec::new();
        let mut with_dv = 0;
        for m in scan.scan_metadata(e).unwrap() {
            let (data, selection) = m.unwrap().scan_files.into_parts();
            let batch = KrishivEngineData::from_engine_data(data)
                .unwrap()
                .into_batch();
            let paths = batch.column_by_name("path").unwrap();
            let path = |i: usize| match paths.data_type() {
                ArrowType::Utf8View => paths.as_string_view().value(i).to_owned(),
                _ => paths.as_string::<i32>().value(i).to_owned(),
            };
            let selection: Vec<bool> = (0..batch.num_rows())
                .map(|i| selection.get(i).copied().unwrap_or(true) && pick(&path(i)))
                .collect();
            if let Some(dv) = batch.column_by_name("deletionVector") {
                with_dv += (0..batch.num_rows())
                    .filter(|i| selection[*i] && dv.is_valid(*i))
                    .count();
            }
            if selection.iter().any(|s| *s) {
                out.push(StagedRemoves {
                    scan_rows: batch,
                    selection,
                });
            }
        }
        (out, with_dv)
    }))
    .unwrap()
}

/// Whether `path` is a data file Krishiv wrote here (Spark's names also start with `part-00000-`).
fn krishiv_file(path: &str) -> bool {
    path.rsplit('/')
        .next()
        .is_some_and(|f| f.starts_with("part-00000-krishiv-"))
}

fn staged(
    name: &str,
    operation: CommitOperation,
    parameters: &[(&str, &str)],
    adds: Vec<RecordBatch>,
    removes: Vec<StagedRemoves>,
) -> StagedCommit {
    StagedCommit {
        table: format!("delta.`{name}`"),
        write_id: format!("rt-{name}-{}", uuid_like()),
        idempotency_key: None,
        operation,
        operation_parameters: parameters
            .iter()
            .map(|(k, v)| ((*k).to_owned(), Some((*v).to_owned())))
            .collect(),
        operation_metrics: Vec::new(),
        adds,
        read_set: if removes.is_empty() {
            ReadSet::default()
        } else {
            ReadSet {
                whole_table: true,
                ..ReadSet::default()
            }
        },
        removes,
        dv_updates: Vec::new(),
        dv_descriptors: std::collections::HashMap::new(),
    }
}

fn uuid_like() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}

/// Commits `staged` on the table's latest snapshot; the new version.
fn commit_on(env: &Env, staged: impl FnOnce(&DeltaTable) -> StagedCommit) -> u64 {
    let table = env
        .block_on(DeltaTable::open(env.engine.clone(), None, &Config::new()))
        .unwrap();
    let s = staged(&table);
    let committed = env
        .block_on(delta_commit(
            &env.engine,
            Arc::clone(table.snapshot()),
            s,
            RetryLimits::default(),
        ))
        .unwrap();
    assert_eq!(committed.version, table.version() + 1);
    committed.version
}

/// Step 2: Krishiv reads Spark's tables and commits an append, a whole-file `DELETE` and an append on each.
#[test]
#[ignore = "step 2 of tests/scripts/delta_roundtrip.sh (needs SPARK-CREATE.json from Spark)"]
fn krishiv_commits_on_spark_tables() {
    let spark = read_json("SPARK-CREATE.json");
    assert_eq!(spark["_oracle"]["delta"], "4.4.1");
    let mut record = serde_json::Map::new();
    for (name, partitioned) in TABLES {
        let env = Env::at(table_dir(name));
        let created = &spark[name];
        let read = krishiv_read(&env, None);
        assert_eq!(
            sorted(read.clone()),
            sorted(spark_rows(&created["rows"])),
            "{name}: Krishiv reads Spark's table"
        );
        let mut steps = Vec::new();
        let mut step = |env: &Env, what: &str, version: u64| {
            steps
                .push(json!({ "what": what, "version": version, "rows": krishiv_read(env, None) }));
        };

        // An append of two files (per partition value on `part`), one row with a NULL id.
        let v = commit_on(&env, |_| {
            let mut adds = write_files(
                &env,
                "a1",
                &(100..104).map(Some).collect::<Vec<_>>(),
                partitioned,
            );
            adds.extend(write_files(
                &env,
                "a2",
                &[None, Some(104), Some(105)],
                partitioned,
            ));
            staged(
                name,
                CommitOperation::Write,
                &[("mode", "Append")],
                adds,
                Vec::new(),
            )
        });
        step(&env, "append", v);

        // DELETE by whole files: partition x = 2 of `part` (Spark's files and Krishiv's); on `dv` Spark's files (one
        // carries Spark's deletion vector, and Delta matches a remove by path and DV id) and Krishiv's first file (ids
        // 100 to 103); elsewhere Krishiv's first file.
        let (pick, predicate): (fn(&str) -> bool, &str) = match name {
            "part" => (|p| p.starts_with("x=2/"), "[\"(x = 2)\"]"),
            "dv" => (
                |p| !krishiv_file(p) || p.starts_with("part-00000-krishiv-a1"),
                "[\"((id < 100) OR (id IN (100, 101, 102, 103)))\"]",
            ),
            _ => (
                |p| p.starts_with("part-00000-krishiv-a1"),
                "[\"(id IN (100, 101, 102, 103))\"]",
            ),
        };
        let v = commit_on(&env, |t| {
            let (r, with_dv) = removes(&env, t, pick);
            assert!(!r.is_empty(), "{name}: files to remove");
            assert_eq!(
                with_dv > 0,
                name == "dv",
                "{name}: a removed file carries a deletion vector"
            );
            staged(
                name,
                CommitOperation::Delete,
                &[("predicate", predicate)],
                Vec::new(),
                r,
            )
        });
        step(&env, "delete", v);

        // A second append.
        let v = commit_on(&env, |_| {
            let adds = write_files(&env, "a3", &[Some(110), Some(111)], partitioned);
            staged(
                name,
                CommitOperation::Write,
                &[("mode", "Append")],
                adds,
                Vec::new(),
            )
        });
        step(&env, "append", v);

        record.insert(name.to_owned(), json!({ "read": read, "steps": steps }));
    }
    race_tables(&spark, &mut record);
    std::fs::write(
        out_dir().join("KRISHIV-RT.json"),
        serde_json::to_vec_pretty(&Value::Object(record)).unwrap(),
    )
    .unwrap();
}

/// The concurrent-writer tables (PR-53b, added 2026-11-20): both partitioned by `x`.
const RACE_TABLES: [&str; 2] = ["race_k", "race_s"];

/// A read of partition `x = value` (a statement whose condition is that partition's).
fn reads_x(value: i32) -> ReadSet {
    ReadSet {
        predicates: vec![Arc::new(Predicate::eq(
            Expression::column(["x"]),
            Expression::literal(value),
        ))],
        ..ReadSet::default()
    }
}

/// Commits `staged` on `table`'s snapshot (perhaps not the latest): the new version, or the error.
fn commit_from(env: &Env, table: &DeltaTable, staged: StagedCommit) -> Result<u64, KrishivError> {
    env.block_on(delta_commit(
        &env.engine,
        Arc::clone(table.snapshot()),
        staged,
        RetryLimits::default(),
    ))
    .map(|c| c.version)
}

fn open_at(env: &Env, version: Option<u64>) -> DeltaTable {
    env.block_on(DeltaTable::open(
        env.engine.clone(),
        version,
        &Config::new(),
    ))
    .unwrap()
}

/// The file actions of the rows `removes` selected.
fn file_actions(staged: &[StagedRemoves]) -> Vec<crate::commit::winner::FileAction> {
    let mut out = Vec::new();
    for s in staged {
        let paths = s.scan_rows.column_by_name("path").unwrap();
        for (i, keep) in s.selection.iter().enumerate() {
            if *keep {
                let path = match paths.data_type() {
                    ArrowType::Utf8View => paths.as_string_view().value(i).to_owned(),
                    _ => paths.as_string::<i32>().value(i).to_owned(),
                };
                out.push(crate::commit::winner::FileAction {
                    path,
                    partition_values: crate::write::dml::partition_values_of(&s.scan_rows, i)
                        .unwrap(),
                });
            }
        }
    }
    out
}

/// A whole-file `DELETE` of the files `pick` accepts in `table`'s snapshot, reading partition
/// `x = read` (the condition's partition conjunct). Its read set has the shape the product
/// `DELETE` gives it (`write/dml.rs`: the removed files plus every live file the read predicate
/// selects) and delta-spark's transaction too (its `readFiles` are the candidate files), so a
/// losing `DELETE` meets delete-read before delete-delete, as `ConflictChecker.checkConflicts`
/// orders them (`ConflictChecker.scala:280-319`: `checkForDeletedFilesAgainstCurrentTxnReadFiles`
/// before `checkForDeletedFilesAgainstCurrentTxnDeletedFiles`).
fn delete_files(
    env: &Env,
    name: &str,
    table: &DeltaTable,
    pick: fn(&str) -> bool,
    read: i32,
) -> StagedCommit {
    let (r, _) = removes(env, table, pick);
    assert!(!r.is_empty(), "{name}: files to remove");
    let candidates: fn(&str) -> bool = match read {
        0 => |p| p.starts_with("x=0/"),
        1 => |p| p.starts_with("x=1/"),
        2 => |p| p.starts_with("x=2/"),
        other => panic!("{name}: no candidate files for x = {other}"),
    };
    let (c, _) = removes(env, table, candidates);
    let mut read_files = file_actions(&r);
    for f in file_actions(&c) {
        if !read_files.iter().any(|g| g.path == f.path) {
            read_files.push(f);
        }
    }
    let predicate = format!("[\"(x = {read})\"]");
    let mut s = staged(
        name,
        CommitOperation::Delete,
        &[("predicate", predicate.as_str())],
        Vec::new(),
        r,
    );
    s.read_set = reads_x(read);
    s.read_set.files = read_files;
    s
}

fn append_ids(env: &Env, name: &str, tag: &str, ids: &[i64]) -> StagedCommit {
    let ids: Vec<Option<i64>> = ids.iter().copied().map(Some).collect();
    staged(
        name,
        CommitOperation::Write,
        &[("mode", "Append")],
        write_files(env, tag, &ids, true),
        Vec::new(),
    )
}

/// One race event: what ran, from which version, and how it ended (`version` when it landed,
/// `class` when it conflicted), with Krishiv's read of the latest version after it.
fn event(env: &Env, what: &str, from: u64, outcome: Result<u64, KrishivError>) -> Value {
    let rows = krishiv_read(env, None);
    match outcome {
        Ok(v) => json!({ "what": what, "from": from, "version": v, "rows": rows }),
        Err(e) => json!({
            "what": what,
            "from": from,
            "class": e.error_class().name(),
            "message": e.to_string(),
            "rows": rows,
        }),
    }
}

/// Step 2's concurrent-writer cases. `race_k`: two Krishiv commits staged from one snapshot:
/// appends (both land), a `DELETE` of the same files (the first lands, the second conflicts), a
/// `DELETE` of different partitions (both land). `race_s`: Krishiv commits from the version
/// before Spark's `DELETE` of `x = 0`: an append (lands), a `DELETE` of `x = 0` (conflicts), a
/// `DELETE` of `x = 1` (lands). The landed commits are the table's `steps` (read back by Spark
/// as the other tables' are), every event in `race`.
fn race_tables(spark: &Value, record: &mut serde_json::Map<String, Value>) {
    for name in RACE_TABLES {
        let env = Env::at(table_dir(name));
        let created = &spark[name];
        let latest = created["version"].as_u64().unwrap();
        let mut events = Vec::new();
        if name == "race_k" {
            let s0 = open_at(&env, None);
            let from = s0.version();
            let a = commit_from(&env, &s0, append_ids(&env, name, "k1", &[203, 206]));
            events.push(event(&env, "append", from, a));
            let b = commit_from(&env, &s0, append_ids(&env, name, "k2", &[209, 212]));
            events.push(event(&env, "append", from, b));
            let s1 = open_at(&env, None);
            let from = s1.version();
            let d1 = delete_files(&env, name, &s1, |p| p.starts_with("x=0/"), 0);
            let d2 = delete_files(&env, name, &s1, |p| p.starts_with("x=0/"), 0);
            events.push(event(&env, "delete", from, commit_from(&env, &s1, d1)));
            events.push(event(&env, "delete", from, commit_from(&env, &s1, d2)));
            let s2 = open_at(&env, None);
            let from = s2.version();
            let e1 = delete_files(&env, name, &s2, |p| p.starts_with("x=1/"), 1);
            // Spark's files of x = 2 only: Krishiv's appended rows stay.
            let e2 = delete_files(
                &env,
                name,
                &s2,
                |p| p.starts_with("x=2/") && !krishiv_file(p),
                2,
            );
            events.push(event(&env, "delete", from, commit_from(&env, &s2, e1)));
            events.push(event(&env, "delete", from, commit_from(&env, &s2, e2)));
        } else {
            // Spark's DELETE of x = 0 is `latest`; Krishiv read the version before it.
            let from = latest - 1;
            let s0 = open_at(&env, Some(from));
            let a = commit_from(&env, &s0, append_ids(&env, name, "s1", &[203, 206]));
            events.push(event(&env, "append", from, a));
            let d = delete_files(&env, name, &s0, |p| p.starts_with("x=0/"), 0);
            events.push(event(&env, "delete", from, commit_from(&env, &s0, d)));
            let e = delete_files(&env, name, &s0, |p| p.starts_with("x=1/"), 1);
            events.push(event(&env, "delete", from, commit_from(&env, &s0, e)));
        }
        let steps: Vec<Value> = events
            .iter()
            .filter(|e| e.get("version").is_some())
            .cloned()
            .collect();
        record.insert(
            name.to_owned(),
            json!({ "read": created["rows"], "steps": steps, "race": events }),
        );
    }
}

/// Step 4's checks of the races: the outcomes the cases expect, a conflict leaves the table
/// readable and unchanged, and Spark reads every landed version as Krishiv read it.
fn check_races(created: &Value, krishiv: &Value, spark: &Value) {
    for name in RACE_TABLES {
        let env = Env::at(table_dir(name));
        let k = &krishiv[name];
        let s = &spark[name];
        let events = k["race"].as_array().unwrap();
        let outcomes: Vec<&str> = events
            .iter()
            .map(|e| {
                if e.get("version").is_some() {
                    "lands"
                } else {
                    "conflicts"
                }
            })
            .collect();
        let expected: &[&str] = if name == "race_k" {
            &["lands", "lands", "lands", "conflicts", "lands", "lands"]
        } else {
            &["lands", "conflicts", "lands"]
        };
        assert_eq!(outcomes, expected, "{name}: outcomes {events:#?}");
        let mut latest = created[name]["version"].as_u64().unwrap();
        let mut rows = spark_rows(&created[name]["rows"]);
        for e in events {
            if let Some(v) = e["version"].as_u64() {
                assert_eq!(v, latest + 1, "{name}: {} lands next", e["what"]);
                latest = v;
                let key = v.to_string();
                assert_eq!(
                    sorted(spark_rows(&s["at"][&key])),
                    sorted(e["rows"].as_array().unwrap().clone()),
                    "{name}: Spark at Krishiv's version {v}"
                );
                let h = &s["history"][&key];
                let op = if e["what"] == "delete" {
                    "DELETE"
                } else {
                    "WRITE"
                };
                assert_eq!(h["operation"], op, "{name}: operation of {v}");
                assert_eq!(
                    h["isolationLevel"], "Serializable",
                    "{name}: isolation of {v}"
                );
                // The snapshot the commit read, also after a retry over a winner (delta-spark's
                // `readVersion` is the transaction's snapshot, `OptimisticTransaction.scala`).
                assert_eq!(h["readVersion"], e["from"], "{name}: readVersion of {v}");
            } else {
                // Each conflict is a DELETE of x = 0 that read the files a winner removed: on
                // race_k Krishiv's first DELETE, on race_s Spark's. Delete-read comes before
                // delete-delete in the checker's order (see `delete_files`).
                let class = e["class"].as_str().unwrap();
                let message = e["message"].as_str().unwrap();
                assert_eq!(
                    class, "DELTA_CONCURRENT_DELETE_READ.WITH_PARTITION_HINT",
                    "{name}: conflict class: {message}"
                );
                assert!(
                    message.contains("[x=0]"),
                    "{name}: the hint names partition x = 0: {message}"
                );
                assert_eq!(
                    sorted(e["rows"].as_array().unwrap().clone()),
                    sorted(rows.clone()),
                    "{name}: a conflict changes nothing and the table stays readable"
                );
            }
            rows = e["rows"].as_array().unwrap().clone();
        }
        // Spark's DELETE of x = 0 (race_s) and Krishiv's deletes leave only rows of x = 2:
        // Krishiv's appends, and on race_s Spark's x = 2 rows.
        let mut left: Vec<i64> = rows.iter().map(|r| r[0].as_i64().unwrap()).collect();
        left.sort_unstable();
        let want: Vec<i64> = if name == "race_k" {
            vec![203, 206, 209, 212]
        } else {
            vec![2, 5, 8, 11, 203, 206]
        };
        assert_eq!(left, want, "{name}: rows left");
        assert_eq!(
            s["before"]["version"].as_u64(),
            Some(latest),
            "{name}: Spark's version"
        );
        assert_eq!(
            sorted(spark_rows(&s["before"]["rows"])),
            sorted(rows),
            "{name}: Spark reads the races' result"
        );
        assert_eq!(
            sorted(krishiv_read(&env, None)),
            sorted(spark_rows(&s["after"]["rows"])),
            "{name}: Krishiv reads Spark's result"
        );
    }
}

/// Step 4: Spark read what Krishiv committed, and Krishiv reads Spark's DML on top.
#[test]
#[ignore = "step 4 of tests/scripts/delta_roundtrip.sh (needs SPARK-READBACK.json from Spark)"]
fn krishiv_reads_spark_on_top() {
    let created = read_json("SPARK-CREATE.json");
    let krishiv = read_json("KRISHIV-RT.json");
    let spark = read_json("SPARK-READBACK.json");
    check_races(&created, &krishiv, &spark);
    for (name, _) in TABLES {
        let env = Env::at(table_dir(name));
        let k = &krishiv[name];
        let s = &spark[name];
        let steps = k["steps"].as_array().unwrap();
        let last = steps.last().unwrap();
        // Versions: Krishiv's commits follow Spark's directly.
        let first = created[name]["version"].as_u64().unwrap();
        let versions: Vec<u64> = steps
            .iter()
            .map(|st| st["version"].as_u64().unwrap())
            .collect();
        assert_eq!(
            versions,
            (first + 1..=first + 3).collect::<Vec<_>>(),
            "{name}: Krishiv's versions"
        );
        assert_eq!(
            s["before"]["version"], last["version"],
            "{name}: Spark sees Krishiv's last version"
        );
        assert_eq!(
            s["before"]["protocol"], created[name]["protocol"],
            "{name}: Krishiv's commits leave the protocol alone"
        );
        assert_eq!(
            sorted(spark_rows(&s["before"]["rows"])),
            sorted(last["rows"].as_array().unwrap().clone()),
            "{name}: Spark reads Krishiv's result"
        );
        for st in steps {
            let v = st["version"].to_string();
            // Spark's time travel to each Krishiv version answers Krishiv's read of it.
            assert_eq!(
                sorted(spark_rows(&s["at"][&v])),
                sorted(st["rows"].as_array().unwrap().clone()),
                "{name}: Spark at Krishiv's version {v}"
            );
            // Spark's history entry of Krishiv's commit.
            let h = &s["history"][&v];
            // Known difference: delta-spark writes `isBlindAppend: false` on a commit that is not a blind append;
            // delta_kernel 0.30.1 owns the field (it replaces a connector's value) and writes it only when true, so
            // Spark's history shows NULL for Krishiv's DELETE. This check fails once the kernel writes `false`.
            let (op, blind) = if st["what"] == "delete" {
                ("DELETE", Value::Null)
            } else {
                ("WRITE", Value::Bool(true))
            };
            assert_eq!(h["operation"], op, "{name}: operation of {v}");
            assert_eq!(h["isBlindAppend"], blind, "{name}: isBlindAppend of {v}");
            assert_eq!(
                h["isolationLevel"], "Serializable",
                "{name}: isolation of {v}"
            );
            assert_eq!(
                h["readVersion"].as_u64(),
                Some(st["version"].as_u64().unwrap() - 1),
                "{name}: readVersion of {v}"
            );
            // Krishiv reads its own versions back as it read them then.
            let at = st["version"].as_u64().unwrap();
            assert_eq!(
                sorted(krishiv_read(&env, Some(at))),
                sorted(st["rows"].as_array().unwrap().clone()),
                "{name}: Krishiv at its version {v}"
            );
        }
        // The change feed of Krishiv's versions: its appends as inserts, the removed file's rows as deletes.
        if name == "cdf" {
            let changes = s["changes"].as_array().unwrap();
            let of = |kind: &str, v: u64| -> Vec<Value> {
                sorted(
                    changes
                        .iter()
                        .filter(|c| {
                            c["_change_type"] == kind && c["_commit_version"].as_u64() == Some(v)
                        })
                        .map(|c| json!([c["id"], c["x"], c["name"]]))
                        .collect(),
                )
            };
            let rows_at = |i: usize| -> Vec<Value> { steps[i]["rows"].as_array().unwrap().clone() };
            let before_first: Vec<Value> = k["read"].as_array().unwrap().clone();
            let diff = |a: &[Value], b: &[Value]| -> Vec<Value> {
                sorted(a.iter().filter(|r| !b.contains(r)).cloned().collect())
            };
            assert_eq!(
                of("insert", versions[0]),
                diff(&rows_at(0), &before_first),
                "cdf: first append"
            );
            assert_eq!(
                of("delete", versions[1]),
                diff(&rows_at(0), &rows_at(1)),
                "cdf: delete"
            );
            assert_eq!(
                of("insert", versions[2]),
                diff(&rows_at(2), &rows_at(1)),
                "cdf: second append"
            );
        }
        // Spark's DML did work on Krishiv's files: each commit's own metrics (delta-spark's names) are non-zero.
        let dml = s["dml"].as_object().unwrap();
        let metric = |op: &str, key: &str| -> u64 {
            let found: Vec<u64> = dml
                .values()
                .filter(|h| h["operation"] == op)
                .map(|h| {
                    h["operationMetrics"][key]
                        .as_str()
                        .and_then(|m| m.parse().ok())
                        .unwrap_or(0)
                })
                .collect();
            assert_eq!(found.len(), 1, "{name}: one {op} commit of Spark's");
            found[0]
        };
        let mut expected = vec![
            ("DELETE", "numDeletedRows"),
            ("UPDATE", "numUpdatedRows"),
            ("MERGE", "numTargetRowsDeleted"),
            ("MERGE", "numTargetRowsInserted"),
        ];
        if name == "dv" || name == "cdf" {
            expected.push(("DELETE", "numDeletionVectorsAdded"));
        }
        for (op, key) in expected {
            assert!(metric(op, key) > 0, "{name}: Spark's {op} {key}");
        }
        // Krishiv reads Spark's DML on top of its commits.
        assert_eq!(
            sorted(krishiv_read(&env, None)),
            sorted(spark_rows(&s["after"]["rows"])),
            "{name}: Krishiv reads Spark's result"
        );
    }
}
