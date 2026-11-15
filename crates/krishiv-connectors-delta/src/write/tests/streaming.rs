// SPDX-License-Identifier: Apache-2.0
//! The streaming writer under memory pressure, exact overwrite removes, the create's landed
//! check, failed writes, post-commit checkpoints and `delta_no_internal_error_on_oracle_tables`.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Date32Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use delta_kernel::expressions::{Expression as KExpr, Predicate as Pred, Scalar};
use krishiv_common::error::ErrorClass;
use krishiv_exec::planner::WriteMode;
use krishiv_plan::{Expr, Literal};

use super::super::NewDeltaTable;
use super::super::sink::{Planned, commit_create, plan_create, select_removes, take_create};
use super::{Env, adds, ids, modes_schema};
use crate::write::files::FileLimits;

/// The data files under `dir` (not the log).
fn data_files(dir: &Path) -> usize {
    let mut n = 0;
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if entry.file_type().unwrap().is_dir() {
            if entry.file_name() != "_delta_log" {
                n += data_files(&path);
            }
        } else if path.extension().is_some_and(|e| e == "parquet") {
            n += 1;
        }
    }
    n
}

/// `n` rows of `(id, v, p)` from `start`, `p` alternating `x`/`y`, `v` 64 characters.
fn wide_rows(start: i64, n: usize) -> RecordBatch {
    let ids: Vec<i64> = (start..).take(n).collect();
    let v: ArrayRef = Arc::new(StringArray::from_iter_values(
        ids.iter().map(|i| format!("{:064}", i.wrapping_mul(7919))),
    ));
    let id: ArrayRef = Arc::new(Int64Array::from(ids));
    let p: ArrayRef = Arc::new(StringArray::from_iter_values(
        (0..n).map(|i| if i % 2 == 0 { "x" } else { "y" }),
    ));
    RecordBatch::try_new(modes_schema(), vec![id, v, p]).unwrap()
}

fn new_table(env: &Env, schema: &SchemaRef, partitions: &[&str]) -> NewDeltaTable {
    NewDeltaTable::new(
        env.engine.clone(),
        schema,
        partitions.iter().map(|p| (*p).to_owned()).collect(),
        BTreeMap::new(),
        None,
    )
    .unwrap()
}

/// A write several times the query's memory streams: rolled files in each partition, the
/// pool's `NeedMemory` met by the driver (whose grant lands on the sink's own reservation) and
/// by spilling the buffered row groups, and the pool empty at the end.
#[test]
fn a_write_larger_than_the_pool_streams_into_rolled_files() {
    const LIMIT: usize = 12 << 20;
    let env = Env::with_limit(LIMIT);
    let table = new_table(&env, &modes_schema(), &["p"]);
    let batches: Vec<RecordBatch> = (0..120).map(|b| wide_rows(b * 8192, 8192)).collect();
    let bytes: usize = batches.iter().map(RecordBatch::get_array_memory_size).sum();
    assert!(
        bytes > 2 * LIMIT,
        "the input ({bytes} bytes) is larger than the pool"
    );
    let rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    env.write(
        &table,
        WriteMode::Append,
        batches,
        &[("maxRecordsPerFile", "100000")],
    )
    .unwrap();
    let metrics = env.metrics.lock().unwrap().clone().unwrap();
    let sink = metrics
        .operators
        .values()
        .find(|e| e.name.contains("DeltaWrite"))
        .unwrap_or_else(|| {
            panic!(
                "{:?}",
                metrics
                    .operators
                    .values()
                    .map(|e| &e.name)
                    .collect::<Vec<_>>()
            )
        });
    // `NeedMemory` was met by spilling the sink (its buffered row groups shipped) and the
    // retried batch drew the grant: the write finished within the pool.
    assert!(sink.metrics.spills > 0, "{:?}", sink.metrics);
    assert!(
        sink.metrics.peak_reserved <= LIMIT as u64,
        "{}",
        sink.metrics.peak_reserved
    );
    let files = adds(&env.log(0));
    for p in ["x", "y"] {
        let n = files
            .iter()
            .filter(|a| a["partitionValues"]["p"] == p)
            .count();
        assert!(n >= 2, "partition {p}: {n} files");
    }
    for a in &files {
        let stats: serde_json::Value = serde_json::from_str(a["stats"].as_str().unwrap()).unwrap();
        assert!(stats["numRecords"].as_i64().unwrap() <= 100_000, "{stats}");
    }
    let back = env.read(&env.open().unwrap());
    assert_eq!(ids(&back).len(), rows);
}

/// A batch the pool can never hold fails with the memory class (no endless `NeedMemory`), and
/// nothing is left behind.
#[test]
fn a_batch_the_pool_cannot_hold_fails_with_the_memory_class() {
    let env = Env::with_limit(256 << 10);
    let table = new_table(&env, &modes_schema(), &["p"]);
    let e = env
        .write(&table, WriteMode::Append, vec![wide_rows(0, 8192)], &[])
        .unwrap_err();
    assert_ne!(e.error_class(), ErrorClass::InternalError, "{e}");
    assert!(e.to_string().to_lowercase().contains("memory"), "{e}");
    assert_eq!(data_files(&env.dir), 0);
}

/// `(id, k)` partitioned by `k` of `k_type`.
fn keyed_schema(k_type: &ArrowType) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int64, true),
        Field::new("k", k_type.clone(), true),
    ]))
}

fn keyed_rows(k_type: &ArrowType, rows: &[(i64, Option<i32>)]) -> Vec<RecordBatch> {
    let id: ArrayRef = Arc::new(Int64Array::from(
        rows.iter().map(|r| r.0).collect::<Vec<_>>(),
    ));
    let k: ArrayRef = match k_type {
        ArrowType::Int32 => Arc::new(Int32Array::from(
            rows.iter().map(|r| r.1).collect::<Vec<_>>(),
        )),
        _ => Arc::new(Date32Array::from(
            rows.iter().map(|r| r.1).collect::<Vec<_>>(),
        )),
    };
    vec![RecordBatch::try_new(keyed_schema(k_type), vec![id, k]).unwrap()]
}

fn k_literal(k_type: &ArrowType, v: i32) -> Expr {
    Expr::literal(match k_type {
        ArrowType::Int32 => Literal::Integer(v),
        _ => Literal::Date(v),
    })
}

fn k_filter(k_type: &ArrowType, op: &str, v: i32) -> Arc<Expr> {
    Arc::new(Expr::function(
        op,
        vec![Expr::column(["k"]), k_literal(k_type, v)],
    ))
}

/// Overwrites on a table with a NULL partition and an int or a date partition column remove
/// exactly the matching partitions: `replaceWhere k = 1`, a static partition (`k <=> 2`) and a
/// dynamic overwrite of the NULL partition.
#[test]
fn overwrites_remove_only_exact_partition_matches() {
    for k_type in [ArrowType::Int32, ArrowType::Date32] {
        let env = Env::new();
        let table = new_table(&env, &keyed_schema(&k_type), &["k"]);
        env.write(
            &table,
            WriteMode::Append,
            keyed_rows(&k_type, &[(1, Some(1)), (2, Some(2)), (3, None)]),
            &[],
        )
        .unwrap();
        // replaceWhere k = 1: the NULL partition stays (NULL is not TRUE).
        env.write(
            &env.open().unwrap(),
            WriteMode::OverwriteByFilter(k_filter(&k_type, "=", 1)),
            keyed_rows(&k_type, &[(10, Some(1))]),
            &[],
        )
        .unwrap();
        assert_eq!(
            ids(&env.read(&env.open().unwrap())),
            vec![2, 3, 10],
            "{k_type}"
        );
        // A static partition: k <=> 2.
        env.write(
            &env.open().unwrap(),
            WriteMode::OverwriteByFilter(k_filter(&k_type, "<=>", 2)),
            keyed_rows(&k_type, &[(20, Some(2))]),
            &[],
        )
        .unwrap();
        assert_eq!(
            ids(&env.read(&env.open().unwrap())),
            vec![3, 10, 20],
            "{k_type}"
        );
        // A dynamic overwrite of the NULL partition only.
        env.write(
            &env.open().unwrap(),
            WriteMode::OverwriteDynamic,
            keyed_rows(&k_type, &[(30, None)]),
            &[],
        )
        .unwrap();
        assert_eq!(
            ids(&env.read(&env.open().unwrap())),
            vec![10, 20, 30],
            "{k_type}"
        );
        assert_eq!(super::removes(&env.log(env.latest())), 1, "{k_type}");
    }
}

/// When the kernel's scan keeps every file (a skipping predicate it cannot build), the exact
/// check still removes only the matching partition.
#[test]
fn removes_are_exact_when_the_scan_keeps_every_file() {
    let env = Env::new();
    let k_type = ArrowType::Int32;
    let table = new_table(&env, &keyed_schema(&k_type), &["k"]);
    env.write(
        &table,
        WriteMode::Append,
        keyed_rows(&k_type, &[(1, Some(1)), (2, Some(2)), (3, None)]),
        &[],
    )
    .unwrap();
    let t = env.open().unwrap();
    let snapshot = Arc::clone(t.snapshot());
    let exact = Arc::new(Pred::eq(
        KExpr::column(["k"]),
        KExpr::literal(Scalar::Integer(1)),
    ));
    let (removes, files, _) = env
        .rt
        .block_on(env.engine.delta_blocking(move |e| {
            select_removes(e, &snapshot, "t", &["k".to_owned()], None, Some(&exact))
        }))
        .unwrap()
        .unwrap();
    assert_eq!(files, 1);
    let selected: usize = removes
        .iter()
        .map(|r| r.selection.iter().filter(|s| **s).count())
        .sum();
    assert_eq!(selected, 1);
}

/// A create whose version 0 already carries its write id has landed (a retried task, or a
/// lost answer), at planning and at the commit (`Conflicted`); another write's version 0 is
/// `TABLE_OR_VIEW_ALREADY_EXISTS`.
#[test]
fn a_create_finds_its_own_version_zero() {
    let env = Env::new();
    let schema = keyed_schema(&ArrowType::Int32);
    let rows = keyed_rows(&ArrowType::Int32, &[(1, Some(1))]);
    // The commit sees `Conflicted`: version 0 appears between the plan and the commit.
    for (winner, ok) in [("w", true), ("other", false)] {
        let env = Env::new();
        let table = new_table(&env, &schema, &["k"]);
        let new = table.new_table();
        let planned = env
            .rt
            .block_on(env.engine.delta_blocking({
                let table = table.clone();
                move |e| plan_create(e, table.new_table(), "w", FileLimits::default())
            }))
            .unwrap()
            .unwrap();
        let Planned::Ready { create, .. } = planned else {
            panic!("not planned")
        };
        let txn = take_create(&create).unwrap();
        env.rt
            .block_on(new_table(&env, &schema, &["k"]).create_empty(winner))
            .unwrap();
        let name = new_name(new);
        let out = env
            .rt
            .block_on(env.engine.delta_blocking(move |e| {
                commit_create(e, txn, Vec::new(), "w", &name)
                    .map_err(super::super::sink::Failure::into_error)
            }))
            .unwrap();
        match out {
            Ok(()) => assert!(ok, "{winner}"),
            Err(e) => {
                assert!(!ok, "{winner}: {e}");
                assert_eq!(e.error_class(), ErrorClass::TableOrViewAlreadyExists);
            }
        }
        assert_eq!(env.latest(), 0);
    }
    // A retried CTAS task: version 0 is there with this write's id, so planning finds it
    // landed and nothing is written twice.
    let table = new_table(&env, &schema, &["k"]);
    env.write_as(&table, WriteMode::Append, rows.clone(), &[], "w1")
        .unwrap();
    let again = new_table(&env, &schema, &["k"]);
    env.write_as(&again, WriteMode::Append, rows.clone(), &[], "w1")
        .unwrap();
    assert_eq!(env.latest(), 0);
    assert_eq!(ids(&env.read(&env.open().unwrap())), vec![1]);
    assert_eq!(data_files(&env.dir), 1);
    let other = new_table(&env, &schema, &["k"]);
    let e = env
        .write_as(&other, WriteMode::Append, rows, &[], "w2")
        .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::TableOrViewAlreadyExists, "{e}");
}

fn new_name(new: &super::super::sink::NewTable) -> String {
    new.name().to_owned()
}

/// A write refused after some of its files are written (the `replaceWhere` constraint fails on
/// a late batch) leaves no data file and no commit, and its memory is given back.
#[test]
fn a_failed_write_leaves_no_data_files() {
    let env = Env::new();
    let table = new_table(&env, &modes_schema(), &["p"]);
    env.write(&table, WriteMode::Append, vec![wide_rows(0, 10)], &[])
        .unwrap();
    let before = data_files(&env.dir);
    let mut batches: Vec<RecordBatch> = (0..8)
        .map(|b| {
            let mut rows = wide_rows(100 + b * 100, 100);
            // Every row in partition `x`.
            let p: ArrayRef = Arc::new(StringArray::from_iter_values((0..100).map(|_| "x")));
            rows = RecordBatch::try_new(
                modes_schema(),
                vec![Arc::clone(rows.column(0)), Arc::clone(rows.column(1)), p],
            )
            .unwrap();
            rows
        })
        .collect();
    // The last batch holds a row of `y`.
    batches.push(wide_rows(5000, 2));
    let filter = Arc::new(Expr::function(
        "==",
        vec![
            Expr::column(["p"]),
            Expr::literal(Literal::String("x".to_owned())),
        ],
    ));
    let e = env
        .write(
            &env.open().unwrap(),
            WriteMode::OverwriteByFilter(filter),
            batches,
            &[("maxRecordsPerFile", "50")],
        )
        .unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::DeltaReplaceWhereMismatch,
        "{e}"
    );
    assert_eq!(env.latest(), 0);
    // Other drivers' files are deleted in the background.
    for _ in 0..100 {
        if data_files(&env.dir) == before {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    if data_files(&env.dir) != before {
        let out = std::process::Command::new("find")
            .arg(&env.dir)
            .output()
            .unwrap();
        panic!("{}", String::from_utf8_lossy(&out.stdout));
    }
}

/// A commit past `delta.checkpointInterval` is followed by a checkpoint (detached from the
/// statement).
#[test]
fn a_checkpoint_follows_the_interval() {
    let env = Env::new();
    let table = NewDeltaTable::new(
        env.engine.clone(),
        &modes_schema(),
        vec!["p".to_owned()],
        BTreeMap::from([("delta.checkpointInterval".to_owned(), "2".to_owned())]),
        None,
    )
    .unwrap();
    env.write(&table, WriteMode::Append, vec![wide_rows(0, 4)], &[])
        .unwrap();
    for i in 1..=3 {
        env.write(
            &env.open().unwrap(),
            WriteMode::Append,
            vec![wide_rows(i * 10, 4)],
            &[],
        )
        .unwrap();
    }
    let last = env.dir.join("_delta_log/_last_checkpoint");
    for _ in 0..200 {
        if last.exists() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(last.exists(), "no checkpoint");
    assert_eq!(ids(&env.read(&env.open().unwrap())).len(), 16);
}

/// No write into an oracle table, of any mode, raises `INTERNAL_ERROR` (design section 3.7:
/// every user-reachable refusal is named).
#[test]
fn delta_no_internal_error_on_oracle_tables() {
    let features = super::expected("write-features/FEATURES.json");
    let mut tried = 0;
    for name in features["tables"].as_object().unwrap().keys() {
        for mode in [
            WriteMode::Append,
            WriteMode::Truncate,
            WriteMode::OverwriteDynamic,
        ] {
            let env = Env::fixture(&format!("write-features/{name}"));
            let table = env.open().unwrap();
            let row = super::one_row(&table);
            if let Err(e) = env.write(&table, mode.clone(), vec![row], &[]) {
                assert_ne!(
                    e.error_class(),
                    ErrorClass::InternalError,
                    "{name} {mode:?}: {e}"
                );
            }
            tried += 1;
        }
    }
    assert!(tried >= 60, "{tried}");
}
