// SPDX-License-Identifier: Apache-2.0
//! Tests of the merge-on-read sink (`mor_sink.rs`, PR-36c part 2): the rows a statement's SQL
//! selects, ending with the row id, through the row-level table's rewriter.

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use krishiv_common::error::ErrorClass;
use krishiv_exec::planner::RowLevelOperation;
use krishiv_exec::{ROW_FILE_COLUMN, ROW_POSITION_COLUMN};
use krishiv_memory::MemoryConfig;

use super::cow_tests::{Dml, append, create, get, live, summary};
use super::provider_tests::{Env, env};
use super::sink_tests::{
    Catalog, PerDriver, expected, parquet_files, range, read_back, run, run_in, settled,
};

const MOR: &[(&str, &str)] = &[
    ("write.delete.mode", "merge-on-read"),
    ("write.update.mode", "merge-on-read"),
];

#[test]
fn a_merge_on_read_delete_writes_a_delete_file_and_no_data() {
    let env = env();
    let t = create(&env, 2, false, MOR);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    let before = live(&env, &catalog.metadata()).0;
    Dml::open(&env, &catalog, RowLevelOperation::Delete)
        .mor_delete("id < 3")
        .unwrap();
    let m = catalog.metadata();
    assert_eq!(read_back(&env, &catalog), expected(3..10));
    let (data, deletes) = live(&env, &m);
    assert_eq!(data, before, "no data file is rewritten");
    assert_eq!(deletes.len(), 1);
    let (_, s) = summary(&m);
    assert_eq!(get(&s, "added-position-delete-files"), Some("1"));
}

#[test]
fn a_merge_on_read_update_deletes_the_old_row_and_adds_the_new() {
    let env = env();
    let t = create(&env, 2, false, MOR);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 9));
    Dml::open(&env, &catalog, RowLevelOperation::Update)
        .mor_update("id = 4", "'u'")
        .unwrap();
    let mut want = expected(0..9);
    want.remove(4);
    want.push("4|1|u".to_owned());
    want.sort();
    let mut got = read_back(&env, &catalog);
    got.sort();
    assert_eq!(got, want);
}

#[test]
fn a_merge_on_read_delete_that_hits_nothing_commits_an_overwrite() {
    let env = env();
    let t = create(&env, 2, false, MOR);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 5));
    let before = catalog.metadata().current_snapshot_id();
    Dml::open(&env, &catalog, RowLevelOperation::Delete)
        .mor_delete("id > 100")
        .unwrap();
    let m = catalog.metadata();
    assert_ne!(m.current_snapshot_id(), before);
    let (op, _) = summary(&m);
    assert_eq!(format!("{op:?}").to_lowercase(), "overwrite");
}

#[test]
fn a_v3_merge_on_read_delete_writes_a_dv() {
    let env = env();
    let t = create(&env, 3, false, MOR);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    Dml::open(&env, &catalog, RowLevelOperation::Delete)
        .mor_delete("id % 2 = 0")
        .unwrap();
    let m = catalog.metadata();
    assert_eq!(read_back(&env, &catalog).len(), 5);
    let (_, s) = summary(&m);
    assert_eq!(get(&s, "added-dvs"), Some("1"));
}

#[test]
fn an_unknown_row_level_mode_fails_the_statement() {
    for (op, key) in [
        (RowLevelOperation::Delete, "write.delete.mode"),
        (RowLevelOperation::Update, "write.update.mode"),
    ] {
        let env = env();
        let t = create(&env, 2, false, &[(key, "mor")]);
        let catalog = Catalog::new(&t);
        append(&env, &catalog, range(0, 3));
        assert!(Dml::open(&env, &catalog, op).scan_is_refused(), "{key}");
    }
}

/// Every data, delete and puffin file under the table directory.
fn table_files(env: &Env) -> Vec<String> {
    fn walk(dir: &std::path::Path, out: &mut Vec<String>) {
        for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p
                .extension()
                .is_some_and(|x| x == "parquet" || x == "puffin")
            {
                out.push(p.display().to_string());
            }
        }
    }
    let mut out = Vec::new();
    walk(env.dir.path(), &mut out);
    out.sort();
    out
}

/// The same rows as `b`, with the position column replaced by `positions`.
fn with_positions(b: &RecordBatch, positions: Vec<i64>) -> RecordBatch {
    let mut cols = b.columns().to_vec();
    *cols.last_mut().unwrap() = Arc::new(Int64Array::from(positions));
    RecordBatch::try_new(b.schema(), cols).unwrap()
}

fn input(
    schema: arrow_schema::SchemaRef,
    batches: Vec<RecordBatch>,
    stall: bool,
) -> Arc<PerDriver> {
    Arc::new(PerDriver {
        schema,
        batches,
        stall,
    })
}

/// R36c1 m6: the positions and the delete file built in memory are charged to the query's
/// pool: a limit too small for them fails the statement with the memory error class, and
/// nothing is committed or left behind.
#[test]
fn the_positions_of_a_delete_are_charged_to_the_pool() {
    for version in [2, 3] {
        let env = env();
        let t = create(&env, version, false, MOR);
        let catalog = Catalog::new(&t);
        append(&env, &catalog, range(0, 200));
        let before = table_files(&env);
        let snapshot = catalog.metadata().current_snapshot_id();
        let dml = Dml::open(&env, &catalog, RowLevelOperation::Delete);
        let select = format!("SELECT {ROW_FILE_COLUMN}, {ROW_POSITION_COLUMN} FROM t");
        let (sink, schema, batches) = dml.mor_parts(&select);
        let config = MemoryConfig {
            limit: Some(64),
            quantum: 8,
            spill_scratch_reserve: 0,
            ..MemoryConfig::default()
        };
        let (out, pool, _runner) = run_in(
            &env,
            sink.unwrap(),
            input(schema, batches, false),
            None,
            &config,
        );
        let e = out.unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::UnableToAcquireMemory,
            "v{version}: {e}"
        );
        assert_eq!(settled(&pool), 0);
        assert_eq!(catalog.metadata().current_snapshot_id(), snapshot);
        assert_eq!(table_files(&env), before);
    }
}

/// One driver deletes rows while another fails: no commit, and nothing the first wrote is left.
#[test]
fn a_failing_driver_leaves_no_commit_and_no_file() {
    for version in [2, 3] {
        let env = env();
        let t = create(&env, version, false, MOR);
        let catalog = Catalog::new(&t);
        append(&env, &catalog, range(0, 10));
        let before = table_files(&env);
        let snapshot = catalog.metadata().current_snapshot_id();
        let dml = Dml::open(&env, &catalog, RowLevelOperation::Delete);
        let select = format!("SELECT {ROW_FILE_COLUMN}, {ROW_POSITION_COLUMN} FROM t WHERE id < 4");
        let (sink, schema, batches) = dml.mor_parts(&select);
        let good = batches.first().unwrap().clone();
        let bad = with_positions(&good, vec![-1; good.num_rows()]);
        let (out, pool, _runner) = run(
            &env,
            sink.unwrap(),
            input(schema, vec![good, bad], false),
            None,
        );
        assert!(out.is_err(), "v{version}");
        assert_eq!(settled(&pool), 0);
        assert_eq!(catalog.metadata().current_snapshot_id(), snapshot);
        // Background deletes of the finished task's file get a moment.
        for _ in 0..500 {
            if table_files(&env) == before {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(table_files(&env), before, "v{version}");
    }
}

/// A statement cancelled before its finalize: no commit, nothing left.
#[test]
fn a_cancelled_merge_on_read_statement_leaves_nothing() {
    let env = env();
    let t = create(&env, 2, false, MOR);
    let catalog = Catalog::new(&t);
    append(&env, &catalog, range(0, 10));
    let before = table_files(&env);
    let snapshot = catalog.metadata().current_snapshot_id();
    let dml = Dml::open(&env, &catalog, RowLevelOperation::Delete);
    let select = format!("SELECT {ROW_FILE_COLUMN}, {ROW_POSITION_COLUMN} FROM t WHERE id < 4");
    let (sink, schema, batches) = dml.mor_parts(&select);
    let (out, pool, _runner) = run(
        &env,
        sink.unwrap(),
        input(schema, batches, true),
        Some(Box::new(|| true)),
    );
    assert!(out.is_err());
    assert_eq!(settled(&pool), 0);
    assert_eq!(catalog.metadata().current_snapshot_id(), snapshot);
    for _ in 0..500 {
        if table_files(&env) == before {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(table_files(&env), before);
}

/// Two drivers deleting rows of one data file: on v2 each task writes a position delete file,
/// on v3 the two merge into one deletion vector; either way the rows are gone.
#[test]
fn two_drivers_deleting_from_one_file_are_merged() {
    for (version, key, want) in [
        (2, "added-position-delete-files", "2"),
        (3, "added-dvs", "1"),
    ] {
        let env = env();
        let t = create(&env, version, false, MOR);
        let catalog = Catalog::new(&t);
        append(&env, &catalog, range(0, 10));
        let dml = Dml::open(&env, &catalog, RowLevelOperation::Delete);
        let select = format!("SELECT {ROW_FILE_COLUMN}, {ROW_POSITION_COLUMN} FROM t WHERE id < 4");
        let (sink, schema, batches) = dml.mor_parts(&select);
        let all = batches.first().unwrap();
        let split = vec![all.slice(0, 2), all.slice(2, all.num_rows() - 2)];
        let (out, pool, _runner) = run(&env, sink.unwrap(), input(schema, split, false), None);
        out.unwrap();
        assert_eq!(settled(&pool), 0);
        let m = catalog.metadata();
        let (_, s) = summary(&m);
        assert_eq!(get(&s, key), Some(want), "v{version}: {s:?}");
        assert_eq!(read_back(&env, &catalog), expected(4..10), "v{version}");
    }
}

/// A v3 merge-on-read `UPDATE` is refused before it runs, whether or not it hits a row.
#[test]
fn a_v3_merge_on_read_update_is_refused_up_front() {
    for cond in ["id = 4", "false"] {
        let env = env();
        let t = create(&env, 3, false, MOR);
        let catalog = Catalog::new(&t);
        append(&env, &catalog, range(0, 9));
        let snapshot = catalog.metadata().current_snapshot_id();
        let e = Dml::open(&env, &catalog, RowLevelOperation::Update)
            .mor_update(cond, "'u'")
            .unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{cond}: {e}"
        );
        assert_eq!(catalog.metadata().current_snapshot_id(), snapshot, "{cond}");
        let _ = parquet_files(&env, false);
    }
}
