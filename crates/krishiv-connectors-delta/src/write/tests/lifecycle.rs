// SPDX-License-Identifier: Apache-2.0
//! A write's lifecycle (review 2): cancellation during the batches and during `finalize`
//! leaves no data files, no commit and an empty pool; the global state is dropped after a
//! write; at most one post-commit checkpoint runs per table root; a scan row without partition
//! values refuses the overwrite.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field, Schema};
use delta_kernel::expressions::{Expression as KExpr, Predicate as Pred, Scalar};
use krishiv_common::error::ErrorClass;
use krishiv_exec::planner::WriteMode;

use super::super::sink::{CheckpointSlot, ExactCheck, checkpoints_skipped};
use super::super::test_hooks::{self, Act, Point};
use super::streaming::{data_files, new_table, wide_rows};
use super::{Env, modes_schema};

/// Waits up to 5 s for `done`.
fn eventually(mut done: impl FnMut() -> bool) -> bool {
    for _ in 0..100 {
        if done() {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    done()
}

/// A write into `env`'s table (version 0, `before` data files) cancelled at `point`: an
/// error, the version unchanged, no new data files, the pool empty and the global state gone.
fn cancelled_at(point: Point, write_id: &str) {
    let env = Env::new();
    let table = new_table(&env, &modes_schema(), &["p"]);
    env.write(&table, WriteMode::Append, vec![wide_rows(0, 10)], &[])
        .unwrap();
    let before = data_files(&env.dir);
    test_hooks::set(
        write_id,
        Arc::new(move |site, query| {
            if site.point != point {
                return Act::NONE;
            }
            query.cancel();
            Act::stall(std::time::Duration::from_secs(30))
        }),
    );
    let batches: Vec<RecordBatch> = (0..6).map(|b| wide_rows(100 + b * 100, 100)).collect();
    let out = env.write_as(
        &env.open().unwrap(),
        WriteMode::Append,
        batches,
        &[("maxRecordsPerFile", "20")],
        write_id,
    );
    assert!(out.is_err(), "a cancelled write succeeded at {point:?}");
    assert_eq!(env.latest(), 0, "a cancelled write committed at {point:?}");
    assert!(
        eventually(|| data_files(&env.dir) == before),
        "a cancelled write at {point:?} left {} data files, {before} before",
        data_files(&env.dir)
    );
    assert_eq!(env.query_pool.reserved(), 0);
    assert!(
        eventually(|| test_hooks::global_alive(write_id) == Some(false)),
        "the global state outlived the cancelled write at {point:?}"
    );
}

#[test]
fn a_write_cancelled_during_its_batches_leaves_nothing() {
    cancelled_at(Point::Sunk, "cancel-during-sink");
}

#[test]
fn a_write_cancelled_during_finalize_leaves_nothing() {
    cancelled_at(Point::Finished, "cancel-during-finalize");
}

/// No reference cycle keeps a write's global state alive after the write.
#[test]
fn the_global_state_is_dropped_after_a_write() {
    let env = Env::new();
    let table = new_table(&env, &modes_schema(), &["p"]);
    env.write_as(
        &table,
        WriteMode::Append,
        vec![wide_rows(0, 10)],
        &[],
        "dropped-create",
    )
    .unwrap();
    env.write_as(
        &env.open().unwrap(),
        WriteMode::Append,
        vec![wide_rows(10, 10)],
        &[],
        "dropped-append",
    )
    .unwrap();
    for id in ["dropped-create", "dropped-append"] {
        assert_eq!(test_hooks::global_alive(id), Some(false), "{id}");
    }
}

/// At most one post-commit checkpoint runs per table root: while one holds the root's slot,
/// commits past the interval skip theirs.
#[test]
fn at_most_one_checkpoint_runs_per_table() {
    let a = CheckpointSlot::begin("file:///a/").unwrap();
    assert!(CheckpointSlot::begin("file:///a/").is_none());
    let b = CheckpointSlot::begin("file:///b/").unwrap();
    drop(a);
    let again = CheckpointSlot::begin("file:///a/").unwrap();
    drop((again, b));

    let env = Env::new();
    let table = super::super::NewDeltaTable::new(
        env.engine.clone(),
        &modes_schema(),
        vec!["p".to_owned()],
        std::collections::BTreeMap::from([("delta.checkpointInterval".to_owned(), "1".to_owned())]),
        None,
    )
    .unwrap();
    env.write(&table, WriteMode::Append, vec![wide_rows(0, 4)], &[])
        .unwrap();
    let held = CheckpointSlot::begin(env.engine.table_root().as_str()).unwrap();
    let skipped = checkpoints_skipped();
    for i in 1..=2 {
        env.write(
            &env.open().unwrap(),
            WriteMode::Append,
            vec![wide_rows(i * 10, 4)],
            &[],
        )
        .unwrap();
    }
    assert!(checkpoints_skipped() >= skipped + 2);
    assert!(!env.dir.join("_delta_log/_last_checkpoint").exists());
    drop(held);
    env.write(
        &env.open().unwrap(),
        WriteMode::Append,
        vec![wide_rows(100, 4)],
        &[],
    )
    .unwrap();
    assert!(
        eventually(|| env.dir.join("_delta_log/_last_checkpoint").exists()),
        "no checkpoint once the slot was free"
    );
}

/// A partitioned table's scan row without partition values refuses the overwrite rather than
/// reading every partition column as NULL.
#[test]
fn a_scan_row_without_partition_values_is_refused() {
    let env = Env::new();
    let table = new_table(&env, &modes_schema(), &["p"]);
    env.write(&table, WriteMode::Append, vec![wide_rows(0, 4)], &[])
        .unwrap();
    let opened = env.open().unwrap();
    let pred = Arc::new(Pred::not(Pred::distinct(
        KExpr::column(["p"]),
        KExpr::literal(Scalar::Null(delta_kernel::schema::DataType::STRING)),
    )));
    let check = ExactCheck::new(
        &env.engine,
        opened.snapshot(),
        &["p".to_owned()],
        &pred,
        "`t`",
    )
    .unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "size",
            ArrowType::Int64,
            true,
        )])),
        vec![Arc::new(Int64Array::from(vec![1_i64])) as ArrayRef],
    )
    .unwrap();
    let mut selection = vec![true];
    let e = check.narrow(&batch, &mut selection).unwrap_err();
    assert_eq!(
        e.error_class(),
        ErrorClass::UnsupportedFeatureTableOperation,
        "{e}"
    );
}
