// SPDX-License-Identifier: Apache-2.0
//! Copy-on-write `DELETE` and `UPDATE` through [`DeltaRowLevel`] (PR-37b part 2), driven as the
//! session drives it (the find scan with row ids, then the rewrite's rows into the rewriter):
//! the row id over deletion vectors, the commit's removes, adds, parameters and metrics, a
//! whole-file delete, the conflict checker against a concurrent remove and append, a cancelled
//! rewrite, and the refusals (append-only, change data feed). The comparison with delta-spark
//! is the session's `delta_dml` test (`tests/fixtures/delta-4.4.1/dml/DML.json`).

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_schema::Schema;
use krishiv_common::error::KrishivError;
use krishiv_exec::planner::{
    RowLevelCounts, RowLevelOperation, RowLevelTable, RowLevelWrite, TableProvider, WriteMode,
};
use krishiv_exec::{ROW_FILE_COLUMN, ROW_POSITION_COLUMN, ScanPushdowns};
use krishiv_io::ColumnPath;
use krishiv_plan::{Expr, Literal};
use serde_json::Value;

use super::super::dml::{
    ROW_LEVEL_AFFECTED_OPTION, ROW_LEVEL_CONDITION_OPTION, ROW_LEVEL_SCAN_MS_OPTION,
};
use super::super::test_hooks::{self, Point};
use super::streaming::{data_files, new_table};
use super::{Env, Leaf, Writing, commit_info, modes_schema, rows, typed_rows};
use crate::table::DeltaTable;

/// `(id, file)` of every live row, by the row-level scan.
fn row_files(env: &Env, table: &DeltaTable, op: RowLevelOperation) -> Vec<(i64, Vec<u8>, i64)> {
    let rl = table.row_level(op).unwrap();
    let source = rl
        .scan()
        .unwrap()
        .build(ScanPushdowns {
            projection: Some(vec![
                ColumnPath::new("id"),
                ColumnPath::new(ROW_FILE_COLUMN),
                ColumnPath::new(ROW_POSITION_COLUMN),
            ]),
            ..ScanPushdowns::default()
        })
        .unwrap();
    let mut out = Vec::new();
    for b in env.run(Arc::new(Leaf(source))).unwrap() {
        let id = arrow_cast::cast(b.column(0), &arrow_schema::DataType::Int64).unwrap();
        let id = id.as_primitive::<Int64Type>();
        let file = b.column(1).as_binary_view();
        let pos = b.column(2).as_primitive::<Int64Type>();
        for i in 0..b.num_rows() {
            out.push((id.value(i), file.value(i).to_vec(), pos.value(i)));
        }
    }
    out.sort_unstable();
    out
}

/// A table of `rows` written in two appends (two files per partition written).
fn two_file_table(env: &Env, partitions: &[&str]) {
    let table = new_table(env, &modes_schema(), partitions);
    env.write(
        &table,
        WriteMode::Append,
        rows(&[
            (1, "a", Some("x")),
            (2, "b", Some("x")),
            (3, "c", Some("y")),
        ]),
        &[],
    )
    .unwrap();
    env.write(
        &env.open().unwrap(),
        WriteMode::Append,
        rows(&[(4, "d", Some("y")), (5, "e", None), (6, "f", Some("z"))]),
        &[],
    )
    .unwrap();
}

/// The rewrite of `touched` with `batches` through `rl`'s rewriter.
#[expect(
    clippy::too_many_arguments,
    reason = "test helper: the request's parts"
)]
#[expect(clippy::needless_pass_by_value, reason = "test helper")]
fn rewrite(
    env: &Env,
    rl: &Arc<dyn RowLevelTable>,
    op: RowLevelOperation,
    touched: Vec<Vec<u8>>,
    batches: Vec<RecordBatch>,
    condition: Option<(&str, Expr)>,
    affected: u64,
    write_id: &str,
) -> Result<(), KrishivError> {
    let mut request = RowLevelWrite::new(
        op,
        "`t`",
        modes_schema(),
        touched,
        condition.as_ref().map(|(_, e)| Arc::new(e.clone())),
    );
    request.write_id = Some(write_id.to_owned());
    request
        .options
        .insert(ROW_LEVEL_AFFECTED_OPTION.to_owned(), affected.to_string());
    request
        .options
        .insert(ROW_LEVEL_SCAN_MS_OPTION.to_owned(), "3".to_owned());
    if let Some((text, _)) = condition {
        request
            .options
            .insert(ROW_LEVEL_CONDITION_OPTION.to_owned(), text.to_owned());
    }
    let sink = rl.rewriter(&request)?;
    let input = krishiv_exec::planner::MemoryTable::new(
        modes_schema(),
        batches
            .into_iter()
            .map(krishiv_types::batch::Batch::new)
            .collect(),
    )
    .scan()?
    .build(ScanPushdowns::default())?;
    let out = env.run(Arc::new(Writing {
        input: Arc::new(Leaf(input)),
        sink,
    }));
    assert_eq!(
        env.query_pool.reserved(),
        0,
        "the rewrite gave its memory back"
    );
    out.map(drop)
}

fn eq(column: &str, value: Literal) -> Expr {
    Expr::function("=", vec![Expr::column([column]), Expr::literal(value)])
}

fn metrics(info: &Value) -> serde_json::Map<String, Value> {
    let mut m = info["operationMetrics"].as_object().unwrap().clone();
    for k in [
        "executionTimeMs",
        "scanTimeMs",
        "rewriteTimeMs",
        "numAddedBytes",
    ] {
        m.remove(k);
    }
    m
}

fn the_file_of(files: &[(i64, Vec<u8>, i64)], id: i64) -> Vec<u8> {
    files.iter().find(|f| f.0 == id).unwrap().1.clone()
}

#[test]
fn the_row_id_is_the_logged_path_and_the_position_in_the_file_past_deletion_vectors() {
    let env = Env::fixture("deletion-vectors");
    let table = env.open().unwrap();
    let paths: Vec<String> = table.units().iter().map(|u| u.path.clone()).collect();
    let rows = row_files(&env, &table, RowLevelOperation::Delete);
    // One file of `range(40000)` written in order: a row's position is its id; the deleted
    // rows' positions are skipped, never renumbered.
    assert!(!rows.is_empty());
    assert!(rows.len() < 40_000);
    for (id, file, pos) in &rows {
        assert_eq!(*pos, *id, "row {id}");
        assert!(paths.iter().any(|p| p.as_bytes() == file.as_slice()));
    }
    assert_eq!(env.query_pool.reserved(), 0);
}

#[test]
fn a_row_level_delete_removes_the_touched_file_and_adds_its_other_rows() {
    let env = Env::new();
    two_file_table(&env, &[]);
    let table = env.open().unwrap();
    let before = table.version();
    let files = row_files(&env, &table, RowLevelOperation::Delete);
    let touched = the_file_of(&files, 2);
    let rl = table.row_level(RowLevelOperation::Delete).unwrap();
    rewrite(
        &env,
        &rl,
        RowLevelOperation::Delete,
        vec![touched.clone()],
        rows(&[(1, "a", Some("x")), (3, "c", Some("y"))]),
        Some(("id = 2", eq("id", Literal::Long(2)))),
        1,
        "dml-delete-rows",
    )
    .unwrap();
    assert_eq!(env.latest(), before + 1);
    let actions = env.log(env.latest());
    let info = commit_info(&actions);
    assert_eq!(info["operation"], "DELETE");
    assert_eq!(info["operationParameters"]["predicate"], r#"["(id = 2)"]"#);
    let removed: Vec<&str> = actions
        .iter()
        .filter_map(|a| a.get("remove")?.get("path")?.as_str())
        .collect();
    assert_eq!(removed, [std::str::from_utf8(&touched).unwrap()]);
    let want: serde_json::Map<String, Value> = serde_json::from_value(serde_json::json!({
        "numRemovedFiles": "1", "numRemovedBytes": info["operationMetrics"]["numRemovedBytes"],
        "numAddedFiles": "1", "numAddedChangeFiles": "0", "numDeletedRows": "1",
        "numCopiedRows": "2", "numDeletionVectorsAdded": "0", "numDeletionVectorsRemoved": "0",
        "numDeletionVectorsUpdated": "0",
    }))
    .unwrap();
    assert_eq!(metrics(&info), want);
    assert_ne!(info["operationMetrics"]["numRemovedBytes"], "0");
    let left = typed_rows(&env.read(&env.open().unwrap()));
    let ids: Vec<i64> = left.iter().map(|r| r[0].as_i64().unwrap()).collect();
    assert_eq!(ids, [1, 3, 4, 5, 6]);
    let out = rl.output(&RowLevelCounts::affected(1)).unwrap();
    assert_eq!(out.schema().field(0).name(), "num_affected_rows");
    assert_eq!(out.column(0).as_primitive::<Int64Type>().value(0), 1);
}

#[test]
fn an_update_rewrites_every_row_of_the_touched_files() {
    let env = Env::new();
    two_file_table(&env, &["p"]);
    let table = env.open().unwrap();
    let files = row_files(&env, &table, RowLevelOperation::Update);
    let touched = the_file_of(&files, 2);
    let rl = table.row_level(RowLevelOperation::Update).unwrap();
    rewrite(
        &env,
        &rl,
        RowLevelOperation::Update,
        vec![touched],
        rows(&[(1, "a", Some("x")), (2, "u", Some("x"))]),
        Some(("v = 'b'", eq("v", Literal::String("b".to_owned())))),
        1,
        "dml-update-rows",
    )
    .unwrap();
    let info = commit_info(&env.log(env.latest()));
    assert_eq!(info["operation"], "UPDATE");
    let m = metrics(&info);
    assert_eq!(m["numUpdatedRows"], "1");
    assert_eq!(m["numCopiedRows"], "1");
    assert_eq!(m["numRemovedFiles"], "1");
    assert_eq!(m["numAddedFiles"], "1");
    assert!(m.get("numDeletedRows").is_none());
    let left = typed_rows(&env.read(&env.open().unwrap()));
    assert_eq!(left[1], serde_json::json!([2, "u", "x"]));
    assert_eq!(left.len(), 6);
}

#[test]
fn a_partition_delete_removes_whole_files_and_counts_rows_from_the_statistics() {
    let env = Env::new();
    two_file_table(&env, &["p"]);
    let table = env.open().unwrap();
    let files = row_files(&env, &table, RowLevelOperation::Delete);
    let touched = vec![the_file_of(&files, 1)];
    let rl = table.row_level(RowLevelOperation::Delete).unwrap();
    rewrite(
        &env,
        &rl,
        RowLevelOperation::Delete,
        touched,
        Vec::new(),
        Some(("p = 'x'", eq("p", Literal::String("x".to_owned())))),
        2,
        "dml-delete-partition",
    )
    .unwrap();
    let info = commit_info(&env.log(env.latest()));
    let m = metrics(&info);
    assert_eq!(m["numDeletedRows"], "2");
    assert_eq!(m["numCopiedRows"], "0");
    assert_eq!(m["numAddedFiles"], "0");
    assert_eq!(info["operationMetrics"]["rewriteTimeMs"], "0");
    let left = typed_rows(&env.read(&env.open().unwrap()));
    assert_eq!(left.len(), 4);
    let out = rl.output(&RowLevelCounts::affected(2)).unwrap();
    assert_eq!(out.column(0).as_primitive::<Int64Type>().value(0), 2);
}

#[test]
fn a_delete_that_matches_nothing_commits_nothing() {
    let env = Env::new();
    two_file_table(&env, &[]);
    let table = env.open().unwrap();
    let before = env.latest();
    let rl = table.row_level(RowLevelOperation::Delete).unwrap();
    rewrite(
        &env,
        &rl,
        RowLevelOperation::Delete,
        Vec::new(),
        Vec::new(),
        Some(("id = 99", eq("id", Literal::Long(99)))),
        0,
        "dml-delete-nothing",
    )
    .unwrap();
    assert_eq!(env.latest(), before);
    let out = rl.output(&RowLevelCounts::affected(0)).unwrap();
    assert_eq!(out.column(0).as_primitive::<Int64Type>().value(0), 0);
}

/// The conflict checker sees the statement as a read of the whole table: a concurrent remove
/// of a read file is a delete-read conflict, a concurrent append an append conflict.
#[test]
fn a_concurrent_change_conflicts_with_a_row_level_change() {
    for (winner, class) in [
        ("truncate", "DELTA_CONCURRENT_DELETE_READ.WITHOUT_HINT"),
        ("append", "DELTA_CONCURRENT_APPEND.WITHOUT_HINT"),
    ] {
        let env = Env::new();
        two_file_table(&env, &[]);
        let table = env.open().unwrap();
        let files = row_files(&env, &table, RowLevelOperation::Delete);
        let touched = the_file_of(&files, 2);
        // Another writer commits after the statement read its snapshot: a `DELETE FROM t`
        // (removes only) or an append.
        if winner == "truncate" {
            let other = env.open().unwrap();
            let all: Vec<Vec<u8>> = other
                .units()
                .iter()
                .map(|u| u.path.as_bytes().to_vec())
                .collect();
            let rl = other.row_level(RowLevelOperation::Delete).unwrap();
            rewrite(
                &env,
                &rl,
                RowLevelOperation::Delete,
                all,
                Vec::new(),
                None,
                6,
                "dml-conflict-winner",
            )
            .unwrap();
        } else {
            env.write(
                &env.open().unwrap(),
                WriteMode::Append,
                rows(&[(9, "w", None)]),
                &[],
            )
            .unwrap();
        }
        let after = env.latest();
        let rl = table.row_level(RowLevelOperation::Delete).unwrap();
        let err = rewrite(
            &env,
            &rl,
            RowLevelOperation::Delete,
            vec![touched],
            rows(&[(1, "a", Some("x")), (3, "c", Some("y"))]),
            Some(("id = 2", eq("id", Literal::Long(2)))),
            1,
            &format!("dml-conflict-{winner}"),
        )
        .unwrap_err();
        assert_eq!(err.error_class().name(), class, "{winner}: {err}");
        assert_eq!(env.latest(), after, "{winner}");
    }
}

#[test]
fn a_rewrite_cancelled_during_its_batches_leaves_the_table_unchanged() {
    let env = Env::new();
    two_file_table(&env, &[]);
    let table = env.open().unwrap();
    let files = row_files(&env, &table, RowLevelOperation::Update);
    let touched = the_file_of(&files, 2);
    let (version, before) = (env.latest(), data_files(&env.dir));
    let id = "dml-cancel-during-sink";
    test_hooks::set(
        id,
        Arc::new(|at, query| {
            if at == Point::Sunk {
                query.cancel();
                true
            } else {
                false
            }
        }),
    );
    let rl = table.row_level(RowLevelOperation::Update).unwrap();
    let out = rewrite(
        &env,
        &rl,
        RowLevelOperation::Update,
        vec![touched],
        rows(&[
            (1, "a", Some("x")),
            (2, "u", Some("x")),
            (3, "c", Some("y")),
        ]),
        Some(("id = 2", eq("id", Literal::Long(2)))),
        1,
        id,
    );
    assert!(out.is_err());
    assert_eq!(env.latest(), version);
    let mut gone = false;
    for _ in 0..100 {
        if data_files(&env.dir) == before {
            gone = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(gone, "the cancelled rewrite left data files");
    assert_eq!(env.query_pool.reserved(), 0);
}

#[test]
fn an_append_only_table_refuses_row_level_changes_before_scanning() {
    let env = Env::fixture("write-features/append-only");
    let table = env.open().unwrap();
    for op in [RowLevelOperation::Delete, RowLevelOperation::Update] {
        let Err(err) = table.row_level(op).unwrap().scan() else {
            panic!("{op:?} scanned an append-only table");
        };
        assert_eq!(
            err.error_class().name(),
            "DELTA_CANNOT_MODIFY_APPEND_ONLY",
            "{op:?}"
        );
        assert_eq!(err.sql_state(), Some("42809"));
    }
}

#[test]
fn a_change_data_feed_table_refuses_rewrites_but_commits_whole_file_deletes() {
    let env = Env::fixture("write-features/change-data-feed");
    let before = env.latest();
    let table = env.open().unwrap();
    let request = |op, filter: Option<Expr>| {
        let mut r = RowLevelWrite::new(
            op,
            "`t`",
            Arc::new(Schema::empty()),
            Vec::new(),
            filter.map(Arc::new),
        );
        r.write_id = Some("dml-cdf".to_owned());
        r
    };
    let update = table.row_level(RowLevelOperation::Update).unwrap();
    let Err(err) = update.rewriter(&request(
        RowLevelOperation::Update,
        Some(eq("id", Literal::Long(1))),
    )) else {
        panic!("an UPDATE of a change-data-feed table was accepted");
    };
    assert_eq!(
        err.error_class().name(),
        "UNSUPPORTED_FEATURE.TABLE_OPERATION"
    );
    assert!(err.to_string().contains("change data feed"), "{err}");
    let delete = table.row_level(RowLevelOperation::Delete).unwrap();
    let Err(err) = delete.rewriter(&request(
        RowLevelOperation::Delete,
        Some(eq("id", Literal::Long(1))),
    )) else {
        panic!("a row DELETE of a change-data-feed table was accepted");
    };
    assert_eq!(
        err.error_class().name(),
        "UNSUPPORTED_FEATURE.TABLE_OPERATION"
    );
    // `DELETE FROM t` removes every file: no rows written, so the kernel takes the commit.
    let files: Vec<Vec<u8>> = table
        .units()
        .iter()
        .map(|u| u.path.as_bytes().to_vec())
        .collect();
    let mut r = request(RowLevelOperation::Delete, None);
    r.touched = files;
    r.input_schema = table.schema();
    let sink = delete.rewriter(&r).unwrap();
    let input = krishiv_exec::planner::MemoryTable::new(table.schema(), Vec::new())
        .scan()
        .unwrap()
        .build(ScanPushdowns::default())
        .unwrap();
    env.run(Arc::new(Writing {
        input: Arc::new(Leaf(input)),
        sink,
    }))
    .unwrap();
    assert_eq!(env.latest(), before + 1);
    let info = commit_info(&env.log(env.latest()));
    assert_eq!(info["operation"], "DELETE");
    assert_eq!(info["operationParameters"]["predicate"], r#"["true"]"#);
    assert!(
        env.read(&env.open().unwrap())
            .iter()
            .all(|b| b.num_rows() == 0)
    );
    assert_eq!(env.query_pool.reserved(), 0);
}
