// SPDX-License-Identifier: Apache-2.0
//! Merge-on-read `DELETE` through [`DeltaRowLevel`] (PR-37d), driven as the session drives it
//! (the row ids of the rows the condition holds for, into the rewriter with
//! `MERGE_ON_READ_OPTION`), on copies of delta-spark's deletion-vector tables
//! (`tests/fixtures/delta-4.4.1/dvs/`): when merge-on-read applies, the conflict checker against
//! concurrent vector updates and whole-file removes (the hooked race of a winner committed
//! after the statement read its snapshot), a failure before the commit deleting the vector
//! file, and an unsupported table refusing the option. The comparison with delta-spark is the
//! session's `delta_dvs` test.

use std::sync::Arc;

use arrow_array::{
    BinaryViewArray, BooleanArray, Int32Array, Int64Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType as ArrowType, Field, Schema, SchemaRef};
use krishiv_common::error::KrishivError;
use krishiv_exec::planner::{
    MERGE_ON_READ_OPTION, ROW_WRITE_COLUMN, RowLevelOperation, RowLevelTable, RowLevelWrite,
    TableProvider,
};
use krishiv_exec::{ROW_FILE_COLUMN, ROW_POSITION_COLUMN, ScanPushdowns};
use krishiv_plan::{Expr, Literal};

use super::super::dml::{
    ROW_LEVEL_AFFECTED_OPTION, ROW_LEVEL_ALIAS_OPTION, ROW_LEVEL_CONDITION_OPTION,
};
use super::super::test_hooks;
use super::dml::row_files;
use super::{Env, Leaf, Writing};

/// The merge-on-read rows of a `DELETE`: the table's columns (NULL) and the row ids.
fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", ArrowType::Int64, true),
        Field::new("v", ArrowType::Utf8, true),
        Field::new("n", ArrowType::Int32, true),
        Field::new(ROW_FILE_COLUMN, ArrowType::BinaryView, true),
        Field::new(ROW_POSITION_COLUMN, ArrowType::Int64, true),
        Field::new(ROW_WRITE_COLUMN, ArrowType::Boolean, false),
    ]))
}

fn marked(rows: &[(Vec<u8>, i64)]) -> RecordBatch {
    let n = rows.len();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(vec![None::<i64>; n])),
            Arc::new(StringArray::from(vec![None::<&str>; n])),
            Arc::new(Int32Array::from(vec![None::<i32>; n])),
            Arc::new(BinaryViewArray::from(
                rows.iter()
                    .map(|r| Some(r.0.as_slice()))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                rows.iter().map(|r| Some(r.1)).collect::<Vec<_>>(),
            )),
            Arc::new(BooleanArray::from(vec![false; n])),
        ],
    )
    .unwrap()
}

fn eq(column: &str, value: Literal) -> Expr {
    Expr::function("=", vec![Expr::column([column]), Expr::literal(value)])
}

/// A merge-on-read `DELETE` of the rows `ids` (`condition` as written and parsed).
#[expect(clippy::unwrap_in_result, reason = "test helper")]
fn delete(
    env: &Env,
    rl: &Arc<dyn RowLevelTable>,
    ids: &[i64],
    condition: (&str, Expr),
    write_id: &str,
) -> Result<(), KrishivError> {
    let files = row_files(env, &env.open().unwrap(), RowLevelOperation::Delete);
    delete_keyed(env, rl, &files, ids, condition, write_id, None)
}

/// [`delete`] of the rows `ids` among `files` (the row ids of the snapshot the statement read),
/// under `krishiv.write.idempotencyKey` = `key` (E34).
#[expect(clippy::needless_pass_by_value, reason = "test helper")]
fn delete_keyed(
    env: &Env,
    rl: &Arc<dyn RowLevelTable>,
    files: &[(i64, Vec<u8>, i64)],
    ids: &[i64],
    condition: (&str, Expr),
    write_id: &str,
    key: Option<&str>,
) -> Result<(), KrishivError> {
    let rows: Vec<(Vec<u8>, i64)> = files
        .iter()
        .filter(|f| ids.contains(&f.0))
        .map(|f| (f.1.clone(), f.2))
        .collect();
    let mut touched: Vec<Vec<u8>> = rows.iter().map(|r| r.0.clone()).collect();
    touched.sort();
    touched.dedup();
    let mut request = RowLevelWrite::new(
        RowLevelOperation::Delete,
        "`t`",
        schema(),
        touched,
        Some(Arc::new(condition.1.clone())),
    );
    request.write_id = Some(write_id.to_owned());
    if let Some(key) = key {
        request
            .options
            .insert("krishiv.write.idempotencyKey".to_owned(), key.to_owned());
    }
    for (k, v) in [
        (ROW_LEVEL_AFFECTED_OPTION, ids.len().to_string()),
        (ROW_LEVEL_ALIAS_OPTION, "t".to_owned()),
        (ROW_LEVEL_CONDITION_OPTION, condition.0.to_owned()),
        (MERGE_ON_READ_OPTION, "true".to_owned()),
    ] {
        request.options.insert(k.to_owned(), v);
    }
    let sink = rl.rewriter(&request)?;
    let input = krishiv_exec::planner::MemoryTable::new(
        schema(),
        vec![krishiv_types::batch::Batch::new(marked(&rows))],
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

/// The deletion vector files under the table root.
fn vector_files(env: &Env) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(&env.dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("deletion_vector_"))
        .collect();
    v.sort();
    v
}

fn ids(env: &Env) -> Vec<i64> {
    let mut out: Vec<i64> = row_files(env, &env.open().unwrap(), RowLevelOperation::Delete)
        .into_iter()
        .map(|f| f.0)
        .collect();
    out.sort_unstable();
    out
}

#[test]
fn merge_on_read_applies_as_delta_spark_decides() {
    let env = Env::fixture("dvs/unpartitioned/before");
    let table = env.open().unwrap();
    for op in [
        RowLevelOperation::Delete,
        RowLevelOperation::Update,
        RowLevelOperation::Merge,
    ] {
        let rl = table.row_level(op).unwrap();
        // Never the one-query (Iceberg) path: the session runs the find query first.
        assert!(!rl.capabilities().merge_on_read, "{op:?}");
        // A data predicate marks rows; no condition (cases 1 and 2) removes whole files.
        let data = eq("id", Literal::Long(2));
        assert!(rl.merge_on_read(Some(&data), Some("t")), "{op:?}");
        assert_eq!(rl.merge_on_read(None, None), op == RowLevelOperation::Merge);
    }
    // A partition-only condition removes or rewrites whole files.
    let env = Env::fixture("dvs/partitioned/before");
    let rl = env
        .open()
        .unwrap()
        .row_level(RowLevelOperation::Delete)
        .unwrap();
    let p = eq("p", Literal::String("x".into()));
    assert!(!rl.merge_on_read(Some(&p), Some("t")));
    // A table without `delta.enableDeletionVectors`: copy-on-write, and the option is refused.
    let env = Env::fixture("dml/unpartitioned/before");
    let rl = env
        .open()
        .unwrap()
        .row_level(RowLevelOperation::Delete)
        .unwrap();
    assert!(!rl.merge_on_read(Some(&eq("id", Literal::Long(2))), None));
    let err = delete(
        &env,
        &rl,
        &[2],
        ("id = 2", eq("id", Literal::Long(2))),
        "dv-refused",
    )
    .unwrap_err();
    assert_eq!(err.error_class().name(), "INTERNAL_ERROR", "{err}");
}

#[test]
fn a_dv_delete_commits_a_vector_and_reads_back() {
    let env = Env::fixture("dvs/unpartitioned/before");
    let version = env.latest();
    let rl = env
        .open()
        .unwrap()
        .row_level(RowLevelOperation::Delete)
        .unwrap();
    delete(
        &env,
        &rl,
        &[2, 11],
        ("id IN (2, 11)", eq("id", Literal::Long(2))),
        "dv-commit",
    )
    .unwrap();
    assert_eq!(env.latest(), version + 1);
    assert_eq!(vector_files(&env).len(), 2, "the old file and this write's");
    let left = ids(&env);
    assert!(!left.contains(&2) && !left.contains(&11) && left.contains(&3));
    assert_eq!(left.len(), 90 - 2);
}

/// E34: a merge-on-read retry under the key of a statement that landed commits nothing, and its
/// own deletion vector file is deleted with its other files (the landed attempt's vector stays).
#[test]
fn a_keyed_dv_delete_retried_leaves_no_vector_file_of_its_own() {
    let env = Env::fixture("dvs/unpartitioned/before");
    let table = env.open().unwrap();
    let rl = table.row_level(RowLevelOperation::Delete).unwrap();
    assert!(rl.capabilities().idempotent_commit);
    let files = row_files(&env, &table, RowLevelOperation::Delete);
    let before = vector_files(&env).len();
    let attempt = |write_id: &str| {
        delete_keyed(
            &env,
            &rl,
            &files,
            &[2],
            ("id = 2", eq("id", Literal::Long(2))),
            write_id,
            Some("job-dv"),
        )
    };
    attempt("dv-key-1").unwrap();
    let version = env.latest();
    let landed = vector_files(&env);
    assert_eq!(landed.len(), before + 1, "the landed attempt's vector");
    attempt("dv-key-2").unwrap();
    assert_eq!(env.latest(), version, "the retry committed nothing");
    assert_eq!(
        vector_files(&env),
        landed,
        "the retry's vector file is deleted"
    );
    assert!(!ids(&env).contains(&2));
}

/// delta-spark's checker on a winner committed after the statement read its snapshot: a
/// vector update of a file is a remove and an add of it, so the statement (a whole-table read:
/// no partition conjunct) meets the added file first (`DELTA_CONCURRENT_APPEND`); a whole-file
/// remove of a file it read is `DELTA_CONCURRENT_DELETE_READ`. The loser leaves the version
/// as the winner left it.
#[test]
fn concurrent_vector_updates_conflict_as_delta_spark_checks() {
    for (winner, class) in [
        ("vector", "DELTA_CONCURRENT_APPEND.WITHOUT_HINT"),
        ("whole-file", "DELTA_CONCURRENT_DELETE_READ.WITHOUT_HINT"),
    ] {
        let env = Env::fixture("dvs/unpartitioned/before");
        let table = env.open().unwrap();
        let loser = table.row_level(RowLevelOperation::Delete).unwrap();
        let other = env
            .open()
            .unwrap()
            .row_level(RowLevelOperation::Delete)
            .unwrap();
        if winner == "vector" {
            delete(
                &env,
                &other,
                &[3],
                ("id = 3", eq("id", Literal::Long(3))),
                "dv-winner",
            )
            .unwrap();
        } else {
            // A `DELETE` of every row of the first file (fully replaced: removed whole).
            delete(
                &env,
                &other,
                &[1, 2, 3],
                ("id <= 3", eq("id", Literal::Long(3))),
                "dv-winner-whole",
            )
            .unwrap();
        }
        let after = env.latest();
        // The loser's row ids are those of its own snapshot.
        let files = row_files(&env, &table, RowLevelOperation::Delete);
        let row = files.iter().find(|f| f.0 == 2).unwrap();
        let mut request = RowLevelWrite::new(
            RowLevelOperation::Delete,
            "`t`",
            schema(),
            vec![row.1.clone()],
            Some(Arc::new(eq("id", Literal::Long(2)))),
        );
        request.write_id = Some(format!("dv-loser-{winner}"));
        for (k, v) in [
            (ROW_LEVEL_AFFECTED_OPTION, "1"),
            (ROW_LEVEL_ALIAS_OPTION, "t"),
            (ROW_LEVEL_CONDITION_OPTION, "id = 2"),
            (MERGE_ON_READ_OPTION, "true"),
        ] {
            request.options.insert(k.to_owned(), v.to_owned());
        }
        let sink = loser.rewriter(&request).unwrap();
        let input = krishiv_exec::planner::MemoryTable::new(
            schema(),
            vec![krishiv_types::batch::Batch::new(marked(&[(
                row.1.clone(),
                row.2,
            )]))],
        )
        .scan()
        .unwrap()
        .build(ScanPushdowns::default())
        .unwrap();
        let err = env
            .run(Arc::new(Writing {
                input: Arc::new(Leaf(input)),
                sink,
            }))
            .unwrap_err();
        assert_eq!(err.error_class().name(), class, "{winner}: {err}");
        assert_eq!(env.latest(), after, "{winner}");
        assert_eq!(env.query_pool.reserved(), 0);
    }
}

/// Vector updates of files in other partitions than the statement reads do not conflict: both
/// commit and both vectors apply.
#[test]
fn vector_updates_of_other_partitions_both_commit() {
    let env = Env::fixture("dvs/partitioned/before");
    let table = env.open().unwrap();
    let loser = table.row_level(RowLevelOperation::Delete).unwrap();
    let other = env
        .open()
        .unwrap()
        .row_level(RowLevelOperation::Delete)
        .unwrap();
    let p = |v: &str| eq("p", Literal::String(v.into()));
    let and = |a: Expr, b: Expr| Expr::function("and", vec![a, b]);
    // `p = 'y' AND id = 4` (the second file of `y`) first, then `p = 'x' AND id = 1`.
    delete(
        &env,
        &other,
        &[4],
        (
            "p = 'y' AND id = 4",
            and(p("y"), eq("id", Literal::Long(4))),
        ),
        "dv-other-partition",
    )
    .unwrap();
    let files = row_files(&env, &table, RowLevelOperation::Delete);
    let row = files.iter().find(|f| f.0 == 1).unwrap().clone();
    let mut request = RowLevelWrite::new(
        RowLevelOperation::Delete,
        "`t`",
        schema(),
        vec![row.1.clone()],
        Some(Arc::new(and(p("x"), eq("id", Literal::Long(1))))),
    );
    request.write_id = Some("dv-this-partition".to_owned());
    for (k, v) in [
        (ROW_LEVEL_AFFECTED_OPTION, "1"),
        (ROW_LEVEL_ALIAS_OPTION, "t"),
        (ROW_LEVEL_CONDITION_OPTION, "p = 'x' AND id = 1"),
        (MERGE_ON_READ_OPTION, "true"),
    ] {
        request.options.insert(k.to_owned(), v.to_owned());
    }
    let sink = loser.rewriter(&request).unwrap();
    let input = krishiv_exec::planner::MemoryTable::new(
        schema(),
        vec![krishiv_types::batch::Batch::new(marked(&[(row.1, row.2)]))],
    )
    .scan()
    .unwrap()
    .build(ScanPushdowns::default())
    .unwrap();
    env.run(Arc::new(Writing {
        input: Arc::new(Leaf(input)),
        sink,
    }))
    .unwrap();
    assert_eq!(ids(&env), vec![2, 3, 5, 6]);
}

#[test]
fn a_failure_before_the_commit_deletes_the_vector_file() {
    let env = Env::fixture("dvs/unpartitioned/before");
    let (version, before) = (env.latest(), vector_files(&env));
    let rl = env
        .open()
        .unwrap()
        .row_level(RowLevelOperation::Delete)
        .unwrap();
    let id = "dv-fail-in-stage";
    test_hooks::fail(id);
    let err = delete(
        &env,
        &rl,
        &[2, 50],
        ("id = 2", eq("id", Literal::Long(2))),
        id,
    )
    .unwrap_err();
    assert!(err.to_string().contains("injected"), "{err}");
    assert_eq!(env.latest(), version);
    assert_eq!(vector_files(&env), before, "the vector file was deleted");
}

/// A merge-on-read `UPDATE`-shaped rewrite: `marks` deleted, `written` rows written (their row
/// ids those of `marks`, one each), through `rl`.
fn update(
    env: &Env,
    rl: &Arc<dyn RowLevelTable>,
    marks: &[(Vec<u8>, i64)],
    written: &[(i64, &str, i32)],
    write_id: &str,
) -> Result<(), KrishivError> {
    let n = marks.len();
    let batch = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(
                (0..n)
                    .map(|i| written.get(i).map(|w| w.0))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (0..n)
                    .map(|i| written.get(i).map(|w| w.1))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                (0..n)
                    .map(|i| written.get(i).map(|w| w.2))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(BinaryViewArray::from(
                marks
                    .iter()
                    .map(|r| Some(r.0.as_slice()))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                marks.iter().map(|r| Some(r.1)).collect::<Vec<_>>(),
            )),
            Arc::new(BooleanArray::from(
                (0..n).map(|i| i < written.len()).collect::<Vec<_>>(),
            )),
        ],
    )
    .map_err(|e| {
        KrishivError::from(krishiv_common::error::InternalError {
            message: e.to_string(),
        })
    })?;
    let mut touched: Vec<Vec<u8>> = marks.iter().map(|r| r.0.clone()).collect();
    touched.sort();
    touched.dedup();
    let mut request = RowLevelWrite::new(
        RowLevelOperation::Update,
        "`t`",
        schema(),
        touched,
        Some(Arc::new(eq("id", Literal::Long(2)))),
    );
    request.write_id = Some(write_id.to_owned());
    for (k, v) in [
        (ROW_LEVEL_AFFECTED_OPTION, n.to_string()),
        (ROW_LEVEL_ALIAS_OPTION, "t".to_owned()),
        (ROW_LEVEL_CONDITION_OPTION, "id = 2".to_owned()),
        (MERGE_ON_READ_OPTION, "true".to_owned()),
    ] {
        request.options.insert(k.to_owned(), v);
    }
    let sink = rl.rewriter(&request)?;
    let input = krishiv_exec::planner::MemoryTable::new(
        schema(),
        vec![krishiv_types::batch::Batch::new(batch)],
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

/// The fixture's first file (ids 1 to 3) with its statistics removed from the log.
fn without_statistics(env: &Env) {
    let path = env.dir.join("_delta_log/00000000000000000001.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<String> = text
        .lines()
        .map(|l| {
            let mut v: serde_json::Value = serde_json::from_str(l).unwrap();
            if let Some(add) = v.get_mut("add").and_then(serde_json::Value::as_object_mut) {
                add.remove("stats");
            }
            v.to_string()
        })
        .collect();
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    let crc = env.dir.join("_delta_log/00000000000000000001.crc");
    let _ = std::fs::remove_file(crc);
}

/// delta-spark's rule for a file whose statistics lack `numRecords`
/// (`tests/fixtures/delta-4.4.1/dvs-no-stats/NOSTATS.json`): the statement commits, the file's
/// new add carries the footer's row count, and a file whose every row goes keeps a vector of
/// all its rows (only statistics prove a file is fully replaced).
#[test]
fn a_file_without_statistics_gets_the_footer_count_and_is_never_removed_whole() {
    let env = Env::fixture("dvs/unpartitioned/before");
    without_statistics(&env);
    let table = env.open().unwrap();
    let file = row_files(&env, &table, RowLevelOperation::Delete)
        .into_iter()
        .find(|f| f.0 == 1)
        .unwrap()
        .1;
    assert!(
        table
            .units()
            .iter()
            .any(|u| u.path.as_bytes() == file && u.records.is_none())
    );
    let rl = table.row_level(RowLevelOperation::Delete).unwrap();
    delete(
        &env,
        &rl,
        &[1, 2, 3],
        ("id <= 3", eq("id", Literal::Long(3))),
        "dv-no-stats",
    )
    .unwrap();
    let log = env.log(env.latest());
    let path = std::str::from_utf8(&file).unwrap();
    let add = log
        .iter()
        .filter_map(|a| a.get("add"))
        .find(|a| a["path"] == path)
        .expect("the file is added again with its vector, not removed");
    assert_eq!(add["deletionVector"]["cardinality"], 3);
    let stats: serde_json::Value = serde_json::from_str(add["stats"].as_str().unwrap()).unwrap();
    assert_eq!(stats["numRecords"], 3);
    assert_eq!(stats["tightBounds"], false);
    let left = ids(&env);
    assert!(!left.contains(&1) && !left.contains(&3) && left.contains(&4));
}

/// A merge-on-read rewrite whose marks the pool cannot hold fails with the memory class, gives
/// the pool back, writes no file and leaves the version.
#[test]
fn marks_the_pool_cannot_hold_fail_the_statement_cleanly() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("t");
    super::copy_dir(&super::fixtures().join("dvs/unpartitioned/before"), &dir);
    let env = Env::at_with(
        tmp,
        dir,
        &krishiv_memory::MemoryConfig {
            limit: Some(4 << 20),
            quantum: 64 << 10,
            ..krishiv_memory::MemoryConfig::default()
        },
    );
    let (version, before) = (env.latest(), vector_files(&env));
    let file = row_files(&env, &env.open().unwrap(), RowLevelOperation::Delete)
        .into_iter()
        .find(|f| f.0 == 2)
        .unwrap()
        .1;
    // Two million marks, one per 2^16-row container: an 8 MiB bitmap bound at least, twice the
    // pool (the batch itself is under 1 MiB of views over one path).
    let marks: Vec<(Vec<u8>, i64)> = (0..2_000_000_i64)
        .map(|i| (file.clone(), i * 70_000))
        .collect();
    let rl = env
        .open()
        .unwrap()
        .row_level(RowLevelOperation::Delete)
        .unwrap();
    let err = update(&env, &rl, &marks, &[], "dv-marks-oom").unwrap_err();
    assert_eq!(err.sql_state(), Some("53200"), "{err}");
    assert_eq!(env.latest(), version);
    assert_eq!(vector_files(&env), before);
    assert_eq!(env.query_pool.reserved(), 0);
}

/// A statement cancelled while it writes its vector file (after a vector, before the file is
/// finished): the version stays, the vector file and the written data file are deleted, the
/// pool is back to zero.
#[test]
fn a_cancel_while_vectors_are_written_leaves_no_file() {
    let env = Env::fixture("dvs/unpartitioned/before");
    let (version, vectors, data) = (
        env.latest(),
        vector_files(&env),
        super::streaming::data_files(&env.dir),
    );
    let table = env.open().unwrap();
    let files = row_files(&env, &table, RowLevelOperation::Update);
    let marks: Vec<(Vec<u8>, i64)> = files
        .iter()
        .filter(|f| f.0 == 2 || f.0 == 11)
        .map(|f| (f.1.clone(), f.2))
        .collect();
    let id = "dv-cancel-vector-write";
    test_hooks::set(
        id,
        Arc::new(|at, query| {
            if at.point == test_hooks::Point::VectorWritten {
                query.cancel();
                test_hooks::Act::stall(std::time::Duration::from_secs(30))
            } else {
                test_hooks::Act::NONE
            }
        }),
    );
    let rl = table.row_level(RowLevelOperation::Update).unwrap();
    let err = update(&env, &rl, &marks, &[(2, "u", 2), (11, "u", 4)], id).unwrap_err();
    assert_eq!(err.sql_state(), Some("HY008"), "{err}");
    assert_eq!(env.latest(), version);
    let mut gone = false;
    for _ in 0..100 {
        if vector_files(&env) == vectors && super::streaming::data_files(&env.dir) == data {
            gone = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(gone, "a cancelled statement left a vector or data file");
    assert_eq!(env.query_pool.reserved(), 0);
}
