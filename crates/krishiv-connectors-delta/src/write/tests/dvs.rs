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
#[expect(clippy::needless_pass_by_value, reason = "test helper")]
fn delete(
    env: &Env,
    rl: &Arc<dyn RowLevelTable>,
    ids: &[i64],
    condition: (&str, Expr),
    write_id: &str,
) -> Result<(), KrishivError> {
    let files = row_files(env, &env.open().unwrap(), RowLevelOperation::Delete);
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
        assert!(rl.capabilities().merge_on_read, "{op:?}");
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
    assert!(!rl.capabilities().merge_on_read);
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
