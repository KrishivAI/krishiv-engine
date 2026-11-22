// SPDX-License-Identifier: Apache-2.0
//! Copy-on-write `MERGE INTO` through [`DeltaRowLevel`] (PR-37c), driven as the session drives
//! it (the touched files and counts from its find query, then the rewrite's rows): the commit's
//! removes, adds, parameters, metrics and result row, an insert-only `MERGE`, the conflict
//! checker against a concurrent change, a cancelled rewrite and the refusals (append-only and
//! change data feed only when a file is removed). The comparison with delta-spark is the
//! session's `delta_merge` test (`tests/fixtures/delta-4.4.1/merge/MERGE.json`).

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_schema::SchemaRef;
use krishiv_common::error::KrishivError;
use krishiv_exec::ScanPushdowns;
use krishiv_exec::planner::{
    RowLevelCounts, RowLevelOperation, RowLevelTable, RowLevelWrite, TableProvider, WriteMode,
};
use krishiv_plan::Expr;

use super::super::dml::{
    ROW_LEVEL_ALIAS_OPTION, ROW_LEVEL_CONDITION_OPTION, ROW_LEVEL_SCAN_MS_OPTION,
};
use super::super::merge::{MERGE_CLAUSES_OPTION, MERGE_COUNTS_OPTION};
use super::super::test_hooks::{self, Act, Point};
use super::dml::{row_files, the_file_of, two_file_table};
use super::streaming::data_files;
use super::{Env, Leaf, Writing, commit_info, modes_schema, rows};
use crate::table::DeltaTable;

const UPSERT: &str = r#"[{"kind":"matched","action":"update","condition":null},
    {"kind":"notMatched","action":"insert","condition":null}]"#;

const INSERT_ONLY: &str = r#"[{"kind":"notMatched","action":"insert","condition":null}]"#;

/// The rewrite of `touched` with `batches` through the `MERGE` table of `table`.
#[expect(
    clippy::too_many_arguments,
    reason = "test helper: the request's parts"
)]
fn merge(
    env: &Env,
    table: &DeltaTable,
    schema: SchemaRef,
    touched: Vec<Vec<u8>>,
    batches: Vec<RecordBatch>,
    counts: &str,
    clauses: &str,
    write_id: &str,
) -> Result<Arc<dyn RowLevelTable>, KrishivError> {
    let rl = table.row_level(RowLevelOperation::Merge).unwrap();
    // A `MERGE` reads an append-only or change-data-feed table: it may only insert.
    rl.scan()?;
    let mut request = RowLevelWrite::new(
        RowLevelOperation::Merge,
        "`t`",
        Arc::clone(&schema),
        touched,
        None::<Arc<Expr>>,
    );
    request.write_id = Some(write_id.to_owned());
    for (k, v) in [
        (MERGE_COUNTS_OPTION, counts),
        (MERGE_CLAUSES_OPTION, clauses),
        (ROW_LEVEL_CONDITION_OPTION, "t.id = s.id"),
        (ROW_LEVEL_SCAN_MS_OPTION, "2"),
        (ROW_LEVEL_ALIAS_OPTION, "t"),
    ] {
        request.options.insert(k.to_owned(), v.to_owned());
    }
    let sink = rl.rewriter(&request)?;
    let input = krishiv_exec::planner::MemoryTable::new(
        schema,
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
    // As in `Env::write_as`: an ended driver's row groups are shipped by a background flush that returns its
    // memory a moment after the query finished.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while env.query_pool.reserved() != 0 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(
        env.query_pool.reserved(),
        0,
        "the rewrite gave its memory back"
    );
    out.map(|_| rl)
}

fn result_row(rl: &Arc<dyn RowLevelTable>) -> Vec<i64> {
    let b = rl.output(&RowLevelCounts::default()).unwrap();
    let names: Vec<String> = b
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(
        names,
        [
            "num_affected_rows",
            "num_updated_rows",
            "num_deleted_rows",
            "num_inserted_rows"
        ]
    );
    b.columns()
        .iter()
        .map(|c| c.as_primitive::<Int64Type>().value(0))
        .collect()
}

#[test]
fn a_merge_removes_the_touched_file_and_adds_its_rows_and_the_inserts() {
    let env = Env::new();
    two_file_table(&env, &[]);
    let table = env.open().unwrap();
    let files = row_files(&env, &table, RowLevelOperation::Merge);
    let touched = the_file_of(&files, 2);
    let before = env.latest();
    let before_rows = super::typed_rows(&env.read(&table));
    // `USING (2 'B', 7 'g') ON t.id = s.id WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED
    // THEN INSERT *`: id 2 updated, 1 and 3 copied, 7 inserted.
    let rl = merge(
        &env,
        &table,
        modes_schema(),
        vec![touched.clone()],
        rows(&[
            (1, "a", Some("x")),
            (2, "B", Some("x")),
            (3, "c", Some("y")),
            (7, "g", None),
        ]),
        r#"{"source_rows":2,"matched_updated":1,"inserted":1,"materialize_ms":1}"#,
        UPSERT,
        "merge-upsert",
    )
    .unwrap();
    assert_eq!(env.latest(), before + 1);
    let log = env.log(env.latest());
    let info = commit_info(&log);
    assert_eq!(info["operation"], "MERGE");
    let p = &info["operationParameters"];
    assert_eq!(p["predicate"], r#"["(t.id = s.id)"]"#);
    assert_eq!(p["matchedPredicates"], r#"[{"actionType":"update"}]"#);
    assert_eq!(p["notMatchedPredicates"], r#"[{"actionType":"insert"}]"#);
    assert_eq!(p["notMatchedBySourcePredicates"], "[]");
    let m = &info["operationMetrics"];
    for (k, v) in [
        ("numSourceRows", "2"),
        ("numTargetRowsUpdated", "1"),
        ("numTargetRowsMatchedUpdated", "1"),
        ("numTargetRowsInserted", "1"),
        ("numTargetRowsCopied", "2"),
        ("numOutputRows", "4"),
        ("numTargetFilesRemoved", "1"),
        ("numTargetFilesAdded", "1"),
        ("materializeSourceTimeMs", "1"),
        ("scanTimeMs", "2"),
    ] {
        assert_eq!(m[k], v, "{k}");
    }
    let removed: Vec<&str> = log
        .iter()
        .filter_map(|a| a.get("remove"))
        .map(|r| r["path"].as_str().unwrap())
        .collect();
    assert_eq!(removed, [std::str::from_utf8(&touched).unwrap()]);
    // Every row as it was but id 2's, which took the source's `v`, plus the insert.
    let mut want: Vec<serde_json::Value> = before_rows
        .into_iter()
        .map(|r| {
            if r[0] == 2 {
                serde_json::json!([2, "B", "x"])
            } else {
                r
            }
        })
        .collect();
    want.push(serde_json::json!([7, "g", null]));
    let mut got = super::typed_rows(&env.read(&env.open().unwrap()));
    let id = |r: &serde_json::Value| r[0].as_i64().unwrap();
    want.sort_by_key(id);
    got.sort_by_key(id);
    assert_eq!(got, want);
    assert_eq!(result_row(&rl), [2, 1, 0, 1]);
}

#[test]
fn an_insert_only_merge_counts_the_written_rows_and_removes_nothing() {
    let env = Env::new();
    two_file_table(&env, &[]);
    let table = env.open().unwrap();
    let rl = merge(
        &env,
        &table,
        modes_schema(),
        Vec::new(),
        rows(&[(7, "g", None), (8, "h", None)]),
        r#"{"source_rows":3,"inserted":null}"#,
        INSERT_ONLY,
        "merge-insert-only",
    )
    .unwrap();
    let log = env.log(env.latest());
    let m = &commit_info(&log)["operationMetrics"];
    assert_eq!(m["numTargetRowsInserted"], "2");
    assert_eq!(m["numTargetFilesRemoved"], "0");
    assert!(log.iter().all(|a| a.get("remove").is_none()));
    assert_eq!(result_row(&rl), [2, 0, 0, 2]);
}

#[test]
fn a_merge_that_changes_nothing_commits_nothing() {
    let env = Env::new();
    two_file_table(&env, &[]);
    let table = env.open().unwrap();
    let before = env.latest();
    let rl = merge(
        &env,
        &table,
        modes_schema(),
        Vec::new(),
        Vec::new(),
        r#"{"source_rows":1,"inserted":0}"#,
        UPSERT,
        "merge-nothing",
    )
    .unwrap();
    assert_eq!(env.latest(), before);
    assert_eq!(result_row(&rl), [0, 0, 0, 0]);
}

/// The read set is the whole table here (no partition conjunct): a concurrent append or a
/// concurrent delete of the touched file conflicts, as delta-spark's checker says.
#[test]
fn a_concurrent_change_conflicts_with_a_merge() {
    for (winner, class) in [
        ("truncate", "DELTA_CONCURRENT_DELETE_READ.WITHOUT_HINT"),
        ("append", "DELTA_CONCURRENT_APPEND.WITHOUT_HINT"),
    ] {
        let env = Env::new();
        two_file_table(&env, &[]);
        let table = env.open().unwrap();
        let files = row_files(&env, &table, RowLevelOperation::Merge);
        let touched = the_file_of(&files, 2);
        if winner == "truncate" {
            // `DELETE FROM t` by another writer: every file removed.
            let other = env.open().unwrap();
            let all: Vec<Vec<u8>> = other
                .units()
                .iter()
                .map(|u| u.path.as_bytes().to_vec())
                .collect();
            let rl = other.row_level(RowLevelOperation::Delete).unwrap();
            let mut request =
                RowLevelWrite::new(RowLevelOperation::Delete, "`t`", modes_schema(), all, None);
            request.write_id = Some("merge-conflict-winner".to_owned());
            let sink = rl.rewriter(&request).unwrap();
            let input = krishiv_exec::planner::MemoryTable::new(modes_schema(), Vec::new())
                .scan()
                .unwrap()
                .build(ScanPushdowns::default())
                .unwrap();
            env.run(Arc::new(Writing {
                input: Arc::new(Leaf(input)),
                sink,
            }))
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
        let err = merge(
            &env,
            &table,
            modes_schema(),
            vec![touched],
            rows(&[(1, "a", Some("x")), (3, "c", Some("y"))]),
            r#"{"source_rows":1,"matched_deleted":1,"inserted":0}"#,
            r#"[{"kind":"matched","action":"delete","condition":null}]"#,
            &format!("merge-conflict-{winner}"),
        )
        .unwrap_err();
        assert_eq!(err.error_class().name(), class, "{winner}: {err}");
        assert_eq!(env.latest(), after, "{winner}");
    }
}

#[test]
fn a_merge_cancelled_during_its_rewrite_leaves_the_table_unchanged() {
    let env = Env::new();
    two_file_table(&env, &[]);
    let table = env.open().unwrap();
    let files = row_files(&env, &table, RowLevelOperation::Merge);
    let touched = the_file_of(&files, 2);
    let (version, before) = (env.latest(), data_files(&env.dir));
    let id = "merge-cancel-during-sink";
    let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let seen = Arc::clone(&fired);
    test_hooks::set(
        id,
        Arc::new(move |site, query| {
            if site.point != Point::Sunk {
                return Act::NONE;
            }
            seen.store(true, std::sync::atomic::Ordering::SeqCst);
            query.cancel();
            Act::stall(std::time::Duration::from_secs(30))
        }),
    );
    let out = merge(
        &env,
        &table,
        modes_schema(),
        vec![touched],
        rows(&[
            (1, "a", Some("x")),
            (2, "B", Some("x")),
            (3, "c", Some("y")),
        ]),
        r#"{"source_rows":1,"matched_updated":1,"inserted":0}"#,
        UPSERT,
        id,
    );
    let Err(e) = out else {
        panic!("a cancelled MERGE committed")
    };
    assert!(
        fired.load(std::sync::atomic::Ordering::SeqCst),
        "the cancel came from the rewrite's hook"
    );
    assert_eq!(e.error_class().name(), "OPERATION_CANCELED", "{e}");
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
}

/// `id BIGINT` rows of the `write-features` fixtures.
fn ids(values: &[i64]) -> (SchemaRef, Vec<RecordBatch>) {
    let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "id",
        arrow_schema::DataType::Int64,
        true,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(arrow_array::Int64Array::from(values.to_vec()))],
    )
    .unwrap();
    (schema, vec![batch])
}

/// delta-spark refuses a `MERGE` that removes a file of an append-only table and commits one
/// that only inserts (`merge/MERGE.json` `append-only`); Krishiv refuses one that removes a
/// file of a change-data-feed table (design section 6.2) and commits one that only inserts.
#[test]
fn append_only_and_change_data_feed_tables_take_only_inserting_merges() {
    for (fixture, class) in [
        (
            "write-features/append-only",
            "DELTA_CANNOT_MODIFY_APPEND_ONLY",
        ),
        (
            "write-features/change-data-feed",
            "UNSUPPORTED_FEATURE.TABLE_OPERATION",
        ),
    ] {
        let env = Env::fixture(fixture);
        let table = env.open().unwrap();
        let before = env.latest();
        let some_file = table.units()[0].path.as_bytes().to_vec();
        let (schema, batches) = ids(&[1]);
        let Err(err) = merge(
            &env,
            &table,
            Arc::clone(&schema),
            vec![some_file],
            batches,
            r#"{"source_rows":1,"matched_deleted":1,"inserted":0}"#,
            r#"[{"kind":"matched","action":"delete","condition":null}]"#,
            "merge-refused",
        ) else {
            panic!("{fixture}: a MERGE removing a file was accepted");
        };
        assert_eq!(err.error_class().name(), class, "{fixture}: {err}");
        assert_eq!(env.latest(), before, "{fixture}");
        let (schema, batches) = ids(&[100, 101]);
        merge(
            &env,
            &table,
            schema,
            Vec::new(),
            batches,
            r#"{"source_rows":2,"inserted":null}"#,
            INSERT_ONLY,
            "merge-inserts",
        )
        .unwrap();
        assert_eq!(env.latest(), before + 1, "{fixture}");
        assert_eq!(commit_info(&env.log(env.latest()))["operation"], "MERGE");
    }
}

/// A `MERGE` that rewrites a file carrying a deletion vector reports no deletion vector removed,
/// as delta-spark 4.4.1's copy-on-write `MERGE` does (`merge/MERGE.json`
/// `deletion-vectors/update_copy_on_write`: a removed file with a DV of cardinality 10, metric
/// `0`). The table is delta-spark's `deletion-vectors` fixture: one file of `range(40000)` with a DV.
#[test]
fn a_merge_that_rewrites_a_file_with_a_deletion_vector_reports_none_removed() {
    let env = Env::fixture("deletion-vectors");
    let table = env.open().unwrap();
    let files = row_files(&env, &table, RowLevelOperation::Merge);
    let touched = files[0].1.clone();
    let live: Vec<i64> = files.iter().map(|(id, ..)| *id).collect();
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("id", arrow_schema::DataType::Int64, true),
        arrow_schema::Field::new("v", arrow_schema::DataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(arrow_array::Int64Array::from(live.clone())),
            Arc::new(arrow_array::Int32Array::from(vec![0; live.len()])),
        ],
    )
    .unwrap();
    merge(
        &env,
        &table,
        schema,
        vec![touched],
        vec![batch],
        r#"{"source_rows":1,"matched_updated":1,"inserted":0,"materialize_ms":0}"#,
        UPSERT,
        "merge-dv",
    )
    .unwrap();
    let info = commit_info(&env.log(env.latest()));
    let m = &info["operationMetrics"];
    assert_eq!(m["numTargetFilesRemoved"], "1");
    assert_eq!(m["numTargetDeletionVectorsRemoved"], "0");
    assert_eq!(m["numTargetDeletionVectorsAdded"], "0");
}

/// The classes a Delta table gives the session planner's problems (`DeltaRowLevel::error`):
/// delta-spark's for a `MERGE`, its `DELETE`/`UPDATE` names for a subquery, and none for what
/// it leaves to the session (a duplicate assignment of a `DELETE` is not a Delta problem).
#[test]
fn the_merge_problems_take_delta_spark_s_classes() {
    use krishiv_common::error::{ErrorClass, InternalError};
    use krishiv_exec::planner::RowLevelProblem;

    let env = Env::new();
    two_file_table(&env, &[]);
    let table = env.open().unwrap();
    let rl = table.row_level(RowLevelOperation::Merge).unwrap();
    let merge = RowLevelOperation::Merge;
    let class = |op: RowLevelOperation, p: &RowLevelProblem<'_>| {
        rl.error(op, p)
            .map(|e| (e.error_class(), e.message().to_owned()))
    };

    let cause: KrishivError = InternalError {
        message: "unresolved".to_owned(),
    }
    .into();
    let (c, m) = class(merge, &RowLevelProblem::Unresolved(&cause)).unwrap();
    assert_eq!(c, ErrorClass::InternalError);
    assert!(m.contains("DELTA_MERGE_UNRESOLVED_EXPRESSION"), "{m}");
    assert!(
        class(
            RowLevelOperation::Update,
            &RowLevelProblem::Unresolved(&cause)
        )
        .is_none()
    );

    let nd = RowLevelProblem::NonDeterministic {
        what: "search condition",
        condition: "rand() > 0.5",
    };
    let (c, m) = class(merge, &nd).unwrap();
    assert_eq!(c, ErrorClass::DeltaNonDeterministicFunctionNotSupported);
    assert!(m.contains("search condition of MERGE operation"), "{m}");
    assert!(m.contains("(condition = rand() > 0.5)"), "{m}");

    let sub = RowLevelProblem::Subquery {
        what: "UPDATE condition",
        condition: "id IN (SELECT 1)",
    };
    let (c, m) = class(merge, &sub).unwrap();
    assert_eq!(c, ErrorClass::DeltaUnsupportedSubquery);
    assert!(m.contains("UPDATE condition of MERGE operation"), "{m}");
    let (_, m) = class(RowLevelOperation::Update, &sub).unwrap();
    assert!(m.contains("UPDATE"), "{m}");
    let (_, m) = class(RowLevelOperation::Delete, &sub).unwrap();
    assert!(m.contains("DELETE"), "{m}");

    let columns = vec!["s".to_owned(), "a`b".to_owned()];
    let dup = RowLevelProblem::DuplicateAssignment { columns: &columns };
    let (c, m) = class(merge, &dup).unwrap();
    assert_eq!(c, ErrorClass::DeltaConflictSetColumn);
    assert!(m.contains("`s`, `a``b`"), "{m}");
    assert!(class(RowLevelOperation::Delete, &dup).is_none());

    let (c, _) = class(merge, &RowLevelProblem::DuplicateInsertColumn).unwrap();
    assert_eq!(c, ErrorClass::DeltaDuplicateColumnsOnInsert);
}
