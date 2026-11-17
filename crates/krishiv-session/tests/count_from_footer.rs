// SPDX-License-Identifier: Apache-2.0
//! A query that needs no column of a Parquet file reads no column: `COUNT(*)`
//! counts the rows the footers declare instead of decoding a column (plan B,
//! ClickBench q01). Other formats still read one column.
#![expect(clippy::unwrap_used, reason = "test code")]

use arrow_array::{Array, Int64Array, RecordBatch};
use futures::executor::block_on;
use krishiv_session::{ExplainMode, Session};

fn write_parquet(s: &Session, path: &str, sql: &str) {
    let df = block_on(s.sql(sql)).unwrap();
    block_on(df.write().parquet(path)).unwrap();
}

fn one_long(batches: &[RecordBatch]) -> i64 {
    let rows: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            let c = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            (0..c.len()).map(|i| c.value(i)).collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(rows.len(), 1, "{batches:?}");
    rows[0]
}

/// Whether the plan's (one) scan reads no column: its line ends `[]`.
fn scans_no_column(plan: &str) -> bool {
    plan.lines()
        .filter(|l| l.contains("Scan "))
        .all(|l| l.trim_end().ends_with("[]"))
}

fn count(s: &Session, sql: &str) -> (i64, String) {
    let df = block_on(s.sql(sql)).unwrap();
    let plan = block_on(df.explain(ExplainMode::Simple)).unwrap();
    (one_long(&block_on(df.collect()).unwrap()), plan)
}

#[test]
fn count_star_over_parquet_reads_no_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t").display().to_string();
    let s = Session::new().unwrap();
    write_parquet(
        &s,
        &path,
        "SELECT id, CAST(id AS STRING) AS s FROM range(0, 1000, 1, 3)",
    );
    let (n, plan) = count(&s, &format!("SELECT COUNT(*) FROM parquet.`{path}`"));
    assert_eq!(n, 1000);
    assert!(scans_no_column(&plan), "{plan}");
    // A literal count and an empty projection with a limit read no column either.
    let (n, plan) = count(&s, &format!("SELECT COUNT(1) FROM parquet.`{path}`"));
    assert_eq!(n, 1000);
    assert!(scans_no_column(&plan), "{plan}");
    let df = block_on(s.sql(&format!("SELECT 1 AS one FROM parquet.`{path}` LIMIT 3"))).unwrap();
    let plan = block_on(df.explain(ExplainMode::Simple)).unwrap();
    assert!(scans_no_column(&plan), "{plan}");
    let rows: usize = block_on(df.collect())
        .unwrap()
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, 3);
}

#[test]
fn a_filter_still_reads_its_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t").display().to_string();
    let s = Session::new().unwrap();
    write_parquet(&s, &path, "SELECT id FROM range(100)");
    let (n, plan) = count(
        &s,
        &format!("SELECT COUNT(*) FROM parquet.`{path}` WHERE id >= 90"),
    );
    assert_eq!(n, 10);
    assert!(!scans_no_column(&plan), "{plan}");
}

#[test]
fn count_star_over_a_partitioned_directory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("p").display().to_string();
    let s = Session::new().unwrap();
    let df = block_on(s.sql("SELECT id, id % 4 AS k FROM range(200)")).unwrap();
    block_on(df.write().partition_by(["k"]).parquet(&path)).unwrap();
    let (n, _) = count(&s, &format!("SELECT COUNT(*) FROM parquet.`{path}`"));
    assert_eq!(n, 200);
    let (n, _) = count(
        &s,
        &format!("SELECT COUNT(*) FROM parquet.`{path}` WHERE k = 1"),
    );
    assert_eq!(n, 50);
}

#[test]
fn other_formats_count_as_before() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c").display().to_string();
    let s = Session::new().unwrap();
    let df = block_on(s.sql("SELECT id FROM range(42)")).unwrap();
    block_on(df.write().csv(&path)).unwrap();
    let (n, plan) = count(&s, &format!("SELECT COUNT(*) FROM csv.`{path}`"));
    assert_eq!(n, 42);
    assert!(!scans_no_column(&plan), "{plan}");
}

#[test]
fn count_star_over_a_temporary_view_and_a_table_reads_no_column() {
    // The bench harness reads ClickBench `hits` through a temporary view;
    // catalog tables over Parquet go through the same file provider.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v").display().to_string();
    let s = Session::new().unwrap();
    write_parquet(
        &s,
        &path,
        "SELECT id, CAST(id AS STRING) AS s FROM range(0, 500, 1, 2)",
    );
    block_on(s.sql(&format!(
        "CREATE TEMPORARY VIEW hits AS SELECT * FROM parquet.`{path}`"
    )))
    .unwrap();
    // A SQL view keeps a projection above the scan, and pruning keeps one
    // entry of a projection, so this form still reads a column (follow-up).
    let (n, _) = count(&s, "SELECT COUNT(*) FROM hits");
    assert_eq!(n, 500);
    let df = s.read().parquet([path.as_str()]);
    block_on(df.create_or_replace_temp_view("hits2")).unwrap();
    let (n, plan) = count(&s, "SELECT COUNT(*) FROM hits2");
    assert_eq!(n, 500);
    assert!(scans_no_column(&plan), "{plan}");
    let src = block_on(s.sql("SELECT id FROM range(77)")).unwrap();
    block_on(src.write().save_as_table("t77")).unwrap();
    let (n, plan) = count(&s, "SELECT COUNT(*) FROM t77");
    assert_eq!(n, 77);
    assert!(scans_no_column(&plan), "{plan}");
}
