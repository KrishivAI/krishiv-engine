// SPDX-License-Identifier: Apache-2.0
//! `DataFrame::estimate` (ADR 0041): what a statement would scan, before it runs.
#![expect(clippy::unwrap_used, reason = "the helper sits outside #[test] fns")]

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;

fn session_with(rows: i64) -> Session {
    let s = Session::new().unwrap();
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from_iter_values(0..rows))],
    )
    .unwrap();
    s.register_batches("t", schema, vec![batch]).unwrap();
    s
}

#[test]
fn a_scan_is_estimated_from_rows_times_the_default_widths() {
    let s = session_with(1000);
    let e = block_on(block_on(s.sql("SELECT a FROM t")).unwrap().estimate()).unwrap();
    // 1000 rows of an 8-byte column plus 8 bytes of overhead each (the widths EXPLAIN COST uses).
    assert_eq!(e.scan_bytes, Some(1000 * 16));
    assert_eq!(e.rows, Some(1000));
    assert!(!e.plan.is_empty());
}

#[test]
fn a_filter_lowers_the_rows_but_not_the_scan() {
    let s = session_with(1000);
    let df = block_on(s.sql("SELECT a FROM t WHERE a < 10")).unwrap();
    let e = block_on(df.estimate()).unwrap();
    assert_eq!(e.scan_bytes, Some(1000 * 16));
    // The oracle is the optimizer's default selectivity of a range predicate on a column without min/max statistics
    // (one third, as Spark's `EXPLAIN COST` estimates it): 1000 rows become 333.
    assert_eq!(e.rows, Some(333), "{e:?}");
}

#[test]
fn a_statement_without_a_table_scans_nothing() {
    let s = Session::new().unwrap();
    let e = block_on(block_on(s.sql("SELECT 1 AS a")).unwrap().estimate()).unwrap();
    assert_eq!(e.scan_bytes, Some(0));
}

#[test]
fn a_join_adds_the_scans_of_both_sides() {
    let s = session_with(100);
    let df = block_on(s.sql("SELECT x.a FROM t x JOIN t y ON x.a = y.a")).unwrap();
    let e = block_on(df.estimate()).unwrap();
    assert_eq!(e.scan_bytes, Some(2 * 100 * 16));
}

#[test]
fn a_subquery_scan_counts() {
    let s = session_with(1000);
    let df = block_on(s.sql("SELECT (SELECT count(*) FROM t) AS n")).unwrap();
    let e = block_on(df.estimate()).unwrap();
    assert_eq!(e.scan_bytes, Some(1000 * 16));
    let df = block_on(s.sql("SELECT a FROM t WHERE a IN (SELECT a FROM t WHERE a < 5)")).unwrap();
    let e = block_on(df.estimate()).unwrap();
    assert_eq!(e.scan_bytes, Some(2 * 1000 * 16));
}

#[test]
fn a_table_without_statistics_has_an_unknown_estimate() {
    let s = Session::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let csv = dir.path().join("t.csv");
    std::fs::write(&csv, "a\n1\n2\n").unwrap();
    let df = s
        .read()
        .format("csv")
        .option("header", "true")
        .load([csv.to_str().unwrap()]);
    let e = block_on(df.estimate()).unwrap();
    assert_eq!(e.scan_bytes, None);
}

#[test]
fn an_insert_is_estimated_from_its_source_and_writes_nothing() {
    let s = session_with(1000);
    block_on(s.sql("CREATE TABLE w (a BIGINT)")).unwrap();
    let e = block_on(s.estimate_sql("INSERT INTO w SELECT a FROM t"))
        .unwrap()
        .unwrap();
    assert_eq!(e.scan_bytes, Some(1000 * 16));
    let rows = block_on(block_on(s.sql("SELECT count(*) FROM w")).unwrap().collect()).unwrap();
    let n = rows[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(n, 0, "estimating wrote nothing");
    // A statement with nothing to estimate answers `None`.
    assert!(block_on(s.estimate_sql("SET a.b = 1")).unwrap().is_none());
    assert!(
        block_on(s.estimate_sql("CREATE TABLE z (a INT)"))
            .unwrap()
            .is_none()
    );
}
