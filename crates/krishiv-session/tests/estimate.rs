// SPDX-License-Identifier: Apache-2.0
//! `DataFrame::estimate` (ADR 0041): what a statement would scan, before it runs.

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
    assert!(e.rows.unwrap() < 1000, "{e:?}");
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
