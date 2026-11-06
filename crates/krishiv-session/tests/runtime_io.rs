// SPDX-License-Identifier: Apache-2.0
//! I/O under one engine runtime (design P2-E1, unit R3): one scheduler for
//! every session, and a query's `system.queries.bytes_read` counts its own
//! reads, never another session's running at the same time.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::{Array, Int64Array, RecordBatch};
use futures::StreamExt;
use futures::executor::block_on;
use krishiv_session::{EngineRuntime, Session};

fn run(s: &Session, sql: &str) -> Vec<RecordBatch> {
    block_on(block_on(s.sql(sql)).unwrap().collect()).unwrap()
}

/// The `bytes_read` of the row whose text is `text`.
fn bytes_read(s: &Session, text: &str) -> i64 {
    let got = run(
        s,
        &format!(
            "SELECT bytes_read FROM system.queries WHERE query_text = '{}'",
            text.replace('\'', "''")
        ),
    );
    let c = got[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(c.len(), 1);
    c.value(0)
}

#[test]
fn a_querys_bytes_are_its_own() {
    let dir = tempfile::tempdir().unwrap();
    let big = dir.path().join("big");
    let small = dir.path().join("small");
    let rt = EngineRuntime::builder().build().unwrap();
    let a = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let b = a.new_session();
    let write = |rows: u64, path: &std::path::Path| {
        let df = block_on(a.sql(&format!("SELECT id, id * 7 AS v FROM range(0, {rows})"))).unwrap();
        block_on(df.write().format("parquet").save(path.to_str().unwrap())).unwrap();
    };
    write(1_000_000, &big);
    write(1_000, &small);
    let size = |p: &std::path::Path| -> u64 {
        std::fs::read_dir(p)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.metadata().unwrap().len())
            .sum()
    };
    let (big_bytes, small_bytes) = (size(&big), size(&small));
    assert!(big_bytes > 20 * small_bytes);
    // Session B scans the big files while session A reads the small ones.
    let big_sql = format!("SELECT * FROM parquet.`{}`", big.display());
    let mut scan = block_on(block_on(b.sql(&big_sql)).unwrap().execute_stream()).unwrap();
    assert!(block_on(scan.next()).unwrap().is_ok());
    let small_sql = format!("SELECT * FROM parquet.`{}`", small.display());
    assert_eq!(
        run(&a, &small_sql)
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>(),
        1_000
    );
    while let Some(batch) = block_on(scan.next()) {
        batch.unwrap();
    }
    let own = u64::try_from(bytes_read(&a, &small_sql)).unwrap();
    assert!(own > 0, "the read was counted");
    assert!(
        own <= 2 * small_bytes,
        "{own} bytes counted for a {small_bytes}-byte read while another session read {big_bytes}"
    );
    let theirs = u64::try_from(bytes_read(&b, &big_sql)).unwrap();
    assert!(theirs >= big_bytes / 2, "{theirs} of {big_bytes}");
}

/// A write query counts the bytes it wrote (its own I/O context).
#[test]
fn a_write_counts_its_bytes_written() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out");
    let s = Session::new().unwrap();
    let df = block_on(s.sql("SELECT id FROM range(0, 100000)")).unwrap();
    block_on(df.write().format("parquet").save(out.to_str().unwrap())).unwrap();
    let written: u64 = std::fs::read_dir(&out)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.metadata().unwrap().len())
        .sum();
    let got = run(
        &s,
        "SELECT bytes_written FROM system.queries WHERE query_text = 'SELECT id FROM range(0, 100000)'",
    );
    let c = got[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let counted = u64::try_from(c.value(0)).unwrap();
    assert!(
        counted > 0 && counted <= written + 4096,
        "{counted} of {written}"
    );
}
