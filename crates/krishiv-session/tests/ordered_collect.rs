// SPDX-License-Identifier: Apache-2.0
//! Without `ORDER BY`, a query's rows come back partition by partition (a
//! `UNION ALL`'s children in order), as Spark returns them, however many
//! drivers read the partitions and however they race.
#![cfg(test)]

use std::sync::Arc;

use arrow_array::cast::AsArray as _;
use arrow_array::types::Int64Type;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_exec::{ExecutorConfig, TokioExecutor};
use krishiv_session::Session;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

fn session(threads: usize) -> Session {
    let executor = TokioExecutor::new(&ExecutorConfig {
        compute_threads: threads,
        io_threads: 1,
        blocking_threads: threads,
        io_blocking_threads: 2,
    })
    .unwrap();
    Session::builder().executor(executor).build().unwrap()
}

/// The first column's values in the order the query returned them.
fn collect_in_order(session: &Session, sql: &str) -> Vec<i64> {
    let df = block_on(session.sql(sql)).unwrap();
    let batches = block_on(df.collect()).unwrap();
    batches
        .iter()
        .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
        .collect()
}

/// Runs `sql` 25 times on each of four threads of one session.
fn repeat_under_load(session: &Session, sql: &str, want: &[i64]) {
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let session = session.clone();
            scope.spawn(move || {
                for _ in 0..25 {
                    assert_eq!(collect_in_order(&session, sql), want, "{sql}");
                }
            });
        }
    });
}

fn ids(range: std::ops::Range<i64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from_iter_values(range))]).unwrap()
}

#[test]
fn union_of_ranges_returns_child_by_child() {
    let want: Vec<i64> = (0..128).collect();
    for threads in [2, 8] {
        let s = session(threads);
        repeat_under_load(
            &s,
            "SELECT * FROM range(0, 64, 1, 8) UNION ALL SELECT * FROM range(64, 128, 1, 8)",
            &want,
        );
    }
}

/// A memory table of five morsels (128K rows each) read by several
/// drivers, through a filter and a projection.
#[test]
fn a_large_memory_table_returns_morsel_by_morsel() {
    let rows: i64 = 600_000;
    let batches: Vec<RecordBatch> = (0..rows)
        .step_by(8192)
        .map(|start| ids(start..(start + 8192).min(rows)))
        .collect();
    let want: Vec<i64> = (0..rows).filter(|i| i % 7 != 0).map(|i| i + 1).collect();
    for threads in [2, 8] {
        let s = session(threads);
        s.register_batches("t", batches[0].schema(), batches.clone())
            .unwrap();
        repeat_under_load(&s, "SELECT id + 1 FROM t WHERE id % 7 != 0", &want);
    }
}

/// Several Parquet files, each its own split: every run returns the rows
/// in the order a single driver reads the splits.
#[test]
fn a_multi_split_parquet_scan_returns_split_by_split() {
    let dir = tempfile::tempdir().unwrap();
    for f in 0..8_i64 {
        let path = dir.path().join(format!("part-{f}.parquet"));
        let file = std::fs::File::create(path).unwrap();
        let batch = ids(f * 10_000..(f + 1) * 10_000);
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(2_500))
            .build();
        let mut w = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }
    let path = dir.path().to_str().unwrap().to_owned();
    let sql = format!("SELECT id FROM parquet.`{path}` WHERE id % 3 != 1");
    let one = session(1);
    let want = collect_in_order(&one, &sql);
    assert_eq!(want.len(), (0..80_000).filter(|i| i % 3 != 1).count());
    for threads in [2, 8] {
        let s = session(threads);
        repeat_under_load(&s, &sql, &want);
    }
}
