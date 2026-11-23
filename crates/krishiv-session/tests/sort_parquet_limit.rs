// SPDX-License-Identifier: Apache-2.0
//! Acceptance SQL-33: a sort feeding a Parquet write under a memory limit far
//! below its input finishes (the acceptance run on the 0.3 wheel neither
//! finished in 600 s under a 1 GB limit nor stayed inside its container
//! without one), spills, writes every row once, and leaves each file in the
//! sort's order. CI size: 400,000 rows of about 60 bytes (about 24 MB) under a
//! 16 MB limit, and under 6 MB, where the Parquet writer itself runs short of
//! memory and must end its file instead of aborting the query.
#![cfg(test)]

use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::executor::block_on;
use krishiv_session::Session;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// Rows of `t`; `k` is a permutation of `0..ROWS`.
const ROWS: i64 = 400_000;

/// A write that has not finished by then counts as the acceptance hang.
const DEADLINE: Duration = Duration::from_secs(300);

fn session(spill: &std::path::Path, limit: &str) -> Session {
    let s = Session::builder()
        .config("krishiv.metrics.enabled", "true")
        .config("krishiv.spill.dir", spill.to_str().unwrap())
        .config("krishiv.memory.limit", limit)
        .build()
        .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    let rows: Vec<i64> = (0..ROWS).collect();
    let batches = rows
        .chunks(8192)
        .map(|c| {
            // 7,919 is prime and does not divide ROWS: `k` visits every value once.
            let k: Vec<i64> = c.iter().map(|i| i * 7_919 % ROWS).collect();
            let text: Vec<String> = k.iter().map(|k| text_of(*k)).collect();
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(k)),
                    Arc::new(StringArray::from(text)),
                ],
            )
            .unwrap()
        })
        .collect();
    s.register_batches("t", schema, batches).unwrap();
    s
}

/// The `s` value written for key `k`.
fn text_of(k: i64) -> String {
    format!("row {k:>12} of the SQL-33 sort-to-parquet case")
}

/// The `k` column of every Parquet file under `dir`, one list per file. Each
/// row's `s` is checked against its `k`, so a write that misaligns or drops
/// the second column fails here.
fn files(dir: &std::path::Path) -> Vec<Vec<i64>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "parquet") {
            continue;
        }
        let file = std::fs::File::open(&path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let mut keys = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            let k = batch.column_by_name("k").unwrap();
            let s = batch.column_by_name("s").unwrap();
            assert_eq!(k.null_count(), 0, "a null key in {}", path.display());
            assert_eq!(s.null_count(), 0, "a null text in {}", path.display());
            let k = k.as_primitive::<Int64Type>().values();
            let texts: Vec<&str> = match s.data_type() {
                DataType::Utf8View => s.as_string_view().iter().flatten().collect(),
                DataType::LargeUtf8 => s.as_string::<i64>().iter().flatten().collect(),
                _ => s.as_string::<i32>().iter().flatten().collect(),
            };
            assert_eq!(texts.len(), k.len());
            for (k, text) in k.iter().zip(texts) {
                assert_eq!(text, text_of(*k), "row of key {k} in {}", path.display());
            }
            keys.extend(k.iter());
        }
        out.push(keys);
    }
    out
}

#[test]
fn a_sort_written_to_parquet_under_a_small_limit_finishes_and_spills() {
    // One file at 16m: the 6m case's extra files come from the writer's roll,
    // not from several write tasks.
    assert_eq!(
        sorted_write_under("16m"),
        1,
        "the write split without a roll"
    );
}

/// At 6 MB the writer's buffered row group cannot grow: it ends the file and
/// opens the next, so the output is several sorted files, not an abort.
#[test]
fn a_sort_written_to_parquet_under_6m_rolls_files_instead_of_aborting() {
    let files = sorted_write_under("6m");
    assert!(
        files > 1,
        "the writer did not roll under 6 MB ({files} file)"
    );
}

/// Runs the sorted write under `limit`, checks it spilled and wrote every row
/// once in sorted files, and returns the number of files.
fn sorted_write_under(limit: &str) -> usize {
    let dir = tempfile::tempdir().unwrap();
    let spill = dir.path().join("spill");
    let out = dir.path().join("out");
    let s = session(&spill, limit);
    let path = out.to_str().unwrap().to_owned();
    let (done, wait) = mpsc::channel();
    let writer = std::thread::spawn(move || {
        let result = block_on(async {
            s.sql("SELECT k, s FROM t ORDER BY k")
                .await?
                .write()
                .parquet(&path)
                .await
        });
        let _ = done.send(());
        (s, result)
    });
    // A hang fails here rather than at the test runner's timeout; a panic of
    // the writer thread (the sender dropped unsent) shows through `join`.
    if let Err(mpsc::RecvTimeoutError::Timeout) = wait.recv_timeout(DEADLINE) {
        panic!("the sorted write did not finish in {DEADLINE:?}");
    }
    let (s, result) = writer.join().unwrap();
    result.unwrap();

    let snap = s.metrics();
    let spilled = snap
        .counter("krishiv.query.spill.files", &[])
        .unwrap_or(0.0);
    assert!(spilled > 0.0, "the sort did not spill under the limit");

    let files = files(&out);
    assert!(!files.is_empty());
    for keys in &files {
        assert!(keys.is_sorted(), "a file is not in the sort's order");
    }
    let count = files.len();
    let mut all: Vec<i64> = files.into_iter().flatten().collect();
    all.sort_unstable();
    assert_eq!(all, (0..ROWS).collect::<Vec<_>>(), "every row once");
    count
}
