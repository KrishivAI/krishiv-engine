// SPDX-License-Identifier: Apache-2.0
//! Queries run through a session spill under a memory limit: the result is
//! right, and the spill files are gone when the query ends, is dropped
//! mid-stream, or the session is dropped.
#![cfg(test)]

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::StreamExt;
use futures::executor::block_on;
use krishiv_session::Session;

const FACT_ROWS: i64 = 300_000;
const KEYS: i64 = 60_000;

const QUERY: &str = "SELECT f.k, sum(f.v) AS s, count(*) AS n \
                     FROM f JOIN d ON f.k = d.k \
                     GROUP BY f.k ORDER BY s DESC, f.k";

/// A spill root under a temp directory; the session creates it (mode 0700).
fn root(tmp: &tempfile::TempDir) -> std::path::PathBuf {
    tmp.path().join("spill")
}

fn session(dir: &Path, limit: Option<&str>) -> Session {
    let mut b = Session::builder().config("krishiv.spill.dir", dir.to_str().unwrap());
    if let Some(limit) = limit {
        b = b.config("krishiv.memory.limit", limit);
    }
    let s = b.build().unwrap();
    let fact = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let rows: Vec<i64> = (0..FACT_ROWS).collect();
    let batches: Vec<RecordBatch> = rows
        .chunks(8192)
        .map(|c| {
            RecordBatch::try_new(
                Arc::clone(&fact),
                vec![
                    Arc::new(Int64Array::from_iter_values(c.iter().map(|i| i % KEYS))),
                    Arc::new(Int64Array::from_iter_values(c.iter().map(|i| i % 1000))),
                ],
            )
            .unwrap()
        })
        .collect();
    s.register_batches("f", fact, batches).unwrap();
    let dim = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
    let keys = RecordBatch::try_new(
        Arc::clone(&dim),
        vec![Arc::new(Int64Array::from_iter_values(0..KEYS))],
    )
    .unwrap();
    s.register_batches("d", dim, vec![keys]).unwrap();
    s
}

fn expected() -> Vec<(i64, i64, i64)> {
    let mut by_key = vec![(0i64, 0i64); usize::try_from(KEYS).unwrap()];
    for i in 0..FACT_ROWS {
        let e = &mut by_key[usize::try_from(i % KEYS).unwrap()];
        e.0 += i % 1000;
        e.1 += 1;
    }
    let mut out: Vec<(i64, i64, i64)> = by_key
        .into_iter()
        .enumerate()
        .map(|(k, (s, n))| (i64::try_from(k).unwrap(), s, n))
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

fn rows(batches: &[RecordBatch]) -> Vec<(i64, i64, i64)> {
    batches
        .iter()
        .flat_map(|b| {
            let k = b.column(0).as_primitive::<Int64Type>();
            let s = b.column(1).as_primitive::<Int64Type>();
            let n = b.column(2).as_primitive::<Int64Type>();
            (0..b.num_rows())
                .map(|i| (k.value(i), s.value(i), n.value(i)))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Every file or directory below the per-query directories of `root`.
fn leftovers(root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    // A missing root is a failure, not a clean directory.
    let sessions = std::fs::read_dir(root).expect("the spill root exists");
    for s in sessions.flatten().filter(|s| s.path().is_dir()) {
        for q in std::fs::read_dir(s.path()).unwrap().flatten() {
            out.push(q.path().display().to_string());
        }
    }
    out
}

fn wait_until_clean(root: &Path) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !leftovers(root).is_empty() {
        assert!(
            Instant::now() < deadline,
            "spill files left: {:?}",
            leftovers(root)
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_join_aggregate_sort_spills_under_a_small_memory_limit() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(&root(&dir), Some("6m"));
    let df = block_on(s.sql(QUERY)).unwrap();
    let got = rows(&block_on(df.collect()).unwrap());
    assert_eq!(got, expected());
    wait_until_clean(&root(&dir));
    drop(df);
    drop(s);
    assert!(
        std::fs::read_dir(root(&dir)).unwrap().next().is_none(),
        "the session directory outlived the session"
    );
}

#[test]
fn dropping_a_query_mid_stream_removes_its_spill_files() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(&root(&dir), Some("6m"));
    let df = block_on(s.sql(QUERY)).unwrap();
    {
        let mut stream = block_on(df.execute_stream()).unwrap();
        let first = block_on(stream.next()).unwrap().unwrap();
        assert!(first.num_rows() > 0);
    }
    wait_until_clean(&root(&dir));
}

#[test]
fn an_unlimited_session_gives_the_same_result() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(&root(&dir), None);
    let df = block_on(s.sql(QUERY)).unwrap();
    assert_eq!(rows(&block_on(df.collect()).unwrap()), expected());
    wait_until_clean(&root(&dir));
}

#[test]
fn df_write_under_a_limit_spills_too() {
    let dir = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let s = session(&root(&dir), Some("6m"));
    let df = block_on(s.sql(QUERY)).unwrap();
    block_on(
        df.write()
            .format("parquet")
            .mode("overwrite")
            .save(out.path().join("o").to_str().unwrap()),
    )
    .unwrap();
    let back = s.read().parquet([out.path().join("o").to_str().unwrap()]);
    assert_eq!(
        block_on(back.count()).unwrap(),
        u64::try_from(KEYS).unwrap()
    );
    wait_until_clean(&root(&dir));
}

#[test]
#[should_panic(expected = "the spill root exists")]
fn a_missing_spill_root_fails_the_cleanup_check() {
    let dir = tempfile::tempdir().unwrap();
    let _ = leftovers(&root(&dir));
}

/// `EXPLAIN ANALYZE` of a spilling query shows the spill files each
/// operator wrote and the query's memory limit and peak.
#[test]
fn explain_analyze_shows_spill_files_and_memory() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(&root(&dir), Some("6m"));
    let df = block_on(s.sql(QUERY)).unwrap();
    let (got, profile) = block_on(df.collect_with_profile()).unwrap();
    assert_eq!(rows(&got), expected());
    let m = &profile.metrics;
    let files: u64 = m.operators.values().map(|e| e.metrics.spill_files).sum();
    let written: u64 = m
        .operators
        .values()
        .map(|e| e.metrics.spill_write_bytes)
        .sum();
    assert!(files > 0 && written > 0, "{files} files, {written} bytes");
    assert!(m.memory.peak_reserved > 0);
    let text = profile.to_text(krishiv_session::TextOptions::default());
    assert!(text.contains("spill files "), "{text}");
    assert!(text.contains("memory: peak reserved "), "{text}");
    let back = krishiv_session::QueryProfile::from_json(&profile.to_json()).unwrap();
    assert_eq!(back, profile);
    wait_until_clean(&root(&dir));
}
