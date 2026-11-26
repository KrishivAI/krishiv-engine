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

/// A join feeding a sort of every joined row spills under 8 MiB instead of
/// aborting the query: the only query holds about half of its limit, so
/// arbitration has room to spill (the sort merge falls back to a smaller
/// fan-in through `Reservation::grow_without_abort` instead of aborting).
#[test]
fn a_join_feeding_a_full_sort_spills_under_eight_mib() {
    let dir = tempfile::tempdir().unwrap();
    let s = session(&root(&dir), Some("8m"));
    let df =
        block_on(s.sql("SELECT f.k, f.v, d.k AS dk FROM f JOIN d ON f.k = d.k ORDER BY f.v, f.k"))
            .unwrap();
    let mut stream = block_on(df.execute_stream()).unwrap();
    let mut got = Vec::new();
    while let Some(batch) = block_on(stream.next()) {
        let batch = batch.unwrap();
        let (k, v, dk) = (
            batch.column(0).as_primitive::<Int64Type>(),
            batch.column(1).as_primitive::<Int64Type>(),
            batch.column(2).as_primitive::<Int64Type>(),
        );
        got.extend((0..batch.num_rows()).map(|i| (v.value(i), k.value(i), dk.value(i))));
    }
    // Every fact row joins its one key; `(v, k)` orders the whole row, so
    // the complete output, duplicates included, is determined.
    let mut want: Vec<(i64, i64, i64)> = (0..FACT_ROWS)
        .map(|i| (i % 1000, i % KEYS, i % KEYS))
        .collect();
    want.sort_unstable();
    assert_eq!(got.len(), want.len());
    if let Some(at) = got.iter().zip(&want).position(|(g, w)| g != w) {
        panic!("row {at}: got {:?}, want {:?}", got[at], want[at]);
    }
    drop(stream);
    wait_until_clean(&root(&dir));
}

/// RS-29 / PY-058: a spill directory that cannot be made (a read-only root, no writable home)
/// does not fail queries that never spill: `select 1` and a plain scan run.
#[test]
fn an_unusable_spill_directory_does_not_fail_a_query_that_never_spills() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("a_file");
    std::fs::write(&file, b"x").unwrap();
    // a directory below a regular file can never be created
    let dir = file.join("spill");
    let s = Session::builder()
        .config("krishiv.spill.dir", dir.to_str().unwrap())
        .build()
        .unwrap();
    let df = block_on(s.sql("SELECT 1 AS one, 'a' AS b")).unwrap();
    let got = block_on(df.collect()).unwrap();
    assert_eq!(got.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
    let df = block_on(s.sql("SELECT count(*) AS n FROM range(1000)")).unwrap();
    let got = block_on(df.collect()).unwrap();
    assert_eq!(got[0].column(0).as_primitive::<Int64Type>().value(0), 1000);
}

/// PY-078: a query that has to spill, with a spill directory that cannot be made, fails naming
/// `krishiv.spill.dir` and the OS error (`KRISHIV_SPILL_IO_ERROR`), not a bare memory error.
#[test]
fn a_spilling_query_with_an_unusable_spill_directory_names_the_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("a_file");
    std::fs::write(&file, b"x").unwrap();
    let dir = file.join("spill");
    let s = session(&dir, Some("4MB"));
    let err = block_on(async { s.sql(QUERY).await.unwrap().collect().await }).unwrap_err();
    assert_eq!(
        err.error_class(),
        krishiv_common::error::ErrorClass::KrishivSpillIoError,
        "{err}"
    );
    assert!(err.to_string().contains("krishiv.spill.dir"), "{err}");
    assert!(err.to_string().contains("a_file"), "{err}");
}

/// PY-078: a write names the directory too; a query that never spills is unaffected.
#[test]
fn a_spilling_write_with_an_unusable_spill_directory_names_the_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("a_file");
    std::fs::write(&file, b"x").unwrap();
    let s = session(&file.join("spill"), Some("4MB"));
    let out = tmp.path().join("o");
    let df = block_on(s.sql(QUERY)).unwrap();
    let err = block_on(df.write().format("parquet").save(out.to_str().unwrap())).unwrap_err();
    assert_eq!(
        err.error_class(),
        krishiv_common::error::ErrorClass::KrishivSpillIoError,
        "{err}"
    );
    assert!(err.to_string().contains("krishiv.spill.dir"), "{err}");
    assert!(err.to_string().contains("a_file"), "{err}");
    let df = block_on(s.sql("SELECT count(*) AS n FROM range(10)")).unwrap();
    let got = block_on(df.collect()).unwrap();
    assert_eq!(got[0].column(0).as_primitive::<Int64Type>().value(0), 10);
}

/// PY-078 (review): the error keeps the memory error as its cause, shows the OS error once, and
/// is what `system.queries` records for the failed query.
#[test]
fn the_spill_failure_is_the_recorded_error_and_keeps_the_memory_error_as_its_cause() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("a_file");
    std::fs::write(&file, b"x").unwrap();
    let s = session(&file.join("spill"), Some("4MB"));
    let err = block_on(async { s.sql(QUERY).await.unwrap().collect().await }).unwrap_err();
    assert_eq!(
        err.cause()
            .map(krishiv_common::error::KrishivError::error_class),
        Some(krishiv_common::error::ErrorClass::UnableToAcquireMemory),
        "{err}"
    );
    assert_eq!(err.to_string().matches("os error").count(), 1, "{err}");
    let rows = block_on(async {
        s.sql("SELECT error_class FROM system.queries WHERE error_class IS NOT NULL")
            .await
            .unwrap()
            .collect()
            .await
    })
    .unwrap();
    let classes: Vec<String> = rows
        .iter()
        .flat_map(|b| {
            let a = b.column(0).as_string::<i32>();
            (0..b.num_rows())
                .map(|i| a.value(i).to_owned())
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(
        classes.iter().any(|c| c == "KRISHIV_SPILL_IO_ERROR"),
        "{classes:?}"
    );
}

/// PY-078 (review): a spill set-up failure that is not an I/O failure keeps its own class.
#[test]
fn an_invalid_spill_setting_keeps_its_own_error_class() {
    let tmp = tempfile::tempdir().unwrap();
    let s = Session::builder()
        .config(
            "krishiv.spill.dir",
            tmp.path().join("spill").to_str().unwrap(),
        )
        .config("krishiv.spill.compression", "zstd")
        .config("krishiv.spill.zstd.level", "9999")
        .config("krishiv.memory.limit", "4MB")
        .build();
    let s = s.expect("a level the registry accepts");
    let err = block_on(async {
        s.sql("SELECT count(*), max(s) FROM (SELECT sha2(cast(id AS string), 256) s FROM range(2000000) ORDER BY s)")
            .await
            .unwrap()
            .collect()
            .await
    })
    .unwrap_err();
    assert_eq!(
        err.error_class(),
        krishiv_common::error::ErrorClass::KrishivInvalidConfigValue,
        "{err}"
    );
    assert!(
        err.to_string().contains("krishiv.spill.zstd.level"),
        "{err}"
    );
    // The cause is the memory refusal that asked for the spill. Which arbitration path refuses
    // first (the operator's own reservation or the query-wide abort) depends on task timing, so
    // either memory class is accepted; the class and message above are what the test pins.
    let cause = err
        .cause()
        .map(krishiv_common::error::KrishivError::error_class);
    assert!(
        matches!(
            cause,
            Some(
                krishiv_common::error::ErrorClass::UnableToAcquireMemory
                    | krishiv_common::error::ErrorClass::KrishivQueryAbortedOutOfMemory
            )
        ),
        "{cause:?}: {err}"
    );
}

/// PY-078 (review): the same frame run again once the directory is usable spills and succeeds.
#[test]
fn a_frame_run_again_after_the_spill_directory_became_usable_succeeds() {
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("a_file");
    std::fs::write(&file, b"x").unwrap();
    let s = session(&file.join("spill"), Some("6m"));
    let df = block_on(s.sql(QUERY)).unwrap();
    let err = block_on(df.collect()).unwrap_err();
    assert_eq!(
        err.error_class(),
        krishiv_common::error::ErrorClass::KrishivSpillIoError,
        "{err}"
    );
    std::fs::remove_file(&file).unwrap();
    let rows: usize = block_on(df.collect())
        .unwrap()
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, usize::try_from(KEYS).unwrap());
}

/// PY-078 (review): the spill directory opened for an earlier query, then made unwritable: the
/// next spilling query still names the directory (the per-query directory cannot be created).
#[cfg(unix)]
#[test]
fn a_spill_directory_that_became_unwritable_after_it_opened_is_named() {
    let tmp = tempfile::tempdir().unwrap();
    let r = root(&tmp);
    let s = session(&r, Some("6m"));
    let rows = |s: &Session| -> usize {
        let df = block_on(s.sql(QUERY)).unwrap();
        block_on(df.collect())
            .unwrap()
            .iter()
            .map(RecordBatch::num_rows)
            .sum()
    };
    assert_eq!(rows(&s), usize::try_from(KEYS).unwrap());
    wait_until_clean(&r);
    let dirs: Vec<_> = std::fs::read_dir(&r)
        .unwrap()
        .flatten()
        .map(|d| d.path())
        .filter(|p| p.is_dir())
        .collect();
    assert!(!dirs.is_empty());
    // A regular file where the session directory was: no child directory can be made below it,
    // for a privileged run as for an unprivileged one.
    let moved: Vec<_> = dirs
        .iter()
        .map(|d| {
            let to = d.with_extension("moved");
            std::fs::rename(d, &to).unwrap();
            std::fs::write(d, b"x").unwrap();
            to
        })
        .collect();
    // Under a loaded runner the tight budget can abort the query before it spills: retry then.
    let mut err = block_on(async { s.sql(QUERY).await.unwrap().collect().await });
    for _ in 0..10 {
        let oom = matches!(
            &err,
            Err(e) if e.error_class()
                == krishiv_common::error::ErrorClass::KrishivQueryAbortedOutOfMemory
        );
        if !oom {
            break;
        }
        err = block_on(async { s.sql(QUERY).await.unwrap().collect().await });
    }
    for (d, to) in dirs.iter().zip(&moved) {
        std::fs::remove_file(d).unwrap();
        std::fs::rename(to, d).unwrap();
    }
    let err = err.unwrap_err();
    assert_eq!(
        err.error_class(),
        krishiv_common::error::ErrorClass::KrishivSpillIoError,
        "{err}"
    );
    assert!(err.to_string().contains("krishiv.spill.dir"), "{err}");
}
