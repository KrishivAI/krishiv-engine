// SPDX-License-Identifier: Apache-2.0
//! The persisted query history: what a finished query leaves, the byte bound, the schema's compatibility
//! and that a failed write never fails a query.

use std::sync::Arc;

use futures::executor::block_on;
use krishiv_exec::profile::QueryProfile;

use super::{StoredProfile, read_history, trim, write_segment};
use crate::{EngineRuntime, Session};

fn row(i: i64, profile: String) -> StoredProfile {
    StoredProfile {
        query_id: format!("q{i}"),
        finished_at_ms: i,
        state: "finished".to_owned(),
        error_class: None,
        principal: "alice".to_owned(),
        sql: Some("SELECT 1".to_owned()),
        duration_ns: i,
        profile_version: 1,
        profile_minor: 5,
        profile,
    }
}

fn runtime(dir: &std::path::Path, max: &str) -> Arc<EngineRuntime> {
    EngineRuntime::builder()
        .config("krishiv.history.path", dir.to_str().unwrap())
        .config("krishiv.history.maxBytes", max)
        .build()
        .unwrap()
}

fn run(session: &Session, sql: &str) {
    let df = block_on(session.sql(sql)).unwrap();
    block_on(df.collect()).unwrap();
}

#[test]
fn a_finished_query_leaves_a_profile_that_reads_back() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(dir.path(), "64m");
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    run(
        &session,
        "SELECT id, id * 2 FROM range(0, 1000) WHERE id > 10",
    );
    rt.profile_history.as_ref().unwrap().flush();
    let stored = read_history(dir.path()).unwrap();
    assert_eq!(stored.len(), 1);
    let q = &stored[0];
    assert_eq!(q.state, "finished");
    assert!(q.sql.as_deref().unwrap().contains("range"));
    assert_eq!((q.profile_version, q.profile_minor), (1, 5));
    let profile = QueryProfile::from_json(&q.profile).unwrap();
    assert!(!profile.metrics.operators.is_empty());
    assert_eq!(profile.identity.unwrap().principal, q.principal);
    assert_eq!(rt.profile_history.as_ref().unwrap().failures(), 0);
}

#[test]
fn a_failed_query_is_kept_with_its_error_class() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(dir.path(), "64m");
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let df = block_on(session.sql("SELECT 1 / 0 FROM range(0, 3)")).unwrap();
    // Division by zero raises under ANSI mode; without it the query succeeds, so either is a valid run.
    let failed = block_on(df.collect()).is_err();
    rt.profile_history.as_ref().unwrap().flush();
    let stored = read_history(dir.path()).unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].state == "failed", failed);
    assert_eq!(stored[0].error_class.is_some(), failed);
}

#[test]
fn history_is_bounded_and_keeps_the_newest() {
    let dir = tempfile::tempdir().unwrap();
    let bound: u64 = 50 << 20;
    // Incompressible text, so the 100,000 rows are well over the bound on disk.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut i = 0;
    while i < 100_000 {
        let rows: Vec<StoredProfile> = (i..i + 256)
            .map(|n| row(n, (0..40).map(|_| format!("{:016x}", next())).collect()))
            .collect();
        let name = format!("history-{i:013}-{:06}.parquet", i / 256);
        write_segment(dir.path(), &name, &rows).unwrap();
        trim(dir.path(), bound).unwrap();
        i += 256;
    }
    let size: u64 = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().metadata().unwrap().len())
        .sum();
    assert!(size <= bound, "{size} bytes is over the bound");
    let stored = read_history(dir.path()).unwrap();
    assert!(stored.len() > 10_000 && stored.len() < 100_000);
    // The newest query is there, and what is kept is the newest run.
    let newest = stored.last().unwrap().finished_at_ms;
    assert!(newest >= 99_700);
    assert_eq!(
        stored.first().unwrap().finished_at_ms + i64::try_from(stored.len()).unwrap() - 1,
        newest
    );
}

#[test]
fn the_newest_segment_stays_even_over_the_bound() {
    let dir = tempfile::tempdir().unwrap();
    write_segment(
        dir.path(),
        "history-0000000000001-000001.parquet",
        &[row(1, "x".repeat(10_000))],
    )
    .unwrap();
    write_segment(
        dir.path(),
        "history-0000000000002-000002.parquet",
        &[row(2, "y".repeat(10_000))],
    )
    .unwrap();
    trim(dir.path(), 1).unwrap();
    let stored = read_history(dir.path()).unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].query_id, "q2");
}

#[test]
fn a_later_schema_with_an_extra_column_still_reads() {
    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};

    let dir = tempfile::tempdir().unwrap();
    // A v1 file as a later build writes it: the v1 columns plus an optional one.
    let mut fields: Vec<Field> = super::schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    fields.push(Field::new("added_later", DataType::Utf8, true));
    let schema = Arc::new(Schema::new(fields));
    let s = |v: &str| -> ArrayRef { Arc::new(StringArray::from(vec![v])) };
    let n = |v: i64| -> ArrayRef { Arc::new(Int64Array::from(vec![v])) };
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            s("q9"),
            n(9),
            s("finished"),
            Arc::new(StringArray::from(vec![None::<&str>])),
            s("bob"),
            s("SELECT 9"),
            n(9),
            n(1),
            n(7),
            s("{}"),
            s("new"),
        ],
    )
    .unwrap();
    let file =
        std::fs::File::create(dir.path().join("history-0000000000009-000001.parquet")).unwrap();
    let mut w = parquet::arrow::ArrowWriter::try_new(file, schema, None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
    let stored = read_history(dir.path()).unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(
        (stored[0].principal.as_str(), stored[0].profile_minor),
        ("bob", 7)
    );
}

#[test]
fn a_checked_in_v1_file_reads() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::copy(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/history-v1.parquet"),
        dir.path().join("history-0000000000001-000001.parquet"),
    )
    .unwrap();
    let stored = read_history(dir.path()).unwrap();
    assert_eq!(stored.len(), 2);
    assert_eq!(stored[0].query_id, "q1");
    assert_eq!(stored[1].error_class.as_deref(), Some("DIVIDE_BY_ZERO"));
}

#[test]
fn a_failed_write_is_counted_and_the_query_still_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history");
    let rt = runtime(&path, "64m");
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    // The directory vanishes after the store opened it: every segment write fails.
    std::fs::remove_dir_all(&path).unwrap();
    run(&session, "SELECT count(*) FROM range(0, 100)");
    rt.profile_history.as_ref().unwrap().flush();
    assert_eq!(rt.profile_history.as_ref().unwrap().failures(), 1);
    let snapshot = session.metrics();
    let gauge = snapshot
        .gauges
        .iter()
        .find(|g| g.name == "krishiv.history.write_failures")
        .expect("the failure gauge is registered when a history path is set");
    assert_eq!(gauge.value.to_bits(), 1.0_f64.to_bits());
}

#[test]
fn nothing_is_stored_without_a_path() {
    let rt = EngineRuntime::builder().build().unwrap();
    assert!(rt.profile_history.is_none());
    let session = Session::builder().runtime(rt).build().unwrap();
    run(&session, "SELECT 1");
}

/// Writes `tests/data/history-v1.parquet`, the checked-in v1 file; run once with `--ignored` when v1 is
/// defined, never to follow a later schema (the point of the fixture is that it stays as v1 wrote it).
#[test]
#[ignore = "regenerates the checked-in fixture"]
fn write_the_v1_fixture() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    std::fs::create_dir_all(&dir).unwrap();
    let mut failed = row(2, "{}".to_owned());
    failed.state = "failed".to_owned();
    failed.error_class = Some("DIVIDE_BY_ZERO".to_owned());
    write_segment(
        &dir,
        "history-v1.parquet",
        &[row(1, "{}".to_owned()), failed],
    )
    .unwrap();
}
