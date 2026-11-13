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
    // The stored document is a profile this build still reads.
    let profile = QueryProfile::from_json(&stored[0].profile).unwrap();
    assert!(!profile.metrics.operators.is_empty());
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

/// Writes `tests/data/history-v1.parquet`, the checked-in v1 file, holding a real profile; run once with
/// `--ignored` when v1 is defined, never to follow a later schema (the point of the fixture is that it stays
/// as v1 wrote it).
#[test]
#[ignore = "regenerates the checked-in fixture"]
fn write_the_v1_fixture() {
    let scratch = tempfile::tempdir().unwrap();
    let rt = runtime(scratch.path(), "64m");
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    run(
        &session,
        "SELECT id % 3 AS k, count(*) FROM range(0, 500) GROUP BY 1",
    );
    rt.profile_history.as_ref().unwrap().flush();
    let profile = read_history(scratch.path()).unwrap().remove(0).profile;
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    std::fs::create_dir_all(&dir).unwrap();
    let mut failed = row(2, profile.clone());
    failed.state = "failed".to_owned();
    failed.error_class = Some("DIVIDE_BY_ZERO".to_owned());
    write_segment(&dir, "history-v1.parquet", &[row(1, profile), failed]).unwrap();
}

const SECRET_TEXT: &str = "hunter2-literal-canary";
const SECRET_NUMBER: &str = "424242424";

fn canary_run(literals: Option<&str>) -> Vec<StoredProfile> {
    let dir = tempfile::tempdir().unwrap();
    let mut b =
        EngineRuntime::builder().config("krishiv.history.path", dir.path().to_str().unwrap());
    if let Some(l) = literals {
        b = b.config("krishiv.history.literals", l);
    }
    let rt = b.build().unwrap();
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    run(
        &session,
        &format!(
            "SELECT id, '{SECRET_TEXT}' AS s FROM range(0, 50) WHERE id < {SECRET_NUMBER} AND id <> 7"
        ),
    );
    rt.profile_history.as_ref().unwrap().flush();
    read_history(dir.path()).unwrap()
}

#[test]
fn literals_are_masked_by_default_in_the_text_and_the_plan() {
    for stored in [canary_run(None), canary_run(Some("mask"))] {
        assert_eq!(stored.len(), 1);
        let q = &stored[0];
        for text in [q.sql.as_deref().unwrap(), q.profile.as_str()] {
            assert!(!text.contains(SECRET_TEXT), "a string literal was stored");
            assert!(!text.contains(SECRET_NUMBER), "a number literal was stored");
        }
        assert!(q.sql.as_deref().unwrap().contains('?'));
        // The plan is still there, and still parses.
        assert!(q.profile.contains("Filter"));
        QueryProfile::from_json(&q.profile).unwrap();
    }
}

#[test]
fn keep_stores_the_literals() {
    let stored = canary_run(Some("keep"));
    assert!(stored[0].sql.as_deref().unwrap().contains(SECRET_TEXT));
    assert!(stored[0].profile.contains(SECRET_NUMBER));
}

#[test]
fn a_profile_is_stored_only_after_its_result_stream_ends() {
    use futures::StreamExt;

    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(dir.path(), "64m");
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let df = block_on(session.sql("SELECT id FROM range(0, 100000)")).unwrap();
    let mut stream = block_on(df.execute_stream()).unwrap();
    let history = rt.profile_history.as_ref().unwrap();
    assert!(block_on(stream.next()).is_some());
    history.flush();
    assert!(
        read_history(dir.path()).unwrap().is_empty(),
        "stored while the stream was open"
    );
    while block_on(stream.next()).is_some() {}
    drop(stream);
    history.flush();
    assert_eq!(read_history(dir.path()).unwrap().len(), 1);
}

#[test]
fn two_engines_on_one_directory_keep_every_segment() {
    let dir = tempfile::tempdir().unwrap();
    let a = runtime(dir.path(), "64m");
    let b = runtime(dir.path(), "64m");
    let sa = Session::builder().runtime(Arc::clone(&a)).build().unwrap();
    let sb = Session::builder().runtime(Arc::clone(&b)).build().unwrap();
    for _ in 0..5 {
        run(&sa, "SELECT 1");
        a.profile_history.as_ref().unwrap().flush();
        run(&sb, "SELECT 2");
        b.profile_history.as_ref().unwrap().flush();
    }
    // Both writers have written a first segment (sequence 1) for the same moment, if the clock is coarse.
    assert_eq!(read_history(dir.path()).unwrap().len(), 10);
    assert_eq!(a.profile_history.as_ref().unwrap().failures(), 0);
    assert_eq!(b.profile_history.as_ref().unwrap().failures(), 0);
}

#[test]
fn a_file_of_a_later_schema_version_is_skipped() {
    use parquet::file::metadata::KeyValue;
    use parquet::file::properties::WriterProperties;

    let dir = tempfile::tempdir().unwrap();
    write_segment(
        dir.path(),
        "history-0000000000001-a-000001.parquet",
        &[row(1, "{}".to_owned())],
    )
    .unwrap();
    // The same columns, marked as a later major version.
    let src = dir.path().join("history-0000000000001-a-000001.parquet");
    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
        std::fs::File::open(&src).unwrap(),
    )
    .unwrap();
    let schema = Arc::clone(builder.schema());
    let batches: Vec<_> = builder.build().unwrap().map(Result::unwrap).collect();
    let props = WriterProperties::builder()
        .set_key_value_metadata(Some(vec![KeyValue::new(
            "krishiv.history.schema_version".to_owned(),
            "2".to_owned(),
        )]))
        .build();
    let file =
        std::fs::File::create(dir.path().join("history-0000000000002-a-000002.parquet")).unwrap();
    let mut w = parquet::arrow::ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    for b in &batches {
        w.write(b).unwrap();
    }
    w.close().unwrap();
    // Only the version 1 file reads.
    assert_eq!(read_history(dir.path()).unwrap().len(), 1);
}

#[test]
fn trim_ignores_a_segment_another_engine_removed() {
    let dir = tempfile::tempdir().unwrap();
    write_segment(
        dir.path(),
        "history-0000000000001-a-000001.parquet",
        &[row(1, "x".repeat(5_000))],
    )
    .unwrap();
    write_segment(
        dir.path(),
        "history-0000000000002-a-000002.parquet",
        &[row(2, "y".repeat(5_000))],
    )
    .unwrap();
    write_segment(
        dir.path(),
        "history-0000000000003-a-000003.parquet",
        &[row(3, "z".repeat(5_000))],
    )
    .unwrap();
    // The listing sees all three; the first is gone by the time trim removes it.
    std::fs::remove_file(dir.path().join("history-0000000000001-a-000001.parquet")).unwrap();
    trim(dir.path(), 1).unwrap();
    assert_eq!(read_history(dir.path()).unwrap().len(), 1);
}

#[test]
fn an_unaliased_literal_projection_is_masked_in_the_plan() {
    let dir = tempfile::tempdir().unwrap();
    let rt = EngineRuntime::builder()
        .config("krishiv.history.path", dir.path().to_str().unwrap())
        .build()
        .unwrap();
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    run(
        &session,
        &format!("SELECT '{SECRET_TEXT}' FROM range(0, 5)"),
    );
    rt.profile_history.as_ref().unwrap().flush();
    let stored = read_history(dir.path()).unwrap();
    assert!(
        !stored[0].profile.contains(SECRET_TEXT),
        "a literal-derived output name was stored"
    );
}

#[test]
fn mask_word_replaces_whole_words_only() {
    assert_eq!(
        super::mask_word("a, alias, a_b, a", "a"),
        "?, alias, a_b, ?"
    );
    assert_eq!(super::mask_word("x", "hunter2"), "x");
}

#[test]
fn a_failed_segment_write_leaves_no_temporary_file() {
    let dir = tempfile::tempdir().unwrap();
    // The final name is a directory, so the rename fails after the file was written.
    std::fs::create_dir(dir.path().join("history-v1.parquet")).unwrap();
    let profile = String::new();
    assert!(write_segment(dir.path(), "history-v1.parquet", &[row(1, profile)]).is_err());
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty());
}

#[test]
fn the_detailed_flag_follows_the_query() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(dir.path(), "64m");
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    run(&session, "SELECT count(*) FROM range(0, 100)");
    rt.profile_history.as_ref().unwrap().flush();
    let stored = read_history(dir.path()).unwrap();
    assert!(
        !QueryProfile::from_json(&stored[0].profile)
            .unwrap()
            .detailed
    );
}

#[test]
fn a_file_write_is_recorded_once_after_its_commit() {
    let dir = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let rt = runtime(dir.path(), "64m");
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let df = block_on(session.sql("SELECT id FROM range(0, 10)")).unwrap();
    block_on(df.write().parquet(out.path().join("t").to_str().unwrap())).unwrap();
    rt.profile_history.as_ref().unwrap().flush();
    let stored = read_history(dir.path()).unwrap();
    assert_eq!(stored.len(), 1, "the write's profile is lost or doubled");
    assert_eq!(stored[0].state, "finished");
}
