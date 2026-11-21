// SPDX-License-Identifier: Apache-2.0
//! The persisted query history: what a finished query leaves, the byte bound, the schema's compatibility
//! and that a failed write never fails a query.

use std::sync::Arc;

use futures::executor::block_on;
use krishiv_exec::profile::QueryProfile;

use super::{StoredProfile, kept_hints, read_history, trim, write_segment};
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

/// The stored entries, which must not be empty: a masking test that loops over none passes by losing every profile.
fn nonempty_history(dir: &std::path::Path) -> Vec<StoredProfile> {
    let stored = read_history(dir).unwrap();
    assert!(!stored.is_empty(), "the history holds no entry to check");
    stored
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

/// The runtime reads one query's profile back by id (what the console's query page asks for): each stored query
/// gets its own document, an id the store does not hold gets none, and a runtime with no history has none to give.
#[test]
fn the_runtime_reads_one_stored_profile_by_query_id() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(dir.path(), "64m");
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    run(&session, "SELECT id FROM range(0, 10) WHERE id > 3");
    run(
        &session,
        "SELECT id % 2 AS k, count(*) AS n FROM range(0, 100) GROUP BY id % 2",
    );
    rt.profile_history.as_ref().unwrap().flush();
    let stored = read_history(dir.path()).unwrap();
    assert_eq!(stored.len(), 2);
    assert_ne!(stored[0].query_id, stored[1].query_id);
    for want in &stored {
        let got = rt.stored_profile(&want.query_id).unwrap();
        assert_eq!(got.as_ref(), Some(want));
        // The document carries the plan the page draws.
        let doc: serde_json::Value = serde_json::from_str(&want.profile).unwrap();
        assert!(
            doc["plan"].as_array().is_some_and(|p| !p.is_empty()),
            "{doc}"
        );
    }
    assert_eq!(rt.stored_profile("no-such-query").unwrap(), None);
    // Queries stored later land in later segments; the earlier ones are still found.
    run(&session, "SELECT 1");
    rt.profile_history.as_ref().unwrap().flush();
    assert_eq!(
        rt.stored_profile(&stored[0].query_id).unwrap().as_ref(),
        Some(&stored[0])
    );
    let plain = EngineRuntime::builder().build().unwrap();
    assert_eq!(plain.stored_profile(&stored[0].query_id).unwrap(), None);
}

/// A query id stored more than once (a client may reuse an operation id) reads back as its newest profile:
/// the newest segment that names it decides, and within a segment the latest finish. A segment that does not
/// read is passed over.
#[test]
fn a_query_id_stored_twice_reads_back_as_the_newest() {
    let dir = tempfile::tempdir().unwrap();
    let again = |finished_at_ms: i64, profile: &str| StoredProfile {
        query_id: "q1".to_owned(),
        ..row(finished_at_ms, profile.to_owned())
    };
    write_segment(
        dir.path(),
        "history-0000000000001-000001.parquet",
        &[row(1, "first".to_owned()), row(2, "other".to_owned())],
    )
    .unwrap();
    write_segment(
        dir.path(),
        "history-0000000000005-000002.parquet",
        &[
            again(7, "third"),
            again(5, "second"),
            row(3, "x".to_owned()),
        ],
    )
    .unwrap();
    std::fs::write(
        dir.path().join("history-0000000000009-000003.parquet"),
        b"not parquet",
    )
    .unwrap();
    let read = |id: &str| super::read_stored(dir.path(), id).unwrap();
    assert_eq!(read("q1"), Some(again(7, "third")));
    // Only the older segment names this one.
    assert_eq!(read("q2"), Some(row(2, "other".to_owned())));
    assert_eq!(read("q9"), None);
    // No store there at all.
    assert!(super::read_stored(&dir.path().join("missing"), "q1").is_err());
}

/// The row is found by its id and finish time, then read alone: a query in any of a segment's row groups
/// reads back as its own row, and of two rows with one id and one finish time the later one is returned.
#[test]
fn a_lookup_reads_the_profile_of_the_one_row_it_returns() {
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    let dir = tempfile::tempdir().unwrap();
    // 3,000 rows in row groups of 1,000; the wanted id is in the last group, between others.
    let rows: Vec<StoredProfile> = (0..3000).map(|i| row(i, format!("doc-{i}"))).collect();
    write_segment(dir.path(), "history-0000000000001-000001.parquet", &rows).unwrap();
    let path = dir.path().join("history-0000000000001-000001.parquet");
    let whole = super::read_segment(&path).unwrap();
    let small = dir.path().join("history-0000000000002-000002.parquet");
    let batch = {
        let reader = super::open_segment(&path)
            .unwrap()
            .with_batch_size(3000)
            .build()
            .unwrap();
        reader.into_iter().next().unwrap().unwrap()
    };
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(1000))
        .build();
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(&small).unwrap(),
        batch.schema(),
        Some(props),
    )
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(
        super::open_segment(&small)
            .unwrap()
            .metadata()
            .num_row_groups(),
        3
    );
    for i in [0_usize, 999, 1000, 2500, 2999] {
        assert_eq!(
            super::read_stored(dir.path(), &format!("q{i}"))
                .unwrap()
                .as_ref(),
            Some(&whole[i]),
            "row {i}"
        );
    }
    assert_eq!(super::read_stored(dir.path(), "q3000").unwrap(), None);
    let twin = |profile: &str| StoredProfile {
        query_id: "twin".to_owned(),
        ..row(9, profile.to_owned())
    };
    write_segment(
        dir.path(),
        "history-0000000000003-000003.parquet",
        &[twin("first"), row(1, "between".to_owned()), twin("last")],
    )
    .unwrap();
    assert_eq!(
        super::read_stored(dir.path(), "twin").unwrap(),
        Some(twin("last"))
    );
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
    assert_eq!((q.profile_version, q.profile_minor), (1, 7));
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
    // Division by zero raises under ANSI mode (Spark's DIVIDE_BY_ZERO), so the run must fail.
    session
        .conf()
        .set("spark.sql.ansi.enabled", "true")
        .unwrap();
    let df = block_on(session.sql("SELECT 1 / 0 FROM range(0, 3)")).unwrap();
    assert!(
        block_on(df.collect()).is_err(),
        "the query must fail under ANSI"
    );
    rt.profile_history.as_ref().unwrap().flush();
    let stored = read_history(dir.path()).unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].state, "failed");
    assert_eq!(stored[0].error_class.as_deref(), Some("DIVIDE_BY_ZERO"));
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
    let listed = super::list_segments(dir.path()).unwrap();
    assert_eq!(listed.len(), 3);
    std::fs::remove_file(dir.path().join("history-0000000000001-a-000001.parquet")).unwrap();
    super::trim_listed(&listed, 1).unwrap();
    let left: Vec<String> = super::list_segments(dir.path())
        .unwrap()
        .into_iter()
        .map(|(p, _)| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(left, ["history-0000000000003-a-000003.parquet"]);
}

#[test]
fn a_segment_that_is_gone_or_not_a_regular_file_has_no_size() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("history-0000000000001-a-000001.parquet");
    write_segment(
        dir.path(),
        "history-0000000000001-a-000001.parquet",
        &[row(1, "x".repeat(5_000))],
    )
    .unwrap();
    assert_eq!(
        super::segment_len(&file).unwrap(),
        Some(std::fs::metadata(&file).unwrap().len())
    );
    // Removed after the directory was listed: gone, not an error.
    std::fs::remove_file(&file).unwrap();
    assert_eq!(super::segment_len(&file).unwrap(), None);
    // A directory is not a segment, and neither is a link (never followed, so a loop cannot fail the listing).
    assert_eq!(super::segment_len(dir.path()).unwrap(), None);
    #[cfg(unix)]
    {
        let link = dir.path().join("history-0000000000002-a-000002.parquet");
        std::os::unix::fs::symlink(&link, &link).unwrap();
        assert_eq!(super::segment_len(&link).unwrap(), None);
        assert!(super::list_segments(dir.path()).unwrap().is_empty());
    }
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
    let stored = nonempty_history(dir.path());
    assert!(
        !stored[0].profile.contains(SECRET_TEXT),
        "a literal-derived output name was stored"
    );
}

#[test]
fn a_view_with_a_literal_output_name_is_masked_in_the_plan() {
    let dir = tempfile::tempdir().unwrap();
    let rt = EngineRuntime::builder()
        .config("krishiv.history.path", dir.path().to_str().unwrap())
        .build()
        .unwrap();
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    run(
        &session,
        &format!("CREATE TEMPORARY VIEW v AS SELECT '{SECRET_TEXT}' FROM range(0, 1)"),
    );
    run(&session, "SELECT * FROM v");
    rt.profile_history.as_ref().unwrap().flush();
    for e in nonempty_history(dir.path()) {
        assert!(
            !e.profile.contains(SECRET_TEXT),
            "a view's literal name was stored"
        );
    }
}

#[test]
fn concatenated_and_punctuated_literal_names_are_masked() {
    let dir = tempfile::tempdir().unwrap();
    let rt = EngineRuntime::builder()
        .config("krishiv.history.path", dir.path().to_str().unwrap())
        .build()
        .unwrap();
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    run(&session, "SELECT 'huntera' 'twob' FROM range(0, 5)");
    run(&session, "SELECT 'hunterc-secretd' FROM range(0, 5)");
    rt.profile_history.as_ref().unwrap().flush();
    for stored in nonempty_history(dir.path()) {
        for leak in ["huntera", "twob", "hunterc", "secretd"] {
            assert!(
                !stored.profile.contains(leak),
                "{leak} stored: {}",
                stored.profile
            );
        }
    }
}

#[test]
fn a_file_write_of_a_literal_holding_as_and_a_hash_is_masked() {
    let dir = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let rt = runtime(dir.path(), "64m");
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let df = block_on(session.sql("SELECT 'hunterx AS y#1hunterz' FROM range(0, 5)")).unwrap();
    block_on(df.write().parquet(out.path().join("t").to_str().unwrap())).unwrap();
    rt.profile_history.as_ref().unwrap().flush();
    let stored = nonempty_history(dir.path());
    for q in stored {
        assert!(!q.profile.contains("hunterx"), "{}", q.profile);
        assert!(!q.profile.contains("hunterz"), "{}", q.profile);
    }
}

#[test]
fn a_file_write_of_a_literal_projection_is_masked() {
    let dir = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let rt = runtime(dir.path(), "64m");
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let df = block_on(session.sql("SELECT 'huntere' FROM range(0, 5)")).unwrap();
    block_on(df.write().parquet(out.path().join("t").to_str().unwrap())).unwrap();
    rt.profile_history.as_ref().unwrap().flush();
    let stored = nonempty_history(dir.path());
    for q in stored {
        assert!(!q.profile.contains("huntere"), "{}", q.profile);
    }
}

#[test]
fn a_partitioned_write_of_a_literal_column_is_masked() {
    let dir = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let rt = runtime(dir.path(), "64m");
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let df = block_on(session.sql("SELECT 'huntery', id FROM range(0, 5)")).unwrap();
    block_on(
        df.write()
            .partition_by(["huntery"])
            .parquet(out.path().join("t").to_str().unwrap()),
    )
    .unwrap();
    rt.profile_history.as_ref().unwrap().flush();
    for q in nonempty_history(dir.path()) {
        assert!(!q.profile.contains("huntery"), "{}", q.profile);
    }
}

#[test]
fn a_view_over_a_literal_is_masked_in_unions_and_distinct() {
    let dir = tempfile::tempdir().unwrap();
    let rt = EngineRuntime::builder()
        .config("krishiv.history.path", dir.path().to_str().unwrap())
        .build()
        .unwrap();
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    run(
        &session,
        "CREATE TEMP VIEW hv AS SELECT 'hunterq' FROM range(0, 1)",
    );
    run(&session, "SELECT * FROM hv UNION ALL SELECT * FROM hv");
    run(&session, "SELECT DISTINCT * FROM hv");
    rt.profile_history.as_ref().unwrap().flush();
    let stored = nonempty_history(dir.path());
    for q in stored {
        assert!(!q.profile.contains("hunterq"), "{}", q.profile);
    }
}

#[test]
fn a_dataframe_write_of_literal_named_keys_is_masked() {
    use crate::functions::{count_star, lit};
    let dir = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let rt = runtime(dir.path(), "64m");
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let df = session
        .range(0, 5, 1, None)
        .select([lit("hunterz[")])
        .group_by(["hunterz["])
        .agg([count_star()]);
    block_on(df.write().parquet(out.path().join("t").to_str().unwrap())).unwrap();
    rt.profile_history.as_ref().unwrap().flush();
    let stored = nonempty_history(dir.path());
    for q in stored {
        assert!(!q.profile.contains("hunterz"), "{}", q.profile);
    }
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

#[test]
fn a_dropped_file_write_is_recorded_cancelled_or_finished_never_lost_or_doubled() {
    use futures::FutureExt;
    let dir = tempfile::tempdir().unwrap();
    let out = tempfile::tempdir().unwrap();
    let rt = runtime(dir.path(), "64m");
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let mut recorded_writes = 0;
    for polls in (0..40_u64).step_by(3) {
        let df = block_on(session.sql("SELECT id FROM range(0, 50000000)")).unwrap();
        let target = out.path().join(format!("t{polls}"));
        let mut write = Box::pin(df.write().parquet(target.to_str().unwrap()));
        // Poll `polls` times, a millisecond apart, then drop the future wherever it stands. 50 million rows take
        // longer than any poll count here, so a drop always catches the query in flight.
        for _ in 0..polls {
            if write.as_mut().now_or_never().is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        drop(write);
        recorded_writes += 1;
    }
    // A write dropped before its query was registered has nothing to record, so the count is not fixed; what must hold
    // is that no entry is doubled and none is lost once it exists. Wait until the writer has taken everything (the count
    // holds still for a second), not for a fixed time.
    let history = rt.profile_history.as_ref().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut last = (usize::MAX, std::time::Instant::now());
    let stored = loop {
        history.flush();
        let now = read_history(dir.path()).unwrap();
        if now.len() != last.0 {
            last = (now.len(), std::time::Instant::now());
        }
        if last.1.elapsed() >= std::time::Duration::from_secs(1)
            || std::time::Instant::now() > deadline
        {
            break now;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    };
    assert!(
        stored.len() <= recorded_writes,
        "{} entries for {recorded_writes} dropped writes",
        stored.len()
    );
    let mut ids: Vec<&str> = stored.iter().map(|q| q.query_id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), stored.len(), "an entry was doubled");
    assert!(
        stored.iter().any(|q| q.state == "cancelled"),
        "a dropped write's held entry was lost: {:?}",
        stored.iter().map(|q| q.state.clone()).collect::<Vec<_>>()
    );
    assert!(
        stored
            .iter()
            .all(|q| ["finished", "cancelled"].contains(&q.state.as_str()))
    );
    assert_eq!(rt.profile_history.as_ref().unwrap().failures(), 0);
}

#[test]
fn the_stored_profile_holds_the_tags_of_the_queries_own_row() {
    let dir = tempfile::tempdir().unwrap();
    let rt = runtime(dir.path(), "64m");
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let df = block_on(session.sql_with("SELECT 1", crate::QueryOptions::new().tags(["job-42"])))
        .unwrap();
    block_on(df.collect()).unwrap();
    rt.profile_history.as_ref().unwrap().flush();
    let stored = read_history(dir.path()).unwrap();
    assert!(
        stored.iter().any(|q| q.profile.contains("job-42")),
        "a request tag was not stored"
    );
}

/// The stored profile's hints are the ones `EXPLAIN ANALYZE` shows, under the default `mask`: the numbers in
/// a message are engine metrics, not data, so masking must not touch them.
#[test]
fn stored_hints_equal_the_explain_analyze_hints_under_masking() {
    let dir = tempfile::tempdir().unwrap();
    let executor = krishiv_exec::TokioExecutor::new(&krishiv_exec::ExecutorConfig {
        compute_threads: 4,
        io_threads: 1,
        blocking_threads: 4,
        io_blocking_threads: 2,
    })
    .unwrap();
    let rt = EngineRuntime::builder()
        .executor(executor)
        .config("krishiv.history.path", dir.path().to_str().unwrap())
        .config("krishiv.advisor.skew.minTime", "1ms")
        .config("krishiv.advisor.skew.minRows", "1")
        .build()
        .unwrap();
    let session = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    let df = block_on(session.sql(
        "SELECT id, sum(id) OVER (PARTITION BY CASE WHEN id % 10 = 0 THEN id ELSE 0 END ORDER BY id) \
         FROM range(0, 3000000)",
    ))
    .unwrap();
    let (_, profile) = block_on(df.collect_with_profile()).unwrap();
    assert!(!profile.hints.is_empty(), "the planted skew raised no hint");
    rt.profile_history.as_ref().unwrap().flush();
    let stored = nonempty_history(dir.path());
    let back = QueryProfile::from_json(&stored[0].profile).unwrap();
    // The JSON object keeps its evidence sorted by name; the code, node, operator and message are the contract.
    let key = |hints: &[krishiv_exec::profile::Hint]| {
        hints
            .iter()
            .map(|h| (h.code, h.node, h.operator, h.message.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(key(&back.hints), key(&profile.hints));
    assert!(
        back.hints
            .iter()
            .any(|h| h.message.chars().any(|c| c.is_ascii_digit())),
        "a hint with numbers is what masking could have broken"
    );
}

#[test]
fn only_a_finished_query_keeps_its_hints() {
    use krishiv_exec::profile::{Hint, HintCode};
    let hint = || Hint {
        code: HintCode::Skew,
        node: Some(1),
        operator: None,
        message: "skew".to_owned(),
        evidence: Vec::new(),
    };
    assert_eq!(kept_hints("finished", vec![hint()]).len(), 1);
    for state in ["failed", "cancelled", "denied"] {
        assert!(kept_hints(state, vec![hint()]).is_empty(), "{state}");
    }
}

#[test]
fn stored_keeps_hints_only_for_a_finished_entry() {
    use krishiv_exec::physical::{ExprEnv, PhysNode, PhysicalPlan};
    use krishiv_exec::planner::ExplainNode;
    use krishiv_exec::profile::{Hint, HintCode, ProfileIdentity};

    let entry = |state: &'static str| super::HistoryEntry {
        query_id: "q".to_owned(),
        finished_at_ms: 1,
        state,
        error_class: None,
        sql: None,
        detailed: false,
        duration_ns: 1,
        plan: PhysicalPlan::new(
            PhysNode::OneRow,
            Arc::new(arrow_schema::Schema::empty()),
            ExplainNode::leaf("OneRowRelation", ""),
            ExprEnv {
                ansi: true,
                legacy_statistical_aggregate: false,
                zone: None,
                zone_name: None,
                query_start_micros: 0,
                user: Arc::from("u"),
                map_key_dedup_last_win: false,
                legacy_java_charsets: false,
                legacy_coding_error_action: false,
                legacy_time_parser: false,
                binary_output_style: None,
            },
        ),
        metrics: krishiv_exec::QueryMetrics::default(),
        identity: ProfileIdentity::default(),
        trace: None,
        scans: Vec::new(),
        hints: vec![Hint {
            code: HintCode::Skew,
            node: Some(1),
            operator: None,
            message: "skew".to_owned(),
            evidence: Vec::new(),
        }],
    };
    for (state, kept) in [
        ("finished", 1),
        ("failed", 0),
        ("cancelled", 0),
        ("denied", 0),
    ] {
        for mask in [false, true] {
            let row = super::stored(entry(state), mask).unwrap();
            let back = QueryProfile::from_json(&row.profile).unwrap();
            assert_eq!(back.hints.len(), kept, "{state}, mask {mask}");
        }
    }
}
