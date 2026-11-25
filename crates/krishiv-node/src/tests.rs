// SPDX-License-Identifier: Apache-2.0
//! The addon's Rust side without Node: a session, a streamed query, cancel, close.

use std::collections::HashMap;

use arrow_array::RecordBatch;
use arrow_ipc::reader::StreamReader;
use krishiv_common::error::KrishivError;

use crate::{NativeQuery, NativeSession};

fn run<T>(f: impl std::future::Future<Output = T>) -> T {
    futures::executor::block_on(f)
}

fn open() -> NativeSession {
    NativeSession::build(HashMap::new()).unwrap()
}

/// Every chunk of `query` concatenated, read back as one IPC stream.
fn drain(query: &NativeQuery) -> Result<Vec<RecordBatch>, KrishivError> {
    let mut bytes = Vec::new();
    while let Some(chunk) = run(query.step())? {
        bytes.extend(chunk);
    }
    Ok(read(&bytes))
}

fn read(bytes: &[u8]) -> Vec<RecordBatch> {
    StreamReader::try_new(bytes, None)
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

#[test]
fn a_query_streams_its_rows_as_ipc() {
    let session = open();
    let query = NativeQuery::new(
        session.session().unwrap(),
        "SELECT id, CAST(id AS STRING) AS s FROM range(5)".to_owned(),
    );
    let batches = drain(&query).unwrap();
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(rows, 5);
    assert_eq!(batches[0].schema().field(1).name(), "s");
    // After the end, `next()` keeps answering `null`.
    assert!(run(query.step()).unwrap().is_none());
}

#[test]
fn a_failure_carries_the_spark_error_class() {
    let session = open();
    let query = NativeQuery::new(
        session.session().unwrap(),
        "SELECT * FROM no_such_table".to_owned(),
    );
    let e = drain(&query).unwrap_err();
    assert_eq!(e.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
    let json: serde_json::Value = serde_json::from_str(&crate::error::to_json(&e)).unwrap();
    assert_eq!(json["sqlState"], "42P01");
    // A failed query is over.
    assert!(run(query.step()).unwrap().is_none());
}

#[test]
fn a_cancelled_query_rejects_with_operation_canceled() {
    let session = open();
    let query = NativeQuery::new(session.session().unwrap(), "SELECT 1".to_owned());
    query.stop();
    let e = run(query.step()).unwrap_err();
    assert_eq!(e.error_class().name(), "OPERATION_CANCELED");
    assert!(run(query.step()).unwrap().is_none());
}

#[test]
fn a_cancel_while_streaming_ends_the_stream() {
    let session = open();
    let query = NativeQuery::new(
        session.session().unwrap(),
        "SELECT id FROM range(10)".to_owned(),
    );
    // The schema chunk: the query is running.
    assert!(run(query.step()).unwrap().is_some());
    query.stop();
    let e = run(query.step()).unwrap_err();
    assert_eq!(e.error_class().name(), "OPERATION_CANCELED");
}

#[test]
fn configuration_reaches_the_session_and_close_is_final() {
    let session = NativeSession::build(HashMap::from([(
        "spark.sql.session.timeZone".to_owned(),
        "UTC".to_owned(),
    )]))
    .unwrap();
    let inner = session.session().unwrap();
    assert_eq!(
        inner.conf().get("spark.sql.session.timeZone").as_deref(),
        Some("UTC")
    );
    session.shut();
    session.shut();
    assert!(session.session().is_err());
}

#[test]
fn get_conf_reads_set_values_and_defaults() {
    let session = open();
    session
        .session()
        .unwrap()
        .conf()
        .set("spark.sql.session.timeZone", "Asia/Kolkata")
        .unwrap();
    assert_eq!(
        session
            .conf_value("spark.sql.session.timeZone")
            .unwrap()
            .as_deref(),
        Some("Asia/Kolkata")
    );
    assert_eq!(
        session
            .conf_value("spark.sql.ansi.enabled")
            .unwrap()
            .as_deref(),
        Some("true")
    );
    assert_eq!(session.conf_value("my.app.flag").unwrap(), None);
    session.shut();
    assert!(session.conf_value("spark.sql.ansi.enabled").is_err());
}
