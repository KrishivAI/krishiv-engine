// SPDX-License-Identifier: Apache-2.0
//! The UI backend over a real engine runtime: the system tables through `SessionSource`.
#![cfg(feature = "session")]
#![expect(
    clippy::unwrap_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::sync::Arc;

use krishiv_server::internal::{SystemTable, UiSource};
use krishiv_server::session_source::{SessionSource, UI_PRINCIPAL};
use krishiv_session::{EngineRuntime, Session};
use serde_json::Value;

fn runtime() -> Arc<EngineRuntime> {
    EngineRuntime::builder().build().unwrap()
}

#[tokio::test]
async fn a_users_query_shows_up_in_the_queries_table_but_the_uis_own_polling_does_not() {
    let rt = runtime();
    let user = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    user.sql("SELECT 41 + 1 AS answer")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let source = SessionSource::new(rt).unwrap();
    let first = source.rows(SystemTable::Queries).await.unwrap();
    let again = source.rows(SystemTable::Queries).await.unwrap();
    assert!(
        first.iter().any(|r| r["query_text"]
            .as_str()
            .is_some_and(|t| t.contains("41 + 1"))),
        "{first:?}"
    );
    assert!(
        first
            .iter()
            .chain(&again)
            .all(|r| r["principal"] != Value::String(UI_PRINCIPAL.to_owned())),
        "the UI's own queries are hidden"
    );
    assert_eq!(
        first.len(),
        again.len(),
        "polling does not add rows to the list"
    );
    let row = first
        .iter()
        .find(|r| {
            r["query_text"]
                .as_str()
                .is_some_and(|t| t.contains("41 + 1"))
        })
        .unwrap();
    assert!(
        row["submitted_at"].is_i64(),
        "timestamps are microseconds: {row:?}"
    );
    assert_eq!(row["state"], "finished");
}

#[tokio::test]
async fn memory_and_caches_come_through_with_numbers_and_flags() {
    let source = SessionSource::new(runtime()).unwrap();
    let memory = source.rows(SystemTable::Memory).await.unwrap();
    assert!(!memory.is_empty());
    assert!(memory[0]["reserved_bytes"].is_i64());
    assert!(memory[0]["spillable"].is_boolean());
    let caches = source.rows(SystemTable::Caches).await.unwrap();
    assert!(caches.iter().all(|r| r["hits"].is_i64()));
}

#[tokio::test]
async fn a_populated_operator_row_converts_every_allowlisted_column() {
    let rt = runtime();
    let user = Session::builder().runtime(Arc::clone(&rt)).build().unwrap();
    user.sql("SELECT id % 3 AS k, sum(id) FROM range(1000) GROUP BY id % 3")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let source = SessionSource::new(rt).unwrap();
    let operators = source.rows(SystemTable::Operators).await.unwrap();
    assert!(!operators.is_empty(), "the query recorded no operators");
    let sink = operators.iter().find(|r| r["name"] == "Result").unwrap();
    assert!(
        sink["query_id"].as_str().is_some_and(|q| q.len() == 36),
        "{sink:?}"
    );
    assert_eq!(sink["role"], "sink");
    for column in [
        "operator_id",
        "drivers",
        "rows_in",
        "compute_ns",
        "max_driver_compute_ns",
        "blocked_io_ns",
        "blocked_memory_ns",
        "blocked_sink_ns",
        "peak_reserved_bytes",
        "spilled_bytes",
    ] {
        // A column the engine has not filled yet is null; every filled one converts to a number.
        assert!(
            sink[column].is_i64() || sink[column].is_null(),
            "{column}: {sink:?}"
        );
    }
    assert_eq!(
        sink["rows_in"], 3,
        "the three groups reached the sink: {sink:?}"
    );
}

#[tokio::test]
async fn cancelling_an_unknown_query_is_false_and_a_malformed_id_is_an_error() {
    let source = SessionSource::new(runtime()).unwrap();
    assert!(
        !source
            .cancel("00112233-4455-6677-8899-aabbccddeeff")
            .await
            .unwrap()
    );
    assert!(
        !source.cancel("not-a-query-id").await.unwrap(),
        "an id that is not an id names no running query"
    );
}
