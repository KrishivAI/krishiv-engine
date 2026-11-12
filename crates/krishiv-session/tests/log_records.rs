// SPDX-License-Identifier: Apache-2.0
//! Structured logs (ADR 0035, P2-T116): a record written while a query runs carries the query's
//! ids, the trace the caller set and the principal; no record holds the statement's literals; and
//! the rate limit keeps a callsite from flooding the output.
#![cfg(test)]

use std::io::Write;
use std::sync::{Arc, Mutex, OnceLock};

use futures::executor::block_on;
use krishiv_session::Session;
use krishiv_session::logging::json_layer;
use serde_json::Value;
use tracing_subscriber::layer::SubscriberExt as _;

#[derive(Clone, Default)]
struct Buf(Arc<Mutex<Vec<u8>>>);

impl Write for Buf {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend_from_slice(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Buf {
    fn records(&self) -> Vec<Value> {
        let text = String::from_utf8(self.0.lock().unwrap().clone()).unwrap();
        text.lines()
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{e}: {l}")))
            .collect()
    }
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }
}

/// One subscriber for the test binary: actions run on the session's executor threads, where a
/// thread-local default is not set.
fn output() -> &'static Buf {
    static OUT: OnceLock<Buf> = OnceLock::new();
    OUT.get_or_init(|| {
        let buf = Buf::default();
        let writer = buf.clone();
        let subscriber =
            tracing_subscriber::registry().with(json_layer(1000, move || writer.clone()));
        let _ = tracing::subscriber::set_global_default(subscriber);
        buf
    })
}

/// The records written inside a `krishiv.query` span of the trace `trace`.
fn of_trace(trace: &str) -> Vec<Value> {
    output()
        .records()
        .into_iter()
        .filter(|r| {
            r["spans"].as_array().is_some_and(|spans| {
                spans
                    .iter()
                    .any(|s| s["krishiv.trace_id"].as_str() == Some(trace))
            })
        })
        .collect()
}

fn session_with_trace(trace: &str) -> Session {
    let s = Session::new().unwrap();
    s.set_trace_context(&format!("00-{trace}-00f067aa0ba902b7-01"))
        .unwrap();
    s
}

#[test]
fn a_record_carries_the_ids_of_its_query_and_no_literal() {
    let _ = output();
    let trace = "4bf92f3577b34da6a3ce929d0e0e4701";
    let s = session_with_trace(trace);
    let rows = block_on(async {
        s.sql("SELECT 'topsecret-value' AS x, 7 AS n")
            .await?
            .collect()
            .await
    })
    .unwrap();
    assert_eq!(
        rows.iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        1
    );

    let records = of_trace(trace);
    let finished = records
        .iter()
        .find(|r| r["fields"]["message"] == "query finished")
        .unwrap_or_else(|| panic!("no 'query finished' record in {records:#?}"));
    assert_eq!(finished["target"], "krishiv::query");
    assert_eq!(finished["fields"]["rows"], 1);
    let span = finished["spans"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["name"] == "krishiv.query")
        .unwrap();
    assert_eq!(span["krishiv.trace_id"], trace);
    assert!(
        span["krishiv.query_id"]
            .as_str()
            .is_some_and(|q| !q.is_empty())
    );
    assert!(
        span["krishiv.session_id"]
            .as_str()
            // The session's `system.queries` id, a UUID.
            .is_some_and(|q| q.len() == 36 && q.matches('-').count() == 4)
    );
    assert!(
        span["krishiv.principal"]
            .as_str()
            .is_some_and(|p| !p.is_empty())
    );
    // The sanitized text is on the span; the literal is on no record.
    assert_eq!(span["db.query.text"], "SELECT ? AS x , ? AS n");
    assert!(
        !output().text().contains("topsecret-value"),
        "a record holds the statement's literal"
    );
}

#[test]
fn a_failing_query_logs_its_error_class() {
    let _ = output();
    let trace = "4bf92f3577b34da6a3ce929d0e0e4702";
    let s = session_with_trace(trace);
    let r = block_on(async {
        s.sql("SELECT CAST(x AS INT) FROM (VALUES ('not a number')) AS t(x)")
            .await?
            .collect()
            .await
    });
    assert!(r.is_err());
    let records = of_trace(trace);
    assert!(
        records
            .iter()
            .all(|r| r["fields"]["message"] != "query finished"),
        "{records:#?}"
    );
    let failed = records
        .iter()
        .find(|r| r["fields"]["message"] == "query failed")
        .unwrap_or_else(|| panic!("no 'query failed' record in {records:#?}"));
    assert_eq!(failed["level"], "WARN");
    assert!(failed["fields"]["error_class"].as_str().is_some());
}

#[test]
fn a_callsite_is_limited_per_second() {
    let buf = Buf::default();
    let writer = buf.clone();
    let subscriber = tracing_subscriber::registry().with(json_layer(2, move || writer.clone()));
    tracing::subscriber::with_default(subscriber, || {
        for i in 0..10 {
            tracing::info!(i, "same callsite");
        }
        tracing::info!("another callsite");
    });
    let records = buf.records();
    assert_eq!(
        records
            .iter()
            .filter(|r| r["fields"]["message"] == "same callsite")
            .count(),
        2
    );
    assert_eq!(
        records
            .iter()
            .filter(|r| r["fields"]["message"] == "another callsite")
            .count(),
        1
    );
}
