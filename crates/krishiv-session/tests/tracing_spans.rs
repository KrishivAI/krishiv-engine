// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.query` span of a statement (ADR 0035, P2-T116): the
//! database attribute set, the sanitized text and the ids.
#![cfg(test)]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use futures::executor::block_on;
use krishiv_session::Session;
use tracing::Subscriber;
use tracing::field::{Field, Visit};
use tracing_subscriber::layer::{Context, SubscriberExt as _};
use tracing_subscriber::registry::LookupSpan;

type Spans = Arc<Mutex<Vec<(String, BTreeMap<String, String>)>>>;

/// Records the name and fields of every span.
struct Capture(Spans);

struct Fields<'a>(&'a mut BTreeMap<String, String>);

impl Visit for Fields<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.0.insert(field.name().to_owned(), format!("{value:?}"));
    }
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().to_owned(), value.to_owned());
    }
}

impl<S: Subscriber + for<'a> LookupSpan<'a>> tracing_subscriber::Layer<S> for Capture {
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        _id: &tracing::span::Id,
        _ctx: Context<'_, S>,
    ) {
        let mut fields = BTreeMap::new();
        attrs.record(&mut Fields(&mut fields));
        if let Ok(mut spans) = self.0.lock() {
            spans.push((attrs.metadata().name().to_owned(), fields));
        }
    }
}

/// One subscriber for the test binary: actions run on the session's executor
/// threads, where a thread-local default is not set. Each test reads only the
/// spans of its own table.
fn spans() -> &'static Spans {
    static SPANS: std::sync::OnceLock<Spans> = std::sync::OnceLock::new();
    SPANS.get_or_init(|| {
        let spans = Spans::default();
        let subscriber = tracing_subscriber::registry().with(Capture(Arc::clone(&spans)));
        // Another test binary in the process cannot exist; ignore a second set.
        let _ = tracing::subscriber::set_global_default(subscriber);
        spans
    })
}

/// The `krishiv.query` spans whose `db.query.summary` mentions `table`.
fn spans_of(table: &str) -> Vec<BTreeMap<String, String>> {
    spans()
        .lock()
        .map(|s| s.clone())
        .unwrap_or_default()
        .into_iter()
        .filter(|(n, f)| {
            n == "krishiv.query"
                && f.get("db.query.summary")
                    .is_some_and(|summary| summary.contains(table))
        })
        .map(|(_, f)| f)
        .collect()
}

fn session(table: &str) -> Session {
    let s = Session::new().expect("session");
    let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "a",
        arrow_schema::DataType::Int32,
        false,
    )]));
    let batch = arrow_array::RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(arrow_array::Int32Array::from(vec![1, 2, 3]))],
    )
    .expect("batch");
    s.register_batches(table, schema, vec![batch])
        .expect("register");
    s
}

#[test]
fn a_query_span_carries_the_database_attributes() {
    spans();
    let s = session("orders_a");
    let df = block_on(s.sql("SELECT a FROM orders_a WHERE a > 1 AND 'secret-token' <> ''"))
        .expect("plans");
    block_on(df.collect()).expect("runs");
    let found = spans_of("orders_a");
    let f = found.first().expect("a krishiv.query span");
    assert_eq!(f["db.system.name"], "krishiv");
    assert_eq!(f["db.operation.name"], "SELECT");
    assert_eq!(f["db.query.summary"], "SELECT orders_a");
    assert_eq!(
        f["db.query.text"],
        "SELECT a FROM orders_a WHERE a > ? AND ? <> ?"
    );
    assert!(f["krishiv.query_id"].starts_with(f["krishiv.session_id"].as_str()));
    for fields in &found {
        for v in fields.values() {
            assert!(!v.contains("secret-token"), "{v}");
        }
    }
}

#[test]
fn statements_get_distinct_query_ids_in_one_session() {
    spans();
    let s = session("orders_b");
    for _ in 0..2 {
        let df = block_on(s.sql("SELECT a FROM orders_b")).expect("plans");
        block_on(df.collect()).expect("runs");
    }
    let found = spans_of("orders_b");
    assert_eq!(found.len(), 2, "{found:?}");
    assert_ne!(found[0]["krishiv.query_id"], found[1]["krishiv.query_id"]);
    assert_eq!(
        found[0]["krishiv.session_id"],
        found[1]["krishiv.session_id"]
    );
}

#[test]
fn a_frame_without_a_statement_has_no_query_text() {
    spans();
    let s = session("orders_c");
    block_on(s.table("orders_c").collect()).expect("runs");
    let plain = spans()
        .lock()
        .map(|s| s.clone())
        .unwrap_or_default()
        .into_iter()
        .filter(|(n, f)| n == "krishiv.query" && !f.contains_key("db.query.text"))
        .count();
    assert!(plain >= 1);
}
