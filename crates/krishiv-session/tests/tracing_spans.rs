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
        id: &tracing::span::Id,
        ctx: Context<'_, S>,
    ) {
        let mut fields = BTreeMap::new();
        attrs.record(&mut Fields(&mut fields));
        // The parent span's name (explicit, or the span current when this one is made).
        let parent = attrs
            .parent()
            .cloned()
            .or_else(|| ctx.current_span().id().cloned())
            .and_then(|pid| ctx.span(&pid).map(|p| p.name().to_owned()));
        if let Some(parent) = parent {
            fields.insert("parent".to_owned(), parent);
        }
        // Where the parent's fields are in `Spans`, to tell one test's child from another's.
        if let Some(pid) = attrs
            .parent()
            .cloned()
            .or_else(|| ctx.current_span().id().cloned())
            && let Some(Slot(slot)) = ctx
                .span(&pid)
                .and_then(|p| p.extensions().get::<Slot>().copied())
        {
            fields.insert("parent_slot".to_owned(), slot.to_string());
        }
        if let Ok(mut spans) = self.0.lock() {
            if let Some(span) = ctx.span(id) {
                span.extensions_mut().insert(Slot(spans.len()));
            }
            spans.push((attrs.metadata().name().to_owned(), fields));
        }
    }

    /// Fields recorded after the span was created (the query attributes are).
    fn on_record(
        &self,
        id: &tracing::span::Id,
        values: &tracing::span::Record<'_>,
        ctx: Context<'_, S>,
    ) {
        let Some(span) = ctx.span(id) else { return };
        let Some(Slot(slot)) = span.extensions().get::<Slot>().copied() else {
            return;
        };
        if let Ok(mut spans) = self.0.lock()
            && let Some((_, fields)) = spans.get_mut(slot)
        {
            values.record(&mut Fields(fields));
        }
    }

    fn on_close(&self, id: tracing::span::Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };
        let Some(Slot(slot)) = span.extensions().get::<Slot>().copied() else {
            return;
        };
        if let Ok(mut spans) = self.0.lock()
            && let Some((_, fields)) = spans.get_mut(slot)
        {
            fields.insert("closed".to_owned(), "true".to_owned());
        }
    }
}

/// Where the span's fields are in `Spans`.
#[derive(Clone, Copy)]
struct Slot(usize);

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
    spans_named("krishiv.query", table)
}

/// The spans called `name` whose `db.query.summary` mentions `table`.
fn spans_named(name: &str, table: &str) -> Vec<BTreeMap<String, String>> {
    spans()
        .lock()
        .map(|s| s.clone())
        .unwrap_or_default()
        .into_iter()
        .filter(|(n, f)| {
            n == name
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

#[test]
fn collect_with_metrics_runs_in_a_query_span() {
    spans();
    let s = session("orders_m");
    let df = block_on(s.sql("SELECT a FROM orders_m")).expect("plans");
    let (batches, _metrics) = block_on(df.collect_with_metrics()).expect("runs");
    assert_eq!(
        batches
            .iter()
            .map(arrow_array::RecordBatch::num_rows)
            .sum::<usize>(),
        3
    );
    assert_eq!(spans_of("orders_m").len(), 1, "one action, one span");
}

#[test]
fn a_command_that_runs_eagerly_has_a_span_around_its_execution() {
    spans();
    let s = session("orders_e");
    block_on(s.sql("CREATE TEMPORARY VIEW v_e AS SELECT a FROM orders_e")).expect("runs");
    let prepared = spans_named("krishiv.sql", "orders_e");
    let p = prepared
        .first()
        .expect("a krishiv.sql span for the statement");
    assert_eq!(p["db.operation.name"], "CREATE");
    assert!(p["db.query.text"].contains("v_e"), "{p:?}");
    // The planning and eager work is not a query span of its own.
    assert!(spans_of("orders_e").is_empty());
}

/// P2-T116: analysis, planning and execution are child spans of the query's span, so a backend
/// shows where a query's time went.
#[test]
fn analysis_planning_and_execution_are_child_spans() {
    spans();
    let s = session("orders_pipeline");
    let df = block_on(s.sql("SELECT a FROM orders_pipeline WHERE a > 1")).expect("plans");
    block_on(df.collect()).expect("runs");
    let all = spans().lock().map(|s| s.clone()).unwrap_or_default();
    for child in ["krishiv.analyze", "krishiv.plan", "krishiv.execute"] {
        assert!(
            all.iter()
                .any(|(n, f)| n == child
                    && f.get("parent").map(String::as_str) == Some("krishiv.query")),
            "no {child} span under krishiv.query in {all:#?}"
        );
    }
}

/// The execution span lasts while the result stream is alive, not only until the stream is handed
/// back.
#[test]
fn the_execution_span_stays_open_while_the_results_are_consumed() {
    use futures::StreamExt as _;
    spans();
    let s = session("orders_open");
    let df = block_on(s.sql("SELECT a FROM orders_open")).expect("plans");
    let mut stream = block_on(df.execute_stream()).expect("starts");
    // This test's execute span: the child of its own query span (other tests share the subscriber).
    let exec = || {
        let all = spans().lock().map(|s| s.clone()).unwrap_or_default();
        let query = all
            .iter()
            .position(|(n, f)| {
                n == "krishiv.query"
                    && f.get("db.query.summary")
                        .is_some_and(|v| v.contains("orders_open"))
            })
            .expect("the query span");
        let mine: Vec<_> = all
            .into_iter()
            .filter(|(n, f)| {
                n == "krishiv.execute"
                    && f.get("parent_slot").map(String::as_str) == Some(&query.to_string())
            })
            .collect();
        assert_eq!(mine.len(), 1, "{mine:?}");
        mine[0].1.contains_key("closed")
    };
    // Open after the first batch, closed only when the stream is dropped.
    let first = block_on(stream.next())
        .expect("a first batch")
        .expect("the first batch is rows, not an error");
    let a = first
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int32Array>()
        .expect("column a is Int32");
    assert_eq!(a.values().to_vec(), vec![1, 2, 3]);
    assert!(!exec(), "closed while the rows are still being read");
    while block_on(stream.next()).is_some() {}
    drop(stream);
    assert!(exec(), "still open after the stream was dropped");
}
