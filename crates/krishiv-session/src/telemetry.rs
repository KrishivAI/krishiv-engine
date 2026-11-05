// SPDX-License-Identifier: Apache-2.0
//! The query span (ADR 0035 section 3, P2-T116): one `krishiv.query` span
//! per action of a frame that `Session::sql` built, with OpenTelemetry's
//! database attributes and the `krishiv.*` ones.
//!
//! The span carries `db.system.name`, `db.operation.name`,
//! `db.query.summary` and the sanitized `db.query.text` (every literal
//! replaced by `?`, comments dropped; [`krishiv_sql::sanitize_query_text`]),
//! computed over the text with its secrets already redacted (`redact_sql`),
//! plus `krishiv.session_id` (the session's `system.queries` id) and `krishiv.query_id`. Ids are minted here
//! until the session context (T136) brings principal, tags and the caller's
//! trace context. Without a subscriber `tracing` makes the span a no-op and
//! the attributes are not built, so an embedded engine pays a disabled-callsite
//! check and the copy of the statement text. The span stays open while the
//! result stream is read (`in_span`), so its duration is the query's.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use tracing::Span;

use crate::session::RecordBatchStream;

/// What a span says about one statement. The attributes are built when the
/// span is, and only if a subscriber wants it: with none installed a statement
/// costs the copy of its text and nothing else.
#[derive(Debug, Clone)]
pub(crate) struct QueryTrace {
    seq: u64,
    sql: String,
}

impl QueryTrace {
    /// The trace of `sql`, the `seq`-th statement of its session.
    pub(crate) fn of(seq: u64, sql: &str) -> Self {
        Self {
            seq,
            sql: sql.to_owned(),
        }
    }

    /// The attributes of the statement: the operation, the summary and the
    /// sanitized text, all computed over the text with its secrets redacted
    /// first (`krishiv_sql::redact_sql`: credential URLs, passwords, keys).
    fn attributes(&self) -> (String, String, String) {
        let redacted = krishiv_sql::redact_sql(&self.sql);
        let sql = redacted.as_str();
        let summary = krishiv_sql::query_summary(sql);
        let operation = summary
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_owned();
        (operation, summary, krishiv_sql::sanitize_query_text(sql))
    }

    /// The span of one action of the statement.
    pub(crate) fn span(&self, session_id: &str) -> Span {
        let span = tracing::info_span!(
            "krishiv.query",
            db.system.name = "krishiv",
            db.operation.name = tracing::field::Empty,
            db.query.summary = tracing::field::Empty,
            db.query.text = tracing::field::Empty,
            krishiv.session_id = %session_id,
            krishiv.query_id = tracing::field::Empty,
        );
        self.fill(&span, session_id);
        span
    }

    /// The span of `Session::sql` itself: parse, analysis and, for a command
    /// that runs eagerly (`INSERT ... SELECT`, DDL), its execution. A query's
    /// rows are produced later, inside the `krishiv.query` span of its action.
    pub(crate) fn prepare_span(&self, session_id: &str) -> Span {
        let span = tracing::info_span!(
            "krishiv.sql",
            db.system.name = "krishiv",
            db.operation.name = tracing::field::Empty,
            db.query.summary = tracing::field::Empty,
            db.query.text = tracing::field::Empty,
            krishiv.session_id = %session_id,
            krishiv.query_id = tracing::field::Empty,
        );
        self.fill(&span, session_id);
        span
    }

    /// Records the attributes, if a subscriber wants the span.
    fn fill(&self, span: &Span, session_id: &str) {
        if span.is_disabled() {
            return;
        }
        let (operation, summary, text) = self.attributes();
        span.record("db.operation.name", operation.as_str());
        span.record("db.query.summary", summary.as_str());
        span.record("db.query.text", text.as_str());
        span.record(
            "krishiv.query_id",
            format!("{session_id}-{}", self.seq).as_str(),
        );
    }
}

/// The span of an action of a frame without a statement (a DataFrame
/// transformation chain): the session and the engine, no query text.
pub(crate) fn plain_span(session_id: &str) -> Span {
    tracing::info_span!(
        "krishiv.query",
        db.system.name = "krishiv",
        krishiv.session_id = %session_id,
    )
}

/// `stream` kept inside `span` until it is dropped, and polled inside it: the
/// span lasts as long as the result is being produced and read, not only the
/// planning. A disabled span adds nothing.
pub(crate) fn in_span(stream: RecordBatchStream, span: Span) -> RecordBatchStream {
    if span.is_disabled() {
        return stream;
    }
    Box::pin(SpanStream {
        inner: stream,
        span,
    })
}

struct SpanStream {
    inner: RecordBatchStream,
    span: Span,
}

impl Stream for SpanStream {
    type Item = <RecordBatchStream as Stream>::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        let _entered = this.span.enter();
        this.inner.as_mut().poll_next(cx)
    }
}

/// An instance id for the OTLP exporter's resource: 16 hex digits, unique
/// within the process and unpredictable across processes (the std hasher's
/// random keys). Sessions use their `system.queries` id instead.
#[cfg(any(test, feature = "telemetry-otlp"))]
pub(crate) fn new_session_id() -> String {
    use std::hash::{BuildHasher, Hasher, RandomState};
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let mut h = RandomState::new().build_hasher();
    h.write_u64(NEXT.fetch_add(1, Ordering::Relaxed));
    format!("{:016x}", h.finish())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;

    #[test]
    fn a_trace_has_no_literals_and_no_secrets() {
        let t = QueryTrace::of(3, "SELECT * FROM orders WHERE name = 'bob' AND n > 7");
        let (operation, summary, text) = t.attributes();
        assert_eq!(operation, "SELECT");
        assert_eq!(summary, "SELECT orders");
        assert_eq!(text, "SELECT * FROM orders WHERE name = ? AND n > ?");
        for sql in [
            "SELECT * FROM parquet.`s3a://AKIA:hunter2@bucket/k`",
            "CREATE TABLE t USING csv OPTIONS (password = hunter2) AS SELECT 1",
            "CREATE TABLE t USING csv OPTIONS (password 'hunter2') AS SELECT 1",
        ] {
            let (_, summary, text) = QueryTrace::of(0, sql).attributes();
            assert!(
                !summary.contains("hunter2") && !text.contains("hunter2"),
                "{sql}: {summary} / {text}"
            );
        }
    }

    #[test]
    fn the_span_lasts_until_the_result_stream_is_dropped() {
        use futures::StreamExt as _;
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        use tracing_subscriber::layer::{Context, SubscriberExt as _};
        use tracing_subscriber::registry::LookupSpan;

        struct Closed(Arc<AtomicBool>);
        impl<S: tracing::Subscriber + for<'a> LookupSpan<'a>> tracing_subscriber::Layer<S> for Closed {
            fn on_close(&self, _id: tracing::span::Id, _ctx: Context<'_, S>) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let closed = Arc::new(AtomicBool::new(false));
        let subscriber = tracing_subscriber::registry().with(Closed(Arc::clone(&closed)));
        tracing::subscriber::with_default(subscriber, || {
            let span = plain_span("s");
            let stream: RecordBatchStream = futures::stream::empty().boxed();
            let stream = in_span(stream, span.clone());
            drop(span);
            assert!(
                !closed.load(Ordering::SeqCst),
                "the stream holds the span open"
            );
            drop(stream);
            assert!(
                closed.load(Ordering::SeqCst),
                "the span ends with the stream"
            );
        });
    }

    #[test]
    fn session_ids_differ() {
        assert_ne!(new_session_id(), new_session_id());
        assert_eq!(new_session_id().len(), 16);
    }
}
