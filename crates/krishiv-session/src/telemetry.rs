// SPDX-License-Identifier: Apache-2.0
//! The query span (ADR 0035 section 3, P2-T116): one `krishiv.query` span
//! per action of a frame that `Session::sql` built, with OpenTelemetry's
//! database attributes and the `krishiv.*` ones.
//!
//! The span carries `db.system.name`, `db.operation.name`,
//! `db.query.summary` and the sanitized `db.query.text` (every literal
//! replaced by `?`, comments dropped; [`krishiv_sql::sanitize_query_text`]),
//! computed over the text with its secrets already redacted (`redact_sql`),
//! plus `krishiv.session_id` (the session's `system.queries` id) and
//! `krishiv.query_id`, minted here until the session context (T136) brings
//! principal, tags and the caller's trace context. Without a subscriber `tracing` makes the span a no-op and
//! the attributes are not built, so an embedded engine pays a disabled-callsite
//! check and the copy of the statement text. The span stays open while the
//! result stream is read (`in_span`), so its duration is the query's.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use tracing::Span;

use crate::session::RecordBatchStream;

/// Who and what a span belongs to: the session, the trace the caller set with
/// `Session::set_trace_context`, and the principal. Every log record written
/// inside the span carries these through the span's fields (the JSON layer
/// lists them), so one query's records join on `krishiv.query_id` and
/// `krishiv.trace_id`.
#[derive(Debug, Clone)]
pub(crate) struct Who {
    pub(crate) session_id: String,
    pub(crate) trace_id: Option<String>,
    /// The caller's span the trace came with: the spans below are its children.
    #[cfg(feature = "telemetry-otlp")]
    pub(crate) remote: Option<crate::system::RemoteParent>,
    pub(crate) principal: String,
    pub(crate) tenant: Option<String>,
}

impl Who {
    /// The identity of `state`'s session at this moment (the trace context can change between
    /// statements).
    pub(crate) fn of(state: &crate::session::SessionState) -> Self {
        Self {
            session_id: state.session_id.clone(),
            trace_id: state.queries.trace_id(),
            #[cfg(feature = "telemetry-otlp")]
            remote: state.queries.trace(),
            principal: state.principal.name.clone(),
            tenant: state.principal.tenant.clone(),
        }
    }

    /// Records the fields on `span`, if a subscriber wants it.
    fn record(&self, span: &Span) {
        if span.is_disabled() {
            return;
        }
        span.record("krishiv.principal", self.principal.as_str());
        if let Some(t) = &self.tenant {
            span.record("krishiv.tenant", t.as_str());
        }
        if let Some(t) = &self.trace_id {
            span.record("krishiv.trace_id", t.as_str());
        }
        #[cfg(feature = "telemetry-otlp")]
        if let Some(remote) = &self.remote {
            use tracing_opentelemetry::OpenTelemetrySpanExt as _;
            if let Some(ctx) = remote_context(remote) {
                // Best effort: without an OpenTelemetry layer this does nothing.
                let _ = span.set_parent(ctx);
            }
        }
    }
}

/// The OpenTelemetry context of a caller's span, so the engine's span joins the caller's trace.
#[cfg(feature = "telemetry-otlp")]
fn remote_context(remote: &crate::system::RemoteParent) -> Option<opentelemetry::Context> {
    use opentelemetry::trace::{
        SpanContext, SpanId, TraceContextExt as _, TraceFlags, TraceId, TraceState,
    };
    let span_context = SpanContext::new(
        TraceId::from_hex(&remote.trace_id).ok()?,
        SpanId::from_hex(&remote.span_id).ok()?,
        TraceFlags::new(remote.flags),
        true,
        TraceState::default(),
    );
    Some(opentelemetry::Context::new().with_remote_span_context(span_context))
}

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
    pub(crate) fn span(&self, who: &Who) -> Span {
        let session_id = who.session_id.as_str();
        let span = tracing::info_span!(
            "krishiv.query",
            db.system.name = "krishiv",
            db.operation.name = tracing::field::Empty,
            db.query.summary = tracing::field::Empty,
            db.query.text = tracing::field::Empty,
            krishiv.session_id = %session_id,
            krishiv.query_id = tracing::field::Empty,
            krishiv.trace_id = tracing::field::Empty,
            krishiv.principal = tracing::field::Empty,
            krishiv.tenant = tracing::field::Empty,
        );
        self.fill(&span, session_id);
        who.record(&span);
        span
    }

    /// The span of `Session::sql` itself: parse, analysis and, for a command
    /// that runs eagerly (`INSERT ... SELECT`, DDL), its execution. A query's
    /// rows are produced later, inside the `krishiv.query` span of its action.
    pub(crate) fn prepare_span(&self, who: &Who) -> Span {
        let session_id = who.session_id.as_str();
        let span = tracing::info_span!(
            "krishiv.sql",
            db.system.name = "krishiv",
            db.operation.name = tracing::field::Empty,
            db.query.summary = tracing::field::Empty,
            db.query.text = tracing::field::Empty,
            krishiv.session_id = %session_id,
            krishiv.query_id = tracing::field::Empty,
            krishiv.trace_id = tracing::field::Empty,
            krishiv.principal = tracing::field::Empty,
            krishiv.tenant = tracing::field::Empty,
        );
        self.fill(&span, session_id);
        who.record(&span);
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
pub(crate) fn plain_span(who: &Who) -> Span {
    let session_id = who.session_id.as_str();
    let span = tracing::info_span!(
        "krishiv.query",
        db.system.name = "krishiv",
        krishiv.session_id = %session_id,
        krishiv.trace_id = tracing::field::Empty,
        krishiv.principal = tracing::field::Empty,
        krishiv.tenant = tracing::field::Empty,
    );
    who.record(&span);
    span
}

/// `stream` kept inside `span` until it is dropped, and polled inside it: the
/// span lasts as long as the result is being produced and read, not only the
/// planning. A disabled span adds nothing.
pub(crate) fn in_span(stream: RecordBatchStream, span: Span) -> RecordBatchStream {
    wrap_span(stream, span)
}

fn wrap_span(stream: RecordBatchStream, span: Span) -> RecordBatchStream {
    if span.is_disabled() {
        return stream;
    }
    Box::pin(SpanStream {
        inner: stream,
        span,
        started: std::time::Instant::now(),
        rows: 0,
        batches: 0,
        done: false,
    })
}

/// `stream` kept open inside `span`: the span closes when the stream is dropped, so a child
/// stage (`krishiv.execute`) lasts while the rows are produced, not only until `run` returns.
/// A disabled span adds nothing.
pub(crate) fn hold_span(stream: RecordBatchStream, span: Span) -> RecordBatchStream {
    if span.is_disabled() {
        return stream;
    }
    Box::pin(HeldSpan {
        inner: stream,
        span,
    })
}

struct HeldSpan {
    inner: RecordBatchStream,
    span: Span,
}

impl Stream for HeldSpan {
    type Item = <RecordBatchStream as Stream>::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        let _entered = this.span.enter();
        this.inner.as_mut().poll_next(cx)
    }
}

struct SpanStream {
    inner: RecordBatchStream,
    span: Span,
    started: std::time::Instant,
    rows: u64,
    batches: u64,
    /// The end (finish or failure) was logged.
    done: bool,
}

impl Stream for SpanStream {
    type Item = <RecordBatchStream as Stream>::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        let _entered = this.span.enter();
        let polled = this.inner.as_mut().poll_next(cx);
        match &polled {
            Poll::Ready(Some(Ok(batch))) => {
                this.rows += batch.num_rows() as u64;
                this.batches += 1;
            }
            // One record at the end of the result, inside the query's span: the span's fields
            // (query id, session id, trace id, principal) are on the record, and nothing of the
            // statement's text or values is (that is the span's sanitized `db.query.text`).
            Poll::Ready(None) if !this.done => {
                this.done = true;
                tracing::info!(
                    target: "krishiv::query",
                    rows = this.rows,
                    batches = this.batches,
                    elapsed_ms = u64::try_from(this.started.elapsed().as_millis()).unwrap_or(u64::MAX),
                    "query finished"
                );
            }
            Poll::Ready(Some(Err(e))) if !this.done => {
                this.done = true;
                tracing::warn!(
                    target: "krishiv::query",
                    error_class = e.error_class().name(),
                    rows = this.rows,
                    elapsed_ms = u64::try_from(this.started.elapsed().as_millis()).unwrap_or(u64::MAX),
                    "query failed"
                );
            }
            _ => {}
        }
        polled
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
            let span = plain_span(&Who {
                session_id: "s".to_owned(),
                trace_id: None,
                #[cfg(feature = "telemetry-otlp")]
                remote: None,
                principal: "p".to_owned(),
                tenant: None,
            });
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
