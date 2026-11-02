// SPDX-License-Identifier: Apache-2.0
//! The query span (ADR 0035 section 3, P2-T116): one `krishiv.query` span
//! per action of a frame that `Session::sql` built, with OpenTelemetry's
//! database attributes and the `krishiv.*` ones.
//!
//! The span carries `db.system.name`, `db.operation.name`,
//! `db.query.summary` and the sanitized `db.query.text` (every literal
//! replaced by `?`, comments dropped; [`krishiv_sql::sanitize_query_text`]),
//! plus `krishiv.session_id` and `krishiv.query_id`. Ids are minted here
//! until the session context (T136) brings principal, tags and the caller's
//! trace context. Without a subscriber `tracing` makes the span a no-op, so
//! an embedded engine pays a disabled-callsite check.

use std::sync::atomic::{AtomicU64, Ordering};

use tracing::Span;

/// What a span says about one statement.
#[derive(Debug, Clone)]
pub(crate) struct QueryTrace {
    pub(crate) query_id: String,
    pub(crate) operation: String,
    pub(crate) summary: String,
    pub(crate) text: String,
}

impl QueryTrace {
    /// The trace of `sql`, the `seq`-th statement of session `session_id`.
    pub(crate) fn of(session_id: &str, seq: u64, sql: &str) -> Self {
        let summary = krishiv_sql::query_summary(sql);
        let operation = summary
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_owned();
        Self {
            query_id: format!("{session_id}-{seq}"),
            operation,
            summary,
            text: krishiv_sql::sanitize_query_text(sql),
        }
    }

    /// The span of one action of the statement.
    pub(crate) fn span(&self, session_id: &str) -> Span {
        tracing::info_span!(
            "krishiv.query",
            db.system.name = "krishiv",
            db.operation.name = %self.operation,
            db.query.summary = %self.summary,
            db.query.text = %self.text,
            krishiv.session_id = %session_id,
            krishiv.query_id = %self.query_id,
        )
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

/// A session id: 16 hex digits, unique within the process and
/// unpredictable across processes (the std hasher's random keys).
pub(crate) fn new_session_id() -> String {
    use std::hash::{BuildHasher, Hasher, RandomState};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let mut h = RandomState::new().build_hasher();
    h.write_u64(NEXT.fetch_add(1, Ordering::Relaxed));
    format!("{:016x}", h.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trace_has_no_literals() {
        let t = QueryTrace::of(
            "abc",
            3,
            "SELECT * FROM orders WHERE name = 'bob' AND n > 7",
        );
        assert_eq!(t.query_id, "abc-3");
        assert_eq!(t.operation, "SELECT");
        assert_eq!(t.summary, "SELECT orders");
        assert_eq!(t.text, "SELECT * FROM orders WHERE name = ? AND n > ?");
    }

    #[test]
    fn session_ids_differ() {
        assert_ne!(new_session_id(), new_session_id());
        assert_eq!(new_session_id().len(), 16);
    }
}
