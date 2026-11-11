// SPDX-License-Identifier: Apache-2.0
//! Rendering the SQL span of a run-time error as its query context: the
//! expression that raised it marked the error with a site
//! (`KrishivError::pending_site`), and the statement's own text, kept in the
//! query environment, is what that span indexes.

use std::sync::Arc;

use futures::StreamExt as _;
use krishiv_common::error::{ErrorQueryContext, KrishivError};
use krishiv_common::redact::RedactedSql;

use crate::session::{QueryEnv, RecordBatchStream};

impl QueryEnv {
    /// Sets the statement text, shown redacted, and keeps the map back to its offsets.
    pub(crate) fn set_text(&mut self, text: &RedactedSql) {
        self.text = Some(Arc::from(text.as_str()));
        self.redacted = Some(Arc::new(text.clone()));
    }

    /// Takes the statement text of `other`.
    pub(crate) fn copy_text_from(&mut self, other: &Self) {
        self.text.clone_from(&other.text);
        self.redacted.clone_from(&other.redacted);
    }
}

/// A run-time error marked with the span of SQL it arose at, with that span rendered as its query
/// context: `text` is the statement's own text, which only a span of source 0 indexes (a span in
/// a SQL function's body keeps no context). An error without a site passes unchanged.
pub(crate) fn resolve_site(error: KrishivError, text: Option<&RedactedSql>) -> KrishivError {
    let Some(site) = error.pending_site() else {
        return error;
    };
    let error = error.without_pending_site();
    match text {
        Some(t) if site.source == 0 && site.start < site.end => {
            sql_context(error, t, site.start as usize, site.end as usize)
        }
        _ => error,
    }
}

/// `stream` with the errors it yields resolved against the statement's text.
pub(crate) fn resolve_sites(stream: RecordBatchStream, env: &QueryEnv) -> RecordBatchStream {
    match env.redacted.clone() {
        Some(text) => Box::pin(stream.map(move |r| r.map_err(|e| resolve_site(e, Some(&text))))),
        None => stream,
    }
}

/// `error` with the SQL span `start..end` of `text` attached as its
/// query context (`stop` is inclusive, as Spark reports it).
pub(crate) fn sql_context(
    error: KrishivError,
    text: &RedactedSql,
    start: usize,
    end: usize,
) -> KrishivError {
    error.with_query_context(ErrorQueryContext::sql(
        text.fragment(start..end),
        start,
        end.saturating_sub(1),
    ))
}

/// The source of an expression written as DataFrame text (`select_expr`, `filter_expr`): no
/// statement's text, so no span of it renders as a context. A SQL statement's own source is 0.
pub(crate) const DF_SOURCE: krishiv_plan::SourceId = krishiv_plan::SourceId(u32::MAX);
