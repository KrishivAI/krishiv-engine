// SPDX-License-Identifier: Apache-2.0
//! What a statement stamps on its history row after it starts: tags and trace id.

use std::sync::Arc;

use super::QueryLog;

impl QueryLog {
    /// The tags row `id` carries now: the snapshot taken when the statement started (the
    /// session's tags then, and the request tags), which later `set_tag` calls do not change.
    pub(crate) fn row_tags(&self, id: &str) -> Option<Vec<(String, String)>> {
        let mut tags = None;
        self.history.update(&self.session_id, id, |row| {
            tags = Some(row.tags.clone());
        });
        tags
    }

    /// The resource group row `id` ran in and the nanoseconds it queued there; `(None, None)` when
    /// no group admitted it.
    pub(crate) fn row_queue(&self, id: &str) -> (Option<String>, Option<u64>) {
        let mut out = (None, None);
        self.history.update(&self.session_id, id, |row| {
            if row.resource_group.is_some() {
                out = (
                    row.resource_group.clone(),
                    row.queued_ns.and_then(|n| u64::try_from(n).ok()),
                );
            }
        });
        out
    }

    /// Sets the trace id of row `id` (a statement with a trace of its own).
    pub(crate) fn set_row_trace(&self, id: &str, trace_id: Option<String>) {
        self.history.update(&self.session_id, id, |row| {
            row.trace_id = trace_id;
        });
    }

    /// Adds the statement's request tags, as `(tag, "")` pairs, to the tags of row `id`.
    pub(crate) fn add_request_tags(&self, id: &str, tags: &[Arc<str>]) {
        if tags.is_empty() {
            return;
        }
        self.history.update(&self.session_id, id, |row| {
            crate::action::merge_request_tags(&mut row.tags, tags);
        });
    }
}
