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

    /// The `planning_ns` of row `id` (MODES-2), for the profile of its query; `None` when the row is gone or
    /// was never planned.
    pub(crate) fn row_planning_ns(&self, id: &str) -> Option<u64> {
        let mut ns = None;
        self.history.update(&self.session_id, id, |row| {
            ns = row.planning_ns.and_then(|p| u64::try_from(p).ok());
        });
        ns
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
