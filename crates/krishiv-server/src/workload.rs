// SPDX-License-Identifier: Apache-2.0
//! The server console's pages about who is connected and how work is admitted: the sessions of every client
//! (`system.sessions`), the resource groups (`system.resource_groups`), and ending a session.
//!
//! Both lists read through [`UiSource`](crate::internal::UiSource) like every other page and keep only the
//! columns named here. A console over one session lists that session alone and ends none; one over stored
//! profiles has neither list.

use axum::Extension;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Response;
use serde_json::{Value, json};

use crate::guard::Caller;
use crate::internal::{
    EDITOR_TAG_PREFIX, Row, SystemTable, UiState, error, int, json_response, pick, source_error,
    table, text,
};

/// Sessions one answer lists: the open ones first, so a long tail of ended sessions is what is cut.
const MAX_SESSIONS: usize = 1000;

const SESSION_COLUMNS: &[&str] = &[
    "session_id",
    "principal",
    "client_user_id",
    "client_type",
    "transport",
    "state",
    "created_at",
    "last_active_at",
    "closed_at",
    "close_reason",
    "idle_timeout_ms",
    "current_catalog",
    "current_namespace",
    "tags",
    "queries_running",
    "queries_started",
    "delegated_by",
];

const GROUP_COLUMNS: &[&str] = &[
    "group_id",
    "parent_id",
    "state",
    "scheduling_policy",
    "scheduling_weight",
    "priority",
    "workload_class",
    "hard_concurrency_limit",
    "max_queued",
    "soft_memory_limit_bytes",
    "per_query_memory_limit_bytes",
    "scan_quota_bytes",
    "scan_quota_period_s",
    "running",
    "queued",
    "memory_reserved_bytes",
    "scan_bytes_in_window",
    "scan_window_resets_at",
];

/// Whether a `system.sessions` row is a session that can still run something.
fn is_open(row: &Row) -> bool {
    text(row, "state") == "active"
}

/// A session's row as the page gets it: the allowlisted columns, without the editor's own tags.
fn pick_session(row: &Row) -> Value {
    let mut picked = pick(row, SESSION_COLUMNS);
    if let Some(Value::Object(tags)) = picked.get_mut("tags") {
        tags.retain(|key, _| !key.starts_with(EDITOR_TAG_PREFIX));
    }
    Value::Object(picked)
}

/// `GET /api/internal/sessions`: the open sessions, then the ended ones the engine still remembers, each
/// part by its last activity, newest first; `truncated` says that more than [`MAX_SESSIONS`] exist.
pub(crate) async fn sessions(State(s): State<UiState>) -> Response {
    let rows = match table(&s, SystemTable::Sessions).await {
        Ok(r) => r,
        Err(e) => return *e,
    };
    let mut listed: Vec<&Row> = rows.iter().collect();
    listed.sort_by_key(|r| (!is_open(r), std::cmp::Reverse(int(r, "last_active_at"))));
    let truncated = listed.len() > MAX_SESSIONS;
    listed.truncate(MAX_SESSIONS);
    json_response(
        StatusCode::OK,
        &json!({
            "sessions": listed.into_iter().map(pick_session).collect::<Vec<_>>(),
            "truncated": truncated,
        }),
    )
}

/// `POST /api/internal/sessions/{id}/close`: ends an open session and cancels what it runs. The router's
/// guard has already asked for the operator's privilege; a session that was ended is one audit record.
pub(crate) async fn close_session(
    State(s): State<UiState>,
    caller: Option<Extension<Caller>>,
    Path(id): Path<String>,
) -> Response {
    match s.source.close_session(&id).await {
        Ok(Some(cancelled)) => {
            let caller = caller.map(|c| c.0);
            tracing::info!(
                target: "krishiv::audit",
                principal = caller.as_ref().map_or("unknown", |c| c.principal.as_str()),
                role = caller.as_ref().and_then(|c| c.role.as_deref()).unwrap_or("none"),
                surface = "ui",
                action = "close_session",
                session = %id,
                queries_cancelled = cancelled,
                "session ended"
            );
            json_response(
                StatusCode::OK,
                &json!({"closed": true, "queries_cancelled": cancelled}),
            )
        }
        Ok(None) => error(
            StatusCode::NOT_FOUND,
            "not_open",
            "no open session with that id can be ended here",
        ),
        Err(e) => source_error(&e),
    }
}

/// `GET /api/internal/groups`: the resource groups in the table's own order, a parent before its children.
pub(crate) async fn groups(State(s): State<UiState>) -> Response {
    let rows = match table(&s, SystemTable::ResourceGroups).await {
        Ok(r) => r,
        Err(e) => return *e,
    };
    json_response(
        StatusCode::OK,
        &json!({
            "groups": rows
                .iter()
                .map(|r| Value::Object(pick(r, GROUP_COLUMNS)))
                .collect::<Vec<_>>(),
        }),
    )
}
