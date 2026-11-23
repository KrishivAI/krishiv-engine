// SPDX-License-Identifier: Apache-2.0
//! The Resources page's Python UDF workers: how many worker processes the server's pool runs now, for whom,
//! and the limits they are counted against.
//!
//! The pool belongs to the server binary, not to this crate: `krishiv server` hands the source a
//! [`UdfWorkersProvider`] and the page reads it through [`UiSource::udf_workers`](crate::internal::UiSource).
//! A console with no pool (one embedded session, stored profiles, a server on a system without the pool)
//! answers `{"pool": null}` and the page shows no table. Nothing here starts or stops a worker.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Response;
use serde_json::{Value, json};

use crate::internal::{UiState, json_response, source_error};

/// Owners one answer lists: the ones with the most workers first, so the cut is the tail of small ones.
pub const MAX_OWNERS: usize = 500;

/// The workers of one owner: a worker serves one principal and one session only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UdfWorkerOwner {
    /// The authenticated principal.
    pub principal: String,
    /// The session id.
    pub session_id: String,
    /// The principal's tenant, on a server that has tenants.
    pub tenant: Option<String>,
    /// Its live workers: idle, running a call or starting.
    pub live: u64,
    /// Those of them that wait for a call.
    pub idle: u64,
}

/// The Python UDF worker pool at one moment.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UdfWorkers {
    /// Live workers in all: idle, running a call or starting.
    pub live: u64,
    /// Those of them that wait for a call.
    pub idle: u64,
    /// `krishiv.python.worker.maxTotal`.
    pub max_total: u64,
    /// `krishiv.python.worker.maxPerSession`.
    pub max_per_session: u64,
    /// `krishiv.python.worker.maxPerTenant`.
    pub max_per_tenant: u64,
    /// Every owner with a live worker, in any order.
    pub owners: Vec<UdfWorkerOwner>,
}

/// Reads the pool now. It is called on a request's task, so it must not block: a snapshot under a short lock.
pub type UdfWorkersProvider = Arc<dyn Fn() -> UdfWorkers + Send + Sync>;

fn owner_json(o: &UdfWorkerOwner) -> Value {
    json!({
        "principal": o.principal,
        "session_id": o.session_id,
        "tenant": o.tenant,
        "live": o.live,
        // A worker that is not idle runs a call or is starting.
        "busy": o.live.saturating_sub(o.idle),
    })
}

/// The pool as the page gets it: the owners with the most workers first (then by tenant, principal and
/// session, so the order is stable between two reads), at most [`MAX_OWNERS`]; `truncated` says more have
/// workers.
fn pool_json(mut pool: UdfWorkers) -> Value {
    pool.owners.sort_by(|a, b| {
        b.live
            .cmp(&a.live)
            .then_with(|| a.tenant.cmp(&b.tenant))
            .then_with(|| a.principal.cmp(&b.principal))
            .then_with(|| a.session_id.cmp(&b.session_id))
    });
    let truncated = pool.owners.len() > MAX_OWNERS;
    pool.owners.truncate(MAX_OWNERS);
    json!({
        "live": pool.live,
        "busy": pool.live.saturating_sub(pool.idle),
        "max_total": pool.max_total,
        "max_per_session": pool.max_per_session,
        "max_per_tenant": pool.max_per_tenant,
        "owners": pool.owners.iter().map(owner_json).collect::<Vec<_>>(),
        "truncated": truncated,
    })
}

/// `GET /api/internal/udf-workers`: `pool` is the worker pool, or `null` when this console has none.
pub(crate) async fn udf_workers(State(s): State<UiState>) -> Response {
    match s.source.udf_workers().await {
        Ok(pool) => json_response(
            StatusCode::OK,
            &json!({"pool": pool.map_or(Value::Null, pool_json)}),
        ),
        Err(e) => source_error(&e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn owner(
        tenant: Option<&str>,
        principal: &str,
        session: &str,
        live: u64,
        idle: u64,
    ) -> UdfWorkerOwner {
        UdfWorkerOwner {
            principal: principal.to_owned(),
            session_id: session.to_owned(),
            tenant: tenant.map(str::to_owned),
            live,
            idle,
        }
    }

    #[test]
    fn the_owners_with_most_workers_come_first_and_busy_is_what_is_not_idle() {
        let body = pool_json(UdfWorkers {
            live: 6,
            idle: 2,
            max_total: 64,
            max_per_session: 4,
            max_per_tenant: 32,
            owners: vec![
                owner(Some("b"), "zoe", "s3", 1, 1),
                owner(Some("a"), "amy", "s2", 1, 0),
                owner(None, "bob", "s1", 4, 1),
            ],
        });
        assert_eq!(body["live"], 6);
        assert_eq!(body["busy"], 4);
        assert_eq!(body["max_total"], 64);
        assert_eq!(body["max_per_session"], 4);
        assert_eq!(body["max_per_tenant"], 32);
        assert_eq!(body["truncated"], false);
        let sessions: Vec<&str> = body["owners"]
            .as_array()
            .map(|o| o.iter().filter_map(|o| o["session_id"].as_str()).collect())
            .unwrap_or_default();
        assert_eq!(sessions, ["s1", "s2", "s3"]);
        assert_eq!(body["owners"][0]["busy"], 3);
        assert_eq!(body["owners"][0]["tenant"], Value::Null);
        assert_eq!(body["owners"][1]["tenant"], "a");
        assert_eq!(body["owners"][2]["busy"], 0);
    }

    #[test]
    fn more_owners_than_the_cap_are_cut_and_said_so_and_a_wrong_idle_count_is_no_underflow() {
        let owners = (0..=MAX_OWNERS)
            .map(|i| owner(None, "p", &format!("s{i:04}"), 1, 2))
            .collect();
        let body = pool_json(UdfWorkers {
            owners,
            ..UdfWorkers::default()
        });
        assert_eq!(body["truncated"], true);
        assert_eq!(body["owners"].as_array().map(Vec::len), Some(MAX_OWNERS));
        assert_eq!(body["owners"][0]["session_id"], "s0000");
        assert_eq!(body["owners"][0]["busy"], 0);
        assert_eq!(body["busy"], 0);
    }
}
