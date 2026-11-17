// SPDX-License-Identifier: Apache-2.0
//! [`AdminApi`] over the engine runtime: the version, the `system.capabilities` rows, and the
//! sessions and queries of the runtime with ADR 0046's rules (design P3-T066 §4, §5.3):
//!
//! - a caller sees and acts on its own work with no privilege; other principals' work needs
//!   `Monitor` on `System("system")` to list and `Operate` on `System("sessions")` /
//!   `System("queries")` to end;
//! - an id the caller may not act on answers exactly like one that does not exist (no existence
//!   oracle), and the refusal is a denial: one `krishiv::audit` record, the denial counter, and an
//!   action record with `result = denied`;
//! - own work ends with the caller's own reason (`User` / the release of a session), other
//!   principals' with `Admin`, so `admin` in `system.queries.cancel_reason` always means someone
//!   else acted.

use std::collections::BTreeMap;
use std::sync::Arc;

use futures::future::{BoxFuture, FutureExt};
use krishiv_session::auth::{Authenticated, DeniedCounts, Role, RolePolicy, Surface};
use krishiv_session::{
    CancelReason, EngineRuntime, PolicyVersion, Principal, PrincipalKind, Privilege, QueryId,
    QueryInfo, Securable, SessionInfo,
};
use prost_types::Timestamp;

use super::action::{self, Action};
use super::page::Page;
use super::{
    ADMIN_V1_REVISION, AdminApi, AdminError, AdminResult, Call, CancelQueriesByTagRequest,
    CancelQueriesByTagResponse, CancelQueryRequest, CancelQueryResponse, Capability, ErrorKind,
    GetCapabilitiesResponse, GetVersionResponse, KillSessionRequest, KillSessionResponse,
    ListQueriesRequest, ListQueriesResponse, ListSessionsRequest, ListSessionsResponse, PREVIEW,
    Query, Session,
};

/// The admin API of a `krishiv server` process. Creating it records the `admin.v1` surface row
/// in `system.capabilities` (`("surface", "admin.v1", <revision>, "preview" | "")`), so a client
/// that needs a field of a newer revision can check the revision first.
#[derive(Clone)]
pub struct SessionAdmin {
    runtime: Arc<EngineRuntime>,
    denials: Option<Arc<DeniedCounts>>,
}

impl std::fmt::Debug for SessionAdmin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionAdmin").finish_non_exhaustive()
    }
}

impl SessionAdmin {
    /// The admin API over `runtime`; registers the `admin.v1` capability row.
    #[must_use]
    pub fn new(runtime: Arc<EngineRuntime>) -> Self {
        krishiv_session::register_surface(
            "admin.v1",
            &ADMIN_V1_REVISION.to_string(),
            if PREVIEW { "preview" } else { "" },
        );
        Self {
            runtime,
            denials: None,
        }
    }

    /// The same, counting its denials into `denials` (`krishiv_auth_denied_total`, ADR 0046 §8).
    #[must_use]
    pub fn with_denials(mut self, denials: Arc<DeniedCounts>) -> Self {
        self.denials = Some(denials);
        self
    }
}

/// What a caller may do, resolved from what the guard authenticated.
struct Who {
    principal: Principal,
    monitor: bool,
    operate_sessions: bool,
    operate_queries: bool,
}

impl Who {
    fn of(call: &Call) -> Self {
        let system = |n: &str| Securable::System(n.to_owned());
        let authenticated = call
            .caller
            .identity
            .as_ref()
            .and_then(|i| i.downcast_ref::<Authenticated>());
        if let Some(a) = authenticated {
            let policy = RolePolicy::new(a.role, Role::Admin, PolicyVersion(0));
            return Self {
                principal: a.principal.clone(),
                monitor: policy.allows(&system("system"), Privilege::Monitor),
                operate_sessions: policy.allows(&system("sessions"), Privilege::Operate),
                operate_queries: policy.allows(&system("queries"), Privilege::Operate),
            };
        }
        // The local administrator of a server without authentication (its start-up token).
        let admin = call
            .caller
            .identity
            .as_ref()
            .is_some_and(|i| i.is::<crate::guard::LocalAdmin>());
        Self {
            principal: Principal::new(&call.caller.principal, PrincipalKind::User),
            monitor: admin,
            operate_sessions: admin,
            operate_queries: admin,
        }
    }
}

fn timestamp(ms: i64) -> Timestamp {
    Timestamp {
        seconds: ms.div_euclid(1000),
        nanos: i32::try_from(ms.rem_euclid(1000) * 1_000_000).unwrap_or(0),
    }
}

fn tags(pairs: &[(String, String)]) -> BTreeMap<String, String> {
    pairs.iter().cloned().collect()
}

fn session_message(s: &SessionInfo) -> Session {
    Session {
        session_id: s.session_id.clone(),
        principal: s.principal.clone(),
        delegated_by: s.delegated_by.clone().unwrap_or_default(),
        client_type: s.client_type.clone().unwrap_or_default(),
        transport: s.transport.clone(),
        state: s.state.clone(),
        created: Some(timestamp(s.created_ms)),
        last_active: Some(timestamp(s.last_active_ms)),
        closed: s.closed_ms.map(timestamp),
        close_reason: s.close_reason.clone().unwrap_or_default(),
        tags: tags(&s.tags),
        queries_running: s.queries_running,
        queries_started: s.queries_started,
    }
}

fn query_message(q: &QueryInfo) -> Query {
    Query {
        query_id: q.id.clone(),
        session_id: q.session_id.clone(),
        principal: q.principal.clone(),
        state: q.state.clone(),
        statement_class: q.statement_class.clone().unwrap_or_default(),
        text: q.text.clone().unwrap_or_default(),
        submitted: Some(timestamp(q.submitted_ms)),
        started: q.started_ms.map(timestamp),
        ended: q.ended_ms.map(timestamp),
        error_class: q.error_class.clone().unwrap_or_default(),
        resource_group: q.resource_group.clone().unwrap_or_default(),
        tags: tags(&q.tags),
    }
}

/// A tag selector `k` (a pair with key `k`) or `k=v` (the exact pair).
fn tag_matches(selector: &str, tags: &[(String, String)]) -> bool {
    tags.iter().any(|(k, v)| match selector.split_once('=') {
        Some((sk, sv)) => k == sk && v == sv,
        None => k == selector,
    })
}

fn not_found(class: &str, what: &str, id: &str) -> AdminError {
    // The SQLSTATE comes from the registry (default profile): `OPERATION_NOT_FOUND` has none of its
    // own (`HY000`), `SESSION_NOT_FOUND` is `08003` only from Spark 4.3.
    let state = if class.ends_with("SESSION_NOT_FOUND") {
        krishiv_common::error::ErrorClass::InvalidHandleSessionNotFound.sql_state()
    } else {
        krishiv_common::error::ErrorClass::InvalidHandleOperationNotFound.sql_state()
    };
    AdminError::new(
        ErrorKind::NotFound,
        class,
        state.unwrap_or("HY000"),
        format!("{what} {id} was not found."),
    )
    .with_parameter("handle", id)
}

impl SessionAdmin {
    /// A refusal that looks like a missing id, written as the denial it is.
    fn denied(
        &self,
        call: &Call,
        privilege: Privilege,
        rpc: &str,
        target: &str,
        error: AdminError,
    ) -> AdminError {
        tracing::warn!(
            target: "krishiv::audit",
            principal = %call.caller.principal,
            role = call.caller.role.as_deref().unwrap_or("none"),
            surface = "admin",
            securable = %Securable::System("sessions".to_owned()),
            privilege = %privilege,
            error_class = "KRISHIV_INSUFFICIENT_PERMISSIONS",
            "access denied"
        );
        if let Some(d) = &self.denials {
            d.record(Surface::Admin, privilege);
        }
        self.log(call, rpc, target, false, "denied", &error.error_class);
        error.disguised_denial()
    }

    fn log(&self, call: &Call, rpc: &str, target: &str, own: bool, result: &str, class: &str) {
        action::record(&Action {
            principal: &call.caller.principal,
            role: call.caller.role.as_deref().unwrap_or("none"),
            auth_method: &call.caller.method,
            rpc,
            transport: call.transport,
            target,
            own,
            request_id: &call.request_id,
            result,
            error_class: class,
            replay: false,
        });
    }
}

impl AdminApi for SessionAdmin {
    fn idempotency_scope(&self, call: &Call) -> String {
        let p = Who::of(call).principal;
        format!(
            "{:?}|{:?}|{}|{:?}",
            p.kind, p.tenant, p.name, p.delegated_by
        )
    }

    fn get_version(&self, _: &Call) -> BoxFuture<'_, AdminResult<GetVersionResponse>> {
        async {
            Ok(GetVersionResponse {
                version: crate::VERSION.to_owned(),
                admin_v1_revision: ADMIN_V1_REVISION,
                preview: PREVIEW,
            })
        }
        .boxed()
    }

    fn get_capabilities(&self, _: &Call) -> BoxFuture<'_, AdminResult<GetCapabilitiesResponse>> {
        async {
            Ok(GetCapabilitiesResponse {
                capabilities: krishiv_session::capabilities()
                    .into_iter()
                    .map(|[kind, name, value, detail]| Capability {
                        kind,
                        name,
                        value,
                        detail,
                    })
                    .collect(),
            })
        }
        .boxed()
    }

    fn list_sessions(
        &self,
        call: &Call,
        request: ListSessionsRequest,
    ) -> BoxFuture<'_, AdminResult<ListSessionsResponse>> {
        let who = Who::of(call);
        async move {
            let filter = "sessions";
            let page = Page::parse(request.page_size, &request.page_token, filter)?;
            let rows: Vec<SessionInfo> = self
                .runtime
                .session_infos()
                .into_iter()
                .filter(|s| who.monitor || s.is_owned_by(&who.principal))
                .collect();
            let (rows, next) = page.apply(rows, |s| (s.created_ms, s.session_id.clone()), filter);
            Ok(ListSessionsResponse {
                sessions: rows.iter().map(session_message).collect(),
                next_page_token: next,
            })
        }
        .boxed()
    }

    fn kill_session(
        &self,
        call: &Call,
        request: KillSessionRequest,
    ) -> BoxFuture<'_, AdminResult<KillSessionResponse>> {
        let who = Who::of(call);
        let call = call.clone();
        async move {
            let id = request.session_id;
            let class = "INVALID_HANDLE.SESSION_NOT_FOUND";
            let found = self
                .runtime
                .session_infos()
                .into_iter()
                .find(|s| s.session_id == id);
            let own = found
                .as_ref()
                .is_some_and(|s| s.is_owned_by(&who.principal));
            if found.is_none() || !(own || who.operate_sessions) {
                let error = not_found(class, "The session", &id);
                // A caller who may act on any session gets the real answer: it does not exist.
                if who.operate_sessions {
                    self.log(&call, "KillSession", &id, false, "failed", class);
                    return Err(error);
                }
                return Err(self.denied(&call, Privilege::Operate, "KillSession", &id, error));
            }
            let reason = if own {
                CancelReason::ClientGone
            } else {
                CancelReason::Admin
            };
            let cancelled = self.runtime.kill_session(&id, reason);
            let Some(n) = cancelled else {
                self.log(&call, "KillSession", &id, own, "failed", class);
                return Err(not_found(class, "The session", &id));
            };
            self.log(&call, "KillSession", &id, own, "ok", "");
            Ok(KillSessionResponse {
                queries_cancelled: i64::try_from(n).unwrap_or(i64::MAX),
            })
        }
        .boxed()
    }

    fn list_queries(
        &self,
        call: &Call,
        request: ListQueriesRequest,
    ) -> BoxFuture<'_, AdminResult<ListQueriesResponse>> {
        let who = Who::of(call);
        async move {
            let states: Vec<String> = if request.states.is_empty() {
                vec!["running".to_owned(), "queued".to_owned()]
            } else {
                request.states.clone()
            };
            let filter = format!(
                "queries|{}|{}|{}|{}",
                states.join(","),
                request.session_id,
                request.principal,
                request.tag
            );
            let page = Page::parse(request.page_size, &request.page_token, &filter)?;
            let rows: Vec<QueryInfo> = self
                .runtime
                .query_infos()
                .into_iter()
                .filter(|q| who.monitor || q.is_owned_by(&who.principal))
                .filter(|q| states.contains(&q.state))
                .filter(|q| request.session_id.is_empty() || q.session_id == request.session_id)
                .filter(|q| request.principal.is_empty() || q.principal == request.principal)
                .filter(|q| request.tag.is_empty() || tag_matches(&request.tag, &q.tags))
                .collect();
            let (rows, next) = page.apply(rows, |q| (q.submitted_ms, q.id.clone()), &filter);
            Ok(ListQueriesResponse {
                queries: rows.iter().map(query_message).collect(),
                next_page_token: next,
            })
        }
        .boxed()
    }

    fn cancel_query(
        &self,
        call: &Call,
        request: CancelQueryRequest,
    ) -> BoxFuture<'_, AdminResult<CancelQueryResponse>> {
        let who = Who::of(call);
        let call = call.clone();
        async move {
            let id = request.query_id;
            let class = "INVALID_HANDLE.OPERATION_NOT_FOUND";
            let parsed = QueryId::parse(&id).ok();
            let found = parsed.as_ref().and_then(|p| {
                self.runtime
                    .query_infos()
                    .into_iter()
                    .find(|q| q.id == p.to_string())
            });
            let own = found
                .as_ref()
                .is_some_and(|q| q.is_owned_by(&who.principal));
            if found.is_none() || !(own || who.operate_queries) {
                let error = not_found(class, "The operation", &id);
                if who.operate_queries {
                    self.log(&call, "CancelQuery", &id, false, "failed", class);
                    return Err(error);
                }
                return Err(self.denied(&call, Privilege::Operate, "CancelQuery", &id, error));
            }
            let reason = if own {
                CancelReason::User
            } else {
                CancelReason::Admin
            };
            let cancelled = parsed.is_some_and(|p| self.runtime.cancel(&p, reason));
            if !cancelled {
                // Already finished: as an unknown id (design §5.3).
                self.log(&call, "CancelQuery", &id, own, "failed", class);
                return Err(not_found(class, "The operation", &id));
            }
            self.log(&call, "CancelQuery", &id, own, "ok", "");
            Ok(CancelQueryResponse {})
        }
        .boxed()
    }

    fn cancel_queries_by_tag(
        &self,
        call: &Call,
        request: CancelQueriesByTagRequest,
    ) -> BoxFuture<'_, AdminResult<CancelQueriesByTagResponse>> {
        let who = Who::of(call);
        let call = call.clone();
        async move {
            if request.selector.is_empty() {
                return Err(AdminError::invalid_argument("the selector is empty"));
            }
            // Without `Operate` only the caller's own queries match, so the answer never depends
            // on other principals' queries.
            let ids =
                self.runtime
                    .cancel_by_tag(&request.selector, &who.principal, who.operate_queries);
            let ids: Vec<String> = ids.iter().map(ToString::to_string).collect();
            let target = format!("{} -> [{}]", request.selector, ids.join(","));
            self.log(
                &call,
                "CancelQueriesByTag",
                &target,
                !who.operate_queries,
                "ok",
                "",
            );
            Ok(CancelQueriesByTagResponse { query_ids: ids })
        }
        .boxed()
    }
}
