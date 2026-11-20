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
use super::{
    DrainRequest, DrainResponse, DrainStatus, GetConfigRequest, GetConfigResponse,
    GetDrainStatusResponse, GetReadinessResponse, GetResourceGroupConfigRequest,
    GetResourceGroupConfigResponse, ListResourceGroupsRequest, ListResourceGroupsResponse,
    ReadinessCheck, ReloadResourceGroupConfigRequest, ReloadResourceGroupConfigResponse,
    ReplaceResourceGroupConfigRequest, ReplaceResourceGroupConfigResponse, ResumeServingRequest,
    ResumeServingResponse, SetConfigDefaultsRequest, SetConfigDefaultsResponse,
    ValidateResourceGroupConfigRequest, ValidateResourceGroupConfigResponse,
};
use crate::drain::{DrainController, DrainState, Exiting, MAX_GRACE, MIN_GRACE};
use crate::readiness::{CheckState, Readiness};

/// What the drain and readiness RPCs act on (design P3-T066 §5.1, §5.2). The closures belong to the
/// process that owns the engine and the listeners: `start` begins (or shortens) a drain and makes
/// sure something drives it to `DRAINED`; `resume` undoes the engine side of a drain; `load` is
/// `(queries running or queued, sessions open)`.
pub struct DrainOps {
    /// The state machine.
    pub drain: Arc<DrainController>,
    /// The cached readiness checks.
    pub readiness: Arc<Readiness>,
    /// Starts or shortens a drain: the grace period and `exit_when_drained`.
    pub start: Box<dyn Fn(std::time::Duration, bool) + Send + Sync>,
    /// Resumes serving: the controller and the engine, together; refused once exiting.
    pub resume: Box<dyn Fn() -> Result<(), Exiting> + Send + Sync>,
    /// `(queries held, sessions open)`.
    pub load: Box<dyn Fn() -> (i64, i64) + Send + Sync>,
    /// The grace period a `Drain` without one gets (`krishiv.server.shutdownTimeout`).
    pub default_grace: std::time::Duration,
}

impl std::fmt::Debug for DrainOps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DrainOps").finish_non_exhaustive()
    }
}

/// The admin API of a `krishiv server` process. Creating it records the `admin.v1` surface row
/// in `system.capabilities` (`("surface", "admin.v1", <revision>, "preview" | "")`), so a client
/// that needs a field of a newer revision can check the revision first.
#[derive(Clone)]
pub struct SessionAdmin {
    runtime: Arc<EngineRuntime>,
    denials: Option<Arc<DeniedCounts>>,
    ops: Option<Arc<DrainOps>>,
    /// The `krishiv.auth.*` settings as the operator wrote them (references, never resolved).
    pub(super) startup_auth: Arc<[(String, String)]>,
}

impl std::fmt::Debug for SessionAdmin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionAdmin").finish_non_exhaustive()
    }
}

impl SessionAdmin {
    pub(super) fn runtime(&self) -> &EngineRuntime {
        &self.runtime
    }

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
            ops: None,
            startup_auth: Arc::from([]),
        }
    }

    /// The same, listing the `krishiv.auth.*` startup settings (as written, so a secret reference
    /// is shown, never its secret) in `GetConfig`: withheld unless the caller holds `Manage`.
    #[must_use]
    pub fn with_startup_auth(mut self, settings: Vec<(String, String)>) -> Self {
        self.startup_auth = settings.into();
        self
    }

    /// The same, serving the drain and readiness RPCs over `ops`.
    #[must_use]
    pub fn with_ops(mut self, ops: Arc<DrainOps>) -> Self {
        self.ops = Some(ops);
        self
    }

    /// The same, counting its denials into `denials` (`krishiv_auth_denied_total`, ADR 0046 §8).
    #[must_use]
    pub fn with_denials(mut self, denials: Arc<DeniedCounts>) -> Self {
        self.denials = Some(denials);
        self
    }
}

/// Runs `work` on the blocking pool; a panicked or cancelled task is an internal error.
async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> AdminResult<T> + Send + 'static,
) -> AdminResult<T> {
    tokio::task::spawn_blocking(work).await.unwrap_or_else(|_| {
        Err(AdminError::new(
            ErrorKind::Internal,
            "INTERNAL_ERROR",
            "XX000",
            "The request could not be completed.",
        ))
    })
}

/// The owner name of a caller with no identity: a control character, so no real principal has it.
const ANONYMOUS_OWNER: &str = "\u{0}anonymous";

/// What a caller may do, resolved from what the guard authenticated.
pub(super) struct Who {
    pub(super) principal: Principal,
    pub(super) monitor: bool,
    pub(super) operate_sessions: bool,
    pub(super) operate_queries: bool,
    pub(super) manage: bool,
}

impl Who {
    pub(super) fn of(call: &Call) -> Self {
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
                manage: policy.allows(&system("cluster"), Privilege::Manage),
            };
        }
        // The local administrator of a server without authentication (its start-up token).
        let admin = call
            .caller
            .identity
            .as_ref()
            .is_some_and(|i| i.is::<crate::guard::LocalAdmin>());
        // A caller with no identity owns nothing: its audit name ("unknown") must not match a real
        // user of that name, so it runs as a name no token or login can carry.
        let name = if call.caller.identity.is_none() {
            ANONYMOUS_OWNER
        } else {
            call.caller.principal.as_str()
        };
        Self {
            principal: Principal::new(name, PrincipalKind::User),
            monitor: admin,
            operate_sessions: admin,
            operate_queries: admin,
            manage: admin,
        }
    }
}

pub(super) fn timestamp(ms: i64) -> Timestamp {
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

/// The runtime's tag-selector rule, so a listing and a cancel by tag see the same queries.
fn tag_matches(selector: &str, tags: &[(String, String)]) -> bool {
    tags.iter()
        .any(|(k, v)| krishiv_session::resource_groups::tag_matches(selector, k, v))
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
        securable: &str,
        target: &str,
        error: AdminError,
    ) -> AdminError {
        tracing::warn!(
            target: "krishiv::audit",
            principal = %call.caller.principal,
            role = call.caller.role.as_deref().unwrap_or("none"),
            surface = "admin",
            securable = %Securable::System(securable.to_owned()),
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

    pub(super) fn log(
        &self,
        call: &Call,
        rpc: &str,
        target: &str,
        own: bool,
        result: &str,
        class: &str,
    ) {
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
    fn list_resource_groups(
        &self,
        call: &Call,
        request: ListResourceGroupsRequest,
    ) -> BoxFuture<'_, AdminResult<ListResourceGroupsResponse>> {
        let call = call.clone();
        async move { self.do_list_resource_groups(&call, &request) }.boxed()
    }

    fn get_resource_group_config(
        &self,
        call: &Call,
        _request: GetResourceGroupConfigRequest,
    ) -> BoxFuture<'_, AdminResult<GetResourceGroupConfigResponse>> {
        let call = call.clone();
        let me = self.clone();
        // File reads, writes and fsyncs: off the async workers (a stalled mount parks one thread).
        async move { blocking(move || me.do_get_resource_group_config(&call)).await }.boxed()
    }

    fn validate_resource_group_config(
        &self,
        call: &Call,
        request: ValidateResourceGroupConfigRequest,
    ) -> BoxFuture<'_, AdminResult<ValidateResourceGroupConfigResponse>> {
        let call = call.clone();
        async move { self.do_validate_resource_group_config(&call, &request) }.boxed()
    }

    fn replace_resource_group_config(
        &self,
        call: &Call,
        request: ReplaceResourceGroupConfigRequest,
    ) -> BoxFuture<'_, AdminResult<ReplaceResourceGroupConfigResponse>> {
        let call = call.clone();
        let me = self.clone();
        // File reads, writes and fsyncs: off the async workers (a stalled mount parks one thread).
        async move { blocking(move || me.do_replace_resource_group_config(&call, &request)).await }
            .boxed()
    }

    fn reload_resource_group_config(
        &self,
        call: &Call,
        _request: ReloadResourceGroupConfigRequest,
    ) -> BoxFuture<'_, AdminResult<ReloadResourceGroupConfigResponse>> {
        let call = call.clone();
        let me = self.clone();
        // File reads, writes and fsyncs: off the async workers (a stalled mount parks one thread).
        async move { blocking(move || me.do_reload_resource_group_config(&call)).await }.boxed()
    }

    fn get_config(
        &self,
        call: &Call,
        request: GetConfigRequest,
    ) -> BoxFuture<'_, AdminResult<GetConfigResponse>> {
        let call = call.clone();
        async move { self.do_get_config(&call, &request) }.boxed()
    }

    fn set_config_defaults(
        &self,
        call: &Call,
        request: SetConfigDefaultsRequest,
    ) -> BoxFuture<'_, AdminResult<SetConfigDefaultsResponse>> {
        let call = call.clone();
        async move { self.do_set_config_defaults(&call, &request) }.boxed()
    }

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
                return Err(self.denied(
                    &call,
                    Privilege::Operate,
                    "KillSession",
                    "sessions",
                    &id,
                    error,
                ));
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
            // Debug text quotes and escapes each field, so no two requests share a filter string.
            let filter = format!(
                "queries{:?}",
                (
                    &states,
                    &request.session_id,
                    &request.principal,
                    &request.tag
                )
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
            let (rows, next) = page.apply(
                rows,
                |q| (q.submitted_ms, format!("{}/{}", q.session_id, q.id)),
                &filter,
            );
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
            // The same id may be live in several sessions: the caller's own row first.
            let found = parsed.as_ref().and_then(|p| {
                let key = p.to_string();
                let mut rows = self
                    .runtime
                    .query_infos()
                    .into_iter()
                    .filter(|q| q.id == key)
                    .peekable();
                let mut first = None;
                for q in rows.by_ref() {
                    if q.is_owned_by(&who.principal) {
                        return Some(q);
                    }
                    first.get_or_insert(q);
                }
                first
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
                return Err(self.denied(
                    &call,
                    Privilege::Operate,
                    "CancelQuery",
                    "queries",
                    &id,
                    error,
                ));
            }
            let reason = if own {
                CancelReason::User
            } else {
                CancelReason::Admin
            };
            // Only the row that passed the ownership check: the same id in another session stays.
            let session_id = found
                .as_ref()
                .map(|q| q.session_id.clone())
                .unwrap_or_default();
            let cancelled =
                parsed.is_some_and(|p| self.runtime.cancel_in_session(&session_id, &p, reason));
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
                let error = AdminError::invalid_argument("the selector is empty");
                self.log(
                    &call,
                    "CancelQueriesByTag",
                    "",
                    false,
                    "failed",
                    &error.error_class,
                );
                return Err(error);
            }
            // Without `Operate` only the caller's own queries match, so the answer never depends
            // on other principals' queries.
            let ids =
                self.runtime
                    .cancel_by_tag(&request.selector, &who.principal, who.operate_queries);
            let mut ids: Vec<String> = ids.iter().map(ToString::to_string).collect();
            ids.sort();
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

    fn get_readiness(&self, call: &Call) -> BoxFuture<'_, AdminResult<GetReadinessResponse>> {
        let who = Who::of(call);
        let call = call.clone();
        async move {
            let ops = self.ops_for("GetReadiness")?;
            self.require(
                &call,
                who.monitor,
                Privilege::Monitor,
                "GetReadiness",
                "system",
            )?;
            let (ready, report) = ops.readiness.snapshot();
            let checks = report
                .into_iter()
                .map(|c| ReadinessCheck {
                    name: c.name,
                    state: match c.state {
                        CheckState::Ok => "ok",
                        CheckState::Failed => "failed",
                        CheckState::Unknown => "unknown",
                    }
                    .to_owned(),
                    required: c.required,
                    consecutive_failures: i32::try_from(c.consecutive_failures).unwrap_or(i32::MAX),
                })
                .collect();
            Ok(GetReadinessResponse { ready, checks })
        }
        .boxed()
    }

    fn get_drain_status(&self, call: &Call) -> BoxFuture<'_, AdminResult<GetDrainStatusResponse>> {
        let who = Who::of(call);
        let call = call.clone();
        async move {
            let ops = self.ops_for("GetDrainStatus")?;
            self.require(
                &call,
                who.monitor,
                Privilege::Monitor,
                "GetDrainStatus",
                "system",
            )?;
            Ok(GetDrainStatusResponse {
                status: Some(drain_status(ops)),
            })
        }
        .boxed()
    }

    fn drain(
        &self,
        call: &Call,
        request: DrainRequest,
    ) -> BoxFuture<'_, AdminResult<DrainResponse>> {
        let who = Who::of(call);
        let call = call.clone();
        async move {
            self.require(&call, who.manage, Privilege::Manage, "Drain", "cluster")?;
            let ops = match self.ops_for("Drain") {
                Ok(ops) => ops,
                Err(e) => {
                    // A refused mutation leaves its record, supported or not.
                    self.log(&call, "Drain", "", false, "failed", &e.error_class);
                    return Err(e);
                }
            };
            let grace = match request.grace_period {
                None => ops.default_grace,
                Some(d) => u64::try_from(d.seconds)
                    .ok()
                    .filter(|_| (0..1_000_000_000).contains(&d.nanos))
                    .map(|s| std::time::Duration::new(s, d.nanos.unsigned_abs()))
                    .filter(|g| (MIN_GRACE..=MAX_GRACE).contains(g))
                    .ok_or_else(|| {
                        let e = AdminError::invalid_argument(
                            "grace_period must be between 1 second and 24 hours",
                        );
                        self.log(&call, "Drain", "", false, "failed", &e.error_class);
                        e
                    })?,
            };
            (ops.start)(grace, request.exit_when_drained);
            self.log(&call, "Drain", "", false, "ok", "");
            Ok(DrainResponse {
                status: Some(drain_status(ops)),
            })
        }
        .boxed()
    }

    fn resume_serving(
        &self,
        call: &Call,
        _request: ResumeServingRequest,
    ) -> BoxFuture<'_, AdminResult<ResumeServingResponse>> {
        let who = Who::of(call);
        let call = call.clone();
        async move {
            self.require(
                &call,
                who.manage,
                Privilege::Manage,
                "ResumeServing",
                "cluster",
            )?;
            let ops = match self.ops_for("ResumeServing") {
                Ok(ops) => ops,
                Err(e) => {
                    // A refused mutation leaves its record, supported or not.
                    self.log(&call, "ResumeServing", "", false, "failed", &e.error_class);
                    return Err(e);
                }
            };
            if (ops.resume)().is_err() {
                let e = AdminError::new(
                    ErrorKind::FailedPrecondition,
                    "KRISHIV_ADMIN_PRECONDITION_FAILED",
                    "55000",
                    "The change was refused because the object is not in the state the request \
                     expects (exiting).",
                )
                .with_parameter("reason", "exiting");
                self.log(&call, "ResumeServing", "", false, "failed", &e.error_class);
                return Err(e);
            }
            self.log(&call, "ResumeServing", "", false, "ok", "");
            Ok(ResumeServingResponse {
                status: Some(drain_status(ops)),
            })
        }
        .boxed()
    }
}

impl SessionAdmin {
    fn ops_for(&self, rpc: &str) -> AdminResult<&DrainOps> {
        self.ops.as_deref().ok_or_else(|| {
            AdminError::new(
                ErrorKind::FailedPrecondition,
                "KRISHIV_UNSUPPORTED_FEATURE",
                "0A000",
                format!("{rpc} is not available on this server."),
            )
        })
    }

    /// Fails with the denial (counted, audited, recorded) unless `allowed`.
    pub(super) fn require(
        &self,
        call: &Call,
        allowed: bool,
        privilege: Privilege,
        rpc: &str,
        securable: &str,
    ) -> AdminResult<()> {
        if allowed {
            return Ok(());
        }
        let error = AdminError::new(
            ErrorKind::PermissionDenied,
            "KRISHIV_INSUFFICIENT_PERMISSIONS",
            "42501",
            "Permission denied.",
        );
        Err(self.denied(call, privilege, rpc, securable, "", error))
    }
}

fn drain_status(ops: &DrainOps) -> DrainStatus {
    let s = ops.drain.status();
    let (queries, sessions) = (ops.load)();
    let deadline = s.deadline_at.map(|at| {
        let since = at.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
        Timestamp {
            seconds: i64::try_from(since.as_secs()).unwrap_or(i64::MAX),
            nanos: i32::try_from(since.subsec_nanos()).unwrap_or(0),
        }
    });
    DrainStatus {
        state: match s.state {
            DrainState::Serving => "serving",
            DrainState::Draining => "draining",
            DrainState::Drained => "drained",
        }
        .to_owned(),
        deadline,
        exit_when_drained: s.exit_when_drained,
        queries_running: queries,
        sessions_open: sessions,
    }
}
