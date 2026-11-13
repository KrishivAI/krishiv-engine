// SPDX-License-Identifier: Apache-2.0
//! Authentication and request classes of the Spark Connect listener (P3-T002; design
//! `design/P3-T002-authentication.md` §3.1 and §3.6, ADR 0046 §2, §3b and §5).
//!
//! - [`AuthLayer`] authenticates **every RPC** from its headers alone, in front of the tonic
//!   service, so nothing of an unauthenticated request's body is read or decoded. Once per
//!   RPC, not per connection: tokens expire and an HTTP/2 connection outlives them.
//!   [`ConnectAuth::authenticate`] hands the bearer token of the `authorization` header and
//!   the peer address (the failure throttle's key) to the server's one [`Authenticator`]. A
//!   failure is `UNAUTHENTICATED` with Spark's messages; a throttled peer or an unreachable
//!   OIDC provider is `UNAVAILABLE`, which clients retry; a request without a peer address is
//!   refused (fail closed). A caller that authenticates but maps to **no role** is refused
//!   before any session exists (`PERMISSION_DENIED`, `KRISHIV_INSUFFICIENT_PERMISSIONS`, ADR
//!   0046 §2), with one `krishiv::audit` record. The result rides in the request extensions;
//!   [`ConnectAuth::caller`] reads it there and fails closed when it is missing. With
//!   authentication off every request is the server's local user, which the loopback-only
//!   listener rule keeps to this machine.
//! - [`SessionOwner`] is what a Connect session is bound to: the method, the principal's kind,
//!   tenant and name, **without its groups** (one OIDC issuer per server, so the method stands
//!   for it). An OIDC user whose group claim changes between two tokens keeps its sessions,
//!   and a static token and an OIDC subject with the same name never share one (design §7,
//!   open question; this is the proposed answer). A session's policy is fixed for its life,
//!   so a caller whose **role** differs from the role its session was built with does not get
//!   that session: the session manager replaces it (`crate::manager`). A session keeps the
//!   principal (groups included) of the caller that created it.
//! - [`check_rpc`] and [`check_command`] give every RPC and every `Command` arm its ADR 0046
//!   §3b class. Neither has a wildcard arm, so a proto update that adds an RPC or a command
//!   does not compile until the new arm is classified.
//! - [`role_sessions`] builds each caller's session as `krishiv server` does:
//!   `PathPrefixPolicy(RolePolicy(role))` (design §3.6).

mod layer;

use std::net::SocketAddr;
use std::sync::Arc;

use krishiv_common::config::Config;
use krishiv_common::error::KrishivError;
use krishiv_common::principal::{Principal, PrincipalKind};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::command::CommandType;
use krishiv_connect::proto::v4_3::ml_command::Command as MlCommandType;
use krishiv_plan::analyzer::insufficient_permissions;
use krishiv_session::auth::{
    AuthFailure, Authenticated, Authenticator, Credentials, Method, Role, RolePolicy, Surface,
};
use krishiv_session::{
    PathPrefixPolicy, PolicyVersion, Privilege, Securable, Session, StatementClass,
};
use tonic::codegen::http::HeaderMap;
use tonic::codegen::http::header::AUTHORIZATION;
use tonic::{Request, Status};

pub use layer::{AuthLayer, AuthService};

use crate::manager::SessionFactory;
use crate::status::status_of;

/// The `system` securable a principal needs `Usage` on to send code the server runs (ADR
/// 0046 §4).
const SYSTEM_UDF: &str = "udf";

/// The catalog the `global_temp` namespace belongs to, whatever catalog is current.
const SESSION_CATALOG: &str = "spark_catalog";

/// The namespace global temporary views live in.
const GLOBAL_TEMP: &str = "global_temp";

/// Who each Connect request acts for.
pub struct ConnectAuth {
    authenticator: Option<Arc<dyn Authenticator>>,
    local: Principal,
}

impl std::fmt::Debug for ConnectAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectAuth")
            .field("enabled", &self.is_enabled())
            .field("local", &self.local)
            .finish()
    }
}

impl Default for ConnectAuth {
    fn default() -> Self {
        Self::off()
    }
}

impl ConnectAuth {
    /// Authentication off: every request is the server's local user (loopback only).
    #[must_use]
    pub fn off() -> Self {
        Self {
            authenticator: None,
            local: Principal::local_user(),
        }
    }

    /// Every request authenticated by `authenticator` (the server's one `ServerAuth`, shared
    /// with the admin listener).
    #[must_use]
    pub fn new(authenticator: Arc<dyn Authenticator>) -> Self {
        Self {
            authenticator: Some(authenticator),
            local: Principal::local_user(),
        }
    }

    /// Whether requests are authenticated.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.authenticator.is_some()
    }

    fn local(&self) -> Authenticated {
        Authenticated::new(self.local.clone(), None, Method::None)
    }

    /// Authenticates one request from its `headers` and `peer` address ([`AuthLayer`] calls
    /// it before the body is read).
    ///
    /// # Errors
    /// `UNAUTHENTICATED` for a missing, malformed, wrong or expired credential (and for a
    /// request without a peer address); `UNAVAILABLE` for a throttled peer or a credential that
    /// cannot be checked now; `PERMISSION_DENIED` (`KRISHIV_INSUFFICIENT_PERMISSIONS`) for a
    /// caller with no role.
    pub async fn authenticate(
        &self,
        headers: &HeaderMap,
        peer: Option<SocketAddr>,
    ) -> Result<Authenticated, Status> {
        let Some(authenticator) = &self.authenticator else {
            return Ok(self.local());
        };
        let Some(peer) = peer else {
            tracing::warn!(
                target: "krishiv::auth",
                surface = Surface::Connect.as_str(),
                "refused a Connect request without a peer address"
            );
            return Err(failure_status(AuthFailure::Missing));
        };
        let mut credentials = Credentials::new(peer, Surface::Connect);
        let mut values = headers.get_all(AUTHORIZATION).iter();
        match (values.next(), values.next()) {
            (None, _) => {}
            // One header, readable as text: its `Bearer` token, if it is one.
            (Some(value), None) => match value.to_str() {
                Ok(text) => credentials = credentials.with_authorization_header(text),
                Err(_) => credentials.malformed = true,
            },
            // Two `authorization` entries: which one a proxy or the client meant is unknown.
            (Some(_), Some(_)) => credentials.malformed = true,
        }
        let caller = authenticator
            .authenticate(&credentials)
            .await
            .map_err(failure_status)?;
        if caller.role.is_none() {
            // ADR 0046 §2: no role, no session. `Browse` is the least any role holds.
            let error = insufficient_permissions(
                &caller.principal,
                &Securable::System("cluster".to_owned()),
                Privilege::Browse,
            );
            tracing::warn!(
                target: "krishiv::audit",
                principal = %caller.principal.name,
                method = caller.method.as_str(),
                surface = Surface::Connect.as_str(),
                error_class = error.error_class().name(),
                "refused a Connect caller with no role"
            );
            return Err(status_of(&error));
        }
        Ok(caller)
    }

    /// The caller of `request`: what [`AuthLayer`] attached to it, or the local user with
    /// authentication off.
    ///
    /// # Errors
    /// `UNAUTHENTICATED` when authentication is on and nothing was attached (the service runs
    /// without the layer): fail closed.
    pub fn caller<T>(&self, request: &Request<T>) -> Result<Authenticated, Status> {
        if !self.is_enabled() {
            return Ok(self.local());
        }
        request
            .extensions()
            .get::<Authenticated>()
            .cloned()
            .ok_or_else(|| {
                tracing::error!(
                    target: "krishiv::auth",
                    surface = Surface::Connect.as_str(),
                    "a Connect request reached the service without authentication"
                );
                failure_status(AuthFailure::Missing)
            })
    }
}

/// The gRPC status of an authentication failure (the failure is logged and counted by the
/// authenticator; the status never carries the credential).
fn failure_status(failure: AuthFailure) -> Status {
    if failure.is_retryable() {
        Status::unavailable(failure.message())
    } else {
        Status::unauthenticated(failure.message())
    }
}

/// The session factory of an authenticated server (design §3.6): each caller's session acts
/// for its principal under `PathPrefixPolicy(RolePolicy(role))` on `base`'s runtime. Client
/// code needs admin until the T014b sandbox merges (ADR 0046 N1). The path confinement of
/// `config` (`krishiv.io.local.allowedRoots`, `krishiv.io.objectStore.allowedPrefixes`) is
/// checked once here, so a bad root fails at startup.
///
/// # Errors
/// What `PathPrefixPolicy::from_config` refuses (`KRISHIV_INVALID_CONFIG_VALUE`).
pub fn role_sessions(base: Session, config: &Config) -> Result<Box<SessionFactory>, KrishivError> {
    let config = config.clone();
    let policy = move |role: Option<Role>| {
        let roles = RolePolicy::new(role, Role::Admin, PolicyVersion(0));
        PathPrefixPolicy::from_config(&config, Arc::new(roles))
    };
    policy(None)?;
    Ok(Box::new(move |caller: &Authenticated| {
        let policy = policy(caller.role)?;
        Ok(base.new_session_as(caller.principal.clone(), Arc::new(policy)))
    }))
}

/// What a Connect session is bound to (module docs): the session manager's key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionOwner {
    method: Method,
    kind: PrincipalKind,
    tenant: Option<String>,
    name: String,
}

impl SessionOwner {
    /// The owner `caller`'s sessions belong to.
    #[must_use]
    pub fn of(caller: &Authenticated) -> Self {
        Self {
            method: caller.method,
            kind: caller.principal.kind,
            tenant: caller.principal.tenant.clone(),
            name: caller.principal.name.clone(),
        }
    }
}

/// A Spark Connect RPC (`spark.connect.SparkConnectService`, Spark 4.3.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rpc {
    /// `ExecutePlan`; its commands are classified by [`check_command`].
    ExecutePlan,
    /// `AnalyzePlan`: `Browse` on what the plan names for most analyses; its `persist` /
    /// `unpersist` subtypes are [`Check::SideEffect`] and `input_files` is `Select` on the
    /// paths (ADR 0046 §3b), classified per subtype when PR-11b implements the RPC.
    AnalyzePlan,
    /// `Config`.
    Config,
    /// `AddArtifacts`.
    AddArtifacts,
    /// `ArtifactStatus`.
    ArtifactStatus,
    /// `Interrupt`.
    Interrupt,
    /// `ReattachExecute`.
    ReattachExecute,
    /// `ReleaseExecute`.
    ReleaseExecute,
    /// `ReleaseSession`.
    ReleaseSession,
    /// `FetchErrorDetails`.
    FetchErrorDetails,
    /// `CloneSession`.
    CloneSession,
    /// `GetStatus`.
    GetStatus,
}

/// What a request is checked as (ADR 0046 §3b).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Check {
    /// Nothing beyond the session lookup, which is keyed by the caller: its own sessions and
    /// operations, every role.
    OwnSession,
    /// The securables of the resolved plan, which the session's analyzer asks the policy
    /// about when it runs the plan (and, for SQL text, the statement classifier, statement by
    /// statement).
    Plan,
    /// A non-durable side effect (`checkpoint_command`; `AnalyzePlan` `persist` / `unpersist`):
    /// the plan's securables plus the operator floor of the ADR 0046 §3 side-effect row
    /// (a viewer is refused even for a plan that reads no table), applied with the SQL
    /// `CACHE TABLE` path when P3-T012 / PR-11b run them.
    SideEffect,
    /// The `Write` class: `Modify` on the target, plus `Create` when the save mode makes one,
    /// plus the source plan's securables, through the session's write path (P3-T012 runs it).
    Write,
    /// `privilege` on `securable`, recorded as `class` when denied, before anything runs.
    Privilege {
        /// What is asked about.
        securable: Securable,
        /// The privilege needed.
        privilege: Privilege,
        /// The statement class an audit record names.
        class: StatementClass,
    },
    /// Refused for every role: not supported in P3 (ADR 0046 §3b); `what` names it.
    Unsupported(&'static str),
}

fn client_code() -> Check {
    Check::Privilege {
        securable: Securable::System(SYSTEM_UDF.to_owned()),
        privilege: Privilege::Usage,
        class: StatementClass::Session,
    }
}

/// The class of `rpc` (ADR 0046 §3b). `ExecutePlan` and `AnalyzePlan` are [`Check::Plan`]: a
/// command is classified further by [`check_command`].
#[must_use]
pub fn check_rpc(rpc: Rpc) -> Check {
    match rpc {
        Rpc::ExecutePlan | Rpc::AnalyzePlan => Check::Plan,
        // Code the server may run: `Usage` on `System("udf")`.
        Rpc::AddArtifacts => client_code(),
        // A `Config.set` of an admin key needs `Manage` on the cluster: the key's
        // `SettableBy`, checked by the same path as SQL `SET`.
        Rpc::Config
        | Rpc::ArtifactStatus
        | Rpc::Interrupt
        | Rpc::ReattachExecute
        | Rpc::ReleaseExecute
        | Rpc::ReleaseSession
        | Rpc::FetchErrorDetails
        | Rpc::CloneSession
        | Rpc::GetStatus => Check::OwnSession,
    }
}

/// The class of `command` (ADR 0046 §3b). A command without a type is [`Check::Plan`]: the
/// translation refuses it.
#[must_use]
pub fn check_command(command: &pb::Command) -> Check {
    let Some(command) = &command.command_type else {
        return Check::Plan;
    };
    match command {
        CommandType::SqlCommand(_) => Check::Plan,
        CommandType::CheckpointCommand(_) => Check::SideEffect,
        CommandType::WriteOperation(_)
        | CommandType::WriteOperationV2(_)
        | CommandType::MergeIntoTableCommand(_) => Check::Write,
        CommandType::CreateDataframeView(view) => {
            if view.is_global {
                // A shared object: the `Ddl` class, as `CREATE GLOBAL TEMPORARY VIEW`.
                Check::Privilege {
                    securable: Securable::Namespace(vec![
                        SESSION_CATALOG.to_owned(),
                        GLOBAL_TEMP.to_owned(),
                    ]),
                    privilege: Privilege::Create,
                    class: StatementClass::Ddl,
                }
            } else {
                // A local temporary view: its plan is checked when it is read.
                Check::OwnSession
            }
        }
        CommandType::RegisterFunction(_)
        | CommandType::RegisterTableFunction(_)
        | CommandType::RegisterDataSource(_) => client_code(),
        CommandType::GetResourcesCommand(_) | CommandType::RemoveCachedRemoteRelationCommand(_) => {
            Check::OwnSession
        }
        CommandType::MlCommand(ml) => match &ml.command {
            // A stock client sends it before every test and at exit: nothing to clean.
            Some(MlCommandType::CleanCache(_)) => Check::OwnSession,
            Some(
                MlCommandType::Fit(_)
                | MlCommandType::Fetch(_)
                | MlCommandType::Delete(_)
                | MlCommandType::Write(_)
                | MlCommandType::Read(_)
                | MlCommandType::Evaluate(_)
                | MlCommandType::GetCacheInfo(_)
                | MlCommandType::CreateSummary(_)
                | MlCommandType::GetModelSize(_),
            )
            | None => Check::Unsupported("ml_command"),
        },
        CommandType::CreateResourceProfileCommand(_) => {
            Check::Unsupported("create_resource_profile_command")
        }
        CommandType::ExecuteExternalCommand(_) => Check::Unsupported("execute_external_command"),
        CommandType::PipelineCommand(_) => Check::Unsupported("pipeline_command"),
        CommandType::WriteStreamOperationStart(_)
        | CommandType::StreamingQueryCommand(_)
        | CommandType::StreamingQueryManagerCommand(_)
        | CommandType::StreamingQueryListenerBusCommand(_) => {
            Check::Unsupported("streaming commands")
        }
        // No extension is registered until P3-T015; one without a declared class is refused.
        CommandType::Extension(_) => Check::Unsupported("extension"),
    }
}

/// Applies the part of `check` that is decided before a request runs: a
/// [`Check::Privilege`] is asked of `session`'s policy. The other classes are applied where
/// the request runs (the session lookup, the analyzer, the write path) or refused by it.
///
/// # Errors
/// `KRISHIV_INSUFFICIENT_PERMISSIONS` (42501) when the session's principal lacks the
/// privilege.
pub async fn enforce(check: &Check, session: &Session) -> Result<(), KrishivError> {
    match check {
        Check::Privilege {
            securable,
            privilege,
            class,
        } => {
            session
                .authorize_privilege(securable, *privilege, *class)
                .await
        }
        Check::OwnSession
        | Check::Plan
        | Check::SideEffect
        | Check::Write
        | Check::Unsupported(_) => Ok(()),
    }
}

#[cfg(test)]
mod tests;
