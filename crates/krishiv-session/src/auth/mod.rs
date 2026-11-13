// SPDX-License-Identifier: Apache-2.0
//! Server authentication (P3-T002; design `design/P3-T002-authentication.md`, ADR 0046).
//!
//! One [`Authenticator`] turns what a transport received (a bearer token today; client
//! certificates in a later unit) into an [`Authenticated`] principal with a [`Role`]. The Spark
//! Connect listener, the admin HTTP server, `/metrics` and the UI all call the same
//! [`ServerAuth`]; none keeps a token of its own.
//!
//! What a role may do is not decided here: that is ADR 0046's matrix, applied by the session's
//! access policy.
//!
//! - [`AuthConfig`] reads the `krishiv.auth.*` settings (and Spark's
//!   `spark.connect.authenticate.token`).
//! - [`StaticTokens`]: one principal per configured token, held as SHA-256 digests and compared
//!   in constant time against every entry.
//! - [`FailureThrottle`]: repeated failures from one peer address are refused for a while.
//! - [`ServerAuth`]: the chain, its failure log record (target `krishiv::auth`, never the
//!   credential) and its failure counters.

mod config;
mod roles;
mod secret_ref;
mod server;
mod throttle;
mod tokens;

#[cfg(test)]
mod tests;

use std::fmt;
use std::net::SocketAddr;

use futures::future::BoxFuture;
use krishiv_common::principal::Principal;

pub use config::{AuthConfig, ThrottleConfig, TokenSpec, is_auth_key};
pub use roles::RolePolicy;
pub use secret_ref::SecretRef;
pub use server::{FailureCount, ServerAuth};
pub use throttle::FailureThrottle;
pub use tokens::StaticTokens;

/// A server role (ADR 0046 §1): a total order, so a check is `role >= required`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    /// Sees metadata and its own work.
    Viewer,
    /// Reads data, runs client code, acts on others' running work.
    Operator,
    /// Everything, including durable writes, DDL and cluster changes.
    Admin,
}

impl Role {
    /// The configuration spelling: `viewer`, `operator` or `admin`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Viewer => "viewer",
            Self::Operator => "operator",
            Self::Admin => "admin",
        }
    }

    /// Parses the configuration spelling, ignoring ASCII case.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        [Self::Viewer, Self::Operator, Self::Admin]
            .into_iter()
            .find(|r| r.as_str().eq_ignore_ascii_case(s))
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The listener a request arrived on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Surface {
    /// The Spark Connect gRPC listener.
    Connect,
    /// The admin HTTP API.
    Admin,
    /// `GET /metrics`.
    Metrics,
    /// The web UI and its `/api/internal` routes.
    Ui,
}

impl Surface {
    /// Every surface, in counter order.
    pub const ALL: [Self; 4] = [Self::Connect, Self::Admin, Self::Metrics, Self::Ui];

    /// The label value: `connect`, `admin`, `metrics` or `ui`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Connect => "connect",
            Self::Admin => "admin",
            Self::Metrics => "metrics",
            Self::Ui => "ui",
        }
    }
}

/// How a request authenticated (or tried to).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Method {
    /// A configured static token.
    Token,
    /// An OIDC access token (a later unit).
    Oidc,
    /// A client certificate (a later unit).
    Mtls,
    /// Refused before any method ran (the peer is throttled).
    Throttle,
    /// No credential was presented.
    None,
}

impl Method {
    /// Every method, in counter order.
    pub const ALL: [Self; 5] = [
        Self::Token,
        Self::Oidc,
        Self::Mtls,
        Self::Throttle,
        Self::None,
    ];

    /// The label value: `token`, `oidc`, `mtls` or `none`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Token => "token",
            Self::Oidc => "oidc",
            Self::Mtls => "mtls",
            Self::Throttle => "throttle",
            Self::None => "none",
        }
    }
}

/// What a transport received, borrowed for one authentication. Never logged.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub struct Credentials<'a> {
    /// The token of an `authorization: Bearer <token>` header or gRPC metadata entry.
    pub bearer: Option<&'a str>,
    /// The peer's address (the throttle's key).
    pub peer: SocketAddr,
    /// The listener.
    pub surface: Surface,
    /// An `Authorization` header was present but not a usable `Bearer` token (`Basic ..`,
    /// `Bearer` with no token): Spark answers "Invalid authentication token", not "missing".
    pub malformed: bool,
}

impl<'a> Credentials<'a> {
    /// Credentials with no bearer token.
    #[must_use]
    pub fn new(peer: SocketAddr, surface: Surface) -> Self {
        Self {
            bearer: None,
            peer,
            surface,
            malformed: false,
        }
    }

    /// The same credentials carrying `token`.
    #[must_use]
    pub fn with_bearer(mut self, token: &'a str) -> Self {
        self.bearer = Some(token);
        self
    }

    /// The same credentials with the token of an `Authorization` header value, if it is a
    /// `Bearer` one (scheme compared ignoring ASCII case, as RFC 9110 §11.1 asks).
    #[must_use]
    pub fn with_authorization_header(mut self, value: &'a str) -> Self {
        self.bearer = bearer_token(value);
        self.malformed = self.bearer.is_none();
        self
    }
}

impl fmt::Debug for Credentials<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Credentials")
            .field(
                "bearer",
                &self.bearer.map(|_| krishiv_common::secret::REDACTED),
            )
            .field("peer", &self.peer)
            .field("surface", &self.surface)
            .field("malformed", &self.malformed)
            .finish()
    }
}

/// The token of an `Authorization` header value `Bearer <token>`; `None` for any other scheme
/// or an empty token.
#[must_use]
pub fn bearer_token(value: &str) -> Option<&str> {
    let (scheme, token) = value.split_once(' ')?;
    let token = token.trim_start_matches(' ');
    (scheme.eq_ignore_ascii_case("Bearer") && !token.is_empty()).then_some(token)
}

/// A successful authentication.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Authenticated {
    /// Who the request acts for.
    pub principal: Principal,
    /// The role; `None` when the credential maps to no role (every request is then refused by
    /// the policy, ADR 0046 §2).
    pub role: Option<Role>,
    /// How it authenticated.
    pub method: Method,
}

impl Authenticated {
    /// An authentication result.
    #[must_use]
    pub fn new(principal: Principal, role: Option<Role>, method: Method) -> Self {
        Self {
            principal,
            role,
            method,
        }
    }
}

/// Why authentication failed. Never carries the credential.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AuthFailure {
    /// No credential where one is required.
    Missing,
    /// A credential that matches nothing.
    Invalid,
    /// A credential that has expired.
    Expired,
    /// The credential cannot be checked now (for example no OIDC key could be fetched): clients
    /// should retry.
    Unavailable,
    /// The peer failed too often and is refused for a while.
    Throttled,
}

impl AuthFailure {
    /// The label value and log field.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::Invalid => "invalid",
            Self::Expired => "expired",
            Self::Unavailable => "unavailable",
            Self::Throttled => "throttled",
        }
    }

    /// The status message: Spark's own two for a missing and a wrong token
    /// (`PreSharedKeyAuthenticationInterceptor`, Spark 4.3.0).
    #[must_use]
    pub fn message(self) -> &'static str {
        match self {
            Self::Missing => "No authentication token provided",
            Self::Invalid => "Invalid authentication token",
            Self::Expired => "Expired authentication token",
            Self::Unavailable => "Authentication is temporarily unavailable",
            Self::Throttled => "Too many failed authentication attempts",
        }
    }

    /// Whether the transport answers "retry later" (gRPC `UNAVAILABLE`, HTTP 503) rather than
    /// "unauthenticated" (gRPC `UNAUTHENTICATED`, HTTP 401).
    #[must_use]
    pub fn is_retryable(self) -> bool {
        matches!(self, Self::Unavailable | Self::Throttled)
    }
}

impl fmt::Display for AuthFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

/// Turns credentials into a principal. One implementation per server ([`ServerAuth`]); tests
/// may supply their own.
pub trait Authenticator: Send + Sync {
    /// Authenticates `credentials`. Asynchronous because an OIDC key may need fetching.
    fn authenticate<'a>(
        &'a self,
        credentials: &'a Credentials<'a>,
    ) -> BoxFuture<'a, Result<Authenticated, AuthFailure>>;
}
