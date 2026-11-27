// SPDX-License-Identifier: Apache-2.0
//! Server authentication (the role model and the token chain).
//!
//! One [`Authenticator`] turns what a transport received (a bearer token, a verified client
//! certificate) into an [`Authenticated`] principal with a [`Role`]. The Spark
//! Connect listener, the admin HTTP server, `/metrics` and the UI all call the same
//! [`ServerAuth`]; none keeps a token of its own.
//!
//! What a role may do is not decided here: that is the role matrix, applied by the session's
//! access policy.
//!
//! - [`AuthConfig`] reads the `krishiv.auth.*` settings (and Spark's
//!   `spark.connect.authenticate.token`).
//! - [`StaticTokens`]: one principal per configured token, held as SHA-256 digests and compared
//!   in constant time against every entry.
//! - [`FailureThrottle`]: repeated failures from one peer address are refused for a while.
//! - [`MtlsConfig`]: client certificates; the one `principalFrom` rule reads the principal out
//!   of the leaf with a minimal DER walk ([`CertNames`]).
//! - [`TlsConfig`] and [`check_listener`]: the `krishiv.server.tls.*` settings and the startup
//!   refusals of an address, authentication and TLS that do not fit together; `ServerTls`
//!   (feature `tls`) is the rustls configuration the listeners serve, reloadable.
//! - [`OidcConfig`]: OIDC access tokens (feature `oidc`): JWS compact tokens parsed by a
//!   strict, fuzzed parser (`jose.rs`), signatures checked on aws-lc-rs with the algorithm pinned
//!   by construction (no `none`, no HMAC), keys from a cached JWK set fetched on a thread of its
//!   own, the role from the groups claim through the role lists.
//! - [`ServerAuth`]: the chain (throttle, client certificate, bearer token), its failure log record (target `krishiv::auth`, never the
//!   credential) and its failure counters.

mod basic;
mod config;
mod credential;
mod denials;
mod der;
mod impersonation;
#[cfg_attr(
    not(feature = "oidc"),
    expect(
        unused_imports,
        reason = "only `Alg` and the principal-name rule are used without the `oidc` verifier"
    )
)]
mod jose;
mod mtls;
#[cfg(feature = "oidc")]
mod oidc;
mod oidc_config;
mod principals;
mod roles;
mod secret_ref;
mod server;
mod throttle;
#[cfg(feature = "tls")]
mod tls;
mod tls_config;
mod tokens;

#[cfg(test)]
#[cfg(feature = "oidc")]
mod assertion_tests;
#[cfg(test)]
mod impersonation_tests;
#[cfg(test)]
mod mtls_tests;
#[cfg(test)]
mod oidc_config_tests;
#[cfg(test)]
#[cfg(feature = "oidc")]
mod oidc_http_tests;
#[cfg(test)]
#[cfg(feature = "oidc")]
mod oidc_tests;
#[cfg(test)]
mod tests;

use std::fmt;
use std::net::SocketAddr;

use futures::future::BoxFuture;
use krishiv_common::principal::Principal;

pub use basic::{BasicCredentials, BasicHeader, basic_header};
pub use config::{AuthConfig, RoleLists, ThrottleConfig, TokenSpec, is_auth_key};
pub use credential::{CredentialKind, ExchangeClaims, InboundCredential};
pub use denials::{DeniedCount, DeniedCounts};
pub use der::{CertNames, DerError};
pub use impersonation::{ActAs, ActAsHint, Denial, ImpersonationConfig};
pub use krishiv_jose::Alg;
pub use mtls::{ClientCertificates, MtlsConfig, PrincipalFrom};
pub use oidc_config::OidcConfig;
pub use principals::{ImpersonationTarget, PrincipalPrivilege, PrincipalRights, PrincipalTable};
pub use roles::RolePolicy;
pub use secret_ref::SecretRef;
pub use server::{FailureCount, ImpersonationCount, ServerAuth};
pub use throttle::FailureThrottle;
#[cfg(feature = "tls")]
pub use tls::ServerTls;
pub use tls_config::{TlsConfig, check_listener};
pub use tokens::StaticTokens;

/// A server role: a total order, so a check is `role >= required`.
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
    /// The Arrow Flight SQL service (on the Spark Connect listener).
    FlightSql,
}

impl Surface {
    /// Every surface, in counter order.
    pub const ALL: [Self; 5] = [
        Self::Connect,
        Self::Admin,
        Self::Metrics,
        Self::Ui,
        Self::FlightSql,
    ];

    /// The surface a session of `transport` is served by.
    ///
    /// The surface whose requests a session of `transport` serves: `Connect`, `FlightSql` and
    /// `Ui` map to themselves; an embedded or MCP session arrives on no listener, so `None`
    /// (a Flight SQL session counts its denials under `flight-sql`).
    #[must_use]
    pub fn of(transport: crate::Transport) -> Option<Self> {
        match transport {
            crate::Transport::Connect => Some(Self::Connect),
            crate::Transport::FlightSql => Some(Self::FlightSql),
            crate::Transport::Ui => Some(Self::Ui),
            crate::Transport::Embedded | crate::Transport::Mcp => None,
        }
    }

    /// The label value: `connect`, `admin`, `metrics`, `ui` or `flight-sql`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Connect => "connect",
            Self::Admin => "admin",
            Self::Metrics => "metrics",
            Self::Ui => "ui",
            Self::FlightSql => "flight-sql",
        }
    }
}

/// How a request authenticated (or tried to).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Method {
    /// A configured static token.
    Token,
    /// An OIDC access token (a JWT the configured issuer signed).
    Oidc,
    /// A client certificate the client CA verified.
    Mtls,
    /// Refused before any method ran (the peer is throttled).
    Throttle,
    /// No credential was presented.
    None,
    /// `Authorization: Basic` on the Flight SQL surface: a bearer token as the password, its
    /// principal's name as the user name.
    Basic,
}

impl Method {
    /// Every method, in counter order.
    pub const ALL: [Self; 6] = [
        Self::Token,
        Self::Oidc,
        Self::Mtls,
        Self::Throttle,
        Self::None,
        Self::Basic,
    ];

    /// The label value: `token`, `oidc`, `mtls`, `throttle`, `none` or `basic`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Token => "token",
            Self::Oidc => "oidc",
            Self::Mtls => "mtls",
            Self::Throttle => "throttle",
            Self::None => "none",
            Self::Basic => "basic",
        }
    }
}

/// What a transport received, borrowed for one authentication. Never logged.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub struct Credentials<'a> {
    /// The token of an `authorization: Bearer <token>` header or gRPC metadata entry.
    pub bearer: Option<&'a str>,
    /// The client's leaf certificate (DER), which the TLS layer verified against the client
    /// CA at the handshake. Only the verified leaf may be passed here.
    pub peer_leaf: Option<&'a [u8]>,
    /// The peer's address (the throttle's key).
    pub peer: SocketAddr,
    /// The listener.
    pub surface: Surface,
    /// An `Authorization` header was present but not a usable `Bearer` token (`Basic ..` off the
    /// Flight SQL surface, `Bearer` with no token): Spark answers "Invalid authentication
    /// token", not "missing".
    pub malformed: bool,
    /// The user name of an `Authorization: Basic` credential (its password is `bearer`): the
    /// token's principal must have this name. Only the Flight SQL surface sets it
    /// ([`Self::with_basic`]); on any other surface it is refused.
    pub basic_user: Option<&'a str>,
}

impl<'a> Credentials<'a> {
    /// Credentials with no bearer token.
    #[must_use]
    pub fn new(peer: SocketAddr, surface: Surface) -> Self {
        Self {
            bearer: None,
            peer_leaf: None,
            peer,
            surface,
            malformed: false,
            basic_user: None,
        }
    }

    /// The same credentials carrying `token`.
    #[must_use]
    pub fn with_bearer(mut self, token: &'a str) -> Self {
        self.bearer = Some(token);
        self
    }

    /// The same credentials with the bearer token of an `Authorization` header value.
    ///
    /// The same credentials with the token of an `Authorization` header value, if it is a
    /// `Bearer` one (scheme compared ignoring ASCII case, as RFC 9110 §11.1 asks).
    #[must_use]
    pub fn with_authorization_header(mut self, value: &'a str) -> Self {
        self.bearer = bearer_token(value);
        self.malformed = self.bearer.is_none();
        self
    }

    /// The same credentials carrying a decoded `Basic` credential: its password as the bearer
    /// token and its user name to check against the token's principal. The
    /// transport decodes `Basic` only on `Surface::FlightSql`.
    #[must_use]
    pub fn with_basic(mut self, basic: &'a BasicCredentials) -> Self {
        self.bearer = Some(basic.password());
        self.basic_user = Some(basic.user());
        self.malformed = false;
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
            .field("basic_user", &self.basic_user)
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
    /// the policy).
    pub role: Option<Role>,
    /// How it authenticated.
    pub method: Method,
    /// The credential's own id (an OIDC token's `jti`), for the audit record; never enforced.
    pub token_id: Option<String>,
    /// The user token itself, kept for delegated catalogs: an OIDC token or a
    /// signed assertion; `None` for a static token, a certificate or plain act-as metadata.
    pub credential: Option<InboundCredential>,
}

impl Authenticated {
    /// An authentication result.
    #[must_use]
    pub fn new(principal: Principal, role: Option<Role>, method: Method) -> Self {
        Self {
            principal,
            role,
            method,
            token_id: None,
            credential: None,
        }
    }

    /// This result keeping `credential`, the user token.
    #[must_use]
    pub fn with_credential(mut self, credential: Option<InboundCredential>) -> Self {
        self.credential = credential;
        self
    }

    /// This result with the credential's id `token_id`.
    #[must_use]
    pub fn with_token_id(mut self, token_id: Option<String>) -> Self {
        self.token_id = token_id;
        self
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

    /// The status message of the refusal.
    ///
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

    /// Whether the refusal means "retry later" rather than "unauthenticated".
    ///
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
