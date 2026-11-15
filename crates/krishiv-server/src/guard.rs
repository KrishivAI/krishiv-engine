// SPDX-License-Identifier: Apache-2.0
//! Who may use an admin-listener route (P3-T002 design §3.1, ADR 0046 §3, §3c, §8).
//!
//! Every guarded route of the admin listener (`/metrics`, the UI's `/api/internal/*`) asks one
//! [`Guard`] whether a request may go on. A route states what it needs as an [`Access`]; the
//! surface extracts the presented token its own way (a header for `/metrics`, a header, cookie
//! or query parameter for the UI) and hands it over in a [`GuardRequest`].
//!
//! - [`TokenGuard`]: one shared secret (the token `krishiv server` prints when no
//!   authentication is configured, loopback only). The token is the whole check: every
//!   [`Access`] is granted to its holder.
//! - `SessionGuard` (feature `session`): the server's `ServerAuth` and the ADR 0046 role
//!   matrix. Failures are `401` (`503` while the peer is throttled), denials `403`.
//!
//! `/healthz` and `/readyz` are never guarded (ADR 0046 §3c).

use std::fmt;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::connect_info::Connected;
use axum::serve::IncomingStream;
use futures::future::BoxFuture;

/// What a route needs beyond an authenticated caller (ADR 0046 §3, §3c).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Access {
    /// Any authenticated principal, whatever its role (`/api/v1/version`).
    Authenticated,
    /// A `/metrics` scrape: `Monitor` on `System("metrics")`, held by every role.
    Metrics,
    /// Other principals' work shown in the UI: `Monitor` on `System("system")` (operator).
    Monitor,
    /// Cancelling another principal's query from the UI: `Operate` on `System("queries")`
    /// (operator).
    Operate,
}

/// The surface a guarded request arrived on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GuardSurface {
    /// `GET /metrics`.
    Metrics,
    /// The web UI and its `/api/internal` routes.
    Ui,
    /// The admin HTTP API.
    Admin,
}

/// What a surface extracted from one request. Never logged: `Debug` hides the token.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub struct GuardRequest<'a> {
    /// The presented token, if the surface found one.
    pub token: Option<&'a str>,
    /// An `Authorization` header was present but held no usable `Bearer` token.
    pub malformed: bool,
    /// The peer's address, when the listener records it ([`PeerAddr`]).
    pub peer: Option<SocketAddr>,
    /// The surface.
    pub surface: GuardSurface,
    /// What the route needs.
    pub access: Access,
}

impl<'a> GuardRequest<'a> {
    /// A request on `surface` for `access`, with no token and no peer.
    #[must_use]
    pub fn new(surface: GuardSurface, access: Access) -> Self {
        Self {
            token: None,
            malformed: false,
            peer: None,
            surface,
            access,
        }
    }

    /// The same request presenting `token`.
    #[must_use]
    pub fn with_token(mut self, token: Option<&'a str>) -> Self {
        self.token = token;
        self
    }

    /// The same request marked as carrying an unusable `Authorization` header.
    #[must_use]
    pub fn with_malformed(mut self, malformed: bool) -> Self {
        self.malformed = malformed;
        self
    }

    /// The same request from `peer`.
    #[must_use]
    pub fn with_peer(mut self, peer: Option<SocketAddr>) -> Self {
        self.peer = peer;
        self
    }
}

impl fmt::Debug for GuardRequest<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GuardRequest")
            .field("token", &self.token.map(|_| "<redacted>"))
            .field("malformed", &self.malformed)
            .field("peer", &self.peer)
            .field("surface", &self.surface)
            .field("access", &self.access)
            .finish()
    }
}

/// Why a request was refused. Carries no credential and no policy reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Refusal {
    /// No valid credential (`401`, `WWW-Authenticate: Bearer`).
    Unauthenticated,
    /// The peer is throttled after repeated failures, or a provider is unreachable (`503`).
    Unavailable,
    /// Authenticated, but the role does not hold what the route needs (`403`).
    Denied,
    /// The guard cannot decide (no peer address recorded): refused, fail closed (`500`).
    Misconfigured,
}

/// Decides whether a request may use a guarded route.
pub trait Guard: Send + Sync + 'static {
    /// `Ok` lets the request through.
    ///
    /// # Errors
    /// The [`Refusal`]; the guard has already logged it.
    fn check<'a>(&'a self, request: GuardRequest<'a>) -> BoxFuture<'a, Result<(), Refusal>>;
}

/// One shared secret: the start-up token of a server without authentication.
#[derive(Clone)]
pub struct TokenGuard {
    token: Arc<str>,
}

impl TokenGuard {
    /// A guard accepting exactly `token`. An empty `token` accepts nothing.
    #[must_use]
    pub fn new(token: &str) -> Self {
        Self {
            token: Arc::from(token),
        }
    }
}

impl fmt::Debug for TokenGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenGuard").finish_non_exhaustive()
    }
}

impl Guard for TokenGuard {
    fn check<'a>(&'a self, request: GuardRequest<'a>) -> BoxFuture<'a, Result<(), Refusal>> {
        let ok = !self.token.is_empty()
            && request
                .token
                .is_some_and(|t| same_secret(t.as_bytes(), self.token.as_bytes()));
        Box::pin(std::future::ready(if ok {
            Ok(())
        } else {
            Err(Refusal::Unauthenticated)
        }))
    }
}

/// Whether `a` and `b` are equal, in time that depends only on their lengths.
pub(crate) fn same_secret(a: &[u8], b: &[u8]) -> bool {
    let mut diff = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        diff |= usize::from(a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0));
    }
    // `black_box` keeps the compiler from turning the fold into an early exit. The length is not
    // secret from a timing observer; the token's bytes are not revealed by the time taken.
    std::hint::black_box(diff) == 0
}

/// The peer address of an admin connection, recorded by [`crate::admin::serve_capped`] as
/// `ConnectInfo<PeerAddr>` for the failure throttle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerAddr(pub SocketAddr);

impl Connected<IncomingStream<'_, crate::admin::CappedListener>> for PeerAddr {
    fn connect_info(stream: IncomingStream<'_, crate::admin::CappedListener>) -> Self {
        Self(*stream.remote_addr())
    }
}

impl Connected<SocketAddr> for PeerAddr {
    fn connect_info(addr: SocketAddr) -> Self {
        Self(addr)
    }
}

/// The peer recorded on `extensions`, if the listener records one.
#[must_use]
pub fn peer_of(extensions: &axum::http::Extensions) -> Option<SocketAddr> {
    extensions
        .get::<axum::extract::ConnectInfo<PeerAddr>>()
        .map(|c| c.0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn run(guard: &dyn Guard, token: Option<&str>) -> Result<(), Refusal> {
        let request = GuardRequest::new(GuardSurface::Ui, Access::Operate).with_token(token);
        guard.check(request).await
    }

    #[tokio::test]
    async fn token_guard_accepts_exactly_its_token() {
        let g = TokenGuard::new("s3cret");
        assert_eq!(run(&g, Some("s3cret")).await, Ok(()));
        assert_eq!(run(&g, Some("s3cre")).await, Err(Refusal::Unauthenticated));
        assert_eq!(
            run(&g, Some("s3cret2")).await,
            Err(Refusal::Unauthenticated)
        );
        assert_eq!(run(&g, None).await, Err(Refusal::Unauthenticated));
        let empty = TokenGuard::new("");
        assert_eq!(run(&empty, Some("")).await, Err(Refusal::Unauthenticated));
    }

    #[test]
    fn debug_never_shows_the_token() {
        let r = GuardRequest::new(GuardSurface::Metrics, Access::Metrics).with_token(Some("x9q"));
        assert!(!format!("{r:?}").contains("x9q"));
        assert!(!format!("{:?}", TokenGuard::new("x9q")).contains("x9q"));
    }

    #[test]
    fn secrets_compare_by_content_and_length() {
        assert!(same_secret(b"abc", b"abc"));
        assert!(!same_secret(b"abc", b"abx"));
        assert!(!same_secret(b"abc", b"abcd"));
        assert!(!same_secret(b"abcd", b"abc"));
        assert!(same_secret(b"", b""));
    }
}
