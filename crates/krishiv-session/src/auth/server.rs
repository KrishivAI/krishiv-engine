// SPDX-License-Identifier: Apache-2.0
//! [`ServerAuth`], the one authenticator of a server (design §3.1).
//!
//! The chain, in a fixed order: a throttled peer is refused first; then a verified client
//! certificate names the principal (when client certificates are on; a bearer token sent with it
//! must name the same principal); then a bearer token shaped like a JWS goes to OIDC when OIDC is
//! configured (feature `oidc`), any other to the static tokens; no credential is
//! [`AuthFailure::Missing`]. An OIDC token whose key is not cached waits for one JWK set fetch
//! in [`Authenticator::authenticate`]; the synchronous [`ServerAuth::authenticate_at`] does not
//! wait and answers [`AuthFailure::Unavailable`] instead. Every failure writes one log
//! record (target `krishiv::auth`: surface, method, peer, failure; never the credential) and
//! counts once per `(surface, method)`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_common::principal::Principal;

use super::config::{AuthConfig, RoleLists, SPARK_TOKEN_NAME};
#[cfg(feature = "oidc")]
use super::jose::{self, Identity};
use super::mtls::{ClientCertificates, MtlsConfig};
#[cfg(feature = "oidc")]
use super::oidc::{Oidc, Verified, unix_now};
use super::throttle::FailureThrottle;
use super::tokens::StaticTokens;
use super::{AuthFailure, Authenticated, Authenticator, Credentials, Method, Surface};

/// The longest bearer token hashed (design §3.3's JWT cap); configured secrets share the cap.
const MAX_TOKEN_BYTES: usize = super::tokens::MAX_SECRET_BYTES;

/// One failure counter (`krishiv_auth_failed_total{surface, method}`, ADR 0046 §8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct FailureCount {
    /// The listener.
    pub surface: Surface,
    /// The method tried.
    pub method: Method,
    /// Failures so far.
    pub count: u64,
}

/// The authenticator a server builds once from its settings.
#[derive(Debug)]
pub struct ServerAuth {
    tokens: StaticTokens,
    mtls: MtlsConfig,
    roles: RoleLists,
    throttle: FailureThrottle,
    /// The OIDC verifier, when configured.
    #[cfg(feature = "oidc")]
    oidc: Option<Oidc>,
    /// The roles of OIDC role-claim values.
    #[cfg(feature = "oidc")]
    oidc_roles: RoleLists,
    /// Indexed by `surface * Method::ALL.len() + method`.
    failures: [AtomicU64; Surface::ALL.len() * Method::ALL.len()],
    /// The denials of this server's callers (its guards and sessions count into it).
    denied: std::sync::Arc<super::DeniedCounts>,
}

impl ServerAuth {
    /// Builds the authenticator: resolves and hashes every token, and starts the OIDC key
    /// fetch thread when OIDC is configured. Logs a warning when Spark's all-powerful
    /// `spark.connect.authenticate.token` is in use.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` from [`StaticTokens::build`], or when the OIDC thread
    /// cannot start.
    pub fn new(config: &AuthConfig) -> Result<Self, KrishivError> {
        #[cfg(feature = "oidc")]
        let oidc = if config.oidc.is_on() {
            Some(Oidc::start(&config.oidc)?)
        } else {
            None
        };
        Self::build(
            config,
            #[cfg(feature = "oidc")]
            oidc,
        )
    }

    /// [`ServerAuth::new`] with an OIDC verifier over an in-memory key source.
    #[cfg(test)]
    #[cfg(feature = "oidc")]
    pub(crate) fn with_key_source(
        config: &AuthConfig,
        source: Box<dyn super::oidc::KeySource>,
    ) -> Result<Self, KrishivError> {
        let oidc = Oidc::with_source(&config.oidc, source)?;
        Self::build(config, Some(oidc))
    }

    /// The OIDC verifier (tests).
    #[cfg(test)]
    #[cfg(feature = "oidc")]
    pub(crate) fn oidc(&self) -> Option<&Oidc> {
        self.oidc.as_ref()
    }

    fn build(
        config: &AuthConfig,
        #[cfg(feature = "oidc")] oidc: Option<Oidc>,
    ) -> Result<Self, KrishivError> {
        let tokens = StaticTokens::build(&config.tokens)?;
        if config.tokens.iter().any(|t| t.name == SPARK_TOKEN_NAME) {
            tracing::warn!(
                target: "krishiv::auth",
                "spark.connect.authenticate.token is set: it is one shared token with the admin \
                 role; prefer krishiv.auth.tokens.<name> with one principal and role per token"
            );
        }
        Ok(Self {
            tokens,
            mtls: config.mtls.clone(),
            roles: config.roles.clone(),
            throttle: FailureThrottle::new(config.throttle),
            #[cfg(feature = "oidc")]
            oidc,
            #[cfg(feature = "oidc")]
            oidc_roles: config.oidc.roles.clone(),
            failures: std::array::from_fn(|_| AtomicU64::new(0)),
            denied: std::sync::Arc::new(super::DeniedCounts::new()),
        })
    }

    /// Whether any method is configured. A server with none runs with authentication off
    /// (loopback only; the listener enforces that).
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        #[cfg(feature = "oidc")]
        let oidc = self.oidc.is_some();
        #[cfg(not(feature = "oidc"))]
        let oidc = false;
        !self.tokens.is_empty() || self.mtls.is_on() || oidc
    }

    /// Authenticates `credentials` at `now` (the clock is a parameter for tests).
    ///
    /// # Errors
    /// The [`AuthFailure`], after its log record and counter.
    pub fn authenticate_at(
        &self,
        credentials: &Credentials<'_>,
        now: Instant,
    ) -> Result<Authenticated, AuthFailure> {
        let ip = credentials.peer.ip();
        if self.throttle.is_blocked(ip, now) {
            return Err(self.fail(credentials, Method::Throttle, AuthFailure::Throttled));
        }
        if self.mtls.is_on() {
            if let Some(leaf) = credentials.peer_leaf {
                return self.authenticate_certificate(credentials, leaf, now);
            }
            if self.mtls.client_certificates == ClientCertificates::Required {
                // The handshake refuses a client without a certificate; a transport that did
                // not pass the leaf through must not fall back to a token.
                return Err(self.fail(credentials, Method::Mtls, AuthFailure::Missing));
            }
        }
        let token = match credentials.bearer {
            Some(token) => Some(token),
            None if credentials.malformed => None,
            None => return Err(self.fail(credentials, Method::None, AuthFailure::Missing)),
        };
        #[cfg(feature = "oidc")]
        if let Some(t) = token
            && self.oidc.is_some()
            && jose::looks_like_jws(t)
        {
            return self.finish_oidc(credentials, self.verify_oidc(t, now, false), now);
        }
        self.authenticate_static(credentials, token, now)
    }

    /// A bearer token (or a malformed `Authorization` header, `None`) against the static
    /// tokens.
    fn authenticate_static(
        &self,
        credentials: &Credentials<'_>,
        token: Option<&str>,
        now: Instant,
    ) -> Result<Authenticated, AuthFailure> {
        // A token longer than any credential this server issues is refused unhashed.
        let token = token.filter(|t| t.len() <= MAX_TOKEN_BYTES);
        match token.and_then(|t| self.tokens.lookup(t)) {
            Some((principal, role)) => Ok(Authenticated::new(
                principal.clone(),
                Some(role),
                Method::Token,
            )),
            None => {
                self.record_failure(credentials, now);
                Err(self.fail(credentials, Method::Token, AuthFailure::Invalid))
            }
        }
    }

    /// Counts a failure toward the peer's throttle, logging when it starts a block.
    fn record_failure(&self, credentials: &Credentials<'_>, now: Instant) {
        let ip = credentials.peer.ip();
        if self.throttle.record_failure(ip, now) {
            tracing::warn!(
                target: "krishiv::auth",
                peer = %ip,
                "authentication failures from this address are throttled"
            );
        }
    }

    /// A client with a verified certificate: the configured rule names the principal. A bearer
    /// token sent as well must name the same principal (a confused gateway is refused). An OIDC
    /// token that cannot be checked now, or has expired, fails as such (retryable, or "expired"),
    /// not as a mismatch.
    fn authenticate_certificate(
        &self,
        credentials: &Credentials<'_>,
        leaf: &[u8],
        now: Instant,
    ) -> Result<Authenticated, AuthFailure> {
        let Some((name, kind)) = self.mtls.principal_of(leaf) else {
            self.record_failure(credentials, now);
            return Err(self.fail(credentials, Method::Mtls, AuthFailure::Invalid));
        };
        match self.bearer_agrees(credentials, &name, now) {
            Ok(true) => {
                let role = self.roles.role_of(&name);
                Ok(Authenticated::new(
                    Principal::new(name, kind),
                    role,
                    Method::Mtls,
                ))
            }
            Ok(false) => {
                self.record_failure(credentials, now);
                Err(self.fail(credentials, Method::Mtls, AuthFailure::Invalid))
            }
            Err(failure) => Err(self.fail(credentials, Method::Oidc, failure)),
        }
    }

    /// Whether the bearer token sent next to a certificate names `name` (no token agrees). An
    /// OIDC token is checked with the cached keys only; one that cannot be checked now or has
    /// expired is that failure.
    fn bearer_agrees(
        &self,
        credentials: &Credentials<'_>,
        name: &str,
        #[cfg_attr(not(feature = "oidc"), expect(unused_variables, reason = "OIDC only"))]
        now: Instant,
    ) -> Result<bool, AuthFailure> {
        match (credentials.bearer, credentials.malformed) {
            (None, false) => Ok(true),
            #[cfg(feature = "oidc")]
            (Some(t), _) if self.oidc.is_some() && jose::looks_like_jws(t) => {
                match self.verify_oidc(t, now, false) {
                    Ok(Some(id)) => Ok(id.principal == name),
                    Ok(None) => Err(AuthFailure::Unavailable),
                    Err(AuthFailure::Invalid) => Ok(false),
                    Err(f) => Err(f),
                }
            }
            (Some(t), _) if t.len() <= MAX_TOKEN_BYTES => {
                Ok(self.tokens.lookup(t).is_some_and(|(p, _)| p.name == name))
            }
            _ => Ok(false),
        }
    }

    /// The denial counters of this server's callers, shared with whatever counts into them
    /// (`krishiv_auth_denied_total`, ADR 0046 §8).
    #[must_use]
    pub fn denials(&self) -> std::sync::Arc<super::DeniedCounts> {
        std::sync::Arc::clone(&self.denied)
    }

    /// `krishiv_auth_failed_total{surface, method}` and `krishiv_auth_denied_total{surface,
    /// privilege}` (ADR 0046 §8), every series including the zeros, for the server's `/metrics`
    /// (merged into the engine's snapshot).
    #[must_use]
    pub fn metrics(&self) -> krishiv_exec::MetricsSnapshot {
        let mut out = krishiv_exec::MetricsSnapshot::default();
        for f in self.failure_counts() {
            out.counters.push(krishiv_exec::CounterSample {
                name: "krishiv.auth.failed",
                unit: "{request}",
                labels: vec![
                    ("surface".to_owned(), f.surface.as_str().to_owned()),
                    ("method".to_owned(), f.method.as_str().to_owned()),
                ],
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "a counter past 2^53 loses only its low bits"
                )]
                value: f.count as f64,
            });
        }
        for d in self.denied.counts() {
            out.counters.push(krishiv_exec::CounterSample {
                name: "krishiv.auth.denied",
                unit: "{request}",
                labels: vec![
                    ("surface".to_owned(), d.surface.as_str().to_owned()),
                    ("privilege".to_owned(), d.privilege.to_string()),
                ],
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "a counter past 2^53 loses only its low bits"
                )]
                value: d.count as f64,
            });
        }
        out
    }

    /// The failure counters, every `(surface, method)` pair in order.
    #[must_use]
    pub fn failure_counts(&self) -> Vec<FailureCount> {
        let mut out = Vec::with_capacity(self.failures.len());
        for surface in Surface::ALL {
            for method in Method::ALL {
                out.push(FailureCount {
                    surface,
                    method,
                    count: self.failures[index(surface, method)].load(Ordering::Relaxed),
                });
            }
        }
        out
    }

    /// Logs and counts `failure`, and returns it.
    fn fail(&self, c: &Credentials<'_>, method: Method, failure: AuthFailure) -> AuthFailure {
        self.failures[index(c.surface, method)].fetch_add(1, Ordering::Relaxed);
        tracing::warn!(
            target: "krishiv::auth",
            surface = c.surface.as_str(),
            method = method.as_str(),
            peer = %c.peer,
            failure = failure.as_str(),
            "authentication failed"
        );
        failure
    }
}

impl Authenticator for ServerAuth {
    fn authenticate<'a>(
        &'a self,
        credentials: &'a Credentials<'a>,
    ) -> BoxFuture<'a, Result<Authenticated, AuthFailure>> {
        Box::pin(async move {
            #[cfg(feature = "oidc")]
            if let Some(oidc) = &self.oidc
                && let Some(token) = self.waiting_oidc_token(credentials)
            {
                let now = Instant::now();
                if self.throttle.is_blocked(credentials.peer.ip(), now) {
                    return Err(self.fail(credentials, Method::Throttle, AuthFailure::Throttled));
                }
                let verdict = match self.verify_oidc(token, now, true) {
                    // No cached key fits: wait for one fetch, then decide on what it brought.
                    Ok(None) => {
                        drop(oidc.refresh().await);
                        self.verify_oidc(token, Instant::now(), false)
                    }
                    v => v,
                };
                return self.finish_oidc(credentials, verdict, Instant::now());
            }
            self.authenticate_at(credentials, Instant::now())
        })
    }
}

#[cfg(feature = "oidc")]
impl ServerAuth {
    /// The OIDC verdict on `token`: `Ok(Some(identity))`, a failure, or `Ok(None)` when the
    /// caller may wait for a key fetch and should.
    fn verify_oidc(
        &self,
        token: &str,
        now: Instant,
        may_refresh: bool,
    ) -> Result<Option<Identity>, AuthFailure> {
        let Some(oidc) = &self.oidc else {
            return Err(AuthFailure::Invalid);
        };
        match oidc.verify(token, now, unix_now(), may_refresh) {
            Verified::Ok(identity) => Ok(Some(identity)),
            Verified::Failed(f) => Err(f),
            Verified::Refresh => Ok(None),
        }
    }

    /// Turns an OIDC verdict into the result: the role from the groups, the failure logged,
    /// counted and (unless the provider is unreachable) throttled.
    fn finish_oidc(
        &self,
        credentials: &Credentials<'_>,
        verdict: Result<Option<Identity>, AuthFailure>,
        now: Instant,
    ) -> Result<Authenticated, AuthFailure> {
        match verdict {
            Ok(Some(identity)) => Ok(self.oidc_principal(identity)),
            // Only `authenticate` asks to wait; reaching here without waiting is a refusal.
            Ok(None) => Err(self.fail(credentials, Method::Oidc, AuthFailure::Unavailable)),
            Err(failure) => {
                // Only a token that is wrong counts toward the peer's throttle: an expired one
                // was signed by the provider (a client retrying it is not guessing), and an
                // unreachable provider is not the client's fault.
                if failure == AuthFailure::Invalid {
                    self.record_failure(credentials, now);
                }
                Err(self.fail(credentials, Method::Oidc, failure))
            }
        }
    }

    /// An OIDC principal: a user, with its groups; the role is the highest any role-claim value
    /// has in `krishiv.auth.oidc.roles.*` (the lists are disjoint, so this is "the first list,
    /// in order admin, operator, viewer, holding any of the values").
    fn oidc_principal(&self, identity: Identity) -> Authenticated {
        let role = identity
            .groups
            .iter()
            .filter_map(|g| self.oidc_roles.role_of(g))
            .max();
        let mut principal = Principal::new(
            identity.principal,
            krishiv_common::principal::PrincipalKind::User,
        );
        principal.groups = identity.groups;
        Authenticated::new(principal, role, Method::Oidc)
    }

    /// The bearer token of a request that goes to OIDC without a client certificate: the one
    /// path that may wait for a key fetch.
    fn waiting_oidc_token<'c>(&self, credentials: &Credentials<'c>) -> Option<&'c str> {
        let certificate_path = self.mtls.is_on()
            && (credentials.peer_leaf.is_some()
                || self.mtls.client_certificates == ClientCertificates::Required);
        if certificate_path {
            return None;
        }
        credentials.bearer.filter(|t| jose::looks_like_jws(t))
    }
}

/// The counter slot of `(surface, method)`.
fn index(surface: Surface, method: Method) -> usize {
    let s = Surface::ALL.iter().position(|x| *x == surface).unwrap_or(0);
    let m = Method::ALL.iter().position(|x| *x == method).unwrap_or(0);
    s * Method::ALL.len() + m
}
