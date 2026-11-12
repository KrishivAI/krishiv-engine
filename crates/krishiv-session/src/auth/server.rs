// SPDX-License-Identifier: Apache-2.0
//! [`ServerAuth`], the one authenticator of a server (design §3.1).
//!
//! The chain, in a fixed order: a throttled peer is refused first; then a verified client
//! certificate names the principal (when client certificates are on; a bearer token sent with it
//! must name the same principal); then the bearer token goes to the static tokens (OIDC joins in
//! a later unit); no credential is [`AuthFailure::Missing`]. Every failure writes one log
//! record (target `krishiv::auth`: surface, method, peer, failure; never the credential) and
//! counts once per `(surface, method)`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use futures::future::BoxFuture;
use krishiv_common::error::KrishivError;
use krishiv_common::principal::Principal;

use super::config::{AuthConfig, RoleLists, SPARK_TOKEN_NAME};
use super::mtls::{ClientCertificates, MtlsConfig};
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
    /// Indexed by `surface * Method::ALL.len() + method`.
    failures: [AtomicU64; Surface::ALL.len() * Method::ALL.len()],
}

impl ServerAuth {
    /// Builds the authenticator: resolves and hashes every token. Logs a warning when Spark's
    /// all-powerful `spark.connect.authenticate.token` is in use.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` from [`StaticTokens::build`].
    pub fn new(config: &AuthConfig) -> Result<Self, KrishivError> {
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
            failures: std::array::from_fn(|_| AtomicU64::new(0)),
        })
    }

    /// Whether any method is configured. A server with none runs with authentication off
    /// (loopback only; the listener enforces that).
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        !self.tokens.is_empty() || self.mtls.is_on()
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
        // A token longer than any credential this server issues is refused unhashed.
        let token = token.filter(|t| t.len() <= MAX_TOKEN_BYTES);
        match token.and_then(|t| self.tokens.lookup(t)) {
            Some((principal, role)) => Ok(Authenticated::new(
                principal.clone(),
                Some(role),
                Method::Token,
            )),
            None => {
                if self.throttle.record_failure(ip, now) {
                    tracing::warn!(
                        target: "krishiv::auth",
                        peer = %ip,
                        "authentication failures from this address are throttled"
                    );
                }
                Err(self.fail(credentials, Method::Token, AuthFailure::Invalid))
            }
        }
    }

    /// A client with a verified certificate: the configured rule names the principal. A bearer
    /// token sent as well must name the same principal (a confused gateway is refused).
    fn authenticate_certificate(
        &self,
        credentials: &Credentials<'_>,
        leaf: &[u8],
        now: Instant,
    ) -> Result<Authenticated, AuthFailure> {
        let ip = credentials.peer.ip();
        let named = self.mtls.principal_of(leaf);
        let agrees = |name: &str| match (credentials.bearer, credentials.malformed) {
            (None, false) => true,
            (Some(t), _) if t.len() <= MAX_TOKEN_BYTES => {
                self.tokens.lookup(t).is_some_and(|(p, _)| p.name == name)
            }
            _ => false,
        };
        match named {
            Some((name, kind)) if agrees(&name) => {
                let role = self.roles.role_of(&name);
                Ok(Authenticated::new(
                    Principal::new(name, kind),
                    role,
                    Method::Mtls,
                ))
            }
            _ => {
                self.throttle.record_failure(ip, now);
                Err(self.fail(credentials, Method::Mtls, AuthFailure::Invalid))
            }
        }
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
        let result = self.authenticate_at(credentials, Instant::now());
        Box::pin(std::future::ready(result))
    }
}

/// The counter slot of `(surface, method)`.
fn index(surface: Surface, method: Method) -> usize {
    let s = Surface::ALL.iter().position(|x| *x == surface).unwrap_or(0);
    let m = Method::ALL.iter().position(|x| *x == method).unwrap_or(0);
    s * Method::ALL.len() + m
}
