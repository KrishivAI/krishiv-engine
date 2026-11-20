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
use super::impersonation::{
    self, ActAs, ActAsHint, Channel, Decision, Denial, ImpersonationConfig,
};
#[cfg(feature = "oidc")]
use super::jose::{self, Identity};
use super::mtls::{ClientCertificates, MtlsConfig};
#[cfg(feature = "oidc")]
use super::oidc::{Oidc, Verified, unix_now};
use super::principals::PrincipalTable;
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
    /// The rights of named principals.
    principals: PrincipalTable,
    /// Trusted impersonation.
    impersonation: ImpersonationConfig,
    /// The verifier of signed act-as assertions, when `signedAssertion` is on: its own issuer,
    /// audience and JWK set cache, apart from OIDC's.
    #[cfg(feature = "oidc")]
    assertions: Option<Oidc>,
    /// `krishiv_auth_impersonation_total{decision}`: allowed, denied.
    impersonations: [AtomicU64; 2],
}

/// The impersonation counters (`krishiv_auth_impersonation_total{decision}`, design P3-T061
/// §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct ImpersonationCount {
    /// Requests that acted for another principal.
    pub allowed: u64,
    /// Requests refused.
    pub denied: u64,
}

impl ServerAuth {
    /// Readiness check 2 of the admin API (design P3-T066 §5.2): `None` when OIDC is not
    /// configured, else `Err(reason)` while the JWK set was never fetched or is older than
    /// three refresh periods.
    #[must_use]
    pub fn jwks_ready(&self) -> Option<Result<(), String>> {
        #[cfg(feature = "oidc")]
        {
            self.oidc.as_ref().map(Oidc::jwks_ready)
        }
        #[cfg(not(feature = "oidc"))]
        {
            None
        }
    }

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
        #[cfg(feature = "oidc")]
        let assertions = if config.impersonation.signed_assertion {
            Some(Oidc::start(&config.impersonation.assertion)?)
        } else {
            None
        };
        Self::build(
            config,
            #[cfg(feature = "oidc")]
            oidc,
            #[cfg(feature = "oidc")]
            assertions,
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
        Self::build(config, Some(oidc), None)
    }

    /// [`ServerAuth::new`] without OIDC, with the signed-assertion verifier over an in-memory
    /// key source.
    #[cfg(test)]
    #[cfg(feature = "oidc")]
    pub(crate) fn with_assertion_source(
        config: &AuthConfig,
        source: Box<dyn super::oidc::KeySource>,
    ) -> Result<Self, KrishivError> {
        let assertions = Oidc::with_source(&config.impersonation.assertion, source)?;
        Self::build(config, None, Some(assertions))
    }

    /// The signed-assertion verifier (tests).
    #[cfg(test)]
    #[cfg(feature = "oidc")]
    pub(crate) fn assertions(&self) -> Option<&Oidc> {
        self.assertions.as_ref()
    }

    /// This chain with `tokens` in place of the configured ones (tests: tokens built at
    /// another time).
    #[cfg(test)]
    pub(crate) fn with_tokens(mut self, tokens: StaticTokens) -> Self {
        self.tokens = tokens;
        self
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
        #[cfg(feature = "oidc")] assertions: Option<Oidc>,
    ) -> Result<Self, KrishivError> {
        // A token's own expiry and its principal's are one rule: the earlier applies.
        let tokens = StaticTokens::build(&super::tokens::with_principal_expiry(
            &config.tokens,
            &config.principals,
        ))?;
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
            principals: config.principals.clone(),
            impersonation: config.impersonation.clone(),
            #[cfg(feature = "oidc")]
            assertions,
            impersonations: [AtomicU64::new(0), AtomicU64::new(0)],
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
        if let Some(refused) = self.basic_off_flight(credentials) {
            return Err(refused);
        }
        let verdict = self.authenticate_credential(credentials, now);
        self.basic_user_agrees(credentials, verdict, now)
    }

    /// [`Self::authenticate_at`] before the `Basic` user name check.
    fn authenticate_credential(
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
            return self.finish_oidc(credentials, t, self.verify_oidc(t, now, false), now);
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
        match token.and_then(|t| self.tokens.lookup_at(t, std::time::SystemTime::now())) {
            // The right token past its date: not a guess, so the throttle does not count it.
            Some((_, _, true)) => Err(self.fail(credentials, Method::Token, AuthFailure::Expired)),
            Some((principal, role, false)) => Ok(Authenticated::new(
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

    /// A `Basic` credential anywhere but the Flight SQL surface: refused as malformed (a transport
    /// that decoded it there is wrong; P3-T101 note §3.1).
    fn basic_off_flight(&self, credentials: &Credentials<'_>) -> Option<AuthFailure> {
        (credentials.basic_user.is_some() && credentials.surface != Surface::FlightSql)
            .then(|| self.fail(credentials, Method::Basic, AuthFailure::Invalid))
    }

    /// With a `Basic` credential, the authenticated principal must have its user name (P3-T101
    /// note §3.1): a mismatch is a failed credential, counted toward the throttle; a match
    /// authenticated by `basic` (by `mtls` when a client certificate named the principal).
    fn basic_user_agrees(
        &self,
        credentials: &Credentials<'_>,
        verdict: Result<Authenticated, AuthFailure>,
        now: Instant,
    ) -> Result<Authenticated, AuthFailure> {
        let Some(user) = credentials.basic_user else {
            return verdict;
        };
        match verdict {
            Ok(mut caller) if caller.principal.name == user => {
                // A client certificate named the principal: it stays an `mtls` caller (its
                // session key does not change with the header it sends).
                if caller.method != Method::Mtls {
                    caller.method = Method::Basic;
                }
                Ok(caller)
            }
            Ok(_) => {
                self.record_failure(credentials, now);
                Err(self.fail(credentials, Method::Basic, AuthFailure::Invalid))
            }
            Err(failure) => Err(failure),
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
                    Ok(Some(id)) => Ok(id.act.is_none() && id.principal == name),
                    Ok(None) => Err(AuthFailure::Unavailable),
                    Err(AuthFailure::Invalid) => Ok(false),
                    Err(f) => Err(f),
                }
            }
            (Some(t), _) if t.len() <= MAX_TOKEN_BYTES => Ok(self
                .tokens
                .lookup_at(t, std::time::SystemTime::now())
                .is_some_and(|(p, _, expired)| !expired && p.name == name)),
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

    /// `caller` (authenticated by this server, on `surface` from `peer`) asking to act as
    /// `act_as` (design P3-T061 §4.2): the end user's result, or the caller itself when
    /// `act_as` names only the caller.
    ///
    /// Every decision on someone else is one `krishiv::audit` record (caller, end user, a
    /// digest of the groups, tenant, decision, reason, the token's `jti`, peer; never a
    /// credential) and one count.
    ///
    /// # Errors
    /// `KRISHIV_AUTH_IMPERSONATION_DENIED` (42501) for every refusal; the reason is in the
    /// audit record only.
    pub fn impersonate(
        &self,
        caller: &Authenticated,
        surface: Surface,
        peer: std::net::SocketAddr,
        act_as: &ActAs,
    ) -> Result<Authenticated, KrishivError> {
        let decision = impersonation::decide(
            &self.principals,
            &self.impersonation,
            caller,
            Channel::Plain(peer),
            act_as,
            std::time::SystemTime::now(),
        );
        self.audit(caller, surface, peer, Some(act_as), None, decision)
    }

    /// `caller` (authenticated by this server, on `surface` from `peer`) presenting the signed
    /// act-as assertion `assertion` (`x-krishiv-act-as-token`, design P3-T061 §4.1, unit I3):
    /// the [`ActAs`] it names and the end user's result.
    ///
    /// The assertion must verify against `krishiv.auth.impersonation.assertion.*` (signature,
    /// `iss`, `aud`, dates, lifetime), carry `act.sub` equal to the caller's name and no `act`
    /// inside `act`; then the rules of [`ServerAuth::impersonate`] apply as for a trusted
    /// channel, from any peer. A key not cached yet waits for one JWK set fetch. The returned
    /// [`ActAs`] is what the transport pins for the session (rule 5): a later call may bring a
    /// fresh assertion, which must name the same end user, groups and tenant. `beside` is the
    /// plain act-as metadata the transport received with the assertion: a part that disagrees
    /// with it is refused (`MetadataMismatch`), audited once with the assertion's `jti`.
    ///
    /// # Errors
    /// `KRISHIV_AUTH_IMPERSONATION_DENIED` (42501) for every refusal, `signedAssertion` off
    /// included; the reason is in the audit record only.
    pub async fn impersonate_signed(
        &self,
        caller: &Authenticated,
        surface: Surface,
        peer: std::net::SocketAddr,
        assertion: &str,
        beside: &ActAsHint,
    ) -> Result<(ActAs, Authenticated), KrishivError> {
        // The caller's own standing first (review m3): a caller that may not impersonate never
        // makes the server verify an assertion or fetch its keys.
        let checked = match impersonation::may_impersonate(
            &self.principals,
            caller,
            std::time::SystemTime::now(),
        ) {
            Ok(()) => self.verify_assertion(caller, assertion).await,
            Err(denial) => Err(denial),
        };
        let (act_as, assertion_id, subject) = match checked {
            Ok(found) => found,
            Err(denial) => {
                return Err(self.refuse(caller, surface, peer, denial));
            }
        };
        // One decision, one audit record (review M3): a disagreeing hint is that decision.
        let decision = if beside.disagrees_with(&act_as, &caller.principal) {
            Decision::Denied(Denial::MetadataMismatch)
        } else {
            impersonation::decide(
                &self.principals,
                &self.impersonation,
                caller,
                Channel::Signed,
                &act_as,
                std::time::SystemTime::now(),
            )
        };
        let end_user = self.audit(
            caller,
            surface,
            peer,
            Some(&act_as),
            assertion_id.as_deref(),
            decision,
        )?;
        // The assertion is the end user's subject token for delegated catalogs (P3-T067 §3.3),
        // marked as an assertion so it is never forwarded. Kept only for someone else: acting
        // as oneself leaves the caller's own credential.
        let end_user = if end_user.principal.delegated_by.is_some() {
            end_user.with_credential(subject)
        } else {
            end_user
        };
        Ok((act_as, end_user))
    }

    /// Audits and counts a refusal the transport makes itself (act-as metadata it cannot read,
    /// or plain metadata that disagrees with the assertion beside it): one `krishiv::audit`
    /// record and one denied count, as [`ServerAuth::impersonate`]'s refusals.
    #[must_use]
    pub fn refuse_impersonation(
        &self,
        caller: &Authenticated,
        surface: Surface,
        peer: std::net::SocketAddr,
        denial: Denial,
    ) -> KrishivError {
        self.refuse(caller, surface, peer, denial)
    }

    /// Audits and counts a refusal that came before any [`ActAs`] was known; the error.
    fn refuse(
        &self,
        caller: &Authenticated,
        surface: Surface,
        peer: std::net::SocketAddr,
        denial: Denial,
    ) -> KrishivError {
        match self.audit(caller, surface, peer, None, None, Decision::Denied(denial)) {
            Err(e) => e,
            Ok(_) => denial.error(),
        }
    }

    /// The [`ActAs`] a signed assertion names for `caller`, and its `jti`.
    #[cfg(feature = "oidc")]
    async fn verify_assertion(
        &self,
        caller: &Authenticated,
        assertion: &str,
    ) -> Result<(ActAs, Option<String>, Option<super::InboundCredential>), Denial> {
        let Some(verifier) = &self.assertions else {
            return Err(Denial::AssertionOff);
        };
        let verdict = match verifier.verify(assertion, Instant::now(), unix_now(), true) {
            // No cached key fits: wait for one fetch, then decide on what it brought.
            Verified::Refresh => {
                drop(verifier.refresh().await);
                verifier.verify(assertion, Instant::now(), unix_now(), false)
            }
            v => v,
        };
        let identity = match verdict {
            Verified::Ok(identity) => identity,
            Verified::Failed(AuthFailure::Unavailable) | Verified::Refresh => {
                return Err(Denial::AssertionUnavailable);
            }
            Verified::Failed(_) => return Err(Denial::BadAssertion),
        };
        let Some(actor) = identity.act else {
            return Err(Denial::BadAssertion);
        };
        if actor.chained {
            return Err(Denial::Chained);
        }
        if actor.subject.as_deref() != Some(caller.principal.name.as_str()) {
            return Err(Denial::ActorMismatch);
        }
        let subject = super::InboundCredential::from_exp(
            super::CredentialKind::SignedAssertion,
            assertion,
            identity.claims,
            identity.exp,
        );
        Ok((
            ActAs::new(identity.principal).with_groups(identity.groups),
            identity.token_id,
            Some(subject),
        ))
    }

    /// Without the `oidc` feature no assertion can be verified.
    #[cfg(not(feature = "oidc"))]
    #[expect(
        clippy::unused_async,
        reason = "the same signature as with the `oidc` feature"
    )]
    async fn verify_assertion(
        &self,
        _caller: &Authenticated,
        _assertion: &str,
    ) -> Result<(ActAs, Option<String>, Option<super::InboundCredential>), Denial> {
        Err(Denial::AssertionOff)
    }

    /// Writes the audit record of `decision` and counts it; the end user, the caller itself
    /// ([`Decision::Itself`], neither recorded nor counted) or the refusal.
    fn audit(
        &self,
        caller: &Authenticated,
        surface: Surface,
        peer: std::net::SocketAddr,
        act_as: Option<&ActAs>,
        assertion_id: Option<&str>,
        decision: Decision,
    ) -> Result<Authenticated, KrishivError> {
        let (outcome, reason) = match &decision {
            Decision::Itself => return Ok(caller.clone()),
            Decision::Allowed(_) => ("allowed", None),
            Decision::Denied(d) => ("denied", Some(d.as_str())),
        };
        self.impersonations[usize::from(reason.is_some())].fetch_add(1, Ordering::Relaxed);
        tracing::info!(
            target: "krishiv::audit",
            listener = surface.as_str(),
            caller = %caller.principal.name,
            // Caller-supplied names are written only once they are known to be well formed
            // (review m2): no control characters, bounded length.
            end_user = act_as
                .filter(|a| a.is_well_formed())
                .map(|a| tracing::field::display(&a.user)),
            malformed = act_as.is_some_and(|a| !a.is_well_formed()),
            groups_hash = act_as.map(|a| impersonation::groups_hash(&a.groups)),
            tenant = caller.principal.tenant.as_deref(),
            decision = outcome,
            reason_code = reason,
            jti = caller.token_id.as_deref(),
            assertion_jti = assertion_id,
            remote_addr = %peer,
            "impersonation"
        );
        match decision {
            Decision::Allowed(end_user) => Ok(end_user),
            Decision::Denied(d) => Err(d.error()),
            Decision::Itself => Ok(caller.clone()),
        }
    }

    /// The impersonation counters so far.
    #[must_use]
    pub fn impersonation_counts(&self) -> ImpersonationCount {
        ImpersonationCount {
            allowed: self.impersonations[0].load(Ordering::Relaxed),
            denied: self.impersonations[1].load(Ordering::Relaxed),
        }
    }

    /// Logs and counts `failure`, and returns it. A `Basic` credential's failures are counted
    /// under `basic` (a throttled peer's under `throttle`).
    fn fail(&self, c: &Credentials<'_>, method: Method, failure: AuthFailure) -> AuthFailure {
        let method = if c.basic_user.is_some() && method != Method::Throttle {
            Method::Basic
        } else {
            method
        };
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
            if let Some(refused) = self.basic_off_flight(credentials) {
                return Err(refused);
            }
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
                let now = Instant::now();
                let verdict = self.finish_oidc(credentials, token, verdict, now);
                return self.basic_user_agrees(credentials, verdict, now);
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
        token: &str,
        verdict: Result<Option<Identity>, AuthFailure>,
        now: Instant,
    ) -> Result<Authenticated, AuthFailure> {
        match verdict {
            // A delegated token (RFC 8693 `act`) is refused as a login (review M4): acting for
            // someone goes through `impersonate_signed`, with its privilege, allowlist and audit.
            Ok(Some(identity)) if identity.act.is_some() => {
                tracing::debug!(target: "krishiv::auth", reason = "delegated token", "OIDC token refused");
                self.record_failure(credentials, now);
                Err(self.fail(credentials, Method::Oidc, AuthFailure::Invalid))
            }
            Ok(Some(identity)) => Ok(self.oidc_principal(identity, token)),
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
    fn oidc_principal(&self, identity: Identity, token: &str) -> Authenticated {
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
        // The token itself is kept for delegated catalogs (P3-T067 §3.3), secret-typed.
        let credential = super::InboundCredential::from_exp(
            super::CredentialKind::Oidc,
            token,
            identity.claims,
            identity.exp,
        );
        Authenticated::new(principal, role, Method::Oidc)
            .with_token_id(identity.token_id)
            .with_credential(Some(credential))
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
