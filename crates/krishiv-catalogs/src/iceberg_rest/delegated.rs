// SPDX-License-Identifier: Apache-2.0
//! The delegated token source of one catalog (design P3-T067 §3.3): the end user's identity, never
//! the server's.
//!
//! - **`forward`.** The caller's own token is sent unchanged, and only when its validated `aud`
//!   names the catalog's audience. A signed assertion (audience Krishiv by construction) never is.
//! - **`exchange` (RFC 8693).** The caller's token is the `subject_token`; the actor is the
//!   server's own client-credentials token for this catalog (`actorToken = client-credentials`) or
//!   none. The result is cached per (identity key, fingerprint of the exchange claims): a fresh
//!   token with the same claims reuses it until it expires, a token with other claims (narrower
//!   scope, other groups) gets its own exchange. The cache lives in this catalog's source, so an
//!   exchanged token never reaches another catalog. A cached token is used until its own expiry
//!   even after the subject token expired; a new exchange needs a subject token that has not. A
//!   token issued without `expires_in` lives no longer than the subject token it was exchanged
//!   for (at most a day), and serves a caller only while that caller's own subject token is valid.
//! - **Refusals.** No user token, the wrong kind, an expired subject token or a token endpoint
//!   that refuses the subject token (400/401/403 with any code but the client ones) is
//!   `KRISHIV_CATALOG_DELEGATION_UNAVAILABLE` (the OAuth `error` code at most, never a token or
//!   `error_description`). A refused subject token is remembered for 30 s by a fingerprint of the
//!   whole token, so repeats fail at once without reaching the IdP. A refusal of the server's own
//!   client (`invalid_client`, `unauthorized_client`, or an actor token that cannot be fetched) is
//!   the catalog's failure (`KRISHIV_CATALOG_FAILURE`): nothing the user can fix, never
//!   remembered against the user's token.
//! - **Limits.** `krishiv.catalog.delegation.maxExchangesPerMinute` per (principal, claims) and
//!   `maxDistinctExchangesPerMinute` per principal, over a sliding minute; the principal key has
//!   no groups (kind, tenant, name, `delegated_by`), so varying groups is no way around them. Past
//!   either a call is `KRISHIV_CATALOG_LIMIT_EXCEEDED`. Only a request sent to the IdP counts.
//! - **Audit.** Every exchange attempt writes one `krishiv::audit` record (an actor token that
//!   cannot be fetched included): catalog, principal, `delegated_by`, outcome; never a token. A
//!   failed actor fetch is repeated from memory for 30 s, so no caller drives the server's client
//!   grant at the IdP. The actor token is sent to the token endpoint only, never to the catalog.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use krishiv_common::secret::{SecretString, wipe_bytes};
use url::Url;

use super::auth::{MAX_TOKEN_LIFETIME, Tokens, refresh_at};
use super::encode::form_body;
use super::error::RestError;
use super::transport::{Request, Transport};
use super::types;

/// A user token as the token source sees it (built from the session's inbound credential).
#[derive(Clone)]
#[non_exhaustive]
pub struct SubjectToken {
    /// The token.
    pub token: SecretString,
    /// Whether it is a signed assertion: exchangeable, never forwarded.
    pub assertion: bool,
    /// Its validated `aud` values.
    pub audiences: Vec<String>,
    /// A fingerprint of its exchange claims (scope/scp, aud, sorted groups).
    pub claims: [u8; 16],
    /// Its `exp`.
    pub expires_at: SystemTime,
}

impl SubjectToken {
    /// A subject token.
    #[must_use]
    pub fn new(
        token: SecretString,
        assertion: bool,
        audiences: Vec<String>,
        claims: [u8; 16],
        expires_at: SystemTime,
    ) -> Self {
        Self {
            token,
            assertion,
            audiences,
            claims,
            expires_at,
        }
    }
}

impl fmt::Debug for SubjectToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubjectToken")
            .field("token", &self.token)
            .field("assertion", &self.assertion)
            .field("audiences", &self.audiences)
            .field("expires_at", &self.expires_at)
            .finish_non_exhaustive()
    }
}

/// Who one call is for (design P3-T067 §3.2). Server-identity sources ignore it.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct RestCaller {
    /// The identity key, encoded; `None` is the server's own identity.
    pub identity: Option<String>,
    /// The principal without its groups (kind, tenant, name, `delegated_by`): the key of the
    /// exchange limits and the audit record's principal.
    pub principal: Option<String>,
    /// The service acting for the principal.
    pub delegated_by: Option<String>,
    /// The user token, when the session has one.
    pub subject: Option<SubjectToken>,
    /// How the session authenticated, for the refusal message (`token`, `mtls`, ...).
    pub authenticated_with: String,
}

impl RestCaller {
    /// The server itself.
    #[must_use]
    pub fn server() -> Self {
        Self::default()
    }

    /// A user: `identity` the encoded identity key, `principal` the principal key without groups.
    #[must_use]
    pub fn user(
        identity: String,
        principal: String,
        delegated_by: Option<String>,
        authenticated_with: impl Into<String>,
        subject: Option<SubjectToken>,
    ) -> Self {
        Self {
            identity: Some(identity),
            principal: Some(principal),
            delegated_by,
            subject,
            authenticated_with: authenticated_with.into(),
        }
    }
}

/// The exchange limits (`krishiv.catalog.delegation.*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DelegationLimits {
    /// Exchanges per minute per (principal, claims).
    pub max_exchanges_per_minute: usize,
    /// Distinct claim sets exchanged per minute per principal.
    pub max_distinct_exchanges_per_minute: usize,
}

impl Default for DelegationLimits {
    fn default() -> Self {
        Self {
            max_exchanges_per_minute: 60,
            max_distinct_exchanges_per_minute: 20,
        }
    }
}

const WINDOW: Duration = Duration::from_secs(60);
const REFUSED_FOR: Duration = Duration::from_secs(30);
/// More entries than this in one map and the stale ones are dropped first.
const PRUNE_AT: usize = 4096;

const GRANT: &str = "urn:ietf:params:oauth:grant-type:token-exchange";
const JWT: &str = "urn:ietf:params:oauth:token-type:jwt";
const ACCESS_TOKEN: &str = "urn:ietf:params:oauth:token-type:access_token";

struct Exchanged {
    token: SecretString,
    refresh_at: Instant,
    /// Issued without `expires_in`: its life is the subject token's, so it serves only a caller
    /// whose own subject token is still valid.
    bounded_by_subject: bool,
}

#[derive(Default)]
struct State {
    /// (identity key, claims) -> the exchanged token.
    exchanged: HashMap<(String, [u8; 16]), Exchanged>,
    /// (principal, claims) -> the times of the exchanges of the last minute.
    per_claims: HashMap<(String, [u8; 16]), VecDeque<Instant>>,
    /// principal -> claims -> the last exchange of that claim set.
    distinct: HashMap<String, HashMap<[u8; 16], Instant>>,
    /// Fingerprint of a whole refused subject token -> when it was refused.
    refused: HashMap<[u8; 16], Instant>,
    /// The last refusal of the actor token (400/401/403), and when: repeated for 30 s without
    /// asking the IdP again, so user calls cannot drive the server's client grant (§3.3). A
    /// timeout or a 5xx is not kept: one blip must not refuse everyone.
    actor_failed: Option<(Instant, RestError)>,
}

impl State {
    fn prune(&mut self, now: Instant) {
        let fresh = |at: &Instant| now.saturating_duration_since(*at) < WINDOW;
        if self.per_claims.len() > PRUNE_AT {
            self.per_claims
                .retain(|_, times| times.back().is_some_and(fresh));
        }
        if self.distinct.len() > PRUNE_AT {
            self.distinct.retain(|_, sets| {
                sets.retain(|_, at| fresh(at));
                !sets.is_empty()
            });
        }
        if self.refused.len() > PRUNE_AT {
            self.refused
                .retain(|_, at| now.saturating_duration_since(*at) < REFUSED_FOR);
        }
        if self.exchanged.len() > PRUNE_AT {
            self.exchanged.retain(|_, e| now < e.refresh_at);
        }
    }
}

/// The delegated source of one catalog.
pub(crate) struct Delegated {
    catalog: String,
    forward: bool,
    audience: String,
    endpoint: Option<Url>,
    scope: Option<String>,
    actor: Option<Tokens>,
    max_depth: usize,
    limits: DelegationLimits,
    /// The deadline of one actor-token fetch, its own (not what is left of a caller's call).
    actor_timeout: Duration,
    state: Mutex<State>,
}

impl fmt::Debug for Delegated {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Delegated")
            .field("catalog", &self.catalog)
            .field("forward", &self.forward)
            .field("audience", &self.audience)
            .field("actor", &self.actor.is_some())
            .finish_non_exhaustive()
    }
}

/// An expiry as the message shows it: seconds since the Unix epoch.
fn shown(at: SystemTime) -> String {
    at.duration_since(UNIX_EPOCH)
        .map_or_else(|_| "0".to_owned(), |d| d.as_secs().to_string())
}

impl Delegated {
    #[expect(
        clippy::too_many_arguments,
        reason = "one constructor, the declaration's fields"
    )]
    pub(crate) fn new(
        catalog: &str,
        forward: bool,
        audience: String,
        endpoint: Option<Url>,
        scope: Option<String>,
        actor: Option<Tokens>,
        max_depth: usize,
        limits: DelegationLimits,
    ) -> Self {
        Self {
            catalog: catalog.to_owned(),
            forward,
            audience,
            endpoint,
            scope,
            actor,
            max_depth,
            limits,
            actor_timeout: Duration::from_secs(30),
            state: Mutex::new(State::default()),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Sets the deadline of one actor-token fetch (the catalog's `request_timeout`).
    pub(crate) fn set_actor_timeout(&mut self, timeout: Duration) {
        self.actor_timeout = timeout;
    }

    pub(crate) fn refreshable(&self) -> bool {
        !self.forward
    }

    /// Drops the exchanged token the catalog just refused, so the retry exchanges again.
    pub(crate) fn invalidate(&self, refused: &SecretString) {
        self.state().exchanged.retain(|_, e| e.token != *refused);
    }

    /// The token for `caller`'s next request.
    pub(crate) fn token(
        &self,
        caller: &RestCaller,
        http: &Transport,
        deadline: Instant,
    ) -> Result<SecretString, RestError> {
        let (Some(identity), Some(principal), Some(subject)) =
            (&caller.identity, &caller.principal, &caller.subject)
        else {
            return Err(RestError::Delegation(format!(
                "this session authenticated with `{}`",
                caller.authenticated_with
            )));
        };
        if self.forward {
            if subject.assertion {
                return Err(RestError::Delegation(format!(
                    "this session authenticated with a signed assertion; catalog `{}` uses \
                     forwarding",
                    self.catalog
                )));
            }
            if !subject.audiences.contains(&self.audience) {
                return Err(RestError::Delegation(
                    "the session's token is not issued for this catalog's audience, and a token \
                     is forwarded only to an audience it names"
                        .to_owned(),
                ));
            }
            if SystemTime::now() >= subject.expires_at {
                return Err(expired(subject));
            }
            return Ok(subject.token.clone());
        }
        let key = (identity.clone(), subject.claims);
        let now = Instant::now();
        let subject_valid = SystemTime::now() < subject.expires_at;
        if let Some(e) = self.state().exchanged.get(&key)
            && now < e.refresh_at
            && (subject_valid || !e.bounded_by_subject)
        {
            return Ok(e.token.clone());
        }
        if !subject_valid {
            return Err(expired(subject));
        }
        let whole = subject.token.fingerprint();
        self.check_refused(whole, now)?;
        let audit = |outcome: &str| {
            tracing::info!(
                target: "krishiv::audit",
                catalog = self.catalog.as_str(),
                principal = principal.as_str(),
                delegated_by = caller.delegated_by.as_deref(),
                outcome,
                "catalog token exchange"
            );
        };
        // The actor first: a server whose own client is refused spends no user's budget.
        let actor = match &self.actor {
            Some(a) => {
                if let Some((at, e)) = &self.state().actor_failed
                    && now.saturating_duration_since(*at) < REFUSED_FOR
                {
                    return Err(e.clone());
                }
                // Its own deadline: a caller's nearly spent one must not fail the actor.
                let started = Instant::now();
                let own = started.checked_add(self.actor_timeout).unwrap_or(started);
                match a.token(&RestCaller::server(), http, own) {
                    Ok(t) => t,
                    Err(e) => {
                        audit("failed");
                        if matches!(e, RestError::TokenRefused { .. }) {
                            self.state().actor_failed = Some((Instant::now(), e.clone()));
                        }
                        return Err(e);
                    }
                }
            }
            None => None,
        };
        self.admit(principal, subject.claims, now)?;
        let outcome = self.exchange(subject, actor.as_ref(), http, deadline);
        // The IdP refused the server's own client: drop the cached actor token, so the next
        // exchange fetches a new one (still under the back-off).
        if let (Err(RestError::TokenRefused { .. }), Some(a), Some(token)) =
            (&outcome, &self.actor, &actor)
        {
            a.invalidate(token);
        }
        audit(match &outcome {
            Ok(_) => "exchanged",
            Err(RestError::Delegation(_)) => "refused",
            Err(_) => "failed",
        });
        let mut state = self.state();
        match outcome {
            Ok(token) => {
                let fetched = Instant::now();
                let (refresh, bounded_by_subject) = match token.expires_in {
                    Some(s) => (refresh_at(fetched, s), false),
                    // No lifetime given: no longer than the subject token, at most a day.
                    None => {
                        let left = subject
                            .expires_at
                            .duration_since(SystemTime::now())
                            .unwrap_or(Duration::ZERO)
                            .min(MAX_TOKEN_LIFETIME);
                        (fetched.checked_add(left).unwrap_or(fetched), true)
                    }
                };
                state.exchanged.insert(
                    key,
                    Exchanged {
                        token: token.access_token.clone(),
                        refresh_at: refresh,
                        bounded_by_subject,
                    },
                );
                Ok(token.access_token)
            }
            Err(e) => {
                if matches!(e, RestError::Delegation(_)) {
                    state.refused.insert(whole, Instant::now());
                }
                Err(e)
            }
        }
    }

    /// Refuses a subject token the IdP refused in the last 30 seconds.
    fn check_refused(&self, whole: [u8; 16], now: Instant) -> Result<(), RestError> {
        let mut state = self.state();
        state.prune(now);
        if state
            .refused
            .get(&whole)
            .is_some_and(|at| now.saturating_duration_since(*at) < REFUSED_FOR)
        {
            return Err(RestError::Delegation(
                "the identity provider refused this token in the last 30 seconds; reconnect with \
                 a fresh token"
                    .to_owned(),
            ));
        }
        Ok(())
    }

    /// Counts one exchange request for `principal` with `claims`, or refuses it at a limit.
    fn admit(&self, principal: &str, claims: [u8; 16], now: Instant) -> Result<(), RestError> {
        let mut state = self.state();
        let fresh = |at: &Instant| now.saturating_duration_since(*at) < WINDOW;
        let sets = state.distinct.entry(principal.to_owned()).or_default();
        sets.retain(|_, at| fresh(at));
        if !sets.contains_key(&claims)
            && sets.len() >= self.limits.max_distinct_exchanges_per_minute
        {
            return Err(RestError::DelegationLimit {
                what: "distinct claim sets exchanged in a minute",
                value: self.limits.max_distinct_exchanges_per_minute,
                config: "krishiv.catalog.delegation.maxDistinctExchangesPerMinute",
            });
        }
        let times = state
            .per_claims
            .entry((principal.to_owned(), claims))
            .or_default();
        while times.front().is_some_and(|at| !fresh(at)) {
            times.pop_front();
        }
        if times.len() >= self.limits.max_exchanges_per_minute {
            return Err(RestError::DelegationLimit {
                what: "token exchanges in a minute",
                value: self.limits.max_exchanges_per_minute,
                config: "krishiv.catalog.delegation.maxExchangesPerMinute",
            });
        }
        times.push_back(now);
        state
            .distinct
            .entry(principal.to_owned())
            .or_default()
            .insert(claims, now);
        Ok(())
    }

    /// One RFC 8693 exchange at the declared token endpoint.
    fn exchange(
        &self,
        subject: &SubjectToken,
        actor: Option<&SecretString>,
        http: &Transport,
        deadline: Instant,
    ) -> Result<types::TokenResponse, RestError> {
        let Some(endpoint) = &self.endpoint else {
            return Err(RestError::Protocol(
                "an exchange without a token endpoint".to_owned(),
            ));
        };
        let mut pairs: Vec<(&str, &str)> = vec![
            ("grant_type", GRANT),
            ("subject_token", subject.token.expose_secret()),
            ("subject_token_type", JWT),
        ];
        if let Some(a) = actor {
            pairs.push(("actor_token", a.expose_secret()));
            pairs.push(("actor_token_type", ACCESS_TOKEN));
        }
        pairs.push(("audience", &self.audience));
        if let Some(s) = &self.scope {
            pairs.push(("scope", s));
        }
        let request = Request {
            post: true,
            url: endpoint.clone(),
            bearer: None,
            form: Some(SecretString::new(form_body(&pairs))),
            json: None,
            headers: Vec::new(),
        };
        let mut response = http.call(request, deadline)?;
        let result = match response.status {
            200 => {
                types::parse_limited(&response.body, self.max_depth).and_then(types::token_response)
            }
            400 | 401 | 403 => {
                let code = types::oauth_error(&response.body);
                match code.as_deref() {
                    // The server's own client was refused: the catalog's fault, not the user's.
                    Some("invalid_client" | "unauthorized_client") => {
                        Err(RestError::TokenRefused {
                            status: response.status,
                            error: code,
                        })
                    }
                    _ => Err(RestError::Delegation(format!(
                        "the identity provider refused the token exchange ({})",
                        code.as_deref().unwrap_or("no error code")
                    ))),
                }
            }
            status => Err(RestError::Status {
                status,
                kind: None,
                message: None,
            }),
        };
        wipe_bytes(&mut response.body);
        result
    }
}

fn expired(subject: &SubjectToken) -> RestError {
    RestError::Delegation(format!(
        "the session's token expired at {} (seconds since 1970); reconnect with a fresh token",
        shown(subject.expires_at)
    ))
}
