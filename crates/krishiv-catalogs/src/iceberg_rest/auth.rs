// SPDX-License-Identifier: Apache-2.0
//! Where a request's bearer token comes from (T067 §3.2, §3.4): one source per catalog, asked once per request, so
//! there is no per-caller client and no token outlives what its source allows. The server-identity forms are a
//! static token and the OAuth2 client-credentials grant; the delegated forms (exchange, forward) are the
//! `delegated` module's, the one source that reads the caller.
//!
//! A client-credentials token is fetched from the declared token endpoint only (never one a server's `/v1/config`
//! names), reused until a tenth of its lifetime (at most a minute) is left, and dropped when the catalog answers
//! 401, so the next request fetches a new one.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use krishiv_common::secret::SecretString;
use url::Url;

use super::config::{DEFAULT_SCOPE, RestAuth};
use super::delegated::{Delegated, RestCaller};
use super::encode::form_body;
use super::error::RestError;
use super::transport::{Request, Transport};
use super::types;

#[derive(Debug)]
pub(crate) struct Cached {
    token: SecretString,
    /// When to stop using it; `None` when the endpoint gave no lifetime (used until a 401).
    refresh_at: Option<Instant>,
}

/// The token source of one catalog.
#[derive(Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "one value per catalog, built once and never moved on a hot path"
)]
pub(crate) enum Tokens {
    None,
    Static(SecretString),
    ClientCredentials {
        client_id: Option<String>,
        client_secret: SecretString,
        endpoint: Url,
        scope: String,
        extra: Vec<(String, String)>,
        max_depth: usize,
        cached: Mutex<Option<Cached>>,
    },
    /// The end user's identity (T067 §3.3).
    Delegated(Box<Delegated>),
}

/// The longest a client-credentials token is reused, whatever lifetime the endpoint claims: an `expires_in` is
/// untrusted input, and a huge one must neither overflow an `Instant` nor keep a token past a day.
pub(crate) const MAX_TOKEN_LIFETIME: Duration = Duration::from_secs(24 * 3600);

/// When to stop using a token fetched at `fetched` that lives `expires_in` seconds: a tenth of its lifetime (at
/// most a minute) before it ends, the lifetime capped at [`MAX_TOKEN_LIFETIME`].
pub(crate) fn refresh_at(fetched: Instant, expires_in: u64) -> Instant {
    let lifetime = Duration::from_secs(expires_in).min(MAX_TOKEN_LIFETIME);
    // Past the clock's range: use the token for this request only.
    fetched
        .checked_add(lifetime.saturating_sub(refresh_margin(lifetime)))
        .unwrap_or(fetched)
}

fn refresh_margin(lifetime: Duration) -> Duration {
    (lifetime / 10).min(Duration::from_secs(60))
}

impl Tokens {
    /// The source for `auth`; `base` is the catalog URI, for the default token endpoint `v1/oauth/tokens`;
    /// `max_depth` bounds the nesting of a token response.
    pub(crate) fn new(
        catalog: &str,
        auth: &RestAuth,
        base: &Url,
        max_depth: usize,
    ) -> Result<Self, RestError> {
        Ok(match auth {
            RestAuth::None => Self::None,
            RestAuth::Token(t) => Self::Static(t.clone()),
            RestAuth::ClientCredentials {
                client_id,
                client_secret,
                token_endpoint,
                scope,
                extra,
            } => Self::ClientCredentials {
                client_id: client_id.clone(),
                client_secret: client_secret.clone(),
                endpoint: match token_endpoint {
                    Some(e) => e.clone(),
                    None => super::client::join(base, &["v1", "oauth", "tokens"])?,
                },
                scope: scope.clone().unwrap_or_else(|| DEFAULT_SCOPE.to_owned()),
                extra: extra.clone(),
                max_depth,
                cached: Mutex::new(None),
            },
            RestAuth::Delegated {
                forward,
                audience,
                token_endpoint,
                scope,
                actor,
                limits,
            } => {
                let actor = match (actor, token_endpoint) {
                    (Some((client_id, client_secret)), Some(endpoint)) => Some(Self::new(
                        catalog,
                        &RestAuth::ClientCredentials {
                            client_id: Some(client_id.clone()),
                            client_secret: client_secret.clone(),
                            token_endpoint: Some(endpoint.clone()),
                            scope: scope.clone(),
                            extra: Vec::new(),
                        },
                        base,
                        max_depth,
                    )?),
                    _ => None,
                };
                Self::Delegated(Box::new(Delegated::new(
                    catalog,
                    *forward,
                    audience.clone(),
                    token_endpoint.clone(),
                    scope.clone(),
                    actor,
                    max_depth,
                    *limits,
                )))
            }
        })
    }

    /// The token for `caller`'s next request (only the delegated source reads the caller).
    pub(crate) fn token(
        &self,
        caller: &RestCaller,
        http: &Transport,
        deadline: Instant,
    ) -> Result<Option<SecretString>, RestError> {
        match self {
            Self::None => Ok(None),
            Self::Delegated(d) => d.token(caller, http, deadline).map(Some),
            Self::Static(t) => Ok(Some(t.clone())),
            Self::ClientCredentials {
                client_id,
                client_secret,
                endpoint,
                scope,
                extra,
                max_depth,
                cached,
            } => {
                let now = Instant::now();
                if let Some(c) = lock(cached).as_ref()
                    && c.refresh_at.is_none_or(|at| now < at)
                {
                    return Ok(Some(c.token.clone()));
                }
                let mut pairs: Vec<(&str, &str)> = vec![("grant_type", "client_credentials")];
                if let Some(id) = client_id {
                    pairs.push(("client_id", id));
                }
                pairs.push(("client_secret", client_secret.expose_secret()));
                pairs.push(("scope", scope));
                pairs.extend(extra.iter().map(|(k, v)| (k.as_str(), v.as_str())));
                let request = Request {
                    post: true,
                    url: endpoint.clone(),
                    bearer: None,
                    form: Some(SecretString::new(form_body(&pairs))),
                    json: None,
                    headers: Vec::new(),
                };
                let mut response = http.call(request, deadline)?;
                let result = if response.status == 200 {
                    types::parse_limited(&response.body, *max_depth).and_then(types::token_response)
                } else if matches!(response.status, 400 | 401 | 403) {
                    Err(RestError::TokenRefused {
                        status: response.status,
                        error: types::oauth_error(&response.body),
                    })
                } else {
                    Err(RestError::Status {
                        status: response.status,
                        kind: None,
                        message: None,
                    })
                };
                krishiv_common::secret::wipe_bytes(&mut response.body);
                let token = result?;
                let fetched = Instant::now();
                let refresh_at = token.expires_in.map(|s| refresh_at(fetched, s));
                *lock(cached) = Some(Cached {
                    token: token.access_token.clone(),
                    refresh_at,
                });
                Ok(Some(token.access_token))
            }
        }
    }

    /// Sets the deadline of a delegated catalog's actor-token fetch (no effect for other modes).
    pub(crate) fn set_actor_timeout(&mut self, timeout: Duration) {
        if let Self::Delegated(d) = self {
            d.set_actor_timeout(timeout);
        }
    }

    /// Whether the catalog sees end users (no identity of its own at the catalog).
    pub(crate) fn is_delegated(&self) -> bool {
        matches!(self, Self::Delegated(_))
    }

    /// Whether a 401 is worth one retry with a new token.
    pub(crate) fn refreshable(&self) -> bool {
        match self {
            Self::ClientCredentials { .. } => true,
            Self::Delegated(d) => d.refreshable(),
            Self::None | Self::Static(_) => false,
        }
    }

    /// Drops the cached token if it is `refused`, the one the catalog just answered 401 to; a newer token another
    /// caller stored meanwhile stays.
    pub(crate) fn invalidate(&self, refused: &SecretString) {
        if let Self::Delegated(d) = self {
            d.invalidate(refused);
        }
        if let Self::ClientCredentials { cached, .. } = self {
            let mut slot = lock(cached);
            if slot.as_ref().is_some_and(|c| c.token == *refused) {
                *slot = None;
            }
        }
    }
}

fn lock(m: &Mutex<Option<Cached>>) -> std::sync::MutexGuard<'_, Option<Cached>> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_refresh_margin_is_a_tenth_up_to_a_minute() {
        assert_eq!(
            refresh_margin(Duration::from_secs(100)),
            Duration::from_secs(10)
        );
        assert_eq!(
            refresh_margin(Duration::from_secs(3600)),
            Duration::from_secs(60)
        );
        assert_eq!(refresh_margin(Duration::ZERO), Duration::ZERO);
    }

    #[test]
    fn a_huge_expires_in_is_capped_and_never_overflows() {
        let now = Instant::now();
        let day = now + MAX_TOKEN_LIFETIME - Duration::from_secs(60);
        assert_eq!(refresh_at(now, u64::MAX), day);
        assert_eq!(refresh_at(now, (i64::MAX as u64) + 1), day);
        assert_eq!(refresh_at(now, 100), now + Duration::from_secs(90));
    }

    #[test]
    fn invalidate_drops_only_the_refused_token() {
        let base = Url::parse("https://h/").unwrap();
        let t = Tokens::new(
            "rest",
            &RestAuth::ClientCredentials {
                client_id: None,
                client_secret: SecretString::new("s"),
                token_endpoint: None,
                scope: None,
                extra: Vec::new(),
            },
            &base,
            64,
        )
        .unwrap();
        let Tokens::ClientCredentials { cached, .. } = &t else {
            panic!("client credentials")
        };
        *lock(cached) = Some(Cached {
            token: SecretString::new("new"),
            refresh_at: None,
        });
        t.invalidate(&SecretString::new("old"));
        assert!(lock(cached).is_some());
        t.invalidate(&SecretString::new("new"));
        assert!(lock(cached).is_none());
    }

    #[test]
    fn the_default_token_endpoint_is_under_the_catalog_uri() {
        let base = Url::parse("https://h/iceberg").unwrap();
        let t = Tokens::new(
            "rest",
            &RestAuth::ClientCredentials {
                client_id: Some("id".into()),
                client_secret: SecretString::new("s"),
                token_endpoint: None,
                scope: None,
                extra: Vec::new(),
            },
            &base,
            64,
        )
        .unwrap();
        let Tokens::ClientCredentials {
            endpoint, scope, ..
        } = t
        else {
            panic!("client credentials")
        };
        assert_eq!(endpoint.as_str(), "https://h/iceberg/v1/oauth/tokens");
        assert_eq!(scope, "catalog");
    }
}
