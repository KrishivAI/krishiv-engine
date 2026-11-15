// SPDX-License-Identifier: Apache-2.0
//! Where a request's bearer token comes from (T067 §3.2, §3.4): one source per catalog, asked once per request, so
//! there is no per-caller client and no token outlives what its source allows. P3 has the two server-identity
//! forms, a static token and the OAuth2 client-credentials grant; the delegated forms (exchange, forward) join here
//! with T067 (PR-58b).
//!
//! A client-credentials token is fetched from the declared token endpoint only (never one a server's `/v1/config`
//! names), reused until a tenth of its lifetime (at most a minute) is left, and dropped when the catalog answers
//! 401, so the next request fetches a new one.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use krishiv_common::secret::SecretString;
use url::Url;

use super::config::{DEFAULT_SCOPE, RestAuth};
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
        cached: Mutex<Option<Cached>>,
    },
}

fn refresh_margin(lifetime: Duration) -> Duration {
    (lifetime / 10).min(Duration::from_secs(60))
}

impl Tokens {
    /// The source for `auth`; `base` is the catalog URI, for the default token endpoint `v1/oauth/tokens`.
    pub(crate) fn new(auth: &RestAuth, base: &Url) -> Result<Self, RestError> {
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
                cached: Mutex::new(None),
            },
        })
    }

    /// The token for the next request.
    pub(crate) fn token(
        &self,
        http: &Transport,
        deadline: Instant,
    ) -> Result<Option<SecretString>, RestError> {
        match self {
            Self::None => Ok(None),
            Self::Static(t) => Ok(Some(t.clone())),
            Self::ClientCredentials {
                client_id,
                client_secret,
                endpoint,
                scope,
                extra,
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
                    headers: Vec::new(),
                };
                let mut response = http.call(request, deadline)?;
                let result = if response.status == 200 {
                    types::parse(&response.body).and_then(types::token_response)
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
                let refresh_at = token.expires_in.map(|s| {
                    let lifetime = Duration::from_secs(s);
                    fetched + lifetime.saturating_sub(refresh_margin(lifetime))
                });
                *lock(cached) = Some(Cached {
                    token: token.access_token.clone(),
                    refresh_at,
                });
                Ok(Some(token.access_token))
            }
        }
    }

    /// Whether a 401 is worth one retry with a new token.
    pub(crate) fn refreshable(&self) -> bool {
        matches!(self, Self::ClientCredentials { .. })
    }

    /// Drops a cached token the catalog refused.
    pub(crate) fn invalidate(&self) {
        if let Self::ClientCredentials { cached, .. } = self {
            *lock(cached) = None;
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
    fn the_default_token_endpoint_is_under_the_catalog_uri() {
        let base = Url::parse("https://h/iceberg").unwrap();
        let t = Tokens::new(
            &RestAuth::ClientCredentials {
                client_id: Some("id".into()),
                client_secret: SecretString::new("s"),
                token_endpoint: None,
                scope: None,
                extra: Vec::new(),
            },
            &base,
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
