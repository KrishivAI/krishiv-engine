// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.auth.oidc.*` settings (design §3.3; ADR 0046 §6).
//!
//! ```text
//! krishiv.auth.oidc.issuer              = https://idp.example.com/  (exact match on `iss`; turns OIDC on)
//! krishiv.auth.oidc.audience            = krishiv                   (must be in `aud`; required)
//! krishiv.auth.oidc.jwksUri             = https://..                (default: discovered from the issuer)
//! krishiv.auth.oidc.algorithms          = RS256                     (allowlist of RS256, RS384, RS512, ES256, ES384)
//! krishiv.auth.oidc.principalClaim      = sub
//! krishiv.auth.oidc.roleClaim           = groups                    (its values are matched against the lists below)
//! krishiv.auth.oidc.roles.{viewer,operator,admin} = a, b           (role-claim values by role)
//! krishiv.auth.oidc.clockSkew           = 60s                       (0s to 5m)
//! krishiv.auth.oidc.jwksRefresh         = 10m                       (1m to 24h)
//! krishiv.auth.oidc.allowInsecureIssuer = false                     (http:// issuer and JWKS; tests only)
//! ```
//!
//! The OIDC role lists are separate from `krishiv.auth.roles.*`, which name certificate
//! principals: a group value and a certificate name never share a namespace, so an identity
//! provider group cannot be named like a certificate principal to take its role, nor the reverse.
//! Lists are comma separated, so a value that contains a comma (an LDAP distinguished name such
//! as `CN=admins,OU=..`) cannot be listed: have the provider emit plain group names.
//!
//! A build without the `oidc` feature refuses every one of these keys, so a configuration that
//! asks for OIDC never starts a server that cannot check its tokens.

use std::time::Duration;

use krishiv_common::error::KrishivError;

use super::config::{RoleLists, parse_duration};
use super::jose::Alg;
use super::secret_ref::invalid;

const PREFIX: &str = "krishiv.auth.oidc.";

/// The OIDC settings.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct OidcConfig {
    /// The issuer; OIDC is on when set.
    pub issuer: Option<String>,
    /// The audience every token must name.
    pub audience: Option<String>,
    /// The JWK set location; discovered from the issuer when unset.
    pub jwks_uri: Option<String>,
    /// The algorithms accepted.
    pub algorithms: Vec<Alg>,
    /// The claim naming the principal.
    pub principal_claim: String,
    /// The claim holding the groups the role lists match.
    pub role_claim: String,
    /// `krishiv.auth.oidc.roles.*`: role-claim values by role.
    pub roles: RoleLists,
    /// Allowed clock difference for `exp` and `nbf`.
    pub clock_skew: Duration,
    /// How often the JWK set is fetched again.
    pub jwks_refresh: Duration,
    /// `http://` issuer and JWK set locations are allowed (tests only).
    pub allow_insecure_issuer: bool,
}

impl Default for OidcConfig {
    fn default() -> Self {
        Self {
            issuer: None,
            audience: None,
            jwks_uri: None,
            algorithms: vec![Alg::Rs256],
            principal_claim: "sub".to_owned(),
            role_claim: "groups".to_owned(),
            roles: RoleLists::default(),
            clock_skew: Duration::from_mins(1),
            jwks_refresh: Duration::from_mins(10),
            allow_insecure_issuer: false,
        }
    }
}

impl OidcConfig {
    /// Applies `key = raw` if it is an OIDC setting; returns whether it was.
    pub(crate) fn apply(&mut self, key: &str, raw: &str) -> Result<bool, KrishivError> {
        let Some(rest) = key.strip_prefix(PREFIX) else {
            return Ok(false);
        };
        let text = || -> Result<String, KrishivError> {
            let v = raw.trim();
            if v.is_empty() {
                return Err(invalid(key, raw, "must not be empty"));
            }
            Ok(v.to_owned())
        };
        match rest {
            "issuer" => self.issuer = Some(text()?),
            "audience" => self.audience = Some(text()?),
            "jwksUri" => self.jwks_uri = Some(text()?),
            "algorithms" => {
                let mut algs = Vec::new();
                for name in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    let alg = Alg::parse(name).ok_or_else(|| {
                        invalid(
                            key,
                            raw,
                            "expected a list of RS256, RS384, RS512, ES256, ES384",
                        )
                    })?;
                    if !algs.contains(&alg) {
                        algs.push(alg);
                    }
                }
                if algs.is_empty() {
                    return Err(invalid(key, raw, "at least one algorithm"));
                }
                self.algorithms = algs;
            }
            "principalClaim" => self.principal_claim = text()?,
            "roleClaim" => self.role_claim = text()?,
            "clockSkew" => {
                self.clock_skew = parse_duration(key, raw)?;
                if self.clock_skew > Duration::from_mins(5) {
                    return Err(invalid(key, raw, "must be between 0s and 5m"));
                }
            }
            "jwksRefresh" => {
                self.jwks_refresh = parse_duration(key, raw)?;
                if !(Duration::from_mins(1)..=Duration::from_hours(24)).contains(&self.jwks_refresh)
                {
                    return Err(invalid(key, raw, "must be between 1m and 24h"));
                }
            }
            _ if rest.starts_with("roles.") => {
                let name = rest.strip_prefix("roles.").unwrap_or_default();
                let Some(role) = super::Role::parse(name).filter(|r| r.as_str() == name) else {
                    return Ok(false);
                };
                self.roles.set(key, role, raw)?;
            }
            "allowInsecureIssuer" => {
                self.allow_insecure_issuer = match raw.trim().to_ascii_lowercase().as_str() {
                    "true" => true,
                    "false" => false,
                    _ => return Err(invalid(key, raw, "expected true or false")),
                };
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Whether OIDC is configured.
    #[must_use]
    pub fn is_on(&self) -> bool {
        self.issuer.is_some()
    }

    /// The location the JWK set is fetched from first: `jwksUri`, else the discovery document
    /// `<issuer>/.well-known/openid-configuration`, whose `jwks_uri` is then used.
    #[must_use]
    pub fn discovery_uri(&self) -> Option<String> {
        self.issuer.as_deref().map(|i| {
            format!(
                "{}/.well-known/openid-configuration",
                i.trim_end_matches('/')
            )
        })
    }

    /// Whether `uri` may be fetched: `https://`, or `http://` with `allowInsecureIssuer`; a
    /// non-empty host; no user information (it would be logged) and no fragment.
    #[must_use]
    pub fn uri_allowed(&self, uri: &str) -> bool {
        let lower = uri.to_ascii_lowercase();
        let rest = if let Some(rest) = lower.strip_prefix("https://") {
            rest
        } else if let Some(rest) = lower.strip_prefix("http://")
            && self.allow_insecure_issuer
        {
            rest
        } else {
            return false;
        };
        let authority = rest.split(['/', '?']).next().unwrap_or_default();
        !authority.is_empty() && !authority.contains('@') && !uri.contains('#')
    }

    /// Startup checks: [`OidcConfig::validate_values`], and a build with the `oidc` feature
    /// when any OIDC key is given.
    pub(crate) fn validate(&self) -> Result<(), KrishivError> {
        self.validate_values()?;
        if !cfg!(feature = "oidc") && *self != Self::default() {
            return Err(invalid(
                &format!("{PREFIX}issuer"),
                self.issuer.as_deref().unwrap_or_default(),
                "this build has no OIDC (the `oidc` feature of krishiv-session is off)",
            ));
        }
        Ok(())
    }

    /// The checks of the values themselves: an issuer with any other OIDC key, an audience
    /// with the issuer, and `https://` locations unless `allowInsecureIssuer` (design §3.5's
    /// fourth refusal).
    pub(crate) fn validate_values(&self) -> Result<(), KrishivError> {
        let Some(issuer) = &self.issuer else {
            if *self != Self::default() {
                return Err(invalid(
                    &format!("{PREFIX}issuer"),
                    "",
                    "required when any krishiv.auth.oidc.* setting is given",
                ));
            }
            return Ok(());
        };
        if self.audience.is_none() {
            return Err(invalid(
                &format!("{PREFIX}audience"),
                "",
                "required with krishiv.auth.oidc.issuer",
            ));
        }
        if !self.uri_allowed(issuer) {
            return Err(invalid(
                &format!("{PREFIX}issuer"),
                issuer,
                "must be an https:// URL (http:// only with krishiv.auth.oidc.allowInsecureIssuer, for tests)",
            ));
        }
        if let Some(uri) = &self.jwks_uri
            && !self.uri_allowed(uri)
        {
            return Err(invalid(
                &format!("{PREFIX}jwksUri"),
                uri,
                "must be an https:// URL (http:// only with krishiv.auth.oidc.allowInsecureIssuer, for tests)",
            ));
        }
        Ok(())
    }
}
