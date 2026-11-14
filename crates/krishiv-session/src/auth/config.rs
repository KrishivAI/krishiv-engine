// SPDX-License-Identifier: Apache-2.0
//! The `krishiv.auth.*` settings (design §3.2, §3.1 throttling).
//!
//! Static tokens are named, so their keys are a pattern the fixed registry cannot hold:
//!
//! ```text
//! krishiv.auth.tokens.<name>.secret    = env:NAME | file:/path   (ADR 0038 reference; required)
//! krishiv.auth.tokens.<name>.principal = prometheus                (required)
//! krishiv.auth.tokens.<name>.kind      = user | service            (default user)
//! krishiv.auth.tokens.<name>.role      = viewer | operator | admin (required)
//! spark.connect.authenticate.token     = <token> | env:.. | file:.. (Spark's key: one admin token)
//! krishiv.auth.sparkTokenPrincipal     = spark                     (that token's principal)
//! krishiv.auth.failureThrottle.maxFailures = 20                    (per peer address and window)
//! krishiv.auth.failureThrottle.window      = 60s
//! krishiv.auth.failureThrottle.blockFor    = 60s
//! krishiv.auth.clientCertificates      = off | optional | required (default off)
//! krishiv.auth.clientCa                = file:/path               (PEM; with client certificates)
//! krishiv.auth.mtls.principalFrom      = san-uri | san-dns | cn   (default san-uri)
//! krishiv.auth.mtls.userPrincipals     = alice, bob               (certificate principals that are users)
//! krishiv.auth.roles.{viewer,operator,admin} = a, b               (certificate principal names; ADR 0046 §6)
//! krishiv.auth.oidc.*                  (OIDC access tokens; see `oidc_config.rs`)
//! krishiv.auth.principals.<name>.*     (rights by principal name; see `principals.rs`)
//! ```
//!
//! A static token names its own role. A certificate principal gets its role from the
//! `krishiv.auth.roles.*` lists by its name, an OIDC principal from `krishiv.auth.oidc.roles.*`
//! by the values of its role claim; a name in none of them has no role,
//! which denies everything.
//!
//! Every key is startup-only. An unknown `krishiv.auth.*` key is refused, so a misspelt
//! setting never leaves a server open.

use std::collections::BTreeMap;
use std::time::Duration;

use krishiv_common::error::KrishivError;
use krishiv_common::principal::PrincipalKind;

use super::Role;
use super::mtls::MtlsConfig;
use super::oidc_config::OidcConfig;
use super::principals::PrincipalTable;
use super::secret_ref::{SecretRef, invalid};

/// The key prefix of every Krishiv authentication setting.
const PREFIX: &str = "krishiv.auth.";
/// Spark's pre-shared token key (`SparkConnectService`, Spark 4.3.0).
pub(crate) const SPARK_TOKEN_KEY: &str = "spark.connect.authenticate.token";
/// The token name the Spark key defines.
pub(crate) const SPARK_TOKEN_NAME: &str = "spark";

/// Whether `key` is an authentication setting [`AuthConfig::from_settings`] reads.
#[must_use]
pub fn is_auth_key(key: &str) -> bool {
    key.starts_with(PREFIX) || key == SPARK_TOKEN_KEY
}

/// One configured static token.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TokenSpec {
    /// The name in `krishiv.auth.tokens.<name>`, or `spark` for Spark's key.
    pub name: String,
    /// Where the secret comes from.
    pub secret: SecretRef,
    /// The principal's name.
    pub principal: String,
    /// User or service principal.
    pub kind: PrincipalKind,
    /// The role.
    pub role: Role,
}

/// Failure throttling (threat model T-A4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ThrottleConfig {
    /// Failures from one peer address within `window` that start a block; 0 turns it off.
    pub max_failures: u32,
    /// The counting window.
    pub window: Duration,
    /// How long a blocked peer is refused.
    pub block_for: Duration,
}

impl Default for ThrottleConfig {
    fn default() -> Self {
        Self {
            max_failures: 20,
            window: Duration::from_mins(1),
            block_for: Duration::from_mins(1),
        }
    }
}

/// The authentication settings of a server.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct AuthConfig {
    /// The static tokens, by name.
    pub tokens: Vec<TokenSpec>,
    /// Failure throttling.
    pub throttle: ThrottleConfig,
    /// Client certificates.
    pub mtls: MtlsConfig,
    /// The roles of principals that do not carry one (certificates, OIDC groups).
    pub roles: RoleLists,
    /// OIDC access tokens.
    pub oidc: OidcConfig,
    /// The rights of named principals beyond their role (P3-T061 §3).
    pub principals: PrincipalTable,
}

/// `krishiv.auth.roles.{viewer,operator,admin}`: principal names by role.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RoleLists {
    by_name: BTreeMap<String, Role>,
}

impl RoleLists {
    /// The role of `name`, if a list names it.
    #[must_use]
    pub fn role_of(&self, name: &str) -> Option<Role> {
        self.by_name.get(name).copied()
    }

    /// Sets the list of `role` from the setting `key = raw` (comma separated names).
    pub(super) fn set(&mut self, key: &str, role: Role, raw: &str) -> Result<(), KrishivError> {
        self.by_name.retain(|_, r| *r != role);
        for name in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            if let Some(other) = self.by_name.get(name)
                && *other != role
            {
                return Err(invalid(
                    key,
                    name,
                    &format!(
                        "`{name}` is already listed as {other}; a principal has exactly one role"
                    ),
                ));
            }
            self.by_name.insert(name.to_owned(), role);
        }
        Ok(())
    }
}

/// The fields of one `krishiv.auth.tokens.<name>` group as given.
#[derive(Default)]
struct TokenFields {
    secret: Option<SecretRef>,
    principal: Option<String>,
    kind: Option<PrincipalKind>,
    role: Option<Role>,
}

impl AuthConfig {
    /// Reads every authentication setting in `settings` (other keys are ignored; see
    /// [`is_auth_key`]). A key given twice keeps its last value.
    ///
    /// # Errors
    /// `KRISHIV_INVALID_CONFIG_VALUE` for an unknown `krishiv.auth.*` key, a bad value, a
    /// token without a secret, principal or role, or a secret that is not a reference.
    pub fn from_settings<'a>(
        settings: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self, KrishivError> {
        let mut groups: BTreeMap<String, TokenFields> = BTreeMap::new();
        let mut spark: Option<SecretRef> = None;
        let mut spark_principal = SPARK_TOKEN_NAME.to_owned();
        let mut throttle = ThrottleConfig::default();
        let mut mtls = MtlsConfig::default();
        let mut oidc = OidcConfig::default();
        let mut principals = PrincipalTable::default();
        let mut role_lists: BTreeMap<Role, (String, String)> = BTreeMap::new();
        for (key, raw) in settings {
            if key == SPARK_TOKEN_KEY {
                spark = Some(SecretRef::parse(key, raw, true)?);
                continue;
            }
            let Some(rest) = key.strip_prefix(PREFIX) else {
                continue;
            };
            if let Some(token) = rest.strip_prefix("tokens.") {
                let Some((name, field)) = token.rsplit_once('.') else {
                    return Err(unknown(key));
                };
                if name.is_empty() || name.contains('.') {
                    return Err(invalid(key, name, "a token name is one non-empty segment"));
                }
                let entry = groups.entry(name.to_owned()).or_default();
                match field {
                    "secret" => entry.secret = Some(SecretRef::parse(key, raw, false)?),
                    "principal" => entry.principal = Some(non_empty(key, raw)?),
                    "kind" => entry.kind = Some(parse_kind(key, raw)?),
                    "role" => entry.role = Some(parse_role(key, raw)?),
                    _ => return Err(unknown(key)),
                }
                continue;
            }
            if let Some(principal) = rest.strip_prefix("principals.") {
                principals.apply(key, principal, raw)?;
                continue;
            }
            if mtls.apply(key, rest, raw)? || oidc.apply(key, raw)? {
                continue;
            }
            if let Some(role) = rest.strip_prefix("roles.") {
                let role = Role::parse(role)
                    .filter(|r| r.as_str() == role)
                    .ok_or_else(|| unknown(key))?;
                role_lists.insert(role, (key.to_owned(), raw.to_owned()));
                continue;
            }
            match rest {
                "sparkTokenPrincipal" => spark_principal = non_empty(key, raw)?,
                "failureThrottle.maxFailures" => {
                    throttle.max_failures = raw
                        .trim()
                        .parse()
                        .map_err(|_| invalid(key, raw, "expected a whole number >= 0"))?;
                }
                "failureThrottle.window" => throttle.window = bounded_duration(key, raw)?,
                "failureThrottle.blockFor" => throttle.block_for = bounded_duration(key, raw)?,
                _ => return Err(unknown(key)),
            }
        }
        let mut tokens = Vec::with_capacity(groups.len() + usize::from(spark.is_some()));
        for (name, f) in groups {
            if name == SPARK_TOKEN_NAME && spark.is_some() {
                return Err(invalid(
                    &format!("{PREFIX}tokens.{name}"),
                    &name,
                    "the name `spark` is taken by spark.connect.authenticate.token",
                ));
            }
            let missing = |field: &str| {
                invalid(
                    &format!("{PREFIX}tokens.{name}.{field}"),
                    "",
                    "required for every configured token",
                )
            };
            tokens.push(TokenSpec {
                secret: f.secret.ok_or_else(|| missing("secret"))?,
                principal: f.principal.ok_or_else(|| missing("principal"))?,
                kind: f.kind.unwrap_or(PrincipalKind::User),
                role: f.role.ok_or_else(|| missing("role"))?,
                name,
            });
        }
        if let Some(secret) = spark {
            // Spark's token is all-powerful; the owner kept that (design §6 question 4).
            tokens.push(TokenSpec {
                name: SPARK_TOKEN_NAME.to_owned(),
                secret,
                principal: spark_principal,
                kind: PrincipalKind::User,
                role: Role::Admin,
            });
        }
        mtls.validate()?;
        oidc.validate()?;
        principals.validate()?;
        // Each list once, whole, so a key given twice keeps its last value.
        let mut roles = RoleLists::default();
        for (role, (key, raw)) in role_lists {
            roles.set(&key, role, &raw)?;
        }
        Ok(Self {
            tokens,
            throttle,
            mtls,
            roles,
            oidc,
            principals,
        })
    }

    /// Whether any authentication method is configured.
    #[must_use]
    pub fn is_configured(&self) -> bool {
        !self.tokens.is_empty() || self.mtls.is_on() || self.oidc.is_on()
    }

    /// Whether a bearer token can authenticate (static tokens or OIDC): such a server needs TLS
    /// off loopback.
    #[must_use]
    pub fn has_bearer_method(&self) -> bool {
        !self.tokens.is_empty() || self.oidc.is_on()
    }
}

fn unknown(key: &str) -> KrishivError {
    invalid(key, key, "not a known authentication setting")
}

fn non_empty(key: &str, raw: &str) -> Result<String, KrishivError> {
    let v = raw.trim();
    if v.is_empty() {
        return Err(invalid(key, raw, "must not be empty"));
    }
    Ok(v.to_owned())
}

fn parse_role(key: &str, raw: &str) -> Result<Role, KrishivError> {
    Role::parse(raw.trim()).ok_or_else(|| invalid(key, raw, "expected viewer, operator or admin"))
}

fn parse_kind(key: &str, raw: &str) -> Result<PrincipalKind, KrishivError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "user" => Ok(PrincipalKind::User),
        "service" => Ok(PrincipalKind::Service),
        _ => Err(invalid(key, raw, "expected user or service")),
    }
}

pub(super) fn parse_duration(key: &str, raw: &str) -> Result<Duration, KrishivError> {
    let s = raw.trim();
    let split = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let (digits, unit) = s.split_at(split);
    let n: u64 = digits
        .parse()
        .map_err(|_| invalid(key, raw, "expected a duration such as 500ms, 60s, 5m or 1h"))?;
    let d = match unit.trim() {
        "ms" => Duration::from_millis(n),
        "s" | "" => Duration::from_secs(n),
        "m" | "min" => Duration::from_secs(n.saturating_mul(60)),
        "h" => Duration::from_secs(n.saturating_mul(3600)),
        _ => {
            return Err(invalid(
                key,
                raw,
                "expected a duration such as 500ms, 60s, 5m or 1h",
            ));
        }
    };
    Ok(d)
}

/// The bounds of the throttle's durations: zero would switch it off silently (use
/// `maxFailures = 0` for that), and a huge value would overflow the clock.
const MIN_THROTTLE: Duration = Duration::from_secs(1);
const MAX_THROTTLE: Duration = Duration::from_hours(24);

fn bounded_duration(key: &str, raw: &str) -> Result<Duration, KrishivError> {
    let d = parse_duration(key, raw)?;
    if !(MIN_THROTTLE..=MAX_THROTTLE).contains(&d) {
        return Err(invalid(key, raw, "must be between 1s and 24h"));
    }
    Ok(d)
}
