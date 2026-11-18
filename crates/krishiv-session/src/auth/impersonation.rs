// SPDX-License-Identifier: Apache-2.0
//! Trusted impersonation (design P3-T061 §4, Q1-Q3, amendments A3-A4): an authenticated
//! service principal acting for an end user.
//!
//! ```text
//! krishiv.auth.impersonation.roles.{viewer,operator,admin} = g1, g2  (end users' groups by role; default viewer)
//! krishiv.auth.impersonation.signedAssertion = false                 (accept signed act-as assertions, unit I3)
//! krishiv.auth.impersonation.assertion.issuer = https://portal.example.com/   (their verifier: the
//!     krishiv.auth.oidc.* keys under this prefix, except roles.*; maxLifetime defaults to 5m)
//! ```
//!
//! The transport hands over what the caller asked for as an [`ActAs`] (Spark Connect: the
//! `UserContext.user_id` plus the `x-krishiv-act-as-*` metadata), and [`decide`] gives either
//! the end user's [`Authenticated`] or the [`Denial`] reason. A signed assertion
//! (`x-krishiv-act-as-token`, RFC 8693) is a JWT from the assertion issuer whose `sub` is the
//! end user, whose groups claim holds the end user's groups and whose `act.sub` is the caller;
//! it becomes the same [`ActAs`] (see `ServerAuth::impersonate_signed`). The rules (§4.2):
//!
//! 1. Asking for nothing but one's own name is no impersonation and needs no right. Anything
//!    else needs `impersonate` (`krishiv.auth.principals.<caller>.privileges`) on a caller of
//!    kind service, whose `expires` has not passed: without it the request is refused, not
//!    ignored, so a misconfigured front end shows at once. Only service principals impersonate
//!    (a static token of kind service, or a client certificate not listed in
//!    `mtls.userPrincipals`), so an OIDC user or a user certificate that happens to carry a
//!    service's name never inherits its rights.
//! 2. The end user must match the caller's `impersonates` list.
//! 3. No chaining, no escalation: a caller that is itself acting for someone is refused, and the
//!    end user's role comes from the impersonation role lists by its groups (default viewer),
//!    capped at the caller's own role: the groups are the caller's claim, and a `group:<g>`
//!    allowlist entry trusts that claim as far as the caller is trusted. The lists are a namespace of their own (A3): an end user's
//!    group can never be named like a certificate principal or an OIDC role-claim value.
//! 4. The result keeps the caller as `Principal::delegated_by`, for the audit and the system
//!    tables; the tenant is the caller's (an end user cannot be placed in another tenant).
//! 5. Plain act-as is accepted only over a client certificate or from loopback (Q2): a front
//!    end across a network hop must sign its assertion (unit I3). A signed assertion is valid
//!    from any peer, but only for the caller its `act.sub` names, and never for a chain (an
//!    `act` inside `act`). The caller still needs `impersonate` and the allowlist: the
//!    assertion proves what the front end asked for, not that it may ask.
//!
//! Fixing the act-as identity for a session's life (rule 5 of §4.2) is the transport's: it
//! keys the session by the result, `delegated_by` included (A4), and compares [`ActAs`] on
//! every later call.

use std::net::SocketAddr;
use std::time::SystemTime;

use krishiv_common::error::KrishivError;
use krishiv_common::principal::{Principal, PrincipalKind};

use super::config::RoleLists;
use super::jose::is_format_char;
use super::oidc_config::OidcConfig;
use super::principals::{PrincipalPrivilege, PrincipalTable};
use super::secret_ref::invalid;
use super::{Authenticated, Method, Role};

/// The longest end-user name accepted, in bytes (as an OIDC principal name).
const MAX_NAME_BYTES: usize = 256;
/// Most groups an act-as request may carry.
const MAX_GROUPS: usize = 64;

/// What a caller asked to act as.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ActAs {
    /// The end user's name.
    pub user: String,
    /// The end user's groups.
    pub groups: Vec<String>,
    /// The end user's tenant, if the caller named one.
    pub tenant: Option<String>,
}

impl ActAs {
    /// Acting as `user`, without groups or tenant.
    #[must_use]
    pub fn new(user: impl Into<String>) -> Self {
        Self {
            user: user.into(),
            groups: Vec::new(),
            tenant: None,
        }
    }

    /// This request with the end user's `groups`.
    #[must_use]
    pub fn with_groups(mut self, groups: Vec<String>) -> Self {
        self.groups = groups;
        self
    }

    /// This request with the end user's `tenant`.
    #[must_use]
    pub fn with_tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = Some(tenant.into());
        self
    }

    /// Whether the name, groups and tenant are usable names, and the groups not too many.
    pub(crate) fn is_well_formed(&self) -> bool {
        is_name(&self.user)
            && self.groups.len() <= MAX_GROUPS
            && self.groups.iter().all(|g| is_name(g))
            && self.tenant.as_deref().is_none_or(is_name)
    }

    /// Whether it asks for nothing but `caller` itself (rule 1: no right needed).
    fn is_self(&self, caller: &Principal) -> bool {
        self.user == caller.name
            && self.groups.is_empty()
            && self
                .tenant
                .as_ref()
                .is_none_or(|t| Some(t) == caller.tenant.as_ref())
    }
}

/// Plain act-as metadata a transport received beside a signed assertion (design P3-T061
/// §4.1; the security review's wiring requirement 5): each part that is present must agree
/// with what the assertion names, or [`super::ServerAuth::impersonate_signed`] refuses
/// (`MetadataMismatch`). Groups compare as sets; a tenant compares with the caller's (A10: the
/// end user is always in the caller's tenant).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ActAsHint {
    /// The end user's name, if sent.
    pub user: Option<String>,
    /// The end user's groups, if sent.
    pub groups: Option<Vec<String>>,
    /// The end user's tenant, if sent.
    pub tenant: Option<String>,
}

impl ActAsHint {
    /// Nothing sent beside the assertion.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// With the end user's name.
    #[must_use]
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// With the end user's groups.
    #[must_use]
    pub fn with_groups(mut self, groups: Vec<String>) -> Self {
        self.groups = Some(groups);
        self
    }

    /// With the end user's tenant.
    #[must_use]
    pub fn with_tenant(mut self, tenant: impl Into<String>) -> Self {
        self.tenant = Some(tenant.into());
        self
    }

    /// Whether a part that is present names something other than `act_as` for `caller`.
    pub(crate) fn disagrees_with(&self, act_as: &ActAs, caller: &Principal) -> bool {
        fn set(g: &[String]) -> std::collections::BTreeSet<&String> {
            g.iter().collect()
        }
        self.user.as_ref().is_some_and(|u| *u != act_as.user)
            || self
                .groups
                .as_ref()
                .is_some_and(|g| set(g) != set(&act_as.groups))
            || self.tenant.is_some() && self.tenant != caller.tenant
    }
}

/// `krishiv.auth.impersonation.*`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ImpersonationConfig {
    /// End users' groups by role.
    pub roles: RoleLists,
    /// Signed act-as assertions are accepted (Q2, unit I3).
    pub signed_assertion: bool,
    /// The verifier of signed assertions: `krishiv.auth.impersonation.assertion.*`.
    pub assertion: OidcConfig,
}

impl Default for ImpersonationConfig {
    fn default() -> Self {
        Self {
            roles: RoleLists::default(),
            signed_assertion: false,
            assertion: OidcConfig::assertion(),
        }
    }
}

impl ImpersonationConfig {
    /// Applies `krishiv.auth.impersonation.<rest>` = `raw`; whether `rest` was one of these.
    pub(super) fn apply(&mut self, key: &str, rest: &str, raw: &str) -> Result<bool, KrishivError> {
        if rest == "signedAssertion" {
            self.signed_assertion = match raw.trim().to_ascii_lowercase().as_str() {
                "true" => true,
                "false" => false,
                _ => return Err(invalid(key, raw, "expected true or false")),
            };
            return Ok(true);
        }
        if rest.starts_with("assertion.") {
            return self.assertion.apply(key, raw);
        }
        let Some(role) = rest.strip_prefix("roles.") else {
            return Ok(false);
        };
        let Some(role) = Role::parse(role).filter(|r| r.as_str() == role) else {
            return Ok(false);
        };
        self.roles.set(key, role, raw)?;
        Ok(true)
    }

    /// Startup checks: `signedAssertion=true` needs the assertion issuer and audience, and
    /// assertion settings without `signedAssertion=true` are refused (they would do nothing).
    pub(super) fn validate(&self) -> Result<(), KrishivError> {
        self.assertion.validate()?;
        let key = |k: &str| format!("krishiv.auth.impersonation.{k}");
        match (self.signed_assertion, self.assertion.is_on()) {
            (true, false) => Err(invalid(
                &key("assertion.issuer"),
                "",
                "required with krishiv.auth.impersonation.signedAssertion=true",
            )),
            (false, true) => Err(invalid(
                &key("signedAssertion"),
                "false",
                "krishiv.auth.impersonation.assertion.* is set: set signedAssertion=true to use it, \
                 or remove it",
            )),
            _ => Ok(()),
        }
    }

    /// The end user's role: the highest any of its groups has, else viewer.
    fn role_of(&self, groups: &[String]) -> Role {
        groups
            .iter()
            .filter_map(|g| self.roles.role_of(g))
            .max()
            .unwrap_or(Role::Viewer)
    }
}

/// Why an impersonation was refused: the audit record's `reason_code`. The caller only ever
/// sees `KRISHIV_AUTH_IMPERSONATION_DENIED`, never which rule refused it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Denial {
    /// The caller does not hold `impersonate`.
    NoPrivilege,
    /// The end user is not in the caller's `impersonates` list.
    NotAllowed,
    /// The caller is itself acting for someone.
    Chained,
    /// Plain act-as from a peer that is neither loopback nor client-certificate authenticated.
    UntrustedChannel,
    /// A name, group or tenant that is empty, too long or holds control characters, or too
    /// many groups.
    Malformed,
    /// An end user in a tenant other than the caller's.
    OtherTenant,
    /// A signed assertion while `signedAssertion` is off.
    AssertionOff,
    /// A signed assertion that does not verify: signature, issuer, audience, dates, lifetime,
    /// or no `act`.
    BadAssertion,
    /// A signed assertion whose key could not be fetched; the client may retry.
    AssertionUnavailable,
    /// A signed assertion whose `act.sub` is not the caller.
    ActorMismatch,
    /// The caller's `krishiv.auth.principals.<name>.expires` has passed.
    Expired,
    /// Plain act-as metadata beside a signed assertion that names someone else, other groups
    /// or another tenant (the transport's check, design P3-T061 §4.1).
    MetadataMismatch,
}

impl Denial {
    /// The `reason_code` value.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoPrivilege => "no_privilege",
            Self::NotAllowed => "not_allowed",
            Self::Chained => "chained",
            Self::UntrustedChannel => "untrusted_channel",
            Self::Malformed => "malformed",
            Self::OtherTenant => "other_tenant",
            Self::AssertionOff => "assertion_off",
            Self::BadAssertion => "bad_assertion",
            Self::AssertionUnavailable => "assertion_unavailable",
            Self::ActorMismatch => "actor_mismatch",
            Self::Expired => "expired",
            Self::MetadataMismatch => "metadata_mismatch",
        }
    }

    /// The error the caller gets: one class for every reason (42501).
    #[must_use]
    pub fn error(self) -> KrishivError {
        krishiv_common::error::KrishivAuthImpersonationDenied {}.into()
    }
}

/// The outcome of [`decide`] for a request that asked for someone else.
pub(crate) enum Decision {
    /// The request named only the caller: the caller, unchanged, and nothing to audit.
    Itself,
    /// The end user, acting through the caller.
    Allowed(Authenticated),
    /// Refused.
    Denied(Denial),
}

/// How the act-as request reached the server (rule 5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Channel {
    /// Plain act-as from this peer: trusted only from loopback or with a client certificate.
    Plain(SocketAddr),
    /// A signed assertion that verified and names the caller in `act.sub`.
    Signed,
}

/// Whether `caller` may ask to act for anyone at `now`: it acts for no one itself, is a
/// service principal (review M1: a user named like a service, from OIDC or a user
/// certificate, never inherits its rights), holds `impersonate` and its
/// `krishiv.auth.principals.<name>.expires` has not passed (review M3).
pub(crate) fn may_impersonate(
    principals: &PrincipalTable,
    caller: &Authenticated,
    now: SystemTime,
) -> Result<(), Denial> {
    let me = &caller.principal;
    if me.delegated_by.is_some() {
        return Err(Denial::Chained);
    }
    let rights = principals
        .get(&me.name)
        .filter(|r| r.has(PrincipalPrivilege::Impersonate) && me.kind == PrincipalKind::Service)
        .ok_or(Denial::NoPrivilege)?;
    if rights.expired_at(now) {
        return Err(Denial::Expired);
    }
    Ok(())
}

/// Rules 1-5 for `caller` (already authenticated) asking, over `channel`, to act as `act_as`
/// at `now`.
pub(crate) fn decide(
    principals: &PrincipalTable,
    config: &ImpersonationConfig,
    caller: &Authenticated,
    channel: Channel,
    act_as: &ActAs,
    now: SystemTime,
) -> Decision {
    let me = &caller.principal;
    if me.delegated_by.is_some() {
        return Decision::Denied(Denial::Chained);
    }
    if act_as.is_self(me) {
        return Decision::Itself;
    }
    if let Err(denial) = may_impersonate(principals, caller, now) {
        return Decision::Denied(denial);
    }
    let trusted = match channel {
        Channel::Plain(peer) => caller.method == Method::Mtls || peer.ip().is_loopback(),
        Channel::Signed => true,
    };
    if !trusted {
        return Decision::Denied(Denial::UntrustedChannel);
    }
    if !act_as.is_well_formed() {
        return Decision::Denied(Denial::Malformed);
    }
    // The end user is in the caller's tenant; a caller outside every tenant cannot place it in
    // one (review m1: the allowlist names users without tenants, so it cannot say which).
    if act_as.tenant.is_some() && act_as.tenant != me.tenant {
        return Decision::Denied(Denial::OtherTenant);
    }
    let allowed = principals
        .get(&me.name)
        .is_some_and(|r| r.may_impersonate(&act_as.user, &act_as.groups));
    if !allowed {
        return Decision::Denied(Denial::NotAllowed);
    }
    let mut principal =
        Principal::new(act_as.user.clone(), PrincipalKind::User).with_delegator(me.name.clone());
    principal.groups.clone_from(&act_as.groups);
    principal.tenant.clone_from(&me.tenant);
    // The groups are the caller's claim, so the role they give is capped at the caller's own
    // (review M2): impersonating never yields more than the caller has.
    let role = caller
        .role
        .map(|own| own.min(config.role_of(&act_as.groups)));
    Decision::Allowed(
        Authenticated::new(principal, role, caller.method).with_token_id(caller.token_id.clone()),
    )
}

/// A usable name: non-empty, bounded, no control or invisible format characters (it is logged
/// and shown in the system tables).
fn is_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= MAX_NAME_BYTES
        && !s.chars().any(|c| c.is_control() || is_format_char(c))
}

/// A short, stable digest of `groups` (sorted, deduplicated) for the audit record: groups can
/// be sensitive, and the record needs only to tell two sets apart.
pub(crate) fn groups_hash(groups: &[String]) -> String {
    use sha2::{Digest, Sha256};
    let mut sorted: Vec<&str> = groups.iter().map(String::as_str).collect();
    sorted.sort_unstable();
    sorted.dedup();
    let mut h = Sha256::new();
    for g in sorted {
        h.update(g.len().to_le_bytes());
        h.update(g.as_bytes());
    }
    h.finalize()[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
