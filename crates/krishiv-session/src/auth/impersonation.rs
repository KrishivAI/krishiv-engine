// SPDX-License-Identifier: Apache-2.0
//! Trusted impersonation (design P3-T061 §4, Q1-Q3, amendments A3-A4): an authenticated
//! service principal acting for an end user.
//!
//! ```text
//! krishiv.auth.impersonation.roles.{viewer,operator,admin} = g1, g2  (end users' groups by role; default viewer)
//! ```
//!
//! The transport hands over what the caller asked for as an [`ActAs`] (Spark Connect: the
//! `UserContext.user_id` plus the `x-krishiv-act-as-*` metadata), and [`decide`] gives either
//! the end user's [`Authenticated`] or the [`Denial`] reason. The rules (§4.2):
//!
//! 1. Asking for nothing but one's own name is no impersonation and needs no right. Anything
//!    else needs `impersonate` (`krishiv.auth.principals.<caller>.privileges`): without it the
//!    request is refused, not ignored, so a misconfigured front end shows at once.
//! 2. The end user must match the caller's `impersonates` list.
//! 3. No chaining, no escalation: a caller that is itself acting for someone is refused, and the
//!    end user's role comes from the impersonation role lists by its groups (default viewer),
//!    never from the caller's role. The lists are a namespace of their own (A3): an end user's
//!    group can never be named like a certificate principal or an OIDC role-claim value.
//! 4. The result keeps the caller as `Principal::delegated_by`, for the audit and the system
//!    tables; the tenant is the caller's (an end user cannot be placed in another tenant).
//! 5. Plain act-as is accepted only over a client certificate or from loopback (Q2): a front
//!    end across a network hop must sign its assertion (unit I3).
//!
//! Fixing the act-as identity for a session's life (rule 5 of §4.2) is the transport's: it
//! keys the session by the result, `delegated_by` included (A4), and compares [`ActAs`] on
//! every later call.

use std::net::SocketAddr;

use krishiv_common::error::KrishivError;
use krishiv_common::principal::{Principal, PrincipalKind};

use super::config::RoleLists;
use super::jose::is_format_char;
use super::principals::{PrincipalPrivilege, PrincipalTable};
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

/// `krishiv.auth.impersonation.*`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ImpersonationConfig {
    /// End users' groups by role.
    pub roles: RoleLists,
}

impl ImpersonationConfig {
    /// Applies `krishiv.auth.impersonation.<rest>` = `raw`; whether `rest` was one of these.
    pub(super) fn apply(&mut self, key: &str, rest: &str, raw: &str) -> Result<bool, KrishivError> {
        let Some(role) = rest.strip_prefix("roles.") else {
            return Ok(false);
        };
        let Some(role) = Role::parse(role).filter(|r| r.as_str() == role) else {
            return Ok(false);
        };
        self.roles.set(key, role, raw)?;
        Ok(true)
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

/// Rules 1-5 for `caller` (already authenticated, from `peer`) asking to act as `act_as`.
pub(crate) fn decide(
    principals: &PrincipalTable,
    config: &ImpersonationConfig,
    caller: &Authenticated,
    peer: SocketAddr,
    act_as: &ActAs,
) -> Decision {
    let me = &caller.principal;
    if me.delegated_by.is_some() {
        return Decision::Denied(Denial::Chained);
    }
    if act_as.is_self(me) {
        return Decision::Itself;
    }
    if !principals.has(&me.name, PrincipalPrivilege::Impersonate) {
        return Decision::Denied(Denial::NoPrivilege);
    }
    if !(caller.method == Method::Mtls || peer.ip().is_loopback()) {
        return Decision::Denied(Denial::UntrustedChannel);
    }
    let well_formed = is_name(&act_as.user)
        && act_as.groups.len() <= MAX_GROUPS
        && act_as.groups.iter().all(|g| is_name(g))
        && act_as.tenant.as_deref().is_none_or(is_name);
    if !well_formed {
        return Decision::Denied(Denial::Malformed);
    }
    let tenant = match (&me.tenant, &act_as.tenant) {
        (Some(mine), Some(theirs)) if mine != theirs => {
            return Decision::Denied(Denial::OtherTenant);
        }
        (Some(mine), _) => Some(mine.clone()),
        // A caller outside every tenant names the end user's.
        (None, theirs) => theirs.clone(),
    };
    let allowed = principals
        .get(&me.name)
        .is_some_and(|r| r.may_impersonate(&act_as.user, &act_as.groups));
    if !allowed {
        return Decision::Denied(Denial::NotAllowed);
    }
    let mut principal =
        Principal::new(act_as.user.clone(), PrincipalKind::User).with_delegator(me.name.clone());
    principal.groups.clone_from(&act_as.groups);
    principal.tenant = tenant;
    let role = config.role_of(&act_as.groups);
    Decision::Allowed(
        Authenticated::new(principal, Some(role), caller.method)
            .with_token_id(caller.token_id.clone()),
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
