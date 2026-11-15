// SPDX-License-Identifier: Apache-2.0
//! The `grants` access-policy provider (P3-T062 unit G2; design `P3-T062-access-policy-providers.md`
//! §2, ADR 0046 §7, ADR 0036 R2).
//!
//! [`Grants`] is the server-wide part: the [`GrantStore`] (roles, grant rows, owners, the epoch)
//! and the denial counters. [`GrantsPolicy`] is one session's policy over it, composed as ADR 0046
//! §7 says: data securables (tables, views, columns, namespaces, functions, paths, secrets,
//! connections) are decided by the grants alone; `System(..)` securables and the reserved
//! `system` catalog's tables by the `system` policy (the role matrix in server mode); an admin
//! passes every data check. Path confinement (`PathPrefixPolicy`) wraps the whole policy, so it
//! is asked first and grants never lift it.
//!
//! Decisions: any grant of the principal's effective grantees (the user, its groups, and their
//! roles transitively) on the securable or a parent allows; nothing else does (no `DENY`, Q2).
//! An owner (the creator, Q4) holds every privilege with the grant option. Reads are decided per
//! column ([`AccessPolicy::column_level`]): `Browse` on a table needs any grant on it (a column
//! grant included), its rows `Select` per referenced column. Each refusal writes one
//! `krishiv::audit` record (`principal`, `securable`, `privilege`, `decision`, `provider`,
//! `policy_epoch`; the query id is the enclosing `krishiv.sql`/`krishiv.query` span's) and counts
//! toward [`Grants::denied_counts`] (`krishiv_access_denied_total{privilege,provider}`).

mod exec;
mod model;
mod store;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use futures::future::BoxFuture;
use krishiv_common::config::{ACCESS_PROVIDER, Config, ConfigValue};
use krishiv_common::error::{KrishivError, KrishivInvalidConfigValue};
use krishiv_plan::analyzer::{
    AccessDecision, AccessPolicy, PolicyVersion, Principal, Privilege, Securable,
};

pub use store::{FileGrantStore, GrantStateHandle, GrantStore, MemoryGrantStore};

use model::ObjectKey;

/// The privileges the counters keep, in label order.
const COUNTED: [Privilege; 10] = [
    Privilege::Select,
    Privilege::Modify,
    Privilege::Create,
    Privilege::Drop,
    Privilege::Monitor,
    Privilege::Usage,
    Privilege::Browse,
    Privilege::Operate,
    Privilege::Manage,
    Privilege::Impersonate,
];

/// Who decided a securable under [`GrantsPolicy`] (the `provider` label).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DecidedBy {
    /// The grants.
    Grants,
    /// The `system` policy (the role matrix).
    Roles,
}

impl DecidedBy {
    /// `grants` or `roles`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Grants => "grants",
            Self::Roles => "roles",
        }
    }
}

/// One denial counter (`krishiv_access_denied_total{privilege,provider}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeniedCount {
    /// The privilege refused.
    pub privilege: Privilege,
    /// Who refused it.
    pub provider: DecidedBy,
    /// Refusals so far.
    pub count: u64,
}

/// The server-wide grants: the store and the denial counters, shared by every session's
/// [`GrantsPolicy`].
pub struct Grants {
    store: Arc<dyn GrantStore>,
    /// Indexed by `provider * COUNTED.len() + privilege`.
    denied: [AtomicU64; 2 * COUNTED.len()],
}

impl std::fmt::Debug for Grants {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Grants")
            .field("epoch", &self.epoch())
            .finish_non_exhaustive()
    }
}

impl Grants {
    /// Grants over `store`.
    #[must_use]
    pub fn new(store: Arc<dyn GrantStore>) -> Arc<Self> {
        Arc::new(Self {
            store,
            denied: std::array::from_fn(|_| AtomicU64::new(0)),
        })
    }

    /// The grants epoch: bumped by every `GRANT`, `REVOKE`, role change and ownership change.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.store.snapshot().0.epoch
    }

    /// The denial counters, every (privilege, provider) pair, zeros included.
    #[must_use]
    pub fn denied_counts(&self) -> Vec<DeniedCount> {
        [DecidedBy::Grants, DecidedBy::Roles]
            .into_iter()
            .enumerate()
            .flat_map(|(p, provider)| {
                COUNTED
                    .iter()
                    .enumerate()
                    .map(move |(i, privilege)| (p, provider, i, *privilege))
            })
            .map(|(p, provider, i, privilege)| DeniedCount {
                privilege,
                provider,
                count: self.denied[p * COUNTED.len() + i].load(Ordering::Relaxed),
            })
            .collect()
    }

    fn count(&self, provider: DecidedBy, privilege: Privilege) {
        let p = match provider {
            DecidedBy::Grants => 0,
            DecidedBy::Roles => 1,
        };
        if let Some(i) = COUNTED.iter().position(|c| *c == privilege) {
            self.denied[p * COUNTED.len() + i].fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn store(&self) -> &dyn GrantStore {
        &*self.store
    }
}

/// One session's policy: grants for data securables, `system` for the rest, admin passes every
/// data check (see the module docs).
pub struct GrantsPolicy {
    grants: Arc<Grants>,
    system: Arc<dyn AccessPolicy>,
    admin: bool,
}

impl std::fmt::Debug for GrantsPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrantsPolicy")
            .field("admin", &self.admin)
            .finish_non_exhaustive()
    }
}

impl GrantsPolicy {
    /// The policy of a session over `grants`, with `system` deciding `System(..)` securables (the
    /// session's `RolePolicy` in server mode) and `admin` for a principal whose role is admin.
    #[must_use]
    pub fn new(grants: Arc<Grants>, system: Arc<dyn AccessPolicy>, admin: bool) -> Self {
        Self {
            grants,
            system,
            admin,
        }
    }

    /// The shared grants.
    #[must_use]
    pub fn grants(&self) -> &Arc<Grants> {
        &self.grants
    }

    /// Whether the principal is an admin (passes every data check, may grant anything).
    #[must_use]
    pub fn is_admin(&self) -> bool {
        self.admin
    }

    /// Who decides `securable`.
    #[must_use]
    pub fn decided_by(securable: &Securable) -> DecidedBy {
        match securable {
            Securable::System(_) => DecidedBy::Roles,
            Securable::Table(t) | Securable::View(t) | Securable::Column { table: t, .. }
                if crate::system::is_system(&t.catalog) =>
            {
                DecidedBy::Roles
            }
            _ => DecidedBy::Grants,
        }
    }

    fn decide(
        &self,
        principal: &Principal,
        securable: &Securable,
        privilege: Privilege,
    ) -> AccessDecision {
        let deny = |reason: &str| AccessDecision::Deny {
            reason: reason.to_owned(),
        };
        // `Impersonate` is never granted, nor implied by admin (ADR 0046 §5).
        if privilege == Privilege::Impersonate {
            return deny("impersonation is not a grant");
        }
        if self.admin {
            return AccessDecision::Allow;
        }
        let Some(key) = ObjectKey::of(securable) else {
            return deny("the securable has no grant key");
        };
        if self
            .grants
            .store
            .snapshot()
            .0
            .allows(principal, &key, privilege)
        {
            AccessDecision::Allow
        } else {
            deny("no grant")
        }
    }
}

/// `version` mixed with `epoch` so that a change of either changes the result (multiplying by
/// an odd constant is a bijection on `u64`).
const fn mix(epoch: u64, version: u64) -> u64 {
    epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ version
}

impl AccessPolicy for GrantsPolicy {
    fn version(&self) -> PolicyVersion {
        PolicyVersion(mix(self.grants.epoch(), self.system.version().0))
    }

    fn check<'a>(
        &'a self,
        principal: &'a Principal,
        securable: &'a Securable,
        privilege: Privilege,
    ) -> BoxFuture<'a, Result<AccessDecision, KrishivError>> {
        match Self::decided_by(securable) {
            DecidedBy::Roles => self.system.check(principal, securable, privilege),
            DecidedBy::Grants => Box::pin(std::future::ready(Ok(
                self.decide(principal, securable, privilege)
            ))),
        }
    }

    fn column_level(&self) -> bool {
        true
    }

    fn denied(&self, principal: &Principal, securable: &Securable, privilege: Privilege) {
        let provider = Self::decided_by(securable);
        if provider == DecidedBy::Roles {
            self.system.denied(principal, securable, privilege);
        }
        self.grants.count(provider, privilege);
        tracing::warn!(
            target: "krishiv::audit",
            principal = %principal.name,
            securable = %securable,
            privilege = %privilege,
            decision = "deny",
            provider = provider.as_str(),
            policy_epoch = self.grants.epoch(),
            "access policy denial"
        );
    }
}

/// The access provider a server configures (`krishiv.access.provider`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AccessProvider {
    /// No provider beyond the role matrix (or `AllowAll` without authentication).
    Roles,
    /// The `grants` provider.
    Grants,
}

/// The provider `config` selects. Unset, it is `grants` when the server authenticates and the
/// role matrix alone otherwise (owner answer Q1: kept here, in one place). `catalog`, `external`
/// and lists (`grants+catalog`) are refused until their units land (G5, G6).
///
/// # Errors
/// `KRISHIV_INVALID_CONFIG_VALUE` naming the key for any other value.
pub fn access_provider(
    config: &Config,
    authenticated: bool,
) -> Result<AccessProvider, KrishivError> {
    let raw = match config.get(&ACCESS_PROVIDER) {
        Some(ConfigValue::String(s)) if !s.trim().is_empty() => s.trim().to_owned(),
        _ => {
            return Ok(if authenticated {
                AccessProvider::Grants
            } else {
                AccessProvider::Roles
            });
        }
    };
    let invalid = |reason: &str| -> KrishivError {
        KrishivInvalidConfigValue {
            value: format!("'{raw}'"),
            key: format!("\"{}\"", ACCESS_PROVIDER.key),
            reason: reason.to_owned(),
        }
        .into()
    };
    match raw.to_ascii_lowercase().as_str() {
        "grants" => Ok(AccessProvider::Grants),
        "roles" => Ok(AccessProvider::Roles),
        "catalog" => Err(invalid(
            "the catalog provider is not available yet (P3-T062 unit G5)",
        )),
        "external" => Err(invalid(
            "the external provider is not available yet (P3-T062 unit G6)",
        )),
        r if r.contains(['+', ',']) => Err(invalid(
            "provider lists are not available yet (P3-T062 units G5 and G6)",
        )),
        _ => Err(invalid("expected grants or roles")),
    }
}

#[cfg(test)]
mod tests;
