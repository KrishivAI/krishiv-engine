// SPDX-License-Identifier: Apache-2.0
//! The `grants` access-policy provider.
//!
//! [`Grants`] is the server-wide part: the [`GrantStore`] (roles, grant rows, owners, the epoch)
//! and the denial counters. [`GrantsPolicy`] is one session's policy over it, composed as follows:
//! data securables (tables, views, columns, namespaces, functions, paths, secrets,
//! connections) are decided by the grants alone; `System(..)` securables and the reserved
//! `system` catalog's tables by the `system` policy (the role matrix in server mode); an admin
//! passes every data check. Path confinement (`PathPrefixPolicy`) wraps the whole policy, so it
//! is asked first and grants never lift it.
//!
//! Decisions: any grant of the principal's effective grantees (the user, its groups, and their
//! roles transitively) on the securable or a parent allows; nothing else does (no `DENY`, Q2).
//! An owner (the creator, Q4) holds every privilege with the grant option. Reads are decided per
//! column ([`AccessPolicy::column_level`]): `Browse` on a table needs any grant on it (a column
//! grant included), its rows `Select` per column the final plan reads. Each refusal is counted
//! here ([`Grants::denied_counts`], `krishiv_access_denied_total{privilege,provider}`, the
//! canonical per-provider counter); the one `krishiv::audit` record is the session's
//! (`access::log_denial`, with `provider` and `policy_epoch`; the query id is the enclosing
//! `krishiv.sql`/`krishiv.query` span's).
//!
//! Names match as the catalog resolves them (review M1): keys keep the catalog's spelling and
//! are never folded; a key is compared ignoring ASCII case only when the session catalog holds
//! no case twin of it (`GrantsPolicy::exact_for`). A user and a service principal of one name
//! share grants and ownership (review m8: principals are keyed by name).

mod exec;
mod index;
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
    /// Serialises the statements that create, drop or rename objects with the ownership records
    /// they change (review B2, B3, m4): an object's existence is read before and after the
    /// change under it, so only the statement that created an object owns it.
    ddl: tokio::sync::Mutex<()>,
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
            ddl: tokio::sync::Mutex::new(()),
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

    pub(crate) async fn ddl(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.ddl.lock().await
    }
}

/// One session's policy: grants for data securables, `system` for the rest, admin passes every
/// data check (see the module docs).
pub struct GrantsPolicy {
    grants: Arc<Grants>,
    system: Arc<dyn AccessPolicy>,
    admin: bool,
    /// The runtime's session catalog, to tell case twins apart and to find the managed table a
    /// warehouse directory belongs to (set by the session that uses the policy; without it
    /// names match exactly and no directory belongs to a table: the stricter rules).
    catalog: std::sync::OnceLock<Arc<crate::store::InMemoryCatalogStore>>,
    /// The catalog's lookup tables, rebuilt when it changes.
    index: index::IndexCache,
    /// Keys this session's running statement is creating (the new object; a managed location
    /// the statement makes for it): allowed until it ends (review open point 2).
    creating: Arc<std::sync::Mutex<Vec<ObjectKey>>>,
}

/// Clears [`GrantsPolicy::creating`] when the creating statement ends (one runs at a time: the
/// grants' DDL lock).
pub(crate) struct Creating {
    keys: Arc<std::sync::Mutex<Vec<ObjectKey>>>,
}

impl Drop for Creating {
    fn drop(&mut self) {
        if let Ok(mut keys) = self.keys.lock() {
            keys.clear();
        }
    }
}

impl std::fmt::Debug for GrantsPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrantsPolicy")
            .field("admin", &self.admin)
            .finish_non_exhaustive()
    }
}

impl GrantsPolicy {
    /// The policy of a session over `grants`.
    ///
    /// The policy of a session over `grants`, with `system` deciding `System(..)` securables (the
    /// session's `RolePolicy` in server mode) and `admin` for a principal whose role is admin.
    #[must_use]
    pub fn new(grants: Arc<Grants>, system: Arc<dyn AccessPolicy>, admin: bool) -> Self {
        Self {
            grants,
            system,
            admin,
            catalog: std::sync::OnceLock::new(),
            index: index::IndexCache::default(),
            creating: Arc::default(),
        }
    }

    /// Allows `key` to this session until the guard drops: the object (or its managed location)
    /// the running statement creates.
    pub(crate) fn creating(&self, key: ObjectKey) -> Creating {
        self.also_creating(key);
        Creating {
            keys: Arc::clone(&self.creating),
        }
    }

    /// Adds `key` to what the running statement creates (cleared with its [`Creating`]).
    pub(crate) fn also_creating(&self, key: ObjectKey) {
        if let Ok(mut keys) = self.creating.lock() {
            keys.push(key);
        }
    }

    /// Whether the running statement is creating the relation `key`.
    pub(crate) fn is_creating(&self, key: &ObjectKey) -> bool {
        let exact = self.exact_for(key);
        self.creating
            .lock()
            .is_ok_and(|keys| keys.iter().any(|k| k.same(key, exact)))
    }

    /// Ties the policy to the runtime's session catalog.
    pub(crate) fn bind_catalog(&self, store: &Arc<crate::store::InMemoryCatalogStore>) {
        let _ = self.catalog.set(Arc::clone(store));
    }

    /// Whether `key` matches grant keys exactly (review M1 re-check): names match as the
    /// catalog resolves them, so a key is compared ignoring ASCII case only when no case twin
    /// of it, or of a parent, exists in the session catalog (then every spelling names the one
    /// object, or none yet). Any other catalog, a function's twin unknown, or no catalog bound:
    /// exactly. Never from a session's `spark.sql.caseSensitive`.
    pub(crate) fn exact_for(&self, key: &ObjectKey) -> bool {
        let Some(store) = self.catalog.get() else {
            return true;
        };
        if key.kind == model::KeyKind::Function {
            // Functions resolve case-insensitively in every mode.
            return false;
        }
        if !matches!(
            key.kind,
            model::KeyKind::Namespace | model::KeyKind::Relation | model::KeyKind::Column
        ) || key
            .parts
            .first()
            .is_none_or(|c| c != crate::catalog::CATALOG)
        {
            return true;
        }
        self.index.get(store).has_twin(key)
    }

    /// Whether `principal` may use the local path `key` as the table whose directory holds it
    /// allows (review R2): the most specific table location containing it decides, so an
    /// external table's location inside another table's managed directory stays that table's
    /// (review re-check 2 r2-m1). A managed table's directory follows its table: reading needs
    /// `Select` there, anything else `Modify` or `Drop` (an owner holds both). An external
    /// table's location (and what is under it) is left to path grants. The lookup is a
    /// precomputed table (no catalog walk, no filesystem call per check).
    fn managed_location_allows(
        &self,
        state: &model::GrantState,
        principal: &Principal,
        key: &ObjectKey,
        privilege: Privilege,
    ) -> bool {
        let Some(store) = self.catalog.get() else {
            return false;
        };
        let index = self.index.get(store);
        let Some((relation, managed)) = index.owner_of_dir(key) else {
            return false;
        };
        if !managed {
            // An external location is left to path grants.
            return false;
        }
        let exact = self.exact_for(relation);
        if privilege == Privilege::Select {
            return state.allows(principal, relation, Privilege::Select, exact);
        }
        state.allows(principal, relation, Privilege::Modify, exact)
            || state.allows(principal, relation, Privilege::Drop, exact)
    }

    /// Whether a listing shows `securable` to `principal` (`model::GrantState::lists`); `None`
    /// for a securable the `system` policy decides (then `Browse` decides).
    pub(crate) fn listable(&self, principal: &Principal, securable: &Securable) -> Option<bool> {
        if Self::decided_by(securable) == DecidedBy::Roles {
            return None;
        }
        if self.admin && principal.definer_for.is_none() {
            return Some(true);
        }
        let Some(key) = ObjectKey::of(securable).and_then(physical_key) else {
            return Some(false);
        };
        let cs = self.exact_for(&key);
        Some(self.grants.store.snapshot().0.lists(principal, &key, cs))
    }

    /// Whether a listing shows the namespace `securable` to `principal` because of something under
    /// it (`model::GrantState::holds_under`): admin always; `None` for a securable the `system`
    /// policy decides. Answered from the grant state alone, with no catalog call.
    pub(crate) fn listable_under(
        &self,
        principal: &Principal,
        securable: &Securable,
    ) -> Option<bool> {
        if Self::decided_by(securable) == DecidedBy::Roles {
            return None;
        }
        if self.admin && principal.definer_for.is_none() {
            return Some(true);
        }
        let Some(key) = ObjectKey::of(securable).and_then(physical_key) else {
            return Some(false);
        };
        let cs = self.exact_for(&key);
        Some(
            self.grants
                .store
                .snapshot()
                .0
                .holds_under(principal, &key, cs),
        )
    }

    /// The owner (creator) recorded for `securable` itself, not a parent's (review r7 M9: the owner
    /// of a stored function whose body a definer view's text calls is asked about the body's reads).
    pub(crate) fn owner_of(&self, securable: &Securable) -> Option<String> {
        let key = ObjectKey::of(securable).and_then(physical_key)?;
        let cs = self.exact_for(&key);
        let state = self.grants.store.snapshot();
        state
            .0
            .owners
            .iter()
            .find(|(k, _)| k.same(&key, cs))
            .map(|(_, owner)| owner.clone())
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
            // The reserved `system` catalog's namespaces and functions (review m1).
            Securable::Namespace(n) | Securable::Function(n)
                if n.first().is_some_and(|c| crate::system::is_system(c)) =>
            {
                DecidedBy::Roles
            }
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
        // A principal acting for another (a definer view's owner for its reader, a refresh's
        // owner for the scheduler; P3-T062 §2.4) holds only its own grants: never the session's
        // admin role, nor what the session's running statement is creating.
        let own = principal.definer_for.is_none();
        if self.admin && own {
            return AccessDecision::Allow;
        }
        let Some(key) = ObjectKey::of(securable).and_then(physical_key) else {
            return deny("the securable has no grant key");
        };
        let cs = self.exact_for(&key);
        // The object the running statement creates, its columns and its managed location: the
        // creator will own it (only set when the statement is really creating it).
        if own
            && self
                .creating
                .lock()
                .is_ok_and(|keys| keys.iter().any(|k| key.covered_by(k, cs)))
        {
            return AccessDecision::Allow;
        }
        let state = self.grants.store.snapshot();
        if state.0.allows(principal, &key, privilege, cs)
            || (key.kind == model::KeyKind::Path
                && self.managed_location_allows(&state.0, principal, &key, privilege))
        {
            AccessDecision::Allow
        } else {
            deny("no grant")
        }
    }
}

/// `key` with a local path's symbolic links resolved (review M2): a path grant is matched on
/// the file the read really opens, the form `PathPrefixPolicy` confinement compares, so a link
/// inside a granted directory never reaches outside it. `None` (denied) for a local path that
/// does not resolve or holds `%` (percent-encoded text would not be decoded the way the
/// comparison reads it). Other keys are returned as they are.
pub(crate) fn physical_key(key: ObjectKey) -> Option<ObjectKey> {
    let Some(local) = (key.kind == model::KeyKind::Path)
        .then(|| key.parts.first().and_then(|p| p.strip_prefix("file://")))
        .flatten()
    else {
        return Some(key);
    };
    if local.contains('%') {
        return None;
    }
    let resolved = crate::access::resolve_links(std::path::Path::new(local))?;
    let text = resolved.to_str()?;
    let normal = model::path_key(text)?;
    Some(ObjectKey::new(model::KeyKind::Path, vec![normal]))
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
        // Counted only: the session writes the one audit record (review M4).
        self.grants.count(provider, privilege);
    }
}

/// Who decided the refusal `e` of a session with the grants provider, read back from the
/// securable it names (as [`GrantsPolicy::decided_by`] routes it), so the audit record's
/// `provider` is the counter's label. A refusal that names no securable (an access check failing
/// closed) is the grants provider's.
pub(crate) fn denial_provider(e: &KrishivError) -> DecidedBy {
    let Some((_, securable)) = e
        .message_parameters()
        .iter()
        .find(|(name, _)| *name == "securable")
    else {
        return DecidedBy::Grants;
    };
    if securable.starts_with("system ") {
        return DecidedBy::Roles;
    }
    let catalog = ["table ", "view ", "namespace ", "function "]
        .iter()
        .find_map(|kind| securable.strip_prefix(kind))
        .or_else(|| {
            // `column `c` of table `cat`...`: skip the column's quoted name.
            let (_, rest) = first_quoted(securable.strip_prefix("column ")?)?;
            rest.strip_prefix(" of table ")
        })
        .and_then(first_quoted);
    match catalog {
        Some((name, _)) if crate::system::is_system(&name) => DecidedBy::Roles,
        _ => DecidedBy::Grants,
    }
}

/// The first `` `quoted` `` identifier of `text` (a doubled backtick is one) and what follows it.
fn first_quoted(text: &str) -> Option<(String, &str)> {
    let mut chars = text.strip_prefix('`')?.char_indices().peekable();
    let mut name = String::new();
    while let Some((i, c)) = chars.next() {
        if c != '`' {
            name.push(c);
        } else if chars.peek().is_some_and(|(_, n)| *n == '`') {
            chars.next();
            name.push('`');
        } else {
            return Some((name, &text[1 + i + 1..]));
        }
    }
    None
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
