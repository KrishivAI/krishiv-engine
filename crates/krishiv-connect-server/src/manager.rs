// SPDX-License-Identifier: Apache-2.0
//! The Connect session manager (design P3-T001 §3.2; Spark's
//! `SparkConnectSessionManager`): sessions keyed by the authenticated
//! principal and the client's session id, created on first use by the RPCs
//! that create, tombstoned when released, swept when idle, capped before
//! admission.
//!
//! Spark keys by `(UserContext.user_id, session_id)`. Krishiv keys by the
//! authenticated caller instead (ADR 0046 §5, threat model T-A8): the
//! client's `user_id` is recorded, unverified, and never decides anything.
//! The key is the caller's [`SessionOwner`] (method, principal kind, tenant
//! and name, without groups). A session id another owner holds is refused as
//! unknown (`INVALID_HANDLE.SESSION_NOT_FOUND`), the answer a lookup of an
//! unknown id gets.
//!
//! A session's policy is fixed when it is built, from the caller's role. A
//! caller whose role differs from the role its live session was built with
//! (a role list changed, an OIDC group claim changed) never gets that
//! session: it is closed, without a tombstone, and the lookup goes on as if
//! the id were free, so a create builds a new session under the new role and
//! a client that saw the old one gets `INVALID_HANDLE.SESSION_CHANGED`; an
//! RPC that creates nothing gets `INVALID_HANDLE.SESSION_NOT_FOUND`. Two
//! clients of one owner using one session id under different roles at once
//! replace each other's session on every request (each keeps its own role).
//!
//! No lock is held across an `await`, nor while a session is built or
//! closed: the map lock guards lookups and inserts only.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use krishiv_common::config::{
    CONNECT_MAX_SESSIONS, CONNECT_MAX_SESSIONS_PER_PRINCIPAL, CONNECT_SESSION_IDLE_TIMEOUT,
    CONNECT_SESSION_TOMBSTONES, Config, ConfigValue,
};
use krishiv_common::error::{
    InvalidHandleSessionChanged, InvalidHandleSessionClosed, InvalidHandleSessionNotFound,
    KrishivError, KrishivSessionLimit,
};
use krishiv_session::auth::Authenticated;
use krishiv_session::{QueryId, Session};

use crate::auth::SessionOwner;
use crate::operation::Operations;

/// Builds the session an authenticated caller gets for a new session id:
/// its principal, and the policy of its role.
pub type SessionFactory = dyn Fn(&Authenticated) -> Result<Session, KrishivError> + Send + Sync;

/// The manager's limits and timers.
#[derive(Debug, Clone)]
pub struct SessionLimits {
    /// Live sessions in the whole server (`krishiv.connect.maxSessions`).
    pub max_sessions: usize,
    /// Live sessions of one principal
    /// (`krishiv.connect.maxSessionsPerPrincipal`).
    pub max_sessions_per_principal: usize,
    /// A session unused this long is released
    /// (`spark.connect.session.manager.defaultSessionTimeout`); `None`
    /// never expires.
    pub idle_timeout: Option<Duration>,
    /// How many released ids are remembered to answer
    /// `INVALID_HANDLE.SESSION_CLOSED`
    /// (`spark.connect.session.manager.closedSessionsTombstonesSize`).
    pub tombstones: usize,
}

impl Default for SessionLimits {
    fn default() -> Self {
        Self {
            max_sessions: 1000,
            max_sessions_per_principal: 100,
            idle_timeout: Some(Duration::from_secs(60 * 60)),
            tombstones: 1000,
        }
    }
}

impl SessionLimits {
    /// The limits `config` gives (`krishiv.connect.maxSessions`,
    /// `maxSessionsPerPrincipal`, `session.idleTimeout` with `0s` for never,
    /// `session.closedTombstones`); defaults otherwise.
    #[must_use]
    pub fn from_config(config: &Config) -> Self {
        let d = Self::default();
        let count = |e| {
            config
                .get(e)
                .and_then(ConfigValue::as_int)
                .and_then(|v| usize::try_from(v).ok())
        };
        let idle_timeout = match config
            .get(&CONNECT_SESSION_IDLE_TIMEOUT)
            .and_then(ConfigValue::as_duration)
        {
            Some(t) if t.is_zero() => None,
            Some(t) => Some(t),
            None => d.idle_timeout,
        };
        Self {
            max_sessions: count(&CONNECT_MAX_SESSIONS).unwrap_or(d.max_sessions),
            max_sessions_per_principal: count(&CONNECT_MAX_SESSIONS_PER_PRINCIPAL)
                .unwrap_or(d.max_sessions_per_principal),
            idle_timeout,
            tombstones: count(&CONNECT_SESSION_TOMBSTONES).unwrap_or(d.tombstones),
        }
    }
}

/// A live Connect session.
#[derive(Debug)]
pub struct SessionEntry {
    session: Session,
    caller: Authenticated,
    client_user_id: String,
    operations: Arc<Operations>,
    /// Milliseconds since the manager's epoch at the last lookup.
    last_access: AtomicU64,
}

impl SessionEntry {
    /// The engine session.
    #[must_use]
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// The server-side session id every response carries
    /// (`server_side_session_id`): the engine session's own id, distinct
    /// from the client's.
    #[must_use]
    pub fn server_side_id(&self) -> &str {
        self.session.id()
    }

    /// The caller the session was built for (its principal and role).
    #[must_use]
    pub fn caller(&self) -> &Authenticated {
        &self.caller
    }

    /// The `UserContext.user_id` the client sent when it created the
    /// session: recorded, never trusted.
    #[must_use]
    pub fn client_user_id(&self) -> &str {
        &self.client_user_id
    }

    /// The session's live operations (`ExecutePlan` calls).
    #[must_use]
    pub fn operations(&self) -> &Arc<Operations> {
        &self.operations
    }

    /// Cancels the session's operations, then closes the session (its
    /// queries cancelled).
    fn close(&self) {
        self.operations.cancel_all();
        self.session.close();
    }
}

/// What the manager holds, for tests and `system.sessions` (T064).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManagerCounts {
    /// Live sessions.
    pub live: usize,
    /// Remembered released ids.
    pub tombstones: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key {
    owner: SessionOwner,
    session_id: String,
}

#[derive(Default)]
struct Inner {
    live: HashMap<Key, Arc<SessionEntry>>,
    /// The owner holding each live session id.
    owners: HashMap<String, SessionOwner>,
    per_principal: HashMap<SessionOwner, usize>,
    closed: Tombstones,
    /// Set by [`SessionManager::close_all`]: no session is created after it.
    shut: bool,
}

/// A bounded set of released keys; the least recently used goes first, as
/// Spark's Guava cache with a maximum size and no expiry.
#[derive(Default)]
struct Tombstones {
    stamp: HashMap<Key, u64>,
    order: BTreeMap<u64, Key>,
    next: u64,
}

impl Tombstones {
    fn contains(&mut self, key: &Key) -> bool {
        // A lookup is an access: it refreshes the entry.
        let Some(old) = self.stamp.get(key).copied() else {
            return false;
        };
        self.order.remove(&old);
        self.next += 1;
        self.order.insert(self.next, key.clone());
        self.stamp.insert(key.clone(), self.next);
        true
    }

    fn insert(&mut self, key: Key, capacity: usize) {
        if let Some(old) = self.stamp.remove(&key) {
            self.order.remove(&old);
        }
        if capacity == 0 {
            return;
        }
        self.next += 1;
        self.order.insert(self.next, key.clone());
        self.stamp.insert(key, self.next);
        while self.stamp.len() > capacity {
            let Some((_, oldest)) = self.order.pop_first() else {
                break;
            };
            self.stamp.remove(&oldest);
        }
    }
}

/// The Connect sessions of one server.
pub struct SessionManager {
    factory: Box<SessionFactory>,
    limits: SessionLimits,
    inner: Mutex<Inner>,
    epoch: Instant,
}

impl std::fmt::Debug for SessionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionManager")
            .field("limits", &self.limits)
            .field("counts", &self.counts())
            .finish_non_exhaustive()
    }
}

impl SessionManager {
    /// A manager whose sessions `factory` builds.
    #[must_use]
    pub fn new(factory: Box<SessionFactory>, limits: SessionLimits) -> Self {
        Self {
            factory,
            limits,
            inner: Mutex::new(Inner::default()),
            epoch: Instant::now(),
        }
    }

    /// A manager whose sessions are new sessions of `base` (same runtime,
    /// profile and function registry; Spark's `baseSession.newSession()`).
    /// Every caller gets `base`'s principal and policy: the server without
    /// authentication (loopback only); with it, [`Self::new`] takes a factory
    /// that builds each caller's session (`Session::new_session_as`).
    #[must_use]
    pub fn from_base(base: Session, limits: SessionLimits) -> Self {
        Self::new(Box::new(move |_| Ok(base.new_session())), limits)
    }

    /// The limits in force.
    #[must_use]
    pub fn limits(&self) -> &SessionLimits {
        &self.limits
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn now_ms(&self) -> u64 {
        u64::try_from(self.epoch.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    fn key(caller: &Authenticated, session_id: &str) -> Key {
        Key {
            owner: SessionOwner::of(caller),
            session_id: session_id.to_owned(),
        }
    }

    /// The live entry for `key` if it was built for `caller`'s role, stamped
    /// as an access (under the lock, so a sweep cannot release it in
    /// between). One built for another role leaves the map without a
    /// tombstone and goes to `retired`, to be closed after the lock is
    /// released (module docs).
    fn live_locked(
        &self,
        inner: &mut Inner,
        key: &Key,
        caller: &Authenticated,
        retired: &mut Option<Arc<SessionEntry>>,
    ) -> Option<Arc<SessionEntry>> {
        let entry = inner.live.get(key).map(Arc::clone)?;
        if entry.caller.role == caller.role {
            entry.last_access.store(self.now_ms(), Ordering::Relaxed);
            return Some(entry);
        }
        *retired = Self::remove_locked(inner, key);
        None
    }

    /// Closes a session [`Self::live_locked`] retired.
    fn close_retired(retired: Option<Arc<SessionEntry>>) {
        if let Some(entry) = retired {
            tracing::info!(
                target: "krishiv::connect",
                session = entry.server_side_id(),
                "closed a Connect session: its caller's role changed"
            );
            entry.close();
        }
    }

    /// The session `session_id` of `caller`, created when missing (the
    /// `ExecutePlan`, `AnalyzePlan` and `Config` RPCs).
    ///
    /// # Errors
    /// `INVALID_HANDLE.FORMAT` for an id that is not a UUID (on create),
    /// `INVALID_HANDLE.SESSION_CLOSED` for a released id,
    /// `INVALID_HANDLE.SESSION_NOT_FOUND` for an id another owner
    /// holds, `INVALID_HANDLE.SESSION_CHANGED` when `observed` is not the
    /// session's server-side id, `KRISHIV_SESSION_LIMIT` past a session
    /// cap, and whatever building the session fails with.
    pub fn get_or_create(
        &self,
        caller: &Authenticated,
        session_id: &str,
        observed: Option<&str>,
        client_user_id: &str,
    ) -> Result<Arc<SessionEntry>, KrishivError> {
        let key = Self::key(caller, session_id);
        if let Some(entry) = self.lookup(&key, caller, observed)? {
            return Ok(entry);
        }
        // Validate before building: Spark's `validateSessionCreate`.
        QueryId::parse(session_id)?;
        if let Some(existing) = self.check_create(&key, caller)? {
            Self::check_observed(&key, &existing, observed)?;
            return Ok(existing);
        }
        // Built without the lock; a racing create of the same key wins and
        // this one is closed unused.
        let session = (self.factory)(caller)?;
        // `system.sessions` shows the client's `user_id` as reported (ADR 0046 §5: recorded,
        // unverified, never the identity).
        session.set_client_info(
            krishiv_session::Transport::Connect,
            None,
            Some(client_user_id),
            Some(session.id()),
        );
        let entry = Arc::new(SessionEntry {
            session,
            caller: caller.clone(),
            client_user_id: client_user_id.to_owned(),
            operations: Arc::default(),
            last_access: AtomicU64::new(self.now_ms()),
        });
        let mut retired = None;
        let raced = {
            let mut inner = self.lock();
            if let Some(existing) = self.live_locked(&mut inner, &key, caller, &mut retired) {
                Ok(Some(existing))
            } else {
                match Self::check_create_locked(&mut inner, &key, &self.limits) {
                    Err(e) => Err(e),
                    Ok(()) => {
                        inner.live.insert(key.clone(), Arc::clone(&entry));
                        inner
                            .owners
                            .insert(key.session_id.clone(), key.owner.clone());
                        *inner.per_principal.entry(key.owner.clone()).or_insert(0) += 1;
                        Ok(None)
                    }
                }
            }
        };
        Self::close_retired(retired);
        match raced {
            Err(e) => {
                // Built, then refused by a check that changed meanwhile.
                entry.session.close();
                Err(e)
            }
            Ok(Some(existing)) => {
                entry.session.close();
                Self::check_observed(&key, &existing, observed)?;
                Ok(existing)
            }
            Ok(None) => {
                Self::check_observed(&key, &entry, observed)?;
                Ok(entry)
            }
        }
    }

    /// The live session `session_id` of `caller` (every RPC that does not
    /// create one).
    ///
    /// # Errors
    /// `INVALID_HANDLE.SESSION_CLOSED` for a released id,
    /// `INVALID_HANDLE.SESSION_NOT_FOUND` for an unknown one (or one another
    /// owner holds, or one built for another role of the caller),
    /// `INVALID_HANDLE.SESSION_CHANGED` when `observed` differs from its
    /// server-side id.
    pub fn get(
        &self,
        caller: &Authenticated,
        session_id: &str,
        observed: Option<&str>,
    ) -> Result<Arc<SessionEntry>, KrishivError> {
        let key = Self::key(caller, session_id);
        if let Some(entry) = self.lookup(&key, caller, observed)? {
            return Ok(entry);
        }
        let closed = self.lock().closed.contains(&key);
        Err(if closed {
            InvalidHandleSessionClosed {
                handle: session_id.to_owned(),
            }
            .into()
        } else {
            InvalidHandleSessionNotFound {
                handle: session_id.to_owned(),
            }
            .into()
        })
    }

    /// The live session, if there is one, without any check (Spark's
    /// `getIsolatedSessionIfPresent`). Counts as an access.
    #[must_use]
    pub fn get_if_present(
        &self,
        caller: &Authenticated,
        session_id: &str,
    ) -> Option<Arc<SessionEntry>> {
        let key = Self::key(caller, session_id);
        let mut retired = None;
        let entry = self.live_locked(&mut self.lock(), &key, caller, &mut retired);
        Self::close_retired(retired);
        entry
    }

    /// Releases session `session_id` of `caller` (`ReleaseSession`): it
    /// leaves the map, its id is tombstoned unless `allow_reconnect`, its
    /// operations are cancelled and it is closed (its queries cancelled). An unknown id is a no-op. Returns
    /// the released session's server-side id.
    pub fn release(
        &self,
        caller: &Authenticated,
        session_id: &str,
        allow_reconnect: bool,
    ) -> Option<String> {
        // Keyed by the owner alone: releasing ends the session whatever role
        // it was built with.
        let key = Self::key(caller, session_id);
        let entry = {
            let mut inner = self.lock();
            let entry = Self::remove_locked(&mut inner, &key)?;
            if !allow_reconnect {
                inner.closed.insert(key, self.limits.tombstones);
            }
            entry
        };
        let id = entry.server_side_id().to_owned();
        entry.close();
        Some(id)
    }

    /// Releases every session idle past the timeout, running no query and
    /// holding no live operation (Spark's periodic maintenance; Krishiv also
    /// keeps a session whose query or response stream is still live). Returns how many it released.
    pub fn sweep(&self) -> usize {
        let Some(timeout) = self.limits.idle_timeout else {
            return 0;
        };
        let timeout = u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX);
        let now = self.now_ms();
        let expired: Vec<Arc<SessionEntry>> = {
            let mut inner = self.lock();
            let keys: Vec<Key> = inner
                .live
                .iter()
                .filter(|(_, e)| {
                    e.last_access
                        .load(Ordering::Relaxed)
                        .saturating_add(timeout)
                        <= now
                        && e.session.running_queries() == 0
                        && e.operations.is_empty()
                })
                .map(|(k, _)| k.clone())
                .collect();
            let mut out = Vec::with_capacity(keys.len());
            for key in keys {
                if let Some(entry) = Self::remove_locked(&mut inner, &key) {
                    inner.closed.insert(key, self.limits.tombstones);
                    out.push(entry);
                }
            }
            out
        };
        for entry in &expired {
            tracing::info!(
                target: "krishiv::connect",
                session = entry.server_side_id(),
                "released an idle Connect session"
            );
            entry.close();
        }
        expired.len()
    }

    /// Closes every live session (shutdown): each one's operations and
    /// queries are cancelled, and no session is created afterwards (a create answers
    /// `INVALID_HANDLE.SESSION_CLOSED`). Returns how many.
    pub fn close_all(&self) -> usize {
        let entries: Vec<Arc<SessionEntry>> = {
            let mut inner = self.lock();
            inner.shut = true;
            inner.owners.clear();
            inner.per_principal.clear();
            inner.live.drain().map(|(_, e)| e).collect()
        };
        for entry in &entries {
            entry.close();
        }
        entries.len()
    }

    /// The running queries plus the live operations of all live sessions:
    /// zero once nothing is left to finish (shutdown waits for it).
    #[must_use]
    pub fn busy(&self) -> usize {
        let entries: Vec<Arc<SessionEntry>> = self.lock().live.values().map(Arc::clone).collect();
        entries
            .iter()
            .map(|e| e.session.running_queries() + e.operations.len())
            .sum()
    }

    /// The running queries of all live sessions.
    #[must_use]
    pub fn running_queries(&self) -> usize {
        let entries: Vec<Arc<SessionEntry>> = self.lock().live.values().map(Arc::clone).collect();
        entries.iter().map(|e| e.session.running_queries()).sum()
    }

    /// Live sessions and tombstones.
    #[must_use]
    pub fn counts(&self) -> ManagerCounts {
        let inner = self.lock();
        ManagerCounts {
            live: inner.live.len(),
            tombstones: inner.closed.stamp.len(),
        }
    }

    /// The live entry for `key` built for `caller`'s role, checked against
    /// `observed`.
    fn lookup(
        &self,
        key: &Key,
        caller: &Authenticated,
        observed: Option<&str>,
    ) -> Result<Option<Arc<SessionEntry>>, KrishivError> {
        let mut retired = None;
        let found = {
            let mut inner = self.lock();
            match self.live_locked(&mut inner, key, caller, &mut retired) {
                Some(e) => Ok(Some(e)),
                // Held by another owner: refused as unknown.
                None if inner
                    .owners
                    .get(&key.session_id)
                    .is_some_and(|o| *o != key.owner) =>
                {
                    Err(KrishivError::from(InvalidHandleSessionNotFound {
                        handle: key.session_id.clone(),
                    }))
                }
                None => Ok(None),
            }
        };
        Self::close_retired(retired);
        let Some(entry) = found? else {
            return Ok(None);
        };
        Self::check_observed(key, &entry, observed)?;
        Ok(Some(entry))
    }

    fn check_observed(
        key: &Key,
        entry: &SessionEntry,
        observed: Option<&str>,
    ) -> Result<(), KrishivError> {
        match observed {
            Some(seen) if seen != entry.server_side_id() => Err(InvalidHandleSessionChanged {
                handle: key.session_id.clone(),
            }
            .into()),
            _ => Ok(()),
        }
    }

    /// The caps, checked before a session is built. A racing create that
    /// inserted the key since the lookup is returned instead: reusing it
    /// takes no slot, so the caps must not refuse it.
    fn check_create(
        &self,
        key: &Key,
        caller: &Authenticated,
    ) -> Result<Option<Arc<SessionEntry>>, KrishivError> {
        let mut retired = None;
        let checked = {
            let mut inner = self.lock();
            match self.live_locked(&mut inner, key, caller, &mut retired) {
                Some(existing) => Ok(Some(existing)),
                None => Self::check_create_locked(&mut inner, key, &self.limits).map(|()| None),
            }
        };
        Self::close_retired(retired);
        checked
    }

    fn check_create_locked(
        inner: &mut Inner,
        key: &Key,
        limits: &SessionLimits,
    ) -> Result<(), KrishivError> {
        // Shutting down: a handler that passed the drain check before
        // `close_all` must not leave a session nothing will close.
        if inner.shut || inner.closed.contains(key) {
            return Err(InvalidHandleSessionClosed {
                handle: key.session_id.clone(),
            }
            .into());
        }
        if inner
            .owners
            .get(&key.session_id)
            .is_some_and(|o| *o != key.owner)
        {
            return Err(InvalidHandleSessionNotFound {
                handle: key.session_id.clone(),
            }
            .into());
        }
        if inner.live.len() >= limits.max_sessions {
            return Err(KrishivSessionLimit {
                limit_name: "krishiv.connect.maxSessions".to_owned(),
                limit: limits.max_sessions.to_string(),
            }
            .into());
        }
        if inner.per_principal.get(&key.owner).copied().unwrap_or(0)
            >= limits.max_sessions_per_principal
        {
            return Err(KrishivSessionLimit {
                limit_name: "krishiv.connect.maxSessionsPerPrincipal".to_owned(),
                limit: limits.max_sessions_per_principal.to_string(),
            }
            .into());
        }
        Ok(())
    }

    fn remove_locked(inner: &mut Inner, key: &Key) -> Option<Arc<SessionEntry>> {
        let entry = inner.live.remove(key)?;
        inner.owners.remove(&key.session_id);
        if let Some(n) = inner.per_principal.get_mut(&key.owner) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                inner.per_principal.remove(&key.owner);
            }
        }
        Some(entry)
    }
}

#[cfg(test)]
mod tests;
