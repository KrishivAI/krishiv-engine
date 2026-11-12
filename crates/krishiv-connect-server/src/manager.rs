// SPDX-License-Identifier: Apache-2.0
//! The Connect session manager (design P3-T001 §3.2; Spark's
//! `SparkConnectSessionManager`): sessions keyed by the authenticated
//! principal and the client's session id, created on first use by the RPCs
//! that create, tombstoned when released, swept when idle, capped before
//! admission.
//!
//! Spark keys by `(UserContext.user_id, session_id)`. Krishiv keys by the
//! authenticated principal instead (ADR 0046 §5, threat model T-A8): the
//! client's `user_id` is recorded, unverified, and never decides anything.
//! A session id another principal holds is refused as unknown
//! (`INVALID_HANDLE.SESSION_NOT_FOUND`), so the refusal reveals nothing.
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
use krishiv_common::principal::Principal;
use krishiv_session::{QueryId, Session};

/// Builds the session a principal gets for a new session id.
pub type SessionFactory = dyn Fn(&Principal) -> Result<Session, KrishivError> + Send + Sync;

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
    client_user_id: String,
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

    /// The `UserContext.user_id` the client sent when it created the
    /// session: recorded, never trusted.
    #[must_use]
    pub fn client_user_id(&self) -> &str {
        &self.client_user_id
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
    principal: Principal,
    session_id: String,
}

#[derive(Default)]
struct Inner {
    live: HashMap<Key, Arc<SessionEntry>>,
    /// The principal holding each live session id.
    owners: HashMap<String, Principal>,
    per_principal: HashMap<Principal, usize>,
    closed: Tombstones,
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
    /// Every principal gets `base`'s principal until authentication (T002)
    /// supplies a factory that sets it.
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

    fn key(principal: &Principal, session_id: &str) -> Key {
        Key {
            principal: principal.clone(),
            session_id: session_id.to_owned(),
        }
    }

    /// The session `session_id` of `principal`, created when missing (the
    /// `ExecutePlan`, `AnalyzePlan` and `Config` RPCs).
    ///
    /// # Errors
    /// `INVALID_HANDLE.FORMAT` for an id that is not a UUID (on create),
    /// `INVALID_HANDLE.SESSION_CLOSED` for a released id,
    /// `INVALID_HANDLE.SESSION_NOT_FOUND` for an id another principal
    /// holds, `INVALID_HANDLE.SESSION_CHANGED` when `observed` is not the
    /// session's server-side id, `KRISHIV_SESSION_LIMIT` past a session
    /// cap, and whatever building the session fails with.
    pub fn get_or_create(
        &self,
        principal: &Principal,
        session_id: &str,
        observed: Option<&str>,
        client_user_id: &str,
    ) -> Result<Arc<SessionEntry>, KrishivError> {
        let key = Self::key(principal, session_id);
        if let Some(entry) = self.lookup(&key, observed)? {
            return Ok(entry);
        }
        // Validate before building: Spark's `validateSessionCreate`.
        QueryId::parse(session_id)?;
        self.check_create(&key)?;
        // Built without the lock; a racing create of the same key wins and
        // this one is closed unused.
        let session = (self.factory)(principal)?;
        let entry = Arc::new(SessionEntry {
            session,
            client_user_id: client_user_id.to_owned(),
            last_access: AtomicU64::new(self.now_ms()),
        });
        let raced = {
            let mut inner = self.lock();
            if let Some(existing) = inner.live.get(&key) {
                Some(Arc::clone(existing))
            } else {
                Self::check_create_locked(&mut inner, &key, &self.limits)?;
                inner.live.insert(key.clone(), Arc::clone(&entry));
                inner
                    .owners
                    .insert(key.session_id.clone(), key.principal.clone());
                *inner
                    .per_principal
                    .entry(key.principal.clone())
                    .or_insert(0) += 1;
                None
            }
        };
        match raced {
            Some(existing) => {
                entry.session.close();
                Self::check_observed(&key, &existing, observed)?;
                existing.last_access.store(self.now_ms(), Ordering::Relaxed);
                Ok(existing)
            }
            None => {
                Self::check_observed(&key, &entry, observed)?;
                Ok(entry)
            }
        }
    }

    /// The live session `session_id` of `principal` (every RPC that does not
    /// create one).
    ///
    /// # Errors
    /// `INVALID_HANDLE.SESSION_CLOSED` for a released id,
    /// `INVALID_HANDLE.SESSION_NOT_FOUND` for an unknown one (or one another
    /// principal holds), `INVALID_HANDLE.SESSION_CHANGED` when `observed`
    /// differs from its server-side id.
    pub fn get(
        &self,
        principal: &Principal,
        session_id: &str,
        observed: Option<&str>,
    ) -> Result<Arc<SessionEntry>, KrishivError> {
        let key = Self::key(principal, session_id);
        if let Some(entry) = self.lookup(&key, observed)? {
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
        principal: &Principal,
        session_id: &str,
    ) -> Option<Arc<SessionEntry>> {
        let key = Self::key(principal, session_id);
        let entry = self.lock().live.get(&key).map(Arc::clone)?;
        entry.last_access.store(self.now_ms(), Ordering::Relaxed);
        Some(entry)
    }

    /// Releases session `session_id` of `principal` (`ReleaseSession`): it
    /// leaves the map, its id is tombstoned unless `allow_reconnect`, and it
    /// is closed (its queries cancelled). An unknown id is a no-op. Returns
    /// the released session's server-side id.
    pub fn release(
        &self,
        principal: &Principal,
        session_id: &str,
        allow_reconnect: bool,
    ) -> Option<String> {
        let key = Self::key(principal, session_id);
        let entry = {
            let mut inner = self.lock();
            let entry = Self::remove_locked(&mut inner, &key)?;
            if !allow_reconnect {
                inner.closed.insert(key, self.limits.tombstones);
            }
            entry
        };
        let id = entry.server_side_id().to_owned();
        entry.session.close();
        Some(id)
    }

    /// Releases every session idle past the timeout and running no query
    /// (Spark's periodic maintenance; Krishiv also keeps a session whose
    /// query is still running). Returns how many it released.
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
            entry.session.close();
        }
        expired.len()
    }

    /// Closes every live session (shutdown): each one's queries are
    /// cancelled. Returns how many.
    pub fn close_all(&self) -> usize {
        let entries: Vec<Arc<SessionEntry>> = {
            let mut inner = self.lock();
            inner.owners.clear();
            inner.per_principal.clear();
            inner.live.drain().map(|(_, e)| e).collect()
        };
        for entry in &entries {
            entry.session.close();
        }
        entries.len()
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

    /// The live entry for `key`, checked against `observed`.
    fn lookup(
        &self,
        key: &Key,
        observed: Option<&str>,
    ) -> Result<Option<Arc<SessionEntry>>, KrishivError> {
        let entry = {
            let inner = self.lock();
            match inner.live.get(key) {
                Some(e) => Arc::clone(e),
                None => {
                    // Held by another principal: refused as unknown.
                    if inner
                        .owners
                        .get(&key.session_id)
                        .is_some_and(|p| *p != key.principal)
                    {
                        return Err(InvalidHandleSessionNotFound {
                            handle: key.session_id.clone(),
                        }
                        .into());
                    }
                    return Ok(None);
                }
            }
        };
        entry.last_access.store(self.now_ms(), Ordering::Relaxed);
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

    fn check_create(&self, key: &Key) -> Result<(), KrishivError> {
        let mut inner = self.lock();
        Self::check_create_locked(&mut inner, key, &self.limits)
    }

    fn check_create_locked(
        inner: &mut Inner,
        key: &Key,
        limits: &SessionLimits,
    ) -> Result<(), KrishivError> {
        if inner.closed.contains(key) {
            return Err(InvalidHandleSessionClosed {
                handle: key.session_id.clone(),
            }
            .into());
        }
        if inner
            .owners
            .get(&key.session_id)
            .is_some_and(|p| *p != key.principal)
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
        if inner
            .per_principal
            .get(&key.principal)
            .copied()
            .unwrap_or(0)
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
        if let Some(n) = inner.per_principal.get_mut(&key.principal) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                inner.per_principal.remove(&key.principal);
            }
        }
        Some(entry)
    }
}

#[cfg(test)]
mod tests;
