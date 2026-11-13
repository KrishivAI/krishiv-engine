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
//! (`INVALID_HANDLE.SESSION_NOT_FOUND`), the answer a lookup of an unknown
//! id gets.
//!
//! No lock is held across an `await`, nor while a session is built or
//! closed: the map lock guards lookups and inserts only. The one exception
//! is each session's statement gate ([`SessionEntry::shared`]), a tokio
//! `RwLock` built to be held across an `await`: `CloneSession` takes it
//! exclusively around the copy (`clone`).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use krishiv_common::config::{
    CONNECT_EXECUTE_ABANDONED_TOMBSTONES, CONNECT_MAX_SESSIONS, CONNECT_MAX_SESSIONS_PER_PRINCIPAL,
    CONNECT_SESSION_IDLE_TIMEOUT, CONNECT_SESSION_TOMBSTONES, Config, ConfigValue,
};
use krishiv_common::error::{
    InvalidHandleSessionChanged, InvalidHandleSessionClosed, InvalidHandleSessionNotFound,
    KrishivError, KrishivSessionLimit,
};
use krishiv_common::principal::Principal;
use krishiv_session::{QueryId, Session};

use crate::artifacts::SessionArtifacts;
use crate::errors::SessionErrors;
use crate::operation::{Operation, Operations};

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
    /// How many abandoned operation ids are remembered to answer
    /// `INVALID_HANDLE.OPERATION_ABANDONED`
    /// (`spark.connect.execute.manager.abandonedTombstonesSize`, 10,000).
    pub abandoned_tombstones: usize,
}

impl Default for SessionLimits {
    fn default() -> Self {
        Self {
            max_sessions: 1000,
            max_sessions_per_principal: 100,
            idle_timeout: Some(Duration::from_secs(60 * 60)),
            tombstones: 1000,
            abandoned_tombstones: 10_000,
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
            abandoned_tombstones: count(&CONNECT_EXECUTE_ABANDONED_TOMBSTONES)
                .unwrap_or(d.abandoned_tombstones),
        }
    }
}

/// A live Connect session.
#[derive(Debug)]
pub struct SessionEntry {
    session: Session,
    client_user_id: String,
    operations: Arc<Operations>,
    /// The errors sent on the session, for `FetchErrorDetails`.
    errors: Arc<SessionErrors>,
    /// The session's artifacts (`AddArtifacts`).
    artifacts: Arc<SessionArtifacts>,
    /// Milliseconds since the manager's epoch at the last lookup.
    last_access: AtomicU64,
    /// Held shared by the RPCs that change the session's state while they
    /// do, exclusively by `CloneSession` around its copy.
    gate: tokio::sync::RwLock<()>,
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

    /// The session's live operations (`ExecutePlan` calls).
    #[must_use]
    pub fn operations(&self) -> &Arc<Operations> {
        &self.operations
    }

    /// The errors sent on the session: [`SessionErrors::status`] turns an
    /// error into the status with its `errorId`.
    #[must_use]
    pub fn errors(&self) -> &Arc<SessionErrors> {
        &self.errors
    }

    /// The session's artifacts.
    #[must_use]
    pub fn artifacts(&self) -> &Arc<SessionArtifacts> {
        &self.artifacts
    }

    /// The statement gate, shared: the RPCs that change the session's state
    /// (`ExecutePlan` until its query is started, `AnalyzePlan`, `Config`;
    /// the commands of T012) hold it while they do, so a `CloneSession` of
    /// this session copies it between two of them, never in the middle of
    /// one. Each RPC holds the gate of its own session only and takes no
    /// other lock while it waits for it, so it cannot deadlock.
    pub async fn shared(&self) -> tokio::sync::RwLockReadGuard<'_, ()> {
        self.gate.read().await
    }

    /// The statement gate, exclusively (`CloneSession` around its copy).
    pub(crate) async fn exclusive(&self) -> tokio::sync::RwLockWriteGuard<'_, ()> {
        self.gate.write().await
    }

    /// Closes the session's operations, then the session (its queries
    /// cancelled), then its artifacts (their directory removed). Returns the
    /// operations it closed.
    fn close(&self) -> Vec<Arc<Operation>> {
        let ops = self.operations.close_all();
        self.session.close();
        self.artifacts.close();
        ops
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

/// An operation of a session (Spark's `ExecuteKey`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct OpKey {
    session: Key,
    operation: QueryId,
}

#[derive(Default)]
struct Inner {
    live: HashMap<Key, Arc<SessionEntry>>,
    /// The principal holding each live session id.
    owners: HashMap<String, Principal>,
    per_principal: HashMap<Principal, usize>,
    closed: Tombstones<Key>,
    /// Abandoned operations (Spark's `abandonedTombstones`).
    abandoned: Tombstones<OpKey>,
    /// Set by [`SessionManager::close_all`]: no session is created after it.
    shut: bool,
}

/// A bounded set of released keys; the least recently used goes first, as
/// Spark's Guava cache with a maximum size and no expiry.
struct Tombstones<K> {
    stamp: HashMap<K, u64>,
    order: BTreeMap<u64, K>,
    next: u64,
}

impl<K> Default for Tombstones<K> {
    fn default() -> Self {
        Self {
            stamp: HashMap::new(),
            order: BTreeMap::new(),
            next: 0,
        }
    }
}

impl<K: Clone + Eq + std::hash::Hash> Tombstones<K> {
    fn contains(&mut self, key: &K) -> bool {
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

    fn insert(&mut self, key: K, capacity: usize) {
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

    /// A new entry for `session`, not yet in the map.
    fn new_entry(&self, session: Session, client_user_id: &str) -> Arc<SessionEntry> {
        Arc::new(SessionEntry {
            errors: Arc::new(SessionErrors::new(session.conf())),
            artifacts: Arc::default(),
            session,
            client_user_id: client_user_id.to_owned(),
            operations: Arc::default(),
            last_access: AtomicU64::new(self.now_ms()),
            gate: tokio::sync::RwLock::new(()),
        })
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
        if let Some(existing) = self.check_create(&key)? {
            Self::check_observed(&key, &existing, observed)?;
            return Ok(existing);
        }
        // Built without the lock; a racing create of the same key wins and
        // this one is closed unused.
        let session = (self.factory)(principal)?;
        let entry = self.new_entry(session, client_user_id);
        let raced = {
            let mut inner = self.lock();
            if let Some(existing) = inner.live.get(&key).map(Arc::clone) {
                existing.last_access.store(self.now_ms(), Ordering::Relaxed);
                Ok(Some(existing))
            } else {
                match Self::check_create_locked(&mut inner, &key, &self.limits) {
                    Err(e) => Err(e),
                    Ok(()) => {
                        inner.live.insert(key.clone(), Arc::clone(&entry));
                        inner
                            .owners
                            .insert(key.session_id.clone(), key.principal.clone());
                        *inner
                            .per_principal
                            .entry(key.principal.clone())
                            .or_insert(0) += 1;
                        Ok(None)
                    }
                }
            }
        };
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
        let inner = self.lock();
        let entry = inner.live.get(&key).map(Arc::clone)?;
        // Stamped under the lock, so a sweep cannot release it in between.
        entry.last_access.store(self.now_ms(), Ordering::Relaxed);
        Some(entry)
    }

    /// Releases session `session_id` of `principal` (`ReleaseSession`): it
    /// leaves the map, its id is tombstoned unless `allow_reconnect`, its
    /// operations are cancelled and it is closed (its queries cancelled). An unknown id is a no-op. Returns
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
                inner.closed.insert(key.clone(), self.limits.tombstones);
            }
            entry
        };
        let id = entry.server_side_id().to_owned();
        // Spark's `removeAllExecutionsForSession` removes them as abandoned.
        let ops = entry.close();
        if !ops.is_empty() {
            let mut inner = self.lock();
            for op in ops {
                inner.abandoned.insert(
                    OpKey {
                        session: key.clone(),
                        operation: op.id().clone(),
                    },
                    self.limits.abandoned_tombstones,
                );
            }
        }
        Some(id)
    }

    /// Whether operation `operation_id` of session `session_id` was
    /// abandoned (a reattach answers `INVALID_HANDLE.OPERATION_ABANDONED`).
    #[must_use]
    pub fn is_abandoned(
        &self,
        principal: &Principal,
        session_id: &str,
        operation_id: &str,
    ) -> bool {
        let Ok(operation) = QueryId::parse(operation_id) else {
            return false;
        };
        let key = OpKey {
            session: Self::key(principal, session_id),
            operation,
        };
        self.lock().abandoned.contains(&key)
    }

    /// Spark's periodic maintenance of executions
    /// (`SparkConnectExecutionManager.periodicMaintenance`, v4.3.0 lines
    /// 342-362): a sender past its deadline is detached first (its client
    /// may have stopped reading, so its stream is never polled again), then
    /// every reattachable operation with no stream attached for `timeout`
    /// is tombstoned as abandoned, removed and closed (its query
    /// interrupted, its responses and budgets returned). Returns how many.
    pub fn abandon_detached(&self, timeout: Duration) -> usize {
        let sessions: Vec<(Key, Arc<SessionEntry>)> = self
            .lock()
            .live
            .iter()
            .map(|(k, e)| (k.clone(), Arc::clone(e)))
            .collect();
        let mut abandoned = 0;
        for (key, entry) in sessions {
            for op in entry.operations().all() {
                op.replay().expire_sender(std::time::Instant::now());
                let expired = op
                    .replay()
                    .detached_since()
                    .is_some_and(|since| since.elapsed() >= timeout);
                if !expired {
                    continue;
                }
                // Tombstoned before it leaves the map, so a racing reattach
                // finds it abandoned rather than unknown (lines 158-166).
                // A reattach between the check above and the close below
                // is closed with it (its retry sees OPERATION_ABANDONED):
                // Spark's `periodicMaintenance` has the same window.
                self.lock().abandoned.insert(
                    OpKey {
                        session: key.clone(),
                        operation: op.id().clone(),
                    },
                    self.limits.abandoned_tombstones,
                );
                if entry.operations().remove(&op) {
                    tracing::info!(
                        target: "krishiv::connect",
                        operation = op.id().as_str(),
                        "abandoned a detached Connect operation"
                    );
                    abandoned += 1;
                }
                op.close();
            }
        }
        abandoned
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
            // It holds no operation (checked above).
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

    /// The live entry for `key`, checked against `observed`.
    fn lookup(
        &self,
        key: &Key,
        observed: Option<&str>,
    ) -> Result<Option<Arc<SessionEntry>>, KrishivError> {
        let entry = {
            let inner = self.lock();
            match inner.live.get(key) {
                Some(e) => {
                    // Stamped under the lock, so a sweep cannot release it
                    // in between.
                    e.last_access.store(self.now_ms(), Ordering::Relaxed);
                    Arc::clone(e)
                }
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
    fn check_create(&self, key: &Key) -> Result<Option<Arc<SessionEntry>>, KrishivError> {
        let mut inner = self.lock();
        if let Some(existing) = inner.live.get(key).map(Arc::clone) {
            existing.last_access.store(self.now_ms(), Ordering::Relaxed);
            return Ok(Some(existing));
        }
        Self::check_create_locked(&mut inner, key, &self.limits).map(|()| None)
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

mod clone;

#[cfg(test)]
mod tests;
