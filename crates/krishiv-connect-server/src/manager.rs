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
//! The key is the caller's [`SessionOwner`] (method, principal kind, tenant,
//! name and delegator; groups only for an impersonated principal, whose
//! act-as identity they pin, P3-T061). `maxSessionsPerPrincipal` counts
//! without those groups (`SessionOwner::cap_key`). A session id another owner holds is refused as
//! unknown (`INVALID_HANDLE.SESSION_NOT_FOUND`), the answer a lookup of an
//! unknown id gets, and writes a `krishiv::audit` denial record (ADR 0046
//! §3b: no existence oracle, but the denial is recorded); so does a release
//! of such an id (a no-op) and, through
//! [`SessionManager::audit_foreign_operation`], an operation id another
//! owner's session holds. These records are not counted in
//! `krishiv_auth_denied_total`, which counts `KRISHIV_INSUFFICIENT_PERMISSIONS`
//! refusals only (§8).
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
//! closed: the map lock guards lookups and inserts only. The one exception
//! is each session's statement gate ([`SessionEntry::shared`]), a tokio
//! `RwLock` built to be held across an `await`: `CloneSession` takes it
//! exclusively around the copy (`clone`).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use krishiv_common::PrincipalKind;
use krishiv_common::config::{
    CONNECT_EXECUTE_ABANDONED_TOMBSTONES, CONNECT_MAX_SESSIONS, CONNECT_MAX_SESSIONS_PER_PRINCIPAL,
    CONNECT_SESSION_IDLE_TIMEOUT, CONNECT_SESSION_TOMBSTONES, Config, ConfigValue,
};
use krishiv_common::error::{
    InvalidHandleSessionChanged, InvalidHandleSessionClosed, InvalidHandleSessionNotFound,
    KrishivError, KrishivSessionLimit,
};
use krishiv_session::auth::{Authenticated, Method, Role, Surface};
use krishiv_session::{DataFrame, QueryId, Session, Transport};

use crate::artifacts::SessionArtifacts;
use crate::auth::SessionOwner;
use crate::errors::SessionErrors;
use crate::operation::{Operation, Operations};
use crate::udf::{FunctionStore, SessionPython, UdfHost};

/// Builds the session an authenticated caller gets for a new session id:
/// its principal, and the policy of its role. The transport is the service the
/// session is for (`Connect`, or `FlightSql` through
/// [`SessionManager::get_or_create_with`], P3-T101 §2 H3), so the session can
/// label what it records (its denials count under `Surface::of(transport)`).
pub type SessionFactory =
    dyn Fn(&Authenticated, Transport) -> Result<Session, KrishivError> + Send + Sync;

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
    caller: Authenticated,
    client_user_id: String,
    /// The session id the client sent: what Spark's own messages name (`SessionHolder.sessionId`).
    client_session_id: String,
    operations: Arc<Operations>,
    /// The errors sent on the session, for `FetchErrorDetails`.
    errors: Arc<SessionErrors>,
    /// The session's artifacts (`AddArtifacts`).
    artifacts: Arc<SessionArtifacts>,
    /// The session's Python UDFs (P3-T014): their runtime exists from the first passed `Usage` check on
    /// `System("udf")`; its workers stop when the session is closed.
    python: Arc<SessionPython>,
    /// The DataFrames the session holds for the client (`checkpoint_command`, Spark's
    /// `SessionHolder.dataFrameCache`): id to frame, ended by
    /// `remove_cached_remote_relation_command` or with the session. A frame's rows and files end
    /// when the last frame that reads them is dropped.
    cached_frames: Mutex<HashMap<String, DataFrame>>,
    /// Milliseconds since the manager's epoch at the last lookup.
    last_access: AtomicU64,
    /// Held shared by the RPCs that change the session's state while they
    /// do, exclusively by `CloneSession` around its copy.
    gate: tokio::sync::RwLock<()>,
}

impl SessionEntry {
    /// Holds `frame` under `id` (a checkpoint's result).
    pub(crate) fn cache_frame(&self, id: String, frame: DataFrame) {
        self.cached_frames
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(id, frame);
    }

    /// The frame held under `id`.
    pub(crate) fn cached_frame(&self, id: &str) -> Option<DataFrame> {
        self.cached_frames
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(id)
            .cloned()
    }

    /// Lets go of the frame held under `id`; an id that is not held is not an error (Spark's
    /// `removeCachedDataFrame`). The frame is dropped outside the lock.
    pub(crate) fn remove_cached_frame(&self, id: &str) {
        let removed = self
            .cached_frames
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(id);
        drop(removed);
    }

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

    /// The session id the client sent (Spark's `sessionId`, as opposed to the server-side one).
    #[must_use]
    pub fn client_session_id(&self) -> &str {
        &self.client_session_id
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

    /// The session's Python UDFs.
    pub(crate) fn python(&self) -> &SessionPython {
        self.python.as_ref()
    }

    /// Closes the session's operations, then the session (its queries
    /// cancelled), then its artifacts (their directory removed), then stops
    /// its Python UDF workers. Returns the operations it closed.
    fn close(&self) -> Vec<Arc<Operation>> {
        let ops = self.operations.close_all();
        self.session.close();
        self.artifacts.close();
        self.python.release();
        // The held DataFrames go with the session, whoever still holds the entry; dropped
        // outside the lock.
        let held = std::mem::take(
            &mut *self
                .cached_frames
                .lock()
                .unwrap_or_else(PoisonError::into_inner),
        );
        drop(held);
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
    owner: SessionOwner,
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
    /// The owner holding each live session id.
    owners: HashMap<String, SessionOwner>,
    per_principal: HashMap<SessionOwner, usize>,
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
    /// The server's Python UDF workers; `None` refuses Python UDFs.
    udfs: Option<Arc<dyn UdfHost>>,
    /// The catalog of persisted Python UDFs; `None`: nothing is persisted.
    function_store: Option<Arc<dyn FunctionStore>>,
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
            udfs: None,
            function_store: None,
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
        Self::new(Box::new(move |_, _| Ok(base.new_session())), limits)
    }

    /// Runs the Python UDFs of every session this manager builds on `host`'s workers (the server's one
    /// pool, `krishiv_udf_pool::Pool` on Linux): each session gets a runtime of its own, keyed by its
    /// principal and server-side id, once it passes the `Usage` check on `System("udf")`. Without a host,
    /// Python UDFs are refused (`KRISHIV_UNSUPPORTED_FEATURE`).
    #[must_use]
    pub fn with_udf_host(mut self, host: Arc<dyn UdfHost>) -> Self {
        self.udfs = Some(host);
        self
    }

    /// Keeps the Python UDFs a client persists (`krishiv.v1.PersistPythonFunction`) in `store`, per tenant, and
    /// loads a tenant's into its sessions (at a session's first request that passes the UDF privilege check).
    #[must_use]
    pub fn with_function_store(mut self, store: Arc<dyn FunctionStore>) -> Self {
        self.function_store = Some(store);
        self
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

    /// A new entry for `session`, built for `caller`, not yet in the map.
    /// `system.sessions` shows `transport` and the client's `user_id` as
    /// reported (ADR 0046 §5: recorded, unverified, never the identity).
    fn new_entry(
        &self,
        caller: &Authenticated,
        session: Session,
        (client_user_id, client_session_id): (&str, &str),
        python: SessionPython,
        transport: Transport,
    ) -> Arc<SessionEntry> {
        let python = Arc::new(python);
        session.set_persisted_functions(Arc::new(crate::udf::SqlPersist(Arc::clone(&python))));
        session.set_client_info(transport, None, Some(client_user_id), Some(session.id()));
        Arc::new(SessionEntry {
            errors: Arc::new(SessionErrors::new(session.conf())),
            artifacts: Arc::default(),
            session,
            caller: caller.clone(),
            client_user_id: client_user_id.to_owned(),
            client_session_id: client_session_id.to_owned(),
            operations: Arc::default(),
            python,
            cached_frames: Mutex::default(),
            last_access: AtomicU64::new(self.now_ms()),
            gate: tokio::sync::RwLock::new(()),
        })
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
        self.get_or_create_with(
            caller,
            session_id,
            observed,
            client_user_id,
            Transport::Connect,
        )
    }

    /// [`Self::get_or_create`] for a session of `transport` (P3-T101 §2 H3):
    /// a new session records `transport` in `system.sessions` and is built by
    /// the factory for it; the ADR 0046 §3b record of an id another owner holds
    /// names `Surface::of(transport)`. An existing session is returned
    /// whatever transport created it.
    ///
    /// # Errors
    /// As [`Self::get_or_create`].
    pub fn get_or_create_with(
        &self,
        caller: &Authenticated,
        session_id: &str,
        observed: Option<&str>,
        client_user_id: &str,
        transport: Transport,
    ) -> Result<Arc<SessionEntry>, KrishivError> {
        let surface = Surface::of(transport).unwrap_or(Surface::Connect);
        let key = Self::key(caller, session_id);
        if let Some(entry) = self.lookup(&key, caller, observed, surface)? {
            return Ok(entry);
        }
        // Validate before building: Spark's `validateSessionCreate`.
        QueryId::parse(session_id)?;
        if let Some(existing) = self.check_create(&key, caller, surface)? {
            Self::check_observed(&key, &existing, observed)?;
            return Ok(existing);
        }
        // Built without the lock; a racing create of the same key wins and
        // this one is closed unused.
        let session = (self.factory)(caller, transport)?;
        let python = SessionPython::new(
            self.udfs.as_ref(),
            &principal_id(caller),
            caller.principal.tenant.as_deref(),
            caller.method != Method::None,
            &session,
        )
        .with_store(self.function_store.as_ref());
        let entry = self.new_entry(
            caller,
            session,
            (client_user_id, session_id),
            python,
            transport,
        );
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
                        *inner.per_principal.entry(key.owner.cap_key()).or_insert(0) += 1;
                        Ok(None)
                    }
                }
            }
        };
        Self::close_retired(retired);
        match raced {
            Err(e) => {
                // Built, then refused by a check that changed meanwhile.
                entry.close();
                Err(audit_foreign(caller, "session", surface, e))
            }
            Ok(Some(existing)) => {
                entry.close();
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
        self.get_observed(caller, session_id, observed, Surface::Connect)
    }

    /// [`Self::get`] for a request of `surface`, without a server-side id to
    /// compare (P3-T101 §2 H3): the ADR 0046 §3b record of an id another owner
    /// holds names `surface`.
    ///
    /// # Errors
    /// As [`Self::get`].
    pub fn get_with(
        &self,
        caller: &Authenticated,
        session_id: &str,
        surface: Surface,
    ) -> Result<Arc<SessionEntry>, KrishivError> {
        self.get_observed(caller, session_id, None, surface)
    }

    fn get_observed(
        &self,
        caller: &Authenticated,
        session_id: &str,
        observed: Option<&str>,
        surface: Surface,
    ) -> Result<Arc<SessionEntry>, KrishivError> {
        let key = Self::key(caller, session_id);
        if let Some(entry) = self.lookup(&key, caller, observed, surface)? {
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
            let Some(entry) = Self::remove_locked(&mut inner, &key) else {
                let foreign = Self::foreign_locked(&inner, &key);
                drop(inner);
                if foreign {
                    write_foreign_record(caller, "session", Surface::Connect);
                }
                return None;
            };
            if !allow_reconnect {
                inner.closed.insert(key.clone(), self.limits.tombstones);
            }
            entry
        };
        Some(self.close_released(&key, &entry, false))
    }

    /// Whether `owner` has the live session `session_id` (P3-T101 §2 H5). Not an
    /// access: unlike [`Self::get_if_present`] it does not stamp the session, so a
    /// caller that polls (Flight SQL's sweep) keeps no session from the idle
    /// timeout. Whatever role the session was built with.
    #[must_use]
    pub fn is_live(&self, owner: &SessionOwner, session_id: &str) -> bool {
        let key = Key {
            owner: owner.clone(),
            session_id: session_id.to_owned(),
        };
        self.lock().live.contains_key(&key)
    }

    /// Releases `owner`'s session `session_id` as [`Self::release`] does, for a
    /// caller outside any request (Flight SQL's sweep and provisional sessions,
    /// P3-T101 §2 H5, §3.2): keyed by the owner, it needs no `Authenticated`,
    /// writes no foreign-id record for an id it does not find, and tombstones as
    /// abandoned only the reattachable operations it cancels (one that is not
    /// can never be reattached). Returns the released session's server-side id.
    pub fn release_owned(
        &self,
        owner: &SessionOwner,
        session_id: &str,
        allow_reconnect: bool,
    ) -> Option<String> {
        let key = Key {
            owner: owner.clone(),
            session_id: session_id.to_owned(),
        };
        let entry = {
            let mut inner = self.lock();
            let entry = Self::remove_locked(&mut inner, &key)?;
            if !allow_reconnect {
                inner.closed.insert(key.clone(), self.limits.tombstones);
            }
            entry
        };
        Some(self.close_released(&key, &entry, true))
    }

    /// Closes `entry`, which left the map as `key`, and tombstones the operations
    /// it cancelled as abandoned (Spark's `removeAllExecutionsForSession`): all of
    /// them, or with `reattachable_only` those a client could reattach to.
    /// Returns its server-side id.
    fn close_released(&self, key: &Key, entry: &SessionEntry, reattachable_only: bool) -> String {
        let id = entry.server_side_id().to_owned();
        let ops: Vec<_> = entry
            .close()
            .into_iter()
            .filter(|op| !reattachable_only || op.reattachable())
            .collect();
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
        id
    }

    /// Whether operation `operation_id` of session `session_id` was
    /// abandoned (a reattach answers `INVALID_HANDLE.OPERATION_ABANDONED`).
    #[must_use]
    pub fn is_abandoned(
        &self,
        caller: &Authenticated,
        session_id: &str,
        operation_id: &str,
    ) -> bool {
        let Ok(operation) = QueryId::parse(operation_id) else {
            return false;
        };
        let key = OpKey {
            session: Self::key(caller, session_id),
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
                if self.abandon(&key, &entry, &op, timeout) {
                    abandoned += 1;
                }
            }
        }
        abandoned
    }

    /// Abandons `op` of session `key` if it has had no stream attached for
    /// `timeout` and is still registered: under the manager's lock it is
    /// interrupted (its final state decided), removed (which records that
    /// state) and tombstoned; it is closed after the lock is released, so
    /// its buffered responses are not freed under it. One a release removed first
    /// is left alone (Spark's `removeExecuteHolder` returns before the
    /// tombstone for a holder no longer in `executions`), so its reattach
    /// answers `OPERATION_NOT_FOUND`. Whether it was abandoned.
    fn abandon(
        &self,
        key: &Key,
        entry: &SessionEntry,
        op: &Arc<Operation>,
        timeout: Duration,
    ) -> bool {
        let expired = op
            .replay()
            .detached_since()
            .is_some_and(|since| since.elapsed() >= timeout);
        if !expired {
            return false;
        }
        {
            // The close, the removal (under the operations' lock, as a
            // release takes it) and the tombstone are one step under the
            // manager's lock: a racing reattach that no longer finds the
            // operation reads `is_abandoned` after it, so it finds it
            // abandoned rather than unknown (Spark's lines 158-166). One a
            // release removed first is left alone. Interrupted before it is
            // removed, so the state it is remembered with is the one the
            // compare-and-swap left (a query that finished just before stays
            // `SUCCEEDED`). Lock order: manager, then operations (as
            // `sweep`).
            let mut inner = self.lock();
            let registered = entry
                .operations()
                .get(op.id().as_str())
                .is_some_and(|o| Arc::ptr_eq(&o, op));
            if !registered {
                return false;
            }
            op.cancel();
            if !entry.operations().remove(op) {
                return false;
            }
            #[cfg(test)]
            crate::execute::hooks::abandon_recorded(op);
            inner.abandoned.insert(
                OpKey {
                    session: key.clone(),
                    operation: op.id().clone(),
                },
                self.limits.abandoned_tombstones,
            );
        }
        // A reattach between the expiry check and this close is closed with
        // it (its retry sees OPERATION_ABANDONED): Spark's
        // `periodicMaintenance` has the same window.
        op.close();
        tracing::info!(
            target: "krishiv::connect",
            operation = op.id().as_str(),
            "abandoned a detached Connect operation"
        );
        true
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

    /// The running queries plus the active operations (still running, or
    /// read by an attached client) of all live sessions: zero once nothing
    /// is left to finish (shutdown waits for it). A detached operation that
    /// has ended does not count; closing its session releases it.
    #[must_use]
    pub fn busy(&self) -> usize {
        let entries: Vec<Arc<SessionEntry>> = self.lock().live.values().map(Arc::clone).collect();
        entries
            .iter()
            .map(|e| e.session.running_queries() + e.operations.active())
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
        surface: Surface,
    ) -> Result<Option<Arc<SessionEntry>>, KrishivError> {
        let mut retired = None;
        let found = {
            let mut inner = self.lock();
            match self.live_locked(&mut inner, key, caller, &mut retired) {
                Some(e) => Ok(Some(e)),
                // Held by another owner: refused as unknown.
                None if Self::foreign_locked(&inner, key) => {
                    Err(KrishivError::from(InvalidHandleSessionNotFound {
                        handle: key.session_id.clone(),
                    }))
                }
                None => Ok(None),
            }
        };
        Self::close_retired(retired);
        let Some(entry) = found.map_err(|e| audit_foreign(caller, "session", surface, e))? else {
            return Ok(None);
        };
        Self::check_observed(key, &entry, observed)?;
        Ok(Some(entry))
    }

    /// Whether another owner holds `key`'s session id.
    fn foreign_locked(inner: &Inner, key: &Key) -> bool {
        inner
            .owners
            .get(&key.session_id)
            .is_some_and(|o| *o != key.owner)
    }

    /// Writes the denial record of ADR 0046 §3b when `operation_id`, unknown in
    /// `caller`'s session, is a live operation of another owner's session (the
    /// caller is answered as for an unknown id either way: `ReattachExecute`,
    /// `ReleaseExecute`, `Interrupt` by id). It looks through every live
    /// session (at most `krishiv.connect.maxSessions`), and runs on every
    /// unknown operation id, which includes a normal client's reattach or
    /// release after its operation finished.
    pub fn audit_foreign_operation(&self, caller: &Authenticated, operation_id: &str) {
        let owner = SessionOwner::of(caller);
        let others: Vec<Arc<SessionEntry>> = self
            .lock()
            .live
            .iter()
            .filter(|(k, _)| k.owner != owner)
            .map(|(_, e)| Arc::clone(e))
            .collect();
        if others
            .iter()
            .any(|e| e.operations().get(operation_id).is_some())
        {
            write_foreign_record(caller, "operation", Surface::Connect);
        }
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
        surface: Surface,
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
        checked.map_err(|e| audit_foreign(caller, "session", surface, e))
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
        if Self::foreign_locked(inner, key) {
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
            .get(&key.owner.cap_key())
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
        let cap = key.owner.cap_key();
        if let Some(n) = inner.per_principal.get_mut(&cap) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                inner.per_principal.remove(&cap);
            }
        }
        Some(entry)
    }
}

mod clone;

/// `error` unchanged, after the ADR 0046 §3b denial record when it is the
/// `INVALID_HANDLE.SESSION_NOT_FOUND` of an id another owner holds (the only
/// `SESSION_NOT_FOUND` [`SessionManager::lookup`] and the create checks give),
/// on a request of `surface`.
fn audit_foreign(
    caller: &Authenticated,
    handle: &'static str,
    surface: Surface,
    error: KrishivError,
) -> KrishivError {
    if error.error_class().name() == "INVALID_HANDLE.SESSION_NOT_FOUND" {
        write_foreign_record(caller, handle, surface);
    }
    error
}

/// The `krishiv::audit` record of a refused id another principal holds: who
/// asked, on which surface, and what kind of id, never the id's owner.
fn write_foreign_record(caller: &Authenticated, handle: &'static str, surface: Surface) {
    tracing::warn!(
        target: "krishiv::audit",
        principal = %caller.principal.name,
        method = caller.method.as_str(),
        role = caller.role.map_or("none", Role::as_str),
        delegated_by = caller.principal.delegated_by.as_deref(),
        surface = surface.as_str(),
        handle,
        "refused an id another principal holds, answered as unknown"
    );
}

#[cfg(test)]
mod tests;

/// The principal as the Python UDF runtime and the function catalog tell it apart: a user and a service of
/// one name are two principals (the persisted-function owner and the untenanted key).
pub(crate) fn principal_id(caller: &Authenticated) -> String {
    let kind = match caller.principal.kind {
        PrincipalKind::User => "user",
        PrincipalKind::Service => "service",
        _ => "other",
    };
    format!("{kind}:{}", caller.principal.name)
}
