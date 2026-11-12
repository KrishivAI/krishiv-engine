// SPDX-License-Identifier: Apache-2.0
//! [`EngineRuntime`]: the part of the engine one process shares (design
//! P2-E1). It owns what the configuration docs call process-wide: the
//! executor, the memory root, the spill directory, the metrics registry, the
//! format registry, the query runner and the running-query registry. A
//! [`Session`](crate::Session) is a light child of one runtime: its own
//! configuration overlay, catalog view, temporary views, variables, UDFs,
//! principal and access policy.
//!
//! `Session::builder().build()` with no runtime builds a private runtime
//! from the builder's configuration, so a session built alone behaves as it
//! always did. A host serving many sessions builds one runtime and hands it
//! to each session ([`SessionBuilder::runtime`](crate::SessionBuilder::runtime)).

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use krishiv_common::config::{Config, SYSTEM_QUERIES_CAPACITY, lookup};
use krishiv_common::error::{CannotModifyStaticConfig, KrishivError};
use krishiv_exec::{
    Executor, ExecutorConfig, MetricsRegistry, QueryContext, QueryRunner, TokioExecutor,
    WorkloadClass,
};
use krishiv_io::{FormatRegistry, IoConfig, IoScheduler, StoreId};
use krishiv_memory::{MemoryConfig, MemoryPool, SpillManager, TreePool};
use object_store::ObjectStore;

use crate::QueryId;
use crate::cancel::{self, CancelReason, Deadlines};
use crate::history::QueryHistory;
use crate::result_cache::ResultCache;
use crate::store::{InMemoryCatalogStore, Warehouse};

/// The process-wide part of the engine, shared by the sessions built on it
/// (always behind an `Arc`). It ends when the last session, the last result
/// stream and the last `Arc<EngineRuntime>` are dropped.
///
/// ```
/// use std::sync::Arc;
/// use krishiv_session::{EngineRuntime, Session};
///
/// let rt = EngineRuntime::builder().config("krishiv.memory.limit", "512m").build()?;
/// let a = Session::builder().runtime(Arc::clone(&rt)).build()?;
/// let b = a.new_session();
/// assert!(Arc::ptr_eq(a.runtime(), b.runtime()));
/// # Ok::<(), krishiv_common::error::KrishivError>(())
/// ```
pub struct EngineRuntime {
    /// The configuration the runtime was built with, as given: every session
    /// of the runtime starts from it, and its runtime-owned keys are fixed.
    pairs: Vec<(String, String)>,
    /// `pairs`, validated: what the runtime's own components are built from,
    /// and the configuration a child session starts from.
    pub(crate) config: Arc<Config>,
    pub(crate) executor: Arc<dyn Executor>,
    pub(crate) runner: QueryRunner,
    /// The memory root: `krishiv.memory.limit` bounds the whole process.
    pub(crate) memory_tree: Arc<TreePool>,
    /// The engine's metrics registry: one scrape per process.
    pub(crate) metrics: Arc<MetricsRegistry>,
    /// The spill directory, opened by the first query that spills.
    pub(crate) spill: Arc<Mutex<Option<SpillManager>>>,
    pub(crate) formats: Arc<FormatRegistry>,
    /// The worker-local index of the next query: unique across the
    /// runtime's sessions, so they share one runner.
    query_index: AtomicU64,
    /// The number of the next session built on the runtime.
    session_seq: AtomicU64,
    /// The `system.queries` rows of every session.
    pub(crate) history: Arc<QueryHistory>,
    /// The result cache: the process-wide one for a private runtime (sessions
    /// built alone share it, as before), a store of its own for a supplied
    /// runtime (ADR 0037 §6 holds per runtime).
    pub(crate) results: Arc<ResultCache>,
    /// The runtime's invalidation hub: a table change from any session
    /// reaches the result cache and every session's relation cache.
    pub(crate) invalidation: Arc<crate::cache::Invalidation>,
    /// The permanent catalog and the managed tables' warehouse, shared by
    /// every session of the runtime (Spark's `SharedState`).
    pub(crate) catalog_store: Arc<InMemoryCatalogStore>,
    pub(crate) warehouse: Arc<Warehouse>,
    /// The statement-timeout timer, started by the first query with a limit.
    deadlines: OnceLock<Deadlines>,
    /// The queries running now, and the closed sessions.
    live: Mutex<LiveSet>,
    /// The sessions, live and recently ended (`system.sessions`).
    pub(crate) sessions: Arc<crate::sessions::SessionRegistry>,
    /// The I/O scheduler: pacing, the file-metadata and block caches and
    /// the local store, one for every session (design P2-E1 R3). Stores built
    /// from a session's `fs.*` settings carry a fingerprint of them in their
    /// id, so sessions with different settings never share a store or a cache
    /// entry (`krishiv_store::StoreRegistry`).
    pub(crate) io: Arc<IoScheduler>,
    /// Whether the runtime was built for one session that owns it (the
    /// default builder), or supplied by the host.
    private: bool,
}

/// The running queries by worker-local index, and the sessions closed
/// meanwhile (one lock: a query never registers on a closed session).
#[derive(Default)]
struct LiveSet {
    queries: HashMap<u64, Live>,
    closed: HashSet<u64>,
}

/// A query running on the runtime, from its `system.queries` row on.
struct Live {
    /// The number of the session that runs it, and its id.
    session: u64,
    session_id: Arc<str>,
    /// Its id in `system.queries`.
    id: Arc<str>,
    /// `None` until the query's context exists ([`EngineRuntime::attach`]).
    query: Option<Weak<QueryContext>>,
    /// A cancellation that came before the context: adopted at `attach`.
    pending: Option<CancelReason>,
}

/// How a reserved query goes on ([`EngineRuntime::attach`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Attached {
    /// It runs.
    Run,
    /// It was cancelled before its context existed: it ends at once.
    Cancelled(CancelReason),
    /// Its session closed meanwhile.
    Closed,
}

impl fmt::Debug for EngineRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EngineRuntime")
            .field("config", &self.pairs.len())
            .field("private", &self.private)
            .finish_non_exhaustive()
    }
}

/// Builds an [`EngineRuntime`].
#[must_use]
pub struct EngineRuntimeBuilder {
    pairs: Vec<(String, String)>,
    executor: Option<Arc<dyn Executor>>,
}

impl fmt::Debug for EngineRuntimeBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EngineRuntimeBuilder")
            .field("config", &self.pairs.len())
            .finish_non_exhaustive()
    }
}

impl EngineRuntimeBuilder {
    /// Sets a configuration entry. Runtime-owned keys (cluster scope, and
    /// the `krishiv.memory.*`, `krishiv.spill.*` and `krishiv.io.*` keys)
    /// configure the runtime; any other key is the starting value of every
    /// session built on it, which a session may still change.
    ///
    /// ```
    /// let rt = krishiv_session::EngineRuntime::builder()
    ///     .config("krishiv.memory.limit", "1g")
    ///     .config("spark.sql.ansi.enabled", "false")
    ///     .build()?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.pairs.push((key.into(), value.into()));
        self
    }

    /// Runs every query of the runtime on `executor` (default: a
    /// [`TokioExecutor`] of the runtime's own).
    pub fn executor(mut self, executor: Arc<dyn Executor>) -> Self {
        self.executor = Some(executor);
        self
    }

    /// The runtime.
    ///
    /// # Errors
    /// Configuration errors; `INTERNAL_ERROR` if the executor's runtimes
    /// cannot start.
    pub fn build(self) -> Result<Arc<EngineRuntime>, KrishivError> {
        EngineRuntime::new(self.pairs, self.executor, false)
    }
}

impl EngineRuntime {
    /// A builder with the defaults.
    pub fn builder() -> EngineRuntimeBuilder {
        EngineRuntimeBuilder {
            pairs: Vec::new(),
            executor: None,
        }
    }

    /// The private runtime of a session built without one.
    pub(crate) fn private(
        pairs: Vec<(String, String)>,
        executor: Option<Arc<dyn Executor>>,
    ) -> Result<Arc<Self>, KrishivError> {
        Self::new(pairs, executor, true)
    }

    fn new(
        pairs: Vec<(String, String)>,
        executor: Option<Arc<dyn Executor>>,
        private: bool,
    ) -> Result<Arc<Self>, KrishivError> {
        // Diagnostics first: installs the tokio-console subscriber when the
        // feature is on and `KRISHIV_TOKIO_CONSOLE=1`, a no-op otherwise.
        krishiv_exec::maybe_init_tokio_console();
        crate::conf::check_pairs(&pairs)?;
        let (config, _) = Config::from_pairs(pairs.iter().cloned())?;
        let executor: Arc<dyn Executor> = match executor {
            Some(e) => e,
            None => TokioExecutor::new(&ExecutorConfig::default().with_config(&config))?,
        };
        let memory_tree = TreePool::new(&MemoryConfig::from_config(&config));
        let io =
            IoScheduler::with_runtime(IoConfig::from_config(&config), executor.io_handle().clone());
        io.attach_memory(&(Arc::<TreePool>::clone(&memory_tree) as Arc<dyn MemoryPool>));
        #[cfg(unix)]
        io.register_whole_listing_store(
            StoreId::new(crate::session::LOCAL_STORE),
            Arc::new(krishiv_io::SettledLocalFileSystem::new()) as Arc<dyn ObjectStore>,
        );
        #[cfg(not(unix))]
        io.register_store(
            StoreId::new(crate::session::LOCAL_STORE),
            Arc::new(object_store::local::LocalFileSystem::new()) as Arc<dyn ObjectStore>,
        );
        let metrics = Arc::new(MetricsRegistry::new());
        let capacity = config
            .get(&SYSTEM_QUERIES_CAPACITY)
            .and_then(krishiv_common::config::ConfigValue::as_int)
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(crate::history::DEFAULT_CAPACITY);
        let results = if private {
            Arc::clone(crate::result_cache::global())
        } else {
            Arc::new(ResultCache::default())
        };
        let invalidation = crate::cache::Invalidation::new(Arc::clone(&results));
        let runtime = Self {
            pairs,
            config: Arc::new(config),
            runner: QueryRunner::new(Arc::clone(&executor)),
            executor,
            memory_tree,
            metrics,
            spill: Arc::new(Mutex::new(None)),
            formats: Arc::new(FormatRegistry::with_defaults()),
            query_index: AtomicU64::new(0),
            session_seq: AtomicU64::new(0),
            history: Arc::new(QueryHistory::new(capacity)),
            catalog_store: Arc::new(InMemoryCatalogStore::with_hub(Arc::clone(&invalidation))),
            invalidation,
            warehouse: Arc::new(Warehouse::default()),
            results,
            deadlines: OnceLock::new(),
            live: Mutex::new(LiveSet::default()),
            sessions: Arc::default(),
            io,
            private,
        };
        runtime.register_gauges()?;
        Ok(Arc::new(runtime))
    }

    /// The memory root every session's pool and query hangs off.
    #[must_use]
    pub fn memory(&self) -> Arc<dyn MemoryPool> {
        Arc::<TreePool>::clone(&self.memory_tree)
    }

    /// The executor every query of the runtime runs on.
    #[must_use]
    pub fn executor(&self) -> &Arc<dyn Executor> {
        &self.executor
    }

    /// The engine's metrics snapshot (every session of the runtime reads
    /// the same one).
    #[must_use]
    pub fn metrics(&self) -> krishiv_exec::MetricsSnapshot {
        self.metrics.snapshot()
    }

    /// The process-wide I/O counters of this runtime's scheduler: requests,
    /// bytes and the read-latency histogram, every session's queries
    /// together (`krishiv_io::IoCounters`; a query's own are in its
    /// `system.queries` row).
    #[must_use]
    pub fn io_totals(&self) -> &krishiv_io::IoCounters {
        self.io.totals()
    }

    /// The configuration a session of this runtime starts from: the
    /// runtime's pairs, then the session's. A runtime-owned key among the
    /// session's pairs fails with `CANNOT_MODIFY_STATIC_CONFIG`, as `SET` of
    /// one does (a private runtime's session passes none: its builder's
    /// pairs are the runtime's).
    pub(crate) fn session_config(
        &self,
        session_pairs: Vec<(String, String)>,
    ) -> Result<Config, KrishivError> {
        if let Some((key, _)) = session_pairs.iter().find(|(k, _)| runtime_owned(k)) {
            return Err(CannotModifyStaticConfig {
                key: format!("\"{key}\""),
            }
            .into());
        }
        crate::conf::check_pairs(&session_pairs)?;
        let pairs = self.pairs.iter().cloned().chain(session_pairs);
        Ok(Config::from_pairs(pairs)?.0)
    }

    /// The number of a new session of this runtime.
    pub(crate) fn next_session(&self) -> u64 {
        self.session_seq.fetch_add(1, Ordering::Relaxed)
    }

    /// A worker-local index outside the registry (tests that run a task
    /// directly, with no `system.queries` row).
    #[cfg(test)]
    pub(crate) fn next_index(&self) -> u64 {
        self.query_index.fetch_add(1, Ordering::Relaxed)
    }

    /// Reserves the worker-local index of a new query of session `session`
    /// (`session_id`), `system.queries` row `id`, before its context
    /// exists: from here a cancellation of `id` is never lost.
    pub(crate) fn reserve(&self, session: u64, session_id: &str, id: &str) -> u64 {
        let index = self.query_index.fetch_add(1, Ordering::Relaxed);
        let mut live = self.live.lock().unwrap_or_else(PoisonError::into_inner);
        // Entries of ended queries go as new ones arrive.
        live.queries.retain(|_, l| {
            l.query
                .as_ref()
                .is_none_or(|q| q.upgrade().is_some_and(|q| !q.is_finished()))
        });
        live.queries.insert(
            index,
            Live {
                session,
                session_id: Arc::from(session_id),
                id: Arc::from(id),
                query: None,
                pending: None,
            },
        );
        index
    }

    /// Attaches `query`, reserved as `index`, decided under the lock
    /// [`Self::cancel_where`] and [`Self::cancel_session`] take. A deadline
    /// of `limit` from now applies when given.
    pub(crate) fn attach(
        &self,
        index: u64,
        query: &Arc<QueryContext>,
        limit: Option<std::time::Duration>,
    ) -> Attached {
        let attached = {
            let mut live = self.live.lock().unwrap_or_else(PoisonError::into_inner);
            let LiveSet { queries, closed } = &mut *live;
            match queries.get_mut(&index) {
                None => Attached::Closed,
                Some(l) if closed.contains(&l.session) => {
                    queries.remove(&index);
                    Attached::Closed
                }
                Some(l) => {
                    l.query = Some(Arc::downgrade(query));
                    l.pending.map_or(Attached::Run, Attached::Cancelled)
                }
            }
        };
        if attached == Attached::Run
            && let Some(limit) = limit
            && let Some((session_id, id)) = self.ids_of(index)
        {
            self.deadlines
                .get_or_init(|| Deadlines::start(&*self.executor, Arc::clone(&self.history)))
                .add(query, limit, &session_id, &id);
        }
        attached
    }

    /// The session id and row id of the query reserved as `index`.
    fn ids_of(&self, index: u64) -> Option<(Arc<str>, Arc<str>)> {
        let live = self.live.lock().unwrap_or_else(PoisonError::into_inner);
        live.queries
            .get(&index)
            .map(|l| (Arc::clone(&l.session_id), Arc::clone(&l.id)))
    }

    /// Cancels, with `reason`, every running query `pick` selects (by
    /// session number and row id); the ids of those it cancelled. A query
    /// whose context does not exist yet keeps the reason for `attach`.
    pub(crate) fn cancel_where(
        &self,
        reason: CancelReason,
        pick: impl Fn(u64, &str) -> bool,
    ) -> Vec<QueryId> {
        let mut pending = Vec::new();
        let picked: Vec<(Arc<QueryContext>, Arc<str>, Arc<str>)> = {
            let mut live = self.live.lock().unwrap_or_else(PoisonError::into_inner);
            let mut out = Vec::new();
            for l in live.queries.values_mut() {
                if !pick(l.session, &l.id) {
                    continue;
                }
                match l.query.as_ref() {
                    None => {
                        if l.pending.is_none() {
                            l.pending = Some(reason);
                            pending.push(QueryId::minted(l.id.to_string()));
                        }
                    }
                    Some(q) => {
                        if let Some(q) = q.upgrade() {
                            out.push((q, Arc::clone(&l.session_id), Arc::clone(&l.id)));
                        }
                    }
                }
            }
            out
        };
        // Cancelled outside the lock: a cancel wakes the query's tasks.
        let mut ids = pending;
        for (query, session_id, id) in picked {
            if cancel::apply(&query, reason, None, &self.history, &session_id, &id) {
                ids.push(QueryId::minted(id.to_string()));
            }
        }
        ids
    }

    /// Cancels the query `id` of any session of the runtime with `reason`
    /// (an operator's kill, a drain): `KRISHIV_QUERY_KILLED` for `Admin`,
    /// `Drain` and `Quota`, `OPERATION_CANCELED` for `User` and `ClientGone`,
    /// `KRISHIV_QUERY_TIMEOUT` for `Timeout`. Not reachable from SQL (an
    /// admin check belongs in front of it). Whether a running query was
    /// cancelled: `false` for an unknown or ended id. Equal caller ids in
    /// several sessions are all cancelled.
    pub fn cancel(&self, id: &crate::QueryId, reason: CancelReason) -> bool {
        !self
            .cancel_where(reason, |_, q| q == id.as_str())
            .is_empty()
    }

    /// Closes session `session`: cancels its running queries with `reason`
    /// and refuses any it attaches later; the number cancelled.
    pub(crate) fn cancel_session(&self, session: u64, reason: CancelReason) -> usize {
        self.live
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .closed
            .insert(session);
        self.cancel_where(reason, |s, _| s == session).len()
    }

    /// The queries each session is running now, counted in one pass.
    pub(crate) fn running_counts(&self) -> HashMap<u64, i64> {
        let live = self.live.lock().unwrap_or_else(PoisonError::into_inner);
        let mut counts = HashMap::new();
        for l in live.queries.values().filter(|l| {
            l.query
                .as_ref()
                .is_none_or(|q| q.upgrade().is_some_and(|q| !q.is_finished()))
        }) {
            *counts.entry(l.session).or_insert(0) += 1;
        }
        counts
    }

    /// The registry as `system.sessions` reads it.
    pub(crate) fn session_view(self: &Arc<Self>) -> Arc<crate::sessions::SessionView> {
        let runtime = Arc::clone(self);
        Arc::new(crate::sessions::SessionView {
            registry: Arc::clone(&self.sessions),
            running: Box::new(move || runtime.running_counts()),
        })
    }

    /// Ends the live session `session_id` for `end`: its running queries are
    /// cancelled with `reason`, it takes no new statement
    /// (`INVALID_HANDLE.SESSION_CLOSED`) and it moves to the ended ring.
    /// `None` for an id that is not live.
    fn end_session(
        &self,
        session_id: &str,
        reason: CancelReason,
        end: crate::sessions::End,
    ) -> Option<usize> {
        let (number, state) = self.sessions.find(session_id)?;
        // One caller ends it: a second kill, expire or `close` finds the flag set.
        if state.closed.swap(true, std::sync::atomic::Ordering::AcqRel) {
            return None;
        }
        let n = self.cancel_session(number, reason);
        self.sessions.retire(number, || {
            crate::sessions::SessionRow::of(&state, Some(end), 0)
        });
        Some(n)
    }

    /// Ends session `session_id` as an administrator would (T066 `KillSession`,
    /// a drain deadline): with `reason` `Admin` or `Drain` its queries end
    /// `KRISHIV_QUERY_KILLED`, and the session's next call is
    /// `INVALID_HANDLE.SESSION_CLOSED`. The number of queries cancelled, or
    /// `None` for an unknown (or already ended) id. The privilege check and the
    /// audit line belong in front of it, as for [`Self::cancel`].
    pub fn kill_session(&self, session_id: &str, reason: CancelReason) -> Option<usize> {
        let end = match reason {
            CancelReason::Drain => crate::sessions::End::Killed("drain"),
            // A caller ending its own session (T066 `KillSession` of its own): recorded as a release.
            CancelReason::ClientGone => crate::sessions::End::Closed("release_session"),
            _ => crate::sessions::End::Killed("admin_kill"),
        };
        self.end_session(session_id, reason, end)
    }

    /// Ends session `session_id` as idle (the server's sweep): its queries
    /// are cancelled as for a vanished client. As [`Self::kill_session`]
    /// otherwise.
    pub fn expire_session(&self, session_id: &str) -> Option<usize> {
        self.end_session(
            session_id,
            CancelReason::ClientGone,
            crate::sessions::End::Expired,
        )
    }

    /// Whether `session_id` ended and is still remembered (a server answers
    /// `INVALID_HANDLE.SESSION_CLOSED` for it, `SESSION_NOT_FOUND` otherwise).
    pub fn session_ended(&self, session_id: &str) -> bool {
        self.sessions.ended(session_id)
    }

    /// Forgets session `session` once it is gone (its closed mark).
    pub(crate) fn forget_session(&self, session: u64) {
        self.live
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .closed
            .remove(&session);
    }

    /// Registers the engine gauges on the runtime's registry: closures over
    /// the owning components, evaluated at scrape time, never pushed. No
    /// gauge holds the runtime itself, so the runtime can end.
    fn register_gauges(&self) -> Result<(), KrishivError> {
        let metrics = &self.metrics;
        {
            let memory = Arc::clone(&self.memory_tree);
            metrics.add_gauge("krishiv.memory.reserved", Vec::new(), move || {
                u64::try_from(memory.reserved()).ok()
            })?;
        }
        {
            let memory = Arc::clone(&self.memory_tree);
            metrics.add_gauge("krishiv.memory.limit", Vec::new(), move || {
                memory.effective_limit().and_then(|l| u64::try_from(l).ok())
            })?;
        }
        {
            let io = Arc::clone(&self.io);
            metrics.add_gauge(
                "krishiv.cache.size",
                vec![("cache".to_owned(), "metadata".to_owned())],
                move || Some(io.cache().bytes()),
            )?;
        }
        {
            let spill = Arc::clone(&self.spill);
            metrics.add_gauge("krishiv.spill.disk.used", Vec::new(), move || {
                let slot = spill.lock().unwrap_or_else(PoisonError::into_inner);
                Some(slot.as_ref().map_or(0, SpillManager::used_bytes))
            })?;
        }
        metrics.add_gauge(
            "krishiv.build.info",
            vec![
                ("version".to_owned(), env!("CARGO_PKG_VERSION").to_owned()),
                (
                    "git_sha".to_owned(),
                    option_env!("KRISHIV_GIT_SHA")
                        .unwrap_or("unknown")
                        .to_owned(),
                ),
            ],
            || Some(1),
        )?;
        for class in [
            WorkloadClass::Interactive,
            WorkloadClass::Batch,
            WorkloadClass::Background,
            WorkloadClass::Agent,
        ] {
            // A weak handle: a strong one would cycle the registry through its
            // own gauges and leak it (and the spill manager) forever.
            let registry = Arc::downgrade(metrics);
            metrics.add_gauge(
                "krishiv.query.active",
                vec![("class".to_owned(), class.as_str().to_owned())],
                move || registry.upgrade().map(|r| r.active_queries(class)),
            )?;
        }
        Ok(())
    }
}

/// Whether `key` belongs to the runtime rather than to a session: a
/// cluster-scope entry, or one of the keys fixed when the runtime's
/// components are built (`krishiv.memory.*`, `krishiv.spill.*`,
/// `krishiv.io.*`).
pub(crate) fn runtime_owned(key: &str) -> bool {
    lookup(key).is_some_and(|entry| {
        entry.scope == krishiv_common::config::Scope::Cluster
            || crate::conf::BUILD_TIME
                .iter()
                .any(|p| entry.key.starts_with(p))
    })
}

#[cfg(test)]
mod tests {
    use krishiv_exec::{ExecConfig, QueryContext, QueryId, WorkloadClass};

    use super::*;

    /// A query attached after its session closed is refused, and a
    /// cancellation that arrives between a query's row and its context is
    /// kept and applied when the context attaches.
    #[test]
    fn closing_and_cancelling_are_never_lost_in_the_start_window() {
        let rt = EngineRuntime::builder().build().unwrap();
        let query = |n: u64| {
            QueryContext::new(
                QueryId(n),
                WorkloadClass::Interactive,
                ExecConfig::default(),
                rt.memory().child("q", None),
            )
        };
        let i = rt.reserve(7, "s7", "a");
        let a = query(i);
        assert_eq!(rt.attach(i, &a, None), Attached::Run);
        // Reserved, context not attached yet: the cancel waits for it.
        let j = rt.reserve(7, "s7", "b");
        let ids = rt.cancel_where(CancelReason::User, |s, q| s == 7 && q == "b");
        assert_eq!(ids.len(), 1);
        let b = query(j);
        assert_eq!(
            rt.attach(j, &b, None),
            Attached::Cancelled(CancelReason::User)
        );
        // The caller applies it (`Session::attach`).
        assert!(b.cancel_with(CancelReason::User.error("b", None)));
        // Closing cancels the attached query and refuses a later attach.
        let k = rt.reserve(7, "s7", "c");
        // `a`, and `c` (reserved: its reason waits); `b` already ended.
        assert_eq!(rt.cancel_session(7, CancelReason::ClientGone), 2);
        assert!(a.is_cancelled());
        assert_eq!(
            a.check_cancelled().unwrap_err().error_class().name(),
            "OPERATION_CANCELED"
        );
        assert_eq!(rt.attach(k, &query(k), None), Attached::Closed);
        // Other sessions are unaffected; a forgotten session's number is free.
        let m = rt.reserve(8, "s8", "d");
        let d = query(m);
        assert_eq!(rt.attach(m, &d, None), Attached::Run);
        rt.forget_session(7);
        let n = rt.reserve(7, "s7", "e");
        assert_eq!(rt.attach(n, &query(n), None), Attached::Run);
        // A kill by id reaches any session, with its own class.
        assert!(rt.cancel(&crate::QueryId::minted("d".to_owned()), CancelReason::Admin));
        assert!(!rt.cancel(
            &crate::QueryId::minted("zz".to_owned()),
            CancelReason::Admin
        ));
    }

    /// One I/O scheduler for every session of the runtime (design P2-E1 R3).
    #[test]
    fn sessions_share_the_runtimes_io_scheduler() {
        let rt = EngineRuntime::builder().build().unwrap();
        let a = crate::Session::builder()
            .runtime(Arc::clone(&rt))
            .build()
            .unwrap();
        let b = a.new_session();
        assert!(Arc::ptr_eq(&a.state.io, &rt.io));
        assert!(Arc::ptr_eq(&b.state.io, &rt.io));
        let gauges = rt
            .metrics()
            .gauges
            .iter()
            .filter(|g| g.name == "krishiv.cache.size")
            .count();
        assert_eq!(gauges, 1);
    }
}
