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

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};

use krishiv_common::config::{Config, SYSTEM_QUERIES_CAPACITY, lookup};
use krishiv_common::error::{CannotModifyConfig, KrishivError};
use krishiv_exec::{
    Executor, ExecutorConfig, MetricsRegistry, QueryContext, QueryRunner, TokioExecutor,
    WorkloadClass,
};
use krishiv_io::{FormatRegistry, IoScheduler};
use krishiv_memory::{MemoryConfig, MemoryPool, SpillManager, TreePool};

use crate::history::QueryHistory;

/// Where Spark's error messages point for documentation.
const SPARK_DOC_ROOT: &str = "https://spark.apache.org/docs/latest";

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
    /// `pairs`, validated: what the runtime's own components are built from.
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
    /// The queries running now, by worker-local index.
    live: Mutex<HashMap<u64, Live>>,
    /// The I/O schedulers of the runtime's sessions, read by the metadata
    /// cache gauge (weak: a scheduler ends with its session).
    io: Arc<Mutex<Vec<Weak<IoScheduler>>>>,
    /// Whether the runtime was built for one session that owns it (the
    /// default builder), or supplied by the host.
    private: bool,
}

/// A query running on the runtime.
struct Live {
    /// The number of the session that runs it.
    session: u64,
    /// Its id in `system.queries`.
    #[expect(dead_code, reason = "read by cancellation by id (E1 unit R5)")]
    id: String,
    query: Weak<QueryContext>,
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
        let (config, _) = Config::from_pairs(pairs.iter().cloned())?;
        let executor: Arc<dyn Executor> = match executor {
            Some(e) => e,
            None => TokioExecutor::new(&ExecutorConfig::default().with_config(&config))?,
        };
        let memory_tree = TreePool::new(&MemoryConfig::from_config(&config));
        let metrics = Arc::new(MetricsRegistry::new());
        let capacity = config
            .get(&SYSTEM_QUERIES_CAPACITY)
            .and_then(krishiv_common::config::ConfigValue::as_int)
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(crate::history::DEFAULT_CAPACITY);
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
            live: Mutex::new(HashMap::new()),
            io: Arc::new(Mutex::new(Vec::new())),
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

    /// The configuration a session of this runtime starts from: the
    /// runtime's pairs, then the session's. A runtime-owned key among the
    /// session's pairs of a supplied runtime fails, as `SET` of one does.
    pub(crate) fn session_config(
        &self,
        session_pairs: Vec<(String, String)>,
    ) -> Result<Config, KrishivError> {
        if !self.private
            && let Some((key, _)) = session_pairs.iter().find(|(k, _)| runtime_owned(k))
        {
            return Err(CannotModifyConfig {
                key: format!("\"{key}\""),
                docroot: SPARK_DOC_ROOT.to_owned(),
            }
            .into());
        }
        let pairs = if self.private {
            session_pairs
        } else {
            self.pairs.iter().cloned().chain(session_pairs).collect()
        };
        Ok(Config::from_pairs(pairs)?.0)
    }

    /// The number of a new session of this runtime.
    pub(crate) fn next_session(&self) -> u64 {
        self.session_seq.fetch_add(1, Ordering::Relaxed)
    }

    /// The worker-local index of a new query.
    pub(crate) fn next_query_index(&self) -> u64 {
        self.query_index.fetch_add(1, Ordering::Relaxed)
    }

    /// Records `query`, run by session `session` as `system.queries` row
    /// `id`, as running until it ends.
    pub(crate) fn track(&self, session: u64, id: &str, query: &Arc<QueryContext>) {
        let mut live = self.live.lock().unwrap_or_else(PoisonError::into_inner);
        // Entries of ended queries go as new ones arrive.
        live.retain(|_, l| l.query.upgrade().is_some_and(|q| !q.is_finished()));
        live.insert(
            query.id().0,
            Live {
                session,
                id: id.to_owned(),
                query: Arc::downgrade(query),
            },
        );
    }

    /// Cancels every running query of session `session`; the number
    /// cancelled.
    pub(crate) fn cancel_session(&self, session: u64) -> usize {
        let queries: Vec<Arc<QueryContext>> = {
            let mut live = self.live.lock().unwrap_or_else(PoisonError::into_inner);
            let mine: Vec<u64> = live
                .iter()
                .filter(|(_, l)| l.session == session)
                .map(|(k, _)| *k)
                .collect();
            mine.into_iter()
                .filter_map(|k| live.remove(&k))
                .filter_map(|l| l.query.upgrade())
                .collect()
        };
        // Cancelled outside the lock: a cancel wakes the query's tasks.
        let n = queries.iter().filter(|q| !q.is_finished()).count();
        for q in &queries {
            q.cancel();
        }
        n
    }

    /// Makes the metadata cache gauge count `io`'s cache too.
    pub(crate) fn add_io(&self, io: &Arc<IoScheduler>) {
        let mut all = self.io.lock().unwrap_or_else(PoisonError::into_inner);
        all.retain(|w| w.strong_count() > 0);
        all.push(Arc::downgrade(io));
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
                move || {
                    let all = io.lock().unwrap_or_else(PoisonError::into_inner);
                    Some(
                        all.iter()
                            .filter_map(Weak::upgrade)
                            .map(|io| io.cache().bytes())
                            .fold(0, u64::saturating_add),
                    )
                },
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
