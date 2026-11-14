// SPDX-License-Identifier: Apache-2.0
//! A session's place on its engine runtime (design P2-E1): child sessions,
//! closing, and per-statement options. A child of `session` so that it reads
//! the session's private state and keeps `session.rs` from growing
//! (`cargo xtask file-size`).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64};

use krishiv_common::config::{Config, ConfigValue, SESSION_MEMORY_LIMIT};
use krishiv_common::error::KrishivError;
use krishiv_exec::planner::DefaultPlanner;
use krishiv_memory::{MemoryPool, RelationStore};
use krishiv_plan::analyzer::FunctionRegistry;
use krishiv_udf_sdk::UdfLibrary;

use super::{Session, SessionParts, SessionState};
use crate::admission::Unlimited;
use crate::cache::RelationCache;
use crate::catalog::SessionCatalog;
use crate::conf::ConfStore;
use crate::dataframe::DataFrame;
use crate::runtime::{Attached, EngineRuntime};
use crate::sql_functions::SqlFunctions;
use crate::udfs::SessionFunctions;
use crate::variables::SessionVariables;

impl Session {
    /// A session on `runtime` with configuration `config`.
    pub(super) fn on_runtime(
        runtime: Arc<EngineRuntime>,
        config: Config,
        parts: SessionParts,
    ) -> Self {
        let catalog = SessionCatalog::over(
            Arc::clone(&runtime.catalog_store),
            Arc::clone(&runtime.warehouse),
        );
        let seed = Seed {
            conf: ConfStore::new(config),
            catalog,
            variables: SessionVariables::default(),
            udfs: Arc::new(UdfLibrary::new()),
            // Per session, as Spark's `spark.udf.register`: a new session does
            // not see another's Python UDFs.
            python_udfs: crate::python_udfs::PythonUdfLibrary::new(),
            python_runtime: None,
            sql_functions: SqlFunctions::default(),
            relations: RelationStore::new(),
            plan_ids: 0,
            sql_sources: 1,
            cwd: None,
        };
        Self::seeded(runtime, parts, seed)
    }

    /// A session of `parts` on `runtime` starting from `seed`'s state.
    fn seeded(runtime: Arc<EngineRuntime>, parts: SessionParts, seed: Seed) -> Self {
        let config = seed.conf.snapshot();
        let queries = Arc::new(crate::system::QueryLog::new(
            Arc::clone(&runtime.history),
            &parts.principal.name,
        ));
        // Without a session limit the session level is skipped: queries stay
        // children of the root, so `system.memory` shows what it always did.
        let memory: Arc<dyn MemoryPool> = match config
            .get(&SESSION_MEMORY_LIMIT)
            .and_then(ConfigValue::as_size)
        {
            Some(limit) => runtime.memory_tree.tenant(
                &format!("session:{}", queries.session_id()),
                Some(usize::try_from(limit).unwrap_or(usize::MAX)),
            ),
            None => runtime.memory(),
        };
        let cache = RelationCache::on(
            &memory,
            &runtime.invalidation,
            Some(crate::cache::CacheDisk {
                executor: Arc::clone(&runtime.executor),
                spill: Arc::clone(&runtime.spill),
                config: krishiv_memory::SpillConfig::from_config(&runtime.config),
            }),
        );
        let io = Arc::clone(&runtime.io);
        let cwd = seed.cwd.unwrap_or_else(|| {
            std::env::current_dir()
                .ok()
                .and_then(|d| d.to_str().map(str::to_owned))
                .unwrap_or_else(|| "/".to_owned())
        });
        let udfs = seed.udfs;
        let python_udfs = Arc::new(seed.python_udfs);
        let base_functions = parts.functions;
        let functions = match base_functions.clone() {
            Some(f) => Arc::new(
                SessionFunctions::with_registry(f, Arc::clone(&udfs))
                    .with_python(Arc::clone(&python_udfs)),
            ) as Arc<dyn FunctionRegistry>,
            None => Arc::new(
                SessionFunctions::new(Arc::clone(&udfs)).with_python(Arc::clone(&python_udfs)),
            ) as Arc<dyn FunctionRegistry>,
        };
        Session {
            locals: Arc::from([]),
            operation: None,
            deadline: None,
            state: Arc::new(SessionState {
                conf: Arc::new(seed.conf),
                catalog: Arc::new(seed.catalog),
                cache,
                analyzer: krishiv_plan::analyzer::CachingAnalyzer::new(),
                variables: Arc::new(seed.variables),
                profile: parts.profile,
                principal: parts.principal,
                policy: parts.policy,
                functions,
                base_functions,
                udfs,
                python_udfs,
                python_runtime: Mutex::new(seed.python_runtime),
                sql_functions: seed.sql_functions,
                commit_log_probes: Mutex::new(HashMap::new()),
                number: runtime.next_session(),
                closed: AtomicBool::new(false),
                runtime,
                memory,
                relations: seed.relations,
                io,
                cwd,
                planner: Arc::new(DefaultPlanner),
                plan_ids: AtomicI64::new(seed.plan_ids),
                // One session id for traces and `system.queries`.
                session_id: queries.session_id().to_owned(),
                queries,
                admission: parts.admission.unwrap_or_else(|| Arc::new(Unlimited)),
                statements: AtomicU64::new(0),
                sql_sources: AtomicU32::new(seed.sql_sources),
                denials: std::sync::OnceLock::new(),
            }),
        }
    }

    /// A new session on the same runtime, run by the same principal under
    /// the same access policy, Spark profile and function registry: fresh
    /// temporary views, variables, UDFs, tags and configuration (the
    /// runtime's, without this session's changes), as Spark's
    /// `newSession()`.
    ///
    /// ```
    /// let a = krishiv_session::Session::builder()
    ///     .config("spark.sql.caseSensitive", "true")
    ///     .build()?;
    /// a.conf().set("spark.sql.ansi.enabled", "false")?;
    /// let b = a.new_session();
    /// assert!(std::sync::Arc::ptr_eq(a.runtime(), b.runtime()));
    /// assert_eq!(b.conf().get("spark.sql.ansi.enabled").as_deref(), Some("true"));
    /// assert_eq!(b.conf().get("spark.sql.caseSensitive").as_deref(), Some("true"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn new_session(&self) -> Self {
        let state = &self.state;
        let runtime = Arc::clone(&state.runtime);
        let parts = SessionParts {
            profile: state.profile,
            principal: Arc::clone(&state.principal),
            policy: Arc::clone(&state.policy),
            functions: state.base_functions.clone(),
            admission: Some(Arc::clone(&state.admission)),
        };
        // The runtime's starting configuration, validated when it was built
        // (for a private runtime, the builder's configuration).
        let config = Config::clone(&runtime.config);
        Self::on_runtime(runtime, config, parts)
    }

    /// [`Self::new_session`] acting for `principal` under `policy` instead of this session's
    /// principal and policy: a server's session for one authenticated caller (P3-T002, ADR
    /// 0046), built from a base session that holds the server's runtime and profile. The
    /// principal and policy are the new session's for life.
    #[must_use]
    pub fn new_session_as(
        &self,
        principal: krishiv_common::principal::Principal,
        policy: Arc<dyn krishiv_plan::analyzer::AccessPolicy>,
    ) -> Self {
        let state = &self.state;
        let runtime = Arc::clone(&state.runtime);
        let parts = SessionParts {
            profile: state.profile,
            principal: Arc::new(principal),
            policy,
            functions: state.base_functions.clone(),
            admission: Some(Arc::clone(&state.admission)),
        };
        let config = Config::clone(&runtime.config);
        Self::on_runtime(runtime, config, parts)
    }

    /// A new session on the same runtime that starts as a copy of this one
    /// (Spark Connect's `CloneSession`, Spark's `SparkSession.cloneSession`):
    /// the same principal, access policy and profile, and copies of the
    /// configuration, temporary tables and views, temporary functions (SQL
    /// bodied and Python), variables, current catalog, namespace and path.
    /// Later changes to either session do not reach the other, except Rust
    /// UDFs (`register_udf`): the two sessions share those registrations (a
    /// recorded difference, design P3-T011a §3.11).
    /// Not copied: cached relations (`CACHE TABLE`, `DataFrame.cache`),
    /// running queries and the results of SQL scripts; the copy has a
    /// session id of its own. `RESET` in the copy restores the runtime's
    /// configuration, without the options this session's builder added, as
    /// Spark's `cloneSession` gives the clone no initial session options.
    /// With `krishiv.session.memoryLimit` (fixed when a session is built)
    /// the copy gets a memory pool of its own with the same limit.
    ///
    /// The copy is not atomic against statements running concurrently on
    /// this session: each piece (configuration, temporary tables and views,
    /// functions, variables) is copied consistently on its own, but a
    /// concurrent DDL statement may land in some pieces and not others.
    /// Callers that need one snapshot serialise the session's statements
    /// around the call (the Connect server's `CloneSession` must do so).
    ///
    /// ```
    /// let s = krishiv_session::Session::new()?;
    /// futures::executor::block_on(s.sql("CREATE TEMPORARY VIEW v AS SELECT 1 AS x"))?;
    /// s.conf().set("my.app.flag", "on")?;
    /// let c = s.clone_session()?;
    /// assert_eq!(c.conf().get("my.app.flag").as_deref(), Some("on"));
    /// futures::executor::block_on(futures::executor::block_on(c.sql("SELECT x FROM v"))?.collect())?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// `INVALID_HANDLE.SESSION_CLOSED` when this session is closed.
    pub fn clone_session(&self) -> Result<Self, KrishivError> {
        self.check_open()?;
        let state = &self.state;
        let parts = SessionParts {
            profile: state.profile,
            principal: Arc::clone(&state.principal),
            policy: Arc::clone(&state.policy),
            functions: state.base_functions.clone(),
            admission: Some(Arc::clone(&state.admission)),
        };
        let python_runtime = state
            .python_runtime
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let sql_functions = state.sql_functions.copy();
        let catalog = state.catalog.copy();
        let seed = Seed {
            conf: state.conf.copy_onto(&state.runtime.config),
            catalog,
            variables: state.variables.copy(),
            // Shared, not copied: the SDK's library has no copy (a public
            // surface), and Connect clients cannot register Rust UDFs.
            udfs: Arc::clone(&state.udfs),
            python_udfs: state.python_udfs.copy(),
            python_runtime,
            sql_functions,
            // Shared: the copied memory tables hold their leases in this
            // store, and the clone's tasks resolve their splits through
            // its own (content-addressed and owner-counted, so sharing is
            // invisible otherwise).
            relations: Arc::clone(&state.relations),
            // Read after the copies: the ids the copied functions, views
            // and DataFrame plans carry are all below these, so the clone
            // never hands out one of them again.
            plan_ids: state.plan_ids.load(Ordering::Acquire),
            sql_sources: state.sql_sources.load(Ordering::Acquire),
            // Relative paths of copied views resolve as they do in the
            // source, wherever the process is now.
            cwd: Some(state.cwd.clone()),
        };
        // A `close` that ran while the copy was made wins: no live clone of
        // a closed session.
        self.check_open()?;
        Ok(Self::seeded(Arc::clone(&state.runtime), parts, seed))
    }

    /// The runtime this session runs on.
    #[must_use]
    pub fn runtime(&self) -> &Arc<EngineRuntime> {
        &self.state.runtime
    }

    /// Closes the session (Spark Connect's `ReleaseSession`): its running
    /// queries are cancelled, so their streams end with
    /// `OPERATION_CANCELED`, and it takes no new statement
    /// (`INVALID_HANDLE.SESSION_CLOSED`). Dropping a session's handles does
    /// not cancel anything: an open result stream keeps its session alive.
    /// Returns the number of queries cancelled; closing again is a no-op.
    ///
    /// ```
    /// let s = krishiv_session::Session::new()?;
    /// s.close();
    /// let e = futures::executor::block_on(s.sql("SELECT 1")).unwrap_err();
    /// assert_eq!(e.error_class().name(), "INVALID_HANDLE.SESSION_CLOSED");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn close(&self) -> usize {
        if self.state.closed.swap(true, Ordering::AcqRel) {
            return 0;
        }
        // The session's cached relations go with it: their memory, and the
        // spill scopes of those on disk (once no cancelled query reads them).
        self.state.cache.close();
        // Under the runtime's registry lock: a query that registers after
        // this is refused there (`EngineRuntime::attach`), so none slips
        // between the flag and the cancellation.
        self.state.runtime.cancel_session(self.state.number)
    }

    /// The session's id: a UUIDv7 minted when it was made, the id
    /// `system.queries` and traces carry (Spark Connect's server-side
    /// session id).
    #[must_use]
    pub fn id(&self) -> &str {
        self.state.queries.session_id()
    }

    /// Whether [`Self::close`] ran.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.state.closed.load(Ordering::Acquire)
    }

    /// The number of this session's queries that are running (registered
    /// on the runtime and not yet ended).
    #[must_use]
    pub fn running_queries(&self) -> usize {
        self.state.runtime.running_in_session(self.state.number)
    }

    /// The error a closed session answers with.
    pub(crate) fn closed_error(&self) -> KrishivError {
        krishiv_common::error::InvalidHandleSessionClosed {
            handle: self.state.queries.session_id().to_owned(),
        }
        .into()
    }

    /// `INVALID_HANDLE.SESSION_CLOSED` once [`Self::close`] ran.
    pub(crate) fn check_open(&self) -> Result<(), KrishivError> {
        if self.state.closed.load(Ordering::Acquire) {
            return Err(self.closed_error());
        }
        Ok(())
    }

    /// As [`Self::sql`], with per-statement options: an operation id names
    /// the queries the statement and its DataFrame's actions run
    /// (`system.queries.query_id`), as Spark Connect's `operation_id` does.
    ///
    /// ```
    /// use krishiv_session::{QueryOptions, Session};
    /// let s = Session::new()?;
    /// let id = "00112233-4455-6677-8899-aabbccddeeff";
    /// let df = futures::executor::block_on(s.sql_with("SELECT 1", QueryOptions::new().operation_id(id)))?;
    /// futures::executor::block_on(df.collect())?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// As [`Self::sql`]; `INVALID_HANDLE.FORMAT` for an operation id that is
    /// not a UUID; `INVALID_HANDLE.OPERATION_ALREADY_EXISTS` while another
    /// statement of this session holds the id (its DataFrame or a result
    /// stream of it is alive); `INVALID_HANDLE.OPERATION_ABANDONED` within
    /// 30 minutes of the id's last use ([`crate::QueryOptions::operation_id`]).
    pub async fn sql_with(
        &self,
        query: &str,
        options: crate::QueryOptions,
    ) -> Result<DataFrame, KrishivError> {
        self.with_options(options)?.sql(query).await
    }

    /// A handle of this session that runs its queries under `options` (the
    /// operation id taken now, held while the handle or a clone lives).
    pub(crate) fn with_options(
        &self,
        options: crate::QueryOptions,
    ) -> Result<Session, KrishivError> {
        self.check_open()?;
        let operation = match options.operation_id {
            None => None,
            Some(text) => {
                let id = crate::QueryId::parse(&text)?;
                let taken = self
                    .state
                    .queries
                    .operations()
                    .take(id, crate::system::now_ms())?;
                Some(Arc::new(taken))
            }
        };
        let handle = Session {
            state: Arc::clone(&self.state),
            locals: Arc::clone(&self.locals),
            operation,
            deadline: options.deadline.or(self.deadline),
        };
        Ok(handle)
    }

    /// The operation id of the statement this handle runs, if the caller
    /// gave one.
    pub(crate) fn operation_id(&self) -> Option<&crate::QueryId> {
        self.operation.as_deref().map(crate::history::Operation::id)
    }

    /// Attaches the query reserved for row `id` to the runtime's registry,
    /// with its time limit: the shorter of `krishiv.query.maxExecutionTime`
    /// and the statement's [`crate::QueryOptions::deadline`]. A cancellation
    /// that arrived before the query's context existed is applied now.
    ///
    /// # Errors
    /// `INVALID_HANDLE.SESSION_CLOSED` when the session closed meanwhile.
    pub(crate) fn attach(
        &self,
        query: &Arc<krishiv_exec::QueryContext>,
        id: &str,
        config: &Config,
    ) -> Result<(), KrishivError> {
        let configured = config
            .get(&krishiv_common::config::QUERY_MAX_EXECUTION_TIME)
            .and_then(krishiv_common::config::ConfigValue::as_duration);
        let limit = match (configured, self.deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let runtime = &self.state.runtime;
        match runtime.attach(query.id().0, query, limit) {
            Attached::Run => Ok(()),
            Attached::Closed => Err(self.closed_error()),
            Attached::Cancelled(reason) => {
                crate::cancel::apply(
                    query,
                    reason,
                    None,
                    &runtime.history,
                    self.state.queries.session_id(),
                    id,
                );
                Ok(())
            }
        }
    }

    /// Cancels the running query `id` of this session (Spark Connect's
    /// `Interrupt` by operation id; a SQL script is one such query, ended between its statements
    /// too): its stream ends with
    /// `OPERATION_CANCELED` at its next batch boundary (a query whose
    /// execution has not begun fails as it begins), and
    /// `system.queries.cancel_reason` shows `user`. Whether a running query
    /// was cancelled: `false` for an unknown id, a query that ended or one
    /// of another session.
    ///
    /// ```
    /// let s = krishiv_session::Session::new()?;
    /// let id = krishiv_session::QueryId::parse("00112233-4455-6677-8899-aabbccddeeff")?;
    /// assert!(!s.cancel(&id));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn cancel(&self, id: &crate::QueryId) -> bool {
        let number = self.state.number;
        !self
            .state
            .runtime
            .cancel_where(crate::CancelReason::User, |s, q| {
                s == number && q == id.as_str()
            })
            .is_empty()
    }

    /// Cancels every running query of this session (Spark Connect's
    /// `Interrupt` of type `ALL`) and returns their ids.
    pub fn interrupt_all(&self) -> Vec<crate::QueryId> {
        let number = self.state.number;
        self.state
            .runtime
            .cancel_where(crate::CancelReason::User, |s, _| s == number)
    }
}

impl Session {
    /// Records a statement refused before it ran (`system.queries`), under
    /// the statement's operation id when the caller gave one.
    pub(crate) fn record_refused(
        &self,
        text: Option<&str>,
        class: Option<&'static str>,
        error: &KrishivError,
    ) -> String {
        let policy = crate::system::TextPolicy::from_config(&self.state.conf.snapshot());
        self.state
            .queries
            .record_refused(policy, text, class, error, self.operation_id())
    }
}

/// The state a new session starts from: empty for a new session, copies of
/// another's for a clone ([`Session::clone_session`]).
struct Seed {
    conf: ConfStore,
    catalog: SessionCatalog,
    variables: SessionVariables,
    udfs: Arc<UdfLibrary>,
    python_udfs: crate::python_udfs::PythonUdfLibrary,
    python_runtime: Option<Arc<dyn krishiv_exec::python_eval::UdfRuntime>>,
    sql_functions: SqlFunctions,
    relations: Arc<RelationStore>,
    /// The first DataFrame plan id the session hands out.
    plan_ids: i64,
    /// The first SQL source id the session hands out.
    sql_sources: u32,
    /// The working directory relative paths resolve against; `None` takes
    /// the process's.
    cwd: Option<String>,
}

#[cfg(test)]
mod tests {
    use krishiv_exec::{ExecConfig, QueryContext, QueryId, WorkloadClass};

    use super::*;

    /// A cancel that arrives after a query's row exists and before its
    /// context does is applied as the context attaches: the query fails at
    /// once with the reason's error, and its row names the reason.
    #[test]
    fn a_cancel_before_the_context_fails_the_query_at_once() {
        let s = Session::new().unwrap();
        let id = s
            .state
            .queries
            .begin_as(crate::system::TextPolicy::default(), Some("q"), None, None)
            .unwrap();
        let runtime = &s.state.runtime;
        let index = runtime.reserve(s.state.number, s.state.queries.session_id(), &id);
        let ids = runtime.cancel_where(crate::CancelReason::User, |_, q| q == id);
        assert_eq!(ids.len(), 1);
        let query = QueryContext::new(
            QueryId(index),
            WorkloadClass::Interactive,
            ExecConfig::default(),
            s.state.memory.child("q", None),
        );
        s.attach(&query, &id, &Config::new()).unwrap();
        assert_eq!(
            query.check_cancelled().unwrap_err().error_class().name(),
            "OPERATION_CANCELED"
        );
        let row = s
            .state
            .queries
            .snapshot(false)
            .into_iter()
            .find(|r| r.id == id)
            .unwrap();
        assert_eq!(row.cancel_reason, Some("user"));
        // Attaching on a closed session is refused.
        let late = s
            .state
            .queries
            .begin_as(
                crate::system::TextPolicy::default(),
                Some("late"),
                None,
                None,
            )
            .unwrap();
        let index = runtime.reserve(s.state.number, s.state.queries.session_id(), &late);
        s.close();
        let query = QueryContext::new(
            QueryId(index),
            WorkloadClass::Interactive,
            ExecConfig::default(),
            s.state.memory.child("late", None),
        );
        let e = s.attach(&query, &late, &Config::new()).unwrap_err();
        assert_eq!(e.error_class().name(), "INVALID_HANDLE.SESSION_CLOSED");
    }

    /// Recorded difference (design P3-T011a §3.11): the Rust UDF library is
    /// shared with the clone, not copied, so a `register_udf` on either side
    /// after the clone reaches the other. A copy needs a `krishiv-udf-sdk`
    /// surface; this test changes on purpose when that lands.
    #[test]
    fn a_clone_shares_the_rust_udf_library() {
        let s = Session::new().unwrap();
        let c = s.clone_session().unwrap();
        assert!(Arc::ptr_eq(&s.state.udfs, &c.state.udfs));
    }

    /// The clone's id counters start where the source's are: SQL source ids
    /// and plan ids it hands out never repeat one a copied function or view
    /// carries.
    #[test]
    fn a_clone_continues_the_source_id_counters() {
        let s = Session::new().unwrap();
        for _ in 0..5 {
            let _ = s.next_plan_id();
        }
        futures::executor::block_on(
            s.sql("CREATE TEMPORARY FUNCTION f(x INT) RETURNS INT RETURN x"),
        )
        .unwrap();
        let c = s.clone_session().unwrap();
        assert_eq!(
            c.state.plan_ids.load(Ordering::Relaxed),
            s.state.plan_ids.load(Ordering::Relaxed)
        );
        assert_eq!(
            c.state.sql_sources.load(Ordering::Relaxed),
            s.state.sql_sources.load(Ordering::Relaxed)
        );
        assert!(c.state.sql_sources.load(Ordering::Relaxed) > 1);
    }
}
