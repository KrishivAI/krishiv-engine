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
use krishiv_io::{IoConfig, IoScheduler, StoreId};
use krishiv_memory::{MemoryPool, RelationStore};
use krishiv_plan::analyzer::FunctionRegistry;
use krishiv_udf_sdk::UdfLibrary;
use object_store::ObjectStore;
#[cfg(not(unix))]
use object_store::local::LocalFileSystem;

use super::{LOCAL_STORE, Session, SessionParts, SessionState};
use crate::admission::Unlimited;
use crate::cache::RelationCache;
use crate::catalog::SessionCatalog;
use crate::conf::ConfStore;
use crate::dataframe::DataFrame;
use crate::runtime::EngineRuntime;
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
        let cache = Arc::new(RelationCache::new(&memory));
        let io = IoScheduler::with_runtime(
            IoConfig::from_config(&runtime.config),
            runtime.executor.io_handle().clone(),
        );
        io.attach_memory(&memory);
        #[cfg(unix)]
        io.register_whole_listing_store(
            StoreId::new(LOCAL_STORE),
            Arc::new(krishiv_io::SettledLocalFileSystem::new()) as Arc<dyn ObjectStore>,
        );
        #[cfg(not(unix))]
        io.register_store(
            StoreId::new(LOCAL_STORE),
            Arc::new(LocalFileSystem::new()) as Arc<dyn ObjectStore>,
        );
        runtime.add_io(&io);
        let cwd = std::env::current_dir()
            .ok()
            .and_then(|d| d.to_str().map(str::to_owned))
            .unwrap_or_else(|| "/".to_owned());
        let udfs = Arc::new(UdfLibrary::new());
        let base_functions = parts.functions;
        let functions = match base_functions.clone() {
            Some(f) => Arc::new(SessionFunctions::with_registry(f, Arc::clone(&udfs)))
                as Arc<dyn FunctionRegistry>,
            None => Arc::new(SessionFunctions::new(Arc::clone(&udfs))) as Arc<dyn FunctionRegistry>,
        };
        Session {
            locals: Arc::from([]),
            operation: None,
            state: Arc::new(SessionState {
                conf: Arc::new(ConfStore::new(config)),
                catalog: Arc::new(SessionCatalog::default()),
                cache,
                variables: Arc::new(SessionVariables::default()),
                profile: parts.profile,
                principal: parts.principal,
                policy: parts.policy,
                functions,
                base_functions,
                udfs,
                sql_functions: SqlFunctions::default(),
                commit_log_probes: Mutex::new(HashMap::new()),
                number: runtime.next_session(),
                closed: AtomicBool::new(false),
                runtime,
                memory,
                relations: RelationStore::new(),
                io,
                cwd,
                planner: Arc::new(DefaultPlanner),
                plan_ids: AtomicI64::new(0),
                // One session id for traces and `system.queries`.
                session_id: queries.session_id().to_owned(),
                queries,
                admission: parts.admission.unwrap_or_else(|| Arc::new(Unlimited)),
                statements: AtomicU64::new(0),
                sql_sources: AtomicU32::new(1),
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
        // Under the runtime's registry lock: a query that registers after
        // this is refused there (`EngineRuntime::track`), so none slips
        // between the flag and the cancellation.
        self.state.runtime.cancel_session(self.state.number)
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
        self.check_open()?;
        let operation = match options.operation_id.as_deref() {
            None => None,
            Some(text) => {
                let id = crate::QueryId::parse(text)?;
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
        };
        handle.sql(query).await
    }

    /// The operation id of the statement this handle runs, if the caller
    /// gave one.
    pub(crate) fn operation_id(&self) -> Option<&crate::QueryId> {
        self.operation.as_deref().map(crate::history::Operation::id)
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
        self.state
            .queries
            .record_refused(text, class, error, self.operation_id())
    }
}
