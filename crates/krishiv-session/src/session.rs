// SPDX-License-Identifier: Apache-2.0
//! [`Session`] and [`SessionBuilder`], and the query pipeline every
//! action runs: analyze → optimize → physical plan → run.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::task::Poll;
use tracing::Instrument as _;

pub(crate) use crate::run_site::sql_context;
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::StreamExt;
use futures::stream::BoxStream;
use krishiv_common::config::{Config, ConfigValue, METRICS_ENABLED, PARTITION_OVERWRITE_MODE};
use krishiv_common::error::{
    AllPartitionColumnsNotAllowed, ErrorQueryContext, InternalError, KrishivError,
    KrishivUnsupportedFeature, PartitionColumnNotFoundInSchema,
};
use krishiv_common::names::check_column_name_duplication;
use krishiv_common::redact::RedactedSql;
use krishiv_common::secret::OptionValue;
use krishiv_exec::files::{FileReads, LocalPaths, ObjectPaths, PathLocator as _};
use krishiv_exec::physical::{SpillScopes, WorkerContext, WritePlan};
use krishiv_exec::planner::{PhysicalPlan, PhysicalPlanner, PlanningContext, ProviderStatistics};
use krishiv_exec::{
    ExecConfig, Executor, MetricsSnapshot, QueryContext, QueryId, Scheduler, WorkloadClass,
};
use krishiv_io::{
    DirectoryWrite, IoScheduler, SaveMode, StoreId, StoreRegistry, is_spark_file_source,
    parse_object_url,
};
use krishiv_memory::{MemoryPool, RelationStore, SpillConfig, SpillManager, SpillScope};
use krishiv_plan::analyzer::QueryContext as PlanQueryContext;
use krishiv_plan::analyzer::{
    AccessPolicy, AllowAll, AnalysisContext, AnalysisError, AnalyzedPlan, Analyzer as _,
    DefaultAnalyzer, FunctionRegistry, Principal, Privilege, ResolutionConf, Securable,
    SparkProfile, authorize,
};
use krishiv_plan::{LogicalPlan, Plan, PlanId, PlanNode, ReadSource, SourceId};
use krishiv_sql::ast::Statement;
use krishiv_sql::{Frontend, ParserOptions, StatementClass};
use krishiv_types::DataType;
use krishiv_types::batch::{Batch, Weights};
use krishiv_udf_sdk::{UdfLibrary, guard_call};
use object_store::ObjectStore;
#[cfg(not(unix))]
use object_store::local::LocalFileSystem;

use crate::admission::{AdmissionController, QueryRequest};
use crate::cache::RelationCache;
use crate::catalog::{FileScans, QueryCatalog, QueryCatalogs, SessionCatalog, TempTable};
use crate::catalog::{is_global_key, is_reserved_view_name, reserved_temp_view_name};
use crate::column::Column;
use crate::conf::{self, ConfStore, RuntimeConf};
use crate::dataframe::{DataFrame, DataFrameReader};
use crate::query_fold::{Admitted, MetricStream, PendingFold, QueryFold, QueryStart, QueuedRow};
use crate::runtime::EngineRuntime;
use crate::sql_functions::{QueryFunctions, SqlFunctions};
use crate::variables::SessionVariables;

mod context;
mod lifecycle;
mod procedure_inputs;
mod traits;
#[cfg(feature = "unstable-ir")]
pub use lifecycle::{SqlOutcome, SqlQuery};
#[cfg(test)]
mod run_task;

/// The store id of the local file system.
pub(crate) const LOCAL_STORE: &str = "file";

/// A stream of result batches; dropping it cancels the query.
pub type RecordBatchStream = BoxStream<'static, Result<RecordBatch, KrishivError>>;

/// A session: configuration, temporary tables and views, and the engine
/// that runs its queries. Cheap to clone; clones share everything.
///
/// Actions (`sql`, `collect`, ...) are `async` and run on the session's own
/// executor, so they work from any async runtime or from a plain
/// `futures::executor::block_on`.
///
/// ```
/// use krishiv_session::Session;
///
/// let session = Session::new()?;
/// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![
/// #     arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false),
/// #     arrow_schema::Field::new("b", arrow_schema::DataType::Utf8, true),
/// # ]));
/// # session.register_batches("t", schema, vec![])?;
/// let df = futures::executor::block_on(session.sql("SELECT a AS one FROM t"))?;
/// let schema = futures::executor::block_on(df.schema())?;
/// assert_eq!(schema.field(0).name(), "one");
/// # Ok::<(), krishiv_common::error::KrishivError>(())
/// ```
#[derive(Clone)]
pub struct Session {
    pub(crate) state: Arc<SessionState>,
    /// The local variables of the SQL script this handle runs, outermost frame first. Empty for a
    /// session a user holds: only the handle a script's statements run on has frames, so no other
    /// query on the session ever sees them (`locals.rs`).
    pub(crate) locals: Arc<[krishiv_plan::analyzer::LocalFrame]>,
    /// The operation the queries this handle starts belong to
    /// ([`Session::sql_with`]); `None` for a handle a user holds.
    pub(crate) operation: Option<Arc<crate::history::Operation>>,
    /// The statement's own time limit ([`crate::QueryOptions::deadline`]).
    pub(crate) deadline: Option<std::time::Duration>,
    /// The statement's fixed query start ([`crate::QueryOptions::query_start`]); `None`: each
    /// plan reads the clock.
    pub(crate) query_start: Option<i64>,
    /// The statement is work accepted before a drain ([`crate::QueryOptions::admitted_before_drain`]).
    pub(crate) drain_exempt: bool,
    /// The statement's request tags ([`crate::QueryOptions::tags`]).
    pub(crate) request_tags: Arc<[Arc<str>]>,
    /// The trace of the request that started the statement ([`crate::QueryOptions::inbound_trace`]):
    /// `Some(None)` is a request without a valid one (no trace), `None` means the session's.
    pub(crate) inbound_trace: Option<Option<Arc<crate::trace_context::RemoteParent>>>,
    /// The baggage that came with it (empty without a valid `traceparent`).
    pub(crate) inbound_baggage: Arc<[(String, String)]>,
    /// The authentication of the request that started the statement, for delegated catalogs.
    pub(crate) request_inbound: Option<crate::catalog_registry::Inbound>,
}

pub(crate) struct SessionState {
    pub(crate) conf: Arc<ConfStore>,
    pub(crate) catalog: Arc<SessionCatalog>,
    pub(crate) cache: Arc<RelationCache>,
    /// Analyses the session's queries and keeps the last root's resolution, so a client that grows a plan
    /// step by step does not pay for the whole prefix again (design P3-T013a).
    pub(crate) analyzer: krishiv_plan::analyzer::CachingAnalyzer,
    pub(crate) variables: Arc<SessionVariables>,
    pub(crate) profile: SparkProfile,
    /// Fixed for the session's life: caches and fingerprints never mix
    /// identities (a new identity gets a new session).
    pub(crate) principal: Arc<Principal>,
    /// Decides when each query may start (E5); the default admits at once.
    pub(crate) admission: Arc<dyn AdmissionController>,
    pub(crate) policy: Arc<dyn AccessPolicy>,
    /// The grants provider this session's `GRANT`, `REVOKE` and `SHOW GRANTS` change and read
    /// (P3-T062); `None`: those statements are refused.
    pub(crate) grants: Option<Arc<crate::grants::GrantsPolicy>>,
    pub(crate) functions: Arc<dyn FunctionRegistry>,
    /// The registry the builder supplied, if any (child sessions use it too).
    base_functions: Option<Arc<dyn FunctionRegistry>>,
    /// The session's registered UDFs, shared with the registry above and
    /// the workers running its plans.
    pub(crate) udfs: Arc<UdfLibrary>,
    /// The session's Python UDFs (P2-T059), shared with the registry above:
    /// plans name them by name and version only.
    pub(crate) python_udfs: Arc<crate::python_udfs::PythonUdfLibrary>,
    /// The runtime that evaluates them in this process (set by the Python
    /// facade); without one a plan calling a Python UDF fails to run.
    pub(crate) python_runtime: Mutex<Option<Arc<dyn krishiv_exec::python_eval::UdfRuntime>>>,
    /// Where persisted Python UDFs are kept (a Connect server's catalog); none for a session alone.
    pub(crate) persisted_functions: Mutex<Option<Arc<dyn crate::python_udfs::PersistedFunctions>>>,
    /// The SQL-bodied functions the session defines.
    pub(crate) sql_functions: SqlFunctions,
    /// The process-wide part of the engine this session runs on.
    pub(crate) runtime: Arc<EngineRuntime>,
    /// The session's number on its runtime.
    pub(crate) number: u64,
    /// Set by [`Session::close`]: the session takes no new statement.
    pub(crate) closed: AtomicBool,
    /// What `system.sessions` shows that changes per call.
    pub(crate) activity: crate::sessions::Activity,
    /// The session's memory pool: the runtime's root, or a pool of its own
    /// under the root (`krishiv.session.memoryLimit`).
    pub(crate) memory: Arc<dyn MemoryPool>,
    /// Whether a store creates objects only if absent (the commit log's
    /// probe), by store id; per session, never process-wide.
    pub(crate) commit_log_probes: Mutex<HashMap<String, bool>>,
    pub(crate) io: Arc<IoScheduler>,
    /// The session's content-addressed relations: every memory table's
    /// batches as Arrow IPC, so workers resolve a batch range through it
    /// instead of reading rows from the plan.
    pub(crate) relations: Arc<RelationStore>,
    cwd: String,
    planner: Arc<dyn PhysicalPlanner>,
    plan_ids: AtomicI64,
    /// The queries `system.queries` lists.
    pub(crate) queries: Arc<crate::system::QueryLog>,
    /// The session's id on spans and logs (`telemetry.rs`), and the count of its statements.
    pub(crate) session_id: String,
    statements: AtomicU64,
    /// SQL source ids handed to view lowerings: every `CREATE TEMPORARY
    /// VIEW` lowers its query with an id of its own, so errors raised
    /// while expanding a view point into the view's text, not the outer
    /// statement's (`SourceId(0)` stays the outer statement's).
    sql_sources: AtomicU32,
    /// Where this session counts its denials (a server's sessions, `Session::count_denials`).
    pub(crate) denials: std::sync::OnceLock<(Arc<crate::auth::DeniedCounts>, crate::auth::Surface)>,
    /// How the latest RPC authenticated and its user token, for delegated catalogs (P3-T067).
    pub(crate) inbound: Mutex<Option<crate::catalog_registry::Inbound>>,
}

/// Builds a [`Session`].
///
/// ```
/// use krishiv_session::Session;
///
/// let session = Session::builder()
///     .config("spark.sql.ansi.enabled", "false")
///     .config("krishiv.memory.limit", "512m")
///     .build()?;
/// assert_eq!(session.conf().get("spark.sql.ansi.enabled").as_deref(), Some("false"));
/// # Ok::<(), krishiv_common::error::KrishivError>(())
/// ```
#[must_use]
pub struct SessionBuilder {
    pairs: Vec<(String, String)>,
    profile: SparkProfile,
    principal: Option<Principal>,
    policy: Option<Arc<dyn AccessPolicy>>,
    grants: Option<Arc<crate::grants::GrantsPolicy>>,
    executor: Option<Arc<dyn Executor>>,
    functions: Option<Arc<dyn FunctionRegistry>>,
    runtime: Option<Arc<EngineRuntime>>,
    admission: Option<Arc<dyn AdmissionController>>,
    source: Option<String>,
}

impl fmt::Debug for SessionBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionBuilder")
            .field("config", &self.pairs.len())
            .field("profile", &self.profile)
            .finish_non_exhaustive()
    }
}

impl SessionBuilder {
    /// Sets a configuration entry, static ones included.
    ///
    /// ```
    /// let s = krishiv_session::Session::builder().config("krishiv.exec.batchSize", "1024").build()?;
    /// assert_eq!(s.conf().get("krishiv.exec.batchSize").as_deref(), Some("1024"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn config(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.pairs.push((key.into(), value.into()));
        self
    }

    /// The Spark version whose behaviour queries follow (default 4.3).
    ///
    /// ```
    /// use krishiv_plan::analyzer::SparkProfile;
    /// let s = krishiv_session::Session::builder().profile(SparkProfile::V4_1).build()?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn profile(mut self, profile: SparkProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Who runs the session's queries (default: the local OS user).
    ///
    /// ```
    /// use krishiv_plan::analyzer::{Principal, PrincipalKind};
    /// let s = krishiv_session::Session::builder()
    ///     .principal(Principal::new("etl", PrincipalKind::Service))
    ///     .build()?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn principal(mut self, principal: Principal) -> Self {
        self.principal = Some(principal);
        self
    }

    /// Who the session says its client is, shown as `system.sessions.client_type` and matched by
    /// the `source` selector of a resource group (design P3-T003 §3; Python sets
    /// `krishiv-python`). Client-asserted: a routing hint, never an identity. A front end that
    /// reports its own client type later does not replace it. `new_session` does not inherit it
    /// (a Connect manager's sessions take each client's own type).
    ///
    /// ```
    /// let s = krishiv_session::Session::builder().source("etl-job").build()?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// The admission controller every query asks for a permit before it
    /// runs (default: [`Unlimited`](crate::Unlimited), which admits at once). While it
    /// waits the query's `system.queries` state is `queued`.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use krishiv_session::{Session, Unlimited};
    /// let s = Session::builder().admission_controller(Arc::new(Unlimited)).build()?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn admission_controller(mut self, controller: Arc<dyn AdmissionController>) -> Self {
        self.admission = Some(controller);
        self
    }

    /// The access policy every statement is checked against (default:
    /// [`AllowAll`]): reads during analysis, writes and DDL before they
    /// change anything, all failing closed with
    /// `KRISHIV_INSUFFICIENT_PERMISSIONS`. Like the principal, it is fixed
    /// for the session's life.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use krishiv_session::{DenyAll, Session};
    ///
    /// let s = Session::builder().access_policy(Arc::new(DenyAll)).build()?;
    /// let e = futures::executor::block_on(s.sql("DROP TABLE t")).unwrap_err();
    /// assert_eq!(e.error_class().name(), "KRISHIV_INSUFFICIENT_PERMISSIONS");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn access_policy(mut self, policy: Arc<dyn AccessPolicy>) -> Self {
        self.policy = Some(policy);
        self
    }

    /// The grants provider (P3-T062) the session's `GRANT`, `REVOKE`, `SHOW GRANTS` and role
    /// statements act on, and whose ownership records its `CREATE` and `DROP` keep. It decides
    /// access only as the session's [access policy](Self::access_policy): pass it there too
    /// (wrapped in a `PathPrefixPolicy` in server mode). A builder with grants but no access
    /// policy is refused, so grants are never managed without being enforced (`tests/grants.rs`).
    pub fn grants(mut self, grants: Arc<crate::grants::GrantsPolicy>) -> Self {
        self.grants = Some(grants);
        self
    }

    /// Runs queries on `executor`, shared with other sessions (default: a
    /// [`TokioExecutor`](krishiv_exec::TokioExecutor) of the session's own).
    ///
    /// ```
    /// use krishiv_exec::{ExecutorConfig, TokioExecutor};
    /// let shared = TokioExecutor::new(&ExecutorConfig::default())?;
    /// let a = krishiv_session::Session::builder().executor(shared.clone()).build()?;
    /// let b = krishiv_session::Session::builder().executor(shared).build()?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn executor(mut self, executor: Arc<dyn Executor>) -> Self {
        self.executor = Some(executor);
        self
    }

    /// Builds the session on `runtime`, shared with its other sessions:
    /// one executor, memory root, spill directory and metrics registry for
    /// all of them. Without it the session builds a private runtime of its
    /// own from its configuration.
    ///
    /// The session starts from the runtime's configuration. Runtime-owned
    /// keys (cluster scope, `krishiv.memory.*`, `krishiv.spill.*`,
    /// `krishiv.io.*`) are the runtime's: setting one here fails with
    /// `CANNOT_MODIFY_STATIC_CONFIG`, as `SET` of one does, and so does
    /// [`Self::executor`].
    ///
    /// ```
    /// use std::sync::Arc;
    /// use krishiv_session::{EngineRuntime, Session};
    /// let rt = EngineRuntime::builder().build()?;
    /// let s = Session::builder().runtime(Arc::clone(&rt)).build()?;
    /// let e = Session::builder().runtime(rt).config("krishiv.memory.limit", "1g").build().unwrap_err();
    /// assert_eq!(e.error_class().name(), "CANNOT_MODIFY_STATIC_CONFIG");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn runtime(mut self, runtime: Arc<EngineRuntime>) -> Self {
        self.runtime = Some(runtime);
        self
    }

    /// Resolves functions with `functions` instead of the built-in
    /// registry.
    ///
    /// ```
    /// use krishiv_plan::analyzer::BuiltinFunctions;
    /// let s = krishiv_session::Session::builder()
    ///     .functions(std::sync::Arc::new(BuiltinFunctions::new()))
    ///     .build()?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn functions(mut self, functions: Arc<dyn FunctionRegistry>) -> Self {
        self.functions = Some(functions);
        self
    }

    /// The session.
    ///
    /// # Errors
    /// Configuration errors; `INTERNAL_ERROR` if the executor's runtimes
    /// cannot start.
    pub fn build(self) -> Result<Session, KrishivError> {
        let (runtime, config) = match self.runtime {
            Some(runtime) => {
                if self.executor.is_some() {
                    return Err(krishiv_common::error::CannotModifyStaticConfig {
                        key: "\"executor\"".to_owned(),
                    }
                    .into());
                }
                let config = runtime.session_config(self.pairs)?;
                (runtime, config)
            }
            None => {
                let runtime = EngineRuntime::private(self.pairs, self.executor)?;
                let config = runtime.session_config(Vec::new())?;
                (runtime, config)
            }
        };
        if self.grants.is_some() && self.policy.is_none() {
            return Err(krishiv_common::error::KrishivInvalidConfigValue {
                value: "grants".to_owned(),
                key: "\"krishiv.access.provider\"".to_owned(),
                reason: "a session with grants needs them as its access policy too".to_owned(),
            }
            .into());
        }
        let parts = SessionParts {
            profile: self.profile,
            principal: Arc::new(self.principal.unwrap_or_else(Principal::local_user)),
            policy: self.policy.unwrap_or_else(|| Arc::new(AllowAll)),
            grants: self.grants,
            functions: self.functions,
            admission: self.admission,
        };
        let session = Session::on_runtime(runtime, config, parts);
        if let Some(source) = self.source.as_deref() {
            session.state.activity.set_client(Some(source), None, None);
        }
        Ok(session)
    }
}

/// What a session keeps from its builder, and a child session from its
/// parent ([`Session::new_session`]).
struct SessionParts {
    profile: SparkProfile,
    principal: Arc<Principal>,
    policy: Arc<dyn AccessPolicy>,
    grants: Option<Arc<crate::grants::GrantsPolicy>>,
    functions: Option<Arc<dyn FunctionRegistry>>,
    admission: Option<Arc<dyn AdmissionController>>,
}

/// Whether queries fold into the metrics registry: `krishiv.metrics.enabled`
/// (default false).
fn metrics_enabled(config: &Config) -> bool {
    config
        .get(&METRICS_ENABLED)
        .and_then(ConfigValue::as_bool)
        .unwrap_or(false)
}

impl Session {
    /// A builder with the defaults.
    ///
    /// ```
    /// let session = krishiv_session::Session::builder().build()?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn builder() -> SessionBuilder {
        SessionBuilder {
            pairs: Vec::new(),
            profile: SparkProfile::default(),
            principal: None,
            policy: None,
            grants: None,
            executor: None,
            functions: None,
            runtime: None,
            admission: None,
            source: None,
        }
    }

    /// A session with the defaults.
    ///
    /// ```
    /// let session = krishiv_session::Session::new()?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// As [`SessionBuilder::build`].
    pub fn new() -> Result<Self, KrishivError> {
        Self::builder().build()
    }

    /// Bytes the session's memory pool holds now, across its queries: what
    /// `krishiv.memory.limit` is checked against (for memory measurements).
    #[must_use]
    pub fn memory_reserved(&self) -> usize {
        self.state.memory.reserved()
    }

    /// The runtime configuration.
    ///
    /// ```
    /// let session = krishiv_session::Session::new()?;
    /// assert!(session.conf().get("spark.sql.ansi.enabled").is_some());
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn conf(&self) -> RuntimeConf {
        RuntimeConf::new(Arc::clone(&self.state.conf))
    }

    /// Runs `query` as Spark's `spark.sql` does: parses and analyzes it now
    /// (errors surface here) and returns its rows as a lazy [`DataFrame`].
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![
    /// #     arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false),
    /// #     arrow_schema::Field::new("b", arrow_schema::DataType::Utf8, true),
    /// # ]));
    /// # session.register_batches("t", schema, vec![])?;
    /// let df = futures::executor::block_on(session.sql("SELECT a, b FROM t"))?;
    /// let e = futures::executor::block_on(session.sql("SELECT nope FROM missing")).unwrap_err();
    /// assert_eq!(e.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Spark's parse and analysis errors; `KRISHIV_UNSUPPORTED_FEATURE`
    /// for statements other than queries and commands.
    pub async fn sql(&self, query: &str) -> Result<DataFrame, KrishivError> {
        self.sql_kind(query).await.map(|(df, _)| df)
    }

    /// Runs `query` on the calling task as `sql` would, for text the session built from a
    /// statement it already substituted (`CACHE TABLE t AS query`): `spans` are the substituted
    /// names in `query`, kept while substituting again leaves the text as it is.
    pub(crate) async fn sql_here_spanned(
        &self,
        query: &str,
        spans: Vec<krishiv_sql::ast::Span>,
    ) -> Result<DataFrame, KrishivError> {
        // Boxed: one more nested future level overflows the compiler's query depth in the binaries.
        Box::pin(self.sql_here_inner(query, Arc::default(), spans, None, false))
            .await
            .and_then(Ran::into_frame)
            .map(|(df, _)| df)
    }

    /// [`Self::run_statement`] without the ownership bookkeeping (`grants/exec.rs`).
    pub(crate) async fn run_statement_here(
        &self,
        statement: Statement,
        env: QueryEnv,
        query: &str,
        redacted: &RedactedSql,
    ) -> Result<DataFrame, KrishivError> {
        crate::collation_defaults::refuse_unhonoured(
            &statement,
            self.parser_options().schema_level_collations,
        )?;
        match statement {
            Statement::Query(q) => {
                let plan = self.lower_identified(&q, &env, redacted).map_err(|e| {
                    crate::execute_immediate::unbound_markers(
                        e,
                        query,
                        env.parser.double_quoted_identifiers,
                    )
                })?;
                let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
                df.analyzed_in_with_text(env, Some(redacted)).await?;
                Ok(df)
            }
            Statement::Explain(e) => self.explain_statement(&e, env, query, redacted).await,
            Statement::Set(s) => self.exec_set(&s, redacted).await,
            Statement::Reset(r) => self.exec_reset(&r).await,
            Statement::CreateView(v) => {
                self.exec_create_view(&v, query, &env.identifier_spans)
                    .await
            }
            Statement::DropView(v) => self.exec_drop_view(&v, &env).await,
            Statement::Cache(c) => {
                Box::pin(self.exec_cache(&c, query, &env.identifier_spans)).await
            }
            Statement::Refresh(r) => self.exec_refresh(&r).await,
            Statement::Call(c) => Box::pin(self.exec_call(&c, query)).await,
            Statement::DeclareVariable(v) => self.exec_declare(&v, query, redacted).await,
            Statement::SetVariable(v) => self.exec_set_variable(&v, query, redacted).await,
            Statement::DropVariable(v) => self.exec_drop_variable(&v, redacted).await,
            Statement::ExecuteImmediate(e) => {
                Box::pin(self.exec_execute_immediate(&e, query, redacted, 0)).await
            }
            Statement::CreateFunction(f) => {
                self.exec_create_function(&f, redacted, &env.identifier_spans)
                    .await
            }
            Statement::CreatePersistedFunction(f) => self.exec_create_persisted_function(&f).await,
            Statement::DropFunction(f) => self.exec_drop_function(&f).await,
            Statement::CreateTable(t) => {
                Box::pin(self.exec_create_table(&t, query, &env.identifier_spans))
                    .await
                    .map_err(|e| {
                        crate::execute_immediate::unbound_markers(
                            e,
                            query,
                            env.parser.double_quoted_identifiers,
                        )
                    })
            }
            Statement::CreateTableLike(t) => Box::pin(self.exec_create_table_like(&t, query)).await,
            Statement::DropTable(d) => self.exec_drop_table(&d).await,
            Statement::CreateNamespace(n) => self.exec_create_namespace(&n, query).await,
            Statement::DropNamespace(n) => self.exec_drop_namespace(&n).await,
            Statement::AlterNamespace(a) => self.exec_alter_namespace(&a, query).await,
            Statement::Repair(r) => self.exec_repair(&r).await,
            Statement::Use(u) => self.exec_use(&u).await,
            Statement::Show(s) => Box::pin(self.exec_show(&s, query)).await,
            Statement::Describe(d) => {
                Box::pin(self.exec_describe(&d, query, &env.identifier_spans)).await
            }
            Statement::CreatePermanentView(v) => {
                let ids = (&*env.identifier_spans, &*env.identifier_written);
                self.exec_create_permanent_view(&v, query, ids).await
            }
            Statement::CreateMetricView(v) => {
                if let Some(e) =
                    ran::metric_view_gate_error(&v, self.state.profile, query, redacted)
                {
                    return Err(e);
                }
                self.exec_create_metric_view(&v, query).await
            }
            Statement::DropPermanentView(d) => self.exec_drop_permanent_view(&d).await,
            Statement::AlterView(a) => {
                let ids = (&*env.identifier_spans, &*env.identifier_written);
                self.exec_alter_view(&a, query, ids).await
            }
            Statement::AlterTable(a) => Box::pin(self.exec_alter_table(&a, query)).await,
            Statement::Insert(i) => Box::pin(self.exec_insert(&i, query, &env.identifier_spans))
                .await
                .map_err(|e| {
                    crate::execute_immediate::unbound_markers(
                        e,
                        query,
                        env.parser.double_quoted_identifiers,
                    )
                }),
            Statement::Truncate(t) => self.exec_truncate(&t).await,
            Statement::Script(c) => {
                let off =
                    crate::conf::get(&self.state.conf.snapshot(), "spark.sql.scripting.enabled")
                        .is_some_and(|v| v.trim().eq_ignore_ascii_case("false"));
                if off {
                    return Err(krishiv_common::error::UnsupportedFeatureSqlScripting {
                        sql_scripting_enabled: "\"spark.sql.scripting.enabled\"".to_owned(),
                    }
                    .into());
                }
                Box::pin(self.exec_script(&c, query, redacted)).await
            }
            Statement::Analyze(a) => Box::pin(self.exec_analyze(&a, query)).await,
            Statement::AnalyzeTables(a) => Box::pin(self.exec_analyze_tables(&a)).await,
            Statement::CreatePipelineDataset(c) => Self::exec_create_pipeline_dataset(&c),
            Statement::CreateFlow(f) => self.exec_create_flow(&f),
            Statement::LoadData(l) => self.exec_load_data(&l),
            Statement::InsertDirectory(d) => {
                Box::pin(self.exec_insert_directory(&d, &env.identifier_spans)).await
            }
            Statement::AccessControl(a) => Box::pin(self.exec_access_control(&a, &env)).await,
            Statement::AddResource(_) => Err(unsupported_statement("ADD JAR | FILE | ARCHIVE")),
            Statement::ListResource(_) => Err(unsupported_statement("LIST JAR | FILE | ARCHIVE")),
            Statement::Delete(d) => {
                let changes = crate::table_commands::RowChanges {
                    targets: &[],
                    values: &[],
                    condition: d.where_clause.as_ref(),
                    options: &d.options,
                    more: &[],
                    sources: &[],
                };
                self.exec_row_level_change(&d.table, d.alias.as_ref(), "DELETE", &changes, query)
                    .await
            }
            Statement::Update(u) => {
                let targets: Vec<_> = u.assignments.iter().map(|a| &a.target).collect();
                let values: Vec<_> = u.assignments.iter().map(|a| &a.value).collect();
                let changes = crate::table_commands::RowChanges {
                    targets: &targets,
                    values: &values,
                    condition: u.where_clause.as_ref(),
                    options: &u.options,
                    more: &[],
                    sources: &[],
                };
                self.exec_row_level_change(
                    &u.table,
                    u.alias.as_ref(),
                    "UPDATE TABLE",
                    &changes,
                    query,
                )
                .await
            }
            Statement::Merge(m) => self.exec_merge(&m, query).await,
            _ => Err(unsupported_statement("statements other than queries")),
        }
    }

    /// `range(start, end, step, numSlices)`, as Spark's `Session.range`
    /// builds it: the same [`PlanNode::Range`] SQL `range` resolves to.
    /// `start` defaults to 0 and `step` to 1 when not given; `step` must
    /// not be 0 and `num_slices` must be positive when given (else the
    /// first action fails, as SQL `range` does).
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// let rows = futures::executor::block_on(session.range(0, 3, 1, None).collect())?;
    /// assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn range(&self, start: i64, end: i64, step: i64, num_slices: Option<u32>) -> DataFrame {
        use krishiv_common::error::{FailedFunctionCall, KrishivInvalidPartitionNumber};
        use krishiv_plan::{BoundColumn, PlanNode};
        if step == 0 {
            return DataFrame::invalid(
                self.clone(),
                Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
                FailedFunctionCall {
                    func_name: "`range`".to_owned(),
                }
                .into(),
            );
        }
        if let Some(n) = num_slices
            && n == 0
        {
            return DataFrame::invalid(
                self.clone(),
                Arc::new(LogicalPlan::new(PlanNode::OneRowRelation)),
                KrishivInvalidPartitionNumber {
                    num_partitions: "0".to_owned(),
                }
                .into(),
            );
        }
        // The analyzer passes a resolved `Range` through with its output;
        // the id is fresh within this fresh plan.
        let output = BoundColumn {
            id: krishiv_plan::AttrId(0),
            name: "id".to_owned(),
            data_type: krishiv_types::DataType::Long,
            nullable: false,
        };
        let plan = LogicalPlan::new(PlanNode::Range {
            start,
            end,
            step,
            num_slices,
            output,
        });
        DataFrame::new(self.clone(), Arc::new(plan))
    }

    /// The table or view `name`, lazily (Spark Connect's `spark.table`: an
    /// unknown name fails at the first action).
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// let df = session.table("missing");
    /// let e = futures::executor::block_on(df.schema()).unwrap_err();
    /// assert_eq!(e.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn table(&self, name: &str) -> DataFrame {
        // Spark parses the name as SQL (`parseMultipartIdentifier`): dots
        // separate parts outside backquotes, and a syntax error is the
        // first action's error.
        match krishiv_sql::parse_multipart_identifier(
            name,
            &parser_options(&self.state.conf.snapshot(), self.state.profile),
        ) {
            Ok(id) => {
                let parts = id.parts.into_iter().map(|p| p.value);
                DataFrame::new(self.clone(), Arc::new(LogicalPlan::table(parts)))
            }
            Err(e) => {
                DataFrame::invalid(self.clone(), Arc::new(LogicalPlan::table([name])), e.error)
            }
        }
    }

    /// Parses `text` as a SQL expression (`F.expr` over this session) with
    /// this session's parser options and lowers it as [`Session::sql`]
    /// lowers `SELECT` items, so it resolves against a DataFrame's columns
    /// in the analyzer.
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// let df = session.table("t").select([session.expr("a + 1").unwrap()]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Spark's parse errors (`PARSE_SYNTAX_ERROR`, ...); lowering errors
    /// for constructs the IR lacks.
    pub fn expr(&self, text: &str) -> Result<Column, KrishivError> {
        let lowered = krishiv_sql::parse_and_lower_named_expression(
            text,
            &self.parser_options(),
            SourceId(0),
        )
        .map_err(|e| e.error)?;
        Ok(Column::new((*lowered).clone()))
    }

    /// Reads files (Spark's `spark.read`).
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// let df = session.read().option("mergeSchema", "false").parquet(["/data/events"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn read(&self) -> DataFrameReader {
        DataFrameReader::new(self.clone())
    }

    /// Registers `batches` (of `schema`) as the temporary table `name`,
    /// replacing any table or view of that name.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use arrow_array::{Int32Array, RecordBatch};
    /// use arrow_schema::{DataType, Field, Schema};
    ///
    /// # let session = krishiv_session::Session::new()?;
    /// let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    /// let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1, 2]))]).unwrap();
    /// session.register_batches("t", schema, vec![batch])?;
    /// let rows = futures::executor::block_on(session.table("t").collect())?;
    /// assert_eq!(rows[0].num_rows(), 2);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a column type Spark has no
    /// equivalent of; `INTERNAL_ERROR` when a batch does not have `schema`;
    /// `UNABLE_TO_ACQUIRE_MEMORY` when the relation store cannot hold the
    /// batches' bytes (its pool is unbounded, so only a shut-down pool
    /// refuses); `KRISHIV_RESERVED_TEMP_VIEW_NAME` when `name` starts with
    /// the reserved snapshot prefix `` `__krishiv_snapshot_` `` or the cache
    /// prefix `` `__krishiv_cache_` `` (in any letter case), which queries
    /// cannot spell either.
    pub fn register_batches(
        &self,
        name: &str,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    ) -> Result<(), KrishivError> {
        if is_reserved_view_name(name) {
            return Err(reserved_temp_view_name(name));
        }
        let (columns, table) = self.memory_table(schema, batches)?;
        self.state.catalog.put(
            name,
            TempTable::Memory {
                columns,
                table: Arc::new(table),
            },
        );
        // Replacing a table drops what was cached from the old one, and the
        // registrations over it, as replacing a temporary view does in Spark.
        self.state.cache.forget_table(name);
        Ok(())
    }

    /// Registers a native Rust scalar UDF under its name
    /// (case-insensitive), replacing any UDF of that name. Built-ins keep precedence on the
    /// bare search path; qualify with `system.session` (or `session`) to
    /// reach the UDF when a built-in shares its name. Every registration
    /// changes the analysis fingerprint, so plans analyzed before it are
    /// never reused after it.
    ///
    /// ```
    /// # use krishiv_session::{ScalarUdf, Session};
    /// # use krishiv_session::{ArgField, InvokeContext, ReturnField, Signature, UdfArg, UdfError, UdfInputType};
    /// # use krishiv_types::DataType;
    /// # use std::sync::Arc;
    /// # use arrow_array::ArrayRef;
    /// # #[derive(Debug)] struct One;
    /// # impl ScalarUdf for One {
    /// #     fn name(&self) -> &str { "one" }
    /// #     fn version(&self) -> u32 { 1 }
    /// #     fn signature(&self) -> &Signature {
    /// #         use std::sync::OnceLock; static SIG: OnceLock<Signature> = OnceLock::new();
    /// #         SIG.get_or_init(|| Signature::Exact(Vec::new()))
    /// #     }
    /// #     fn return_field(&self, _: &[ArgField]) -> Result<ReturnField, UdfError> {
    /// #         Ok(ReturnField { data_type: DataType::Integer, nullable: false })
    /// #     }
    /// #     fn invoke(&self, _: &[UdfArg<'_>], _: usize, _: &InvokeContext) -> Result<ArrayRef, UdfError> {
    /// #         Ok(Arc::new(arrow_array::Int32Array::from(vec![1])) as ArrayRef)
    /// #     }
    /// # }
    /// let session = Session::new()?;
    /// session.register_udf(One)?;
    /// let df = futures::executor::block_on(session.sql("SELECT one()"))?;
    /// let rows = futures::executor::block_on(df.collect())?;
    /// assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// `FAILED_EXECUTE_UDF` (naming `unknown` if there is no name) when the
    /// UDF's `name` panics, is empty, or holds a `.` or a backtick.
    pub fn register_udf(
        &self,
        udf: impl krishiv_udf_sdk::ScalarUdf,
    ) -> Result<(), krishiv_common::error::KrishivError> {
        let udf: Arc<dyn krishiv_udf_sdk::ScalarUdf> = Arc::new(udf);
        let named = || guard_call("unknown", "name", || udf.name().to_owned());
        let done = self.state.udfs.register(Arc::clone(&udf));
        done.map_err(|e| crate::udfs::registration_error(named(), &e))
    }

    /// Registers a native Rust aggregate UDF (ADR 0045), replacing any UDF
    /// of its name, scalar or aggregate; otherwise as [`register_udf`](Self::register_udf).
    ///
    /// # Errors
    /// As [`register_udf`](Self::register_udf).
    pub fn register_udaf(
        &self,
        udaf: impl krishiv_udf_sdk::AggregateUdf,
    ) -> Result<(), krishiv_common::error::KrishivError> {
        let udaf: Arc<dyn krishiv_udf_sdk::AggregateUdf> = Arc::new(udaf);
        let named = || guard_call("unknown", "name", || udaf.name().to_owned());
        let done = self.state.udfs.register_aggregate(Arc::clone(&udaf));
        done.map_err(|e| crate::udfs::registration_error(named(), &e))
    }

    /// The temporary tables and views, by name as registered.
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// assert!(session.temp_table_names().is_empty());
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn temp_table_names(&self) -> Vec<String> {
        self.state.catalog.names()
    }

    /// Drops the temporary table or view `name`; whether it existed
    /// (Spark's `spark.catalog.dropTempView`). Dropping a view also
    /// removes the snapshot pins it owned that no other live view still
    /// owns. Hidden snapshot names never exist for this call.
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// assert!(!session.drop_temp_view("t"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use = "whether the view existed"]
    pub fn drop_temp_view(&self, name: &str) -> bool {
        if is_global_key(name) {
            return false;
        }
        let dropped = self.state.catalog.drop_table(name);
        self.state.cache.forget_table(name);
        dropped
    }

    /// Drops every cached relation (Spark's `CLEAR CACHE` and
    /// `spark.catalog.clearCache`): later actions run uncached, and the
    /// `relation_cache` pool returns to baseline.
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// session.clear_cache();
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn clear_cache(&self) {
        self.state.cache.clear();
        // The key counts' answers (design P2b A5 (d)) are counted again.
        self.state.io.key_facts().clear();
    }

    /// Drops the cached relations reading the table `name` (Spark's
    /// `UNCACHE TABLE`): later actions over it run uncached.
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// session.uncache_table("t");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn uncache_table(&self, name: &str) {
        self.state.cache.forget_table(name);
    }

    /// Drops the cached relations reading the table `name` (Spark's
    /// `REFRESH TABLE`): the next action reads the table again. Unlike
    /// `CLEAR CACHE`, other tables' entries survive.
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// session.refresh_table("t");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn refresh_table(&self, name: &str) {
        self.state.cache.invalidate_table(name);
        // The key facts are keyed by file version, not by table: every one
        // is counted again (a file rewritten in place within the store's
        // modification-time resolution keeps its version).
        self.state.io.key_facts().clear();
    }

    /// Drops the cached relations reading files by path (Spark's
    /// `REFRESH path`): the next action lists the files again. The engine
    /// also calls this after its own file writes (`df.write ...`), whose
    /// target may be a cached path; an external change to a cached path is
    /// deliberately invisible and drops nothing.
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// session.refresh_files();
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn refresh_files(&self) {
        self.state.cache.invalidate_files();
    }

    /// Bytes the relation cache holds now (the `relation_cache` pool's
    /// reservation): what `unpersist` and the invalidation calls above
    /// return to baseline.
    ///
    /// ```
    /// # let session = krishiv_session::Session::new()?;
    /// assert_eq!(session.cache_reserved(), 0);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn cache_reserved(&self) -> usize {
        self.state.cache.reserved()
    }

    /// A DataFrame of the logical plan `plan` (the Python facade, Spark
    /// Connect and tests build plans themselves).
    ///
    /// ```
    /// use krishiv_plan::{LogicalPlan, Plan};
    /// # let session = krishiv_session::Session::new()?;
    /// let df = session.from_plan(&Plan::try_new(LogicalPlan::table(["t"]))?);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[cfg(feature = "unstable-ir")]
    #[must_use]
    pub fn from_plan(&self, plan: &Plan) -> DataFrame {
        Self::plan_frame(self.clone(), Arc::clone(plan.root()))
    }

    /// As [`Self::from_plan`], with per-statement options, as
    /// [`Self::sql_with`] takes them for SQL: an operation id names the
    /// queries the DataFrame's actions run (Spark Connect's `ExecutePlan`
    /// `operation_id`) and is held while the DataFrame or a result stream of
    /// it is alive; a deadline limits each of them.
    ///
    /// ```
    /// use krishiv_plan::{LogicalPlan, Plan};
    /// use krishiv_session::QueryOptions;
    /// # let session = krishiv_session::Session::new()?;
    /// let plan = Plan::try_new(LogicalPlan::table(["t"]))?;
    /// let id = "00112233-4455-6677-8899-aabbccddeeff";
    /// let df = session.from_plan_with(&plan, QueryOptions::new().operation_id(id))?;
    /// // The id is taken while `df` lives.
    /// assert!(session.from_plan_with(&plan, QueryOptions::new().operation_id(id)).is_err());
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// `INVALID_HANDLE.SESSION_CLOSED` on a closed session; for the
    /// operation id, the errors of [`Self::sql_with`] (`INVALID_HANDLE.FORMAT`,
    /// `OPERATION_ALREADY_EXISTS`, `OPERATION_ABANDONED`).
    #[cfg(feature = "unstable-ir")]
    pub fn from_plan_with(
        &self,
        plan: &Plan,
        options: crate::QueryOptions,
    ) -> Result<DataFrame, KrishivError> {
        Ok(Self::plan_frame(
            self.with_options(options)?,
            Arc::clone(plan.root()),
        ))
    }

    /// As [`Self::from_plan_with`], with each read of a table named in `frames` replaced by that
    /// frame's plan (the read node's origin, so a client's plan id still names it): a DataFrame
    /// the host holds under an id (Spark Connect's `CachedRemoteRelation`, the result of a
    /// checkpoint). The new frame owns what the frames own, so a checkpoint's rows live as long
    /// as it does even when the host lets go of the frame. A frame that is a lazy checkpoint takes
    /// its rows here. A read inside a subquery expression is not replaced.
    ///
    /// # Errors
    /// The errors of [`Self::from_plan_with`], and of taking a lazy checkpoint's rows.
    #[cfg(feature = "unstable-ir")]
    pub async fn from_plan_over(
        &self,
        plan: &Plan,
        options: crate::QueryOptions,
        frames: &[(Vec<String>, DataFrame)],
    ) -> Result<DataFrame, KrishivError> {
        let session = self.with_options(options)?;
        if frames.is_empty() {
            return Ok(Self::plan_frame(session, Arc::clone(plan.root())));
        }
        // The statement comes from the client's plan, before the cached frames' own plans
        // (which may hold `spark.sql` relations of their own) are spliced in.
        let statement = session.plan_statement(plan.root());
        let frames = frames.to_vec();
        let root = Arc::clone(plan.root());
        let (root, keep) = self
            .on_executor(async move {
                let mut keep: Vec<Arc<dyn std::any::Any + Send + Sync>> = Vec::new();
                let mut plans = Vec::with_capacity(frames.len());
                for (name, frame) in &frames {
                    plans.push((name.clone(), frame.plan_with_checkpoints().await?));
                    for k in &frame.keep {
                        if !keep.iter().any(|have| Arc::ptr_eq(have, k)) {
                            keep.push(Arc::clone(k));
                        }
                    }
                }
                let root = root.transform_down(&mut |node| {
                    let PlanNode::Read(ReadSource::Table { name, .. }) = &node.node else {
                        return Ok::<_, std::convert::Infallible>(node);
                    };
                    Ok(match plans.iter().find(|(n, _)| n == name) {
                        Some((_, plan)) => {
                            Arc::new(LogicalPlan::clone(plan).with_origin(node.origin))
                        }
                        None => node,
                    })
                });
                Ok((root.unwrap_or_else(|e| match e {}), keep))
            })
            .await?;
        let df = DataFrame::over_plan(session, root, keep);
        Ok(match statement {
            Some(env) => df.with_first_statement(env),
            None => df,
        })
    }

    /// Runs `work` on the session's executor, so that it finds the
    /// runtime its I/O needs whatever runtime (if any) polls the caller;
    /// dropping the returned future aborts the work.
    pub(crate) async fn on_executor<T: Send + 'static>(
        &self,
        work: impl Future<Output = Result<T, KrishivError>> + Send + 'static,
    ) -> Result<T, KrishivError> {
        // Boxed: the work is held once, not in both futures (`clippy::large_futures` at the callers).
        self.on_executor_reasoned(Box::pin(work)).await.0
    }

    /// [`Self::on_executor`], also returning the reason the action was ended
    /// with before any query of it was reserved, if it was (for the row of a
    /// statement that ended before its query existed).
    pub(crate) async fn on_executor_reasoned<T: Send + 'static>(
        &self,
        work: impl Future<Output = Result<T, KrishivError>> + Send + 'static,
    ) -> (Result<T, KrishivError>, Option<crate::CancelReason>) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        // Registered from here on: `cancel` and `interrupt_all` reach the
        // action during analysis and planning too (`action.rs`).
        let action = self.state.runtime.actions.begin_tagged(
            self.state.number,
            self.operation_id().map(crate::QueryId::as_str),
            self.tagging(),
        );
        let reason = action.reason();
        let work = ran::in_running_command(work);
        let task = self.state.runtime.executor.spawn(
            WorkloadClass::Interactive,
            Box::pin(async move {
                // The caller may be gone; its result is then not needed.
                let _ = tx.send(action.run(work).await);
            }),
        );
        let _abort = AbortOnDrop(task);
        let result = match rx.await {
            Ok(result) => result,
            Err(_) => Err(internal("a session task ended without a result".to_owned())),
        };
        (result, reason.ended_with())
    }

    /// A fresh DataFrame plan id.
    pub(crate) fn next_plan_id(&self) -> PlanId {
        PlanId(self.state.plan_ids.fetch_add(1, Ordering::Relaxed))
    }

    /// A fresh SQL source id for a view lowering (see `sql_sources`).
    pub(crate) fn next_source_id(&self) -> SourceId {
        SourceId(self.state.sql_sources.fetch_add(1, Ordering::Relaxed))
    }

    /// The block-cache scope of this session's stores: its principal. The session's storage
    /// credentials come from its own configuration, so one principal's queries may share blocks
    /// and no other principal's may (E15).
    ///
    /// The session's own `fs.*` credentials are not part of the scope: `SET` of them keeps it
    /// (store ids change, by their `#cfg:` hash), so blocks cached under the old credentials may
    /// stay readable with the new ones. Swap storage credentials with a new session.
    pub(crate) fn credential_scope(&self) -> krishiv_io::CacheScope {
        krishiv_io::CacheScope::Principal(self.state.principal.id())
    }

    /// The I/O context of this session's principal (E15, E12).
    fn scoped_io(&self) -> krishiv_io::IoContext {
        self.state
            .io
            .context()
            .with_principal(Arc::clone(&self.state.principal))
    }

    /// A store registry reading with this session's credential scope (E15).
    fn scoped_stores(&self, config: &Arc<Config>) -> Arc<StoreRegistry> {
        Arc::new(
            krishiv_io::query_registry(Arc::clone(config), &self.state.io)
                .with_cache_scope(self.credential_scope()),
        )
    }

    /// The analysis inputs of one query.
    pub(crate) fn query_env(&self) -> QueryEnv {
        // Before any analysis loads a table: a fill is refused if an
        // invalidation ran after this point (it may hold the replaced rows).
        let cache_generation = self.state.cache.generation();
        let (config, conf_version) = self.state.conf.query_snapshot();
        // Cloud stores are built from this snapshot, lazily per query:
        // plans carry only path strings, and the worker resolves the store
        // at listing from the read's option values (P2-T084).
        let stores = self.scoped_stores(&config);
        // One I/O context for the query's reads, scans and writes: its
        // counters are the query's own bytes (`system.queries`); its metadata reads charge the
        // session's pool while planning, the query's once it runs (`start_query`).
        let io = self.scoped_io().with_memory(Arc::clone(&self.state.memory));
        let paths: Arc<dyn krishiv_exec::files::PathLocator> = Arc::new(ObjectPaths::new(
            StoreId::new(LOCAL_STORE),
            self.state.cwd.clone(),
            Arc::clone(&self.state.io),
            Arc::clone(&stores),
        ));
        let files = Arc::new(FileReads::new(
            io.clone(),
            Arc::clone(&self.state.runtime.formats),
            Arc::clone(&paths),
            Arc::clone(&config),
        ));
        let declared_paths = Arc::clone(&paths);
        let delta = Arc::new(crate::delta::DeltaReads::new(crate::delta::DeltaInputs {
            io: io.clone(),
            paths,
            runtime: self.state.runtime.executor.io_handle().clone(),
            config: Arc::clone(&config),
            cache: Arc::clone(&self.state.runtime.delta),
            credential: self.state.principal.id(),
        }));
        let scans = FileScans {
            io: io.clone(),
            config: Arc::clone(&config),
            // File partitions are cut for this many slots. Under `krishiv.deterministicOrder` it
            // is a constant, never the machine's thread count: a morsel never spans a partition,
            // so the partitions decide which rows a lane reads (T094 G1).
            slots: if config
                .get(&krishiv_common::config::DETERMINISTIC_ORDER)
                .and_then(krishiv_common::config::ConfigValue::as_bool)
                .unwrap_or(false)
            {
                16
            } else {
                u64::try_from(self.state.runtime.executor.parallelism().max(1)).unwrap_or(1)
            },
            runtime: self.state.runtime.executor.io_handle().clone(),
            relations: Arc::clone(&self.state.relations),
            memory: Arc::clone(&self.state.memory),
        };
        let declared =
            self.declared_catalogs()
                .for_query(crate::catalog_registry::ScanInputs::new(
                    io.clone(),
                    declared_paths,
                    Arc::clone(&config),
                    scans.slots,
                    self.declared_stores(&config),
                ));
        let catalog = Arc::new(QueryCatalog::new(
            Arc::clone(&self.state.catalog),
            Arc::clone(&self.state.cache),
            Arc::clone(&files),
            Arc::clone(&delta),
            scans,
            &config
                .get(&krishiv_common::config::SQL_DEFAULT_PATH)
                .map(ToString::to_string)
                .unwrap_or_default(),
            crate::system::Sources {
                memory: Arc::clone(&self.state.memory),
                queries: Arc::clone(&self.state.queries),
                io: Arc::clone(&self.state.io),
                relations: Arc::clone(&self.state.cache),
                metrics: Arc::clone(&self.state.runtime.metrics),
                sessions: self.state.runtime.session_view(),
                groups: self.state.runtime.resource_groups.clone(),
                advice: Arc::new(crate::partition_advice::AdviceCell::new(
                    self.state.runtime.profile_history.clone(),
                    Arc::clone(&self.state.runtime.executor),
                )),
                catalogs: declared,
                information: Some(crate::information_schema::Source {
                    store: Arc::clone(&self.state.catalog.store),
                    principal: Arc::clone(&self.state.principal),
                    policy: Arc::clone(&self.state.policy),
                    grants: self.state.grants.clone(),
                }),
                monitor: Arc::new(crate::system::Monitor::new(
                    Arc::clone(&self.state.principal),
                    Arc::clone(&self.state.policy),
                )),
            },
        ));
        let resolution = {
            let values = conf_values(&config, &ResolutionConf::KEYS);
            ResolutionConf::from_conf(|k| values.get(k).map(String::as_str))
        };
        let parser = parser_options(&config, self.state.profile);
        // The cache cap is the query's own configuration snapshot.
        let cache_cap = RelationCache::cap_of(&config);
        // One snapshot of the variables per query, like the configuration
        // snapshot: every reference in the query reads the same values.
        let variables = self.state.variables.snapshot();
        let variables = if self.locals.is_empty() {
            variables
        } else {
            Arc::new(krishiv_plan::analyzer::VariableSnapshot {
                locals: self.locals.to_vec(),
                ..(*variables).clone()
            })
        };
        let sql_functions = Arc::new(
            QueryFunctions::new(
                Arc::clone(&self.state.functions),
                self.state.sql_functions.snapshot(),
                Arc::clone(&self.state.catalog),
                resolution.case_sensitive,
                Some(crate::sql_functions::ViewSource {
                    parser,
                    next_source: {
                        let state = Arc::clone(&self.state);
                        Arc::new(move || {
                            SourceId(state.sql_sources.fetch_add(1, Ordering::Relaxed))
                        })
                    },
                }),
            )
            .with_path(catalog.sql_path_snapshot())
            .outside_session(!catalog.in_session_catalog()),
        );
        let functions: Arc<dyn FunctionRegistry> = Arc::clone(&sql_functions) as _;
        let functions_version = functions.version();
        let udtfs = self.table_functions();
        let analysis = AnalysisContext {
            profile: self.state.profile,
            conf: resolution,
            principal: Arc::clone(&self.state.principal),
            policy: Arc::clone(&self.state.policy),
            catalogs: Arc::new(QueryCatalogs(Arc::clone(&catalog))),
            functions,
            table_functions: Arc::clone(&udtfs) as _,
            sql: Some(Arc::new(Frontend { options: parser })),
            files: Some(Arc::new(crate::delta::SessionFiles { files, delta })),
            variables: Arc::clone(&variables),
            // `DataFrame::analyzed_for_metadata` / `schema_for_metadata` switch one analysis to
            // `Metadata`; every other analysis may run the plan.
            mode: krishiv_plan::analyzer::AnalysisMode::Execute,
            // Filled per analysis by `Session::authorize_definer` callers; none by default.
            definer_texts: krishiv_plan::analyzer::views::DefinerTexts::default(),
            // Set per analysis to the permanent views it inlined (review r5 Z1, Z2).
            stored_view_texts: krishiv_plan::analyzer::views::DefinerTexts::default(),
            // Set per analysis to the temporary views it inlined.
            temp_view_texts: krishiv_plan::analyzer::views::DefinerTexts::default(),
        };
        // The epoch the query's catalog reads at: entries it loads are at least as new.
        let catalog_epoch = krishiv_plan::analyzer::Catalog::epoch(&*catalog);
        QueryEnv {
            io,
            config,
            history_hold: None,
            parser,
            analysis,
            catalog,
            conf_version,
            catalog_epoch,
            variables_version: variables.version,
            policy_version: self.state.policy.version().0,
            text: None,
            redacted: None,
            statement_class: None,
            row: Arc::default(),
            identifier_spans: Arc::default(),
            identifier_written: Arc::default(),
            observers: Arc::default(),
            withhold_profile: Arc::default(),
            cache_generation,
            cache_cap,
            sql_functions,
            functions_version,
            sample_salt: {
                use std::hash::BuildHasher as _;
                std::collections::hash_map::RandomState::new().hash_one(0_u8)
            },
            quiet: QuietDenials::default(),
            udtfs,
            spill_failure: Arc::default(),
            planning: Arc::default(),
        }
    }

    /// Analyzes `plan` (`text` is the redacted query the plan came
    /// from, if it came from SQL text: analysis errors carry its span
    /// as their query context). Also returns what the temporary view entries the analysis
    /// inlined kept (`TempTable::View::keeps`), the query's and its function bodies': the handles
    /// of the definitions the query runs, whatever replaced them since.
    pub(crate) async fn analyze_in(
        &self,
        plan: &Arc<LogicalPlan>,
        env: &QueryEnv,
        text: Option<&RedactedSql>,
    ) -> Result<(AnalyzedPlan, Vec<crate::catalog::Keep>), KrishivError> {
        // A store setting this session redirected (endpoint, region, account) could send the
        // server's credentials to another host:
        // asked once per analysis, before any table or file is read (P3-T067b).
        let r = self
            .check_store_options(&*env.analysis.policy, &env.config, std::iter::empty())
            .await;
        crate::access::audit_read(self, env, r)?;
        // Temporary views over queries inline as subqueries, so the
        // analyzer sees one plan however the view was made.
        let next_source = || self.next_source_id();
        let mut expansion = crate::catalog::expand_views(
            &self.state.catalog,
            plan,
            env.analysis.conf.case_sensitive,
            Some(&crate::catalog::StoredViews {
                parser: env.parser,
                next_source: &next_source,
                namespace: env
                    .catalog
                    .namespace()
                    .first()
                    .cloned()
                    .unwrap_or_else(|| self.state.catalog.current_namespace()),
                path: env.catalog.sql_path_snapshot(),
                binding_enabled: crate::view_schema::binding_enabled(&env.config),
                outside_session: !env.catalog.in_session_catalog(),
                cte_precedence: env.analysis.conf.cte_precedence,
                qualify: Some(&|e, namespace, path, rule| {
                    env.sql_functions
                        .qualify_calls(e, namespace, path, self.state.profile, rule)
                }),
                qualify_table: Some(&|name, namespace, path, rule| {
                    env.sql_functions.qualify_table_call(
                        name,
                        namespace,
                        path,
                        self.state.profile,
                        rule,
                    )
                }),
                stored_body: false,
            }),
        )?;
        let mut inlined = std::mem::take(&mut expansion.views);
        // What a definer view being created pins (review r10 M10).
        env.sql_functions.note_views(&expansion.reached);
        let mut keeps = std::mem::take(&mut expansion.keeps);
        // A permanent view is inlined before binding, so the analyzer never asks the
        // policy about it: ask here, before any read and before the text of a stored view
        // can fail, so a principal who may not read it learns nothing from it.
        let mut views = self.authorize_reads(&expansion.reads, env).await?;
        if let Some(e) = expansion.deferred {
            // Inside a definer view's text it is the definer's failure (review B2).
            if expansion.deferred_in_definer {
                return Err(self.definer_failure(env, e));
            }
            let e = self.view_text_failure(&views, env, e).await;
            return crate::access::audit_read(self, env, Err(e));
        }
        // `SQL SECURITY DEFINER` views (P3-T062 §2.4): the reader may read the view (asked above);
        // its text is read with its definer's rights, and the reader's analysis skips what it reads.
        let definers = std::mem::take(&mut expansion.definers);
        // Trusted by node identity (review B1): the inlined queries themselves, never a number.
        let texts = krishiv_plan::analyzer::views::DefinerTexts::new(
            definers.iter().filter_map(|d| d.query.clone()).collect(),
        );
        for d in &definers {
            match Box::pin(self.authorize_definer(d, &texts, env)).await {
                Ok(owner_reads) => {
                    if let Some(v) = inlined.get_mut(d.view) {
                        v.owner_reads = owner_reads;
                    }
                }
                Err(e) => return Err(self.definer_failure(env, e)),
            }
        }
        // What only their texts call was asked of their definers (review r5 M7).
        env.sql_functions
            .settle_definer_calls(&expansion.plan, &texts, self.state.profile);
        // The profile shows the definers' plans only to a principal that may see them (M2).
        if !definers.is_empty() && !self.may_see_definer_reads(&inlined).await {
            env.withhold_profile
                .store(true, std::sync::atomic::Ordering::Release);
        }
        // Every permanent view's text, kept by identity: nothing of the reading session resolves in
        // it (review r5 Z1, Z2).
        let stored_texts = krishiv_plan::analyzer::views::DefinerTexts::new(
            expansion
                .stored_reads
                .iter()
                .map(|r| Arc::clone(&r.query))
                .collect(),
        );
        // Every temporary view's text: the metric-view rule stops there, as at a permanent view's.
        let temp_texts =
            krishiv_plan::analyzer::views::DefinerTexts::new(expansion.temp_texts.clone());
        let trusting_views =
            definer::trusting_context(&env.analysis, &texts, &stored_texts, &temp_texts);
        let views_analysis = trusting_views.as_ref().unwrap_or(&env.analysis);
        // (Boxed: `analyze_in` runs on the deepest stacks, where the size of its
        // future counts.)
        let (synced, renames) =
            match Box::pin(self.check_stored_views(&expansion.stored_reads, env, views_analysis))
                .await
            {
                Ok(checked) => checked,
                // A definer view's binding, or one inside a definer's text, is the definer's (review r2 R1).
                Err((e, true)) => return Err(self.definer_failure(env, e)),
                Err((e, false)) => {
                    let e = self.view_text_failure(&views, env, e).await;
                    return crate::access::audit_read(self, env, Err(e));
                }
            };
        // A rename rebuilds the nodes above it: the trusted queries follow their rebuilt nodes.
        let (renamed, mut kept) = crate::view_schema::apply_renames_keeping(
            &expansion.plan,
            &renames,
            &[&texts, &stored_texts, &temp_texts],
        );
        let temp_texts = kept.pop().unwrap_or_default();
        let stored_texts = kept.pop().unwrap_or_default();
        let texts = kept.pop().unwrap_or_default();
        let trusting = definer::trusting_context(&env.analysis, &texts, &stored_texts, &temp_texts);
        let analysis = trusting.as_ref().unwrap_or(&env.analysis);
        let plan = Plan::try_new(renamed)?;
        // The bodies of the functions the plan calls read views the same way:
        // the policy is asked about them before the analysis' own result (or
        // a failure inside a view's text) is shown. One boxed future for both
        // steps: `analyze_in` runs on the deepest stacks, where the size of
        // its own future counts.
        let (analyzed, body_views, calls) = Box::pin(async {
            let analyzed = self.state.analyzer.analyze(&plan, analysis).await;
            (
                analyzed,
                self.authorize_body_reads(env).await,
                self.authorize_function_calls(env).await,
            )
        })
        .await;
        let analyzed = analyzed.map_err(|e| {
            // An error inside a SQL function's body points into the
            // statement that defined the function.
            let own = e
                .origin
                .sql
                .and_then(|(source, _)| env.sql_functions.source_text(source));
            attach_analysis(e, own.as_ref().or(text))
        });
        views.extend(body_views?);
        // The stored functions it calls, before the analysis' own result (P3-T062 §2.4).
        crate::access::audit_read(self, env, calls)?;
        let analyzed = match analyzed {
            Ok(analyzed) => Ok(analyzed),
            Err(e) => Err(self.view_text_failure(&views, env, e).await),
        };
        // Analysis checks every read: its denials are the reads' denials.
        let mut analyzed = crate::access::audit_read(self, env, analyzed)?;
        // Spark's `ViewSyncSchemaToMetaStore`: once the query analyzed, evolved views store their new schema. A
        // concurrent change of the view wins (its entry is no longer the one the read found).
        // A difference from Spark (ADR 0008: explicit): a metadata analysis (Connect `AnalyzePlan`, for every
        // role, admins too) never stores it. Spark syncs in the analysis of every Dataset, `.schema` included;
        // here an `AnalyzePlan` must not write the catalog (a viewer has `Browse` only and no side effects).
        // It returns the evolved schema all the same, and the next ordinary query stores it.
        if env.analysis.mode == krishiv_plan::analyzer::AnalysisMode::Execute {
            for (found, changed) in synced {
                crate::store::CatalogStore::replace_table(
                    &*self.state.catalog.store,
                    &found,
                    changed,
                );
            }
        }
        // The view side table: where each view the query and its functions' bodies inlined landed.
        inlined.extend(env.sql_functions.body_views());
        keeps.extend(env.sql_functions.body_keeps());
        analyzed.views = krishiv_plan::analyzer::views::locate_views(analyzed.plan.root(), inlined);
        for view in views {
            let securable = krishiv_plan::analyzer::Securable::View(view);
            if !analyzed.securables.iter().any(|s| s.securable == securable) {
                analyzed
                    .securables
                    .push(krishiv_plan::analyzer::SecurableUse {
                        securable,
                        privilege: env.analysis.mode.read_privilege(),
                        columns: std::collections::BTreeSet::new(),
                    });
            }
        }
        Ok((analyzed, keeps))
    }

    /// Runs `physical`; the stream's batches are its result rows.
    pub(crate) async fn run(
        &self,
        physical: &PhysicalPlan,
        env: &QueryEnv,
    ) -> Result<RecordBatchStream, KrishivError> {
        Ok(self.run_with_query(physical, env).await?.0)
    }

    /// As [`Self::run`], also returning the query's context: its metrics
    /// are complete once the stream has ended.
    pub(crate) async fn run_with_query(
        &self,
        physical: &PhysicalPlan,
        env: &QueryEnv,
    ) -> Result<(RecordBatchStream, Arc<QueryContext>), KrishivError> {
        self.run_on_query(physical, self.worker(env), env, false)
            .await
    }

    /// As [`Self::run_with_query`], collecting the detailed metrics of a
    /// profile (`EXPLAIN ANALYZE`).
    pub(crate) async fn run_profiled(
        &self,
        physical: &PhysicalPlan,
        env: &QueryEnv,
    ) -> Result<(RecordBatchStream, Arc<QueryContext>, String), KrishivError> {
        // The row id too: the profile reads the query's `planning_ns` from its row (MODES-2).
        let (stream, query, log_id) = self
            .run_on_query_logged(physical, self.worker(env), env, true, None)
            .await?;
        Ok((crate::run_site::resolve_sites(stream, env), query, log_id))
    }

    /// The session's relation store (tests): every registered memory
    /// table's batches as content-addressed IPC.
    #[cfg(test)]
    pub(crate) fn relations(&self) -> &Arc<RelationStore> {
        &self.state.relations
    }

    /// Runs `physical` instantiated with `worker`.
    async fn run_on_query(
        &self,
        physical: &PhysicalPlan,
        worker: WorkerContext,
        env: &QueryEnv,
        detailed: bool,
    ) -> Result<(RecordBatchStream, Arc<QueryContext>), KrishivError> {
        let (stream, query, _) = self
            .run_on_query_logged(physical, worker, env, detailed, None)
            .await?;
        Ok((crate::run_site::resolve_sites(stream, env), query))
    }

    /// `error`, when it is a memory error of a query whose spill directory could not be set up,
    /// as that failure (the memory error stays its cause): the query may have needed to spill
    /// (PY-078). A refused reservation and an arbitration abort are both such errors: which
    /// one ends a query that cannot spill depends on timing. Applied below the metrics fold,
    /// so `system.queries` and the caller agree.
    fn name_spill_failure(
        error: KrishivError,
        failure: &std::sync::Mutex<Option<KrishivError>>,
    ) -> KrishivError {
        use krishiv_common::error::ErrorClass;
        if !matches!(
            error.error_class(),
            ErrorClass::UnableToAcquireMemory | ErrorClass::KrishivQueryAbortedOutOfMemory
        ) {
            return error;
        }
        let found = failure
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        match found {
            Some(f) => f.with_cause(error),
            None => error,
        }
    }

    /// The tags a query started now carries: the session's, then the statement's request tags as
    /// `(tag, "")` pairs (a key appears once; design P3-T064 §4.3).
    pub(crate) fn effective_tags(&self) -> Vec<(String, String)> {
        let mut tags = self.state.queries.tags_now();
        crate::action::merge_request_tags(&mut tags, &self.request_tags);
        tags
    }

    /// [`Self::tagging`] with the tags row `log_id` was recorded with, so a `set_tag` while the
    /// query waited for admission does not change what cancel by tag sees.
    pub(crate) fn tagging_of_row(&self, log_id: &str) -> crate::action::Tagging {
        let mut tagging = self.tagging();
        if let Some(tags) = self.state.queries.row_tags(log_id) {
            tagging.tags = tags.into();
        }
        tagging
    }

    /// The trace a statement started now belongs to: its request's
    /// ([`crate::QueryOptions::inbound_trace`]) when it has one, else the session's.
    pub(crate) fn trace_now(&self) -> Option<crate::trace_context::RemoteParent> {
        match &self.inbound_trace {
            Some(t) => t.as_deref().cloned(),
            None => self.state.queries.trace(),
        }
    }

    /// Stamps the history row `id` with what this statement adds to it: its request tags and, when
    /// it has its own trace, the trace id.
    pub(crate) fn stamp_row(&self, id: &str) {
        self.state.queries.add_request_tags(id, &self.request_tags);
        if self.inbound_trace.is_some() {
            self.state
                .queries
                .set_row_trace(id, self.trace_now().map(|t| t.trace_id));
        }
    }

    /// The principal and tags of a statement started now, for cancel by tag.
    pub(crate) fn tagging(&self) -> crate::action::Tagging {
        crate::action::Tagging {
            owner: Arc::from(crate::sessions::owner_key(&self.state.principal).as_str()),
            tags: self.effective_tags().into(),
            request_tags: Arc::clone(&self.request_tags),
        }
    }

    /// As [`Self::run_on_query`], also returning the id of the `system.queries` row this execution
    /// began, so the caller keeps it for itself: the statement's slot in `env` is shared by every
    /// execution of the same analyzed frame.
    ///
    /// Every run completes the frame's observations: a delivered stream as it ends, a run that
    /// fails before it has one (closed session, refused admission, failed submit) with its
    /// error, and a run dropped during the setup with the reason it was dropped for (the action's
    /// cancellation, a panic, else the caller giving up: `OPERATION_CANCELED`).
    async fn run_on_query_logged(
        &self,
        physical: &PhysicalPlan,
        worker: WorkerContext,
        env: &QueryEnv,
        detailed: bool,
        admitted: Option<Admitted>,
    ) -> Result<(RecordBatchStream, Arc<QueryContext>, String), KrishivError> {
        let pending = crate::dataframe::PendingObservations::new(&env.observers);
        let started = self
            .start_query(physical, worker, env, detailed, admitted)
            .await;
        pending.settle(started.as_ref().err());
        started
    }

    /// [`Self::run_on_query_logged`] up to the delivered stream.
    async fn start_query(
        &self,
        physical: &PhysicalPlan,
        worker: WorkerContext,
        env: &QueryEnv,
        detailed: bool,
        admitted: Option<Admitted>,
    ) -> Result<(RecordBatchStream, Arc<QueryContext>, String), KrishivError> {
        self.check_open()?;
        // The history row and the permit first (unless the caller took them already, before it
        // changed anything): the row's id names the query's memory pool, so
        // `system.memory.query_id` joins `system.queries.query_id`.
        let (log_id, permit) = match admitted {
            Some(admitted) => admitted.into_parts(),
            None => self
                .admit_query(
                    env,
                    env.catalog.reads_only_system(physical),
                    krishiv_sql::StatementClass::Read,
                )
                .await?
                .into_parts(),
        };
        // The plan is made: what the statement spent planning, the queue excluded (MODES-2).
        self.state
            .queries
            .set_planning_ns(&log_id, env.planning.take());
        self.state.activity.query_started();
        let id = QueryId(self.state.runtime.reserve_tagged(
            self.state.number,
            self.state.queries.session_id(),
            &log_id,
            self.tagging_of_row(&log_id),
        ));
        let pool = self.state.memory.child_with_priority(
            &crate::system::pool_name(&log_id),
            permit.memory_budget(),
            permit
                .priority()
                .unwrap_or(krishiv_memory::DEFAULT_PRIORITY),
        );
        let query = QueryContext::new(
            id,
            permit.class().unwrap_or(WorkloadClass::Interactive),
            ExecConfig {
                detailed_metrics: detailed,
                ..ExecConfig::from_config(&env.config)
            },
            pool,
        );
        // The query's row in `system.queries`, and (when metrics are on) its
        // one metrics fold, both closed by the guard when the stream ends,
        // fails or is dropped.
        let metrics_on = metrics_enabled(&env.config);
        if metrics_on {
            self.state.runtime.metrics.query_begin(query.class());
        }
        let row_id = log_id.clone();
        // A definer's plan the principal may not see stays out of its profile (review M2).
        let withheld = env
            .withhold_profile
            .load(std::sync::atomic::Ordering::Acquire);
        let mut pending = PendingFold(
            Some(QueryFold {
                query: Arc::clone(&query),
                state: Arc::clone(&self.state),
                start: QueryStart::capture(&self.state, &env.io),
                _io_memory: Some(env.io.bind_query_memory(Arc::clone(query.pool()))),
                metrics_on,
                log_id,
                _permit: permit,
                _operation: self.operation.clone(),
                history: if withheld {
                    None
                } else {
                    env.history_job(self, &row_id, physical, detailed)
                },
                advisor: if withheld {
                    None
                } else {
                    crate::query_fold::AdvisorJob::of(physical, &env.config)
                },
                withhold_operators: withheld,
            }),
            None,
        );
        // Closed after `check_open`: the pending fold cancels the query
        // and records it as it drops.
        self.attach(&query, &row_id, &env.config)?;
        // A plan the wire format cannot carry fails as a submit failure does:
        // its row records the error.
        let submitted = match crate::plan_round_trip::round_trip(physical, &env.config) {
            Ok(decoded) => {
                let physical = decoded.as_ref().unwrap_or(physical);
                // Submitted from a task of its own: instantiation runs on the blocking pool and
                // goes on when the caller gives up, so the admission permit must outlive it
                // (see `PendingFold`). The task hands the stream over, or cancels it if the
                // caller is gone.
                // The decided scans read first (`after_reads`), on this async task, against the query's cancel.
                let (plan, reading) = (physical.clone(), Arc::clone(&query));
                let submit = self.state.runtime.runner.submit(
                    plan.clone(),
                    worker.clone(),
                    Arc::clone(&reading),
                );
                let (stream_tx, stream_rx) = tokio::sync::oneshot::channel();
                let (done_tx, done_rx) = tokio::sync::oneshot::channel::<bool>();
                pending.1 = Some(done_rx);
                let abandoned = Arc::clone(&query);
                drop(
                    self.state.runtime.executor.spawn(
                        WorkloadClass::Interactive,
                        Box::pin(
                            async move {
                                let result = worker.after_reads(&plan, reading, submit).await;
                                let started = result.is_ok();
                                if let Err(Ok(stream)) = stream_tx.send(result) {
                                    abandoned.cancel();
                                    drop(stream);
                                }
                                let _ = done_tx.send(started);
                            }
                            .instrument(tracing::Span::current()),
                        ),
                    ),
                );
                stream_rx
                    .await
                    .unwrap_or_else(|_| Err(internal("a query ended during its setup".to_owned())))
            }
            Err(error) => Err(error),
        };
        match submitted {
            Err(error) => {
                let error = Self::name_spill_failure(error, &env.spill_failure);
                if let Some(fold) = pending.0.take() {
                    fold.finish(Some(&error));
                }
                // No stream completes the observations: the failure does, in the caller.
                Err(error)
            }
            Ok(stream) => {
                let failure = Arc::clone(&env.spill_failure);
                let inner = Box::pin(
                    stream.map(move |r| r.map_err(|e| Self::name_spill_failure(e, &failure))),
                );
                let fold = pending.0.take();
                let stream = MetricStream { inner, fold };
                let stream = crate::dataframe::deliver(
                    Box::pin(stream.map(|b| b.and_then(into_record_batch))),
                    Arc::clone(&query),
                    &env.observers,
                );
                Ok((stream, query, row_id))
            }
        }
    }

    /// The engine's metrics snapshot: the folded series of ended queries
    /// plus the gauges read now. Query series appear only when
    /// `krishiv.metrics.enabled` was true as the queries ran. Only queries
    /// that reached execution are counted: statements that fail in parsing,
    /// analysis or an access check are not.
    ///
    /// ```
    /// let session = krishiv_session::Session::new()?;
    /// let snapshot = session.metrics();
    /// assert!(snapshot.gauges.iter().any(|g| g.name == "krishiv.build.info"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn metrics(&self) -> MetricsSnapshot {
        self.state.runtime.metrics.snapshot()
    }
}

#[cfg(test)]
mod tests;

mod env;
pub(crate) use env::*;

mod spill;
pub(crate) use spill::*;

pub(crate) mod writes;
pub(crate) use writes::partition_overwrite_is_dynamic;
use writes::*;

mod ops;
use ops::AbortOnDrop;

mod planning;

#[cfg(feature = "unstable-ir")]
mod plan_frames;

// `SQL SECURITY DEFINER` reads (P3-T062 §2.4).
mod definer;

mod ran;
mod search_path;
pub(crate) use ran::{
    Ran, as_evaluation, as_listed_script_statement, as_script_statement, command_statement,
    evaluation_failed,
};
