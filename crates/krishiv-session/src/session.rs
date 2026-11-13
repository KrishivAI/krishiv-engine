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
    BuiltinTableFunctions, DefaultAnalyzer, FunctionRegistry, Principal, Privilege, ResolutionConf,
    Securable, SparkProfile, authorize,
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
#[cfg(test)]
mod run_task;

/// The store id of the local file system.
pub(crate) const LOCAL_STORE: &str = "file";

/// Now, as microseconds since the epoch.
fn query_start_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_micros()).ok())
        .unwrap_or_default()
}

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
}

impl fmt::Debug for Session {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Session")
            .field("profile", &self.state.profile)
            .field("principal", &self.state.principal.name)
            .finish_non_exhaustive()
    }
}

pub(crate) struct SessionState {
    pub(crate) conf: Arc<ConfStore>,
    pub(crate) catalog: Arc<SessionCatalog>,
    pub(crate) cache: Arc<RelationCache>,
    pub(crate) variables: Arc<SessionVariables>,
    pub(crate) profile: SparkProfile,
    /// Fixed for the session's life: caches and fingerprints never mix
    /// identities (a new identity gets a new session).
    pub(crate) principal: Arc<Principal>,
    /// Decides when each query may start (E5); the default admits at once.
    pub(crate) admission: Arc<dyn AdmissionController>,
    pub(crate) policy: Arc<dyn AccessPolicy>,
    pub(crate) functions: Arc<dyn FunctionRegistry>,
    /// The registry the builder supplied, if any (child sessions use it too).
    base_functions: Option<Arc<dyn FunctionRegistry>>,
    /// The session's registered UDFs, shared with the registry above and
    /// the workers running its plans.
    udfs: Arc<UdfLibrary>,
    /// The session's Python UDFs (P2-T059), shared with the registry above:
    /// plans name them by name and version only.
    pub(crate) python_udfs: Arc<crate::python_udfs::PythonUdfLibrary>,
    /// The runtime that evaluates them in this process (set by the Python
    /// facade); without one a plan calling a Python UDF fails to run.
    pub(crate) python_runtime: Mutex<Option<Arc<dyn krishiv_exec::python_eval::UdfRuntime>>>,
    /// The SQL-bodied functions the session defines.
    pub(crate) sql_functions: SqlFunctions,
    /// The process-wide part of the engine this session runs on.
    pub(crate) runtime: Arc<EngineRuntime>,
    /// The session's number on its runtime.
    pub(crate) number: u64,
    /// Set by [`Session::close`]: the session takes no new statement.
    closed: AtomicBool,
    /// The session's memory pool: the runtime's root, or a pool of its own
    /// under the root (`krishiv.session.memoryLimit`).
    pub(crate) memory: Arc<dyn MemoryPool>,
    /// Whether a store creates objects only if absent (the commit log's
    /// probe), by store id; per session, never process-wide.
    commit_log_probes: Mutex<HashMap<String, bool>>,
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
}

impl Drop for SessionState {
    fn drop(&mut self) {
        self.runtime.forget_session(self.number);
    }
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
    executor: Option<Arc<dyn Executor>>,
    functions: Option<Arc<dyn FunctionRegistry>>,
    runtime: Option<Arc<EngineRuntime>>,
    admission: Option<Arc<dyn AdmissionController>>,
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

    /// The Spark version whose behaviour queries follow (default 4.2).
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
        let parts = SessionParts {
            profile: self.profile,
            principal: Arc::new(self.principal.unwrap_or_else(Principal::local_user)),
            policy: self.policy.unwrap_or_else(|| Arc::new(AllowAll)),
            functions: self.functions,
            admission: self.admission,
        };
        Ok(Session::on_runtime(runtime, config, parts))
    }
}

/// What a session keeps from its builder, and a child session from its
/// parent ([`Session::new_session`]).
struct SessionParts {
    profile: SparkProfile,
    principal: Arc<Principal>,
    policy: Arc<dyn AccessPolicy>,
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
            executor: None,
            functions: None,
            runtime: None,
            admission: None,
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
        self.check_open()?;
        let (session, text) = (self.clone(), query.to_owned());
        let row: Arc<std::sync::Mutex<Option<String>>> = Arc::default();
        let marker = Arc::clone(&row);
        let result = self
            .on_executor(async move {
                let seq = session.state.statements.fetch_add(1, Ordering::Relaxed);
                use tracing::Instrument as _;
                let trace = crate::telemetry::QueryTrace::of(seq, &text);
                let span = trace.prepare_span(&crate::telemetry::Who::of(&session.state));
                session
                    .sql_here_marked(&text, marker)
                    .instrument(span)
                    .await
                    .map(|df| df.with_trace(trace))
            })
            .await;
        // A statement refused before it ran (parse, analysis, access) is a
        // row of `system.queries` too, with the text as the session may show it.
        // ...unless a query already ran inside `sql` (an `EXPLAIN ANALYZE`
        // that failed while it ran): that one has its row.
        if let Err(error) = &result
            && row.lock().unwrap_or_else(PoisonError::into_inner).is_none()
        {
            self.state.queries.record_refused(
                crate::system::TextPolicy::from_config(&self.state.conf.snapshot()),
                Some(krishiv_sql::redact_sql(query).as_str()),
                None,
                error,
                self.operation_id(),
            );
        }
        result
    }

    /// `query` with its `${name}` references replaced, unless `spark.sql.variable.substitute` is false.
    pub(crate) fn substituted<'q>(&self, query: &'q str) -> std::borrow::Cow<'q, str> {
        let config = self.state.conf.snapshot();
        if crate::conf::get(&config, "spark.sql.variable.substitute")
            .is_none_or(|v| !v.trim().eq_ignore_ascii_case("false"))
        {
            crate::variable_substitution::substitute(query, &|k| crate::conf::get(&config, k))
        } else {
            std::borrow::Cow::Borrowed(query)
        }
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
        Box::pin(self.sql_here_inner(query, Arc::default(), spans)).await
    }

    /// Runs `query` on the calling task as `sql` would, with `row` as the statement's slot for
    /// the id of the `system.queries` row a query of it began.
    pub(crate) async fn sql_here_marked(
        &self,
        query: &str,
        row: Arc<std::sync::Mutex<Option<String>>>,
    ) -> Result<DataFrame, KrishivError> {
        // Boxed: one more nested future level overflows the compiler's query depth in the binaries.
        Box::pin(self.sql_here_inner(query, row, Vec::new())).await
    }

    async fn sql_here_inner(
        &self,
        query: &str,
        row: Arc<std::sync::Mutex<Option<String>>>,
        spans: Vec<krishiv_sql::ast::Span>,
    ) -> Result<DataFrame, KrishivError> {
        let mut env = self.query_env();
        env.row = row;
        let written = query;
        let query = self.substituted(query);
        let query: &str = &query;
        // `IDENTIFIER(expr)` names are evaluated first: parsing, access checks and the run see them.
        let resolved = self.identifier_clauses_into(query, &mut env).await?;
        let named_by_clause = matches!(resolved, std::borrow::Cow::Owned(_));
        if !spans.is_empty() && !named_by_clause && query == written {
            env.identifier_spans = spans.into();
        }
        let query: &str = &resolved;
        let redacted = krishiv_sql::redact_sql(query);
        let statement = krishiv_sql::parse_statement(query, &env.parser)
            .map_err(|e| sql_context(e.error, &redacted, e.span.start, e.span.end))?;
        // Spark refuses a database on a temporary function named by an `IDENTIFIER(...)`
        // clause when it drops it; the parser sees only the evaluated name.
        if named_by_clause
            && let Statement::DropFunction(f) = &statement
            && f.temporary
        {
            krishiv_sql::check_temporary_function_name(&f.name, f.span)
                .map_err(|e| sql_context(e.error, &redacted, e.span.start, e.span.end))?;
        }
        // Before any side effect: what the statement writes, creates or
        // drops (its reads are checked by analysis, first thing).
        let class = krishiv_sql::classify(&statement);
        self.authorize_statement(&statement, class, &env).await?;
        env.statement_class = Some(class.name());
        let df = self.run_statement(statement, env, query, &redacted).await?;
        // Whatever frame the statement returns (a command's result too) shows the statement's
        // text and class when it runs.
        df.with_statement(&redacted, class.name());
        Ok(df)
    }

    /// Runs the parsed, authorized `statement`.
    pub(crate) async fn run_statement(
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
                let plan = self.lower_identified(&q, &env, redacted)?;
                let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
                df.analyzed_in_with_text(env, Some(redacted)).await?;
                Ok(df)
            }
            Statement::Explain(e) => self.explain_statement(&e, env, redacted).await,
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
            Statement::DropFunction(f) => self.exec_drop_function(&f).await,
            Statement::CreateTable(t) => {
                Box::pin(self.exec_create_table(&t, query, &env.identifier_spans)).await
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
            Statement::DropPermanentView(d) => self.exec_drop_permanent_view(&d).await,
            Statement::AlterView(a) => {
                let ids = (&*env.identifier_spans, &*env.identifier_written);
                self.exec_alter_view(&a, query, ids).await
            }
            Statement::AlterTable(a) => Box::pin(self.exec_alter_table(&a, query)).await,
            Statement::Insert(i) => {
                Box::pin(self.exec_insert(&i, query, &env.identifier_spans)).await
            }
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
            Statement::AddResource(_) => Err(unsupported_statement("ADD JAR | FILE | ARCHIVE")),
            Statement::ListResource(_) => Err(unsupported_statement("LIST JAR | FILE | ARCHIVE")),
            Statement::Delete(d) => {
                let changes = crate::table_commands::RowChanges {
                    targets: &[],
                    values: &[],
                    condition: d.where_clause.as_ref(),
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
            Statement::Merge(m) => {
                let changes = crate::table_commands::RowChanges {
                    targets: &[],
                    values: &[],
                    condition: None,
                };
                self.exec_row_level_change(&m.target, None, "MERGE INTO TABLE", &changes, query)
                    .await
            }
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
            &parser_options(&self.state.conf.snapshot()),
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

    /// This session's parser options, from its configuration.
    pub(crate) fn parser_options(&self) -> ParserOptions {
        parser_options(&self.state.conf.snapshot())
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
        // Replacing a table drops what was cached from the old one, as
        // replacing a temporary view does in Spark.
        self.state.cache.invalidate_table(name);
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
        self.state.cache.invalidate_table(name);
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
        self.state.cache.invalidate_table(name);
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
        DataFrame::new(self.clone(), Arc::clone(plan.root()))
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
        Ok(DataFrame::new(
            self.with_options(options)?,
            Arc::clone(plan.root()),
        ))
    }

    /// Runs `work` on the session's executor, so that it finds the
    /// runtime its I/O needs whatever runtime (if any) polls the caller;
    /// dropping the returned future aborts the work.
    pub(crate) async fn on_executor<T: Send + 'static>(
        &self,
        work: impl Future<Output = Result<T, KrishivError>> + Send + 'static,
    ) -> Result<T, KrishivError> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let task = self.state.runtime.executor.spawn(
            WorkloadClass::Interactive,
            Box::pin(async move {
                // The caller may be gone; its result is then not needed.
                let _ = tx.send(work.await);
            }),
        );
        let _abort = AbortOnDrop(task);
        rx.await
            .map_err(|_| internal("a session task ended without a result".to_owned()))?
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
        // counters are the query's own bytes (`system.queries`).
        let io = self.scoped_io();
        let files = Arc::new(FileReads::new(
            io.clone(),
            Arc::clone(&self.state.runtime.formats),
            Arc::new(ObjectPaths::new(
                StoreId::new(LOCAL_STORE),
                self.state.cwd.clone(),
                Arc::clone(&self.state.io),
                Arc::clone(&stores),
            )),
            Arc::clone(&config),
        ));
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
        };
        let catalog = Arc::new(QueryCatalog::new(
            Arc::clone(&self.state.catalog),
            Arc::clone(&self.state.cache),
            Arc::clone(&files),
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
        let parser = parser_options(&config);
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
            .in_system(catalog.in_system()),
        );
        let functions: Arc<dyn FunctionRegistry> = Arc::clone(&sql_functions) as _;
        let functions_version = functions.version();
        let analysis = AnalysisContext {
            profile: self.state.profile,
            conf: resolution,
            principal: Arc::clone(&self.state.principal),
            policy: Arc::clone(&self.state.policy),
            catalogs: Arc::new(QueryCatalogs(Arc::clone(&catalog))),
            functions,
            table_functions: Arc::new(BuiltinTableFunctions::new()),
            sql: Some(Arc::new(Frontend { options: parser })),
            files: Some(files),
            variables: Arc::clone(&variables),
        };
        // The epoch the query's catalog reads at: entries it loads are at
        // least as new.
        let catalog_epoch = krishiv_plan::analyzer::Catalog::epoch(&*catalog);
        QueryEnv {
            io,
            config,
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
            cache_generation,
            cache_cap,
            sql_functions,
            functions_version,
        }
    }

    /// Analyzes `plan` (`text` is the redacted query the plan came
    /// from, if it came from SQL text: analysis errors carry its span
    /// as their query context).
    pub(crate) async fn analyze_in(
        &self,
        plan: &Arc<LogicalPlan>,
        env: &QueryEnv,
        text: Option<&RedactedSql>,
    ) -> Result<AnalyzedPlan, KrishivError> {
        // A store setting this session redirected (endpoint, region, account) could send the
        // server's credentials to another host:
        // asked once per analysis, before any table or file is read (P3-T067b).
        self.authorize_store_options(StatementClass::Read, &env.config, std::iter::empty())
            .await?;
        // Temporary views over queries inline as subqueries, so the
        // analyzer sees one plan however the view was made.
        let next_source = || self.next_source_id();
        let expansion = crate::catalog::expand_views(
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
                in_system: env.catalog.in_system(),
                qualify: Some(&|e, namespace, path| {
                    env.sql_functions
                        .qualify_calls(e, namespace, path, self.state.profile, true)
                }),
            }),
        )?;
        // A permanent view is inlined before binding, so the analyzer never asks the
        // policy about it: ask here, before any read and before the text of a stored view
        // can fail, so a principal who may not read it learns nothing from it.
        let mut views = self.authorize_reads(&expansion.reads, env).await?;
        if let Some(e) = expansion.deferred {
            return Err(e);
        }
        // (Boxed: `analyze_in` runs on the deepest stacks, where the size of its
        // future counts.)
        let (synced, renames) =
            Box::pin(self.check_stored_views(&expansion.stored_reads, env)).await?;
        let plan = Plan::try_new(crate::view_schema::apply_renames(&expansion.plan, &renames))?;
        // The bodies of the functions the plan calls read views the same way:
        // the policy is asked about them before the analysis' own result (or
        // a failure inside a view's text) is shown. One boxed future for both
        // steps: `analyze_in` runs on the deepest stacks, where the size of
        // its own future counts.
        let (analyzed, body_views) = Box::pin(async {
            let analyzed = DefaultAnalyzer.analyze(&plan, &env.analysis).await;
            (analyzed, self.authorize_body_reads(env).await)
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
        // Analysis checks every read: its denials are the reads' denials.
        let mut analyzed = crate::access::audit(self, StatementClass::Read, analyzed)?;
        // Spark's `ViewSyncSchemaToMetaStore`: once the query analyzed, evolved views store their new schema. A
        // concurrent change of the view wins (its entry is no longer the one the read found).
        for (found, changed) in synced {
            crate::store::CatalogStore::replace_table(&*self.state.catalog.store, &found, changed);
        }
        for view in views {
            let securable = krishiv_plan::analyzer::Securable::View(view);
            if !analyzed.securables.iter().any(|s| s.securable == securable) {
                analyzed
                    .securables
                    .push(krishiv_plan::analyzer::SecurableUse {
                        securable,
                        privilege: krishiv_plan::analyzer::Privilege::Select,
                        columns: std::collections::BTreeSet::new(),
                    });
            }
        }
        Ok(analyzed)
    }

    /// The views the bodies of the functions a query called read, once the
    /// policy allows them; a failure inside a view's text after that.
    ///
    /// Boxed: `analyze_in` runs on the deepest stacks (a script's nested
    /// statements), where the size of its future counts.
    fn authorize_body_reads<'a>(
        &'a self,
        env: &'a QueryEnv,
    ) -> futures::future::BoxFuture<'a, Result<Vec<krishiv_plan::analyzer::TableIdent>, KrishivError>>
    {
        Box::pin(async move {
            let (reads, deferred) = env.sql_functions.body_reads();
            let views = if reads.is_empty() {
                Vec::new()
            } else {
                Box::pin(self.authorize_reads(&reads, env)).await?
            };
            match deferred {
                Some(e) => Err(e),
                None => Ok(views),
            }
        })
    }

    /// The schema binding of each stored view `reads` inlined (`view_schema`): its query analyzed alone, inner
    /// views first. Returns the evolved entries to store once the whole query analyzed, and the renames of reads.
    ///
    /// # Errors
    /// What [`crate::view_schema::check_read`] raises.
    async fn check_stored_views(
        &self,
        reads: &[crate::view_schema::StoredRead],
        env: &QueryEnv,
    ) -> Result<crate::view_schema::StoredViewChecks, KrishivError> {
        let compensation = crate::view_schema::compensation(&env.config);
        let mut synced = Vec::new();
        let mut renames = Vec::new();
        for read in reads {
            // A view whose query does not analyze alone is left to the outer query.
            let Ok(plan) = Plan::try_new(crate::view_schema::apply_renames(&read.query, &renames))
            else {
                continue;
            };
            let Ok(analyzed) = DefaultAnalyzer.analyze(&plan, &env.analysis).await else {
                continue;
            };
            let case_sensitive = env.analysis.conf.case_sensitive;
            let check =
                crate::view_schema::check_read(read, &analyzed, case_sensitive, compensation)?;
            if let Some(changed) = check.changed {
                synced.push((Arc::clone(&read.entry), changed));
            }
            renames.extend(check.rename);
        }
        Ok((synced, renames))
    }

    /// The physical plan of `analyzed`, once the files it scans are
    /// listed and their footers read.
    pub(crate) async fn physical(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
    ) -> Result<(Plan, PhysicalPlan), KrishivError> {
        let (optimized, physical, _) = self.physical_with(analyzed, env, false).await?;
        Ok((optimized, physical))
    }

    /// As [`Self::physical`] for a plan that is only inspected (EXPLAIN,
    /// estimates): a recursive CTE is not run and reads as an empty table.
    pub(crate) async fn physical_inspected(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
    ) -> Result<(Plan, PhysicalPlan), KrishivError> {
        let (optimized, physical, _) = self.physical_with(analyzed, env, false).await?;
        Ok((optimized, physical))
    }

    /// As [`Self::physical_inspected`] (the plan is only shown, never run);
    /// with `trace`, also every optimizer rule application that changed the
    /// plan (`EXPLAIN VERBOSE`).
    pub(crate) async fn physical_traced(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
        trace: bool,
    ) -> Result<(Plan, PhysicalPlan, Vec<krishiv_optimizer::RuleStep>), KrishivError> {
        self.physical_with(analyzed, env, trace).await
    }

    async fn physical_with(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
        trace: bool,
    ) -> Result<(Plan, PhysicalPlan, Vec<krishiv_optimizer::RuleStep>), KrishivError> {
        let start = query_start_micros();
        // A recursive CTE runs as an operator of the physical plan; the
        // session only tells each loop how many rows its readers need.
        let limited;
        let analyzed = if crate::recursion::has_loop(analyzed.plan.root()) {
            limited = crate::recursion::push_limits(analyzed)?;
            &limited
        } else {
            analyzed
        };
        env.catalog.open_files(analyzed.plan.root()).await?;
        let tables =
            Arc::clone(&env.catalog) as Arc<dyn krishiv_exec::planner::TableProviderResolver>;
        let stats = ProviderStatistics(Arc::clone(&tables));
        let (optimized, steps) = if trace {
            krishiv_optimizer::optimize_traced_with_udfs(
                &analyzed.plan,
                &stats,
                &self.state.udfs,
                &env.config,
            )?
        } else {
            (
                krishiv_optimizer::optimize_with_udfs(
                    &analyzed.plan,
                    &stats,
                    &self.state.udfs,
                    &env.config,
                )?,
                Vec::new(),
            )
        };
        let cx = PlanningContext::new(tables, Arc::clone(&env.config))
            .with_query_start(start)
            .with_user(self.state.principal.name.as_str())
            .with_udfs(Arc::clone(&self.state.udfs));
        let physical = self.state.planner.plan(&optimized, &cx)?;
        // The admission request's estimate, only for a controller that reads it.
        if self.state.admission.uses_estimate() {
            env.catalog.set_estimate(crate::explain::estimate_of(
                &optimized,
                &physical,
                &env.catalog,
            ));
        }
        // Spark logs what analysis and planning ignore (hints) as
        // warnings; `DataFrame::warnings` returns them too.
        for w in analyzed.warnings.iter().chain(physical.warnings()) {
            tracing::warn!(target: "krishiv::session::warnings", "{w}");
        }
        Ok((optimized, physical, steps))
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
    ) -> Result<(RecordBatchStream, Arc<QueryContext>), KrishivError> {
        self.run_on_query(physical, self.worker(env), env, true)
            .await
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

    /// Asks the admission controller for a permit for the query whose
    /// `system.queries` row is `log_id`. A controller that answers at once
    /// leaves the row `running`; one that waits shows it `queued`. A refusal
    /// closes the row as failed with the controller's error class, and a
    /// caller that drops the query while it is queued closes it `cancelled`.
    async fn admit(
        &self,
        log_id: &str,
        env: &QueryEnv,
        exempt: bool,
    ) -> Result<crate::admission::Permit, KrishivError> {
        // A query that reads only `system.*` is never queued behind the load it inspects.
        if exempt {
            return Ok(self.exempt_permit(log_id));
        }
        let request = QueryRequest {
            principal: Arc::clone(&self.state.principal),
            tags: self.state.queries.tags_now(),
            class: WorkloadClass::Interactive,
            estimate: env.catalog.estimate(),
        };
        let controller = Arc::clone(&self.state.admission);
        let mut admit = std::pin::pin!(controller.admit(&request));
        let mut queued = QueuedRow {
            state: Arc::clone(&self.state),
            id: log_id,
            open: true,
        };
        let admitted = match std::future::poll_fn(|cx| Poll::Ready(admit.as_mut().poll(cx))).await {
            Poll::Ready(done) => done,
            Poll::Pending => {
                self.state.queries.set_state(log_id, "queued");
                let done = admit.await;
                if done.is_ok() {
                    self.state.queries.set_state(log_id, "running");
                }
                done
            }
        };
        queued.open = false;
        match admitted {
            Ok(permit) => {
                self.state.queries.mark_started(log_id);
                Ok(permit)
            }
            Err(error) => {
                self.state.queries.record_failed_row(log_id, &error);
                Err(error)
            }
        }
    }

    /// Begins the `system.queries` row of a query and waits for its admission permit. A caller
    /// that is about to change something (a write replacing files) takes the permit first and
    /// passes it on, so a refused or cancelled query has changed nothing.
    pub(crate) async fn admit_query(
        &self,
        env: &QueryEnv,
        exempt: bool,
    ) -> Result<Admitted, KrishivError> {
        // The policy of the query's own configuration snapshot, passed to its row (no shared slot).
        let log_id = self.state.queries.begin_as(
            crate::system::TextPolicy::from_config(&env.config),
            env.text.as_deref(),
            env.statement_class,
            self.operation_id(),
        )?;
        *env.row.lock().unwrap_or_else(PoisonError::into_inner) = Some(log_id.clone());
        let permit = self.admit(&log_id, env, exempt).await?;
        Ok(Admitted::new(Arc::clone(&self.state), log_id, permit))
    }

    /// As [`Self::run_on_query`], also returning the id of the `system.queries` row this execution
    /// began, so the caller keeps it for itself: the statement's slot in `env` is shared by every
    /// execution of the same analyzed frame.
    async fn run_on_query_logged(
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
                .admit_query(env, env.catalog.reads_only_system(physical))
                .await?
                .into_parts(),
        };
        let id = QueryId(self.state.runtime.reserve(
            self.state.number,
            self.state.queries.session_id(),
            &log_id,
        ));
        let pool = self
            .state
            .memory
            .child(&crate::system::pool_name(&log_id), permit.memory_budget());
        let query = QueryContext::new(
            id,
            WorkloadClass::Interactive,
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
        let mut pending = PendingFold(
            Some(QueryFold {
                query: Arc::clone(&query),
                state: Arc::clone(&self.state),
                start: QueryStart::capture(&self.state, &env.io),
                metrics_on,
                log_id,
                _permit: permit,
                _operation: self.operation.clone(),
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
                let submit =
                    self.state
                        .runtime
                        .runner
                        .submit(physical.clone(), worker, Arc::clone(&query));
                let (stream_tx, stream_rx) = tokio::sync::oneshot::channel();
                let (done_tx, done_rx) = tokio::sync::oneshot::channel::<bool>();
                pending.1 = Some(done_rx);
                let abandoned = Arc::clone(&query);
                drop(
                    self.state.runtime.executor.spawn(
                        WorkloadClass::Interactive,
                        Box::pin(
                            async move {
                                let result = submit.await;
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
                if let Some(fold) = pending.0.take() {
                    fold.finish(Some(&error));
                }
                Err(error)
            }
            Ok(stream) => {
                let stream = MetricStream {
                    inner: stream,
                    fold: pending.0.take(),
                };
                Ok((
                    Box::pin(stream.map(|b| b.and_then(into_record_batch))),
                    query,
                    row_id,
                ))
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

impl Session {
    /// What a query's plans are instantiated with in this process: the
    /// query's tables, its UDFs, and its spill scope under the session's
    /// spill directory (created at instantiation). Writes build their sink
    /// from the plan with the query's I/O context, formats and configuration;
    /// batch splits resolve through the session's relation store.
    fn worker(&self, env: &QueryEnv) -> WorkerContext {
        let worker = WorkerContext::new(
            Arc::clone(&env.catalog) as Arc<dyn krishiv_exec::planner::TableProviderResolver>
        )
        .with_udfs(Arc::clone(&self.state.udfs))
        .with_relations(Arc::clone(&self.state.relations))
        .with_spill_scopes(Arc::new(QuerySpill {
            state: Arc::clone(&self.state),
        }))
        .with_io(env.io.clone())
        .with_formats(Arc::clone(&self.state.runtime.formats))
        .with_config(Arc::clone(&env.config));
        self.with_python_runtime(worker)
    }
}

/// Aborts a task when dropped.
struct AbortOnDrop(krishiv_exec::TaskHandle);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[cfg(test)]
mod tests;

mod env;
pub(crate) use env::*;

mod spill;
pub(crate) use spill::*;

mod writes;
use writes::*;

mod ops;
