// SPDX-License-Identifier: Apache-2.0
//! [`Session`] and [`SessionBuilder`], and the query pipeline every
//! action runs: analyze → optimize → physical plan → run.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};
use std::time::Instant;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::Stream;
use futures::StreamExt;
use futures::stream::BoxStream;
use krishiv_common::config::{Config, ConfigValue, METRICS_ENABLED, PARTITION_OVERWRITE_MODE};
use krishiv_common::error::{
    AllPartitionColumnsNotAllowed, ColumnAlreadyExists, ErrorQueryContext, InternalError,
    KrishivError, KrishivUnsupportedFeature, PartitionColumnNotFoundInSchema,
};
use krishiv_common::redact::RedactedSql;
use krishiv_common::secret::OptionValue;
use krishiv_exec::files::{FileReads, LocalPaths, ObjectPaths, PathLocator as _};
use krishiv_exec::physical::{SpillScopes, WorkerContext, WritePlan};
use krishiv_exec::planner::{
    DefaultPlanner, PhysicalPlan, PhysicalPlanner, PlanningContext, ProviderStatistics,
};
use krishiv_exec::{
    ExecConfig, Executor, ExecutorConfig, MetricsRegistry, MetricsSnapshot, QueryContext, QueryId,
    QueryOutcome, QueryRunner, Scheduler, Status, TokioExecutor, WorkloadClass,
};
use krishiv_io::{
    DirectoryWrite, FormatRegistry, IoConfig, IoCounters, IoScheduler, SaveMode, StoreId,
    StoreRegistry, is_spark_file_source, parse_object_url,
};
use krishiv_memory::{
    ArbitrationStats, MemoryConfig, MemoryPool, RelationStore, SpillConfig, SpillManager,
    SpillScope, TreePool,
};
use krishiv_plan::analyzer::QueryContext as PlanQueryContext;
use krishiv_plan::analyzer::{
    AccessPolicy, AllowAll, AnalysisContext, AnalysisError, AnalyzedPlan, Analyzer as _,
    BuiltinTableFunctions, DefaultAnalyzer, FunctionRegistry, Principal, Privilege, ResolutionConf,
    Securable, SparkProfile, TableColumn, authorize,
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

use crate::cache::RelationCache;
use crate::catalog::{FileScans, QueryCatalog, QueryCatalogs, SessionCatalog, TempTable};
use crate::catalog::{is_global_key, is_reserved_view_name, reserved_temp_view_name};
use crate::column::Column;
use crate::conf::{self, ConfStore, RuntimeConf};
use crate::dataframe::{DataFrame, DataFrameReader};
use crate::sql_functions::{QueryFunctions, SqlFunctions};
use crate::udfs::SessionFunctions;
use crate::variables::SessionVariables;

/// The store id of the local file system.
const LOCAL_STORE: &str = "file";

tokio::task_local! {
    /// The query-start instant (microseconds since the epoch) of the
    /// physical planning in progress on this task.
    static QUERY_START: i64;
}

/// Now, as microseconds since the epoch.
fn query_start_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|d| i64::try_from(d.as_micros()).ok())
        .unwrap_or_default()
}

/// Unpins the hidden tables a planning call pinned when dropped.
struct PinGuard {
    catalog: Arc<crate::catalog::QueryCatalog>,
    names: Vec<String>,
}

impl Drop for PinGuard {
    fn drop(&mut self) {
        self.catalog.unpin_cached(&self.names);
    }
}

/// Unpins a run's per-execution tables when dropped: the run's stream ends
/// or is dropped, or starting it failed.
struct Release {
    catalog: Arc<crate::catalog::QueryCatalog>,
    names: Vec<String>,
}

impl Drop for Release {
    fn drop(&mut self) {
        self.catalog.unpin_cached(&self.names);
    }
}

/// A run's stream, holding the [`Release`] of its tables.
struct Released {
    inner: RecordBatchStream,
    _release: Release,
}

impl futures::Stream for Released {
    type Item = Result<RecordBatch, KrishivError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.inner.poll_next_unpin(cx)
    }
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
    pub(crate) policy: Arc<dyn AccessPolicy>,
    pub(crate) functions: Arc<dyn FunctionRegistry>,
    /// The session's registered UDFs, shared with the registry above and
    /// the workers running its plans.
    udfs: Arc<UdfLibrary>,
    /// The SQL-bodied functions the session defines.
    pub(crate) sql_functions: SqlFunctions,
    executor: Arc<dyn Executor>,
    runner: QueryRunner,
    memory: Arc<dyn MemoryPool>,
    /// The concrete pool tree behind [`Self::memory`]: its arbitration
    /// counters feed the metrics fold.
    memory_tree: Arc<TreePool>,
    /// The engine's metrics registry: every query that reaches execution
    /// folds into it once at query end (when `krishiv.metrics.enabled`).
    metrics: Arc<MetricsRegistry>,
    /// The spill directory, opened by the first query that plans for
    /// execution (`krishiv.spill.*` are build-time keys).
    spill: Arc<Mutex<Option<SpillManager>>>,
    /// Whether a store creates objects only if absent (the commit log's
    /// probe), by store id; per session, never process-wide.
    commit_log_probes: Mutex<HashMap<String, bool>>,
    io: Arc<IoScheduler>,
    formats: Arc<FormatRegistry>,
    /// The session's content-addressed relations: every memory table's
    /// batches as Arrow IPC, so workers resolve a batch range through it
    /// instead of reading rows from the plan.
    relations: Arc<RelationStore>,
    cwd: String,
    planner: Arc<dyn PhysicalPlanner>,
    plan_ids: AtomicI64,
    query_ids: AtomicU64,
    /// SQL source ids handed to view lowerings: every `CREATE TEMPORARY
    /// VIEW` lowers its query with an id of its own, so errors raised
    /// while expanding a view point into the view's text, not the outer
    /// statement's (`SourceId(0)` stays the outer statement's).
    sql_sources: AtomicU32,
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
    /// [`TokioExecutor`] of the session's own).
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
        // Diagnostics first: installs the tokio-console subscriber when the
        // feature is on and `KRISHIV_TOKIO_CONSOLE=1`, a no-op otherwise.
        krishiv_exec::maybe_init_tokio_console();
        let (config, _) = Config::from_pairs(self.pairs)?;
        let executor: Arc<dyn Executor> = match self.executor {
            Some(e) => e,
            None => TokioExecutor::new(&ExecutorConfig::default().with_config(&config))?,
        };
        let memory_tree = TreePool::new(&MemoryConfig::from_config(&config));
        let memory: Arc<dyn MemoryPool> = Arc::<TreePool>::clone(&memory_tree);
        let cache = Arc::new(RelationCache::new(&memory));
        let io =
            IoScheduler::with_runtime(IoConfig::from_config(&config), executor.io_handle().clone());
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
        let metrics = Arc::new(MetricsRegistry::new());
        register_gauges(&metrics, &memory, &io)?;
        let cwd = std::env::current_dir()
            .ok()
            .and_then(|d| d.to_str().map(str::to_owned))
            .unwrap_or_else(|| "/".to_owned());
        let spill = Arc::new(Mutex::new(None));
        {
            let spill = Arc::clone(&spill);
            metrics.add_gauge("krishiv.spill.disk.used", Vec::new(), move || {
                let slot = spill.lock().unwrap_or_else(PoisonError::into_inner);
                Some(slot.as_ref().map_or(0, SpillManager::used_bytes))
            })?;
        }
        let udfs = Arc::new(UdfLibrary::new());
        let functions = match self.functions {
            Some(f) => Arc::new(SessionFunctions::with_registry(f, Arc::clone(&udfs)))
                as Arc<dyn FunctionRegistry>,
            None => Arc::new(SessionFunctions::new(Arc::clone(&udfs))) as Arc<dyn FunctionRegistry>,
        };
        Ok(Session {
            locals: Arc::from([]),
            state: Arc::new(SessionState {
                conf: Arc::new(ConfStore::new(config)),
                catalog: Arc::new(SessionCatalog::default()),
                cache,
                variables: Arc::new(SessionVariables::default()),
                profile: self.profile,
                principal: Arc::new(self.principal.unwrap_or_else(Principal::local_user)),
                policy: self.policy.unwrap_or_else(|| Arc::new(AllowAll)),
                functions,
                udfs,
                sql_functions: SqlFunctions::default(),
                commit_log_probes: Mutex::new(HashMap::new()),
                runner: QueryRunner::new(Arc::clone(&executor)),
                executor,
                memory,
                memory_tree,
                metrics,
                relations: RelationStore::new(),
                spill,
                io,
                formats: Arc::new(FormatRegistry::with_defaults()),
                cwd,
                planner: Arc::new(DefaultPlanner),
                plan_ids: AtomicI64::new(0),
                query_ids: AtomicU64::new(0),
                sql_sources: AtomicU32::new(1),
            }),
        })
    }
}

/// Registers the engine gauges on `metrics`: closures over the owning
/// components, evaluated at scrape time, never pushed.
fn register_gauges(
    metrics: &Arc<MetricsRegistry>,
    memory: &Arc<dyn MemoryPool>,
    io: &Arc<IoScheduler>,
) -> Result<(), KrishivError> {
    {
        let memory = Arc::clone(memory);
        metrics.add_gauge("krishiv.memory.reserved", Vec::new(), move || {
            u64::try_from(memory.reserved()).ok()
        })?;
    }
    {
        let memory = Arc::clone(memory);
        metrics.add_gauge("krishiv.memory.limit", Vec::new(), move || {
            memory.effective_limit().and_then(|l| u64::try_from(l).ok())
        })?;
    }
    {
        let io = Arc::clone(io);
        metrics.add_gauge(
            "krishiv.cache.size",
            vec![("cache".to_owned(), "metadata".to_owned())],
            move || Some(io.cache().bytes()),
        )?;
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
        // own gauges and leak every session's registry (and its spill
        // manager) forever.
        let registry = Arc::downgrade(metrics);
        metrics.add_gauge(
            "krishiv.query.active",
            vec![("class".to_owned(), class.as_str().to_owned())],
            move || registry.upgrade().map(|r| r.active_queries(class)),
        )?;
    }
    Ok(())
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
        let (session, query) = (self.clone(), query.to_owned());
        self.on_executor(async move { session.sql_here(&query).await })
            .await
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

    pub(crate) async fn sql_here(&self, query: &str) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let query = self.substituted(query);
        let query: &str = &query;
        // `IDENTIFIER(expr)` names are evaluated first, so the statement is parsed, authorized and
        // run with the names they stand for.
        let resolved = self.resolve_identifier_clauses(query, &env.parser).await?;
        let query: &str = &resolved;
        let redacted = krishiv_sql::redact_sql(query);
        let statement = krishiv_sql::parse_statement(query, &env.parser)
            .map_err(|e| sql_context(e.error, &redacted, e.span.start, e.span.end))?;
        // Before any side effect: what the statement writes, creates or
        // drops (its reads are checked by analysis, first thing).
        let class = krishiv_sql::classify(&statement);
        self.authorize_statement(&statement, class, &env).await?;
        self.run_statement(statement, env, query, &redacted).await
    }

    /// Runs the parsed, authorized `statement`.
    pub(crate) async fn run_statement(
        &self,
        statement: Statement,
        env: QueryEnv,
        query: &str,
        redacted: &RedactedSql,
    ) -> Result<DataFrame, KrishivError> {
        match statement {
            Statement::Query(q) => {
                let plan = krishiv_sql::lower(&q, SourceId(0))
                    .map_err(|e| sql_context(e.error, redacted, e.span.start, e.span.end))?;
                let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
                df.analyzed_in_with_text(env, Some(redacted)).await?;
                Ok(df)
            }
            Statement::Explain(e) => self.explain_statement(&e).await,
            Statement::Set(s) => self.exec_set(&s, redacted).await,
            Statement::Reset(r) => self.exec_reset(&r).await,
            Statement::CreateView(v) => self.exec_create_view(&v).await,
            Statement::DropView(v) => self.exec_drop_view(&v).await,
            Statement::Cache(c) => Box::pin(self.exec_cache(&c, query)).await,
            Statement::Refresh(r) => self.exec_refresh(&r).await,
            Statement::DeclareVariable(v) => self.exec_declare(&v, query, redacted).await,
            Statement::SetVariable(v) => self.exec_set_variable(&v, query, redacted).await,
            Statement::DropVariable(v) => self.exec_drop_variable(&v, redacted).await,
            Statement::ExecuteImmediate(e) => {
                Box::pin(self.exec_execute_immediate(&e, query, redacted, 0)).await
            }
            Statement::CreateFunction(f) => self.exec_create_function(&f, redacted).await,
            Statement::DropFunction(f) => self.exec_drop_function(&f).await,
            Statement::CreateTable(t) => Box::pin(self.exec_create_table(&t, query)).await,
            Statement::CreateTableLike(t) => Box::pin(self.exec_create_table_like(&t, query)).await,
            Statement::DropTable(d) => self.exec_drop_table(&d).await,
            Statement::CreateNamespace(n) => self.exec_create_namespace(&n, query).await,
            Statement::DropNamespace(n) => self.exec_drop_namespace(&n).await,
            Statement::Use(u) => self.exec_use(&u).await,
            Statement::Show(s) => Box::pin(self.exec_show(&s, query)).await,
            Statement::Describe(d) => Box::pin(self.exec_describe(&d, query)).await,
            Statement::CreatePermanentView(v) => self.exec_create_permanent_view(&v, query).await,
            Statement::DropPermanentView(d) => self.exec_drop_permanent_view(&d).await,
            Statement::AlterView(a) => self.exec_alter_view(&a, query).await,
            Statement::AlterTable(a) => Box::pin(self.exec_alter_table(&a, query)).await,
            Statement::Insert(i) => Box::pin(self.exec_insert(&i, query)).await,
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
        let columns = schema
            .fields()
            .iter()
            .map(|f| {
                Ok(TableColumn {
                    name: f.name().clone(),
                    data_type: DataType::from_arrow_field(f)?.refuse_collations()?,
                    nullable: f.is_nullable(),
                })
            })
            .collect::<Result<Vec<_>, KrishivError>>()?;
        if let Some(b) = batches.iter().find(|b| b.schema() != schema) {
            return Err(internal(format!(
                "a batch of schema {} for a table of schema {schema}",
                b.schema()
            )));
        }
        let schema = crate::carrier::canonical_schema(schema);
        let batches = batches
            .into_iter()
            .map(|b| crate::carrier::recarried(&b, &schema).map(Batch::new))
            .collect::<Result<Vec<Batch>, KrishivError>>()?;
        // The batches as content-addressed IPC in the session's store: the
        // table's scans split into batch ranges that workers resolve
        // through it. The table holds the lease but not the batches, so
        // they live once, memory-accounted; the lease releases the entry
        // when the table is dropped or replaced.
        let rows: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
        let lease = self.state.relations.insert(&schema, &batches)?;
        let table = krishiv_exec::planner::MemoryTable::from_lease(schema, rows, lease);
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
    /// `FAILED_EXECUTE_UDF` when the UDF's `name` panics, is empty, or
    /// contains a `.` or a backtick, since such a UDF can never be called
    /// by name (the function name is `unknown` when there is none).
    pub fn register_udf(
        &self,
        udf: impl krishiv_udf_sdk::ScalarUdf,
    ) -> Result<(), krishiv_common::error::KrishivError> {
        let udf: Arc<dyn krishiv_udf_sdk::ScalarUdf> = Arc::new(udf);
        if let Err(e) = self.state.udfs.register(Arc::clone(&udf)) {
            let name = guard_call("unknown", "name", || udf.name().to_owned())
                .map(|name| name.to_lowercase())
                .unwrap_or_else(|_| "unknown".to_owned());
            let name = if name.is_empty() {
                "unknown".to_owned()
            } else {
                name
            };
            return Err(krishiv_functions::scalar::failed_execute(
                &name,
                "unknown",
                "unknown",
                e.message(),
            ));
        }
        Ok(())
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

    /// Runs `work` on the session's executor, so that it finds the
    /// runtime its I/O needs whatever runtime (if any) polls the caller;
    /// dropping the returned future aborts the work.
    pub(crate) async fn on_executor<T: Send + 'static>(
        &self,
        work: impl Future<Output = Result<T, KrishivError>> + Send + 'static,
    ) -> Result<T, KrishivError> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let task = self.state.executor.spawn(
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

    /// The analysis inputs of one query.
    pub(crate) fn query_env(&self) -> QueryEnv {
        // Before any analysis loads a table: a fill is refused if an
        // invalidation ran after this point (it may hold the replaced rows).
        let cache_generation = self.state.cache.generation();
        let (config, conf_version) = self.state.conf.query_snapshot();
        // Cloud stores are built from this snapshot, lazily per query:
        // plans carry only path strings, and the worker resolves the store
        // at listing from the read's option values (P2-T084).
        let stores = Arc::new(StoreRegistry::new(Arc::clone(&config)));
        let files = Arc::new(FileReads::new(
            self.state.io.context(),
            Arc::clone(&self.state.formats),
            Arc::new(ObjectPaths::new(
                StoreId::new(LOCAL_STORE),
                self.state.cwd.clone(),
                Arc::clone(&self.state.io),
                Arc::clone(&stores),
            )),
            Arc::clone(&config),
        ));
        let scans = FileScans {
            io: self.state.io.context(),
            config: Arc::clone(&config),
            slots: u64::try_from(self.state.executor.parallelism().max(1)).unwrap_or(1),
            runtime: self.state.executor.io_handle().clone(),
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
            .with_path(catalog.sql_path_snapshot()),
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
            config,
            parser,
            analysis,
            catalog,
            conf_version,
            catalog_epoch,
            variables_version: variables.version,
            policy_version: self.state.policy.version().0,
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
        let synced = Box::pin(self.check_stored_views(&expansion.stored_reads, env)).await?;
        let plan = expansion.plan;
        let plan = Plan::try_new(plan)?;
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
            crate::store::CatalogStore::replace_table(&self.state.catalog.store, &found, changed);
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
    /// views first. Returns the evolved entries to store once the whole query analyzed. A query that does not
    /// analyze alone is left to the analysis of the whole query, which reports it as it would without views.
    ///
    /// # Errors
    /// What [`crate::view_schema::check_read`] raises.
    async fn check_stored_views(
        &self,
        reads: &[crate::view_schema::StoredRead],
        env: &QueryEnv,
    ) -> Result<Vec<(Arc<crate::store::CatalogTable>, crate::store::CatalogTable)>, KrishivError>
    {
        let compensation = crate::view_schema::compensation(&env.config);
        let mut synced = Vec::new();
        for read in reads {
            // A view whose query does not analyze is left to the outer query, which raises that error itself;
            // the views after it still get their checks.
            let Ok(plan) = Plan::try_new(Arc::clone(&read.query)) else {
                continue;
            };
            let Ok(analyzed) = DefaultAnalyzer.analyze(&plan, &env.analysis).await else {
                continue;
            };
            if let Some(changed) = crate::view_schema::check_read(
                read,
                &analyzed,
                env.analysis.conf.case_sensitive,
                compensation,
            )? {
                synced.push((Arc::clone(&read.entry), changed));
            }
        }
        Ok(synced)
    }

    /// The physical plan of `analyzed`, once the files it scans are
    /// listed and their footers read.
    pub(crate) async fn physical(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
    ) -> Result<(Plan, PhysicalPlan), KrishivError> {
        let (optimized, physical, _) = self.physical_with(analyzed, env, false, true).await?;
        Ok((optimized, physical))
    }

    /// An empty reservation in a pool of its own under the session's, for
    /// the rows a recursive CTE keeps.
    pub(crate) fn recursion_reservation(&self) -> krishiv_memory::Reservation {
        let pool = self.state.memory.child("recursive-cte", None);
        let consumer = pool.register("recursive-cte", None);
        krishiv_memory::Reservation::new(pool, consumer)
    }

    /// As [`Self::physical`] for a plan that is only inspected (EXPLAIN,
    /// estimates): a recursive CTE is not run and reads as an empty table.
    pub(crate) async fn physical_inspected(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
    ) -> Result<(Plan, PhysicalPlan), KrishivError> {
        let (optimized, physical, _) = self.physical_with(analyzed, env, false, false).await?;
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
        self.physical_with(analyzed, env, trace, false).await
    }

    /// `run_loops` false: recursive CTEs are not run (see
    /// [`Self::physical_inspected`]).
    async fn physical_with(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
        trace: bool,
        run_loops: bool,
    ) -> Result<(Plan, PhysicalPlan, Vec<krishiv_optimizer::RuleStep>), KrishivError> {
        // One query-start instant for the plan and every recursion round it
        // plans (`current_timestamp()` is constant within a query): the
        // outermost call takes it, nested ones read it.
        let start = QUERY_START
            .try_with(|s| *s)
            .unwrap_or_else(|_| query_start_micros());
        // The hidden tables the planning pins are released when this call
        // ends in an error, is cancelled, or only inspects the plan; a plan
        // that will run keeps them, and the run releases the ones its scans
        // name (`Self::run_on_query`).
        let mut pins = PinGuard {
            catalog: Arc::clone(&env.catalog),
            names: Vec::new(),
        };
        let planned = QUERY_START
            .scope(
                start,
                self.physical_pinned(analyzed, env, trace, run_loops, &mut pins.names, start),
            )
            .await;
        if run_loops && planned.is_ok() {
            env.catalog.mark_releasable(std::mem::take(&mut pins.names));
        }
        planned
    }

    async fn physical_pinned(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
        trace: bool,
        run_loops: bool,
        pins: &mut Vec<String>,
        start: i64,
    ) -> Result<(Plan, PhysicalPlan, Vec<krishiv_optimizer::RuleStep>), KrishivError> {
        // Planning makes no spill directory: a query gets its scope when
        // its plan is instantiated (`Self::worker`), so a plan that is only
        // inspected (EXPLAIN, warnings) makes none.
        let expanded;
        let analyzed = if crate::recursion::has_loop(analyzed.plan.root()) {
            let (plan, hidden) = self.expand_loops(analyzed, env, run_loops).await?;
            pins.extend(hidden);
            expanded = plan;
            &expanded
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
        // Spark logs what analysis and planning ignore (hints) as
        // warnings; `DataFrame::warnings` returns them too.
        for w in analyzed.warnings.iter().chain(physical.warnings()) {
            tracing::warn!(target: "krishiv::session", "{w}");
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

    /// Runs `physical` as `task` in this process (tests): the worker of
    /// [`Self::worker`] instantiating with the task's scan splits, executed
    /// on the session's executor (I/O needs its runtime).
    #[cfg(test)]
    pub(crate) async fn run_task(
        &self,
        physical: &PhysicalPlan,
        env: &QueryEnv,
        task: &krishiv_exec::physical::TaskSpec,
    ) -> Result<Vec<RecordBatch>, KrishivError> {
        use krishiv_exec::physical::instantiate_task;
        let worker = self.worker(env);
        let node = instantiate_task(physical, &worker, task)?;
        let state = Arc::clone(&self.state);
        let config = Arc::clone(&env.config);
        self.on_executor(async move {
            let id = QueryId(state.query_ids.fetch_add(1, Ordering::Relaxed));
            let pool = state.memory.child(&format!("query-task-{}", id.0), None);
            let query = QueryContext::new(
                id,
                WorkloadClass::Interactive,
                ExecConfig::from_config(&config),
                pool,
            );
            let mut stream = state.runner.execute(node, Arc::clone(&query)).await?;
            let mut out = Vec::new();
            while let Some(batch) = stream.next().await {
                out.push(batch?.into_parts().0);
            }
            query.finished().await;
            Ok(out)
        })
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
        let release = Release {
            catalog: Arc::clone(&env.catalog),
            names: env
                .catalog
                .take_releasable(&crate::recursion::loop_tables(physical)),
        };
        let (stream, query) = self
            .run_on_query_inner(physical, worker, env, detailed)
            .await?;
        Ok((
            Box::pin(Released {
                inner: stream,
                _release: release,
            }),
            query,
        ))
    }

    async fn run_on_query_inner(
        &self,
        physical: &PhysicalPlan,
        worker: WorkerContext,
        env: &QueryEnv,
        detailed: bool,
    ) -> Result<(RecordBatchStream, Arc<QueryContext>), KrishivError> {
        let id = QueryId(self.state.query_ids.fetch_add(1, Ordering::Relaxed));
        let pool = self.state.memory.child(&format!("query-{}", id.0), None);
        let query = QueryContext::new(
            id,
            WorkloadClass::Interactive,
            ExecConfig {
                detailed_metrics: detailed,
                ..ExecConfig::from_config(&env.config)
            },
            pool,
        );
        if !metrics_enabled(&env.config) {
            let stream = self
                .state
                .runner
                .submit(physical.clone(), worker, Arc::clone(&query))
                .await?;
            return Ok((
                Box::pin(stream.map(|b| b.and_then(into_record_batch))),
                query,
            ));
        }
        // The fold below runs exactly once per started query; without it a
        // query is invisible to the registry. The guard folds it when this
        // future is dropped before the stream exists (a cancel during
        // `submit`).
        self.state.metrics.query_begin(query.class());
        let mut pending = PendingFold(Some(QueryFold {
            query: Arc::clone(&query),
            state: Arc::clone(&self.state),
            start: QueryStart::capture(&self.state),
        }));
        match self
            .state
            .runner
            .submit(physical.clone(), worker, Arc::clone(&query))
            .await
        {
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
        self.state.metrics.snapshot()
    }
}

/// The process counters a query-end fold attributes, and when the query
/// started: read at query start so the fold records the deltas.
struct QueryStart {
    started: Instant,
    arbitration: ArbitrationStats,
    cache_hits: u64,
    cache_misses: u64,
}

impl QueryStart {
    fn capture(state: &SessionState) -> Self {
        let totals = state.io.totals();
        Self {
            started: Instant::now(),
            arbitration: state.memory_tree.stats(),
            cache_hits: IoCounters::get(&totals.cache_hits),
            cache_misses: IoCounters::get(&totals.cache_misses),
        }
    }

    /// Folds the query's metrics with the deltas since [`Self::capture`]:
    /// the one fold of this query. `error` is the stream's terminal error,
    /// if any; otherwise the query's own error decides the outcome (none is
    /// success). A zero wall time means the query never finished (a
    /// cancelled stream dropped early): the duration then runs to now.
    fn fold(self, state: &SessionState, query: &QueryContext, error: Option<&KrishivError>) {
        let stored = query.error();
        let (status, error_class) = Status::from_error(error.or(stored.as_ref()));
        let metrics = query.metrics();
        let wall = metrics.wall_time;
        // A zero wall time means the query never finished (a cancelled
        // stream dropped early): the duration then runs from query start
        // to the fold.
        let duration = if wall.is_zero() {
            self.started.elapsed()
        } else {
            wall
        };
        let totals = state.io.totals();
        let arbitration = state.memory_tree.stats();
        state.metrics.fold(&QueryOutcome {
            class: query.class(),
            status,
            error_class: error_class.as_deref(),
            duration_secs: duration.as_secs_f64(),
            metrics: &metrics,
            arbitration_spills: arbitration
                .spill_requests
                .saturating_sub(self.arbitration.spill_requests),
            arbitration_aborts: arbitration.aborts.saturating_sub(self.arbitration.aborts),
            cache_hits: IoCounters::get(&totals.cache_hits).saturating_sub(self.cache_hits),
            cache_misses: IoCounters::get(&totals.cache_misses).saturating_sub(self.cache_misses),
        });
    }
}

/// The one fold of a running query, taken when its stream ends.
struct QueryFold {
    query: Arc<QueryContext>,
    state: Arc<SessionState>,
    start: QueryStart,
}

impl QueryFold {
    fn finish(self, error: Option<&KrishivError>) {
        self.start.fold(&self.state, &self.query, error);
    }
}

/// The fold of a query whose stream does not exist yet: dropped while still
/// holding it (the caller cancelled), it cancels the query and folds it.
struct PendingFold(Option<QueryFold>);

impl Drop for PendingFold {
    fn drop(&mut self) {
        if let Some(fold) = self.0.take() {
            fold.query.cancel();
            fold.finish(None);
        }
    }
}

/// The result stream with the query's fold attached: the stream's end
/// (batches, the query's error, or a drop) folds the query's metrics once.
struct MetricStream {
    inner: BoxStream<'static, Result<Batch, KrishivError>>,
    fold: Option<QueryFold>,
}

impl Stream for MetricStream {
    type Item = Result<Batch, KrishivError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(None) => {
                if let Some(fold) = self.fold.take() {
                    fold.finish(None);
                }
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(error))) => {
                if let Some(fold) = self.fold.take() {
                    fold.finish(Some(&error));
                }
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(Some(Ok(batch))) => Poll::Ready(Some(Ok(batch))),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for MetricStream {
    fn drop(&mut self) {
        if let Some(fold) = self.fold.take() {
            // The caller went away: cancel first (a recorded error wins),
            // then fold what the drivers merged so far.
            fold.query.cancel();
            fold.finish(None);
        }
    }
}

impl Session {
    /// Writes the result of `physical` to the directory `path` in
    /// `format` with `mode` (`df.write`): the files under their final
    /// names, then the directory's commit (`_SUCCESS`, the removal of
    /// orphaned attempt files, and the removal of what an overwrite
    /// replaces). A failed or cancelled write deletes its files and leaves
    /// the directory as it was. The plan carries the write as data; the
    /// commit runs from the plan's result rows (one per file: `path`,
    /// `size`, `rows`). Returns the directories below `path` that hold the
    /// committed files (`col=value/...` of a partitioned write, in their
    /// escaped form; empty for an unpartitioned one or when nothing was
    /// written).
    pub(crate) async fn write_files(
        &self,
        physical: PhysicalPlan,
        env: &QueryEnv,
        target: WriteTarget,
    ) -> Result<Vec<String>, KrishivError> {
        let WriteTarget {
            format,
            mode,
            path,
            options,
            partition_by,
        } = target;
        let mode = SaveMode::parse(&mode)?;
        unique_columns(physical.schema(), env.analysis.conf.case_sensitive)?;
        let format = match self.state.formats.get(&format) {
            Ok(format) => format,
            Err(e) => {
                if is_spark_file_source(&format) {
                    return Err(KrishivUnsupportedFeature {
                        feature: format!("writing the {format} format"),
                    }
                    .into());
                }
                return Err(e);
            }
        };
        let (parsed, _) = format.options().parse(&options, &env.config)?;
        let Some(text) = path.as_plain() else {
            return Err(KrishivUnsupportedFeature {
                feature: "write paths with credentials".to_owned(),
            }
            .into());
        };
        let schema = Arc::clone(physical.schema());
        let partitions =
            resolve_partitions(&schema, &partition_by, env.analysis.conf.case_sensitive)?;
        if !partitions.is_empty() && partitions.len() == schema.fields().len() {
            return Err(AllPartitionColumnsNotAllowed {}.into());
        }
        let dynamic =
            !partitions.is_empty() && partition_overwrite_is_dynamic(&env.config, &options)?;
        let max_records = max_records_per_file(&options, format.name())?;
        // Cloud stores are built from the query snapshot with the write's
        // per-relation `fs.*` options overriding it (Spark's per-relation
        // Hadoop configuration).
        let stores = Arc::new(StoreRegistry::new(Arc::clone(&env.config)));
        let at = ObjectPaths::new(
            StoreId::new(LOCAL_STORE),
            self.state.cwd.clone(),
            Arc::clone(&self.state.io),
            stores,
        )
        .locate_with_options(text, &options)?;
        // The directory as it physically is (links in its existing part
        // resolved): the policy is asked about the directory that is
        // written, never about a name that a link sends elsewhere.
        let at = crate::access::physical(at)?;
        let io = self.state.io.context();
        // Before anything is written or replaced (an overwrite deletes
        // first): `Modify` and `Create`, both whether or not the path
        // exists, so a denial never tells whether it does.
        let securable = Securable::Path(OptionValue::from_user("path", at.display.clone()));
        for privilege in [Privilege::Modify, Privilege::Create] {
            let r = authorize(
                &*self.state.policy,
                &self.state.principal,
                &securable,
                privilege,
            )
            .await;
            crate::access::audit(self, StatementClass::Write, r)?;
        }
        // A link swapped into the checked path since fails the write
        // before anything is replaced, and again before the commit deletes
        // what an overwrite replaces.
        crate::access::still_physical(self, &at)?;
        let job = job_uuid();
        // Every name of the write is resolved relative to the directory
        // opened here, never through a link (P2-T199). The registration
        // lives until the write is over.
        let (store, _registration) = self.anchor_local_write(&at, &job)?;
        let written = async {
            let Some(write) = DirectoryWrite::begin(
                &io,
                store.clone(),
                at.path.clone(),
                mode,
                job.clone(),
                dynamic,
            )
            .await?
            .map(|w| w.partitioned(!partitions.is_empty())) else {
                // `ignore` found the path: nothing is written.
                return Ok(Vec::new());
            };
            // Boxed: the statement futures are on the recursion of nested
            // scripts, so a larger write future would overflow their stack.
            let write = if Box::pin(uses_commit_log(
                &env.config,
                &self.state.commit_log_probes,
                &io,
                &store,
                &at,
                &job,
            ))
            .await?
            {
                Box::pin(write.logged(&io)).await?
            } else {
                write
            };
            let plan = physical.into_write(WritePlan {
                format: format.name().to_owned(),
                options: options.clone(),
                store: store.key().to_owned(),
                dir: at.path.as_ref().to_owned(),
                job_uuid: job,
                max_records_per_file: max_records,
                target_file_bytes: 0,
                partition_by: partitions.iter().map(|(i, _)| *i).collect(),
            })?;
            let worker = self.worker(env);
            let mut rows = self.run_on_query(&plan, worker, env, false).await?.0;
            let mut batches = Vec::new();
            while let Some(batch) = rows.next().await {
                batches.push(batch?);
            }
            drop(rows);
            let files = committed_files(&batches)?;
            if let Err(denied) = crate::access::still_physical(self, &at) {
                // The files the query wrote are under the anchor: remove
                // them (best effort) before refusing.
                for f in &files {
                    drop(io.delete(&store, &f.path).await);
                }
                return Err(denied);
            }
            let committed = write
                .commit(&io, format.as_ref(), schema, &parsed, files)
                .await?;
            // The engine's own write may have replaced a cached path: entries
            // reading files are dropped, so the next action lists them again.
            // (An external change drops nothing and stays invisible.)
            self.refresh_files();
            let mut dirs = std::collections::BTreeSet::new();
            for f in &committed {
                let below: Vec<String> = f
                    .path
                    .prefix_match(&at.path)
                    .map(|parts| parts.map(|p| percent_decoded(p.as_ref())).collect())
                    .unwrap_or_default();
                if let Some((_, leaf)) = below.split_last()
                    && !leaf.is_empty()
                {
                    dirs.insert(leaf.join("/"));
                }
            }
            Ok(dirs.into_iter().collect())
        }
        .await;
        // A store error after the anchor is open may be a link swapped into
        // the path: the audited denial is what the caller sees then.
        if written.is_err()
            && let Err(denied) = crate::access::still_physical(self, &at)
            && denied.error_class().name() == "KRISHIV_INSUFFICIENT_PERMISSIONS"
        {
            return Err(denied);
        }
        written
    }

    /// The store a directory write at `at` goes through: for a local path
    /// on Unix a [`krishiv_io::BeneathStore`] anchored at the deepest
    /// existing directory above the target, registered for this write only
    /// (messages and `EXPLAIN` still name `file`); otherwise the plain
    /// store of `at`.
    fn anchor_local_write(
        &self,
        at: &krishiv_exec::files::LocatedPath,
        job: &str,
    ) -> Result<(StoreId, Option<StoreRegistration>), KrishivError> {
        #[cfg(unix)]
        if let Some(local) = at.display.strip_prefix("file:") {
            match krishiv_io::BeneathStore::anchor(std::path::Path::new(local)) {
                Ok(anchored) => {
                    let id = at.store.private_to(job);
                    self.state
                        .io
                        .register_store(id.clone(), Arc::new(anchored) as Arc<dyn ObjectStore>);
                    let registration = StoreRegistration {
                        io: Arc::clone(&self.state.io),
                        id: id.clone(),
                    };
                    return Ok((id, Some(registration)));
                }
                Err(krishiv_io::AnchorError::Swapped(_)) => {
                    // A link or file where a directory was checked: the
                    // audited denial if the path now resolves elsewhere.
                    crate::access::still_physical(self, at)?;
                    return Err(anchor_failed(
                        at,
                        "a component of the path is not a directory",
                    ));
                }
                Err(krishiv_io::AnchorError::UnsupportedName(name)) => {
                    return Err(KrishivUnsupportedFeature {
                        feature: format!(
                            "writing below the directory {name:?} (a name with non-ASCII or \
                             reserved characters above the target)"
                        ),
                    }
                    .into());
                }
                Err(krishiv_io::AnchorError::Io(e)) => {
                    return Err(anchor_failed(at, &e.to_string()));
                }
            }
        }
        let _ = job;
        Ok((at.store.clone(), None))
    }
}

/// Whether the write overwrites dynamically: the `partitionOverwriteMode`
/// write option (any case), else `spark.sql.sources.partitionOverwriteMode`,
/// is `DYNAMIC`.
///
/// # Errors
/// `KRISHIV_INVALID_FORMAT_OPTION_VALUE` for an option value other than
/// `STATIC` or `DYNAMIC`.
fn partition_overwrite_is_dynamic(
    config: &Config,
    options: &BTreeMap<String, OptionValue>,
) -> Result<bool, KrishivError> {
    let from_option = options
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("partitionOverwriteMode"));
    if let Some((key, value)) = from_option {
        let text = value.as_plain().unwrap_or("<secret>");
        return match text.to_ascii_uppercase().as_str() {
            "DYNAMIC" => Ok(true),
            "STATIC" => Ok(false),
            _ => Err(krishiv_common::error::KrishivInvalidFormatOptionValue {
                value: format!("'{text}'"),
                option: format!("\"{key}\""),
                format: "\"partitioned write\"".to_owned(),
                reason: "expected STATIC or DYNAMIC".to_owned(),
            }
            .into()),
        };
    }
    Ok(config
        .get(&PARTITION_OVERWRITE_MODE)
        .and_then(ConfigValue::as_str)
        == Some("DYNAMIC"))
}

/// Whether this write goes through the commit log:
/// `krishiv.sql.files.commitProtocol` `LOG`, or `AUTO` where the store
/// creates objects only if absent (probed once per write; a store that
/// fails the probe falls back to the plain commit with a warning).
async fn uses_commit_log(
    config: &Config,
    probes: &Mutex<HashMap<String, bool>>,
    io: &krishiv_io::IoContext,
    store: &StoreId,
    at: &krishiv_exec::files::LocatedPath,
    job: &str,
) -> Result<bool, KrishivError> {
    let protocol = config
        .get(&krishiv_common::config::FILES_COMMIT_PROTOCOL)
        .and_then(ConfigValue::as_str)
        .map(str::to_ascii_uppercase);
    if protocol.as_deref() == Some("NONE") {
        return Ok(false);
    }
    let log = krishiv_io::commit_log::CommitLog::new(
        io,
        store.clone(),
        at.path.clone(),
        at.display.clone(),
    );
    if log.exists().await? {
        // A directory that already has a log stays logged.
        return Ok(true);
    }
    let known = probes
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(at.store.key())
        .copied();
    let supported = match known {
        Some(supported) => supported,
        None => {
            let supported = log.supports_put_if_absent(job).await?;
            probes
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(at.store.key().to_owned(), supported);
            supported
        }
    };
    if !supported {
        if protocol.as_deref() == Some("LOG") {
            return Err(krishiv_common::error::KrishivUnsupportedFeature {
                feature: format!(
                    "the commit log on a store without put-if-absent ({})",
                    at.display
                ),
            }
            .into());
        }
        tracing::warn!(
            path = %at.display,
            "the store does not create objects only if absent: writing without the commit log"
        );
    }
    Ok(supported)
}

/// The `maxRecordsPerFile` write option (`FILE_SOURCE_OPTIONS`), case
/// insensitive; 0 (unlimited) when unset.
fn max_records_per_file(
    options: &BTreeMap<String, OptionValue>,
    format: &str,
) -> Result<u64, KrishivError> {
    let Some((key, value)) = options
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("maxRecordsPerFile"))
    else {
        return Ok(0);
    };
    let Some(raw) = value.as_plain() else {
        return Err(krishiv_common::error::KrishivInvalidFormatOptionValue {
            value: "<secret>".to_owned(),
            option: format!("\"{key}\""),
            format: format!("\"{format}\""),
            reason: "a secret is not allowed here".to_owned(),
        }
        .into());
    };
    raw.trim().parse::<u64>().map_err(|_| {
        krishiv_common::error::KrishivInvalidFormatOptionValue {
            value: format!("'{raw}'"),
            option: format!("\"{key}\""),
            format: format!("\"{format}\""),
            reason: "expected an integer in 0..=18446744073709551615".to_owned(),
        }
        .into()
    })
}

/// A write directory that could not be opened, as the path-based store
/// reports it (`KRISHIV_OBJECT_STORE_REQUEST_FAILED`).
fn anchor_failed(at: &krishiv_exec::files::LocatedPath, reason: &str) -> KrishivError {
    krishiv_common::error::KrishivObjectStoreRequestFailed {
        path: at.display.clone(),
        attempts: "1".to_owned(),
        reason: reason.to_owned(),
    }
    .into()
}

/// Removes a write's private store registration when the write is over.
struct StoreRegistration {
    io: Arc<IoScheduler>,
    id: StoreId,
}

impl Drop for StoreRegistration {
    fn drop(&mut self) {
        self.io.unregister_store(&self.id);
    }
}

/// The commit's files from the write plan's result rows (one per file:
/// `path`, `size`, `rows`).
fn committed_files(
    batches: &[RecordBatch],
) -> Result<Vec<krishiv_io::WrittenFileInfo>, KrishivError> {
    use arrow_array::Int64Array;
    use arrow_array::StringViewArray;
    use krishiv_io::{FileStatistics, Precision};
    let mut out = Vec::new();
    for batch in batches {
        if batch.num_columns() != 3 {
            return Err(internal(
                "a write result without path, size and rows".to_owned(),
            ));
        }
        let path = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .ok_or_else(|| internal("a write result whose path is not a string".to_owned()))?;
        let size = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| internal("a write result whose size is not an integer".to_owned()))?;
        let rows = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| internal("a write result whose rows are not an integer".to_owned()))?;
        for i in 0..batch.num_rows() {
            let size = u64::try_from(size.value(i))
                .map_err(|_| internal(format!("a written file of {} bytes", size.value(i))))?;
            let rows = u64::try_from(rows.value(i))
                .map_err(|_| internal(format!("a written file of {} rows", rows.value(i))))?;
            out.push(krishiv_io::WrittenFileInfo {
                path: object_store::path::Path::from(path.value(i)),
                size,
                rows,
                statistics: FileStatistics::with_rows(Precision::Exact(rows)),
            });
        }
    }
    Ok(out)
}

/// The `partitionBy` columns as `(index, name)` pairs in `partitionBy`
/// order, deduplicated.
///
/// # Errors
/// `PARTITION_COLUMN_NOT_FOUND_IN_SCHEMA` for a name the schema has no
/// column for.
fn resolve_partitions(
    schema: &SchemaRef,
    partition_by: &[String],
    case_sensitive: bool,
) -> Result<Vec<(usize, String)>, KrishivError> {
    let mut out = Vec::new();
    for name in partition_by {
        let mut matches = schema.fields().iter().enumerate().filter(|(_, f)| {
            if case_sensitive {
                f.name() == name
            } else {
                f.name().eq_ignore_ascii_case(name)
            }
        });
        let Some((index, field)) = matches.next() else {
            return Err(PartitionColumnNotFoundInSchema {
                column: format!("`{}`", name.replace('`', "``")),
                schema: schema_simple_string(schema),
            }
            .into());
        };
        if matches.next().is_some() {
            return Err(internal(format!(
                "partition column `{name}` matches several columns"
            )));
        }
        // Spark rejects a repeated partition column
        // (`checkColumnNameDuplication`).
        if out.iter().any(|(i, _)| *i == index) {
            return Err(ColumnAlreadyExists {
                column_name: format!("`{}`", field.name().replace('`', "``")),
            }
            .into());
        }
        // Checked here so that nothing is started for a column type that
        // has no partition path text.
        krishiv_io::check_partition_column(field.name(), field.data_type())?;
        out.push((index, field.name().clone()));
    }
    Ok(out)
}

/// `schema` as Spark prints one (`struct<id:int,...>`).
fn schema_simple_string(schema: &SchemaRef) -> String {
    let fields: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| {
            let ty = DataType::from_arrow(f.data_type())
                .map(|t| t.simple_string())
                .unwrap_or_else(|_| f.data_type().to_string());
            format!("{}:{ty}", f.name())
        })
        .collect();
    format!("struct<{}>", fields.join(","))
}

impl Session {
    /// `path` in the qualified form reads and writes use (`file:/a/b`),
    /// which the access policy is asked about.
    pub(crate) fn qualify_path(&self, path: &str) -> Result<String, KrishivError> {
        // No store is built here: cloud paths qualify through their canonical
        // display (as reads qualify them), local paths through `LocalPaths`.
        if let Ok(url) = parse_object_url(path) {
            return Ok(url.display);
        }
        Ok(
            LocalPaths::new(StoreId::new(LOCAL_STORE), self.state.cwd.clone())
                .locate(path)?
                .display,
        )
    }

    /// What a query's plans are instantiated with in this process: the
    /// query's tables, its UDFs, and its spill scope under the session's
    /// spill directory (created at instantiation). Writes build their sink
    /// from the plan with the query's I/O context, formats and configuration;
    /// batch splits resolve through the session's relation store.
    fn worker(&self, env: &QueryEnv) -> WorkerContext {
        WorkerContext::new(
            Arc::clone(&env.catalog) as Arc<dyn krishiv_exec::planner::TableProviderResolver>
        )
        .with_udfs(Arc::clone(&self.state.udfs))
        .with_relations(Arc::clone(&self.state.relations))
        .with_spill_scopes(Arc::new(QuerySpill {
            state: Arc::clone(&self.state),
            config: Arc::clone(&env.config),
        }))
        .with_io(self.state.io.context())
        .with_formats(Arc::clone(&self.state.formats))
        .with_config(Arc::clone(&env.config))
    }
}

/// The spill scope of one query: its own directory under the session's
/// spill directory (opened by the first query that runs; `krishiv.spill.*`
/// are build-time keys), removed when the query's operators and streams
/// are dropped (cancellation included).
struct QuerySpill {
    state: Arc<SessionState>,
    config: Arc<Config>,
}

impl fmt::Debug for QuerySpill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuerySpill").finish_non_exhaustive()
    }
}

impl SpillScopes for QuerySpill {
    fn scope(&self) -> krishiv_exec::Result<SpillScope> {
        let mut slot = self
            .state
            .spill
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if slot.is_none() {
            let manager = SpillManager::open(&SpillConfig::from_config(&self.config))
                .map_err(|e| internal(format!("cannot set up the spill directory: {e}")))?;
            *slot = Some(manager);
        }
        slot.as_ref().map_or_else(
            || Err(internal("the spill directory is not open".to_owned())),
            |m| {
                m.scope().map_err(|e| {
                    internal(format!("cannot create the query's spill directory: {e}"))
                })
            },
        )
    }
}

/// Spark's `COLUMN_ALREADY_EXISTS` for a written schema with a repeated
/// name (ignoring case unless `case_sensitive`, then named in lower case,
/// as Spark 4.2 reports it).
fn unique_columns(schema: &SchemaRef, case_sensitive: bool) -> Result<(), KrishivError> {
    let mut seen = std::collections::HashSet::new();
    for f in schema.fields() {
        let name = if case_sensitive {
            f.name().clone()
        } else {
            f.name().to_lowercase()
        };
        if !seen.insert(name.clone()) {
            return Err(ColumnAlreadyExists {
                column_name: format!("`{}`", name.replace('`', "``")),
            }
            .into());
        }
    }
    Ok(())
}

/// Where and how `df.write` writes.
pub(crate) struct WriteTarget {
    pub(crate) format: String,
    pub(crate) mode: String,
    pub(crate) path: OptionValue,
    pub(crate) options: BTreeMap<String, OptionValue>,
    pub(crate) partition_by: Vec<String>,
}

/// A fresh job id naming a write's files (`part-00000-<id>-c000...`), in
/// UUID form. Its bits come from the standard library's per-process
/// random hash keys, a clock reading and a counter, so two writes never
/// share one, within or across processes.
fn job_uuid() -> String {
    use std::hash::{BuildHasher, Hasher};
    static WRITES: AtomicU64 = AtomicU64::new(0);
    let n = WRITES.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let half = |salt: u64| {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(salt);
        h.write_u64(n);
        h.write_u128(nanos);
        h.write_u32(std::process::id());
        h.finish()
    };
    let (hi, lo) = (half(1), half(2));
    // Version 4, RFC 4122 variant.
    let hi = (hi & !0xF000) | 0x4000;
    let lo = (lo & !(0xC << 60)) | (0x8 << 60);
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        hi >> 32,
        (hi >> 16) & 0xFFFF,
        hi & 0xFFFF,
        lo >> 48,
        lo & 0xFFFF_FFFF_FFFF
    )
}

/// The parser's configs (`ParserOptions::from_conf` reads these).
const PARSER_KEYS: [&str; 11] = [
    "spark.sql.ansi.enabled",
    "spark.sql.ansi.enforceReservedKeywords",
    "spark.sql.ansi.doubleQuotedIdentifiers",
    "spark.sql.legacy.setopsPrecedence.enabled",
    "spark.sql.parser.escapedStringLiterals",
    "spark.sql.legacy.bangEqualsNot",
    "spark.sql.legacy.identifierClause",
    "spark.sql.legacy.consecutiveStringLiterals.enabled",
    "spark.sql.sources.bucketing.maxBuckets",
    "spark.sql.parser.singleCharacterPipeOperator.enabled",
    "spark.sql.legacy.viewSchemaBindingMode",
];

/// `KRISHIV_UNSUPPORTED_FEATURE` for a statement `Session::sql` cannot run yet.
pub(crate) fn unsupported_statement(what: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: format!("{what} in sessions"),
    }
    .into()
}

/// An empty redaction, for analyses with no SQL text (DataFrame errors
/// carry a DataFrame context, which needs no text).
fn empty_redacted() -> RedactedSql {
    RedactedSql::opaque("")
}

/// `error` with the SQL span `start..end` of `text` attached as its
/// query context (`stop` is inclusive, as Spark reports it).
pub(crate) fn sql_context(
    error: KrishivError,
    text: &RedactedSql,
    start: usize,
    end: usize,
) -> KrishivError {
    error.with_query_context(ErrorQueryContext::sql(
        text.fragment(start..end),
        start,
        end.saturating_sub(1),
    ))
}

/// An analysis error with its query context attached: the SQL span of
/// `text`, or the DataFrame operation that raised it. With no text, a
/// SQL span has no fragment to attach, so only DataFrame contexts cross.
fn attach_analysis(e: AnalysisError, text: Option<&RedactedSql>) -> KrishivError {
    let empty;
    let redacted = match text {
        Some(t) => t,
        None => {
            empty = empty_redacted();
            &empty
        }
    };
    match e.query_context(redacted) {
        Some(PlanQueryContext::Sql {
            start,
            stop,
            fragment,
        }) if text.is_some() => e
            .error
            .with_query_context(ErrorQueryContext::sql(fragment, start, stop)),
        Some(PlanQueryContext::DataFrame { .. }) => {
            e.error.with_query_context(ErrorQueryContext::dataframe())
        }
        // A DataFrame action with no origin (e.g. an unbound `col("x")`):
        // the error still arose from a DataFrame operation.
        None if text.is_none() => e.error.with_query_context(ErrorQueryContext::dataframe()),
        _ => e.error,
    }
}

/// The parser options `config` sets.
fn parser_options(config: &Config) -> ParserOptions {
    let values = conf_values(config, &PARSER_KEYS);
    ParserOptions::from_conf(|k| values.get(k).map(String::as_str))
}

/// The values of `keys` that are set or have a default.
fn conf_values<'k>(config: &Config, keys: &[&'k str]) -> BTreeMap<&'k str, String> {
    keys.iter()
        .filter_map(|k| conf::get(config, k).map(|v| (*k, v)))
        .collect()
}

/// The analysis inputs of one query and the versions they were taken at.
pub(crate) struct QueryEnv {
    pub(crate) config: Arc<Config>,
    pub(crate) parser: ParserOptions,
    pub(crate) analysis: AnalysisContext,
    pub(crate) catalog: Arc<QueryCatalog>,
    pub(crate) conf_version: u64,
    pub(crate) catalog_epoch: u64,
    /// The variable version the query's snapshot was taken at: a cached
    /// resolved plan is reused only while it is unchanged.
    pub(crate) variables_version: u64,
    /// The cache's invalidation generation before the query was analyzed.
    pub(crate) cache_generation: u64,
    /// `krishiv.cache.maxBytes` of the query's configuration snapshot.
    pub(crate) cache_cap: usize,
    /// The registry in `analysis`, as itself: the function map the query
    /// reads and the texts of function bodies.
    pub(crate) sql_functions: Arc<QueryFunctions>,
    /// [`FunctionRegistry::version`] of `analysis.functions`.
    pub(crate) functions_version: u64,
    /// The access policy's version when the query started: an analysis made under another one
    /// is not reused, since its reads were authorized under the old rules.
    pub(crate) policy_version: u64,
}

/// A batch query's result rows: insert-only, so a weight other than 1 is a
/// broken invariant, never dropped silently.
fn into_record_batch(batch: Batch) -> Result<RecordBatch, KrishivError> {
    match batch.into_parts() {
        (data, Weights::Const(1)) => Ok(data),
        _ => Err(internal("a batch query produced weighted rows".to_owned())),
    }
}

pub(crate) fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// The `path` option values of `paths`, typed (a URL's credentials stay
/// secret).
pub(crate) fn typed_paths<S: AsRef<str>>(paths: impl IntoIterator<Item = S>) -> Vec<OptionValue> {
    paths
        .into_iter()
        .map(|p| OptionValue::from_user("path", p.as_ref()))
        .collect()
}

/// A read by path of `format`.
pub(crate) fn read_files(
    format: &str,
    paths: Vec<OptionValue>,
    options: BTreeMap<String, OptionValue>,
) -> LogicalPlan {
    LogicalPlan::new(PlanNode::Read(ReadSource::Files {
        format: format.to_owned(),
        paths,
        options,
    }))
}

/// Aborts a task when dropped.
struct AbortOnDrop(krishiv_exec::TaskHandle);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// An object-store path part as the file system names it: object_store percent-encodes characters it does
/// not keep in a part (`%` among them), so `13%253A33` is the directory `13%3A33`.
fn percent_decoded(part: &str) -> String {
    let bytes = part.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = |b: u8| char::from(b).to_digit(16);
        if bytes[i] == b'%'
            && let (Some(Some(h)), Some(Some(l))) = (
                bytes.get(i + 1).map(|b| hex(*b)),
                bytes.get(i + 2).map(|b| hex(*b)),
            )
        {
            out.push(u8::try_from(h * 16 + l).unwrap_or(b'?'));
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::{committed_files, max_records_per_file};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use arrow_array::{Int32Array, Int64Array, RecordBatch, StringViewArray};
    use arrow_schema::{DataType, Field, Schema};
    use krishiv_common::secret::OptionValue;

    fn options(pairs: &[(&str, &str)]) -> BTreeMap<String, OptionValue> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), OptionValue::from_user(k, *v)))
            .collect()
    }

    /// The access policy is asked about the qualified path: a cloud URL
    /// keeps its canonical form (no store is built), a local path gets the
    /// `file:` form.
    #[test]
    fn qualify_path_keeps_cloud_urls_and_qualifies_local_paths() {
        let session = super::Session::new().unwrap();
        assert_eq!(
            session.qualify_path("s3a://bucket/k").unwrap(),
            "s3a://bucket/k"
        );
        assert_eq!(session.qualify_path("/a/b").unwrap(), "file:/a/b");
    }

    #[test]
    fn max_records_per_file_reads_the_write_option() {
        assert_eq!(
            max_records_per_file(&BTreeMap::new(), "parquet").unwrap(),
            0
        );
        assert_eq!(
            max_records_per_file(&options(&[("maxRecordsPerFile", "10")]), "parquet").unwrap(),
            10
        );
        assert_eq!(
            max_records_per_file(&options(&[("MAXRECORDSPERFILE", " 7 ")]), "parquet").unwrap(),
            7
        );
        let e =
            max_records_per_file(&options(&[("maxRecordsPerFile", "ten")]), "parquet").unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_INVALID_FORMAT_OPTION_VALUE"
        );
        let secret = BTreeMap::from([(
            "maxRecordsPerFile".to_owned(),
            OptionValue::reference("s", "k"),
        )]);
        let e = max_records_per_file(&secret, "parquet").unwrap_err();
        assert_eq!(
            e.error_class().name(),
            "KRISHIV_INVALID_FORMAT_OPTION_VALUE"
        );
    }

    fn result_batch(paths: &[&str], sizes: &[i64], rows: &[i64]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("path", DataType::Utf8View, false),
            Field::new("size", DataType::Int64, false),
            Field::new("rows", DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringViewArray::from_iter_values(paths.iter().copied())),
                Arc::new(Int64Array::from(sizes.to_vec())),
                Arc::new(Int64Array::from(rows.to_vec())),
            ],
        )
        .unwrap()
    }

    #[test]
    fn committed_files_reads_path_size_and_rows() {
        let batch = result_batch(&["a/p", "a/q"], &[10, 20], &[1, 2]);
        let files = committed_files(std::slice::from_ref(&batch)).unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].path.as_ref(), "a/p");
        assert_eq!((files[0].size, files[0].rows), (10, 1));
        assert_eq!((files[1].size, files[1].rows), (20, 2));
        assert!(committed_files(&[]).unwrap().is_empty());
    }

    #[test]
    fn committed_files_refuses_a_malformed_result() {
        let batch = result_batch(&["a/p"], &[10], &[1]);
        // Fewer columns.
        let narrow = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "path",
                DataType::Utf8View,
                false,
            )])),
            vec![Arc::clone(batch.column(0))],
        )
        .unwrap();
        assert!(
            committed_files(std::slice::from_ref(&narrow))
                .unwrap_err()
                .to_string()
                .contains("path, size and rows")
        );
        // A path that is not a string.
        let schema = Arc::new(Schema::new(vec![
            Field::new("path", DataType::Int32, false),
            Field::new("size", DataType::Int64, false),
            Field::new("rows", DataType::Int64, false),
        ]));
        let bad = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(Int64Array::from(vec![10])),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        assert!(
            committed_files(std::slice::from_ref(&bad))
                .unwrap_err()
                .to_string()
                .contains("not a string")
        );
        // Negative sizes and rows.
        let bad = result_batch(&["a/p"], &[-1], &[1]);
        assert!(
            committed_files(std::slice::from_ref(&bad))
                .unwrap_err()
                .to_string()
                .contains("bytes")
        );
        let bad = result_batch(&["a/p"], &[10], &[-1]);
        assert!(
            committed_files(std::slice::from_ref(&bad))
                .unwrap_err()
                .to_string()
                .contains("rows")
        );
    }
}
