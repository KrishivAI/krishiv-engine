// SPDX-License-Identifier: Apache-2.0
//! [`Session`] and [`SessionBuilder`], and the query pipeline every
//! action runs: analyze → optimize → physical plan → run.

use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::StreamExt;
use futures::stream::BoxStream;
use krishiv_common::config::Config;
use krishiv_common::error::{
    ColumnAlreadyExists, ErrorQueryContext, InternalError, KrishivError, KrishivUnsupportedFeature,
};
use krishiv_common::redact::RedactedSql;
use krishiv_common::secret::OptionValue;
use krishiv_exec::FileWriteSink;
use krishiv_exec::files::{FileReads, ObjectPaths, PathLocator as _};
use krishiv_exec::physical::{SpillScopes, WorkerContext};
use krishiv_exec::planner::{
    DefaultPlanner, PhysicalPlan, PhysicalPlanner, PlanningContext, ProviderStatistics,
};
use krishiv_exec::{
    ExecConfig, Executor, ExecutorConfig, QueryContext, QueryId, QueryRunner, Scheduler,
    TokioExecutor, WorkloadClass,
};
use krishiv_io::parquet::ParquetFormat;
use krishiv_io::{
    DirectoryWrite, FileFormat, IoConfig, IoScheduler, SaveMode, StoreId, StoreRegistry,
};
use krishiv_memory::{MemoryConfig, MemoryPool, SpillConfig, SpillManager, SpillScope, TreePool};
use krishiv_plan::analyzer::QueryContext as PlanQueryContext;
use krishiv_plan::analyzer::{
    AllowAll, AnalysisContext, AnalysisError, AnalyzedPlan, Analyzer as _, BuiltinFunctions,
    DefaultAnalyzer, FunctionRegistry, Principal, ResolutionConf, SparkProfile, TableColumn,
};
use krishiv_plan::{LogicalPlan, Plan, PlanId, PlanNode, ReadSource, SourceId};
use krishiv_sql::ast::Statement;
use krishiv_sql::{Frontend, ParserOptions};
use krishiv_types::DataType;
use krishiv_types::batch::{Batch, Weights};
use object_store::ObjectStore;
use object_store::local::LocalFileSystem;

use crate::catalog::{FileScans, QueryCatalog, QueryCatalogs, SessionCatalog, TempTable};
use crate::catalog::{is_hidden_name, reserved_temp_view_name};
use crate::column::Column;
use crate::conf::{self, ConfStore, RuntimeConf};
use crate::dataframe::{DataFrame, DataFrameReader};
use crate::variables::SessionVariables;

/// The store id of the local file system.
const LOCAL_STORE: &str = "file";

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
    pub(crate) variables: Arc<SessionVariables>,
    profile: SparkProfile,
    principal: Arc<Principal>,
    functions: Arc<dyn FunctionRegistry>,
    executor: Arc<dyn Executor>,
    runner: QueryRunner,
    memory: Arc<dyn MemoryPool>,
    /// The spill directory, opened by the first query that plans for
    /// execution (`krishiv.spill.*` are build-time keys).
    spill: Mutex<Option<SpillManager>>,
    io: Arc<IoScheduler>,
    formats: Vec<Arc<dyn FileFormat>>,
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
        let (config, _) = Config::from_pairs(self.pairs)?;
        let executor: Arc<dyn Executor> = match self.executor {
            Some(e) => e,
            None => TokioExecutor::new(&ExecutorConfig::default().with_config(&config))?,
        };
        let memory: Arc<dyn MemoryPool> = TreePool::new(&MemoryConfig::from_config(&config));
        let io =
            IoScheduler::with_runtime(IoConfig::from_config(&config), executor.io_handle().clone());
        io.attach_memory(&memory);
        io.register_store(
            StoreId::new(LOCAL_STORE),
            Arc::new(LocalFileSystem::new()) as Arc<dyn ObjectStore>,
        );
        let cwd = std::env::current_dir()
            .ok()
            .and_then(|d| d.to_str().map(str::to_owned))
            .unwrap_or_else(|| "/".to_owned());
        Ok(Session {
            state: Arc::new(SessionState {
                conf: Arc::new(ConfStore::new(config)),
                catalog: Arc::new(SessionCatalog::default()),
                variables: Arc::new(SessionVariables::default()),
                profile: self.profile,
                principal: Arc::new(self.principal.unwrap_or_else(Principal::local_user)),
                functions: self
                    .functions
                    .unwrap_or_else(|| Arc::new(BuiltinFunctions::new())),
                runner: QueryRunner::new(Arc::clone(&executor)),
                executor,
                memory,
                spill: Mutex::new(None),
                io,
                formats: vec![Arc::new(ParquetFormat)],
                cwd,
                planner: Arc::new(DefaultPlanner),
                plan_ids: AtomicI64::new(0),
                query_ids: AtomicU64::new(0),
                sql_sources: AtomicU32::new(1),
            }),
        })
    }
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

    async fn sql_here(&self, query: &str) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
        let redacted = krishiv_sql::redact_sql(query);
        let statement = krishiv_sql::parse_statement(query, &env.parser)
            .map_err(|e| sql_context(e.error, &redacted, e.span.start, e.span.end))?;
        match statement {
            Statement::Query(q) => {
                let plan = krishiv_sql::lower(&q, SourceId(0))
                    .map_err(|e| sql_context(e.error, &redacted, e.span.start, e.span.end))?;
                let df = DataFrame::new(self.clone(), Arc::clone(plan.root()));
                df.analyzed_in_with_text(env, Some(&redacted)).await?;
                Ok(df)
            }
            Statement::Explain(e) => self.explain_statement(&e).await,
            Statement::Set(s) => self.exec_set(&s, &redacted).await,
            Statement::Reset(r) => self.exec_reset(&r).await,
            Statement::CreateView(v) => self.exec_create_view(&v).await,
            Statement::DropView(v) => self.exec_drop_view(&v).await,
            Statement::DeclareVariable(v) => self.exec_declare(&v, query, &redacted).await,
            Statement::SetVariable(v) => self.exec_set_variable(&v, query, &redacted).await,
            Statement::DropVariable(v) => self.exec_drop_variable(&v, &redacted).await,
            _ => Err(unsupported_statement("statements other than queries")),
        }
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
    /// `KRISHIV_RESERVED_TEMP_VIEW_NAME` when `name` starts with the
    /// reserved snapshot prefix `` `__krishiv_snapshot_` `` (in any letter
    /// case), which queries cannot spell either.
    pub fn register_batches(
        &self,
        name: &str,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    ) -> Result<(), KrishivError> {
        if is_hidden_name(name) {
            return Err(reserved_temp_view_name(name));
        }
        let columns = schema
            .fields()
            .iter()
            .map(|f| {
                Ok(TableColumn {
                    name: f.name().clone(),
                    data_type: DataType::from_arrow_field(f)?,
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
        let batches = batches.into_iter().map(Batch::new).collect();
        let table = krishiv_exec::planner::MemoryTable::new(schema, batches);
        self.state.catalog.put(
            name,
            TempTable::Memory {
                columns,
                table: Arc::new(table),
            },
        );
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
        self.state.catalog.drop_table(name)
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
        let (config, conf_version) = self.state.conf.versioned_snapshot();
        // Cloud stores are built from this snapshot, lazily per query:
        // plans carry only path strings, and the worker resolves the store
        // at instantiation (P2-T084).
        let stores = Arc::new(StoreRegistry::new(Arc::clone(&config)));
        let files = Arc::new(FileReads::new(
            self.state.io.context(),
            self.state.formats.clone(),
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
            Arc::clone(&files),
            scans,
        ));
        let resolution = {
            let values = conf_values(&config, &ResolutionConf::KEYS);
            ResolutionConf::from_conf(|k| values.get(k).map(String::as_str))
        };
        let parser = parser_options(&config);
        // One snapshot of the variables per query, like the configuration
        // snapshot: every reference in the query reads the same values.
        let variables = self.state.variables.snapshot();
        let analysis = AnalysisContext {
            profile: self.state.profile,
            conf: resolution,
            principal: Arc::clone(&self.state.principal),
            policy: Arc::new(AllowAll),
            catalogs: Arc::new(QueryCatalogs(Arc::clone(&catalog))),
            functions: Arc::clone(&self.state.functions),
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
        let plan = crate::catalog::expand_temp_views(
            &self.state.catalog,
            plan,
            env.analysis.conf.case_sensitive,
        )?;
        let plan = Plan::try_new(plan)?;
        DefaultAnalyzer
            .analyze(&plan, &env.analysis)
            .await
            .map_err(|e| attach_analysis(e, text))
    }

    /// The physical plan of `analyzed`, once the files it scans are
    /// listed and their footers read.
    pub(crate) async fn physical(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
    ) -> Result<(Plan, PhysicalPlan), KrishivError> {
        let (optimized, physical, _) = self.physical_traced(analyzed, env, false).await?;
        Ok((optimized, physical))
    }

    /// As [`Self::physical`]; with `trace`, also every optimizer rule
    /// application that changed the plan (`EXPLAIN VERBOSE`).
    pub(crate) async fn physical_traced(
        &self,
        analyzed: &AnalyzedPlan,
        env: &QueryEnv,
        trace: bool,
    ) -> Result<(Plan, PhysicalPlan, Vec<krishiv_optimizer::RuleStep>), KrishivError> {
        // Planning makes no spill directory: a query gets its scope when
        // its plan is instantiated (`Self::worker`), so a plan that is only
        // inspected (EXPLAIN, warnings) makes none.
        env.catalog.open_files(analyzed.plan.root()).await?;
        let tables =
            Arc::clone(&env.catalog) as Arc<dyn krishiv_exec::planner::TableProviderResolver>;
        let stats = ProviderStatistics(Arc::clone(&tables));
        let (optimized, steps) = if trace {
            krishiv_optimizer::optimize_traced(&analyzed.plan, &stats)?
        } else {
            (
                krishiv_optimizer::optimize_with(&analyzed.plan, &stats)?,
                Vec::new(),
            )
        };
        let cx = PlanningContext::new(tables, Arc::clone(&env.config))
            .with_user(self.state.principal.name.as_str());
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

    /// Runs `physical` instantiated with `worker`.
    async fn run_on(
        &self,
        physical: &PhysicalPlan,
        worker: WorkerContext,
        env: &QueryEnv,
    ) -> Result<RecordBatchStream, KrishivError> {
        Ok(self.run_on_query(physical, worker, env, false).await?.0)
    }

    async fn run_on_query(
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
        let stream = self
            .state
            .runner
            .submit(physical.clone(), worker, Arc::clone(&query))
            .await?;
        Ok((
            Box::pin(stream.map(|b| b.and_then(into_record_batch))),
            query,
        ))
    }
}

impl Session {
    /// Writes the result of `physical` to the directory `path` in
    /// `format` with `mode` (`df.write`): the files under their final
    /// names, then the directory's commit (`_SUCCESS`, and the removal of
    /// what an overwrite replaces). A failed or cancelled write deletes its
    /// files and leaves the directory as it was.
    pub(crate) async fn write_files(
        &self,
        physical: PhysicalPlan,
        env: &QueryEnv,
        target: WriteTarget,
    ) -> Result<(), KrishivError> {
        let WriteTarget {
            format,
            mode,
            path,
            options,
        } = target;
        let mode = SaveMode::parse(&mode)?;
        unique_columns(physical.schema(), env.analysis.conf.case_sensitive)?;
        let Some(format) = self
            .state
            .formats
            .iter()
            .find(|f| f.name().eq_ignore_ascii_case(&format))
            .cloned()
        else {
            return Err(KrishivUnsupportedFeature {
                feature: format!("writing the {format} format"),
            }
            .into());
        };
        let (parsed, _) = format.options().parse(&options, &env.config)?;
        let Some(text) = path.as_plain() else {
            return Err(KrishivUnsupportedFeature {
                feature: "write paths with credentials".to_owned(),
            }
            .into());
        };
        let stores = Arc::new(StoreRegistry::new(Arc::clone(&env.config)));
        let at = ObjectPaths::new(
            StoreId::new(LOCAL_STORE),
            self.state.cwd.clone(),
            Arc::clone(&self.state.io),
            stores,
        )
        .locate_with_options(text, &options)?;
        let io = self.state.io.context();
        let Some(write) = DirectoryWrite::begin(&io, at.store, at.path, mode, job_uuid()).await?
        else {
            // `ignore` found the path: nothing is written.
            return Ok(());
        };
        let schema = Arc::clone(physical.schema());
        let sink = Arc::new(FileWriteSink::try_new(
            Arc::clone(&format),
            io.clone(),
            Arc::clone(&schema),
            parsed.clone(),
            write.write_options(0),
        )?);
        let plan = physical.into_write(&sink)?;
        let worker = self.worker(env).with_write_sink(Arc::clone(&sink));
        let mut rows = self.run_on(&plan, worker, env).await?;
        while let Some(batch) = rows.next().await {
            batch?;
        }
        drop(rows);
        write
            .commit(&io, format.as_ref(), schema, &parsed, sink.files())
            .await?;
        Ok(())
    }
}

impl Session {
    /// What a query's plans are instantiated with in this process: the
    /// query's tables, and its spill scope under the session's spill
    /// directory (created at instantiation).
    fn worker(&self, env: &QueryEnv) -> WorkerContext {
        WorkerContext::new(
            Arc::clone(&env.catalog) as Arc<dyn krishiv_exec::planner::TableProviderResolver>
        )
        .with_spill_scopes(Arc::new(QuerySpill {
            state: Arc::clone(&self.state),
            config: Arc::clone(&env.config),
        }))
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
const PARSER_KEYS: [&str; 7] = [
    "spark.sql.ansi.enabled",
    "spark.sql.ansi.enforceReservedKeywords",
    "spark.sql.ansi.doubleQuotedIdentifiers",
    "spark.sql.legacy.setopsPrecedence.enabled",
    "spark.sql.parser.escapedStringLiterals",
    "spark.sql.legacy.bangEqualsNot",
    "spark.sql.legacy.identifierClause",
];

/// `KRISHIV_UNSUPPORTED_FEATURE` for a statement `Session::sql` cannot run yet.
fn unsupported_statement(what: &str) -> KrishivError {
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
