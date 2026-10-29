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
use krishiv_exec::files::{FileReads, LocalPaths, ObjectPaths, PathLocator as _};
use krishiv_exec::physical::{SpillScopes, WorkerContext, WritePlan};
use krishiv_exec::planner::{
    DefaultPlanner, PhysicalPlan, PhysicalPlanner, PlanningContext, ProviderStatistics,
};
use krishiv_exec::{
    ExecConfig, Executor, ExecutorConfig, QueryContext, QueryId, QueryRunner, Scheduler,
    TokioExecutor, WorkloadClass,
};
use krishiv_io::{
    DirectoryWrite, FormatRegistry, IoConfig, IoScheduler, SaveMode, StoreId, StoreRegistry,
    is_spark_file_source, parse_object_url,
};
use krishiv_memory::{
    MemoryConfig, MemoryPool, RelationStore, SpillConfig, SpillManager, SpillScope, TreePool,
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
use object_store::local::LocalFileSystem;

use crate::catalog::{FileScans, QueryCatalog, QueryCatalogs, SessionCatalog, TempTable};
use crate::catalog::{is_hidden_name, reserved_temp_view_name};
use crate::column::Column;
use crate::conf::{self, ConfStore, RuntimeConf};
use crate::dataframe::{DataFrame, DataFrameReader};
use crate::udfs::SessionFunctions;
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
    /// Fixed for the session's life: caches and fingerprints never mix
    /// identities (a new identity gets a new session).
    pub(crate) principal: Arc<Principal>,
    pub(crate) policy: Arc<dyn AccessPolicy>,
    functions: Arc<dyn FunctionRegistry>,
    /// The session's registered UDFs, shared with the registry above and
    /// the workers running its plans.
    udfs: Arc<UdfLibrary>,
    executor: Arc<dyn Executor>,
    runner: QueryRunner,
    memory: Arc<dyn MemoryPool>,
    /// The spill directory, opened by the first query that plans for
    /// execution (`krishiv.spill.*` are build-time keys).
    spill: Mutex<Option<SpillManager>>,
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
        let udfs = Arc::new(UdfLibrary::new());
        let functions = match self.functions {
            Some(f) => Arc::new(SessionFunctions::with_registry(f, Arc::clone(&udfs)))
                as Arc<dyn FunctionRegistry>,
            None => Arc::new(SessionFunctions::new(Arc::clone(&udfs))) as Arc<dyn FunctionRegistry>,
        };
        Ok(Session {
            state: Arc::new(SessionState {
                conf: Arc::new(ConfStore::new(config)),
                catalog: Arc::new(SessionCatalog::default()),
                variables: Arc::new(SessionVariables::default()),
                profile: self.profile,
                principal: Arc::new(self.principal.unwrap_or_else(Principal::local_user)),
                policy: self.policy.unwrap_or_else(|| Arc::new(AllowAll)),
                functions,
                udfs,
                runner: QueryRunner::new(Arc::clone(&executor)),
                executor,
                memory,
                relations: RelationStore::new(),
                spill: Mutex::new(None),
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

    async fn sql_here(&self, query: &str) -> Result<DataFrame, KrishivError> {
        let env = self.query_env();
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
    async fn run_statement(
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
            Statement::DeclareVariable(v) => self.exec_declare(&v, query, redacted).await,
            Statement::SetVariable(v) => self.exec_set_variable(&v, query, redacted).await,
            Statement::DropVariable(v) => self.exec_drop_variable(&v, redacted).await,
            Statement::CreateTable(t) => self.exec_create_table(&t, redacted).await,
            Statement::CreateTableLike(t) => self.exec_create_table_like(&t, redacted).await,
            Statement::DropTable(d) => self.exec_drop_table(&d).await,
            Statement::Insert(i) => self.exec_insert(&i).await,
            Statement::Truncate(t) => self.exec_truncate(&t).await,
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
    /// with the reserved snapshot prefix `` `__krishiv_snapshot_` `` (in any
    /// letter case), which queries cannot spell either.
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
        let batches: Vec<Batch> = batches.into_iter().map(Batch::new).collect();
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
            policy: Arc::clone(&self.state.policy),
            catalogs: Arc::new(QueryCatalogs(Arc::clone(&catalog))),
            functions: Arc::clone(&self.state.functions),
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
        let analyzed = DefaultAnalyzer
            .analyze(&plan, &env.analysis)
            .await
            .map_err(|e| attach_analysis(e, text));
        // Analysis checks every read: its denials are the reads' denials.
        crate::access::audit(self, StatementClass::Read, analyzed)
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
    /// names, then the directory's commit (`_SUCCESS`, the removal of
    /// orphaned attempt files, and the removal of what an overwrite
    /// replaces). A failed or cancelled write deletes its files and leaves
    /// the directory as it was. The plan carries the write as data; the
    /// commit runs from the plan's result rows (one per file: `path`,
    /// `size`, `rows`).
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
            partition_by,
        } = target;
        let mode = SaveMode::parse(&mode)?;
        if !partition_by.is_empty() {
            return Err(KrishivUnsupportedFeature {
                feature: "writing partitioned output".to_owned(),
            }
            .into());
        }
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
        let Some(write) =
            DirectoryWrite::begin(&io, at.store.clone(), at.path.clone(), mode, job.clone())
                .await?
        else {
            // `ignore` found the path: nothing is written.
            return Ok(());
        };
        let schema = Arc::clone(physical.schema());
        let plan = physical.into_write(WritePlan {
            format: format.name().to_owned(),
            options: options.clone(),
            store: at.store.to_string(),
            dir: at.path.as_ref().to_owned(),
            job_uuid: job,
            max_records_per_file: max_records_per_file(&options, format.name())?,
            target_file_bytes: 0,
        });
        let worker = self.worker(env);
        let mut rows = self.run_on(&plan, worker, env).await?;
        let mut batches = Vec::new();
        while let Some(batch) = rows.next().await {
            batches.push(batch?);
        }
        drop(rows);
        let files = committed_files(&batches)?;
        crate::access::still_physical(self, &at)?;
        write
            .commit(&io, format.as_ref(), schema, &parsed, files)
            .await?;
        Ok(())
    }
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
const PARSER_KEYS: [&str; 9] = [
    "spark.sql.ansi.enabled",
    "spark.sql.ansi.enforceReservedKeywords",
    "spark.sql.ansi.doubleQuotedIdentifiers",
    "spark.sql.legacy.setopsPrecedence.enabled",
    "spark.sql.parser.escapedStringLiterals",
    "spark.sql.legacy.bangEqualsNot",
    "spark.sql.legacy.identifierClause",
    "spark.sql.sources.bucketing.maxBuckets",
    "spark.sql.parser.singleCharacterPipeOperator.enabled",
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
