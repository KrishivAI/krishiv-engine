// SPDX-License-Identifier: Apache-2.0
//! [`DataFrame`], [`GroupedData`] and [`DataFrameReader`].
//!
//! Transformations build the unresolved logical plan and never fail; the
//! plan is analyzed at the first action (Spark Connect's behaviour; `sql`
//! analyzes at once, as Spark does). An analysis is reused while the
//! configuration and the temporary tables stay the same, except for plans
//! that read files, which list them again at every action.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, PoisonError};

use arrow_array::RecordBatch;
use arrow_schema::{Schema, SchemaRef};
use futures::StreamExt;
use krishiv_common::error::KrishivError;
use krishiv_common::secret::OptionValue;
use krishiv_plan::analyzer::{AnalyzedPlan, TableColumn};
use krishiv_plan::{
    Expr, Grouping, JoinCondition, JoinKind, Literal, LogicalPlan, Origin, PlanId, PlanNode,
    ReadSource,
};

use crate::catalog::TempTable;
use crate::column::Column;
use crate::session::{QueryEnv, RecordBatchStream, Session, WriteTarget, read_files, typed_paths};

/// How [`DataFrame::join`] combines rows (Spark's `how`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum JoinType {
    /// `inner`.
    Inner,
    /// `left` / `left_outer`.
    LeftOuter,
    /// `right` / `right_outer`.
    RightOuter,
    /// `full` / `full_outer`.
    FullOuter,
    /// `left_semi`.
    LeftSemi,
    /// `left_anti`.
    LeftAnti,
}

impl JoinType {
    fn kind(self) -> JoinKind {
        match self {
            Self::Inner => JoinKind::Inner,
            Self::LeftOuter => JoinKind::LeftOuter,
            Self::RightOuter => JoinKind::RightOuter,
            Self::FullOuter => JoinKind::FullOuter,
            Self::LeftSemi => JoinKind::LeftSemi,
            Self::LeftAnti => JoinKind::LeftAnti,
        }
    }
}

/// What [`DataFrame::explain`] prints (Spark's modes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExplainMode {
    /// The physical plan.
    Simple,
    /// The parsed, analyzed, optimized and physical plans.
    Extended,
    /// The physical plan outline, then each node's details.
    Formatted,
    /// The optimized plan with statistics, then the physical plan.
    Cost,
    /// The physical plan's generated-code subtrees (none: operators are
    /// vectorized, see ADR 0023).
    Codegen,
}

/// An analysis and the query inputs it was made with.
pub(crate) struct Analysis {
    pub(crate) env: Arc<QueryEnv>,
    pub(crate) analyzed: Arc<AnalyzedPlan>,
}

/// A lazy query: its rows are computed by actions (`collect`,
/// `execute_stream`, ...). Cheap to clone.
///
/// ```
/// use krishiv_session::Session;
/// use krishiv_session::functions::{col, lit};
///
/// let session = Session::new()?;
/// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![
/// #     arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false),
/// #     arrow_schema::Field::new("b", arrow_schema::DataType::Utf8, true),
/// # ]));
/// # session.register_batches("t", schema, vec![])?;
/// let df = futures::executor::block_on(session.sql("SELECT a, b FROM t"))?
///     .select([col("b"), col("a").alias("first")]);
/// let schema = futures::executor::block_on(df.schema())?;
/// assert_eq!(schema.field(1).name(), "first");
/// # Ok::<(), krishiv_common::error::KrishivError>(())
/// ```
#[derive(Clone)]
pub struct DataFrame {
    pub(crate) session: Session,
    pub(crate) plan: Arc<LogicalPlan>,
    analysis: Arc<Mutex<Option<Arc<Analysis>>>>,
    /// An error found while building the frame (an invalid table name),
    /// raised by its first action, as Spark Connect raises it.
    invalid: Option<KrishivError>,
}

impl fmt::Debug for DataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DataFrame")
            .field("plan", &self.plan.to_string())
            .finish_non_exhaustive()
    }
}

impl DataFrame {
    /// A DataFrame of `plan`; its root gets a plan id of its own, so that
    /// [`Self::col`] names its columns.
    pub(crate) fn new(session: Session, plan: Arc<LogicalPlan>) -> Self {
        let plan = if plan.origin.plan_id.is_some() {
            plan
        } else {
            let mut origin = plan.origin;
            origin.plan_id = Some(session.next_plan_id());
            Arc::new(LogicalPlan::clone(&plan).with_origin(origin))
        };
        Self {
            session,
            plan,
            analysis: Arc::default(),
            invalid: None,
        }
    }

    /// A DataFrame that fails with `error` at its first action; `plan`
    /// stands in for the plan it could not build.
    pub(crate) fn invalid(session: Session, plan: Arc<LogicalPlan>, error: KrishivError) -> Self {
        Self {
            invalid: Some(error),
            ..Self::new(session, plan)
        }
    }

    fn id(&self) -> Option<PlanId> {
        self.plan.origin.plan_id
    }

    /// A new DataFrame of `node` over this one's session.
    fn derive(&self, node: PlanNode) -> Self {
        let id = self.session.next_plan_id();
        let plan = LogicalPlan::new(node).with_origin(Origin::plan(id));
        Self {
            session: self.session.clone(),
            plan: Arc::new(plan),
            analysis: Arc::default(),
            invalid: self.invalid.clone(),
        }
    }

    /// The column `name` of this DataFrame (PySpark's `df["a"]`): it
    /// resolves among this DataFrame's columns even where another relation
    /// has a column of the same name, as in a self-join.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![
    /// #     arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false),
    /// #     arrow_schema::Field::new("b", arrow_schema::DataType::Utf8, true),
    /// # ]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let df = futures::executor::block_on(s.sql("SELECT a FROM t"))?;
    /// let left = df.clone().alias("l");
    /// let c = df.col("a");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn col(&self, name: &str) -> Column {
        Column::named(name, self.id())
    }

    /// `SELECT cols` (PySpark's `select`).
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").select([col("a"), col("b").alias("c")]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn select(&self, cols: impl IntoIterator<Item = impl Into<Column>>) -> Self {
        self.derive(PlanNode::Project {
            input: Arc::clone(&self.plan),
            exprs: exprs(cols),
        })
    }

    /// Rows where `condition` is true (PySpark's `filter`).
    ///
    /// ```
    /// # use krishiv_session::functions::{col, lit};
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").filter(col("a").gt(lit(1)));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn filter(&self, condition: Column) -> Self {
        self.derive(PlanNode::Filter {
            input: Arc::clone(&self.plan),
            predicate: Arc::new(condition.expr),
        })
    }

    /// The same as [`Self::filter`] (PySpark's `where`).
    ///
    /// ```
    /// # use krishiv_session::functions::{col, lit};
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").where_(col("a").is_not_null());
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn where_(&self, condition: Column) -> Self {
        self.filter(condition)
    }

    /// This DataFrame with the column `name` added, or replaced
    /// (PySpark's `withColumn`).
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").with_column("doubled", col("a") * 2);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn with_column(&self, name: &str, value: Column) -> Self {
        self.derive(PlanNode::WithColumns {
            input: Arc::clone(&self.plan),
            columns: vec![(name.to_owned(), Arc::new(value.expr))],
        })
    }

    /// Groups rows by `keys`, for [`GroupedData::agg`] (PySpark's
    /// `groupBy`).
    ///
    /// ```
    /// # use krishiv_session::functions::{col, sum};
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").group_by([col("k")]).agg([sum(col("v")).alias("total")]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn group_by(&self, keys: impl IntoIterator<Item = impl Into<Column>>) -> GroupedData {
        GroupedData {
            df: self.clone(),
            keys: keys.into_iter().map(|k| k.into().expr).collect(),
        }
    }

    /// Aggregates over all rows (PySpark's `agg`).
    ///
    /// ```
    /// # use krishiv_session::functions::{col, max};
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").agg([max(col("v"))]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn agg(&self, aggregates: impl IntoIterator<Item = Column>) -> Self {
        self.group_by(Vec::<Column>::new()).agg(aggregates)
    }

    /// Joins with `right` on `condition` (PySpark's `join(other, on, how)`
    /// with a column expression).
    ///
    /// ```
    /// # use krishiv_session::JoinType;
    /// # let s = krishiv_session::Session::new()?;
    /// let (l, r) = (s.table("l"), s.table("r"));
    /// let df = l.join(&r, l.col("id").eq(r.col("id")), JoinType::Inner);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn join(&self, right: &Self, condition: Column, how: JoinType) -> Self {
        self.join_with(
            right,
            how.kind(),
            JoinCondition::On(Arc::new(condition.expr)),
        )
    }

    /// Joins with `right` on equal values of the columns `using`, which
    /// appear once in the result (PySpark's `join(other, ["id"], how)`).
    ///
    /// ```
    /// # use krishiv_session::JoinType;
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("l").join_using(&s.table("r"), ["id"], JoinType::LeftOuter);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn join_using<'a>(
        &self,
        right: &Self,
        using: impl IntoIterator<Item = &'a str>,
        how: JoinType,
    ) -> Self {
        let using = using.into_iter().map(str::to_owned).collect();
        self.join_with(right, how.kind(), JoinCondition::Using(using))
    }

    /// Every pair of rows (PySpark's `crossJoin`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("l").cross_join(&s.table("r"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn cross_join(&self, right: &Self) -> Self {
        self.join_with(right, JoinKind::Cross, JoinCondition::None)
    }

    fn join_with(&self, right: &Self, kind: JoinKind, condition: JoinCondition) -> Self {
        let mut joined = self.derive(PlanNode::Join {
            left: Arc::clone(&self.plan),
            right: Arc::clone(&right.plan),
            kind,
            condition,
        });
        if joined.invalid.is_none() {
            joined.invalid.clone_from(&right.invalid);
        }
        joined
    }

    /// Rows sorted by `keys`, ascending unless a key says
    /// [`Column::desc`] (PySpark's `orderBy`).
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").order_by([col("a").desc(), col("b")]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn order_by(&self, keys: impl IntoIterator<Item = impl Into<Column>>) -> Self {
        self.derive(PlanNode::Sort {
            input: Arc::clone(&self.plan),
            order: keys.into_iter().map(|k| k.into().sort_order()).collect(),
            global: true,
        })
    }

    /// The same as [`Self::order_by`] (PySpark's `sort`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").sort(["a"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn sort(&self, keys: impl IntoIterator<Item = impl Into<Column>>) -> Self {
        self.order_by(keys)
    }

    /// The first `n` rows (PySpark's `limit`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").limit(10);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn limit(&self, n: i32) -> Self {
        self.derive(PlanNode::Limit {
            input: Arc::clone(&self.plan),
            limit: Arc::new(Expr::literal(Literal::Integer(n))),
        })
    }

    /// This DataFrame without its first `n` rows (PySpark's `offset`);
    /// `df.offset(m).limit(n)` is `LIMIT n OFFSET m`.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").offset(10).limit(5);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn offset(&self, n: i32) -> Self {
        self.derive(PlanNode::Offset {
            input: Arc::clone(&self.plan),
            offset: Arc::new(Expr::literal(Literal::Integer(n))),
        })
    }

    /// This DataFrame under the name `alias`, for qualified column names
    /// (PySpark's `alias`).
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").alias("x").select([col("x.a")]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn alias(&self, alias: &str) -> Self {
        self.derive(PlanNode::SubqueryAlias {
            input: Arc::clone(&self.plan),
            alias: alias.to_owned(),
            columns: Vec::new(),
        })
    }

    /// The unresolved logical plan.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let plan = s.table("t").limit(1).plan();
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[cfg(feature = "unstable-ir")]
    #[must_use]
    pub fn plan(&self) -> &Arc<LogicalPlan> {
        &self.plan
    }

    /// The analysis of this DataFrame's plan.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![
    /// #     arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false),
    /// #     arrow_schema::Field::new("b", arrow_schema::DataType::Utf8, true),
    /// # ]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let df = futures::executor::block_on(s.sql("SELECT a FROM t"))?;
    /// let analyzed = futures::executor::block_on(df.analyzed())?;
    /// assert_eq!(analyzed.output[0].name, "a");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Spark's analysis errors.
    #[cfg(feature = "unstable-ir")]
    pub async fn analyzed(&self) -> Result<Arc<AnalyzedPlan>, KrishivError> {
        let df = self.clone();
        self.session
            .on_executor(async move { Ok(Arc::clone(&df.analysis().await?.analyzed)) })
            .await
    }

    /// The analysis to run with: the cached one while the configuration and
    /// the temporary tables are unchanged and no files are read (by path or
    /// through a file view), else a fresh one, which lists them again.
    pub(crate) async fn analysis(&self) -> Result<Arc<Analysis>, KrishivError> {
        let cached = self
            .analysis
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let env = self.session.query_env();
        if let Some(a) = cached
            && !a.analyzed.reads_files
            && !a.env.catalog.loaded_file_views()
            && a.env.conf_version == env.conf_version
            && a.env.catalog_epoch == env.catalog_epoch
        {
            return Ok(a);
        }
        self.analyzed_in(env).await
    }

    /// Analyzes the plan in `env` and keeps the result.
    pub(crate) async fn analyzed_in(&self, env: QueryEnv) -> Result<Arc<Analysis>, KrishivError> {
        self.analyzed_in_with_text(env, None).await
    }

    /// As [`Self::analyzed_in`], with `text` the redacted query the plan
    /// came from, so an analysis error carries its SQL span as context.
    pub(crate) async fn analyzed_in_with_text(
        &self,
        env: QueryEnv,
        text: Option<&krishiv_common::redact::RedactedSql>,
    ) -> Result<Arc<Analysis>, KrishivError> {
        if let Some(e) = &self.invalid {
            return Err(e.clone());
        }
        let analyzed = self.session.analyze_in(&self.plan, &env, text).await?;
        let analysis = Arc::new(Analysis {
            env: Arc::new(env),
            analyzed: Arc::new(analyzed),
        });
        *self.analysis.lock().unwrap_or_else(PoisonError::into_inner) = Some(Arc::clone(&analysis));
        Ok(analysis)
    }

    /// The result's schema: names, Arrow types and nullability.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![
    /// #     arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false),
    /// #     arrow_schema::Field::new("b", arrow_schema::DataType::Utf8, true),
    /// # ]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let df = futures::executor::block_on(s.sql("SELECT a, b FROM t"))?;
    /// let schema = futures::executor::block_on(df.schema())?;
    /// assert_eq!(schema.fields().len(), 2);
    /// assert!(!schema.field(0).is_nullable());
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Spark's analysis errors.
    pub async fn schema(&self) -> Result<SchemaRef, KrishivError> {
        let df = self.clone();
        self.session
            .on_executor(async move { df.schema_here().await })
            .await
    }

    async fn schema_here(&self) -> Result<SchemaRef, KrishivError> {
        let a = self.analysis().await?;
        let fields: Vec<_> = a
            .analyzed
            .output
            .iter()
            .map(|c| c.data_type.to_arrow_field(c.name.clone(), c.nullable))
            .collect();
        Ok(Arc::new(Schema::new(fields)))
    }

    /// Runs the query; its result rows, in batches. Dropping the stream
    /// cancels the query.
    ///
    /// ```
    /// use futures::StreamExt;
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # let batch = arrow_array::RecordBatch::try_new(schema.clone(), vec![std::sync::Arc::new(arrow_array::Int32Array::from(vec![1, 2, 3]))]).unwrap();
    /// # s.register_batches("t", schema, vec![batch])?;
    /// let mut stream = futures::executor::block_on(s.table("t").execute_stream())?;
    /// let first = futures::executor::block_on(stream.next()).unwrap()?;
    /// assert_eq!(first.num_rows(), 3);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Analysis and planning errors; execution errors arrive on the
    /// stream.
    pub async fn execute_stream(&self) -> Result<RecordBatchStream, KrishivError> {
        let df = self.clone();
        self.session
            .on_executor(async move { df.execute_stream_here().await })
            .await
    }

    async fn execute_stream_here(&self) -> Result<RecordBatchStream, KrishivError> {
        let a = self.analysis().await?;
        let (_, physical) = self.session.physical(&a.analyzed, &a.env).await?;
        self.session.run(&physical, &a.env).await
    }

    /// Runs the query and returns its rows. Dropping the future cancels
    /// the query.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # let batch = arrow_array::RecordBatch::try_new(schema.clone(), vec![std::sync::Arc::new(arrow_array::Int32Array::from(vec![3, 1, 2]))]).unwrap();
    /// # s.register_batches("t", schema, vec![batch])?;
    /// let rows = futures::executor::block_on(s.table("t").order_by(["a"]).limit(2).collect())?;
    /// assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Analysis, planning and execution errors.
    pub async fn collect(&self) -> Result<Vec<RecordBatch>, KrishivError> {
        let mut stream = self.execute_stream().await?;
        let mut out = Vec::new();
        while let Some(batch) = stream.next().await {
            out.push(batch?);
        }
        Ok(out)
    }

    /// Runs the query and returns its rows with the executor's per-operator
    /// metrics (`EXPLAIN ANALYZE`'s data): rows, batches and time of every
    /// source, operator and sink, summed over the query's drivers. The
    /// metrics are plain data owned by the caller; nothing is registered
    /// globally. Dropping the future cancels the query.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # let batch = arrow_array::RecordBatch::try_new(schema.clone(), vec![std::sync::Arc::new(arrow_array::Int32Array::from(vec![3, 1, 2]))]).unwrap();
    /// # s.register_batches("t", schema, vec![batch])?;
    /// let (rows, metrics) = futures::executor::block_on(s.table("t").collect_with_metrics())?;
    /// assert_eq!(rows.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
    /// assert!(!metrics.operators.is_empty());
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// As [`Self::collect`].
    pub async fn collect_with_metrics(
        &self,
    ) -> Result<(Vec<RecordBatch>, krishiv_exec::QueryMetrics), KrishivError> {
        let df = self.clone();
        self.session
            .on_executor(async move {
                let a = df.analysis().await?;
                let (_, physical) = df.session.physical(&a.analyzed, &a.env).await?;
                let (mut stream, query) = df.session.run_with_query(&physical, &a.env).await?;
                let mut out = Vec::new();
                while let Some(batch) = stream.next().await {
                    out.push(batch?);
                }
                drop(stream);
                Ok((out, query.metrics()))
            })
            .await
    }

    /// The number of result rows.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # let batch = arrow_array::RecordBatch::try_new(schema.clone(), vec![std::sync::Arc::new(arrow_array::Int32Array::from(vec![3, 1, 2]))]).unwrap();
    /// # s.register_batches("t", schema, vec![batch])?;
    /// assert_eq!(futures::executor::block_on(s.table("t").count())?, 3);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// As [`Self::collect`].
    pub async fn count(&self) -> Result<u64, KrishivError> {
        let mut stream = self.execute_stream().await?;
        let mut rows = 0_u64;
        while let Some(batch) = stream.next().await {
            rows += batch?.num_rows() as u64;
        }
        Ok(rows)
    }

    /// The warnings analysing and planning the query raise: hints Spark
    /// ignores (an unknown hint, a hint naming no relation, a join hint
    /// the join cannot follow), analysis first. Planning logs them too,
    /// at `WARN` (target `krishiv::session`), as Spark logs them.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let df = futures::executor::block_on(s.sql("SELECT /*+ NOPE */ a FROM t"))?;
    /// let warnings = futures::executor::block_on(df.warnings())?;
    /// assert!(warnings[0].starts_with("Unrecognized hint: NOPE"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Analysis and planning errors.
    pub async fn warnings(&self) -> Result<Vec<String>, KrishivError> {
        let df = self.clone();
        self.session
            .on_executor(async move {
                let a = df.analysis().await?;
                let (_, physical) = df.session.physical(&a.analyzed, &a.env).await?;
                let mut all = a.analyzed.warnings.clone();
                all.extend(physical.warnings().iter().cloned());
                Ok(all)
            })
            .await
    }

    /// The query's plans as text (PySpark's `explain(mode)`).
    ///
    /// ```
    /// use krishiv_session::ExplainMode;
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let text = futures::executor::block_on(s.table("t").limit(1).explain(ExplainMode::Extended))?;
    /// assert!(text.starts_with("== Parsed Logical Plan =="));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Analysis and planning errors.
    pub async fn explain(&self, mode: ExplainMode) -> Result<String, KrishivError> {
        let df = self.clone();
        self.session
            .on_executor(async move { df.explain_text(mode, false).await })
            .await
    }

    /// Writes this DataFrame's rows to files (PySpark's `df.write`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let out = dir.path().join("out");
    /// # let out = out.to_str().unwrap();
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # let batch = arrow_array::RecordBatch::try_new(schema.clone(), vec![std::sync::Arc::new(arrow_array::Int32Array::from(vec![1, 2]))]).unwrap();
    /// # s.register_batches("t", schema, vec![batch])?;
    /// futures::executor::block_on(s.table("t").write().mode("overwrite").parquet(out))?;
    /// assert_eq!(futures::executor::block_on(s.read().parquet([out]).count())?, 2);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn write(&self) -> DataFrameWriter {
        DataFrameWriter {
            df: self.clone(),
            format: "parquet".to_owned(),
            mode: "errorifexists".to_owned(),
            options: BTreeMap::new(),
        }
    }

    /// Registers this DataFrame as the temporary view `name`, replacing any
    /// table or view of that name (PySpark's `createOrReplaceTempView`).
    /// A view over a read by path lists its files again at every query;
    /// any other view inlines its plan as a subquery wherever it is read.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let e = futures::executor::block_on(s.table("t").create_or_replace_temp_view("v")).unwrap_err();
    /// assert_eq!(e.error_class().name(), "TABLE_OR_VIEW_NOT_FOUND");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Analysis errors, and `KRISHIV_RESERVED_TEMP_VIEW_NAME` when `name`
    /// starts with the reserved snapshot prefix `` `__krishiv_snapshot_` ``
    /// (in any letter case), which [`Session::sql`] cannot spell either.
    pub async fn create_or_replace_temp_view(&self, name: &str) -> Result<(), KrishivError> {
        let (df, name) = (self.clone(), name.to_owned());
        self.session
            .on_executor(async move { df.temp_view_here(&name).await })
            .await
    }

    async fn temp_view_here(&self, name: &str) -> Result<(), KrishivError> {
        if crate::catalog::is_hidden_name(name) {
            return Err(crate::catalog::reserved_temp_view_name(name));
        }
        // Spark analyzes the view's plan first: its errors come first.
        let analysis = self.analysis().await?;
        let columns: Vec<TableColumn> = analysis
            .analyzed
            .output
            .iter()
            .map(|c| TableColumn {
                name: c.name.clone(),
                data_type: c.data_type.clone(),
                nullable: c.nullable,
            })
            .collect();
        // A view over files keeps listing them: files added or changed
        // later show up in the next query.
        if let PlanNode::Read(source @ ReadSource::Files { .. }) = &self.plan.node {
            self.session.state.catalog.put(
                name,
                TempTable::Files {
                    source: source.clone(),
                    columns,
                },
            );
            return Ok(());
        }
        // Any other view snapshots its input: the analyzed plan is stored,
        // with the temporary tables it reads pinned under hidden names, so
        // a view built from itself reads the old definition, as in Spark.
        // (SQL `CREATE VIEW` keeps the unresolved query instead, so it
        // follows the tables it reads.)
        let (expanded, pins, new_pins) = crate::catalog::pin_temp_tables(
            &self.session.state.catalog,
            analysis.analyzed.plan.root(),
            analysis.env.analysis.conf.case_sensitive,
        )?;
        self.session
            .state
            .catalog
            .put_view(name, expanded, columns, pins, new_pins);
        Ok(())
    }
}

/// Rows grouped by keys ([`DataFrame::group_by`]).
#[derive(Debug, Clone)]
pub struct GroupedData {
    df: DataFrame,
    keys: Vec<Expr>,
}

impl GroupedData {
    /// One row per group: the keys, then `aggregates` (PySpark's
    /// `groupBy(...).agg(...)`).
    ///
    /// ```
    /// # use krishiv_session::functions::{col, count_star};
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").group_by(["k"]).agg([count_star().alias("n")]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn agg(&self, aggregates: impl IntoIterator<Item = Column>) -> DataFrame {
        let keys: Vec<Arc<Expr>> = self.keys.iter().cloned().map(Arc::new).collect();
        let mut output = keys.clone();
        output.extend(aggregates.into_iter().map(|c| Arc::new(c.expr)));
        self.df.derive(PlanNode::Aggregate {
            input: Arc::clone(&self.df.plan),
            grouping: Grouping::GroupBy(keys),
            aggregates: output,
        })
    }
}

/// Reads files ([`Session::read`]; PySpark's `DataFrameReader`).
#[derive(Debug, Clone)]
#[must_use]
pub struct DataFrameReader {
    session: Session,
    format: String,
    options: BTreeMap<String, OptionValue>,
}

impl DataFrameReader {
    pub(crate) fn new(session: Session) -> Self {
        Self {
            session,
            format: "parquet".to_owned(),
            options: BTreeMap::new(),
        }
    }

    /// The file format (default `parquet`, as Spark's
    /// `spark.sql.sources.default`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.read().format("parquet").load(["/data/t"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn format(mut self, format: &str) -> Self {
        self.format = format.to_owned();
        self
    }

    /// A read option; values are typed as secrets where Spark redacts them.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.read().option("mergeSchema", "false").parquet(["/data/t"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn option(mut self, key: &str, value: &str) -> Self {
        self.options
            .insert(key.to_owned(), OptionValue::from_user(key, value));
        self
    }

    /// Reads `paths` in the format.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.read().load(["/data/t", "/data/u"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn load<S: AsRef<str>>(self, paths: impl IntoIterator<Item = S>) -> DataFrame {
        let plan = read_files(&self.format, typed_paths(paths), self.options);
        DataFrame::new(self.session, Arc::new(plan))
    }

    /// Reads Parquet files (PySpark's `read.parquet`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.read().parquet(["/data/t"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn parquet<S: AsRef<str>>(self, paths: impl IntoIterator<Item = S>) -> DataFrame {
        self.format("parquet").load(paths)
    }
}

/// Writes a DataFrame's rows to files ([`DataFrame::write`]; PySpark's
/// `DataFrameWriter`).
#[derive(Debug, Clone)]
#[must_use]
pub struct DataFrameWriter {
    df: DataFrame,
    format: String,
    mode: String,
    options: BTreeMap<String, OptionValue>,
}

impl DataFrameWriter {
    /// What to do when the path exists (Spark's save modes, ASCII
    /// case-insensitive): `errorifexists` (also `error`, `default`; the
    /// default) fails with `PATH_ALREADY_EXISTS`, `ignore` writes nothing,
    /// `append` adds files, `overwrite` replaces what is there once the
    /// new files are complete. Another name fails the write with
    /// `INVALID_SAVE_MODE`.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write().mode("append");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn mode(mut self, mode: &str) -> Self {
        mode.clone_into(&mut self.mode);
        self
    }

    /// The file format (default `parquet`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write().format("parquet");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn format(mut self, format: &str) -> Self {
        format.clone_into(&mut self.format);
        self
    }

    /// A write option (`compression`, ...); values are typed as secrets
    /// where Spark redacts them.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write().option("compression", "zstd");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn option(mut self, key: &str, value: &str) -> Self {
        self.options
            .insert(key.to_owned(), OptionValue::from_user(key, value));
        self
    }

    /// Writes the rows to the directory `path` in the format. Dropping
    /// the future before the write commits cancels it and deletes its
    /// files.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let out = dir.path().join("t");
    /// futures::executor::block_on(s.table("t").write().format("parquet").save(out.to_str().unwrap()))?;
    /// assert!(out.join("_SUCCESS").exists());
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Analysis and planning errors; `INVALID_SAVE_MODE`;
    /// `PATH_ALREADY_EXISTS` (`errorifexists` on an existing path, or an
    /// append to a file); the format's option errors; write failures.
    pub async fn save(self, path: &str) -> Result<(), KrishivError> {
        let target = WriteTarget {
            format: self.format,
            mode: self.mode,
            path: OptionValue::from_user("path", path),
            options: self.options,
        };
        let df = self.df;
        let session = df.session.clone();
        session
            .on_executor(async move {
                let a = df.analysis().await?;
                let (_, physical) = df.session.physical(&a.analyzed, &a.env).await?;
                df.session.write_files(physical, &a.env, target).await
            })
            .await
    }

    /// Writes the rows as Parquet files under `path` (PySpark's
    /// `write.parquet`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let out = dir.path().join("t");
    /// futures::executor::block_on(s.table("t").write().parquet(out.to_str().unwrap()))?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// As [`Self::save`].
    pub async fn parquet(self, path: &str) -> Result<(), KrishivError> {
        self.format("parquet").save(path).await
    }
}

/// The expressions of `cols`.
fn exprs(cols: impl IntoIterator<Item = impl Into<Column>>) -> Vec<Arc<Expr>> {
    cols.into_iter().map(|c| Arc::new(c.into().expr)).collect()
}
