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
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};

use arrow_array::RecordBatch;
use arrow_schema::{Schema, SchemaRef};
use futures::StreamExt;
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_common::secret::OptionValue;
use krishiv_plan::analyzer::{AnalyzedPlan, TableColumn};
use krishiv_plan::cache::CacheKey;
use krishiv_plan::{
    Expr, Grouping, JoinCondition, JoinKind, Literal, LogicalPlan, Origin, PlanId, PlanNode,
    ReadSource, SetOperation, SourceId,
};

use crate::cache::RelationCache;
use crate::catalog::TempTable;
use crate::column::Column;
use crate::session::{QueryEnv, RecordBatchStream, Session, WriteTarget, read_files, typed_paths};

/// A storage level for [`DataFrame::persist`], as PySpark's
/// `StorageLevel`. Every level maps to memory: the relation cache holds
/// Arrow batches under the `relation_cache` pool, so a level asking for
/// disk behaves like its memory-only counterpart for now (disk levels
/// spill there later).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageLevel {
    /// Keep entries on disk (accepted; stored in memory for now).
    pub use_disk: bool,
    /// Keep entries in memory.
    pub use_memory: bool,
    /// Keep entries off-heap (accepted; stored in memory for now).
    pub use_off_heap: bool,
    /// Keep entries deserialized.
    pub deserialized: bool,
    /// How many copies to keep (accepted; one copy is kept for now).
    pub replication: i32,
}

impl StorageLevel {
    /// `MEMORY_ONLY`: one in-memory copy.
    pub const MEMORY_ONLY: Self = Self {
        use_disk: false,
        use_memory: true,
        use_off_heap: false,
        deserialized: true,
        replication: 1,
    };
    /// `MEMORY_ONLY_2`: accepted as `MEMORY_ONLY`.
    pub const MEMORY_ONLY_2: Self = Self {
        replication: 2,
        ..Self::MEMORY_ONLY
    };
    /// `MEMORY_AND_DISK`: one in-memory copy (`cache`'s level).
    pub const MEMORY_AND_DISK: Self = Self {
        use_disk: true,
        ..Self::MEMORY_ONLY
    };
    /// `MEMORY_AND_DISK_2`: accepted as `MEMORY_AND_DISK`.
    pub const MEMORY_AND_DISK_2: Self = Self {
        use_disk: true,
        ..Self::MEMORY_ONLY_2
    };
    /// `DISK_ONLY`: accepted as `MEMORY_ONLY`.
    pub const DISK_ONLY: Self = Self {
        use_disk: true,
        use_memory: false,
        deserialized: false,
        ..Self::MEMORY_ONLY
    };
    /// `DISK_ONLY_2`: accepted as `MEMORY_ONLY`.
    pub const DISK_ONLY_2: Self = Self {
        replication: 2,
        ..Self::DISK_ONLY
    };
    /// `OFF_HEAP`: accepted as `MEMORY_ONLY`.
    pub const OFF_HEAP: Self = Self {
        use_off_heap: true,
        ..Self::MEMORY_ONLY
    };
}

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

/// What an action stores into the relation cache when its stream completes.
pub(crate) enum CacheFill {
    /// This frame asked for no cache, or the whole plan already hit.
    No,
    /// Materialize the stream's batches under `key` when it completes.
    Fill {
        key: CacheKey,
        fallback: SchemaRef,
        reads_files: bool,
        tables: Vec<String>,
        /// The cache's invalidation generation when the fill started.
        generation: u64,
    },
}

impl CacheFill {
    /// `stream`, filling `cache` when it ends cleanly (see [`FillStream`]).
    fn fill(self, stream: RecordBatchStream, cache: &Arc<RelationCache>) -> RecordBatchStream {
        match self {
            Self::No => stream,
            Self::Fill {
                key,
                fallback,
                reads_files,
                tables,
                generation,
            } => Box::pin(FillStream {
                inner: stream,
                cache: Arc::clone(cache),
                key,
                schema: None,
                fallback,
                reads_files,
                tables,
                batches: Vec::new(),
                bytes: 0,
                cap: cache.cap(),
                generation,
                filled: false,
            }),
        }
    }
}

/// A result stream that keeps its batches for the relation cache. When the
/// inner stream ends, the batches are stored under the fill's key; a cap
/// overrun stores nothing, so the query stays uncached instead of failing.
/// An error, a cancellation (dropping before the end) or a mid-stream schema
/// change stores nothing either: a cached relation is one complete snapshot,
/// never a prefix of one. The empty result stores an empty snapshot, so
/// later actions read the snapshot, not the source. The buffer is bounded by
/// the cache cap: once the batches kept so far exceed it, they are dropped
/// and the rest of the stream passes through unbuffered.
struct FillStream {
    inner: RecordBatchStream,
    cache: Arc<RelationCache>,
    key: CacheKey,
    /// The first batch's schema: what the snapshot is stored as. Operators
    /// may widen the analysis types (large string views, ...), so the cached
    /// table carries the executed schema, as the scan of a created
    /// DataFrame carries the registered one. The analysis schema is only
    /// the fallback for an empty result.
    schema: Option<SchemaRef>,
    fallback: SchemaRef,
    reads_files: bool,
    tables: Vec<String>,
    batches: Vec<RecordBatch>,
    /// Bytes of `batches`.
    bytes: usize,
    /// The cache cap when the fill started.
    cap: usize,
    generation: u64,
    filled: bool,
}

impl futures::Stream for FillStream {
    type Item = Result<RecordBatch, KrishivError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                if self.filled {
                    // Abandoned: over the cap or a changed shape.
                    return Poll::Ready(Some(Ok(batch)));
                }
                if self.schema.is_none() {
                    self.schema = Some(Arc::clone(batch.schema_ref()));
                } else if self
                    .schema
                    .as_ref()
                    .is_some_and(|s| s.as_ref() != batch.schema().as_ref())
                {
                    // A stream that changes shape mid-way is not one snapshot.
                    self.batches.clear();
                    self.schema = None;
                    self.filled = true;
                    return Poll::Ready(Some(Ok(batch)));
                }
                self.bytes = self.bytes.saturating_add(crate::cache::batch_bytes(&batch));
                if self.bytes > self.cap {
                    // The snapshot cannot be stored: stop holding it.
                    self.batches = Vec::new();
                    self.schema = None;
                    self.filled = true;
                    return Poll::Ready(Some(Ok(batch)));
                }
                self.batches.push(batch.clone());
                Poll::Ready(Some(Ok(batch)))
            }
            Poll::Ready(None) => {
                if !self.filled {
                    self.filled = true;
                    // The empty result still caches (as an empty snapshot),
                    // so later actions read the snapshot, not the source.
                    let this = &mut *self;
                    let batches = std::mem::take(&mut this.batches);
                    let schema = this
                        .schema
                        .take()
                        .unwrap_or_else(|| Arc::clone(&this.fallback));
                    let cache = Arc::clone(&this.cache);
                    cache.insert(
                        this.key.clone(),
                        schema,
                        batches,
                        this.reads_files,
                        std::mem::take(&mut this.tables),
                        this.generation,
                    );
                }
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(e))) => {
                self.batches.clear();
                self.filled = true;
                Poll::Ready(Some(Err(e)))
            }
            Poll::Pending => Poll::Pending,
        }
    }
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
    /// The storage level a `cache` or `persist` asked for: the next action
    /// materializes this plan into the relation cache. Derived frames do
    /// not inherit it, but their actions still read matching entries.
    cache_request: Option<StorageLevel>,
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
            cache_request: None,
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
            cache_request: None,
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

    /// `SELECT` of SQL expression strings (PySpark's `selectExpr`): each
    /// string holds one expression with an optional alias
    /// (`expr [[AS] alias]`), so `*` and aliases work as they do in a
    /// `SELECT` list. The strings parse with this session's parser options.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").select_expr(["a", "b AS second"]).unwrap();
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Spark's parse errors (`PARSE_SYNTAX_ERROR`, ...); lowering errors
    /// for constructs the IR lacks.
    pub fn select_expr(
        &self,
        exprs: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<Self, KrishivError> {
        let options = self.session.parser_options();
        let mut lowered = Vec::new();
        for text in exprs {
            lowered.push(
                krishiv_sql::parse_and_lower_named_expression(text.as_ref(), &options, SourceId(0))
                    .map_err(|e| e.error)?,
            );
        }
        Ok(self.derive(PlanNode::Project {
            input: Arc::clone(&self.plan),
            exprs: lowered,
        }))
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

    /// Rows where the SQL expression `condition` is true (PySpark's
    /// `filter(str)` and `where(str)`). The string parses with this
    /// session's parser options.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").filter_expr("a IS NOT NULL").unwrap();
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Spark's parse errors (`PARSE_SYNTAX_ERROR`, ...); lowering errors
    /// for constructs the IR lacks.
    pub fn filter_expr(&self, condition: &str) -> Result<Self, KrishivError> {
        let predicate = krishiv_sql::parse_and_lower_named_expression(
            condition,
            &self.session.parser_options(),
            SourceId(0),
        )
        .map_err(|e| e.error)?;
        Ok(self.derive(PlanNode::Filter {
            input: Arc::clone(&self.plan),
            predicate,
        }))
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
        self.grouped(keys, Sets::GroupBy)
    }

    /// Groups rows by `keys` and by each prefix of them, down to the grand
    /// total, for [`GroupedData::agg`] (PySpark's `rollup`): a key a row is
    /// rolled up over reads `NULL`.
    ///
    /// ```
    /// # use krishiv_session::functions::{col, sum};
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").rollup([col("a"), col("b")]).agg([sum(col("v"))]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn rollup(&self, keys: impl IntoIterator<Item = impl Into<Column>>) -> GroupedData {
        self.grouped(keys, Sets::Rollup)
    }

    /// Groups rows by every subset of `keys`, for [`GroupedData::agg`]
    /// (PySpark's `cube`): a key a row is rolled up over reads `NULL`.
    ///
    /// ```
    /// # use krishiv_session::functions::{col, sum};
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").cube([col("a"), col("b")]).agg([sum(col("v"))]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn cube(&self, keys: impl IntoIterator<Item = impl Into<Column>>) -> GroupedData {
        self.grouped(keys, Sets::Cube)
    }

    /// Groups rows by each of `sets` (grouping expressions among `keys`),
    /// for [`GroupedData::agg`] (PySpark's `groupingSets`): the output has
    /// the columns of `keys`, `NULL` where a set does not group by them.
    ///
    /// ```
    /// # use krishiv_session::functions::{col, sum};
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s
    ///     .table("t")
    ///     .grouping_sets([vec![col("a")], vec![col("b")], vec![]], [col("a"), col("b")])
    ///     .agg([sum(col("v"))]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn grouping_sets(
        &self,
        sets: impl IntoIterator<Item = impl IntoIterator<Item = Column>>,
        keys: impl IntoIterator<Item = impl Into<Column>>,
    ) -> GroupedData {
        let sets = sets
            .into_iter()
            .map(|set| set.into_iter().map(|c| c.expr).collect())
            .collect();
        self.grouped(keys, Sets::Explicit(sets))
    }

    fn grouped(
        &self,
        keys: impl IntoIterator<Item = impl Into<Column>>,
        sets: Sets,
    ) -> GroupedData {
        GroupedData {
            df: self.clone(),
            keys: keys.into_iter().map(|k| k.into().expr).collect(),
            sets,
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

    /// A new DataFrame of `node` over this frame and `other` (as
    /// [`Self::join_with`]): either frame's deferred error fails the new
    /// frame's first action.
    fn pair(&self, other: &Self, node: PlanNode) -> Self {
        let mut out = self.derive(node);
        if out.invalid.is_none() {
            out.invalid.clone_from(&other.invalid);
        }
        out
    }

    /// [`Self::intersect`], [`Self::intersect_all`],
    /// [`Self::subtract`] and [`Self::except_all`].
    fn set_op(&self, other: &Self, op: SetOperation, all: bool) -> Self {
        self.pair(
            other,
            PlanNode::SetOp {
                left: Arc::clone(&self.plan),
                right: Arc::clone(&other.plan),
                op,
                all,
            },
        )
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

    /// The distinct rows (PySpark's `distinct`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").distinct();
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn distinct(&self) -> Self {
        self.derive(PlanNode::Distinct {
            input: Arc::clone(&self.plan),
        })
    }

    /// Every row of this DataFrame and `other`, columns matched by
    /// position (PySpark's `union` and `unionAll`; SQL's `UNION ALL`).
    /// Column types widen per position; a different column count fails
    /// analysis.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("l").union(&s.table("r"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        self.pair(
            other,
            PlanNode::Union {
                inputs: vec![Arc::clone(&self.plan), Arc::clone(&other.plan)],
                by_name: false,
                allow_missing_columns: false,
            },
        )
    }

    /// Every row of this DataFrame and `other`, columns matched by name
    /// (PySpark's `unionByName`): `other`'s columns are reordered to this
    /// frame's, and with `allow_missing_columns` a column one side lacks
    /// reads NULL there.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("l").union_by_name(&s.table("r"), true);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn union_by_name(&self, other: &Self, allow_missing_columns: bool) -> Self {
        self.pair(
            other,
            PlanNode::Union {
                inputs: vec![Arc::clone(&self.plan), Arc::clone(&other.plan)],
                by_name: true,
                allow_missing_columns,
            },
        )
    }

    /// The rows of this DataFrame the `other` one has too, without
    /// duplicates (PySpark's `intersect`; SQL's `INTERSECT`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("l").intersect(&s.table("r"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self {
        self.set_op(other, SetOperation::Intersect, false)
    }

    /// The rows of this DataFrame the `other` one has too, duplicates kept
    /// (PySpark's `intersectAll`; SQL's `INTERSECT ALL`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("l").intersect_all(&s.table("r"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn intersect_all(&self, other: &Self) -> Self {
        self.set_op(other, SetOperation::Intersect, true)
    }

    /// The rows of this DataFrame the `other` one lacks, without
    /// duplicates (PySpark's `subtract`; SQL's `EXCEPT`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("l").subtract(&s.table("r"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn subtract(&self, other: &Self) -> Self {
        self.set_op(other, SetOperation::Except, false)
    }

    /// The rows of this DataFrame the `other` one lacks, duplicates kept
    /// (PySpark's `exceptAll`; SQL's `EXCEPT ALL`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("l").except_all(&s.table("r"));
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn except_all(&self, other: &Self) -> Self {
        self.set_op(other, SetOperation::Except, true)
    }

    /// This DataFrame with the hint `name` and `params` (PySpark's
    /// `hint`): a join strategy hint wraps the relations `params` name
    /// (columns and string literals name relations); an unknown hint and
    /// a hint naming no relation come back as warnings, and a
    /// partitioning hint (`COALESCE`, `REPARTITION`, ...) is checked and
    /// otherwise dropped: a single-node plan has no partitions to set.
    /// The hint never changes the rows.
    ///
    /// ```
    /// # use krishiv_session::functions::lit;
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").hint("BROADCAST", [lit("t")]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn hint(&self, name: &str, params: impl IntoIterator<Item = impl Into<Column>>) -> Self {
        self.derive(PlanNode::Hint {
            input: Arc::clone(&self.plan),
            name: name.to_owned(),
            params: params
                .into_iter()
                .map(|c| Arc::new(c.into().expr))
                .collect(),
        })
    }

    /// This DataFrame with its columns renamed to `names`, in order
    /// (PySpark's `toDF`; SQL's `AS (columns)`). A different count fails
    /// analysis.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").to_df(["x", "y"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn to_df(&self, names: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        self.derive(PlanNode::SubqueryAlias {
            input: Arc::clone(&self.plan),
            alias: String::new(),
            columns: names.into_iter().map(|n| n.as_ref().to_owned()).collect(),
        })
    }

    /// A sample of the rows (PySpark's `sample`). The engine cannot
    /// sample yet: the returned frame fails its first action with
    /// `KRISHIV_UNSUPPORTED_FEATURE`, as do the schema and the plans.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").sample(false, 0.5, 3);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE`: sampling is not supported yet.
    #[must_use]
    pub fn sample(&self, _with_replacement: bool, _fraction: f64, _seed: i64) -> Self {
        Self::invalid(
            self.session.clone(),
            Arc::clone(&self.plan),
            self.invalid.clone().unwrap_or_else(|| {
                KrishivUnsupportedFeature {
                    feature: "DataFrame.sample".to_owned(),
                }
                .into()
            }),
        )
    }

    /// The wide frame unpivoted to long rows (PySpark's `unpivot`):
    /// one row per value column per input row, with the identifier
    /// columns `ids`, the value's label in `variable` and the value in
    /// `value`. `values` holds `(label, column)` pairs: the label is the
    /// variable a value column unpivots under (Spark uses the column
    /// name, or the given alias). Value columns widen to a common type,
    /// as in `union`; NULL values are kept, as in Spark's DataFrame API
    /// (SQL's `UNPIVOT` drops them unless told otherwise). An empty
    /// `values` fails the first action with
    /// `KRISHIV_UNSUPPORTED_FEATURE`, as Spark leaves the plan
    /// unresolved.
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// # let s = krishiv_session::Session::new()?;
    /// let df = s.table("t").unpivot(
    ///     [col("id")],
    ///     [("int".to_owned(), col("int")), ("double".to_owned(), col("double"))],
    ///     "var",
    ///     "val",
    /// );
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE`: with no value columns.
    #[must_use]
    pub fn unpivot(
        &self,
        ids: impl IntoIterator<Item = Column>,
        values: impl IntoIterator<Item = (String, Column)>,
        variable: &str,
        value: &str,
    ) -> Self {
        let ids: Vec<Column> = ids.into_iter().collect();
        let mut values = values.into_iter();
        let Some((label, column)) = values.next() else {
            return Self::invalid(
                self.session.clone(),
                Arc::clone(&self.plan),
                self.invalid.clone().unwrap_or_else(|| {
                    KrishivUnsupportedFeature {
                        feature: "UNPIVOT without value columns".to_owned(),
                    }
                    .into()
                }),
            );
        };
        // One union over every branch: the common type is computed once
        // and each branch is cast to it directly.
        let first = self.branch(&ids, label, column, variable, value);
        let rest: Vec<Self> = values
            .map(|(label, column)| self.branch(&ids, label, column, variable, value))
            .collect();
        if rest.is_empty() {
            return first;
        }
        let inputs = std::iter::once(&first)
            .chain(&rest)
            .map(|b| Arc::clone(&b.plan))
            .collect();
        self.derive(PlanNode::Union {
            inputs,
            by_name: false,
            allow_missing_columns: false,
        })
    }

    /// One unpivot branch: the identifiers, the `label` in `variable`
    /// and the `column` in `value`.
    fn branch(
        &self,
        ids: &[Column],
        label: String,
        column: Column,
        variable: &str,
        value: &str,
    ) -> Self {
        let mut exprs = ids.to_vec();
        exprs.push(crate::functions::lit(label).alias(variable));
        exprs.push(column.alias(value));
        self.select(exprs)
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
            && a.env.variables_version == env.variables_version
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
        let (physical, fill) = self.physical_cached(&a).await?;
        let stream = self.session.run(&physical, &a.env).await?;
        Ok(fill.fill(stream, &self.session.state.cache))
    }

    /// The physical plan with every cached (sub)plan substituted, and what
    /// to fill into the cache when the stream completes: the key and the
    /// snapshot shape, or `None` when this frame asked for no cache or the
    /// whole plan already hit.
    pub(crate) async fn physical_cached(
        &self,
        a: &Arc<Analysis>,
    ) -> Result<(krishiv_exec::planner::PhysicalPlan, CacheFill), KrishivError> {
        let key = CacheKey::of(&a.analyzed);
        let generation = self.session.state.cache.generation();
        let hit = self.session.state.cache.lookup(&key).is_some();
        let substituted = self.session.state.cache.substitute(&a.analyzed);
        let owned;
        let analyzed = match substituted {
            Some(sub) => {
                // The query keeps its snapshots: clearing the cache before
                // it is planned must not fail it.
                for (name, table) in sub.tables {
                    a.env.catalog.pin_cached(name, table);
                }
                owned = AnalyzedPlan {
                    plan: krishiv_plan::Plan::try_new(sub.plan)?,
                    ..(*a.analyzed).clone()
                };
                &owned
            }
            None => &a.analyzed,
        };
        let (_, physical) = self.session.physical(analyzed, &a.env).await?;
        let fill = if hit || self.cache_request.is_none() {
            CacheFill::No
        } else {
            let fields: Vec<_> = a
                .analyzed
                .output
                .iter()
                .map(|c| c.data_type.to_arrow_field(c.name.clone(), c.nullable))
                .collect();
            CacheFill::Fill {
                key,
                fallback: Arc::new(Schema::new(fields)),
                reads_files: a.analyzed.reads_files,
                generation,
                tables: {
                    let mut tables = crate::cache::read_tables(a.analyzed.plan.root());
                    tables.extend(crate::cache::unresolved_tables(&self.plan));
                    tables.sort();
                    tables.dedup();
                    tables
                },
            }
        };
        Ok((physical, fill))
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
                let (physical, fill) = df.physical_cached(&a).await?;
                let (stream, query) = df.session.run_with_query(&physical, &a.env).await?;
                let mut stream = fill.fill(stream, &df.session.state.cache);
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

    /// Caches this frame's rows in memory (PySpark's `cache`, Spark's
    /// `MEMORY_AND_DISK` default): the next action materializes the plan
    /// into the session's relation cache, and later actions of this or any
    /// frame with the same result read the snapshot instead of recomputing
    /// it. Materialization is lazy and all-or-nothing; when the snapshot
    /// would exceed `krishiv.cache.maxBytes` the query runs uncached.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let df = s.table("t").cache();
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn cache(&self) -> Self {
        self.persist(StorageLevel::MEMORY_AND_DISK)
    }

    /// Caches this frame's rows at `level` (PySpark's `persist`): every
    /// level maps to memory for now (see [`StorageLevel`]).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let df = s.table("t").persist(krishiv_session::StorageLevel::DISK_ONLY);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn persist(&self, level: StorageLevel) -> Self {
        Self {
            cache_request: Some(level),
            ..self.clone()
        }
    }

    /// Drops this frame's cached snapshot, if any (PySpark's `unpersist`),
    /// returning the pool memory to baseline. Frames derived from this one
    /// are unaffected: they read the cache only while an entry matches.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let df = futures::executor::block_on(s.table("t").cache().unpersist())?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Spark's analysis errors.
    pub async fn unpersist(&self) -> Result<Self, KrishivError> {
        let df = self.clone();
        self.session
            .on_executor(async move {
                let a = df.analysis().await?;
                df.session.state.cache.remove(&CacheKey::of(&a.analyzed));
                Ok(Self {
                    cache_request: None,
                    ..df
                })
            })
            .await
    }

    /// Whether this frame's plan is cached now: an entry matches it, under
    /// this principal and policy version.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// assert!(!futures::executor::block_on(s.table("t").is_cached())?);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Spark's analysis errors.
    pub async fn is_cached(&self) -> Result<bool, KrishivError> {
        let df = self.clone();
        self.session
            .on_executor(async move {
                let a = df.analysis().await?;
                Ok(df.session.state.cache.contains(&CacheKey::of(&a.analyzed)))
            })
            .await
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
            mode: "error".to_owned(),
            options: BTreeMap::new(),
            partition_by: Vec::new(),
        }
    }

    /// Writes this DataFrame's rows to the directory `path` in `format`
    /// with `mode` (PySpark's `DataFrameWriter.save` collected as one
    /// call): `options` pairs and the `partition_by` columns. The engine
    /// futures run is the same as [`DataFrameWriter::save`]; a partitioned
    /// write fails with `KRISHIV_UNSUPPORTED_FEATURE`.
    ///
    /// # Errors
    /// As [`DataFrameWriter::save`].
    pub async fn write_to(
        &self,
        format: &str,
        mode: &str,
        path: &str,
        options: Vec<(String, String)>,
        partition_by: Vec<String>,
    ) -> Result<(), KrishivError> {
        let mut writer = self.write().format(format).mode(mode);
        for (key, value) in &options {
            writer = writer.option(key, value);
        }
        if !partition_by.is_empty() {
            writer = writer.partition_by(partition_by.iter().map(String::as_str));
        }
        writer.save(path).await
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
    /// or the cache prefix `` `__krishiv_cache_` `` (in any letter case),
    /// which [`Session::sql`] cannot spell either.
    pub async fn create_or_replace_temp_view(&self, name: &str) -> Result<(), KrishivError> {
        let (df, name) = (self.clone(), name.to_owned());
        self.session
            .on_executor(async move { df.temp_view_here(&name).await })
            .await
    }

    async fn temp_view_here(&self, name: &str) -> Result<(), KrishivError> {
        if crate::catalog::is_hidden_name(name) || crate::cache::is_cache_name(name) {
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
            self.session.state.cache.invalidate_table(name);
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
        self.session.state.cache.invalidate_table(name);
        Ok(())
    }
}

/// Which groups of its keys a [`GroupedData`] aggregates.
#[derive(Debug, Clone)]
enum Sets {
    /// All the keys together.
    GroupBy,
    /// Each prefix of the keys.
    Rollup,
    /// Each subset of the keys.
    Cube,
    /// These sets.
    Explicit(Vec<Vec<Expr>>),
}

/// Rows grouped by keys ([`DataFrame::group_by`], [`DataFrame::rollup`],
/// [`DataFrame::cube`], [`DataFrame::grouping_sets`]).
#[derive(Debug, Clone)]
pub struct GroupedData {
    df: DataFrame,
    keys: Vec<Expr>,
    sets: Sets,
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
        let grouping = match &self.sets {
            Sets::GroupBy => Grouping::GroupBy(keys),
            Sets::Rollup => Grouping::Rollup(keys),
            Sets::Cube => Grouping::Cube(keys),
            Sets::Explicit(sets) => Grouping::GroupingSets {
                exprs: keys,
                sets: sets
                    .iter()
                    .map(|set| set.iter().cloned().map(Arc::new).collect())
                    .collect(),
            },
        };
        self.df.derive(PlanNode::Aggregate {
            input: Arc::clone(&self.df.plan),
            grouping,
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
    partition_by: Vec<String>,
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

    /// Several write options at once (PySpark's `options(**options)`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write().options([("compression", "zstd")]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn options(
        mut self,
        options: impl IntoIterator<Item = (impl AsRef<str>, impl AsRef<str>)>,
    ) -> Self {
        for (key, value) in options {
            self.options.insert(
                key.as_ref().to_owned(),
                OptionValue::from_user(key.as_ref(), value.as_ref()),
            );
        }
        self
    }

    /// Partition the output by `cols` (PySpark's `partitionBy`); each call
    /// replaces the columns of the one before, so an empty call clears them.
    /// The engine writes no partitioned layout yet: any column fails the
    /// write with `KRISHIV_UNSUPPORTED_FEATURE`, rather than silently
    /// ignoring it.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write().partition_by(["a"]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn partition_by(mut self, cols: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        self.partition_by = cols.into_iter().map(|c| c.as_ref().to_owned()).collect();
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
    /// Analysis and planning errors; `INVALID_SAVE_MODE` (checked before
    /// partitioning, so a bad mode wins over a partitioned write);
    /// `PATH_ALREADY_EXISTS` (`errorifexists` on an existing path, or an
    /// append to a file); the format's option errors; write failures;
    /// `KRISHIV_UNSUPPORTED_FEATURE` when partitioning was requested.
    pub async fn save(self, path: &str) -> Result<(), KrishivError> {
        let target = WriteTarget {
            format: self.format,
            mode: self.mode,
            path: OptionValue::from_user("path", path),
            options: self.options,
            partition_by: self.partition_by,
        };
        let df = self.df;
        let session = df.session.clone();
        session
            .on_executor(async move {
                let a = df.analysis().await?;
                // A write reads the cache like any other action.
                let (physical, _) = df.physical_cached(&a).await?;
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

    /// Writes the rows as CSV files under `path` (PySpark's `write.csv`).
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let out = dir.path().join("t");
    /// futures::executor::block_on(s.table("t").write().csv(out.to_str().unwrap()))?;
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// As [`Self::save`].
    pub async fn csv(self, path: &str) -> Result<(), KrishivError> {
        self.format("csv").save(path).await
    }

    /// Writes the rows as JSON files under `path` (PySpark's `write.json`).
    /// The engine has no JSON writer yet: the write fails with
    /// `KRISHIV_UNSUPPORTED_FEATURE`.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let dir = tempfile::tempdir().unwrap();
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let out = dir.path().join("t");
    /// let e = futures::executor::block_on(s.table("t").write().json(out.to_str().unwrap())).unwrap_err();
    /// assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// As [`Self::save`].
    pub async fn json(self, path: &str) -> Result<(), KrishivError> {
        self.format("json").save(path).await
    }
}

/// The expressions of `cols`.
fn exprs(cols: impl IntoIterator<Item = impl Into<Column>>) -> Vec<Arc<Expr>> {
    cols.into_iter().map(|c| Arc::new(c.into().expr)).collect()
}
