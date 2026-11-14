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
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_common::secret::OptionValue;
use krishiv_plan::analyzer::{AnalyzedPlan, TableColumn};
use krishiv_plan::{
    Expr, Grouping, JoinCondition, JoinKind, Literal, LogicalPlan, Origin, PlanId, PlanNode,
    ReadSource, SetOperation,
};

use crate::cache::SnapshotLease;
use crate::catalog::TempTable;
use crate::column::Column;
use crate::run_site::DF_SOURCE;
use crate::session::{QueryEnv, RecordBatchStream, Session, WriteTarget, read_files, typed_paths};

pub(crate) mod checkpoint;
mod fill_stream;
mod geo_gate;
mod observe;
pub use observe::ObservedSink;
pub(crate) use observe::{Observers, deliver};
mod traced;

pub(crate) use fill_stream::{CacheFill, FillKind};

/// A storage level for [`DataFrame::persist`], as PySpark's
/// `StorageLevel`. Memory levels keep the snapshot as Arrow batches under the
/// `relation_cache` pool and drop it under memory pressure (the next use
/// refills it); `MEMORY_AND_DISK` moves it to a spill file instead, and so
/// does a snapshot over `krishiv.cache.maxBytes`; `DISK_ONLY` writes it to a
/// spill file when it fills. Off-heap and replication are accepted and
/// reported as asked: one copy is kept, in the engine's memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageLevel {
    /// Keep entries on disk (when they do not fit in memory, or always
    /// without `use_memory`).
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
    /// `NONE`: what an uncached frame reports; persisting at it registers nothing.
    pub const NONE: Self = Self {
        use_disk: false,
        use_memory: false,
        use_off_heap: false,
        deserialized: false,
        replication: 1,
    };
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
    /// `MEMORY_AND_DISK`: in memory, moved to disk under pressure (`cache`'s level).
    pub const MEMORY_AND_DISK: Self = Self {
        use_disk: true,
        ..Self::MEMORY_ONLY
    };
    /// `MEMORY_AND_DISK_2`: accepted as `MEMORY_AND_DISK`.
    pub const MEMORY_AND_DISK_2: Self = Self {
        use_disk: true,
        ..Self::MEMORY_ONLY_2
    };
    /// `DISK_ONLY`: written to a spill file when it fills.
    pub const DISK_ONLY: Self = Self {
        use_disk: true,
        use_memory: false,
        deserialized: false,
        ..Self::MEMORY_ONLY
    };
    /// `DISK_ONLY_2`: accepted as `DISK_ONLY`.
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
    /// What the frame's actions say on their `krishiv.query` span: set for
    /// a frame `Session::sql` returned, absent for one a transformation
    /// built.
    trace: Option<Arc<crate::telemetry::QueryTrace>>,
    /// Owns what the plan reads by a name no SQL spells (a script result's table), as long as it lives.
    pub(crate) keep: Vec<Arc<dyn std::any::Any + Send + Sync>>,
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
            trace: None,
            keep: Vec::new(),
        }
    }

    /// A frame over `plan`, which keeps its plan id, owning `keep`: the frame a registration's
    /// fill runs.
    pub(crate) fn over_plan(
        session: Session,
        plan: Arc<LogicalPlan>,
        keep: Vec<Arc<dyn std::any::Any + Send + Sync>>,
    ) -> Self {
        Self {
            keep,
            ..Self::new(session, plan)
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
            trace: None,
            keep: self.keep.clone(),
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
                krishiv_sql::parse_and_lower_named_expression(text.as_ref(), &options, DF_SOURCE)
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
            DF_SOURCE,
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
        self.pair(
            right,
            PlanNode::Join {
                left: Arc::clone(&self.plan),
                right: Arc::clone(&right.plan),
                kind,
                condition,
            },
        )
    }

    /// A new DataFrame of `node` over this frame and `other` (as
    /// [`Self::join_with`]): either frame's deferred error fails the new
    /// frame's first action.
    fn pair(&self, other: &Self, node: PlanNode) -> Self {
        let mut out = self.derive(node);
        if out.invalid.is_none() {
            out.invalid.clone_from(&other.invalid);
        }
        out.keep.extend(other.keep.iter().cloned());
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

    /// The analysis to run with: the cached one while the configuration, the
    /// temporary tables and the session's functions are unchanged, no cache
    /// invalidation ran since and no files are read (by path or through a file view), else a fresh one,
    /// which lists them again.
    pub(crate) async fn analysis(&self) -> Result<Arc<Analysis>, KrishivError> {
        let cached = self
            .analysis
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let mut env = self.session.query_env();
        // A fresh analysis is still the same statement: keep what
        // `system.queries` shows of it.
        if let Some(a) = &cached {
            env.copy_text_from(&a.env);
            env.statement_class = a.env.statement_class;
            env.row = Arc::clone(&a.env.row);
        }
        if let Some(a) = cached
            && !a.analyzed.reads_files
            && !a.env.catalog.loaded_file_views()
            && a.env.conf_version == env.conf_version
            && a.env.catalog_epoch == env.catalog_epoch
            && a.env.variables_version == env.variables_version
            && a.env.cache_generation == env.cache_generation
            && a.env.functions_version == env.functions_version
            && a.env.policy_version == env.policy_version
        {
            return Ok(a);
        }
        self.analyzed_in(env).await
    }

    /// [`Self::analysis`] for an action that runs the query: a re-analysis
    /// that fails (the catalog or the configuration changed since the frame
    /// analyzed) is a row of `system.queries`, with the statement's text and
    /// class. A first analysis that fails is left to the caller, as before:
    /// inside `sql` its refusal handler records it.
    pub(crate) async fn action_analysis(&self) -> Result<Arc<Analysis>, KrishivError> {
        let error = match self.analysis().await {
            Ok(a) => return Ok(a),
            Err(error) => error,
        };
        let cached = self
            .analysis
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if let Some(a) = cached {
            let id =
                self.session
                    .record_refused(a.env.text.as_deref(), a.env.statement_class, &error);
            // Shared with the statement's marker, so `sql` adds no second row.
            *a.env.row.lock().unwrap_or_else(PoisonError::into_inner) = Some(id);
        }
        Err(error)
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
        let plan = self.plan_with_checkpoints().await?;
        let analyzed = self.session.analyze_in(&plan, &env, text).await?;
        let mut env = env;
        if let Some(t) = text {
            env.set_text(t);
        }
        let through_views = analyzed
            .views
            .iter()
            .filter(|v| v.security == krishiv_plan::analyzer::views::ViewSecurity::Temporary)
            .flat_map(|v| self.session.state.catalog.view_keeps(&v.name))
            .collect();
        env.observers = Arc::new(self.observers_through(through_views));
        let analysis = Arc::new(Analysis {
            env: Arc::new(env),
            analyzed: Arc::new(analyzed),
        });
        *self.analysis.lock().unwrap_or_else(PoisonError::into_inner) = Some(Arc::clone(&analysis));
        Ok(analysis)
    }

    /// Makes the statement `text` of class `class` the one this frame's executions show in
    /// `system.queries`, when its analysis has none yet (a frame a command made).
    pub(crate) fn with_statement(
        &self,
        text: &krishiv_common::redact::RedactedSql,
        class: &'static str,
    ) {
        let mut cached = self.analysis.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(a) = cached.as_ref()
            && a.env.text.is_none()
        {
            let mut env = QueryEnv::clone(&a.env);
            env.set_text(text);
            env.statement_class = Some(class);
            *cached = Some(Arc::new(Analysis {
                env: Arc::new(env),
                analyzed: Arc::clone(&a.analyzed),
            }));
        }
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
        Ok(crate::carrier::client_schema(&Schema::new(fields)))
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
            .on_executor(async move { df.execute_stream_here(true).await })
            .await
    }

    /// The result batches on the engine's own carriers, before the session edge converts them (a calendar
    /// interval as its struct of microseconds, which holds values `Interval(MonthDayNano)` cannot): for
    /// harnesses that render rows as Spark's SQL tests do, without Arrow. Not covered by semver.
    ///
    /// # Errors
    /// As [`Self::collect`].
    #[cfg(feature = "unstable-ir")]
    pub async fn collect_engine(&self) -> Result<Vec<RecordBatch>, KrishivError> {
        let df = self.clone();
        let mut stream = self
            .session
            .on_executor(async move { df.execute_stream_here(false).await })
            .await?;
        let mut out = Vec::new();
        while let Some(batch) = stream.next().await {
            out.push(batch?);
        }
        Ok(out)
    }

    /// The analyzed plan with cached subplans replaced by scans of their
    /// snapshots (the plan that is planned and run), or `None` when no
    /// subplan is cached, and the snapshots it reads, held by the action.
    pub(crate) fn cache_substituted(
        &self,
        a: &Arc<Analysis>,
    ) -> Result<(Option<AnalyzedPlan>, SnapshotLease), KrishivError> {
        let cache = &self.session.state.cache;
        let Some(sub) = cache.substitute(&a.analyzed) else {
            return Ok((None, SnapshotLease::default()));
        };
        // The query keeps its snapshots: clearing the cache before it is
        // planned must not fail it, nor free memory it still reads.
        let lease = SnapshotLease::pin(&a.env.catalog, cache, sub.tables);
        let plan = AnalyzedPlan {
            plan: krishiv_plan::Plan::try_new(sub.plan)?,
            ..(*a.analyzed).clone()
        };
        Ok((Some(plan), lease))
    }

    /// The physical plan with every cached (sub)plan substituted, and what
    /// to fill into the cache when the stream completes: the key and the
    /// snapshot shape when the whole plan is registered but holds no data
    /// yet, else nothing. A registered proper subtree without data is run
    /// first, as a query of its own, so the plan reads its snapshot.
    pub(crate) async fn physical_cached(
        &self,
        a: &Arc<Analysis>,
    ) -> Result<(krishiv_exec::planner::PhysicalPlan, CacheFill), KrishivError> {
        self.session.register_pending_caches().await;
        self.session.fill_registered_subtrees(&a.analyzed).await;
        let key = crate::cache::key_of(a.analyzed.plan.root(), &a.analyzed);
        let hit = self.session.state.cache.lookup(&key).is_some();
        let mut lease = SnapshotLease::default();
        let planned = match self.cache_substituted(a) {
            Ok((owned, held)) => {
                lease = held;
                let analyzed = owned.as_ref().unwrap_or(&a.analyzed);
                self.session.physical(analyzed, &a.env).await
            }
            Err(error) => Err(error),
        };
        let physical = match planned {
            Ok((_, physical)) => physical,
            Err(error) => {
                // The query analyzed but never ran: an action's planning failure is a row of
                // `system.queries` too (no execution began, so it has no other row).
                let id = self.session.record_refused(
                    a.env.text.as_deref(),
                    a.env.statement_class,
                    &error,
                );
                *a.env.row.lock().unwrap_or_else(PoisonError::into_inner) = Some(id);
                return Err(error);
            }
        };
        let claim = if hit {
            None
        } else {
            self.session.state.cache.claim_fill(&key)
        };
        let kind = match claim {
            None => FillKind::No,
            Some(claim) => {
                let fields: Vec<_> = a
                    .analyzed
                    .output
                    .iter()
                    .map(|c| c.data_type.to_arrow_field(c.name.clone(), c.nullable))
                    .collect();
                let disk = self
                    .session
                    .state
                    .cache
                    .level_of(&key)
                    .is_some_and(|l| l.use_disk);
                FillKind::Fill {
                    disk,
                    key,
                    fallback: Arc::new(Schema::new(fields)),
                    reads_files: a.analyzed.reads_files,
                    generation: a.env.cache_generation,
                    cap: a.env.cache_cap,
                    claim,
                    tables: {
                        let mut tables = crate::cache::read_tables(a.analyzed.plan.root());
                        tables.extend(crate::cache::unresolved_tables(&self.plan));
                        tables.sort();
                        tables.dedup();
                        tables
                    },
                }
            }
        };
        Ok((physical, CacheFill { kind, lease }))
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

    /// Caches this frame's rows (PySpark's `cache`, at
    /// `spark.sql.defaultCacheStorageLevel`, `MEMORY_AND_DISK` unless set): the plan is registered at once
    /// ([`Self::is_cached`] is true from this call on), and the first action
    /// of any frame that uses it materializes it into the session's relation
    /// cache; later actions read the snapshot instead of recomputing it.
    /// Materialization is lazy and all-or-nothing; when the snapshot would
    /// exceed `krishiv.cache.maxBytes` the query runs uncached.
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
        self.persist(StorageLevel::default_of(
            &self.session.state.conf.snapshot(),
        ))
    }

    /// Caches this frame's rows at `level` (PySpark's `persist`; where each
    /// level keeps them: [`StorageLevel`]). Persisting a plan
    /// that is already cached keeps the first level (Spark logs a warning),
    /// and [`StorageLevel::NONE`] does nothing.
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
        if self.invalid.is_none() {
            self.session
                .state
                .cache
                .request(Arc::clone(&self.plan), self.keep.clone(), level);
        }
        self.clone()
    }

    /// Drops this frame's cached snapshot and its registration, if any
    /// (PySpark's `unpersist`), returning the pool memory to baseline.
    /// Frames derived from this one are unaffected: they read the cache only
    /// while an entry matches. Unpersisting an uncached frame does nothing.
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
                df.session.register_pending_caches().await;
                let a = df.analysis().await?;
                df.session
                    .state
                    .cache
                    .remove(&crate::cache::key_of(a.analyzed.plan.root(), &a.analyzed));
                Ok(df)
            })
            .await
    }

    /// Whether this frame's plan is cached: registered by `cache()` or
    /// `persist()` (whether or not it holds data yet), under this principal
    /// and policy version.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// assert!(!futures::executor::block_on(s.table("t").is_cached())?);
    /// assert!(futures::executor::block_on(s.table("t").cache().is_cached())?);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Spark's analysis errors.
    pub async fn is_cached(&self) -> Result<bool, KrishivError> {
        Ok(self.storage_level().await? != StorageLevel::NONE)
    }

    /// The level this frame's plan is cached at (PySpark's `storageLevel`):
    /// the level of the first `persist`, or [`StorageLevel::NONE`] when it is
    /// not cached.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # let schema = std::sync::Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new("a", arrow_schema::DataType::Int32, false)]));
    /// # s.register_batches("t", schema, vec![])?;
    /// let level = futures::executor::block_on(s.table("t").persist(krishiv_session::StorageLevel::DISK_ONLY).storage_level())?;
    /// assert_eq!(level, krishiv_session::StorageLevel::DISK_ONLY);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    ///
    /// # Errors
    /// Spark's analysis errors.
    pub async fn storage_level(&self) -> Result<StorageLevel, KrishivError> {
        let df = self.clone();
        self.session
            .on_executor(async move {
                df.session.register_pending_caches().await;
                let a = df.analysis().await?;
                Ok(df
                    .session
                    .state
                    .cache
                    .level_of(&crate::cache::key_of(a.analyzed.plan.root(), &a.analyzed))
                    .unwrap_or(StorageLevel::NONE))
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
                let (_, physical) = df.session.physical_inspected(&a.analyzed, &a.env).await?;
                let mut all = a.analyzed.warnings.clone();
                all.extend(physical.warnings().iter().cloned());
                Ok(all)
            })
            .await
    }

    /// The tables the query reads, in first-use order, as written
    /// (dotted, e.g. `tpch.lineitem`). Reads by path contribute no names.
    /// Pure and sync: walks the unresolved plan, subqueries included.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// # for v in ["t", "u"] {
    /// #     futures::executor::block_on(s.sql(&format!("CREATE TEMPORARY VIEW {v} AS SELECT 1 AS a")))?;
    /// # }
    /// let df = futures::executor::block_on(s.sql("SELECT t.a FROM t JOIN u ON t.a = u.a"))?;
    /// assert_eq!(df.input_tables(), vec!["t".to_owned(), "u".to_owned()]);
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    #[must_use]
    pub fn input_tables(&self) -> Vec<String> {
        use std::collections::HashSet;

        use krishiv_plan::ExprNode;

        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let mut visited: HashSet<*const ()> = HashSet::new();
        let mut plans = vec![Arc::clone(&self.plan)];
        while let Some(plan) = plans.pop() {
            let key = Arc::as_ptr(&plan).cast::<()>();
            if !visited.insert(key) {
                continue;
            }
            if let PlanNode::Read(ReadSource::Table { name, .. }) = &plan.node {
                let dotted = name.join(".");
                if seen.insert(dotted.clone()) {
                    out.push(dotted);
                }
            }
            // Children visit left-to-right: the stack pops the last push
            // first, so collected children go on reversed.
            let mut kids = Vec::new();
            plan.for_each_child(&mut |c| kids.push(Arc::clone(c)));
            kids.reverse();
            plans.extend(kids);
            let mut exprs = Vec::new();
            plan.for_each_expr(&mut |e| exprs.push(Arc::clone(e)));
            exprs.reverse();
            let mut estack = exprs;
            while let Some(expr) = estack.pop() {
                if let ExprNode::Subquery { plan, .. } = &expr.node {
                    plans.push(Arc::clone(plan));
                }
                expr.for_each_child(&mut |c| estack.push(Arc::clone(c)));
            }
        }
        out
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
            table_options: BTreeMap::new(),
            partition_by: Vec::new(),
            bucket_by: None,
            sort_by: None,
            format_set: false,
        }
    }

    /// Writes this DataFrame's rows to the table `table` (PySpark's
    /// `df.writeTo`): a [`DataFrameWriterV2`] collecting the request; the
    /// table's schema is this DataFrame's.
    ///
    /// ```
    /// # let s = krishiv_session::Session::new()?;
    /// let w = s.table("t").write_to_table("u").using("parquet");
    /// # Ok::<(), krishiv_common::error::KrishivError>(())
    /// ```
    pub fn write_to_table(&self, table: &str) -> DataFrameWriterV2 {
        DataFrameWriterV2 {
            df: self.clone(),
            table: table.to_owned(),
            provider: None,
            options: BTreeMap::new(),
            properties: BTreeMap::new(),
            partitioned_by: Vec::new(),
            cluster_by: Vec::new(),
        }
    }

    /// Runs a collected `writeTo` request in one call (the Python facade):
    /// the builder state plus the terminal `mode` (`create`, `replace`,
    /// `createOrReplace`, `append`, `overwrite`, `overwritePartitions`) and
    /// the overwrite `condition`, if given. The mapping from the facade's
    /// collected writer state to [`DataFrameWriterV2`] lives here, in
    /// `krishiv-session`, not in the facade.
    ///
    /// # Errors
    /// As the terminal action reports.
    #[expect(
        clippy::too_many_arguments,
        reason = "one bridge call site carrying the whole collected writer state"
    )]
    pub async fn write_to_table_with(
        &self,
        table: &str,
        provider: Option<String>,
        options: Vec<(String, String)>,
        properties: Vec<(String, String)>,
        partitioned_by: Vec<String>,
        cluster_by: Vec<String>,
        mode: &str,
        condition: Option<String>,
    ) -> Result<(), KrishivError> {
        let mut writer = self.write_to_table(table);
        if let Some(provider) = provider.as_deref() {
            writer = writer.using(provider);
        }
        for (key, value) in &options {
            writer = writer.option(key, value);
        }
        for (key, value) in &properties {
            writer = writer.table_property(key, value);
        }
        if !partitioned_by.is_empty() {
            writer = writer.partitioned_by(partitioned_by.iter().map(String::as_str));
        }
        if !cluster_by.is_empty() {
            writer = writer.cluster_by(cluster_by.iter().map(String::as_str));
        }
        match mode {
            "create" => writer.create().await,
            "replace" => writer.replace().await,
            "createOrReplace" => writer.create_or_replace().await,
            "append" => writer.append().await,
            "overwrite" => writer.overwrite(condition.as_deref().unwrap_or("")).await,
            "overwritePartitions" => writer.overwrite_partitions().await,
            _ => Err(crate::session::internal(format!(
                "unknown writer v2 mode {mode}"
            ))),
        }
    }

    /// Writes this DataFrame's rows to the directory `path` in `format`
    /// with `mode` (PySpark's `DataFrameWriter.save` collected as one
    /// call): `options` pairs and the `partition_by` columns. The engine
    /// futures run is the same as [`DataFrameWriter::save`].
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
        self.write_to_with(format, mode, path, options, partition_by, None, None)
            .await
    }

    /// [`Self::write_to`] with the `bucket_by` (count and columns) and
    /// `sort_by` state of the writer, which `save` refuses as Spark does.
    ///
    /// # Errors
    /// As [`DataFrameWriter::save`].
    #[expect(
        clippy::too_many_arguments,
        reason = "one bridge call carrying the whole writer state"
    )]
    pub async fn write_to_with(
        &self,
        format: &str,
        mode: &str,
        path: &str,
        options: Vec<(String, String)>,
        partition_by: Vec<String>,
        bucket_by: Option<(i32, Vec<String>)>,
        sort_by: Option<Vec<String>>,
    ) -> Result<(), KrishivError> {
        let mut writer = self.write().format(format).mode(mode);
        for (key, value) in &options {
            writer = writer.option(key, value);
        }
        if !partition_by.is_empty() {
            writer = writer.partition_by(partition_by.iter().map(String::as_str));
        }
        if let Some((buckets, cols)) = bucket_by {
            writer = writer.bucket_by(buckets, cols);
        }
        if let Some(cols) = sort_by {
            writer = writer.sort_by(cols);
        }
        writer.save(path).await
    }

    /// Writes this DataFrame's rows to the table `table` (PySpark's
    /// `DataFrameWriter.saveAsTable`, or `insertInto` when `insert_into` is
    /// set, collected as one call): `format`, `mode`, `options` pairs and the
    /// `partition_by`, `bucket_by` (count and columns) and `sort_by` state.
    ///
    /// # Errors
    /// As [`DataFrameWriter::save_as_table`] or [`DataFrameWriter::insert_into`].
    #[expect(
        clippy::too_many_arguments,
        reason = "one bridge call carrying the whole writer state"
    )]
    pub async fn write_table_with(
        &self,
        insert_into: bool,
        table: &str,
        format: Option<&str>,
        mode: &str,
        options: Vec<(String, String)>,
        partition_by: Vec<String>,
        bucket_by: Option<(i32, Vec<String>)>,
        sort_by: Option<Vec<String>>,
    ) -> Result<(), KrishivError> {
        let mut writer = self.write().mode(mode);
        if let Some(format) = format {
            writer = writer.format(format);
        }
        for (key, value) in &options {
            writer = writer.option(key, value);
        }
        if !partition_by.is_empty() {
            writer = writer.partition_by(partition_by.iter().map(String::as_str));
        }
        if let Some((buckets, cols)) = bucket_by {
            writer = writer.bucket_by(buckets, cols);
        }
        if let Some(cols) = sort_by {
            writer = writer.sort_by(cols);
        }
        if insert_into {
            writer.insert_into(table).await
        } else {
            writer.save_as_table(table).await
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
    /// or the cache prefix `` `__krishiv_cache_` `` (in any letter case),
    /// which [`Session::sql`] cannot spell either.
    pub async fn create_or_replace_temp_view(&self, name: &str) -> Result<(), KrishivError> {
        let (df, name) = (self.clone(), name.to_owned());
        self.session
            .on_executor(async move { df.temp_view_here(&name).await })
            .await
    }

    async fn temp_view_here(&self, name: &str) -> Result<(), KrishivError> {
        if crate::catalog::is_reserved_view_name(name) {
            return Err(crate::catalog::reserved_temp_view_name(name));
        }
        // Spark analyzes the view's plan first: its errors come first.
        let analysis = self.analysis().await?;
        let columns: Vec<TableColumn> =
            crate::char_write::refuse_indeterminate_schema(&analysis.analyzed)?
                .output
                .iter()
                .map(|c| TableColumn::new(c.name.clone(), c.data_type.clone(), c.nullable))
                .collect();
        // A view over files keeps listing them: files added or changed
        // later show up in the next query.
        // (Not a lazy checkpoint of one: its rows were taken at the analysis above.)
        if let (PlanNode::Read(source @ ReadSource::Files { .. }), false) =
            (&self.plan.node, checkpoint::has_lazy_checkpoint(&self.keep))
        {
            self.session.state.catalog.put(
                name,
                TempTable::Files {
                    source: source.clone(),
                    columns,
                    known: false,
                    permanent: false,
                    partitions: Some(Default::default()),
                },
            );
            self.session.state.cache.forget_table(name);
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
        // A query that reads the view runs its observed nodes: it completes the same observations.
        self.session
            .state
            .catalog
            .set_view_keeps(name, self.keep.clone());
        self.session.state.cache.forget_table(name);
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

/// The expressions of `cols`.
fn exprs(cols: impl IntoIterator<Item = impl Into<Column>>) -> Vec<Arc<Expr>> {
    cols.into_iter().map(|c| Arc::new(c.into().expr)).collect()
}

mod writer;
pub use writer::{DataFrameWriter, DataFrameWriterV2};
