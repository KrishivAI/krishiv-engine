// SPDX-License-Identifier: Apache-2.0
//! The physical plan as data, and its instantiation on a worker.
//!
//! # Two steps
//! 1. **Planning** ([`PhysicalPlanner`](crate::planner::PhysicalPlanner))
//!    turns an optimized logical plan into a [`PhysicalPlan`]: a tree of
//!    [`PhysNode`]s holding only values (resolved IR expressions with the
//!    attribute ids of their input columns, schemas, column indexes, ids,
//!    options). It reads table metadata (schemas, statistics, what a scan
//!    can filter) and builds no scan source, unless a scan factory cannot
//!    state its output schema without one
//!    ([`ScanPushdown::output_schema`](crate::ScanPushdown::output_schema)).
//! 2. **Instantiation** ([`instantiate`]) builds the executable graph of
//!    operators from a plan on the worker that runs it, with the
//!    resources of that worker ([`WorkerContext`]): table providers and
//!    their scan factories, the spill scope, the I/O context, formats and
//!    configuration a write builds its sink from. Expressions
//!    are compiled for the worker's kernels; runtime filters, subquery
//!    results and broadcast builds are published and read through the
//!    worker's [`ValueService`](crate::values::ValueService), one per
//!    instantiation, so a plan can be instantiated any number of times.
//!
//! A [`Scheduler`](crate::Scheduler) takes a [`PhysicalPlan`]; the
//! in-process [`QueryRunner`](crate::QueryRunner) instantiates it and runs
//! the graph.
//!
//! # Invariants
//! - Every type a plan holds implements the sealed [`PlanData`] marker: a
//!   node holding a service handle, a lock, an atomic or a cell does not
//!   compile.
//! - Ids are assigned by the planner and are plan data: [`ScanId`],
//!   [`FilterId`], [`SubqueryId`], [`ExchangeId`]; [`ValueId`] names the
//!   values among them. A runtime filter is described twice: the scan it
//!   filters lists it ([`ScanRuntimeFilter`]: column and key type), and
//!   the join that produces it lists it ([`JoinFilterPlan`]: key and
//!   options); instantiation connects the two through the value service.
//! - A subquery's value is read by the expressions of the node that lists
//!   it ([`SubqueryPlan`]) as the column of attribute [`SubqueryId::attr`].
//! - Data moves between partitions only at [`PhysNode::Exchange`] nodes,
//!   which the planner's property pass places where a node's required
//!   distribution is not met (ADR 0012). The plan holds no partition count:
//!   the count and the assignment of partitions to tasks belong to the task
//!   that runs a fragment (design P2-T171, I5). On one worker each exchange
//!   is lowered into the operators around it ([`instantiate`]).

pub(crate) mod data;
mod instantiate;
pub mod splits;

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use arrow_array::Array;
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use krishiv_common::secret::OptionValue;
use krishiv_io::{ColumnPath, FilterId, KeyFilterOptions};
use krishiv_kernels::hash::routing;
use krishiv_plan::{AttrId, Expr, Scan};

use data::plan_data;
pub use data::{PlanData, assert_plan_data};
#[cfg(test)]
mod shared_tests;

#[cfg(test)]
pub(crate) use instantiate::node_ids;
pub use instantiate::{SpillScopes, TaskSpec, WorkerContext, instantiate, instantiate_task};
pub(crate) use instantiate::{
    Unbuilt, Values, aggregate_node, aggregate_spec, expand_op, filter_op, generate_op,
    group_limit_op, hash_join, nested_loop_join, project_op, replicate_op, scan_filter, sort_op,
    top_n_op,
};
pub use krishiv_memory::{RelationLease, RelationRef, RelationStore};
pub use splits::{
    ScanSplits, SplitFileRef, assign_batch_splits, assign_file_splits, deal_ranges, files_of,
    partitions_of, plan_file_partitions,
};

use crate::aggregate::{AggregateExpr, AggregateOutput};
use crate::error::Result;
use crate::join::JoinType;
use crate::operators::{RowStageSettings, SortKey};
use crate::planner::ExplainNode;
use crate::window::{GroupLimitPlan, WindowPlan};

/// A scan of a plan, numbered by the planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScanId(pub u64);

/// An uncorrelated subquery of a plan. Its value is read as the column of
/// attribute [`Self::attr`]: the planner numbers subqueries in a range
/// above every attribute id it assigns, so no input column has it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubqueryId(pub u64);

impl SubqueryId {
    /// The attribute expressions read the value as.
    #[must_use]
    pub fn attr(self) -> AttrId {
        AttrId(self.0)
    }
}

/// An exchange of a plan, numbered by the planner. Shuffle output is keyed
/// by it (with the stage, map partition and attempt; ADR 0013, I10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExchangeId(pub u64);

/// A shared plan of a plan, numbered by the planner. A separate space from
/// [`ExchangeId`]: on one worker a shared plan is not an exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SharedId(pub u64);

/// A subplan run once per query instantiation and read in full by each
/// [`PhysNode::SharedRead`]: a CTE body several references would otherwise
/// compute separately.
#[derive(Debug, Clone, PartialEq)]
pub struct SharedPlan {
    /// Its number, unique in the plan.
    pub id: SharedId,
    /// The subplan.
    pub plan: PhysNode,
    /// Its printable tree.
    pub explain: ExplainNode,
    /// Whether it computes the same rows when run again: a lost output of
    /// one that does not forces its readers to run again too.
    pub deterministic: bool,
    /// Whether readers must see its rows in the order it produces them.
    pub ordered: bool,
}

/// A value computed while a plan runs and read by other operators of the
/// plan: a runtime filter, a subquery's result or a broadcast exchange's
/// build (design P2-T171, I3). Its producers publish partials of it, one
/// per producer partition, and its consumers read it once every producer
/// partition published ([`crate::values`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ValueId {
    /// The runtime filter a join's build publishes.
    Filter(FilterId),
    /// A subquery's result.
    Subquery(SubqueryId),
    /// The rows of a broadcast exchange.
    Broadcast(ExchangeId),
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Filter(id) => write!(f, "runtime filter {id}"),
            Self::Subquery(id) => write!(f, "subquery #{}", id.0),
            Self::Broadcast(id) => write!(f, "broadcast exchange #{}", id.0),
        }
    }
}

/// The version of a value: the epoch its producers computed it at. A
/// batch query has one, [`Version::BATCH`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version {
    /// The epoch.
    pub epoch: u64,
}

impl Version {
    /// The only version of a batch query's values (epoch 0).
    pub const BATCH: Self = Self { epoch: 0 };
}

/// How an [`PhysNode::Exchange`] places its input's rows into partitions.
/// Columns are indices into the exchange's input (and output: an exchange
/// keeps its input's columns).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Partitioning {
    /// By the routing hash `h` of the key columns: a row goes to the
    /// partition that owns its key group (design P2-T171 §5.1 and §7.2,
    /// T157 Appendix C). `h` is seedless, so equal keys meet in one
    /// partition in every process; a partition owns a contiguous range of
    /// key groups, so for `2^k` partitions it is the top `k` bits of `h`,
    /// and a radix partition of a breaker is a union of partitions.
    Hash {
        /// The key columns, in hash order.
        keys: Vec<usize>,
        /// The routing function.
        scheme: HashScheme,
    },
    /// Every row in one partition (a gather). Partitions are read in
    /// partition order, so a range-partitioned sorted input stays sorted.
    Single,
    /// Every partition receives every row.
    Broadcast,
    /// Rows dealt to partitions in turn. Each producer starts at an offset
    /// derived from its partition index, never from its attempt, and deals
    /// its rows in input order (I10: a retried task writes the same rows to
    /// the same partitions).
    RoundRobin,
    /// By ranges of the sort keys, partition `i` before partition `i + 1`
    /// in key order. Bounds are sampled when the exchange runs.
    Range {
        /// The sort keys.
        keys: Vec<SortKey>,
    },
}

/// The function a [`Partitioning::Hash`] exchange routes rows by. It is
/// part of the plan, so every worker that runs a fragment of a query routes
/// a key to the same partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HashScheme {
    /// No function named: a plan written before the routing hash was
    /// fixed. Its exchanges are only ever lowered on one worker, where no
    /// row is routed; such a plan cannot be split across workers.
    Unpinned,
    /// Key-group ranges of the routing hash `H_kg` version 1
    /// (`krishiv_kernels::hash::routing`): XXH64, seed 0, of the normalized
    /// key bytes, folded over several key columns. Every exchange the
    /// planner places uses it.
    KeyGroupV1,
}

impl HashScheme {
    /// The partition of `n` that each row of `keys` (the key columns, in
    /// hash order) goes to, written to `out` (cleared first).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for [`Unpinned`](Self::Unpinned), which names no
    /// function to route by; `KRISHIV_UNSUPPORTED_FEATURE` for a key type
    /// the routing hash does not cover.
    pub fn partitions(self, keys: &[&dyn Array], n: u32, out: &mut Vec<u32>) -> Result<()> {
        match self {
            Self::Unpinned => Err(crate::error::internal(
                "an exchange without a routing function cannot route rows".to_owned(),
            )),
            Self::KeyGroupV1 => {
                let mut h = Vec::new();
                routing::hash_columns(keys, &mut h)?;
                out.clear();
                out.extend(h.iter().map(|&h| routing::partition_of(h, n)));
                Ok(())
            }
        }
    }
}

/// An exchange: `input`'s rows placed into partitions by `partitioning`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExchangePlan {
    /// Its number in the plan.
    pub id: ExchangeId,
    /// The input.
    pub input: Box<PhysNode>,
    /// Where each row goes.
    pub partitioning: Partitioning,
}

/// How a hash join gets matching rows of its two inputs into one task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinDistribution {
    /// The build input is broadcast to every task of the probe input,
    /// which stays where it is.
    Broadcast,
    /// Both inputs are hash-partitioned on their keys, the same way: each
    /// task builds and probes one partition.
    Partitioned,
}

/// What every expression of a plan is evaluated with, fixed at planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExprEnv {
    /// `spark.sql.ansi.enabled`.
    pub ansi: bool,
    /// `spark.sql.legacy.statisticalAggregate`: the sample moment aggregates
    /// give NaN instead of NULL where Spark divides by zero.
    pub legacy_statistical_aggregate: bool,
    /// The session time zone as a fixed offset in seconds, when it is one.
    pub zone: Option<i32>,
    /// `spark.sql.session.timeZone` as configured, when it differs from what
    /// `zone` renders (`UTC`, `+05:30`). A plan that sets it needs the
    /// `session_zone_name` codec feature.
    pub zone_name: Option<Arc<str>>,
    /// The query's clock (microseconds since the epoch, UTC), read once at
    /// planning: every instantiation sees the same `current_date()`.
    pub query_start_micros: i64,
    /// The session user (`current_user()`).
    pub user: Arc<str>,
    /// `spark.sql.mapKeyDedupPolicy = LAST_WIN`: duplicate map keys keep
    /// the value inserted last. A plan that sets it needs the
    /// `map_key_dedup_policy` codec feature.
    pub map_key_dedup_last_win: bool,
    /// `spark.sql.legacy.javaCharsets`: `encode` and `decode` take Java's
    /// names. A plan that sets it needs the `legacy_charset_switches` codec
    /// feature.
    pub legacy_java_charsets: bool,
    /// `spark.sql.legacy.codingErrorAction`: `encode` and `decode` replace
    /// what they cannot map instead of raising. A plan that sets it needs
    /// the `legacy_charset_switches` codec feature.
    pub legacy_coding_error_action: bool,
}

impl ExprEnv {
    /// Whether the zone name, if any, names the zone's offset (a region or
    /// malformed name has none). A plan that does not hold this would print
    /// one zone and compute with another.
    #[must_use]
    pub fn zone_name_matches_zone(&self) -> bool {
        self.zone_name.as_deref().is_none_or(|name| {
            krishiv_kernels::cast::FixedOffset::parse(name).map(|o| o.0) == self.zone
        })
    }
}

/// A node of a physical plan. Children come first in each variant;
/// expressions are resolved IR over the attribute ids `layout` lists for
/// the input's columns.
#[derive(Debug, Clone, PartialEq)]
pub enum PhysNode {
    /// A table scan.
    Scan(ScanNode),
    /// One row without columns (`SELECT` without `FROM`).
    OneRow,
    /// A range of `id BIGINT NOT NULL` values.
    Range(RangeNode),
    /// The rows of `input` for which `predicate` is true.
    Filter {
        /// The input.
        input: Box<PhysNode>,
        /// The attribute of each input column.
        layout: Vec<AttrId>,
        /// A boolean expression.
        predicate: Arc<Expr>,
        /// The subqueries `predicate` reads.
        subqueries: Vec<SubqueryPlan>,
    },
    /// Columns of `input` and computed columns.
    Project {
        /// The input.
        input: Box<PhysNode>,
        /// The attribute of each input column.
        layout: Vec<AttrId>,
        /// The output columns.
        items: Vec<ProjectItemPlan>,
        /// Output nullability overriding the items' own (the branches of a
        /// `UNION ALL`, nullable where any branch is).
        nullable: Option<Vec<bool>>,
        /// The subqueries the items read.
        subqueries: Vec<SubqueryPlan>,
    },
    /// `limit` rows of `input` after skipping the first `offset`.
    Limit {
        /// The input.
        input: Box<PhysNode>,
        /// The number of leading rows to skip.
        offset: u64,
        /// The row count.
        limit: u64,
    },
    /// Each row of `input` `count` times (`INTERSECT ALL`, `EXCEPT ALL`).
    ReplicateRows {
        /// The input.
        input: Box<PhysNode>,
        /// The attribute of each input column.
        layout: Vec<AttrId>,
        /// The replication count.
        count: Arc<Expr>,
        /// The subqueries `count` reads.
        subqueries: Vec<SubqueryPlan>,
    },
    /// Each row of `input` once per element of `arg` (`explode`,
    /// `posexplode`, `inline`): the input's columns, then the generated ones.
    Generate {
        /// The input.
        input: Box<PhysNode>,
        /// The attribute of each input column.
        layout: Vec<AttrId>,
        /// What is generated.
        kind: krishiv_plan::GeneratorKind,
        /// Keep a row whose argument has no element, with NULL columns.
        outer: bool,
        /// The array or map column expression.
        arg: Arc<Expr>,
        /// The subqueries `arg` reads.
        subqueries: Vec<SubqueryPlan>,
    },
    /// Each row of `input` once per projection (`ROLLUP`, `CUBE`,
    /// `GROUPING SETS`).
    Expand {
        /// The input.
        input: Box<PhysNode>,
        /// The attribute of each input column.
        layout: Vec<AttrId>,
        /// The projections: the same output columns, each computed one way.
        projections: Vec<Vec<ProjectItemPlan>>,
        /// Whether each output column is nullable.
        nullable: Vec<bool>,
        /// The subqueries the projections read.
        subqueries: Vec<SubqueryPlan>,
    },
    /// `input` sorted by `keys` (columns of `input`).
    Sort {
        /// The input.
        input: Box<PhysNode>,
        /// The sort keys.
        keys: Vec<SortKey>,
    },
    /// The first `limit` rows of `input` in the order of `keys`.
    TopN {
        /// The input.
        input: Box<PhysNode>,
        /// The sort keys.
        keys: Vec<SortKey>,
        /// The row count.
        limit: u64,
    },
    /// Window functions over `input`, which is sorted by the plan's
    /// partitioning, then its ordering: the input's columns, then one per
    /// function. Its state is one partition at a time: it requires
    /// [`WindowPlan::required_distribution`] of its input.
    Window {
        /// The sorted input.
        input: Box<PhysNode>,
        /// Partitioning, ordering and functions.
        window: WindowPlan,
    },
    /// The rows of each partition of `input` ranked within a limit by a
    /// rank-like function (Spark's `WindowGroupLimit`), below the window
    /// whose filter keeps only those: `input` is sorted as the window's is,
    /// and a sort directly below also drops rows past the limit within
    /// each of its sorted batches.
    WindowGroupLimit {
        /// The sorted input.
        input: Box<PhysNode>,
        /// Partitioning, ordering, function and limit.
        limit: GroupLimitPlan,
    },
    /// A hash aggregate.
    Aggregate {
        /// The input.
        input: Box<PhysNode>,
        /// Keys and aggregates.
        aggregate: AggregatePlan,
    },
    /// A hash join building its right input.
    HashJoin(HashJoinPlan),
    /// A nested-loop join building its right input.
    NestedLoopJoin(NestedLoopPlan),
    /// `UNION ALL` of inputs of one schema.
    Union {
        /// The inputs.
        inputs: Vec<PhysNode>,
    },
    /// Writes `input`'s rows as files described by `write`; one row per
    /// file written (`path`, `size`, `rows`). Instantiation builds the
    /// file sink from `(write, task partition, task attempt)` with the
    /// worker's own I/O context, so the same plan bytes run on every
    /// worker. The task number in file names is the partition index.
    /// `None` is a plan written before the `write_plan` feature: it
    /// encodes exactly as before (no `write` field) and lists no feature,
    /// and instantiation refuses it.
    WriteFiles {
        /// The rows to write.
        input: Box<PhysNode>,
        /// What to write: format, options (secret references only),
        /// store, directory, job id and file limits. Plain data; absent
        /// in plans that predate the `write_plan` feature.
        write: Option<WritePlan>,
    },
    /// Moves rows between partitions.
    Exchange(ExchangePlan),
    /// The rows of a shared plan, under the reader's attribute ids.
    SharedRead {
        /// The plan read.
        id: SharedId,
        /// The attribute id of each of its output columns here.
        layout: Vec<AttrId>,
    },
}

/// A table scan.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanNode {
    /// The scan's number in its plan.
    pub id: ScanId,
    /// The resolved scan: source, output attributes, pinned version.
    pub scan: Scan,
    /// The table columns read, in output order.
    pub projection: Vec<ColumnPath>,
    /// A filter pushed into the scan.
    pub filter: Option<ScanFilter>,
    /// The runtime filters joins above publish to this scan.
    pub runtime_filters: Vec<ScanRuntimeFilter>,
    /// The runtime filters' row-stage settings.
    pub row_stage: RowStageSettings,
}

/// A range source: `range(start, end, step, numSlices)` (Spark's `Range`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeNode {
    /// First value (inclusive).
    pub start: i64,
    /// End (exclusive).
    pub end: i64,
    /// Step (never zero).
    pub step: i64,
    /// The partition count, when `numSlices` was written. `None` means the
    /// task's default parallelism decides at execution; the fragment never
    /// stores a count it did not get from `numSlices`.
    pub num_slices: Option<u32>,
    /// The output attribute.
    pub output: AttrId,
}

/// A filter pushed into a scan, over the table schema.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanFilter {
    /// The attribute of each table column.
    pub layout: Vec<AttrId>,
    /// The predicate.
    pub predicate: Arc<Expr>,
    /// The scan applies it exactly, so the plan has no `Filter` above the
    /// scan. Instantiation checks the scan still does.
    pub exact: bool,
}

/// A runtime filter as the scan it filters sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanRuntimeFilter {
    /// The filter.
    pub id: FilterId,
    /// The table column it filters.
    pub column: ColumnPath,
    /// The type of the join key it is collected from.
    pub key_type: DataType,
}

/// A runtime filter as the join that collects it sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct JoinFilterPlan {
    /// The filter.
    pub id: FilterId,
    /// The join key it is collected from (an index into the keys).
    pub key: usize,
    /// What the build collects.
    pub options: KeyFilterOptions,
}

/// One output column of a [`PhysNode::Project`].
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectItemPlan {
    /// An input column, optionally renamed.
    Column {
        /// Its index in the input.
        index: usize,
        /// The output name; the input's by default.
        name: Option<String>,
    },
    /// A computed column.
    Expr {
        /// The output name.
        name: String,
        /// The value.
        expr: Arc<Expr>,
    },
    /// A column of nulls in the Arrow type of an input column (the cell an
    /// expansion nulls).
    NullOf {
        /// The input column whose type it takes.
        index: usize,
        /// The output name.
        name: String,
    },
}

/// Scalar or `EXISTS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubqueryValueKind {
    /// The first column of at most one row; NULL without a row.
    Scalar,
    /// Whether the subquery returns a row.
    Exists,
}

/// An uncorrelated subquery, run before the node that reads its value.
#[derive(Debug, Clone, PartialEq)]
pub struct SubqueryPlan {
    /// Its id; expressions read the value as [`SubqueryId::attr`].
    pub id: SubqueryId,
    /// Scalar or `EXISTS`.
    pub kind: SubqueryValueKind,
    /// The value's type.
    pub data_type: DataType,
    /// The subquery (one column for a scalar subquery).
    pub plan: PhysNode,
}

/// The keys and aggregates of a [`PhysNode::Aggregate`] (the
/// [`AggregateSpec`](crate::aggregate::AggregateSpec) without its
/// worker resources).
#[derive(Debug, Clone, PartialEq)]
pub struct AggregatePlan {
    /// Grouping key columns of the input.
    pub group_by: Vec<usize>,
    /// The aggregates.
    pub aggregates: Vec<AggregateExpr>,
    /// Which aggregates are `DISTINCT`, one flag per aggregate.
    pub distinct: Vec<bool>,
    /// Final results or partial states.
    pub output: AggregateOutput,
}

/// The conjuncts of a join's residual filter, over probe then build
/// columns.
#[derive(Debug, Clone, PartialEq)]
pub struct Residual {
    /// The attribute of each probe column, then of each build column.
    pub layout: Vec<AttrId>,
    /// The conjuncts.
    pub conjuncts: Vec<Arc<Expr>>,
}

/// A hash join: `build` (the right input) builds the table `probe`
/// probes.
#[derive(Debug, Clone, PartialEq)]
pub struct HashJoinPlan {
    /// The probe input.
    pub probe: Box<PhysNode>,
    /// The build input.
    pub build: Box<PhysNode>,
    /// The join type.
    pub join_type: JoinType,
    /// Key columns of the probe input.
    pub probe_keys: Vec<usize>,
    /// Key columns of the build input.
    pub build_keys: Vec<usize>,
    /// Per key, whether NULLs match (`<=>`).
    pub null_safe: Vec<bool>,
    /// The non-equality conditions.
    pub residual: Option<Residual>,
    /// A left outer join that fails when a probe row matches more than
    /// once (a correlated scalar subquery without an aggregate). Only
    /// `LeftOuter` joins set it.
    pub single_match: bool,
    /// The build's rows are never output and the join reads nothing of
    /// them but their keys (a left semi, left anti or existence join
    /// without a residual condition), so the build may keep each key once
    /// (design P2-T166 §3.2). An executor that ignores it stores every row
    /// and returns the same result; this one does so for now.
    pub distinct_build: bool,
    /// The runtime filters the build publishes.
    pub runtime_filters: Vec<JoinFilterPlan>,
    /// Broadcast or partitioned.
    pub distribution: JoinDistribution,
}

impl HashJoinPlan {
    /// Whether [`distinct_build`](Self::distinct_build) may be set: the
    /// join outputs no build rows and reads no build column but the keys
    /// (a left semi, left anti or existence join without a residual).
    #[must_use]
    pub fn distinct_build_allowed(&self) -> bool {
        self.residual.is_none()
            && matches!(
                self.join_type,
                JoinType::LeftSemi | JoinType::LeftAnti | JoinType::Existence
            )
    }
}

/// A nested-loop join over every pair of rows.
#[derive(Debug, Clone, PartialEq)]
pub struct NestedLoopPlan {
    /// The probe input.
    pub probe: Box<PhysNode>,
    /// The build input.
    pub build: Box<PhysNode>,
    /// The join type.
    pub join_type: JoinType,
    /// The join condition.
    pub residual: Option<Residual>,
}

/// What a [`PhysNode::WriteFiles`] writes: the file format and its options
/// (secret references only, never inline secrets or URLs with
/// credentials), the store and directory, the job UUID naming its files,
/// and the file limits. Plain data: the same bytes on every worker, and
/// the sink is built at instantiation from `(self, task partition, task
/// attempt)`.
#[derive(Debug, Clone, PartialEq)]
pub struct WritePlan {
    /// The format name (for example `parquet`).
    pub format: String,
    /// The format options, as given (references only on the wire).
    pub options: BTreeMap<String, OptionValue>,
    /// The store id, as text.
    pub store: String,
    /// The target directory, as text.
    pub dir: String,
    /// The job UUID naming the write's files.
    pub job_uuid: String,
    /// `maxRecordsPerFile`; 0 means unlimited.
    pub max_records_per_file: u64,
    /// Roll over to a new file once this many bytes were produced; 0 means
    /// never.
    pub target_file_bytes: u64,
    /// Partition column indices into the writer's input, in `partitionBy`
    /// order (Spark's directory layout); empty for unpartitioned output.
    /// A partitioned write sorts its input by these columns directly
    /// below the writer.
    pub partition_by: Vec<usize>,
}

impl WritePlan {
    /// The schema of a write's result rows (`path`, `size`, `rows`): the
    /// commit message. It matches the file sink's result schema exactly.
    #[must_use]
    pub fn result_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("path", krishiv_types::DataType::String.to_arrow(), false),
            Field::new("size", DataType::Int64, false),
            Field::new("rows", DataType::Int64, false),
        ]))
    }

    /// The `EXPLAIN` detail (`<format> to <store>/<dir>`).
    #[must_use]
    pub fn describe(&self) -> String {
        format!("{} to {}/{}", self.format, self.store, self.dir)
    }
}

plan_data!(tuple ScanId(id));
plan_data!(tuple SubqueryId(id));
plan_data!(tuple ExchangeId(id));
plan_data!(tuple SharedId(id));
plan_data!(struct SharedPlan { id, plan, explain, deterministic, ordered });
plan_data!(
    enum ValueId {
        Filter(id),
        Subquery(id),
        Broadcast(id),
    }
);
plan_data!(struct Version { epoch });
plan_data!(
    enum HashScheme {
        Unpinned,
        KeyGroupV1,
    }
);
plan_data!(enum Partitioning {
    Hash { keys, scheme },
    Single,
    Broadcast,
    RoundRobin,
    Range { keys },
});
plan_data!(struct ExchangePlan { id, input, partitioning });
plan_data!(
    enum JoinDistribution {
        Broadcast,
        Partitioned,
    }
);
plan_data!(struct ExprEnv {
    ansi,
    legacy_statistical_aggregate,
    zone,
    zone_name,
    query_start_micros,
    user,
    map_key_dedup_last_win,
    legacy_java_charsets,
    legacy_coding_error_action
});
plan_data!(enum PhysNode {
    Scan(scan),
    OneRow,
    Range(range),
    Filter { input, layout, predicate, subqueries },
    Project { input, layout, items, nullable, subqueries },
    Limit { input, offset, limit },
    ReplicateRows { input, layout, count, subqueries },
    Generate { input, layout, kind, outer, arg, subqueries },
    Expand { input, layout, projections, nullable, subqueries },
    Sort { input, keys },
    TopN { input, keys, limit },
    Window { input, window },
    WindowGroupLimit { input, limit },
    Aggregate { input, aggregate },
    HashJoin(join),
    NestedLoopJoin(join),
    Union { inputs },
    WriteFiles { input, write },
    Exchange(exchange),
    SharedRead { id, layout },
});
plan_data!(struct WritePlan {
    format,
    options,
    store,
    dir,
    job_uuid,
    max_records_per_file,
    target_file_bytes,
    partition_by
});
plan_data!(struct ScanNode { id, scan, projection, filter, runtime_filters, row_stage });
plan_data!(struct RangeNode { start, end, step, num_slices, output });
plan_data!(struct ScanFilter { layout, predicate, exact });
plan_data!(struct ScanRuntimeFilter { id, column, key_type });
plan_data!(struct JoinFilterPlan { id, key, options });
plan_data!(enum ProjectItemPlan { Column { index, name }, Expr { name, expr }, NullOf { index, name } });
plan_data!(
    enum SubqueryValueKind {
        Scalar,
        Exists,
    }
);
plan_data!(struct SubqueryPlan { id, kind, data_type, plan });
plan_data!(struct AggregatePlan { group_by, aggregates, distinct, output });
plan_data!(struct Residual { layout, conjuncts });
plan_data!(struct HashJoinPlan {
    probe, build, join_type, probe_keys, build_keys, null_safe, residual, single_match,
    distinct_build, runtime_filters, distribution
});
plan_data!(struct NestedLoopPlan { probe, build, join_type, residual });
plan_data!(struct PhysicalPlan { root, shared, schema, explain, warnings, env });

impl PhysNode {
    /// The plans this node reads: its inputs, then the subqueries it runs
    /// first.
    pub fn children(&self) -> Vec<&PhysNode> {
        match self {
            Self::Scan(_) | Self::OneRow | Self::Range(_) | Self::SharedRead { .. } => Vec::new(),
            Self::Filter {
                input, subqueries, ..
            }
            | Self::Project {
                input, subqueries, ..
            }
            | Self::Expand {
                input, subqueries, ..
            }
            | Self::ReplicateRows {
                input, subqueries, ..
            }
            | Self::Generate {
                input, subqueries, ..
            } => std::iter::once(input.as_ref())
                .chain(subqueries.iter().map(|s| &s.plan))
                .collect(),
            Self::Limit { input, .. }
            | Self::Sort { input, .. }
            | Self::TopN { input, .. }
            | Self::Window { input, .. }
            | Self::WindowGroupLimit { input, .. }
            | Self::Aggregate { input, .. }
            | Self::WriteFiles { input, .. } => vec![input.as_ref()],
            Self::Exchange(e) => vec![e.input.as_ref()],
            Self::HashJoin(j) => vec![j.probe.as_ref(), j.build.as_ref()],
            Self::NestedLoopJoin(j) => vec![j.probe.as_ref(), j.build.as_ref()],
            Self::Union { inputs } => inputs.iter().collect(),
        }
    }

    /// [`Self::children`], mutable.
    pub(crate) fn children_mut(&mut self) -> Vec<&mut PhysNode> {
        match self {
            Self::Scan(_) | Self::OneRow | Self::Range(_) | Self::SharedRead { .. } => Vec::new(),
            Self::Filter {
                input, subqueries, ..
            }
            | Self::Project {
                input, subqueries, ..
            }
            | Self::Expand {
                input, subqueries, ..
            }
            | Self::ReplicateRows {
                input, subqueries, ..
            }
            | Self::Generate {
                input, subqueries, ..
            } => std::iter::once(input.as_mut())
                .chain(subqueries.iter_mut().map(|s| &mut s.plan))
                .collect(),
            Self::Limit { input, .. }
            | Self::Sort { input, .. }
            | Self::TopN { input, .. }
            | Self::Window { input, .. }
            | Self::WindowGroupLimit { input, .. }
            | Self::Aggregate { input, .. }
            | Self::WriteFiles { input, .. } => vec![input.as_mut()],
            Self::Exchange(e) => vec![e.input.as_mut()],
            Self::HashJoin(j) => vec![j.probe.as_mut(), j.build.as_mut()],
            Self::NestedLoopJoin(j) => vec![j.probe.as_mut(), j.build.as_mut()],
            Self::Union { inputs } => inputs.iter_mut().collect(),
        }
    }
}

/// An executable plan as data: the root node, the schema of its result
/// batches, the printable tree and the expression environment. Hand it to
/// a [`Scheduler`](crate::Scheduler); [`instantiate`] builds the operators.
#[derive(Clone, PartialEq)]
pub struct PhysicalPlan {
    root: Arc<PhysNode>,
    shared: Vec<SharedPlan>,
    schema: SchemaRef,
    explain: ExplainNode,
    warnings: Vec<String>,
    env: ExprEnv,
}

impl PhysicalPlan {
    /// A plan of `root`, whose result batches have `schema`, printed as
    /// `explain`, evaluating its expressions with `env`.
    pub fn new(root: PhysNode, schema: SchemaRef, explain: ExplainNode, env: ExprEnv) -> Self {
        Self {
            root: Arc::new(root),
            shared: Vec::new(),
            schema,
            explain,
            warnings: Vec::new(),
            env,
        }
    }

    /// The root node.
    pub fn root(&self) -> &PhysNode {
        &self.root
    }

    /// The plans [`PhysNode::SharedRead`]s read, inner first.
    pub fn shared(&self) -> &[SharedPlan] {
        &self.shared
    }

    /// Every tree of the plan: the shared plans in order, then the root. A
    /// walk over the plan's nodes starts here.
    pub fn roots(&self) -> impl Iterator<Item = &PhysNode> {
        self.shared
            .iter()
            .map(|s| &s.plan)
            .chain(std::iter::once(&*self.root))
    }

    /// The same plan reading `shared`, listed inner first.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if two shared plans have one id, or a node reads a
    /// shared plan that is not listed before it.
    pub fn with_shared(mut self, shared: Vec<SharedPlan>) -> Result<Self> {
        let mut known = std::collections::HashSet::with_capacity(shared.len());
        for s in &shared {
            check_shared_reads(&s.plan, &known)?;
            if !known.insert(s.id) {
                return Err(crate::error::internal(format!(
                    "two shared plans numbered {}",
                    s.id.0
                )));
            }
        }
        check_shared_reads(&self.root, &known)?;
        self.shared = shared;
        Ok(self)
    }

    /// The schema of the result batches: the logical plan's output names.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// The printable tree of the root.
    pub fn explain(&self) -> &ExplainNode {
        &self.explain
    }

    /// The printable tree of the whole plan: as [`Self::explain`], with the
    /// shared plans printed first when there are any.
    #[must_use]
    pub fn explain_all(&self) -> ExplainNode {
        if self.shared.is_empty() {
            return self.explain.clone();
        }
        let mut children: Vec<ExplainNode> = self
            .shared
            .iter()
            .map(|s| shared_explain(s.id, s.explain.clone()))
            .collect();
        children.push(self.explain.clone());
        ExplainNode::new(WITH_SHARED, "", children)
    }

    /// The expression environment.
    pub fn env(&self) -> &ExprEnv {
        &self.env
    }

    /// Warnings planning produced (an ignored hint, as Spark logs it).
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// The plan writing this plan's rows as the files `write` describes
    /// (`df.write`): its result is one row per file written (`path`,
    /// `size`, `rows`), the commit message. The worker it runs on builds
    /// the file sink at instantiation from `(write, task partition, task
    /// attempt)` with its own I/O context. The caller starts the write
    /// before (`DirectoryWrite::begin`) and commits it from the plan's
    /// result rows once the plan has run.
    ///
    /// A partitioned write (`write.partition_by`) sorts the rows by the
    /// partition columns directly below the writer (Spark's required
    /// ordering, in the existing spillable local sort), so each task keeps
    /// exactly one open file however many partitions it sees.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if a partition column is out of range.
    pub fn into_write(self, write: WritePlan) -> Result<Self> {
        let fields = self.schema.fields();
        let mut names = Vec::with_capacity(write.partition_by.len());
        for &c in &write.partition_by {
            let Some(field) = fields.get(c) else {
                return Err(crate::error::internal(format!(
                    "partition column {c} is out of range"
                )));
            };
            names.push(field.name().as_str());
        }
        let (root, explain) = if names.is_empty() {
            (
                Arc::unwrap_or_clone(self.root),
                ExplainNode::new("WriteFiles", write.describe(), vec![self.explain]),
            )
        } else {
            let keys: Vec<SortKey> = write
                .partition_by
                .iter()
                .map(|&c| SortKey::asc(c))
                .collect();
            let detail = format!(
                "[{}]",
                names
                    .iter()
                    .map(|n| format!("{n} ASC NULLS FIRST"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let sort = ExplainNode::new("Sort", detail, vec![self.explain]);
            let explain = ExplainNode::new(
                "WriteFiles",
                format!("{} partitioned by [{}]", write.describe(), names.join(", ")),
                vec![sort],
            );
            (
                PhysNode::Sort {
                    input: Box::new(Arc::unwrap_or_clone(self.root)),
                    keys,
                },
                explain,
            )
        };
        Ok(Self {
            schema: WritePlan::result_schema(),
            root: Arc::new(PhysNode::WriteFiles {
                input: Box::new(root),
                write: Some(write),
            }),
            shared: self.shared,
            explain,
            warnings: self.warnings,
            env: self.env,
        })
    }

    /// The same plan with `warnings` added.
    #[must_use]
    pub fn with_warnings(mut self, warnings: Vec<String>) -> Self {
        self.warnings.extend(warnings);
        self
    }
}

impl fmt::Debug for PhysicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PhysicalPlan")
            .field("schema", &self.schema)
            .field("explain", &self.explain.to_string())
            .field("warnings", &self.warnings)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for PhysicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.explain_all().fmt(f)
    }
}

/// The node [`PhysicalPlan::explain_all`] prints above a plan with shared
/// plans.
pub(crate) const WITH_SHARED: &str = "WithShared";
/// The node printed above a shared plan.
pub(crate) const SHARED_PLAN: &str = "SharedPlan";

fn shared_explain(id: SharedId, body: ExplainNode) -> ExplainNode {
    ExplainNode::new(SHARED_PLAN, format!("#{}", id.0), vec![body])
}

/// Every [`PhysNode::SharedRead`] under `node` reads a plan in `known`.
fn check_shared_reads(node: &PhysNode, known: &std::collections::HashSet<SharedId>) -> Result<()> {
    let mut stack = vec![node];
    while let Some(n) = stack.pop() {
        if let PhysNode::SharedRead { id, .. } = n
            && !known.contains(id)
        {
            return Err(crate::error::internal(format!(
                "a read of shared plan {}, which is not listed before it",
                id.0
            )));
        }
        stack.extend(n.children());
    }
    Ok(())
}
