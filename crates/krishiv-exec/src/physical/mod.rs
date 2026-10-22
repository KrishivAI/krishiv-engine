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
//!    their scan factories, the spill scope, the write sink. Expressions
//!    are compiled for the worker's kernels; runtime-filter cells and
//!    subquery value cells are created here, one set per instantiation, so
//!    a plan can be instantiated any number of times.
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
//!   [`FilterId`], [`SubqueryId`]. A runtime filter is described twice:
//!   the scan it filters lists it ([`ScanRuntimeFilter`]: column and key
//!   type), and the join that produces it lists it ([`JoinFilterPlan`]: key
//!   and options); instantiation connects the two through one cell per id.
//! - A subquery's value is read by the expressions of the node that lists
//!   it ([`SubqueryPlan`]) as the column of attribute [`SubqueryId::attr`].
//! - Data moves between partitions only at [`PhysNode::Exchange`] nodes,
//!   which the planner's property pass places where a node's required
//!   distribution is not met (ADR 0012). The plan holds no partition count:
//!   the count and the assignment of partitions to tasks belong to the task
//!   that runs a fragment (design P2-T171, I5). On one worker each exchange
//!   is lowered into the operators around it ([`instantiate`]).

mod data;
mod instantiate;

use std::fmt;
use std::sync::Arc;

use arrow_schema::{DataType, SchemaRef};
use krishiv_io::{ColumnPath, FilterId, KeyFilterOptions};
use krishiv_plan::{AttrId, Expr, Scan};

use data::plan_data;
pub use data::{PlanData, assert_plan_data};
#[cfg(test)]
pub(crate) use instantiate::node_ids;
pub use instantiate::{SpillScopes, WorkerContext, instantiate};
pub(crate) use instantiate::{
    Unbuilt, Values, aggregate_node, aggregate_spec, filter_op, hash_join, nested_loop_join,
    project_op, replicate_op, scan_filter, sort_op, top_n_op,
};

use crate::aggregate::{AggregateExpr, AggregateOutput};
use crate::error::Result;
use crate::join::JoinType;
use crate::operators::{FileWriteSink, RowStageSettings, SortKey};
use crate::planner::ExplainNode;

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

/// How an [`PhysNode::Exchange`] places its input's rows into partitions.
/// Columns are indices into the exchange's input (and output: an exchange
/// keeps its input's columns).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Partitioning {
    /// By the routing hash `h` of the key columns: a row goes to the
    /// partition its top bits name (design P2-T171 §5.1, T157 Appendix
    /// C). `h` is seedless, so equal keys meet in one partition in every
    /// process; partition counts are powers of two, so a partition nests
    /// in a key group and a radix partition of a breaker is a union of
    /// partitions.
    Hash {
        /// The key columns, in hash order.
        keys: Vec<usize>,
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
    /// The session time zone as a fixed offset in seconds, when it is one.
    pub zone: Option<i32>,
    /// The query's clock (microseconds since the epoch, UTC), read once at
    /// planning: every instantiation sees the same `current_date()`.
    pub query_start_micros: i64,
    /// The session user (`current_user()`).
    pub user: Arc<str>,
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
    /// The first `limit` rows of `input`.
    Limit {
        /// The input.
        input: Box<PhysNode>,
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
    /// Writes `input`'s rows through the worker's write sink
    /// ([`WorkerContext::with_write_sink`]); one row per file written.
    WriteFiles {
        /// The rows to write.
        input: Box<PhysNode>,
    },
    /// Moves rows between partitions.
    Exchange(ExchangePlan),
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
    /// The runtime filters the build publishes.
    pub runtime_filters: Vec<JoinFilterPlan>,
    /// Broadcast or partitioned.
    pub distribution: JoinDistribution,
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

plan_data!(tuple ScanId(id));
plan_data!(tuple SubqueryId(id));
plan_data!(tuple ExchangeId(id));
plan_data!(enum Partitioning {
    Hash { keys },
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
plan_data!(struct ExprEnv { ansi, zone, query_start_micros, user });
plan_data!(enum PhysNode {
    Scan(scan),
    OneRow,
    Filter { input, layout, predicate, subqueries },
    Project { input, layout, items, nullable, subqueries },
    Limit { input, limit },
    ReplicateRows { input, layout, count, subqueries },
    Sort { input, keys },
    TopN { input, keys, limit },
    Aggregate { input, aggregate },
    HashJoin(join),
    NestedLoopJoin(join),
    Union { inputs },
    WriteFiles { input },
    Exchange(exchange),
});
plan_data!(struct ScanNode { id, scan, projection, filter, runtime_filters, row_stage });
plan_data!(struct ScanFilter { layout, predicate, exact });
plan_data!(struct ScanRuntimeFilter { id, column, key_type });
plan_data!(struct JoinFilterPlan { id, key, options });
plan_data!(enum ProjectItemPlan { Column { index, name }, Expr { name, expr } });
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
    probe, build, join_type, probe_keys, build_keys, null_safe, residual, runtime_filters,
    distribution
});
plan_data!(struct NestedLoopPlan { probe, build, join_type, residual });
plan_data!(struct PhysicalPlan { root, schema, explain, warnings, env });

impl PhysNode {
    /// The plans this node reads: its inputs, then the subqueries it runs
    /// first.
    pub fn children(&self) -> Vec<&PhysNode> {
        match self {
            Self::Scan(_) | Self::OneRow => Vec::new(),
            Self::Filter {
                input, subqueries, ..
            }
            | Self::Project {
                input, subqueries, ..
            }
            | Self::ReplicateRows {
                input, subqueries, ..
            } => std::iter::once(input.as_ref())
                .chain(subqueries.iter().map(|s| &s.plan))
                .collect(),
            Self::Limit { input, .. }
            | Self::Sort { input, .. }
            | Self::TopN { input, .. }
            | Self::Aggregate { input, .. }
            | Self::WriteFiles { input } => vec![input.as_ref()],
            Self::Exchange(e) => vec![e.input.as_ref()],
            Self::HashJoin(j) => vec![j.probe.as_ref(), j.build.as_ref()],
            Self::NestedLoopJoin(j) => vec![j.probe.as_ref(), j.build.as_ref()],
            Self::Union { inputs } => inputs.iter().collect(),
        }
    }

    /// [`Self::children`], mutable.
    pub(crate) fn children_mut(&mut self) -> Vec<&mut PhysNode> {
        match self {
            Self::Scan(_) | Self::OneRow => Vec::new(),
            Self::Filter {
                input, subqueries, ..
            }
            | Self::Project {
                input, subqueries, ..
            }
            | Self::ReplicateRows {
                input, subqueries, ..
            } => std::iter::once(input.as_mut())
                .chain(subqueries.iter_mut().map(|s| &mut s.plan))
                .collect(),
            Self::Limit { input, .. }
            | Self::Sort { input, .. }
            | Self::TopN { input, .. }
            | Self::Aggregate { input, .. }
            | Self::WriteFiles { input } => vec![input.as_mut()],
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

    /// The schema of the result batches: the logical plan's output names.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// The printable tree.
    pub fn explain(&self) -> &ExplainNode {
        &self.explain
    }

    /// The expression environment.
    pub fn env(&self) -> &ExprEnv {
        &self.env
    }

    /// Warnings planning produced (an ignored hint, as Spark logs it).
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// The plan writing this plan's rows through `sink` (`df.write`): its
    /// result is one row per file written (`path`, `size`, `rows`). The
    /// worker it runs on writes through the sink its [`WorkerContext`]
    /// holds ([`WorkerContext::with_write_sink`]), which must write this
    /// plan's schema. The caller starts the write before
    /// (`DirectoryWrite::begin`, whose write options built `sink` over
    /// [`Self::schema`]) and commits it with [`FileWriteSink::files`] once
    /// the plan has run.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if `sink` writes another schema.
    pub fn into_write(self, sink: &FileWriteSink) -> Result<Self> {
        if sink.input_schema().fields() != self.schema.fields() {
            return Err(crate::error::internal(
                "a write sink of another schema than its plan".to_owned(),
            ));
        }
        let explain = ExplainNode::new("WriteFiles", sink.describe(), vec![self.explain]);
        let root = Arc::unwrap_or_clone(self.root);
        Ok(Self {
            schema: crate::operator::Source::schema(sink),
            root: Arc::new(PhysNode::WriteFiles {
                input: Box::new(root),
            }),
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
        self.explain.fmt(f)
    }
}
