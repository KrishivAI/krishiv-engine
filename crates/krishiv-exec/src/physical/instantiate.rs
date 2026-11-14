// SPDX-License-Identifier: Apache-2.0
//! Instantiation: a [`PhysicalPlan`] and a worker's resources in, the
//! executable operator graph out.
//!
//! The operator builders here are shared with the planner, which calls
//! them with throw-away cells to validate a node and learn its schema:
//! a plan that planned is a plan that instantiates, with the same schemas,
//! unless the worker's tables changed in between (each scan checks that
//! its pushed filter is still honoured as planned, and the root's schema
//! is checked against the plan's).
//!
//! # Exchanges on one worker
//! A worker that runs a whole plan lowers each exchange into the operators
//! around it, without moving rows (design P2-T171 §5, ADR 0006 update of
//! 2026-10-19): exchanges are distribution boundaries, and morsel-driven
//! drivers stay the parallelism within a worker.
//!
//! | Partitioning | Lowered into |
//! |---|---|
//! | `Hash` below an aggregate | the aggregate's breaker, whose partitions are the hash's |
//! | `Hash` below a sort by its keys first (a window's) | the sort's hash buckets, merged and read in parallel |
//! | `Hash` on both inputs of a join | the build's partitions; the probe needs none, as the build is complete |
//! | `Broadcast` | the shared, immutable build, published as a value |
//! | `Single` | the breaker's combine and merge, or the result stream |
//! | `Range` | the sort's run merge |
//! | `RoundRobin` | the drivers' morsels |
//!
//! Every one of them is the identity on the rows a worker computes from
//! the whole input, so an exchange instantiates as its input.
//!
//! # Values
//! Runtime filters, subquery results and broadcast builds go through the
//! worker's [`ValueService`] (a fresh [`LocalValues`] per instantiation
//! unless the context sets one): instantiation declares each filter there,
//! scans read the cells it hands out, and producers publish their partials
//! to it.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use arrow_schema::{Field, Schema, SchemaRef};
use krishiv_common::config::Config;
use krishiv_common::error::KrishivUnsupportedFeature;
use krishiv_io::{
    ColumnPath, FilterId, FilterSnapshot, FormatRegistry, IoContext, KeyFilter, StoreId,
    WriteOptions, effective_job_uuid, is_spark_file_source,
};
use krishiv_memory::SpillScope;
use krishiv_plan::{AttrId, Expr};
use krishiv_udf_sdk::UdfLibrary;
use object_store::path::Path;

use super::{
    AggregatePlan, ExchangeId, ExchangePlan, HashJoinPlan, NestedLoopPlan, Partitioning, PhysNode,
    PhysicalPlan, ProjectItemPlan, Residual, ScanId, ScanNode, ScanSplits, SharedPlan,
    SortMergeJoinPlan, SubqueryPlan, ValueId, Version, WritePlan, files_of, partitions_of,
};
use crate::aggregate::{
    AggInput, AggregateExpr, AggregateNode, AggregateOutput, AggregateSpec, HashAggregate,
    distinct_plan,
};
use crate::error::{Result, internal};
use crate::exchange::{Exchange, ExchangeRead, ExchangeSink, ExchangeWrite, MapOutputId};
use crate::join::{
    BroadcastKey, BroadcastTables, BuildJob, HashJoin, HashJoinNode, JoinRuntimeFilter, JoinSpec,
    NestedLoopJoin, NestedLoopJoinNode, NestedLoopSpec, PrebuiltSink, SharedBuildJoinNode,
    SortMergeJoin, SortMergeJoinNode,
};
use crate::loops::UnionLoopSource;
use crate::operator::{Operator, Sink, Source};
use crate::operators::{
    Expand, FileWrite, FileWriteSink, Filter, FilterPushdown, Generate, LateFetch, NullOf,
    Predicate, Project, ProjectItem, PushedFilter, ReplicateRows, RuntimeFilters, ScanPushdown,
    ScanPushdowns, Sort, SortKey, TopN,
};
use crate::pipeline::{ExecNode, PipelineBuilder};
use crate::planner::TableProviderResolver;
use crate::planner::eval::{self, Compiled, EvalEnv, Input, PhysExpr};
use crate::planner::nodes::{BreakerNode, FanInNode, OperatorNode, OrderedNode, SourceNode};
use crate::planner::subquery::{SubquerySink, SubqueryValue, WithSubqueries};
use crate::shared::{Output, SharedSink, SharedSource};
use crate::values::{
    BroadcastBuild, LocalValues, Partial, Producer, Publication, Value, ValueService,
};
use crate::window::{GroupLimitPlan, WindowGroupLimit};

mod detached;
mod distributed;
mod filter;
mod order;
mod python_eval;
pub(crate) use detached::{Detached, LoopRun};
mod join;
use distributed::planned_schema_holds;
pub(crate) use join::hash_join;
use order::{range_ordered_evals, reads_range_in_order};

/// Creates the spill scope of one instantiation: the query's own spill
/// directory on this worker, removed when the scope's last holder (the
/// operators, the result stream) is dropped.
pub trait SpillScopes: Send + Sync + fmt::Debug {
    /// A new scope.
    ///
    /// # Errors
    /// When the spill directory cannot be created.
    fn scope(&self) -> Result<SpillScope>;
}

/// What a worker resolves a plan's references against: the tables its
/// scans read, the session UDFs its function calls invoke, the relation store
/// its batch splits resolve through, the spill scope of the query, the I/O
/// context, formats and configuration a write builds its sink from, and the
/// values of the plan. One per query
/// and worker.
#[derive(Clone)]
pub struct WorkerContext {
    tables: Arc<dyn TableProviderResolver>,
    udfs: Option<Arc<UdfLibrary>>,
    /// Resolves the Python UDFs a plan names (design P2-T059).
    python: Option<Arc<dyn crate::python_eval::UdfRuntime>>,
    relations: Option<Arc<krishiv_memory::RelationStore>>,
    spill: Option<SpillScope>,
    scopes: Option<Arc<dyn SpillScopes>>,
    io: Option<IoContext>,
    formats: Arc<FormatRegistry>,
    config: Option<Arc<Config>>,
    /// The plan's values; a fresh [`LocalValues`] per instantiation when
    /// unset.
    values: Option<Arc<dyn ValueService>>,
    producer: Producer,
    /// Where exchanges between stages are written and read.
    exchange: Option<Arc<dyn Exchange>>,
    /// The job's broadcast builds on this worker, shared by its tasks.
    broadcasts: Option<Arc<BroadcastTables>>,
    /// Whether a partial and a final aggregate around an exchange run as one
    /// breaker (the default; see [`fused_aggregate`]).
    fuse_aggregates: bool,
}

/// Which task of a stage instantiates a plan fragment: the stage, its
/// attempt, and the partition within it, with the scan splits of the tasks
/// that read them. The same fragment bytes run on every worker; the task
/// tells instantiation which partition's files a write names, which
/// attempt's suffix keeps a retry from overwriting a committed file, and
/// which splits each scan reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskSpec {
    /// The stage.
    pub stage: u32,
    /// The stage attempt.
    pub stage_attempt: u32,
    /// The partition index (the task number in file names).
    pub partition: u32,
    /// The partition count of the stage.
    pub partitions: u32,
    /// The task attempt (0 is the first try).
    pub attempt: u32,
    /// The splits each scan of the fragment reads, by scan id. A scan
    /// without an entry plans its own splits exactly as today, so
    /// single-process behaviour is unchanged.
    pub scans: BTreeMap<ScanId, ScanSplits>,
    /// What each [`PhysNode::ExchangeSource`] of the fragment reads, by
    /// exchange id (design P2-T176 §2.2).
    pub exchanges: BTreeMap<ExchangeId, ExchangeInput>,
    /// The reduce partitions the fragment's root exchange writes, when the
    /// task is a map task of that exchange; `None` lowers a root exchange
    /// as on one worker.
    pub output: Option<u32>,
    /// The runtime filters earlier stages computed for the fragment's
    /// scans, as `KeyFilter` v1 bytes (design P2-T177, cross-stage
    /// filters). A scan's filter without an entry stays pending and
    /// applies nothing. Each value is complete (merged from every task of
    /// the stage that collected it) and is published as the sole producer
    /// partition of the task's value service, so that service must count
    /// one producer for it (the default).
    pub filters: BTreeMap<FilterId, Vec<u8>>,
}

/// What one task reads from one exchange of another stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExchangeInput {
    /// The attempt of the stage that wrote it.
    pub stage_attempt: u32,
    /// The map partitions it was written by: `0..maps`.
    pub maps: u32,
    /// The reduce partitions this task reads.
    pub reduce: Vec<u32>,
}

impl TaskSpec {
    /// Partition 0 of 1, attempt 0, reading every scan whole: what a plan
    /// instantiated without a distributed scheduler runs as.
    #[must_use]
    pub fn whole() -> Self {
        Self {
            stage: 0,
            stage_attempt: 0,
            partition: 0,
            partitions: 1,
            attempt: 0,
            scans: BTreeMap::new(),
            exchanges: BTreeMap::new(),
            output: None,
            filters: BTreeMap::new(),
        }
    }
}

impl WorkerContext {
    /// A context reading `tables`, without spilling.
    pub fn new(tables: Arc<dyn TableProviderResolver>) -> Self {
        Self {
            tables,
            udfs: None,
            python: None,
            relations: None,
            spill: None,
            scopes: None,
            io: None,
            formats: Arc::new(FormatRegistry::new()),
            config: None,
            values: None,
            producer: Producer::default(),
            exchange: None,
            broadcasts: None,
            fuse_aggregates: true,
        }
    }

    /// The same context invoking session UDFs from `udfs`. A plan calling
    /// a UDF the context lacks fails with `KRISHIV_UDF_NOT_AVAILABLE`.
    #[must_use]
    pub fn with_udfs(mut self, udfs: Arc<UdfLibrary>) -> Self {
        self.udfs = Some(udfs);
        self
    }

    /// The same context resolving a scan's batch splits (the
    /// [`ScanSplits::Batches`] of its task) through `relations`: the
    /// session's relation store.
    #[must_use]
    pub fn with_relations(mut self, relations: Arc<krishiv_memory::RelationStore>) -> Self {
        self.relations = Some(relations);
        self
    }

    /// The same context spilling into `scope` (one per query).
    #[must_use]
    pub fn with_spill(mut self, scope: SpillScope) -> Self {
        self.spill = Some(scope);
        self
    }

    /// The same context spilling into a scope `scopes` creates for each
    /// instantiation (unless a fixed scope is set with
    /// [`Self::with_spill`]).
    #[must_use]
    pub fn with_spill_scopes(mut self, scopes: Arc<dyn SpillScopes>) -> Self {
        self.scopes = Some(scopes);
        self
    }

    /// The same context with the I/O context `io`: what a
    /// [`WriteFiles`](PhysNode::WriteFiles) node builds its file sink
    /// with.
    #[must_use]
    pub fn with_io(mut self, io: IoContext) -> Self {
        self.io = Some(io);
        self
    }

    /// The same context resolving a write's format name against `formats`.
    #[must_use]
    pub fn with_formats(mut self, formats: Arc<FormatRegistry>) -> Self {
        self.formats = formats;
        self
    }

    /// The same context parsing a write's format options against `config`
    /// (session fallbacks and strict mode).
    #[must_use]
    pub fn with_config(mut self, config: Arc<Config>) -> Self {
        self.config = Some(config);
        self
    }

    /// The same context publishing and reading the plan's values through
    /// `values`. A service serves one instantiation; reusing it is
    /// unsupported: a second plan with runtime filters fails to declare
    /// them, and a subquery-only plan silently reads the first one's values.
    #[must_use]
    pub fn with_values(mut self, values: Arc<dyn ValueService>) -> Self {
        self.values = Some(values);
        self
    }

    /// The same context publishing values as `producer` (partition 0,
    /// attempt 0 by default).
    #[must_use]
    pub fn with_producer(mut self, producer: Producer) -> Self {
        self.producer = producer;
        self
    }

    /// The same context writing and reading the exchanges between stages
    /// through `exchange`: what a task's root exchange and its
    /// [`PhysNode::ExchangeSource`]s instantiate with.
    #[must_use]
    pub fn with_exchange(mut self, exchange: Arc<dyn Exchange>) -> Self {
        self.exchange = Some(exchange);
        self
    }

    /// The same context building each broadcast hash join's table once per
    /// job on this worker, in `broadcasts`, and sharing it among the job's
    /// tasks here (design P2-T177 unit 3; see [`BroadcastTables`]).
    #[must_use]
    pub fn with_broadcasts(mut self, broadcasts: Arc<BroadcastTables>) -> Self {
        self.broadcasts = Some(broadcasts);
        self
    }

    /// The same context running a partial aggregate, its exchange and its
    /// final aggregate as written (`false`) instead of as one breaker: the
    /// operators a worker runs on each side of a cut exchange, for tests.
    #[must_use]
    pub fn with_fused_aggregates(mut self, fuse: bool) -> Self {
        self.fuse_aggregates = fuse;
        self
    }

    /// The fixed spill scope, if one is set ([`Self::with_spill`]).
    pub fn spill(&self) -> Option<&SpillScope> {
        self.spill.as_ref()
    }

    /// The scope an instantiation spills into: the fixed one, else a new
    /// one from the factory ([`Self::with_spill_scopes`]); `None` without
    /// spilling.
    ///
    /// # Errors
    /// When the factory cannot create the spill directory.
    pub fn spill_scope(&self) -> Result<Option<SpillScope>> {
        match (&self.spill, &self.scopes) {
            (Some(scope), _) => Ok(Some(scope.clone())),
            (None, Some(scopes)) => Ok(Some(scopes.scope()?)),
            (None, None) => Ok(None),
        }
    }

    /// The relation store, if batch splits resolve through one.
    pub fn relations(&self) -> Option<&Arc<krishiv_memory::RelationStore>> {
        self.relations.as_ref()
    }
}

impl fmt::Debug for WorkerContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerContext")
            .field("tables", &self.tables)
            .field("udfs", &self.udfs.is_some())
            .field("relations", &self.relations.is_some())
            .field("spill", &self.spill.is_some())
            .field("spill_scopes", &self.scopes.is_some())
            .field("io", &self.io.is_some())
            .field("formats", &self.formats.names())
            .field("config", &self.config.is_some())
            .field("values", &self.values.is_some())
            .field("producer", &self.producer)
            .field("exchange", &self.exchange.is_some())
            .field("broadcasts", &self.broadcasts.is_some())
            .field("fuse_aggregates", &self.fuse_aggregates)
            .finish_non_exhaustive()
    }
}
/// The executable graph of `task` of `plan` on the worker `worker`
/// describes. Each call creates its own operators and, unless `worker` sets
/// a value service, its own values: a plan can be instantiated any number
/// of times.
///
/// # Errors
/// The table providers' errors; `INTERNAL_ERROR` when the plan does not
/// match the worker's tables any more (a pushed filter honoured
/// differently, another result schema), names a runtime filter no scan
/// declares, or writes without an I/O context, formats or configuration.
pub fn instantiate_task(
    plan: &PhysicalPlan,
    worker: &WorkerContext,
    task: &TaskSpec,
) -> Result<Arc<dyn ExecNode>> {
    Ok(instantiate_with_schemas(plan, worker, task)?.0)
}

/// The columns of the input of each exchange of `plan` (shared plans
/// included), as `worker` instantiates it whole: what a fragmenter gives
/// the [`PhysNode::ExchangeSource`] it puts in an exchange's place. The
/// operators built are dropped unrun.
///
/// # Errors
/// As [`instantiate_task`].
pub fn exchange_schemas(
    plan: &PhysicalPlan,
    worker: &WorkerContext,
) -> Result<BTreeMap<ExchangeId, SchemaRef>> {
    // A fused aggregate never builds the exchange between its phases, which
    // a fragmenter cuts like any other.
    let written = worker.clone().with_fused_aggregates(false);
    Ok(instantiate_with_schemas(plan, &written, &TaskSpec::whole())?.1)
}

/// [`exchange_schemas`] from the plan alone: the planner's schema of every
/// exchange ([`ExchangePlan::schema`], T176 unit 1 review m4), with no
/// operator built. `None` when some exchange has none (a plan written
/// without them): instantiate it instead.
#[must_use]
pub fn planned_exchange_schemas(plan: &PhysicalPlan) -> Option<BTreeMap<ExchangeId, SchemaRef>> {
    let mut out = BTreeMap::new();
    let mut stack: Vec<&PhysNode> = plan.roots().collect();
    while let Some(n) = stack.pop() {
        if let PhysNode::Exchange(e) = n {
            out.insert(e.id, SchemaRef::clone(e.schema.as_ref()?));
        }
        stack.extend(n.children());
    }
    Some(out)
}

/// [`instantiate_task`], with the schema of each exchange's input.
fn instantiate_with_schemas(
    plan: &PhysicalPlan,
    worker: &WorkerContext,
    task: &TaskSpec,
) -> Result<(Arc<dyn ExecNode>, BTreeMap<ExchangeId, SchemaRef>)> {
    let run = || {
        let spill = worker.spill_scope()?;
        let mut env = EvalEnv::from_plan(plan.env());
        // The UDFs travel with the worker, never with the plan.
        env.udfs = worker.udfs.clone();
        let mut cx = Instantiation {
            worker,
            task: task.clone(),
            spill,
            env,
            values: worker
                .values
                .clone()
                .unwrap_or_else(|| Arc::new(LocalValues::new())),
            ids: node_ids(plan.roots()),
            shared: HashMap::new(),
            shared_plans: Arc::from(plan.shared()),
            loops: HashMap::new(),
            schemas: BTreeMap::new(),
            unit_weights: HashSet::new(),
            range_ordered: range_ordered_evals(plan.roots(), Some(plan.root())),
            salts: HashMap::new(),
            stages: HashMap::new(),
        };
        for root in plan.roots() {
            cx.declare_filters(root)?;
        }
        cx.take_filters()?;
        cx.build_shared()?;
        let built = match (task.output, plan.root()) {
            (Some(partitions), PhysNode::Exchange(e)) => cx.exchange_write(e, partitions)?,
            (Some(_), _) => {
                return Err(internal(
                    "a map task whose fragment does not end in an exchange".to_owned(),
                ));
            }
            (None, root) => cx.node(root)?,
        };
        if built.schema.fields() != plan.schema().fields() {
            return Err(internal(format!(
                "the plan instantiates to {:?} where it planned {:?}",
                built.schema.fields(),
                plan.schema().fields()
            )));
        }
        // A `range` streamed to the result through row-wise nodes only
        // returns its rows in slice order, as Spark collects partitions in
        // order: one driver keeps it (several would hand their batches to
        // the result in completion order).
        let node: Arc<dyn ExecNode> = if reads_range_in_order(plan.root()) {
            Arc::new(OrderedNode { inner: built.node })
        } else {
            built.node
        };
        Ok((node, cx.schemas))
    };
    // Instantiation recurses once per node, as planning does.
    krishiv_common::stack::on_big_stack(run)
        .ok_or_else(|| internal("cannot start an instantiation thread".to_owned()))?
}

/// The executable graph of `plan` on the worker `worker` describes, as
/// partition 0 of 1, attempt 0. Each call creates its own operators and,
/// unless `worker` sets a value service, its own values: a plan can be
/// instantiated any number of times.
///
/// # Errors
/// As [`instantiate_task`].
pub fn instantiate(plan: &PhysicalPlan, worker: &WorkerContext) -> Result<Arc<dyn ExecNode>> {
    instantiate_task(plan, worker, &TaskSpec::whole())
}

struct Instantiation<'a> {
    worker: &'a WorkerContext,
    task: TaskSpec,
    /// The query's spill scope on this worker.
    spill: Option<SpillScope>,
    env: EvalEnv,
    /// The plan's values: runtime filters, subquery results, broadcasts.
    values: Arc<dyn ValueService>,
    /// Each node's pre-order index, by address (the plan is borrowed for
    /// the whole instantiation, so addresses are stable).
    ids: HashMap<usize, u32>,
    /// The shared plans built so far (a shared plan reads only earlier ones).
    shared: HashMap<super::SharedId, Arc<SharedProducer>>,
    /// The plan's shared plans, inner first (a recursive round builds them
    /// again).
    shared_plans: Arc<[SharedPlan]>,
    /// The table each enclosing recursive CTE's [`PhysNode::LoopRef`]s read,
    /// by loop id: the previous round, for a round being instantiated.
    loops: HashMap<u32, Arc<Output>>,
    /// The columns of each exchange's input built so far.
    schemas: BTreeMap<ExchangeId, SchemaRef>,
    /// Aggregate nodes (by address) that refuse weights other than 1: the
    /// partial phase of a written-out `DISTINCT` aggregation.
    unit_weights: HashSet<usize>,
    /// `PythonEval` nodes (by address) keeping a `range`'s order (`order.rs`).
    range_ordered: HashSet<usize>,
    /// The first unseeded-random salt of each shared plan, as the
    /// instantiation that built it first numbered it (`detached.rs`).
    salts: HashMap<super::SharedId, u64>,
    /// Each scan's runtime filters, for the probe it may hand its row stage.
    stages: HashMap<ScanId, RuntimeFilters>,
}

/// A shared plan's body and the sink that keeps its rows; every reader's pipeline depends on this one.
struct SharedProducer {
    node: Arc<dyn ExecNode>,
    sink: Arc<SharedSink>,
    schema: SchemaRef,
    /// The body ends in a sort: readers keep the list's order.
    ordered: bool,
}

/// One `SharedRead`: its own source over the shared plan's list.
struct SharedReadNode {
    id: u64,
    producer: Arc<SharedProducer>,
    source: Arc<SharedSource>,
}

impl ExecNode for SharedReadNode {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        let sink = b.shared_pipeline(
            self.id,
            Arc::clone(&self.producer.sink) as Arc<dyn Sink>,
            Arc::clone(&self.producer.node),
        )?;
        b.set_breaker_source(Arc::clone(&self.source) as Arc<dyn Source>, sink)?;
        if self.producer.ordered {
            // One driver reads the list in the order it was produced.
            b.preserve_order();
        }
        Ok(())
    }
}

/// The pre-order index of every node under `roots`, numbered on from one
/// root to the next (children in
/// [`PhysNode::children`] order), by address: `EXPLAIN ANALYZE`'s node ids.
pub(crate) fn node_ids<'a>(roots: impl IntoIterator<Item = &'a PhysNode>) -> HashMap<usize, u32> {
    let mut ids = HashMap::new();
    let mut next = 0_u32;
    for root in roots {
        let mut stack = vec![root];
        while let Some(n) = stack.pop() {
            ids.insert(std::ptr::from_ref(n).addr(), next);
            next = next.saturating_add(1);
            stack.extend(n.children().into_iter().rev());
        }
    }
    ids
}

/// The final phase reading the output of `partial` (its keys, then each
/// aggregate's state columns): grouped by those keys, each aggregate merging
/// its states. [`fused_aggregate`] recognizes the pair.
pub(crate) fn final_of(partial: &AggregatePlan) -> Result<AggregatePlan> {
    let k = partial.group_by.len();
    let mut at = k;
    let mut aggregates = Vec::with_capacity(partial.aggregates.len());
    for a in &partial.aggregates {
        let m = a.merging(at);
        at += m.args.len();
        aggregates.push(m);
    }
    if partial.output != AggregateOutput::Partial || partial.distinct.iter().any(|d| *d) {
        return Err(internal(
            "a final phase over a plan that is not a partial one".to_owned(),
        ));
    }
    Ok(AggregatePlan {
        group_by: (0..k).collect(),
        distinct: vec![false; aggregates.len()],
        aggregates,
        output: AggregateOutput::Final,
    })
}

/// `Final ← Exchange(Hash | Single) ← Partial` as one aggregation (design
/// P2-T174 §2.2): the partial node, the node below it, and the plan of the
/// single breaker over that node, whose radix partitions stand for the
/// exchange's and whose finalize merge is its reader side. `None` for any
/// other shape: `aggregate` is not exactly the final phase
/// [`final_of`] plans over the partial.
pub(crate) fn fused_aggregate<'p>(
    aggregate: &AggregatePlan,
    input: &'p PhysNode,
) -> Option<(&'p PhysNode, &'p PhysNode, AggregatePlan)> {
    let PhysNode::Exchange(e) = input else {
        return None;
    };
    if !matches!(
        e.partitioning,
        Partitioning::Hash { .. } | Partitioning::Single
    ) {
        return None;
    }
    let partial = e.input.as_ref();
    let PhysNode::Aggregate {
        input: below,
        aggregate: p,
    } = partial
    else {
        return None;
    };
    if p.output != AggregateOutput::Partial || final_of(p).ok().as_ref() != Some(aggregate) {
        return None;
    }
    Some((
        partial,
        below.as_ref(),
        AggregatePlan {
            output: AggregateOutput::Final,
            ..p.clone()
        },
    ))
}

/// The phases of `single`, an aggregation with `DISTINCT` aggregates over raw
/// input, around an exchange (Spark's single-distinct plan, as
/// [`distinct_plan`] runs it in one place): a partial phase grouped by the
/// keys and the distinct arguments computing the other aggregates' states; a
/// partial merge after the exchange on the same columns, which removes the
/// pairs several tasks sent; and the final phase grouped by the keys, which
/// aggregates each distinct value once (as plain aggregates over the now
/// distinct argument columns) and merges the other states. `None` where
/// [`distinct_plan`] has no such plan: no distinct aggregate, distinct
/// aggregates over different arguments or with a filter, or input that is not
/// raw.
pub(crate) fn distinct_phases(single: &AggregatePlan) -> Option<[AggregatePlan; 3]> {
    if single.output != AggregateOutput::Final
        || single.distinct.len() != single.aggregates.len()
        || single.aggregates.iter().any(|a| a.input != AggInput::Raw)
    {
        return None;
    }
    let mut dargs: Option<&[usize]> = None;
    for (a, &d) in single.aggregates.iter().zip(&single.distinct) {
        if d && (a.mask.is_some() || *dargs.get_or_insert(&a.args) != a.args.as_slice()) {
            return None;
        }
    }
    let dargs = dargs?;
    let k = single.group_by.len();
    let width = k + dargs.len();
    let others: Vec<AggregateExpr> = single
        .aggregates
        .iter()
        .zip(&single.distinct)
        .filter(|(_, d)| !**d)
        .map(|(a, _)| a.clone())
        .collect();
    let first = AggregatePlan {
        group_by: single.group_by.iter().chain(dargs).copied().collect(),
        distinct: vec![false; others.len()],
        aggregates: others,
        output: AggregateOutput::Partial,
    };
    let mut at = width;
    let mut merged = Vec::with_capacity(first.aggregates.len());
    for a in &first.aggregates {
        let m = a.merging(at);
        at += m.args.len();
        merged.push(m);
    }
    let merge = AggregatePlan {
        group_by: (0..width).collect(),
        distinct: vec![false; merged.len()],
        aggregates: merged.clone(),
        output: AggregateOutput::Partial,
    };
    let mut merged = merged.into_iter();
    let mut aggregates = Vec::with_capacity(single.aggregates.len());
    for (a, &d) in single.aggregates.iter().zip(&single.distinct) {
        aggregates.push(if d {
            AggregateExpr {
                args: (k..width).collect(),
                ..a.clone()
            }
        } else {
            merged.next()?
        });
    }
    let last = AggregatePlan {
        group_by: (0..k).collect(),
        distinct: vec![false; aggregates.len()],
        aggregates,
        output: AggregateOutput::Final,
    };
    Some([first, merge, last])
}

/// `Final ← PartialMerge ← Exchange(Hash | Single) ← Partial`, the phases
/// [`distinct_phases`] plans, as the one aggregation they came from: the node
/// the partial node, the node below it and that aggregation. `None` for any
/// other shape.
pub(crate) fn fused_distinct<'p>(
    last: &AggregatePlan,
    input: &'p PhysNode,
) -> Option<(&'p PhysNode, &'p PhysNode, AggregatePlan)> {
    let PhysNode::Aggregate {
        input: exchange,
        aggregate: merge,
    } = input
    else {
        return None;
    };
    let PhysNode::Exchange(e) = exchange.as_ref() else {
        return None;
    };
    if !matches!(
        e.partitioning,
        Partitioning::Hash { .. } | Partitioning::Single
    ) {
        return None;
    }
    let partial = e.input.as_ref();
    let PhysNode::Aggregate {
        input: below,
        aggregate: first,
    } = partial
    else {
        return None;
    };
    let k = last.group_by.len();
    let dargs = first.group_by.get(k..)?;
    // The final phase's raw aggregates are the distinct ones; the others
    // merge the partial phase's states, in order.
    let mut others = first.aggregates.iter();
    let mut aggregates = Vec::with_capacity(last.aggregates.len());
    let mut distinct = Vec::with_capacity(last.aggregates.len());
    for a in &last.aggregates {
        if a.input == AggInput::Raw {
            aggregates.push(AggregateExpr {
                args: dargs.to_vec(),
                ..a.clone()
            });
            distinct.push(true);
        } else {
            aggregates.push(others.next()?.clone());
            distinct.push(false);
        }
    }
    let single = AggregatePlan {
        group_by: first.group_by.get(..k)?.to_vec(),
        aggregates,
        distinct,
        output: AggregateOutput::Final,
    };
    let phases = distinct_phases(&single)?;
    (phases[0] == *first && phases[1] == *merge && phases[2] == *last).then_some((
        partial,
        below.as_ref(),
        single,
    ))
}

/// A fused aggregation (see [`fused_aggregate`]): one [`HashAggregate`]
/// whose sink side is tagged with the partial node's id.
struct FusedAggregateNode {
    partial: Option<u32>,
    input: Arc<dyn ExecNode>,
    agg: Arc<HashAggregate>,
}

impl ExecNode for FusedAggregateNode {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        if let Some(id) = self.partial {
            b.enter_node(id);
        }
        let sink = Arc::clone(&self.agg) as Arc<dyn Sink>;
        let id = b.child_pipeline(sink, Arc::clone(&self.input));
        if self.partial.is_some() {
            b.exit_node();
        }
        b.set_breaker_source(Arc::clone(&self.agg) as Arc<dyn Source>, id?)
            .map(drop)
    }
}

/// A plan node whose operators report under its id.
struct Tagged {
    id: u32,
    inner: Arc<dyn ExecNode>,
}

impl ExecNode for Tagged {
    fn build_pipelines(self: Arc<Self>, b: &mut PipelineBuilder) -> Result<()> {
        b.enter_node(self.id);
        let built = Arc::clone(&self.inner).build_pipelines(b);
        b.exit_node();
        built
    }
}

struct Built {
    node: Arc<dyn ExecNode>,
    schema: SchemaRef,
}

impl Instantiation<'_> {
    /// Builds the shared plans, inner first: a shared plan reads only earlier
    /// ones.
    fn build_shared(&mut self) -> Result<()> {
        let plans = Arc::clone(&self.shared_plans);
        // A shared plan that reads a loop's table (or a shared plan that
        // does) can only be built where that table is bound: in a round of
        // the loop. Elsewhere it is left out; nothing reads it there.
        // Each deferred plan keeps the loops it still waits for, so a plan
        // that reads it is deferred only for those the plan does not bind
        // itself (a round-local plan read inside a loop is built in its
        // rounds, not at the enclosing producer).
        let mut deferred: HashMap<_, Vec<_>> = HashMap::new();
        for sp in plans.iter() {
            let mut stack = vec![&sp.plan];
            // Loops this plan defines bind their own references.
            let mut local = HashSet::new();
            let mut refs = Vec::new();
            while let Some(n) = stack.pop() {
                match n {
                    PhysNode::UnionLoop { id, .. } => {
                        local.insert(*id);
                    }
                    PhysNode::LoopRef { id, .. } => refs.push(*id),
                    PhysNode::SharedRead { id, .. } => {
                        if let Some(waits) = deferred.get(id) {
                            refs.extend(waits.iter().copied());
                        }
                    }
                    _ => {}
                }
                stack.extend(n.children());
            }
            refs.retain(|id| !self.loops.contains_key(id) && !local.contains(id));
            if !refs.is_empty() {
                refs.sort_unstable();
                refs.dedup();
                deferred.insert(sp.id, refs);
                continue;
            }
            let body = self.shared_body(sp)?;
            let producer = Arc::new(SharedProducer {
                sink: SharedSink::new(Arc::clone(&body.schema), self.spill.clone()),
                schema: Arc::clone(&body.schema),
                node: body.node,
                ordered: sp.ordered,
            });
            self.shared.insert(sp.id, producer);
        }
        Ok(())
    }

    /// Declares every runtime filter a scan declares to the value service.
    fn declare_filters(&self, n: &PhysNode) -> Result<()> {
        let mut stack = vec![n];
        while let Some(n) = stack.pop() {
            if let PhysNode::Scan(s) = n {
                for f in &s.runtime_filters {
                    self.values
                        .declare_filter(f.id, f.column.clone(), f.key_type.clone())?;
                }
            }
            stack.extend(n.children());
        }
        Ok(())
    }

    /// Builds `n`, tagged with its pre-order index so the operators it
    /// adds report under it.
    fn node(&mut self, n: &PhysNode) -> Result<Built> {
        let built = self.untagged(n)?;
        Ok(self.tag(n, built))
    }

    /// `built` (the operators of `n`) under `n`'s `EXPLAIN ANALYZE` id.
    fn tag(&self, n: &PhysNode, built: Built) -> Built {
        match self.ids.get(&std::ptr::from_ref(n).addr()) {
            Some(&id) => Built {
                schema: built.schema,
                node: Arc::new(Tagged {
                    id,
                    inner: built.node,
                }),
            },
            None => built,
        }
    }

    fn untagged(&mut self, n: &PhysNode) -> Result<Built> {
        match n {
            PhysNode::PythonEval { .. } => self.python_eval(n),
            PhysNode::Scan(s) => self.scan(s),
            PhysNode::Range(r) => {
                let source = Arc::new(crate::operators::Range::new(
                    r.start,
                    r.end,
                    r.step,
                    r.num_slices,
                ));
                Ok(Built {
                    schema: source.schema(),
                    node: Arc::new(SourceNode(source)),
                })
            }
            PhysNode::SharedRead { id, .. } => {
                let producer = self.shared.get(id).cloned().ok_or_else(|| {
                    internal(format!("shared plan {} is read before it is built", id.0))
                })?;
                Ok(Built {
                    schema: Arc::clone(&producer.schema),
                    node: Arc::new(SharedReadNode {
                        id: id.0,
                        source: SharedSource::new(Arc::clone(&producer.schema)),
                        producer,
                    }),
                })
            }
            PhysNode::LoopRef { id, .. } => {
                let table = self.loops.get(id).cloned().ok_or_else(|| {
                    internal(format!(
                        "recursive reference {id} is read outside its round"
                    ))
                })?;
                let source = SharedSource::over(table);
                Ok(Built {
                    schema: source.schema(),
                    node: Arc::new(SourceNode(source)),
                })
            }
            PhysNode::UnionLoop {
                id,
                anchor,
                recursive,
                nullable,
                limits,
                ..
            } => {
                // The outer build of the anchor below gives the loop its schema only: every run
                // (the anchor's too) is instantiated again from this plan (`detached.rs`).
                let anchor_plan = Arc::new((**anchor).clone());
                let anchor = self.node(anchor)?;
                // The loop's rows are the anchor's, with the nullability the
                // plan fixed (the recursive term may produce nulls the
                // anchor does not).
                let fields: Vec<arrow_schema::FieldRef> = anchor
                    .schema
                    .fields()
                    .iter()
                    .zip(nullable)
                    .map(|(f, n)| Arc::new(f.as_ref().clone().with_nullable(*n)))
                    .collect();
                if fields.len() != anchor.schema.fields().len() {
                    return Err(internal(
                        "a recursive CTE whose anchor has another width".to_owned(),
                    ));
                }
                let schema: SchemaRef = Arc::new(arrow_schema::Schema::new(fields));
                let detached = Arc::new(Detached {
                    worker: self.worker.clone(),
                    task: self.task.clone(),
                    spill: self.spill.clone(),
                    env: self.env.clone(),
                    shared_plans: Arc::clone(&self.shared_plans),
                    loops: self.loops.clone(),
                    salts: self.salts.clone(),
                });
                let source = UnionLoopSource::new(
                    *id,
                    Arc::clone(&schema),
                    anchor_plan,
                    (**recursive).clone(),
                    *limits,
                    detached,
                );
                Ok(Built {
                    schema,
                    node: Arc::new(SourceNode(source)),
                })
            }
            PhysNode::OneRow => {
                let source = one_row()?;
                Ok(Built {
                    schema: source.schema(),
                    node: Arc::new(SourceNode(source)),
                })
            }
            PhysNode::Filter {
                input,
                layout,
                predicate,
                subqueries,
                row_stage: Some(stage),
            } if subqueries.is_empty() => {
                self.filter_with_row_stage(input, layout, predicate, stage)
            }
            PhysNode::Filter {
                input,
                layout,
                predicate,
                subqueries,
                ..
            } => self.with_subqueries(input, subqueries, |input, v| {
                filter_op(input, layout, predicate, v)
            }),
            PhysNode::Project {
                input,
                layout,
                items,
                nullable,
                subqueries,
            } => self.with_subqueries(input, subqueries, |input, v| {
                Ok(project_op(input, layout, items, nullable.as_deref(), v)? as Arc<dyn Operator>)
            }),
            PhysNode::ReplicateRows {
                input,
                layout,
                count,
                subqueries,
            } => self.with_subqueries(input, subqueries, |input, v| {
                replicate_op(input, layout, count, v)
            }),
            PhysNode::Generate {
                input,
                layout,
                kind,
                outer,
                arg,
                subqueries,
            } => self.with_subqueries(input, subqueries, |input, v| {
                generate_op(input, layout, *kind, *outer, arg, v)
            }),
            PhysNode::Expand {
                input,
                layout,
                projections,
                nullable,
                subqueries,
            } => self.with_subqueries(input, subqueries, |input, v| {
                expand_op(input, layout, projections, nullable, v)
            }),
            PhysNode::Limit {
                input,
                offset,
                limit,
            } => {
                let ordered = reads_range_in_order(input);
                let input = self.node(input)?;
                let op = crate::operators::Limit::with_offset(
                    Arc::clone(&input.schema),
                    *offset,
                    *limit,
                );
                let built = operator(input, op);
                // Which rows a parallel limit passes follows the batch
                // order; a `range` has a defined order, so one driver keeps it.
                Ok(if ordered {
                    Built {
                        schema: built.schema,
                        node: Arc::new(OrderedNode { inner: built.node }),
                    }
                } else {
                    built
                })
            }
            PhysNode::Sort { input, keys } => {
                let buckets = hash_buckets(input, keys);
                let input = self.node(input)?;
                match buckets {
                    // A hash exchange below lowers into the sort's buckets.
                    Some(columns) => {
                        let sort = Sort::new(Arc::clone(&input.schema), keys, self.spill.clone())?
                            .with_hash_buckets(columns)?;
                        Ok(bucketed(input, Arc::new(sort)))
                    }
                    None => {
                        let sort = sort_op(&input.schema, keys, self.spill.clone())?;
                        Ok(breaker(input, Arc::clone(&sort) as Arc<dyn Sink>, sort))
                    }
                }
            }
            PhysNode::Window { input, window } => {
                let input = self.node(input)?;
                let op = crate::window::Window::with_env(
                    Arc::clone(&input.schema),
                    window.clone(),
                    &self.env,
                    self.spill.clone(),
                )?;
                Ok(operator(input, Arc::new(op)))
            }
            PhysNode::WindowGroupLimit { input, limit } => {
                // Directly over its sort, the sort also prunes each sorted
                // batch (the limit's `Partial` mode).
                let input = match input.as_ref() {
                    sort_node @ PhysNode::Sort { input, keys } if limit.sorted_by(keys) => {
                        let buckets = hash_buckets(input, keys);
                        let input = self.node(input)?;
                        let sort = Sort::new(Arc::clone(&input.schema), keys, self.spill.clone())?
                            .with_group_limit(keys, limit)?;
                        let built = match buckets {
                            // As a plain sort: a hash exchange below lowers
                            // into the sort's buckets.
                            Some(columns) => {
                                bucketed(input, Arc::new(sort.with_hash_buckets(columns)?))
                            }
                            None => {
                                let sort = Arc::new(sort);
                                breaker(input, Arc::clone(&sort) as Arc<dyn Sink>, sort)
                            }
                        };
                        // Built here, not through `node`: tag it as `node`
                        // does, for `EXPLAIN ANALYZE`.
                        self.tag(sort_node, built)
                    }
                    other => self.node(other)?,
                };
                let op = group_limit_op(&input.schema, limit)?;
                Ok(operator(input, op))
            }
            PhysNode::LateFetch {
                input,
                scan,
                file_col,
                row_col,
                output,
            } => {
                let input = self.node(input)?;
                // Not a scan of its own: the row id names each file, so the
                // task's splits don't apply.
                let fetch = Arc::new(LateFetch::new(
                    &input.schema,
                    self.worker.tables.provider(&scan.scan)?.scan()?,
                    scan.projection.clone(),
                    *file_col,
                    *row_col,
                    output.clone(),
                )?);
                Ok(breaker(input, Arc::clone(&fetch) as Arc<dyn Sink>, fetch))
            }
            PhysNode::TopN {
                input,
                keys,
                limit,
                bound,
            } => {
                let input = self.node(input)?;
                let top = top_n(&input.schema, keys, *limit)?;
                let top = Arc::new(match bound {
                    Some(id) => top.with_bound(Arc::clone(&self.values), *id),
                    None => top,
                });
                Ok(breaker(input, Arc::clone(&top) as Arc<dyn Sink>, top))
            }
            PhysNode::Aggregate { input, aggregate } => {
                if self.worker.fuse_aggregates
                    && let Some((partial, below, fused)) = fused_aggregate(aggregate, input)
                {
                    return self.fused(partial, below, &fused);
                }
                if let Some((partial, below, single)) = fused_distinct(aggregate, input) {
                    if self.worker.fuse_aggregates {
                        let input = self.node(below)?;
                        let spec =
                            aggregate_spec(&input.schema, &single, &self.env, self.spill.clone());
                        let schema = spec.output_schema()?;
                        let node = aggregate_node(input.node, spec, &single.distinct)?;
                        return Ok(Built { node, schema });
                    }
                    // Written out, the partial phase is where a row's weight
                    // is lost: it refuses weights other than 1, as the fused
                    // plan's first phase does.
                    self.unit_weights.insert(std::ptr::from_ref(partial).addr());
                }
                let unit = self.unit_weights.contains(&std::ptr::from_ref(n).addr());
                let input = self.node(input)?;
                let mut spec =
                    aggregate_spec(&input.schema, aggregate, &self.env, self.spill.clone());
                spec.require_unit_weights = unit;
                let schema = spec.output_schema()?;
                let node = aggregate_node(input.node, spec, &aggregate.distinct)?;
                Ok(Built { node, schema })
            }
            PhysNode::HashJoin(j) => self.hash_join(j),
            PhysNode::SortMergeJoin(j) => self.merge_join(j),
            PhysNode::NestedLoopJoin(j) => {
                let probe = self.node(&j.probe)?;
                let build = self.node(&j.build)?;
                let join = nested_loop_join(
                    &probe.schema,
                    &build.schema,
                    j,
                    &self.env,
                    self.spill.clone(),
                )?;
                Ok(Built {
                    schema: join.schema(),
                    node: Arc::new(NestedLoopJoinNode {
                        join,
                        build: build.node,
                        probe: probe.node,
                    }),
                })
            }
            PhysNode::Union { inputs } => {
                let mut nodes = Vec::with_capacity(inputs.len());
                let mut schema: Option<SchemaRef> = None;
                for i in inputs {
                    let built = self.node(i)?;
                    match &schema {
                        None => schema = Some(built.schema),
                        Some(s) if s.fields() != built.schema.fields() => {
                            return Err(internal(
                                "union inputs of different physical types".to_owned(),
                            ));
                        }
                        Some(_) => {}
                    }
                    nodes.push(built.node);
                }
                Ok(Built {
                    schema: schema.ok_or_else(|| internal("a union without inputs".to_owned()))?,
                    node: Arc::new(FanInNode { inputs: nodes }),
                })
            }
            PhysNode::WriteFiles { input, write } => self.write_files(input, write),
            // See the module docs: the operators around an exchange
            // implement it on one worker.
            PhysNode::Exchange(e) if e.partitioning == Partitioning::Broadcast => self.broadcast(e),
            PhysNode::Exchange(e) => {
                let input = self.node(&e.input)?;
                // Unused on one worker: checked in debug builds (every test).
                if cfg!(debug_assertions) {
                    planned_schema_holds(e, &input.schema)?;
                }
                self.schemas.insert(e.id, Arc::clone(&input.schema));
                Ok(input)
            }
            PhysNode::ExchangeSource {
                id,
                partitioning,
                schema,
            } => self.exchange_source(*id, partitioning, schema),
        }
    }

    fn scan(&mut self, s: &ScanNode) -> Result<Built> {
        let factory = self.worker.tables.provider(&s.scan)?.scan()?;
        let factory = self.splits(s.id, factory)?;
        if s.row_id && !factory.supports_take() {
            return Err(internal(format!(
                "scan {} has a row id its table can't read back",
                s.id.0
            )));
        }
        let table = factory.table_schema();
        let mut filters = Vec::new();
        if let Some(f) = &s.filter {
            let pushed = scan_filter(&table, &f.layout, &f.predicate, &s.scan, &self.env)?;
            let exact = factory.filter_pushdown(&pushed) == FilterPushdown::Exact;
            if exact != f.exact {
                return Err(internal(format!(
                    "scan {} honours its pushed filter differently than when it was planned",
                    s.id.0
                )));
            }
            filters.push(pushed);
        }
        let dynamic = RuntimeFilters::default();
        for f in &s.runtime_filters {
            dynamic.push(self.values.filter(f.id)?);
        }
        if !s.runtime_filters.is_empty() {
            dynamic.set_settings(s.row_stage);
            self.stages.insert(s.id, dynamic.clone());
        }
        let source = factory.build(ScanPushdowns {
            projection: Some(s.projection.clone()),
            filters,
            dynamic,
            row_id: s.row_id,
            metadata: crate::operators::scan_metadata(&s.scan),
            ..ScanPushdowns::default()
        })?;
        Ok(Built {
            schema: source.schema(),
            node: Arc::new(SourceNode(source)),
        })
    }

    /// The scan factory with this task's splits applied: the listed files
    /// and partitions for a file split (reads stay pinned to the listed
    /// versions), or the batch range of the resolved relation for a batch
    /// split. A scan the task names nothing for builds exactly as today.
    fn splits(&self, id: ScanId, factory: Arc<dyn ScanPushdown>) -> Result<Arc<dyn ScanPushdown>> {
        let Some(splits) = self.task.scans.get(&id) else {
            return Ok(factory);
        };
        match splits {
            ScanSplits::Files { .. } => {
                let partitions = partitions_of(splits)?;
                factory.with_file_splits(files_of(splits)?, partitions)
            }
            ScanSplits::Batches { relation, range } => {
                let Some(store) = &self.worker.relations else {
                    return Err(internal(
                        "a batch split without a relation store".to_owned(),
                    ));
                };
                let batches = store.resolve(&relation.hash, range.clone())?;
                factory.with_memory_batches(batches)
            }
        }
    }

    /// The file write of `write` over `input`, as task `self.task` runs
    /// it: the sink is built here from the plan with the worker's own I/O
    /// context, so the same plan bytes run on every worker. The task
    /// number in file names is the partition index; attempt 0 names files
    /// exactly as before, and attempt `k > 0` appends `_<k>` to the job-id
    /// segment, so a late attempt can never overwrite a committed file.
    fn write_files(&mut self, input: &PhysNode, write: &Option<WritePlan>) -> Result<Built> {
        let Some(write) = write else {
            return Err(internal(
                "a write without a write plan (written by a writer before the write_plan feature)"
                    .to_owned(),
            ));
        };
        let io = self
            .worker
            .io
            .clone()
            .ok_or_else(|| internal("a write without an I/O context".to_owned()))?;
        let format = match self.worker.formats.get(&write.format) {
            Ok(format) => format,
            Err(e) => {
                if is_spark_file_source(&write.format) {
                    return Err(KrishivUnsupportedFeature {
                        feature: format!("writing the {} format", write.format),
                    }
                    .into());
                }
                return Err(e);
            }
        };
        let Some(config) = self.worker.config.as_deref() else {
            return Err(internal("a write without a configuration".to_owned()));
        };
        let (options, _) = format.options().parse(&write.options, config)?;
        let input = self.node(input)?;
        let mut fields: Vec<Arc<Field>> = input.schema.fields().to_vec();
        for (column, spark_type) in &write.column_types {
            let f = fields.get_mut(*column).ok_or_else(|| {
                internal(format!(
                    "write column type of column {column}, out of range"
                ))
            })?;
            *f = Arc::new(spark_type.to_arrow_field_with_storage(
                f.name(),
                f.data_type().clone(),
                f.is_nullable(),
            )?);
        }
        let written = Arc::new(Schema::new_with_metadata(
            fields,
            input.schema.metadata().clone(),
        ));
        let sink = Arc::new(FileWriteSink::try_new(
            Arc::clone(&format),
            io,
            written,
            options,
            WriteOptions {
                store: StoreId::new(&write.store),
                dir: Path::from(write.dir.as_str()),
                task: self.task.partition,
                job_uuid: effective_job_uuid(&write.job_uuid, self.task.attempt),
                max_records_per_file: write.max_records_per_file,
                target_file_bytes: write.target_file_bytes,
                partition_by: write.partition_by.clone(),
                zone: self.env.zone,
                zone_name: self.env.zone_name.clone(),
                partition_names: write
                    .partition_by
                    .iter()
                    .map(|&c| {
                        input
                            .schema
                            .fields()
                            .get(c)
                            .map(|f| f.name().clone())
                            .ok_or_else(|| {
                                internal(format!("partition column {c} is out of range"))
                            })
                    })
                    .collect::<Result<_>>()?,
            },
        )?);
        Ok(Built {
            schema: Source::schema(sink.as_ref()),
            node: Arc::new(FileWrite {
                input: input.node,
                sink,
            }),
        })
    }

    /// `op` over `input`, after the pipelines of the subqueries its
    /// expressions read.
    fn with_subqueries(
        &mut self,
        input: &PhysNode,
        subqueries: &[SubqueryPlan],
        op: impl FnOnce(&SchemaRef, &Values<'_>) -> Result<Arc<dyn Operator>>,
    ) -> Result<Built> {
        let input = self.node(input)?;
        let mut plans = Vec::with_capacity(subqueries.len());
        let mut values = Vec::with_capacity(subqueries.len());
        for s in subqueries {
            let plan = self.node(&s.plan)?;
            let sink = SubquerySink::new(
                s.id,
                s.kind,
                s.data_type.clone(),
                Arc::clone(&self.values),
                self.worker.producer,
            );
            plans.push((plan.node, sink));
            values.push((
                s.id.attr(),
                Arc::new(SubqueryValue {
                    kind: s.kind,
                    value: self.values.subscribe(ValueId::Subquery(s.id))?,
                    data_type: s.data_type.clone(),
                }) as Arc<dyn PhysExpr>,
            ));
        }
        let op = op(
            &input.schema,
            &Values {
                env: &self.env,
                subqueries: &values,
            },
        )?;
        if plans.is_empty() {
            return Ok(operator(input, op));
        }
        Ok(Built {
            schema: op.schema(),
            node: Arc::new(WithSubqueries {
                subqueries: plans,
                input: input.node,
                op,
            }),
        })
    }

    /// The aggregation `fused` over `below` as one breaker (see
    /// [`fused_aggregate`]): its sink side reports under the partial node
    /// `partial`, its source side under the final one.
    fn fused(
        &mut self,
        partial: &PhysNode,
        below: &PhysNode,
        fused: &AggregatePlan,
    ) -> Result<Built> {
        let input = self.node(below)?;
        let spec = aggregate_spec(&input.schema, fused, &self.env, self.spill.clone());
        let schema = spec.output_schema()?;
        let node = Arc::new(FusedAggregateNode {
            partial: self.ids.get(&std::ptr::from_ref(partial).addr()).copied(),
            input: input.node,
            agg: HashAggregate::fused(spec)?,
        });
        Ok(Built { node, schema })
    }

    fn merge_join(&mut self, j: &SortMergeJoinPlan) -> Result<Built> {
        let probe = self.node(&j.probe)?;
        let build = self.node(&j.build)?;
        let join = sort_merge_join(
            &probe.schema,
            &build.schema,
            j,
            &self.env,
            self.spill.clone(),
        )?;
        Ok(Built {
            schema: join.schema(),
            node: Arc::new(SortMergeJoinNode {
                join: Arc::new(join),
                probe: probe.node,
                build: build.node,
            }),
        })
    }

    fn hash_join(&mut self, j: &HashJoinPlan) -> Result<Built> {
        if let Some(shared) = self.shared_broadcast_join(j)? {
            return Ok(shared);
        }
        let probe = self.node(&j.probe)?;
        let build = self.node(&j.build)?;
        let join = hash_join(
            &probe.schema,
            &build.schema,
            j,
            &self.env,
            self.spill.clone(),
            &self.values,
            self.worker.producer,
        )?;
        let join = self.hand_row_stage(join, j)?;
        Ok(Built {
            schema: join.schema(),
            node: Arc::new(HashJoinNode {
                join,
                build: build.node,
                probe: probe.node,
            }),
        })
    }
}

fn operator(input: Built, op: Arc<dyn Operator>) -> Built {
    Built {
        schema: op.schema(),
        node: Arc::new(OperatorNode {
            input: input.node,
            op,
        }),
    }
}

/// The hash keys of the exchange below a sort by those keys first (the
/// sort below a window with partitioning): the sort's buckets, which the
/// exchange lowers into on one worker (design P2-T055 4a).
fn hash_buckets<'p>(input: &'p PhysNode, keys: &[SortKey]) -> Option<&'p [usize]> {
    // A reduce task's sort reads a hash exchange's partition: the buckets
    // split it again by the sort's own table hash (P2-T176 unit 1).
    let partitioning = match input {
        PhysNode::Exchange(e) => &e.partitioning,
        PhysNode::ExchangeSource { partitioning, .. } => partitioning,
        _ => return None,
    };
    let Partitioning::Hash { keys: hash, .. } = partitioning else {
        return None;
    };
    let mut lead: Vec<usize> = keys.get(..hash.len())?.iter().map(|k| k.column).collect();
    let mut want = hash.clone();
    lead.sort_unstable();
    want.sort_unstable();
    (!hash.is_empty() && lead == want).then_some(hash.as_slice())
}

/// A bucketed sort's breaker: its buckets are read by several drivers, so
/// the order holds within each bucket only.
fn bucketed(input: Built, sort: Arc<Sort>) -> Built {
    Built {
        schema: Source::schema(&*sort),
        node: Arc::new(BreakerNode {
            input: input.node,
            sink: Arc::clone(&sort) as Arc<dyn Sink>,
            source: sort,
            ordered: false,
        }),
    }
}

/// A breaker keeping its output order (sorts).
fn breaker(input: Built, sink: Arc<dyn Sink>, source: Arc<dyn Source>) -> Built {
    Built {
        schema: source.schema(),
        node: Arc::new(BreakerNode {
            input: input.node,
            sink,
            source,
            ordered: true,
        }),
    }
}

/// What expressions are compiled with: the expression environment and the
/// values of the subqueries they read, by [`SubqueryId::attr`](super::SubqueryId::attr).
pub(crate) struct Values<'a> {
    pub(crate) env: &'a EvalEnv,
    pub(crate) subqueries: &'a [(AttrId, Arc<dyn PhysExpr>)],
}

/// `exprs` compiled over an input of `schema` whose columns are the
/// attributes `layout`.
pub(crate) fn compile_all(
    exprs: &[&Expr],
    layout: &[AttrId],
    schema: &SchemaRef,
    v: &Values<'_>,
) -> Result<Vec<Arc<dyn PhysExpr>>> {
    let input = Input {
        layout,
        schema,
        env: v.env,
        subqueries: v.subqueries,
    };
    exprs.iter().map(|e| eval::compile(e, input)).collect()
}

fn compile_one(
    e: &Expr,
    layout: &[AttrId],
    schema: &SchemaRef,
    v: &Values<'_>,
) -> Result<Arc<dyn PhysExpr>> {
    compile_all(&[e], layout, schema, v)?
        .pop()
        .ok_or_else(|| internal("an expression compiled to nothing".to_owned()))
}

pub(crate) fn filter_op(
    schema: &SchemaRef,
    layout: &[AttrId],
    predicate: &Expr,
    v: &Values<'_>,
) -> Result<Arc<dyn Operator>> {
    let e = compile_one(predicate, layout, schema, v)?;
    Ok(Filter::new(Arc::clone(schema), Arc::new(Compiled(e))))
}

pub(crate) fn project_op(
    schema: &SchemaRef,
    layout: &[AttrId],
    items: &[ProjectItemPlan],
    nullable: Option<&[bool]>,
    v: &Values<'_>,
) -> Result<Arc<Project>> {
    let items = items
        .iter()
        .map(|item| {
            Ok(match item {
                ProjectItemPlan::Column { index, name } => ProjectItem::Column {
                    index: *index,
                    name: name.clone(),
                },
                ProjectItemPlan::Expr { name, expr } => ProjectItem::expr(
                    name.clone(),
                    Arc::new(Compiled(compile_one(expr, layout, schema, v)?)),
                ),
                ProjectItemPlan::NullOf { index, name } => {
                    ProjectItem::expr(name.clone(), Arc::new(NullOf::new(schema, *index)?))
                }
            })
        })
        .collect::<Result<Vec<_>>>()?;
    match nullable {
        Some(nullable) => Project::try_new_nullable(schema, items, nullable),
        None => Project::try_new(schema, items),
    }
}

pub(crate) fn expand_op(
    schema: &SchemaRef,
    layout: &[AttrId],
    projections: &[Vec<ProjectItemPlan>],
    nullable: &[bool],
    v: &Values<'_>,
) -> Result<Arc<dyn Operator>> {
    let projects = projections
        .iter()
        .map(|items| project_op(schema, layout, items, Some(nullable), v))
        .collect::<Result<Vec<_>>>()?;
    Ok(Expand::try_new(projects)?)
}

pub(crate) fn replicate_op(
    schema: &SchemaRef,
    layout: &[AttrId],
    count: &Expr,
    v: &Values<'_>,
) -> Result<Arc<dyn Operator>> {
    let e = compile_one(count, layout, schema, v)?;
    Ok(ReplicateRows::new(
        Arc::clone(schema),
        Arc::new(Compiled(e)),
    ))
}

/// The generator `kind` over `arg`, on batches of `schema`.
pub(crate) fn generate_op(
    schema: &SchemaRef,
    layout: &[AttrId],
    kind: krishiv_plan::GeneratorKind,
    outer: bool,
    arg: &Expr,
    v: &Values<'_>,
) -> Result<Arc<dyn Operator>> {
    let e = compile_one(arg, layout, schema, v)?;
    // The argument is a column of the input, read but not carried.
    let skip = krishiv_plan::generator_arg(arg).and_then(|id| layout.iter().position(|a| *a == id));
    Ok(Generate::try_new(
        schema,
        kind,
        outer,
        Arc::new(Compiled(e)),
        skip,
    )?)
}

/// The exact window group limit `plan` over batches of `schema`.
pub(crate) fn group_limit_op(
    schema: &SchemaRef,
    plan: &GroupLimitPlan,
) -> Result<Arc<WindowGroupLimit>> {
    Ok(Arc::new(WindowGroupLimit::new(
        Arc::clone(schema),
        plan.clone(),
    )?))
}

pub(crate) fn sort_op(
    schema: &SchemaRef,
    keys: &[SortKey],
    spill: Option<SpillScope>,
) -> Result<Arc<Sort>> {
    Ok(Arc::new(Sort::new(Arc::clone(schema), keys, spill)?))
}

pub(crate) fn top_n_op(schema: &SchemaRef, keys: &[SortKey], limit: u64) -> Result<Arc<TopN>> {
    Ok(Arc::new(top_n(schema, keys, limit)?))
}

fn top_n(schema: &SchemaRef, keys: &[SortKey], limit: u64) -> Result<TopN> {
    let limit =
        usize::try_from(limit).map_err(|_| internal("top-N limit out of range".to_owned()))?;
    TopN::new(Arc::clone(schema), keys, 0, limit)
}

/// The filter pushed into a scan of `scan` over its table schema `table`.
pub(crate) fn scan_filter(
    table: &SchemaRef,
    layout: &[AttrId],
    predicate: &Expr,
    scan: &krishiv_plan::Scan,
    env: &EvalEnv,
) -> Result<PushedFilter> {
    let v = Values {
        env,
        subqueries: &[],
    };
    let compiled = compile_one(predicate, layout, table, &v)?;
    Ok(PushedFilter {
        predicate: Arc::new(Compiled(compiled)),
        expr: crate::planner::pushdown::translate(predicate, &scan.output),
        rows: crate::planner::pushdown::row_level(predicate, &scan.output, env.ansi),
    })
}

pub(crate) fn aggregate_spec(
    input: &SchemaRef,
    a: &AggregatePlan,
    env: &EvalEnv,
    spill: Option<SpillScope>,
) -> AggregateSpec {
    AggregateSpec {
        input: Arc::clone(input),
        group_by: a.group_by.clone(),
        aggregates: a.aggregates.clone(),
        output: a.output,
        ansi: env.ansi,
        spill,
        require_unit_weights: false,
        legacy_statistical: env.legacy_statistical_aggregate,
        udafs: crate::aggregate::UdafEnv::of(env.udfs.as_ref(), env.zone),
    }
}

/// The aggregate of `spec` over `input` (two for `DISTINCT` aggregates).
pub(crate) fn aggregate_node(
    input: Arc<dyn ExecNode>,
    spec: AggregateSpec,
    distinct: &[bool],
) -> Result<Arc<dyn ExecNode>> {
    if distinct.iter().any(|d| *d) {
        distinct_plan(input, spec, distinct)
    } else {
        Ok(AggregateNode::new(input, HashAggregate::new(spec)?))
    }
}

/// The conjunction of a join's residual conjuncts over probe then build
/// columns.
fn residual(
    probe: &SchemaRef,
    build: &SchemaRef,
    r: Option<&Residual>,
    env: &EvalEnv,
) -> Result<Option<Arc<dyn Predicate>>> {
    let Some(r) = r else {
        return Ok(None);
    };
    let both = crate::planner::concat_schemas(probe, build);
    let conjuncts: Vec<&Expr> = r.conjuncts.iter().map(AsRef::as_ref).collect();
    let v = Values {
        env,
        subqueries: &[],
    };
    let mut exprs = compile_all(&conjuncts, &r.layout, &both, &v)?;
    let mut all = exprs
        .pop()
        .ok_or_else(|| internal("a residual without conjuncts".to_owned()))?;
    while let Some(e) = exprs.pop() {
        all = eval::and(e, all);
    }
    Ok(Some(Arc::new(Compiled(all)) as Arc<dyn Predicate>))
}

pub(crate) fn sort_merge_join(
    probe: &SchemaRef,
    build: &SchemaRef,
    j: &SortMergeJoinPlan,
    env: &EvalEnv,
    spill: Option<SpillScope>,
) -> Result<SortMergeJoin> {
    SortMergeJoin::try_new(JoinSpec {
        join_type: j.join_type,
        probe_schema: Arc::clone(probe),
        build_schema: Arc::clone(build),
        probe_keys: j.probe_keys.clone(),
        build_keys: j.build_keys.clone(),
        null_safe: j.null_safe.clone(),
        filter: residual(probe, build, j.residual.as_ref(), env)?,
        single_match: j.single_match,
        spill,
    })
}

pub(crate) fn nested_loop_join(
    probe: &SchemaRef,
    build: &SchemaRef,
    j: &NestedLoopPlan,
    env: &EvalEnv,
    spill: Option<SpillScope>,
) -> Result<NestedLoopJoin> {
    NestedLoopJoin::try_new(NestedLoopSpec {
        join_type: j.join_type,
        probe_schema: Arc::clone(probe),
        build_schema: Arc::clone(build),
        filter: residual(probe, build, j.residual.as_ref(), env)?,
        spill,
    })
}

/// The source of one row without columns.
pub(crate) fn one_row() -> Result<Arc<dyn Source>> {
    let schema = Arc::new(arrow_schema::Schema::empty());
    let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(1));
    let batch =
        arrow_array::RecordBatch::try_new_with_options(Arc::clone(&schema), Vec::new(), &options)
            .map_err(|e| internal(e.to_string()))?;
    let scan = crate::operators::MemoryScan::try_new(
        schema,
        vec![krishiv_types::batch::Batch::new(batch)],
    )?;
    crate::operators::ScanPushdown::build(&scan, ScanPushdowns::default())
}

/// An input that is never run: the planner validates an aggregate over it.
pub(crate) struct Unbuilt;

impl ExecNode for Unbuilt {
    fn build_pipelines(self: Arc<Self>, _b: &mut PipelineBuilder) -> Result<()> {
        Err(internal("a validation-only plan node was run".to_owned()))
    }
}

#[cfg(test)]
mod tests;
