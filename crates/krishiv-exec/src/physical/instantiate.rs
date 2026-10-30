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

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use krishiv_common::config::Config;
use krishiv_common::error::KrishivUnsupportedFeature;
use krishiv_io::{
    FormatRegistry, IoContext, StoreId, WriteOptions, effective_job_uuid, is_spark_file_source,
};
use krishiv_memory::SpillScope;
use krishiv_plan::{AttrId, Expr};
use krishiv_udf_sdk::UdfLibrary;
use object_store::path::Path;

use super::{
    AggregatePlan, HashJoinPlan, NestedLoopPlan, Partitioning, PhysNode, PhysicalPlan,
    ProjectItemPlan, Residual, ScanId, ScanNode, ScanSplits, SubqueryPlan, ValueId, Version,
    WritePlan, files_of, partitions_of,
};
use crate::aggregate::{AggregateNode, AggregateSpec, HashAggregate, distinct_plan};
use crate::error::{Result, internal};
use crate::join::{
    HashJoin, HashJoinNode, JoinRuntimeFilter, JoinSpec, NestedLoopJoin, NestedLoopJoinNode,
    NestedLoopSpec,
};
use crate::operator::{Operator, Sink, Source};
use crate::operators::{
    Expand, FileWrite, FileWriteSink, Filter, FilterPushdown, NullOf, Predicate, Project,
    ProjectItem, PushedFilter, ReplicateRows, RuntimeFilters, ScanPushdown, ScanPushdowns, Sort,
    SortKey, TopN,
};
use crate::pipeline::{ExecNode, PipelineBuilder};
use crate::planner::TableProviderResolver;
use crate::planner::eval::{self, Compiled, EvalEnv, Input, PhysExpr};
use crate::planner::nodes::{BreakerNode, FanInNode, OperatorNode, OrderedNode, SourceNode};
use crate::planner::subquery::{SubquerySink, SubqueryValue, WithSubqueries};
use crate::values::{
    BroadcastBuild, LocalValues, Partial, Producer, Publication, Value, ValueService,
};
use crate::window::{GroupLimitPlan, WindowGroupLimit};

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
        }
    }
}

impl WorkerContext {
    /// A context reading `tables`, without spilling.
    pub fn new(tables: Arc<dyn TableProviderResolver>) -> Self {
        Self {
            tables,
            udfs: None,
            relations: None,
            spill: None,
            scopes: None,
            io: None,
            formats: Arc::new(FormatRegistry::new()),
            config: None,
            values: None,
            producer: Producer::default(),
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

    /// The spill scope, if spilling.
    pub fn spill(&self) -> Option<&SpillScope> {
        self.spill.as_ref()
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
    let run = || {
        let spill = match (&worker.spill, &worker.scopes) {
            (Some(scope), _) => Some(scope.clone()),
            (None, Some(scopes)) => Some(scopes.scope()?),
            (None, None) => None,
        };
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
            ids: node_ids(plan.root()),
        };
        cx.declare_filters(plan.root())?;
        let built = cx.node(plan.root())?;
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
        Ok(if reads_range_in_order(plan.root()) {
            Arc::new(OrderedNode { inner: built.node })
        } else {
            built.node
        })
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
}

/// The pre-order index of every node under `root` (children in
/// [`PhysNode::children`] order), by address: `EXPLAIN ANALYZE`'s node ids.
pub(crate) fn node_ids(root: &PhysNode) -> HashMap<usize, u32> {
    let mut ids = HashMap::new();
    let mut stack = vec![root];
    let mut next = 0_u32;
    while let Some(n) = stack.pop() {
        ids.insert(std::ptr::from_ref(n).addr(), next);
        next = next.saturating_add(1);
        stack.extend(n.children().into_iter().rev());
    }
    ids
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
                let op = crate::window::Window::new(
                    Arc::clone(&input.schema),
                    window.clone(),
                    self.env.ansi,
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
            PhysNode::TopN { input, keys, limit } => {
                let input = self.node(input)?;
                let top = top_n_op(&input.schema, keys, *limit)?;
                Ok(breaker(input, Arc::clone(&top) as Arc<dyn Sink>, top))
            }
            PhysNode::Aggregate { input, aggregate } => {
                let input = self.node(input)?;
                let spec = aggregate_spec(&input.schema, aggregate, &self.env, self.spill.clone());
                let schema = spec.output_schema()?;
                let node = aggregate_node(input.node, spec, &aggregate.distinct)?;
                Ok(Built { node, schema })
            }
            PhysNode::HashJoin(j) => self.hash_join(j),
            PhysNode::NestedLoopJoin(j) => {
                let probe = self.node(&j.probe)?;
                let build = self.node(&j.build)?;
                let join = nested_loop_join(&probe.schema, &build.schema, j, &self.env)?;
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
            PhysNode::Exchange(e) if e.partitioning == Partitioning::Broadcast => {
                self.broadcast(e.id, &e.input)
            }
            PhysNode::Exchange(e) => self.node(&e.input),
        }
    }

    fn scan(&self, s: &ScanNode) -> Result<Built> {
        let factory = self.worker.tables.provider(&s.scan)?.scan()?;
        let factory = self.splits(s.id, factory)?;
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
        }
        let source = factory.build(ScanPushdowns {
            projection: Some(s.projection.clone()),
            filters,
            dynamic,
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
        let sink = Arc::new(FileWriteSink::try_new(
            Arc::clone(&format),
            io,
            Arc::clone(&input.schema),
            options,
            WriteOptions {
                store: StoreId::new(&write.store),
                dir: Path::from(write.dir.as_str()),
                task: self.task.partition,
                job_uuid: effective_job_uuid(&write.job_uuid, self.task.attempt),
                max_records_per_file: write.max_records_per_file,
                target_file_bytes: write.target_file_bytes,
                partition_by: write.partition_by.clone(),
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

    /// The input of the broadcast exchange `id`, published as this
    /// producer's partial of its value: on one worker the join over it
    /// shares one build among its drivers.
    fn broadcast(&mut self, id: super::ExchangeId, input: &PhysNode) -> Result<Built> {
        let input = self.node(input)?;
        let value = ValueId::Broadcast(id);
        let published = self.values.publish(Partial {
            value,
            version: Version::BATCH,
            producer: self.worker.producer,
            payload: Value::Broadcast(BroadcastBuild::new(&input.node)),
        })?;
        if published != Publication::Complete {
            // In one process every producer partition is this one: another
            // partial (`Waiting`) or a second exchange of this id
            // (`Ignored`) cannot be lowered.
            return Err(internal(format!(
                "{value} is {published:?} in this instantiation"
            )));
        }
        // The value names this input without owning it (see
        // `BroadcastBuild`): the plan holds it.
        Ok(input)
    }

    fn hash_join(&mut self, j: &HashJoinPlan) -> Result<Built> {
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

/// Whether `n` streams a `range` through row-wise nodes only (filters,
/// projections and limits), so its rows reach a `LIMIT` or the result in a
/// defined order.
fn reads_range_in_order(n: &PhysNode) -> bool {
    match n {
        PhysNode::Range(_) => true,
        PhysNode::Filter { input, .. }
        | PhysNode::Project { input, .. }
        | PhysNode::Limit { input, .. } => reads_range_in_order(input),
        _ => false,
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
    let PhysNode::Exchange(e) = input else {
        return None;
    };
    let Partitioning::Hash { keys: hash, .. } = &e.partitioning else {
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
    let limit =
        usize::try_from(limit).map_err(|_| internal("top-N limit out of range".to_owned()))?;
    Ok(Arc::new(TopN::new(Arc::clone(schema), keys, 0, limit)?))
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

/// The hash join of `j` over inputs of `probe` and `build`, publishing its
/// runtime filters to `values` (which declares them) as `producer`.
pub(crate) fn hash_join(
    probe: &SchemaRef,
    build: &SchemaRef,
    j: &HashJoinPlan,
    env: &EvalEnv,
    spill: Option<SpillScope>,
    values: &Arc<dyn ValueService>,
    producer: Producer,
) -> Result<HashJoin> {
    if j.distinct_build && !j.distinct_build_allowed() {
        return Err(internal(format!(
            "a distinct build on a {:?} join{}",
            j.join_type,
            if j.residual.is_some() {
                " with a residual condition"
            } else {
                ""
            }
        )));
    }
    let join = HashJoin::try_new(JoinSpec {
        join_type: j.join_type,
        probe_schema: Arc::clone(probe),
        build_schema: Arc::clone(build),
        probe_keys: j.probe_keys.clone(),
        build_keys: j.build_keys.clone(),
        null_safe: j.null_safe.clone(),
        filter: residual(probe, build, j.residual.as_ref(), env)?,
        single_match: j.single_match,
        spill,
    })?;
    if j.runtime_filters.is_empty() {
        return Ok(join);
    }
    let filters = j
        .runtime_filters
        .iter()
        .map(|f| {
            Ok(JoinRuntimeFilter {
                key: f.key,
                filter: values.filter(f.id)?,
                options: f.options,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    join.with_filter_values(filters, Arc::clone(values), producer)
}

pub(crate) fn nested_loop_join(
    probe: &SchemaRef,
    build: &SchemaRef,
    j: &NestedLoopPlan,
    env: &EvalEnv,
) -> Result<NestedLoopJoin> {
    NestedLoopJoin::try_new(NestedLoopSpec {
        join_type: j.join_type,
        probe_schema: Arc::clone(probe),
        build_schema: Arc::clone(build),
        filter: residual(probe, build, j.residual.as_ref(), env)?,
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
mod tests {
    use std::sync::Arc;

    use super::Unbuilt;
    use crate::pipeline::{ExecNode as _, PipelineBuilder};

    /// The pipelines `root` instantiates to, one `(preserve_order)` per pipeline.
    fn driver_orders(root: crate::physical::PhysNode) -> Vec<bool> {
        use arrow_schema::{DataType, Field, Schema};

        let plan = crate::physical::PhysicalPlan::new(
            root,
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
            crate::planner::ExplainNode::leaf("Range", ""),
            crate::physical::ExprEnv {
                ansi: true,
                legacy_statistical_aggregate: false,
                zone: Some(0),
                query_start_micros: 0,
                user: Arc::from("u"),
                map_key_dedup_last_win: false,
            },
        );
        let worker = super::WorkerContext::new(Arc::new(crate::planner::MemoryTables::new()));
        let node = super::instantiate(&plan, &worker).unwrap();
        let mut b = PipelineBuilder::new(Arc::new(crate::testing::Count::default()));
        node.build_pipelines(&mut b).unwrap();
        let (pipelines, _) = b.finish().unwrap();
        pipelines.iter().map(|p| p.preserve_order).collect()
    }

    fn range(num_slices: u32) -> crate::physical::PhysNode {
        crate::physical::PhysNode::Range(crate::physical::RangeNode {
            start: 0,
            end: 64,
            step: 1,
            num_slices: Some(num_slices),
            output: krishiv_plan::AttrId(0),
        })
    }

    /// A multi-slice range streamed to the result runs on one driver, so its
    /// batches reach the result in slice order whatever the scheduling.
    #[test]
    fn a_range_streamed_to_the_result_keeps_one_driver() {
        assert_eq!(driver_orders(range(8)), [true]);
        let limited = crate::physical::PhysNode::Limit {
            input: Box::new(range(8)),
            offset: 0,
            limit: 5,
        };
        assert_eq!(driver_orders(limited), [true]);
    }

    #[test]
    fn a_validation_input_never_runs() {
        let mut b = PipelineBuilder::new(Arc::new(crate::testing::Count::default()));
        let e = Arc::new(Unbuilt).build_pipelines(&mut b).unwrap_err();
        assert!(e.to_string().contains("validation-only"), "{e}");
    }
}
