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

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use krishiv_io::{FilterId, RuntimeFilter};
use krishiv_memory::SpillScope;
use krishiv_plan::{AttrId, Expr};

use super::{
    AggregatePlan, HashJoinPlan, NestedLoopPlan, PhysNode, PhysicalPlan, ProjectItemPlan, Residual,
    ScanNode, SubqueryPlan,
};
use crate::aggregate::{AggregateNode, AggregateSpec, HashAggregate, distinct_plan};
use crate::error::{Result, internal};
use crate::join::{
    HashJoin, HashJoinNode, JoinRuntimeFilter, JoinSpec, NestedLoopJoin, NestedLoopJoinNode,
    NestedLoopSpec,
};
use crate::operator::{Operator, Sink, Source};
use crate::operators::{
    FileWrite, FileWriteSink, Filter, FilterPushdown, Predicate, Project, ProjectItem,
    PushedFilter, ReplicateRows, RuntimeFilters, ScanPushdowns, Sort, SortKey, TopN,
};
use crate::pipeline::{ExecNode, PipelineBuilder};
use crate::planner::TableProviderResolver;
use crate::planner::eval::{self, Compiled, EvalEnv, Input, PhysExpr};
use crate::planner::nodes::{BreakerNode, FanInNode, OperatorNode, SourceNode};
use crate::planner::subquery::{Slot, SubquerySink, SubqueryValue, WithSubqueries};

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
/// scans read, the spill scope of the query, the sink of a write. One per
/// query and worker.
#[derive(Clone)]
pub struct WorkerContext {
    tables: Arc<dyn TableProviderResolver>,
    spill: Option<SpillScope>,
    scopes: Option<Arc<dyn SpillScopes>>,
    write: Option<Arc<FileWriteSink>>,
}

impl WorkerContext {
    /// A context reading `tables`, without spilling.
    pub fn new(tables: Arc<dyn TableProviderResolver>) -> Self {
        Self {
            tables,
            spill: None,
            scopes: None,
            write: None,
        }
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

    /// The same context writing a plan's
    /// [`WriteFiles`](PhysNode::WriteFiles) rows through `sink`.
    #[must_use]
    pub fn with_write_sink(mut self, sink: Arc<FileWriteSink>) -> Self {
        self.write = Some(sink);
        self
    }

    /// The spill scope, if spilling.
    pub fn spill(&self) -> Option<&SpillScope> {
        self.spill.as_ref()
    }
}

impl fmt::Debug for WorkerContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkerContext")
            .field("tables", &self.tables)
            .field("spill", &self.spill.is_some())
            .field("spill_scopes", &self.scopes.is_some())
            .field("write", &self.write.is_some())
            .finish()
    }
}

/// The executable graph of `plan` on the worker `worker` describes. Each
/// call creates its own operators and cells: a plan can be instantiated
/// any number of times.
///
/// # Errors
/// The table providers' errors; `INTERNAL_ERROR` when the plan does not
/// match the worker's tables any more (a pushed filter honoured
/// differently, another result schema), names a runtime filter no scan
/// declares, or writes without a write sink.
pub fn instantiate(plan: &PhysicalPlan, worker: &WorkerContext) -> Result<Arc<dyn ExecNode>> {
    let run = || {
        let spill = match (&worker.spill, &worker.scopes) {
            (Some(scope), _) => Some(scope.clone()),
            (None, Some(scopes)) => Some(scopes.scope()?),
            (None, None) => None,
        };
        let mut cx = Instantiation {
            worker,
            spill,
            env: EvalEnv::from_plan(plan.env()),
            filters: HashMap::new(),
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
        Ok(built.node)
    };
    // Instantiation recurses once per node, as planning does.
    krishiv_common::stack::on_big_stack(run)
        .ok_or_else(|| internal("cannot start an instantiation thread".to_owned()))?
}

struct Instantiation<'a> {
    worker: &'a WorkerContext,
    /// The query's spill scope on this worker.
    spill: Option<SpillScope>,
    env: EvalEnv,
    /// One cell per runtime filter of the plan.
    filters: HashMap<FilterId, Arc<RuntimeFilter>>,
}

struct Built {
    node: Arc<dyn ExecNode>,
    schema: SchemaRef,
}

impl Instantiation<'_> {
    /// Creates the cell of every runtime filter a scan declares.
    fn declare_filters(&mut self, n: &PhysNode) -> Result<()> {
        let mut stack = vec![n];
        while let Some(n) = stack.pop() {
            if let PhysNode::Scan(s) = n {
                for f in &s.runtime_filters {
                    let cell = Arc::new(RuntimeFilter::new(
                        f.id,
                        f.column.clone(),
                        f.key_type.clone(),
                    ));
                    if self.filters.insert(f.id, cell).is_some() {
                        return Err(internal(format!("runtime filter {} declared twice", f.id)));
                    }
                }
            }
            stack.extend(n.children());
        }
        Ok(())
    }

    fn cell(&self, id: FilterId) -> Result<Arc<RuntimeFilter>> {
        self.filters
            .get(&id)
            .cloned()
            .ok_or_else(|| internal(format!("runtime filter {id} has no scan")))
    }

    fn node(&mut self, n: &PhysNode) -> Result<Built> {
        match n {
            PhysNode::Scan(s) => self.scan(s),
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
            PhysNode::Limit { input, limit } => {
                let input = self.node(input)?;
                let op = crate::operators::Limit::new(Arc::clone(&input.schema), *limit);
                Ok(operator(input, op))
            }
            PhysNode::Sort { input, keys } => {
                let input = self.node(input)?;
                let sort = sort_op(&input.schema, keys, self.spill.clone())?;
                Ok(breaker(input, Arc::clone(&sort) as Arc<dyn Sink>, sort))
            }
            PhysNode::TopN { input, keys, limit } => {
                let input = self.node(input)?;
                let top = top_n_op(&input.schema, keys, *limit)?;
                Ok(breaker(input, Arc::clone(&top) as Arc<dyn Sink>, top))
            }
            PhysNode::Aggregate { input, aggregate } => {
                let input = self.node(input)?;
                let spec =
                    aggregate_spec(&input.schema, aggregate, self.env.ansi, self.spill.clone());
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
            PhysNode::WriteFiles { input } => {
                let sink = self
                    .worker
                    .write
                    .clone()
                    .ok_or_else(|| internal("a write without a write sink".to_owned()))?;
                let input = self.node(input)?;
                if sink.input_schema().fields() != input.schema.fields() {
                    return Err(internal(
                        "a write sink of another schema than its plan".to_owned(),
                    ));
                }
                Ok(Built {
                    schema: Source::schema(sink.as_ref()),
                    node: Arc::new(FileWrite {
                        input: input.node,
                        sink,
                    }),
                })
            }
        }
    }

    fn scan(&self, s: &ScanNode) -> Result<Built> {
        let factory = self.worker.tables.provider(&s.scan)?.scan()?;
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
            dynamic.push(self.cell(f.id)?);
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
            let slot = Arc::new(Slot::default());
            let sink = SubquerySink::new(s.kind, s.data_type.clone(), Arc::clone(&slot));
            plans.push((plan.node, sink));
            values.push((
                s.id.attr(),
                Arc::new(SubqueryValue {
                    kind: s.kind,
                    slot,
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

    fn hash_join(&mut self, j: &HashJoinPlan) -> Result<Built> {
        let probe = self.node(&j.probe)?;
        let build = self.node(&j.build)?;
        let join = hash_join(
            &probe.schema,
            &build.schema,
            j,
            &self.env,
            self.spill.clone(),
            &mut |id| self.cell(id),
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

fn operator(input: Built, op: Arc<dyn Operator>) -> Built {
    Built {
        schema: op.schema(),
        node: Arc::new(OperatorNode {
            input: input.node,
            op,
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
            })
        })
        .collect::<Result<Vec<_>>>()?;
    match nullable {
        Some(nullable) => Project::try_new_nullable(schema, items, nullable),
        None => Project::try_new(schema, items),
    }
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
    })
}

pub(crate) fn aggregate_spec(
    input: &SchemaRef,
    a: &AggregatePlan,
    ansi: bool,
    spill: Option<SpillScope>,
) -> AggregateSpec {
    AggregateSpec {
        input: Arc::clone(input),
        group_by: a.group_by.clone(),
        aggregates: a.aggregates.clone(),
        output: a.output,
        ansi,
        spill,
        require_unit_weights: false,
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

/// The hash join of `j` over inputs of `probe` and `build`; `cell` gives
/// the runtime filter of an id.
pub(crate) fn hash_join(
    probe: &SchemaRef,
    build: &SchemaRef,
    j: &HashJoinPlan,
    env: &EvalEnv,
    spill: Option<SpillScope>,
    cell: &mut dyn FnMut(FilterId) -> Result<Arc<RuntimeFilter>>,
) -> Result<HashJoin> {
    let join = HashJoin::try_new(JoinSpec {
        join_type: j.join_type,
        probe_schema: Arc::clone(probe),
        build_schema: Arc::clone(build),
        probe_keys: j.probe_keys.clone(),
        build_keys: j.build_keys.clone(),
        null_safe: j.null_safe.clone(),
        filter: residual(probe, build, j.residual.as_ref(), env)?,
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
                filter: cell(f.id)?,
                options: f.options,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    join.with_runtime_filters(filters)
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

    #[test]
    fn a_validation_input_never_runs() {
        let mut b = PipelineBuilder::new(Arc::new(crate::testing::Count::default()));
        let e = Arc::new(Unbuilt).build_pipelines(&mut b).unwrap_err();
        assert!(e.to_string().contains("validation-only"), "{e}");
    }
}
