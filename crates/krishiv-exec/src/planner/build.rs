// SPDX-License-Identifier: Apache-2.0
//! [`DefaultPlanner`]: logical relations to plan nodes.
//!
//! Each relation becomes a [`Planned`] node: the plan node, the schema of
//! its batches and its **layout**, the attribute id of each column. Names
//! are for display only; expressions find their inputs by attribute id, so
//! a self-join instance ([`PlanNode::Shift`]) or a reordered join only
//! changes the layout. Columns the planner adds (computed join and sort
//! keys, aggregate results) get fresh ids above every id of the plan. The
//! root gets a final projection that puts the logical output columns in
//! order under their output names.
//!
//! A `WITH` definition is planned again at each reference (Spark inlines
//! deterministic CTEs too); a shared subtree (`Arc`) is planned once per
//! parent, because plan nodes hold per-query state.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{Schema, SchemaRef};
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_io::ColumnPath;
use krishiv_plan::{
    AttrId, BoundColumn, Expr, ExprNode, Literal, LogicalPlan, NullOrdering, Plan, PlanNode,
    ResolvedExpr, SortDirection, SortOrder, SubqueryKind,
};

use super::eval::{self, Compiled, EvalEnv, Input, PhysExpr};
use super::explain::ExplainNode;
use super::nodes::{BreakerNode, OperatorNode, SourceNode};
use super::subquery::{Kind, Slot, SubquerySink, SubqueryValue, WithSubqueries};
use super::{PhysicalPlan, PhysicalPlanner, PlanningContext, pushdown};
use crate::error::Result;
use crate::operator::{Operator, Sink, Source};
use crate::operators::{
    Filter, FilterPushdown, Limit, Project, ProjectItem, PushedFilter, ScanPushdowns, Sort,
    SortKey, TopN,
};
use crate::pipeline::ExecNode;

/// The embedded physical planner.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultPlanner;

impl PhysicalPlanner for DefaultPlanner {
    fn plan(&self, plan: &Plan, cx: &PlanningContext) -> Result<PhysicalPlan> {
        let root = plan.root();
        if !root.is_resolved() {
            return Err(internal("the physical planner needs a resolved plan"));
        }
        let output = root
            .resolved_output()
            .ok_or_else(|| internal("the plan has no resolved output"))?;
        // Planning recurses once per relation; plans nest up to
        // `krishiv_plan::MAX_DEPTH` levels.
        let run = || {
            let mut planner = Planner::new(cx, root);
            let planned = planner.plan(root)?;
            let planned = select(planned, &output, "Project")?;
            Ok::<_, KrishivError>(
                PhysicalPlan::new(planned.node, planned.schema, planned.explain)
                    .with_warnings(planner.warnings),
            )
        };
        krishiv_common::stack::on_big_stack(run)
            .ok_or_else(|| internal("cannot start a planner thread"))?
    }
}

/// A planned relation.
pub(crate) struct Planned {
    pub(crate) node: Arc<dyn ExecNode>,
    pub(crate) schema: SchemaRef,
    /// The attribute id of each column of `schema`.
    pub(crate) layout: Vec<AttrId>,
    pub(crate) explain: ExplainNode,
}

impl Planned {
    /// The column holding `id`.
    pub(crate) fn column(&self, id: AttrId) -> Result<usize> {
        self.layout
            .iter()
            .position(|a| *a == id)
            .ok_or_else(|| internal(&format!("attribute #{} is not an input column", id.0)))
    }

    /// `op` streaming over this relation.
    pub(crate) fn operator(
        self,
        op: Arc<dyn Operator>,
        layout: Vec<AttrId>,
        explain: (&str, String),
    ) -> Self {
        Self {
            schema: op.schema(),
            node: Arc::new(OperatorNode {
                input: self.node,
                op,
            }),
            layout,
            explain: ExplainNode::new(explain.0, explain.1, vec![self.explain]),
        }
    }
}

/// Compiled expressions and the subqueries they read.
pub(crate) struct CompiledExprs {
    pub(crate) exprs: Vec<Arc<dyn PhysExpr>>,
    pub(crate) subqueries: Vec<(Arc<dyn ExecNode>, Arc<SubquerySink>)>,
    pub(crate) explain: Vec<ExplainNode>,
}

pub(crate) struct Planner<'a> {
    pub(crate) cx: &'a PlanningContext,
    pub(crate) env: EvalEnv,
    pub(crate) warnings: Vec<String>,
    /// `WITH` definitions in scope, by their output attributes.
    ctes: HashMap<Vec<AttrId>, Arc<LogicalPlan>>,
    next_id: u64,
}

impl<'a> Planner<'a> {
    fn new(cx: &'a PlanningContext, root: &LogicalPlan) -> Self {
        Self {
            cx,
            env: EvalEnv::new(&cx.config),
            warnings: Vec::new(),
            ctes: HashMap::new(),
            next_id: max_id(root).saturating_add(1).max(1 << 62),
        }
    }

    /// A fresh attribute id for a column the planner adds.
    pub(crate) fn fresh_id(&mut self) -> AttrId {
        let id = AttrId(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    pub(crate) fn plan(&mut self, p: &Arc<LogicalPlan>) -> Result<Planned> {
        match &p.node {
            PlanNode::Scan(scan) => self.scan(scan, None),
            PlanNode::Filter { input, predicate } => {
                if let PlanNode::Scan(scan) = &input.node
                    && !has_subquery(predicate)
                {
                    return self.scan(scan, Some(predicate));
                }
                let input = self.plan(input)?;
                self.filter(input, predicate)
            }
            PlanNode::Project { input, exprs } => {
                let input = self.plan(input)?;
                self.project(input, exprs)
            }
            PlanNode::Aggregate {
                input,
                grouping,
                aggregates,
            } => {
                let input = self.plan(input)?;
                self.aggregate(input, grouping, aggregates)
            }
            PlanNode::Join {
                left,
                right,
                kind,
                condition,
            } => self.join(left, right, *kind, condition),
            PlanNode::SubqueryAlias { input, .. } => self.plan(input),
            PlanNode::Shift { input, offset } => {
                let mut planned = self.plan(input)?;
                for id in &mut planned.layout {
                    id.0 =
                        id.0.checked_add(*offset)
                            .ok_or_else(|| internal("attribute id overflow"))?;
                }
                // The subtree prints its own ids; its output reads shifted.
                planned
                    .explain
                    .detail
                    .push_str(&format!(" (ids +{offset})"));
                Ok(planned)
            }
            PlanNode::WithCte { input, ctes } => {
                for cte in ctes {
                    let out = cte
                        .plan
                        .resolved_output()
                        .ok_or_else(|| internal("a CTE without output"))?;
                    self.ctes
                        .insert(out.iter().map(|c| c.id).collect(), Arc::clone(&cte.plan));
                }
                self.plan(input)
            }
            PlanNode::CteRef(r) => {
                let def = self
                    .ctes
                    .get(&r.source)
                    .cloned()
                    .ok_or_else(|| internal(&format!("CTE {} is not in scope", r.name)))?;
                let planned = self.plan(&def)?;
                // Exactly the definition's columns, under this reference's
                // attributes.
                let source: Vec<BoundColumn> = r
                    .source
                    .iter()
                    .zip(&r.output)
                    .map(|(s, o)| BoundColumn {
                        id: *s,
                        ..o.clone()
                    })
                    .collect();
                let mut planned = select(planned, &source, "CteRef")?;
                planned.layout = r.output.iter().map(|c| c.id).collect();
                planned.explain.detail = format!("{} [{}]", r.name, columns(&r.output));
                Ok(planned)
            }
            PlanNode::Limit { input, limit } => {
                let n = limit_value(limit)?;
                if let PlanNode::Sort {
                    input,
                    order,
                    global: true,
                } = &input.node
                    && usize::try_from(n).is_ok_and(|n| n <= TopN::MAX_ROWS)
                {
                    let input = self.plan(input)?;
                    return self.top_n(input, order, n);
                }
                let input = self.plan(input)?;
                let op = Limit::new(Arc::clone(&input.schema), n);
                let layout = input.layout.clone();
                Ok(input.operator(op, layout, ("Limit", n.to_string())))
            }
            PlanNode::Sort { input, order, .. } => {
                let input = self.plan(input)?;
                self.sort(input, order)
            }
            _ => {
                let text = p.to_string();
                let name = text.split_whitespace().next().unwrap_or_default();
                Err(unsupported(&format!(
                    "the {name} relation in physical plans"
                )))
            }
        }
    }

    /// Compiles `exprs` over `input`, planning the uncorrelated subqueries
    /// they read.
    pub(crate) fn compile(&mut self, exprs: &[&Expr], input: &Planned) -> Result<CompiledExprs> {
        let mut found = Vec::new();
        for e in exprs {
            subqueries(e, &mut found);
        }
        let mut slots: Vec<(usize, Arc<dyn PhysExpr>)> = Vec::new();
        let mut plans = Vec::new();
        let mut explain = Vec::new();
        for (kind, plan) in found {
            let key = Arc::as_ptr(&plan) as usize;
            if slots.iter().any(|(k, _)| *k == key) {
                continue;
            }
            let out = plan
                .resolved_output()
                .ok_or_else(|| internal("a subquery without output"))?;
            let (kind, data_type) = match (kind, out.as_slice()) {
                (Kind::Scalar, [c]) => (Kind::Scalar, c.data_type.to_arrow()),
                (Kind::Exists, _) => (Kind::Exists, arrow_schema::DataType::Boolean),
                _ => return Err(internal("a scalar subquery of several columns")),
            };
            let planned = self.plan(&plan)?;
            let planned = select(
                planned,
                &out[..usize::from(kind == Kind::Scalar)],
                "Project",
            )?;
            let slot = Arc::new(Slot::default());
            let sink = SubquerySink::new(kind, data_type.clone(), Arc::clone(&slot));
            explain.push(ExplainNode::new(
                "Subquery",
                if kind == Kind::Scalar {
                    "scalar"
                } else {
                    "exists"
                },
                vec![planned.explain],
            ));
            plans.push((planned.node, sink));
            slots.push((
                key,
                Arc::new(SubqueryValue {
                    kind,
                    slot,
                    data_type,
                }),
            ));
        }
        let at = Input {
            layout: &input.layout,
            schema: &input.schema,
            env: &self.env,
            subqueries: &slots,
        };
        let exprs = exprs
            .iter()
            .map(|e| eval::compile(e, at))
            .collect::<Result<Vec<_>>>()?;
        Ok(CompiledExprs {
            exprs,
            subqueries: plans,
            explain,
        })
    }

    /// `op` over `input`, after the subqueries its expressions read.
    pub(crate) fn attach(
        input: Planned,
        op: Arc<dyn Operator>,
        compiled: CompiledExprs,
        layout: Vec<AttrId>,
        explain: (&str, String),
    ) -> Planned {
        if compiled.subqueries.is_empty() {
            return input.operator(op, layout, explain);
        }
        let mut children = vec![input.explain];
        children.extend(compiled.explain);
        Planned {
            schema: op.schema(),
            node: Arc::new(WithSubqueries {
                subqueries: compiled.subqueries,
                input: input.node,
                op,
            }),
            layout,
            explain: ExplainNode::new(explain.0, explain.1, children),
        }
    }

    fn scan(&mut self, scan: &krishiv_plan::Scan, filter: Option<&Arc<Expr>>) -> Result<Planned> {
        let provider = self.cx.tables.provider(scan)?;
        let factory = provider.scan()?;
        let table = factory.table_schema();
        let mut projection = Vec::with_capacity(scan.output.len());
        for c in &scan.output {
            let field = table
                .fields()
                .iter()
                .find(|f| f.name() == &c.name)
                .ok_or_else(|| internal(&format!("the table has no column {}", c.name)))?;
            projection.push(ColumnPath::new(field.name().clone()));
        }
        let mut filters = Vec::new();
        let mut exact = false;
        let mut detail = format!("{} [{}]", source_name(scan), columns(&scan.output));
        if let Some(predicate) = filter {
            // The scan evaluates pushed filters over its table schema.
            let layout: Vec<AttrId> = table
                .fields()
                .iter()
                .map(|f| match scan.output.iter().find(|c| c.name == *f.name()) {
                    Some(c) => c.id,
                    None => self.fresh_id(),
                })
                .collect();
            let compiled = eval::compile(
                predicate,
                Input {
                    layout: &layout,
                    schema: &table,
                    env: &self.env,
                    subqueries: &[],
                },
            )?;
            let typed = pushdown::translate(predicate, &scan.output);
            let pushed = PushedFilter {
                predicate: Arc::new(Compiled(compiled)),
                expr: typed,
            };
            exact = factory.filter_pushdown(&pushed) == FilterPushdown::Exact;
            detail.push_str(&format!(", PushedFilters: [{predicate}]"));
            filters.push(pushed);
        }
        let source = factory.build(ScanPushdowns {
            projection: Some(projection),
            filters,
            ..ScanPushdowns::default()
        })?;
        let planned = Planned {
            schema: source.schema(),
            node: Arc::new(SourceNode(source)),
            layout: scan.output.iter().map(|c| c.id).collect(),
            explain: ExplainNode::leaf("Scan", detail),
        };
        match filter {
            Some(predicate) if !exact => self.filter(planned, predicate),
            _ => Ok(planned),
        }
    }

    fn filter(&mut self, input: Planned, predicate: &Expr) -> Result<Planned> {
        let mut compiled = self.compile(&[predicate], &input)?;
        let e = compiled
            .exprs
            .pop()
            .ok_or_else(|| internal("a filter without a predicate"))?;
        let op = Filter::new(Arc::clone(&input.schema), Arc::new(Compiled(e)));
        let layout = input.layout.clone();
        Ok(Self::attach(
            input,
            op,
            compiled,
            layout,
            ("Filter", predicate.to_string()),
        ))
    }

    /// `SELECT exprs`: plain columns pass through without a copy.
    pub(crate) fn project(&mut self, input: Planned, exprs: &[Arc<Expr>]) -> Result<Planned> {
        let computed: Vec<&Expr> = exprs
            .iter()
            .filter(|e| column_of(e).is_none())
            .map(AsRef::as_ref)
            .collect();
        let compiled = self.compile(&computed, &input)?;
        let mut values = compiled.exprs.iter();
        let mut items = Vec::with_capacity(exprs.len());
        let mut layout = Vec::with_capacity(exprs.len());
        for e in exprs {
            let (name, id) = match &e.node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) => (c.name.clone(), c.id),
                ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => {
                    (output.name.clone(), output.id)
                }
                _ => (e.to_string(), self.fresh_id()),
            };
            match column_of(e) {
                Some(c) => items.push(ProjectItem::aliased(input.column(c)?, name)),
                None => {
                    let value = values.next().ok_or_else(|| internal("projection arity"))?;
                    items.push(ProjectItem::expr(
                        name,
                        Arc::new(Compiled(Arc::clone(value))),
                    ));
                }
            }
            layout.push(id);
        }
        let op = Project::try_new(&input.schema, items)?;
        Ok(Self::attach(
            input,
            op,
            compiled,
            layout,
            ("Project", list(exprs)),
        ))
    }

    /// `input` with a column for each sort key that is not one already, and
    /// the keys.
    fn sort_keys(
        &mut self,
        input: Planned,
        order: &[SortOrder],
    ) -> Result<(Planned, Vec<SortKey>)> {
        let exprs: Vec<Arc<Expr>> = order.iter().map(|o| Arc::clone(&o.expr)).collect();
        let input = self.with_columns(input, &exprs)?;
        let keys = order
            .iter()
            .zip(&input.1)
            .map(|(o, column)| {
                let key = match o.direction {
                    SortDirection::Ascending => SortKey::asc(*column),
                    SortDirection::Descending => SortKey::desc(*column),
                };
                key.nulls_first(o.nulls == NullOrdering::NullsFirst)
            })
            .collect();
        Ok((input.0, keys))
    }

    /// `input` with a computed column appended for each of `exprs` that is
    /// not a column of it, and the column of each.
    pub(crate) fn with_columns(
        &mut self,
        input: Planned,
        exprs: &[Arc<Expr>],
    ) -> Result<(Planned, Vec<usize>)> {
        let computed: Vec<&Expr> = exprs
            .iter()
            .filter(|e| column_of(e).is_none_or(|id| input.column(id).is_err()))
            .map(AsRef::as_ref)
            .collect();
        if computed.is_empty() {
            let columns = exprs
                .iter()
                .map(|e| input.column(column_of(e).unwrap_or(AttrId(u64::MAX))))
                .collect::<Result<Vec<_>>>()?;
            return Ok((input, columns));
        }
        let compiled = self.compile(&computed, &input)?;
        let mut items: Vec<ProjectItem> =
            (0..input.layout.len()).map(ProjectItem::column).collect();
        let mut layout = input.layout.clone();
        let mut columns = Vec::with_capacity(exprs.len());
        let mut values = compiled.exprs.iter();
        for e in exprs {
            match column_of(e).and_then(|id| input.column(id).ok()) {
                Some(c) => columns.push(c),
                None => {
                    let value = values.next().ok_or_else(|| internal("key arity"))?;
                    columns.push(items.len());
                    let id = self.fresh_id();
                    items.push(ProjectItem::expr(
                        format!("_key{}", items.len()),
                        Arc::new(Compiled(Arc::clone(value))),
                    ));
                    layout.push(id);
                }
            }
        }
        let detail = list(exprs);
        let op = Project::try_new(&input.schema, items)?;
        Ok((
            Self::attach(input, op, compiled, layout, ("Project", detail)),
            columns,
        ))
    }

    fn sort(&mut self, input: Planned, order: &[SortOrder]) -> Result<Planned> {
        let (input, keys) = self.sort_keys(input, order)?;
        let sort = Arc::new(Sort::new(
            Arc::clone(&input.schema),
            &keys,
            self.cx.spill.clone(),
        )?);
        Ok(breaker(
            input,
            Arc::clone(&sort) as Arc<dyn Sink>,
            sort,
            ("Sort", orders(order)),
            true,
        ))
    }

    fn top_n(&mut self, input: Planned, order: &[SortOrder], n: u64) -> Result<Planned> {
        let (input, keys) = self.sort_keys(input, order)?;
        let limit = usize::try_from(n).map_err(|_| internal("top-N limit out of range"))?;
        let top = Arc::new(TopN::new(Arc::clone(&input.schema), &keys, 0, limit)?);
        Ok(breaker(
            input,
            Arc::clone(&top) as Arc<dyn Sink>,
            top,
            ("TopN", format!("{n}, {}", orders(order))),
            true,
        ))
    }
}

/// The largest attribute id in `root` (expressions and subqueries
/// included), to number the planner's own columns above it.
fn max_id(root: &LogicalPlan) -> u64 {
    let mut max = 0;
    let mut stack = vec![root];
    let mut seen = std::collections::HashSet::new();
    while let Some(p) = stack.pop() {
        if !seen.insert(std::ptr::from_ref(p)) {
            continue;
        }
        if let Some(out) = p.resolved_output() {
            max = out.iter().map(|c| c.id.0).fold(max, u64::max);
        }
        if let PlanNode::Shift { offset, input } = &p.node
            && let Some(out) = input.resolved_output()
        {
            max = out
                .iter()
                .map(|c| c.id.0.saturating_add(*offset))
                .fold(max, u64::max);
        }
        p.for_each_child(&mut |c| stack.push(c));
        p.for_each_expr(&mut |e| {
            let mut exprs = vec![e.as_ref()];
            while let Some(e) = exprs.pop() {
                if let ExprNode::Subquery { plan, .. } = &e.node {
                    stack.push(plan);
                }
                e.for_each_child(&mut |c| exprs.push(c));
            }
        });
    }
    max
}

/// The id of a plain (or aliased plain) column expression.
pub(crate) fn column_of(e: &Expr) -> Option<AttrId> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => column_of(expr),
        _ => None,
    }
}

/// The uncorrelated subqueries `e` reads (not inside subquery plans).
fn subqueries(e: &Expr, out: &mut Vec<(Kind, Arc<LogicalPlan>)>) {
    if let ExprNode::Subquery { kind, plan } = &e.node {
        match kind {
            SubqueryKind::Scalar => out.push((Kind::Scalar, Arc::clone(plan))),
            SubqueryKind::Exists => out.push((Kind::Exists, Arc::clone(plan))),
            // `IN (subquery)` is a semi join after analysis; one left here
            // fails when compiled.
            _ => {}
        }
    }
    e.for_each_child(&mut |c| subqueries(c, out));
}

pub(crate) fn has_subquery(e: &Expr) -> bool {
    let mut found = Vec::new();
    subqueries(e, &mut found);
    !found.is_empty()
}

/// A breaker over `input`.
pub(crate) fn breaker(
    input: Planned,
    sink: Arc<dyn Sink>,
    source: Arc<dyn Source>,
    explain: (&str, String),
    ordered: bool,
) -> Planned {
    Planned {
        schema: source.schema(),
        layout: input.layout.clone(),
        node: Arc::new(BreakerNode {
            input: input.node,
            sink,
            source,
            ordered,
        }),
        explain: ExplainNode::new(explain.0, explain.1, vec![input.explain]),
    }
}

/// Exactly `output`'s columns of `planned`, in order, under their names.
pub(crate) fn select(planned: Planned, output: &[BoundColumn], name: &str) -> Result<Planned> {
    let ids: Vec<AttrId> = output.iter().map(|c| c.id).collect();
    let named = planned.schema.fields().len() == output.len()
        && planned
            .schema
            .fields()
            .iter()
            .zip(output)
            .all(|(f, c)| f.name() == &c.name);
    if planned.layout == ids && named {
        return Ok(planned);
    }
    let items = output
        .iter()
        .map(|c| Ok(ProjectItem::aliased(planned.column(c.id)?, &c.name)))
        .collect::<Result<Vec<_>>>()?;
    let op = Project::try_new(&planned.schema, items)?;
    let detail = columns(output);
    Ok(planned.operator(op, ids, (name, detail)))
}

/// The schema of `l`'s columns then `r`'s.
pub(crate) fn concat_schemas(l: &SchemaRef, r: &SchemaRef) -> SchemaRef {
    let fields: Vec<_> = l.fields().iter().chain(r.fields()).cloned().collect();
    Arc::new(Schema::new(fields))
}

/// A `LIMIT`'s literal row count.
fn limit_value(e: &Expr) -> Result<u64> {
    match &e.node {
        ExprNode::Literal(Literal::Integer(v)) => {
            u64::try_from(*v).map_err(|_| internal("a negative LIMIT reached the planner"))
        }
        _ => Err(eval::unsupported_expr(e)),
    }
}

fn source_name(scan: &krishiv_plan::Scan) -> String {
    match &scan.source {
        krishiv_plan::ReadSource::Table { name, .. } => name.join("."),
        krishiv_plan::ReadSource::Files { format, .. } => format!("{format} files"),
        _ => "source".to_owned(),
    }
}

pub(crate) fn columns(cols: &[BoundColumn]) -> String {
    cols.iter()
        .map(|c| format!("{}#{}", c.name, c.id.0))
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn list(exprs: &[Arc<Expr>]) -> String {
    format!(
        "[{}]",
        exprs
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn orders(order: &[SortOrder]) -> String {
    let text = order.iter().map(|o| {
        let dir = match o.direction {
            SortDirection::Ascending => "ASC",
            SortDirection::Descending => "DESC",
        };
        let nulls = match o.nulls {
            NullOrdering::NullsFirst => "NULLS FIRST",
            NullOrdering::NullsLast => "NULLS LAST",
        };
        format!("{} {dir} {nulls}", o.expr)
    });
    format!("[{}]", text.collect::<Vec<_>>().join(", "))
}

pub(crate) fn internal(message: &str) -> KrishivError {
    InternalError {
        message: message.to_owned(),
    }
    .into()
}

pub(crate) fn unsupported(feature: &str) -> KrishivError {
    KrishivUnsupportedFeature {
        feature: feature.to_owned(),
    }
    .into()
}
