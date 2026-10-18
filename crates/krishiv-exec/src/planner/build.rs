// SPDX-License-Identifier: Apache-2.0
//! [`DefaultPlanner`]: logical relations to plan nodes.
//!
//! Each relation becomes a [`Planned`] node: the plan node, the schema of
//! its batches and its **layout**, the attribute id of each column. Names
//! are for display only; expressions find their inputs by attribute id, so
//! a self-join instance ([`PlanNode::Shift`]) or a reordered join only
//! changes the layout. The root gets a final projection that puts the
//! logical output columns in order under their output names.

use std::sync::Arc;

use arrow_schema::SchemaRef;
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_io::ColumnPath;
use krishiv_plan::{
    AttrId, BoundColumn, Expr, ExprNode, Literal, LogicalPlan, NullOrdering, Plan, PlanNode,
    ResolvedExpr, SortDirection, SortOrder,
};

use super::explain::ExplainNode;
use super::nodes::{BreakerNode, OperatorNode, SourceNode};
use super::{PhysicalPlan, PhysicalPlanner, PlanningContext};
use crate::error::Result;
use crate::operator::{Operator, Sink, Source};
use crate::operators::{Limit, Project, ProjectItem, ScanPushdowns, Sort, SortKey, TopN};
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
            let mut planner = Planner {
                cx,
                warnings: Vec::new(),
            };
            let planned = planner.plan(root)?;
            let planned = finish(planned, &output)?;
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

    fn operator(self, op: Arc<dyn Operator>, layout: Vec<AttrId>, explain: (&str, String)) -> Self {
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

struct Planner<'a> {
    cx: &'a PlanningContext,
    warnings: Vec<String>,
}

impl Planner<'_> {
    fn plan(&mut self, p: &Arc<LogicalPlan>) -> Result<Planned> {
        match &p.node {
            PlanNode::Scan(scan) => self.scan(scan),
            PlanNode::Project { input, exprs } => {
                let input = self.plan(input)?;
                project(input, exprs)
            }
            PlanNode::SubqueryAlias { input, .. } => self.plan(input),
            PlanNode::Shift { input, offset } => {
                let mut planned = self.plan(input)?;
                for id in &mut planned.layout {
                    id.0 =
                        id.0.checked_add(*offset)
                            .ok_or_else(|| internal("attribute id overflow"))?;
                }
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
                    return top_n(input, order, n);
                }
                let input = self.plan(input)?;
                let op = Limit::new(Arc::clone(&input.schema), n);
                let layout = input.layout.clone();
                Ok(input.operator(op, layout, ("Limit", n.to_string())))
            }
            PlanNode::Sort { input, order, .. } => {
                let input = self.plan(input)?;
                sort(input, order, self.cx)
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

    fn scan(&self, scan: &krishiv_plan::Scan) -> Result<Planned> {
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
        let source = factory.build(ScanPushdowns {
            projection: Some(projection),
            ..ScanPushdowns::default()
        })?;
        let detail = format!("{} [{}]", source_name(scan), columns(&scan.output));
        Ok(Planned {
            schema: source.schema(),
            node: Arc::new(SourceNode(source)),
            layout: scan.output.iter().map(|c| c.id).collect(),
            explain: ExplainNode::leaf("Scan", detail),
        })
    }
}

/// `SELECT exprs`: plain and renamed columns.
fn project(input: Planned, exprs: &[Arc<Expr>]) -> Result<Planned> {
    let mut items = Vec::with_capacity(exprs.len());
    let mut layout = Vec::with_capacity(exprs.len());
    for e in exprs {
        match &e.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => {
                items.push(ProjectItem::column(input.column(c.id)?));
                layout.push(c.id);
            }
            ExprNode::Resolved(ResolvedExpr::Alias { expr, output })
                if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &expr.node =>
            {
                items.push(ProjectItem::aliased(input.column(c.id)?, &output.name));
                layout.push(output.id);
            }
            _ => return Err(unsupported_expr(e)),
        }
    }
    let op = Project::try_new(&input.schema, items)?;
    Ok(input.operator(op, layout, ("Project", list(exprs))))
}

/// The sort keys of `order` over `input`'s columns.
fn sort_keys(input: &Planned, order: &[SortOrder]) -> Result<Vec<SortKey>> {
    order
        .iter()
        .map(|o| {
            let ExprNode::Resolved(ResolvedExpr::Column(c)) = &o.expr.node else {
                return Err(unsupported_expr(&o.expr));
            };
            let column = input.column(c.id)?;
            let key = match o.direction {
                SortDirection::Ascending => SortKey::asc(column),
                SortDirection::Descending => SortKey::desc(column),
            };
            Ok(key.nulls_first(o.nulls == NullOrdering::NullsFirst))
        })
        .collect()
}

fn sort(input: Planned, order: &[SortOrder], cx: &PlanningContext) -> Result<Planned> {
    let keys = sort_keys(&input, order)?;
    let sort = Arc::new(Sort::new(
        Arc::clone(&input.schema),
        &keys,
        cx.spill.clone(),
    )?);
    Ok(breaker(
        input,
        Arc::clone(&sort) as Arc<dyn Sink>,
        sort,
        ("Sort", orders(order)),
    ))
}

fn top_n(input: Planned, order: &[SortOrder], n: u64) -> Result<Planned> {
    let keys = sort_keys(&input, order)?;
    let limit = usize::try_from(n).map_err(|_| internal("top-N limit out of range"))?;
    let top = Arc::new(TopN::new(Arc::clone(&input.schema), &keys, 0, limit)?);
    Ok(breaker(
        input,
        Arc::clone(&top) as Arc<dyn Sink>,
        top,
        ("TopN", format!("{n}, {}", orders(order))),
    ))
}

/// A breaker whose output order is kept: it sorts.
fn breaker(
    input: Planned,
    sink: Arc<dyn Sink>,
    source: Arc<dyn Source>,
    explain: (&str, String),
) -> Planned {
    Planned {
        schema: source.schema(),
        layout: input.layout.clone(),
        node: Arc::new(BreakerNode {
            input: input.node,
            sink,
            source,
            ordered: true,
        }),
        explain: ExplainNode::new(explain.0, explain.1, vec![input.explain]),
    }
}

/// The logical output columns, in order, under their output names.
fn finish(planned: Planned, output: &[BoundColumn]) -> Result<Planned> {
    let ids: Vec<AttrId> = output.iter().map(|c| c.id).collect();
    let named = planned
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
    Ok(planned.operator(op, ids, ("Project", detail)))
}

/// A `LIMIT`'s literal row count.
fn limit_value(e: &Expr) -> Result<u64> {
    match &e.node {
        ExprNode::Literal(Literal::Integer(v)) => {
            u64::try_from(*v).map_err(|_| internal("a negative LIMIT reached the planner"))
        }
        _ => Err(unsupported_expr(e)),
    }
}

fn source_name(scan: &krishiv_plan::Scan) -> String {
    match &scan.source {
        krishiv_plan::ReadSource::Table { name, .. } => name.join("."),
        krishiv_plan::ReadSource::Files { format, .. } => format!("{format} files"),
        _ => "source".to_owned(),
    }
}

fn columns(cols: &[BoundColumn]) -> String {
    cols.iter()
        .map(|c| format!("{}#{}", c.name, c.id.0))
        .collect::<Vec<_>>()
        .join(", ")
}

fn list(exprs: &[Arc<Expr>]) -> String {
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

fn unsupported_expr(e: &Expr) -> KrishivError {
    unsupported(&format!("the expression {e} in physical plans"))
}
