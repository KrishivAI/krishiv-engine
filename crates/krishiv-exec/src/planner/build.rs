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
//! parent.
//!
//! The planner produces data ([`PhysNode`]s); it validates each node and
//! learns its schema with the operator builders instantiation uses
//! ([`crate::physical`]), over throw-away runtime-filter and subquery
//! cells, and opens no scan.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow_schema::{Schema, SchemaRef};
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_io::{ColumnPath, FilterId};
use krishiv_optimizer::PlanStats;
use krishiv_plan::{
    AttrId, BoundColumn, CastMode, Expr, ExprNode, Literal, LogicalPlan, NullOrdering, Plan,
    PlanNode, ResolvedExpr, SortDirection, SortOrder, SubqueryKind,
};

use super::eval::{self, EvalEnv, PhysExpr};
use super::explain::ExplainNode;
use super::properties::{Distribution, Placement, range_keys};
use super::runtime_filter::{Settings, Site, column_type, kept, key_source, renamed};
use super::subquery::SubqueryValue;
use super::{PhysicalPlanner, PlanningContext};
use crate::error::Result;
use crate::operator::{Operator, Source};
use crate::operators::{FilterPushdown, Limit, Project, ProjectItem, ScanPushdowns, SortKey, TopN};
use crate::physical::{
    PhysNode, PhysicalPlan, ProjectItemPlan, ScanFilter, ScanId, ScanNode, ScanRuntimeFilter,
    SubqueryId, SubqueryPlan, SubqueryValueKind, ValueId, Values, expand_op, filter_op, project_op,
    replicate_op, scan_filter, sort_op, top_n_op,
};
use crate::values::{LocalValues, ValueService};

/// How far above the planner's first fresh attribute subquery ids start:
/// subquery values are read as attributes of these numbers, so they must
/// never meet an attribute of the plan or one the planner adds.
const SUBQUERY_GAP: u64 = 1 << 62;

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
            let mut planned = select(planned, &output, "Project")?;
            planner.attach_runtime_filters(&mut planned.node);
            Ok::<_, KrishivError>(
                PhysicalPlan::new(
                    planned.node,
                    planned.schema,
                    planned.explain,
                    planner.env.to_plan(),
                )
                .with_warnings(planner.warnings),
            )
        };
        krishiv_common::stack::on_big_stack(run)
            .ok_or_else(|| internal("cannot start a planner thread"))?
    }
}

/// A planned relation.
pub(crate) struct Planned {
    pub(crate) node: PhysNode,
    pub(crate) schema: SchemaRef,
    /// The attribute id of each column of `schema`.
    pub(crate) layout: Vec<AttrId>,
    pub(crate) explain: ExplainNode,
    /// Scan columns a runtime filter from above may target.
    pub(crate) sites: Vec<Site>,
    /// How its rows are placed in partitions.
    pub(crate) placement: Placement,
}

impl Planned {
    /// The column holding `id`.
    pub(crate) fn column(&self, id: AttrId) -> Result<usize> {
        self.layout
            .iter()
            .position(|a| *a == id)
            .ok_or_else(|| internal(&format!("attribute #{} is not an input column", id.0)))
    }

    /// The Arrow type column `index` is stored in.
    fn stored(&self, index: usize) -> Result<&arrow_schema::DataType> {
        self.schema
            .fields()
            .get(index)
            .map(|f| f.data_type())
            .ok_or_else(|| internal(&format!("column {index} is not in the schema")))
    }

    /// The attribute of column `c`.
    pub(crate) fn column_attr(&self, c: usize) -> Result<AttrId> {
        self.layout
            .get(c)
            .copied()
            .ok_or_else(|| internal(&format!("column {c} is not an input column")))
    }

    /// The node `make` builds over this relation, with output `schema`
    /// and `layout`.
    pub(crate) fn wrap(
        self,
        make: impl FnOnce(Box<PhysNode>) -> PhysNode,
        schema: SchemaRef,
        layout: Vec<AttrId>,
        explain: (&str, String),
    ) -> Self {
        Self {
            schema,
            node: make(Box::new(self.node)),
            sites: kept(&self.sites, &layout),
            placement: self.placement.restrict(&layout),
            layout,
            explain: ExplainNode::new(explain.0, explain.1, vec![self.explain]),
        }
    }
}

/// Expressions ready for a node: subqueries replaced by the columns of
/// their values, and the subqueries planned.
pub(crate) struct Prepared {
    pub(crate) exprs: Vec<Arc<Expr>>,
    pub(crate) subqueries: Vec<SubqueryPlan>,
    /// Placeholder values the planner validates the expressions with.
    values: Vec<(AttrId, Arc<dyn PhysExpr>)>,
    explain: Vec<ExplainNode>,
}

pub(crate) struct Planner<'a> {
    pub(crate) cx: &'a PlanningContext,
    pub(crate) env: EvalEnv,
    pub(crate) warnings: Vec<String>,
    /// `WITH` definitions in scope, by their output attributes.
    ctes: HashMap<Vec<AttrId>, Arc<LogicalPlan>>,
    /// The join hints a join read, by address.
    pub(crate) consumed_hints: HashSet<usize>,
    next_id: u64,
    /// Runtime filter settings.
    pub(crate) rf: Settings,
    /// The next runtime filter's id.
    pub(crate) next_filter: u64,
    /// The next scan's id.
    next_scan: u64,
    /// The next subquery's id.
    next_subquery: u64,
    /// The next exchange's id.
    pub(crate) next_exchange: u64,
    /// The runtime filters each scan receives, by scan.
    pub(crate) scan_filters: HashMap<ScanId, Vec<ScanRuntimeFilter>>,
    /// The estimates of every relation of the plan, computed once.
    pub(crate) stats: PlanStats,
}

impl<'a> Planner<'a> {
    fn new(cx: &'a PlanningContext, root: &Arc<LogicalPlan>) -> Self {
        let next_id = max_id(root).saturating_add(1).max(1 << 62);
        let mut env = EvalEnv::new(&cx.config, Arc::clone(&cx.user));
        // Planning validates UDF calls against the session's library; the
        // implementation a plan runs comes from its worker.
        env.udfs = cx.udfs.clone();
        let provided = super::ProviderStatistics(Arc::clone(&cx.tables));
        Self {
            stats: PlanStats::compute(root, &provided),
            cx,
            env,
            warnings: Vec::new(),
            ctes: HashMap::new(),
            consumed_hints: HashSet::new(),
            next_id,
            rf: Settings::new(&cx.config),
            next_filter: 1,
            next_scan: 0,
            next_subquery: next_id.saturating_add(SUBQUERY_GAP),
            next_exchange: 0,
            scan_filters: HashMap::new(),
        }
    }

    /// A fresh attribute id for a column the planner adds.
    pub(crate) fn fresh_id(&mut self) -> AttrId {
        let id = AttrId(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    /// The values expressions are validated with at planning.
    pub(crate) fn values<'p>(&'p self, prepared: &'p Prepared) -> Values<'p> {
        Values {
            env: &self.env,
            subqueries: &prepared.values,
        }
    }

    /// Placeholder values declaring the runtime filters `ids` (planning
    /// validates joins with them).
    pub(crate) fn placeholder_values(
        &self,
        ids: impl IntoIterator<Item = FilterId>,
    ) -> Result<Arc<dyn ValueService>> {
        let values = LocalValues::new();
        for id in ids {
            let f = self
                .scan_filters
                .values()
                .flatten()
                .find(|f| f.id == id)
                .ok_or_else(|| internal(&format!("runtime filter {id} has no scan")))?;
            values.declare_filter(f.id, f.column.clone(), f.key_type.clone())?;
        }
        Ok(Arc::new(values))
    }

    /// Hands each scan of `node` the runtime filters joins publish to it.
    fn attach_runtime_filters(&mut self, node: &mut PhysNode) {
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            if let PhysNode::Scan(s) = n
                && let Some(filters) = self.scan_filters.remove(&s.id)
            {
                s.runtime_filters = filters;
            }
            stack.extend(n.children_mut());
        }
    }

    pub(crate) fn plan(&mut self, p: &Arc<LogicalPlan>) -> Result<Planned> {
        match &p.node {
            PlanNode::Scan(scan) => self.scan(p, scan, None),
            PlanNode::Filter { input, predicate } => {
                if let PlanNode::Scan(scan) = &input.node
                    && !has_subquery(predicate)
                {
                    return self.scan(p, scan, Some(predicate));
                }
                if let PlanNode::Window {
                    input: below,
                    exprs,
                    partition_by,
                    order_by,
                } = &input.node
                {
                    // Spark's window group limit (`InferWindowGroupLimit`).
                    let limit = krishiv_optimizer::infer_window_group_limit(
                        predicate,
                        input,
                        self.window_group_limit_threshold(),
                    );
                    let below = self.plan(below)?;
                    let window = self.window(below, exprs, partition_by, order_by, limit)?;
                    return self.filter(window, predicate);
                }
                let mut input = self.plan(input)?;
                if matches!(input.node, PhysNode::Scan(_)) {
                    self.no_runtime_filters_before(&mut input, predicate);
                }
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
                for site in &mut planned.sites {
                    site.attr.0 = site
                        .attr
                        .0
                        .checked_add(*offset)
                        .ok_or_else(|| internal("attribute id overflow"))?;
                }
                // The layout's ids shifted without overflow, so do these.
                planned.placement = std::mem::take(&mut planned.placement)
                    .map_attrs(|a| a.0.checked_add(*offset).map(AttrId));
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
                planned.placement = std::mem::take(&mut planned.placement).map_attrs(|a| {
                    r.source
                        .iter()
                        .position(|s| *s == a)
                        .and_then(|i| r.output.get(i))
                        .map(|o| o.id)
                });
                // Each reference is planned anew: its scans are its own.
                planned.sites = renamed(&planned.sites, &r.source, &planned.layout);
                planned.explain.detail = format!("{} [{}]", r.name, columns(&r.output));
                Ok(planned)
            }
            PlanNode::Limit { input, limit } => {
                let n = limit_value(limit)?;
                // `LIMIT n OFFSET m` is the limit over the offset: skip, then keep.
                match &input.node {
                    PlanNode::Offset { input, offset } => {
                        self.limit(input, limit_value(offset)?, n)
                    }
                    _ => self.limit(input, 0, n),
                }
            }
            PlanNode::Offset { input, offset } => self.limit(input, limit_value(offset)?, u64::MAX),
            PlanNode::Sort {
                input,
                order,
                global,
            } => {
                let input = self.plan(input)?;
                self.sort(input, order, *global)
            }
            PlanNode::JoinHint { input, strategy } => {
                if !self.consumed_hints.contains(&(Arc::as_ptr(p) as usize)) {
                    self.warnings.push(format!(
                        "A join hint (strategy={}) is specified but it is not part of a join relation.",
                        strategy.name()
                    ));
                }
                self.plan(input)
            }
            PlanNode::Union { inputs, .. } => self.union(p, inputs),
            PlanNode::ReplicateRows { input, count } => {
                let input = self.plan(input)?;
                self.replicate(input, count)
            }
            PlanNode::Expand {
                input,
                projections,
                output,
            } => {
                let input = self.plan(input)?;
                self.expand(input, projections, output)
            }
            PlanNode::Window {
                input,
                exprs,
                partition_by,
                order_by,
            } => {
                let input = self.plan(input)?;
                self.window(input, exprs, partition_by, order_by, None)
            }
            PlanNode::OneRowRelation => Ok(one_row()),
            PlanNode::Range {
                start,
                end,
                step,
                num_slices,
                output,
            } => Ok(range_node(*start, *end, *step, *num_slices, output)),
            _ => {
                let text = p.to_string();
                let name = text.split_whitespace().next().unwrap_or_default();
                Err(unsupported(&format!(
                    "the {name} relation in physical plans"
                )))
            }
        }
    }

    /// `input` without its first `skip` rows, then its first `n`
    /// (`u64::MAX` for all the rest). Over a total sort the rows come from
    /// one top-N of `skip + n`.
    fn limit(&mut self, input: &Arc<LogicalPlan>, skip: u64, n: u64) -> Result<Planned> {
        if let PlanNode::Sort {
            input: sorted,
            order,
            global: true,
        } = &input.node
            && let Some(total) = n.checked_add(skip)
            && usize::try_from(total).is_ok_and(|t| t <= TopN::MAX_ROWS)
        {
            let sorted = self.plan(sorted)?;
            let top = self.top_n(sorted, order, total)?;
            if skip == 0 {
                return Ok(top);
            }
            return Ok(Self::skip_rows(top, skip, n));
        }
        let input = self.plan(input)?;
        // Each partition keeps at most `skip + n` rows before the gather;
        // the skip applies only after it.
        let input = if input.placement.satisfies(Distribution::Single) {
            input
        } else if let Some(total) = skip.checked_add(n) {
            Self::local_limit(input, total)
        } else {
            input
        };
        let input = self.require(input, Distribution::Single)?;
        Ok(Self::skip_rows(input, skip, n))
    }

    /// The `Limit` operator over `input`.
    fn skip_rows(input: Planned, offset: u64, limit: u64) -> Planned {
        let schema = Limit::with_offset(Arc::clone(&input.schema), offset, limit).schema();
        let layout = input.layout.clone();
        let detail = match (offset, limit) {
            (0, n) => n.to_string(),
            (m, u64::MAX) => format!("offset {m}"),
            (m, n) => format!("{n} offset {m}"),
        };
        let mut planned = input.wrap(
            |input| PhysNode::Limit {
                input,
                offset,
                limit,
            },
            schema,
            layout,
            ("Limit", detail),
        );
        // A filter below a limit would change which rows it keeps.
        planned.sites.clear();
        planned
    }

    /// `UNION ALL`: each input projected to the union's columns (the first
    /// input's attributes, nullable where any input's column is), fanned
    /// into the pipeline above. Where the inputs store a column of one Spark
    /// type in different Arrow types (`Utf8` beside `Utf8View`), every input
    /// yields the union carrier ([`union_carriers`]): the branches that store
    /// it otherwise cast it.
    fn union(&mut self, p: &Arc<LogicalPlan>, inputs: &[Arc<LogicalPlan>]) -> Result<Planned> {
        let out = p
            .resolved_output()
            .ok_or_else(|| internal("a union without output"))?;
        let layout: Vec<AttrId> = out.iter().map(|c| c.id).collect();
        let nullable: Vec<bool> = out.iter().map(|c| c.nullable).collect();
        let mut branches = Vec::with_capacity(inputs.len());
        for input in inputs {
            let cols = input
                .resolved_output()
                .ok_or_else(|| internal("a union input without output"))?;
            if cols.len() != out.len() {
                return Err(internal("union inputs of different widths"));
            }
            branches.push((cols, self.plan(input)?));
        }
        let carriers = union_carriers(&out, &branches)?;
        let mut schema: Option<SchemaRef> = None;
        let mut nodes = Vec::with_capacity(inputs.len());
        let mut explain = Vec::with_capacity(inputs.len());
        let mut sites = Vec::new();
        for (cols, planned) in branches {
            let mut cast = HashSet::new();
            let mut items = Vec::with_capacity(cols.len());
            for ((c, o), (carrier, mixed)) in cols.iter().zip(&out).zip(&carriers) {
                let index = planned.column(c.id)?;
                // A binary-safe `string` union (mixed `BinaryView` and
                // `Utf8View` storage) rides on `BinaryView`: every branch
                // casts to the carrier itself, so all branches meet byte for
                // byte. (A cast through the canonical `Utf8View` would
                // validate the carried bytes and reject them.) Branches that
                // agree keep their carrier: no cast, no copy.
                if planned.stored(index)? == carrier && !mixed {
                    items.push(ProjectItemPlan::Column {
                        index,
                        name: Some(o.name.clone()),
                    });
                } else {
                    let node = ExprNode::Cast {
                        expr: Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
                            c.clone(),
                        )))),
                        to: carrier_cast_target(&o.data_type, carrier),
                        mode: CastMode::Ansi,
                    };
                    cast.insert(c.id);
                    items.push(ProjectItemPlan::Expr {
                        name: o.name.clone(),
                        expr: Arc::new(Expr::new(node)),
                    });
                }
            }
            let op = project_op(
                &planned.schema,
                &planned.layout,
                &items,
                Some(&nullable),
                &self.no_values(),
            )?;
            let branch: Vec<AttrId> = cols.iter().map(|c| c.id).collect();
            // A cast column no longer is the scan's column.
            let passing: Vec<Site> = planned
                .sites
                .iter()
                .filter(|s| !cast.contains(&s.attr))
                .cloned()
                .collect();
            sites.extend(renamed(&passing, &branch, &layout));
            let input_layout = planned.layout.clone();
            let nullable = nullable.clone();
            let planned = planned.wrap(
                |input| PhysNode::Project {
                    input,
                    layout: input_layout,
                    items,
                    nullable: Some(nullable),
                    subqueries: Vec::new(),
                },
                op.schema(),
                layout.clone(),
                ("Project", columns(&cols)),
            );
            match &schema {
                None => schema = Some(Arc::clone(&planned.schema)),
                Some(s) if s.fields() != planned.schema.fields() => {
                    return Err(internal("union inputs of different physical types"));
                }
                Some(_) => {}
            }
            nodes.push(planned.node);
            explain.push(planned.explain);
        }
        let schema = schema.ok_or_else(|| internal("a union without inputs"))?;
        Ok(Planned {
            node: PhysNode::Union { inputs: nodes },
            schema,
            layout,
            explain: ExplainNode::new("Union", columns(&out), explain),
            sites,
            placement: Placement::Unknown,
        })
    }

    /// Values for expressions that read no subquery.
    pub(crate) fn no_values(&self) -> Values<'_> {
        Values {
            env: &self.env,
            subqueries: &[],
        }
    }

    /// Plans the uncorrelated subqueries `exprs` read and replaces each by
    /// the column of its value.
    pub(crate) fn prepare(&mut self, exprs: &[&Expr]) -> Result<Prepared> {
        let mut found = Vec::new();
        for e in exprs {
            subqueries(e, &mut found);
        }
        let mut by_plan: HashMap<usize, BoundColumn> = HashMap::new();
        let mut prepared = Prepared {
            exprs: Vec::with_capacity(exprs.len()),
            subqueries: Vec::new(),
            values: Vec::new(),
            explain: Vec::new(),
        };
        for (kind, plan) in found {
            let key = Arc::as_ptr(&plan) as usize;
            if by_plan.contains_key(&key) {
                continue;
            }
            let out = plan
                .resolved_output()
                .ok_or_else(|| internal("a subquery without output"))?;
            let (kind, data_type) = match (kind, out.as_slice()) {
                (SubqueryValueKind::Scalar, [c]) => {
                    (SubqueryValueKind::Scalar, c.data_type.clone())
                }
                (SubqueryValueKind::Exists, _) => {
                    (SubqueryValueKind::Exists, krishiv_types::DataType::Boolean)
                }
                _ => return Err(internal("a scalar subquery of several columns")),
            };
            // One partition holds the whole result: a scalar subquery must
            // see every row to fail on a second one, and `EXISTS` must see
            // every row to be false.
            let planned = self.plan(&plan)?;
            let planned = self.require(planned, Distribution::Single)?;
            let scalar = kind == SubqueryValueKind::Scalar;
            let planned = select(planned, &out[..usize::from(scalar)], "Project")?;
            let id = SubqueryId(self.next_subquery);
            self.next_subquery = self
                .next_subquery
                .checked_add(1)
                .ok_or_else(|| internal("subquery id overflow"))?;
            let arrow = data_type.to_arrow();
            prepared.explain.push(ExplainNode::new(
                "Subquery",
                if scalar { "scalar" } else { "exists" },
                vec![planned.explain],
            ));
            prepared.subqueries.push(SubqueryPlan {
                id,
                kind,
                data_type: arrow.clone(),
                plan: planned.node,
            });
            prepared.values.push((
                id.attr(),
                Arc::new(SubqueryValue {
                    kind,
                    value: LocalValues::new().subscribe(ValueId::Subquery(id))?,
                    data_type: arrow,
                }),
            ));
            by_plan.insert(
                key,
                BoundColumn {
                    id: id.attr(),
                    name: if scalar {
                        "scalarsubquery()"
                    } else {
                        "exists()"
                    }
                    .to_owned(),
                    data_type,
                    nullable: scalar,
                },
            );
        }
        for e in exprs {
            let e = Arc::new((*e).clone());
            prepared.exprs.push(if by_plan.is_empty() {
                e
            } else {
                with_subquery_columns(e, &by_plan)?
            });
        }
        Ok(prepared)
    }

    /// The node `make` builds over `input` from `prepared`'s subqueries:
    /// `make` gets the input, its layout and the subqueries.
    pub(crate) fn attach(
        input: Planned,
        prepared: Prepared,
        schema: SchemaRef,
        layout: Vec<AttrId>,
        explain: (&str, String),
        make: impl FnOnce(Box<PhysNode>, Vec<AttrId>, Vec<SubqueryPlan>) -> PhysNode,
    ) -> Planned {
        let mut children = vec![input.explain];
        children.extend(prepared.explain);
        Planned {
            sites: kept(&input.sites, &layout),
            placement: input.placement.restrict(&layout),
            schema,
            node: make(Box::new(input.node), input.layout, prepared.subqueries),
            layout,
            explain: ExplainNode::new(explain.0, explain.1, children),
        }
    }

    /// The estimate of `plan`'s rows (join and runtime filter costs), read
    /// from the plan's statistics: each relation is estimated once.
    pub(crate) fn estimated_rows(&self, plan: &Arc<LogicalPlan>) -> f64 {
        self.stats.rows(plan)
    }

    /// Whether every table `plan` reads reports statistics, so its estimate
    /// does not rest on default row counts. A `WITH` reference counts as
    /// default-sized, as the estimator treats it.
    pub(crate) fn has_statistics(&self, plan: &Arc<LogicalPlan>) -> bool {
        self.stats.has_statistics(plan)
    }

    fn scan(
        &mut self,
        plan: &Arc<LogicalPlan>,
        scan: &krishiv_plan::Scan,
        filter: Option<&Arc<Expr>>,
    ) -> Result<Planned> {
        // Table metadata only: the scan itself is built when the plan is
        // instantiated.
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
        let mut pushed = Vec::new();
        let mut scan_filter_plan = None;
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
            let filter = scan_filter(&table, &layout, predicate, scan, &self.env)?;
            let exact = factory.filter_pushdown(&filter) == FilterPushdown::Exact;
            detail.push_str(&format!(", PushedFilters: [{predicate}]"));
            pushed.push(filter);
            scan_filter_plan = Some(ScanFilter {
                layout,
                predicate: Arc::clone(predicate),
                exact,
            });
        }
        let schema = factory.output_schema(&ScanPushdowns {
            projection: Some(projection.clone()),
            filters: pushed,
            ..ScanPushdowns::default()
        })?;
        let id = ScanId(self.next_scan);
        self.next_scan = self
            .next_scan
            .checked_add(1)
            .ok_or_else(|| internal("scan id overflow"))?;
        let sites = if self.rf.enabled() {
            let scan_rows = self.estimated_rows(plan);
            let name = source_name(scan);
            scan.output
                .iter()
                .zip(&projection)
                .map(|(c, column)| Site {
                    attr: c.id,
                    column: column.clone(),
                    scan: id,
                    scan_rows,
                    scan_name: name.clone(),
                })
                .collect()
        } else {
            Vec::new()
        };
        let exact = scan_filter_plan.as_ref().is_some_and(|f| f.exact);
        let mut planned = Planned {
            schema,
            node: PhysNode::Scan(ScanNode {
                id,
                scan: scan.clone(),
                projection,
                filter: scan_filter_plan,
                runtime_filters: Vec::new(),
                row_stage: self.rf.row_stage(),
            }),
            layout: scan.output.iter().map(|c| c.id).collect(),
            explain: ExplainNode::leaf("Scan", detail),
            sites,
            placement: Placement::Unknown,
        };
        if let Some(predicate) = filter {
            self.no_runtime_filters_before(&mut planned, predicate);
        }
        match filter {
            Some(predicate) if !exact => self.filter(planned, predicate),
            _ => Ok(planned),
        }
    }

    /// A scan whose filter has a conjunct that may raise gets no runtime
    /// filter: Spark evaluates its runtime filter as the last conjunct of
    /// the filter over the scan, so it never keeps that conjunct from
    /// seeing a row (it raises on a row whose key matches nothing, even
    /// alone in its file). Dropping the scan's sites keeps joins above from
    /// targeting it.
    fn no_runtime_filters_before(&self, scan: &mut Planned, predicate: &Expr) {
        if krishiv_plan::may_raise(predicate, self.env.ansi) {
            scan.sites.clear();
        }
    }

    fn filter(&mut self, input: Planned, predicate: &Expr) -> Result<Planned> {
        let prepared = self.prepare(&[predicate])?;
        let e = prepared
            .exprs
            .first()
            .cloned()
            .ok_or_else(|| internal("a filter without a predicate"))?;
        let op = filter_op(&input.schema, &input.layout, &e, &self.values(&prepared))?;
        let layout = input.layout.clone();
        Ok(Self::attach(
            input,
            prepared,
            op.schema(),
            layout,
            ("Filter", predicate.to_string()),
            |input, layout, subqueries| PhysNode::Filter {
                input,
                layout,
                predicate: e,
                subqueries,
            },
        ))
    }

    /// Each row `count` times (`INTERSECT ALL` / `EXCEPT ALL`).
    fn replicate(&mut self, input: Planned, count: &Expr) -> Result<Planned> {
        let prepared = self.prepare(&[count])?;
        let e = prepared
            .exprs
            .first()
            .cloned()
            .ok_or_else(|| internal("a replication without a count"))?;
        let op = replicate_op(&input.schema, &input.layout, &e, &self.values(&prepared))?;
        let layout = input.layout.clone();
        let mut planned = Self::attach(
            input,
            prepared,
            op.schema(),
            layout,
            ("ReplicateRows", count.to_string()),
            |input, layout, subqueries| PhysNode::ReplicateRows {
                input,
                layout,
                count: e,
                subqueries,
            },
        );
        // The replication count is computed over groups the filter cuts.
        planned.sites.clear();
        Ok(planned)
    }

    /// Each row once per projection (`ROLLUP`, `CUBE`, `GROUPING SETS`):
    /// plain columns pass through without a copy.
    fn expand(
        &mut self,
        input: Planned,
        projections: &[Vec<Arc<Expr>>],
        output: &[BoundColumn],
    ) -> Result<Planned> {
        if projections.iter().any(|row| row.len() != output.len()) {
            return Err(internal("an expansion row of the wrong width"));
        }
        // A column an expansion nulls keeps the Arrow type of the column
        // another projection copies.
        let copied: Vec<Option<usize>> = (0..output.len())
            .map(|j| {
                projections
                    .iter()
                    .find_map(|row| row.get(j).and_then(|e| column_of(e)))
                    .and_then(|c| input.column(c).ok())
            })
            .collect();
        let null_of = |j: usize, e: &Expr| {
            let null = matches!(&e.node, ExprNode::Literal(Literal::Null(Some(_))));
            copied.get(j).copied().flatten().filter(|_| null)
        };
        let computed: Vec<&Expr> = projections
            .iter()
            .flat_map(|row| row.iter().enumerate())
            .filter(|(j, e)| column_of(e).is_none() && null_of(*j, e).is_none())
            .map(|(_, e)| e.as_ref())
            .collect();
        let prepared = self.prepare(&computed)?;
        let mut values = prepared.exprs.iter();
        let mut rows = Vec::with_capacity(projections.len());
        for row in projections {
            let mut items = Vec::with_capacity(row.len());
            for (j, (e, out)) in row.iter().zip(output).enumerate() {
                let name = out.name.clone();
                if let Some(c) = column_of(e) {
                    items.push(ProjectItemPlan::Column {
                        index: input.column(c)?,
                        name: Some(name),
                    });
                } else if let Some(index) = null_of(j, e) {
                    items.push(ProjectItemPlan::NullOf { index, name });
                } else {
                    let expr = values.next().ok_or_else(|| internal("expansion arity"))?;
                    items.push(ProjectItemPlan::Expr {
                        name,
                        expr: Arc::clone(expr),
                    });
                }
            }
            rows.push(items);
        }
        let nullable: Vec<bool> = output.iter().map(|c| c.nullable).collect();
        let layout: Vec<AttrId> = output.iter().map(|c| c.id).collect();
        let op = expand_op(
            &input.schema,
            &input.layout,
            &rows,
            &nullable,
            &self.values(&prepared),
        )?;
        let names: Vec<&str> = output.iter().map(|c| c.name.as_str()).collect();
        let text = names.join(", ");
        let mut planned = Self::attach(
            input,
            prepared,
            op.schema(),
            layout,
            (
                "Expand",
                format!("{} projections [{text}]", projections.len()),
            ),
            |input, layout, subqueries| PhysNode::Expand {
                input,
                layout,
                projections: rows,
                nullable,
                subqueries,
            },
        );
        // The rows are repeated with columns nulled: no runtime filter may
        // reach a scan through the expansion.
        planned.sites.clear();
        Ok(planned)
    }

    /// `SELECT exprs`: plain columns pass through without a copy.
    pub(crate) fn project(&mut self, input: Planned, exprs: &[Arc<Expr>]) -> Result<Planned> {
        let computed: Vec<&Expr> = exprs
            .iter()
            .filter(|e| column_of(e).is_none())
            .map(AsRef::as_ref)
            .collect();
        let prepared = self.prepare(&computed)?;
        let mut values = prepared.exprs.iter();
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
                Some(c) => items.push(ProjectItemPlan::Column {
                    index: input.column(c)?,
                    name: Some(name),
                }),
                None => {
                    let value = values.next().ok_or_else(|| internal("projection arity"))?;
                    items.push(ProjectItemPlan::Expr {
                        name,
                        expr: Arc::clone(value),
                    });
                }
            }
            layout.push(id);
        }
        let op = project_op(
            &input.schema,
            &input.layout,
            &items,
            None,
            &self.values(&prepared),
        )?;
        // A site passes as each output that is its column (renamed) or a
        // widening cast of it.
        let types = |id: AttrId| column_type(&input, id);
        let mut sites = Vec::new();
        // The input's placement follows each plain column to every name the
        // projection gives it.
        let placed = input.placement.clone();
        let renames: Vec<(AttrId, AttrId)> = exprs
            .iter()
            .zip(&layout)
            .filter_map(|(e, out)| column_of(e).map(|src| (src, *out)))
            .collect();
        for (e, out) in exprs.iter().zip(&layout) {
            if let Some(src) = key_source(e, &types) {
                sites.extend(input.sites.iter().filter(|s| s.attr == src).map(|s| Site {
                    attr: *out,
                    ..s.clone()
                }));
            }
        }
        let mut planned = Self::attach(
            input,
            prepared,
            op.schema(),
            layout,
            ("Project", list(exprs)),
            |input, layout, subqueries| PhysNode::Project {
                input,
                layout,
                items,
                nullable: None,
                subqueries,
            },
        );
        planned.sites = sites;
        planned.placement = placed.project(&renames);
        Ok(planned)
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
        let prepared = self.prepare(&computed)?;
        let mut items: Vec<ProjectItemPlan> = (0..input.layout.len())
            .map(|index| ProjectItemPlan::Column { index, name: None })
            .collect();
        let mut layout = input.layout.clone();
        let mut columns = Vec::with_capacity(exprs.len());
        let mut values = prepared.exprs.iter();
        for e in exprs {
            match column_of(e).and_then(|id| input.column(id).ok()) {
                Some(c) => columns.push(c),
                None => {
                    let value = values.next().ok_or_else(|| internal("key arity"))?;
                    columns.push(items.len());
                    let id = self.fresh_id();
                    items.push(ProjectItemPlan::Expr {
                        name: format!("_key{}", items.len()),
                        expr: Arc::clone(value),
                    });
                    layout.push(id);
                }
            }
        }
        let detail = list(exprs);
        let op = project_op(
            &input.schema,
            &input.layout,
            &items,
            None,
            &self.values(&prepared),
        )?;
        Ok((
            Self::attach(
                input,
                prepared,
                op.schema(),
                layout,
                ("Project", detail),
                |input, layout, subqueries| PhysNode::Project {
                    input,
                    layout,
                    items,
                    nullable: None,
                    subqueries,
                },
            ),
            columns,
        ))
    }

    /// A sort; a global one (`ORDER BY`) needs its input in ranges of the
    /// keys, a local one (`SORT BY`) sorts each partition.
    fn sort(&mut self, input: Planned, order: &[SortOrder], global: bool) -> Result<Planned> {
        let (input, keys) = self.sort_keys(input, order)?;
        let input = if global {
            let by = range_keys(&keys, &input.layout)?;
            self.require(input, Distribution::Ordered(&by))?
        } else {
            input
        };
        let schema = sort_op(&input.schema, &keys, None)?.schema();
        let layout = input.layout.clone();
        let sites = input.sites.clone();
        let mut planned = input.wrap(
            |input| PhysNode::Sort { input, keys },
            schema,
            layout,
            ("Sort", orders(order)),
        );
        planned.sites = sites;
        Ok(planned)
    }

    /// `input` cut to its first `n` rows by `keys` in each partition,
    /// shown with `detail`.
    pub(crate) fn local_top_n(
        input: Planned,
        keys: &[SortKey],
        n: u64,
        detail: String,
    ) -> Result<Planned> {
        let schema = top_n_op(&input.schema, keys, n)?.schema();
        let layout = input.layout.clone();
        let local = keys.to_vec();
        let mut local = input.wrap(
            |input| PhysNode::TopN {
                input,
                keys: local,
                limit: n,
            },
            schema,
            layout,
            ("LocalTopN", detail),
        );
        // A runtime filter below would drop rows the top-N ranks.
        local.sites.clear();
        Ok(local)
    }

    /// `input` cut to `n` rows in each partition.
    pub(crate) fn local_limit(input: Planned, n: u64) -> Planned {
        let schema = Arc::clone(&input.schema);
        let layout = input.layout.clone();
        let mut local = input.wrap(
            |input| PhysNode::Limit {
                input,
                offset: 0,
                limit: n,
            },
            schema,
            layout,
            ("LocalLimit", n.to_string()),
        );
        local.sites.clear();
        local
    }

    fn top_n(&mut self, input: Planned, order: &[SortOrder], n: u64) -> Result<Planned> {
        let (input, keys) = self.sort_keys(input, order)?;
        // Each partition keeps its best `n` rows before the gather.
        let input = if input.placement.satisfies(Distribution::Single) {
            input
        } else {
            Self::local_top_n(input, &keys, n, format!("{n}, {}", orders(order)))?
        };
        let input = self.require(input, Distribution::Single)?;
        let schema = top_n_op(&input.schema, &keys, n)?.schema();
        let layout = input.layout.clone();
        let mut planned = input.wrap(
            |input| PhysNode::TopN {
                input,
                keys,
                limit: n,
            },
            schema,
            layout,
            ("TopN", format!("{n}, {}", orders(order))),
        );
        planned.sites.clear();
        Ok(planned)
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

/// The Arrow type each column of a union has in every input, and whether
/// every branch converts to it: the type the inputs all store it in (kept
/// as is), or, where they differ, the canonical Arrow type of its Spark
/// type — except a `string` column with a binary-safe branch (non-UTF-8
/// bytes in a binary carrier), which rides on `BinaryView`: the canonical
/// `Utf8View` would route the branch through the validating `STRING` cast
/// and raise `INVALID_UTF8_STRING`. An input column of another Spark type
/// than the union's is an error.
fn union_carriers(
    out: &[BoundColumn],
    branches: &[(Vec<BoundColumn>, Planned)],
) -> Result<Vec<(arrow_schema::DataType, bool)>> {
    let mut carriers = Vec::with_capacity(out.len());
    for (i, o) in out.iter().enumerate() {
        let mut stored = Vec::with_capacity(branches.len());
        for (cols, planned) in branches {
            let c = cols
                .get(i)
                .ok_or_else(|| internal("union inputs of different widths"))?;
            stored.push((c, planned.stored(planned.column(c.id)?)?));
        }
        let first = stored.first().map(|(_, t)| *t);
        let (carrier, mixed) = if stored.iter().all(|(_, t)| Some(*t) == first) {
            (first.cloned(), false)
        } else if stored.iter().all(|(c, _)| c.data_type == o.data_type) {
            let binary = is_binary_safe_string(&o.data_type, stored.iter().map(|(_, t)| *t));
            (
                Some(if binary {
                    arrow_schema::DataType::BinaryView
                } else {
                    o.data_type.to_arrow()
                }),
                binary,
            )
        } else {
            return Err(internal("union inputs of different physical types"));
        };
        carriers.push((
            carrier.ok_or_else(|| internal("a union without inputs"))?,
            mixed,
        ));
    }
    Ok(carriers)
}

/// Whether the `string` column stored as `stored` needs a binary union
/// carrier: some branch holds it in a binary Arrow type, whose bytes are not
/// necessarily valid UTF-8, so the validating `STRING` cast is not a way to
/// meet the branches.
fn is_binary_safe_string<'a>(
    spark: &krishiv_types::DataType,
    mut stored: impl Iterator<Item = &'a arrow_schema::DataType>,
) -> bool {
    *spark == krishiv_types::DataType::String
        && stored.any(|t| {
            matches!(
                t,
                arrow_schema::DataType::BinaryView
                    | arrow_schema::DataType::Binary
                    | arrow_schema::DataType::LargeBinary
            )
        })
}

/// The Spark type a union branch casts to so its column lands on `carrier`:
/// the column's own Spark type — unless the carrier is a binary-safe
/// `string` union's `BinaryView`, whose Spark type is `binary`, so the cast
/// is a byte-preserving reinterpretation, never a validating string cast.
fn carrier_cast_target(
    spark: &krishiv_types::DataType,
    carrier: &arrow_schema::DataType,
) -> krishiv_types::DataType {
    if *spark == krishiv_types::DataType::String && *carrier == arrow_schema::DataType::BinaryView {
        krishiv_types::DataType::Binary
    } else {
        spark.clone()
    }
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
fn subqueries(e: &Expr, out: &mut Vec<(SubqueryValueKind, Arc<LogicalPlan>)>) {
    if let ExprNode::Subquery { kind, plan } = &e.node {
        match kind {
            SubqueryKind::Scalar => out.push((SubqueryValueKind::Scalar, Arc::clone(plan))),
            SubqueryKind::Exists => out.push((SubqueryValueKind::Exists, Arc::clone(plan))),
            // `IN (subquery)` is a semi join after analysis; one left here
            // fails when compiled.
            _ => {}
        }
    }
    e.for_each_child(&mut |c| subqueries(c, out));
}

/// `e` with each subquery of `columns` (by the address of its plan)
/// replaced by the column of its value.
fn with_subquery_columns(e: Arc<Expr>, columns: &HashMap<usize, BoundColumn>) -> Result<Arc<Expr>> {
    e.transform_down(&mut |x: Arc<Expr>| {
        if let ExprNode::Subquery { plan, .. } = &x.node
            && let Some(c) = columns.get(&(Arc::as_ptr(plan) as usize))
        {
            return Ok::<_, KrishivError>(Arc::new(Expr::new(ExprNode::Resolved(
                ResolvedExpr::Column(c.clone()),
            ))));
        }
        Ok(x)
    })
}

pub(crate) fn has_subquery(e: &Expr) -> bool {
    let mut found = Vec::new();
    subqueries(e, &mut found);
    !found.is_empty()
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
    let mut items = Vec::with_capacity(output.len());
    let mut columns_only = Vec::with_capacity(output.len());
    for c in output {
        let index = planned.column(c.id)?;
        items.push(ProjectItemPlan::Column {
            index,
            name: Some(c.name.clone()),
        });
        columns_only.push(ProjectItem::aliased(index, &c.name));
    }
    // Columns only: the schema needs no expression environment.
    let schema = Project::try_new(&planned.schema, columns_only)?.schema();
    let detail = columns(output);
    let input_layout = planned.layout.clone();
    Ok(planned.wrap(
        |input| PhysNode::Project {
            input,
            layout: input_layout,
            items,
            nullable: None,
            subqueries: Vec::new(),
        },
        schema,
        ids,
        (name, detail),
    ))
}

/// `range(start, end, step, numSlices)`: one `id BIGINT NOT NULL` column.
fn range_node(
    start: i64,
    end: i64,
    step: i64,
    num_slices: Option<u32>,
    output: &krishiv_plan::BoundColumn,
) -> Planned {
    use arrow_schema::{DataType, Field, Schema};
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    Planned {
        schema,
        node: PhysNode::Range(crate::physical::RangeNode {
            start,
            end,
            step,
            num_slices,
            output: output.id,
        }),
        layout: vec![output.id],
        explain: ExplainNode::leaf(
            "Range",
            &match num_slices {
                Some(n) => format!("[{start}, {end}, step {step}, slices {n}]"),
                None => format!("[{start}, {end}, step {step}]"),
            },
        ),
        sites: Vec::new(),
        placement: Placement::Unknown,
    }
}

/// One row without columns (`SELECT` without `FROM`).
fn one_row() -> Planned {
    Planned {
        schema: Arc::new(Schema::empty()),
        node: PhysNode::OneRow,
        layout: Vec::new(),
        explain: ExplainNode::leaf("OneRowRelation", ""),
        sites: Vec::new(),
        placement: Placement::Single,
    }
}

/// The schema of `l`'s columns then `r`'s.
pub(crate) fn concat_schemas(l: &SchemaRef, r: &SchemaRef) -> SchemaRef {
    let fields: Vec<_> = l.fields().iter().chain(r.fields()).cloned().collect();
    Arc::new(Schema::new(fields))
}

/// A `LIMIT` or `OFFSET`'s literal row count.
fn limit_value(e: &Expr) -> Result<u64> {
    match &e.node {
        ExprNode::Literal(Literal::Integer(v)) => u64::try_from(*v)
            .map_err(|_| internal("a negative LIMIT or OFFSET reached the planner")),
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

pub(crate) fn orders(order: &[SortOrder]) -> String {
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
