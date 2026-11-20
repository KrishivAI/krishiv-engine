// SPDX-License-Identifier: Apache-2.0
//! Reads of a metric view (Spark 4.2's `MetricViewPlaceholder` and `ResolveMetricView`; design note
//! `P3-metric-views.md` §4).
//!
//! A read resolves to a [`PlanNode::MetricViewRead`] whose output is the stored schema (the
//! dimensions, then the measures). An aggregate whose input reaches one through filters,
//! projections and aliases is resolved twice:
//! 1. as written, over the read, so names, `MISSING_AGGREGATION` and the other analysis errors
//!    are Spark's (the view's columns are what a query sees). Then Spark's rule checks: a measure
//!    column read anywhere but as the bare argument of `measure(..)` is missing once the read is
//!    replaced (`MISSING_ATTRIBUTES`, a Spark bug reproduced by class), and `measure(<dimension>)`
//!    is Spark's internal error;
//! 2. as Spark's rewrite gives it: the read replaced by a projection of the source columns no
//!    dimension shadows and each dimension cast to its stored type under its name, over the
//!    source with the view's filter, and each `measure(m)` replaced by `m`'s parsed expression,
//!    which resolves over that projection (so a measure sees a dimension that shadows a source
//!    column). Measures are not additive: each group computes its own.
//!
//! A read no aggregate rewrote stays in the plan: analysis succeeds and planning refuses it
//! ([`refuse_unplannable`]), as Spark's planner does.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_common::error::{
    InternalError, KrishivError, MissingAttributesResolvedAttributeMissingFromInput,
};

use super::AnalysisError;
use super::resolve::{Attr, Env, Rel, Resolver};
use crate::walk::{Node, visit_distinct};
use crate::{
    AttrId, BoundColumn, Expr, ExprNode, FunctionId, Grouping, LogicalPlan, MetricColumn, PlanNode,
    ResolvedExpr,
};

/// The name of Spark's `Measure` aggregate.
const MEASURE: &str = "measure";

/// Spark's message for a plan its planner has no strategy for (`QueryExecution`).
const PLANNING_FAILED: &str = "The Spark SQL phase planning failed with an internal error. You hit a bug in Spark or the Spark plugins you use. Please, report this bug to the corresponding communities or vendors, and provide the full stack trace.";

/// Spark's message for a `Measure` left in an aggregate.
const UNEVALUABLE: &str = "UnevaluableAggregateFunc.aggBufferAttributes should not be called.";

impl Resolver<'_> {
    /// A [`PlanNode::MetricViewRead`]: its source resolved, its output the stored columns
    /// (unqualified: a metric view's name never qualifies its columns).
    #[expect(
        clippy::too_many_arguments,
        reason = "the fields of the node, as the other node resolvers take them"
    )]
    pub(super) fn metric_view_read(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        name: &[String],
        dimensions: &[MetricColumn],
        measures: &[MetricColumn],
        output: &[BoundColumn],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        for m in dimensions.iter().chain(measures) {
            if let Some(feature) = self.unsupported_in(&m.expr) {
                return Err(AnalysisError::new(
                    krishiv_common::error::KrishivUnsupportedFeature { feature },
                    m.expr.origin,
                ));
            }
        }
        let c = self.child(input, env)?;
        let qual: Arc<[String]> = Arc::from(Vec::new());
        let mut out = Vec::with_capacity(dimensions.len() + measures.len());
        for (i, m) in dimensions.iter().chain(measures).enumerate() {
            let id = match output.get(i) {
                Some(o) => self.built_id(o.id),
                None => self.fresh(),
            };
            out.push(Attr {
                col: BoundColumn {
                    id,
                    name: m.name.clone(),
                    data_type: m.data_type.clone(),
                    nullable: m.nullable,
                },
                qual: Arc::clone(&qual),
                source: None,
                duplicate: false,
            });
        }
        let node = PlanNode::MetricViewRead {
            input: Arc::clone(&c.plan),
            name: name.to_vec(),
            dimensions: dimensions.to_vec(),
            measures: measures.to_vec(),
            output: out.iter().map(|a| a.col.clone()).collect(),
        };
        Ok(self.make(p.origin, node, out, &[&c]))
    }

    /// An aggregate, with Spark's metric-view rule when its input reads a metric view.
    pub(super) fn aggregate_node(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        grouping: &Grouping,
        aggregates: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        let Some(read) = chain_read(&c.plan) else {
            return self.aggregate(p, input, grouping, aggregates, env);
        };
        let read = Arc::clone(read);
        // Spark's rule never looks through a view (a `View` node ends the chain) and never fires
        // for a metric view over a metric view: the `measure(..)` stays, and planning refuses it.
        if self.crosses_view(input) || reads_metric_view_below(&read) {
            return self.aggregate(p, input, grouping, aggregates, env);
        }
        // Pass 1: as written, over the read.
        let first = self.aggregate(p, input, grouping, aggregates, env)?;
        let PlanNode::MetricViewRead {
            input: source,
            dimensions,
            measures,
            output,
            ..
        } = &read.node
        else {
            return Ok(first);
        };
        let meas = output.get(dimensions.len()..).unwrap_or_default();
        let meas_ids: HashSet<AttrId> = meas.iter().map(|c| c.id).collect();
        let source_names: Vec<String> = source
            .resolved_output()
            .unwrap_or_default()
            .into_iter()
            .map(|c| c.name)
            .collect();
        let missing = |cols: &[&BoundColumn], operator: &str| {
            let names: Vec<String> = cols.iter().map(|c| format!("\"{}\"", c.name)).collect();
            let input: Vec<String> = source_names
                .iter()
                .filter(|n| !dimensions.iter().any(|d| self.same_name(&d.name, n)))
                .chain(dimensions.iter().map(|d| &d.name))
                .map(|n| format!("\"{n}\""))
                .collect();
            AnalysisError::new(
                MissingAttributesResolvedAttributeMissingFromInput {
                    missing_attributes: names.join(", "),
                    input: input.join(", "),
                    operator: operator.to_owned(),
                },
                p.origin,
            )
        };
        // A grouping-set expansion passes every column of the read on, the measures included
        // (Spark's `Expand`), so they are missing once the read is replaced.
        if !matches!(grouping, Grouping::GroupBy(_)) && !meas.is_empty() {
            return Err(missing(&meas.iter().collect::<Vec<_>>(), "Project"));
        }
        // The operators between the aggregate and the read, then the aggregate.
        let mut at = &c.plan;
        while !std::ptr::eq(&**at, &*read) {
            let mut used = Vec::new();
            at.for_each_expr(&mut |e| reads_measures(e, &meas_ids, &HashSet::new(), &mut used));
            if !used.is_empty() {
                let operator = match &at.node {
                    PlanNode::Filter { .. } => "Filter",
                    _ => "Project",
                };
                let cols: Vec<&BoundColumn> =
                    meas.iter().filter(|c| used.contains(&c.id)).collect();
                return Err(missing(&cols, operator));
            }
            let mut next = None;
            at.for_each_child(&mut |k| next = Some(k));
            match next {
                Some(k) => at = k,
                None => break,
            }
        }
        if let Some(PlanNode::Aggregate {
            grouping: g,
            aggregates: items,
            ..
        }) = aggregate_of(&first.plan).map(|a| &a.node)
        {
            // `measure(<column>)` of a column that is not a measure: Spark's rule looks the
            // attribute up among the measures.
            let mut allowed = HashSet::new();
            let mut not_measure = None;
            for e in items {
                let _ = visit_distinct(Node::Expr(e), |n| {
                    if let Node::Expr(x) = n
                        && let Some(arg) = measure_arg(x)
                    {
                        match &arg.node {
                            ExprNode::Resolved(ResolvedExpr::Column(c))
                                if meas_ids.contains(&c.id) =>
                            {
                                allowed.insert(std::ptr::from_ref::<Expr>(arg) as usize);
                            }
                            // Any other attribute is looked up among the measures and fails
                            // (`measuresMap(a.exprId)`): a dimension, or a column a projection
                            // above the read made.
                            ExprNode::Resolved(ResolvedExpr::Column(c)) => {
                                not_measure.get_or_insert(c.id);
                            }
                            _ => {}
                        }
                    }
                    std::ops::ControlFlow::Continue(())
                });
            }
            if let Some(id) = not_measure {
                return Err(AnalysisError::new(
                    KrishivError::from(InternalError {
                        message: format!("key not found: ExprId({},{})", id.0, "metric-view"),
                    }),
                    p.origin,
                ));
            }
            let keys: Vec<Arc<Expr>> = match g {
                Grouping::GroupBy(k) => k.clone(),
                _ => Vec::new(),
            };
            let mut used = Vec::new();
            for e in keys.iter().chain(items) {
                reads_measures(e, &meas_ids, &allowed, &mut used);
            }
            if !used.is_empty() {
                let cols: Vec<&BoundColumn> =
                    meas.iter().filter(|c| used.contains(&c.id)).collect();
                return Err(missing(&cols, "Aggregate"));
            }
        }
        // Pass 2: the rewrite.
        let names: Vec<String> = first.out.iter().map(|a| a.col.name.clone()).collect();
        let mut items = Vec::with_capacity(aggregates.len());
        for (i, e) in aggregates.iter().enumerate() {
            let rewritten = Arc::clone(e).transform_up(&mut |x| {
                Ok::<_, AnalysisError>(self.measure_expr(&x, measures).unwrap_or(x))
            })?;
            let item = if Arc::ptr_eq(&rewritten, e) || matches!(e.node, ExprNode::Alias { .. }) {
                rewritten
            } else {
                match names.get(i).filter(|_| names.len() == aggregates.len()) {
                    Some(n) => {
                        Arc::new((*rewritten).clone().alias(n.clone()).with_origin(e.origin))
                    }
                    None => rewritten,
                }
            };
            items.push(item);
        }
        let Some(rewritten_input) = self.replace_read(input, &source_names, dimensions) else {
            return Ok(first);
        };
        let agg = Arc::new(
            LogicalPlan::new(PlanNode::Aggregate {
                input: rewritten_input,
                grouping: grouping.clone(),
                aggregates: items,
            })
            .with_origin(p.origin),
        );
        self.rel(&agg, env)
    }

    /// Whether the unresolved chain from `input` down to its metric-view read passes the inlined
    /// text of a permanent view.
    fn crosses_view(&self, input: &Arc<LogicalPlan>) -> bool {
        let mut at = input;
        loop {
            if self.ctx.stored_view_texts.contains(at) {
                return true;
            }
            match &at.node {
                PlanNode::Filter { input, .. }
                | PlanNode::Project { input, .. }
                | PlanNode::SubqueryAlias { input, .. } => at = input,
                _ => return false,
            }
        }
    }

    /// What a dimension or measure `e` (unresolved) holds that a read does not support yet: a
    /// subquery, or a call of a SQL function or a UDF. Bind and the session's lookups never see
    /// these expressions, so the tables and functions they reach would enter the plan without the
    /// reader being asked: they are refused (fail closed, review 2026-11-20 finding 1).
    fn unsupported_in(&self, e: &Expr) -> Option<String> {
        let mut found = None;
        let _ = visit_distinct(Node::Expr(e), |n| {
            if let Node::Expr(x) = n {
                match &x.node {
                    ExprNode::Subquery { .. } => {
                        found =
                            Some("a subquery in a metric view's dimension or measure".to_owned());
                    }
                    ExprNode::Function(f) => {
                        let mut name = f.qualifier.clone();
                        name.push(f.name.clone());
                        let other = match self.ctx.functions.lookup(&name, self.ctx.profile) {
                            super::FunctionLookup::Sql(_) => true,
                            super::FunctionLookup::Found(entry) => !matches!(
                                entry.id,
                                FunctionId::Builtin(_) | FunctionId::Registry { .. }
                            ),
                            super::FunctionLookup::KnownUnsupported
                            | super::FunctionLookup::NotFound => false,
                        };
                        if other {
                            found = Some(format!(
                                "calling `{}` (not a built-in function) in a metric view's dimension or measure",
                                name.join(".")
                            ));
                        }
                    }
                    _ => {}
                }
            }
            if found.is_some() {
                std::ops::ControlFlow::Break(())
            } else {
                std::ops::ControlFlow::Continue(())
            }
        });
        found
    }

    /// Whether two column names are the same under the session's case rule.
    fn same_name(&self, a: &str, b: &str) -> bool {
        if self.ctx.conf.case_sensitive {
            a == b
        } else {
            a.eq_ignore_ascii_case(b)
        }
    }

    /// `measure(m)` with `m` naming a measure of the view: that measure's expression.
    fn measure_expr(&self, x: &Arc<Expr>, measures: &[MetricColumn]) -> Option<Arc<Expr>> {
        let ExprNode::Function(f) = &x.node else {
            return None;
        };
        if !f.name.eq_ignore_ascii_case(MEASURE) || f.args.len() != 1 {
            return None;
        }
        let ExprNode::Column(c) = &f.args.first()?.node else {
            return None;
        };
        let name = c.name.last()?;
        measures
            .iter()
            .find(|m| self.same_name(&m.name, name))
            .map(|m| Arc::clone(&m.expr))
    }

    /// `plan` (unresolved) with the metric-view read it reaches through filters, projections and
    /// aliases replaced by Spark's projection of the source.
    fn replace_read(
        &self,
        plan: &Arc<LogicalPlan>,
        source_names: &[String],
        dimensions: &[MetricColumn],
    ) -> Option<Arc<LogicalPlan>> {
        let node = match &plan.node {
            PlanNode::MetricViewRead { input, .. } => {
                let mut exprs: Vec<Arc<Expr>> = source_names
                    .iter()
                    .filter(|n| !dimensions.iter().any(|d| self.same_name(&d.name, n)))
                    .map(|n| Arc::new(Expr::column([n.clone()]).with_origin(plan.origin)))
                    .collect();
                for d in dimensions {
                    exprs.push(Arc::new(
                        (*d.expr)
                            .clone()
                            .cast(d.data_type.clone())
                            .with_origin(d.expr.origin)
                            .alias(d.name.clone())
                            .with_origin(d.expr.origin),
                    ));
                }
                PlanNode::Project {
                    input: Arc::clone(input),
                    exprs,
                }
            }
            PlanNode::Filter { input, predicate } => PlanNode::Filter {
                input: self.replace_read(input, source_names, dimensions)?,
                predicate: Arc::clone(predicate),
            },
            PlanNode::Project { input, exprs } => PlanNode::Project {
                input: self.replace_read(input, source_names, dimensions)?,
                exprs: exprs.clone(),
            },
            PlanNode::SubqueryAlias {
                input,
                alias,
                columns,
            } => PlanNode::SubqueryAlias {
                input: self.replace_read(input, source_names, dimensions)?,
                alias: alias.clone(),
                columns: columns.clone(),
            },
            _ => return None,
        };
        Some(Arc::new(LogicalPlan::new(node).with_origin(plan.origin)))
    }
}

/// The metric-view read `plan` (resolved) reaches through filters, projections and aliases
/// (Spark's `collectMetricViewNode`).
fn chain_read(plan: &Arc<LogicalPlan>) -> Option<&Arc<LogicalPlan>> {
    let mut at = plan;
    loop {
        match &at.node {
            PlanNode::MetricViewRead { .. } => return Some(at),
            PlanNode::Filter { input, .. }
            | PlanNode::Project { input, .. }
            | PlanNode::SubqueryAlias { input, .. } => at = input,
            _ => return None,
        }
    }
}

/// Whether the source of the metric-view read `read` reads a metric view.
fn reads_metric_view_below(read: &LogicalPlan) -> bool {
    let mut found = false;
    read.for_each_child(&mut |source| {
        let _ = visit_distinct(Node::Plan(source), |n| {
            if matches!(n, Node::Plan(p) if matches!(p.node, PlanNode::MetricViewRead { .. })) {
                found = true;
                return std::ops::ControlFlow::Break(());
            }
            std::ops::ControlFlow::Continue(())
        });
    });
    found
}

/// Whether `plan` (resolved) reads a metric view through filters, projections and aliases.
pub(super) fn reads_metric_view(plan: &Arc<LogicalPlan>) -> bool {
    chain_read(plan).is_some()
}

/// Whether `plan` (resolved) is an aggregate relation.
pub(super) fn is_aggregate(plan: &Arc<LogicalPlan>) -> bool {
    aggregate_of(plan).is_some()
}

/// The aggregate node at the top of a resolved aggregate relation (below the projections a
/// select list's windows or generators add).
fn aggregate_of(plan: &Arc<LogicalPlan>) -> Option<&Arc<LogicalPlan>> {
    let mut at = plan;
    loop {
        match &at.node {
            PlanNode::Aggregate { .. } => return Some(at),
            PlanNode::Project { input, .. }
            | PlanNode::Filter { input, .. }
            | PlanNode::Window { input, .. }
            | PlanNode::Generate { input, .. } => at = input,
            _ => return None,
        }
    }
}

/// The argument of a resolved `measure(..)` call.
fn measure_arg(e: &Expr) -> Option<&Arc<Expr>> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) if matches!(&f.id, FunctionId::Registry { name, .. } if name == MEASURE) => {
            f.args.first()
        }
        _ => None,
    }
}

/// Adds to `used` each measure column `e` reads, other than the `allowed` references (by
/// address).
fn reads_measures(
    e: &Expr,
    measures: &HashSet<AttrId>,
    allowed: &HashSet<usize>,
    used: &mut Vec<AttrId>,
) {
    let _ = visit_distinct(Node::Expr(e), |n| {
        if let Node::Expr(x) = n
            && let ExprNode::Resolved(ResolvedExpr::Column(c)) = &x.node
            && measures.contains(&c.id)
            && !allowed.contains(&(std::ptr::from_ref::<Expr>(x) as usize))
            && !used.contains(&c.id)
        {
            used.push(c.id);
        }
        std::ops::ControlFlow::Continue(())
    });
}

/// Planning's refusal of what only an aggregate over a metric view may hold (design note §4,
/// golden results 3): a `measure(..)` left in a plan is its unevaluable aggregate's failure,
/// then a read no aggregate rewrote Spark's planning failure.
///
/// # Errors
/// `INTERNAL_ERROR` with Spark's message for the first found.
pub fn refuse_unplannable(plan: &LogicalPlan) -> Result<(), KrishivError> {
    let mut read = false;
    let mut measure = false;
    let _ = visit_distinct(Node::Plan(plan), |n| {
        match n {
            Node::Plan(p) => read |= matches!(p.node, PlanNode::MetricViewRead { .. }),
            Node::Expr(e) => measure |= measure_arg(e).is_some(),
        }
        std::ops::ControlFlow::Continue(())
    });
    if measure {
        return Err(InternalError {
            message: UNEVALUABLE.to_owned(),
        }
        .into());
    }
    if read {
        return Err(InternalError {
            message: PLANNING_FAILED.to_owned(),
        }
        .into());
    }
    Ok(())
}

/// Whether `e` is one of [`refuse_unplannable`]'s errors (`EXPLAIN` prints it as Spark's planning
/// failure, in every mode).
#[must_use]
pub fn is_unplannable(e: &KrishivError) -> bool {
    e.error_class() == krishiv_common::error::ErrorClass::InternalError
        && e.message_parameters()
            .iter()
            .any(|(k, v)| *k == "message" && (v == PLANNING_FAILED || v == UNEVALUABLE))
}
