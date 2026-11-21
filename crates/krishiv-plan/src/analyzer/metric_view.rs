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
    CannotUpCastDatatype, InternalError, KrishivError,
    MissingAttributesResolvedAttributeMissingFromInput,
};

use super::AnalysisError;
use super::exprs::Bound;
use super::resolve::{Attr, Env, Rel, Resolver};
use crate::function::Builtin;
use crate::origin::SourceId;
use crate::walk::{Node, visit_distinct};
use crate::{
    AttrId, BoundColumn, Expr, ExprNode, FunctionId, FunctionKind, Grouping, LogicalPlan,
    MetricColumn, PlanNode, ResolvedExpr,
};
use krishiv_types::DataType;

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
        ansi: Option<bool>,
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
            ansi,
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
            ansi,
            ..
        } = &read.node
        else {
            return Ok(first);
        };
        let meas = output.get(dimensions.len()..).unwrap_or_default();
        let meas_ids: HashSet<AttrId> = meas.iter().map(|c| c.id).collect();
        let source_columns: Vec<BoundColumn> = source.resolved_output().unwrap_or_default();
        let source_names: Vec<String> = source_columns.iter().map(|c| c.name.clone()).collect();
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
                // Spark's text is `key not found: ExprId(<id>,<jvm uuid>)`: matched by its prefix,
                // so the second field is a placeholder, never a real UUID.
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
        self.check_dimension_up_casts(input, dimensions, env, p.origin)?;
        let written: Vec<&MetricColumn> = dimensions.iter().chain(measures).collect();
        let Some(rewritten_input) = self.replace_read(input, &source_columns, dimensions, &written)
        else {
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
        let mut rel = self.rel(&agg, env)?;
        // The view's operators keep the creator's mode (review b047d8901 F5): in the plan, and in
        // the `AggInfo` a later clause rebuilds the aggregate from (review 60b9e05fb finding 1).
        let pin = ansi.filter(|a| *a != self.ctx.conf.ansi).map(|creator| {
            // The operators the view's text wrote: its expressions' and its source's sources.
            let sources: HashSet<SourceId> = dimensions
                .iter()
                .chain(measures)
                .map(|m| m.expr.origin)
                .chain(std::iter::once(source.origin))
                .filter_map(|o| o.sql.map(|(s, _)| s))
                .collect();
            Pin {
                sources,
                creator,
                reader: self.ctx.conf.ansi,
            }
        });
        let mut info = rel.agg.as_deref().cloned();
        if let Some(pin) = &pin {
            rel.plan = pin.plan(&rel.plan);
            if let Some(info) = &mut info {
                info.input.plan = pin.plan(&info.input.plan);
                info.grouping = info.grouping.iter().map(|e| pin.expr(e)).collect();
                info.aggregates = info.aggregates.iter().map(|e| pin.expr(e)).collect();
                if let Some(top) = &info.top {
                    info.top = Some(
                        top.iter()
                            .map(|b| Bound {
                                expr: pin.expr(&b.expr),
                                ..b.clone()
                            })
                            .collect(),
                    );
                }
            }
        }
        if let Some(info) = &mut info
            && holds_no_aggregate(grouping, aggregates)
        {
            let input: Vec<String> = source_names
                .iter()
                .filter(|n| !dimensions.iter().any(|d| self.same_name(&d.name, n)))
                .chain(dimensions.iter().map(|d| &d.name))
                .map(|n| format!("\"{n}\""))
                .collect();
            info.metric = Some(Arc::new(Deferred {
                first,
                measures: measures.clone(),
                input: input.join(", "),
                origin: p.origin,
                pin,
            }));
        }
        rel.agg = info.map(Arc::new);
        Ok(rel)
    }

    /// The expressions of an `ORDER BY` (`op` `Sort`) or `HAVING` (`Filter`) over an aggregate of a
    /// metric-view read whose select list holds no aggregate (`m`), as Spark's rule order gives
    /// them (probed on Spark 4.2.1): Spark resolves the clause over the aggregate before the
    /// view's rewrite, so its `measure(m)` calls are added to the aggregate and rewritten with it,
    /// and each is replaced here by `m`'s expression. When the clause's expressions, typed over the
    /// read (a `measure(m)` has its measure's type), need an implicit cast anywhere (`ORDER BY
    /// measure(m) + 1` over a `BIGINT`, `HAVING measure(m) > 100`), Spark resolves the clause one
    /// pass later, after the rewrite, and its measures are missing from the rewritten input
    /// (`MISSING_ATTRIBUTES`). Expressions without `measure(..)` are returned as they are.
    pub(super) fn deferred_measures(
        &mut self,
        m: &Deferred,
        exprs: &[Arc<Expr>],
        env: &Arc<Env>,
        op: &'static str,
    ) -> Result<Vec<Arc<Expr>>, AnalysisError> {
        let mut used: Vec<String> = Vec::new();
        let mut rewritten = Vec::with_capacity(exprs.len());
        for e in exprs {
            rewritten.push(Arc::clone(e).transform_up(&mut |x| {
                Ok::<_, AnalysisError>(match self.measure_expr(&x, &m.measures) {
                    Some(r) => {
                        if let Some(name) = measure_name(&x)
                            && !used.iter().any(|u| self.same_name(u, name))
                        {
                            used.push(name.to_owned());
                        }
                        r
                    }
                    None => x,
                })
            })?);
        }
        if used.is_empty() {
            return Ok(exprs.to_vec());
        }
        let before = self.implicit_casts.get();
        // Typing over pass 1 only tells whether Spark would coerce. An error there (a clause that
        // does not type even over the read) is the rewrite's to report: resolving the rewritten
        // clause raises the same class, at the same expression.
        let typed = self.over_agg(&m.first, exprs, env, op);
        if typed.is_ok() && self.implicit_casts.get() != before {
            let names: Vec<String> = used.iter().map(|n| format!("\"{n}\"")).collect();
            return Err(AnalysisError::new(
                MissingAttributesResolvedAttributeMissingFromInput {
                    missing_attributes: names.join(", "),
                    input: m.input.clone(),
                    operator: "Aggregate".to_owned(),
                },
                m.origin,
            ));
        }
        Ok(rewritten)
    }

    /// Whether the unresolved chain from `input` down to its metric-view read passes the inlined
    /// text of a permanent or a temporary view (Spark's `View` node).
    fn crosses_view(&self, input: &Arc<LogicalPlan>) -> bool {
        let mut at = input;
        loop {
            if self.ctx.stored_view_texts.contains(at) || self.ctx.temp_view_texts.contains(at) {
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
                        // Looked up as Resolve looks it up: `builtin.f` / `system.builtin.f` by
                        // the bare name (whatever that answers is refused unless built in).
                        let builtin = match f.qualifier.as_slice() {
                            [b] => b.eq_ignore_ascii_case("builtin"),
                            [s, b] => {
                                s.eq_ignore_ascii_case("system")
                                    && b.eq_ignore_ascii_case("builtin")
                            }
                            _ => false,
                        };
                        let mut name = if builtin {
                            Vec::new()
                        } else {
                            f.qualifier.clone()
                        };
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

    /// `measure(m)` with `m` naming a measure of the view: that measure's expression (`m` itself:
    /// [`measure_name`]).
    fn measure_expr(&self, x: &Arc<Expr>, measures: &[MetricColumn]) -> Option<Arc<Expr>> {
        let name = measure_name(x)?;
        measures
            .iter()
            .find(|m| self.same_name(&m.name, name))
            .map(|m| Arc::clone(&m.expr))
    }

    /// Spark's `UpCast` of each dimension to its stored type (`ResolveMetricView`'s projection of the
    /// source): a source column whose type changed since the create so that the dimension no longer
    /// up-casts fails the read, the first such dimension in YAML order, whatever the reader's ANSI
    /// mode. Spark's `ResolveUpCast` raises it before any `Cast` exists, so each dimension is
    /// resolved here alone over the source of the read `input` reaches, uncast, and its type
    /// checked with `can_up_cast` (an `ARRAY<INT>` or a `DATE` source of an `INT` dimension is
    /// `CANNOT_UP_CAST_DATATYPE`, not the cast's type error; Spark 4.2.1 run, review c856eb6ee
    /// finding 1). A dimension that does not resolve alone is left to the read's own resolution,
    /// whose type errors Spark raises later. The `expression` parameter is Krishiv's pretty form,
    /// where Spark qualifies the columns (`spark_catalog.default.t.k`;
    /// docs/concepts/known-differences.md).
    fn check_dimension_up_casts(
        &mut self,
        input: &Arc<LogicalPlan>,
        dimensions: &[MetricColumn],
        env: &Arc<Env>,
        origin: crate::Origin,
    ) -> Result<(), AnalysisError> {
        let Some(source) = unresolved_source(input) else {
            return Ok(());
        };
        for d in dimensions {
            let alone = Arc::new(
                LogicalPlan::new(PlanNode::Project {
                    input: Arc::clone(source),
                    exprs: vec![Arc::new(
                        (*d.expr)
                            .clone()
                            .alias(d.name.clone())
                            .with_origin(d.expr.origin),
                    )],
                })
                .with_origin(d.expr.origin),
            );
            let Ok(rel) = self.rel(&alone, env) else {
                continue;
            };
            let PlanNode::Project { exprs, .. } = &rel.plan.node else {
                continue;
            };
            let Some(x) = exprs.first().map(|e| match &e.node {
                ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => expr,
                _ => e,
            }) else {
                continue;
            };
            let Some(from) = x.resolved_type() else {
                continue;
            };
            if !super::can_up_cast(&from, &d.data_type) {
                return Err(AnalysisError::new(
                    CannotUpCastDatatype {
                        expression: super::pretty_sql(x),
                        source_type: format!("\"{}\"", from.sql()),
                        target_type: format!("\"{}\"", d.data_type.sql()),
                        details: "The type path of the target object is:\n\nYou can either add an \
                                  explicit cast to the input data or choose a higher precision type \
                                  of the field in the target object"
                            .to_owned(),
                    },
                    origin,
                ));
            }
        }
        Ok(())
    }

    /// `plan` (unresolved) with the metric-view read it reaches through filters, projections and
    /// aliases replaced by Spark's projection of the source.
    fn replace_read(
        &self,
        plan: &Arc<LogicalPlan>,
        source_columns: &[BoundColumn],
        dimensions: &[MetricColumn],
        written: &[&MetricColumn],
    ) -> Option<Arc<LogicalPlan>> {
        let node = match &plan.node {
            PlanNode::MetricViewRead { input, .. } => {
                // The source's columns by name, each once. A name the source repeats is ambiguous
                // by name, so it is projected only when a dimension or measure names it (then
                // `AMBIGUOUS_REFERENCE`, as in Spark); otherwise nothing can read it (the view's
                // output is its dimensions and measures), and Spark reads the view (#941 review).
                let mut exprs: Vec<Arc<Expr>> = Vec::new();
                for (i, c) in source_columns.iter().enumerate() {
                    let n = &c.name;
                    let seen = source_columns[..i]
                        .iter()
                        .any(|o| self.same_name(&o.name, n));
                    let repeated = source_columns
                        .iter()
                        .filter(|o| self.same_name(&o.name, n))
                        .count()
                        > 1;
                    if seen
                        || dimensions.iter().any(|d| self.same_name(&d.name, n))
                        || (repeated && !names_column(written, n, |a, b| self.same_name(a, b)))
                    {
                        continue;
                    }
                    exprs.push(Arc::new(Expr::column([n.clone()]).with_origin(plan.origin)));
                }
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
                input: self.replace_read(input, source_columns, dimensions, written)?,
                predicate: Arc::clone(predicate),
            },
            PlanNode::Project { input, exprs } => PlanNode::Project {
                input: self.replace_read(input, source_columns, dimensions, written)?,
                exprs: exprs.clone(),
            },
            PlanNode::SubqueryAlias {
                input,
                alias,
                columns,
            } => PlanNode::SubqueryAlias {
                input: self.replace_read(input, source_columns, dimensions, written)?,
                alias: alias.clone(),
                columns: columns.clone(),
            },
            _ => return None,
        };
        Some(Arc::new(LogicalPlan::new(node).with_origin(plan.origin)))
    }
}

/// `plan` with each arithmetic operator lowered from one of `sources` (a metric view's own text)
/// evaluating in ANSI mode `creator`, its operands in the reader's (`reader`): Spark's parser builds
/// `+`, `-`, `*`, `/`, `%`, `div` and unary minus with the mode it parses the view's text under,
/// while the functions a read resolves (`sum`, `abs`) take the reader's. An aggregate or window call
/// is an operand as it is: it reads its operator's mode, which is the reader's.
fn pin_written_operators(
    plan: &Arc<LogicalPlan>,
    sources: &HashSet<SourceId>,
    creator: bool,
    reader: bool,
) -> Arc<LogicalPlan> {
    let done: Result<Arc<LogicalPlan>, std::convert::Infallible> =
        Arc::clone(plan).transform_up(&mut |n| {
            n.map_expressions(|e| e.transform_up(&mut |x| Ok(pin(x, sources, creator, reader))))
        });
    done.unwrap_or_else(|never| match never {})
}

/// `e` evaluated in ANSI mode `ansi`.
fn in_mode(e: Arc<Expr>, ansi: bool, data_type: DataType, nullable: bool) -> Arc<Expr> {
    let origin = e.origin;
    Arc::new(
        Expr::new(ExprNode::Resolved(ResolvedExpr::EvalMode {
            ansi,
            expr: e,
            data_type,
            nullable,
        }))
        .with_origin(origin),
    )
}

/// [`pin_written_operators`] of one expression node.
fn pin(x: Arc<Expr>, sources: &HashSet<SourceId>, creator: bool, reader: bool) -> Arc<Expr> {
    let ExprNode::Resolved(ResolvedExpr::Function(f)) = &x.node else {
        return x;
    };
    let written = x.origin.sql.is_some_and(|(s, _)| sources.contains(&s));
    let arithmetic = matches!(
        f.id,
        FunctionId::Builtin(
            Builtin::Add
                | Builtin::Subtract
                | Builtin::Multiply
                | Builtin::Divide
                | Builtin::Remainder
                | Builtin::IntegralDivide
                | Builtin::Negative
        )
    );
    if !written || !arithmetic {
        return x;
    }
    let mut f = f.clone();
    f.args = f.args.iter().map(|a| operand(a, reader)).collect();
    let (data_type, nullable) = (f.data_type.clone(), f.nullable);
    let call =
        Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f))).with_origin(x.origin));
    in_mode(call, creator, data_type, nullable)
}

/// An operand of a pinned operator, evaluating in the reader's mode `reader`: a scalar call or a
/// cast is wrapped; a column, a literal, an aggregate or window call or a pinned operator is not.
fn operand(a: &Arc<Expr>, reader: bool) -> Arc<Expr> {
    match &a.node {
        ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Scalar => {
            in_mode(Arc::clone(a), reader, f.data_type.clone(), f.nullable)
        }
        ExprNode::Cast { to, .. } => in_mode(Arc::clone(a), reader, to.clone(), true),
        _ => Arc::clone(a),
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

/// What an `ORDER BY` or `HAVING` over a metric-view aggregate whose select list holds no aggregate
/// needs ([`Resolver::deferred_measures`]).
pub(super) struct Deferred {
    /// The aggregate as written, over the read (pass 1): the clause is typed over it.
    pub(super) first: Rel,
    /// The view's measures.
    pub(super) measures: Vec<MetricColumn>,
    /// The rewritten input's columns, as `MISSING_ATTRIBUTES` lists them.
    pub(super) input: String,
    /// The aggregate's origin.
    pub(super) origin: crate::Origin,
    /// The creator's ANSI mode for the view's operators, when it differs from the reader's: the
    /// clause's rewritten expressions and the aggregates it adds are pinned with it.
    pub(super) pin: Option<Pin>,
}

/// The view's own operators (`sources`) evaluate in the creator's ANSI mode, their operands in
/// the reader's ([`pin_written_operators`]).
pub(super) struct Pin {
    sources: HashSet<SourceId>,
    creator: bool,
    reader: bool,
}

impl Pin {
    /// `plan` (resolved) pinned.
    fn plan(&self, plan: &Arc<LogicalPlan>) -> Arc<LogicalPlan> {
        pin_written_operators(plan, &self.sources, self.creator, self.reader)
    }

    /// `e` (resolved) pinned.
    pub(super) fn expr(&self, e: &Arc<Expr>) -> Arc<Expr> {
        let done: Result<Arc<Expr>, std::convert::Infallible> = Arc::clone(e)
            .transform_up(&mut |x| Ok(pin(x, &self.sources, self.creator, self.reader)));
        done.unwrap_or_else(|never| match never {})
    }
}

/// Whether Spark's rule leaves an aggregate over a metric-view read for a later pass: its select
/// list holds only columns and literals (aliased or not) and it groups by columns, so it holds no
/// aggregate (probed on Spark 4.2.1: `upper(region)` in the select list or as a grouping key, and
/// `GROUP BY 1`, have Spark rewrite it at once). `GROUP BY ALL` is the column `all` here.
fn holds_no_aggregate(grouping: &Grouping, aggregates: &[Arc<Expr>]) -> bool {
    let plain = |e: &Expr| matches!(e.node, ExprNode::Column(_) | ExprNode::Literal(_));
    let Grouping::GroupBy(keys) = grouping else {
        return false;
    };
    !keys.is_empty()
        && keys.iter().all(|k| matches!(k.node, ExprNode::Column(_)))
        && aggregates.iter().all(|e| match &e.node {
            ExprNode::Alias { expr, .. } => plain(expr),
            _ => plain(e),
        })
}

/// Whether an expression of `columns` (unresolved) names a column `name` (its last part).
fn names_column(columns: &[&MetricColumn], name: &str, same: impl Fn(&str, &str) -> bool) -> bool {
    columns.iter().any(|m| {
        let mut found = false;
        let _ = visit_distinct(Node::Expr(&m.expr), |n| {
            if let Node::Expr(x) = n
                && let ExprNode::Column(c) = &x.node
                && c.name.last().is_some_and(|l| same(l, name))
            {
                found = true;
            }
            std::ops::ControlFlow::<()>::Continue(())
        });
        found
    })
}

/// The measure `measure(m)` names (unresolved). Only the built-in reads a measure: a call
/// qualified by anything but `builtin.` / `system.builtin.` is a catalog function of that name
/// (Spark 4.2.1: `default.measure(m)` over a measure is `MISSING_AGGREGATION`, #941 review).
fn measure_name(x: &Expr) -> Option<&str> {
    let ExprNode::Function(f) = &x.node else {
        return None;
    };
    if !f.name.eq_ignore_ascii_case(MEASURE)
        || f.args.len() != 1
        || !(f.qualifier.is_empty() || super::bind::is_builtin_qualifier(&f.qualifier))
    {
        return None;
    }
    let ExprNode::Column(c) = &f.args.first()?.node else {
        return None;
    };
    c.name.last().map(String::as_str)
}

/// The source (unresolved) of the metric-view read `plan` reaches through filters, projections and
/// aliases, as [`Resolver::replace_read`] walks them.
fn unresolved_source(plan: &Arc<LogicalPlan>) -> Option<&Arc<LogicalPlan>> {
    match &plan.node {
        PlanNode::MetricViewRead { input, .. } => Some(input),
        PlanNode::Filter { input, .. }
        | PlanNode::Project { input, .. }
        | PlanNode::SubqueryAlias { input, .. } => unresolved_source(input),
        _ => None,
    }
}
