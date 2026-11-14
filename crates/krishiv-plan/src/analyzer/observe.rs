// SPDX-License-Identifier: Apache-2.0
//! `Dataset.observe(name, metrics)` (Spark's `CollectMetrics`): the
//! metrics are a global aggregate over the input that the rows pass
//! through, so they resolve as the select list of an aggregate with no
//! grouping, and `CheckAnalysis.checkMetric` (4.2.0) limits what they may
//! hold. Rows are untouched: the node's output is its input's.
//!
//! Order of errors, as Spark raises them: the analyzer's (an unresolved
//! column, a generator) first, then the missing name, then per metric in
//! order the first violation of a pre-order walk (window, nested
//! aggregate, DISTINCT, FILTER, a column or a non-deterministic call
//! outside any aggregate). The expression shown is the whole aliased
//! metric. A name used by two relations that are not the same relation
//! (equal field by field, `plan_id` included) is `DUPLICATED_METRICS_NAME`;
//! the same relation reached twice (a self-join, a self-union) is legal.

use std::sync::Arc;

use krishiv_common::error::{
    DuplicatedMetricsName, InvalidObservedMetricsAggregateExpressionWithDistinctUnsupported,
    InvalidObservedMetricsAggregateExpressionWithFilterUnsupported,
    InvalidObservedMetricsMissingName, InvalidObservedMetricsNestedAggregatesUnsupported,
    InvalidObservedMetricsNonAggregateFuncArgIsAttribute,
    InvalidObservedMetricsNonAggregateFuncArgIsNonDeterministic,
    InvalidObservedMetricsWindowExpressionsUnsupported, UnsupportedGeneratorOutsideSelect,
};

use super::AnalysisError;
use super::bind::addr;
use super::exprs::pretty;
use super::ops::has_generator;
use super::resolve::{Env, Rel, Resolver};
use crate::{Expr, ExprNode, FunctionKind, LogicalPlan, PlanNode, ResolvedExpr};

/// What an observed metric may not hold, in Spark's order of checks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Violation {
    Window,
    Nested,
    Distinct,
    Filter,
    Attribute,
    NonDeterministic,
}

/// The first violation in a pre-order walk of `metric`, `None` when the
/// metric is allowed.
fn violation(metric: &Arc<Expr>) -> Option<Violation> {
    let mut stack = vec![(Arc::clone(metric), false)];
    while let Some((e, seen)) = stack.pop() {
        let mut inside = seen;
        match &e.node {
            ExprNode::Window(_) => return Some(Violation::Window),
            ExprNode::Resolved(ResolvedExpr::Function(f)) if f.kind == FunctionKind::Aggregate => {
                if seen {
                    return Some(Violation::Nested);
                }
                if f.distinct {
                    return Some(Violation::Distinct);
                }
                if f.filter.is_some() {
                    return Some(Violation::Filter);
                }
                inside = true;
            }
            ExprNode::Resolved(ResolvedExpr::Column(_)) if !seen => {
                return Some(Violation::Attribute);
            }
            ExprNode::Resolved(ResolvedExpr::Alias { .. }) => {}
            _ if !seen && !crate::raises::is_deterministic(&e) => {
                return Some(Violation::NonDeterministic);
            }
            _ => {}
        }
        let mut kids = Vec::new();
        e.for_each_child(&mut |k| kids.push(Arc::clone(k)));
        stack.extend(kids.into_iter().rev().map(|k| (k, inside)));
    }
    None
}

impl Resolver<'_> {
    /// Resolves `metrics` over `input` as a global aggregate would and
    /// keeps the node with them bound; its output is the input's.
    pub(super) fn collect_metrics(
        &mut self,
        p: &Arc<LogicalPlan>,
        input: &Arc<LogicalPlan>,
        name: &str,
        metrics: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let o = p.origin;
        let c = self.child(input, env)?;
        self.observing = true;
        let items = self.select_items(&c, metrics, env, "CollectMetrics");
        self.observing = false;
        let items = items?;
        let zone = self.ctx.conf.session_zone;
        let shown = |b: &super::exprs::Bound| format!("\"{}\"", pretty(&b.shown(), zone));
        if let Some(b) = items.iter().find(|b| has_generator(&b.expr)) {
            return Err(AnalysisError::new(
                UnsupportedGeneratorOutsideSelect {
                    plan: format!("CollectMetrics {name}, [{}]", pretty(&b.shown(), zone)),
                },
                o,
            ));
        }
        if name.is_empty() || items.is_empty() {
            let listed: Vec<String> = items.iter().map(|b| pretty(&b.shown(), zone)).collect();
            return Err(AnalysisError::new(
                InvalidObservedMetricsMissingName {
                    operator: format!("CollectMetrics {name}, [{}]", listed.join(", ")),
                },
                o,
            ));
        }
        for b in &items {
            let Some(v) = violation(&b.expr) else {
                continue;
            };
            let expr = shown(b);
            let e: krishiv_common::error::KrishivError = match v {
                Violation::Window => {
                    InvalidObservedMetricsWindowExpressionsUnsupported { expr }.into()
                }
                Violation::Nested => {
                    InvalidObservedMetricsNestedAggregatesUnsupported { expr }.into()
                }
                Violation::Distinct => {
                    InvalidObservedMetricsAggregateExpressionWithDistinctUnsupported { expr }.into()
                }
                Violation::Filter => {
                    InvalidObservedMetricsAggregateExpressionWithFilterUnsupported { expr }.into()
                }
                Violation::Attribute => {
                    InvalidObservedMetricsNonAggregateFuncArgIsAttribute { expr }.into()
                }
                Violation::NonDeterministic => {
                    InvalidObservedMetricsNonAggregateFuncArgIsNonDeterministic { expr }.into()
                }
            };
            return Err(AnalysisError::new(e, o));
        }
        // One relation per name: the same relation (its canonical
        // representative) may appear again, another may not.
        let key = addr(&*self.rep(p));
        match self.observed.get(name) {
            Some(first) if *first != key => {
                return Err(AnalysisError::new(
                    DuplicatedMetricsName {
                        metric_name: name.to_owned(),
                    },
                    o,
                ));
            }
            Some(_) => {}
            None => {
                self.observed.insert(name.to_owned(), key);
            }
        }
        let node = PlanNode::CollectMetrics {
            input: Arc::clone(&c.plan),
            name: name.to_owned(),
            metrics: items.into_iter().map(|b| b.expr).collect(),
        };
        let mut r = self.make(o, node, c.out.to_vec(), &[&c]);
        r.hidden = Arc::clone(&c.hidden);
        Ok(r)
    }
}
