// SPDX-License-Identifier: Apache-2.0
//! `DISTRIBUTE BY` (Spark's `RepartitionByExpression`; `CLUSTER BY` and
//! `SORT BY` add a local sort above it).
//!
//! The expressions resolve against the input, a column a projection below
//! dropped included (Spark lets the node host missing attributes). They may
//! be anything a projection allows (a map, `rand()`, a literal) except an
//! aggregate, a window or a generator, and they may not read the outer
//! query. The node stays in the resolved plan with the bound expressions:
//! Spark evaluates them while it shuffles, so an expression that fails at
//! run time (`x + 'a'`) fails the query, and the physical planner places a
//! hash exchange on their values. An expression that reads a column a
//! projection below dropped widens the input with it, as a filter's does,
//! and a projection above restores the node's output.

use std::sync::Arc;

use krishiv_common::error::{UnsupportedExprForOperator, UnsupportedGeneratorOutsideSelect};

use super::AnalysisError;
use super::exprs::{Bound, pretty};
use super::ops::has_generator;
use super::resolve::{Env, Rel, Resolver, Widen};
use super::subquery::correlated_reference;
use crate::{Expr, ExprNode, FunctionKind, LogicalPlan, Origin, PlanNode, ResolvedExpr};

/// Spark's operator name, as its errors show it.
const OPERATOR: &str = "RepartitionByExpression";

impl Resolver<'_> {
    /// Checks `exprs` against `input` and resolves to a
    /// [`PlanNode::Repartition`] over it with the bound expressions; a sort
    /// above reads the input's columns through it.
    pub(super) fn repartition(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        exprs: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let c = self.child(input, env)?;
        let (bounds, wide) = self.bind_missing(&c, exprs, env, OPERATOR, false)?;
        let o = p.origin;
        if bounds.iter().any(|b| b.outer.is_some()) {
            let shown: Vec<Arc<Expr>> = bounds.iter().map(Bound::shown).collect();
            return Err(correlated_reference(&shown, o));
        }
        if let Some(b) = bounds.iter().find(|b| has_generator(&b.expr)) {
            return Err(AnalysisError::new(
                UnsupportedGeneratorOutsideSelect {
                    plan: format!("{OPERATOR} [{}]", pretty(&b.shown(), None)),
                },
                o,
            ));
        }
        if let Some(e) = self.misplaced_aggregates(&bounds, o) {
            return Err(e);
        }
        if let Some(b) = bounds.iter().find(|b| b.win.is_some()) {
            return Err(super::windows::misplaced(
                super::exprs::WindowUse::Other,
                b,
                o,
            ));
        }
        let wide = self.reveal(&wide, &bounds)?;
        let node = PlanNode::Repartition {
            input: Arc::clone(&wide.plan),
            exprs: bounds.into_iter().map(|b| b.expr).collect(),
        };
        let mut r = self.make(o, node, wide.out.to_vec(), &[&wide]);
        let extended = wide.out.len() > c.out.len();
        // A sort above reaches the input's columns through the node, and a
        // grouping key under it (Spark 4.2.0 resolves `GROUP BY y
        // DISTRIBUTE BY c SORT BY y`), but grows no aggregate across it
        // (`... SORT BY max(x)` is UNRESOLVED_COLUMN): `through_to_agg`
        // stops at the node.
        r.widen = Some(Arc::new(Widen::Through { child: wide }));
        let mut r = if extended {
            let mut kept = self.restore(&r, &c.out);
            kept.hidden = Arc::clone(&c.hidden);
            kept
        } else {
            r
        };
        // Spark resolves an aggregate call in a sort only directly over the
        // aggregate: `GROUP BY y DISTRIBUTE BY y SORT BY count(*)` fails.
        r.agg = None;
        Ok(r)
    }

    /// `UNSUPPORTED_EXPR_FOR_OPERATOR` for the aggregate calls in `bounds`
    /// of an operator that has no aggregate to run them (a repartition, a
    /// sort over no aggregate); `None` when there is none.
    pub(super) fn misplaced_aggregates(
        &self,
        bounds: &[Bound],
        o: Origin,
    ) -> Option<AnalysisError> {
        let zone = self.ctx.conf.session_zone;
        let calls: Vec<String> = bounds
            .iter()
            .filter(|b| b.agg.is_some())
            .flat_map(|b| aggregate_calls(&b.shown()))
            .map(|c| format!("\"{}\"", pretty(&c, zone)))
            .collect();
        (!calls.is_empty()).then(|| {
            AnalysisError::new(
                UnsupportedExprForOperator {
                    invalid_expr_sqls: calls.join(", "),
                },
                o,
            )
        })
    }
}

/// The aggregate calls in `e`, left to right, outside subqueries and not
/// inside another call (Spark's `Invalid expressions` list).
fn aggregate_calls(e: &Arc<Expr>) -> Vec<Arc<Expr>> {
    let mut found = Vec::new();
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &x.node
            && f.kind == FunctionKind::Aggregate
        {
            found.push(x);
            continue;
        }
        let mut kids = Vec::new();
        x.for_each_child(&mut |k| kids.push(Arc::clone(k)));
        stack.extend(kids.into_iter().rev());
    }
    found
}
