// SPDX-License-Identifier: Apache-2.0
//! `DISTRIBUTE BY` (Spark's `RepartitionByExpression`; `CLUSTER BY` and
//! `SORT BY` add a local sort above it).
//!
//! The expressions resolve against the input, a column a projection below
//! dropped included (Spark lets the node host missing attributes). They may
//! be anything a projection allows (a map, `rand()`, a literal) except an
//! aggregate, a window or a generator, and they may not read the outer
//! query. The node is then dropped, as the partitioning hints are: one node
//! has no partitions to set. Spark also evaluates the expressions while it
//! shuffles, so an expression that fails at run time (`x + 'a'`) fails the
//! query there; here it is never evaluated.

use std::sync::Arc;

use krishiv_common::error::{UnsupportedExprForOperator, UnsupportedGeneratorOutsideSelect};

use super::AnalysisError;
use super::exprs::{Bound, pretty};
use super::ops::has_generator;
use super::resolve::{Env, Rel, Resolver};
use super::subquery::correlated_reference;
use crate::{Expr, ExprNode, FunctionKind, LogicalPlan, Origin, ResolvedExpr};

/// Spark's operator name, as its errors show it.
const OPERATOR: &str = "RepartitionByExpression";

impl Resolver<'_> {
    /// Checks `exprs` against `input` and resolves to the input, which a
    /// sort above reads as it would without the node.
    pub(super) fn repartition(
        &mut self,
        p: &LogicalPlan,
        input: &Arc<LogicalPlan>,
        exprs: &[Arc<Expr>],
        env: &Arc<Env>,
    ) -> Result<Rel, AnalysisError> {
        let mut c = self.child(input, env)?;
        let (bounds, _) = self.bind_missing(&c, exprs, env, OPERATOR, false)?;
        let o = p.origin;
        if bounds.iter().any(|b| b.outer.is_some()) {
            let shown: Vec<Arc<Expr>> = bounds.iter().map(Bound::shown).collect();
            return Err(correlated_reference(&shown, o));
        }
        if let Some(b) = bounds.iter().find(|b| has_generator(&b.expr)) {
            return Err(AnalysisError::new(
                UnsupportedGeneratorOutsideSelect {
                    plan: format!("'{OPERATOR} [{}]'", pretty(&b.shown(), None)),
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
        // Spark resolves an aggregate call in a sort only directly over the
        // aggregate: `GROUP BY y DISTRIBUTE BY y SORT BY count(*)` fails.
        c.agg = None;
        Ok(c)
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
