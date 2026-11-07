// SPDX-License-Identifier: Apache-2.0
//! The uncorrelated `EXISTS` or scalar subquery a join condition keeps (after the three-valued
//! `IN` rewrite, or written there): one constant for every row pair, so the left side carries
//! its value as a column (the executor evaluates subquery values in a projection, not in a join
//! condition). A scalar subquery over an empty left side is never evaluated, so its
//! `SCALAR_SUBQUERY_TOO_MANY_ROWS` is not raised there, where Spark raises it.

use std::collections::HashMap;
use std::sync::Arc;

use super::bind::addr;
use super::ops::col_expr;
use super::subquery::{R, Rewriter, alias, node, output, subqueries};
use crate::{BoundColumn, Expr, ExprNode, JoinKind, LogicalPlan, Origin, PlanNode, SubqueryKind};

impl Rewriter {
    /// Whether `e` holds an uncorrelated `EXISTS` or scalar subquery: what a join condition
    /// cannot evaluate where it stands, and [`Self::lift_exists`] moves to the left side.
    pub(super) fn liftable(&mut self, e: &Arc<Expr>) -> bool {
        subqueries(e).iter().any(|s| self.liftable_subquery(s))
    }

    /// Whether `sub` is an uncorrelated `EXISTS`, or a scalar subquery of one column.
    fn liftable_subquery(&mut self, sub: &Arc<Expr>) -> bool {
        let ExprNode::Subquery { kind, plan } = &sub.node else {
            return false;
        };
        if self.correlated(sub) {
            return false;
        }
        match kind {
            SubqueryKind::Exists => true,
            SubqueryKind::Scalar => output(plan).is_ok_and(|o| o.len() == 1),
            _ => false,
        }
    }

    /// The fresh column a liftable subquery `sub` becomes: a boolean for `EXISTS`, the value's
    /// own type (nullable) for a scalar.
    fn lift_column(&mut self, sub: &Arc<Expr>) -> Option<BoundColumn> {
        if !self.liftable_subquery(sub) {
            return None;
        }
        let ExprNode::Subquery { kind, plan } = &sub.node else {
            return None;
        };
        if matches!(kind, SubqueryKind::Exists) {
            return Some(JoinKind::existence_column(self.fresh()));
        }
        let value = output(plan).ok()?.into_iter().next()?;
        Some(BoundColumn {
            id: self.fresh(),
            name: "scalar".to_owned(),
            data_type: value.data_type,
            nullable: true,
        })
    }

    /// `conds` with each uncorrelated `EXISTS` or scalar subquery replaced by a column that
    /// `side` gains (a projection of its columns and the subquery values), and `side` with
    /// those columns.
    pub(super) fn lift_exists(
        &mut self,
        conds: Vec<Arc<Expr>>,
        side: Arc<LogicalPlan>,
    ) -> R<(Vec<Arc<Expr>>, Arc<LogicalPlan>)> {
        let mut lifted: Vec<Arc<Expr>> = Vec::new();
        let mut columns_of: HashMap<usize, Arc<Expr>> = HashMap::new();
        for c in &conds {
            for sub in subqueries(c) {
                if columns_of.contains_key(&addr(&*sub)) {
                    continue;
                }
                let Some(column) = self.lift_column(&sub) else {
                    continue;
                };
                lifted.push(alias(Arc::clone(&sub), &column));
                columns_of.insert(addr(&*sub), col_expr(&column, sub.origin));
            }
        }
        if lifted.is_empty() {
            return Ok((conds, side));
        }
        let conds = conds
            .into_iter()
            .map(|c| {
                c.transform_down(&mut |x: Arc<Expr>| {
                    Ok::<_, std::convert::Infallible>(
                        columns_of.get(&addr(&*x)).cloned().unwrap_or(x),
                    )
                })
                .unwrap_or_else(|never| match never {})
            })
            .collect();
        let mut exprs: Vec<Arc<Expr>> = output(&side)?
            .iter()
            .map(|c| col_expr(c, Origin::UNKNOWN))
            .collect();
        exprs.extend(lifted);
        let origin = side.origin;
        Ok((
            conds,
            node(PlanNode::Project { input: side, exprs }, origin),
        ))
    }
}
