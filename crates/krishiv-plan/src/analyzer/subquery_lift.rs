// SPDX-License-Identifier: Apache-2.0
//! The uncorrelated `EXISTS` a join condition keeps after the three-valued `IN` rewrite: one
//! constant for every row pair, so the left side carries its value as a column (the executor
//! evaluates subquery values in a projection, not in a join condition).

use std::collections::HashMap;
use std::sync::Arc;

use super::bind::addr;
use super::ops::col_expr;
use super::subquery::{R, Rewriter, alias, node, output, subqueries};
use crate::{Expr, ExprNode, JoinKind, LogicalPlan, Origin, PlanNode, SubqueryKind};

impl Rewriter {
    /// `conds` with each uncorrelated `EXISTS` replaced by a boolean column that `side` gains
    /// (a projection of its columns and the subquery values), and `side` with those columns.
    pub(super) fn lift_exists(
        &mut self,
        conds: Vec<Arc<Expr>>,
        side: Arc<LogicalPlan>,
    ) -> R<(Vec<Arc<Expr>>, Arc<LogicalPlan>)> {
        let mut lifted: Vec<Arc<Expr>> = Vec::new();
        let mut columns_of: HashMap<usize, Arc<Expr>> = HashMap::new();
        for c in &conds {
            for sub in subqueries(c) {
                if !matches!(
                    sub.node,
                    ExprNode::Subquery {
                        kind: SubqueryKind::Exists,
                        ..
                    }
                ) || self.correlated(&sub)
                    || columns_of.contains_key(&addr(&*sub))
                {
                    continue;
                }
                let column = JoinKind::existence_column(self.fresh());
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
