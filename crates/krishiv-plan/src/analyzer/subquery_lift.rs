// SPDX-License-Identifier: Apache-2.0
//! The uncorrelated `EXISTS` or scalar subquery a join condition keeps (after the three-valued
//! `IN` rewrite, or written there): one constant for every row pair, so the left side carries
//! its value as a column (the executor evaluates subquery values in a projection, not in a join
//! condition).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::bind::addr;
use super::ops::col_expr;
use super::subquery::{
    R, Rewriter, alias, conjuncts, inner_columns, node, output, reads_left, subqueries,
};
use crate::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, JoinKind, LogicalPlan, Origin,
    PlanNode, ResolvedExpr, SubqueryKind,
};

/// The conjuncts of a join condition after the lift, and the join's left and right inputs.
type Lifted = (Vec<Arc<Expr>>, Arc<LogicalPlan>, Arc<LogicalPlan>);

impl Rewriter {
    /// Whether a join of `kind` with condition `on` is rewritten by `join()`: it holds a subquery
    /// that becomes a join, or one whose lift the join needs (see [`Self::needs_lift`]), and its
    /// right side is not a LATERAL one (which reads the left and keeps the lateral rewrite).
    pub(super) fn joins_here(
        &mut self,
        on: &Arc<Expr>,
        kind: &JoinKind,
        left: &Arc<LogicalPlan>,
        right: &Arc<LogicalPlan>,
    ) -> bool {
        if self.existential(on) {
            return true;
        }
        if reads_left(left, right) {
            return false;
        }
        let (Ok(l), Ok(r)) = (output(left), output(right)) else {
            return false;
        };
        let ids = [
            l.iter().map(|c| c.id).collect::<HashSet<_>>(),
            r.iter().map(|c| c.id).collect::<HashSet<_>>(),
        ];
        conjuncts(on)
            .iter()
            .any(|c| self.needs_lift(c, kind, &ids[0], &ids[1]))
    }

    /// Whether the conjunct `c` of a join of `kind` holds an uncorrelated `EXISTS` or scalar
    /// subquery that the join cannot evaluate where it stands, so it must be lifted. A subquery
    /// the planner already pushes into one input as a filter is left alone, so those plans do not
    /// change: a FULL OUTER join takes none in its condition; any join takes none under an `OR`;
    /// a LEFT join takes none in a condition on the left (preserved) input alone, a RIGHT join
    /// none on the right alone.
    fn needs_lift(
        &mut self,
        c: &Arc<Expr>,
        kind: &JoinKind,
        left_ids: &HashSet<AttrId>,
        right_ids: &HashSet<AttrId>,
    ) -> bool {
        if !self.liftable(c) {
            return false;
        }
        let cols = inner_columns(c);
        let on_side =
            |ids: &HashSet<AttrId>| !cols.is_empty() && cols.iter().all(|x| ids.contains(&x.id));
        match kind {
            JoinKind::FullOuter => true,
            _ if has_or(c) => true,
            JoinKind::LeftOuter => on_side(left_ids),
            JoinKind::RightOuter => on_side(right_ids),
            _ => false,
        }
    }

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

    /// [`Self::lift_exists`] for both inputs of a join, for the conjuncts that need it (see
    /// [`Self::needs_lift`]): one that reads only `right`'s columns lifts onto the right input (so
    /// it can still be pushed into it as a filter), every other onto the left. Returns the
    /// conjuncts and the two inputs.
    pub(super) fn lift_sides(
        &mut self,
        conds: Vec<Arc<Expr>>,
        kind: &JoinKind,
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        ids: [&HashSet<AttrId>; 2],
    ) -> R<Lifted> {
        let (mut keep, mut to_left, mut to_right) = (Vec::new(), Vec::new(), Vec::new());
        for c in conds {
            if !self.needs_lift(&c, kind, ids[0], ids[1]) {
                keep.push(c);
                continue;
            }
            let cols = inner_columns(&c);
            if !cols.is_empty() && cols.iter().all(|x| ids[1].contains(&x.id)) {
                to_right.push(c);
            } else {
                to_left.push(c);
            }
        }
        let (to_left, left) = self.lift_exists(to_left, left)?;
        let (to_right, right) = self.lift_exists(to_right, right)?;
        keep.extend(to_left);
        keep.extend(to_right);
        Ok((keep, left, right))
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

/// Whether `e` has an `OR` anywhere in it.
fn has_or(e: &Arc<Expr>) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(e)];
    while let Some(x) = stack.pop() {
        if !seen.insert(addr(&*x)) {
            continue;
        }
        if matches!(&x.node, ExprNode::Resolved(ResolvedExpr::Function(f))
            if f.id == FunctionId::Builtin(Builtin::Or))
        {
            return true;
        }
        x.for_each_child(&mut |k| stack.push(Arc::clone(k)));
    }
    false
}
