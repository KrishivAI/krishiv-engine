// SPDX-License-Identifier: Apache-2.0
//! A fresh instance of a correlated subtree, for the second side of a
//! self-join or set operation inside a subquery (two correlated filters
//! over one view, `... WHERE k = outer.k INTERSECT ... WHERE k = outer.k`).
//!
//! A [`PlanNode::Shift`] renumbers every id inside it, so it never stands
//! over an outer reference. Here the closed subtrees below the correlation
//! (the shared view) get a `Shift`, and the correlated operators above them
//! are rebuilt with their own columns renumbered by the same offset; outer
//! references keep their ids. Only filters, projections, `GROUP BY`
//! aggregates, sorts, limits, offsets and aliases are rebuilt, over
//! expressions without subqueries or lambdas: any other correlated operator
//! stays refused.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;

use super::AnalysisError;
use super::bind::{addr, unsupported};
use super::resolve::{Rel, Resolver};
use super::subquery::has_outer;
use crate::walk::{Node, visit_distinct};
use crate::{
    AttrId, BoundColumn, Expr, ExprNode, Grouping, LogicalPlan, PlanNode, ResolvedExpr, SortOrder,
};

type R<T> = Result<T, AnalysisError>;

impl Resolver<'_> {
    /// [`Self::instance`] for a subtree that reads an enclosing query: the
    /// ids it defines move to fresh ones, its outer references stay.
    pub(super) fn correlated_instance(&mut self, c: &Rel) -> R<Rel> {
        let (lo, hi) = c.ids.unwrap_or((0, 0));
        let offset = self.next_id.saturating_sub(lo);
        let mut memo = HashMap::new();
        let plan = renumbered(&c.plan, Span { lo, hi, offset }, &mut memo)?;
        let mut r = self.shifted(c, offset, Some(plan))?;
        self.next_id += hi - lo + 1;
        r.outer = c.outer;
        // A `_metadata` widening would shift the correlated plan whole.
        r.widen = None;
        Ok(r)
    }
}

/// The ids a subtree defines, `lo..=hi`, and what each gains.
#[derive(Clone, Copy)]
pub(super) struct Span {
    pub(super) lo: u64,
    pub(super) hi: u64,
    pub(super) offset: u64,
}

impl Span {
    pub(super) fn column(self, c: &BoundColumn) -> BoundColumn {
        let id = if (self.lo..=self.hi).contains(&c.id.0) {
            AttrId(c.id.0 + self.offset)
        } else {
            c.id
        };
        BoundColumn { id, ..c.clone() }
    }

    /// `e` reading the instance's columns; `None` when it holds a
    /// subquery or a lambda, whose ids this rewrite does not follow.
    fn expr(self, e: &Arc<Expr>) -> Option<Arc<Expr>> {
        let foreign = visit_distinct(Node::Expr(e), |n| match n {
            Node::Expr(x)
                if matches!(x.node, ExprNode::Subquery { .. } | ExprNode::Lambda { .. }) =>
            {
                std::ops::ControlFlow::Break(())
            }
            _ => std::ops::ControlFlow::Continue(()),
        });
        if foreign.is_break() {
            return None;
        }
        let out = Arc::clone(e).transform_up(&mut |x: Arc<Expr>| {
            let node = match &x.node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) => {
                    ExprNode::Resolved(ResolvedExpr::Column(self.column(c)))
                }
                ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => {
                    ExprNode::Resolved(ResolvedExpr::Alias {
                        expr: Arc::clone(expr),
                        output: self.column(output),
                    })
                }
                _ => return Ok::<_, Infallible>(x),
            };
            Ok(Arc::new(Expr::new(node).with_origin(x.origin)))
        });
        match out {
            Ok(x) => Some(x),
            Err(never) => match never {},
        }
    }
}

/// `p` with its closed subtrees under a `Shift` by `s.offset` and the
/// correlated operators above them reading the shifted ids.
pub(super) fn renumbered(
    p: &Arc<LogicalPlan>,
    s: Span,
    memo: &mut HashMap<usize, Arc<LogicalPlan>>,
) -> R<Arc<LogicalPlan>> {
    if let Some(done) = memo.get(&addr(&**p)) {
        return Ok(Arc::clone(done));
    }
    let out = if has_outer(Node::Plan(p)) {
        let refused = || unsupported("a self-join of a correlated subtree", p.origin);
        let ex = |e: &Arc<Expr>| s.expr(e).ok_or_else(refused);
        let node = match &p.node {
            PlanNode::Filter { input, predicate } => PlanNode::Filter {
                input: renumbered(input, s, memo)?,
                predicate: ex(predicate)?,
            },
            PlanNode::Project { input, exprs } => PlanNode::Project {
                input: renumbered(input, s, memo)?,
                exprs: exprs.iter().map(ex).collect::<R<_>>()?,
            },
            PlanNode::Sort {
                input,
                order,
                global,
            } => PlanNode::Sort {
                input: renumbered(input, s, memo)?,
                order: order
                    .iter()
                    .map(|o| {
                        Ok(SortOrder {
                            expr: ex(&o.expr)?,
                            ..o.clone()
                        })
                    })
                    .collect::<R<_>>()?,
                global: *global,
            },
            PlanNode::Aggregate {
                input,
                grouping: Grouping::GroupBy(keys),
                aggregates,
            } => PlanNode::Aggregate {
                input: renumbered(input, s, memo)?,
                grouping: Grouping::GroupBy(keys.iter().map(ex).collect::<R<_>>()?),
                aggregates: aggregates.iter().map(ex).collect::<R<_>>()?,
            },
            PlanNode::Limit { input, limit } => PlanNode::Limit {
                input: renumbered(input, s, memo)?,
                limit: Arc::clone(limit),
            },
            PlanNode::Repartition { input, exprs } => PlanNode::Repartition {
                input: renumbered(input, s, memo)?,
                exprs: exprs.iter().map(ex).collect::<R<_>>()?,
            },
            PlanNode::Offset { input, offset } => PlanNode::Offset {
                input: renumbered(input, s, memo)?,
                offset: Arc::clone(offset),
            },
            PlanNode::Sample {
                input,
                bounds,
                seed,
            } => PlanNode::Sample {
                input: renumbered(input, s, memo)?,
                bounds: *bounds,
                seed: *seed,
            },
            PlanNode::SubqueryAlias {
                input,
                alias,
                columns,
            } => PlanNode::SubqueryAlias {
                input: renumbered(input, s, memo)?,
                alias: alias.clone(),
                columns: columns.clone(),
            },
            _ => return Err(refused()),
        };
        Arc::new(LogicalPlan::new(node).with_origin(p.origin))
    } else {
        Arc::new(LogicalPlan::new(PlanNode::Shift {
            input: Arc::clone(p),
            offset: s.offset,
        }))
    };
    memo.insert(addr(&**p), Arc::clone(&out));
    Ok(out)
}
