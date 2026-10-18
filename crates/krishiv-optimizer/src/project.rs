// SPDX-License-Identifier: Apache-2.0
//! Projection rules: adjacent projections collapse into one (also into an
//! aggregate below), and a projection that passes its input through
//! unchanged disappears, as does an alias that renames nothing.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use krishiv_plan::{AttrId, Expr, ExprNode, Grouping, LogicalPlan, PlanNode, ResolvedExpr};

use crate::expr::{has_subquery, is_deterministic, references, strip_alias, substitute};

/// The rule at a `SubqueryAlias` that renames nothing: the analyzer has
/// resolved every name, so it is gone (Spark's `EliminateSubqueryAliases`).
pub(crate) fn remove_alias(node: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    match &node.node {
        PlanNode::SubqueryAlias { input, columns, .. } if columns.is_empty() => Arc::clone(input),
        _ => node,
    }
}

/// The rule at a `Project`.
pub(crate) fn collapse(node: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    let PlanNode::Project { input, exprs } = &node.node else {
        return node;
    };
    if let Some(out) = input.resolved_output()
        && exprs.len() == out.len()
        && exprs
            .iter()
            .zip(&out)
            .all(|(e, c)| matches!(&e.node, ExprNode::Resolved(ResolvedExpr::Column(x)) if x == c))
    {
        return Arc::clone(input);
    }
    let rebuilt = match &input.node {
        PlanNode::Project {
            input: below,
            exprs: inner,
        } => inline(exprs, inner, false).map(|exprs| PlanNode::Project {
            input: Arc::clone(below),
            exprs,
        }),
        PlanNode::Aggregate {
            input: below,
            grouping: grouping @ Grouping::GroupBy(_),
            aggregates,
        } => inline(exprs, aggregates, true).map(|aggregates| PlanNode::Aggregate {
            input: Arc::clone(below),
            grouping: grouping.clone(),
            aggregates,
        }),
        _ => None,
    };
    match rebuilt {
        Some(n) => Arc::new(LogicalPlan {
            node: n,
            origin: node.origin,
        }),
        None => node,
    }
}

/// `outer` with the aliases of `inner` inlined, when that evaluates nothing
/// more often than before: every inlined producer is deterministic and is
/// either cheap or used once. A subquery that reads an inlined alias, and
/// (into an aggregate) any subquery, blocks it.
fn inline(outer: &[Arc<Expr>], inner: &[Arc<Expr>], aggregate: bool) -> Option<Vec<Arc<Expr>>> {
    let producers: HashMap<AttrId, Arc<Expr>> = inner
        .iter()
        .filter_map(|e| match &e.node {
            ExprNode::Resolved(ResolvedExpr::Alias { output, .. }) => {
                Some((output.id, Arc::clone(strip_alias(e))))
            }
            _ => None,
        })
        .collect();
    let mut uses: HashMap<AttrId, usize> = HashMap::new();
    let mut seen = HashSet::new();
    for e in outer {
        let hits = references(e).iter().any(|id| producers.contains_key(id));
        if has_subquery(e) && (hits || aggregate) {
            return None;
        }
        count_uses(e, &mut uses, &mut seen);
    }
    for (id, n) in &uses {
        if let Some(p) = producers.get(id) {
            let cheap = matches!(
                p.node,
                ExprNode::Literal(_) | ExprNode::Resolved(ResolvedExpr::Column(_))
            );
            if !is_deterministic(p) || (!cheap && *n > 1) {
                return None;
            }
        }
    }
    Some(
        outer
            .iter()
            .map(|e| match &e.node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) => match producers.get(&c.id) {
                    Some(p) => Arc::new(Expr {
                        node: ExprNode::Resolved(ResolvedExpr::Alias {
                            expr: Arc::clone(p),
                            output: c.clone(),
                        }),
                        origin: e.origin,
                    }),
                    None => Arc::clone(e),
                },
                _ => substitute(e, &producers),
            })
            .collect(),
    )
}

/// Adds to `uses` how often each column is read by `e`. A subexpression
/// shared through `Arc` is walked once; its columns then count as read
/// twice.
fn count_uses(e: &Arc<Expr>, uses: &mut HashMap<AttrId, usize>, seen: &mut HashSet<*const Expr>) {
    let mut stack = vec![e];
    while let Some(x) = stack.pop() {
        if !seen.insert(Arc::as_ptr(x)) {
            for id in references(x) {
                *uses.entry(id).or_default() += 2;
            }
            continue;
        }
        if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &x.node {
            *uses.entry(c.id).or_default() += 1;
        }
        x.for_each_child(&mut |c| stack.push(c));
    }
}
