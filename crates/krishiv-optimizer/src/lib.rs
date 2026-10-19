// SPDX-License-Identifier: Apache-2.0
//! The logical optimizer (layer L3; ADR 0012): resolved plan in, an
//! equivalent resolved plan out.
//!
//! Rules run in batches, each batch to a fixed point (at most
//! [`MAX_ITERATIONS`] rounds, as Spark's `spark.sql.optimizer.maxIterations`):
//!
//! 0. **Implied join filters**, once: each input of an inner join is
//!    filtered with what its condition's disjunctions imply on it alone.
//! 1. **Rewrite**: constant folding and boolean simplification, filter
//!    pushdown (through projections, aggregates onto their grouping keys,
//!    sorts, aliases, CTE bodies, self-join instances and joins, turning
//!    outer joins inner where a filter rejects their NULL rows), join
//!    conditions into the side they read, projection collapse and removal,
//!    limit pushdown.
//! 2. **Restrict**, once: an aggregate below a join that drops its
//!    unmatched groups groups only the input rows whose keys the join's
//!    other side holds (a semi join on that side's key domain, see
//!    `restrict`), when estimates say the domain is small.
//! 3. **Join order**: each tree of inner joins is re-planned from
//!    cardinality estimates ([`Statistics`]): exact dynamic programming up
//!    to [`DP_LIMIT`] inputs, greedy above, never a cross product the query
//!    does not have.
//! 4. **Prune**: column pruning over the whole plan (projection pushdown
//!    into scans), once.
//! 5. **Tidy**: the rewrite batch again over the pruned plan.
//!
//! Every rule applies inside subquery plans too, and rewrites a subtree
//! shared through `Arc` once, so the plan stays a DAG. A rule that changes
//! nothing returns its input (`Arc::ptr_eq`), which is how a batch detects
//! its fixed point. [`TopN::detect`] finds the `LIMIT` over `ORDER BY` the
//! physical planner runs as a top-N operator.
//!
//! The optimizer changes neither the output attributes nor the results;
//! the crate's tests check the TPC-H and per-rule plans against snapshots.

#![forbid(unsafe_code)]

mod dag;
mod expr;
mod fold;
mod join_order;
mod limit;
mod project;
mod prune;
mod pushdown;
mod restrict;
mod stats;

use std::sync::Arc;

use krishiv_common::error::{InternalError, KrishivError};
use krishiv_plan::{Expr, ExprNode, Grouping, LogicalPlan, Plan, PlanNode, ResolvedExpr};

pub use join_order::DP_LIMIT;
pub use limit::TopN;
pub use stats::{
    ColumnStats, DEFAULT_ROWS, DEFAULT_SELECTIVITY, NoStatistics, Statistics, TableStats,
    estimated_rows,
};

/// The most rounds a rule batch runs before its result is taken as is.
pub const MAX_ITERATIONS: usize = 100;

/// A rewrite of one relation whose inputs are already rewritten.
type Rule = fn(Arc<LogicalPlan>) -> Arc<LogicalPlan>;

/// The rewrite batch, in order.
const REWRITE: [Rule; 6] = [
    project::remove_alias,
    fold_node,
    pushdown::push_filter,
    pushdown::push_join_condition,
    project::collapse,
    limit::push_limit,
];

/// Optimises a resolved plan without statistics; see [`optimize_with`].
///
/// # Errors
/// As [`optimize_with`].
pub fn optimize(plan: &Plan) -> Result<Plan, KrishivError> {
    optimize_with(plan, &NoStatistics)
}

/// Optimises a resolved plan, ordering joins by `stats`.
///
/// # Errors
/// `INTERNAL_ERROR` when `plan` is not resolved or no optimizer thread can
/// be started. A rewrite nesting deeper than [`krishiv_plan::MAX_DEPTH`] is
/// not an error: the input plan is returned unoptimised.
pub fn optimize_with(plan: &Plan, stats: &dyn Statistics) -> Result<Plan, KrishivError> {
    run(plan, Some(stats))
}

/// As [`optimize_with`], keeping the joins in the order the query wrote
/// them (a baseline for the join order, as benchmarks compare it).
///
/// # Errors
/// As [`optimize_with`].
pub fn optimize_as_written(plan: &Plan) -> Result<Plan, KrishivError> {
    run(plan, None)
}

fn run(plan: &Plan, stats: Option<&dyn Statistics>) -> Result<Plan, KrishivError> {
    if !plan.root().is_resolved() {
        return Err(InternalError {
            message: "the optimizer needs a resolved plan".to_owned(),
        }
        .into());
    }
    let root = Arc::clone(plan.root());
    let run = move || -> Result<Arc<LogicalPlan>, KrishivError> {
        let root = dag::rewrite_up(root, &mut |p| Ok(pushdown::imply_join_condition(p)))?;
        let root = batch(root)?;
        // Before join ordering, which then sees the restricted estimates.
        let root = restrict::restrict(root, stats.unwrap_or(&NoStatistics))?;
        let root = match stats {
            Some(stats) => join_order::reorder(&root, stats),
            None => root,
        };
        let root = prune::prune(&root);
        batch(root)
    };
    // Passes recurse once per level and nest; rewrites may deepen the plan.
    let root = krishiv_common::stack::on_big_stack(run).ok_or_else(|| InternalError {
        message: "cannot start an optimizer thread".to_owned(),
    })??;
    // A plan the rewrites made deeper than the IR allows runs unoptimised:
    // the input was valid, and the optimizer never makes a query fail.
    Ok(Plan::try_new(root).unwrap_or_else(|_| plan.clone()))
}

/// Runs [`REWRITE`] to a fixed point.
fn batch(mut root: Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, KrishivError> {
    for _ in 0..MAX_ITERATIONS {
        let before = Arc::clone(&root);
        for rule in REWRITE {
            root = dag::rewrite_up(root, &mut |p| Ok(rule(p)))?;
        }
        if Arc::ptr_eq(&before, &root) {
            break;
        }
    }
    Ok(root)
}

/// Folds the expressions of one relation. An aggregate grouping by
/// expressions (not just columns) is left alone: its outputs refer to those
/// expressions by shape, and folding either side could break the match.
fn fold_node(p: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
    if let PlanNode::Aggregate { grouping, .. } = &p.node
        && !groups_by_columns(grouping)
    {
        return p;
    }
    let out = p.map_expressions(|e| Ok::<_, std::convert::Infallible>(fold::fold(&e)));
    match out {
        Ok(v) => v,
        Err(never) => match never {},
    }
}

fn groups_by_columns(g: &Grouping) -> bool {
    let column = |e: &Arc<Expr>| matches!(e.node, ExprNode::Resolved(ResolvedExpr::Column(_)));
    match g {
        Grouping::GroupBy(v) | Grouping::Rollup(v) | Grouping::Cube(v) => v.iter().all(column),
        Grouping::GroupingSets { exprs, sets } => {
            exprs.iter().chain(sets.iter().flatten()).all(column)
        }
        _ => false,
    }
}
