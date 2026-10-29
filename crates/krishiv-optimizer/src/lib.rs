// SPDX-License-Identifier: Apache-2.0
//! The logical optimizer (layer L3; ADR 0012): resolved plan in, an
//! equivalent resolved plan out.
//!
//! Rules run in batches, each batch to a fixed point (at most
//! [`MAX_ITERATIONS`] rounds, as Spark's `spark.sql.optimizer.maxIterations`):
//!
//! 0. **Inline CTEs**, once: each reference to a deterministic (or
//!    singly referenced) `WITH` definition becomes a fresh instance of it
//!    (Spark's `InlineCTE`), so the rules below see through it.
//!    **Implied join filters**, once: each input of an inner join is
//!    filtered with what its condition's disjunctions imply on it alone.
//! 1. **Rewrite**: constant folding and boolean simplification, filter
//!    pushdown (through projections, aggregates onto their grouping keys,
//!    sorts, aliases, CTE bodies, self-join instances and joins, turning
//!    outer joins inner where a filter rejects their NULL rows), join
//!    conditions into the side they read, projection collapse and removal,
//!    limit pushdown.
//!    **Algebraic sum**, once: `SUM(x + c)` over a narrow integral column
//!    becomes `SUM(x) + c * COUNT(x)` (`sum_rewrite`).
//! 2. **Restrict**, then alternating with join order until neither
//!    changes the plan: an aggregate below a join that drops its unmatched
//!    groups groups only the input rows whose keys the join's other side
//!    holds (a semi join on that side's key domain, see `restrict`), when
//!    estimates say the domain is small.
//! 3. **Join order**: each tree of inner joins is re-planned from
//!    cardinality estimates ([`Statistics`]): exact dynamic programming up
//!    to [`DP_LIMIT`] inputs, greedy above, never a cross product the query
//!    does not have.
//! 4. **Prune**: column pruning over the whole plan (projection pushdown
//!    into scans), once.
//! 5. **Tidy**: the rewrite batch again over the pruned plan; prune and
//!    tidy repeat until neither changes the plan (folding can leave columns
//!    unread).
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
mod group_limit;
mod inline_cte;
mod join_order;
mod limit;
mod project;
mod prune;
mod pushdown;
mod restrict;
mod stats;
mod sum_rewrite;

use std::sync::Arc;

use krishiv_common::config::{ANSI_ENABLED, Config, SESSION_TIME_ZONE};
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_plan::{Expr, ExprNode, Grouping, LogicalPlan, Plan, PlanNode, ResolvedExpr};
use krishiv_udf_sdk::UdfLibrary;

pub use group_limit::{
    DEFAULT_WINDOW_GROUP_LIMIT_THRESHOLD, RankLikeFunction, WindowGroupLimit,
    infer_window_group_limit,
};
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

/// The rewrite batch, in order, with the rules' names.
const REWRITE: [(&str, Rule); 6] = [
    ("RemoveAlias", project::remove_alias),
    ("ConstantFolding", fold_node),
    ("PushDownFilter", pushdown::push_filter),
    ("PushDownJoinCondition", pushdown::push_join_condition),
    ("CollapseProject", project::collapse),
    ("PushDownLimit", limit::push_limit),
];

/// A rule application that changed the plan (`EXPLAIN VERBOSE`).
#[derive(Debug, Clone)]
pub struct RuleStep {
    /// The batch it ran in: `ImpliedJoinFilters`, `Rewrite`, `Restrict`,
    /// `JoinOrder` (join ordering alternating with restriction), `Prune` or
    /// `Tidy`.
    pub batch: &'static str,
    /// The round of a fixed-point batch, from 1.
    pub round: usize,
    /// The rule.
    pub rule: &'static str,
    /// The plan after it.
    pub plan: Arc<LogicalPlan>,
}

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
    Ok(run(plan, Some(stats), false, None)?.0)
}

/// As [`optimize_with`], also folding deterministic session-UDF calls of
/// literal arguments to their values (`udfs`, with the session settings in
/// `config`). A failure leaves the call in the plan, so the error surfaces
/// at execution.
///
/// # Errors
/// As [`optimize_with`].
pub fn optimize_with_udfs(
    plan: &Plan,
    stats: &dyn Statistics,
    udfs: &UdfLibrary,
    config: &Config,
) -> Result<Plan, KrishivError> {
    Ok(run(plan, Some(stats), false, Some((udfs, config)))?.0)
}

/// As [`optimize_with`], also returning every rule application that
/// changed the plan, in order (`EXPLAIN VERBOSE`). Tracing keeps each
/// changed plan (they share unchanged subtrees); the plan it returns is
/// the one [`optimize_with`] returns.
///
/// # Errors
/// As [`optimize_with`].
pub fn optimize_traced(
    plan: &Plan,
    stats: &dyn Statistics,
) -> Result<(Plan, Vec<RuleStep>), KrishivError> {
    run(plan, Some(stats), true, None)
}

/// As [`optimize_traced`], also folding deterministic session-UDF calls of
/// literal arguments to their values (see [`optimize_with_udfs`]).
///
/// # Errors
/// As [`optimize_with`].
pub fn optimize_traced_with_udfs(
    plan: &Plan,
    stats: &dyn Statistics,
    udfs: &UdfLibrary,
    config: &Config,
) -> Result<(Plan, Vec<RuleStep>), KrishivError> {
    run(plan, Some(stats), true, Some((udfs, config)))
}

/// As [`optimize_with`], keeping the joins in the order the query wrote
/// them (a baseline for the join order, as benchmarks compare it).
///
/// # Errors
/// As [`optimize_with`].
pub fn optimize_as_written(plan: &Plan) -> Result<Plan, KrishivError> {
    Ok(run(plan, None, false, None)?.0)
}

/// Records the steps that changed the plan, when tracing.
struct Trace {
    on: bool,
    steps: Vec<RuleStep>,
}

impl Trace {
    fn step(
        &mut self,
        batch: &'static str,
        round: usize,
        rule: &'static str,
        before: &Arc<LogicalPlan>,
        after: &Arc<LogicalPlan>,
    ) {
        if self.on && !Arc::ptr_eq(before, after) {
            self.steps.push(RuleStep {
                batch,
                round,
                rule,
                plan: Arc::clone(after),
            });
        }
    }

    /// Runs the one-off `rule` of `batch` and records it.
    fn once(
        &mut self,
        batch: &'static str,
        root: &Arc<LogicalPlan>,
        rule: impl FnOnce(&Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, KrishivError>,
    ) -> Result<Arc<LogicalPlan>, KrishivError> {
        let after = rule(root)?;
        self.step(batch, 1, batch, root, &after);
        Ok(after)
    }
}

fn run(
    plan: &Plan,
    stats: Option<&dyn Statistics>,
    tracing: bool,
    udfs: Option<(&UdfLibrary, &Config)>,
) -> Result<(Plan, Vec<RuleStep>), KrishivError> {
    if !plan.root().is_resolved() {
        return Err(InternalError {
            message: "the optimizer needs a resolved plan".to_owned(),
        }
        .into());
    }
    let root = Arc::clone(plan.root());
    let run = move || -> Result<(Arc<LogicalPlan>, Vec<RuleStep>), KrishivError> {
        let mut t = Trace {
            on: tracing,
            steps: Vec::new(),
        };
        let root = t.once("InlineCTE", &root, |r| inline_cte::inline(Arc::clone(r)))?;
        let root = t.once("ImpliedJoinFilters", &root, |r| {
            dag::rewrite_up(Arc::clone(r), &mut |p| {
                Ok(pushdown::imply_join_condition(p))
            })
        })?;
        // UDF calls of literals fold with the rewrite batch to a fixed
        // point: folding exposes the constants rewriting folds, and
        // rewriting exposes the literals folding folds (casts, arithmetic
        // and nested calls), so neither alone reaches the fixed point.
        // Only this pass needs the UDFs.
        let mut root = root;
        if let Some((library, config)) = udfs {
            for round in 1..=MAX_ITERATIONS {
                let before = Arc::clone(&root);
                let prev = Arc::clone(&root);
                root = dag::rewrite_up(Arc::clone(&root), &mut |p| {
                    Ok(fold::fold_plan(p, &fold_ctx(library, config)))
                })?;
                t.step("FoldUdfs", round, "FoldUdfs", &prev, &root);
                root = batch(root, "Rewrite", &mut t)?;
                if Arc::ptr_eq(&before, &root) {
                    break;
                }
            }
        } else {
            root = batch(root, "Rewrite", &mut t)?;
        }
        let root = t.once("AlgebraicSum", &root, |r| {
            sum_rewrite::rewrite(Arc::clone(r))
        })?;
        // Restriction and join ordering alternate until neither changes
        // the plan: ordering sees the restricted estimates, and restriction
        // sees every join the ordering forms (which side an aggregate joins
        // decides its domain), so a second optimization finds nothing new.
        let estimates = stats.unwrap_or(&NoStatistics);
        let mut root = t.once("Restrict", &root, |r| {
            restrict::restrict(Arc::clone(r), estimates)
        })?;
        if let Some(stats) = stats {
            for round in 1..=MAX_ITERATIONS {
                let ordered = join_order::reorder(&root, stats);
                t.step("JoinOrder", round, "JoinOrder", &root, &ordered);
                let restricted = restrict::restrict(Arc::clone(&ordered), estimates)?;
                t.step("JoinOrder", round, "Restrict", &ordered, &restricted);
                let done = Arc::ptr_eq(&restricted, &ordered);
                root = restricted;
                if done {
                    break;
                }
            }
        }
        let pruned = t.once("Prune", &root, |r| Ok(prune::prune(r)))?;
        let mut root = batch(Arc::clone(&pruned), "Tidy", &mut t)?;
        // Tidying can fold a filter to a constant and leave columns unread
        // (a `false` that reached a filter on `b` frees `b`), so after a tidy
        // that changed the plan, prune and tidy again until a tidy changes
        // nothing: then the plan is pruned and tidy, a fixed point. A tidy
        // batch that changes nothing returns its input, so the common case
        // (tidying a pruned plan changes nothing) costs no extra pass.
        let mut changed = !Arc::ptr_eq(&root, &pruned);
        let mut round = 2;
        while changed && round <= MAX_ITERATIONS {
            let pruned = prune::prune(&root);
            t.step("Prune", round, "Prune", &root, &pruned);
            let tidied = batch(Arc::clone(&pruned), "Tidy", &mut t)?;
            changed = !Arc::ptr_eq(&tidied, &pruned);
            root = tidied;
            round += 1;
        }
        Ok((root, t.steps))
    };
    // Passes recurse once per level and nest; rewrites may deepen the plan.
    let (root, steps) =
        krishiv_common::stack::on_big_stack(run).ok_or_else(|| InternalError {
            message: "cannot start an optimizer thread".to_owned(),
        })??;
    // A plan the rewrites made deeper than the IR allows runs unoptimised:
    // the input was valid, and the optimizer never makes a query fail.
    match Plan::try_new(root) {
        Ok(p) => Ok((p, steps)),
        Err(_) => Ok((plan.clone(), Vec::new())),
    }
}

/// The UDF folding context of one optimization: the session's library and
/// the plain values a folded UDF call may read.
fn fold_ctx<'a>(library: &'a UdfLibrary, config: &'a Config) -> fold::UdfFoldCtx<'a> {
    use krishiv_common::config::ConfigValue;
    use krishiv_kernels::cast::FixedOffset;

    let ansi = config
        .get(&ANSI_ENABLED)
        .and_then(ConfigValue::as_bool)
        .unwrap_or(true);
    let zone = config
        .get(&SESSION_TIME_ZONE)
        .and_then(|v| v.as_str())
        .and_then(|z| FixedOffset::parse(z).map(|o| o.0));
    fold::UdfFoldCtx {
        library,
        ansi,
        zone,
    }
}

/// Runs [`REWRITE`] to a fixed point.
fn batch(
    mut root: Arc<LogicalPlan>,
    name: &'static str,
    t: &mut Trace,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    for round in 1..=MAX_ITERATIONS {
        let before = Arc::clone(&root);
        for (rule_name, rule) in REWRITE {
            let prev = Arc::clone(&root);
            root = dag::rewrite_up(root, &mut |p| Ok(rule(p)))?;
            t.step(name, round, rule_name, &prev, &root);
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

pub(crate) fn groups_by_columns(g: &Grouping) -> bool {
    let column = |e: &Arc<Expr>| matches!(e.node, ExprNode::Resolved(ResolvedExpr::Column(_)));
    match g {
        Grouping::GroupBy(v) | Grouping::Rollup(v) | Grouping::Cube(v) => v.iter().all(column),
        Grouping::GroupingSets { exprs, sets } => {
            exprs.iter().chain(sets.iter().flatten()).all(column)
        }
        _ => false,
    }
}
