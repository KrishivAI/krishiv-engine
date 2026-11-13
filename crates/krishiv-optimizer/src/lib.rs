// SPDX-License-Identifier: Apache-2.0
//! The logical optimizer (layer L3; ADR 0012): resolved plan in, an
//! equivalent resolved plan out.
//!
//! Rules run in batches, each batch to a fixed point (at most
//! [`MAX_ITERATIONS`] rounds, as Spark's `spark.sql.optimizer.maxIterations`):
//!
//! 0. **Collation keys**, once: the key positions of a collated value compare its key
//!    (`collation_keys`).
//!    **Inline CTEs**, once: each reference to a deterministic (or
//!    singly referenced) `WITH` definition becomes a fresh instance of it
//!    (Spark's `InlineCTE`), so the rules below see through it.
//!    **Implied join filters**, once: each input of an inner join is
//!    filtered with what its condition's disjunctions imply on it alone.
//! 1. **Rewrite**: constant folding and boolean simplification, filter
//!    pushdown (through projections, aggregates onto their grouping keys,
//!    sorts, aliases, CTE bodies, self-join instances and joins, turning
//!    outer joins inner where a filter rejects their NULL rows), join
//!    conditions into the side they read, projection collapse and removal,
//!    limit pushdown, a `DISTRIBUTE BY` directly over another dropping the
//!    inner one (Spark's `CollapseRepartition`).
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
//!    **Group-key code**, once, with statistics (`group_code`, design P2b
//!    D2): an aggregate's keys from one dimension travel as one integer
//!    code, decoded above it.
//! 4. **Prune**: column pruning over the whole plan (projection pushdown
//!    into scans), once.
//! 5. **Tidy**: the rewrite batch again over the pruned plan; prune and
//!    tidy repeat until neither changes the plan (folding can leave columns
//!    unread).
//! 6. **CSE** (`cse`, once): common-subexpression elimination within one
//!    operator; disabled by [`Options::cse`].
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

mod collation_keys;
mod constraints;
mod cse;
mod dag;
mod eager;
mod estimate;
mod expr;
mod fold;
mod group_code;
mod group_limit;
mod inline_cte;
mod join_order;
mod limit;
mod narrow;
mod plan_stats;
mod project;
mod prune;
mod pushdown;
mod repartition;
mod restrict;
mod share_cte;
mod stats;
mod sum_rewrite;
mod win_magic;

use std::sync::Arc;

use krishiv_common::config::{
    ANSI_ENABLED, Config, ConfigValue, EAGER_AGGREGATION, GROUP_KEY_CODE, JOIN_COST, JOIN_DP_LIMIT,
    JOIN_ENUMERATOR, SESSION_TIME_ZONE, SHARE_CTE_INSTANCES, SHARE_REPEATED_SUBTREES,
    SUBQUERY_AS_WINDOW,
};
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_plan::{
    AttrId, BoundColumn, Expr, ExprNode, Grouping, LogicalPlan, Origin, Plan, PlanNode,
    ResolvedExpr,
};
use krishiv_udf_sdk::UdfLibrary;

pub use group_limit::{
    DEFAULT_WINDOW_GROUP_LIMIT_THRESHOLD, RankLikeFunction, WindowGroupLimit,
    infer_window_group_limit,
};
pub use join_order::DP_LIMIT;
pub use limit::TopN;
pub use plan_stats::PlanStats;
pub use stats::{
    AttrStats, ColumnStats, DEFAULT_ROWS, DEFAULT_SELECTIVITY, NoStatistics, RelStats, StatSource,
    Statistics, TableStats, estimated_rows,
};

/// The most rounds a rule batch runs before its result is taken as is.
pub const MAX_ITERATIONS: usize = 100;

/// A rewrite of one relation whose inputs are already rewritten.
type Rule = fn(Arc<LogicalPlan>) -> Arc<LogicalPlan>;

/// The rewrite batch, in order, with the rules' names.
const REWRITE: [(&str, Rule); 7] = [
    ("RemoveAlias", project::remove_alias),
    ("CollapseRepartition", repartition::collapse),
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
    /// `JoinOrder` (join ordering alternating with restriction; a `Rewrite`
    /// batch also runs inside it after an outer join turned inner, its rounds
    /// counting from 1 each time; a `Rewrite` batch also follows
    /// `InferJoinFilters` whenever that rule changed the plan), `Prune`, `Tidy` or `Cse`.
    pub batch: &'static str,
    /// The round of a fixed-point batch, from 1.
    pub round: usize,
    /// The rule.
    pub rule: &'static str,
    /// The plan after it.
    pub plan: Arc<LogicalPlan>,
    /// What the rule decided and why, when it says (a rule that declines
    /// leaves the plan as it was); else empty.
    pub note: String,
}

/// The default of `krishiv.optimizer.shareRepeatedSubtrees` when no
/// configuration value is set (on since the P3 W1 flip, unit 6).
const SHARE_REPEATED_SUBTREES_DEFAULT: bool = true;

/// A test-only switch per optimizer rule: which rewrites run. All are on
/// by default; the differential harnesses turn one off at a time and
/// compare results with it on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Options {
    /// Common-subexpression elimination within one operator (`cse`).
    pub cse: bool,
    /// Statistics propagation: estimates carry column ranges, null
    /// fractions and distinct counts through filters, joins, aggregates,
    /// projections and unions. Off, only row counts scale (the estimates
    /// that order joins and choose restrictions).
    pub stats_propagation: bool,
    /// Eager aggregation (`eager`): aggregates run below joins where the
    /// statistics say it pays. Needs statistics; a session configuration,
    /// when given, can switch it off.
    pub eager_aggregation: bool,
    /// Subquery as window (`win_magic`, P3 W1 shape (a)): a correlated
    /// aggregate subquery over the outer block's own join runs as a window.
    /// Runs only when `krishiv.optimizer.subqueryAsWindow` is also on.
    pub subquery_as_window: bool,
    /// Sharing a subtree the main block and a subquery both read (`ShareCte`
    /// direct members, P3 W1 shape (b)). Runs only when
    /// `krishiv.optimizer.shareRepeatedSubtrees` is also on.
    pub share_repeated_subtrees: bool,
    /// Group-key codes (`group_code`, design P2b D2): an aggregate's keys
    /// from one dimension travel as one integer code. Needs statistics; a
    /// session configuration, when given, must also switch it on
    /// (`krishiv.optimizer.groupKeyCode`).
    pub group_key_code: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            cse: true,
            stats_propagation: true,
            eager_aggregation: true,
            subquery_as_window: true,
            share_repeated_subtrees: true,
            group_key_code: true,
        }
    }
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
/// not an error: the input plan is returned unoptimised, with only its
/// collation keys; `FAILED_TO_PARSE_TOO_COMPLEX` when those keys alone nest
/// past the cap. `KRISHIV_UNSUPPORTED_FEATURE` for a collated value the
/// engine has no key form for yet, in a plan that did not come through the
/// analyzer (whose guard refuses it first).
pub fn optimize_with(plan: &Plan, stats: &dyn Statistics) -> Result<Plan, KrishivError> {
    optimize_with_options(plan, stats, &Options::default())
}

/// As [`optimize_with`], with per-rule switches.
///
/// # Errors
/// As [`optimize_with`].
pub fn optimize_with_options(
    plan: &Plan,
    stats: &dyn Statistics,
    options: &Options,
) -> Result<Plan, KrishivError> {
    Ok(run(plan, Some(stats), false, None, options)?.0)
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
    Ok(run(
        plan,
        Some(stats),
        false,
        Some((udfs, config)),
        &Options::default(),
    )?
    .0)
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
    run(plan, Some(stats), true, None, &Options::default())
}

/// As [`optimize_traced`], with per-rule switches.
///
/// # Errors
/// As [`optimize_with`].
pub fn optimize_traced_with_options(
    plan: &Plan,
    stats: &dyn Statistics,
    options: &Options,
) -> Result<(Plan, Vec<RuleStep>), KrishivError> {
    run(plan, Some(stats), true, None, options)
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
    run(
        plan,
        Some(stats),
        true,
        Some((udfs, config)),
        &Options::default(),
    )
}

/// As [`optimize_with`], keeping the joins in the order the query wrote
/// them (a baseline for the join order, as benchmarks compare it).
///
/// # Errors
/// As [`optimize_with`].
pub fn optimize_as_written(plan: &Plan) -> Result<Plan, KrishivError> {
    Ok(run(plan, None, false, None, &Options::default())?.0)
}

/// As [`optimize_as_written`], with per-rule switches.
///
/// # Errors
/// As [`optimize_with`].
pub fn optimize_as_written_with_options(
    plan: &Plan,
    options: &Options,
) -> Result<Plan, KrishivError> {
    Ok(run(plan, None, false, None, options)?.0)
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
                note: String::new(),
            });
        }
    }

    /// Records what a rule decided, one step per note, on the plan it left.
    fn notes(&mut self, rule: &'static str, after: &Arc<LogicalPlan>, notes: Vec<String>) {
        if self.on {
            self.steps.extend(notes.into_iter().map(|note| RuleStep {
                batch: rule,
                round: 1,
                rule,
                plan: Arc::clone(after),
                note,
            }));
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

/// What one optimizer run gives: the plan with only its collation keys, the optimized plan and
/// the rule steps.
type Ran = (Arc<LogicalPlan>, Arc<LogicalPlan>, Vec<RuleStep>);

fn run(
    plan: &Plan,
    stats: Option<&dyn Statistics>,
    tracing: bool,
    udfs: Option<(&UdfLibrary, &Config)>,
    options: &Options,
) -> Result<(Plan, Vec<RuleStep>), KrishivError> {
    if !plan.root().is_resolved() {
        return Err(InternalError {
            message: "the optimizer needs a resolved plan".to_owned(),
        }
        .into());
    }
    let root = Arc::clone(plan.root());
    let run = move || -> Result<Ran, KrishivError> {
        let mut t = Trace {
            on: tracing,
            steps: Vec::new(),
        };
        // Collated keys first: every later rule then sees operators that compare bytes.
        let root = t.once("CollationKeys", &root, |r| {
            collation_keys::rewrite(Arc::clone(r))
        })?;
        let keyed = Arc::clone(&root);
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
        // Join ordering's settings: the window rule's guard orders with them too.
        let joins = join_settings(udfs.map(|(_, config)| config), tracing);
        // P3 W1 shape (a): a correlated aggregate subquery over the outer block's own
        // relations becomes a window (`krishiv.optimizer.subqueryAsWindow`, on by default).
        let window = options.subquery_as_window
            && udfs.is_some_and(|(_, config)| {
                config
                    .get(&SUBQUERY_AS_WINDOW)
                    .and_then(ConfigValue::as_bool)
                    .unwrap_or(false)
            });
        let root = if window {
            let mut notes = Vec::new();
            let after = t.once("SubqueryAsWindow", &root, |r| {
                let (after, n) = win_magic::rewrite(
                    Arc::clone(r),
                    stats.unwrap_or(&NoStatistics),
                    options.stats_propagation,
                    joins,
                )?;
                notes = n;
                Ok(after)
            })?;
            t.notes("SubqueryAsWindow", &after, notes);
            after
        } else {
            root
        };
        // After the rewrite batch, which moves `WHERE` equalities into the
        // joins (comma joins are cross joins before it); the batch runs
        // again to push what the rule added, only when it added something.
        // What it added can turn an outer join inner or one-sided (an `isnotnull` on
        // its null-supplied side), which lets the batch move that join's `ON`
        // conjuncts into the side they read; the rule then runs over the joins
        // below them in join ordering (or once more after restriction), not
        // only in a second optimization.
        let inferred = t.once("InferJoinFilters", &root, infer_join_filters)?;
        let root = if Arc::ptr_eq(&inferred, &root) {
            inferred
        } else {
            batch(inferred, "Rewrite", &mut t)?
        };
        let root = t.once("AlgebraicSum", &root, |r| {
            sum_rewrite::rewrite(Arc::clone(r))
        })?;
        // Restriction and join ordering alternate until neither changes
        // the plan: ordering sees the restricted estimates, and restriction
        // sees every join the ordering forms (which side an aggregate joins
        // decides its domain), and an outer join their filters make inner
        // is ordered with its tree, so a second optimization finds nothing new.
        let estimates = stats.unwrap_or(&NoStatistics);
        let mut root = t.once("Restrict", &root, |r| {
            restrict::restrict(Arc::clone(r), estimates, options.stats_propagation)
        })?;
        if let Some(stats) = stats {
            root = order_and_restrict(root, stats, options.stats_propagation, joins, &mut t)?;
        }
        // After the plan's joins are final, so the rule reads the join it
        // moves below; before pruning, which drops what it leaves unread.
        let eager = options.eager_aggregation
            && udfs.is_none_or(|(_, config)| {
                config
                    .get(&EAGER_AGGREGATION)
                    .and_then(ConfigValue::as_bool)
                    .unwrap_or(true)
            });
        let root = match stats {
            Some(stats) if eager => {
                let (after, notes) =
                    eager::rewrite(Arc::clone(&root), stats, options.stats_propagation)?;
                t.notes("EagerAggregate", &after, notes);
                // The moved aggregate changes what its join's inputs estimate,
                // and so their canonical order.
                if Arc::ptr_eq(&after, &root) {
                    after
                } else {
                    order_and_restrict(after, stats, options.stats_propagation, joins, &mut t)?
                }
            }
            _ => root,
        };
        // After eager aggregation (which may pre-empt it) and the final join
        // order, before pruning: an aggregate's dimension keys travel as one
        // code (design P2b D2).
        // `krishiv.experimental.groupKeyCodeForce` (measurement only) also
        // switches it on, without the statistics test.
        let forced = udfs.is_some_and(|(_, config)| {
            krishiv_common::experimental::is_enabled(config, "groupKeyCodeForce")
        });
        let coded = options.group_key_code
            && (forced
                || udfs.is_none_or(|(_, config)| {
                    config
                        .get(&GROUP_KEY_CODE)
                        .and_then(ConfigValue::as_bool)
                        .unwrap_or(false)
                }));
        let root = match stats {
            Some(stats) if coded => {
                let (after, notes) = group_code::rewrite(
                    Arc::clone(&root),
                    stats,
                    options.stats_propagation,
                    forced,
                )?;
                t.step("GroupKeyCode", 1, "GroupKeyCode", &root, &after);
                t.notes("GroupKeyCode", &after, notes);
                // The decode join is a new inner join: ordered (and its
                // inputs' canonical order settled) here, as a second
                // optimization would.
                if Arc::ptr_eq(&after, &root) {
                    after
                } else {
                    order_and_restrict(after, stats, options.stats_propagation, joins, &mut t)?
                }
            }
            _ => root,
        };
        // Without join ordering, once more after restriction (its semi
        // joins); with it, `order_and_restrict` already ran the rule on the
        // final joins. Tidying pushes what it adds.
        // The rule and the batch alternate until the rule adds nothing, as in
        // `order_and_restrict`: what it adds can make another outer join
        // one-sided, whose `ON` conjuncts the batch then moves.
        let root = if stats.is_none() {
            let mut root = root;
            for _ in 1..=MAX_ITERATIONS {
                let inferred = t.once("InferJoinFilters", &root, infer_join_filters)?;
                if Arc::ptr_eq(&inferred, &root) {
                    break;
                }
                root = batch(inferred, "Rewrite", &mut t)?;
            }
            root
        } else {
            root
        };
        let pruned = t.once("Prune", &root, |r| Ok(prune::prune(r)))?;
        let mut root = batch(Arc::clone(&pruned), "Tidy", &mut t)?;
        // Tidying can fold a filter to a constant `false` and leave columns
        // unread (a `false` that reached a filter on `b` frees `b`; constants
        // pass grouped aggregates), so after a tidy that changed the plan and
        // left such a filter, prune and tidy again until nothing changes:
        // then the plan is a fixed point. Other plans (no `Filter false`, or
        // a tidy that changed nothing) take no extra pass.
        let mut changed = !Arc::ptr_eq(&root, &pruned) && has_false_filter(&root);
        let mut round = 2;
        while changed && round <= MAX_ITERATIONS {
            let pruned = prune::prune(&root);
            t.step("Prune", round, "Prune", &root, &pruned);
            if Arc::ptr_eq(&pruned, &root) {
                break;
            }
            let tidied = batch(Arc::clone(&pruned), "Tidy", &mut t)?;
            changed = !Arc::ptr_eq(&tidied, &pruned);
            root = tidied;
            round += 1;
        }
        // Common-subexpression elimination runs once after the final
        // rewrite batch: the conjunct order of filters above is then the
        // order that executes, so "first operand" is execution order.
        let root = if options.cse {
            let before = Arc::clone(&root);
            let origin = before.origin;
            let output = before.resolved_output();
            let after = cse::eliminate(root);
            let after = restore_output(after, &output, origin);
            t.step("Cse", 1, "CommonSubexpressionElimination", &before, &after);
            // Reading a `cse_*` column in place of the expression it names
            // can leave a column unread (an aggregate output, a join side's
            // column): prune again so that a second optimization finds
            // nothing to remove.
            let pruned = prune::prune(&after);
            t.step("Cse", 1, "Prune", &after, &pruned);
            if Arc::ptr_eq(&pruned, &after) {
                after
            } else {
                batch(pruned, "Tidy", &mut t)?
            }
        } else {
            root
        };
        // Identical deterministic CTE instances share one computation
        // (P2-T198): only where a session configuration is given, and the
        // switch is on.
        let share = udfs.is_some_and(|(_, config)| {
            config
                .get(&SHARE_CTE_INSTANCES)
                .and_then(ConfigValue::as_bool)
                .unwrap_or(true)
        });
        // P3 W1 shape (b): the places that read a shared body directly
        // count as members too (`krishiv.optimizer.shareRepeatedSubtrees`).
        let direct = options.share_repeated_subtrees
            && udfs.is_some_and(|(_, config)| {
                config
                    .get(&SHARE_REPEATED_SUBTREES)
                    .and_then(ConfigValue::as_bool)
                    .unwrap_or(SHARE_REPEATED_SUBTREES_DEFAULT)
            });
        let root = if share {
            t.once("ShareCte", &root, |r| {
                share_cte::share(Arc::clone(r), direct)
            })?
        } else {
            root
        };
        Ok((keyed, root, t.steps))
    };
    // Passes recurse once per level and nest; rewrites may deepen the plan.
    let (keyed, root, steps) =
        krishiv_common::stack::on_big_stack(run).ok_or_else(|| InternalError {
            message: "cannot start an optimizer thread".to_owned(),
        })??;
    match Plan::try_new(root) {
        Ok(p) => Ok((p, steps)),
        Err(_) => Ok((unoptimised(plan, keyed)?, Vec::new())),
    }
}

/// The plan to run when the rewrites made `plan` deeper than the IR allows: the input with only
/// its collation keys (`keyed`, the first rule's output), so a collated value still compares by
/// its key. Without collations that is the input itself, which was valid.
///
/// # Errors
/// `FAILED_TO_PARSE_TOO_COMPLEX` when the keys alone nest past the cap: running the unkeyed
/// input would compare collated values as bytes.
fn unoptimised(plan: &Plan, keyed: Arc<LogicalPlan>) -> Result<Plan, KrishivError> {
    if Arc::ptr_eq(&keyed, plan.root()) {
        return Ok(plan.clone());
    }
    Plan::try_new(keyed)
}

/// [`constraints::infer_join_filters`] at every join, bottom up.
fn infer_join_filters(root: &Arc<LogicalPlan>) -> Result<Arc<LogicalPlan>, KrishivError> {
    let mut memo = constraints::Memo::for_root(root);
    dag::rewrite_up_traced(Arc::clone(root), &mut |original, p| {
        // A shared relation rebuilt by this pass (a join inside it got its
        // own filters) is still shared.
        memo.carry_shared(original, &p);
        let out = constraints::infer_join_filters(p, &mut memo);
        memo.carry_shared(original, &out);
        Ok(out)
    })
}

/// The join-ordering settings of a session configuration (defaults
/// without one), describing each tree when `trace`.
fn join_settings(config: Option<&Config>, trace: bool) -> join_order::Settings {
    let mut s = join_order::Settings {
        trace,
        ..join_order::Settings::default()
    };
    if let Some(config) = config {
        if let Some(name) = config.get(&JOIN_ENUMERATOR).and_then(ConfigValue::as_str) {
            s.enumerator = join_order::Enumerator::parse(name);
        }
        if let Some(name) = config.get(&JOIN_COST).and_then(ConfigValue::as_str) {
            s.cost = join_order::JoinCost::parse(name);
        }
        if let Some(limit) = config.get(&JOIN_DP_LIMIT).and_then(ConfigValue::as_int) {
            s.dp_limit = usize::try_from(limit)
                .unwrap_or(join_order::MAX_DP_LIMIT)
                .min(join_order::MAX_DP_LIMIT);
        }
    }
    s
}

/// Join ordering alternating with restriction and inferred join filters,
/// until neither changes the plan. The trees of the last ordering, the
/// plan's own, are described in the trace.
fn order_and_restrict(
    mut root: Arc<LogicalPlan>,
    stats: &dyn Statistics,
    propagate: bool,
    joins: join_order::Settings,
    t: &mut Trace,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    let mut notes = Vec::new();
    for round in 1..=MAX_ITERATIONS {
        let input = Arc::clone(&root);
        let (ordered, trees) = join_order::reorder(&root, stats, propagate, joins);
        notes = trees;
        t.step("JoinOrder", round, "JoinOrder", &root, &ordered);
        let restricted = restrict::restrict(Arc::clone(&ordered), stats, propagate)?;
        t.step("JoinOrder", round, "Restrict", &ordered, &restricted);
        // A filter ordering or restriction leaves can reject the NULL rows of an outer join's
        // null-supplied side (an equality implied across two joins, `a = c` from `a = b` and
        // `b = c`, over the outer join below). Such a join turns inner here, and the rewrite batch
        // then settles the filters, so the next ordering sees the larger inner-join tree instead of
        // a second optimization finding it. Nothing else runs, and nothing at all unless a join
        // turned inner.
        let narrowed = dag::rewrite_up(Arc::clone(&restricted), &mut |p| {
            Ok(pushdown::eliminate_outer_joins(p))
        })?;
        let rewritten = if Arc::ptr_eq(&narrowed, &restricted) {
            restricted
        } else {
            t.step(
                "JoinOrder",
                round,
                "EliminateOuterJoins",
                &restricted,
                &narrowed,
            );
            batch(narrowed, "Rewrite", t)?
        };
        // The filters of the joins ordering and restriction formed: ordering
        // keys its canonical input order on them, so they are part of the
        // loop.
        let inferred = infer_join_filters(&rewritten)?;
        t.step(
            "JoinOrder",
            round,
            "InferJoinFilters",
            &rewritten,
            &inferred,
        );
        let mut done = Arc::ptr_eq(&inferred, &ordered);
        // Pushed down and merged before ordering reads them again, so the
        // next round sees the inputs a second `optimize` would see.
        root = if Arc::ptr_eq(&inferred, &rewritten) {
            inferred
        } else {
            let tidied = batch(inferred, "Tidy", t)?;
            // Tidying can return the round's input (what the rule added was
            // already there, spelled another way): nothing to do again.
            if same_plan(&tidied, &input) {
                done = true;
                input
            } else {
                tidied
            }
        };
        if done {
            break;
        }
    }
    t.notes("JoinOrder", &root, notes);
    Ok(root)
}

/// Whether two relations are structurally equal (their canonical encodings).
fn same_plan(a: &Arc<LogicalPlan>, b: &Arc<LogicalPlan>) -> bool {
    Arc::ptr_eq(a, b)
        || matches!(
            (Plan::try_new(LogicalPlan::clone(a)), Plan::try_new(LogicalPlan::clone(b))),
            (Ok(x), Ok(y)) if x.structurally_eq(&y)
        )
}

/// Whether a `Filter` whose predicate is the literal `false` is anywhere in
/// `root` (subquery plans included).
fn has_false_filter(root: &Arc<LogicalPlan>) -> bool {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![Arc::clone(root)];
    while let Some(p) = stack.pop() {
        if let PlanNode::Filter { predicate, .. } = &p.node
            && matches!(
                predicate.node,
                ExprNode::Literal(krishiv_plan::Literal::Boolean(false))
            )
        {
            return true;
        }
        if !seen.insert(Arc::as_ptr(&p)) {
            continue;
        }
        p.for_each_expr(&mut |e| stack.extend(restrict::subquery_plans(e)));
        p.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    false
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

/// The plan with the output attributes `output` (when CSE widened a
/// pass-through operator below, a filter's output is its input's, now with
/// the extracted columns): a projection back to them on top, or `root`
/// itself when they already match, so re-optimising returns the same plan.
fn restore_output(
    root: Arc<LogicalPlan>,
    output: &Option<Vec<BoundColumn>>,
    origin: Origin,
) -> Arc<LogicalPlan> {
    let Some(want) = output else {
        return root;
    };
    if want.is_empty() {
        return root;
    }
    let have: Vec<AttrId> = root
        .resolved_output()
        .unwrap_or_default()
        .iter()
        .map(|c| c.id)
        .collect();
    if have.len() == want.len() && have.iter().zip(want).all(|(a, b)| a == &b.id) {
        return root;
    }
    if !want.iter().all(|c| have.contains(&c.id)) {
        return root;
    }
    Arc::new(LogicalPlan {
        node: PlanNode::Project {
            input: root,
            exprs: want
                .iter()
                .map(|c| {
                    Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
                        c.clone(),
                    ))))
                })
                .collect(),
        },
        origin,
    })
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

/// Whether every expression of `plan` (subquery plans included) is deterministic: a shared plan that is not is
/// run again, with its readers, when its output is lost (P2-T198).
#[must_use]
pub fn plan_is_deterministic(plan: &Arc<LogicalPlan>) -> bool {
    inline_cte::deterministic(plan)
}

#[cfg(test)]
mod tests {
    use krishiv_common::error::ErrorClass;
    use krishiv_plan::{Expr, MAX_DEPTH};

    use super::*;

    fn filters(n: usize) -> Arc<LogicalPlan> {
        Arc::new((0..n).fold(LogicalPlan::table(["t"]), |p, _| {
            p.filter(Expr::column(["a"]))
        }))
    }

    /// Past the depth cap the plan runs with its keys, or fails: never as the unkeyed input,
    /// which would compare collated values as bytes.
    #[test]
    fn an_unoptimised_plan_keeps_its_collation_keys() {
        let input = Plan::try_new(filters(3)).unwrap();
        // No collations: the keyed plan is the input.
        let same = unoptimised(&input, Arc::clone(input.root())).unwrap();
        assert!(Arc::ptr_eq(same.root(), input.root()));
        // Keys added: the keyed plan, not the input.
        let keyed = filters(4);
        let out = unoptimised(&input, Arc::clone(&keyed)).unwrap();
        assert!(Arc::ptr_eq(out.root(), &keyed));
        // Keys that nest past the cap: the error.
        let err = unoptimised(&input, filters(MAX_DEPTH + 1)).unwrap_err();
        assert_eq!(err.error_class(), ErrorClass::FailedToParseTooComplex);
    }
}
