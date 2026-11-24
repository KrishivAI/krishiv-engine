// SPDX-License-Identifier: Apache-2.0
//! `PayloadAboveAggregate`: a dimension's payload joins above the aggregate
//! that groups by it (design `design/P2b-A5-payload-above-aggregate.md` in
//! krishiv-internal; Yan and Larson 1994, Theorem 1).
//!
//! ```text
//! Aggregate[K_X ∪ d ∪ P; calls(X)](Path(X ⋈[x = d] D'))
//!   =>  Project[output](Aggregate[K_X ∪ x; calls(X)](Path'(X)) ⋈[x = d] D')
//! ```
//!
//! `D'` is one side of an inner equi-join, `d` its key columns, `P` the other
//! columns of `D'` the aggregate groups by, `K_X` its grouping columns from
//! the other side. The rows are the same exactly when Yan and Larson's FD1 and
//! FD2 hold: FD1 (`d` is a function of the new groups) holds by `x = d` with
//! every column of `d` grouped; FD2 (each new group meets at most one `D'`
//! row) holds when `d` is unique among the non-NULL keys of `D'`. Then each new
//! group meets one `D'` row carrying the old group's `P`, and the calls saw
//! the same rows of `X`: the same multiset of rows, for any aggregate.
//!
//! # FD2: where the key comes from
//!
//! Only a proof ([`unique`]): `D'` (and each side it absorbed) is, through
//! filters, plain column projections and inner joins on another relation's
//! unique key, an aggregate grouped by a subset of `d` (the plan), or a scan
//! whose table declares a subset of `d` unique ([`Statistics::unique_keys`]:
//! statistics). Nothing is guessed: without a proof the rule does not fire.
//!
//! Where only a base relation's fact is missing (a scan of a table or of
//! files with no declared key), the candidate goes on through every other
//! gate with the missing facts pending; when all of them pass, each is
//! asked for ([`Statistics::want_unique`]: a request, never a fact, design
//! P2b A5 (d)) and the rule refuses with an "awaiting a key count" note. The
//! session may count the columns (`krishiv.optimizer.discoverKeys`) and
//! optimize once more, when the answer is a fact like any other.
//!
//! # Absorption
//!
//! An inner join on the path whose condition reads `D'`'s columns and only
//! the other input `E`'s columns, with `E` proven unique on the columns its
//! equalities compare, joins `D'` instead (TPC-H q10's `nation`, above the
//! `customer` join): each `D'` row meets at most one `E` row, so `D' ⋈ E` is
//! unique on `d` too, and `E`'s columns join `P`.
//!
//! # When it applies (each refusal is noted for `EXPLAIN VERBOSE`)
//!
//! - The aggregate groups by columns (no grouping sets, no `Expand` below);
//!   the join is inner, its condition only `=` of a column of each side, of a
//!   type whose equality is identity (integers, decimals, dates, timestamps,
//!   strings and binary of the binary collation: no floating point, no
//!   collation);
//! - the path from the join to the aggregate holds only deterministic
//!   filters, projections and joins on the side whose rows they keep, reading
//!   none of `D'`'s columns (a projection may pass one through), apart from
//!   absorbed joins;
//! - every call reads only the other side (arguments and `FILTER`);
//! - errors (review B1): the new aggregate and `Path'` also see the `X` rows
//!   the join dropped first (no match, NULL keys), so every path expression,
//!   absorbed condition, call argument and `FILTER` must run on any row
//!   (`runs_on_any_row`: cannot raise, or exact decimal arithmetic), the calls
//!   are the built-in (registry, never a session UDAF of the same name)
//!   `count`, `min`, `max`, and integer or decimal `sum` and `avg` (floating
//!   point never: the summation order changes), and every `sum` and `avg`
//!   buffer is proven not to overflow (integer `avg`: its double sum stays
//!   below 2^53) from the argument's range times the rows of `Path'(X)`,
//!   estimated from statistics only, with eager aggregation's 8× margin;
//! - `P` holds a variable-width column or at least
//!   [`MIN_PAYLOAD_BYTES`] a row;
//! - the join keeps at least [`MIN_RETENTION`] of `X`'s rows, both estimated
//!   from statistics (a selective `D'` pruned `X` before the aggregate);
//! - no node the rule rebuilds or moves is read by another parent;
//! - batch plans only. No gate is needed: the session's batch planning
//!   (`krishiv-session` `planning.rs`) is the optimizer's only production
//!   caller, and streaming plans never pass through `optimize*`. A streaming
//!   caller must switch the rule off (`Options::payload_above_aggregate`).
//!
//! Key types: every equality the proofs read (the moved join's, an absorbed
//! join's, an inner join's in [`unique`]) must be of one type whose `=` is
//! identity ([`identity_key`]), so that a key unique under grouping (a fact or
//! an aggregate) is unique under the join's `=`.
//!
//! # Residual assumption
//!
//! The overflow proof bounds each argument soundly (footer ranges, or the
//! type's range) but multiplies by the *estimated* rows of `Path'(X)` with an
//! 8× margin, as eager aggregation's rule 2 does. If the true rows exceed the
//! estimate by more than 8× and the proven bound is close to the buffer's
//! limit, a group the join would have dropped could overflow under ANSI where
//! Spark raises nothing. The design note (§2) accepts this.
//!
//! # Not here
//!
//! With `K_X` non-empty the new aggregate is partitioned on `K_X ∪ x` and the
//! payload join on `x`: a second shuffle. Partitioning the aggregate on `x`
//! alone (design §4, review minor 8) is the physical planner's and is
//! deferred; it costs time, never rows. q10 has an empty `K_X`.
//!
//! # Fixed point
//!
//! The rule runs after join ordering and pruning (`lib.rs`). The moved join
//! sits above an aggregate, which join ordering treats as an input and never
//! looks through, so ordering cannot move `D'` back below it. The new
//! aggregate is tried again at once; after a move `lib.rs` re-runs join
//! ordering, eager aggregation and the group-key code (when on) and then the
//! rule again until it moves nothing, so a second optimization finds nothing
//! the first left.

use std::collections::HashSet;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, Grouping, JoinCondition, JoinKind,
    LogicalPlan, PlanNode, ResolvedExpr, ResolvedFunction, runs_on_any_row,
};
use krishiv_types::DataType;

use crate::dag;
use crate::eager::{MIN_RETENTION, distinct_calls, replace_calls};
use crate::estimate::Estimator;
use crate::expr::{
    Attrs, builtin, column, conjuncts, has_outer_reference, has_subquery, is_deterministic,
    output_of, references, references_into,
};
use crate::late_payload::{
    MIN_PAYLOAD_BYTES, bare_column, children, fixed_width, plain_column, shared_nodes, with_child,
};
use crate::restrict::{MAX_ID, key_type, max_id};
use crate::stats::{RelStats, Statistics, number};
use unique::{Key, Want, push_want, unique};

/// The most moves below one aggregate (the new aggregate is tried again).
const MAX_MOVES: usize = 8;

/// The most plan nodes one walk down from an aggregate visits.
const MAX_WALK: usize = 256;

/// The margin on the estimated rows when proving a sum cannot overflow.
const ROW_MARGIN: f64 = 8.0;

/// 2^63: a `bigint` sum overflows at it.
const LONG_LIMIT: f64 = 9_223_372_036_854_775_808.0;

/// The largest magnitude below which every integer is a double (2^53).
const EXACT_DOUBLE: f64 = 9_007_199_254_740_992.0;

/// Moves dimension payloads above the aggregates that group by them,
/// entering subquery plans. Also returns one line per move, and per aggregate
/// left alone the gate that refused each candidate (`EXPLAIN VERBOSE`).
pub(crate) fn rewrite(
    root: Arc<LogicalPlan>,
    stats: &dyn Statistics,
    propagate: bool,
) -> Result<(Arc<LogicalPlan>, Vec<String>), KrishivError> {
    let mut ctx = Context {
        next: max_id(&root, &mut crate::expr::AddrMap::default()).saturating_add(1),
        notes: Vec::new(),
        shared: shared_nodes(&root),
        stats,
    };
    let mut est = Estimator::with_propagation(stats, propagate);
    let out = dag::rewrite_up(root, &mut |p| {
        Ok(at_aggregate(&p, &mut est, &mut ctx, MAX_MOVES).unwrap_or(p))
    })?;
    Ok((out, ctx.notes))
}

/// What one pass carries between aggregates.
struct Context<'a> {
    /// The next free attribute id.
    next: u64,
    /// `EXPLAIN VERBOSE` lines, a refusal once.
    notes: Vec<String>,
    /// Nodes read by more than one parent in the input plan.
    shared: HashSet<*const LogicalPlan>,
    /// Where key facts come from.
    stats: &'a dyn Statistics,
}

impl Context<'_> {
    fn note(&mut self, line: String) {
        if line.starts_with("moved") || !self.notes.contains(&line) {
            self.notes.push(line);
        }
    }

    fn is_shared(&self, p: &Arc<LogicalPlan>) -> bool {
        self.shared.contains(&Arc::as_ptr(p))
    }
}

/// One step of the path: a node and the index of the child the path came from.
#[derive(Clone)]
struct Step {
    node: Arc<LogicalPlan>,
    child: usize,
}

/// A join below the aggregate: the path from just above it to the
/// aggregate's input (bottom first), the join, which side is the dimension.
struct Candidate {
    path: Vec<Step>,
    join: Arc<LogicalPlan>,
    dim_left: bool,
}

/// What one candidate gave.
enum Outcome {
    /// Not a candidate: the aggregate groups by none of the side's columns.
    Skip,
    /// Refused, and why.
    Refused(String),
    /// The rewritten aggregate and the note.
    Moved(Arc<LogicalPlan>, String),
}

/// `agg` rewritten, if it is an aggregate some candidate below qualifies for.
fn at_aggregate(
    agg: &Arc<LogicalPlan>,
    est: &mut Estimator<'_>,
    ctx: &mut Context<'_>,
    depth: usize,
) -> Option<Arc<LogicalPlan>> {
    if depth == 0 {
        return None;
    }
    let PlanNode::Aggregate {
        input,
        grouping,
        aggregates: outputs,
    } = &agg.node
    else {
        return None;
    };
    let mut found = Vec::new();
    let mut budget = MAX_WALK;
    let mut expand = false;
    walk(input, &mut Vec::new(), &mut found, &mut budget, &mut expand);
    let Grouping::GroupBy(keys) = grouping else {
        if !found.is_empty() {
            ctx.note("not applied: grouping sets".to_owned());
        }
        return None;
    };
    if expand {
        ctx.note("not applied: grouping sets (an Expand below the aggregate)".to_owned());
        return None;
    }
    let mut refusals = Vec::new();
    for cand in &found {
        match try_candidate(agg, keys, outputs, cand, est, ctx) {
            Outcome::Skip => {}
            Outcome::Refused(why) => refusals.push(why),
            Outcome::Moved(out, note) => {
                ctx.note(note);
                // The new aggregate may group by another dimension's payload.
                return Some(again(&out, est, ctx, depth - 1));
            }
        }
    }
    for r in refusals {
        ctx.note(r);
    }
    None
}

/// `Project(Join(agg, D'))` with the rule applied to `agg` once more.
fn again(
    out: &Arc<LogicalPlan>,
    est: &mut Estimator<'_>,
    ctx: &mut Context<'_>,
    depth: usize,
) -> Arc<LogicalPlan> {
    let PlanNode::Project { input: join, .. } = &out.node else {
        return Arc::clone(out);
    };
    let PlanNode::Join { left, .. } = &join.node else {
        return Arc::clone(out);
    };
    match at_aggregate(left, est, ctx, depth) {
        Some(inner) => dag::with_input(out, with_child(join, 0, inner)),
        None => Arc::clone(out),
    }
}

/// Every inner join reachable from `at` through nodes a path may pass, with
/// the path above it; `expand` is set when an `Expand` is met.
fn walk(
    at: &Arc<LogicalPlan>,
    path: &mut Vec<Step>,
    found: &mut Vec<Candidate>,
    budget: &mut usize,
    expand: &mut bool,
) {
    if *budget == 0 {
        return;
    }
    *budget -= 1;
    match &at.node {
        PlanNode::Expand { .. } => *expand = true,
        PlanNode::Join {
            kind: JoinKind::Inner,
            condition: JoinCondition::On(_),
            ..
        } => {
            for dim_left in [true, false] {
                found.push(Candidate {
                    path: path.iter().rev().cloned().collect(),
                    join: Arc::clone(at),
                    dim_left,
                });
            }
        }
        _ => {}
    }
    for (i, kid) in children(at).iter().enumerate() {
        if may_pass(at, i) {
            path.push(Step {
                node: Arc::clone(at),
                child: i,
            });
            walk(kid, path, found, budget, expand);
            path.pop();
        }
    }
}

/// Whether a path may come up through `node` from child `i`: a projection,
/// a filter, or a join on the side whose rows it keeps.
fn may_pass(node: &LogicalPlan, i: usize) -> bool {
    match &node.node {
        PlanNode::Project { .. } | PlanNode::Filter { .. } => i == 0,
        PlanNode::Join {
            kind,
            condition: JoinCondition::On(_),
            ..
        } => match kind {
            JoinKind::Inner => true,
            JoinKind::LeftSemi
            | JoinKind::LeftAnti
            | JoinKind::LeftOuter
            | JoinKind::Existence(_) => i == 0,
            JoinKind::RightOuter => i == 1,
            _ => false,
        },
        _ => false,
    }
}

/// The candidate's join read: `(x, d)` per equality, the dimension and the
/// other side.
struct Join<'a> {
    eqs: Vec<(BoundColumn, BoundColumn)>,
    dim: &'a Arc<LogicalPlan>,
    other: &'a Arc<LogicalPlan>,
    condition: &'a JoinCondition,
}

impl<'a> Join<'a> {
    fn of(c: &'a Candidate) -> Result<Self, String> {
        let PlanNode::Join {
            left,
            right,
            condition,
            ..
        } = &c.join.node
        else {
            return Err("not a join".to_owned());
        };
        let JoinCondition::On(on) = condition else {
            return Err("a join without a condition".to_owned());
        };
        let (dim, other) = if c.dim_left {
            (left, right)
        } else {
            (right, left)
        };
        let dim_ids = ids(dim).ok_or("a join side without a resolved output")?;
        let other_ids = ids(other).ok_or("a join side without a resolved output")?;
        let mut eqs = Vec::new();
        for e in conjuncts(on) {
            let pair = builtin(&e).and_then(|(op, args, _)| match (op, args) {
                (Builtin::Eq, [a, b]) => Some((bare_column(a)?, bare_column(b)?)),
                _ => None,
            });
            let Some((a, b)) = pair else {
                return Err("the join condition is not only equalities of columns".to_owned());
            };
            let (x, d) = if other_ids.contains(&a.id) && dim_ids.contains(&b.id) {
                (a, b)
            } else if dim_ids.contains(&a.id) && other_ids.contains(&b.id) {
                (b, a)
            } else {
                return Err("an equality does not compare the two sides".to_owned());
            };
            eqs.push((x.clone(), d.clone()));
        }
        Ok(Self {
            eqs,
            dim,
            other,
            condition,
        })
    }

    fn keys_text(&self) -> String {
        let v: Vec<String> = self
            .eqs
            .iter()
            .map(|(x, d)| format!("{} = {}", x.name, d.name))
            .collect();
        v.join(", ")
    }
}

/// The output ids of `p`.
fn ids(p: &LogicalPlan) -> Option<Attrs> {
    Some(p.resolved_output()?.iter().map(|c| c.id).collect())
}

/// Whether `t`'s `=` is identity, so `x = d` makes `x` and `d` the same value.
/// Every proof of uniqueness rests on it too: a key unique under grouping is
/// unique under `=` only for these types.
fn identity_key(t: &DataType) -> bool {
    key_type(t) && !t.contains_collation()
}

/// The aggregate rewritten over `c`, why not, or not a candidate.
#[expect(
    clippy::too_many_lines,
    reason = "one rule's gates in the order the design note gives them"
)]
fn try_candidate(
    agg: &Arc<LogicalPlan>,
    keys: &[Arc<Expr>],
    outputs: &[Arc<Expr>],
    c: &Candidate,
    est: &mut Estimator<'_>,
    ctx: &mut Context<'_>,
) -> Outcome {
    let join = match Join::of(c) {
        Ok(j) => j,
        Err(why) => {
            // A refusal only where the aggregate groups by the side's columns.
            let side = children(&c.join)
                .get(usize::from(!c.dim_left))
                .and_then(|p| ids(p));
            let grouped_side = side.is_some_and(|s| {
                keys.iter()
                    .filter_map(|k| bare_column(k))
                    .any(|k| s.contains(&k.id))
            });
            return if grouped_side {
                Outcome::Refused(format!("not applied: {why}"))
            } else {
                Outcome::Skip
            };
        }
    };
    // Grouping keys are columns.
    let mut grouped = Attrs::default();
    for k in keys {
        match bare_column(k) {
            Some(col) => {
                grouped.insert(col.id);
            }
            None => return Outcome::Skip,
        }
    }
    let Some(dim_ids) = ids(join.dim) else {
        return Outcome::Skip;
    };
    if dim_ids.is_disjoint(&grouped) {
        return Outcome::Skip;
    }
    let refuse = |why: String| {
        Outcome::Refused(format!(
            "not applied to the join on [{}]: {why}",
            join.keys_text()
        ))
    };

    // FD1: every key column of the dimension is grouped.
    if let Some((_, d)) = join.eqs.iter().find(|(_, d)| !grouped.contains(&d.id)) {
        return refuse(format!("the dimension's key {} is not all grouped", d.name));
    }
    if let Some((x, _)) = join.eqs.iter().find(|(x, _)| !identity_key(&x.data_type)) {
        return refuse(format!("a {} key: equal values may differ", x.data_type));
    }
    if ctx.is_shared(&c.join)
        || ctx.is_shared(join.dim)
        || c.path.iter().any(|s| ctx.is_shared(&s.node))
    {
        return refuse("the join, its path or the dimension is read by another parent".to_owned());
    }

    // The path, bottom up: absorb, or keep.
    let mut dprime = Arc::clone(join.dim);
    let mut dcols = dim_ids;
    let mut absorbed: Vec<String> = Vec::new();
    let mut kept: Vec<&Step> = Vec::new();
    let mut proofs: Vec<String> = Vec::new();
    // Key counts a proof still waits for (design P2b A5 (d)): asked for only
    // once every other gate passed.
    let mut wanted: Vec<Want> = Vec::new();
    for s in &c.path {
        match &s.node.node {
            PlanNode::Project { exprs, .. } => {
                for e in exprs {
                    if !is_deterministic(e) {
                        return refuse("a projection on the path is nondeterministic".to_owned());
                    }
                    if !references(e).is_disjoint(&dcols) {
                        if !plain_column(e) {
                            return refuse(
                                "a projection on the path computes from the dimension".to_owned(),
                            );
                        }
                    } else if !runs_on_any_row(e, true) {
                        return refuse("a projection on the path may raise".to_owned());
                    }
                }
                kept.push(s);
            }
            PlanNode::Filter { predicate, .. } => {
                if !references(predicate).is_disjoint(&dcols) {
                    return refuse("a filter on the path reads the dimension".to_owned());
                }
                if !runs_on_any_row(predicate, true) {
                    return refuse("a filter on the path may raise".to_owned());
                }
                kept.push(s);
            }
            PlanNode::Join {
                kind,
                condition: JoinCondition::On(on),
                ..
            } => {
                if !runs_on_any_row(on, true) {
                    return refuse("a join condition on the path may raise".to_owned());
                }
                let refs = references(on);
                if refs.is_disjoint(&dcols) {
                    kept.push(s);
                    continue;
                }
                if *kind != JoinKind::Inner {
                    return refuse(format!("a {kind:?} join on the path reads the dimension"));
                }
                let kids = children(&s.node);
                let Some(e) = kids.get(usize::from(s.child == 0)) else {
                    return refuse("a join without two inputs".to_owned());
                };
                let Some(e_ids) = ids(e) else {
                    return refuse("a join side without a resolved output".to_owned());
                };
                if !refs.iter().all(|r| dcols.contains(r) || e_ids.contains(r)) {
                    return refuse(
                        "a join on the path reads the dimension and the other side".to_owned(),
                    );
                }
                if ctx.is_shared(e) {
                    return refuse("an absorbed input is read by another parent".to_owned());
                }
                // E's columns compared with D''s by its equalities.
                let mut e_keys = Attrs::default();
                for conj in conjuncts(on) {
                    if let Some((Builtin::Eq, [a, b], _)) = builtin(&conj)
                        && let (Some(a), Some(b)) = (bare_column(a), bare_column(b))
                        && identity_key(&a.data_type)
                        && a.data_type == b.data_type
                    {
                        if dcols.contains(&a.id) && e_ids.contains(&b.id) {
                            e_keys.insert(b.id);
                        } else if dcols.contains(&b.id) && e_ids.contains(&a.id) {
                            e_keys.insert(a.id);
                        }
                    }
                }
                let name = relation_name(e);
                match unique(e, &e_keys, ctx.stats) {
                    Some(Key::Proven(proof)) => proofs.push(format!("{name} by {proof}")),
                    Some(Key::Wanted(w)) => {
                        for w in w {
                            push_want(&mut wanted, w);
                        }
                    }
                    None => {
                        return refuse(format!(
                            "the joined {name} is not proven unique on its join key"
                        ));
                    }
                }
                absorbed.push(name);
                let (left, right) = if s.child == 0 {
                    (dprime, Arc::clone(e))
                } else {
                    (Arc::clone(e), dprime)
                };
                dprime = with_child(&with_child(&s.node, 0, left), 1, right);
                dcols.extend(e_ids);
            }
            _ => return refuse("the path holds another operator".to_owned()),
        }
    }

    // What the aggregate reads.
    if keys
        .iter()
        .chain(outputs)
        .any(|e| has_subquery(e) || has_outer_reference(e) || !is_deterministic(e))
    {
        return refuse(
            "the aggregate holds a subquery or a nondeterministic expression".to_owned(),
        );
    }
    let calls = distinct_calls(outputs);
    for (_, f) in &calls {
        if let Err(why) = call_reads_x(f, &dcols) {
            return refuse(why);
        }
    }
    let d_ids: Attrs = join.eqs.iter().map(|(_, d)| d.id).collect();
    let top_out = c
        .path
        .last()
        .map_or(&c.join, |s| &s.node)
        .resolved_output()
        .unwrap_or_default();
    let payload: Vec<BoundColumn> = top_out
        .iter()
        .filter(|o| dcols.contains(&o.id) && grouped.contains(&o.id) && !d_ids.contains(&o.id))
        .cloned()
        .collect();
    let width: Option<usize> = payload.iter().map(|p| fixed_width(&p.data_type)).sum();
    if width.is_some_and(|w| w < MIN_PAYLOAD_BYTES) {
        return refuse(format!(
            "a payload of {} B a row is under {MIN_PAYLOAD_BYTES} B",
            width.unwrap_or(0)
        ));
    }

    // FD2.
    let proof = match unique(join.dim, &d_ids, ctx.stats) {
        Some(Key::Proven(proof)) => Some(proof),
        Some(Key::Wanted(w)) => {
            for w in w {
                push_want(&mut wanted, w);
            }
            None
        }
        None => {
            return refuse(format!(
                "no fact proves the dimension unique on [{}]",
                names(join.eqs.iter().map(|(_, d)| d))
            ));
        }
    };

    // Retention, from statistics.
    let (je, xe) = (est.plan(&c.join), est.plan(join.other));
    if !je.from_statistics || !xe.from_statistics {
        return refuse("the estimates rest on default row counts".to_owned());
    }
    if je.rows < MIN_RETENTION * xe.rows {
        return refuse(format!(
            "retention below {MIN_RETENTION}: the join keeps {:.0} of {:.0} rows",
            je.rows, xe.rows
        ));
    }

    // Path'(X): the kept nodes over X, without D''s columns, with x kept.
    let xs: Vec<BoundColumn> = {
        let mut v: Vec<BoundColumn> = Vec::new();
        for (x, _) in &join.eqs {
            if !v.iter().any(|y| y.id == x.id) {
                v.push(x.clone());
            }
        }
        v
    };
    let mut below = Arc::clone(join.other);
    for s in kept {
        below = match &s.node.node {
            PlanNode::Project { exprs, .. } => {
                let mut items: Vec<Arc<Expr>> = exprs
                    .iter()
                    .filter(|e| output_of(e).is_none_or(|o| !dcols.contains(&o.id)))
                    .cloned()
                    .collect();
                for x in &xs {
                    if !items
                        .iter()
                        .any(|e| output_of(e).is_some_and(|o| o.id == x.id))
                    {
                        items.push(column(x));
                    }
                }
                let mut p = LogicalPlan::clone(&s.node);
                if let PlanNode::Project { input, exprs } = &mut p.node {
                    *input = below;
                    *exprs = items;
                }
                Arc::new(p)
            }
            _ => with_child(&s.node, s.child, below),
        };
    }

    // The error gate's accumulation half: no sum buffer can overflow.
    let path_est = est.plan(&below);
    // The bound multiplies by these rows: never a default count.
    if !path_est.from_statistics {
        return refuse(
            "the rows of the aggregate's new input rest on default row counts".to_owned(),
        );
    }
    for (_, f) in &calls {
        if let Err(why) = accumulates_safely(f, &path_est) {
            return refuse(why);
        }
    }

    // The new aggregate: K_X and x, each call under a fresh column.
    let mut group_cols: Vec<BoundColumn> = Vec::new();
    for k in keys {
        if let Some(col) = bare_column(k)
            && !dcols.contains(&col.id)
            && !group_cols.iter().any(|g| g.id == col.id)
        {
            group_cols.push(col.clone());
        }
    }
    for x in &xs {
        if !group_cols.iter().any(|g| g.id == x.id) {
            group_cols.push(x.clone());
        }
    }
    let first = ctx.next;
    let Some(next) = first
        .checked_add(calls.len() as u64)
        .filter(|n| *n <= MAX_ID)
    else {
        return refuse("the plan has no attribute ids left".to_owned());
    };
    // Every other gate passed: what is still unproven is asked for, and the
    // rule waits for the answer (a later optimization reads it as a fact).
    if !wanted.is_empty() {
        for w in &wanted {
            if let PlanNode::Scan(scan) = &w.scan.node {
                ctx.stats.want_unique(scan, &w.columns);
            }
        }
        let texts: Vec<String> = wanted.iter().map(Want::text).collect();
        return refuse(format!(
            "no fact proves the dimension unique on [{}] yet: awaiting a key count of {}",
            names(join.eqs.iter().map(|(_, d)| d)),
            texts.join(", ")
        ));
    }
    let Some(proof) = proof else {
        return refuse("no fact proves the dimension unique".to_owned());
    };
    ctx.next = next;
    let mut merged: Vec<(Arc<Expr>, Arc<Expr>)> = Vec::new();
    let mut agg_outputs: Vec<Arc<Expr>> = group_cols.iter().map(column).collect();
    for (i, (call, f)) in calls.iter().enumerate() {
        let out = BoundColumn {
            id: AttrId(first + i as u64),
            name: format!("{}_{}", f.id.name(), first + i as u64),
            data_type: f.data_type.clone(),
            nullable: f.nullable,
        };
        agg_outputs.push(Arc::new(Expr::new(ExprNode::Resolved(
            ResolvedExpr::Alias {
                expr: Arc::clone(call),
                output: out.clone(),
            },
        ))));
        merged.push((Arc::clone(call), column(&out)));
    }
    let new_agg = Arc::new(
        LogicalPlan::new(PlanNode::Aggregate {
            input: below,
            grouping: Grouping::GroupBy(group_cols.iter().map(column).collect()),
            aggregates: agg_outputs,
        })
        .with_origin(agg.origin),
    );
    let joined = Arc::new(
        LogicalPlan::new(PlanNode::Join {
            left: new_agg,
            right: dprime,
            kind: JoinKind::Inner,
            condition: join.condition.clone(),
        })
        .with_origin(c.join.origin),
    );
    let top = Arc::new(
        LogicalPlan::new(PlanNode::Project {
            input: joined,
            exprs: outputs.iter().map(|e| replace_calls(e, &merged)).collect(),
        })
        .with_origin(agg.origin),
    );
    let mut source = format!("the dimension by {proof}");
    for p in proofs {
        source.push_str(", ");
        source.push_str(&p);
    }
    let note = format!(
        "moved the join on [{}] above the aggregate: grouped by [{}], payload [{}], absorbed [{}]; unique: {source}",
        join.keys_text(),
        names(group_cols.iter()),
        names(payload.iter()),
        absorbed.join(", "),
    );
    Outcome::Moved(top, note)
}

fn names<'a>(cols: impl Iterator<Item = &'a BoundColumn>) -> String {
    let v: Vec<&str> = cols.map(|c| c.name.as_str()).collect();
    v.join(", ")
}

/// The table a relation reads, for notes: the first scan's name.
fn relation_name(p: &Arc<LogicalPlan>) -> String {
    let mut stack = vec![Arc::clone(p)];
    while let Some(x) = stack.pop() {
        if let PlanNode::Scan(s) = &x.node
            && let krishiv_plan::ReadSource::Table { name, .. } = &s.source
            && let Some(n) = name.last()
        {
            return n.clone();
        }
        x.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    "input".to_owned()
}

/// Why `f` cannot run over the other side's rows alone: it reads the
/// dimension, or its argument or `FILTER` may raise on a row the join dropped.
fn call_reads_x(f: &ResolvedFunction, dcols: &Attrs) -> Result<(), String> {
    let name = f.id.name();
    let mut read = Attrs::default();
    for a in &f.args {
        references_into(a, &mut read);
    }
    if let Some(filter) = &f.filter {
        references_into(filter, &mut read);
    }
    if !read.is_disjoint(dcols) {
        return Err(format!("{name} reads the dimension"));
    }
    if !f.deterministic {
        return Err(format!("{name} is nondeterministic"));
    }
    if f.args
        .iter()
        .chain(f.filter.iter())
        .any(|a| !runs_on_any_row(a, true) || has_outer_reference(a))
    {
        return Err(format!(
            "{name}'s argument may raise on a row the join drops"
        ));
    }
    Ok(())
}

/// Why `f`'s accumulation may raise over `rel`'s rows: a call outside
/// `count`, `min`, `max`, `sum`, `avg`, or a sum buffer whose bound is not
/// proven (argument magnitude × rows × 8 below the buffer's limit).
fn accumulates_safely(f: &ResolvedFunction, rel: &RelStats) -> Result<(), String> {
    let name = f.id.name();
    // By identity, never by name: a session UDAF named `max` is not `max`.
    if !matches!(f.id, FunctionId::Registry { .. }) {
        return Err(format!(
            "{name} is not a built-in aggregate: its body may raise on a row the join drops"
        ));
    }
    let arg = f.args.first();
    let arg_type = arg.and_then(|a| crate::expr::data_type(a));
    let limit = match (name, &arg_type) {
        // No buffer that raises and no order-dependent rounding: `count`,
        // `min`, `max`. Floating-point `sum` and `avg` never move: removing
        // the join changes the summation order, and so the rounding (as
        // `eager.rs` rule 2).
        ("count" | "min" | "max", _) => return Ok(()),
        // `avg` of integers sums in a double: exact, so order-free, while
        // every partial sum stays below 2^53.
        (
            "avg" | "mean",
            Some(DataType::Byte | DataType::Short | DataType::Integer | DataType::Long),
        ) => EXACT_DOUBLE,
        ("sum", Some(DataType::Byte | DataType::Short | DataType::Integer | DataType::Long)) => {
            LONG_LIMIT
        }
        ("sum", Some(DataType::Decimal(_))) => match &f.data_type {
            DataType::Decimal(r) => 10_f64.powi(i32::from(r.precision()) - i32::from(r.scale())),
            _ => return Err(format!("{name} without a decimal result")),
        },
        ("avg" | "mean", Some(DataType::Decimal(t))) => {
            // The buffer is decimal(min(38, p + 10), s).
            let p = (i32::from(t.precision()) + 10).min(38);
            10_f64.powi(p - i32::from(t.scale()))
        }
        _ => {
            return Err(format!(
                "{name} over {arg_type:?} is not known to accumulate without raising"
            ));
        }
    };
    let magnitude = arg
        .and_then(|a| magnitude(a, rel))
        .or_else(|| arg_type.as_ref().and_then(type_magnitude));
    let Some(m) = magnitude else {
        return Err(format!("{name}'s overflow bound is unknown"));
    };
    if m * rel.rows.max(1.0) * ROW_MARGIN < limit {
        Ok(())
    } else {
        Err(format!(
            "{name} may overflow: |argument| <= {m:e} over {:.0} rows",
            rel.rows
        ))
    }
}

/// The largest magnitude a value of `t` holds, for integers and decimals.
fn type_magnitude(t: &DataType) -> Option<f64> {
    Some(match t {
        DataType::Byte => 128.0,
        DataType::Short => 32_768.0,
        DataType::Integer => 2_147_483_648.0,
        DataType::Long => LONG_LIMIT,
        DataType::Decimal(d) => 10_f64.powi(i32::from(d.precision()) - i32::from(d.scale())),
        _ => return None,
    })
}

/// A bound on `|e|` over the rows `rel` estimates, from its columns' ranges
/// (or, without one, their types')
/// (interval arithmetic over `+`, `-`, `*`, negation, casts and literals).
fn magnitude(e: &Expr, rel: &RelStats) -> Option<f64> {
    match &e.node {
        ExprNode::Literal(l) => Some(number(l)?.abs()),
        ExprNode::Resolved(ResolvedExpr::Column(c)) => {
            // The column's range, else every value its type holds.
            let range = rel
                .cols
                .get(&c.id)
                .and_then(|s| Some(s.min?.abs().max(s.max?.abs())));
            range.or_else(|| type_magnitude(&c.data_type))
        }
        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) | ExprNode::Cast { expr, .. } => {
            magnitude(expr, rel)
        }
        _ => {
            let (op, args, _) = builtin(e)?;
            match (op, args) {
                (Builtin::Add | Builtin::Subtract, [a, b]) => {
                    Some(magnitude(a, rel)? + magnitude(b, rel)?)
                }
                (Builtin::Multiply, [a, b]) => Some(magnitude(a, rel)? * magnitude(b, rel)?),
                (Builtin::Negative, [a]) => magnitude(a, rel),
                _ => None,
            }
        }
    }
}

#[cfg(test)]
mod tests;
/// Whether `p` is unique on a subset of `cols` by a proof (the plan or a
/// [`Statistics::unique_keys`] fact) or could be once a base relation's key
/// count answers a request ([`Statistics::want_unique`]). Eager aggregation
/// leaves such a join to this rule first, so the rule can fire or issue the
/// request; eager runs again after it.
pub(crate) fn keyed(p: &Arc<LogicalPlan>, cols: &Attrs, stats: &dyn Statistics) -> bool {
    unique(p, cols, stats).is_some()
}

mod unique;
