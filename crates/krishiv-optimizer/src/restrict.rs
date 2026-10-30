// SPDX-License-Identifier: Apache-2.0
//! Restricting a grouped aggregate to the keys a join can match: a
//! semi-join reduction of the aggregate's input by the domain of the
//! join's other side.
//!
//! A join that drops the aggregate's rows without a match (an inner or
//! semi join, either side; the null-supplying side of an outer join; the
//! right side of an anti join) only ever reads the groups whose keys meet
//! a value of the other side under the condition's equalities `k = o`. So
//! the aggregate may group only the input rows whose keys do:
//!
//! ```text
//! Join(A, Aggregate[g](I), k = o)
//!   =>  Join(A, Aggregate[g](I ⋉ Shift(π_o(D)), g = o), k = o)
//! ```
//!
//! where `D` is `A` or a subtree of it that holds every value of `o` that
//! `A` does (a superset of the domain is enough). Each kept group sees all
//! of its rows, so its value is unchanged; groups no row of `A` asks for
//! are never formed. This is how a decorrelated scalar subquery such as
//! TPC-H q17's `(select 0.2 * avg(l_quantity) from lineitem where
//! l_partkey = p_partkey)`, planned as an aggregate grouped by its
//! correlation keys and joined back, groups only the parts the outer query
//! keeps instead of every part. Prior art: magic-set rewriting
//! (Seshadri et al., "Cost-Based Optimization for Magic: Algebra and
//! Implementation", SIGMOD 1996) and the duplicate-free outer domain of
//! dependent-join unnesting (Neumann and Kemper, "Unnesting Arbitrary
//! Queries", BTW 2015).
//!
//! The result is ordinary operators (a semi join below the aggregate, the
//! domain recomputed under a [`PlanNode::Shift`]), so it plans and
//! distributes like any query; nothing is shared between them at run
//! time. The semi join's build side also feeds the hash join's runtime
//! filter on the aggregate's scan.
//!
//! Applied only when it is sound and pays off:
//! - the aggregate is a plain `GROUP BY` with at least one key (grouping
//!   sets add rows with NULL keys), and the aggregate, its input and the
//!   domain subtree call no nondeterministic function and consist only of
//!   scans, filters, projections, joins, aggregates and shifts (`D` is
//!   computed twice and must agree);
//! - each restricting equality is a plain `=` (never `<=>`; NULL keys
//!   match nothing either way) of columns or integral widenings (the semi
//!   join evaluates them on rows the join never reads, so they must not
//!   raise) whose aggregate side reads only grouping keys of types whose grouping and equality agree (no floating point:
//!   grouping normalizes `-0.0` and NaN); other conjuncts, non-equality
//!   correlation included, are left alone, and without an equality the
//!   rule does nothing;
//! - the domain's estimated keys are at most half the aggregate's groups,
//!   it keeps at most half the rows it reads (a whole table's keys, as for
//!   a foreign key, restrict nothing), and computing it scans no more
//!   rows than the aggregate's input does.
//!
//! Like a runtime filter, the restriction may skip rows whose evaluation
//! would have raised an ANSI error (a `sum` that overflows in a group no
//! row asks for); it never changes a value. This is a documented
//! divergence from Spark, which computes every group.

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, Grouping, JoinCondition, JoinKind, LogicalPlan,
    Origin, PlanNode, ResolvedExpr,
};
use krishiv_types::DataType;

use crate::dag::{self, with_input};
use crate::expr::{
    Attrs, and_all, builtin, column, conjuncts, has_outer_reference, has_subquery,
    is_deterministic, recall, references, same, strip_alias, substitute,
};
use crate::stats::{Est, Estimator, Statistics};

/// The domain must have at most `1 / LOPSIDED` of the aggregate's groups.
const LOPSIDED: f64 = 2.0;

/// The domain moves into a subtree of the other side only while its keys
/// grow by at most this factor.
const DESCEND: f64 = 2.0;

/// The largest attribute id a plan may hold (as the analyzer allows).
pub(crate) const MAX_ID: u64 = 1 << 62;

/// Restricts the aggregates below joins in `root` (see the module docs).
pub(crate) fn restrict(
    root: Arc<LogicalPlan>,
    stats: &dyn Statistics,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    let mut r = Restrict {
        est: Estimator::new(stats),
        root: Arc::clone(&root),
        next: None,
    };
    dag::rewrite_up(root, &mut |p| Ok(r.join(p)))
}

struct Restrict<'a> {
    est: Estimator<'a>,
    /// The plan being rewritten, for [`Self::next`].
    root: Arc<LogicalPlan>,
    /// One past every attribute id the plan holds, Shift ranges included:
    /// computed at the first restriction (most plans have none).
    next: Option<u64>,
}

/// One restricting equality: `agg = other`, each over its own side.
struct Pair<'e> {
    agg: &'e Arc<Expr>,
    other: &'e Arc<Expr>,
    eq: &'e krishiv_plan::ResolvedFunction,
}

impl Restrict<'_> {
    fn join(&mut self, p: Arc<LogicalPlan>) -> Arc<LogicalPlan> {
        let PlanNode::Join {
            left,
            right,
            kind,
            condition: JoinCondition::On(cond),
        } = &p.node
        else {
            return p;
        };
        // Which sides drop their unmatched rows.
        let (l, r) = match kind {
            JoinKind::Inner | JoinKind::Cross | JoinKind::LeftSemi => (true, true),
            // An existence join keeps every left row (`exists` says whether
            // it matched), so only the right side may be restricted.
            JoinKind::LeftOuter | JoinKind::LeftAnti | JoinKind::Existence(_) => (false, true),
            JoinKind::RightOuter => (true, false),
            _ => (false, false),
        };
        let conj = conjuncts(cond);
        let mut out_left = Arc::clone(left);
        let mut out_right = Arc::clone(right);
        if r && let Some(new) = self.side(right, left, &conj) {
            out_right = new;
        }
        if l && let Some(new) = self.side(left, right, &conj) {
            out_left = new;
        }
        if Arc::ptr_eq(&out_left, left) && Arc::ptr_eq(&out_right, right) {
            return p;
        }
        let mut node = LogicalPlan::clone(&p);
        if let PlanNode::Join { left, right, .. } = &mut node.node {
            *left = out_left;
            *right = out_right;
        }
        Arc::new(node)
    }

    /// `side` with the aggregate at its top restricted to `other`'s keys
    /// under the equalities among `conj`, when that applies.
    fn side(
        &mut self,
        side: &Arc<LogicalPlan>,
        other: &Arc<LogicalPlan>,
        conj: &[Arc<Expr>],
    ) -> Option<Arc<LogicalPlan>> {
        let mine = output_ids(side)?;
        let theirs = output_ids(other)?;
        let mut pairs: Vec<Pair<'_>> = conj
            .iter()
            .filter_map(|c| {
                let (op, args, eq) = builtin(c)?;
                let [a, b] = args else { return None };
                if op != Builtin::Eq {
                    return None;
                }
                // A subquery would be instantiated twice, unshifted.
                if has_subquery(c) {
                    return None;
                }
                let (ra, rb) = (references(a), references(b));
                let over = |r: &Attrs, s: &Attrs| !r.is_empty() && r.is_subset(s);
                if over(&ra, &mine) && over(&rb, &theirs) {
                    Some(Pair {
                        agg: a,
                        other: b,
                        eq,
                    })
                } else if over(&rb, &mine) && over(&ra, &theirs) {
                    Some(Pair {
                        agg: b,
                        other: a,
                        eq,
                    })
                } else {
                    None
                }
            })
            .collect();
        if pairs.is_empty() {
            return None;
        }
        // Down to the aggregate, each pair's aggregate side rewritten over
        // the node below.
        let mut path = Vec::new();
        let mut keys: Vec<Arc<Expr>> = pairs.iter().map(|p| Arc::clone(p.agg)).collect();
        let mut at = Arc::clone(side);
        loop {
            if !node_exprs_deterministic(&at) {
                return None;
            }
            match &at.node {
                PlanNode::Aggregate { .. } => break,
                PlanNode::Filter { input, .. } => {
                    let next = Arc::clone(input);
                    path.push(at);
                    at = next;
                }
                // Through inner joins too, into the input that holds the
                // keys: a group that meets no value of `other` only
                // feeds rows the join above drops. Join ordering moves
                // aggregates in and out of such trees, so without this the
                // result would depend on the order the query wrote.
                PlanNode::Join {
                    left,
                    right,
                    kind: JoinKind::Inner | JoinKind::Cross,
                    ..
                } => {
                    let within = |c: &Arc<LogicalPlan>, k: &Arc<Expr>| {
                        output_ids(c).is_some_and(|o| references(k).is_subset(&o))
                    };
                    let next = if keys.iter().any(|k| within(left, k)) {
                        Arc::clone(left)
                    } else {
                        Arc::clone(right)
                    };
                    let keep: Vec<bool> = keys.iter().map(|k| within(&next, k)).collect();
                    let mut it = keep.iter();
                    pairs.retain(|_| *it.next().unwrap_or(&false));
                    let mut it = keep.iter();
                    keys.retain(|_| *it.next().unwrap_or(&false));
                    if keys.is_empty() {
                        return None;
                    }
                    path.push(at);
                    at = next;
                }
                PlanNode::Project { input, exprs } => {
                    let map: HashMap<AttrId, Arc<Expr>> = exprs
                        .iter()
                        .filter_map(|e| {
                            let out = crate::expr::output_of(e)?;
                            Some((out.id, Arc::clone(strip_alias(e))))
                        })
                        .collect();
                    keys = keys.iter().map(|k| substitute(k, &map)).collect();
                    let next = Arc::clone(input);
                    path.push(at);
                    at = next;
                }
                _ => return None,
            }
        }
        // A plain GROUP BY (grouping sets add rows with NULL keys).
        let (input, groups, aggregates) = match &at.node {
            PlanNode::Aggregate {
                input,
                grouping: Grouping::GroupBy(groups),
                aggregates,
            } if !groups.is_empty() && repeatable(input) => (input, groups, aggregates),
            _ => return None,
        };
        // Each grouping key's output attribute, to its expression.
        let by_output: HashMap<AttrId, Arc<Expr>> = aggregates
            .iter()
            .filter_map(|a| {
                let g = groups.iter().find(|g| same(strip_alias(a), g))?;
                Some((crate::expr::output_of(a)?.id, Arc::clone(g)))
            })
            .collect();
        // The pairs whose aggregate side reads grouping keys only, over
        // the aggregate's input.
        let mut inner = Vec::new();
        let mut kept = Vec::new();
        for (pair, key) in pairs.iter().zip(&keys) {
            let reads = references(key);
            let ok = reads.iter().all(|id| {
                by_output
                    .get(id)
                    .and_then(|g| crate::expr::data_type(g))
                    .is_some_and(|t| key_type(&t))
            });
            // The semi join evaluates both sides on rows the join never
            // reads (groups a HAVING drops, domain rows below a filter), so
            // neither may raise: columns and integral widenings only.
            if ok && !reads.is_empty() && cannot_raise(key) && cannot_raise(pair.other) {
                inner.push(substitute(key, &by_output));
                kept.push(pair);
            }
        }
        if kept.is_empty() || restricted(input, &inner) {
            return None;
        }
        // The domain: the deepest subtree of `other` that still holds the
        // keys' values without many more of them.
        let mut want = Attrs::new();
        for p in &kept {
            want.extend(references(p.other));
        }
        let domain = self.domain(other, &want)?;
        if !repeatable(&domain) {
            return None;
        }
        // Pays off: few keys against many groups, a domain that drops
        // most of what it reads (without distinct counts the groups are
        // overestimated, and a whole table's keys restrict nothing), and
        // a domain no costlier to read than the aggregate's input.
        let groups_est = self.est.plan(&at).rows;
        let dom_est = self.est.plan(&domain);
        let dom_scanned = self.scanned(&domain);
        if key_ndv(&dom_est, &want) * LOPSIDED > groups_est
            || dom_est.rows * LOPSIDED > dom_scanned
            || dom_scanned > self.scanned(input)
        {
            return None;
        }
        // Fresh ids for the domain's instance.
        let offset = match self.next {
            Some(n) => n,
            None => max_id(&self.root, &mut HashMap::new()).saturating_add(1),
        };
        let top = max_id(&domain, &mut HashMap::new());
        self.next = Some(
            offset
                .checked_add(top)?
                .checked_add(1)
                .filter(|n| *n <= MAX_ID)?,
        );
        let cols: Vec<BoundColumn> = output(&domain)?
            .into_iter()
            .filter(|c| want.contains(&c.id))
            .collect();
        let exprs = cols.iter().map(column).collect();
        let projected = Arc::new(LogicalPlan::new(PlanNode::Project {
            input: domain,
            exprs,
        }));
        let shifted = Arc::new(LogicalPlan::new(PlanNode::Shift {
            input: projected,
            offset,
        }));
        let eqs = kept.iter().zip(inner);
        let cond =
            and_all(eqs.map(|(p, i)| recall(p.eq, Builtin::Eq, vec![i, shift(p.other, offset)])))?;
        let semi = Arc::new(LogicalPlan::new(PlanNode::Join {
            left: Arc::clone(input),
            right: shifted,
            kind: JoinKind::LeftSemi,
            condition: JoinCondition::On(cond),
        }));
        // The semi join may restrict an aggregate below it in turn (one
        // pass, so the result is a fixed point).
        let semi = self.join(semi);
        let mut out = with_input(&at, semi);
        let mut below = at;
        for node in path.iter().rev() {
            let rebuilt = match &node.node {
                PlanNode::Join { left, right, .. } => {
                    let mut join = LogicalPlan::clone(node);
                    if let PlanNode::Join {
                        left: l, right: r, ..
                    } = &mut join.node
                    {
                        if Arc::ptr_eq(left, &below) {
                            *l = out;
                        } else {
                            debug_assert!(Arc::ptr_eq(right, &below));
                            *r = out;
                        }
                    }
                    Arc::new(join)
                }
                _ => with_input(node, out),
            };
            below = Arc::clone(node);
            out = rebuilt;
        }
        Some(out)
    }

    /// The deepest subtree of `p` that holds every value the attributes
    /// `want` take in `p`, while its estimated distinct keys stay within
    /// [`DESCEND`] times `p`'s.
    fn domain(&mut self, p: &Arc<LogicalPlan>, want: &Attrs) -> Option<Arc<LogicalPlan>> {
        let mut at = Arc::clone(p);
        loop {
            let next = match &at.node {
                PlanNode::Filter { input, .. } => Some(Arc::clone(input)),
                // Only columns passed through unchanged.
                PlanNode::Project { input, exprs } => {
                    let through: Attrs = exprs
                        .iter()
                        .filter_map(|e| match &e.node {
                            ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
                            _ => None,
                        })
                        .collect();
                    want.is_subset(&through).then(|| Arc::clone(input))
                }
                // A side's columns hold its values or NULL, which matches
                // nothing.
                PlanNode::Join {
                    left, right, kind, ..
                } => {
                    let in_left = output_ids(left).is_some_and(|o| want.is_subset(&o));
                    let in_right = !matches!(
                        kind,
                        JoinKind::LeftSemi | JoinKind::LeftAnti | JoinKind::Existence(_)
                    ) && output_ids(right).is_some_and(|o| want.is_subset(&o));
                    if in_left {
                        Some(Arc::clone(left))
                    } else if in_right {
                        Some(Arc::clone(right))
                    } else {
                        None
                    }
                }
                _ => None,
            };
            let Some(next) = next else {
                return Some(at);
            };
            let here = key_ndv(&self.est.plan(&at), want);
            let there = key_ndv(&self.est.plan(&next), want);
            if there > here * DESCEND {
                return Some(at);
            }
            at = next;
        }
    }

    /// The estimated rows the scans of `p` read.
    fn scanned(&mut self, p: &Arc<LogicalPlan>) -> f64 {
        let mut seen = HashSet::new();
        let mut stack = vec![Arc::clone(p)];
        let mut total = 0.0;
        while let Some(x) = stack.pop() {
            if seen.insert(Arc::as_ptr(&x)) {
                if matches!(x.node, PlanNode::Scan(_)) {
                    total += self.est.plan(&x).rows;
                }
                x.for_each_child(&mut |c| stack.push(Arc::clone(c)));
            }
        }
        total
    }
}

/// The estimated distinct values of the attributes `want` together.
fn key_ndv(est: &Rc<Est>, want: &Attrs) -> f64 {
    want.iter()
        .map(|id| est.cols.get(id).map_or(est.rows, |c| c.ndv))
        .product::<f64>()
        .min(est.rows)
}

/// Whether `input` is already reduced by a semi join on `keys` (this
/// pass's own, or one the query wrote): restricting again adds little.
/// Only the join's kind and condition count, never the shape of its
/// right side, which pruning changes between optimizer runs (a `Shift`
/// gains a projection above it), so the answer is the same on every run
/// and the optimizer stays a fixed point.
fn restricted(input: &Arc<LogicalPlan>, keys: &[Arc<Expr>]) -> bool {
    let mut at = input;
    loop {
        match &at.node {
            PlanNode::Filter { input, .. } | PlanNode::Project { input, .. } => at = input,
            PlanNode::Join {
                kind: JoinKind::LeftSemi,
                condition: JoinCondition::On(cond),
                left,
                ..
            } => {
                let conj = conjuncts(cond);
                let has = |k: &Arc<Expr>| {
                    conj.iter().any(|c| {
                        builtin(c).is_some_and(|(op, args, _)| {
                            op == Builtin::Eq && args.iter().any(|a| same(a, k))
                        })
                    })
                };
                if keys.iter().all(has) {
                    return true;
                }
                at = left;
            }
            _ => return false,
        }
    }
}

/// Whether evaluating `e` can never raise: a column, or a widening of an
/// integral value to a wider integral type (deterministic by construction).
fn cannot_raise(e: &Arc<Expr>) -> bool {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(_)) => true,
        ExprNode::Cast { expr, to, .. } => {
            let rank = |t: &DataType| match t {
                DataType::Byte => Some(1),
                DataType::Short => Some(2),
                DataType::Integer => Some(3),
                DataType::Long => Some(4),
                _ => None,
            };
            let from = crate::expr::data_type(expr);
            matches!((from.as_ref().and_then(rank), rank(to)), (Some(a), Some(b)) if a <= b)
                && cannot_raise(expr)
        }
        _ => false,
    }
}

/// Types whose grouping and `=` agree value for value.
fn key_type(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Boolean
            | DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Long
            | DataType::String
            | DataType::Binary
            | DataType::Decimal(_)
            | DataType::Date
            | DataType::Timestamp
            | DataType::TimestampNtz
    )
}

fn output(p: &LogicalPlan) -> Option<Vec<BoundColumn>> {
    p.resolved_output()
}

fn output_ids(p: &LogicalPlan) -> Option<Attrs> {
    Some(output(p)?.into_iter().map(|c| c.id).collect())
}

/// Whether this node's own expressions call only deterministic functions.
fn node_exprs_deterministic(p: &LogicalPlan) -> bool {
    let mut ok = true;
    p.for_each_expr(&mut |e| ok &= is_deterministic(e));
    ok
}

/// Whether `p` computes the same rows each time and may be instantiated
/// again under a [`PlanNode::Shift`]: only relations whose rows are a
/// function of their inputs (no limit or sample; no CTE reference, which a
/// shift does not renumber), deterministic expressions (subquery plans
/// included) and no outer references.
fn repeatable(p: &Arc<LogicalPlan>) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![Arc::clone(p)];
    while let Some(x) = stack.pop() {
        if seen.insert(Arc::as_ptr(&x)) {
            let plain = matches!(
                x.node,
                PlanNode::Scan(_)
                    | PlanNode::Filter { .. }
                    | PlanNode::Project { .. }
                    | PlanNode::Join { .. }
                    | PlanNode::Aggregate { .. }
                    | PlanNode::Shift { .. }
            );
            let mut ok = plain;
            x.for_each_expr(&mut |e| ok &= is_deterministic(e) && !has_outer_reference(e));
            if !ok {
                return false;
            }
            x.for_each_child(&mut |c| stack.push(Arc::clone(c)));
        }
    }
    true
}

/// `e` with every column id raised by `offset` (no subqueries inside: the
/// restricting equalities never hold one).
fn shift(e: &Arc<Expr>, offset: u64) -> Arc<Expr> {
    let out = Arc::clone(e).transform_up(&mut |x| {
        Ok::<_, std::convert::Infallible>(match &x.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => {
                let id = AttrId(c.id.0 + offset);
                Arc::new(
                    Expr::new(ExprNode::Resolved(ResolvedExpr::Column(BoundColumn {
                        id,
                        ..c.clone()
                    })))
                    .with_origin(Origin::UNKNOWN),
                )
            }
            _ => x,
        })
    });
    match out {
        Ok(v) => v,
        Err(never) => match never {},
    }
}

/// The largest attribute id `p` holds: outputs, references and subquery
/// plans, each shift at its offset.
pub(crate) fn max_id(p: &Arc<LogicalPlan>, memo: &mut HashMap<*const LogicalPlan, u64>) -> u64 {
    if let Some(m) = memo.get(&Arc::as_ptr(p)) {
        return *m;
    }
    let mut m = output(p)
        .unwrap_or_default()
        .iter()
        .map(|c| c.id.0)
        .max()
        .unwrap_or(0);
    let mut kids = Vec::new();
    p.for_each_child(&mut |c| kids.push(Arc::clone(c)));
    let offset = match &p.node {
        PlanNode::Shift { offset, .. } => *offset,
        _ => 0,
    };
    for c in &kids {
        m = m.max(max_id(c, memo).saturating_add(offset));
    }
    let mut exprs = Vec::new();
    p.for_each_expr(&mut |e| exprs.push(Arc::clone(e)));
    for e in &exprs {
        let mut ids = Attrs::new();
        crate::expr::references_into(e, &mut ids);
        m = m.max(
            ids.iter()
                .map(|i| i.0)
                .max()
                .unwrap_or(0)
                .saturating_add(offset),
        );
        for plan in subquery_plans(e) {
            m = m.max(max_id(&plan, memo).saturating_add(offset));
        }
    }
    memo.insert(Arc::as_ptr(p), m);
    m
}

/// The subquery plans directly inside `e`.
pub(crate) fn subquery_plans(e: &Arc<Expr>) -> Vec<Arc<LogicalPlan>> {
    let mut out = Vec::new();
    let mut stack = vec![Arc::clone(e)];
    let mut seen = HashSet::new();
    while let Some(x) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&x)) {
            continue;
        }
        if let ExprNode::Subquery { plan, .. } = &x.node {
            out.push(Arc::clone(plan));
        }
        x.for_each_child(&mut |c| stack.push(Arc::clone(c)));
    }
    out
}
