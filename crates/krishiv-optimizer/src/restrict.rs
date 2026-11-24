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
//! `A` does (a superset of the domain is enough). Through an inner join of
//! `A` with a plain `o = y` (one type), `D` may be the subtree holding `y`
//! instead, the semi join then on `g = y`: each row of the join has `o` equal
//! to `y`. Of the subtrees that keep about as few keys, the domain is the one
//! scanning fewest rows, so how join ordering placed the aggregate does not
//! decide it (TPC-H q02's minimum is restricted to the filtered parts
//! whether it joins part or partsupp joined to part). Each kept group sees all
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
//! ## The preserved side of an outer join (perf lever K6)
//!
//! With `krishiv.optimizer.preservedSideSemiJoin` on, the walk down from
//! the join may also stop at a `LEFT` or `RIGHT OUTER` join (never `FULL`)
//! whose preserved input holds every key, through filters, projections and
//! inner joins only:
//!
//! ```text
//! Join(A, OuterJoin(P, N), k = o)
//!   =>  Join(A, OuterJoin(P ⋉ Shift(π_o(D)), N), k = o)
//! ```
//!
//! Sound because each row of `P` reaches the outer join's output with its
//! own key values (matched or null-extended), and the null-supplying side
//! `N` emits no row of its own: a row of `P` whose keys meet no value of
//! `A` only feeds output rows the join above drops, matched or not. The
//! semi join sits on `P`, never on `N`. Its gates, on top of the ones
//! above for the path and the equalities: every node on the path is
//! deterministic, the domain is `repeatable` (computed twice), the keys
//! are columns or integral widenings of `P`'s columns, statistics say the
//! semi join keeps at most a quarter of `P` and `N` estimates at least
//! twice the reduced `P` (so the hash join can build `P`), and `P` does
//! not already hold a semi join on the keys (`reduced`). `P` itself need
//! not be deterministic: the semi join only drops rows of it after they
//! were computed once, so a nondeterministic `P` keeps its values.
//!
//! ## Errors
//!
//! Like a runtime filter, the restriction may skip rows whose evaluation
//! would have raised an ANSI error (a `sum` that overflows in a group no
//! row asks for); it never changes a value. On the preserved side the
//! skipped work is wider: the outer join's condition and every filter and
//! projection between it and the join above are no longer evaluated on the
//! removed rows, so an error Spark raises there (a failing cast in a
//! projection above the outer join, on a row the inner join later drops)
//! is not raised. This is a documented divergence from Spark, which
//! computes every group and every row.

use crate::expr::AttrMap;
use std::sync::Arc;

use krishiv_common::error::KrishivError;
use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, Grouping, JoinCondition, JoinKind, LogicalPlan,
    Origin, PlanNode, ResolvedExpr,
};
use krishiv_types::DataType;

use crate::dag::{self, with_input};
use crate::estimate::{Estimator, Memo};
use crate::expr::{
    Attrs, and_all, builtin, column, conjuncts, has_outer_reference, has_subquery,
    is_deterministic, recall, references, same, strip_alias, substitute,
};
use crate::stats::{RelStats, Statistics};

/// The domain must have at most `1 / LOPSIDED` of the aggregate's groups.
const LOPSIDED: f64 = 2.0;

/// The domain moves into a subtree of the other side only while its keys
/// grow by at most this factor.
const DESCEND: f64 = 2.0;

/// The largest attribute id a plan may hold (as the analyzer allows).
pub(crate) const MAX_ID: u64 = 1 << 62;

/// The preserved side's estimated rows after the semi join must be at most
/// `1 / PRESERVED_KEEP` of its rows before.
const PRESERVED_KEEP: f64 = 4.0;

/// The outer join's other input must estimate at least this many times the
/// preserved side's rows after the semi join (the physical planner's
/// `STATS_FACTOR`: only then can it build the preserved side).
const PRESERVED_FLIP: f64 = 2.0;

/// Restricts the aggregates below joins in `root` (see the module docs);
/// with `preserved`, also the preserved sides of outer joins below them
/// (`krishiv.optimizer.preservedSideSemiJoin`, module docs). Estimates start
/// from `memo` and are left in it.
pub(crate) fn restrict(
    root: Arc<LogicalPlan>,
    stats: &dyn Statistics,
    propagate: bool,
    preserved: bool,
    memo: &mut Memo,
) -> Result<Arc<LogicalPlan>, KrishivError> {
    let mut r = Restrict {
        est: Estimator::resume(stats, propagate, std::mem::take(memo)),
        root: Arc::clone(&root),
        next: None,
        preserved,
        reaches: Default::default(),
    };
    let out = dag::rewrite_up(root, &mut |p| Ok(r.join(p)));
    *memo = r.est.into_memo();
    out
}

struct Restrict<'a> {
    est: Estimator<'a>,
    /// The plan being rewritten, for [`Self::next`].
    root: Arc<LogicalPlan>,
    /// One past every attribute id the plan holds, Shift ranges included:
    /// computed at the first restriction (most plans have none).
    next: Option<u64>,
    /// Whether the preserved side of an outer join is a target too.
    preserved: bool,
    /// Per relation (kept alive, so its address stays unique): whether
    /// [`Self::side`]'s descent can reach a target from it.
    reaches: crate::expr::AddrMap<LogicalPlan, (Arc<LogicalPlan>, bool)>,
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
            JoinKind::LeftOuter
            | JoinKind::Single
            | JoinKind::LeftAnti
            | JoinKind::Existence(_) => (false, true),
            JoinKind::RightOuter => (true, false),
            _ => (false, false),
        };
        // Most sides hold no aggregate the descent reaches: those skip
        // collecting either side's outputs.
        let r = r && self.reaches(right);
        let l = l && self.reaches(left);
        if !l && !r {
            return p;
        }
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

    /// Whether [`Self::side`]'s descent from `p` can reach a target: an
    /// aggregate or, when [`Self::preserved`], an outer join, through
    /// filters, projections and inner joins only (either input of a join,
    /// so a superset of what the descent follows).
    fn reaches(&mut self, p: &Arc<LogicalPlan>) -> bool {
        if let Some((_, r)) = self.reaches.get(&Arc::as_ptr(p)) {
            return *r;
        }
        let r = match &p.node {
            PlanNode::Aggregate { .. } => true,
            PlanNode::Join {
                kind: JoinKind::LeftOuter | JoinKind::RightOuter,
                ..
            } => self.preserved,
            PlanNode::Filter { input, .. } | PlanNode::Project { input, .. } => self.reaches(input),
            PlanNode::Join {
                left,
                right,
                kind: JoinKind::Inner | JoinKind::Cross,
                ..
            } => self.reaches(left) || self.reaches(right),
            _ => false,
        };
        self.reaches.insert(Arc::as_ptr(p), (Arc::clone(p), r));
        r
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
                // The preserved side of an outer join: every row of it reaches
                // the join's output with its own values, so one whose keys
                // meet no value of `other` only feeds rows the join above
                // drops.
                PlanNode::Join {
                    left,
                    right,
                    kind: kind @ (JoinKind::LeftOuter | JoinKind::RightOuter),
                    ..
                } if self.preserved => {
                    let (kept, nulls) = if matches!(kind, JoinKind::LeftOuter) {
                        (left, right)
                    } else {
                        (right, left)
                    };
                    let within = output_ids(kept)
                        .is_some_and(|o| keys.iter().all(|k| references(k).is_subset(&o)));
                    if !within {
                        return None;
                    }
                    let (kept, nulls) = (Arc::clone(kept), Arc::clone(nulls));
                    path.push(at);
                    return self.preserved_side(&path, &kept, &nulls, &pairs, &keys, other);
                }
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
                    let map: AttrMap<Arc<Expr>> = exprs
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
        let by_output: AttrMap<Arc<Expr>> = aggregates
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
        let mut want = Attrs::default();
        for p in &kept {
            want.extend(references(p.other));
        }
        let Domain {
            plan: domain,
            want,
            rename,
        } = self.domain(other, &want)?;
        if !repeatable(&domain) {
            return None;
        }
        // Pays off: few keys against many groups, a domain that drops
        // most of what it reads (without distinct counts the groups are
        // overestimated, and a whole table's keys restrict nothing), and
        // a domain no costlier to read than the aggregate's input.
        let groups_est = self.est.plan(&at).rows_for_costing();
        let dom_est = self.est.plan(&domain);
        let dom_scanned = self.scanned(&domain);
        if key_ndv(&dom_est, &want) * LOPSIDED > groups_est
            || dom_est.rows_for_costing() * LOPSIDED > dom_scanned
            || dom_scanned > self.scanned(input)
        {
            return None;
        }
        let semi = self.semi(input, &kept, inner, domain, &want, &rename)?;
        // The semi join may restrict an aggregate below it in turn (one
        // pass, so the result is a fixed point).
        let semi = self.join(semi);
        let out = with_input(&at, semi);
        Some(rebuild(&path, at, out))
    }

    /// The target of [`Self::side`] when its walk meets an outer join whose
    /// preserved input `kept` holds the keys (`path` ends at that join):
    /// `kept` semi-joined to `other`'s domain, when the reduction lets the
    /// physical planner build `kept` instead of `nulls`, the outer join's
    /// other input (TPC-DS q40, q80: a small filtered fact against all of
    /// its returns).
    fn preserved_side(
        &mut self,
        path: &[Arc<LogicalPlan>],
        kept: &Arc<LogicalPlan>,
        nulls: &Arc<LogicalPlan>,
        pairs: &[Pair<'_>],
        keys: &[Arc<Expr>],
        other: &Arc<LogicalPlan>,
    ) -> Option<Arc<LogicalPlan>> {
        // `other`'s domain is computed twice, so it must agree with itself;
        // `kept` need not be deterministic: the semi join sits above it and
        // only drops rows computed once (module docs).
        if !repeatable(other) {
            return None;
        }
        // Each key is evaluated on every row of `kept` (the join above reads
        // them all, as the outer join keeps them), the domain side on rows of
        // the domain the join may never read: neither may raise.
        let mut inner = Vec::new();
        let mut kept_pairs = Vec::new();
        for (pair, key) in pairs.iter().zip(keys) {
            let typed = crate::expr::data_type(key).is_some_and(|t| key_type(&t));
            if typed && cannot_raise(key) && cannot_raise(pair.other) {
                inner.push(Arc::clone(key));
                kept_pairs.push(pair);
            }
        }
        if kept_pairs.is_empty() || reduced(kept, &inner) {
            return None;
        }
        let mut want = Attrs::default();
        for p in &kept_pairs {
            want.extend(references(p.other));
        }
        let Domain {
            plan: domain,
            want,
            rename,
        } = self.domain(other, &want)?;
        if !repeatable(&domain) {
            return None;
        }
        // The domain drops most of what it reads (a whole table's keys
        // restrict nothing) and reads no more than `kept` does.
        let dom_est = self.est.plan(&domain);
        let dom_scanned = self.scanned(&domain);
        if dom_est.rows_for_costing() * LOPSIDED > dom_scanned || dom_scanned > self.scanned(kept) {
            return None;
        }
        let next = self.next;
        let semi = self.semi(kept, &kept_pairs, inner, domain, &want, &rename)?;
        // Pays off: the semi join keeps at most a quarter of `kept`, and
        // the outer join's other input is then at least twice as large, so the
        // planner can build the reduced side.
        let before = self.est.plan(kept).rows_for_costing();
        let after = self.est.plan(&semi).rows_for_costing();
        let nulls_rows = self.est.plan(nulls).rows_for_costing();
        if after * PRESERVED_KEEP > before || nulls_rows < after * PRESERVED_FLIP {
            // The ids taken for the refused instance are given back.
            self.next = next;
            return None;
        }
        let at = Arc::clone(kept);
        Some(rebuild(path, at, semi))
    }

    /// `input ⋉ Shift(π_want(domain))` on the equalities of `kept`, each
    /// pair's aggregate side as `inner` reads it over `input`.
    fn semi(
        &mut self,
        input: &Arc<LogicalPlan>,
        kept: &[&Pair<'_>],
        inner: Vec<Arc<Expr>>,
        domain: Arc<LogicalPlan>,
        want: &Attrs,
        rename: &AttrMap<Arc<Expr>>,
    ) -> Option<Arc<LogicalPlan>> {
        // Fresh ids for the domain's instance.
        let offset = match self.next {
            Some(n) => n,
            None => max_id(&self.root, &mut Default::default()).saturating_add(1),
        };
        let top = max_id(&domain, &mut Default::default());
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
        let cond = and_all(eqs.map(|(p, i)| {
            recall(
                p.eq,
                Builtin::Eq,
                vec![i, shift(&substitute(p.other, rename), offset)],
            )
        }))?;
        Some(Arc::new(LogicalPlan::new(PlanNode::Join {
            left: Arc::clone(input),
            right: shifted,
            kind: JoinKind::LeftSemi,
            condition: JoinCondition::On(cond),
        })))
    }

    /// The deepest subtree of `p` that holds every value the attributes
    /// `want` take in `p` (under the columns an inner join's equalities
    /// rename them to), while its estimated distinct keys stay within
    /// [`DESCEND`] times `p`'s; of several such subtrees at a step, the one
    /// scanning fewest rows.
    fn domain(&mut self, p: &Arc<LogicalPlan>, want: &Attrs) -> Option<Domain> {
        let mut at = Arc::clone(p);
        let mut want = want.clone();
        // Each wanted column as `other` reads it, the column it is read as
        // now, and that column where an equality moved it to another input.
        let mut track: Vec<(AttrId, AttrId, Option<BoundColumn>)> =
            want.iter().map(|&id| (id, id, None)).collect();
        loop {
            // The subtrees the domain may move into, each with the columns
            // an equality of the join renames `want` to there.
            let mut next: Vec<(Arc<LogicalPlan>, AttrMap<BoundColumn>)> = Vec::new();
            match &at.node {
                PlanNode::Filter { input, .. } => {
                    next.push((Arc::clone(input), AttrMap::default()));
                }
                // Only columns passed through unchanged.
                PlanNode::Project { input, exprs } => {
                    let through: Attrs = exprs
                        .iter()
                        .filter_map(|e| match &e.node {
                            ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
                            _ => None,
                        })
                        .collect();
                    if want.is_subset(&through) {
                        next.push((Arc::clone(input), AttrMap::default()));
                    }
                }
                // A side's columns hold its values or NULL, which matches
                // nothing.
                PlanNode::Join {
                    left,
                    right,
                    kind,
                    condition,
                } => {
                    let in_left = output_ids(left).is_some_and(|o| want.is_subset(&o));
                    let in_right = !matches!(
                        kind,
                        JoinKind::LeftSemi | JoinKind::LeftAnti | JoinKind::Existence(_)
                    ) && output_ids(right).is_some_and(|o| want.is_subset(&o));
                    if in_left {
                        next.push((Arc::clone(left), AttrMap::default()));
                    } else if in_right {
                        next.push((Arc::clone(right), AttrMap::default()));
                    }
                    // Through an inner join's equalities into the other
                    // input: a row of the join has `w = y`, so the join's
                    // values of `w` are among that input's values of `y`
                    // (TPC-H q02: partsupp's partkey is part's, so the
                    // filtered parts are the domain, not partsupp joined
                    // to them).
                    if let (JoinKind::Inner, JoinCondition::On(cond)) = (kind, condition) {
                        for side in [right, left] {
                            if let Some(map) = partners(cond, &want, side) {
                                next.push((Arc::clone(side), map));
                            }
                        }
                    }
                }
                _ => {}
            }
            let here = key_ndv(&self.est.plan(&at), &want);
            // Of the subtrees whose keys grow by at most `DESCEND`, the one
            // that scans the fewest rows (the first on a tie).
            let mut best: Option<(f64, usize, Attrs)> = None;
            for (k, (c, map)) in next.iter().enumerate() {
                let moved: Attrs = want
                    .iter()
                    .map(|id| map.get(id).map_or(*id, |b| b.id))
                    .collect();
                let there = key_ndv(&self.est.plan(c), &moved);
                if there > here * DESCEND {
                    continue;
                }
                let cost = self.scanned(c);
                if best.as_ref().is_none_or(|(b, _, _)| cost < *b) {
                    best = Some((cost, k, moved));
                }
            }
            let Some((_, k, moved)) = best else {
                let rename = track
                    .into_iter()
                    .filter_map(|(id, _, c)| Some((id, column(&c?))))
                    .collect();
                return Some(Domain {
                    plan: at,
                    want,
                    rename,
                });
            };
            let (c, map) = next.swap_remove(k);
            for t in &mut track {
                if let Some(b) = map.get(&t.1) {
                    t.1 = b.id;
                    t.2 = Some(b.clone());
                }
            }
            want = moved;
            at = c;
        }
    }

    /// The estimated rows the scans of `p` read.
    fn scanned(&mut self, p: &Arc<LogicalPlan>) -> f64 {
        let mut seen = crate::expr::AddrSet::default();
        let mut stack = vec![Arc::clone(p)];
        let mut total = 0.0;
        while let Some(x) = stack.pop() {
            if seen.insert(Arc::as_ptr(&x)) {
                if matches!(x.node, PlanNode::Scan(_)) {
                    total += self.est.plan(&x).rows_for_costing();
                }
                x.for_each_child(&mut |c| stack.push(Arc::clone(c)));
            }
        }
        total
    }
}

/// `path` (from the joined side down to the node above `below`) rebuilt over
/// `out` in place of `below`.
fn rebuild(
    path: &[Arc<LogicalPlan>],
    mut below: Arc<LogicalPlan>,
    mut out: Arc<LogicalPlan>,
) -> Arc<LogicalPlan> {
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
    out
}

/// The domain of a restriction: the subtree, the columns it holds the keys
/// in, and the columns `other` reads that it holds as others (moved by
/// join equalities, `w` to `y` of `w = y`).
struct Domain {
    plan: Arc<LogicalPlan>,
    want: Attrs,
    rename: AttrMap<Arc<Expr>>,
}

/// For every column of `want` (none of them in `side`), a column of `side`
/// it equals under a conjunct of `cond`: a plain `=` of two columns of one
/// type that groups as it compares (a row of the join then holds both, and
/// one value).
fn partners(
    cond: &Arc<Expr>,
    want: &Attrs,
    side: &Arc<LogicalPlan>,
) -> Option<AttrMap<BoundColumn>> {
    let cols = output(side)?;
    if want.iter().any(|w| cols.iter().any(|c| c.id == *w)) {
        return None;
    }
    let col = |e: &Arc<Expr>| match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.clone()),
        _ => None,
    };
    let mut map = AttrMap::default();
    for c in conjuncts(cond) {
        let Some((Builtin::Eq, [a, b], _)) = builtin(&c) else {
            continue;
        };
        let (Some(a), Some(b)) = (col(a), col(b)) else {
            continue;
        };
        if a.data_type != b.data_type || !key_type(&a.data_type) {
            continue;
        }
        for (w, y) in [(&a, &b), (&b, &a)] {
            if want.contains(&w.id) && cols.iter().any(|c| c.id == y.id) {
                map.entry(w.id).or_insert_with(|| y.clone());
            }
        }
    }
    (map.len() == want.len()).then_some(map)
}

/// The estimated distinct values of the attributes `want` together.
fn key_ndv(est: &Arc<RelStats>, want: &Attrs) -> f64 {
    want.iter()
        .map(|id| est.cols.get(id).map_or(est.rows_for_costing(), |c| c.ndv))
        .product::<f64>()
        .min(est.rows_for_costing())
}

/// Whether `input` is already reduced by a semi join on `keys` (this
/// pass's own, or one the query wrote): restricting again adds little.
/// The walk passes the left input of anti and existence joins too: semi/anti
/// ordering may sort this pass's semi join below one, and the reduction
/// still holds there (an anti join only drops rows, an existence join only
/// adds a column).
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
                kind: JoinKind::LeftAnti | JoinKind::Existence(_),
                left,
                ..
            } => at = left,
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

/// Whether the preserved input `kept` already holds a semi join on `keys`:
/// [`restricted`], also through inner joins into the input holding the keys
/// (where `LatePayloadJoin` leaves its key-only semi join), so neither rule
/// adds a second one and the plan stays a fixed point.
fn reduced(kept: &Arc<LogicalPlan>, keys: &[Arc<Expr>]) -> bool {
    let mut at = kept;
    loop {
        if restricted(at, keys) {
            return true;
        }
        match &at.node {
            PlanNode::Filter { input, .. } | PlanNode::Project { input, .. } => at = input,
            PlanNode::Join {
                kind: JoinKind::LeftAnti | JoinKind::Existence(_) | JoinKind::LeftSemi,
                left,
                ..
            } => at = left,
            PlanNode::Join {
                kind: JoinKind::Inner | JoinKind::Cross,
                left,
                right,
                ..
            } => {
                let holds = |c: &Arc<LogicalPlan>| {
                    output_ids(c).is_some_and(|o| keys.iter().all(|k| references(k).is_subset(&o)))
                };
                if holds(left) {
                    at = left;
                } else if holds(right) {
                    at = right;
                } else {
                    return false;
                }
            }
            _ => return false,
        }
    }
}

/// Whether evaluating `e` can never raise: a column, or a widening of an
/// integral value to a wider integral type (deterministic by construction).
pub(crate) fn cannot_raise(e: &Arc<Expr>) -> bool {
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
pub(crate) fn key_type(t: &DataType) -> bool {
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
    let mut ids = Attrs::default();
    crate::pushdown::collect_ids(p, &mut ids).then_some(ids)
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
pub(crate) fn repeatable(p: &Arc<LogicalPlan>) -> bool {
    let mut seen = crate::expr::AddrSet::default();
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
pub(crate) fn shift(e: &Arc<Expr>, offset: u64) -> Arc<Expr> {
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
pub(crate) fn max_id(
    p: &Arc<LogicalPlan>,
    memo: &mut crate::expr::AddrMap<LogicalPlan, u64>,
) -> u64 {
    if let Some(m) = memo.get(&Arc::as_ptr(p)) {
        return *m;
    }
    // A node that passes its inputs' columns on has no id of its own above
    // theirs (taken below); its output would be rebuilt from the whole
    // subtree at every level.
    let passes_on = match &p.node {
        PlanNode::Filter { .. }
        | PlanNode::Sort { .. }
        | PlanNode::Limit { .. }
        | PlanNode::Offset { .. }
        | PlanNode::Tail { .. }
        | PlanNode::WithCte { .. }
        | PlanNode::JoinHint { .. }
        | PlanNode::Repartition { .. }
        | PlanNode::Sample { .. }
        | PlanNode::ReplicateRows { .. } => true,
        PlanNode::Join { kind, .. } => !matches!(kind, JoinKind::Existence(_)),
        _ => false,
    };
    let mut m = if passes_on {
        0
    } else {
        output(p)
            .unwrap_or_default()
            .iter()
            .map(|c| c.id.0)
            .max()
            .unwrap_or(0)
    };
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
        let mut ids = Attrs::default();
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
    let mut seen = crate::expr::AddrSet::default();
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
