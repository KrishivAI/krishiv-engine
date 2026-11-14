// SPDX-License-Identifier: Apache-2.0
//! The estimator: what each operator leaves known about its output.
//!
//! Rows scale by selectivities; each column carries a distinct count, a null
//! fraction and bounds ([`AttrStats`]) through the operators, only ever
//! staying or widening from what the surviving rows hold:
//!
//! - **Filter**: see [`crate::narrow`]; every column the predicate does not
//!   read keeps the distinct values the kept rows are expected to hold
//!   ([`kept_distinct`]).
//! - **Join**: each equi-key's distinct count is the smaller of the two
//!   sides' (an outer join's preserved side keeps its own: all its values
//!   stay); an inner join's keys intersect their ranges and lose their
//!   nulls; the null-supplying side of an outer join gains the unmatched
//!   fraction as nulls; every other column keeps the distinct values the
//!   matching fraction of its side's rows is expected to hold
//!   ([`kept_distinct`]: a column of few values over many rows keeps them
//!   all, a unique column keeps that fraction).
//! - **Aggregate**: rows are the product of the keys' distinct counts, at
//!   most the input's; keys keep their statistics; `min`, `max` and `count`
//!   results get bounds from their input.
//! - **Project**: a column keeps its statistics, a cast keeps what it
//!   cannot disturb ([`carries`]), anything else is a guess.
//! - **Union**: rows add; distinct counts are the most of any input.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use krishiv_plan::{
    AttrId, BoundColumn, Builtin, Expr, ExprNode, FunctionId, FunctionKind, Grouping,
    JoinCondition, JoinKind, Literal, LogicalPlan, PlanNode, ResolvedExpr, Scan,
};
use krishiv_types::{DataType, DecimalType};

use crate::expr::{builtin, conjuncts, output_of, references, strip_alias};
use crate::limit::limit_value;
use crate::narrow::{View, apply};
use crate::stats::{
    AttrStats, DEFAULT_ROWS, RelStats, StatSource, Statistics, cap, cse_definitions, number,
    reads_cse, resolve_cse, to_f64,
};

/// How a cast carries a column's statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Carries {
    /// The distinct count, null fraction and bounds (injective, non-raising
    /// and order-preserving).
    Bounds,
    /// As [`Self::Bounds`], with the bounds converted from days to
    /// microseconds, widened by a day for the session time zone.
    DaysToMicros,
    /// The distinct count and null fraction only (injective and
    /// non-raising, but not order-preserving).
    Distinct,
}

/// What a cast from `from` to `to` carries: only a cast that is injective
/// and cannot raise or produce NULL for any value of `from`.
fn carries(from: &DataType, to: &DataType) -> Option<Carries> {
    use DataType as T;
    // The digits an integral type needs before the decimal point.
    let digits = |t: &DataType| match t {
        T::Byte => Some(3),
        T::Short => Some(5),
        T::Integer => Some(10),
        T::Long => Some(19),
        _ => None,
    };
    let room = |d: &DecimalType| i32::from(d.precision()) - i32::from(d.scale());
    if from == to {
        return Some(Carries::Bounds);
    }
    match (from, to) {
        (T::Byte, T::Short | T::Integer | T::Long)
        | (T::Short, T::Integer | T::Long)
        | (T::Integer, T::Long)
        | (T::Byte | T::Short, T::Float | T::Double)
        | (T::Integer | T::Float, T::Double) => Some(Carries::Bounds),
        (T::Byte | T::Short | T::Integer | T::Long, T::Decimal(d)) => {
            (digits(from)? <= room(d)).then_some(Carries::Bounds)
        }
        (T::Decimal(a), T::Decimal(b)) => {
            (b.scale() >= a.scale() && room(b) >= room(a)).then_some(Carries::Bounds)
        }
        (T::Date, T::Timestamp | T::TimestampNtz) => Some(Carries::DaysToMicros),
        (T::Byte | T::Short | T::Integer | T::Long | T::Date, T::String) => Some(Carries::Distinct),
        _ => None,
    }
}

/// Microseconds in a day.
const DAY_MICROS: f64 = 86_400_000_000.0;

/// What a projection or aggregate output `e` carries over from the input's
/// statistics: a column reference, a cast that [`carries`] them, the bounds
/// of `min`, `max` and `count`; `None` for anything else.
fn derived(e: &Expr, input: &RelStats, rows: f64) -> Option<AttrStats> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => input.cols.get(&c.id).cloned(),
        ExprNode::Cast { expr, to, .. } => {
            let ExprNode::Resolved(ResolvedExpr::Column(c)) = &expr.node else {
                return None;
            };
            let mut s = input.cols.get(&c.id)?.clone();
            match carries(&c.data_type, to)? {
                Carries::Bounds => {}
                Carries::DaysToMicros => {
                    s.min = s.min.map(|d| (d - 1.0) * DAY_MICROS);
                    s.max = s.max.map(|d| (d + 1.0) * DAY_MICROS);
                }
                Carries::Distinct => {
                    s.min = None;
                    s.max = None;
                }
            }
            Some(s)
        }
        // A session UDF may share a built-in's name: only registry
        // aggregates are known.
        ExprNode::Resolved(ResolvedExpr::Function(f))
            if f.kind == FunctionKind::Aggregate && matches!(f.id, FunctionId::Registry { .. }) =>
        {
            let mut s = AttrStats::guess(rows);
            match (f.id.name(), f.args.as_slice()) {
                ("min" | "max", [arg]) => {
                    let ExprNode::Resolved(ResolvedExpr::Column(c)) = &arg.node else {
                        return None;
                    };
                    let from = input.cols.get(&c.id)?;
                    (s.min, s.max) = (from.min, from.max);
                    // A group of NULLs yields NULL: the input's fraction bounds it.
                    s.null_frac = from.null_frac;
                }
                ("count", _) => (s.min, s.max) = (Some(0.0), Some(input.rows)),
                _ => return None,
            }
            Some(s)
        }
        _ => None,
    }
}

/// The estimate of a projection or aggregate's outputs over `rows` rows. A
/// column keeps its statistics; with `propagate`, so do casts and the
/// aggregates [`derived`] reads; everything else is a guess.
fn outputs(input: &RelStats, exprs: &[Arc<Expr>], rows: f64, propagate: bool) -> RelStats {
    let cols = exprs
        .iter()
        .filter_map(|e| {
            let out = output_of(e)?;
            let inner = strip_alias(e);
            let col = if propagate {
                derived(inner, input, rows)
            } else {
                match &inner.node {
                    ExprNode::Resolved(ResolvedExpr::Column(c)) => input.cols.get(&c.id).cloned(),
                    _ => None,
                }
            };
            Some((out.id, col.unwrap_or_else(|| AttrStats::guess(rows))))
        })
        .collect();
    RelStats {
        rows,
        cols,
        from_statistics: input.from_statistics,
        // A projection keeps the rows; an aggregate does not.
        unfiltered_rows: if rows == input.rows {
            input.unfiltered_rows
        } else {
            0.0
        },
    }
    .with_rows(rows)
}

/// The rows left of `rows` after a predicate of selectivity `sel`: at least
/// one, unless the predicate proves the relation empty.
fn kept_rows(rows: f64, sel: f64, empty: bool) -> f64 {
    if empty || rows <= 0.0 {
        0.0
    } else {
        (rows * sel).max(1.0)
    }
}

/// `est` filtered by `pred`: rows scaled by its selectivity; with
/// `propagate`, the columns it reads narrowed and every other column keeping
/// the distinct values the kept rows are expected to hold ([`kept_distinct`]).
fn filtered(est: &RelStats, pred: &Expr, propagate: bool) -> RelStats {
    let mut v = View::new(&est.cols, propagate);
    let sel = apply(pred, &mut v);
    let rows = kept_rows(est.rows, sel, v.empty);
    if !propagate {
        return est.clone().with_rows(rows.max(1.0));
    }
    let read = references(pred);
    let kept = if est.rows > 0.0 { rows / est.rows } else { 1.0 };
    let cols = est
        .cols
        .iter()
        .map(|(id, base)| {
            let mut c = v.over.get(id).unwrap_or(base).clone();
            c.ndv = if read.contains(id) {
                cap(c.ndv, rows)
            } else {
                cap(kept_distinct(c.ndv, kept, est.rows), rows)
            };
            (*id, c)
        })
        .collect();
    RelStats {
        rows,
        cols,
        from_statistics: est.from_statistics,
        unfiltered_rows: est.base_rows(),
    }
}

/// The pairs `(left column, right column)` of the equalities between a
/// column of `l` and a column of `r` among the conjuncts of `pred`.
fn key_pairs(pred: &Arc<Expr>, l: &RelStats, r: &RelStats) -> Vec<(AttrId, AttrId)> {
    let column = |e: &Expr| match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some(c.id),
        _ => None,
    };
    conjuncts(pred)
        .iter()
        .filter_map(|c| {
            let (Builtin::Eq | Builtin::EqNullSafe, [a, b], _) = builtin(c)? else {
                return None;
            };
            let (a, b) = (column(a)?, column(b)?);
            if l.cols.contains_key(&a) && r.cols.contains_key(&b) {
                Some((a, b))
            } else if l.cols.contains_key(&b) && r.cols.contains_key(&a) {
                Some((b, a))
            } else {
                None
            }
        })
        .collect()
}

/// The factor that turns the product of independent key selectivities
/// (`prod 1 / max(ndv)` over the equalities, what `narrow::apply` gives) into
/// the composite-key selectivity: several equalities between the same two
/// sides are a foreign key into the side with fewer distinct key tuples, so
/// the selectivity is `1 / min(prod ndv left, prod ndv right)` (each product
/// over the distinct key columns, at most the side's rows before its filters: a filtered key
/// side keeps its fraction of the join), the rule the
/// join orderer's `composite_keys` applies, except that the cap is the side's rows before its
/// filters (the orderer caps by the leaf's filtered rows). Never below 1: the independent
/// product is already the smaller estimate then.
fn composite_factor(pairs: &[(AttrId, AttrId)], l: &RelStats, r: &RelStats) -> f64 {
    let (mut lc, mut rc): (Vec<AttrId>, Vec<AttrId>) = pairs.iter().copied().unzip();
    lc.sort_unstable();
    lc.dedup();
    rc.sort_unstable();
    rc.dedup();
    // A column in two equalities narrows sequentially in `narrow::apply`; the product of original
    // distinct counts below would not match it, so such keys get no factor.
    if lc.len().min(rc.len()) < 2 || lc.len() != pairs.len() || rc.len() != pairs.len() {
        return 1.0;
    }
    // A key column without known distinct counts (a guess) makes the tuple count unknown: no
    // factor. A filtered side's key domain is that of the unfiltered side: each column's count
    // is taken back through the filter's shrink (`unfiltered_distinct`), so a filter on a key
    // keeps its reduction in the join's rows and one on another column leaves the domain alone.
    let tuples = |ids: &[AttrId], s: &RelStats| -> Option<f64> {
        let base = s.base_rows().max(1.0);
        let kept = s.rows.max(1.0) / base;
        let product = ids
            .iter()
            .map(|i| {
                s.cols
                    .get(i)
                    .filter(|c| c.ndv_known())
                    .map(|c| unfiltered_distinct(c.ndv.max(1.0), kept, base))
            })
            .product::<Option<f64>>()?;
        Some(product.min(base))
    };
    let (Some(lt), Some(rt)) = (tuples(&lc, l), tuples(&rc, r)) else {
        return 1.0;
    };
    let composite = 1.0 / lt.min(rt).max(1.0);
    let independent: f64 = pairs
        .iter()
        .filter_map(|(a, b)| Some(1.0 / l.cols.get(a)?.ndv.max(r.cols.get(b)?.ndv).max(1.0)))
        .product();
    if independent > 0.0 && composite > independent {
        composite / independent
    } else {
        1.0
    }
}

/// Estimates relations, memoised per relation.
pub(crate) struct Estimator<'a> {
    stats: &'a dyn Statistics,
    /// Whether filters, joins, aggregates, projections and unions carry
    /// column statistics (ranges, null fractions, distinct counts) through,
    /// or only scale row counts.
    propagate: bool,
    done: HashMap<usize, (Arc<LogicalPlan>, Arc<RelStats>)>,
    /// How many relations were estimated (not looked up).
    computed: usize,
    /// The `WITH` definitions in scope, by their output attributes.
    ctes: HashMap<Vec<AttrId>, Arc<LogicalPlan>>,
    /// The definitions being estimated: a reference to one reads as unknown.
    open: HashSet<Vec<AttrId>>,
}

impl<'a> Estimator<'a> {
    pub(crate) fn new(stats: &'a dyn Statistics) -> Self {
        Self::with_propagation(stats, true)
    }

    pub(crate) fn with_propagation(stats: &'a dyn Statistics, propagate: bool) -> Self {
        Self {
            stats,
            propagate,
            done: HashMap::new(),
            computed: 0,
            ctes: HashMap::new(),
            open: HashSet::new(),
        }
    }

    /// The number of relations estimated so far: each once, however often
    /// it is asked for.
    pub(crate) fn computed(&self) -> usize {
        self.computed
    }

    /// The estimate of `p`.
    pub(crate) fn plan(&mut self, p: &Arc<LogicalPlan>) -> Arc<RelStats> {
        if let Some((_, e)) = self.done.get(&Arc::as_ptr(p).addr()) {
            return Arc::clone(e);
        }
        let mut est = self.compute(p);
        est.from_statistics = match &p.node {
            PlanNode::Scan(_) | PlanNode::CteRef(_) => est.from_statistics,
            _ => {
                let mut all = true;
                p.for_each_child(&mut |c| all &= self.plan(c).from_statistics);
                all
            }
        };
        self.computed += 1;
        let est = Arc::new(est);
        self.done
            .insert(Arc::as_ptr(p).addr(), (Arc::clone(p), Arc::clone(&est)));
        est
    }

    fn compute(&mut self, p: &LogicalPlan) -> RelStats {
        let ids = || {
            p.resolved_output()
                .unwrap_or_default()
                .into_iter()
                .map(|c| c.id)
        };
        match &p.node {
            PlanNode::Scan(scan) => self.scan(scan),
            PlanNode::Filter { input, predicate } => {
                let est = self.plan(input);
                // Estimates see through common-subexpression projections:
                // the column holds its definition's values.
                if reads_cse(predicate) {
                    let defs = cse_definitions(input);
                    filtered(&est, &resolve_cse(predicate, &defs), self.propagate)
                } else {
                    let mut out = filtered(&est, &same_renames(predicate, input), self.propagate);
                    // Every rename of a column holds the same values: the
                    // narrowing the canonical one got is theirs too.
                    for (renamed, canon) in renames_to_canonical(input) {
                        if let Some(c) = out.cols.get(&canon.id).cloned() {
                            out.cols.insert(renamed, c);
                        }
                    }
                    out
                }
            }
            PlanNode::Project { input, exprs } => {
                let est = self.plan(input);
                outputs(&est, exprs, est.rows, self.propagate)
            }
            PlanNode::Aggregate {
                input,
                grouping,
                aggregates,
            } => {
                let est = self.plan(input);
                let rows = match grouping {
                    Grouping::GroupBy(keys) if keys.is_empty() => 1.0,
                    Grouping::GroupBy(keys) => keys
                        .iter()
                        .map(|k| column_ndv(k, &est).unwrap_or(est.rows))
                        .product::<f64>()
                        .min(est.rows),
                    _ => est.rows,
                };
                outputs(&est, aggregates, rows, self.propagate)
            }
            PlanNode::Join {
                left,
                right,
                kind,
                condition,
            } => {
                let (l, r) = (self.plan(left), self.plan(right));
                self.join(p, &l, &r, *kind, condition)
            }
            PlanNode::Sort { input, .. }
            | PlanNode::SubqueryAlias { input, .. }
            | PlanNode::JoinHint { input, .. }
            | PlanNode::Repartition { input, .. }
            // A window adds columns (without statistics), never rows.
            | PlanNode::Window { input, .. }
            // Each distinct row of an `INTERSECT ALL` / `EXCEPT ALL` count:
            // no count is known, the input's rows bound it from below.
            | PlanNode::ReplicateRows { input, .. } => RelStats::clone(&self.plan(input)),
            // Every input row once per projection.
            PlanNode::Expand {
                input,
                projections,
                output,
            } => {
                let est = RelStats::clone(&self.plan(input));
                expand(est, projections, output)
            }
            PlanNode::Union {
                inputs, by_name, ..
            } => {
                let ests: Vec<Arc<RelStats>> = inputs.iter().map(|i| self.plan(i)).collect();
                let rows = ests.iter().map(|e| e.rows).sum();
                if self.propagate && !by_name {
                    let positions: Vec<Vec<AttrId>> = inputs
                        .iter()
                        .map(|i| {
                            i.resolved_output()
                                .unwrap_or_default()
                                .into_iter()
                                .map(|c| c.id)
                                .collect()
                        })
                        .collect();
                    union(&ests, &positions, ids(), rows)
                } else {
                    RelStats::unknown(rows, ids())
                }
            }
            PlanNode::Limit { input, limit } => {
                let est = RelStats::clone(&self.plan(input));
                let n = limit_value(limit).map_or(est.rows, to_f64);
                let rows = est.rows.min(n);
                est.with_rows(rows)
            }
            PlanNode::Offset { input, offset } => {
                let est = RelStats::clone(&self.plan(input));
                let skipped = limit_value(offset).map_or(0.0, to_f64);
                let rows = (est.rows - skipped).max(0.0);
                est.with_rows(rows)
            }
            PlanNode::WithCte { input, ctes } => {
                for cte in ctes {
                    let source = cte
                        .plan
                        .resolved_output()
                        .unwrap_or_default()
                        .iter()
                        .map(|c| c.id)
                        .collect();
                    self.ctes.insert(source, Arc::clone(&cte.plan));
                }
                RelStats::clone(&self.plan(input))
            }
            // A reference holds what its definition holds, under its own
            // attributes, as a `Shift` instance of the same relation does.
            PlanNode::CteRef(r) => {
                let Some(def) = self.ctes.get(&r.source).cloned() else {
                    return RelStats::unknown(DEFAULT_ROWS, ids());
                };
                if !self.open.insert(r.source.clone()) {
                    return RelStats::unknown(DEFAULT_ROWS, ids());
                }
                let est = self.plan(&def);
                self.open.remove(&r.source);
                RelStats {
                    rows: est.rows,
                    cols: r
                        .source
                        .iter()
                        .zip(&r.output)
                        // A column the definition's estimate does not hold
                        // (a window's output) stays unknown, as under a
                        // `Shift`: `ShareCte` turning one into the other
                        // must not change the estimate.
                        .filter_map(|(s, o)| est.cols.get(s).map(|c| (o.id, c.clone())))
                        .collect(),
                    from_statistics: est.from_statistics,
                    unfiltered_rows: est.unfiltered_rows,
                }
            }
            PlanNode::Shift { input, offset } => {
                let est = self.plan(input);
                RelStats {
                    rows: est.rows,
                    cols: est
                        .cols
                        .iter()
                        .map(|(id, c)| (AttrId(id.0.saturating_add(*offset)), c.clone()))
                        .collect(),
                    from_statistics: est.from_statistics,
                    unfiltered_rows: est.unfiltered_rows,
                }
            }
            PlanNode::PartialAggregate {
                input,
                keys,
                states,
                ..
            } => {
                let est = self.plan(input);
                let rows = keys
                    .iter()
                    .map(|k| column_ndv(k, &est).unwrap_or(est.rows))
                    .product::<f64>()
                    .min(est.rows);
                let mut out = outputs(&est, keys, rows, self.propagate);
                out.cols
                    .extend(states.iter().map(|c| (c.id, AttrStats::guess(rows))));
                out
            }
            _ => RelStats::unknown(DEFAULT_ROWS, ids()),
        }
    }

    /// A join of `l` and `r`: rows from the condition's selectivity and the
    /// join kind; with propagation, key columns narrowed by their equality
    /// (an inner join's keys also lose their nulls), the null-supplying
    /// side of an outer join gaining the unmatched fraction as nulls, and
    /// every other column keeping the distinct values the fraction of its
    /// side that finds a match is expected to hold ([`kept_distinct`]).
    fn join(
        &self,
        p: &LogicalPlan,
        l: &RelStats,
        r: &RelStats,
        kind: JoinKind,
        condition: &JoinCondition,
    ) -> RelStats {
        let mut cols = l.cols.clone();
        cols.extend(r.cols.iter().map(|(k, v)| (*k, v.clone())));
        let cross = l.rows * r.rows;
        let pred = match condition {
            JoinCondition::On(c) if reads_cse(c) => Some(resolve_cse(c, &cse_definitions(p))),
            JoinCondition::On(c) => Some(Arc::clone(c)),
            _ => None,
        };
        let mut v = View::new(&cols, self.propagate);
        let composite = pred
            .as_ref()
            // Inner joins only: an outer or semi join's row rule (`max(l.rows)`, `min(inner,
            // l.rows)`) and its unmatched fraction would read the larger inner estimate (TPC-DS
            // q78's anti join collapsed to one row).
            .filter(|_| self.propagate && matches!(kind, JoinKind::Inner))
            .map_or(1.0, |c| composite_factor(&key_pairs(c, l, r), l, r));
        let sel = pred.as_ref().map_or(1.0, |c| apply(c, &mut v)) * composite;
        // `inf * 0` is NaN: a zero selectivity is zero rows.
        let inner = if v.empty || sel <= 0.0 {
            0.0
        } else {
            cross * sel
        };
        let rows = match kind {
            JoinKind::LeftOuter | JoinKind::Single => inner.max(l.rows),
            JoinKind::RightOuter => inner.max(r.rows),
            JoinKind::FullOuter => inner.max(l.rows).max(r.rows),
            JoinKind::LeftSemi => inner.min(l.rows),
            JoinKind::LeftAnti => (l.rows - inner.min(l.rows)).max(1.0),
            // Every left row once, with `exists` set or not.
            JoinKind::Existence(_) => l.rows,
            _ => inner,
        };
        let proven_empty = self.propagate && (v.empty || cross <= 0.0);
        let rows = if proven_empty && rows < 1.0 {
            rows
        } else {
            rows.max(1.0)
        };
        if !self.propagate {
            if matches!(
                kind,
                JoinKind::LeftSemi | JoinKind::LeftAnti | JoinKind::Existence(_)
            ) {
                cols.clone_from(&l.cols);
            }
            return RelStats {
                rows,
                cols,
                from_statistics: false,
                unfiltered_rows: 0.0,
            }
            .with_rows(rows);
        }
        let read = pred.as_ref().map(|c| references(c)).unwrap_or_default();
        let pairs = pred
            .as_ref()
            .map(|c| key_pairs(c, l, r))
            .unwrap_or_default();
        // The fraction of a side's rows with a match: the keys' surviving
        // distinct values over its own, else the output over its rows.
        let matched = |side: &RelStats, other: &RelStats, left: bool| -> f64 {
            pairs
                .iter()
                .filter_map(|&(a, b)| {
                    let (mine, theirs) = if left { (a, b) } else { (b, a) };
                    let (m, o) = (side.cols.get(&mine)?, other.cols.get(&theirs)?);
                    (m.ndv_known() && o.ndv_known()).then(|| (m.ndv.min(o.ndv) / m.ndv).min(1.0))
                })
                .reduce(f64::min)
                .unwrap_or_else(|| {
                    if side.rows > 0.0 {
                        (inner / side.rows).min(1.0)
                    } else {
                        1.0
                    }
                })
        };
        // A side's columns: narrowed by the condition (or as they were),
        // with `nulls` of the output rows supplied as NULL.
        let side = |s: &RelStats, matched: f64, narrowed: bool, nulls: f64| {
            let nulls = if rows > 0.0 {
                (nulls / rows).clamp(0.0, 1.0)
            } else {
                0.0
            };
            s.cols
                .iter()
                .map(|(id, base)| {
                    let mut c = if narrowed {
                        v.over.get(id).unwrap_or(base).clone()
                    } else {
                        base.clone()
                    };
                    c.ndv = if narrowed && !read.contains(id) {
                        cap(kept_distinct(c.ndv, matched, s.rows), rows)
                    } else {
                        cap(c.ndv, rows)
                    };
                    c.null_frac = nulls + (1.0 - nulls) * c.null_frac;
                    (*id, c)
                })
                .collect::<Vec<_>>()
        };
        let (left_only, right_only) = ((l.rows - inner).max(0.0), (r.rows - inner).max(0.0));
        let (ml, mr) = (matched(l, r, true), matched(r, l, false));
        let cols: HashMap<AttrId, AttrStats> = match kind {
            JoinKind::LeftOuter | JoinKind::Single => side(l, ml, false, 0.0)
                .into_iter()
                .chain(side(r, mr, true, left_only))
                .collect(),
            JoinKind::RightOuter => side(l, ml, true, right_only)
                .into_iter()
                .chain(side(r, mr, false, 0.0))
                .collect(),
            JoinKind::FullOuter => side(l, ml, false, right_only)
                .into_iter()
                .chain(side(r, mr, false, left_only))
                .collect(),
            JoinKind::LeftSemi => side(l, ml, true, 0.0).into_iter().collect(),
            // An existence join keeps the left rows unchanged (`exists` has
            // no statistics).
            JoinKind::LeftAnti | JoinKind::Existence(_) => {
                side(l, ml, false, 0.0).into_iter().collect()
            }
            _ => side(l, ml, true, 0.0)
                .into_iter()
                .chain(side(r, mr, true, 0.0))
                .collect(),
        };
        RelStats {
            rows,
            cols,
            from_statistics: false,
            unfiltered_rows: rows
                * (l.base_rows() / l.rows.max(1.0))
                * (r.base_rows() / r.rows.max(1.0)),
        }
    }

    fn scan(&self, scan: &Scan) -> RelStats {
        let ids = scan.output.iter().map(|c| c.id);
        let Some(stats) = self.stats.table(scan) else {
            return RelStats::unknown(DEFAULT_ROWS, ids);
        };
        let rows = to_f64(stats.rows).max(1.0);
        let cols = scan
            .output
            .iter()
            .map(|c| {
                let s = stats.columns.get(&c.name).cloned().unwrap_or_default();
                let col = AttrStats {
                    ndv: s.distinct.map_or(rows, to_f64).clamp(1.0, rows),
                    null_frac: s.nulls.map_or(0.0, |n| (to_f64(n) / rows).min(1.0)),
                    min: s.min.as_ref().and_then(number),
                    max: s.max.as_ref().and_then(number),
                    source: if s.distinct.is_some() {
                        s.source
                    } else {
                        StatSource::Guess
                    },
                };
                (c.id, col)
            })
            .collect();
        RelStats {
            rows,
            cols,
            from_statistics: true,
            unfiltered_rows: 0.0,
        }
    }
}

/// The estimate of a union of `ests`: rows add, a column's distinct values
/// are the most of any input (branches usually share a key domain: the store
/// key of sales and of returns; at most the rows), its bounds are the
/// widest, its nulls the row-weighted mean. `positions` are the input columns' attributes, `ids`
/// the output's.
fn union(
    ests: &[Arc<RelStats>],
    positions: &[Vec<AttrId>],
    ids: impl IntoIterator<Item = AttrId>,
    rows: f64,
) -> RelStats {
    let cols = ids
        .into_iter()
        .enumerate()
        .map(|(j, out)| {
            let mut c = AttrStats {
                ndv: 0.0,
                null_frac: 0.0,
                min: Some(f64::INFINITY),
                max: Some(f64::NEG_INFINITY),
                source: StatSource::Provided,
            };
            let mut nulls = 0.0;
            for (est, ids) in ests.iter().zip(positions) {
                match ids.get(j).and_then(|id| est.cols.get(id)) {
                    Some(a) => {
                        c.ndv = c.ndv.max(a.ndv);
                        nulls += a.null_frac * est.rows;
                        c.min = c.min.zip(a.min).map(|(x, y)| x.min(y));
                        c.max = c.max.zip(a.max).map(|(x, y)| x.max(y));
                        c.source = c.source.min(a.source);
                    }
                    None => {
                        c.ndv = c.ndv.max(est.rows);
                        c.min = None;
                        c.max = None;
                        c.source = StatSource::Guess;
                    }
                }
            }
            c.ndv = cap(c.ndv, rows);
            c.null_frac = if rows > 0.0 {
                (nulls / rows).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (out, c)
        })
        .collect();
    RelStats {
        rows,
        cols,
        from_statistics: false,
        unfiltered_rows: 0.0,
    }
}

/// Every row of `est` once per projection, over the columns `output`.
fn expand(est: RelStats, projections: &[Vec<Arc<Expr>>], output: &[BoundColumn]) -> RelStats {
    let rows = est.rows * to_f64(u64::try_from(projections.len()).unwrap_or(u64::MAX));
    let rows = rows.max(1.0);
    if projections.is_empty()
        || output.is_empty()
        || projections.iter().any(|p| p.len() != output.len())
    {
        return est.with_rows(rows);
    }
    let per = est.rows;
    let mut cols = HashMap::with_capacity(output.len());
    for (j, out) in output.iter().enumerate() {
        let mut null_rows = 0.0;
        let mut col_id: Option<AttrId> = None;
        let mut mixed = false;
        let mut literals: Vec<&Literal> = Vec::new();
        let mut other = false;
        // Whether every input column's count comes from statistics
        // (literals are counted exactly), and the weakest source.
        let mut known = true;
        let mut source = StatSource::Provided;
        for proj in projections {
            let s = strip_alias(&proj[j]);
            match &s.node {
                ExprNode::Literal(Literal::Null(_)) => null_rows += per,
                ExprNode::Literal(l) => {
                    if col_id.is_some() {
                        mixed = true;
                    }
                    if !literals.contains(&l) {
                        literals.push(l);
                    }
                }
                ExprNode::Resolved(ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c)) => {
                    match est.cols.get(&c.id) {
                        Some(col) => {
                            null_rows += col.null_frac * per;
                            known &= col.ndv_known();
                            source = source.min(col.source);
                            match col_id {
                                None => col_id = Some(c.id),
                                Some(id) if id == c.id => {}
                                Some(_) => mixed = true,
                            }
                            if !literals.is_empty() {
                                mixed = true;
                            }
                        }
                        None => other = true,
                    }
                }
                _ => other = true,
            }
        }
        let nulls = (null_rows / rows).clamp(0.0, 1.0);
        let distinct_literals = to_f64(u64::try_from(literals.len()).unwrap_or(u64::MAX));
        let ndv = if other {
            rows
        } else if let Some(id) = col_id {
            let of = |id: &AttrId| est.cols.get(id).map_or(per, |c| c.ndv);
            let base = if mixed {
                // Different input columns (or columns plus literals): at
                // most their sum.
                let mut seen = Vec::new();
                let mut sum = 0.0;
                for proj in projections {
                    let s = strip_alias(&proj[j]);
                    if let ExprNode::Resolved(
                        ResolvedExpr::Column(c) | ResolvedExpr::OuterColumn(c),
                    ) = &s.node
                        && !seen.contains(&c.id)
                    {
                        seen.push(c.id);
                        sum += of(&c.id);
                    }
                }
                sum + distinct_literals
            } else if literals.is_empty() {
                of(&id)
            } else {
                of(&id) + distinct_literals
            };
            cap(base, rows)
        } else if literals.is_empty() {
            // Every projection is NULL.
            1.0
        } else {
            cap(distinct_literals, rows)
        };
        let bounds = if !mixed && !other && literals.is_empty() {
            col_id.and_then(|id| est.cols.get(&id))
        } else {
            None
        };
        cols.insert(
            out.id,
            AttrStats {
                ndv: cap(ndv, rows),
                null_frac: nulls,
                min: bounds.and_then(|c| c.min),
                max: bounds.and_then(|c| c.max),
                source: if known && !other {
                    source
                } else {
                    StatSource::Guess
                },
            },
        );
    }
    RelStats {
        rows,
        cols,
        from_statistics: est.from_statistics,
        unfiltered_rows: 0.0,
    }
}

/// The distinct count of the column `e` reads directly.
fn column_ndv(e: &Expr, est: &RelStats) -> Option<f64> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => est.cols.get(&c.id).map(|c| c.ndv),
        _ => None,
    }
}

/// The distinct values expected among a fraction `kept` of `rows` rows that
/// hold `ndv` equally frequent values: `ndv * (1 - (1 - kept)^(rows / ndv))`
/// (Cardenas 1975). Each value has `rows / ndv` rows (at least one), and is
/// lost only when every one of them is: a column of few values over many rows
/// keeps them all, a unique column keeps the fraction `kept` of them.
fn kept_distinct(ndv: f64, kept: f64, rows: f64) -> f64 {
    // NaN inputs keep the count as it was.
    if ndv.is_nan() || kept.is_nan() || ndv <= 0.0 || kept >= 1.0 {
        return ndv;
    }
    if kept <= 0.0 {
        return 0.0;
    }
    let per_value = (rows / ndv).max(1.0);
    ndv * (1.0 - (1.0 - kept).powf(per_value))
}

/// The distinct count a column held before a filter that kept the fraction `kept` of `rows` rows
/// left it with `ndv`: the inverse of [`kept_distinct`] (monotone in the count, found by
/// bisection between `ndv` and `rows`). A column of few values over many rows comes back as it
/// is; a unique one comes back at `ndv / kept`.
fn unfiltered_distinct(ndv: f64, kept: f64, rows: f64) -> f64 {
    if !(kept > 0.0 && kept < 1.0) || ndv >= rows {
        return ndv;
    }
    let (mut lo, mut hi) = (ndv, rows);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if kept_distinct(mid, kept, rows) < ndv {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

/// `predicate` over the projection `input` with every output that renames
/// an input column read as the first output renaming that column: over
/// `Project [a AS c0, a AS c1]`, `c0 = c1` is `c0 = c0`, which keeps every
/// non-NULL row (`narrow`), as `a = a` does below the projection once
/// pushdown moves it there. Estimated as two columns, the conjunct join
/// ordering derives for the equal pair made its input look one row large on
/// the first optimization and a million on the next.
pub(crate) fn same_renames(predicate: &Arc<Expr>, input: &LogicalPlan) -> Arc<Expr> {
    let to = renames_to_canonical(input);
    if to.is_empty() {
        return Arc::clone(predicate);
    }
    let out: Result<Arc<Expr>, std::convert::Infallible> =
        Arc::clone(predicate).transform_up(&mut |n| {
            Ok(match &n.node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) => match to.get(&c.id) {
                    Some(canon) => Arc::new(
                        Expr::new(ExprNode::Resolved(ResolvedExpr::Column(canon.clone())))
                            .with_origin(n.origin),
                    ),
                    None => n,
                },
                _ => n,
            })
        });
    match out {
        Ok(p) => p,
        Err(never) => match never {},
    }
}

/// The outputs of the projection under `input` (through filters) that
/// rename an input column another output already names, by attribute: that
/// first output (the canonical name of the column).
fn renames_to_canonical(input: &LogicalPlan) -> HashMap<AttrId, BoundColumn> {
    // Filters and subquery aliases keep their input's columns (attributes):
    // the projection may sit below them.
    let mut input = input;
    while let PlanNode::Filter { input: below, .. } | PlanNode::SubqueryAlias { input: below, .. } =
        &input.node
    {
        input = below;
    }
    let PlanNode::Project { exprs, .. } = &input.node else {
        return HashMap::new();
    };
    let mut first: HashMap<AttrId, BoundColumn> = HashMap::new();
    let mut to: HashMap<AttrId, BoundColumn> = HashMap::new();
    for e in exprs {
        // `a AS c` renames `a`; a bare `a` passes it through under its own
        // name (`Project [a, a AS c0]`: `a = c0` is `a = a`).
        let (source, output) = match &e.node {
            ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) => match &expr.node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) => (c.id, output),
                _ => continue,
            },
            ExprNode::Resolved(ResolvedExpr::Column(c)) => (c.id, c),
            _ => continue,
        };
        let canon = first.entry(source).or_insert_with(|| output.clone());
        if canon.id != output.id {
            to.insert(output.id, canon.clone());
        }
    }
    to
}
