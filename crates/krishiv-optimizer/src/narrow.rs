// SPDX-License-Identifier: Apache-2.0
//! Predicates over estimates: a predicate's selectivity and what it leaves
//! known about the columns it reads.
//!
//! [`apply`] walks a predicate once. A conjunction applies its conjuncts in
//! order, so a later one reads the range an earlier one narrowed; a
//! disjunction applies each branch to the same columns and keeps the union
//! of the ranges (the widest bounds, the most nulls); `NOT` and anything
//! unknown leave the columns as they were. Statistics only ever stay or
//! widen from what the rows that survive hold: a range is never narrowed
//! past what a predicate proves. A predicate that proves the relation empty
//! (an equality with a literal outside `[min, max]`, a constant `false`) sets
//! [`View::empty`]: the estimate is zero rows, and nothing is removed from
//! the plan.

use std::collections::HashSet;

use krishiv_plan::{AttrId, Builtin, Expr, ExprNode, Literal, ResolvedExpr};

use crate::expr::builtin;
use crate::stats::{AttrStats, DEFAULT_SELECTIVITY, RelStats, number, to_f64};

/// The columns of a relation as a predicate sees them: the base estimates
/// under the columns the predicate narrowed.
#[derive(Clone)]
pub(crate) struct View<'a> {
    base: &'a std::collections::HashMap<AttrId, AttrStats>,
    /// The columns the predicate changed, by attribute.
    pub(crate) over: std::collections::HashMap<AttrId, AttrStats>,
    /// The predicate proves the relation empty.
    pub(crate) empty: bool,
    /// Whether to record what the predicate leaves known (off: only the
    /// selectivity is read, as before statistics propagated).
    narrow: bool,
}

impl<'a> View<'a> {
    pub(crate) fn new(
        base: &'a std::collections::HashMap<AttrId, AttrStats>,
        narrow: bool,
    ) -> Self {
        Self {
            base,
            over: std::collections::HashMap::new(),
            empty: false,
            narrow,
        }
    }

    /// The column `id` after the predicate so far.
    pub(crate) fn get(&self, id: AttrId) -> Option<&AttrStats> {
        self.over.get(&id).or_else(|| self.base.get(&id))
    }

    /// The column `e` reads directly.
    fn column(&self, e: &Expr) -> Option<(AttrId, &AttrStats)> {
        match &e.node {
            ExprNode::Resolved(ResolvedExpr::Column(c)) => Some((c.id, self.get(c.id)?)),
            _ => None,
        }
    }

    fn update(&mut self, id: AttrId, f: impl FnOnce(&mut AttrStats)) {
        if !self.narrow {
            return;
        }
        if let Some(mut c) = self.get(id).cloned() {
            f(&mut c);
            self.over.insert(id, c);
        }
    }
}

fn literal(e: &Expr) -> Option<&Literal> {
    match &e.node {
        ExprNode::Literal(l) => Some(l),
        _ => None,
    }
}

fn is_null_literal(e: &Expr) -> bool {
    matches!(literal(e), Some(Literal::Null(_)))
}

/// The fraction of `[lo, hi]` below `x`.
fn below(lo: f64, hi: f64, x: f64) -> f64 {
    if hi > lo {
        ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
    } else if x > lo {
        1.0
    } else if x < lo {
        0.0
    } else {
        0.5
    }
}

fn max_of(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    }
}

fn min_of(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// `c` with its bounds cut to `[lo, hi]` (either side may be open). A cut
/// that would leave no values (`lo > hi`) is not made: the bounds stay.
fn cut(c: &mut AttrStats, lo: Option<f64>, hi: Option<f64>) {
    let (min, max) = (max_of(c.min, lo), min_of(c.max, hi));
    if let (Some(a), Some(b)) = (min, max)
        && a > b
    {
        return;
    }
    c.min = min;
    c.max = max;
}

/// `c` after a predicate that keeps the fraction `sel` of its non-null
/// values, none of them null: its distinct values scale by `sel`, at least 1.
fn keep(c: &mut AttrStats, sel: f64) {
    c.null_frac = 0.0;
    c.ndv = (c.ndv * sel).clamp(1.0, c.ndv.max(1.0));
}

/// The fraction of the rows of `cols` that satisfy `pred`; the columns
/// `pred` narrows are recorded in `v`.
pub(crate) fn apply(pred: &Expr, v: &mut View<'_>) -> f64 {
    let s = match (builtin(pred), literal(pred)) {
        (_, Some(Literal::Boolean(true))) => 1.0,
        (_, Some(_)) => {
            v.empty |= v.narrow;
            0.0
        }
        (Some((Builtin::And, [a, b], _)), _) => {
            let x = apply(a, v);
            x * apply(b, v)
        }
        (Some((Builtin::Or, [a, b], _)), _) => or(a, b, v),
        (Some((Builtin::Not, [a], _)), _) => 1.0 - apply(a, &mut v.clone()),
        (Some((op, args, _)), _) => leaf(op, args, v),
        _ => DEFAULT_SELECTIVITY,
    };
    if s.is_finite() {
        s.clamp(0.0, 1.0)
    } else {
        DEFAULT_SELECTIVITY
    }
}

/// `a OR b`: each branch over the same columns; what is known afterwards
/// holds for either branch's rows.
fn or(a: &Expr, b: &Expr, v: &mut View<'_>) -> f64 {
    let (mut x, mut y) = (v.clone(), v.clone());
    let (sx, sy) = (apply(a, &mut x), apply(b, &mut y));
    if v.narrow {
        v.empty |= x.empty && y.empty;
        let ids: HashSet<AttrId> = x.over.keys().chain(y.over.keys()).copied().collect();
        for id in ids {
            let (Some(p), Some(q)) = (x.get(id), y.get(id)) else {
                continue;
            };
            let before = v.get(id).map_or(f64::INFINITY, |c| c.ndv);
            let merged = AttrStats {
                ndv: (p.ndv + q.ndv).min(before).max(1.0),
                null_frac: p.null_frac.max(q.null_frac),
                min: p.min.zip(q.min).map(|(m, n)| m.min(n)),
                max: p.max.zip(q.max).map(|(m, n)| m.max(n)),
                source: p.source.min(q.source),
            };
            v.over.insert(id, merged);
        }
    }
    sx + sy - sx * sy
}

fn leaf(op: Builtin, args: &[std::sync::Arc<Expr>], v: &mut View<'_>) -> f64 {
    match (op, args) {
        // `c <=> NULL` holds exactly where `c` is NULL, as `c IS NULL`.
        (Builtin::EqNullSafe, [a, b]) if is_null_literal(a) || is_null_literal(b) => {
            let c = if is_null_literal(a) { b } else { a };
            v.column(c)
                .map_or(DEFAULT_SELECTIVITY, |(_, c)| c.null_frac)
        }
        (Builtin::Eq | Builtin::EqNullSafe, [a, b]) => equality(op, a, b, v),
        (Builtin::Lt | Builtin::LtEq | Builtin::Gt | Builtin::GtEq, [a, b]) => compare(op, a, b, v),
        (Builtin::Between, [x, lo, hi]) => between(x, lo, hi, v),
        (Builtin::In, [x, list @ ..]) => in_list(x, list, v),
        (Builtin::IsNull, [a]) => v
            .column(a)
            .map_or(DEFAULT_SELECTIVITY, |(_, c)| c.null_frac),
        (Builtin::IsNotNull, [a]) => match v.column(a).map(|(id, c)| (id, c.null_frac)) {
            Some((id, nulls)) => {
                v.update(id, |c| c.null_frac = 0.0);
                1.0 - nulls
            }
            None => 1.0 - DEFAULT_SELECTIVITY,
        },
        _ => DEFAULT_SELECTIVITY,
    }
}

/// `a = b` or `a <=> b`.
fn equality(op: Builtin, a: &Expr, b: &Expr, v: &mut View<'_>) -> f64 {
    let (x, y) = (
        v.column(a).map(|(i, c)| (i, c.clone())),
        v.column(b).map(|(i, c)| (i, c.clone())),
    );
    match (x, y) {
        (Some((i, ci)), Some((j, cj))) => {
            // `1 / max(NDV)` over the distinct counts that come from
            // statistics; a count that is only the default guess is not one.
            let known = [&ci, &cj]
                .into_iter()
                .filter(|c| c.ndv_known())
                .map(|c| c.ndv)
                .reduce(f64::max);
            let sel = known.map_or(DEFAULT_SELECTIVITY, |n| 1.0 / n);
            let (lo, hi) = (max_of(ci.min, cj.min), min_of(ci.max, cj.max));
            let (ndv, source) = (ci.ndv.min(cj.ndv), ci.source.min(cj.source));
            for id in [i, j] {
                v.update(id, |c| {
                    c.ndv = ndv;
                    c.source = source;
                    cut(c, lo, hi);
                    if op == Builtin::Eq {
                        c.null_frac = 0.0;
                    }
                });
            }
            sel
        }
        (Some((id, c)), None) | (None, Some((id, c))) => {
            let other = if v.column(a).is_some() { b } else { a };
            match literal(other).filter(|l| !matches!(l, Literal::Null(_))) {
                None if literal(other).is_some() => DEFAULT_SELECTIVITY,
                None => {
                    if c.ndv_known() {
                        1.0 / c.ndv
                    } else {
                        DEFAULT_SELECTIVITY
                    }
                }
                Some(l) => {
                    let x = number(l);
                    let outside = x.is_some_and(|x| {
                        c.min.is_some_and(|m| x < m) || c.max.is_some_and(|m| x > m)
                    });
                    if outside {
                        v.empty |= v.narrow;
                    }
                    v.update(id, |c| {
                        c.ndv = 1.0;
                        c.null_frac = 0.0;
                        if let Some(x) = x {
                            c.min = Some(x);
                            c.max = Some(x);
                        }
                    });
                    if outside {
                        0.0
                    } else if c.ndv_known() {
                        1.0 / c.ndv
                    } else {
                        DEFAULT_SELECTIVITY
                    }
                }
            }
        }
        (None, None) => DEFAULT_SELECTIVITY,
    }
}

fn flip(op: Builtin) -> Builtin {
    match op {
        Builtin::Lt => Builtin::Gt,
        Builtin::LtEq => Builtin::GtEq,
        Builtin::Gt => Builtin::Lt,
        Builtin::GtEq => Builtin::LtEq,
        other => other,
    }
}

/// `a < b`, `a <= b`, `a > b`, `a >= b`.
fn compare(op: Builtin, a: &Expr, b: &Expr, v: &mut View<'_>) -> f64 {
    let (col, lit, op) = if v.column(a).is_some() && literal(b).is_some() {
        (a, b, op)
    } else if v.column(b).is_some() && literal(a).is_some() {
        (b, a, flip(op))
    } else {
        // Two columns (or something else): every comparison rejects NULL.
        for e in [a, b] {
            if let Some((id, _)) = v.column(e) {
                v.update(id, |c| c.null_frac = 0.0);
            }
        }
        return 1.0 / 3.0;
    };
    let Some((id, c)) = v.column(col).map(|(i, c)| (i, c.clone())) else {
        return 1.0 / 3.0;
    };
    let x = literal(lit).and_then(number);
    let upper = matches!(op, Builtin::Lt | Builtin::LtEq);
    let sel = match (x, c.min.zip(c.max)) {
        (Some(x), Some((lo, hi))) => {
            let f = below(lo, hi, x);
            if upper { f } else { 1.0 - f }
        }
        _ => 1.0 / 3.0,
    };
    v.update(id, |c| {
        if let Some(x) = x {
            if upper {
                cut(c, None, Some(x));
            } else {
                cut(c, Some(x), None);
            }
        }
        keep(c, sel);
    });
    sel
}

/// `x BETWEEN lo AND hi`.
fn between(x: &Expr, lo: &Expr, hi: &Expr, v: &mut View<'_>) -> f64 {
    let Some((id, c)) = v.column(x).map(|(i, c)| (i, c.clone())) else {
        return DEFAULT_SELECTIVITY;
    };
    let (l, h) = (literal(lo).and_then(number), literal(hi).and_then(number));
    let sel = match (c.min.zip(c.max), l, h) {
        (Some((a, b)), Some(l), Some(h)) => below(a, b, h) - below(a, b, l),
        _ => DEFAULT_SELECTIVITY,
    };
    let sel = sel.clamp(0.0, 1.0);
    v.update(id, |c| {
        cut(c, l, h);
        keep(c, sel);
    });
    sel
}

/// `x IN (list)`.
fn in_list(x: &Expr, list: &[std::sync::Arc<Expr>], v: &mut View<'_>) -> f64 {
    let Some((id, c)) = v.column(x).map(|(i, c)| (i, c.clone())) else {
        return DEFAULT_SELECTIVITY;
    };
    let k = to_f64(u64::try_from(list.len()).unwrap_or(u64::MAX));
    let sel = if c.ndv_known() {
        k / c.ndv
    } else {
        DEFAULT_SELECTIVITY
    };
    let numbers: Option<Vec<f64>> = list.iter().map(|e| literal(e).and_then(number)).collect();
    v.update(id, |c| {
        c.null_frac = 0.0;
        c.ndv = c.ndv.min(k).max(1.0);
        if let Some(n) = numbers {
            cut(
                c,
                n.iter().copied().reduce(f64::min),
                n.iter().copied().reduce(f64::max),
            );
        }
    });
    sel
}

/// The fraction of the rows of `est` that satisfy `pred`.
pub(crate) fn selectivity(pred: &Expr, est: &RelStats) -> f64 {
    apply(pred, &mut View::new(&est.cols, true))
}

/// The two columns of an equality between columns, with their distinct
/// counts.
pub(crate) fn equi_columns(pred: &Expr, est: &RelStats) -> Option<[(AttrId, f64); 2]> {
    let Some((Builtin::Eq | Builtin::EqNullSafe, [a, b], _)) = builtin(pred) else {
        return None;
    };
    let side = |e: &Expr| match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some((c.id, est.cols.get(&c.id)?.ndv)),
        _ => None,
    };
    Some([side(a)?, side(b)?])
}
