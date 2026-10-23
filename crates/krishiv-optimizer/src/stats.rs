// SPDX-License-Identifier: Apache-2.0
//! Table statistics and cardinality estimates.
//!
//! Estimates follow the textbook independence assumptions: a predicate's
//! selectivity is `1 / NDV` for an equality with a literal (when the
//! column's distinct count is known, else [`DEFAULT_SELECTIVITY`]), the covered
//! fraction of `[min, max]` for a range, `1 / max(NDV)` for an equi-join,
//! and [`DEFAULT_SELECTIVITY`] when nothing better is known; conjunctions
//! multiply and disjunctions add. Without statistics a table holds
//! [`DEFAULT_ROWS`] rows whose columns are all distinct.

use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;

use krishiv_plan::{
    AttrId, Builtin, Expr, ExprNode, Grouping, JoinCondition, JoinKind, Literal, LogicalPlan,
    PlanNode, ResolvedExpr, Scan,
};

use crate::expr::{builtin, output_of, strip_alias};
use crate::limit::limit_value;

/// The rows of a table without statistics.
pub const DEFAULT_ROWS: f64 = 1_000_000.0;

/// The selectivity of a predicate the estimator cannot read.
pub const DEFAULT_SELECTIVITY: f64 = 0.2;

/// Statistics of one column.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ColumnStats {
    /// The number of distinct non-null values.
    pub distinct: Option<u64>,
    /// The number of nulls.
    pub nulls: Option<u64>,
    /// The smallest value.
    pub min: Option<Literal>,
    /// The largest value.
    pub max: Option<Literal>,
}

/// Statistics of a table: its row count and, by column name, its columns'
/// statistics (from a Parquet footer, table-format metadata or `ANALYZE`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TableStats {
    /// The number of rows.
    pub rows: u64,
    /// Column statistics by column name.
    pub columns: HashMap<String, ColumnStats>,
}

/// Where the optimizer reads statistics from.
pub trait Statistics: Send + Sync + fmt::Debug {
    /// The statistics of the table `scan` reads at its pinned version, when
    /// known.
    fn table(&self, scan: &Scan) -> Option<TableStats>;
}

/// No statistics: every table is estimated by default.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoStatistics;

impl Statistics for NoStatistics {
    fn table(&self, _: &Scan) -> Option<TableStats> {
        None
    }
}

/// The estimated number of rows `plan` returns, as the join orderer sees it
/// (at least 1; [`DEFAULT_ROWS`] per table without statistics).
#[must_use]
pub fn estimated_rows(plan: &Arc<LogicalPlan>, stats: &dyn Statistics) -> f64 {
    Estimator::new(stats).plan(plan).rows
}

/// What is known of one attribute.
#[derive(Debug, Clone)]
pub(crate) struct Col {
    /// Distinct non-null values, at least 1. Without statistics, every
    /// value is taken as distinct (`ndv` = rows).
    pub(crate) ndv: f64,
    /// Whether `ndv` comes from statistics rather than that assumption.
    pub(crate) ndv_known: bool,
    /// The fraction of rows that are null.
    pub(crate) nulls: f64,
    /// Bounds, as numbers on one scale.
    pub(crate) range: Option<(f64, f64)>,
}

/// The estimated output of a relation.
#[derive(Debug, Clone, Default)]
pub(crate) struct Est {
    /// Rows, at least 1.
    pub(crate) rows: f64,
    /// Per output attribute.
    pub(crate) cols: HashMap<AttrId, Col>,
}

impl Est {
    fn unknown(rows: f64, ids: impl IntoIterator<Item = AttrId>) -> Self {
        let cols = ids.into_iter().map(|id| (id, distinct(rows))).collect();
        Self { rows, cols }
    }

    /// The same columns over `rows` rows: no column has more distinct values
    /// than rows.
    fn with_rows(mut self, rows: f64) -> Self {
        self.rows = rows.max(1.0);
        for c in self.cols.values_mut() {
            c.ndv = c.ndv.min(self.rows);
        }
        self
    }
}

fn distinct(rows: f64) -> Col {
    Col {
        ndv: rows.max(1.0),
        ndv_known: false,
        nulls: 0.0,
        range: None,
    }
}

/// A literal as a number on the scale its type compares on.
#[expect(clippy::cast_precision_loss, reason = "estimates only")]
fn number(l: &Literal) -> Option<f64> {
    let v = match l {
        Literal::Byte(v) => f64::from(*v),
        Literal::Short(v) => f64::from(*v),
        Literal::Integer(v) | Literal::Date(v) => f64::from(*v),
        Literal::Long(v) | Literal::Timestamp(v) | Literal::TimestampNtz(v) => *v as f64,
        Literal::Float(v) => f64::from(*v),
        Literal::Double(v) => *v,
        Literal::Decimal {
            unscaled,
            data_type,
        } => *unscaled as f64 / 10_f64.powi(i32::from(data_type.scale())),
        _ => return None,
    };
    v.is_finite().then_some(v)
}

#[expect(clippy::cast_precision_loss, reason = "estimates only")]
fn to_f64(v: u64) -> f64 {
    v as f64
}

/// Estimates relations, memoised per relation.
pub(crate) struct Estimator<'a> {
    stats: &'a dyn Statistics,
    done: HashMap<*const LogicalPlan, (Arc<LogicalPlan>, Rc<Est>)>,
}

impl<'a> Estimator<'a> {
    pub(crate) fn new(stats: &'a dyn Statistics) -> Self {
        Self {
            stats,
            done: HashMap::new(),
        }
    }

    /// The estimate of `p`.
    pub(crate) fn plan(&mut self, p: &Arc<LogicalPlan>) -> Rc<Est> {
        if let Some((_, e)) = self.done.get(&Arc::as_ptr(p)) {
            return Rc::clone(e);
        }
        let est = Rc::new(self.compute(p));
        self.done
            .insert(Arc::as_ptr(p), (Arc::clone(p), Rc::clone(&est)));
        est
    }

    fn compute(&mut self, p: &LogicalPlan) -> Est {
        let ids = || {
            p.resolved_output()
                .unwrap_or_default()
                .into_iter()
                .map(|c| c.id)
        };
        match &p.node {
            PlanNode::Scan(scan) => self.scan(scan),
            PlanNode::Filter { input, predicate } => {
                let est = Est::clone(&self.plan(input));
                let rows = est.rows * selectivity(predicate, &est);
                est.with_rows(rows)
            }
            PlanNode::Project { input, exprs } => {
                let est = self.plan(input);
                outputs(&est, exprs, est.rows)
            }
            PlanNode::Aggregate {
                input,
                grouping,
                aggregates,
            } => {
                let est = self.plan(input);
                let rows = match grouping {
                    Grouping::GroupBy(keys) => keys
                        .iter()
                        .map(|k| column_ndv(k, &est).unwrap_or(est.rows))
                        .product::<f64>()
                        .min(est.rows),
                    _ => est.rows,
                };
                outputs(&est, aggregates, rows)
            }
            PlanNode::Join {
                left,
                right,
                kind,
                condition,
            } => {
                let (l, r) = (self.plan(left), self.plan(right));
                let mut both = Est {
                    rows: l.rows * r.rows,
                    cols: l.cols.clone(),
                };
                both.cols
                    .extend(r.cols.iter().map(|(k, v)| (*k, v.clone())));
                let inner = match condition {
                    JoinCondition::On(c) => both.rows * selectivity(c, &both),
                    _ => both.rows,
                };
                let rows = match kind {
                    JoinKind::LeftOuter => inner.max(l.rows),
                    JoinKind::RightOuter => inner.max(r.rows),
                    JoinKind::FullOuter => inner.max(l.rows).max(r.rows),
                    JoinKind::LeftSemi => inner.min(l.rows),
                    JoinKind::LeftAnti => (l.rows - inner.min(l.rows)).max(1.0),
                    _ => inner,
                };
                if matches!(kind, JoinKind::LeftSemi | JoinKind::LeftAnti) {
                    both.cols = l.cols.clone();
                }
                both.with_rows(rows)
            }
            PlanNode::Sort { input, .. }
            | PlanNode::SubqueryAlias { input, .. }
            | PlanNode::WithCte { input, .. }
            | PlanNode::JoinHint { input, .. }
            // Each distinct row of an `INTERSECT ALL` / `EXCEPT ALL` count:
            // no count is known, the input's rows bound it from below.
            | PlanNode::ReplicateRows { input, .. } => Est::clone(&self.plan(input)),
            PlanNode::Union { inputs, .. } => {
                let rows = inputs.iter().map(|i| self.plan(i).rows).sum();
                Est::unknown(rows, ids())
            }
            PlanNode::Limit { input, limit } => {
                let est = Est::clone(&self.plan(input));
                let n = limit_value(limit).map_or(est.rows, to_f64);
                let rows = est.rows.min(n);
                est.with_rows(rows)
            }
            PlanNode::Offset { input, offset } => {
                let est = Est::clone(&self.plan(input));
                let skipped = limit_value(offset).map_or(0.0, to_f64);
                let rows = (est.rows - skipped).max(0.0);
                est.with_rows(rows)
            }
            PlanNode::Shift { input, offset } => {
                let est = self.plan(input);
                Est {
                    rows: est.rows,
                    cols: est
                        .cols
                        .iter()
                        .map(|(id, c)| (AttrId(id.0.saturating_add(*offset)), c.clone()))
                        .collect(),
                }
            }
            _ => Est::unknown(DEFAULT_ROWS, ids()),
        }
    }

    fn scan(&self, scan: &Scan) -> Est {
        let ids = scan.output.iter().map(|c| c.id);
        let Some(stats) = self.stats.table(scan) else {
            return Est::unknown(DEFAULT_ROWS, ids);
        };
        let rows = to_f64(stats.rows).max(1.0);
        let cols = scan
            .output
            .iter()
            .map(|c| {
                let s = stats.columns.get(&c.name).cloned().unwrap_or_default();
                let col = Col {
                    ndv: s.distinct.map_or(rows, to_f64).clamp(1.0, rows),
                    ndv_known: s.distinct.is_some(),
                    nulls: s.nulls.map_or(0.0, |n| (to_f64(n) / rows).min(1.0)),
                    range: s
                        .min
                        .as_ref()
                        .and_then(number)
                        .zip(s.max.as_ref().and_then(number)),
                };
                (c.id, col)
            })
            .collect();
        Est { rows, cols }
    }
}

/// The estimate of a projection or aggregate's outputs over `rows` rows.
fn outputs(input: &Est, exprs: &[Arc<Expr>], rows: f64) -> Est {
    let cols = exprs
        .iter()
        .filter_map(|e| {
            let out = output_of(e)?;
            let col = match &strip_alias(e).node {
                ExprNode::Resolved(ResolvedExpr::Column(c)) => input.cols.get(&c.id).cloned(),
                _ => None,
            };
            Some((out.id, col.unwrap_or_else(|| distinct(rows))))
        })
        .collect();
    Est { rows, cols }.with_rows(rows)
}

/// The column `e` reads directly.
fn column<'e>(e: &'e Expr, est: &'e Est) -> Option<&'e Col> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => est.cols.get(&c.id),
        _ => None,
    }
}

fn column_ndv(e: &Expr, est: &Est) -> Option<f64> {
    column(e, est).map(|c| c.ndv)
}

fn literal(e: &Expr) -> Option<&Literal> {
    match &e.node {
        ExprNode::Literal(l) => Some(l),
        _ => None,
    }
}

/// The two columns of an equality between columns, with their distinct
/// counts.
pub(crate) fn equi_columns(pred: &Expr, est: &Est) -> Option<[(AttrId, f64); 2]> {
    let Some((Builtin::Eq | Builtin::EqNullSafe, [a, b], _)) = builtin(pred) else {
        return None;
    };
    let side = |e: &Expr| match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => Some((c.id, est.cols.get(&c.id)?.ndv)),
        _ => None,
    };
    Some([side(a)?, side(b)?])
}

/// The fraction of rows of `est` that satisfy `pred`.
pub(crate) fn selectivity(pred: &Expr, est: &Est) -> f64 {
    let s = match (builtin(pred), literal(pred)) {
        (_, Some(Literal::Boolean(true))) => 1.0,
        (_, Some(_)) => 0.0,
        (Some((Builtin::And, [a, b], _)), _) => selectivity(a, est) * selectivity(b, est),
        (Some((Builtin::Or, [a, b], _)), _) => {
            let (x, y) = (selectivity(a, est), selectivity(b, est));
            x + y - x * y
        }
        (Some((Builtin::Not, [a], _)), _) => 1.0 - selectivity(a, est),
        (Some((Builtin::Eq | Builtin::EqNullSafe, [a, b], _)), _) => {
            match (column(a, est), column(b, est)) {
                (Some(x), Some(y)) => 1.0 / x.ndv.max(y.ndv),
                // `col = literal`: one value of the column's. Without a
                // distinct count, `1 / rows` would claim every value is
                // unique (one matching row); a low-cardinality column
                // (`l_returnflag = 'R'`) would then look tiny and be
                // built.
                (Some(x), None) | (None, Some(x)) => {
                    let with_literal = literal(a).is_some() || literal(b).is_some();
                    if with_literal && !x.ndv_known {
                        DEFAULT_SELECTIVITY
                    } else {
                        1.0 / x.ndv
                    }
                }
                (None, None) => DEFAULT_SELECTIVITY,
            }
        }
        (
            Some((op @ (Builtin::Lt | Builtin::LtEq | Builtin::Gt | Builtin::GtEq), [a, b], _)),
            _,
        ) => range(op, a, b, est),
        (Some((Builtin::Between, [v, lo, hi], _)), _) => {
            match (below(v, lo, est), below(v, hi, est)) {
                (Some(a), Some(b)) => b - a,
                _ => DEFAULT_SELECTIVITY,
            }
        }
        (Some((Builtin::In, [v, list @ ..], _)), _) => match column(v, est) {
            Some(c) if c.ndv_known => to_f64(u64::try_from(list.len()).unwrap_or(u64::MAX)) / c.ndv,
            _ => DEFAULT_SELECTIVITY,
        },
        (Some((Builtin::IsNull, [a], _)), _) => {
            column(a, est).map_or(DEFAULT_SELECTIVITY, |c| c.nulls)
        }
        (Some((Builtin::IsNotNull, [a], _)), _) => {
            column(a, est).map_or(1.0 - DEFAULT_SELECTIVITY, |c| 1.0 - c.nulls)
        }
        _ => DEFAULT_SELECTIVITY,
    };
    if s.is_finite() {
        s.clamp(0.0, 1.0)
    } else {
        DEFAULT_SELECTIVITY
    }
}

/// The fraction of the column `v`'s range below the literal `bound`, from
/// the column's bounds.
fn below(v: &Expr, bound: &Expr, est: &Est) -> Option<f64> {
    let col = column(v, est)?;
    let ((lo, hi), x) = (col.range?, number(literal(bound)?)?);
    (hi > lo).then(|| ((x - lo) / (hi - lo)).clamp(0.0, 1.0))
}

/// The selectivity of `a op b` from a column's bounds; a third without them.
fn range(op: Builtin, a: &Expr, b: &Expr, est: &Est) -> f64 {
    let (fraction, upper) = match (below(a, b, est), below(b, a, est)) {
        // `col < lit` keeps the part below; `lit < col` the part above.
        (Some(f), _) => (f, matches!(op, Builtin::Lt | Builtin::LtEq)),
        (_, Some(f)) => (f, matches!(op, Builtin::Gt | Builtin::GtEq)),
        _ => return 1.0 / 3.0,
    };
    if upper { fraction } else { 1.0 - fraction }
}

#[cfg(test)]
mod tests {
    use krishiv_plan::Literal;
    use krishiv_types::DecimalType;

    use super::number;

    #[test]
    fn literals_compare_on_one_scale() {
        let dec = Literal::Decimal {
            unscaled: 150,
            data_type: DecimalType::try_new(5, 2).unwrap(),
        };
        let cases = [
            (Literal::Byte(1), Some(1.0)),
            (Literal::Short(2), Some(2.0)),
            (Literal::Long(3), Some(3.0)),
            (Literal::Float(0.5), Some(0.5)),
            (dec, Some(1.5)),
            (Literal::Double(f64::NAN), None),
            (Literal::String("x".into()), None),
        ];
        for (l, want) in cases {
            assert_eq!(number(&l), want, "{l:?}");
        }
    }
}
