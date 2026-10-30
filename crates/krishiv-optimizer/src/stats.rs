// SPDX-License-Identifier: Apache-2.0
//! Table statistics and cardinality estimates.
//!
//! Estimates follow the textbook independence assumptions: a predicate's
//! selectivity is `1 / NDV` for an equality with a literal (when the
//! column's distinct count is known, else [`DEFAULT_SELECTIVITY`]), the covered
//! fraction of `[min, max]` for a range, `1 / max(NDV)` over the known
//! distinct counts for an equi-join (a count that is only the all-distinct
//! guess is never used), and [`DEFAULT_SELECTIVITY`] when nothing better is
//! known; conjunctions multiply and disjunctions add. Without statistics a
//! table holds [`DEFAULT_ROWS`] rows whose columns are all distinct.
//!
//! What a relation's columns look like after each operator (ranges, null
//! fractions, distinct counts) is propagated by `estimate`; the types it
//! produces are here.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use krishiv_plan::{AttrId, Expr, ExprNode, Literal, LogicalPlan, ResolvedExpr, Scan};

use crate::estimate::Estimator;
use crate::expr::substitute;

/// Whether `e` reads a common-subexpression column (an alias named `cse_*`,
/// see `crate::cse`).
pub(crate) fn reads_cse(e: &Expr) -> bool {
    let mut found = false;
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(x) = stack.pop() {
        if !seen.insert(std::ptr::from_ref(x)) {
            continue;
        }
        if let ExprNode::Resolved(ResolvedExpr::Column(c)) = &x.node
            && crate::cse::is_cse_column(c)
        {
            found = true;
            break;
        }
        x.for_each_child(&mut |c| stack.push(c));
    }
    found
}

/// The definitions of the common-subexpression columns computed anywhere
/// below `p` (subquery plans excluded), by attribute id. Each holds exactly
/// its definition's values, so estimates read the definition instead of
/// the column: the same rows are estimated before and after the
/// common-subexpression pass runs.
pub(crate) fn cse_definitions(p: &LogicalPlan) -> HashMap<AttrId, Arc<Expr>> {
    let mut defs = HashMap::new();
    let mut plans = vec![p];
    let mut seen_plans = std::collections::HashSet::new();
    while let Some(q) = plans.pop() {
        if !seen_plans.insert(std::ptr::from_ref(q)) {
            continue;
        }
        let mut stack: Vec<&Expr> = Vec::new();
        q.for_each_expr(&mut |e| stack.push(e));
        let mut seen = std::collections::HashSet::new();
        while let Some(e) = stack.pop() {
            if !seen.insert(std::ptr::from_ref(e)) {
                continue;
            }
            if let ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) = &e.node
                && crate::cse::is_cse_column(output)
            {
                defs.insert(output.id, Arc::clone(expr));
            }
            e.for_each_child(&mut |c| stack.push(c));
        }
        q.for_each_child(&mut |c| plans.push(c));
    }
    defs
}

/// `e` with every common-subexpression column replaced by its definition
/// (see [`cse_definitions`]); `e` itself when there is nothing to replace.
/// Definitions hold no `cse_*` references themselves, so one pass suffices.
pub(crate) fn resolve_cse(e: &Arc<Expr>, defs: &HashMap<AttrId, Arc<Expr>>) -> Arc<Expr> {
    if defs.is_empty() || !reads_cse(e) {
        return Arc::clone(e);
    }
    substitute(e, defs)
}

/// The rows of a table without statistics.
pub const DEFAULT_ROWS: f64 = 1_000_000.0;

/// The selectivity of a predicate the estimator cannot read.
pub const DEFAULT_SELECTIVITY: f64 = 0.2;

/// Where a column's distinct count comes from, weakest first. When
/// statistics combine (a join of two columns, a union), the weakest input's
/// source is the result's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[non_exhaustive]
pub enum StatSource {
    /// The default guess: no statistics, every value taken as distinct.
    Guess,
    /// Bounded by an integral column's exact range (`c_nationkey` in
    /// `0..=24` has at most 25 values), from the table's bounds.
    IntegralRange,
    /// Reported by the table provider (Parquet footers, table-format
    /// metadata) or supplied with the table.
    #[default]
    Provided,
}

/// Statistics of one column.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ColumnStats {
    /// Where `distinct` comes from; without `distinct` the count is a
    /// guess whatever this says.
    pub source: StatSource,
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
    Estimator::new(stats).plan(plan).rows_for_costing()
}

/// What is estimated of one attribute of a relation.
#[derive(Debug, Clone, PartialEq)]
pub struct AttrStats {
    /// Distinct non-null values, at least 1. Without statistics, every
    /// value is taken as distinct (`ndv` = rows); [`Self::ndv_known`] tells
    /// the two apart.
    pub ndv: f64,
    /// The fraction of rows that are null.
    pub null_frac: f64,
    /// A lower bound of the non-null values, as a number on the scale the
    /// type compares on (`None` when unknown).
    pub min: Option<f64>,
    /// An upper bound of the non-null values.
    pub max: Option<f64>,
    /// Where the distinct count comes from.
    pub source: StatSource,
}

impl AttrStats {
    /// An attribute of `rows` rows nothing is known of: all values distinct.
    #[must_use]
    pub fn guess(rows: f64) -> Self {
        Self {
            ndv: rows.max(1.0),
            null_frac: 0.0,
            min: None,
            max: None,
            source: StatSource::Guess,
        }
    }

    /// Whether the distinct count comes from statistics rather than the
    /// all-distinct guess. Only then may a selectivity divide by it.
    #[must_use]
    pub fn ndv_known(&self) -> bool {
        self.source != StatSource::Guess
    }
}

/// The estimated output of a relation.
#[derive(Debug, Clone, Default)]
pub struct RelStats {
    /// Rows. Zero only where a predicate proves the relation empty; cost
    /// decisions read at least 1 ([`Self::rows_for_costing`]).
    pub rows: f64,
    /// Per output attribute.
    pub cols: HashMap<AttrId, AttrStats>,
    /// Whether every table below reports statistics, so the estimate does
    /// not rest on default row counts (a `WITH` reference counts as
    /// default-sized).
    pub from_statistics: bool,
}

impl RelStats {
    pub(crate) fn unknown(rows: f64, ids: impl IntoIterator<Item = AttrId>) -> Self {
        let cols = ids
            .into_iter()
            .map(|id| (id, AttrStats::guess(rows)))
            .collect();
        Self {
            rows,
            cols,
            from_statistics: false,
        }
    }

    /// The same columns over `rows` rows: no column has more distinct values
    /// than rows.
    pub(crate) fn with_rows(mut self, rows: f64) -> Self {
        self.rows = rows.max(0.0);
        for c in self.cols.values_mut() {
            c.ndv = cap(c.ndv, self.rows);
        }
        self
    }

    /// The rows cost decisions read: at least 1.
    #[must_use]
    pub fn rows_for_costing(&self) -> f64 {
        self.rows.max(1.0)
    }
}

/// `ndv` limited to `rows`, but at least 1.
pub(crate) fn cap(ndv: f64, rows: f64) -> f64 {
    ndv.min(rows).max(1.0)
}

/// A literal as a number on the scale its type compares on.
#[expect(clippy::cast_precision_loss, reason = "estimates only")]
pub(crate) fn number(l: &Literal) -> Option<f64> {
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
pub(crate) fn to_f64(v: u64) -> f64 {
    v as f64
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
