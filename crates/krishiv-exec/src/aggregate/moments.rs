// SPDX-License-Identifier: Apache-2.0
//! The moment aggregates: `stddev`, `variance`, `skewness`, `kurtosis`,
//! `covar_*` and `corr`, with Spark 4.2's buffers and arithmetic
//! (`CentralMomentAgg`, `Covariance` and `PearsonCorrelation` of
//! `org.apache.spark.sql.catalyst.expressions.aggregate`).
//!
//! A group's state is Spark's aggregation buffer, `double`s all: a count
//! `n`, the running mean(s) and sums of powers of the deviations from the
//! mean(s):
//!
//! | function | buffer |
//! |---|---|
//! | `stddev_*`, `var_*` | `n`, `avg`, `m2` |
//! | `skewness` | `n`, `avg`, `m2`, `m3` |
//! | `kurtosis` | `n`, `avg`, `m2`, `m3`, `m4` |
//! | `covar_*` | `n`, `xAvg`, `yAvg`, `ck` |
//! | `corr` | `n`, `xAvg`, `yAvg`, `ck`, `xMk`, `yMk` |
//!
//! A row folds in with the one-pass update of Meng's "Simpler Online
//! Updates for Arbitrary-Order Central Moments" ([`update`]); two buffers
//! combine with the pairwise formulas of the same paper ([`merge`]). The
//! operations are applied in Spark's order, so one pass over rows in one
//! order gives Spark's bits. The naive sum-of-squares formulas are not
//! used: they lose all precision for values far from zero.
//!
//! [`update`] and [`merge`] are the only arithmetic: the default and the
//! row-layout tables fold rows and merge spilled or partial states through
//! the same [`Moments`], so they cannot disagree.
//!
//! # Floating-point tolerance
//! One pass in a fixed order, and any fixed partitioning merged in a fixed
//! order, are deterministic to the bit. Different partitionings (split the
//! input in two, in four, merge in another order) give the same value only
//! up to rounding: the merge reassociates additions and products, as it does
//! in Spark. For well-conditioned data (the mean is not many orders of
//! magnitude above the spread) the relative difference is a few ulps; the
//! tests allow `1e-9`.
//!
//! # NULL, NaN, infinity
//! A row with a NULL in any argument is skipped. NaN and infinities flow
//! through the arithmetic as IEEE doubles do (`stddev` of `1, NaN` is NaN;
//! an infinity makes the deviations NaN); a NaN count of rows never occurs.
//!
//! # Division by zero
//! Where Spark divides by zero for want of rows or spread (one row of a
//! sample statistic, no variance for `skewness` and `kurtosis`) the value is
//! NULL, or NaN under `spark.sql.legacy.statisticalAggregate`. `corr` of
//! several rows of which one column is constant is not governed by that
//! flag: it is `DIVIDE_BY_ZERO` under ANSI and NULL without.

use arrow_array::{ArrayRef, Float64Array};
use arrow_schema::{DataType, Field};

use krishiv_common::error::{DivideByZero, KrishivError};

use super::{ANSI_CONF, AggregateFunction};
use crate::error::{Result, internal};

/// The widest buffer (`corr`'s).
const WIDTH: usize = 6;

/// One group's buffer: the first [`Family::width`] entries are used.
type Row = [f64; WIDTH];

/// The buffer attributes of `f`, in Spark's order; `None` for an aggregate
/// that is not a moment aggregate.
pub(super) fn buffer(f: &AggregateFunction) -> Option<&'static [&'static str]> {
    use AggregateFunction as F;
    Some(match f {
        F::StddevSamp | F::StddevPop | F::VarSamp | F::VarPop => &["n", "avg", "m2"],
        F::Skewness => &["n", "avg", "m2", "m3"],
        F::Kurtosis => &["n", "avg", "m2", "m3", "m4"],
        F::CovarSamp | F::CovarPop => &["n", "xAvg", "yAvg", "ck"],
        // The regression aggregates share `corr`'s buffer (which holds
        // the marginal sums of squares too): only rows where both
        // arguments are non-NULL count, as for `corr` and `covar_*`.
        F::Corr
        | F::RegrCount
        | F::RegrAvgx
        | F::RegrAvgy
        | F::RegrSxx
        | F::RegrSyy
        | F::RegrSxy
        | F::RegrSlope
        | F::RegrIntercept
        | F::RegrR2 => &["n", "xAvg", "yAvg", "ck", "xMk", "yMk"],
        _ => return None,
    })
}

/// Whether a moment aggregate reads a column of type `t` (as doubles).
pub(super) fn reads_as_double(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Float32
            | DataType::Float64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
    )
}

/// The partial-state columns of moment aggregate `f` (names derive from
/// the aggregate's `name`): Spark's buffer, non-nullable `double`s.
pub(super) fn state_fields(f: &AggregateFunction, name: &str) -> Vec<Field> {
    buffer(f)
        .unwrap_or_default()
        .iter()
        .map(|attr| Field::new(format!("{name}#{attr}"), DataType::Float64, false))
        .collect()
}

/// The arguments of an aggregate other than `count`: two for the pair
/// statistics (`corr`, `covar_*`, `regr_*`), else one.
pub(crate) fn arity(f: &AggregateFunction) -> usize {
    match f {
        AggregateFunction::CovarSamp
        | AggregateFunction::CovarPop
        | AggregateFunction::Corr
        | AggregateFunction::RegrCount
        | AggregateFunction::RegrAvgx
        | AggregateFunction::RegrAvgy
        | AggregateFunction::RegrSxx
        | AggregateFunction::RegrSyy
        | AggregateFunction::RegrSxy
        | AggregateFunction::RegrSlope
        | AggregateFunction::RegrIntercept
        | AggregateFunction::RegrR2 => 2,
        _ => 1,
    }
}

/// The buffer's shape: which update and merge apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    /// `CentralMomentAgg` of this order (2, 3 or 4).
    Central(usize),
    /// `Covariance` (`marginals`: false) or `PearsonCorrelation` (true).
    Pair { marginals: bool },
}

impl Family {
    fn of(f: &AggregateFunction) -> Option<Self> {
        let width = buffer(f)?.len();
        Some(if arity(f) == 2 {
            Self::Pair {
                marginals: width == WIDTH,
            }
        } else {
            Self::Central(width - 1)
        })
    }

    fn width(self) -> usize {
        match self {
            Self::Central(order) => order + 1,
            Self::Pair { marginals: false } => 4,
            Self::Pair { marginals: true } => WIDTH,
        }
    }
}

/// Folds one row into a buffer: `x` (and `y` for the pair statistics).
/// Spark's `updateExpressions`.
fn update(family: Family, s: &mut Row, x: f64, y: f64) {
    match family {
        Family::Central(order) => {
            let [n, avg, m2, m3, m4, _] = *s;
            let new_n = n + 1.0;
            let delta = x - avg;
            let delta_n = delta / new_n;
            let new_avg = avg + delta_n;
            let new_m2 = m2 + delta * (delta - delta_n);
            let delta2 = delta * delta;
            let delta_n2 = delta_n * delta_n;
            let new_m3 = if order >= 3 {
                m3 - 3.0 * delta_n * new_m2 + delta * (delta2 - delta_n2)
            } else {
                0.0
            };
            let new_m4 = if order >= 4 {
                m4 - 4.0 * delta_n * new_m3 - 6.0 * delta_n2 * new_m2
                    + delta * (delta * delta2 - delta_n * delta_n2)
            } else {
                0.0
            };
            *s = [new_n, new_avg, new_m2, new_m3, new_m4, 0.0];
        }
        Family::Pair { marginals } => {
            let [n, x_avg, y_avg, ck, x_mk, y_mk] = *s;
            let new_n = n + 1.0;
            let dx = x - x_avg;
            let dx_n = dx / new_n;
            let dy = y - y_avg;
            let dy_n = dy / new_n;
            let new_x_avg = x_avg + dx_n;
            let new_y_avg = y_avg + dy_n;
            let new_ck = ck + dx * (y - new_y_avg);
            let (new_x_mk, new_y_mk) = if marginals {
                (x_mk + dx * (x - new_x_avg), y_mk + dy * (y - new_y_avg))
            } else {
                (0.0, 0.0)
            };
            *s = [new_n, new_x_avg, new_y_avg, new_ck, new_x_mk, new_y_mk];
        }
    }
}

/// Combines buffer `b` into buffer `a`. Spark's `mergeExpressions` (`a` is
/// the left buffer, `b` the right one). Applied even when a side is empty,
/// as Spark does.
fn merge(family: Family, a: &mut Row, b: &Row) {
    let new_n = a[0] + b[0];
    match family {
        Family::Central(order) => {
            let [n1, avg1, m2_1, m3_1, m4_1, _] = *a;
            let [n2, avg2, m2_2, m3_2, m4_2, _] = *b;
            let delta = avg2 - avg1;
            let delta_n = if new_n == 0.0 { 0.0 } else { delta / new_n };
            let new_avg = avg1 + delta_n * n2;
            let new_m2 = m2_1 + m2_2 + delta * delta_n * n1 * n2;
            let new_m3 = if order >= 3 {
                m3_1 + m3_2
                    + delta_n * delta_n * delta * n1 * n2 * (n1 - n2)
                    + 3.0 * delta_n * (n1 * m2_2 - n2 * m2_1)
            } else {
                0.0
            };
            let new_m4 = if order >= 4 {
                m4_1 + m4_2
                    + delta_n * delta_n * delta_n * delta * n1 * n2 * (n1 * n1 - n1 * n2 + n2 * n2)
                    + 6.0 * delta_n * delta_n * (n1 * n1 * m2_2 + n2 * n2 * m2_1)
                    + 4.0 * delta_n * (n1 * m3_2 - n2 * m3_1)
            } else {
                0.0
            };
            *a = [new_n, new_avg, new_m2, new_m3, new_m4, 0.0];
        }
        Family::Pair { marginals } => {
            let [n1, x_avg1, y_avg1, ck1, x_mk1, y_mk1] = *a;
            let [n2, x_avg2, y_avg2, ck2, x_mk2, y_mk2] = *b;
            let dx = x_avg2 - x_avg1;
            let dx_n = if new_n == 0.0 { 0.0 } else { dx / new_n };
            let dy = y_avg2 - y_avg1;
            let dy_n = if new_n == 0.0 { 0.0 } else { dy / new_n };
            let new_x_avg = x_avg1 + dx_n * n2;
            let new_y_avg = y_avg1 + dy_n * n2;
            let new_ck = ck1 + ck2 + dx * dy_n * n1 * n2;
            let (new_x_mk, new_y_mk) = if marginals {
                (
                    x_mk1 + x_mk2 + dx * dx_n * n1 * n2,
                    y_mk1 + y_mk2 + dy * dy_n * n1 * n2,
                )
            } else {
                (0.0, 0.0)
            };
            *a = [new_n, new_x_avg, new_y_avg, new_ck, new_x_mk, new_y_mk];
        }
    }
}

/// The value of `f` over buffer `s`: NULL for no rows; where Spark divides
/// by zero, NULL, or NaN with `legacy` (`spark.sql.legacy.statisticalAggregate`).
/// `corr` of several rows over a constant column is the exception: a
/// division by zero whatever `legacy` says, `DIVIDE_BY_ZERO` with `ansi` and
/// NULL without. Spark's `evaluateExpression`.
///
/// The pair buffer's slots hold `x` (the first argument) in `xAvg`/`xMk`
/// and `y` (the second) in `yAvg`/`yMk`, as `update` folds them; the
/// regression aggregates read them as `regr_*(y, x)` documents them (so
/// `regr_avgx` is the `yAvg` slot, the mean of the second argument).
fn finish(f: &AggregateFunction, s: &Row, legacy: bool, ansi: bool) -> Result<Option<f64>> {
    use AggregateFunction as F;
    // `regr_count` counts the pairs (never NULL, even with none).
    if f == &F::RegrCount {
        return Ok(Some(s[0]));
    }
    let [n, _, m2, _, m4, _] = *s;
    if n == 0.0 {
        return Ok(None);
    }
    let divide_by_zero = legacy.then_some(f64::NAN);
    Ok(match f {
        F::StddevPop => Some((m2 / n).sqrt()),
        F::StddevSamp | F::VarSamp | F::CovarSamp | F::Corr if n == 1.0 => divide_by_zero,
        F::StddevSamp => Some((m2 / (n - 1.0)).sqrt()),
        F::VarPop => Some(m2 / n),
        F::VarSamp => Some(m2 / (n - 1.0)),
        F::Skewness | F::Kurtosis if m2 == 0.0 => divide_by_zero,
        F::Skewness => Some(n.sqrt() * s[3] / (m2 * m2 * m2).sqrt()),
        F::Kurtosis => Some(n * m4 / (m2 * m2) - 3.0),
        // The pair statistics keep their `ck` in the fourth slot and
        // `xMk`, `yMk` in the last two.
        F::CovarPop => Some(s[3] / n),
        F::CovarSamp => Some(s[3] / (n - 1.0)),
        F::Corr if s[4] * s[5] == 0.0 => {
            if ansi {
                return Err(KrishivError::from(DivideByZero {
                    config: ANSI_CONF.to_owned(),
                }));
            }
            None
        }
        F::Corr => Some(s[3] / (s[4] * s[5]).sqrt()),
        // The regression aggregates: `x` is the second argument (the
        // independent variable, in `yAvg`/`yMk`) and `y` the first (in
        // `xAvg`/`xMk`); Spark's `regr_*` evaluate expressions.
        // (`regr_count` returns above: it is never NULL, even with none.)
        F::RegrAvgx => Some(s[2]),
        F::RegrAvgy => Some(s[1]),
        F::RegrSxx => Some(s[5]),
        F::RegrSyy => Some(s[4]),
        F::RegrSxy => Some(s[3]),
        // NULL when `x` does not vary.
        F::RegrSlope | F::RegrIntercept | F::RegrR2 if s[5] == 0.0 => None,
        F::RegrSlope => Some(s[3] / s[5]),
        F::RegrIntercept => Some(s[1] - s[3] / s[5] * s[2]),
        // `1` when only `y` does not vary, NULL when `x` does not;
        // otherwise the squared correlation, as Spark squares `corr`.
        F::RegrR2 if s[4] == 0.0 => Some(1.0),
        F::RegrR2 => {
            let corr = s[3] / (s[4] * s[5]).sqrt();
            Some(corr * corr)
        }
        _ => None,
    })
}

/// The states of one moment aggregate for every group, one vector per
/// buffer attribute (column-oriented, as the other accumulators).
#[derive(Debug)]
pub(super) struct Moments {
    function: AggregateFunction,
    family: Family,
    /// NaN instead of NULL where Spark divides by zero.
    legacy: bool,
    /// `corr` over a constant column is `DIVIDE_BY_ZERO`, not NULL.
    ansi: bool,
    cols: Vec<Vec<f64>>,
}

impl Moments {
    /// No groups yet, for moment aggregate `function`.
    pub(super) fn new(function: AggregateFunction, ansi: bool) -> Result<Self> {
        let family = Family::of(&function)
            .ok_or_else(|| internal(format!("{function:?} is not a moment aggregate")))?;
        Ok(Self {
            function,
            family,
            legacy: false,
            ansi,
            cols: vec![Vec::new(); family.width()],
        })
    }

    pub(super) fn legacy(&mut self, on: bool) {
        self.legacy = on;
    }

    /// The number of state columns.
    pub(super) fn width(&self) -> usize {
        self.family.width()
    }

    /// Each vector as (length, capacity, element size).
    pub(super) fn vecs(&self) -> Vec<(usize, usize, usize)> {
        self.cols
            .iter()
            .map(|c| (c.len(), c.capacity(), size_of::<f64>()))
            .collect()
    }

    /// Adds groups up to `n` with empty states.
    pub(super) fn resize(&mut self, n: usize) {
        self.cols.iter_mut().for_each(|c| c.resize(n, 0.0));
    }

    /// Reserves room for `new` more groups.
    pub(super) fn reserve(&mut self, new: usize) {
        self.cols.iter_mut().for_each(|c| c.reserve(new));
    }

    fn load(&self, g: u32) -> Result<Row> {
        let g = g as usize;
        let mut row = [0.0; WIDTH];
        for (slot, col) in row.iter_mut().zip(&self.cols) {
            *slot = *col
                .get(g)
                .ok_or_else(|| internal(format!("aggregate group {g} out of range")))?;
        }
        Ok(row)
    }

    fn store(&mut self, g: u32, row: &Row) -> Result<()> {
        for (col, v) in self.cols.iter_mut().zip(row) {
            *col.get_mut(g as usize)
                .ok_or_else(|| internal(format!("aggregate group {g} out of range")))? = *v;
        }
        Ok(())
    }

    /// Folds rows into `groups` (one group id per row). `args` are the
    /// argument columns as doubles; `valid(i)` says whether row `i` has no
    /// NULL argument.
    pub(super) fn update(
        &mut self,
        groups: &[u32],
        args: &[&[f64]],
        valid: impl Fn(usize) -> bool,
    ) -> Result<()> {
        let (Some(x), y) = (args.first(), args.get(1)) else {
            return Err(internal("a moment aggregate without arguments".to_owned()));
        };
        if x.len() != groups.len()
            || args.len() != arity(&self.function)
            || y.is_some_and(|y| y.len() != groups.len())
        {
            return Err(internal(
                "a moment aggregate over columns of other lengths than its rows".to_owned(),
            ));
        }
        for (i, &g) in groups.iter().enumerate() {
            if valid(i) {
                let mut s = self.load(g)?;
                update(self.family, &mut s, x[i], y.map_or(0.0, |y| y[i]));
                self.store(g, &s)?;
            }
        }
        Ok(())
    }

    /// Merges partial states: `states` are the state columns, in buffer
    /// order, one row per entry of `groups`.
    pub(super) fn merge(&mut self, groups: &[u32], states: &[&[f64]]) -> Result<()> {
        if states.len() != self.width() || states.iter().any(|c| c.len() != groups.len()) {
            return Err(internal(
                "moment states of another shape than the rows merged".to_owned(),
            ));
        }
        for (i, &g) in groups.iter().enumerate() {
            let mut incoming = [0.0; WIDTH];
            for (slot, col) in incoming.iter_mut().zip(states) {
                *slot = col[i];
            }
            let mut s = self.load(g)?;
            merge(self.family, &mut s, &incoming);
            self.store(g, &s)?;
        }
        Ok(())
    }

    /// The partial states of groups `ids`, one `double` column per buffer
    /// attribute.
    pub(super) fn state(&self, ids: &[u32]) -> Result<Vec<ArrayRef>> {
        let rows = ids
            .iter()
            .map(|&g| self.load(g))
            .collect::<Result<Vec<_>>>()?;
        Ok((0..self.width())
            .map(|k| {
                let v: Float64Array = rows.iter().map(|r| r[k]).collect();
                std::sync::Arc::new(v) as ArrayRef
            })
            .collect())
    }

    /// The final values of groups `ids`.
    pub(super) fn evaluate(&self, ids: &[u32]) -> Result<ArrayRef> {
        let v = ids
            .iter()
            .map(|&g| finish(&self.function, &self.load(g)?, self.legacy, self.ansi))
            .collect::<Result<Float64Array>>()?;
        Ok(std::sync::Arc::new(v))
    }
}

#[cfg(test)]
mod tests;
