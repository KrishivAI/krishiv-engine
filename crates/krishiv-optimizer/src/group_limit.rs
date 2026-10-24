// SPDX-License-Identifier: Apache-2.0
//! Spark's `InferWindowGroupLimit`: a filter that keeps only the rows a
//! rank-like window function (`row_number`, `rank`, `dense_rank`) ranks at
//! most `k` lets the rows past rank `k` of each partition go before the
//! window is evaluated. The physical planner applies the decision
//! ([`infer_window_group_limit`]); results are the same either way.
//!
//! The conditions are Spark's (checked on Spark 4.2.0):
//! - the rule is off when `spark.sql.optimizer.windowGroupLimitThreshold`
//!   is -1;
//! - the window has an ordering, and every one of its expressions runs over
//!   `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` and is not
//!   size-based (`ntile`, `percent_rank`, `cume_dist`) or look-around
//!   (`lag`, `lead`);
//! - a conjunct of the filter compares a rank-like column with an `INT`
//!   literal: `= k`, `<= k`, `< k + 1` (literal on either side); the
//!   smallest such `k` of a column is its limit (`< k` computes `k - 1` in
//!   32-bit arithmetic, as Spark's `Int` does);
//! - `row_number` limits are taken over the others (cheaper), then the
//!   smallest limit, the first one on ties;
//! - the limit is at most the threshold and below the window input's
//!   most rows, when that is bounded;
//! - a limit below 1 keeps no rows; `row_number` without partitioning is a
//!   plain limit over the window (Spark's top-K path).
//!
//! Plans differ from Spark's in two places, never in results: the bound on
//! the window input's rows is this crate's (a `LIMIT`, a global aggregate,
//! ...; `VALUES` and `range` are unbounded here), and the physical planner
//! takes the top-K path as a top-N below the window rather than a limit
//! above it.

use std::sync::Arc;

use krishiv_plan::{
    Builtin, Expr, ExprNode, FrameBound, FrameUnits, FunctionKind, Literal, LogicalPlan, PlanNode,
    ResolvedExpr, WindowSpecRef,
};

use krishiv_types::DataType;

use crate::expr::{builtin, conjuncts};
use crate::limit::max_rows;

/// Spark's default `spark.sql.optimizer.windowGroupLimitThreshold`.
pub const DEFAULT_WINDOW_GROUP_LIMIT_THRESHOLD: i64 = 1000;

/// A rank-like window function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankLikeFunction {
    /// `row_number()`.
    RowNumber,
    /// `rank()`.
    Rank,
    /// `dense_rank()`.
    DenseRank,
}

/// What a filter over a window infers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowGroupLimit {
    /// No row passes the filter (a limit below 1): Spark's empty relation.
    Empty,
    /// `row_number` without partitioning: the first `limit` rows of the
    /// window (Spark's `Limit` over the window).
    Limit(u64),
    /// The rows of each partition `function` ranks at most `limit`.
    Group {
        /// The rank-like function.
        function: RankLikeFunction,
        /// The highest rank kept (at least 1).
        limit: u64,
    },
}

/// The window group limit of `predicate` filtering `window` (a
/// [`PlanNode::Window`]), under `threshold`
/// (`spark.sql.optimizer.windowGroupLimitThreshold`; -1 turns it off), as
/// Spark's `InferWindowGroupLimit` decides it (see the module docs).
#[must_use]
pub fn infer_window_group_limit(
    predicate: &Arc<Expr>,
    window: &LogicalPlan,
    threshold: i64,
) -> Option<WindowGroupLimit> {
    if threshold == -1 {
        return None;
    }
    let PlanNode::Window {
        input,
        exprs,
        partition_by,
        order_by,
    } = &window.node
    else {
        return None;
    };
    if order_by.is_empty() {
        return None;
    }
    let calls = exprs.iter().map(|e| call(e)).collect::<Option<Vec<_>>>()?;
    if !calls.iter().all(|c| c.expanding) {
        return None;
    }
    let conjuncts = conjuncts(predicate);
    let limits: Vec<(i32, RankLikeFunction)> = calls
        .iter()
        .filter_map(|c| {
            let f = c.rank_like?;
            conjuncts
                .iter()
                .filter_map(|e| limit_of(e, c.output))
                .min()
                .map(|l| (l, f))
        })
        .collect();
    let row_number: Vec<_> = limits
        .iter()
        .copied()
        .filter(|(_, f)| *f == RankLikeFunction::RowNumber)
        .collect();
    let selected = if row_number.is_empty() {
        limits
    } else {
        row_number
    };
    // The first smallest, as Scala's `minBy`.
    let (limit, function) = selected
        .into_iter()
        .reduce(|best, l| if l.0 < best.0 { l } else { best })?;
    let limit = i64::from(limit);
    if limit > threshold
        || max_rows(input).is_some_and(|m| i64::try_from(m).is_ok_and(|m| m <= limit))
    {
        return None;
    }
    let Ok(limit) = u64::try_from(limit) else {
        return Some(WindowGroupLimit::Empty);
    };
    if limit == 0 {
        return Some(WindowGroupLimit::Empty);
    }
    if function == RankLikeFunction::RowNumber && partition_by.is_empty() {
        return Some(WindowGroupLimit::Limit(limit));
    }
    Some(WindowGroupLimit::Group { function, limit })
}

/// One window expression as the rule reads it.
struct Call {
    output: krishiv_plan::AttrId,
    rank_like: Option<RankLikeFunction>,
    /// Over `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, neither
    /// size-based nor look-around.
    expanding: bool,
}

fn call(e: &Expr) -> Option<Call> {
    let ExprNode::Resolved(ResolvedExpr::Alias { expr, output }) = &e.node else {
        return None;
    };
    let ExprNode::Window(w) = &expr.node else {
        return None;
    };
    let ExprNode::Resolved(ResolvedExpr::Function(f)) = &w.function.node else {
        return None;
    };
    let WindowSpecRef::Spec(spec) = &w.window else {
        return None;
    };
    let name = f.id.name();
    let window_fn = f.kind == FunctionKind::Window;
    let rank_like = match name {
        "row_number" if window_fn => Some(RankLikeFunction::RowNumber),
        "rank" if window_fn => Some(RankLikeFunction::Rank),
        "dense_rank" if window_fn => Some(RankLikeFunction::DenseRank),
        _ => None,
    };
    let size_based = window_fn && matches!(name, "ntile" | "percent_rank" | "cume_dist");
    let look_around = window_fn && matches!(name, "lag" | "lead");
    let running = spec.frame.as_ref().is_some_and(|frame| {
        frame.units == FrameUnits::Rows
            && frame.lower == FrameBound::UnboundedPreceding
            && frame.upper == FrameBound::CurrentRow
    });
    Some(Call {
        output: output.id,
        rank_like,
        expanding: running && !size_based && !look_around,
    })
}

/// The comparisons Spark's `extractLimits` reads, with the column on the
/// left.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Cmp {
    Eq,
    Lt,
    LtEq,
}

/// The limit conjunct `e` puts on column `attr`, as Spark's
/// `extractLimits` reads it: `attr = k`, `attr < k`, `attr <= k`, or the
/// same with the literal first (`k = attr`, `k > attr`, `k >= attr`).
/// Where the analyzer widened the column (`CAST(attr AS BIGINT) <= 2L`,
/// `... < 2.5`), the comparison is read as Spark's
/// `UnwrapCastInBinaryComparison` leaves it over the `INT` column.
fn limit_of(e: &Expr, attr: krishiv_plan::AttrId) -> Option<i32> {
    let (op, args, _) = builtin(e)?;
    let [a, b] = args else {
        return None;
    };
    let (cmp, column, value) = match op {
        Builtin::Eq if column_of(a, attr).is_some() => (Cmp::Eq, a, b),
        Builtin::Eq => (Cmp::Eq, b, a),
        Builtin::Lt => (Cmp::Lt, a, b),
        Builtin::Gt => (Cmp::Lt, b, a),
        Builtin::LtEq => (Cmp::LtEq, a, b),
        Builtin::GtEq => (Cmp::LtEq, b, a),
        _ => return None,
    };
    let ExprNode::Literal(value) = &value.node else {
        return None;
    };
    match column_of(column, attr)? {
        // Spark's `Int` arithmetic: `attr < k` is a limit of `k - 1`.
        false => match value {
            Literal::Integer(k) if cmp == Cmp::Lt => Some(k.wrapping_sub(1)),
            Literal::Integer(k) => Some(*k),
            _ => None,
        },
        true => {
            let (floor, exact) = floor_of(value)?;
            let k = match cmp {
                Cmp::Eq if exact => floor,
                Cmp::Eq => return None,
                Cmp::Lt if exact => floor.checked_sub(1)?,
                Cmp::Lt | Cmp::LtEq => floor,
            };
            i32::try_from(k).ok()
        }
    }
}

/// Whether `e` is column `attr` (`Some(false)`) or `attr` widened by a cast
/// to a wider numeric type (`Some(true)`).
fn column_of(e: &Expr, attr: krishiv_plan::AttrId) -> Option<bool> {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) if c.id == attr => Some(false),
        ExprNode::Cast {
            expr,
            to: DataType::Long | DataType::Float | DataType::Double | DataType::Decimal(_),
            ..
        } => column_of(expr, attr).filter(|&cast| !cast).map(|_| true),
        _ => None,
    }
}

/// A numeric literal's floor and whether it is integral.
fn floor_of(l: &Literal) -> Option<(i128, bool)> {
    match l {
        Literal::Integer(v) => Some((i128::from(*v), true)),
        Literal::Long(v) => Some((i128::from(*v), true)),
        Literal::Decimal {
            unscaled,
            data_type,
        } => {
            let scale = 10_i128.checked_pow(u32::from(data_type.scale()))?;
            let floor = unscaled.div_euclid(scale);
            Some((floor, unscaled.rem_euclid(scale) == 0))
        }
        Literal::Double(v) => float_floor(*v),
        Literal::Float(v) => float_floor(f64::from(*v)),
        _ => None,
    }
}

fn float_floor(v: f64) -> Option<(i128, bool)> {
    // Beyond an `INT` the limit is out of range anyway.
    if !v.is_finite() || v.abs() > 4e9 {
        return None;
    }
    let floor = v.floor();
    #[expect(clippy::cast_possible_truncation, reason = "a finite value within 4e9")]
    Some((floor as i128, floor == v))
}
