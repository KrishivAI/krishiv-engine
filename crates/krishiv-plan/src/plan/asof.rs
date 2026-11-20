// SPDX-License-Identifier: Apache-2.0
//! The match of an as-of join ([`PlanNode::AsOfJoin`](super::PlanNode::AsOfJoin)): Spark 4.3's
//! SQL `ASOF JOIN ... MATCH_CONDITION (l op r)` and Spark Connect's `AsOfJoin` relation.

use std::sync::Arc;

use crate::Expr;

/// Which right rows an as-of join may pick for a left row, by their as-of value `R`
/// against the left row's `L`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AsOfDirection {
    /// The greatest `R` at most `L` (`L >= R`; `L > R` without exact matches).
    Backward,
    /// The least `R` at least `L` (`L <= R`; `L < R` without exact matches).
    Forward,
    /// The `R` nearest to `L` either way (Spark Connect only).
    Nearest,
}

impl AsOfDirection {
    /// The direction after swapping the operands (`a >= b` is `b <= a`).
    #[must_use]
    pub const fn flip(self) -> Self {
        match self {
            Self::Backward => Self::Forward,
            Self::Forward => Self::Backward,
            Self::Nearest => Self::Nearest,
        }
    }

    /// The SQL comparison of `L` and `R` this direction stands for (`None` for
    /// [`AsOfDirection::Nearest`]).
    #[must_use]
    pub const fn sql_operator(self, allow_exact: bool) -> Option<&'static str> {
        match (self, allow_exact) {
            (Self::Backward, true) => Some(">="),
            (Self::Backward, false) => Some(">"),
            (Self::Forward, true) => Some("<="),
            (Self::Forward, false) => Some("<"),
            (Self::Nearest, _) => None,
        }
    }
}

/// The match of an as-of join: the as-of values of the two sides, the direction, whether
/// an equal value matches, and the largest distance allowed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsOfMatch {
    /// The left row's as-of value `L` (resolved: reads only the left input).
    pub left: Arc<Expr>,
    /// The right row's as-of value `R` (resolved: reads only the right input).
    pub right: Arc<Expr>,
    /// Which right rows may match.
    pub direction: AsOfDirection,
    /// Whether `R = L` matches.
    pub allow_exact: bool,
    /// The largest `|L - R|` that matches: a foldable value at least zero (Connect only).
    pub tolerance: Option<Arc<Expr>>,
    /// Resolved: Spark's distance (`orderExpression`) over the operands as normalized, where
    /// their type subtracts: `R - L` forward, and backward `L - R` when
    /// [`AsOfMatch::distance_on_every_pair`]. The executor evaluates it where Spark does, so
    /// what it raises (an ANSI overflow) is Spark's error. `None` before analysis, and when no
    /// row can raise.
    pub distance: Option<Arc<Expr>>,
    /// Where Spark evaluates the distance. `false` (SQL, Spark's `SortMergeAsOfJoinExec` scan):
    /// forward only, on each left row's matched right row and the next right row that passes.
    /// `true` (Connect, whose `AsOfJoin` Spark rewrites into `min_by` over every passing right
    /// row, `RewriteAsOfJoin`): both directions, on every pair that passes. Set by the front
    /// end and kept through analysis.
    pub distance_on_every_pair: bool,
    /// Unresolved SQL: the operands are as written and the analyzer puts the one that
    /// reads the left input first (Spark's `normalizeMatchOperands`); `false` once
    /// resolved and for Connect, whose operands are already per side.
    pub as_written: bool,
}

impl AsOfMatch {
    /// The match of a SQL `MATCH_CONDITION (left op right)`: `op` is `>=`, `>`, `<=` or
    /// `<`; `None` for any other.
    #[must_use]
    pub fn sql(left: Arc<Expr>, op: &str, right: Arc<Expr>) -> Option<Self> {
        let (direction, allow_exact) = match op {
            ">=" => (AsOfDirection::Backward, true),
            ">" => (AsOfDirection::Backward, false),
            "<=" => (AsOfDirection::Forward, true),
            "<" => (AsOfDirection::Forward, false),
            _ => return None,
        };
        Some(Self {
            left,
            right,
            direction,
            allow_exact,
            tolerance: None,
            distance: None,
            distance_on_every_pair: false,
            as_written: true,
        })
    }

    /// Its expressions in order: the two values, the tolerance, then the distance.
    pub fn exprs(&self) -> impl Iterator<Item = &Arc<Expr>> {
        [&self.left, &self.right]
            .into_iter()
            .chain(self.tolerance.as_ref())
            .chain(self.distance.as_ref())
    }

    /// Its expressions in order, mutably.
    pub fn exprs_mut(&mut self) -> impl Iterator<Item = &mut Arc<Expr>> {
        [&mut self.left, &mut self.right]
            .into_iter()
            .chain(self.tolerance.as_mut())
            .chain(self.distance.as_mut())
    }
}
