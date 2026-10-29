// SPDX-License-Identifier: Apache-2.0
//! [`WindowSpec`]: the window of a [`Column::over`], as PySpark's
//! `pyspark.sql.Window` and `WindowSpec`. Builders only assemble the same
//! unresolved [`krishiv_plan::WindowSpec`] the SQL path lowers from a
//! `WINDOW` clause or an inline `OVER (...)` (partitioning, ordering and
//! frame); the analyzer checks and defaults it, as for SQL.

use std::sync::Arc;

use krishiv_plan::{
    Builtin, Expr, ExprNode, FrameBound, FrameUnits, Literal, SortOrder, WindowExpr, WindowFrame,
    WindowSpec as PlanWindowSpec, WindowSpecRef,
};

use crate::Column;

/// The window a [`Column::over`] evaluates its function over: partitioning
/// expressions, ordering expressions and an optional frame (built by
/// [`WindowSpec::rows_between`] or [`WindowSpec::range_between`]).
///
/// ```
/// use krishiv_session::WindowSpec;
/// use krishiv_session::functions::{col, sum};
///
/// let window = WindowSpec::new()
///     .partition_by(["dept"])
///     .order_by([col("salary").desc()])
///     .rows_between(-1, 1);
/// let total = sum(col("salary")).over(window);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowSpec {
    partition_by: Vec<Expr>,
    order_by: Vec<SortOrder>,
    frame: Option<WindowFrame>,
}

impl Default for WindowSpec {
    /// A window over all rows, in no order, with no frame: the analyzer
    /// applies Spark's default frame for the function.
    fn default() -> Self {
        Self::new()
    }
}

impl WindowSpec {
    /// A window over all rows, in no order, with no frame.
    ///
    /// ```
    /// # use krishiv_session::WindowSpec;
    /// let window = WindowSpec::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            partition_by: Vec::new(),
            order_by: Vec::new(),
            frame: None,
        }
    }

    /// This window partitioning by `cols`, replacing any partitioning
    /// (PySpark's `partitionBy`; a repeated call replaces, as Spark's
    /// does). A `&str` names a column, a [`Column`] is an expression.
    ///
    /// ```
    /// # use krishiv_session::WindowSpec;
    /// # use krishiv_session::functions::col;
    /// let window = WindowSpec::new().partition_by(["dept", "team"]);
    /// ```
    #[must_use]
    pub fn partition_by(self, cols: impl IntoIterator<Item = impl Into<Column>>) -> Self {
        Self {
            partition_by: cols.into_iter().map(|c| c.into().expr).collect(),
            ..self
        }
    }

    /// This window ordering by `cols`, replacing any ordering (PySpark's
    /// `orderBy`). A [`Column`] built by [`Column::asc`] or
    /// [`Column::desc`] orders that way, anything else ascends with nulls
    /// first, as in an `ORDER BY`.
    ///
    /// ```
    /// # use krishiv_session::WindowSpec;
    /// # use krishiv_session::functions::col;
    /// let window = WindowSpec::new().order_by([col("salary").desc()]);
    /// ```
    #[must_use]
    pub fn order_by(self, cols: impl IntoIterator<Item = impl Into<Column>>) -> Self {
        Self {
            order_by: cols.into_iter().map(|c| c.into().sort_order()).collect(),
            ..self
        }
    }

    /// This window with the `ROWS` frame from `start` to `end` (PySpark's
    /// `rowsBetween`): offsets count rows from the current row,
    /// `i64::MIN` is `UNBOUNDED PRECEDING`, `i64::MAX` is `UNBOUNDED
    /// FOLLOWING` and `0` is `CURRENT ROW`. A repeated call replaces the
    /// frame.
    ///
    /// ```
    /// # use krishiv_session::WindowSpec;
    /// let window = WindowSpec::new().rows_between(-1, 1);
    /// ```
    #[must_use]
    pub fn rows_between(self, start: i64, end: i64) -> Self {
        self.framed(FrameUnits::Rows, start, end)
    }

    /// This window with the `RANGE` frame from `start` to `end`
    /// (PySpark's `rangeBetween`), with the same bounds as
    /// [`Self::rows_between`]. A repeated call replaces the frame.
    ///
    /// ```
    /// # use krishiv_session::WindowSpec;
    /// let window = WindowSpec::new().range_between(i64::MIN, 0);
    /// ```
    #[must_use]
    pub fn range_between(self, start: i64, end: i64) -> Self {
        self.framed(FrameUnits::Range, start, end)
    }

    fn framed(mut self, units: FrameUnits, start: i64, end: i64) -> Self {
        self.frame = Some(WindowFrame {
            units,
            lower: bound(start),
            upper: bound(end),
        });
        self
    }

    /// The plan window: partitioning and ordering expressions with the
    /// frame, as SQL lowering builds them.
    pub(crate) fn into_plan(self) -> PlanWindowSpec {
        PlanWindowSpec {
            partition_by: self.partition_by.into_iter().map(Arc::new).collect(),
            order_by: self.order_by,
            frame: self.frame,
        }
    }
}

/// A frame bound for `value`, as SQL lowering writes it: `i64::MIN` is
/// `UNBOUNDED PRECEDING`, `i64::MAX` is `UNBOUNDED FOLLOWING`, `0` is
/// `CURRENT ROW`, a negative offset is `-(n)` (`n PRECEDING`) and a
/// positive one is `n` (`n FOLLOWING`). The literal is an `int` in the
/// `int` range and a `long` past it, as the SQL parser types integer
/// literals, so a bound past the `int` range fails analysis with the
/// engine's own `SPECIFIED_WINDOW_FRAME_UNACCEPTED_TYPE`, as the SQL form
/// does.
fn bound(value: i64) -> FrameBound {
    if value == i64::MIN {
        FrameBound::UnboundedPreceding
    } else if value == i64::MAX {
        FrameBound::UnboundedFollowing
    } else if value == 0 {
        FrameBound::CurrentRow
    } else if value < 0 {
        FrameBound::Offset(Arc::new(Expr::builtin(
            Builtin::Negative,
            vec![int_literal(-value)],
        )))
    } else {
        FrameBound::Offset(Arc::new(int_literal(value)))
    }
}

/// An `int` literal of `value`, a `long` one past the `int` range (as the
/// SQL parser types integer literals), so a bound past the `int` range
/// fails analysis with the engine's own
/// `SPECIFIED_WINDOW_FRAME_UNACCEPTED_TYPE`, as the SQL form does.
fn int_literal(value: i64) -> Expr {
    i32::try_from(value).map_or_else(
        |_| Expr::literal(Literal::Long(value)),
        |v| Expr::literal(Literal::Integer(v)),
    )
}

impl Column {
    /// `self OVER window` (PySpark's `Column.over`): the same window
    /// expression the SQL path builds for `function OVER (...)`. The
    /// analyzer checks the function and the window, and fails with the
    /// engine's own error where the SQL form fails.
    ///
    /// ```
    /// use krishiv_session::WindowSpec;
    /// use krishiv_session::functions::{col, sum};
    ///
    /// let window = WindowSpec::new().partition_by(["dept"]).rows_between(-1, 1);
    /// let total = sum(col("salary")).over(window);
    /// ```
    #[must_use]
    pub fn over(self, window: WindowSpec) -> Self {
        Self::new(Expr::new(ExprNode::Window(WindowExpr {
            function: Arc::new(self.expr),
            window: WindowSpecRef::Spec(window.into_plan()),
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_of(column: &Column) -> PlanWindowSpec {
        let ExprNode::Window(w) = &column.expr.node else {
            panic!("expected a window expression");
        };
        let WindowSpecRef::Spec(spec) = &w.window else {
            panic!("expected an inline window");
        };
        spec.clone()
    }

    #[test]
    fn empty_window_has_no_partitioning_ordering_or_frame() {
        let spec = spec_of(&crate::functions::col("a").over(WindowSpec::new()));
        assert!(spec.partition_by.is_empty());
        assert!(spec.order_by.is_empty());
        assert!(spec.frame.is_none());
    }

    #[test]
    fn partitioning_names_columns_and_replaces() {
        let window = WindowSpec::new()
            .partition_by(["a"])
            .partition_by([crate::functions::col("b")]);
        let spec = spec_of(&crate::functions::col("x").over(window));
        assert_eq!(spec.partition_by.len(), 1);
    }

    #[test]
    fn ordering_keeps_the_columns_sort_order() {
        use krishiv_plan::{NullOrdering, SortDirection};

        let window = WindowSpec::new().order_by([
            crate::functions::col("a"),
            crate::functions::col("b").desc(),
        ]);
        let spec = spec_of(&crate::functions::col("x").over(window));
        assert_eq!(spec.order_by.len(), 2);
        assert_eq!(
            (spec.order_by[0].direction, spec.order_by[0].nulls),
            (SortDirection::Ascending, NullOrdering::NullsFirst)
        );
        assert_eq!(
            (spec.order_by[1].direction, spec.order_by[1].nulls),
            (SortDirection::Descending, NullOrdering::NullsLast)
        );
    }

    #[test]
    fn frame_bounds_match_sql_lowering() {
        let frame = spec_of(
            &crate::functions::col("x").over(WindowSpec::new().rows_between(i64::MIN, i64::MAX)),
        )
        .frame
        .expect("a frame");
        assert!(matches!(frame.lower, FrameBound::UnboundedPreceding));
        assert!(matches!(frame.upper, FrameBound::UnboundedFollowing));

        let frame =
            spec_of(&crate::functions::col("x").over(WindowSpec::new().rows_between(-1, 1)))
                .frame
                .expect("a frame");
        assert_eq!(frame.units, FrameUnits::Rows);
        let ExprNode::Function(negative) = &as_offset(&frame.lower).node else {
            panic!("a preceding bound is a negation");
        };
        assert_eq!(negative.name, Builtin::Negative.name());
        assert!(matches!(
            &as_offset(&frame.upper).node,
            ExprNode::Literal(Literal::Integer(1))
        ));
    }

    #[test]
    fn bounds_past_the_int_range_are_long_literals() {
        let frame = spec_of(
            &crate::functions::col("x")
                .over(WindowSpec::new().rows_between(-4_000_000_000, 4_000_000_000)),
        )
        .frame
        .expect("a frame");
        let ExprNode::Function(negative) = &as_offset(&frame.lower).node else {
            panic!("a preceding bound is a negation");
        };
        let [magnitude] = &negative.args[..] else {
            panic!("a negation of one literal");
        };
        assert!(matches!(
            &magnitude.node,
            ExprNode::Literal(Literal::Long(4_000_000_000))
        ));
        assert!(matches!(
            &as_offset(&frame.upper).node,
            ExprNode::Literal(Literal::Long(4_000_000_000))
        ));
    }

    #[test]
    fn current_row_is_zero() {
        let frame = spec_of(&crate::functions::col("x").over(WindowSpec::new().rows_between(0, 0)))
            .frame
            .expect("a frame");
        assert!(matches!(frame.lower, FrameBound::CurrentRow));
        assert!(matches!(frame.upper, FrameBound::CurrentRow));
    }

    #[test]
    fn range_frames_keep_their_units() {
        let frame = spec_of(
            &crate::functions::col("x").over(WindowSpec::new().range_between(i64::MIN, i64::MAX)),
        )
        .frame
        .expect("a frame");
        assert_eq!(frame.units, FrameUnits::Range);
    }

    fn as_offset(bound: &FrameBound) -> &Arc<Expr> {
        let FrameBound::Offset(expr) = bound else {
            panic!("expected an offset bound");
        };
        expr
    }
}
