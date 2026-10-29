// SPDX-License-Identifier: Apache-2.0
//! [`Column`]: an expression of a DataFrame, built with [`crate::functions`]
//! and the methods and operators here, as PySpark's `Column`.

use std::ops;
use std::sync::Arc;

use krishiv_plan::{
    Builtin, CastMode, ColumnRef, Expr, ExprNode, FieldOp, FunctionCall, Literal, NullOrdering,
    PlanId, SortDirection, SortOrder, Star,
};
use krishiv_types::DataType;

/// An expression over a DataFrame's columns. Comparisons and boolean
/// logic are methods (`eq`, `lt`, `and`, ...), arithmetic uses Rust's
/// operators.
///
/// ```
/// use krishiv_session::functions::{col, lit};
///
/// let total = (col("price") * col("qty")).alias("total");
/// let cheap = col("price").lt(lit(10)).and(col("qty").gt(lit(0)));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Column {
    pub(crate) expr: Expr,
    /// The sort order `asc`/`desc` gave; used only by `order_by`.
    pub(crate) order: Option<(SortDirection, NullOrdering)>,
}

impl Column {
    pub(crate) fn new(expr: Expr) -> Self {
        Self { expr, order: None }
    }

    /// The column named `name` (`a`, `t.a`, `` `a.b` ``, `*`, `t.*`),
    /// optionally of the DataFrame `plan_id`.
    pub(crate) fn named(name: &str, plan_id: Option<PlanId>) -> Self {
        let mut parts = split_name(name);
        let node = if parts.last().is_some_and(|p| p == "*") {
            parts.pop();
            ExprNode::Star(Star {
                qualifier: parts,
                except: Vec::new(),
                plan_id,
            })
        } else {
            ExprNode::Column(ColumnRef {
                name: parts,
                plan_id,
            })
        };
        Self::new(Expr::new(node))
    }

    fn call(op: Builtin, args: Vec<Self>) -> Self {
        Self::new(Expr::builtin(
            op,
            args.into_iter().map(|c| c.expr).collect(),
        ))
    }

    /// The expression under `name` (SQL's `AS`).
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// let c = col("a").alias("b");
    /// ```
    #[must_use]
    pub fn alias(self, name: &str) -> Self {
        Self::new(self.expr.alias(name))
    }

    /// `CAST(self AS to)`.
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// use krishiv_types::DataType;
    /// let c = col("a").cast(DataType::Long);
    /// ```
    #[must_use]
    pub fn cast(self, to: DataType) -> Self {
        Self::new(self.expr.cast(to))
    }

    /// `TRY_CAST(self AS to)`: invalid input becomes `NULL`, even under ANSI.
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// use krishiv_types::DataType;
    /// let c = col("a").try_cast(DataType::Long);
    /// ```
    #[must_use]
    pub fn try_cast(self, to: DataType) -> Self {
        Self::new(Expr::new(ExprNode::Cast {
            expr: Arc::new(self.expr),
            to,
            mode: CastMode::Try,
        }))
    }

    /// `self = other`.
    ///
    /// ```
    /// # use krishiv_session::functions::{col, lit};
    /// let c = col("a").eq(lit(1));
    /// ```
    #[must_use]
    pub fn eq(self, other: impl Into<Self>) -> Self {
        Self::call(Builtin::Eq, vec![self, other.into()])
    }

    /// `self <> other`.
    ///
    /// ```
    /// # use krishiv_session::functions::{col, lit};
    /// let c = col("a").not_eq(lit(1));
    /// ```
    #[must_use]
    pub fn not_eq(self, other: impl Into<Self>) -> Self {
        !self.eq(other)
    }

    /// `self <=> other` (null-safe equality).
    ///
    /// ```
    /// # use krishiv_session::functions::{col, lit};
    /// let c = col("a").eq_null_safe(lit(1));
    /// ```
    #[must_use]
    pub fn eq_null_safe(self, other: impl Into<Self>) -> Self {
        Self::call(Builtin::EqNullSafe, vec![self, other.into()])
    }

    /// `self < other`.
    ///
    /// ```
    /// # use krishiv_session::functions::{col, lit};
    /// let c = col("a").lt(lit(1));
    /// ```
    #[must_use]
    pub fn lt(self, other: impl Into<Self>) -> Self {
        Self::call(Builtin::Lt, vec![self, other.into()])
    }

    /// `self <= other`.
    ///
    /// ```
    /// # use krishiv_session::functions::{col, lit};
    /// let c = col("a").lt_eq(lit(1));
    /// ```
    #[must_use]
    pub fn lt_eq(self, other: impl Into<Self>) -> Self {
        Self::call(Builtin::LtEq, vec![self, other.into()])
    }

    /// `self > other`.
    ///
    /// ```
    /// # use krishiv_session::functions::{col, lit};
    /// let c = col("a").gt(lit(1));
    /// ```
    #[must_use]
    pub fn gt(self, other: impl Into<Self>) -> Self {
        Self::call(Builtin::Gt, vec![self, other.into()])
    }

    /// `self >= other`.
    ///
    /// ```
    /// # use krishiv_session::functions::{col, lit};
    /// let c = col("a").gt_eq(lit(1));
    /// ```
    #[must_use]
    pub fn gt_eq(self, other: impl Into<Self>) -> Self {
        Self::call(Builtin::GtEq, vec![self, other.into()])
    }

    /// `self AND other`.
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// let c = col("a").and(col("b"));
    /// ```
    #[must_use]
    pub fn and(self, other: impl Into<Self>) -> Self {
        Self::call(Builtin::And, vec![self, other.into()])
    }

    /// `self OR other`.
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// let c = col("a").or(col("b"));
    /// ```
    #[must_use]
    pub fn or(self, other: impl Into<Self>) -> Self {
        Self::call(Builtin::Or, vec![self, other.into()])
    }

    /// `self IS NULL`.
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// let c = col("a").is_null();
    /// ```
    #[must_use]
    pub fn is_null(self) -> Self {
        Self::call(Builtin::IsNull, vec![self])
    }

    /// `self IS NOT NULL`.
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// let c = col("a").is_not_null();
    /// ```
    #[must_use]
    pub fn is_not_null(self) -> Self {
        Self::call(Builtin::IsNotNull, vec![self])
    }

    /// `self LIKE pattern`.
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// let c = col("name").like("a%");
    /// ```
    #[must_use]
    pub fn like(self, pattern: &str) -> Self {
        Self::call(Builtin::Like, vec![self, lit_str(pattern)])
    }

    /// `self IN (values)`.
    ///
    /// ```
    /// # use krishiv_session::functions::{col, lit};
    /// let c = col("a").is_in([lit(1), lit(2)]);
    /// ```
    #[must_use]
    pub fn is_in(self, values: impl IntoIterator<Item = Self>) -> Self {
        let mut args = vec![self];
        args.extend(values);
        Self::call(Builtin::In, args)
    }

    /// `self BETWEEN low AND high`.
    ///
    /// ```
    /// # use krishiv_session::functions::{col, lit};
    /// let c = col("a").between(lit(1), lit(9));
    /// ```
    #[must_use]
    pub fn between(self, low: impl Into<Self>, high: impl Into<Self>) -> Self {
        Self::call(Builtin::Between, vec![self, low.into(), high.into()])
    }

    /// The field `name` of this struct (`self.name` in SQL reads the same
    /// node), as PySpark's `Column.getField`.
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// let c = col("s").get_field("a");
    /// ```
    #[must_use]
    pub fn get_field(self, name: &str) -> Self {
        Self::new(Expr::new(ExprNode::Field {
            base: Arc::new(self.expr),
            name: name.to_owned(),
        }))
    }

    /// The struct with the field `name` added or replaced by `value`, as
    /// PySpark's `Column.withField`: a match (per
    /// `spark.sql.caseSensitive`) is replaced in place, else the field is
    /// appended last. A dotted name is a path into nested structs (Spark
    /// Connect reads it so): `withField("a.b", v)` updates `b` inside `a`,
    /// a NULL `a` stays NULL, and a missing `a` is `FIELD_NOT_FOUND`.
    ///
    /// ```
    /// # use krishiv_session::functions::{col, lit};
    /// let c = col("s").with_field("a", lit(1));
    /// ```
    #[must_use]
    pub fn with_field(self, name: &str, value: Self) -> Self {
        let parts: Vec<&str> = name.split('.').collect();
        Self::new(with_path(self.expr, &parts, value.expr))
    }

    /// The struct without the named fields, as PySpark's
    /// `Column.dropFields`: every match (per `spark.sql.caseSensitive`) is
    /// removed, names that match nothing are ignored. A dotted name is a
    /// path into nested structs, applied one name after the other.
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// let c = col("s").drop_fields(vec!["a".to_owned()]);
    /// ```
    #[must_use]
    pub fn drop_fields(self, names: Vec<String>) -> Self {
        if names.iter().all(|n| !n.contains('.')) {
            return Self::new(Expr::new(ExprNode::UpdateFields {
                base: Arc::new(self.expr),
                ops: names
                    .into_iter()
                    .map(|name| FieldOp::Drop { name })
                    .collect(),
            }));
        }
        let expr = names.iter().fold(self.expr, |acc, name| {
            let parts: Vec<&str> = name.split('.').collect();
            drop_path(acc, &parts)
        });
        Self::new(expr)
    }

    /// Ascending, nulls first (Spark's default), for `order_by`.
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// let c = col("a").asc();
    /// ```
    #[must_use]
    pub fn asc(self) -> Self {
        self.ordered(SortDirection::Ascending, NullOrdering::NullsFirst)
    }

    /// Ascending, nulls last, for `order_by`.
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// let c = col("a").asc_nulls_last();
    /// ```
    #[must_use]
    pub fn asc_nulls_last(self) -> Self {
        self.ordered(SortDirection::Ascending, NullOrdering::NullsLast)
    }

    /// Descending, nulls last (Spark's default), for `order_by`.
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// let c = col("a").desc();
    /// ```
    #[must_use]
    pub fn desc(self) -> Self {
        self.ordered(SortDirection::Descending, NullOrdering::NullsLast)
    }

    /// Descending, nulls first, for `order_by`.
    ///
    /// ```
    /// # use krishiv_session::functions::col;
    /// let c = col("a").desc_nulls_first();
    /// ```
    #[must_use]
    pub fn desc_nulls_first(self) -> Self {
        self.ordered(SortDirection::Descending, NullOrdering::NullsFirst)
    }

    fn ordered(mut self, direction: SortDirection, nulls: NullOrdering) -> Self {
        self.order = Some((direction, nulls));
        self
    }

    /// The sort key: ascending unless `asc`/`desc` said otherwise.
    pub(crate) fn sort_order(self) -> SortOrder {
        let (direction, nulls) = self
            .order
            .unwrap_or((SortDirection::Ascending, NullOrdering::NullsFirst));
        SortOrder {
            expr: Arc::new(self.expr),
            direction,
            nulls,
        }
    }

    /// The expression of an aggregate call `name(args)`. Dots separate
    /// name parts outside backquotes, as in SQL (`system.session.f`).
    pub(crate) fn function(name: &str, args: Vec<Self>, distinct: bool) -> Self {
        let mut parts = split_name(name);
        let func = parts.pop().unwrap_or_default();
        Self::new(Expr::new(ExprNode::Function(FunctionCall {
            name: func,
            qualifier: parts,
            arg_names: Vec::new(),
            args: args.into_iter().map(|c| Arc::new(c.expr)).collect(),
            distinct,
            filter: None,
        })))
    }
}

fn lit_str(s: &str) -> Column {
    Column::new(Literal::String(s.to_owned()).into())
}

/// The parts of a column name: dots separate parts outside backquotes, and
/// a doubled backquote inside them is one.
fn split_name(name: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut part = String::new();
    let mut quoted = false;
    let mut chars = name.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '`' if quoted && chars.peek() == Some(&'`') => {
                chars.next();
                part.push('`');
            }
            '`' => quoted = !quoted,
            '.' if !quoted => parts.push(std::mem::take(&mut part)),
            c => part.push(c),
        }
    }
    parts.push(part);
    parts
}

impl From<&str> for Column {
    /// The column named so (see [`crate::functions::col`]).
    fn from(name: &str) -> Self {
        Self::named(name, None)
    }
}

macro_rules! binary_op {
    ($trait:ident, $method:ident, $op:ident) => {
        impl<T: Into<Column>> ops::$trait<T> for Column {
            type Output = Column;
            fn $method(self, rhs: T) -> Column {
                Column::call(Builtin::$op, vec![self, rhs.into()])
            }
        }
    };
}

binary_op!(Add, add, Add);
binary_op!(Sub, sub, Subtract);
binary_op!(Mul, mul, Multiply);
binary_op!(Div, div, Divide);
binary_op!(Rem, rem, Remainder);

impl ops::Neg for Column {
    type Output = Column;
    fn neg(self) -> Column {
        Column::call(Builtin::Negative, vec![self])
    }
}

impl ops::Not for Column {
    type Output = Column;
    fn not(self) -> Column {
        Column::call(Builtin::Not, vec![self])
    }
}

/// `base` with the field at `parts` set to `value`: one update for a plain
/// name, else an update of the first part whose value is the nested update
/// of that field.
fn with_path(base: Expr, parts: &[&str], value: Expr) -> Expr {
    let (first, rest) = parts.split_first().map_or(("", &[][..]), |(f, r)| (*f, r));
    let new_value = if rest.is_empty() {
        value
    } else {
        let inner = Expr::new(ExprNode::Field {
            base: Arc::new(base.clone()),
            name: first.to_owned(),
        });
        with_path(inner, rest, value)
    };
    Expr::new(ExprNode::UpdateFields {
        base: Arc::new(base),
        ops: vec![FieldOp::With {
            name: first.to_owned(),
            value: Arc::new(new_value),
        }],
    })
}

/// `base` without the field at `parts` (a nested path updates its parent).
fn drop_path(base: Expr, parts: &[&str]) -> Expr {
    let (first, rest) = parts.split_first().map_or(("", &[][..]), |(f, r)| (*f, r));
    if rest.is_empty() {
        return Expr::new(ExprNode::UpdateFields {
            base: Arc::new(base),
            ops: vec![FieldOp::Drop {
                name: first.to_owned(),
            }],
        });
    }
    let inner = Expr::new(ExprNode::Field {
        base: Arc::new(base.clone()),
        name: first.to_owned(),
    });
    Expr::new(ExprNode::UpdateFields {
        base: Arc::new(base),
        ops: vec![FieldOp::With {
            name: first.to_owned(),
            value: Arc::new(drop_path(inner, rest)),
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_split_like_spark() {
        assert_eq!(split_name("a"), ["a"]);
        assert_eq!(split_name("t.a"), ["t", "a"]);
        assert_eq!(split_name("`a.b`.c"), ["a.b", "c"]);
        assert_eq!(split_name("`a``b`"), ["a`b"]);
        let star = Column::named("t.*", None);
        assert!(matches!(&star.expr.node, ExprNode::Star(s) if s.qualifier == ["t"]));
        let all = Column::named("*", Some(PlanId(3)));
        assert!(
            matches!(&all.expr.node, ExprNode::Star(s) if s.qualifier.is_empty() && s.plan_id == Some(PlanId(3)))
        );
    }

    #[test]
    fn try_cast_uses_try_mode() {
        let c = Column::from("a").try_cast(DataType::Long);
        assert!(matches!(
            &c.expr.node,
            ExprNode::Cast {
                mode: CastMode::Try,
                ..
            }
        ));
        let c = Column::from("a").cast(DataType::Long);
        assert!(matches!(
            &c.expr.node,
            ExprNode::Cast {
                mode: CastMode::Default,
                ..
            }
        ));
    }

    #[test]
    fn orders_default_like_spark() {
        let c = Column::from("a");
        assert_eq!(
            c.clone().sort_order(),
            SortOrder::asc(Arc::new(c.clone().expr))
        );
        assert_eq!(
            c.clone().desc().sort_order(),
            SortOrder::desc(Arc::new(c.clone().expr))
        );
        let o = c.clone().asc_nulls_last().sort_order();
        assert_eq!(
            (o.direction, o.nulls),
            (SortDirection::Ascending, NullOrdering::NullsLast)
        );
        let o = c.desc_nulls_first().sort_order();
        assert_eq!(
            (o.direction, o.nulls),
            (SortDirection::Descending, NullOrdering::NullsFirst)
        );
    }
}
