// SPDX-License-Identifier: Apache-2.0
//! Column builders, as PySpark's `pyspark.sql.functions`. Functions resolve
//! by name in the session's function registry when a query is analyzed.

use krishiv_plan::{Expr, ExprNode, Literal, Star};

use crate::Column;

/// The column named `name` (`a`, `t.a`, `` `a.b` ``, `*`, `t.*`).
///
/// ```
/// use krishiv_session::functions::col;
/// let a = col("a");
/// let all = col("*");
/// ```
#[must_use]
pub fn col(name: &str) -> Column {
    Column::named(name, None)
}

/// A literal value.
///
/// ```
/// use krishiv_session::functions::lit;
/// let (one, pi, yes, text) = (lit(1), lit(3.14), lit(true), lit("x"));
/// ```
#[must_use]
pub fn lit(value: impl IntoLiteral) -> Column {
    Column::new(Expr::literal(value.into_literal()))
}

/// Values [`lit`] accepts, with the Spark type each becomes.
pub trait IntoLiteral {
    /// The literal.
    fn into_literal(self) -> Literal;
}

macro_rules! into_literal {
    ($($t:ty => $variant:ident),* $(,)?) => {
        $(impl IntoLiteral for $t {
            fn into_literal(self) -> Literal {
                Literal::$variant(self)
            }
        })*
    };
}

into_literal!(
    bool => Boolean,
    i8 => Byte,
    i16 => Short,
    i32 => Integer,
    i64 => Long,
    f32 => Float,
    f64 => Double,
    String => String,
);

impl IntoLiteral for &str {
    fn into_literal(self) -> Literal {
        Literal::String(self.to_owned())
    }
}

/// Numbers and booleans convert into a [`Column`] directly
/// (`col("a") + 1`); a `&str` converts into a column name instead.
macro_rules! into_column {
    ($($t:ty),*) => {
        $(impl From<$t> for Column {
            fn from(value: $t) -> Self {
                lit(value)
            }
        })*
    };
}

into_column!(bool, i8, i16, i32, i64, f32, f64);

/// A call of the function `name` (Spark's `call_function`).
///
/// ```
/// use krishiv_session::functions::{call_function, col};
/// let c = call_function("upper", [col("name")]);
/// ```
#[must_use]
pub fn call_function(name: &str, args: impl IntoIterator<Item = Column>) -> Column {
    Column::function(name, args.into_iter().collect(), false)
}

/// `sum(c)`.
///
/// ```
/// use krishiv_session::functions::{col, sum};
/// let c = sum(col("x"));
/// ```
#[must_use]
pub fn sum(c: Column) -> Column {
    call_function("sum", [c])
}

/// `avg(c)`.
///
/// ```
/// use krishiv_session::functions::{avg, col};
/// let c = avg(col("x"));
/// ```
#[must_use]
pub fn avg(c: Column) -> Column {
    call_function("avg", [c])
}

/// `min(c)`.
///
/// ```
/// use krishiv_session::functions::{col, min};
/// let c = min(col("x"));
/// ```
#[must_use]
pub fn min(c: Column) -> Column {
    call_function("min", [c])
}

/// `max(c)`.
///
/// ```
/// use krishiv_session::functions::{col, max};
/// let c = max(col("x"));
/// ```
#[must_use]
pub fn max(c: Column) -> Column {
    call_function("max", [c])
}

/// `count(c)`; `count(col("*"))` counts rows.
///
/// ```
/// use krishiv_session::functions::{col, count};
/// let rows = count(col("*"));
/// ```
#[must_use]
pub fn count(c: Column) -> Column {
    call_function("count", [c])
}

/// `count(*)`.
///
/// ```
/// use krishiv_session::functions::count_star;
/// let rows = count_star();
/// ```
#[must_use]
pub fn count_star() -> Column {
    let star = Column::new(Expr::new(ExprNode::Star(Star {
        qualifier: Vec::new(),
        plan_id: None,
    })));
    call_function("count", [star])
}

/// `count(DISTINCT cols)`.
///
/// ```
/// use krishiv_session::functions::{col, count_distinct};
/// let c = count_distinct([col("a"), col("b")]);
/// ```
#[must_use]
pub fn count_distinct(cols: impl IntoIterator<Item = Column>) -> Column {
    Column::function("count", cols.into_iter().collect(), true)
}

/// `coalesce(cols)`: the first non-null value.
///
/// ```
/// use krishiv_session::functions::{coalesce, col, lit};
/// let c = coalesce([col("a"), lit(0)]);
/// ```
#[must_use]
pub fn coalesce(cols: impl IntoIterator<Item = Column>) -> Column {
    call_function("coalesce", cols)
}
