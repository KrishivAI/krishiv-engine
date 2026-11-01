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

/// Parses `text` as a SQL expression (`F.expr`) with the default parser
/// options and lowers it as `Session::sql` lowers `SELECT` items, so it
/// resolves against a DataFrame's columns in the analyzer (`*` stays a
/// star, as in a `SELECT` list; an alias stays, as in Spark's
/// `parseExpression`).
///
/// ```
/// use krishiv_session::functions::expr;
/// let total = expr("price * qty").unwrap();
/// let cheap = expr("price < 10").unwrap();
/// ```
///
/// # Errors
/// Spark's parse errors (`PARSE_SYNTAX_ERROR`, ...); lowering errors for
/// constructs the IR lacks (`KRISHIV_UNSUPPORTED_FEATURE`).
pub fn expr(text: &str) -> Result<Column, krishiv_common::error::KrishivError> {
    let lowered = krishiv_sql::parse_and_lower_named_expression(
        text,
        &krishiv_sql::ParserOptions::default(),
        krishiv_plan::SourceId(0),
    )
    .map_err(|e| e.error)?;
    Ok(Column::new((*lowered).clone()))
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

/// A `MAP<STRING, STRING>` literal, from its entries in order: the JSON
/// options of one call, as the Python wrappers build them.
///
/// ```
/// use krishiv_session::functions::map_literal;
/// let options = map_literal(vec![("mode".to_owned(), "FAILFAST".to_owned())]);
/// let shown = format!("{options:?}");
/// assert!(shown.contains("Map"), "{shown}");
/// assert!(shown.contains("mode"), "{shown}");
/// assert!(shown.contains("FAILFAST"), "{shown}");
/// ```
#[must_use]
pub fn map_literal(pairs: Vec<(String, String)>) -> Column {
    Column::new(Expr::literal(Literal::Map(pairs)))
}

/// The untyped `NULL` literal (`lit(None)` in PySpark; Spark's `void` type).
///
/// ```
/// use krishiv_session::functions::lit_null;
/// let nothing = lit_null();
/// ```
#[must_use]
pub fn lit_null() -> Column {
    Column::new(Expr::literal(Literal::Null(None)))
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
        replace: Vec::new(),
        position: None,
        qualifier: Vec::new(),
        except: Vec::new(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lit_null_is_the_untyped_null_literal() {
        let null = lit_null();
        assert_eq!(
            format!("{null:?}"),
            format!("{:?}", Column::new(Expr::literal(Literal::Null(None))))
        );
    }
}
