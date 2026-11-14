// SPDX-License-Identifier: Apache-2.0
//! Krishiv's pushed filters as the kernel's `Predicate` (design note P3-T021 section 3.4): a
//! conservative translation the kernel prunes files with (partition values and statistics).
//!
//! Only comparisons, `IS [NOT] NULL` and `IN` (as an OR of equalities) on columns with a literal
//! of exactly the column's type, combined by AND, OR and NOT, translate. Floating-point columns
//! never do (their NaN ordering is Spark's, not the statistics'), nor do non-binary collations.
//! A conjunct that does not translate is left out of an AND, which only keeps more files; an OR
//! or a NOT needs every part translated exactly. The filter above the scan always stays, so a
//! translation only ever prunes.

use delta_kernel::expressions::{Expression as Expr, Predicate as Pred, Scalar};
use delta_kernel::schema::{DataType, PrimitiveType, StructType};
use krishiv_io::{CmpOp, Collation, ColumnPath, PushdownExpr, ScalarValue};

/// `expr` as a kernel predicate over the table `schema` (logical names), keeping a superset of
/// its matching rows; `None` when nothing of it translates.
#[must_use]
pub fn to_kernel(expr: &PushdownExpr, schema: &StructType) -> Option<Pred> {
    translate(expr, schema).map(|(p, _)| p)
}

/// The translation and whether it is exact (not a superset).
fn translate(expr: &PushdownExpr, schema: &StructType) -> Option<(Pred, bool)> {
    match expr {
        PushdownExpr::Cmp {
            column,
            op,
            value,
            collation,
        } => {
            binary(collation)?;
            let (c, v) = operands(column, value, schema)?;
            Some((
                match op {
                    CmpOp::Eq => Pred::eq(c, v),
                    CmpOp::NotEq => Pred::ne(c, v),
                    CmpOp::Lt => Pred::lt(c, v),
                    CmpOp::LtEq => Pred::le(c, v),
                    CmpOp::Gt => Pred::gt(c, v),
                    CmpOp::GtEq => Pred::ge(c, v),
                },
                true,
            ))
        }
        PushdownExpr::In {
            column,
            values,
            collation,
        } => {
            binary(collation)?;
            if values.is_empty() {
                return None;
            }
            let eqs = values
                .iter()
                .map(|v| operands(column, v, schema).map(|(c, v)| Pred::eq(c, v)))
                .collect::<Option<Vec<_>>>()?;
            Some((Pred::or_from(eqs), true))
        }
        PushdownExpr::IsNull(column) => {
            column_type(column, schema)?;
            Some((Pred::is_null(column_expr(column)), true))
        }
        PushdownExpr::IsNotNull(column) => {
            column_type(column, schema)?;
            Some((Pred::is_not_null(column_expr(column)), true))
        }
        PushdownExpr::And(parts) => {
            let translated: Vec<(Pred, bool)> =
                parts.iter().filter_map(|p| translate(p, schema)).collect();
            let exact = translated.len() == parts.len() && translated.iter().all(|(_, e)| *e);
            match translated.len() {
                0 => None,
                _ => Some((
                    Pred::and_from(translated.into_iter().map(|(p, _)| p)),
                    exact,
                )),
            }
        }
        PushdownExpr::Or(parts) => {
            if parts.is_empty() {
                return None;
            }
            let translated = parts
                .iter()
                .map(|p| translate(p, schema))
                .collect::<Option<Vec<_>>>()?;
            let exact = translated.iter().all(|(_, e)| *e);
            Some((Pred::or_from(translated.into_iter().map(|(p, _)| p)), exact))
        }
        PushdownExpr::Not(inner) => match translate(inner, schema)? {
            (p, true) => Some((Pred::not(p), true)),
            (_, false) => None,
        },
        // Statistics cannot answer these, or they are runtime-only.
        _ => None,
    }
}

fn binary(collation: &Collation) -> Option<()> {
    matches!(collation, Collation::Utf8Binary).then_some(())
}

fn column_expr(column: &ColumnPath) -> Expr {
    Expr::column(column.0.iter().map(String::as_str))
}

/// The type of the (possibly nested) column `column` of `schema`.
fn column_type<'a>(column: &ColumnPath, schema: &'a StructType) -> Option<&'a DataType> {
    let (first, rest) = column.0.split_first()?;
    let mut t = schema.field(first)?.data_type();
    for name in rest {
        let DataType::Struct(s) = t else {
            return None;
        };
        t = s.field(name)?.data_type();
    }
    Some(t)
}

/// The column and the literal of `column op value`, when the literal has exactly the column's
/// type.
fn operands(column: &ColumnPath, value: &ScalarValue, schema: &StructType) -> Option<(Expr, Expr)> {
    let DataType::Primitive(p) = column_type(column, schema)? else {
        return None;
    };
    let scalar = match (p, value) {
        (PrimitiveType::Boolean, ScalarValue::Boolean(v)) => Scalar::Boolean(*v),
        (PrimitiveType::Byte, ScalarValue::Int8(v)) => Scalar::Byte(*v),
        (PrimitiveType::Short, ScalarValue::Int16(v)) => Scalar::Short(*v),
        (PrimitiveType::Integer, ScalarValue::Int32(v)) => Scalar::Integer(*v),
        (PrimitiveType::Long, ScalarValue::Int64(v)) => Scalar::Long(*v),
        (PrimitiveType::String, ScalarValue::Utf8(v)) => Scalar::String(v.clone()),
        (PrimitiveType::Date, ScalarValue::Date32(v)) => Scalar::Date(*v),
        (PrimitiveType::Timestamp, ScalarValue::TimestampMicros(v)) => Scalar::Timestamp(*v),
        (PrimitiveType::TimestampNtz, ScalarValue::TimestampMicros(v)) => Scalar::TimestampNtz(*v),
        (
            PrimitiveType::Decimal(d),
            ScalarValue::Decimal128 {
                unscaled,
                precision,
                scale,
            },
        ) if d.precision() == *precision && i16::from(d.scale()) == i16::from(*scale) => {
            Scalar::decimal(*unscaled, d.precision(), d.scale()).ok()?
        }
        _ => return None,
    };
    Some((column_expr(column), Expr::literal(scalar)))
}
