// SPDX-License-Identifier: Apache-2.0
//! The filter of an overwrite by filter (`replaceWhere`, `INSERT OVERWRITE ... PARTITION`) as a
//! kernel predicate over the partition columns (design note P3-T021 section 4.1, "Overwrite").
//!
//! Only partition columns may be named: the files a partition predicate selects are exactly
//! the files to remove, and each written partition is checked against it (delta-spark's
//! `replaceWhere` constraint check, `DELTA_REPLACE_WHERE_MISMATCH`). A filter on data columns
//! (delta-spark rewrites the matching files' other rows) is refused by name. The forms
//! translated: `=`, `==`, `<=>`, `<`, `<=`, `>`, `>=`, `!=`, `<>`, `AND`, `OR`, `NOT`, `IS [NOT]
//! NULL`, `IN` and literals; a literal is taken in its column's type when it is exactly
//! representable there.

use delta_kernel::expressions::{Expression as KExpr, Predicate as Pred, Scalar};
use delta_kernel::schema::{DataType, PrimitiveType, StructType};
use krishiv_plan::{Expr, ExprNode, Literal};

/// Why a filter does not translate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterRefusal {
    /// It names a column that is not a partition column (the column).
    DataColumn(String),
    /// It uses a form outside the list (its description).
    Unsupported(String),
}

/// `filter` as a predicate over the partition columns `partitions` of the table `schema`.
///
/// # Errors
/// [`FilterRefusal`] when it names a data column or uses an untranslated form.
pub fn partition_predicate(
    filter: &Expr,
    schema: &StructType,
    partitions: &[String],
) -> Result<Pred, FilterRefusal> {
    Translate { schema, partitions }.pred(filter)
}

struct Translate<'a> {
    schema: &'a StructType,
    partitions: &'a [String],
}

impl Translate<'_> {
    fn pred(&self, e: &Expr) -> Result<Pred, FilterRefusal> {
        match &e.node {
            ExprNode::Literal(Literal::Boolean(b)) => Ok(Pred::literal(*b)),
            ExprNode::Function(f) if f.qualifier.is_empty() => {
                let name = f.name.to_ascii_lowercase();
                let args = &f.args;
                match (name.as_str(), args.as_slice()) {
                    ("and", [a, b]) => Ok(Pred::and(self.pred(a)?, self.pred(b)?)),
                    ("or", [a, b]) => Ok(Pred::or(self.pred(a)?, self.pred(b)?)),
                    ("not" | "!", [a]) => Ok(Pred::not(self.pred(a)?)),
                    ("isnull", [a]) => Ok(Pred::is_null(self.column(a)?.0)),
                    ("isnotnull", [a]) => Ok(Pred::is_not_null(self.column(a)?.0)),
                    ("in", [a, rest @ ..]) if !rest.is_empty() => {
                        let (c, t) = self.column(a)?;
                        let eqs = rest
                            .iter()
                            .map(|v| Ok(Pred::eq(c.clone(), self.literal(v, &t)?)))
                            .collect::<Result<Vec<_>, FilterRefusal>>()?;
                        Ok(Pred::or_from(eqs))
                    }
                    (op, [a, b]) => self.comparison(op, a, b),
                    _ => Err(FilterRefusal::Unsupported(format!(
                        "the function {}",
                        f.name
                    ))),
                }
            }
            _ => Err(FilterRefusal::Unsupported(
                "an expression that is not a comparison of a partition column with a literal"
                    .to_owned(),
            )),
        }
    }

    fn comparison(&self, op: &str, a: &Expr, b: &Expr) -> Result<Pred, FilterRefusal> {
        // `column op literal` or `literal op column` (flipped).
        let (column, literal, flipped) = match (self.column(a), self.column(b)) {
            (Ok(c), _) => (c, b, false),
            (Err(FilterRefusal::DataColumn(c)), _) => return Err(FilterRefusal::DataColumn(c)),
            (_, Ok(c)) => (c, a, true),
            (_, Err(e)) => return Err(e),
        };
        let (c, t) = column;
        let v = self.literal(literal, &t)?;
        let (l, r) = if flipped { (v, c) } else { (c, v) };
        Ok(match op {
            "=" | "==" => Pred::eq(l, r),
            "<=>" => Pred::not(Pred::distinct(l, r)),
            "!=" | "<>" => Pred::ne(l, r),
            "<" => Pred::lt(l, r),
            "<=" => Pred::le(l, r),
            ">" => Pred::gt(l, r),
            ">=" => Pred::ge(l, r),
            other => return Err(FilterRefusal::Unsupported(format!("the operator {other}"))),
        })
    }

    /// A partition column reference and its type.
    fn column(&self, e: &Expr) -> Result<(KExpr, DataType), FilterRefusal> {
        let ExprNode::Column(c) = &e.node else {
            return Err(FilterRefusal::Unsupported(
                "a non-column operand".to_owned(),
            ));
        };
        let Some(name) = c.name.last() else {
            return Err(FilterRefusal::Unsupported(
                "an empty column name".to_owned(),
            ));
        };
        let Some(p) = self
            .partitions
            .iter()
            .find(|p| p.eq_ignore_ascii_case(name))
        else {
            return Err(FilterRefusal::DataColumn(name.clone()));
        };
        let f = self
            .schema
            .field(p)
            .ok_or_else(|| FilterRefusal::DataColumn(name.clone()))?;
        Ok((KExpr::column([p.as_str()]), f.data_type().clone()))
    }

    fn literal(&self, e: &Expr, t: &DataType) -> Result<KExpr, FilterRefusal> {
        let ExprNode::Literal(lit) = &e.node else {
            return Err(FilterRefusal::Unsupported(
                "a non-literal operand".to_owned(),
            ));
        };
        let refuse = || FilterRefusal::Unsupported(format!("the literal {lit:?} for a {t} column"));
        let DataType::Primitive(p) = t else {
            return Err(refuse());
        };
        let int = |v: i64| -> Option<Scalar> {
            Some(match p {
                PrimitiveType::Byte => Scalar::Byte(i8::try_from(v).ok()?),
                PrimitiveType::Short => Scalar::Short(i16::try_from(v).ok()?),
                PrimitiveType::Integer => Scalar::Integer(i32::try_from(v).ok()?),
                PrimitiveType::Long => Scalar::Long(v),
                _ => return None,
            })
        };
        let s = match lit {
            Literal::Null(_) => Some(Scalar::Null(t.clone())),
            Literal::Byte(v) => int(i64::from(*v)),
            Literal::Short(v) => int(i64::from(*v)),
            Literal::Integer(v) => int(i64::from(*v)),
            Literal::Long(v) => int(*v),
            Literal::Boolean(v) if *p == PrimitiveType::Boolean => Some(Scalar::Boolean(*v)),
            Literal::Date(v) if *p == PrimitiveType::Date => Some(Scalar::Date(*v)),
            Literal::Timestamp(v) if *p == PrimitiveType::Timestamp => Some(Scalar::Timestamp(*v)),
            Literal::TimestampNtz(v) if *p == PrimitiveType::TimestampNtz => {
                Some(Scalar::TimestampNtz(*v))
            }
            Literal::String(v) => p.parse_scalar(v).ok(),
            _ => None,
        };
        s.map(KExpr::literal).ok_or_else(refuse)
    }
}
