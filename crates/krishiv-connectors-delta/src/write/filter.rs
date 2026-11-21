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
//!
//! A column is a one-part name, or a two-part name whose first part is one of the caller's
//! qualifiers (a row-level statement's alias or table name) and no top-level column: `s.p` with a
//! struct column `s` is the struct's field, never the partition column `p`. A string literal
//! compared with a `TIMESTAMP` column is refused: Spark casts it in the session time zone, which
//! the translation does not know.

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

/// `filter` as a predicate over the partition columns `partitions` of the table `schema`; a
/// two-part column name may start with one of `qualifiers` (see the module docs).
///
/// # Errors
/// [`FilterRefusal`] when it names a data column or uses an untranslated form.
pub fn partition_predicate(
    filter: &Expr,
    schema: &StructType,
    partitions: &[String],
    qualifiers: &[&str],
) -> Result<Pred, FilterRefusal> {
    Translate {
        schema,
        partitions,
        qualifiers,
    }
    .pred(filter)
}

/// The top-level `AND` conjuncts of `filter` that translate by [`partition_predicate`], as one
/// predicate (`None`: none does). Rows the filter holds for satisfy it, so it bounds what a
/// statement with that condition read.
#[must_use]
pub fn partition_conjuncts(
    filter: &Expr,
    schema: &StructType,
    partitions: &[String],
    qualifiers: &[&str],
) -> Option<Pred> {
    fn split<'e>(e: &'e Expr, out: &mut Vec<&'e Expr>) {
        match &e.node {
            ExprNode::Function(f)
                if f.qualifier.is_empty()
                    && f.name.eq_ignore_ascii_case("and")
                    && f.args.len() == 2 =>
            {
                for a in &f.args {
                    split(a, out);
                }
            }
            _ => out.push(e),
        }
    }
    let mut conjuncts = Vec::new();
    split(filter, &mut conjuncts);
    let t = Translate {
        schema,
        partitions,
        qualifiers,
    };
    let mut kept: Vec<Pred> = conjuncts
        .into_iter()
        .filter_map(|c| t.pred(c).ok())
        .collect();
    match kept.len() {
        0 | 1 => kept.pop(),
        _ => Some(Pred::and_from(kept)),
    }
}

struct Translate<'a> {
    schema: &'a StructType,
    partitions: &'a [String],
    qualifiers: &'a [&'a str],
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
        let name = match c.name.as_slice() {
            [name] => name,
            [q, name]
                if self.qualifiers.iter().any(|x| x.eq_ignore_ascii_case(q))
                    && !self
                        .schema
                        .fields()
                        .any(|f| f.name().eq_ignore_ascii_case(q)) =>
            {
                name
            }
            [] => {
                return Err(FilterRefusal::Unsupported(
                    "an empty column name".to_owned(),
                ));
            }
            parts => return Err(FilterRefusal::DataColumn(parts.join("."))),
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
            // Spark casts the string in the session time zone.
            Literal::String(_) if *p == PrimitiveType::Timestamp => None,
            Literal::String(v) => p.parse_scalar(v).ok(),
            _ => None,
        };
        s.map(KExpr::literal).ok_or_else(refuse)
    }
}

#[cfg(test)]
mod tests {
    use delta_kernel::schema::{DataType, StructField, StructType};
    use krishiv_plan::{Expr, Literal};

    use super::{FilterRefusal, partition_conjuncts, partition_predicate};

    /// `id BIGINT, s STRUCT<p INT>, p INT, ts TIMESTAMP, ntz TIMESTAMP_NTZ`, partitioned by
    /// `p`, `ts` and `ntz`.
    fn schema() -> (StructType, Vec<String>) {
        let s = StructType::new_unchecked([StructField::nullable("p", DataType::INTEGER)]);
        (
            StructType::new_unchecked([
                StructField::nullable("id", DataType::LONG),
                StructField::nullable("s", s),
                StructField::nullable("p", DataType::INTEGER),
                StructField::nullable("ts", DataType::TIMESTAMP),
                StructField::nullable("ntz", DataType::TIMESTAMP_NTZ),
            ]),
            vec!["p".to_owned(), "ts".to_owned(), "ntz".to_owned()],
        )
    }

    fn eq(column: &[&str], value: Literal) -> Expr {
        Expr::function(
            "=",
            vec![Expr::column(column.iter().copied()), Expr::literal(value)],
        )
    }

    fn translate(e: &Expr, qualifiers: &[&str]) -> Result<(), FilterRefusal> {
        let (schema, partitions) = schema();
        partition_predicate(e, &schema, &partitions, qualifiers).map(drop)
    }

    #[test]
    fn a_struct_field_is_never_the_partition_column_of_its_last_name() {
        assert_eq!(
            translate(&eq(&["s", "p"], Literal::Integer(1)), &["t"]),
            Err(FilterRefusal::DataColumn("s.p".to_owned()))
        );
        // The statement's alias names the partition column.
        assert_eq!(
            translate(&eq(&["t", "p"], Literal::Integer(1)), &["t"]),
            Ok(())
        );
        assert_eq!(
            translate(&eq(&["T", "p"], Literal::Integer(1)), &["t"]),
            Ok(())
        );
        // An alias that is also a column's name is ambiguous: refused.
        assert!(translate(&eq(&["s", "p"], Literal::Integer(1)), &["s"]).is_err());
        // No qualifier given: only one-part names.
        assert!(translate(&eq(&["t", "p"], Literal::Integer(1)), &[]).is_err());
        assert!(translate(&eq(&["db", "t", "p"], Literal::Integer(1)), &["t"]).is_err());
        assert_eq!(translate(&eq(&["p"], Literal::Integer(1)), &[]), Ok(()));
    }

    #[test]
    fn a_string_compared_with_a_timestamp_partition_is_refused() {
        let text = || Literal::String("2024-01-01 00:00:00".to_owned());
        assert!(matches!(
            translate(&eq(&["ts"], text()), &[]),
            Err(FilterRefusal::Unsupported(_))
        ));
        // A wall-clock `TIMESTAMP_NTZ` reads no zone.
        assert_eq!(translate(&eq(&["ntz"], text()), &[]), Ok(()));
    }

    #[test]
    fn the_partition_conjuncts_of_a_condition_are_kept() {
        let (schema, partitions) = schema();
        let both = Expr::function(
            "and",
            vec![
                eq(&["p"], Literal::Integer(1)),
                eq(&["id"], Literal::Long(3)),
            ],
        );
        let p = partition_conjuncts(&both, &schema, &partitions, &[]).unwrap();
        assert_eq!(
            p.to_string(),
            partition_predicate(&eq(&["p"], Literal::Integer(1)), &schema, &partitions, &[])
                .unwrap()
                .to_string()
        );
        assert!(
            partition_conjuncts(&eq(&["id"], Literal::Long(3)), &schema, &partitions, &[])
                .is_none()
        );
        // An `OR` with a data column bounds nothing.
        let or = Expr::function(
            "or",
            vec![
                eq(&["p"], Literal::Integer(1)),
                eq(&["id"], Literal::Long(3)),
            ],
        );
        assert!(partition_conjuncts(&or, &schema, &partitions, &[]).is_none());
        assert!(
            partition_conjuncts(
                &eq(&["s", "p"], Literal::Integer(1)),
                &schema,
                &partitions,
                &["t"]
            )
            .is_none()
        );
    }

    /// The translation of every operator and literal type (one-part names, no qualifier).
    mod translation {
        use delta_kernel::schema::StructField;

        use super::super::*;

        fn schema() -> StructType {
            StructType::try_new([
                StructField::nullable("b", DataType::BYTE),
                StructField::nullable("s", DataType::SHORT),
                StructField::nullable("i", DataType::INTEGER),
                StructField::nullable("l", DataType::LONG),
                StructField::nullable("flag", DataType::BOOLEAN),
                StructField::nullable("d", DataType::DATE),
                StructField::nullable("ts", DataType::TIMESTAMP),
                StructField::nullable("ntz", DataType::TIMESTAMP_NTZ),
                StructField::nullable("str", DataType::STRING),
                StructField::nullable("x", DataType::DOUBLE),
                StructField::nullable("data", DataType::LONG),
            ])
            .unwrap()
        }

        fn translate(filter: &Expr) -> Result<Pred, FilterRefusal> {
            let partitions =
                ["b", "s", "i", "l", "flag", "d", "ts", "ntz", "str", "x"].map(str::to_owned);
            partition_predicate(filter, &schema(), &partitions, &[])
        }

        fn col(n: &str) -> Expr {
            Expr::column([n])
        }

        fn lit(v: Literal) -> Expr {
            Expr::literal(v)
        }

        fn call(name: &str, args: Vec<Expr>) -> Expr {
            Expr::function(name, args)
        }

        fn kcol(n: &str) -> KExpr {
            KExpr::column([n])
        }

        fn eq(c: &str, v: Literal) -> Expr {
            call("=", vec![col(c), lit(v)])
        }

        fn keq(c: &str, s: Scalar) -> Pred {
            Pred::eq(kcol(c), KExpr::literal(s))
        }

        #[test]
        fn connectives_and_null_tests_translate() {
            let and = call(
                "AND",
                vec![eq("i", Literal::Integer(1)), eq("l", Literal::Long(2))],
            );
            assert_eq!(
                translate(&and).unwrap(),
                Pred::and(keq("i", Scalar::Integer(1)), keq("l", Scalar::Long(2)))
            );
            let or = call(
                "or",
                vec![eq("i", Literal::Integer(1)), lit(Literal::Boolean(false))],
            );
            assert_eq!(
                translate(&or).unwrap(),
                Pred::or(keq("i", Scalar::Integer(1)), Pred::literal(false))
            );
            for not in ["not", "!"] {
                assert_eq!(
                    translate(&call(not, vec![eq("i", Literal::Integer(1))])).unwrap(),
                    Pred::not(keq("i", Scalar::Integer(1)))
                );
            }
            assert_eq!(
                translate(&call("isnull", vec![col("S")])).unwrap(),
                Pred::is_null(kcol("s"))
            );
            assert_eq!(
                translate(&call("isNotNull", vec![col("d")])).unwrap(),
                Pred::is_not_null(kcol("d"))
            );
            let inl = call(
                "in",
                vec![col("b"), lit(Literal::Integer(1)), lit(Literal::Long(-2))],
            );
            assert_eq!(
                translate(&inl).unwrap(),
                Pred::or_from([keq("b", Scalar::Byte(1)), keq("b", Scalar::Byte(-2))])
            );
        }

        #[test]
        fn literals_take_the_column_type() {
            let cases = [
                ("b", Literal::Byte(3), Scalar::Byte(3)),
                ("s", Literal::Short(-4), Scalar::Short(-4)),
                ("s", Literal::Integer(300), Scalar::Short(300)),
                ("i", Literal::Long(7), Scalar::Integer(7)),
                ("l", Literal::Byte(5), Scalar::Long(5)),
                ("l", Literal::Short(6), Scalar::Long(6)),
                ("i", Literal::Null(None), Scalar::Null(DataType::INTEGER)),
                ("flag", Literal::Boolean(true), Scalar::Boolean(true)),
                ("d", Literal::Date(19_000), Scalar::Date(19_000)),
                ("ts", Literal::Timestamp(1_000), Scalar::Timestamp(1_000)),
                (
                    "ntz",
                    Literal::TimestampNtz(2_000),
                    Scalar::TimestampNtz(2_000),
                ),
                (
                    "str",
                    Literal::String("a".into()),
                    Scalar::String("a".into()),
                ),
                ("i", Literal::String("42".into()), Scalar::Integer(42)),
            ];
            for (c, v, want) in cases {
                assert_eq!(translate(&eq(c, v)).unwrap(), keq(c, want), "column {c}");
            }
        }

        #[test]
        fn comparisons_and_flipped_operands() {
            let one = || KExpr::literal(Scalar::Integer(1));
            let cases: [(&str, Pred); 8] = [
                ("==", Pred::eq(kcol("i"), one())),
                ("<=>", Pred::not(Pred::distinct(kcol("i"), one()))),
                ("!=", Pred::ne(kcol("i"), one())),
                ("<>", Pred::ne(kcol("i"), one())),
                ("<", Pred::lt(kcol("i"), one())),
                ("<=", Pred::le(kcol("i"), one())),
                (">", Pred::gt(kcol("i"), one())),
                (">=", Pred::ge(kcol("i"), one())),
            ];
            for (op, want) in cases {
                let e = call(op, vec![col("i"), lit(Literal::Integer(1))]);
                assert_eq!(translate(&e).unwrap(), want, "operator {op}");
            }
            let flipped = call("<", vec![lit(Literal::Integer(1)), col("i")]);
            assert_eq!(translate(&flipped).unwrap(), Pred::lt(one(), kcol("i")));
        }

        #[test]
        fn refusals() {
            let unsupported = |e: &Expr| matches!(translate(e), Err(FilterRefusal::Unsupported(_)));
            // Out of range for the column type, wrong literal type, a non-primitive-mapped literal.
            assert!(unsupported(&eq("b", Literal::Integer(1_000))));
            assert!(unsupported(&eq("i", Literal::Long(i64::MAX))));
            assert!(unsupported(&eq("flag", Literal::Integer(1))));
            assert!(unsupported(&eq("d", Literal::Boolean(true))));
            assert!(unsupported(&eq("x", Literal::Double(1.0))));
            assert!(unsupported(&eq("i", Literal::String("nope".into()))));
            // Non-literal and non-column operands, an unknown operator or function, a bare column.
            assert!(unsupported(&call("=", vec![col("i"), col("l")])));
            assert!(unsupported(&call(
                "=",
                vec![lit(Literal::Integer(1)), lit(Literal::Integer(2))]
            )));
            assert!(unsupported(&call(
                "like",
                vec![col("str"), lit(Literal::String("a%".into()))]
            )));
            assert!(unsupported(&call("coalesce", vec![col("i")])));
            assert!(unsupported(&col("i")));
            assert!(unsupported(&Expr::column(Vec::<String>::new())));
            assert!(unsupported(&call(
                "isnull",
                vec![Expr::column(Vec::<String>::new())]
            )));
            // A data column, on either side and inside a connective.
            assert_eq!(
                translate(&eq("data", Literal::Long(1))),
                Err(FilterRefusal::DataColumn("data".to_owned()))
            );
            assert_eq!(
                translate(&call("<", vec![lit(Literal::Long(1)), col("data")])),
                Err(FilterRefusal::DataColumn("data".to_owned()))
            );
            assert_eq!(
                translate(&call("in", vec![col("data"), lit(Literal::Long(1))])),
                Err(FilterRefusal::DataColumn("data".to_owned()))
            );
            // A partition name missing from the schema reads as a data column.
            let missing = partition_predicate(
                &eq("gone", Literal::Long(1)),
                &schema(),
                &["gone".to_owned()],
                &[],
            );
            assert_eq!(missing, Err(FilterRefusal::DataColumn("gone".to_owned())));
        }
    }
}
