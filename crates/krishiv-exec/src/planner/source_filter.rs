// SPDX-License-Identifier: Apache-2.0
//! A conjunct of the filter over a scan as Spark's data source filter translation sees it
//! (`DataSourceStrategy.translateFilterWithMapping` and the Python runner's serializability rule,
//! v4.3.0): a leaf or a `Not` of a leaf reaches Python; anything else Spark translates is kept in the
//! residual without being sent; the rest is not translated at all.

use krishiv_io::{ColumnPath, ScalarValue};
use krishiv_plan::{BoundColumn, Builtin, Expr, ExprNode, Literal, ResolvedExpr};

use super::pushdown::{builtin, cmp_op, value_or_null};
use crate::operators::{SourceConjunct, SourceFilter};

/// The state of `e` (one conjunct over the columns of `scan`) in Spark's filter translation.
pub(crate) fn classify(e: &Expr, scan: &[BoundColumn]) -> SourceConjunct {
    match translate(e, scan) {
        Tr::Leaf(f) => SourceConjunct::Serializable(f),
        Tr::Translated => SourceConjunct::Translated,
        Tr::No => SourceConjunct::Untranslated,
    }
}

/// `parts` joined by `AND`, left-deep, in the order given; `None` for no parts.
pub(crate) fn and_all(parts: Vec<&Expr>) -> Option<Expr> {
    parts.into_iter().cloned().reduce(|l, r| {
        Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            krishiv_plan::ResolvedFunction {
                id: krishiv_plan::FunctionId::Builtin(Builtin::And),
                kind: krishiv_plan::FunctionKind::Scalar,
                args: vec![std::sync::Arc::new(l), std::sync::Arc::new(r)],
                distinct: false,
                filter: None,
                within_group: Vec::new(),
                data_type: krishiv_types::DataType::Boolean,
                nullable: true,
                deterministic: true,
            },
        )))
    })
}

/// The `IsNotNull(column)` conjuncts Spark's `InferFiltersFromConstraints` adds for the nullable scan
/// columns `e` (one conjunct) rejects nulls of (`constructIsNotNullConstraints`): every attribute under
/// a chain of null-intolerant expressions (comparisons, `LIKE`, arithmetic, casts), not `<=>`,
/// `IS NULL`, `IN`, `AND` or `OR`. First-appearance order.
pub(crate) fn inferred_not_null(e: &Expr, scan: &[BoundColumn]) -> Vec<Expr> {
    let mut found: Vec<&BoundColumn> = Vec::new();
    null_rejected(e, scan, &mut found);
    found
        .into_iter()
        .map(|b| {
            let column = Expr::new(ExprNode::Resolved(ResolvedExpr::Column(b.clone())));
            Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
                krishiv_plan::ResolvedFunction {
                    id: krishiv_plan::FunctionId::Builtin(Builtin::IsNotNull),
                    kind: krishiv_plan::FunctionKind::Scalar,
                    args: vec![std::sync::Arc::new(column)],
                    distinct: false,
                    filter: None,
                    within_group: Vec::new(),
                    data_type: krishiv_types::DataType::Boolean,
                    nullable: false,
                    deterministic: true,
                },
            )))
        })
        .collect()
}

/// Collects the nullable scan columns `e` is NULL without (Spark's `scanNullIntolerantAttribute`).
fn null_rejected<'a>(e: &Expr, scan: &'a [BoundColumn], out: &mut Vec<&'a BoundColumn>) {
    match &e.node {
        ExprNode::Resolved(ResolvedExpr::Column(c)) => {
            if let Some(b) = scan.iter().find(|s| s.id == c.id)
                && b.nullable
                && !out.iter().any(|o| o.id == b.id)
            {
                out.push(b);
            }
        }
        ExprNode::Cast { expr, .. } | ExprNode::Collate { expr, .. } => {
            null_rejected(expr, scan, out);
        }
        _ => {
            if let Some((b, args)) = builtin(e)
                && matches!(
                    b,
                    Builtin::Eq
                        | Builtin::Lt
                        | Builtin::LtEq
                        | Builtin::Gt
                        | Builtin::GtEq
                        | Builtin::Add
                        | Builtin::Subtract
                        | Builtin::Multiply
                        | Builtin::Divide
                        | Builtin::Remainder
                        | Builtin::IntegralDivide
                        | Builtin::Negative
                        | Builtin::Like
                        | Builtin::ILike
                        | Builtin::RLike
                        | Builtin::Concat
                )
            {
                for a in args {
                    null_rejected(a, scan, out);
                }
            }
        }
    }
}

enum Tr {
    /// A serializable leaf or `Not` of a leaf.
    Leaf(SourceFilter),
    /// Translated by Spark, dropped by the Python runner.
    Translated,
    /// No data source filter form.
    No,
}

fn translate(e: &Expr, scan: &[BoundColumn]) -> Tr {
    if let ExprNode::Literal(Literal::Boolean(_)) = &e.node {
        return Tr::Translated;
    }
    if let ExprNode::Resolved(ResolvedExpr::Column(_)) = &e.node {
        // A boolean column alone is `EqualTo(column, true)`.
        return match attribute(e, scan) {
            Some(Attr::Plain(c, b)) if b.data_type == krishiv_types::DataType::Boolean => {
                Tr::Leaf(SourceFilter::EqualTo(c, ScalarValue::Boolean(true)))
            }
            Some(Attr::Collated) => Tr::Translated,
            _ => Tr::No,
        };
    }
    let Some((b, args)) = builtin(e) else {
        return Tr::No;
    };
    match (b, args.as_slice()) {
        (Builtin::And | Builtin::Or, [l, r]) => both(translate(l, scan), translate(r, scan)),
        (Builtin::Not, [x]) => match translate(x, scan) {
            Tr::Leaf(SourceFilter::Not(_)) | Tr::Translated => Tr::Translated,
            Tr::Leaf(f) => Tr::Leaf(SourceFilter::Not(Box::new(f))),
            Tr::No => Tr::No,
        },
        (Builtin::IsNull | Builtin::IsNotNull, [x]) => match attribute(x, scan) {
            Some(Attr::Plain(c, _)) => Tr::Leaf(if b == Builtin::IsNull {
                SourceFilter::IsNull(c)
            } else {
                SourceFilter::IsNotNull(c)
            }),
            Some(Attr::Collated) => Tr::Translated,
            None => Tr::No,
        },
        (Builtin::Eq | Builtin::Lt | Builtin::LtEq | Builtin::Gt | Builtin::GtEq, [l, r]) => {
            compare(b, l, r, scan)
        }
        (Builtin::EqNullSafe, [l, r]) => compare(b, l, r, scan),
        (Builtin::In, [x, list @ ..]) => {
            let Some(attr) = attribute(x, scan) else {
                return Tr::No;
            };
            let Attr::Plain(c, _) = attr else {
                return Tr::Translated;
            };
            let values: Option<Vec<Option<ScalarValue>>> =
                list.iter().map(|v| value_or_null(v)).collect();
            match values {
                Some(values) => Tr::Leaf(SourceFilter::In(
                    c,
                    values
                        .into_iter()
                        .map(|v| v.unwrap_or(ScalarValue::Null))
                        .collect(),
                )),
                None => Tr::No,
            }
        }
        (Builtin::Like, [x, p]) => like(x, p, scan),
        _ => Tr::No,
    }
}

fn both(l: Tr, r: Tr) -> Tr {
    match (l, r) {
        (Tr::No, _) | (_, Tr::No) => Tr::No,
        _ => Tr::Translated,
    }
}

/// `l op r`, the attribute on either side (the operator flips for a literal on the left).
fn compare(b: Builtin, l: &Expr, r: &Expr, scan: &[BoundColumn]) -> Tr {
    let (attr, lit, flipped) = match (attribute(l, scan), attribute(r, scan)) {
        (Some(a), None) => (a, r, false),
        (None, Some(a)) => (a, l, true),
        _ => return Tr::No,
    };
    let Some(value) = value_or_null(lit) else {
        return Tr::No;
    };
    let Attr::Plain(c, _) = attr else {
        return Tr::Translated;
    };
    // A NULL literal has no variant a source could use; the optimizer folds the comparison first.
    let Some(v) = value else {
        return Tr::Translated;
    };
    let f = match b {
        Builtin::EqNullSafe => SourceFilter::EqualNullSafe(c, v),
        _ => {
            let op = cmp_op(b).map(|op| {
                if flipped {
                    super::pushdown::flip(op)
                } else {
                    op
                }
            });
            match op {
                Some(krishiv_io::CmpOp::Eq) => SourceFilter::EqualTo(c, v),
                Some(krishiv_io::CmpOp::Lt) => SourceFilter::LessThan(c, v),
                Some(krishiv_io::CmpOp::LtEq) => SourceFilter::LessThanOrEqual(c, v),
                Some(krishiv_io::CmpOp::Gt) => SourceFilter::GreaterThan(c, v),
                Some(krishiv_io::CmpOp::GtEq) => SourceFilter::GreaterThanOrEqual(c, v),
                _ => return Tr::No,
            }
        }
    };
    Tr::Leaf(f)
}

/// `x LIKE 'p%'`, `'%p'` or `'%p%'` with no other wildcard or escape.
fn like(x: &Expr, p: &Expr, scan: &[BoundColumn]) -> Tr {
    let Some(attr) = attribute(x, scan) else {
        return Tr::No;
    };
    let ExprNode::Literal(Literal::String(p)) = &p.node else {
        return Tr::No;
    };
    let Attr::Plain(c, _) = attr else {
        return Tr::Translated;
    };
    let plain = |s: &str| !s.is_empty() && !s.contains(['%', '_', '\\']);
    if let Some(n) = p.strip_prefix('%').and_then(|s| s.strip_suffix('%'))
        && plain(n)
    {
        return Tr::Leaf(SourceFilter::StringContains(c, n.to_owned()));
    }
    if let Some(n) = p.strip_suffix('%')
        && plain(n)
    {
        return Tr::Leaf(SourceFilter::StringStartsWith(c, n.to_owned()));
    }
    if let Some(n) = p.strip_prefix('%')
        && plain(n)
    {
        return Tr::Leaf(SourceFilter::StringEndsWith(c, n.to_owned()));
    }
    Tr::No
}

enum Attr<'a> {
    Plain(ColumnPath, &'a BoundColumn),
    /// A column of a non-binary collation: Spark wraps its filters, the runner drops them.
    Collated,
}

/// The scan column `e` names, if it is one.
fn attribute<'a>(e: &Expr, scan: &'a [BoundColumn]) -> Option<Attr<'a>> {
    let ExprNode::Resolved(ResolvedExpr::Column(c)) = &e.node else {
        return None;
    };
    let b = scan.iter().find(|s| s.id == c.id)?;
    Some(if b.data_type.contains_collation() {
        Attr::Collated
    } else {
        Attr::Plain(ColumnPath::new(b.name.clone()), b)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use krishiv_plan::{AttrId, FunctionId, FunctionKind, ResolvedFunction};
    use krishiv_types::DataType;

    use super::*;

    fn col(id: u64, name: &str, t: DataType) -> BoundColumn {
        BoundColumn {
            id: AttrId(id),
            name: name.to_owned(),
            data_type: t,
            nullable: true,
        }
    }

    fn c(b: &BoundColumn) -> Expr {
        Expr::new(ExprNode::Resolved(ResolvedExpr::Column(b.clone())))
    }

    fn int(v: i32) -> Expr {
        Expr::new(ExprNode::Literal(Literal::Integer(v)))
    }

    fn s(v: &str) -> Expr {
        Expr::new(ExprNode::Literal(Literal::String(v.to_owned())))
    }

    fn call(b: Builtin, args: Vec<Expr>) -> Expr {
        Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: FunctionId::Builtin(b),
                kind: FunctionKind::Scalar,
                args: args.into_iter().map(Arc::new).collect(),
                distinct: false,
                filter: None,
                within_group: Vec::new(),
                data_type: DataType::Boolean,
                nullable: true,
                deterministic: true,
            },
        )))
    }

    fn x() -> BoundColumn {
        col(1, "x", DataType::Integer)
    }

    #[test]
    fn a_leaf_serializes_with_the_attribute_on_either_side() {
        let x = x();
        let scan = [x.clone()];
        assert_eq!(
            classify(&call(Builtin::Gt, vec![c(&x), int(1)]), &scan),
            SourceConjunct::Serializable(SourceFilter::GreaterThan(
                ColumnPath::new("x"),
                ScalarValue::Int32(1)
            ))
        );
        // `1 < x` is `x > 1`.
        assert_eq!(
            classify(&call(Builtin::Lt, vec![int(1), c(&x)]), &scan),
            SourceConjunct::Serializable(SourceFilter::GreaterThan(
                ColumnPath::new("x"),
                ScalarValue::Int32(1)
            ))
        );
        assert_eq!(
            classify(&call(Builtin::IsNotNull, vec![c(&x)]), &scan),
            SourceConjunct::Serializable(SourceFilter::IsNotNull(ColumnPath::new("x")))
        );
        assert_eq!(
            classify(&call(Builtin::EqNullSafe, vec![c(&x), int(2)]), &scan),
            SourceConjunct::Serializable(SourceFilter::EqualNullSafe(
                ColumnPath::new("x"),
                ScalarValue::Int32(2)
            ))
        );
    }

    #[test]
    fn in_and_like_leaves() {
        let x = x();
        let t = col(2, "t", DataType::String);
        let scan = [x.clone(), t.clone()];
        let Some(SourceConjunct::Serializable(SourceFilter::In(_, v))) = Some(classify(
            &call(Builtin::In, vec![c(&x), int(1), int(2)]),
            &scan,
        )) else {
            panic!("IN is a leaf");
        };
        assert_eq!(v, vec![ScalarValue::Int32(1), ScalarValue::Int32(2)]);
        let like = |p: &str| classify(&call(Builtin::Like, vec![c(&t), s(p)]), &scan);
        let path = ColumnPath::new("t");
        assert_eq!(
            like("a%"),
            SourceConjunct::Serializable(SourceFilter::StringStartsWith(path.clone(), "a".into()))
        );
        assert_eq!(
            like("%a"),
            SourceConjunct::Serializable(SourceFilter::StringEndsWith(path.clone(), "a".into()))
        );
        assert_eq!(
            like("%a%"),
            SourceConjunct::Serializable(SourceFilter::StringContains(path, "a".into()))
        );
        assert_eq!(like("a_b"), SourceConjunct::Untranslated);
    }

    #[test]
    fn composites_are_translated_but_not_serialized() {
        let x = x();
        let scan = [x.clone()];
        let a = call(Builtin::Eq, vec![c(&x), int(1)]);
        let b = call(Builtin::Gt, vec![c(&x), int(2)]);
        assert_eq!(
            classify(&call(Builtin::Or, vec![a.clone(), b]), &scan),
            SourceConjunct::Translated
        );
        // `Not` of a leaf serializes; a double `Not` does not.
        let not = call(Builtin::Not, vec![a.clone()]);
        assert!(matches!(
            classify(&not, &scan),
            SourceConjunct::Serializable(SourceFilter::Not(_))
        ));
        assert_eq!(
            classify(&call(Builtin::Not, vec![not]), &scan),
            SourceConjunct::Translated
        );
        assert_eq!(
            classify(&Expr::new(ExprNode::Literal(Literal::Boolean(true))), &scan),
            SourceConjunct::Translated
        );
        // A part with no filter form makes the whole untranslatable.
        let f = call(
            Builtin::Gt,
            vec![call(Builtin::Add, vec![c(&x), int(1)]), int(2)],
        );
        assert_eq!(classify(&f, &scan), SourceConjunct::Untranslated);
        assert_eq!(
            classify(&call(Builtin::And, vec![a, f]), &scan),
            SourceConjunct::Untranslated
        );
    }

    #[test]
    fn a_boolean_column_is_equal_to_true_and_a_null_literal_is_translated() {
        let flag = col(3, "flag", DataType::Boolean);
        let x = x();
        let scan = [flag.clone(), x.clone()];
        assert_eq!(
            classify(&c(&flag), &scan),
            SourceConjunct::Serializable(SourceFilter::EqualTo(
                ColumnPath::new("flag"),
                ScalarValue::Boolean(true)
            ))
        );
        let null = Expr::new(ExprNode::Literal(Literal::Null(None)));
        assert_eq!(
            classify(&call(Builtin::Eq, vec![c(&x), null]), &scan),
            SourceConjunct::Translated
        );
        assert_eq!(classify(&c(&x), &scan), SourceConjunct::Untranslated);
    }

    #[test]
    fn not_null_is_inferred_for_every_nullable_attribute_under_a_null_intolerant_chain() {
        let x = x();
        let mut y = col(2, "y", DataType::Integer);
        y.nullable = false;
        let scan = [x.clone(), y.clone()];
        let names = |e: &Expr| -> Vec<String> {
            inferred_not_null(e, &scan)
                .iter()
                .map(ToString::to_string)
                .collect()
        };
        // `x % 5 = 1` rejects null x; the non-nullable y is skipped.
        let rem = call(Builtin::Remainder, vec![c(&x), int(5)]);
        assert_eq!(names(&call(Builtin::Eq, vec![rem, int(1)])).len(), 1);
        let both = call(Builtin::Eq, vec![c(&x), c(&y)]);
        assert_eq!(names(&both).len(), 1);
        // `<=>`, `IS NULL` and `OR` reject nothing.
        assert!(names(&call(Builtin::EqNullSafe, vec![c(&x), int(1)])).is_empty());
        assert!(names(&call(Builtin::IsNull, vec![c(&x)])).is_empty());
    }
}
