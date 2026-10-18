// SPDX-License-Identifier: Apache-2.0
//! Filters as the typed predicates file formats prune with
//! ([`PushdownExpr`]). A translation must keep every row the filter keeps
//! (pruning keeps a superset), so anything not understood is left out of a
//! conjunction and makes a disjunction or negation untranslatable.

use krishiv_io::{CmpOp, ColumnPath, PushdownExpr, ScalarValue};
use krishiv_plan::{
    BoundColumn, Builtin, Expr, ExprNode, FunctionId, Literal, ResolvedExpr, ResolvedFunction,
};

/// The typed form of `e` over the columns of `scan`, when there is one.
pub(crate) fn translate(e: &Expr, scan: &[BoundColumn]) -> Option<PushdownExpr> {
    tr(e, scan, false)
}

/// `exact`: the translation must keep exactly the rows `e` keeps (under a
/// negation, where a superset would turn into a subset).
fn tr(e: &Expr, scan: &[BoundColumn], exact: bool) -> Option<PushdownExpr> {
    let f = builtin(e)?;
    let (b, args) = f;
    match (b, args.as_slice()) {
        (Builtin::And, [l, r]) => match (tr(l, scan, exact), tr(r, scan, exact)) {
            (Some(l), Some(r)) => Some(PushdownExpr::And(vec![l, r])),
            // Dropping a conjunct keeps a superset.
            (one, None) | (None, one) if !exact => one,
            _ => None,
        },
        (Builtin::Or, [l, r]) => Some(PushdownExpr::Or(vec![
            tr(l, scan, exact)?,
            tr(r, scan, exact)?,
        ])),
        (Builtin::Not, [x]) => Some(PushdownExpr::Not(Box::new(tr(x, scan, true)?))),
        (Builtin::IsNull, [x]) => Some(PushdownExpr::IsNull(column(x, scan)?)),
        (Builtin::IsNotNull, [x]) => Some(PushdownExpr::IsNotNull(column(x, scan)?)),
        (Builtin::Eq | Builtin::Lt | Builtin::LtEq | Builtin::Gt | Builtin::GtEq, [l, r]) => {
            let op = cmp_op(b)?;
            if let (Some(c), Some(v)) = (column(l, scan), value(r)) {
                Some(PushdownExpr::cmp(c, op, v))
            } else {
                let (c, v) = (column(r, scan)?, value(l)?);
                Some(PushdownExpr::cmp(c, flip(op), v))
            }
        }
        (Builtin::In, [x, list @ ..]) => {
            let c = column(x, scan)?;
            let values = list
                .iter()
                .map(|v| value_or_null(v))
                .collect::<Option<Vec<_>>>()?;
            // A NULL element never makes a row match, so dropping it keeps
            // the rows `x IN (...)` keeps. But a non-matching row is then
            // NULL rather than false, so `NOT (x IN (1, NULL))` keeps no
            // row: under a negation the list is not translatable.
            if exact && values.iter().any(Option::is_none) {
                return None;
            }
            let values: Vec<ScalarValue> = values.into_iter().flatten().collect();
            Some(PushdownExpr::In {
                column: c,
                values,
                collation: krishiv_io::Collation::Utf8Binary,
            })
        }
        (Builtin::Between, [x, lo, hi]) => {
            let c = column(x, scan)?;
            Some(PushdownExpr::And(vec![
                PushdownExpr::cmp(c.clone(), CmpOp::GtEq, value(lo)?),
                PushdownExpr::cmp(c, CmpOp::LtEq, value(hi)?),
            ]))
        }
        (Builtin::Like, [x, p]) => {
            let c = column(x, scan)?;
            let ExprNode::Literal(Literal::String(p)) = &p.node else {
                return None;
            };
            // `prefix%` with no other wildcard or escape.
            let prefix = p.strip_suffix('%')?;
            if prefix.is_empty() || prefix.contains(['%', '_', '\\']) {
                return None;
            }
            Some(PushdownExpr::StartsWith {
                column: c,
                prefix: prefix.to_owned(),
                collation: krishiv_io::Collation::Utf8Binary,
            })
        }
        _ => None,
    }
}

fn builtin(e: &Expr) -> Option<(Builtin, Vec<&Expr>)> {
    let ExprNode::Resolved(ResolvedExpr::Function(ResolvedFunction {
        id: FunctionId::Builtin(b),
        args,
        ..
    })) = &e.node
    else {
        return None;
    };
    Some((*b, args.iter().map(AsRef::as_ref).collect()))
}

fn cmp_op(b: Builtin) -> Option<CmpOp> {
    Some(match b {
        Builtin::Eq => CmpOp::Eq,
        Builtin::Lt => CmpOp::Lt,
        Builtin::LtEq => CmpOp::LtEq,
        Builtin::Gt => CmpOp::Gt,
        Builtin::GtEq => CmpOp::GtEq,
        _ => return None,
    })
}

/// `l op r` as `r op' l`.
fn flip(op: CmpOp) -> CmpOp {
    match op {
        CmpOp::Lt => CmpOp::Gt,
        CmpOp::LtEq => CmpOp::GtEq,
        CmpOp::Gt => CmpOp::Lt,
        CmpOp::GtEq => CmpOp::LtEq,
        other => other,
    }
}

fn column(e: &Expr, scan: &[BoundColumn]) -> Option<ColumnPath> {
    let ExprNode::Resolved(ResolvedExpr::Column(c)) = &e.node else {
        return None;
    };
    scan.iter()
        .find(|s| s.id == c.id)
        .map(|s| ColumnPath::new(s.name.clone()))
}

/// A non-null literal's value.
fn value(e: &Expr) -> Option<ScalarValue> {
    value_or_null(e).flatten()
}

/// A literal's value: `Some(None)` for NULL, `None` for anything else.
fn value_or_null(e: &Expr) -> Option<Option<ScalarValue>> {
    let ExprNode::Literal(l) = &e.node else {
        return None;
    };
    Some(Some(match l {
        Literal::Null(_) => return Some(None),
        Literal::Boolean(v) => ScalarValue::Boolean(*v),
        Literal::Byte(v) => ScalarValue::Int8(*v),
        Literal::Short(v) => ScalarValue::Int16(*v),
        Literal::Integer(v) => ScalarValue::Int32(*v),
        Literal::Long(v) => ScalarValue::Int64(*v),
        Literal::Float(v) => ScalarValue::Float32(*v),
        Literal::Double(v) => ScalarValue::Float64(*v),
        Literal::Decimal {
            unscaled,
            data_type,
        } => ScalarValue::Decimal128 {
            unscaled: *unscaled,
            precision: data_type.precision(),
            scale: i8::try_from(data_type.scale()).ok()?,
        },
        Literal::String(v) => ScalarValue::Utf8(v.clone()),
        Literal::Binary(v) => ScalarValue::Binary(v.clone()),
        Literal::Date(v) => ScalarValue::Date32(*v),
        Literal::Timestamp(v) | Literal::TimestampNtz(v) => ScalarValue::TimestampMicros(*v),
        _ => return None,
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use krishiv_plan::{AttrId, FunctionKind};
    use krishiv_types::DataType;

    use super::*;

    fn col() -> (Expr, Vec<BoundColumn>) {
        let c = BoundColumn {
            id: AttrId(1),
            name: "a".to_owned(),
            data_type: DataType::Integer,
            nullable: true,
        };
        (
            Expr::new(ExprNode::Resolved(ResolvedExpr::Column(c.clone()))),
            vec![c],
        )
    }

    fn call(b: Builtin, args: Vec<Expr>) -> Expr {
        Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id: FunctionId::Builtin(b),
                kind: FunctionKind::Scalar,
                args: args.into_iter().map(Arc::new).collect(),
                distinct: false,
                filter: None,
                data_type: DataType::Boolean,
                nullable: true,
            },
        )))
    }

    fn in_list(with_null: bool) -> (Expr, Vec<BoundColumn>) {
        let (a, scan) = col();
        let mut args = vec![a, Expr::new(ExprNode::Literal(Literal::Integer(1)))];
        if with_null {
            args.push(Expr::new(ExprNode::Literal(Literal::Null(None))));
        }
        (call(Builtin::In, args), scan)
    }

    #[test]
    fn in_with_a_null_element_is_translatable_only_outside_a_negation() {
        let (e, scan) = in_list(true);
        let Some(PushdownExpr::In { values, .. }) = translate(&e, &scan) else {
            panic!("a IN (1, NULL) keeps a subset of a IN (1)");
        };
        assert_eq!(values, vec![ScalarValue::Int32(1)]);
        // Spark keeps no row for NOT (a IN (1, NULL)); NOT (a IN (1)) keeps
        // every non-null a <> 1.
        assert_eq!(translate(&call(Builtin::Not, vec![e]), &scan), None);
        let (e, scan) = in_list(false);
        assert!(matches!(
            translate(&call(Builtin::Not, vec![e]), &scan),
            Some(PushdownExpr::Not(_))
        ));
    }
}
