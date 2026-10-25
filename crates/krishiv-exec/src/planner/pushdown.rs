// SPDX-License-Identifier: Apache-2.0
//! Filters as the typed predicates file formats prune with
//! ([`PushdownExpr`]). A translation must keep every row the filter keeps
//! (pruning keeps a superset), so anything not understood is left out of a
//! conjunction and makes a disjunction or negation untranslatable.

use krishiv_io::{CmpOp, ColumnPath, PushdownExpr, RowLevel, ScalarValue};
use krishiv_plan::{
    BoundColumn, Builtin, Expr, ExprNode, FunctionId, Literal, ResolvedExpr, ResolvedFunction,
};

/// The typed form of `e` over the columns of `scan`, when there is one.
pub(crate) fn translate(e: &Expr, scan: &[BoundColumn]) -> Option<PushdownExpr> {
    tr(e, scan, false)
}

/// What a scan may drop rows by before the filter `e` sees them, under
/// `spark.sql.ansi.enabled = ansi`.
///
/// The filter evaluates its conjuncts in written order, each on the rows
/// no earlier one is FALSE on (Kleene `AND` with short circuit). So while
/// no conjunct may raise ([`krishiv_plan::may_raise`]), dropping any row
/// the filter drops is invisible: [`RowLevel::Full`]. Otherwise a row may
/// be dropped only where a conjunct written before the first raising one
/// is FALSE: each such conjunct becomes `c OR col IS NULL` for every
/// column it reads (it is NULL only where one of them is, being a
/// comparison of columns with non-null literals), translated exactly.
/// Untranslatable conjuncts are left out (fewer rows dropped).
pub(crate) fn row_level(e: &Expr, scan: &[BoundColumn], ansi: bool) -> RowLevel {
    let conjuncts = krishiv_plan::conjuncts(e);
    let Some(first) = conjuncts
        .iter()
        .position(|c| krishiv_plan::may_raise(c, ansi))
    else {
        return RowLevel::Full;
    };
    let mut prefix: Vec<PushdownExpr> = conjuncts[..first]
        .iter()
        .filter_map(|c| not_false(c, scan))
        .collect();
    RowLevel::Prefix(match prefix.len() {
        0 | 1 => prefix.pop(),
        _ => Some(PushdownExpr::And(prefix)),
    })
}

/// A typed predicate TRUE on every row `c` is not FALSE on, and never NULL.
fn not_false(c: &Expr, scan: &[BoundColumn]) -> Option<PushdownExpr> {
    let t = tr(c, scan, true)?;
    let mut columns = Vec::new();
    columns_of(&t, &mut columns);
    if columns.is_empty() {
        return Some(t);
    }
    let mut any = vec![t];
    any.extend(columns.into_iter().map(PushdownExpr::IsNull));
    Some(PushdownExpr::Or(any))
}

/// The columns `e` reads, each once.
fn columns_of(e: &PushdownExpr, out: &mut Vec<ColumnPath>) {
    let mut add = |c: &ColumnPath| {
        if !out.contains(c) {
            out.push(c.clone());
        }
    };
    match e {
        PushdownExpr::Cmp { column, .. }
        | PushdownExpr::In { column, .. }
        | PushdownExpr::StartsWith { column, .. }
        | PushdownExpr::Contains { column, .. }
        | PushdownExpr::Membership { column, .. }
        | PushdownExpr::IsNull(column)
        | PushdownExpr::IsNotNull(column) => add(column),
        PushdownExpr::And(xs) | PushdownExpr::Or(xs) => {
            xs.iter().for_each(|x| columns_of(x, out));
        }
        PushdownExpr::Not(x) => columns_of(x, out),
        _ => {}
    }
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
            let plain = |s: &str| !s.is_empty() && !s.contains(['%', '_', '\\']);
            // `%needle%` with no other wildcard or escape.
            if let Some(needle) = p.strip_prefix('%').and_then(|s| s.strip_suffix('%'))
                && plain(needle)
            {
                return Some(PushdownExpr::Contains {
                    column: c,
                    needle: needle.to_owned(),
                    collation: krishiv_io::Collation::Utf8Binary,
                });
            }
            // `prefix%` with no other wildcard or escape.
            let prefix = p.strip_suffix('%')?;
            if !plain(prefix) {
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
                deterministic: true,
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

    fn like(pattern: &str) -> Option<PushdownExpr> {
        let c = BoundColumn {
            id: AttrId(2),
            name: "s".to_owned(),
            data_type: DataType::String,
            nullable: true,
        };
        let e = call(
            Builtin::Like,
            vec![
                Expr::new(ExprNode::Resolved(ResolvedExpr::Column(c.clone()))),
                Expr::new(ExprNode::Literal(Literal::String(pattern.to_owned()))),
            ],
        );
        translate(&e, &[c])
    }

    #[test]
    fn like_patterns_with_one_wildcard_run_are_prefixes_or_substrings() {
        let utf8 = krishiv_io::Collation::Utf8Binary;
        assert_eq!(
            like("%google%"),
            Some(PushdownExpr::Contains {
                column: ColumnPath::new("s"),
                needle: "google".to_owned(),
                collation: utf8.clone(),
            })
        );
        assert_eq!(
            like("goo%"),
            Some(PushdownExpr::StartsWith {
                column: ColumnPath::new("s"),
                prefix: "goo".to_owned(),
                collation: utf8,
            })
        );
        // Other wildcards, escapes, an empty needle or a suffix: not pushed.
        for p in [
            "%go_gle%", "%go%gle%", "%go\\%%", "%%", "%", "%google", "go%gle",
        ] {
            assert_eq!(like(p), None, "{p}");
        }
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

    /// `CAST(s AS INT) > 0` in `mode` (it may raise under ANSI).
    fn cast_gt0(mode: krishiv_plan::CastMode) -> (Expr, BoundColumn) {
        let s = BoundColumn {
            id: AttrId(2),
            name: "s".to_owned(),
            data_type: DataType::String,
            nullable: true,
        };
        let cast = Expr::new(ExprNode::Cast {
            expr: Arc::new(Expr::new(ExprNode::Resolved(ResolvedExpr::Column(
                s.clone(),
            )))),
            to: DataType::Integer,
            mode,
        });
        (
            call(
                Builtin::Gt,
                vec![cast, Expr::new(ExprNode::Literal(Literal::Integer(0)))],
            ),
            s,
        )
    }

    #[test]
    fn rows_are_dropped_only_by_conjuncts_written_before_one_that_may_raise() {
        let (a, mut scan) = col();
        let (cast, s) = cast_gt0(krishiv_plan::CastMode::Ansi);
        let (legacy, _) = cast_gt0(krishiv_plan::CastMode::Legacy);
        scan.push(s);
        let a_is_1 = call(
            Builtin::Eq,
            vec![a, Expr::new(ExprNode::Literal(Literal::Integer(1)))],
        );
        let and = |l: &Expr, r: &Expr| call(Builtin::And, vec![l.clone(), r.clone()]);
        // `a = 1` before the cast: rows where it is FALSE (not NULL).
        let a1 = PushdownExpr::cmp(ColumnPath::new("a"), CmpOp::Eq, ScalarValue::Int32(1));
        assert_eq!(
            row_level(&and(&a_is_1, &cast), &scan, true),
            RowLevel::Prefix(Some(PushdownExpr::Or(vec![
                a1,
                PushdownExpr::IsNull(ColumnPath::new("a")),
            ])))
        );
        // Nothing before it; with ANSI off the cast gives NULL.
        assert_eq!(
            row_level(&and(&cast, &a_is_1), &scan, true),
            RowLevel::Prefix(None)
        );
        assert_eq!(
            row_level(&and(&legacy, &a_is_1), &scan, false),
            RowLevel::Full
        );
        // `a IN (1, NULL)` is NULL, never FALSE, on rows it does not hold:
        // it cannot drop a row before the cast.
        let (in_null, _) = in_list(true);
        assert_eq!(
            row_level(&and(&in_null, &cast), &scan, true),
            RowLevel::Prefix(None)
        );
        assert_eq!(
            row_level(&and(&in_null, &a_is_1), &scan, true),
            RowLevel::Full
        );
    }
}
