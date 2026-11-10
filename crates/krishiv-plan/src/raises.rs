// SPDX-License-Identifier: Apache-2.0
//! Which resolved expressions may raise an error on some row.
//!
//! A scan may drop a row before the filter above it sees it only when no
//! conjunct that still has to see the row can raise: Spark evaluates a
//! filter's conjuncts in written order on every row its reader returns,
//! and its reader drops no row on its own. So a stage that drops rows
//! early must know which conjuncts can raise.
//!
//! The rule is an allow-list per function **and argument types**, with
//! "may raise" as the default. The same operator raises or not depending
//! on its operand types: with `spark.sql.ansi.enabled=false`, `+`, `*`, `/`
//! and `CAST` give `NULL` (or wrap) on integral, decimal, floating and
//! string operands, but raise on interval, date and timestamp operands
//! (`CAST('abc' AS INTERVAL DAY)` raises `INVALID_INTERVAL_FORMAT` with
//! ANSI off, as Spark 4.2 does). Every arithmetic, cast or function over
//! such an operand is therefore "may raise" whatever the ANSI setting.

use krishiv_types::DataType;

use crate::expr::{CastMode, Expr, ExprNode, ResolvedExpr, SubqueryKind};
use crate::function::{Builtin, FunctionId, FunctionKind};
use crate::literal::Literal;

/// Whether evaluating `e` may raise an error on some row, under
/// `spark.sql.ansi.enabled = ansi` (casts carry their own mode). `true`
/// unless every node of `e` is on the allow-list (see the module docs):
///
/// - columns, literals, comparisons, `AND`/`OR`/`NOT`, `IS [NOT] NULL`,
///   `IN` lists, `BETWEEN`, `CASE`, `coalesce`, `startswith`/`endswith`/
///   `contains`, `substring`, and `LIKE`/`ILIKE` with a literal pattern that holds no
///   escape;
/// - arithmetic only with ANSI off and numeric (or `NULL`) operands;
/// - `TRY_CAST`, and other casts between boolean, numeric and string types
///   that cannot fail in their mode;
/// - `EXISTS` and `IN (subquery)` (their subquery runs once, not per row).
///
/// Anything else, including any operand of an interval, date or timestamp
/// type, a scalar subquery (`SCALAR_SUBQUERY_TOO_MANY_ROWS`) and every
/// registry function not listed, may raise.
#[must_use]
pub fn may_raise(e: &Expr, ansi: bool) -> bool {
    !never_raises(e, ansi)
}

/// Whether `e` may run on rows Spark would not evaluate it on, with no
/// observable effect: it is deterministic, holds no subquery, and raises on
/// no row, either by [`may_raise`] or because it is exact decimal arithmetic
/// ([`exact_decimal`], which also holds with ANSI on).
#[must_use]
pub fn runs_on_any_row(e: &Expr, ansi: bool) -> bool {
    is_deterministic(e) && !has_subquery(e) && (exact_decimal(e) || !may_raise(e, ansi))
}

fn has_subquery(e: &Expr) -> bool {
    let mut found = false;
    let _ = crate::walk::visit_distinct(crate::walk::Node::Expr(e), |n| {
        if let crate::walk::Node::Expr(x) = n
            && matches!(x.node, ExprNode::Subquery { .. })
        {
            found = true;
            return std::ops::ControlFlow::Break(());
        }
        std::ops::ControlFlow::Continue(())
    });
    found
}

/// Whether `e` is decimal `+`, `-` and `*` whose every result is exact: each
/// node's resolved type is the type Spark gives its operands before any
/// `allowPrecisionLoss` adjustment (`(p1 + p2 + 1, s1 + s2)` for `*`,
/// `(max(p1 - s1, p2 - s2) + max(s1, s2) + 1, max(s1, s2))` for `+`/`-`),
/// so it is at most 38 digits and no value can overflow it: no
/// `NUMERIC_VALUE_OUT_OF_RANGE` with ANSI on, no NULL with it off, under
/// either precision-loss setting. The leaves are decimal columns and
/// literals, integral columns cast to a decimal that holds every value, and
/// integral literals cast to a decimal that holds them (the analyzer's
/// `cast(1 AS decimal(1,0))`).
#[must_use]
pub fn exact_decimal(e: &Expr) -> bool {
    match &e.node {
        ExprNode::Literal(Literal::Decimal { .. }) => true,
        ExprNode::Resolved(ResolvedExpr::Column(c)) => matches!(c.data_type, DataType::Decimal(_)),
        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => exact_decimal(expr),
        ExprNode::Cast {
            expr,
            to: DataType::Decimal(to),
            ..
        } => match (&expr.node, type_of(expr)) {
            (ExprNode::Literal(l), _) => integral_literal_digits(l)
                .is_some_and(|d| d <= i32::from(to.precision()) - i32::from(to.scale())),
            (_, Some(from @ DataType::Decimal(_))) => {
                exact_decimal(expr) && widens(&from, &DataType::Decimal(*to))
            }
            (
                ExprNode::Resolved(ResolvedExpr::Column(_)),
                Some(
                    from @ (DataType::Byte | DataType::Short | DataType::Integer | DataType::Long),
                ),
            ) => widens(&from, &DataType::Decimal(*to)),
            _ => false,
        },
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            if f.kind != FunctionKind::Scalar || f.filter.is_some() {
                return false;
            }
            let FunctionId::Builtin(op) = &f.id else {
                return false;
            };
            let [l, r] = f.args.as_slice() else {
                return false;
            };
            let (Some(DataType::Decimal(a)), Some(DataType::Decimal(b)), DataType::Decimal(out)) =
                (type_of(l), type_of(r), &f.data_type)
            else {
                return false;
            };
            let (p1, s1) = (i32::from(a.precision()), i32::from(a.scale()));
            let (p2, s2) = (i32::from(b.precision()), i32::from(b.scale()));
            let exact = match op {
                Builtin::Multiply => (p1 + p2 + 1, s1 + s2),
                Builtin::Add | Builtin::Subtract => {
                    let s = s1.max(s2);
                    ((p1 - s1).max(p2 - s2) + s + 1, s)
                }
                _ => return false,
            };
            exact == (i32::from(out.precision()), i32::from(out.scale()))
                && exact_decimal(l)
                && exact_decimal(r)
        }
        _ => false,
    }
}

/// The decimal digits of an integral literal's absolute value.
fn integral_literal_digits(l: &Literal) -> Option<i32> {
    let v: u128 = match l {
        Literal::Byte(v) => u128::from(v.unsigned_abs()),
        Literal::Short(v) => u128::from(v.unsigned_abs()),
        Literal::Integer(v) => u128::from(v.unsigned_abs()),
        Literal::Long(v) => u128::from(v.unsigned_abs()),
        _ => return None,
    };
    let mut digits = 1;
    let mut rest = v / 10;
    while rest > 0 {
        digits += 1;
        rest /= 10;
    }
    Some(digits)
}

/// Whether `e` holds no nondeterministic call (`rand`, `monotonically_increasing_id`,
/// ...), so evaluating it twice gives the same values. An unresolved call
/// counts as nondeterministic.
#[must_use]
pub fn is_deterministic(e: &Expr) -> bool {
    let mut nondeterministic = false;
    let _ = crate::walk::visit_distinct(crate::walk::Node::Expr(e), |n| {
        if let crate::walk::Node::Expr(x) = n {
            match &x.node {
                ExprNode::Function(_) => {
                    nondeterministic = true;
                    return std::ops::ControlFlow::Break(());
                }
                ExprNode::Resolved(ResolvedExpr::Function(f)) if !f.deterministic => {
                    nondeterministic = true;
                    return std::ops::ControlFlow::Break(());
                }
                _ => {}
            }
        }
        std::ops::ControlFlow::Continue(())
    });
    !nondeterministic
}

/// The conjuncts of `e` in written order (`a AND (b AND c)` gives `a`,
/// `b`, `c`): the order a filter short-circuits in.
#[must_use]
pub fn conjuncts(e: &Expr) -> Vec<&Expr> {
    let mut out = Vec::new();
    let mut stack = vec![e];
    while let Some(x) = stack.pop() {
        match &x.node {
            ExprNode::Resolved(ResolvedExpr::Function(f))
                if f.id == FunctionId::Builtin(Builtin::And) && f.args.len() == 2 =>
            {
                // Right first, so the left conjunct pops first.
                stack.extend(f.args.iter().rev().map(AsRef::as_ref));
            }
            _ => out.push(x),
        }
    }
    out
}

fn never_raises(e: &Expr, ansi: bool) -> bool {
    match &e.node {
        ExprNode::Literal(_)
        | ExprNode::Resolved(ResolvedExpr::Column(_) | ResolvedExpr::OuterColumn(_)) => true,
        ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => never_raises(expr, ansi),
        ExprNode::Resolved(ResolvedExpr::Field { base, .. }) => never_raises(base, ansi),
        // The pinned mode decides inside (T184).
        ExprNode::Resolved(ResolvedExpr::EvalMode { ansi, expr, .. }) => never_raises(expr, *ansi),
        ExprNode::Cast { expr, to, mode } => {
            never_raises(expr, ansi)
                && type_of(expr).is_some_and(|from| cast_never_raises(&from, to, *mode))
        }
        ExprNode::Subquery { kind, .. } => match kind {
            SubqueryKind::Exists => true,
            SubqueryKind::In(values) => values.iter().all(|v| never_raises(v, ansi)),
            // `SCALAR_SUBQUERY_TOO_MANY_ROWS`, and anything else.
            _ => false,
        },
        ExprNode::Resolved(ResolvedExpr::Function(f)) => {
            f.kind == FunctionKind::Scalar
                && f.filter.is_none()
                && f.args.iter().all(|a| never_raises(a, ansi))
                && function_never_raises(&f.id, &f.args, ansi)
        }
        _ => false,
    }
}

/// Whether the call itself (its arguments already checked) never raises.
fn function_never_raises(id: &FunctionId, args: &[std::sync::Arc<Expr>], ansi: bool) -> bool {
    let numeric = || {
        args.iter()
            .all(|a| type_of(a).is_some_and(|t| is_numeric(&t) || t == DataType::Null))
    };
    match id {
        FunctionId::Builtin(b) => match b {
            Builtin::Eq
            | Builtin::EqNullSafe
            | Builtin::Lt
            | Builtin::LtEq
            | Builtin::Gt
            | Builtin::GtEq
            | Builtin::And
            | Builtin::Or
            | Builtin::Not
            | Builtin::IsNull
            | Builtin::IsNotNull
            | Builtin::In
            | Builtin::Between
            | Builtin::When => true,
            // An invalid pattern raises; a literal pattern without the
            // default escape character cannot be invalid.
            Builtin::Like | Builtin::ILike => match args {
                [_, p] => matches!(
                    &p.node,
                    ExprNode::Literal(Literal::String(p)) if !p.contains('\\')
                ),
                _ => false,
            },
            Builtin::Add
            | Builtin::Subtract
            | Builtin::Multiply
            | Builtin::Divide
            | Builtin::Remainder
            | Builtin::IntegralDivide
            | Builtin::Negative => !ansi && numeric(),
            _ => false,
        },
        FunctionId::Registry { name, .. } => match name.as_str() {
            "coalesce" | "startswith" | "endswith" | "contains" | "substring" | "substr" => true,
            "try_add" | "try_subtract" | "try_multiply" | "try_divide" | "try_mod" => numeric(),
            _ => false,
        },
        // User code may raise (or fail its contract): never assumed pure.
        FunctionId::Udf { .. } | FunctionId::PythonUdf { .. } => false,
    }
}

/// Whether `CAST(x AS to)` from `from` in `mode` never raises.
fn cast_never_raises(from: &DataType, to: &DataType, mode: CastMode) -> bool {
    let plain = |t: &DataType| {
        is_numeric(t) || matches!(t, DataType::Boolean | DataType::String | DataType::Null)
    };
    if !plain(from) || !plain(to) {
        // Interval, date, timestamp and other types: may raise with ANSI
        // off too.
        return false;
    }
    match mode {
        CastMode::Try | CastMode::Legacy => true,
        // ANSI: only casts that cannot fail on any value.
        _ => from == to || *from == DataType::Null || *to == DataType::String || widens(from, to),
    }
}

/// Whether every value of `from` is a value of `to` (no overflow).
fn widens(from: &DataType, to: &DataType) -> bool {
    let rank = |t: &DataType| match t {
        DataType::Byte => Some(1),
        DataType::Short => Some(2),
        DataType::Integer => Some(3),
        DataType::Long => Some(4),
        _ => None,
    };
    match (from, to) {
        (DataType::Boolean, _) => false,
        (_, DataType::Float | DataType::Double) => is_numeric(from),
        (DataType::Decimal(f), DataType::Decimal(t)) => {
            t.scale() >= f.scale()
                && i32::from(t.precision()) - i32::from(t.scale())
                    >= i32::from(f.precision()) - i32::from(f.scale())
        }
        (_, DataType::Decimal(t)) => rank(from).is_some_and(|r| {
            // Digits of the integral type's largest value.
            let digits = [0, 3, 5, 10, 19][r];
            i32::from(t.precision()) - i32::from(t.scale()) >= digits
        }),
        _ => match (rank(from), rank(to)) {
            (Some(f), Some(t)) => f <= t,
            _ => false,
        },
    }
}

fn is_numeric(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Long
            | DataType::Float
            | DataType::Double
            | DataType::Decimal(_)
    )
}

/// The type of a resolved expression, when its node says it.
pub(crate) fn type_of(e: &Expr) -> Option<DataType> {
    match &e.node {
        ExprNode::Resolved(
            ResolvedExpr::Column(c)
            | ResolvedExpr::OuterColumn(c)
            | ResolvedExpr::LambdaVar(c)
            | ResolvedExpr::Alias { output: c, .. },
        ) => Some(c.data_type.clone()),
        ExprNode::Resolved(
            ResolvedExpr::Field { data_type, .. } | ResolvedExpr::EvalMode { data_type, .. },
        ) => Some(data_type.clone()),
        ExprNode::Resolved(ResolvedExpr::Function(f)) => Some(f.data_type.clone()),
        ExprNode::Cast { to, .. } => Some(to.clone()),
        ExprNode::Literal(l) => literal_type(l),
        _ => None,
    }
}

/// The type of a literal of the types the allow-list knows (`None` for
/// the others: dates, timestamps, intervals, binary, nested).
fn literal_type(l: &Literal) -> Option<DataType> {
    Some(match l {
        Literal::Null(t) => t.clone().unwrap_or(DataType::Null),
        Literal::Boolean(_) => DataType::Boolean,
        Literal::Byte(_) => DataType::Byte,
        Literal::Short(_) => DataType::Short,
        Literal::Integer(_) => DataType::Integer,
        Literal::Long(_) => DataType::Long,
        Literal::Float(_) => DataType::Float,
        Literal::Double(_) => DataType::Double,
        Literal::Decimal { data_type, .. } => DataType::Decimal(*data_type),
        Literal::String(_) => DataType::String,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use krishiv_types::DecimalType;

    use super::*;
    use crate::expr::{BoundColumn, ResolvedFunction};
    use crate::{AttrId, DayTimeField};

    fn col(name: &str, t: DataType) -> Expr {
        Expr::new(ExprNode::Resolved(ResolvedExpr::Column(BoundColumn {
            id: AttrId(u64::try_from(name.len()).unwrap()),
            name: name.to_owned(),
            data_type: t,
            nullable: true,
        })))
    }

    fn lit(l: Literal) -> Expr {
        Expr::new(ExprNode::Literal(l))
    }

    fn call(id: FunctionId, args: Vec<Expr>, t: DataType) -> Expr {
        Expr::new(ExprNode::Resolved(ResolvedExpr::Function(
            ResolvedFunction {
                id,
                kind: FunctionKind::Scalar,
                args: args.into_iter().map(Arc::new).collect(),
                distinct: false,
                filter: None,
                within_group: Vec::new(),
                data_type: t,
                nullable: true,
                deterministic: true,
            },
        )))
    }

    fn op(b: Builtin, args: Vec<Expr>, t: DataType) -> Expr {
        call(FunctionId::Builtin(b), args, t)
    }

    fn named(name: &str, args: Vec<Expr>, t: DataType) -> Expr {
        call(
            FunctionId::Registry {
                name: name.to_owned(),
                id: 0,
            },
            args,
            t,
        )
    }

    fn cast(e: Expr, to: DataType, mode: CastMode) -> Expr {
        Expr::new(ExprNode::Cast {
            expr: Arc::new(e),
            to,
            mode,
        })
    }

    fn gt0(e: Expr) -> Expr {
        op(
            Builtin::Gt,
            vec![e, lit(Literal::Integer(0))],
            DataType::Boolean,
        )
    }

    fn dec(p: u8, s: u8) -> DataType {
        DataType::Decimal(DecimalType::try_new(u32::from(p), u32::from(s)).unwrap())
    }

    #[test]
    fn predicates_without_arithmetic_or_casts_never_raise() {
        let k = col("k", DataType::Integer);
        let s = col("s", DataType::String);
        let eq = op(
            Builtin::Eq,
            vec![k, lit(Literal::Integer(1))],
            DataType::Boolean,
        );
        let like = op(
            Builtin::Like,
            vec![s.clone(), lit(Literal::String("ab%".to_owned()))],
            DataType::Boolean,
        );
        let isnull = op(Builtin::IsNull, vec![s.clone()], DataType::Boolean);
        let both = op(
            Builtin::Or,
            vec![
                eq.clone(),
                op(Builtin::Not, vec![like.clone()], DataType::Boolean),
            ],
            DataType::Boolean,
        );
        for ansi in [false, true] {
            for e in [&eq, &like, &isnull, &both] {
                assert!(!may_raise(e, ansi), "{e}");
            }
        }
        // A pattern with the escape character may be malformed.
        let escaped = op(
            Builtin::Like,
            vec![s.clone(), lit(Literal::String("ab\\".to_owned()))],
            DataType::Boolean,
        );
        assert!(may_raise(&escaped, false));
        // A column pattern may be malformed.
        let column_pattern = op(Builtin::Like, vec![s.clone(), s], DataType::Boolean);
        assert!(may_raise(&column_pattern, false));
    }

    #[test]
    fn numeric_arithmetic_and_casts_raise_only_under_ansi() {
        let k = col("k", DataType::Integer);
        let s = col("s", DataType::String);
        let sum = gt0(op(
            Builtin::Add,
            vec![k.clone(), lit(Literal::Integer(1))],
            DataType::Integer,
        ));
        let div = gt0(op(
            Builtin::Divide,
            vec![col("d", dec(10, 2)), k.clone()],
            dec(21, 11),
        ));
        assert!(!may_raise(&sum, false));
        assert!(!may_raise(&div, false));
        assert!(may_raise(&sum, true));
        assert!(may_raise(&div, true));
        // CAST(s AS INT): NULL with ANSI off, CAST_INVALID_INPUT with it on.
        assert!(!may_raise(
            &gt0(cast(s.clone(), DataType::Integer, CastMode::Legacy)),
            true
        ));
        assert!(may_raise(
            &gt0(cast(s.clone(), DataType::Integer, CastMode::Ansi)),
            true
        ));
        assert!(!may_raise(
            &gt0(cast(s, DataType::Integer, CastMode::Try)),
            true
        ));
        // Widening casts cannot fail, ANSI or not; narrowing ones can.
        for (from, to, raises) in [
            (DataType::Integer, DataType::Long, false),
            (DataType::Integer, dec(10, 0), false),
            (DataType::Integer, dec(9, 0), true),
            (dec(10, 2), dec(12, 2), false),
            (dec(10, 2), dec(10, 3), true),
            (dec(10, 2), DataType::Double, false),
            (DataType::Long, DataType::Integer, true),
            (DataType::Long, DataType::String, false),
        ] {
            let e = cast(col("x", from.clone()), to.clone(), CastMode::Ansi);
            assert_eq!(may_raise(&e, true), raises, "{from} -> {to}");
        }
        // try_* forms over numbers never raise.
        let try_add = named("try_add", vec![k.clone(), k], DataType::Integer);
        assert!(!may_raise(&gt0(try_add), true));
    }

    /// Spark 4.2 with `spark.sql.ansi.enabled=false` raises on interval,
    /// date and timestamp operands: `CAST('abc' AS INTERVAL DAY)` is
    /// `INVALID_INTERVAL_FORMAT`, `INTERVAL '1' DAY / 0` is
    /// `INTERVAL_DIVIDED_BY_ZERO`, interval multiplication overflows.
    #[test]
    fn interval_date_and_timestamp_operands_may_raise_with_ansi_off() {
        let ts = col("ts", DataType::Timestamp);
        let d = col("d", DataType::Date);
        let day = lit(Literal::DayTimeInterval {
            micros: 86_400_000_000,
            start: DayTimeField::Day,
            end: DayTimeField::Day,
        });
        let plus = op(
            Builtin::Add,
            vec![ts.clone(), day.clone()],
            DataType::Timestamp,
        );
        assert!(may_raise(&plus, false));
        let date_minus = op(
            Builtin::Subtract,
            vec![d.clone(), col("n", DataType::Integer)],
            DataType::Date,
        );
        assert!(may_raise(&date_minus, false));
        let date_add = named(
            "date_add",
            vec![d.clone(), col("n", DataType::Integer)],
            DataType::Date,
        );
        assert!(may_raise(&date_add, false));
        let interval_div = op(
            Builtin::Divide,
            vec![day, col("n", DataType::Integer)],
            DataType::Null,
        );
        assert!(may_raise(&interval_div, false));
        // Casts to or from dates and timestamps, in every mode but TRY.
        for mode in [CastMode::Legacy, CastMode::Ansi] {
            assert!(may_raise(
                &cast(col("s", DataType::String), DataType::Date, mode),
                false
            ));
            assert!(may_raise(
                &cast(d.clone(), DataType::Timestamp, mode),
                false
            ));
            assert!(may_raise(&cast(ts.clone(), DataType::String, mode), false));
        }
    }

    #[test]
    fn unknown_functions_and_scalar_subqueries_may_raise() {
        let s = col("s", DataType::String);
        assert!(may_raise(
            &named("assert_true", vec![s.clone()], DataType::Null),
            false
        ));
        assert!(may_raise(
            &op(
                Builtin::RLike,
                vec![s.clone(), lit(Literal::String("a".to_owned()))],
                DataType::Boolean
            ),
            false
        ));
        assert!(!may_raise(
            &named("startswith", vec![s.clone(), s.clone()], DataType::Boolean),
            false
        ));
        // User code may raise (or fail its contract): never assumed pure.
        let udf = call(
            FunctionId::Udf {
                name: "my_udf".to_owned(),
                version: 1,
            },
            vec![s],
            DataType::Boolean,
        );
        assert!(may_raise(&udf, false));
        assert!(may_raise(&udf, true));
    }

    #[test]
    fn conjuncts_come_in_written_order() {
        let a = col("a", DataType::Boolean);
        let b = col("bb", DataType::Boolean);
        let c = col("ccc", DataType::Boolean);
        let and = |l: Expr, r: Expr| op(Builtin::And, vec![l, r], DataType::Boolean);
        let left = and(and(a.clone(), b.clone()), c.clone());
        let right = and(a.clone(), and(b.clone(), c.clone()));
        for e in [&left, &right] {
            let got: Vec<&Expr> = conjuncts(e);
            assert_eq!(got, vec![&a, &b, &c]);
        }
    }

    /// q09's `amount`: `l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity`,
    /// typed as Spark types it, with the analyzer's `cast(1 AS decimal(1,0))`.
    fn amount() -> Expr {
        let one = cast(lit(Literal::Integer(1)), dec(1, 0), CastMode::Ansi);
        let disc = op(
            Builtin::Subtract,
            vec![one, col("l_discount", dec(15, 2))],
            dec(16, 2),
        );
        let rev = op(
            Builtin::Multiply,
            vec![col("l_extendedprice", dec(15, 2)), disc],
            dec(32, 4),
        );
        let cost = op(
            Builtin::Multiply,
            vec![
                col("ps_supplycost", dec(15, 2)),
                col("l_quantity", dec(15, 2)),
            ],
            dec(31, 4),
        );
        op(Builtin::Subtract, vec![rev, cost], dec(33, 4))
    }

    #[test]
    fn exact_decimal_arithmetic_runs_on_any_row_under_ansi() {
        let a = amount();
        assert!(exact_decimal(&a));
        assert!(may_raise(&a, true));
        for ansi in [false, true] {
            assert!(runs_on_any_row(&a, ansi), "{a}");
        }
        // An integral column widened to a decimal that holds it.
        let widened = op(
            Builtin::Add,
            vec![
                cast(col("k", DataType::Integer), dec(10, 0), CastMode::Ansi),
                col("d", dec(10, 2)),
            ],
            dec(13, 2),
        );
        assert!(exact_decimal(&widened));
    }

    #[test]
    fn inexact_decimal_arithmetic_is_refused() {
        // decimal(38,0) * decimal(2,0): Spark's unadjusted (41,0) is cut to 38.
        let wide = op(
            Builtin::Multiply,
            vec![col("a", dec(38, 0)), col("b", dec(2, 0))],
            dec(38, 0),
        );
        assert!(!exact_decimal(&wide));
        assert!(!runs_on_any_row(&wide, true));
        assert!(runs_on_any_row(&wide, false));
        // A resolved type that is not the exact one (precision loss applied).
        let cut = op(
            Builtin::Add,
            vec![col("a", dec(15, 2)), col("b", dec(15, 2))],
            dec(15, 2),
        );
        assert!(!exact_decimal(&cut));
        // Division never.
        let div = op(
            Builtin::Divide,
            vec![col("a", dec(10, 2)), col("b", dec(10, 2))],
            dec(23, 13),
        );
        assert!(!exact_decimal(&div));
        // A literal that does not fit its cast.
        let big = cast(lit(Literal::Integer(100)), dec(2, 0), CastMode::Ansi);
        assert!(!exact_decimal(&big));
        // Integer arithmetic is not decimal.
        let int = op(
            Builtin::Add,
            vec![col("k", DataType::Integer), lit(Literal::Integer(1))],
            DataType::Integer,
        );
        assert!(!exact_decimal(&int));
        assert!(!runs_on_any_row(&int, true));
    }
}
