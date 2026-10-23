// SPDX-License-Identifier: Apache-2.0
//! The value of a constant argument, for the functions whose typing reads
//! it (`round`'s scale, `extract`'s field).
//!
//! Spark folds a foldable argument before it types the call, so
//! `round(x, CAST(1.5 AS FLOAT))` has scale 1. Literals and casts of
//! literals to a number fold here, through the same cast kernels that
//! evaluate them (so overflow, malformed input and ANSI/legacy behaviour
//! are the kernels'). A cast that fails folds to [`Constant::Error`]: the
//! function that reads the value raises it, any other ignores it, so an
//! unselected branch never raises.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Decimal128Array, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, StringArray, new_null_array,
};
use arrow_schema::DataType as ArrowType;
use krishiv_common::error::{ErrorClass, KrishivError};
use krishiv_kernels::cast::{CastMode as KernelMode, CastOptions, FixedOffset, cast};
use krishiv_types::DataType;

use super::Constant;
use crate::{Builtin, CastMode, Expr, ExprNode, FunctionId, FunctionKind, Literal, ResolvedExpr};

/// The value of `e` as a constant, when it is a literal, a cast of one,
/// or integral `+`/`-`/`*`/unary `-` over constants (what a `LIMIT`
/// count folds before Spark's `checkLimitLikeClause`).
pub(super) fn constant(e: &Expr, ansi: bool) -> Option<Constant> {
    if let ExprNode::Resolved(ResolvedExpr::Function(f)) = &e.node {
        let op = match &f.id {
            FunctionId::Builtin(op) => *op,
            _ => return None,
        };
        match (op, f.args.as_slice()) {
            (Builtin::Add | Builtin::Subtract | Builtin::Multiply, [a, b]) => {
                return arith(op, a, b, ansi);
            }
            (Builtin::Negative, [a]) => return negative(a, ansi),
            _ => return None,
        }
    }
    let folded = match &e.node {
        ExprNode::Literal(Literal::Null(_)) => return Some(Constant::Null),
        ExprNode::Literal(Literal::Boolean(v)) => return Some(Constant::Boolean(*v)),
        ExprNode::Literal(Literal::String(v)) => return Some(Constant::String(v.clone())),
        ExprNode::Literal(_) => fold(e, ansi)?,
        ExprNode::Cast { expr, to, mode } => match fold(e, ansi) {
            Some(folded) => folded,
            None => {
                // A cast of an arithmetic constant (`CAST(1 - 2 AS INT)`):
                // fold the inner constant through the cast kernels so
                // overflow and ANSI/legacy rules match a cast of a literal.
                match constant(expr, ansi)? {
                    Constant::Null => return Some(Constant::Null),
                    Constant::Error(err) => return Some(Constant::Error(err)),
                    Constant::Integral(v) => {
                        let inner = Expr::new(ExprNode::Literal(Literal::Long(v)));
                        let tmp = Expr::new(ExprNode::Cast {
                            expr: Arc::new(inner),
                            to: to.clone(),
                            mode: *mode,
                        });
                        fold(&tmp, ansi)?
                    }
                    _ => return None,
                }
            }
        },
        _ => return None,
    };
    Some(match folded.and_then(|a| value(&a, ansi)) {
        Ok(c) => c,
        Err(e) => Constant::Error(e),
    })
}

/// Whether `e` is foldable for Spark's `checkLimitLikeClause`: deterministic
/// with no input references, so it can be evaluated without rows. Columns,
/// subqueries and nondeterministic or aggregate calls are not foldable.
/// Iterative over an explicit stack so deep expressions do not overflow the
/// call stack (the analyzer resolves deep plans on a small stack).
pub(super) fn is_foldable(e: &Expr) -> bool {
    let mut stack = vec![e];
    while let Some(x) = stack.pop() {
        match &x.node {
            ExprNode::Literal(_) => {}
            ExprNode::Cast { expr, .. }
            | ExprNode::Alias { expr, .. }
            | ExprNode::Resolved(ResolvedExpr::Alias { expr, .. }) => {
                stack.push(expr);
            }
            ExprNode::Resolved(ResolvedExpr::Function(f)) => {
                if f.kind != FunctionKind::Scalar
                    || !f.deterministic
                    || f.filter.is_some()
                    || f.distinct
                {
                    return false;
                }
                stack.extend(f.args.iter().map(|a| &**a));
            }
            _ => return false,
        }
    }
    true
}

/// Integral `+`/`-`/`*` over constants, with NULL propagation. Overflow is
/// not folded (the expression stays foldable but unevaluated, as before).
fn arith(op: Builtin, a: &Expr, b: &Expr, ansi: bool) -> Option<Constant> {
    let (x, y) = match (constant(a, ansi)?, constant(b, ansi)?) {
        (Constant::Null, _) | (_, Constant::Null) => return Some(Constant::Null),
        (Constant::Error(e), _) | (_, Constant::Error(e)) => {
            return Some(Constant::Error(e));
        }
        (Constant::Integral(x), Constant::Integral(y)) => (x, y),
        _ => return None,
    };
    let v = match op {
        Builtin::Add => x.checked_add(y),
        Builtin::Subtract => x.checked_sub(y),
        _ => x.checked_mul(y),
    }?;
    Some(Constant::Integral(v))
}

/// Unary `-` over a constant, with NULL propagation.
fn negative(a: &Expr, ansi: bool) -> Option<Constant> {
    match constant(a, ansi)? {
        Constant::Null => Some(Constant::Null),
        Constant::Error(e) => Some(Constant::Error(e)),
        Constant::Integral(v) => v.checked_neg().map(Constant::Integral),
        _ => None,
    }
}

/// A folded array as a constant. A fractional value is what the implicit
/// cast to `INT` of an integral argument gives (the scale of `round`).
fn value(a: &ArrayRef, ansi: bool) -> Result<Constant, KrishivError> {
    if a.null_count() > 0 || a.data_type() == &ArrowType::Null {
        return Ok(Constant::Null);
    }
    Ok(match a.data_type() {
        ArrowType::Boolean => Constant::Boolean(a.as_boolean().value(0)),
        ArrowType::Int8 => Constant::Integral(i64::from(a.as_primitive::<Int8Type>().value(0))),
        ArrowType::Int16 => Constant::Integral(i64::from(a.as_primitive::<Int16Type>().value(0))),
        ArrowType::Int32 => Constant::Integral(i64::from(a.as_primitive::<Int32Type>().value(0))),
        ArrowType::Int64 => Constant::Integral(a.as_primitive::<Int64Type>().value(0)),
        ArrowType::Float32 | ArrowType::Float64 | ArrowType::Decimal128(..) => {
            let int = cast(
                a.as_ref(),
                &ArrowType::Int32,
                &options(mode(ansi), &FixedOffset(0)),
            )
            .map_err(KrishivError::from)?;
            if int.null_count() > 0 {
                Constant::Null
            } else {
                Constant::Fractional(i64::from(int.as_primitive::<Int32Type>().value(0)))
            }
        }
        _ => return Err(unsupported()),
    })
}

fn mode(ansi: bool) -> KernelMode {
    if ansi {
        KernelMode::Ansi
    } else {
        KernelMode::Legacy
    }
}

fn options<'a>(mode: KernelMode, zone: &'a FixedOffset) -> CastOptions<'a> {
    CastOptions {
        mode,
        time_zone: zone,
    }
}

fn unsupported() -> KrishivError {
    krishiv_common::error::KrishivUnsupportedFeature {
        feature: "folding this constant".to_owned(),
    }
    .into()
}

/// `e` as a one-row array: `None` when it is not a literal, a cast of one
/// to a number, or a cast the kernels do not implement.
fn fold(e: &Expr, ansi: bool) -> Option<Result<ArrayRef, KrishivError>> {
    match &e.node {
        ExprNode::Literal(l) => literal(l).map(Ok),
        ExprNode::Cast { expr, to, mode: m } => {
            if !matches!(
                to,
                DataType::Byte
                    | DataType::Short
                    | DataType::Integer
                    | DataType::Long
                    | DataType::Float
                    | DataType::Double
                    | DataType::Decimal(_)
            ) {
                return None;
            }
            let inner = match fold(expr, ansi)? {
                Ok(a) => a,
                Err(e) => return Some(Err(e)),
            };
            // A NULL of any type casts to a NULL of the target type.
            if inner.null_count() > 0 || inner.data_type() == &ArrowType::Null {
                return Some(Ok(new_null_array(&to.to_arrow(), 1)));
            }
            let kernel_mode = match m {
                CastMode::Try => KernelMode::Try,
                CastMode::Legacy => KernelMode::Legacy,
                CastMode::Ansi => KernelMode::Ansi,
                _ => mode(ansi),
            };
            match cast(
                inner.as_ref(),
                &to.to_arrow(),
                &options(kernel_mode, &FixedOffset(0)),
            ) {
                Ok(a) => Some(Ok(a)),
                Err(err) => {
                    let err = KrishivError::from(err);
                    (err.error_class() != ErrorClass::KrishivUnsupportedFeature).then_some(Err(err))
                }
            }
        }
        _ => None,
    }
}

/// A literal as a one-row array of its Spark type's carrier.
fn literal(l: &Literal) -> Option<ArrayRef> {
    Some(match l {
        Literal::Null(Some(t)) => new_null_array(&t.to_arrow(), 1),
        Literal::Null(None) => new_null_array(&ArrowType::Null, 1),
        Literal::Boolean(v) => Arc::new(BooleanArray::from(vec![*v])),
        Literal::Byte(v) => Arc::new(Int8Array::from(vec![*v])),
        Literal::Short(v) => Arc::new(Int16Array::from(vec![*v])),
        Literal::Integer(v) => Arc::new(Int32Array::from(vec![*v])),
        Literal::Long(v) => Arc::new(Int64Array::from(vec![*v])),
        Literal::Float(v) => Arc::new(Float32Array::from(vec![*v])),
        Literal::Double(v) => Arc::new(Float64Array::from(vec![*v])),
        Literal::Decimal {
            unscaled,
            data_type,
        } => Arc::new(
            Decimal128Array::from(vec![*unscaled])
                .with_precision_and_scale(
                    data_type.precision(),
                    i8::try_from(data_type.scale()).ok()?,
                )
                .ok()?,
        ),
        Literal::String(v) => Arc::new(StringArray::from(vec![v.as_str()])),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use krishiv_common::error::ErrorClass;
    use krishiv_types::DecimalType;

    use super::*;

    fn lit(v: Literal) -> Expr {
        Expr::new(ExprNode::Literal(v))
    }

    fn cast(v: Literal, to: DataType) -> Expr {
        cast_of(lit(v), to)
    }

    fn cast_of(inner: Expr, to: DataType) -> Expr {
        Expr::new(ExprNode::Cast {
            expr: Arc::new(inner),
            to,
            mode: CastMode::Default,
        })
    }

    fn dec(unscaled: i128, p: u32, s: u32) -> Literal {
        Literal::Decimal {
            unscaled,
            data_type: DecimalType::try_new(p, s).unwrap(),
        }
    }

    fn class(c: Option<Constant>) -> Option<ErrorClass> {
        match c {
            Some(Constant::Error(e)) => Some(e.error_class()),
            _ => None,
        }
    }

    /// Spark 4.2: `round(x, CAST(1.5 AS FLOAT))` has scale 1.
    #[test]
    fn casts_of_literals_to_numbers_fold() {
        let n = |v: i64| Some(Constant::Integral(v));
        let f = |v: i64| Some(Constant::Fractional(v));
        let ansi = true;
        assert_eq!(
            constant(&cast(Literal::Integer(7), DataType::Byte), ansi),
            n(7)
        );
        assert_eq!(
            constant(&cast(Literal::Byte(-3), DataType::Long), ansi),
            n(-3)
        );
        assert_eq!(constant(&cast(dec(15, 2, 1), DataType::Float), ansi), f(1));
        assert_eq!(
            constant(&cast(dec(-15, 2, 1), DataType::Double), ansi),
            f(-1)
        );
        assert_eq!(
            constant(&cast(Literal::Double(2.9), DataType::Float), ansi),
            f(2)
        );
        // A decimal target rounds half up first: 1.5 is 2 at scale 0.
        assert_eq!(
            constant(
                &cast(
                    dec(15, 2, 1),
                    DataType::Decimal(DecimalType::try_new(10, 0).unwrap())
                ),
                ansi
            ),
            f(2)
        );
        assert_eq!(
            constant(
                &cast(
                    dec(15, 2, 1),
                    DataType::Decimal(DecimalType::try_new(10, 2).unwrap())
                ),
                ansi
            ),
            f(1)
        );
        // Strings parse as the cast kernel parses them.
        assert_eq!(
            constant(
                &cast(Literal::String("2.7".to_owned()), DataType::Float),
                ansi
            ),
            f(2)
        );
        assert_eq!(
            constant(
                &cast(Literal::String(" 12 ".to_owned()), DataType::Integer),
                ansi
            ),
            n(12)
        );
        // Casts nest.
        let inner = cast(Literal::Integer(3), DataType::Float);
        assert_eq!(constant(&cast_of(inner, DataType::Double), ansi), f(3));
        // A NULL stays NULL through any cast.
        assert_eq!(
            constant(&cast(Literal::Null(None), DataType::Float), ansi),
            Some(Constant::Null)
        );
        // Literals alone: a fractional one is the INT it casts to.
        assert_eq!(constant(&lit(Literal::Double(-2.5)), ansi), f(-2));
        assert_eq!(constant(&lit(dec(15, 2, 1)), ansi), f(1));
        assert_eq!(constant(&lit(Literal::Long(5)), ansi), n(5));
    }

    /// A cast Spark raises on is an error value, raised by the function
    /// that reads it; legacy mode follows the kernels' legacy rules.
    #[test]
    fn failing_casts_fold_to_errors() {
        let ansi = true;
        assert_eq!(
            class(constant(&cast(Literal::Integer(300), DataType::Byte), ansi)),
            Some(ErrorClass::CastOverflow)
        );
        assert_eq!(
            class(constant(
                &cast(Literal::Double(1e30), DataType::Integer),
                ansi
            )),
            Some(ErrorClass::CastOverflow)
        );
        assert_eq!(
            class(constant(&lit(Literal::Double(1e30)), ansi)),
            Some(ErrorClass::CastOverflow)
        );
        assert_eq!(
            class(constant(
                &cast(Literal::String("x".to_owned()), DataType::Float),
                ansi
            )),
            Some(ErrorClass::CastInvalidInput)
        );
        assert_eq!(
            class(constant(
                &cast(
                    dec(995, 3, 1),
                    DataType::Decimal(DecimalType::try_new(2, 0).unwrap())
                ),
                ansi
            )),
            Some(ErrorClass::NumericValueOutOfRangeWithSuggestion)
        );
        // Not ANSI: the malformed string is NULL, the wrapped byte is 44.
        assert_eq!(
            constant(
                &cast(Literal::String("x".to_owned()), DataType::Float),
                false
            ),
            Some(Constant::Null)
        );
        assert_eq!(
            constant(&cast(Literal::Integer(300), DataType::Byte), false),
            Some(Constant::Integral(44))
        );
    }

    /// What is not a literal or a cast of one to a number is not folded.
    #[test]
    fn other_expressions_are_not_constants() {
        let ansi = true;
        assert_eq!(
            constant(&cast(Literal::Integer(1), DataType::String), ansi),
            None
        );
        let col = Expr::new(ExprNode::Column(crate::ColumnRef {
            name: vec!["a".to_owned()],
            plan_id: None,
        }));
        assert_eq!(constant(&col, ansi), None);
        assert_eq!(constant(&cast_of(col, DataType::Float), ansi), None);
    }

    fn resolved_arith(op: crate::Builtin, args: Vec<Expr>, data_type: DataType) -> Expr {
        use crate::{FunctionId, FunctionKind, ResolvedExpr, ResolvedFunction};
        let f = ResolvedFunction {
            id: FunctionId::Builtin(op),
            kind: FunctionKind::Scalar,
            args: args.into_iter().map(std::sync::Arc::new).collect(),
            distinct: false,
            filter: None,
            data_type,
            nullable: false,
            deterministic: true,
        };
        Expr::new(ExprNode::Resolved(ResolvedExpr::Function(f)))
    }

    fn sub(a: Expr, b: Expr) -> Expr {
        resolved_arith(crate::Builtin::Subtract, vec![a, b], DataType::Integer)
    }

    /// Integral `+`/`-`/`*`/unary `-` fold, with NULL propagation and error
    /// passthrough; overflow stays unfoldered as before.
    #[test]
    fn arithmetic_constants_fold_for_limit_checks() {
        use crate::Builtin;
        let ansi = true;
        let n = |v: i64| Some(Constant::Integral(v));
        let one = lit(Literal::Integer(1));
        let two = lit(Literal::Integer(2));
        // `1 - 2` is -1, `1 + 2` is 3, `2 * 3` is 6.
        assert_eq!(constant(&sub(one.clone(), two.clone()), ansi), n(-1));
        assert!(is_foldable(&sub(one.clone(), two.clone())));
        let add = resolved_arith(
            Builtin::Add,
            vec![one.clone(), two.clone()],
            DataType::Integer,
        );
        assert_eq!(constant(&add, ansi), n(3));
        let mul = resolved_arith(
            Builtin::Multiply,
            vec![two, lit(Literal::Integer(3))],
            DataType::Integer,
        );
        assert_eq!(constant(&mul, ansi), n(6));
        // `NULL + 1` stays NULL.
        let null = cast(Literal::Null(None), DataType::Integer);
        assert_eq!(
            constant(&sub(null.clone(), one.clone()), ansi),
            Some(Constant::Null)
        );
        assert!(is_foldable(&sub(null, one.clone())));
        // An error inside stays an error for the reader to raise.
        let bad = cast(Literal::String("x".to_owned()), DataType::Integer);
        assert!(matches!(
            constant(&sub(bad.clone(), one.clone()), ansi),
            Some(Constant::Error(_))
        ));
        assert!(is_foldable(&sub(bad, one.clone())));
        // Overflow is not folded, but stays foldable.
        let max = lit(Literal::Long(i64::MAX));
        let overflow = sub(max, lit(Literal::Integer(-1)));
        assert_eq!(constant(&overflow, ansi), None);
        assert!(is_foldable(&overflow));
        // A non-integral arithmetic result is not a limit constant, but the
        // count is still foldable.
        let boolean = resolved_arith(
            Builtin::Add,
            vec![lit(Literal::Boolean(true)), one.clone()],
            DataType::Boolean,
        );
        assert_eq!(constant(&boolean, ansi), None);
        assert!(is_foldable(&boolean));
        // Unary minus folds as well, including NULL and error passthrough.
        let neg = resolved_arith(Builtin::Negative, vec![one], DataType::Integer);
        assert_eq!(constant(&neg, ansi), n(-1));
        assert!(is_foldable(&neg));
        let neg_null = resolved_arith(
            Builtin::Negative,
            vec![cast(Literal::Null(None), DataType::Integer)],
            DataType::Integer,
        );
        assert_eq!(constant(&neg_null, ansi), Some(Constant::Null));
        let neg_bad = resolved_arith(
            Builtin::Negative,
            vec![cast(Literal::String("x".to_owned()), DataType::Integer)],
            DataType::Integer,
        );
        assert!(matches!(constant(&neg_bad, ansi), Some(Constant::Error(_))));
        let neg_bool = resolved_arith(
            Builtin::Negative,
            vec![lit(Literal::Boolean(true))],
            DataType::Boolean,
        );
        assert_eq!(constant(&neg_bool, ansi), None);
        assert!(is_foldable(&neg_bool));
    }

    /// What `LIMIT` accepts as foldable: literals and deterministic scalar
    /// calls of foldable arguments, nothing with rows or state.
    #[test]
    fn limit_foldability_needs_no_rows_or_state() {
        assert!(is_foldable(&lit(Literal::Integer(1))));
        assert!(is_foldable(&cast(Literal::Integer(1), DataType::Integer)));
        let one = lit(Literal::Integer(1));
        assert!(is_foldable(&sub(one.clone(), one.clone())));
        let col = Expr::new(ExprNode::Column(crate::ColumnRef {
            name: vec!["a".to_owned()],
            plan_id: None,
        }));
        assert!(!is_foldable(&col));
        assert!(!is_foldable(&cast_of(col, DataType::Integer)));
        // A cast that the kernels do not implement is still foldable when its
        // input is, but has no constant.
        let to_string = cast_of(one, DataType::String);
        assert!(is_foldable(&to_string));
        assert_eq!(constant(&to_string, true), None);
        // A cast of a non-numeric literal has no constant either, but stays
        // foldable.
        let bool_to_string = cast_of(lit(Literal::Boolean(true)), DataType::String);
        assert!(is_foldable(&bool_to_string));
        assert_eq!(constant(&bool_to_string, true), None);
    }
}
