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
use crate::{CastMode, Expr, ExprNode, Literal};

/// The value of `e` as a constant, when it is a literal or a cast of one.
pub(super) fn constant(e: &Expr, ansi: bool) -> Option<Constant> {
    let folded = match &e.node {
        ExprNode::Literal(Literal::Null(_)) => return Some(Constant::Null),
        ExprNode::Literal(Literal::Boolean(v)) => return Some(Constant::Boolean(*v)),
        ExprNode::Literal(Literal::String(v)) => return Some(Constant::String(v.clone())),
        ExprNode::Literal(_) | ExprNode::Cast { .. } => fold(e, ansi)?,
        _ => return None,
    };
    Some(match folded.and_then(|a| value(&a, ansi)) {
        Ok(c) => c,
        Err(e) => Constant::Error(e),
    })
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
}
