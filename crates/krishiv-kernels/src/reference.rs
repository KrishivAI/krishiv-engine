// SPDX-License-Identifier: Apache-2.0
//! Row-at-a-time reference implementations (the oracle).
//!
//! Each function evaluates one row at a time on `Option` values with the
//! overflow policy as a runtime value, using exact wide arithmetic instead
//! of the fast paths' flags. They define the expected result of every
//! specialization in property tests, the differential fuzz target and the
//! benchmarks (where they are the "generic path"). They are deliberately
//! simple, not fast; production code uses [`crate::integer`],
//! [`crate::decimal`] and [`crate::float`].

use std::cmp::Ordering;

use arrow_array::types::{Decimal128Type, Int64Type};
use arrow_array::{Array, ArrowPrimitiveType, BooleanArray, Decimal128Array, PrimitiveArray};
use arrow_buffer::i256;

use crate::compare::CompareKind;
use crate::decimal::DecimalPlan;
use crate::dispatch::KernelCtx;
use crate::error::KernelError;
use crate::float::SparkFloat;
use crate::integer::{SparkIntegral, UnaryOp};
use crate::ops::{ArithOp, Operand};
use crate::policy::{Action, Fault, OverflowMode};
use crate::version::SparkVersion;

/// Row `i` of an operand (`None` for NULL).
fn value<T: ArrowPrimitiveType>(o: Operand<'_, T>, i: usize) -> Option<T::Native> {
    match o {
        Operand::Array(a) => a.is_valid(i).then(|| a.value(i)),
        Operand::Scalar(v) => v,
    }
}

fn length<T: ArrowPrimitiveType>(
    l: Operand<'_, T>,
    r: Operand<'_, T>,
) -> Result<usize, KernelError> {
    match (l, r) {
        (Operand::Array(a), Operand::Array(b)) if a.len() != b.len() => {
            Err(KernelError::internal("operand lengths differ"))
        }
        (Operand::Array(a), _) | (_, Operand::Array(a)) => Ok(a.len()),
        _ => Err(KernelError::internal("two scalar operands")),
    }
}

fn exact_i128(op: ArithOp, a: i128, b: i128) -> Option<i128> {
    match op {
        ArithOp::Add => a.checked_add(b),
        ArithOp::Subtract => a.checked_sub(b),
        ArithOp::Multiply => a.checked_mul(b),
    }
}

/// What policy `mode` makes of a faulting row: the wrapped value, NULL or
/// the error.
fn settle<T>(
    mode: OverflowMode,
    fault: Fault,
    wrapped: T,
    error: impl FnOnce() -> KernelError,
) -> Result<Option<T>, KernelError> {
    match mode.action(fault) {
        Action::Wrap => Ok(Some(wrapped)),
        Action::Null => Ok(None),
        Action::Raise => Err(error()),
    }
}

/// Row-at-a-time loop over two operands; `f` maps each non-NULL pair.
fn rows<T: ArrowPrimitiveType, U: ArrowPrimitiveType>(
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    mut f: impl FnMut(usize, T::Native, T::Native) -> Result<Option<U::Native>, KernelError>,
) -> Result<PrimitiveArray<U>, KernelError> {
    let len = length(l, r)?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(match (value(l, i), value(r, i)) {
            (Some(a), Some(b)) => f(i, a, b)?,
            _ => None,
        });
    }
    Ok(out.into_iter().collect())
}

/// Reference for [`crate::integer::binary`].
///
/// # Errors
/// As the fast path.
pub fn integer_binary<T: SparkIntegral>(
    op: ArithOp,
    mode: OverflowMode,
    version: SparkVersion,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
) -> Result<PrimitiveArray<T>, KernelError> {
    let ctx = KernelCtx::new().with_spark_version(version);
    rows(l, r, |i, a, b| {
        // Every Spark integral fits an i128 with room for + - *.
        let exact = exact_i128(op, T::widen(a), T::widen(b)).unwrap_or_default();
        match T::narrow(exact) {
            (v, false) => Ok(Some(v)),
            (v, true) => settle(mode, Fault::Overflow, v, || {
                crate::integer::overflow_error::<T>(&ctx, i, op, a, b)
            }),
        }
    })
}

/// Reference for [`crate::integer::remainder`].
///
/// # Errors
/// As the fast path.
pub fn integer_remainder<T: SparkIntegral>(
    mode: OverflowMode,
    version: SparkVersion,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
) -> Result<PrimitiveArray<T>, KernelError> {
    rows(l, r, |i, a, b| {
        let (a, b) = (T::widen(a), T::widen(b));
        if b == 0 {
            return settle(mode, Fault::DivideByZero, T::Native::default(), || {
                KernelError::remainder_by_zero(i, version)
            });
        }
        // Truncated remainder: the sign of the dividend, |result| < |b|.
        Ok(Some(T::narrow(a.checked_rem(b).unwrap_or_default()).0))
    })
}

/// Reference for [`crate::integer::integral_divide`].
///
/// # Errors
/// As the fast path.
pub fn integer_divide(
    mode: OverflowMode,
    l: Operand<'_, Int64Type>,
    r: Operand<'_, Int64Type>,
) -> Result<PrimitiveArray<Int64Type>, KernelError> {
    rows(l, r, |i, a, b| {
        if b == 0 {
            return settle(mode, Fault::DivideByZero, 0, || {
                KernelError::divide_by_zero(i)
            });
        }
        // b != 0 and i64 operands: exact in i128.
        let q = i128::from(a).checked_div(i128::from(b)).unwrap_or_default();
        match Int64Type::narrow(q) {
            (v, false) => Ok(Some(v)),
            (v, true) => settle(mode, Fault::Overflow, v, || {
                KernelError::arithmetic_overflow(
                    i,
                    "Overflow in integral divide",
                    Some("try_divide"),
                )
            }),
        }
    })
}

/// Reference for [`crate::integer::unary`].
///
/// # Errors
/// As the fast path.
pub fn integer_unary<T: SparkIntegral>(
    op: UnaryOp,
    mode: OverflowMode,
    version: SparkVersion,
    a: &PrimitiveArray<T>,
) -> Result<PrimitiveArray<T>, KernelError> {
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        if a.is_null(i) {
            out.push(None);
            continue;
        }
        let v = T::widen(a.value(i));
        let exact = match op {
            UnaryOp::Negate => v.checked_neg(),
            UnaryOp::Abs => v.checked_abs(),
        };
        // Integral values never reach i128::MIN.
        out.push(match T::narrow(exact.unwrap_or_default()) {
            (n, false) => Some(n),
            (n, true) => settle(mode, Fault::Overflow, n, || {
                KernelError::arithmetic_overflow(i, version.overflow_message(T::WIDTH), None)
            })?,
        });
    }
    Ok(out.into_iter().collect())
}

/// Reference for [`crate::decimal::decimal_binary`]: exact 256-bit
/// arithmetic, range-checked by counting digits.
///
/// # Errors
/// As the fast path (policy/plan mismatches excepted).
pub fn decimal_binary(
    plan: &DecimalPlan,
    mode: OverflowMode,
    l: Operand<'_, Decimal128Type>,
    r: Operand<'_, Decimal128Type>,
) -> Result<Decimal128Array, KernelError> {
    let len = length(l, r)?;
    let (op, (_, ls), (_, rs), (out_p, os)) = plan.types();
    let ten = i256::from_i128(10);
    let scale_up = |v: i128, from: i8| {
        let k = u32::try_from(os.saturating_sub(from)).unwrap_or(0);
        i256::from_i128(v).wrapping_mul(ten.checked_pow(k).unwrap_or(i256::ONE))
    };
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let (Some(a), Some(b)) = (value(l, i), value(r, i)) else {
            out.push(None);
            continue;
        };
        let exact = match op {
            ArithOp::Add => scale_up(a, ls).wrapping_add(scale_up(b, rs)),
            ArithOp::Subtract => scale_up(a, ls).wrapping_sub(scale_up(b, rs)),
            ArithOp::Multiply => i256::from_i128(a).wrapping_mul(i256::from_i128(b)),
        };
        let digits = exact.to_string().trim_start_matches('-').len();
        let fits = digits <= usize::from(out_p);
        let fits_i128 = exact.to_i128();
        out.push(match (mode, fits, fits_i128) {
            (_, true, Some(v)) => Some(v),
            (OverflowMode::AnsiError, _, _) => {
                return Err(KernelError::decimal_out_of_range(
                    i,
                    plan.spark_value(exact),
                    out_p,
                    os,
                ));
            }
            _ => None,
        });
    }
    let a: Decimal128Array = out.into_iter().collect();
    a.with_precision_and_scale(out_p, os)
        .map_err(|e| KernelError::internal(e.to_string()))
}

/// Spark's `SQLOrderingUtil.compareDoubles` / `compareFloats`.
fn spark_order<F: SparkFloat>(x: F, y: F) -> Ordering {
    if x == y {
        return Ordering::Equal;
    }
    match (x.is_nan(), y.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => x.partial_cmp(&y).unwrap_or(Ordering::Equal),
    }
}

/// Reference for [`crate::compare::compare`] on floats.
///
/// # Errors
/// As the fast path.
pub fn compare<T>(
    kind: CompareKind,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
) -> Result<BooleanArray, KernelError>
where
    T: ArrowPrimitiveType<Native: SparkFloat>,
{
    compare_by(kind, l, r, spark_order)
}

/// Reference for [`crate::compare::compare`] on integral, decimal, date
/// and timestamp values (their natural order).
///
/// # Errors
/// As the fast path.
pub fn compare_ord<T>(
    kind: CompareKind,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
) -> Result<BooleanArray, KernelError>
where
    T: ArrowPrimitiveType<Native: Ord>,
{
    compare_by(kind, l, r, |a: T::Native, b| a.cmp(&b))
}

fn compare_by<T: ArrowPrimitiveType>(
    kind: CompareKind,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    ord: impl Fn(T::Native, T::Native) -> Ordering,
) -> Result<BooleanArray, KernelError> {
    let len = length(l, r)?;
    Ok((0..len)
        .map(|i| Some(kind.holds(ord(value(l, i)?, value(r, i)?))))
        .collect())
}

/// Reference for [`crate::compare::compare_bytes`]: `None` rows are NULL.
#[must_use]
pub fn compare_bytes(kind: CompareKind, l: &[Option<&[u8]>], r: &[Option<&[u8]>]) -> BooleanArray {
    l.iter()
        .zip(r)
        .map(|(a, b)| Some(kind.holds((*a)?.cmp((*b)?))))
        .collect()
}

/// Reference for [`crate::compare::null_safe`] over row values.
#[must_use]
pub fn null_safe_equal<T: PartialEq>(l: &[Option<T>], r: &[Option<T>]) -> BooleanArray {
    l.iter().zip(r).map(|(a, b)| Some(a == b)).collect()
}

/// Reference for [`crate::float::binary`], [`crate::float::divide`]
/// (`op == None`, `rem == false`) and [`crate::float::remainder`]
/// (`rem == true`).
///
/// # Errors
/// As the fast path.
pub fn float_arith<T>(
    op: Option<ArithOp>,
    rem: bool,
    mode: OverflowMode,
    version: SparkVersion,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
) -> Result<PrimitiveArray<T>, KernelError>
where
    T: ArrowPrimitiveType<Native: SparkFloat>,
{
    rows(l, r, |i, a, b| match op {
        Some(op) => Ok(Some(a.arith(op, b))),
        None if b.is_zero() => settle(mode, Fault::DivideByZero, a, || {
            if rem {
                KernelError::remainder_by_zero(i, version)
            } else {
                KernelError::divide_by_zero(i)
            }
        }),
        None if rem => Ok(Some(a.rem(b))),
        None => Ok(Some(a.div(b))),
    })
}

#[cfg(test)]
mod tests {
    use arrow_array::{Float32Array, Int64Array};

    use super::*;

    #[test]
    fn engine_errors() {
        let a = Int64Array::from(vec![1, 2]);
        let b = Int64Array::from(vec![1]);
        let add = |l, r| {
            integer_binary::<Int64Type>(
                ArithOp::Add,
                OverflowMode::AnsiError,
                SparkVersion::V4_2,
                l,
                r,
            )
        };
        assert!(add(Operand::Array(&a), Operand::Array(&b)).is_err());
        assert!(add(Operand::Scalar(Some(1)), Operand::Scalar(Some(1))).is_err());
        assert_eq!(
            add(Operand::Scalar(Some(1)), Operand::Array(&a)).unwrap(),
            Int64Array::from(vec![2, 3])
        );
    }

    #[test]
    fn float_order() {
        assert_eq!(spark_order(f32::NAN, 1.0), Ordering::Greater);
        assert_eq!(spark_order(-0.0_f64, 0.0), Ordering::Equal);
        assert_eq!(spark_order(1.0_f64, f64::NAN), Ordering::Less);
        let a = Float32Array::from(vec![f32::NAN]);
        let ge = compare(
            CompareKind::GreaterEq,
            Operand::Array(&a),
            Operand::Scalar(Some(f32::INFINITY)),
        );
        assert_eq!(ge.unwrap(), BooleanArray::from(vec![true]));
    }
}
