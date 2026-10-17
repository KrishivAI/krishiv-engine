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

use arrow_array::types::Decimal128Type;
use arrow_array::{Array, ArrowPrimitiveType, BooleanArray, Decimal128Array, PrimitiveArray};
use arrow_buffer::i256;

use crate::decimal::DecimalPlan;
use crate::error::KernelError;
use crate::float::SparkFloat;
use crate::integer::SparkIntegral;
use crate::ops::{ArithOp, Operand};
use crate::policy::OverflowMode;

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

/// Reference for [`crate::integer::binary`].
///
/// # Errors
/// As the fast path.
pub fn integer_binary<T: SparkIntegral>(
    op: ArithOp,
    mode: OverflowMode,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
) -> Result<PrimitiveArray<T>, KernelError> {
    let len = length(l, r)?;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let (Some(a), Some(b)) = (value(l, i), value(r, i)) else {
            out.push(None);
            continue;
        };
        // i32/i64 operands never overflow i128.
        let exact = exact_i128(op, T::widen(a), T::widen(b)).unwrap_or_default();
        let (wrapped, overflow) = T::narrow(exact);
        out.push(match (mode, overflow) {
            (_, false) | (OverflowMode::Unchecked | OverflowMode::JavaWrap, true) => Some(wrapped),
            (OverflowMode::NullOnOverflow, true) => None,
            (OverflowMode::AnsiError, true) => {
                return Err(KernelError::arithmetic_overflow(i, op));
            }
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

/// A comparison operator, for [`compare`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareKind {
    /// `=`
    Equal,
    /// `<`
    Less,
    /// `<=`
    LessEq,
    /// `>`
    Greater,
    /// `>=`
    GreaterEq,
}

/// Reference for [`crate::float::compare`].
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
    let len = length(l, r)?;
    Ok((0..len)
        .map(|i| {
            let (a, b) = (value(l, i)?, value(r, i)?);
            let o = spark_order(a, b);
            Some(match kind {
                CompareKind::Equal => o.is_eq(),
                CompareKind::Less => o.is_lt(),
                CompareKind::LessEq => o.is_le(),
                CompareKind::Greater => o.is_gt(),
                CompareKind::GreaterEq => o.is_ge(),
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use arrow_array::types::Int64Type;
    use arrow_array::{Float32Array, Int64Array};

    use super::*;

    #[test]
    fn engine_errors() {
        let a = Int64Array::from(vec![1, 2]);
        let b = Int64Array::from(vec![1]);
        let add = |l, r| integer_binary::<Int64Type>(ArithOp::Add, OverflowMode::AnsiError, l, r);
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
