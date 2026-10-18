// SPDX-License-Identifier: Apache-2.0
//! `pmod(a, n)`: the positive remainder, `((a % n) + n) % n` when `a % n` is
//! negative and `a % n` otherwise. A zero divisor raises `REMAINDER_BY_ZERO`
//! under ANSI (`DIVIDE_BY_ZERO` in Spark 4.0) and is NULL otherwise.

use arrow_array::types::Decimal128Type;
use arrow_array::{Array, ArrowPrimitiveType, Decimal128Array, PrimitiveArray};
use arrow_buffer::{ScalarBuffer, i256};

use crate::decimal::{DecimalOp, DecimalPlan, DecimalType, decimal_arith, result_type};
use crate::dispatch::KernelCtx;
use crate::error::KernelError;
use crate::float::SparkFloat;
use crate::integer::SparkIntegral;
use crate::ops::Operand;
use crate::policy::{AnsiError, NullOnOverflow, OverflowMode};
use crate::version::IntWidth;

/// Row `i` of an operand.
fn at<T: ArrowPrimitiveType>(o: Operand<'_, T>, i: usize) -> Option<T::Native> {
    match o {
        Operand::Array(a) => a.is_valid(i).then(|| a.value(i)),
        Operand::Scalar(v) => v,
    }
}

fn rows<T: ArrowPrimitiveType>(l: Operand<'_, T>, r: Operand<'_, T>) -> Result<usize, KernelError> {
    match (l, r) {
        (Operand::Array(a), Operand::Array(b)) if a.len() != b.len() => Err(KernelError::internal(
            format!("pmod operands differ in length: {} vs {}", a.len(), b.len()),
        )),
        (Operand::Array(a), _) | (_, Operand::Array(a)) => Ok(a.len()),
        (Operand::Scalar(_), Operand::Scalar(_)) => Ok(1),
    }
}

/// `pmod` of two integral operands.
///
/// # Errors
/// `REMAINDER_BY_ZERO` (`DIVIDE_BY_ZERO` in Spark 4.0) under
/// `OverflowMode::AnsiError`; `INTERNAL_ERROR` for columns of different lengths.
pub fn pmod_integral<T: SparkIntegral>(
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    overflow: OverflowMode,
    ctx: &KernelCtx,
) -> Result<PrimitiveArray<T>, KernelError> {
    let n = rows(l, r)?;
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| KernelError::out_of_memory(n))?;
    for i in 0..n {
        out.push(match (at(l, i), at(r, i)) {
            (Some(a), Some(d)) => {
                let (a, d) = (T::widen(a), T::widen(d));
                if d == 0 {
                    if overflow == OverflowMode::AnsiError {
                        return Err(KernelError::remainder_by_zero(i, ctx.spark_version()));
                    }
                    None
                } else {
                    let rem = a.checked_rem(d).unwrap_or(0);
                    // |rem| < |d|, so the result fits the type.
                    let v = if rem < 0 {
                        // Spark adds in Java `int` / `long`: the sum wraps at
                        // 32 and 64 bits (tinyint and smallint widen to int).
                        let sum = rem.wrapping_add(d);
                        let sum = match T::WIDTH {
                            IntWidth::Int | IntWidth::Long => T::widen(T::narrow(sum).0),
                            IntWidth::Byte | IntWidth::Short => sum,
                        };
                        sum.checked_rem(d).unwrap_or(0)
                    } else {
                        rem
                    };
                    Some(T::narrow(v).0)
                }
            }
            _ => None,
        });
    }
    Ok(out.into_iter().collect())
}

/// `pmod` of two `float` or `double` operands (IEEE remainder, Java's `%`).
///
/// # Errors
/// As [`pmod_integral`].
pub fn pmod_float<T>(
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    overflow: OverflowMode,
    ctx: &KernelCtx,
) -> Result<PrimitiveArray<T>, KernelError>
where
    T: ArrowPrimitiveType,
    T::Native: SparkFloat,
{
    let n = rows(l, r)?;
    let zero = T::Native::default();
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| KernelError::out_of_memory(n))?;
    for i in 0..n {
        out.push(match (at(l, i), at(r, i)) {
            (Some(a), Some(d)) => {
                if d.is_zero() {
                    if overflow == OverflowMode::AnsiError {
                        return Err(KernelError::remainder_by_zero(i, ctx.spark_version()));
                    }
                    None
                } else {
                    let rem = a.rem(d);
                    Some(if rem < zero {
                        rem.arith(crate::ops::ArithOp::Add, d).rem(d)
                    } else {
                        rem
                    })
                }
            }
            _ => None,
        });
    }
    Ok(out.into_iter().collect())
}

fn pow10_256(k: u32) -> i256 {
    (0..k).fold(i256::ONE, |acc, _| acc.wrapping_mul(i256::from_i128(10)))
}

/// `pmod` of two decimal operands; the result type is that of `%`.
///
/// # Errors
/// As `decimal_arith` for `%`; `NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION`
/// under `AnsiError` (NULL otherwise) when the positive remainder does not
/// fit the result type.
pub fn pmod_decimal(
    l: Operand<'_, Decimal128Type>,
    r: Operand<'_, Decimal128Type>,
    left: DecimalType,
    right: DecimalType,
    allow_precision_loss: bool,
    overflow: OverflowMode,
    ctx: &KernelCtx,
) -> Result<Decimal128Array, KernelError> {
    let out = result_type(DecimalOp::Remainder, left, right, allow_precision_loss)?;
    let plan = DecimalPlan::new(DecimalOp::Remainder, left, right, out)?;
    let rem = if overflow == OverflowMode::AnsiError {
        decimal_arith::<AnsiError>(l, r, &plan, ctx)?
    } else {
        decimal_arith::<NullOnOverflow>(l, r, &plan, ctx)?
    };
    let scale = u32::from(out.1.unsigned_abs());
    let divisor_scale = u32::from(right.1.unsigned_abs());
    let limit = pow10_256(u32::from(out.0));
    let mut values = Vec::new();
    values
        .try_reserve_exact(rem.len())
        .map_err(|_| KernelError::out_of_memory(rem.len()))?;
    let mut valid = Vec::new();
    valid
        .try_reserve_exact(rem.len())
        .map_err(|_| KernelError::out_of_memory(rem.len()))?;
    for (i, v) in rem.values().iter().enumerate() {
        let divisor = at(r, i);
        let (value, ok) = match divisor {
            Some(d) if rem.is_valid(i) && *v < 0 => {
                let n =
                    i256::from_i128(d).wrapping_mul(pow10_256(scale.saturating_sub(divisor_scale)));
                let fixed = i256::from_i128(*v).wrapping_add(n).wrapping_rem(n);
                if fixed.wrapping_abs() < limit {
                    (fixed.to_i128().unwrap_or(0), true)
                } else if overflow == OverflowMode::AnsiError {
                    return Err(KernelError::decimal_out_of_range(
                        i,
                        fixed.to_i128().map_or_else(
                            || fixed.to_string(),
                            |v| super::round::plain_string(v, i32::from(out.1)),
                        ),
                        out.0,
                        out.1,
                    ));
                } else {
                    (0, false)
                }
            }
            _ => (*v, rem.is_valid(i)),
        };
        values.push(value);
        valid.push(ok);
    }
    let nulls = arrow_buffer::NullBuffer::from(valid);
    Decimal128Array::new(ScalarBuffer::from(values), Some(nulls))
        .with_precision_and_scale(out.0, out.1)
        .map_err(|e| KernelError::internal(e.to_string()))
}
