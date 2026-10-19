// SPDX-License-Identifier: Apache-2.0
//! `float`/`double` arithmetic with Spark semantics: `+ - *` ([`binary`]),
//! `/` and `%` ([`divide`], [`remainder`]) and unary `-` / `abs`
//! ([`unary`]). (Comparisons are in [`crate::compare`].)
//!
//! IEEE arithmetic never overflows (it saturates to infinity) and NaN
//! propagates. Spark's one exception is a zero divisor (`0.0` or `-0.0`,
//! whatever the dividend, NaN included): NULL in legacy mode (`JavaWrap` /
//! `NullOnOverflow`) and `DIVIDE_BY_ZERO` / `REMAINDER_BY_ZERO` under
//! `AnsiError`. `%` is Java's (`fmod`: the sign of the dividend).

use arrow_array::{Array, ArrowPrimitiveType, PrimitiveArray};

use crate::chunk::{self, LaneOp, Resolved};
use crate::dispatch::KernelCtx;
use crate::error::KernelError;
use crate::integer::{UnaryOp, finish};
use crate::ops::{ArithOp, Operand};
use crate::policy::{Fault, Overflow, Unchecked};

mod sealed {
    pub trait Sealed {}
}

/// `f32` and `f64`.
pub trait SparkFloat:
    sealed::Sealed + arrow_buffer::ArrowNativeType + PartialOrd + Default
{
    /// Whether the value is a NaN.
    fn is_nan(self) -> bool;
    /// `self op rhs` in IEEE arithmetic.
    fn arith(self, op: ArithOp, rhs: Self) -> Self;
    /// `self / rhs`.
    fn div(self, rhs: Self) -> Self;
    /// `self % rhs` (Java's remainder).
    fn rem(self, rhs: Self) -> Self;
    /// `-self`.
    fn neg(self) -> Self;
    /// `|self|`.
    fn abs(self) -> Self;
    /// Whether the value is `0.0` or `-0.0`.
    fn is_zero(self) -> bool;
}

macro_rules! spark_float {
    ($($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl SparkFloat for $t {
            #[inline(always)]
            fn is_nan(self) -> bool { <$t>::is_nan(self) }
            #[inline(always)]
            fn arith(self, op: ArithOp, rhs: Self) -> Self {
                match op {
                    ArithOp::Add => self + rhs,
                    ArithOp::Subtract => self - rhs,
                    ArithOp::Multiply => self * rhs,
                }
            }
            #[inline(always)]
            fn div(self, rhs: Self) -> Self { self / rhs }
            #[inline(always)]
            fn rem(self, rhs: Self) -> Self { self % rhs }
            #[inline(always)]
            fn neg(self) -> Self { -self }
            #[inline(always)]
            fn abs(self) -> Self { <$t>::abs(self) }
            #[inline(always)]
            fn is_zero(self) -> bool { self == 0.0 }
        }
    )*};
}
spark_float!(f32, f64);

/// `+ - *` as a lane (the operator is a constant of the monomorph).
#[derive(Clone, Copy)]
struct ArithLane<const OP: u8>;

const fn arith_op(op: u8) -> ArithOp {
    match op {
        0 => ArithOp::Add,
        1 => ArithOp::Subtract,
        _ => ArithOp::Multiply,
    }
}

impl<F: SparkFloat, const OP: u8> LaneOp<F> for ArithLane<OP> {
    type Out = F;
    #[inline(always)]
    fn wrapping(self, a: F, b: F) -> F {
        a.arith(arith_op(OP), b)
    }
    #[inline(always)]
    fn checked(self, a: F, b: F) -> (F, bool) {
        (self.wrapping(a, b), false)
    }
    #[inline(always)]
    fn fault(self, _a: F, _b: F) -> Option<Fault> {
        None
    }
}

/// `l op r` for floats (`op` is `+`, `-` or `*`; nothing can fail).
///
/// # Errors
/// `INTERNAL_ERROR` for two scalars or arrays of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be allocated.
pub fn binary<T>(
    op: ArithOp,
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    ctx: &KernelCtx,
) -> Result<PrimitiveArray<T>, KernelError>
where
    T: ArrowPrimitiveType<Native: SparkFloat>,
{
    let (l, r, nulls, len) = match chunk::resolve(l, r)? {
        Resolved::AllNull(len) => return Ok(PrimitiveArray::new_null(len)),
        Resolved::Lanes { l, r, nulls, len } => (l, r, nulls, len),
    };
    let fail = |_, _| KernelError::internal("float + - * cannot fail");
    let c = match op {
        ArithOp::Add => {
            chunk::compute::<_, _, Unchecked>(ctx, l, r, nulls, len, ArithLane::<0>, fail)
        }
        ArithOp::Subtract => {
            chunk::compute::<_, _, Unchecked>(ctx, l, r, nulls, len, ArithLane::<1>, fail)
        }
        ArithOp::Multiply => {
            chunk::compute::<_, _, Unchecked>(ctx, l, r, nulls, len, ArithLane::<2>, fail)
        }
    }?;
    finish(c)
}

/// `/` (`REM == false`) or `%` (`REM == true`) with Spark's zero-divisor rule.
#[derive(Clone, Copy)]
struct DivLane<const REM: bool>;

impl<F: SparkFloat, const REM: bool> LaneOp<F> for DivLane<REM> {
    type Out = F;
    const MAY_DIVIDE_BY_ZERO: bool = true;
    #[inline(always)]
    fn wrapping(self, a: F, b: F) -> F {
        if REM { a.rem(b) } else { a.div(b) }
    }
    #[inline(always)]
    fn checked(self, a: F, b: F) -> (F, bool) {
        (self.wrapping(a, b), b.is_zero())
    }
    #[inline(always)]
    fn fault(self, _a: F, b: F) -> Option<Fault> {
        b.is_zero().then_some(Fault::DivideByZero)
    }
}

/// `l / r`: a zero divisor is NULL under `JavaWrap`/`NullOnOverflow` (also
/// `try_divide`) and raises under `AnsiError`/`Unchecked`.
///
/// # Errors
/// `DIVIDE_BY_ZERO` with the row; engine and memory errors as [`binary`].
pub fn divide<T, O>(
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    ctx: &KernelCtx,
) -> Result<PrimitiveArray<T>, KernelError>
where
    T: ArrowPrimitiveType<Native: SparkFloat>,
    O: Overflow,
{
    let (l, r, nulls, len) = match chunk::resolve(l, r)? {
        Resolved::AllNull(len) => return Ok(PrimitiveArray::new_null(len)),
        Resolved::Lanes { l, r, nulls, len } => (l, r, nulls, len),
    };
    let c = chunk::compute::<_, _, O>(ctx, l, r, nulls, len, DivLane::<false>, |row, _| {
        KernelError::divide_by_zero(row)
    })?;
    finish(c)
}

/// `l % r`, with [`divide`]'s zero-divisor rule.
///
/// # Errors
/// `REMAINDER_BY_ZERO` (Spark 4.0: `DIVIDE_BY_ZERO`) with the row; engine
/// and memory errors as [`binary`].
pub fn remainder<T, O>(
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    ctx: &KernelCtx,
) -> Result<PrimitiveArray<T>, KernelError>
where
    T: ArrowPrimitiveType<Native: SparkFloat>,
    O: Overflow,
{
    let (l, r, nulls, len) = match chunk::resolve(l, r)? {
        Resolved::AllNull(len) => return Ok(PrimitiveArray::new_null(len)),
        Resolved::Lanes { l, r, nulls, len } => (l, r, nulls, len),
    };
    let c = chunk::compute::<_, _, O>(ctx, l, r, nulls, len, DivLane::<true>, |row, _| {
        KernelError::remainder_by_zero(row, ctx.spark_version())
    })?;
    finish(c)
}

/// `-a` or `abs(a)` (never fails; `abs(-0.0) = 0.0`).
///
/// # Errors
/// `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be allocated.
pub fn unary<T>(op: UnaryOp, a: &PrimitiveArray<T>) -> Result<PrimitiveArray<T>, KernelError>
where
    T: ArrowPrimitiveType<Native: SparkFloat>,
{
    let mut out = crate::error::with_capacity::<T::Native>(a.len())?;
    out.extend(a.values().iter().map(|&v| match op {
        UnaryOp::Negate => v.neg(),
        UnaryOp::Abs => v.abs(),
    }));
    finish(chunk::Computed {
        values: out,
        nulls: a.nulls().cloned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lanes_follow_ieee_and_java() {
        assert_eq!(ArithLane::<1>.wrapping(1.0_f64, 3.0), -2.0);
        assert_eq!(DivLane::<true>.wrapping(-7.5_f64, 2.0), -1.5);
        assert!(DivLane::<false>.checked(1.0_f32, -0.0).1);
        assert_eq!(LaneOp::<f64>::fault(DivLane::<false>, 1.0, f64::NAN), None);
        assert_eq!(LaneOp::<f64>::fault(ArithLane::<2>, 1.0, 0.0), None);
        assert_eq!(ArithLane::<0>.checked(1.0_f64, 2.0), (3.0, false));
    }
}
