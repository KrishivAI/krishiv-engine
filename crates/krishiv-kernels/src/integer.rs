// SPDX-License-Identifier: Apache-2.0
//! Integral arithmetic for Spark's `tinyint`, `smallint`, `int` and
//! `bigint`: `+ - *` ([`binary`]), `%` ([`remainder`]), `div`
//! ([`integral_divide`]) and unary `-` / `abs` ([`unary`]).
//!
//! Policies (chosen by the planner from the function and
//! `spark.sql.ansi.enabled`): `JavaWrap` is Spark's legacy behaviour (wrap;
//! a zero divisor gives NULL), `AnsiError` raises at the first failing
//! non-NULL row, `NullOnOverflow` is `try_add` and friends, and `Unchecked`
//! is for plans that proved overflow impossible. Integral `/` is not here:
//! Spark divides integers as `double`.
//!
//! Errors follow the context's [`crate::version::SparkVersion`]:
//! `ARITHMETIC_OVERFLOW` (`int`/`bigint`, unary ops, `div`),
//! `BINARY_ARITHMETIC_OVERFLOW` (`tinyint`/`smallint` `+ - *`),
//! `DIVIDE_BY_ZERO` (`div`) and `REMAINDER_BY_ZERO` (`%`).

use std::marker::PhantomData;

use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
use arrow_array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow_buffer::{NullBuffer, ScalarBuffer};

use crate::chunk::{self, LaneOp, Resolved, Side};
use crate::dispatch::KernelCtx;
use crate::error::KernelError;
use crate::ops::{ArithOp, BinaryOp, IntNative, Operand};
use crate::policy::{Fault, Overflow};
use crate::version::IntWidth;

mod sealed {
    pub trait Sealed {}
}

/// Arrow types of Spark's integral types.
pub trait SparkIntegral: sealed::Sealed + ArrowPrimitiveType<Native: IntNative> {
    /// The Spark type's width.
    const WIDTH: IntWidth;
    /// The value as an `i128` (exact).
    fn widen(v: Self::Native) -> i128;
    /// `v` wrapped to the native type, and whether it did not fit.
    fn narrow(v: i128) -> (Self::Native, bool);
}

macro_rules! spark_integral {
    ($($t:ty => $n:ty, $w:ident),*) => {$(
        impl sealed::Sealed for $t {}
        impl SparkIntegral for $t {
            const WIDTH: IntWidth = IntWidth::$w;
            fn widen(v: $n) -> i128 {
                i128::from(v)
            }
            #[expect(clippy::cast_possible_truncation, reason = "wrapping is the documented result")]
            fn narrow(v: i128) -> ($n, bool) {
                let n = v as $n;
                (n, i128::from(n) != v)
            }
        }
    )*};
}
spark_integral!(
    Int8Type => i8, Byte,
    Int16Type => i16, Short,
    Int32Type => i32, Int,
    Int64Type => i64, Long
);

/// The overflow error of `l op r` at `row`.
pub(crate) fn overflow_error<T: SparkIntegral>(
    ctx: &KernelCtx,
    row: usize,
    op: ArithOp,
    l: T::Native,
    r: T::Native,
) -> KernelError {
    match T::WIDTH {
        IntWidth::Byte | IntWidth::Short => {
            // Byte and short values always fit an i64.
            let v = |x| i64::try_from(T::widen(x)).unwrap_or_default();
            KernelError::binary_arithmetic_overflow(row, v(l), op.symbol(), v(r), op.try_function())
        }
        IntWidth::Int | IntWidth::Long => KernelError::arithmetic_overflow(
            row,
            ctx.spark_version().overflow_message(T::WIDTH),
            Some(op.try_function()),
        ),
    }
}

/// Row `row` of a side.
pub(crate) fn at<T: Copy + Default>(s: Side<'_, T>, row: usize) -> T {
    match s {
        Side::Array(v) => v.get(row).copied().unwrap_or_default(),
        Side::Scalar(v) => v,
    }
}

/// [`LaneOp`] for a [`BinaryOp`].
#[derive(Clone, Copy)]
struct IntLane<Op>(PhantomData<Op>);

impl<T: IntNative, Op: BinaryOp> LaneOp<T> for IntLane<Op> {
    type Out = T;
    #[inline(always)]
    fn wrapping(self, a: T, b: T) -> T {
        Op::wrapping(a, b)
    }
    #[inline(always)]
    fn checked(self, a: T, b: T) -> (T, bool) {
        Op::overflowing(a, b)
    }
    #[inline(always)]
    fn fault(self, a: T, b: T) -> Option<Fault> {
        Op::overflowing(a, b).1.then_some(Fault::Overflow)
    }
}

/// `l Op r` with overflow policy `O`. The result is NULL where either input
/// is NULL (and, under `NullOnOverflow`, where the result overflows).
///
/// # Errors
/// - `ARITHMETIC_OVERFLOW` (`int`, `bigint`) or `BINARY_ARITHMETIC_OVERFLOW`
///   (`tinyint`, `smallint`), with [`KernelError::row`], under `AnsiError`;
/// - `INTERNAL_ERROR` for two scalars or arrays of different lengths;
/// - `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be allocated.
///
/// ```
/// use arrow_array::Int64Array;
/// use krishiv_kernels::integer::binary;
/// use krishiv_kernels::{Add, AnsiError, KernelCtx, Operand};
///
/// let a = Int64Array::from(vec![Some(1), None, Some(i64::MAX)]);
/// let ok = binary::<_, Add, AnsiError>(Operand::Array(&a), Operand::Scalar(Some(0)), &KernelCtx::new())?;
/// assert_eq!(ok, a);
/// let err = binary::<_, Add, AnsiError>(Operand::Array(&a), Operand::Scalar(Some(1)), &KernelCtx::new())
///     .unwrap_err();
/// assert_eq!(err.row(), Some(2));
/// # Ok::<(), krishiv_kernels::KernelError>(())
/// ```
pub fn binary<T, Op, O>(
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    ctx: &KernelCtx,
) -> Result<PrimitiveArray<T>, KernelError>
where
    T: SparkIntegral,
    Op: BinaryOp,
    O: Overflow,
{
    binary_impl::<T, Op, O>(l, r, ctx, false)
}

/// [`binary`] with an all-valid bitmap attached even when no row is NULL,
/// so benchmarks can show that carrying validity costs nothing measurable
/// (the reason elementwise kernels have no has-nulls axis). Not for
/// production use.
#[doc(hidden)]
pub fn binary_nulls_path<T, Op, O>(
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    ctx: &KernelCtx,
) -> Result<PrimitiveArray<T>, KernelError>
where
    T: SparkIntegral,
    Op: BinaryOp,
    O: Overflow,
{
    binary_impl::<T, Op, O>(l, r, ctx, true)
}

fn binary_impl<T, Op, O>(
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    ctx: &KernelCtx,
    force_nulls: bool,
) -> Result<PrimitiveArray<T>, KernelError>
where
    T: SparkIntegral,
    Op: BinaryOp,
    O: Overflow,
{
    let (l, r, nulls, len) = match chunk::resolve(l, r)? {
        Resolved::AllNull(len) => return Ok(PrimitiveArray::new_null(len)),
        Resolved::Lanes { l, r, nulls, len } => (l, r, nulls, len),
    };
    let nulls = nulls.or_else(|| force_nulls.then(|| NullBuffer::new_valid(len)));
    let lane = IntLane::<Op>(PhantomData);
    let c = chunk::compute::<T::Native, _, O>(ctx, l, r, nulls, len, lane, |row, fault| {
        KernelError::for_fault(row, fault, |row| {
            overflow_error::<T>(ctx, row, Op::OP, at(l, row), at(r, row))
        })
    })?;
    finish(c)
}

/// The computed values as an array.
pub(crate) fn finish<T: ArrowPrimitiveType>(
    c: chunk::Computed<T::Native>,
) -> Result<PrimitiveArray<T>, KernelError> {
    PrimitiveArray::try_new(ScalarBuffer::from(c.values), c.nulls)
        .map_err(|e| KernelError::internal(e.to_string()))
}

/// `%`: Java's remainder (the sign follows the dividend; `MIN % -1 == 0`).
#[derive(Clone, Copy)]
struct RemLane;

impl<T: IntNative> LaneOp<T> for RemLane {
    type Out = T;
    const MAY_DIVIDE_BY_ZERO: bool = true;
    #[inline(always)]
    fn wrapping(self, a: T, b: T) -> T {
        a.wrapping_rem(b)
    }
    #[inline(always)]
    fn checked(self, a: T, b: T) -> (T, bool) {
        (a.wrapping_rem(b), b == T::default())
    }
    #[inline(always)]
    fn fault(self, _a: T, b: T) -> Option<Fault> {
        (b == T::default()).then_some(Fault::DivideByZero)
    }
}

/// `l % r` (Spark `Remainder`, also `mod` and `try_mod`). It never
/// overflows; a zero divisor is NULL under `JavaWrap`/`NullOnOverflow`
/// and raises under `AnsiError`/`Unchecked`.
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
    T: SparkIntegral,
    O: Overflow,
{
    let (l, r, nulls, len) = match chunk::resolve(l, r)? {
        Resolved::AllNull(len) => return Ok(PrimitiveArray::new_null(len)),
        Resolved::Lanes { l, r, nulls, len } => (l, r, nulls, len),
    };
    let c = chunk::compute::<T::Native, _, O>(ctx, l, r, nulls, len, RemLane, |row, _| {
        KernelError::remainder_by_zero(row, ctx.spark_version())
    })?;
    finish(c)
}

/// `div` on `bigint`: truncating division.
#[derive(Clone, Copy)]
struct DivLane;

impl LaneOp<i64> for DivLane {
    type Out = i64;
    const MAY_DIVIDE_BY_ZERO: bool = true;
    #[inline(always)]
    fn wrapping(self, a: i64, b: i64) -> i64 {
        IntNative::wrapping_div(a, b)
    }
    #[inline(always)]
    fn checked(self, a: i64, b: i64) -> (i64, bool) {
        let bad = (b == 0) | ((a == i64::MIN) & (b == -1));
        (IntNative::wrapping_div(a, b), bad)
    }
    #[inline(always)]
    fn fault(self, a: i64, b: i64) -> Option<Fault> {
        match (a, b) {
            (_, 0) => Some(Fault::DivideByZero),
            (i64::MIN, -1) => Some(Fault::Overflow),
            _ => None,
        }
    }
}

/// `l div r` (Spark `IntegralDivide`) on `bigint`: Spark casts integral
/// operands to `bigint` first, so `-128Y div -1Y` is `128`. Only
/// `MIN div -1` overflows (it wraps to `MIN` under `JavaWrap`); a zero
/// divisor is NULL under `JavaWrap`/`NullOnOverflow`.
///
/// # Errors
/// `DIVIDE_BY_ZERO` and `ARITHMETIC_OVERFLOW` ("Overflow in integral
/// divide") with the row under `AnsiError`; engine and memory errors as
/// [`binary`].
pub fn integral_divide<O: Overflow>(
    l: Operand<'_, Int64Type>,
    r: Operand<'_, Int64Type>,
    ctx: &KernelCtx,
) -> Result<PrimitiveArray<Int64Type>, KernelError> {
    let (l, r, nulls, len) = match chunk::resolve(l, r)? {
        Resolved::AllNull(len) => return Ok(PrimitiveArray::new_null(len)),
        Resolved::Lanes { l, r, nulls, len } => (l, r, nulls, len),
    };
    let c = chunk::compute::<i64, _, O>(ctx, l, r, nulls, len, DivLane, |row, fault| {
        KernelError::for_fault(row, fault, |row| {
            KernelError::arithmetic_overflow(row, "Overflow in integral divide", Some("try_divide"))
        })
    })?;
    finish(c)
}

/// A unary integral operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// Unary `-` (`negative`).
    Negate,
    /// `abs`.
    Abs,
}

/// A unary operation as a lane (the right side is an ignored scalar).
#[derive(Clone, Copy)]
struct UnaryLane(UnaryOp);

impl<T: IntNative> LaneOp<T> for UnaryLane {
    type Out = T;
    #[inline(always)]
    fn wrapping(self, a: T, _b: T) -> T {
        match self.0 {
            UnaryOp::Negate => a.wrapping_neg(),
            UnaryOp::Abs => a.wrapping_abs(),
        }
    }
    #[inline(always)]
    fn checked(self, a: T, b: T) -> (T, bool) {
        (self.wrapping(a, b), a == T::MIN)
    }
    #[inline(always)]
    fn fault(self, a: T, _b: T) -> Option<Fault> {
        (a == T::MIN).then_some(Fault::Overflow)
    }
}

/// `op a`: only `MIN` overflows (it wraps to `MIN` under `JavaWrap`).
///
/// # Errors
/// `ARITHMETIC_OVERFLOW` (no `try_*` hint) with the row under `AnsiError`;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be allocated.
pub fn unary<T, O>(
    op: UnaryOp,
    a: &PrimitiveArray<T>,
    ctx: &KernelCtx,
) -> Result<PrimitiveArray<T>, KernelError>
where
    T: SparkIntegral,
    O: Overflow,
{
    let nulls = a.nulls().filter(|n| n.null_count() > 0).cloned();
    let (l, r) = (
        Side::Array(a.values().as_ref()),
        Side::Scalar(T::Native::default()),
    );
    let c =
        chunk::compute::<T::Native, _, O>(ctx, l, r, nulls, a.len(), UnaryLane(op), |row, _| {
            KernelError::arithmetic_overflow(
                row,
                ctx.spark_version().overflow_message(T::WIDTH),
                None,
            )
        })?;
    finish(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The unchecked forms the chunk engine uses for non-checking policies
    /// (dividing lanes always check, so only these tests call them).
    #[test]
    fn lane_wrapping_forms_are_total() {
        assert_eq!(LaneOp::<i32>::wrapping(RemLane, 7, 0), 0);
        assert_eq!(LaneOp::<i32>::wrapping(RemLane, -7, 3), -1);
        assert_eq!(DivLane.wrapping(i64::MIN, -1), i64::MIN);
        assert_eq!(DivLane.wrapping(7, 0), 7);
    }
}
