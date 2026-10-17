// SPDX-License-Identifier: Apache-2.0
//! Integral `+ - *` for Spark `int` and `bigint` (the framework's proof
//! kernel family; the full arithmetic set is built on it later).
//!
//! `JavaWrap` is Spark's legacy (non-ANSI) behaviour, `AnsiError` raises
//! `ARITHMETIC_OVERFLOW` at the first overflowing non-NULL row, and
//! `Unchecked` is for plans that proved overflow impossible (e.g. operands
//! widened from a narrower type). `NullOnOverflow` is accepted too (Spark
//! has no such integral mode, but `try_add` and friends have exactly this
//! semantics).

use std::marker::PhantomData;

use arrow_array::types::{Int32Type, Int64Type};
use arrow_array::{ArrowPrimitiveType, PrimitiveArray};
use arrow_buffer::ScalarBuffer;

use crate::chunk::{self, LaneOp, Resolved};
use crate::dispatch::KernelCtx;
use crate::error::{KernelError, zeroed};
use crate::ops::{BinaryOp, IntNative, Operand};
use crate::policy::Overflow;

mod sealed {
    pub trait Sealed {}
}

/// Arrow types of Spark's `int` and `bigint`. (`tinyint`/`smallint` raise a
/// different error class, `BINARY_ARITHMETIC_OVERFLOW`, and arrive with the
/// full arithmetic set.)
pub trait SparkIntegral: sealed::Sealed + ArrowPrimitiveType<Native: IntNative> {
    /// The value as an `i128` (exact).
    fn widen(v: Self::Native) -> i128;
    /// `v` wrapped to the native type, and whether it did not fit.
    fn narrow(v: i128) -> (Self::Native, bool);
}

macro_rules! spark_integral {
    ($($t:ty => $n:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl SparkIntegral for $t {
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
spark_integral!(Int32Type => i32, Int64Type => i64);

/// [`LaneOp`] for a [`BinaryOp`].
#[derive(Clone, Copy)]
struct IntLane<Op>(PhantomData<Op>);

impl<T: IntNative, Op: BinaryOp> LaneOp<T> for IntLane<Op> {
    #[inline(always)]
    fn wrapping(self, a: T, b: T) -> T {
        Op::wrapping(a, b)
    }
    #[inline(always)]
    fn checked(self, a: T, b: T) -> (T, bool) {
        Op::overflowing(a, b)
    }
    #[inline(always)]
    fn is_bad(self, a: T, b: T) -> bool {
        Op::overflowing(a, b).1
    }
}

/// `l Op r` with overflow policy `O`. The result is NULL where either input
/// is NULL (and, under `NullOnOverflow`, where the result overflows).
///
/// # Errors
/// - `ARITHMETIC_OVERFLOW` (with [`KernelError::row`]) under `AnsiError`;
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
    let (l, r, nulls, len) = match chunk::resolve(l, r)? {
        Resolved::AllNull(len) => return Ok(PrimitiveArray::new_null(len)),
        Resolved::Lanes { l, r, nulls, len } => (l, r, nulls, len),
    };
    let out = zeroed(len)?;
    let lane = IntLane::<Op>(PhantomData);
    let c = chunk::compute::<T::Native, _, O>(ctx, l, r, nulls, out, lane)
        .map_err(|row| KernelError::arithmetic_overflow(row, Op::OP))?;
    PrimitiveArray::try_new(ScalarBuffer::from(c.values), c.nulls)
        .map_err(|e| KernelError::internal(e.to_string()))
}
