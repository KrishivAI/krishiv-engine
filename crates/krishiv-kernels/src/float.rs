// SPDX-License-Identifier: Apache-2.0
//! Floating-point comparisons with Spark semantics.
//!
//! Spark compares `float`/`double` with `SQLOrderingUtil.compareDoubles`:
//! `NaN = NaN`, `NaN` is greater than every other value, and `-0.0 = 0.0`.
//! IEEE comparison (`NaN != NaN`) and arrow-rs's `total_cmp` kernels
//! (`-0.0 < 0.0`) both differ, so these kernels are our own. The result is
//! NULL where either input is NULL.

use arrow_array::{ArrowPrimitiveType, BooleanArray};
use arrow_buffer::{ArrowNativeType, BooleanBuffer, Buffer};

use crate::chunk::{self, CHUNK, Resolved, Side, split};
use crate::dispatch::KernelCtx;
use crate::error::KernelError;
use crate::ops::Operand;
use crate::unsafe_dispatch::{Kernel, run};

mod sealed {
    pub trait Sealed {}
}

/// `f32` and `f64`.
pub trait SparkFloat: sealed::Sealed + ArrowNativeType + PartialOrd + Default {
    /// Whether the value is a NaN.
    fn is_nan(self) -> bool;
}

macro_rules! spark_float {
    ($($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl SparkFloat for $t {
            #[inline(always)]
            fn is_nan(self) -> bool { <$t>::is_nan(self) }
        }
    )*};
}
spark_float!(f32, f64);

/// A Spark comparison operator, as a zero-sized type.
pub trait CompareOp: Copy + Send + Sync + 'static {
    /// `a op b` under Spark's float ordering; branch-free.
    fn compare<F: SparkFloat>(a: F, b: F) -> bool;
}

/// `=` (`EqualTo`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Equal;
/// `<`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Less;
/// `<=`.
#[derive(Debug, Clone, Copy, Default)]
pub struct LessEq;
/// `>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Greater;
/// `>=`.
#[derive(Debug, Clone, Copy, Default)]
pub struct GreaterEq;

impl CompareOp for Equal {
    #[inline(always)]
    fn compare<F: SparkFloat>(a: F, b: F) -> bool {
        // IEEE `==` already equates -0.0 and 0.0.
        (a == b) | (a.is_nan() & b.is_nan())
    }
}
impl CompareOp for Less {
    #[inline(always)]
    fn compare<F: SparkFloat>(a: F, b: F) -> bool {
        (a < b) | (b.is_nan() & !a.is_nan())
    }
}
impl CompareOp for LessEq {
    #[inline(always)]
    fn compare<F: SparkFloat>(a: F, b: F) -> bool {
        (a <= b) | b.is_nan()
    }
}
impl CompareOp for Greater {
    #[inline(always)]
    fn compare<F: SparkFloat>(a: F, b: F) -> bool {
        Less::compare(b, a)
    }
}
impl CompareOp for GreaterEq {
    #[inline(always)]
    fn compare<F: SparkFloat>(a: F, b: F) -> bool {
        LessEq::compare(b, a)
    }
}

/// `l op r` for `float`/`double` arrays.
///
/// # Errors
/// `INTERNAL_ERROR` for two scalars or arrays of different lengths;
/// `UNABLE_TO_ACQUIRE_MEMORY` if the output cannot be allocated.
///
/// ```
/// use arrow_array::{BooleanArray, Float64Array};
/// use krishiv_kernels::float::{compare, Equal};
/// use krishiv_kernels::{KernelCtx, Operand};
///
/// let a = Float64Array::from(vec![f64::NAN, -0.0, 1.0]);
/// let b = Float64Array::from(vec![f64::NAN, 0.0, 2.0]);
/// let eq = compare::<_, Equal>(Operand::Array(&a), Operand::Array(&b), &KernelCtx::new())?;
/// assert_eq!(eq, BooleanArray::from(vec![true, true, false]));
/// # Ok::<(), krishiv_kernels::KernelError>(())
/// ```
pub fn compare<T, Op>(
    l: Operand<'_, T>,
    r: Operand<'_, T>,
    ctx: &KernelCtx,
) -> Result<BooleanArray, KernelError>
where
    T: ArrowPrimitiveType<Native: SparkFloat>,
    Op: CompareOp,
{
    let (l, r, nulls, len) = match chunk::resolve(l, r)? {
        Resolved::AllNull(len) => return Ok(BooleanArray::new_null(len)),
        Resolved::Lanes { l, r, nulls, len } => (l, r, nulls, len),
    };
    let n_words = len.div_ceil(CHUNK);
    let mut words = Vec::new();
    words
        .try_reserve_exact(n_words)
        .map_err(|_| KernelError::out_of_memory(n_words.saturating_mul(8)))?;
    macro_rules! go {
        ($ls:literal, $rs:literal) => {
            run(
                ctx,
                Compare::<'_, T::Native, Op, $ls, $rs> {
                    l,
                    r,
                    len,
                    words: &mut words,
                    op: std::marker::PhantomData,
                },
            )
        };
    }
    match (l, r) {
        (Side::Array(_), Side::Array(_)) => go!(false, false),
        (Side::Array(_), Side::Scalar(_)) => go!(false, true),
        (Side::Scalar(_), _) => go!(true, false),
    }
    Ok(BooleanArray::new(bits(&words, len), nulls))
}

fn bits(words: &[u64], len: usize) -> BooleanBuffer {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    BooleanBuffer::new(Buffer::from_vec(bytes), 0, len)
}

/// The comparison loop for one operand shape; `run` compiles it per CPU target.
struct Compare<'a, F, Op, const L_SCALAR: bool, const R_SCALAR: bool> {
    l: Side<'a, F>,
    r: Side<'a, F>,
    len: usize,
    words: &'a mut Vec<u64>,
    op: std::marker::PhantomData<Op>,
}

impl<F: SparkFloat, Op: CompareOp, const L_SCALAR: bool, const R_SCALAR: bool> Kernel
    for Compare<'_, F, Op, L_SCALAR, R_SCALAR>
{
    type Output = ();

    #[inline(always)]
    fn run(self) {
        let (lc, lt, ls) = split(self.l);
        let (rc, rt, rs) = split(self.r);
        let zero = [F::default(); CHUNK];
        let full = self.len / CHUNK;
        let word = |a: &[F; CHUNK], b: &[F; CHUNK]| {
            let mut w = 0_u64;
            for ((&x, &y), bit) in a.iter().zip(b).zip(0_u32..) {
                let x = if L_SCALAR { ls } else { x };
                let y = if R_SCALAR { rs } else { y };
                w |= u64::from(Op::compare(x, y)).wrapping_shl(bit);
            }
            w
        };
        for i in 0..full {
            let a = lc.get(i).unwrap_or(&zero);
            let b = rc.get(i).unwrap_or(&zero);
            self.words.push(word(a, b));
        }
        if !self.len.is_multiple_of(CHUNK) {
            let (mut a, mut b) = (zero, zero);
            for (d, s) in a.iter_mut().zip(lt) {
                *d = *s;
            }
            for (d, s) in b.iter_mut().zip(rt) {
                *d = *s;
            }
            self.words.push(word(&a, &b));
        }
    }
}
