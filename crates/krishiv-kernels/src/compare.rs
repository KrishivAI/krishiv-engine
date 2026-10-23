// SPDX-License-Identifier: Apache-2.0
//! Spark comparisons: `= != < <= > >=` and the null-safe `<=>`.
//!
//! - Primitives ([`compare`]): integral, `date` (`Date32`), timestamps
//!   (`Timestamp(Microsecond)`), decimals of one type (`Decimal64` or
//!   `Decimal128`; the planner casts both sides to a common type and
//!   carrier first) and floats. Floats
//!   use `SQLOrderingUtil.compareDoubles`: `NaN = NaN`, `NaN` above every
//!   other value, `-0.0 = 0.0` (IEEE comparison and arrow-rs's `total_cmp`
//!   both differ).
//! - Strings and binary ([`compare_bytes`]): unsigned byte order (Spark's
//!   `UTF8String.compareTo`, i.e. code-point order; no collation).
//! - Booleans ([`compare_bool`]): `false < true`.
//!
//! Results are NULL where either input is NULL; [`null_safe`] turns an
//! equality into `<=>` (never NULL).

use std::marker::PhantomData;

use arrow_array::types::ByteArrayType;
use arrow_array::{Array, ArrowPrimitiveType, BooleanArray, GenericByteArray};
use arrow_buffer::{ArrowNativeType, BooleanBuffer, Buffer, NullBuffer};

use crate::chunk::{self, CHUNK, Resolved, Side, split};
use crate::dispatch::Dispatch;
use crate::error::{KernelError, with_capacity};
use crate::ops::Operand;
use crate::unsafe_dispatch::{Kernel, run, scalar_loops};

mod sealed {
    pub trait Sealed {}
}

/// A native value with Spark's total order.
pub trait SparkOrd: sealed::Sealed + ArrowNativeType + Default {
    /// `a = b`.
    fn spark_eq(a: Self, b: Self) -> bool;
    /// `a < b`.
    fn spark_lt(a: Self, b: Self) -> bool;
    /// `a <= b`.
    fn spark_le(a: Self, b: Self) -> bool;
}

macro_rules! ord_int {
    ($($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl SparkOrd for $t {
            #[inline(always)]
            fn spark_eq(a: Self, b: Self) -> bool { a == b }
            #[inline(always)]
            fn spark_lt(a: Self, b: Self) -> bool { a < b }
            #[inline(always)]
            fn spark_le(a: Self, b: Self) -> bool { a <= b }
        }
    )*};
}
ord_int!(i8, i16, i32, i64, i128);

macro_rules! ord_float {
    ($($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl SparkOrd for $t {
            #[inline(always)]
            fn spark_eq(a: Self, b: Self) -> bool {
                // IEEE `==` already equates -0.0 and 0.0.
                (a == b) | (a.is_nan() & b.is_nan())
            }
            #[inline(always)]
            fn spark_lt(a: Self, b: Self) -> bool {
                (a < b) | (b.is_nan() & !a.is_nan())
            }
            #[inline(always)]
            fn spark_le(a: Self, b: Self) -> bool {
                (a <= b) | b.is_nan()
            }
        }
    )*};
}
ord_float!(f32, f64);

/// A comparison operator, as a zero-sized type.
pub trait CompareOp: Copy + Send + Sync + 'static {
    /// The operator.
    const KIND: CompareKind;
    /// `a op b` under Spark's order; branch-free.
    #[inline(always)]
    fn compare<F: SparkOrd>(a: F, b: F) -> bool {
        match Self::KIND {
            CompareKind::Equal => F::spark_eq(a, b),
            CompareKind::NotEqual => !F::spark_eq(a, b),
            CompareKind::Less => F::spark_lt(a, b),
            CompareKind::LessEq => F::spark_le(a, b),
            CompareKind::Greater => F::spark_lt(b, a),
            CompareKind::GreaterEq => F::spark_le(b, a),
        }
    }
}

/// A comparison operator's runtime name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareKind {
    /// `=`
    Equal,
    /// `!=` / `<>`
    NotEqual,
    /// `<`
    Less,
    /// `<=`
    LessEq,
    /// `>`
    Greater,
    /// `>=`
    GreaterEq,
}

impl CompareKind {
    /// Whether `ord` (of `a` against `b`) satisfies `a op b`.
    #[must_use]
    pub fn holds(self, ord: std::cmp::Ordering) -> bool {
        match self {
            Self::Equal => ord.is_eq(),
            Self::NotEqual => ord.is_ne(),
            Self::Less => ord.is_lt(),
            Self::LessEq => ord.is_le(),
            Self::Greater => ord.is_gt(),
            Self::GreaterEq => ord.is_ge(),
        }
    }
}

macro_rules! compare_ops {
    ($($(#[$m:meta])* $t:ident),*) => {$(
        $(#[$m])*
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $t;
        impl CompareOp for $t {
            const KIND: CompareKind = CompareKind::$t;
        }
    )*};
}
compare_ops!(
    /// `=` (`EqualTo`).
    Equal,
    /// `!=` / `<>`.
    NotEqual,
    /// `<`.
    Less,
    /// `<=`.
    LessEq,
    /// `>`.
    Greater,
    /// `>=`.
    GreaterEq
);

/// `l op r` for primitive arrays.
///
/// # Errors
/// `INTERNAL_ERROR` for two scalars, arrays of different lengths or (for
/// decimals) of different types; `UNABLE_TO_ACQUIRE_MEMORY` if the output
/// cannot be allocated.
///
/// ```
/// use arrow_array::{BooleanArray, Float64Array};
/// use krishiv_kernels::compare::{compare, Equal};
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
    ctx: &impl Dispatch,
) -> Result<BooleanArray, KernelError>
where
    T: ArrowPrimitiveType<Native: SparkOrd>,
    Op: CompareOp,
{
    if let (Operand::Array(a), Operand::Array(b)) = (l, r)
        && a.data_type() != b.data_type()
    {
        return Err(KernelError::internal(format!(
            "comparison of {} with {}",
            a.data_type(),
            b.data_type()
        )));
    }
    let (l, r, nulls, len) = match chunk::resolve(l, r)? {
        Resolved::AllNull(len) => return Ok(BooleanArray::new_null(len)),
        Resolved::Lanes { l, r, nulls, len } => (l, r, nulls, len),
    };
    let mut words = with_capacity::<u64>(len.div_ceil(CHUNK))?;
    macro_rules! go {
        ($ls:literal, $rs:literal, $l:expr, $r:expr) => {
            run(
                ctx,
                Compare::<'_, T::Native, Op, $ls, $rs> {
                    l: $l,
                    r: $r,
                    len,
                    words: &mut words,
                    op: PhantomData,
                },
            )
        };
    }
    if scalar_loops(ctx) {
        match (l, r) {
            (Side::Array(_), Side::Array(_)) => go!(false, false, l, r),
            (Side::Array(_), Side::Scalar(_)) => go!(false, true, l, r),
            (Side::Scalar(_), _) => go!(true, false, l, r),
        }
    } else {
        let (mut lb, mut rb) = (Vec::new(), Vec::new());
        go!(
            false,
            false,
            l.to_array(len, &mut lb)?,
            r.to_array(len, &mut rb)?
        );
    }
    Ok(BooleanArray::new(bits(words, len), nulls))
}

/// The bitmap of `words`, without copying (Arrow bitmaps are little-endian).
fn bits(mut words: Vec<u64>, len: usize) -> BooleanBuffer {
    for w in &mut words {
        *w = w.to_le();
    }
    BooleanBuffer::new(Buffer::from_vec(words), 0, len)
}

/// The comparison loop for one operand shape; `run` compiles it per CPU target.
struct Compare<'a, F, Op, const L_SCALAR: bool, const R_SCALAR: bool> {
    l: Side<'a, F>,
    r: Side<'a, F>,
    len: usize,
    words: &'a mut Vec<u64>,
    op: PhantomData<Op>,
}

impl<F: SparkOrd, Op: CompareOp, const L_SCALAR: bool, const R_SCALAR: bool> Kernel
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

/// One side of a string or binary comparison: an array, or a scalar
/// (`None` is a NULL literal).
#[derive(Debug)]
pub enum BytesOperand<'a, T: ByteArrayType> {
    /// A column.
    Array(&'a GenericByteArray<T>),
    /// A literal's bytes.
    Scalar(Option<&'a [u8]>),
}

impl<T: ByteArrayType> Clone for BytesOperand<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: ByteArrayType> Copy for BytesOperand<'_, T> {}

impl<T: ByteArrayType> BytesOperand<'_, T> {
    fn len(&self) -> Option<usize> {
        match self {
            Self::Array(a) => Some(a.len()),
            Self::Scalar(_) => None,
        }
    }

    fn nulls(&self) -> Option<&NullBuffer> {
        match self {
            Self::Array(a) => a.nulls(),
            Self::Scalar(_) => None,
        }
    }

    /// Row `i`'s bytes (the slot's bytes even under a NULL).
    fn at(&self, i: usize) -> &[u8] {
        match self {
            Self::Array(a) => a.value(i).as_ref(),
            Self::Scalar(v) => v.unwrap_or_default(),
        }
    }
}

/// The common length of two operands, or the engine error.
fn common_len(l: Option<usize>, r: Option<usize>) -> Result<usize, KernelError> {
    match (l, r) {
        (Some(a), Some(b)) if a != b => Err(KernelError::internal(format!(
            "comparison operands differ in length: {a} vs {b}"
        ))),
        (Some(a), _) | (_, Some(a)) => Ok(a),
        (None, None) => Err(KernelError::internal(
            "comparison needs at least one array operand",
        )),
    }
}

/// `l op r` for strings or binary, in unsigned byte order.
///
/// # Errors
/// `INTERNAL_ERROR` for two scalars or arrays of different lengths.
pub fn compare_bytes<T: ByteArrayType, Op: CompareOp>(
    l: BytesOperand<'_, T>,
    r: BytesOperand<'_, T>,
) -> Result<BooleanArray, KernelError> {
    let len = common_len(l.len(), r.len())?;
    if matches!(l, BytesOperand::Scalar(None)) || matches!(r, BytesOperand::Scalar(None)) {
        return Ok(BooleanArray::new_null(len));
    }
    // Byte slices compare lexicographically as unsigned bytes; the kind's
    // truth table turns the ordering into the result.
    let values = BooleanBuffer::collect_bool(len, |i| Op::KIND.holds(l.at(i).cmp(r.at(i))));
    Ok(BooleanArray::new(
        values,
        NullBuffer::union(l.nulls(), r.nulls()),
    ))
}

/// One side of a boolean comparison.
#[derive(Debug, Clone, Copy)]
pub enum BoolOperand<'a> {
    /// A column.
    Array(&'a BooleanArray),
    /// A literal (`None` is NULL).
    Scalar(Option<bool>),
}

/// `l op r` for booleans (`false < true`), on whole bitmaps.
///
/// # Errors
/// `INTERNAL_ERROR` for two scalars or arrays of different lengths.
pub fn compare_bool<Op: CompareOp>(
    l: BoolOperand<'_>,
    r: BoolOperand<'_>,
) -> Result<BooleanArray, KernelError> {
    let len_of = |o: BoolOperand<'_>| match o {
        BoolOperand::Array(a) => Some(a.len()),
        BoolOperand::Scalar(_) => None,
    };
    let len = common_len(len_of(l), len_of(r))?;
    let bits = |o: BoolOperand<'_>| match o {
        BoolOperand::Array(a) => Some((a.values().clone(), a.nulls().cloned())),
        BoolOperand::Scalar(Some(true)) => Some((BooleanBuffer::new_set(len), None)),
        BoolOperand::Scalar(Some(false)) => Some((BooleanBuffer::new_unset(len), None)),
        BoolOperand::Scalar(None) => None,
    };
    let (Some((a, an)), Some((b, bn))) = (bits(l), bits(r)) else {
        return Ok(BooleanArray::new_null(len));
    };
    let values = match Op::KIND {
        CompareKind::Equal => !&(&a ^ &b),
        CompareKind::NotEqual => &a ^ &b,
        CompareKind::Less => &!&a & &b,
        CompareKind::LessEq => &!&a | &b,
        CompareKind::Greater => &a & &!&b,
        CompareKind::GreaterEq => &a | &!&b,
    };
    Ok(BooleanArray::new(
        values,
        NullBuffer::union(an.as_ref(), bn.as_ref()),
    ))
}

/// Validity of one `<=>` operand.
#[derive(Debug, Clone, Copy)]
pub enum Validity<'a> {
    /// An array's validity (`None`: no NULLs).
    Array(Option<&'a NullBuffer>),
    /// A scalar: `true` when it is NULL.
    Scalar {
        /// The literal is NULL.
        null: bool,
    },
}

/// `l <=> r` from the equality `eq` of the same operands (its values under
/// NULL rows are ignored): true where both are NULL, false where one is,
/// else the equality. Never NULL.
///
/// # Errors
/// `INTERNAL_ERROR` if a validity differs in length from `eq`.
pub fn null_safe(
    eq: &BooleanArray,
    l: Validity<'_>,
    r: Validity<'_>,
) -> Result<BooleanArray, KernelError> {
    let len = eq.len();
    for v in [l, r] {
        if let Validity::Array(Some(n)) = v {
            common_len(Some(len), Some(n.len()))?;
        }
    }
    let valid = |v: Validity<'_>| match v {
        Validity::Array(Some(n)) => n.inner().clone(),
        Validity::Array(None) | Validity::Scalar { null: false } => BooleanBuffer::new_set(len),
        Validity::Scalar { null: true } => BooleanBuffer::new_unset(len),
    };
    let (lv, rv) = (valid(l), valid(r));
    let both = &(&lv & &rv) & eq.values();
    let neither = &!&lv & &!&rv;
    Ok(BooleanArray::new(&both | &neither, None))
}
