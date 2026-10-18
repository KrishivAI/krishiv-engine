// SPDX-License-Identifier: Apache-2.0
//! Operands and the arithmetic operations kernels are generic over.

use arrow_array::{ArrowPrimitiveType, PrimitiveArray};
use arrow_buffer::ArrowNativeType;

/// One side of a binary kernel: an array, or a scalar (`None` is a NULL
/// literal) broadcast over the other side.
#[derive(Debug)]
pub enum Operand<'a, T: ArrowPrimitiveType> {
    /// A column.
    Array(&'a PrimitiveArray<T>),
    /// A literal.
    Scalar(Option<T::Native>),
}

// Manual impls: a derive would require `T: Clone`/`T: Copy`.
impl<T: ArrowPrimitiveType> Clone for Operand<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: ArrowPrimitiveType> Copy for Operand<'_, T> {}

/// A Spark binary arithmetic operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArithOp {
    /// `+`
    Add,
    /// `-`
    Subtract,
    /// `*`
    Multiply,
}

impl ArithOp {
    /// The operator as Spark writes it in error messages.
    #[must_use]
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Subtract => "-",
            Self::Multiply => "*",
        }
    }

    /// The `try_*` function Spark suggests in its overflow error.
    #[must_use]
    pub fn try_function(self) -> &'static str {
        match self {
            Self::Add => "try_add",
            Self::Subtract => "try_subtract",
            Self::Multiply => "try_multiply",
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// Native integers kernels compute on (`i8` to `i128`).
pub trait IntNative: sealed::Sealed + ArrowNativeType + Ord + Default {
    /// `self + rhs`, wrapping.
    fn wrapping_add(self, rhs: Self) -> Self;
    /// `self - rhs`, wrapping.
    fn wrapping_sub(self, rhs: Self) -> Self;
    /// `self * rhs`, wrapping.
    fn wrapping_mul(self, rhs: Self) -> Self;
    /// `self + rhs` and whether it overflowed.
    fn overflowing_add(self, rhs: Self) -> (Self, bool);
    /// `self - rhs` and whether it overflowed.
    fn overflowing_sub(self, rhs: Self) -> (Self, bool);
    /// `self * rhs` and whether it overflowed.
    fn overflowing_mul(self, rhs: Self) -> (Self, bool);
    /// `self / rhs`, wrapping (`MIN / -1 == MIN`); `rhs` must not be zero.
    fn wrapping_div(self, rhs: Self) -> Self;
    /// `self % rhs`, wrapping (`MIN % -1 == 0`); `rhs` must not be zero.
    fn wrapping_rem(self, rhs: Self) -> Self;
    /// `-self`, wrapping.
    fn wrapping_neg(self) -> Self;
    /// `|self|`, wrapping.
    fn wrapping_abs(self) -> Self;
    /// `self`, or 1 where `self` is zero: a divisor that never traps.
    fn nonzero(self) -> Self;
    /// The type's minimum.
    const MIN: Self;
}

macro_rules! int_native {
    ($($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl IntNative for $t {
            #[inline(always)]
            fn wrapping_add(self, rhs: Self) -> Self { <$t>::wrapping_add(self, rhs) }
            #[inline(always)]
            fn wrapping_sub(self, rhs: Self) -> Self { <$t>::wrapping_sub(self, rhs) }
            #[inline(always)]
            fn wrapping_mul(self, rhs: Self) -> Self { <$t>::wrapping_mul(self, rhs) }
            #[inline(always)]
            fn overflowing_add(self, rhs: Self) -> (Self, bool) { <$t>::overflowing_add(self, rhs) }
            #[inline(always)]
            fn overflowing_sub(self, rhs: Self) -> (Self, bool) { <$t>::overflowing_sub(self, rhs) }
            #[inline(always)]
            fn overflowing_mul(self, rhs: Self) -> (Self, bool) { <$t>::overflowing_mul(self, rhs) }
            #[inline(always)]
            fn wrapping_div(self, rhs: Self) -> Self { <$t>::wrapping_div(self, rhs.nonzero()) }
            #[inline(always)]
            fn wrapping_rem(self, rhs: Self) -> Self { <$t>::wrapping_rem(self, rhs.nonzero()) }
            #[inline(always)]
            fn wrapping_neg(self) -> Self { <$t>::wrapping_neg(self) }
            #[inline(always)]
            fn wrapping_abs(self) -> Self { <$t>::wrapping_abs(self) }
            #[inline(always)]
            fn nonzero(self) -> Self { self | <$t>::from(self == 0) }
            const MIN: Self = <$t>::MIN;
        }
    )*};
}
int_native!(i8, i16, i32, i64, i128);

/// A binary arithmetic operation, as a zero-sized type so each operation is
/// its own monomorph. Neither method panics.
pub trait BinaryOp: Copy + Send + Sync + 'static {
    /// The Spark operator.
    const OP: ArithOp;
    /// The wrapped (two's-complement) result.
    fn wrapping<T: IntNative>(a: T, b: T) -> T;
    /// The wrapped result and whether it overflowed; branch-free.
    fn overflowing<T: IntNative>(a: T, b: T) -> (T, bool);
}

/// `+`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Add;
/// `-`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Sub;
/// `*`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Mul;

macro_rules! binary_op {
    ($($t:ident: $op:ident, $w:ident, $o:ident);*) => {$(
        impl BinaryOp for $t {
            const OP: ArithOp = ArithOp::$op;
            #[inline(always)]
            fn wrapping<T: IntNative>(a: T, b: T) -> T { a.$w(b) }
            #[inline(always)]
            fn overflowing<T: IntNative>(a: T, b: T) -> (T, bool) { a.$o(b) }
        }
    )*};
}
binary_op!(
    Add: Add, wrapping_add, overflowing_add;
    Sub: Subtract, wrapping_sub, overflowing_sub;
    Mul: Multiply, wrapping_mul, overflowing_mul
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ops_wrap_and_flag() {
        assert_eq!(Add::overflowing(i64::MAX, 1), (i64::MIN, true));
        assert_eq!(Sub::overflowing(i32::MIN, 1), (i32::MAX, true));
        assert_eq!(Mul::overflowing(i8::MIN, -1), (i8::MIN, true));
        assert_eq!(Mul::wrapping(3_i16, 4), 12);
        assert_eq!(Sub::wrapping(1_i128, 2), -1);
        assert_eq!(Add::wrapping(1_i8, 2), 3);
        assert_eq!(Mul::OP.try_function(), "try_multiply");
        assert_eq!(
            [Add::OP, Sub::OP, Mul::OP].map(ArithOp::symbol),
            ["+", "-", "*"]
        );
        assert_eq!(IntNative::wrapping_div(i8::MIN, -1), i8::MIN);
        assert_eq!(IntNative::wrapping_rem(i64::MIN, -1), 0);
        assert_eq!(IntNative::wrapping_div(7_i32, 0), 7);
        assert_eq!(IntNative::wrapping_rem(-7_i16, 0), 0);
        assert_eq!((0_i128.nonzero(), (-3_i128).nonzero()), (1, -3));
        assert_eq!(
            (
                IntNative::wrapping_neg(i32::MIN),
                IntNative::wrapping_abs(-5_i64)
            ),
            (i32::MIN, 5)
        );
        assert_eq!(Sub::OP.try_function(), "try_subtract");
        let o: Operand<'_, arrow_array::types::Int64Type> = Operand::Scalar(Some(1));
        let c = o;
        assert!(matches!(
            (o, c),
            (Operand::Scalar(Some(1)), Operand::Scalar(_))
        ));
        assert!(format!("{:?}", o.clone()).contains("Scalar"));
    }
}
