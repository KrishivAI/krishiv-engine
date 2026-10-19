// SPDX-License-Identifier: Apache-2.0
//! Decimal physical carriers (design P2-T160): `Decimal64` (`i64` values,
//! precision at most 18) and `Decimal128` (`i128`, any precision). The
//! kernels take either carrier on each side and write the result in the
//! carrier the caller asks for; the Spark type `(precision, scale)` alone
//! decides the result values, so both carriers give the same results.

use std::sync::Arc;

use arrow_array::types::{Decimal64Type, Decimal128Type};
use arrow_array::{Array, ArrayRef, Decimal64Array, Decimal128Array, PrimitiveArray};
use arrow_buffer::{NullBuffer, ScalarBuffer};
use arrow_schema::DataType;

use super::{DecimalType, MAX_I64_PRECISION};
use crate::error::KernelError;
use crate::ops::Operand;

mod sealed {
    pub trait Sealed {}
    impl Sealed for i64 {}
    impl Sealed for i128 {}
}

/// The native value of a decimal carrier: `i64` (`Decimal64`) or `i128`
/// (`Decimal128`).
pub trait DecimalNative:
    sealed::Sealed + arrow_buffer::ArrowNativeType + Default + Send + Sync + 'static
{
    /// The value as an `i128` (exact).
    fn wide(self) -> i128;
    /// The low 64 bits, and whether they lost information.
    fn low64(self) -> (i64, bool);
    /// The ones'-complement magnitude (`|v|` for `v >= 0`, `|v| - 1`
    /// otherwise): branch-free, and `|v| <= m + 1`.
    fn ones_magnitude(self) -> u128;
    /// `-v`, wrapping.
    fn negate(self) -> Self;
    /// `|v|`, wrapping.
    fn abs(self) -> Self;
}

impl DecimalNative for i64 {
    #[inline(always)]
    fn wide(self) -> i128 {
        i128::from(self)
    }
    #[inline(always)]
    fn low64(self) -> (i64, bool) {
        (self, false)
    }
    #[inline(always)]
    fn ones_magnitude(self) -> u128 {
        u128::from((self ^ (self >> 63)).cast_unsigned())
    }
    #[inline(always)]
    fn negate(self) -> Self {
        self.wrapping_neg()
    }
    #[inline(always)]
    fn abs(self) -> Self {
        self.wrapping_abs()
    }
}

impl DecimalNative for i128 {
    #[inline(always)]
    fn wide(self) -> i128 {
        self
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "taking the low 64 bits is the point; a lane that does not fit is flagged"
    )]
    #[inline(always)]
    fn low64(self) -> (i64, bool) {
        let low = self as i64;
        (low, i128::from(low) != self)
    }
    #[inline(always)]
    fn ones_magnitude(self) -> u128 {
        (self ^ (self >> 127)).cast_unsigned()
    }
    #[inline(always)]
    fn negate(self) -> Self {
        self.wrapping_neg()
    }
    #[inline(always)]
    fn abs(self) -> Self {
        self.wrapping_abs()
    }
}

/// A decimal result native: the lanes write `i64` straight into a
/// `Decimal64` result, `i128` otherwise.
pub(crate) trait DecimalOut: Copy + Default {
    fn of64(v: i64) -> Self;
    /// The result array of `carrier` and type `t` (validity `nulls`). Every
    /// valid value is within the type's precision, so narrowing an `i128`
    /// into a `Decimal64` keeps it; values under NULL slots are arbitrary.
    fn finish(
        values: Vec<Self>,
        nulls: Option<NullBuffer>,
        carrier: DecimalCarrier,
        t: DecimalType,
    ) -> Result<ArrayRef, KernelError>;
}

fn internal(e: &arrow_schema::ArrowError) -> KernelError {
    KernelError::internal(e.to_string())
}

impl DecimalOut for i64 {
    #[inline(always)]
    fn of64(v: i64) -> Self {
        v
    }
    fn finish(
        values: Vec<Self>,
        nulls: Option<NullBuffer>,
        carrier: DecimalCarrier,
        (p, s): DecimalType,
    ) -> Result<ArrayRef, KernelError> {
        Ok(match carrier {
            DecimalCarrier::Decimal64 => Arc::new(
                Decimal64Array::try_new(ScalarBuffer::from(values), nulls)
                    .and_then(|a| a.with_precision_and_scale(p, s))
                    .map_err(|e| internal(&e))?,
            ),
            DecimalCarrier::Decimal128 => {
                let wide: Vec<i128> = values.into_iter().map(i128::from).collect();
                <i128 as DecimalOut>::finish(wide, nulls, carrier, (p, s))?
            }
        })
    }
}

impl DecimalOut for i128 {
    #[inline(always)]
    fn of64(v: i64) -> Self {
        i128::from(v)
    }
    fn finish(
        values: Vec<Self>,
        nulls: Option<NullBuffer>,
        carrier: DecimalCarrier,
        (p, s): DecimalType,
    ) -> Result<ArrayRef, KernelError> {
        Ok(match carrier {
            DecimalCarrier::Decimal128 => Arc::new(
                Decimal128Array::try_new(ScalarBuffer::from(values), nulls)
                    .and_then(|a| a.with_precision_and_scale(p, s))
                    .map_err(|e| internal(&e))?,
            ),
            DecimalCarrier::Decimal64 => {
                let narrow: Vec<i64> = values.into_iter().map(|v| v.low64().0).collect();
                <i64 as DecimalOut>::finish(narrow, nulls, carrier, (p, s))?
            }
        })
    }
}

/// The physical carrier of a decimal array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DecimalCarrier {
    /// Arrow `Decimal64`: `i64` values, precision at most 18.
    Decimal64,
    /// Arrow `Decimal128`: `i128` values, any precision up to 38.
    Decimal128,
}

impl DecimalCarrier {
    /// The Arrow type of `(p, s)` on this carrier.
    #[must_use]
    pub fn data_type(self, (p, s): DecimalType) -> DataType {
        match self {
            Self::Decimal64 => DataType::Decimal64(p, s),
            Self::Decimal128 => DataType::Decimal128(p, s),
        }
    }

    /// Checks that this carrier can hold precision `p`.
    pub(crate) fn check(self, p: u8) -> Result<(), KernelError> {
        if self == Self::Decimal64 && p > MAX_I64_PRECISION {
            return Err(KernelError::internal(format!(
                "decimal precision {p} on a Decimal64 carrier"
            )));
        }
        Ok(())
    }
}

/// One side of a decimal kernel on either carrier.
#[derive(Debug, Clone, Copy)]
pub enum DecimalOperand<'a> {
    /// A `Decimal64` column.
    Decimal64(&'a Decimal64Array),
    /// A `Decimal128` column.
    Decimal128(&'a Decimal128Array),
    /// A raw unscaled literal in the plan's type for its side (`None` is a
    /// NULL literal); literals carry no type, so it is not checked.
    Scalar(Option<i128>),
}

impl<'a> From<Operand<'a, Decimal128Type>> for DecimalOperand<'a> {
    fn from(o: Operand<'a, Decimal128Type>) -> Self {
        match o {
            Operand::Array(a) => Self::Decimal128(a),
            Operand::Scalar(v) => Self::Scalar(v),
        }
    }
}

impl<'a> From<Operand<'a, Decimal64Type>> for DecimalOperand<'a> {
    fn from(o: Operand<'a, Decimal64Type>) -> Self {
        match o {
            Operand::Array(a) => Self::Decimal64(a),
            Operand::Scalar(v) => Self::Scalar(v.map(i128::from)),
        }
    }
}

impl<'a> DecimalOperand<'a> {
    /// The operand of a decimal column of either carrier.
    ///
    /// # Errors
    /// `INTERNAL_ERROR` for a non-decimal (or `Decimal32`/`Decimal256`) column.
    pub fn of_array(a: &'a dyn Array) -> Result<Self, KernelError> {
        use arrow_array::cast::AsArray;
        match a.data_type() {
            DataType::Decimal64(..) => Ok(Self::Decimal64(a.as_primitive())),
            DataType::Decimal128(..) => Ok(Self::Decimal128(a.as_primitive())),
            t => Err(KernelError::internal(format!(
                "decimal kernel operand of type {t}"
            ))),
        }
    }

    /// `(precision, scale)` of a column (`None` for a literal).
    pub(crate) fn decimal_type(self) -> Option<DecimalType> {
        match self {
            Self::Decimal64(a) => Some((a.precision(), a.scale())),
            Self::Decimal128(a) => Some((a.precision(), a.scale())),
            Self::Scalar(_) => None,
        }
    }

    /// The Arrow type of a column, for messages.
    pub(crate) fn arrow_type(self) -> Option<DataType> {
        match self {
            Self::Decimal64(a) => Some(a.data_type().clone()),
            Self::Decimal128(a) => Some(a.data_type().clone()),
            Self::Scalar(_) => None,
        }
    }

    /// The operand on 64-bit lanes: a `Decimal64` column, or a literal that
    /// fits an `i64`.
    pub(crate) fn narrow(self) -> Option<Operand<'a, Decimal64Type>> {
        match self {
            Self::Decimal64(a) => Some(Operand::Array(a)),
            Self::Scalar(None) => Some(Operand::Scalar(None)),
            Self::Scalar(Some(v)) => i64::try_from(v).ok().map(|v| Operand::Scalar(Some(v))),
            Self::Decimal128(_) => None,
        }
    }

    /// The operand on 128-bit lanes; a `Decimal64` column is widened (a
    /// copy, only for mixed-carrier operands and the 256-bit path).
    pub(crate) fn wide(self) -> Result<WideOperand<'a>, KernelError> {
        Ok(match self {
            Self::Decimal128(a) => WideOperand::Borrowed(Operand::Array(a)),
            Self::Scalar(v) => WideOperand::Borrowed(Operand::Scalar(v)),
            Self::Decimal64(a) => {
                let w: PrimitiveArray<Decimal128Type> = a
                    .unary::<_, Decimal128Type>(i128::from)
                    .with_precision_and_scale(a.precision(), a.scale())
                    .map_err(|e| internal(&e))?;
                WideOperand::Owned(w)
            }
        })
    }
}

/// A `Decimal128` operand, borrowed or widened from a `Decimal64` column.
pub(crate) enum WideOperand<'a> {
    Borrowed(Operand<'a, Decimal128Type>),
    Owned(Decimal128Array),
}

impl WideOperand<'_> {
    pub(crate) fn operand(&self) -> Operand<'_, Decimal128Type> {
        match self {
            Self::Borrowed(o) => *o,
            Self::Owned(a) => Operand::Array(a),
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Decimal64Type, Decimal128Type};

    use super::*;

    /// Either result native lands on either carrier with the same values.
    #[test]
    fn finish_on_either_carrier() {
        let nulls = Some(NullBuffer::from(vec![true, false]));
        let a = <i64 as DecimalOut>::finish(
            vec![-7, 3],
            nulls.clone(),
            DecimalCarrier::Decimal128,
            (5, 2),
        )
        .unwrap();
        assert_eq!(a.data_type(), &DataType::Decimal128(5, 2));
        assert_eq!(
            (a.as_primitive::<Decimal128Type>().value(0), a.null_count()),
            (-7, 1)
        );
        let b = <i128 as DecimalOut>::finish(vec![-7, 3], nulls, DecimalCarrier::Decimal64, (5, 2))
            .unwrap();
        assert_eq!(b.data_type(), &DataType::Decimal64(5, 2));
        assert_eq!(b.as_primitive::<Decimal64Type>().value(0), -7);
        assert!(DecimalCarrier::Decimal64.check(19).is_err());
        assert!(DecimalCarrier::Decimal128.check(38).is_ok());
    }

    #[test]
    fn operands() {
        let a = arrow_array::Decimal64Array::from(vec![5_i64])
            .with_precision_and_scale(3, 1)
            .unwrap();
        let o = DecimalOperand::from(Operand::Array(&a));
        assert_eq!(o.arrow_type(), Some(DataType::Decimal64(3, 1)));
        assert!(matches!(
            DecimalOperand::from(Operand::<Decimal64Type>::Scalar(Some(-4))),
            DecimalOperand::Scalar(Some(-4))
        ));
        assert_eq!(DecimalOperand::Scalar(None).arrow_type(), None);
        assert_eq!((-5_i64).abs(), 5);
    }

    #[test]
    fn natives() {
        assert_eq!(
            i64::MIN.ones_magnitude(),
            u128::from(i64::MAX.cast_unsigned())
        );
        assert_eq!((-1_i128).ones_magnitude(), 0);
        assert_eq!(((1_i128 << 64) + 5).low64(), (5, true));
        assert_eq!(i64::MIN.low64(), (i64::MIN, false));
        assert_eq!((i64::MIN.negate(), (-3_i128).abs()), (i64::MIN, 3));
    }
}
