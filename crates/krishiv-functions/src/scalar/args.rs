// SPDX-License-Identifier: Apache-2.0
//! Arguments at invocation and their conversion to kernel operands.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Int8Type, Int16Type, Int32Type, TimestampMicrosecondType};
use arrow_array::{
    Array, ArrayRef, ArrowPrimitiveType, BinaryArray, Int32Array, PrimitiveArray, StringArray,
    UInt32Array,
};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::error::{InternalError, KrishivError, KrishivUnsupportedFeature};
use krishiv_kernels::Operand;
use krishiv_kernels::string::{BinOperand, StrOperand};

/// An argument at invocation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Arg {
    /// A column of `rows` values.
    Array(ArrayRef),
    /// A literal, as a one-row array.
    Scalar(ArrayRef),
}

impl Arg {
    /// The values.
    #[must_use]
    pub fn array(&self) -> &ArrayRef {
        match self {
            Self::Array(a) | Self::Scalar(a) => a,
        }
    }

    /// The same values as a column.
    pub(super) fn as_column(&self) -> Self {
        Self::Array(Arc::clone(self.array()))
    }

    /// Checks the argument against its bound type and the batch's rows.
    pub(super) fn check(&self, bound: &DataType, rows: usize, f: &str) -> Result<(), KrishivError> {
        let (a, want) = match self {
            Self::Array(a) => (a, rows),
            Self::Scalar(a) => (a, 1),
        };
        if a.len() != want || a.data_type() != bound {
            return Err(internal(format!(
                "{f}: argument of {} rows of {} where {want} rows of {bound} were bound",
                a.len(),
                a.data_type()
            )));
        }
        Ok(())
    }
}

/// `INTERNAL_ERROR`.
pub(super) fn internal(message: String) -> KrishivError {
    InternalError { message }.into()
}

/// `KRISHIV_UNSUPPORTED_FEATURE`.
pub(super) fn unsupported(feature: String) -> KrishivError {
    KrishivUnsupportedFeature { feature }.into()
}

/// `a`'s single row, `rows` times.
pub(super) fn broadcast(a: &ArrayRef, rows: usize) -> Result<ArrayRef, KrishivError> {
    let zeros = UInt32Array::from(vec![0_u32; rows]);
    arrow_select::take::take(a.as_ref(), &zeros, None).map_err(|e| internal(e.to_string()))
}

/// Whether `t` is a string carrier.
pub(super) fn is_string(t: &DataType) -> bool {
    matches!(t, DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8)
}

/// Whether `t` is a binary carrier.
pub(super) fn is_binary(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Binary | DataType::BinaryView | DataType::LargeBinary
    )
}

/// Whether `t` is a `timestamp` (an instant) or `timestamp_ntz`.
pub(super) fn is_timestamp(t: &DataType) -> bool {
    matches!(t, DataType::Timestamp(TimeUnit::Microsecond, _))
}

/// The arguments of one invocation, all of one length (`rows`) or literals.
pub(super) struct Inputs<'a> {
    args: &'a [Arg],
    rows: usize,
}

/// A string argument, as `Utf8`.
pub(super) enum Str {
    Array(StringArray),
    Scalar(Option<String>),
}

impl Str {
    pub(super) fn operand(&self) -> StrOperand<'_, i32> {
        match self {
            Self::Array(a) => StrOperand::Array(a),
            Self::Scalar(s) => StrOperand::Scalar(s.as_deref()),
        }
    }
}

/// A binary argument, as `Binary`.
pub(super) enum Bin {
    Array(BinaryArray),
    Scalar(Option<Vec<u8>>),
}

impl Bin {
    pub(super) fn operand(&self) -> BinOperand<'_, i32> {
        match self {
            Self::Array(a) => BinOperand::Array(a),
            Self::Scalar(s) => BinOperand::Scalar(s.as_deref()),
        }
    }
}

/// A primitive argument.
pub(super) enum Prim<T: ArrowPrimitiveType> {
    Array(PrimitiveArray<T>),
    Scalar(Option<T::Native>),
}

impl<T: ArrowPrimitiveType> Prim<T> {
    pub(super) fn operand(&self) -> Operand<'_, T> {
        match self {
            Self::Array(a) => Operand::Array(a),
            Self::Scalar(v) => Operand::Scalar(*v),
        }
    }
}

pub(super) fn to_utf8(a: &dyn Array) -> Result<StringArray, KrishivError> {
    match a.data_type() {
        DataType::Utf8 => Ok(a.as_string::<i32>().clone()),
        DataType::Utf8View => Ok(a.as_string_view().iter().collect()),
        DataType::LargeUtf8 => Ok(a.as_string::<i64>().iter().collect()),
        t => Err(internal(format!("a string argument of type {t}"))),
    }
}

pub(super) fn to_binary(a: &dyn Array) -> Result<BinaryArray, KrishivError> {
    match a.data_type() {
        DataType::Binary => Ok(a.as_binary::<i32>().clone()),
        DataType::BinaryView => Ok(a.as_binary_view().iter().collect()),
        DataType::LargeBinary => Ok(a.as_binary::<i64>().iter().collect()),
        t => Err(internal(format!("a binary argument of type {t}"))),
    }
}

impl<'a> Inputs<'a> {
    pub(super) fn new(args: &'a [Arg], rows: usize) -> Self {
        Self { args, rows }
    }

    pub(super) fn rows(&self) -> usize {
        self.rows
    }

    /// Argument `i` as a column of `rows` values, with string and binary
    /// carriers read as `Utf8` and `Binary`.
    pub(super) fn plain_column(&self, i: usize) -> Result<ArrayRef, KrishivError> {
        let c = self.column(i)?;
        Ok(match c.data_type() {
            t if is_string(t) => Arc::new(to_utf8(c.as_ref())?),
            t if is_binary(t) => Arc::new(to_binary(c.as_ref())?),
            _ => c,
        })
    }

    /// Argument `i` as a primitive column of `rows` values.
    pub(super) fn array<T: ArrowPrimitiveType>(
        &self,
        i: usize,
    ) -> Result<PrimitiveArray<T>, KrishivError> {
        let c = self.column(i)?;
        c.as_primitive_opt::<T>().cloned().ok_or_else(|| {
            internal(format!(
                "argument {i} of type {} read as {}",
                c.data_type(),
                T::DATA_TYPE
            ))
        })
    }

    /// Argument `i` as a boolean column of `rows` values.
    pub(super) fn booleans(&self, i: usize) -> Result<arrow_array::BooleanArray, KrishivError> {
        let c = self.column(i)?;
        c.as_boolean_opt()
            .cloned()
            .ok_or_else(|| internal(format!("a boolean argument of type {}", c.data_type())))
    }

    /// Argument `i`.
    fn arg(&self, i: usize) -> Result<&'a Arg, KrishivError> {
        self.args
            .get(i)
            .ok_or_else(|| internal(format!("missing argument {i}")))
    }

    /// Argument `i` as a column of `rows` values (a literal is repeated).
    pub(super) fn column(&self, i: usize) -> Result<ArrayRef, KrishivError> {
        match self.arg(i)? {
            Arg::Array(a) => Ok(Arc::clone(a)),
            Arg::Scalar(a) => broadcast(a, self.rows),
        }
    }

    /// Argument `i` as a string column of `rows` values.
    pub(super) fn strings(&self, i: usize) -> Result<StringArray, KrishivError> {
        to_utf8(self.column(i)?.as_ref())
    }

    /// Argument `i` as a binary column of `rows` values.
    pub(super) fn binaries(&self, i: usize) -> Result<BinaryArray, KrishivError> {
        to_binary(self.column(i)?.as_ref())
    }

    /// Argument `i` as a string operand.
    pub(super) fn str(&self, i: usize) -> Result<Str, KrishivError> {
        Ok(match self.arg(i)? {
            Arg::Array(a) => Str::Array(to_utf8(a.as_ref())?),
            Arg::Scalar(a) => {
                let s = to_utf8(a.as_ref())?;
                Str::Scalar(s.is_valid(0).then(|| s.value(0).to_owned()))
            }
        })
    }

    /// Argument `i` as a binary operand.
    pub(super) fn bin(&self, i: usize) -> Result<Bin, KrishivError> {
        Ok(match self.arg(i)? {
            Arg::Array(a) => Bin::Array(to_binary(a.as_ref())?),
            Arg::Scalar(a) => {
                let b = to_binary(a.as_ref())?;
                Bin::Scalar(b.is_valid(0).then(|| b.value(0).to_vec()))
            }
        })
    }

    /// Argument `i` as a primitive operand of type `T`.
    pub(super) fn prim<T: ArrowPrimitiveType>(&self, i: usize) -> Result<Prim<T>, KrishivError> {
        let a = self.arg(i)?;
        let v = a.array().as_primitive_opt::<T>().ok_or_else(|| {
            internal(format!(
                "argument {i} of type {} read as {}",
                a.array().data_type(),
                T::DATA_TYPE
            ))
        })?;
        Ok(match a {
            Arg::Array(_) => Prim::Array(v.clone()),
            Arg::Scalar(_) => Prim::Scalar(v.is_valid(0).then(|| v.value(0))),
        })
    }

    /// Argument `i`, a `tinyint`, `smallint` or `int`, as an `int` operand.
    pub(super) fn int(&self, i: usize) -> Result<Prim<Int32Type>, KrishivError> {
        let a = self.arg(i)?;
        let widened: Int32Array = match a.array().data_type() {
            DataType::Int32 => return self.prim::<Int32Type>(i),
            DataType::Int16 => a.array().as_primitive::<Int16Type>().unary(i32::from),
            DataType::Int8 => a.array().as_primitive::<Int8Type>().unary(i32::from),
            t => return Err(internal(format!("an int argument of type {t}"))),
        };
        Ok(match a {
            Arg::Array(_) => Prim::Array(widened),
            Arg::Scalar(_) => Prim::Scalar(widened.is_valid(0).then(|| widened.value(0))),
        })
    }

    /// Argument `i` as a `date` operand.
    pub(super) fn date(&self, i: usize) -> Result<Prim<Date32Type>, KrishivError> {
        self.prim::<Date32Type>(i)
    }

    /// Argument `i` as a date column of `rows` values.
    pub(super) fn dates(&self, i: usize) -> Result<PrimitiveArray<Date32Type>, KrishivError> {
        let a = self.column(i)?;
        a.as_primitive_opt::<Date32Type>()
            .cloned()
            .ok_or_else(|| internal(format!("a date argument of type {}", a.data_type())))
    }

    /// Argument `i` as a timestamp column of `rows` values.
    pub(super) fn timestamps(
        &self,
        i: usize,
    ) -> Result<PrimitiveArray<TimestampMicrosecondType>, KrishivError> {
        let a = self.column(i)?;
        a.as_primitive_opt::<TimestampMicrosecondType>()
            .cloned()
            .ok_or_else(|| internal(format!("a timestamp argument of type {}", a.data_type())))
    }
}
