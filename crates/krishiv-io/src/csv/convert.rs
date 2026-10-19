// SPDX-License-Identifier: Apache-2.0
//! Values of a record as typed columns: Spark's `UnivocityParser` value
//! converters, and the column builders.

use std::sync::Arc;

use arrow_array::builder::{
    BinaryBuilder, BinaryViewBuilder, BooleanBuilder, Date32Builder, Decimal128Builder,
    Float32Builder, Float64Builder, Int8Builder, Int16Builder, Int32Builder, Int64Builder,
    LargeBinaryBuilder, LargeStringBuilder, StringBuilder, StringViewBuilder,
    TimestampMicrosecondBuilder,
};
use arrow_array::{ArrayRef, new_null_array};
use arrow_schema::{DataType, TimeUnit};
use krishiv_common::error::KrishivError;
use krishiv_kernels::text::{
    java_parse_decimal, java_parse_double, java_parse_float, java_parse_integral,
};

use super::options::CsvOptions;
use crate::error::unsupported;
use crate::text::{Datetimes, spark_type_name};

/// A string column of any Arrow string type.
pub(crate) enum StrBuilder {
    View(StringViewBuilder),
    Utf8(StringBuilder),
    Large(LargeStringBuilder),
}

impl StrBuilder {
    /// A builder for `dt`, if it is a string type.
    fn for_type(dt: &DataType) -> Option<Self> {
        match dt {
            DataType::Utf8View => Some(Self::View(StringViewBuilder::new())),
            DataType::Utf8 => Some(Self::Utf8(StringBuilder::new())),
            DataType::LargeUtf8 => Some(Self::Large(LargeStringBuilder::new())),
            _ => None,
        }
    }

    fn append(&mut self, s: Option<&str>) {
        match (self, s) {
            (Self::View(b), s) => b.append_option(s),
            (Self::Utf8(b), s) => b.append_option(s),
            (Self::Large(b), s) => b.append_option(s),
        }
    }

    fn finish(&mut self) -> ArrayRef {
        match self {
            Self::View(b) => Arc::new(b.finish()),
            Self::Utf8(b) => Arc::new(b.finish()),
            Self::Large(b) => Arc::new(b.finish()),
        }
    }
}

/// The Spark type of a column, as the CSV reader handles it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Kind {
    Bool,
    Int(u32),
    F32,
    F64,
    Decimal(u8, i8),
    Str,
    Bin,
    Date,
    Timestamp,
    TimestampNtz,
    Null,
}

impl Kind {
    /// The kind for an Arrow type.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for nested and interval types.
    pub(crate) fn of(dt: &DataType) -> Result<Self, KrishivError> {
        Ok(match dt {
            DataType::Boolean => Self::Bool,
            DataType::Int8 => Self::Int(8),
            DataType::Int16 => Self::Int(16),
            DataType::Int32 => Self::Int(32),
            DataType::Int64 => Self::Int(64),
            DataType::Float32 => Self::F32,
            DataType::Float64 => Self::F64,
            DataType::Decimal128(p, s) => Self::Decimal(*p, *s),
            DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 => Self::Str,
            DataType::Binary | DataType::BinaryView | DataType::LargeBinary => Self::Bin,
            DataType::Date32 => Self::Date,
            DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => Self::Timestamp,
            DataType::Timestamp(TimeUnit::Microsecond, None) => Self::TimestampNtz,
            DataType::Null => Self::Null,
            other => {
                return Err(unsupported(format!(
                    "the CSV data source with a column of type {}",
                    spark_type_name(other)
                )));
            }
        })
    }
}

/// One converted value.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Val {
    Null,
    Bool(bool),
    I64(i64),
    I128(i128),
    F32(f32),
    F64(f64),
    Text(String),
    Bytes(Vec<u8>),
}

/// Why a value did not convert.
#[derive(Debug)]
pub(crate) enum Conv {
    /// The text is not a value of the type: the record is malformed.
    Malformed,
    /// The conversion cannot be done at all (a region time zone).
    Fatal(KrishivError),
}

impl From<KrishivError> for Conv {
    fn from(e: KrishivError) -> Self {
        Self::Fatal(e)
    }
}

/// Converts the tokens of records.
pub(crate) struct Converter<'a> {
    pub o: &'a CsvOptions,
    pub dt: &'a Datetimes,
}

impl Converter<'_> {
    /// `token` as a value of `kind`.
    ///
    /// # Errors
    /// [`Conv::Malformed`] for text that is not a value of the type.
    pub(crate) fn convert(&self, kind: &Kind, token: Option<&str>) -> Result<Val, Conv> {
        let Some(s) = token else {
            return Ok(Val::Null);
        };
        if s == self.o.null_value {
            return Ok(Val::Null);
        }
        let bad = || Conv::Malformed;
        Ok(match kind {
            Kind::Null => Val::Null,
            Kind::Str => Val::Text(s.to_owned()),
            Kind::Bin => Val::Bytes(s.as_bytes().to_vec()),
            Kind::Bool => {
                if s.eq_ignore_ascii_case("true") {
                    Val::Bool(true)
                } else if s.eq_ignore_ascii_case("false") {
                    Val::Bool(false)
                } else {
                    return Err(bad());
                }
            }
            Kind::Int(bits) => Val::I64(java_parse_integral(s, *bits).ok_or_else(bad)?),
            Kind::F32 => Val::F32(if s == self.o.nan {
                f32::NAN
            } else if s == self.o.neg_inf {
                f32::NEG_INFINITY
            } else if s == self.o.pos_inf {
                f32::INFINITY
            } else {
                java_parse_float(s).ok_or_else(bad)?
            }),
            Kind::F64 => Val::F64(if s == self.o.nan {
                f64::NAN
            } else if s == self.o.neg_inf {
                f64::NEG_INFINITY
            } else if s == self.o.pos_inf {
                f64::INFINITY
            } else {
                java_parse_double(s).ok_or_else(bad)?
            }),
            Kind::Decimal(p, sc) => {
                let sc = u8::try_from(*sc).map_err(|_| bad())?;
                let v = java_parse_decimal(&s.replace(',', ""), *p, sc).ok_or_else(bad)?;
                Val::I128(v)
            }
            Kind::Date => Val::I64(i64::from(self.dt.parse_date(s)?.ok_or_else(bad)?)),
            Kind::Timestamp => Val::I64(self.dt.parse_timestamp(s)?.ok_or_else(bad)?),
            Kind::TimestampNtz => Val::I64(self.dt.parse_ntz(s)?.ok_or_else(bad)?),
        })
    }
}

/// Builds one output column.
pub(crate) enum Column {
    Bool(BooleanBuilder),
    I8(Int8Builder),
    I16(Int16Builder),
    I32(Int32Builder),
    I64(Int64Builder),
    F32(Float32Builder),
    F64(Float64Builder),
    Decimal(Decimal128Builder),
    Str(StrBuilder),
    Binary(BinaryBuilder),
    BinaryView(BinaryViewBuilder),
    LargeBinary(LargeBinaryBuilder),
    Date(Date32Builder),
    Timestamp(TimestampMicrosecondBuilder),
    Null(usize, DataType),
}

impl Column {
    /// A builder for the Arrow type `dt`.
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for a type CSV cannot hold.
    pub(crate) fn new(dt: &DataType) -> Result<Self, KrishivError> {
        Kind::of(dt)?;
        Ok(match dt {
            DataType::Boolean => Self::Bool(BooleanBuilder::new()),
            DataType::Int8 => Self::I8(Int8Builder::new()),
            DataType::Int16 => Self::I16(Int16Builder::new()),
            DataType::Int32 => Self::I32(Int32Builder::new()),
            DataType::Int64 => Self::I64(Int64Builder::new()),
            DataType::Float32 => Self::F32(Float32Builder::new()),
            DataType::Float64 => Self::F64(Float64Builder::new()),
            DataType::Decimal128(p, s) => Self::Decimal(
                Decimal128Builder::new()
                    .with_precision_and_scale(*p, *s)
                    .map_err(|e| crate::error::internal(e.to_string()))?,
            ),
            DataType::Binary => Self::Binary(BinaryBuilder::new()),
            DataType::BinaryView => Self::BinaryView(BinaryViewBuilder::new()),
            DataType::LargeBinary => Self::LargeBinary(LargeBinaryBuilder::new()),
            DataType::Date32 => Self::Date(Date32Builder::new()),
            DataType::Timestamp(_, tz) => {
                let b = TimestampMicrosecondBuilder::new();
                Self::Timestamp(match tz {
                    Some(tz) => b.with_timezone(Arc::clone(tz)),
                    None => b,
                })
            }
            DataType::Null => Self::Null(0, DataType::Null),
            other => match StrBuilder::for_type(other) {
                Some(b) => Self::Str(b),
                None => return Err(unsupported(format!("CSV column of type {other}"))),
            },
        })
    }

    /// Appends a value (a mismatched value is null).
    pub(crate) fn append(&mut self, v: &Val) {
        match (self, v) {
            (Self::Bool(b), Val::Bool(x)) => b.append_value(*x),
            (Self::I8(b), Val::I64(x)) => b.append_option(i8::try_from(*x).ok()),
            (Self::I16(b), Val::I64(x)) => b.append_option(i16::try_from(*x).ok()),
            (Self::I32(b), Val::I64(x)) => b.append_option(i32::try_from(*x).ok()),
            (Self::I64(b), Val::I64(x)) => b.append_value(*x),
            (Self::F32(b), Val::F32(x)) => b.append_value(*x),
            (Self::F64(b), Val::F64(x)) => b.append_value(*x),
            (Self::Decimal(b), Val::I128(x)) => b.append_value(*x),
            (Self::Str(b), Val::Text(x)) => b.append(Some(x)),
            (Self::Binary(b), Val::Bytes(x)) => b.append_value(x),
            (Self::BinaryView(b), Val::Bytes(x)) => b.append_value(x),
            (Self::LargeBinary(b), Val::Bytes(x)) => b.append_value(x),
            (Self::Date(b), Val::I64(x)) => b.append_option(i32::try_from(*x).ok()),
            (Self::Timestamp(b), Val::I64(x)) => b.append_value(*x),
            (Self::Null(n, _), _) => *n += 1,
            (c, _) => c.append_null(),
        }
    }

    /// Appends a NULL.
    pub(crate) fn append_null(&mut self) {
        match self {
            Self::Bool(b) => b.append_null(),
            Self::I8(b) => b.append_null(),
            Self::I16(b) => b.append_null(),
            Self::I32(b) => b.append_null(),
            Self::I64(b) => b.append_null(),
            Self::F32(b) => b.append_null(),
            Self::F64(b) => b.append_null(),
            Self::Decimal(b) => b.append_null(),
            Self::Str(b) => b.append(None),
            Self::Binary(b) => b.append_null(),
            Self::BinaryView(b) => b.append_null(),
            Self::LargeBinary(b) => b.append_null(),
            Self::Date(b) => b.append_null(),
            Self::Timestamp(b) => b.append_null(),
            Self::Null(n, _) => *n += 1,
        }
    }

    /// The column built so far (the builder starts over).
    pub(crate) fn finish(&mut self) -> ArrayRef {
        match self {
            Self::Bool(b) => Arc::new(b.finish()),
            Self::I8(b) => Arc::new(b.finish()),
            Self::I16(b) => Arc::new(b.finish()),
            Self::I32(b) => Arc::new(b.finish()),
            Self::I64(b) => Arc::new(b.finish()),
            Self::F32(b) => Arc::new(b.finish()),
            Self::F64(b) => Arc::new(b.finish()),
            Self::Decimal(b) => Arc::new(b.finish()),
            Self::Str(b) => b.finish(),
            Self::Binary(b) => Arc::new(b.finish()),
            Self::BinaryView(b) => Arc::new(b.finish()),
            Self::LargeBinary(b) => Arc::new(b.finish()),
            Self::Date(b) => Arc::new(b.finish()),
            Self::Timestamp(b) => Arc::new(b.finish()),
            Self::Null(n, dt) => {
                let rows = std::mem::take(n);
                new_null_array(dt, rows)
            }
        }
    }
}
