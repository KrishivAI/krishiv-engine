// SPDX-License-Identifier: Apache-2.0
//! Spark data types and their Spark renderings (`simpleString`, `catalogString`,
//! `sql`, `typeName`).

use std::fmt;
use std::str::FromStr;

use krishiv_common::error::KrishivError;

use crate::error::{self, Result};

/// A Spark SQL data type.
///
/// Covers void (the type of a bare `NULL`), boolean, integral, floating-point, string, binary, decimal, date and
/// timestamp types. Nested, interval, char/varchar and variant types are not
/// represented yet; parsing them yields `KRISHIV_UNSUPPORTED_FEATURE`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DataType {
    /// `VOID`: the type of an untyped `NULL`; every value is null.
    Null,
    /// `BOOLEAN`.
    Boolean,
    /// `TINYINT` (8-bit signed).
    Byte,
    /// `SMALLINT` (16-bit signed).
    Short,
    /// `INT` (32-bit signed).
    Integer,
    /// `BIGINT` (64-bit signed).
    Long,
    /// `FLOAT` (IEEE 754 binary32).
    Float,
    /// `DOUBLE` (IEEE 754 binary64).
    Double,
    /// `STRING`: binary-safe, may hold bytes that are not valid UTF-8.
    String,
    /// `BINARY`.
    Binary,
    /// `DECIMAL(p, s)`.
    Decimal(DecimalType),
    /// `DATE`: days since 1970-01-01.
    Date,
    /// `TIMESTAMP` (`TIMESTAMP_LTZ`): microseconds since the epoch, a UTC
    /// instant shown in the session time zone.
    Timestamp,
    /// `TIMESTAMP_NTZ`: microseconds since the epoch, no time zone.
    TimestampNtz,
}

impl DataType {
    /// Spark's `DataType.simpleString`, e.g. `int`, `decimal(10,2)`.
    pub fn simple_string(&self) -> String {
        match self {
            Self::Decimal(d) => format!("decimal({},{})", d.precision, d.scale),
            other => other.keyword().to_owned(),
        }
    }

    /// Spark's `DataType.catalogString`; equal to [`Self::simple_string`] for
    /// every type represented here.
    pub fn catalog_string(&self) -> String {
        self.simple_string()
    }

    /// Spark's `DataType.sql`, e.g. `INT`, `DECIMAL(10,2)`.
    pub fn sql(&self) -> String {
        self.simple_string().to_ascii_uppercase()
    }

    /// Spark's `DataType.typeName` (the name used in Spark's JSON schemas),
    /// e.g. `integer`, `long`, `decimal(10,2)`.
    pub fn type_name(&self) -> String {
        match self {
            Self::Byte => "byte".to_owned(),
            Self::Short => "short".to_owned(),
            Self::Integer => "integer".to_owned(),
            Self::Long => "long".to_owned(),
            other => other.simple_string(),
        }
    }

    /// The `simpleString` keyword of a non-parameterized type.
    fn keyword(&self) -> &'static str {
        match self {
            Self::Null => "void",
            Self::Boolean => "boolean",
            Self::Byte => "tinyint",
            Self::Short => "smallint",
            Self::Integer => "int",
            Self::Long => "bigint",
            Self::Float => "float",
            Self::Double => "double",
            Self::String => "string",
            Self::Binary => "binary",
            Self::Decimal(_) => "decimal",
            Self::Date => "date",
            Self::Timestamp => "timestamp",
            Self::TimestampNtz => "timestamp_ntz",
        }
    }
}

/// Formats as Spark's `simpleString`.
impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.simple_string())
    }
}

/// Parses a Spark DDL type string; see [`crate::parse_data_type`].
impl FromStr for DataType {
    type Err = KrishivError;

    fn from_str(s: &str) -> Result<Self> {
        crate::parse_data_type(s)
    }
}

/// How a decimal's unscaled value is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DecimalStorage {
    /// Precision ≤ 18: the unscaled value fits an `i64` (the fast path).
    I64,
    /// Precision 19..=38: the unscaled value needs an `i128`.
    I128,
}

/// Spark's `DecimalType(precision, scale)`, with Spark's bounds:
/// `0 ≤ scale ≤ precision ≤ 38`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecimalType {
    precision: u8,
    scale: u8,
}

impl DecimalType {
    /// Spark's `DecimalType.MAX_PRECISION`.
    pub const MAX_PRECISION: u8 = 38;
    /// Spark's `DecimalType.MAX_SCALE`.
    pub const MAX_SCALE: u8 = 38;
    /// Spark's `Decimal.MAX_LONG_DIGITS`: the largest precision whose unscaled
    /// values always fit an `i64`.
    pub const MAX_LONG_DIGITS: u8 = 18;
    /// Spark's `DecimalType.USER_DEFAULT`, `decimal(10,0)`: the type of a bare
    /// `DECIMAL` in DDL.
    pub const USER_DEFAULT: Self = Self {
        precision: 10,
        scale: 0,
    };
    /// Spark's `DecimalType.SYSTEM_DEFAULT`, `decimal(38,18)`.
    pub const SYSTEM_DEFAULT: Self = Self {
        precision: 38,
        scale: 18,
    };

    /// Builds `decimal(precision, scale)`, failing as Spark's `DecimalType`
    /// constructor does.
    ///
    /// # Errors
    /// `DECIMAL_PRECISION_EXCEEDS_MAX_PRECISION` when `precision > 38`; an
    /// error when `scale > precision` (checked first, as in Spark).
    pub fn try_new(precision: u32, scale: u32) -> Result<Self> {
        if scale > precision {
            return Err(error::scale_exceeds_precision(
                &precision.to_string(),
                &scale.to_string(),
            ));
        }
        match (u8::try_from(precision), u8::try_from(scale)) {
            (Ok(p), Ok(s)) if p <= Self::MAX_PRECISION => Ok(Self {
                precision: p,
                scale: s,
            }),
            _ => Err(error::precision_exceeds_max(&precision.to_string())),
        }
    }

    /// Total number of digits.
    pub const fn precision(self) -> u8 {
        self.precision
    }

    /// Digits after the decimal point.
    pub const fn scale(self) -> u8 {
        self.scale
    }

    /// The physical storage of the unscaled value: `i64` for precision ≤ 18.
    pub const fn storage(self) -> DecimalStorage {
        if self.precision <= Self::MAX_LONG_DIGITS {
            DecimalStorage::I64
        } else {
            DecimalStorage::I128
        }
    }
}

impl From<DecimalType> for DataType {
    fn from(d: DecimalType) -> Self {
        Self::Decimal(d)
    }
}
