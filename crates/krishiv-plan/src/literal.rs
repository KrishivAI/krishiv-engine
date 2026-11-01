// SPDX-License-Identifier: Apache-2.0
//! Literal values.

use krishiv_types::{DataType, DecimalType};
pub use krishiv_types::{DayTimeField, YearMonthField};

/// A literal value, in Spark's physical representation for its type.
///
/// Equality compares floating-point values by their bits, so `NaN` equals
/// itself and `0.0` differs from `-0.0`: two literals are equal exactly
/// when they are the same literal. A string holding a URL with credentials
/// never prints them, through `Display` or `Debug` (ADR 0038).
#[derive(Clone)]
#[non_exhaustive]
pub enum Literal {
    /// `NULL`: of the given type, or untyped (Spark's `void`) when `None`.
    Null(Option<DataType>),
    /// `BOOLEAN`.
    Boolean(bool),
    /// `TINYINT`.
    Byte(i8),
    /// `SMALLINT`.
    Short(i16),
    /// `INT`.
    Integer(i32),
    /// `BIGINT`.
    Long(i64),
    /// `FLOAT`.
    Float(f32),
    /// `DOUBLE`.
    Double(f64),
    /// `DECIMAL(p, s)`: the value is `unscaled * 10^-s`.
    Decimal {
        /// The unscaled value.
        unscaled: i128,
        /// Precision and scale.
        data_type: DecimalType,
    },
    /// `STRING`.
    String(String),
    /// A `MAP<STRING, STRING>` literal: the JSON options of one call, as
    /// the Python wrappers build them (SQL has no map-literal syntax).
    Map(Vec<(String, String)>),
    /// `BINARY`.
    Binary(Vec<u8>),
    /// `DATE`: days since 1970-01-01.
    Date(i32),
    /// `TIMESTAMP`: microseconds since the epoch (UTC).
    Timestamp(i64),
    /// `TIMESTAMP_NTZ`: microseconds since the epoch, no time zone.
    TimestampNtz(i64),
    /// `INTERVAL ... start [TO end]` of type `INTERVAL YEAR [TO MONTH]` or
    /// `INTERVAL MONTH`.
    YearMonthInterval {
        /// The value, in months.
        months: i32,
        /// The leading field.
        start: YearMonthField,
        /// The trailing field; equal to `start` for a single-field type.
        end: YearMonthField,
    },
    /// `INTERVAL ... start [TO end]` of a day-time interval type, e.g.
    /// `INTERVAL DAY` or `INTERVAL HOUR TO SECOND`.
    DayTimeInterval {
        /// The value, in microseconds.
        micros: i64,
        /// The leading field.
        start: DayTimeField,
        /// The trailing field; equal to `start` for a single-field type.
        end: DayTimeField,
    },
    /// A calendar `INTERVAL`, which only casts and calendar arithmetic make.
    CalendarInterval {
        /// Months.
        months: i32,
        /// Days.
        days: i32,
        /// Microseconds.
        micros: i64,
    },
}

impl std::fmt::Debug for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Null(t) => f.debug_tuple("Null").field(t).finish(),
            Self::Boolean(v) => f.debug_tuple("Boolean").field(v).finish(),
            Self::Byte(v) => f.debug_tuple("Byte").field(v).finish(),
            Self::Short(v) => f.debug_tuple("Short").field(v).finish(),
            Self::Integer(v) => f.debug_tuple("Integer").field(v).finish(),
            Self::Long(v) => f.debug_tuple("Long").field(v).finish(),
            Self::Float(v) => f.debug_tuple("Float").field(v).finish(),
            Self::Double(v) => f.debug_tuple("Double").field(v).finish(),
            Self::Decimal {
                unscaled,
                data_type,
            } => f
                .debug_struct("Decimal")
                .field("unscaled", unscaled)
                .field("data_type", data_type)
                .finish(),
            Self::String(s) => match krishiv_common::secret::SecretUrl::parse(s) {
                Some(url) => f.debug_tuple("String").field(&url).finish(),
                None => f.debug_tuple("String").field(s).finish(),
            },
            Self::Map(pairs) => f.debug_tuple("Map").field(pairs).finish(),
            Self::Binary(v) => f.debug_tuple("Binary").field(v).finish(),
            Self::Date(v) => f.debug_tuple("Date").field(v).finish(),
            Self::Timestamp(v) => f.debug_tuple("Timestamp").field(v).finish(),
            Self::TimestampNtz(v) => f.debug_tuple("TimestampNtz").field(v).finish(),
            Self::YearMonthInterval { months, start, end } => f
                .debug_struct("YearMonthInterval")
                .field("months", months)
                .field("start", start)
                .field("end", end)
                .finish(),
            Self::DayTimeInterval { micros, start, end } => f
                .debug_struct("DayTimeInterval")
                .field("micros", micros)
                .field("start", start)
                .field("end", end)
                .finish(),
            Self::CalendarInterval {
                months,
                days,
                micros,
            } => f
                .debug_struct("CalendarInterval")
                .field("months", months)
                .field("days", days)
                .field("micros", micros)
                .finish(),
        }
    }
}

/// The value of a [`Literal`] with floats as bits, so it derives `Eq`.
#[derive(PartialEq, Eq)]
enum Key<'a> {
    Null(Option<&'a DataType>),
    Boolean(bool),
    Byte(i8),
    Short(i16),
    Integer(i32),
    Long(i64),
    Float(u32),
    Double(u64),
    Decimal(i128, DecimalType),
    String(&'a str),
    Map(&'a [(String, String)]),
    Binary(&'a [u8]),
    Date(i32),
    Timestamp(i64),
    TimestampNtz(i64),
    YearMonth(i32, YearMonthField, YearMonthField),
    DayTime(i64, DayTimeField, DayTimeField),
    Calendar(i32, i32, i64),
}

impl Literal {
    fn key(&self) -> Key<'_> {
        match self {
            Self::Null(t) => Key::Null(t.as_ref()),
            Self::Boolean(v) => Key::Boolean(*v),
            Self::Byte(v) => Key::Byte(*v),
            Self::Short(v) => Key::Short(*v),
            Self::Integer(v) => Key::Integer(*v),
            Self::Long(v) => Key::Long(*v),
            Self::Float(v) => Key::Float(v.to_bits()),
            Self::Double(v) => Key::Double(v.to_bits()),
            Self::Decimal {
                unscaled,
                data_type,
            } => Key::Decimal(*unscaled, *data_type),
            Self::String(v) => Key::String(v),
            Self::Map(pairs) => Key::Map(pairs),
            Self::Binary(v) => Key::Binary(v),
            Self::Date(v) => Key::Date(*v),
            Self::Timestamp(v) => Key::Timestamp(*v),
            Self::TimestampNtz(v) => Key::TimestampNtz(*v),
            Self::YearMonthInterval { months, start, end } => Key::YearMonth(*months, *start, *end),
            Self::DayTimeInterval { micros, start, end } => Key::DayTime(*micros, *start, *end),
            Self::CalendarInterval {
                months,
                days,
                micros,
            } => Key::Calendar(*months, *days, *micros),
        }
    }
}

impl PartialEq for Literal {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Eq for Literal {}
