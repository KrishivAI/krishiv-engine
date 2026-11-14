// SPDX-License-Identifier: Apache-2.0
//! Literal values.

use std::sync::Arc;

use krishiv_common::error::{FailedToParseTooComplex, InternalError, KrishivError};
use krishiv_types::{ArrayType, DataType, DecimalType, MapType, StructType};
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
    /// An `ARRAY` value (Spark Connect `Literal.Array`). Build it with
    /// [`Literal::array`]; a null element is `Null(Some(element type))`.
    Array {
        /// The array type, element type and `containsNull`.
        data_type: Arc<ArrayType>,
        /// The elements, each of the element type.
        elements: Vec<Literal>,
    },
    /// A `MAP` value of any key and value type (not the options-only
    /// [`Literal::Map`]). Build it with [`Literal::key_value_map`]; keys are
    /// never null.
    KeyValueMap {
        /// The map type.
        data_type: Arc<MapType>,
        /// The entries, in order, as `(key, value)`.
        entries: Vec<(Literal, Literal)>,
    },
    /// A `STRUCT` value. Build it with [`Literal::struct_value`].
    Struct {
        /// The struct type.
        data_type: Arc<StructType>,
        /// One literal per field, in field order.
        fields: Vec<Literal>,
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
            Self::Array {
                data_type,
                elements,
            } => f
                .debug_struct("Array")
                .field("data_type", data_type)
                .field("elements", elements)
                .finish(),
            Self::KeyValueMap { data_type, entries } => f
                .debug_struct("KeyValueMap")
                .field("data_type", data_type)
                .field("entries", entries)
                .finish(),
            Self::Struct { data_type, fields } => f
                .debug_struct("Struct")
                .field("data_type", data_type)
                .field("fields", fields)
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
    Array(&'a ArrayType, &'a [Literal]),
    Pairs(&'a MapType, &'a [(Literal, Literal)]),
    Struct(&'a StructType, &'a [Literal]),
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
            Self::Array {
                data_type,
                elements,
            } => Key::Array(data_type, elements),
            Self::KeyValueMap { data_type, entries } => Key::Pairs(data_type, entries),
            Self::Struct { data_type, fields } => Key::Struct(data_type, fields),
        }
    }
}

impl PartialEq for Literal {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Eq for Literal {}

fn invalid(message: String) -> KrishivError {
    InternalError { message }.into()
}

fn check_depth(t: &DataType) -> Result<(), KrishivError> {
    if t.nesting_depth() > DataType::MAX_NESTING_DEPTH {
        return Err(FailedToParseTooComplex {}.into());
    }
    Ok(())
}

/// Checks that `value` is a literal of `expected` (a typed null included),
/// and that a null is allowed here.
fn check_member(
    what: &str,
    value: &Literal,
    expected: &DataType,
    nullable: bool,
) -> Result<(), KrishivError> {
    if matches!(value, Literal::Null(_)) && !nullable {
        return Err(invalid(format!(
            "a null {what} in a nested literal whose type forbids nulls"
        )));
    }
    match value.data_type() {
        Some(t) if t == *expected => Ok(()),
        Some(t) => Err(invalid(format!(
            "a nested literal {what} of type {} where the type says {}",
            t.simple_string(),
            expected.simple_string()
        ))),
        None => Err(invalid(format!(
            "an untyped null {what} in a nested literal; nulls inside carry their type"
        ))),
    }
}

impl Literal {
    /// The type of this literal; `None` for an untyped `NULL`.
    ///
    /// Golden-pending (P3-T012a): a typed null of `VOID` is also `None`, so
    /// an `ARRAY<VOID>`/`MAP<_, VOID>` literal holding a null member is
    /// refused as an untyped null; whether Spark's `typedLit` reaches that
    /// is settled with the literal goldens in part 2b.
    #[must_use]
    pub fn data_type(&self) -> Option<DataType> {
        Some(match self {
            Self::Null(t) => return t.clone().filter(|t| *t != DataType::Null),
            Self::Boolean(_) => DataType::Boolean,
            Self::Byte(_) => DataType::Byte,
            Self::Short(_) => DataType::Short,
            Self::Integer(_) => DataType::Integer,
            Self::Long(_) => DataType::Long,
            Self::Float(_) => DataType::Float,
            Self::Double(_) => DataType::Double,
            Self::Decimal { data_type, .. } => DataType::Decimal(*data_type),
            Self::String(_) => DataType::String,
            Self::Map(_) => {
                DataType::Map(MapType::new(DataType::String, DataType::String, false).into())
            }
            Self::Binary(_) => DataType::Binary,
            Self::Date(_) => DataType::Date,
            Self::Timestamp(_) => DataType::Timestamp,
            Self::TimestampNtz(_) => DataType::TimestampNtz,
            Self::YearMonthInterval { start, end, .. } => DataType::YearMonthInterval {
                start: *start,
                end: *end,
            },
            Self::DayTimeInterval { start, end, .. } => DataType::DayTimeInterval {
                start: *start,
                end: *end,
            },
            Self::CalendarInterval { .. } => DataType::CalendarInterval,
            Self::Array { data_type, .. } => DataType::Array(Arc::clone(data_type)),
            Self::KeyValueMap { data_type, .. } => DataType::Map(Arc::clone(data_type)),
            Self::Struct { data_type, .. } => DataType::Struct((**data_type).clone()),
        })
    }

    /// An array literal: every element is a literal of the element type
    /// (a null is `Null(Some(element type))` and needs `contains_null`).
    ///
    /// # Errors
    /// `FAILED_TO_PARSE_TOO_COMPLEX` when the type nests deeper than
    /// [`DataType::MAX_NESTING_DEPTH`]; `INTERNAL_ERROR` when an element
    /// has another type or is a forbidden or untyped null.
    pub fn array(data_type: ArrayType, elements: Vec<Literal>) -> Result<Self, KrishivError> {
        check_depth(&DataType::Array(Arc::new(data_type.clone())))?;
        for e in &elements {
            check_member(
                "element",
                e,
                data_type.element_type(),
                data_type.contains_null(),
            )?;
        }
        Ok(Self::Array {
            data_type: Arc::new(data_type),
            elements,
        })
    }

    /// A map literal of `(key, value)` entries of the map's key and value
    /// types; keys are never null.
    ///
    /// # Errors
    /// As [`Literal::array`].
    pub fn key_value_map(
        data_type: MapType,
        entries: Vec<(Literal, Literal)>,
    ) -> Result<Self, KrishivError> {
        check_depth(&DataType::Map(Arc::new(data_type.clone())))?;
        for (k, v) in &entries {
            check_member("key", k, data_type.key_type(), false)?;
            check_member(
                "value",
                v,
                data_type.value_type(),
                data_type.value_contains_null(),
            )?;
        }
        Ok(Self::KeyValueMap {
            data_type: Arc::new(data_type),
            entries,
        })
    }

    /// A struct literal: one literal per field, of the field's type (a null
    /// needs a nullable field).
    ///
    /// # Errors
    /// As [`Literal::array`]; also when the count of values differs from the
    /// count of fields.
    pub fn struct_value(data_type: StructType, fields: Vec<Literal>) -> Result<Self, KrishivError> {
        check_depth(&DataType::Struct(data_type.clone()))?;
        if data_type.fields().len() != fields.len() {
            return Err(invalid(format!(
                "a struct literal of {} values for a type of {} fields",
                fields.len(),
                data_type.fields().len()
            )));
        }
        for (f, v) in data_type.fields().iter().zip(&fields) {
            check_member("field", v, f.data_type(), f.nullable())?;
        }
        Ok(Self::Struct {
            data_type: Arc::new(data_type),
            fields,
        })
    }
}
