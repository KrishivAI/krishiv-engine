// SPDX-License-Identifier: Apache-2.0
//! Spark data types and their Spark renderings (`simpleString`, `catalogString`,
//! `sql`, `typeName`).

use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use krishiv_common::error::KrishivError;

use crate::collation::Collation;
use crate::error::{self, Result};
use crate::nested::{ArrayType, MapType, StructType};

/// A Spark SQL data type.
///
/// Covers void (the type of a bare `NULL`), boolean, integral, floating-point, string, binary, decimal, date,
/// timestamp, time, interval, geometry and geography types and the nested array, map and struct types. Char/varchar
/// and variant types are not represented yet; parsing them yields `KRISHIV_UNSUPPORTED_FEATURE`.
///
/// Build `Time`, `Geometry` and `Geography` with [`Self::time`],
/// [`Self::geometry`] and [`Self::geography`], which check Spark's bounds.
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
    /// `STRING COLLATE x`: a string compared under a collation other than
    /// `UTF8_BINARY` (which is plain [`Self::String`]). The values are the
    /// same UTF-8 text; the collation is part of the type (ADR 0043).
    CollatedString(Collation),
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
    /// `TIME(p)`: nanoseconds since midnight, `0 ≤ p ≤ 6` fractional
    /// digits of seconds. The precision is part of the type only: as in
    /// Spark, a value may carry more digits than `p`.
    Time {
        /// Fractional digits of seconds, at most [`Self::TIME_MAX_PRECISION`].
        precision: u8,
    },
    /// `GEOMETRY(srid)`: a spatial value in a Cartesian reference system.
    Geometry {
        /// The spatial reference of every value, or [`Srid::Any`].
        srid: Srid,
    },
    /// `GEOGRAPHY(srid)`: a spatial value in a geographic reference system.
    Geography {
        /// The spatial reference of every value, or [`Srid::Any`].
        srid: Srid,
    },
    /// `INTERVAL YEAR`, `INTERVAL MONTH` or `INTERVAL YEAR TO MONTH`: a count
    /// of months. Build it with [`Self::year_month_interval`].
    YearMonthInterval {
        /// The leading field.
        start: YearMonthField,
        /// The trailing field; equal to `start` for a single-field type.
        end: YearMonthField,
    },
    /// `INTERVAL DAY`, `INTERVAL HOUR TO SECOND`, ...: a count of microseconds.
    /// Build it with [`Self::day_time_interval`].
    DayTimeInterval {
        /// The leading field.
        start: DayTimeField,
        /// The trailing field; equal to `start` for a single-field type.
        end: DayTimeField,
    },
    /// `INTERVAL`: Spark's `CalendarIntervalType`, months, days and
    /// microseconds, each counted on its own.
    CalendarInterval,
    /// `ARRAY<element>`.
    Array(Arc<ArrayType>),
    /// `MAP<key, value>`.
    Map(Arc<MapType>),
    /// `STRUCT<name: type, ...>`.
    Struct(StructType),
}

/// A field of a year-month interval type (Spark's `YearMonthIntervalType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum YearMonthField {
    /// `YEAR`.
    Year,
    /// `MONTH`.
    Month,
}

impl YearMonthField {
    /// The field's name in a `simpleString`.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Year => "year",
            Self::Month => "month",
        }
    }
}

/// A field of a day-time interval type (Spark's `DayTimeIntervalType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DayTimeField {
    /// `DAY`.
    Day,
    /// `HOUR`.
    Hour,
    /// `MINUTE`.
    Minute,
    /// `SECOND`.
    Second,
}

impl DayTimeField {
    /// The field's name in a `simpleString`.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Day => "day",
            Self::Hour => "hour",
            Self::Minute => "minute",
            Self::Second => "second",
        }
    }
}

/// `interval start` or `interval start to end`.
fn interval_string(start: &str, end: &str) -> String {
    if start == end {
        format!("interval {start}")
    } else {
        format!("interval {start} to {end}")
    }
}

/// The spatial reference of a geometry or geography type: one SRID for
/// every value, or `ANY` (each value carries its own).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Srid {
    /// Every value has this SRID.
    Fixed(i32),
    /// Values may have different SRIDs (Spark's `GEOMETRY(ANY)`).
    Any,
}

impl Srid {
    /// The integer Spark uses for [`Srid::Any`] where an SRID is an
    /// integer (Spark Connect types and Arrow field metadata).
    pub const MIXED: i32 = -1;

    /// The SRID as Spark Connect and Arrow metadata write it:
    /// [`Self::MIXED`] for `ANY`.
    pub const fn to_i32(self) -> i32 {
        match self {
            Self::Fixed(srid) => srid,
            Self::Any => Self::MIXED,
        }
    }

    /// The inverse of [`Self::to_i32`].
    pub const fn from_i32(srid: i32) -> Self {
        if srid == Self::MIXED {
            Self::Any
        } else {
            Self::Fixed(srid)
        }
    }

    /// The coordinate reference system Spark names a fixed SRID with in
    /// its JSON schemas and file metadata, if Spark supports the SRID:
    /// `SRID:0`, `EPSG:3857` and `OGC:CRS84` (4326). `geographic` limits
    /// the answer to geographic reference systems (4326 only).
    pub fn crs(srid: i32, geographic: bool) -> Option<&'static str> {
        // Spark 4.2's supported reference systems: (SRID, CRS, geographic).
        const SYSTEMS: [(i32, &str, bool); 3] = [
            (0, "SRID:0", false),
            (3857, "EPSG:3857", false),
            (4326, "OGC:CRS84", true),
        ];
        SYSTEMS
            .iter()
            .find(|(s, _, g)| *s == srid && (*g || !geographic))
            .map(|(_, crs, _)| *crs)
    }
}

impl fmt::Display for Srid {
    /// `ANY` renders as Spark's `simpleString` does, in lower case.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fixed(srid) => write!(f, "{srid}"),
            Self::Any => f.write_str("any"),
        }
    }
}

impl DataType {
    /// The deepest nesting of array, map and struct types accepted from
    /// outside (a type string, Arrow, JSON, Spark Connect, a plan). Spark has
    /// no limit; this one keeps every recursive walk over a type, and the
    /// JSON form of a type, within bounds.
    pub const MAX_NESTING_DEPTH: usize = 32;

    /// Spark's default `spark.sql.debug.maxToStringFields`: a struct's
    /// `simpleString` lists at most this many fields.
    pub const MAX_TO_STRING_FIELDS: usize = 25;

    /// Spark's `TimeType.MAX_PRECISION`.
    pub const TIME_MAX_PRECISION: u8 = 6;
    /// Spark's default precision of `TIME` (`TimeType.DEFAULT_PRECISION`).
    pub const TIME_DEFAULT_PRECISION: u8 = 6;

    /// Spark's `YearMonthIntervalType()`, `INTERVAL YEAR TO MONTH`.
    pub const DEFAULT_YEAR_MONTH_INTERVAL: Self = Self::YearMonthInterval {
        start: YearMonthField::Year,
        end: YearMonthField::Month,
    };
    /// Spark's `DayTimeIntervalType()`, `INTERVAL DAY TO SECOND`.
    pub const DEFAULT_DAY_TIME_INTERVAL: Self = Self::DayTimeInterval {
        start: DayTimeField::Day,
        end: DayTimeField::Second,
    };

    /// `INTERVAL start [TO end]` of the year-month family.
    ///
    /// # Errors
    /// `INVALID_INTERVAL_FORMAT.UNSUPPORTED_FROM_TO_EXPRESSION` when `end` comes
    /// before `start`.
    pub fn year_month_interval(start: YearMonthField, end: YearMonthField) -> Result<Self> {
        if end < start {
            return Err(error::interval_from_to(start.name(), end.name()));
        }
        Ok(Self::YearMonthInterval { start, end })
    }

    /// `INTERVAL start [TO end]` of the day-time family.
    ///
    /// # Errors
    /// `INVALID_INTERVAL_FORMAT.UNSUPPORTED_FROM_TO_EXPRESSION` when `end` comes
    /// before `start`.
    pub fn day_time_interval(start: DayTimeField, end: DayTimeField) -> Result<Self> {
        if end < start {
            return Err(error::interval_from_to(start.name(), end.name()));
        }
        Ok(Self::DayTimeInterval { start, end })
    }

    /// Whether a collated string occurs in this type, at any depth.
    #[must_use]
    pub fn contains_collation(&self) -> bool {
        match self {
            Self::CollatedString(_) => true,
            Self::Array(a) => a.element_type().contains_collation(),
            Self::Map(m) => {
                m.key_type().contains_collation() || m.value_type().contains_collation()
            }
            Self::Struct(s) => s
                .fields()
                .iter()
                .any(|f| f.data_type().contains_collation()),
            _ => false,
        }
    }

    /// `self`, or `KRISHIV_UNSUPPORTED_FEATURE` if it holds a collated
    /// string. The analyzer does not take collated types yet (ADR 0043,
    /// unit 2a): a user-facing entry point that cannot honour a collation
    /// refuses it here, so no operator compares such a column as bytes.
    ///
    /// # Errors
    /// As above.
    pub fn refuse_collations(self) -> Result<Self> {
        if self.contains_collation() {
            return Err(error::unsupported(format!(
                "collated strings (\"{}\")",
                self.sql()
            )));
        }
        Ok(self)
    }

    /// `STRING COLLATE collation`: [`Self::String`] for `UTF8_BINARY`, else
    /// [`Self::CollatedString`].
    #[must_use]
    pub fn collated_string(collation: Collation) -> Self {
        if collation.is_utf8_binary() {
            Self::String
        } else {
            Self::CollatedString(collation)
        }
    }

    /// `TIME(precision)`.
    ///
    /// # Errors
    /// `UNSUPPORTED_TIME_PRECISION` when `precision > 6`.
    pub fn time(precision: u32) -> Result<Self> {
        match u8::try_from(precision) {
            Ok(p) if p <= Self::TIME_MAX_PRECISION => Ok(Self::Time { precision: p }),
            _ => Err(error::time_precision(&precision.to_string())),
        }
    }

    /// `GEOMETRY(srid)`.
    ///
    /// # Errors
    /// `ST_INVALID_SRID_VALUE` for an SRID Spark does not support (see
    /// [`Srid::crs`]).
    pub fn geometry(srid: Srid) -> Result<Self> {
        check_srid(srid, false).map(|srid| Self::Geometry { srid })
    }

    /// `GEOGRAPHY(srid)`.
    ///
    /// # Errors
    /// `ST_INVALID_SRID_VALUE` for any SRID other than 4326.
    pub fn geography(srid: Srid) -> Result<Self> {
        check_srid(srid, true).map(|srid| Self::Geography { srid })
    }

    /// Spark's `DataType.simpleString`, e.g. `int`, `decimal(10,2)`,
    /// `struct<a:int>` (at most [`Self::MAX_TO_STRING_FIELDS`] fields of a
    /// struct are listed).
    pub fn simple_string(&self) -> String {
        self.render(Render::Simple)
    }

    /// Spark's `DataType.catalogString`: [`Self::simple_string`] with every
    /// struct field listed.
    pub fn catalog_string(&self) -> String {
        self.render(Render::Catalog)
    }

    /// Spark's `DataType.sql`, e.g. `INT`, `DECIMAL(10,2)`,
    /// `STRUCT<a: INT NOT NULL COMMENT 'c'>`.
    pub fn sql(&self) -> String {
        self.render(Render::Sql)
    }

    /// Spark's `DataType.typeName` (the name used in Spark's JSON schemas),
    /// e.g. `integer`, `long`, `decimal(10,2)`, `array`.
    pub fn type_name(&self) -> String {
        match self {
            Self::Byte => "byte".to_owned(),
            Self::Short => "short".to_owned(),
            Self::Integer => "integer".to_owned(),
            Self::Long => "long".to_owned(),
            Self::Decimal(_)
            | Self::CollatedString(_)
            | Self::Time { .. }
            | Self::Geometry { .. }
            | Self::Geography { .. }
            | Self::YearMonthInterval { .. }
            | Self::DayTimeInterval { .. } => self.atomic_simple_string(),
            other => other.keyword().to_owned(),
        }
    }

    /// The `simpleString` of an atomic type.
    fn atomic_simple_string(&self) -> String {
        match self {
            Self::Decimal(d) => format!("decimal({},{})", d.precision, d.scale),
            Self::CollatedString(c) => format!("string collate {c}"),
            Self::Time { precision } => format!("time({precision})"),
            Self::Geometry { srid } => format!("geometry({srid})"),
            Self::Geography { srid } => format!("geography({srid})"),
            Self::YearMonthInterval { start, end } => interval_string(start.name(), end.name()),
            Self::DayTimeInterval { start, end } => interval_string(start.name(), end.name()),
            other => other.keyword().to_owned(),
        }
    }

    fn render(&self, how: Render) -> String {
        let mut out = String::new();
        self.render_into(how, &mut out);
        out
    }

    fn render_into(&self, how: Render, out: &mut String) {
        let sql = how == Render::Sql;
        let (open, sep) = if sql { ("<", ", ") } else { ("<", ",") };
        match self {
            Self::Array(a) => {
                out.push_str(if sql { "ARRAY" } else { "array" });
                out.push_str(open);
                a.element_type().render_into(how, out);
                out.push('>');
            }
            Self::Map(m) => {
                out.push_str(if sql { "MAP" } else { "map" });
                out.push_str(open);
                m.key_type().render_into(how, out);
                out.push_str(sep);
                m.value_type().render_into(how, out);
                out.push('>');
            }
            Self::Struct(s) => {
                out.push_str(if sql { "STRUCT" } else { "struct" });
                out.push_str(open);
                let fields = s.fields();
                let shown = match how {
                    Render::Simple => fields.len().min(Self::MAX_TO_STRING_FIELDS),
                    _ => fields.len(),
                };
                for (i, f) in fields.iter().take(shown).enumerate() {
                    if i > 0 {
                        out.push_str(sep);
                    }
                    if sql {
                        out.push_str(&quote_if_needed(f.name()));
                        out.push_str(": ");
                    } else {
                        out.push_str(f.name());
                        out.push(':');
                    }
                    f.data_type().render_into(how, out);
                    if sql {
                        if !f.nullable() {
                            out.push_str(" NOT NULL");
                        }
                        if let Some(c) = f.metadata().comment() {
                            out.push_str(" COMMENT '");
                            out.push_str(&c.replace('\'', "\\'"));
                            out.push('\'');
                        }
                    }
                }
                // Spark's `truncatedString`: the listed fields, a separator,
                // then how many are left out.
                if shown < fields.len() {
                    out.push_str(sep);
                    out.push_str(&format!("... {} more fields", fields.len() - shown));
                }
                out.push('>');
            }
            Self::CollatedString(c) => {
                out.push_str(if sql {
                    "STRING COLLATE "
                } else {
                    "string collate "
                });
                out.push_str(&c.to_string());
            }
            atomic if sql => out.push_str(&atomic.atomic_simple_string().to_ascii_uppercase()),
            atomic => out.push_str(&atomic.atomic_simple_string()),
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
            Self::String | Self::CollatedString(_) => "string",
            Self::Binary => "binary",
            Self::Decimal(_) => "decimal",
            Self::Date => "date",
            Self::Timestamp => "timestamp",
            Self::TimestampNtz => "timestamp_ntz",
            Self::Time { .. } => "time",
            Self::Geometry { .. } => "geometry",
            Self::Geography { .. } => "geography",
            Self::YearMonthInterval { .. }
            | Self::DayTimeInterval { .. }
            | Self::CalendarInterval => "interval",
            Self::Array(_) => "array",
            Self::Map(_) => "map",
            Self::Struct(_) => "struct",
        }
    }
}

/// Which of Spark's renderings of a type.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Render {
    Simple,
    Catalog,
    Sql,
}

/// Spark's `QuotingUtils.quoteIfNeeded`: a name that is not a plain
/// identifier (`[a-zA-Z_][a-zA-Z0-9_]*`) is backquoted, with backquotes
/// doubled.
pub(crate) fn quote_if_needed(name: &str) -> String {
    let mut chars = name.chars();
    let plain = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if plain {
        name.to_owned()
    } else {
        format!("`{}`", name.replace('`', "``"))
    }
}

fn check_srid(srid: Srid, geographic: bool) -> Result<Srid> {
    match srid {
        Srid::Fixed(s) if Srid::crs(s, geographic).is_none() => {
            Err(error::invalid_srid(&s.to_string()))
        }
        _ => Ok(srid),
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
