// SPDX-License-Identifier: Apache-2.0
//! Spark data type ↔ Arrow mapping (ADR 0009).
//!
//! | Spark | Canonical Arrow type | Also accepted as storage |
//! |---|---|---|
//! | `boolean` | `Boolean` | |
//! | `tinyint` / `smallint` / `int` / `bigint` | `Int8` / `Int16` / `Int32` / `Int64` | |
//! | `float` / `double` | `Float32` / `Float64` | |
//! | `string` | `Utf8View` | `Utf8`, `LargeUtf8`; `BinaryView`, `Binary`, `LargeBinary` for bytes that are not UTF-8 |
//! | `binary` | `BinaryView` | `Binary`, `LargeBinary` |
//! | `decimal(p,s)` | `Decimal64(p, s)` for `p <= 18`, else `Decimal128(p, s)`; `Decimal64(1, 0)` for `decimal(0,0)` | `Decimal32`, `Decimal128` |
//! | `date` | `Date32` | |
//! | `timestamp` | `Timestamp(Microsecond, "UTC")` | |
//! | `timestamp_ntz` | `Timestamp(Microsecond, None)` | |
//! | `time(p)` | `Time64(Nanosecond)`, tagged when `p < 6` | |
//! | `interval year to month` | `Interval(YearMonth)`, tagged for other fields | |
//! | `interval day to second` | `Duration(Microsecond)`, tagged for other fields | |
//! | `interval` | `Struct<months: Int32, days: Int32, microseconds: Int64>`, children non-null and marked with [`CALENDAR_CHILD_METADATA_KEY`] (Spark's `CalendarInterval`; ADR 0009 update 2026-11-06) | `Interval(MonthDayNano)` (the session edges convert to and from it) |
//! | `geometry(s)` / `geography(s)` | `Struct<srid: Int32, wkb: BinaryView>`, both children non-null, always tagged | |
//! | `array<e>` | `List` of a field `element` (nullable when `containsNull`) | `LargeList`, any child name |
//! | `map<k,v>` | `Map` of a non-null struct `entries` of `key` (non-null) and `value` (nullable when `valueContainsNull`), unsorted | any child names |
//! | `struct<...>` | `Struct` of the fields | |
//! | `variant` | `Struct<metadata: BinaryView, value: BinaryView>`, both children non-null and marked with [`VARIANT_CHILD_METADATA_KEY`]; the field also carries the Arrow extension name `arrow.parquet.variant` | |
//!
//! Nested children follow Spark's `ArrowUtils` names. Each child is a field
//! built as [`DataType::to_arrow_field`] builds a column, so it carries its
//! own type tag when it needs one, and a struct field's non-empty metadata
//! rides in [`SPARK_METADATA_KEY`]. Nesting deeper than
//! [`DataType::MAX_NESTING_DEPTH`] is refused.
//!
//! The plan schema (Spark [`DataType`]) is the source of truth; Arrow types
//! and the [`SPARK_TYPE_METADATA_KEY`] field tag only carry it through storage
//! and edge round trips.
//!
//! - **Binary-safe strings.** A string column whose bytes are not valid UTF-8
//!   is stored as a binary Arrow type, and its field carries the Spark type
//!   under [`SPARK_TYPE_METADATA_KEY`] so it reads back as `string`.
//! - **Decimals.** The carrier follows the declared precision alone
//!   ([`DECIMAL64_MAX_PRECISION`], design P2-T160), so one Spark type has one
//!   carrier between operators, in spill files and in shuffle. Client edges
//!   (Connect/Flight, Python, UDFs) widen to `Decimal128`, what Spark sends.
//!   Arrow rejects precision 0, so `decimal(0,0)` (whose only value is 0) is
//!   stored with precision 1 and the field tag; use the field form for it.
//! - **Timestamps.** `timestamp` values are UTC instants. An input Arrow
//!   timestamp with any zone label is a `timestamp`; re-label it to `"UTC"` at
//!   ingest ([`normalize_ingest_type`]). Client egress re-labels to the session
//!   time zone and converts strings to `Utf8`, as Spark does.
//! - **Times.** A `time` value is nanoseconds since midnight, the long Spark
//!   hashes and sends over Spark Connect; the precision lives in the field
//!   tag only.
//! - **Intervals.** The fields of an ANSI interval live in the field tag only,
//!   as Spark's `ArrowUtils` loses them too. A calendar interval's nanoseconds
//!   are always a whole number of microseconds.
//! - **Spatial values.** Each row holds its SRID and its WKB bytes
//!   (little-endian, as [`crate::geo::rewrite`] writes it for
//!   [`crate::geo::ByteOrder::Ndr`]), so a `geometry(any)`
//!   column can mix SRIDs. The struct alone does not say geometry or
//!   geography, so these fields always carry the tag.
//! - **Variants.** Each row holds its own metadata dictionary and value bytes
//!   (Parquet Variant v1, unshredded: shredding exists only in files). A
//!   null struct slot is SQL `NULL`; a Variant null is the one-byte value
//!   `0x00`. The field carries the canonical Arrow extension
//!   name ([`VARIANT_EXTENSION_NAME`]) so Arrow consumers can recognise it;
//!   the marked children make the tag unnecessary.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{DataType as ArrowType, Field, Fields, IntervalUnit, TimeUnit};

use crate::error::{self, Result};
use crate::nested::{ArrayType, MapType, Metadata, StructField, StructType};
use crate::{DataType, DecimalType};

/// Arrow field-metadata key holding the Spark type (its `catalogString`)
/// when the field's Arrow type alone does not determine it.
pub const SPARK_TYPE_METADATA_KEY: &str = "krishiv.spark_type";

/// Arrow field-metadata key holding a struct field's Spark `Metadata` (its
/// JSON), when it has any.
pub const SPARK_METADATA_KEY: &str = "krishiv.spark_metadata";

/// Time zone of the canonical Arrow type for `timestamp` values, which are
/// UTC instants; the session time zone is applied only at the edges.
pub const TIMESTAMP_ARROW_TIME_ZONE: &str = "UTC";

/// Name of the SRID child (`Int32`) of a spatial value's struct.
pub const SPATIAL_SRID_FIELD: &str = "srid";

/// Name of the WKB child of a spatial value's struct.
pub const SPATIAL_WKB_FIELD: &str = "wkb";

/// Name of the metadata-dictionary child of a variant value's struct.
pub const VARIANT_METADATA_FIELD: &str = "metadata";

/// Name of the value child of a variant value's struct.
pub const VARIANT_VALUE_FIELD: &str = "value";

/// Arrow field-metadata key on the two children of a variant struct (value
/// `metadata` / `value`), so the struct's Arrow type alone says `variant`
/// and never reads as a user `struct<metadata: binary, value: binary>`.
pub const VARIANT_CHILD_METADATA_KEY: &str = "krishiv.variant";

/// The canonical Arrow extension name of a variant field.
pub const VARIANT_EXTENSION_NAME: &str = "arrow.parquet.variant";

/// Arrow's field-metadata key for an extension type name.
const ARROW_EXTENSION_NAME_KEY: &str = "ARROW:extension:name";

/// The largest precision on the `Decimal64` carrier; above it a decimal is on
/// `Decimal128`. Setting this to 0 puts every decimal back on `Decimal128`.
pub const DECIMAL64_MAX_PRECISION: u8 = 18;

/// The in-memory struct of a geometry or geography value.
fn spatial_struct() -> ArrowType {
    ArrowType::Struct(Fields::from(vec![
        Field::new(SPATIAL_SRID_FIELD, ArrowType::Int32, false),
        Field::new(SPATIAL_WKB_FIELD, ArrowType::BinaryView, false),
    ]))
}

/// The in-memory struct of a variant value: `metadata`, then `value`.
fn variant_struct() -> ArrowType {
    let child = |name: &str| {
        Field::new(name, ArrowType::BinaryView, false).with_metadata(HashMap::from([(
            VARIANT_CHILD_METADATA_KEY.to_owned(),
            name.to_owned(),
        )]))
    };
    ArrowType::Struct(Fields::from(vec![
        child(VARIANT_METADATA_FIELD),
        child(VARIANT_VALUE_FIELD),
    ]))
}

/// The child marker of a calendar interval's struct (its value is the child's name).
pub const CALENDAR_CHILD_METADATA_KEY: &str = "krishiv.calendar";
/// The months child of a calendar interval's struct.
pub const CALENDAR_MONTHS_FIELD: &str = "months";
/// The days child of a calendar interval's struct.
pub const CALENDAR_DAYS_FIELD: &str = "days";
/// The microseconds child of a calendar interval's struct.
pub const CALENDAR_MICROS_FIELD: &str = "microseconds";

/// The Arrow carrier of `CALENDAR INTERVAL`: Spark's `CalendarInterval` layout, so microseconds past the `i64`
/// nanoseconds of `Interval(MonthDayNano)` are representable. The children are marked, so a user struct with the
/// same names is never read as an interval.
#[must_use]
pub fn calendar_struct() -> ArrowType {
    let child = |name: &str, t: ArrowType| {
        Field::new(name, t, false).with_metadata(HashMap::from([(
            CALENDAR_CHILD_METADATA_KEY.to_owned(),
            name.to_owned(),
        )]))
    };
    ArrowType::Struct(Fields::from(vec![
        child(CALENDAR_MONTHS_FIELD, ArrowType::Int32),
        child(CALENDAR_DAYS_FIELD, ArrowType::Int32),
        child(CALENDAR_MICROS_FIELD, ArrowType::Int64),
    ]))
}

/// Whether `arrow` is the struct of a calendar interval.
#[must_use]
pub fn is_calendar_struct(arrow: &ArrowType) -> bool {
    // Only a three-field struct whose first child carries the mark can be one,
    // so other types never build the carrier to compare against.
    matches!(arrow, ArrowType::Struct(f) if f.len() == 3
        && f[0].metadata().contains_key(CALENDAR_CHILD_METADATA_KEY))
        && *arrow == calendar_struct()
}

/// Whether `arrow` is the struct of a variant value.
#[must_use]
pub fn is_variant_struct(arrow: &ArrowType) -> bool {
    *arrow == variant_struct()
}

/// The Arrow type an ingested column should carry: a zoned microsecond
/// timestamp is re-labelled to `"UTC"` (the values are already UTC instants);
/// every other type is returned unchanged.
pub fn normalize_ingest_type(arrow: &ArrowType) -> ArrowType {
    match arrow {
        ArrowType::Timestamp(TimeUnit::Microsecond, Some(_)) => DataType::Timestamp.to_arrow(),
        other => other.clone(),
    }
}

/// `arrow` with each decimal, nested ones included, mapped by `to`.
fn map_decimals(arrow: &ArrowType, to: fn(u8, i8) -> ArrowType) -> ArrowType {
    let field = |f: &Field| f.clone().with_data_type(map_decimals(f.data_type(), to));
    match arrow {
        ArrowType::Decimal32(p, s) | ArrowType::Decimal64(p, s) | ArrowType::Decimal128(p, s) => {
            to(*p, *s)
        }
        ArrowType::List(f) => ArrowType::List(Arc::new(field(f))),
        ArrowType::LargeList(f) => ArrowType::LargeList(Arc::new(field(f))),
        ArrowType::FixedSizeList(f, n) => ArrowType::FixedSizeList(Arc::new(field(f)), *n),
        ArrowType::Map(f, sorted) => ArrowType::Map(Arc::new(field(f)), *sorted),
        ArrowType::Struct(fields) => {
            ArrowType::Struct(fields.iter().map(|f| field(f)).collect::<Fields>())
        }
        other => other.clone(),
    }
}

/// `arrow` with each decimal on the canonical carrier of its precision and each calendar interval on its struct:
/// what an Arrow value of any carrier becomes when it enters a query.
#[must_use]
pub fn canonical_arrow_type(arrow: &ArrowType) -> ArrowType {
    let arrow = map_calendar(arrow, false);
    map_decimals(&arrow, |p, s| {
        if p <= DECIMAL64_MAX_PRECISION {
            ArrowType::Decimal64(p, s)
        } else {
            ArrowType::Decimal128(p, s)
        }
    })
}

/// `arrow` with each decimal on `Decimal128`: what a client reads.
#[must_use]
pub fn client_arrow_type(arrow: &ArrowType) -> ArrowType {
    map_decimals(arrow, ArrowType::Decimal128)
}

/// `arrow` as a caller outside the engine reads it: [`client_arrow_type`], and each calendar interval as Arrow's
/// `Interval(MonthDayNano)` (Spark's own Arrow form of it; ADR 0009 update 2026-11-06).
#[must_use]
pub fn edge_arrow_type(arrow: &ArrowType) -> ArrowType {
    map_calendar(&client_arrow_type(arrow), true)
}

/// `arrow` with each calendar interval on the struct carrier (`to_edge == false`) or on `Interval(MonthDayNano)`.
fn map_calendar(arrow: &ArrowType, to_edge: bool) -> ArrowType {
    let field = |f: &Field| {
        f.clone()
            .with_data_type(map_calendar(f.data_type(), to_edge))
    };
    match arrow {
        ArrowType::Struct(_) if to_edge && is_calendar_struct(arrow) => {
            ArrowType::Interval(IntervalUnit::MonthDayNano)
        }
        ArrowType::Interval(IntervalUnit::MonthDayNano) if !to_edge => calendar_struct(),
        ArrowType::List(f) => ArrowType::List(Arc::new(field(f))),
        ArrowType::LargeList(f) => ArrowType::LargeList(Arc::new(field(f))),
        ArrowType::FixedSizeList(f, n) => ArrowType::FixedSizeList(Arc::new(field(f)), *n),
        ArrowType::Map(f, sorted) => ArrowType::Map(Arc::new(field(f)), *sorted),
        ArrowType::Struct(fields) => {
            ArrowType::Struct(fields.iter().map(|f| field(f)).collect::<Fields>())
        }
        other => other.clone(),
    }
}

impl DataType {
    /// The canonical Arrow type for this Spark type.
    pub fn to_arrow(&self) -> ArrowType {
        match self {
            Self::Null => ArrowType::Null,
            Self::Boolean => ArrowType::Boolean,
            Self::Byte => ArrowType::Int8,
            Self::Short => ArrowType::Int16,
            Self::Integer => ArrowType::Int32,
            Self::Long => ArrowType::Int64,
            Self::Float => ArrowType::Float32,
            Self::Double => ArrowType::Float64,
            Self::String | Self::CollatedString(_) => ArrowType::Utf8View,
            Self::Binary => ArrowType::BinaryView,
            Self::Decimal(d) if d.precision().max(1) <= DECIMAL64_MAX_PRECISION => {
                ArrowType::Decimal64(d.precision().max(1), arrow_scale(*d))
            }
            Self::Decimal(d) => ArrowType::Decimal128(d.precision().max(1), arrow_scale(*d)),
            Self::Date => ArrowType::Date32,
            Self::Timestamp => ArrowType::Timestamp(
                TimeUnit::Microsecond,
                Some(Arc::from(TIMESTAMP_ARROW_TIME_ZONE)),
            ),
            Self::TimestampNtz => ArrowType::Timestamp(TimeUnit::Microsecond, None),
            Self::Time { .. } => ArrowType::Time64(TimeUnit::Nanosecond),
            Self::YearMonthInterval { .. } => ArrowType::Interval(IntervalUnit::YearMonth),
            Self::DayTimeInterval { .. } => ArrowType::Duration(TimeUnit::Microsecond),
            Self::CalendarInterval => calendar_struct(),
            Self::Geometry { .. } | Self::Geography { .. } => spatial_struct(),
            Self::Variant => variant_struct(),
            Self::Array(a) => ArrowType::List(Arc::new(
                a.element_type()
                    .to_arrow_field("element", a.contains_null()),
            )),
            Self::Map(m) => {
                let entries = ArrowType::Struct(
                    vec![
                        m.key_type().to_arrow_field("key", false),
                        m.value_type()
                            .to_arrow_field("value", m.value_contains_null()),
                    ]
                    .into(),
                );
                ArrowType::Map(Arc::new(Field::new("entries", entries, false)), false)
            }
            Self::Struct(s) => {
                ArrowType::Struct(s.fields().iter().map(struct_field_to_arrow).collect())
            }
        }
    }

    /// The Spark type of an Arrow type, inverting [`Self::to_arrow`] and
    /// accepting the storage alternatives listed in the module docs (a
    /// binary Arrow type maps to `binary`; see [`Self::from_arrow_field`]).
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for an Arrow type with no Spark type
    /// here (including non-microsecond timestamps and `Decimal256`), and the
    /// Spark decimal bound errors for an out-of-range decimal.
    pub fn from_arrow(arrow: &ArrowType) -> Result<Self> {
        Self::from_arrow_at(arrow, 0)
    }

    /// [`Self::from_arrow`] of a type nested `depth` levels deep.
    fn from_arrow_at(arrow: &ArrowType, depth: usize) -> Result<Self> {
        let nested = matches!(
            arrow,
            ArrowType::List(_)
                | ArrowType::LargeList(_)
                | ArrowType::Map(..)
                | ArrowType::Struct(_)
        );
        if nested && depth >= Self::MAX_NESTING_DEPTH {
            return Err(error::too_deep());
        }
        Ok(match arrow {
            ArrowType::Null => Self::Null,
            ArrowType::Boolean => Self::Boolean,
            ArrowType::Int8 => Self::Byte,
            ArrowType::Int16 => Self::Short,
            ArrowType::Int32 => Self::Integer,
            ArrowType::Int64 => Self::Long,
            ArrowType::Float32 => Self::Float,
            ArrowType::Float64 => Self::Double,
            ArrowType::Utf8View | ArrowType::Utf8 | ArrowType::LargeUtf8 => Self::String,
            ArrowType::BinaryView | ArrowType::Binary | ArrowType::LargeBinary => Self::Binary,
            ArrowType::Decimal32(p, s)
            | ArrowType::Decimal64(p, s)
            | ArrowType::Decimal128(p, s) => Self::Decimal(decimal_from_arrow(*p, *s)?),
            ArrowType::Date32 => Self::Date,
            ArrowType::Timestamp(TimeUnit::Microsecond, Some(_)) => Self::Timestamp,
            ArrowType::Timestamp(TimeUnit::Microsecond, None) => Self::TimestampNtz,
            ArrowType::Time64(TimeUnit::Nanosecond) => Self::Time {
                precision: Self::TIME_DEFAULT_PRECISION,
            },
            ArrowType::Interval(IntervalUnit::YearMonth) => Self::DEFAULT_YEAR_MONTH_INTERVAL,
            ArrowType::Duration(TimeUnit::Microsecond) => Self::DEFAULT_DAY_TIME_INTERVAL,
            ArrowType::Interval(IntervalUnit::MonthDayNano) => Self::CalendarInterval,
            ArrowType::List(f) | ArrowType::LargeList(f) => {
                ArrayType::new(Self::from_arrow_field_at(f, depth + 1)?, f.is_nullable()).into()
            }
            ArrowType::Map(entries, _) => match entries.data_type() {
                ArrowType::Struct(kv) if kv.len() == 2 => MapType::new(
                    Self::from_arrow_field_at(&kv[0], depth + 1)?,
                    Self::from_arrow_field_at(&kv[1], depth + 1)?,
                    kv[1].is_nullable(),
                )
                .into(),
                other => return Err(error::unsupported(format!("Arrow map entries {other}"))),
            },
            ArrowType::Struct(_) if is_variant_struct(arrow) => Self::Variant,
            ArrowType::Struct(_) if is_calendar_struct(arrow) => Self::CalendarInterval,
            ArrowType::Struct(fields) => StructType::new(
                fields
                    .iter()
                    .map(|f| struct_field_from_arrow(f, depth + 1))
                    .collect::<Result<Vec<_>>>()?,
            )
            .into(),
            other => return Err(error::unsupported(format!("Arrow type {other}"))),
        })
    }

    /// Whether `arrow` is a valid physical storage for this Spark type
    /// (the canonical type or one of the alternatives in the module docs).
    pub(crate) fn is_valid_storage(&self, arrow: &ArrowType) -> bool {
        match (self, arrow) {
            (Self::String, ArrowType::BinaryView | ArrowType::Binary | ArrowType::LargeBinary)
            | (
                Self::CollatedString(_),
                ArrowType::Utf8View | ArrowType::Utf8 | ArrowType::LargeUtf8,
            ) => true,
            (
                Self::Decimal(d),
                ArrowType::Decimal32(p, s)
                | ArrowType::Decimal64(p, s)
                | ArrowType::Decimal128(p, s),
            ) => (*p, *s) == (d.precision().max(1), arrow_scale(*d)),
            (Self::Timestamp, ArrowType::Timestamp(..))
            | (
                Self::Time { .. }
                | Self::Geometry { .. }
                | Self::Geography { .. }
                | Self::Variant
                | Self::YearMonthInterval { .. }
                | Self::DayTimeInterval { .. },
                _,
            ) => *arrow == self.to_arrow(),
            // The element, key, value or field types decide, so a tag on the
            // top-level field can say what plain nested fields don't.
            (Self::Array(a), ArrowType::List(e) | ArrowType::LargeList(e)) => {
                a.element_type().is_valid_storage(e.data_type())
            }
            (Self::Map(m), ArrowType::Map(entries, _)) => match entries.data_type() {
                ArrowType::Struct(kv) => match (kv.first(), kv.get(1)) {
                    (Some(k), Some(v)) => {
                        m.key_type().is_valid_storage(k.data_type())
                            && m.value_type().is_valid_storage(v.data_type())
                    }
                    _ => false,
                },
                _ => false,
            },
            (Self::Struct(s), ArrowType::Struct(fields)) => {
                s.fields().len() == fields.len()
                    && s.fields().iter().zip(fields.iter()).all(|(a, b)| {
                        a.name() == b.name() && a.data_type().is_valid_storage(b.data_type())
                    })
            }
            _ => Self::from_arrow(arrow).is_ok_and(|t| t == *self),
        }
    }

    /// A field of the canonical Arrow type, tagged with the Spark type when
    /// the Arrow type alone does not determine it (`decimal(0,0)`, `time(p)`
    /// below precision 6, intervals with other than the default fields,
    /// geometry, geography and variant).
    pub fn to_arrow_field(&self, name: impl Into<String>, nullable: bool) -> Field {
        self.tagged_field(name, self.to_arrow(), nullable)
    }

    /// A field whose Arrow type is `storage`, a valid storage for this type.
    /// The Spark type is recorded in [`SPARK_TYPE_METADATA_KEY`] when
    /// `storage` alone would read back as another type (a binary-safe string).
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` when `storage` cannot hold this type.
    pub fn to_arrow_field_with_storage(
        &self,
        name: impl Into<String>,
        storage: ArrowType,
        nullable: bool,
    ) -> Result<Field> {
        if !self.is_valid_storage(&storage) {
            return Err(error::unsupported(format!(
                "Arrow type {storage} as storage for {}",
                self.sql()
            )));
        }
        Ok(self.tagged_field(name, storage, nullable))
    }

    fn tagged_field(&self, name: impl Into<String>, storage: ArrowType, nullable: bool) -> Field {
        let needs_tag = Self::from_arrow(&storage).ok().as_ref() != Some(self);
        let field = Field::new(name, storage, nullable);
        let mut metadata = HashMap::new();
        if needs_tag {
            metadata.insert(SPARK_TYPE_METADATA_KEY.to_owned(), self.catalog_string());
        }
        if *self == Self::Variant {
            metadata.insert(
                ARROW_EXTENSION_NAME_KEY.to_owned(),
                VARIANT_EXTENSION_NAME.to_owned(),
            );
        }
        if metadata.is_empty() {
            field
        } else {
            field.with_metadata(metadata)
        }
    }

    /// The Spark type of a field: the type in [`SPARK_TYPE_METADATA_KEY`] if
    /// present (which must be valid for the field's Arrow type), otherwise
    /// [`Self::from_arrow`] of the field's type.
    ///
    /// # Errors
    /// As [`Self::from_arrow`]; a parse error for malformed metadata; and
    /// `KRISHIV_UNSUPPORTED_FEATURE` when the metadata type cannot be stored
    /// in the field's Arrow type.
    pub fn from_arrow_field(field: &Field) -> Result<Self> {
        Self::from_arrow_field_at(field, 0)
    }

    fn from_arrow_field_at(field: &Field, depth: usize) -> Result<Self> {
        match field.metadata().get(SPARK_TYPE_METADATA_KEY) {
            None if field
                .metadata()
                .get(ARROW_EXTENSION_NAME_KEY)
                .is_some_and(|n| n == VARIANT_EXTENSION_NAME)
                && Self::Variant.is_valid_storage(field.data_type()) =>
            {
                Ok(Self::Variant)
            }
            None => Self::from_arrow_at(field.data_type(), depth),
            Some(tag) => {
                let t = crate::parse_data_type(tag)?;
                if depth + t.nesting_depth() > Self::MAX_NESTING_DEPTH {
                    return Err(error::too_deep());
                }
                if t.is_valid_storage(field.data_type()) {
                    Ok(t)
                } else {
                    Err(error::unsupported(format!(
                        "Arrow type {} as storage for {}",
                        field.data_type(),
                        t.sql()
                    )))
                }
            }
        }
    }
}

/// A struct field as an Arrow field: its metadata in [`SPARK_METADATA_KEY`].
fn struct_field_to_arrow(f: &StructField) -> Field {
    let field = f.data_type().to_arrow_field(f.name(), f.nullable());
    if f.metadata().is_empty() {
        return field;
    }
    let mut metadata = field.metadata().clone();
    metadata.insert(
        SPARK_METADATA_KEY.to_owned(),
        crate::json::metadata_json(f.metadata()),
    );
    field.with_metadata(metadata)
}

fn struct_field_from_arrow(f: &Field, depth: usize) -> Result<StructField> {
    let metadata = match f.metadata().get(SPARK_METADATA_KEY) {
        Some(json) => crate::parse_metadata(json)?,
        None => Metadata::empty(),
    };
    Ok(StructField::new(
        f.name(),
        DataType::from_arrow_field_at(f, depth)?,
        f.is_nullable(),
    )
    .with_metadata(metadata))
}

/// Spark scales are 0..=38, so they always fit Arrow's `i8` scale.
fn arrow_scale(d: DecimalType) -> i8 {
    i8::try_from(d.scale()).unwrap_or(i8::MAX)
}

fn decimal_from_arrow(precision: u8, scale: i8) -> Result<DecimalType> {
    let scale = u32::try_from(scale).map_err(|_| error::negative_scale(i64::from(scale)))?;
    DecimalType::try_new(u32::from(precision), scale)
}
