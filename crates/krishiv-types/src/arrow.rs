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
//! | `decimal(p,s)` | `Decimal128(p, s)`; `Decimal128(1, 0)` for `decimal(0,0)` | |
//! | `date` | `Date32` | |
//! | `timestamp` | `Timestamp(Microsecond, "UTC")` | |
//! | `timestamp_ntz` | `Timestamp(Microsecond, None)` | |
//!
//! The plan schema (Spark [`DataType`]) is the source of truth; Arrow types
//! and the [`SPARK_TYPE_METADATA_KEY`] field tag only carry it through storage
//! and edge round trips.
//!
//! - **Binary-safe strings.** A string column whose bytes are not valid UTF-8
//!   is stored as a binary Arrow type, and its field carries the Spark type
//!   under [`SPARK_TYPE_METADATA_KEY`] so it reads back as `string`.
//! - **Decimals.** Only `Decimal128` crosses an edge (Connect/Flight, IPC,
//!   spill, shuffle). The i64 representation for precision ≤ 18
//!   ([`crate::DecimalStorage::I64`]) is internal to kernels; Arrow `Decimal64`
//!   is not used until that fast path is benchmarked. Arrow rejects
//!   precision 0, so `decimal(0,0)` (whose only value is 0) is stored as
//!   `Decimal128(1, 0)` with the field tag; use the field form for it.
//! - **Timestamps.** `timestamp` values are UTC instants. An input Arrow
//!   timestamp with any zone label is a `timestamp`; re-label it to `"UTC"` at
//!   ingest ([`normalize_ingest_type`]). Client egress re-labels to the session
//!   time zone and converts strings to `Utf8`, as Spark does.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::{DataType as ArrowType, Field, TimeUnit};

use crate::error::{self, Result};
use crate::{DataType, DecimalType};

/// Arrow field-metadata key holding the Spark type (its `catalogString`)
/// when the field's Arrow type alone does not determine it.
pub const SPARK_TYPE_METADATA_KEY: &str = "krishiv.spark_type";

/// Time zone of the canonical Arrow type for `timestamp` values, which are
/// UTC instants; the session time zone is applied only at the edges.
pub const TIMESTAMP_ARROW_TIME_ZONE: &str = "UTC";

/// The Arrow type an ingested column should carry: a zoned microsecond
/// timestamp is re-labelled to `"UTC"` (the values are already UTC instants);
/// every other type is returned unchanged.
pub fn normalize_ingest_type(arrow: &ArrowType) -> ArrowType {
    match arrow {
        ArrowType::Timestamp(TimeUnit::Microsecond, Some(_)) => DataType::Timestamp.to_arrow(),
        other => other.clone(),
    }
}

impl DataType {
    /// The canonical Arrow type for this Spark type.
    pub fn to_arrow(&self) -> ArrowType {
        match self {
            Self::Boolean => ArrowType::Boolean,
            Self::Byte => ArrowType::Int8,
            Self::Short => ArrowType::Int16,
            Self::Integer => ArrowType::Int32,
            Self::Long => ArrowType::Int64,
            Self::Float => ArrowType::Float32,
            Self::Double => ArrowType::Float64,
            Self::String => ArrowType::Utf8View,
            Self::Binary => ArrowType::BinaryView,
            Self::Decimal(d) => ArrowType::Decimal128(d.precision().max(1), arrow_scale(*d)),
            Self::Date => ArrowType::Date32,
            Self::Timestamp => ArrowType::Timestamp(
                TimeUnit::Microsecond,
                Some(Arc::from(TIMESTAMP_ARROW_TIME_ZONE)),
            ),
            Self::TimestampNtz => ArrowType::Timestamp(TimeUnit::Microsecond, None),
        }
    }

    /// The Spark type of an Arrow type, inverting [`Self::to_arrow`] and
    /// accepting the storage alternatives listed in the module docs (a
    /// binary Arrow type maps to `binary`; see [`Self::from_arrow_field`]).
    ///
    /// # Errors
    /// `KRISHIV_UNSUPPORTED_FEATURE` for an Arrow type with no Spark type
    /// here (including non-microsecond timestamps and `Decimal64`), and the
    /// Spark decimal bound errors for an out-of-range decimal.
    pub fn from_arrow(arrow: &ArrowType) -> Result<Self> {
        Ok(match arrow {
            ArrowType::Boolean => Self::Boolean,
            ArrowType::Int8 => Self::Byte,
            ArrowType::Int16 => Self::Short,
            ArrowType::Int32 => Self::Integer,
            ArrowType::Int64 => Self::Long,
            ArrowType::Float32 => Self::Float,
            ArrowType::Float64 => Self::Double,
            ArrowType::Utf8View | ArrowType::Utf8 | ArrowType::LargeUtf8 => Self::String,
            ArrowType::BinaryView | ArrowType::Binary | ArrowType::LargeBinary => Self::Binary,
            ArrowType::Decimal128(p, s) => Self::Decimal(decimal_from_arrow(*p, *s)?),
            ArrowType::Date32 => Self::Date,
            ArrowType::Timestamp(TimeUnit::Microsecond, Some(_)) => Self::Timestamp,
            ArrowType::Timestamp(TimeUnit::Microsecond, None) => Self::TimestampNtz,
            other => return Err(error::unsupported(format!("Arrow type {other}"))),
        })
    }

    /// Whether `arrow` is a valid physical storage for this Spark type
    /// (the canonical type or one of the alternatives in the module docs).
    pub(crate) fn is_valid_storage(&self, arrow: &ArrowType) -> bool {
        match (self, arrow) {
            (Self::String, ArrowType::BinaryView | ArrowType::Binary | ArrowType::LargeBinary) => {
                true
            }
            (Self::Decimal(_), ArrowType::Decimal128(..))
            | (Self::Timestamp, ArrowType::Timestamp(..)) => *arrow == self.to_arrow(),
            _ => Self::from_arrow(arrow).is_ok_and(|t| t == *self),
        }
    }

    /// A field of the canonical Arrow type, tagged with the Spark type when
    /// the Arrow type alone does not determine it (`decimal(0,0)`).
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
        if needs_tag {
            field.with_metadata(HashMap::from([(
                SPARK_TYPE_METADATA_KEY.to_owned(),
                self.catalog_string(),
            )]))
        } else {
            field
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
        match field.metadata().get(SPARK_TYPE_METADATA_KEY) {
            None => Self::from_arrow(field.data_type()),
            Some(tag) => {
                let t = crate::parse_data_type(tag)?;
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

/// Spark scales are 0..=38, so they always fit Arrow's `i8` scale.
fn arrow_scale(d: DecimalType) -> i8 {
    i8::try_from(d.scale()).unwrap_or(i8::MAX)
}

fn decimal_from_arrow(precision: u8, scale: i8) -> Result<DecimalType> {
    let scale = u32::try_from(scale).map_err(|_| error::negative_scale(i64::from(scale)))?;
    DecimalType::try_new(u32::from(precision), scale)
}
