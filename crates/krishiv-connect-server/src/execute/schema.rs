// SPDX-License-Identifier: Apache-2.0
//! A result schema as the `schema` response carries it: Spark's
//! `DataTypeProtoConverter.toConnectProtoType` (v4.3.0) over the Spark type
//! of each result field.

use arrow_schema::Schema;
use krishiv_common::error::{InternalError, KrishivError};
use krishiv_connect::proto::v4_3 as pb;
use krishiv_connect::proto::v4_3::data_type::{self as dt, Kind};
use krishiv_types::arrow::SPARK_METADATA_KEY;
use krishiv_types::{DataType, DayTimeField, Metadata, StructField, YearMonthField};

/// The result schema as a Connect struct type. Each field's Spark type is
/// read from its Arrow field ([`DataType::from_arrow_field`]), its user
/// metadata from [`SPARK_METADATA_KEY`] (Spark sends non-empty metadata).
///
/// # Errors
/// The error of a field whose Spark type cannot be read.
pub fn proto_schema(schema: &Schema) -> Result<pb::DataType, KrishivError> {
    let fields = schema
        .fields()
        .iter()
        .map(|f| {
            let t = DataType::from_arrow_field(f)?;
            // The user metadata (a column comment, ...) rides in its own
            // key, apart from the engine's annotations of the field.
            let metadata = match f.metadata().get(SPARK_METADATA_KEY) {
                Some(json) => krishiv_types::parse_metadata(json)?,
                None => Metadata::empty(),
            };
            Ok(StructField::new(f.name(), t, f.is_nullable()).with_metadata(metadata))
        })
        .collect::<Result<Vec<_>, KrishivError>>()?;
    proto_type(&DataType::Struct(krishiv_types::StructType::new(fields)))
}

fn kind(kind: Kind) -> pb::DataType {
    pb::DataType { kind: Some(kind) }
}

/// Spark's ordinal of a year-month field (`YearMonthIntervalType.YEAR` 0).
fn year_month(f: YearMonthField) -> i32 {
    match f {
        YearMonthField::Year => 0,
        YearMonthField::Month => 1,
    }
}

/// Spark's ordinal of a day-time field (`DayTimeIntervalType.DAY` 0).
fn day_time(f: DayTimeField) -> i32 {
    match f {
        DayTimeField::Day => 0,
        DayTimeField::Hour => 1,
        DayTimeField::Minute => 2,
        DayTimeField::Second => 3,
    }
}

/// One Spark type as a Connect type.
///
/// # Errors
/// `INTERNAL_ERROR` for a type this mapping does not know (none today: a
/// type added to `krishiv-types` later must be added here).
pub fn proto_type(t: &DataType) -> Result<pb::DataType, KrishivError> {
    let v = 0;
    Ok(match t {
        DataType::Null => kind(Kind::Null(dt::Null {
            type_variation_reference: v,
        })),
        DataType::Boolean => kind(Kind::Boolean(dt::Boolean {
            type_variation_reference: v,
        })),
        DataType::Byte => kind(Kind::Byte(dt::Byte {
            type_variation_reference: v,
        })),
        DataType::Short => kind(Kind::Short(dt::Short {
            type_variation_reference: v,
        })),
        DataType::Integer => kind(Kind::Integer(dt::Integer {
            type_variation_reference: v,
        })),
        DataType::Long => kind(Kind::Long(dt::Long {
            type_variation_reference: v,
        })),
        DataType::Float => kind(Kind::Float(dt::Float {
            type_variation_reference: v,
        })),
        DataType::Double => kind(Kind::Double(dt::Double {
            type_variation_reference: v,
        })),
        // The default `STRING` sends no collation; an explicit one its name.
        DataType::String => kind(Kind::String(dt::String {
            type_variation_reference: v,
            collation: String::new(),
        })),
        DataType::CollatedString(c) => kind(Kind::String(dt::String {
            type_variation_reference: v,
            collation: c.name(),
        })),
        DataType::Binary => kind(Kind::Binary(dt::Binary {
            type_variation_reference: v,
        })),
        DataType::Decimal(d) => kind(Kind::Decimal(dt::Decimal {
            precision: Some(i32::from(d.precision())),
            scale: Some(i32::from(d.scale())),
            type_variation_reference: v,
        })),
        DataType::Date => kind(Kind::Date(dt::Date {
            type_variation_reference: v,
        })),
        DataType::Timestamp => kind(Kind::Timestamp(dt::Timestamp {
            type_variation_reference: v,
        })),
        DataType::TimestampNtz => kind(Kind::TimestampNtz(dt::TimestampNtz {
            type_variation_reference: v,
        })),
        DataType::Time { precision } => kind(Kind::Time(dt::Time {
            precision: Some(i32::from(*precision)),
            type_variation_reference: v,
        })),
        DataType::Geometry { srid } => kind(Kind::Geometry(dt::Geometry {
            srid: srid.to_i32(),
            type_variation_reference: v,
        })),
        DataType::Geography { srid } => kind(Kind::Geography(dt::Geography {
            srid: srid.to_i32(),
            type_variation_reference: v,
        })),
        DataType::YearMonthInterval { start, end } => {
            kind(Kind::YearMonthInterval(dt::YearMonthInterval {
                start_field: Some(year_month(*start)),
                end_field: Some(year_month(*end)),
                type_variation_reference: v,
            }))
        }
        DataType::DayTimeInterval { start, end } => {
            kind(Kind::DayTimeInterval(dt::DayTimeInterval {
                start_field: Some(day_time(*start)),
                end_field: Some(day_time(*end)),
                type_variation_reference: v,
            }))
        }
        DataType::CalendarInterval => kind(Kind::CalendarInterval(dt::CalendarInterval {
            type_variation_reference: v,
        })),
        DataType::Variant => kind(Kind::Variant(dt::Variant {
            type_variation_reference: v,
        })),
        DataType::Array(a) => kind(Kind::Array(Box::new(dt::Array {
            element_type: Some(Box::new(proto_type(a.element_type())?)),
            contains_null: a.contains_null(),
            type_variation_reference: v,
        }))),
        DataType::Map(m) => kind(Kind::Map(Box::new(dt::Map {
            key_type: Some(Box::new(proto_type(m.key_type())?)),
            value_type: Some(Box::new(proto_type(m.value_type())?)),
            value_contains_null: m.value_contains_null(),
            type_variation_reference: v,
        }))),
        DataType::Struct(s) => {
            let fields = s
                .fields()
                .iter()
                .map(|f| {
                    let metadata = f.metadata();
                    Ok(dt::StructField {
                        name: f.name().to_owned(),
                        data_type: Some(proto_type(f.data_type())?),
                        nullable: f.nullable(),
                        // Spark sets it only when the metadata is not empty.
                        metadata: (!metadata.is_empty()).then(|| {
                            serde_json::Value::Object(metadata.as_map().clone()).to_string()
                        }),
                    })
                })
                .collect::<Result<Vec<_>, KrishivError>>()?;
            kind(Kind::Struct(dt::Struct {
                fields,
                type_variation_reference: v,
            }))
        }
        // A type added to krishiv-types later (the enum is non-exhaustive).
        other => {
            return Err(InternalError {
                message: format!("no Spark Connect type for {}", other.sql()),
            }
            .into());
        }
    })
}
