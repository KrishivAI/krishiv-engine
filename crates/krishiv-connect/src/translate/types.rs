// SPDX-License-Identifier: Apache-2.0
//! Connect `DataType` to Krishiv's.

use krishiv_common::error::{KrishivError, UnsupportedTimePrecision};
use krishiv_types::{
    ArrayType, Collation, DataType, DecimalType, MapType, Srid, StructField, StructType,
};

use super::errors::{assertion, not_set, unsupported_variant};
use super::literal;
use crate::proto::v4_2::{DataType as Proto, data_type::Kind};

/// Maps a Connect data type, following Spark's `DataTypeProtoConverter`:
/// a decimal with no precision is `decimal(10,0)`, one with a precision but
/// no scale has scale 0, a time with no precision is `time(6)`, an SRID of
/// -1 is `ANY`, `string` with the default collation is `STRING` and with another one a collated string,
/// and a struct field's metadata is Spark's `Metadata` JSON. An unset
/// element, key, value or field type is an unset kind.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a type Krishiv does not have yet
/// (`NULL`, `CHAR`, `VARCHAR`, intervals, ...), `COLLATION_INVALID_NAME` for an unknown collation,
/// `CONNECT_INVALID_PLAN.*` for an unset kind or an invalid decimal,
/// `INVALID_JSON_DATA_TYPE` for field metadata that is not a JSON object,
/// `FAILED_TO_PARSE_TOO_COMPLEX` for a type nested deeper than
/// `DataType::MAX_NESTING_DEPTH`, `UNSUPPORTED_TIME_PRECISION` and
/// `ST_INVALID_SRID_VALUE` for a time or spatial type Spark rejects.
pub(crate) fn data_type(proto: &Proto) -> Result<DataType, KrishivError> {
    data_type_at(proto, 0)
}

fn data_type_at(proto: &Proto, depth: usize) -> Result<DataType, KrishivError> {
    let child = |t: Option<&Proto>| {
        if depth >= DataType::MAX_NESTING_DEPTH {
            return Err(krishiv_common::error::FailedToParseTooComplex {}.into());
        }
        data_type_at(t.unwrap_or(&Proto::default()), depth + 1)
    };
    let Some(kind) = &proto.kind else {
        return Err(not_set("spark.connect.DataType", "KIND_NOT_SET"));
    };
    Ok(match kind {
        Kind::Boolean(_) => DataType::Boolean,
        Kind::Byte(_) => DataType::Byte,
        Kind::Short(_) => DataType::Short,
        Kind::Integer(_) => DataType::Integer,
        Kind::Long(_) => DataType::Long,
        Kind::Float(_) => DataType::Float,
        Kind::Double(_) => DataType::Double,
        Kind::Binary(_) => DataType::Binary,
        Kind::Date(_) => DataType::Date,
        Kind::Timestamp(_) => DataType::Timestamp,
        Kind::TimestampNtz(_) => DataType::TimestampNtz,
        Kind::Decimal(d) => DataType::Decimal(decimal_type(d.precision, d.scale)?),
        Kind::Time(t) => match t.precision {
            None => DataType::Time {
                precision: DataType::TIME_DEFAULT_PRECISION,
            },
            Some(p) => match u32::try_from(p) {
                Ok(p) => DataType::time(p)?,
                Err(_) => {
                    return Err(UnsupportedTimePrecision {
                        precision: p.to_string(),
                    }
                    .into());
                }
            },
        },
        Kind::YearMonthInterval(t) => {
            let (start, end) =
                literal::fields(t.start_field, t.end_field, literal::YEAR_MONTH_DEFAULT);
            DataType::year_month_interval(
                literal::year_month_field(start)?,
                literal::year_month_field(end)?,
            )?
        }
        Kind::DayTimeInterval(t) => {
            let (start, end) =
                literal::fields(t.start_field, t.end_field, literal::DAY_TIME_DEFAULT);
            DataType::day_time_interval(
                literal::day_time_field(start)?,
                literal::day_time_field(end)?,
            )?
        }
        Kind::CalendarInterval(_) => DataType::CalendarInterval,
        Kind::Variant(_) => DataType::Variant,
        Kind::Geometry(g) => DataType::geometry(Srid::from_i32(g.srid))?,
        Kind::Geography(g) => DataType::geography(Srid::from_i32(g.srid))?,
        Kind::String(s) => {
            // An empty name or `UTF8_BINARY` is the plain `STRING`: PySpark sends the default
            // string with the name `UTF8_BINARY`, so Connect cannot tell a written
            // `COLLATE UTF8_BINARY` from it (the SQL path keeps that as the display-only marker).
            // Any other name is a collated string (`COLLATION_INVALID_NAME` for a name Spark does
            // not know).
            if s.collation.is_empty() {
                DataType::String
            } else {
                match Collation::parse(&s.collation)? {
                    c if c.is_utf8_binary() => DataType::String,
                    c => DataType::collated_string(c),
                }
            }
        }
        Kind::Array(a) => ArrayType::new(child(a.element_type.as_deref())?, a.contains_null).into(),
        Kind::Map(m) => MapType::new(
            child(m.key_type.as_deref())?,
            child(m.value_type.as_deref())?,
            m.value_contains_null,
        )
        .into(),
        Kind::Struct(s) => {
            let fields = s
                .fields
                .iter()
                .map(|f| {
                    let field = StructField::new(&f.name, child(f.data_type.as_ref())?, f.nullable);
                    Ok(match &f.metadata {
                        Some(json) => field.with_metadata(krishiv_types::parse_metadata(json)?),
                        None => field,
                    })
                })
                .collect::<Result<Vec<_>, KrishivError>>()?;
            StructType::new(fields).into()
        }
        other => {
            return Err(unsupported_variant(
                "data type",
                "spark.connect.DataType",
                other.variant(),
            ));
        }
    })
}

/// A decimal type from the optional precision and scale of a Connect decimal.
pub(crate) fn decimal_type(
    precision: Option<i32>,
    scale: Option<i32>,
) -> Result<DecimalType, KrishivError> {
    let (precision, scale) = match (precision, scale) {
        (Some(p), Some(s)) => (p, s),
        (Some(p), None) => (p, 0),
        (None, _) => {
            let default = DecimalType::USER_DEFAULT;
            (i32::from(default.precision()), i32::from(default.scale()))
        }
    };
    match (u32::try_from(precision), u32::try_from(scale)) {
        (Ok(p), Ok(s)) => DecimalType::try_new(p, s),
        _ => Err(assertion(format!(
            "invalid decimal type with precision {precision} and scale {scale}"
        ))),
    }
}

#[cfg(test)]
mod tests;
