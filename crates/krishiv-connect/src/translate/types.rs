// SPDX-License-Identifier: Apache-2.0
//! Connect `DataType` to Krishiv's.

use krishiv_common::error::KrishivError;
use krishiv_types::{ArrayType, DataType, DecimalType, MapType, StructField, StructType};

use super::errors::{assertion, not_set, unsupported_field, unsupported_variant};
use crate::proto::v4_2::{DataType as Proto, data_type::Kind};

/// The collation Spark 4 reports for the default `STRING`.
const DEFAULT_COLLATION: &str = "UTF8_BINARY";

/// Maps a Connect data type, following Spark's `DataTypeProtoConverter`:
/// a decimal with no precision is `decimal(10,0)`, one with a precision but
/// no scale has scale 0, `string` with the default collation is `STRING`,
/// and a struct field's metadata is Spark's `Metadata` JSON. An unset
/// element, key, value or field type is an unset kind.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a type Krishiv does not have yet
/// (`NULL`, `CHAR`, `VARCHAR`, intervals, ...) or a non-default collation,
/// `CONNECT_INVALID_PLAN.*` for an unset kind or an invalid decimal,
/// `INVALID_JSON_DATA_TYPE` for field metadata that is not a JSON object and
/// `FAILED_TO_PARSE_TOO_COMPLEX` for a type nested deeper than
/// `DataType::MAX_NESTING_DEPTH`.
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
        Kind::String(s) => {
            if !s.collation.is_empty() && !s.collation.eq_ignore_ascii_case(DEFAULT_COLLATION) {
                return Err(unsupported_field(
                    "DataType.String",
                    "collation",
                    "string types with a collation",
                ));
            }
            DataType::String
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
