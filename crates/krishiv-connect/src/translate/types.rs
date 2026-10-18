// SPDX-License-Identifier: Apache-2.0
//! Connect `DataType` to Krishiv's.

use krishiv_common::error::KrishivError;
use krishiv_types::{DataType, DecimalType};

use super::errors::{assertion, not_set, unsupported_field, unsupported_variant};
use crate::proto::v4_2::{DataType as Proto, data_type::Kind};

/// The collation Spark 4 reports for the default `STRING`.
const DEFAULT_COLLATION: &str = "UTF8_BINARY";

/// Maps a Connect data type, following Spark's `DataTypeProtoConverter`:
/// a decimal with no precision is `decimal(10,0)`, one with a precision but
/// no scale has scale 0, and `string` with the default collation is `STRING`.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a type Krishiv does not have yet
/// (`NULL`, `CHAR`, `VARCHAR`, intervals, nested types, ...) or a
/// non-default collation, `CONNECT_INVALID_PLAN.*` for an unset kind or an
/// invalid decimal.
pub(crate) fn data_type(proto: &Proto) -> Result<DataType, KrishivError> {
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
