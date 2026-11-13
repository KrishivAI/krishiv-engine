// SPDX-License-Identifier: Apache-2.0
//! Iceberg types as Spark sees them (`iceberg_type_mapping`): the mapping of
//! Iceberg's `TypeToSparkType` at apache-iceberg 1.12.0, Spark 4.1 module
//! (`research/p3/t022a-iceberg-spark-types.md`). Field nullability is the field's
//! `optional`, a list's `containsNull` its element's, a map's `valueContainsNull`
//! its value's; a doc becomes the `comment` metadata; field ids are not carried.

use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use krishiv_common::error::{KrishivError, KrishivUnsupportedFeature};
use krishiv_types::{ArrayType, DataType, DecimalType, MapType, Metadata, StructField, StructType};

fn unsupported(message: String) -> KrishivError {
    KrishivUnsupportedFeature { feature: message }.into()
}

/// The Spark type of `t`.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for the types Spark's Iceberg integration refuses,
/// with its message: `time` ("Spark does not support time fields") and the
/// nanosecond timestamps ("Cannot convert unsupported type to Spark: ..."); a
/// decimal Spark cannot represent fails with Spark's decimal class.
pub(crate) fn spark_type(t: &Type) -> Result<DataType, KrishivError> {
    Ok(match t {
        Type::Primitive(p) => match p {
            PrimitiveType::Boolean => DataType::Boolean,
            PrimitiveType::Int => DataType::Integer,
            PrimitiveType::Long => DataType::Long,
            PrimitiveType::Float => DataType::Float,
            PrimitiveType::Double => DataType::Double,
            PrimitiveType::Decimal { precision, scale } => {
                DataType::Decimal(DecimalType::try_new(*precision, *scale)?)
            }
            PrimitiveType::Date => DataType::Date,
            PrimitiveType::Time => {
                return Err(unsupported("Spark does not support time fields".to_owned()));
            }
            PrimitiveType::Timestamp => DataType::TimestampNtz,
            PrimitiveType::Timestamptz => DataType::Timestamp,
            PrimitiveType::TimestampNs | PrimitiveType::TimestamptzNs => {
                return Err(unsupported(format!(
                    "Cannot convert unsupported type to Spark: {p}"
                )));
            }
            PrimitiveType::String | PrimitiveType::Uuid => DataType::String,
            PrimitiveType::Fixed(_) | PrimitiveType::Binary => DataType::Binary,
        },
        Type::Struct(s) => DataType::Struct(StructType::new(
            s.fields()
                .iter()
                .map(|f| spark_field(f))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        Type::List(l) => DataType::Array(std::sync::Arc::new(ArrayType::new(
            spark_type(&l.element_field.field_type)?,
            !l.element_field.required,
        ))),
        Type::Map(m) => DataType::Map(std::sync::Arc::new(MapType::new(
            spark_type(&m.key_field.field_type)?,
            spark_type(&m.value_field.field_type)?,
            !m.value_field.required,
        ))),
        Type::Variant(_) => DataType::Variant,
    })
}

/// The Spark field of `f`.
///
/// # Errors
/// As [`spark_type`].
pub(crate) fn spark_field(f: &NestedField) -> Result<StructField, KrishivError> {
    let field = StructField::new(f.name.clone(), spark_type(&f.field_type)?, !f.required);
    Ok(match &f.doc {
        Some(doc) => field.with_metadata(Metadata::with_comment(doc.clone())),
        None => field,
    })
}

/// The Spark schema of `schema`.
///
/// # Errors
/// As [`spark_type`].
pub(crate) fn spark_schema(schema: &Schema) -> Result<StructType, KrishivError> {
    Ok(StructType::new(
        schema
            .as_struct()
            .fields()
            .iter()
            .map(|f| spark_field(f))
            .collect::<Result<Vec<_>, _>>()?,
    ))
}
