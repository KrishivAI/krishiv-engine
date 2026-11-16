// SPDX-License-Identifier: Apache-2.0
//! Iceberg types as Spark sees them (`iceberg_type_mapping`): the mapping of
//! Iceberg's `TypeToSparkType` at apache-iceberg 1.12.0, Spark 4.1 module
//! (`research/p3/t022a-iceberg-spark-types.md`). Field nullability is the field's
//! `optional`, a list's `containsNull` its element's, a map's `valueContainsNull`
//! its value's; a doc becomes the `comment` metadata; field ids are not carried by the Spark
//! schema, only by the Arrow table schema a scan maps files onto ([`arrow_schema`]).
//! This is the Spark 4.x mapping (Spark 3.5 has no `variant`). Column default values
//! (format v3) are refused until Spark's `EXISTS_DEFAULT` / `CURRENT_DEFAULT` metadata
//! and the fill of missing columns exist.

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
    if f.initial_default.is_some() || f.write_default.is_some() {
        // Spark shows them as `EXISTS_DEFAULT`/`CURRENT_DEFAULT` SQL text and fills
        // missing columns with them; neither is built yet, so refuse rather than read
        // the column as NULL.
        return Err(unsupported(format!(
            "Iceberg column default values (column {})",
            f.name
        )));
    }
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

/// The Arrow form of [`spark_schema`] (the table schema a scan maps files
/// onto), with every field's Iceberg id in its [`krishiv_io::FIELD_ID_KEY`]
/// metadata, nested fields included, so files are read by id.
///
/// # Errors
/// As [`spark_type`].
pub(crate) fn arrow_schema(schema: &Schema) -> Result<arrow_schema::SchemaRef, KrishivError> {
    let spark = spark_schema(schema)?;
    let fields: Vec<arrow_schema::Field> = spark
        .fields()
        .iter()
        .zip(schema.as_struct().fields())
        .map(|(s, f)| {
            with_ids(
                s.data_type().to_arrow_field(s.name(), s.nullable()),
                f.id,
                &f.field_type,
            )
        })
        .collect();
    Ok(std::sync::Arc::new(arrow_schema::Schema::new(fields)))
}

/// `field` (the Arrow form of an Iceberg field of type `ty`) with `id` and
/// the ids of its nested fields.
fn with_ids(field: arrow_schema::Field, id: i32, ty: &Type) -> arrow_schema::Field {
    use arrow_schema::DataType as A;
    use std::sync::Arc;
    let child = |f: &arrow_schema::FieldRef, n: &NestedField| {
        Arc::new(with_ids(f.as_ref().clone(), n.id, &n.field_type))
    };
    let data_type = match (field.data_type(), ty) {
        (A::Struct(children), Type::Struct(s)) if children.len() == s.fields().len() => A::Struct(
            children
                .iter()
                .zip(s.fields())
                .map(|(c, n)| child(c, n))
                .collect(),
        ),
        (A::List(e), Type::List(l)) => A::List(child(e, &l.element_field)),
        (A::Map(entries, sorted), Type::Map(m)) => match entries.data_type() {
            A::Struct(kv) if kv.len() == 2 => A::Map(
                Arc::new(entries.as_ref().clone().with_data_type(A::Struct(
                    vec![child(&kv[0], &m.key_field), child(&kv[1], &m.value_field)].into(),
                ))),
                *sorted,
            ),
            _ => field.data_type().clone(),
        },
        _ => field.data_type().clone(),
    };
    let mut metadata = field.metadata().clone();
    metadata.insert(krishiv_io::FIELD_ID_KEY.to_owned(), id.to_string());
    field.with_data_type(data_type).with_metadata(metadata)
}

/// One field of an Iceberg name mapping as serialised (table spec "Column
/// Projection": `field-id` optional, `names`, nested `fields`).
#[derive(serde::Deserialize)]
struct MappedFieldJson {
    #[serde(rename = "field-id")]
    field_id: Option<i32>,
    names: Vec<String>,
    #[serde(default)]
    fields: Vec<MappedFieldJson>,
}

impl From<MappedFieldJson> for krishiv_io::MappedField {
    fn from(f: MappedFieldJson) -> Self {
        Self {
            id: f.field_id,
            names: f.names,
            fields: f.fields.into_iter().map(Into::into).collect(),
        }
    }
}

/// The name mapping `text` (the table property
/// `schema.name-mapping.default`, Java's `NameMappingParser.fromJson`).
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` naming the parse error for text that is not
/// a name mapping (a field without `names`, as Java requires them) or that
/// gives one name twice at a level (Java's `MappedFields` refuses it): files
/// without ids are then never read by a guess.
pub(crate) fn parse_name_mapping(text: &str) -> Result<krishiv_io::NameMapping, KrishivError> {
    let bad = |e: String| {
        unsupported(format!(
            "reading the Iceberg name mapping schema.name-mapping.default: {e}"
        ))
    };
    let fields: Vec<MappedFieldJson> =
        serde_json::from_str(text).map_err(|e| bad(e.to_string()))?;
    fn unique(level: &[MappedFieldJson]) -> Result<(), String> {
        let mut seen = std::collections::BTreeSet::new();
        for f in level {
            for n in &f.names {
                if !seen.insert(n.as_str()) {
                    return Err(format!("the name {n} is mapped twice"));
                }
            }
            unique(&f.fields)?;
        }
        Ok(())
    }
    unique(&fields).map_err(bad)?;
    Ok(krishiv_io::NameMapping::new(
        fields.into_iter().map(Into::into).collect(),
    ))
}
