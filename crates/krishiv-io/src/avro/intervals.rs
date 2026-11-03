// SPDX-License-Identifier: Apache-2.0
//! ANSI intervals in Avro, as Spark stores them (design P2-T001b §3): a
//! year-month interval is an `int`, a day-time interval a `long`, and the Spark
//! type survives only in the schema's [`CATALYST_TYPE`] property.
//!
//! `apache_avro` drops unknown properties of a primitive type, so [`typed`]
//! reads them from the schema's JSON, beside the parsed schema.

use apache_avro::Schema as AvroSchema;
use arrow_schema::{DataType, Field, IntervalUnit, TimeUnit};
use krishiv_types::DataType as SparkType;
use serde_json::Value as J;

use super::schema::{UnionShape, schema_kind, union_shape};
use crate::error::{FormatError, FormatResult, unsupported};

/// The property that names a primitive's Spark type.
const CATALYST_TYPE: &str = "spark.sql.catalyst.type";

/// The fragment of an ANSI interval column of Arrow type `dt`: its Spark type
/// is `spark` when that is an interval, else the type's default fields.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for an Arrow type with no interval form.
pub(super) fn fragment(dt: &DataType, spark: Option<&SparkType>) -> FormatResult<String> {
    let avro = match dt {
        DataType::Interval(IntervalUnit::YearMonth) => "int",
        DataType::Duration(TimeUnit::Microsecond) => "long",
        other => return Err(refused(format!("writing the Arrow type {other} as Avro"))),
    };
    let name = match spark {
        Some(t @ (SparkType::YearMonthInterval { .. } | SparkType::DayTimeInterval { .. })) => {
            t.catalog_string()
        }
        _ => SparkType::from_arrow(dt)
            .map_err(FormatError::Other)?
            .catalog_string(),
    };
    Ok(format!(r#"{{"type":"{avro}","{CATALYST_TYPE}":"{name}"}}"#))
}

/// `fields`, the Arrow fields of the record `schema`, with the interval types
/// that `json`, the same schema's text, names. Text that doesn't parse leaves
/// them alone; `apache_avro` has parsed it already.
///
/// # Errors
/// `KRISHIV_UNSUPPORTED_FEATURE` for a property other than an interval of the
/// primitive's family, or the primitive's own type.
pub(super) fn typed_fields(
    fields: Vec<Field>,
    schema: &AvroSchema,
    json: &str,
) -> FormatResult<Vec<Field>> {
    let Ok(json) = serde_json::from_str::<J>(json) else {
        return Ok(fields);
    };
    match schema {
        AvroSchema::Record(r) => record_fields(fields.into_iter(), &r.fields, &json),
        _ => Ok(fields),
    }
}

fn record_fields(
    fields: impl Iterator<Item = Field>,
    schemas: &[apache_avro::schema::RecordField],
    record: &J,
) -> FormatResult<Vec<Field>> {
    let raw = type_of(record).get("fields").and_then(J::as_array);
    fields
        .zip(schemas)
        .enumerate()
        .map(|(i, (f, s))| {
            // A property on the record field, beside its `type`, is not the
            // type's.
            match raw.and_then(|r| r.get(i)).and_then(|r| r.get("type")) {
                Some(t) => typed(f, &s.schema, t),
                None => Ok(f),
            }
        })
        .collect()
}

/// The node that holds a schema: `{"type": {...}}` is its inner object.
fn type_of(json: &J) -> &J {
    match json.get("type") {
        Some(inner @ (J::Object(_) | J::Array(_))) => type_of(inner),
        _ => json,
    }
}

fn is_null_branch(json: &J) -> bool {
    json.as_str() == Some("null") || type_of(json).get("type").and_then(J::as_str) == Some("null")
}

/// `field`, the Arrow field of `schema`, with the interval types `json` names.
fn typed(field: Field, schema: &AvroSchema, json: &J) -> FormatResult<Field> {
    let json = type_of(json);
    match schema {
        AvroSchema::Int | AvroSchema::Long => primitive(field, schema, json),
        AvroSchema::Union(u) => {
            let Some(raw) = json.as_array() else {
                return Ok(field);
            };
            let branches = raw.iter().filter(|b| !is_null_branch(b));
            match union_shape(u.variants()) {
                UnionShape::Single(inner) => match branches.clone().next() {
                    Some(b) => typed(field, inner, b),
                    None => Ok(field),
                },
                UnionShape::Long | UnionShape::Double => Ok(field),
                UnionShape::Members => {
                    let DataType::Struct(members) = field.data_type() else {
                        return Ok(field);
                    };
                    let variants = u
                        .variants()
                        .iter()
                        .filter(|s| !matches!(s, AvroSchema::Null));
                    let members = members
                        .iter()
                        .zip(variants)
                        .zip(branches)
                        .map(|((m, s), b)| typed(m.as_ref().clone(), s, b))
                        .collect::<FormatResult<Vec<_>>>()?;
                    Ok(field.with_data_type(DataType::Struct(members.into())))
                }
            }
        }
        AvroSchema::Record(r) => {
            let DataType::Struct(fields) = field.data_type() else {
                return Ok(field);
            };
            let fields = record_fields(fields.iter().map(|f| f.as_ref().clone()), &r.fields, json)?;
            Ok(field.with_data_type(DataType::Struct(fields.into())))
        }
        AvroSchema::Array(a) => {
            let (DataType::List(item), Some(items)) = (field.data_type(), json.get("items")) else {
                return Ok(field);
            };
            let item = typed(item.as_ref().clone(), &a.items, items)?;
            Ok(field.with_data_type(DataType::List(item.into())))
        }
        AvroSchema::Map(m) => {
            let (DataType::Map(entries, sorted), Some(values)) =
                (field.data_type(), json.get("values"))
            else {
                return Ok(field);
            };
            let DataType::Struct(kv) = entries.data_type() else {
                return Ok(field);
            };
            let (Some(key), Some(value)) = (kv.first(), kv.get(1)) else {
                return Ok(field);
            };
            let value = typed(value.as_ref().clone(), &m.types, values)?;
            let kv = vec![key.as_ref().clone(), value];
            let entries = entries
                .as_ref()
                .clone()
                .with_data_type(DataType::Struct(kv.into()));
            let sorted = *sorted;
            Ok(field.with_data_type(DataType::Map(entries.into(), sorted)))
        }
        _ => Ok(field),
    }
}

/// `field` under the type a primitive's property names. Spark widens an `int`
/// named `bigint` and fails on the other mismatches as it reads; Krishiv
/// refuses them all by name.
fn primitive(field: Field, schema: &AvroSchema, json: &J) -> FormatResult<Field> {
    let Some(text) = json.get(CATALYST_TYPE).and_then(J::as_str) else {
        return Ok(field);
    };
    let refuse = || {
        refused(format!(
            "the Avro type {} with {CATALYST_TYPE} {text}",
            schema_kind(schema)
        ))
    };
    let spark = krishiv_types::parse_data_type(text).map_err(|_| refuse())?;
    match (schema, &spark) {
        (AvroSchema::Int, SparkType::Integer) | (AvroSchema::Long, SparkType::Long) => Ok(field),
        (AvroSchema::Int, SparkType::YearMonthInterval { .. })
        | (AvroSchema::Long, SparkType::DayTimeInterval { .. }) => {
            Ok(spark.to_arrow_field(field.name(), field.is_nullable()))
        }
        _ => Err(refuse()),
    }
}

/// `KRISHIV_UNSUPPORTED_FEATURE` as a format error.
fn refused(feature: String) -> FormatError {
    FormatError::Other(unsupported(feature))
}

#[cfg(test)]
mod tests {
    use apache_avro::Schema as AvroSchema;
    use arrow_schema::{DataType, Field, Fields, IntervalUnit, Schema, TimeUnit};
    use krishiv_common::error::ErrorClass;
    use krishiv_types::DataType as SparkType;

    use super::super::schema::{fields_of_record, schema_json_of_arrow};
    use super::typed_fields;
    use crate::error::{FormatError, FormatResult};

    /// The types of the fields of a record of one field `a` of type `t`.
    fn read(t: &str) -> FormatResult<Field> {
        let json =
            format!(r#"{{"type":"record","name":"r","fields":[{{"name":"a","type":{t}}}]}}"#);
        field_of(&json)
    }

    fn field_of(json: &str) -> FormatResult<Field> {
        let schema =
            AvroSchema::parse_str(json).map_err(|e| FormatError::Corrupt(e.to_string()))?;
        let fields = typed_fields(fields_of_record(&schema)?, &schema, json)?;
        Ok(fields[0].clone())
    }

    fn spark(f: &Field) -> String {
        SparkType::from_arrow_field(f).map_or_else(|e| e.to_string(), |t| t.simple_string())
    }

    fn prop(avro: &str, spark: &str) -> String {
        format!(r#"{{"type":"{avro}","spark.sql.catalyst.type":"{spark}"}}"#)
    }

    fn parsed(t: &str) -> SparkType {
        krishiv_types::parse_data_type(t).unwrap()
    }

    /// Each from Spark 4.2.0 over `avroSchema`.
    #[test]
    fn a_property_names_the_primitive_s_interval() -> FormatResult<()> {
        for (t, expected) in [
            (prop("int", "interval year to month"), "interval year to month"),
            (prop("int", "interval month"), "interval month"),
            (prop("long", "interval day to hour"), "interval day to hour"),
            (prop("long", "INTERVAL DAY TO SECOND"), "interval day to second"),
            (format!(r#"["null",{}]"#, prop("long", "interval day")), "interval day"),
            (format!(r#"[{},"null"]"#, prop("long", "interval day")), "interval day"),
            (
                format!(r#"{{"type":"array","items":{}}}"#, prop("int", "interval year")),
                "array<interval year>",
            ),
            (
                format!(r#"{{"type":"array","items":[{},"null"]}}"#, prop("int", "interval year")),
                "array<interval year>",
            ),
            (
                format!(r#"{{"type":"map","values":{}}}"#, prop("long", "interval hour")),
                "map<string,interval hour>",
            ),
            (
                format!(
                    r#"{{"type":"record","name":"n","fields":[{{"name":"x","type":{}}}]}}"#,
                    prop("long", "interval day")
                ),
                "struct<x:interval day>",
            ),
            (prop("int", "int"), "int"),
            (prop("long", "bigint"), "bigint"),
            // Not a primitive `int` or `long`, so the property is not read.
            (prop("string", "interval day"), "string"),
            (prop("double", "interval day"), "double"),
            (
                r#"{"type":"long","logicalType":"timestamp-micros","spark.sql.catalyst.type":"interval day"}"#
                    .to_owned(),
                "timestamp",
            ),
            // Not text: `getProp` has no answer.
            (r#"{"type":"int","spark.sql.catalyst.type":1}"#.to_owned(), "int"),
            // An `int` and `long` union is a `long`, whatever a branch says.
            (format!(r#"[{},"long"]"#, prop("int", "interval month")), "bigint"),
        ] {
            assert_eq!(spark(&read(&t)?), expected, "{t}");
        }
        Ok(())
    }

    #[test]
    fn a_union_s_members_keep_their_own_types() -> FormatResult<()> {
        let f = read(&format!(
            r#"["null",{},"string"]"#,
            prop("long", "interval day")
        ))?;
        assert_eq!(spark(&f), "struct<member0:interval day,member1:string>");
        Ok(())
    }

    #[test]
    fn a_property_on_the_record_field_is_not_the_type_s() -> FormatResult<()> {
        let f = field_of(
            r#"{"type":"record","name":"r","fields":[{"name":"a","type":"int","spark.sql.catalyst.type":"interval month"}]}"#,
        )?;
        assert_eq!(spark(&f), "int");
        Ok(())
    }

    #[test]
    fn another_property_is_refused_by_name() {
        for t in [
            prop("int", "interval day"),
            prop("long", "interval year"),
            prop("int", "bigint"),
            prop("int", "string"),
            prop("long", "int"),
            prop("int", "garbage!"),
        ] {
            let e = read(&t).expect_err(&t);
            let FormatError::Other(e) = e else {
                panic!("{e}");
            };
            assert_eq!(
                e.error_class(),
                ErrorClass::KrishivUnsupportedFeature,
                "{t}"
            );
            assert!(e.to_string().contains("spark.sql.catalyst.type"), "{e}");
        }
    }

    #[test]
    fn a_written_interval_names_its_spark_type_at_any_depth() -> FormatResult<()> {
        let tagged = |name: &str, t: &SparkType| t.to_arrow_field(name, false);
        let ym = parsed("interval year");
        let dh = parsed("interval day to hour");
        let plain_list = DataType::List(
            Field::new(
                "element",
                DataType::Interval(IntervalUnit::YearMonth),
                false,
            )
            .into(),
        );
        let list = parsed("array<interval year>")
            .to_arrow_field_with_storage("l", plain_list, false)
            .map_err(FormatError::Other)?;
        let inner: Fields = vec![tagged("x", &dh)].into();
        let schema = Schema::new(vec![
            tagged("a", &ym),
            tagged("b", &dh),
            list,
            Field::new("s", DataType::Struct(inner), false),
            Field::new("d", DataType::Duration(TimeUnit::Microsecond), false),
        ]);
        let json = schema_json_of_arrow(&schema, "r", "")?;
        for fragment in [
            prop("int", "interval year"),
            prop("long", "interval day to hour"),
            format!(
                r#"{{"type":"array","items":{}}}"#,
                prop("int", "interval year")
            ),
            prop("long", "interval day to second"),
        ] {
            assert!(json.contains(&fragment), "{fragment} in {json}");
        }
        Ok(())
    }
}
