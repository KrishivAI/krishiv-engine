// SPDX-License-Identifier: Apache-2.0
//! Collated strings in the Parquet footer (ADR 0043, unit 4).
//!
//! Parquet has no collation: a column is a plain `STRING`. Spark records the
//! column's Spark type in the footer key
//! `org.apache.spark.sql.parquet.row.metadata` (its schema JSON, where a
//! collated leaf is `"string"` plus `__COLLATIONS` field metadata).
//!
//! - **Write.** The key is written only when a column holds a collated
//!   string, so files without collations are unchanged byte for byte.
//! - **Read.** The key is read only to restore collations: every other part
//!   of the schema comes from the Parquet types, as for files without the
//!   key. A key that is missing, is not a struct, or does not fit the file
//!   (other names, other shapes) is ignored.

use std::sync::Arc;

use arrow_schema::{DataType as ArrowType, Field, Schema};
use krishiv_types::arrow::SPARK_TYPE_METADATA_KEY;
use krishiv_types::{DataType, StructField, StructType};
use parquet::file::metadata::KeyValue;

/// Spark's footer key for the schema JSON.
pub(crate) const ROW_METADATA_KEY: &str = "org.apache.spark.sql.parquet.row.metadata";

/// Whether `f` or a field nested in it is a collated string.
fn has_collation(f: &Field) -> bool {
    f.metadata()
        .get(SPARK_TYPE_METADATA_KEY)
        .is_some_and(|t| t.starts_with("string collate "))
        || match f.data_type() {
            ArrowType::Struct(fields) => fields.iter().any(|c| has_collation(c)),
            ArrowType::List(c) | ArrowType::LargeList(c) | ArrowType::Map(c, _) => has_collation(c),
            _ => false,
        }
}

/// Spark's schema JSON of `schema`, when a column holds a collated string.
pub(crate) fn row_metadata(schema: &Schema) -> Option<KeyValue> {
    if !schema.fields().iter().any(|f| has_collation(f)) {
        return None;
    }
    let fields: Option<Vec<StructField>> = schema
        .fields()
        .iter()
        .map(|f| {
            DataType::from_arrow_field(f)
                .ok()
                .map(|t| StructField::new(f.name().clone(), t, f.is_nullable()))
        })
        .collect();
    let json = DataType::from(StructType::new(fields?)).json();
    Some(KeyValue::new(ROW_METADATA_KEY.to_owned(), json))
}

/// `schema` with the collations that the footer's Spark schema JSON records,
/// as the field tag (only the tag changes; types and nullability stay).
/// `None` when nothing changes.
pub(crate) fn restore(schema: &Schema, kv: Option<&Vec<KeyValue>>) -> Option<Schema> {
    let json = kv?
        .iter()
        .find(|k| k.key == ROW_METADATA_KEY)?
        .value
        .as_deref()?;
    if !json.contains("__COLLATIONS") {
        return None;
    }
    let DataType::Struct(spark) = DataType::from_json(json).ok()? else {
        return None;
    };
    let mut changed = false;
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| {
            let Some(s) = spark.fields().iter().find(|s| s.name() == f.name()) else {
                return f.as_ref().clone();
            };
            let restored = tag(f, s.data_type());
            changed |= restored != **f;
            restored
        })
        .collect();
    changed.then(|| Schema::new_with_metadata(fields, schema.metadata().clone()))
}

/// `f` with the field tag of every collated string of `t` below it.
fn tag(f: &Field, t: &DataType) -> Field {
    match (t, f.data_type()) {
        (
            DataType::CollatedString(_),
            ArrowType::Utf8 | ArrowType::LargeUtf8 | ArrowType::Utf8View,
        ) => {
            let mut meta = f.metadata().clone();
            meta.insert(SPARK_TYPE_METADATA_KEY.to_owned(), t.catalog_string());
            f.clone().with_metadata(meta)
        }
        (DataType::Struct(st), ArrowType::Struct(children)) => {
            let children: Vec<Arc<Field>> = children
                .iter()
                .map(
                    |c| match st.fields().iter().find(|s| s.name() == c.name()) {
                        Some(s) => Arc::new(tag(c, s.data_type())),
                        None => Arc::clone(c),
                    },
                )
                .collect();
            f.clone().with_data_type(ArrowType::Struct(children.into()))
        }
        (DataType::Array(a), ArrowType::List(c)) => f
            .clone()
            .with_data_type(ArrowType::List(Arc::new(tag(c, a.element_type())))),
        (DataType::Array(a), ArrowType::LargeList(c)) => f
            .clone()
            .with_data_type(ArrowType::LargeList(Arc::new(tag(c, a.element_type())))),
        (DataType::Map(m), ArrowType::Map(entries, sorted)) => {
            let ArrowType::Struct(kv) = entries.data_type() else {
                return f.clone();
            };
            let [k, v] = kv.as_ref() else {
                return f.clone();
            };
            let kv = vec![
                Arc::new(tag(k, m.key_type())),
                Arc::new(tag(v, m.value_type())),
            ];
            let entries = entries
                .as_ref()
                .clone()
                .with_data_type(ArrowType::Struct(kv.into()));
            f.clone()
                .with_data_type(ArrowType::Map(Arc::new(entries), *sorted))
        }
        _ => f.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use krishiv_types::Collation;

    fn lcase() -> DataType {
        DataType::collated_string(Collation::parse("UTF8_LCASE").unwrap())
    }

    fn kv(json: &str) -> Vec<KeyValue> {
        vec![KeyValue::new(ROW_METADATA_KEY.to_owned(), json.to_owned())]
    }

    /// Spark 4.2's JSON for `s string collate UTF8_LCASE, st struct<x: string
    /// collate UNICODE_CI, n int>, arr array<string collate UTF8_LCASE>,
    /// m map<string, string collate UTF8_LCASE>, p string`.
    const SPARK_JSON: &str = r#"{"type":"struct","fields":[
        {"name":"s","type":"string","nullable":false,"metadata":{"__COLLATIONS":{"s":"spark.UTF8_LCASE"}}},
        {"name":"st","type":{"type":"struct","fields":[
            {"name":"x","type":"string","nullable":false,"metadata":{"__COLLATIONS":{"x":"icu.UNICODE_CI"}}},
            {"name":"n","type":"integer","nullable":false,"metadata":{}}]},"nullable":false,"metadata":{}},
        {"name":"arr","type":{"type":"array","elementType":"string","containsNull":false},"nullable":false,"metadata":{"__COLLATIONS":{"arr.element":"spark.UTF8_LCASE"}}},
        {"name":"m","type":{"type":"map","keyType":"string","valueType":"string","valueContainsNull":false},"nullable":false,"metadata":{"__COLLATIONS":{"m.value":"spark.UTF8_LCASE"}}},
        {"name":"p","type":"string","nullable":false,"metadata":{}}]}"#;

    fn plain_schema() -> Schema {
        let item = |t| Arc::new(Field::new("element", t, false));
        let entries = Field::new(
            "key_value",
            ArrowType::Struct(
                vec![
                    Field::new("key", ArrowType::Utf8, false),
                    Field::new("value", ArrowType::Utf8, false),
                ]
                .into(),
            ),
            false,
        );
        Schema::new(vec![
            Field::new("s", ArrowType::Utf8, false),
            Field::new(
                "st",
                ArrowType::Struct(
                    vec![
                        Field::new("x", ArrowType::Utf8, false),
                        Field::new("n", ArrowType::Int32, false),
                    ]
                    .into(),
                ),
                false,
            ),
            Field::new("arr", ArrowType::List(item(ArrowType::Utf8)), false),
            Field::new("m", ArrowType::Map(Arc::new(entries), false), false),
            Field::new("p", ArrowType::Utf8, false),
        ])
    }

    #[test]
    fn spark_row_metadata_restores_nested_collations() {
        let restored = restore(&plain_schema(), Some(&kv(SPARK_JSON))).unwrap();
        let ty = |name: &str| DataType::from_arrow_field(restored.field_with_name(name).unwrap());
        assert_eq!(ty("s").unwrap(), lcase());
        assert_eq!(ty("p").unwrap(), DataType::String);
        let DataType::Struct(st) = ty("st").unwrap() else {
            panic!("struct");
        };
        assert_eq!(
            st.fields()[0].data_type().sql(),
            "STRING COLLATE UNICODE_CI"
        );
        assert_eq!(st.fields()[1].data_type(), &DataType::Integer);
        let DataType::Array(a) = ty("arr").unwrap() else {
            panic!("array");
        };
        assert_eq!(a.element_type(), &lcase());
        let DataType::Map(m) = ty("m").unwrap() else {
            panic!("map");
        };
        assert_eq!(
            (m.key_type(), m.value_type()),
            (&DataType::String, &lcase())
        );
    }

    #[test]
    fn unusable_row_metadata_changes_nothing() {
        let schema = plain_schema();
        assert!(restore(&schema, None).is_none());
        // No collations in the JSON.
        let plain = r#"{"type":"struct","fields":[{"name":"s","type":"string","nullable":true,"metadata":{}}]}"#;
        assert!(restore(&schema, Some(&kv(plain))).is_none());
        // Not JSON, not a struct, other names.
        assert!(restore(&schema, Some(&kv("__COLLATIONS nonsense"))).is_none());
        assert!(restore(&schema, Some(&kv(r#""string""#))).is_none());
        let other = SPARK_JSON.replace("\"name\":\"s\"", "\"name\":\"zz\"");
        let r = restore(&schema, Some(&kv(&other))).unwrap();
        assert_eq!(
            DataType::from_arrow_field(r.field_with_name("s").unwrap()).unwrap(),
            DataType::String
        );
    }

    #[test]
    fn row_metadata_is_written_only_for_collated_columns() {
        assert!(row_metadata(&plain_schema()).is_none());
        let schema = Schema::new(vec![
            Field::new("i", ArrowType::Int32, true),
            lcase().to_arrow_field("s", true),
        ]);
        let kv = row_metadata(&schema).unwrap();
        assert_eq!(kv.key, ROW_METADATA_KEY);
        let json = kv.value.unwrap();
        assert!(
            json.contains(r#""__COLLATIONS":{"s":"spark.UTF8_LCASE"}"#),
            "{json}"
        );
        // Reading the key back restores the tag.
        let back = restore(
            &Schema::new(vec![
                Field::new("i", ArrowType::Int32, true),
                Field::new("s", ArrowType::Utf8View, true),
            ]),
            Some(&vec![KeyValue::new(ROW_METADATA_KEY.to_owned(), json)]),
        )
        .unwrap();
        assert_eq!(
            DataType::from_arrow_field(back.field_with_name("s").unwrap()).unwrap(),
            lcase()
        );
    }
}
