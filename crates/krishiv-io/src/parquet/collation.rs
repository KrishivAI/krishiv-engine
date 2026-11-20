// SPDX-License-Identifier: Apache-2.0
//! Collated strings in the Parquet footer (ADR 0043, unit 4).
//!
//! Parquet has no collation: a column is a plain `STRING`. Spark records the
//! column's Spark type in the footer key
//! `org.apache.spark.sql.parquet.row.metadata` (its schema JSON, where a
//! collated leaf is `"string"` plus `__COLLATIONS` field metadata).
//!
//! - **Write.** The key is written for every file (`intervals::row_metadata`,
//!   Spark's schema JSON of the Arrow schema), which carries a collated leaf as
//!   `"string"` plus `__COLLATIONS`.
//! - **Read.** The key is read only to restore collations: every other part
//!   of the schema comes from the Parquet types, as for files without the
//!   key. A key that is missing, is not a struct, or does not fit the file
//!   (other names, other shapes) is ignored.

use std::sync::Arc;

use arrow_schema::{DataType as ArrowType, Field, Schema};
use krishiv_types::DataType;
use krishiv_types::arrow::SPARK_TYPE_METADATA_KEY;
use parquet::file::metadata::KeyValue;

/// Spark's footer key for the schema JSON.
pub(crate) const ROW_METADATA_KEY: &str = "org.apache.spark.sql.parquet.row.metadata";

/// `schema` with the collations that the footer's Spark schema JSON records,
/// as the field tag (only the tag changes; types and nullability stay).
/// `None` when nothing changes.
pub(crate) fn restore(schema: &Schema, kv: Option<&Vec<KeyValue>>) -> Option<Schema> {
    let json = kv?
        .iter()
        .find(|k| k.key == ROW_METADATA_KEY)?
        .value
        .as_deref()?;
    // Collations, and user-defined types (P3-T079): Parquet holds a UDT as its storage type.
    // A cheap prefilter; the JSON may carry whitespace around the colon (`"type": "udt"`), so look for the value.
    if !json.contains("__COLLATIONS") && !json.contains(r#""udt""#) {
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

/// `f` with the field tag of every collated string and user-defined type of `t` below it.
fn tag(f: &Field, t: &DataType) -> Field {
    match (t, f.data_type()) {
        // A UDT is tagged with its JSON when its storage type is what the file holds; a footer whose storage
        // type does not fit the column leaves it alone.
        (DataType::Udt(u), carrier) => {
            let mut meta = f.metadata().clone();
            meta.insert(SPARK_TYPE_METADATA_KEY.to_owned(), t.json());
            let mut tagged = f.clone().with_metadata(meta);
            // A file without Arrow metadata holds an interval as a plain integer: restore the interval carrier
            // so the storage check sees the type the footer declares.
            let integer = matches!(carrier, ArrowType::Int32 | ArrowType::Int64);
            if integer
                && matches!(
                    u.sql_type(),
                    DataType::YearMonthInterval { .. } | DataType::DayTimeInterval { .. }
                )
            {
                tagged = tagged.with_data_type(u.sql_type().to_arrow());
            }
            if DataType::from_arrow_field(&tagged).is_ok() {
                tagged
            } else {
                f.clone()
            }
        }
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
    use krishiv_types::{ArrayType, Collation, UdtType};

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
    fn the_written_row_metadata_carries_and_restores_collations() {
        let schema = Schema::new(vec![
            Field::new("i", ArrowType::Int32, true),
            lcase().to_arrow_field("s", true),
        ]);
        let json = super::super::intervals::row_metadata(&schema).unwrap();
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

    /// A Spark footer that names a UDT tags the column, so it reads back as the UDT (P3-T079); a footer whose
    /// storage type does not fit the column changes nothing.
    #[test]
    fn spark_row_metadata_restores_a_udt() {
        let storage = DataType::Array(Arc::new(ArrayType::new(DataType::Double, false)));
        let udt = DataType::Udt(Arc::new(UdtType::jvm(
            storage.clone(),
            "org.example.PointUDT".to_owned(),
            Some("pkg.PointUDT".to_owned()),
        )));
        let footer = DataType::Struct(krishiv_types::StructType::new(vec![
            krishiv_types::StructField::new("p", udt.clone(), true),
        ]))
        .json();
        let schema = Schema::new(vec![storage.to_arrow_field("p", true)]);
        let restored = restore(&schema, Some(&kv(&footer))).expect("tagged");
        assert_eq!(DataType::from_arrow_field(restored.field(0)).unwrap(), udt);
        // A column of another storage type is left alone.
        let other = Schema::new(vec![DataType::Integer.to_arrow_field("p", true)]);
        assert!(restore(&other, Some(&kv(&footer))).is_none());
    }

    /// Spark writes `"pyClass":null` for a JVM UDT with no Python class, and a Python-only UDT as `pyClass`,
    /// `serializedClass` (a real 4.2.0 run, `tests/spark-golden/udt`); both footers restore the UDT.
    #[test]
    fn spark_row_metadata_restores_a_udt_with_a_null_py_class_and_a_python_only_one() {
        let storage = DataType::Array(Arc::new(ArrayType::new(DataType::Double, false)));
        let sql = r#"{"type":"array","elementType":"double","containsNull":false}"#;
        let schema = Schema::new(vec![storage.to_arrow_field("p", true)]);
        let jvm = DataType::Udt(Arc::new(UdtType::jvm(
            storage.clone(),
            "org.example.PointUDT".to_owned(),
            None,
        )));
        let python = DataType::Udt(Arc::new(UdtType::python(
            storage,
            "__main__.PyOnlyUDT".to_owned(),
            "gAWV".to_owned(),
        )));
        for (udt, want) in [
            (
                format!(
                    r#"{{"type":"udt","class":"org.example.PointUDT","pyClass":null,"sqlType":{sql}}}"#
                ),
                jvm,
            ),
            (
                format!(
                    r#"{{"type":"udt","pyClass":"__main__.PyOnlyUDT","serializedClass":"gAWV","sqlType":{sql}}}"#
                ),
                python,
            ),
        ] {
            let footer = format!(
                r#"{{"type":"struct","fields":[{{"name":"p","type":{udt},"nullable":true,"metadata":{{}}}}]}}"#
            );
            let restored = restore(&schema, Some(&kv(&footer))).expect("tagged");
            let got = DataType::from_arrow_field(restored.field(0)).unwrap();
            assert_eq!(got, want, "{footer}");
            let (DataType::Udt(g), DataType::Udt(w)) = (&got, &want) else {
                panic!("{footer}")
            };
            assert_eq!(g.python_class(), w.python_class(), "{footer}");
            assert_eq!(
                g.serialized_python_class(),
                w.serialized_python_class(),
                "{footer}"
            );
        }
    }

    #[test]
    fn a_footer_with_whitespace_in_its_json_still_restores_a_udt() {
        let storage = DataType::Array(Arc::new(ArrayType::new(DataType::Double, false)));
        let schema = Schema::new(vec![storage.to_arrow_field("p", true)]);
        let footer = r#"{"type": "struct", "fields": [{"name": "p", "type": {"type": "udt", "class": "org.example.PointUDT", "pyClass": null, "sqlType": {"type": "array", "elementType": "double", "containsNull": false}}, "nullable": true, "metadata": {}}]}"#;
        let restored = restore(&schema, Some(&kv(footer))).expect("tagged");
        assert!(matches!(
            DataType::from_arrow_field(restored.field(0)).unwrap(),
            DataType::Udt(_)
        ));
    }

    /// A UDT over a collated string: the footer holds the collation inline (Spark 4.2.0), and a read whose Arrow
    /// schema carries no tag (the restore path alone) brings the collated UDT back.
    #[test]
    fn a_udt_over_a_collated_string_restores_from_the_footer_alone() {
        let udt = DataType::Udt(Arc::new(UdtType::python(
            lcase(),
            "__main__.LUDT".to_owned(),
            "gAWV".to_owned(),
        )));
        let schema = Schema::new(vec![udt.to_arrow_field("p", true)]);
        let json = super::super::intervals::row_metadata(&schema).unwrap();
        assert!(
            json.contains(r#""sqlType":"string collate UTF8_LCASE""#),
            "{json}"
        );
        let untagged = Schema::new(vec![Field::new("p", ArrowType::Utf8View, true)]);
        let back = restore(&untagged, Some(&kv(&json))).expect("restored");
        assert_eq!(DataType::from_arrow_field(back.field(0)).unwrap(), udt);
    }

    /// A UDT over an ANSI interval in a file with no Arrow metadata: the column is a plain integer, and the tag
    /// restores both the UDT and the interval carrier.
    #[test]
    fn a_udt_over_an_interval_restores_from_the_footer_alone() {
        let storage = DataType::YearMonthInterval {
            start: krishiv_types::YearMonthField::Year,
            end: krishiv_types::YearMonthField::Month,
        };
        let udt = DataType::Udt(Arc::new(UdtType::python(
            storage,
            "__main__.IUDT".to_owned(),
            "gAWV".to_owned(),
        )));
        let schema = Schema::new(vec![udt.to_arrow_field("p", true)]);
        let json = super::super::intervals::row_metadata(&schema).unwrap();
        let untagged = Schema::new(vec![Field::new("p", ArrowType::Int32, true)]);
        let back = restore(&untagged, Some(&kv(&json))).expect("restored");
        assert_eq!(DataType::from_arrow_field(back.field(0)).unwrap(), udt);
    }
}
