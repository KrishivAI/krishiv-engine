// SPDX-License-Identifier: Apache-2.0
//! Unit tests of the schema texts: tree strings, DDL and type parsing.
use std::sync::Arc;

use krishiv_types::{ArrayType, DataType, DecimalType, MapType, Metadata, StructField, StructType};

use super::{json_to_ddl, parse_datatype_string, to_ddl, tree_string};

fn schema() -> StructType {
    let inner = StructType::new([StructField::new(
        "c",
        DataType::Array(Arc::new(ArrayType::new(DataType::Long, true))),
        true,
    )]);
    StructType::new([
        StructField::new("a", DataType::Integer, true),
        StructField::new("b", DataType::Struct(inner), false),
        StructField::new(
            "m",
            DataType::Map(Arc::new(MapType::new(
                DataType::String,
                DataType::Decimal(DecimalType::try_new(10, 2).unwrap()),
                true,
            ))),
            true,
        ),
    ])
}

#[test]
fn tree_string_prints_every_level() {
    assert_eq!(
        tree_string(&schema(), None),
        "root\n\
         \x20|-- a: integer (nullable = true)\n\
         \x20|-- b: struct (nullable = false)\n\
         \x20|    |-- c: array (nullable = true)\n\
         \x20|    |    |-- element: long (containsNull = true)\n\
         \x20|-- m: map (nullable = true)\n\
         \x20|    |-- key: string\n\
         \x20|    |-- value: decimal(10,2) (valueContainsNull = true)\n"
    );
    // A level of 0 or below is every level, as Spark's `treeString(0)`.
    assert_eq!(
        tree_string(&schema(), Some(0)),
        tree_string(&schema(), None)
    );
    assert_eq!(
        tree_string(&schema(), Some(-3)),
        tree_string(&schema(), None)
    );
}

#[test]
fn tree_string_stops_at_the_level() {
    assert_eq!(
        tree_string(&schema(), Some(1)),
        "root\n |-- a: integer (nullable = true)\n |-- b: struct (nullable = false)\n \
         |-- m: map (nullable = true)\n"
    );
    assert_eq!(
        tree_string(&schema(), Some(2)),
        "root\n\
         \x20|-- a: integer (nullable = true)\n\
         \x20|-- b: struct (nullable = false)\n\
         \x20|    |-- c: array (nullable = true)\n\
         \x20|-- m: map (nullable = true)\n\
         \x20|    |-- key: string\n\
         \x20|    |-- value: decimal(10,2) (valueContainsNull = true)\n"
    );
}

#[test]
fn tree_string_escapes_meta_characters_in_names() {
    let s = StructType::new([StructField::new("a\nb\t\u{7}", DataType::Byte, true)]);
    assert_eq!(
        tree_string(&s, None),
        "root\n |-- a\\nb\\t\\a: byte (nullable = true)\n"
    );
}

#[test]
fn to_ddl_quotes_names_and_keeps_not_null_default_and_comment() {
    let s = StructType::new([
        StructField::new("id", DataType::Long, false),
        StructField::new("my col", DataType::String, true)
            .with_metadata(Metadata::with_comment("it's \\ here")),
        StructField::new("n", DataType::Integer, true).with_metadata(Metadata::from_map(
            serde_json::from_str(r#"{"CURRENT_DEFAULT":"42"}"#).unwrap(),
        )),
    ]);
    assert_eq!(
        to_ddl(&s),
        "id BIGINT NOT NULL,`my col` STRING COMMENT 'it\\'s \\\\ here',n INT DEFAULT 42"
    );
}

#[test]
fn json_to_ddl_needs_a_struct() {
    let json = r#"{"type":"struct","fields":[{"name":"a","type":"integer","nullable":true,"metadata":{}},{"name":"b","type":{"type":"array","elementType":"string","containsNull":true},"nullable":false,"metadata":{}}]}"#;
    assert_eq!(json_to_ddl(json).unwrap(), "a INT,b ARRAY<STRING> NOT NULL");
    let e = json_to_ddl("\"integer\"").unwrap_err();
    assert_eq!(e.error_class().name(), "INVALID_SCHEMA.NON_STRUCT_TYPE");
    let e = json_to_ddl("{").unwrap_err();
    assert_eq!(e.error_class().name(), "INVALID_JSON_DATA_TYPE");
}

#[test]
fn parse_datatype_string_tries_schema_type_then_struct() {
    let t = parse_datatype_string("a INT, b STRING NOT NULL").unwrap();
    assert_eq!(t.simple_string(), "struct<a:int,b:string>");
    assert_eq!(parse_datatype_string("int").unwrap(), DataType::Integer);
    assert_eq!(
        parse_datatype_string("array<bigint>")
            .unwrap()
            .simple_string(),
        "array<bigint>"
    );
    assert_eq!(
        parse_datatype_string(" a: int ").unwrap().simple_string(),
        "struct<a:int>"
    );
    assert!(parse_datatype_string("a nonsense_type").is_err());
}
