// SPDX-License-Identifier: Apache-2.0
//! Tests of the schema texts.

use krishiv_types::DataType;

use super::{parse_datatype_string, parse_local_schema, parse_table_schema};

#[test]
fn a_local_schema_is_ddl_a_type_or_json_and_a_type_is_wrapped() {
    assert_eq!(
        parse_local_schema("a INT, b STRING NOT NULL").unwrap(),
        parse_local_schema(
            r#"{"type":"struct","fields":[{"name":"a","type":"integer","nullable":true,"metadata":{}},{"name":"b","type":"string","nullable":false,"metadata":{}}]}"#
        )
        .unwrap()
    );
    // A bare type becomes `struct<value: t>`.
    let wrapped = parse_local_schema("int").unwrap();
    assert_eq!(wrapped.fields().len(), 1);
    assert_eq!(wrapped.fields()[0].name(), "value");
    let json_type = parse_local_schema("\"long\"").unwrap();
    assert_eq!(json_type.fields()[0].name(), "value");
    // Neither DDL nor JSON: the DDL error.
    assert!(parse_local_schema("a nonsense_type").is_err());
}

#[test]
fn a_data_source_schema_must_be_a_struct() {
    let s = parse_table_schema("a INT, b ARRAY<STRING>").unwrap();
    assert_eq!(
        DataType::Struct(s).simple_string(),
        "struct<a:int,b:array<string>>"
    );
    let json = parse_table_schema(
        r#"{"type":"struct","fields":[{"name":"a","type":"integer","nullable":false,"metadata":{}}]}"#,
    )
    .unwrap();
    assert!(!json.fields()[0].nullable());
    let e = parse_table_schema("\"integer\"").unwrap_err();
    assert_eq!(
        e.error_class().name(),
        "CONNECT_INVALID_PLAN.INVALID_SCHEMA_NON_STRUCT_TYPE"
    );
    // Not DDL, not JSON: the DDL error, never an echo of the text.
    assert!(parse_table_schema("a b c d").is_err());
}

#[test]
fn datatype_strings_try_schema_type_then_struct() {
    assert_eq!(
        parse_datatype_string(" a: int ").unwrap().simple_string(),
        "struct<a:int>"
    );
    assert_eq!(parse_datatype_string("int").unwrap().simple_string(), "int");
}
