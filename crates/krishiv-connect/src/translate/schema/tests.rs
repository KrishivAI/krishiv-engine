// SPDX-License-Identifier: Apache-2.0
//! Tests of the schema texts.

use krishiv_sql::ParserOptions;
use krishiv_types::{DataType, StructType};

use super::{parse_datatype_string, parse_local_schema, parse_table_schema};

#[test]
fn a_local_schema_is_ddl_a_type_or_json_and_a_type_is_wrapped() {
    assert_eq!(
        parse_local_schema("a INT, b STRING NOT NULL", &ParserOptions::default()).unwrap(),
        parse_local_schema(
            r#"{"type":"struct","fields":[{"name":"a","type":"integer","nullable":true,"metadata":{}},{"name":"b","type":"string","nullable":false,"metadata":{}}]}"#, &ParserOptions::default())
        .unwrap()
    );
    // A bare type becomes `struct<value: t>`.
    let wrapped = parse_local_schema("int", &ParserOptions::default()).unwrap();
    assert_eq!(wrapped.fields().len(), 1);
    assert_eq!(wrapped.fields()[0].name(), "value");
    assert_eq!(wrapped.fields()[0].data_type(), &DataType::Integer);
    assert!(wrapped.fields()[0].nullable());
    let json_type = parse_local_schema("\"long\"", &ParserOptions::default()).unwrap();
    assert_eq!(json_type.fields()[0].name(), "value");
    assert_eq!(json_type.fields()[0].data_type(), &DataType::Long);
    assert!(json_type.fields()[0].nullable());
    // Neither DDL nor JSON: the DDL error.
    assert!(parse_local_schema("a nonsense_type", &ParserOptions::default()).is_err());
}

#[test]
fn a_data_source_schema_must_be_a_struct() {
    let s = parse_table_schema("a INT, b ARRAY<STRING>", &ParserOptions::default()).unwrap();
    assert_eq!(
        DataType::Struct(s).simple_string(),
        "struct<a:int,b:array<string>>"
    );
    let json = parse_table_schema(
        r#"{"type":"struct","fields":[{"name":"a","type":"integer","nullable":false,"metadata":{}}]}"#, &ParserOptions::default())
    .unwrap();
    assert!(!json.fields()[0].nullable());
    let e = parse_table_schema("\"integer\"", &ParserOptions::default()).unwrap_err();
    assert_eq!(
        e.error_class().name(),
        "CONNECT_INVALID_PLAN.INVALID_SCHEMA_NON_STRUCT_TYPE"
    );
    // Not DDL, not JSON: the DDL error, never an echo of the text.
    assert!(parse_table_schema("a b c d", &ParserOptions::default()).is_err());
}

#[test]
fn datatype_strings_try_schema_type_then_struct() {
    assert_eq!(
        parse_datatype_string(" a: int ", &ParserOptions::default())
            .unwrap()
            .simple_string(),
        "struct<a:int>"
    );
    assert_eq!(
        parse_datatype_string("int", &ParserOptions::default())
            .unwrap()
            .simple_string(),
        "int"
    );
}

#[test]
fn a_trailing_line_comment_does_not_swallow_the_closing_delimiter() {
    // Spark parses a table schema directly, so a line comment may end at the end of the text.
    for text in ["a INT -- note", "a INT, b STRING -- note"] {
        let local = parse_local_schema(text, &ParserOptions::default()).unwrap();
        let table = parse_table_schema(text, &ParserOptions::default()).unwrap();
        assert_eq!(local, table, "{text}");
        assert_eq!(local.fields()[0].name(), "a", "{text}");
    }
    assert_eq!(
        DataType::Struct(
            parse_table_schema("a INT, b STRING -- note", &ParserOptions::default()).unwrap()
        )
        .simple_string(),
        "struct<a:int,b:string>"
    );
}

/// Schema texts follow `spark.sql.timestampType` for a bare `TIMESTAMP`, as Spark Connect's
/// planner parses them with the session's parser.
#[test]
fn schema_texts_follow_the_session_timestamp_type() {
    let ntz = ParserOptions {
        timestamp_ntz: true,
        ..ParserOptions::default()
    };
    let shown = |s: StructType| DataType::Struct(s).simple_string();
    assert_eq!(
        shown(parse_table_schema("a TIMESTAMP", &ntz).unwrap()),
        "struct<a:timestamp_ntz>"
    );
    assert_eq!(
        shown(parse_local_schema("a TIMESTAMP, b TIMESTAMP_LTZ", &ntz).unwrap()),
        "struct<a:timestamp_ntz,b:timestamp>"
    );
    assert_eq!(
        parse_datatype_string("timestamp", &ntz).unwrap(),
        DataType::TimestampNtz
    );
    assert_eq!(
        shown(parse_table_schema("a TIMESTAMP", &ParserOptions::default()).unwrap()),
        "struct<a:timestamp>"
    );
}
