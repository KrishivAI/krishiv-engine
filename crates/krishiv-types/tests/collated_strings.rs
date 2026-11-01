// SPDX-License-Identifier: Apache-2.0
//! Collated string types against Spark 4.2.0 (a PySpark 4.2.0 run for the JSON
//! forms; `typeof` and `SHOW CREATE TABLE` for the renderings): names,
//! renderings, Arrow field tags, DDL strings and the `__COLLATIONS` JSON.

#![expect(clippy::panic, reason = "test helpers sit outside #[test] fns")]

use krishiv_common::error::ErrorClass;
use krishiv_types::{Collation, DataType, parse_data_type};
use serde_json::Value;

fn t(ddl: &str) -> DataType {
    parse_data_type(ddl).unwrap_or_else(|e| panic!("{ddl}: {e}"))
}

#[test]
fn utf8_binary_is_the_plain_string() {
    assert_eq!(t("string collate utf8_binary"), DataType::String);
    assert_eq!(
        t("STRING COLLATE SYSTEM.BUILTIN.UTF8_BINARY"),
        DataType::String
    );
    assert_ne!(t("string collate utf8_binary_rtrim"), DataType::String);
}

#[test]
fn renderings_are_spark_s() {
    let x = t("string collate utf8_lcase");
    assert_eq!(x.simple_string(), "string collate UTF8_LCASE");
    assert_eq!(x.catalog_string(), "string collate UTF8_LCASE");
    assert_eq!(x.sql(), "STRING COLLATE UTF8_LCASE");
    assert_eq!(x.type_name(), "string collate UTF8_LCASE");
    assert_eq!(
        t("string collate SR_ci_ai").sql(),
        "STRING COLLATE sr_CI_AI"
    );
    assert_eq!(
        t("struct<a: string collate unicode_ci, b: array<string collate utf8_lcase>>")
            .simple_string(),
        "struct<a:string collate UNICODE_CI,b:array<string collate UTF8_LCASE>>"
    );
}

#[test]
fn an_unknown_collation_is_an_error_with_proposals() {
    let e = parse_data_type("string collate utf8_lcas").unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::CollationInvalidName);
    assert!(parse_data_type("string collate utf8_lcase extra").is_err());
}

#[test]
fn the_arrow_field_tags_the_collation_and_reads_it_back() {
    for ddl in [
        "string collate utf8_lcase",
        "string collate unicode_ci_ai",
        "array<string collate utf8_binary_rtrim>",
    ] {
        let ty = t(ddl);
        let field = ty.to_arrow_field("c", true);
        assert_eq!(DataType::from_arrow_field(&field), Ok(ty.clone()), "{ddl}");
        assert!(
            !field.metadata().is_empty() || ty.nesting_depth() > 0,
            "{ddl}"
        );
    }
    // Without the tag the carrier is a plain string.
    let plain = t("string collate utf8_lcase").to_arrow();
    assert_eq!(DataType::from_arrow(&plain), Ok(DataType::String));
}

#[test]
fn json_matches_pyspark() {
    let spark = |s: &str| serde_json::from_str::<Value>(s).unwrap();
    assert_eq!(
        spark(&t("string collate utf8_lcase").json()),
        spark(r#""string collate UTF8_LCASE""#)
    );
    assert_eq!(
        spark(&t("array<string collate utf8_lcase>").json()),
        spark(r#"{"containsNull":true,"elementType":"string collate UTF8_LCASE","type":"array"}"#)
    );
    let ddl = "struct<a: string collate utf8_lcase, n: array<string collate utf8_lcase>, \
               m: map<string, string collate sr_ci_ai_rtrim>, \
               s: struct<x: string collate utf8_binary_rtrim>>";
    let expected = r#"{"fields":[{"metadata":{"__COLLATIONS":{"a":"spark.UTF8_LCASE"}},"name":"a","nullable":true,"type":"string"},{"metadata":{"__COLLATIONS":{"n.element":"spark.UTF8_LCASE"}},"name":"n","nullable":true,"type":{"containsNull":true,"elementType":"string","type":"array"}},{"metadata":{"__COLLATIONS":{"m.value":"icu.sr_CI_AI_RTRIM"}},"name":"m","nullable":true,"type":{"keyType":"string","type":"map","valueContainsNull":true,"valueType":"string"}},{"metadata":{},"name":"s","nullable":true,"type":{"fields":[{"metadata":{"__COLLATIONS":{"x":"spark.UTF8_BINARY_RTRIM"}},"name":"x","nullable":true,"type":"string"}],"type":"struct"}}],"type":"struct"}"#;
    let ty = t(ddl);
    assert_eq!(spark(&ty.json()), spark(expected));
    // And back.
    assert_eq!(DataType::from_json(expected), Ok(ty));
    assert_eq!(
        DataType::from_json(r#""string collate UNICODE_CI""#),
        Ok(DataType::CollatedString(
            Collation::parse("unicode_ci").unwrap()
        ))
    );
}

#[test]
fn json_refuses_a_bad_provider_and_a_collation_on_a_non_string() {
    let field = |ty: &str, meta: &str| {
        format!(
            r#"{{"type":"struct","fields":[{{"name":"a","type":{ty},"nullable":true,"metadata":{meta}}}]}}"#
        )
    };
    let e = DataType::from_json(&field(
        r#""string""#,
        r#"{"__COLLATIONS":{"a":"zzz.UTF8_LCASE"}}"#,
    ))
    .unwrap_err();
    assert_eq!(e.error_class(), ErrorClass::CollationInvalidProvider);
    let e = DataType::from_json(&field(
        r#""integer""#,
        r#"{"__COLLATIONS":{"a":"spark.UTF8_LCASE"}}"#,
    ))
    .unwrap_err();
    assert_eq!(
        e.error_class().name(),
        "INVALID_JSON_DATA_TYPE_FOR_COLLATIONS"
    );
}

#[test]
fn user_facing_entry_points_can_refuse_a_collation_until_the_analyzer_takes_it() {
    for ddl in [
        "string collate utf8_lcase",
        "array<string collate unicode_ci>",
        "map<int, struct<a: string collate utf8_lcase>>",
    ] {
        let e = t(ddl).refuse_collations().unwrap_err();
        assert_eq!(
            e.error_class(),
            ErrorClass::KrishivUnsupportedFeature,
            "{ddl}"
        );
    }
    assert_eq!(
        t("array<string>").refuse_collations(),
        Ok(t("array<string>"))
    );
}
