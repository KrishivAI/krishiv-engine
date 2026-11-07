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
fn an_explicit_utf8_binary_is_a_display_only_marker() {
    // Written explicitly, it is the display-only marker (ADR 0043, update 2026-11-07): it prints
    // as Spark prints it and is no collation.
    for ddl in [
        "string collate utf8_binary",
        "STRING COLLATE SYSTEM.BUILTIN.UTF8_BINARY",
    ] {
        let ty = t(ddl);
        assert_eq!(ty.simple_string(), "string collate UTF8_BINARY");
        assert!(ty.is_string_type() && !ty.contains_collation() && !ty.is_collated_string());
        assert!(ty.is_binary_marker());
        // Spark's `StringType.equals` ignores whether the collation was written.
        assert_eq!(ty, DataType::String);
        assert_eq!(DataType::String, ty);
        assert_eq!(hash_of(&ty), hash_of(&DataType::String));
    }
    assert_eq!(t("string"), DataType::String);
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

fn hash_of(t: &DataType) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    t.hash(&mut h);
    h.finish()
}

#[test]
fn the_marker_equals_string_inside_nested_types_and_no_collation_does() {
    let marked = t("array<string collate utf8_binary>");
    let plain = t("array<string>");
    assert_eq!(marked, plain);
    assert_eq!(hash_of(&marked), hash_of(&plain));
    assert_eq!(
        t("map<string collate utf8_binary, struct<a: string>>"),
        t("map<string, struct<a: string collate utf8_binary>>")
    );
    assert_ne!(t("string collate utf8_lcase"), DataType::String);
    assert_ne!(
        t("string collate utf8_lcase"),
        t("string collate utf8_binary")
    );
    assert_ne!(t("array<string collate utf8_lcase>"), plain);
    // Every other type still equals itself and only itself.
    assert_eq!(DataType::Integer, DataType::Integer);
    assert_ne!(DataType::Integer, DataType::Long);
    assert_eq!(t("decimal(10,2)"), t("decimal(10,2)"));
    assert_ne!(t("decimal(10,2)"), t("decimal(10,3)"));
    for ty in [
        "void",
        "boolean",
        "binary",
        "date",
        "timestamp",
        "timestamp_ntz",
        "variant",
        "interval",
    ] {
        assert_eq!(t(ty), t(ty), "{ty}");
    }
}

#[test]
fn hashing_agrees_with_equality_for_every_payload_type() {
    let equal_pairs = [
        ("decimal(10,2)", "decimal(10,2)"),
        ("time(3)", "time(3)"),
        ("geometry(4326)", "geometry(4326)"),
        ("geography(4326)", "geography(4326)"),
        ("interval year to month", "interval year to month"),
        ("interval day to second", "interval day to second"),
        ("array<int>", "array<int>"),
        ("map<string, int>", "map<string collate utf8_binary, int>"),
        (
            "struct<a: int, b: string>",
            "struct<a: int, b: string collate utf8_binary>",
        ),
        ("string collate utf8_lcase", "string collate utf8_lcase"),
    ];
    for (a, b) in equal_pairs {
        let (x, y) = (t(a), t(b));
        assert_eq!(x, y, "{a} / {b}");
        assert_eq!(hash_of(&x), hash_of(&y), "{a} / {b}");
    }
    assert_ne!(t("time(3)"), t("time(4)"));
    assert_ne!(t("geometry(4326)"), t("geography(4326)"));
    assert_ne!(t("interval year to month"), t("interval day to second"));
}

#[test]
fn without_the_marker_a_type_is_the_plain_type_at_any_depth() {
    for (marked, plain) in [
        ("string collate utf8_binary", "string"),
        ("array<string collate utf8_binary>", "array<string>"),
        (
            "map<string collate utf8_binary, array<string collate utf8_binary>>",
            "map<string, array<string>>",
        ),
        (
            "struct<a: string collate utf8_binary, b: int>",
            "struct<a: string, b: int>",
        ),
        (
            "array<string collate utf8_lcase>",
            "array<string collate utf8_lcase>",
        ),
        ("int", "int"),
    ] {
        let stripped = t(marked).without_binary_marker();
        assert_eq!(
            stripped.simple_string(),
            t(plain).simple_string(),
            "{marked}"
        );
        assert!(
            !stripped.any_type(&krishiv_types::DataType::is_binary_marker),
            "{marked}"
        );
    }
}

#[test]
fn the_marker_accepts_the_storage_a_plain_string_does() {
    use arrow_schema::DataType as Arrow;
    let marker = t("string collate utf8_binary");
    for storage in [
        Arrow::Utf8,
        Arrow::LargeUtf8,
        Arrow::Utf8View,
        Arrow::BinaryView,
        Arrow::Binary,
    ] {
        assert!(
            marker
                .to_arrow_field_with_storage("c", storage.clone(), true)
                .is_ok(),
            "{storage}"
        );
    }
    let lcase = t("string collate utf8_lcase");
    assert!(
        lcase
            .to_arrow_field_with_storage("c", Arrow::Utf8, true)
            .is_ok()
    );
    assert!(
        lcase
            .to_arrow_field_with_storage("c", Arrow::BinaryView, true)
            .is_err()
    );
}
