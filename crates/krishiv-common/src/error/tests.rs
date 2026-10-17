// SPDX-License-Identifier: Apache-2.0
use std::collections::BTreeSet;

use super::template;
use super::*;

#[test]
fn sqlstate_and_name_lookup() {
    assert_eq!(ErrorClass::DivideByZero.name(), "DIVIDE_BY_ZERO");
    assert_eq!(ErrorClass::DivideByZero.sql_state(), Some("22012"));
    assert_eq!(ErrorClass::ArithmeticOverflow.sql_state(), Some("22003"));
    assert_eq!(ErrorClass::CastInvalidInput.sql_state(), Some("22018"));
    assert_eq!(ErrorClass::TableOrViewNotFound.sql_state(), Some("42P01"));
    assert_eq!(ErrorClass::ParseSyntaxError.sql_state(), Some("42601"));
    assert_eq!(ErrorClass::InternalError.sql_state(), Some("XX000"));
    assert_eq!(
        ErrorClass::from_name("DIVIDE_BY_ZERO"),
        Some(ErrorClass::DivideByZero)
    );
    assert_eq!(ErrorClass::from_name("NO_SUCH_CLASS"), None);
}

#[test]
fn sub_classes_inherit_sqlstate_and_use_dotted_names() {
    let c = ErrorClass::UnresolvedColumnWithSuggestion;
    assert_eq!(c.name(), "UNRESOLVED_COLUMN.WITH_SUGGESTION");
    assert_eq!(c.sql_state(), Some("42703"));
    assert_eq!(
        ErrorClass::from_name("UNRESOLVED_COLUMN.WITHOUT_SUGGESTION"),
        Some(ErrorClass::UnresolvedColumnWithoutSuggestion)
    );
    // A class with sub-classes is only usable through a sub-class (as in Spark).
    assert_eq!(ErrorClass::from_name("UNRESOLVED_COLUMN"), None);
}

#[test]
fn renders_message_and_spark_display_format() {
    let e = KrishivError::from(DivideByZero {
        config: "\"spark.sql.ansi.enabled\"".to_owned(),
    });
    assert_eq!(e.error_class(), ErrorClass::DivideByZero);
    assert_eq!(e.sql_state(), Some("22012"));
    assert_eq!(
        e.message(),
        "Division by zero. Use `try_divide` to tolerate divisor being 0 and return NULL \
         instead. If necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error."
    );
    assert_eq!(
        e.to_string(),
        format!("[DIVIDE_BY_ZERO] {} SQLSTATE: 22012", e.message())
    );
}

#[test]
fn sub_class_template_is_main_plus_space_plus_sub() {
    let e = KrishivError::from(UnresolvedColumnWithSuggestion {
        object_name: "`x`".to_owned(),
        proposal: "`a`, `b`".to_owned(),
    });
    assert_eq!(
        e.message(),
        "A column, variable, or function parameter with name `x` cannot be resolved. \
         Did you mean one of the following? [`a`, `b`]."
    );
    let e = KrishivError::from(UnresolvedColumnWithoutSuggestion {
        object_name: "`x`".to_owned(),
    });
    // Spark joins with a space even when the sub-class message is empty.
    assert!(e.message().ends_with("cannot be resolved. "));
}

#[test]
fn multi_line_messages_join_with_newline() {
    let t = ErrorClass::TableOrViewNotFound.message_template();
    assert_eq!(t.lines().count(), 4);
    assert!(t.starts_with("The table or view <relationName> cannot be found."));
}

#[test]
fn message_parameters_use_spark_names_in_template_order() {
    let e = KrishivError::from(CastInvalidInput {
        expression: "'abc'".to_owned(),
        source_type: "\"STRING\"".to_owned(),
        target_type: "\"INT\"".to_owned(),
    });
    let names: Vec<&str> = e.message_parameters().iter().map(|(k, _)| *k).collect();
    assert_eq!(names, ["expression", "sourceType", "targetType"]);
    assert_eq!(
        ErrorClass::CastInvalidInput.parameters(),
        ["expression", "sourceType", "targetType"]
    );
}

#[test]
fn parameter_values_are_not_resubstituted() {
    let e = KrishivError::from(ParseSyntaxError {
        error: "'<hint>'".to_owned(),
        hint: "".to_owned(),
    });
    assert_eq!(e.message(), "Syntax error at or near '<hint>'.");
}

#[test]
fn krishiv_specific_classes_are_prefixed() {
    let e = KrishivError::from(KrishivUnsupportedFeature {
        feature: "RDD API".to_owned(),
    });
    assert_eq!(e.error_class().name(), "KRISHIV_UNSUPPORTED_FEATURE");
    assert_eq!(e.sql_state(), Some("0A000"));
}

/// Test-time guard on the registry: declared parameters are exactly the
/// placeholders of each template, and every class renders completely.
#[test]
fn registry_is_consistent() {
    let mut names = BTreeSet::new();
    for &c in ErrorClass::ALL {
        assert!(names.insert(c.name()), "duplicate class {}", c.name());
        assert_eq!(ErrorClass::from_name(c.name()), Some(c));
        if let Some(s) = c.sql_state() {
            assert!(
                s.len() == 5
                    && s.bytes()
                        .all(|b| b.is_ascii_digit() || b.is_ascii_uppercase()),
                "bad SQLSTATE {s} for {}",
                c.name()
            );
        } else {
            assert!(
                !c.name().starts_with("KRISHIV_"),
                "Krishiv class {} needs a SQLSTATE",
                c.name()
            );
        }
        assert_eq!(
            c.parameters(),
            template::placeholders(c.message_template()).as_slice(),
            "parameters out of sync for {}",
            c.name()
        );
        let params = c
            .parameters()
            .iter()
            .map(|p| (*p, format!("v_{p}")))
            .collect();
        let e = KrishivError::from_parts(c, params);
        assert!(
            template::placeholders(e.message()).is_empty(),
            "unrendered placeholder in {}",
            c.name()
        );
    }
    assert!(names.len() >= 10);
}

#[test]
fn placeholder_scanner_matches_spark_pattern() {
    assert_eq!(
        template::placeholders("a <x> b <y-z_1> <x> <> < bad> <=3 <ok>"),
        ["x", "y-z_1", "ok"]
    );
}

#[test]
fn render_leaves_unknown_placeholders_and_is_single_pass() {
    let out = template::render("<a>-<b>-<a>", &[("a", "<b>".to_owned())]);
    assert_eq!(out, "<b>-<b>-<b>");
}

#[test]
fn docs_markdown_lists_every_class_with_escaped_templates() {
    let md = docs_markdown();
    assert!(md.starts_with("# Krishiv error classes\n"));
    for &c in ErrorClass::ALL {
        assert!(md.contains(&format!(
            "| `{}` | `{}` |",
            c.name(),
            c.sql_state().unwrap_or("-")
        )));
    }
    // `<param>` must not be read as an HTML tag.
    assert!(md.contains("&lt;config&gt;"));
    assert!(!md.contains("<config>"));
    assert_eq!(md, docs_markdown());
}

const SPARK_JSON: &str = include_str!("../../vendored/spark-error-conditions.json");
const RENDER_FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/spark-4.2.0/error-classes/render.json"
));

fn json(text: &str) -> serde_json::Value {
    serde_json::from_str(text).expect("valid JSON")
}

/// P2-T003: the registry has exactly the classes of Spark 4.2.0's JSON.
#[test]
fn class_count_equals_spark_4_2_0() {
    let doc = json(SPARK_JSON);
    let doc = doc.as_object().expect("object");
    let leaves: usize = doc
        .values()
        .map(|v| {
            v.get("subClass")
                .and_then(|s| s.as_object())
                .map_or(1, serde_json::Map::len)
        })
        .sum();
    let spark: Vec<&ErrorClass> = ErrorClass::ALL
        .iter()
        .filter(|c| !c.name().starts_with("KRISHIV_"))
        .collect();
    assert_eq!(spark.len(), leaves);
    let top: BTreeSet<&str> = spark
        .iter()
        .map(|c| c.name().split('.').next().unwrap_or_default())
        .collect();
    assert_eq!(top.len(), doc.len());
    // The count recorded by the generator from the live container agrees.
    let fixture = json(RENDER_FIXTURE);
    assert_eq!(fixture["top_level_classes"], doc.len());
    assert_eq!(fixture["leaf_classes"], leaves);
    // Every class has the SQLSTATE the JSON declares (or none).
    for (name, info) in doc {
        let want = info.get("sqlState").and_then(|v| v.as_str());
        let subs = info.get("subClass").and_then(|s| s.as_object());
        let leaf_names: Vec<String> = subs.map_or_else(
            || vec![name.clone()],
            |s| s.keys().map(|k| format!("{name}.{k}")).collect(),
        );
        for leaf in leaf_names {
            let c = ErrorClass::from_name(&leaf).expect("registered");
            assert_eq!(c.sql_state(), want, "{leaf}");
        }
    }
}

/// P2-T003: 20 classes raised in a real Spark 4.2.0 render byte-identically.
#[test]
fn messages_are_byte_identical_to_spark_4_2_0() {
    let fixture = json(RENDER_FIXTURE);
    let cases = fixture["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 20);
    assert!(
        cases
            .iter()
            .any(|c| c["class"].as_str().is_some_and(|n| n.contains('.')))
    );
    assert!(
        cases
            .iter()
            .any(|c| c["parameters"].as_object().is_some_and(|p| p.len() >= 3))
    );
    for case in cases {
        let name = case["class"].as_str().expect("class");
        let class = ErrorClass::from_name(name).expect("registered class");
        let values = case["parameters"].as_object().expect("parameters");
        assert_eq!(class.parameters().len(), values.len(), "{name}");
        let params = class
            .parameters()
            .iter()
            .map(|p| (*p, values[*p].as_str().expect("string").to_owned()))
            .collect();
        let e = KrishivError::from_parts(class, params);
        assert_eq!(
            e.to_string(),
            case["message"].as_str().expect("message"),
            "{name}"
        );
        assert_eq!(e.sql_state(), case["sql_state"].as_str(), "{name}");
    }
}

#[test]
fn class_without_sqlstate_has_no_suffix() {
    let e = KrishivError::from(LegacyErrorTemp3092 {});
    assert_eq!(e.sql_state(), None);
    assert!(!e.to_string().contains("SQLSTATE"));
}
