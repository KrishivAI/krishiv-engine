// SPDX-License-Identifier: Apache-2.0
//! Tests for error classes: SQLSTATE and name lookup, sub-classes, message templates and parameters, and the generated docs.
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

const SPARK_4_3_JSON: &str = include_str!("../../vendored/spark-error-conditions.json");
const SPARK_4_2_JSON: &str = include_str!("../../vendored/spark-error-conditions-4.2.1.json");
const RENDER_FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/spark-4.2.0/error-classes/render.json"
));

fn json(text: &str) -> serde_json::Value {
    serde_json::from_str(text).expect("valid JSON")
}

const DELTA_JSON: &str = include_str!("../../vendored/delta-error-classes.json");

/// P3-T021: the registry holds exactly the classes of the vendored Delta subset
/// (delta-spark 4.4.1), each with that file's SQLSTATE and template (a sub-class
/// joins its parent's text with a space, as for Spark's classes).
#[test]
fn delta_classes_equal_the_vendored_subset() {
    let doc = json(DELTA_JSON);
    let doc = doc.as_object().expect("object");
    let mut want = Vec::new();
    for (name, info) in doc {
        let lines = |v: &serde_json::Value| -> String {
            v["message"]
                .as_array()
                .expect("message lines")
                .iter()
                .map(|l| l.as_str().expect("line"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let main = lines(info);
        let state = info["sqlState"]
            .as_str()
            .expect("Delta classes have a SQLSTATE");
        match info.get("subClass").and_then(|s| s.as_object()) {
            None => want.push((name.clone(), state, main)),
            Some(subs) => {
                for (sub, sub_info) in subs {
                    want.push((
                        format!("{name}.{sub}"),
                        state,
                        format!("{main} {}", lines(sub_info)),
                    ));
                }
            }
        }
    }
    let got: Vec<&ErrorClass> = ErrorClass::ALL
        .iter()
        .filter(|c| c.name().starts_with("DELTA_"))
        .collect();
    assert_eq!(got.len(), want.len());
    for (name, state, template) in &want {
        let c = ErrorClass::from_name(name).expect("registered");
        assert_eq!(c.sql_state(), Some(*state), "{name}");
        assert_eq!(c.message_template(), template, "{name}");
    }
    // The classes the Delta connector names (design note P3-T021 sections 2.2, 3.4, 3.7).
    for (name, state) in [
        ("DELTA_INVALID_PROTOCOL_VERSION", "KD004"),
        ("DELTA_UNSUPPORTED_FEATURES_FOR_READ", "56038"),
        ("DELTA_VERSION_NOT_FOUND", "22003"),
        ("DELTA_TIMESTAMP_GREATER_THAN_COMMIT", "42816"),
        ("DELTA_CONCURRENT_APPEND.WITH_PARTITION_HINT", "2D521"),
        ("DELTA_MAX_COMMIT_RETRIES_EXCEEDED", "40000"),
    ] {
        assert_eq!(
            ErrorClass::from_name(name).and_then(ErrorClass::sql_state),
            Some(state),
            "{name}"
        );
    }
    let e: KrishivError = DeltaVersionNotFound {
        user_version: "7".to_owned(),
        earliest: "0".to_owned(),
        latest: "3".to_owned(),
    }
    .into();
    assert_eq!(
        e.to_string(),
        "[DELTA_VERSION_NOT_FOUND] Cannot time travel Delta table to version 7. Available versions: [0, 3]. SQLSTATE: 22003"
    );
}

/// P3-T021: the catalog parse-limit class names the limit, its value and the fix.
#[test]
fn catalog_limit_exceeded_names_the_limit() {
    let e: KrishivError = KrishivCatalogLimitExceeded {
        source: "Delta log file t/_delta_log/0.json".to_owned(),
        limit: "the size limit".to_owned(),
        value: "268435456 bytes".to_owned(),
        config: "krishiv.catalog.maxMetadataBytes".to_owned(),
    }
    .into();
    assert_eq!(e.error_class(), ErrorClass::KrishivCatalogLimitExceeded);
    assert_eq!(e.error_class().sql_state(), Some("54000"));
    assert!(
        e.to_string().contains("krishiv.catalog.maxMetadataBytes"),
        "{e}"
    );
}

/// One release's leaf classes: name -> (SQLSTATE, template), as Spark joins a
/// main message and its sub-class message.
fn release_leaves(
    doc: &serde_json::Value,
) -> std::collections::BTreeMap<String, (Option<String>, String)> {
    let join = |v: &serde_json::Value| -> String {
        v["message"]
            .as_array()
            .expect("message array")
            .iter()
            .map(|l| l.as_str().expect("line"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let mut out = std::collections::BTreeMap::new();
    for (name, info) in doc.as_object().expect("object") {
        let state = info
            .get("sqlState")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        match info.get("subClass").and_then(|s| s.as_object()) {
            None => {
                out.insert(name.clone(), (state, join(info)));
            }
            Some(subs) => {
                for (sub, sub_info) in subs {
                    out.insert(
                        format!("{name}.{sub}"),
                        (state.clone(), format!("{} {}", join(info), join(sub_info))),
                    );
                }
            }
        }
    }
    out
}

/// PR-64: the Spark classes are exactly the union of 4.2.1's and 4.3.0's, each with the SQLSTATE the JSON
/// declares, and the generator's 4.2.0 count (from the live container) plus 4.2.1's two added classes is
/// 4.2.1's count (`PARQUET_FIELD_REQUESTED_MORE_THAN_ONCE`, `WINDOW_FUNCTION_PARTITION_SIZE_EXCEEDS_LIMIT`).
#[test]
fn registry_is_the_union_of_spark_4_2_1_and_4_3_0() {
    let v42 = release_leaves(&json(SPARK_4_2_JSON));
    let v43 = release_leaves(&json(SPARK_4_3_JSON));
    let union: BTreeSet<&String> = v42.keys().chain(v43.keys()).collect();
    let spark: BTreeSet<String> = ErrorClass::ALL
        .iter()
        .filter(|c| !c.name().starts_with("KRISHIV_") && !c.name().starts_with("DELTA_"))
        .map(|c| c.name().to_owned())
        .collect();
    assert_eq!(spark.iter().collect::<BTreeSet<_>>(), union);
    assert_eq!(v42.len(), 1931);
    assert_eq!(v43.len(), 2001);
    let fixture = json(RENDER_FIXTURE);
    assert_eq!(fixture["leaf_classes"].as_u64(), Some(v42.len() as u64 - 2));
    for name in &spark {
        let c = ErrorClass::from_name(name).expect("registered");
        let want = v42
            .get(name)
            .or_else(|| v43.get(name))
            .map(|(s, _)| s.as_deref());
        assert_eq!(Some(c.sql_state()), want, "{name}");
    }
}

/// PR-64: each release's own text and parameters, `None` where the class is not in that release, and the
/// default view (`message_template`) is 4.2's, or 4.3's for a class only 4.3.0 has.
#[test]
fn per_release_templates_match_the_vendored_registries() {
    let v42 = release_leaves(&json(SPARK_4_2_JSON));
    let v43 = release_leaves(&json(SPARK_4_3_JSON));
    for c in ErrorClass::ALL
        .iter()
        .filter(|c| !c.name().starts_with("KRISHIV_") && !c.name().starts_with("DELTA_"))
    {
        let name = c.name().to_owned();
        let t42 = v42.get(&name).map(|(_, t)| t.as_str());
        let t43 = v43.get(&name).map(|(_, t)| t.as_str());
        assert_eq!(c.message_template_for(SparkRelease::V4_2), t42, "{name}");
        assert_eq!(c.message_template_for(SparkRelease::V4_3), t43, "{name}");
        assert_eq!(
            c.message_template(),
            t42.or(t43).expect("in a release"),
            "{name}"
        );
        for (release, t) in [(SparkRelease::V4_2, t42), (SparkRelease::V4_3, t43)] {
            let want = t.map(template::placeholders);
            assert_eq!(
                c.parameters_for(release),
                want.as_deref(),
                "{name} {release:?}"
            );
        }
    }
    // Krishiv's own classes exist in every release, with one text.
    let k = ErrorClass::KrishivSpillCapExceeded;
    assert_eq!(
        k.message_template_for(SparkRelease::V4_2),
        Some(k.message_template())
    );
    assert_eq!(
        k.message_template_for(SparkRelease::V4_3),
        Some(k.message_template())
    );
}

/// PR-64: the differences the 4.3.0 migration brings, by name: a replaced message, a class added in 4.3.0, a
/// class only the 4.2 goldens raise, and a new parameter the default view does not have.
#[test]
fn spark_4_3_0_differences_are_keyed_by_release() {
    let p = ErrorClass::UnsupportedTimePrecision;
    assert!(
        p.message_template_for(SparkRelease::V4_2)
            .is_some_and(|t| t.contains("[0, 6]"))
    );
    assert!(
        p.message_template_for(SparkRelease::V4_3)
            .is_some_and(|t| t.contains("[0, 9]"))
    );
    let added = ErrorClass::from_name("INVALID_SQL_SYNTAX.INVALID_WINDOW_FRAME_BOUND")
        .expect("added in 4.3.0");
    assert_eq!(added.message_template_for(SparkRelease::V4_2), None);
    assert!(added.message_template_for(SparkRelease::V4_3).is_some());
    let dropped =
        ErrorClass::from_name("DATETIME_FIELD_OUT_OF_BOUNDS.WITH_SUGGESTION").expect("4.2 only");
    assert!(dropped.message_template_for(SparkRelease::V4_2).is_some());
    assert_eq!(dropped.message_template_for(SparkRelease::V4_3), None);
    // 4.3.0 names the operator; the default view (and so the 4.2 goldens' `messageParameters`) does not.
    let op = ErrorClass::UnsupportedExprForOperator;
    assert_eq!(op.parameters(), &["invalidExprSqls"]);
    assert_eq!(
        op.parameters_for(SparkRelease::V4_2),
        Some(&["invalidExprSqls"][..])
    );
    assert_eq!(
        op.parameters_for(SparkRelease::V4_3),
        Some(&["operator", "invalidExprSqls"][..])
    );
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

#[test]
fn every_legacy_class_omits_the_class_prefix() {
    // Spark 4.2.0: `new SparkException("_LEGACY_ERROR_USER_RAISED_EXCEPTION",
    // Map("errorClass" -> "X", "errorMessage" -> "boom"), null).getMessage`.
    let e = KrishivError::from(LegacyErrorUserRaisedException {
        error_message: "boom".to_owned(),
    });
    assert_eq!(e.to_string(), "boom SQLSTATE: P0001");
}

#[test]
fn cause_chain_like_spark() {
    use std::error::Error as _;
    let dup = KrishivError::from(LegacyErrorTemp2093 {
        required_field_name: "a".to_owned(),
        matched_orc_fields: "[a, A]".to_owned(),
    });
    let read = KrishivError::from(FailedReadFileNoHint {
        path: "file:/t/p.parquet".to_owned(),
    });
    assert_eq!(read.cause(), None);
    assert!(read.source().is_none());
    let e = read.clone().with_cause(dup.clone());
    assert_eq!(e.cause(), Some(&dup));
    assert_eq!(e.causes().collect::<Vec<_>>(), [&dup]);
    // Display is Spark's getMessage: the cause is not part of it.
    assert_eq!(e.to_string(), read.to_string());
    assert_ne!(e, read);
    let source = e.source().expect("source");
    assert_eq!(source.to_string(), dup.to_string());
    // Two levels; with_cause replaces the direct cause.
    let top = KrishivError::from(OperationCanceled {}).with_cause(e.clone());
    let chain: Vec<ErrorClass> = top.causes().map(KrishivError::error_class).collect();
    assert_eq!(
        chain,
        [
            ErrorClass::FailedReadFileNoHint,
            ErrorClass::LegacyErrorTemp2093
        ]
    );
    assert_eq!(top.clone().with_cause(dup).causes().count(), 1);
    // Redaction covers the causes.
    let r = top.redacted();
    assert!(r.causes().all(|c| {
        c.message_parameters()
            .iter()
            .all(|(_, v)| v == crate::secret::REDACTED)
    }));
    assert_eq!(r.causes().count(), 2);
}

#[test]
fn audience_comes_from_the_json_files() {
    assert_eq!(ErrorClass::DivideByZero.audience(), Audience::User);
    assert_eq!(ErrorClass::KrishivUnknownConfig.audience(), Audience::User);
    // Krishiv class marked in krishiv-error-classes.json.
    assert_eq!(
        ErrorClass::KrishivSpillIoError.audience(),
        Audience::Operator
    );
    assert_eq!(
        ErrorClass::KrishivObjectStoreRequestFailed.audience(),
        Audience::Operator
    );
    // Spark class named in error-audience.json, and its sub-classes.
    assert_eq!(ErrorClass::InternalError.audience(), Audience::Operator);
    assert!(
        ErrorClass::ALL
            .iter()
            .filter(|c| c.name().starts_with("FAILED_READ_FILE."))
            .all(|c| c.audience() == Audience::Operator)
    );
}

#[test]
fn for_client_keeps_user_classes_and_cuts_the_cause() {
    let cause = KrishivError::from(KrishivSpillIoError {
        operation: "write".into(),
        path: "/var/spill/x".into(),
        reason: "disk full".into(),
    });
    let e = KrishivError::from(DivideByZero {
        config: "\"spark.sql.ansi.enabled\"".into(),
    })
    .with_cause(cause);
    let c = e.for_client();
    assert_eq!(c.error().error_class(), ErrorClass::DivideByZero);
    assert_eq!(c.error().message(), e.message());
    assert!(c.error().cause().is_none());
    assert!(std::error::Error::source(c.error()).is_none());
    assert!(std::error::Error::source(&c).is_none());
    assert!(!c.error_id().is_empty());
    assert_eq!(c.to_string(), c.error().to_string());
    // The original is untouched.
    assert!(e.cause().is_some());
}

#[test]
fn for_client_redacts_operator_classes() {
    let e = KrishivError::from(KrishivSpillIoError {
        operation: "write".into(),
        path: "/var/spill/tenant-7/x".into(),
        reason: "disk full on host-12".into(),
    });
    let c = e.for_client();
    assert_eq!(c.error().error_class(), e.error_class());
    assert_eq!(c.error().sql_state(), e.sql_state());
    let shown = c.to_string();
    for secret in ["/var/spill", "tenant-7", "host-12", "disk full"] {
        assert!(!shown.contains(secret), "{shown}");
        assert!(!format!("{c:?}").contains(secret));
    }
    assert!(
        c.error()
            .message_parameters()
            .iter()
            .all(|(_, v)| v == crate::secret::REDACTED)
    );
    // The full form still has the detail.
    assert!(e.to_string().contains("/var/spill/tenant-7/x"));
}

#[test]
fn error_id_joins_the_full_and_client_forms() {
    let e = KrishivError::from(KrishivSpillIoError {
        operation: "read".into(),
        path: "p".into(),
        reason: "r".into(),
    });
    let id = new_error_id();
    assert_eq!(e.for_client_with_id(id.clone()).error_id(), id);
    // Fresh ids differ.
    let a = KrishivError::from(DivideByZero { config: "c".into() });
    assert_ne!(a.for_client().error_id(), a.for_client().error_id());
    assert_ne!(new_error_id(), new_error_id());
}

#[test]
fn for_client_keeps_query_contexts() {
    let e = KrishivError::from(DivideByZero { config: "c".into() })
        .with_query_context(ErrorQueryContext::sql("1/0", 0, 2));
    assert_eq!(e.for_client().error().query_contexts(), e.query_contexts());
}

#[test]
fn every_internal_error_class_is_operator_only() {
    let internal: Vec<_> = ErrorClass::ALL
        .iter()
        .filter(|c| c.name().starts_with("INTERNAL_ERROR"))
        .collect();
    assert!(internal.len() > 1);
    assert!(internal.iter().all(|c| c.audience() == Audience::Operator));
    for c in [
        ErrorClass::PathNotFound,
        ErrorClass::KrishivConcurrentCommit,
    ] {
        assert_eq!(c.audience(), Audience::Operator, "{}", c.name());
    }
}

#[test]
fn python_exception_renders_message_and_traceback() {
    // Spark 4.2.0 `PYTHON_EXCEPTION`: `<msg>: <traceback>`, SQLSTATE 38000.
    assert_eq!(ErrorClass::PythonException.name(), "PYTHON_EXCEPTION");
    let e = KrishivError::from(PythonException {
        msg: "An exception was thrown from the Python worker".to_owned(),
        traceback: "ValueError: boom".to_owned(),
    });
    assert_eq!(e.sql_state(), Some("38000"));
    assert!(
        e.to_string()
            .contains("An exception was thrown from the Python worker: ValueError: boom"),
        "{e}"
    );
}
