// SPDX-License-Identifier: Apache-2.0
//! `CREATE VIEW ... WITH METRICS` (P3 4.2-MV) against the metric-view goldens recorded from Spark
//! 4.2.1 and 4.3.0 (`tests/fixtures/spark-{4.2.1,4.3.0}/metric-views/cases.json`): every create
//! statement parses as Spark's does, and the errors Spark raises while parsing match in class,
//! parameters and position.

use super::*;
use crate::ast::CreateMetricView;

/// Error classes Spark's parser raises for a metric-view statement (the rest come later).
const PARSER_CLASSES: &[&str] = &[
    "PARSE_SYNTAX_ERROR",
    "DUPLICATE_CLAUSES",
    "CREATE_VIEW_WITH_IF_NOT_EXISTS_AND_REPLACE",
    "_LEGACY_ERROR_TEMP_0035",
];

/// A `ParseException`'s `(line 1, pos N)`: the character offset of the offending token.
fn parse_position(message: &str) -> Option<usize> {
    let rest = message.split("(line 1, pos ").nth(1)?;
    rest.split(')').next()?.parse().ok()
}

fn metric(sql: &str) -> CreateMetricView {
    match parse_statement(sql, &ParserOptions::default()) {
        Ok(Statement::CreateMetricView(v)) => *v,
        other => panic!("{sql}: {other:?}"),
    }
}

#[test]
fn create_statements_agree_with_spark() {
    let mut checked = 0;
    let mut failures = vec![];
    for version in ["4.2.1", "4.3.0"] {
        let path = format!(
            "{}/../../tests/fixtures/spark-{version}/metric-views/cases.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        for case in doc["cases"].as_array().unwrap() {
            if case["group"] == "gate" {
                continue;
            }
            for step in case["steps"].as_array().unwrap() {
                let sql = step["sql"].as_str().unwrap_or_default();
                let upper = sql.to_ascii_uppercase();
                if !(upper.starts_with("CREATE") && upper.contains("VIEW"))
                    || !(upper.contains("METRICS") || sql.contains('$'))
                {
                    continue;
                }
                checked += 1;
                let ours = parse_statement(sql, &ParserOptions::default());
                let error = &step["outcome"]["error"];
                let class = error["condition"].as_str().unwrap_or_default();
                // A syntax error without a `(line 1, pos N)` is in a YAML expression: the session
                // parses those when it creates the view.
                let message = error["message"].as_str().unwrap_or_default();
                let statement_error = PARSER_CLASSES.contains(&class)
                    && (class != "PARSE_SYNTAX_ERROR" || parse_position(message).is_some());
                let agrees = if statement_error {
                    ours.as_ref().is_err_and(|e| {
                        let params: serde_json::Map<String, serde_json::Value> = e
                            .error
                            .message_parameters()
                            .iter()
                            .map(|(k, v)| ((*k).to_owned(), v.clone().into()))
                            .collect();
                        let at = sql.get(..e.span.start).map(|p| p.chars().count());
                        let position =
                            parse_position(error["message"].as_str().unwrap()).unwrap_or(0);
                        e.error.error_class().name() == class
                            && Some(&params) == error["parameters"].as_object()
                            && at == Some(position)
                    })
                } else {
                    // Spark's parser took it; anything it raised came later.
                    matches!(ours, Ok(Statement::CreateMetricView(_)))
                };
                if !agrees {
                    failures.push(format!(
                        "{version} {}: {sql:?}\n  spark: {class} {}\n  ours: {:?}",
                        case["name"],
                        error["message"],
                        ours.as_ref()
                            .map(drop)
                            .map_err(|e| (e.error.to_string(), e.span)),
                    ));
                }
            }
        }
    }
    assert!(checked > 300, "only {checked} create statements");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_tree_records_the_clauses_as_written() {
    let v = metric(
        "CREATE OR REPLACE VIEW db.mv (a COMMENT 'x', b) COMMENT 'c' WITH METRICS \
         TBLPROPERTIES ('k' = 'v') LANGUAGE yaml AS $t$version: 0.1$t$",
    );
    assert!(v.or_replace && !v.if_not_exists);
    assert_eq!(v.columns.len(), 2);
    assert_eq!(v.comment.as_deref(), Some("c"));
    assert_eq!(v.tblproperties.len(), 1);
    assert_eq!(v.language.as_ref().map(|l| l.value.as_str()), Some("yaml"));
    assert!(v.with_metrics.is_some());
    assert_eq!(v.duplicate_clause, None);
    assert_eq!(v.body, "version: 0.1");
}

#[test]
fn missing_clauses_are_left_to_the_session() {
    // Spark 4.2 and 4.3 raise different errors here; the parser records what is missing.
    let v = metric("CREATE VIEW mv AS $$x$$");
    assert!(v.language.is_none() && v.with_metrics.is_none());
    let v = metric("CREATE VIEW mv WITH METRICS AS $$x$$");
    assert!(v.language.is_none() && v.with_metrics.is_some());
}

#[test]
fn language_names_follow_the_routine_language_rule() {
    let opts = ParserOptions::default();
    // `SQL` is the one keyword taken; it is reported by its constant.
    let e =
        parse_statement("CREATE VIEW mv WITH METRICS LANGUAGE sql AS $$x$$", &opts).unwrap_err();
    assert!(
        e.error.to_string().contains("metric view: SQL"),
        "{}",
        e.error
    );
    // Another keyword is no `IDENTIFIER` token.
    let e = parse_statement(
        "CREATE VIEW mv WITH METRICS LANGUAGE METRICS AS $$x$$",
        &opts,
    )
    .unwrap_err();
    assert_eq!(e.error.error_class().name(), "PARSE_SYNTAX_ERROR");
    // A plain view keeps parsing as one, and a temporary view takes no metric clause.
    assert!(matches!(
        parse_statement("CREATE VIEW v COMMENT 'c' AS SELECT 1", &opts),
        Ok(Statement::CreatePermanentView(_))
    ));
}
