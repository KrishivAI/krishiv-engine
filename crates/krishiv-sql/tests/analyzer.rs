// SPDX-License-Identifier: Apache-2.0
//! SQL lowering and the analyzer against Spark 4.2.
//!
//! - Analyzed, they and a name-resolution corpus give Spark's output names
//!   and nullability, or Spark's error class, message parameters and query
//!   context (`tests/fixtures/spark-4.2.0/analyzer-goldens.json`, generated
//!   by `tests/scripts/gen_analyzer_goldens.py` from a real Spark run).
//!   Output types are left to type coercion (T071b). The resolved TPC-H
//!   plans are insta snapshots.
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]
#![expect(clippy::unwrap_in_result, reason = "a malformed fixture is a test bug")]

use std::path::Path;

use krishiv_plan::SourceId;
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    AnalysisError, AnalyzedPlan, Analyzer as _, DefaultAnalyzer, MemoryCatalog, QueryContext,
    ResolutionConf, TableColumn,
};
use krishiv_sql::{ParserOptions, parse_and_lower, redact_sql};
use serde_json::Value;

fn root() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

fn goldens() -> Value {
    let path = root().join("tests/fixtures/spark-4.2.0/analyzer-goldens.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn tpch(id: &str) -> String {
    let path = root().join(format!("bench/queries/tpch/{id}.sql"));
    let text = std::fs::read_to_string(path).unwrap();
    text.trim().trim_end_matches(';').to_owned()
}

/// The fixture's tables in `spark_catalog.default`.
fn catalog(g: &Value) -> MemoryCatalog {
    let mut c = MemoryCatalog::new("spark_catalog");
    for (name, cols) in g["tables"].as_object().unwrap() {
        let columns = cols
            .as_array()
            .unwrap()
            .iter()
            .map(|f| TableColumn {
                name: f["name"].as_str().unwrap().to_owned(),
                data_type: krishiv_types::parse_data_type(f["type"].as_str().unwrap()).unwrap(),
                nullable: f["nullable"].as_bool().unwrap(),
            })
            .collect();
        c = c.with_table(&["default"], name, columns, None);
    }
    c
}

fn conf(c: &Value) -> ResolutionConf {
    let map = c.as_object().cloned().unwrap_or_default();
    ResolutionConf::from_conf(|k| map.get(k).and_then(Value::as_str))
}

fn analyze(g: &Value, sql: &str, conf: ResolutionConf) -> Result<AnalyzedPlan, AnalysisError> {
    let plan = parse_and_lower(sql, &ParserOptions::default(), SourceId(0)).map_err(|e| {
        AnalysisError {
            error: e.error,
            origin: krishiv_plan::Origin::UNKNOWN,
        }
    })?;
    block_on(DefaultAnalyzer.analyze(&plan, &context(catalog(g), conf)))
}

/// Mismatches between `got` and Spark's `want`, as text.
fn compare(sql: &str, got: &Result<AnalyzedPlan, AnalysisError>, want: &Value) -> Option<String> {
    let text = redact_sql(sql);
    match (got, want["ok"].as_bool().unwrap()) {
        (Ok(a), true) => {
            let got: Vec<(String, bool)> = a
                .output
                .iter()
                .map(|c| (c.name.clone(), c.nullable))
                .collect();
            let want: Vec<(String, bool)> = want["output"]
                .as_array()
                .unwrap()
                .iter()
                .map(|f| {
                    (
                        f["name"].as_str().unwrap().to_owned(),
                        f["nullable"].as_bool().unwrap(),
                    )
                })
                .collect();
            (got != want).then(|| format!("output {got:?}, Spark {want:?}"))
        }
        (Err(e), false) => {
            let class = e.error.error_class().name();
            let params: serde_json::Map<String, Value> = e
                .error
                .message_parameters()
                .iter()
                .map(|(k, v)| ((*k).to_owned(), Value::String(v.clone())))
                .collect();
            let mut diff = Vec::new();
            if class != want["error_class"] {
                diff.push(format!("class {class}, Spark {}", want["error_class"]));
            } else if Value::Object(params.clone()) != want["message_parameters"] {
                diff.push(format!(
                    "params {params:?}, Spark {}",
                    want["message_parameters"]
                ));
            }
            if let Some(ctx) = want.get("context").filter(|c| c.get("start").is_some()) {
                let got = e.query_context(&text);
                let got = match got {
                    Some(QueryContext::Sql {
                        start,
                        stop,
                        fragment,
                    }) => Some((start as u64, stop as u64, fragment)),
                    _ => None,
                };
                let want = (
                    ctx["start"].as_u64().unwrap(),
                    ctx["stop"].as_u64().unwrap(),
                    ctx["fragment"].as_str().unwrap().to_owned(),
                );
                if got.as_ref() != Some(&want) {
                    diff.push(format!("context {got:?}, Spark {want:?}"));
                }
            }
            (!diff.is_empty()).then(|| diff.join("; "))
        }
        (Ok(a), false) => Some(format!(
            "resolved to {:?}, Spark raised {}",
            a.output.iter().map(|c| &c.name).collect::<Vec<_>>(),
            want["error_class"]
        )),
        (Err(e), true) => Some(format!("failed with {e}, Spark resolved it")),
    }
}

#[test]
fn tpch_resolves_as_spark_does() {
    let g = goldens();
    let mut failures = Vec::new();
    for case in g["tpch"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let got = analyze(&g, &tpch(id), ResolutionConf::default());
        if let Some(diff) = compare(&tpch(id), &got, case) {
            failures.push(format!("{id}: {diff}"));
            continue;
        }
        let a = got.unwrap();
        assert!(a.plan.root().is_resolved(), "{id}");
        assert_eq!(
            a.plan.root().resolved_output(),
            Some(a.output.clone()),
            "{id}"
        );
        insta::assert_snapshot!(
            format!("tpch_{}", id.to_lowercase()),
            a.plan.root().to_string()
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn resolution_corpus_matches_spark() {
    let g = goldens();
    let mut failures = Vec::new();
    let mut raised = std::collections::BTreeSet::new();
    for case in g["corpus"].as_array().unwrap() {
        let sql = case["sql"].as_str().unwrap();
        let conf = conf(&case["conf"]);
        let got = analyze(&g, sql, conf);
        if let Some(diff) = compare(sql, &got, case) {
            failures.push(format!("{sql} {}: {diff}", case["conf"]));
        }
        if let Err(e) = &got {
            raised.insert(e.error.error_class().name());
        }
        // A resolved plan states its output and resolves to itself.
        if let Ok(a) = got {
            let ctx = context(catalog(&g), conf);
            let again = block_on(DefaultAnalyzer.analyze(&a.plan, &ctx)).unwrap();
            if !again.plan.structurally_eq(&a.plan)
                || a.plan.root().resolved_output() != Some(a.output)
            {
                failures.push(format!("{sql}: not a fixed point\n{}", again.plan.root()));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    // Classes of lateral aliases and ordinals, each matched against Spark
    // above.
    for class in [
        "AMBIGUOUS_LATERAL_COLUMN_ALIAS",
        "GROUP_BY_POS_AGGREGATE",
        "GROUP_BY_POS_OUT_OF_RANGE",
        "ORDER_BY_POS_OUT_OF_RANGE",
        "STAR_GROUP_BY_POS",
        "UNSUPPORTED_FEATURE.LATERAL_COLUMN_ALIAS_IN_AGGREGATE_FUNC",
        "UNSUPPORTED_FEATURE.LATERAL_COLUMN_ALIAS_IN_GROUP_BY",
    ] {
        assert!(raised.contains(class), "{class} not raised");
    }
}

/// SQL text inside a plan resolves through the front end, which keeps it
/// redacted; its parse errors keep their span and never echo a secret.
#[test]
fn sql_text_in_plans_goes_through_the_front_end() {
    use std::sync::Arc;

    use krishiv_plan::analyzer::SqlFrontend as _;
    use krishiv_plan::{LogicalPlan, Plan, PlanNode, SqlArgs};
    use krishiv_sql::Frontend;

    let g = goldens();
    let mut ctx = context(catalog(&g), ResolutionConf::default());
    ctx.sql = Some(Arc::new(Frontend::default()));
    let sql = |q: &str| {
        Plan::try_new(LogicalPlan::new(PlanNode::Sql {
            query: q.to_owned(),
            args: SqlArgs::None,
        }))
        .unwrap()
    };
    let text = "SELECT a FROM t WITH ('password' = 'Canary7f3a9c')";
    let a = block_on(DefaultAnalyzer.analyze(&sql(text), &ctx)).unwrap();
    assert_eq!(a.output[0].name, "a");
    let shown = a.query_text.unwrap().to_string();
    assert!(!shown.contains("Canary7f3a9c"), "{shown}");
    assert_eq!(a.plan.root().resolved_output(), Some(a.output));
    let e = block_on(DefaultAnalyzer.analyze(
        &sql("SELECT a FROM t WITH (password = Canary7f3a9c x)"),
        &ctx,
    ))
    .unwrap_err();
    assert!(e.origin.sql.is_some());
    assert!(!format!("{e} {e:?}").contains("Canary7f3a9c"));
    let ansi = Frontend {
        options: ParserOptions {
            enforce_reserved_keywords: true,
            ..ParserOptions::default()
        },
    };
    assert_ne!(ansi.fingerprint(), Frontend::default().fingerprint());
}
