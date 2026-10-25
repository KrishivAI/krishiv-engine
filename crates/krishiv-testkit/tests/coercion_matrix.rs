// SPDX-License-Identifier: Apache-2.0
//! The implicit-coercion matrix against Spark 4.2 (P2-T071d unit 1): for
//! every form (binary arithmetic, comparisons, `IN`, `CASE`, `IF`,
//! `COALESCE`, `NVL`, `NULLIF`, `greatest`, `least`, `concat`, `elt`,
//! `array(...)`, `map(...)` and `UNION ALL` columns) and every pair of
//! operand types in both ANSI modes, the analyzer must resolve the
//! expression to Spark's result type, or fail with Spark's analysis error
//! class and message parameters. The ground truth is
//! `tests/fixtures/spark-4.2.0/coercion-matrix.json` (generated from a real
//! Spark 4.2 run by `tests/scripts/gen_coercion_matrix.py`); `typeof` is
//! analysis-transparent, so the test resolves the inner expression (the
//! `UNION ALL` query itself for `union` rows). The rows the analyzer cannot
//! express yet (types or functions the engine lacks, or a rendering it does
//! not match) are listed, each with a reason, in
//! `tests/fixtures/analysis/coercion-matrix-manifest.json`; the list must
//! match the failures and their reasons exactly, so a fix shrinks it and a
//! regression fails. Rows need the engine (P2-T074): this checks analysis
//! only.
//!
//! `KRISHIV_UPDATE_MANIFEST=1` rewrites the manifest from the failures.
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::collections::BTreeMap;

use krishiv_common::error::KrishivError;
use krishiv_plan::SourceId;
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf};
use krishiv_sql::{ParserOptions, parse_and_lower};
use serde_json::{Value, json};

const FIXTURE: &str = "fixtures/spark-4.2.0/coercion-matrix.json";
const MANIFEST: &str = "fixtures/analysis/coercion-matrix-manifest.json";

/// One matrix row: what Spark resolved the query to.
struct Row {
    id: String,
    sql: String,
    query: String,
    ansi: bool,
    want_type: Option<String>,
    want_class: Option<String>,
    want_params: BTreeMap<String, String>,
}

/// The fixture rows: `SELECT <expr>` for expression rows (the `UNION ALL`
/// query itself for `union` rows).
fn rows(root: &std::path::Path) -> Vec<Row> {
    let doc: Value =
        serde_json::from_str(&std::fs::read_to_string(root.join(FIXTURE)).unwrap()).unwrap();
    assert_eq!(doc["format_version"], 1);
    assert_eq!(doc["spark_version"], "4.2.0");
    doc["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let text = |k: &str| r[k].as_str().unwrap().to_owned();
            let query = match r.get("expr") {
                Some(e) => format!("SELECT {}", e.as_str().unwrap()),
                None => text("sql"),
            };
            Row {
                id: text("id"),
                sql: text("sql"),
                query,
                ansi: r["mode"].as_str().unwrap() == "ansi_on",
                want_type: r.get("type").and_then(|t| t.as_str()).map(str::to_owned),
                want_class: r
                    .get("error_class")
                    .and_then(|t| t.as_str())
                    .map(str::to_owned),
                want_params: r
                    .get("message_parameters")
                    .and_then(|p| p.as_object())
                    .map(|o| {
                        o.iter()
                            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_owned()))
                            .collect()
                    })
                    .unwrap_or_default(),
            }
        })
        .collect()
}

/// An error: its class and its message parameters, sorted.
type Failure = (String, BTreeMap<String, String>);

fn failure(e: &KrishivError) -> Failure {
    (
        e.error_class().name().to_owned(),
        e.message_parameters()
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect(),
    )
}

/// Our analysis of a row's query: the result type, or the error.
fn analyze(query: &str, ansi: bool) -> Result<String, Failure> {
    let plan = parse_and_lower(query, &ParserOptions::default(), SourceId(0))
        .map_err(|e| failure(&e.error))?;
    let conf = ResolutionConf {
        ansi,
        ..ResolutionConf::default()
    };
    let catalog = MemoryCatalog::new("spark_catalog");
    let a = block_on(DefaultAnalyzer.analyze(&plan, &context(catalog, conf)))
        .map_err(|e| failure(&e.error))?;
    a.output
        .first()
        .map(|c| c.data_type.simple_string())
        .ok_or_else(|| ("KRISHIV_NO_OUTPUT".to_owned(), BTreeMap::new()))
}

/// Why a row fails, or `None` when it passes.
fn verdict(row: &Row) -> Option<String> {
    let got = analyze(&row.query, row.ansi);
    match (&row.want_type, &row.want_class, got) {
        (Some(want), None, Ok(have)) if have == *want => None,
        (Some(want), None, Ok(have)) => Some(format!("resolves to {have} where Spark has {want}")),
        (Some(want), None, Err((class, params))) => {
            Some(format!("{class} {params:?} where Spark resolves to {want}"))
        }
        (None, Some(class), Err((have, params))) if have == *class && params == row.want_params => {
            None
        }
        (None, Some(class), Err((have, params))) => Some(format!(
            "{have} {params:?} where Spark has {class} {:?}",
            row.want_params
        )),
        (None, Some(class), Ok(have)) => {
            Some(format!("resolves to {have} where Spark has {class}"))
        }
        _ => Some("row is neither a type nor an error".to_owned()),
    }
}

#[test]
fn coercion_matrix_analyzes_as_spark_does() {
    let root = krishiv_testkit::tests_dir();
    let rows = rows(&root);
    let mut failures: BTreeMap<String, String> = BTreeMap::new();
    let mut by_form: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for row in &rows {
        let form = row.id.split('/').next().unwrap_or_default().to_owned();
        let tally = by_form.entry(form).or_default();
        tally.0 += 1;
        match verdict(row) {
            None => tally.1 += 1,
            Some(why) => {
                failures.insert(row.id.clone(), format!("{} | {why}", row.sql));
            }
        }
    }
    let (total, passed) = (rows.len(), rows.len() - failures.len());
    // Of 15,038 matrix rows, 7,371 resolve as Spark resolves them.
    assert_eq!((total, passed), (15038, 7371));
    let path = root.join(MANIFEST);
    if std::env::var_os("KRISHIV_UPDATE_MANIFEST").is_some() {
        let entries: Vec<Value> = failures
            .iter()
            .map(|(id, reason)| json!({"id": id, "reason": reason}))
            .collect();
        let doc = json!({"format_version": 1, "rows": total, "pass": passed, "entries": entries});
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap() + "\n").unwrap();
    }
    let manifest: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let listed: BTreeMap<String, String> = manifest["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            let text = |k: &str| e[k].as_str().unwrap().to_owned();
            (text("id"), text("reason"))
        })
        .collect();
    assert_eq!(
        (manifest["rows"].as_u64(), manifest["pass"].as_u64()),
        (Some(total as u64), Some(passed as u64)),
        "row totals changed: run with KRISHIV_UPDATE_MANIFEST=1 and review the diff"
    );
    let new: Vec<_> = failures
        .iter()
        .filter(|(k, _)| !listed.contains_key(*k))
        .collect();
    let fixed: Vec<_> = listed
        .keys()
        .filter(|k| !failures.contains_key(*k))
        .collect();
    // A listed row that now fails otherwise: the reason changed.
    let changed: Vec<_> = failures
        .iter()
        .filter(|(k, why)| listed.get(*k).is_some_and(|was| was != *why))
        .map(|(k, why)| (k, &listed[k], why))
        .collect();
    assert!(
        new.is_empty() && fixed.is_empty() && changed.is_empty(),
        "unlisted failures: {new:#?}\nlisted but passing: {fixed:#?}\n\
         failing for another reason (listed, now): {changed:#?}"
    );
}
