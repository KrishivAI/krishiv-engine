// SPDX-License-Identifier: Apache-2.0
//! Spark's `typeCoercion/native` golden files through the analyzer
//! (P2-T071b): every query statement resolves to Spark's result schema
//! (names and types), or fails with Spark's analysis error class; a
//! statement Spark fails only at run time must resolve. The statements
//! that use what Krishiv does not have yet are listed, each with a reason,
//! in `tests/fixtures/analysis/type-coercion-manifest.json`; the list must
//! match the failures and their reasons exactly, so a fix shrinks it and a
//! regression fails.
//! Rows need the engine (P2-T074): this checks analysis only.
//!
//! `KRISHIV_UPDATE_MANIFEST=1` rewrites the manifest from the failures.
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use krishiv_common::error::KrishivError;
use krishiv_plan::SourceId;
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_sql::{ParserOptions, parse_and_lower};
use krishiv_testkit::golden::parse::{Expected, parse_results};
use krishiv_testkit::golden::setup::FileSetup;
use krishiv_types::DataType;
use serde_json::{Value, json};

const DIR: &str = "spark-golden/v4.2.0";
const MANIFEST: &str = "fixtures/analysis/type-coercion-manifest.json";

/// The view `CREATE TEMPORARY VIEW name AS SELECT v [AS c], ...` defines,
/// for the literal forms the files use: an integer or a string.
fn view(sql: &str) -> Option<(String, Vec<TableColumn>)> {
    let lower = sql.to_ascii_lowercase();
    let rest = lower.strip_prefix("create temporary view ")?;
    let (name, body) = rest.split_once(" as select ")?;
    let start = sql.len() - body.len();
    let mut columns = Vec::new();
    for item in sql[start..].split(',') {
        let item = item.trim();
        let (value, alias) = match item.to_ascii_lowercase().rfind(" as ") {
            Some(i) => (item[..i].trim(), Some(item[i + 4..].trim())),
            None => (item, None),
        };
        let data_type = if value.parse::<i32>().is_ok() {
            DataType::Integer
        } else if value.starts_with('\'') && value.ends_with('\'') {
            DataType::String
        } else {
            return None;
        };
        let shown = value.trim_matches('\'');
        columns.push(TableColumn {
            name: alias.unwrap_or(shown).to_owned(),
            data_type,
            nullable: false,
        });
    }
    Some((name.trim().to_owned(), columns))
}

/// An error: its class and what it names (the feature, function or
/// table), if anything.
type Failure = (String, String);

fn failure(e: &KrishivError) -> Failure {
    let named = e
        .message_parameters()
        .iter()
        .find(|(k, _)| matches!(*k, "feature" | "routineName" | "relationName"))
        .map(|(_, v)| v.clone())
        .unwrap_or_default();
    (e.error_class().name().to_owned(), named)
}

/// Our outcome of a query: its schema as Spark prints it, or the error.
fn analyze(sql: &str, catalog: &MemoryCatalog, conf: ResolutionConf) -> Result<String, Failure> {
    let plan = parse_and_lower(sql, &ParserOptions::default(), SourceId(0))
        .map_err(|e| failure(&e.error))?;
    let a = block_on(DefaultAnalyzer.analyze(&plan, &context(catalog.clone(), conf)))
        .map_err(|e| failure(&e.error))?;
    let fields: Vec<String> = a
        .output
        .iter()
        .map(|c| format!("{}:{}", c.name, c.data_type.simple_string()))
        .collect();
    Ok(format!("struct<{}>", fields.join(",")))
}

/// Why a statement fails, from our outcome, for the manifest.
fn reason(got: &Result<String, Failure>) -> String {
    match got {
        Err((class, named)) if named.is_empty() => class.clone(),
        Err((class, named)) => format!("{class}: {named}"),
        Ok(schema) => format!("resolves to {schema}"),
    }
}

/// Every failing statement of the files, `(file, query) -> reason`, and
/// the number of query statements.
fn failures(root: &Path) -> (BTreeMap<(String, String), String>, usize) {
    let results = root.join(DIR).join("results/typeCoercion/native");
    let mut files: Vec<_> = std::fs::read_dir(&results)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "out"))
        .collect();
    files.sort();
    let mut out = BTreeMap::new();
    let mut queries = BTreeSet::new();
    for path in files {
        let stem = path.file_name().unwrap().to_str().unwrap();
        let name = format!("typeCoercion/native/{}", stem.trim_end_matches(".sql.out"));
        let input = root.join(DIR).join(format!("inputs/{name}.sql"));
        let setup = FileSetup::parse(&name, &std::fs::read_to_string(input).unwrap());
        let statements = parse_results(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for session in setup.sessions() {
            let mut conf: BTreeMap<String, String> = session.configs.into_iter().collect();
            let mut catalog = MemoryCatalog::new("spark_catalog");
            for st in &statements {
                let q = st.query.trim();
                let lower = q.to_ascii_lowercase();
                if let Some((k, v)) = lower.strip_prefix("set ").and_then(|kv| kv.split_once('=')) {
                    conf.insert(k.trim().to_owned(), v.trim().to_owned());
                    continue;
                }
                if lower.starts_with("create") {
                    if let Some((table, columns)) = view(q) {
                        catalog = catalog.with_table(&["default"], &table, columns, None);
                    }
                    continue;
                }
                if !(lower.starts_with("select") || lower.starts_with("with")) {
                    continue;
                }
                queries.insert((name.clone(), q.to_owned()));
                let rc = ResolutionConf::from_conf(|k| conf.get(k).map(String::as_str));
                let got = analyze(q, &catalog, rc);
                let pass = match (&st.expected, &got) {
                    (Expected::Rows(_), Ok(schema)) => *schema == st.schema,
                    (Expected::Error(e), Err((class, _))) => {
                        e.exception.ends_with("AnalysisException")
                            && e.error_class.as_deref() == Some(class.as_str())
                    }
                    // Spark failed only at run time: analysis succeeds.
                    (Expected::Error(e), Ok(_)) => !e.exception.ends_with("AnalysisException"),
                    (Expected::Rows(_), Err(_)) => false,
                };
                if !pass {
                    out.insert((name.clone(), q.to_owned()), reason(&got));
                }
            }
        }
    }
    (out, queries.len())
}

#[test]
fn type_coercion_goldens_analyze_as_spark_does() {
    let root = krishiv_testkit::tests_dir();
    let (failures, queries) = failures(&root);
    // Of 2,858 query statements, 2,635 resolve as Spark resolves them.
    assert_eq!((queries, queries - failures.len()), (2858, 2635));
    let path = root.join(MANIFEST);
    if std::env::var_os("KRISHIV_UPDATE_MANIFEST").is_some() {
        let entries: Vec<Value> = failures
            .iter()
            .map(|((file, query), reason)| json!({"file": file, "query": query, "reason": reason}))
            .collect();
        let doc = json!({"format_version": 1, "entries": entries});
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap() + "\n").unwrap();
    }
    let manifest: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let listed: BTreeMap<(String, String), String> = manifest["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            let text = |k: &str| e[k].as_str().unwrap().to_owned();
            ((text("file"), text("query")), text("reason"))
        })
        .collect();
    let new: Vec<_> = failures
        .iter()
        .filter(|(k, _)| !listed.contains_key(*k))
        .collect();
    let fixed: Vec<_> = listed
        .keys()
        .filter(|k| !failures.contains_key(*k))
        .collect();
    // A listed statement that now fails otherwise: the reason changed.
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
