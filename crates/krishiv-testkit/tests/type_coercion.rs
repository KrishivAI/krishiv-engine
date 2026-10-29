// SPDX-License-Identifier: Apache-2.0
//! Spark's `typeCoercion/native` golden files through the analyzer
//! (P2-T071b): every query statement resolves to Spark's result schema
//! (names and types), or fails with Spark's analysis error class; a
//! statement Spark fails only at run time must resolve. The statements
//! that use what Krishiv does not have yet are listed, each with a reason,
//! in `tests/fixtures/analysis/type-coercion/`, one file per coercion
//! input file; the list must match the failures and their reasons exactly,
//! so a fix shrinks it and a regression fails.
//! Rows need the engine (P2-T074): this checks analysis only.
//!
//! `KRISHIV_UPDATE_MANIFEST=1` rewrites only the per-file manifests whose
//! content changed and deletes the files with no failures left.
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use krishiv_common::error::KrishivError;
use krishiv_plan::SourceId;
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_sql::{ParserOptions, parse_and_lower};
use krishiv_testkit::golden::parse::{Expected, parse_results};
use krishiv_testkit::golden::setup::FileSetup;
use krishiv_testkit::manifest_dir::{read_json_dir, sync_json_dir};
use krishiv_types::DataType;
use serde_json::{Value, json};

const DIR: &str = "spark-golden/v4.2.0";
/// One file per coercion input file (e.g. `typeCoercion/native/concat.json`)
/// under `tests/fixtures/analysis/`.
const MANIFEST_DIR: &str = "fixtures/analysis/type-coercion";

/// Manifest file format version this test reads and writes.
const FORMAT_VERSION: u64 = 1;

/// The listed failures (`(file, query) -> reason`) across the per-file
/// manifests. Each file holds `{"format_version", "file", "entries":
/// [{query, reason}]}` at the path of its coercion input file.
fn load_listed(dir: &Path) -> BTreeMap<(String, String), String> {
    let mut listed = BTreeMap::new();
    for (rel, text) in read_json_dir(dir).unwrap() {
        let stem = rel
            .to_str()
            .unwrap()
            .strip_suffix(".json")
            .unwrap_or_default()
            .replace('\\', "/");
        let file: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            file["format_version"].as_u64(),
            Some(FORMAT_VERSION),
            "{}: bad format_version",
            rel.display()
        );
        let name = file["file"].as_str().unwrap().to_owned();
        assert_eq!(
            name,
            stem,
            "{}: file name does not match its coercion file",
            rel.display()
        );
        for e in file["entries"].as_array().unwrap() {
            let query = e["query"].as_str().unwrap().to_owned();
            let reason = e["reason"].as_str().unwrap().to_owned();
            assert!(
                listed
                    .insert((name.clone(), query.clone()), reason)
                    .is_none(),
                "{name}: {query} is listed twice"
            );
        }
    }
    listed
}

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
    // Of 2,858 query statements, 2,705 resolve as Spark resolves them.
    assert_eq!((queries, queries - failures.len()), (2858, 2705));
    let dir = root.join(MANIFEST_DIR);
    if std::env::var_os("KRISHIV_UPDATE_MANIFEST").is_some() {
        let mut by_file: BTreeMap<&str, Vec<(&String, &String)>> = BTreeMap::new();
        for ((file, query), reason) in &failures {
            by_file.entry(file).or_default().push((query, reason));
        }
        let mut wanted: BTreeMap<PathBuf, String> = BTreeMap::new();
        for (file, rows) in &by_file {
            let entries: Vec<Value> = rows
                .iter()
                .map(|(query, reason)| json!({"query": query, "reason": reason}))
                .collect();
            let doc = json!({
                "format_version": FORMAT_VERSION,
                "file": file,
                "entries": entries,
            });
            wanted.insert(
                PathBuf::from(format!("{file}.json")),
                serde_json::to_string_pretty(&doc).unwrap() + "\n",
            );
        }
        sync_json_dir(&dir, &wanted).unwrap();
    }
    let listed = load_listed(&dir);
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
