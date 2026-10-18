// SPDX-License-Identifier: Apache-2.0
//! The built-in function registry against the Spark function matrix
//! (P2-T153): every matrix case of a registered function is analyzed with
//! the production registry, in both ANSI modes. A case passes when the
//! analyzer gives Spark's result type and nullability, or fails with
//! Spark's analysis error class; a case Spark fails only at run time must
//! analyze (to Spark's type, when Spark reported one). The cases that do
//! not pass are listed, each with a reason, in
//! `tests/fixtures/analysis/registry-matrix-manifest.json`; the list must
//! match exactly, so a fix shrinks it and a regression fails.
//!
//! `KRISHIV_UPDATE_MANIFEST=1` rewrites the manifest from the failures.

use std::collections::BTreeMap;

use krishiv_functions::registry::{Lookup, builtins};
use krishiv_functions::signature::SparkProfile;
use krishiv_plan::SourceId;
use krishiv_plan::analyzer::testing::{block_on, context};
use krishiv_plan::analyzer::{
    Analyzer as _, DefaultAnalyzer, MemoryCatalog, ResolutionConf, TableColumn,
};
use krishiv_sql::{ParserOptions, parse_and_lower};
use krishiv_testkit::fn_matrix::{Expected, Field, FnMatrix, MatrixCase, Mode};
use krishiv_types::DataType;
use serde_json::{Value, json};

const MANIFEST: &str = "fixtures/analysis/registry-matrix-manifest.json";

/// Whether `sql` has a `FROM` clause at the top level (outside brackets
/// and quotes: `extract(YEAR FROM d)` is not one).
fn has_from(sql: &str) -> bool {
    let (mut depth, mut quote) = (0_i32, None);
    let bytes = sql.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match (quote, b) {
            (Some(q), _) if b == q => quote = None,
            (None, b'\'' | b'"' | b'`') => quote = Some(b),
            (None, b'(') => depth += 1,
            (None, b')') => depth -= 1,
            (None, _) if depth == 0 => {
                let word = sql.get(i..i + 5).unwrap_or_default();
                let before = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
                if before && word.eq_ignore_ascii_case("from ") {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Our analysis of a case: the first column's type and nullability, or the
/// error class (and what it names). The parser has no `SELECT` without
/// `FROM` yet, so such a case reads from a one-row table `one`.
fn analyze(case: &MatrixCase) -> Result<(String, bool), String> {
    let sql = if has_from(&case.sql) {
        case.sql.clone()
    } else {
        format!("{} FROM one", case.sql)
    };
    let plan = parse_and_lower(&sql, &ParserOptions::default(), SourceId(0)).map_err(|e| {
        let params = e.error.message_parameters();
        let named = params
            .iter()
            .find(|(k, _)| *k == "feature")
            .map(|(_, v)| format!(": {v}"))
            .unwrap_or_default();
        format!("parse: {}{named}", e.error.error_class().name())
    })?;
    let conf = ResolutionConf {
        ansi: case.mode == Mode::AnsiOn,
        ..ResolutionConf::default()
    };
    let one = TableColumn {
        name: "one_".to_owned(),
        data_type: DataType::Integer,
        nullable: false,
    };
    let catalog =
        MemoryCatalog::new("spark_catalog").with_table(&["default"], "one", vec![one], None);
    let ctx = context(catalog, conf);
    let a = block_on(DefaultAnalyzer.analyze(&plan, &ctx)).map_err(|e| {
        let named = e
            .error
            .message_parameters()
            .iter()
            .find(|(k, _)| matches!(*k, "feature" | "routineName"))
            .map(|(_, v)| format!(": {v}"))
            .unwrap_or_default();
        format!("{}{named}", e.error.error_class().name())
    })?;
    let c = a.output.first().ok_or("no columns")?;
    Ok((c.data_type.simple_string(), c.nullable))
}

/// Why a case does not pass, or `None` when it does.
fn verdict(case: &MatrixCase) -> Option<String> {
    let got = analyze(case);
    let typed = |schema: &[Field], got: &(String, bool)| {
        let want = schema.first()?;
        (want.data_type != got.0 || want.nullable != got.1).then(|| {
            format!(
                "{}{} where Spark has {}{}",
                got.0,
                if got.1 { "" } else { " not null" },
                want.data_type,
                if want.nullable { "" } else { " not null" }
            )
        })
    };
    match (&case.expected, got) {
        (Expected::Rows { schema, .. } | Expected::Nondeterministic { schema }, Ok(g)) => {
            typed(schema, &g)
        }
        (Expected::Error { error, .. }, Err(class)) if error.phase == "analysis" => {
            let ours = class.split(':').next().unwrap_or_default();
            (ours != error.error_class)
                .then(|| format!("{class} where Spark has {}", error.error_class))
        }
        (Expected::Error { error, schema }, Ok(g)) if error.phase != "analysis" => {
            typed(schema.as_deref().unwrap_or_default(), &g)
        }
        (Expected::Error { error, .. }, Ok(g)) => Some(format!(
            "resolves to {} where Spark has {}",
            g.0, error.error_class
        )),
        (_, Err(class)) => Some(class),
    }
}

#[test]
fn registered_functions_analyze_as_spark_does() {
    let matrix = FnMatrix::load_default().unwrap();
    let registry = builtins();
    let registered = |f: &str| matches!(registry.lookup(f, SparkProfile::V4_2), Lookup::Found(_));
    let mut failures: BTreeMap<String, String> = BTreeMap::new();
    let mut per_function: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for case in matrix.cases.iter().filter(|c| registered(&c.function)) {
        let tally = per_function.entry(&case.function).or_default();
        tally.0 += 1;
        match verdict(case) {
            None => tally.1 += 1,
            Some(why) => {
                failures.insert(case.id.clone(), format!("{} | {why}", case.sql));
            }
        }
    }
    // Every registered function has matrix cases.
    let missing: Vec<&str> = registry
        .functions()
        .iter()
        .map(|d| d.name)
        .filter(|n| !per_function.contains_key(n))
        .collect();
    assert!(missing.is_empty(), "no matrix cases for {missing:?}");
    let (total, passed) = per_function
        .values()
        .fold((0, 0), |(t, p), (ct, cp)| (t + ct, p + cp));
    let path = krishiv_testkit::tests_dir().join(MANIFEST);
    if std::env::var_os("KRISHIV_UPDATE_MANIFEST").is_some() {
        let functions: Vec<Value> = per_function
            .iter()
            .map(|(f, (t, p))| json!({"function": f, "cases": t, "pass": p}))
            .collect();
        let entries: Vec<Value> = failures
            .iter()
            .map(|(id, reason)| json!({"id": id, "reason": reason}))
            .collect();
        let doc = json!({
            "format_version": 1,
            "cases": total,
            "pass": passed,
            "functions": functions,
            "entries": entries,
        });
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
        (manifest["cases"].as_u64(), manifest["pass"].as_u64()),
        (Some(total as u64), Some(passed as u64)),
        "case totals changed: run with KRISHIV_UPDATE_MANIFEST=1 and review the diff"
    );
    let differ: Vec<_> = failures
        .iter()
        .filter(|(id, why)| listed.get(*id) != Some(*why))
        .chain(listed.iter().filter(|(id, _)| !failures.contains_key(*id)))
        .take(20)
        .collect();
    assert!(differ.is_empty(), "manifest differs: {differ:#?}");
}
