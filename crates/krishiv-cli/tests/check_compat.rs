// SPDX-License-Identifier: Apache-2.0
//! `krishiv check-compat` end to end (P3-T029): the checked-in corpus, exit codes,
//! the JSON envelope, and false-positive sanity runs over code that is not ours.
//!
//! The corpus has three parts under `tests/check_compat_corpus/`:
//! - `handwritten/`: realistic SQL, PySpark and a notebook, with a hand-checked
//!   `expected.json` that does not come from the matrix;
//! - `supported_only/`: files that must give no finding at all;
//! - `generated/`: a volume check built by `tests/scripts/gen_check_compat_corpus.py`.

#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::Value;

fn corpus() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/check_compat_corpus")
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run(args: &[&str], path: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_krishiv"))
        .arg("check-compat")
        .args(args)
        .arg(path)
        .output()
        .expect("run check-compat")
}

fn report(path: &Path) -> (Option<i32>, Value) {
    let out = run(&["--format", "json"], path);
    let v = serde_json::from_slice(&out.stdout).expect("json report");
    (out.status.code(), v)
}

/// (file name, line, function, status) of each finding of a report or expected file.
type Key = (String, u64, String, String);

fn keys(findings: &Value) -> BTreeSet<Key> {
    findings
        .as_array()
        .expect("array")
        .iter()
        .map(|f| {
            let file = f["file"].as_str().expect("file");
            let file = file.rsplit('/').next().expect("name").to_string();
            (
                file,
                f["line"].as_u64().expect("line"),
                f["name"].as_str().expect("name").to_string(),
                f["status"].as_str().expect("status").to_string(),
            )
        })
        .collect()
}

fn is_sql(k: &Key) -> bool {
    k.0.ends_with(".sql")
}

/// Asserts precision and recall of at least 95% for the keys accepted by `kind`.
fn score(label: &str, got: &BTreeSet<Key>, want: &BTreeSet<Key>, kind: fn(&Key) -> bool) {
    let got: BTreeSet<_> = got.iter().filter(|k| kind(k)).collect();
    let want: BTreeSet<_> = want.iter().filter(|k| kind(k)).collect();
    assert!(!want.is_empty(), "{label}: nothing expected");
    let hits = got.intersection(&want).count() as f64;
    let precision = hits / got.len() as f64;
    let recall = hits / want.len() as f64;
    let missed: Vec<_> = want.difference(&got).collect();
    let extra: Vec<_> = got.difference(&want).collect();
    assert!(
        precision >= 0.95 && recall >= 0.95,
        "{label}: precision {precision:.3}, recall {recall:.3}; missed {missed:?}; extra {extra:?}"
    );
}

fn check_part(part: &str) {
    let dir = corpus().join(part);
    let (code, rep) = report(&dir);
    assert_eq!(code, Some(1), "findings exit 1");
    let want: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("expected.json")).expect("expected.json"),
    )
    .expect("json");
    let got = keys(&rep["findings"]);
    let want = keys(&want);
    score(&format!("{part} sql"), &got, &want, is_sql);
    score(&format!("{part} python"), &got, &want, |k| !is_sql(k));
}

#[test]
fn handwritten_corpus_findings_match_the_hand_checked_ones() {
    check_part("handwritten");
}

#[test]
fn generated_corpus_findings_match_the_planted_ones() {
    check_part("generated");
}

#[test]
fn handwritten_notebook_findings_carry_file_line_and_cell() {
    let (_, rep) = report(&corpus().join("handwritten/notebook.ipynb"));
    let f: Vec<_> = rep["findings"]
        .as_array()
        .expect("findings")
        .iter()
        .map(|f| (f["line"].as_u64(), f["cell"].as_u64(), f["name"].as_str()))
        .collect();
    assert_eq!(
        f,
        [
            (Some(27), Some(2), Some("xpath_float")),
            (Some(37), Some(3), Some("legacy_score"))
        ]
    );
}

#[test]
fn supported_only_files_give_no_findings() {
    let (code, rep) = report(&corpus().join("supported_only"));
    assert_eq!(rep["findings"], serde_json::json!([]), "{rep}");
    assert_eq!(code, Some(0));
}

#[test]
fn json_report_is_a_versioned_envelope() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("a.sql"), "SELECT nopefn(1)\n").expect("write");
    std::fs::write(dir.path().join("bad.ipynb"), "{not json").expect("write");
    let (code, rep) = report(dir.path());
    assert_eq!(code, Some(2));
    assert_eq!(rep["schema_version"], 1);
    assert_eq!(rep["files_scanned"], 2);
    assert_eq!(rep["findings"][0]["name"], "nopefn");
    assert_eq!(rep["findings"][0]["status"], "unknown");
    assert!(
        rep["errors"][0]["path"]
            .as_str()
            .is_some_and(|p| p.ends_with("bad.ipynb")),
        "{rep}"
    );
}

#[test]
fn text_report_spells_out_the_statuses() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("a.sql"),
        "SELECT nopefn(1),\n  sentences(2)\n",
    )
    .expect("write");
    let out = run(&[], dir.path());
    let text = String::from_utf8(out.stdout).expect("utf8");
    let lines: Vec<_> = text.lines().collect();
    assert_eq!(lines.len(), 2, "{text}");
    assert!(
        lines[0].ends_with("a.sql:1: nopefn: unknown, not supported"),
        "{text}"
    );
    assert!(lines[1].ends_with("a.sql:2: sentences: not yet"), "{text}");
}

#[test]
fn exit_codes_follow_the_findings_and_read_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("ok.sql"), "SELECT abs(1)\n").expect("write");
    let code = |p: &Path| run(&[], p).status.code();
    assert_eq!(code(dir.path()), Some(0));
    std::fs::write(dir.path().join("bad.ipynb"), "{not json").expect("write");
    assert_eq!(code(dir.path()), Some(2));
    assert_eq!(code(&dir.path().join("missing")), Some(2));
}

#[test]
fn walk_skips_environments_and_rejects_other_named_files() {
    let dir = tempfile::tempdir().expect("tempdir");
    let venv = dir.path().join(".venv/lib/site-packages");
    std::fs::create_dir_all(&venv).expect("mkdir");
    std::fs::write(
        venv.join("x.py"),
        "import pyspark.sql.functions as F\nF.sentences(1)\n",
    )
    .expect("write");
    std::fs::write(dir.path().join("notes.txt"), "sentences(1)").expect("write");
    assert_eq!(run(&[], dir.path()).status.code(), Some(0));
    assert_eq!(
        run(&[], &dir.path().join("notes.txt")).status.code(),
        Some(2)
    );
}

/// SQL from Spark's own test inputs is full of table aliases, CTE column lists, casts
/// and type names: none of them may be reported, and nothing is `partial`.
#[test]
fn vendored_spark_sql_inputs_give_no_false_positives() {
    let dir = repo_root().join("tests/spark-golden/v4.2.0/inputs");
    let (_, rep) = report(&dir);
    let findings = rep["findings"].as_array().expect("findings");
    assert!(findings.len() > 100, "scanned {}", findings.len());
    let names: BTreeSet<_> = findings
        .iter()
        .filter(|f| f["status"] == "unknown")
        .map(|f| f["name"].as_str().expect("name"))
        .collect();
    for bad in [
        "t",
        "t1",
        "t2",
        "tab",
        "v",
        "r",
        "tbl",
        "identifier",
        "try_cast",
        "cast",
        "decimal",
        "varchar",
        "numeric",
        "columns",
        "variable",
        "until",
    ] {
        assert!(!names.contains(bad), "`{bad}` reported as a function");
    }
    assert!(
        findings.iter().all(|f| f["status"] != "partial"),
        "partial reported"
    );
}

/// Python code that is not PySpark (the repository's bench harness and package, with
/// docstrings and messages full of parentheses) gives no findings.
#[test]
fn non_spark_python_gives_no_findings() {
    for rel in ["bench/src", "xtask"] {
        let dir = repo_root().join(rel);
        let (_, rep) = report(&dir);
        assert_eq!(rep["findings"], serde_json::json!([]), "{rel}: {rep}");
    }
}
