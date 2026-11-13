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
    // The expected files are complete (planted or hand-checked), so any miss or extra fails.
    let missed: Vec<_> = want.difference(&got).collect();
    let extra: Vec<_> = got.difference(&want).collect();
    assert!(
        missed.is_empty() && extra.is_empty(),
        "{part}: missed {missed:?}; extra {extra:?}"
    );
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
    let (code, rep) = report(&dir);
    assert_eq!(code, Some(1), "{rep}");
    assert_eq!(rep["errors"], serde_json::json!([]));
    let findings = rep["findings"].as_array().expect("findings");
    assert!(findings.len() > 100, "scanned {}", findings.len());
    // Precision against the source text: every finding sits on a line that has the
    // reported name as a whole word (independent of the scanner's own state).
    let mut lines_of: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    for f in findings {
        let file = f["file"].as_str().expect("file");
        let text = lines_of.entry(file.to_string()).or_insert_with(|| {
            let t = std::fs::read_to_string(file).expect("read");
            t.lines().map(str::to_ascii_lowercase).collect()
        });
        let line = usize::try_from(f["line"].as_u64().expect("line")).expect("line fits");
        let name = f["name"].as_str().expect("name");
        let has_word = text.get(line - 1).is_some_and(|l| {
            l.match_indices(name).any(|(i, _)| {
                let word = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
                !word(l[..i].chars().next_back()) && !word(l[i + name.len()..].chars().next())
            })
        });
        assert!(has_word, "{file}:{line}: `{name}` not on that line");
    }
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
        let (code, rep) = report(&dir);
        assert_eq!(code, Some(0), "{rel}: {rep}");
        assert_eq!(rep["errors"], serde_json::json!([]), "{rel}");
        // `xtask` is Rust; only the Python trees must actually have been read.
        if rel != "xtask" {
            assert!(
                rep["files_scanned"].as_u64() > Some(0),
                "{rel}: nothing scanned"
            );
        }
        assert_eq!(rep["findings"], serde_json::json!([]), "{rel}: {rep}");
    }
}

/// Recall over Spark's own SQL inputs: every call of a function the matrix lists as
/// "not yet" (found by a plain `name(` token scan) is reported, at least 95% of them.
#[test]
fn vendored_spark_sql_inputs_report_at_least_95_percent_of_not_yet_calls() {
    let matrix: Value = serde_json::from_str(include_str!("../src/check_compat/functions.json"))
        .expect("matrix json");
    let dir = repo_root().join("tests/spark-golden/v4.2.0/inputs");
    let (_, rep) = report(&dir);
    let reported: BTreeSet<(String, u64, String)> = rep["findings"]
        .as_array()
        .expect("findings")
        .iter()
        .filter(|f| f["status"] == "not yet")
        .map(|f| {
            (
                f["file"].as_str().expect("file").to_string(),
                f["line"].as_u64().expect("line"),
                f["name"].as_str().expect("name").to_string(),
            )
        })
        .collect();
    let mut want = BTreeSet::new();
    let mut stack = vec![dir];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).expect("read_dir") {
            let path = e.expect("entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|x| x == "sql") {
                let text = std::fs::read_to_string(&path).expect("read");
                let file = path.to_string_lossy().to_string();
                for (n, line) in text.lines().enumerate() {
                    if line.trim_start().starts_with("--") {
                        continue;
                    }
                    let b = line.as_bytes();
                    let mut i = 0;
                    while i < b.len() {
                        if b[i].is_ascii_alphabetic() || b[i] == b'_' {
                            let s = i;
                            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                                i += 1;
                            }
                            let dotted = s > 0 && b[s - 1] == b'.';
                            let word = line[s..i].to_ascii_lowercase();
                            let next = line[i..].trim_start().as_bytes().first().copied();
                            if next == Some(b'(') && !dotted && matrix[&word] == "not yet" {
                                want.insert((file.clone(), n as u64 + 1, word));
                            }
                        } else {
                            i += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(want.len() > 100, "only {} expected calls", want.len());
    let hit = want.iter().filter(|w| reported.contains(*w)).count();
    let pct = hit * 100 / want.len();
    assert!(pct >= 95, "recall {pct}% ({hit}/{})", want.len());
}

#[cfg(unix)]
#[test]
fn directory_symlink_cycles_are_not_followed() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("a.sql"), "SELECT abs(1)\n").expect("write");
    std::os::unix::fs::symlink(".", dir.path().join("again")).expect("symlink");
    let (code, rep) = report(dir.path());
    assert_eq!(code, Some(0), "{rep}");
    assert_eq!(rep["files_scanned"], 1, "{rep}");
}

#[test]
fn findings_carry_an_alternative_where_one_is_listed() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("a.sql"),
        "SELECT hll_sketch_estimate(hll_sketch_agg(c)), sentences(s) FROM t\n",
    )
    .expect("write");
    let (_, rep) = report(dir.path());
    let by_name = |n: &str| {
        rep["findings"]
            .as_array()
            .and_then(|a| a.iter().find(|f| f["name"] == n))
            .cloned()
            .expect("finding")
    };
    let alt = by_name("hll_sketch_agg")["alternative"].clone();
    assert!(
        alt.as_str()
            .is_some_and(|a| a.contains("approx_count_distinct")),
        "{alt}"
    );
    assert!(by_name("sentences").get("alternative").is_none());
    let text = run(&[], dir.path());
    let out = String::from_utf8_lossy(&text.stdout);
    assert!(
        out.contains("hll_sketch_agg: not yet (alternative: approx_count_distinct"),
        "{out}"
    );
}

/// P3-T029 speed: 1,000 files in under 10 s (the CI box gets the same bound; the real
/// run takes well under a second).
#[test]
fn a_thousand_files_scan_in_under_ten_seconds() {
    let dir = tempfile::tempdir().expect("tempdir");
    for i in 0..1000 {
        let (name, body) = if i % 2 == 0 {
            (
                format!("q{i}.sql"),
                "WITH a AS (SELECT id, upper(name) AS n FROM t WHERE id > 1)\nSELECT n, count(*), sentences(n) FROM a GROUP BY n;\n".repeat(20),
            )
        } else {
            (
                format!("j{i}.py"),
                "import pyspark.sql.functions as F\ndf = spark.table('t').select(F.col('a'), F.mystery_fn('b'))\nspark.sql('SELECT abs(x) FROM t')\n".repeat(20),
            )
        };
        std::fs::write(dir.path().join(name), body).expect("write");
    }
    let start = std::time::Instant::now();
    let (code, rep) = report(dir.path());
    let took = start.elapsed();
    assert_eq!(code, Some(1));
    assert_eq!(rep["files_scanned"], 1000);
    assert!(took.as_secs_f64() < 10.0, "took {took:?}");
}

/// A report that cannot be written is a failure, not a clean run.
#[cfg(target_os = "linux")]
#[test]
fn an_unwritable_report_exits_2() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("a.sql"), "SELECT abs(1)\n").expect("write");
    for args in [&["--format", "json"][..], &[][..]] {
        let full = std::fs::OpenOptions::new()
            .write(true)
            .open("/dev/full")
            .expect("/dev/full");
        // Findings make the text report non-empty; the JSON report is always written.
        std::fs::write(dir.path().join("b.sql"), "SELECT sentences(x)\n").expect("write");
        let out = Command::new(env!("CARGO_BIN_EXE_krishiv"))
            .arg("check-compat")
            .args(args)
            .arg(dir.path())
            .stdout(full)
            .output()
            .expect("run");
        assert_eq!(out.status.code(), Some(2), "{args:?}");
    }
}

/// P3-T074: the Python of the quickstart and the migration guide, and the sample notebooks,
/// call only supported functions (the same scan a user runs on their own project).
#[test]
fn the_docs_examples_pass_check_compat() {
    let docs = repo_root().join("docs");
    let scratch = tempfile::tempdir().expect("tempdir");
    for (i, page) in ["get-started/quickstart.md", "guides/pyspark-migration.md"]
        .iter()
        .enumerate()
    {
        let text = std::fs::read_to_string(docs.join(page)).expect("docs page");
        let mut code = String::from("import krishiv\nfrom krishiv.sql import functions as F\n");
        let mut in_py = false;
        for line in text.lines() {
            match (in_py, line) {
                (false, "```python") => in_py = true,
                (true, "```") => in_py = false,
                (true, l) => {
                    code.push_str(l);
                    code.push('\n');
                }
                _ => {}
            }
        }
        assert!(code.lines().count() > 5, "{page}: no python blocks found");
        std::fs::write(scratch.path().join(format!("page{i}.py")), code).expect("write");
    }
    let (code, v) = report(scratch.path());
    assert_eq!(code, Some(0), "docs pages: {v}");
    let (code, v) = report(&docs.join("notebooks"));
    assert_eq!(code, Some(0), "notebooks: {v}");
    assert!(
        v["files_scanned"].as_u64() >= Some(3),
        "notebooks scanned: {v}"
    );
}
