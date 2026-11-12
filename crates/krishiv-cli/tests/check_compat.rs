// SPDX-License-Identifier: Apache-2.0
//! `krishiv check-compat` over the checked-in corpus (P3-T029): 30 PySpark
//! scripts and 30 SQL files with the findings planted in them
//! (`tests/scripts/gen_check_compat_corpus.py`).

#![expect(
    clippy::expect_used,
    reason = "integration-test helpers are outside #[test] fns"
)]

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

fn corpus() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/check_compat_corpus")
}

/// (file name, line, function, status) of each finding in a JSON report.
fn keys(report: &Value) -> BTreeSet<(String, u64, String, String)> {
    report
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

#[test]
fn corpus_findings_match_the_planted_ones() {
    let out = Command::new(env!("CARGO_BIN_EXE_krishiv"))
        .args(["check-compat", "--format", "json"])
        .arg(corpus())
        .output()
        .expect("run check-compat");
    assert_eq!(out.status.code(), Some(1), "findings exit 1");
    let got = keys(&serde_json::from_slice(&out.stdout).expect("json"));
    let want = keys(
        &serde_json::from_str(
            &std::fs::read_to_string(corpus().join("expected.json")).expect("expected.json"),
        )
        .expect("json"),
    );
    let hits = got.intersection(&want).count() as f64;
    let precision = hits / got.len() as f64;
    let recall = hits / want.len() as f64;
    let missed: Vec<_> = want.difference(&got).collect();
    let extra: Vec<_> = got.difference(&want).collect();
    assert!(
        precision >= 0.95 && recall >= 0.95,
        "precision {precision:.3}, recall {recall:.3}; missed {missed:?}; extra {extra:?}"
    );
}

#[test]
fn exit_codes_follow_the_findings_and_read_errors() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("ok.sql"), "SELECT abs(1)\n").expect("write");
    let run = |p: &std::path::Path| {
        Command::new(env!("CARGO_BIN_EXE_krishiv"))
            .arg("check-compat")
            .arg(p)
            .output()
            .expect("run")
            .status
            .code()
    };
    assert_eq!(run(dir.path()), Some(0));
    std::fs::write(dir.path().join("bad.ipynb"), "{not json").expect("write");
    assert_eq!(run(dir.path()), Some(2));
    assert_eq!(run(&dir.path().join("missing")), Some(2));
}
