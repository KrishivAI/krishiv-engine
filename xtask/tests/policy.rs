// SPDX-License-Identifier: Apache-2.0
//! Grep-style tests that keep agent and review policy files honest
//! (P2-T111, P2-T122) and run the git hook shell test (P2-T134).
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test helpers sit outside #[test] fns"
)]

use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(root().join(rel)).unwrap_or_else(|e| panic!("cannot read {rel}: {e}"))
}

const PERF_ITEMS: [&str; 5] = [
    "allocations per row",
    "copies",
    "dyn dispatch in hot loops",
    "locks across `.await`",
    "blocking in async",
];

#[test]
fn reviewer_agent_has_the_perf_checklist_and_perf_output_section() {
    let reviewer = read(".claude/agents/reviewer.md").to_lowercase();
    for item in PERF_ITEMS {
        assert!(
            reviewer.contains(item),
            "reviewer.md lacks perf item {item:?}"
        );
    }
    let output = reviewer
        .split("## output")
        .nth(1)
        .expect("reviewer.md has an Output section");
    assert!(
        output.contains("**perf** section"),
        "reviewer output format lacks a perf section"
    );
}

#[test]
fn pr_template_has_the_checklists() {
    let t = read(".github/pull_request_template.md").to_lowercase();
    for item in PERF_ITEMS {
        assert!(t.contains(item), "PR template lacks perf item {item:?}");
    }
    for needle in [
        "new dependency checklist (adr 0028)",
        "rust api guidelines",
        "tests run",
        "signed off",
        "spdx-license-identifier",
    ] {
        assert!(t.contains(needle), "PR template lacks {needle:?}");
    }
}

#[test]
fn spark_semantics_agent_requires_executed_expectations() {
    let a = read(".claude/agents/spark-semantics.md");
    assert!(a.contains("produced by executing Spark in the `krishiv-bench` container"));
    assert!(a.contains("T106a"));
    assert!(a.contains("Never transcribe"));
    assert!(a.contains("only for error classes"));
    assert!(a.contains("cite the\nSpark test file"));
}

#[test]
fn notice_skeleton_exists() {
    let n = read("NOTICE");
    assert!(n.contains("Apache License, Version 2.0"));
    assert!(n.contains("Apache Spark"));
}

#[test]
fn pre_push_hook_refuses_main_and_force_pushes() {
    let script = root().join(".githooks/test-pre-push.sh");
    let out = Command::new("sh").arg(&script).output().unwrap();
    assert!(
        out.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn deny_toml_guards_object_store_coupling() {
    let deny = read("deny.toml");
    assert!(deny.contains("[[bans.features]]"));
    assert!(deny.contains("crate = \"parquet\""));
    assert!(deny.contains("deny = [\"object_store\"]"));
    assert!(deny.contains("{ crate = \"object_store\", deny-multiple-versions = true }"));
    let cargo = read("Cargo.toml");
    assert!(cargo.contains("object_store = { version = \"=0.14.2\""));
}

#[test]
fn ci_policy_wiring() {
    let ci = read(".github/workflows/ci.yml");
    for needle in [
        "cargo xtask pr-size",
        "cargo xtask dco",
        "cargo xtask workflow-lint",
        "cargo xtask spdx",
        "cargo xtask source-guard",
        "cargo xtask changes",
        "size-approved",
        "cargo hack check --each-feature",
    ] {
        assert!(ci.contains(needle), "ci.yml lacks {needle:?}");
    }
    assert!(
        !ci.contains("--all-features --locked --no-tests"),
        "PR nextest must use default features"
    );
    let nightly = read(".github/workflows/nightly.yml");
    assert!(nightly.contains("[self-hosted, krishiv-heavy]"));
    assert!(nightly.contains("--all-features"));
    assert!(nightly.contains("sccache"));
    assert!(!nightly.contains("ubuntu-latest"));
}
