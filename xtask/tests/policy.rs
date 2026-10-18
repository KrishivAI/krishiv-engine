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
    for needle in ["cargo xtask changes", "cargo hack check --each-feature"] {
        assert!(ci.contains(needle), "ci.yml lacks {needle:?}");
    }
    // Label-dependent policy lives in its own workflow so label changes never
    // restart (and cancel) the full CI run; the job name is a required check.
    let types_line = ci
        .lines()
        .find(|l| l.trim_start().starts_with("types:"))
        .expect("ci.yml pull_request types");
    assert!(
        !types_line.contains("labeled"),
        "ci.yml must not trigger on label changes"
    );
    assert!(!ci.contains("cargo xtask pr-size"));
    assert!(!ci.contains("  policy:\n"));
    let policy = read(".github/workflows/pr-policy.yml");
    for needle in [
        "  policy:\n",
        "cargo xtask pr-size",
        "cargo xtask dco",
        "cargo xtask workflow-lint",
        "cargo xtask spdx",
        "cargo xtask source-guard",
        "size-approved",
        "labeled, unlabeled",
        "group: pr-policy-",
    ] {
        assert!(policy.contains(needle), "pr-policy.yml lacks {needle:?}");
    }
    // Budgets: ci.yml only measures (no label reads); the label-aware gate in
    // pr-policy.yml compares that measurement, so a label re-runs only the gate.
    for needle in [
        "  budgets-measure:\n",
        "cargo xtask budgets --build-time --measure-only",
        "name: budgets-current",
    ] {
        assert!(ci.contains(needle), "ci.yml lacks {needle:?}");
    }
    for label in ["budget-approved", "unsafe-reviewed", "size-approved"] {
        assert!(
            !ci.contains(&format!("approval-label.sh {label}")),
            "ci.yml must not resolve the {label} label (labels do not re-run it)"
        );
    }
    for needle in [
        "  budgets:\n",
        ".github/scripts/budgets-measurement.sh",
        "git show FETCH_HEAD:budgets.json > budgets.json",
        "approval-label.sh budget-approved",
        "approval-label.sh unsafe-reviewed",
        "cargo xtask budgets --compare",
    ] {
        assert!(policy.contains(needle), "pr-policy.yml lacks {needle:?}");
    }
    // The gate finds the measurement by job and artifact name.
    let fetch = read(".github/scripts/budgets-measurement.sh");
    for needle in [
        "actions/workflows/ci.yml/runs",
        "select(.name == \"budgets-measure\")",
        "--name budgets-current",
    ] {
        assert!(
            fetch.contains(needle),
            "budgets-measurement.sh lacks {needle:?}"
        );
    }
    assert!(
        !ci.contains("--all-features --locked --no-tests"),
        "PR nextest must use default features"
    );
    let nightly = read(".github/workflows/nightly.yml");
    assert!(nightly.contains("runs-on: ubuntu-latest"));
    assert!(!nightly.contains("self-hosted"));
    assert!(!nightly.contains("krishiv-heavy"));
    assert!(nightly.contains("--all-features"));
    assert!(nightly.contains("sccache"));
    assert!(!nightly.contains("/home/"));
}

#[test]
fn budgets_measurement_script_waits_and_fetches() {
    let script = root().join(".github/scripts/test-budgets-measurement.sh");
    let out = Command::new("bash").arg(&script).output().unwrap();
    assert!(
        out.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Every job that runs steps starts with Harden-Runner (audit mode), and
/// every checkout drops its credentials (zizmor `artipacked`). Jobs that
/// call a reusable workflow have no steps of their own.
#[test]
fn every_job_hardens_its_runner_first() {
    let dir = root().join(".github/workflows");
    let mut jobs = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if line.trim_end() == "    steps:" {
                jobs += 1;
                let first: String = lines[i + 1..(i + 3).min(lines.len())].join("\n");
                assert!(
                    first.contains("uses: step-security/harden-runner@")
                        && lines[i + 1..(i + 5).min(lines.len())]
                            .iter()
                            .any(|l| l.trim() == "egress-policy: audit"),
                    "{name}:{}: the job's first step is not Harden-Runner in audit mode",
                    i + 1
                );
            }
            if line
                .trim_start()
                .trim_start_matches("- ")
                .starts_with("uses: actions/checkout@")
            {
                let block: Vec<&str> = lines[i + 1..]
                    .iter()
                    .take_while(|l| !l.trim_start().starts_with("- ") && !l.trim().is_empty())
                    .copied()
                    .collect();
                assert!(
                    block
                        .iter()
                        .any(|l| l.trim() == "persist-credentials: false"),
                    "{name}:{}: checkout keeps its credentials",
                    i + 1
                );
            }
        }
    }
    assert!(jobs > 20, "found only {jobs} jobs");
}

#[test]
fn security_workflows_are_wired() {
    let sec = read(".github/workflows/security.yml");
    for needle in [
        "actions/dependency-review-action@",
        "fail-on-severity: high",
        "zizmorcore/zizmor-action@",
        "fail-on-scopes: runtime, development, unknown",
        "advanced-security: false",
        "advanced-security: true",
    ] {
        assert!(sec.contains(needle), "security.yml lacks {needle:?}");
    }
    // The license allow list matches deny.toml's.
    let deny = read("deny.toml");
    let allow = deny
        .split("[licenses]")
        .nth(1)
        .and_then(|s| s.split("allow = [").nth(1))
        .and_then(|s| s.split(']').next())
        .unwrap();
    let licenses: Vec<&str> = allow.lines().filter_map(|l| l.split('"').nth(1)).collect();
    assert!(licenses.len() > 5);
    let listed = sec
        .split("allow-licenses: >-")
        .nth(1)
        .unwrap()
        .lines()
        .take_while(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ");
    let listed: Vec<&str> = listed.split(',').map(str::trim).collect();
    assert_eq!(
        listed, licenses,
        "dependency-review allow-licenses != deny.toml"
    );
    let osv = read(".github/workflows/osv-scanner.yml");
    for lock in ["Cargo.lock", "python/uv.lock", "bench/uv.lock"] {
        assert!(
            osv.contains(&format!("--lockfile={lock}")),
            "osv lacks {lock}"
        );
    }
    assert!(osv.contains("osv-scanner-reusable-pr.yml@"));
    assert!(osv.contains("osv-scanner-reusable.yml@"));
    assert!(osv.contains("schedule:"));
}
