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

/// The text of the top-level job `name` of a workflow: from its key to the
/// next job's key (or the end).
fn job<'a>(workflow: &'a str, name: &str) -> &'a str {
    let key = format!("\n  {name}:\n");
    let start = workflow
        .find(&key)
        .unwrap_or_else(|| panic!("no job {name:?}"))
        + 1;
    let rest = &workflow[start..];
    let end = rest
        .match_indices('\n')
        .map(|(i, _)| i + 1)
        .find(|&i| {
            let line = rest[i..].lines().next().unwrap_or_default();
            line.starts_with("  ") && !line.starts_with("   ") && !line.starts_with("  #")
        })
        .unwrap_or(rest.len());
    &rest[..end]
}

#[test]
fn ci_policy_wiring() {
    let ci = read(".github/workflows/ci.yml");
    for needle in ["cargo xtask changes", "cargo hack check --each-feature"] {
        assert!(ci.contains(needle), "ci.yml lacks {needle:?}");
    }
    // The fuzz targets are built by their own job, on every Rust change.
    let fuzz_build = job(&ci, "fuzz-build");
    for needle in [
        "    needs: changes\n",
        "    if: needs.changes.outputs.rust == 'true' && github.event_name != 'push' && github.event_name != 'merge_group'\n",
        "        run: cargo \"+$NIGHTLY\" fuzz build --target x86_64-unknown-linux-gnu\n",
    ] {
        // Whole lines: a deeper-indented step-level `if:` or `run:` must not match.
        assert!(
            fuzz_build.lines().any(|line| line == needle.trim_end()),
            "fuzz-build lacks {needle:?}"
        );
    }
    // The merge queue runs the build, tests, clippy and the cheap lints; the long checks stay
    // on pull requests and run on main nightly. Each stays a job that reports (skipped passes).
    assert!(
        ci.contains("  schedule:\n    - cron:"),
        "ci.yml runs on main nightly"
    );
    for name in ["miri-kernels", "wheels"] {
        let body = job(&ci, name);
        assert!(
            body.contains("github.event_name != 'merge_group'"),
            "{name} must not run in the merge queue"
        );
    }
    for name in ["test", "clippy", "fmt", "layering"] {
        assert!(
            !job(&ci, name).contains("merge_group"),
            "{name} must run in the merge queue"
        );
    }
    // The nightly and the push run must not cancel each other, and a nightly failure is reported.
    assert!(
        ci.contains("group: ci-${{ github.workflow }}-${{ github.event_name }}-"),
        "the concurrency group must include the event"
    );
    assert!(
        job(&ci, "nightly-failure-notice").contains(
            "if: always() && github.event_name == 'schedule' && contains(needs.*.result, 'failure')"
        ),
        "the failure notice must run after a failed dependency, on the schedule only"
    );
    assert!(job(&ci, "nightly-failure-notice").contains("issues: write"));
    // A release or a published binary needs the latest nightly run green on its commit.
    for wf in ["publish-testpypi.yml", "binaries.yml"] {
        let text = read(&format!(".github/workflows/{wf}"));
        assert!(
            text.contains(".github/scripts/require-green-nightly.sh"),
            "{wf} must require a green nightly run"
        );
    }
    // Diff coverage tests only the packages the PR touches and their dependents.
    let coverage = read(".github/workflows/coverage.yml");
    for needle in [
        "cargo xtask coverage-scope",
        "steps.scope.outputs.packages",
        "cargo xtask coverage --no-branch $PACKAGES",
    ] {
        assert!(coverage.contains(needle), "coverage.yml lacks {needle:?}");
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
        // The commit the measurement was built on, for the baseline selection.
        "git rev-parse HEAD^1 > target/measured-base-sha",
        // HEAD^1 needs the parent: the default depth-1 checkout has none.
        "fetch-depth: 2",
        // Main is measured by main-measure.yml, so ci.yml builds no wheels there.
        "github.event_name != 'push'",
    ] {
        assert!(ci.contains(needle), "ci.yml lacks {needle:?}");
    }
    assert!(
        ci.contains(
            "if: github.event_name == 'pull_request' && needs.changes.outputs.rust == 'true'"
        ),
        "budgets-measure is for pull requests"
    );
    // Main measurement: every commit, never cancelled, 90 days.
    let main_measure = read(".github/workflows/main-measure.yml");
    for needle in [
        "branches: [main]",
        "group: main-measure-${{ github.sha }}",
        "cancel-in-progress: false",
        "cargo xtask budgets --build-time --measure-only",
        "name: budgets-main",
        "retention-days: 90",
        "uses: $/.github/workflows/wheels.yml",
        "actions: read",
        // The drift report never fails the run.
        "continue-on-error: true",
    ] {
        assert!(
            main_measure.contains(needle),
            "main-measure.yml lacks {needle:?}"
        );
    }
    assert!(!main_measure.contains("pull_request"));
    for label in ["budget-approved", "unsafe-reviewed", "size-approved"] {
        assert!(
            !ci.contains(&format!("approval-label.sh {label}")),
            "ci.yml must not resolve the {label} label (labels do not re-run it)"
        );
    }
    for needle in [
        "  budgets:\n",
        ".github/scripts/budgets-measurement.sh",
        ".github/scripts/main-baseline.sh",
        "ARTIFACT: budgets-main",
        "approval-label.sh budget-approved",
        "approval-label.sh unsafe-reviewed",
        "cargo xtask budgets --compare \"$MEASURED\" --baseline \"$BASELINE\"",
    ] {
        assert!(policy.contains(needle), "pr-policy.yml lacks {needle:?}");
    }
    // The baseline is main's measurement: the gates never take it from the PR's
    // checkout (a PR must not raise its own baseline), and the selection reads
    // the base commit's file only as the fallback.
    assert!(
        !policy.contains("git show FETCH_HEAD")
            && policy.contains(
                "if [ \"$MODE\" = main ]; then\n            cargo xtask budgets --compare"
            ),
        "the budgets gate must compare with an explicit baseline (legacy base aside)"
    );
    let select = read(".github/scripts/main-baseline.sh");
    for needle in [
        "TARGET_SHA_FILE",
        "git show \"$BASE_SHA:$fallback_path\"",
        "--event push",
        "--branch \"$branch\"",
    ] {
        assert!(select.contains(needle), "main-baseline.sh lacks {needle:?}");
    }
    // The selection runs on a full history and with the least privilege that
    // can list and download artifacts (`actions: read`).
    let at = |job: &str| policy.find(job).unwrap();
    let (budgets_at, wheels_at) = (at("  budgets:\n"), at("  wheel-sizes:\n"));
    for (job, body) in [
        ("budgets", &policy[budgets_at..wheels_at]),
        ("wheel-sizes", &policy[wheels_at..]),
    ] {
        assert!(body.contains("fetch-depth: 0"), "{job} needs full history");
        // The gate runs the base commit's scripts and xtask, never the PR's:
        // the only checkout is of the base sha, before anything is executed.
        assert_eq!(
            body.matches("actions/checkout@").count(),
            1,
            "{job} has one checkout"
        );
        let checkout = body.find("actions/checkout@").unwrap();
        let first_run = body.find("run:").unwrap();
        assert!(checkout < first_run);
        assert!(
            body[checkout..first_run].contains("ref: ${{ github.event.pull_request.base.sha }}"),
            "{job} must check out the base commit, not the pull request"
        );
        assert!(
            !body.contains("head.ref") && !body.contains("refs/pull/"),
            "{job} must not check out or run pull request code"
        );
        assert!(
            body.contains("TARGET_SHA_FILE"),
            "{job} selects by the measured base"
        );
        assert!(body.contains("actions: read"), "{job} needs actions: read");
        // Bootstrap: a base without the selection script or `--baseline` is
        // judged by its own old gate (committed budgets.json), and the
        // selection is skipped. No rust-cache, so a PR cannot seed the build.
        assert!(
            body.contains("id: gate")
                && body.contains("[ -f .github/scripts/main-baseline.sh ]")
                && body.contains("grep -qF '\"--baseline\"' xtask/src/budgets.rs")
                && body.contains("mode=legacy")
                && body.contains("steps.gate.outputs.mode == 'main'"),
            "{job} must fall back to the base's own gate when it lacks --baseline"
        );
        assert!(!body.contains("rust-cache"), "{job} must not use a cache");
        assert!(!body.contains("actions: write") && !body.contains("contents: write"));
    }
    // The gate finds the measurement by job and artifact name.
    let fetch = read(".github/scripts/budgets-measurement.sh");
    for needle in [
        "actions/workflows/ci.yml/runs",
        "job_name=\"${JOB:-budgets-measure}\"",
        "artifact=\"${ARTIFACT:-budgets-current}\"",
        "--name \"$artifact\"",
    ] {
        assert!(
            fetch.contains(needle),
            "budgets-measurement.sh lacks {needle:?}"
        );
    }
    // Wheel sizes: wheels.yml only measures on a pull request (and reads no
    // label); the label-aware gate in pr-policy.yml compares the measurement.
    let ci = read(".github/workflows/ci.yml");
    assert!(
        ci.contains("github.event_name != 'push' && github.event_name != 'merge_group'"),
        "wheels-targets is skipped on push and in the merge queue"
    );
    let wheels = read(".github/workflows/wheels.yml");
    for needle in [
        "cargo xtask wheel-sizes dist --measure-only",
        "'wheel-sizes-main' || 'wheel-sizes'",
        "retention-days: ${{ github.event_name == 'push' && 90 || 14 }}",
        "git rev-parse HEAD^1 > \"$RUNNER_TEMP/wheel-sizes/measured-base-sha\"",
        "fetch-depth: ${{ github.event_name == 'pull_request' && 2 || 0 }}",
        // Nightly and manual runs are gated on main's measurement; a push
        // only reports the drift.
        "WORKFLOW: main-measure.yml",
        "git merge-base HEAD origin/main",
        "continue-on-error: ${{ github.event_name == 'push' }}",
        "actions: read",
    ] {
        assert!(wheels.contains(needle), "wheels.yml lacks {needle:?}");
    }
    assert!(
        !wheels.contains("approval-label.sh") && !wheels.contains("HAS_LABEL"),
        "wheels.yml must not resolve labels from its (stale) event payload"
    );
    for needle in [
        "  wheel-sizes:\n",
        "JOB: wheels-targets / sizes",
        "SKIP_JOB: wheels-targets",
        "approval-label.sh budget-approved",
        "cargo xtask wheel-sizes --compare",
        "ARTIFACT: wheel-sizes-main",
        "flag=--baseline; else flag=--budgets",
    ] {
        assert!(policy.contains(needle), "pr-policy.yml lacks {needle:?}");
    }
    assert!(ci.contains("  wheels-targets:\n") && ci.contains("  wheels-ok:\n"));
    // Python diff coverage is gated in the required wheels job (python/krishiv)
    // and in the diff-coverage job (bench/src), each at 90% on its own.
    for needle in [
        "--cov-report=lcov:python-coverage.lcov",
        "--lcov python-coverage.lcov --fail-under 90",
        "fetch-depth: ${{ github.event_name == 'pull_request' && 2 || 1 }}",
    ] {
        assert!(ci.contains(needle), "ci.yml lacks {needle:?}");
    }
    let coverage = read(".github/workflows/coverage.yml");
    for needle in [
        "--cov-report=\"lcov:$RUNNER_TEMP/bench.lcov\"",
        "--lcov \"$RUNNER_TEMP/bench.lcov:bench\" --fail-under 90",
        "cargo xtask diff-coverage --base \"$BASE_SHA\" --head \"$HEAD_SHA\" --fail-under 90",
    ] {
        assert!(coverage.contains(needle), "coverage.yml lacks {needle:?}");
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

#[test]
fn main_baseline_script_selects_the_merge_base_measurement() {
    let script = root().join(".github/scripts/test-main-baseline.sh");
    let out = Command::new("bash").arg(&script).output().unwrap();
    assert!(
        out.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn large_file_guard_rejects_new_large_files() {
    let script = root().join(".github/scripts/test-check-large-files.sh");
    let out = Command::new("bash").arg(&script).output().unwrap();
    assert!(
        out.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let policy = read(".github/workflows/pr-policy.yml");
    assert!(policy.contains(".github/scripts/check-large-files.sh"));
    assert!(read(".githooks/pre-commit").contains("check-large-files.sh"));
}

#[test]
fn ai_attribution_guard_passes_its_tests_and_is_wired() {
    let script = root().join(".github/scripts/test-check-ai-attribution.sh");
    let out = Command::new("bash").arg(&script).output().unwrap();
    assert!(
        out.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let policy = read(".github/workflows/pr-policy.yml");
    assert!(
        policy.contains("run: .github/scripts/check-ai-attribution.sh \"$BASE_SHA\" \"$HEAD_SHA\""),
        "pr-policy.yml must run the attribution gate (a mention of the script is not enough)"
    );
    assert!(policy.contains("PR_BODY: ${{ github.event.pull_request.body }}"));
    assert!(read(".githooks/commit-msg").contains("check-ai-attribution.sh"));
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
    // The dependency-review allow list is deny.toml's plus exactly one deliberate extra:
    // CC0-1.0, allowed for Python dependencies only (numpy's compound license), never for
    // Rust crates (no patent grant in the engine binary). Any other difference fails.
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
    let mut expected = licenses.clone();
    expected.push("CC0-1.0");
    assert_eq!(
        listed, expected,
        "dependency-review allow-licenses must be deny.toml's list + CC0-1.0 (Python-only)"
    );
    assert!(
        !licenses.contains(&"CC0-1.0"),
        "CC0-1.0 must stay out of deny.toml's global allow list"
    );
    // Per-package exceptions, in two deliberate categories only: Rust crates, which must be
    // exactly deny.toml's `[licenses] exceptions`, and GitHub Actions used only in CI
    // (tooling on the build runner, never distributed or linked), which must really be used
    // by a workflow and never be a dependency of the workspace.
    let listed = sec
        .split("allow-dependencies-licenses: >-")
        .nth(1)
        .unwrap()
        .lines()
        .skip(1)
        .take_while(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ");
    let purls: Vec<&str> = listed.split(',').map(str::trim).collect();
    // The exceptions array, on one line or one entry per line.
    let exceptions = deny.split("\nexceptions = [").nth(1).unwrap();
    let exceptions = &exceptions[..exceptions
        .find("\n]")
        .or_else(|| exceptions.find("}]"))
        .unwrap()];
    let exception_crates: Vec<&str> = exceptions
        .split("crate = \"")
        .skip(1)
        .filter_map(|l| l.split('"').next())
        .collect();
    let cargo: Vec<&str> = purls
        .iter()
        .filter_map(|p| p.strip_prefix("pkg:cargo/"))
        .collect();
    assert_eq!(
        cargo, exception_crates,
        "pkg:cargo purls must be deny.toml's exceptions"
    );
    let workflows: String = std::fs::read_dir(root().join(".github/workflows"))
        .unwrap()
        .map(|e| std::fs::read_to_string(e.unwrap().path()).unwrap())
        .collect();
    for purl in &purls {
        if let Some(action) = purl.strip_prefix("pkg:githubactions/") {
            assert!(
                workflows.contains(&format!("uses: {action}@")),
                "{purl} is a CI-only exception but no workflow uses {action}"
            );
        } else {
            assert!(
                purl.starts_with("pkg:cargo/"),
                "unknown exception category: {purl}"
            );
        }
    }
    assert!(
        purls.contains(&"pkg:githubactions/Swatinem/rust-cache"),
        "the CI-only rust-cache exception is missing"
    );
    let osv = read(".github/workflows/osv-scanner.yml");
    // The scan walks the tree instead of listing lock files, so a lock
    // file a pull request adds does not fail the base scan.
    let blocks: Vec<&str> = osv.split("scan-args: |-").skip(1).collect();
    assert_eq!(
        blocks.len(),
        2,
        "osv-scanner.yml must have a scan-pr and a scan job"
    );
    for block in &blocks {
        for needle in ["--recursive", "--experimental-exclude=ui", "./"] {
            assert!(
                block.contains(needle),
                "osv scan job lacks {needle:?}: {block}"
            );
        }
    }
    assert!(
        !osv.contains("--lockfile="),
        "osv-scanner.yml must not list lock files"
    );
    assert!(osv.contains("osv-scanner-reusable-pr.yml@"));
    assert!(osv.contains("osv-scanner-reusable.yml@"));
    assert!(osv.contains("schedule:"));
}

/// Commands and actions that publish or release something.
const PUBLISH_MARKERS: [&str; 10] = [
    "gh-action-pypi-publish",
    "twine",
    "maturin publish",
    "maturin upload",
    "uv publish",
    "release-plz release",
    "action-gh-release",
    "gh release",
    "npm publish",
    "pnpm publish",
];

/// Nothing publishes on merge or on a tag (P2-T147, RELEASING.md): the only
/// publishing path is the TestPyPI workflow, which runs on manual dispatch
/// alone and waits for an approved environment; the release switches stay off
/// and every crate stays unpublishable.
#[test]
fn release_publishing_is_manual_and_gated() {
    let dir = root().join(".github");
    let mut files = Vec::new();
    for sub in ["workflows", "scripts"] {
        for entry in std::fs::read_dir(dir.join(sub)).unwrap() {
            files.push(entry.unwrap().path());
        }
    }
    let mut publishers = Vec::new();
    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let text = std::fs::read_to_string(path).unwrap();
        // Comments may talk about publishing; code may not.
        let code: Vec<&str> = text
            .lines()
            .filter(|l| !l.trim_start().starts_with('#'))
            .collect();
        for line in &code {
            // A dry run packages and verifies and never uploads.
            if line.contains("cargo publish") {
                assert!(
                    line.contains("--dry-run") && name == "publish-dry-run.sh",
                    "{name}: `cargo publish` outside the dry-run script or without --dry-run: {line}"
                );
            }
            if PUBLISH_MARKERS.iter().any(|m| line.contains(m)) {
                publishers.push(name.clone());
            }
        }
    }
    publishers.sort();
    publishers.dedup();
    assert_eq!(
        publishers,
        ["publish-testpypi.yml"],
        "only the gated TestPyPI workflow may publish"
    );

    let wf = read(".github/workflows/publish-testpypi.yml");
    let triggers: Vec<&str> = wf
        .lines()
        .skip_while(|l| *l != "on:")
        .skip(1)
        .take_while(|l| l.starts_with(' '))
        .map(str::trim)
        .collect();
    assert_eq!(
        triggers,
        ["workflow_dispatch:"],
        "the TestPyPI workflow must run on manual dispatch only"
    );
    // Scope the gate assertions to the `publish` job: a match anywhere else
    // in the file (another job, a comment) must not satisfy them.
    let publish = wf
        .find("\n  publish:\n")
        .map(|i| &wf[i..])
        .expect("publish-testpypi.yml has a `publish` job");
    let env = publish
        .find("environment:\n      name: testpypi")
        .expect("the publish job must declare the testpypi environment");
    let upload = publish
        .find("uses: pypa/gh-action-pypi-publish@")
        .expect("the publish job must upload via gh-action-pypi-publish");
    assert!(
        env < upload,
        "the publish job must declare the testpypi environment before the upload step"
    );
    assert!(wf.contains("github.ref == 'refs/heads/main'") && wf.contains("refs/tags/v"));
    // The upload step's effective `repository-url` must be TestPyPI: scoping
    // to the step's `with:` block keeps a production URL anywhere else (or
    // next to the TestPyPI one) from passing.
    let after_upload = &publish[upload..];
    let with_off = after_upload
        .find("\n        with:\n")
        .expect("the upload step must have a `with:` block");
    let with_block: Vec<&str> = after_upload[with_off..]
        .lines()
        .skip_while(|l| l.trim() != "with:")
        .skip(1)
        .take_while(|l| l.trim().is_empty() || l.starts_with("          "))
        .collect();
    let urls: Vec<&&str> = with_block
        .iter()
        .filter(|l| l.contains("repository-url:"))
        .collect();
    assert_eq!(
        urls.len(),
        1,
        "the upload step must set exactly one repository-url, got: {with_block:?}"
    );
    assert_eq!(
        urls[0].trim(),
        "repository-url: https://test.pypi.org/legacy/",
        "the upload step must target TestPyPI"
    );
    assert!(
        !publish.contains("//pypi.org/legacy/") && !publish.contains("upload.pypi.org"),
        "the publish job must not reference production PyPI"
    );
    assert!(
        !wf.contains("secrets."),
        "upload is by trusted publishing, no token"
    );

    // release-plz stays a dry run until the owner changes it.
    let plz = read("release-plz.toml");
    let code: Vec<&str> = plz
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect();
    for off in [
        "release = false",
        "publish = false",
        "git_release_enable = false",
        "git_tag_enable = false",
    ] {
        assert!(
            code.iter()
                .any(|l| l.split('#').next().unwrap().trim() == off),
            "release-plz.toml must keep `{off}`"
        );
    }
    for on in [
        "release = true",
        "publish = true",
        "git_release_enable = true",
        "git_tag_enable = true",
        "release_always = true",
    ] {
        assert!(
            !code.iter().any(|l| l.trim().starts_with(on)),
            "release-plz.toml turns on `{on}`"
        );
    }

    // No crate can be published from this repository.
    let mut manifests = vec![root().join("xtask/Cargo.toml")];
    for entry in std::fs::read_dir(root().join("crates")).unwrap() {
        manifests.push(entry.unwrap().path().join("Cargo.toml"));
    }
    for m in manifests.iter().filter(|m| m.exists()) {
        let text = std::fs::read_to_string(m).unwrap();
        assert!(
            text.lines().any(|l| l.trim() == "publish = false"),
            "{} must keep `publish = false`",
            m.display()
        );
    }

    // The dry run and the quickstart are wired into CI.
    let ci = read(".github/workflows/ci.yml");
    for needle in [
        "publish-dry-run.sh",
        "quickstart-test.sh",
        "  release-dry-run:\n",
    ] {
        assert!(ci.contains(needle), "ci.yml lacks {needle:?}");
    }
    // The README fixture uses `tempfile.mkdtemp()`; the script must point it
    // inside the scratch directory so the exit trap removes it.
    assert!(
        read(".github/scripts/quickstart-test.sh").contains("TMPDIR=\"$work\""),
        "quickstart-test.sh must run Python with TMPDIR=\"$work\""
    );
}

/// The `cargo test` commands of a loom job that do not select exactly the
/// `loom_model` modules (`--lib`, filter `loom_model::`). The `run:` value
/// is YAML-quoted (a trailing `::` is not a valid plain scalar), so
/// surrounding quotes are stripped before matching.
fn loom_steps_off_the_models(loom_job: &str) -> Vec<String> {
    loom_job
        .lines()
        .map(|l| l.trim().trim_start_matches("- "))
        .filter_map(|l| l.strip_prefix("run: "))
        .map(|cmd| {
            cmd.trim()
                .trim_matches(|c| c == '"' || c == '\'')
                .trim()
                .to_owned()
        })
        .filter(|cmd| cmd.starts_with("cargo test"))
        .filter(|cmd| {
            let words: Vec<&str> = cmd.split_whitespace().collect();
            let last = words
                .last()
                .map(|w| w.trim_matches(|c| c == '"' || c == '\''));
            !(words.contains(&"--lib") && last == Some("loom_model::"))
        })
        .collect()
}

/// The crates whose sources build a `loom::model`, each checked to keep its
/// models in a module named `loom_model`.
fn crates_with_loom_models() -> Vec<String> {
    fn walk(dir: &Path, files: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, files);
            } else if path.extension().is_some_and(|e| e == "rs") {
                files.push(path);
            }
        }
    }
    let mut crates = Vec::new();
    for entry in std::fs::read_dir(root().join("crates")).unwrap() {
        let krate = entry.unwrap().path();
        let src = krate.join("src");
        if !src.is_dir() {
            continue;
        }
        let mut files = Vec::new();
        walk(&src, &mut files);
        for file in files {
            let text = std::fs::read_to_string(&file).unwrap();
            if !text.contains("loom::model") {
                continue;
            }
            assert!(
                file.file_stem().is_some_and(|s| s == "loom_model")
                    || text.contains("mod loom_model {"),
                "{} builds a loom model outside a `loom_model` module",
                file.display()
            );
            let name = krate.file_name().unwrap().to_string_lossy().into_owned();
            if !crates.contains(&name) {
                crates.push(name);
            }
        }
    }
    crates.sort();
    crates
}

/// A loom primitive used outside `loom::model` panics, so the nightly loom
/// steps select only the `loom_model` modules, and every crate with loom
/// models has a step.
#[test]
fn loom_steps_select_only_loom_models() {
    let nightly = read(".github/workflows/nightly.yml");
    let loom = job(&nightly, "loom");
    assert_eq!(loom_steps_off_the_models(loom), Vec::<String>::new());
    let crates = crates_with_loom_models();
    assert!(!crates.is_empty(), "no loom models found");
    for krate in crates {
        let step = format!("run: \"cargo test -p {krate} --lib --release --locked loom_model::\"");
        assert!(loom.contains(&step), "the nightly loom job lacks {step:?}");
    }
}

#[test]
fn loom_step_check_rejects_name_filters_and_whole_crates() {
    let job = "    steps:\n      - run: rustup show\n      \
               - run: cargo test -p a --lib --release --locked runtime_filter\n      \
               - run: cargo test -p b --lib --release --locked\n      \
               - run: cargo test -p c --release --locked loom_model::\n      \
               - run: cargo test -p d --lib --release --locked loom_model::\n      \
               - run: \"cargo test -p e --lib --release --locked loom_model::\"\n      \
               - run: 'cargo test -p f --lib --release --locked loom_model::'\n      \
               - run: cargo test -p g --lib --release --locked \"loom_model::\"\n";
    assert_eq!(
        loom_steps_off_the_models(job),
        [
            "cargo test -p a --lib --release --locked runtime_filter",
            "cargo test -p b --lib --release --locked",
            "cargo test -p c --release --locked loom_model::",
        ]
    );
}
