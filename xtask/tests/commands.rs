// SPDX-License-Identifier: Apache-2.0
//! End-to-end tests of the policy subcommands against temporary fixtures.
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::path::Path;
use std::process::{Command, Output};

fn xtask(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(args)
        .env_remove("SIZE_APPROVED")
        .output()
        .unwrap()
}

/// A temporary directory, removed on drop (also when the test panics).
struct Tmp(tempfile::TempDir);

impl std::ops::Deref for Tmp {
    type Target = Path;
    fn deref(&self) -> &Path {
        self.0.path()
    }
}

impl AsRef<Path> for Tmp {
    fn as_ref(&self) -> &Path {
        self.0.path()
    }
}

fn tmp(name: &str) -> Tmp {
    Tmp(tempfile::Builder::new()
        .prefix(&format!("xtask-cmd-{name}-"))
        .tempdir()
        .unwrap())
}

fn write(root: &Path, rel: &str, text: &str) {
    let path = root.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn git(dir: &Path, args: &[&str]) {
    let out = Command::new("git")
        .args(args)
        .current_dir(dir)
        .env("GIT_AUTHOR_NAME", "T")
        .env("GIT_AUTHOR_EMAIL", "t@example.com")
        .env("GIT_COMMITTER_NAME", "T")
        .env("GIT_COMMITTER_EMAIL", "t@example.com")
        .output()
        .unwrap();
    assert!(out.status.success(), "git {args:?}: {out:?}");
}

fn init_repo(name: &str) -> Tmp {
    let dir = tmp(name);
    git(&dir, &["init", "-q", "-b", "main"]);
    git(&dir, &["config", "commit.gpgsign", "false"]);
    write(&dir, "README.md", "hi\n");
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "base"]);
    dir
}

#[test]
fn workflow_lint_fails_on_pull_request_target() {
    let dir = tmp("wf");
    write(&dir, "bad.yml", "on:\n  pull_request_target:\njobs: {}\n");
    let out = xtask(&["workflow-lint", dir.to_str().unwrap()]);
    assert!(!out.status.success());
    write(&dir, "bad.yml", "on:\n  pull_request:\njobs: {}\n");
    assert!(
        xtask(&["workflow-lint", dir.to_str().unwrap()])
            .status
            .success()
    );
}

#[test]
fn repo_workflows_pass_the_lint() {
    assert!(xtask(&["workflow-lint"]).status.success());
}

#[test]
fn spdx_fails_on_a_file_without_header() {
    let dir = tmp("spdx");
    write(
        &dir,
        "crates/a/src/lib.rs",
        "// SPDX-License-Identifier: Apache-2.0\n",
    );
    assert!(xtask(&["spdx", dir.to_str().unwrap()]).status.success());
    write(&dir, "crates/a/src/bad.rs", "fn main() {}\n");
    let out = xtask(&["spdx", dir.to_str().unwrap()]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("crates/a/src/bad.rs"));
}

#[test]
fn spdx_fails_when_the_tag_is_only_in_a_string_literal() {
    let dir = tmp("spdx-literal");
    write(
        &dir,
        "crates/a/src/lib.rs",
        "//! doc\npub const TAG: &str = \"SPDX-License-Identifier: Apache-2.0\";\n",
    );
    let out = xtask(&["spdx", dir.to_str().unwrap()]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("crates/a/src/lib.rs"));
}

#[test]
fn repo_sources_all_have_spdx_headers() {
    assert!(xtask(&["spdx"]).status.success());
}

#[test]
fn source_guard_fails_on_parquet_object_reader_fixture() {
    let dir = tmp("guard");
    write(&dir, "crates/a/src/lib.rs", "fn ok() {}\n");
    assert!(
        xtask(&["source-guard", dir.to_str().unwrap()])
            .status
            .success()
    );
    write(
        &dir,
        "crates/a/src/io.rs",
        "use parquet::arrow::async_reader::ParquetObjectReader;\n",
    );
    let out = xtask(&["source-guard", dir.to_str().unwrap()]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("ParquetObjectReader"));
}

#[test]
fn source_guard_confines_kernel_unsafe() {
    let dir = tmp("guard-unsafe");
    write(
        &dir,
        "crates/krishiv-kernels/src/unsafe_x.rs",
        "unsafe { f() }\n",
    );
    assert!(
        xtask(&["source-guard", dir.to_str().unwrap()])
            .status
            .success()
    );
    write(
        &dir,
        "crates/krishiv-kernels/src/x.rs",
        "fn a() { unsafe { f() } }\n",
    );
    let out = xtask(&["source-guard", dir.to_str().unwrap()]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("x.rs:1"));
}

#[test]
fn repo_sources_pass_the_source_guard() {
    assert!(xtask(&["source-guard"]).status.success());
}

#[test]
fn dco_check_on_a_real_repo() {
    let dir = init_repo("dco");
    let repo = dir.to_str().unwrap();
    write(&dir, "a.txt", "1\n");
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-s", "-m", "signed"]);
    assert!(
        xtask(&["dco", "--repo", repo, "--base", "main~1"])
            .status
            .success()
    );
    write(&dir, "b.txt", "2\n");
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "unsigned"]);
    assert!(
        !xtask(&["dco", "--repo", repo, "--base", "main~2"])
            .status
            .success()
    );
}

#[test]
fn pr_size_and_changes_on_a_real_repo() {
    let dir = init_repo("size");
    let repo = dir.to_str().unwrap();
    git(&dir, &["switch", "-q", "-c", "feature"]);
    write(&dir, "docs/a.md", "doc\n");
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "docs only"]);
    let changes = xtask(&["changes", "--repo", repo, "--base", "main"]);
    assert!(String::from_utf8_lossy(&changes.stdout).contains("rust=false"));

    let big: String = (0..3100).map(|i| format!("line {i}\n")).collect();
    write(&dir, "crates/x/src/big.rs", &big);
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "big"]);
    let changes = xtask(&["changes", "--repo", repo, "--base", "main"]);
    assert!(String::from_utf8_lossy(&changes.stdout).contains("rust=true"));
    assert!(
        !xtask(&["pr-size", "--repo", repo, "--base", "main"])
            .status
            .success()
    );
    let approved = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["pr-size", "--repo", repo, "--base", "main"])
        .env("SIZE_APPROVED", "1")
        .output()
        .unwrap();
    assert!(approved.status.success());
}

#[test]
fn big_fixture_diff_does_not_count() {
    let dir = init_repo("fixture");
    let repo = dir.to_str().unwrap();
    git(&dir, &["switch", "-q", "-c", "feature"]);
    let big: String = (0..1600).map(|i| format!("{i}\n")).collect();
    write(&dir, "tests/fixtures/data.json", &big);
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "fixture"]);
    assert!(
        xtask(&["pr-size", "--repo", repo, "--base", "main"])
            .status
            .success()
    );
}

#[test]
fn diff_coverage_gate_passes_and_fails_on_a_real_repo() {
    let dir = init_repo("diffcov");
    let repo = dir.to_str().unwrap();
    git(&dir, &["switch", "-q", "-c", "feature"]);
    let src: String = (1..=10).map(|i| format!("let x{i} = {i};\n")).collect();
    write(&dir, "crates/x/src/lib.rs", &src);
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "add"]);
    // Relative `SF:` paths match the diff's paths whatever the xtask root is.
    let lcov = |hit: u32| {
        let da: String = (1..=10)
            .map(|i| format!("DA:{i},{}\n", u32::from(i <= hit)))
            .collect();
        let path = dir.join(format!("lcov-{hit}.info"));
        std::fs::write(
            &path,
            format!("SF:crates/x/src/lib.rs\n{da}end_of_record\n"),
        )
        .unwrap();
        path.to_str().unwrap().to_owned()
    };
    let run = |lcov: &str, extra: &[&str]| {
        let mut args = vec![
            "diff-coverage",
            "--repo",
            repo,
            "--base",
            "main",
            "--lcov",
            lcov,
        ];
        args.extend_from_slice(extra);
        xtask(&args)
    };
    let nine = lcov(9);
    let eight = lcov(8);
    // Report only: never fails.
    let report = run(&eight, &[]);
    assert!(report.status.success());
    assert!(String::from_utf8_lossy(&report.stdout).contains("8 of 10"));
    // The gate: 9/10 meets 90%, 8/10 does not and names the uncovered lines.
    assert!(run(&nine, &["--fail-under", "90"]).status.success());
    let failed = run(&eight, &["--fail-under", "90"]);
    assert!(!failed.status.success());
    let out = String::from_utf8_lossy(&failed.stdout);
    assert!(out.contains("uncovered lines 9, 10"), "{out}");
    assert!(String::from_utf8_lossy(&failed.stderr).contains("below 90%"));
    // Bad thresholds are usage errors.
    for bad in ["101", "x"] {
        assert!(!run(&nine, &["--fail-under", bad]).status.success());
    }
    assert!(!run(&nine, &["--fail-under"]).status.success());
}

#[test]
fn diff_coverage_ignores_what_the_base_gained_after_the_branch_point() {
    let dir = init_repo("diffcov-behind");
    let repo = dir.to_str().unwrap();
    git(&dir, &["switch", "-q", "-c", "feature"]);
    write(&dir, "crates/x/src/lib.rs", "let a = 1;\nlet b = 2;\n");
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "feature"]);
    // `main` moves on with uncovered lines the branch never touched.
    git(&dir, &["switch", "-q", "main"]);
    let other: String = (1..=20).map(|i| format!("let y{i} = {i};\n")).collect();
    write(&dir, "crates/y/src/lib.rs", &other);
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "main moves on"]);
    git(&dir, &["switch", "-q", "feature"]);
    let lcov = dir.join("lcov.info");
    std::fs::write(
        &lcov,
        "SF:crates/x/src/lib.rs\nDA:1,1\nDA:2,1\nend_of_record\n\
         SF:crates/y/src/lib.rs\nDA:1,0\nend_of_record\n",
    )
    .unwrap();
    let out = xtask(&[
        "diff-coverage",
        "--repo",
        repo,
        "--base",
        "main",
        "--lcov",
        lcov.to_str().unwrap(),
        "--fail-under",
        "90",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("2 of 2"));
}

/// `budgets --compare FILE` (the CI gate) against the repository baseline.
#[test]
fn budgets_compare_gates_growth_on_the_approval_labels() {
    let dir = tmp("budgets");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let base: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("budgets.json")).unwrap()).unwrap();
    let measured = |edit: &dyn Fn(&mut serde_json::Value)| {
        let mut v = base.clone();
        edit(&mut v);
        let path = dir.join("m.json");
        std::fs::write(&path, v.to_string()).unwrap();
        path
    };
    let summary = dir.join("summary.md");
    let run = |file: &Path, env: &[(&str, &str)]| {
        let mut c = Command::new(env!("CARGO_BIN_EXE_xtask"));
        c.args(["budgets", "--compare", file.to_str().unwrap()])
            .env_remove("BUDGET_APPROVED")
            .env_remove("UNSAFE_APPROVED")
            .env("GITHUB_STEP_SUMMARY", &summary);
        for (k, v) in env {
            c.env(k, v);
        }
        c.output().unwrap()
    };
    // Unchanged: passes, and the table lands in the job summary.
    let same = measured(&|_| {});
    let out = run(&same, &[]);
    assert!(out.status.success(), "{out:?}");
    assert!(
        std::fs::read_to_string(&summary)
            .unwrap()
            .contains("| transitive crates |")
    );
    // Growth needs budget-approved; unsafe growth needs unsafe-reviewed.
    let grown = measured(&|v| {
        v["transitive_crates"] = (v["transitive_crates"].as_u64().unwrap() * 2).into();
    });
    let out = run(&grown, &[]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("budget-approved"));
    let out = run(&grown, &[("BUDGET_APPROVED", "1")]);
    assert!(out.status.success(), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("Refresh the baseline"));
    let unsafe_grown = measured(&|v| {
        v["unsafe_sites"]["krishiv-new"] = 1.into();
    });
    let out = run(&unsafe_grown, &[("BUDGET_APPROVED", "1")]);
    assert!(String::from_utf8_lossy(&out.stderr).contains("unsafe-reviewed"));
    assert!(
        run(&unsafe_grown, &[("UNSAFE_APPROVED", "1")])
            .status
            .success()
    );
    // A missing or malformed measurement is an error, never a pass.
    let out = run(&dir.join("missing.json"), &[]);
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot read"));
    std::fs::write(dir.join("bad.json"), "{").unwrap();
    let out = run(&dir.join("bad.json"), &[]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("bad "));
    // Flags that make no sense together are a usage error.
    assert!(
        !xtask(&["budgets", "--compare", "x", "--build-time"])
            .status
            .success()
    );
}

/// `budgets` measuring through a fake `cargo` (`$CARGO`): the measurement
/// is written for the CI artifact, and the default mode also compares it.
#[cfg(unix)]
#[test]
fn budgets_measure_writes_the_measurement_with_a_fake_cargo() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tmp("budgets-measure");
    let target = dir.join("target");
    let fake = dir.join("cargo");
    std::fs::write(
        &fake,
        r#"#!/bin/sh
case "$1" in
  metadata) printf '{"workspace_members":["w"],"resolve":{"nodes":[{"id":"w"},{"id":"dep"}]},"target_directory":"%s"}' "$FAKE_TARGET" ;;
  tree) printf 'krishiv-cli v0.1.0 (/w)\ndep v1.0.0\n' ;;
  build)
    [ -n "$FAKE_BUILD_FAIL" ] && exit 1
    while [ $# -gt 0 ]; do [ "$1" = --target-dir ] && d=$2; shift; done
    mkdir -p "$d/release" && printf 'bin' > "$d/release/krishiv" ;;
  *) exit 2 ;;
esac
"#,
    )
    .unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    let run = |args: &[&str], fail_build: bool| {
        let mut c = Command::new(env!("CARGO_BIN_EXE_xtask"));
        c.arg("budgets")
            .args(args)
            .env("CARGO", &fake)
            .env("FAKE_TARGET", &target)
            .env_remove("GITHUB_STEP_SUMMARY");
        if fail_build {
            c.env("FAKE_BUILD_FAIL", "1");
        }
        c.output().unwrap()
    };
    // --build-time measures in a fresh directory: a stale one is removed.
    std::fs::create_dir_all(target.join("budget-clean/stale")).unwrap();
    let out = run(&["--build-time", "--measure-only"], false);
    assert!(out.status.success(), "{out:?}");
    assert!(!target.join("budget-clean/stale").exists());
    let m: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(target.join("budgets-current.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(m["transitive_crates"], 1);
    assert_eq!(m["transitive_crates_all_features"], 1);
    assert_eq!(m["binary_bytes"], 3);
    assert!(m["clean_build_secs"].is_u64());
    assert!(m["unsafe_sites"].is_object());
    // Default mode: measure, then compare (everything shrank: passes).
    let out = run(&[], false);
    assert!(out.status.success(), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("## Budgets"));
    let out = run(&["--measure-only"], true);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("release build of krishiv-cli failed"));
}
