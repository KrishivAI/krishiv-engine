// SPDX-License-Identifier: Apache-2.0
//! End-to-end tests of the policy subcommands against temporary fixtures.
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn xtask(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(args)
        .env_remove("SIZE_APPROVED")
        .output()
        .unwrap()
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("xtask-cmd-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
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

fn init_repo(name: &str) -> PathBuf {
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

    let big: String = (0..1600).map(|i| format!("line {i}\n")).collect();
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
