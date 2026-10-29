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
fn setup_installs_all_three_hooks() {
    let dir = tmp("setup");
    git(&dir, &["init", "-q", "-b", "main"]);
    for hook in ["pre-commit", "pre-push", "commit-msg"] {
        write(&dir, &format!(".githooks/{hook}"), "#!/bin/sh\n");
    }
    xtask::policy::run_setup(&dir).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for hook in ["pre-commit", "pre-push", "commit-msg"] {
            let mode = std::fs::metadata(dir.join(".githooks").join(hook))
                .unwrap()
                .permissions()
                .mode();
            assert!(mode & 0o111 != 0, "{hook} is not executable");
        }
    }
    let out = Command::new("git")
        .args(["config", "core.hooksPath"])
        .current_dir(&dir)
        .output()
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), ".githooks");
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
fn source_guard_rejects_blocking_calls_in_exec() {
    let dir = tmp("guard-blocking");
    write(&dir, "crates/krishiv-exec/src/lib.rs", "fn ok() {}\n");
    assert!(
        xtask(&["source-guard", dir.to_str().unwrap()])
            .status
            .success()
    );
    write(
        &dir,
        "crates/krishiv-exec/src/worker.rs",
        "fn idle() { std::thread::sleep(d); }\n",
    );
    let out = xtask(&["source-guard", dir.to_str().unwrap()]);
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("worker.rs:1"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // Test-only code is exempt: a `#[cfg(test)]` module and a `tests.rs`
    // file may block (they never run on a compute worker).
    write(
        &dir,
        "crates/krishiv-exec/src/worker.rs",
        "#[cfg(test)]\nmod tests { fn idle(rt: &Runtime) { rt.block_on(run()); } }\n",
    );
    write(
        &dir,
        "crates/krishiv-exec/src/tests.rs",
        "fn idle() { std::fs::read_to_string(p); }\n",
    );
    assert!(
        xtask(&["source-guard", dir.to_str().unwrap()])
            .status
            .success()
    );
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

    // Under the 10,000-line cap (and over the old 5,000 one): accepted without approval.
    let at_cap: String = (0..9_900).map(|i| format!("line {i}\n")).collect();
    write(&dir, "crates/x/src/big.rs", &at_cap);
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "at cap"]);
    assert!(
        xtask(&["pr-size", "--repo", repo, "--base", "main"])
            .status
            .success()
    );
    // One line over the cap: the size gate fails without approval.
    let big: String = (0..10_001).map(|i| format!("line {i}\n")).collect();
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
fn coverage_scope_selects_the_touched_packages_and_their_dependents() {
    let dir = tmp("scope");
    git(&dir, &["init", "-q", "-b", "main"]);
    git(&dir, &["config", "commit.gpgsign", "false"]);
    write(
        &dir,
        "Cargo.toml",
        "[workspace]\nresolver = \"2\"\nmembers = [\"crates/a\", \"crates/b\", \"crates/c\"]\n",
    );
    for (name, deps) in [("a", ""), ("b", "a = { path = \"../a\" }\n"), ("c", "")] {
        write(
            &dir,
            &format!("crates/{name}/Cargo.toml"),
            &format!(
                "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\n{deps}"
            ),
        );
        write(
            &dir,
            &format!("crates/{name}/src/lib.rs"),
            "pub fn f() {}\n",
        );
    }
    write(&dir, "crates/c/src/old.rs", "pub fn g() {}\n");
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "base"]);
    git(&dir, &["switch", "-q", "-c", "feature"]);
    let repo = dir.to_str().unwrap();
    let scope = |out_name: &str| {
        let out = dir.join(out_name);
        let o = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["coverage-scope", "--repo", repo, "--base", "main"])
            .env("GITHUB_OUTPUT", &out)
            .output()
            .unwrap();
        assert!(o.status.success(), "{o:?}");
        std::fs::read_to_string(out).unwrap()
    };

    // Docs and CI only: no Rust coverage run.
    write(&dir, "docs/a.md", "doc\n");
    write(&dir, ".github/workflows/x.yml", "name: x\n");
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "docs"]);
    assert_eq!(scope("o1"), "rust=false\npackages=\nworkspace=false\n");

    // A lock file and a shared fixture add no executable line.
    write(&dir, "Cargo.lock", "# lock\n");
    write(&dir, "tests/fixtures/x.json", "{}\n");
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "data"]);
    assert_eq!(scope("o2"), "rust=false\npackages=\nworkspace=false\n");

    // A source change: the package and the package that depends on it.
    write(
        &dir,
        "crates/a/src/lib.rs",
        "pub fn f() {}\npub fn g() {}\n",
    );
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "a"]);
    assert_eq!(
        scope("o3"),
        "rust=true\npackages=-p a -p b\nworkspace=false\n"
    );
    // A relative `--repo` selects the same packages (cargo reports absolute manifest paths).
    let relative = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(dir.parent().unwrap())
        .args([
            "coverage-scope",
            "--repo",
            dir.file_name().unwrap().to_str().unwrap(),
            "--base",
            "main",
        ])
        .output()
        .unwrap();
    assert!(relative.status.success(), "{relative:?}");
    assert!(
        String::from_utf8_lossy(&relative.stdout).contains("packages=-p a -p b"),
        "{relative:?}"
    );

    // A rename lists the old and the new path; a non-ASCII path stays verbatim.
    git(
        &dir,
        &["mv", "crates/c/src/old.rs", "crates/c/src/\u{e9}.rs"],
    );
    git(&dir, &["commit", "-q", "-am", "rename"]);
    assert_eq!(
        scope("o3b"),
        "rust=true\npackages=-p a -p b -p c\nworkspace=false\n"
    );

    // The gate's own workflow: every package.
    write(&dir, ".github/workflows/coverage.yml", "name: Coverage\n");
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "gate"]);
    assert_eq!(scope("o4"), "rust=true\npackages=\nworkspace=true\n");

    assert!(!xtask(&["coverage-scope", "--repo", repo]).status.success());
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
fn diff_coverage_merges_rust_and_python_tracefiles() {
    let dir = init_repo("diffcov-py");
    let repo = dir.to_str().unwrap();
    git(&dir, &["switch", "-q", "-c", "feature"]);
    let rs: String = (1..=5).map(|i| format!("let x{i} = {i};\n")).collect();
    write(&dir, "crates/x/src/lib.rs", &rs);
    // Six Python lines, line 3 a comment that coverage.py does not report.
    write(
        &dir,
        "bench/src/krishiv_bench/m.py",
        "a = 1\nb = 2\n# note\nc = 3\nd = 4\ne = 5\n",
    );
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "add"]);
    let rust = dir.join("rust.info");
    std::fs::write(
        &rust,
        "SF:crates/x/src/lib.rs\nDA:1,1\nDA:2,1\nDA:3,1\nDA:4,1\nDA:5,1\nend_of_record\n",
    )
    .unwrap();
    // coverage.py run from bench/: paths relative to that directory.
    let py = dir.join("py.info");
    std::fs::write(
        &py,
        "SF:src/krishiv_bench/m.py\nDA:1,1\nDA:2,1\nDA:4,1\nDA:5,0\nDA:6,0\nend_of_record\n",
    )
    .unwrap();
    let py_spec = format!("{}:bench", py.to_str().unwrap());
    let run = |extra: &[&str]| {
        let mut args = vec!["diff-coverage", "--repo", repo, "--base", "main"];
        args.extend_from_slice(extra);
        xtask(&args)
    };
    let both = run(&[
        "--lcov",
        rust.to_str().unwrap(),
        "--lcov",
        &py_spec,
        "--fail-under",
        "70",
    ]);
    let out = String::from_utf8_lossy(&both.stdout);
    // 5 Rust + 5 executable Python lines; 3 of the Python ones ran.
    assert!(both.status.success(), "{out}");
    assert!(out.contains("8 of 10"), "{out}");
    assert!(
        out.contains("bench/src/krishiv_bench/m.py`: uncovered lines 5, 6"),
        "{out}"
    );
    // Python alone is 3/5 = 60%: below 90, and the failure names its file.
    let alone = run(&["--lcov", &py_spec, "--fail-under", "90"]);
    assert!(!alone.status.success());
    assert!(String::from_utf8_lossy(&alone.stderr).contains("3 of 5"));
    // Without the prefix the relative path matches no diff path: nothing counted.
    let unmapped = run(&["--lcov", py.to_str().unwrap()]);
    assert!(String::from_utf8_lossy(&unmapped.stdout).contains("0 of 0"));
    // A bad prefix is a usage error.
    let bad = run(&["--lcov", &format!("{}:../x", py.to_str().unwrap())]);
    assert!(String::from_utf8_lossy(&bad.stderr).contains("plain repository-relative"));
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
    assert!(String::from_utf8_lossy(&out.stdout).contains("Nothing to record"));
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

/// `budgets --compare FILE --baseline MAIN`: the baseline is the given file
/// (a measurement of main), never the repository's `budgets.json`, so a PR is
/// gated only on its own growth.
#[test]
fn budgets_compare_with_an_explicit_baseline() {
    let dir = tmp("budgets-baseline");
    let write = |name: &str, crates: u64, unsafe_krishiv: u64| {
        let path = dir.join(name);
        std::fs::write(
            &path,
            serde_json::json!({
                "transitive_crates": crates,
                "binary_bytes": null,
                "clean_build_secs": null,
                "unsafe_sites": {"krishiv-x": unsafe_krishiv},
            })
            .to_string(),
        )
        .unwrap();
        path
    };
    // The repository's budgets.json is far from these numbers: it must not matter.
    let main = write("main.json", 1000, 3);
    let run = |measured: &Path, baseline: &Path, env: &[(&str, &str)]| {
        let mut c = Command::new(env!("CARGO_BIN_EXE_xtask"));
        c.args(["budgets", "--compare"])
            .arg(measured)
            .arg("--baseline")
            .arg(baseline)
            .env_remove("BUDGET_APPROVED")
            .env_remove("UNSAFE_APPROVED")
            .env_remove("GITHUB_STEP_SUMMARY");
        for (k, v) in env {
            c.env(k, v);
        }
        c.output().unwrap()
    };
    // Equal to main: passes although budgets.json differs.
    let out = run(&write("same.json", 1000, 3), &main, &[]);
    assert!(out.status.success(), "{out:?}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("| transitive crates | 1000 | 1000 |"));
    // Within 5% passes; 6% fails and names the label.
    assert!(run(&write("a.json", 1050, 3), &main, &[]).status.success());
    let out = run(&write("edge.json", 1051, 3), &main, &[]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("transitive crates: 1000 -> 1051"));
    let out = run(&write("b.json", 1060, 3), &main, &[]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("budget-approved"));
    assert!(
        run(
            &write("b.json", 1060, 3),
            &main,
            &[("BUDGET_APPROVED", "1")]
        )
        .status
        .success()
    );
    // Shrinking passes; unsafe growth over main needs its own label.
    assert!(run(&write("c.json", 900, 3), &main, &[]).status.success());
    let out = run(
        &write("d.json", 1000, 4),
        &main,
        &[("BUDGET_APPROVED", "1")],
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("unsafe-reviewed"));
    // A missing baseline is an error, never a pass.
    let out = run(&main, &dir.join("absent.json"), &[]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot read"));
    // `--baseline` belongs to `--compare` only.
    assert!(
        !xtask(&["budgets", "--baseline", main.to_str().unwrap()])
            .status
            .success()
    );
}

/// `wheel-sizes --compare FILE --baseline MAIN`: another measurement is the
/// baseline; `--budgets` and `--baseline` are exclusive.
#[test]
fn wheel_sizes_compare_with_a_measured_baseline() {
    let dir = tmp("wheel-sizes-baseline");
    let write = |name: &str, bytes: u64| {
        let path = dir.join(name);
        std::fs::write(
            &path,
            format!(r#"{{"cp311-abi3-manylinux_2_28_x86_64": {bytes}}}"#),
        )
        .unwrap();
        path
    };
    let main = write("main.json", 1000);
    let run = |measured: &Path, extra: &[&str], approved: bool| {
        let mut c = Command::new(env!("CARGO_BIN_EXE_xtask"));
        c.args(["wheel-sizes", "--compare"])
            .arg(measured)
            .args(extra)
            .env_remove("GITHUB_STEP_SUMMARY")
            .env_remove("BUDGET_APPROVED");
        if approved {
            c.env("BUDGET_APPROVED", "1");
        }
        c.output().unwrap()
    };
    let base = ["--baseline", main.to_str().unwrap()];
    assert!(
        run(&write("same.json", 1000), &base, false)
            .status
            .success()
    );
    assert!(
        run(&write("small.json", 1050), &base, false)
            .status
            .success()
    );
    let out = run(&write("edge.json", 1051), &base, false);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("grew from 1000 to 1051"));
    let out = run(&write("big.json", 1100), &base, false);
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("grew from 1000 to 1100"),
        "{out:?}"
    );
    assert!(run(&write("big.json", 1100), &base, true).status.success());
    // A wheel main has no size for is reported as new, as with budgets.json.
    std::fs::write(
        dir.join("other.json"),
        r#"{"cp311-abi3-macosx_11_0_arm64": 5}"#,
    )
    .unwrap();
    let out = run(&dir.join("other.json"), &base, false);
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("no recorded size"),
        "{out:?}"
    );
    // Errors: empty or missing baseline, both baseline flags.
    std::fs::write(dir.join("empty.json"), "{}").unwrap();
    assert!(
        !run(
            &main,
            &["--baseline", dir.join("empty.json").to_str().unwrap()],
            false
        )
        .status
        .success()
    );
    assert!(
        !run(
            &main,
            &["--baseline", dir.join("none.json").to_str().unwrap()],
            false
        )
        .status
        .success()
    );
    let both = [
        "--baseline",
        main.to_str().unwrap(),
        "--budgets",
        main.to_str().unwrap(),
    ];
    assert!(!run(&main, &both, false).status.success());
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
    mkdir -p "$d/dist" && printf 'bin' > "$d/dist/krishiv" ;;
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
            // The unsafe-site gate compares the real workspace with
            // budgets.json, which a PR adding reviewed `unsafe` changes; it
            // has its own test, so this one, about the measurement, approves it.
            .env("UNSAFE_APPROVED", "1")
            .env_remove("GITHUB_STEP_SUMMARY")
            // The measurement must land in FAKE_TARGET, whatever the caller exports.
            .env_remove("CARGO_TARGET_DIR");
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
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("dist-profile build of krishiv-cli failed")
    );
}

/// `wheel-sizes`: records the wheel sizes, then gates growth past the
/// budget and wheels with no recorded size (`BUDGET_APPROVED=1` approves).
#[test]
fn wheel_sizes_record_and_gate() {
    let dir = tmp("wheel-sizes");
    let wheels = dir.join("dist");
    std::fs::create_dir_all(&wheels).unwrap();
    let budgets = dir.join("budgets.json");
    std::fs::write(
        &budgets,
        r#"{"transitive_crates": 1, "binary_bytes": null, "clean_build_secs": null}"#,
    )
    .unwrap();
    let wheel = wheels.join("krishiv-0.1.0-cp311-abi3-manylinux_2_28_x86_64.whl");
    std::fs::write(&wheel, vec![0_u8; 1000]).unwrap();
    std::fs::write(wheels.join("README.txt"), "ignored").unwrap();
    let run = |extra: &[&str], approved: bool| {
        let mut c = Command::new(env!("CARGO_BIN_EXE_xtask"));
        c.arg("wheel-sizes")
            .arg(&wheels)
            .arg("--budgets")
            .arg(&budgets)
            .args(extra)
            .env_remove("GITHUB_STEP_SUMMARY")
            .env_remove("BUDGET_APPROVED");
        if approved {
            c.env("BUDGET_APPROVED", "1");
        }
        c.output().unwrap()
    };
    // No baseline at all yet: the gate only reports.
    let out = run(&[], false);
    assert!(out.status.success(), "{out:?}");
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("no baseline yet"),
        "{out:?}"
    );
    // Record, then the same wheel passes.
    assert!(run(&["--write"], false).status.success());
    let recorded: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&budgets).unwrap()).unwrap();
    assert_eq!(
        recorded["wheel_bytes"]["cp311-abi3-manylinux_2_28_x86_64"],
        1000
    );
    assert_eq!(recorded["transitive_crates"], 1);
    // Recording a subset keeps the other targets' entries.
    let other = dir.join("other");
    std::fs::create_dir_all(&other).unwrap();
    std::fs::write(
        other.join("krishiv-0.1.0-cp314-cp314t-macosx_11_0_arm64.whl"),
        vec![0_u8; 7],
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .arg("wheel-sizes")
        .arg(&other)
        .arg("--budgets")
        .arg(&budgets)
        .arg("--write")
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    let merged: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&budgets).unwrap()).unwrap();
    assert_eq!(
        merged["wheel_bytes"]["cp311-abi3-manylinux_2_28_x86_64"],
        1000
    );
    assert_eq!(merged["wheel_bytes"]["cp314-cp314t-macosx_11_0_arm64"], 7);
    let out = run(&[], false);
    assert!(out.status.success(), "{out:?}");
    assert!(
        String::from_utf8_lossy(&out.stdout)
            .contains("| cp311-abi3-manylinux_2_28_x86_64 | 1000 | 1000 | +0.00% |")
    );
    // 10% larger: over budget.
    std::fs::write(&wheel, vec![0_u8; 1100]).unwrap();
    let out = run(&[], false);
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("grew from 1000 to 1100"),
        "{out:?}"
    );
    assert!(run(&[], true).status.success(), "approved growth passes");
    std::fs::write(&wheel, vec![0_u8; 1000]).unwrap();
    // A wheel with no recorded size once a baseline exists: a violation.
    let new_target = wheels.join("krishiv-0.1.0-cp311-abi3-musllinux_1_2_x86_64.whl");
    std::fs::write(&new_target, "x").unwrap();
    let out = run(&[], false);
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("no recorded size"),
        "{out:?}"
    );
    std::fs::remove_file(&new_target).unwrap();
    // Measure once, gate later (what the wheels workflow and the pr-policy
    // gate do): --measure-only gates nothing, --compare gates the file.
    // The output directory need not exist (CI points it into a fresh temp dir).
    let nested = dir.join("not/yet/there/sizes.json");
    let out = run(&["--measure-only", nested.to_str().unwrap()], false);
    assert!(out.status.success(), "{out:?}");
    assert!(nested.is_file());
    let sizes = dir.join("sizes.json");
    let out = run(&["--measure-only", sizes.to_str().unwrap()], false);
    assert!(out.status.success(), "{out:?}");
    let compare = |approved: bool| {
        let mut c = Command::new(env!("CARGO_BIN_EXE_xtask"));
        c.args(["wheel-sizes", "--compare"])
            .arg(&sizes)
            .arg("--budgets")
            .arg(&budgets)
            .env_remove("GITHUB_STEP_SUMMARY")
            .env_remove("BUDGET_APPROVED");
        if approved {
            c.env("BUDGET_APPROVED", "1");
        }
        c.output().unwrap()
    };
    assert!(compare(false).status.success());
    std::fs::write(&wheel, vec![0_u8; 1100]).unwrap();
    let grown = dir.join("grown.json");
    assert!(
        run(&["--measure-only", grown.to_str().unwrap()], false)
            .status
            .success(),
        "measuring never fails on growth"
    );
    std::fs::write(&wheel, vec![0_u8; 1000]).unwrap();
    std::fs::rename(&grown, &sizes).unwrap();
    let out = compare(false);
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("grew from 1000 to 1100"),
        "{out:?}"
    );
    assert!(compare(true).status.success(), "approved growth passes");
    std::fs::write(&sizes, "{}").unwrap();
    assert!(!compare(false).status.success(), "empty measurement fails");
    std::fs::write(&sizes, "nope").unwrap();
    assert!(!compare(false).status.success(), "bad measurement fails");
    std::fs::remove_file(&sizes).unwrap();
    assert!(
        !compare(false).status.success(),
        "missing measurement fails"
    );
    assert!(!run(&["--measure-only"], false).status.success());
    assert!(
        !run(&["--measure-only", "x.json", "--write"], false)
            .status
            .success()
    );
    // Usage and input errors.
    assert!(!run(&["--bogus"], false).status.success());
    assert!(!run(&["--budgets"], false).status.success());
    std::fs::write(wheels.join("bad.whl"), "x").unwrap();
    let out = run(&[], false);
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("not a wheel file name"),
        "{out:?}"
    );
    std::fs::remove_file(wheels.join("bad.whl")).unwrap();
    std::fs::write(
        wheels.join("krishiv-0.2.0-cp311-abi3-manylinux_2_28_x86_64.whl"),
        "x",
    )
    .unwrap();
    let out = run(&[], false);
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("two wheels"),
        "{out:?}"
    );
    let empty = dir.join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .arg("wheel-sizes")
        .arg(&empty)
        .arg("--budgets")
        .arg(&budgets)
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("no wheels"),
        "{out:?}"
    );
}

/// `budgets --cache-main` + `budgets --precheck`: the local equivalent of
/// CI's measure/compare split, through a fake `cargo` and a throwaway repo.
/// Without a cached main measurement the step skips (never a false failure);
/// with one it gates the delta with the CI rule. The throwaway repo has no
/// `budgets.json` at all, so a passing run also proves the stale absolute
/// baseline is never read.
#[cfg(unix)]
#[test]
fn budgets_precheck_gates_the_delta_against_the_cached_main() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tmp("budgets-precheck");
    git(&dir, &["init", "-q", "-b", "main"]);
    write(&dir, "README.md", "hi\n");
    // The unsafe-site scan reads `crates/`; an empty one measures nothing.
    std::fs::create_dir_all(dir.join("crates")).unwrap();
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "base"]);
    git(&dir, &["switch", "-q", "-c", "feature"]);
    write(&dir, "feature.txt", "work\n");
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "work"]);
    let repo = dir.to_str().unwrap().to_owned();

    // A fixed measurement (1 transitive crate, a 10-byte binary), whatever
    // the checkout holds; the target directory comes from `FAKE_TARGET`.
    let target = dir.join("target");
    let fake = dir.join("cargo");
    std::fs::write(
        &fake,
        r#"#!/bin/sh
case "$1" in
  metadata) printf '{"workspace_members":["w"],"resolve":{"nodes":[{"id":"w"},{"id":"dep"}]},"target_directory":"%s"}' "$FAKE_TARGET" ;;
  tree) printf 'krishiv-cli v0.1.0 (/w)\ndep v1.0.0\n' ;;
  build)
    while [ $# -gt 0 ]; do [ "$1" = --target-dir ] && d=$2; shift; done
    mkdir -p "$d/dist" && printf '0123456789' > "$d/dist/krishiv" ;;
  *) exit 2 ;;
esac
"#,
    )
    .unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    let sha_of = |rev: &str| {
        let out = Command::new("git")
            .args(["rev-parse", rev])
            .current_dir(&dir)
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
        String::from_utf8_lossy(&out.stdout).trim().to_owned()
    };
    let run = |args: &[&str], approved: bool| {
        let mut c = Command::new(env!("CARGO_BIN_EXE_xtask"));
        c.arg("budgets")
            .args(args)
            .env("CARGO", &fake)
            .env("FAKE_TARGET", &target)
            .env("CARGO_TARGET_DIR", &target)
            .env_remove("GITHUB_STEP_SUMMARY")
            .env_remove("BUDGET_APPROVED")
            .env_remove("UNSAFE_APPROVED");
        if approved {
            c.env("BUDGET_APPROVED", "1");
        }
        c.output().unwrap()
    };
    let precheck = ["--precheck", "--repo", &repo, "--base", "main"];

    // No cached measurement yet: skip, never fail.
    let out = run(&precheck, false);
    assert!(out.status.success(), "{out:?}");
    assert!(
        String::from_utf8_lossy(&out.stdout)
            .contains("budgets: skipped (no main baseline; CI checks it)"),
        "{out:?}"
    );
    // An unknown base is an error, not a silent skip.
    let out = run(
        &["--precheck", "--repo", &repo, "--base", "no-such-ref"],
        false,
    );
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("merge-base"),
        "{out:?}"
    );

    // Cache the main measurement (as on a main checkout), then switch back:
    // the feature branch's merge-base is that main commit.
    git(&dir, &["switch", "-q", "main"]);
    let out = run(&["--cache-main", "--repo", &repo], false);
    assert!(out.status.success(), "{out:?}");
    let cache = target
        .join("precheck")
        .join(format!("budgets-main-{}.json", sha_of("HEAD")));
    assert!(cache.is_file(), "cache-main wrote {cache:?}");
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("cached the main measurement"),
        "{out:?}"
    );
    // Back on the branch: the delta against the cached main passes. The
    // feature commit only adds a text file outside `crates/`, which the
    // measurement does not see.
    git(&dir, &["switch", "-q", "feature"]);
    let out = run(&precheck, false);
    assert!(out.status.success(), "{out:?}");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(stdout.contains("| transitive crates | 1 | 1 |"), "{out:?}");
    assert!(
        stdout.contains("| dist binary bytes | 10 | 10 |"),
        "{out:?}"
    );

    // Growth over the cached main fails without the label, passes with it.
    let mut measured: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&cache).unwrap()).unwrap();
    measured["binary_bytes"] = serde_json::json!(9);
    std::fs::write(&cache, serde_json::to_string_pretty(&measured).unwrap()).unwrap();
    let out = run(&precheck, false);
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("dist binary bytes: 9 -> 10"),
        "{out:?}"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("budget-approved"),
        "{out:?}"
    );
    assert!(
        run(&precheck, true).status.success(),
        "approved growth passes"
    );
    // A corrupt cache is an error, never a silent pass.
    std::fs::write(&cache, "{").unwrap();
    let out = run(&precheck, false);
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("bad "),
        "{out:?}"
    );
    // Flag soup is a usage error.
    assert!(
        !run(&["--precheck", "--compare", "x"], false)
            .status
            .success()
    );
}

/// `budgets --cache-main` refuses a dirty tree: the cache is keyed by
/// `HEAD`, so measuring with uncommitted changes would file a measurement
/// that does not match the commit. Untracked `target/` output never counts.
#[test]
fn budgets_cache_main_rejects_a_dirty_working_tree() {
    let dir = init_repo("budgets-dirty");
    let repo = dir.to_str().unwrap();
    // Tracked modification without a commit: the gate fails before measuring.
    write(&dir, "README.md", "dirty\n");
    let out = xtask(&["budgets", "--cache-main", "--repo", repo]);
    assert!(!out.status.success(), "{out:?}");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        stderr.contains("uncommitted changes") && stderr.contains("commit or stash"),
        "{out:?}"
    );
}

/// The real `git`, for fake-`git` scripts to delegate to.
fn real_git() -> std::path::PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default())
        .map(|d| d.join("git"))
        .find(|p| p.is_file())
        .unwrap()
}

/// A `git` that answers `subcommand` with `answer` and delegates
/// everything else to the real one, first on `PATH` for the child.
#[cfg(unix)]
fn fake_git(bin: &Path, subcommand: &str, answer: &str) {
    use std::os::unix::fs::PermissionsExt;
    write(
        bin,
        "git",
        &format!(
            "#!/bin/sh\nif [ \"$1\" = \"{subcommand}\" ]; then printf '%s' \"{answer}\"; exit 0; fi\nexec \"$REAL_GIT\" \"$@\"\n"
        ),
    );
    std::fs::set_permissions(bin.join("git"), std::fs::Permissions::from_mode(0o755)).unwrap();
}

fn xtask_with_path(args: &[&str], bin: &Path) -> Output {
    let path = format!(
        "{}:{}",
        bin.to_str().unwrap(),
        std::env::var("PATH").unwrap_or_default()
    );
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(args)
        .env("PATH", path)
        .env("REAL_GIT", real_git())
        .output()
        .unwrap()
}

/// `budgets --precheck` rejects a merge-base that is not a commit id: the
/// cache lookup must never escape its directory.
#[cfg(unix)]
#[test]
fn budgets_precheck_rejects_an_unsafe_merge_base() {
    let dir = init_repo("budgets-bad-base");
    let repo = dir.to_str().unwrap().to_owned();
    let bin = dir.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    fake_git(&bin, "merge-base", "../evil");
    let out = xtask_with_path(
        &["budgets", "--precheck", "--repo", &repo, "--base", "main"],
        &bin,
    );
    assert!(!out.status.success(), "{out:?}");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("bad merge-base"),
        "{out:?}"
    );
}

/// `budgets --cache-main` rejects a `HEAD` id that is not a commit id.
#[cfg(unix)]
#[test]
fn budgets_cache_main_rejects_an_unsafe_head_id() {
    let dir = init_repo("budgets-bad-head");
    let repo = dir.to_str().unwrap().to_owned();
    let bin = dir.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    fake_git(&bin, "rev-parse", "not a sha");
    let out = xtask_with_path(&["budgets", "--cache-main", "--repo", &repo], &bin);
    assert!(!out.status.success(), "{out:?}");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("bad HEAD id"),
        "{out:?}"
    );
}

/// `budgets --cache-main` in a repository without a Rust project reports
/// the cargo failure instead of caching anything.
#[test]
fn budgets_cache_main_reports_a_cargo_metadata_failure() {
    let dir = init_repo("budgets-no-cargo");
    let repo = dir.to_str().unwrap();
    let out = xtask(&["budgets", "--cache-main", "--repo", repo]);
    assert!(!out.status.success(), "{out:?}");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("cargo metadata failed"),
        "{out:?}"
    );
}

/// `budgets --cache-main` reports a cache write failure: measuring may
/// succeed while filing the measurement fails.
#[cfg(unix)]
#[test]
fn budgets_cache_main_reports_an_unwritable_cache() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tmp("budgets-cache-unwritable");
    git(&dir, &["init", "-q", "-b", "main"]);
    write(&dir, "README.md", "hi\n");
    // The unsafe-site scan reads `crates/`; an empty one measures nothing.
    std::fs::create_dir_all(dir.join("crates")).unwrap();
    git(&dir, &["add", "."]);
    git(&dir, &["commit", "-q", "-m", "base"]);
    let repo = dir.to_str().unwrap().to_owned();
    // The cache directory as a file: measuring succeeds, filing it fails.
    let target = dir.join("target");
    std::fs::create_dir_all(&target).unwrap();
    std::fs::write(target.join("precheck"), "blocker\n").unwrap();
    let fake = dir.join("cargo");
    std::fs::write(
        &fake,
        r#"#!/bin/sh
case "$1" in
  metadata) printf '{"workspace_members":["w"],"resolve":{"nodes":[{"id":"w"},{"id":"dep"}]},"target_directory":"%s"}' "$FAKE_TARGET" ;;
  tree) printf 'krishiv-cli v0.1.0 (/w)\ndep v1.0.0\n' ;;
  build)
    while [ $# -gt 0 ]; do [ "$1" = --target-dir ] && d=$2; shift; done
    mkdir -p "$d/dist" && printf '0123456789' > "$d/dist/krishiv" ;;
  *) exit 2 ;;
esac
"#,
    )
    .unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["budgets", "--cache-main", "--repo", &repo])
        .env("CARGO", &fake)
        .env("FAKE_TARGET", &target)
        .env("CARGO_TARGET_DIR", &target)
        .env_remove("GITHUB_STEP_SUMMARY")
        .output()
        .unwrap();
    assert!(!out.status.success(), "{out:?}");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("cannot create"),
        "{out:?}"
    );
}
