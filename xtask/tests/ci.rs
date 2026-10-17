// SPDX-License-Identifier: Apache-2.0
//! Tests for the PR size cap, path filters and DCO check.

use xtask::ci::{
    MAX_CHANGED_LINES, commits_missing_signoff, is_size_excluded, measure_numstat, rust_relevant,
};

fn numstat(path: &str, added: u64) -> String {
    format!("{added}\t0\t{path}\n")
}

#[test]
fn diff_of_1600_non_fixture_lines_fails_the_cap() {
    let report = measure_numstat(&numstat("crates/krishiv-exec/src/lib.rs", 1600)).unwrap();
    assert_eq!(report.counted, 1600);
    assert!(!report.within_cap());
}

#[test]
fn cap_boundary_is_inclusive() {
    let at = measure_numstat(&numstat("a.rs", MAX_CHANGED_LINES)).unwrap();
    assert!(at.within_cap());
    let over = measure_numstat(&numstat("a.rs", MAX_CHANGED_LINES + 1)).unwrap();
    assert!(!over.within_cap());
}

#[test]
fn goldens_fixtures_vendored_and_generated_are_excluded() {
    let input = [
        numstat("tests/spark-golden/functions/abs/v4.2.0/abs.out", 5000),
        numstat("tests/fixtures/spark-4.2/hash.json", 4000),
        numstat("Cargo.lock", 900),
        numstat("vendor/foo/src/lib.rs", 3000),
        numstat("crates/x/src/generated/ids.rs", 2000),
        numstat("crates/x/src/lib.rs", 10),
    ]
    .concat();
    let report = measure_numstat(&input).unwrap();
    assert_eq!(report.counted, 10);
    assert_eq!(report.excluded, 5000 + 4000 + 900 + 3000 + 2000);
    assert!(report.within_cap());
}

#[test]
fn deletions_count_binary_files_do_not_and_renames_resolve() {
    let input = "0\t700\tcrates/a.rs\n800\t0\tcrates/b.rs\n-\t-\tlogo.png\n5\t5\ttests/{old => fixtures}/data.json\n";
    let report = measure_numstat(input).unwrap();
    assert_eq!(report.counted, 1500);
    assert_eq!(report.excluded, 10);
    assert!(measure_numstat("garbage").is_err());
}

#[test]
fn allowlist_matches_expected_paths() {
    assert!(is_size_excluded("tests/sqllogictest/x.snap"));
    assert!(is_size_excluded("ui/pnpm-lock.yaml"));
    assert!(!is_size_excluded("crates/krishiv-sql/src/parser.rs"));
    assert!(!is_size_excluded("xtask/src/ci.rs"));
}

#[test]
fn docs_bench_and_plan_only_changes_skip_rust_jobs() {
    assert!(!rust_relevant(["docs/guide/intro.md", "docs/img/a.png"]));
    assert!(!rust_relevant(["bench/src/krishiv_bench/cli.py"]));
    assert!(!rust_relevant([
        "README.md",
        ".superpowers/sdd/x/report.md"
    ]));
    assert!(!rust_relevant(Vec::<&str>::new()));
    assert!(rust_relevant([
        "docs/a.md",
        "crates/krishiv-common/src/lib.rs"
    ]));
    assert!(rust_relevant(["Cargo.toml"]));
    assert!(rust_relevant([".github/workflows/ci.yml"]));
    assert!(rust_relevant([".github/pull_request_template.md"]));
    assert!(rust_relevant([".claude/agents/reviewer.md"]));
}

const GOOD: &str = "aaa\u{1f}dev@example.com\u{1f}feat: x\n\nSigned-off-by: Dev One <dev@example.com>\nCo-Authored-By: X <x@y>\n\u{1e}\n";
const BAD: &str = "bbb\u{1f}dev@example.com\u{1f}fix: y\n\nno trailer here\n\u{1e}\n";
const BOT: &str =
    "ccc\u{1f}1+renovate[bot]@users.noreply.github.com\u{1f}chore(deps): bump\n\u{1e}\n";

#[test]
fn dco_flags_unsigned_commits_only() {
    assert!(commits_missing_signoff(GOOD).is_empty());
    assert_eq!(commits_missing_signoff(&format!("{GOOD}{BAD}")), ["bbb"]);
    assert!(commits_missing_signoff(BOT).is_empty());
    let malformed = "ddd\u{1f}a@b.c\u{1f}x\n\nSigned-off-by: nobody\n\u{1e}\n";
    assert_eq!(commits_missing_signoff(malformed), ["ddd"]);
}
