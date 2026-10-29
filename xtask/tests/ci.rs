// SPDX-License-Identifier: Apache-2.0
//! Tests for the PR size cap, path filters and DCO check.

use xtask::ci::{
    MAX_CHANGED_LINES, TEST_LINE_DIVISOR, commits_missing_signoff, is_size_excluded, is_test_path,
    measure_numstat, rust_relevant,
};

fn numstat(path: &str, added: u64) -> String {
    format!("{added}\t0\t{path}\0")
}

#[test]
fn diff_over_the_cap_in_non_fixture_lines_fails() {
    let lines = MAX_CHANGED_LINES + 100;
    let report = measure_numstat(&numstat("crates/krishiv-exec/src/lib.rs", lines)).unwrap();
    assert_eq!(report.counted, lines);
    assert!(!report.within_cap());
}

#[test]
fn the_cap_is_ten_thousand_counted_lines() {
    assert_eq!(MAX_CHANGED_LINES, 10_000);
    assert!(
        measure_numstat(&numstat("a.rs", 10_000))
            .unwrap()
            .within_cap()
    );
    assert!(
        !measure_numstat(&numstat("a.rs", 10_001))
            .unwrap()
            .within_cap()
    );
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
        numstat("crates/x/vendored/spec.json", 2000),
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
    let deleted = MAX_CHANGED_LINES - 2300;
    let input = format!(
        "0\t{deleted}\tcrates/a.rs\x002300\t0\tcrates/b.rs\x00-\t-\tlogo.png\x005\t5\t\x00tests/old/data.json\x00tests/fixtures/data.json\x00"
    );
    let mut report = measure_numstat(&input).unwrap();
    assert_eq!(report.counted, MAX_CHANGED_LINES);
    assert_eq!(report.excluded, 10);
    assert_eq!(report.binaries, vec!["logo.png".to_owned()]);
    // A 6,401-byte binary counts as 101 lines and pushes the diff over the cap.
    report.add_binary(6401);
    assert_eq!(report.counted, MAX_CHANGED_LINES + 101);
    assert!(!report.within_cap());
    assert!(measure_numstat("garbage").is_err());
    // Paths are taken verbatim: no quoting, and ` => ` in a name is literal.
    let odd = measure_numstat("3\t0\tdocs/a => b.rs\x00").unwrap();
    assert_eq!(odd.counted, 3);
    assert!(measure_numstat("1\t1\t\x00only-old\x00").is_err());
}

#[test]
fn allowlist_matches_expected_paths() {
    assert!(is_size_excluded("tests/sqllogictest/x.snap"));
    assert!(is_size_excluded("ui/pnpm-lock.yaml"));
    assert!(!is_size_excluded("crates/krishiv-sql/src/parser.rs"));
    assert!(!is_size_excluded("xtask/src/ci.rs"));
    // Data directories count only at their fixed place under a crate.
    assert!(is_size_excluded("crates/krishiv-common/vendored/x.json"));
    assert!(is_size_excluded(
        "proto/spark-connect/4.2/spark/connect/base.proto"
    ));
    assert!(!is_size_excluded("proto/krishiv/v1/plan.proto"));
    assert!(is_size_excluded(
        "crates/krishiv-io/tests/fixtures/a.parquet"
    ));
    assert!(is_size_excluded(
        "python/krishiv/sql/_generated_functions.py"
    ));
    assert!(is_size_excluded(
        "python/krishiv/sql/_generated_functions.pyi"
    ));
    assert!(!is_size_excluded("python/krishiv/sql/_function_support.py"));
    assert!(!is_size_excluded("python/krishiv/sql/functions.py"));
    assert!(!is_size_excluded("crates/x/src/generated/ids.rs"));
    assert!(!is_size_excluded("crates/x/src/fixtures/helpers.rs"));
    assert!(!is_size_excluded("docs/notes.out"));
    assert!(!is_size_excluded("crates/x/src/source.map"));
    assert!(is_size_excluded(
        "crates/krishiv-ui-assets/assets/app.js.map"
    ));
}

#[test]
fn docs_bench_and_plan_only_changes_skip_rust_jobs() {
    assert!(!rust_relevant(["docs/guide/intro.md", "docs/img/a.png"]));
    assert!(!rust_relevant(["bench/src/krishiv_bench/cli.py"]));
    assert!(!rust_relevant([
        "CHANGELOG.md",
        ".superpowers/sdd/x/report.md"
    ]));
    assert!(!rust_relevant(Vec::<&str>::new()));
    assert!(rust_relevant([
        "docs/a.md",
        "crates/krishiv-common/src/lib.rs"
    ]));
    assert!(rust_relevant(["Cargo.toml"]));
    // The README's Quickstart is run by CI, so editing it runs the wheel jobs.
    assert!(rust_relevant(["README.md"]));
    assert!(!rust_relevant([
        "CHANGELOG.md",
        "RELEASING.md",
        "docs/README.md"
    ]));
    assert!(rust_relevant([".github/workflows/ci.yml"]));
    assert!(rust_relevant([".github/pull_request_template.md"]));
    assert!(rust_relevant([".claude/agents/reviewer.md"]));
    // Crate Markdown can be included into rustdoc.
    assert!(rust_relevant(["crates/krishiv-sql/README.md"]));
    assert!(rust_relevant(["xtask/README.md"]));
    assert!(!rust_relevant(["CONTRIBUTING.md", "docs/README.md"]));
}

const GOOD: &str = "aaa\u{1f}dev@example.com\u{1f}dev@example.com\u{1f}feat: x\n\nSigned-off-by: Dev One <dev@example.com>\nCo-Authored-By: X <x@y>\n\u{1e}\n";
const BAD: &str =
    "bbb\u{1f}dev@example.com\u{1f}dev@example.com\u{1f}fix: y\n\nno trailer here\n\u{1e}\n";
const BOT: &str = "ccc\u{1f}1+renovate[bot]@users.noreply.github.com\u{1f}noreply@github.com\u{1f}chore(deps): bump\n\u{1e}\n";
const FAKE_BOT: &str = "eee\u{1f}evil[bot]@example.com\u{1f}evil[bot]@example.com\u{1f}x\n\u{1e}\n";
const BOT_AUTHOR_HUMAN_COMMITTER: &str =
    "fff\u{1f}renovate[bot]@users.noreply.github.com\u{1f}dev@example.com\u{1f}x\n\u{1e}\n";

#[test]
fn dco_flags_unsigned_commits_only() {
    assert!(commits_missing_signoff(GOOD).is_empty());
    assert_eq!(commits_missing_signoff(&format!("{GOOD}{BAD}")), ["bbb"]);
    assert!(commits_missing_signoff(BOT).is_empty());
    let malformed = "ddd\u{1f}a@b.c\u{1f}a@b.c\u{1f}x\n\nSigned-off-by: nobody\n\u{1e}\n";
    assert_eq!(commits_missing_signoff(malformed), ["ddd"]);
}

#[test]
fn dco_bot_exemption_is_an_explicit_allowlist_and_checks_committer() {
    assert_eq!(commits_missing_signoff(FAKE_BOT), ["eee"]);
    assert_eq!(commits_missing_signoff(BOT_AUTHOR_HUMAN_COMMITTER), ["fff"]);
}

#[test]
fn license_is_not_counted_towards_the_size_cap() {
    assert!(is_size_excluded("LICENSE"));
    assert!(!is_size_excluded("NOTICE"));
}

#[test]
fn test_code_counts_at_reduced_weight() {
    assert_eq!(TEST_LINE_DIVISOR, 2);
    assert!(is_test_path("crates/krishiv-exec/tests/join.rs"));
    assert!(is_test_path("crates/krishiv-exec/src/aggregate/tests.rs"));
    assert!(is_test_path("crates/krishiv-io/src/parquet/write_tests.rs"));
    assert!(is_test_path("tests/scripts/gen_goldens.py"));
    assert!(is_test_path("python/tests/test_session.py"));
    assert!(!is_test_path("crates/krishiv-exec/src/join.rs"));
    assert!(!is_test_path("crates/krishiv-exec/src/testing.rs"));
    assert!(!is_test_path("xtask/src/ci.rs"));
    let input = [
        numstat("crates/x/src/lib.rs", MAX_CHANGED_LINES - 1000),
        numstat("crates/x/tests/it.rs", 1999),
        numstat("crates/x/src/tests.rs", 2),
    ]
    .concat();
    let report = measure_numstat(&input).unwrap();
    assert_eq!(report.test_lines, 2001);
    // (cap - 1,000) + ceil(2,001 / 2)
    assert_eq!(report.counted, MAX_CHANGED_LINES + 1);
    assert!(!report.within_cap());
    // Fixtures stay excluded even though they sit under `tests/`.
    let fixtures = measure_numstat(&numstat("tests/fixtures/a.json", 900)).unwrap();
    assert_eq!(
        (fixtures.counted, fixtures.test_lines, fixtures.excluded),
        (0, 0, 900)
    );
}
