// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask file-size` on fixture workspaces: the ratchet on files over the line limit.
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::fs;
use std::path::Path;

use xtask::file_size::{BASELINE, read_baseline, run_file_size, scan};
use xtask::map::MAX_LINES;

fn write(root: &Path, rel: &str, text: &str) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, text).unwrap();
}

fn lines(n: usize) -> String {
    "// line\n".repeat(n)
}

/// A workspace with one short file, one file over the limit, a generated file over it and tests outside `src/`.
fn fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let r = dir.path();
    write(r, "crates/a/src/short.rs", &lines(10));
    write(r, "crates/a/src/long.rs", &lines(MAX_LINES + 300));
    write(r, "crates/a/src/gen.pb.rs", &lines(MAX_LINES + 900));
    write(
        r,
        "crates/a/src/marked.rs",
        &format!("// @generated\n{}", lines(MAX_LINES + 50)),
    );
    write(r, "crates/a/tests/big.rs", &lines(MAX_LINES + 500));
    write(r, "xtask/src/tool.rs", &lines(MAX_LINES + 1));
    dir
}

fn check(root: &Path) -> Result<(), String> {
    run_file_size(root, &[])
}

#[test]
fn the_scan_lists_hand_written_src_files_over_the_limit_only() {
    let dir = fixture();
    let found = scan(dir.path()).unwrap();
    let names: Vec<&str> = found.keys().map(String::as_str).collect();
    // Generated files and tests outside `src/` are not looked at; `xtask/src` is.
    assert_eq!(names, ["crates/a/src/long.rs", "xtask/src/tool.rs"]);
    assert_eq!(found["crates/a/src/long.rs"], MAX_LINES + 300);
}

#[test]
fn without_a_baseline_the_check_says_how_to_make_one() {
    let dir = fixture();
    let e = check(dir.path()).unwrap_err();
    assert!(e.contains("--write-baseline"), "{e}");
}

#[test]
fn a_baseline_written_from_the_tree_passes_and_round_trips() {
    let dir = fixture();
    run_file_size(dir.path(), &["--write-baseline"]).unwrap();
    assert!(dir.path().join(BASELINE).is_file());
    let b = read_baseline(dir.path()).unwrap();
    assert_eq!(b.max_lines, MAX_LINES);
    assert_eq!(b.files.len(), 2);
    check(dir.path()).unwrap();
}

#[test]
fn a_file_that_grew_fails_until_its_allowance_is_raised_by_hand() {
    let dir = fixture();
    run_file_size(dir.path(), &["--write-baseline"]).unwrap();
    write(dir.path(), "crates/a/src/long.rs", &lines(MAX_LINES + 301));
    assert!(check(dir.path()).unwrap_err().contains("1 finding"));
    // Raising the entry is a reviewed edit of the baseline.
    let text = fs::read_to_string(dir.path().join(BASELINE)).unwrap();
    let raised = text.replace(
        &(MAX_LINES + 300).to_string(),
        &(MAX_LINES + 301).to_string(),
    );
    write(dir.path(), BASELINE, &raised);
    check(dir.path()).unwrap();
}

#[test]
fn a_file_that_shrank_must_lower_its_allowance() {
    let dir = fixture();
    run_file_size(dir.path(), &["--write-baseline"]).unwrap();
    write(dir.path(), "crates/a/src/long.rs", &lines(MAX_LINES + 100));
    assert!(check(dir.path()).is_err());
    run_file_size(dir.path(), &["--write-baseline"]).unwrap();
    check(dir.path()).unwrap();
    // Down to the limit: no longer listed, and its entry has to go.
    write(dir.path(), "crates/a/src/long.rs", &lines(MAX_LINES));
    assert!(check(dir.path()).is_err());
    run_file_size(dir.path(), &["--write-baseline"]).unwrap();
    assert_eq!(read_baseline(dir.path()).unwrap().files.len(), 1);
    check(dir.path()).unwrap();
}

#[test]
fn a_new_file_over_the_limit_fails_and_a_generated_one_does_not() {
    let dir = fixture();
    run_file_size(dir.path(), &["--write-baseline"]).unwrap();
    write(
        dir.path(),
        "crates/a/src/other_gen.pb.rs",
        &lines(MAX_LINES * 2),
    );
    check(dir.path()).unwrap();
    write(
        dir.path(),
        "crates/a/src/new_long.rs",
        &lines(MAX_LINES + 1),
    );
    assert!(check(dir.path()).unwrap_err().contains("1 finding"));
}

#[test]
fn a_baseline_for_another_limit_and_unknown_flags_are_refused() {
    let dir = fixture();
    write(
        dir.path(),
        BASELINE,
        "{\"format_version\":1,\"max_lines\":999,\"files\":{}}",
    );
    assert!(check(dir.path()).unwrap_err().contains("limit"));
    assert!(run_file_size(dir.path(), &["--bogus"]).is_err());
}
