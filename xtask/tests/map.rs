// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask map` and `maps-check` (P2-T193) on fixture crates.
#![expect(clippy::unwrap_used, reason = "test helpers sit outside #[test] fns")]

use std::fs;
use std::path::Path;

use xtask::map::{MAX_LINES, check_crate, run_map, run_maps_check, scan_crate};

fn write(root: &Path, rel: &str, text: &str) {
    let path = root.join(rel);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, text).unwrap();
}

/// A workspace with a clean crate and a crate with each kind of finding.
fn fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write(
        root,
        "crates/clean/src/lib.rs",
        "//! A clean crate.\n/// Adds.\npub fn add() {}\n",
    );
    write(root, "crates/clean/src/inner/mod.rs", "//! Inner.\n");
    write(
        root,
        "crates/clean/CLAUDE.md",
        "# clean\n## Layout\n`lib.rs` the API · `inner/` · `tests/fixtures/x.json`\n",
    );
    write(
        root,
        "crates/bad/src/lib.rs",
        "//! A crate with findings.\n",
    );
    write(root, "crates/bad/fuzz/Cargo.toml", "[package]\n");
    write(root, "crates/bad/src/nodoc.rs", "// SPDX\npub fn f() {}\n");
    let long = format!("//! Long.\n{}", "// line\n".repeat(MAX_LINES));
    write(root, "crates/bad/src/long.rs", &long);
    write(
        root,
        "crates/bad/src/gen.pb.rs",
        &"// generated\n".repeat(MAX_LINES + 1),
    );
    write(
        root,
        "crates/bad/CLAUDE.md",
        "# bad\n## Layout\n`lib.rs` · `ghost.rs` · `missing/` · `gen.pb.rs` · `fuzz/`\n",
    );
    dir
}

#[test]
fn a_clean_crate_has_no_findings() {
    let dir = fixture();
    let found = check_crate(&dir.path().join("crates/clean")).unwrap();
    assert_eq!(found.count(), 0, "{found:?}");
}

#[test]
fn each_kind_of_finding_is_detected() {
    let dir = fixture();
    let found = check_crate(&dir.path().join("crates/bad")).unwrap();
    assert_eq!(found.missing_docs, ["nodoc.rs"]);
    assert_eq!(found.stale_names, ["ghost.rs", "missing/"]);
    assert_eq!(found.long_files, [("long.rs".to_owned(), MAX_LINES + 1)]);
}

#[test]
fn the_command_fails_on_findings_unless_it_only_reports() {
    let dir = fixture();
    let e = run_maps_check(dir.path(), &[]).unwrap_err();
    assert!(e.contains("4 finding(s)"), "{e}");
    run_maps_check(dir.path(), &["--report"]).unwrap();
    assert!(run_maps_check(dir.path(), &["--other"]).is_err());
    fs::remove_dir_all(dir.path().join("crates/bad")).unwrap();
    run_maps_check(dir.path(), &[]).unwrap();
}

#[test]
fn the_map_names_files_docs_and_public_items() {
    let dir = fixture();
    let files = scan_crate(&dir.path().join("crates/clean")).unwrap();
    assert_eq!(files.len(), 2);
    assert_eq!(files[0].path, "inner/mod.rs");
    assert_eq!(files[1].path, "lib.rs");
    assert_eq!(files[1].doc.as_deref(), Some("A clean crate."));
    assert_eq!(files[1].items, [("fn add".to_owned(), "Adds.".to_owned())]);
    run_map(dir.path(), &["clean"]).unwrap();
    assert!(run_map(dir.path(), &["nope"]).is_err());
    assert!(run_map(dir.path(), &[]).is_err());
}
