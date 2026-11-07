// SPDX-License-Identifier: Apache-2.0
//! Tests for the docs emitter.

use xtask::docs::{Doc, emit};

#[test]
fn renders_all_documents() {
    assert!(
        Doc::Config
            .markdown()
            .contains("`krishiv.sql.ansi.enabled`")
    );
    assert!(
        Doc::Errors
            .markdown()
            .contains("`DIVIDE_BY_ZERO` | `22012`")
    );
    assert!(Doc::Metrics.markdown().contains("`krishiv.query.count`"));
    assert_eq!(Doc::from_command("docs-config"), Some(Doc::Config));
    assert_eq!(Doc::from_command("docs-errors"), Some(Doc::Errors));
    assert_eq!(Doc::from_command("docs-metrics"), Some(Doc::Metrics));
    assert_eq!(Doc::from_command("layering"), None);
}

#[test]
fn emits_to_stdout_or_path() {
    let mut buf = Vec::new();
    emit("# hi\n", None, &mut buf).unwrap();
    assert_eq!(buf, b"# hi\n");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nested/config.md");
    let mut buf = Vec::new();
    emit("# hi\n", Some(&path), &mut buf).unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "# hi\n");
    assert!(buf.is_empty());
}

#[test]
fn the_checked_in_configuration_guide_page_matches_the_registry() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let on_disk = std::fs::read_to_string(root.join("docs/configuration.md")).unwrap();
    assert_eq!(
        on_disk,
        Doc::Config.markdown(),
        "run `cargo xtask docs-config docs/configuration.md`"
    );
}

#[test]
fn the_checked_in_python_reference_matches_the_generator() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    if std::process::Command::new("python3")
        .arg("--version")
        .output()
        .is_err()
    {
        return; // no Python on this machine; the python-quality job runs it
    }
    xtask::docs_ext::run_python(root, true).unwrap();
}
