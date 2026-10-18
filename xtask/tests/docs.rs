// SPDX-License-Identifier: Apache-2.0
//! Tests for the docs emitter.

use xtask::docs::{Doc, emit};

#[test]
fn renders_both_documents() {
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
    assert_eq!(Doc::from_command("docs-config"), Some(Doc::Config));
    assert_eq!(Doc::from_command("docs-errors"), Some(Doc::Errors));
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
