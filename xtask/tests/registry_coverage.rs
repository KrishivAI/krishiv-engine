// SPDX-License-Identifier: Apache-2.0
//! Tests for the registry coverage check.

use xtask::registry_coverage::{
    ClassEntry, collect_tests, parse_error_classes, references, run, test_text, uncovered,
};

const JSON: &str = r#"{
  "ALPHA": {"message": ["a"], "sqlState": "22003"},
  "PARENT": {"message": ["p"], "sqlState": "42000",
             "subClass": {"CHILD": {"message": ["c"]}}}
}"#;

#[test]
fn parses_leaves_like_the_generator() {
    let got = parse_error_classes(JSON).unwrap();
    let names: Vec<_> = got
        .iter()
        .map(|c| (c.name.as_str(), c.ident.as_str()))
        .collect();
    assert_eq!(names, [("ALPHA", "Alpha"), ("PARENT.CHILD", "ParentChild")]);
    assert!(parse_error_classes("nope").is_err());
}

#[test]
fn token_matching_is_exact() {
    assert!(references("x(\"ALPHA\")", "ALPHA"));
    assert!(!references("ALPHA_BETA", "ALPHA"));
    assert!(!references("XALPHA", "ALPHA"));
    assert!(!references("PARENT.CHILD", "PARENT"));
    assert!(references("PARENT.CHILD", "PARENT.CHILD"));
    assert!(references("PARENT. done", "PARENT"));
    assert!(!references("krishiv.a.b.c", "krishiv.a.b"));
    assert!(!references("anything", ""));
}

#[test]
fn selects_test_text() {
    assert!(test_text("crates/x/tests/a.rs", "body").is_some());
    assert!(test_text("crates/x/src/tests.rs", "body").is_some());
    assert!(test_text("crates/x/src/lib.rs", "body").is_none());
    assert_eq!(
        test_text("crates/x/src/lib.rs", "code\n#[cfg(test)]\nmod t {}"),
        Some("#[cfg(test)]\nmod t {}")
    );
}

#[test]
fn fixture_with_unreferenced_entries_fails() {
    let classes = parse_error_classes(JSON).unwrap();
    let tests = vec!["assert_eq!(c.name(), \"ALPHA\"); use krishiv.cfg.seen;".to_owned()];
    let missing = uncovered(
        &classes,
        &["krishiv.cfg.seen", "krishiv.cfg.unseen"],
        &tests,
    );
    assert_eq!(missing.len(), 2, "{missing:?}");
    assert!(missing[0].contains("PARENT.CHILD"));
    assert!(missing[1].contains("krishiv.cfg.unseen"));
}

#[test]
fn rust_identifier_counts_as_reference() {
    let classes = vec![ClassEntry {
        name: "ALPHA".into(),
        ident: "Alpha".into(),
    }];
    assert!(uncovered(&classes, &[], &["ErrorClass::Alpha".to_owned()]).is_empty());
}

#[test]
fn fixture_tree_is_scanned() {
    let dir = std::env::temp_dir().join(format!("xtask-regcov-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("c/tests")).unwrap();
    std::fs::create_dir_all(dir.join("target")).unwrap();
    std::fs::write(dir.join("c/tests/t.rs"), "ALPHA").unwrap();
    std::fs::write(dir.join("c/lib.rs"), "BETA").unwrap();
    std::fs::write(dir.join("target/tests.rs"), "GAMMA").unwrap();
    let texts = collect_tests(&dir);
    assert_eq!(texts, ["ALPHA"]);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn current_registries_are_covered() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    if let Err(e) = run(root) {
        panic!("{e}");
    }
}

#[test]
fn method_call_on_rust_identifier_counts() {
    assert!(references("ErrorClass::Alpha.sql_state()", "Alpha"));
}
