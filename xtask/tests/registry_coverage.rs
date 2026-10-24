// SPDX-License-Identifier: Apache-2.0
//! Tests for the registry coverage check.

use xtask::registry_coverage::{
    ClassEntry, collect_engine, collect_tests, engine_text, parse_error_classes, raised,
    references, run, test_text, uncovered,
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
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    std::fs::create_dir_all(dir.join("c/tests")).unwrap();
    std::fs::create_dir_all(dir.join("target")).unwrap();
    std::fs::write(dir.join("c/tests/t.rs"), "ALPHA").unwrap();
    std::fs::write(dir.join("c/lib.rs"), "BETA").unwrap();
    std::fs::write(dir.join("target/tests.rs"), "GAMMA").unwrap();
    let texts = collect_tests(dir);
    assert_eq!(texts, ["ALPHA"]);
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

#[test]
fn engine_text_skips_tests_comments_and_other_trees() {
    let src = "use a::Alpha;\n// Beta in a comment\n/// Gamma in docs\nfn f() {}\n#[cfg(test)]\nmod t { Delta }";
    let got = engine_text("crates/x/src/lib.rs", src).unwrap();
    assert!(got.contains("Alpha") && got.contains("fn f"));
    assert!(!got.contains("Beta") && !got.contains("Gamma") && !got.contains("Delta"));
    assert!(engine_text("crates/x/src/tests.rs", "Alpha").is_none());
    assert!(engine_text("crates/x/tests/t.rs", "Alpha").is_none());
    assert!(engine_text("crates/x/build.rs", "Alpha").is_none());
    assert!(engine_text("xtask/src/lib.rs", "Alpha").is_none());
}

/// A catalog class the engine does not raise needs no test; one it raises does.
#[test]
fn only_engine_raised_classes_need_tests() {
    let catalog = parse_error_classes(JSON).unwrap();
    let engine = vec!["return Err(ErrorClass::Alpha.into());".to_owned()];
    let in_use = raised(&catalog, &engine);
    assert_eq!(in_use.len(), 1);
    assert_eq!(in_use[0].name, "ALPHA");
    // Catalog entries nothing raises are not demanded...
    assert!(uncovered(&in_use, &[], &[]).len() == 1);
    assert!(uncovered(&in_use, &[], &["ALPHA".to_owned()]).is_empty());
    // ...so the rest of the catalog passes without any test text.
    let none = raised(&catalog, &[]);
    assert!(uncovered(&none, &[], &[]).is_empty());
    // Raising it by Spark name counts too.
    let by_name = raised(&catalog, &["throw(\"PARENT.CHILD\")".to_owned()]);
    assert_eq!(by_name.len(), 1);
    assert_eq!(by_name[0].ident, "ParentChild");
}

#[test]
fn engine_raised_set_of_this_repo_is_a_small_subset_of_the_catalog() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let json = std::fs::read_to_string(
        root.join("crates/krishiv-common/vendored/spark-error-conditions.json"),
    )
    .unwrap();
    let catalog = parse_error_classes(&json).unwrap();
    assert!(catalog.len() > 1_900, "full Spark 4.2.0 catalog is loaded");
    let in_use = raised(&catalog, &collect_engine(root));
    // A detector that matched the whole catalog (the bug this guards
    // against) would far exceed this bound; the engine's raised set grows
    // with each function family (193 of the 1,929 classes after the bitwise
    // and number-format functions), so the bound is a quarter, not a tenth.
    assert!(in_use.len() < catalog.len() / 4, "{}", in_use.len());
    // The comment and test filter removes real references in this repo:
    // the same sources read whole name strictly more classes.
    let mut raw = Vec::new();
    let mut dirs = vec![root.join("crates")];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().is_some_and(|e| e == "rs")
                && path.components().any(|c| c.as_os_str() == "src")
            {
                raw.push(std::fs::read_to_string(&path).unwrap());
            }
        }
    }
    let in_raw = |c: &ClassEntry| raised(std::slice::from_ref(c), &raw).len() == 1;
    assert!(
        in_use.iter().all(in_raw),
        "engine text is a subset of the raw sources"
    );
    let filtered_out = catalog
        .iter()
        .find(|c| !in_use.contains(c) && in_raw(c))
        .map(|c| c.name.clone());
    assert!(filtered_out.is_some(), "the filter drops no reference");
}
