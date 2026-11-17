// SPDX-License-Identifier: Apache-2.0

use krishiv_common::error::ErrorClass;

use super::{ArtifactKind, parse_artifact_name};

fn class(name: &str) -> ErrorClass {
    match parse_artifact_name(name) {
        Ok(n) => panic!("{name:?} was accepted as {n:?}"),
        Err(e) => e.error_class(),
    }
}

#[test]
fn artifact_names_accepted() {
    let n = parse_artifact_name("pyfiles/util.py").unwrap();
    assert_eq!(n.kind(), ArtifactKind::PyFiles);
    assert_eq!(n.path(), "pyfiles/util.py");
    assert_eq!(n.segments().collect::<Vec<_>>(), ["pyfiles", "util.py"]);
    assert_eq!(n.fragment(), None);

    let n = parse_artifact_name("files/sub/dir/data.txt").unwrap();
    assert_eq!(n.kind(), ArtifactKind::Files);
    assert_eq!(n.path(), "files/sub/dir/data.txt");

    // An archive may carry one `#fragment` (the directory it unpacks to).
    let n = parse_artifact_name("archives/env.tar.gz#env").unwrap();
    assert_eq!(n.kind(), ArtifactKind::Archives);
    assert_eq!(n.path(), "archives/env.tar.gz");
    assert_eq!(n.fragment(), Some("env"));
    // Java's `split("#")` drops a trailing empty part: no fragment.
    let n = parse_artifact_name("archives/env.zip#").unwrap();
    assert_eq!((n.path(), n.fragment()), ("archives/env.zip", None));
    // Elsewhere `#` is part of the file name.
    let n = parse_artifact_name("files/a#b").unwrap();
    assert_eq!((n.path(), n.fragment()), ("files/a#b", None));

    let hash = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";
    let n = parse_artifact_name(&format!("cache/{hash}")).unwrap();
    assert_eq!(n.kind(), ArtifactKind::Cache);
    assert_eq!(n.cache_hash(), Some(hash));
    assert_eq!(n.path(), format!("cache/{hash}"));
    // Non-ASCII file names are kept as given.
    assert!(parse_artifact_name("files/données.csv").is_ok());
    assert!(parse_artifact_name("files/.hidden").is_ok());
    assert!(parse_artifact_name("files/..x").is_ok());
}

#[test]
fn artifact_names_refused() {
    let invalid = ErrorClass::InvalidArtifactPath;
    for name in [
        "",
        "pyfiles",
        "pyfiles/",
        "/pyfiles/a.py",
        "pyfiles//a.py",
        "pyfiles/./a.py",
        "pyfiles/../a.py",
        "pyfiles/sub/../../a.py",
        "files/..",
        "files/.",
        "pyfiles/a.py/",
        "pyfiles\\..\\a.py",
        "files/a\\b",
        "files/a\0b",
        "files/a\nb",
        "files/a\u{7f}b",
        // Bidirectional overrides and zero-width characters hide a name's
        // real text in logs and listings.
        "files/a\u{202e}yp.exe",
        "files/a\u{200b}b",
        "files/\u{feff}a",
        "archives/a.zip#b#c",
        "archives/a.zip#../b",
        "archives/a.zip#b/c",
        "archives/#b",
        "unknown/a",
        "pyfiles.py",
        "Cache/abc",
        "cache/",
        "cache/a/b",
        "cache/../x",
        "cache/not-hex!",
    ] {
        assert_eq!(class(name), invalid, "{name:?}");
    }
    // A segment longer than a file name may be, and a name past the cap.
    assert_eq!(class(&format!("files/{}", "a".repeat(256))), invalid);
    let deep = format!("files/{}a", "d/".repeat(40));
    assert_eq!(class(&deep), invalid);
    let long = format!("files/{}", vec!["b".repeat(200); 30].join("/"));
    assert_eq!(class(&long), invalid);
    assert_eq!(class(&format!("cache/{}", "a".repeat(129))), invalid);
}

#[test]
fn jvm_and_forwarded_artifacts_are_unsupported() {
    for name in ["jars/udf.jar", "classes/A.class", "forward_to_fs/tmp/x"] {
        assert_eq!(
            class(name),
            ErrorClass::KrishivUnsupportedFeature,
            "{name:?}"
        );
    }
    // Still checked as a path first.
    assert_eq!(class("jars/../x"), ErrorClass::InvalidArtifactPath);
}

#[test]
fn refusals_name_the_artifact() {
    let e = parse_artifact_name("pyfiles/../a.py").unwrap_err();
    assert_eq!(
        e.message_parameters(),
        &[("name", "pyfiles/../a.py".to_owned())]
    );
    assert_eq!(e.sql_state(), Some("22023"));
}
