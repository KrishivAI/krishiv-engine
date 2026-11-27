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
    let on_disk = std::fs::read_to_string(root.join("docs/reference/config/index.md")).unwrap();
    assert_eq!(
        on_disk,
        Doc::Config.markdown(),
        "run `cargo xtask docs-config docs/reference/config/index.md`"
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

#[test]
fn generated_reference_pages_name_no_internal_documents() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let mut files = vec![root.join("docs/reference/config/index.md")];
    for dir in ["sql/functions", "python", "rust", "errors"] {
        let d = root.join("docs/reference").join(dir);
        for e in std::fs::read_dir(&d).unwrap() {
            files.push(e.unwrap().path());
        }
    }
    for f in files {
        let Ok(text) = std::fs::read_to_string(&f) else {
            continue;
        };
        for (i, line) in text.lines().enumerate() {
            let b = line.as_bytes();
            let task = b.windows(2).enumerate().any(|(k, w)| {
                w[0] == b'P'
                    && w[1].is_ascii_digit()
                    && line[k + 1..]
                        .trim_start_matches(|c: char| c.is_ascii_digit())
                        .starts_with("-T")
            });
            let adr = line.contains("ADR 0");
            let layer = line.contains("layer L") && line.contains("(layer L");
            assert!(
                !(task || adr || layer || line.contains("design/")),
                "{}:{}: internal reference on a public page: {line}",
                f.display(),
                i + 1
            );
        }
    }
}

/// The fenced blocks of `text` (language line and body), optionally only those between the
/// README's `quickstart:begin` and `quickstart:end` comments.
fn fenced_blocks(text: &str, between_markers: bool) -> Vec<String> {
    let mut on = !between_markers;
    let (mut open, mut cur, mut out) = (false, String::new(), Vec::new());
    for line in text.lines() {
        if line.contains("quickstart:begin") {
            on = true;
        } else if line.contains("quickstart:end") {
            on = !between_markers;
        } else if on && line.starts_with("```") {
            if open {
                out.push(std::mem::take(&mut cur));
            } else {
                cur = format!("{line}\n");
            }
            open = !open;
        } else if on && open {
            cur.push_str(line);
            cur.push('\n');
        }
    }
    out
}

/// The `python` blocks of a page; for the README, only the Quickstart's.
#[expect(
    clippy::unwrap_used,
    reason = "a test helper: a missing page fails the test"
)]
fn python_blocks(path: &str) -> Vec<String> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let text = std::fs::read_to_string(root.join(path)).unwrap();
    fenced_blocks(&text, path == "README.md")
        .into_iter()
        .filter(|b| b.starts_with("```python"))
        .collect()
}

#[test]
fn the_quickstart_page_runs_the_readme_code() {
    let readme = python_blocks("README.md");
    for page in [
        "docs/get-started/index.md",
        "docs/get-started/quickstart.md",
    ] {
        assert_eq!(
            readme,
            python_blocks(page),
            "{page} drifted from the README"
        );
    }
}

/// The first result comes before the longer example: import, a session, one query.
#[test]
fn the_quickstart_starts_with_three_lines() {
    let first = &python_blocks("README.md")[0];
    let code: Vec<&str> = first.lines().skip(1).filter(|l| !l.is_empty()).collect();
    assert_eq!(
        code,
        [
            "import krishiv",
            "spark = krishiv.session()  # a PySpark-style SparkSession, in process",
            "spark.sql(\"SELECT 1 AS answer\").show()",
        ]
    );
}
