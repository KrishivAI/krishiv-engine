// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask entry-dirs [--check | --write]`: registries kept as one file per entry, so two
//! changes that each add an entry never edit the same file (the conflict churn of shared sorted
//! lists). Collected where they are used (`krishiv-common`'s `build.rs` reads the error classes,
//! `krishiv-plan-codec`'s reads its per-entry test files) and checked here; CI's `pr-policy`
//! job and `cargo xtask precheck` run `--check`.
//!
//! - `crates/krishiv-common/krishiv-error-classes/<CLASS>.json`: one Krishiv error class per file,
//!   the class body (`message`, `sqlState`, optional `audience`, `subClass`), named
//!   `KRISHIV_[A-Z0-9_]+`, pretty-printed as `serde_json` prints it (`--write` rewrites).
//! - `crates/krishiv-plan-codec/tests/{features,roundtrip}/<name>.rs`: codec tests, one module per
//!   file of the `features` / `roundtrip` test (snake_case name, not `main`, SPDX header);
//!   `tests/features.rs` and `tests/roundtrip.rs` may not gain test functions (their counts are
//!   capped below), so new tests land in new files.

use std::path::Path;

/// The error-class directory, relative to the workspace root.
pub const ERROR_CLASSES: &str = "crates/krishiv-common/krishiv-error-classes";
/// The per-entry codec test directories (relative to the workspace root) and the file each
/// belongs to.
pub const CODEC_TEST_DIRS: &[(&str, &str)] = &[
    (
        "crates/krishiv-plan-codec/tests/features",
        "tests/features.rs",
    ),
    (
        "crates/krishiv-plan-codec/tests/roundtrip",
        "tests/roundtrip.rs",
    ),
];
/// Files whose `#[test]` count may only fall: (path, cap). Lower a cap when tests move out.
pub const CAPPED_TEST_FILES: &[(&str, usize)] = &[
    ("crates/krishiv-plan-codec/tests/features.rs", 58),
    ("crates/krishiv-plan-codec/tests/roundtrip.rs", 55),
];

/// The Krishiv error classes under `root` as one JSON object (`{"KRISHIV_X": {...}, ...}`), the
/// shape of the vendored Spark registry.
///
/// # Errors
/// An unreadable directory or file, or a file that is not a JSON object.
pub fn error_classes_json(root: &Path) -> Result<String, String> {
    let mut all = serde_json::Map::new();
    for (name, path) in error_class_files(root)? {
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        all.insert(name, parse_body(&path, &text)?);
    }
    serde_json::to_string(&all).map_err(|e| e.to_string())
}

fn error_class_files(root: &Path) -> Result<Vec<(String, std::path::PathBuf)>, String> {
    let dir = root.join(ERROR_CLASSES);
    let entries =
        std::fs::read_dir(&dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
    let mut out = Vec::new();
    for entry in entries {
        let path = entry.map_err(|e| e.to_string())?.path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".json"))
            .ok_or_else(|| format!("{}: not a <CLASS>.json file", path.display()))?;
        out.push((name.to_owned(), path));
    }
    out.sort();
    Ok(out)
}

fn parse_body(path: &Path, text: &str) -> Result<serde_json::Value, String> {
    let body: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("{}: {e}", path.display()))?;
    if body.is_object() {
        Ok(body)
    } else {
        Err(format!(
            "{}: a class body must be a JSON object",
            path.display()
        ))
    }
}

/// Whether `name` is a Krishiv error-class name: `KRISHIV_` then `[A-Z0-9_]`, not ending in `_`.
#[must_use]
pub fn is_class_name(name: &str) -> bool {
    name.strip_prefix("KRISHIV_").is_some_and(|rest| {
        !rest.is_empty()
            && !rest.ends_with('_')
            && rest
                .bytes()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
    })
}

/// Whether `name` (a file stem) is a snake_case Rust module name.
#[must_use]
pub fn is_module_name(name: &str) -> bool {
    // `main.rs` in a directory under `tests/` would also be a test target of its own.
    name != "main"
        && name.starts_with(|c: char| c.is_ascii_lowercase())
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

/// The canonical text of a class body: `serde_json`'s pretty form and a final newline.
///
/// # Errors
/// Never in practice (a parsed value always serialises).
pub fn canonical(body: &serde_json::Value) -> Result<String, String> {
    serde_json::to_string_pretty(body)
        .map(|s| s + "\n")
        .map_err(|e| e.to_string())
}

/// The test functions in `text`: lines starting with a test attribute (`#[test]`,
/// `#[tokio::test]`, `#[tokio::test(...)]`), the function on the same line or the next.
#[must_use]
pub fn test_count(text: &str) -> usize {
    text.lines()
        .map(str::trim_start)
        .filter(|l| l.starts_with("#[test]") || l.starts_with("#[tokio::test"))
        .count()
}

/// Runs the checks; with `write`, rewrites error-class files into their canonical form first.
///
/// # Errors
/// Every problem found, one per line.
pub fn run(root: &Path, write: bool) -> Result<(), String> {
    let mut problems = Vec::new();
    let classes = match error_class_files(root) {
        Ok(c) => c,
        Err(e) => {
            problems.push(e);
            Vec::new()
        }
    };
    for (name, path) in classes {
        if !is_class_name(&name) {
            problems.push(format!(
                "{}: an error-class file is named KRISHIV_<NAME>.json in [A-Z0-9_]",
                path.display()
            ));
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let want = match parse_body(&path, &text).and_then(|b| canonical(&b)) {
            Ok(w) => w,
            Err(e) => {
                problems.push(e);
                continue;
            }
        };
        if text != want {
            if write {
                std::fs::write(&path, want)
                    .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
            } else {
                problems.push(format!(
                    "{}: not in canonical form; run `cargo xtask entry-dirs --write`",
                    path.display()
                ));
            }
        }
    }
    for (dir, of) in CODEC_TEST_DIRS {
        let Ok(entries) = std::fs::read_dir(root.join(dir)) else {
            continue;
        };
        for entry in entries {
            let path = match entry {
                Ok(e) => e.path(),
                Err(e) => {
                    problems.push(format!("{dir}: {e}"));
                    continue;
                }
            };
            let stem = path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix(".rs"));
            match stem {
                Some(stem) if is_module_name(stem) => match std::fs::read_to_string(&path) {
                    Ok(text) if text.starts_with("// SPDX-License-Identifier: Apache-2.0") => {}
                    Ok(_) => problems.push(format!("{}: missing the SPDX header", path.display())),
                    Err(e) => problems.push(format!("{}: {e}", path.display())),
                },
                _ => problems.push(format!(
                    "{}: a test file of {of} is a snake_case <name>.rs other than main.rs",
                    path.display()
                )),
            }
        }
    }
    for (file, cap) in CAPPED_TEST_FILES {
        let path = root.join(file);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let n = test_count(&text);
        if n > *cap {
            problems.push(format!(
                "{file}: {n} test functions, more than its cap {cap}; put new tests in a new file \
                 of {} (one module per file, collected by build.rs)",
                file.trim_end_matches(".rs")
            ));
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_and_module_names() {
        assert!(is_class_name("KRISHIV_CATALOG_FAILURE"));
        assert!(is_class_name("KRISHIV_A1"));
        assert!(!is_class_name("KRISHIV_"));
        assert!(!is_class_name("KRISHIV_A_"));
        assert!(!is_class_name("KRISHIV_lower"));
        assert!(!is_class_name("DELTA_X"));
        assert!(is_module_name("nested_literals"));
        assert!(!is_module_name("Nested"));
        assert!(!is_module_name("1x"));
        assert!(!is_module_name("a-b"));
        assert!(!is_module_name("main"));
    }

    #[test]
    fn canonical_form_and_test_count() {
        let body = serde_json::json!({"sqlState": "58030", "message": ["m"]});
        assert_eq!(
            canonical(&body).unwrap(),
            "{\n  \"message\": [\n    \"m\"\n  ],\n  \"sqlState\": \"58030\"\n}\n"
        );
        assert_eq!(
            test_count("#[test]\nfn a() {}\n    #[test] fn b() {}\n#[tokio::test]\n// #[test]\n"),
            3
        );
    }

    #[test]
    fn the_repository_passes() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        run(root, false).unwrap();
        let all: serde_json::Value =
            serde_json::from_str(&error_classes_json(root).unwrap()).unwrap();
        assert!(all.get("KRISHIV_CATALOG_FAILURE").is_some());
    }
}
