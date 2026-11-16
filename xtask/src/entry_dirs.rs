// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask entry-dirs [--check | --write]`: registries kept as one file per entry, so two
//! changes that each add an entry never edit the same file (the conflict churn of shared sorted
//! lists). Collected where they are used (`krishiv-common`'s `build.rs` reads the error classes,
//! `krishiv-plan-codec`'s reads its per-feature test files) and checked here; CI's `generated`
//! job runs `--check`.
//!
//! - `crates/krishiv-common/krishiv-error-classes/<CLASS>.json`: one Krishiv error class per file,
//!   the class body (`message`, `sqlState`, optional `audience`, `subClass`), named
//!   `KRISHIV_[A-Z0-9_]+`, pretty-printed as `serde_json` prints it (`--write` rewrites).
//! - `crates/krishiv-plan-codec/tests/features/<name>.rs`: a feature's own codec tests, one module
//!   per file (snake_case name, SPDX header); `tests/features.rs` and `tests/roundtrip.rs` may not
//!   gain `#[test]` functions (their counts are capped below), so new tests land in new files.

use std::path::Path;

/// The error-class directory, relative to the workspace root.
pub const ERROR_CLASSES: &str = "crates/krishiv-common/krishiv-error-classes";
/// The per-feature codec test directory, relative to the workspace root.
pub const CODEC_FEATURE_TESTS: &str = "crates/krishiv-plan-codec/tests/features";
/// Files whose `#[test]` count may only fall: (path, cap). Lower a cap when tests move out.
pub const CAPPED_TEST_FILES: &[(&str, usize)] = &[
    ("crates/krishiv-plan-codec/tests/features.rs", 58),
    ("crates/krishiv-plan-codec/tests/roundtrip.rs", 56),
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
    name.starts_with(|c: char| c.is_ascii_lowercase())
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

/// The `#[test]` functions in `text`.
#[must_use]
pub fn test_count(text: &str) -> usize {
    text.lines().filter(|l| l.trim() == "#[test]").count()
}

/// Runs the checks; with `write`, rewrites error-class files into their canonical form first.
///
/// # Errors
/// Every problem found, one per line.
pub fn run(root: &Path, write: bool) -> Result<(), String> {
    let mut problems = Vec::new();
    for (name, path) in error_class_files(root)? {
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
    let tests_dir = root.join(CODEC_FEATURE_TESTS);
    if let Ok(entries) = std::fs::read_dir(&tests_dir) {
        for entry in entries {
            let path = entry.map_err(|e| e.to_string())?.path();
            let stem = path
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix(".rs"));
            match stem {
                Some(stem) if is_module_name(stem) => {
                    let text = std::fs::read_to_string(&path)
                        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
                    if !text.starts_with("// SPDX-License-Identifier: Apache-2.0") {
                        problems.push(format!("{}: missing the SPDX header", path.display()));
                    }
                }
                _ => problems.push(format!(
                    "{}: a codec test file is a snake_case <name>.rs",
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
                "{file}: {n} #[test] functions, more than its cap {cap}; put new tests in \
                 {CODEC_FEATURE_TESTS}/<feature>.rs (one file per feature, collected by build.rs)"
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
    }

    #[test]
    fn canonical_form_and_test_count() {
        let body = serde_json::json!({"sqlState": "58030", "message": ["m"]});
        assert_eq!(
            canonical(&body).unwrap(),
            "{\n  \"message\": [\n    \"m\"\n  ],\n  \"sqlState\": \"58030\"\n}\n"
        );
        assert_eq!(
            test_count("#[test]\nfn a() {}\n    #[test]\nfn b() {}\n#[tokio::test]\n"),
            2
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
