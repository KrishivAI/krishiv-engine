// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask registry-coverage` (P2-T109): every error class the engine can
//! raise and every config key in the registries must be referenced by at least
//! one test.
//!
//! # What is checked
//!
//! The error registry holds the **full Spark catalog** (about 1,900 classes,
//! generated from Spark's `error-conditions.json`, P2-T003) plus the
//! `KRISHIV_` classes. The catalog is data: its integrity is verified by
//! `krishiv-common`'s class-count test and its sampled byte-identical render
//! test, not by this check. This check covers the **engine-raised set**: the
//! catalog classes that engine code refers to, by Spark name (`"DIVIDE_BY_ZERO"`)
//! or Rust identifier (`ErrorClass::DivideByZero`, `DivideByZero { .. }`).
//! Engine code is every `.rs` file under `crates/*/src/` outside test code
//! (see [`engine_text`]). A class nothing raises needs no test; the moment
//! engine code starts raising one, a test must reference it too. Config keys
//! are all checked.
//!
//! A registry entry is *referenced* when one of its identifiers appears as a
//! whole token in test text:
//! - error class: its Spark name (`DIVIDE_BY_ZERO`, `UNRESOLVED_COLUMN.WITH_SUGGESTION`)
//!   or its Rust name (`DivideByZero`);
//! - config key: the key string (`krishiv.sql.ansi.enabled`).
//!
//! Test text is every `.rs` file under a `tests/` directory or named
//! `tests.rs`, plus the part of any other `.rs` file from its first
//! `#[cfg(test)]` onward. A token match requires that the characters on both
//! sides are not identifier characters, and that a `.` does not join the
//! match to a longer dotted name (so a parent class is not credited by its
//! sub-class). Entries that cannot be tested yet go in [`ALLOWLIST`].

use std::path::{Path, PathBuf};

/// Entries exempt from the check, each with the reason it is untestable yet.
/// Keep this empty unless there is no way to write the test today.
pub const ALLOWLIST: &[(&str, &str)] = &[];

/// One error class, as the generator names it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassEntry {
    /// Spark name, e.g. `UNRESOLVED_COLUMN.WITH_SUGGESTION`.
    pub name: String,
    /// Rust identifier, e.g. `UnresolvedColumnWithSuggestion`.
    pub ident: String,
}

fn ident_of(name: &str) -> String {
    name.split(['_', '.'])
        .map(|w| {
            let mut c = w.chars();
            c.next().map_or_else(String::new, |f| {
                f.to_string() + &c.as_str().to_ascii_lowercase()
            })
        })
        .collect()
}

/// Lists the leaf classes of an error-class JSON document (a class with
/// sub-classes contributes one leaf per sub-class, as in the generator).
///
/// # Errors
/// Returns a message when the JSON is malformed.
pub fn parse_error_classes(json: &str) -> Result<Vec<ClassEntry>, String> {
    let root: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(json).map_err(|e| format!("invalid error-class JSON: {e}"))?;
    let mut out = Vec::new();
    for (name, info) in &root {
        match info.get("subClass").and_then(serde_json::Value::as_object) {
            Some(subs) => {
                for sub in subs.keys() {
                    let full = format!("{name}.{sub}");
                    out.push(ClassEntry {
                        ident: ident_of(&full),
                        name: full,
                    });
                }
            }
            None => out.push(ClassEntry {
                ident: ident_of(name),
                name: name.clone(),
            }),
        }
    }
    Ok(out)
}

fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// True when `needle` occurs in `text` as a whole token.
#[must_use]
pub fn references(text: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let bytes = text.as_bytes();
    // Dotted joining applies to Spark names and config keys, not to Rust
    // identifiers (`Foo.method()` still references `Foo`).
    let dotted = needle.contains('.') || !needle.bytes().any(|b| b.is_ascii_lowercase());
    text.match_indices(needle).any(|(i, _)| {
        let before = i.checked_sub(1).map(|j| bytes[j]);
        let after = bytes.get(i + needle.len()).copied();
        let joins = |dot: Option<u8>, other: Option<u8>| {
            dot == Some(b'.') && other.is_some_and(is_ident_char)
        };
        let before_ok = before.is_none_or(|b| !is_ident_char(b));
        let after_ok = after.is_none_or(|b| !is_ident_char(b));
        let dotted_after = joins(after, bytes.get(i + needle.len() + 1).copied());
        let dotted_before = joins(before, i.checked_sub(2).map(|j| bytes[j]));
        before_ok && after_ok && !(dotted && (dotted_after || dotted_before))
    })
}

/// The test portion of a source file, or `None` if it has none.
#[must_use]
pub fn test_text<'a>(rel_path: &str, text: &'a str) -> Option<&'a str> {
    let p = rel_path.replace('\\', "/");
    let in_tests_dir = p.split('/').any(|c| c == "tests");
    let file = p.rsplit('/').next().unwrap_or(&p);
    if in_tests_dir || file == "tests.rs" {
        return Some(text);
    }
    text.find("#[cfg(test)]").map(|i| &text[i..])
}

/// The engine (non-test, non-comment) part of a source file, or `None` for
/// files that are not engine code: anything outside `crates/*/src/`, test
/// files, and `build.rs`. Comment lines (including doc comments and their
/// examples) are dropped, so documentation never makes a class "raised".
#[must_use]
pub fn engine_text(rel_path: &str, text: &str) -> Option<String> {
    let p = rel_path.replace('\\', "/");
    let parts: Vec<&str> = p.split('/').collect();
    let in_src = parts.first() == Some(&"crates") && parts.get(2) == Some(&"src");
    let file = parts.last().copied().unwrap_or_default();
    if !in_src || parts.contains(&"tests") || file == "tests.rs" {
        return None;
    }
    let code = text.find("#[cfg(test)]").map_or(text, |i| &text[..i]);
    Some(
        code.lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

/// The classes that engine code refers to (the engine-raised set).
#[must_use]
pub fn raised(catalog: &[ClassEntry], engine: &[String]) -> Vec<ClassEntry> {
    catalog
        .iter()
        .filter(|c| {
            engine
                .iter()
                .any(|t| references(t, &c.name) || references(t, &c.ident))
        })
        .cloned()
        .collect()
}

/// Entries with no referencing test, as messages. Empty means success.
#[must_use]
pub fn uncovered(classes: &[ClassEntry], keys: &[&str], tests: &[String]) -> Vec<String> {
    let allowed = |id: &str| ALLOWLIST.iter().any(|(a, _)| *a == id);
    let mut out = Vec::new();
    for c in classes {
        let hit = tests
            .iter()
            .any(|t| references(t, &c.name) || references(t, &c.ident));
        if !hit && !allowed(&c.name) {
            out.push(format!(
                "error class `{}` ({}) has no referencing test",
                c.name, c.ident
            ));
        }
    }
    for k in keys {
        if !tests.iter().any(|t| references(t, k)) && !allowed(k) {
            out.push(format!("config key `{k}` has no referencing test"));
        }
    }
    out
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let path = entry.path();
        let name = entry.file_name();
        if path.is_dir() {
            let skip = name == "target" || name.to_string_lossy().starts_with('.');
            if !skip {
                walk(&path, out);
            }
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Collects the test text of every `.rs` file under `root`.
#[must_use]
pub fn collect_tests(root: &Path) -> Vec<String> {
    let mut files = Vec::new();
    walk(root, &mut files);
    files
        .iter()
        .filter_map(|f| {
            let rel = f.strip_prefix(root).unwrap_or(f).to_string_lossy();
            let text = std::fs::read_to_string(f).ok()?;
            test_text(&rel, &text).map(str::to_owned)
        })
        .collect()
}

/// Collects the engine text of every `.rs` file under `root`.
#[must_use]
pub fn collect_engine(root: &Path) -> Vec<String> {
    let mut files = Vec::new();
    walk(root, &mut files);
    files
        .iter()
        .filter_map(|f| {
            let rel = f.strip_prefix(root).unwrap_or(f).to_string_lossy();
            let text = std::fs::read_to_string(f).ok()?;
            engine_text(&rel, &text)
        })
        .collect()
}

/// `xtask registry-coverage`: checks the engine-raised error classes and all
/// config keys against the tests under `root`.
///
/// # Errors
/// Returns every uncovered entry.
pub fn run(root: &Path) -> Result<(), String> {
    let mut catalog = Vec::new();
    for file in [
        "vendored/spark-error-conditions.json",
        "krishiv-error-classes.json",
    ] {
        let path = root.join("crates/krishiv-common").join(file);
        let json = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        catalog.extend(parse_error_classes(&json)?);
    }
    let classes = raised(&catalog, &collect_engine(root));
    let keys: Vec<&str> = krishiv_common::config::REGISTRY
        .iter()
        .map(|e| e.key)
        .collect();
    let tests = collect_tests(root);
    let missing = uncovered(&classes, &keys, &tests);
    if missing.is_empty() {
        Ok(())
    } else {
        Err(missing
            .iter()
            .map(|m| format!("error: {m}"))
            .collect::<Vec<_>>()
            .join("\n"))
    }
}
