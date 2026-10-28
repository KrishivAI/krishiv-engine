// SPDX-License-Identifier: Apache-2.0
#![expect(clippy::print_stdout, reason = "command output is the interface")]
//! `cargo xtask check-format-neutral`: production code outside
//! `crates/krishiv-io/src/parquet/` must not name a concrete format
//! (`parquet::` or `krishiv_io::parquet`), so connectors can add formats
//! without touching production code. The only exemption is the registry
//! constructor (`FormatRegistry::with_defaults`); test code and
//! `crates/krishiv-io/benches/io.rs` are excluded.

use std::path::{Path, PathBuf};

/// A violation: the file and line naming a concrete format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Violation {
    /// Repository-relative path.
    pub path: String,
    /// 1-based line number.
    pub line: usize,
    /// The matched marker.
    pub marker: &'static str,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}: `{}` names a concrete format",
            self.path, self.line, self.marker
        )
    }
}

/// Whether `rel` (repository-relative, `/`-separated) is scanned at all.
#[must_use]
pub fn is_exempt_path(rel: &str) -> bool {
    // Only the Parquet implementation lives under `parquet/`; everything
    // else production is checked.
    if rel.starts_with("crates/krishiv-io/src/parquet/") {
        return true;
    }
    // Test support and fuzz targets are test code, not production code.
    if rel.starts_with("crates/krishiv-testkit/") || rel.starts_with("fuzz/") {
        return true;
    }
    // Test code: integration tests, unit-test files and bench sources.
    if rel.contains("/tests/") || rel.ends_with("/tests.rs") || rel.ends_with("tests.rs") {
        return true;
    }
    if rel.starts_with("crates/krishiv-io/benches/") {
        return true;
    }
    if rel.contains("/benches/") || rel.contains("benches/") {
        return true;
    }
    false
}

/// Whether the line is inside `with_defaults` of the registry constructor
/// (the one exemption): `false` outside `registry.rs`.
fn in_constructor(path: &str, lines: &[&str], line: usize) -> bool {
    if path != "crates/krishiv-io/src/registry.rs" {
        return false;
    }
    let mut depth = 0_i64;
    let mut inside = false;
    let mut opened = false;
    for (i, l) in lines.iter().enumerate() {
        let n = i + 1;
        if l.contains("fn with_defaults") {
            inside = true;
            depth = 0;
            opened = false;
        }
        if inside {
            depth += i64::try_from(l.bytes().filter(|&b| b == b'{').count()).unwrap_or(i64::MAX);
            depth -= i64::try_from(l.bytes().filter(|&b| b == b'}').count()).unwrap_or(i64::MAX);
            if depth > 0 {
                opened = true;
            }
            if n == line {
                return opened && depth > 0;
            }
            if opened && depth <= 0 {
                // Left the function: only lines between the fn start and
                // its closing brace count.
                if n >= line {
                    return false;
                }
                inside = false;
            }
        }
    }
    false
}

/// Violations in one file's text (empty when exempt).
#[must_use]
pub fn check_file(path: &str, text: &str) -> Vec<Violation> {
    if is_exempt_path(path) {
        return Vec::new();
    }
    let lines: Vec<&str> = text.lines().collect();
    let test_lines = test_region_lines(&lines);
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let n = i + 1;
        if test_lines.contains(&n) {
            continue;
        }
        // Comments still couple to the format; flag them too.
        let marker = if line.contains("krishiv_io::parquet") {
            "krishiv_io::parquet"
        } else if line.contains("parquet::") {
            "parquet::"
        } else {
            continue;
        };
        if in_constructor(path, &lines, n) {
            continue;
        }
        out.push(Violation {
            path: path.to_owned(),
            line: n,
            marker,
        });
    }
    out
}

/// 1-based line numbers inside `#[cfg(test)]` modules (inline test code).
fn test_region_lines(lines: &[&str]) -> std::collections::HashSet<usize> {
    let mut out = std::collections::HashSet::new();
    let mut pending_cfg = false;
    let mut pending_mod_brace = false;
    let mut depths: Vec<i64> = Vec::new();
    for (i, raw) in lines.iter().enumerate() {
        let n = i + 1;
        let line = strip_code(raw);
        // `#[cfg(test)]` and `mod` on one line, or the attribute alone.
        if line.contains("#[cfg(test)]") {
            if is_mod_decl(&line) {
                // Inline (`{`) or out-of-line (`;`): only inline counts.
                if line.contains('{') {
                    out.insert(n);
                    depths.push(brace_delta(&line));
                } else if !line.contains(';') {
                    pending_mod_brace = true;
                    out.insert(n);
                }
            } else {
                pending_cfg = true;
            }
            continue;
        }
        if pending_cfg {
            pending_cfg = false;
            if is_mod_decl(&line) {
                if line.contains('{') {
                    out.insert(n);
                    depths.push(brace_delta(&line));
                } else if !line.contains(';') {
                    pending_mod_brace = true;
                    out.insert(n);
                }
                continue;
            }
            // Not a module: the attribute belonged to something else.
        }
        if pending_mod_brace {
            out.insert(n);
            let delta = brace_delta(&line);
            if line.contains('{') {
                depths.push(delta);
                pending_mod_brace = false;
            } else if line.contains(';') {
                // `mod tests;`: out-of-line, no inline region.
                pending_mod_brace = false;
            }
            continue;
        }
        if let Some(d) = depths.last_mut() {
            out.insert(n);
            *d += brace_delta(&line);
            if *d <= 0 {
                depths.pop();
            }
        }
    }
    out
}

/// Whether `line` (code without strings/comments) declares a module.
fn is_mod_decl(line: &str) -> bool {
    let cleaned = line.replace(['(', ')', '{', '}', ';', ','], " ");
    let tokens = cleaned.split_whitespace().peekable();
    for tok in tokens {
        if tok == "mod" {
            return true;
        }
        // `pub mod`, `pub(crate) mod`, `pub(super) mod`: skip qualifiers.
        if tok == "pub" {
            continue;
        }
        if tok.starts_with("pub") {
            continue;
        }
    }
    false
}

/// The brace delta of `line`, ignoring braces in strings and comments.
fn brace_delta(line: &str) -> i64 {
    let opens = line.bytes().filter(|&b| b == b'{').count();
    let closes = line.bytes().filter(|&b| b == b'}').count();
    i64::try_from(opens).unwrap_or(i64::MAX) - i64::try_from(closes).unwrap_or(i64::MAX)
}

/// `line` without `//` comments and `"..."` string contents (braces there
/// do not open test modules or constructor bodies).
fn strip_code(line: &str) -> String {
    let code = match line.find("//") {
        Some(i) => &line[..i],
        None => line,
    };
    let mut out = String::with_capacity(code.len());
    let mut in_str = false;
    let mut escape = false;
    for c in code.chars() {
        if in_str {
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        } else if c == '"' {
            in_str = true;
            continue;
        }
        out.push(c);
    }
    out
}

/// Scans `root/crates/**/*.rs` for concrete-format references.
///
/// # Errors
/// Returns every violation found, or a filesystem error message.
pub fn run_check(root: &Path) -> Result<(), String> {
    let mut files = Vec::new();
    collect(&root.join("crates"), &mut files)?;
    let mut hits = Vec::new();
    for path in &files {
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .display()
            .to_string()
            .replace('\\', "/");
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        hits.extend(check_file(&rel, &text));
    }
    if hits.is_empty() {
        println!("check-format-neutral: OK ({} files)", files.len());
        Ok(())
    } else {
        Err(hits
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n"))
    }
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let read = std::fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
    let mut entries: Vec<std::fs::DirEntry> = Vec::new();
    for entry in read {
        entries.push(entry.map_err(|e| format!("cannot read an entry in {}: {e}", dir.display()))?);
    }
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            let name = entry.file_name();
            if name != "target" {
                collect(&path, out)?;
            }
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_concrete_format_names_in_production_code() {
        let hits = check_file(
            "crates/krishiv-exec/src/files.rs",
            "use krishiv_io::parquet::ParquetFormat;\n",
        );
        assert_eq!(hits.len(), 1);
        let hits = check_file(
            "crates/krishiv-session/src/session.rs",
            "let x = parquet::foo();\n",
        );
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn ignores_tests_benches_and_the_constructor() {
        assert!(
            check_file(
                "crates/krishiv-io/src/commit/tests.rs",
                "use crate::parquet::X;\n"
            )
            .is_empty()
        );
        assert!(
            check_file(
                "crates/krishiv-testkit/tests/x.rs",
                "use parquet::arrow::Y;\n"
            )
            .is_empty()
        );
        assert!(
            check_file(
                "crates/krishiv-io/benches/io.rs",
                "use krishiv_io::parquet::ParquetFormat;\n"
            )
            .is_empty()
        );
        let constructor = "impl FormatRegistry {\n    pub fn with_defaults() -> Self {\n        r.register(Arc::new(crate::parquet::ParquetFormat));\n    }\n}\n";
        assert!(check_file("crates/krishiv-io/src/registry.rs", constructor).is_empty());
        // The same reference outside the constructor still fails.
        let outside = "use crate::parquet::ParquetFormat;\nimpl FormatRegistry {\n    pub fn with_defaults() -> Self {\n        Self::new()\n    }\n}\n";
        assert_eq!(
            check_file("crates/krishiv-io/src/registry.rs", outside).len(),
            1
        );
        // The implementation itself is never checked.
        assert!(
            check_file(
                "crates/krishiv-io/src/parquet/mod.rs",
                "use parquet::basic::X;\n"
            )
            .is_empty()
        );
    }

    #[test]
    fn the_constructor_exemption_ends_at_its_closing_brace() {
        let after = "impl FormatRegistry {\n    pub fn with_defaults() -> Self {\n        Self::new()\n    }\n}\nuse crate::parquet::ParquetFormat;\n";
        let hits = check_file("crates/krishiv-io/src/registry.rs", after);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].line, 6);
        // Only the real registry file is exempt, not any `registry.rs`.
        let other = "pub fn with_defaults() -> Self {\n    crate::parquet::X::new()\n}\n";
        assert_eq!(check_file("crates/other/src/registry.rs", other).len(), 1);
    }

    #[test]
    fn test_modules_count_whatever_their_header_shape() {
        for (text, line) in [
            (
                "#[cfg(test)]\nmod tests {\nuse parquet::X;\n}\nuse parquet::Y;\n",
                5,
            ),
            (
                "#[cfg(test)] mod tests {\nuse parquet::X;\n}\nuse parquet::Y;\n",
                4,
            ),
            (
                "#[cfg(test)]\npub mod tests {\nuse parquet::X;\n}\nuse parquet::Y;\n",
                5,
            ),
            (
                "#[cfg(test)]\nmod tests\n{\nuse parquet::X;\n}\nuse parquet::Y;\n",
                6,
            ),
        ] {
            let hits = check_file("crates/krishiv-exec/src/files.rs", text);
            assert_eq!(hits.len(), 1, "{text:?}: {hits:?}");
            assert_eq!(hits[0].line, line);
        }
        // Braces in strings and comments do not close the test module.
        let tricky =
            "#[cfg(test)]\nmod tests {\nlet s = \"{\"; // }\nuse parquet::X;\n}\nuse parquet::Y;\n";
        let hits = check_file("crates/krishiv-exec/src/files.rs", tricky);
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].line, 6);
    }
}
