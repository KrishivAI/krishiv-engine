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
    if !path.ends_with("crates/krishiv-io/src/registry.rs") && !path.ends_with("registry.rs") {
        return false;
    }
    let mut depth = 0_i64;
    let mut inside = false;
    for (i, l) in lines.iter().enumerate() {
        let n = i + 1;
        if l.contains("fn with_defaults") {
            inside = true;
            depth = 0;
        }
        if inside {
            depth += i64::try_from(l.bytes().filter(|&b| b == b'{').count()).unwrap_or(i64::MAX);
            depth -= i64::try_from(l.bytes().filter(|&b| b == b'}').count()).unwrap_or(i64::MAX);
            if n == line {
                return true;
            }
            if depth <= 0 && n > line_of_fn(lines, "fn with_defaults") {
                // Left the function without hitting the line: only lines
                // between the fn start and its closing brace count.
                if n >= line {
                    return false;
                }
            }
        }
    }
    false
}

fn line_of_fn(lines: &[&str], pat: &str) -> usize {
    lines
        .iter()
        .position(|l| l.contains(pat))
        .map_or(usize::MAX, |i| i + 1)
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
    let mut depth: Option<i64> = None;
    for (i, line) in lines.iter().enumerate() {
        let n = i + 1;
        let trimmed = line.trim_start();
        if trimmed.starts_with("#[cfg(test)]") {
            pending_cfg = true;
            continue;
        }
        if pending_cfg {
            if trimmed.starts_with("mod ") {
                let opens =
                    i64::try_from(line.bytes().filter(|&b| b == b'{').count()).unwrap_or(i64::MAX);
                let closes =
                    i64::try_from(line.bytes().filter(|&b| b == b'}').count()).unwrap_or(i64::MAX);
                out.insert(n);
                depth = (opens - closes > 0).then_some(opens - closes);
            }
            pending_cfg = false;
            continue;
        }
        if let Some(d) = depth {
            out.insert(n);
            let opens =
                i64::try_from(line.bytes().filter(|&b| b == b'{').count()).unwrap_or(i64::MAX);
            let closes =
                i64::try_from(line.bytes().filter(|&b| b == b'}').count()).unwrap_or(i64::MAX);
            let nd = d + opens - closes;
            depth = (nd > 0).then_some(nd);
        }
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
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
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
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
    let mut entries: Vec<_> = entries.flatten().collect();
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
}
