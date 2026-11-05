// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask file-size [--write-baseline]`: no hand-written Rust file under a crate's `src/` grows
//! past [`MAX_LINES`] lines (owner rule, 2026-11-01).
//!
//! Files already over the limit are listed with their size in `xtask/file-size-baseline.txt` (one
//! `<lines> <path>` per line, sorted by path): they may shrink, never grow. A new file over the limit
//! fails; a file that grew past its baseline fails. A PR that needs a bigger file raises that one line
//! in the baseline, which shows in review. A shrink passes and is reported, so concurrent PRs do not
//! all have to edit the same baseline line; `--write-baseline` rewrites the file from the tree and
//! drops files that are back under the limit. Test files count when they sit under `src/`; the `tests/`
//! directories, generated code (`*.pb.rs`, `@generated`) and fixtures are not scanned.
#![expect(clippy::print_stdout, reason = "command output is the interface")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::map::{MAX_LINES, is_generated};

/// The checked-in baseline, relative to the repository root.
pub const BASELINE: &str = "xtask/file-size-baseline.txt";

/// What a comparison of the tree with the baseline found.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Files over the limit that are not in the baseline: `(path, lines)`.
    pub new_over: Vec<(String, usize)>,
    /// Baseline files that grew: `(path, baseline, lines)`.
    pub grown: Vec<(String, usize, usize)>,
    /// Baseline files that shrank but are still over the limit: `(path, baseline, lines)`.
    pub shrunk: Vec<(String, usize, usize)>,
    /// Baseline entries whose file is back under the limit or gone.
    pub stale: Vec<String>,
}

impl Report {
    /// The number of findings that fail the check.
    #[must_use]
    pub fn failures(&self) -> usize {
        self.new_over.len() + self.grown.len()
    }
}

/// Compares `current` (every hand-written file's line count) with `baseline`.
#[must_use]
pub fn compare(current: &BTreeMap<String, usize>, baseline: &BTreeMap<String, usize>) -> Report {
    let mut r = Report::default();
    for (path, &lines) in current {
        match baseline.get(path) {
            _ if lines <= MAX_LINES => {}
            None => r.new_over.push((path.clone(), lines)),
            Some(&was) if lines > was => r.grown.push((path.clone(), was, lines)),
            Some(&was) if lines < was => r.shrunk.push((path.clone(), was, lines)),
            Some(_) => {}
        }
    }
    for path in baseline.keys() {
        if current.get(path).is_none_or(|&l| l <= MAX_LINES) {
            r.stale.push(path.clone());
        }
    }
    r
}

/// Parses the baseline text (`<lines> <path>` per line; blank lines and `#` comments are skipped).
///
/// # Errors
/// Returns an error naming the first malformed line.
pub fn parse_baseline(text: &str) -> Result<BTreeMap<String, usize>, String> {
    let mut out = BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (n, path) = line
            .split_once(' ')
            .ok_or_else(|| format!("{BASELINE}:{}: expected `<lines> <path>`", i + 1))?;
        let n = n
            .parse()
            .map_err(|_| format!("{BASELINE}:{}: `{n}` is not a line count", i + 1))?;
        out.insert(path.trim().to_owned(), n);
    }
    Ok(out)
}

/// The baseline text for the files over the limit in `current`.
#[must_use]
pub fn render_baseline(current: &BTreeMap<String, usize>) -> String {
    let mut out = format!(
        "# Hand-written Rust files over {MAX_LINES} lines: `<lines> <path>`. They may shrink, never grow\n\
         # (`cargo xtask file-size`; rewrite with `--write-baseline`). Raising a line needs a reason in review.\n"
    );
    for (path, lines) in current.iter().filter(|&(_, &l)| l > MAX_LINES) {
        out.push_str(&format!("{lines} {path}\n"));
    }
    out
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_dir() {
            rs_files(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// Line counts of every hand-written `.rs` file under `root/crates/*/src`, keyed by path from `root`.
///
/// # Errors
/// Returns an error when a directory or file cannot be read.
pub fn scan(root: &Path) -> Result<BTreeMap<String, usize>, String> {
    let crates = root.join("crates");
    let mut files = Vec::new();
    for entry in std::fs::read_dir(&crates).map_err(|e| format!("{}: {e}", crates.display()))? {
        let src = entry.map_err(|e| e.to_string())?.path().join("src");
        if src.is_dir() {
            rs_files(&src, &mut files)?;
        }
    }
    let mut out = BTreeMap::new();
    for file in files {
        let rel = file
            .strip_prefix(root)
            .unwrap_or(&file)
            .to_string_lossy()
            .replace('\\', "/");
        let text =
            std::fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?;
        if !is_generated(&rel, &text) {
            out.insert(rel, text.lines().count());
        }
    }
    Ok(out)
}

/// `xtask file-size [--write-baseline]`.
///
/// # Errors
/// Returns an error for unknown flags, unreadable files, a malformed baseline, or any failing finding.
pub fn run(root: &Path, flags: &[&str]) -> Result<(), String> {
    let write = match flags {
        [] => false,
        ["--write-baseline"] => true,
        other => return Err(format!("unknown flags {other:?}")),
    };
    let current = scan(root)?;
    let path = root.join(BASELINE);
    if write {
        std::fs::write(&path, render_baseline(&current))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        println!("file-size: wrote {BASELINE}");
        return Ok(());
    }
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let report = compare(&current, &parse_baseline(&text)?);
    for (f, n) in &report.new_over {
        println!("{f}: {n} lines, over {MAX_LINES} and not in {BASELINE}: split it");
    }
    for (f, was, n) in &report.grown {
        println!(
            "{f}: grew from {was} to {n} lines: split it, or raise its line in {BASELINE} with a reason"
        );
    }
    for (f, was, n) in &report.shrunk {
        println!(
            "note: {f} shrank from {was} to {n} lines: lower its baseline (`--write-baseline`)"
        );
    }
    for f in &report.stale {
        println!(
            "note: {f} is under the limit or gone: drop it from {BASELINE} (`--write-baseline`)"
        );
    }
    println!("file-size: {} failing finding(s)", report.failures());
    if report.failures() == 0 {
        Ok(())
    } else {
        Err(format!(
            "file-size: {} failing finding(s)",
            report.failures()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(items: &[(&str, usize)]) -> BTreeMap<String, usize> {
        items.iter().map(|(p, n)| ((*p).to_owned(), *n)).collect()
    }

    #[test]
    fn a_file_over_the_limit_that_grew_fails() {
        let r = compare(&map(&[("a.rs", 2600)]), &map(&[("a.rs", 2500)]));
        assert_eq!(r.grown, [("a.rs".to_owned(), 2500, 2600)]);
        assert_eq!(r.failures(), 1);
    }

    #[test]
    fn a_file_that_shrank_passes_and_is_reported() {
        let r = compare(&map(&[("a.rs", 2400)]), &map(&[("a.rs", 2500)]));
        assert_eq!(r.failures(), 0);
        assert_eq!(r.shrunk, [("a.rs".to_owned(), 2500, 2400)]);
    }

    #[test]
    fn a_new_file_over_the_limit_fails_and_one_at_the_limit_passes() {
        let r = compare(&map(&[("n.rs", 2001), ("ok.rs", 2000)]), &map(&[]));
        assert_eq!(r.new_over, [("n.rs".to_owned(), 2001)]);
        assert_eq!(r.failures(), 1);
    }

    #[test]
    fn an_unchanged_file_passes_and_a_file_back_under_the_limit_is_stale() {
        let r = compare(
            &map(&[("same.rs", 2500), ("small.rs", 1900)]),
            &map(&[("same.rs", 2500), ("small.rs", 2100), ("gone.rs", 3000)]),
        );
        assert_eq!(r.failures(), 0);
        assert_eq!(r.stale, ["gone.rs", "small.rs"]);
    }

    #[test]
    fn the_baseline_round_trips() {
        let cur = map(&[("a.rs", 2500), ("b.rs", 100), ("c/d.rs", 2001)]);
        let text = render_baseline(&cur);
        assert_eq!(
            parse_baseline(&text).unwrap(),
            map(&[("a.rs", 2500), ("c/d.rs", 2001)])
        );
        assert!(parse_baseline("12x a.rs").is_err());
        assert!(parse_baseline("nospace").is_err());
    }

    #[test]
    fn generated_files_and_tests_directories_are_not_scanned() {
        let dir = tempfile::tempdir().unwrap();
        let big = "fn f() {}\n".repeat(2500);
        let put = |rel: &str, text: &str| {
            let p = dir.path().join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, text).unwrap();
        };
        put("crates/c/src/hand.rs", &big);
        put("crates/c/src/gen.pb.rs", &big);
        put("crates/c/src/marked.rs", &format!("// @generated\n{big}"));
        put("crates/c/tests/it.rs", &big);
        let got = scan(dir.path()).unwrap();
        assert_eq!(got.keys().collect::<Vec<_>>(), ["crates/c/src/hand.rs"]);
    }
}
