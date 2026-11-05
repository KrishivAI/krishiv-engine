// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask file-size [--write-baseline]`: no Rust source file grows past [`MAX_LINES`] lines.
//!
//! A ratchet on hand-written files under `crates/*/src/` and `xtask/src/` (generated files, `*.pb.rs` or
//! `@generated` in the first lines, are ignored; tests and fixtures outside `src/` are not looked at):
//!
//! * a file over the limit that is not in the baseline `xtask/file-size-baseline.json` fails (a new file, or an
//!   existing one that grew past the limit);
//! * a file in the baseline that is longer than its recorded size fails (it may shrink, never grow);
//! * a file in the baseline that is shorter than recorded, or no longer over the limit, or gone, fails too until
//!   the baseline is lowered (`cargo xtask file-size --write-baseline`), so the allowance only ever goes down.
//!
//! A PR that needs a file to grow raises that file's entry in the baseline by hand; the change shows in review.
//! `maps-check` still lists the long files; this is the gate.
#![expect(clippy::print_stdout, reason = "command output is the interface")]

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::map::{MAX_LINES, is_generated, rs_files};

/// The baseline file, relative to the repository root.
pub const BASELINE: &str = "xtask/file-size-baseline.json";

/// The on-disk baseline: the files over the limit and their allowed sizes.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Baseline {
    /// Format version of this file.
    pub format_version: u32,
    /// The limit the baseline was written for ([`MAX_LINES`]).
    pub max_lines: usize,
    /// Repository-relative path to the allowed number of lines.
    pub files: BTreeMap<String, usize>,
}

/// What a comparison of the current sizes with the baseline found.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Verdict {
    /// Over the limit and not in the baseline: `(path, lines)`.
    pub new_over: Vec<(String, usize)>,
    /// In the baseline and longer than recorded: `(path, allowed, lines)`.
    pub grown: Vec<(String, usize, usize)>,
    /// In the baseline but shorter, no longer over the limit or gone: `(path, allowed, lines or None)`.
    pub stale: Vec<(String, usize, Option<usize>)>,
}

impl Verdict {
    /// How many findings there are.
    #[must_use]
    pub fn count(&self) -> usize {
        self.new_over.len() + self.grown.len() + self.stale.len()
    }
}

/// The hand-written Rust files under `root` that are over [`MAX_LINES`] lines, with their line counts.
///
/// # Errors
/// Returns an error when a directory or file cannot be read.
pub fn scan(root: &Path) -> Result<BTreeMap<String, usize>, String> {
    let mut dirs = vec![root.join("xtask/src")];
    let crates = root.join("crates");
    if crates.is_dir() {
        let entries =
            std::fs::read_dir(&crates).map_err(|e| format!("{}: {e}", crates.display()))?;
        for entry in entries {
            let src = entry.map_err(|e| e.to_string())?.path().join("src");
            if src.is_dir() {
                dirs.push(src);
            }
        }
    }
    let mut out = BTreeMap::new();
    for dir in dirs.into_iter().filter(|d| d.is_dir()) {
        let mut files = Vec::new();
        rs_files(&dir, &mut files)?;
        for file in files {
            let text =
                std::fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?;
            let rel = file
                .strip_prefix(root)
                .map_err(|e| e.to_string())?
                .to_string_lossy()
                .replace('\\', "/");
            let lines = text.lines().count();
            if lines > MAX_LINES && !is_generated(&rel, &text) {
                out.insert(rel, lines);
            }
        }
    }
    Ok(out)
}

/// Compares the current sizes (files over the limit) with the baseline's allowances.
#[must_use]
pub fn compare(current: &BTreeMap<String, usize>, allowed: &BTreeMap<String, usize>) -> Verdict {
    let mut v = Verdict::default();
    for (path, &lines) in current {
        match allowed.get(path) {
            None => v.new_over.push((path.clone(), lines)),
            Some(&max) if lines > max => v.grown.push((path.clone(), max, lines)),
            Some(&max) if lines < max => v.stale.push((path.clone(), max, Some(lines))),
            Some(_) => {}
        }
    }
    for (path, &max) in allowed {
        if !current.contains_key(path) {
            v.stale.push((path.clone(), max, None));
        }
    }
    v
}

/// Reads the baseline at `root`/[`BASELINE`].
///
/// # Errors
/// Returns an error when the file is missing, unreadable, malformed or written for another limit.
pub fn read_baseline(root: &Path) -> Result<Baseline, String> {
    let path = root.join(BASELINE);
    let text = std::fs::read_to_string(&path).map_err(|e| {
        format!(
            "{}: {e} (create it with `cargo xtask file-size --write-baseline`)",
            path.display()
        )
    })?;
    let baseline: Baseline =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    if baseline.max_lines != MAX_LINES {
        return Err(format!(
            "{BASELINE} is for a limit of {} lines, the limit is {MAX_LINES}: rewrite it with --write-baseline",
            baseline.max_lines
        ));
    }
    Ok(baseline)
}

/// Writes the baseline for the current sizes.
///
/// # Errors
/// Returns an error when the file cannot be written.
pub fn write_baseline(root: &Path, files: BTreeMap<String, usize>) -> Result<(), String> {
    let baseline = Baseline {
        format_version: 1,
        max_lines: MAX_LINES,
        files,
    };
    let path = root.join(BASELINE);
    let mut text = serde_json::to_string_pretty(&baseline).map_err(|e| e.to_string())?;
    text.push('\n');
    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))
}

/// `xtask file-size [--write-baseline]`.
///
/// # Errors
/// Returns an error for unknown flags, unreadable files, a missing baseline or any finding.
pub fn run_file_size(root: &Path, flags: &[&str]) -> Result<(), String> {
    let current = scan(root)?;
    match flags {
        [] => {}
        ["--write-baseline"] => {
            let n = current.len();
            write_baseline(root, current)?;
            println!("file-size: wrote {BASELINE} ({n} file(s) over {MAX_LINES} lines)");
            return Ok(());
        }
        other => return Err(format!("unknown flags {other:?}")),
    }
    let baseline = read_baseline(root)?;
    let v = compare(&current, &baseline.files);
    for (path, lines) in &v.new_over {
        println!("{path}: {lines} lines, over {MAX_LINES} and not in {BASELINE}: split it");
    }
    for (path, max, lines) in &v.grown {
        println!(
            "{path}: {lines} lines, grew past its allowance of {max}: split it, or raise its entry in {BASELINE}"
        );
    }
    for (path, max, lines) in &v.stale {
        match lines {
            Some(n) => println!(
                "{path}: {n} lines, allowance {max}: lower it (`cargo xtask file-size --write-baseline`)"
            ),
            None => println!(
                "{path}: allowance {max} but the file is under {MAX_LINES} lines or gone: drop its entry (`cargo xtask file-size --write-baseline`)"
            ),
        }
    }
    println!("file-size: {} finding(s)", v.count());
    if v.count() == 0 {
        Ok(())
    } else {
        Err(format!("file-size: {} finding(s)", v.count()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
        entries.iter().map(|&(p, n)| (p.to_owned(), n)).collect()
    }

    #[test]
    fn a_file_that_matches_its_allowance_passes() {
        let v = compare(&map(&[("a.rs", 2500)]), &map(&[("a.rs", 2500)]));
        assert_eq!(v, Verdict::default());
    }

    #[test]
    fn a_file_that_grew_fails() {
        let v = compare(&map(&[("a.rs", 2501)]), &map(&[("a.rs", 2500)]));
        assert_eq!(v.grown, vec![("a.rs".to_owned(), 2500, 2501)]);
        assert_eq!(v.count(), 1);
    }

    #[test]
    fn a_file_that_shrank_asks_for_a_lower_baseline() {
        let v = compare(&map(&[("a.rs", 2400)]), &map(&[("a.rs", 2500)]));
        assert_eq!(v.stale, vec![("a.rs".to_owned(), 2500, Some(2400))]);
    }

    #[test]
    fn a_new_file_over_the_limit_fails() {
        let v = compare(&map(&[("b.rs", 2001)]), &map(&[]));
        assert_eq!(v.new_over, vec![("b.rs".to_owned(), 2001)]);
    }

    #[test]
    fn a_file_that_left_the_baseline_by_shrinking_under_the_limit_or_vanishing_is_stale() {
        let v = compare(&map(&[]), &map(&[("a.rs", 2500)]));
        assert_eq!(v.stale, vec![("a.rs".to_owned(), 2500, None)]);
    }
}
