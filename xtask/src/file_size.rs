// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask file-size [--write-baseline [--only PATH]... | --all]`: no hand-written Rust file under a crate's `src/` grows
//! past [`MAX_LINES`] lines (owner rule, 2026-11-01).
//!
//! Files already over the limit are listed with their size in `xtask/file-size-baseline.txt` (one
//! `<lines> <path>` per line, sorted by path): they may shrink, never grow. A new file over the limit
//! fails; a file that grew past its baseline fails. A PR that needs a bigger file raises that one line
//! in the baseline, which shows in review. A shrink passes and is reported, so concurrent PRs do not
//! all have to edit the same baseline line. `--write-baseline` updates only the lines of files this
//! branch changed (against the merge base with `origin/main`, working tree included), or of the
//! paths named with `--only`; every other line is kept as it is, so two PRs never fight over a file
//! neither changed. `--write-baseline --all` rewrites everything from the tree (for a scheduled
//! main-only job). Test files count when they sit under `src/`; the `tests/`
//! directories, generated code (`*.pb.rs`, `@generated`) and fixtures are not scanned.
#![expect(clippy::print_stdout, reason = "command output is the interface")]

use std::collections::{BTreeMap, BTreeSet};
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

/// The baseline text after updating only the lines of `only`: those paths take their size from
/// `current` (dropped when back under the limit or gone). Every other line of `existing_text`,
/// comments included, is kept verbatim; new entries go in sorted position.
#[must_use]
pub fn render_baseline_for(
    current: &BTreeMap<String, usize>,
    existing_text: &str,
    only: &BTreeSet<String>,
) -> String {
    let entry_path = |line: &str| -> Option<String> {
        let line = line.trim();
        if line.starts_with('#') {
            return None;
        }
        line.split_once(' ').map(|(_, p)| p.trim().to_owned())
    };
    let mut pending: BTreeMap<&String, usize> = only
        .iter()
        .filter_map(|p| {
            current
                .get(p)
                .copied()
                .filter(|&n| n > MAX_LINES)
                .map(|n| (p, n))
        })
        .collect();
    let mut out = String::new();
    // Comment lines stay attached to the entry that follows them.
    let mut comments = String::new();
    let mut wrote_entry = false;
    for line in existing_text.lines() {
        match entry_path(line) {
            Some(p) => {
                // New entries that sort before this line go first, so the file stays sorted.
                let before: Vec<(&String, usize)> = pending
                    .iter()
                    .filter(|(q, _)| q.as_str() < p.as_str())
                    .map(|(q, &n)| (*q, n))
                    .collect();
                if !before.is_empty() && !wrote_entry {
                    // The file header stays on top.
                    out.push_str(&std::mem::take(&mut comments));
                }
                for (q, n) in before {
                    out.push_str(&format!("{n} {q}\n"));
                    pending.remove(q);
                }
                wrote_entry = true;
                if only.contains(&p) {
                    match pending.remove(&p) {
                        Some(n) => {
                            out.push_str(&std::mem::take(&mut comments));
                            out.push_str(&format!("{n} {p}\n"));
                        }
                        None => comments.clear(),
                    }
                } else {
                    out.push_str(&std::mem::take(&mut comments));
                    out.push_str(line);
                    out.push('\n');
                }
            }
            None => {
                comments.push_str(line);
                comments.push('\n');
            }
        }
    }
    for (q, n) in pending {
        out.push_str(&format!("{n} {q}\n"));
    }
    out.push_str(&comments);
    out
}

fn git_lines(root: &Path, args: &[&str]) -> Result<Vec<String>, String> {
    let out = std::process::Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|e| format!("git {args:?}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_owned)
        .collect())
}

/// Files this branch changed against the merge base with `origin/main`, committed or not.
fn changed_files(root: &Path) -> Result<BTreeSet<String>, String> {
    let base = git_lines(root, &["merge-base", "origin/main", "HEAD"])
        .map_err(|e| format!("{e}; pass `--only <path>` or `--all`"))?
        .into_iter()
        .next()
        .ok_or("no merge base; pass `--only <path>` or `--all`")?;
    let mut out: BTreeSet<String> = git_lines(root, &["diff", "--name-only", &base])?
        .into_iter()
        .collect();
    out.extend(git_lines(
        root,
        &["ls-files", "--others", "--exclude-standard"],
    )?);
    Ok(out)
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

/// `xtask file-size [--write-baseline [--only PATH]... | --all]`.
///
/// # Errors
/// Returns an error for unknown flags, unreadable files, a malformed baseline, or any failing finding.
pub fn run(root: &Path, flags: &[&str]) -> Result<(), String> {
    /// What `--write-baseline` rewrites.
    enum Scope {
        Branch,
        All,
        Only(BTreeSet<String>),
    }
    let write = match flags {
        [] => None,
        ["--write-baseline"] => Some(Scope::Branch),
        ["--write-baseline", "--all"] => Some(Scope::All),
        ["--write-baseline", rest @ ..] if !rest.is_empty() => {
            let mut only = BTreeSet::new();
            let mut it = rest.iter();
            while let Some(&flag) = it.next() {
                match (flag, it.next()) {
                    ("--only", Some(&p)) => {
                        only.insert(p.to_owned());
                    }
                    _ => return Err(format!("unknown flags {flags:?}")),
                }
            }
            Some(Scope::Only(only))
        }
        other => return Err(format!("unknown flags {other:?}")),
    };
    let current = scan(root)?;
    let path = root.join(BASELINE);
    if let Some(scope) = write {
        let text = match scope {
            Scope::All => render_baseline(&current),
            Scope::Branch | Scope::Only(_) => {
                let only = match scope {
                    Scope::Only(o) => o,
                    _ => changed_files(root)?,
                };
                // A missing baseline starts empty, so the first write works.
                let existing = std::fs::read_to_string(&path)
                    .unwrap_or_else(|_| render_baseline(&BTreeMap::new()));
                parse_baseline(&existing)?;
                render_baseline_for(&current, &existing, &only)
            }
        };
        std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
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
            "note: {f} shrank from {was} to {n} lines: lower its baseline (`--write-baseline`, which touches only files you changed)"
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

    #[test]
    fn writing_for_some_paths_leaves_every_other_line_alone() {
        let current = map(&[
            ("mine.rs", 2300),
            ("shrunk_elsewhere.rs", 4543),
            ("fixed.rs", 1500),
            ("new.rs", 2100),
        ]);
        let existing =
            "# header\n2200 mine.rs\n# keep me\n4582 shrunk_elsewhere.rs\n2100 fixed.rs\n";
        let only: BTreeSet<String> = ["mine.rs", "fixed.rs", "new.rs"].map(str::to_owned).into();
        let text = render_baseline_for(&current, existing, &only);
        assert_eq!(
            text,
            "# header\n2300 mine.rs\n2100 new.rs\n# keep me\n4582 shrunk_elsewhere.rs\n"
        );
    }

    #[test]
    fn the_branch_scope_is_the_files_changed_since_the_merge_base() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let git = |args: &[&str]| {
            let st = std::process::Command::new("git")
                .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
                .args(args)
                .current_dir(root)
                .status()
                .unwrap();
            assert!(st.success(), "git {args:?}");
        };
        let src = root.join("crates/c/src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(root.join("xtask")).unwrap();
        let lines = |n: usize| "fn f() {}\n".repeat(n);
        std::fs::write(src.join("untouched.rs"), lines(2100)).unwrap();
        std::fs::write(src.join("mine.rs"), lines(2100)).unwrap();
        std::fs::write(
            root.join(BASELINE),
            "# keep\n2100 crates/c/src/mine.rs\n2200 crates/c/src/untouched.rs\n",
        )
        .unwrap();
        git(&["init", "-q", "-b", "main"]);
        git(&["add", "-A"]);
        git(&["commit", "-q", "-m", "base"]);
        git(&["update-ref", "refs/remotes/origin/main", "HEAD"]);
        git(&["checkout", "-q", "-b", "topic"]);
        // One committed change, one uncommitted, one new file: all count as changed.
        std::fs::write(src.join("mine.rs"), lines(2300)).unwrap();
        git(&["commit", "-q", "-am", "grow"]);
        std::fs::write(src.join("added.rs"), lines(2050)).unwrap();
        let changed = changed_files(root).unwrap();
        assert!(changed.contains("crates/c/src/mine.rs"));
        assert!(changed.contains("crates/c/src/added.rs"));
        assert!(!changed.contains("crates/c/src/untouched.rs"));
        run(root, &["--write-baseline"]).unwrap();
        assert_eq!(
            std::fs::read_to_string(root.join(BASELINE)).unwrap(),
            "# keep\n2050 crates/c/src/added.rs\n2300 crates/c/src/mine.rs\n2200 crates/c/src/untouched.rs\n"
        );
    }

    #[test]
    fn only_and_all_flags_parse_and_a_bad_flag_fails() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("crates/c/src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(dir.path().join("xtask")).unwrap();
        let lines = |n: usize| "fn f() {}\n".repeat(n);
        std::fs::write(src.join("a.rs"), lines(2100)).unwrap();
        std::fs::write(src.join("b.rs"), lines(2200)).unwrap();
        run(
            dir.path(),
            &["--write-baseline", "--only", "crates/c/src/a.rs"],
        )
        .unwrap();
        let t = std::fs::read_to_string(dir.path().join(BASELINE)).unwrap();
        assert!(t.contains("2100 crates/c/src/a.rs") && !t.contains("b.rs"));
        assert!(run(dir.path(), &["--write-baseline", "--only"]).is_err());
        assert!(run(dir.path(), &["--write-baseline", "--wat"]).is_err());
        // Outside a git repository the branch scope cannot be found: the error names the way out.
        let e = run(dir.path(), &["--write-baseline"]).unwrap_err();
        assert!(e.contains("--only"), "{e}");
    }

    #[test]
    fn run_writes_the_baseline_then_fails_on_growth_and_new_files() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("crates/c/src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(dir.path().join("xtask")).unwrap();
        let lines = |n: usize| "fn f() {}\n".repeat(n);
        std::fs::write(src.join("big.rs"), lines(2100)).unwrap();
        std::fs::write(src.join("small.rs"), lines(10)).unwrap();
        assert!(run(dir.path(), &["--bogus"]).is_err());
        // No baseline yet: the big file is new.
        assert!(run(dir.path(), &[]).is_err());
        run(dir.path(), &["--write-baseline", "--all"]).unwrap();
        let text = std::fs::read_to_string(dir.path().join(BASELINE)).unwrap();
        assert!(text.contains("2100 crates/c/src/big.rs") && !text.contains("small.rs"));
        run(dir.path(), &[]).unwrap();
        // A shrink and a stale entry pass (they are only reported).
        std::fs::write(src.join("big.rs"), lines(2050)).unwrap();
        run(dir.path(), &[]).unwrap();
        std::fs::write(src.join("big.rs"), lines(100)).unwrap();
        run(dir.path(), &[]).unwrap();
        // Growth and a new file over the limit fail.
        std::fs::write(src.join("big.rs"), lines(2200)).unwrap();
        assert!(run(dir.path(), &[]).is_err());
        std::fs::write(src.join("big.rs"), lines(2100)).unwrap();
        std::fs::write(src.join("new.rs"), lines(2001)).unwrap();
        assert!(run(dir.path(), &[]).is_err());
        // A malformed baseline is an error.
        std::fs::write(dir.path().join(BASELINE), "oops").unwrap();
        assert!(run(dir.path(), &[]).is_err());
    }
}
