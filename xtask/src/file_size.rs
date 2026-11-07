// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask file-size [--write-baseline [--base REF] [--only PATH]...]`: no hand-written Rust file under a crate's `src/` grows
//! past [`MAX_LINES`] lines (owner rule, 2026-11-01).
//!
//! `xtask/file-size-baseline.txt` is unused (the check no longer reads it) and goes away in a follow-up; it stays
//! untouched until then so PRs that still edit it merge cleanly.
//!
//! Files already over the limit are listed with their size in `xtask/file-size-baseline/<crate>.txt` (one
//! `<lines> <path>` per line, sorted by path; one file per crate so PRs in different crates never
//! conflict on the list): they may shrink, never grow. A new file over the limit
//! fails; a file that grew past its baseline fails. A PR that needs a bigger file raises that one line
//! in the baseline, which shows in review. A shrink passes and is reported, so concurrent PRs do not
//! all have to edit the same baseline line. `--write-baseline` updates only the lines of files this
//! branch changed (against the merge base with `origin/main`, plus uncommitted and untracked files, or
//! the paths given with `--only`): it sets a changed file's line and drops it when it is back under the
//! limit, and leaves every other line alone, so two PRs never fight over a file neither changed. Lowering
//! the lines of unchanged files is left to a main-only job. Test files count when they sit under `src/`; the `tests/`
//! directories, generated code (`*.pb.rs`, `@generated`) and fixtures are not scanned.
#![expect(clippy::print_stdout, reason = "command output is the interface")]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::map::{MAX_LINES, is_generated};

/// The checked-in baseline directory, relative to the repository root: one `<crate>.txt` per crate.
pub const BASELINE: &str = "xtask/file-size-baseline";

/// The crate a source path belongs to (`crates/<crate>/...`), or `other`; names the baseline file.
fn crate_of(path: &str) -> &str {
    path.strip_prefix("crates/")
        .and_then(|rest| rest.split('/').next())
        .unwrap_or("other")
}

/// Reads every `*.txt` under the baseline directory into one map.
///
/// # Errors
/// Returns an error when the directory or a file cannot be read, a line is malformed, or a path is
/// listed twice.
pub fn load_baseline(dir: &Path) -> Result<BTreeMap<String, usize>, String> {
    let mut out = BTreeMap::new();
    // Git does not track an empty directory: no baseline files means nothing is over the limit.
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(format!("{}: {e}", dir.display())),
    };
    for entry in entries {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.extension().is_none_or(|e| e != "txt") {
            continue;
        }
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        for (file, lines) in parse_baseline(&text)? {
            if out.insert(file.clone(), lines).is_some() {
                return Err(format!(
                    "{}: {file} is listed in two baseline files",
                    path.display()
                ));
            }
        }
    }
    Ok(out)
}

/// Whether baseline text still has an entry (not just comments and blank lines).
fn has_entries(text: &str) -> bool {
    text.lines()
        .any(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
}

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
            .ok_or_else(|| format!("baseline line {}: expected `<lines> <path>`", i + 1))?;
        let n = n
            .parse()
            .map_err(|_| format!("baseline line {}: `{n}` is not a line count", i + 1))?;
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

/// `old_text` (the baseline file) with only the `touched` files' entries changed: a touched file's line is
/// replaced by its current size, dropped when it is back under the limit, or inserted in path order when
/// new. Comment lines (the reasons for raised lines) and every other entry stay exactly as they were.
#[must_use]
pub fn merge_baseline_text(
    old_text: &str,
    current: &BTreeMap<String, usize>,
    touched: &BTreeSet<String>,
) -> String {
    let mut pending: BTreeMap<&str, usize> = touched
        .iter()
        .filter_map(|p| match current.get(p) {
            Some(&l) if l > MAX_LINES => Some((p.as_str(), l)),
            _ => None,
        })
        .collect();
    let mut out = String::new();
    for line in old_text.lines() {
        let entry = if line.trim_start().starts_with('#') {
            None
        } else {
            line.trim().split_once(' ').map(|(_, p)| p.trim())
        };
        if let Some(path) = entry {
            // Insert new files that sort before this entry.
            let before: Vec<&str> = pending.keys().copied().filter(|k| *k < path).collect();
            for k in before {
                if let Some(l) = pending.remove(k) {
                    out.push_str(&format!("{l} {k}\n"));
                }
            }
            if touched.contains(path) {
                if let Some(l) = pending.remove(path) {
                    out.push_str(&format!("{l} {path}\n"));
                }
                continue;
            }
        }
        out.push_str(line);
        out.push('\n');
    }
    for (k, l) in pending {
        out.push_str(&format!("{l} {k}\n"));
    }
    out
}

fn git_lines(root: &Path, args: &[&str]) -> Result<Vec<String>, String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
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

/// The files this branch changed: committed since the merge base with `base`, uncommitted, and untracked.
fn changed_files(root: &Path, base: &str) -> Result<BTreeSet<String>, String> {
    let range = format!("{base}...HEAD");
    let mut out = BTreeSet::new();
    out.extend(git_lines(root, &["diff", "--name-only", &range])?);
    out.extend(git_lines(root, &["diff", "--name-only", "HEAD"])?);
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

/// `xtask file-size [--write-baseline [--base REF] [--only PATH]...]`.
///
/// # Errors
/// Returns an error for unknown flags, unreadable files, a malformed baseline, a failed `git` call, or
/// any failing finding.
pub fn run(root: &Path, flags: &[&str]) -> Result<(), String> {
    let mut write = false;
    let mut base = "origin/main".to_owned();
    let mut only: Vec<String> = Vec::new();
    let mut it = flags.iter();
    while let Some(&flag) = it.next() {
        match flag {
            "--write-baseline" => write = true,
            "--base" => base = (*it.next().ok_or("--base needs a ref")?).to_owned(),
            "--only" => only.push((*it.next().ok_or("--only needs a path")?).to_owned()),
            other => return Err(format!("unknown flags {other:?}")),
        }
    }
    if !write && (!only.is_empty() || base != "origin/main") {
        return Err("--base and --only go with --write-baseline".to_owned());
    }
    let current = scan(root)?;
    let dir = root.join(BASELINE);
    if write {
        let touched = if only.is_empty() {
            changed_files(root, &base)?
        } else {
            only.into_iter().collect()
        };
        let mut by_crate: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
        for file in &touched {
            by_crate
                .entry(crate_of(file))
                .or_default()
                .insert(file.clone());
        }
        // A touched file that is not under a crate's src/ is never in the baseline; skip crates with no change to make.
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for (krate, files) in by_crate {
            let path = dir.join(format!("{krate}.txt"));
            let old = match std::fs::read_to_string(&path) {
                Ok(text) => text,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    render_baseline(&BTreeMap::new())
                }
                Err(e) => return Err(format!("{}: {e}", path.display())),
            };
            parse_baseline(&old)?;
            let new = merge_baseline_text(&old, &current, &files);
            if has_entries(&new) {
                if new != old {
                    std::fs::write(&path, new).map_err(|e| format!("{}: {e}", path.display()))?;
                }
            } else if path.exists() {
                std::fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            }
        }
        println!("file-size: updated {BASELINE}/ for the files this branch changed");
        return Ok(());
    }
    let report = compare(&current, &load_baseline(&dir)?);
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
            "note: {f} shrank from {was} to {n} lines: lower its baseline (`--write-baseline --only {f}`, or leave it to the main-only refresh)"
        );
    }
    for f in &report.stale {
        println!(
            "note: {f} is under the limit or gone: drop it from {BASELINE} (`--write-baseline --only {f}`, or leave it to the main-only refresh)"
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
    fn writing_the_baseline_touches_only_the_changed_files_and_keeps_comments() {
        let old = "# header\n4582 a.rs\n3000 b.rs\n# why c grew\n2500 c.rs\n";
        let cur = map(&[
            ("a.rs", 4543),
            ("b.rs", 3100),
            ("c.rs", 1000),
            ("a0.rs", 2100),
            ("z.rs", 2200),
        ]);
        let touched: BTreeSet<String> = ["b.rs", "c.rs", "a0.rs", "z.rs"].map(str::to_owned).into();
        // a.rs shrank on its own but was not changed here: its line stays.
        assert_eq!(
            merge_baseline_text(old, &cur, &touched),
            "# header\n4582 a.rs\n2100 a0.rs\n3100 b.rs\n# why c grew\n2200 z.rs\n"
        );
    }

    #[test]
    fn write_baseline_with_only_leaves_other_lines_alone() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("crates/c/src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(dir.path().join(BASELINE)).unwrap();
        let lines = |n: usize| "fn f() {}\n".repeat(n);
        std::fs::write(src.join("a.rs"), lines(2100)).unwrap();
        std::fs::write(src.join("b.rs"), lines(2100)).unwrap();
        std::fs::write(
            dir.path().join(BASELINE).join("c.txt"),
            "2300 crates/c/src/a.rs\n2100 crates/c/src/b.rs\n",
        )
        .unwrap();
        std::fs::write(src.join("b.rs"), lines(2200)).unwrap();
        run(
            dir.path(),
            &["--write-baseline", "--only", "crates/c/src/b.rs"],
        )
        .unwrap();
        let text = std::fs::read_to_string(dir.path().join(BASELINE).join("c.txt")).unwrap();
        assert!(text.contains("2300 crates/c/src/a.rs") && text.contains("2200 crates/c/src/b.rs"));
        assert!(run(dir.path(), &["--only", "x"]).is_err());
    }

    #[test]
    fn write_baseline_finds_the_changed_files_through_git() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let git = |args: &[&str]| {
            let st = std::process::Command::new("git")
                .arg("-C")
                .arg(root)
                .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
                .args(args)
                .status()
                .unwrap();
            assert!(st.success(), "git {args:?}");
        };
        let src = root.join("crates/c/src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(root.join(BASELINE)).unwrap();
        let lines = |n: usize| "fn f() {}\n".repeat(n);
        std::fs::write(src.join("a.rs"), lines(2100)).unwrap();
        std::fs::write(src.join("b.rs"), lines(2100)).unwrap();
        std::fs::write(
            root.join(BASELINE).join("c.txt"),
            "# why\n2300 crates/c/src/a.rs\n2100 crates/c/src/b.rs\n",
        )
        .unwrap();
        git(&["init", "-q", "-b", "main"]);
        git(&["add", "-A"]);
        git(&["commit", "-q", "-m", "base"]);
        git(&["checkout", "-q", "-b", "topic"]);
        // b.rs grows in a commit, new.rs is untracked, a.rs shrank but is not changed here.
        std::fs::write(src.join("b.rs"), lines(2200)).unwrap();
        git(&["commit", "-q", "-am", "grow b"]);
        std::fs::write(src.join("new.rs"), lines(2001)).unwrap();
        run(root, &["--write-baseline", "--base", "main"]).unwrap();
        let text = std::fs::read_to_string(root.join(BASELINE).join("c.txt")).unwrap();
        assert_eq!(
            text,
            "# why\n2300 crates/c/src/a.rs\n2200 crates/c/src/b.rs\n2001 crates/c/src/new.rs\n"
        );
        assert!(run(root, &["--write-baseline", "--base", "nope"]).is_err());
        assert!(run(root, &["--base"]).is_err());
        assert!(run(root, &["--only"]).is_err());
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
    fn run_writes_the_baseline_then_fails_on_growth_and_new_files() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("crates/c/src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(dir.path().join(BASELINE)).unwrap();
        let lines = |n: usize| "fn f() {}\n".repeat(n);
        std::fs::write(src.join("big.rs"), lines(2100)).unwrap();
        std::fs::write(src.join("small.rs"), lines(10)).unwrap();
        assert!(run(dir.path(), &["--bogus"]).is_err());
        // No baseline yet: the big file is new.
        assert!(run(dir.path(), &[]).is_err());
        run(
            dir.path(),
            &["--write-baseline", "--only", "crates/c/src/big.rs"],
        )
        .unwrap();
        let text = std::fs::read_to_string(dir.path().join(BASELINE).join("c.txt")).unwrap();
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
        std::fs::write(dir.path().join(BASELINE).join("c.txt"), "oops").unwrap();
        assert!(run(dir.path(), &[]).is_err());
    }

    #[test]
    fn the_baseline_is_one_file_per_crate_and_a_path_listed_twice_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let put = |rel: &str, text: &str| {
            let p = root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, text).unwrap();
        };
        let lines = |n: usize| "fn f() {}\n".repeat(n);
        put("crates/a/src/x.rs", &lines(2100));
        put("crates/b/src/y.rs", &lines(2100));
        put("xtask/ignored.md", "not a baseline");
        run(
            root,
            &[
                "--write-baseline",
                "--only",
                "crates/a/src/x.rs",
                "--only",
                "crates/b/src/y.rs",
            ],
        )
        .unwrap();
        let read = |c: &str| std::fs::read_to_string(root.join(BASELINE).join(c)).unwrap();
        assert!(
            read("a.txt").contains("2100 crates/a/src/x.rs") && !read("a.txt").contains("y.rs")
        );
        assert!(read("b.txt").contains("2100 crates/b/src/y.rs"));
        run(root, &[]).unwrap();
        // A file back under the limit empties its crate's baseline file, which is then removed.
        put("crates/a/src/x.rs", &lines(10));
        run(root, &["--write-baseline", "--only", "crates/a/src/x.rs"]).unwrap();
        assert!(
            !root.join(BASELINE).join("a.txt").exists()
                && root.join(BASELINE).join("b.txt").exists()
        );
        // The same path in two files is a malformed baseline.
        put(
            "xtask/file-size-baseline/dup.txt",
            "2100 crates/b/src/y.rs\n",
        );
        assert!(run(root, &[]).is_err());
        // No baseline directory at all is an empty baseline: only the files over the limit fail.
        std::fs::remove_dir_all(root.join(BASELINE)).unwrap();
        put("crates/b/src/y.rs", &lines(10));
        run(root, &[]).unwrap();
    }
}
