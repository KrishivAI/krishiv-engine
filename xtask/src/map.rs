// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask map <crate>` and `cargo xtask maps-check` (P2-T193): a code map of a crate, and the
//! check that keeps the navigation aids true.
//!
//! `map` prints, for every `src/` file of a crate, its first `//!` line, its line count and its public
//! items (`pub fn`, `struct`, `enum`, `trait`, `const`, `static`, `type`, `mod`) with the first line of
//! their doc. The scanner is line-based (no parser): it reads items outside the first `#[cfg(test)]`
//! and ignores `pub(crate)` and `pub(super)`.
//!
//! `maps-check` fails (or with `--report`, only reports) on three things, in every crate under `crates/`:
//! (a) a `src/` file without a `//!` line; (b) a file or directory named in backticks in the crate's
//! `CLAUDE.md` `## Layout` section that is not in `src/` (`name.rs`, `name/`; longer paths are other
//! places and are not checked); (c) a hand-written file over [`MAX_LINES`] lines. Generated files
//! (`*.pb.rs`, or `@generated` in the first lines) are skipped.
#![expect(clippy::print_stdout, reason = "command output is the interface")]

use std::path::{Path, PathBuf};

/// Hand-written lines a file may have before it is listed.
pub const MAX_LINES: usize = 2000;

/// One `src/` file of a crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileMap {
    /// Path below the crate's `src/`.
    pub path: String,
    /// The first `//!` line, if any.
    pub doc: Option<String>,
    /// Lines in the file.
    pub lines: usize,
    /// Public items: `(declaration head, first doc line)`.
    pub items: Vec<(String, String)>,
}

/// What `maps-check` found in one crate.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Findings {
    /// `src/` files without a `//!` line.
    pub missing_docs: Vec<String>,
    /// Names in the `CLAUDE.md` layout that are not in `src/`.
    pub stale_names: Vec<String>,
    /// Hand-written files over [`MAX_LINES`], with their line counts.
    pub long_files: Vec<(String, usize)>,
}

impl Findings {
    /// How many findings there are.
    #[must_use]
    pub fn count(&self) -> usize {
        self.missing_docs.len() + self.stale_names.len() + self.long_files.len()
    }
}

/// The first `//!` line with text in the first 60 lines of `text`.
#[must_use]
pub fn module_doc(text: &str) -> Option<String> {
    text.lines().take(60).find_map(|l| {
        let rest = l.trim_start().strip_prefix("//!")?.trim();
        (!rest.is_empty()).then(|| rest.to_owned())
    })
}

/// Whether the file is generated code, which no hand edits or docs are expected for.
#[must_use]
pub fn is_generated(path: &str, text: &str) -> bool {
    path.ends_with(".pb.rs")
        || text
            .lines()
            .take(5)
            .any(|l| l.contains("@generated") || l.contains("Code generated"))
}

const ITEM_KEYWORDS: &[&str] = &[
    "fn",
    "async fn",
    "const fn",
    "unsafe fn",
    "struct",
    "enum",
    "trait",
    "type",
    "const",
    "static",
    "mod",
];

/// The public items of `text` outside its first `#[cfg(test)]`: the declaration up to its name, and
/// the first line of the doc above it.
#[must_use]
pub fn public_items(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut doc: Option<String> = None;
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with("#[cfg(test)]") {
            break;
        }
        if let Some(d) = t.strip_prefix("///") {
            if doc.is_none() && !d.trim().is_empty() {
                doc = Some(d.trim().to_owned());
            }
            continue;
        }
        if t.starts_with("#[") || t.starts_with("//") {
            continue;
        }
        if let Some(rest) = t.strip_prefix("pub ")
            && let Some(kw) = ITEM_KEYWORDS
                .iter()
                .find(|k| rest.strip_prefix(**k).is_some_and(|r| r.starts_with(' ')))
        {
            let after = rest[kw.len()..].trim_start();
            let name: String = after
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                out.push((format!("{kw} {name}"), doc.take().unwrap_or_default()));
            }
        }
        doc = None;
    }
    out
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for entry in entries {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_dir() {
            rs_files(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// The map of the crate at `crate_dir`: its hand-written `src/` files, sorted by path.
///
/// # Errors
/// Returns an error when `src/` or a file cannot be read.
pub fn scan_crate(crate_dir: &Path) -> Result<Vec<FileMap>, String> {
    let src = crate_dir.join("src");
    let mut files = Vec::new();
    rs_files(&src, &mut files)?;
    files.sort();
    let mut out = Vec::new();
    for file in files {
        let rel = file
            .strip_prefix(&src)
            .unwrap_or(&file)
            .to_string_lossy()
            .replace('\\', "/");
        let text =
            std::fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?;
        if is_generated(&rel, &text) {
            continue;
        }
        out.push(FileMap {
            doc: module_doc(&text),
            lines: text.lines().count(),
            items: public_items(&text),
            path: rel,
        });
    }
    Ok(out)
}

/// The text of `map`: a block per file.
#[must_use]
pub fn render_map(files: &[FileMap]) -> String {
    let mut out = String::new();
    for f in files {
        out.push_str(&format!(
            "{} ({} lines): {}\n",
            f.path,
            f.lines,
            f.doc.as_deref().unwrap_or("(no module doc)")
        ));
        for (head, doc) in &f.items {
            if doc.is_empty() {
                out.push_str(&format!("  {head}\n"));
            } else {
                out.push_str(&format!("  {head} - {doc}\n"));
            }
        }
    }
    out
}

/// The names in backticks in the `## Layout` section of a `CLAUDE.md` that look like a file in
/// `src/` (`name.rs`) or a directory of it (`name/`).
#[must_use]
pub fn layout_names(claude_md: &str) -> Vec<String> {
    let mut in_layout = false;
    let mut names = Vec::new();
    for line in claude_md.lines() {
        if let Some(h) = line.strip_prefix("## ") {
            in_layout = h.trim().starts_with("Layout");
            continue;
        }
        if !in_layout {
            continue;
        }
        for (i, part) in line.split('`').enumerate() {
            if i % 2 == 0 {
                continue;
            }
            let word = part.trim();
            let stem = word
                .strip_suffix(".rs")
                .or_else(|| word.strip_suffix('/'))
                .unwrap_or("");
            if !stem.is_empty()
                && stem.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                && !names.iter().any(|n| n == word)
            {
                names.push(word.to_owned());
            }
        }
    }
    names
}

/// What `maps-check` finds in the crate at `crate_dir`.
///
/// # Errors
/// Returns an error when the crate's files cannot be read.
pub fn check_crate(crate_dir: &Path) -> Result<Findings, String> {
    let files = scan_crate(crate_dir)?;
    let mut found = Findings::default();
    for f in &files {
        if f.doc.is_none() {
            found.missing_docs.push(f.path.clone());
        }
        if f.lines > MAX_LINES {
            found.long_files.push((f.path.clone(), f.lines));
        }
    }
    if let Ok(claude) = std::fs::read_to_string(crate_dir.join("CLAUDE.md")) {
        let exists = |name: &str| {
            files.iter().any(|f| {
                if let Some(dir) = name.strip_suffix('/') {
                    f.path.split('/').rev().skip(1).any(|p| p == dir)
                } else {
                    f.path.rsplit('/').next() == Some(name)
                }
            })
        };
        // Generated files are not in the map but are real files, and a crate keeps `fuzz/`,
        // `tests/` and the like beside `src/`: look at the disk for those.
        let on_disk = |name: &str| {
            if crate_dir.join(name.trim_end_matches('/')).exists() {
                return true;
            }
            let mut all = Vec::new();
            rs_files(&crate_dir.join("src"), &mut all).is_ok()
                && all.iter().any(|p| {
                    p.file_name().is_some_and(|n| n == name)
                        || name
                            .strip_suffix('/')
                            .is_some_and(|d| p.components().any(|c| c.as_os_str() == d))
                })
        };
        for name in layout_names(&claude) {
            if !exists(&name) && !on_disk(&name) {
                found.stale_names.push(name);
            }
        }
    }
    Ok(found)
}

/// The crates under `root/crates`, sorted.
fn crate_dirs(root: &Path) -> Result<Vec<PathBuf>, String> {
    let dir = root.join("crates");
    let mut out: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("src").is_dir())
        .collect();
    out.sort();
    Ok(out)
}

/// `xtask map <crate>`: prints the map of `crates/<crate>`.
///
/// # Errors
/// Returns an error for a missing crate or unreadable files.
pub fn run_map(root: &Path, flags: &[&str]) -> Result<(), String> {
    let [name] = flags else {
        return Err("usage: cargo xtask map <crate>".to_owned());
    };
    let dir = root.join("crates").join(name);
    if !dir.join("src").is_dir() {
        return Err(format!("no crate {name} under crates/"));
    }
    print!("{}", render_map(&scan_crate(&dir)?));
    Ok(())
}

/// `xtask maps-check [--report]`: lists the findings of every crate; fails when there are any, unless
/// `--report` (report only, for `precheck` until the check is a gate).
///
/// # Errors
/// Returns an error for unknown flags, unreadable files, or (without `--report`) any finding.
pub fn run_maps_check(root: &Path, flags: &[&str]) -> Result<(), String> {
    let report_only = match flags {
        [] => false,
        ["--report"] => true,
        other => return Err(format!("unknown flags {other:?}")),
    };
    let mut total = 0;
    for dir in crate_dirs(root)? {
        let name = dir
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        let found = check_crate(&dir)?;
        total += found.count();
        for f in &found.missing_docs {
            println!("{name}: src/{f} has no //! line");
        }
        for n in &found.stale_names {
            println!("{name}: CLAUDE.md layout names `{n}`, which is not in src/");
        }
        for (f, lines) in &found.long_files {
            println!("{name}: src/{f} has {lines} lines (over {MAX_LINES})");
        }
    }
    println!("maps-check: {total} finding(s)");
    if total == 0 || report_only {
        Ok(())
    } else {
        Err(format!("maps-check: {total} finding(s)"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_module_doc_is_the_first_line_with_text() {
        assert_eq!(
            module_doc("// SPDX\n//!\n//! Joins rows.\n//! More.\nuse a;").as_deref(),
            Some("Joins rows.")
        );
        assert_eq!(module_doc("// SPDX\nuse a;\n"), None);
    }

    #[test]
    fn public_items_skip_restricted_and_test_code() {
        let text = "/// Adds.\n#[must_use]\npub fn add() {}\npub(crate) fn hidden() {}\n\npub struct S;\n/// A kind.\n/// More.\npub enum Kind {}\npub async fn go() {}\n#[cfg(test)]\nmod tests { pub fn t() {} }\n";
        assert_eq!(
            public_items(text),
            [
                ("fn add".to_owned(), "Adds.".to_owned()),
                ("struct S".to_owned(), String::new()),
                ("enum Kind".to_owned(), "A kind.".to_owned()),
                ("async fn go".to_owned(), String::new()),
            ]
        );
    }

    #[test]
    fn layout_names_are_the_files_and_directories_in_backticks() {
        let md = "# c\n## Purpose\n`skip.rs`\n## Layout (P2)\n`a.rs` and `dir/` · `tests/fixtures/x.json` · `Foo::bar` `a.rs`\n## Next\n`later.rs`\n";
        assert_eq!(layout_names(md), ["a.rs", "dir/"]);
    }

    #[test]
    fn generated_files_are_recognised() {
        assert!(is_generated("x/a.pb.rs", ""));
        assert!(is_generated("a.rs", "// @generated by prost\n"));
        assert!(!is_generated("a.rs", "//! Hand written.\n"));
    }

    #[test]
    fn the_map_lists_each_file_with_its_items() {
        let files = [FileMap {
            path: "a.rs".to_owned(),
            doc: None,
            lines: 3,
            items: vec![
                ("fn f".to_owned(), "Does f.".to_owned()),
                ("struct S".to_owned(), String::new()),
            ],
        }];
        assert_eq!(
            render_map(&files),
            "a.rs (3 lines): (no module doc)\n  fn f - Does f.\n  struct S\n"
        );
    }
}
