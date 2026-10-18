// SPDX-License-Identifier: Apache-2.0
//! Repository lints (P2-T145, P2-T111, P2-T112): workflow policy, SPDX
//! headers and the banned `parquet` object_store adapters.

/// The SPDX tag every source file carries (ADR 0033).
pub const SPDX_TAG: &str = "SPDX-License-Identifier: Apache-2.0";

/// Identifiers that must not appear in the source (ADR 0028: the parquet
/// crate's `object_store` feature is banned, so these adapters cannot exist).
pub const BANNED_IDENTS: &[&str] = &["ParquetObjectReader", "ParquetObjectWriter"];

fn code_lines(text: &str) -> impl Iterator<Item = &str> {
    text.lines().filter(|l| !l.trim_start().starts_with('#'))
}

/// Triggers that run with the base repository's secrets and write token on
/// events an outsider controls, so a workflow must never use them here.
const BANNED_TRIGGERS: &[&str] = &["pull_request_target", "workflow_run", "issue_comment"];

/// Whether `label` names a GitHub-hosted runner image.
fn is_github_hosted(label: &str) -> bool {
    ["ubuntu-", "macos-", "windows-"]
        .iter()
        .any(|p| label.starts_with(p))
}

/// Whether `rest` (what follows `uses:`) is pinned: a local action, a Docker
/// image by digest, or `owner/repo[/path]@<40-hex commit SHA>`.
fn is_pinned(rest: &str) -> bool {
    let target = rest
        .split('#')
        .next()
        .unwrap_or("")
        .trim()
        .trim_matches(['"', '\'']);
    if target.starts_with("./") {
        return true;
    }
    if let Some(image) = target.strip_prefix("docker://") {
        return image.contains("@sha256:");
    }
    target
        .rsplit_once('@')
        .is_some_and(|(_, r)| r.len() == 40 && r.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// The runner labels of the `runs-on:` key on line `i`: inline (`x`, `[x, y]`)
/// or as a following `- x` list.
fn runs_on_labels(lines: &[&str], i: usize, rest: &str) -> Vec<String> {
    let clean = |s: &str| s.trim().trim_matches(['"', '\'']).to_owned();
    let rest = rest.split('#').next().unwrap_or("").trim();
    if !rest.is_empty() {
        return rest
            .trim_start_matches('[')
            .trim_end_matches(']')
            .split(',')
            .map(clean)
            .filter(|l| !l.is_empty())
            .collect();
    }
    let indent = lines[i].len() - lines[i].trim_start().len();
    lines[i + 1..]
        .iter()
        .take_while(|l| l.len() - l.trim_start().len() > indent && l.trim_start().starts_with("- "))
        .map(|l| clean(&l.trim_start()[2..]))
        .collect()
}

/// Policy violations in one workflow file's text.
///
/// - Triggers that run privileged on outsider-controlled events
///   (`pull_request_target`, `workflow_run`, `issue_comment`) are banned.
/// - Only GitHub-hosted runners are allowed (public repository policy):
///   every `runs-on` label must be a literal `ubuntu-*`, `macos-*` or
///   `windows-*` image. Self-hosted or custom labels and expressions fail.
/// - Every `uses:` is pinned to a full commit SHA (or a local action, or a
///   Docker image by digest).
#[must_use]
pub fn workflow_violations(name: &str, text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    for trigger in BANNED_TRIGGERS {
        let used = code_lines(text).any(|l| {
            l.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .any(|w| w == *trigger)
        });
        if used {
            out.push(format!("{name}: the `{trigger}` trigger is banned"));
        }
    }
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        if t.starts_with('#') {
            continue;
        }
        if let Some(rest) = t.strip_prefix("runs-on:") {
            let labels = runs_on_labels(&lines, i, rest);
            if labels.is_empty() {
                out.push(format!(
                    "{name}:{}: `runs-on` has no label this lint can read; use a literal GitHub-hosted label",
                    i + 1
                ));
            }
            let bad: Vec<String> = labels
                .into_iter()
                .filter(|l| !is_github_hosted(l))
                .collect();
            if !bad.is_empty() {
                out.push(format!(
                    "{name}:{}: runner labels {bad:?} are not GitHub-hosted; self-hosted, custom and expression labels are banned",
                    i + 1
                ));
            }
        }
        let uses = t
            .strip_prefix("- uses:")
            .or_else(|| t.strip_prefix("uses:"));
        if let Some(rest) = uses
            && !is_pinned(rest)
        {
            out.push(format!(
                "{name}:{}: `uses: {}` is not pinned to a full commit SHA",
                i + 1,
                rest.trim()
            ));
        }
    }
    out
}

/// Whether a comment line in the first lines of `text` carries the SPDX tag.
/// A string literal that merely contains the tag does not count.
#[must_use]
pub fn has_spdx_header(text: &str) -> bool {
    text.lines().take(5).any(|l| {
        let t = l.trim_start();
        (t.starts_with("//") || t.starts_with('#')) && t.contains(SPDX_TAG)
    })
}

/// An identifier allowed only in the listed files (paths from the repository
/// root): `RedactedSql` is built only by the token redactor (ADR 0038).
pub const CONFINED_IDENTS: &[(&str, &[&str])] = &[(
    "from_marks_unchecked",
    &[
        "crates/krishiv-common/src/redact.rs",
        "crates/krishiv-sql/src/redact.rs",
    ],
)];

/// Confined identifiers used outside their files, as `(line, identifier)`.
#[must_use]
pub fn confined_idents_in(rel_path: &str, text: &str) -> Vec<(usize, &'static str)> {
    let mut out = Vec::new();
    for (ident, allowed) in CONFINED_IDENTS {
        if allowed.contains(&rel_path) {
            continue;
        }
        for (n, line) in text.lines().enumerate() {
            if line.contains(ident) {
                out.push((n + 1, *ident));
            }
        }
    }
    out
}

/// Banned identifiers found in `text` as `(line number, identifier)`.
#[must_use]
pub fn banned_idents_in(text: &str) -> Vec<(usize, &'static str)> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        for ident in BANNED_IDENTS {
            if line.contains(ident) {
                out.push((n + 1, *ident));
            }
        }
    }
    out
}

/// Lines (1-based) holding an `unsafe` keyword in code: the whole word
/// `unsafe` outside comments (`//`, nested `/* */`), string literals (with
/// escapes and line continuations, `b`/`c` strings and raw `r`/`br`/`cr`
/// strings with any number of `#`, across lines) and char
/// literals. `unsafe_code`, `unsafe_op_in_unsafe_fn` and
/// `undocumented_unsafe_blocks` do not count. One site per line
/// (`unsafe { .. }`, `unsafe fn`, `unsafe impl`).
///
/// A small lexer for the budget and the source guard, not a full Rust
/// lexer; where it is unsure (an unterminated literal) it keeps scanning
/// as code, so it can over-count but does not hide an `unsafe` in code.
#[must_use]
pub fn unsafe_lines(text: &str) -> Vec<usize> {
    let chars: Vec<char> = text.chars().collect();
    let at = |i: usize| chars.get(i).copied().unwrap_or('\0');
    let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut out = Vec::new();
    let mut line = 1;
    let mut word = String::new();
    let flush = |word: &mut String, line: usize, out: &mut Vec<usize>| {
        if word == "unsafe" && out.last() != Some(&line) {
            out.push(line);
        }
        word.clear();
    };
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if is_ident(c) {
            // Raw string prefixes: `r`, `br`, `cr`, each with any number of
            // `#`. (`b"`/`c"` are normal strings, handled at the quote.)
            let prefix = match (c, at(i + 1)) {
                ('r', _) => 1,
                ('b' | 'c', 'r') => 2,
                _ => 0,
            };
            if prefix > 0 && word.is_empty() {
                let mut j = i + prefix;
                let mut hashes = 0;
                while at(j) == '#' {
                    hashes += 1;
                    j += 1;
                }
                if at(j) == '"' {
                    // Find `"` followed by `hashes` `#`.
                    let mut k = j + 1;
                    let end = loop {
                        if k >= chars.len() {
                            break None;
                        }
                        if chars[k] == '"' && (1..=hashes).all(|h| at(k + h) == '#') {
                            break Some(k + hashes);
                        }
                        k += 1;
                    };
                    if let Some(end) = end {
                        line += chars[i..=end].iter().filter(|&&ch| ch == '\n').count();
                        i = end + 1;
                        continue;
                    }
                }
            }
            word.push(c);
            i += 1;
            continue;
        }
        flush(&mut word, line, &mut out);
        match c {
            '\n' => {
                line += 1;
                i += 1;
            }
            '/' if at(i + 1) == '/' => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '/' if at(i + 1) == '*' => {
                let mut depth = 1;
                let mut k = i + 2;
                while k < chars.len() && depth > 0 {
                    match (chars[k], at(k + 1)) {
                        ('/', '*') => {
                            depth += 1;
                            k += 1;
                        }
                        ('*', '/') => {
                            depth -= 1;
                            k += 1;
                        }
                        ('\n', _) => line += 1,
                        _ => {}
                    }
                    k += 1;
                }
                i = k;
            }
            '"' => {
                // A normal (or byte) string: `\\` escapes the next char.
                let mut k = i + 1;
                let mut lines = 0;
                while k < chars.len() && chars[k] != '"' {
                    if chars[k] == '\\' {
                        // The escaped char may be a newline (a line continuation).
                        k += 1;
                    }
                    if at(k) == '\n' {
                        lines += 1;
                    }
                    k += 1;
                }
                if k < chars.len() {
                    line += lines;
                    i = k + 1;
                } else {
                    // Unterminated: scan the rest as code.
                    i += 1;
                }
            }
            '\'' => {
                // A char literal (`'x'`, `'\n'`, `'\u{..}'`) or a lifetime.
                let end = if at(i + 1) == '\\' {
                    (i + 3..i + 12).find(|&k| at(k) == '\'')
                } else if at(i + 2) == '\'' {
                    Some(i + 2)
                } else {
                    None
                };
                i = end.map_or(i + 1, |e| e + 1);
            }
            _ => i += 1,
        }
    }
    flush(&mut word, line, &mut out);
    out
}

/// Crate directories (under `crates/`) whose `unsafe` must live in files
/// named `unsafe_*.rs` (the kernel unsafe policy, ADR 0029): every unsafe
/// site sits in a small, separately reviewed helper module.
pub const UNSAFE_FILE_CRATES: &[&str] = &["krishiv-kernels"];

/// Whether `rel_path` (relative to the repository root) may contain
/// `unsafe`: outside [`UNSAFE_FILE_CRATES`] the crate's own
/// `forbid(unsafe_code)` decides; inside, only `unsafe_*.rs` files may.
#[must_use]
pub fn unsafe_allowed_in(rel_path: &str) -> bool {
    let p = rel_path.replace('\\', "/");
    let governed = UNSAFE_FILE_CRATES
        .iter()
        .any(|c| p.starts_with(&format!("crates/{c}/")));
    !governed
        || p.rsplit('/')
            .next()
            .is_some_and(|f| f.starts_with("unsafe_"))
}

/// Lines with an `unsafe` keyword per crate under `root/crates/*/src`
/// (crates with none are omitted), for `xtask budgets`. Symlinks are not
/// followed (no loops or double counting).
///
/// # Errors
/// Any unreadable directory or file: a skipped file would hide its sites
/// and make the gate fail open.
pub fn unsafe_sites_per_crate(
    root: &std::path::Path,
) -> Result<std::collections::BTreeMap<String, u64>, String> {
    fn walk(dir: &std::path::Path, n: &mut u64) -> Result<(), String> {
        let entries =
            std::fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
        for e in entries {
            let e = e.map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
            let p = e.path();
            let kind = e
                .file_type()
                .map_err(|e| format!("cannot stat {}: {e}", p.display()))?;
            if kind.is_dir() {
                walk(&p, n)?;
            } else if kind.is_file() && p.extension().is_some_and(|x| x == "rs") {
                let text = std::fs::read_to_string(&p)
                    .map_err(|e| format!("cannot read {}: {e}", p.display()))?;
                let lines = u64::try_from(unsafe_lines(&text).len()).unwrap_or(u64::MAX);
                *n = n.saturating_add(lines);
            }
        }
        Ok(())
    }
    let crates = root.join("crates");
    let mut out = std::collections::BTreeMap::new();
    let entries =
        std::fs::read_dir(&crates).map_err(|e| format!("cannot read {}: {e}", crates.display()))?;
    for c in entries {
        let c = c.map_err(|e| format!("cannot read {}: {e}", crates.display()))?;
        let src = c.path().join("src");
        if !src.is_dir() {
            continue;
        }
        let mut n = 0;
        walk(&src, &mut n)?;
        if n > 0 {
            out.insert(c.file_name().to_string_lossy().into_owned(), n);
        }
    }
    Ok(out)
}
