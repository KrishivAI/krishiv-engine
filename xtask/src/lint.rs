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
/// `unsafe` outside `//` comments, string literals and `'"'` char literals
/// (`unsafe_code`, `unsafe_op_in_unsafe_fn` and `undocumented_unsafe_blocks`
/// do not count). One site per line (`unsafe { .. }`, `unsafe fn`,
/// `unsafe impl`).
///
/// A line-based heuristic for the budget and the source guard, not a Rust
/// lexer: `/* */` block comments and raw strings are read as code, so an
/// `unsafe` inside them is over-counted (never hidden).
#[must_use]
pub fn unsafe_lines(text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let mut code = String::new();
        let mut in_str = false;
        let mut prev = '\0';
        let mut chars = line.chars().peekable();
        while let Some(c) = chars.next() {
            if !in_str && c == '/' && chars.peek() == Some(&'/') {
                break;
            }
            // A `'"'` char literal does not open a string.
            if !in_str && c == '\'' && chars.peek() == Some(&'"') {
                let mut ahead = chars.clone();
                ahead.next();
                if ahead.next() == Some('\'') {
                    chars = ahead;
                    code.push_str("' '");
                    prev = '\'';
                    continue;
                }
            }
            if c == '"' && prev != '\\' {
                in_str = !in_str;
            }
            code.push(if in_str { ' ' } else { c });
            prev = c;
        }
        if code
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .any(|w| w == "unsafe")
        {
            out.push(n + 1);
        }
    }
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
/// (crates with none are omitted), for `xtask budgets`.
#[must_use]
pub fn unsafe_sites_per_crate(root: &std::path::Path) -> std::collections::BTreeMap<String, u64> {
    fn walk(dir: &std::path::Path, n: &mut u64) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, n);
            } else if p.extension().is_some_and(|x| x == "rs")
                && let Ok(text) = std::fs::read_to_string(&p)
            {
                let lines = u64::try_from(unsafe_lines(&text).len()).unwrap_or(u64::MAX);
                *n = n.saturating_add(lines);
            }
        }
    }
    let mut out = std::collections::BTreeMap::new();
    for c in std::fs::read_dir(root.join("crates"))
        .into_iter()
        .flatten()
        .flatten()
    {
        let mut n = 0;
        walk(&c.path().join("src"), &mut n);
        if n > 0 {
            out.insert(c.file_name().to_string_lossy().into_owned(), n);
        }
    }
    out
}
