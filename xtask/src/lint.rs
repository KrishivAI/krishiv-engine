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
            let bad: Vec<String> = runs_on_labels(&lines, i, rest)
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
