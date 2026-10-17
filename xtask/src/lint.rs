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

/// Policy violations in one workflow file's text.
///
/// - `pull_request_target` is banned everywhere (it runs with secrets on
///   untrusted code).
/// - A workflow with a self-hosted job must not trigger on `pull_request`
///   (ops/ci-runner.md: untrusted code never reaches the runner).
#[must_use]
pub fn workflow_violations(name: &str, text: &str) -> Vec<String> {
    let mut out = Vec::new();
    if code_lines(text).any(|l| l.contains("pull_request_target")) {
        out.push(format!("{name}: `pull_request_target` is banned"));
    }
    let triggers = text.split("\njobs:").next().unwrap_or(text);
    let pr_trigger = code_lines(triggers).any(|l| {
        l.match_indices("pull_request").any(|(i, m)| {
            let after = l[i + m.len()..].chars().next();
            !after.is_some_and(|c| c == '_' || c.is_alphanumeric())
        })
    });
    if pr_trigger && code_lines(text).any(|l| l.contains("self-hosted")) {
        out.push(format!(
            "{name}: self-hosted jobs must not run in a workflow triggered by pull_request"
        ));
    }
    out
}

/// Whether the first lines of `text` carry the SPDX tag.
#[must_use]
pub fn has_spdx_header(text: &str) -> bool {
    text.lines().take(5).any(|l| l.contains(SPDX_TAG))
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
