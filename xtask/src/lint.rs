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

/// Policy violations in one workflow file's text.
///
/// - `pull_request_target` is banned everywhere (it runs with secrets on
///   untrusted code).
/// - Self-hosted runners are banned (public repository policy): any
///   `self-hosted` or `krishiv-heavy` runner label fails the lint.
#[must_use]
pub fn workflow_violations(name: &str, text: &str) -> Vec<String> {
    let mut out = Vec::new();
    if code_lines(text).any(|l| l.contains("pull_request_target")) {
        out.push(format!("{name}: `pull_request_target` is banned"));
    }
    let self_hosted = code_lines(text).any(|l| {
        let t = l.trim();
        (l.contains("runs-on") || t.starts_with("- "))
            && (l.contains("self-hosted") || l.contains("krishiv-heavy"))
    });
    if self_hosted {
        out.push(format!(
            "{name}: self-hosted runners (`self-hosted`, `krishiv-heavy`) are banned; use GitHub-hosted runners"
        ));
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
