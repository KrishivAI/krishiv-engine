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

/// Triggers a workflow with self-hosted jobs may use. Anything that can run
/// untrusted code (pull_request*, workflow_run, issue_comment, ...) is refused.
const SELF_HOSTED_TRIGGERS: &[&str] = &["schedule", "workflow_dispatch", "push"];

/// Trigger names of a workflow, with the text of each trigger's block.
fn triggers(text: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = code_lines(text).collect();
    let mut out: Vec<(String, String)> = Vec::new();
    let Some(start) = lines
        .iter()
        .position(|l| l.starts_with("on:") || l.starts_with("\"on\":") || l.starts_with("true:"))
    else {
        return out;
    };
    let rest = lines[start].split_once(':').map_or("", |(_, r)| r).trim();
    if !rest.is_empty() {
        for name in rest.split(|c: char| c == '[' || c == ']' || c == ',' || c.is_whitespace()) {
            if !name.is_empty() {
                out.push((name.to_owned(), String::new()));
            }
        }
        return out;
    }
    let mut child_indent = None;
    for line in &lines[start + 1..] {
        if line.trim().is_empty() {
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        if indent == 0 {
            break;
        }
        let ci = *child_indent.get_or_insert(indent);
        if indent == ci {
            let name = line.trim().split(':').next().unwrap_or("").trim();
            out.push((name.to_owned(), String::new()));
        } else if let Some(last) = out.last_mut() {
            last.1.push_str(line);
            last.1.push('\n');
        }
    }
    out
}

/// Policy violations in one workflow file's text.
///
/// - `pull_request_target` is banned everywhere (it runs with secrets on
///   untrusted code).
/// - A workflow with a job on a self-hosted runner (`self-hosted` or the
///   `krishiv-heavy` label) may only trigger on `schedule`, `workflow_dispatch`
///   or `push` to `main` (ops/ci-runner.md: untrusted code never reaches the
///   runner).
#[must_use]
pub fn workflow_violations(name: &str, text: &str) -> Vec<String> {
    let mut out = Vec::new();
    if code_lines(text).any(|l| l.contains("pull_request_target")) {
        out.push(format!("{name}: `pull_request_target` is banned"));
    }
    let self_hosted = code_lines(text).any(|l| {
        l.contains("runs-on") && (l.contains("self-hosted") || l.contains("krishiv-heavy"))
    }) || code_lines(text).any(|l| {
        let t = l.trim();
        t.starts_with("- self-hosted") || t.starts_with("- krishiv-heavy")
    });
    if self_hosted {
        for (trigger, block) in triggers(text) {
            let ok = SELF_HOSTED_TRIGGERS.contains(&trigger.as_str())
                && (trigger != "push" || block.contains("[main]") || block.contains("- main"));
            if !ok {
                out.push(format!(
                    "{name}: self-hosted jobs must not run on trigger `{trigger}` (allowed: schedule, workflow_dispatch, push to main)"
                ));
            }
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
