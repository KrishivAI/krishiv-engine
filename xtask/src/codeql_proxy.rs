// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask codeql-proxy`: a cheap local stand-in for the one CodeQL query
//! that has failed PRs, `rust/cleartext-logging`.
//!
//! CodeQL itself only runs in CI. That query treats a variable whose *name*
//! looks like a credential (`secret`, `password`, ...) as sensitive and flags
//! any log, print, panic or assertion message it flows into, even in tests
//! that merely check a bundle does not leak the value. The proxy scans the
//! lines the branch adds for such a name inside a format string (`{secret}`)
//! or as a bare macro argument, and fails with the fix: rename the variable
//! (`needle`) or leave the value out of the message.
//!
//! It over-approximates on purpose (a flagged line is cheap to rename); CodeQL
//! stays the authority.
#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "CLI command: the findings are its interface"
)]

use std::path::Path;
use std::process::Command;

/// Name fragments CodeQL's sensitive-data heuristics treat as credentials.
const SENSITIVE: &[&str] = &[
    "secret",
    "password",
    "passwd",
    "apikey",
    "api_key",
    "credential",
    "privatekey",
    "private_key",
    "accesskey",
    "access_key",
    "bearer",
];

/// Macros that write to a log or the terminal.
const LOG_SINKS: &[&str] = &[
    "println!",
    "eprintln!",
    "print!",
    "eprint!",
    "trace!",
    "debug!",
    "info!",
    "warn!",
    "error!",
];

/// One suspicious added line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// File path.
    pub file: String,
    /// 1-based line in the new file.
    pub line: usize,
    /// The line, trimmed.
    pub text: String,
}

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Whether `s` contains an identifier that contains a sensitive fragment.
fn has_sensitive_ident(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    lower
        .split(|c: char| !is_ident(c))
        .any(|w| SENSITIVE.iter().any(|f| w.contains(f)))
}

/// `{name}` / `{name:?}` interpolations whose name is sensitive.
fn interpolates_sensitive(line: &str) -> bool {
    let mut rest = line;
    while let Some(i) = rest.find('{') {
        rest = &rest[i + 1..];
        if rest.starts_with('{') {
            rest = &rest[1..];
            continue;
        }
        let name: String = rest.chars().take_while(|c| is_ident(*c)).collect();
        if !name.is_empty() && has_sensitive_ident(&name) {
            return true;
        }
    }
    false
}

/// Whether one added line looks like the flagged pattern.
#[must_use]
pub fn line_is_suspicious(line: &str) -> bool {
    let t = line.trim();
    if t.starts_with("//") || t.starts_with("///") || t.contains("codeql-allow") {
        return false;
    }
    // A message string naming a credential-like variable (the #229 alert: an assert message
    // spread over several lines, so the sink is not on this line). A `format!` that only
    // builds a value is not a sink.
    if t.contains('"') && !t.contains("format!(") && interpolates_sensitive(t) {
        return true;
    }
    // A credential-named identifier passed as a bare argument of a logging macro.
    LOG_SINKS.iter().any(|m| {
        t.find(m).is_some_and(|i| {
            let mut outside = String::new();
            let mut in_str = false;
            for c in t[i + m.len()..].chars() {
                if c == '"' {
                    in_str = !in_str;
                } else if !in_str {
                    outside.push(c);
                }
            }
            has_sensitive_ident(&outside)
        })
    })
}

/// Added lines of a zero-context unified diff of `*.rs` files that look suspicious.
#[must_use]
pub fn scan(diff: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    let (mut file, mut line) = (String::new(), 0usize);
    for l in diff.lines() {
        if let Some(f) = l.strip_prefix("+++ ") {
            f.strip_prefix("b/").unwrap_or(f).clone_into(&mut file);
        } else if let Some(h) = l.strip_prefix("@@ ") {
            // "@@ -a,b +c,d @@"
            line = h
                .split_whitespace()
                .find(|p| p.starts_with('+'))
                .and_then(|p| p[1..].split(',').next())
                .and_then(|n| n.parse().ok())
                .unwrap_or(0);
        } else if let Some(added) = l.strip_prefix('+') {
            if file.ends_with(".rs") && line_is_suspicious(added) {
                out.push(Finding {
                    file: file.clone(),
                    line,
                    text: added.trim().to_owned(),
                });
            }
            line += 1;
        }
    }
    out
}

/// `xtask codeql-proxy [--base REF]`.
///
/// # Errors
/// Returns the git failure, or the findings.
pub fn run(root: &Path, flags: &[&str]) -> Result<(), String> {
    let mut base = "origin/main";
    let mut it = flags.iter();
    while let Some(f) = it.next() {
        match *f {
            "--base" => base = it.next().ok_or("--base needs a ref")?,
            other => return Err(format!("unknown codeql-proxy flag {other}")),
        }
    }
    let mb = Command::new("git")
        .args(["merge-base", base, "HEAD"])
        .current_dir(root)
        .output()
        .map_err(|e| format!("failed to run git: {e}"))?;
    if !mb.status.success() {
        return Err(format!("no merge base with {base}"));
    }
    let mb = String::from_utf8_lossy(&mb.stdout).trim().to_owned();
    // Working tree against the merge base: committed and uncommitted changes alike.
    let diff = Command::new("git")
        .args(["diff", "-U0", "--no-color", &mb, "--", "*.rs"])
        .current_dir(root)
        .output()
        .map_err(|e| format!("failed to run git: {e}"))?;
    if !diff.status.success() {
        return Err("git diff failed".to_owned());
    }
    let found = scan(&String::from_utf8_lossy(&diff.stdout));
    if found.is_empty() {
        println!("codeql-proxy: OK (CodeQL itself runs only in CI)");
        return Ok(());
    }
    for f in &found {
        eprintln!("{}:{}: {}", f.file, f.line, f.text);
    }
    Err(format!(
        "codeql-proxy: {} line(s) put a credential-named variable in a log, print, panic or assert message \
         (CodeQL rust/cleartext-logging); rename the variable (e.g. `needle`) or leave the value out. \
         Add `codeql-allow` to a line only after CodeQL has accepted it",
        found.len()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_the_pattern_codeql_flagged() {
        assert!(line_is_suspicious(
            r#"            "bundle leaks {secret}: {}","#
        ));
        assert!(line_is_suspicious(r#"println!("{password:?}")"#));
        assert!(line_is_suspicious("tracing::info!(secret_value);"));
    }

    #[test]
    fn leaves_ordinary_lines_alone() {
        assert!(!line_is_suspicious(
            r#"assert!(!all_text.contains(needle), "bundle leaks {needle}")"#
        ));
        assert!(!line_is_suspicious("let secret = load();"));
        assert!(!line_is_suspicious(
            r#"format!("{{secret}} is a template")"#
        ));
        assert!(!line_is_suspicious("// println!(\"{secret}\")"));
        assert!(!line_is_suspicious(r#"println!("password policy")"#));
    }

    #[test]
    fn scan_reports_new_file_line_numbers() {
        let diff = "+++ b/a/x.rs\n@@ -1,0 +10,2 @@\n+let a = 1;\n+println!(\"{secret}\");\n+++ b/README.md\n@@ -1 +1 @@\n+{secret}\n";
        let f = scan(diff);
        assert_eq!(f.len(), 1);
        assert_eq!((f[0].file.as_str(), f[0].line), ("a/x.rs", 11));
    }
}
