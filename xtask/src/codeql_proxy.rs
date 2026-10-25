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

use std::collections::{BTreeMap, BTreeSet};
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

/// Macros whose message arguments CodeQL checks too, with the number of
/// leading arguments that are conditions or operands, not message.
const MESSAGE_SINKS: &[(&str, usize)] = &[
    ("assert!", 1),
    ("assert_eq!", 2),
    ("assert_ne!", 2),
    ("debug_assert!", 1),
    ("debug_assert_eq!", 2),
    ("debug_assert_ne!", 2),
    ("panic!", 0),
    ("unreachable!", 0),
    ("todo!", 0),
    ("unimplemented!", 0),
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

/// Position of the macro `name` (`println!`, ...) in `s`, not as the tail of a longer
/// identifier (`eprint!` is not `print!`, `debug_assert!` is not `assert!`).
fn find_macro(s: &str, name: &str) -> Option<usize> {
    s.match_indices(name)
        .map(|(i, _)| i)
        .find(|&i| !s[..i].chars().next_back().is_some_and(is_ident))
}

/// `src` with comments blanked out (newlines kept, so positions and line numbers hold).
fn blank_comments(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let (mut i, mut depth) = (0, 0usize);
    let mut in_str = false;
    while i < b.len() {
        let c = b[i];
        let next = b.get(i + 1).copied();
        if depth > 0 {
            if c == b'*' && next == Some(b'/') {
                depth -= 1;
                out.push_str("  ");
                i += 2;
            } else if c == b'/' && next == Some(b'*') {
                depth += 1;
                out.push_str("  ");
                i += 2;
            } else {
                out.push(if c == b'\n' { '\n' } else { ' ' });
                i += 1;
            }
        } else if in_str {
            // Copy whole characters: only ASCII bytes are special here.
            let ch = src[i..].chars().next().unwrap_or(' ');
            if c == b'\\' {
                out.push(ch);
                i += 1;
                if let Some(e) = src[i..].chars().next() {
                    out.push(e);
                    i += e.len_utf8();
                }
                continue;
            }
            in_str = c != b'"';
            out.push(ch);
            i += ch.len_utf8();
        } else if c == b'/' && next == Some(b'/') {
            while i < b.len() && b[i] != b'\n' {
                out.push(' ');
                i += 1;
            }
        } else if c == b'/' && next == Some(b'*') {
            depth = 1;
            out.push_str("  ");
            i += 2;
        } else if let Some(n) = char_literal_len(&b[i..]) {
            // A `'"'` must not open a string: keep the quotes, blank the character.
            out.push('\'');
            out.push_str(&" ".repeat(n - 2));
            out.push('\'');
            i += n;
        } else {
            let ch = src[i..].chars().next().unwrap_or(' ');
            in_str = c == b'"';
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// Length in bytes of the ASCII character literal (`'x'`, `'\n'`, `'\''`) at the start of `b`.
/// A lifetime (`'a`) is not one.
fn char_literal_len(b: &[u8]) -> Option<usize> {
    if b.first() != Some(&b'\'') {
        return None;
    }
    match (b.get(1)?, b.get(2)?) {
        (b'\\', _) => (3..8).find(|&n| b.get(n) == Some(&b'\'')).map(|n| n + 1),
        (c, b'\'') if c.is_ascii() => Some(3),
        _ => None,
    }
}

/// The text of the macro call whose opening bracket is at `open`, split at top-level commas.
/// Returns the arguments and the byte offset after the closing bracket.
fn call_args(src: &str, open: usize) -> Option<(Vec<String>, usize)> {
    let b = src.as_bytes();
    let (mut args, mut cur) = (Vec::new(), String::new());
    let (mut depth, mut in_str) = (0usize, false);
    let mut i = open;
    while i < b.len() {
        let ch = src[i..].chars().next()?;
        let c = b[i];
        if in_str {
            cur.push(ch);
            if c == b'\\' {
                if let Some(e) = src[i + 1..].chars().next() {
                    cur.push(e);
                    i += e.len_utf8();
                }
            } else if c == b'"' {
                in_str = false;
            }
        } else {
            match c {
                b'"' => {
                    in_str = true;
                    cur.push(ch);
                }
                b'(' | b'[' | b'{' => {
                    depth += 1;
                    if depth > 1 {
                        cur.push(ch);
                    }
                }
                b')' | b']' | b'}' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        args.push(std::mem::take(&mut cur));
                        return Some((args, i + 1));
                    }
                    cur.push(ch);
                }
                b',' if depth == 1 => args.push(std::mem::take(&mut cur)),
                _ => cur.push(ch),
            }
        }
        i += ch.len_utf8();
    }
    None
}

/// The contents of the string literals in `s` and the text outside them.
fn split_strings(s: &str) -> (String, String) {
    let (mut inside, mut outside) = (String::new(), String::new());
    let (mut in_str, mut esc) = (false, false);
    for c in s.chars() {
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
                inside.push(' ');
                continue;
            }
            inside.push(c);
        } else if c == '"' {
            in_str = true;
        } else {
            outside.push(c);
        }
    }
    (inside, outside)
}

/// Whether any message argument of the call (the arguments after `skip` operands) names a
/// credential: inside a string as `{name}`, or outside strings as a bare identifier.
fn message_is_sensitive(args: &[String], skip: usize) -> bool {
    args.iter().skip(skip).any(|a| {
        let (inside, outside) = split_strings(a);
        interpolates_sensitive(&inside) || has_sensitive_ident(&outside)
    })
}

/// One sink call of a source file: where it starts, its arguments, the macro's operand count to
/// skip before the message, and its 1-based first and last lines.
struct SinkCall {
    args: Vec<String>,
    skip: usize,
    first: usize,
    last: usize,
}

/// Every sink call in `src` (the whole file, so a call may span lines).
fn sink_calls(src: &str) -> Vec<SinkCall> {
    let code = blank_comments(src);
    let line_of = |off: usize| code[..off].bytes().filter(|&b| b == b'\n').count() + 1;
    let mut out = Vec::new();
    let sinks = LOG_SINKS
        .iter()
        .map(|m| (*m, 0))
        .chain(MESSAGE_SINKS.iter().copied());
    for (m, skip) in sinks {
        for (at, _) in code.match_indices(m) {
            if code[..at].chars().next_back().is_some_and(is_ident) {
                continue;
            }
            let rest = &code[at + m.len()..];
            let open = at + m.len() + (rest.len() - rest.trim_start().len());
            if !code[open..].starts_with(['(', '[', '{']) {
                continue;
            }
            let Some((args, end)) = call_args(&code, open) else {
                continue;
            };
            out.push(SinkCall {
                args,
                skip,
                first: line_of(at),
                last: line_of(end.saturating_sub(1)),
            });
        }
    }
    out
}

/// Whether a line of `src` carries the `codeql-allow` marker.
fn has_allow(lines: &[&str], line: usize) -> bool {
    lines
        .get(line - 1)
        .is_some_and(|l| l.contains("codeql-allow"))
}

/// Every line of a sink call that carries `codeql-allow` on any of its lines: the marker exempts
/// the whole call, also from the line-based pass.
#[must_use]
pub fn allowed_call_lines(src: &str) -> BTreeSet<usize> {
    let lines: Vec<&str> = src.lines().collect();
    sink_calls(src)
        .into_iter()
        .filter(|c| (c.first..=c.last).any(|l| has_allow(&lines, l)))
        .flat_map(|c| c.first..=c.last)
        .collect()
}

/// Sink calls in `src` (the whole file, so a call may span lines) that put a credential-named
/// value in their message and touch a line for which `added` is true. The finding is on the
/// line of the macro name. A `codeql-allow` on any line of the call exempts it.
#[must_use]
pub fn scan_calls(file: &str, src: &str, added: &dyn Fn(usize) -> bool) -> Vec<Finding> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    for call in sink_calls(src) {
        let (first, last) = (call.first, call.last);
        if !(first..=last).any(added)
            || (first..=last).any(|l| has_allow(&lines, l))
            || !message_is_sensitive(&call.args, call.skip)
        {
            continue;
        }
        out.push(Finding {
            file: file.to_owned(),
            line: first,
            text: lines
                .get(first - 1)
                .copied()
                .unwrap_or("")
                .trim()
                .to_owned(),
        });
    }
    out
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
    // Only a standalone `format!` is exempt: `println!("{}", format!("{x}"))` is a sink.
    let has_sink = LOG_SINKS
        .iter()
        .chain(MESSAGE_SINKS.iter().map(|(m, _)| m))
        .any(|m| find_macro(t, m).is_some());
    if t.contains('"') && (has_sink || !t.contains("format!(")) && interpolates_sensitive(t) {
        return true;
    }
    // A credential-named identifier passed as a bare argument of a logging macro.
    LOG_SINKS.iter().any(|m| {
        find_macro(t, m).is_some_and(|i| {
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
        return Err(format!(
            "no merge base with {base}: fetch it (`git fetch origin main`) or pass `--base <ref>` \
             with a ref this checkout has"
        ));
    }
    let mb = String::from_utf8_lossy(&mb.stdout).trim().to_owned();
    // Working tree against the merge base: committed and uncommitted changes alike.
    let diff = git_output(root, &["diff", "-U0", "--no-color", &mb, "--", "*.rs"])
        .map_err(|e| format!("git diff failed: {e}"))?;
    // Files git does not track yet are all new: scan them whole.
    let untracked = git_output(
        root,
        &[
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "*.rs",
        ],
    )
    .map_err(|e| format!("git ls-files failed: {e}"))?;
    let mut found = scan(&diff);
    for (file, lines) in added_lines(&diff) {
        if let Ok(src) = std::fs::read_to_string(root.join(&file)) {
            found.extend(scan_calls(&file, &src, &|l| lines.contains(&l)));
        }
    }
    for file in untracked.split('\0').filter(|f| !f.is_empty()) {
        if let Ok(src) = std::fs::read_to_string(root.join(file)) {
            found.extend(scan_source(file, &src));
        }
    }
    // `codeql-allow` anywhere in a call exempts the whole call from the line-based pass too.
    let mut allowed: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    found.retain(|f| {
        !allowed
            .entry(f.file.clone())
            .or_insert_with(|| {
                std::fs::read_to_string(root.join(&f.file))
                    .map(|src| allowed_call_lines(&src))
                    .unwrap_or_default()
            })
            .contains(&f.line)
    });
    found.sort_by(|a, b| (&a.file, a.line).cmp(&(&b.file, b.line)));
    found.dedup_by(|a, b| a.file == b.file && a.line == b.line);
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

fn git_output(root: &Path, args: &[&str]) -> Result<String, String> {
    // `core.quotePath=false`: non-ASCII paths reach the `.rs` filters unquoted.
    let out = Command::new("git")
        .args(["-c", "core.quotePath=false"])
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|e| format!("failed to run git: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_owned());
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Per `*.rs` file of a zero-context diff, the 1-based lines it adds.
fn added_lines(diff: &str) -> BTreeMap<String, BTreeSet<usize>> {
    let mut out: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    let (mut file, mut line) = (String::new(), 0usize);
    for l in diff.lines() {
        if let Some(f) = l.strip_prefix("+++ ") {
            f.strip_prefix("b/").unwrap_or(f).clone_into(&mut file);
        } else if let Some(h) = l.strip_prefix("@@ ") {
            line = h
                .split_whitespace()
                .find(|p| p.starts_with('+'))
                .and_then(|p| p[1..].split(',').next())
                .and_then(|n| n.parse().ok())
                .unwrap_or(0);
        } else if l.starts_with('+') {
            if file.ends_with(".rs") {
                out.entry(file.clone()).or_default().insert(line);
            }
            line += 1;
        }
    }
    out
}

/// Every line of a file that git does not track yet counts as added.
fn scan_source(file: &str, src: &str) -> Vec<Finding> {
    let mut out: Vec<Finding> = src
        .lines()
        .enumerate()
        .filter(|(_, l)| line_is_suspicious(l))
        .map(|(i, l)| Finding {
            file: file.to_owned(),
            line: i + 1,
            text: l.trim().to_owned(),
        })
        .collect();
    out.extend(scan_calls(file, src, &|_| true));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sensitive word, assembled so that this file does not trip its own scan.
    fn word() -> String {
        ["sec", "ret"].concat()
    }

    #[test]
    fn flags_the_pattern_codeql_flagged() {
        let w = word();
        assert!(line_is_suspicious(&format!(
            "\"bundle leaks {{{w}}}: {{}}\","
        )));
        assert!(line_is_suspicious(&format!("println!(\"{{{w}:?}}\")")));
        assert!(line_is_suspicious(&format!("tracing::info!({w}_value);")));
    }

    #[test]
    fn leaves_ordinary_lines_alone() {
        let w = word();
        assert!(!line_is_suspicious(
            r#"assert!(!all_text.contains(needle), "bundle leaks {needle}")"#
        ));
        assert!(!line_is_suspicious(&format!("let {w} = load();")));
        assert!(!line_is_suspicious(&format!("format!(\"{{{w}}}\")")));
        assert!(!line_is_suspicious(&format!(
            "{{{{{w}}}}} escaped braces \"x\""
        )));
        assert!(!line_is_suspicious(&format!("// println!(\"{{{w}}}\")")));
        assert!(!line_is_suspicious(&format!(
            "println!(\"{w} policy\") // codeql-allow"
        )));
        assert!(!line_is_suspicious(r#"println!("password policy")"#));
    }

    #[test]
    fn scan_reports_new_file_line_numbers() {
        let w = word();
        let diff = format!(
            "+++ b/a/x.rs\n@@ -1,0 +10,2 @@\n+let a = 1;\n+println!(\"{{{w}}}\");\n+++ b/README.md\n@@ -1 +1 @@\n+{{{w}}}\n"
        );
        let f = scan(&diff);
        assert_eq!(f.len(), 1);
        assert_eq!((f[0].file.as_str(), f[0].line), ("a/x.rs", 11));
    }

    fn git(dir: &Path, args: &[&str]) {
        // Local overrides: the test must not depend on the developer's signing or hook setup.
        let st = Command::new("git")
            .args([
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
            ])
            .args(args)
            .current_dir(dir)
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .env("GIT_AUTHOR_NAME", "t")
            .env("GIT_AUTHOR_EMAIL", "t@example.com")
            .env("GIT_COMMITTER_NAME", "t")
            .env("GIT_COMMITTER_EMAIL", "t@example.com")
            .status()
            .unwrap();
        assert!(st.success(), "git {args:?}");
    }

    #[test]
    fn run_fails_on_a_flagged_line_and_passes_without_one() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        git(d, &["init", "-q", "-b", "main"]);
        std::fs::write(d.join("a.rs"), "fn a() {}\n").unwrap();
        git(d, &["add", "."]);
        git(d, &["commit", "-q", "-m", "base"]);
        assert!(run(d, &["--base", "main"]).is_ok());
        let w = word();
        std::fs::write(
            d.join("a.rs"),
            format!("fn a() {{ println!(\"{{{w}}}\"); }}\n"),
        )
        .unwrap();
        let err = run(d, &["--base", "main"]).unwrap_err();
        assert!(err.contains("1 line(s)"), "{err}");
        assert!(run(d, &["--bogus"]).is_err());
        assert!(run(d, &["--base"]).is_err());
        assert!(run(d, &["--base", "nope"]).is_err());
    }
    fn calls(src: &str) -> Vec<usize> {
        scan_calls("x.rs", src, &|_| true)
            .iter()
            .map(|f| f.line)
            .collect()
    }

    #[test]
    fn a_nested_format_inside_a_sink_is_still_a_sink() {
        let w = word();
        assert!(line_is_suspicious(&format!(
            "println!(\"{{}}\", format!(\"{{{w}}}\"))"
        )));
        assert_eq!(
            calls(&format!("println!(\"{{}}\", format!(\"{{{w}}}\"));")),
            [1]
        );
    }

    #[test]
    fn assert_and_panic_messages_are_checked_but_not_the_condition() {
        let w = word();
        assert_eq!(calls(&format!("assert!(ok, \"v: {{}}\", {w});")), [1]);
        assert_eq!(calls(&format!("panic!(\"v: {{}}\", {w});")), [1]);
        assert_eq!(calls(&format!("assert_eq!(a, b, \"v: {{}}\", {w});")), [1]);
        assert_eq!(calls(&format!("debug_assert!(ok, \"v: {{{w}}}\");")), [1]);
        // The condition and the operands are not messages.
        assert!(calls(&format!("assert!({w}.is_empty());")).is_empty());
        assert!(calls(&format!("assert_eq!({w}, other);")).is_empty());
        assert!(calls(&format!("assert_eq!({w}, other, \"ok\");")).is_empty());
        // Not a sink: a longer macro name, a comment, a plain `format!`.
        assert!(calls(&format!("my_panic!(\"v: {{}}\", {w});")).is_empty());
        assert!(calls(&format!("// println!(\"{{{w}}}\");")).is_empty());
        assert!(calls(&format!("let m = format!(\"{{{w}}}\");")).is_empty());
    }

    #[test]
    fn a_call_spread_over_lines_is_judged_whole() {
        let w = word();
        let src = format!(
            "fn f() {{\n    println!(\n        \"value: {{}}\",\n        {w}\n    );\n}}\n"
        );
        let f = scan_calls("x.rs", &src, &|l| l == 4);
        assert_eq!(f.len(), 1);
        assert_eq!((f[0].line, f[0].text.as_str()), (2, "println!("));
        // Only when the call has an added line.
        assert!(scan_calls("x.rs", &src, &|l| l == 1).is_empty());
        // `codeql-allow` anywhere in the call exempts it.
        let allowed = src.replace(&format!("{w}\n"), &format!("{w} // codeql-allow\n"));
        assert!(scan_calls("x.rs", &allowed, &|_| true).is_empty());
        // Parentheses, braces, char literals and comments inside the call.
        let src = format!("log!(a);\nwarn!(\"(\", '\"', /* ) */ f({w}));\n");
        assert_eq!(calls(&src), [2]);
    }

    #[test]
    fn run_scans_untracked_files_and_names_a_remedy_for_a_missing_base() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        git(d, &["init", "-q", "-b", "main"]);
        std::fs::write(d.join("a.rs"), "fn a() {}\n").unwrap();
        git(d, &["add", "."]);
        git(d, &["commit", "-q", "-m", "base"]);
        let w = word();
        std::fs::write(
            d.join("new.rs"),
            format!("fn b() {{\n    println!(\n        \"v: {{}}\",\n        {w}\n    );\n}}\n"),
        )
        .unwrap();
        let err = run(d, &["--base", "main"]).unwrap_err();
        assert!(err.contains("1 line(s)"), "{err}");
        // A tracked, committed file is scanned as a whole call as well.
        git(d, &["add", "."]);
        git(d, &["commit", "-q", "-m", "more"]);
        git(d, &["branch", "base", "HEAD~1"]);
        let err = run(d, &["--base", "base"]).unwrap_err();
        assert!(err.contains("1 line(s)"), "{err}");
        let err = run(d, &["--base", "nope"]).unwrap_err();
        assert!(err.contains("git fetch"), "{err}");
    }

    #[test]
    fn one_allow_marker_exempts_a_multi_line_call_in_every_pass() {
        let w = word();
        let src = format!(
            "fn f() {{\n    println!( // codeql-allow\n        \"value: {{{w}}}\",\n    );\n}}\n"
        );
        assert_eq!(allowed_call_lines(&src), BTreeSet::from([2, 3, 4]));
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        git(d, &["init", "-q", "-b", "main"]);
        std::fs::write(d.join("a.rs"), "fn a() {}\n").unwrap();
        git(d, &["add", "."]);
        git(d, &["commit", "-q", "-m", "base"]);
        // Untracked (whole-file scan) and tracked-and-modified (diff scan).
        std::fs::write(d.join("new.rs"), &src).unwrap();
        assert!(run(d, &["--base", "main"]).is_ok());
        git(d, &["add", "."]);
        git(d, &["commit", "-q", "-m", "more"]);
        git(d, &["branch", "base", "HEAD~1"]);
        assert!(run(d, &["--base", "base"]).is_ok());
        // Without the marker the same call is flagged.
        std::fs::write(d.join("new.rs"), src.replace(" // codeql-allow", "")).unwrap();
        assert!(run(d, &["--base", "base"]).is_err());
    }

    #[test]
    fn non_ascii_paths_are_scanned() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        git(d, &["init", "-q", "-b", "main"]);
        std::fs::write(d.join("a.rs"), "fn a() {}\n").unwrap();
        git(d, &["add", "."]);
        git(d, &["commit", "-q", "-m", "base"]);
        let w = word();
        std::fs::write(
            d.join("na\u{ef}ve.rs"),
            format!("fn a() {{ println!(\"{{{w}}}\"); }}\n"),
        )
        .unwrap();
        git(d, &["add", "."]);
        git(d, &["commit", "-q", "-m", "tracked"]);
        git(d, &["branch", "base", "HEAD~1"]);
        let err = run(d, &["--base", "base"]).unwrap_err();
        assert!(err.contains("1 line(s)"), "{err}");
    }
}
