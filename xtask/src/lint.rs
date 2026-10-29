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

/// Whether `rest` (what follows `uses:`) is pinned: a local action (`./` or `$/`), a Docker
/// image by digest, or `owner/repo[/path]@<40-hex commit SHA>`.
fn is_pinned(rest: &str) -> bool {
    let target = rest
        .split('#')
        .next()
        .unwrap_or("")
        .trim()
        .trim_matches(['"', '\'']);
    // `./path` and GitHub's self-repository form `$/path` both name a file in
    // this repository at the workflow's own commit.
    if target.starts_with("./") || target.starts_with("$/") {
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
    let rel_path = rel_path.replace('\\', "/");
    let mut out = Vec::new();
    for (ident, allowed) in CONFINED_IDENTS {
        if allowed.contains(&rel_path.as_str()) {
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

/// Live-handle markers a physical-plan file must not use in code (design
/// P2-T171 invariant I1; the `PlanData` marker is the compile-time check,
/// this is the backstop).
const PLAN_HANDLE_MARKERS: &[&str] = &[
    "Arc<dyn", "Box<dyn", "Mutex", "RwLock", "Atomic", "OnceLock", "Cell<", "dyn Fn",
];

/// The physical-plan sources the backstop covers: the plan data types,
/// not the instantiation that builds operators from them.
fn is_plan_data_source(rel_path: &str) -> bool {
    rel_path.starts_with("crates/krishiv-exec/src/physical/")
        && rel_path.ends_with(".rs")
        && !rel_path.ends_with("/instantiate.rs")
}

/// Live-handle markers in code lines (not `//` comments) of a physical-plan
/// source, as `(line, marker)`.
#[must_use]
pub fn plan_handles_in(rel_path: &str, text: &str) -> Vec<(usize, &'static str)> {
    let rel_path = rel_path.replace('\\', "/");
    if !is_plan_data_source(&rel_path) {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let code = line.split("//").next().unwrap_or_default();
        for marker in PLAN_HANDLE_MARKERS {
            if code.contains(marker) {
                out.push((n + 1, *marker));
            }
        }
    }
    out
}

/// Whether `code` (a line without its comment) ends a match arm with the
/// `Admin` statement class: `=> StatementClass::Admin`, or `=> Admin` in a
/// file that names `StatementClass`.
fn is_admin_arm(code: &str, names_class: bool) -> bool {
    let ends_word = |rest: &str| {
        !rest
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
    };
    let compact: String = code.split_whitespace().collect::<Vec<_>>().join(" ");
    let found = |needle: &str| {
        compact
            .match_indices(needle)
            .any(|(i, _)| ends_word(&compact[i + needle.len()..]))
    };
    found("=> StatementClass::Admin") || names_class && found("=> Admin")
}

/// Match arms that classify a statement as `Admin` without a
/// `// cluster-scope: <reason>` comment on the arm or in the comments and
/// pattern lines above it (back to the end of the previous arm), as
/// 1-based lines. `Admin` is the fail-closed class: an arm may choose it
/// only with a stated reason, so it never becomes a silent catch-all.
#[must_use]
pub fn unjustified_admin_arms(text: &str) -> Vec<usize> {
    let names_class = text.contains("StatementClass");
    let lines: Vec<&str> = text.lines().collect();
    let justified = |line: &str| {
        line.split_once("// cluster-scope:")
            .is_some_and(|(_, reason)| !reason.trim().is_empty())
    };
    let mut out = Vec::new();
    for (n, line) in lines.iter().enumerate() {
        let code = line.split("//").next().unwrap_or_default();
        if !is_admin_arm(code, names_class) {
            continue;
        }
        let mut ok = justified(line);
        for above in lines[..n].iter().rev() {
            if ok {
                break;
            }
            let trimmed = above.trim();
            if trimmed.starts_with("//") {
                ok = justified(trimmed);
                continue;
            }
            let code = trimmed.split("//").next().unwrap_or_default().trim_end();
            if code.is_empty() || code.ends_with(',') || code.ends_with('{') || code.ends_with('}')
            {
                break;
            }
            ok = justified(trimmed);
        }
        if !ok {
            out.push(n + 1);
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

/// Blocking-call markers banned in `crates/krishiv-exec/src` production
/// code (P2-T061 yield discipline): the same names the workspace
/// `clippy.toml` denies in that crate. `object_store` needs no entry: the
/// executor only names its paths (`object_store::path::Path`) and opens
/// stores; every read goes through the async `IoScheduler` on the I/O
/// runtime, so no synchronous `object_store` API is used in `krishiv-exec`.
pub const EXEC_BLOCKING_MARKERS: &[&str] = &["block_on", "std::fs::", "std::thread::sleep"];

/// Whether `rel_path` (relative to the repository root) is test-only by
/// name: compiled under `#[cfg(test)]` (or only reached from it), so the
/// blocking-call ban does not apply. File-module test code the ban would
/// otherwise flag (`mod tests;` under `#[cfg(test)]`) is found by
/// [`exec_test_only_files`].
#[must_use]
pub fn is_exec_test_file(rel_path: &str) -> bool {
    let p = rel_path.replace('\\', "/");
    if !p.starts_with("crates/krishiv-exec/src/") || !p.ends_with(".rs") {
        return false;
    }
    let name = p.rsplit('/').next().unwrap_or(&p);
    name == "tests.rs"
        || name == "loom_model.rs"
        || name.ends_with("_tests.rs")
        || p.contains("/tests/")
}

/// Whether a `#[cfg(...)]` attribute line gates on `test`: the negated
/// `not(test)` form does not count (code compiled only outside tests is
/// production code, so the blocking-call ban still applies to it).
fn mentions_test_cfg(attr: &str) -> bool {
    if !attr.contains("cfg(") || !attr.contains("test") {
        return false;
    }
    let compact: String = attr.chars().filter(|c| !c.is_whitespace()).collect();
    !compact.contains("not(test")
}

/// Files under `crates/krishiv-exec/src` that are only compiled for tests:
/// `files` is every `(path, text)` pair of that directory. A `mod name;`
/// declaration whose contiguous `#[cfg]` block gates on `test` (but not
/// the negated `not(test)` form) with no `feature` resolves to a test-only
/// file (`pages`/`table` in `rows` are also compiled with a feature, so
/// they stay production).
#[must_use]
pub fn exec_test_only_files(files: &[(&str, &str)]) -> Vec<String> {
    let mut out = Vec::new();
    for (path, text) in files {
        let p = path.replace('\\', "/");
        if !p.starts_with("crates/krishiv-exec/src/") || !p.ends_with(".rs") {
            continue;
        }
        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;
        while i < lines.len() {
            let t = lines[i].trim_start();
            if t.starts_with("#[") {
                let mut j = i;
                let mut mentions_test = false;
                let mut mentions_feature = false;
                while j < lines.len() && lines[j].trim_start().starts_with("#[") {
                    let a = lines[j].trim_start();
                    mentions_test = mentions_test || mentions_test_cfg(a);
                    mentions_feature = mentions_feature || a.contains("feature");
                    j += 1;
                }
                if mentions_test && !mentions_feature && j < lines.len() {
                    // `mod name;` (a file), not `mod name {` (inline, which
                    // the per-file scan skips itself).
                    let item = lines[j].trim_start();
                    let item = item
                        .strip_prefix("pub(crate) ")
                        .or_else(|| item.strip_prefix("pub(super) "))
                        .or_else(|| item.strip_prefix("pub "))
                        .unwrap_or(item);
                    if let Some(name) = item
                        .strip_prefix("mod ")
                        .and_then(|r| r.strip_suffix(';'))
                        .map(str::trim)
                        .filter(|n| {
                            !n.is_empty()
                                && n.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
                        })
                    {
                        for candidate in [
                            format!("{}/{name}.rs", parent_dir(&p)),
                            format!("{}/{name}/mod.rs", parent_dir(&p)),
                            format!("{}/{name}.rs", stem_dir(&p)),
                            format!("{}/{name}/mod.rs", stem_dir(&p)),
                        ] {
                            if files.iter().any(|(f, _)| f.replace('\\', "/") == candidate) {
                                out.push(candidate);
                            }
                        }
                    }
                }
                i = j;
            } else {
                i += 1;
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// The directory of `p` (`a/b/c.rs` to `a/b`).
fn parent_dir(p: &str) -> &str {
    p.rsplit_once('/').map_or("", |(d, _)| d)
}

/// The file's own stem directory (`a/b/c.rs` to `a/b/c`): where Rust looks
/// for `mod name;` declared in a non-`mod.rs` file.
fn stem_dir(p: &str) -> String {
    let stem = p
        .rsplit('/')
        .next()
        .unwrap_or(p)
        .strip_suffix(".rs")
        .unwrap_or(p);
    format!("{}/{stem}", parent_dir(p))
}

/// `count` narrowed to `i32`, saturating (brace counts per line never reach
/// it): the depth arithmetic below stays in `i32` without a truncating cast.
fn narrow(count: usize) -> i32 {
    i32::try_from(count).unwrap_or(i32::MAX)
}

/// Masks a Rust source into code lines: comments (`//`, nested `/* */`),
/// string literals (with escapes, `b`/`c` prefixes and raw `r`/`br`/`cr`
/// strings with any number of `#`) and char literals become spaces, so a
/// search over the result only sees code. One output line per input line.
fn code_mask(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let at = |i: usize| chars.get(i).copied().unwrap_or('\0');
    let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let mut masked = vec![' '; chars.len()];
    let mut line_of = vec![0usize; chars.len()];
    let mut line = 0;
    for (i, c) in chars.iter().enumerate() {
        if *c == '\n' {
            line += 1;
        } else {
            line_of[i] = line;
        }
    }
    let mut lines: Vec<String> = vec![String::new(); line + 1];
    let mut i = 0;
    let put = |masked: &mut Vec<char>, i: usize| masked[i] = chars[i];
    while i < chars.len() {
        let c = chars[i];
        // Raw strings first: `r`, `br` or `cr` with any number of `#`.
        let mut prefix = 0;
        if c == 'r' && at(i + 1) == '"' || c == 'r' && at(i + 1) == '#' {
            prefix = 1;
        } else if (c == 'b' || c == 'c') && at(i + 1) == 'r' {
            prefix = 2;
        }
        if prefix > 0 && (i == 0 || !is_ident(at(i.wrapping_sub(1)))) {
            let mut j = i + prefix;
            let mut hashes = 0;
            while at(j) == '#' {
                hashes += 1;
                j += 1;
            }
            if at(j) == '"' {
                let mut k = j + 1;
                loop {
                    if k >= chars.len() {
                        break;
                    }
                    if chars[k] == '"' && (0..hashes).all(|h| at(k + 1 + h) == '#') {
                        k += 1 + hashes;
                        break;
                    }
                    k += 1;
                }
                i = k;
                continue;
            }
        }
        match c {
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
                        _ => {}
                    }
                    k += 1;
                }
                i = k;
            }
            '"' => {
                let mut k = i + 1;
                while k < chars.len() && chars[k] != '"' {
                    if chars[k] == '\\' {
                        k += 1;
                    }
                    k += 1;
                }
                i = if k < chars.len() { k + 1 } else { k };
            }
            '\'' => {
                let end = if at(i + 1) == '\\' {
                    (i + 3..i + 12).find(|&k| at(k) == '\'')
                } else if at(i + 2) == '\'' {
                    Some(i + 2)
                } else {
                    None
                };
                i = end.map_or(i + 1, |e| e + 1);
            }
            _ => {
                put(&mut masked, i);
                i += 1;
            }
        }
    }
    for (i, c) in masked.iter().enumerate() {
        if chars[i] != '\n' {
            lines[line_of[i]].push(*c);
        }
    }
    lines
}

/// Whether `marker` occurs in `code` outside a larger identifier.
fn has_marker(code: &str, marker: &str) -> bool {
    let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    let bytes = code.as_bytes();
    let mark = marker.as_bytes();
    if mark.is_empty() || mark.len() > bytes.len() {
        return false;
    }
    (0..=bytes.len() - mark.len()).any(|i| {
        // A marker ending in `::` is a path prefix (`std::fs::` covers every
        // sync file call); only its start needs a boundary.
        let trailing = mark.ends_with(b":");
        code.is_char_boundary(i)
            && code.is_char_boundary(i + mark.len())
            && &bytes[i..i + mark.len()] == mark
            && (i == 0 || !is_ident(code[..i].chars().next_back().unwrap_or(' ')))
            && (trailing
                || (i + mark.len() == bytes.len()
                    || !is_ident(code[i + mark.len()..].chars().next().unwrap_or(' '))))
    })
}

/// Whether the masked line starts an `fn` or `mod` item, and whether it is
/// bodyless (`mod name;`, `fn f();`): only items with bodies are skipped.
fn fn_or_mod_item(t: &str) -> (bool, bool) {
    let mut r = t;
    loop {
        let before = r.len();
        for prefix in [
            "pub(crate) ",
            "pub(super) ",
            "pub ",
            "async ",
            "unsafe ",
            "const ",
            "default ",
        ] {
            r = r.strip_prefix(prefix).unwrap_or(r);
        }
        if r.len() == before {
            break;
        }
    }
    let is = r.starts_with("fn ") || r.starts_with("fn(") || r.starts_with("mod ");
    (is, is && t.contains(';') && !t.contains('{'))
}

/// Blocking-call markers in the production code of `text` as `(line,
/// marker)`: `[]` for files outside `crates/krishiv-exec/src` or for
/// test-only files. Skips comments, string literals, test-gated items
/// (`#[cfg(test...)]` but not `#[cfg(not(test))]`, `#[test]`,
/// `#[tokio::test...]`) and items the crate's
/// deliberate `#[expect(clippy::disallowed_methods, ...)]` covers (that
/// attribute, or one of the two lines above the hit, names it).
#[must_use]
pub fn exec_blocking_calls_in(rel_path: &str, text: &str) -> Vec<(usize, &'static str)> {
    let p = rel_path.replace('\\', "/");
    if !p.starts_with("crates/krishiv-exec/src/") || !p.ends_with(".rs") || is_exec_test_file(&p) {
        return Vec::new();
    }
    let masked = code_mask(text);
    let raw: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut depth: i32 = 0;
    let mut pending_test = false;
    let mut pending_expect = false;
    let mut pending_test_fn = false;
    let mut skipping = false;
    let mut skip_depth = 0i32;
    let mut entered = false;
    // Brackets opened by an attribute and not yet closed: attribute
    // continuations (`reason = "..."`) keep the pending flags.
    let mut attr_depth: i32 = 0;
    let brackets = |t: &str| {
        narrow(t.bytes().filter(|&b| b == b'(' || b == b'[').count())
            - narrow(t.bytes().filter(|&b| b == b')' || b == b']').count())
    };
    for (n, code) in masked.iter().enumerate() {
        let t = code.trim_start();
        let opens = narrow(code.bytes().filter(|&b| b == b'{').count());
        let closes = narrow(code.bytes().filter(|&b| b == b'}').count());
        let blank = t.is_empty();
        if skipping {
            depth += opens - closes;
            if opens > 0 {
                entered = true;
            }
            if entered && depth <= skip_depth {
                skipping = false;
                entered = false;
            }
            continue;
        }
        if blank {
            // Blank lines and comments (masked out) never end an
            // attribute run.
        } else if t.starts_with("#[") || attr_depth != 0 {
            if mentions_test_cfg(t) {
                pending_test = true;
            }
            if t.contains("disallowed_methods") {
                pending_expect = true;
            }
            if t.starts_with("#[test") || t.starts_with("#[tokio::test") {
                pending_test_fn = true;
            }
            attr_depth += brackets(t);
        } else {
            let (is_fm, bodyless) = fn_or_mod_item(t);
            if (pending_test || pending_test_fn || pending_expect) && is_fm && !bodyless {
                pending_test = false;
                pending_expect = false;
                pending_test_fn = false;
                if opens > 0 && opens == closes {
                    // The whole item is on this line (`fn f() { ... }`):
                    // suppress it, but nothing after it.
                    depth += opens - closes;
                    continue;
                }
                skipping = true;
                skip_depth = depth;
                entered = opens > 0;
                depth += opens - closes;
                continue;
            }
            // Any other line (a statement, a `mod name;` declaration, an
            // import naming `std::fs`) ends the attribute run.
            pending_test = false;
            pending_expect = false;
            pending_test_fn = false;
            for marker in EXEC_BLOCKING_MARKERS {
                if has_marker(code, marker) && !expect_guarded(&raw, n) {
                    out.push((n + 1, *marker));
                }
            }
        }
        depth += opens - closes;
    }
    out
}

/// Whether `raw[n]` (or one of the two lines above it) names the deliberate
/// `#[expect(clippy::disallowed_methods, ...)]` covering the hit.
fn expect_guarded(raw: &[&str], n: usize) -> bool {
    if raw.is_empty() {
        return false;
    }
    raw[n.saturating_sub(2)..=n.min(raw.len() - 1)]
        .iter()
        .any(|l| l.contains("disallowed_methods"))
}

#[cfg(test)]
mod exec_discipline_tests {
    use super::*;

    const PATH: &str = "crates/krishiv-exec/src/driver.rs";

    #[test]
    fn ignores_expect_covered_multiline_functions() {
        let text = "    #[expect(\n        clippy::disallowed_methods,\n        reason = \"x\"\n    )]\n    fn finalize<'a>(\n        &'a self,\n    ) -> BoxFuture<'a, Result<()>> {\n        std::thread::sleep(d);\n    }\n";
        assert_eq!(exec_blocking_calls_in(PATH, text), []);
    }

    #[test]
    fn flags_blocking_calls_in_production_code() {
        let text = "fn f() {\n    let _ = std::fs::read_to_string(\"x\");\n    std::thread::sleep(d);\n    rt.block_on(g()).await;\n}\n";
        let hits = exec_blocking_calls_in(PATH, text);
        assert_eq!(
            hits,
            [(2, "std::fs::"), (3, "std::thread::sleep"), (4, "block_on")]
        );
    }

    #[test]
    fn ignores_cfg_test_modules() {
        let text = "fn f() {}\n#[cfg(test)]\nmod tests {\n    fn g(rt: &Runtime) { rt.block_on(h()); }\n}\nfn k() {}\n";
        assert!(exec_blocking_calls_in(PATH, text).is_empty());
    }

    #[test]
    fn ignores_stacked_cfg_test_attributes() {
        let text = "#[cfg(test)]\n#[cfg(not(loom))]\nmod proof_tests {\n    let t = std::fs::read_to_string(p);\n}\n";
        assert!(exec_blocking_calls_in(PATH, text).is_empty());
    }

    #[test]
    fn flags_blocking_calls_under_not_test() {
        // `not(test)` compiles only outside tests: production code, so the
        // ban applies.
        let text = "#[cfg(not(test))]\nfn f() {\n    std::thread::sleep(d);\n}\n";
        assert_eq!(
            exec_blocking_calls_in(PATH, text),
            [(3, "std::thread::sleep")]
        );
    }

    #[test]
    fn ignores_comments_and_strings() {
        let text = "// rt.block_on(g())\n/* std::fs::read */\nlet s = \"std::thread::sleep\";\nlet c = 'x';\n";
        assert!(exec_blocking_calls_in(PATH, text).is_empty());
    }

    #[test]
    fn ignores_expect_covered_lines() {
        let text = "#[expect(clippy::disallowed_methods, reason = \"x\")]\nfn f() { std::fs::File::open(p); }\n";
        assert!(exec_blocking_calls_in(PATH, text).is_empty());
    }

    #[test]
    fn ignores_expect_covered_function_bodies() {
        let text = "#[expect(\n    clippy::disallowed_methods,\n    reason = \"blocking pool\"\n)]\nfn finalize() {\n    let d = Duration::from_millis(1);\n    std::thread::sleep(d);\n}\nfn other() { std::thread::sleep(d); }\n";
        assert_eq!(
            exec_blocking_calls_in(PATH, text),
            [(9, "std::thread::sleep")]
        );
    }

    #[test]
    fn ignores_test_annotated_functions() {
        let text = "#[test]\nfn spinning() { std::thread::sleep(d); }\nfn prod() { std::thread::sleep(d); }\n";
        assert_eq!(
            exec_blocking_calls_in(PATH, text),
            [(3, "std::thread::sleep")]
        );
    }

    #[test]
    fn does_not_match_larger_identifiers() {
        let text = "fn block_one() {}\nlet block_on_io = 1;\n";
        assert!(exec_blocking_calls_in(PATH, text).is_empty());
    }

    #[test]
    fn skips_files_outside_exec_production_sources() {
        let text = "fn f() { std::thread::sleep(d); }\n";
        assert!(exec_blocking_calls_in("crates/krishiv-io/src/scan.rs", text).is_empty());
        assert!(exec_blocking_calls_in("crates/krishiv-exec/tests/executor.rs", text).is_empty());
        assert!(
            exec_blocking_calls_in(
                "crates/krishiv-exec/src/window/tests.rs",
                "fn f() { rt.block_on(g()); }\n"
            )
            .is_empty()
        );
        assert!(
            exec_blocking_calls_in(
                "crates/krishiv-exec/src/loom_model.rs",
                "fn f() { block_on(g()); }\n"
            )
            .is_empty()
        );
    }

    #[test]
    fn finds_test_only_file_modules() {
        let files = [
            (
                "crates/krishiv-exec/src/aggregate/mod.rs",
                "#[cfg(test)]\nmod tests;\n",
            ),
            (
                "crates/krishiv-exec/src/aggregate/tests.rs",
                "fn f() { rt.block_on(g()); }\n",
            ),
            (
                "crates/krishiv-exec/src/rows/mod.rs",
                "#[cfg(any(test, feature = \"row-table-prototype\"))]\nmod pages;\n",
            ),
            ("crates/krishiv-exec/src/rows/pages.rs", "fn f() {}\n"),
        ];
        // `tests.rs` resolves through its `mod tests;` declaration; `pages.rs`
        // stays production: its gate also compiles it with a feature.
        assert_eq!(
            exec_test_only_files(&files),
            ["crates/krishiv-exec/src/aggregate/tests.rs".to_owned()]
        );
        assert!(is_exec_test_file(
            "crates/krishiv-exec/src/aggregate/tests.rs"
        ));
        assert!(!is_exec_test_file("crates/krishiv-exec/src/rows/pages.rs"));
    }

    #[test]
    fn resolves_non_mod_rs_test_modules() {
        let files = [
            (
                "crates/krishiv-exec/src/aggregate/accumulator.rs",
                "#[cfg(test)]\n#[cfg(not(loom))]\nmod differential;\n",
            ),
            (
                "crates/krishiv-exec/src/aggregate/accumulator/differential.rs",
                "fn f() { rt.block_on(g()); }\n",
            ),
        ];
        assert_eq!(
            exec_test_only_files(&files),
            ["crates/krishiv-exec/src/aggregate/accumulator/differential.rs".to_owned()]
        );
    }

    #[test]
    fn not_test_modules_stay_production() {
        // `not(test)` compiles only outside tests: the file stays
        // production, so the blocking-call ban still covers it.
        let files = [
            (
                "crates/krishiv-exec/src/driver.rs",
                "#[cfg(not(test))]\nmod shim;\n",
            ),
            (
                "crates/krishiv-exec/src/shim.rs",
                "fn f() { rt.block_on(g()); }\n",
            ),
        ];
        assert!(exec_test_only_files(&files).is_empty());
    }
}
