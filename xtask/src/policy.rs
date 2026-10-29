// SPDX-License-Identifier: Apache-2.0
//! I/O wrappers for the CI policy checks in [`crate::ci`] and [`crate::lint`].
#![expect(clippy::print_stdout, reason = "command output is the interface")]

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{ci, lint};

/// Parsed `--repo/--base/--head` flags.
#[derive(Debug)]
pub struct DiffArgs {
    repo: PathBuf,
    base: String,
    head: String,
}

impl DiffArgs {
    /// Parses `[--repo DIR] --base REF [--head REF]` (head defaults to `HEAD`).
    ///
    /// # Errors
    /// Returns a usage error for unknown flags or a missing `--base`.
    pub fn parse(flags: &[&str], default_repo: &Path) -> Result<Self, String> {
        let (mut repo, mut base, mut head) = (default_repo.to_path_buf(), None, "HEAD".to_owned());
        let mut it = flags.iter();
        while let Some(flag) = it.next() {
            let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
            match *flag {
                "--repo" => repo = PathBuf::from(value),
                "--base" => base = Some((*value).to_owned()),
                "--head" => (*value).clone_into(&mut head),
                other => return Err(format!("unknown flag {other}")),
            }
        }
        let base = base.ok_or("--base is required")?;
        Ok(Self { repo, base, head })
    }

    fn git(&self, args: &[&str]) -> Result<String, String> {
        let out = Command::new("git")
            .args(args)
            .current_dir(&self.repo)
            .output()
            .map_err(|e| format!("failed to run git: {e}"))?;
        if out.status.success() {
            Ok(String::from_utf8_lossy(&out.stdout).into_owned())
        } else {
            Err(format!(
                "git {} failed: {}",
                args.join(" "),
                String::from_utf8_lossy(&out.stderr)
            ))
        }
    }

    /// `git diff -U0` of the range, for line-level reports.
    ///
    /// # Errors
    /// Returns the git failure message.
    pub fn unified_zero_diff(&self) -> Result<String, String> {
        self.git(&["diff", "-U0", "--no-color", &self.range()])
    }

    /// `git diff --name-only` of the range, one repository-relative path per entry.
    ///
    /// # Errors
    /// Returns the git failure message.
    pub fn changed_paths(&self) -> Result<Vec<String>, String> {
        // NUL-separated and without rename detection: non-ASCII paths stay verbatim, and a rename
        // lists both its old and new path, so both packages are scoped.
        Ok(self
            .git(&["diff", "--name-only", "-z", "--no-renames", &self.range()])?
            .split('\0')
            .filter(|p| !p.is_empty())
            .map(str::to_owned)
            .collect())
    }

    /// The repository the range is taken in.
    #[must_use]
    pub fn repo(&self) -> &Path {
        &self.repo
    }

    fn range(&self) -> String {
        format!("{}...{}", self.base, self.head)
    }

    fn head_ref(&self) -> &str {
        &self.head
    }
}

fn step_summary(text: &str) {
    if let Some(path) = std::env::var_os("GITHUB_STEP_SUMMARY") {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
        {
            let _ = writeln!(f, "{text}");
        }
    }
}

/// `xtask pr-size`: fails when the diff exceeds the cap, unless `SIZE_APPROVED=1`.
///
/// # Errors
/// Returns the failure message when the cap is exceeded or git fails.
pub fn run_pr_size(args: &DiffArgs) -> Result<(), String> {
    let numstat = args.git(&["diff", "--numstat", "-z", &args.range()])?;
    let mut report = ci::measure_numstat(&numstat)?;
    let deleted = args.git(&[
        "diff",
        "--name-only",
        "-z",
        "--diff-filter=D",
        &args.range(),
    ])?;
    let deleted: Vec<&str> = deleted.split('\0').filter(|p| !p.is_empty()).collect();
    for path in report.binaries.clone() {
        if deleted.contains(&path.as_str()) {
            continue; // a deleted binary adds nothing
        }
        // Fail closed: a binary present at the head must be sized.
        let spec = format!("{}:{path}", args.head_ref());
        let size = args.git(&["cat-file", "-s", &spec])?;
        let bytes = size
            .trim()
            .parse::<u64>()
            .map_err(|e| format!("cannot size binary {path}: {e} ({size:?})"))?;
        report.add_binary(bytes);
    }
    let binaries = if report.binaries.is_empty() {
        String::new()
    } else {
        format!(
            " Binary files (counted at {} bytes per line): {}.",
            ci::BINARY_BYTES_PER_LINE,
            report.binaries.join(", ")
        )
    };
    let summary = format!(
        "## PR size\n\n{} counted changed lines (cap {}), including {} test lines at 1/{} weight; {} more in allowlisted data files (lock files, snapshots, Spark goldens, test fixtures, vendored data; see `xtask/src/ci.rs`).{binaries}",
        report.counted,
        ci::MAX_CHANGED_LINES,
        report.test_lines,
        ci::TEST_LINE_DIVISOR,
        report.excluded
    );
    println!("{summary}");
    step_summary(&summary);
    if report.within_cap() {
        return Ok(());
    }
    if std::env::var("SIZE_APPROVED").is_ok_and(|v| v == "1") {
        println!("Over the cap but approved via the `size-approved` label.");
        return Ok(());
    }
    Err(format!(
        "PR changes {} lines (cap {}). Split it, or ask the maintainer to apply the `size-approved` label.",
        report.counted,
        ci::MAX_CHANGED_LINES
    ))
}

/// `xtask changes`: prints (and exports to `GITHUB_OUTPUT`) `rust=true|false`.
///
/// # Errors
/// Returns an error when git fails.
pub fn run_changes(args: &DiffArgs) -> Result<(), String> {
    let names = args.git(&["diff", "--name-only", &args.range()])?;
    let rust = ci::rust_relevant(names.lines());
    println!("rust={rust}");
    if let Some(path) = std::env::var_os("GITHUB_OUTPUT") {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
            .map_err(|e| format!("cannot open GITHUB_OUTPUT: {e}"))?;
        writeln!(f, "rust={rust}").map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// `xtask dco`: every non-merge commit in `base..head` must be signed off.
///
/// # Errors
/// Returns the offending commits.
pub fn run_dco(args: &DiffArgs) -> Result<(), String> {
    let log = args.git(&[
        "log",
        "--no-merges",
        "--format=%H%x1f%ae%x1f%ce%x1f%B%x1e",
        &format!("{}..{}", args.base, args.head),
    ])?;
    let bad = ci::commits_missing_signoff(&log);
    if bad.is_empty() {
        println!("dco: OK");
        return Ok(());
    }
    Err(format!(
        "commits without a `Signed-off-by:` trailer (use `git commit -s`; fix with `git rebase --signoff`):\n{}",
        bad.iter()
            .map(|s| format!("  {s}"))
            .collect::<Vec<_>>()
            .join("\n")
    ))
}

fn collect(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let path = entry.path();
        let name = entry.file_name();
        if path.is_dir() {
            if name != "target" && name != ".venv" && name != "node_modules" {
                collect(&path, ext, out);
            }
        } else if path.extension().is_some_and(|e| e == ext) {
            out.push(path);
        }
    }
}

/// `xtask workflow-lint`: scans `dir/*.yml|yaml` for banned patterns.
///
/// # Errors
/// Returns every violation found.
pub fn run_workflow_lint(dir: &Path) -> Result<(), String> {
    let mut violations = Vec::new();
    let mut files = Vec::new();
    collect(dir, "yml", &mut files);
    collect(dir, "yaml", &mut files);
    for file in &files {
        let text = std::fs::read_to_string(file)
            .map_err(|e| format!("cannot read {}: {e}", file.display()))?;
        violations.extend(lint::workflow_violations(
            &file.display().to_string(),
            &text,
        ));
    }
    if violations.is_empty() {
        println!("workflow-lint: OK ({} files)", files.len());
        Ok(())
    } else {
        Err(violations.join("\n"))
    }
}

/// `xtask spdx`: `.rs` files under `crates/` and `xtask/` and `.py` files under
/// `python/` need the SPDX header. (`bench/` and `tests/` join once their
/// owners have added headers.)
///
/// # Errors
/// Returns the files without a header.
pub fn run_spdx(root: &Path) -> Result<(), String> {
    let mut files = Vec::new();
    for dir in ["crates", "xtask"] {
        collect(&root.join(dir), "rs", &mut files);
    }
    collect(&root.join("python"), "py", &mut files);
    let missing: Vec<String> = files
        .iter()
        .filter(|f| !std::fs::read_to_string(f).is_ok_and(|t| lint::has_spdx_header(&t)))
        .map(|f| f.strip_prefix(root).unwrap_or(f).display().to_string())
        .collect();
    if missing.is_empty() {
        println!("spdx: OK ({} files)", files.len());
        Ok(())
    } else {
        Err(format!(
            "missing `{}` in the first lines of:\n{}",
            lint::SPDX_TAG,
            missing.join("\n")
        ))
    }
}

/// `xtask source-guard`: bans the parquet `object_store` adapters in `crates/`,
/// confines `unsafe` in the kernel crates to `unsafe_*.rs` modules, requires
/// a `// cluster-scope:` reason on every `Admin` statement-class arm, and bans
/// blocking calls in `crates/krishiv-exec/src` production code (P2-T061 yield
/// discipline; the compile-time backstop is the workspace `clippy.toml`
/// `disallowed-methods` list denied at that crate's root).
///
/// # Errors
/// Returns each banned use with its location.
pub fn run_source_guard(root: &Path) -> Result<(), String> {
    let mut files = Vec::new();
    collect(&root.join("crates"), "rs", &mut files);
    // Every exec source, to resolve test-only file modules (`mod tests;`
    // under `#[cfg(test)]`) before scanning for blocking calls.
    let mut exec_sources = Vec::new();
    for file in &files {
        let Ok(text) = std::fs::read_to_string(file) else {
            continue;
        };
        let rel = file
            .strip_prefix(root)
            .unwrap_or(file)
            .display()
            .to_string();
        if rel.starts_with("crates/krishiv-exec/src/") {
            exec_sources.push((rel, text));
        }
    }
    let exec_test_only: std::collections::BTreeSet<String> = lint::exec_test_only_files(
        &exec_sources
            .iter()
            .map(|(p, t)| (p.as_str(), t.as_str()))
            .collect::<Vec<_>>(),
    )
    .into_iter()
    .collect();
    let mut hits = Vec::new();
    for file in &files {
        let Ok(text) = std::fs::read_to_string(file) else {
            continue;
        };
        let rel = file
            .strip_prefix(root)
            .unwrap_or(file)
            .display()
            .to_string();
        for (line, ident) in lint::banned_idents_in(&text) {
            hits.push(format!(
                "{rel}:{line}: `{ident}` is banned (parquet's object_store feature is off; ADR 0028)"
            ));
        }
        for (line, ident) in lint::confined_idents_in(&rel, &text) {
            hits.push(format!(
                "{rel}:{line}: `{ident}` is for the SQL token redactor only (ADR 0038): use `krishiv_sql::redact_sql`"
            ));
        }
        for (line, marker) in lint::plan_handles_in(&rel, &text) {
            hits.push(format!(
                "{rel}:{line}: `{marker}` in a physical-plan type: plans are data, live resources are resolved at instantiation (design P2-T171, I1)"
            ));
        }
        for line in lint::unjustified_admin_arms(&text) {
            hits.push(format!(
                "{rel}:{line}: an `Admin` statement-class arm needs a `// cluster-scope: <reason>` comment (Admin is fail-closed, never a catch-all)"
            ));
        }
        if rel.starts_with("crates/krishiv-exec/src/") && !exec_test_only.contains(&rel) {
            for (line, marker) in lint::exec_blocking_calls_in(&rel, &text) {
                hits.push(format!(
                    "{rel}:{line}: `{marker}` blocks a compute worker: await on the I/O runtime or use TaskContext::run_blocking (yield discipline, P2-T061)"
                ));
            }
        }
        if !lint::unsafe_allowed_in(&rel) {
            for line in lint::unsafe_lines(&text) {
                hits.push(format!(
                    "{rel}:{line}: `unsafe` belongs in an `unsafe_*.rs` helper module in this crate (ADR 0029)"
                ));
            }
        }
    }
    if hits.is_empty() {
        println!("source-guard: OK ({} files)", files.len());
        Ok(())
    } else {
        Err(hits.join("\n"))
    }
}

/// `xtask setup`: points git at the versioned hooks.
///
/// # Errors
/// Returns an error when git or chmod fails.
pub fn run_setup(root: &Path) -> Result<(), String> {
    let status = Command::new("git")
        .args(["config", "core.hooksPath", ".githooks"])
        .current_dir(root)
        .status()
        .map_err(|e| format!("failed to run git: {e}"))?;
    if !status.success() {
        return Err("git config core.hooksPath failed".to_owned());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for hook in ["pre-commit", "pre-push", "commit-msg"] {
            let path = root.join(".githooks").join(hook);
            let mut perms = std::fs::metadata(&path)
                .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
                .permissions();
            perms.set_mode(perms.mode() | 0o111);
            std::fs::set_permissions(&path, perms)
                .map_err(|e| format!("cannot chmod {}: {e}", path.display()))?;
        }
    }
    println!("setup: core.hooksPath = .githooks (pre-commit, pre-push, commit-msg installed)");
    Ok(())
}
