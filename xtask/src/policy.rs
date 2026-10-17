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

    fn range(&self) -> String {
        format!("{}...{}", self.base, self.head)
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
    let numstat = args.git(&["diff", "--numstat", &args.range()])?;
    let report = ci::measure_numstat(&numstat)?;
    let summary = format!(
        "## PR size\n\n{} counted changed lines (cap {}); {} more in goldens/fixtures/vendored/generated files (excluded).",
        report.counted,
        ci::MAX_CHANGED_LINES,
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
        "--format=%H%x1f%ae%x1f%B%x1e",
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

/// `xtask source-guard`: bans the parquet `object_store` adapters in `crates/`.
///
/// # Errors
/// Returns each banned use with its location.
pub fn run_source_guard(root: &Path) -> Result<(), String> {
    let mut files = Vec::new();
    collect(&root.join("crates"), "rs", &mut files);
    let mut hits = Vec::new();
    for file in &files {
        let Ok(text) = std::fs::read_to_string(file) else {
            continue;
        };
        for (line, ident) in lint::banned_idents_in(&text) {
            hits.push(format!(
                "{}:{line}: `{ident}` is banned (parquet's object_store feature is off; ADR 0028)",
                file.strip_prefix(root).unwrap_or(file).display()
            ));
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
        for hook in ["pre-commit", "pre-push"] {
            let path = root.join(".githooks").join(hook);
            let mut perms = std::fs::metadata(&path)
                .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
                .permissions();
            perms.set_mode(perms.mode() | 0o111);
            std::fs::set_permissions(&path, perms)
                .map_err(|e| format!("cannot chmod {}: {e}", path.display()))?;
        }
    }
    println!("setup: core.hooksPath = .githooks (pre-commit, pre-push installed)");
    Ok(())
}
