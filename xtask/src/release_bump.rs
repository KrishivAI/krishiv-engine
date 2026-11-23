// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask release-bump VERSION [--check]`: moves the workspace to a release version in one step
//! (RELEASING.md, "Release checklist").
//!
//! It rewrites `[workspace.package] version` in the root manifest and every exact `version = "=X"` pin of an
//! internal path dependency in the crate manifests, then prints the lockfile step. `--check` writes nothing and
//! fails when any file would change. It touches no prose (README, docs): those name a release by hand.
//! Python's version is dynamic (`pyproject.toml`) and follows the crate version, so it needs no edit.

#![expect(clippy::print_stdout, reason = "command output is the interface")]

use std::path::{Path, PathBuf};

use crate::release_check::{is_version, workspace_version};

/// Directories that never hold a manifest of ours.
const SKIP: &[&str] = &["target", ".git", ".worktrees", "node_modules", ".venv"];

/// The byte range of the version digits of an exact pin (`version = "=X.Y.Z"`) on a one-line dependency that also
/// declares `path`. A pin on a dependency without `path` is an external crate and is not ours to move.
fn internal_pin(line: &str) -> Option<(usize, usize)> {
    if !line.contains("path") {
        return None;
    }
    let key = line.find("version")?;
    let rest = line[key + "version".len()..].trim_start();
    let rest = rest.strip_prefix('=')?.trim_start().strip_prefix("\"=")?;
    let lo = line.len() - rest.len();
    Some((lo, lo + rest.find('"')?))
}

/// The byte range of the value of a `version = "X.Y.Z"` line, in any spacing and with a trailing comment.
fn version_value(line: &str) -> Option<(usize, usize)> {
    let rest = line.trim_start().strip_prefix("version")?;
    let rest = rest
        .trim_start()
        .strip_prefix('=')?
        .trim_start()
        .strip_prefix('"')?;
    let lo = line.len() - rest.len();
    Some((lo, lo + rest.find('"')?))
}

/// `text` with the workspace version line (the root manifest) and every exact internal path pin moved to `new`,
/// whatever version they hold now (so a half-done bump is repaired). `None` when nothing changes.
fn bump_manifest(text: &str, new: &str, root: bool) -> Option<String> {
    let mut changed = false;
    let mut in_package = false;
    let mut out = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_package = trimmed == "[workspace.package]";
        }
        let span = if root && in_package {
            version_value(line)
        } else {
            internal_pin(line)
        };
        match span {
            Some((lo, hi)) if &line[lo..hi] != new => {
                out.push_str(&line[..lo]);
                out.push_str(new);
                out.push_str(&line[hi..]);
                changed = true;
            }
            _ => out.push_str(line),
        }
    }
    changed.then_some(out)
}

/// Every `Cargo.toml` under `dir`, skipping build output and worktrees.
fn manifests(dir: &Path, found: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let name = entry.file_name();
        if path.is_dir() {
            if !SKIP.iter().any(|s| name == *s) {
                manifests(&path, found)?;
            }
        } else if name == "Cargo.toml" {
            found.push(path);
        }
    }
    Ok(())
}

/// Runs the command.
///
/// # Errors
/// An unreadable manifest, a bad version, or (`--check`) a manifest that is not at the version.
pub fn run(root: &Path, flags: &[&str]) -> Result<(), String> {
    let (version, check) = match flags {
        [v] => (*v, false),
        [v, "--check"] | ["--check", v] => (*v, true),
        other => {
            return Err(format!(
                "usage: release-bump VERSION [--check], got {other:?}"
            ));
        }
    };
    if !is_version(version) {
        return Err(format!("{version} is not a version (X.Y.Z[-pre])"));
    }
    let root_manifest = root.join("Cargo.toml");
    let text = std::fs::read_to_string(&root_manifest).map_err(|e| format!("Cargo.toml: {e}"))?;
    let old = workspace_version(&text)?;
    let mut files = Vec::new();
    manifests(root, &mut files)?;
    // The root manifest goes last: a failed write leaves the old root version, so a re-run still finds the work.
    files.sort_by_key(|p| (*p == root_manifest, p.clone()));
    let mut touched = 0;
    for path in files {
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let Some(new_text) = bump_manifest(&text, version, path == root_manifest) else {
            continue;
        };
        touched += 1;
        let rel = path.strip_prefix(root).unwrap_or(&path);
        if check {
            println!("would change {}", rel.display());
        } else {
            std::fs::write(&path, new_text).map_err(|e| format!("{}: {e}", path.display()))?;
            println!("bumped {}", rel.display());
        }
    }
    if touched == 0 {
        println!("release-bump: the workspace is already at {version}");
        return Ok(());
    }
    if check {
        return Err(format!(
            "release-bump: {touched} manifest(s) are not at {version}"
        ));
    }
    println!(
        "release-bump: {touched} manifest(s) moved {old} -> {version}; now run `cargo update --workspace` \
         (refreshes Cargo.lock) and `cargo xtask release-check {version}`"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_root_version_and_the_pins_move() {
        let root = "[workspace.package]\nversion = \"0.2.1\"\n[workspace.dependencies]\nlibc = { version = \"0.2.1\" }\n";
        let out = bump_manifest(root, "0.3.0", true).unwrap_or_default();
        assert!(out.contains("[workspace.package]\nversion = \"0.3.0\""));
        assert!(out.contains("libc = { version = \"0.2.1\" }"), "{out}");
        let krate =
            "[dependencies]\na = { path = \"../a\", version = \"=0.2.1\", optional = true }\n";
        assert_eq!(
            bump_manifest(krate, "0.3.0", false).unwrap_or_default(),
            "[dependencies]\na = { path = \"../a\", version = \"=0.3.0\", optional = true }\n"
        );
    }

    #[test]
    fn a_manifest_at_another_version_is_untouched() {
        assert!(
            bump_manifest(
                "a = { version = \"=0.3.0\", path = \"../a\" }\n",
                "0.3.0",
                false
            )
            .is_none()
        );
        assert!(bump_manifest("[package]\nversion = \"0.2.1\"\n", "0.3.0", false).is_none());
    }

    #[test]
    fn a_workspace_moves_and_check_reports_the_rest() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let root = dir.path();
        std::fs::create_dir_all(root.join("crates/a")).unwrap_or_default();
        std::fs::create_dir_all(root.join("target/x")).unwrap_or_default();
        let write = |p: &str, t: &str| std::fs::write(root.join(p), t).unwrap_or_default();
        write("Cargo.toml", "[workspace.package]\nversion = \"0.2.1\"\n");
        write(
            "crates/a/Cargo.toml",
            "b = { path = \"../b\", version = \"=0.2.1\" }\n",
        );
        write("target/x/Cargo.toml", "b = { version = \"=0.2.1\" }\n");
        assert!(run(root, &["0.3.0", "--check"]).is_err());
        assert!(run(root, &["0.3"]).is_err());
        assert!(run(root, &["0.3.0"]).is_ok());
        let read = |p: &str| std::fs::read_to_string(root.join(p)).unwrap_or_default();
        assert!(read("Cargo.toml").contains("version = \"0.3.0\""));
        assert!(read("crates/a/Cargo.toml").contains("=0.3.0"));
        assert!(
            read("target/x/Cargo.toml").contains("=0.2.1"),
            "skipped dirs stay"
        );
        assert!(run(root, &["0.3.0", "--check"]).is_ok());
    }

    #[test]
    fn an_external_pin_stays_and_root_spacing_is_free() {
        let ext = "x = { version = \"=0.2.1\" }\n";
        assert!(bump_manifest(ext, "0.3.0", false).is_none());
        let root = "[workspace.package]\nversion=\"0.2.1\" # release\n";
        assert_eq!(
            bump_manifest(root, "0.3.0", true).unwrap_or_default(),
            "[workspace.package]\nversion=\"0.3.0\" # release\n"
        );
    }

    #[test]
    fn a_root_at_the_version_with_a_stale_pin_is_repaired() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let root = dir.path();
        std::fs::create_dir_all(root.join("crates/a")).unwrap_or_default();
        let write = |p: &str, t: &str| std::fs::write(root.join(p), t).unwrap_or_default();
        write("Cargo.toml", "[workspace.package]\nversion = \"0.3.0\"\n");
        write(
            "crates/a/Cargo.toml",
            "b = { path = \"../b\", version = \"=0.2.1\" }\n",
        );
        assert!(run(root, &["0.3.0", "--check"]).is_err());
        assert!(run(root, &["0.3.0"]).is_ok());
        let pin = std::fs::read_to_string(root.join("crates/a/Cargo.toml")).unwrap_or_default();
        assert!(pin.contains("=0.3.0"), "{pin}");
        assert!(run(root, &["0.3.0", "--check"]).is_ok());
    }
}
