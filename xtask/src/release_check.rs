// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask release-check [VERSION]`: a read-only preflight for a release (RELEASING.md, "Release checklist").
//!
//! It publishes nothing and writes nothing. It checks what can be checked from the checkout: the workspace
//! version, that every crate inherits it, that `.github/scripts/publish-crates.txt` names real crates in
//! dependency order, that the changelog has the section (or fragments to assemble it), that the release
//! workflows and the stability doc exist, and that the tag is not taken. Every line is `ok` or `FAIL`; any
//! `FAIL` makes the command exit non-zero.

#![expect(clippy::print_stdout, reason = "command output is the interface")]

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

/// Files a release depends on; a missing one is a failed check.
const REQUIRED_FILES: &[&str] = &[
    "RELEASING.md",
    "CHANGELOG.md",
    "docs/concepts/stability.md",
    ".github/workflows/release.yml",
    ".github/workflows/publish-pypi.yml",
    ".github/workflows/publish-crates.yml",
    ".github/workflows/publish-testpypi.yml",
    ".github/scripts/publish-crates.txt",
];

/// `X.Y.Z` with an optional `-pre` suffix: the shape cargo and PyPI both accept here.
fn is_version(v: &str) -> bool {
    let (core, pre) = v.split_once('-').map_or((v, None), |(c, p)| (c, Some(p)));
    let parts: Vec<&str> = core.split('.').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
        && pre.is_none_or(|p| {
            !p.is_empty() && p.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'.')
        })
}

/// The workspace version from the root manifest.
fn workspace_version(root_manifest: &str) -> Result<String, String> {
    let doc: toml::Table = toml::from_str(root_manifest).map_err(|e| format!("Cargo.toml: {e}"))?;
    doc.get("workspace")
        .and_then(|w| w.get("package"))
        .and_then(|p| p.get("version"))
        .and_then(toml::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| "Cargo.toml: no [workspace.package] version".to_owned())
}

/// True when a crate manifest takes its version from the workspace.
fn inherits_version(manifest: &str) -> bool {
    toml::from_str::<toml::Table>(manifest)
        .ok()
        .and_then(|d| {
            d.get("package")?
                .get("version")?
                .get("workspace")?
                .as_bool()
        })
        .unwrap_or(false)
}

/// The `krishiv*` crates a manifest depends on at build or run time (not dev-dependencies), by name.
fn internal_deps(manifest: &str) -> BTreeSet<String> {
    let Ok(doc) = toml::from_str::<toml::Table>(manifest) else {
        return BTreeSet::new();
    };
    let mut out = BTreeSet::new();
    for section in ["dependencies", "build-dependencies"] {
        if let Some(t) = doc.get(section).and_then(toml::Value::as_table) {
            out.extend(t.keys().filter(|k| k.starts_with("krishiv")).cloned());
        }
    }
    out
}

/// Crates of `order` that depend on a crate listed after them (or not listed), as `(crate, dependency)`.
fn order_violations(
    order: &[String],
    deps: &BTreeMap<String, BTreeSet<String>>,
) -> Vec<(String, String)> {
    let position: BTreeMap<&str, usize> = order
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let mut bad = Vec::new();
    for (i, name) in order.iter().enumerate() {
        for dep in deps.get(name).into_iter().flatten() {
            match position.get(dep.as_str()) {
                Some(&j) if j > i => bad.push((name.clone(), dep.clone())),
                None => bad.push((name.clone(), format!("{dep} (not in the list)"))),
                _ => {}
            }
        }
    }
    bad
}

/// Whether the changelog has a `## <version>` section.
fn has_section(changelog: &str, version: &str) -> bool {
    let heading = format!("## {version}");
    changelog.lines().any(|l| l.trim_end() == heading)
}

#[derive(Default)]
struct Report {
    lines: Vec<String>,
    failed: usize,
}

impl Report {
    fn check(&mut self, ok: bool, what: impl Into<String>) {
        let what = what.into();
        if !ok {
            self.failed += 1;
        }
        self.lines
            .push(format!("{} {what}", if ok { "ok  " } else { "FAIL" }));
    }
}

/// Entry point of `cargo xtask release-check [VERSION]`.
///
/// # Errors
/// Returns the failed checks, or a message when a file cannot be read.
pub fn run(root: &Path, flags: &[&str]) -> Result<(), String> {
    let read = |p: &str| std::fs::read_to_string(root.join(p)).map_err(|e| format!("{p}: {e}"));
    let current = workspace_version(&read("Cargo.toml")?)?;
    let version = match flags {
        [] => current.clone(),
        [v] => (*v).to_owned(),
        other => return Err(format!("unknown flags {other:?}")),
    };
    let mut r = Report::default();
    r.check(
        is_version(&version),
        format!("{version} is a version (X.Y.Z[-pre])"),
    );
    r.check(
        current == version,
        format!("[workspace.package] version is {current} (release is {version})"),
    );

    for f in REQUIRED_FILES {
        r.check(root.join(f).is_file(), format!("{f} exists"));
    }

    let mut manifests = BTreeMap::new();
    let entries = std::fs::read_dir(root.join("crates")).map_err(|e| format!("crates: {e}"))?;
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Ok(text) = std::fs::read_to_string(entry.path().join("Cargo.toml")) {
            manifests.insert(name, text);
        }
    }
    let loose: Vec<_> = manifests
        .iter()
        .filter(|(_, m)| !inherits_version(m))
        .map(|(n, _)| n.as_str())
        .collect();
    r.check(
        loose.is_empty(),
        format!("every crate inherits the workspace version (not: {loose:?})"),
    );

    let list = read(".github/scripts/publish-crates.txt").unwrap_or_default();
    let order: Vec<String> = list
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_owned)
        .collect();
    let unique: BTreeSet<_> = order.iter().collect();
    r.check(
        unique.len() == order.len(),
        "publish-crates.txt has no duplicates",
    );
    let missing: Vec<_> = order
        .iter()
        .filter(|n| !manifests.contains_key(*n))
        .collect();
    r.check(
        missing.is_empty(),
        format!("publish-crates.txt names only crates in crates/ (unknown: {missing:?})"),
    );
    let deps: BTreeMap<_, _> = order
        .iter()
        .filter_map(|n| manifests.get(n).map(|m| (n.clone(), internal_deps(m))))
        .collect();
    let bad = order_violations(&order, &deps);
    r.check(
        bad.is_empty(),
        format!("publish-crates.txt is in dependency order (violations: {bad:?})"),
    );

    let changelog = read("CHANGELOG.md").unwrap_or_default();
    let fragments = std::fs::read_dir(root.join("changelog.d")).map_or(0, |d| {
        d.filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "md"))
            .count()
    });
    r.check(
        has_section(&changelog, &version) || fragments > 0,
        format!(
            "CHANGELOG.md has `## {version}` or changelog.d has fragments to assemble ({fragments} pending)"
        ),
    );

    let tag = format!("v{version}");
    let taken = Command::new("git")
        .args(["tag", "--list", &tag])
        .current_dir(root)
        .output()
        .is_ok_and(|o| !String::from_utf8_lossy(&o.stdout).trim().is_empty());
    r.check(!taken, format!("tag {tag} is not taken locally"));

    for l in &r.lines {
        println!("{l}");
    }
    if r.failed == 0 {
        println!("release-check: {version} is ready to hand to the owner");
        Ok(())
    } else {
        Err(format!("release-check: {} check(s) failed", r.failed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions() {
        assert!(is_version("0.3.0"));
        assert!(is_version("0.3.0-rc.1"));
        assert!(!is_version("0.3"));
        assert!(!is_version("v0.3.0"));
        assert!(!is_version("0.3.x"));
        assert!(!is_version("0.3.0-"));
    }

    #[test]
    fn workspace_version_is_read() {
        let m = "[workspace]\nmembers = []\n[workspace.package]\nversion = \"0.3.0\"\n";
        assert_eq!(workspace_version(m).unwrap_or_default(), "0.3.0");
        assert!(workspace_version("[workspace]\n").is_err());
    }

    #[test]
    fn inheriting_the_version() {
        assert!(inherits_version(
            "[package]\nname = \"a\"\nversion.workspace = true\n"
        ));
        assert!(!inherits_version(
            "[package]\nname = \"a\"\nversion = \"1.0.0\"\n"
        ));
    }

    #[test]
    fn internal_dependencies_skip_dev() {
        let m = "[dependencies]\nkrishiv-a = { workspace = true }\nserde = \"1\"\n[dev-dependencies]\nkrishiv-b = { workspace = true }\n[build-dependencies]\nkrishiv-c = { workspace = true }\n";
        let d: Vec<_> = internal_deps(m).into_iter().collect();
        assert_eq!(d, ["krishiv-a", "krishiv-c"]);
    }

    #[test]
    fn dependency_order() {
        let order = vec!["a".to_owned(), "b".to_owned()];
        let mut deps = BTreeMap::new();
        deps.insert("a".to_owned(), BTreeSet::from(["b".to_owned()]));
        assert_eq!(
            order_violations(&order, &deps),
            [("a".to_owned(), "b".to_owned())]
        );
        deps.insert("a".to_owned(), BTreeSet::new());
        deps.insert("b".to_owned(), BTreeSet::from(["a".to_owned()]));
        assert!(order_violations(&order, &deps).is_empty());
        deps.insert("b".to_owned(), BTreeSet::from(["z".to_owned()]));
        assert_eq!(order_violations(&order, &deps).len(), 1);
    }

    #[test]
    fn changelog_section() {
        assert!(has_section("# C\n\n## 0.3.0\n\nx\n", "0.3.0"));
        assert!(!has_section("## 0.3.0-rc.1\n", "0.3.0"));
    }
    fn fixture(version: &str, list: &str, b_dep: &str) -> tempfile::TempDir {
        let d = tempfile::tempdir().expect("tempdir");
        let w = |p: &str, t: &str| {
            let f = d.path().join(p);
            std::fs::create_dir_all(f.parent().expect("parent")).expect("mkdir");
            std::fs::write(f, t).expect("write");
        };
        w(
            "Cargo.toml",
            &format!("[workspace.package]\nversion = \"{version}\"\n"),
        );
        for f in REQUIRED_FILES {
            w(f, "");
        }
        w(".github/scripts/publish-crates.txt", list);
        w("CHANGELOG.md", "## 0.3.0\n");
        std::fs::create_dir_all(d.path().join("changelog.d")).expect("mkdir");
        let inherit = "[package]\nname = \"x\"\nversion.workspace = true\n";
        w("crates/krishiv-a/Cargo.toml", inherit);
        w(
            "crates/krishiv-b/Cargo.toml",
            &format!("{inherit}[dependencies]\n{b_dep}\n"),
        );
        d
    }

    #[test]
    fn a_consistent_tree_passes() {
        let d = fixture(
            "0.3.0",
            "# order\nkrishiv-a\nkrishiv-b\n",
            "krishiv-a = \"1\"",
        );
        run(d.path(), &[]).expect("passes");
        run(d.path(), &["0.3.0"]).expect("passes with the version given");
    }

    #[test]
    fn a_broken_tree_fails() {
        let d = fixture(
            "0.3.0",
            "krishiv-b\nkrishiv-a\nkrishiv-a\nkrishiv-zz\n",
            "krishiv-a = \"1\"",
        );
        let err = run(d.path(), &["0.4.0"]).expect_err("fails");
        assert!(err.contains("check(s) failed"), "{err}");
        assert!(run(d.path(), &["1", "2"]).is_err());
    }

    #[test]
    fn a_missing_manifest_is_an_error() {
        let d = tempfile::tempdir().expect("tempdir");
        assert!(run(d.path(), &[]).is_err());
    }
}
