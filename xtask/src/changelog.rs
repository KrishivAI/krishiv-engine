// SPDX-License-Identifier: Apache-2.0
//! Changelog fragments: one `changelog.d/<pr>.<type>.md` per pull request, checked in CI and assembled at release.
#![expect(clippy::print_stdout, reason = "command output is the interface")]

use std::path::Path;

use crate::policy::DiffArgs;

/// Fragment types in the order their sections appear in a release, with the section heading.
pub const SECTIONS: [(&str, &str); 8] = [
    ("breaking", "Breaking changes"),
    ("added", "Added"),
    ("performance", "Performance"),
    ("compat", "Spark compatibility"),
    ("fixed", "Fixed"),
    ("deprecated", "Deprecated"),
    ("removed", "Removed"),
    ("security", "Security"),
];

/// The label that exempts a CI-only or docs-only pull request from needing a fragment.
pub const EXEMPT_LABEL: &str = "no-changelog";

/// One fragment: the pull request number, its type and the user-facing text.
#[derive(Debug, PartialEq, Eq)]
pub struct Fragment {
    /// Pull request number from the file name.
    pub pr: u64,
    /// One of the type names in [`SECTIONS`].
    pub kind: String,
    /// The user-facing lines, trimmed.
    pub text: String,
}

/// Parses `<pr>.<type>.md`; `None` when the name is not a fragment name.
#[must_use]
pub fn parse_name(file_name: &str) -> Option<(u64, &str)> {
    let stem = file_name.strip_suffix(".md")?;
    let (pr, kind) = stem.split_once('.')?;
    let pr = pr.parse().ok()?;
    SECTIONS
        .iter()
        .any(|(k, _)| *k == kind)
        .then_some((pr, kind))
}

/// Whether a repository path is a fragment file.
#[must_use]
pub fn is_fragment_path(path: &str) -> bool {
    path.strip_prefix("changelog.d/")
        .is_some_and(|name| !name.contains('/') && parse_name(name).is_some())
}

/// The CI check over the files a pull request adds: a fragment, or the exempt label.
///
/// # Errors
/// Returns the instruction to add a fragment or the label.
pub fn check(added: &[&str], labels: &[&str]) -> Result<(), String> {
    if added.iter().any(|p| is_fragment_path(p)) || labels.contains(&EXEMPT_LABEL) {
        return Ok(());
    }
    Err(format!(
        "no changelog fragment: add changelog.d/<pr>.<type>.md (type: {}) with one or two lines a user can read, \
         or label the PR `{EXEMPT_LABEL}` if it is CI or docs only (CONTRIBUTING.md, \"Changelog\")",
        SECTIONS.map(|(k, _)| k).join(", ")
    ))
}

/// Renders fragments as the sections of a release, in the fixed order; empty sections are left out.
#[must_use]
pub fn render_sections(fragments: &[Fragment]) -> String {
    let mut out = String::new();
    for (kind, heading) in SECTIONS {
        let mut of_kind: Vec<&Fragment> = fragments.iter().filter(|f| f.kind == kind).collect();
        if of_kind.is_empty() {
            continue;
        }
        of_kind.sort_by_key(|f| f.pr);
        out.push_str(&format!("### {heading}\n\n"));
        for f in of_kind {
            let mut lines = f.text.lines();
            if let Some(first) = lines.next() {
                out.push_str(&format!("- {first} (#{})\n", f.pr));
            }
            for line in lines {
                out.push_str(&format!("  {}\n", line.trim_end()));
            }
        }
        out.push('\n');
    }
    out
}

/// Reads every fragment under `dir`, sorted by pull request number.
///
/// # Errors
/// Returns an unreadable or empty fragment.
pub fn read_fragments(dir: &Path) -> Result<Vec<Fragment>, String> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(out);
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == "README.md" {
            continue;
        }
        let (pr, kind) = parse_name(&name).ok_or_else(|| {
            format!(
                "{}: not a fragment name (<pr>.<type>.md)",
                entry.path().display()
            )
        })?;
        let text = std::fs::read_to_string(entry.path())
            .map_err(|e| format!("{}: {e}", entry.path().display()))?;
        let text = text.trim().to_owned();
        if text.is_empty() {
            return Err(format!("{}: empty fragment", entry.path().display()));
        }
        out.push(Fragment {
            pr,
            kind: kind.to_owned(),
            text,
        });
    }
    out.sort_by_key(|f| f.pr);
    Ok(out)
}

/// Puts `section` (a whole `## <version>` block) under `## Unreleased` and resets Unreleased to the pending marker.
///
/// # Errors
/// Returns an error when the changelog has no `## Unreleased` heading.
pub fn insert_release(changelog: &str, section: &str) -> Result<String, String> {
    let marker = "## Unreleased\n";
    let at = changelog
        .find(marker)
        .ok_or("CHANGELOG.md has no `## Unreleased` heading")?;
    let after = at + marker.len();
    let rest = &changelog[after..];
    let next = rest.find("\n## ").map_or(rest.len(), |i| i + 1);
    Ok(format!(
        "{}{marker}\nNothing yet.\n\n{section}{}",
        &changelog[..at],
        &rest[next..]
    ))
}

/// `xtask changelog check --base REF [--head REF] [--labels a,b]`.
///
/// # Errors
/// Returns the failed check or a usage error.
pub fn run_check(args: &DiffArgs, labels: &[&str]) -> Result<(), String> {
    let added = args.git_added_files()?;
    let added: Vec<&str> = added.iter().map(String::as_str).collect();
    check(&added, labels)?;
    println!("changelog: OK");
    Ok(())
}

/// `xtask changelog pending [DIR]`: prints what the next release would list.
///
/// # Errors
/// Returns an unreadable fragment.
pub fn run_pending(root: &Path) -> Result<(), String> {
    let fragments = read_fragments(&root.join("changelog.d"))?;
    if fragments.is_empty() {
        println!("no pending fragments");
    } else {
        print!("{}", render_sections(&fragments));
    }
    Ok(())
}

/// `xtask changelog assemble VERSION [--highlights FILE]`: writes the release section into `CHANGELOG.md` and
/// deletes the fragments.
///
/// # Errors
/// Returns an unreadable file, a missing `## Unreleased` heading, or no fragments.
pub fn run_assemble(root: &Path, version: &str, highlights: Option<&Path>) -> Result<(), String> {
    let dir = root.join("changelog.d");
    let fragments = read_fragments(&dir)?;
    if fragments.is_empty() {
        return Err("no fragments in changelog.d".to_owned());
    }
    let mut section = format!("## {version}\n\n");
    if let Some(path) = highlights {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        section.push_str(&format!("### Highlights\n\n{}\n\n", text.trim()));
    }
    section.push_str(&render_sections(&fragments));
    let path = root.join("CHANGELOG.md");
    let old = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let new = insert_release(&old, &section)?;
    std::fs::write(&path, new).map_err(|e| format!("{}: {e}", path.display()))?;
    for entry in std::fs::read_dir(&dir)
        .map_err(|e| e.to_string())?
        .flatten()
    {
        if entry.file_name() != "README.md" {
            std::fs::remove_file(entry.path()).map_err(|e| e.to_string())?;
        }
    }
    println!(
        "changelog: {} fragments assembled into {version}",
        fragments.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frag(pr: u64, kind: &str, text: &str) -> Fragment {
        Fragment {
            pr,
            kind: kind.to_owned(),
            text: text.to_owned(),
        }
    }

    #[test]
    fn names_parse() {
        assert_eq!(parse_name("703.added.md"), Some((703, "added")));
        assert_eq!(parse_name("703.bogus.md"), None);
        assert_eq!(parse_name("README.md"), None);
        assert_eq!(parse_name("x.added.md"), None);
    }

    #[test]
    fn check_needs_fragment_or_label() {
        assert!(check(&["changelog.d/1.fixed.md"], &[]).is_ok());
        assert!(check(&["src/a.rs"], &["no-changelog"]).is_ok());
        assert!(check(&["src/a.rs", "changelog.d/README.md"], &[]).is_err());
        assert!(check(&["changelog.d/sub/1.fixed.md"], &[]).is_err());
    }

    #[test]
    fn sections_follow_the_fixed_order() {
        let out = render_sections(&[
            frag(9, "fixed", "A crash is gone."),
            frag(3, "breaking", "`x` is renamed.\nUse `y`."),
            frag(5, "added", "New thing."),
        ]);
        let b = out.find("### Breaking").unwrap();
        let a = out.find("### Added").unwrap();
        let f = out.find("### Fixed").unwrap();
        assert!(b < a && a < f);
        assert!(out.contains("- `x` is renamed. (#3)\n  Use `y`.\n"));
        assert!(!out.contains("### Security"));
    }

    #[test]
    fn release_goes_under_unreleased_and_resets_it() {
        let log = "# Changelog\n\n## Unreleased\n\nNothing yet.\n\n## 0.2.0\n\nold\n";
        let out = insert_release(log, "## 0.3.0\n\n### Added\n\n- x (#1)\n\n").unwrap();
        assert_eq!(
            out,
            "# Changelog\n\n## Unreleased\n\nNothing yet.\n\n## 0.3.0\n\n### Added\n\n- x (#1)\n\n## 0.2.0\n\nold\n"
        );
        assert!(insert_release("# no heading", "x").is_err());
    }
}
