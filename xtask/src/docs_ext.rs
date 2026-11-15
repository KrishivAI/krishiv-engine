// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask docs-python [--check]` and `cargo xtask docs-rust [--check]`: the Python and Rust
//! API references, generated into `docs/reference/python/` and `docs/reference/rust/`.
//!
//! Python: `tests/scripts/gen_python_docs.py` (standard library `ast`; nothing imported).
//! Rust: rustdoc JSON (nightly rustdoc) converted to Markdown by `rustdoc-md` (a tool installed with
//! `cargo install rustdoc-md --version 0.2.0 --locked`, not a workspace dependency; ADR 0028).

use std::path::{Path, PathBuf};
use std::process::Command;

/// The nightly that produces rustdoc JSON; its format version must be one `rustdoc-md` reads.
pub const RUSTDOC_NIGHTLY: &str = "nightly-2026-08-12";

/// The crates whose public API is documented on the site: the facade, the session and the UDF SDK (the connector SDK is an empty skeleton; `krishiv-common` is mostly
/// registry constants, covered by the configuration reference).
pub const RUST_CRATES: &[&str] = &[
    "krishiv",
    "krishiv-session",
    "krishiv-udf-sdk",
    "krishiv-types",
];

/// Runs the Python generator (`--check` compares instead of writing).
///
/// # Errors
/// When Python is missing or the generator fails.
pub fn run_python(root: &Path, check: bool) -> Result<(), String> {
    let mut cmd = Command::new("python3");
    cmd.current_dir(root)
        .args(["-I", "tests/scripts/gen_python_docs.py"]);
    if check {
        cmd.arg("--check");
    }
    match cmd.status() {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(format!("gen_python_docs.py failed: {s}")),
        Err(e) => Err(format!("cannot run python3: {e}")),
    }
}

fn markdown_of(json: &Path, out: &Path) -> Result<String, String> {
    let status = Command::new("rustdoc-md")
        .arg("--path")
        .arg(json)
        .arg("--output")
        .arg(out)
        .status()
        .map_err(|e| format!("cannot run rustdoc-md ({e}); install it: cargo install rustdoc-md --version 0.2.0 --locked"))?;
    if !status.success() {
        return Err(format!("rustdoc-md failed on {}: {status}", json.display()));
    }
    let text = std::fs::read_to_string(out).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(tidy(&text))
}

/// Rewrites a rustdoc intra-doc link whose target is an item path (`[`name`](ScalarUdf::name)`) to
/// its label: the converter copies the path through as if it were a file, and the link checker of
/// the docs (`tests/scripts/check_docs_links.py`) rightly rejects it.
fn drop_item_links(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(open) = rest.find("](") {
        let (head, tail) = rest.split_at(open);
        let after = &tail[2..];
        let item = after.find(')').and_then(|end| {
            let target = &after[..end];
            let is_item = target.contains("::")
                && target
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == ':');
            head.rfind('[')
                .filter(|_| is_item)
                .map(|open_label| (open_label, end))
        });
        if let Some((open_label, end)) = item {
            out.push_str(&head[..open_label]);
            out.push_str(&head[open_label + 1..]);
            rest = &after[end + 1..];
        } else {
            out.push_str(head);
            out.push_str("](");
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// Drops the converter's version banner (the format version changes with the nightly, not the API).
fn tidy(text: &str) -> String {
    let mut lines = text.lines().peekable();
    let mut out = Vec::new();
    while let Some(l) = lines.next() {
        if l.starts_with("**Format Version:**") || l == "# Crate Documentation" {
            while lines.peek().is_some_and(|n| n.is_empty()) {
                lines.next();
            }
            continue;
        }
        out.push(drop_item_links(l));
    }
    format!("{}\n", out.join("\n").trim())
}

/// Builds rustdoc JSON for [`RUST_CRATES`] and writes (or with `check` compares) the Markdown pages.
///
/// # Errors
/// When the nightly or `rustdoc-md` is missing, a build fails, or with `check` a page is stale.
pub fn run_rust(root: &Path, check: bool) -> Result<(), String> {
    let target = root.join("target/rustdoc-json");
    let mut cmd = Command::new("cargo");
    cmd.current_dir(root)
        .arg(format!("+{RUSTDOC_NIGHTLY}"))
        .args(["doc", "--no-deps", "--locked"])
        .env("CARGO_TARGET_DIR", &target)
        .env("RUSTDOCFLAGS", "-Zunstable-options --output-format json");
    for c in RUST_CRATES {
        cmd.args(["-p", c]);
    }
    let status = cmd.status().map_err(|e| format!("cannot run cargo: {e}"))?;
    if !status.success() {
        return Err(format!("cargo doc ({RUSTDOC_NIGHTLY}) failed: {status}"));
    }
    let out_dir = root.join("docs/reference/rust");
    let tmp = target.join("md");
    std::fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
    let mut mds: Vec<(&str, String)> = Vec::new();
    for c in RUST_CRATES {
        let stem = c.replace('-', "_");
        mds.push((
            c,
            markdown_of(
                &target.join(format!("doc/{stem}.json")),
                &tmp.join(format!("{c}.md")),
            )?,
        ));
    }
    write_or_check(&out_dir, &rust_pages(&out_dir, &mds), check)
}

/// The crate pages plus the index and `_meta.json`, as `(path, text)`.
fn rust_pages(out_dir: &Path, mds: &[(&str, String)]) -> Vec<(PathBuf, String)> {
    let mut pages: Vec<(PathBuf, String)> = Vec::new();
    let mut index = String::from(
        "# Rust API reference\n\n<!-- @generated by `cargo xtask docs-rust`; do not edit. -->\n\n\
         The public API of the crates users build against. Published versions are also on docs.rs.\n\n\
         | Crate | Summary |\n|---|---|\n",
    );
    for (c, md) in mds {
        let summary = md
            .lines()
            .skip_while(|l| !l.starts_with("# Module"))
            .skip(1)
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .replace('|', "\\|");
        index.push_str(&format!("| [`{c}`]({c}.md) | {summary} |\n"));
        pages.push((out_dir.join(format!("{c}.md")), md.clone()));
    }
    pages.push((out_dir.join("index.md"), index));
    pages.push((
        out_dir.join("_meta.json"),
        "{\n  \"index\": \"Rust API\"\n}\n".to_owned(),
    ));
    pages
}

/// Writes the pages, or with `check` compares them with the files on disk.
fn write_or_check(out_dir: &Path, pages: &[(PathBuf, String)], check: bool) -> Result<(), String> {
    if check {
        let bad: Vec<String> = pages
            .iter()
            .filter(|(p, t)| std::fs::read_to_string(p).ok().as_deref() != Some(t.as_str()))
            .map(|(p, _)| p.display().to_string())
            .collect();
        return if bad.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "docs/reference/rust is stale (first: {}); run `cargo xtask docs-rust`",
                bad[0]
            ))
        };
    }
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    for (p, t) in pages {
        std::fs::write(p, t).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_python_and_rust_report_a_missing_root() {
        let nowhere = Path::new("/no/such/root");
        assert!(run_python(nowhere, true).unwrap_err().contains("python3"));
        assert!(run_rust(nowhere, true).is_err());
    }

    #[test]
    fn tidy_drops_the_banner_and_the_blank_lines_after_it() {
        let t = tidy("# Crate Documentation\n\n**Format Version:** 57\n\n# Module `x`\n\nBody\n");
        assert_eq!(t, "# Module `x`\n\nBody\n");
    }

    #[test]
    fn item_path_links_become_their_label_and_file_links_stay() {
        assert_eq!(
            drop_item_links(
                "by [`name`](ScalarUdf::name) and [`v`](ScalarUdf::version) only, see [a](b.md#c) [x](Y::z"
            ),
            "by `name` and `v` only, see [a](b.md#c) [x](Y::z"
        );
        // A link whose `)` is on the next line, or an item link with no label bracket, is left alone.
        assert_eq!(drop_item_links("see [`a`](B::c"), "see [`a`](B::c");
        assert_eq!(drop_item_links("](A::b) and é[é](É::é)"), "](A::b) and éé");
    }

    #[test]
    fn the_index_lists_each_crate_with_its_first_module_line() {
        let out = Path::new("o");
        let pages = rust_pages(
            out,
            &[(
                "krishiv",
                "# Module `krishiv`\n\nThe | facade.\n".to_owned(),
            )],
        );
        let index = &pages[1].1;
        assert!(
            index.contains("| [`krishiv`](krishiv.md) | The \\| facade. |"),
            "{index}"
        );
        assert_eq!(pages.len(), 3);
        assert!(pages[2].0.ends_with("_meta.json"));
    }

    #[test]
    fn write_then_check_then_stale() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| panic!("{e}"));
        let out = dir.path().join("rust");
        let pages = || vec![(out.join("a.md"), "A\n".to_owned())];
        assert!(write_or_check(&out, &pages(), true).is_err());
        assert!(write_or_check(&out, &pages(), false).is_ok());
        assert!(write_or_check(&out, &pages(), true).is_ok());
        std::fs::write(out.join("a.md"), "B\n").unwrap_or_default();
        let err = write_or_check(&out, &pages(), true)
            .err()
            .unwrap_or_default();
        assert!(err.contains("stale"), "{err}");
    }
}
