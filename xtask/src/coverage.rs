// SPDX-License-Identifier: Apache-2.0
//! Coverage tooling (P2-T105): `coverage`, `coverage-summary`, `diff-coverage`.
//!
//! Report only: nothing here gates a merge. The diff-coverage ratchet is
//! switched on at P2a exit (P2-T105b). Codecov is informational as well.
#![expect(clippy::print_stdout, reason = "command output is the interface")]

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Covered/total counts for one metric.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    /// Items hit at least once.
    pub covered: u64,
    /// Instrumented items.
    pub total: u64,
}

impl Counts {
    /// Percentage, or `None` when nothing is instrumented.
    #[must_use]
    #[expect(clippy::cast_precision_loss, reason = "report percentage")]
    pub fn percent(self) -> Option<f64> {
        (self.total > 0).then(|| self.covered as f64 * 100.0 / self.total as f64)
    }

    fn add(&mut self, other: Self) {
        self.covered += other.covered;
        self.total += other.total;
    }

    fn show(self) -> String {
        self.percent()
            .map_or_else(|| "-".to_owned(), |p| format!("{p:.1}%"))
    }
}

/// Line and branch coverage of one crate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CrateCov {
    /// Line coverage.
    pub lines: Counts,
    /// Branch coverage (zero total without `--branch`).
    pub branches: Counts,
}

/// Crate a source path belongs to: `crates/<name>/...` or `xtask/...`.
#[must_use]
pub fn crate_of(path: &str) -> Option<String> {
    let mut parts = path
        .split('/')
        .skip_while(|p| *p != "crates" && *p != "xtask");
    match parts.next()? {
        "xtask" => Some("xtask".to_owned()),
        _ => parts.next().map(str::to_owned),
    }
}

/// Per-crate coverage from `cargo llvm-cov report --json --summary-only`.
///
/// # Errors
/// Returns an error when the JSON is not an llvm-cov export.
pub fn summarize(json: &str) -> Result<BTreeMap<String, CrateCov>, String> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("bad coverage JSON: {e}"))?;
    let files = v
        .pointer("/data/0/files")
        .and_then(serde_json::Value::as_array)
        .ok_or("coverage JSON has no data[0].files")?;
    let count = |s: &serde_json::Value, k: &str| Counts {
        covered: s
            .pointer(&format!("/{k}/covered"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
        total: s
            .pointer(&format!("/{k}/count"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0),
    };
    let mut out: BTreeMap<String, CrateCov> = BTreeMap::new();
    for f in files {
        let name = f
            .get("filename")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let (Some(krate), Some(s)) = (crate_of(name), f.get("summary")) else {
            continue;
        };
        let e = out.entry(krate).or_default();
        e.lines.add(count(s, "lines"));
        e.branches.add(count(s, "branches"));
    }
    Ok(out)
}

/// Renders the per-crate table (plain text, or Markdown with `markdown`).
#[must_use]
pub fn render_summary(map: &BTreeMap<String, CrateCov>, markdown: bool) -> String {
    let mut total = CrateCov::default();
    let mut rows = Vec::new();
    for (name, c) in map {
        total.lines.add(c.lines);
        total.branches.add(c.branches);
        rows.push((name.as_str(), *c));
    }
    rows.push(("TOTAL", total));
    let mut out = String::new();
    if markdown {
        out.push_str("| crate | line % | branch % |\n|---|---:|---:|\n");
    } else {
        out.push_str(&format!(
            "{:<32} {:>8} {:>8}\n",
            "crate", "line %", "branch %"
        ));
    }
    for (name, c) in rows {
        if markdown {
            out.push_str(&format!(
                "| {name} | {} | {} |\n",
                c.lines.show(),
                c.branches.show()
            ));
        } else {
            out.push_str(&format!(
                "{name:<32} {:>8} {:>8}\n",
                c.lines.show(),
                c.branches.show()
            ));
        }
    }
    out
}

/// Executable lines and hit counts per file from an lcov tracefile.
#[must_use]
pub fn parse_lcov(text: &str) -> HashMap<String, BTreeMap<u32, u64>> {
    let mut out: HashMap<String, BTreeMap<u32, u64>> = HashMap::new();
    let mut cur: Option<String> = None;
    for line in text.lines() {
        if let Some(f) = line.strip_prefix("SF:") {
            cur = Some(f.to_owned());
        } else if let (Some(da), Some(file)) = (line.strip_prefix("DA:"), &cur) {
            let mut it = da.split(',');
            if let (Some(Ok(n)), Some(Ok(h))) = (
                it.next().map(str::parse::<u32>),
                it.next().map(str::parse::<u64>),
            ) {
                *out.entry(file.clone()).or_default().entry(n).or_default() += h;
            }
        }
    }
    out
}

/// Added line numbers per file from `git diff -U0` output.
#[must_use]
pub fn parse_added_lines(diff: &str) -> BTreeMap<String, Vec<u32>> {
    let mut out: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    let mut cur: Option<String> = None;
    for line in diff.lines() {
        if let Some(p) = line.strip_prefix("+++ ") {
            cur = p.strip_prefix("b/").map(str::to_owned);
        } else if let (Some(h), Some(file)) = (line.strip_prefix("@@ "), &cur) {
            let new = h
                .split_whitespace()
                .find(|t| t.starts_with('+'))
                .unwrap_or("+0");
            let mut it = new.trim_start_matches('+').split(',');
            let start: u32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            let len: u32 = it.next().and_then(|s| s.parse().ok()).unwrap_or(1);
            out.entry(file.clone())
                .or_default()
                .extend(start..start + len);
        }
    }
    out
}

/// Diff-coverage result.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct DiffCov {
    /// Added executable lines that ran.
    pub covered: u64,
    /// Added executable lines.
    pub total: u64,
    /// Uncovered added lines per file.
    pub uncovered: BTreeMap<String, Vec<u32>>,
}

/// Coverage of the added lines. Only lines the tracefile instruments count;
/// `root` is stripped from absolute `SF:` paths.
#[must_use]
pub fn diff_coverage(
    added: &BTreeMap<String, Vec<u32>>,
    lcov: &HashMap<String, BTreeMap<u32, u64>>,
    root: &str,
) -> DiffCov {
    let mut res = DiffCov::default();
    let by_rel: HashMap<&str, &BTreeMap<u32, u64>> = lcov
        .iter()
        .map(|(k, v)| {
            (
                k.strip_prefix(root)
                    .map_or(k.as_str(), |r| r.trim_start_matches('/')),
                v,
            )
        })
        .collect();
    for (file, lines) in added {
        let Some(hits) = by_rel.get(file.as_str()) else {
            continue;
        };
        for n in lines {
            match hits.get(n) {
                Some(0) => {
                    res.total += 1;
                    res.uncovered.entry(file.clone()).or_default().push(*n);
                }
                Some(_) => {
                    res.total += 1;
                    res.covered += 1;
                }
                None => {}
            }
        }
    }
    res
}

/// Renders a diff-coverage report as Markdown.
#[must_use]
pub fn render_diff(d: &DiffCov) -> String {
    let c = Counts {
        covered: d.covered,
        total: d.total,
    };
    let mut out = format!(
        "### Diff coverage (report only)\n\n{} of {} added executable lines covered ({}).\n",
        d.covered,
        d.total,
        c.show()
    );
    for (f, lines) in &d.uncovered {
        let list: Vec<String> = lines.iter().map(u32::to_string).collect();
        out.push_str(&format!("\n- `{f}`: uncovered lines {}", list.join(", ")));
    }
    out.push('\n');
    out
}

fn coverage_dir(root: &Path) -> PathBuf {
    std::env::var_os("CARGO_TARGET_DIR")
        .map_or_else(|| root.join("target"), PathBuf::from)
        .join("coverage")
}

fn cargo() -> Command {
    Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
}

fn run(mut cmd: Command, what: &str) -> Result<(), String> {
    let status = cmd
        .status()
        .map_err(|e| format!("failed to run {what}: {e}"))?;
    status
        .success()
        .then_some(())
        .ok_or_else(|| format!("{what} failed"))
}

/// `xtask coverage [--no-branch] [--all-features] [--html]`: runs the suite
/// under `cargo-llvm-cov`, writing `lcov.info` and `summary.json` (and an HTML
/// report) under `<target>/coverage`. Branch coverage needs the nightly
/// toolchain with `llvm-tools-preview`.
///
/// # Errors
/// Returns an error for unknown flags or when a cargo step fails.
pub fn run_coverage(root: &Path, flags: &[&str]) -> Result<(), String> {
    let (mut branch, mut all, mut html) = (true, false, false);
    for f in flags {
        match *f {
            "--no-branch" => branch = false,
            "--all-features" => all = true,
            "--html" => html = true,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let dir = coverage_dir(root);
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let base = |sub: &[&str]| {
        // `$CARGO` is the pinned toolchain's own binary and would ignore
        // RUSTUP_TOOLCHAIN, so branch mode goes through the rustup proxy.
        let mut c = if branch {
            let mut c = Command::new("cargo");
            c.env("RUSTUP_TOOLCHAIN", "nightly");
            c
        } else {
            cargo()
        };
        c.args(["llvm-cov"]).args(sub).current_dir(root);
        c
    };
    let mut test = base(&[
        "nextest",
        "--workspace",
        "--locked",
        "--no-tests=pass",
        "--no-report",
    ]);
    if all {
        test.arg("--all-features");
    }
    if branch {
        test.arg("--branch");
    }
    run(test, "cargo llvm-cov nextest")?;
    let report = |extra: &[&str], out: &Path| {
        let mut c = base(&["report"]);
        if branch {
            c.arg("--branch");
        }
        c.args(extra).arg(out);
        c
    };
    run(
        report(&["--lcov", "--output-path"], &dir.join("lcov.info")),
        "cargo llvm-cov report --lcov",
    )?;
    run(
        report(
            &["--json", "--summary-only", "--output-path"],
            &dir.join("summary.json"),
        ),
        "cargo llvm-cov report --json",
    )?;
    if html {
        run(
            report(&["--html", "--output-dir"], &dir.join("html")),
            "cargo llvm-cov report --html",
        )?;
    }
    println!("coverage: wrote {}", dir.display());
    run_summary(root, &[])
}

/// `xtask coverage-summary [--markdown] [--json] [PATH]`: per-crate line and
/// branch percentages from `summary.json`.
///
/// # Errors
/// Returns an error when the summary cannot be read or parsed.
pub fn run_summary(root: &Path, flags: &[&str]) -> Result<(), String> {
    let (mut markdown, mut json, mut path) =
        (false, false, coverage_dir(root).join("summary.json"));
    for f in flags {
        match *f {
            "--markdown" => markdown = true,
            "--json" => json = true,
            p if !p.starts_with("--") => path = PathBuf::from(p),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let text = std::fs::read_to_string(&path).map_err(|e| {
        format!(
            "cannot read {} (run `cargo xtask coverage` first): {e}",
            path.display()
        )
    })?;
    let map = summarize(&text)?;
    if json {
        let obj: serde_json::Map<String, serde_json::Value> = map
            .iter()
            .map(|(k, c)| {
                (k.clone(), serde_json::json!({ "line_pct": c.lines.percent(), "branch_pct": c.branches.percent() }))
            })
            .collect();
        println!("{}", serde_json::Value::Object(obj));
    } else {
        print!("{}", render_summary(&map, markdown));
    }
    Ok(())
}

/// `xtask diff-coverage --base REF [--head REF] [--lcov PATH]`: report-only
/// coverage of the lines a PR adds; appended to the step summary in CI.
///
/// # Errors
/// Returns an error for bad flags, a missing tracefile or a git failure.
pub fn run_diff_coverage(root: &Path, flags: &[&str]) -> Result<(), String> {
    let mut lcov = coverage_dir(root).join("lcov.info");
    let mut rest = Vec::new();
    let mut it = flags.iter();
    while let Some(f) = it.next() {
        if *f == "--lcov" {
            lcov = PathBuf::from(it.next().ok_or("--lcov needs a value")?);
        } else {
            rest.push(*f);
        }
    }
    let args = crate::policy::DiffArgs::parse(&rest, root)?;
    let text = std::fs::read_to_string(&lcov)
        .map_err(|e| format!("cannot read {}: {e}", lcov.display()))?;
    let diff = args.unified_zero_diff()?;
    let d = diff_coverage(
        &parse_added_lines(&diff),
        &parse_lcov(&text),
        &root.display().to_string(),
    );
    let md = render_diff(&d);
    print!("{md}");
    if let Some(path) = std::env::var_os("GITHUB_STEP_SUMMARY") {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
        {
            let _ = writeln!(f, "{md}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXPORT: &str = r#"{"data":[{"files":[
      {"filename":"/w/crates/a/src/lib.rs","summary":{"lines":{"count":10,"covered":9},"branches":{"count":4,"covered":2}}},
      {"filename":"/w/crates/a/src/x.rs","summary":{"lines":{"count":10,"covered":1},"branches":{"count":0,"covered":0}}},
      {"filename":"/w/xtask/src/main.rs","summary":{"lines":{"count":4,"covered":4},"branches":{"count":0,"covered":0}}},
      {"filename":"/w/build.rs","summary":{"lines":{"count":9,"covered":9},"branches":{"count":0,"covered":0}}}
    ]}]}"#;

    #[test]
    fn crates_are_named_from_paths() {
        assert_eq!(
            crate_of("/w/crates/krishiv-sql/src/a.rs").as_deref(),
            Some("krishiv-sql")
        );
        assert_eq!(crate_of("/w/xtask/src/a.rs").as_deref(), Some("xtask"));
        assert_eq!(crate_of("/w/build.rs"), None);
    }

    #[test]
    fn summary_aggregates_per_crate() {
        let m = summarize(EXPORT).unwrap_or_default();
        assert_eq!(m.len(), 2);
        let a = m["a"];
        assert_eq!(
            a.lines,
            Counts {
                covered: 10,
                total: 20
            }
        );
        assert_eq!(a.branches.percent(), Some(50.0));
        assert_eq!(m["xtask"].branches.percent(), None);
        assert!(summarize("{}").is_err());
        let text = render_summary(&m, false);
        assert!(text.contains("a ") && text.contains("50.0%") && text.contains("TOTAL"));
        assert!(render_summary(&m, true).contains("| xtask | 100.0% | - |"));
    }

    #[test]
    fn diff_coverage_counts_only_instrumented_added_lines() {
        let lcov = parse_lcov(
            "SF:/w/crates/a/src/lib.rs\nDA:1,3\nDA:2,0\nDA:3,0\nDA:5,1\nend_of_record\n",
        );
        let diff = "+++ b/crates/a/src/lib.rs\n@@ -1,0 +2,2 @@\n+x\n+y\n@@ -9 +5 @@\n-a\n+b\n+++ b/README.md\n@@ -1 +1 @@\n";
        let added = parse_added_lines(diff);
        assert_eq!(added["crates/a/src/lib.rs"], vec![2, 3, 5]);
        let d = diff_coverage(&added, &lcov, "/w");
        assert_eq!((d.covered, d.total), (1, 3));
        assert_eq!(d.uncovered["crates/a/src/lib.rs"], vec![2, 3]);
        let md = render_diff(&d);
        assert!(
            md.contains("1 of 3") && md.contains("33.3%") && md.contains("uncovered lines 2, 3")
        );
    }

    #[test]
    fn empty_diff_is_not_a_division_by_zero() {
        assert!(
            render_diff(&DiffCov::default()).contains("0 of 0 added executable lines covered (-)")
        );
    }
}
