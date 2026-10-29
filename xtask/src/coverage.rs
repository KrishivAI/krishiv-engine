// SPDX-License-Identifier: Apache-2.0
//! Coverage tooling (P2-T105): `coverage`, `coverage-summary`, `diff-coverage`.
//!
//! `coverage` and `coverage-summary` only report; `diff-coverage --fail-under`
//! is the merge gate (Rust, and Python through coverage.py lcov tracefiles).
//! Codecov is informational.
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

/// Hit counts of the executable lines of each file, keyed by path.
pub type LineHits = HashMap<String, BTreeMap<u32, u64>>;

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

/// A tracefile argument: `PATH` or `PATH:PREFIX`. `PREFIX` is the directory,
/// relative to the repository root, that the tracefile's relative `SF:` paths
/// are relative to (coverage.py writes paths relative to the directory pytest
/// ran in, e.g. `src/krishiv_bench/x.py` from `bench/`).
///
/// # Errors
/// Returns an error for an empty path or a prefix containing `..`.
pub fn parse_lcov_spec(spec: &str) -> Result<(PathBuf, String), String> {
    let (path, prefix) = spec.split_once(':').unwrap_or((spec, ""));
    if path.is_empty() {
        return Err(format!("--lcov {spec:?} has an empty path"));
    }
    let prefix = prefix.trim_matches('/');
    if prefix.split('/').any(|c| c == "..") {
        return Err(format!(
            "--lcov prefix {prefix:?} must be a plain repository-relative directory"
        ));
    }
    Ok((PathBuf::from(path), prefix.to_owned()))
}

/// Repository-relative form of an `SF:` path: an absolute path under `root`
/// loses it; a relative path loses leading `./` and gains `prefix`. Other
/// absolute paths are left alone (they match no diff path).
#[must_use]
pub fn normalize_sf(sf: &str, root: &str, prefix: &str) -> String {
    let root = root.trim_end_matches('/');
    if !root.is_empty()
        && let Some(rest) = sf.strip_prefix(root).and_then(|r| r.strip_prefix('/'))
    {
        return rest.to_owned();
    }
    if sf.starts_with('/') {
        return sf.to_owned();
    }
    let mut rel = sf;
    while let Some(r) = rel.strip_prefix("./") {
        rel = r;
    }
    if prefix.is_empty() {
        rel.to_owned()
    } else {
        format!("{prefix}/{rel}")
    }
}

/// Merges tracefiles into one map keyed by repository-relative path, summing
/// hit counts of lines seen in several files. Each part is a parsed
/// tracefile and its `PREFIX` (see [`parse_lcov_spec`]).
#[must_use]
pub fn merge_lcov(parts: &[(LineHits, String)], root: &str) -> LineHits {
    let mut out = LineHits::new();
    for (map, prefix) in parts {
        for (file, lines) in map {
            let dst = out.entry(normalize_sf(file, root, prefix)).or_default();
            for (n, h) in lines {
                *dst.entry(*n).or_default() += h;
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

impl DiffCov {
    /// Whether at least `min_percent` of the added executable lines ran. A
    /// diff with no executable lines meets any threshold.
    #[must_use]
    pub fn meets(&self, min_percent: u64) -> bool {
        self.total == 0 || self.covered * 100 >= min_percent * self.total
    }
}

/// Coverage of the added lines. Only lines the tracefile instruments count;
/// `root` is stripped from absolute `SF:` paths. Files on the PR-size
/// allowlist (generated, vendored and fixture files, `crate::ci::is_size_excluded`)
/// are not counted: nobody writes them, so a test cannot be asked for.
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
        if crate::ci::is_size_excluded(file) {
            continue;
        }
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
        "### Diff coverage\n\n{} of {} added executable lines covered ({}).\n",
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

/// Which packages the PR coverage run has to test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoverageScope {
    /// Nothing that compiles changed (docs, CI, agent config, bench): no Rust coverage run.
    None,
    /// The gate's own workflow changed: all packages, as a self-test of the scoping.
    Workspace,
    /// The changed packages and every package that depends on them, transitively. A changed
    /// line is covered by its own tests and by the tests of its dependents.
    Packages(Vec<String>),
}

/// The workflow that runs the gate: a change to it is tested against the whole workspace.
const COVERAGE_WORKFLOW: &str = ".github/workflows/coverage.yml";

/// Whether `path` cannot change what the Rust test suite covers.
fn is_coverage_neutral(path: &str) -> bool {
    // Markdown under `crates/` or `xtask/` can be compiled into rustdoc (`include_str!`) and
    // belongs to its package; Markdown elsewhere is documentation only.
    [
        "docs/",
        "bench/",
        ".github/",
        ".claude/",
        ".githooks/",
        ".superpowers/",
    ]
    .iter()
    .any(|d| path.starts_with(d))
        || (path.ends_with(".md") && !path.starts_with("crates/") && !path.starts_with("xtask/"))
}

/// Decides the coverage scope of a change from its paths and the workspace packages.
///
/// Diff coverage counts the executable lines a PR adds, and those lie in package sources. A
/// path inside a package directory therefore touches that package; every other path (manifests
/// and the lock file, toolchain pins, protos, shared test data, `fuzz/`, `ui/`) adds no
/// executable line of its own and is ignored here. Whether the rest of the workspace still
/// builds and passes is the job of the `test` and `clippy` checks, not of the coverage gate.
#[must_use]
pub fn coverage_scope(paths: &[String], pkgs: &[crate::precheck::Pkg]) -> CoverageScope {
    let mut touched = std::collections::BTreeSet::new();
    for path in paths.iter().map(|p| p.trim()).filter(|p| !p.is_empty()) {
        if path == COVERAGE_WORKFLOW {
            return CoverageScope::Workspace;
        }
        if is_coverage_neutral(path) {
            continue;
        }
        // Longest directory first so a nested package wins over its parent.
        let owner = pkgs
            .iter()
            .filter(|pkg| {
                !pkg.dir.is_empty()
                    && path
                        .strip_prefix(pkg.dir.as_str())
                        .is_some_and(|rest| rest.starts_with('/'))
            })
            .max_by_key(|pkg| pkg.dir.len());
        if let Some(pkg) = owner {
            touched.insert(pkg.name.as_str());
        }
    }
    if touched.is_empty() {
        CoverageScope::None
    } else {
        CoverageScope::Packages(crate::precheck::dependents_closure(&touched, pkgs))
    }
}

/// The `rust` and `packages` outputs of a scope: `packages` is `-p a -p b`, empty for the
/// whole workspace.
#[must_use]
pub fn scope_outputs(scope: &CoverageScope) -> (bool, String) {
    match scope {
        CoverageScope::None => (false, String::new()),
        CoverageScope::Workspace => (true, String::new()),
        CoverageScope::Packages(names) => (
            true,
            names
                .iter()
                .map(|n| format!("-p {n}"))
                .collect::<Vec<_>>()
                .join(" "),
        ),
    }
}

/// `xtask coverage-scope --base REF [--head REF] [--repo DIR]`: prints and exports to
/// `GITHUB_OUTPUT` `rust=true|false` and `packages=-p a -p b` (empty for the whole workspace),
/// the inputs of the PR coverage run.
///
/// # Errors
/// Returns an error for bad flags, a git failure or a cargo metadata failure.
pub fn run_coverage_scope(root: &Path, flags: &[&str]) -> Result<(), String> {
    let args = crate::policy::DiffArgs::parse(flags, root)?;
    let pkgs = crate::precheck::workspace_packages(args.repo())?;
    let scope = coverage_scope(&args.changed_paths()?, &pkgs);
    let (rust, packages) = scope_outputs(&scope);
    println!("coverage scope: {scope:?}");
    println!("rust={rust}");
    println!("packages={packages}");
    if let Some(path) = std::env::var_os("GITHUB_OUTPUT") {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(path)
            .map_err(|e| format!("cannot open GITHUB_OUTPUT: {e}"))?;
        writeln!(f, "rust={rust}\npackages={packages}").map_err(|e| e.to_string())?;
    }
    Ok(())
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

/// Parsed `xtask coverage` flags.
#[derive(Debug, PartialEq, Eq)]
struct CoverageFlags {
    branch: bool,
    all: bool,
    html: bool,
    packages: Vec<String>,
}

impl CoverageFlags {
    fn parse(flags: &[&str]) -> Result<Self, String> {
        let mut o = Self {
            branch: true,
            all: false,
            html: false,
            packages: Vec::new(),
        };
        let mut it = flags.iter();
        while let Some(f) = it.next() {
            match *f {
                "--no-branch" => o.branch = false,
                "--all-features" => o.all = true,
                "--html" => o.html = true,
                "-p" | "--package" => o
                    .packages
                    .push((*it.next().ok_or("-p needs a package name")?).to_owned()),
                other => return Err(format!("unknown flag {other}")),
            }
        }
        Ok(o)
    }
}

/// `--workspace`, or `-p NAME` per package.
fn package_args(packages: &[String]) -> Vec<String> {
    if packages.is_empty() {
        vec!["--workspace".to_owned()]
    } else {
        packages
            .iter()
            .flat_map(|p| ["-p".to_owned(), p.clone()])
            .collect()
    }
}

/// `xtask coverage [--no-branch] [--all-features] [--html] [-p PACKAGE]...`: runs the suite
/// under `cargo-llvm-cov`, writing `lcov.info` and `summary.json` (and an HTML
/// report) under `<target>/coverage`. Branch coverage needs the nightly
/// toolchain with `llvm-tools-preview`.
///
/// # Errors
/// Returns an error for unknown flags or when a cargo step fails.
pub fn run_coverage(root: &Path, flags: &[&str]) -> Result<(), String> {
    let CoverageFlags {
        branch,
        all,
        html,
        packages,
    } = CoverageFlags::parse(flags)?;
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
    let mut test = base(&["nextest", "--locked", "--no-tests=pass", "--no-report"]);
    test.args(package_args(&packages));
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

/// `xtask diff-coverage --base REF [--head REF] [--lcov PATH[:PREFIX]]... [--fail-under PCT]`:
/// coverage of the lines a PR adds, appended to the step summary in CI. `--lcov`
/// may repeat (Rust llvm-cov and Python coverage.py tracefiles are merged; the
/// threshold applies to the union, so CI gates each language in its own
/// invocation, which keeps a well-covered diff in one from masking the other). With
/// `--fail-under`, fails when fewer than `PCT`% of the added executable lines
/// ran (the merge gate is 90%).
///
/// # Errors
/// Returns an error for bad flags, a missing tracefile or a git failure.
pub fn run_diff_coverage(root: &Path, flags: &[&str]) -> Result<(), String> {
    let mut lcovs: Vec<(PathBuf, String)> = Vec::new();
    let mut fail_under = None;
    let mut rest = Vec::new();
    let mut it = flags.iter();
    while let Some(f) = it.next() {
        if *f == "--lcov" {
            lcovs.push(parse_lcov_spec(it.next().ok_or("--lcov needs a value")?)?);
        } else if *f == "--fail-under" {
            let v = it.next().ok_or("--fail-under needs a percentage")?;
            let pct: u64 = v
                .parse()
                .ok()
                .filter(|p| *p <= 100)
                .ok_or_else(|| format!("--fail-under expects 0..=100, got {v}"))?;
            fail_under = Some(pct);
        } else {
            rest.push(*f);
        }
    }
    let args = crate::policy::DiffArgs::parse(&rest, root)?;
    if lcovs.is_empty() {
        lcovs.push((coverage_dir(root).join("lcov.info"), String::new()));
    }
    let root_str = root.display().to_string();
    let mut parts = Vec::new();
    for (path, prefix) in lcovs {
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        parts.push((parse_lcov(&text), prefix));
    }
    let diff = args.unified_zero_diff()?;
    let d = diff_coverage(
        &parse_added_lines(&diff),
        &merge_lcov(&parts, &root_str),
        &root_str,
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
    match fail_under {
        Some(min) if !d.meets(min) => Err(format!(
            "diff coverage {} of {} added lines is below {min}%; add tests for the uncovered lines listed above",
            d.covered, d.total
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn coverage_flags_collect_packages() {
        let f = super::CoverageFlags::parse(&["--no-branch", "-p", "a", "--package", "b"]).unwrap();
        assert!(!f.branch && !f.all && !f.html);
        assert_eq!(super::package_args(&f.packages), ["-p", "a", "-p", "b"]);
        assert_eq!(super::package_args(&[]), ["--workspace"]);
        assert!(super::CoverageFlags::parse(&["-p"]).is_err());
        assert!(super::CoverageFlags::parse(&["--bogus"]).is_err());
        assert!(
            super::CoverageFlags::parse(&["--all-features", "--html"])
                .unwrap()
                .html
        );
    }

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
    fn lcov_specs_split_path_and_prefix() {
        assert_eq!(
            parse_lcov_spec("a/lcov.info").unwrap(),
            (PathBuf::from("a/lcov.info"), String::new())
        );
        assert_eq!(
            parse_lcov_spec("py.lcov:/bench/").unwrap(),
            (PathBuf::from("py.lcov"), "bench".to_owned())
        );
        for (bad, want) in [
            (":bench", "empty path"),
            ("x:../y", "plain repository-relative"),
            ("x:a/../b", "plain repository-relative"),
        ] {
            let e = parse_lcov_spec(bad).unwrap_err();
            assert!(e.contains(want), "{bad}: {e}");
        }
    }

    #[test]
    fn sf_paths_are_made_repository_relative() {
        assert_eq!(
            normalize_sf("/w/crates/a/lib.rs", "/w", ""),
            "crates/a/lib.rs"
        );
        assert_eq!(normalize_sf("/w/x.py", "/w/", ""), "x.py");
        // The root must end at a path boundary.
        assert_eq!(normalize_sf("/wx/y.py", "/w", ""), "/wx/y.py");
        assert_eq!(
            normalize_sf("src/krishiv_bench/a.py", "/w", "bench"),
            "bench/src/krishiv_bench/a.py"
        );
        assert_eq!(
            normalize_sf("./krishiv/a.py", "/w", "python"),
            "python/krishiv/a.py"
        );
        assert_eq!(
            normalize_sf("python/krishiv/a.py", "/w", ""),
            "python/krishiv/a.py"
        );
        assert_eq!(
            normalize_sf("/usr/lib/a.py", "/w", "bench"),
            "/usr/lib/a.py"
        );
    }

    #[test]
    fn rust_and_python_tracefiles_merge_and_gate() {
        let rust = parse_lcov("SF:/w/crates/a/src/lib.rs\nDA:1,1\nDA:2,0\nend_of_record\n");
        let bench = parse_lcov("SF:src/krishiv_bench/a.py\nDA:1,1\nDA:2,0\nend_of_record\n");
        let again = parse_lcov("SF:/w/bench/src/krishiv_bench/a.py\nDA:2,3\nend_of_record\n");
        let merged = merge_lcov(
            &[
                (rust, String::new()),
                (bench, "bench".to_owned()),
                (again, String::new()),
            ],
            "/w",
        );
        assert_eq!(merged.len(), 2);
        assert_eq!(merged["bench/src/krishiv_bench/a.py"][&2], 3);
        let diff = "+++ b/crates/a/src/lib.rs\n@@ -0,0 +1,2 @@\n+a\n+b\n+++ b/bench/src/krishiv_bench/a.py\n@@ -0,0 +1,3 @@\n+a\n+b\n+c\n";
        let d = diff_coverage(&parse_added_lines(diff), &merged, "/w");
        // Python line 3 has no DA record (not executable), so it is not counted;
        // Python line 2 was hit via the second tracefile.
        assert_eq!((d.covered, d.total), (3, 4));
        assert_eq!(d.uncovered["crates/a/src/lib.rs"], vec![2]);
    }

    #[test]
    fn diff_coverage_skips_generated_files() {
        let lcov = parse_lcov(
            "SF:/w/crates/a/src/proto/x.pb.rs\nDA:1,0\nend_of_record\nSF:/w/crates/a/src/lib.rs\nDA:1,1\nend_of_record\n",
        );
        let diff = "+++ b/crates/a/src/proto/x.pb.rs\n@@ -0,0 +1 @@\n+x\n+++ b/crates/a/src/lib.rs\n@@ -0,0 +1 @@\n+y\n";
        let d = diff_coverage(&parse_added_lines(diff), &lcov, "/w");
        assert_eq!((d.covered, d.total), (1, 1));
        assert!(d.uncovered.is_empty());
    }

    #[test]
    fn empty_diff_is_not_a_division_by_zero() {
        assert!(
            render_diff(&DiffCov::default()).contains("0 of 0 added executable lines covered (-)")
        );
    }

    #[test]
    fn diff_coverage_threshold() {
        let d = |covered, total| DiffCov {
            covered,
            total,
            uncovered: BTreeMap::new(),
        };
        assert!(d(0, 0).meets(90));
        assert!(d(9, 10).meets(90));
        assert!(!d(8, 10).meets(90));
        assert!(d(899, 999).meets(90) == (899 * 100 >= 90 * 999));
        assert!(d(1, 1).meets(100));
        assert!(d(0, 5).meets(0));
    }

    fn pkg(name: &str, dir: &str, deps: &[&str]) -> crate::precheck::Pkg {
        crate::precheck::Pkg {
            name: name.to_owned(),
            dir: dir.to_owned(),
            deps: deps.iter().map(|d| (*d).to_owned()).collect(),
        }
    }

    fn paths(p: &[&str]) -> Vec<String> {
        p.iter().map(|s| (*s).to_owned()).collect()
    }

    fn workspace() -> Vec<crate::precheck::Pkg> {
        vec![
            pkg("krishiv-common", "crates/krishiv-common", &[]),
            pkg("krishiv-exec", "crates/krishiv-exec", &["krishiv-common"]),
            pkg(
                "krishiv-session",
                "crates/krishiv-session",
                &["krishiv-exec"],
            ),
            pkg("krishiv-ui", "crates/krishiv-ui", &[]),
            pkg("xtask", "xtask", &["krishiv-common"]),
        ]
    }

    #[test]
    fn docs_and_ci_only_changes_need_no_coverage_run() {
        let s = super::coverage_scope(
            &paths(&[
                "docs/guide.md",
                ".github/workflows/ci.yml",
                "README.md",
                "bench/src/krishiv_bench/run.py",
                ".claude/agents/x.md",
            ]),
            &workspace(),
        );
        assert_eq!(s, super::CoverageScope::None);
        assert_eq!(
            super::coverage_scope(&[], &workspace()),
            super::CoverageScope::None
        );
        assert_eq!(
            super::coverage_scope(&paths(&["  "]), &workspace()),
            super::CoverageScope::None
        );
    }

    #[test]
    fn a_change_selects_the_package_and_its_dependents() {
        let s = super::coverage_scope(&paths(&["crates/krishiv-exec/src/lib.rs"]), &workspace());
        assert_eq!(
            s,
            super::CoverageScope::Packages(vec![
                "krishiv-exec".to_owned(),
                "krishiv-session".to_owned()
            ])
        );
        let leaf = super::coverage_scope(&paths(&["crates/krishiv-ui/src/a.rs"]), &workspace());
        assert_eq!(
            leaf,
            super::CoverageScope::Packages(vec!["krishiv-ui".to_owned()])
        );
    }

    #[test]
    fn markdown_inside_a_crate_belongs_to_the_crate() {
        let s = super::coverage_scope(&paths(&["crates/krishiv-ui/README.md"]), &workspace());
        assert_eq!(
            s,
            super::CoverageScope::Packages(vec!["krishiv-ui".to_owned()])
        );
        let x = super::coverage_scope(&paths(&["xtask/NOTES.md"]), &workspace());
        assert_eq!(x, super::CoverageScope::Packages(vec!["xtask".to_owned()]));
    }

    #[test]
    fn paths_outside_every_package_add_no_executable_lines() {
        for p in [
            "Cargo.lock",
            "Cargo.toml",
            "rust-toolchain.toml",
            "proto/krishiv/plan/v1/plan.proto",
            "tests/spark-golden/known-failures.json",
            "fuzz/fuzz_targets/a.rs",
        ] {
            assert_eq!(
                super::coverage_scope(&paths(&[p]), &workspace()),
                super::CoverageScope::None,
                "{p}"
            );
            assert_eq!(
                super::coverage_scope(&paths(&["crates/krishiv-ui/src/a.rs", p]), &workspace()),
                super::CoverageScope::Packages(vec!["krishiv-ui".to_owned()]),
                "{p}"
            );
        }
    }

    #[test]
    fn changing_the_gate_workflow_tests_the_whole_workspace() {
        assert_eq!(
            super::coverage_scope(
                &paths(&[
                    "crates/krishiv-ui/src/a.rs",
                    ".github/workflows/coverage.yml"
                ]),
                &workspace()
            ),
            super::CoverageScope::Workspace
        );
    }

    #[test]
    fn a_nested_package_wins_over_its_parent() {
        let mut ws = workspace();
        ws.push(pkg("inner", "crates/krishiv-ui/inner", &[]));
        let s = super::coverage_scope(&paths(&["crates/krishiv-ui/inner/src/a.rs"]), &ws);
        assert_eq!(s, super::CoverageScope::Packages(vec!["inner".to_owned()]));
    }

    #[test]
    fn scope_outputs_name_the_packages() {
        assert_eq!(
            super::scope_outputs(&super::CoverageScope::None),
            (false, String::new())
        );
        assert_eq!(
            super::scope_outputs(&super::CoverageScope::Workspace),
            (true, String::new())
        );
        assert_eq!(
            super::scope_outputs(&super::CoverageScope::Packages(vec![
                "a".to_owned(),
                "b".to_owned()
            ])),
            (true, "-p a -p b".to_owned())
        );
    }
}
