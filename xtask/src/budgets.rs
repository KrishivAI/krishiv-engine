// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask budgets`: dependency, build-time and binary-size budgets.
//!
//! Measures are compared with a baseline; growth of more than
//! [`MAX_GROWTH_PERCENT`] in any gated metric is a failure unless the caller
//! opts out (CI sets `BUDGET_APPROVED=1` for `budget-approved` PRs). Any
//! growth of a crate's unsafe sites needs its own approval
//! (`UNSAFE_APPROVED=1`, label `unsafe-reviewed`). The clean build time is
//! reported but never gated (shared runners are too noisy for a 5% rule).
//!
//! CI splits the work in two: `ci.yml` measures a pull request (`--measure-only`,
//! uploads `budgets-current.json`) and `main-measure.yml` every push to main
//! (artifact `budgets-main`),
//! and the label-aware gate in `pr-policy.yml` compares a pull request's
//! measurement with the main measurement for the commit it was built on
//! (`--compare FILE --baseline MAIN_FILE`), so adding an approval label
//! re-runs only the cheap gate and a PR is gated only on the growth it adds.
//! Without `--baseline` the baseline is the committed `budgets.json`, a
//! reference record that is not updated per merge (the main measurement is
//! authoritative); the gate falls back to it only when main has no
//! measurement.
//!
//! `cargo xtask precheck` runs the same delta locally through `--precheck`:
//! it measures the checkout and compares it with the cached measurement of
//! the merge-base with `--base` (see [`NO_MAIN_BASELINE`]), applying the same
//! [`compare`] rule as CI. `budgets.json` is never read on that path, so a
//! stale absolute baseline cannot fail the step. `--cache-main` records the
//! measurement (run it on a main checkout); the cache lives per main commit
//! under the target directory, so it is measured once.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

/// What one `cargo xtask budgets` invocation does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode {
    /// Measure, then compare with `budgets.json` and apply the verdict.
    Check {
        /// Also measure a clean `dist`-profile build's wall time.
        build_time: bool,
    },
    /// Measure and write `budgets-current.json` only; no verdict.
    Measure {
        /// Also measure a clean `dist`-profile build's wall time.
        build_time: bool,
    },
    /// Measure and overwrite the `budgets.json` baseline.
    WriteBaseline {
        /// Also measure a clean `dist`-profile build's wall time.
        build_time: bool,
    },
    /// Compare an existing measurement file with a baseline file (the
    /// committed `budgets.json` when `baseline` is `None`).
    Compare {
        /// The measurement (`budgets-current.json`).
        measured: PathBuf,
        /// The baseline to compare with, e.g. a measurement of main.
        baseline: Option<PathBuf>,
    },
    /// Measure this checkout and compare it with the cached measurement of
    /// the merge-base with `base` (the precheck budgets step). When no cached
    /// measurement exists the step skips (see [`NO_MAIN_BASELINE`]) instead
    /// of failing: CI checks the delta. `budgets.json` is never read.
    Precheck {
        /// The main ref the merge-base is taken against (`origin/main`).
        base: String,
        /// Repository to measure; `None` is the workspace root.
        repo: Option<PathBuf>,
    },
    /// Measure this checkout and cache it as the main measurement of `HEAD`,
    /// so later `--precheck` runs compare against it. Run on a main checkout.
    CacheMain {
        /// Repository to measure; `None` is the workspace root.
        repo: Option<PathBuf>,
    },
}

/// What `--precheck` prints when no cached main measurement exists yet.
pub const NO_MAIN_BASELINE: &str = "budgets: skipped (no main baseline; CI checks it)";

/// Whether `id` is safe to embed in a cache file name: a non-empty hex
/// commit id, so a hostile `--base` cannot escape the cache directory.
///
/// # Examples
///
/// ```rust
/// assert!(xtask::budgets::is_safe_sha(&"a".repeat(40)));
/// assert!(!xtask::budgets::is_safe_sha("../nope"));
/// ```
#[must_use]
pub fn is_safe_sha(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && id.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Where the main measurement for `sha` is cached: `budgets-main-<sha>.json`
/// under `<target>/precheck`, or `None` when `sha` is not a safe id.
///
/// # Examples
///
/// ```rust
/// let dir = std::path::Path::new("/t");
/// let sha = "a".repeat(40);
/// assert_eq!(
///     xtask::budgets::main_baseline_path(dir, &sha),
///     Some(dir.join("precheck").join(format!("budgets-main-{sha}.json")))
/// );
/// assert_eq!(xtask::budgets::main_baseline_path(dir, "../x"), None);
/// ```
#[must_use]
pub fn main_baseline_path(target_dir: &Path, sha: &str) -> Option<PathBuf> {
    is_safe_sha(sha).then(|| {
        target_dir
            .join("precheck")
            .join(format!("budgets-main-{sha}.json"))
    })
}

fn git_output(root: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|e| format!("failed to run git: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

/// The merge-base of `HEAD` and `base`: the main commit a `--precheck` run
/// gates its delta against.
///
/// # Errors
/// Returns the git failure, or when it reports something that is not a
/// commit id.
pub fn merge_base(root: &Path, base: &str) -> Result<String, String> {
    let sha = git_output(root, &["merge-base", "HEAD", base])?;
    if is_safe_sha(&sha) {
        Ok(sha)
    } else {
        Err(format!("bad merge-base for {base:?}: {sha:?}"))
    }
}

/// The id of `HEAD`, the key `--cache-main` files the measurement under.
///
/// # Errors
/// Returns the git failure, or when it reports something that is not a
/// commit id.
pub fn head_sha(root: &Path) -> Result<String, String> {
    let sha = git_output(root, &["rev-parse", "HEAD"])?;
    if is_safe_sha(&sha) {
        Ok(sha)
    } else {
        Err(format!("bad HEAD id: {sha:?}"))
    }
}

/// Parses the `budgets` flags; `None` on an unknown flag or a conflicting
/// combination (`--compare` measures nothing, so it takes no other flag
/// besides `--baseline`, which only `--compare` accepts; `--precheck` takes
/// only `--base` and `--repo`, `--cache-main` only `--repo`).
#[must_use]
pub fn parse_flags(flags: &[&str]) -> Option<Mode> {
    let (mut build_time, mut measure_only, mut write_baseline) = (false, false, false);
    let (mut precheck, mut cache_main) = (false, false);
    let (mut compare, mut baseline) = (None, None);
    let (mut base, mut repo): (Option<String>, Option<PathBuf>) = (None, None);
    let mut it = flags.iter();
    while let Some(f) = it.next() {
        match *f {
            "--build-time" => build_time = true,
            "--measure-only" => measure_only = true,
            "--write-baseline" => write_baseline = true,
            "--precheck" => precheck = true,
            "--cache-main" => cache_main = true,
            "--compare" => compare = Some(PathBuf::from(it.next()?)),
            "--baseline" => baseline = Some(PathBuf::from(it.next()?)),
            "--base" => base = Some(it.next()?.to_string()),
            "--repo" => repo = Some(PathBuf::from(it.next()?)),
            _ => return None,
        }
    }
    if baseline.is_some() && compare.is_none() {
        return None;
    }
    if !precheck && base.is_some() {
        return None;
    }
    if repo.is_some() && !(precheck || cache_main) {
        return None;
    }
    if precheck && (compare.is_some() || measure_only || write_baseline || build_time) {
        return None;
    }
    if cache_main
        && (compare.is_some()
            || measure_only
            || write_baseline
            || precheck
            || build_time
            || base.is_some())
    {
        return None;
    }
    match (compare, measure_only, write_baseline) {
        (Some(measured), false, false) if !build_time && !precheck && !cache_main => {
            Some(Mode::Compare { measured, baseline })
        }
        (None, false, false) if !precheck && !cache_main && base.is_none() && repo.is_none() => {
            Some(Mode::Check { build_time })
        }
        (None, true, false) if !precheck && !cache_main => Some(Mode::Measure { build_time }),
        (None, false, true) if !precheck && !cache_main => Some(Mode::WriteBaseline { build_time }),
        (None, false, false) if precheck => Some(Mode::Precheck {
            base: base.unwrap_or_else(|| "origin/main".to_owned()),
            repo,
        }),
        (None, false, false) if cache_main => Some(Mode::CacheMain { repo }),
        _ => None,
    }
}

/// Largest tolerated growth of any metric, in percent.
pub const MAX_GROWTH_PERCENT: u64 = 5;

/// One measurement (or baseline) set. `None` means "not measured".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budgets {
    /// Unique third-party crates in the shipped binary's graph: normal
    /// (non-dev, non-build-only) dependencies of `krishiv-cli` with default
    /// features (`cargo tree -p krishiv-cli -e normal`). Gated.
    pub transitive_crates: u64,
    /// Third-party crates in the all-features workspace graph (including
    /// dev/test crates). Informational, never gated.
    #[serde(default)]
    pub transitive_crates_all_features: Option<u64>,
    /// Size in bytes of the `dist`-profile `krishiv` binary.
    pub binary_bytes: Option<u64>,
    /// Clean `dist`-profile build wall time in seconds (CI only).
    pub clean_build_secs: Option<u64>,
    /// Lines with an `unsafe` keyword per crate (`crates/<name>/src`), for
    /// crates that have any. Any growth needs approval (ADR 0029: every new
    /// unsafe site is reviewed); `None` means not measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unsafe_sites: Option<BTreeMap<String, u64>>,
    /// Size in bytes of each Python wheel, keyed by its tags (P2-T103).
    /// Measured and gated by `cargo xtask wheel-sizes` in the wheels
    /// workflow, not by `budgets`; `--write-baseline` keeps the recorded
    /// values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wheel_bytes: Option<BTreeMap<String, u64>>,
}

#[derive(Debug, Deserialize)]
struct Metadata {
    workspace_members: Vec<String>,
    resolve: Option<Resolve>,
}

#[derive(Debug, Deserialize)]
struct Resolve {
    nodes: Vec<Node>,
}

#[derive(Debug, Deserialize)]
struct Node {
    id: String,
}

/// Counts unique non-workspace packages in `cargo tree --prefix none` output
/// (lines like `name vX.Y.Z`, `name vX.Y.Z (/path)` for path crates,
/// `(*)` and `[build-dependencies]` headers ignored).
#[must_use]
pub fn count_tree_crates(tree: &str) -> u64 {
    let set: BTreeSet<String> = tree
        .lines()
        .map(strip_ansi)
        .filter_map(|l| {
            let name = l.trim_end_matches(" (*)").trim_end_matches(" (proc-macro)");
            (!name.is_empty() && !name.starts_with('[') && !name.contains(" (/"))
                .then(|| name.to_owned())
        })
        .collect();
    u64::try_from(set.len()).unwrap_or(u64::MAX)
}

/// Removes ANSI escape sequences (`ESC [` ... final byte in `@`..=`~`), which
/// cargo emits when `CARGO_TERM_COLOR=always`.
fn strip_ansi(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.clone().next() == Some('[') {
            chars.next();
            for c in chars.by_ref() {
                if ('@'..='~').contains(&c) {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Counts resolved packages that are not workspace members.
///
/// # Errors
/// Returns an error if the JSON is malformed or has no resolve graph.
pub fn count_transitive(metadata_json: &str) -> Result<u64, String> {
    let meta: Metadata =
        serde_json::from_str(metadata_json).map_err(|e| format!("bad cargo metadata: {e}"))?;
    let resolve = meta.resolve.ok_or("cargo metadata has no resolve graph")?;
    let members: BTreeSet<&str> = meta.workspace_members.iter().map(String::as_str).collect();
    let n = resolve
        .nodes
        .iter()
        .filter(|n| !members.contains(n.id.as_str()))
        .count();
    Ok(u64::try_from(n).unwrap_or(u64::MAX))
}

fn over_budget(base: u64, now: u64) -> bool {
    // now > base * 1.05, in integers.
    u128::from(now) * 100 > u128::from(base) * u128::from(100 + MAX_GROWTH_PERCENT)
}

fn growth_pct(base: u64, now: u64) -> f64 {
    if base == 0 {
        return 0.0;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "ratio of small counts; precision loss is irrelevant"
    )]
    {
        (now as f64 - base as f64) / base as f64 * 100.0
    }
}

/// Result of comparing a measurement with the baseline.
#[derive(Debug)]
pub struct Comparison {
    /// Markdown table for the job summary.
    pub markdown: String,
    /// Metrics that grew by more than the allowed percentage.
    pub violations: Vec<String>,
    /// Crates whose unsafe-site count grew (reviewed separately, label
    /// `unsafe-reviewed` / `UNSAFE_APPROVED=1`).
    pub unsafe_violations: Vec<String>,
}

impl Comparison {
    /// The outcome given the two approvals: `Ok(true)` when something grew
    /// but was approved, `Ok(false)` when nothing grew, `Err` naming the
    /// missing approval otherwise.
    ///
    /// # Errors
    /// When a growth lacks its approval.
    pub fn verdict(&self, budget_approved: bool, unsafe_approved: bool) -> Result<bool, String> {
        let mut missing = Vec::new();
        if !self.violations.is_empty() && !budget_approved {
            missing
                .push("budget exceeded; add the `budget-approved` label or set BUDGET_APPROVED=1");
        }
        if !self.unsafe_violations.is_empty() && !unsafe_approved {
            missing
                .push("new unsafe sites; add the `unsafe-reviewed` label or set UNSAFE_APPROVED=1");
        }
        if missing.is_empty() {
            Ok(!self.violations.is_empty() || !self.unsafe_violations.is_empty())
        } else {
            Err(missing.join("\n"))
        }
    }
}

/// Compares `now` with `base`. Metrics missing on either side are skipped.
#[must_use]
pub fn compare(base: &Budgets, now: &Budgets) -> Comparison {
    // (name, baseline, current, gated). The build time is wall-clock on a
    // shared runner: shown, never gated.
    let rows = [
        (
            "transitive crates",
            Some(base.transitive_crates),
            Some(now.transitive_crates),
            true,
        ),
        (
            "dist binary bytes",
            base.binary_bytes,
            now.binary_bytes,
            true,
        ),
        (
            "clean build seconds",
            base.clean_build_secs,
            now.clean_build_secs,
            false,
        ),
    ];
    let mut markdown =
        String::from("| metric | baseline | current | change |\n|---|---|---|---|\n");
    let mut violations = Vec::new();
    let mut unsafe_violations = Vec::new();
    for (name, b, n, gated) in rows {
        match (b, n) {
            (Some(b), Some(n)) => {
                let pct = growth_pct(b, n);
                let flag = if gated && over_budget(b, n) {
                    violations.push(format!(
                        "{name}: {b} -> {n} ({pct:+.1}%) exceeds {MAX_GROWTH_PERCENT}% budget"
                    ));
                    " OVER BUDGET"
                } else {
                    ""
                };
                markdown.push_str(&format!("| {name} | {b} | {n} | {pct:+.1}%{flag} |\n"));
            }
            (b, n) => {
                let show = |v: Option<u64>| v.map_or_else(|| "n/a".to_owned(), |v| v.to_string());
                markdown.push_str(&format!(
                    "| {name} | {} | {} | not compared |\n",
                    show(b),
                    show(n)
                ));
            }
        }
    }
    if let (Some(b), Some(n)) = (&base.unsafe_sites, &now.unsafe_sites) {
        for (krate, &count) in n {
            let was = b.get(krate).copied().unwrap_or(0);
            let flag = if count > was {
                unsafe_violations.push(format!(
                    "unsafe sites in {krate}: {was} -> {count}; new unsafe needs review"
                ));
                " NEEDS REVIEW"
            } else {
                ""
            };
            markdown.push_str(&format!(
                "| unsafe sites: {krate} | {was} | {count} | {flag} |\n"
            ));
        }
    }
    Comparison {
        markdown,
        violations,
        unsafe_violations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(c: u64, bytes: Option<u64>) -> Budgets {
        Budgets {
            transitive_crates: c,
            transitive_crates_all_features: None,
            binary_bytes: bytes,
            clean_build_secs: None,
            unsafe_sites: None,
            wheel_bytes: None,
        }
    }

    #[test]
    fn any_unsafe_growth_is_a_violation() {
        let with = |n: u64| Budgets {
            unsafe_sites: Some(BTreeMap::from([("k".to_owned(), n)])),
            wheel_bytes: None,
            ..b(1, None)
        };
        assert!(compare(&with(2), &with(2)).unsafe_violations.is_empty());
        assert!(compare(&with(3), &with(2)).unsafe_violations.is_empty());
        let c = compare(&with(2), &with(3));
        assert_eq!((c.unsafe_violations.len(), c.violations.len()), (1, 0));
        assert!(c.markdown.contains("NEEDS REVIEW"));
        // Only the unsafe label approves unsafe growth.
        assert!(
            c.verdict(true, false)
                .unwrap_err()
                .contains("unsafe-reviewed")
        );
        assert_eq!(c.verdict(false, true), Ok(true));
        // Unmeasured baseline: not compared.
        assert_eq!(compare(&b(1, None), &with(1)).unsafe_violations.len(), 0);
        let base = Budgets {
            unsafe_sites: Some(BTreeMap::new()),
            wheel_bytes: None,
            ..b(1, None)
        };
        assert_eq!(compare(&base, &with(1)).unsafe_violations.len(), 1);
    }

    #[test]
    fn parses_precheck_and_cache_main_flags() {
        assert_eq!(
            parse_flags(&["--precheck", "--base", "origin/main"]),
            Some(Mode::Precheck {
                base: "origin/main".to_owned(),
                repo: None
            })
        );
        assert_eq!(
            parse_flags(&["--precheck"]),
            Some(Mode::Precheck {
                base: "origin/main".to_owned(),
                repo: None
            })
        );
        assert_eq!(
            parse_flags(&["--precheck", "--base", "b", "--repo", "r"]),
            Some(Mode::Precheck {
                base: "b".to_owned(),
                repo: Some(PathBuf::from("r"))
            })
        );
        assert_eq!(
            parse_flags(&["--cache-main"]),
            Some(Mode::CacheMain { repo: None })
        );
        assert_eq!(
            parse_flags(&["--cache-main", "--repo", "r"]),
            Some(Mode::CacheMain {
                repo: Some(PathBuf::from("r"))
            })
        );
        for bad in [
            &["--precheck", "--compare", "m.json"][..],
            &["--precheck", "--baseline", "b.json"],
            &["--precheck", "--build-time"],
            &["--precheck", "--measure-only"],
            &["--precheck", "--write-baseline"],
            &["--cache-main", "--build-time"],
            &["--cache-main", "--measure-only"],
            &["--cache-main", "--base", "b"],
            &["--cache-main", "--compare", "m.json"],
            &["--cache-main", "--precheck"],
            &["--base", "b"],
            &["--repo", "r"],
            &["--compare", "m.json", "--base", "b"],
            &["--compare", "m.json", "--repo", "r"],
            &["--measure-only", "--repo", "r"],
            &["--precheck", "--base"],
            &["--precheck", "--repo"],
        ] {
            assert_eq!(parse_flags(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn precheck_delta_applies_the_ci_rule_not_the_absolute_baseline() {
        // The branch equals the cached main measurement, but both sit far
        // above the stale absolute baseline: the old gate (against
        // budgets.json) would fail, the delta gate passes.
        let stale = b(100, Some(1_000_000));
        let main = b(150, Some(2_000_000));
        let branch = b(150, Some(2_000_000));
        assert!(!compare(&stale, &branch).violations.is_empty());
        assert_eq!(compare(&main, &branch).verdict(false, false), Ok(false));
        // Past the budget over main still fails without the label (the gate
        // is live, not skipped): 160 is more than 5% above 150.
        let grown = b(160, Some(2_000_000));
        assert!(
            compare(&main, &grown)
                .verdict(false, false)
                .unwrap_err()
                .contains("budget-approved")
        );
        assert_eq!(compare(&main, &grown).verdict(true, false), Ok(true));
    }

    #[test]
    fn main_baseline_cache_paths_are_keyed_by_sha() {
        let dir = Path::new("/t");
        let sha = "a".repeat(40);
        assert_eq!(
            main_baseline_path(dir, &sha),
            Some(
                dir.join("precheck")
                    .join(format!("budgets-main-{sha}.json"))
            )
        );
        assert!(is_safe_sha(&sha));
        assert!(is_safe_sha(&"9".repeat(64)));
        for bad in ["", "../x", "a/b", "a b", "ab!c", "g", &"a".repeat(65)] {
            assert!(!is_safe_sha(bad), "{bad:?}");
            assert_eq!(main_baseline_path(dir, bad), None);
        }
    }

    #[test]
    fn parses_flags() {
        let check = |b| Some(Mode::Check { build_time: b });
        assert_eq!(parse_flags(&[]), check(false));
        assert_eq!(parse_flags(&["--build-time"]), check(true));
        assert_eq!(
            parse_flags(&["--build-time", "--measure-only"]),
            Some(Mode::Measure { build_time: true })
        );
        assert_eq!(
            parse_flags(&["--write-baseline"]),
            Some(Mode::WriteBaseline { build_time: false })
        );
        assert_eq!(
            parse_flags(&["--compare", "m.json"]),
            Some(Mode::Compare {
                measured: PathBuf::from("m.json"),
                baseline: None
            })
        );
        assert_eq!(
            parse_flags(&["--baseline", "main.json", "--compare", "m.json"]),
            Some(Mode::Compare {
                measured: PathBuf::from("m.json"),
                baseline: Some(PathBuf::from("main.json"))
            })
        );
        for bad in [
            &["--baseline", "b.json"][..],
            &["--baseline"],
            &["--baseline", "b.json", "--build-time"],
            &["--compare", "m.json", "--baseline"],
            &["--compare"][..],
            &["--compare", "m.json", "--build-time"],
            &["--compare", "m.json", "--measure-only"],
            &["--measure-only", "--write-baseline"],
            &["--bogus"],
        ] {
            assert_eq!(parse_flags(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn verdicts() {
        let ok = compare(&b(100, None), &b(100, None));
        assert_eq!(ok.verdict(false, false), Ok(false));
        let grew = compare(&b(100, None), &b(200, None));
        assert!(
            grew.verdict(false, true)
                .unwrap_err()
                .contains("budget-approved")
        );
        assert_eq!(grew.verdict(true, false), Ok(true));
    }

    #[test]
    fn five_percent_is_allowed_six_is_not() {
        assert!(compare(&b(100, None), &b(105, None)).violations.is_empty());
        assert_eq!(compare(&b(100, None), &b(106, None)).violations.len(), 1);
    }

    #[test]
    fn build_time_is_reported_but_never_gated() {
        let mut base = b(100, None);
        base.clean_build_secs = Some(100);
        let mut now = b(100, None);
        now.clean_build_secs = Some(400);
        let c = compare(&base, &now);
        assert!(c.violations.is_empty());
        assert!(
            c.markdown
                .contains("| clean build seconds | 100 | 400 | +300.0% |")
        );
    }

    #[test]
    fn shrinking_and_unmeasured_pass() {
        assert!(
            compare(&b(100, Some(10)), &b(90, None))
                .violations
                .is_empty()
        );
    }

    #[test]
    fn counts_non_members() {
        let json = r#"{"workspace_members":["a 1"],"resolve":{"nodes":[{"id":"a 1"},{"id":"b 1"},{"id":"c 1"}]}}"#;
        assert_eq!(count_transitive(json), Ok(2));
    }

    #[test]
    fn counts_tree_crates_unique_and_skips_members() {
        let t = "krishiv-cli v0.1.0 (/x/cli)\nclap v4.6.7\nclap_builder v4.6.7\nclap v4.6.7 (*)\nsyn v2.0.1 (proc-macro)\nsyn v2.0.1\nkrishiv v0.1.0 (/x/k)\n";
        assert_eq!(count_tree_crates(t), 3);
    }

    #[test]
    fn colored_tree_output_counts_like_plain() {
        // `CARGO_TERM_COLOR=always` wraps the `(*)` and `(proc-macro)` markers
        // and headers in ANSI escapes; they must not change the count.
        let plain = "krishiv-cli v0.1.0 (/x/cli)\nclap v4.6.7\nclap_builder v4.6.7\nclap v4.6.7 (*)\nsyn v2.0.1 (proc-macro)\nsyn v2.0.1\nproc-macro2 v1.0.107 (*)\n[build-dependencies]\nquote v1.0.40\n";
        let colored = "krishiv-cli v0.1.0 (/x/cli)\nclap v4.6.7\nclap_builder v4.6.7\nclap v4.6.7 \x1b[33m\x1b[2m(*)\x1b[39m\x1b[22m\nsyn v2.0.1 \x1b[33m(proc-macro)\x1b[39m\nsyn v2.0.1\nproc-macro2 v1.0.107 \x1b[33m\x1b[2m(*)\x1b[39m\x1b[22m\n\x1b[1m[build-dependencies]\x1b[0m\nquote v1.0.40\n";
        assert_eq!(count_tree_crates(colored), count_tree_crates(plain));
        assert_eq!(count_tree_crates(colored), 5);
    }

    #[test]
    fn missing_resolve_is_error() {
        assert!(count_transitive(r#"{"workspace_members":[],"resolve":null}"#).is_err());
    }
}
