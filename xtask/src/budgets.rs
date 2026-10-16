//! `cargo xtask budgets`: dependency, build-time and binary-size budgets.
//!
//! Measures are compared with the committed `budgets.json` baseline; growth
//! of more than [`MAX_GROWTH_PERCENT`] in any metric is a failure unless the
//! caller opts out (CI sets `BUDGET_APPROVED=1` for `budget-approved` PRs).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

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
    /// Size in bytes of the release `krishiv` binary.
    pub binary_bytes: Option<u64>,
    /// Clean release build wall time in seconds (CI only).
    pub clean_build_secs: Option<u64>,
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
    #[allow(clippy::cast_precision_loss)]
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
}

/// Compares `now` with `base`. Metrics missing on either side are skipped.
#[must_use]
pub fn compare(base: &Budgets, now: &Budgets) -> Comparison {
    let rows = [
        (
            "transitive crates",
            Some(base.transitive_crates),
            Some(now.transitive_crates),
        ),
        ("release binary bytes", base.binary_bytes, now.binary_bytes),
        (
            "clean build seconds",
            base.clean_build_secs,
            now.clean_build_secs,
        ),
    ];
    let mut markdown =
        String::from("| metric | baseline | current | change |\n|---|---|---|---|\n");
    let mut violations = Vec::new();
    for (name, b, n) in rows {
        match (b, n) {
            (Some(b), Some(n)) => {
                let pct = growth_pct(b, n);
                let flag = if over_budget(b, n) {
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
    Comparison {
        markdown,
        violations,
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
        }
    }

    #[test]
    fn five_percent_is_allowed_six_is_not() {
        assert!(compare(&b(100, None), &b(105, None)).violations.is_empty());
        assert_eq!(compare(&b(100, None), &b(106, None)).violations.len(), 1);
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
