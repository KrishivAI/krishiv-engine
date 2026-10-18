// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask wheel-sizes DIR [--write]`: the size budget of the Python
//! wheels (P2-T103), recorded in `budgets.json` under `wheel_bytes`.
//!
//! Each wheel is keyed by its tags (`cp311-abi3-manylinux_2_28_x86_64`), so
//! one entry per target and flavour. A wheel more than
//! [`crate::budgets::MAX_GROWTH_PERCENT`] larger than its entry, or a wheel
//! with no entry, needs the `budget-approved` label (`BUDGET_APPROVED=1`);
//! `--write` records the measured sizes.

use std::collections::BTreeMap;

use crate::budgets::MAX_GROWTH_PERCENT;

/// The tags of a wheel file name (`{name}-{version}(-{build})?-{python}-{abi}-{platform}.whl`):
/// `python-abi-platform`, or `None` for a name that is not a wheel's.
#[must_use]
pub fn wheel_key(file_name: &str) -> Option<String> {
    let stem = file_name.strip_suffix(".whl")?;
    let parts: Vec<&str> = stem.split('-').collect();
    if !(5..=6).contains(&parts.len()) || parts.iter().any(|p| p.is_empty()) {
        return None;
    }
    Some(parts[parts.len() - 3..].join("-"))
}

/// Result of comparing measured wheels with the recorded sizes.
#[derive(Debug, Default)]
pub struct WheelComparison {
    /// Markdown table for the job summary.
    pub markdown: String,
    /// Wheels over budget or without a recorded size.
    pub violations: Vec<String>,
}

/// Compares the `measured` wheel sizes with the `recorded` ones.
#[must_use]
pub fn compare(
    recorded: &BTreeMap<String, u64>,
    measured: &BTreeMap<String, u64>,
) -> WheelComparison {
    let mut out = WheelComparison {
        markdown: String::from("| wheel | recorded | current | change |\n|---|---|---|---|\n"),
        ..WheelComparison::default()
    };
    for (key, &now) in measured {
        match recorded.get(key) {
            Some(&base) => {
                let over =
                    u128::from(now) * 100 > u128::from(base) * u128::from(100 + MAX_GROWTH_PERCENT);
                let change = if base == 0 {
                    "n/a".to_owned()
                } else {
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "a percentage for display; wheel sizes fit f64 exactly"
                    )]
                    let pct = (now as f64 - base as f64) / base as f64 * 100.0;
                    format!("{pct:+.2}%")
                };
                out.markdown
                    .push_str(&format!("| {key} | {base} | {now} | {change} |\n"));
                if over {
                    out.violations.push(format!(
                        "wheel {key} grew from {base} to {now} bytes (more than {MAX_GROWTH_PERCENT}%)"
                    ));
                }
            }
            None => {
                out.markdown
                    .push_str(&format!("| {key} | — | {now} | new |\n"));
                out.violations.push(format!(
                    "wheel {key} ({now} bytes) has no recorded size in budgets.json"
                ));
            }
        }
    }
    for key in recorded.keys().filter(|k| !measured.contains_key(*k)) {
        out.markdown
            .push_str(&format!("| {key} | {} | — | not built |\n", recorded[key]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_the_wheel_tags() {
        assert_eq!(
            wheel_key("krishiv-0.1.0.dev0-cp311-abi3-manylinux_2_28_x86_64.whl").as_deref(),
            Some("cp311-abi3-manylinux_2_28_x86_64")
        );
        assert_eq!(
            wheel_key("krishiv-0.1.0-1-cp314-cp314t-macosx_11_0_arm64.whl").as_deref(),
            Some("cp314-cp314t-macosx_11_0_arm64")
        );
        assert_eq!(wheel_key("krishiv-0.1.0.tar.gz"), None);
        assert_eq!(wheel_key("a-b.whl"), None);
        assert_eq!(wheel_key("a--b-c-d.whl"), None);
    }

    fn map(entries: &[(&str, u64)]) -> BTreeMap<String, u64> {
        entries.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect()
    }

    #[test]
    fn growth_within_the_budget_passes() {
        let cmp = compare(&map(&[("a", 1000), ("gone", 5)]), &map(&[("a", 1050)]));
        assert!(cmp.violations.is_empty(), "{:?}", cmp.violations);
        assert!(
            cmp.markdown.contains("| a | 1000 | 1050 | +5.00% |"),
            "{}",
            cmp.markdown
        );
        assert!(cmp.markdown.contains("| gone | 5 | — | not built |"));
    }

    #[test]
    fn growth_past_the_budget_and_new_wheels_fail() {
        let cmp = compare(&map(&[("a", 1000)]), &map(&[("a", 1051), ("b", 7)]));
        assert_eq!(cmp.violations.len(), 2, "{:?}", cmp.violations);
        assert!(cmp.violations[0].contains("grew from 1000 to 1051"));
        assert!(cmp.violations[1].contains("no recorded size"));
        assert!(cmp.markdown.contains("| b | — | 7 | new |"));
    }

    #[test]
    fn shrinking_passes() {
        let cmp = compare(
            &map(&[("a", 1000), ("s", 1000), ("z", 0)]),
            &map(&[("a", 900), ("s", 995), ("z", 0)]),
        );
        assert!(cmp.violations.is_empty());
        assert!(
            cmp.markdown.contains("| a | 1000 | 900 | -10.00% |"),
            "{}",
            cmp.markdown
        );
        assert!(
            cmp.markdown.contains("| s | 1000 | 995 | -0.50% |"),
            "{}",
            cmp.markdown
        );
        assert!(cmp.markdown.contains("| z | 0 | 0 | n/a |"));
    }
}
