// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask quarantine [--list]`: the flaky-test quarantine registry (owner policy 2026-11-07).
//!
//! `tests/quarantine.toml` lists every quarantined test with an owner, an issue, the day it was added
//! and a deadline at most three days later. The check fails when a field is missing, a deadline has
//! passed (fix the test or remove it), or `.config/nextest.toml` disagrees with the registry: the
//! default profile must skip exactly the registered tests and the `quarantine` profile run exactly them.

#![expect(clippy::print_stdout, reason = "command output is the interface")]

use std::collections::BTreeSet;
use std::path::Path;

use serde::Deserialize;

/// Longest allowed stay in quarantine, in days.
const MAX_DAYS: i64 = 3;

#[derive(Debug, Deserialize)]
struct Registry {
    #[serde(default)]
    test: Vec<Entry>,
}

#[derive(Debug, Deserialize)]
struct Entry {
    name: String,
    package: String,
    owner: String,
    issue: String,
    added: String,
    deadline: String,
}

/// Days since 1970-01-01 of a `YYYY-MM-DD` date.
fn days(date: &str) -> Result<i64, String> {
    let bad = || format!("{date:?} is not a YYYY-MM-DD date");
    let mut it = date.split('-');
    let mut next = |len: usize| -> Result<i64, String> {
        let p = it.next().ok_or_else(bad)?;
        if p.len() != len {
            return Err(bad());
        }
        p.parse::<i64>().map_err(|_| bad())
    };
    let (y, m, d) = (next(4)?, next(2)?, next(2)?);
    if it.next().is_some() || !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return Err(bad());
    }
    // Howard Hinnant's days_from_civil.
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Ok(era * 146_097 + doe - 719_468)
}

fn today() -> i64 {
    i64::try_from(crate::profile::utc_now_millis() / 86_400_000).unwrap_or(0)
}

/// Names inside `test(=name)` terms of a nextest filter expression.
fn filter_names(filter: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut rest = filter;
    while let Some(i) = rest.find("test(=") {
        rest = &rest[i + 6..];
        if let Some(j) = rest.find(')') {
            out.insert(rest[..j].to_owned());
            rest = &rest[j..];
        }
    }
    out
}

fn profile_filter(nextest: &toml::Table, profile: &str) -> String {
    nextest
        .get("profile")
        .and_then(|p| p.get(profile))
        .and_then(|p| p.get("default-filter"))
        .and_then(toml::Value::as_str)
        .unwrap_or("")
        .to_owned()
}

/// Problems of a registry against the nextest config, as of day `today` (days since the epoch).
fn problems(registry: &str, nextest: &str, today: i64) -> Vec<String> {
    let mut out = Vec::new();
    let reg: Registry = match toml::from_str(registry) {
        Ok(r) => r,
        Err(e) => return vec![format!("tests/quarantine.toml: {e}")],
    };
    let nx: toml::Table = match toml::from_str(nextest) {
        Ok(t) => t,
        Err(e) => return vec![format!(".config/nextest.toml: {e}")],
    };
    let mut names = BTreeSet::new();
    for e in &reg.test {
        let n = &e.name;
        if [&e.name, &e.package, &e.owner, &e.issue]
            .iter()
            .any(|s| s.trim().is_empty())
        {
            out.push(format!(
                "{n}: name, package, owner and issue must all be set"
            ));
        }
        if !names.insert(e.name.clone()) {
            out.push(format!("{n}: listed twice"));
        }
        match (days(&e.added), days(&e.deadline)) {
            (Ok(a), Ok(d)) => {
                if d < a || d - a > MAX_DAYS {
                    out.push(format!(
                        "{n}: deadline must be 0..={MAX_DAYS} days after `added`"
                    ));
                }
                if d < today {
                    out.push(format!(
                        "{n}: deadline {} has passed: fix the test or remove it",
                        e.deadline
                    ));
                }
            }
            (a, d) => out.extend(
                [a.err(), d.err()]
                    .into_iter()
                    .flatten()
                    .map(|m| format!("{n}: {m}")),
            ),
        }
    }
    for (profile, what) in [("default", "skip"), ("quarantine", "run")] {
        let in_config = filter_names(&profile_filter(&nx, profile));
        for missing in names.difference(&in_config) {
            out.push(format!(
                "{missing}: registered but the `{profile}` profile does not {what} it"
            ));
        }
        for extra in in_config.difference(&names) {
            out.push(format!("{extra}: the `{profile}` profile {what}s it but tests/quarantine.toml does not list it"));
        }
    }
    out
}

/// Entry point of `cargo xtask quarantine [--list]`.
///
/// # Errors
/// Returns the problems found, or a message when a file cannot be read.
pub fn run(root: &Path, flags: &[&str]) -> Result<(), String> {
    let read = |p: &str| std::fs::read_to_string(root.join(p)).map_err(|e| format!("{p}: {e}"));
    let registry = read("tests/quarantine.toml")?;
    let nextest = read(".config/nextest.toml")?;
    match flags {
        [] => {}
        ["--list"] => {
            let reg: Registry = toml::from_str(&registry).map_err(|e| e.to_string())?;
            for e in reg.test {
                println!(
                    "{} ({}) owner {} issue {} deadline {}",
                    e.name, e.package, e.owner, e.issue, e.deadline
                );
            }
            return Ok(());
        }
        other => return Err(format!("unknown flags {other:?}")),
    }
    let found = problems(&registry, &nextest, today());
    if found.is_empty() {
        println!("quarantine: registry and nextest profiles agree");
        Ok(())
    } else {
        Err(found.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NX: &str = "[profile.default]\ndefault-filter = \"not test(=a)\"\n[profile.quarantine]\ndefault-filter = \"test(=a)\"\n";
    const REG: &str = "[[test]]\nname=\"a\"\npackage=\"p\"\nowner=\"o\"\nissue=\"#1\"\nadded=\"2026-11-07\"\ndeadline=\"2026-11-10\"\n";

    #[test]
    fn a_consistent_registry_passes() {
        let d = days("2026-11-08").unwrap();
        assert!(problems(REG, NX, d).is_empty());
    }

    #[test]
    fn a_passed_deadline_fails() {
        let d = days("2026-11-11").unwrap();
        assert!(problems(REG, NX, d)[0].contains("has passed"));
    }

    #[test]
    fn a_registry_and_config_that_disagree_fail() {
        let d = days("2026-11-08").unwrap();
        let nx = NX.replace("(=a)", "(=b)");
        assert_eq!(problems(REG, &nx, d).len(), 4);
    }

    #[test]
    fn more_than_three_days_fails() {
        let d = days("2026-11-08").unwrap();
        let reg = REG.replace("2026-11-10", "2026-11-12");
        assert!(problems(&reg, NX, d)[0].contains("days after"));
    }

    #[test]
    fn dates_count_days() {
        assert_eq!(days("1970-01-02").unwrap(), 1);
        assert!(days("2026-13-01").is_err());
    }
}
