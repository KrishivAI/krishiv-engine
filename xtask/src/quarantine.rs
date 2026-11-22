// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask quarantine [--list | --flaky <nextest-output>]`: the flaky-test quarantine registry (owner policy 2026-11-07).
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
#[serde(deny_unknown_fields)]
struct Registry {
    #[serde(default)]
    test: Vec<Entry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
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
        if !p.bytes().all(|b| b.is_ascii_digit()) {
            return Err(bad());
        }
        p.parse::<i64>().map_err(|_| bad())
    };
    let (y, m, d) = (next(4)?, next(2)?, next(2)?);
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let month_days = match m {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    if it.next().is_some() || !(1..=12).contains(&m) || !(1..=month_days).contains(&d) {
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

/// The exact `default-filter` a profile must carry for the registered `names` (in registry order):
/// the default profile skips each one, the quarantine profile runs exactly them. Comparing whole
/// strings means a missing `not`, an extra `and not package(x)` or a stray `|` fails the gate.
fn expected_filter(profile: &str, names: &[&str]) -> String {
    match (profile, names.is_empty()) {
        ("default", true) => String::new(),
        ("default", false) => names
            .iter()
            .map(|n| format!("not test(={n})"))
            .collect::<Vec<_>>()
            .join(" and "),
        (_, true) => "none()".to_owned(),
        (_, false) => names
            .iter()
            .map(|n| format!("test(={n})"))
            .collect::<Vec<_>>()
            .join(" | "),
    }
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
    let listed: Vec<&str> = reg.test.iter().map(|e| e.name.as_str()).collect();
    for profile in ["default", "quarantine"] {
        let want = expected_filter(profile, &listed);
        let have = profile_filter(&nx, profile);
        if have.trim() != want {
            out.push(format!(
                "the `{profile}` profile's default-filter must be exactly {want:?} for tests/quarantine.toml, found {have:?}"
            ));
        }
    }
    out
}

/// Tests nextest reported as `FLAKY` (failed, then passed on a retry) in a captured run, as
/// `(package, test name)` pairs. Colour codes are ignored.
fn flaky_tests(log: &str) -> BTreeSet<(String, String)> {
    let mut out = BTreeSet::new();
    for line in log.lines() {
        let plain = strip_ansi(line);
        let Some(rest) = plain.trim_start().strip_prefix("FLAKY") else {
            continue;
        };
        let Some((_, after)) = rest.split_once(']') else {
            continue;
        };
        // nextest 0.9.x prints a counter such as `(2/2)` or `(───)` after the duration.
        // The counter pads with spaces (`(  2/100)`), so drop everything up to its closing `)`.
        let after = after.trim_start();
        let after = if after.starts_with('(') {
            after.split_once(')').map_or(after, |(_, r)| r)
        } else {
            after
        };
        let mut words = after.split_whitespace();
        if let (Some(binary), Some(name)) = (words.next(), words.next()) {
            let package = binary.split("::").next().unwrap_or(binary);
            out.insert((package.to_owned(), name.to_owned()));
        }
    }
    out
}

/// Names of the tests the default profile retries on purpose (`[[profile.default.overrides]]` with `retries`):
/// they flake by design on loaded runners, so a `FLAKY` line for them is not a finding.
fn retried_on_purpose(nextest: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let Ok(doc) = toml::from_str::<toml::Table>(nextest) else {
        return out;
    };
    let overrides = doc
        .get("profile")
        .and_then(|p| p.get("default"))
        .and_then(|p| p.get("overrides"))
        .and_then(toml::Value::as_array);
    for o in overrides.into_iter().flatten() {
        if o.get("retries").is_none() {
            continue;
        }
        let filter = o.get("filter").and_then(toml::Value::as_str).unwrap_or("");
        for part in filter.split("test(=").skip(1) {
            if let Some((name, _)) = part.split_once(')') {
                out.insert(name.trim().to_owned());
            }
        }
    }
    out
}

fn strip_ansi(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for n in chars.by_ref() {
                if n.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Entry point of `cargo xtask quarantine [--list | --flaky <nextest-output>]`.
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
        ["--flaky", log] => {
            let text = std::fs::read_to_string(log).map_err(|e| format!("{log}: {e}"))?;
            let reg: Registry = toml::from_str(&registry).map_err(|e| e.to_string())?;
            let known: BTreeSet<&str> = reg.test.iter().map(|e| e.name.as_str()).collect();
            let tolerated = retried_on_purpose(&nextest);
            let new: Vec<_> = flaky_tests(&text)
                .into_iter()
                .filter(|(_, n)| !known.contains(n.as_str()) && !tolerated.contains(n))
                .collect();
            if new.is_empty() {
                println!("quarantine: no unregistered flaky tests in {log}");
                return Ok(());
            }
            for (package, name) in &new {
                println!(
                    "[[test]]\nname = \"{name}\"\npackage = \"{package}\"\nowner = \"<lane>\"\nissue = \"<#issue>\"\nadded = \"<YYYY-MM-DD today>\"\ndeadline = \"<added + 3 days>\"\n"
                );
            }
            return Err(format!(
                "{} flaky test(s) passed only on a retry and are not in tests/quarantine.toml: open an issue, fix within 3 days or register them with an owner; the stanza goes with `default-filter = \"not test(=NAME)\"` on the default profile and `test(=NAME)` on the quarantine profile in .config/nextest.toml (cargo xtask quarantine checks both)",
                new.len()
            ));
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
        assert_eq!(problems(REG, &nx, d).len(), 2);
    }

    #[test]
    fn a_wrong_filter_shape_fails() {
        let d = days("2026-11-08").unwrap();
        for bad in [
            NX.replace("\"not test(=a)\"", "\"test(=a)\""),
            NX.replace("\"test(=a)\"", "\"not test(=a)\""),
            NX.replace("\"not test(=a)\"", "\"not test(=a) | not test(=b)\""),
            NX.replace("\"not test(=a)\"", "\"not test(=a) and not package(x)\""),
        ] {
            assert_eq!(problems(REG, &bad, d).len(), 1, "{bad}");
        }
    }

    #[test]
    fn unknown_fields_and_impossible_dates_fail() {
        let d = days("2026-11-08").unwrap();
        assert!(problems(&format!("{REG}extra=1\n"), NX, d)[0].contains("extra"));
        assert!(days("2026-02-31").is_err());
        assert!(days("+202-01-01").is_err());
        assert!(days("2024-02-29").is_ok());
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

    #[test]
    fn broken_files_and_empty_fields_are_reported() {
        let d = days("2026-11-08").unwrap();
        assert!(problems("[[test", NX, d)[0].starts_with("tests/quarantine.toml"));
        assert!(problems(REG, "[[x", d)[0].starts_with(".config/nextest.toml"));
        let empty = REG.replace("owner=\"o\"", "owner=\" \"");
        assert!(problems(&empty, NX, d)[0].contains("must all be set"));
        let bad_date = REG.replace("2026-11-10", "tomorrow");
        assert!(problems(&bad_date, NX, d)[0].contains("not a YYYY-MM-DD"));
        let twice = format!("{REG}{REG}");
        assert!(
            problems(&twice, NX, d)
                .iter()
                .any(|m| m.contains("listed twice"))
        );
    }

    #[test]
    fn several_entries_and_an_empty_registry_have_their_own_filters() {
        let d = days("2026-11-08").unwrap();
        let two = format!("{REG}{}", REG.replace("name=\"a\"", "name=\"b\""));
        let nx = "[profile.default]\ndefault-filter = \"not test(=a) and not test(=b)\"\n[profile.quarantine]\ndefault-filter = \"test(=a) | test(=b)\"\n";
        assert!(problems(&two, nx, d).is_empty());
        let none = "[profile.quarantine]\ndefault-filter = \"none()\"\n";
        assert!(problems("", none, d).is_empty());
        assert_eq!(problems("", NX, d).len(), 2);
    }

    fn repo(name: &str, registry: &str, nextest: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("xtask-quarantine-{name}-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("tests")).unwrap();
        std::fs::create_dir_all(dir.join(".config")).unwrap();
        std::fs::write(dir.join("tests/quarantine.toml"), registry).unwrap();
        std::fs::write(dir.join(".config/nextest.toml"), nextest).unwrap();
        dir
    }

    #[test]
    fn tests_retried_on_purpose_are_read_from_the_default_overrides() {
        let nx = "[[profile.default.overrides]]\nfilter = \"test(=a) | test(=b_c)\"\nretries = 2\n[[profile.default.overrides]]\nfilter = \"test(=d)\"\nthreads-required = 2\n";
        let got: Vec<_> = retried_on_purpose(nx).into_iter().collect();
        assert_eq!(got, ["a", "b_c"]);
    }

    #[test]
    fn flaky_lines_are_found_with_or_without_colour() {
        let log = "        FLAKY 2/3 [   0.005s] (2/2) flk flips\n        FLAKY 2/3 [   0.005s] (  2/100) pad padded\n        PASS [   0.1s] a::b ok\n\u{1b}[33m       FLAKY\u{1b}[0m 2/3 [   0.5s] krishiv-session::udaf window_x\n        FLAKY 2/3 [   0.5s] krishiv-exec lib_test\n";
        let got: Vec<_> = flaky_tests(log).into_iter().collect();
        assert_eq!(
            got,
            vec![
                ("flk".to_owned(), "flips".to_owned()),
                ("krishiv-exec".to_owned(), "lib_test".to_owned()),
                ("krishiv-session".to_owned(), "window_x".to_owned()),
                ("pad".to_owned(), "padded".to_owned())
            ]
        );
    }

    #[test]
    fn a_registered_flaky_test_passes_and_an_unregistered_one_fails() {
        let dir = repo("flaky", REG, NX);
        let log = dir.join("run.log");
        std::fs::write(&log, "FLAKY 2/3 [ 1s] p::bin a\n").unwrap();
        let path = log.to_str().unwrap();
        assert!(run(&dir, &["--flaky", path]).is_ok());
        std::fs::write(&log, "FLAKY 2/3 [ 1s] p::bin zzz\n").unwrap();
        assert!(
            run(&dir, &["--flaky", path])
                .unwrap_err()
                .contains("1 flaky")
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn run_checks_lists_and_refuses_unknown_flags() {
        let ok = repo(
            "ok",
            "",
            "[profile.quarantine]\ndefault-filter = \"none()\"\n",
        );
        assert!(run(&ok, &[]).is_ok());
        assert!(run(&ok, &["--list"]).is_ok());
        assert!(run(&ok, &["--nope"]).unwrap_err().contains("unknown flags"));
        let bad = repo("bad", REG, NX.replace("(=a)", "(=b)").as_str());
        assert!(run(&bad, &[]).unwrap_err().contains("default-filter"));
        assert!(run(&std::env::temp_dir().join("xtask-quarantine-missing"), &[]).is_err());
        for d in [ok, bad] {
            std::fs::remove_dir_all(d).unwrap();
        }
    }
}
