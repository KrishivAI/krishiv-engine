// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask ci-time`: the wall time of the CI PR gate, measured from real runs.
//!
//! The release-engineering row asks for "measured PR-gate wall time before/after" a CI change. This command
//! reads the run list `gh run list --json databaseId,startedAt,updatedAt,conclusion,headBranch` prints (or
//! fetches it itself) and reports count, median, p90 and mean of `updatedAt - startedAt` over the successful
//! runs. `updatedAt` is the last update of the run, which for a finished run is its completion.
//!
//! ```text
//! cargo xtask ci-time [--workflow ci.yml] [--limit 50] [--event pull_request]   # fetch with gh, report
//! cargo xtask ci-time --file runs.json                                          # report one saved list
//! cargo xtask ci-time --compare before.json after.json                          # the change in the median and p90
//! ```
//!
//! Save the "before" list with `gh run list ... --json ... > before.json` before the change merges, the
//! "after" list once enough runs exist, and compare them. Nothing is written and nothing is published.

#![expect(clippy::print_stdout, reason = "command output is the interface")]

use std::path::Path;
use std::process::Command;

/// One finished workflow run, reduced to what the wall time needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Run {
    /// GitHub's run id (kept so a report can name the slowest run).
    pub id: u64,
    /// Seconds from `startedAt` to `updatedAt`.
    pub seconds: u64,
}

/// Wall-time statistics of a set of runs, in seconds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stats {
    /// Number of runs measured.
    pub count: usize,
    /// Median (the upper middle for an even count).
    pub median: u64,
    /// 90th percentile (nearest rank).
    pub p90: u64,
    /// Mean, rounded down.
    pub mean: u64,
    /// The slowest run's id and seconds.
    pub slowest: (u64, u64),
}

/// Days from 1970-01-01 to a civil date (Howard Hinnant's algorithm), so no date crate is needed.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Epoch seconds of a `YYYY-MM-DDTHH:MM:SSZ` timestamp (the only form GitHub prints).
pub fn parse_utc(ts: &str) -> Result<i64, String> {
    let bad = || format!("not a UTC timestamp: {ts:?}");
    let b = ts.as_bytes();
    if b.len() != 20
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
        || b[19] != b'Z'
    {
        return Err(bad());
    }
    let num = |r: std::ops::Range<usize>| -> Result<i64, String> {
        ts[r].parse::<i64>().map_err(|_| bad())
    };
    let (y, mo, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    let (h, mi, s) = (num(11..13)?, num(14..16)?, num(17..19)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || s > 60 {
        return Err(bad());
    }
    Ok(days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + s)
}

/// The successful runs of a `gh run list --json ...` document, with their wall times. Runs that did not
/// succeed, or whose clock runs backwards (a re-run), are skipped: they would skew the measurement.
pub fn parse_runs(json: &str) -> Result<Vec<Run>, String> {
    let doc: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("run list: {e}"))?;
    let items = doc.as_array().ok_or("run list: expected a JSON array")?;
    let mut runs = Vec::new();
    for item in items {
        let text = |k: &str| item.get(k).and_then(serde_json::Value::as_str);
        if text("conclusion").is_some_and(|c| c != "success") {
            continue;
        }
        let (Some(start), Some(end)) = (text("startedAt"), text("updatedAt")) else {
            continue;
        };
        let id = item
            .get("databaseId")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let (s, e) = (parse_utc(start)?, parse_utc(end)?);
        if let Ok(seconds) = u64::try_from(e - s) {
            runs.push(Run { id, seconds });
        }
    }
    Ok(runs)
}

/// Statistics over runs; `None` for an empty set.
pub fn stats(runs: &[Run]) -> Option<Stats> {
    if runs.is_empty() {
        return None;
    }
    let mut secs: Vec<u64> = runs.iter().map(|r| r.seconds).collect();
    secs.sort_unstable();
    let n = secs.len();
    // Nearest rank: the smallest value with at least 90% of the runs at or below it.
    let rank = (n * 9).div_ceil(10).max(1);
    let slow = runs.iter().max_by_key(|r| r.seconds)?;
    Some(Stats {
        count: n,
        median: secs[n / 2],
        p90: secs[rank - 1],
        mean: secs.iter().sum::<u64>() / u64::try_from(n).ok()?,
        slowest: (slow.id, slow.seconds),
    })
}

/// `1h02m03s`-style text for a number of seconds.
pub fn fmt_duration(secs: u64) -> String {
    let (h, m, s) = (secs / 3600, secs % 3600 / 60, secs % 60);
    if h > 0 {
        format!("{h}h{m:02}m{s:02}s")
    } else {
        format!("{m}m{s:02}s")
    }
}

/// The report for one set of runs.
pub fn report(label: &str, st: &Stats) -> String {
    format!(
        "{label}: {} runs, median {}, p90 {}, mean {}, slowest run {} ({})",
        st.count,
        fmt_duration(st.median),
        fmt_duration(st.p90),
        fmt_duration(st.mean),
        st.slowest.0,
        fmt_duration(st.slowest.1),
    )
}

/// The signed change from `before` to `after`, as text with a percentage (negative is faster).
pub fn delta(name: &str, before: u64, after: u64) -> String {
    let diff = i128::from(after) - i128::from(before);
    let sign = if diff < 0 { '-' } else { '+' };
    let abs = u64::try_from(diff.unsigned_abs()).unwrap_or(u64::MAX);
    let pct = if before == 0 {
        0.0
    } else {
        diff as f64 * 100.0 / before as f64
    };
    format!("{name}: {sign}{} ({pct:+.1}%)", fmt_duration(abs))
}

fn load(path: &str) -> Result<Vec<Run>, String> {
    let text = std::fs::read_to_string(Path::new(path)).map_err(|e| format!("{path}: {e}"))?;
    parse_runs(&text)
}

fn fetch(workflow: &str, limit: &str, event: &str) -> Result<Vec<Run>, String> {
    let out = Command::new("gh")
        .args([
            "run",
            "list",
            "--workflow",
            workflow,
            "--event",
            event,
            "--status",
            "success",
            "--limit",
            limit,
            "--json",
            "databaseId,startedAt,updatedAt,conclusion,headBranch",
        ])
        .output()
        .map_err(|e| format!("gh: {e} (install gh, or pass --file)"))?;
    if !out.status.success() {
        return Err(format!(
            "gh run list failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    parse_runs(&String::from_utf8_lossy(&out.stdout))
}

/// Entry point for `cargo xtask ci-time`.
pub fn run(flags: &[&str]) -> Result<(), String> {
    match flags {
        ["--compare", before, after] => {
            let (b, a) = (stats(&load(before)?), stats(&load(after)?));
            let (Some(b), Some(a)) = (b, a) else {
                return Err("a run list has no successful runs".to_owned());
            };
            println!("{}", report("before", &b));
            println!("{}", report("after", &a));
            println!("{}", delta("median", b.median, a.median));
            println!("{}", delta("p90", b.p90, a.p90));
            Ok(())
        }
        ["--file", path] => print_one(&load(path)?),
        rest => {
            let (mut workflow, mut limit, mut event) = ("ci.yml", "50", "pull_request");
            let mut it = rest.iter();
            while let Some(flag) = it.next() {
                let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
                match *flag {
                    "--workflow" => workflow = value,
                    "--limit" => limit = value,
                    "--event" => event = value,
                    other => return Err(format!("unknown flag {other}")),
                }
            }
            print_one(&fetch(workflow, limit, event)?)
        }
    }
}

fn print_one(runs: &[Run]) -> Result<(), String> {
    let st = stats(runs).ok_or("no successful runs to measure")?;
    println!("{}", report("PR gate", &st));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_json(id: u64, start: &str, end: &str, conclusion: &str) -> String {
        format!(
            r#"{{"databaseId":{id},"startedAt":"{start}","updatedAt":"{end}","conclusion":"{conclusion}"}}"#
        )
    }

    #[test]
    fn timestamps_parse_to_epoch_seconds() {
        assert_eq!(parse_utc("1970-01-01T00:00:00Z"), Ok(0));
        assert_eq!(parse_utc("2000-03-01T00:00:00Z"), Ok(951_868_800));
        assert_eq!(parse_utc("2026-11-22T02:50:07Z"), Ok(1_795_315_807));
        // A leap day and a year boundary.
        assert_eq!(
            parse_utc("2024-03-01T00:00:00Z").unwrap() - parse_utc("2024-02-28T00:00:00Z").unwrap(),
            2 * 86_400
        );
    }

    #[test]
    fn malformed_timestamps_are_refused() {
        for bad in [
            "",
            "2026-11-22 02:50:07Z",
            "2026-13-01T00:00:00Z",
            "2026-11-22T25:00:00Z",
            "2026-11-22T02:50:07+01:00",
            "20x6-11-22T02:50:07Z",
        ] {
            assert!(parse_utc(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn only_successful_forward_running_runs_count() {
        let json = format!(
            "[{},{},{},{}]",
            run_json(1, "2026-11-22T00:00:00Z", "2026-11-22T00:10:00Z", "success"),
            run_json(2, "2026-11-22T00:00:00Z", "2026-11-22T00:20:00Z", "failure"),
            run_json(3, "2026-11-22T01:00:00Z", "2026-11-22T00:30:00Z", "success"),
            run_json(4, "2026-11-22T02:00:00Z", "2026-11-22T02:05:30Z", "success"),
        );
        let runs = parse_runs(&json).unwrap();
        assert_eq!(
            runs,
            vec![
                Run {
                    id: 1,
                    seconds: 600
                },
                Run {
                    id: 4,
                    seconds: 330
                }
            ]
        );
    }

    #[test]
    fn a_document_that_is_not_an_array_is_refused() {
        assert!(parse_runs("{}").is_err());
        assert!(parse_runs("not json").is_err());
        assert_eq!(parse_runs("[]"), Ok(vec![]));
    }

    #[test]
    fn stats_use_nearest_rank_for_p90() {
        let runs: Vec<Run> = (1..=10)
            .map(|i| Run {
                id: i,
                seconds: i * 100,
            })
            .collect();
        let st = stats(&runs).unwrap();
        assert_eq!(st.count, 10);
        assert_eq!(st.p90, 900);
        assert_eq!(st.median, 600);
        assert_eq!(st.mean, 550);
        assert_eq!(st.slowest, (10, 1000));
        assert!(stats(&[]).is_none());
        let one = stats(&[Run { id: 7, seconds: 42 }]).unwrap();
        assert_eq!((one.median, one.p90, one.mean), (42, 42, 42));
    }

    #[test]
    fn durations_and_deltas_read_plainly() {
        assert_eq!(fmt_duration(59), "0m59s");
        assert_eq!(fmt_duration(3723), "1h02m03s");
        assert_eq!(delta("median", 1000, 800), "median: -3m20s (-20.0%)");
        assert_eq!(delta("p90", 600, 660), "p90: +1m00s (+10.0%)");
        assert_eq!(delta("p90", 0, 60), "p90: +1m00s (+0.0%)");
    }

    #[test]
    fn the_report_names_the_slowest_run() {
        let st = stats(&[
            Run { id: 5, seconds: 60 },
            Run {
                id: 9,
                seconds: 120,
            },
        ])
        .unwrap();
        let text = report("PR gate", &st);
        assert!(text.starts_with("PR gate: 2 runs, median 2m00s"), "{text}");
        assert!(text.contains("slowest run 9 (2m00s)"), "{text}");
    }

    #[test]
    fn run_reads_a_saved_list_and_compares_two() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, secs: u64| {
            let end = format!("2026-11-22T00:{:02}:{:02}Z", secs / 60, secs % 60);
            let p = dir.path().join(name);
            std::fs::write(
                &p,
                format!("[{}]", run_json(1, "2026-11-22T00:00:00Z", &end, "success")),
            )
            .unwrap();
            p.to_string_lossy().into_owned()
        };
        let (before, after) = (write("b.json", 1200), write("a.json", 900));
        assert!(run(&["--file", &before]).is_ok());
        assert!(run(&["--compare", &before, &after]).is_ok());
        let empty = dir.path().join("e.json");
        std::fs::write(&empty, "[]").unwrap();
        let empty = empty.to_string_lossy().into_owned();
        assert!(run(&["--file", &empty]).is_err());
        assert!(run(&["--compare", &before, &empty]).is_err());
        assert!(run(&["--file", "/nonexistent/runs.json"]).is_err());
        assert!(run(&["--bogus", "x"]).is_err());
        assert!(run(&["--limit"]).is_err());
    }
}
