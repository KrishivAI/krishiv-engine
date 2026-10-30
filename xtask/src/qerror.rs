// SPDX-License-Identifier: Apache-2.0
//! `cargo xtask qerror [--data DIR] [--scale N] [--suite tpch|tpcds]...`: the
//! q-error report of the optimizer's cardinality estimates over TPC-H and
//! TPC-DS (median, 95th percentile and largest of `max(est / actual, actual /
//! est)` per join and aggregate, with statistics propagation on and off).
//!
//! The report itself is the `krishiv-optimizer` test `qerror` (it needs the
//! reference interpreter and the in-process data generators, which are test
//! support); this runs it with the settings it reads, so `xtask` stays free of
//! an engine dependency. Without `--data` the data are generated in process
//! at a small scale.

use std::path::Path;
use std::process::Command;

/// The parsed command line.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QerrorArgs {
    /// `--data DIR`: pipe-delimited table files instead of generated data.
    pub data: Option<String>,
    /// `--scale N`: TPC-H orders of the generated data.
    pub scale: Option<u64>,
    /// `--suite NAME`, repeatable: `tpch` and `tpcds` (both by default).
    pub suites: Vec<String>,
}

/// Parses the command line.
///
/// # Errors
/// A usage error for an unknown flag, a missing or invalid value, or an
/// unknown suite.
pub fn parse_args(flags: &[&str]) -> Result<QerrorArgs, String> {
    let mut args = QerrorArgs::default();
    let mut it = flags.iter();
    while let Some(f) = it.next() {
        match *f {
            "--data" => {
                let dir = it.next().ok_or("--data needs a directory")?;
                args.data = Some((*dir).to_owned());
            }
            "--scale" => {
                let n = it.next().ok_or("--scale needs a number of orders")?;
                match n.parse::<u64>() {
                    Ok(v) if v > 0 => args.scale = Some(v),
                    _ => return Err(format!("--scale expects a positive number, got {n:?}")),
                }
            }
            "--suite" => {
                let s = it.next().ok_or("--suite needs tpch or tpcds")?;
                if !["tpch", "tpcds"].contains(s) {
                    return Err(format!("--suite expects tpch or tpcds, got {s:?}"));
                }
                if !args.suites.iter().any(|x| x == s) {
                    args.suites.push((*s).to_owned());
                }
            }
            other => {
                return Err(format!(
                    "unknown flag {other:?}; usage: cargo xtask qerror [--data DIR] [--scale N] [--suite tpch|tpcds]..."
                ));
            }
        }
    }
    Ok(args)
}

/// The cargo arguments and the environment of the run.
#[must_use]
pub fn invocation(args: &QerrorArgs) -> (Vec<String>, Vec<(String, String)>) {
    let cargo = [
        "test",
        "-p",
        "krishiv-optimizer",
        "--test",
        "it",
        "--",
        "qerror::q_error_report_on_generated_data",
        "--exact",
        "--nocapture",
    ]
    .map(str::to_owned)
    .to_vec();
    let mut env = Vec::new();
    if let Some(dir) = &args.data {
        env.push(("KRISHIV_QERROR_DIR".to_owned(), dir.clone()));
    }
    if let Some(n) = args.scale {
        env.push(("KRISHIV_QERROR_SCALE".to_owned(), n.to_string()));
    }
    if !args.suites.is_empty() {
        env.push(("KRISHIV_QERROR_SUITES".to_owned(), args.suites.join(",")));
    }
    (cargo, env)
}

/// Runs the report.
///
/// # Errors
/// A usage error, a data directory that does not exist, or the test failing.
pub fn run_qerror(root: &Path, flags: &[&str]) -> Result<(), String> {
    let mut args = parse_args(flags)?;
    if let Some(dir) = &args.data {
        // The test runs in the package directory: a relative path would
        // name another directory there.
        let abs = std::fs::canonicalize(dir)
            .ok()
            .filter(|p| p.is_dir())
            .ok_or_else(|| format!("--data {dir:?} is not a directory"))?;
        args.data = Some(abs.to_string_lossy().into_owned());
    }
    let (cargo_args, env) = invocation(&args);
    let mut cmd = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    // The flags describe the run: nothing inherited from the shell.
    for var in [
        "KRISHIV_QERROR_DIR",
        "KRISHIV_QERROR_SCALE",
        "KRISHIV_QERROR_SUITES",
    ] {
        cmd.env_remove(var);
    }
    cmd.current_dir(root).args(cargo_args).envs(env);
    match cmd.status() {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!("the q-error report failed: {status}")),
        Err(e) => Err(format!("cannot run cargo: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_flags_run_both_suites_on_generated_data() {
        let args = parse_args(&[]).unwrap();
        assert_eq!(args, QerrorArgs::default());
        let (cargo, env) = invocation(&args);
        assert_eq!(
            &cargo[..5],
            ["test", "-p", "krishiv-optimizer", "--test", "it"]
        );
        assert!(cargo.contains(&"--nocapture".to_owned()));
        assert!(env.is_empty());
    }

    #[test]
    fn flags_become_the_reports_settings() {
        let args = parse_args(&[
            "--data",
            "/data/tpch",
            "--scale",
            "5000",
            "--suite",
            "tpch",
            "--suite",
            "tpch",
        ])
        .unwrap();
        assert_eq!(args.suites, ["tpch"], "a repeated suite counts once");
        let (_, env) = invocation(&args);
        assert_eq!(
            env,
            [
                ("KRISHIV_QERROR_DIR".to_owned(), "/data/tpch".to_owned()),
                ("KRISHIV_QERROR_SCALE".to_owned(), "5000".to_owned()),
                ("KRISHIV_QERROR_SUITES".to_owned(), "tpch".to_owned()),
            ]
        );
    }

    #[test]
    fn bad_command_lines_are_usage_errors() {
        for (flags, want) in [
            (vec!["--data"], "needs a directory"),
            (vec!["--scale"], "needs a number"),
            (vec!["--scale", "0"], "positive"),
            (vec!["--scale", "many"], "positive"),
            (vec!["--suite", "clickbench"], "tpch or tpcds"),
            (vec!["--suite"], "needs tpch or tpcds"),
            (vec!["--fast"], "unknown flag"),
        ] {
            let e = parse_args(&flags).unwrap_err();
            assert!(e.contains(want), "{flags:?}: {e}");
        }
    }

    #[test]
    fn a_missing_data_directory_is_reported_before_running() {
        let e = run_qerror(Path::new("."), &["--data", "/nonexistent/krishiv-qerror"]).unwrap_err();
        assert!(e.contains("not a directory"), "{e}");
    }
}
