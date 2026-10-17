// SPDX-License-Identifier: Apache-2.0
//! Runs the Spark `SQLQueryTestSuite` golden files against an engine and writes the report
//! (three numbers, per-file denominators, exclusion manifest) as JSON.
//!
//! `cargo run -p krishiv-testkit --bin golden-report -- [--tag TAG] [--jdk 17|21] [--manifest FILE] [--out FILE]`
//! (or `cargo xtask golden`). The default engine is the reject-all stub (0/0/0 with the real
//! denominators); `--engine sql-parse` measures the `krishiv-sql` parser on the DQL statements
//! ([`krishiv_testkit::golden::sql_parse`]).
#![forbid(unsafe_code)]

use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use krishiv_testkit::golden::manifest::ExclusionManifest;
use krishiv_testkit::golden::run::{RejectAllEngine, run_with};
use krishiv_testkit::golden::sql_parse::{ParseOnlyEngine, in_dql_scope};
use krishiv_testkit::golden::{DEFAULT_TAG, GoldenSet, Jdk, report};

const USAGE: &str = "usage: golden-report [--tag TAG] [--jdk 17|21] [--manifest FILE] [--out FILE] [--engine reject-all|sql-parse]";

fn run_cli() -> Result<(), String> {
    let mut tag = DEFAULT_TAG.to_owned();
    let mut manifest_path = ExclusionManifest::default_path();
    let mut out = None;
    let mut jdk = Jdk::default();
    let mut sql_parse = false;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| USAGE.to_owned())?;
        match flag.as_str() {
            "--tag" => tag = value,
            "--manifest" => manifest_path = PathBuf::from(value),
            "--jdk" => {
                jdk = match value.as_str() {
                    "17" => Jdk::Java17,
                    "21" => Jdk::Java21,
                    _ => return Err(USAGE.to_owned()),
                }
            }
            "--out" => out = Some(PathBuf::from(value)),
            "--engine" => {
                sql_parse = match value.as_str() {
                    "reject-all" => false,
                    "sql-parse" => true,
                    _ => return Err(USAGE.to_owned()),
                }
            }
            _ => return Err(USAGE.to_owned()),
        }
    }
    let manifest = ExclusionManifest::load(&manifest_path).map_err(|e| e.to_string())?;
    let dir = GoldenSet::tag_dir(&tag);
    let set = GoldenSet::scan_for(&dir, jdk).map_err(|e| format!("{}: {e}", dir.display()))?;
    let result = if sql_parse {
        run_with(
            &mut ParseOnlyEngine::default(),
            &set,
            &manifest,
            &in_dql_scope,
        )
    } else {
        run_with(&mut RejectAllEngine, &set, &manifest, &|_| true)
    }
    .map_err(|e| e.to_string())?;
    let json = report::to_json_string(&report::to_json(&result, &tag, &manifest))
        .map_err(|e| e.to_string())?;
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(report::summary(&result).as_bytes())
        .map_err(|e| e.to_string())?;
    if let Some(path) = out {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        std::fs::write(&path, json).map_err(|e| format!("{}: {e}", path.display()))?;
        writeln!(stdout, "report written to {}", path.display()).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn main() -> ExitCode {
    match run_cli() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            let _ = writeln!(std::io::stderr().lock(), "{msg}");
            ExitCode::FAILURE
        }
    }
}
