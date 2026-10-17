// SPDX-License-Identifier: Apache-2.0
//! Renders the Spark function coverage matrix (P2-T106a) as Markdown for STATUS.md and CI.
//!
//! `cargo run -p krishiv-testkit --bin fn-matrix-report -- [--fixtures DIR] [--results FILE] [--out FILE]`
//!
//! `--results` is engine output `{"engine": "...", "passed": [ids], "failed": [ids]}`; without it every
//! case counts as not run, so the matrix reports 0% green.
#![forbid(unsafe_code)]

use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use krishiv_testkit::fn_matrix::{CoverageReport, EngineResults, FnMatrix};

const USAGE: &str = "usage: fn-matrix-report [--fixtures DIR] [--results FILE] [--out FILE]";

fn run() -> Result<(), String> {
    let mut fixtures = FnMatrix::default_dir();
    let mut results = None;
    let mut out = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || {
            args.next()
                .map(PathBuf::from)
                .ok_or_else(|| USAGE.to_owned())
        };
        match flag.as_str() {
            "--fixtures" => fixtures = value()?,
            "--results" => results = Some(value()?),
            "--out" => out = Some(value()?),
            _ => return Err(USAGE.to_owned()),
        }
    }
    let matrix = FnMatrix::load(&fixtures).map_err(|e| e.to_string())?;
    let results = match results {
        Some(p) => EngineResults::parse(
            &std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?,
        )
        .map_err(|e| e.to_string())?,
        None => EngineResults {
            engine: "krishiv (no results)".to_owned(),
            ..EngineResults::default()
        },
    };
    results.check_ids(&matrix).map_err(|e| e.to_string())?;
    let md =
        CoverageReport::compute(&matrix, &results.engine, |c| results.outcome(c)).render_markdown();
    match out {
        Some(p) => std::fs::write(&p, md).map_err(|e| format!("{}: {e}", p.display())),
        None => std::io::stdout()
            .lock()
            .write_all(md.as_bytes())
            .map_err(|e| e.to_string()),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            let _ = writeln!(std::io::stderr().lock(), "{msg}");
            ExitCode::FAILURE
        }
    }
}
