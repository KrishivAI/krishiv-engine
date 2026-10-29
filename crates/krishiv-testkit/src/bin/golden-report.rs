// SPDX-License-Identifier: Apache-2.0
//! Runs the Spark `SQLQueryTestSuite` golden files against an engine and writes the report
//! (three numbers, per-file denominators, exclusion manifest) as JSON.
//!
//! `cargo run -p krishiv-testkit --bin golden-report -- [--tag TAG] [--jdk 17|21] [--manifest FILE]
//! [--out FILE] [--engine reject-all|sql-parse|session] [--cases FILE] [--ratchet check|update]
//! [--known-failures FILE]` (or `cargo xtask golden`).
//!
//! * The default engine is the reject-all stub (0/0/0 with the real denominators);
//!   `--engine sql-parse` measures the `krishiv-sql` parser on the DQL statements
//!   ([`krishiv_testkit::golden::sql_parse`]); `--engine session` runs every statement through
//!   `krishiv-session` ([`krishiv_testkit::golden::session_engine`]).
//! * `--cases FILE` writes every statement that did not pass, with why.
//! * `--ratchet check` compares the run with the frozen baseline
//!   (`tests/spark-golden/known-failures.json` with its per-category and per-file directories)
//!   and fails on any difference;
//!   `--ratchet update` rewrites the baseline: passing statements leave it, new failures are
//!   classified by its rules (the command fails, writing nothing, if a failure matches no rule).
#![forbid(unsafe_code)]

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use krishiv_testkit::golden::known_failures::KnownFailures;
use krishiv_testkit::golden::manifest::ExclusionManifest;
use krishiv_testkit::golden::run::{Engine, RejectAllEngine, RunResult, run_with};
use krishiv_testkit::golden::session_engine::SessionEngine;
use krishiv_testkit::golden::sql_parse::{ParseOnlyEngine, in_dql_scope};
use krishiv_testkit::golden::{DEFAULT_TAG, GoldenSet, Jdk, report};

const USAGE: &str = "usage: golden-report [--tag TAG] [--jdk 17|21] [--manifest FILE] [--out FILE] \
[--engine reject-all|sql-parse|session] [--cases FILE] [--ratchet check|update] [--reclassify CATEGORY] [--known-failures FILE]";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ratchet {
    Off,
    Check,
    Update,
}

fn ratchet(
    mode: Ratchet,
    reclassify: Option<&str>,
    path: &Path,
    set: &GoldenSet,
    result: &RunResult,
) -> Result<String, String> {
    match mode {
        Ratchet::Check => KnownFailures::check_file(path, result),
        Ratchet::Update => KnownFailures::update_file(path, result, set, reclassify),
        Ratchet::Off => Ok(String::new()),
    }
}

fn run_cli() -> Result<(), String> {
    let mut tag = DEFAULT_TAG.to_owned();
    let mut manifest_path = ExclusionManifest::default_path();
    let mut known_path = KnownFailures::default_path();
    let mut out = None;
    let mut jdk = Jdk::default();
    let mut engine_name = "reject-all".to_owned();
    let mut cases_out = None;
    let mut mode = Ratchet::Off;
    let mut reclassify: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| USAGE.to_owned())?;
        match flag.as_str() {
            "--tag" => tag = value,
            "--reclassify" => reclassify = Some(value),
            "--manifest" => manifest_path = PathBuf::from(value),
            "--known-failures" => known_path = PathBuf::from(value),
            "--jdk" => {
                jdk = match value.as_str() {
                    "17" => Jdk::Java17,
                    "21" => Jdk::Java21,
                    _ => return Err(USAGE.to_owned()),
                }
            }
            "--out" => out = Some(PathBuf::from(value)),
            "--engine" => match value.as_str() {
                "reject-all" | "sql-parse" | "session" => engine_name = value,
                _ => return Err(USAGE.to_owned()),
            },
            "--cases" => cases_out = Some(PathBuf::from(value)),
            "--ratchet" => {
                mode = match value.as_str() {
                    "check" => Ratchet::Check,
                    "update" => Ratchet::Update,
                    _ => return Err(USAGE.to_owned()),
                }
            }
            _ => return Err(USAGE.to_owned()),
        }
    }
    if mode != Ratchet::Off && engine_name != "session" {
        return Err("--ratchet needs --engine session".to_owned());
    }
    if reclassify.is_some() && mode != Ratchet::Update {
        return Err("--reclassify needs --ratchet update".to_owned());
    }
    let manifest = ExclusionManifest::load(&manifest_path).map_err(|e| e.to_string())?;
    let dir = GoldenSet::tag_dir(&tag);
    let set = GoldenSet::scan_for(&dir, jdk).map_err(|e| format!("{}: {e}", dir.display()))?;
    let result = match engine_name.as_str() {
        "sql-parse" => run_with(
            &mut ParseOnlyEngine::default(),
            &set,
            &manifest,
            &in_dql_scope,
        ),
        "session" => {
            let mut engine = SessionEngine::new().map_err(|e| e.message)?;
            run_with(&mut engine as &mut dyn Engine, &set, &manifest, &|_| true)
        }
        _ => run_with(&mut RejectAllEngine, &set, &manifest, &|_| true),
    }
    .map_err(|e| e.to_string())?;
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(report::summary(&result).as_bytes())
        .map_err(|e| e.to_string())?;
    if let Some(path) = &cases_out {
        let text =
            report::to_json_string(&report::cases_json(&result)).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    if let Some(path) = out {
        let json = report::to_json_string(&report::to_json(&result, &tag, &manifest))
            .map_err(|e| e.to_string())?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        std::fs::write(&path, json).map_err(|e| format!("{}: {e}", path.display()))?;
        writeln!(stdout, "report written to {}", path.display()).map_err(|e| e.to_string())?;
    }
    if mode != Ratchet::Off {
        let msg = ratchet(mode, reclassify.as_deref(), &known_path, &set, &result)?;
        writeln!(stdout, "{msg}").map_err(|e| e.to_string())?;
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
