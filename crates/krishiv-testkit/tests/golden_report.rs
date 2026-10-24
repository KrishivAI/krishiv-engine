// SPDX-License-Identifier: Apache-2.0
//! CLI test for `golden-report`: the `--engine` flag and the parse number of
//! the `krishiv-sql` engine on the DQL statements.
#![forbid(unsafe_code)]

use std::path::Path;
use std::process::Command;

type R = Result<(), Box<dyn std::error::Error>>;

fn report(args: &[&str]) -> Result<(bool, String, String), std::io::Error> {
    let out = Command::new(env!("CARGO_BIN_EXE_golden-report"))
        .args(args)
        .output()?;
    Ok((
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    ))
}

#[test]
fn sql_parse_engine_reports_the_dql_parse_rate() -> R {
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("golden-sql-parse.json");
    let (ok, stdout, stderr) = report(&["--engine", "sql-parse", "--out", &out.to_string_lossy()])?;
    assert!(ok, "{stderr}");
    assert!(
        stdout.contains("krishiv-sql parse-only (DQL subset)"),
        "{stdout}"
    );
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&out)?)?;
    let parse = &json["totals"]["parse"];
    let (passed, total) = (
        parse["passed"].as_u64().ok_or("passed")?,
        parse["total"].as_u64().ok_or("total")?,
    );
    assert!(total > 13_000, "{parse}");
    assert!(passed * 1000 >= total * 999, "{parse}");
    Ok(())
}

#[test]
fn reject_all_engine_is_the_default_stub() -> R {
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("golden-reject-all.json");
    let (ok, stdout, stderr) =
        report(&["--engine", "reject-all", "--out", &out.to_string_lossy()])?;
    assert!(ok, "{stderr}");
    assert!(stdout.contains("reject-all"), "{stdout}");
    Ok(())
}

#[test]
fn unknown_engine_is_a_usage_error() -> R {
    let (ok, _, stderr) = report(&["--engine", "spark"])?;
    assert!(!ok);
    assert!(stderr.contains("--engine reject-all|sql-parse"), "{stderr}");
    Ok(())
}

#[test]
fn ratchet_flags_are_checked() -> R {
    let (ok, _, stderr) = report(&["--ratchet", "check"])?;
    assert!(!ok);
    assert!(
        stderr.contains("--ratchet needs --engine session"),
        "{stderr}"
    );
    let (ok, _, stderr) = report(&["--engine", "session", "--ratchet", "maybe"])?;
    assert!(!ok);
    assert!(stderr.contains("[--ratchet check|update]"), "{stderr}");
    Ok(())
}

#[test]
fn reject_all_writes_every_failing_case() -> R {
    let cases = Path::new(env!("CARGO_TARGET_TMPDIR")).join("golden-reject-all-cases.json");
    let (ok, _, stderr) = report(&[
        "--engine",
        "reject-all",
        "--cases",
        &cases.to_string_lossy(),
        "--known-failures",
        "unused.json",
    ])?;
    assert!(ok, "{stderr}");
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cases)?)?;
    let first = &json["array"][0];
    assert_eq!(first["index"], 0);
    assert!(
        first["detail"]
            .as_str()
            .is_some_and(|d| d.contains("RejectAllEngine")),
        "{first}"
    );
    Ok(())
}
