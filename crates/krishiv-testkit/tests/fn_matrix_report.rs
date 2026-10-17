// SPDX-License-Identifier: Apache-2.0
//! CLI test for `fn-matrix-report` (P2-T106a): fixtures dir, `--results`, stale-id rejection.
#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

type R = Result<(), Box<dyn std::error::Error>>;

const MANIFEST: &str = r#"{"format_version":1,"spark_version":"4.2.0","base_conf":{},
  "show_functions":["abs"],
  "functions":[{"name":"abs","family":"math","registry":"function","cases":1}],
  "families":[{"family":"math","shard":"cases/math.jsonl","functions":1,"cases":1}]}"#;
const SHARD: &str = concat!(
    r#"{"id":"abs#1","function":"abs","kind":"example0","sql":"SELECT abs(-1)","conf":{},"#,
    r#""ansi_on":{"schema":[{"name":"abs(-1)","type":"int","nullable":false}],"rows":[["1"]],"#,
    r#""typed_rows":[[1]],"render":["cast"],"ordered":false,"truncated":false},"#,
    r#""ansi_off":{"same_as":"ansi_on"}}"#,
    "\n"
);

fn fixture_dir(name: &str) -> Result<PathBuf, std::io::Error> {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(dir.join("cases"))?;
    std::fs::write(dir.join("manifest.json"), MANIFEST)?;
    std::fs::write(dir.join("cases/math.jsonl"), SHARD)?;
    Ok(dir)
}

fn report(args: &[&str]) -> Result<(bool, String, String), std::io::Error> {
    let out = Command::new(env!("CARGO_BIN_EXE_fn-matrix-report"))
        .args(args)
        .output()?;
    Ok((
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    ))
}

#[test]
fn renders_zero_and_partial_coverage() -> R {
    let dir = fixture_dir("fn-matrix-cli")?;
    let fixtures = dir.to_str().ok_or("non-UTF-8 path")?;
    let (ok, md, err) = report(&["--fixtures", fixtures])?;
    assert!(ok, "{err}");
    assert!(
        md.contains("engine `krishiv (no results)`): 0% green"),
        "{md}"
    );

    let results = dir.join("results.json");
    std::fs::write(
        &results,
        r#"{"engine":"k","passed":["abs#1/ansi_on"],"failed":["abs#1/ansi_off"]}"#,
    )?;
    let res = results.to_str().ok_or("non-UTF-8 path")?;
    let (ok, md, err) = report(&["--fixtures", fixtures, "--results", res])?;
    assert!(ok, "{err}");
    assert!(md.contains("engine `k`): 50.0% green"), "{md}");
    Ok(())
}

#[test]
fn rejects_stale_results_and_bad_flags() -> R {
    let dir = fixture_dir("fn-matrix-cli-stale")?;
    let fixtures = dir.to_str().ok_or("non-UTF-8 path")?;
    let results = dir.join("stale.json");
    std::fs::write(&results, r#"{"engine":"k","passed":["gone#1/ansi_on"]}"#)?;
    let res = results.to_str().ok_or("non-UTF-8 path")?;
    let (ok, _, err) = report(&["--fixtures", fixtures, "--results", res])?;
    assert!(!ok && err.contains("stale results"), "{err}");
    let (ok, _, err) = report(&["--bogus"])?;
    assert!(!ok && err.contains("usage"), "{err}");
    Ok(())
}
