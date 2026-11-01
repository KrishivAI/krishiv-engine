// SPDX-License-Identifier: Apache-2.0
//! Replays an expression-differential fixture (P2-T101) on `krishiv-session`; the nightly run.
//!
//! `expr-diff --fixture FILE [--known FILE] [--out FILE]` prints the minimal mismatches, one per
//! line (`ansi|sql`, then Spark's answer and the engine's), writes them as JSON to `--out`, and
//! exits 1 when one is not in the `--known` list. The fixture comes from
//! `tests/scripts/gen_expr_differential.py --roots 10000`.
#![forbid(unsafe_code)]

use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use krishiv_testkit::expr_diff::{Fixture, SessionEvaluator, parse_known, ratchet, run};

const USAGE: &str = "usage: expr-diff --fixture FILE [--known FILE] [--out FILE]";

fn go() -> Result<bool, String> {
    let (mut fixture, mut known, mut out) = (None, None, None);
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let v = args
            .next()
            .map(PathBuf::from)
            .ok_or_else(|| USAGE.to_owned())?;
        match flag.as_str() {
            "--fixture" => fixture = Some(v),
            "--known" => known = Some(v),
            "--out" => out = Some(v),
            _ => return Err(USAGE.to_owned()),
        }
    }
    let path = fixture.ok_or_else(|| USAGE.to_owned())?;
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let fixture = Fixture::parse(&text)?;
    let mut ev = SessionEvaluator::new(&fixture.setup)?;
    let found = run(&fixture, &mut ev);
    let mut stdout = std::io::stdout().lock();
    for m in &found {
        writeln!(
            stdout,
            "{}\n  spark: {}\n  got:   {}",
            m.key(),
            m.spark,
            m.got
        )
        .map_err(|e| e.to_string())?;
    }
    let known = match known {
        Some(p) => {
            parse_known(&std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?)?
        }
        None => Vec::new(),
    };
    let r = ratchet(&found, &known);
    writeln!(
        std::io::stderr(),
        "{} cases, {} minimal mismatches, {} new, {} fixed",
        fixture.cases.len(),
        found.len(),
        r.new.len(),
        r.fixed.len()
    )
    .map_err(|e| e.to_string())?;
    if let Some(p) = out {
        let rows: Vec<serde_json::Value> = found
            .iter()
            .map(|m| {
                serde_json::json!({
                    "key": m.key(), "spark": m.spark.to_string(), "got": m.got.to_string(),
                    "new": r.new.contains(&m.key()),
                })
            })
            .collect();
        let text = serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())?;
        std::fs::write(&p, text).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    Ok(r.new.is_empty())
}

fn main() -> ExitCode {
    match go() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            drop(writeln!(std::io::stderr(), "expr-diff: {e}"));
            ExitCode::from(2)
        }
    }
}
