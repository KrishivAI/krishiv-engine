// SPDX-License-Identifier: Apache-2.0
//! The function-matrix cells of `+ - * / % div`, comparisons and casts run on
//! `krishiv-kernels` (P2-T021a): no cell may fail; cells not run carry a
//! reason (analysis-time errors belong to the analyzer, other constructs to
//! later tasks).
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt::Write as _;

use krishiv_testkit::fn_matrix::{FnMatrix, Mode};
use krishiv_testkit::kernel_eval::{CellOutcome, FUNCTIONS, run_cell};

type R = Result<(), Box<dyn std::error::Error>>;

#[test]
fn kernel_cells_match_spark() -> R {
    let matrix = FnMatrix::load_default()?;
    // function -> [pass, fail, not run] per mode.
    let mut tally: BTreeMap<&str, [[usize; 3]; 2]> = BTreeMap::new();
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    let mut failures = String::new();
    let (mut passed, mut failed) = (Vec::new(), Vec::new());
    for c in matrix
        .cases
        .iter()
        .filter(|c| FUNCTIONS.contains(&c.function.as_str()))
    {
        let m = usize::from(c.mode == Mode::AnsiOff);
        let t = tally.entry(c.function.as_str()).or_default();
        match run_cell(c) {
            CellOutcome::Pass => {
                t[m][0] += 1;
                passed.push(c.id.clone());
            }
            CellOutcome::Fail(why) => {
                t[m][1] += 1;
                failed.push(c.id.clone());
                writeln!(failures, "{} [{}] {}: {why}", c.id, c.mode.key(), c.sql)?;
            }
            CellOutcome::NotRun(why) => {
                t[m][2] += 1;
                *reasons.entry(why).or_default() += 1;
            }
        }
    }
    let mut report =
        String::from("function | ansi_on pass/fail/not-run | ansi_off pass/fail/not-run\n");
    for (f, [on, off]) in &tally {
        writeln!(
            report,
            "{f} | {}/{}/{} | {}/{}/{}",
            on[0], on[1], on[2], off[0], off[1], off[2]
        )?;
    }
    writeln!(report, "\nnot run:")?;
    for (why, n) in &reasons {
        writeln!(report, "{n:5} {why}")?;
    }
    let results = serde_json::json!({"engine": "krishiv-kernels (P2-T021a kernel evaluator)", "passed": passed, "failed": failed});
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    std::fs::write(dir.join("kernel-matrix-report.txt"), &report)?;
    std::fs::write(dir.join("kernel-matrix-results.json"), results.to_string())?;
    println!("{report}");
    assert!(failures.is_empty(), "{failures}");
    Ok(())
}
