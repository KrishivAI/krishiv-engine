// SPDX-License-Identifier: Apache-2.0
//! Replays an expression-differential fixture (P2-T101) on `krishiv-session`; the nightly run.
//!
//! `expr-diff --fixture FILE [--known FILE] [--out FILE]` prints the minimal mismatches, one per
//! line (`ansi|sql`, then Spark's answer and the engine's), writes them as JSON to `--out`, and
//! exits 1 when one is not in the `--known` list. The fixture comes from
//! `tests/scripts/gen_expr_differential.py --roots 10000`.
#![forbid(unsafe_code)]

use std::process::ExitCode;

use krishiv_testkit::expr_diff::cli;

fn main() -> ExitCode {
    let (mut out, mut err) = (std::io::stdout().lock(), std::io::stderr().lock());
    match cli(std::env::args().skip(1), &mut out, &mut err) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            drop(std::io::Write::write_fmt(
                &mut err,
                format_args!("expr-diff: {e}\n"),
            ));
            ExitCode::from(2)
        }
    }
}
