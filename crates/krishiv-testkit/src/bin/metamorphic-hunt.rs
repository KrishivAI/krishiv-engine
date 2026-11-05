// SPDX-License-Identifier: Apache-2.0
//! Hunts for TLP and NoREC violations on `krishiv-session` for a time budget (P2-T110b); the nightly run.
//!
//! `metamorphic-hunt --data DIR [--seed N] [--seconds N]` draws batches of 25 predicates from
//! consecutive seeds until the budget ends, prints each violation reduced to its smallest
//! predicate, and the line `queries_checked_per_s`. Exits 1 on a violation, 2 on a usage error.
#![forbid(unsafe_code)]

use std::io::Write as _;
use std::process::ExitCode;

use krishiv_testkit::metamorphic::hunt_cli;

fn main() -> ExitCode {
    match hunt_cli(std::env::args().skip(1), &mut std::io::stdout().lock()) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            drop(writeln!(std::io::stderr(), "metamorphic-hunt: {e}"));
            ExitCode::from(2)
        }
    }
}
