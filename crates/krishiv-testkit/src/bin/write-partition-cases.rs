// SPDX-License-Identifier: Apache-2.0
//! Writes every case of a partitioned-write case file with Krishiv, one directory per case:
//! `write-partition-cases CASES.json OUT_DIR`. `tests/scripts/gen_partitioned_write_goldens.py` runs it, then reads
//! the directories with Spark 4.2.0 (P2-T082b acceptance).

use std::io::Write as _;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match krishiv_testkit::partitioned_writes::cli(&args) {
        Ok(dirs) => {
            let mut out = std::io::stdout().lock();
            for d in dirs {
                if writeln!(out, "{d}").is_err() {
                    return ExitCode::FAILURE;
                }
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            let _ = writeln!(std::io::stderr(), "write-partition-cases: {e}");
            ExitCode::FAILURE
        }
    }
}
