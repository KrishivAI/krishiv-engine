// SPDX-License-Identifier: Apache-2.0
//! `krishiv-bench-runner --data DIR [--threads N] [--memory-limit 12g]` (or `--build-info`): see the library docs for
//! the stdin/stdout protocol.

#![forbid(unsafe_code)]

use std::io::Write;
use std::process::ExitCode;

use krishiv_bench_runner::{Runner, RunnerConfig, serve};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() -> ExitCode {
    // First, so a CPU older than the build targets gets a message instead of SIGILL.
    krishiv_common::cpu::check_or_exit();
    if std::env::args().nth(1).as_deref() == Some("--build-info") {
        return match writeln!(std::io::stdout(), "{}", krishiv_bench_runner::BUILD_INFO) {
            Ok(()) => ExitCode::SUCCESS,
            Err(_) => ExitCode::FAILURE,
        };
    }
    let result = RunnerConfig::from_args(std::env::args().skip(1))
        .map_err(|e| e.to_string())
        .and_then(|config| Runner::new(&config).map_err(|e| e.to_string()))
        .and_then(|runner| {
            serve(
                &runner,
                std::io::stdin().lock(),
                &mut std::io::stdout().lock(),
            )
            .map_err(|e| e.to_string())
        });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            // Best effort: a closed stderr leaves nothing to report to.
            let _ = writeln!(std::io::stderr(), "krishiv-bench-runner: {e}");
            ExitCode::FAILURE
        }
    }
}
