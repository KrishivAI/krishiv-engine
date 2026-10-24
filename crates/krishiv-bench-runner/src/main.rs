// SPDX-License-Identifier: Apache-2.0
//! `krishiv-bench-runner --data DIR [--threads N] [--memory-limit 12g] [--conf k=v]...` (or `--build-info`): see the library docs for
//! the stdin/stdout protocol.

#![forbid(unsafe_code)]

use std::io::Write;
use std::process::ExitCode;

use krishiv_bench_runner::{Runner, RunnerConfig, serve};

#[cfg(not(feature = "heap-stats"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// A measurement build: mimalloc with its live heap counted.
#[cfg(feature = "heap-stats")]
#[global_allocator]
static GLOBAL: krishiv_alloc::heap::Counting<mimalloc::MiMalloc> =
    krishiv_alloc::heap::Counting(mimalloc::MiMalloc);

fn main() -> ExitCode {
    // First, so a CPU older than the build targets gets a message instead of SIGILL.
    // `run` holds everything else (see `krishiv_common::cpu`).
    krishiv_common::cpu::check_or_exit();
    // Allocator policy is this binary's (it owns the global allocator): with a
    // configured limit that is not generous, mimalloc purges promptly (set
    // before `main`, see `krishiv_alloc`), and under memory pressure threads
    // return freed memory.
    krishiv_memory::set_release_hook(krishiv_alloc::install());
    run()
}

#[inline(never)]
fn run() -> ExitCode {
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
