// SPDX-License-Identifier: Apache-2.0
//! Krishiv's isolated Python UDF worker pool (P3-T014, PR-26b; design
//! `design/P3-T014-connect-udfs-sandbox.md` §3 in krishiv-internal).
//!
//! In server mode, Python UDFs run in **worker processes**, never in the server: each worker serves one
//! principal and one session, runs inside the OS sandbox of `krishiv-udf-sandbox` (started through the
//! server binary's hidden [`WORKER_SUBCOMMAND`]), and exchanges data with the server through memfds (wire
//! `krishiv.udf.wire.v1`, [`wire`]). The worker program is Python code embedded in this crate
//! (`python/krishiv/_udf`), so worker interpreters need `pyspark-client`, `pyarrow` and `pandas` but not the
//! `krishiv` wheel.
//!
//! - [`Pool`]: the workers, keyed by [`Owner`] and interpreter, with caps, reuse and an idle timeout.
//! - [`SessionUdfs`]: a session's registered UDFs as an executor `UdfRuntime`.
//! - [`Interpreter`]: the administrator's interpreters, recorded at startup.
//! - [`run_trampoline_if_requested`]: the binary's dispatch of the hidden subcommand.
#![forbid(unsafe_code)]

use std::ffi::OsString;
use std::process::ExitCode;

#[cfg(target_os = "linux")]
mod interpreter;
#[cfg(target_os = "linux")]
mod package;
#[cfg(target_os = "linux")]
mod pool;
#[cfg(target_os = "linux")]
mod runtime;
#[cfg(target_os = "linux")]
pub mod wire;
#[cfg(target_os = "linux")]
mod worker;

#[cfg(target_os = "linux")]
pub use interpreter::{Interpreter, choose};
#[cfg(target_os = "linux")]
pub use pool::{Launcher, Owner, Pool, PoolConfig};
#[cfg(target_os = "linux")]
pub use runtime::SessionUdfs;
#[cfg(target_os = "linux")]
pub use worker::{PythonPayload, decode_result};

/// The hidden subcommand of the server's binary that runs the sandbox trampoline.
pub const WORKER_SUBCOMMAND: &str = "__udf-worker";

/// Runs the sandbox trampoline when the process was started as a UDF worker (`<binary> __udf-worker ...`)
/// and returns its exit code; `None` otherwise. A binary that hosts a [`Pool`] calls this first thing in
/// `main`, before it starts any thread or runtime: the trampoline confines the calling thread and then
/// replaces the process.
pub fn run_trampoline_if_requested() -> Option<ExitCode> {
    let mut args = std::env::args_os();
    let _program = args.next();
    if args.next().as_deref() != Some(std::ffi::OsStr::new(WORKER_SUBCOMMAND)) {
        return None;
    }
    let rest: Vec<OsString> = args.collect();
    Some(krishiv_udf_sandbox::trampoline_main(rest))
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_normal_start_is_not_the_trampoline() {
        // The test binary's arguments never start with the subcommand.
        assert!(super::run_trampoline_if_requested().is_none());
    }
}
