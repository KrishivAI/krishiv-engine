// SPDX-License-Identifier: Apache-2.0
//! The OS sandbox for Krishiv's Python UDF workers (P3-T014b; design
//! `design/P3-T014-connect-udfs-sandbox.md` §4 in the internal docs).
//!
//! A cloudpickled UDF is client code by design. In server mode each worker process runs it confined, with no
//! root and no user namespace (both are unavailable on common hosts and in Docker's default profile):
//! - a cleared environment and the interpreter's isolated mode;
//! - **Landlock**: execute only the interpreter, read only the listed paths, write only a private directory,
//!   TCP only to listed ports, signals and abstract sockets scoped to the worker;
//! - **seccomp**: no `socket()` (the worker uses only the descriptors the server hands it), no `fork` or
//!   `vfork` and no `clone` without `CLONE_THREAD`, no tracing, mounts, namespaces, keyrings, BPF or kernel
//!   modules, no signals to other processes;
//! - **rlimits**: address space, file size, open files, no core files, optional CPU seconds;
//! - a parent-death signal, so a worker never outlives the server.
//!
//! The server starts its own executable as a **trampoline** (a hidden subcommand that calls
//! [`trampoline_main`] with [`SandboxSpec::to_args`]); the trampoline confines itself and `exec`s the
//! interpreter, which starts confined because Landlock and seccomp survive `execve`. No `pre_exec` hook
//! runs after `fork` in the server.
//!
//! [`probe`] reports what the kernel can enforce; [`Capabilities::require`] turns a missing layer into
//! `KRISHIV_UDF_SANDBOX_UNAVAILABLE`. The pool decides what to do with it (`krishiv.udf.sandbox=off`).
//!
//! **Layer:** L1, depends on `krishiv-common` only among Krishiv crates. `#![forbid(unsafe_code)]`: the
//! `landlock` and `seccompiler` crates hold the system-call `unsafe`.
#![forbid(unsafe_code)]

mod error;
mod probe;
mod spec;

#[cfg(target_os = "linux")]
mod enter;
#[cfg(target_os = "linux")]
mod fs_rules;
#[cfg(target_os = "linux")]
mod seccomp;

pub use error::SandboxError;
pub use probe::{Capabilities, probe};
pub use spec::{DEFAULT_FILE_SIZE, DEFAULT_OPEN_FILES, Limits, Network, SandboxSpec};

use std::ffi::OsString;
use std::io::Write;
use std::process::ExitCode;

use krishiv_common::error::KrishivError;

/// The exit status of a trampoline that could not enter the sandbox or start the interpreter. Its standard
/// error (the worker's log pipe) holds the error, starting with its class.
pub const EXIT_SANDBOX_FAILED: u8 = 125;

/// Enters the sandbox described by `spec` and replaces the process with its interpreter.
///
/// Returns only on failure: on success the process is the confined interpreter. Call it only in a process
/// that exists to become the worker; it restricts the calling thread for good.
#[cfg(target_os = "linux")]
pub fn enter_and_exec(spec: &SandboxSpec) -> SandboxError {
    enter::enter_and_exec(spec)
}

/// Applies every layer of `spec` to the calling thread and the process's limits, without `exec`.
///
/// Landlock, seccomp and `PR_SET_NO_NEW_PRIVS` bind the calling thread and whatever it later starts or
/// executes; threads that already exist stay unconfined. So it confines a whole process only when called
/// before any other thread starts. [`enter_and_exec`] is this followed by `exec`; the tests use `confine`
/// directly to check the layers from Rust in a process that then exits normally.
///
/// # Errors
/// [`SandboxError::Unavailable`] when a layer is missing or cannot be applied. Layers applied before the
/// failure stay applied.
#[cfg(target_os = "linux")]
pub fn confine(spec: &SandboxSpec) -> Result<(), SandboxError> {
    enter::confine(spec)
}

/// Applies every layer of `spec` to the calling thread.
///
/// # Errors
/// The sandbox needs Linux, so on this platform it always fails with `KRISHIV_UDF_SANDBOX_UNAVAILABLE`.
#[cfg(not(target_os = "linux"))]
pub fn confine(_spec: &SandboxSpec) -> Result<(), SandboxError> {
    Err(SandboxError::Unavailable {
        reason: "the UDF sandbox needs Linux".to_owned(),
    })
}

/// Enters the sandbox described by `spec` and replaces the process with its interpreter.
///
/// The sandbox needs Linux, so on this platform it always fails with `KRISHIV_UDF_SANDBOX_UNAVAILABLE`.
#[cfg(not(target_os = "linux"))]
pub fn enter_and_exec(_spec: &SandboxSpec) -> SandboxError {
    SandboxError::Unavailable {
        reason: "the UDF sandbox needs Linux".to_owned(),
    }
}

/// The trampoline's whole program: decodes `args` (made by [`SandboxSpec::to_args`]), enters the sandbox and
/// `exec`s the interpreter. On failure it writes the error to standard error and returns
/// [`EXIT_SANDBOX_FAILED`].
pub fn trampoline_main(args: impl IntoIterator<Item = OsString>) -> ExitCode {
    let error = match SandboxSpec::from_args(args) {
        Ok(spec) => enter_and_exec(&spec),
        Err(e) => e,
    };
    let error = KrishivError::from(&error);
    // Standard error is the worker's log pipe; if it is gone there is no one left to tell.
    let _ = writeln!(std::io::stderr(), "{error}");
    ExitCode::from(EXIT_SANDBOX_FAILED)
}
