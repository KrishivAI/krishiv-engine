// SPDX-License-Identifier: Apache-2.0
//! Entering the sandbox, then `exec` of the interpreter (note §4.2, the order at its end).
//!
//! 1. a new session (`setsid`), so the worker has no controlling terminal and gets no terminal signals sent
//!    to the server's process group; the parent-death signal, then the check that the parent is still the
//!    server;
//! 2. `PR_SET_NO_NEW_PRIVS` and the rlimits;
//! 3. Landlock;
//! 4. seccomp, last, because it forbids what the earlier steps need;
//! 5. `execve` of the interpreter with a cleared environment.
//!
//! Landlock and seccomp both survive `execve`, so the interpreter starts confined. File descriptors above 2
//! are closed by the worker module before it reads any client code (`os.closerange`), not here: closing an
//! inherited descriptor by number needs `unsafe` in Rust, and the worker runs only trusted code (the
//! interpreter's startup and Krishiv's worker module) before that point.

use std::os::unix::process::CommandExt;
use std::process::Command;

use rustix::process::{Pid, Resource, Rlimit, Signal};

use crate::seccomp::{self, FilterSpec};
use crate::{SandboxError, SandboxSpec, fs_rules, probe};

/// Enters the sandbox described by `spec` and replaces the process with the interpreter. Returns only on
/// failure; on success the process image is the confined interpreter.
///
/// Call it from a process that exists to become the worker (the trampoline), never from the server: it
/// restricts the calling thread for good and changes process-wide limits.
pub(crate) fn enter_and_exec(spec: &SandboxSpec) -> SandboxError {
    match confine(spec) {
        Ok(()) => {
            let error = Command::new(&spec.interpreter)
                .args(&spec.args)
                .env_clear()
                .envs(spec.env.iter().map(|(k, v)| (k, v)))
                .current_dir(&spec.private_dir)
                .exec();
            SandboxError::unavailable(format!(
                "starting {} in the sandbox failed: {error}",
                spec.interpreter.display()
            ))
        }
        Err(e) => e,
    }
}

/// Confines the calling thread as `enter_and_exec` does, without the `exec` (see
/// `crate::confine_current_thread`).
pub(crate) fn confine(spec: &SandboxSpec) -> Result<(), SandboxError> {
    probe::probe().require(spec)?;
    rustix::process::setsid()
        .map_err(|e| SandboxError::unavailable(format!("setsid failed: {e}")))?;
    rustix::process::set_parent_process_death_signal(Some(Signal::KILL))
        .map_err(|e| SandboxError::unavailable(format!("PR_SET_PDEATHSIG failed: {e}")))?;
    if let Some(expected) = spec.parent_pid {
        let parent = rustix::process::getppid().map(|p| p.as_raw_nonzero().get());
        if parent.and_then(|p| u32::try_from(p).ok()) != Some(expected) {
            return Err(SandboxError::unavailable(format!(
                "the server (pid {expected}) exited before the worker started"
            )));
        }
    }
    rustix::thread::set_no_new_privs(true)
        .map_err(|e| SandboxError::unavailable(format!("PR_SET_NO_NEW_PRIVS failed: {e}")))?;
    set_limits(spec)?;
    fs_rules::restrict(spec)?;
    let own_pid = Pid::as_raw(Some(rustix::process::getpid()));
    let programs = seccomp::programs(&FilterSpec {
        own_pid: own_pid.unsigned_abs(),
        network: &spec.network,
        allow_fork: spec.allow_fork,
    })?;
    for program in &programs {
        seccompiler::apply_filter(program).map_err(|e| {
            SandboxError::unavailable(format!("installing the seccomp filter failed: {e}"))
        })?;
    }
    Ok(())
}

fn set_limits(spec: &SandboxSpec) -> Result<(), SandboxError> {
    let limits = spec.limits;
    let mut wanted = vec![
        (Resource::Core, Some(0)),
        (Resource::Nofile, Some(limits.open_files)),
    ];
    wanted.push((Resource::As, limits.address_space));
    wanted.push((Resource::Fsize, limits.file_size));
    wanted.push((Resource::Cpu, limits.cpu_seconds));
    for (resource, value) in wanted {
        let Some(value) = value else { continue };
        // An unprivileged process may only lower its hard limit, so never ask above the inherited one.
        let inherited = rustix::process::getrlimit(resource).maximum;
        let value = inherited.map_or(value, |max| value.min(max));
        rustix::process::setrlimit(
            resource,
            Rlimit {
                current: Some(value),
                maximum: Some(value),
            },
        )
        .map_err(|e| {
            SandboxError::unavailable(format!("setrlimit({resource:?}, {value}) failed: {e}"))
        })?;
    }
    Ok(())
}
