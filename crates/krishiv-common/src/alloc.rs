// SPDX-License-Identifier: Apache-2.0
//! Allocator settings that must be in place before the allocator starts.
//!
//! mimalloc 3 keeps freed memory committed for `purge_delay` (1 s by
//! default) and commits its arenas eagerly on Linux. A query engine frees
//! large column buffers on one thread that another thread allocated, at a
//! high rate, so the process's resident set ran 2-3x above the bytes the
//! memory pool tracks (TPC-H SF1 at a 200 MiB limit: 650 MiB resident for
//! about 200 MiB live). [`ALLOCATOR_ENV`] purges after 10 ms (mimalloc 2's
//! default) and commits arena memory on demand (which also leaves the
//! arenas without transparent huge pages), which kept the resident set
//! within about 1.3-1.5x of the live heap. The cost is page faults when freed
//! memory is used again: about 3% on TPC-H SF1 without a limit and up to
//! about 15% under a tight limit, where operators free and reallocate the
//! most.
//!
//! mimalloc reads these settings from the environment when the process
//! starts, before `main`; setting them later through `mi_option_set` does
//! not reach the arena it already reserved (measured: the resident set
//! stayed at 1.9-2.5x). So a binary that uses mimalloc calls
//! [`reexec_with_allocator_env`] first thing in `main`: it starts the same
//! program again, once, with the settings the user has not set themselves.
//! A Python process loads the engine as a module, after its own start: the
//! package sets the same variables before it loads the native module.

/// Environment for the mimalloc global allocator: `(variable, value)`.
/// Values the user set win.
pub const ALLOCATOR_ENV: [(&str, &str); 2] = [
    ("MIMALLOC_PURGE_DELAY", "10"),
    ("MIMALLOC_ARENA_EAGER_COMMIT", "0"),
];

/// Set (to `1`) in the environment of the re-executed program, so it never
/// re-executes again.
pub const REEXEC_GUARD: &str = "KRISHIV_ALLOCATOR_ENV";

/// The [`ALLOCATOR_ENV`] entries `get` (an environment lookup) does not
/// already have, or nothing if `get` has [`REEXEC_GUARD`].
pub fn missing_allocator_env(
    get: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Vec<(&'static str, &'static str)> {
    if get(REEXEC_GUARD).is_some() {
        return Vec::new();
    }
    ALLOCATOR_ENV
        .into_iter()
        .filter(|(k, _)| get(k).is_none())
        .collect()
}

/// Replaces the running program with itself, same arguments and `argv[0]`,
/// with the [`ALLOCATOR_ENV`] entries that are not set added to its
/// environment (and [`REEXEC_GUARD`]). Returns without doing anything when
/// every entry is set or this is the re-executed program; returns the error
/// when the program cannot be executed again, in which case it keeps
/// running with the allocator's defaults. Call it first thing in `main`,
/// before any thread starts. A no-op off Unix.
#[must_use]
pub fn reexec_with_allocator_env() -> Option<std::io::Error> {
    let missing = missing_allocator_env(|k| std::env::var_os(k));
    if missing.is_empty() {
        return None;
    }
    reexec(&missing)
}

#[cfg(unix)]
fn reexec(missing: &[(&str, &str)]) -> Option<std::io::Error> {
    use std::os::unix::process::CommandExt;
    // `/proc/self/exe` is this very program even if its file was replaced;
    // elsewhere fall back to the path the OS reports.
    let program = if std::path::Path::new("/proc/self/exe").exists() {
        std::path::PathBuf::from("/proc/self/exe")
    } else {
        match std::env::current_exe() {
            Ok(p) => p,
            Err(e) => return Some(e),
        }
    };
    let mut args = std::env::args_os();
    let mut cmd = std::process::Command::new(program);
    if let Some(arg0) = args.next() {
        cmd.arg0(arg0);
    }
    cmd.args(args)
        .envs(missing.iter().copied())
        .env(REEXEC_GUARD, "1");
    // Returns only on failure.
    Some(cmd.exec())
}

#[cfg(not(unix))]
fn reexec(_missing: &[(&str, &str)]) -> Option<std::io::Error> {
    None
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ffi::OsString;

    use super::*;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> {
        let m: HashMap<String, OsString> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), OsString::from(v)))
            .collect();
        move |k| m.get(k).cloned()
    }

    #[test]
    fn only_unset_entries_are_added() {
        assert_eq!(missing_allocator_env(env(&[])), ALLOCATOR_ENV.to_vec());
        assert_eq!(
            missing_allocator_env(env(&[("MIMALLOC_PURGE_DELAY", "0")])),
            vec![("MIMALLOC_ARENA_EAGER_COMMIT", "0")]
        );
        let all: Vec<(&str, &str)> = ALLOCATOR_ENV.iter().map(|(k, _)| (*k, "5")).collect();
        assert!(missing_allocator_env(env(&all)).is_empty());
    }

    #[test]
    fn the_reexecuted_program_never_reexecutes() {
        assert!(missing_allocator_env(env(&[(REEXEC_GUARD, "1")])).is_empty());
    }
}
