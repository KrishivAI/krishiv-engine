// SPDX-License-Identifier: Apache-2.0
//! mimalloc policy for the binaries whose global allocator is mimalloc
//! (`krishiv` CLI, `krishiv-bench-runner`). Libraries, the Python module and
//! embedding hosts never depend on this crate: the host owns its allocator
//! and its policy (P2-T012), and can install its own release hook with
//! `krishiv_memory::set_release_hook`.
//!
//! **Why.** mimalloc 3 keeps freed memory committed for `purge_delay`
//! (1,000 ms by default) and backs its arenas with transparent huge pages,
//! of which it purges no range under 2 MiB, so the many smaller ranges a
//! query engine frees stay resident: under a 200 MiB limit the resident set
//! ran 2.5-3.3x above the live heap (TPC-H SF1) and processes were
//! OOM-killed under a cgroup cap at the memory bound.
//!
//! **What, only under a configured memory limit.** [`LIMITED_OPTIONS`]:
//! purge after 10 ms (mimalloc 2's default), no huge pages, purge ranges
//! from 64 KiB. They must be in force before mimalloc reserves its first
//! arena, at the process's first allocation, before `main` (set from `main`
//! they did not prevent the OOM kills, measured): an ELF `.init_array`
//! constructor reads the program's arguments (glibc passes them to
//! constructors) and sets them only when the arguments configure a memory
//! limit ([`limit_in_args`]) and the user has not set the variable. Without
//! a limit nothing changes (the options cost about 7-9% on TPC-H SF1).
//! [`release`] is the pressure release hook (`mi_collect(true)`).

use std::ffi::{CStr, c_char, c_int, c_long};

/// `(environment variable, mimalloc option number, value)` set under a
/// memory limit. The numbers are mimalloc 3's `mi_option_e` positions
/// (`mi_option_purge_delay` = 15, `mi_option_allow_thp` = 43,
/// `mi_option_minimal_purge_size` = 44, in KiB); a test pins them to the
/// linked mimalloc.
pub const LIMITED_OPTIONS: [(&CStr, c_int, c_long); 3] = [
    (c"MIMALLOC_PURGE_DELAY", 15, 10),
    (c"MIMALLOC_ALLOW_THP", 43, 0),
    (c"MIMALLOC_MINIMAL_PURGE_SIZE", 44, 64),
];

/// Whether program arguments configure a memory limit: a `--memory-limit`
/// flag (the bench runner) or a `krishiv.memory.limit=` setting (`--conf`).
pub fn limit_in_args<'a>(args: impl IntoIterator<Item = &'a [u8]>) -> bool {
    const KEY: &[u8] = b"krishiv.memory.limit=";
    args.into_iter().any(|a| {
        a == b"--memory-limit"
            || a.starts_with(b"--memory-limit=")
            || a.windows(KEY.len()).any(|w| w == KEY)
    })
}

unsafe extern "C" {
    fn getenv(name: *const c_char) -> *const c_char;
}

/// The constructor: allocates nothing (the options must be set before the
/// first allocation).
extern "C" fn init(argc: c_int, argv: *const *const c_char, _envp: *const *const c_char) {
    if argv.is_null() {
        return;
    }
    let count = usize::try_from(argc).unwrap_or(0);
    // SAFETY: glibc calls `.init_array` functions with the program's `argc`
    // and `argv`: `argv` holds `argc` pointers to NUL-terminated strings,
    // alive for the whole process.
    let args = (0..count).filter_map(|i| unsafe {
        let p = *argv.add(i);
        (!p.is_null()).then(|| CStr::from_ptr(p).to_bytes())
    });
    if !limit_in_args(args) {
        return;
    }
    for &(var, option, value) in &LIMITED_OPTIONS {
        // SAFETY: `var` is NUL-terminated; constructors run before any
        // other thread, so the environment and mimalloc's option table are
        // not accessed concurrently.
        unsafe {
            if getenv(var.as_ptr()).is_null() {
                libmimalloc_sys::mi_option_set(option, value);
            }
        }
    }
}

/// Runs [`LIMITED_OPTIONS`]' configuration before the Rust runtime's first
/// allocation (a reference from [`install`] keeps it linked).
#[used]
#[unsafe(link_section = ".init_array")]
static INIT: extern "C" fn(c_int, *const *const c_char, *const *const c_char) = init;

/// Returns the calling thread's freed memory to the OS: the pressure
/// release hook.
pub fn release() {
    // SAFETY: `mi_collect` has no preconditions; it only touches mimalloc's
    // state for the calling thread and the arenas.
    unsafe { libmimalloc_sys::mi_collect(true) };
}

/// The binary's half of the policy, called first thing in `main`: keeps
/// the constructor linked and returns [`release`], the hook to install with
/// `krishiv_memory::set_release_hook`.
pub fn install() -> fn() {
    std::hint::black_box(&INIT);
    release
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_limit_comes_from_the_flag_or_the_conf_key() {
        let has = |a: &[&str]| limit_in_args(a.iter().map(|s| s.as_bytes()));
        assert!(has(&["runner", "--data", "d", "--memory-limit", "2g"]));
        assert!(has(&["runner", "--memory-limit=2g"]));
        assert!(has(&[
            "krishiv",
            "sql",
            "--conf",
            "krishiv.memory.limit=2g"
        ]));
        assert!(has(&["krishiv", "sql", "--conf=krishiv.memory.limit=2g"]));
        assert!(!has(&["runner", "--data", "d", "--threads", "4"]));
        assert!(!has(&[
            "krishiv",
            "sql",
            "--conf",
            "krishiv.io.prefetchBytes=1m"
        ]));
        assert!(!has(&["krishiv", "sql", "--memory-limits"]));
    }

    /// The option numbers belong to the linked mimalloc (version 3) and
    /// name the options this crate means.
    #[test]
    fn the_option_numbers_are_the_linked_mimalloc_s() {
        // SAFETY: plain getters.
        let (version, delay, thp) = unsafe {
            (
                libmimalloc_sys::mi_version(),
                libmimalloc_sys::mi_option_get(15),
                libmimalloc_sys::mi_option_get(43),
            )
        };
        // `major * 10000 + minor * 100 + patch` (3.3.2 is 30302).
        assert!((30_000..40_000).contains(&version), "mimalloc {version}");
        // This test binary has no limit argument: mimalloc 3's defaults.
        assert_eq!((delay, thp), (1000, 1));
        assert_eq!(install() as usize, release as fn() as usize);
        release();
    }
}
