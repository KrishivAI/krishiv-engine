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
//! they did not prevent the OOM kills, measured). So they apply only when
//! the limit is known at process start: on Linux with glibc an ELF
//! `.init_array` constructor sets them when the environment has
//! `KRISHIV_MEMORY_LIMIT` (the documented, reliable way) or the arguments
//! configure a limit ([`limit_in_args`]: `--memory-limit`,
//! `krishiv.memory.limit=`), and the user has not set the `MIMALLOC_*`
//! variable. Without a limit nothing changes (the options cost about 7-9%
//! on TPC-H SF1). A limit from a configuration file or a runtime `SET` is
//! still enforced by the memory pools and uses [`release`], but without
//! these options. On other targets (musl, macOS) there is no constructor:
//! set the `MIMALLOC_*` variables of [`LIMITED_OPTIONS`] in the
//! environment instead. [`release`] is the pressure release hook
//! (`mi_collect(true)`).

use std::ffi::{CStr, c_int, c_long};

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

/// The environment variable that configures a memory limit for the
/// allocator settings (module docs).
pub const LIMIT_VAR: &CStr = c"KRISHIV_MEMORY_LIMIT";

#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod init {
    use std::ffi::{CStr, c_char, c_int};

    use super::{LIMIT_VAR, LIMITED_OPTIONS, limit_in_args};

    unsafe extern "C" {
        fn getenv(name: *const c_char) -> *const c_char;
    }

    /// Whether `var` is set (to anything non-empty).
    fn is_set(var: &CStr) -> bool {
        // SAFETY: `var` is NUL-terminated, and a non-null `getenv` result is
        // a NUL-terminated string; constructors run before any other thread,
        // so nothing modifies the environment concurrently.
        unsafe {
            let value = getenv(var.as_ptr());
            !value.is_null() && !CStr::from_ptr(value).is_empty()
        }
    }

    /// The constructor; allocates nothing (the options must be set before
    /// the first allocation).
    extern "C" fn init(argc: c_int, argv: *const *const c_char, _envp: *const *const c_char) {
        let count = if argv.is_null() {
            0
        } else {
            usize::try_from(argc).unwrap_or(0)
        };
        // SAFETY: glibc (and only glibc, hence the `cfg`) calls
        // `.init_array` functions with the program's `argc`, `argv` and
        // `envp`: `argv` holds `argc` pointers to NUL-terminated strings
        // that live for the whole process.
        let args = (0..count).filter_map(|i| unsafe {
            let p = *argv.add(i);
            (!p.is_null()).then(|| CStr::from_ptr(p).to_bytes())
        });
        if !is_set(LIMIT_VAR) && !limit_in_args(args) {
            return;
        }
        for &(var, option, value) in &LIMITED_OPTIONS {
            if !is_set(var) {
                // SAFETY: only stores the value in mimalloc's option table;
                // no other thread exists yet.
                unsafe { libmimalloc_sys::mi_option_set(option, value) };
            }
        }
    }

    /// Runs [`init`] before the Rust runtime's first allocation (a reference
    /// from [`super::install`] keeps it linked).
    #[used]
    #[unsafe(link_section = ".init_array")]
    pub(super) static INIT: extern "C" fn(c_int, *const *const c_char, *const *const c_char) = init;

    #[cfg(test)]
    mod tests {
        /// Without arguments (a null `argv`) and without the variable, the
        /// constructor sets nothing and does not read `argv`.
        #[test]
        fn a_null_argv_sets_nothing() {
            // (The test process has no `KRISHIV_MEMORY_LIMIT`, as the other
            // tests that read mimalloc's defaults assume.)
            super::init(0, std::ptr::null(), std::ptr::null());
            // SAFETY: plain getters.
            let got = unsafe {
                (
                    libmimalloc_sys::mi_option_get(15),
                    libmimalloc_sys::mi_option_get(43),
                    libmimalloc_sys::mi_option_get(44),
                )
            };
            assert_eq!(got, (1000, 1, 0));
        }
    }
}

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
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    std::hint::black_box(&init::INIT);
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

    /// The option numbers belong to the linked mimalloc (version 3); this
    /// test process has no limit, so mimalloc's defaults are in force.
    #[test]
    fn the_option_numbers_are_the_linked_mimalloc_s() {
        assert_eq!(read(), (1000, 1, 0));
        // SAFETY: plain getter.
        let version = unsafe { libmimalloc_sys::mi_version() };
        // `major * 10000 + minor * 100 + patch` (3.3.2 is 30302).
        assert!((30_000..40_000).contains(&version), "mimalloc {version}");
        assert_eq!(install() as usize, release as fn() as usize);
        release();
    }

    /// `(purge_delay, allow_thp, minimal_purge_size)` in force.
    fn read() -> (c_long, c_long, c_long) {
        // SAFETY: plain getters.
        unsafe {
            (
                libmimalloc_sys::mi_option_get(15),
                libmimalloc_sys::mi_option_get(43),
                libmimalloc_sys::mi_option_get(44),
            )
        }
    }

    /// In a child of this test binary: the options the constructor set.
    #[test]
    fn child_reports_options() {
        if std::env::var_os("KRISHIV_ALLOC_CHILD").is_some() {
            // Whether huge pages are off for the process: mimalloc calls
            // `PR_SET_THP_DISABLE` at its own initialisation, so this shows
            // the constructor ran before it (not only that the option is set).
            println!("OPTIONS {:?} THP {}", read(), thp_enabled());
        }
    }

    /// `THP_enabled` of this process (`?` if the kernel does not report it).
    fn thp_enabled() -> String {
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        status
            .lines()
            .find_map(|l| l.strip_prefix("THP_enabled:"))
            .map_or("?", str::trim)
            .to_owned()
    }

    /// Runs `child_reports_options` in a child process with `arg` and `env`.
    fn child(arg: Option<&str>, env: &[(&str, &str)]) -> String {
        let mut c = std::process::Command::new(std::env::current_exe().unwrap());
        c.args(["tests::child_reports_options", "--exact", "--nocapture"])
            .env("KRISHIV_ALLOC_CHILD", "1")
            .env_remove("KRISHIV_MEMORY_LIMIT");
        for (k, ..) in LIMITED_OPTIONS {
            c.env_remove(k.to_str().unwrap());
        }
        // A second filter: libtest runs tests matching either; the
        // constructor sees it in `argv`.
        c.args(arg);
        c.envs(env.iter().copied());
        let out = c.output().unwrap();
        assert!(out.status.success(), "{out:?}");
        let text = String::from_utf8(out.stdout).unwrap();
        text.lines()
            .find_map(|l| l.strip_prefix("OPTIONS "))
            .unwrap()
            .to_owned()
    }

    /// The constructor applies the options exactly when the process starts
    /// with a limit (argument or environment), and the user's variables win.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[test]
    fn a_limit_at_start_applies_the_options() {
        // Without a limit the child has this (unlimited) process's huge-page
        // state: 1, or 0 on a kernel without transparent huge pages. With a
        // limit it is 0 either way.
        let unlimited = format!("(1000, 1, 0) THP {}", thp_enabled());
        assert_eq!(child(None, &[]), unlimited);
        assert_eq!(
            child(Some("krishiv.memory.limit=1g"), &[]),
            "(10, 0, 64) THP 0"
        );
        assert_eq!(
            child(None, &[("KRISHIV_MEMORY_LIMIT", "1g")]),
            "(10, 0, 64) THP 0"
        );
        assert_eq!(
            child(
                None,
                &[
                    ("KRISHIV_MEMORY_LIMIT", "1g"),
                    ("MIMALLOC_PURGE_DELAY", "50")
                ]
            ),
            "(50, 0, 64) THP 0"
        );
        assert_eq!(child(None, &[("KRISHIV_MEMORY_LIMIT", "")]), unlimited);
    }
}
