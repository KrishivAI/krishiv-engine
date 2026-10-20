// SPDX-License-Identifier: Apache-2.0
//! mimalloc settings for binaries that use mimalloc as their global
//! allocator (feature `mimalloc`; Linux). Libraries never set the global
//! allocator, so this is opt-in for the program that owns it.
//!
//! **Why.** mimalloc 3 keeps freed memory committed for `purge_delay`
//! (1,000 ms by default), and with transparent huge pages enabled it purges
//! no range under 2 MiB (`minimal_purge_size`), so the many smaller ranges a
//! query engine frees stay resident: the resident set ran 2.5-3.3x above the
//! live heap (TPC-H SF1 at a 200 MiB limit). With `purge_delay` 10 ms
//! (mimalloc 2's default) and a 64 KiB minimal purge, plus the pressure
//! release hook, it stays within about 1.3-1.5x; arenas keep their huge
//! pages (committing them on demand instead cost about 9% without a limit).
//!
//! **When.** mimalloc reserves its first arena at the process's first
//! allocation, before `main` (the Rust runtime allocates), with the options
//! in force then; setting them in `main` did not bring the resident set
//! down (measured), setting them before the first allocation did. [`CONFIGURE`] therefore runs from the
//! ELF `.init_array`, before the runtime's first allocation, and sets each
//! option only when its `MIMALLOC_*` environment variable is unset, so the
//! user's environment still wins. [`install`] keeps it linked and installs
//! [`release`] as the pressure release hook ([`crate::set_release_hook`]).

use std::ffi::{CStr, c_char, c_int, c_long};

/// `(environment variable, mimalloc option number, value)`. The numbers are
/// mimalloc 3's `mi_option_e` positions (`mi_option_purge_delay` = 15,
/// `mi_option_minimal_purge_size` = 44, in KiB); a test pins them to the
/// linked version.
pub const OPTIONS: [(&CStr, c_int, c_long); 2] = [
    (c"MIMALLOC_PURGE_DELAY", 15, 10),
    (c"MIMALLOC_MINIMAL_PURGE_SIZE", 44, 64),
];

unsafe extern "C" {
    fn getenv(name: *const c_char) -> *const c_char;
}

/// The options of [`OPTIONS`] whose variable `is_set` does not report set.
pub fn to_set(is_set: impl Fn(&CStr) -> bool) -> Vec<(c_int, c_long)> {
    OPTIONS
        .iter()
        .filter(|(var, ..)| !is_set(var))
        .map(|&(_, option, value)| (option, value))
        .collect()
}

extern "C" fn configure() {
    // No allocation before the options are set: `getenv` reads the
    // environment in place, and the loop below allocates nothing.
    for &(var, option, value) in &OPTIONS {
        // SAFETY: `var` is a NUL-terminated constant; `getenv` only reads
        // it. Constructors run single-threaded, so nothing modifies the
        // environment concurrently.
        let set = unsafe { !getenv(var.as_ptr()).is_null() };
        if !set {
            // SAFETY: setting an option has no preconditions; it runs
            // before any other thread exists.
            unsafe { libmimalloc_sys::mi_option_set(option, value) };
        }
    }
}

/// Runs the configuration (each unset option of [`OPTIONS`]) before the
/// Rust runtime's first allocation.
#[used]
#[unsafe(link_section = ".init_array.00099")]
pub static CONFIGURE: extern "C" fn() = configure;

/// Returns the calling thread's freed memory to the OS: the release hook
/// (`mi_collect(true)`: frees the thread's empty pages and purges what the
/// arenas scheduled, now).
pub fn release() {
    // SAFETY: `mi_collect` has no preconditions and only touches mimalloc's
    // own state for the calling thread and the arenas.
    unsafe { libmimalloc_sys::mi_collect(true) };
}

/// Called first thing in `main` by a binary whose global allocator is
/// mimalloc: keeps [`CONFIGURE`] linked and installs [`release`] as the
/// pressure release hook. Returns `false` if a hook was installed already.
pub fn install() -> bool {
    // A use of the constructor, so the linker keeps its section.
    std::hint::black_box(&CONFIGURE);
    crate::set_release_hook(release)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_variables_get_their_values_set_ones_are_left() {
        assert_eq!(to_set(|_| false), vec![(15, 10), (44, 64)]);
        assert_eq!(to_set(|v| v == c"MIMALLOC_PURGE_DELAY"), vec![(44, 64)]);
        assert!(to_set(|_| true).is_empty());
    }

    /// The option numbers belong to the linked mimalloc: it is version 3,
    /// and the constructor's values are in force unless the environment
    /// set them.
    #[test]
    fn the_constructor_set_the_linked_mimalloc_options() {
        // SAFETY: plain getters.
        let version = unsafe { libmimalloc_sys::mi_version() };
        // `major * 10000 + minor * 100 + patch` (3.3.2 is 30302).
        assert!((30_000..40_000).contains(&version), "mimalloc {version}");
        for &(var, option, value) in &OPTIONS {
            // SAFETY: plain getter; `var` is NUL-terminated.
            let (got, env) = unsafe {
                (
                    libmimalloc_sys::mi_option_get(option),
                    !getenv(var.as_ptr()).is_null(),
                )
            };
            if !env {
                assert_eq!(got, value, "{var:?}");
            }
        }
    }
}
