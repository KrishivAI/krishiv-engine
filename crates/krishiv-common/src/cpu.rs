// SPDX-License-Identifier: Apache-2.0
//! Start-up check of the CPU features this build was compiled for.
//!
//! The shipped builds target a CPU level above the architecture baseline
//! (`x86-64-v3` on x86-64, `neoverse-n1` on aarch64; see `README.md`,
//! "Supported CPUs"). On an older CPU the first instruction of that level
//! faults with SIGILL and gives the user nothing to act on. [`check`] runs
//! once, as early as possible, compares every feature the compiler was allowed
//! to assume with what the CPU reports, and returns an error that names the
//! missing features and the compat build to use instead.
//!
//! What is checked is decided at compile time with `cfg!(target_feature)`, so
//! a baseline (compat) build checks nothing and costs nothing.
//!
//! Limitation: the check cannot run before the program starts. Code that runs
//! first (the Rust runtime start-up, the dynamic loader, and for the Python
//! module the interpreter importing the extension) was not necessarily built
//! for the baseline, and the compiler may use a higher level anywhere in the
//! binary, including in code that runs before `main`. In practice nothing
//! before `main` is hot or vectorised, so the check is reliable on the CPUs it
//! targets, but it is best-effort rather than a guarantee.

use std::fmt;
use std::io::Write;

/// Name of the compat Python distribution (it imports as `krishiv`).
pub const COMPAT_WHEEL: &str = "krishiv-compat";
/// Name of the compat command line binary.
pub const COMPAT_BINARY: &str = "krishiv-compat";

/// Features the compiler was allowed to assume that the CPU does not report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingCpuFeatures {
    level: &'static str,
    missing: Vec<&'static str>,
}

impl MissingCpuFeatures {
    /// The CPU level this build targets, e.g. `x86-64-v3`.
    pub fn level(&self) -> &'static str {
        self.level
    }

    /// The missing feature names, in a fixed order.
    pub fn missing(&self) -> &[&'static str] {
        &self.missing
    }
}

impl fmt::Display for MissingCpuFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "this build of Krishiv requires a {} CPU, but this CPU lacks: {}. \
             Use the compat build instead: `pip install {COMPAT_WHEEL}` for Python, \
             or the `{COMPAT_BINARY}` binary.",
            self.level,
            self.missing.join(", ")
        )
    }
}

impl std::error::Error for MissingCpuFeatures {}

/// The features among `required` whose detection result is `false`.
fn missing_of(required: &[(&'static str, bool)]) -> Vec<&'static str> {
    required
        .iter()
        .filter(|(_, present)| !present)
        .map(|(n, _)| *n)
        .collect()
}

fn verdict(
    level: &'static str,
    required: &[(&'static str, bool)],
) -> Result<(), MissingCpuFeatures> {
    let missing = missing_of(required);
    if missing.is_empty() {
        Ok(())
    } else {
        Err(MissingCpuFeatures { level, missing })
    }
}

/// One `(name, detected)` entry for each listed feature this build was compiled with.
macro_rules! detect {
    ($detect:ident, $($feature:tt),+ $(,)?) => {{
        let mut v: Vec<(&'static str, bool)> = Vec::new();
        $(
            if cfg!(target_feature = $feature) {
                v.push(($feature, std::arch::$detect!($feature)));
            }
        )+
        v
    }};
}

/// Checks the running CPU against the features this build was compiled for.
///
/// # Errors
/// [`MissingCpuFeatures`] when the CPU lacks a feature the build assumes.
#[cfg(target_arch = "x86_64")]
pub fn check() -> Result<(), MissingCpuFeatures> {
    // The x86-64-v3 set (v2 and v3 of the x86-64 psABI levels); features
    // not enabled at compile time are skipped.
    let required = detect!(
        is_x86_feature_detected,
        "sse3",
        "ssse3",
        "sse4.1",
        "sse4.2",
        "popcnt",
        "cmpxchg16b",
        "avx",
        "avx2",
        "bmi1",
        "bmi2",
        "f16c",
        "fma",
        "lzcnt",
        "movbe",
        "xsave",
    );
    verdict("x86-64-v3", &required)
}

/// Checks the running CPU against the features this build was compiled for.
///
/// # Errors
/// [`MissingCpuFeatures`] when the CPU lacks a feature the build assumes.
#[cfg(target_arch = "aarch64")]
pub fn check() -> Result<(), MissingCpuFeatures> {
    // The neoverse-n1 features that are runtime-detectable.
    let required = detect!(
        is_aarch64_feature_detected,
        "aes",
        "crc",
        "dotprod",
        "fp16",
        "lse",
        "rcpc",
        "rdm",
        "sha2",
    );
    verdict("neoverse-n1", &required)
}

/// Checks the running CPU against the features this build was compiled for.
///
/// # Errors
/// Never on this architecture: nothing is checked.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn check() -> Result<(), MissingCpuFeatures> {
    Ok(())
}

/// [`check`] for a binary's `main`: prints the error to stderr and exits with
/// status 1 when the CPU is too old.
pub fn check_or_exit() {
    if let Err(e) = check() {
        // Best effort: a closed stderr leaves nothing to report to.
        let _ = writeln!(std::io::stderr(), "error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_missing_is_ok() {
        assert_eq!(
            verdict("x86-64-v3", &[("avx2", true), ("fma", true)]),
            Ok(())
        );
        assert_eq!(verdict("x86-64-v3", &[]), Ok(()));
    }

    #[test]
    fn missing_features_are_named_in_order() {
        let err = verdict(
            "x86-64-v3",
            &[("avx2", false), ("bmi2", true), ("fma", false)],
        )
        .unwrap_err();
        assert_eq!(err.missing(), ["avx2", "fma"]);
        assert_eq!(err.level(), "x86-64-v3");
    }

    #[test]
    fn message_names_features_and_compat_packages() {
        let err = verdict("x86-64-v3", &[("avx2", false)]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("avx2"), "{msg}");
        assert!(msg.contains("x86-64-v3"), "{msg}");
        assert!(msg.contains(COMPAT_WHEEL), "{msg}");
        assert!(msg.contains(COMPAT_BINARY), "{msg}");
    }

    /// The machine running the tests built them, so it must pass its own check.
    #[test]
    fn running_cpu_satisfies_its_own_build() {
        assert_eq!(check(), Ok(()));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn checked_set_follows_compile_time_features() {
        let compiled = detect!(is_x86_feature_detected, "avx2", "fma");
        assert_eq!(
            compiled.len(),
            usize::from(cfg!(target_feature = "avx2")) + usize::from(cfg!(target_feature = "fma"))
        );
    }
}
