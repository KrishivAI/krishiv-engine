// SPDX-License-Identifier: Apache-2.0
//! Runtime CPU-feature dispatch.
//!
//! Hot loops are written once and compiled three times on x86-64: for the
//! build baseline, for x86-64-v3 (AVX2, BMI1/2, FMA, LZCNT, MOVBE) and for
//! x86-64-v4 (v3 plus AVX-512 F/BW/CD/DQ/VL). The best target is detected
//! once per process; a [`KernelCtx`] carries the chosen target, and
//! [`KernelCtx::with_target`] forces any target the CPU supports (tests run
//! every available target, benchmarks compare them). Other architectures
//! run the baseline only.

use std::sync::OnceLock;

use crate::error::KernelError;

/// A code-generation target for kernel loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Target {
    /// The build's own target features (always available).
    Baseline,
    /// x86-64-v3: AVX2, BMI1, BMI2, FMA, LZCNT, MOVBE, POPCNT.
    X86_64V3,
    /// x86-64-v4: x86-64-v3 plus AVX-512 F, BW, CD, DQ and VL.
    X86_64V4,
}

impl Target {
    /// Every target, from the most portable to the most specific.
    pub const ALL: [Self; 3] = [Self::Baseline, Self::X86_64V3, Self::X86_64V4];

    /// Whether this CPU can run code compiled for the target.
    #[must_use]
    pub fn is_available(self) -> bool {
        match self {
            Self::Baseline => true,
            Self::X86_64V3 => has_v3(),
            Self::X86_64V4 => has_v3() && has_v4_extra(),
        }
    }

    /// The most specific available target (detected once, then cached).
    #[must_use]
    pub fn best() -> Self {
        static BEST: OnceLock<Target> = OnceLock::new();
        *BEST.get_or_init(|| {
            Self::ALL
                .into_iter()
                .rev()
                .find(|t| t.is_available())
                .unwrap_or(Self::Baseline)
        })
    }

    /// The targets this CPU supports, baseline first.
    pub fn available() -> impl Iterator<Item = Self> {
        Self::ALL.into_iter().filter(|t| t.is_available())
    }
}

#[cfg(target_arch = "x86_64")]
fn has_v3() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("bmi1")
        && std::arch::is_x86_feature_detected!("bmi2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("lzcnt")
        && std::arch::is_x86_feature_detected!("movbe")
        && std::arch::is_x86_feature_detected!("popcnt")
}

#[cfg(target_arch = "x86_64")]
fn has_v4_extra() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512cd")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512vl")
}

#[cfg(not(target_arch = "x86_64"))]
fn has_v3() -> bool {
    false
}

#[cfg(not(target_arch = "x86_64"))]
fn has_v4_extra() -> bool {
    false
}

/// Per-call kernel context. Kernels never read configuration; the context
/// only carries the code-generation target, which is always one the CPU
/// supports (the invariant that makes dispatch sound).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelCtx {
    target: Target,
}

impl KernelCtx {
    /// A context using the best available target.
    #[must_use]
    pub fn new() -> Self {
        Self {
            target: Target::best(),
        }
    }

    /// A context forced to `target` (tests and benchmarks).
    ///
    /// # Errors
    /// `INTERNAL_ERROR` if this CPU cannot run `target`.
    pub fn with_target(target: Target) -> Result<Self, KernelError> {
        if target.is_available() {
            Ok(Self { target })
        } else {
            Err(KernelError::internal(format!(
                "kernel target {target:?} is not supported by this CPU"
            )))
        }
    }

    /// The target kernels run on.
    #[must_use]
    pub fn target(&self) -> Target {
        self.target
    }
}

impl Default for KernelCtx {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_always_available_and_best_is_available() {
        assert!(Target::Baseline.is_available());
        assert!(Target::best().is_available());
        assert_eq!(Target::available().next(), Some(Target::Baseline));
        assert_eq!(KernelCtx::default().target(), Target::best());
        for t in Target::ALL {
            match KernelCtx::with_target(t) {
                Ok(ctx) => assert_eq!(ctx.target(), t),
                Err(e) => {
                    assert!(!t.is_available());
                    assert_eq!(
                        e.error_class(),
                        krishiv_common::error::ErrorClass::InternalError
                    );
                }
            }
        }
        // v4 implies v3.
        assert!(!Target::X86_64V4.is_available() || Target::X86_64V3.is_available());
    }
}
