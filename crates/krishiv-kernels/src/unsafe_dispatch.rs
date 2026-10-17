// SPDX-License-Identifier: Apache-2.0
//! The only `unsafe` in `krishiv-kernels` (form (a) of the kernel unsafe
//! policy: calling a `#[target_feature]` clone after feature detection).
//! `cargo xtask source-guard` rejects `unsafe` in any other file of this
//! crate, and `cargo xtask budgets` counts the blocks.

use crate::dispatch::{KernelCtx, Target};

/// A kernel body: everything [`Kernel::run`] calls must be `#[inline]`d into
/// it so the whole loop is compiled with the clone's target features.
pub(crate) trait Kernel {
    /// The kernel's result.
    type Output;
    /// Runs the kernel. Implementations are `#[inline(always)]`.
    fn run(self) -> Self::Output;
}

/// Runs `kernel` compiled for the context's target.
#[inline]
pub(crate) fn run<K: Kernel>(ctx: &KernelCtx, kernel: K) -> K::Output {
    match ctx.target() {
        Target::Baseline => kernel.run(),
        #[cfg(target_arch = "x86_64")]
        Target::X86_64V3 => {
            debug_assert!(Target::X86_64V3.is_available());
            // SAFETY: a `KernelCtx` only holds a target that `Target::is_available`
            // confirmed on this CPU (`KernelCtx::new`/`with_target` are its only
            // constructors), so every feature `unsafe_run_v3` enables is present.
            unsafe { unsafe_run_v3(kernel) }
        }
        #[cfg(target_arch = "x86_64")]
        Target::X86_64V4 => {
            debug_assert!(Target::X86_64V4.is_available());
            // SAFETY: as above, for the x86-64-v4 feature set.
            unsafe { unsafe_run_v4(kernel) }
        }
        // Non-x86 CPUs never report the x86 targets available, so no context
        // holds them; run the portable code regardless.
        #[cfg(not(target_arch = "x86_64"))]
        Target::X86_64V3 | Target::X86_64V4 => kernel.run(),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,fma,lzcnt,movbe,popcnt")]
fn unsafe_run_v3<K: Kernel>(kernel: K) -> K::Output {
    kernel.run()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(
    enable = "avx2,bmi1,bmi2,fma,lzcnt,movbe,popcnt,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
)]
fn unsafe_run_v4<K: Kernel>(kernel: K) -> K::Output {
    kernel.run()
}
