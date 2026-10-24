// SPDX-License-Identifier: Apache-2.0
//! The only `unsafe` in `krishiv-kernels` (form (a) of the kernel unsafe
//! policy: calling a `#[target_feature]` clone after feature detection).
//! `cargo xtask source-guard` rejects `unsafe` in any other file of this
//! crate, and `cargo xtask budgets` counts the blocks.

use crate::dispatch::{Baseline, KernelCtx, Target};

/// A kernel body: everything [`Kernel::run`] calls must be `#[inline]`d into
/// it so the whole loop is compiled with the clone's target features.
/// (Nominally `pub` inside this private module: unnameable outside the
/// crate, as [`Run`] needs.)
pub trait Kernel {
    /// The kernel's result.
    type Output;
    /// Runs the kernel. Implementations are `#[inline(always)]`.
    fn run(self) -> Self::Output;
}

/// How a [`crate::Dispatch`] runs a kernel body. Sealed: this module is
/// private, so no type outside the crate can implement it.
pub trait Run {
    /// Whether a scalar operand gets its own loop (`<L_SCALAR, R_SCALAR>`
    /// monomorphs). Without them a kernel compiles one loop and a scalar is
    /// repeated into an array first: a copy per call, nothing for one row.
    const SCALAR_LOOPS: bool;
    /// Runs `kernel` compiled for the target this dispatch chooses.
    fn run_kernel<K: Kernel>(&self, kernel: K) -> K::Output;
}

/// [`Run::SCALAR_LOOPS`] of `dispatch`'s type.
#[inline(always)]
pub(crate) fn scalar_loops<D: Run + ?Sized>(_dispatch: &D) -> bool {
    D::SCALAR_LOOPS
}

/// Runs `kernel` as `dispatch` chooses.
#[inline]
pub(crate) fn run<D: Run + ?Sized, K: Kernel>(dispatch: &D, kernel: K) -> K::Output {
    dispatch.run_kernel(kernel)
}

/// The build baseline only: no `#[target_feature]` clone is instantiated, and
/// no scalar-operand loop, so a kernel reached only through [`Baseline`]
/// compiles one loop instead of nine.
impl Run for Baseline {
    const SCALAR_LOOPS: bool = false;

    #[inline]
    fn run_kernel<K: Kernel>(&self, kernel: K) -> K::Output {
        kernel.run()
    }
}

/// The context's target, detected at run time.
impl Run for KernelCtx {
    const SCALAR_LOOPS: bool = true;

    #[inline]
    fn run_kernel<K: Kernel>(&self, kernel: K) -> K::Output {
        match self.target() {
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
