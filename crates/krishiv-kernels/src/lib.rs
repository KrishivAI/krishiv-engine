// SPDX-License-Identifier: Apache-2.0
//! Vectorised compute kernels with Spark semantics (layer L2).
//!
//! # The specialization framework
//!
//! Every kernel is written once as a generic chunk loop and instantiated
//! along compile-time axes, chosen **once per batch** (never per row):
//!
//! - **operand shape** — `const L_SCALAR` / `const R_SCALAR` parameters: a
//!   scalar operand is a register, not a broadcast array;
//! - **overflow policy** — a zero-sized [`policy::Overflow`] type:
//!   [`policy::Unchecked`], [`policy::JavaWrap`], [`policy::NullOnOverflow`]
//!   or [`policy::AnsiError`].
//!
//! Elementwise kernels compute through NULL slots, so they have **no**
//! has-nulls axis: the loop is identical either way (measured ~1.0x on the
//! same data). Kernels whose work changes with NULLs (division, strings)
//! opt into a `const NULLS: bool` axis with [`specialize`].
//!
//! Loops run over 64-row chunks aligned with validity words, so LLVM can
//! autovectorize them. Values under NULL slots are computed and discarded
//! (compute-through-nulls); overflow flags are masked with the validity word
//! before anything is raised, so **a NULL row never raises**. ANSI kernels
//! check a whole chunk branch-free and only re-scan a chunk whose masked flag
//! is set, to name the exact failing row.
//!
//! Hot loops are also compiled for x86-64-v3 and x86-64-v4 and selected at
//! runtime ([`dispatch`]); [`dispatch::KernelCtx::with_target`] forces any
//! available target (tests and benchmarks).
//!
//! Arithmetic never relies on the build profile: `clippy::arithmetic_side_effects`
//! is denied, so every operation is an explicit `wrapping_*`, `overflowing_*`
//! or `checked_*` call.
//!
//! [`reference`](mod@reference) holds the row-at-a-time oracle every fast path is tested
//! against (property tests, the differential fuzz target, the benchmarks).

#![deny(clippy::arithmetic_side_effects)]

pub mod cast;
mod chunk;
pub mod compare;
pub mod csv;
pub mod datetime;
pub mod datetime_pattern;
pub mod decimal;
pub mod dispatch;
pub mod error;
pub mod float;
pub mod hash;
pub mod integer;
pub mod json;
pub mod list;
pub mod map;
pub mod math;
pub mod number_format;
pub mod ops;
pub mod policy;
pub mod reference;
pub mod specialize;
pub mod string;
pub mod r#struct;
pub mod text;
mod unsafe_dispatch;
pub mod version;

pub use dispatch::{Baseline, Dispatch, KernelCtx, Target};
pub use error::KernelError;
pub use ops::{Add, ArithOp, Mul, Operand, Sub};
pub use policy::{AnsiError, JavaWrap, NullOnOverflow, Overflow, OverflowMode, Unchecked};
pub use version::SparkVersion;
