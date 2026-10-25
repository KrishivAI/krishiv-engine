// SPDX-License-Identifier: Apache-2.0
//! Spark-compatible built-in functions (layer L2).
//!
//! - [`registry`]: Spark's built-in names bound to their analysis
//!   signature ([`signature`]: implicit casts, result type, arity),
//!   determinism, per-version availability and execution.
//! - [`scalar`]: the execution binding. [`scalar::bind_scalar`] turns a
//!   scalar function's name and argument types into a
//!   [`scalar::ScalarFunction`] over the `krishiv-kernels` kernels, which
//!   the physical planner calls once per batch.
//!
//! - `uniform`: the `uniform` result-type rule both layers share (not a
//!   layer: the analyzer and the executor map their own types to its kinds).
//!
//! Aggregates are executed by `krishiv-exec`; the canonical operators and
//! `CAST` by the expression evaluator.

#![forbid(unsafe_code)]

pub mod registry;
pub mod scalar;
pub mod signature;
pub(crate) mod uniform;
