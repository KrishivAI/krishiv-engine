// SPDX-License-Identifier: Apache-2.0
//! Spark-compatible built-in functions (layer L2).
//!
//! - [`scalar`]: the execution binding. [`scalar::bind_scalar`] turns a
//!   scalar function's name and argument types into a
//!   [`scalar::ScalarFunction`] over the `krishiv-kernels` kernels, which
//!   the physical planner calls once per batch.
//!
//! Aggregates are executed by `krishiv-exec`; the canonical operators and
//! `CAST` by the expression evaluator.

#![forbid(unsafe_code)]

pub mod scalar;
