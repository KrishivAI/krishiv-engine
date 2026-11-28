// SPDX-License-Identifier: Apache-2.0
//! The Apache DataSketches families behind Spark's `hll_*`, `theta_*`, `kll_*` and `tuple_*` functions
//! (P3-T086, design note section 3.3): argument checks, Spark's error classes, the per-type update
//! encoding and the byte mapping, over the `datasketches` crate. The scalar and aggregate bindings sit on top.

pub mod count_min;
pub mod hll;
pub mod theta;
