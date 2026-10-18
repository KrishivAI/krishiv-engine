// SPDX-License-Identifier: Apache-2.0
//! Optimizer tests over resolved plans: TPC-H and per-rule snapshots.

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test helpers sit outside #[test] fns"
)]

mod rules;
mod support;
mod tpch;
