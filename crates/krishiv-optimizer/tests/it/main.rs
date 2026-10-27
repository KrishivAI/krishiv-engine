// SPDX-License-Identifier: Apache-2.0
//! Optimizer tests over resolved plans: TPC-H and per-rule snapshots,
//! results against unoptimised plans, and properties of random queries.

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test helpers sit outside #[test] fns"
)]

mod cse;
mod join_order;
mod plan_stats;
mod props;
mod qerror;
mod rules;
mod sum_rewrite;
mod support;
mod tpcds;
mod tpch;
mod udf;
