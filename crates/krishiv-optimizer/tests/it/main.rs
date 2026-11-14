// SPDX-License-Identifier: Apache-2.0
//! Optimizer tests over resolved plans: TPC-H and per-rule snapshots,
//! results against unoptimised plans, and properties of random queries.

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test helpers sit outside #[test] fns"
)]

mod cse;
mod group_code;
mod join_cost;
mod join_enumerator;
mod join_order;
mod oracle;
mod plan_stats;
mod props;
mod python_relations;
mod qerror;
mod rules;
mod share_subtree;
mod sum_rewrite;
mod support;
mod tpcds;
mod tpch;
mod udf;
mod win_magic;
