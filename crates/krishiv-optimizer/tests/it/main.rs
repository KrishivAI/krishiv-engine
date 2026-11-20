// SPDX-License-Identifier: Apache-2.0
//! Optimizer tests over resolved plans: TPC-H and per-rule snapshots,
//! results against unoptimised plans, and properties of random queries.

#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test helpers sit outside #[test] fns"
)]

mod as_of;
mod barrier;
mod cse;
mod derived_keys;
mod eager_null_filter;
mod group_code;
mod join_cost;
mod join_enumerator;
mod join_order;
mod late_payload;
mod one_row_scalar;
mod oracle;
mod payload_above;
mod plan_stats;
mod pre_expand;
mod preserved_side;
mod props;
mod prune_first;
mod python_relations;
mod qerror;
mod rollup_chain;
mod rule_differential;
mod rules;
mod semi_order;
mod share_literal;
mod share_scan;
mod share_subtree;
mod sum_rewrite;
mod support;
mod tpcds;
mod tpch;
mod udf;
mod win_magic;
