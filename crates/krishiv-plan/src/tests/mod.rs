// SPDX-License-Identifier: Apache-2.0
//! In-crate tests: they match the `#[non_exhaustive]` node enums without
//! wildcard arms, so a new node fails to compile here until it is covered
//! by the inventory, the generator and the mutation sites.

mod analyzer;
mod barrier_widen;
mod column_names;
mod dag;
mod definer_sources;
mod depth;
mod display;
mod display_more;
mod dsl;
mod e2_planning_speed;
mod encode;
mod files;
mod function_call_forms;
mod input_file;
mod inventory;
mod json_tuple;
mod lineage;
mod local;
mod metadata_mode;
mod nested_literals;
mod nodes;
mod observe;
mod pivot;
mod prefetch;
mod prefix_cache;
mod props;
mod python_relation_analysis;
mod python_relations;
mod rewrite;
mod secrets;
mod setops;
mod sqlfn;
mod time_window;
mod tpch;
