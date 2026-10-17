// SPDX-License-Identifier: Apache-2.0
//! In-crate tests: they match the `#[non_exhaustive]` node enums without
//! wildcard arms, so a new node fails to compile here until it is covered
//! by the inventory, the generator and the mutation sites.

mod dag;
mod depth;
mod display;
mod display_more;
mod dsl;
mod encode;
mod inventory;
mod nodes;
mod props;
mod rewrite;
mod secrets;
mod tpch;
