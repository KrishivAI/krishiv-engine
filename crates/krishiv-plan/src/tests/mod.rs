// SPDX-License-Identifier: Apache-2.0
//! In-crate tests. The compile-time guard for new node variants is the
//! exhaustive matches in the child walkers (`LogicalPlan::for_each_child`,
//! `LogicalPlan::for_each_expr`, `Expr::for_each_child`): `#[non_exhaustive]`
//! does not apply inside this crate, so a new variant fails to compile there
//! until it is handled.

mod nodes;
