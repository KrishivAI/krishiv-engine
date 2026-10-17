// SPDX-License-Identifier: Apache-2.0
//! Workspace automation logic, split out of `main` so it can be tested.

#![forbid(unsafe_code)]

pub mod budgets;
pub mod ci;
pub mod docs;
pub mod layering;
pub mod lint;
pub mod policy;
