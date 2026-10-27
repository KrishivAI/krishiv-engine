// SPDX-License-Identifier: Apache-2.0
//! Workspace automation logic, split out of `main` so it can be tested.

#![forbid(unsafe_code)]

pub mod budgets;
pub mod ci;
pub mod codeql_proxy;
pub mod coverage;
pub mod dev;
pub mod docs;
pub mod layering;
pub mod lint;
pub mod metrics_gate;
pub mod policy;
pub mod precheck;
pub mod profile;
pub mod registry_coverage;
pub mod wheels;
