// SPDX-License-Identifier: Apache-2.0
//! Workspace automation logic, split out of `main` so it can be tested.

#![forbid(unsafe_code)]

pub mod budgets;
pub mod ci;
pub mod codeql_proxy;
pub mod compat;
pub mod connect_coverage;
pub mod console;
pub mod coverage;
pub mod dev;
pub mod docs;
pub mod docs_examples;
pub mod docs_ext;
pub mod docs_sql;
pub mod file_size;
pub mod format_neutral;
pub mod layering;
pub mod lint;
pub mod map;
pub mod metrics_gate;
pub mod policy;
pub mod precheck;
pub mod profile;
pub mod qerror;
pub mod quarantine;
pub mod registry_coverage;
pub mod wheels;
