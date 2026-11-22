// SPDX-License-Identifier: Apache-2.0
//! The AWS Glue Data Catalog read-only client (feature `glue`; PR-17 leftover, design P3-lakehouse §2.3): Krishiv's own
//! small client for the Glue JSON API (`GetDatabases`, `GetTables`, `GetTable`) over `reqwest`, `serde_json` and a
//! Signature Version 4 signer on `sha2`, with no AWS SDK in the tree (the SDK is large and would take a credential
//! into code Krishiv does not control; a non-default feature at most).
//!
//! [`config`] the declaration (region or endpoint, static keys, catalog id, limits), [`client`] the three read calls,
//! [`error`] the failures and their Krishiv errors, [`catalog`] `GlueCatalog`, a read-only `Catalog` (no write half:
//! writes and DDL are refused by the capability check with `UNSUPPORTED_FEATURE.CATALOG_OPERATION`). `sigv4` and
//! `transport` are internal and share no code with the other clients, so each feature builds without the others.
//! The Iceberg table of a Glue catalog is exposed by its metadata location; opening it is the Iceberg connector's job.

pub mod catalog;
pub mod client;
pub mod config;
pub mod error;
mod sigv4;
#[cfg(test)]
mod tests;
mod transport;

pub use catalog::{GlueCatalog, GlueFormat, GlueTable};
pub use client::GlueClient;
pub use config::{GlueConfig, GlueLimits};
pub use error::{GlueError, GlueLimit};
