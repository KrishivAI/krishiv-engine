// SPDX-License-Identifier: Apache-2.0
//! The Unity Catalog read-only client (feature `unity`; PR-16, design P3-lakehouse §3 and §5.6): Krishiv's own small
//! client for the open-source UC REST read endpoints (v0.7.0), over `reqwest` and `serde_json`, no UC SDK crate.
//!
//! [`config`] the declaration (server URI, one UC catalog, static bearer token, limits), [`client`] the four read calls,
//! [`error`] the failures and their Krishiv errors, [`catalog`] `UnityCatalog`, a read-only `Catalog` (no write half:
//! writes and DDL are refused by the capability check with `UNSUPPORTED_FEATURE.CATALOG_OPERATION`). `transport` is
//! internal and shares no code with the Iceberg REST client, so either feature builds without the other.
//!
//! Iceberg tables of UC are not read here: UC serves them through its own Iceberg REST endpoint
//! (`/api/2.1/unity-catalog/iceberg`), which this client does not build; they are
//! `UNSUPPORTED_FEATURE.TABLE_OPERATION`. Whether the OSS server in the test image serves that endpoint is unverified
//! (the Docker check is deferred).

pub mod catalog;
pub mod client;
pub mod config;
pub mod error;
#[cfg(test)]
mod tests;
mod transport;

pub use catalog::{UnityCatalog, UnityFormat, UnityTable};
pub use client::UnityClient;
pub use config::{UnityConfig, UnityLimits};
pub use error::{UnityError, UnityLimit};
