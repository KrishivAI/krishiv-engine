// SPDX-License-Identifier: Apache-2.0
//! The Iceberg REST catalog client (feature `iceberg-rest`; PR-17, T022b; decision 8 and the owner's decision of
//! 2026-11-15): Krishiv's own small read client, not `iceberg-catalog-rest` 0.11, which applies a server's `uri`
//! override and then sends the caller's token there. Wire forms follow the spec and the Java client at
//! `apache-iceberg-1.12.0`.
//!
//! [`config`] the declaration (URI, warehouse, auth, limits, the OneLake form, the S3 Tables refusal), [`client`]
//! the calls, [`types`] the response bodies, [`error`] the failures and their Krishiv errors; `auth`, `transport`
//! and `encode` are internal. The `Catalog` over it (metadata mapped through the Iceberg connector) is a later step.

mod auth;
pub mod client;
pub mod config;
mod encode;
pub mod error;
#[cfg(test)]
mod fixture_tests;
#[cfg(test)]
mod test_server;
#[cfg(test)]
mod tests;
mod transport;
pub mod types;

pub use client::RestClient;
pub use config::{ONELAKE_URI, RestAuth, RestConfig, RestLimits};
pub use error::{RestError, RestLimit};
pub use types::{CatalogConfig, LoadedTable, StorageCredential, TableIdent};
