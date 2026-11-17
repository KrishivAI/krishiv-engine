// SPDX-License-Identifier: Apache-2.0
//! External catalogs behind `krishiv_plan`'s `Catalog` trait (layer L5; design P3-lakehouse §2.3, §3, §5.6): the
//! Hive metastore, Iceberg REST and Unity Catalog (read-only). A catalog is input, not a trusted party: every location it
//! returns goes through [`location`] before anything reads it.

#![forbid(unsafe_code)]

#[cfg(feature = "session")]
pub mod declared;
#[cfg(feature = "hive")]
pub mod hive;
#[cfg(feature = "iceberg-rest")]
pub mod iceberg_rest;
pub mod location;
#[cfg(feature = "sqlite")]
pub mod sqlite;
