// SPDX-License-Identifier: Apache-2.0
//! External catalogs behind `krishiv_plan`'s `Catalog` trait (layer L5; design P3-lakehouse §2.3, §3, §5.6): the
//! Hive metastore, Iceberg REST, Unity Catalog and AWS Glue (read-only). A catalog is input, not a trusted party: every location it
//! returns goes through [`location`] before anything reads it.

#![forbid(unsafe_code)]

#[cfg(feature = "session")]
pub mod declared;
#[cfg(feature = "glue")]
pub mod glue;
#[cfg(feature = "hive")]
pub mod hive;
#[cfg(test)]
#[cfg(any(feature = "iceberg-rest", feature = "unity", feature = "glue"))]
mod http_test_server;
#[cfg(feature = "iceberg-rest")]
pub mod iceberg_rest;
pub mod location;
#[cfg(any(test, feature = "testing"))]
#[cfg(feature = "session")]
#[expect(
    clippy::unwrap_used,
    reason = "a test server: a poisoned lock or a failed bind ends the test"
)]
mod s3_test_server;

/// Test helpers for other crates' tests (feature `testing`; never in a product build).
#[cfg(feature = "testing")]
pub mod testing {
    pub use crate::s3_test_server::{S3Seen, S3Server, serve_s3, write_s3_table};
}
#[cfg(feature = "sqlite")]
pub mod sqlite;
#[cfg(feature = "unity")]
pub mod unity;
