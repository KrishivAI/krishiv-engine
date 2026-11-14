// SPDX-License-Identifier: Apache-2.0
//! The Hive metastore catalog (feature `hive`; design P3-lakehouse §2.3): the hand-written structs of
//! `hive_metastore.thrift` ([`types`]) and a blocking client of its read calls ([`client`]), over the plain
//! `thrift` runtime, and [`catalog::HiveCatalog`], the read-only `Catalog` over it.

pub mod catalog;
pub mod client;
pub mod limits;
#[cfg(test)]
mod test_server;
pub mod types;
