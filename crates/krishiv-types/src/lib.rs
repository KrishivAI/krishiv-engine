// SPDX-License-Identifier: Apache-2.0
//! Internal type representations (layer L1, ADR 0009).
//!
//! [`DataType`] models Spark SQL data types with Spark's renderings
//! (`simpleString`, `catalogString`, `sql`, `typeName`), maps them to and from
//! Arrow ([`arrow`]), and parses Spark DDL type strings ([`parse_data_type`]).

#![forbid(unsafe_code)]

pub mod arrow;
mod data_type;
mod ddl;
pub mod error;

pub use data_type::{DataType, DecimalStorage, DecimalType};
pub use ddl::parse_data_type;

#[cfg(test)]
mod tests;
