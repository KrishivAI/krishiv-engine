// SPDX-License-Identifier: Apache-2.0
//! Internal type representations (layer L1, ADR 0009).
//!
//! [`DataType`] models Spark SQL data types with Spark's renderings
//! (`simpleString`, `catalogString`, `sql`, `typeName`, `json`), maps them to and from
//! Arrow ([`arrow`]), and parses Spark DDL type strings ([`parse_data_type`]).
//! [`batch::Batch`] is the engine's unit of data: Arrow columns plus row weights.

#![forbid(unsafe_code)]

pub mod arrow;
pub mod batch;
mod collation;
mod data_type;
mod ddl;
pub mod error;
pub mod geo;
pub mod interval;
mod json;
mod nested;
pub mod variant;

pub use collation::{CatalogEntry, Collation, CollationFamily};
pub use data_type::{DataType, DayTimeField, DecimalStorage, DecimalType, Srid, YearMonthField};
pub use ddl::parse_data_type;
pub use json::parse_metadata;
pub use nested::{ArrayType, MapType, Metadata, StructField, StructType, equals_ignore_case};

#[cfg(test)]
mod tests;
