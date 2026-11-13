// SPDX-License-Identifier: Apache-2.0
//! Iceberg connector (layer L5; P3-T022a so far: iceberg-rust's file access over
//! Krishiv's I/O).
//!
//! iceberg-rust 0.11 (on Arrow 58) is confined to the private module `iceberg_v0`:
//! no iceberg or Arrow 58 type appears in this crate's public items
//! (`design/P3-lakehouse.md` §5.2). Values iceberg-rust returns as Arrow 58 reach
//! Arrow 60 through `krishiv-arrow-bridge`; data batches never cross.

#![forbid(unsafe_code)]

mod iceberg_v0;

pub use iceberg_v0::storage::StorageRegistration;
pub use iceberg_v0::table::IcebergTable;
